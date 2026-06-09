//! Real-time packet-dropout plugin.
//!
//! Reproduces the bursty packet loss of the reference `LossGen` snippet live:
//! the stream is split into fixed-size packets, the `LossGen` model decides
//! keep/drop per packet, and dropped packets are hard-zeroed.

use truce::prelude::*;
use truce_slint::{PluginContext, SlintEditor, SyncFn};

use crate::dsp::PacketLossMasker;

mod dsp;
mod model;

slint::include_modules!();

// --- Parameters ---

use DropoutPluginParamsParamId as P;

#[derive(Params)]
pub struct DropoutPluginParams {
    /// Target loss fraction in [0, 1]; this is exactly the model's `perc`
    /// input. truce's `%` unit renders it as a percentage (0.2 → "20%").
    #[param(name = "Loss", range = "linear(0, 1)", unit = "%", default = 0.2)]
    pub loss: FloatParam,

    /// Packet size in milliseconds — the granularity of each drop decision.
    #[param(name = "Packet", range = "linear(2, 80)", unit = "ms", default = 20)]
    pub packet_ms: FloatParam,

    /// Seed for the deterministic loss pattern. Not shown in the GUI (a
    /// set-and-forget control), but still saved with the project and
    /// automatable from the host's generic parameter view.
    #[param(name = "Seed", range = "discrete(0, 99)", default = 42)]
    pub seed: IntParam,

    /// Clean passthrough when enabled.
    #[param(name = "Bypass", default = 0)]
    pub bypass: BoolParam,

    /// Lights up while a packet is being dropped (audio-thread → UI signal).
    #[meter]
    pub activity: MeterSlot,
}

// --- Plugin ---

pub struct DropoutPlugin {
    params: Arc<DropoutPluginParams>,
    masker: PacketLossMasker,
    sample_rate: f64,
    last_seed: i64,
    last_packet_ms: f32,
}

impl DropoutPlugin {
    pub fn new(params: Arc<DropoutPluginParams>) -> Self {
        let packet_ms = params.packet_ms.read();
        let seed = params.seed.value();
        let masker = PacketLossMasker::new(48_000.0, packet_ms, seed as u64);
        Self {
            params,
            masker,
            sample_rate: 48_000.0,
            last_seed: seed,
            last_packet_ms: packet_ms,
        }
    }
}

impl PluginLogic for DropoutPlugin {
    fn reset(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.params.set_sample_rate(sample_rate);
        self.params.snap_smoothers();
        self.sample_rate = sample_rate;

        let packet_ms = self.params.packet_ms.read();
        let seed = self.params.seed.value();
        self.last_packet_ms = packet_ms;
        self.last_seed = seed;
        self.masker.set_packet_size(sample_rate, packet_ms);
        self.masker.reset(seed as u64);
    }

    fn process(
        &mut self,
        buffer: &mut AudioBuffer,
        _events: &EventList,
        context: &mut ProcessContext,
    ) -> ProcessStatus {
        let bypass = self.params.bypass.value();

        if bypass {
            // Wrapper doesn't copy input → output, so passthrough is explicit.
            for ch in 0..buffer.channels() {
                let (inp, out) = buffer.io(ch);
                out.copy_from_slice(inp);
            }
            context.set_meter(P::Activity, 0.0);
            return ProcessStatus::Normal;
        }

        // Pick up control-rate parameter changes at the block boundary.
        let seed = self.params.seed.value();
        if seed != self.last_seed {
            self.last_seed = seed;
            self.masker.reset(seed as u64);
        }
        let packet_ms = self.params.packet_ms.read();
        if (packet_ms - self.last_packet_ms).abs() > f32::EPSILON {
            self.last_packet_ms = packet_ms;
            self.masker.set_packet_size(self.sample_rate, packet_ms);
        }
        let perc = self.params.loss.read(); // already a fraction in [0, 1]

        for i in 0..buffer.num_samples() {
            // One decision per sample-position, shared across all channels so
            // the whole stream drops together (like real packet loss).
            let gain = self.masker.next_sample_gain(perc);
            for ch in 0..buffer.channels() {
                let (inp, out) = buffer.io(ch);
                out[i] = inp[i] * gain;
            }
        }

        context.set_meter(P::Activity, if self.masker.is_dropping() { 1.0 } else { 0.0 });
        ProcessStatus::Normal
    }

    fn editor(&self) -> Box<dyn Editor> {
        SlintEditor::new(
            self.params.clone(),
            (300, 320),
            |state: PluginContext<DropoutPluginParams>| -> SyncFn<DropoutPluginParams> {
                let ui = DropoutUi::new().unwrap();

                // UI → host
                let s = state.clone();
                ui.on_loss_changed(move |v| s.automate(P::Loss, f64::from(v)));
                let s = state.clone();
                ui.on_packet_changed(move |v| s.automate(P::PacketMs, f64::from(v)));
                let s = state.clone();
                ui.on_bypass_changed(move |v| s.automate(P::Bypass, if v { 1.0 } else { 0.0 }));

                // host → UI (every frame)
                Box::new(move |state: &PluginContext<DropoutPluginParams>| {
                    ui.set_loss(state.get_param(P::Loss));
                    ui.set_packet(state.get_param(P::PacketMs));
                    ui.set_bypass(state.get_param(P::Bypass) > 0.5);
                    ui.set_loss_text(slint::SharedString::from(state.format_param(P::Loss)));
                    ui.set_packet_text(slint::SharedString::from(state.format_param(P::PacketMs)));
                    ui.set_dropped(state.get_meter(P::Activity) > 0.5);
                })
            },
        )
        .min_size((240, 280))
        .max_size((900, 900))
        .into_editor()
    }
}

truce::plugin! {
    logic: DropoutPlugin,
    params: DropoutPluginParams,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_is_valid() {
        truce_test::assert_valid_info::<Plugin>();
    }

    #[test]
    fn has_editor() {
        truce_test::assert_has_editor::<Plugin>();
    }

    #[test]
    fn state_round_trips() {
        truce_test::assert_state_round_trip::<Plugin>();
    }

    // Pins the rendered editor against screenshots/default.png. Regenerate the
    // baseline with `cargo truce screenshot --out screenshots/default.png`.
    #[cfg(target_os = "macos")]
    #[test]
    fn gui_screenshot() {
        truce_test::screenshot!(Plugin, "screenshots/default.png")
            .pixel_threshold(2)
            .run();
    }
}
