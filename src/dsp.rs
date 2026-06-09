//! Streaming packet-loss masker.
//!
//! Splits the incoming sample stream into fixed-size packets and, at each
//! packet boundary, runs the [`LossGen`] model once to decide keep/drop. A
//! dropped packet is hard-zeroed (`gain = 0.0`), faithful to the reference
//! snippet's `signal * (1 - mask)`. The model's autoregressive conditioning on
//! the previous decision produces realistic *bursty* loss.
//!
//! Everything here is allocation-free and runs on the audio thread.

use crate::model::LossGen;

/// Minimal PCG32 PRNG (PCG-XSH-RR 64/32). Deterministic from a seed, no heap,
/// no `std::random`. Used to draw the per-packet decision threshold.
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    const MULT: u64 = 6_364_136_223_846_793_005;

    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (seed << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed ^ 0x853c_49e6_748f_ea9b);
        rng.next_u32();
        rng
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULT).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform `f32` in `[0, 1)` using the top 24 bits.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Per-packet keep/drop generator + sample-rate masking.
pub struct PacketLossMasker {
    model: LossGen,
    rng: Pcg32,
    /// Previous drop decision fed back into the model (0.0 keep / 1.0 drop).
    last: f32,
    /// Samples per packet (>= 1).
    packet_size: usize,
    /// Samples elapsed into the current packet.
    pos: usize,
    /// Whether the current packet is dropped.
    current_dropped: bool,
}

impl PacketLossMasker {
    pub fn new(sample_rate: f64, packet_ms: f32, seed: u64) -> Self {
        let mut m = Self {
            model: LossGen::new(),
            rng: Pcg32::new(seed),
            last: 0.0,
            packet_size: 1,
            pos: 0,
            current_dropped: false,
        };
        m.set_packet_size(sample_rate, packet_ms);
        m.reset(seed);
        m
    }

    /// Recompute the packet length in samples. Called on sample-rate or
    /// packet-size changes; clamps the in-packet cursor so the next sample
    /// lands on a fresh boundary if the old cursor overran.
    pub fn set_packet_size(&mut self, sample_rate: f64, packet_ms: f32) {
        let samples = (sample_rate * f64::from(packet_ms) / 1000.0).round() as usize;
        self.packet_size = samples.max(1);
        if self.pos >= self.packet_size {
            self.pos = 0;
        }
    }

    /// Restart the loss pattern deterministically: reseed the PRNG, zero the
    /// model state and feedback, and force a fresh decision on the next sample.
    pub fn reset(&mut self, seed: u64) {
        self.rng = Pcg32::new(seed);
        self.model.reset_state();
        self.last = 0.0;
        self.pos = 0;
        self.current_dropped = false;
    }

    /// Run one packet decision. `perc` is the target loss as a fraction in
    /// `[0, 1]` (the model's training units).
    #[inline]
    fn decide(&mut self, perc: f32) {
        let prob = self.model.prob(self.last, perc);
        let threshold = self.rng.next_f32();
        let dropped = threshold < prob; // 1 -> drop, mirrors the snippet
        self.current_dropped = dropped;
        self.last = if dropped { 1.0 } else { 0.0 };
    }

    /// Gain for the next output sample: `0.0` if the current packet is dropped,
    /// else `1.0`. Advances the packet cursor and triggers a new decision at
    /// each boundary.
    #[inline]
    pub fn next_sample_gain(&mut self, perc: f32) -> f32 {
        if self.pos == 0 {
            self.decide(perc);
        }
        let gain = if self.current_dropped { 0.0 } else { 1.0 };
        self.pos += 1;
        if self.pos >= self.packet_size {
            self.pos = 0;
        }
        gain
    }

    /// Whether the most recent packet is dropped (for the UI indicator).
    pub fn is_dropping(&self) -> bool {
        self.current_dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_zero_and_passthrough() {
        // packet_size = 4 samples; drive decisions via a fixed model is hard,
        // so instead verify the gain holds constant across a packet and the
        // cursor wraps correctly by inspecting two full packets.
        let mut m = PacketLossMasker::new(48_000.0, 1000.0 / 12_000.0, 0);
        // packet_size = round(48000 * (1000/12000)/1000) = round(4) = 4
        assert_eq!(m.packet_size, 4);
        let g0 = m.next_sample_gain(0.5);
        // gain is constant within the packet
        for _ in 0..3 {
            assert_eq!(m.next_sample_gain(0.5), g0);
        }
        // new packet boundary -> a (possibly different) constant gain
        let g1 = m.next_sample_gain(0.5);
        for _ in 0..3 {
            assert_eq!(m.next_sample_gain(0.5), g1);
        }
        assert!(g0 == 0.0 || g0 == 1.0);
        assert!(g1 == 0.0 || g1 == 1.0);
    }

    #[test]
    fn reset_is_deterministic() {
        let mut a = PacketLossMasker::new(48_000.0, 20.0, 42);
        let seq_a: Vec<f32> = (0..5000).map(|_| a.next_sample_gain(0.3)).collect();
        let mut b = PacketLossMasker::new(48_000.0, 20.0, 42);
        let seq_b: Vec<f32> = (0..5000).map(|_| b.next_sample_gain(0.3)).collect();
        assert_eq!(seq_a, seq_b, "same seed must produce identical patterns");
    }

    #[test]
    fn realized_loss_tracks_target() {
        // perc is a fraction; realized drop ratio should land near it.
        let mut m = PacketLossMasker::new(48_000.0, 20.0, 7);
        let packet = m.packet_size;
        let packets = 20_000;
        let mut dropped = 0usize;
        for _ in 0..packets {
            let g = m.next_sample_gain(0.2);
            if g == 0.0 {
                dropped += 1;
            }
            // skip the rest of the packet (gain constant within it)
            for _ in 1..packet {
                m.next_sample_gain(0.2);
            }
        }
        let ratio = dropped as f32 / packets as f32;
        assert!(
            (ratio - 0.2).abs() < 0.05,
            "realized loss {ratio} should track target 0.2"
        );
    }
}
