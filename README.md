# Dropout Plugin

A real-time VST3 / CLAP / AU audio plugin that simulates **realistic, bursty
packet loss** — the same behaviour as the reference `LossGen` Python snippet —
live in your DAW. Built with [truce.audio](https://truce.audio) (Rust plugin
framework) and a [Slint](https://slint.dev) GUI.

## How it works

The incoming audio stream is split into fixed-size packets. At each packet
boundary, the tiny recurrent **LossGen** model (~1.4K params: `dense_in 2→8`,
`GRU 8→8`, `GRU 8→16`, `dense_out 16→1`) emits a drop probability conditioned on
the *previous* decision and the target loss fraction. A seeded PRNG threshold
turns that into a keep/drop choice. Dropped packets are **hard-zeroed**
(`out = in * (1 - mask)`), faithful to the snippet.

The model's autoregressive conditioning is what makes the loss *bursty* (drops
cluster) instead of independent random — the whole point of the model.

The model is hand-ported to pure Rust (no ONNX/Torch at runtime); weights are
baked into the binary as `const` arrays. A model step is a few hundred
multiply-adds with no allocation, so it runs directly on the audio thread.

## Controls

| Control | Range | Notes |
|---------|-------|-------|
| **Loss**   | 0–100 % | Target loss fraction, fed to the model as `perc`. |
| **Packet** | 2–80 ms | Packet size = decision granularity. |
| **Bypass** | on/off  | Clean passthrough. |
| LOSS LED   | —       | Lights while a packet is being dropped. |

A **Seed** parameter (default 42) controls the deterministic loss pattern. It's
not on the GUI — it's a set-and-forget control — but it's saved with the project
and automatable from the host's generic parameter view.

## Build & run

```bash
# Standalone app (no DAW needed)
cargo truce run

# Build + install into the DAW plug-in folders (CLAP + VST3 by default)
cargo truce install --clap --vst3        # user folders
cargo truce install --au2                # AU (add the feature if needed)

# Tests (model parity, masker behaviour, GUI screenshot)
cargo test

# Re-render the GUI baseline
cargo truce screenshot --out screenshots/default.png
```

## Regenerating model weights

The Rust weights are generated from the PyTorch checkpoint. Re-run after
replacing `lossgen_2000.pth`:

```bash
python3 tools/export_weights.py      # -> src/model/weights.rs
python3 tools/dump_reference.py      # -> src/model/test_vectors.rs (parity test)
cargo test                           # confirm the port still matches PyTorch
```

## Layout

```
src/lib.rs            Plugin: params, process(), Slint editor wiring
src/model/mod.rs      Pure-Rust LossGen (GRU math)
src/model/weights.rs  @generated const weights
src/dsp.rs            PacketLossMasker + PCG32 PRNG
ui/main.slint         GUI markup
tools/                weight + reference-vector exporters
```
