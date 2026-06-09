//! Pure-Rust port of the `LossGen` recurrent packet-loss model.
//!
//! Mirrors the reference PyTorch module exactly:
//!
//! ```text
//! x = tanh(dense_in(cat([loss, perc])))   // Linear(2 -> 8) + tanh
//! gru1_out, h1 = gru1(x,        h1)        // GRU(8  -> 8)
//! gru2_out, h2 = gru2(gru1_out, h2)        // GRU(8  -> 16)
//! logit        = dense_out(gru2_out)       // Linear(16 -> 1)
//! ```
//!
//! `step()` advances one packet and returns the raw logit; the caller applies
//! `sigmoid` to get the drop probability. All buffers are fixed-size on the
//! stack, so a step performs no allocation and is safe on the audio thread.

mod weights;
use weights::*;

/// Logistic sigmoid.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Dot product of two equal-length vectors.
#[inline]
fn dot<const N: usize>(w: &[f32; N], x: &[f32; N]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..N {
        acc += w[i] * x[i];
    }
    acc
}

/// One PyTorch-convention GRU step, in place on `h`.
///
/// Gate rows are stacked `[ r ; z ; n ]`, each `H` tall, so row `i` is the
/// reset gate, `H + i` the update gate, `2H + i` the new gate. The new gate
/// applies the reset to the *hidden* contribution only:
///
/// ```text
/// r = σ(W_ir·x + b_ir + W_hr·h + b_hr)
/// z = σ(W_iz·x + b_iz + W_hz·h + b_hz)
/// n = tanh(W_in·x + b_in + r ∘ (W_hn·h + b_hn))
/// h' = (1 − z) ∘ n + z ∘ h
/// ```
fn gru_step<const IN: usize, const H: usize, const H3: usize>(
    h: &mut [f32; H],
    x: &[f32; IN],
    w_ih: &[[f32; IN]; H3],
    w_hh: &[[f32; H]; H3],
    b_ih: &[f32; H3],
    b_hh: &[f32; H3],
) {
    let mut next = [0.0f32; H];
    for i in 0..H {
        let ri = i;
        let zi = H + i;
        let ni = 2 * H + i;

        let r = sigmoid(dot(&w_ih[ri], x) + b_ih[ri] + dot(&w_hh[ri], h) + b_hh[ri]);
        let z = sigmoid(dot(&w_ih[zi], x) + b_ih[zi] + dot(&w_hh[zi], h) + b_hh[zi]);
        let n = (dot(&w_ih[ni], x) + b_ih[ni] + r * (dot(&w_hh[ni], h) + b_hh[ni])).tanh();

        next[i] = (1.0 - z) * n + z * h[i];
    }
    *h = next;
}

/// The `LossGen` model. Holds only the two GRU hidden states; weights are
/// `const` in the binary.
pub struct LossGen {
    h1: [f32; GRU1_SIZE],
    h2: [f32; GRU2_SIZE],
}

impl Default for LossGen {
    fn default() -> Self {
        Self::new()
    }
}

impl LossGen {
    pub fn new() -> Self {
        Self {
            h1: [0.0; GRU1_SIZE],
            h2: [0.0; GRU2_SIZE],
        }
    }

    /// Zero the recurrent state (equivalent to `states = None` in the snippet).
    pub fn reset_state(&mut self) {
        self.h1 = [0.0; GRU1_SIZE];
        self.h2 = [0.0; GRU2_SIZE];
    }

    /// Advance one packet. `last` is the previous drop decision (0.0 keep /
    /// 1.0 drop), `perc` the target loss in the model's training units.
    /// Returns the raw logit (apply [`sigmoid`] for the drop probability).
    pub fn step(&mut self, last: f32, perc: f32) -> f32 {
        let inp = [last, perc];
        let mut x = [0.0f32; 8];
        for i in 0..8 {
            x[i] = (dot(&DENSE_IN_W[i], &inp) + DENSE_IN_B[i]).tanh();
        }

        gru_step(&mut self.h1, &x, &GRU1_W_IH, &GRU1_W_HH, &GRU1_B_IH, &GRU1_B_HH);
        // GRU output == hidden state; gru1's output feeds gru2 as its input.
        let g1 = self.h1;
        gru_step(&mut self.h2, &g1, &GRU2_W_IH, &GRU2_W_HH, &GRU2_B_IH, &GRU2_B_HH);

        dot(&DENSE_OUT_W[0], &self.h2) + DENSE_OUT_B[0]
    }

    /// Convenience: drop probability for this packet.
    pub fn prob(&mut self, last: f32, perc: f32) -> f32 {
        sigmoid(self.step(last, perc))
    }
}

#[cfg(test)]
mod parity_test;
