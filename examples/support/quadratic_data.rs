//! Shared deterministic input fixture for the batch and recursive tutorials.
//! Contains no fitting algorithm, estimator state, or reporting machinery.

// ANCHOR: inputs
pub(crate) const SAMPLE_COUNT: usize = 40;
pub(crate) const CURVE_COUNT: usize = 201;
pub(crate) const REFERENCE_COEFFICIENTS: [f64; 3] = [0.6, -0.8, 1.5];
pub(crate) const NOISE_SCALE: f64 = 1.0;
// Fixed perturbations: mean approximately zero, RMS approximately 0.7.
// These are reproducible teaching inputs, not a runtime random-number stream.
#[rustfmt::skip]
pub(crate) const NOISE: [f64; SAMPLE_COUNT] = [
    -0.924, -0.068, 0.513, -0.043, 0.157, -0.789, -0.072, -0.157, 0.607, 0.161, 0.06, -0.352,
    -0.414, 0.065, 0.828, -1.027, -1.552, 1.223, 1.006, 0.455, 0.294, -0.154, -0.125, -0.611,
    0.435, -0.036, -1.039, -0.215, -0.236, -0.295, 1.211, 0.715, -1.759, 0.37, 0.707, 0.856,
    0.552, -1.076, 0.6, 0.129,
];
// ANCHOR_END: inputs

pub(crate) fn sample_x(row: usize) -> f64 {
    -3.0 + 6.0 * row as f64 / (SAMPLE_COUNT - 1) as f64
}

pub(crate) fn curve_x(row: usize) -> f64 {
    -3.0 + 6.0 * row as f64 / (CURVE_COUNT - 1) as f64
}

pub(crate) fn reference_y(x: f64) -> f64 {
    let [a, b, c] = REFERENCE_COEFFICIENTS;
    a * x * x + b * x + c
}
