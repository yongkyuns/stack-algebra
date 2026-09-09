#![no_std]

// Compile the very same application helper for a bare-metal target; no allocator.
#[path = "../../../examples/support/quadratic_rls.rs"]
pub mod rls;

pub fn update_f32(
    estimator: &mut rls::QuadraticRls<f32>,
    x: f32,
    y: f32,
) -> Result<rls::Update<f32>, rls::RlsError> {
    estimator.update(x, y)
}

pub fn update_f64(
    estimator: &mut rls::QuadraticRls<f64>,
    x: f64,
    y: f64,
) -> Result<rls::Update<f64>, rls::RlsError> {
    estimator.update(x, y)
}
