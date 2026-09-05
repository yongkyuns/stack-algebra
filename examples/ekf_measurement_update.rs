use stack_algebra::Matrix;

fn main() {
    // Six-state covariance and a two-dimensional linear measurement.
    let covariance = Matrix::<6, 6, f32>::eye();
    let h = Matrix::<2, 6, f32>::from_rows([
        [1.0, 0.0, 0.2, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.1, 0.0, 0.0],
    ]);
    let measurement_noise = Matrix::<2, 2, f32>::from_rows([[0.04, 0.0], [0.0, 0.09]]);
    let residual = Matrix::<2, 1, f32>::from_rows([[0.5], [-0.25]]);

    // S = H P H^T + R. Solve S K^T = H P rather than forming S^-1.
    let innovation = h * covariance * h.transpose() + measurement_noise;
    let factor = innovation
        .try_cholesky()
        .expect("innovation covariance should be positive definite");
    let gain = factor.solve(&(h * covariance)).transpose();
    let correction = gain * residual;

    // Joseph-form covariance update preserves symmetry/PSD better than P - KHP.
    let identity = Matrix::<6, 6, f32>::eye();
    let ikh = identity - gain * h;
    let updated_covariance =
        ikh * covariance * ikh.transpose() + gain * measurement_noise * gain.transpose();

    assert!(correction.norm() > 0.0);
    assert!((updated_covariance - updated_covariance.transpose()).norm() < 1.0e-5);
}
