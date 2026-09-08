//! Correctness checks kept separate from the mapped line-fitting tutorial.

#[path = "../examples/mapped_least_squares.rs"]
mod example;

use example::{fit_line, DESIGN_STORAGE, OBSERVATIONS};
use stack_algebra::{DecompositionError, Map, Matrix};

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn sample_fit_matches_independent_scalar_regression() {
    let design = Map::<5, 2, f64>::from_slice(&DESIGN_STORAGE).unwrap();
    let observations = Matrix::<5, 1, f64>::from_columns([OBSERVATIONS]);
    let coefficients = fit_line(&design, &observations).unwrap();

    // Independent reference: centered scalar least squares, without QR/matrices.
    let x = &DESIGN_STORAGE[..5];
    let mean_x = x.iter().sum::<f64>() / 5.0;
    let mean_y = OBSERVATIONS.iter().sum::<f64>() / 5.0;
    let numerator: f64 = x
        .iter()
        .zip(OBSERVATIONS)
        .map(|(&x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let denominator: f64 = x.iter().map(|&x| (x - mean_x).powi(2)).sum();
    let slope = numerator / denominator;
    let intercept = mean_y - slope * mean_x;
    assert_close(coefficients[(0, 0)], slope);
    assert_close(coefficients[(1, 0)], intercept);
    assert_close(slope, 1.97);
    assert_close(intercept, 1.06);
}

#[test]
fn fitted_values_and_nonzero_residual_match_hand_calculation() {
    let design = Map::<5, 2, f64>::from_slice(&DESIGN_STORAGE).unwrap();
    let observations = Matrix::<5, 1, f64>::from_columns([OBSERVATIONS]);
    let coefficients = fit_line(&design, &observations).unwrap();
    let mut fitted = Matrix::<5, 1, f64>::zeros();
    design.matvec_into(&coefficients, &mut fitted);
    let residuals = observations - fitted;

    let expected_fitted = [1.06, 3.03, 5.0, 6.97, 8.94];
    let expected_residuals = [0.04, -0.13, 0.20, -0.17, 0.06];
    for row in 0..5 {
        assert_close(fitted[(row, 0)], expected_fitted[row]);
        assert_close(residuals[(row, 0)], expected_residuals[row]);
    }
    assert_close(residuals.norm(), 0.091_f64.sqrt());
    assert!(residuals.norm() > 0.3);
    // At the least-squares optimum the residual is orthogonal to both columns.
    assert_close(residuals.as_slice().iter().sum(), 0.0);
    let x_dot_residual: f64 = (0..5)
        .map(|row| design[(row, 0)] * residuals[(row, 0)])
        .sum();
    assert_close(x_dot_residual, 0.0);
}

#[test]
fn exact_line_recovers_coefficients_with_reordered_samples() {
    let storage = [3.0, -1.0, 2.0, 0.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let design = Map::<5, 2, f64>::from_slice(&storage).unwrap();
    let observations = Matrix::<5, 1, f64>::from_fn(|row, _| -0.5 * storage[row] + 2.0);
    let coefficients = fit_line(&design, &observations).unwrap();
    assert_close(coefficients[(0, 0)], -0.5);
    assert_close(coefficients[(1, 0)], 2.0);
}

#[test]
fn map_borrows_caller_storage_and_fit_preserves_inputs() {
    let storage = DESIGN_STORAGE;
    let original = storage;
    let observations = Matrix::<5, 1, f64>::from_columns([OBSERVATIONS]);
    let original_observations = observations;
    let design = Map::<5, 2, f64>::from_slice(&storage).unwrap();
    assert!(core::ptr::eq(design.as_slice().as_ptr(), storage.as_ptr()));
    fit_line(&design, &observations).unwrap();
    assert_eq!(storage, original);
    assert_eq!(observations, original_observations);
}

#[test]
fn identical_x_values_are_rank_deficient() {
    let storage = [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let design = Map::<5, 2, f64>::from_slice(&storage).unwrap();
    let observations = Matrix::<5, 1, f64>::from_columns([OBSERVATIONS]);
    assert_eq!(
        fit_line(&design, &observations),
        Err(DecompositionError::Singular)
    );
}

#[test]
fn nonfinite_design_or_observation_is_rejected() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut storage = DESIGN_STORAGE;
        storage[2] = invalid;
        let design = Map::<5, 2, f64>::from_slice(&storage).unwrap();
        let observations = Matrix::<5, 1, f64>::from_columns([OBSERVATIONS]);
        assert_eq!(
            fit_line(&design, &observations),
            Err(DecompositionError::NonFinite)
        );

        let design = Map::<5, 2, f64>::from_slice(&DESIGN_STORAGE).unwrap();
        let mut observations = observations;
        observations[(2, 0)] = invalid;
        assert_eq!(
            fit_line(&design, &observations),
            Err(DecompositionError::NonFinite)
        );
    }
}
