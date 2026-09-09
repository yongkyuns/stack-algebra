//! Independent checks for the noisy quadratic tutorial and its dense plot grid.

#[path = "../examples/mapped_least_squares.rs"]
mod example;

use example::{
    curve_design, curve_x, design_storage, fit_quadratic, make_observations, reference_y, sample_x,
    CURVE_COUNT, NOISE, NOISE_SCALE, REFERENCE_COEFFICIENTS, SAMPLE_COUNT,
};
use stack_algebra::{DecompositionError, Map, Matrix};

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        (actual - expected).abs() <= 1.0e-11,
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn sample_fit_matches_independent_orthogonal_projection() {
    let storage = design_storage();
    let design = Map::<SAMPLE_COUNT, 3, f64>::from_slice(&storage).unwrap();
    let observations = make_observations();
    let coefficients = fit_quadratic(&design, &observations).unwrap();

    // On this symmetric grid, 1, x, and (x^2 - mean(x^2)) are orthogonal.
    // Scalar projection gives an independent reference, without QR or a solver.
    let n = SAMPLE_COUNT as f64;
    let mean_y = observations.as_slice().iter().sum::<f64>() / n;
    let sum_x2: f64 = (0..SAMPLE_COUNT).map(|i| sample_x(i).powi(2)).sum();
    let mean_x2 = sum_x2 / n;
    let mut q2_y = 0.0;
    let mut q2_norm = 0.0;
    let mut x_y = 0.0;
    for row in 0..SAMPLE_COUNT {
        let x = sample_x(row);
        let q2 = x * x - mean_x2;
        q2_y += q2 * observations[(row, 0)];
        q2_norm += q2 * q2;
        x_y += x * observations[(row, 0)];
    }
    let a = q2_y / q2_norm;
    let b = x_y / sum_x2;
    let c = mean_y - a * mean_x2;
    assert_close(coefficients[(0, 0)], a);
    assert_close(coefficients[(1, 0)], b);
    assert_close(coefficients[(2, 0)], c);
}

#[test]
fn synthetic_inputs_have_visible_reproducible_noise_and_correct_layout() {
    let storage = design_storage();
    let observations = make_observations();
    let [a, b, c] = REFERENCE_COEFFICIENTS;
    for (row, &noise) in NOISE.iter().enumerate() {
        let x = sample_x(row);
        assert_close(storage[row], x * x);
        assert_close(storage[SAMPLE_COUNT + row], x);
        assert_close(storage[2 * SAMPLE_COUNT + row], 1.0);
        assert_close(reference_y(x), a * x * x + b * x + c);
        assert_close(observations[(row, 0)], reference_y(x) + NOISE_SCALE * noise);
    }
    assert_close(sample_x(0), -3.0);
    assert_close(sample_x(SAMPLE_COUNT - 1), 3.0);
    assert_close(NOISE.iter().sum::<f64>(), 0.0);
    let rms = (NOISE.iter().map(|n| n * n).sum::<f64>() / SAMPLE_COUNT as f64).sqrt();
    assert!((0.69..0.71).contains(&rms));
}

#[test]
fn predictions_and_residuals_match_scalar_evaluation_and_normality_conditions() {
    let storage = design_storage();
    let design = Map::<SAMPLE_COUNT, 3, f64>::from_slice(&storage).unwrap();
    let observations = make_observations();
    let coefficients = fit_quadratic(&design, &observations).unwrap();
    let mut fitted = Matrix::<SAMPLE_COUNT, 1, f64>::zeros();
    design.matvec_into(&coefficients, &mut fitted);
    let residuals = observations - fitted;
    let [a, b, c] = [coefficients[(0, 0)], coefficients[(1, 0)], coefficients[(2, 0)]];
    for row in 0..SAMPLE_COUNT {
        let x = sample_x(row);
        assert_close(fitted[(row, 0)], (a * x + b) * x + c);
        assert_close(residuals[(row, 0)], observations[(row, 0)] - fitted[(row, 0)]);
    }
    assert!(residuals.norm() > 3.0);
    for column in 0..3 {
        let projection: f64 = (0..SAMPLE_COUNT)
            .map(|row| design[(row, column)] * residuals[(row, 0)])
            .sum();
        assert_close(projection, 0.0);
    }
}

#[test]
fn exact_quadratic_recovers_coefficients_after_reordering() {
    let mut storage = design_storage();
    for column in storage.chunks_exact_mut(SAMPLE_COUNT) {
        column.reverse();
    }
    let design = Map::<SAMPLE_COUNT, 3, f64>::from_slice(&storage).unwrap();
    let observations = Matrix::<SAMPLE_COUNT, 1, f64>::from_fn(|row, _| {
        let x = design[(row, 1)];
        -0.25 * x * x + 1.2 * x + 4.5
    });
    let coefficients = fit_quadratic(&design, &observations).unwrap();
    for (row, expected) in [-0.25, 1.2, 4.5].into_iter().enumerate() {
        assert_close(coefficients[(row, 0)], expected);
    }
}

#[test]
fn dense_grid_evaluates_fitted_coefficients_not_the_reference_curve() {
    let coefficients = Matrix::<3, 1, f64>::from_columns([[0.2, 0.3, 2.1]]);
    let grid = curve_design();
    let curve = grid * coefficients;
    assert_close(curve_x(0), -3.0);
    assert_close(curve_x(CURVE_COUNT - 1), 3.0);
    for row in 0..CURVE_COUNT {
        let x = curve_x(row);
        assert_close(grid[(row, 0)], x * x);
        assert_close(grid[(row, 1)], x);
        assert_close(grid[(row, 2)], 1.0);
        assert_close(curve[(row, 0)], (0.2 * x + 0.3) * x + 2.1);
        if row > 0 {
            assert!(x > curve_x(row - 1));
        }
    }
    assert!((curve[(0, 0)] - reference_y(-3.0)).abs() > 1.0);
}

#[test]
fn map_borrows_caller_storage_and_fit_preserves_inputs() {
    let storage = design_storage();
    let original = storage;
    let observations = make_observations();
    let original_observations = observations;
    let design = Map::<SAMPLE_COUNT, 3, f64>::from_slice(&storage).unwrap();
    assert!(core::ptr::eq(design.as_slice().as_ptr(), storage.as_ptr()));
    fit_quadratic(&design, &observations).unwrap();
    assert_eq!(storage, original);
    assert_eq!(observations, original_observations);
}

#[test]
fn fewer_than_three_distinct_inputs_are_rank_deficient() {
    for distinct in [1, 2] {
        let mut storage = design_storage();
        for row in 0..SAMPLE_COUNT {
            let x = (row % distinct) as f64;
            storage[row] = x * x;
            storage[SAMPLE_COUNT + row] = x;
        }
        let design = Map::<SAMPLE_COUNT, 3, f64>::from_slice(&storage).unwrap();
        assert_eq!(
            fit_quadratic(&design, &make_observations()),
            Err(DecompositionError::Singular)
        );
    }
}

#[test]
fn nonfinite_design_or_observation_is_rejected() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut storage = design_storage();
        storage[2] = invalid;
        let design = Map::<SAMPLE_COUNT, 3, f64>::from_slice(&storage).unwrap();
        assert_eq!(
            fit_quadratic(&design, &make_observations()),
            Err(DecompositionError::NonFinite)
        );
        let storage = design_storage();
        let design = Map::<SAMPLE_COUNT, 3, f64>::from_slice(&storage).unwrap();
        let mut observations = make_observations();
        observations[(2, 0)] = invalid;
        assert_eq!(
            fit_quadratic(&design, &observations),
            Err(DecompositionError::NonFinite)
        );
    }
}
