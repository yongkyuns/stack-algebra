//! Correctness checks for the teaching example, not a navigation qualification.

#[path = "../examples/kalman_1d.rs"]
mod example;

use example::{
    predict, update_position, ACCELERATION_VARIANCE, DT, MEASUREMENTS, MEASUREMENT_VARIANCE,
};
use stack_algebra::Matrix;

fn assert_close(actual: f32, expected: f64, tolerance: f64) {
    assert!(actual.is_finite());
    assert!(
        (f64::from(actual) - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

fn assert_covariance(covariance: &Matrix<2, 2, f32>) {
    assert!(covariance.as_slice().iter().all(|value| value.is_finite()));
    let a = f64::from(covariance[(0, 0)]);
    let b = f64::from(covariance[(0, 1)]);
    let c = f64::from(covariance[(1, 0)]);
    let d = f64::from(covariance[(1, 1)]);
    assert!(a >= 0.0 && d >= 0.0);
    assert!((b - c).abs() <= 2.0e-6);
    let off_diagonal = 0.5 * (b + c);
    assert!(a * d - off_diagonal * off_diagonal >= -1.0e-7);
}

#[test]
fn prediction_matches_hand_computed_nonunit_time_step() {
    let mut state = Matrix::<2, 1, f32>::from_rows([[2.0], [-0.5]]);
    let mut covariance = Matrix::<2, 2, f32>::from_rows([[4.0, 1.0], [1.0, 3.0]]);
    predict(&mut state, &mut covariance, 0.5, 0.16);

    assert_close(state[(0, 0)], 1.75, 1.0e-6);
    assert_close(state[(1, 0)], -0.5, 1.0e-6);
    // F P F^T = [[5.75, 2.5], [2.5, 3]], Q = [[0.0025, 0.01], [0.01, 0.04]].
    for (row, expected) in [[5.7525, 2.51], [2.51, 3.04]].iter().enumerate() {
        for (column, &value) in expected.iter().enumerate() {
            assert_close(covariance[(row, column)], value, 1.0e-6);
        }
    }
    assert_covariance(&covariance);
}

#[test]
fn scalar_update_matches_hand_computed_correlated_prior() {
    let mut state = Matrix::<2, 1, f32>::from_rows([[2.0], [1.0]]);
    let mut covariance = Matrix::<2, 2, f32>::from_rows([[4.0, 2.0], [2.0, 3.0]]);
    update_position(&mut state, &mut covariance, 3.0, 1.0);

    // Innovation = 1, S = 5, K = [0.8, 0.4]^T. Velocity changes through P_vp.
    assert_close(state[(0, 0)], 2.8, 5.0e-7);
    assert_close(state[(1, 0)], 1.4, 5.0e-7);
    for (row, expected) in [[0.8, 0.4], [0.4, 2.2]].iter().enumerate() {
        for (column, &value) in expected.iter().enumerate() {
            assert_close(covariance[(row, column)], value, 5.0e-7);
        }
    }
    assert_covariance(&covariance);
}

#[test]
fn zero_innovation_preserves_state_but_reduces_uncertainty() {
    let mut state = Matrix::<2, 1, f32>::from_rows([[2.0], [-1.0]]);
    let original = state;
    let mut covariance = Matrix::<2, 2, f32>::from_rows([[4.0, 2.0], [2.0, 3.0]]);
    update_position(&mut state, &mut covariance, 2.0, 1.0);

    assert_eq!(state, original);
    assert_close(covariance[(0, 0)], 0.8, 5.0e-7);
    assert_close(covariance[(1, 1)], 2.2, 5.0e-7);
    assert_covariance(&covariance);
}

// Independent f64 reference: expanded scalar prediction and the optimal-gain
// subtractive covariance identity. No stack-algebra operations or Joseph product.
fn reference_step(state: &mut [f64; 2], covariance: &mut [f64; 3], position: f64) {
    let dt = f64::from(DT);
    let variance = f64::from(ACCELERATION_VARIANCE);
    let r = f64::from(MEASUREMENT_VARIANCE);
    let [pp, pv, vv] = *covariance;
    let a = pp + 2.0 * dt * pv + dt * dt * vv + 0.25 * dt.powi(4) * variance;
    let b = pv + dt * vv + 0.5 * dt.powi(3) * variance;
    let d = vv + dt * dt * variance;
    state[0] += dt * state[1];
    let innovation = position - state[0];
    let s = a + r;
    state[0] += a / s * innovation;
    state[1] += b / s * innovation;
    *covariance = [a - a * a / s, b - a * b / s, d - b * b / s];
}

#[test]
fn sample_sequence_matches_independent_reference_at_every_step() {
    let mut state = Matrix::<2, 1, f32>::zeros();
    let mut covariance = Matrix::<2, 2, f32>::eye();
    let mut reference_state = [0.0_f64; 2];
    let mut reference_covariance = [1.0_f64, 0.0, 1.0];

    for position in MEASUREMENTS {
        predict(&mut state, &mut covariance, DT, ACCELERATION_VARIANCE);
        update_position(&mut state, &mut covariance, position, MEASUREMENT_VARIANCE);
        reference_step(
            &mut reference_state,
            &mut reference_covariance,
            f64::from(position),
        );
        assert_close(state[(0, 0)], reference_state[0], 3.0e-6);
        assert_close(state[(1, 0)], reference_state[1], 3.0e-6);
        assert_close(covariance[(0, 0)], reference_covariance[0], 2.0e-6);
        assert_close(covariance[(0, 1)], reference_covariance[1], 2.0e-6);
        assert_close(covariance[(1, 0)], reference_covariance[1], 2.0e-6);
        assert_close(covariance[(1, 1)], reference_covariance[2], 2.0e-6);
        assert_covariance(&covariance);
    }
}

#[test]
fn repeated_updates_remain_finite_symmetric_and_track_constant_velocity() {
    let mut state = Matrix::<2, 1, f32>::zeros();
    let mut covariance = Matrix::<2, 2, f32>::eye();
    for step in 1..=1000 {
        let position = step as f32 * DT;
        predict(&mut state, &mut covariance, DT, ACCELERATION_VARIANCE);
        assert_covariance(&covariance);
        update_position(&mut state, &mut covariance, position, MEASUREMENT_VARIANCE);
        assert!(state.as_slice().iter().all(|value| value.is_finite()));
        assert_covariance(&covariance);
    }
    assert_close(state[(0, 0)], 1000.0 * f64::from(DT), 2.0e-3);
    assert_close(state[(1, 0)], 1.0, 2.0e-4);
}
