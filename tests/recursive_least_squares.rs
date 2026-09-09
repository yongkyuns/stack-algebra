//! Same estimator as the tutorial, compared to a different batch QR algorithm.
#[allow(dead_code)]
#[path = "../examples/support/quadratic_data.rs"]
mod data;
#[path = "../examples/support/quadratic_rls.rs"]
mod rls;

use rls::{QuadraticRls, RlsError};
use stack_algebra::{ColPivHouseholderQr, Matrix, MatrixScalar, Real};

fn sample(index: usize) -> (f64, f64) {
    let row = (17 * index) % data::SAMPLE_COUNT;
    let x = data::sample_x(row);
    (
        x,
        data::reference_y(x) + data::NOISE_SCALE * data::NOISE[row],
    )
}

fn prefix_parity<T: Real + MatrixScalar>(forgetting: f64, tolerance: f64) {
    let convert = |v: f64| T::from(v).unwrap();
    let initial = Matrix::from_columns([[convert(0.2), convert(-0.3), convert(0.1)]]);
    let delta = convert(0.01);
    let scale = convert(3.0);
    let lambda = convert(forgetting);
    let mut online = QuadraticRls::<T>::new(scale, delta, lambda, initial).unwrap();
    for count in 1..=data::SAMPLE_COUNT {
        let (x, y) = sample(count - 1);
        online.update(convert(x), convert(y)).unwrap();
        // Three prior rows + observed prefix; remaining rows are zero-weight padding.
        // lambda^count also weights the prior: it fades when forgetting is enabled.
        let mut design = Matrix::<43, 3, f64>::zeros();
        let mut rhs = Matrix::<43, 1, f64>::zeros();
        let l = lambda.to_f64().unwrap();
        let prior_weight = (delta.to_f64().unwrap() * l.powi(count as i32)).sqrt();
        for j in 0..3 {
            design[(j, j)] = prior_weight;
            rhs[(j, 0)] = prior_weight * initial[(j, 0)].to_f64().unwrap();
        }
        for i in 0..count {
            let (x, y) = sample(i);
            let u = convert(x).to_f64().unwrap() / scale.to_f64().unwrap();
            let weight = l.powi((count - 1 - i) as i32).sqrt();
            for (j, value) in [u * u, u, 1.0].into_iter().enumerate() {
                design[(3 + i, j)] = weight * value;
            }
            rhs[(3 + i, 0)] = weight * convert(y).to_f64().unwrap();
        }
        let qr = ColPivHouseholderQr::try_decompose_view(&design).unwrap();
        let expected = qr.try_solve_least_squares(&rhs).unwrap();
        let actual = online.normalized_coefficients();
        for j in 0..3 {
            let error = (actual[(j, 0)].to_f64().unwrap() - expected[(j, 0)]).abs();
            assert!(
                error < tolerance,
                "prefix {count}, coefficient {j}: error {error}"
            );
        }
        assert_eq!(online.samples(), count);
    }
}

#[test]
fn f64_matches_regularized_qr_at_every_prefix() {
    prefix_parity::<f64>(1.0, 1e-9);
}
#[test]
fn f32_matches_regularized_qr_at_every_prefix() {
    prefix_parity::<f32>(1.0, 5e-4);
}
#[test]
fn forgetting_matches_weighted_qr_including_fading_prior() {
    prefix_parity::<f64>(0.98, 1e-9);
    prefix_parity::<f32>(0.98, 5e-4);
}

#[test]
fn innovation_uses_the_pre_update_estimate() {
    let mut estimator = QuadraticRls::new(3.0_f64, 0.01, 1.0, Matrix::zeros()).unwrap();
    let update = estimator.update(0.0, 2.0).unwrap();
    assert_eq!(update.prediction_before, 0.0);
    assert_eq!(update.innovation, 2.0);
    assert!((estimator.predict(0.0).unwrap() - 2.0 / 1.01).abs() < 1e-12);
    assert!((update.coefficients[(2, 0)] - 2.0 / 1.01).abs() < 1e-12);
}

#[test]
fn invalid_inputs_and_overflow_reject_atomically_then_recover() {
    let mut estimator = QuadraticRls::new(3.0_f64, 0.01, 1.0, Matrix::zeros()).unwrap();
    estimator.update(-1.0, 2.0).unwrap();
    for (x, y, error) in [
        (f64::NAN, 1.0, RlsError::NonFiniteInput),
        (f64::INFINITY, 1.0, RlsError::NonFiniteInput),
        (0.0, f64::NEG_INFINITY, RlsError::NonFiniteInput),
        (0.0, f64::NAN, RlsError::NonFiniteInput),
        (f64::MAX, 1.0, RlsError::NumericalBreakdown),
    ] {
        let before = estimator;
        assert_eq!(estimator.update(x, y).unwrap_err(), error);
        assert_eq!(estimator, before);
        let mut control = before;
        estimator.update(1.0, 3.0).unwrap();
        control.update(1.0, 3.0).unwrap();
        assert_eq!(estimator, control);
    }
}

fn staged_overflow<T: Real + MatrixScalar>() {
    let large = T::max_value() * T::from(0.9).unwrap();
    let initial = Matrix::from_columns([[T::zero(), T::zero(), large]]);
    let mut estimator = QuadraticRls::new(T::one(), T::one(), T::one(), initial).unwrap();
    let before = estimator;
    // Finite inputs and zero innovation, but the candidate's transformed RHS overflows.
    let error = estimator.update(T::zero(), large).err();
    assert_eq!(error, Some(RlsError::NumericalBreakdown));
    assert!(estimator == before);
    let mut control = before;
    assert!(estimator.update(T::zero(), T::zero()).is_ok());
    assert!(control.update(T::zero(), T::zero()).is_ok());
    assert!(estimator == control);
}

#[test]
fn late_numerical_failure_preserves_state_in_both_precisions() {
    staged_overflow::<f32>();
    staged_overflow::<f64>();
}

#[test]
fn invalid_configuration_is_rejected() {
    for bad in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
        assert!(QuadraticRls::new(bad, 0.01, 1.0, Matrix::zeros()).is_err());
        assert!(QuadraticRls::new(3.0, bad, 1.0, Matrix::zeros()).is_err());
        assert!(QuadraticRls::new(3.0, 0.01, bad, Matrix::zeros()).is_err());
    }
    assert!(QuadraticRls::new(3.0_f64, 0.01, 1.01, Matrix::zeros()).is_err());
    let initial = Matrix::from_columns([[0.0, f64::NAN, 0.0]]);
    assert!(QuadraticRls::new(3.0_f64, 0.01, 1.0, initial).is_err());
}

#[test]
fn repeated_inputs_do_not_magically_identify_all_coefficients() {
    let mut estimator = QuadraticRls::new(3.0_f64, 0.01, 1.0, Matrix::zeros()).unwrap();
    for _ in 0..1000 {
        estimator.update(0.0, 2.0).unwrap();
    }
    let theta = estimator.coefficients();
    assert_eq!(theta[(0, 0)], 0.0);
    assert_eq!(theta[(1, 0)], 0.0);
    assert!((theta[(2, 0)] - 2000.0 / 1000.01).abs() < 1e-9);
}

#[test]
fn no_forgetting_is_order_independent_with_the_same_prior() {
    let mut left = QuadraticRls::new(3.0_f64, 0.01, 1.0, Matrix::zeros()).unwrap();
    let mut right = left;
    for i in 0..data::SAMPLE_COUNT {
        let (x, y) = sample(i);
        left.update(x, y).unwrap();
        let (x, y) = sample(data::SAMPLE_COUNT - 1 - i);
        right.update(x, y).unwrap();
    }
    assert!((left.coefficients() - right.coefficients()).norm() < 1e-10);
}

#[test]
fn long_f32_stream_stays_finite_with_constant_storage() {
    let mut estimator = QuadraticRls::new(3.0_f32, 0.01, 1.0, Matrix::zeros()).unwrap();
    let bytes = core::mem::size_of_val(&estimator);
    for i in 0..10000 {
        let x = -3.0 + 6.0 * (i % 41) as f32 / 40.0;
        let y = (0.6 * x - 0.8) * x + 1.5;
        let update = estimator.update(x, y).unwrap();
        assert!(update.coefficients.as_slice().iter().all(|v| v.is_finite()));
        assert_eq!(core::mem::size_of_val(&estimator), bytes);
    }
    let theta = estimator.coefficients();
    for (j, expected) in [0.6, -0.8, 1.5].into_iter().enumerate() {
        assert!((theta[(j, 0)] - expected).abs() < 2e-3);
    }
}
