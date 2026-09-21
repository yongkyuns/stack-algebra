//! Absolute rank-cutoff regressions needed by robotics controller migration.
use stack_algebra::{matrix, Matrix};

#[test]
fn f32_factorization_retains_small_nonzero_directions() {
    let a = matrix![1.0e6_f32, 0.0; 0.0, 1.0e-3];
    let factor = a.try_svd().unwrap();
    assert_eq!(factor.rank(), 1); // Default relative rank still excludes it.
    assert_eq!(factor.singular_values()[1], 1.0e-3);
    assert_eq!(factor.with_threshold(0.0).rank(), 2);
    let inverse = factor.pseudo_inverse_with_absolute_threshold(1.0e-6);
    assert!((a * inverse - Matrix::eye()).norm() < 2.0e-6);
    assert!(factor.pseudo_inverse()[(1, 1)] == 0.0);
}

#[test]
fn tall_qr_reduction_preserves_absolute_cutoff_choice() {
    let a = matrix![1.0e6_f32, 0.0; 0.0, 1.0e-3; 0.0, 0.0];
    let factor = a.try_svd().unwrap();
    let inverse = factor.pseudo_inverse_with_absolute_threshold(1.0e-6);
    assert!((inverse * a - Matrix::eye()).norm() < 2.0e-6);
}

#[test]
fn cutoff_is_absolute_and_strict_and_handles_zero() {
    let a = matrix![10.0_f64, 0.0; 0.0, 0.25];
    let factor = a.try_svd().unwrap();
    assert_eq!(
        factor.pseudo_inverse_with_absolute_threshold(0.25)[(1, 1)],
        0.0
    );
    assert_eq!(
        factor.pseudo_inverse_with_absolute_threshold(0.125)[(1, 1)],
        4.0
    );
    assert_eq!(
        factor.pseudo_inverse_with_absolute_threshold(f64::INFINITY),
        Matrix::zeros()
    );
    assert_eq!(
        Matrix::<2, 2, f64>::zeros()
            .try_svd()
            .unwrap()
            .pseudo_inverse_with_absolute_threshold(0.0),
        Matrix::zeros()
    );
}

#[test]
fn agrees_with_nalgebra_absolute_cutoff_on_scaled_correlated_systems() {
    for scale in [1.0e-4_f64, 1.0, 1.0e4] {
        let a = matrix![4.0, 1.0; 1.0, 3.0] * scale;
        let reference = nalgebra::SMatrix::<f64, 2, 2>::from_column_slice(a.as_slice());
        for cutoff in [0.0, 1.0e-6, 3.0 * scale, 10.0 * scale] {
            let expected = reference.pseudo_inverse(cutoff).unwrap();
            let actual = a
                .try_svd()
                .unwrap()
                .pseudo_inverse_with_absolute_threshold(cutoff);
            let error = actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(error <= 1.0e-10 * (1.0 + expected.norm()));
        }
    }
}

#[test]
#[should_panic(expected = "SVD cutoff")]
fn negative_absolute_cutoff_is_rejected() {
    Matrix::<2, 2, f32>::eye()
        .try_svd()
        .unwrap()
        .pseudo_inverse_with_absolute_threshold(-1.0);
}

#[test]
#[should_panic(expected = "SVD cutoff")]
fn nan_absolute_cutoff_is_rejected() {
    Matrix::<2, 2, f32>::eye()
        .try_svd()
        .unwrap()
        .pseudo_inverse_with_absolute_threshold(f32::NAN);
}
