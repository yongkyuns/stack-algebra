use approx::assert_relative_eq;
use stack_algebra::{
    matrix, vector, SparseCholeskyError, StaticCscLdlt, StaticCscLdltPattern, StaticCscMatrix,
    StaticCscOrdering,
};

type Indefinite = StaticCscMatrix<3, 3, 6, f64>;

fn indefinite_matrix() -> Indefinite {
    Indefinite::from_pattern(
        &[4.0, 1.0, 2.0, -3.0, 1.0, 2.0],
        &[0, 1, 2, 1, 2, 2],
        &[0, 3, 5, 6],
    )
    .unwrap()
}

#[test]
fn sparse_ldlt_solves_indefinite_system() {
    let matrix = indefinite_matrix();
    let dense = matrix![4.0, 1.0, 2.0; 1.0, -3.0, 1.0; 2.0, 1.0, 2.0];
    let rhs = vector![1.0; 2.0; 3.0];
    let factor = StaticCscLdlt::<3, 6, f64>::decompose(&matrix).unwrap();

    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);
    assert_eq!(factor.lower().get(0, 0), Some(&1.0));
    assert!(factor.diagonal().iter().all(|value| value.is_finite()));
}

#[test]
fn sparse_ldlt_unified_factor_uses_native_sparse_path() {
    let matrix = indefinite_matrix();
    let dense = matrix![4.0, 1.0, 2.0; 1.0, -3.0, 1.0; 2.0, 1.0, 2.0];
    let rhs = vector![1.0; 2.0; 3.0];
    let factor = matrix.try_ldlt_with_dense_fallback::<6>().unwrap();

    assert!(!factor.uses_dense_fallback());
    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);

    let mut in_place = rhs;
    factor.solve_in_place(&mut in_place);
    assert_relative_eq!(dense * in_place, rhs, max_relative = 1e-12);
}

#[test]
fn sparse_ldlt_reuses_factor_storage() {
    let matrix = indefinite_matrix();
    let mut factor = StaticCscLdlt::<3, 6, f64>::decompose(&matrix).unwrap();
    let pattern = StaticCscLdltPattern::<3, 6>::analyze(&matrix).unwrap();
    factor.recompute_with_pattern(&pattern, &matrix).unwrap();
    let rhs = vector![2.0; -1.0; 4.0];
    let dense = matrix![4.0, 1.0, 2.0; 1.0, -3.0, 1.0; 2.0, 1.0, 2.0];

    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);
}

#[test]
fn sparse_ldlt_reports_zero_pivot_without_pivoting() {
    type Singular = StaticCscMatrix<2, 2, 3, f64>;
    let matrix = Singular::from_pattern(&[0.0, 1.0, 2.0], &[0, 1, 1], &[0, 2, 3]).unwrap();

    assert_eq!(
        StaticCscLdlt::<2, 3, f64>::decompose(&matrix),
        Err(SparseCholeskyError::ZeroPivot)
    );
}

#[test]
fn sparse_ldlt_diagonal_pivoting_recovers_zero_leading_diagonal() {
    type Pivoted = StaticCscMatrix<2, 2, 3, f64>;
    let matrix = Pivoted::from_pattern(&[0.0, 1.0, 2.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let dense = matrix![0.0, 1.0; 1.0, 2.0];
    let rhs = vector![3.0; 4.0];
    let factor =
        StaticCscLdlt::<2, 3, f64>::decompose_with_diagonal_pivoting(&matrix, 1e-12).unwrap();

    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);
    assert_ne!(factor.ordering().permutation(), &[0, 1]);
}

#[test]
fn sparse_ldlt_diagonal_pivoting_enforces_threshold() {
    type NearZero = StaticCscMatrix<2, 2, 3, f64>;
    let matrix = NearZero::from_pattern(&[1.0e-8, 0.0, 2.0], &[0, 1, 1], &[0, 2, 3]).unwrap();

    assert_eq!(
        StaticCscLdlt::<2, 3, f64>::decompose_with_diagonal_pivoting(&matrix, 1.0e-6),
        Err(SparseCholeskyError::ZeroPivot)
    );
    assert!(StaticCscLdlt::<2, 3, f64>::decompose_with_diagonal_pivoting(&matrix, 1.0e-10).is_ok());
}

#[test]
fn sparse_ldlt_diagonal_pivoting_rejects_non_finite_threshold() {
    type NearZero = StaticCscMatrix<2, 2, 3, f64>;
    let matrix = NearZero::from_pattern(&[1.0, 0.0, 2.0], &[0, 1, 1], &[0, 2, 3]).unwrap();

    assert_eq!(
        StaticCscLdlt::<2, 3, f64>::decompose_with_diagonal_pivoting(&matrix, f64::NAN,),
        Err(SparseCholeskyError::NonFinite)
    );
}

#[test]
fn sparse_ldlt_dense_fallback_handles_two_by_two_pivot() {
    type Pivoted = StaticCscMatrix<2, 2, 3, f64>;
    let matrix = Pivoted::from_pattern(&[1.0e-6, 1.0, 1.0e-6], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let dense = matrix![1.0e-6, 1.0; 1.0, 1.0e-6];
    let rhs = vector![3.0; 4.0];
    let factor = matrix.try_dense_ldlt().unwrap();

    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-10);
    assert_eq!(factor.pivot_blocks(), &[2, 3]);
}

#[test]
fn sparse_ldlt_unified_factor_falls_back_for_two_by_two_pivot() {
    type Pivoted = StaticCscMatrix<2, 2, 3, f64>;
    let matrix = Pivoted::from_pattern(&[0.0, 1.0, 0.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let dense = matrix![0.0, 1.0; 1.0, 0.0];
    let rhs = vector![3.0; 4.0];
    let factor = matrix.try_ldlt_with_dense_fallback::<3>().unwrap();

    assert!(factor.uses_dense_fallback());
    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);

    let mut in_place = rhs;
    factor.solve_in_place(&mut in_place);
    assert_relative_eq!(dense * in_place, rhs, max_relative = 1e-12);
}

#[test]
fn sparse_ldlt_unified_factor_switches_on_numeric_recompute() {
    type Pivoted = StaticCscMatrix<2, 2, 3, f64>;
    let native = Pivoted::from_pattern(&[2.0, 1.0, 3.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let requires_two_by_two =
        Pivoted::from_pattern(&[0.0, 1.0, 0.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let dense = matrix![0.0, 1.0; 1.0, 0.0];
    let rhs = vector![3.0; 4.0];
    let mut factor = native.try_ldlt_with_dense_fallback::<3>().unwrap();

    assert!(!factor.uses_dense_fallback());
    factor
        .recompute_with_dense_fallback(&requires_two_by_two)
        .unwrap();
    assert!(factor.uses_dense_fallback());
    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);
}

#[test]
fn sparse_ldlt_unified_factor_prefers_sparse_diagonal_pivoting() {
    type Pivoted = StaticCscMatrix<2, 2, 3, f64>;
    let matrix = Pivoted::from_pattern(&[0.0, 1.0, 2.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let dense = matrix![0.0, 1.0; 1.0, 2.0];
    let rhs = vector![3.0; 4.0];
    let factor = matrix.try_ldlt_with_dense_fallback::<3>().unwrap();

    assert!(!factor.uses_dense_fallback());
    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);
}

#[test]
fn sparse_ldlt_unified_factor_uses_scale_relative_threshold() {
    type Scaled = StaticCscMatrix<2, 2, 3, f64>;
    let matrix = Scaled::from_pattern(&[0.0, 1.0e-20, 2.0e-20], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let dense = matrix![0.0, 1.0e-20; 1.0e-20, 2.0e-20];
    let rhs = vector![3.0e-20; 4.0e-20];
    let factor = matrix.try_ldlt_with_dense_fallback::<3>().unwrap();

    assert!(!factor.uses_dense_fallback());
    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);
}

#[test]
fn sparse_ldlt_unified_factor_accepts_explicit_threshold() {
    type Pivoted = StaticCscMatrix<2, 2, 3, f64>;
    let matrix = Pivoted::from_pattern(&[0.0, 1.0, 2.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let factor = matrix
        .try_ldlt_with_dense_fallback_threshold::<3>(10.0)
        .unwrap();

    assert!(factor.uses_dense_fallback());
}

#[test]
fn sparse_ldlt_failed_fallback_recompute_preserves_previous_factor() {
    type Pivoted = StaticCscMatrix<2, 2, 3, f64>;
    let valid = Pivoted::from_pattern(&[2.0, 1.0, 3.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let singular = Pivoted::from_pattern(&[0.0, 0.0, 0.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let dense = matrix![2.0, 1.0; 1.0, 3.0];
    let rhs = vector![3.0; 4.0];
    let mut factor = valid.try_ldlt_with_dense_fallback::<3>().unwrap();

    assert!(!factor.uses_dense_fallback());
    assert_eq!(
        factor.recompute_with_dense_fallback(&singular),
        Err(SparseCholeskyError::ZeroPivot)
    );
    assert!(!factor.uses_dense_fallback());
    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);
}

#[test]
fn sparse_ldlt_ordering_preserves_solution() {
    type Star = StaticCscMatrix<4, 4, 10, f64>;
    let matrix = Star::from_pattern(
        &[4.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
        &[0, 1, 2, 3, 1, 2, 3],
        &[0, 4, 5, 6, 7],
    )
    .unwrap();
    let ordering = StaticCscOrdering::minimum_degree(&matrix);
    let pattern = StaticCscLdltPattern::<4, 10>::analyze_with_ordering(&matrix, ordering).unwrap();
    let ordered = pattern.prepare_ordered(&matrix).unwrap();
    let factor = pattern.factor_ldlt_ordered(&ordered).unwrap();
    let mut reusable_factor = pattern.factor_ldlt_ordered(&ordered).unwrap();
    reusable_factor.recompute_ordered(&ordered).unwrap();
    let dense =
        matrix![4.0, 1.0, 1.0, 1.0; 1.0, 4.0, 0.0, 0.0; 1.0, 0.0, 4.0, 0.0; 1.0, 0.0, 0.0, 4.0];
    let rhs = vector![1.0; 2.0; 3.0; 4.0];

    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);
    assert_relative_eq!(
        dense * reusable_factor.solve(&rhs),
        rhs,
        max_relative = 1e-12
    );
}
