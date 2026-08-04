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
fn sparse_ldlt_reuses_factor_storage() {
    let matrix = indefinite_matrix();
    let pattern = StaticCscLdltPattern::<3, 6>::analyze(&matrix).unwrap();
    let mut factor = StaticCscLdlt::<3, 6, f64>::decompose(&matrix).unwrap();
    pattern.factor_ldlt_into(&matrix, &mut factor).unwrap();
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
    let dense =
        matrix![4.0, 1.0, 1.0, 1.0; 1.0, 4.0, 0.0, 0.0; 1.0, 0.0, 4.0, 0.0; 1.0, 0.0, 0.0, 4.0];
    let rhs = vector![1.0; 2.0; 3.0; 4.0];

    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);
}
