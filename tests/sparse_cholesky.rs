use approx::assert_relative_eq;
use stack_algebra::{
    matrix, vector, SparseCholeskyError, StaticCscCholesky, StaticCscCholeskyPattern,
    StaticCscMatrix, StaticCscOrdering,
};

type Sparse = StaticCscMatrix<3, 3, 7, f64>;

fn tridiagonal_spd(diagonal: [f64; 3]) -> Sparse {
    Sparse::from_pattern(
        &[diagonal[0], 1.0, 1.0, diagonal[1], 1.0, 1.0, diagonal[2]],
        &[0, 1, 0, 1, 2, 1, 2],
        &[0, 2, 5, 7],
    )
    .unwrap()
}

#[test]
fn spd_factorization_solves_and_reconstructs_action() {
    let matrix = tridiagonal_spd([4.0, 3.0, 2.0]);
    let factor = StaticCscCholesky::<3, 5, f64>::decompose(&matrix).unwrap();
    let rhs = vector![1.0; 2.0; 3.0];

    let solution = factor.solve(&rhs);
    assert_relative_eq!(matrix.matvec(&solution), rhs, max_relative = 1e-12);

    let mut output = vector![0.0; 0.0; 0.0];
    factor.solve_into(&rhs, &mut output);
    assert_relative_eq!(output, solution, max_relative = 1e-12);

    // The factor must contain the expected lower-triangular structural
    // entries and no entries above the diagonal.
    let lower = factor.lower();
    for column in 0..3 {
        for row in 0..3 {
            if row < column {
                assert_eq!(lower.get(row, column), None);
            }
        }
    }
    assert!(lower.get(0, 0).is_some());
    assert!(lower.get(1, 0).is_some());
    assert!(lower.get(2, 1).is_some());
}

#[test]
fn symbolic_pattern_can_be_reused_for_new_numeric_values() {
    let mut matrix = tridiagonal_spd([4.0, 3.0, 2.0]);
    let symbolic = StaticCscCholeskyPattern::<3, 5>::analyze(&matrix).unwrap();
    let rhs = vector![2.0; -1.0; 4.0];
    let first = symbolic.factor(&matrix).unwrap().solve(&rhs);

    matrix
        .set_values(&[5.0, 1.0, 1.0, 4.0, 1.0, 1.0, 3.0])
        .unwrap();
    let second_factor = symbolic.factor(&matrix).unwrap();
    let second = second_factor.solve(&rhs);

    assert_relative_eq!(matrix.matvec(&second), rhs, max_relative = 1e-12);
    assert_ne!(first, second);
}

#[test]
fn symbolic_pattern_rejects_new_structural_entries() {
    type Sparse2 = StaticCscMatrix<2, 2, 4, f64>;
    let diagonal = Sparse2::from_pattern(&[2.0, 3.0], &[0, 1], &[0, 1, 2]).unwrap();
    let pattern = stack_algebra::StaticCscCholeskyPattern::<2, 2>::analyze(&diagonal).unwrap();
    let with_off_diagonal =
        Sparse2::from_pattern(&[2.0, 1.0, 1.0, 3.0], &[0, 1, 0, 1], &[0, 2, 4]).unwrap();
    assert_eq!(
        pattern.factor(&with_off_diagonal),
        Err(SparseCholeskyError::PatternMismatch)
    );
}

#[test]
fn singular_and_non_positive_definite_inputs_are_rejected() {
    let singular = StaticCscMatrix::<2, 2, 4, f64>::from_pattern(
        &[1.0, 1.0, 1.0, 1.0],
        &[0, 1, 0, 1],
        &[0, 2, 4],
    )
    .unwrap();
    assert!(StaticCscCholesky::<2, 4, f64>::decompose(&singular).is_err());

    let indefinite = StaticCscMatrix::<2, 2, 4, f64>::from_pattern(
        &[1.0, 2.0, 2.0, 1.0],
        &[0, 1, 0, 1],
        &[0, 2, 4],
    )
    .unwrap();
    assert!(StaticCscCholesky::<2, 4, f64>::decompose(&indefinite).is_err());
}

#[test]
fn insufficient_factor_capacity_is_reported() {
    let matrix = tridiagonal_spd([4.0, 3.0, 2.0]);
    let result = StaticCscCholesky::<3, 2, f64>::decompose(&matrix);
    assert!(matches!(result, Err(SparseCholeskyError::CapacityExceeded)));
}

#[test]
fn factorization_handles_f32() {
    type SparseF32 = StaticCscMatrix<2, 2, 4, f32>;
    let matrix = SparseF32::from_pattern(&[4.0, 1.0, 1.0, 3.0], &[0, 1, 0, 1], &[0, 2, 4]).unwrap();
    let rhs = matrix![1.0_f32; 2.0];
    let factor = StaticCscCholesky::<2, 3, f32>::decompose(&matrix).unwrap();
    assert_relative_eq!(matrix.matvec(&factor.solve(&rhs)), rhs, max_relative = 1e-5);
}

#[test]
fn asymmetric_and_non_finite_inputs_are_rejected() {
    let asymmetric = StaticCscMatrix::<2, 2, 4, f64>::from_pattern(
        &[4.0, 1.0, 2.0, 3.0],
        &[0, 1, 0, 1],
        &[0, 2, 4],
    )
    .unwrap();
    assert!(matches!(
        StaticCscCholesky::<2, 4, f64>::decompose(&asymmetric),
        Err(SparseCholeskyError::NonSymmetric)
    ));

    let non_finite =
        StaticCscMatrix::<1, 1, 1, f64>::from_pattern(&[f64::NAN], &[0], &[0, 1]).unwrap();
    assert!(matches!(
        StaticCscCholesky::<1, 1, f64>::decompose(&non_finite),
        Err(SparseCholeskyError::NonFinite)
    ));
}

#[test]
fn symbolic_fill_in_is_bounded() {
    type Star = StaticCscMatrix<4, 4, 10, f64>;
    let matrix = Star::from_pattern(
        &[4.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0],
        &[0, 1, 2, 3, 0, 1, 0, 2, 0, 3],
        &[0, 4, 6, 8, 10],
    )
    .unwrap();
    assert!(matches!(
        StaticCscCholesky::<4, 9, f64>::decompose(&matrix),
        Err(SparseCholeskyError::CapacityExceeded)
    ));
    let factor = StaticCscCholesky::<4, 10, f64>::decompose(&matrix).unwrap();
    assert_eq!(factor.lower().nnz(), 10);
}

#[test]
fn minimum_degree_ordering_reduces_star_fill_and_solves() {
    type Star = StaticCscMatrix<4, 4, 10, f64>;
    let matrix = Star::from_pattern(
        &[4.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
        &[0, 1, 2, 3, 1, 2, 3],
        &[0, 4, 5, 6, 7],
    )
    .unwrap();
    let ordering = StaticCscOrdering::minimum_degree(&matrix);
    let symbolic =
        StaticCscCholeskyPattern::<4, 10>::analyze_with_ordering(&matrix, ordering).unwrap();
    assert!(symbolic.lower().nnz() < 10);

    let factor = symbolic.factor(&matrix).unwrap();
    let ordered = symbolic.prepare_ordered(&matrix).unwrap();
    let ordered_factor = symbolic.factor_ordered(&ordered).unwrap();
    let mut reusable_factor = StaticCscCholesky::<4, 10, f64>::decompose(&matrix).unwrap();
    symbolic.factor_into(&matrix, &mut reusable_factor).unwrap();
    let mut reusable_ordered_factor = StaticCscCholesky::<4, 10, f64>::decompose(&matrix).unwrap();
    symbolic
        .factor_ordered_into(&ordered, &mut reusable_ordered_factor)
        .unwrap();
    let dense =
        matrix![4.0, 1.0, 1.0, 1.0; 1.0, 4.0, 0.0, 0.0; 1.0, 0.0, 4.0, 0.0; 1.0, 0.0, 0.0, 4.0];
    let rhs = vector![1.0; 2.0; 3.0; 4.0];
    assert_relative_eq!(dense * factor.solve(&rhs), rhs, max_relative = 1e-12);
    assert_relative_eq!(
        dense * ordered_factor.solve(&rhs),
        rhs,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        dense * reusable_factor.solve(&rhs),
        rhs,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        dense * reusable_ordered_factor.solve(&rhs),
        rhs,
        max_relative = 1e-12
    );
}

#[test]
fn invalid_ordering_is_rejected() {
    assert_eq!(
        StaticCscOrdering::<3>::from_permutation(&[0, 0, 2]),
        Err(stack_algebra::CscError::InvalidPermutation)
    );
}

#[test]
fn ordered_lower_semantics_ignore_upper_only_entries() {
    let upper_only =
        StaticCscMatrix::<2, 2, 2, f64>::from_pattern(&[99.0], &[0], &[0, 0, 1]).unwrap();
    let ordered = StaticCscOrdering::identity()
        .permute(&upper_only)
        .expect("identity ordering is valid");

    assert_eq!(ordered.nnz(), 0);
}
