use core::mem::MaybeUninit;
use stack_algebra::{
    Matrix, SparseCholeskyError, StaticCscLdlt, StaticCscLdltPattern, StaticCscMatrix,
    StaticCscOrdering,
};

type Pattern2 = StaticCscLdltPattern<2, 3>;
type Factor2 = StaticCscLdlt<2, 3, f64>;

fn lower2() -> StaticCscMatrix<2, 2, 3, f64> {
    StaticCscMatrix::from_pattern(&[4.0, 1.0, 3.0], &[0, 1, 1], &[0, 2, 3]).unwrap()
}

fn diagonal2() -> StaticCscMatrix<2, 2, 2, f64> {
    StaticCscMatrix::from_pattern(&[5.0, 6.0], &[0, 1], &[0, 1, 2]).unwrap()
}

fn assert_solves<const N: usize, const CAP: usize>(
    factor: &StaticCscLdlt<N, CAP, f64>,
    dense: Matrix<N, N, f64>,
) {
    let rhs = Matrix::<N, 2, f64>::from_fn(|row, column| (row + 2 * column + 1) as f64);
    let residual = dense * factor.solve(&rhs) - rhs;
    assert!(residual.norm() < 1.0e-11, "residual: {residual:?}");
}

#[test]
fn cached_ldlt_handles_smaller_input_capacity() {
    let pattern = Pattern2::analyze(&lower2()).unwrap();
    let factor = pattern.factor_ldlt(&diagonal2()).unwrap();
    assert_solves(&factor, Matrix::from_rows([[5.0, 0.0], [0.0, 6.0]]));
}

#[test]
fn cached_ldlt_handles_upper_triangle_layout_changes() {
    let lower = lower2();
    let full = StaticCscMatrix::<2, 2, 4, f64>::from_pattern(
        &[4.0, 1.0, 1.0, 3.0],
        &[0, 1, 0, 1],
        &[0, 2, 4],
    )
    .unwrap();
    let dense = Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]);
    let lower_pattern = Pattern2::analyze(&lower).unwrap();
    let full_pattern = Pattern2::analyze(&full).unwrap();
    assert_solves(&lower_pattern.factor_ldlt(&full).unwrap(), dense);
    assert_solves(&full_pattern.factor_ldlt(&lower).unwrap(), dense);
}

#[test]
fn cached_ldlt_handles_equal_nnz_with_different_coordinates() {
    let star = StaticCscMatrix::<3, 3, 5, f64>::from_pattern(
        &[8.0, 1.0, 2.0, 9.0, 10.0],
        &[0, 1, 2, 1, 2],
        &[0, 3, 4, 5],
    )
    .unwrap();
    let chain = StaticCscMatrix::<3, 3, 5, f64>::from_pattern(
        &[8.0, 1.0, 9.0, 2.0, 10.0],
        &[0, 1, 1, 2, 2],
        &[0, 2, 4, 5],
    )
    .unwrap();
    let pattern = StaticCscLdltPattern::<3, 6>::analyze(&star).unwrap();
    assert_eq!(star.nnz(), chain.nnz());
    let factor = pattern.factor_ldlt(&chain).unwrap();
    assert_solves(
        &factor,
        Matrix::from_rows([[8.0, 1.0, 0.0], [1.0, 9.0, 2.0], [0.0, 2.0, 10.0]]),
    );
}

#[test]
fn ordered_ldlt_handles_changed_input_layout_and_uninitialized_output() {
    let pattern = Pattern2::analyze(&lower2()).unwrap();
    let diagonal = diagonal2();
    let dense = Matrix::from_rows([[5.0, 0.0], [0.0, 6.0]]);
    assert_solves(&pattern.factor_ldlt_ordered(&diagonal).unwrap(), dense);
    let mut output = MaybeUninit::<Factor2>::uninit();
    pattern
        .factor_ldlt_ordered_into(&diagonal, &mut output)
        .unwrap();
    // SAFETY: the factorization returned Ok and initialized the entire output.
    assert_solves(&unsafe { output.assume_init() }, dense);
}

#[test]
fn cached_ldlt_recompute_handles_changed_input_layout() {
    let original = lower2();
    let pattern = Pattern2::analyze(&original).unwrap();
    let diagonal = diagonal2();
    let dense = Matrix::from_rows([[5.0, 0.0], [0.0, 6.0]]);
    let mut factor = pattern.factor_ldlt(&original).unwrap();
    factor.recompute_with_pattern(&pattern, &diagonal).unwrap();
    assert_solves(&factor, dense);
    let mut factor = pattern.factor_ldlt(&original).unwrap();
    factor
        .recompute_ordered_with_pattern(&pattern, &diagonal)
        .unwrap();
    assert_solves(&factor, dense);
    factor.recompute(&diagonal).unwrap();
    assert_solves(&factor, dense);
}

#[test]
fn ordered_ldlt_recompute_updates_destination_pattern() {
    let matrix = lower2();
    let pattern = Pattern2::analyze(&matrix).unwrap();
    let mut factor = Factor2::decompose(&diagonal2()).unwrap();
    assert_ne!(factor.lower().pattern(), pattern.lower());
    factor
        .recompute_ordered_with_pattern(&pattern, &matrix)
        .unwrap();
    assert_eq!(factor.lower().pattern(), pattern.lower());
    assert_solves(&factor, Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]));
}

#[test]
fn uncovered_structure_is_rejected_before_reusable_output_changes() {
    let diagonal = diagonal2();
    let pattern = Pattern2::analyze(&diagonal).unwrap();
    let matrix = lower2();
    assert_eq!(
        pattern.factor_ldlt(&matrix),
        Err(SparseCholeskyError::PatternMismatch)
    );
    assert_eq!(
        pattern.factor_ldlt_ordered(&matrix),
        Err(SparseCholeskyError::PatternMismatch)
    );
    let mut factor = pattern.factor_ldlt(&diagonal).unwrap();
    let before = factor;
    assert_eq!(
        factor.recompute_with_pattern(&pattern, &matrix),
        Err(SparseCholeskyError::PatternMismatch)
    );
    assert_eq!(factor, before);
    assert_eq!(
        factor.recompute_ordered_with_pattern(&pattern, &matrix),
        Err(SparseCholeskyError::PatternMismatch)
    );
    assert_eq!(factor, before);
}

#[test]
fn cached_ldlt_preserves_matching_numeric_updates() {
    let original = lower2();
    let pattern = Pattern2::analyze(&original).unwrap();
    let next =
        StaticCscMatrix::<2, 2, 5, f64>::from_pattern(&[5.0, 2.0, 6.0], &[0, 1, 1], &[0, 2, 3])
            .unwrap();
    let dense = Matrix::from_rows([[5.0, 2.0], [2.0, 6.0]]);
    assert_solves(&pattern.factor_ldlt(&next).unwrap(), dense);
    let mut factor = pattern.factor_ldlt(&original).unwrap();
    factor.recompute_with_pattern(&pattern, &next).unwrap();
    assert_solves(&factor, dense);
}

#[test]
fn cached_ldlt_handles_empty_replacement_without_reading_stale_indices() {
    let pattern = Pattern2::analyze(&lower2()).unwrap();
    let empty = StaticCscMatrix::<2, 2, 0, f64>::from_pattern(&[], &[], &[0, 0, 0]).unwrap();
    assert_eq!(
        pattern.factor_ldlt(&empty),
        Err(SparseCholeskyError::ZeroPivot)
    );
    assert_eq!(
        pattern.factor_ldlt_ordered(&empty),
        Err(SparseCholeskyError::ZeroPivot)
    );
}

#[test]
fn reordered_ldlt_handles_changed_source_layout() {
    let original = lower2();
    let ordering = StaticCscOrdering::from_permutation(&[1, 0]).unwrap();
    let pattern = Pattern2::analyze_with_ordering(&original, ordering).unwrap();
    let diagonal = diagonal2();
    let ordered = pattern.prepare_ordered(&diagonal).unwrap();
    let dense = Matrix::from_rows([[5.0, 0.0], [0.0, 6.0]]);
    assert_solves(&pattern.factor_ldlt(&diagonal).unwrap(), dense);
    assert_solves(&pattern.factor_ldlt_ordered(&ordered).unwrap(), dense);
    let mut factor = pattern.factor_ldlt(&original).unwrap();
    factor.recompute_with_pattern(&pattern, &diagonal).unwrap();
    assert_eq!(factor.ordering(), ordering);
    assert_solves(&factor, dense);
    factor
        .recompute_ordered_with_pattern(&pattern, &ordered)
        .unwrap();
    assert_eq!(factor.ordering(), ordering);
    assert_solves(&factor, dense);
}

#[test]
fn cached_ldlt_handles_changed_source_layout_for_f32() {
    let original =
        StaticCscMatrix::<2, 2, 3, f32>::from_pattern(&[4.0, 1.0, 3.0], &[0, 1, 1], &[0, 2, 3])
            .unwrap();
    let diagonal =
        StaticCscMatrix::<2, 2, 2, f32>::from_pattern(&[5.0, 6.0], &[0, 1], &[0, 1, 2]).unwrap();
    let pattern = Pattern2::analyze(&original).unwrap();
    let factor = pattern.factor_ldlt(&diagonal).unwrap();
    let rhs = Matrix::<2, 2, f32>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    let dense = Matrix::from_rows([[5.0, 0.0], [0.0, 6.0]]);
    assert!((dense * factor.solve(&rhs) - rhs).norm() < 1.0e-5);
}

mod cholesky_reuse {
    use super::{diagonal2, lower2, Matrix, Pattern2, SparseCholeskyError, StaticCscMatrix};
    use stack_algebra::{StaticCscCholesky, StaticCscCholeskyPattern, StaticCscOrdering};

    type Factor = StaticCscCholesky<2, 3, f64>;

    fn assert_solves<const N: usize, const CAP: usize>(
        factor: &StaticCscCholesky<N, CAP, f64>,
        dense: Matrix<N, N, f64>,
    ) {
        let rhs = Matrix::<N, 2, f64>::from_fn(|row, column| (row + 2 * column + 1) as f64);
        let residual = dense * factor.solve(&rhs) - rhs;
        assert!(residual.norm() < 1.0e-11, "residual: {residual:?}");
    }

    #[test]
    fn ordered_factor_rejects_uncovered_structure() {
        let pattern = Pattern2::analyze(&diagonal2()).unwrap();
        assert_eq!(
            pattern.factor_ordered(&lower2()),
            Err(SparseCholeskyError::PatternMismatch)
        );
    }

    #[test]
    fn all_recompute_entry_points_reject_structure_before_mutation() {
        let diagonal = diagonal2();
        let pattern = Pattern2::analyze(&diagonal).unwrap();
        let input = lower2();
        let original = pattern.factor(&diagonal).unwrap();
        for mode in 0..4 {
            let mut factor = original;
            let result = match mode {
                0 => factor.recompute(&input),
                1 => factor.recompute_with_pattern(&pattern, &input),
                2 => factor.recompute_ordered(&input),
                _ => factor.recompute_ordered_with_pattern(&pattern, &input),
            };
            assert_eq!(result, Err(SparseCholeskyError::PatternMismatch));
            assert_eq!(factor, original);
        }
    }

    #[test]
    fn ordered_recompute_installs_larger_factor_pattern() {
        let input = lower2();
        let pattern = Pattern2::analyze(&input).unwrap();
        let mut factor = Factor::decompose(&diagonal2()).unwrap();
        factor
            .recompute_ordered_with_pattern(&pattern, &input)
            .unwrap();
        assert_eq!(factor.lower().pattern(), pattern.lower());
        assert_solves(&factor, Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]));
    }

    #[test]
    fn ordered_recompute_installs_smaller_factor_pattern() {
        let input = diagonal2();
        let pattern = Pattern2::analyze(&input).unwrap();
        let mut factor = Factor::decompose(&lower2()).unwrap();
        factor
            .recompute_ordered_with_pattern(&pattern, &input)
            .unwrap();
        assert_eq!(factor.lower().pattern(), pattern.lower());
        assert_solves(&factor, Matrix::from_rows([[5.0, 0.0], [0.0, 6.0]]));
    }

    #[test]
    fn structural_rejection_preserves_a_different_destination_pattern() {
        let input = lower2();
        let pattern = Pattern2::analyze(&diagonal2()).unwrap();
        let original = Factor::decompose(&input).unwrap();
        for ordered in [false, true] {
            let mut factor = original;
            let result = if ordered {
                factor.recompute_ordered_with_pattern(&pattern, &input)
            } else {
                factor.recompute_with_pattern(&pattern, &input)
            };
            assert_eq!(result, Err(SparseCholeskyError::PatternMismatch));
            assert_eq!(factor, original);
        }
    }

    #[test]
    fn reordered_calls_validate_structure_and_synchronize_output() {
        let ordering = StaticCscOrdering::from_permutation(&[1, 0]).unwrap();
        let diagonal = diagonal2();
        let pattern = Pattern2::analyze_with_ordering(&diagonal, ordering).unwrap();
        let original = pattern.factor(&diagonal).unwrap();
        let mut factor = original;
        assert_eq!(
            factor.recompute_with_pattern(&pattern, &lower2()),
            Err(SparseCholeskyError::PatternMismatch)
        );
        assert_eq!(factor, original);

        let input = lower2();
        let pattern = Pattern2::analyze_with_ordering(&input, ordering).unwrap();
        let ordered = pattern.prepare_ordered(&input).unwrap();
        factor
            .recompute_ordered_with_pattern(&pattern, &ordered)
            .unwrap();
        assert_eq!(factor.pattern().ordering(), ordering);
        assert_eq!(factor.lower().pattern(), pattern.lower());
        assert_solves(&factor, Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]));
    }

    #[test]
    fn compatible_full_lower_and_smaller_layouts_still_solve() {
        let full = StaticCscMatrix::<2, 2, 4, f64>::from_pattern(
            &[4.0, 1.0, 1.0, 3.0],
            &[0, 1, 0, 1],
            &[0, 2, 4],
        )
        .unwrap();
        let pattern = Pattern2::analyze(&full).unwrap();
        let mut factor = pattern.factor_ordered(&lower2()).unwrap();
        assert_solves(&factor, Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]));
        factor
            .recompute_with_pattern(&pattern, &diagonal2())
            .unwrap();
        assert_solves(&factor, Matrix::from_rows([[5.0, 0.0], [0.0, 6.0]]));
        factor
            .recompute_ordered_with_pattern(&pattern, &full)
            .unwrap();
        assert_solves(&factor, Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]));
    }

    #[test]
    fn factor_fill_covers_changed_input_coordinates() {
        let star = StaticCscMatrix::<3, 3, 5, f64>::from_pattern(
            &[8.0, 1.0, 2.0, 9.0, 10.0],
            &[0, 1, 2, 1, 2],
            &[0, 3, 4, 5],
        )
        .unwrap();
        let chain = StaticCscMatrix::<3, 3, 5, f64>::from_pattern(
            &[8.0, 1.0, 9.0, 2.0, 10.0],
            &[0, 1, 1, 2, 2],
            &[0, 2, 4, 5],
        )
        .unwrap();
        let pattern = StaticCscCholeskyPattern::<3, 6>::analyze(&star).unwrap();
        let dense = Matrix::from_rows([[8.0, 1.0, 0.0], [1.0, 9.0, 2.0], [0.0, 2.0, 10.0]]);
        let mut factor = pattern.factor_ordered(&chain).unwrap();
        assert_solves(&factor, dense);
        factor.recompute_with_pattern(&pattern, &chain).unwrap();
        assert_solves(&factor, dense);
    }

    #[test]
    fn f32_reuse_validates_structure_and_updates_destination() {
        let diagonal =
            StaticCscMatrix::<2, 2, 2, f32>::from_pattern(&[5.0, 6.0], &[0, 1], &[0, 1, 2])
                .unwrap();
        let input =
            StaticCscMatrix::<2, 2, 3, f32>::from_pattern(&[4.0, 1.0, 3.0], &[0, 1, 1], &[0, 2, 3])
                .unwrap();
        let diagonal_pattern = Pattern2::analyze(&diagonal).unwrap();
        assert_eq!(
            diagonal_pattern.factor_ordered(&input),
            Err(SparseCholeskyError::PatternMismatch)
        );
        let pattern = Pattern2::analyze(&input).unwrap();
        let mut factor = diagonal_pattern.factor(&diagonal).unwrap();
        factor
            .recompute_ordered_with_pattern(&pattern, &input)
            .unwrap();
        assert_eq!(factor.lower().pattern(), pattern.lower());
        let rhs = Matrix::<2, 2, f32>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
        let dense = Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]);
        assert!((dense * factor.solve(&rhs) - rhs).norm() < 1.0e-5);
    }

    #[test]
    fn missing_pivots_keep_existing_error() {
        let pattern = Pattern2::analyze(&lower2()).unwrap();
        let missing = StaticCscMatrix::<2, 2, 0, f64>::from_pattern(&[], &[], &[0, 0, 0]).unwrap();
        assert_eq!(
            pattern.factor_ordered(&missing),
            Err(SparseCholeskyError::NotPositiveDefinite)
        );
    }
}
