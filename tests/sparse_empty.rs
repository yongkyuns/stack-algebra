use core::mem::MaybeUninit;
use stack_algebra::{
    CscError, Matrix, SparseCholeskyError, StaticCscCholesky, StaticCscCholeskyPattern,
    StaticCscLdlt, StaticCscMatrix, StaticCscOrdering, StaticCscPattern,
};

fn empty<const CAP: usize>() -> StaticCscMatrix<0, 0, CAP, f64> {
    StaticCscMatrix::from_pattern(&[], &[], &[0]).unwrap()
}

fn check_analysis<const A: usize, const L: usize>() {
    let matrix = empty::<A>();
    let ordering = StaticCscOrdering::from_permutation(&[]).unwrap();
    let pattern = StaticCscCholeskyPattern::<0, L>::analyze(&matrix).unwrap();
    assert_eq!(pattern.lower(), &StaticCscPattern::new());
    assert_eq!(pattern.ordering(), ordering);
    assert_eq!(
        StaticCscCholeskyPattern::<0, L>::analyze_with_minimum_degree(&matrix).unwrap(),
        pattern
    );
    assert_eq!(
        StaticCscCholeskyPattern::<0, L>::analyze_with_ordering(&matrix, ordering).unwrap(),
        pattern
    );
    assert_eq!(
        StaticCscCholeskyPattern::<0, L>::analyze_with_diagonal_pivoting(&matrix, 0.0).unwrap(),
        pattern
    );
}

#[test]
fn empty_analysis_supports_destination_scratch() {
    check_analysis::<0, 0>();
    check_analysis::<0, 4>();
    check_analysis::<3, 5>();
}

#[test]
fn empty_analysis_supports_local_scratch() {
    check_analysis::<4, 0>();
    check_analysis::<5, 3>();
}

fn check_analysis_into<const A: usize, const L: usize>() {
    let matrix = empty::<A>();
    let ordering = StaticCscOrdering::identity();
    let mut output = MaybeUninit::<StaticCscCholeskyPattern<0, L>>::uninit();
    StaticCscCholeskyPattern::analyze_with_ordering_into(&matrix, ordering, &mut output).unwrap();
    // SAFETY: successful analysis initializes the entire output, including unused capacity.
    let pattern = unsafe { output.assume_init() };
    let mut ordered_output = MaybeUninit::<StaticCscCholeskyPattern<0, L>>::uninit();
    StaticCscCholeskyPattern::analyze_ordered_with_ordering_into(
        &matrix,
        ordering,
        &mut ordered_output,
    )
    .unwrap();
    // SAFETY: successful ordered analysis initializes every output field.
    let ordered_pattern = unsafe { ordered_output.assume_init() };
    assert_eq!(pattern, ordered_pattern);
    assert_eq!(pattern, StaticCscCholeskyPattern::analyze(&matrix).unwrap());
}

#[test]
fn empty_analysis_initializes_caller_owned_storage() {
    check_analysis_into::<0, 0>();
    check_analysis_into::<0, 4>();
    check_analysis_into::<4, 0>();
    check_analysis_into::<5, 3>();
}

#[test]
fn empty_cholesky_factors_recompute_and_solve() {
    let matrix = empty::<0>();
    let pattern = StaticCscCholeskyPattern::<0, 0>::analyze(&matrix).unwrap();
    let mut factor = StaticCscCholesky::<0, 0, f64>::decompose(&matrix).unwrap();
    assert_eq!(
        factor,
        StaticCscCholesky::decompose_with_minimum_degree(&matrix).unwrap()
    );
    assert_eq!(factor, pattern.factor(&matrix).unwrap());
    assert_eq!(factor, pattern.factor_ordered(&matrix).unwrap());
    factor.recompute(&empty::<4>()).unwrap();
    factor.recompute_with_pattern(&pattern, &matrix).unwrap();
    factor.recompute_ordered(&matrix).unwrap();
    factor
        .recompute_ordered_with_pattern(&pattern, &matrix)
        .unwrap();
    assert_eq!(factor.lower().nnz(), 0);
    assert_eq!(factor.pattern(), pattern);

    let rhs = Matrix::<0, 3, f64>::zeros();
    let mut output = rhs;
    assert_eq!(factor.solve(&rhs), rhs);
    factor.solve_into(&rhs, &mut output);
    factor.solve_in_place(&mut output);
    assert_eq!(output, rhs);
    let empty_rhs = Matrix::<0, 0, f64>::zeros();
    assert!(factor.solve(&empty_rhs).as_slice().is_empty());
}

#[test]
fn empty_ldlt_factors_recompute_and_solve() {
    let matrix = empty::<4>();
    let pattern = StaticCscCholeskyPattern::<0, 0>::analyze(&matrix).unwrap();
    let mut factor = StaticCscLdlt::<0, 0, f64>::decompose(&matrix).unwrap();
    assert_eq!(factor, pattern.factor_ldlt(&matrix).unwrap());
    assert_eq!(factor, pattern.factor_ldlt_ordered(&matrix).unwrap());
    assert_eq!(
        factor,
        StaticCscLdlt::decompose_with_minimum_degree(&matrix).unwrap()
    );
    assert_eq!(
        factor,
        StaticCscLdlt::decompose_with_diagonal_pivoting(&matrix, 0.0).unwrap()
    );
    factor.recompute(&empty::<0>()).unwrap();
    factor.recompute_with_pattern(&pattern, &matrix).unwrap();
    factor.recompute_ordered(&matrix).unwrap();
    factor
        .recompute_ordered_with_pattern(&pattern, &matrix)
        .unwrap();
    assert_eq!(factor.lower().nnz(), 0);
    assert!(factor.diagonal().is_empty());
    assert_eq!(factor.pattern(), pattern);

    let rhs = Matrix::<0, 3, f64>::zeros();
    let mut output = rhs;
    let mut workspace = rhs;
    assert_eq!(factor.solve(&rhs), rhs);
    factor.solve_into(&rhs, &mut output);
    factor.solve_in_place(&mut output);
    factor.solve_in_place_with_workspace(&mut output, &mut workspace);
    assert_eq!(output, rhs);
    let empty_rhs = Matrix::<0, 0, f64>::zeros();
    assert!(factor.solve(&empty_rhs).as_slice().is_empty());
}

#[test]
fn empty_ordered_preparation_and_numeric_output_are_valid() {
    let matrix = empty::<4>();
    let pattern = StaticCscCholeskyPattern::<0, 3>::analyze(&matrix).unwrap();
    let ordered = pattern.prepare_ordered(&matrix).unwrap();
    let permutation = pattern
        .ordering()
        .permutation_for_pattern(matrix.pattern())
        .unwrap();
    assert_eq!(
        pattern
            .prepare_ordered_with_permutation(&permutation, &matrix)
            .unwrap(),
        ordered
    );
    let mut output = MaybeUninit::<StaticCscLdlt<0, 3, f64>>::uninit();
    pattern
        .factor_ldlt_ordered_into(&ordered, &mut output)
        .unwrap();
    // SAFETY: successful factorization initializes all fields and unused capacity.
    let factor = unsafe { output.assume_init() };
    assert_eq!(factor, pattern.factor_ldlt_ordered(&ordered).unwrap());
    assert_eq!(factor.lower().values(), &[]);
}

#[test]
fn empty_f32_factorizations_and_sparse_fallback_wrapper_work() {
    let matrix = StaticCscMatrix::<0, 0, 2, f32>::from_pattern(&[], &[], &[0]).unwrap();
    let rhs = Matrix::<0, 2, f32>::zeros();
    let cholesky = StaticCscCholesky::<0, 0, f32>::decompose(&matrix).unwrap();
    let ldlt = StaticCscLdlt::<0, 0, f32>::decompose(&matrix).unwrap();
    assert_eq!(cholesky.solve(&rhs), rhs);
    assert_eq!(ldlt.solve(&rhs), rhs);
    let mut unified = matrix.try_ldlt_with_dense_fallback::<0>().unwrap();
    assert!(!unified.uses_dense_fallback());
    unified.recompute_with_dense_fallback(&matrix).unwrap();
    assert_eq!(unified.solve(&rhs), rhs);
}

#[test]
fn rectangular_empty_storage_and_matvec_remain_valid() {
    let pattern = StaticCscPattern::<3, 0, 4>::from_arrays(&[], &[0]).unwrap();
    let matrix = StaticCscMatrix::<3, 0, 4, f64>::with_pattern(pattern, &[]).unwrap();
    assert_eq!(matrix.matvec(&Matrix::zeros()), Matrix::<3, 1, f64>::zeros());
    assert_eq!(matrix.column_end(0), None);
    assert_eq!(matrix.get(0, 0), None);

    let mut output = MaybeUninit::<StaticCscMatrix<3, 0, 4, f64>>::uninit();
    StaticCscMatrix::zero_with_pattern_into(pattern, &mut output);
    // SAFETY: zero_with_pattern_into initializes the complete matrix before returning.
    assert_eq!(unsafe { output.assume_init() }, matrix);
    let zero_rows = StaticCscMatrix::<0, 3, 4, f64>::from_pattern(&[], &[], &[0, 0, 0, 0]).unwrap();
    assert!(zero_rows.matvec(&Matrix::ones()).as_slice().is_empty());
}

#[test]
fn zero_columns_reject_hidden_entries_and_invalid_pointers() {
    type Pattern = StaticCscPattern<3, 0, 4>;
    assert_eq!(Pattern::from_arrays(&[], &[]), Err(CscError::LengthMismatch));
    assert_eq!(
        Pattern::from_arrays(&[], &[1]),
        Err(CscError::InvalidColumnPointers)
    );
    assert_eq!(
        Pattern::from_arrays(&[0], &[0]),
        Err(CscError::InvalidColumnPointers)
    );
    assert_eq!(
        Pattern::from_arrays(&[0], &[1]),
        Err(CscError::InvalidColumnPointers)
    );
    assert_eq!(
        StaticCscMatrix::<3, 0, 4, f64>::from_pattern(&[1.0], &[0], &[0]),
        Err(CscError::InvalidColumnPointers)
    );
}

#[test]
fn nonempty_missing_pivots_are_not_treated_as_empty_systems() {
    let matrix = StaticCscMatrix::<1, 1, 0, f64>::from_pattern(&[], &[], &[0, 0]).unwrap();
    assert_eq!(
        StaticCscCholesky::<1, 1, f64>::decompose(&matrix),
        Err(SparseCholeskyError::NotPositiveDefinite)
    );
    assert_eq!(
        StaticCscLdlt::<1, 1, f64>::decompose(&matrix),
        Err(SparseCholeskyError::ZeroPivot)
    );
}
