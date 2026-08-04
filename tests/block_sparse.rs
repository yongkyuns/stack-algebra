use approx::assert_relative_eq;
use stack_algebra::{
    matrix, Matrix, SparseCholeskyError, StaticBlockCscCholesky, StaticBlockCscLdlt,
    StaticBlockCscMatrix, StaticBlockCsrMatrix, StaticCscCholesky, StaticCscLdlt,
    StaticCscOrdering,
};

type Blocks = StaticBlockCscMatrix<2, 2, 2, 2, 3, f64>;
type CsrBlocks = StaticBlockCsrMatrix<2, 2, 2, 2, 3, f64>;

#[test]
fn block_csc_reports_scalar_dimensions_and_blocks() {
    let values = [
        Matrix::from_rows([[1.0, 0.0], [0.0, 1.0]]),
        Matrix::from_rows([[2.0, 0.0], [0.0, 2.0]]),
    ];
    let matrix = Blocks::from_pattern(&values, &[0, 1], &[0, 1, 2]).unwrap();
    assert_eq!(matrix.block_rows(), 2);
    assert_eq!(matrix.block_cols(), 2);
    assert_eq!(matrix.rows(), 4);
    assert_eq!(matrix.cols(), 4);
    assert_eq!(matrix.block(0, 0), Some(&values[0]));
    assert_eq!(matrix.block(1, 1), Some(&values[1]));
    assert!(matrix.block(1, 0).is_none());
    assert!(matrix.block(0, 1).is_none());
}

#[test]
fn block_csc_matvec_matches_expanded_matrix() {
    let values = [
        Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]),
        Matrix::from_rows([[5.0, 6.0], [7.0, 8.0]]),
    ];
    let matrix = Blocks::from_pattern(&values, &[0, 1], &[0, 1, 2]).unwrap();
    let rhs = [1.0, 2.0, 3.0, 4.0];
    let mut actual = [0.0; 4];
    matrix.matvec_into(&rhs, &mut actual).unwrap();

    let expanded = matrix![
        1.0, 2.0, 0.0, 0.0;
        3.0, 4.0, 0.0, 0.0;
        0.0, 0.0, 5.0, 6.0;
        0.0, 0.0, 7.0, 8.0;
    ];
    let expected = expanded * Matrix::<4, 1, f64>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    let actual = Matrix::<4, 1, f64>::from_columns([actual]);
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn block_csr_reports_scalar_dimensions_and_blocks() {
    let values = [
        Matrix::from_rows([[1.0, 0.0], [0.0, 1.0]]),
        Matrix::from_rows([[2.0, 0.0], [0.0, 2.0]]),
    ];
    let matrix = CsrBlocks::from_pattern(&values, &[0, 1], &[0, 1, 2]).unwrap();
    assert_eq!(matrix.block_rows(), 2);
    assert_eq!(matrix.block_cols(), 2);
    assert_eq!(matrix.rows(), 4);
    assert_eq!(matrix.cols(), 4);
    assert_eq!(matrix.block(0, 0), Some(&values[0]));
    assert_eq!(matrix.block(1, 1), Some(&values[1]));
    assert!(matrix.block(1, 0).is_none());
    assert!(matrix.block(0, 1).is_none());
}

#[test]
fn block_csr_matvec_matches_expanded_matrix() {
    let values = [
        Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]),
        Matrix::from_rows([[5.0, 6.0], [7.0, 8.0]]),
        Matrix::from_rows([[2.0, 0.0], [0.0, 2.0]]),
    ];
    let matrix = CsrBlocks::from_pattern(&values, &[0, 1, 0], &[0, 2, 3]).unwrap();
    let rhs = [1.0, 2.0, 3.0, 4.0];
    let mut actual = [0.0; 4];
    matrix.matvec_into(&rhs, &mut actual).unwrap();

    let expanded = matrix![
        1.0, 2.0, 5.0, 6.0;
        3.0, 4.0, 7.0, 8.0;
        2.0, 0.0, 0.0, 0.0;
        0.0, 2.0, 0.0, 0.0;
    ];
    let expected = expanded * Matrix::<4, 1, f64>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    let actual = Matrix::<4, 1, f64>::from_columns([actual]);
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn block_csr_rejects_noncanonical_patterns() {
    let values = [Matrix::from_rows([[1.0_f64, 0.0], [0.0, 1.0]])];
    assert!(CsrBlocks::from_pattern(&values, &[2], &[0, 1, 1]).is_err());
    assert!(CsrBlocks::from_pattern(&values, &[0], &[0, 1]).is_err());
}

#[test]
fn block_sparse_storage_footprints_are_compile_time_constants() {
    const CSC_BYTES: usize = Blocks::storage_bytes();
    const CSR_BYTES: usize = CsrBlocks::storage_bytes();
    assert_eq!(CSC_BYTES, core::mem::size_of::<Blocks>());
    assert_eq!(CSR_BYTES, core::mem::size_of::<CsrBlocks>());
    assert!(CSC_BYTES >= 3 * core::mem::size_of::<Matrix<2, 2, f64>>());
    assert!(CSR_BYTES >= 3 * core::mem::size_of::<Matrix<2, 2, f64>>());
}

#[test]
fn block_csc_expands_and_factors_without_heap_storage() {
    type BlockMatrix = StaticBlockCscMatrix<2, 2, 2, 2, 4, f64>;
    let values = [
        Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]),
        Matrix::from_rows([[1.0, 0.0], [0.0, 0.5]]),
        Matrix::from_rows([[1.0, 0.0], [0.0, 0.5]]),
        Matrix::from_rows([[3.0, 0.0], [0.0, 2.0]]),
    ];
    let block = BlockMatrix::from_pattern(&values, &[0, 1, 0, 1], &[0, 2, 4]).unwrap();
    let scalar = block.to_scalar_csc::<4, 4, 16>().unwrap();
    assert_eq!(scalar.nnz(), 16);
    assert!(block.to_scalar_csc::<4, 4, 15>().is_err());
    assert!(block.to_scalar_csc::<3, 4, 16>().is_err());

    let dense: Matrix<4, 4, f64> = Matrix::from_fn(|row, column| {
        scalar
            .get(row, column)
            .or_else(|| scalar.get(column, row))
            .copied()
            .unwrap_or(0.0)
    });
    let factor = block.cholesky::<4, 16, 16>().unwrap();
    let native: StaticBlockCscCholesky<2, 2, 2, 2, 4, f64> =
        StaticBlockCscCholesky::decompose(&block).unwrap();
    let rhs = Matrix::<4, 1, f64>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    let solution = factor.solve(&rhs);
    let native_solution = native.try_solve::<4, 1>(&rhs).unwrap();
    assert_relative_eq!(dense * solution, rhs, epsilon = 1e-12, max_relative = 1e-12);
    assert_relative_eq!(
        native_solution,
        solution,
        epsilon = 1e-12,
        max_relative = 1e-12
    );

    let mut reused = native;
    reused.recompute(&block).unwrap();
    assert_relative_eq!(
        reused.try_solve::<4, 1>(&rhs).unwrap(),
        solution,
        epsilon = 1e-12
    );
}

#[test]
fn native_block_cholesky_handles_block_fill_in() {
    type Star = StaticBlockCscMatrix<1, 1, 3, 3, 5, f64>;
    type Factor = StaticBlockCscCholesky<1, 1, 3, 3, 6, f64>;
    let values = [
        Matrix::from_rows([[4.0]]),
        Matrix::from_rows([[1.0]]),
        Matrix::from_rows([[1.0]]),
        Matrix::from_rows([[3.0]]),
        Matrix::from_rows([[3.0]]),
    ];
    let matrix = Star::from_pattern(&values, &[0, 1, 2, 1, 2], &[0, 3, 4, 5]).unwrap();
    let native: Factor = Factor::decompose(&matrix).unwrap();
    assert_eq!(native.lower().nnz(), 6);

    let scalar = matrix.to_scalar_csc::<3, 3, 5>().unwrap();
    let scalar_factor = matrix.cholesky::<3, 5, 6>().unwrap();
    let rhs = Matrix::<3, 1, f64>::from_columns([[1.0, 2.0, 3.0]]);
    assert_relative_eq!(
        native.try_solve::<3, 1>(&rhs).unwrap(),
        scalar_factor.solve(&rhs),
        epsilon = 1e-12,
        max_relative = 1e-12
    );
    let dense: Matrix<3, 3, f64> = Matrix::from_fn(|row, column| {
        scalar
            .get(row, column)
            .or_else(|| scalar.get(column, row))
            .copied()
            .unwrap_or(0.0)
    });
    assert_relative_eq!(
        dense * native.try_solve::<3, 1>(&rhs).unwrap(),
        rhs,
        epsilon = 1e-12,
        max_relative = 1e-12
    );
}

#[test]
fn native_block_ldlt_matches_scalar_expansion() {
    type BlockMatrix = StaticBlockCscMatrix<2, 2, 2, 2, 4, f64>;
    type Factor = StaticBlockCscLdlt<2, 2, 2, 2, 4, f64>;
    let values = [
        Matrix::from_rows([[4.0, 1.0], [1.0, -3.0]]),
        Matrix::from_rows([[1.0, 0.2], [0.5, 0.3]]),
        Matrix::from_rows([[1.0, 0.5], [0.2, 0.3]]),
        Matrix::from_rows([[3.0, 0.4], [0.4, 2.0]]),
    ];
    let block = BlockMatrix::from_pattern(&values, &[0, 1, 0, 1], &[0, 2, 4]).unwrap();
    let scalar = block.to_scalar_csc::<4, 4, 16>().unwrap();
    let scalar_factor = StaticCscLdlt::<4, 16, f64>::decompose(&scalar).unwrap();
    let native: Factor = Factor::decompose(&block).unwrap();
    let rhs = Matrix::<4, 2, f64>::from_columns([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 3.0]]);
    let native_solution = native.try_solve::<4, 2>(&rhs).unwrap();

    assert_relative_eq!(
        native_solution,
        scalar_factor.solve(&rhs),
        epsilon = 1e-12,
        max_relative = 1e-12
    );
    let mut reused = native;
    reused.recompute(&block).unwrap();
    assert_relative_eq!(
        reused.try_solve::<4, 2>(&rhs).unwrap(),
        scalar_factor.solve(&rhs),
        epsilon = 1e-12,
        max_relative = 1e-12
    );
}

#[test]
fn native_block_cholesky_ordering_preserves_solution() {
    type Star = StaticBlockCscMatrix<1, 1, 4, 4, 10, f64>;
    let values = [
        Matrix::from_rows([[4.0]]),
        Matrix::from_rows([[1.0]]),
        Matrix::from_rows([[1.0]]),
        Matrix::from_rows([[1.0]]),
        Matrix::from_rows([[4.0]]),
        Matrix::from_rows([[4.0]]),
        Matrix::from_rows([[4.0]]),
    ];
    let matrix = Star::from_pattern(&values, &[0, 1, 2, 3, 1, 2, 3], &[0, 4, 5, 6, 7]).unwrap();
    let scalar = matrix.to_scalar_csc::<4, 4, 10>().unwrap();
    let ordering = StaticCscOrdering::minimum_degree(&scalar);
    let native =
        StaticBlockCscCholesky::<1, 1, 4, 4, 10, f64>::decompose_with_ordering(&matrix, ordering)
            .unwrap();
    let scalar_factor = StaticCscCholesky::<4, 10, f64>::decompose(&scalar).unwrap();
    let rhs = Matrix::<4, 1, f64>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    assert_relative_eq!(
        native.try_solve::<4, 1>(&rhs).unwrap(),
        scalar_factor.solve(&rhs),
        epsilon = 1e-12,
        max_relative = 1e-12
    );
}

#[test]
fn native_block_ldlt_ordering_preserves_solution() {
    type Star = StaticBlockCscMatrix<1, 1, 4, 4, 10, f64>;
    let values = [
        Matrix::from_rows([[4.0]]),
        Matrix::from_rows([[1.0]]),
        Matrix::from_rows([[1.0]]),
        Matrix::from_rows([[1.0]]),
        Matrix::from_rows([[4.0]]),
        Matrix::from_rows([[4.0]]),
        Matrix::from_rows([[4.0]]),
    ];
    let matrix = Star::from_pattern(&values, &[0, 1, 2, 3, 1, 2, 3], &[0, 4, 5, 6, 7]).unwrap();
    let ordering = StaticCscOrdering::minimum_degree(&matrix.to_scalar_csc::<4, 4, 10>().unwrap());
    let native =
        StaticBlockCscLdlt::<1, 1, 4, 4, 10, f64>::decompose_with_ordering(&matrix, ordering)
            .unwrap();
    let scalar = matrix.to_scalar_csc::<4, 4, 10>().unwrap();
    let scalar_factor = StaticCscLdlt::<4, 10, f64>::decompose(&scalar).unwrap();
    let rhs = Matrix::<4, 1, f64>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    assert_relative_eq!(
        native.try_solve::<4, 1>(&rhs).unwrap(),
        scalar_factor.solve(&rhs),
        epsilon = 1e-12,
        max_relative = 1e-12
    );
}

#[test]
fn native_block_ldlt_diagonal_pivoting_recovers_zero_leading_block() {
    type Pivoted = StaticBlockCscMatrix<1, 1, 2, 2, 3, f64>;
    let matrix = Pivoted::from_pattern(
        &[
            Matrix::from_rows([[0.0]]),
            Matrix::from_rows([[1.0]]),
            Matrix::from_rows([[2.0]]),
        ],
        &[0, 1, 1],
        &[0, 2, 3],
    )
    .unwrap();
    assert_eq!(
        StaticBlockCscLdlt::<1, 1, 2, 2, 3, f64>::decompose(&matrix),
        Err(SparseCholeskyError::ZeroPivot)
    );

    let scalar = matrix.to_scalar_csc::<2, 2, 3>().unwrap();
    let scalar_factor =
        StaticCscLdlt::<2, 3, f64>::decompose_with_diagonal_pivoting(&scalar, 1e-12).unwrap();
    let native =
        StaticBlockCscLdlt::<1, 1, 2, 2, 3, f64>::decompose_with_diagonal_pivoting(&matrix, 1e-12)
            .unwrap();
    assert_ne!(native.ordering().permutation(), &[0, 1]);
    let rhs = Matrix::<2, 1, f64>::from_columns([[3.0, 4.0]]);
    assert_relative_eq!(
        native.try_solve::<2, 1>(&rhs).unwrap(),
        scalar_factor.solve(&rhs),
        epsilon = 1e-12,
        max_relative = 1e-12
    );
}

#[test]
fn dense_block_ldlt_handles_cross_block_two_by_two_pivot() {
    type ScalarBlocks = StaticBlockCscMatrix<1, 1, 2, 2, 3, f64>;
    let matrix = ScalarBlocks::from_pattern(
        &[
            Matrix::from_rows([[0.0]]),
            Matrix::from_rows([[1.0]]),
            Matrix::from_rows([[0.0]]),
        ],
        &[0, 1, 1],
        &[0, 2, 3],
    )
    .unwrap();
    let factor = matrix.try_dense_ldlt::<2>().unwrap();
    assert_eq!(factor.pivot_blocks(), &[2, 3]);
    let rhs = Matrix::<2, 1, f64>::from_columns([[3.0, 4.0]]);
    assert_relative_eq!(
        factor.solve(&rhs),
        Matrix::from_columns([[4.0, 3.0]]),
        epsilon = 1e-12,
        max_relative = 1e-12
    );
}

#[test]
fn native_block_ldlt_handles_local_two_by_two_pivot() {
    type SingleBlock = StaticBlockCscMatrix<2, 2, 1, 1, 1, f64>;
    type Factor = StaticBlockCscLdlt<2, 2, 1, 1, 1, f64>;
    let matrix = SingleBlock::from_pattern(
        &[Matrix::from_rows([[0.0, 1.0], [1.0, 0.0]])],
        &[0],
        &[0, 1],
    )
    .unwrap();
    let factor = Factor::decompose(&matrix).unwrap();
    assert_eq!(factor.local_pivot_blocks(), &[[2, 3]]);
    assert_eq!(factor.local_permutations(), &[[0, 1]]);
    let rhs = Matrix::<2, 1, f64>::from_columns([[2.0, -3.0]]);
    assert_relative_eq!(
        factor.try_solve::<2, 1>(&rhs).unwrap(),
        Matrix::from_columns([[-3.0, 2.0]]),
        epsilon = 1e-12,
        max_relative = 1e-12
    );
}

#[test]
fn native_block_ldlt_handles_local_symmetric_permutation() {
    type SingleBlock = StaticBlockCscMatrix<2, 2, 1, 1, 1, f64>;
    type Factor = StaticBlockCscLdlt<2, 2, 1, 1, 1, f64>;
    let matrix = SingleBlock::from_pattern(
        &[Matrix::from_rows([[0.0, 1.0], [1.0, 4.0]])],
        &[0],
        &[0, 1],
    )
    .unwrap();
    let factor = Factor::decompose(&matrix).unwrap();
    assert_eq!(factor.local_pivot_blocks(), &[[1, 1]]);
    assert_eq!(factor.local_permutations(), &[[1, 0]]);
    let rhs = Matrix::<2, 1, f64>::from_columns([[2.0, -3.0]]);
    assert_relative_eq!(
        matrix![0.0_f64, 1.0; 1.0, 4.0] * factor.try_solve::<2, 1>(&rhs).unwrap(),
        rhs,
        epsilon = 1e-12,
        max_relative = 1e-12
    );
}

#[test]
fn native_block_ldlt_propagates_local_pivots_through_off_diagonal_blocks() {
    type Blocks = StaticBlockCscMatrix<2, 2, 2, 2, 3, f64>;
    type Factor = StaticBlockCscLdlt<2, 2, 2, 2, 3, f64>;
    let matrix = Blocks::from_pattern(
        &[
            Matrix::from_rows([[0.0, 1.0], [1.0, 4.0]]),
            Matrix::from_rows([[1.0, 0.0], [0.0, 1.0]]),
            Matrix::from_rows([[0.0, 1.0], [1.0, 5.0]]),
        ],
        &[0, 1, 1],
        &[0, 2, 3],
    )
    .unwrap();
    let factor = Factor::decompose(&matrix).unwrap();
    assert_eq!(factor.local_permutations(), &[[1, 0], [0, 1]]);
    let rhs = Matrix::<4, 1, f64>::from_columns([[1.0, -2.0, 3.0, 4.0]]);
    let solution = factor.try_solve::<4, 1>(&rhs).unwrap();
    let dense = matrix![
        0.0_f64, 1.0, 1.0, 0.0;
        1.0, 4.0, 0.0, 1.0;
        1.0, 0.0, 0.0, 1.0;
        0.0, 1.0, 1.0, 5.0;
    ];
    assert_relative_eq!(dense * solution, rhs, epsilon = 1e-12, max_relative = 1e-12);
}

#[test]
fn native_block_ldlt_propagates_local_two_by_two_d_blocks() {
    type Blocks = StaticBlockCscMatrix<2, 2, 2, 2, 3, f64>;
    type Factor = StaticBlockCscLdlt<2, 2, 2, 2, 3, f64>;
    let matrix = Blocks::from_pattern(
        &[
            Matrix::from_rows([[0.0, 1.0], [1.0, 0.0]]),
            Matrix::from_rows([[1.0, 0.0], [0.0, 1.0]]),
            Matrix::from_rows([[3.0, 0.0], [0.0, 4.0]]),
        ],
        &[0, 1, 1],
        &[0, 2, 3],
    )
    .unwrap();
    let factor = Factor::decompose(&matrix).unwrap();
    assert_eq!(factor.local_pivot_blocks(), &[[2, 3], [1, 1]]);
    let rhs = Matrix::<4, 1, f64>::from_columns([[1.0, -2.0, 3.0, 4.0]]);
    let solution = factor.try_solve::<4, 1>(&rhs).unwrap();
    let dense = matrix![
        0.0_f64, 1.0, 1.0, 0.0;
        1.0, 0.0, 0.0, 1.0;
        1.0, 0.0, 3.0, 0.0;
        0.0, 1.0, 0.0, 4.0;
    ];
    assert_relative_eq!(dense * solution, rhs, epsilon = 1e-12, max_relative = 1e-12);
}
