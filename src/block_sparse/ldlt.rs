//! Native fixed-capacity block sparse LDLT factorization.

use crate::sparse::map_ldlt_error;
use crate::{
    CscError, Ldlt, Matrix, MatrixScalar, Real, SparseCholeskyError, StaticCscOrdering, Zero,
};

use super::{
    detail::find_block_index,
    storage::StaticBlockCscMatrix,
    symbolic::{permute_block_matrix, StaticBlockCscCholeskyPattern, StaticBlockCscLdltPattern},
};

/// Native fixed-capacity block sparse LDLᵀ factorization.
///
/// Diagonal blocks use compact `L·D` storage with local Bunch–Kaufman scalar
/// pivots. Local permutations are retained separately so off-diagonal blocks
/// remain dense and allocation-free.
///
/// This is the block analogue of [`crate::StaticCscLdlt`]. It supports local
/// scalar pivoting inside diagonal blocks and exposes the selected pivots via
/// [`Self::local_pivot_blocks`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticBlockCscLdlt<
    const BLOCK_ROWS: usize,
    const BLOCK_COLS: usize,
    const BLOCK_GRID_ROWS: usize,
    const BLOCK_GRID_COLS: usize,
    const MAX_L_BLOCK_NNZ: usize,
    T = f32,
> where
    T: Copy + Zero,
{
    lower: StaticBlockCscMatrix<
        BLOCK_ROWS,
        BLOCK_COLS,
        BLOCK_GRID_ROWS,
        BLOCK_GRID_COLS,
        MAX_L_BLOCK_NNZ,
        T,
    >,
    ordering: StaticCscOrdering<BLOCK_GRID_ROWS>,
    local_permutations: [[usize; BLOCK_ROWS]; BLOCK_GRID_ROWS],
    local_pivots: [[u8; BLOCK_ROWS]; BLOCK_GRID_ROWS],
}

impl<
        const BLOCK_ROWS: usize,
        const BLOCK_COLS: usize,
        const BLOCK_GRID_ROWS: usize,
        const BLOCK_GRID_COLS: usize,
        const MAX_L_BLOCK_NNZ: usize,
    >
    StaticBlockCscCholeskyPattern<
        BLOCK_ROWS,
        BLOCK_COLS,
        BLOCK_GRID_ROWS,
        BLOCK_GRID_COLS,
        MAX_L_BLOCK_NNZ,
    >
{
    /// Computes native block LDLᵀ values from this symbolic pattern.
    #[inline]
    pub fn factor_ldlt<const MAX_A_BLOCK_NNZ: usize, T: Real + MatrixScalar>(
        &self,
        matrix: &StaticBlockCscMatrix<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_A_BLOCK_NNZ,
            T,
        >,
    ) -> Result<
        StaticBlockCscLdlt<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_L_BLOCK_NNZ,
            T,
        >,
        SparseCholeskyError,
    > {
        let mut output = StaticBlockCscLdlt {
            lower: StaticBlockCscMatrix {
                values: [Matrix::zeros(); MAX_L_BLOCK_NNZ],
                pattern: self.lower,
            },
            ordering: self.ordering,
            local_permutations: core::array::from_fn(|_| core::array::from_fn(|index| index)),
            local_pivots: [[1; BLOCK_ROWS]; BLOCK_GRID_ROWS],
        };
        self.factor_ldlt_into(matrix, &mut output)?;
        Ok(output)
    }

    fn factor_ldlt_into<const MAX_A_BLOCK_NNZ: usize, T: Real + MatrixScalar>(
        &self,
        matrix: &StaticBlockCscMatrix<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_A_BLOCK_NNZ,
            T,
        >,
        output: &mut StaticBlockCscLdlt<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_L_BLOCK_NNZ,
            T,
        >,
    ) -> Result<(), SparseCholeskyError> {
        if BLOCK_ROWS != BLOCK_COLS || BLOCK_GRID_ROWS != BLOCK_GRID_COLS {
            return Err(SparseCholeskyError::Csc(CscError::LengthMismatch));
        }
        if !self.ordering.is_identity() {
            let permuted = permute_block_matrix(matrix, self.ordering)?;
            self.factor_ldlt_natural_into(&permuted, output)?;
            output.ordering = self.ordering;
            return Ok(());
        }
        self.factor_ldlt_natural_into(matrix, output)
    }

    fn factor_ldlt_natural_into<const MAX_A_BLOCK_NNZ: usize, T: Real + MatrixScalar>(
        &self,
        matrix: &StaticBlockCscMatrix<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_A_BLOCK_NNZ,
            T,
        >,
        output: &mut StaticBlockCscLdlt<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_L_BLOCK_NNZ,
            T,
        >,
    ) -> Result<(), SparseCholeskyError> {
        for column in 0..BLOCK_GRID_COLS {
            let start = matrix.pattern.column_starts()[column];
            let end = matrix.pattern.column_end(column).unwrap_or(matrix.nnz());
            for index in start..end {
                let row = matrix.pattern.row_indices()[index];
                if row >= column && find_block_index(&self.lower, row, column).is_none() {
                    return Err(SparseCholeskyError::PatternMismatch);
                }
            }
        }

        output.lower.pattern = self.lower;
        output.local_permutations = core::array::from_fn(|_| core::array::from_fn(|index| index));
        output.local_pivots = [[1; BLOCK_ROWS]; BLOCK_GRID_ROWS];
        for column in 0..BLOCK_GRID_COLS {
            let mut work = [Matrix::<BLOCK_ROWS, BLOCK_COLS, T>::zeros(); BLOCK_GRID_ROWS];
            let matrix_start = matrix.pattern.column_starts()[column];
            let matrix_end = matrix.pattern.column_end(column).unwrap_or(matrix.nnz());
            for index in matrix_start..matrix_end {
                let row = matrix.pattern.row_indices()[index];
                if row >= column {
                    work[row] = matrix.values()[index];
                }
            }

            for previous in 0..column {
                if let Some(column_index) = find_block_index(&self.lower, column, previous) {
                    let column_block = output.lower.values()[column_index];
                    let diagonal_index = find_block_index(&self.lower, previous, previous)
                        .ok_or(SparseCholeskyError::PatternMismatch)?;
                    let diagonal = output.lower.values()[diagonal_index];
                    let lower_start = self.lower.column_starts()[previous];
                    let lower_end = self.lower.column_end(previous).unwrap_or(self.lower.nnz());
                    for index in lower_start..lower_end {
                        let row = self.lower.row_indices()[index];
                        if row >= column {
                            block_rank_update_ldlt_sub(
                                &mut work[row],
                                &output.lower.values()[index],
                                &diagonal,
                                &output.local_pivots[previous],
                                &column_block,
                            );
                        }
                    }
                }
            }

            let mut diagonal_input = Matrix::<BLOCK_ROWS, BLOCK_ROWS, T>::zeros();
            for row in 0..BLOCK_ROWS {
                for local_column in 0..BLOCK_ROWS {
                    diagonal_input[(row, local_column)] = work[column][(row, local_column)];
                }
            }
            let diagonal_factor = Ldlt::try_decompose(&diagonal_input).map_err(map_ldlt_error)?;
            let permutation = *diagonal_factor.permutation_indices();
            output.local_permutations[column] = permutation;
            output.local_pivots[column] = *diagonal_factor.pivot_blocks();

            for previous in 0..column {
                if let Some(index) = find_block_index(&self.lower, column, previous) {
                    let block = output.lower.values()[index];
                    output.lower.values_mut()[index] = permute_block_rows(&block, &permutation);
                }
            }
            for block in work.iter_mut().skip(column + 1) {
                let current = *block;
                *block = permute_block_columns(&current, &permutation);
            }

            let diagonal_index = find_block_index(&self.lower, column, column)
                .ok_or(SparseCholeskyError::PatternMismatch)?;
            output.lower.values_mut()[diagonal_index] =
                store_ldlt_block::<BLOCK_ROWS, BLOCK_COLS, T>(&diagonal_factor);

            let lower_start = self.lower.column_starts()[column];
            let lower_end = self.lower.column_end(column).unwrap_or(self.lower.nnz());
            let diagonal = output.lower.values()[diagonal_index];
            let pivots = diagonal_factor.pivot_blocks();
            for index in (lower_start + 1)..lower_end {
                let row = self.lower.row_indices()[index];
                block_solve_right_ldlt_transpose(&mut work[row], &diagonal, pivots)?;
                output.lower.values_mut()[index] = work[row];
            }
        }
        Ok(())
    }
}

impl<
        const BLOCK_ROWS: usize,
        const BLOCK_COLS: usize,
        const BLOCK_GRID_ROWS: usize,
        const BLOCK_GRID_COLS: usize,
        const MAX_L_BLOCK_NNZ: usize,
        T,
    >
    StaticBlockCscLdlt<BLOCK_ROWS, BLOCK_COLS, BLOCK_GRID_ROWS, BLOCK_GRID_COLS, MAX_L_BLOCK_NNZ, T>
where
    T: Real + MatrixScalar,
{
    /// Performs symbolic analysis and native block LDLᵀ factorization.
    #[inline]
    pub fn decompose<const MAX_A_BLOCK_NNZ: usize>(
        matrix: &StaticBlockCscMatrix<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_A_BLOCK_NNZ,
            T,
        >,
    ) -> Result<Self, SparseCholeskyError> {
        StaticBlockCscCholeskyPattern::analyze(matrix)?.factor_ldlt(matrix)
    }

    /// Performs native block LDLᵀ factorization with a fixed block ordering.
    #[inline]
    pub fn decompose_with_ordering<const MAX_A_BLOCK_NNZ: usize>(
        matrix: &StaticBlockCscMatrix<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_A_BLOCK_NNZ,
            T,
        >,
        ordering: StaticCscOrdering<BLOCK_GRID_ROWS>,
    ) -> Result<Self, SparseCholeskyError> {
        StaticBlockCscCholeskyPattern::analyze_with_ordering(matrix, ordering)?.factor_ldlt(matrix)
    }

    /// Performs native block LDLᵀ factorization with analysis-time diagonal
    /// block pivoting.
    #[inline]
    pub fn decompose_with_diagonal_pivoting<const MAX_A_BLOCK_NNZ: usize>(
        matrix: &StaticBlockCscMatrix<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_A_BLOCK_NNZ,
            T,
        >,
        threshold: T,
    ) -> Result<Self, SparseCholeskyError> {
        StaticBlockCscCholeskyPattern::analyze_with_diagonal_pivoting(matrix, threshold)?
            .factor_ldlt(matrix)
    }

    /// Returns the reusable symbolic block pattern.
    #[inline]
    pub fn pattern(
        &self,
    ) -> StaticBlockCscLdltPattern<
        BLOCK_ROWS,
        BLOCK_COLS,
        BLOCK_GRID_ROWS,
        BLOCK_GRID_COLS,
        MAX_L_BLOCK_NNZ,
    > {
        StaticBlockCscCholeskyPattern {
            lower: self.lower.pattern,
            ordering: self.ordering,
        }
    }

    /// Recomputes numeric values using the existing block factor pattern.
    #[inline]
    pub fn recompute<const MAX_A_BLOCK_NNZ: usize>(
        &mut self,
        matrix: &StaticBlockCscMatrix<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_A_BLOCK_NNZ,
            T,
        >,
    ) -> Result<(), SparseCholeskyError> {
        let pattern = self.pattern();
        pattern.factor_ldlt_into(matrix, self)
    }

    /// Returns the native block lower factor with compact diagonal `L·D`
    /// blocks.
    #[inline]
    pub fn lower(
        &self,
    ) -> &StaticBlockCscMatrix<
        BLOCK_ROWS,
        BLOCK_COLS,
        BLOCK_GRID_ROWS,
        BLOCK_GRID_COLS,
        MAX_L_BLOCK_NNZ,
        T,
    > {
        &self.lower
    }

    /// Returns the block ordering used by this factorization.
    #[inline]
    pub const fn ordering(&self) -> StaticCscOrdering<BLOCK_GRID_ROWS> {
        self.ordering
    }

    /// Returns the local scalar pivot layout for each diagonal block.
    #[inline]
    pub fn local_pivot_blocks(&self) -> &[[u8; BLOCK_ROWS]; BLOCK_GRID_ROWS] {
        &self.local_pivots
    }

    /// Returns the local scalar permutations for each diagonal block.
    #[inline]
    pub fn local_permutations(&self) -> &[[usize; BLOCK_ROWS]; BLOCK_GRID_ROWS] {
        &self.local_permutations
    }

    /// Returns the exact inline storage footprint of this numeric factor.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Solves a block system for one or more scalar RHS columns.
    #[inline]
    pub fn try_solve<const SCALAR_DIM: usize, const P: usize>(
        &self,
        rhs: &Matrix<SCALAR_DIM, P, T>,
    ) -> Result<Matrix<SCALAR_DIM, P, T>, SparseCholeskyError> {
        let mut output = *rhs;
        self.try_solve_in_place(&mut output)?;
        Ok(output)
    }

    /// Solves a block system into caller-provided output storage.
    #[inline]
    pub fn try_solve_into<const SCALAR_DIM: usize, const P: usize>(
        &self,
        rhs: &Matrix<SCALAR_DIM, P, T>,
        output: &mut Matrix<SCALAR_DIM, P, T>,
    ) -> Result<(), SparseCholeskyError> {
        *output = *rhs;
        self.try_solve_in_place(output)
    }

    /// Solves a block system in place.
    #[inline]
    pub fn try_solve_in_place<const SCALAR_DIM: usize, const P: usize>(
        &self,
        rhs: &mut Matrix<SCALAR_DIM, P, T>,
    ) -> Result<(), SparseCholeskyError> {
        if BLOCK_ROWS != BLOCK_COLS
            || BLOCK_GRID_ROWS != BLOCK_GRID_COLS
            || SCALAR_DIM != BLOCK_ROWS * BLOCK_GRID_ROWS
        {
            return Err(SparseCholeskyError::Csc(CscError::LengthMismatch));
        }

        if !self.ordering.is_identity() {
            let mut permuted = Matrix::<SCALAR_DIM, P, T>::zeros();
            for ordered_block in 0..BLOCK_GRID_ROWS {
                let original_block = self.ordering.permutation()[ordered_block];
                for local_row in 0..BLOCK_ROWS {
                    for rhs_column in 0..P {
                        permuted[(ordered_block * BLOCK_ROWS + local_row, rhs_column)] =
                            rhs[(original_block * BLOCK_ROWS + local_row, rhs_column)];
                    }
                }
            }
            self.permute_local_rhs(&mut permuted);
            self.try_solve_ldlt_natural_in_place(&mut permuted)?;
            self.unpermute_local_rhs(&mut permuted);
            for ordered_block in 0..BLOCK_GRID_ROWS {
                let original_block = self.ordering.permutation()[ordered_block];
                for local_row in 0..BLOCK_ROWS {
                    for rhs_column in 0..P {
                        rhs[(original_block * BLOCK_ROWS + local_row, rhs_column)] =
                            permuted[(ordered_block * BLOCK_ROWS + local_row, rhs_column)];
                    }
                }
            }
            return Ok(());
        }
        self.permute_local_rhs(rhs);
        let result = self.try_solve_ldlt_natural_in_place(rhs);
        self.unpermute_local_rhs(rhs);
        result
    }

    fn try_solve_ldlt_natural_in_place<const SCALAR_DIM: usize, const P: usize>(
        &self,
        rhs: &mut Matrix<SCALAR_DIM, P, T>,
    ) -> Result<(), SparseCholeskyError> {
        for block_column in 0..BLOCK_GRID_COLS {
            let diagonal_index = find_block_index(&self.lower.pattern, block_column, block_column)
                .ok_or(SparseCholeskyError::PatternMismatch)?;
            let diagonal = &self.lower.values()[diagonal_index];
            for rhs_column in 0..P {
                for local_row in 0..BLOCK_ROWS {
                    let scalar_row = block_column * BLOCK_ROWS + local_row;
                    let mut value = rhs[(scalar_row, rhs_column)];
                    for previous in 0..block_column {
                        if let Some(index) =
                            find_block_index(&self.lower.pattern, block_column, previous)
                        {
                            let block = &self.lower.values()[index];
                            for local_column in 0..BLOCK_COLS {
                                value = value
                                    - block[(local_row, local_column)]
                                        * rhs[(previous * BLOCK_COLS + local_column, rhs_column)];
                            }
                        }
                    }
                    for local_column in 0..local_row {
                        let l = block_l_value(
                            diagonal,
                            &self.local_pivots[block_column],
                            local_row,
                            local_column,
                        );
                        value =
                            value - l * rhs[(scalar_row - local_row + local_column, rhs_column)];
                    }
                    if !value.is_finite() {
                        return Err(SparseCholeskyError::NonFinite);
                    }
                    rhs[(scalar_row, rhs_column)] = value;
                }
            }
        }

        for block_column in 0..BLOCK_GRID_COLS {
            let diagonal_index = find_block_index(&self.lower.pattern, block_column, block_column)
                .ok_or(SparseCholeskyError::PatternMismatch)?;
            let diagonal = &self.lower.values()[diagonal_index];
            let pivots = self.local_pivots[block_column];
            for rhs_column in 0..P {
                let mut local_row = 0;
                while local_row < BLOCK_ROWS {
                    if pivots[local_row] == 2 {
                        let first = rhs[(block_column * BLOCK_ROWS + local_row, rhs_column)];
                        let second = rhs[(block_column * BLOCK_ROWS + local_row + 1, rhs_column)];
                        let d11 = diagonal[(local_row, local_row)];
                        let d12 = diagonal[(local_row + 1, local_row)];
                        let d22 = diagonal[(local_row + 1, local_row + 1)];
                        let (first_result, second_result) =
                            solve_two_by_two(first, second, d11, d12, d22)?;
                        rhs[(block_column * BLOCK_ROWS + local_row, rhs_column)] = first_result;
                        rhs[(block_column * BLOCK_ROWS + local_row + 1, rhs_column)] =
                            second_result;
                        local_row += 2;
                    } else {
                        let scalar_row = block_column * BLOCK_ROWS + local_row;
                        let value =
                            rhs[(scalar_row, rhs_column)] / diagonal[(local_row, local_row)];
                        if !value.is_finite() {
                            return Err(SparseCholeskyError::NonFinite);
                        }
                        rhs[(scalar_row, rhs_column)] = value;
                        local_row += 1;
                    }
                }
            }
        }

        for block_row in (0..BLOCK_GRID_ROWS).rev() {
            let diagonal_index = find_block_index(&self.lower.pattern, block_row, block_row)
                .ok_or(SparseCholeskyError::PatternMismatch)?;
            let diagonal = &self.lower.values()[diagonal_index];
            let pivots = self.local_pivots[block_row];
            for rhs_column in 0..P {
                for local_row in (0..BLOCK_ROWS).rev() {
                    let scalar_row = block_row * BLOCK_ROWS + local_row;
                    let mut value = rhs[(scalar_row, rhs_column)];
                    for next in (local_row + 1)..BLOCK_ROWS {
                        let l = block_l_value(diagonal, &pivots, next, local_row);
                        value = value - l * rhs[(block_row * BLOCK_ROWS + next, rhs_column)];
                    }
                    for next_block in (block_row + 1)..BLOCK_GRID_ROWS {
                        if let Some(index) =
                            find_block_index(&self.lower.pattern, next_block, block_row)
                        {
                            let block = &self.lower.values()[index];
                            for next_local in 0..BLOCK_ROWS {
                                value = value
                                    - block[(next_local, local_row)]
                                        * rhs[(next_block * BLOCK_ROWS + next_local, rhs_column)];
                            }
                        }
                    }
                    if !value.is_finite() {
                        return Err(SparseCholeskyError::NonFinite);
                    }
                    rhs[(scalar_row, rhs_column)] = value;
                }
            }
        }

        Ok(())
    }

    #[inline]
    fn permute_local_rhs<const SCALAR_DIM: usize, const P: usize>(
        &self,
        rhs: &mut Matrix<SCALAR_DIM, P, T>,
    ) {
        let original = *rhs;
        for block in 0..BLOCK_GRID_ROWS {
            let permutation = &self.local_permutations[block];
            for local_row in 0..BLOCK_ROWS {
                for rhs_column in 0..P {
                    rhs[(block * BLOCK_ROWS + local_row, rhs_column)] =
                        original[(block * BLOCK_ROWS + permutation[local_row], rhs_column)];
                }
            }
        }
    }

    #[inline]
    fn unpermute_local_rhs<const SCALAR_DIM: usize, const P: usize>(
        &self,
        rhs: &mut Matrix<SCALAR_DIM, P, T>,
    ) {
        let solved = *rhs;
        for block in 0..BLOCK_GRID_ROWS {
            let permutation = &self.local_permutations[block];
            for local_row in 0..BLOCK_ROWS {
                for rhs_column in 0..P {
                    rhs[(block * BLOCK_ROWS + permutation[local_row], rhs_column)] =
                        solved[(block * BLOCK_ROWS + local_row, rhs_column)];
                }
            }
        }
    }
}

#[inline]
pub(super) fn block_rank_update_ldlt_sub<const ROWS: usize, const COLS: usize, T: Real>(
    target: &mut Matrix<ROWS, COLS, T>,
    lhs: &Matrix<ROWS, COLS, T>,
    diagonal: &Matrix<ROWS, COLS, T>,
    pivots: &[u8; ROWS],
    rhs: &Matrix<ROWS, COLS, T>,
) {
    for row in 0..ROWS {
        for column in 0..COLS {
            let mut value = target[(row, column)];
            for left in 0..ROWS {
                for right in 0..ROWS {
                    let d = block_d_value(diagonal, pivots, left, right);
                    value = value - lhs[(row, left)] * d * rhs[(column, right)];
                }
            }
            target[(row, column)] = value;
        }
    }
}

#[inline]
pub(super) fn block_ldlt<const ROWS: usize, const COLS: usize, T: Real>(
    block: &mut Matrix<ROWS, COLS, T>,
) -> Result<(), SparseCholeskyError> {
    for column in 0..ROWS {
        for row in column..ROWS {
            let mut value = block[(row, column)];
            for previous in 0..column {
                value = value
                    - block[(row, previous)]
                        * block[(previous, previous)]
                        * block[(column, previous)];
            }
            if !value.is_finite() {
                return Err(SparseCholeskyError::NonFinite);
            }
            if row == column {
                if value == T::zero() {
                    return Err(SparseCholeskyError::ZeroPivot);
                }
                block[(row, column)] = value;
            } else {
                let result = value / block[(column, column)];
                if !result.is_finite() {
                    return Err(SparseCholeskyError::NonFinite);
                }
                block[(row, column)] = result;
            }
        }
    }
    Ok(())
}

#[inline]
fn solve_two_by_two<T: Real>(
    first: T,
    second: T,
    d11: T,
    d12: T,
    d22: T,
) -> Result<(T, T), SparseCholeskyError> {
    let scale = first
        .abs()
        .max(second.abs())
        .max(d11.abs())
        .max(d12.abs())
        .max(d22.abs());
    if !scale.is_finite() {
        return Err(SparseCholeskyError::NonFinite);
    }
    if scale == T::zero() {
        return Err(SparseCholeskyError::ZeroPivot);
    }

    let normalized_first = first / scale;
    let normalized_second = second / scale;
    let normalized_d11 = d11 / scale;
    let normalized_d12 = d12 / scale;
    let normalized_d22 = d22 / scale;
    let determinant = normalized_d11 * normalized_d22 - normalized_d12 * normalized_d12;
    if !determinant.is_finite() {
        return Err(SparseCholeskyError::NonFinite);
    }
    if determinant == T::zero() {
        return Err(SparseCholeskyError::ZeroPivot);
    }

    let first_result =
        (normalized_first * normalized_d22 - normalized_second * normalized_d12) / determinant;
    let second_result =
        (normalized_second * normalized_d11 - normalized_first * normalized_d12) / determinant;
    if !first_result.is_finite() || !second_result.is_finite() {
        return Err(SparseCholeskyError::NonFinite);
    }
    Ok((first_result, second_result))
}

#[inline]
pub(super) fn block_solve_right_ldlt_transpose<const ROWS: usize, const COLS: usize, T: Real>(
    block: &mut Matrix<ROWS, COLS, T>,
    diagonal: &Matrix<ROWS, COLS, T>,
    pivots: &[u8; ROWS],
) -> Result<(), SparseCholeskyError> {
    for row in 0..ROWS {
        for column in 0..COLS {
            let mut value = block[(row, column)];
            for previous in 0..column {
                let l = if pivots[previous] == 2 && column == previous + 1 {
                    T::zero()
                } else {
                    diagonal[(column, previous)]
                };
                value = value - block[(row, previous)] * l;
            }
            if !value.is_finite() {
                return Err(SparseCholeskyError::NonFinite);
            }
            block[(row, column)] = value;
        }
        let mut column = 0;
        while column < COLS {
            if pivots[column] == 2 {
                let first = block[(row, column)];
                let second = block[(row, column + 1)];
                let d11 = diagonal[(column, column)];
                let d12 = diagonal[(column + 1, column)];
                let d22 = diagonal[(column + 1, column + 1)];
                let (first_result, second_result) = solve_two_by_two(first, second, d11, d12, d22)?;
                block[(row, column)] = first_result;
                block[(row, column + 1)] = second_result;
                column += 2;
            } else {
                let result = block[(row, column)] / diagonal[(column, column)];
                if !result.is_finite() {
                    return Err(SparseCholeskyError::NonFinite);
                }
                block[(row, column)] = result;
                column += 1;
            }
        }
    }
    Ok(())
}

#[inline]
fn block_d_value<const ROWS: usize, const COLS: usize, T: Copy + Zero>(
    diagonal: &Matrix<ROWS, COLS, T>,
    pivots: &[u8; ROWS],
    row: usize,
    column: usize,
) -> T {
    if row == column || (row == column + 1 && pivots[column] == 2) {
        diagonal[(row, column)]
    } else if column == row + 1 && pivots[row] == 2 {
        diagonal[(column, row)]
    } else {
        T::zero()
    }
}

#[inline]
fn block_l_value<const ROWS: usize, const COLS: usize, T: Copy + Zero>(
    diagonal: &Matrix<ROWS, COLS, T>,
    pivots: &[u8; ROWS],
    row: usize,
    column: usize,
) -> T {
    if pivots[column] == 2 && row == column + 1 {
        T::zero()
    } else {
        diagonal[(row, column)]
    }
}

#[inline]
fn permute_block_rows<const ROWS: usize, const COLS: usize, T: Copy + Zero>(
    block: &Matrix<ROWS, COLS, T>,
    permutation: &[usize; ROWS],
) -> Matrix<ROWS, COLS, T> {
    Matrix::from_fn(|row, column| block[(permutation[row], column)])
}

#[inline]
fn permute_block_columns<const ROWS: usize, const COLS: usize, T: Copy + Zero>(
    block: &Matrix<ROWS, COLS, T>,
    permutation: &[usize; ROWS],
) -> Matrix<ROWS, COLS, T> {
    Matrix::from_fn(|row, column| block[(row, permutation[column])])
}

#[inline]
fn store_ldlt_block<const D: usize, const COLS: usize, T: Real + crate::MatrixScalar>(
    factor: &Ldlt<D, T>,
) -> Matrix<D, COLS, T> {
    let lower = factor.lower();
    let diagonal = factor.diagonal_matrix();
    let pivots = factor.pivot_blocks();
    Matrix::from_fn(|row, column| {
        if row == column || (row == column + 1 && pivots[column] == 2) {
            diagonal[(row, column)]
        } else if row > column && column < D {
            lower[(row, column)]
        } else {
            T::zero()
        }
    })
}
