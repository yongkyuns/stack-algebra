//! Native block sparse Cholesky factorization.

use crate::{CscError, Matrix, Real, SparseCholeskyError, StaticCscOrdering, Zero};

use super::{
    detail::find_block_index,
    symbolic::{permute_block_matrix, StaticBlockCscCholeskyPattern},
    StaticBlockCscMatrix,
};

/// Native block sparse Cholesky factorization.
///
/// The factor stores dense lower-triangular diagonal blocks and dense
/// off-diagonal blocks. It requires square blocks and square block grids at
/// runtime; the explicit dimensions remain part of the type so storage is
/// still fully bounded at compile time.
///
/// `decompose` combines analysis and factorization. For repeated solves,
/// retain [`Self::pattern`] and use the symbolic pattern to recompute numeric
/// values instead of redoing fill analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticBlockCscCholesky<
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
    /// Computes numeric block Cholesky values from this pattern.
    #[inline]
    pub fn factor<const MAX_A_BLOCK_NNZ: usize, T: Real>(
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
        StaticBlockCscCholesky<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_L_BLOCK_NNZ,
            T,
        >,
        SparseCholeskyError,
    > {
        let mut output = StaticBlockCscCholesky {
            lower: StaticBlockCscMatrix {
                values: [Matrix::zeros(); MAX_L_BLOCK_NNZ],
                pattern: self.lower,
            },
            ordering: self.ordering,
        };
        self.factor_into(matrix, &mut output)?;
        Ok(output)
    }

    fn factor_into<const MAX_A_BLOCK_NNZ: usize, T: Real>(
        &self,
        matrix: &StaticBlockCscMatrix<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_A_BLOCK_NNZ,
            T,
        >,
        output: &mut StaticBlockCscCholesky<
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
            self.factor_natural_into(&permuted, output)?;
            output.ordering = self.ordering;
            return Ok(());
        }
        self.factor_natural_into(matrix, output)
    }

    fn factor_natural_into<const MAX_A_BLOCK_NNZ: usize, T: Real>(
        &self,
        matrix: &StaticBlockCscMatrix<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_A_BLOCK_NNZ,
            T,
        >,
        output: &mut StaticBlockCscCholesky<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_L_BLOCK_NNZ,
            T,
        >,
    ) -> Result<(), SparseCholeskyError> {
        output.lower.pattern = self.lower;
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
                    let lower_start = self.lower.column_starts()[previous];
                    let lower_end = self.lower.column_end(previous).unwrap_or(self.lower.nnz());
                    for index in lower_start..lower_end {
                        let row = self.lower.row_indices()[index];
                        if row >= column {
                            block_rank_update_sub(
                                &mut work[row],
                                &output.lower.values()[index],
                                &column_block,
                            );
                        }
                    }
                }
            }

            block_cholesky(&mut work[column])?;
            let diagonal_index = find_block_index(&self.lower, column, column)
                .ok_or(SparseCholeskyError::PatternMismatch)?;
            output.lower.values_mut()[diagonal_index] = work[column];

            let lower_start = self.lower.column_starts()[column];
            let lower_end = self.lower.column_end(column).unwrap_or(self.lower.nnz());
            let diagonal = work[column];
            for index in (lower_start + 1)..lower_end {
                let row = self.lower.row_indices()[index];
                block_solve_right_transpose(&mut work[row], &diagonal)?;
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
    StaticBlockCscCholesky<
        BLOCK_ROWS,
        BLOCK_COLS,
        BLOCK_GRID_ROWS,
        BLOCK_GRID_COLS,
        MAX_L_BLOCK_NNZ,
        T,
    >
where
    T: Real,
{
    /// Performs symbolic analysis and native block Cholesky factorization.
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
        StaticBlockCscCholeskyPattern::analyze(matrix)?.factor(matrix)
    }

    /// Performs native block Cholesky factorization with a fixed block ordering.
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
        StaticBlockCscCholeskyPattern::analyze_with_ordering(matrix, ordering)?.factor(matrix)
    }

    /// Returns the reusable symbolic block pattern.
    #[inline]
    pub fn pattern(
        &self,
    ) -> StaticBlockCscCholeskyPattern<
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
        pattern.factor_into(matrix, self)
    }

    /// Returns the native block lower factor.
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
            self.try_solve_cholesky_natural_in_place(&mut permuted)?;
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
        self.try_solve_cholesky_natural_in_place(rhs)
    }

    fn try_solve_cholesky_natural_in_place<const SCALAR_DIM: usize, const P: usize>(
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
                        value = value
                            - diagonal[(local_row, local_column)]
                                * rhs[(scalar_row - local_row + local_column, rhs_column)];
                    }
                    let result = value / diagonal[(local_row, local_row)];
                    if !result.is_finite() {
                        return Err(SparseCholeskyError::NonFinite);
                    }
                    rhs[(scalar_row, rhs_column)] = result;
                }
            }
        }

        for block_row in (0..BLOCK_GRID_ROWS).rev() {
            let diagonal_index = find_block_index(&self.lower.pattern, block_row, block_row)
                .ok_or(SparseCholeskyError::PatternMismatch)?;
            let diagonal = &self.lower.values()[diagonal_index];
            for rhs_column in 0..P {
                for local_row in (0..BLOCK_ROWS).rev() {
                    let scalar_row = block_row * BLOCK_ROWS + local_row;
                    let mut value = rhs[(scalar_row, rhs_column)];
                    for next in (local_row + 1)..BLOCK_ROWS {
                        value = value
                            - diagonal[(next, local_row)]
                                * rhs[(block_row * BLOCK_ROWS + next, rhs_column)];
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
                    let result = value / diagonal[(local_row, local_row)];
                    if !result.is_finite() {
                        return Err(SparseCholeskyError::NonFinite);
                    }
                    rhs[(scalar_row, rhs_column)] = result;
                }
            }
        }
        Ok(())
    }
}

#[inline]
fn block_rank_update_sub<const ROWS: usize, const COLS: usize, T: Real>(
    target: &mut Matrix<ROWS, COLS, T>,
    lhs: &Matrix<ROWS, COLS, T>,
    rhs: &Matrix<ROWS, COLS, T>,
) {
    for row in 0..ROWS {
        for column in 0..COLS {
            let mut value = target[(row, column)];
            for inner in 0..ROWS {
                value = value - lhs[(row, inner)] * rhs[(column, inner)];
            }
            target[(row, column)] = value;
        }
    }
}

#[inline]
fn block_cholesky<const ROWS: usize, const COLS: usize, T: Real>(
    block: &mut Matrix<ROWS, COLS, T>,
) -> Result<(), SparseCholeskyError> {
    for column in 0..ROWS {
        for row in column..ROWS {
            let mut value = block[(row, column)];
            for previous in 0..column {
                value = value - block[(row, previous)] * block[(column, previous)];
            }
            if !value.is_finite() {
                return Err(SparseCholeskyError::NonFinite);
            }
            if row == column {
                if value <= T::zero() {
                    return Err(SparseCholeskyError::NotPositiveDefinite);
                }
                block[(row, column)] = value.sqrt();
            } else {
                block[(row, column)] = value / block[(column, column)];
            }
        }
    }
    Ok(())
}

#[inline]
fn block_solve_right_transpose<const ROWS: usize, const COLS: usize, T: Real>(
    block: &mut Matrix<ROWS, COLS, T>,
    diagonal: &Matrix<ROWS, COLS, T>,
) -> Result<(), SparseCholeskyError> {
    for row in 0..ROWS {
        for column in 0..ROWS {
            let mut value = block[(row, column)];
            for previous in 0..column {
                value = value - diagonal[(column, previous)] * block[(row, previous)];
            }
            let result = value / diagonal[(column, column)];
            if !result.is_finite() {
                return Err(SparseCholeskyError::NonFinite);
            }
            block[(row, column)] = result;
        }
    }
    Ok(())
}
