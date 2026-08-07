//! Symbolic analysis and ordering for block sparse factors.

use crate::{
    CscError, Matrix, Real, SparseCholeskyError, StaticCscMatrix, StaticCscOrdering,
    StaticCscPattern, Zero,
};

use super::{
    block_ldlt, block_rank_update_ldlt_sub, block_solve_right_ldlt_transpose, StaticBlockCscMatrix,
};

/// Symbolic pattern for a native block sparse Cholesky factor.
///
/// Create one with [`Self::analyze`] (or an ordering/pivoting variant), then
/// reuse it with [`Self::factor`] for changing numeric block values. The
/// factor pattern owns only bounded arrays and can be stored alongside a
/// generated solver state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticBlockCscCholeskyPattern<
    const BLOCK_ROWS: usize,
    const BLOCK_COLS: usize,
    const BLOCK_GRID_ROWS: usize,
    const BLOCK_GRID_COLS: usize,
    const MAX_L_BLOCK_NNZ: usize,
> {
    pub(super) lower: StaticCscPattern<BLOCK_GRID_ROWS, BLOCK_GRID_COLS, MAX_L_BLOCK_NNZ>,
    pub(super) ordering: StaticCscOrdering<BLOCK_GRID_ROWS>,
}

/// Symbolic pattern shared by native block Cholesky and LDLᵀ factors.
pub type StaticBlockCscLdltPattern<
    const BLOCK_ROWS: usize,
    const BLOCK_COLS: usize,
    const BLOCK_GRID_ROWS: usize,
    const BLOCK_GRID_COLS: usize,
    const MAX_L_BLOCK_NNZ: usize,
> = StaticBlockCscCholeskyPattern<
    BLOCK_ROWS,
    BLOCK_COLS,
    BLOCK_GRID_ROWS,
    BLOCK_GRID_COLS,
    MAX_L_BLOCK_NNZ,
>;

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
    /// Returns the exact inline storage footprint of this symbolic factor.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Analyzes lower-triangular block structure and computes block fill.
    #[inline]
    pub fn analyze<const MAX_A_BLOCK_NNZ: usize, T: Copy + Zero>(
        matrix: &StaticBlockCscMatrix<
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_GRID_ROWS,
            BLOCK_GRID_COLS,
            MAX_A_BLOCK_NNZ,
            T,
        >,
    ) -> Result<Self, SparseCholeskyError> {
        if BLOCK_ROWS != BLOCK_COLS || BLOCK_GRID_ROWS != BLOCK_GRID_COLS {
            return Err(SparseCholeskyError::Csc(CscError::LengthMismatch));
        }

        let mut structure =
            StaticCscMatrix::<BLOCK_GRID_ROWS, BLOCK_GRID_COLS, MAX_L_BLOCK_NNZ, u8>::new();
        for column in 0..BLOCK_GRID_COLS {
            let mut reachable = [false; BLOCK_GRID_ROWS];
            let start = matrix.pattern.column_starts()[column];
            let end = matrix.pattern.column_end(column).unwrap_or(matrix.nnz());
            for index in start..end {
                let row = matrix.pattern.row_indices()[index];
                if row >= column {
                    reachable[row] = true;
                }
            }
            reachable[column] = true;

            for previous in 0..column {
                if structure.get(column, previous).is_some() {
                    let previous_start = structure.column_starts()[previous];
                    let previous_end = structure.column_end(previous).unwrap_or(structure.nnz());
                    for index in previous_start..previous_end {
                        let row = structure.row_indices()[index];
                        if row >= column {
                            reachable[row] = true;
                        }
                    }
                }
            }

            for (row, &is_reachable) in reachable.iter().enumerate().skip(column) {
                if is_reachable {
                    structure
                        .insert(row, column, 1)
                        .map_err(SparseCholeskyError::from)?;
                }
            }
        }

        Ok(Self {
            lower: *structure.pattern(),
            ordering: StaticCscOrdering::identity(),
        })
    }

    /// Analyzes block structure after applying a symmetric block ordering.
    #[inline]
    pub fn analyze_with_ordering<const MAX_A_BLOCK_NNZ: usize, T: Copy + Zero>(
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
        if BLOCK_ROWS != BLOCK_COLS || BLOCK_GRID_ROWS != BLOCK_GRID_COLS {
            return Err(SparseCholeskyError::Csc(CscError::LengthMismatch));
        }
        if ordering.is_identity() {
            return Self::analyze(matrix);
        }
        let permuted = permute_block_matrix(matrix, ordering)?;
        let natural = Self::analyze(&permuted)?;
        Ok(Self {
            lower: natural.lower,
            ordering,
        })
    }

    /// Analyzes the matrix using fixed-workspace block diagonal pivoting.
    ///
    /// The pivot search selects the largest-magnitude diagonal entry from
    /// each remaining diagonal block. This supports block permutations while
    /// retaining compact factor storage inside each dense block.
    #[inline]
    pub fn analyze_with_diagonal_pivoting<const MAX_A_BLOCK_NNZ: usize, T: Real>(
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
        let ordering = block_diagonal_pivot_ordering(matrix, threshold)?;
        Self::analyze_with_ordering(matrix, ordering)
    }

    /// Returns the analyzed lower block pattern.
    #[inline]
    pub fn lower(&self) -> &StaticCscPattern<BLOCK_GRID_ROWS, BLOCK_GRID_COLS, MAX_L_BLOCK_NNZ> {
        &self.lower
    }

    /// Returns the block ordering used by this symbolic factorization.
    #[inline]
    pub const fn ordering(&self) -> StaticCscOrdering<BLOCK_GRID_ROWS> {
        self.ordering
    }
}

pub(super) fn permute_block_matrix<
    const BLOCK_ROWS: usize,
    const BLOCK_COLS: usize,
    const BLOCK_GRID_ROWS: usize,
    const BLOCK_GRID_COLS: usize,
    const MAX_BLOCK_NNZ: usize,
    T: Copy + Zero,
>(
    matrix: &StaticBlockCscMatrix<
        BLOCK_ROWS,
        BLOCK_COLS,
        BLOCK_GRID_ROWS,
        BLOCK_GRID_COLS,
        MAX_BLOCK_NNZ,
        T,
    >,
    ordering: StaticCscOrdering<BLOCK_GRID_ROWS>,
) -> Result<
    StaticBlockCscMatrix<
        BLOCK_ROWS,
        BLOCK_COLS,
        BLOCK_GRID_ROWS,
        BLOCK_GRID_COLS,
        MAX_BLOCK_NNZ,
        T,
    >,
    SparseCholeskyError,
> {
    if BLOCK_GRID_ROWS != BLOCK_GRID_COLS {
        return Err(SparseCholeskyError::Csc(CscError::LengthMismatch));
    }

    let mut present = [[false; BLOCK_GRID_ROWS]; BLOCK_GRID_COLS];
    let mut mapped =
        [[Matrix::<BLOCK_ROWS, BLOCK_COLS, T>::zeros(); BLOCK_GRID_ROWS]; BLOCK_GRID_COLS];
    for column in 0..BLOCK_GRID_COLS {
        let start = matrix.pattern.column_starts()[column];
        let end = matrix.pattern.column_end(column).unwrap_or(matrix.nnz());
        for index in start..end {
            let row = matrix.pattern.row_indices()[index];
            if row < column {
                continue;
            }
            let ordered_row = ordering.inverse()[row];
            let ordered_column = ordering.inverse()[column];
            let (lower_row, lower_column) = if ordered_row >= ordered_column {
                (ordered_row, ordered_column)
            } else {
                (ordered_column, ordered_row)
            };
            if !present[lower_column][lower_row] {
                present[lower_column][lower_row] = true;
                mapped[lower_column][lower_row] = matrix.values()[index];
            }
        }
    }

    let mut values = [Matrix::<BLOCK_ROWS, BLOCK_COLS, T>::zeros(); MAX_BLOCK_NNZ];
    let mut row_indices = [0usize; MAX_BLOCK_NNZ];
    let mut column_starts = [0usize; BLOCK_GRID_COLS];
    let mut position = 0;
    for column in 0..BLOCK_GRID_COLS {
        column_starts[column] = position;
        for row in column..BLOCK_GRID_ROWS {
            if present[column][row] {
                if position == MAX_BLOCK_NNZ {
                    return Err(SparseCholeskyError::CapacityExceeded);
                }
                row_indices[position] = row;
                values[position] = mapped[column][row];
                position += 1;
            }
        }
    }
    let mut pattern = StaticCscPattern::new();
    pattern.row_indices[..position].copy_from_slice(&row_indices[..position]);
    pattern.column_starts = column_starts;
    pattern.nnz = position;
    Ok(StaticBlockCscMatrix { values, pattern })
}

#[allow(clippy::needless_range_loop)]
fn block_diagonal_pivot_ordering<
    const BLOCK_ROWS: usize,
    const BLOCK_COLS: usize,
    const BLOCK_GRID_ROWS: usize,
    const BLOCK_GRID_COLS: usize,
    const MAX_BLOCK_NNZ: usize,
    T: Real,
>(
    matrix: &StaticBlockCscMatrix<
        BLOCK_ROWS,
        BLOCK_COLS,
        BLOCK_GRID_ROWS,
        BLOCK_GRID_COLS,
        MAX_BLOCK_NNZ,
        T,
    >,
    threshold: T,
) -> Result<StaticCscOrdering<BLOCK_GRID_ROWS>, SparseCholeskyError> {
    if BLOCK_ROWS != BLOCK_COLS || BLOCK_GRID_ROWS != BLOCK_GRID_COLS {
        return Err(SparseCholeskyError::Csc(CscError::LengthMismatch));
    }
    if !threshold.is_finite() {
        return Err(SparseCholeskyError::NonFinite);
    }
    let threshold = threshold.abs();
    let mut work =
        [[Matrix::<BLOCK_ROWS, BLOCK_COLS, T>::zeros(); BLOCK_GRID_ROWS]; BLOCK_GRID_COLS];
    for column in 0..BLOCK_GRID_COLS {
        let start = matrix.pattern.column_starts()[column];
        let end = matrix.pattern.column_end(column).unwrap_or(matrix.nnz());
        for index in start..end {
            let row = matrix.pattern.row_indices()[index];
            if row < column {
                continue;
            }
            let block = matrix.values()[index];
            if row == column {
                work[row][column] = block_symmetrize_lower(&block);
            } else {
                work[row][column] = block;
                work[column][row] = block_transpose(&block);
            }
        }
    }

    let mut permutation: [usize; BLOCK_GRID_ROWS] = core::array::from_fn(|index| index);
    for position in 0..BLOCK_GRID_ROWS {
        let mut selected = position;
        let mut selected_magnitude = T::zero();
        for candidate in position..BLOCK_GRID_ROWS {
            let magnitude = block_diagonal_magnitude(&work[candidate][candidate]);
            if !magnitude.is_finite() {
                return Err(SparseCholeskyError::NonFinite);
            }
            if candidate == position || magnitude > selected_magnitude {
                selected = candidate;
                selected_magnitude = magnitude;
            }
        }
        if selected_magnitude <= threshold {
            return Err(SparseCholeskyError::ZeroPivot);
        }
        if selected != position {
            for row in 0..BLOCK_GRID_ROWS {
                work[row].swap(position, selected);
            }
            work.swap(position, selected);
            permutation.swap(position, selected);
        }

        let mut diagonal = work[position][position];
        let pivots = [1; BLOCK_ROWS];
        block_ldlt(&mut diagonal)?;
        work[position][position] = diagonal;
        for row in (position + 1)..BLOCK_GRID_ROWS {
            block_solve_right_ldlt_transpose(&mut work[row][position], &diagonal, &pivots)?;
        }
        for row in (position + 1)..BLOCK_GRID_ROWS {
            for column in (position + 1)..=row {
                let lhs = work[row][position];
                let rhs = work[column][position];
                block_rank_update_ldlt_sub(&mut work[row][column], &lhs, &diagonal, &pivots, &rhs);
            }
        }
    }
    StaticCscOrdering::from_permutation(&permutation).map_err(SparseCholeskyError::from)
}

#[inline]
fn block_diagonal_magnitude<const ROWS: usize, const COLS: usize, T: Real>(
    block: &Matrix<ROWS, COLS, T>,
) -> T {
    let mut magnitude = T::zero();
    for diagonal in 0..ROWS.min(COLS) {
        magnitude = magnitude.max(block[(diagonal, diagonal)].abs());
    }
    magnitude
}

#[inline]
fn block_transpose<const ROWS: usize, const COLS: usize, T: Copy + Zero>(
    block: &Matrix<ROWS, COLS, T>,
) -> Matrix<ROWS, COLS, T> {
    Matrix::from_fn(|row, column| block[(column, row)])
}

#[inline]
fn block_symmetrize_lower<const ROWS: usize, const COLS: usize, T: Copy + Zero>(
    block: &Matrix<ROWS, COLS, T>,
) -> Matrix<ROWS, COLS, T> {
    Matrix::from_fn(|row, column| {
        if row >= column {
            block[(row, column)]
        } else {
            block[(column, row)]
        }
    })
}
