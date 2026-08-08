//! Fixed-capacity block-compressed sparse column storage.

use crate::sparse::map_ldlt_error;
use crate::{
    CscError, Ldlt, Matrix, MatrixScalar, Real, SparseCholeskyError, StaticCscCholesky,
    StaticCscMatrix, StaticCscPattern, Zero,
};

/// A fixed-capacity block-compressed sparse column matrix.
///
/// The CSC pattern indexes block rows and block columns. Each stored value is
/// a dense `BLOCK_ROWS x BLOCK_COLS` block, so the scalar dimensions are
/// `BLOCK_GRID_ROWS * BLOCK_ROWS` by `BLOCK_GRID_COLS * BLOCK_COLS`.
///
/// Block CSC uses block-column pointers, so `block_column_pointers` has
/// `BLOCK_GRID_COLS + 1` entries and block row indices must be strictly
/// increasing within each block column. Numeric updates can use
/// [`Self::set_values`] without rebuilding the pattern.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticBlockCscMatrix<
    const BLOCK_ROWS: usize,
    const BLOCK_COLS: usize,
    const BLOCK_GRID_ROWS: usize,
    const BLOCK_GRID_COLS: usize,
    const MAX_BLOCK_NNZ: usize,
    T = f32,
> {
    pub(super) values: [Matrix<BLOCK_ROWS, BLOCK_COLS, T>; MAX_BLOCK_NNZ],
    pub(super) pattern: StaticCscPattern<BLOCK_GRID_ROWS, BLOCK_GRID_COLS, MAX_BLOCK_NNZ>,
}

impl<
        const BLOCK_ROWS: usize,
        const BLOCK_COLS: usize,
        const BLOCK_GRID_ROWS: usize,
        const BLOCK_GRID_COLS: usize,
        const MAX_BLOCK_NNZ: usize,
        T,
    >
    StaticBlockCscMatrix<BLOCK_ROWS, BLOCK_COLS, BLOCK_GRID_ROWS, BLOCK_GRID_COLS, MAX_BLOCK_NNZ, T>
where
    T: Copy + Zero,
{
    /// Creates an empty block matrix.
    #[inline]
    pub fn new() -> Self {
        Self {
            values: [Matrix::zeros(); MAX_BLOCK_NNZ],
            pattern: StaticCscPattern::new(),
        }
    }

    /// Returns the exact inline storage footprint of this block matrix.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Creates a block matrix from canonical block CSC arrays.
    #[inline]
    pub fn from_pattern(
        values: &[Matrix<BLOCK_ROWS, BLOCK_COLS, T>],
        block_row_indices: &[usize],
        block_column_pointers: &[usize],
    ) -> Result<Self, CscError> {
        if values.len() != block_row_indices.len() || values.len() > MAX_BLOCK_NNZ {
            return Err(CscError::LengthMismatch);
        }
        let pattern = StaticCscPattern::from_arrays(block_row_indices, block_column_pointers)?;
        let mut output = Self::new();
        output.values[..values.len()].copy_from_slice(values);
        output.pattern = pattern;
        Ok(output)
    }

    /// Returns the block pattern.
    #[inline]
    pub fn pattern(&self) -> &StaticCscPattern<BLOCK_GRID_ROWS, BLOCK_GRID_COLS, MAX_BLOCK_NNZ> {
        &self.pattern
    }

    /// Returns the active block values.
    #[inline]
    pub fn values(&self) -> &[Matrix<BLOCK_ROWS, BLOCK_COLS, T>] {
        &self.values[..self.nnz()]
    }

    /// Returns mutable active block values.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [Matrix<BLOCK_ROWS, BLOCK_COLS, T>] {
        let nnz = self.nnz();
        &mut self.values[..nnz]
    }

    /// Returns the number of block rows.
    #[inline]
    pub const fn block_rows(&self) -> usize {
        BLOCK_GRID_ROWS
    }

    /// Returns the number of block columns.
    #[inline]
    pub const fn block_cols(&self) -> usize {
        BLOCK_GRID_COLS
    }

    /// Returns the scalar row count.
    #[inline]
    pub const fn rows(&self) -> usize {
        BLOCK_GRID_ROWS * BLOCK_ROWS
    }

    /// Returns the scalar column count.
    #[inline]
    pub const fn cols(&self) -> usize {
        BLOCK_GRID_COLS * BLOCK_COLS
    }

    /// Returns the maximum number of stored blocks.
    #[inline]
    pub const fn capacity(&self) -> usize {
        MAX_BLOCK_NNZ
    }

    /// Returns the number of stored blocks.
    #[inline]
    pub const fn nnz(&self) -> usize {
        self.pattern.nnz()
    }

    /// Returns block row indices in block CSC order.
    #[inline]
    pub fn block_row_indices(&self) -> &[usize] {
        self.pattern.row_indices()
    }

    /// Returns the start pointer for every block column.
    #[inline]
    pub fn block_column_starts(&self) -> &[usize; BLOCK_GRID_COLS] {
        self.pattern.column_starts()
    }

    /// Returns the exclusive end pointer of a block column.
    #[inline]
    pub fn block_column_end(&self, column: usize) -> Option<usize> {
        self.pattern.column_end(column)
    }

    /// Returns a stored block, or `None` when the block is absent or out of bounds.
    #[inline]
    pub fn block(&self, row: usize, column: usize) -> Option<&Matrix<BLOCK_ROWS, BLOCK_COLS, T>> {
        let start = *self.pattern.column_starts().get(column)?;
        let end = self.pattern.column_end(column)?;
        for index in start..end {
            match self.pattern.row_indices()[index].cmp(&row) {
                core::cmp::Ordering::Equal => return Some(&self.values[index]),
                core::cmp::Ordering::Greater => return None,
                core::cmp::Ordering::Less => {}
            }
        }
        None
    }

    /// Replaces all block values while preserving the validated pattern.
    #[inline]
    pub fn set_values(
        &mut self,
        values: &[Matrix<BLOCK_ROWS, BLOCK_COLS, T>],
    ) -> Result<(), CscError> {
        if values.len() != self.nnz() {
            return Err(CscError::LengthMismatch);
        }
        let nnz = self.nnz();
        self.values[..nnz].copy_from_slice(values);
        Ok(())
    }

    /// Computes the block matrix-vector product into caller-provided slices.
    ///
    /// `rhs` and `output` use scalar column-major vector storage. The slices
    /// must have lengths `self.cols()` and `self.rows()` respectively.
    #[inline]
    pub fn matvec_into(&self, rhs: &[T], output: &mut [T]) -> Result<(), CscError>
    where
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
    {
        if rhs.len() != self.cols() || output.len() != self.rows() {
            return Err(CscError::LengthMismatch);
        }
        for value in output.iter_mut() {
            *value = T::zero();
        }
        for block_column in 0..BLOCK_GRID_COLS {
            let column_start = self.pattern.column_starts()[block_column];
            let column_end = self.pattern.column_end(block_column).unwrap_or(self.nnz());
            for index in column_start..column_end {
                let block_row = self.pattern.row_indices()[index];
                let block = &self.values[index];
                for local_column in 0..BLOCK_COLS {
                    let rhs_value = rhs[block_column * BLOCK_COLS + local_column];
                    for local_row in 0..BLOCK_ROWS {
                        let output_index = block_row * BLOCK_ROWS + local_row;
                        output[output_index] =
                            output[output_index] + block[(local_row, local_column)] * rhs_value;
                    }
                }
            }
        }
        Ok(())
    }

    /// Expands this block CSC matrix into a scalar CSC matrix in fixed storage.
    ///
    /// The expansion preserves canonical column-major ordering and performs
    /// no heap allocation. `SCALAR_ROWS`, `SCALAR_COLS`, and
    /// `MAX_SCALAR_NNZ` make the resulting footprint explicit to embedded
    /// callers. This remains an interoperability path; native block
    /// factorization avoids expansion when the block dimensions are supported.
    #[inline]
    pub fn to_scalar_csc<
        const SCALAR_ROWS: usize,
        const SCALAR_COLS: usize,
        const MAX_SCALAR_NNZ: usize,
    >(
        &self,
    ) -> Result<StaticCscMatrix<SCALAR_ROWS, SCALAR_COLS, MAX_SCALAR_NNZ, T>, CscError> {
        if SCALAR_ROWS != BLOCK_GRID_ROWS * BLOCK_ROWS
            || SCALAR_COLS != BLOCK_GRID_COLS * BLOCK_COLS
        {
            return Err(CscError::LengthMismatch);
        }

        let mut values = [T::zero(); MAX_SCALAR_NNZ];
        let mut row_indices = [0usize; MAX_SCALAR_NNZ];
        let mut column_starts = [0usize; SCALAR_COLS];
        let mut nnz = 0;
        for block_column in 0..BLOCK_GRID_COLS {
            let block_start = self.pattern.column_starts()[block_column];
            let block_end = self.pattern.column_end(block_column).unwrap_or(self.nnz());
            for local_column in 0..BLOCK_COLS {
                let scalar_column = block_column * BLOCK_COLS + local_column;
                column_starts[scalar_column] = nnz;
                for index in block_start..block_end {
                    let block_row = self.pattern.row_indices()[index];
                    let block = &self.values[index];
                    for local_row in 0..BLOCK_ROWS {
                        if nnz == MAX_SCALAR_NNZ {
                            return Err(CscError::CapacityExceeded);
                        }
                        row_indices[nnz] = block_row * BLOCK_ROWS + local_row;
                        values[nnz] = block[(local_row, local_column)];
                        nnz += 1;
                    }
                }
            }
        }
        let pattern = StaticCscPattern {
            row_indices,
            column_starts,
            nnz,
        };
        Ok(StaticCscMatrix { values, pattern })
    }

    /// Computes a scalar-expanded sparse Cholesky factorization.
    ///
    /// This is a bounded interoperability adapter around
    /// [`StaticCscCholesky`]. It is allocation-free, but currently factors the
    /// expanded scalar CSC representation rather than using block kernels.
    #[inline]
    pub fn cholesky<
        const SCALAR_DIM: usize,
        const MAX_SCALAR_A_NNZ: usize,
        const MAX_L_NNZ: usize,
    >(
        &self,
    ) -> Result<StaticCscCholesky<SCALAR_DIM, MAX_L_NNZ, T>, SparseCholeskyError>
    where
        T: Real,
    {
        let scalar = self.to_scalar_csc::<SCALAR_DIM, SCALAR_DIM, MAX_SCALAR_A_NNZ>()?;
        StaticCscCholesky::decompose(&scalar)
    }

    /// Computes a fixed-size dense LDLᵀ factorization with global scalar
    /// Bunch–Kaufman pivoting.
    ///
    /// This is an explicit cross-block pivot fallback: the symmetric lower
    /// block structure is expanded into caller-selected stack storage and
    /// factored by [`Ldlt`]. Native block LDLᵀ remains preferable when pivots
    /// stay within dense diagonal blocks.
    #[inline]
    pub fn try_dense_ldlt<const SCALAR_DIM: usize>(
        &self,
    ) -> Result<Ldlt<SCALAR_DIM, T>, SparseCholeskyError>
    where
        T: Real + MatrixScalar,
    {
        if BLOCK_ROWS != BLOCK_COLS
            || BLOCK_GRID_ROWS != BLOCK_GRID_COLS
            || SCALAR_DIM != BLOCK_ROWS * BLOCK_GRID_ROWS
        {
            return Err(SparseCholeskyError::Csc(CscError::LengthMismatch));
        }

        let mut dense = Matrix::<SCALAR_DIM, SCALAR_DIM, T>::zeros();
        for block_column in 0..BLOCK_GRID_COLS {
            let start = self.pattern.column_starts()[block_column];
            let end = self.pattern.column_end(block_column).unwrap_or(self.nnz());
            for index in start..end {
                let block_row = self.pattern.row_indices()[index];
                if block_row < block_column {
                    continue;
                }
                let block = &self.values[index];
                if block_row == block_column {
                    for local_column in 0..BLOCK_COLS {
                        for local_row in local_column..BLOCK_ROWS {
                            let row = block_row * BLOCK_ROWS + local_row;
                            let column = block_column * BLOCK_COLS + local_column;
                            let value = block[(local_row, local_column)];
                            dense[(row, column)] = value;
                            dense[(column, row)] = value;
                        }
                    }
                } else {
                    for local_column in 0..BLOCK_COLS {
                        for local_row in 0..BLOCK_ROWS {
                            let row = block_row * BLOCK_ROWS + local_row;
                            let column = block_column * BLOCK_COLS + local_column;
                            let value = block[(local_row, local_column)];
                            dense[(row, column)] = value;
                            dense[(column, row)] = value;
                        }
                    }
                }
            }
        }
        Ldlt::try_decompose(&dense).map_err(map_ldlt_error)
    }
}
impl<
        const BLOCK_ROWS: usize,
        const BLOCK_COLS: usize,
        const BLOCK_GRID_ROWS: usize,
        const BLOCK_GRID_COLS: usize,
        const MAX_BLOCK_NNZ: usize,
        T,
    > Default
    for StaticBlockCscMatrix<
        BLOCK_ROWS,
        BLOCK_COLS,
        BLOCK_GRID_ROWS,
        BLOCK_GRID_COLS,
        MAX_BLOCK_NNZ,
        T,
    >
where
    T: Copy + Zero,
{
    fn default() -> Self {
        Self::new()
    }
}
