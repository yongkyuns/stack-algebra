//! Fixed-capacity block-compressed sparse row storage.

use crate::{CscError, Matrix, Zero};

/// A fixed-capacity block-compressed sparse row matrix.
///
/// The CSR pattern indexes block rows and block columns. Each stored value is
/// a dense `BLOCK_ROWS x BLOCK_COLS` block, so the scalar dimensions are
/// `BLOCK_GRID_ROWS * BLOCK_ROWS` by `BLOCK_GRID_COLS * BLOCK_COLS`.
///
/// CSR is convenient for row-oriented products. Its row pointers have
/// `BLOCK_GRID_ROWS + 1` entries and block column indices are strictly
/// increasing within each block row. The storage is still fixed-capacity and
/// allocation-free; use CSC when sparse factorization is required.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticBlockCsrMatrix<
    const BLOCK_ROWS: usize,
    const BLOCK_COLS: usize,
    const BLOCK_GRID_ROWS: usize,
    const BLOCK_GRID_COLS: usize,
    const MAX_BLOCK_NNZ: usize,
    T = f32,
> {
    values: [Matrix<BLOCK_ROWS, BLOCK_COLS, T>; MAX_BLOCK_NNZ],
    block_column_indices: [usize; MAX_BLOCK_NNZ],
    block_row_starts: [usize; BLOCK_GRID_ROWS],
    nnz: usize,
}

impl<
        const BLOCK_ROWS: usize,
        const BLOCK_COLS: usize,
        const BLOCK_GRID_ROWS: usize,
        const BLOCK_GRID_COLS: usize,
        const MAX_BLOCK_NNZ: usize,
        T,
    >
    StaticBlockCsrMatrix<BLOCK_ROWS, BLOCK_COLS, BLOCK_GRID_ROWS, BLOCK_GRID_COLS, MAX_BLOCK_NNZ, T>
where
    T: Copy + Zero,
{
    /// Creates an empty block matrix.
    #[inline]
    pub fn new() -> Self {
        Self {
            values: [Matrix::zeros(); MAX_BLOCK_NNZ],
            block_column_indices: [0; MAX_BLOCK_NNZ],
            block_row_starts: [0; BLOCK_GRID_ROWS],
            nnz: 0,
        }
    }

    /// Returns the exact inline storage footprint of this block matrix.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Creates a block matrix from canonical CSR arrays.
    #[inline]
    pub fn from_pattern(
        values: &[Matrix<BLOCK_ROWS, BLOCK_COLS, T>],
        block_column_indices: &[usize],
        block_row_pointers: &[usize],
    ) -> Result<Self, CscError> {
        if values.len() != block_column_indices.len()
            || values.len() > MAX_BLOCK_NNZ
            || block_row_pointers.len() != BLOCK_GRID_ROWS + 1
        {
            return Err(CscError::LengthMismatch);
        }
        if block_row_pointers[0] != 0 {
            return Err(CscError::InvalidColumnPointers);
        }
        let nnz = values.len();
        let mut previous = 0;
        for &pointer in block_row_pointers {
            if pointer < previous || pointer > nnz {
                return Err(CscError::InvalidColumnPointers);
            }
            previous = pointer;
        }
        if block_row_pointers[BLOCK_GRID_ROWS] != nnz {
            return Err(CscError::InvalidColumnPointers);
        }
        for row in 0..BLOCK_GRID_ROWS {
            let start = block_row_pointers[row];
            let end = block_row_pointers[row + 1];
            let mut previous_column = None;
            for &column in &block_column_indices[start..end] {
                if column >= BLOCK_GRID_COLS
                    || previous_column.is_some_and(|previous| column <= previous)
                {
                    return Err(CscError::InvalidRowIndices);
                }
                previous_column = Some(column);
            }
        }
        let mut output = Self::new();
        output.values[..nnz].copy_from_slice(values);
        output.block_column_indices[..nnz].copy_from_slice(block_column_indices);
        output
            .block_row_starts
            .copy_from_slice(&block_row_pointers[..BLOCK_GRID_ROWS]);
        output.nnz = nnz;
        Ok(output)
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
        self.nnz
    }

    /// Returns the active block values in CSR order.
    #[inline]
    pub fn values(&self) -> &[Matrix<BLOCK_ROWS, BLOCK_COLS, T>] {
        &self.values[..self.nnz]
    }

    /// Returns mutable active block values.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [Matrix<BLOCK_ROWS, BLOCK_COLS, T>] {
        &mut self.values[..self.nnz]
    }

    /// Returns block column indices in block CSR order.
    #[inline]
    pub fn block_column_indices(&self) -> &[usize] {
        &self.block_column_indices[..self.nnz]
    }

    /// Returns the start pointer for every block row.
    #[inline]
    pub fn block_row_starts(&self) -> &[usize; BLOCK_GRID_ROWS] {
        &self.block_row_starts
    }

    /// Returns the exclusive end pointer of a block row.
    #[inline]
    pub fn block_row_end(&self, row: usize) -> Option<usize> {
        if row >= BLOCK_GRID_ROWS {
            return None;
        }
        Some(if row + 1 < BLOCK_GRID_ROWS {
            self.block_row_starts[row + 1]
        } else {
            self.nnz
        })
    }

    /// Returns a stored block, or `None` when the block is absent or out of bounds.
    #[inline]
    pub fn block(&self, row: usize, column: usize) -> Option<&Matrix<BLOCK_ROWS, BLOCK_COLS, T>> {
        let start = *self.block_row_starts.get(row)?;
        let end = self.block_row_end(row)?;
        for index in start..end {
            match self.block_column_indices[index].cmp(&column) {
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
        if values.len() != self.nnz {
            return Err(CscError::LengthMismatch);
        }
        self.values[..self.nnz].copy_from_slice(values);
        Ok(())
    }

    /// Computes the block matrix-vector product into caller-provided slices.
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
        for block_row in 0..BLOCK_GRID_ROWS {
            let row_start = self.block_row_starts[block_row];
            let row_end = self.block_row_end(block_row).unwrap_or(self.nnz);
            for index in row_start..row_end {
                let block_column = self.block_column_indices[index];
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
}

impl<
        const BLOCK_ROWS: usize,
        const BLOCK_COLS: usize,
        const BLOCK_GRID_ROWS: usize,
        const BLOCK_GRID_COLS: usize,
        const MAX_BLOCK_NNZ: usize,
        T,
    > Default
    for StaticBlockCsrMatrix<
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
