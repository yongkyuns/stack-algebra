//! Fixed-capacity block sparse storage and factorization.
//!
//! Block sparse matrices store dense compile-time-sized blocks in CSC or CSR
//! structure. This is useful when a problem is naturally partitioned into
//! state-variable blocks: the block pattern stays compact while each block
//! retains dense, cache-friendly arithmetic. All storage is inline and bounded
//! by `MAX_BLOCK_NNZ`.
//!
//! ```
//! use stack_algebra::{Matrix, StaticBlockCscMatrix};
//!
//! type Blocks = StaticBlockCscMatrix<1, 1, 2, 2, 3, f64>;
//! let a = Blocks::from_pattern(
//!     &[
//!         Matrix::from_rows([[4.0]]),
//!         Matrix::from_rows([[1.0]]),
//!         Matrix::from_rows([[3.0]]),
//!     ],
//!     &[0, 1, 1],
//!     &[0, 2, 3],
//! ).unwrap();
//! let mut y = [0.0; 2];
//! a.matvec_into(&[1.0, 2.0], &mut y).unwrap();
//! assert_eq!(y, [4.0, 7.0]);
//! ```
//!
//! Native block Cholesky and LDLᵀ require square blocks and square block
//! grids. For rectangular or unsupported block layouts, use
//! [`StaticBlockCscMatrix::to_scalar_csc`] as an explicit bounded adapter.

use crate::{
    CscError, DecompositionError, Ldlt, Matrix, MatrixScalar, Real, SparseCholeskyError,
    StaticCscCholesky, StaticCscMatrix, StaticCscOrdering, StaticCscPattern, Zero,
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
    values: [Matrix<BLOCK_ROWS, BLOCK_COLS, T>; MAX_BLOCK_NNZ],
    pattern: StaticCscPattern<BLOCK_GRID_ROWS, BLOCK_GRID_COLS, MAX_BLOCK_NNZ>,
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
    lower: StaticCscPattern<BLOCK_GRID_ROWS, BLOCK_GRID_COLS, MAX_L_BLOCK_NNZ>,
    ordering: StaticCscOrdering<BLOCK_GRID_ROWS>,
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

/// Native block sparse Cholesky factorization.
///
/// The factor stores dense lower-triangular diagonal blocks and dense
/// off-diagonal blocks. It requires square block and grid dimensions at
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
                        let determinant = d11 * d22 - d12 * d12;
                        let first_result = (first * d22 - second * d12) / determinant;
                        let second_result = (second * d11 - first * d12) / determinant;
                        if !determinant.is_finite()
                            || determinant == T::zero()
                            || !first_result.is_finite()
                            || !second_result.is_finite()
                        {
                            return Err(SparseCholeskyError::ZeroPivot);
                        }
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

fn permute_block_matrix<
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

#[inline]
fn find_block_index<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize>(
    pattern: &StaticCscPattern<ROWS, COLS, MAX_NNZ>,
    row: usize,
    column: usize,
) -> Option<usize> {
    let start = *pattern.column_starts().get(column)?;
    let end = pattern.column_end(column)?;
    pattern.row_indices()[start..end]
        .iter()
        .position(|&candidate| candidate == row)
        .map(|offset| start + offset)
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
fn block_rank_update_ldlt_sub<const ROWS: usize, const COLS: usize, T: Real>(
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

#[inline]
fn block_ldlt<const ROWS: usize, const COLS: usize, T: Real>(
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
fn block_solve_right_ldlt_transpose<const ROWS: usize, const COLS: usize, T: Real>(
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
                let determinant = d11 * d22 - d12 * d12;
                let first_result = (first * d22 - second * d12) / determinant;
                let second_result = (second * d11 - first * d12) / determinant;
                if !determinant.is_finite()
                    || !first_result.is_finite()
                    || !second_result.is_finite()
                {
                    return Err(SparseCholeskyError::NonFinite);
                }
                if determinant == T::zero() {
                    return Err(SparseCholeskyError::ZeroPivot);
                }
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

#[inline]
fn map_ldlt_error(error: DecompositionError) -> SparseCholeskyError {
    match error {
        DecompositionError::NonFinite => SparseCholeskyError::NonFinite,
        DecompositionError::ZeroPivot
        | DecompositionError::Singular
        | DecompositionError::NotPositiveDefinite
        | DecompositionError::NotSymmetric
        | DecompositionError::NoConvergence
        | DecompositionError::InvalidView => SparseCholeskyError::ZeroPivot,
    }
}

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
