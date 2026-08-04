use core::ops::{Add, Mul};

use crate::{Matrix, Real, Zero};

mod errors;

pub use errors::{CscError, SparseCholeskyError};

/// A fixed-capacity symmetric permutation for sparse factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticCscOrdering<const N: usize> {
    permutation: [usize; N],
    inverse: [usize; N],
}

impl<const N: usize> StaticCscOrdering<N> {
    /// Returns the identity ordering.
    #[inline]
    pub const fn identity() -> Self {
        let mut permutation = [0; N];
        let mut index = 0;
        while index < N {
            permutation[index] = index;
            index += 1;
        }
        Self {
            permutation,
            inverse: permutation,
        }
    }

    /// Validates an ordering given as ordered-index to original-index maps.
    #[inline]
    pub fn from_permutation(permutation: &[usize]) -> Result<Self, CscError> {
        if permutation.len() != N {
            return Err(CscError::LengthMismatch);
        }
        let mut output = Self {
            permutation: [0; N],
            inverse: [0; N],
        };
        let mut seen = [false; N];
        for (ordered, &original) in permutation.iter().enumerate() {
            if original >= N || seen[original] {
                return Err(CscError::InvalidPermutation);
            }
            seen[original] = true;
            output.permutation[ordered] = original;
            output.inverse[original] = ordered;
        }
        Ok(output)
    }

    /// Computes a deterministic fixed-workspace minimum-degree ordering.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub fn minimum_degree<const MAX_NNZ: usize, T: Copy + Zero>(
        matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
    ) -> Self {
        let mut adjacency = [[false; N]; N];
        for column in 0..N {
            let start = matrix.column_starts()[column];
            let end = matrix.column_end(column).unwrap_or(matrix.nnz());
            for index in start..end {
                let row = matrix.row_indices()[index];
                if row != column {
                    adjacency[row][column] = true;
                    adjacency[column][row] = true;
                }
            }
        }

        let mut eliminated = [false; N];
        let mut permutation = [0; N];
        for slot in permutation.iter_mut() {
            let mut selected = 0;
            let mut selected_degree = usize::MAX;
            for candidate in 0..N {
                if !eliminated[candidate] {
                    let degree = (0..N)
                        .filter(|&neighbor| !eliminated[neighbor] && adjacency[candidate][neighbor])
                        .count();
                    if degree < selected_degree {
                        selected = candidate;
                        selected_degree = degree;
                    }
                }
            }
            *slot = selected;
            eliminated[selected] = true;

            let mut neighbors = [0; N];
            let mut neighbor_count = 0;
            for neighbor in 0..N {
                if !eliminated[neighbor] && adjacency[selected][neighbor] {
                    neighbors[neighbor_count] = neighbor;
                    neighbor_count += 1;
                }
            }
            for left in 0..neighbor_count {
                for right in (left + 1)..neighbor_count {
                    let first = neighbors[left];
                    let second = neighbors[right];
                    adjacency[first][second] = true;
                    adjacency[second][first] = true;
                }
            }
        }
        Self::from_permutation(&permutation).expect("minimum-degree ordering is a permutation")
    }

    /// Returns the ordered-index to original-index map.
    #[inline]
    pub const fn permutation(&self) -> &[usize; N] {
        &self.permutation
    }

    /// Returns the original-index to ordered-index map.
    #[inline]
    pub const fn inverse(&self) -> &[usize; N] {
        &self.inverse
    }

    /// Applies this symmetric ordering and returns a lower-triangular CSC
    /// matrix in ordered coordinates.
    #[inline]
    pub fn permute<const MAX_NNZ: usize, T: Copy + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
    ) -> Result<StaticCscMatrix<N, N, MAX_NNZ, T>, CscError> {
        permute_matrix(matrix, *self)
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.permutation == Self::identity().permutation
    }
}

/// An immutable fixed-capacity CSC sparsity pattern.
///
/// Keeping the symbolic structure separate allows generated code to reuse a
/// validated pattern while replacing numeric values at every iteration.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticCscPattern<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize> {
    row_indices: [usize; MAX_NNZ],
    column_starts: [usize; COLS],
    nnz: usize,
}

impl<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize>
    StaticCscPattern<ROWS, COLS, MAX_NNZ>
{
    /// Creates an empty pattern.
    #[inline]
    pub const fn new() -> Self {
        Self {
            row_indices: [0; MAX_NNZ],
            column_starts: [0; COLS],
            nnz: 0,
        }
    }

    /// Creates a pattern from canonical CSC row and column-pointer arrays.
    #[inline]
    pub fn from_arrays(row_indices: &[usize], column_pointers: &[usize]) -> Result<Self, CscError> {
        if row_indices.len() > MAX_NNZ || column_pointers.len() != COLS + 1 {
            return Err(CscError::LengthMismatch);
        }
        if column_pointers[0] != 0 {
            return Err(CscError::InvalidColumnPointers);
        }
        let nnz = row_indices.len();
        let mut previous = 0;
        for &pointer in column_pointers {
            if pointer < previous || pointer > nnz {
                return Err(CscError::InvalidColumnPointers);
            }
            previous = pointer;
        }
        if column_pointers[COLS] != nnz {
            return Err(CscError::InvalidColumnPointers);
        }
        for column in 0..COLS {
            let start = column_pointers[column];
            let end = column_pointers[column + 1];
            let mut previous_row = None;
            for &row in &row_indices[start..end] {
                if row >= ROWS || previous_row.is_some_and(|previous| row <= previous) {
                    return Err(CscError::InvalidRowIndices);
                }
                previous_row = Some(row);
            }
        }
        let mut pattern = Self::new();
        pattern.row_indices[..nnz].copy_from_slice(row_indices);
        pattern
            .column_starts
            .copy_from_slice(&column_pointers[..COLS]);
        pattern.nnz = nnz;
        Ok(pattern)
    }

    /// Returns the number of stored entries.
    #[inline]
    pub const fn nnz(&self) -> usize {
        self.nnz
    }

    /// Returns the active row indices.
    #[inline]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices[..self.nnz]
    }

    /// Returns the start pointer for each column.
    #[inline]
    pub fn column_starts(&self) -> &[usize; COLS] {
        &self.column_starts
    }

    /// Returns the exclusive end pointer of a column.
    #[inline]
    pub fn column_end(&self, column: usize) -> Option<usize> {
        if column >= COLS {
            return None;
        }
        Some(if column + 1 < COLS {
            self.column_starts[column + 1]
        } else {
            self.nnz
        })
    }
}

impl<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize> Default
    for StaticCscPattern<ROWS, COLS, MAX_NNZ>
{
    fn default() -> Self {
        Self::new()
    }
}

/// A fixed-capacity compressed sparse column (CSC) matrix.
///
/// The matrix owns its storage and never allocates.  The sparsity pattern is
/// stored in canonical CSC form: column starts are stored for each of the
/// `COLS` columns and the final column end is `nnz()`. Row indices are
/// strictly increasing within each column. `MAX_NNZ`
/// bounds the number of stored entries; unused capacity remains in the
/// backing arrays but is not exposed through the active slices.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticCscMatrix<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize, T = f32> {
    values: [T; MAX_NNZ],
    pattern: StaticCscPattern<ROWS, COLS, MAX_NNZ>,
}

impl<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize, T>
    StaticCscMatrix<ROWS, COLS, MAX_NNZ, T>
where
    T: Copy + Zero,
{
    /// Creates an empty matrix with the requested compile-time dimensions.
    #[inline]
    pub fn new() -> Self {
        Self {
            values: [T::zero(); MAX_NNZ],
            pattern: StaticCscPattern::new(),
        }
    }

    /// Creates a matrix from canonical CSC arrays.
    #[inline]
    pub fn from_pattern(
        values: &[T],
        row_indices: &[usize],
        column_pointers: &[usize],
    ) -> Result<Self, CscError> {
        if values.len() != row_indices.len() || values.len() > MAX_NNZ {
            return Err(CscError::LengthMismatch);
        }
        let pattern = StaticCscPattern::from_arrays(row_indices, column_pointers)?;
        let nnz = values.len();
        let mut matrix = Self::new();
        matrix.values[..nnz].copy_from_slice(values);
        matrix.pattern = pattern;
        Ok(matrix)
    }

    /// Creates a matrix from a validated pattern and numeric values.
    #[inline]
    pub fn with_pattern(
        pattern: StaticCscPattern<ROWS, COLS, MAX_NNZ>,
        values: &[T],
    ) -> Result<Self, CscError> {
        if values.len() != pattern.nnz() {
            return Err(CscError::LengthMismatch);
        }
        let mut matrix = Self::new();
        matrix.values[..values.len()].copy_from_slice(values);
        matrix.pattern = pattern;
        Ok(matrix)
    }

    /// Returns the validated symbolic sparsity pattern.
    #[inline]
    pub fn pattern(&self) -> &StaticCscPattern<ROWS, COLS, MAX_NNZ> {
        &self.pattern
    }

    /// Returns the number of rows.
    #[inline]
    pub const fn rows(&self) -> usize {
        ROWS
    }

    /// Returns the number of columns.
    #[inline]
    pub const fn cols(&self) -> usize {
        COLS
    }

    /// Returns the compile-time nonzero capacity.
    #[inline]
    pub const fn capacity(&self) -> usize {
        MAX_NNZ
    }

    /// Returns the number of stored entries.
    #[inline]
    pub const fn nnz(&self) -> usize {
        self.pattern.nnz()
    }

    /// Returns the active values in column-major CSC order.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values[..self.nnz()]
    }

    /// Returns the active row indices in column-major CSC order.
    #[inline]
    pub fn row_indices(&self) -> &[usize] {
        self.pattern.row_indices()
    }

    /// Returns the start pointer for each column.
    ///
    /// The end pointer for column `j` is `column_start(j + 1)`, or `nnz()`
    /// for the final column. This representation avoids requiring unstable
    /// generic-const arithmetic for a `[usize; COLS + 1]` field.
    #[inline]
    pub fn column_starts(&self) -> &[usize; COLS] {
        self.pattern.column_starts()
    }

    /// Returns the exclusive end pointer of a column.
    #[inline]
    pub fn column_end(&self, column: usize) -> Option<usize> {
        self.pattern.column_end(column)
    }

    /// Alias for [`Self::row_indices`] using Eigen's inner-index terminology.
    #[inline]
    pub fn inner_indices(&self) -> &[usize] {
        self.row_indices()
    }

    /// Alias for [`Self::column_starts`] using Eigen's outer-index terminology.
    #[inline]
    pub fn outer_starts(&self) -> &[usize; COLS] {
        self.column_starts()
    }

    /// Returns a mutable view of the active values while preserving the pattern.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        let nnz = self.pattern.nnz();
        &mut self.values[..nnz]
    }

    /// Replaces all stored values without changing the sparsity pattern.
    #[inline]
    pub fn set_values(&mut self, values: &[T]) -> Result<(), CscError> {
        if values.len() != self.nnz() {
            return Err(CscError::LengthMismatch);
        }
        self.values[..self.pattern.nnz()].copy_from_slice(values);
        Ok(())
    }

    /// Clears all entries and resets the sparsity pattern.
    #[inline]
    pub fn clear(&mut self) {
        self.pattern = StaticCscPattern::new();
    }

    /// Returns the value at `(row, column)`, if the entry is present.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        let index = self.find_entry(row, column).ok()??;
        Some(&self.values[index])
    }

    /// Sets the value of an existing entry without changing the pattern.
    #[inline]
    pub fn set_value(&mut self, row: usize, column: usize, value: T) -> Result<(), CscError> {
        let index = self
            .find_entry(row, column)?
            .ok_or(CscError::EntryNotFound)?;
        self.values[index] = value;
        Ok(())
    }

    /// Inserts a new entry or overwrites an existing one.
    #[inline]
    pub fn insert(&mut self, row: usize, column: usize, value: T) -> Result<(), CscError> {
        if row >= ROWS || column >= COLS {
            return Err(CscError::IndexOutOfBounds);
        }
        let start = self.pattern.column_starts()[column];
        let end = self.column_end(column).unwrap_or(self.nnz());
        let mut position = end;
        for index in start..end {
            let existing = self.pattern.row_indices[index];
            if existing == row {
                self.values[index] = value;
                return Ok(());
            }
            if existing > row {
                position = index;
                break;
            }
        }
        if self.nnz() == MAX_NNZ {
            return Err(CscError::CapacityExceeded);
        }

        for index in (position..self.nnz()).rev() {
            self.values[index + 1] = self.values[index];
            self.pattern.row_indices[index + 1] = self.pattern.row_indices[index];
        }
        self.values[position] = value;
        self.pattern.row_indices[position] = row;
        self.pattern.nnz += 1;
        for pointer in &mut self.pattern.column_starts[(column + 1)..] {
            *pointer += 1;
        }
        Ok(())
    }

    /// Computes `self * vector` into caller-provided fixed-size storage.
    #[inline]
    pub fn matvec_into(&self, vector: &Matrix<COLS, 1, T>, output: &mut Matrix<ROWS, 1, T>)
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        for row in output.iter_mut() {
            *row = T::zero();
        }
        for column in 0..COLS {
            let value = vector[column];
            for index in
                self.pattern.column_starts[column]..self.column_end(column).unwrap_or(self.nnz())
            {
                let row = self.pattern.row_indices[index];
                output[row] = output[row] + self.values[index] * value;
            }
        }
    }

    /// Computes `self * vector` and returns an owning fixed-size vector.
    #[inline]
    pub fn matvec(&self, vector: &Matrix<COLS, 1, T>) -> Matrix<ROWS, 1, T>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        let mut output = Matrix::<ROWS, 1, T>::zeros();
        self.matvec_into(vector, &mut output);
        output
    }

    fn find_entry(&self, row: usize, column: usize) -> Result<Option<usize>, CscError> {
        if row >= ROWS || column >= COLS {
            return Err(CscError::IndexOutOfBounds);
        }
        let start = self.pattern.column_starts()[column];
        let end = self.column_end(column).unwrap_or(self.nnz());
        for index in start..end {
            match self.pattern.row_indices[index].cmp(&row) {
                core::cmp::Ordering::Equal => return Ok(Some(index)),
                core::cmp::Ordering::Greater => break,
                core::cmp::Ordering::Less => {}
            }
        }
        Ok(None)
    }
}

/// Symbolic pattern for a fixed-capacity simplicial sparse Cholesky factor.
///
/// The pattern includes the lower-triangular fill generated by the input
/// structure. It can be retained and reused while numeric values change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticCscCholeskyPattern<const N: usize, const MAX_L_NNZ: usize> {
    lower: StaticCscPattern<N, N, MAX_L_NNZ>,
    ordering: StaticCscOrdering<N>,
    update_starts: [usize; N],
    update_indices: [usize; MAX_L_NNZ],
    update_columns: [usize; MAX_L_NNZ],
    update_nnz: usize,
}

/// Symbolic CSC pattern shared by sparse LLT and no-pivot LDLᵀ.
pub type StaticCscLdltPattern<const N: usize, const MAX_L_NNZ: usize> =
    StaticCscCholeskyPattern<N, MAX_L_NNZ>;

impl<const N: usize, const MAX_L_NNZ: usize> StaticCscCholeskyPattern<N, MAX_L_NNZ> {
    fn from_lower(lower: StaticCscPattern<N, N, MAX_L_NNZ>) -> Self {
        Self::from_lower_with_ordering(lower, StaticCscOrdering::identity())
    }

    fn from_lower_with_ordering(
        lower: StaticCscPattern<N, N, MAX_L_NNZ>,
        ordering: StaticCscOrdering<N>,
    ) -> Self {
        let mut update_counts = [0usize; N];
        for column in 0..N {
            let start = lower.column_starts()[column];
            let end = lower.column_end(column).unwrap_or(lower.nnz());
            for index in (start + 1)..end {
                update_counts[lower.row_indices()[index]] += 1;
            }
        }

        let mut update_starts = [0usize; N];
        let mut update_nnz = 0;
        for row in 0..N {
            update_starts[row] = update_nnz;
            update_nnz += update_counts[row];
        }
        let mut update_indices = [0usize; MAX_L_NNZ];
        let mut update_columns = [0usize; MAX_L_NNZ];
        let mut update_cursor = update_starts;
        for column in 0..N {
            let start = lower.column_starts()[column];
            let end = lower.column_end(column).unwrap_or(lower.nnz());
            for index in (start + 1)..end {
                let row = lower.row_indices()[index];
                let position = update_cursor[row];
                update_indices[position] = index;
                update_columns[position] = column;
                update_cursor[row] += 1;
            }
        }

        Self {
            lower,
            ordering,
            update_starts,
            update_indices,
            update_columns,
            update_nnz,
        }
    }

    /// Analyzes the lower-triangular structure of a symmetric CSC matrix.
    ///
    /// The analysis is conservative: entries that may cancel numerically are
    /// still retained in the factor pattern, which makes the pattern reusable
    /// across iterations with different values.
    #[inline]
    pub fn analyze<const MAX_A_NNZ: usize, T: Real + Copy + Zero>(
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<Self, SparseCholeskyError> {
        validate_symmetric_structure(matrix)?;
        let mut structure = StaticCscMatrix::<N, N, MAX_L_NNZ, u8>::new();
        for column in 0..N {
            let mut reachable = [false; N];
            for (row, reachable_row) in reachable.iter_mut().enumerate().skip(column) {
                *reachable_row = matrix.get(row, column).is_some();
            }
            reachable[column] = true;

            // A previous column contributes fill to this column when its
            // factor column contains the current row.
            for previous in 0..column {
                if structure.get(column, previous).is_some() {
                    let start = structure.column_starts()[previous];
                    let end = structure.column_end(previous).unwrap_or(structure.nnz());
                    for index in start..end {
                        let row = structure.row_indices()[index];
                        if row >= column {
                            reachable[row] = true;
                        }
                    }
                }
            }

            for (row, &is_reachable) in reachable.iter().enumerate().skip(column) {
                if is_reachable {
                    structure.insert(row, column, 1)?;
                }
            }
        }

        Ok(Self::from_lower(*structure.pattern()))
    }

    /// Analyzes the matrix after applying a symmetric fixed-size ordering.
    #[inline]
    pub fn analyze_with_ordering<const MAX_A_NNZ: usize, T: Real + Copy + Zero>(
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        ordering: StaticCscOrdering<N>,
    ) -> Result<Self, SparseCholeskyError> {
        validate_symmetric_structure(matrix)?;
        if ordering.is_identity() {
            return Self::analyze(matrix);
        }
        let permuted = ordering.permute(matrix)?;
        let natural = Self::analyze(&permuted)?;
        Ok(Self::from_lower_with_ordering(natural.lower, ordering))
    }

    /// Analyzes a matrix using fixed-workspace diagonal pivot selection.
    ///
    /// The returned pattern captures the numeric permutation selected from
    /// this matrix. A nonzero absolute pivot must exceed `threshold`.
    #[inline]
    pub fn analyze_with_diagonal_pivoting<const MAX_A_NNZ: usize, T: Real + Copy + Zero>(
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        threshold: T,
    ) -> Result<Self, SparseCholeskyError> {
        validate_symmetric_structure(matrix)?;
        let ordering = diagonal_pivot_ordering(matrix, threshold)?;
        Self::analyze_with_ordering(matrix, ordering)
    }

    /// Returns the analyzed lower-factor pattern.
    #[inline]
    pub fn lower(&self) -> &StaticCscPattern<N, N, MAX_L_NNZ> {
        &self.lower
    }

    /// Returns the ordering used by this symbolic factorization.
    #[inline]
    pub const fn ordering(&self) -> StaticCscOrdering<N> {
        self.ordering
    }

    /// Validates and transforms a symmetric matrix into reusable ordered
    /// lower-triangular coordinates.
    #[inline]
    pub fn prepare_ordered<const MAX_A_NNZ: usize, T: Real + Copy + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<StaticCscMatrix<N, N, MAX_A_NNZ, T>, SparseCholeskyError> {
        validate_symmetric_structure(matrix)?;
        let ordered = self.ordering.permute(matrix)?;
        self.validate_factor_pattern(&ordered)?;
        Ok(ordered)
    }

    /// Computes numeric values using this symbolic pattern.
    ///
    /// The input is interpreted using its lower triangle, as in Eigen's
    /// `UpLo=Lower` sparse LLT mode. A matching upper-triangle entry may also
    /// be present; it is validated and ignored for the numeric update.
    #[inline]
    pub fn factor<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<StaticCscCholesky<N, MAX_L_NNZ, T>, SparseCholeskyError> {
        let mut factor = StaticCscCholesky {
            lower: StaticCscMatrix {
                values: [T::zero(); MAX_L_NNZ],
                pattern: self.lower,
            },
            ordering: StaticCscOrdering::identity(),
        };
        self.factor_into(matrix, &mut factor)?;
        Ok(factor)
    }

    /// Computes numeric values into reusable factor storage.
    #[inline]
    pub fn factor_into<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        output: &mut StaticCscCholesky<N, MAX_L_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        validate_symmetric_structure(matrix)?;
        if !self.ordering.is_identity() {
            let permuted = self.ordering.permute(matrix)?;
            self.factor_natural_into(&permuted, true, output)?;
            output.ordering = self.ordering;
            return Ok(());
        }
        self.factor_natural_into(matrix, true, output)
    }

    /// Refactors an analyzed matrix into reusable storage without repeating
    /// symmetry or pattern validation. The matrix must retain its analyzed
    /// sparsity pattern; use [`Self::factor_into`] when that cannot be
    /// guaranteed.
    #[inline]
    pub fn factor_reuse_into<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        output: &mut StaticCscCholesky<N, MAX_L_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        if !self.ordering.is_identity() {
            let permuted = self.ordering.permute(matrix)?;
            self.factor_natural_into(&permuted, false, output)?;
            output.ordering = self.ordering;
            return Ok(());
        }
        self.factor_natural_into(matrix, false, output)
    }

    /// Factors a matrix that has already been transformed into this pattern's
    /// ordered coordinates. This avoids repeating the symmetric permutation
    /// and structural validation when numeric values are updated under a
    /// reused symbolic pattern. The matrix must retain the analyzed ordered
    /// sparsity pattern; use [`Self::factor`] when that cannot be guaranteed.
    #[inline]
    pub fn factor_ordered<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<StaticCscCholesky<N, MAX_L_NNZ, T>, SparseCholeskyError> {
        let mut factor = StaticCscCholesky {
            lower: StaticCscMatrix {
                values: [T::zero(); MAX_L_NNZ],
                pattern: self.lower,
            },
            ordering: self.ordering,
        };
        self.factor_ordered_into(matrix, &mut factor)?;
        Ok(factor)
    }

    /// Factors ordered coordinates into reusable factor storage.
    #[inline]
    pub fn factor_ordered_into<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        output: &mut StaticCscCholesky<N, MAX_L_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        self.factor_natural_into(matrix, false, output)?;
        output.ordering = self.ordering;
        Ok(())
    }

    /// Computes a fixed-capacity sparse LDLᵀ factorization using this pattern.
    #[inline]
    pub fn factor_ldlt<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<StaticCscLdlt<N, MAX_L_NNZ, T>, SparseCholeskyError> {
        let mut factor = StaticCscLdlt {
            lower: StaticCscMatrix {
                values: [T::zero(); MAX_L_NNZ],
                pattern: self.lower,
            },
            diagonal: [T::zero(); N],
            ordering: StaticCscOrdering::identity(),
        };
        self.factor_ldlt_into(matrix, &mut factor)?;
        Ok(factor)
    }

    /// Computes sparse LDLᵀ values into reusable factor storage.
    #[inline]
    pub fn factor_ldlt_into<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        output: &mut StaticCscLdlt<N, MAX_L_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        validate_symmetric_structure(matrix)?;
        if !self.ordering.is_identity() {
            let permuted = self.ordering.permute(matrix)?;
            self.factor_ldlt_natural_into(&permuted, true, output)?;
            output.ordering = self.ordering;
            return Ok(());
        }
        self.factor_ldlt_natural_into(matrix, true, output)
    }

    /// Refactors sparse LDLᵀ values without repeating symmetry or pattern
    /// validation. The matrix must retain its analyzed sparsity pattern.
    #[inline]
    pub fn factor_ldlt_reuse_into<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        output: &mut StaticCscLdlt<N, MAX_L_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        if !self.ordering.is_identity() {
            let permuted = self.ordering.permute(matrix)?;
            self.factor_ldlt_natural_into(&permuted, false, output)?;
            output.ordering = self.ordering;
            return Ok(());
        }
        self.factor_ldlt_natural_into(matrix, false, output)
    }

    /// Factors already ordered lower-triangular coordinates with LDLᵀ.
    #[inline]
    pub fn factor_ldlt_ordered<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<StaticCscLdlt<N, MAX_L_NNZ, T>, SparseCholeskyError> {
        let mut factor = StaticCscLdlt {
            lower: StaticCscMatrix {
                values: [T::zero(); MAX_L_NNZ],
                pattern: self.lower,
            },
            diagonal: [T::zero(); N],
            ordering: self.ordering,
        };
        self.factor_ldlt_ordered_into(matrix, &mut factor)?;
        Ok(factor)
    }

    /// Factors ordered coordinates into reusable LDLᵀ storage.
    #[inline]
    pub fn factor_ldlt_ordered_into<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        output: &mut StaticCscLdlt<N, MAX_L_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        self.factor_ldlt_natural_into(matrix, false, output)?;
        output.ordering = self.ordering;
        Ok(())
    }

    fn factor_ldlt_natural_into<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        validate_pattern: bool,
        output: &mut StaticCscLdlt<N, MAX_L_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        if validate_pattern {
            self.validate_factor_pattern(matrix)?;
        }
        output.lower.pattern = self.lower;
        output.ordering = StaticCscOrdering::identity();

        for column in 0..N {
            let mut work = [T::zero(); N];
            let matrix_start = matrix.column_starts()[column];
            let matrix_end = matrix.column_end(column).unwrap_or(matrix.nnz());
            for index in matrix_start..matrix_end {
                let row = matrix.row_indices()[index];
                if row >= column {
                    work[row] = matrix.values()[index];
                }
            }

            let update_start = self.update_starts[column];
            let update_end = if column + 1 < N {
                self.update_starts[column + 1]
            } else {
                self.update_nnz
            };
            for update in update_start..update_end {
                let lower_index = self.update_indices[update];
                let previous = self.update_columns[update];
                let scale = output.lower.values()[lower_index] * output.diagonal[previous];
                let end = output
                    .lower
                    .column_end(previous)
                    .unwrap_or(output.lower.nnz());
                for index in lower_index..end {
                    let row = output.lower.row_indices()[index];
                    work[row] = work[row] - output.lower.values()[index] * scale;
                }
            }

            let diagonal = work[column];
            if !diagonal.is_finite() {
                return Err(SparseCholeskyError::NonFinite);
            }
            if diagonal == T::zero() {
                return Err(SparseCholeskyError::ZeroPivot);
            }
            output.diagonal[column] = diagonal;
            let start = output.lower.column_starts()[column];
            let end = output
                .lower
                .column_end(column)
                .unwrap_or(output.lower.nnz());
            output.lower.values_mut()[start] = T::one();
            for index in (start + 1)..end {
                let row = output.lower.row_indices()[index];
                let value = work[row] / diagonal;
                if !value.is_finite() {
                    return Err(SparseCholeskyError::NonFinite);
                }
                output.lower.values_mut()[index] = value;
            }
        }

        Ok(())
    }

    fn factor_natural_into<const MAX_A_NNZ: usize, T: Real + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        validate_pattern: bool,
        output: &mut StaticCscCholesky<N, MAX_L_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        if validate_pattern {
            self.validate_factor_pattern(matrix)?;
        }
        output.lower.pattern = self.lower;
        output.ordering = StaticCscOrdering::identity();

        for column in 0..N {
            let mut work = [T::zero(); N];
            let matrix_start = matrix.column_starts()[column];
            let matrix_end = matrix.column_end(column).unwrap_or(matrix.nnz());
            for index in matrix_start..matrix_end {
                let row = matrix.row_indices()[index];
                if row >= column {
                    work[row] = matrix.values()[index];
                }
            }

            let update_start = self.update_starts[column];
            let update_end = if column + 1 < N {
                self.update_starts[column + 1]
            } else {
                self.update_nnz
            };
            for update in update_start..update_end {
                let lower_index = self.update_indices[update];
                let previous = self.update_columns[update];
                let lower_column = output.lower.values()[lower_index];
                let end = output
                    .lower
                    .column_end(previous)
                    .unwrap_or(output.lower.nnz());
                for index in lower_index..end {
                    let row = output.lower.row_indices()[index];
                    work[row] = work[row] - output.lower.values()[index] * lower_column;
                }
            }

            let diagonal = work[column];
            if !diagonal.is_finite() {
                return Err(SparseCholeskyError::NonFinite);
            }
            if diagonal <= T::zero() {
                return Err(SparseCholeskyError::NotPositiveDefinite);
            }
            let root = diagonal.sqrt();
            if !root.is_finite() || root <= T::zero() {
                return Err(SparseCholeskyError::NonFinite);
            }
            let start = output.lower.column_starts()[column];
            let end = output
                .lower
                .column_end(column)
                .unwrap_or(output.lower.nnz());
            output.lower.values_mut()[start] = root;
            for index in (start + 1)..end {
                let row = output.lower.row_indices()[index];
                let value = work[row] / root;
                if !value.is_finite() {
                    return Err(SparseCholeskyError::NonFinite);
                }
                output.lower.values_mut()[index] = value;
            }
        }

        Ok(())
    }

    fn validate_factor_pattern<const MAX_A_NNZ: usize, T: Copy + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        for column in 0..N {
            let start = matrix.column_starts()[column];
            let end = matrix.column_end(column).unwrap_or(matrix.nnz());
            for index in start..end {
                let row = matrix.row_indices()[index];
                let (lower_row, lower_column) = if row >= column {
                    (row, column)
                } else {
                    (column, row)
                };
                let lower_start = self.lower.column_starts()[lower_column];
                let lower_end = self
                    .lower
                    .column_end(lower_column)
                    .unwrap_or(self.lower.nnz());
                if !self.lower.row_indices()[lower_start..lower_end].contains(&lower_row) {
                    return Err(SparseCholeskyError::PatternMismatch);
                }
            }
        }
        Ok(())
    }
}

fn validate_symmetric_structure<const N: usize, const MAX_NNZ: usize, T: Real + Copy>(
    matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
) -> Result<(), SparseCholeskyError> {
    for column in 0..N {
        let start = matrix.column_starts()[column];
        let end = matrix.column_end(column).unwrap_or(matrix.nnz());
        for index in start..end {
            let row = matrix.row_indices()[index];
            let value = matrix.values()[index];
            if !value.is_finite() {
                return Err(SparseCholeskyError::NonFinite);
            }
            if row < column {
                let transpose = matrix
                    .get(column, row)
                    .copied()
                    .ok_or(SparseCholeskyError::NonSymmetric)?;
                if !transpose.is_finite() {
                    return Err(SparseCholeskyError::NonFinite);
                }
                let scale = T::one().max(value.abs()).max(transpose.abs());
                let tolerance = T::epsilon() * T::from(100).unwrap_or(T::one()) * scale;
                if (value - transpose).abs() > tolerance {
                    return Err(SparseCholeskyError::NonSymmetric);
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn diagonal_pivot_ordering<const N: usize, const MAX_NNZ: usize, T: Real + Copy + Zero>(
    matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
    threshold: T,
) -> Result<StaticCscOrdering<N>, SparseCholeskyError> {
    if !threshold.is_finite() {
        return Err(SparseCholeskyError::NonFinite);
    }
    let threshold = threshold.abs();
    let mut work = [[T::zero(); N]; N];
    for column in 0..N {
        let start = matrix.column_starts()[column];
        let end = matrix.column_end(column).unwrap_or(matrix.nnz());
        for index in start..end {
            let row = matrix.row_indices()[index];
            if row >= column {
                let value = matrix.values()[index];
                work[row][column] = value;
                work[column][row] = value;
            }
        }
    }

    let mut permutation = [0; N];
    for (index, value) in permutation.iter_mut().enumerate() {
        *value = index;
    }
    for pivot_position in 0..N {
        let mut pivot = pivot_position;
        let mut magnitude = work[pivot_position][pivot_position].abs();
        for candidate in (pivot_position + 1)..N {
            let candidate_magnitude = work[candidate][candidate].abs();
            if candidate_magnitude > magnitude {
                magnitude = candidate_magnitude;
                pivot = candidate;
            }
        }
        if !magnitude.is_finite() {
            return Err(SparseCholeskyError::NonFinite);
        }
        if magnitude <= threshold {
            return Err(SparseCholeskyError::ZeroPivot);
        }
        if pivot != pivot_position {
            work.swap(pivot_position, pivot);
            for index in 0..N {
                work[index].swap(pivot_position, pivot);
            }
            permutation.swap(pivot_position, pivot);
        }

        let diagonal = work[pivot_position][pivot_position];
        for row in (pivot_position + 1)..N {
            work[row][pivot_position] = work[row][pivot_position] / diagonal;
            work[pivot_position][row] = work[row][pivot_position];
            if !work[row][pivot_position].is_finite() {
                return Err(SparseCholeskyError::NonFinite);
            }
        }
        for column in (pivot_position + 1)..N {
            let lower_column = work[column][pivot_position];
            for row in column..N {
                work[row][column] =
                    work[row][column] - work[row][pivot_position] * diagonal * lower_column;
                work[column][row] = work[row][column];
                if !work[row][column].is_finite() {
                    return Err(SparseCholeskyError::NonFinite);
                }
            }
        }
    }
    StaticCscOrdering::from_permutation(&permutation).map_err(SparseCholeskyError::from)
}

fn permute_matrix<const N: usize, const MAX_NNZ: usize, T: Copy + Zero>(
    matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
    ordering: StaticCscOrdering<N>,
) -> Result<StaticCscMatrix<N, N, MAX_NNZ, T>, CscError> {
    let mut present = [[false; N]; N];
    let mut mapped_values = [[T::zero(); N]; N];
    for column in 0..N {
        let start = matrix.column_starts()[column];
        let end = matrix.column_end(column).unwrap_or(matrix.nnz());
        for index in start..end {
            let row = matrix.row_indices()[index];
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
                mapped_values[lower_column][lower_row] = matrix.values()[index];
            }
        }
    }

    let mut output = StaticCscMatrix::new();
    let mut position = 0;
    for column in 0..N {
        output.pattern.column_starts[column] = position;
        for row in column..N {
            if present[column][row] {
                if position == MAX_NNZ {
                    return Err(CscError::CapacityExceeded);
                }
                output.pattern.row_indices[position] = row;
                output.values[position] = mapped_values[column][row];
                position += 1;
            }
        }
    }
    output.pattern.nnz = position;
    Ok(output)
}

/// Numeric fixed-capacity simplicial sparse Cholesky factorization.
///
/// The factor stores a lower-triangular `L` in CSC form such that
/// `A = L * Lᵀ`. Any symbolic fill must fit in `MAX_L_NNZ` entries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticCscCholesky<const N: usize, const MAX_L_NNZ: usize, T = f32> {
    lower: StaticCscMatrix<N, N, MAX_L_NNZ, T>,
    ordering: StaticCscOrdering<N>,
}

impl<const N: usize, const MAX_L_NNZ: usize, T: Real> StaticCscCholesky<N, MAX_L_NNZ, T> {
    /// Performs symbolic analysis and numeric factorization in one step.
    #[inline]
    pub fn decompose<const MAX_A_NNZ: usize>(
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<Self, SparseCholeskyError> {
        StaticCscCholeskyPattern::analyze(matrix)?.factor(matrix)
    }

    /// Returns the reusable symbolic factor pattern.
    #[inline]
    pub fn pattern(&self) -> StaticCscCholeskyPattern<N, MAX_L_NNZ> {
        StaticCscCholeskyPattern::from_lower_with_ordering(*self.lower.pattern(), self.ordering)
    }

    /// Returns the sparse lower-triangular factor.
    #[inline]
    pub fn lower(&self) -> &StaticCscMatrix<N, N, MAX_L_NNZ, T> {
        &self.lower
    }

    /// Solves `A * X = B` using the sparse factor.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<N, P, T>) -> Matrix<N, P, T> {
        let mut output = *rhs;
        self.solve_in_place(&mut output);
        output
    }

    /// Solves `A * X = B` into caller-provided storage.
    #[inline]
    pub fn solve_into<const P: usize>(&self, rhs: &Matrix<N, P, T>, output: &mut Matrix<N, P, T>) {
        *output = *rhs;
        self.solve_in_place(output);
    }

    /// Solves `A * X = B` in place using forward and backward substitution.
    #[inline]
    pub fn solve_in_place<const P: usize>(&self, rhs: &mut Matrix<N, P, T>) {
        if !self.ordering.is_identity() {
            let mut permuted = Matrix::<N, P, T>::zeros();
            for ordered in 0..N {
                let original = self.ordering.permutation()[ordered];
                for column in 0..P {
                    permuted[(ordered, column)] = rhs[(original, column)];
                }
            }
            self.solve_natural_in_place(&mut permuted);
            for ordered in 0..N {
                let original = self.ordering.permutation()[ordered];
                for column in 0..P {
                    rhs[(original, column)] = permuted[(ordered, column)];
                }
            }
            return;
        }
        self.solve_natural_in_place(rhs);
    }

    fn solve_natural_in_place<const P: usize>(&self, rhs: &mut Matrix<N, P, T>) {
        for row in 0..N {
            let start = self.lower.column_starts()[row];
            let end = self.lower.column_end(row).unwrap_or(self.lower.nnz());
            let diagonal = self.lower.values()[start];
            for column in 0..P {
                rhs[(row, column)] = rhs[(row, column)] / diagonal;
            }
            for index in (start + 1)..end {
                let target = self.lower.row_indices()[index];
                let value = self.lower.values()[index];
                for column in 0..P {
                    rhs[(target, column)] = rhs[(target, column)] - value * rhs[(row, column)];
                }
            }
        }

        for row in (0..N).rev() {
            let start = self.lower.column_starts()[row];
            let end = self.lower.column_end(row).unwrap_or(self.lower.nnz());
            let diagonal = self.lower.values()[start];
            for index in (start + 1)..end {
                let source = self.lower.row_indices()[index];
                let value = self.lower.values()[index];
                for column in 0..P {
                    rhs[(row, column)] = rhs[(row, column)] - value * rhs[(source, column)];
                }
            }
            for column in 0..P {
                rhs[(row, column)] = rhs[(row, column)] / diagonal;
            }
        }
    }
}

/// Numeric fixed-capacity simplicial sparse LDLᵀ factorization without
/// diagonal pivoting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticCscLdlt<const N: usize, const MAX_L_NNZ: usize, T = f32> {
    lower: StaticCscMatrix<N, N, MAX_L_NNZ, T>,
    diagonal: [T; N],
    ordering: StaticCscOrdering<N>,
}

impl<const N: usize, const MAX_L_NNZ: usize, T: Real> StaticCscLdlt<N, MAX_L_NNZ, T> {
    /// Performs symbolic analysis and numeric LDLᵀ factorization in one step.
    #[inline]
    pub fn decompose<const MAX_A_NNZ: usize>(
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<Self, SparseCholeskyError> {
        StaticCscCholeskyPattern::analyze(matrix)?.factor_ldlt(matrix)
    }

    /// Performs analysis-time diagonal pivoting followed by sparse LDLᵀ.
    ///
    /// This supports 1x1 symmetric diagonal pivots. Matrices requiring a
    /// 2x2 pivot block still return [`SparseCholeskyError::ZeroPivot`].
    #[inline]
    pub fn decompose_with_diagonal_pivoting<const MAX_A_NNZ: usize>(
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        threshold: T,
    ) -> Result<Self, SparseCholeskyError> {
        StaticCscCholeskyPattern::analyze_with_diagonal_pivoting(matrix, threshold)?
            .factor_ldlt(matrix)
    }

    /// Returns the reusable symbolic factor pattern.
    #[inline]
    pub fn pattern(&self) -> StaticCscCholeskyPattern<N, MAX_L_NNZ> {
        StaticCscCholeskyPattern::from_lower_with_ordering(*self.lower.pattern(), self.ordering)
    }

    /// Returns the unit lower-triangular factor.
    #[inline]
    pub fn lower(&self) -> &StaticCscMatrix<N, N, MAX_L_NNZ, T> {
        &self.lower
    }

    /// Returns the diagonal factor `D`.
    #[inline]
    pub fn diagonal(&self) -> &[T; N] {
        &self.diagonal
    }

    /// Returns the ordering used by this factorization.
    #[inline]
    pub const fn ordering(&self) -> StaticCscOrdering<N> {
        self.ordering
    }

    /// Solves `A * X = B` using the sparse factor.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<N, P, T>) -> Matrix<N, P, T> {
        let mut output = *rhs;
        self.solve_in_place(&mut output);
        output
    }

    /// Solves `A * X = B` into caller-provided output storage.
    #[inline]
    pub fn solve_into<const P: usize>(&self, rhs: &Matrix<N, P, T>, output: &mut Matrix<N, P, T>) {
        *output = *rhs;
        self.solve_in_place(output);
    }

    /// Solves `A * X = B` in place using sparse LDLᵀ substitution.
    #[inline]
    pub fn solve_in_place<const P: usize>(&self, rhs: &mut Matrix<N, P, T>) {
        if !self.ordering.is_identity() {
            let mut permuted = Matrix::<N, P, T>::zeros();
            for ordered in 0..N {
                let original = self.ordering.permutation()[ordered];
                for column in 0..P {
                    permuted[(ordered, column)] = rhs[(original, column)];
                }
            }
            self.solve_natural_in_place(&mut permuted);
            for ordered in 0..N {
                let original = self.ordering.permutation()[ordered];
                for column in 0..P {
                    rhs[(original, column)] = permuted[(ordered, column)];
                }
            }
            return;
        }
        self.solve_natural_in_place(rhs);
    }

    fn solve_natural_in_place<const P: usize>(&self, rhs: &mut Matrix<N, P, T>) {
        for row in 0..N {
            let start = self.lower.column_starts()[row];
            let end = self.lower.column_end(row).unwrap_or(self.lower.nnz());
            for index in (start + 1)..end {
                let target = self.lower.row_indices()[index];
                let value = self.lower.values()[index];
                for column in 0..P {
                    rhs[(target, column)] = rhs[(target, column)] - value * rhs[(row, column)];
                }
            }
        }

        for row in 0..N {
            let diagonal = self.diagonal[row];
            for column in 0..P {
                rhs[(row, column)] = rhs[(row, column)] / diagonal;
            }
        }

        for row in (0..N).rev() {
            let start = self.lower.column_starts()[row];
            let end = self.lower.column_end(row).unwrap_or(self.lower.nnz());
            for index in (start + 1)..end {
                let source = self.lower.row_indices()[index];
                let value = self.lower.values()[index];
                for column in 0..P {
                    rhs[(row, column)] = rhs[(row, column)] - value * rhs[(source, column)];
                }
            }
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize, T> Default
    for StaticCscMatrix<ROWS, COLS, MAX_NNZ, T>
where
    T: Copy + Zero,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector;

    type Csc = StaticCscMatrix<3, 2, 4, f64>;

    #[test]
    fn insertion_keeps_canonical_csc_pattern() {
        let mut matrix = Csc::new();
        matrix.insert(2, 1, 5.0).unwrap();
        matrix.insert(0, 0, 1.0).unwrap();
        matrix.insert(1, 0, 2.0).unwrap();
        matrix.insert(0, 1, 3.0).unwrap();
        assert_eq!(matrix.values(), &[1.0, 2.0, 3.0, 5.0]);
        assert_eq!(matrix.row_indices(), &[0, 1, 0, 2]);
        assert_eq!(matrix.column_starts(), &[0, 2]);
        assert_eq!(matrix.column_end(0), Some(2));
        assert_eq!(matrix.column_end(1), Some(4));
    }

    #[test]
    fn matvec_and_value_updates_work_without_repacking() {
        let mut matrix = Csc::from_pattern(&[1.0, 2.0, 3.0], &[0, 2, 1], &[0, 2, 3]).unwrap();
        let vector = vector![4.0; 5.0];
        assert_eq!(matrix.matvec(&vector), vector![4.0; 15.0; 8.0]);
        matrix.set_value(2, 0, 7.0).unwrap();
        assert_eq!(matrix.matvec(&vector), vector![4.0; 15.0; 28.0]);
        matrix.set_values(&[2.0, 4.0, 6.0]).unwrap();
        assert_eq!(matrix.values(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn malformed_patterns_are_rejected() {
        assert_eq!(
            Csc::from_pattern(&[1.0], &[0], &[0, 1]),
            Err(CscError::LengthMismatch)
        );
        assert_eq!(
            Csc::from_pattern(&[1.0], &[3], &[0, 1, 1]),
            Err(CscError::InvalidRowIndices)
        );
        assert_eq!(
            Csc::from_pattern(&[1.0], &[0], &[1, 1, 1]),
            Err(CscError::InvalidColumnPointers)
        );
    }

    #[test]
    fn capacity_and_bounds_are_checked() {
        let mut matrix = StaticCscMatrix::<1, 1, 1, f64>::new();
        assert_eq!(matrix.insert(1, 0, 1.0), Err(CscError::IndexOutOfBounds));
        matrix.insert(0, 0, 1.0).unwrap();
        assert_eq!(matrix.insert(0, 0, 2.0), Ok(()));
        assert_eq!(matrix.get(0, 0), Some(&2.0));
        assert_eq!(matrix.set_value(0, 1, 1.0), Err(CscError::IndexOutOfBounds));
    }
}
