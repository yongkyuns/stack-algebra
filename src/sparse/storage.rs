use core::ops::{Add, AddAssign, Mul};

use crate::{Matrix, Zero};

use super::errors::CscError;

/// An immutable fixed-capacity CSC sparsity pattern.
///
/// Keeping the symbolic structure separate allows generated code to reuse a
/// validated pattern while replacing numeric values at every iteration.
///
/// The arrays use canonical CSC ordering. For a `2 x 2` diagonal pattern,
/// `row_indices = [0, 1]` and `column_pointers = [0, 1, 2]`:
///
/// ```
/// use stack_algebra::StaticCscPattern;
///
/// let pattern = StaticCscPattern::<2, 2, 2>::from_arrays(&[0, 1], &[0, 1, 2]).unwrap();
/// assert_eq!(pattern.nnz(), 2);
/// assert_eq!(pattern.column_end(1), Some(2));
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticCscPattern<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize> {
    pub(crate) row_indices: [usize; MAX_NNZ],
    pub(crate) column_starts: [usize; COLS],
    pub(crate) nnz: usize,
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

    /// Returns the exact inline storage footprint of this symbolic pattern.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
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

    pub(crate) fn from_parts(
        row_indices: &[usize],
        column_starts: &[usize; COLS],
        nnz: usize,
    ) -> Result<Self, CscError> {
        if nnz != row_indices.len() || nnz > MAX_NNZ {
            return Err(CscError::LengthMismatch);
        }
        if COLS == 0 || column_starts[0] != 0 {
            return Err(CscError::InvalidColumnPointers);
        }
        let mut previous = 0;
        for &pointer in column_starts {
            if pointer < previous || pointer > nnz {
                return Err(CscError::InvalidColumnPointers);
            }
            previous = pointer;
        }
        for column in 0..COLS {
            let start = column_starts[column];
            let end = if column + 1 < COLS {
                column_starts[column + 1]
            } else {
                nnz
            };
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
        pattern.column_starts = *column_starts;
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

    /// Returns the value-array index for an existing `(row, column)` entry.
    #[inline]
    pub fn entry_index(&self, row: usize, column: usize) -> Option<usize> {
        if row >= ROWS || column >= COLS {
            return None;
        }
        let start = self.column_starts[column];
        let end = self.column_end(column).unwrap_or(self.nnz);
        self.row_indices[start..end]
            .binary_search(&row)
            .ok()
            .map(|offset| start + offset)
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
/// The matrix owns its storage and never allocates. The sparsity pattern is
/// stored in canonical CSC form: column starts are stored for each of the
/// `COLS` columns and the final column end is `nnz()`. Row indices are
/// strictly increasing within each column. `MAX_NNZ` bounds the number of
/// stored entries; unused capacity remains in the backing arrays but is not
/// exposed through the active slices.
///
/// Use [`Self::from_pattern`] when a complete canonical pattern is available,
/// or [`Self::new`] and [`Self::insert`] when assembling a small pattern. The
/// latter keeps row indices sorted and returns an error instead of allocating
/// when `MAX_NNZ` is full.
///
/// ```
/// use stack_algebra::{Matrix, StaticCscMatrix};
///
/// let mut a = StaticCscMatrix::<2, 2, 3, f32>::new();
/// a.insert(1, 0, 2.0).unwrap();
/// a.insert(0, 0, 4.0).unwrap();
/// a.insert(1, 1, 3.0).unwrap();
/// let x = Matrix::<2, 1, f32>::from_rows([[1.0], [2.0]]);
/// assert_eq!(a.matvec(&x), Matrix::from_rows([[4.0], [8.0]]));
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticCscMatrix<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize, T = f32> {
    pub(crate) values: [T; MAX_NNZ],
    pub(crate) pattern: StaticCscPattern<ROWS, COLS, MAX_NNZ>,
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

    /// Returns the exact inline storage footprint of this sparse matrix.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
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

    /// Creates a zero-valued matrix from a validated symbolic pattern.
    #[inline]
    pub fn zero_with_pattern(pattern: StaticCscPattern<ROWS, COLS, MAX_NNZ>) -> Self {
        Self {
            values: [T::zero(); MAX_NNZ],
            pattern,
        }
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

    /// Adds to an existing stored entry without changing the sparsity pattern.
    ///
    /// This is useful when assembling a normal equation matrix whose symbolic
    /// CSC pattern was prepared before numeric factorization.
    #[inline]
    pub fn add_to_value(&mut self, row: usize, column: usize, value: T) -> Result<(), CscError>
    where
        T: AddAssign,
    {
        let index = self
            .find_entry(row, column)?
            .ok_or(CscError::EntryNotFound)?;
        self.values[index] += value;
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
        Ok(self.pattern.row_indices[start..end]
            .binary_search(&row)
            .ok()
            .map(|offset| start + offset))
    }
}
