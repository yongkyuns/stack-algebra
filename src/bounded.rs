//! Stack-backed matrices with runtime active dimensions.
//!
//! [`MatrixBuf`] is useful when a device receives a matrix size at runtime but
//! has a known upper bound. It reserves its complete capacity inline and only
//! tracks the active rectangle, so construction and resizing never allocate.
//! The active values are still column-major, matching [`Matrix`](crate::Matrix).
//!
//! # Example
//!
//! ```
//! use stack_algebra::{matrix, MatrixBuf};
//!
//! let mut buffer = MatrixBuf::<4, 4, f32>::new(2, 3).unwrap();
//! buffer[(1, 2)] = 7.0;
//! buffer.resize(3, 2).unwrap();
//! assert_eq!(buffer.rows(), 3);
//! assert_eq!(buffer.columns(), 2);
//! assert!(buffer.get(1, 2).is_none());
//!
//! let source = matrix![1_i32, 2, 3; 4, 5, 6];
//! let bounded = MatrixBuf::<4, 4, _>::from_matrix(&source).unwrap();
//! assert_eq!(bounded.to_matrix::<2, 3>(), Ok(source));
//! ```

use core::ops::{Index, IndexMut};

use crate::view::{MatrixRead, MatrixWrite};
use crate::{Matrix, Zero};

/// Describes why a [`MatrixBuf`] operation could not satisfy a requested shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixBufError {
    /// A requested active shape exceeds the compile-time capacity.
    CapacityExceeded {
        /// Requested row count.
        rows: usize,
        /// Requested column count.
        columns: usize,
        /// Maximum supported row count.
        max_rows: usize,
        /// Maximum supported column count.
        max_columns: usize,
    },
    /// A column-major input slice has the wrong number of values.
    LengthMismatch {
        /// Number of values implied by the requested shape.
        expected: usize,
        /// Number of values supplied by the caller.
        actual: usize,
    },
    /// A shape calculation overflowed `usize`.
    SizeOverflow,
    /// A fixed-size target does not match the buffer's active shape.
    ShapeMismatch {
        /// Row count expected by the target.
        expected_rows: usize,
        /// Column count expected by the target.
        expected_columns: usize,
        /// Active row count in the buffer.
        actual_rows: usize,
        /// Active column count in the buffer.
        actual_columns: usize,
    },
}

/// A stack-allocated matrix with runtime active dimensions and compile-time
/// maximum capacity.
///
/// Storage remains column-major and reserves `MAX_ROWS * MAX_COLS` scalar
/// slots. Changing the active dimensions never allocates or moves storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatrixBuf<const MAX_ROWS: usize, const MAX_COLS: usize, T = f32> {
    data: [[T; MAX_ROWS]; MAX_COLS],
    rows: usize,
    columns: usize,
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, T> MatrixBuf<MAX_ROWS, MAX_COLS, T>
where
    T: Copy + Zero,
{
    /// Creates a zero-filled buffer with the requested active dimensions.
    ///
    /// Returns [`MatrixBufError::CapacityExceeded`] when either requested
    /// dimension exceeds the type-level capacity. The entire backing array is
    /// initialized inline.
    #[inline]
    pub fn new(rows: usize, columns: usize) -> Result<Self, MatrixBufError> {
        if rows > MAX_ROWS || columns > MAX_COLS {
            return Err(MatrixBufError::CapacityExceeded {
                rows,
                columns,
                max_rows: MAX_ROWS,
                max_columns: MAX_COLS,
            });
        }
        Ok(Self {
            data: [[T::zero(); MAX_ROWS]; MAX_COLS],
            rows,
            columns,
        })
    }

    /// Returns the exact inline storage footprint of this bounded buffer.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Creates a bounded matrix from an owning fixed-size matrix.
    #[inline]
    pub fn from_matrix<const M: usize, const N: usize>(
        matrix: &Matrix<M, N, T>,
    ) -> Result<Self, MatrixBufError> {
        let mut output = Self::new(M, N)?;
        for column in 0..N {
            for row in 0..M {
                output.data[column][row] = matrix[(row, column)];
            }
        }
        Ok(output)
    }

    /// Creates a bounded matrix from active column-major values.
    #[inline]
    pub fn from_column_major(
        rows: usize,
        columns: usize,
        values: &[T],
    ) -> Result<Self, MatrixBufError> {
        let expected = rows
            .checked_mul(columns)
            .ok_or(MatrixBufError::SizeOverflow)?;
        if values.len() != expected {
            return Err(MatrixBufError::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }
        let mut output = Self::new(rows, columns)?;
        for column in 0..columns {
            let source = &values[column * rows..(column + 1) * rows];
            output.data[column][..rows].copy_from_slice(source);
        }
        Ok(output)
    }

    /// Returns the active row count.
    #[inline]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the active column count.
    #[inline]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the compile-time row capacity.
    #[inline]
    pub const fn max_rows(&self) -> usize {
        MAX_ROWS
    }

    /// Returns the compile-time column capacity.
    #[inline]
    pub const fn max_columns(&self) -> usize {
        MAX_COLS
    }

    /// Changes the active dimensions without reallocating or clearing storage.
    ///
    /// Newly exposed elements retain whatever values were previously stored in
    /// the inline capacity. Initialize them before reading when growing a
    /// buffer.
    #[inline]
    pub fn resize(&mut self, rows: usize, columns: usize) -> Result<(), MatrixBufError> {
        if rows > MAX_ROWS || columns > MAX_COLS {
            return Err(MatrixBufError::CapacityExceeded {
                rows,
                columns,
                max_rows: MAX_ROWS,
                max_columns: MAX_COLS,
            });
        }
        self.rows = rows;
        self.columns = columns;
        Ok(())
    }

    /// Returns an active element, or `None` outside the active dimensions.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        if row < self.rows && column < self.columns {
            Some(&self.data[column][row])
        } else {
            None
        }
    }

    /// Returns an active mutable element, or `None` outside the active dimensions.
    #[inline]
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        if row < self.rows && column < self.columns {
            Some(&mut self.data[column][row])
        } else {
            None
        }
    }

    /// Returns the active portion of a column.
    #[inline]
    pub fn column(&self, column: usize) -> Option<&[T]> {
        if column < self.columns {
            Some(&self.data[column][..self.rows])
        } else {
            None
        }
    }

    /// Returns the active portion of a column mutably.
    #[inline]
    pub fn column_mut(&mut self, column: usize) -> Option<&mut [T]> {
        if column < self.columns {
            Some(&mut self.data[column][..self.rows])
        } else {
            None
        }
    }

    /// Copies active values into an owning matrix with matching dimensions.
    #[inline]
    pub fn copy_into<const M: usize, const N: usize>(
        &self,
        output: &mut Matrix<M, N, T>,
    ) -> Result<(), MatrixBufError> {
        if self.rows != M || self.columns != N {
            return Err(MatrixBufError::ShapeMismatch {
                expected_rows: M,
                expected_columns: N,
                actual_rows: self.rows,
                actual_columns: self.columns,
            });
        }
        for column in 0..N {
            for row in 0..M {
                output[(row, column)] = self.data[column][row];
            }
        }
        Ok(())
    }

    /// Returns the active matrix as an owning fixed-size matrix.
    #[inline]
    pub fn to_matrix<const M: usize, const N: usize>(
        &self,
    ) -> Result<Matrix<M, N, T>, MatrixBufError> {
        let mut output = Matrix::<M, N, T>::zeros();
        self.copy_into(&mut output)?;
        Ok(output)
    }

    /// Borrows the active values as a fixed-size read-only view.
    #[inline]
    pub fn as_view<const M: usize, const N: usize>(
        &self,
    ) -> Result<MatrixBufView<'_, MAX_ROWS, MAX_COLS, M, N, T>, MatrixBufError> {
        if self.rows == M && self.columns == N {
            Ok(MatrixBufView { buffer: self })
        } else {
            Err(MatrixBufError::ShapeMismatch {
                expected_rows: M,
                expected_columns: N,
                actual_rows: self.rows,
                actual_columns: self.columns,
            })
        }
    }

    /// Borrows the active values as a fixed-size mutable view.
    #[inline]
    pub fn as_view_mut<const M: usize, const N: usize>(
        &mut self,
    ) -> Result<MatrixBufViewMut<'_, MAX_ROWS, MAX_COLS, M, N, T>, MatrixBufError> {
        if self.rows == M && self.columns == N {
            Ok(MatrixBufViewMut { buffer: self })
        } else {
            Err(MatrixBufError::ShapeMismatch {
                expected_rows: M,
                expected_columns: N,
                actual_rows: self.rows,
                actual_columns: self.columns,
            })
        }
    }
}

/// Fixed-size read-only view into a matching active `MatrixBuf` region.
#[derive(Clone, Copy)]
pub struct MatrixBufView<
    'a,
    const MAX_ROWS: usize,
    const MAX_COLS: usize,
    const M: usize,
    const N: usize,
    T,
> {
    buffer: &'a MatrixBuf<MAX_ROWS, MAX_COLS, T>,
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, const M: usize, const N: usize, T>
    MatrixBufView<'_, MAX_ROWS, MAX_COLS, M, N, T>
where
    T: Copy + Zero,
{
    /// Returns the element at `(row, column)`, or `None` when out of bounds.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.buffer.get(row, column)
    }

    /// Copies the view into an owning fixed-size matrix.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<M, N, T> {
        self.buffer
            .to_matrix::<M, N>()
            .expect("view dimensions match the active buffer")
    }
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, const M: usize, const N: usize, T>
    Index<(usize, usize)> for MatrixBufView<'_, MAX_ROWS, MAX_COLS, M, N, T>
where
    T: Copy + Zero,
{
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("bounded matrix view index is out of bounds")
    }
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, const M: usize, const N: usize, T>
    MatrixRead<M, N, T> for MatrixBufView<'_, MAX_ROWS, MAX_COLS, M, N, T>
where
    T: Copy + Zero,
{
    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.buffer.get(row, column)
    }

    #[inline]
    fn get_in_bounds(&self, row: usize, column: usize) -> &T {
        &self.buffer.data[column][row]
    }
}

/// Fixed-size mutable view into a matching active `MatrixBuf` region.
pub struct MatrixBufViewMut<
    'a,
    const MAX_ROWS: usize,
    const MAX_COLS: usize,
    const M: usize,
    const N: usize,
    T,
> {
    buffer: &'a mut MatrixBuf<MAX_ROWS, MAX_COLS, T>,
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, const M: usize, const N: usize, T>
    MatrixBufViewMut<'_, MAX_ROWS, MAX_COLS, M, N, T>
where
    T: Copy + Zero,
{
    /// Reborrows the mutable view as a read-only view.
    #[inline]
    pub fn as_view(&self) -> MatrixBufView<'_, MAX_ROWS, MAX_COLS, M, N, T> {
        MatrixBufView {
            buffer: &*self.buffer,
        }
    }

    /// Returns the element at `(row, column)`, or `None` when out of bounds.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.buffer.get(row, column)
    }

    /// Returns a mutable element at `(row, column)`, or `None` when out of bounds.
    #[inline]
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        self.buffer.get_mut(row, column)
    }
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, const M: usize, const N: usize, T>
    Index<(usize, usize)> for MatrixBufViewMut<'_, MAX_ROWS, MAX_COLS, M, N, T>
where
    T: Copy + Zero,
{
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("bounded matrix view index is out of bounds")
    }
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, const M: usize, const N: usize, T>
    IndexMut<(usize, usize)> for MatrixBufViewMut<'_, MAX_ROWS, MAX_COLS, M, N, T>
where
    T: Copy + Zero,
{
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1)
            .expect("bounded matrix view index is out of bounds")
    }
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, const M: usize, const N: usize, T>
    MatrixRead<M, N, T> for MatrixBufViewMut<'_, MAX_ROWS, MAX_COLS, M, N, T>
where
    T: Copy + Zero,
{
    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.buffer.get(row, column)
    }

    #[inline]
    fn get_in_bounds(&self, row: usize, column: usize) -> &T {
        &self.buffer.data[column][row]
    }
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, const M: usize, const N: usize, T>
    MatrixWrite<M, N, T> for MatrixBufViewMut<'_, MAX_ROWS, MAX_COLS, M, N, T>
where
    T: Copy + Zero,
{
    #[inline]
    fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        self.buffer.get_mut(row, column)
    }
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, T> Index<(usize, usize)>
    for MatrixBuf<MAX_ROWS, MAX_COLS, T>
where
    T: Copy + Zero,
{
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("matrix buffer index is outside active dimensions")
    }
}

impl<const MAX_ROWS: usize, const MAX_COLS: usize, T> IndexMut<(usize, usize)>
    for MatrixBuf<MAX_ROWS, MAX_COLS, T>
where
    T: Copy + Zero,
{
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1)
            .expect("matrix buffer index is outside active dimensions")
    }
}

#[cfg(test)]
mod tests {
    use super::{MatrixBuf, MatrixBufError};
    use crate::{matrix, Cholesky, Matrix};

    #[test]
    fn bounded_matrix_tracks_active_dimensions() {
        let mut buffer = MatrixBuf::<4, 3, i32>::new(2, 3).unwrap();
        assert_eq!(buffer.rows(), 2);
        assert_eq!(buffer.columns(), 3);
        assert_eq!(buffer.max_rows(), 4);
        assert_eq!(buffer.max_columns(), 3);
        buffer[(1, 2)] = 7;
        assert_eq!(buffer.get(1, 2), Some(&7));
        assert!(buffer.get(2, 0).is_none());
        assert_eq!(
            buffer.resize(5, 1),
            Err(MatrixBufError::CapacityExceeded {
                rows: 5,
                columns: 1,
                max_rows: 4,
                max_columns: 3,
            })
        );
        buffer.resize(4, 2).unwrap();
        assert_eq!(buffer.rows(), 4);
        assert_eq!(buffer.columns(), 2);
    }

    #[test]
    fn bounded_matrix_round_trips_column_major_values() {
        let source = matrix![1_i32, 2, 3; 4, 5, 6];
        let buffer = MatrixBuf::<4, 4, i32>::from_matrix(&source).unwrap();
        assert_eq!(buffer.column(0), Some(&[1, 4][..]));
        assert_eq!(buffer.to_matrix::<2, 3>(), Ok(source));

        let from_values = MatrixBuf::<3, 2, i32>::from_column_major(2, 2, &[1, 3, 2, 4]).unwrap();
        assert_eq!(
            from_values.to_matrix::<2, 2>(),
            Ok(Matrix::from_rows([[1, 2], [3, 4]]))
        );
    }

    #[test]
    fn reports_inline_storage_footprint() {
        assert_eq!(
            MatrixBuf::<4, 3, i32>::storage_bytes(),
            core::mem::size_of::<MatrixBuf<4, 3, i32>>()
        );
        const FOOTPRINT: usize = MatrixBuf::<4, 3, i32>::storage_bytes();
        assert!(FOOTPRINT >= 4 * 3 * core::mem::size_of::<i32>());
    }

    #[test]
    fn fixed_views_match_active_dimensions() {
        let mut buffer = MatrixBuf::<4, 4, f64>::from_matrix(&matrix![4.0, 1.0; 1.0, 3.0]).unwrap();
        let view = buffer.as_view::<2, 2>().unwrap();
        assert_eq!(view.get(1, 0), Some(&1.0));
        assert_eq!(view[(0, 1)], 1.0);
        assert_eq!(view.to_matrix(), matrix![4.0, 1.0; 1.0, 3.0]);
        let factor = Cholesky::try_decompose_view(&view).unwrap();
        assert_eq!(factor.lower()[(0, 0)], 2.0);

        let mut view = buffer.as_view_mut::<2, 2>().unwrap();
        *view.get_mut(1, 0).unwrap() = 2.0;
        view[(0, 1)] = 2.0;
        assert_eq!(view.as_view()[(0, 1)], 2.0);
        assert_eq!(buffer[(1, 0)], 2.0);
        assert!(matches!(
            buffer.as_view::<3, 2>(),
            Err(MatrixBufError::ShapeMismatch {
                expected_rows: 3,
                expected_columns: 2,
                actual_rows: 2,
                actual_columns: 2,
            })
        ));
    }

    #[test]
    fn bounded_matrix_reports_constructor_errors() {
        assert_eq!(
            MatrixBuf::<2, 3, i32>::new(3, 1),
            Err(MatrixBufError::CapacityExceeded {
                rows: 3,
                columns: 1,
                max_rows: 2,
                max_columns: 3,
            })
        );
        assert_eq!(
            MatrixBuf::<2, 3, i32>::from_column_major(2, 2, &[1, 2, 3]),
            Err(MatrixBufError::LengthMismatch {
                expected: 4,
                actual: 3,
            })
        );
        assert_eq!(
            MatrixBuf::<2, 3, i32>::from_column_major(usize::MAX, 2, &[]),
            Err(MatrixBufError::SizeOverflow)
        );
    }
}
