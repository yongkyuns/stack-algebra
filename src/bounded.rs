use core::ops::{Index, IndexMut};

use crate::{Matrix, Zero};

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
    #[inline]
    pub fn new(rows: usize, columns: usize) -> Option<Self> {
        if rows > MAX_ROWS || columns > MAX_COLS {
            return None;
        }
        Some(Self {
            data: [[T::zero(); MAX_ROWS]; MAX_COLS],
            rows,
            columns,
        })
    }

    /// Creates a bounded matrix from an owning fixed-size matrix.
    #[inline]
    pub fn from_matrix<const M: usize, const N: usize>(matrix: &Matrix<M, N, T>) -> Option<Self> {
        let mut output = Self::new(M, N)?;
        for column in 0..N {
            for row in 0..M {
                output.data[column][row] = matrix[(row, column)];
            }
        }
        Some(output)
    }

    /// Creates a bounded matrix from active column-major values.
    #[inline]
    pub fn from_column_major(rows: usize, columns: usize, values: &[T]) -> Option<Self> {
        if values.len() != rows.checked_mul(columns)? {
            return None;
        }
        let mut output = Self::new(rows, columns)?;
        for column in 0..columns {
            let source = &values[column * rows..(column + 1) * rows];
            output.data[column][..rows].copy_from_slice(source);
        }
        Some(output)
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
    #[inline]
    pub fn resize(&mut self, rows: usize, columns: usize) -> Option<()> {
        if rows > MAX_ROWS || columns > MAX_COLS {
            return None;
        }
        self.rows = rows;
        self.columns = columns;
        Some(())
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
    ) -> Option<()> {
        if self.rows != M || self.columns != N {
            return None;
        }
        for column in 0..N {
            for row in 0..M {
                output[(row, column)] = self.data[column][row];
            }
        }
        Some(())
    }

    /// Returns the active matrix as an owning fixed-size matrix.
    #[inline]
    pub fn to_matrix<const M: usize, const N: usize>(&self) -> Option<Matrix<M, N, T>> {
        let mut output = Matrix::<M, N, T>::zeros();
        self.copy_into(&mut output)?;
        Some(output)
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
    use super::MatrixBuf;
    use crate::{matrix, Matrix};

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
        assert!(buffer.resize(5, 1).is_none());
        buffer.resize(4, 2).unwrap();
        assert_eq!(buffer.rows(), 4);
        assert_eq!(buffer.columns(), 2);
    }

    #[test]
    fn bounded_matrix_round_trips_column_major_values() {
        let source = matrix![1_i32, 2, 3; 4, 5, 6];
        let buffer = MatrixBuf::<4, 4, i32>::from_matrix(&source).unwrap();
        assert_eq!(buffer.column(0), Some(&[1, 4][..]));
        assert_eq!(buffer.to_matrix::<2, 3>(), Some(source));

        let from_values = MatrixBuf::<3, 2, i32>::from_column_major(2, 2, &[1, 3, 2, 4]).unwrap();
        assert_eq!(
            from_values.to_matrix::<2, 2>(),
            Some(Matrix::from_rows([[1, 2], [3, 4]]))
        );
    }
}
