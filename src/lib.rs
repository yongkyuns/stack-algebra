#![no_std]

mod num;
mod util;

#[doc(hidden)]
pub use vectrix_macro as proc_macro;

/// Represents a matrix with constant `M` rows and constant `N` columns.
///
/// The underlying data is represented as an array and is always stored in
/// column-major order.
///
/// See the [crate root][crate] for usage examples.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Matrix<const M: usize, const N: usize, T = f32> {
    data: [[T; M]; N],
}

/// A matrix with one row and `N` columns.
pub type RowVector<const N: usize, T> = Matrix<1, N, T>;

/// A matrix with one column and `M` rows.
pub type Vector<const M: usize, T> = Matrix<M, 1, T>;

////////////////////////////////////////////////////////////////////////////////
// Matrix<T, M, N> methods
////////////////////////////////////////////////////////////////////////////////
impl<const M: usize, const N: usize, T> Matrix<M, N, T> {
    /// Create a new matrix from an array of arrays in column-major order.
    #[doc(hidden)]
    #[inline]
    pub const fn from_column_major_order(data: [[T; M]; N]) -> Self {
        Self { data }
    }
}

impl<const M: usize, const N: usize, T> Matrix<M, N, T>
where
    T: num::Zero + Copy,
{
    /// Create a new matrix from an array of arrays in column-major order.
    #[doc(hidden)]
    #[inline]
    fn zeros() -> Self {
        Self::from_column_major_order([[T::zero(); M]; N])
    }
}

impl<const M: usize, const N: usize, T> Matrix<M, N, T>
where
    T: num::One + Copy,
{
    /// Create a new matrix from an array of arrays in column-major order.
    #[doc(hidden)]
    #[inline]
    fn ones() -> Self {
        Self::from_column_major_order([[T::one(); M]; N])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        assert_eq!(m.data[0][0], 1.0);
        assert_eq!(m.data[2][1], 6.0);

        let v = vector![1.0, 2.0, 3.0];
        assert_eq!(v.data[0][0], 1.0);
        assert_eq!(v.data[2][0], 3.0);

        let z = zeros!(2, 3);
        assert_eq!(z.data[0][0], 0.0);
        assert_eq!(z.data[2][1], 0.0);

        let z = zeros!(3);
        assert_eq!(z.data[2][2], 0.0);

        let o = ones!(2, 3);
        assert_eq!(o.data[0][0], 1.0);
        assert_eq!(o.data[2][1], 1.0);

        let o = ones!(3);
        assert_eq!(o.data[2][2], 1.0);
    }
}
