use crate::num::{One, Zero};
use crate::Matrix;

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
    T: Zero + Copy,
{
    /// Create a new matrix from an array of arrays in column-major order.
    #[doc(hidden)]
    #[inline]
    pub fn zeros() -> Self {
        Self::from_column_major_order([[T::zero(); M]; N])
    }
}

impl<const M: usize, const N: usize, T> Matrix<M, N, T>
where
    T: One + Copy,
{
    /// Create a new matrix from an array of arrays in column-major order.
    #[doc(hidden)]
    #[inline]
    pub fn ones() -> Self {
        Self::from_column_major_order([[T::one(); M]; N])
    }
}

#[macro_export]
macro_rules! matrix {
    ($($data:tt)*) => {
        $crate::Matrix::from_column_major_order($crate::proc_macro::matrix!($($data)*))
    };
}

/// A macro for composing vectors.
#[macro_export]
macro_rules! vector {
    ($($data:tt)*) => {
        $crate::Matrix::from_column_major_order($crate::proc_macro::matrix!($($data)*))
    };
}

#[macro_export]
macro_rules! zeros {
    ($cols:expr) => {
        $crate::Matrix::<$cols, $cols>::zeros()
    };
    ($rows:expr, $cols:expr) => {{
        $crate::Matrix::<$rows, $cols>::zeros()
    }};
    ($rows:expr, $cols:expr, $ty:ty) => {{
        $crate::Matrix::<$rows, $cols, $ty>::zeros()
    }};
}

#[macro_export]
macro_rules! ones {
    ($cols:expr) => {
        $crate::Matrix::<$cols, $cols>::ones()
    };
    ($rows:expr, $cols:expr) => {{
        $crate::Matrix::<$rows, $cols>::ones()
    }};
    ($rows:expr, $cols:expr, $ty:ty) => {{
        $crate::Matrix::<$rows, $cols, $ty>::ones()
    }};
}
