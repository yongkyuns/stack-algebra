#![no_std]

mod index;
mod new;
mod num;
mod ops;
mod util;

use core::slice;

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

impl<const M: usize, const N: usize, T> Matrix<M, N, T> {
    /// Returns a raw pointer to the underlying data.
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr() as *const T
    }

    /// Returns an unsafe mutable pointer to the underlying data.
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr() as *mut T
    }

    /// Views the underlying data as a contiguous slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), M * N) }
    }

    /// Views the underlying data as a contiguous mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), M * N) }
    }
}

/// A matrix with one row and `N` columns.
pub type RowVector<const N: usize, T> = Matrix<1, N, T>;

/// A matrix with one column and `M` rows.
pub type Vector<const M: usize, T> = Matrix<M, 1, T>;

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

    #[test]
    fn index() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        assert_eq!(m[1], 4.0);
        assert_eq!(m[(1, 2)], 6.0);

        let mut s = m.as_slice().iter();
        assert_eq!(s.next(), Some(&1.0));
        assert_eq!(s.next(), Some(&4.0));
        assert_eq!(s.next(), Some(&2.0));
        assert_eq!(s.next(), Some(&5.0));
        assert_eq!(s.next(), Some(&3.0));
        assert_eq!(s.next(), Some(&6.0));
        assert_eq!(s.next(), None);
    }
}
