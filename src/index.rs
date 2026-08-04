//! Indexing helpers for fixed-size matrices.
//!
//! Most users should use `matrix[row]`, `matrix[(row, column)]`, `get`, or
//! `get_mut` directly. [`MatrixIndex`] is the sealed extension point used by
//! those methods; it is public so generic code can name the associated output
//! type, but external implementations are intentionally not permitted.
//!
//! # Example
//!
//! ```
//! use stack_algebra::matrix;
//! let mut m = matrix![1_i32, 2; 3, 4];
//! assert_eq!(m[0], 1); // column-major flat index
//! assert_eq!(m[(1, 0)], 3); // row, column
//! assert_eq!(m.get((0, 2)), None);
//! *m.get_mut((0, 1)).unwrap() = 20;
//! assert_eq!(m[(0, 1)], 20);
//! ```

use crate::Matrix;

mod private {

    pub trait Sealed {}
    impl Sealed for usize {}
    impl Sealed for (usize, usize) {}
}

/// A helper trait used for indexing operations.
///
/// This is the [`Matrix`] version of [`SliceIndex`][`core::slice::SliceIndex`].
/// You should not use or implement this trait directly but instead use the
/// corresponding methods on [`Matrix`].
///
/// The built-in indices are `usize` for column-major flat access and
/// `(usize, usize)` for `(row, column)` access. Both checked and unchecked
/// forms use the same indexing convention.
///
/// # Safety
///
/// Implementations of this trait have to promise that if the argument
/// to `get_(mut_)unchecked` is a safe reference, then so is the result.
pub unsafe trait MatrixIndex<T: ?Sized>: private::Sealed {
    /// The output type returned by methods.
    type Output: ?Sized;

    /// Returns a shared reference to the output at this location, if in
    /// bounds.
    fn get(self, matrix: &T) -> Option<&Self::Output>;

    /// Returns a mutable reference to the output at this location, if in
    /// bounds.
    fn get_mut(self, matrix: &mut T) -> Option<&mut Self::Output>;

    /// Returns a shared reference to the output at this location, without
    /// performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index or a dangling `matrix`
    /// pointer is *[undefined behavior]* even if the resulting reference is not
    /// used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked(self, matrix: *const T) -> *const Self::Output;

    /// Returns a mutable reference to the output at this location, without
    /// performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index or a dangling `matrix`
    /// pointer is *[undefined behavior]* even if the resulting reference is not
    /// used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked_mut(self, stride: *mut T) -> *mut Self::Output;

    /// Returns a shared reference to the output at this location, panicking
    /// if out of bounds.
    #[track_caller]
    fn index(self, stride: &T) -> &Self::Output;

    /// Returns a mutable reference to the output at this location, panicking
    /// if out of bounds.
    #[track_caller]
    fn index_mut(self, stride: &mut T) -> &mut Self::Output;
}

unsafe impl<T, const M: usize, const N: usize> MatrixIndex<Matrix<M, N, T>> for usize {
    type Output = T;

    #[inline]
    fn get(self, matrix: &Matrix<M, N, T>) -> Option<&Self::Output> {
        matrix.as_slice().get(self)
    }

    #[inline]
    fn get_mut(self, matrix: &mut Matrix<M, N, T>) -> Option<&mut Self::Output> {
        matrix.as_mut_slice().get_mut(self)
    }

    #[inline]
    unsafe fn get_unchecked(self, matrix: *const Matrix<M, N, T>) -> *const Self::Output {
        // SAFETY: it is the caller's responsibility not to call this with an
        // out-of-bounds index or a dangling `matrix` pointer.
        let matrix = unsafe { (*matrix).as_slice() };
        unsafe { matrix.get_unchecked(self) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, matrix: *mut Matrix<M, N, T>) -> *mut Self::Output {
        // SAFETY: it is the caller's responsibility not to call this with an
        // out-of-bounds index or a dangling `matrix` pointer.
        let matrix = unsafe { (*matrix).as_mut_slice() };
        unsafe { matrix.get_unchecked_mut(self) }
    }

    #[track_caller]
    #[inline]
    fn index(self, matrix: &Matrix<M, N, T>) -> &Self::Output {
        &matrix.as_slice()[self]
    }

    #[track_caller]
    #[inline]
    fn index_mut(self, matrix: &mut Matrix<M, N, T>) -> &mut Self::Output {
        &mut matrix.as_mut_slice()[self]
    }
}

unsafe impl<T, const M: usize, const N: usize> MatrixIndex<Matrix<M, N, T>> for (usize, usize) {
    type Output = T;

    #[inline]
    fn get(self, matrix: &Matrix<M, N, T>) -> Option<&Self::Output> {
        matrix.as_slice().get(self.1 * M + self.0)
    }

    #[inline]
    fn get_mut(self, matrix: &mut Matrix<M, N, T>) -> Option<&mut Self::Output> {
        matrix.as_mut_slice().get_mut(self.1 * M + self.0)
    }

    #[inline]
    unsafe fn get_unchecked(self, matrix: *const Matrix<M, N, T>) -> *const Self::Output {
        // SAFETY: it is the caller's responsibility not to call this with an
        // out-of-bounds index or a dangling `matrix` pointer.
        let matrix = unsafe { (*matrix).as_slice() };
        unsafe { matrix.get_unchecked(self.1 * M + self.0) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, matrix: *mut Matrix<M, N, T>) -> *mut Self::Output {
        // SAFETY: it is the caller's responsibility not to call this with an
        // out-of-bounds index or a dangling `matrix` pointer.
        let matrix = unsafe { (*matrix).as_mut_slice() };
        unsafe { matrix.get_unchecked_mut(self.1 * M + self.0) }
    }

    #[track_caller]
    #[inline]
    fn index(self, matrix: &Matrix<M, N, T>) -> &Self::Output {
        &matrix.as_slice()[self.1 * M + self.0]
    }

    #[track_caller]
    #[inline]
    fn index_mut(self, matrix: &mut Matrix<M, N, T>) -> &mut Self::Output {
        &mut matrix.as_mut_slice()[self.1 * M + self.0]
    }
}
