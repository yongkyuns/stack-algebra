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
