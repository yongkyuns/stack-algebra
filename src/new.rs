use crate::num::{One, Zero};
use crate::Matrix;

use core::hint;
use core::mem;
use core::mem::MaybeUninit;
use core::ptr;

////////////////////////////////////////////////////////////////////////////////
// Matrix<M,N,T> methods
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

impl<const D: usize, T> Matrix<D, D, T>
where
    T: Zero + One + Copy,
{
    /// Create a new matrix from an array of arrays in column-major order.
    #[doc(hidden)]
    #[inline]
    pub fn eye() -> Self {
        let mut m = Self::from_column_major_order([[T::zero(); D]; D]);
        for i in 0..D {
            m[(i, i)] = T::one();
        }
        m
    }
}

/// A macro for creating a matrix.
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

#[macro_export]
macro_rules! eye {
    ($dim:expr) => {
        $crate::Matrix::<$dim, $dim>::eye()
    };
    ($dim:expr, $ty:ty) => {{
        $crate::Matrix::<$dim, $dim, $ty>::eye()
    }};
}

#[macro_export]
macro_rules! diag {
    ($d1:expr, $d2:expr) => {{
        let mut m = $crate::Matrix::<2, 2>::zeros();
        m[(0, 0)] = $d1;
        m[(1, 1)] = $d2;
        m
    }};
    ($d1:expr, $d2:expr, $ty:ty) => {{
        let mut m = $crate::Matrix::<2, 2, $ty>::zeros();
        m[(0, 0)] = $d1;
        m[(1, 1)] = $d2;
        m
    }};
    ($d1:expr, $d2:expr, $d3:expr) => {{
        let mut m = $crate::Matrix::<3, 3>::zeros();
        m[(0, 0)] = $d1;
        m[(1, 1)] = $d2;
        m[(2, 2)] = $d3;
        m
    }};
    ($d1:expr, $d2:expr, $d3:expr, $ty:ty) => {{
        let mut m = $crate::Matrix::<3, 3, $ty>::zeros();
        m[(0, 0)] = $d1;
        m[(1, 1)] = $d2;
        m[(2, 2)] = $d3;
        m
    }};
    ($d1:expr, $d2:expr, $d3:expr, $d4:expr) => {{
        let mut m = $crate::Matrix::<4, 4>::zeros();
        m[(0, 0)] = $d1;
        m[(1, 1)] = $d2;
        m[(2, 2)] = $d3;
        m[(3, 3)] = $d4;
        m
    }};
    ($d1:expr, $d2:expr, $d3:expr, $d4:expr, $ty:ty) => {{
        let mut m = $crate::Matrix::<4, 4, $ty>::zeros();
        m[(0, 0)] = $d1;
        m[(1, 1)] = $d2;
        m[(2, 2)] = $d3;
        m[(3, 3)] = $d4;
        m
    }};
    ($d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr) => {{
        let mut m = $crate::Matrix::<5, 5>::zeros();
        m[(0, 0)] = $d1;
        m[(1, 1)] = $d2;
        m[(2, 2)] = $d3;
        m[(3, 3)] = $d4;
        m[(4, 4)] = $d5;
        m
    }};
    ($d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr, $ty:ty) => {{
        let mut m = $crate::Matrix::<5, 5, $ty>::zeros();
        m[(0, 0)] = $d1;
        m[(1, 1)] = $d2;
        m[(2, 2)] = $d3;
        m[(3, 3)] = $d4;
        m[(4, 4)] = $d5;
        m
    }};
    ($d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr, $d6:expr) => {{
        let mut m = $crate::Matrix::<6, 6>::zeros();
        m[(0, 0)] = $d1;
        m[(1, 1)] = $d2;
        m[(2, 2)] = $d3;
        m[(3, 3)] = $d4;
        m[(4, 4)] = $d5;
        m[(5, 5)] = $d6;
        m
    }};
    ($d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr, $d6:expr, $ty:ty) => {{
        let mut m = $crate::Matrix::<6, 6, $ty>::zeros();
        m[(0, 0)] = $d1;
        m[(1, 1)] = $d2;
        m[(2, 2)] = $d3;
        m[(3, 3)] = $d4;
        m[(4, 4)] = $d5;
        m[(5, 5)] = $d6;
        m
    }};
}

////////////////////////////////////////////////////////////////////////////////
// Uninit related methods
////////////////////////////////////////////////////////////////////////////////

/// Size-heterogeneous transmutation.
///
/// This is required because the compiler doesn't yet know how to deal with the
/// size of const arrays. We should be able to use [`mem::transmute()`] but it
/// doesn't work yet :(.
#[inline]
pub unsafe fn transmute_unchecked<A, B>(a: A) -> B {
    let b = unsafe { ptr::read(&a as *const A as *const B) };
    mem::forget(a);
    b
}

impl<T, const M: usize, const N: usize> Matrix<M, N, MaybeUninit<T>> {
    /// Create a new matrix with uninitialized contents.
    #[inline]
    pub(crate) fn uninit() -> Self {
        // SAFETY: The `assume_init` is safe because the type we are claiming to
        // have initialized here is a bunch of `MaybeUninit`s, which do not
        // require initialization. Additionally, `Matrix` is `repr(transparent)`
        // with an array of arrays.
        //
        // Note: this is not the most ideal way of doing this. In the future
        // when Rust allows inline const expressions we might be able to use
        // `Self { data: [const { MaybeUninit::<T>::uninit() }; M] ; N] }`
        //
        // See https://doc.rust-lang.org/std/mem/union.MaybeUninit.html#initializing-an-array-element-by-element
        let matrix = MaybeUninit::uninit();
        unsafe { matrix.assume_init() }
    }

    /// Assumes the data is initialized and extracts each element as `T`.
    ///
    /// # Safety
    ///
    /// As with [`MaybeUninit::assume_init`], it is up to the caller to
    /// guarantee that the matrix is really is in an initialized state. Calling
    /// this when the contents are not yet fully initialized causes immediate
    /// undefined behavior.
    #[inline]
    pub(crate) unsafe fn assume_init(self) -> Matrix<M, N, T> {
        // SAFETY: The caller is responsible for all the elements being
        // initialized. Additionally, we know that `T` is the same size as
        // `MaybeUninit<T>`.
        unsafe { transmute_unchecked(self) }
    }
}

////////////////////////////////////////////////////////////////////////////////
// FromIterator
////////////////////////////////////////////////////////////////////////////////

/// Pulls `M * N` items from `iter` and fills a matrix. If the iterator yields
/// fewer than `M * N` items, `Err(_)` is returned and all already yielded items
/// are dropped.
///
/// If `iter.next()` panics, all items already yielded by the iterator are
/// dropped.
pub fn collect<I, T, const M: usize, const N: usize>(mut iter: I) -> Result<Matrix<M, N, T>, usize>
where
    I: Iterator<Item = T>,
{
    struct Guard<'a, T, const M: usize, const N: usize> {
        matrix: &'a mut Matrix<M, N, MaybeUninit<T>>,
        init: usize,
    }

    impl<T, const M: usize, const N: usize> Drop for Guard<'_, T, M, N> {
        fn drop(&mut self) {
            for elem in &mut self.matrix.as_mut_slice()[..self.init] {
                // SAFETY: this raw slice up to `self.len` will only contain
                // the initialized objects.
                unsafe { ptr::drop_in_place(elem.as_mut_ptr()) };
            }
        }
    }

    let mut matrix: Matrix<M, N, MaybeUninit<T>> = Matrix::uninit();
    let mut guard = Guard {
        matrix: &mut matrix,
        init: 0,
    };

    for _ in 0..(M * N) {
        match iter.next() {
            Some(item) => {
                // SAFETY: `guard.init` starts at zero, is increased by 1 each
                // iteration of the loop, and the loop is aborted once M * N
                // is reached, which is the length of the matrix.
                unsafe { guard.matrix.get_unchecked_mut(guard.init).write(item) };
                guard.init += 1;
            }
            None => {
                return Err(guard.init);
                // <-- guard is dropped here with already initialized elements
            }
        }
    }

    mem::forget(guard);
    // SAFETY: the loop above loops exactly M * N times which is the size of the
    // matrix, so all elements in the matrix are initialized.
    Ok(unsafe { matrix.assume_init() })
}

/// Like [`collect()`] except the caller must guarantee that the iterator will
/// yield enough elements to fill the matrix.
pub unsafe fn collect_unchecked<I, T, const M: usize, const N: usize>(iter: I) -> Matrix<M, N, T>
where
    I: IntoIterator<Item = T>,
{
    match collect(iter.into_iter()) {
        Ok(matrix) => matrix,
        Err(_) => {
            // SAFETY: the caller guarantees the iterator will yield enough
            // elements, so this error case can never be reached.
            unsafe { hint::unreachable_unchecked() }
        }
    }
}

impl<T, const M: usize, const N: usize> FromIterator<T> for Matrix<M, N, T> {
    /// Create a new matrix from an iterator.
    ///
    /// Elements will be filled in column-major order.
    ///
    /// # Panics
    ///
    /// If the iterator doesn't yield enough elements to fill the matrix.
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        collect(iter.into_iter()).unwrap_or_else(|len| collect_panic::<M, N>(len))
    }
}

#[cold]
fn collect_panic<const M: usize, const N: usize>(len: usize) -> ! {
    if N == 1 {
        panic!("collect iterator of length {} into `Vector<_, {}>`", len, M);
    } else if M == 1 {
        panic!(
            "collect iterator of length {} into `RowVector<_, {}>`",
            len, N
        );
    } else {
        panic!(
            "collect iterator of length {} into `Matrix<_, {}, {}>`",
            len, M, N
        );
    }
}

#[cfg(test)]
mod new_test {
    use approx::assert_relative_eq;
    #[test]
    fn diag() {
        let d = diag!(0.1, 0.2);
        let e = matrix![
        0.1, 0.0;
        0.0, 0.2;
        ];
        assert_relative_eq!(d, e, max_relative = 1e-6);
        let d = diag!(0.1, 0.2, f32);
        assert_relative_eq!(d, e, max_relative = 1e-6);

        let d = diag!(0.1, 0.2, 0.3);
        let e = matrix![
        0.1, 0.0, 0.0;
        0.0, 0.2, 0.0;
        0.0, 0.0, 0.3;
        ];
        assert_relative_eq!(d, e, max_relative = 1e-6);
        let d = diag!(0.1, 0.2, 0.3, f32);
        assert_relative_eq!(d, e, max_relative = 1e-6);

        let d = diag!(0.1, 0.2, 0.3, 0.4);
        let e = matrix![
        0.1, 0.0, 0.0, 0.0;
        0.0, 0.2, 0.0, 0.0;
        0.0, 0.0, 0.3, 0.0;
        0.0, 0.0, 0.0, 0.4;
        ];
        assert_relative_eq!(d, e, max_relative = 1e-6);
        let d = diag!(0.1, 0.2, 0.3, 0.4, f32);
        assert_relative_eq!(d, e, max_relative = 1e-6);

        let d = diag!(0.1, 0.2, 0.3, 0.4, 0.5);
        let e = matrix![
        0.1, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.2, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.3, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.4, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.5;
        ];
        assert_relative_eq!(d, e, max_relative = 1e-6);
        let d = diag!(0.1, 0.2, 0.3, 0.4, 0.5, f32);
        assert_relative_eq!(d, e, max_relative = 1e-6);

        let d = diag!(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        let e = matrix![
        0.1, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.2, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.3, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.4, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.5, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 0.6;
        ];
        assert_relative_eq!(d, e, max_relative = 1e-6);
        let d = diag!(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, f32);
        assert_relative_eq!(d, e, max_relative = 1e-6);
    }
}
