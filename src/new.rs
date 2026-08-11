//! Constructors and literal macros for fixed-size matrices.
//!
//! Matrices store columns in column-major order, but [`Matrix::from_rows`]
//! accepts the more familiar row-major notation. The `matrix!` and `vector!`
//! macros are usually the most readable choice in application code; use
//! `from_fn` when values come from an index-based formula.
//!
//! # Example
//!
//! ```
//! use stack_algebra::{diag, eye, matrix, vector, Matrix};
//!
//! let a = matrix![1_i32, 2; 3, 4];
//! let b = Matrix::<2, 2, i32>::from_fn(|row, column| (row + column) as i32);
//! let unit = eye!(3, f32);
//! let diagonal = diag!(1.0_f32, 2.0, 3.0);
//! let point = vector![1.0_f32; 2.0; 3.0];
//! assert_eq!(a[(1, 0)], 3);
//! assert_eq!(b[(1, 1)], 2);
//! assert_eq!(unit[(2, 2)], 1.0);
//! assert_eq!(diagonal[(1, 1)], 2.0);
//! assert_eq!(point[2], 3.0);
//! ```

use crate::num::{One, Zero};
use crate::Matrix;

use core::mem;
use core::mem::MaybeUninit;
use core::ptr;

////////////////////////////////////////////////////////////////////////////////
// Matrix<M,N,T> methods
////////////////////////////////////////////////////////////////////////////////
impl<const M: usize, const N: usize, T> Matrix<M, N, T> {
    /// Creates a new matrix from its columns in column-major order.
    ///
    /// The outer array has `N` columns and each inner array has `M` rows. For
    /// row-major input, prefer [`Matrix::from_rows`] or [`matrix!`](crate::matrix!).
    #[inline]
    pub const fn from_columns(data: [[T; M]; N]) -> Self {
        Self { data }
    }

    /// Creates a new matrix by evaluating `f` for each `(row, column)` pair.
    ///
    /// The closure is called exactly `M * N` times and the result is fully
    /// initialized without an intermediate heap allocation.
    #[inline]
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut matrix = Matrix::<M, N, MaybeUninit<T>>::uninit();
        for column in 0..N {
            for row in 0..M {
                matrix[(row, column)].write(f(row, column));
            }
        }

        // SAFETY: every matrix element is initialized exactly once above.
        unsafe { matrix.assume_init() }
    }

    /// Creates a new matrix from rows in row-major order.
    ///
    /// This is often the clearest constructor for hand-written constants.
    #[inline]
    pub fn from_rows(data: [[T; N]; M]) -> Self
    where
        T: Copy,
    {
        Self::from_fn(|row, column| data[row][column])
    }

    /// Creates a new matrix from an array of arrays in column-major order.
    #[doc(hidden)]
    #[inline]
    pub const fn from_column_major_order(data: [[T; M]; N]) -> Self {
        Self::from_columns(data)
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
        Self::from_columns([[T::zero(); M]; N])
    }

    /// Initializes zero-valued matrix storage directly in caller-owned memory.
    ///
    /// This is useful for heap-allocated fixed-capacity workspaces because it avoids constructing
    /// the full matrix as an intermediate return value.
    pub fn zeros_into(output: &mut MaybeUninit<Self>) {
        // SAFETY: the matrix data is the only field and is initialized exactly once.
        unsafe {
            let data = ptr::addr_of_mut!((*output.as_mut_ptr()).data).cast::<T>();
            for index in 0..(M * N) {
                data.add(index).write(T::zero());
            }
        }
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
        Self::from_columns([[T::one(); M]; N])
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
        let mut m = Self::from_columns([[T::zero(); D]; D]);
        for i in 0..D {
            m[(i, i)] = T::one();
        }
        m
    }
}

/// Creates a matrix from row-major literal syntax.
///
/// A semicolon separates rows; commas separate values within each row. The
/// dimensions and scalar type are inferred from the literal.
///
/// ```
/// use stack_algebra::{matrix, Matrix};
/// let m: Matrix<2, 3, i32> = matrix![1, 2, 3; 4, 5, 6];
/// assert_eq!(m[(1, 2)], 6);
/// ```
#[macro_export]
macro_rules! matrix {
    ($($data:tt)*) => {
        $crate::Matrix::from_columns($crate::proc_macro::matrix!($($data)*))
    };
}

/// Creates a row or column vector from literal syntax.
///
/// Commas produce a one-row [`RowVector`](crate::RowVector); semicolons
/// produce a one-column [`Vector`](crate::Vector).
///
/// ```
/// use stack_algebra::{vector, RowVector, Vector};
/// let row: RowVector<3, i32> = vector![1, 2, 3];
/// let column: Vector<3, i32> = vector![1; 2; 3];
/// assert_eq!(row[(0, 2)], 3);
/// assert_eq!(column[2], 3);
/// ```
#[macro_export]
macro_rules! vector {
    ($($data:tt)*) => {
        $crate::Matrix::from_columns($crate::proc_macro::matrix!($($data)*))
    };
}

/// Creates a zero-filled square or rectangular matrix.
///
/// `zeros!(n)` creates `n`-by-`n`; `zeros!(rows, columns)` creates a
/// rectangular matrix; and `zeros!(rows, columns, Scalar)` selects the scalar
/// type explicitly.
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

/// Creates a one-filled square or rectangular matrix.
///
/// The argument forms mirror [`zeros!`].
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

/// Creates an identity matrix with ones on the main diagonal.
///
/// The matrix is always square. Pass a scalar type as the second argument when
/// inference cannot determine whether `f32` or `f64` is desired.
#[macro_export]
macro_rules! eye {
    ($dim:expr) => {
        $crate::Matrix::<$dim, $dim>::eye()
    };
    ($dim:expr, $ty:ty) => {{
        $crate::Matrix::<$dim, $dim, $ty>::eye()
    }};
}

/// Creates a diagonal matrix for two through six diagonal values.
///
/// Values not supplied on the diagonal are zero. This macro is intended for
/// small fixed-size matrices; use [`Matrix::from_fn`] for larger dimensions.
#[macro_export]
macro_rules! diag {
    ($d1:expr, $d2:expr) => {{
        let mut m = $crate::Matrix::<2, 2>::zeros();
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
    ($d1:expr, $d2:expr, $d3:expr, $d4:expr) => {{
        let mut m = $crate::Matrix::<4, 4>::zeros();
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
}

////////////////////////////////////////////////////////////////////////////////
// Uninit related methods
////////////////////////////////////////////////////////////////////////////////

/// Size-heterogeneous transmutation.
///
/// This is required because the compiler doesn't yet know how to deal with the
/// size of const arrays. We should be able to use [`mem::transmute()`] but it
/// doesn't work yet :(.
///
/// # Safety
///
/// The caller must ensure that `A` and `B` have identical sizes and compatible
/// alignments, and that the bit pattern of `a` is valid for `B`. Ownership of
/// `a` is consumed; its destructor is intentionally skipped.
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
        // require initialization. Additionally, `Matrix` is `repr(C)` with an
        // array-of-arrays representation.
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
    /// guarantee that the matrix is really in an initialized state. Calling
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

        let d = diag!(0.1, 0.2, 0.3);
        let e = matrix![
        0.1, 0.0, 0.0;
        0.0, 0.2, 0.0;
        0.0, 0.0, 0.3;
        ];
        assert_relative_eq!(d, e, max_relative = 1e-6);

        let d = diag!(0.1, 0.2, 0.3, 0.4);
        let e = matrix![
        0.1, 0.0, 0.0, 0.0;
        0.0, 0.2, 0.0, 0.0;
        0.0, 0.0, 0.3, 0.0;
        0.0, 0.0, 0.0, 0.4;
        ];
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
    }
}
