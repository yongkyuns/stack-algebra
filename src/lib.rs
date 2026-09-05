#![no_std]
#![deny(missing_docs)]
//! Fixed-size, inline-storage linear algebra for `no_std` Rust.
//!
//! [`Matrix`] stores its dimensions in const generics, so a value such as
//! `Matrix<3, 3, f64>` has no run-time shape metadata and no heap allocation.
//! The scalar type is the final generic parameter (`f32` is the default); use
//! `f64` explicitly when double precision is required.
//!
//! # Quick start
//!
//! ```
//! use stack_algebra::{matrix, Matrix};
//!
//! let lhs: Matrix<2, 3, f32> = matrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0];
//! let rhs: Matrix<3, 1, f32> = matrix![1.0; 0.0; -1.0];
//! let product = lhs * rhs;
//! assert_eq!(product, matrix![-2.0; -2.0]);
//!
//! let precise: Matrix<2, 2, f64> = Matrix::eye();
//! assert_eq!(precise[(0, 0)], 1.0);
//! ```
//!
//! Matrix values are column-major, matching common numerical and Eigen-style
//! kernels. Use `from_rows` for readable row-major literals, `from_columns`
//! when data already comes from a column-major buffer, or `Map`/`StridedMap`
//! to borrow external storage without copying.
//!
//! # Compile-time shape and scalar checks
//!
//! Matrix multiplication only accepts matching inner dimensions:
//!
//! ```compile_fail
//! use stack_algebra::{matrix, Matrix};
//!
//! let lhs: Matrix<2, 3, f32> = matrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0];
//! let rhs: Matrix<2, 2, f32> = matrix![1.0, 2.0; 3.0, 4.0];
//! let _ = lhs * rhs;
//! ```
//!
//! Arithmetic does not implicitly mix scalar types. Use [`Matrix::cast`] when
//! an explicit conversion is intended:
//!
//! ```compile_fail
//! use stack_algebra::{matrix, Matrix};
//!
//! let single: Matrix<2, 2, f32> = matrix![1.0, 0.0; 0.0, 1.0];
//! let double: Matrix<2, 2, f64> = matrix![1.0, 0.0; 0.0, 1.0];
//! let _ = single * double;
//! ```
//!
//! Decomposition solves also require a right-hand side with the factor's
//! compile-time row count:
//!
//! ```compile_fail
//! use stack_algebra::{matrix, Matrix};
//!
//! let coefficient: Matrix<2, 2, f64> = matrix![2.0, 0.0; 0.0, 3.0];
//! let factor = coefficient.try_cholesky().unwrap();
//! let rhs: Matrix<3, 1, f64> = matrix![1.0; 2.0; 3.0];
//! let _ = factor.solve(&rhs);
//! ```

mod algebra;
mod block_sparse;
mod bounded;
mod fmt;
mod geometry;
mod index;
mod iter;
mod kernels;
mod new;
mod num;
mod ops;
mod sparse;
mod util;
mod view;

use core::{
    ops::{Add, Mul, Sub},
    slice,
};

pub use algebra::{
    Cholesky, ColPivHouseholderQr, DecompositionError, HouseholderQr, Ldlt, LowerTriangular,
    PartialPivLu, SelfAdjointEigen, SelfAdjointEigenWorkspace, SelfAdjointLower, SelfAdjointUpper,
    SelfAdjointView, Svd, UpperTriangular,
};
pub use block_sparse::{
    StaticBlockCscCholesky, StaticBlockCscCholeskyPattern, StaticBlockCscLdlt,
    StaticBlockCscLdltPattern, StaticBlockCscMatrix, StaticBlockCsrMatrix,
};
pub use bounded::{MatrixBuf, MatrixBufError, MatrixBufView, MatrixBufViewMut};
pub use geometry::{AffineTransform, AngleAxis, Isometry, Quaternion, RotationMatrix};
pub use index::MatrixIndex;
pub use kernels::{FactorizationScalar, MatrixScalar, ReductionScalar};
pub use num::{AsPrimitive, Float, One, Real, Zero};
pub use ops::{matmul_view_into, matvec_view, matvec_view_into};
pub use sparse::{
    CscError, SparseCholeskyError, StaticCscCholesky, StaticCscCholeskyPattern, StaticCscLdlt,
    StaticCscLdltFactor, StaticCscLdltPattern, StaticCscMatrix, StaticCscOrdering,
    StaticCscPattern, StaticCscPermutation,
};
pub use view::{
    Block, BlockMut, Column, Map, MapMut, MatrixRead, MatrixWrite, Row, StrideAxis, StridedMap,
    StridedMapMut, ViewError,
};

#[doc(hidden)]
pub use vectrix_macro as proc_macro;

/// A fixed-size matrix with `M` rows and `N` columns.
///
/// The underlying data is represented as an array and is always stored in
/// column-major order. `M` and `N` are compile-time dimensions, while `T` is
/// the scalar type (`f32` by default). Matrix operations therefore remain
/// allocation-free and dimension mismatches are rejected by the compiler.
///
/// # Examples
///
/// ```
/// use stack_algebra::{matrix, Matrix};
///
/// let a: Matrix<2, 2, f64> = matrix![2.0, 1.0; 1.0, 3.0];
/// let b = Matrix::<2, 2, f64>::eye();
/// let product = a * b;
/// assert_eq!(product, a);
/// assert_eq!(a.as_slice(), &[2.0, 1.0, 1.0, 3.0]);
/// ```
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Matrix<const M: usize, const N: usize, T = f32> {
    data: [[T; M]; N],
}

impl<const M: usize, const N: usize, T> Matrix<M, N, T> {
    /// Returns the exact inline storage footprint of this fixed-size matrix.
    ///
    /// This is `M * N * size_of::<T>()` for the current representation and is
    /// useful when budgeting stack or embedded static storage.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Copies a fixed-size matrix view into an owning matrix.
    ///
    /// This is useful when a borrowed `Map`, `StridedMap`, or block needs to
    /// be retained after the source storage is released.
    #[inline]
    pub fn from_view<V>(view: &V) -> Self
    where
        T: Copy,
        V: view::MatrixRead<M, N, T>,
    {
        Self::from_fn(|row, column| {
            *view
                .get(row, column)
                .expect("matrix view dimensions match the destination")
        })
    }

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

    /// Views the underlying data as a contiguous column-major slice.
    ///
    /// Element `(row, column)` is at `column * M + row`.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `Matrix` is `repr(C)` with an array-of-arrays layout, so its
        // elements are contiguous in column-major order and initialized.
        unsafe { slice::from_raw_parts(self.as_ptr(), M * N) }
    }

    /// Views the underlying data as a mutable contiguous column-major slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `Matrix` is `repr(C)` with an array-of-arrays layout, so its
        // elements are contiguous in column-major order and initialized. The
        // exclusive borrow guarantees unique access for the returned slice.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), M * N) }
    }

    /// Converts every matrix element to `U` using Rust's primitive cast semantics.
    ///
    /// This is an explicit conversion; arithmetic never mixes scalar types
    /// implicitly. For example, convert an `f32` matrix to `f64` with
    /// `let wide: Matrix<2, 2, f64> = matrix.cast();`.
    #[inline]
    pub fn cast<U>(&self) -> Matrix<M, N, U>
    where
        T: Copy + AsPrimitive<U>,
        U: 'static + Copy,
    {
        Matrix::from_fn(|row, column| self[(row, column)].as_())
    }

    /// Returns a reference to the `i`-th row of this matrix.
    ///
    /// Rows are strided in column-major storage; use [`Self::as_slice`] when a
    /// contiguous buffer is required.
    #[inline]
    pub fn row(&self, i: usize) -> &Row<M, N, T> {
        assert!(i < M, "row index out of bounds");
        Row::new(&self.as_slice()[i..])
    }

    /// Returns a mutable reference to the `i`-th row of this matrix.
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut Row<M, N, T> {
        assert!(i < M, "row index out of bounds");
        Row::new_mut(&mut self.as_mut_slice()[i..])
    }

    /// Returns a reference to the `i`-th column of this matrix.
    ///
    /// A column is contiguous and can be passed to APIs that consume a slice.
    #[inline]
    pub fn column(&self, i: usize) -> &Column<M, N, T> {
        Column::new(&self.data[i])
    }

    /// Returns a mutable reference to the `i`-th column of this matrix.
    #[inline]
    pub fn column_mut(&mut self, i: usize) -> &mut Column<M, N, T> {
        Column::new_mut(&mut self.data[i])
    }

    /// Returns a fixed-size block view, or `None` when it exceeds the matrix.
    ///
    /// The returned view borrows `self` and preserves compile-time block
    /// dimensions; no elements are copied.
    #[inline]
    pub fn block<const R: usize, const C: usize>(
        &self,
        row_offset: usize,
        column_offset: usize,
    ) -> Option<Block<'_, M, N, R, C, T>> {
        let row_end = row_offset.checked_add(R)?;
        let column_end = column_offset.checked_add(C)?;
        if row_end <= M && column_end <= N {
            Some(Block::new(self, row_offset, column_offset))
        } else {
            None
        }
    }

    /// Returns a mutable fixed-size block view, or `None` when it exceeds the matrix.
    #[inline]
    pub fn block_mut<const R: usize, const C: usize>(
        &mut self,
        row_offset: usize,
        column_offset: usize,
    ) -> Option<BlockMut<'_, M, N, R, C, T>> {
        let row_end = row_offset.checked_add(R)?;
        let column_end = column_offset.checked_add(C)?;
        if row_end <= M && column_end <= N {
            Some(BlockMut::new(self, row_offset, column_offset))
        } else {
            None
        }
    }

    /// Returns a reference to an element in the matrix or `None` if out of
    /// bounds.
    #[inline]
    pub fn get<I>(&self, i: I) -> Option<&I::Output>
    where
        I: MatrixIndex<Self>,
    {
        i.get(self)
    }

    /// Returns a mutable reference to an element in the matrix or `None` if out
    /// of bounds.
    #[inline]
    pub fn get_mut<I>(&mut self, i: I) -> Option<&mut I::Output>
    where
        I: MatrixIndex<Self>,
    {
        i.get_mut(self)
    }

    /// Returns a reference to an element in the matrix without doing any bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// *[undefined behavior]* even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub unsafe fn get_unchecked<I>(&self, i: I) -> &I::Output
    where
        I: MatrixIndex<Self>,
    {
        unsafe { &*i.get_unchecked(self) }
    }

    /// Returns a mutable reference to an element in the matrix without doing
    /// any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// *[undefined behavior]* even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub unsafe fn get_unchecked_mut<I>(&mut self, i: I) -> &mut I::Output
    where
        I: MatrixIndex<Self>,
    {
        unsafe { &mut *i.get_unchecked_mut(self) }
    }

    /// Returns an iterator over the underlying data.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the underlying data.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// Swap the two given rows of this matrix
    #[inline]
    pub fn swap_rows(&mut self, r1: usize, r2: usize)
    where
        T: Copy,
    {
        if r1 < M && r2 < M {
            for i in 0..N {
                let tmp = self[(r1, i)];
                self[(r1, i)] = self[(r2, i)];
                self[(r2, i)] = tmp;
            }
        }
    }

    /// Swap the two given columns of this matrix
    #[inline]
    pub fn swap_columns(&mut self, c1: usize, c2: usize)
    where
        T: Copy,
    {
        if c1 < N && c2 < N {
            for i in 0..M {
                let tmp = self[(i, c1)];
                self[(i, c1)] = self[(i, c2)];
                self[(i, c2)] = tmp;
            }
        }
    }

    /// Returns a transposed copy of the matrix.
    #[inline]
    pub fn transpose(&self) -> Matrix<N, M, T>
    where
        T: Clone,
    {
        Matrix::from_fn(|row, column| self[(column, row)].clone())
    }

    /// Writes the transpose of this matrix into `output`.
    #[inline]
    pub fn transpose_into(&self, output: &mut Matrix<N, M, T>)
    where
        T: Clone,
    {
        for column in 0..N {
            for row in 0..M {
                output[(column, row)] = self[(row, column)].clone();
            }
        }
    }

    /// Shorthand for [`Self::transpose`].
    #[allow(non_snake_case)]
    #[inline]
    pub fn T(&self) -> Matrix<N, M, T>
    where
        T: Clone,
    {
        self.transpose()
    }

    /// Returns the raw sum of squares of all entries.
    ///
    /// This is the direct `aᵢⱼ²` accumulation and can overflow or underflow for
    /// extreme finite values. Use [`Self::norm`] when a scale-stable magnitude
    /// is required.
    #[inline]
    pub fn squared_norm(&self) -> T
    where
        T: Real + ReductionScalar,
    {
        T::squared_norm(self)
    }

    /// Returns the Frobenius norm, `sqrt(sum(aᵢⱼ²))`.
    ///
    /// The reduction is scaled during accumulation, so finite inputs avoid
    /// intermediate overflow and underflow where the mathematical result is
    /// representable.
    #[inline]
    pub fn norm(&self) -> T
    where
        T: Real + ReductionScalar,
    {
        T::norm(self)
    }

    /// Returns a normalized copy divided by the Frobenius norm.
    pub fn normalize(self) -> Self
    where
        T: Real + ReductionScalar,
    {
        self / self.norm()
    }
}

impl<const M: usize, T> Matrix<M, 1, T>
where
    T: ReductionScalar,
{
    /// Computes the dot product of two fixed-size vectors.
    ///
    /// Both vectors must have the same compile-time length and scalar type.
    #[inline]
    pub fn dot(&self, other: &Self) -> T {
        T::dot(self, other)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Square matrix functions
////////////////////////////////////////////////////////////////////////////////
impl<const N: usize, T> Matrix<N, N, T> {
    /// Computes the sum of the diagonal elements.
    pub fn trace(&self) -> T
    where
        T: Zero,
        for<'a> &'a T: Add<&'a T, Output = T>,
    {
        let mut t = T::zero();
        for i in 0..N {
            t = &t + &self[(i, i)];
        }
        t
    }
}

impl<T> Matrix<3, 1, T> {
    /// Computes the 3D cross product.
    pub fn cross(&self, other: &Self) -> Self
    where
        for<'a> &'a T: Mul<&'a T, Output = T> + Sub<&'a T, Output = T>,
    {
        Self::from_columns([[
            &(&self[1] * &other[2]) - &(&self[2] * &other[1]),
            &(&self[2] * &other[0]) - &(&self[0] * &other[2]),
            &(&self[0] * &other[1]) - &(&self[1] * &other[0]),
        ]])
    }
}

/// Computes the 3D cross product of two column vectors.
///
/// This free function is equivalent to [`Matrix::cross`] and is convenient in
/// generic code where the method form would be less readable.
pub fn cross<T>(a: &Matrix<3, 1, T>, b: &Matrix<3, 1, T>) -> Matrix<3, 1, T>
where
    for<'a> &'a T: Mul<&'a T, Output = T> + Sub<&'a T, Output = T>,
{
    a.cross(b)
}

////////////////////////////////////////////////////////////////////////////////
// 3D/4D Vector Type Conversion to Tuple
////////////////////////////////////////////////////////////////////////////////

impl<T: Copy> From<(T, T, T)> for Matrix<3, 1, T> {
    fn from(src: (T, T, T)) -> Self {
        matrix![src.0; src.1; src.2]
    }
}

impl<T: Copy> From<(T, T, T)> for Matrix<1, 3, T> {
    fn from(src: (T, T, T)) -> Self {
        matrix![src.0, src.1, src.2]
    }
}

impl<T: Copy> From<Matrix<3, 1, T>> for (T, T, T) {
    fn from(src: Matrix<3, 1, T>) -> Self {
        (src[0], src[1], src[2])
    }
}

impl<T: Copy> From<Matrix<1, 3, T>> for (T, T, T) {
    fn from(src: Matrix<1, 3, T>) -> Self {
        (src[0], src[1], src[2])
    }
}

impl<T: Copy> From<(T, T, T, T)> for Matrix<4, 1, T> {
    fn from(src: (T, T, T, T)) -> Self {
        matrix![src.0; src.1; src.2; src.3]
    }
}

impl<T: Copy> From<(T, T, T, T)> for Matrix<1, 4, T> {
    fn from(src: (T, T, T, T)) -> Self {
        matrix![src.0, src.1, src.2, src.3]
    }
}

impl<T: Copy> From<Matrix<4, 1, T>> for (T, T, T, T) {
    fn from(src: Matrix<4, 1, T>) -> Self {
        (src[0], src[1], src[2], src[3])
    }
}

impl<T: Copy> From<Matrix<1, 4, T>> for (T, T, T, T) {
    fn from(src: Matrix<1, 4, T>) -> Self {
        (src[0], src[1], src[2], src[3])
    }
}

// #[cfg(test)]
impl<const M: usize, const N: usize, T: approx::AbsDiffEq> approx::AbsDiffEq for Matrix<M, N, T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let mut eq = true;
        for j in 0..N {
            for i in 0..M {
                eq = eq && T::abs_diff_eq(&self[(i, j)], &other[(i, j)], epsilon);
                if !eq {
                    return false;
                }
            }
        }
        true
    }
}

// #[cfg(test)]
impl<const M: usize, const N: usize, T: approx::RelativeEq> approx::RelativeEq for Matrix<M, N, T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        let mut eq = true;
        for j in 0..N {
            for i in 0..M {
                eq = eq && T::relative_eq(&self[(i, j)], &other[(i, j)], epsilon, max_relative);
                if !eq {
                    return false;
                }
            }
        }
        true
    }
}

/// A matrix with one row and `N` columns.
pub type RowVector<const N: usize, T = f32> = Matrix<1, N, T>;

/// A matrix with one column and `M` rows.
pub type Vector<const M: usize, T = f32> = Matrix<M, 1, T>;

/// A 2-by-2 matrix.
pub type Matrix2<T = f32> = Matrix<2, 2, T>;
/// A 3-by-3 matrix.
pub type Matrix3<T = f32> = Matrix<3, 3, T>;
/// A 4-by-4 matrix.
pub type Matrix4<T = f32> = Matrix<4, 4, T>;
/// A 2-by-2 matrix of `f32` values.
pub type Matrix2f = Matrix2<f32>;
/// A 3-by-3 matrix of `f32` values.
pub type Matrix3f = Matrix3<f32>;
/// A 4-by-4 matrix of `f32` values.
pub type Matrix4f = Matrix4<f32>;
/// A 2-by-2 matrix of `f64` values.
pub type Matrix2d = Matrix2<f64>;
/// A 3-by-3 matrix of `f64` values.
pub type Matrix3d = Matrix3<f64>;
/// A 4-by-4 matrix of `f64` values.
pub type Matrix4d = Matrix4<f64>;
/// A 3-element vector of `f32` values.
pub type Vector3f = Vector<3, f32>;
/// A 3-element vector of `f64` values.
pub type Vector3d = Vector<3, f64>;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn create() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 2)], 6.0);

        let v = vector![1.0, 2.0, 3.0];
        assert_eq!(v[0], 1.0);
        assert_eq!(v[2], 3.0);

        let z = zeros!(2, 3);
        assert_eq!(z[(0, 0)], 0.0);
        assert_eq!(z[(1, 2)], 0.0);

        let z = zeros!(3);
        assert_eq!(z[(2, 2)], 0.0);

        let o = ones!(2, 3);
        assert_eq!(o[(0, 0)], 1.0);
        assert_eq!(o[(1, 2)], 1.0);

        let o = ones!(3);
        assert_eq!(o[(2, 2)], 1.0);
    }

    #[test]
    fn constructors_and_casts_preserve_layout() {
        let matrix = Matrix::<2, 3, f32>::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(matrix.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let generated = Matrix::<2, 3, i32>::from_fn(|row, column| (row * 10 + column) as i32);
        assert_eq!(generated[(1, 2)], 12);

        let widened: Matrix<2, 3, f64> = matrix.cast();
        assert_eq!(widened[(1, 2)], 6.0);
        let narrowed: Matrix<2, 3, f32> = widened.cast();
        assert_eq!(narrowed, matrix);
    }

    #[test]
    #[should_panic(expected = "row index out of bounds")]
    fn row_rejects_out_of_bounds_index() {
        let matrix = Matrix::<2, 2, i32>::zeros();
        let _ = matrix.row(2);
    }

    #[test]
    #[should_panic(expected = "row index out of bounds")]
    fn row_mut_rejects_out_of_bounds_index() {
        let mut matrix = Matrix::<2, 2, i32>::zeros();
        let _ = matrix.row_mut(2);
    }

    #[test]
    fn mul_into_matches_operator_for_rectangular_matrices() {
        let lhs = Matrix::<2, 3, f64>::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let rhs = Matrix::<3, 2, f64>::from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
        let mut output = Matrix::<2, 2, f64>::ones();
        lhs.mul_into(&rhs, &mut output);
        assert_eq!(output, lhs * rhs);
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

    #[test]
    fn tuple_get_rejects_large_indices_without_overflow() {
        let mut matrix = Matrix::<1, 1, i32>::zeros();
        assert!(matrix.get((usize::MAX, usize::MAX)).is_none());
        assert!(matrix.get_mut((usize::MAX, usize::MAX)).is_none());
    }
    #[test]
    fn swap() {
        let mut m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
        ];
        m.swap_rows(0, 2);
        let exp = matrix![
            7.0, 8.0, 9.0;
            4.0, 5.0, 6.0;
            1.0, 2.0, 3.0;
        ];
        assert_eq!(m, exp);
        m.swap_columns(0, 2);
        let exp = matrix![
            9.0, 8.0, 7.0;
            6.0, 5.0, 4.0;
            3.0, 2.0, 1.0;
        ];
        assert_eq!(m, exp);
    }
    #[test]
    fn transpose() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        let t = matrix![
            1.0, 4.0;
            2.0, 5.0;
            3.0, 6.0;
        ];
        assert_eq!(m.transpose(), t);
    }
    #[test]
    fn clone() {
        let a = matrix![
            1.0, 2.0, 3.0;
            5.0, 6.0, 4.0;
        ];
        assert_eq!(a.clone(), a);
    }
    #[test]
    fn norm() {
        let m = matrix![
            1.0,-2.0;
           -3.0, 6.0;
        ];
        assert_eq!(m.squared_norm(), 50.0);
        assert_relative_eq!(m.norm(), 7.0710678, max_relative = 1e-6);
    }

    #[test]
    fn norm_avoids_overflow_and_underflow() {
        let large = Matrix::<2, 1, f64>::from_rows([[1.0e308], [1.0e308]]);
        let small = Matrix::<2, 1, f64>::from_rows([[1.0e-300], [1.0e-300]]);
        assert_relative_eq!(large.norm(), 2.0_f64.sqrt() * 1.0e308, max_relative = 1e-14);
        assert_relative_eq!(
            small.norm(),
            2.0_f64.sqrt() * 1.0e-300,
            max_relative = 1e-14
        );
    }

    #[test]
    fn vector_dot_and_matvec() {
        let lhs = Matrix::<5, 1, f64>::from_rows([[1.0], [-2.0], [3.0], [4.0], [-5.0]]);
        let rhs = Matrix::<5, 1, f64>::from_rows([[2.0], [3.0], [-4.0], [5.0], [6.0]]);
        assert_eq!(lhs.dot(&rhs), -26.0);

        let matrix = Matrix::<4, 3, f64>::from_rows([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]);
        let vector = Matrix::<3, 1, f64>::from_rows([[2.0], [-1.0], [0.5]]);
        let expected = Matrix::<4, 1, f64>::from_rows([[1.5], [6.0], [10.5], [15.0]]);
        assert_eq!(matrix.matvec(&vector), expected);

        let mut output = Vector::<4, f64>::zeros();
        matrix.matvec_into(&vector, &mut output);
        assert_eq!(output, expected);
    }

    #[test]
    fn cross() {
        let a = vector![3.0;-3.0; 1.0];
        let b = vector![4.0; 9.0; 2.0];
        let exp = vector![-15.0; -2.0; 39.0];
        assert_relative_eq!(a.cross(&b), exp, max_relative = 1e-6);
    }

    #[test]
    fn trace() {
        let m = matrix![
            9.0, 8.0, 7.0;
            6.0, 5.0, 4.0;
            3.0, 2.0, 1.0;
        ];
        assert_eq!(m.trace(), 15.0);
    }
}
