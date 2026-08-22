//! Arithmetic operations for fixed-size matrices and matrix-shaped views.
//!
//! Operators such as `+`, `-`, and `*` return a new fixed-size value. The
//! `*_into` methods and view functions write into caller-provided storage,
//! which is useful in control loops where avoiding an intermediate is
//! important. All dimensions are checked by the type system at compile time.
//!
//! # Example
//!
//! ```
//! use stack_algebra::{matrix, Matrix};
//!
//! let lhs = matrix![1.0_f32, 2.0; 3.0, 4.0];
//! let rhs = matrix![2.0_f32, 0.0; 1.0, 2.0];
//! let mut product = Matrix::<2, 2, f32>::zeros();
//! lhs.mul_into(&rhs, &mut product);
//! assert_eq!(product, matrix![4.0, 4.0; 10.0, 8.0]);
//! ```

use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub,
    SubAssign,
};

use crate::index::MatrixIndex;
use crate::kernels::{matmul, matvec, MatrixScalar, ReductionScalar};
use crate::num::Zero;
use crate::view::{Map, MatrixRead, StridedMap};
use crate::{Matrix, Vector};

#[inline]
fn column_major_matrix_ref<const M: usize, const N: usize, T>(
    data: &[T],
) -> Option<&Matrix<M, N, T>> {
    let required = M.checked_mul(N)?;
    let data = data.get(..required)?;
    if core::mem::size_of_val(data) != core::mem::size_of::<Matrix<M, N, T>>() {
        return None;
    }

    // SAFETY: `Matrix` is `repr(C)` over the nested array `[[T; M]; N]`.
    // Arrays are contiguous with the same element alignment as `T`, and the
    // slice above contains exactly `M * N` initialized `T` values in the
    // matrix's column-major order. The returned borrow cannot outlive `data`.
    Some(unsafe { &*data.as_ptr().cast::<Matrix<M, N, T>>() })
}

#[inline]
fn strided_column_major_matrix_ref<'a, const M: usize, const N: usize, T>(
    matrix: &'a StridedMap<'_, M, N, T>,
) -> Option<&'a Matrix<M, N, T>> {
    if M > 1 && matrix.inner_stride() != 1 {
        return None;
    }
    if N > 1 && matrix.outer_stride() != M {
        return None;
    }
    column_major_matrix_ref::<M, N, T>(matrix.as_slice())
}

#[inline]
fn strided_has_unit_inner<const M: usize, const N: usize, T>(
    matrix: &StridedMap<'_, M, N, T>,
) -> bool {
    M <= 1 || matrix.inner_stride() == 1
}

/// Multiplies column-major views that may have padding between columns.
///
/// This keeps caller-owned padded storage zero-copy while using a
/// column/shared/row traversal that streams contiguous columns. Exact
/// contiguous layouts still use the target-specific owned `matmul` backend.
#[inline]
fn matmul_leading_dimension_into<
    const M: usize,
    const N: usize,
    const P: usize,
    T: MatrixScalar,
>(
    lhs: &StridedMap<'_, M, N, T>,
    rhs: &StridedMap<'_, N, P, T>,
    output: &mut Matrix<M, P, T>,
) {
    let lhs_data = lhs.as_slice();
    let rhs_data = rhs.as_slice();
    let lhs_outer = lhs.outer_stride();
    let rhs_outer = rhs.outer_stride();
    let output_data = output.as_mut_slice();

    for value in output_data.iter_mut() {
        *value = T::zero();
    }

    for column in 0..P {
        let output_start = column * M;
        for shared in 0..N {
            let rhs_value = rhs_data[column * rhs_outer + shared];
            let lhs_start = shared * lhs_outer;
            for row in 0..M {
                let output_index = output_start + row;
                output_data[output_index] = T::mul_add(
                    lhs_data[lhs_start + row],
                    rhs_value,
                    output_data[output_index],
                );
            }
        }
    }
}

/// Computes `matrix * vector` directly from a fixed-size matrix view.
///
/// The view is read through [`MatrixRead`](crate::MatrixRead), so this works
/// for an owning matrix, block, map, or strided map. The returned vector is
/// fixed-size and stack allocated.
///
/// # Example
///
/// ```
/// use stack_algebra::{matrix, matvec_view, vector, Map};
///
/// let storage = [1_i32, 3, 2, 4];
/// let matrix = Map::<2, 2, _>::from_slice(&storage).unwrap();
/// let result = matvec_view(&matrix, &vector![5; 6]);
/// assert_eq!(result, vector![17; 39]);
/// ```
#[inline]
pub fn matvec_view<const M: usize, const N: usize, T, V>(
    matrix: &V,
    vector: &Vector<N, T>,
) -> Vector<M, T>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
    V: MatrixRead<M, N, T>,
{
    let mut output = Vector::<M, T>::zeros();
    matvec_view_into(matrix, vector, &mut output);
    output
}

/// Computes `matrix * vector` directly from a fixed-size matrix view.
///
/// This is the allocation-free counterpart to [`matvec_view`]. `output` may
/// be reused across iterations; it must have the matrix's compile-time row
/// count.
#[inline]
pub fn matvec_view_into<const M: usize, const N: usize, T, V>(
    matrix: &V,
    vector: &Vector<N, T>,
    output: &mut Vector<M, T>,
) where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
    V: MatrixRead<M, N, T>,
{
    for row in 0..M {
        let mut value = T::zero();
        for column in 0..N {
            value = value + *matrix.get_in_bounds(row, column) * vector[column];
        }
        output[row] = value;
    }
}

/// Computes `lhs * rhs` directly from fixed-size matrix views.
///
/// Both operands may be borrowed views with different backing layouts. The
/// destination is an owning fixed-size matrix supplied by the caller. This
/// generic path accepts arbitrary [`MatrixRead`] implementations; the direct
/// `Map` and `StridedMap` methods use the optimized owned-matrix kernels when
/// their storage is exactly column-major contiguous.
///
/// # Example
///
/// ```
/// use stack_algebra::{matmul_view_into, matrix, Map, Matrix};
///
/// let lhs_storage = [1_i32, 3, 2, 4];
/// let rhs_storage = [5_i32, 7, 6, 8];
/// let lhs = Map::<2, 2, _>::from_slice(&lhs_storage).unwrap();
/// let rhs = Map::<2, 2, _>::from_slice(&rhs_storage).unwrap();
/// let mut output = Matrix::<2, 2, i32>::zeros();
/// matmul_view_into(&lhs, &rhs, &mut output);
/// assert_eq!(output, matrix![19, 22; 43, 50]);
/// ```
#[inline]
pub fn matmul_view_into<const M: usize, const N: usize, const P: usize, T, Lhs, Rhs>(
    lhs: &Lhs,
    rhs: &Rhs,
    output: &mut Matrix<M, P, T>,
) where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
    Lhs: MatrixRead<M, N, T>,
    Rhs: MatrixRead<N, P, T>,
{
    for column in 0..P {
        for row in 0..M {
            let mut value = T::zero();
            for shared in 0..N {
                value =
                    value + *lhs.get_in_bounds(row, shared) * *rhs.get_in_bounds(shared, column);
            }
            output[(row, column)] = value;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Optimized mapped-view products
////////////////////////////////////////////////////////////////////////////////

impl<const M: usize, const N: usize, T> Map<'_, M, N, T>
where
    T: MatrixScalar,
{
    /// Multiplies two contiguous column-major maps using the same optimized
    /// kernel path as owned matrices.
    #[inline]
    pub fn mul_into<const P: usize>(&self, rhs: &Map<'_, N, P, T>, output: &mut Matrix<M, P, T>) {
        let lhs = column_major_matrix_ref::<M, N, T>(self.as_slice())
            .expect("Map storage always matches its compile-time matrix shape");
        let rhs = column_major_matrix_ref::<N, P, T>(rhs.as_slice())
            .expect("Map storage always matches its compile-time matrix shape");
        matmul(lhs, rhs, output);
    }

    /// Multiplies this contiguous map by an owned matrix using the optimized
    /// owned-matrix kernel path.
    #[inline]
    pub fn mul_matrix_into<const P: usize>(
        &self,
        rhs: &Matrix<N, P, T>,
        output: &mut Matrix<M, P, T>,
    ) {
        let lhs = column_major_matrix_ref::<M, N, T>(self.as_slice())
            .expect("Map storage always matches its compile-time matrix shape");
        matmul(lhs, rhs, output);
    }

    /// Multiplies this contiguous map by a strided view. Exact column-major
    /// storage uses the optimized kernel; other layouts use the generic view
    /// path without repacking.
    #[inline]
    pub fn mul_strided_into<const P: usize>(
        &self,
        rhs: &StridedMap<'_, N, P, T>,
        output: &mut Matrix<M, P, T>,
    ) {
        if let Some(rhs) = strided_column_major_matrix_ref(rhs) {
            let lhs = column_major_matrix_ref::<M, N, T>(self.as_slice())
                .expect("Map storage always matches its compile-time matrix shape");
            matmul(lhs, rhs, output);
        } else {
            matmul_view_into(self, rhs, output);
        }
    }
}

impl<const M: usize, const N: usize, T> Map<'_, M, N, T>
where
    T: ReductionScalar,
{
    /// Multiplies this contiguous map by a vector using the same optimized
    /// reduction kernel as an owned matrix.
    #[inline]
    pub fn matvec(&self, vector: &Vector<N, T>) -> Vector<M, T> {
        let mut output = Vector::<M, T>::zeros();
        self.matvec_into(vector, &mut output);
        output
    }

    /// Multiplies this contiguous map by a vector into caller-owned output.
    #[inline]
    pub fn matvec_into(&self, vector: &Vector<N, T>, output: &mut Vector<M, T>) {
        let matrix = column_major_matrix_ref::<M, N, T>(self.as_slice())
            .expect("Map storage always matches its compile-time matrix shape");
        matvec(matrix, vector, output);
    }
}

impl<const M: usize, const N: usize, T> StridedMap<'_, M, N, T>
where
    T: MatrixScalar,
{
    /// Multiplies two strided views. Exact contiguous column-major layouts use
    /// the target-specific owned kernel. Unit-inner-stride column-major views
    /// with padded leading dimensions use a direct zero-copy streaming path;
    /// arbitrary inner strides use the generic view path.
    #[inline]
    pub fn mul_into<const P: usize>(
        &self,
        rhs: &StridedMap<'_, N, P, T>,
        output: &mut Matrix<M, P, T>,
    ) {
        if let (Some(lhs), Some(rhs)) = (
            strided_column_major_matrix_ref(self),
            strided_column_major_matrix_ref(rhs),
        ) {
            matmul(lhs, rhs, output);
        } else if strided_has_unit_inner(self) && strided_has_unit_inner(rhs) {
            matmul_leading_dimension_into(self, rhs, output);
        } else {
            matmul_view_into(self, rhs, output);
        }
    }

    /// Multiplies this strided view by an owned matrix. Exact column-major
    /// storage uses the optimized kernel; other layouts use direct strided
    /// reads without materializing an intermediate matrix.
    #[inline]
    pub fn mul_matrix_into<const P: usize>(
        &self,
        rhs: &Matrix<N, P, T>,
        output: &mut Matrix<M, P, T>,
    ) {
        if let Some(lhs) = strided_column_major_matrix_ref(self) {
            matmul(lhs, rhs, output);
        } else {
            matmul_view_into(self, rhs, output);
        }
    }

    /// Multiplies this strided view by a contiguous map, reusing the optimized
    /// kernel when this view is also exact column-major contiguous.
    #[inline]
    pub fn mul_map_into<const P: usize>(
        &self,
        rhs: &Map<'_, N, P, T>,
        output: &mut Matrix<M, P, T>,
    ) {
        if let Some(lhs) = strided_column_major_matrix_ref(self) {
            let rhs = column_major_matrix_ref::<N, P, T>(rhs.as_slice())
                .expect("Map storage always matches its compile-time matrix shape");
            matmul(lhs, rhs, output);
        } else {
            matmul_view_into(self, rhs, output);
        }
    }
}

impl<const M: usize, const N: usize, T> StridedMap<'_, M, N, T>
where
    T: ReductionScalar,
{
    /// Multiplies this strided view by a vector. Exact column-major contiguous
    /// storage uses the optimized reduction kernel; arbitrary strides use the
    /// generic zero-copy view path.
    #[inline]
    pub fn matvec(&self, vector: &Vector<N, T>) -> Vector<M, T> {
        let mut output = Vector::<M, T>::zeros();
        self.matvec_into(vector, &mut output);
        output
    }

    /// Multiplies this strided view by a vector into caller-owned output.
    #[inline]
    pub fn matvec_into(&self, vector: &Vector<N, T>, output: &mut Vector<M, T>) {
        if let Some(matrix) = strided_column_major_matrix_ref(self) {
            matvec(matrix, vector, output);
        } else {
            matvec_view_into(self, vector, output);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Indexing
////////////////////////////////////////////////////////////////////////////////

impl<T, I, const M: usize, const N: usize> Index<I> for Matrix<M, N, T>
where
    I: MatrixIndex<Self>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, I, const M: usize, const N: usize> IndexMut<I> for Matrix<M, N, T>
where
    I: MatrixIndex<Self>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Matrix + T
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_op_scalar {
    ($trt:ident, $meth:ident) => {
        // Matrix + T
        impl<T, const M: usize, const N: usize> $trt<T> for Matrix<M, N, T>
        where
            T: Copy + $trt<Output = T>,
        {
            type Output = Matrix<M, N, T>;

            fn $meth(mut self, other: T) -> Self::Output {
                #[allow(clippy::suspicious_arithmetic_impl)]
                for i in 0..(M * N) {
                    self[i] = self[i].$meth(other);
                }
                self
            }
        }

        // Matrix + &T
        impl<T, const M: usize, const N: usize> $trt<&T> for Matrix<M, N, T>
        where
            T: Copy + $trt<Output = T>,
        {
            type Output = Matrix<M, N, T>;

            fn $meth(mut self, other: &T) -> Self::Output {
                #[allow(clippy::suspicious_arithmetic_impl)]
                for i in 0..(M * N) {
                    self[i] = self[i].$meth(*other);
                }
                self
            }
        }

        // &Matrix + T
        impl<T, const M: usize, const N: usize> $trt<T> for &Matrix<M, N, T>
        where
            T: Copy + Zero + $trt<Output = T>,
        {
            type Output = Matrix<M, N, T>;

            fn $meth(self, other: T) -> Self::Output {
                let mut matrix = Self::Output::zeros();
                #[allow(clippy::suspicious_arithmetic_impl)]
                for i in 0..(M * N) {
                    matrix[i] = self[i].$meth(other);
                }
                matrix
            }
        }

        // &Matrix + &T
        impl<T, const M: usize, const N: usize> $trt<&T> for &Matrix<M, N, T>
        where
            T: Copy + Zero + $trt<Output = T>,
        {
            type Output = Matrix<M, N, T>;

            fn $meth(self, other: &T) -> Self::Output {
                let mut matrix = Self::Output::zeros();
                #[allow(clippy::suspicious_arithmetic_impl)]
                for i in 0..(M * N) {
                    matrix[i] = self[i].$meth(*other);
                }
                matrix
            }
        }
    };
}

impl_op_scalar! { Add, add }
impl_op_scalar! { Sub, sub }
impl_op_scalar! { Mul, mul }
impl_op_scalar! { Div, div }
impl_op_scalar! { Rem, rem }

////////////////////////////////////////////////////////////////////////////////
// Matrix += T
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_op_assign_scalar {
    ($trt:ident, $meth:ident) => {
        // Matrix += T
        impl<'a, T, const M: usize, const N: usize> $trt<T> for Matrix<M, N, T>
        where
            T: Copy + $trt<T>,
        {
            fn $meth(&mut self, other: T) {
                #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
                for i in 0..(M * N) {
                    self[i].$meth(other);
                }
            }
        }

        // Matrix += &T
        impl<T, const M: usize, const N: usize> $trt<&T> for Matrix<M, N, T>
        where
            T: Copy + $trt<T>,
        {
            fn $meth(&mut self, other: &T) {
                #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
                for i in 0..(M * N) {
                    self[i].$meth(*other);
                }
            }
        }
    };
}

impl_op_assign_scalar! { AddAssign, add_assign }
impl_op_assign_scalar! { SubAssign, sub_assign }
impl_op_assign_scalar! { MulAssign, mul_assign }
impl_op_assign_scalar! { DivAssign, div_assign }
impl_op_assign_scalar! { RemAssign, rem_assign }

////////////////////////////////////////////////////////////////////////////////
// Matrix + Matrix
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_op {
    ($trt:ident, $meth:ident) => {
        // Matrix + Matrix
        impl<T, const M: usize, const N: usize> $trt<Matrix<M, N, T>> for Matrix<M, N, T>
        where
            T: Copy + $trt<Output = T>,
        {
            type Output = Matrix<M, N, T>;

            fn $meth(mut self, other: Matrix<M, N, T>) -> Self::Output {
                #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
                for i in 0..(M * N) {
                    self[i] = self[i].$meth(other[i]);
                }
                self
            }
        }

        // Matrix + &Matrix
        impl<T, const M: usize, const N: usize> $trt<&Matrix<M, N, T>> for Matrix<M, N, T>
        where
            T: Copy + $trt<Output = T>,
        {
            type Output = Matrix<M, N, T>;

            fn $meth(mut self, other: &Matrix<M, N, T>) -> Self::Output {
                #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
                for i in 0..(M * N) {
                    self[i] = self[i].$meth(other[i]);
                }
                self
            }
        }

        // &Matrix + Matrix
        impl<T, const M: usize, const N: usize> $trt<Matrix<M, N, T>> for &Matrix<M, N, T>
        where
            T: Copy + Zero + $trt<Output = T>,
        {
            type Output = Matrix<M, N, T>;

            fn $meth(self, other: Matrix<M, N, T>) -> Self::Output {
                let mut matrix = *self;
                #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
                for i in 0..(M * N) {
                    matrix[i] = self[i].$meth(other[i]);
                }
                matrix
            }
        }

        // &Matrix + &Matrix
        impl<T, const M: usize, const N: usize> $trt<&Matrix<M, N, T>> for &Matrix<M, N, T>
        where
            T: Copy + Zero + $trt<Output = T>,
        {
            type Output = Matrix<M, N, T>;

            fn $meth(self, other: &Matrix<M, N, T>) -> Self::Output {
                let mut matrix = *self;
                #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
                for i in 0..(M * N) {
                    matrix[i] = self[i].$meth(other[i]);
                }
                matrix
            }
        }
    };
}

impl_op! { Add, add }
impl_op! { Sub, sub }

////////////////////////////////////////////////////////////////////////////////
// Matrix * Matrix
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_op_mul {
    ($lhs:ty, $rhs:ty) => {
        impl<T, const N: usize, const M: usize, const P: usize> Mul<$rhs> for $lhs
        where
            T: MatrixScalar,
        {
            type Output = Matrix<M, P, T>;

            fn mul(self, rhs: $rhs) -> Self::Output {
                let mut matrix = Self::Output::zeros();
                let lhs: &Matrix<M, N, T> = &self;
                let rhs: &Matrix<N, P, T> = &rhs;
                lhs.mul_into(rhs, &mut matrix);
                matrix
            }
        }
    };
}

impl_op_mul! {  Matrix<M,N,T>,  Matrix<N,P,T> }
impl_op_mul! {  Matrix<M,N,T>, &Matrix<N,P,T> }
impl_op_mul! { &Matrix<M,N,T>,  Matrix<N,P,T> }
impl_op_mul! { &Matrix<M,N,T>, &Matrix<N,P,T> }

impl<const M: usize, const N: usize, T> Matrix<M, N, T>
where
    T: MatrixScalar,
{
    /// Multiplies this matrix by `rhs` and writes the result into `output`.
    ///
    /// The inputs and output use column-major traversal and do not allocate.
    /// `output` may alias neither input; pass a separate matrix when updating
    /// in place is required.
    #[inline]
    pub fn mul_into<const P: usize>(&self, rhs: &Matrix<N, P, T>, output: &mut Matrix<M, P, T>) {
        matmul(self, rhs, output);
    }

    /// Multiplies this matrix by a contiguous column-major map using the same
    /// optimized kernel as an owned right-hand side.
    #[inline]
    pub fn mul_map_into<const P: usize>(
        &self,
        rhs: &Map<'_, N, P, T>,
        output: &mut Matrix<M, P, T>,
    ) {
        let rhs = column_major_matrix_ref::<N, P, T>(rhs.as_slice())
            .expect("Map storage always matches its compile-time matrix shape");
        matmul(self, rhs, output);
    }

    /// Multiplies this matrix by a strided view. Exact column-major contiguous
    /// storage uses the optimized kernel; arbitrary strides use the generic
    /// zero-copy view path.
    #[inline]
    pub fn mul_strided_into<const P: usize>(
        &self,
        rhs: &StridedMap<'_, N, P, T>,
        output: &mut Matrix<M, P, T>,
    ) {
        if let Some(rhs) = strided_column_major_matrix_ref(rhs) {
            matmul(self, rhs, output);
        } else {
            matmul_view_into(self, rhs, output);
        }
    }

    /// Updates this matrix in place with `self += alpha * x`.
    ///
    /// For built-in floating-point scalars, the per-element multiply-add uses
    /// the scalar backend's fused operation where the target supports it.
    #[inline]
    pub fn axpy_in_place(&mut self, alpha: T, x: &Self) {
        for (output, &input) in self.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *output = T::mul_add(alpha, input, *output);
        }
    }

    /// Writes `alpha * self + y` into caller-provided `output`.
    ///
    /// This is the non-aliasing output form of an AXPY-style update and avoids
    /// the temporary matrix created by `self * alpha + y`.
    #[inline]
    pub fn axpy_into(&self, alpha: T, y: &Self, output: &mut Self) {
        for ((output, &x), &y) in output
            .as_mut_slice()
            .iter_mut()
            .zip(self.as_slice())
            .zip(y.as_slice())
        {
            *output = T::mul_add(alpha, x, y);
        }
    }

    /// Writes `alpha * self + beta * rhs` into caller-provided `output`.
    ///
    /// The operation is explicit rather than expression-template based, so the
    /// destination storage remains obvious while one intermediate matrix is
    /// eliminated from common estimation/control update patterns.
    #[inline]
    pub fn linear_combination_into(&self, alpha: T, rhs: &Self, beta: T, output: &mut Self) {
        for ((output, &lhs), &rhs) in output
            .as_mut_slice()
            .iter_mut()
            .zip(self.as_slice())
            .zip(rhs.as_slice())
        {
            *output = T::mul_add(alpha, lhs, beta * rhs);
        }
    }
}

impl<const M: usize, const N: usize, T> Matrix<M, N, T>
where
    T: ReductionScalar,
{
    /// Multiplies this matrix by a fixed-size vector without allocating
    /// intermediate storage.
    ///
    /// The result has one entry per matrix row. For repeated operations, use
    /// [`Matrix::matvec_into`](Self::matvec_into) to reuse an output buffer.
    #[inline]
    pub fn matvec(&self, vector: &Vector<N, T>) -> Vector<M, T> {
        let mut output = Vector::<M, T>::zeros();
        self.matvec_into(vector, &mut output);
        output
    }

    /// Multiplies this matrix by a fixed-size vector and writes into `output`.
    ///
    /// This method is suitable for allocation-free control-loop code. The
    /// matrix and output dimensions are encoded in their types.
    #[inline]
    pub fn matvec_into(&self, vector: &Vector<N, T>, output: &mut Vector<M, T>) {
        matvec(self, vector, output);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Matrix += Matrix
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_op_assign {
    (impl $trt:ident<$rhs:ty>, $meth:ident) => {
        impl<T, const M: usize, const N: usize> $trt<$rhs> for Matrix<M, N, T>
        where
            T: Copy + $trt,
        {
            fn $meth(&mut self, other: $rhs) {
                #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
                for i in 0..(M * N) {
                    self[i].$meth(other[i]);
                }
            }
        }
    };
}

impl_op_assign! { impl AddAssign< Matrix<M,N,T>>, add_assign }
impl_op_assign! { impl AddAssign<&Matrix<M,N,T>>, add_assign }
impl_op_assign! { impl SubAssign< Matrix<M,N,T>>, sub_assign }
impl_op_assign! { impl SubAssign<&Matrix<M,N,T>>, sub_assign }

////////////////////////////////////////////////////////////////////////////////
// -Matrix
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_op_unary {
    ($trt:ident, $meth:ident) => {
        impl<T, const M: usize, const N: usize> $trt for Matrix<M, N, T>
        where
            T: Copy + Zero + $trt<Output = T>,
        {
            type Output = Matrix<M, N, T>;

            fn $meth(mut self) -> Self::Output {
                #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
                for i in 0..(M * N) {
                    self[i] = self[i].$meth();
                }
                self
            }
        }

        impl<T, const M: usize, const N: usize> $trt for &Matrix<M, N, T>
        where
            T: Copy + Zero + $trt<Output = T>,
        {
            type Output = Matrix<M, N, T>;

            fn $meth(self) -> Self::Output {
                let mut matrix = Self::Output::zeros();
                #[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
                for i in 0..(M * N) {
                    matrix[i] = self[i].$meth();
                }
                matrix
            }
        }
    };
}

impl_op_unary! { Neg, neg }
impl_op_unary! { Not, not }

#[cfg(test)]
mod tests {
    use crate::*;
    extern crate std;

    #[ignore]
    #[test]
    fn time() {
        use std::println;
        use std::time::Instant;

        let m = matrix![
              2.0_f32, 3.0, 0.0, 9.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0;
              1.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 9.0, 2.0, 3.0, 1.0;
              1.0, 4.0, 0.0, 2.0, 8.0, 5.0, 0.0, 3.0, 6.0, 1.0, 9.0;
              0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0;
              2.0, 2.0, 4.0, 1.0, 1.0, 2.0, 1.0, 6.0, 9.0, 0.0, 7.0;
              0.0, 0.0, 0.0, 6.0, 0.0, 7.0, 0.0, 1.0, 0.0, 0.0, 0.0;
              2.0, 5.0, 0.0, 7.0, 0.0, 4.0, 6.0, 8.0, 5.0, 1.0, 3.0;
              0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0;
              0.0, 0.0, 0.0, 8.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0;
              2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0;
              2.0, 6.0, 0.0, 1.0, 0.0,30.0, 0.0, 2.0, 3.0, 2.0, 1.0;
        ];

        let begin = Instant::now();
        const N: usize = 1000000;
        for _ in 0..N {
            let _ = m * m;
        }
        let elapsed = (Instant::now() - begin).as_nanos();
        println!(
            "11x11 Matrix Multiplication: {} ns/call",
            elapsed as f32 / N as f32
        );

        let begin = Instant::now();
        for _ in 0..N {
            let _ = m.inverse();
        }
        let elapsed = (Instant::now() - begin).as_nanos();
        println!(
            "11x11 Matrix Inverse: {} ns/call",
            elapsed as f32 / N as f32
        );
    }
    #[test]
    fn scalar() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        let res = m + 3.0;
        let exp = matrix![
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
        ];
        assert_eq!(res, exp);
        let res = res - 3.0;
        assert_eq!(res, m);
    }
    #[test]
    fn mat_add() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        let m2 = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        let exp = matrix![
            2.0, 4.0, 6.0;
            8.0, 10.0, 12.0;
        ];
        assert_eq!(m + m2, exp);
    }
    #[test]
    fn mat_mul() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        let m2 = matrix![
            1.0, 2.0;
            3.0, 4.0;
            5.0, 6.0;
        ];
        let exp = matrix![
            22.0, 28.0;
            49.0, 64.0;
        ];
        assert_eq!(m * m2, exp);
    }
}
