use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub,
    SubAssign,
};

use crate::index::MatrixIndex;
use crate::kernels::{matmul, matvec, MatrixScalar, ReductionScalar};
use crate::num::Zero;
use crate::{Matrix, Vector};

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
    #[inline]
    pub fn mul_into<const P: usize>(&self, rhs: &Matrix<N, P, T>, output: &mut Matrix<M, P, T>) {
        matmul(self, rhs, output);
    }
}

impl<const M: usize, const N: usize, T> Matrix<M, N, T>
where
    T: ReductionScalar,
{
    /// Multiplies this matrix by a fixed-size vector without allocating
    /// intermediate storage.
    #[inline]
    pub fn matvec(&self, vector: &Vector<N, T>) -> Vector<M, T> {
        let mut output = Vector::<M, T>::zeros();
        self.matvec_into(vector, &mut output);
        output
    }

    /// Multiplies this matrix by a fixed-size vector and writes into `output`.
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
              0.0, 0.0, 0.0, 8.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0;
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
