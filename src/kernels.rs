use core::ops::{Add, Div, Mul, Sub};

use crate::num::Zero;
use crate::{Matrix, Vector};

mod portable;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod arm;

#[doc(hidden)]
pub trait MatmulBackend<T> {
    /// Multiplies fixed-size matrices into caller-provided output storage.
    ///
    /// Implementations are selected through [`MatrixScalar::Matmul`], so the
    /// dispatch is resolved at compile time and does not add a runtime branch.
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, T>,
        rhs: &Matrix<N, P, T>,
        output: &mut Matrix<M, P, T>,
    );

    fn dot(lhs: &[T], rhs: &[T], initial: T) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        let mut result = initial;
        for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
            result = result + *lhs_value * *rhs_value;
        }
        result
    }

    fn symmetric_rank_k_update<const D: usize>(
        matrix: &mut Matrix<D, D, T>,
        block_start: usize,
        block_end: usize,
    ) where
        T: Copy + Mul<Output = T> + Sub<Output = T>,
    {
        for row in block_end..D {
            for column in block_end..=row {
                let mut value = matrix[(row, column)];
                for index in block_start..block_end {
                    value = value
                        - matrix[(row, index)] * matrix[(index, index)] * matrix[(column, index)];
                }
                matrix[(row, column)] = value;
            }
        }
    }

    fn rank_update_sub(target: &mut [T], source: &[T], scale: T)
    where
        T: Copy + Mul<Output = T> + Sub<Output = T>,
    {
        for (target_value, source_value) in target.iter_mut().zip(source.iter()) {
            *target_value = *target_value - *source_value * scale;
        }
    }

    fn rank_update_two_sub(
        target: &mut [T],
        source_first: &[T],
        scale_first: T,
        source_second: &[T],
        scale_second: T,
    ) where
        T: Copy + Mul<Output = T> + Sub<Output = T>,
    {
        for ((target_value, first_value), second_value) in target
            .iter_mut()
            .zip(source_first.iter())
            .zip(source_second.iter())
        {
            *target_value =
                *target_value - *first_value * scale_first - *second_value * scale_second;
        }
    }

    fn scale_divide(target: &mut [T], divisor: T)
    where
        T: Copy + Div<Output = T>,
    {
        for value in target {
            *value = *value / divisor;
        }
    }
}

#[doc(hidden)]
pub trait ReductionBackend<T> {
    /// Computes a fixed-size vector dot product.
    fn dot<const M: usize>(lhs: &Vector<M, T>, rhs: &Vector<M, T>) -> T;

    fn squared_norm<const M: usize, const N: usize>(matrix: &Matrix<M, N, T>) -> T;

    fn matvec<const M: usize, const N: usize>(
        matrix: &Matrix<M, N, T>,
        vector: &Vector<N, T>,
        output: &mut Vector<M, T>,
    );
}

#[doc(hidden)]
pub use portable::{ScalarMatmul, ScalarReduction};

/// Associates a scalar type with its compile-time matrix multiplication kernel.
///
/// Implement this trait for custom scalar types to enable matrix products. The
/// `ScalarMatmul` kernel provides the portable fallback; specialized kernels
/// can be associated when the scalar type has a matching implementation.
pub trait MatrixScalar: Copy + Zero + Add<Output = Self> + Mul<Output = Self> {
    /// Backend used by [`crate::Matrix::mul_into`] and matrix multiplication.
    ///
    /// The default scalar implementations use [`ScalarMatmul`]. Architecture
    /// modules can provide a packet backend for a scalar type when the target
    /// exposes a suitable SIMD instruction set.
    type Matmul: MatmulBackend<Self>;
}

/// Associates a scalar type with its compile-time reduction kernels.
pub trait ReductionScalar: MatrixScalar {
    /// Backend used by fixed-size dot, norm, and matrix-vector reductions.
    ///
    /// The default implementation is [`ScalarReduction`]. As with matrix
    /// multiplication, choosing a backend is a type-level operation with no
    /// runtime backend lookup.
    type Reduction: ReductionBackend<Self>;
}

macro_rules! impl_scalar_matrix_scalar {
    ($($scalar:ty),+ $(,)?) => {
        $(impl MatrixScalar for $scalar {
            type Matmul = ScalarMatmul;
        })+
    };
}

macro_rules! impl_scalar_reduction_scalar {
    ($($scalar:ty),+ $(,)?) => {
        $(impl ReductionScalar for $scalar {
            type Reduction = ScalarReduction;
        })+
    };
}

impl_scalar_matrix_scalar!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
impl_scalar_reduction_scalar!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
impl_scalar_matrix_scalar!(f32, f64);

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
impl_scalar_reduction_scalar!(f32, f64);

#[inline]
pub(crate) fn matmul<const M: usize, const N: usize, const P: usize, T>(
    lhs: &Matrix<M, N, T>,
    rhs: &Matrix<N, P, T>,
    output: &mut Matrix<M, P, T>,
) where
    T: MatrixScalar,
{
    T::Matmul::run(lhs, rhs, output);
}

#[inline]
pub(crate) fn matvec<const M: usize, const N: usize, T>(
    matrix: &Matrix<M, N, T>,
    vector: &Vector<N, T>,
    output: &mut Vector<M, T>,
) where
    T: ReductionScalar,
{
    T::Reduction::matvec(matrix, vector, output);
}

#[cfg(test)]
pub(crate) use portable::matmul_scalar;

#[cfg(test)]
mod tests {
    use super::{matmul_scalar, MatmulBackend};
    use crate::{Matrix, MatrixScalar};

    fn next_value(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mantissa = *state >> 11;
        (mantissa as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
    }

    fn generated_f64<const R: usize, const C: usize>(seed: u64) -> Matrix<R, C, f64> {
        let mut state = seed;
        Matrix::from_fn(|_, _| next_value(&mut state))
    }

    fn generated_f32<const R: usize, const C: usize>(seed: u64) -> Matrix<R, C, f32> {
        generated_f64::<R, C>(seed).cast()
    }

    fn check_f64<const M: usize, const N: usize, const P: usize>() {
        let lhs = generated_f64::<M, N>(11);
        let rhs = generated_f64::<N, P>(29);
        let mut reference = Matrix::<M, P, f64>::zeros();
        let mut output = Matrix::<M, P, f64>::zeros();
        matmul_scalar(&lhs, &rhs, &mut reference);
        lhs.mul_into(&rhs, &mut output);

        for column in 0..P {
            for row in 0..M {
                let mut expected = 0.0;
                for shared in 0..N {
                    expected += lhs[(row, shared)] * rhs[(shared, column)];
                }
                let actual = reference[(row, column)];
                assert!((actual - expected).abs() <= 1e-14 + 1e-14 * expected.abs());
                let actual = output[(row, column)];
                assert!((actual - expected).abs() <= 1e-13 + 1e-13 * expected.abs());
            }
        }
    }

    fn check_f32<const M: usize, const N: usize, const P: usize>() {
        let lhs = generated_f32::<M, N>(11);
        let rhs = generated_f32::<N, P>(29);
        let mut reference = Matrix::<M, P, f32>::zeros();
        let mut output = Matrix::<M, P, f32>::zeros();
        matmul_scalar(&lhs, &rhs, &mut reference);
        lhs.mul_into(&rhs, &mut output);

        for column in 0..P {
            for row in 0..M {
                let mut expected = 0.0_f64;
                for shared in 0..N {
                    expected += lhs[(row, shared)] as f64 * rhs[(shared, column)] as f64;
                }
                let actual = reference[(row, column)] as f64;
                assert!((actual - expected).abs() <= 1e-6 + 1e-6 * expected.abs());
                let actual = output[(row, column)] as f64;
                assert!((actual - expected).abs() <= 2e-6 + 2e-6 * expected.abs());
            }
        }
    }

    #[test]
    fn scalar_f64_matmul_covers_tails_and_rectangular_shapes() {
        check_f64::<1, 1, 1>();
        check_f64::<2, 3, 4>();
        check_f64::<3, 5, 2>();
        check_f64::<5, 7, 3>();
        check_f64::<7, 4, 9>();
        check_f64::<9, 6, 5>();
        check_f64::<15, 3, 6>();
        check_f64::<6, 15, 6>();
        check_f64::<15, 15, 15>();
        check_f64::<16, 16, 16>();
    }

    #[test]
    fn scalar_f32_matmul_covers_tails_and_rectangular_shapes() {
        check_f32::<1, 1, 1>();
        check_f32::<2, 3, 4>();
        check_f32::<3, 5, 2>();
        check_f32::<5, 7, 3>();
        check_f32::<7, 4, 9>();
        check_f32::<9, 6, 5>();
        check_f32::<15, 3, 6>();
        check_f32::<6, 15, 6>();
        check_f32::<15, 15, 15>();
        check_f32::<16, 16, 16>();
    }

    #[test]
    fn rank_update_two_sub_covers_simd_tails() {
        let mut output = [1.0_f64, -2.0, 3.5, 4.0, -5.0, 6.25, 7.0];
        let first = [0.5_f64, 1.0, -2.0, 3.0, 4.0, -1.5, 2.5];
        let second = [-1.0_f64, 2.0, 0.5, -2.5, 1.5, 3.0, -4.0];
        <f64 as MatrixScalar>::Matmul::rank_update_two_sub(
            &mut output,
            &first,
            1.25,
            &second,
            -0.75,
        );

        let expected = [-0.375, -1.75, 6.375, -1.625, -8.875, 10.375, 0.875];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-14);
        }
    }

    #[test]
    fn rank_update_two_sub_f32_covers_simd_tails() {
        let mut output = [1.0_f32, -2.0, 3.5, 4.0, -5.0, 6.25, 7.0];
        let first = [0.5_f32, 1.0, -2.0, 3.0, 4.0, -1.5, 2.5];
        let second = [-1.0_f32, 2.0, 0.5, -2.5, 1.5, 3.0, -4.0];
        <f32 as MatrixScalar>::Matmul::rank_update_two_sub(
            &mut output,
            &first,
            1.25,
            &second,
            -0.75,
        );

        let expected = [-0.375, -1.75, 6.375, -1.625, -8.875, 10.375, 0.875];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }
}
