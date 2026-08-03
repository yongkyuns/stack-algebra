use core::ops::{Add, Div, Mul, Sub};

use crate::num::Zero;
use crate::{Matrix, Vector};

mod portable;

#[cfg(target_arch = "x86_64")]
mod x86;

#[doc(hidden)]
pub trait MatmulBackend<T> {
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, T>,
        rhs: &Matrix<N, P, T>,
        output: &mut Matrix<M, P, T>,
    );

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
    type Matmul: MatmulBackend<Self>;
}

/// Associates a scalar type with its compile-time reduction kernels.
pub trait ReductionScalar: MatrixScalar {
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

#[cfg(not(target_arch = "x86_64"))]
impl_scalar_matrix_scalar!(f32, f64);

#[cfg(not(target_arch = "x86_64"))]
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
    use super::matmul_scalar;
    use crate::Matrix;

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
        assert_eq!(output, reference);

        for column in 0..P {
            for row in 0..M {
                let mut expected = 0.0;
                for shared in 0..N {
                    expected += lhs[(row, shared)] * rhs[(shared, column)];
                }
                let actual = reference[(row, column)];
                assert!((actual - expected).abs() <= 1e-14 + 1e-14 * expected.abs());
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
        assert_eq!(output, reference);

        for column in 0..P {
            for row in 0..M {
                let mut expected = 0.0_f64;
                for shared in 0..N {
                    expected += lhs[(row, shared)] as f64 * rhs[(shared, column)] as f64;
                }
                let actual = reference[(row, column)] as f64;
                assert!((actual - expected).abs() <= 1e-6 + 1e-6 * expected.abs());
            }
        }
    }

    #[test]
    fn scalar_f64_matmul_covers_tails_and_robotics_shapes() {
        check_f64::<1, 1, 1>();
        check_f64::<2, 3, 4>();
        check_f64::<3, 5, 2>();
        check_f64::<5, 7, 3>();
        check_f64::<7, 4, 9>();
        check_f64::<9, 6, 5>();
        check_f64::<15, 3, 6>();
        check_f64::<6, 15, 6>();
    }

    #[test]
    fn scalar_f32_matmul_covers_tails_and_robotics_shapes() {
        check_f32::<1, 1, 1>();
        check_f32::<2, 3, 4>();
        check_f32::<3, 5, 2>();
        check_f32::<5, 7, 3>();
        check_f32::<7, 4, 9>();
        check_f32::<9, 6, 5>();
        check_f32::<15, 3, 6>();
        check_f32::<6, 15, 6>();
    }
}
