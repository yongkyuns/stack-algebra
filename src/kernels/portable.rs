use core::ops::{Add, Mul};

use crate::num::Zero;
use crate::{Matrix, Vector};

use super::{MatmulBackend, ReductionBackend};

#[doc(hidden)]
/// Portable scalar reduction kernels used when no architecture-specific
/// implementation is selected.
pub struct ScalarReduction;

impl<T> ReductionBackend<T> for ScalarReduction
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    #[inline]
    fn dot<const M: usize>(lhs: &Vector<M, T>, rhs: &Vector<M, T>) -> T {
        let mut result = T::zero();
        for index in 0..M {
            result = result + lhs[index] * rhs[index];
        }
        result
    }

    #[inline]
    fn squared_norm<const M: usize, const N: usize>(matrix: &Matrix<M, N, T>) -> T {
        let mut result = T::zero();
        for &value in matrix.as_slice() {
            result = result + value * value;
        }
        result
    }

    #[inline]
    fn matvec<const M: usize, const N: usize>(
        matrix: &Matrix<M, N, T>,
        vector: &Vector<N, T>,
        output: &mut Vector<M, T>,
    ) {
        for row in 0..M {
            output[row] = T::zero();
        }
        for column in 0..N {
            let vector_value = vector[column];
            for row in 0..M {
                output[row] = output[row] + matrix[(row, column)] * vector_value;
            }
        }
    }
}

#[doc(hidden)]
/// Portable scalar matrix-multiplication kernels used as the fallback backend.
pub struct ScalarMatmul;

impl<T> MatmulBackend<T> for ScalarMatmul
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, T>,
        rhs: &Matrix<N, P, T>,
        output: &mut Matrix<M, P, T>,
    ) {
        matmul_scalar(lhs, rhs, output);
    }

    #[inline]
    fn rank_update_sub(target: &mut [T], source: &[T], scale: T)
    where
        T: Copy + Mul<Output = T> + core::ops::Sub<Output = T>,
    {
        for (target_value, source_value) in target.iter_mut().zip(source.iter()) {
            *target_value = *target_value - *source_value * scale;
        }
    }
}

#[inline]
pub(crate) fn matmul_scalar<const M: usize, const N: usize, const P: usize, T>(
    lhs: &Matrix<M, N, T>,
    rhs: &Matrix<N, P, T>,
    output: &mut Matrix<M, P, T>,
) where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    for column in 0..P {
        for row in 0..M {
            output[(row, column)] = T::zero();
        }

        for shared in 0..N {
            let rhs_value = rhs[(shared, column)];
            for row in 0..M {
                output[(row, column)] = output[(row, column)] + lhs[(row, shared)] * rhs_value;
            }
        }
    }
}
