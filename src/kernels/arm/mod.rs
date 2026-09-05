use super::{FactorizationScalar, MatmulBackend, MatrixScalar, ReductionBackend, ReductionScalar};
use crate::{Matrix, Vector};

mod neon;

use neon::{NeonMatmul, NeonReduction};

#[inline]
fn primitive_mul_add_f32(lhs: f32, rhs: f32, addend: f32) -> f32 {
    f32::mul_add(lhs, rhs, addend)
}

#[inline]
fn primitive_mul_add_f64(lhs: f64, rhs: f64, addend: f64) -> f64 {
    f64::mul_add(lhs, rhs, addend)
}

macro_rules! impl_neon_scalar {
    ($scalar:ty, $mul_add:ident) => {
        impl FactorizationScalar for $scalar {
            #[inline]
            fn symmetric_rank_k_update<const D: usize>(
                matrix: &mut Matrix<D, D, Self>,
                block_start: usize,
                block_end: usize,
            ) {
                <NeonMatmul as MatmulBackend<$scalar>>::symmetric_rank_k_update(
                    matrix,
                    block_start,
                    block_end,
                );
            }

            #[inline]
            fn rank_update_sub(target: &mut [Self], source: &[Self], scale: Self) {
                <NeonMatmul as MatmulBackend<$scalar>>::rank_update_sub(target, source, scale);
            }

            #[inline]
            fn rank_update_two_sub(
                target: &mut [Self],
                source_first: &[Self],
                scale_first: Self,
                source_second: &[Self],
                scale_second: Self,
            ) {
                <NeonMatmul as MatmulBackend<$scalar>>::rank_update_two_sub(
                    target,
                    source_first,
                    scale_first,
                    source_second,
                    scale_second,
                );
            }

            #[inline]
            fn scale_divide(target: &mut [Self], divisor: Self) {
                <NeonMatmul as MatmulBackend<$scalar>>::scale_divide(target, divisor);
            }

            #[inline]
            fn cholesky_update_column<const D: usize>(
                matrix: &mut Matrix<D, D, Self>,
                column: usize,
                diagonal: Self,
            ) {
                <NeonMatmul as MatmulBackend<$scalar>>::cholesky_update_column(
                    matrix, column, diagonal,
                );
            }
        }

        impl MatrixScalar for $scalar {
            #[inline]
            fn mul_add(lhs: Self, rhs: Self, addend: Self) -> Self {
                $mul_add(lhs, rhs, addend)
            }

            #[inline]
            fn matmul<const M: usize, const N: usize, const P: usize>(
                lhs: &Matrix<M, N, Self>,
                rhs: &Matrix<N, P, Self>,
                output: &mut Matrix<M, P, Self>,
            ) {
                <NeonMatmul as MatmulBackend<$scalar>>::run(lhs, rhs, output);
            }

            #[inline]
            fn dot_accumulate(lhs: &[Self], rhs: &[Self], initial: Self) -> Self {
                <NeonMatmul as MatmulBackend<$scalar>>::dot(lhs, rhs, initial)
            }
        }

        impl ReductionScalar for $scalar {
            #[inline]
            fn dot<const M: usize>(lhs: &Vector<M, Self>, rhs: &Vector<M, Self>) -> Self {
                <NeonReduction as ReductionBackend<$scalar>>::dot(lhs, rhs)
            }

            #[inline]
            fn squared_norm<const M: usize, const N: usize>(matrix: &Matrix<M, N, Self>) -> Self {
                <NeonReduction as ReductionBackend<$scalar>>::squared_norm(matrix)
            }

            #[inline]
            fn matvec<const M: usize, const N: usize>(
                matrix: &Matrix<M, N, Self>,
                vector: &Vector<N, Self>,
                output: &mut Vector<M, Self>,
            ) {
                <NeonReduction as ReductionBackend<$scalar>>::matvec(matrix, vector, output);
            }
        }
    };
}

impl_neon_scalar!(f32, primitive_mul_add_f32);
impl_neon_scalar!(f64, primitive_mul_add_f64);
