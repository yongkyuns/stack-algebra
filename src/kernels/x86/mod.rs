use super::{FactorizationScalar, MatmulBackend, MatrixScalar, ReductionBackend, ReductionScalar};
use crate::{Matrix, Vector};

#[cfg(target_feature = "avx2")]
mod avx2;
#[cfg(all(target_feature = "avx2", target_feature = "fma"))]
mod avx2_fma;
#[cfg(all(target_feature = "avx2", not(target_feature = "fma")))]
use avx2::{X86Avx2Matmul, X86Avx2Reduction};
#[cfg(all(target_feature = "avx2", target_feature = "fma"))]
use avx2_fma::{X86Avx2FmaMatmul as X86Avx2Matmul, X86Avx2FmaReduction as X86Avx2Reduction};

#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
mod sse2;
#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
use sse2::{X86Sse2Matmul, X86Sse2Reduction};

#[inline]
fn primitive_mul_add_f32(lhs: f32, rhs: f32, addend: f32) -> f32 {
    f32::mul_add(lhs, rhs, addend)
}

#[inline]
fn primitive_mul_add_f64(lhs: f64, rhs: f64, addend: f64) -> f64 {
    f64::mul_add(lhs, rhs, addend)
}

macro_rules! impl_kernel_scalar {
    ($scalar:ty, $mul_add:ident, $matmul:ty, $reduction:ty) => {
        impl FactorizationScalar for $scalar {
            #[inline]
            fn symmetric_rank_k_update<const D: usize>(
                matrix: &mut Matrix<D, D, Self>,
                block_start: usize,
                block_end: usize,
            ) {
                <$matmul as MatmulBackend<$scalar>>::symmetric_rank_k_update(
                    matrix,
                    block_start,
                    block_end,
                );
            }

            #[inline]
            fn rank_update_sub(target: &mut [Self], source: &[Self], scale: Self) {
                <$matmul as MatmulBackend<$scalar>>::rank_update_sub(target, source, scale);
            }

            #[inline]
            fn rank_update_two_sub(
                target: &mut [Self],
                source_first: &[Self],
                scale_first: Self,
                source_second: &[Self],
                scale_second: Self,
            ) {
                <$matmul as MatmulBackend<$scalar>>::rank_update_two_sub(
                    target,
                    source_first,
                    scale_first,
                    source_second,
                    scale_second,
                );
            }

            #[inline]
            fn rotate_columns(first: &mut [Self], second: &mut [Self], cosine: Self, sine: Self) {
                <$matmul as MatmulBackend<$scalar>>::rotate_columns(first, second, cosine, sine);
            }

            #[inline]
            fn scale_divide(target: &mut [Self], divisor: Self) {
                <$matmul as MatmulBackend<$scalar>>::scale_divide(target, divisor);
            }

            #[inline]
            fn cholesky_update_column<const D: usize>(
                matrix: &mut Matrix<D, D, Self>,
                column: usize,
                diagonal: Self,
            ) {
                <$matmul as MatmulBackend<$scalar>>::cholesky_update_column(
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
                <$matmul as MatmulBackend<$scalar>>::run(lhs, rhs, output);
            }

            #[inline]
            fn dot_accumulate(lhs: &[Self], rhs: &[Self], initial: Self) -> Self {
                <$matmul as MatmulBackend<$scalar>>::dot(lhs, rhs, initial)
            }

            #[inline]
            fn symmetric_dot(lhs: &[Self], rhs: &[Self]) -> (Self, Self, Self) {
                <$matmul as MatmulBackend<$scalar>>::symmetric_dot(lhs, rhs)
            }
        }

        impl ReductionScalar for $scalar {
            #[inline]
            fn dot<const M: usize>(lhs: &Vector<M, Self>, rhs: &Vector<M, Self>) -> Self {
                <$reduction as ReductionBackend<$scalar>>::dot(lhs, rhs)
            }

            #[inline]
            fn squared_norm<const M: usize, const N: usize>(matrix: &Matrix<M, N, Self>) -> Self {
                <$reduction as ReductionBackend<$scalar>>::squared_norm(matrix)
            }

            #[inline]
            fn matvec<const M: usize, const N: usize>(
                matrix: &Matrix<M, N, Self>,
                vector: &Vector<N, Self>,
                output: &mut Vector<M, Self>,
            ) {
                <$reduction as ReductionBackend<$scalar>>::matvec(matrix, vector, output);
            }
        }
    };
}

#[cfg(target_feature = "avx2")]
impl_kernel_scalar!(f32, primitive_mul_add_f32, X86Avx2Matmul, X86Avx2Reduction);
#[cfg(target_feature = "avx2")]
impl_kernel_scalar!(f64, primitive_mul_add_f64, X86Avx2Matmul, X86Avx2Reduction);

#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
impl_kernel_scalar!(f32, primitive_mul_add_f32, X86Sse2Matmul, X86Sse2Reduction);
#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
impl_kernel_scalar!(f64, primitive_mul_add_f64, X86Sse2Matmul, X86Sse2Reduction);
