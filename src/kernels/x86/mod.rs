use super::{MatmulBackend, MatrixScalar, ReductionBackend, ReductionScalar};
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

macro_rules! impl_kernel_scalar {
    ($scalar:ty, $matmul:ty, $reduction:ty) => {
        impl MatrixScalar for $scalar {
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
            fn scale_divide(target: &mut [Self], divisor: Self) {
                <$matmul as MatmulBackend<$scalar>>::scale_divide(target, divisor);
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
impl_kernel_scalar!(f32, X86Avx2Matmul, X86Avx2Reduction);
#[cfg(target_feature = "avx2")]
impl_kernel_scalar!(f64, X86Avx2Matmul, X86Avx2Reduction);

#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
impl_kernel_scalar!(f32, X86Sse2Matmul, X86Sse2Reduction);
#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
impl_kernel_scalar!(f64, X86Sse2Matmul, X86Sse2Reduction);
