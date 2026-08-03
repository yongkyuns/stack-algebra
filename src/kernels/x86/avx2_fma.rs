use crate::{Matrix, Vector};

use super::super::{MatmulBackend, ReductionBackend};

#[doc(hidden)]
pub struct X86Avx2FmaReduction;

impl ReductionBackend<f32> for X86Avx2FmaReduction {
    #[inline]
    fn dot<const M: usize>(lhs: &Vector<M, f32>, rhs: &Vector<M, f32>) -> f32 {
        unsafe { reduction_dot_f32(lhs, rhs) }
    }

    #[inline]
    fn squared_norm<const M: usize, const N: usize>(matrix: &Matrix<M, N, f32>) -> f32 {
        unsafe { reduction_squared_norm_f32(matrix) }
    }

    #[inline]
    fn matvec<const M: usize, const N: usize>(
        matrix: &Matrix<M, N, f32>,
        vector: &Vector<N, f32>,
        output: &mut Vector<M, f32>,
    ) {
        unsafe { super::avx2::reduction_matvec_f32(matrix, vector, output) }
    }
}

impl ReductionBackend<f64> for X86Avx2FmaReduction {
    #[inline]
    fn dot<const M: usize>(lhs: &Vector<M, f64>, rhs: &Vector<M, f64>) -> f64 {
        unsafe { reduction_dot_f64(lhs, rhs) }
    }

    #[inline]
    fn squared_norm<const M: usize, const N: usize>(matrix: &Matrix<M, N, f64>) -> f64 {
        unsafe { reduction_squared_norm_f64(matrix) }
    }

    #[inline]
    fn matvec<const M: usize, const N: usize>(
        matrix: &Matrix<M, N, f64>,
        vector: &Vector<N, f64>,
        output: &mut Vector<M, f64>,
    ) {
        unsafe { super::avx2::reduction_matvec_f64(matrix, vector, output) }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn reduction_dot_f32<const M: usize>(lhs: &Vector<M, f32>, rhs: &Vector<M, f32>) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    if M < 8 {
        let mut result = 0.0_f32;
        let mut index = 0;
        while index < M {
            result += lhs[index] * rhs[index];
            index += 1;
        }
        return result;
    }

    let lhs_values = lhs.as_slice();
    let rhs_values = rhs.as_slice();
    if M < 32 {
        let mut accumulator = _mm256_setzero_ps();
        let mut index = 0;
        while index + 8 <= M {
            accumulator = _mm256_fmadd_ps(
                _mm256_loadu_ps(lhs_values.as_ptr().add(index)),
                _mm256_loadu_ps(rhs_values.as_ptr().add(index)),
                accumulator,
            );
            index += 8;
        }
        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        let mut result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
        result += lanes[4] + lanes[5] + lanes[6] + lanes[7];
        while index < M {
            result += lhs_values[index] * rhs_values[index];
            index += 1;
        }
        return result;
    }

    let mut accumulator0 = _mm256_setzero_ps();
    let mut accumulator1 = _mm256_setzero_ps();
    let mut accumulator2 = _mm256_setzero_ps();
    let mut accumulator3 = _mm256_setzero_ps();
    let mut index = 0;
    while index + 32 <= M {
        accumulator0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(lhs_values.as_ptr().add(index)),
            _mm256_loadu_ps(rhs_values.as_ptr().add(index)),
            accumulator0,
        );
        accumulator1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(lhs_values.as_ptr().add(index + 8)),
            _mm256_loadu_ps(rhs_values.as_ptr().add(index + 8)),
            accumulator1,
        );
        accumulator2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(lhs_values.as_ptr().add(index + 16)),
            _mm256_loadu_ps(rhs_values.as_ptr().add(index + 16)),
            accumulator2,
        );
        accumulator3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(lhs_values.as_ptr().add(index + 24)),
            _mm256_loadu_ps(rhs_values.as_ptr().add(index + 24)),
            accumulator3,
        );
        index += 32;
    }
    while index + 8 <= M {
        accumulator0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(lhs_values.as_ptr().add(index)),
            _mm256_loadu_ps(rhs_values.as_ptr().add(index)),
            accumulator0,
        );
        index += 8;
    }

    accumulator0 = _mm256_add_ps(accumulator0, accumulator1);
    accumulator2 = _mm256_add_ps(accumulator2, accumulator3);
    accumulator0 = _mm256_add_ps(accumulator0, accumulator2);
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator0);
    let mut result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    result += lanes[4] + lanes[5] + lanes[6] + lanes[7];
    while index < M {
        result += lhs_values[index] * rhs_values[index];
        index += 1;
    }
    result
}

#[target_feature(enable = "avx2,fma")]
unsafe fn reduction_dot_f64<const M: usize>(lhs: &Vector<M, f64>, rhs: &Vector<M, f64>) -> f64 {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };

    if M < 4 {
        let mut result = 0.0_f64;
        let mut index = 0;
        while index < M {
            result += lhs[index] * rhs[index];
            index += 1;
        }
        return result;
    }

    let lhs_values = lhs.as_slice();
    let rhs_values = rhs.as_slice();
    if M < 16 {
        let mut accumulator = _mm256_setzero_pd();
        let mut index = 0;
        while index + 4 <= M {
            accumulator = _mm256_fmadd_pd(
                _mm256_loadu_pd(lhs_values.as_ptr().add(index)),
                _mm256_loadu_pd(rhs_values.as_ptr().add(index)),
                accumulator,
            );
            index += 4;
        }
        let mut lanes = [0.0_f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), accumulator);
        let mut result = lanes[0] + lanes[1];
        result += lanes[2] + lanes[3];
        while index < M {
            result += lhs_values[index] * rhs_values[index];
            index += 1;
        }
        return result;
    }

    let mut accumulator0 = _mm256_setzero_pd();
    let mut accumulator1 = _mm256_setzero_pd();
    let mut accumulator2 = _mm256_setzero_pd();
    let mut accumulator3 = _mm256_setzero_pd();
    let mut index = 0;
    while index + 16 <= M {
        accumulator0 = _mm256_fmadd_pd(
            _mm256_loadu_pd(lhs_values.as_ptr().add(index)),
            _mm256_loadu_pd(rhs_values.as_ptr().add(index)),
            accumulator0,
        );
        accumulator1 = _mm256_fmadd_pd(
            _mm256_loadu_pd(lhs_values.as_ptr().add(index + 4)),
            _mm256_loadu_pd(rhs_values.as_ptr().add(index + 4)),
            accumulator1,
        );
        accumulator2 = _mm256_fmadd_pd(
            _mm256_loadu_pd(lhs_values.as_ptr().add(index + 8)),
            _mm256_loadu_pd(rhs_values.as_ptr().add(index + 8)),
            accumulator2,
        );
        accumulator3 = _mm256_fmadd_pd(
            _mm256_loadu_pd(lhs_values.as_ptr().add(index + 12)),
            _mm256_loadu_pd(rhs_values.as_ptr().add(index + 12)),
            accumulator3,
        );
        index += 16;
    }
    while index + 4 <= M {
        accumulator0 = _mm256_fmadd_pd(
            _mm256_loadu_pd(lhs_values.as_ptr().add(index)),
            _mm256_loadu_pd(rhs_values.as_ptr().add(index)),
            accumulator0,
        );
        index += 4;
    }

    accumulator0 = _mm256_add_pd(accumulator0, accumulator1);
    accumulator2 = _mm256_add_pd(accumulator2, accumulator3);
    accumulator0 = _mm256_add_pd(accumulator0, accumulator2);
    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), accumulator0);
    let mut result = lanes[0] + lanes[1];
    result += lanes[2] + lanes[3];
    while index < M {
        result += lhs_values[index] * rhs_values[index];
        index += 1;
    }
    result
}

#[target_feature(enable = "avx2,fma")]
unsafe fn reduction_squared_norm_f32<const M: usize, const N: usize>(
    matrix: &Matrix<M, N, f32>,
) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let values = matrix.as_slice();
    if values.len() < 8 {
        let mut result = 0.0_f32;
        for &value in values {
            result += value * value;
        }
        return result;
    }

    if values.len() < 32 {
        let mut accumulator = _mm256_setzero_ps();
        let mut index = 0;
        while index + 8 <= values.len() {
            let packet = _mm256_loadu_ps(values.as_ptr().add(index));
            accumulator = _mm256_fmadd_ps(packet, packet, accumulator);
            index += 8;
        }
        let mut lanes = [0.0_f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
        let mut result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
        result += lanes[4] + lanes[5] + lanes[6] + lanes[7];
        while index < values.len() {
            result += values[index] * values[index];
            index += 1;
        }
        return result;
    }

    let mut accumulator0 = _mm256_setzero_ps();
    let mut accumulator1 = _mm256_setzero_ps();
    let mut accumulator2 = _mm256_setzero_ps();
    let mut accumulator3 = _mm256_setzero_ps();
    let mut index = 0;
    while index + 32 <= values.len() {
        accumulator0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(values.as_ptr().add(index)),
            _mm256_loadu_ps(values.as_ptr().add(index)),
            accumulator0,
        );
        accumulator1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(values.as_ptr().add(index + 8)),
            _mm256_loadu_ps(values.as_ptr().add(index + 8)),
            accumulator1,
        );
        accumulator2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(values.as_ptr().add(index + 16)),
            _mm256_loadu_ps(values.as_ptr().add(index + 16)),
            accumulator2,
        );
        accumulator3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(values.as_ptr().add(index + 24)),
            _mm256_loadu_ps(values.as_ptr().add(index + 24)),
            accumulator3,
        );
        index += 32;
    }
    while index + 8 <= values.len() {
        let packet = _mm256_loadu_ps(values.as_ptr().add(index));
        accumulator0 = _mm256_fmadd_ps(packet, packet, accumulator0);
        index += 8;
    }

    accumulator0 = _mm256_add_ps(accumulator0, accumulator1);
    accumulator2 = _mm256_add_ps(accumulator2, accumulator3);
    accumulator0 = _mm256_add_ps(accumulator0, accumulator2);
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator0);
    let mut result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    result += lanes[4] + lanes[5] + lanes[6] + lanes[7];
    while index < values.len() {
        result += values[index] * values[index];
        index += 1;
    }
    result
}

#[target_feature(enable = "avx2,fma")]
unsafe fn reduction_squared_norm_f64<const M: usize, const N: usize>(
    matrix: &Matrix<M, N, f64>,
) -> f64 {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };
    let values = matrix.as_slice();
    if values.len() < 4 {
        let mut result = 0.0_f64;
        for &value in values {
            result += value * value;
        }
        return result;
    }

    if values.len() < 16 {
        let mut accumulator = _mm256_setzero_pd();
        let mut index = 0;
        while index + 4 <= values.len() {
            let packet = _mm256_loadu_pd(values.as_ptr().add(index));
            accumulator = _mm256_fmadd_pd(packet, packet, accumulator);
            index += 4;
        }
        let mut lanes = [0.0_f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), accumulator);
        let mut result = lanes[0] + lanes[1];
        result += lanes[2] + lanes[3];
        while index < values.len() {
            result += values[index] * values[index];
            index += 1;
        }
        return result;
    }

    let mut accumulator0 = _mm256_setzero_pd();
    let mut accumulator1 = _mm256_setzero_pd();
    let mut accumulator2 = _mm256_setzero_pd();
    let mut accumulator3 = _mm256_setzero_pd();
    let mut index = 0;
    while index + 16 <= values.len() {
        accumulator0 = _mm256_fmadd_pd(
            _mm256_loadu_pd(values.as_ptr().add(index)),
            _mm256_loadu_pd(values.as_ptr().add(index)),
            accumulator0,
        );
        accumulator1 = _mm256_fmadd_pd(
            _mm256_loadu_pd(values.as_ptr().add(index + 4)),
            _mm256_loadu_pd(values.as_ptr().add(index + 4)),
            accumulator1,
        );
        accumulator2 = _mm256_fmadd_pd(
            _mm256_loadu_pd(values.as_ptr().add(index + 8)),
            _mm256_loadu_pd(values.as_ptr().add(index + 8)),
            accumulator2,
        );
        accumulator3 = _mm256_fmadd_pd(
            _mm256_loadu_pd(values.as_ptr().add(index + 12)),
            _mm256_loadu_pd(values.as_ptr().add(index + 12)),
            accumulator3,
        );
        index += 16;
    }
    while index + 4 <= values.len() {
        let packet = _mm256_loadu_pd(values.as_ptr().add(index));
        accumulator0 = _mm256_fmadd_pd(packet, packet, accumulator0);
        index += 4;
    }

    accumulator0 = _mm256_add_pd(accumulator0, accumulator1);
    accumulator2 = _mm256_add_pd(accumulator2, accumulator3);
    accumulator0 = _mm256_add_pd(accumulator0, accumulator2);
    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), accumulator0);
    let mut result = lanes[0] + lanes[1];
    result += lanes[2] + lanes[3];
    while index < values.len() {
        result += values[index] * values[index];
        index += 1;
    }
    result
}

#[doc(hidden)]
pub struct X86Avx2FmaMatmul;

impl MatmulBackend<f32> for X86Avx2FmaMatmul {
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, f32>,
        rhs: &Matrix<N, P, f32>,
        output: &mut Matrix<M, P, f32>,
    ) {
        unsafe { super::avx2::matmul_f32(lhs, rhs, output) }
    }

    #[inline]
    fn rank_update_sub(target: &mut [f32], source: &[f32], scale: f32) {
        unsafe { super::avx2::rank_update_sub_f32(target, source, scale) }
    }

    #[inline]
    fn symmetric_rank_k_update<const D: usize>(
        matrix: &mut Matrix<D, D, f32>,
        block_start: usize,
        block_end: usize,
    ) {
        unsafe { super::avx2::rank_k_update_f32(matrix, block_start, block_end) }
    }

    #[inline]
    fn scale_divide(target: &mut [f32], divisor: f32) {
        unsafe { super::avx2::scale_divide_f32(target, divisor) }
    }
}

impl MatmulBackend<f64> for X86Avx2FmaMatmul {
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, f64>,
        rhs: &Matrix<N, P, f64>,
        output: &mut Matrix<M, P, f64>,
    ) {
        unsafe { super::avx2::matmul_f64(lhs, rhs, output) }
    }

    #[inline]
    fn rank_update_sub(target: &mut [f64], source: &[f64], scale: f64) {
        unsafe { super::avx2::rank_update_sub_f64(target, source, scale) }
    }

    #[inline]
    fn symmetric_rank_k_update<const D: usize>(
        matrix: &mut Matrix<D, D, f64>,
        block_start: usize,
        block_end: usize,
    ) {
        unsafe { super::avx2::rank_k_update_f64(matrix, block_start, block_end) }
    }

    #[inline]
    fn scale_divide(target: &mut [f64], divisor: f64) {
        unsafe { super::avx2::scale_divide_f64(target, divisor) }
    }
}
