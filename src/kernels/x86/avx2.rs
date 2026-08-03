use crate::{Matrix, Vector};

#[cfg(not(target_feature = "fma"))]
use super::super::{MatmulBackend, ReductionBackend};

#[cfg(not(target_feature = "fma"))]
#[doc(hidden)]
pub struct X86Avx2Reduction;

#[cfg(not(target_feature = "fma"))]
impl ReductionBackend<f32> for X86Avx2Reduction {
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
        unsafe { reduction_matvec_f32(matrix, vector, output) }
    }
}

#[cfg(not(target_feature = "fma"))]
impl ReductionBackend<f64> for X86Avx2Reduction {
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
        unsafe { reduction_matvec_f64(matrix, vector, output) }
    }
}

#[cfg(not(target_feature = "fma"))]
#[target_feature(enable = "avx2")]
unsafe fn reduction_dot_f32<const M: usize>(lhs: &Vector<M, f32>, rhs: &Vector<M, f32>) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let mut accumulator = _mm256_setzero_ps();
    let mut index = 0;
    while index + 8 <= M {
        let left = _mm256_loadu_ps(lhs.as_slice().as_ptr().add(index));
        let right = _mm256_loadu_ps(rhs.as_slice().as_ptr().add(index));
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(left, right));
        index += 8;
    }
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
    let mut result =
        lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    while index < M {
        result += lhs[index] * rhs[index];
        index += 1;
    }
    result
}

#[cfg(not(target_feature = "fma"))]
#[target_feature(enable = "avx2")]
unsafe fn reduction_dot_f64<const M: usize>(lhs: &Vector<M, f64>, rhs: &Vector<M, f64>) -> f64 {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };
    let mut accumulator = _mm256_setzero_pd();
    let mut index = 0;
    while index + 4 <= M {
        let left = _mm256_loadu_pd(lhs.as_slice().as_ptr().add(index));
        let right = _mm256_loadu_pd(rhs.as_slice().as_ptr().add(index));
        accumulator = _mm256_add_pd(accumulator, _mm256_mul_pd(left, right));
        index += 4;
    }
    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), accumulator);
    let mut result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while index < M {
        result += lhs[index] * rhs[index];
        index += 1;
    }
    result
}

#[cfg(not(target_feature = "fma"))]
#[target_feature(enable = "avx2")]
unsafe fn reduction_squared_norm_f32<const M: usize, const N: usize>(
    matrix: &Matrix<M, N, f32>,
) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let values = matrix.as_slice();
    let mut accumulator = _mm256_setzero_ps();
    let mut index = 0;
    while index + 8 <= values.len() {
        let packet = _mm256_loadu_ps(values.as_ptr().add(index));
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(packet, packet));
        index += 8;
    }
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator);
    let mut result =
        lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    while index < values.len() {
        result += values[index] * values[index];
        index += 1;
    }
    result
}

#[cfg(not(target_feature = "fma"))]
#[target_feature(enable = "avx2")]
unsafe fn reduction_squared_norm_f64<const M: usize, const N: usize>(
    matrix: &Matrix<M, N, f64>,
) -> f64 {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };
    let values = matrix.as_slice();
    let mut accumulator = _mm256_setzero_pd();
    let mut index = 0;
    while index + 4 <= values.len() {
        let packet = _mm256_loadu_pd(values.as_ptr().add(index));
        accumulator = _mm256_add_pd(accumulator, _mm256_mul_pd(packet, packet));
        index += 4;
    }
    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), accumulator);
    let mut result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while index < values.len() {
        result += values[index] * values[index];
        index += 1;
    }
    result
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn reduction_matvec_f32<const M: usize, const N: usize>(
    matrix: &Matrix<M, N, f32>,
    vector: &Vector<N, f32>,
    output: &mut Vector<M, f32>,
) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };
    let mut row = 0;
    while row + 8 <= M {
        let mut accumulator = _mm256_setzero_ps();
        for column in 0..N {
            let packet = _mm256_loadu_ps(matrix.as_slice().as_ptr().add(column * M + row));
            accumulator = _mm256_add_ps(
                accumulator,
                _mm256_mul_ps(packet, _mm256_set1_ps(vector[column])),
            );
        }
        _mm256_storeu_ps(output.as_mut_slice().as_mut_ptr().add(row), accumulator);
        row += 8;
    }
    while row < M {
        let mut value = 0.0_f32;
        for column in 0..N {
            value += matrix[(row, column)] * vector[column];
        }
        output[row] = value;
        row += 1;
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn reduction_matvec_f64<const M: usize, const N: usize>(
    matrix: &Matrix<M, N, f64>,
    vector: &Vector<N, f64>,
    output: &mut Vector<M, f64>,
) {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd,
        _mm256_storeu_pd,
    };
    let mut row = 0;
    while row + 4 <= M {
        let mut accumulator = _mm256_setzero_pd();
        for column in 0..N {
            let packet = _mm256_loadu_pd(matrix.as_slice().as_ptr().add(column * M + row));
            accumulator = _mm256_add_pd(
                accumulator,
                _mm256_mul_pd(packet, _mm256_set1_pd(vector[column])),
            );
        }
        _mm256_storeu_pd(output.as_mut_slice().as_mut_ptr().add(row), accumulator);
        row += 4;
    }
    while row < M {
        let mut value = 0.0_f64;
        for column in 0..N {
            value += matrix[(row, column)] * vector[column];
        }
        output[row] = value;
        row += 1;
    }
}

#[cfg(not(target_feature = "fma"))]
#[doc(hidden)]
pub struct X86Avx2Matmul;

#[cfg(not(target_feature = "fma"))]
impl MatmulBackend<f32> for X86Avx2Matmul {
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, f32>,
        rhs: &Matrix<N, P, f32>,
        output: &mut Matrix<M, P, f32>,
    ) {
        unsafe { matmul_f32(lhs, rhs, output) }
    }

    #[inline]
    fn symmetric_rank_k_update<const D: usize>(
        matrix: &mut Matrix<D, D, f32>,
        block_start: usize,
        block_end: usize,
    ) {
        unsafe { rank_k_update_f32(matrix, block_start, block_end) }
    }

    #[inline]
    fn scale_divide(target: &mut [f32], divisor: f32) {
        unsafe { scale_divide_f32(target, divisor) }
    }
}

#[cfg(not(target_feature = "fma"))]
impl MatmulBackend<f64> for X86Avx2Matmul {
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, f64>,
        rhs: &Matrix<N, P, f64>,
        output: &mut Matrix<M, P, f64>,
    ) {
        unsafe { matmul_f64(lhs, rhs, output) }
    }

    #[inline]
    fn symmetric_rank_k_update<const D: usize>(
        matrix: &mut Matrix<D, D, f64>,
        block_start: usize,
        block_end: usize,
    ) {
        unsafe { rank_k_update_f64(matrix, block_start, block_end) }
    }

    #[inline]
    fn scale_divide(target: &mut [f64], divisor: f64) {
        unsafe { scale_divide_f64(target, divisor) }
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn scale_divide_f32(target: &mut [f32], divisor: f32) {
    use core::arch::x86_64::{_mm256_div_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};
    let mut index = 0;
    let divisor_packet = _mm256_set1_ps(divisor);
    while index + 8 <= target.len() {
        let value = _mm256_loadu_ps(target.as_ptr().add(index));
        _mm256_storeu_ps(
            target.as_mut_ptr().add(index),
            _mm256_div_ps(value, divisor_packet),
        );
        index += 8;
    }
    while index < target.len() {
        *target.get_unchecked_mut(index) = *target.get_unchecked(index) / divisor;
        index += 1;
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn scale_divide_f64(target: &mut [f64], divisor: f64) {
    use core::arch::x86_64::{_mm256_div_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};
    let mut index = 0;
    let divisor_packet = _mm256_set1_pd(divisor);
    while index + 4 <= target.len() {
        let value = _mm256_loadu_pd(target.as_ptr().add(index));
        _mm256_storeu_pd(
            target.as_mut_ptr().add(index),
            _mm256_div_pd(value, divisor_packet),
        );
        index += 4;
    }
    while index < target.len() {
        *target.get_unchecked_mut(index) = *target.get_unchecked(index) / divisor;
        index += 1;
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn rank_k_update_f32<const D: usize>(
    matrix: &mut Matrix<D, D, f32>,
    block_start: usize,
    block_end: usize,
) {
    use core::arch::x86_64::{
        _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };
    let data = matrix.as_mut_slice();
    for column in block_end..D {
        let mut row = column;
        while row + 8 <= D {
            let output = data.as_ptr().add(column * D + row);
            let mut accumulator = _mm256_loadu_ps(output);
            for index in block_start..block_end {
                let left = _mm256_loadu_ps(data.as_ptr().add(index * D + row));
                let scale = _mm256_set1_ps(data[index * D + index] * data[index * D + column]);
                accumulator = _mm256_sub_ps(accumulator, _mm256_mul_ps(left, scale));
            }
            _mm256_storeu_ps(data.as_mut_ptr().add(column * D + row), accumulator);
            row += 8;
        }
        while row < D {
            let mut value = data[column * D + row];
            for index in block_start..block_end {
                value -= data[index * D + row] * data[index * D + index] * data[index * D + column];
            }
            data[column * D + row] = value;
            row += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn rank_k_update_f64<const D: usize>(
    matrix: &mut Matrix<D, D, f64>,
    block_start: usize,
    block_end: usize,
) {
    use core::arch::x86_64::{
        _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    let data = matrix.as_mut_slice();
    for column in block_end..D {
        let mut row = column;
        while row + 4 <= D {
            let mut accumulator = _mm256_loadu_pd(data.as_ptr().add(column * D + row));
            for index in block_start..block_end {
                let left = _mm256_loadu_pd(data.as_ptr().add(index * D + row));
                let scale = _mm256_set1_pd(data[index * D + index] * data[index * D + column]);
                accumulator = _mm256_sub_pd(accumulator, _mm256_mul_pd(left, scale));
            }
            _mm256_storeu_pd(data.as_mut_ptr().add(column * D + row), accumulator);
            row += 4;
        }
        while row < D {
            let mut value = data[column * D + row];
            for index in block_start..block_end {
                value -= data[index * D + row] * data[index * D + index] * data[index * D + column];
            }
            data[column * D + row] = value;
            row += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn rank_update_sub_f32(target: &mut [f32], source: &[f32], scale: f32) {
    use core::arch::x86_64::{
        _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };
    let mut index = 0;
    let scale_packet = _mm256_set1_ps(scale);
    while index + 8 <= target.len() {
        let value = _mm256_loadu_ps(target.as_ptr().add(index));
        let product = _mm256_mul_ps(_mm256_loadu_ps(source.as_ptr().add(index)), scale_packet);
        _mm256_storeu_ps(
            target.as_mut_ptr().add(index),
            _mm256_sub_ps(value, product),
        );
        index += 8;
    }
    while index < target.len() {
        *target.get_unchecked_mut(index) =
            *target.get_unchecked(index) - *source.get_unchecked(index) * scale;
        index += 1;
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn rank_update_sub_f64(target: &mut [f64], source: &[f64], scale: f64) {
    use core::arch::x86_64::{
        _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    let mut index = 0;
    let scale_packet = _mm256_set1_pd(scale);
    while index + 4 <= target.len() {
        let value = _mm256_loadu_pd(target.as_ptr().add(index));
        let product = _mm256_mul_pd(_mm256_loadu_pd(source.as_ptr().add(index)), scale_packet);
        _mm256_storeu_pd(
            target.as_mut_ptr().add(index),
            _mm256_sub_pd(value, product),
        );
        index += 4;
    }
    while index < target.len() {
        *target.get_unchecked_mut(index) =
            *target.get_unchecked(index) - *source.get_unchecked(index) * scale;
        index += 1;
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn matmul_f32<const M: usize, const N: usize, const P: usize>(
    lhs: &Matrix<M, N, f32>,
    rhs: &Matrix<N, P, f32>,
    output: &mut Matrix<M, P, f32>,
) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };
    for column in 0..P {
        let mut row = 0;
        while row + 8 <= M {
            let mut accumulator = _mm256_setzero_ps();
            for shared in 0..N {
                let lhs_ptr = lhs.as_slice().as_ptr().add(shared * M + row);
                let lhs_packet = _mm256_loadu_ps(lhs_ptr);
                let rhs_packet = _mm256_set1_ps(rhs[(shared, column)]);
                accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(lhs_packet, rhs_packet));
            }
            let output_ptr = output.as_mut_slice().as_mut_ptr().add(column * M + row);
            _mm256_storeu_ps(output_ptr, accumulator);
            row += 8;
        }
        for row in row..M {
            let mut accumulator = 0.0_f32;
            for shared in 0..N {
                accumulator += lhs[(row, shared)] * rhs[(shared, column)];
            }
            output[(row, column)] = accumulator;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn matmul_f64<const M: usize, const N: usize, const P: usize>(
    lhs: &Matrix<M, N, f64>,
    rhs: &Matrix<N, P, f64>,
    output: &mut Matrix<M, P, f64>,
) {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd,
        _mm256_storeu_pd,
    };
    for column in 0..P {
        let mut row = 0;
        while row + 4 <= M {
            let mut accumulator = _mm256_setzero_pd();
            for shared in 0..N {
                let lhs_ptr = lhs.as_slice().as_ptr().add(shared * M + row);
                let lhs_packet = _mm256_loadu_pd(lhs_ptr);
                let rhs_packet = _mm256_set1_pd(rhs[(shared, column)]);
                accumulator = _mm256_add_pd(accumulator, _mm256_mul_pd(lhs_packet, rhs_packet));
            }
            let output_ptr = output.as_mut_slice().as_mut_ptr().add(column * M + row);
            _mm256_storeu_pd(output_ptr, accumulator);
            row += 4;
        }
        for row in row..M {
            let mut accumulator = 0.0_f64;
            for shared in 0..N {
                accumulator += lhs[(row, shared)] * rhs[(shared, column)];
            }
            output[(row, column)] = accumulator;
        }
    }
}
