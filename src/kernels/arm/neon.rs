use crate::{Matrix, Vector};

use super::super::{MatmulBackend, ReductionBackend};

#[doc(hidden)]
pub struct NeonMatmul;

#[doc(hidden)]
pub struct NeonReduction;

impl MatmulBackend<f32> for NeonMatmul {
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, f32>,
        rhs: &Matrix<N, P, f32>,
        output: &mut Matrix<M, P, f32>,
    ) {
        unsafe { matmul_f32(lhs, rhs, output) }
    }

    #[inline]
    fn dot(lhs: &[f32], rhs: &[f32], initial: f32) -> f32 {
        unsafe { dot_slices_f32(lhs, rhs, initial) }
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
    fn rank_update_sub(target: &mut [f32], source: &[f32], scale: f32) {
        unsafe { rank_update_sub_f32(target, source, scale) }
    }

    #[inline]
    fn rank_update_two_sub(
        target: &mut [f32],
        source_first: &[f32],
        scale_first: f32,
        source_second: &[f32],
        scale_second: f32,
    ) {
        unsafe {
            rank_update_two_sub_f32(
                target,
                source_first,
                scale_first,
                source_second,
                scale_second,
            )
        }
    }

    #[inline]
    fn scale_divide(target: &mut [f32], divisor: f32) {
        unsafe { scale_divide_f32(target, divisor) }
    }
}

impl MatmulBackend<f64> for NeonMatmul {
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, f64>,
        rhs: &Matrix<N, P, f64>,
        output: &mut Matrix<M, P, f64>,
    ) {
        unsafe { matmul_f64(lhs, rhs, output) }
    }

    #[inline]
    fn dot(lhs: &[f64], rhs: &[f64], initial: f64) -> f64 {
        unsafe { dot_slices_f64(lhs, rhs, initial) }
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
    fn rank_update_sub(target: &mut [f64], source: &[f64], scale: f64) {
        unsafe { rank_update_sub_f64(target, source, scale) }
    }

    #[inline]
    fn rank_update_two_sub(
        target: &mut [f64],
        source_first: &[f64],
        scale_first: f64,
        source_second: &[f64],
        scale_second: f64,
    ) {
        unsafe {
            rank_update_two_sub_f64(
                target,
                source_first,
                scale_first,
                source_second,
                scale_second,
            )
        }
    }

    #[inline]
    fn scale_divide(target: &mut [f64], divisor: f64) {
        unsafe { scale_divide_f64(target, divisor) }
    }
}

impl ReductionBackend<f32> for NeonReduction {
    #[inline]
    fn dot<const M: usize>(lhs: &Vector<M, f32>, rhs: &Vector<M, f32>) -> f32 {
        unsafe { dot_f32(lhs, rhs) }
    }

    #[inline]
    fn squared_norm<const M: usize, const N: usize>(matrix: &Matrix<M, N, f32>) -> f32 {
        unsafe { squared_norm_f32(matrix) }
    }

    #[inline]
    fn matvec<const M: usize, const N: usize>(
        matrix: &Matrix<M, N, f32>,
        vector: &Vector<N, f32>,
        output: &mut Vector<M, f32>,
    ) {
        unsafe { matvec_f32(matrix, vector, output) }
    }
}

impl ReductionBackend<f64> for NeonReduction {
    #[inline]
    fn dot<const M: usize>(lhs: &Vector<M, f64>, rhs: &Vector<M, f64>) -> f64 {
        unsafe { dot_f64(lhs, rhs) }
    }

    #[inline]
    fn squared_norm<const M: usize, const N: usize>(matrix: &Matrix<M, N, f64>) -> f64 {
        unsafe { squared_norm_f64(matrix) }
    }

    #[inline]
    fn matvec<const M: usize, const N: usize>(
        matrix: &Matrix<M, N, f64>,
        vector: &Vector<N, f64>,
        output: &mut Vector<M, f64>,
    ) {
        unsafe { matvec_f64(matrix, vector, output) }
    }
}

#[target_feature(enable = "neon")]
unsafe fn matmul_f32<const M: usize, const N: usize, const P: usize>(
    lhs: &Matrix<M, N, f32>,
    rhs: &Matrix<N, P, f32>,
    output: &mut Matrix<M, P, f32>,
) {
    use core::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};
    for column in 0..P {
        let mut row = 0;
        while row + 4 <= M {
            let mut accumulator = vdupq_n_f32(0.0);
            for shared in 0..N {
                let left = vld1q_f32(lhs.as_slice().as_ptr().add(shared * M + row));
                let right = vdupq_n_f32(rhs[(shared, column)]);
                accumulator = vaddq_f32(accumulator, vmulq_f32(left, right));
            }
            vst1q_f32(
                output.as_mut_slice().as_mut_ptr().add(column * M + row),
                accumulator,
            );
            row += 4;
        }
        while row < M {
            let mut value = 0.0;
            for shared in 0..N {
                value += lhs[(row, shared)] * rhs[(shared, column)];
            }
            output[(row, column)] = value;
            row += 1;
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn matmul_f64<const M: usize, const N: usize, const P: usize>(
    lhs: &Matrix<M, N, f64>,
    rhs: &Matrix<N, P, f64>,
    output: &mut Matrix<M, P, f64>,
) {
    use core::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};
    for column in 0..P {
        let mut row = 0;
        while row + 2 <= M {
            let mut accumulator = vdupq_n_f64(0.0);
            for shared in 0..N {
                let left = vld1q_f64(lhs.as_slice().as_ptr().add(shared * M + row));
                let right = vdupq_n_f64(rhs[(shared, column)]);
                accumulator = vaddq_f64(accumulator, vmulq_f64(left, right));
            }
            vst1q_f64(
                output.as_mut_slice().as_mut_ptr().add(column * M + row),
                accumulator,
            );
            row += 2;
        }
        while row < M {
            let mut value = 0.0;
            for shared in 0..N {
                value += lhs[(row, shared)] * rhs[(shared, column)];
            }
            output[(row, column)] = value;
            row += 1;
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn rank_k_update_f32<const D: usize>(
    matrix: &mut Matrix<D, D, f32>,
    block_start: usize,
    block_end: usize,
) {
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32};
    let data = matrix.as_mut_slice();
    for column in block_end..D {
        let mut row = column;
        while row + 4 <= D {
            let mut accumulator = vld1q_f32(data.as_ptr().add(column * D + row));
            for index in block_start..block_end {
                let left = vld1q_f32(data.as_ptr().add(index * D + row));
                let scale = vdupq_n_f32(data[index * D + index] * data[index * D + column]);
                accumulator = vsubq_f32(accumulator, vmulq_f32(left, scale));
            }
            vst1q_f32(data.as_mut_ptr().add(column * D + row), accumulator);
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

#[target_feature(enable = "neon")]
unsafe fn rank_k_update_f64<const D: usize>(
    matrix: &mut Matrix<D, D, f64>,
    block_start: usize,
    block_end: usize,
) {
    use core::arch::aarch64::{vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64};
    let data = matrix.as_mut_slice();
    for column in block_end..D {
        let mut row = column;
        while row + 2 <= D {
            let mut accumulator = vld1q_f64(data.as_ptr().add(column * D + row));
            for index in block_start..block_end {
                let left = vld1q_f64(data.as_ptr().add(index * D + row));
                let scale = vdupq_n_f64(data[index * D + index] * data[index * D + column]);
                accumulator = vsubq_f64(accumulator, vmulq_f64(left, scale));
            }
            vst1q_f64(data.as_mut_ptr().add(column * D + row), accumulator);
            row += 2;
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

#[target_feature(enable = "neon")]
unsafe fn rank_update_sub_f32(target: &mut [f32], source: &[f32], scale: f32) {
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32};
    let mut index = 0;
    let scale_packet = vdupq_n_f32(scale);
    while index + 4 <= target.len() {
        let value = vld1q_f32(target.as_ptr().add(index));
        let product = vmulq_f32(vld1q_f32(source.as_ptr().add(index)), scale_packet);
        vst1q_f32(target.as_mut_ptr().add(index), vsubq_f32(value, product));
        index += 4;
    }
    while index < target.len() {
        *target.get_unchecked_mut(index) =
            *target.get_unchecked(index) - *source.get_unchecked(index) * scale;
        index += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn rank_update_sub_f64(target: &mut [f64], source: &[f64], scale: f64) {
    use core::arch::aarch64::{vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64};
    let mut index = 0;
    let scale_packet = vdupq_n_f64(scale);
    while index + 2 <= target.len() {
        let value = vld1q_f64(target.as_ptr().add(index));
        let product = vmulq_f64(vld1q_f64(source.as_ptr().add(index)), scale_packet);
        vst1q_f64(target.as_mut_ptr().add(index), vsubq_f64(value, product));
        index += 2;
    }
    while index < target.len() {
        *target.get_unchecked_mut(index) =
            *target.get_unchecked(index) - *source.get_unchecked(index) * scale;
        index += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn rank_update_two_sub_f32(
    target: &mut [f32],
    source_first: &[f32],
    scale_first: f32,
    source_second: &[f32],
    scale_second: f32,
) {
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32};
    let mut index = 0;
    let scale_first_packet = vdupq_n_f32(scale_first);
    let scale_second_packet = vdupq_n_f32(scale_second);
    while index + 4 <= target.len() {
        let value = vld1q_f32(target.as_ptr().add(index));
        let first = vmulq_f32(
            vld1q_f32(source_first.as_ptr().add(index)),
            scale_first_packet,
        );
        let second = vmulq_f32(
            vld1q_f32(source_second.as_ptr().add(index)),
            scale_second_packet,
        );
        vst1q_f32(
            target.as_mut_ptr().add(index),
            vsubq_f32(vsubq_f32(value, first), second),
        );
        index += 4;
    }
    while index < target.len() {
        *target.get_unchecked_mut(index) = *target.get_unchecked(index)
            - *source_first.get_unchecked(index) * scale_first
            - *source_second.get_unchecked(index) * scale_second;
        index += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn rank_update_two_sub_f64(
    target: &mut [f64],
    source_first: &[f64],
    scale_first: f64,
    source_second: &[f64],
    scale_second: f64,
) {
    use core::arch::aarch64::{vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64};
    let mut index = 0;
    let scale_first_packet = vdupq_n_f64(scale_first);
    let scale_second_packet = vdupq_n_f64(scale_second);
    while index + 2 <= target.len() {
        let value = vld1q_f64(target.as_ptr().add(index));
        let first = vmulq_f64(
            vld1q_f64(source_first.as_ptr().add(index)),
            scale_first_packet,
        );
        let second = vmulq_f64(
            vld1q_f64(source_second.as_ptr().add(index)),
            scale_second_packet,
        );
        vst1q_f64(
            target.as_mut_ptr().add(index),
            vsubq_f64(vsubq_f64(value, first), second),
        );
        index += 2;
    }
    while index < target.len() {
        *target.get_unchecked_mut(index) = *target.get_unchecked(index)
            - *source_first.get_unchecked(index) * scale_first
            - *source_second.get_unchecked(index) * scale_second;
        index += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scale_divide_f32(target: &mut [f32], divisor: f32) {
    use core::arch::aarch64::{vdivq_f32, vdupq_n_f32, vld1q_f32, vst1q_f32};
    let mut index = 0;
    let divisor_packet = vdupq_n_f32(divisor);
    while index + 4 <= target.len() {
        vst1q_f32(
            target.as_mut_ptr().add(index),
            vdivq_f32(vld1q_f32(target.as_ptr().add(index)), divisor_packet),
        );
        index += 4;
    }
    while index < target.len() {
        *target.get_unchecked_mut(index) = *target.get_unchecked(index) / divisor;
        index += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scale_divide_f64(target: &mut [f64], divisor: f64) {
    use core::arch::aarch64::{vdivq_f64, vdupq_n_f64, vld1q_f64, vst1q_f64};
    let mut index = 0;
    let divisor_packet = vdupq_n_f64(divisor);
    while index + 2 <= target.len() {
        vst1q_f64(
            target.as_mut_ptr().add(index),
            vdivq_f64(vld1q_f64(target.as_ptr().add(index)), divisor_packet),
        );
        index += 2;
    }
    while index < target.len() {
        *target.get_unchecked_mut(index) = *target.get_unchecked(index) / divisor;
        index += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn dot_f32<const M: usize>(lhs: &Vector<M, f32>, rhs: &Vector<M, f32>) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32};
    let mut index = 0;
    let mut accumulator = vdupq_n_f32(0.0);
    while index + 4 <= M {
        accumulator = vaddq_f32(
            accumulator,
            vmulq_f32(
                vld1q_f32(lhs.as_slice().as_ptr().add(index)),
                vld1q_f32(rhs.as_slice().as_ptr().add(index)),
            ),
        );
        index += 4;
    }
    let mut result = vaddvq_f32(accumulator);
    while index < M {
        result += lhs[index] * rhs[index];
        index += 1;
    }
    result
}

#[target_feature(enable = "neon")]
unsafe fn dot_f64<const M: usize>(lhs: &Vector<M, f64>, rhs: &Vector<M, f64>) -> f64 {
    use core::arch::aarch64::{vaddq_f64, vaddvq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64};
    let mut index = 0;
    let mut accumulator = vdupq_n_f64(0.0);
    while index + 2 <= M {
        accumulator = vaddq_f64(
            accumulator,
            vmulq_f64(
                vld1q_f64(lhs.as_slice().as_ptr().add(index)),
                vld1q_f64(rhs.as_slice().as_ptr().add(index)),
            ),
        );
        index += 2;
    }
    let mut result = vaddvq_f64(accumulator);
    while index < M {
        result += lhs[index] * rhs[index];
        index += 1;
    }
    result
}

#[target_feature(enable = "neon")]
unsafe fn dot_slices_f32(lhs: &[f32], rhs: &[f32], initial: f32) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32};
    let mut index = 0;
    let mut accumulator = vdupq_n_f32(0.0);
    while index + 4 <= lhs.len() {
        accumulator = vaddq_f32(
            accumulator,
            vmulq_f32(
                vld1q_f32(lhs.as_ptr().add(index)),
                vld1q_f32(rhs.as_ptr().add(index)),
            ),
        );
        index += 4;
    }
    let mut result = initial + vaddvq_f32(accumulator);
    while index < lhs.len() {
        result += lhs[index] * rhs[index];
        index += 1;
    }
    result
}

#[target_feature(enable = "neon")]
unsafe fn dot_slices_f64(lhs: &[f64], rhs: &[f64], initial: f64) -> f64 {
    use core::arch::aarch64::{vaddq_f64, vaddvq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64};
    let mut index = 0;
    let mut accumulator = vdupq_n_f64(0.0);
    while index + 2 <= lhs.len() {
        accumulator = vaddq_f64(
            accumulator,
            vmulq_f64(
                vld1q_f64(lhs.as_ptr().add(index)),
                vld1q_f64(rhs.as_ptr().add(index)),
            ),
        );
        index += 2;
    }
    let mut result = initial + vaddvq_f64(accumulator);
    while index < lhs.len() {
        result += lhs[index] * rhs[index];
        index += 1;
    }
    result
}

#[target_feature(enable = "neon")]
unsafe fn squared_norm_f32<const M: usize, const N: usize>(matrix: &Matrix<M, N, f32>) -> f32 {
    dot_f32_slice(matrix.as_slice())
}

#[target_feature(enable = "neon")]
unsafe fn squared_norm_f64<const M: usize, const N: usize>(matrix: &Matrix<M, N, f64>) -> f64 {
    dot_f64_slice(matrix.as_slice())
}

#[target_feature(enable = "neon")]
unsafe fn dot_f32_slice(values: &[f32]) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32};
    let mut index = 0;
    let mut accumulator = vdupq_n_f32(0.0);
    while index + 4 <= values.len() {
        let packet = vld1q_f32(values.as_ptr().add(index));
        accumulator = vaddq_f32(accumulator, vmulq_f32(packet, packet));
        index += 4;
    }
    let mut result = vaddvq_f32(accumulator);
    while index < values.len() {
        result += values[index] * values[index];
        index += 1;
    }
    result
}

#[target_feature(enable = "neon")]
unsafe fn dot_f64_slice(values: &[f64]) -> f64 {
    use core::arch::aarch64::{vaddq_f64, vaddvq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64};
    let mut index = 0;
    let mut accumulator = vdupq_n_f64(0.0);
    while index + 2 <= values.len() {
        let packet = vld1q_f64(values.as_ptr().add(index));
        accumulator = vaddq_f64(accumulator, vmulq_f64(packet, packet));
        index += 2;
    }
    let mut result = vaddvq_f64(accumulator);
    while index < values.len() {
        result += values[index] * values[index];
        index += 1;
    }
    result
}

#[target_feature(enable = "neon")]
unsafe fn matvec_f32<const M: usize, const N: usize>(
    matrix: &Matrix<M, N, f32>,
    vector: &Vector<N, f32>,
    output: &mut Vector<M, f32>,
) {
    matmul_f32(matrix, vector, output);
}

#[target_feature(enable = "neon")]
unsafe fn matvec_f64<const M: usize, const N: usize>(
    matrix: &Matrix<M, N, f64>,
    vector: &Vector<N, f64>,
    output: &mut Vector<M, f64>,
) {
    matmul_f64(matrix, vector, output);
}
