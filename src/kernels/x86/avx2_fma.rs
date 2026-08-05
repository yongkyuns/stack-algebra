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

#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_f32<const M: usize, const N: usize, const P: usize>(
    lhs: &Matrix<M, N, f32>,
    rhs: &Matrix<N, P, f32>,
    output: &mut Matrix<M, P, f32>,
) {
    use core::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_maskload_ps, _mm256_maskstore_ps, _mm256_set1_ps,
        _mm256_set_epi32, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let lhs_values = lhs.as_slice();
    let rhs_values = rhs.as_slice();
    let output_values = output.as_mut_slice();
    let mut column = 0;
    while P >= 8 && P % 8 <= 2 && column + 8 <= P {
        let rhs0 = rhs_values.as_ptr().add(column * N);
        let rhs1 = rhs_values.as_ptr().add((column + 1) * N);
        let rhs2 = rhs_values.as_ptr().add((column + 2) * N);
        let rhs3 = rhs_values.as_ptr().add((column + 3) * N);
        let rhs4 = rhs_values.as_ptr().add((column + 4) * N);
        let rhs5 = rhs_values.as_ptr().add((column + 5) * N);
        let rhs6 = rhs_values.as_ptr().add((column + 6) * N);
        let rhs7 = rhs_values.as_ptr().add((column + 7) * N);
        let mut row = 0;
        while row + 8 <= M {
            let mut accumulator0 = _mm256_setzero_ps();
            let mut accumulator1 = _mm256_setzero_ps();
            let mut accumulator2 = _mm256_setzero_ps();
            let mut accumulator3 = _mm256_setzero_ps();
            let mut accumulator4 = _mm256_setzero_ps();
            let mut accumulator5 = _mm256_setzero_ps();
            let mut accumulator6 = _mm256_setzero_ps();
            let mut accumulator7 = _mm256_setzero_ps();
            for shared in 0..N {
                let lhs_packet = _mm256_loadu_ps(lhs_values.as_ptr().add(shared * M + row));
                accumulator0 =
                    _mm256_fmadd_ps(lhs_packet, _mm256_set1_ps(*rhs0.add(shared)), accumulator0);
                accumulator1 =
                    _mm256_fmadd_ps(lhs_packet, _mm256_set1_ps(*rhs1.add(shared)), accumulator1);
                accumulator2 =
                    _mm256_fmadd_ps(lhs_packet, _mm256_set1_ps(*rhs2.add(shared)), accumulator2);
                accumulator3 =
                    _mm256_fmadd_ps(lhs_packet, _mm256_set1_ps(*rhs3.add(shared)), accumulator3);
                accumulator4 =
                    _mm256_fmadd_ps(lhs_packet, _mm256_set1_ps(*rhs4.add(shared)), accumulator4);
                accumulator5 =
                    _mm256_fmadd_ps(lhs_packet, _mm256_set1_ps(*rhs5.add(shared)), accumulator5);
                accumulator6 =
                    _mm256_fmadd_ps(lhs_packet, _mm256_set1_ps(*rhs6.add(shared)), accumulator6);
                accumulator7 =
                    _mm256_fmadd_ps(lhs_packet, _mm256_set1_ps(*rhs7.add(shared)), accumulator7);
            }
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add(column * M + row),
                accumulator0,
            );
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add((column + 1) * M + row),
                accumulator1,
            );
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add((column + 2) * M + row),
                accumulator2,
            );
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add((column + 3) * M + row),
                accumulator3,
            );
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add((column + 4) * M + row),
                accumulator4,
            );
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add((column + 5) * M + row),
                accumulator5,
            );
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add((column + 6) * M + row),
                accumulator6,
            );
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add((column + 7) * M + row),
                accumulator7,
            );
            row += 8;
        }
        while row < M {
            for offset in 0..8 {
                let current_column = column + offset;
                let rhs_column = rhs_values.as_ptr().add(current_column * N);
                let mut accumulator = 0.0_f32;
                for shared in 0..N {
                    accumulator += lhs_values[shared * M + row] * *rhs_column.add(shared);
                }
                output_values[current_column * M + row] = accumulator;
            }
            row += 1;
        }
        column += 8;
    }
    while column + 4 <= P {
        let mut row = 0;
        while row + 8 <= M {
            let mut accumulator0 = _mm256_setzero_ps();
            let mut accumulator1 = _mm256_setzero_ps();
            let mut accumulator2 = _mm256_setzero_ps();
            let mut accumulator3 = _mm256_setzero_ps();
            for shared in 0..N {
                let lhs_packet = _mm256_loadu_ps(lhs_values.as_ptr().add(shared * M + row));
                accumulator0 = _mm256_fmadd_ps(
                    lhs_packet,
                    _mm256_set1_ps(*rhs_values.as_ptr().add(column * N + shared)),
                    accumulator0,
                );
                accumulator1 = _mm256_fmadd_ps(
                    lhs_packet,
                    _mm256_set1_ps(*rhs_values.as_ptr().add((column + 1) * N + shared)),
                    accumulator1,
                );
                accumulator2 = _mm256_fmadd_ps(
                    lhs_packet,
                    _mm256_set1_ps(*rhs_values.as_ptr().add((column + 2) * N + shared)),
                    accumulator2,
                );
                accumulator3 = _mm256_fmadd_ps(
                    lhs_packet,
                    _mm256_set1_ps(*rhs_values.as_ptr().add((column + 3) * N + shared)),
                    accumulator3,
                );
            }
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add(column * M + row),
                accumulator0,
            );
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add((column + 1) * M + row),
                accumulator1,
            );
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add((column + 2) * M + row),
                accumulator2,
            );
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add((column + 3) * M + row),
                accumulator3,
            );
            row += 8;
        }
        while row < M {
            let remaining = M - row;
            let mask = _mm256_set_epi32(
                if remaining > 7 { -1 } else { 0 },
                if remaining > 6 { -1 } else { 0 },
                if remaining > 5 { -1 } else { 0 },
                if remaining > 4 { -1 } else { 0 },
                if remaining > 3 { -1 } else { 0 },
                if remaining > 2 { -1 } else { 0 },
                if remaining > 1 { -1 } else { 0 },
                if remaining > 0 { -1 } else { 0 },
            );
            let mut accumulator0 = _mm256_setzero_ps();
            let mut accumulator1 = _mm256_setzero_ps();
            let mut accumulator2 = _mm256_setzero_ps();
            let mut accumulator3 = _mm256_setzero_ps();
            for shared in 0..N {
                let lhs_packet =
                    _mm256_maskload_ps(lhs_values.as_ptr().add(shared * M + row), mask);
                accumulator0 = _mm256_fmadd_ps(
                    lhs_packet,
                    _mm256_set1_ps(*rhs_values.as_ptr().add(column * N + shared)),
                    accumulator0,
                );
                accumulator1 = _mm256_fmadd_ps(
                    lhs_packet,
                    _mm256_set1_ps(*rhs_values.as_ptr().add((column + 1) * N + shared)),
                    accumulator1,
                );
                accumulator2 = _mm256_fmadd_ps(
                    lhs_packet,
                    _mm256_set1_ps(*rhs_values.as_ptr().add((column + 2) * N + shared)),
                    accumulator2,
                );
                accumulator3 = _mm256_fmadd_ps(
                    lhs_packet,
                    _mm256_set1_ps(*rhs_values.as_ptr().add((column + 3) * N + shared)),
                    accumulator3,
                );
            }
            _mm256_maskstore_ps(
                output_values.as_mut_ptr().add(column * M + row),
                mask,
                accumulator0,
            );
            _mm256_maskstore_ps(
                output_values.as_mut_ptr().add((column + 1) * M + row),
                mask,
                accumulator1,
            );
            _mm256_maskstore_ps(
                output_values.as_mut_ptr().add((column + 2) * M + row),
                mask,
                accumulator2,
            );
            _mm256_maskstore_ps(
                output_values.as_mut_ptr().add((column + 3) * M + row),
                mask,
                accumulator3,
            );
            row = M;
        }
        column += 4;
    }
    while column < P {
        let mut row = 0;
        while row + 8 <= M {
            let mut accumulator = _mm256_setzero_ps();
            for shared in 0..N {
                accumulator = _mm256_fmadd_ps(
                    _mm256_loadu_ps(lhs_values.as_ptr().add(shared * M + row)),
                    _mm256_set1_ps(*rhs_values.as_ptr().add(column * N + shared)),
                    accumulator,
                );
            }
            _mm256_storeu_ps(
                output_values.as_mut_ptr().add(column * M + row),
                accumulator,
            );
            row += 8;
        }
        while row < M {
            let remaining = M - row;
            let mask = _mm256_set_epi32(
                if remaining > 7 { -1 } else { 0 },
                if remaining > 6 { -1 } else { 0 },
                if remaining > 5 { -1 } else { 0 },
                if remaining > 4 { -1 } else { 0 },
                if remaining > 3 { -1 } else { 0 },
                if remaining > 2 { -1 } else { 0 },
                if remaining > 1 { -1 } else { 0 },
                if remaining > 0 { -1 } else { 0 },
            );
            let mut accumulator = _mm256_setzero_ps();
            for shared in 0..N {
                accumulator = _mm256_fmadd_ps(
                    _mm256_maskload_ps(lhs_values.as_ptr().add(shared * M + row), mask),
                    _mm256_set1_ps(*rhs_values.as_ptr().add(column * N + shared)),
                    accumulator,
                );
            }
            _mm256_maskstore_ps(
                output_values.as_mut_ptr().add(column * M + row),
                mask,
                accumulator,
            );
            row = M;
        }
        column += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_f64<const M: usize, const N: usize, const P: usize>(
    lhs: &Matrix<M, N, f64>,
    rhs: &Matrix<N, P, f64>,
    output: &mut Matrix<M, P, f64>,
) {
    use core::arch::x86_64::{
        _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_maskload_pd, _mm256_maskstore_pd, _mm256_set1_pd,
        _mm256_set_epi64x, _mm256_setzero_pd, _mm256_storeu_pd,
    };
    let lhs_values = lhs.as_slice();
    let rhs_values = rhs.as_slice();
    let output_values = output.as_mut_slice();
    let mut column = 0;
    while P >= 8 && P % 8 <= 2 && column + 8 <= P {
        let rhs0 = rhs_values.as_ptr().add(column * N);
        let rhs1 = rhs_values.as_ptr().add((column + 1) * N);
        let rhs2 = rhs_values.as_ptr().add((column + 2) * N);
        let rhs3 = rhs_values.as_ptr().add((column + 3) * N);
        let rhs4 = rhs_values.as_ptr().add((column + 4) * N);
        let rhs5 = rhs_values.as_ptr().add((column + 5) * N);
        let rhs6 = rhs_values.as_ptr().add((column + 6) * N);
        let rhs7 = rhs_values.as_ptr().add((column + 7) * N);
        let mut row = 0;
        while row + 4 <= M {
            let mut accumulator0 = _mm256_setzero_pd();
            let mut accumulator1 = _mm256_setzero_pd();
            let mut accumulator2 = _mm256_setzero_pd();
            let mut accumulator3 = _mm256_setzero_pd();
            let mut accumulator4 = _mm256_setzero_pd();
            let mut accumulator5 = _mm256_setzero_pd();
            let mut accumulator6 = _mm256_setzero_pd();
            let mut accumulator7 = _mm256_setzero_pd();
            for shared in 0..N {
                let lhs_packet = _mm256_loadu_pd(lhs_values.as_ptr().add(shared * M + row));
                accumulator0 =
                    _mm256_fmadd_pd(lhs_packet, _mm256_set1_pd(*rhs0.add(shared)), accumulator0);
                accumulator1 =
                    _mm256_fmadd_pd(lhs_packet, _mm256_set1_pd(*rhs1.add(shared)), accumulator1);
                accumulator2 =
                    _mm256_fmadd_pd(lhs_packet, _mm256_set1_pd(*rhs2.add(shared)), accumulator2);
                accumulator3 =
                    _mm256_fmadd_pd(lhs_packet, _mm256_set1_pd(*rhs3.add(shared)), accumulator3);
                accumulator4 =
                    _mm256_fmadd_pd(lhs_packet, _mm256_set1_pd(*rhs4.add(shared)), accumulator4);
                accumulator5 =
                    _mm256_fmadd_pd(lhs_packet, _mm256_set1_pd(*rhs5.add(shared)), accumulator5);
                accumulator6 =
                    _mm256_fmadd_pd(lhs_packet, _mm256_set1_pd(*rhs6.add(shared)), accumulator6);
                accumulator7 =
                    _mm256_fmadd_pd(lhs_packet, _mm256_set1_pd(*rhs7.add(shared)), accumulator7);
            }
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add(column * M + row),
                accumulator0,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 1) * M + row),
                accumulator1,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 2) * M + row),
                accumulator2,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 3) * M + row),
                accumulator3,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 4) * M + row),
                accumulator4,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 5) * M + row),
                accumulator5,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 6) * M + row),
                accumulator6,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 7) * M + row),
                accumulator7,
            );
            row += 4;
        }
        while row < M {
            for offset in 0..8 {
                let current_column = column + offset;
                let rhs_column = rhs_values.as_ptr().add(current_column * N);
                let mut accumulator = 0.0_f64;
                for shared in 0..N {
                    accumulator += lhs_values[shared * M + row] * *rhs_column.add(shared);
                }
                output_values[current_column * M + row] = accumulator;
            }
            row += 1;
        }
        column += 8;
    }
    while column + 4 <= P {
        let mut row = 0;
        while row + 8 <= M {
            let mut accumulator00 = _mm256_setzero_pd();
            let mut accumulator01 = _mm256_setzero_pd();
            let mut accumulator02 = _mm256_setzero_pd();
            let mut accumulator03 = _mm256_setzero_pd();
            let mut accumulator10 = _mm256_setzero_pd();
            let mut accumulator11 = _mm256_setzero_pd();
            let mut accumulator12 = _mm256_setzero_pd();
            let mut accumulator13 = _mm256_setzero_pd();
            for shared in 0..N {
                let lhs_packet0 = _mm256_loadu_pd(lhs_values.as_ptr().add(shared * M + row));
                let lhs_packet1 = _mm256_loadu_pd(lhs_values.as_ptr().add(shared * M + row + 4));
                let rhs0 = _mm256_set1_pd(*rhs_values.as_ptr().add(column * N + shared));
                let rhs1 = _mm256_set1_pd(*rhs_values.as_ptr().add((column + 1) * N + shared));
                let rhs2 = _mm256_set1_pd(*rhs_values.as_ptr().add((column + 2) * N + shared));
                let rhs3 = _mm256_set1_pd(*rhs_values.as_ptr().add((column + 3) * N + shared));
                accumulator00 = _mm256_fmadd_pd(lhs_packet0, rhs0, accumulator00);
                accumulator01 = _mm256_fmadd_pd(lhs_packet0, rhs1, accumulator01);
                accumulator02 = _mm256_fmadd_pd(lhs_packet0, rhs2, accumulator02);
                accumulator03 = _mm256_fmadd_pd(lhs_packet0, rhs3, accumulator03);
                accumulator10 = _mm256_fmadd_pd(lhs_packet1, rhs0, accumulator10);
                accumulator11 = _mm256_fmadd_pd(lhs_packet1, rhs1, accumulator11);
                accumulator12 = _mm256_fmadd_pd(lhs_packet1, rhs2, accumulator12);
                accumulator13 = _mm256_fmadd_pd(lhs_packet1, rhs3, accumulator13);
            }
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add(column * M + row),
                accumulator00,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 1) * M + row),
                accumulator01,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 2) * M + row),
                accumulator02,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 3) * M + row),
                accumulator03,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add(column * M + row + 4),
                accumulator10,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 1) * M + row + 4),
                accumulator11,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 2) * M + row + 4),
                accumulator12,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 3) * M + row + 4),
                accumulator13,
            );
            row += 8;
        }
        while row + 4 <= M {
            let mut accumulator0 = _mm256_setzero_pd();
            let mut accumulator1 = _mm256_setzero_pd();
            let mut accumulator2 = _mm256_setzero_pd();
            let mut accumulator3 = _mm256_setzero_pd();
            for shared in 0..N {
                let lhs_packet = _mm256_loadu_pd(lhs_values.as_ptr().add(shared * M + row));
                accumulator0 = _mm256_fmadd_pd(
                    lhs_packet,
                    _mm256_set1_pd(*rhs_values.as_ptr().add(column * N + shared)),
                    accumulator0,
                );
                accumulator1 = _mm256_fmadd_pd(
                    lhs_packet,
                    _mm256_set1_pd(*rhs_values.as_ptr().add((column + 1) * N + shared)),
                    accumulator1,
                );
                accumulator2 = _mm256_fmadd_pd(
                    lhs_packet,
                    _mm256_set1_pd(*rhs_values.as_ptr().add((column + 2) * N + shared)),
                    accumulator2,
                );
                accumulator3 = _mm256_fmadd_pd(
                    lhs_packet,
                    _mm256_set1_pd(*rhs_values.as_ptr().add((column + 3) * N + shared)),
                    accumulator3,
                );
            }
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add(column * M + row),
                accumulator0,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 1) * M + row),
                accumulator1,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 2) * M + row),
                accumulator2,
            );
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add((column + 3) * M + row),
                accumulator3,
            );
            row += 4;
        }
        while row < M {
            let remaining = M - row;
            let mask = _mm256_set_epi64x(
                if remaining > 3 { -1 } else { 0 },
                if remaining > 2 { -1 } else { 0 },
                if remaining > 1 { -1 } else { 0 },
                if remaining > 0 { -1 } else { 0 },
            );
            let mut accumulator0 = _mm256_setzero_pd();
            let mut accumulator1 = _mm256_setzero_pd();
            let mut accumulator2 = _mm256_setzero_pd();
            let mut accumulator3 = _mm256_setzero_pd();
            for shared in 0..N {
                let lhs_packet =
                    _mm256_maskload_pd(lhs_values.as_ptr().add(shared * M + row), mask);
                accumulator0 = _mm256_fmadd_pd(
                    lhs_packet,
                    _mm256_set1_pd(*rhs_values.as_ptr().add(column * N + shared)),
                    accumulator0,
                );
                accumulator1 = _mm256_fmadd_pd(
                    lhs_packet,
                    _mm256_set1_pd(*rhs_values.as_ptr().add((column + 1) * N + shared)),
                    accumulator1,
                );
                accumulator2 = _mm256_fmadd_pd(
                    lhs_packet,
                    _mm256_set1_pd(*rhs_values.as_ptr().add((column + 2) * N + shared)),
                    accumulator2,
                );
                accumulator3 = _mm256_fmadd_pd(
                    lhs_packet,
                    _mm256_set1_pd(*rhs_values.as_ptr().add((column + 3) * N + shared)),
                    accumulator3,
                );
            }
            _mm256_maskstore_pd(
                output_values.as_mut_ptr().add(column * M + row),
                mask,
                accumulator0,
            );
            _mm256_maskstore_pd(
                output_values.as_mut_ptr().add((column + 1) * M + row),
                mask,
                accumulator1,
            );
            _mm256_maskstore_pd(
                output_values.as_mut_ptr().add((column + 2) * M + row),
                mask,
                accumulator2,
            );
            _mm256_maskstore_pd(
                output_values.as_mut_ptr().add((column + 3) * M + row),
                mask,
                accumulator3,
            );
            row = M;
        }
        column += 4;
    }
    while column < P {
        let mut row = 0;
        while row + 4 <= M {
            let mut accumulator = _mm256_setzero_pd();
            for shared in 0..N {
                accumulator = _mm256_fmadd_pd(
                    _mm256_loadu_pd(lhs_values.as_ptr().add(shared * M + row)),
                    _mm256_set1_pd(*rhs_values.as_ptr().add(column * N + shared)),
                    accumulator,
                );
            }
            _mm256_storeu_pd(
                output_values.as_mut_ptr().add(column * M + row),
                accumulator,
            );
            row += 4;
        }
        while row < M {
            let remaining = M - row;
            let mask = _mm256_set_epi64x(
                if remaining > 3 { -1 } else { 0 },
                if remaining > 2 { -1 } else { 0 },
                if remaining > 1 { -1 } else { 0 },
                if remaining > 0 { -1 } else { 0 },
            );
            let mut accumulator = _mm256_setzero_pd();
            for shared in 0..N {
                accumulator = _mm256_fmadd_pd(
                    _mm256_maskload_pd(lhs_values.as_ptr().add(shared * M + row), mask),
                    _mm256_set1_pd(*rhs_values.as_ptr().add(column * N + shared)),
                    accumulator,
                );
            }
            _mm256_maskstore_pd(
                output_values.as_mut_ptr().add(column * M + row),
                mask,
                accumulator,
            );
            row = M;
        }
        column += 1;
    }
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
        unsafe { matmul_f32(lhs, rhs, output) }
    }

    #[inline]
    fn dot(lhs: &[f32], rhs: &[f32], initial: f32) -> f32 {
        unsafe { super::avx2::dot_slices_f32(lhs, rhs, initial) }
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
    fn rank_update_two_sub(
        target: &mut [f32],
        source_first: &[f32],
        scale_first: f32,
        source_second: &[f32],
        scale_second: f32,
    ) {
        unsafe {
            super::avx2::rank_update_two_sub_f32(
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
        unsafe { matmul_f64(lhs, rhs, output) }
    }

    #[inline]
    fn dot(lhs: &[f64], rhs: &[f64], initial: f64) -> f64 {
        unsafe { super::avx2::dot_slices_f64(lhs, rhs, initial) }
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
    fn rank_update_two_sub(
        target: &mut [f64],
        source_first: &[f64],
        scale_first: f64,
        source_second: &[f64],
        scale_second: f64,
    ) {
        unsafe {
            super::avx2::rank_update_two_sub_f64(
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
        unsafe { super::avx2::scale_divide_f64(target, divisor) }
    }
}
