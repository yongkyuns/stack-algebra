use core::ops::{Add, Mul};

use crate::num::Zero;
use crate::Matrix;

#[doc(hidden)]
pub trait MatmulBackend<T> {
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, T>,
        rhs: &Matrix<N, P, T>,
        output: &mut Matrix<M, P, T>,
    );
}

#[doc(hidden)]
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
}

/// Associates a scalar type with its compile-time matrix multiplication kernel.
///
/// Implement this trait for custom scalar types to enable matrix products. The
/// `ScalarMatmul` kernel provides the portable fallback; specialized kernels
/// can be associated when the scalar type has a matching implementation.
pub trait MatrixScalar: Copy + Zero + Add<Output = Self> + Mul<Output = Self> {
    type Matmul: MatmulBackend<Self>;
}

macro_rules! impl_scalar_matrix_scalar {
    ($($scalar:ty),+ $(,)?) => {
        $(impl MatrixScalar for $scalar {
            type Matmul = ScalarMatmul;
        })+
    };
}

impl_scalar_matrix_scalar!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
impl_scalar_matrix_scalar!(f32, f64);

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl MatrixScalar for f32 {
    type Matmul = X86Avx2Matmul;
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl MatrixScalar for f64 {
    type Matmul = X86Avx2Matmul;
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
impl MatrixScalar for f32 {
    type Matmul = X86Sse2Matmul;
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
impl MatrixScalar for f64 {
    type Matmul = X86Sse2Matmul;
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[doc(hidden)]
pub struct X86Avx2Matmul;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl MatmulBackend<f32> for X86Avx2Matmul {
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, f32>,
        rhs: &Matrix<N, P, f32>,
        output: &mut Matrix<M, P, f32>,
    ) {
        unsafe { matmul_avx2_f32(lhs, rhs, output) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl MatmulBackend<f64> for X86Avx2Matmul {
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, f64>,
        rhs: &Matrix<N, P, f64>,
        output: &mut Matrix<M, P, f64>,
    ) {
        unsafe { matmul_avx2_f64(lhs, rhs, output) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn matmul_avx2_f32<const M: usize, const N: usize, const P: usize>(
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn matmul_avx2_f64<const M: usize, const N: usize, const P: usize>(
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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
#[doc(hidden)]
pub struct X86Sse2Matmul;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
impl MatmulBackend<f32> for X86Sse2Matmul {
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, f32>,
        rhs: &Matrix<N, P, f32>,
        output: &mut Matrix<M, P, f32>,
    ) {
        unsafe { matmul_sse2_f32(lhs, rhs, output) }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
impl MatmulBackend<f64> for X86Sse2Matmul {
    #[inline]
    fn run<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, f64>,
        rhs: &Matrix<N, P, f64>,
        output: &mut Matrix<M, P, f64>,
    ) {
        unsafe { matmul_sse2_f64(lhs, rhs, output) }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
#[target_feature(enable = "sse2")]
unsafe fn matmul_sse2_f32<const M: usize, const N: usize, const P: usize>(
    lhs: &Matrix<M, N, f32>,
    rhs: &Matrix<N, P, f32>,
    output: &mut Matrix<M, P, f32>,
) {
    use core::arch::x86_64::{
        _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    };

    for column in 0..P {
        let mut row = 0;
        while row + 4 <= M {
            let mut accumulator = _mm_setzero_ps();
            for shared in 0..N {
                let lhs_ptr = lhs.as_slice().as_ptr().add(shared * M + row);
                let lhs_packet = _mm_loadu_ps(lhs_ptr);
                let rhs_packet = _mm_set1_ps(rhs[(shared, column)]);
                accumulator = _mm_add_ps(accumulator, _mm_mul_ps(lhs_packet, rhs_packet));
            }
            let output_ptr = output.as_mut_slice().as_mut_ptr().add(column * M + row);
            _mm_storeu_ps(output_ptr, accumulator);
            row += 4;
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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
#[target_feature(enable = "sse2")]
unsafe fn matmul_sse2_f64<const M: usize, const N: usize, const P: usize>(
    lhs: &Matrix<M, N, f64>,
    rhs: &Matrix<N, P, f64>,
    output: &mut Matrix<M, P, f64>,
) {
    use core::arch::x86_64::{
        _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_setzero_pd, _mm_storeu_pd,
    };

    for column in 0..P {
        let mut row = 0;
        while row + 2 <= M {
            let mut accumulator = _mm_setzero_pd();
            for shared in 0..N {
                let lhs_ptr = lhs.as_slice().as_ptr().add(shared * M + row);
                let lhs_packet = _mm_loadu_pd(lhs_ptr);
                let rhs_packet = _mm_set1_pd(rhs[(shared, column)]);
                accumulator = _mm_add_pd(accumulator, _mm_mul_pd(lhs_packet, rhs_packet));
            }
            let output_ptr = output.as_mut_slice().as_mut_ptr().add(column * M + row);
            _mm_storeu_pd(output_ptr, accumulator);
            row += 2;
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

/// Dispatches matrix multiplication through the selected internal backend.
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

/// Portable scalar matrix multiplication used as the correctness reference.
///
/// Optimized backends should preserve this traversal order where practical so
/// that floating-point differences remain predictable.
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
