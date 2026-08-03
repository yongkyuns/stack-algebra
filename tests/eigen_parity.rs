#![cfg(feature = "eigen-compare")]

use stack_algebra::{Matrix, Vector};

unsafe extern "C" {
    fn sa_eigen_add_f32(
        lhs: *const f32,
        rhs: *const f32,
        rows: usize,
        columns: usize,
        output: *mut f32,
    );
    fn sa_eigen_matmul_f32(
        lhs: *const f32,
        rhs: *const f32,
        rows: usize,
        shared: usize,
        columns: usize,
        output: *mut f32,
    );
    fn sa_eigen_determinant_f32(input: *const f32, dimension: usize) -> f32;
    fn sa_eigen_inverse_f32(input: *const f32, dimension: usize, output: *mut f32);
    fn sa_eigen_solve_f32(
        input: *const f32,
        rhs: *const f32,
        dimension: usize,
        columns: usize,
        output: *mut f32,
    );
    fn sa_eigen_llt_solve_f32(
        input: *const f32,
        rhs: *const f32,
        dimension: usize,
        columns: usize,
        output: *mut f32,
    ) -> i32;
    fn sa_eigen_ldlt_solve_f32(
        input: *const f32,
        rhs: *const f32,
        dimension: usize,
        columns: usize,
        output: *mut f32,
    ) -> i32;
    fn sa_eigen_add_f64(
        lhs: *const f64,
        rhs: *const f64,
        rows: usize,
        columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_transpose_f32(input: *const f32, rows: usize, columns: usize, output: *mut f32);
    fn sa_eigen_norm_f32(input: *const f32, rows: usize, columns: usize) -> f32;
    fn sa_eigen_squared_norm_f32(input: *const f32, rows: usize, columns: usize) -> f32;
    fn sa_eigen_dot_f32(lhs: *const f32, rhs: *const f32, size: usize) -> f32;
    fn sa_eigen_normalize_f32(input: *const f32, rows: usize, columns: usize, output: *mut f32);
    fn sa_eigen_transpose_f64(input: *const f64, rows: usize, columns: usize, output: *mut f64);
    fn sa_eigen_norm_f64(input: *const f64, rows: usize, columns: usize) -> f64;
    fn sa_eigen_squared_norm_f64(input: *const f64, rows: usize, columns: usize) -> f64;
    fn sa_eigen_dot_f64(lhs: *const f64, rhs: *const f64, size: usize) -> f64;
    fn sa_eigen_normalize_f64(input: *const f64, rows: usize, columns: usize, output: *mut f64);
    fn sa_eigen_matmul_f64(
        lhs: *const f64,
        rhs: *const f64,
        rows: usize,
        shared: usize,
        columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_determinant_f64(input: *const f64, dimension: usize) -> f64;
    fn sa_eigen_inverse_f64(input: *const f64, dimension: usize, output: *mut f64);
    fn sa_eigen_solve_f64(
        input: *const f64,
        rhs: *const f64,
        dimension: usize,
        columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_llt_solve_f64(
        input: *const f64,
        rhs: *const f64,
        dimension: usize,
        columns: usize,
        output: *mut f64,
    ) -> i32;
    fn sa_eigen_ldlt_solve_f64(
        input: *const f64,
        rhs: *const f64,
        dimension: usize,
        columns: usize,
        output: *mut f64,
    ) -> i32;
}

fn matrix<const R: usize, const C: usize>() -> Matrix<R, C, f64> {
    Matrix::from_fn(|row, column| {
        let value = (row * C + column + 1) as f64;
        value / 7.0 - 1.0
    })
}

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

fn assert_close_f64(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite() == expected.is_finite());
        let error = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs());
        assert!(error <= 1e-12 + 1e-12 * scale, "{actual} != {expected}");
    }
}

fn assert_close_f32(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite() == expected.is_finite());
        let error = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs());
        assert!(error <= 1e-5 + 1e-5 * scale, "{actual} != {expected}");
    }
}

fn assert_same_fp_class_f64(actual: f64, expected: f64) {
    assert_eq!(actual.is_nan(), expected.is_nan(), "{actual} != {expected}");
    assert_eq!(
        actual.is_infinite(),
        expected.is_infinite(),
        "{actual} != {expected}"
    );
    if actual.is_infinite() && expected.is_infinite() {
        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
    }
}

fn assert_same_fp_class_f32(actual: f32, expected: f32) {
    assert_eq!(actual.is_nan(), expected.is_nan(), "{actual} != {expected}");
    assert_eq!(
        actual.is_infinite(),
        expected.is_infinite(),
        "{actual} != {expected}"
    );
    if actual.is_infinite() && expected.is_infinite() {
        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
    }
}

fn compare_norm_and_normalize_f64<const M: usize, const N: usize>(seed: u64) {
    let input = generated_f64::<M, N>(seed);
    let mut eigen_normalized = Matrix::<M, N, f64>::zeros();
    let eigen_norm = unsafe { sa_eigen_norm_f64(input.as_slice().as_ptr(), M, N) };
    unsafe {
        sa_eigen_normalize_f64(
            input.as_slice().as_ptr(),
            M,
            N,
            eigen_normalized.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f64(&[input.norm()], &[eigen_norm]);
    assert_close_f64(input.normalize().as_slice(), eigen_normalized.as_slice());
}

fn compare_norm_and_normalize_f32<const M: usize, const N: usize>(seed: u64) {
    let input = generated_f32::<M, N>(seed);
    let mut eigen_normalized = Matrix::<M, N, f32>::zeros();
    let eigen_norm = unsafe { sa_eigen_norm_f32(input.as_slice().as_ptr(), M, N) };
    unsafe {
        sa_eigen_normalize_f32(
            input.as_slice().as_ptr(),
            M,
            N,
            eigen_normalized.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f32(&[input.norm()], &[eigen_norm]);
    assert_close_f32(input.normalize().as_slice(), eigen_normalized.as_slice());
}

fn compare_dot_f64<const N: usize>(seed: u64) {
    let lhs = generated_f64::<N, 1>(seed);
    let rhs = generated_f64::<N, 1>(seed.wrapping_add(17));
    let eigen_dot =
        unsafe { sa_eigen_dot_f64(lhs.as_slice().as_ptr(), rhs.as_slice().as_ptr(), N) };
    assert_close_f64(&[lhs.dot(&rhs)], &[eigen_dot]);
}

fn compare_dot_f32<const N: usize>(seed: u64) {
    let lhs = generated_f32::<N, 1>(seed);
    let rhs = generated_f32::<N, 1>(seed.wrapping_add(17));
    let eigen_dot =
        unsafe { sa_eigen_dot_f32(lhs.as_slice().as_ptr(), rhs.as_slice().as_ptr(), N) };
    assert_close_f32(&[lhs.dot(&rhs)], &[eigen_dot]);
}

#[test]
fn elementary_operations_match_eigen_bits() {
    let lhs = Matrix::<2, 3, f64>::from_rows([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]);
    let rhs = Matrix::<2, 3, f64>::from_rows([[0.5, 2.0, -3.0], [7.0, -1.0, 6.0]]);
    let mut eigen = Matrix::<2, 3, f64>::zeros();
    unsafe {
        sa_eigen_add_f64(
            lhs.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            2,
            3,
            eigen.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_eq!((&lhs + &rhs).as_slice(), eigen.as_slice());

    let mut eigen_transpose = Matrix::<3, 2, f64>::zeros();
    unsafe {
        sa_eigen_transpose_f64(
            lhs.as_slice().as_ptr(),
            2,
            3,
            eigen_transpose.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_eq!(lhs.transpose().as_slice(), eigen_transpose.as_slice());

    let lhs_f32 = lhs.cast::<f32>();
    let mut eigen_transpose_f32 = Matrix::<3, 2, f32>::zeros();
    unsafe {
        sa_eigen_transpose_f32(
            lhs_f32.as_slice().as_ptr(),
            2,
            3,
            eigen_transpose_f32.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_eq!(
        lhs_f32.transpose().as_slice(),
        eigen_transpose_f32.as_slice()
    );
}

#[test]
fn reductions_and_normalization_match_eigen() {
    let input_f64 = Matrix::<3, 2, f64>::from_rows([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]]);
    let mut eigen_normalized_f64 = Matrix::<3, 2, f64>::zeros();
    let eigen_norm_f64 = unsafe { sa_eigen_norm_f64(input_f64.as_slice().as_ptr(), 3, 2) };
    unsafe {
        sa_eigen_normalize_f64(
            input_f64.as_slice().as_ptr(),
            3,
            2,
            eigen_normalized_f64.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f64(&[input_f64.norm()], &[eigen_norm_f64]);
    let eigen_squared_norm_f64 =
        unsafe { sa_eigen_squared_norm_f64(input_f64.as_slice().as_ptr(), 3, 2) };
    assert_close_f64(&[input_f64.squared_norm()], &[eigen_squared_norm_f64]);
    assert_close_f64(
        input_f64.normalize().as_slice(),
        eigen_normalized_f64.as_slice(),
    );

    let input_f32 = input_f64.cast::<f32>();
    let mut eigen_normalized_f32 = Matrix::<3, 2, f32>::zeros();
    let eigen_norm_f32 = unsafe { sa_eigen_norm_f32(input_f32.as_slice().as_ptr(), 3, 2) };
    unsafe {
        sa_eigen_normalize_f32(
            input_f32.as_slice().as_ptr(),
            3,
            2,
            eigen_normalized_f32.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f32(&[input_f32.norm()], &[eigen_norm_f32]);
    let eigen_squared_norm_f32 =
        unsafe { sa_eigen_squared_norm_f32(input_f32.as_slice().as_ptr(), 3, 2) };
    assert_close_f32(&[input_f32.squared_norm()], &[eigen_squared_norm_f32]);
    assert_close_f32(
        input_f32.normalize().as_slice(),
        eigen_normalized_f32.as_slice(),
    );

    let lhs_f64 = Vector::<4, f64>::from_columns([[1.0, -2.0, 3.0, 4.0]]);
    let rhs_f64 = Vector::<4, f64>::from_columns([[0.5, 2.0, -3.0, 6.0]]);
    let eigen_dot_f64 =
        unsafe { sa_eigen_dot_f64(lhs_f64.as_slice().as_ptr(), rhs_f64.as_slice().as_ptr(), 4) };
    assert_close_f64(&[lhs_f64.dot(&rhs_f64)], &[eigen_dot_f64]);

    let lhs_f32 = lhs_f64.cast::<f32>();
    let rhs_f32 = rhs_f64.cast::<f32>();
    let eigen_dot_f32 =
        unsafe { sa_eigen_dot_f32(lhs_f32.as_slice().as_ptr(), rhs_f32.as_slice().as_ptr(), 4) };
    assert_close_f32(&[lhs_f32.dot(&rhs_f32)], &[eigen_dot_f32]);
}

#[test]
fn cholesky_solves_match_eigen() {
    let matrix_f64 =
        Matrix::<3, 3, f64>::from_rows([[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]);
    let rhs_f64 = Vector::<3, f64>::from_columns([[1.0, 2.0, 3.0]]);
    let mut eigen_solution_f64 = Vector::<3, f64>::zeros();
    let eigen_status_f64 = unsafe {
        sa_eigen_llt_solve_f64(
            matrix_f64.as_slice().as_ptr(),
            rhs_f64.as_slice().as_ptr(),
            3,
            1,
            eigen_solution_f64.as_mut_slice().as_mut_ptr(),
        )
    };
    assert_eq!(eigen_status_f64, 1);
    let solution_f64 = matrix_f64
        .cholesky()
        .expect("matrix is positive-definite")
        .solve(&rhs_f64);
    assert_close_f64(solution_f64.as_slice(), eigen_solution_f64.as_slice());

    let matrix_f32 = matrix_f64.cast::<f32>();
    let rhs_f32 = rhs_f64.cast::<f32>();
    let mut eigen_solution_f32 = Vector::<3, f32>::zeros();
    let eigen_status_f32 = unsafe {
        sa_eigen_llt_solve_f32(
            matrix_f32.as_slice().as_ptr(),
            rhs_f32.as_slice().as_ptr(),
            3,
            1,
            eigen_solution_f32.as_mut_slice().as_mut_ptr(),
        )
    };
    assert_eq!(eigen_status_f32, 1);
    let solution_f32 = matrix_f32
        .cholesky()
        .expect("matrix is positive-definite")
        .solve(&rhs_f32);
    assert_close_f32(solution_f32.as_slice(), eigen_solution_f32.as_slice());
}

#[test]
fn ldlt_solves_match_eigen() {
    let matrix_f64 =
        Matrix::<3, 3, f64>::from_rows([[0.0, 2.0, 1.0], [2.0, 3.0, 4.0], [1.0, 4.0, 5.0]]);
    let rhs_f64 = Vector::<3, f64>::from_columns([[1.0, 2.0, 3.0]]);
    let mut eigen_solution_f64 = Vector::<3, f64>::zeros();
    let eigen_status_f64 = unsafe {
        sa_eigen_ldlt_solve_f64(
            matrix_f64.as_slice().as_ptr(),
            rhs_f64.as_slice().as_ptr(),
            3,
            1,
            eigen_solution_f64.as_mut_slice().as_mut_ptr(),
        )
    };
    assert_eq!(eigen_status_f64, 1);
    let solution_f64 = matrix_f64
        .ldlt()
        .expect("matrix is nonsingular")
        .solve(&rhs_f64);
    assert_close_f64(solution_f64.as_slice(), eigen_solution_f64.as_slice());

    let matrix_f32 = matrix_f64.cast::<f32>();
    let rhs_f32 = rhs_f64.cast::<f32>();
    let mut eigen_solution_f32 = Vector::<3, f32>::zeros();
    let eigen_status_f32 = unsafe {
        sa_eigen_ldlt_solve_f32(
            matrix_f32.as_slice().as_ptr(),
            rhs_f32.as_slice().as_ptr(),
            3,
            1,
            eigen_solution_f32.as_mut_slice().as_mut_ptr(),
        )
    };
    assert_eq!(eigen_status_f32, 1);
    let solution_f32 = matrix_f32
        .ldlt()
        .expect("matrix is nonsingular")
        .solve(&rhs_f32);
    assert_close_f32(solution_f32.as_slice(), eigen_solution_f32.as_slice());
}

fn compare_matvec_f64<const M: usize, const N: usize>(seed: u64) {
    let lhs = generated_f64::<M, N>(seed);
    let rhs = generated_f64::<N, 1>(seed.wrapping_add(31));
    let mut eigen = Matrix::<M, 1, f64>::zeros();
    unsafe {
        sa_eigen_matmul_f64(
            lhs.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            M,
            N,
            1,
            eigen.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f64(lhs.matvec(&rhs).as_slice(), eigen.as_slice());
}

fn compare_matvec_f32<const M: usize, const N: usize>(seed: u64) {
    let lhs = generated_f32::<M, N>(seed);
    let rhs = generated_f32::<N, 1>(seed.wrapping_add(31));
    let mut eigen = Matrix::<M, 1, f32>::zeros();
    unsafe {
        sa_eigen_matmul_f32(
            lhs.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            M,
            N,
            1,
            eigen.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f32(lhs.matvec(&rhs).as_slice(), eigen.as_slice());
}

#[test]
fn matrix_vector_products_match_eigen_across_shapes() {
    compare_matvec_f64::<1, 1>(31);
    compare_matvec_f64::<2, 3>(32);
    compare_matvec_f64::<7, 4>(33);
    compare_matvec_f64::<9, 6>(34);
    compare_matvec_f64::<15, 3>(35);

    compare_matvec_f32::<1, 1>(31);
    compare_matvec_f32::<2, 3>(32);
    compare_matvec_f32::<7, 4>(33);
    compare_matvec_f32::<9, 6>(34);
    compare_matvec_f32::<15, 3>(35);
}

#[test]
fn randomized_reductions_match_eigen_across_shapes() {
    compare_norm_and_normalize_f64::<1, 1>(1);
    compare_norm_and_normalize_f64::<1, 7>(2);
    compare_norm_and_normalize_f64::<7, 1>(3);
    compare_norm_and_normalize_f64::<2, 3>(4);
    compare_norm_and_normalize_f64::<5, 7>(5);
    compare_norm_and_normalize_f64::<9, 6>(6);
    compare_norm_and_normalize_f64::<15, 15>(7);

    compare_norm_and_normalize_f32::<1, 1>(1);
    compare_norm_and_normalize_f32::<1, 7>(2);
    compare_norm_and_normalize_f32::<7, 1>(3);
    compare_norm_and_normalize_f32::<2, 3>(4);
    compare_norm_and_normalize_f32::<5, 7>(5);
    compare_norm_and_normalize_f32::<9, 6>(6);
    compare_norm_and_normalize_f32::<15, 15>(7);

    compare_dot_f64::<1>(11);
    compare_dot_f64::<2>(12);
    compare_dot_f64::<3>(13);
    compare_dot_f64::<5>(14);
    compare_dot_f64::<8>(15);
    compare_dot_f64::<17>(16);
    compare_dot_f64::<33>(17);

    compare_dot_f32::<1>(11);
    compare_dot_f32::<2>(12);
    compare_dot_f32::<3>(13);
    compare_dot_f32::<5>(14);
    compare_dot_f32::<8>(15);
    compare_dot_f32::<17>(16);
    compare_dot_f32::<33>(17);
}

fn compare_transpose_f64<const M: usize, const N: usize>(seed: u64) {
    let input = generated_f64::<M, N>(seed);
    let mut eigen = Matrix::<N, M, f64>::zeros();
    unsafe {
        sa_eigen_transpose_f64(
            input.as_slice().as_ptr(),
            M,
            N,
            eigen.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_eq!(input.transpose().as_slice(), eigen.as_slice());
}

fn compare_transpose_f32<const M: usize, const N: usize>(seed: u64) {
    let input = generated_f32::<M, N>(seed);
    let mut eigen = Matrix::<N, M, f32>::zeros();
    unsafe {
        sa_eigen_transpose_f32(
            input.as_slice().as_ptr(),
            M,
            N,
            eigen.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_eq!(input.transpose().as_slice(), eigen.as_slice());
}

#[test]
fn transpose_edge_shapes_match_eigen() {
    compare_transpose_f64::<1, 1>(21);
    compare_transpose_f64::<1, 7>(22);
    compare_transpose_f64::<7, 1>(23);
    compare_transpose_f64::<2, 3>(24);
    compare_transpose_f64::<9, 4>(25);

    compare_transpose_f32::<1, 1>(21);
    compare_transpose_f32::<1, 7>(22);
    compare_transpose_f32::<7, 1>(23);
    compare_transpose_f32::<2, 3>(24);
    compare_transpose_f32::<9, 4>(25);
}

#[test]
fn special_values_match_eigen_classification() {
    let nan_f64 = Matrix::<2, 2, f64>::from_rows([[f64::NAN, 1.0], [2.0, -3.0]]);
    let mut eigen_normalized_f64 = Matrix::<2, 2, f64>::zeros();
    let eigen_norm_f64 = unsafe { sa_eigen_norm_f64(nan_f64.as_slice().as_ptr(), 2, 2) };
    unsafe {
        sa_eigen_normalize_f64(
            nan_f64.as_slice().as_ptr(),
            2,
            2,
            eigen_normalized_f64.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_same_fp_class_f64(nan_f64.norm(), eigen_norm_f64);
    assert!(nan_f64
        .normalize()
        .as_slice()
        .iter()
        .all(|value| value.is_nan()));
    assert!(eigen_normalized_f64
        .as_slice()
        .iter()
        .all(|value| value.is_nan()));

    let inf_f32 = Matrix::<2, 2, f32>::from_rows([[f32::INFINITY, 1.0], [2.0, f32::NEG_INFINITY]]);
    let eigen_norm_f32 = unsafe { sa_eigen_norm_f32(inf_f32.as_slice().as_ptr(), 2, 2) };
    assert_same_fp_class_f32(inf_f32.norm(), eigen_norm_f32);
    assert!(inf_f32.norm().is_infinite());

    let nan_lhs_f32 = Vector::<3, f32>::from_columns([[f32::NAN, 1.0, 2.0]]);
    let nan_rhs_f32 = Vector::<3, f32>::from_columns([[1.0, 2.0, 3.0]]);
    let eigen_dot_f32 = unsafe {
        sa_eigen_dot_f32(
            nan_lhs_f32.as_slice().as_ptr(),
            nan_rhs_f32.as_slice().as_ptr(),
            3,
        )
    };
    assert!(nan_lhs_f32.dot(&nan_rhs_f32).is_nan());
    assert!(eigen_dot_f32.is_nan());
}

#[test]
fn transpose_preserves_signed_zero_bits() {
    let input = Matrix::<2, 2, f64>::from_rows([[0.0, -0.0], [-0.0, 1.0]]);
    let mut eigen = Matrix::<2, 2, f64>::zeros();
    unsafe {
        sa_eigen_transpose_f64(
            input.as_slice().as_ptr(),
            2,
            2,
            eigen.as_mut_slice().as_mut_ptr(),
        );
    }
    let actual = input.transpose();
    assert_eq!(
        actual
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        eigen
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

fn compare_square_matmul<const D: usize>() {
    let lhs = matrix::<D, D>();
    let rhs: Matrix<D, D, f64> =
        Matrix::from_fn(|row, column| (row + 2 * column + 3) as f64 / 11.0);
    let mut eigen = Matrix::<D, D, f64>::zeros();
    unsafe {
        sa_eigen_matmul_f64(
            lhs.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            D,
            D,
            D,
            eigen.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f64((&lhs * &rhs).as_slice(), eigen.as_slice());
}

fn compare_rectangular_f64<const M: usize, const N: usize, const P: usize>() {
    let lhs = generated_f64::<M, N>(11);
    let rhs = generated_f64::<N, P>(29);
    let mut eigen = Matrix::<M, P, f64>::zeros();
    unsafe {
        sa_eigen_matmul_f64(
            lhs.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            M,
            N,
            P,
            eigen.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f64((&lhs * &rhs).as_slice(), eigen.as_slice());
}

fn compare_rectangular_f32<const M: usize, const N: usize, const P: usize>() {
    let lhs = generated_f32::<M, N>(11);
    let rhs = generated_f32::<N, P>(29);
    let mut eigen = Matrix::<M, P, f32>::zeros();
    unsafe {
        sa_eigen_matmul_f32(
            lhs.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            M,
            N,
            P,
            eigen.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f32((&lhs * &rhs).as_slice(), eigen.as_slice());
}

#[test]
fn fixed_size_matrix_products_match_eigen() {
    compare_square_matmul::<2>();
    compare_square_matmul::<3>();
    compare_square_matmul::<4>();
    compare_square_matmul::<6>();
    compare_square_matmul::<9>();
    compare_square_matmul::<15>();
}

#[test]
fn rectangular_matrix_products_match_eigen_across_shapes() {
    compare_rectangular_f64::<1, 1, 1>();
    compare_rectangular_f64::<2, 3, 4>();
    compare_rectangular_f64::<3, 5, 2>();
    compare_rectangular_f64::<5, 7, 3>();
    compare_rectangular_f64::<7, 4, 9>();
    compare_rectangular_f64::<9, 6, 5>();
    compare_rectangular_f64::<2, 3, 2>();
    compare_rectangular_f64::<3, 6, 3>();
    compare_rectangular_f64::<6, 15, 6>();

    compare_rectangular_f32::<1, 1, 1>();
    compare_rectangular_f32::<2, 3, 4>();
    compare_rectangular_f32::<3, 5, 2>();
    compare_rectangular_f32::<5, 7, 3>();
    compare_rectangular_f32::<7, 4, 9>();
    compare_rectangular_f32::<9, 6, 5>();
    compare_rectangular_f32::<2, 3, 2>();
    compare_rectangular_f32::<3, 6, 3>();
    compare_rectangular_f32::<6, 15, 6>();
}

#[test]
fn f32_products_match_eigen() {
    let lhs = Matrix::<3, 2, f32>::from_rows([[1.0, -2.0], [3.0, 4.0], [5.0, -6.0]]);
    let rhs = Matrix::<2, 4, f32>::from_rows([[0.5, 2.0, -3.0, 4.0], [7.0, -1.0, 6.0, 8.0]]);
    let mut eigen = Matrix::<3, 4, f32>::zeros();
    unsafe {
        sa_eigen_matmul_f32(
            lhs.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            2,
            4,
            eigen.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f32((&lhs * &rhs).as_slice(), eigen.as_slice());

    let mut eigen_add = Matrix::<3, 2, f32>::zeros();
    unsafe {
        sa_eigen_add_f32(
            lhs.as_slice().as_ptr(),
            lhs.as_slice().as_ptr(),
            3,
            2,
            eigen_add.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_eq!((&lhs + &lhs).as_slice(), eigen_add.as_slice());
}

#[test]
fn f32_determinant_inverse_and_solve_match_eigen() {
    let matrix =
        Matrix::<3, 3, f32>::from_rows([[6.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0, 4.0, 9.0]]);
    let rhs: Vector<3, f32> = Matrix::from_columns([[1.0, 2.0, 3.0]]);

    let eigen_det = unsafe { sa_eigen_determinant_f32(matrix.as_slice().as_ptr(), 3) };
    assert_close_f32(&[matrix.determinant()], &[eigen_det]);

    let mut eigen_inverse = Matrix::<3, 3, f32>::zeros();
    unsafe {
        sa_eigen_inverse_f32(
            matrix.as_slice().as_ptr(),
            3,
            eigen_inverse.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f32(matrix.inverse().as_slice(), eigen_inverse.as_slice());

    let mut eigen_solution = Vector::<3, f32>::zeros();
    unsafe {
        sa_eigen_solve_f32(
            matrix.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            1,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f32(
        matrix.partial_piv_lu().solve(&rhs).as_slice(),
        eigen_solution.as_slice(),
    );
}

#[test]
fn determinant_inverse_and_solve_match_eigen() {
    let matrix =
        Matrix::<3, 3, f64>::from_rows([[6.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0, 4.0, 9.0]]);
    let rhs: Vector<3, f64> = Matrix::from_columns([[1.0, 2.0, 3.0]]);

    let eigen_det = unsafe { sa_eigen_determinant_f64(matrix.as_slice().as_ptr(), 3) };
    assert_close_f64(&[matrix.determinant()], &[eigen_det]);

    let mut eigen_inverse = Matrix::<3, 3, f64>::zeros();
    unsafe {
        sa_eigen_inverse_f64(
            matrix.as_slice().as_ptr(),
            3,
            eigen_inverse.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f64(matrix.inverse().as_slice(), eigen_inverse.as_slice());

    let mut eigen_solution = Vector::<3, f64>::zeros();
    unsafe {
        sa_eigen_solve_f64(
            matrix.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            1,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f64(
        matrix.partial_piv_lu().solve(&rhs).as_slice(),
        eigen_solution.as_slice(),
    );
}
