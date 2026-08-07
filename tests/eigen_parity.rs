#![cfg(feature = "eigen-compare")]

use std::ffi::c_void;

use stack_algebra::{
    AffineTransform, Isometry, Matrix, Quaternion, StaticCscCholesky, StaticCscCholeskyPattern,
    StaticCscLdlt, StaticCscMatrix, StaticCscOrdering, Vector,
};

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
    fn sa_eigen_qr_solve_f32(
        input: *const f32,
        rhs: *const f32,
        rows: usize,
        columns: usize,
        rhs_columns: usize,
        output: *mut f32,
    );
    fn sa_eigen_col_piv_qr_solve_f32(
        input: *const f32,
        rhs: *const f32,
        rows: usize,
        columns: usize,
        rhs_columns: usize,
        output: *mut f32,
    );
    fn sa_eigen_svd_singular_values_f32(
        input: *const f32,
        rows: usize,
        columns: usize,
        output: *mut f32,
    );
    fn sa_eigen_svd_solve_f32(
        input: *const f32,
        rhs: *const f32,
        rows: usize,
        columns: usize,
        rhs_columns: usize,
        output: *mut f32,
    );
    fn sa_eigen_self_adjoint_eigenvalues_f32(input: *const f32, dimension: usize, output: *mut f32);
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
    fn sa_eigen_qr_solve_f64(
        input: *const f64,
        rhs: *const f64,
        rows: usize,
        columns: usize,
        rhs_columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_col_piv_qr_solve_f64(
        input: *const f64,
        rhs: *const f64,
        rows: usize,
        columns: usize,
        rhs_columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_svd_singular_values_f64(
        input: *const f64,
        rows: usize,
        columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_svd_solve_f64(
        input: *const f64,
        rhs: *const f64,
        rows: usize,
        columns: usize,
        rhs_columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_self_adjoint_eigenvalues_f64(input: *const f64, dimension: usize, output: *mut f64);
    fn sa_eigen_lower_triangular_solve_f64(
        input: *const f64,
        rhs: *const f64,
        dimension: usize,
        columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_upper_triangular_solve_f64(
        input: *const f64,
        rhs: *const f64,
        dimension: usize,
        columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_lower_triangular_mul_f64(
        input: *const f64,
        rhs: *const f64,
        dimension: usize,
        columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_upper_triangular_mul_f64(
        input: *const f64,
        rhs: *const f64,
        dimension: usize,
        columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_quaternion_rotation_f64(
        quaternion: *const f64,
        matrix_output: *mut f64,
        vector_output: *mut f64,
        vector_input: *const f64,
    );
    fn sa_eigen_isometry_transform_f64(
        quaternion: *const f64,
        translation: *const f64,
        point: *const f64,
        matrix_output: *mut f64,
        point_output: *mut f64,
    );
    fn sa_eigen_affine_transform_f64(matrix: *const f64, point: *const f64, point_output: *mut f64);
    fn sa_eigen_self_adjoint_eigenvectors_f64(
        input: *const f64,
        dimension: usize,
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
    fn sa_eigen_sparse_llt_create_f64(
        row_indices: *const usize,
        column_starts: *const usize,
        values: *const f64,
        dimension: usize,
        nonzeros: usize,
    ) -> *mut c_void;
    fn sa_eigen_sparse_llt_analyze_f64(context: *mut c_void) -> i32;
    fn sa_eigen_sparse_llt_factorize_f64(context: *mut c_void) -> i32;
    fn sa_eigen_sparse_llt_solve_f64(
        context: *mut c_void,
        rhs: *const f64,
        columns: usize,
        output: *mut f64,
    ) -> i32;
    fn sa_eigen_sparse_llt_destroy_f64(context: *mut c_void);
    fn sa_eigen_sparse_ldlt_create_f64(
        row_indices: *const usize,
        column_starts: *const usize,
        values: *const f64,
        dimension: usize,
        nonzeros: usize,
    ) -> *mut c_void;
    fn sa_eigen_sparse_ldlt_analyze_f64(context: *mut c_void) -> i32;
    fn sa_eigen_sparse_ldlt_factorize_f64(context: *mut c_void) -> i32;
    fn sa_eigen_sparse_ldlt_solve_f64(
        context: *mut c_void,
        rhs: *const f64,
        columns: usize,
        output: *mut f64,
    ) -> i32;
    fn sa_eigen_sparse_ldlt_destroy_f64(context: *mut c_void);
    fn sa_eigen_sparse_ldlt_create_f32(
        row_indices: *const usize,
        column_starts: *const usize,
        values: *const f32,
        dimension: usize,
        nonzeros: usize,
    ) -> *mut c_void;
    fn sa_eigen_sparse_ldlt_analyze_f32(context: *mut c_void) -> i32;
    fn sa_eigen_sparse_ldlt_factorize_f32(context: *mut c_void) -> i32;
    fn sa_eigen_sparse_ldlt_solve_f32(
        context: *mut c_void,
        rhs: *const f32,
        columns: usize,
        output: *mut f32,
    ) -> i32;
    fn sa_eigen_sparse_ldlt_destroy_f32(context: *mut c_void);
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

fn assert_eigenvectors_match_f64<const D: usize>(
    actual: &Matrix<D, D, f64>,
    expected: &Matrix<D, D, f64>,
) {
    for column in 0..D {
        let mut dot = 0.0;
        for row in 0..D {
            dot += actual[(row, column)] * expected[(row, column)];
        }
        let sign = if dot.is_sign_negative() { -1.0 } else { 1.0 };
        for row in 0..D {
            let aligned = sign * expected[(row, column)];
            assert!((actual[(row, column)] - aligned).abs() <= 1e-11);
        }
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
    assert_eq!((lhs + rhs).as_slice(), eigen.as_slice());

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
    let factor_f64 = matrix_f64.cholesky().expect("matrix is positive-definite");
    let solution_f64 = factor_f64.solve(&rhs_f64);
    let mut output_f64 = Matrix::<3, 1, f64>::zeros();
    factor_f64.solve_into(&rhs_f64, &mut output_f64);
    assert_close_f64(solution_f64.as_slice(), eigen_solution_f64.as_slice());
    assert_close_f64(output_f64.as_slice(), eigen_solution_f64.as_slice());

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
    let factor_f32 = matrix_f32.cholesky().expect("matrix is positive-definite");
    let solution_f32 = factor_f32.solve(&rhs_f32);
    let mut output_f32 = Matrix::<3, 1, f32>::zeros();
    factor_f32.solve_into(&rhs_f32, &mut output_f32);
    assert_close_f32(solution_f32.as_slice(), eigen_solution_f32.as_slice());
    assert_close_f32(output_f32.as_slice(), eigen_solution_f32.as_slice());
}

#[test]
fn sparse_cholesky_solves_match_eigen() {
    let dense = Matrix::<3, 3, f64>::from_rows([[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]);
    let sparse = StaticCscMatrix::<3, 3, 5, f64>::from_pattern(
        &[4.0, 1.0, 1.0, 3.0, 2.0],
        &[0, 1, 2, 1, 2],
        &[0, 3, 4, 5],
    )
    .expect("lower CSC pattern is valid");
    let rhs = Vector::<3, f64>::from_columns([[1.0, 2.0, 3.0]]);
    let mut eigen_solution = Vector::<3, f64>::zeros();
    let eigen_status = unsafe {
        sa_eigen_llt_solve_f64(
            dense.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            1,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        )
    };
    assert_eq!(eigen_status, 1);
    let sparse_solution = StaticCscCholesky::<3, 6, f64>::decompose(&sparse)
        .expect("sparse matrix is positive-definite")
        .solve(&rhs);
    assert_close_f64(sparse_solution.as_slice(), eigen_solution.as_slice());

    let eigen_sparse_context = unsafe {
        sa_eigen_sparse_llt_create_f64(
            sparse.row_indices().as_ptr(),
            sparse.column_starts().as_ptr(),
            sparse.values().as_ptr(),
            sparse.rows(),
            sparse.nnz(),
        )
    };
    assert!(!eigen_sparse_context.is_null());
    assert_eq!(
        unsafe { sa_eigen_sparse_llt_analyze_f64(eigen_sparse_context) },
        1
    );
    assert_eq!(
        unsafe { sa_eigen_sparse_llt_factorize_f64(eigen_sparse_context) },
        1
    );
    let mut eigen_sparse_solution = Vector::<3, f64>::zeros();
    assert_eq!(
        unsafe {
            sa_eigen_sparse_llt_solve_f64(
                eigen_sparse_context,
                rhs.as_slice().as_ptr(),
                1,
                eigen_sparse_solution.as_mut_slice().as_mut_ptr(),
            )
        },
        1
    );
    assert_close_f64(eigen_sparse_solution.as_slice(), eigen_solution.as_slice());
    unsafe { sa_eigen_sparse_llt_destroy_f64(eigen_sparse_context) };
}

#[test]
fn sparse_banded_cholesky_solves_match_eigen() {
    let dense = Matrix::<4, 4, f64>::from_rows([
        [4.0, 1.0, 1.0, 0.0],
        [1.0, 4.0, 1.0, 1.0],
        [1.0, 1.0, 4.0, 1.0],
        [0.0, 1.0, 1.0, 4.0],
    ]);
    let sparse = StaticCscMatrix::<4, 4, 9, f64>::from_pattern(
        &[4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 4.0],
        &[0, 1, 2, 1, 2, 3, 2, 3, 3],
        &[0, 3, 6, 8, 9],
    )
    .expect("banded lower CSC pattern is valid");
    let rhs = Vector::<4, f64>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    let mut eigen_solution = Vector::<4, f64>::zeros();
    let eigen_status = unsafe {
        sa_eigen_llt_solve_f64(
            dense.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            4,
            1,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        )
    };
    assert_eq!(eigen_status, 1);
    let sparse_solution = stack_algebra::StaticCscCholesky::<4, 16, f64>::decompose(&sparse)
        .expect("sparse matrix is positive-definite")
        .solve(&rhs);
    assert_close_f64(sparse_solution.as_slice(), eigen_solution.as_slice());
}

#[test]
fn sparse_star_fill_cholesky_solves_match_eigen() {
    let dense = Matrix::<4, 4, f64>::from_rows([
        [4.0, 1.0, 1.0, 1.0],
        [1.0, 4.0, 0.0, 0.0],
        [1.0, 0.0, 4.0, 0.0],
        [1.0, 0.0, 0.0, 4.0],
    ]);
    let sparse = StaticCscMatrix::<4, 4, 10, f64>::from_pattern(
        &[4.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
        &[0, 1, 2, 3, 1, 2, 3],
        &[0, 4, 5, 6, 7],
    )
    .expect("star lower CSC pattern is valid");
    let rhs = Vector::<4, f64>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    let mut eigen_solution = Vector::<4, f64>::zeros();
    let eigen_status = unsafe {
        sa_eigen_llt_solve_f64(
            dense.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            4,
            1,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        )
    };
    assert_eq!(eigen_status, 1);
    let sparse_solution = stack_algebra::StaticCscCholesky::<4, 10, f64>::decompose(&sparse)
        .expect("sparse matrix is positive-definite")
        .solve(&rhs);
    assert_close_f64(sparse_solution.as_slice(), eigen_solution.as_slice());

    let ordering = StaticCscOrdering::minimum_degree(&sparse);
    let ordered_pattern =
        StaticCscCholeskyPattern::<4, 10>::analyze_with_ordering(&sparse, ordering)
            .expect("ordered star pattern is valid");
    let ordered = ordered_pattern
        .prepare_ordered(&sparse)
        .expect("ordered star matrix is valid");
    let ordered_solution = ordered_pattern
        .factor_ordered(&ordered)
        .expect("ordered star matrix is positive-definite")
        .solve(&rhs);
    assert_close_f64(ordered_solution.as_slice(), eigen_solution.as_slice());
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
    let factor_f64 = matrix_f64.ldlt().expect("matrix is nonsingular");
    let solution_f64 = factor_f64.solve(&rhs_f64);
    let mut output_f64 = Matrix::<3, 1, f64>::zeros();
    factor_f64.solve_into(&rhs_f64, &mut output_f64);
    assert_close_f64(solution_f64.as_slice(), eigen_solution_f64.as_slice());
    assert_close_f64(output_f64.as_slice(), eigen_solution_f64.as_slice());

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
    let factor_f32 = matrix_f32.ldlt().expect("matrix is nonsingular");
    let solution_f32 = factor_f32.solve(&rhs_f32);
    let mut output_f32 = Matrix::<3, 1, f32>::zeros();
    factor_f32.solve_into(&rhs_f32, &mut output_f32);
    assert_close_f32(solution_f32.as_slice(), eigen_solution_f32.as_slice());
    assert_close_f32(output_f32.as_slice(), eigen_solution_f32.as_slice());

    let sparse_dense =
        Matrix::<3, 3, f64>::from_rows([[4.0, 1.0, 2.0], [1.0, -3.0, 1.0], [2.0, 1.0, 2.0]]);
    let sparse = StaticCscMatrix::<3, 3, 6, f64>::from_pattern(
        &[4.0, 1.0, 2.0, -3.0, 1.0, 2.0],
        &[0, 1, 2, 1, 2, 2],
        &[0, 3, 5, 6],
    )
    .expect("sparse indefinite CSC pattern is valid");
    let sparse_rhs = Vector::<3, f64>::from_columns([[1.0, 2.0, 3.0]]);
    let mut sparse_eigen_solution = Vector::<3, f64>::zeros();
    let sparse_eigen_status = unsafe {
        sa_eigen_ldlt_solve_f64(
            sparse_dense.as_slice().as_ptr(),
            sparse_rhs.as_slice().as_ptr(),
            3,
            1,
            sparse_eigen_solution.as_mut_slice().as_mut_ptr(),
        )
    };
    assert_eq!(sparse_eigen_status, 1);
    let sparse_solution = StaticCscLdlt::<3, 6, f64>::decompose(&sparse)
        .expect("sparse indefinite matrix is nonsingular")
        .solve(&sparse_rhs);
    assert_close_f64(sparse_solution.as_slice(), sparse_eigen_solution.as_slice());

    let eigen_sparse_context = unsafe {
        sa_eigen_sparse_ldlt_create_f64(
            sparse.row_indices().as_ptr(),
            sparse.column_starts().as_ptr(),
            sparse.values().as_ptr(),
            sparse.rows(),
            sparse.nnz(),
        )
    };
    assert!(!eigen_sparse_context.is_null());
    assert_eq!(
        unsafe { sa_eigen_sparse_ldlt_analyze_f64(eigen_sparse_context) },
        1
    );
    assert_eq!(
        unsafe { sa_eigen_sparse_ldlt_factorize_f64(eigen_sparse_context) },
        1
    );
    let mut eigen_sparse_solution = Vector::<3, f64>::zeros();
    assert_eq!(
        unsafe {
            sa_eigen_sparse_ldlt_solve_f64(
                eigen_sparse_context,
                sparse_rhs.as_slice().as_ptr(),
                1,
                eigen_sparse_solution.as_mut_slice().as_mut_ptr(),
            )
        },
        1
    );
    assert_close_f64(
        eigen_sparse_solution.as_slice(),
        sparse_eigen_solution.as_slice(),
    );
    unsafe { sa_eigen_sparse_ldlt_destroy_f64(eigen_sparse_context) };

    let pivoted_sparse = StaticCscMatrix::<3, 3, 6, f64>::from_pattern(
        &[0.0, 2.0, 1.0, 3.0, 4.0, 5.0],
        &[0, 1, 2, 1, 2, 2],
        &[0, 3, 5, 6],
    )
    .expect("pivoted sparse CSC pattern is valid");
    let pivoted_solution =
        StaticCscLdlt::<3, 6, f64>::decompose_with_diagonal_pivoting(&pivoted_sparse, 1e-12)
            .expect("diagonal pivoting should recover the sparse factorization")
            .solve(&rhs_f64);
    assert_close_f64(pivoted_solution.as_slice(), eigen_solution_f64.as_slice());
}

#[test]
fn sparse_and_dense_fallback_multi_rhs_match_eigen() {
    let sparse_dense =
        Matrix::<3, 3, f64>::from_rows([[4.0, 1.0, 2.0], [1.0, -3.0, 1.0], [2.0, 1.0, 2.0]]);
    let sparse = StaticCscMatrix::<3, 3, 6, f64>::from_pattern(
        &[4.0, 1.0, 2.0, -3.0, 1.0, 2.0],
        &[0, 1, 2, 1, 2, 2],
        &[0, 3, 5, 6],
    )
    .expect("sparse indefinite CSC pattern is valid");
    let rhs = Matrix::<3, 2, f64>::from_rows([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    let mut eigen_solution = Matrix::<3, 2, f64>::zeros();
    assert_eq!(
        unsafe {
            sa_eigen_ldlt_solve_f64(
                sparse_dense.as_slice().as_ptr(),
                rhs.as_slice().as_ptr(),
                3,
                2,
                eigen_solution.as_mut_slice().as_mut_ptr(),
            )
        },
        1
    );
    let factor = StaticCscLdlt::<3, 6, f64>::decompose(&sparse).unwrap();
    assert_close_f64(factor.solve(&rhs).as_slice(), eigen_solution.as_slice());

    let eigen_sparse_context = unsafe {
        sa_eigen_sparse_ldlt_create_f64(
            sparse.row_indices().as_ptr(),
            sparse.column_starts().as_ptr(),
            sparse.values().as_ptr(),
            sparse.rows(),
            sparse.nnz(),
        )
    };
    assert!(!eigen_sparse_context.is_null());
    assert_eq!(
        unsafe { sa_eigen_sparse_ldlt_analyze_f64(eigen_sparse_context) },
        1
    );
    assert_eq!(
        unsafe { sa_eigen_sparse_ldlt_factorize_f64(eigen_sparse_context) },
        1
    );
    let mut eigen_sparse_solution = Matrix::<3, 2, f64>::zeros();
    assert_eq!(
        unsafe {
            sa_eigen_sparse_ldlt_solve_f64(
                eigen_sparse_context,
                rhs.as_slice().as_ptr(),
                2,
                eigen_sparse_solution.as_mut_slice().as_mut_ptr(),
            )
        },
        1
    );
    assert_close_f64(eigen_sparse_solution.as_slice(), eigen_solution.as_slice());
    unsafe { sa_eigen_sparse_ldlt_destroy_f64(eigen_sparse_context) };

    let fallback = StaticCscMatrix::<2, 2, 3, f64>::from_pattern(
        &[1.0e-6, 1.0, 1.0e-6],
        &[0, 1, 1],
        &[0, 2, 3],
    )
    .unwrap();
    let fallback_dense = Matrix::<2, 2, f64>::from_rows([[1.0e-6, 1.0], [1.0, 1.0e-6]]);
    let fallback_rhs = Matrix::<2, 2, f64>::from_rows([[3.0, 5.0], [4.0, 6.0]]);
    let mut eigen_fallback_solution = Matrix::<2, 2, f64>::zeros();
    assert_eq!(
        unsafe {
            sa_eigen_ldlt_solve_f64(
                fallback_dense.as_slice().as_ptr(),
                fallback_rhs.as_slice().as_ptr(),
                2,
                2,
                eigen_fallback_solution.as_mut_slice().as_mut_ptr(),
            )
        },
        1
    );
    let factor = fallback.try_dense_ldlt().unwrap();
    let fallback_solution = factor.solve(&fallback_rhs);
    for (actual, expected) in fallback_solution
        .as_slice()
        .iter()
        .zip(eigen_fallback_solution.as_slice())
    {
        let error = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs());
        assert!(error <= 1e-9 + 1e-9 * scale, "{actual} != {expected}");
    }

    let sparse_dense_f32 =
        Matrix::<3, 3, f32>::from_rows([[4.0, 1.0, 2.0], [1.0, -3.0, 1.0], [2.0, 1.0, 2.0]]);
    let sparse_f32 = StaticCscMatrix::<3, 3, 6, f32>::from_pattern(
        &[4.0, 1.0, 2.0, -3.0, 1.0, 2.0],
        &[0, 1, 2, 1, 2, 2],
        &[0, 3, 5, 6],
    )
    .expect("f32 sparse CSC pattern is valid");
    let rhs_f32 = Matrix::<3, 2, f32>::from_rows([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    let mut eigen_dense_f32 = Matrix::<3, 2, f32>::zeros();
    assert_eq!(
        unsafe {
            sa_eigen_ldlt_solve_f32(
                sparse_dense_f32.as_slice().as_ptr(),
                rhs_f32.as_slice().as_ptr(),
                3,
                2,
                eigen_dense_f32.as_mut_slice().as_mut_ptr(),
            )
        },
        1
    );
    let factor_f32 = StaticCscLdlt::<3, 6, f32>::decompose(&sparse_f32).unwrap();
    assert_close_f32(
        factor_f32.solve(&rhs_f32).as_slice(),
        eigen_dense_f32.as_slice(),
    );

    let eigen_sparse_context_f32 = unsafe {
        sa_eigen_sparse_ldlt_create_f32(
            sparse_f32.row_indices().as_ptr(),
            sparse_f32.column_starts().as_ptr(),
            sparse_f32.values().as_ptr(),
            sparse_f32.rows(),
            sparse_f32.nnz(),
        )
    };
    assert!(!eigen_sparse_context_f32.is_null());
    assert_eq!(
        unsafe { sa_eigen_sparse_ldlt_analyze_f32(eigen_sparse_context_f32) },
        1
    );
    assert_eq!(
        unsafe { sa_eigen_sparse_ldlt_factorize_f32(eigen_sparse_context_f32) },
        1
    );
    let mut eigen_sparse_f32 = Matrix::<3, 2, f32>::zeros();
    assert_eq!(
        unsafe {
            sa_eigen_sparse_ldlt_solve_f32(
                eigen_sparse_context_f32,
                rhs_f32.as_slice().as_ptr(),
                2,
                eigen_sparse_f32.as_mut_slice().as_mut_ptr(),
            )
        },
        1
    );
    assert_close_f32(
        factor_f32.solve(&rhs_f32).as_slice(),
        eigen_sparse_f32.as_slice(),
    );
    unsafe { sa_eigen_sparse_ldlt_destroy_f32(eigen_sparse_context_f32) };
}

fn targeted_ldlt_case<const D: usize>(case: usize) -> Matrix<D, D, f64> {
    Matrix::from_fn(|row, column| match case {
        0 => {
            let mut value = 0.0;
            for shared in 0..D {
                let left = (shared + 3 * row + 1) as f64 / 23.0;
                let right = (shared + 3 * column + 1) as f64 / 23.0;
                value += left * right;
            }
            value + if row == column { D as f64 } else { 0.0 }
        }
        1 => {
            if row == column {
                if row % 2 == 0 {
                    -(D as f64)
                } else {
                    (D + 1) as f64
                }
            } else {
                (row + column + 1) as f64 / 29.0
            }
        }
        2 => {
            if row == 0 && column == 0 {
                1.0e-6
            } else if (row == 1 && column == 0) || (row == 0 && column == 1) {
                1.0
            } else if row == 1 && column == 1 {
                2.0
            } else if row == column {
                (D + row + 1) as f64
            } else {
                (row + column + 1) as f64 / 29.0
            }
        }
        _ => unreachable!(),
    })
}

#[test]
fn targeted_ldlt_cases_match_eigen() {
    for case in 0..3 {
        let matrix_f64 = targeted_ldlt_case::<8>(case);
        let rhs_f64 = Vector::<8, f64>::from_fn(|row, _| (row + 1) as f64 / 3.0);
        let mut eigen_solution_f64 = Vector::<8, f64>::zeros();
        assert_eq!(
            unsafe {
                sa_eigen_ldlt_solve_f64(
                    matrix_f64.as_slice().as_ptr(),
                    rhs_f64.as_slice().as_ptr(),
                    8,
                    1,
                    eigen_solution_f64.as_mut_slice().as_mut_ptr(),
                )
            },
            1
        );
        let solution_f64 = matrix_f64
            .ldlt()
            .expect("targeted matrix is nonsingular")
            .solve(&rhs_f64);
        assert_close_f64(solution_f64.as_slice(), eigen_solution_f64.as_slice());

        let matrix_f32 = matrix_f64.cast::<f32>();
        let rhs_f32 = rhs_f64.cast::<f32>();
        let mut eigen_solution_f32 = Vector::<8, f32>::zeros();
        assert_eq!(
            unsafe {
                sa_eigen_ldlt_solve_f32(
                    matrix_f32.as_slice().as_ptr(),
                    rhs_f32.as_slice().as_ptr(),
                    8,
                    1,
                    eigen_solution_f32.as_mut_slice().as_mut_ptr(),
                )
            },
            1
        );
        let solution_f32 = matrix_f32
            .ldlt()
            .expect("targeted matrix is nonsingular")
            .solve(&rhs_f32);
        assert_close_f32(solution_f32.as_slice(), eigen_solution_f32.as_slice());
    }
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
    assert_close_f64((lhs * rhs).as_slice(), eigen.as_slice());
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
    assert_close_f64((lhs * rhs).as_slice(), eigen.as_slice());
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
    assert_close_f32((lhs * rhs).as_slice(), eigen.as_slice());
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
    assert_close_f32((lhs * rhs).as_slice(), eigen.as_slice());

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
    assert_eq!((lhs + lhs).as_slice(), eigen_add.as_slice());
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
fn f32_qr_least_squares_matches_eigen() {
    let design = Matrix::<4, 2, f32>::from_rows([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
    let rhs = Matrix::<4, 2, f32>::from_rows([[3.0, 1.0], [5.0, 0.0], [7.0, -1.0], [9.0, -2.0]]);
    let mut eigen_solution = Matrix::<2, 2, f32>::zeros();
    unsafe {
        sa_eigen_qr_solve_f32(
            design.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            4,
            2,
            2,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    let solution = design
        .householder_qr()
        .solve_least_squares(&rhs)
        .expect("design matrix is full rank");
    assert_close_f32(solution.as_slice(), eigen_solution.as_slice());
}

#[test]
fn f32_column_pivoted_qr_matches_eigen() {
    let design = Matrix::<3, 2, f32>::from_rows([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]);
    let rhs = Matrix::<3, 1, f32>::from_columns([[-1.0, 0.0, 1.0]]);
    let mut eigen_solution = Matrix::<2, 1, f32>::zeros();
    unsafe {
        sa_eigen_col_piv_qr_solve_f32(
            design.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            2,
            1,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    let solution = design
        .col_piv_householder_qr()
        .solve_least_squares(&rhs)
        .expect("pivoted design matrix is full rank");
    assert_close_f32(solution.as_slice(), eigen_solution.as_slice());
}

#[test]
fn f32_svd_matches_eigen() {
    let design = Matrix::<4, 2, f32>::from_rows([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [5.0, 2.0]]);
    let rhs = Matrix::<4, 1, f32>::from_rows([[1.0], [2.0], [4.0], [3.0]]);
    let mut eigen_values = Matrix::<2, 1, f32>::zeros();
    let mut eigen_solution = Matrix::<2, 1, f32>::zeros();
    unsafe {
        sa_eigen_svd_singular_values_f32(
            design.as_slice().as_ptr(),
            4,
            2,
            eigen_values.as_mut_slice().as_mut_ptr(),
        );
        sa_eigen_svd_solve_f32(
            design.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            4,
            2,
            1,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    let svd = design.svd().expect("tall matrix is supported");
    assert_close_f32(svd.singular_values().as_slice(), eigen_values.as_slice());
    assert_close_f32(svd.solve(&rhs).as_slice(), eigen_solution.as_slice());
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
    let factor = matrix.partial_piv_lu();
    assert_close_f64(factor.solve(&rhs).as_slice(), eigen_solution.as_slice());
    let mut output = Vector::<3, f64>::zeros();
    factor.solve_into(&rhs, &mut output);
    assert_close_f64(output.as_slice(), eigen_solution.as_slice());
}

#[test]
fn qr_least_squares_matches_eigen() {
    let design = Matrix::<4, 2, f64>::from_rows([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
    let rhs = Matrix::<4, 2, f64>::from_rows([[3.0, 1.0], [5.0, 0.0], [7.0, -1.0], [9.0, -2.0]]);
    let mut eigen_solution = Matrix::<2, 2, f64>::zeros();
    unsafe {
        sa_eigen_qr_solve_f64(
            design.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            4,
            2,
            2,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    let factor = design.householder_qr();
    let solution = factor
        .solve_least_squares(&rhs)
        .expect("design matrix is full rank");
    let mut output = Matrix::<2, 2, f64>::zeros();
    factor
        .solve_least_squares_into(&rhs, &mut output)
        .expect("design matrix is full rank");
    assert_close_f64(solution.as_slice(), eigen_solution.as_slice());
    assert_close_f64(output.as_slice(), eigen_solution.as_slice());
}

#[test]
fn column_pivoted_qr_matches_eigen() {
    let design = Matrix::<3, 2, f64>::from_rows([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]);
    let rhs = Matrix::<3, 1, f64>::from_columns([[-1.0, 0.0, 1.0]]);
    let mut eigen_solution = Matrix::<2, 1, f64>::zeros();
    unsafe {
        sa_eigen_col_piv_qr_solve_f64(
            design.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            2,
            1,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    let factor = design.col_piv_householder_qr();
    let solution = factor
        .solve_least_squares(&rhs)
        .expect("pivoted design matrix is full rank");
    let mut output = Matrix::<2, 1, f64>::zeros();
    factor
        .solve_least_squares_into(&rhs, &mut output)
        .expect("pivoted design matrix is full rank");
    assert_close_f64(solution.as_slice(), eigen_solution.as_slice());
    assert_close_f64(output.as_slice(), eigen_solution.as_slice());
}

#[test]
fn svd_matches_eigen() {
    let design = Matrix::<4, 2, f64>::from_rows([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [5.0, 2.0]]);
    let rhs = Matrix::<4, 1, f64>::from_rows([[1.0], [2.0], [4.0], [3.0]]);
    let mut eigen_values = Matrix::<2, 1, f64>::zeros();
    let mut eigen_solution = Matrix::<2, 1, f64>::zeros();
    unsafe {
        sa_eigen_svd_singular_values_f64(
            design.as_slice().as_ptr(),
            4,
            2,
            eigen_values.as_mut_slice().as_mut_ptr(),
        );
        sa_eigen_svd_solve_f64(
            design.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            4,
            2,
            1,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    let svd = design.svd().expect("tall matrix is supported");
    assert_close_f64(svd.singular_values().as_slice(), eigen_values.as_slice());
    assert_close_f64(svd.solve(&rhs).as_slice(), eigen_solution.as_slice());
    let mut output = Matrix::<2, 1, f64>::zeros();
    svd.solve_into(&rhs, &mut output);
    assert_close_f64(output.as_slice(), eigen_solution.as_slice());
}

#[test]
fn randomized_svd_matches_eigen() {
    let design = generated_f64::<6, 3>(91);
    let rhs = generated_f64::<6, 2>(127);
    let mut eigen_values = Matrix::<3, 1, f64>::zeros();
    let mut eigen_solution = Matrix::<3, 2, f64>::zeros();
    unsafe {
        sa_eigen_svd_singular_values_f64(
            design.as_slice().as_ptr(),
            6,
            3,
            eigen_values.as_mut_slice().as_mut_ptr(),
        );
        sa_eigen_svd_solve_f64(
            design.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            6,
            3,
            2,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    let svd = design.svd().expect("tall matrix is supported");
    assert_close_f64(svd.singular_values().as_slice(), eigen_values.as_slice());
    assert_close_f64(svd.solve(&rhs).as_slice(), eigen_solution.as_slice());
}

#[test]
fn self_adjoint_eigenvalues_match_eigen() {
    let matrix =
        Matrix::<3, 3, f64>::from_rows([[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]]);
    let mut eigen_values = Matrix::<3, 1, f64>::zeros();
    unsafe {
        sa_eigen_self_adjoint_eigenvalues_f64(
            matrix.as_slice().as_ptr(),
            3,
            eigen_values.as_mut_slice().as_mut_ptr(),
        );
    }
    let eigen = matrix.self_adjoint_eigen().expect("matrix is symmetric");
    assert_close_f64(eigen.eigenvalues().as_slice(), eigen_values.as_slice());
    let mut eigen_vectors = Matrix::<3, 3, f64>::zeros();
    unsafe {
        sa_eigen_self_adjoint_eigenvectors_f64(
            matrix.as_slice().as_ptr(),
            3,
            eigen_vectors.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_eigenvectors_match_f64(eigen.eigenvectors(), &eigen_vectors);
    assert_close_f64(eigen.reconstruct().as_slice(), matrix.as_slice());
}

#[test]
fn f32_self_adjoint_eigenvalues_match_eigen() {
    let matrix =
        Matrix::<3, 3, f32>::from_rows([[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]]);
    let mut eigen_values = Matrix::<3, 1, f32>::zeros();
    unsafe {
        sa_eigen_self_adjoint_eigenvalues_f32(
            matrix.as_slice().as_ptr(),
            3,
            eigen_values.as_mut_slice().as_mut_ptr(),
        );
    }
    let eigen = matrix.self_adjoint_eigen().expect("matrix is symmetric");
    assert_close_f32(eigen.eigenvalues().as_slice(), eigen_values.as_slice());
}

#[test]
fn triangular_views_match_eigen() {
    let lower =
        Matrix::<3, 3, f64>::from_rows([[2.0, 0.0, 0.0], [3.0, 4.0, 0.0], [-1.0, 2.0, 5.0]]);
    let upper =
        Matrix::<3, 3, f64>::from_rows([[2.0, -1.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 5.0]]);
    let rhs = Matrix::<3, 2, f64>::from_rows([[2.0, 4.0], [11.0, 6.0], [7.0, 13.0]]);

    let mut eigen_lower_solution = Matrix::<3, 2, f64>::zeros();
    let mut eigen_lower_product = Matrix::<3, 2, f64>::zeros();
    let mut eigen_upper_solution = Matrix::<3, 2, f64>::zeros();
    let mut eigen_upper_product = Matrix::<3, 2, f64>::zeros();
    unsafe {
        sa_eigen_lower_triangular_solve_f64(
            lower.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            2,
            eigen_lower_solution.as_mut_slice().as_mut_ptr(),
        );
        sa_eigen_lower_triangular_mul_f64(
            lower.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            2,
            eigen_lower_product.as_mut_slice().as_mut_ptr(),
        );
        sa_eigen_upper_triangular_solve_f64(
            upper.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            2,
            eigen_upper_solution.as_mut_slice().as_mut_ptr(),
        );
        sa_eigen_upper_triangular_mul_f64(
            upper.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            2,
            eigen_upper_product.as_mut_slice().as_mut_ptr(),
        );
    }

    let lower_view = lower.lower_triangular();
    let upper_view = upper.upper_triangular();
    assert_close_f64(
        lower_view.solve(&rhs).as_slice(),
        eigen_lower_solution.as_slice(),
    );
    assert_close_f64(
        upper_view.solve(&rhs).as_slice(),
        eigen_upper_solution.as_slice(),
    );
    let mut lower_product = Matrix::<3, 2, f64>::zeros();
    let mut upper_product = Matrix::<3, 2, f64>::zeros();
    lower_view.mul_into(&rhs, &mut lower_product);
    upper_view.mul_into(&rhs, &mut upper_product);
    assert_close_f64(lower_product.as_slice(), eigen_lower_product.as_slice());
    assert_close_f64(upper_product.as_slice(), eigen_upper_product.as_slice());
}

#[test]
fn quaternion_rotation_matches_eigen() {
    let axis = Vector::<3, f64>::from_columns([[0.0, 0.0, 1.0]]);
    let quaternion =
        Quaternion::from_axis_angle(&axis, core::f64::consts::FRAC_PI_2).expect("axis is nonzero");
    let vector = Vector::<3, f64>::from_columns([[1.0, 0.0, 0.0]]);
    let quaternion_input = [
        quaternion.scalar(),
        quaternion.vector()[0],
        quaternion.vector()[1],
        quaternion.vector()[2],
    ];
    let mut eigen_matrix = Matrix::<3, 3, f64>::zeros();
    let mut eigen_vector = Vector::<3, f64>::zeros();
    unsafe {
        sa_eigen_quaternion_rotation_f64(
            quaternion_input.as_ptr(),
            eigen_matrix.as_mut_slice().as_mut_ptr(),
            eigen_vector.as_mut_slice().as_mut_ptr(),
            vector.as_slice().as_ptr(),
        );
    }

    let rotation = quaternion
        .to_rotation_matrix()
        .expect("quaternion is valid");
    assert_close_f64(rotation.matrix().as_slice(), eigen_matrix.as_slice());
    assert_close_f64(rotation.apply(&vector).as_slice(), eigen_vector.as_slice());
}

#[test]
fn isometry_transform_matches_eigen() {
    let quaternion = Quaternion::from_axis_angle(
        &Vector::<3, f64>::from_columns([[0.0, 0.0, 1.0]]),
        core::f64::consts::FRAC_PI_2,
    )
    .expect("axis is nonzero");
    let translation = Vector::<3, f64>::from_columns([[1.0, 2.0, 3.0]]);
    let point = Vector::<3, f64>::from_columns([[0.5, -1.0, 2.0]]);
    let quaternion_input = [
        quaternion.scalar(),
        quaternion.vector()[0],
        quaternion.vector()[1],
        quaternion.vector()[2],
    ];
    let rotation = quaternion
        .to_rotation_matrix()
        .expect("quaternion is valid");
    let isometry = Isometry::from_parts(rotation, translation);
    let mut eigen_matrix = Matrix::<4, 4, f64>::zeros();
    let mut eigen_point = Vector::<3, f64>::zeros();
    unsafe {
        sa_eigen_isometry_transform_f64(
            quaternion_input.as_ptr(),
            translation.as_slice().as_ptr(),
            point.as_slice().as_ptr(),
            eigen_matrix.as_mut_slice().as_mut_ptr(),
            eigen_point.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f64(
        isometry.to_homogeneous().as_slice(),
        eigen_matrix.as_slice(),
    );
    assert_close_f64(
        isometry.apply_point(&point).as_slice(),
        eigen_point.as_slice(),
    );
}

#[test]
fn affine_transform_matches_eigen() {
    let affine_matrix = Matrix::<4, 4, f64>::from_rows([
        [2.0, 0.25, 0.0, 1.0],
        [0.0, 3.0, -0.5, -2.0],
        [0.25, 0.0, 1.5, 0.75],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    let point = Vector::<3, f64>::from_columns([[0.5, -1.0, 2.0]]);
    let affine = AffineTransform::from_matrix(affine_matrix).expect("valid affine matrix");
    let mut eigen_point = Vector::<3, f64>::zeros();
    unsafe {
        sa_eigen_affine_transform_f64(
            affine_matrix.as_slice().as_ptr(),
            point.as_slice().as_ptr(),
            eigen_point.as_mut_slice().as_mut_ptr(),
        );
    }
    assert_close_f64(
        affine.apply_point(&point).as_slice(),
        eigen_point.as_slice(),
    );
}

#[test]
fn tall_qr_matches_eigen() {
    let design = Matrix::<6, 3, f64>::from_rows([
        [1.0, 0.0, 2.0],
        [2.0, 1.0, 1.0],
        [3.0, 1.0, 0.0],
        [4.0, 2.0, 1.0],
        [5.0, 3.0, 2.0],
        [6.0, 5.0, 1.0],
    ]);
    let rhs = Matrix::<6, 2, f64>::from_rows([
        [1.0, -2.0],
        [2.0, 0.5],
        [4.0, 1.0],
        [5.0, 2.5],
        [7.0, 3.0],
        [8.0, 4.5],
    ]);
    let mut eigen_solution = Matrix::<3, 2, f64>::zeros();
    unsafe {
        sa_eigen_qr_solve_f64(
            design.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            6,
            3,
            2,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    let solution = design
        .householder_qr()
        .solve_least_squares(&rhs)
        .expect("tall design matrix is full rank");
    assert_close_f64(solution.as_slice(), eigen_solution.as_slice());
}

#[test]
fn tall_f32_qr_matches_eigen() {
    let design = Matrix::<6, 3, f32>::from_rows([
        [1.0, 0.0, 2.0],
        [2.0, 1.0, 1.0],
        [3.0, 1.0, 0.0],
        [4.0, 2.0, 1.0],
        [5.0, 3.0, 2.0],
        [6.0, 5.0, 1.0],
    ]);
    let rhs = Matrix::<6, 2, f32>::from_rows([
        [1.0, -2.0],
        [2.0, 0.5],
        [4.0, 1.0],
        [5.0, 2.5],
        [7.0, 3.0],
        [8.0, 4.5],
    ]);
    let mut eigen_solution = Matrix::<3, 2, f32>::zeros();
    unsafe {
        sa_eigen_qr_solve_f32(
            design.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            6,
            3,
            2,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    let solution = design
        .householder_qr()
        .solve_least_squares(&rhs)
        .expect("tall design matrix is full rank");
    assert_close_f32(solution.as_slice(), eigen_solution.as_slice());
}

#[test]
fn rank_deficient_basic_qr_matches_eigen() {
    let design = Matrix::<3, 2, f64>::from_rows([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]);
    let rhs = Matrix::<3, 1, f64>::from_columns([[1.0, 1.0, 1.0]]);
    let mut eigen_solution = Matrix::<2, 1, f64>::zeros();
    unsafe {
        sa_eigen_col_piv_qr_solve_f64(
            design.as_slice().as_ptr(),
            rhs.as_slice().as_ptr(),
            3,
            2,
            1,
            eigen_solution.as_mut_slice().as_mut_ptr(),
        );
    }
    let solution = design
        .col_piv_householder_qr()
        .solve_least_squares_basic(&rhs)
        .expect("tall rank-deficient solve");
    assert_close_f64(solution.as_slice(), eigen_solution.as_slice());
}
