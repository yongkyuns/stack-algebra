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
    fn sa_eigen_add_f64(
        lhs: *const f64,
        rhs: *const f64,
        rows: usize,
        columns: usize,
        output: *mut f64,
    );
    fn sa_eigen_transpose_f64(input: *const f64, rows: usize, columns: usize, output: *mut f64);
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
