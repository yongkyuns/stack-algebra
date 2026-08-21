//! Stabilization tests for numerical behavior and capacity contracts.
//!
//! Numerical tests intentionally focus on mathematical invariants rather than
//! factor-storage parity with another library. The dimensions straddle common
//! SIMD packet widths so optimized and scalar-tail paths remain covered by the
//! same solve contract.

use stack_algebra::{Cholesky, CscError, Matrix, StaticBlockCscMatrix, StaticBlockCsrMatrix};

fn check_cholesky_f64<const N: usize>() {
    let seed = Matrix::<N, N, f64>::from_fn(|row, column| {
        let signed = ((row * 13 + column * 7 + 3) % 17) as f64 - 8.0;
        signed / 9.0
    });
    let matrix = seed.transpose() * seed + Matrix::<N, N, f64>::eye();
    let rhs = Matrix::<N, 1, f64>::from_fn(|row, _| (row as f64 + 1.0) / 3.0 - 0.5);

    let factor = Cholesky::try_decompose(&matrix).expect("constructed matrix is SPD");
    let solution = factor.solve(&rhs);
    let residual = matrix * solution - rhs;
    let scale = rhs.norm().max(1.0);

    assert!(
        residual.norm() <= 2.0e-10 * scale,
        "f64 Cholesky residual {} exceeds scale {} for N={N}",
        residual.norm(),
        scale
    );
}

fn check_cholesky_f32<const N: usize>() {
    let seed = Matrix::<N, N, f32>::from_fn(|row, column| {
        let signed = ((row * 11 + column * 5 + 1) % 13) as f32 - 6.0;
        signed / 7.0
    });
    let matrix = seed.transpose() * seed + Matrix::<N, N, f32>::eye();
    let rhs = Matrix::<N, 1, f32>::from_fn(|row, _| (row as f32 + 1.0) / 4.0 - 0.5);

    let factor = Cholesky::try_decompose(&matrix).expect("constructed matrix is SPD");
    let solution = factor.solve(&rhs);
    let residual = matrix * solution - rhs;
    let scale = rhs.norm().max(1.0);

    assert!(
        residual.norm() <= 4.0e-4 * scale,
        "f32 Cholesky residual {} exceeds scale {} for N={N}",
        residual.norm(),
        scale
    );
}

#[test]
fn f64_cholesky_contract_straddles_four_lane_boundaries() {
    check_cholesky_f64::<1>();
    check_cholesky_f64::<2>();
    check_cholesky_f64::<3>();
    check_cholesky_f64::<4>();
    check_cholesky_f64::<5>();
    check_cholesky_f64::<7>();
    check_cholesky_f64::<8>();
    check_cholesky_f64::<9>();
}

#[test]
fn f32_cholesky_contract_straddles_eight_lane_boundary() {
    check_cholesky_f32::<3>();
    check_cholesky_f32::<4>();
    check_cholesky_f32::<7>();
    check_cholesky_f32::<8>();
    check_cholesky_f32::<9>();
}

#[test]
fn block_sparse_constructors_report_required_capacity() {
    let block = Matrix::from_rows([[1.0_f32]]);
    let values = [block, block];

    let csc =
        StaticBlockCscMatrix::<1, 1, 2, 2, 1, f32>::from_pattern(&values, &[0, 1], &[0, 1, 2]);
    assert_eq!(
        csc,
        Err(CscError::CapacityExceeded {
            required: 2,
            capacity: 1,
        })
    );

    let csr =
        StaticBlockCsrMatrix::<1, 1, 2, 2, 1, f32>::from_pattern(&values, &[0, 1], &[0, 1, 2]);
    assert_eq!(
        csr,
        Err(CscError::CapacityExceeded {
            required: 2,
            capacity: 1,
        })
    );
}
