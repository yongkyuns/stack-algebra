//! Cross-solver numerical contracts independent of factor storage layout.

use stack_algebra::{
    ColPivHouseholderQr, HouseholderQr, Ldlt, Matrix, PartialPivLu, SelfAdjointEigen, StridedMap,
    Svd,
};

fn generated<const M: usize, const N: usize>(scale: f64) -> Matrix<M, N, f64> {
    Matrix::from_fn(|row, column| {
        let value = ((row * 17 + column * 11 + 3) % 19) as f64 / 13.0 - 0.5;
        if row == column {
            scale * (value + 2.0)
        } else {
            scale * value
        }
    })
}

fn max_abs<const M: usize, const N: usize>(matrix: &Matrix<M, N, f64>) -> f64 {
    matrix
        .as_slice()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
}

fn assert_close<const M: usize, const N: usize>(
    actual: &Matrix<M, N, f64>,
    expected: &Matrix<M, N, f64>,
    relative_tolerance: f64,
) {
    let error = max_abs(&(*actual - *expected));
    let scale = max_abs(expected).max(1.0);
    assert!(
        error <= relative_tolerance * scale,
        "maximum error {error:e} exceeds tolerance {relative_tolerance:e} at scale {scale:e}"
    );
}

fn assert_orthonormal_columns<const M: usize, const N: usize>(
    columns: &Matrix<M, N, f64>,
    tolerance: f64,
) {
    assert_close(
        &(columns.transpose() * *columns),
        &Matrix::<N, N, f64>::eye(),
        tolerance,
    );
}

fn svd_reconstruct<const M: usize, const N: usize>(svd: &Svd<M, N, f64>) -> Matrix<M, N, f64> {
    let diagonal = Matrix::from_fn(|row, column| {
        if row == column {
            svd.singular_values()[row]
        } else {
            0.0
        }
    });
    *svd.u() * diagonal * svd.v().transpose()
}

#[test]
fn f64_dense_solver_contracts_across_scales() {
    for scale in [0.25, 1.0, 10_000.0] {
        let tall = generated::<4, 3>(scale);
        let rhs = tall * Matrix::from_rows([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]]);

        let qr = HouseholderQr::try_decompose(&tall).expect("finite full-rank input");
        assert_close(&qr.apply_q(&qr.r()), &tall, 1.0e-11);
        assert_orthonormal_columns(&qr.apply_q(&Matrix::<4, 4, f64>::eye()), 1.0e-11);
        assert_close(
            &(tall * qr.try_solve_least_squares(&rhs).expect("full rank")),
            &rhs,
            1.0e-11,
        );

        let pivoted = ColPivHouseholderQr::try_decompose(&tall).expect("finite full-rank input");
        let permuted = Matrix::from_fn(|row, column| tall[(row, pivoted.permutation()[column])]);
        assert_eq!(pivoted.rank(), 3);
        assert_close(&pivoted.apply_q(&pivoted.r()), &permuted, 1.0e-11);
        assert_close(
            &(tall * pivoted.try_solve_least_squares(&rhs).expect("full rank")),
            &rhs,
            1.0e-11,
        );

        let square = generated::<3, 3>(scale);
        let square_rhs = Matrix::from_rows([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]]);
        let lu = PartialPivLu::try_decompose(&square).expect("finite input");
        assert_close(
            &(lu.permutation() * square),
            &(lu.lower() * lu.upper()),
            1.0e-11,
        );
        assert_close(&(square * lu.solve(&square_rhs)), &square_rhs, 1.0e-11);

        let symmetric = square.transpose() * square
            + Matrix::<3, 3, f64>::from_rows([
                [scale, 0.0, 0.0],
                [0.0, scale, 0.0],
                [0.0, 0.0, scale],
            ]);
        let ldlt = Ldlt::try_decompose(&symmetric).expect("nonsingular symmetric input");
        let permuted = ldlt.permutation() * symmetric * ldlt.permutation().transpose();
        assert_close(
            &permuted,
            &(ldlt.lower() * ldlt.diagonal_matrix() * ldlt.lower().transpose()),
            1.0e-11,
        );
        assert_close(&(symmetric * ldlt.solve(&square_rhs)), &square_rhs, 1.0e-11);

        let svd = Svd::try_decompose(&tall).expect("finite input converges");
        assert_close(&svd_reconstruct(&svd), &tall, 1.0e-10);
        assert_orthonormal_columns(svd.u(), 1.0e-10);
        assert_orthonormal_columns(svd.v(), 1.0e-10);
        assert!(svd
            .singular_values()
            .as_slice()
            .windows(2)
            .all(|pair| pair[0] >= pair[1]));
        assert_eq!(svd.rank(), 3);
        assert_close(&(tall * svd.solve(&rhs)), &rhs, 1.0e-10);

        let eigen = SelfAdjointEigen::try_decompose(&symmetric).expect("symmetric input");
        assert_close(&eigen.reconstruct(), &symmetric, 1.0e-10);
        assert_orthonormal_columns(eigen.eigenvectors(), 1.0e-10);
        assert!(eigen
            .eigenvalues()
            .as_slice()
            .windows(2)
            .all(|pair| pair[0] <= pair[1]));
    }
}

#[test]
fn f32_dense_solver_contracts() {
    let tall = generated::<4, 3>(1.0).cast::<f32>();
    let rhs = tall * Matrix::from_rows([[1.0_f32], [0.5], [-1.5]]);

    let qr = tall.try_householder_qr().expect("finite full-rank input");
    assert!((qr.apply_q(&qr.r()) - tall).norm() < 2.0e-5);
    assert!((tall * qr.try_solve_least_squares(&rhs).expect("full rank") - rhs).norm() < 2.0e-5);

    let pivoted = tall
        .try_col_piv_householder_qr()
        .expect("finite full-rank input");
    assert_eq!(pivoted.rank(), 3);
    assert!(
        (tall * pivoted.try_solve_least_squares(&rhs).expect("full rank") - rhs).norm() < 2.0e-5
    );

    let square = generated::<3, 3>(1.0).cast::<f32>();
    let square_rhs = Matrix::from_rows([[1.0_f32], [0.5], [-1.5]]);
    let lu = square.try_partial_piv_lu().expect("finite input");
    assert!((square * lu.solve(&square_rhs) - square_rhs).norm() < 2.0e-5);

    let symmetric = square.transpose() * square + Matrix::<3, 3, f32>::eye();
    let ldlt = symmetric.try_ldlt().expect("nonsingular symmetric input");
    assert!((symmetric * ldlt.solve(&square_rhs) - square_rhs).norm() < 3.0e-5);

    let svd = tall.try_svd().expect("finite input converges");
    let diagonal = Matrix::from_fn(|row, column| {
        if row == column {
            svd.singular_values()[row]
        } else {
            0.0
        }
    });
    assert!((*svd.u() * diagonal * svd.v().transpose() - tall).norm() < 3.0e-5);
    assert!((tall * svd.solve(&rhs) - rhs).norm() < 3.0e-5);

    let eigen = symmetric.try_self_adjoint_eigen().expect("symmetric input");
    assert!((eigen.reconstruct() - symmetric).norm() < 3.0e-5);
}

#[test]
fn dense_solver_view_contracts_match_owned_input() {
    let input =
        Matrix::<3, 3, f64>::from_rows([[5.0, 1.0, 0.5], [1.0, 4.0, 0.25], [0.5, 0.25, 3.0]]);
    let row_major = [5.0, 1.0, 0.5, 1.0, 4.0, 0.25, 0.5, 0.25, 3.0];
    let view = StridedMap::<3, 3, f64>::from_slice(&row_major, 3, 1).expect("row-major input fits");
    let rhs = Matrix::from_rows([[1.0], [-2.0], [0.5]]);

    let qr = HouseholderQr::try_decompose_view(&view).expect("finite input");
    assert_close(&qr.apply_q(&qr.r()), &input, 1.0e-12);
    let pivoted = ColPivHouseholderQr::try_decompose_view(&view).expect("finite input");
    let permuted = Matrix::from_fn(|row, column| input[(row, pivoted.permutation()[column])]);
    assert_close(&pivoted.apply_q(&pivoted.r()), &permuted, 1.0e-12);

    let lu = PartialPivLu::try_decompose_view(&view).expect("finite input");
    assert_close(&(input * lu.solve(&rhs)), &rhs, 1.0e-12);
    let ldlt = Ldlt::try_decompose_view(&view).expect("nonsingular symmetric input");
    assert_close(&(input * ldlt.solve(&rhs)), &rhs, 1.0e-12);

    let svd = Svd::try_decompose_view(&view).expect("finite input converges");
    assert_close(&svd_reconstruct(&svd), &input, 1.0e-10);
    let eigen = SelfAdjointEigen::try_decompose_view(&view).expect("symmetric input");
    assert_close(&eigen.reconstruct(), &input, 1.0e-10);
}
