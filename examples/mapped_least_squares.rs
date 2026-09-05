use stack_algebra::{ColPivHouseholderQr, Map, Matrix};

fn main() {
    // Column-major Jacobian supplied by an external/generated buffer.
    let storage = [
        1.0_f64, 0.0, 1.0, 2.0, // column 0
        0.0, 1.0, 1.0, -1.0, // column 1
        1.0, 1.0, 0.0, 1.0, // column 2
    ];
    let jacobian = Map::<4, 3, f64>::from_slice(&storage).expect("4x3 buffer");
    let owned = Matrix::<4, 3, f64>::from_fn(|row, column| jacobian[(row, column)]);
    let expected = Matrix::<3, 1, f64>::from_rows([[0.5], [-1.0], [2.0]]);
    let rhs = owned * expected;

    // The factorization reads the mapped input directly; no owning input copy is required.
    let factor =
        ColPivHouseholderQr::try_decompose_view(&jacobian).expect("finite, full-rank Jacobian");
    let solution = factor
        .try_solve_least_squares(&rhs)
        .expect("full-rank least-squares system");

    assert!((owned * solution - rhs).norm() < 1.0e-10);
    assert!((solution - expected).norm() < 1.0e-10);
}
