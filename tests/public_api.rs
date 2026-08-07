//! Compile-time coverage for the primary public API forms.
//!
//! The `compile_fail` rustdoc examples in `src/lib.rs` cover rejected shape
//! and scalar combinations. This test ensures the corresponding supported
//! f32, f64, view, and solver forms remain straightforward to use.

use stack_algebra::{matrix, matvec_view, vector, Map, Matrix};

#[test]
fn public_fixed_matrix_view_and_solver_forms_compile_and_run() {
    let single: Matrix<2, 2, f32> = matrix![1.0, 2.0; 3.0, 4.0];
    let single_product = single * Matrix::<2, 2, f32>::eye();
    assert_eq!(single_product, single);

    let double: Matrix<2, 2, f64> = matrix![4.0, 1.0; 1.0, 3.0];
    let factor = double.try_cholesky().expect("matrix is positive definite");
    let solution = factor.solve(&matrix![1.0_f64; 2.0]);
    assert_eq!(double * solution, matrix![1.0_f64; 2.0]);

    let storage = [1.0_f32, 3.0, 2.0, 4.0];
    let mapped = Map::<2, 2, f32>::from_slice(&storage).expect("storage fits shape");
    assert_eq!(
        matvec_view(&mapped, &vector![5.0_f32; 6.0]),
        vector![17.0; 39.0]
    );

    let widened: Matrix<2, 2, f64> = single.cast();
    assert_eq!(widened, matrix![1.0_f64, 2.0; 3.0, 4.0]);
}
