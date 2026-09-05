//! Compile-time coverage for the primary public API forms.
//!
//! The `compile_fail` rustdoc examples in `src/lib.rs` cover rejected shape
//! and scalar combinations. This test ensures the corresponding supported
//! dense, bounded, view, solver, geometry, sparse, and scalar-extension forms
//! remain straightforward to use.

use stack_algebra::{
    matrix, matvec_view, vector, FactorizationScalar, Map, Matrix, MatrixBuf, MatrixScalar,
    Quaternion, ReductionScalar, StaticCscPattern, Zero,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExternalScalar(i32);

impl core::ops::Add for ExternalScalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl core::ops::Mul for ExternalScalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Zero for ExternalScalar {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

// Downstream scalar types opt into portable factor/update behavior without
// naming or implementing any architecture-specific backend types.
impl FactorizationScalar for ExternalScalar {}
impl MatrixScalar for ExternalScalar {}
impl ReductionScalar for ExternalScalar {}

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

#[test]
fn public_bounded_geometry_and_sparse_forms_compile_and_run() {
    let source = matrix![1.0_f32, 2.0; 3.0, 4.0];
    let mut bounded = MatrixBuf::<4, 4, f32>::from_matrix(&source).expect("capacity fits source");
    bounded
        .resize_zeroed(3, 3)
        .expect("expanded shape fits bounded capacity");
    bounded
        .resize_with(2, 2, 9.0)
        .expect("shrunk shape fits bounded capacity");
    let view = bounded.as_view::<2, 2>().expect("active dimensions match");
    assert_eq!(view.to_matrix(), source);

    let rotation =
        Quaternion::from_axis_angle(&vector![0.0_f32; 0.0; 1.0], core::f32::consts::FRAC_PI_2)
            .expect("axis is nonzero and finite");
    let rotated = rotation
        .rotate_vector(&vector![1.0_f32; 0.0; 0.0])
        .expect("quaternion represents a rotation");
    assert!(rotated[0].abs() < 1.0e-5);
    assert!((rotated[1] - 1.0).abs() < 1.0e-5);

    let pattern = StaticCscPattern::<2, 2, 2>::from_arrays(&[0, 1], &[0, 1, 2])
        .expect("canonical diagonal pattern is valid");
    assert_eq!(pattern.nnz(), 2);
    assert_eq!(pattern.entry_index(1, 1), Some(1));
}

#[test]
fn external_scalar_uses_portable_contract_without_backend_types() {
    let lhs = Matrix::<2, 2, ExternalScalar>::from_rows([
        [ExternalScalar(1), ExternalScalar(2)],
        [ExternalScalar(3), ExternalScalar(4)],
    ]);
    let rhs = Matrix::<2, 1, ExternalScalar>::from_rows([[ExternalScalar(5)], [ExternalScalar(6)]]);
    assert_eq!(
        lhs * rhs,
        Matrix::from_rows([[ExternalScalar(17)], [ExternalScalar(39)]])
    );

    let vector =
        Matrix::<2, 1, ExternalScalar>::from_rows([[ExternalScalar(5)], [ExternalScalar(6)]]);
    assert_eq!(
        lhs.matvec(&vector),
        Matrix::from_rows([[ExternalScalar(17)], [ExternalScalar(39)]])
    );
}
