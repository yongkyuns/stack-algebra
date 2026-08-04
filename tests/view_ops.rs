use approx::assert_relative_eq;
use stack_algebra::{matmul_view_into, matrix, matvec_view, Map, StridedMap};

#[test]
fn view_matvec_matches_owned_matrix() {
    let storage = [1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let view = Map::<2, 3, _>::from_slice(&storage).unwrap();
    let vector = matrix![2.0_f64; 3.0; 4.0];

    let actual = matvec_view(&view, &vector).unwrap();
    let expected = view.to_matrix().matvec(&vector);
    assert_relative_eq!(actual, expected, epsilon = 1e-12, max_relative = 1e-12);
}

#[test]
fn strided_views_matmul_without_repacking_inputs() {
    let lhs_storage = [1.0_f64, 2.0, 99.0, 3.0, 4.0, 88.0];
    let rhs_storage = [5.0_f64, 6.0, 7.0, 8.0];
    let lhs = StridedMap::<2, 2, _>::from_slice(&lhs_storage, 1, 3).unwrap();
    let rhs = Map::<2, 2, _>::from_slice(&rhs_storage).unwrap();
    let mut actual = stack_algebra::Matrix::<2, 2, f64>::zeros();

    matmul_view_into(&lhs, &rhs, &mut actual).unwrap();

    let expected = lhs.to_matrix() * rhs.to_matrix();
    assert_relative_eq!(actual, expected, epsilon = 1e-12, max_relative = 1e-12);
}
