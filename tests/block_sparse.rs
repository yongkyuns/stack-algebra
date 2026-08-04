use approx::assert_relative_eq;
use stack_algebra::{matrix, Matrix, StaticBlockCscMatrix};

type Blocks = StaticBlockCscMatrix<2, 2, 2, 2, 3, f64>;

#[test]
fn block_csc_reports_scalar_dimensions_and_blocks() {
    let values = [
        Matrix::from_rows([[1.0, 0.0], [0.0, 1.0]]),
        Matrix::from_rows([[2.0, 0.0], [0.0, 2.0]]),
    ];
    let matrix = Blocks::from_pattern(&values, &[0, 1], &[0, 1, 2]).unwrap();
    assert_eq!(matrix.block_rows(), 2);
    assert_eq!(matrix.block_cols(), 2);
    assert_eq!(matrix.rows(), 4);
    assert_eq!(matrix.cols(), 4);
    assert_eq!(matrix.block(0, 0), Some(&values[0]));
    assert_eq!(matrix.block(1, 1), Some(&values[1]));
    assert!(matrix.block(1, 0).is_none());
    assert!(matrix.block(0, 1).is_none());
}

#[test]
fn block_csc_matvec_matches_expanded_matrix() {
    let values = [
        Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]),
        Matrix::from_rows([[5.0, 6.0], [7.0, 8.0]]),
    ];
    let matrix = Blocks::from_pattern(&values, &[0, 1], &[0, 1, 2]).unwrap();
    let rhs = [1.0, 2.0, 3.0, 4.0];
    let mut actual = [0.0; 4];
    matrix.matvec_into(&rhs, &mut actual).unwrap();

    let expanded = matrix![
        1.0, 2.0, 0.0, 0.0;
        3.0, 4.0, 0.0, 0.0;
        0.0, 0.0, 5.0, 6.0;
        0.0, 0.0, 7.0, 8.0;
    ];
    let expected = expanded * Matrix::<4, 1, f64>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    let actual = Matrix::<4, 1, f64>::from_columns([actual]);
    assert_relative_eq!(actual, expected, epsilon = 1e-12);
}
