use stack_algebra::{vector, CscError, StaticCscMatrix, StaticCscPattern};

type Matrix = StaticCscMatrix<4, 3, 5, f64>;

#[test]
fn canonical_pattern_preserves_empty_columns_and_orders_insertions() {
    let mut matrix = Matrix::new();

    matrix.insert(3, 2, 4.0).unwrap();
    matrix.insert(2, 0, 2.0).unwrap();
    matrix.insert(0, 0, 1.0).unwrap();
    matrix.insert(1, 2, 3.0).unwrap();

    assert_eq!(matrix.nnz(), 4);
    assert_eq!(matrix.values(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(matrix.row_indices(), &[0, 2, 1, 3]);
    assert_eq!(matrix.column_starts(), &[0, 2, 2]);
    assert_eq!(matrix.column_end(0), Some(2));
    assert_eq!(matrix.column_end(1), Some(2));
    assert_eq!(matrix.column_end(2), Some(4));

    // Inserting an existing coordinate updates its value instead of adding a
    // duplicate entry.
    matrix.insert(2, 0, 20.0).unwrap();
    assert_eq!(matrix.nnz(), 4);
    assert_eq!(matrix.get(2, 0), Some(&20.0));
    assert_eq!(matrix.row_indices(), &[0, 2, 1, 3]);
}

#[test]
fn matvec_handles_empty_columns_and_repeated_value_updates() {
    let mut matrix =
        Matrix::from_pattern(&[1.0, 2.0, 3.0, 4.0], &[0, 2, 1, 3], &[0, 2, 2, 4]).unwrap();
    let input = vector![5.0; 7.0; 11.0];

    assert_eq!(matrix.matvec(&input), vector![5.0; 33.0; 10.0; 44.0]);

    let mut output = vector![0.0; 0.0; 0.0; 0.0];
    matrix.matvec_into(&input, &mut output);
    assert_eq!(output, vector![5.0; 33.0; 10.0; 44.0]);

    matrix.values_mut()[0] = 2.0;
    matrix.set_value(3, 2, 8.0).unwrap();
    assert_eq!(matrix.matvec(&input), vector![10.0; 33.0; 10.0; 88.0]);

    matrix.set_values(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(matrix.values(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        matrix.set_values(&[1.0, 2.0]),
        Err(CscError::LengthMismatch)
    );
}

#[test]
fn sparse_storage_footprints_are_compile_time_constants() {
    type MatrixType = StaticCscMatrix<3, 3, 6, f64>;
    type Pattern = StaticCscPattern<3, 3, 6>;
    const MATRIX_BYTES: usize = MatrixType::storage_bytes();
    const PATTERN_BYTES: usize = Pattern::storage_bytes();
    assert_eq!(MATRIX_BYTES, core::mem::size_of::<MatrixType>());
    assert_eq!(PATTERN_BYTES, core::mem::size_of::<Pattern>());
}

#[test]
fn malformed_patterns_are_rejected_without_partial_construction() {
    assert_eq!(
        Matrix::from_pattern(&[1.0], &[], &[0, 1, 1, 1]),
        Err(CscError::LengthMismatch)
    );
    assert_eq!(
        Matrix::from_pattern(&[1.0], &[0], &[0, 1, 1]),
        Err(CscError::LengthMismatch)
    );
    assert_eq!(
        Matrix::from_pattern(&[1.0], &[0], &[1, 1, 1, 1]),
        Err(CscError::InvalidColumnPointers)
    );
    assert_eq!(
        Matrix::from_pattern(&[1.0], &[0], &[0, 2, 2, 1]),
        Err(CscError::InvalidColumnPointers)
    );
    assert_eq!(
        Matrix::from_pattern(&[1.0], &[0], &[0, 0, 0, 0]),
        Err(CscError::InvalidColumnPointers)
    );
    assert_eq!(
        Matrix::from_pattern(&[1.0], &[4], &[0, 1, 1, 1]),
        Err(CscError::InvalidRowIndices)
    );
    assert_eq!(
        Matrix::from_pattern(&[1.0, 2.0], &[2, 1], &[0, 2, 2, 2]),
        Err(CscError::InvalidRowIndices)
    );
    assert_eq!(
        Matrix::from_pattern(&[1.0], &[0], &[0, 0, 0, 0]),
        Err(CscError::InvalidColumnPointers)
    );
}

#[test]
fn capacity_and_index_errors_are_reported() {
    let mut matrix = StaticCscMatrix::<2, 2, 2, f64>::new();
    matrix.insert(0, 0, 1.0).unwrap();
    matrix.insert(1, 1, 2.0).unwrap();
    assert_eq!(matrix.insert(0, 1, 3.0), Err(CscError::CapacityExceeded));

    assert_eq!(matrix.insert(2, 0, 1.0), Err(CscError::IndexOutOfBounds));
    assert_eq!(matrix.insert(0, 2, 1.0), Err(CscError::IndexOutOfBounds));
    assert_eq!(matrix.set_value(1, 0, 1.0), Err(CscError::EntryNotFound));
    assert_eq!(matrix.set_value(0, 2, 1.0), Err(CscError::IndexOutOfBounds));
    assert_eq!(matrix.get(1, 0), None);
    assert_eq!(matrix.get(2, 0), None);
}

#[test]
fn clear_resets_pattern_and_values() {
    let mut matrix = Matrix::from_pattern(&[1.0, 2.0], &[0, 1], &[0, 1, 2, 2]).unwrap();
    matrix.clear();
    assert_eq!(matrix.nnz(), 0);
    assert!(matrix.values().is_empty());
    assert!(matrix.row_indices().is_empty());
    assert_eq!(matrix.column_starts(), &[0, 0, 0]);
    assert_eq!(matrix.column_end(0), Some(0));
    assert_eq!(matrix.column_end(2), Some(0));

    matrix.insert(3, 2, 9.0).unwrap();
    assert_eq!(matrix.get(3, 2), Some(&9.0));
}

#[test]
fn validated_pattern_can_be_reused_for_numeric_updates() {
    let pattern = StaticCscPattern::<4, 3, 5>::from_arrays(&[0, 2, 1, 3], &[0, 2, 2, 4]).unwrap();
    assert_eq!(pattern.nnz(), 4);
    assert_eq!(pattern.column_starts(), &[0, 2, 2]);
    assert_eq!(pattern.column_end(2), Some(4));

    let mut matrix = StaticCscMatrix::with_pattern(pattern, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(matrix.pattern(), &pattern);
    matrix.set_values(&[10.0, 20.0, 30.0, 40.0]).unwrap();
    assert_eq!(matrix.values(), &[10.0, 20.0, 30.0, 40.0]);
    assert_eq!(
        StaticCscMatrix::<4, 3, 5, f64>::with_pattern(pattern, &[1.0]),
        Err(CscError::LengthMismatch)
    );
}
