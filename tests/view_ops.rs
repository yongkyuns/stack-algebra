use approx::assert_relative_eq;
use stack_algebra::{
    matmul_view_into, matrix, matvec_view, Map, MapMut, Matrix, StrideAxis, StridedMap,
    StridedMapMut, ViewError,
};

#[test]
fn view_matvec_matches_owned_matrix() {
    let storage = [1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let view = Map::<2, 3, _>::from_slice(&storage).unwrap();
    let vector = matrix![2.0_f64; 3.0; 4.0];

    let actual = matvec_view(&view, &vector);
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

    matmul_view_into(&lhs, &rhs, &mut actual);

    let expected = lhs.to_matrix() * rhs.to_matrix();
    assert_relative_eq!(actual, expected, epsilon = 1e-12, max_relative = 1e-12);
}

#[test]
fn mapped_views_preserve_buffer_boundaries_and_mutation() {
    let storage = [1, 4, 2, 5, 3, 6, 99];
    let mapped = Map::<2, 3, _>::from_slice(&storage).unwrap();
    assert_eq!(mapped.as_slice(), &storage[..6]);
    assert_eq!(mapped[(1, 2)], 6);
    assert_eq!(mapped.get(2, 0), None);
    assert_eq!(mapped.get(0, 3), None);
    assert!(matches!(
        Map::<3, 3, _>::from_slice(&storage),
        Err(ViewError::BufferTooShort {
            required: 9,
            available: 7,
        })
    ));

    let mut mutable_storage = storage;
    {
        let mut mapped = MapMut::<2, 3, _>::from_slice(&mut mutable_storage).unwrap();
        *mapped.get_mut(1, 1).unwrap() = 50;
        mapped[(0, 2)] = 30;

        let read_only = mapped.as_map();
        assert_eq!(read_only.to_matrix(), matrix![1, 2, 30; 4, 50, 6]);
    }
    assert_eq!(mutable_storage, [1, 4, 2, 50, 30, 6, 99]);

    let mut too_short = [0_i32; 5];
    assert!(matches!(
        MapMut::<2, 3, _>::from_slice(&mut too_short),
        Err(ViewError::BufferTooShort {
            required: 6,
            available: 5,
        })
    ));

    let empty: [u8; 0] = [];
    assert!(matches!(
        Map::<{ usize::MAX }, 2, _>::from_slice(&empty),
        Err(ViewError::SizeOverflow)
    ));

    let mut empty: [u8; 0] = [];
    assert!(matches!(
        MapMut::<{ usize::MAX }, 2, _>::from_slice(&mut empty),
        Err(ViewError::SizeOverflow)
    ));
}

#[test]
fn strided_mapped_views_validate_layout_and_reborrow_safely() {
    let storage = [1, 2, 3, 99, 4, 5, 6, 88];
    let mapped = StridedMap::<2, 3, _>::from_slice(&storage, 4, 1).unwrap();
    assert_eq!(mapped.to_matrix(), matrix![1, 2, 3; 4, 5, 6]);
    assert_eq!(mapped.as_slice(), &storage[..7]);
    assert_eq!(mapped.get(2, 0), None);
    assert_eq!(mapped.get(0, 3), None);

    let mut mutable_storage = storage;
    {
        let mut mapped = StridedMapMut::<2, 3, _>::from_slice(&mut mutable_storage, 4, 1).unwrap();
        *mapped.get_mut(1, 2).unwrap() = 60;
        mapped[(0, 1)] = 20;

        let read_only = mapped.as_map();
        assert_eq!(read_only.to_matrix(), matrix![1, 20, 3; 4, 5, 60]);
    }
    assert_eq!(mutable_storage, [1, 20, 3, 99, 4, 5, 60, 88]);

    assert!(matches!(
        StridedMap::<2, 3, _>::from_slice(&storage[..6], 4, 1),
        Err(ViewError::BufferTooShort {
            required: 7,
            available: 6,
        })
    ));
    assert!(matches!(
        StridedMap::<2, 1, _>::from_slice(&storage, 0, 1),
        Err(ViewError::ZeroStride {
            axis: StrideAxis::Inner,
        })
    ));
    assert!(matches!(
        StridedMap::<1, 2, _>::from_slice(&storage, 1, 0),
        Err(ViewError::ZeroStride {
            axis: StrideAxis::Outer,
        })
    ));
    assert!(matches!(
        StridedMap::<2, 2, _>::from_slice(&storage, usize::MAX, 1),
        Err(ViewError::SizeOverflow)
    ));

    let mut mutable_storage = storage;
    assert!(matches!(
        StridedMapMut::<1, 2, _>::from_slice(&mut mutable_storage, 1, 0),
        Err(ViewError::ZeroStride {
            axis: StrideAxis::Outer,
        })
    ));
}

#[test]
fn block_views_update_only_their_declared_region() {
    let mut matrix = Matrix::<3, 4, i32>::from_rows([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    let block = matrix.block::<2, 2>(1, 1).unwrap();
    assert_eq!(block.to_matrix(), matrix![6, 7; 10, 11]);
    assert_eq!(block.get(2, 0), None);
    assert!(matrix.block::<2, 2>(2, 3).is_none());
    assert!(matrix.block::<1, 1>(usize::MAX, 0).is_none());

    {
        let mut block = matrix.block_mut::<2, 2>(1, 1).unwrap();
        *block.get_mut(0, 1).unwrap() = 70;
        block[(1, 0)] = 100;
        assert_eq!(block.to_matrix(), matrix![6, 70; 100, 11]);
    }
    assert_eq!(matrix, matrix![1, 2, 3, 4; 5, 6, 70, 8; 9, 100, 11, 12]);
}

#[test]
fn row_and_column_views_reborrow_without_crossing_storage() {
    let mut matrix = Matrix::<3, 3, i32>::from_rows([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let row = matrix.row(1);
    assert_eq!([row[0], row[1], row[2]], [4, 5, 6]);
    let column = matrix.column(2);
    assert_eq!([column[0], column[1], column[2]], [3, 6, 9]);

    {
        let row = matrix.row_mut(1);
        row[0] = 40;
        row[2] = 60;
    }
    {
        let column = matrix.column_mut(1);
        column[0] = 20;
        column[2] = 80;
    }

    assert_eq!(matrix, matrix![1, 20, 3; 40, 5, 60; 7, 80, 9]);
}
