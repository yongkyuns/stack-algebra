use stack_algebra::{Matrix, StridedMap};

fn assert_close<const M: usize, const N: usize>(
    actual: &Matrix<M, N, f64>,
    expected: &Matrix<M, N, f64>,
) {
    for column in 0..N {
        for row in 0..M {
            let error = (actual[(row, column)] - expected[(row, column)]).abs();
            assert!(error <= 1.0e-12, "({row}, {column}) error = {error}");
        }
    }
}

#[test]
fn padded_column_major_mul_matches_owned_product() {
    const D: usize = 4;
    const LEADING: usize = 6;

    let lhs_owned = Matrix::<D, D, f64>::from_fn(|row, column| (1 + row + 3 * column) as f64);
    let rhs_owned = Matrix::<D, D, f64>::from_fn(|row, column| (2 + 2 * row + column) as f64);

    let mut lhs_storage = [0.0_f64; LEADING * D];
    let mut rhs_storage = [0.0_f64; LEADING * D];
    for column in 0..D {
        for row in 0..D {
            lhs_storage[row + column * LEADING] = lhs_owned[(row, column)];
            rhs_storage[row + column * LEADING] = rhs_owned[(row, column)];
        }
    }

    let lhs = StridedMap::<D, D, f64>::from_slice(&lhs_storage, 1, LEADING).unwrap();
    let rhs = StridedMap::<D, D, f64>::from_slice(&rhs_storage, 1, LEADING).unwrap();
    let mut output = Matrix::<D, D, f64>::zeros();
    lhs.mul_leading_dimension_into(&rhs, &mut output);

    assert_close(&output, &(lhs_owned * rhs_owned));
}

#[test]
fn exact_column_major_layout_still_matches_owned_product() {
    const D: usize = 4;

    let lhs_owned = Matrix::<D, D, f64>::from_fn(|row, column| (1 + row + column) as f64);
    let rhs_owned = Matrix::<D, D, f64>::from_fn(|row, column| (2 + row + 2 * column) as f64);
    let lhs = StridedMap::<D, D, f64>::from_slice(lhs_owned.as_slice(), 1, D).unwrap();
    let rhs = StridedMap::<D, D, f64>::from_slice(rhs_owned.as_slice(), 1, D).unwrap();
    let mut output = Matrix::<D, D, f64>::zeros();
    lhs.mul_leading_dimension_into(&rhs, &mut output);

    assert_close(&output, &(lhs_owned * rhs_owned));
}

#[test]
fn arbitrary_inner_stride_falls_back_without_semantic_change() {
    const D: usize = 3;
    const INNER: usize = 2;
    const OUTER: usize = 7;
    const STORAGE: usize = OUTER * D;

    let lhs_owned = Matrix::<D, D, f64>::from_fn(|row, column| (1 + row + column) as f64);
    let rhs_owned = Matrix::<D, D, f64>::from_fn(|row, column| (3 + row + 2 * column) as f64);

    let mut lhs_storage = [0.0_f64; STORAGE];
    let mut rhs_storage = [0.0_f64; STORAGE];
    for column in 0..D {
        for row in 0..D {
            lhs_storage[row * INNER + column * OUTER] = lhs_owned[(row, column)];
            rhs_storage[row * INNER + column * OUTER] = rhs_owned[(row, column)];
        }
    }

    let lhs = StridedMap::<D, D, f64>::from_slice(&lhs_storage, INNER, OUTER).unwrap();
    let rhs = StridedMap::<D, D, f64>::from_slice(&rhs_storage, INNER, OUTER).unwrap();
    let mut output = Matrix::<D, D, f64>::zeros();
    lhs.mul_leading_dimension_into(&rhs, &mut output);

    assert_close(&output, &(lhs_owned * rhs_owned));
}
