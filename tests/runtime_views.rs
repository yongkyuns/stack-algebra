//! Runtime layout and operation contracts, with independent numerical references.
use stack_algebra::runtime::{copy_into, matmul_into, transpose_into, Error, MatrixMut, MatrixRef};

#[test]
fn packed_layouts_copy_and_transpose_without_touching_trailing_capacity() {
    let rows = [1_i32, 2, 3, 4, 5, 6];
    let input = MatrixRef::from_row_major(&rows, 3, 2).unwrap();
    let mut columns = [-99; 8];
    copy_into(input, &mut MatrixMut::from_column_major(&mut columns, 3, 2).unwrap()).unwrap();
    assert_eq!(columns, [1, 3, 5, 2, 4, 6, -99, -99]);
    let mut transposed = [0; 6];
    transpose_into(input, &mut MatrixMut::from_row_major(&mut transposed, 2, 3).unwrap()).unwrap();
    assert_eq!(transposed, [1, 3, 5, 2, 4, 6]);
    assert_eq!(input.transpose().transpose().shape(), input.shape());
    assert_eq!(input.transpose().transpose().strides(), input.strides());
}

#[test]
fn shared_views_copy_without_requiring_copy_scalars() {
    let data = [String::from("borrowed")];
    let view = MatrixRef::from_column_major(&data, 1, 1).unwrap();
    let other = view;
    assert!(core::ptr::eq(view.get(0, 0).unwrap(), other.get(0, 0).unwrap()));
}

#[test]
fn transposed_submatrices_preserve_parent_padding_and_unselected_elements() {
    let mut source = [-999_i32; 30];
    for column in 0..5 {
        for row in 0..4 {
            source[column * 6 + row] = (10 * row + column) as i32;
        }
    }
    let source_before = source;
    let input = MatrixRef::from_strides(&source, 4, 5, 1, 6).unwrap();
    let block = input.submatrix(1, 1, 2, 3).unwrap().transpose();
    let mut destination = [-999; 35];
    {
        let mut parent = MatrixMut::from_strides(&mut destination, 5, 4, 7, 1).unwrap();
        let mut output = parent.reborrow().submatrix(1, 1, 3, 2).unwrap();
        copy_into(block, &mut output).unwrap();
        assert_eq!(parent[(1, 1)], 11);
        assert_eq!(parent[(3, 2)], 23);
    }
    for (index, value) in destination.iter().enumerate() {
        let row = index / 7;
        let column = index % 7;
        let expected = if (1..4).contains(&row) && (1..3).contains(&column) {
            (10 * column + row) as i32
        } else {
            -999
        };
        assert_eq!(*value, expected, "index={index}");
    }
    assert_eq!(source, source_before);
}

#[test]
fn mutable_transpose_reborrows_and_read_only_reborrows_preserve_layout() {
    let mut data = [0_i32; 6];
    let mut view = MatrixMut::from_column_major(&mut data, 2, 3).unwrap();
    {
        let mut transposed = view.reborrow().transpose();
        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed.strides(), (2, 1));
        transposed[(2, 1)] = 42;
    }
    assert_eq!(view.as_ref()[(1, 2)], 42);
    assert_eq!(view[(1, 2)], 42);
    assert_eq!(data, [0, 0, 0, 0, 0, 42]);
}

#[test]
fn rectangular_padded_products_match_nalgebra_across_runtime_shapes() {
    for rows in 0..=4 {
        for inner in 0..=4 {
            for columns in 0..=4 {
                let a = nalgebra::DMatrix::<f64>::from_fn(rows, inner, |r, c| {
                    ((r * 7 + c * 3 + 1) as f64).sin()
                });
                let b = nalgebra::DMatrix::<f64>::from_fn(inner, columns, |r, c| {
                    ((r * 2 + c * 5 + 2) as f64).cos()
                });
                let expected = &a * &b;
                let a_stride = rows + 2;
                let b_stride = columns + 3;
                let out_stride = columns + 1;
                let mut a_data = vec![f64::NAN; a_stride * inner];
                let mut b_data = vec![f64::NAN; b_stride * inner];
                let mut output = vec![12345.0; out_stride * rows + 1];
                for r in 0..rows {
                    for c in 0..inner {
                        a_data[c * a_stride + r] = a[(r, c)];
                    }
                }
                for r in 0..inner {
                    for c in 0..columns {
                        b_data[r * b_stride + c] = b[(r, c)];
                    }
                }
                matmul_into(
                    MatrixRef::from_strides(&a_data, rows, inner, 1, a_stride).unwrap(),
                    MatrixRef::from_strides(&b_data, inner, columns, b_stride, 1).unwrap(),
                    &mut MatrixMut::from_strides(&mut output, rows, columns, out_stride, 1).unwrap(),
                ).unwrap();
                for r in 0..rows {
                    for c in 0..columns {
                        let error = (output[r * out_stride + c] - expected[(r, c)]).abs();
                        assert!(error <= 1.0e-12 * (1.0 + expected[(r, c)].abs()));
                    }
                    assert_eq!(output[r * out_stride + columns], 12345.0);
                }
                assert_eq!(*output.last().unwrap(), 12345.0);
            }
        }
    }
}

#[test]
fn f32_shared_inputs_may_alias_each_other() {
    let data = [1.0_f32, -2.0, 3.0, 0.5, 4.0, 2.0];
    let a = MatrixRef::from_row_major(&data, 2, 3).unwrap();
    let mut output = [f32::NAN; 4];
    matmul_into(a, a.transpose(), &mut MatrixMut::from_row_major(&mut output, 2, 2).unwrap()).unwrap();
    assert_eq!(output, [14.0, -1.5, -1.5, 20.25]);
}

#[test]
fn transposed_inputs_and_transposed_output_match_a_scalar_reference() {
    let left = [1_i32, 2, 3, 4, 5, 6];
    let right = [-1_i32, 2, 0, 3, 4, 1];
    let a = MatrixRef::from_column_major(&left, 3, 2).unwrap().transpose();
    let b = MatrixRef::from_row_major(&right, 2, 3).unwrap().transpose();
    let mut data = [-99; 9];
    {
        let parent = MatrixMut::from_strides(&mut data, 3, 2, 1, 5).unwrap();
        let mut output = parent.submatrix(0, 0, 2, 2).unwrap().transpose();
        matmul_into(a, b, &mut output).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                let expected: i32 = (0..3).map(|k| left[r * 3 + k] * right[c * 3 + k]).sum();
                assert_eq!(output[(r, c)], expected);
            }
        }
    }
    for index in [2, 3, 4, 7, 8] {
        assert_eq!(data[index], -99);
    }
}

#[test]
fn repeated_read_only_elements_are_allowed_but_not_mutable_elements() {
    let values = [2_i32, 3];
    let input = MatrixRef::from_strides(&values, 3, 2, 0, 1).unwrap();
    let mut output = [0; 6];
    copy_into(input, &mut MatrixMut::from_column_major(&mut output, 3, 2).unwrap()).unwrap();
    assert_eq!(output, [2, 2, 2, 3, 3, 3]);
    let mut repeated = values;
    let error = MatrixMut::from_strides(&mut repeated, 3, 2, 0, 1).unwrap_err();
    assert_eq!(error, Error::OverlappingElements);
    assert_eq!(repeated, values);
}

#[test]
fn mutable_layout_admission_matches_exhaustive_distinct_offset_oracle() {
    for rows in 0..=4 {
        for columns in 0..=4 {
            for row_stride in 0..=5 {
                for column_stride in 0..=5 {
                    let mut offsets = Vec::new();
                    for row in 0..rows {
                        for column in 0..columns {
                            offsets.push(row * row_stride + column * column_stride);
                        }
                    }
                    offsets.sort_unstable();
                    let unique = !offsets.windows(2).any(|pair| pair[0] == pair[1]);
                    let mut data = [-7_i32; 64];
                    let actual = MatrixMut::from_strides(&mut data, rows, columns, row_stride, column_stride);
                    assert_eq!(actual.is_ok(), unique, "shape={rows}x{columns}, strides={row_stride},{column_stride}");
                    assert_eq!(data, [-7; 64]);
                }
            }
        }
    }
}

#[test]
fn empty_shapes_and_edge_blocks_do_not_calculate_nonexistent_addresses() {
    let empty = MatrixRef::<i32>::from_strides(&[], 0, usize::MAX, usize::MAX, usize::MAX).unwrap();
    assert_eq!(empty.shape(), (0, usize::MAX));
    assert_eq!(empty.submatrix(0, usize::MAX, 0, 0).unwrap().shape(), (0, 0));
    assert!(empty.get(0, 0).is_none());
    let data = [1_i32, 2, 3, 4];
    let view = MatrixRef::from_column_major(&data, 2, 2).unwrap();
    assert_eq!(view.submatrix(2, 2, 0, 0).unwrap().shape(), (0, 0));
    assert_eq!(view.submatrix(2, 0, 0, 2).unwrap().shape(), (0, 2));
    let mut mutable = data;
    let edge = MatrixMut::from_column_major(&mut mutable, 2, 2).unwrap().submatrix(2, 2, 0, 0).unwrap();
    assert_eq!(edge.shape(), (0, 0));
    assert_eq!(mutable, data);
    // Returning immediately for an empty output matters even when its other
    // dimension is enormous. No iteration over that dimension is needed.
    let mut out = [];
    matmul_into(
        MatrixRef::from_column_major(&[], 0, 0).unwrap(),
        empty,
        &mut MatrixMut::from_column_major(&mut out, 0, usize::MAX).unwrap(),
    ).unwrap();
}

#[test]
fn zero_inner_dimension_writes_zeros_only_to_active_output() {
    let mut output = [99_i32; 8];
    matmul_into(
        MatrixRef::from_column_major(&[], 2, 0).unwrap(),
        MatrixRef::from_column_major(&[], 0, 2).unwrap(),
        &mut MatrixMut::from_strides(&mut output, 2, 2, 1, 4).unwrap(),
    ).unwrap();
    assert_eq!(output, [0, 0, 99, 99, 0, 0, 99, 99]);
}

#[test]
fn shape_rejection_leaves_all_output_storage_unchanged() {
    let input = [1_i32; 6];
    let lhs = MatrixRef::from_column_major(&input, 2, 3).unwrap();
    let wrong_rhs = MatrixRef::from_column_major(&input, 2, 2).unwrap();
    let rhs = MatrixRef::from_column_major(&input, 3, 1).unwrap();
    let mut data = [71; 8];
    {
        let mut output = MatrixMut::from_strides(&mut data, 2, 2, 1, 4).unwrap();
        assert!(matches!(matmul_into(lhs, wrong_rhs, &mut output), Err(Error::ShapeMismatch { .. })));
        assert!(matches!(matmul_into(lhs, rhs, &mut output), Err(Error::ShapeMismatch { .. })));
        assert!(matches!(copy_into(lhs, &mut output), Err(Error::ShapeMismatch { .. })));
        assert!(matches!(transpose_into(lhs, &mut output), Err(Error::ShapeMismatch { .. })));
    }
    assert_eq!(data, [71; 8]);
}

#[test]
fn layout_overflow_short_buffers_and_block_bounds_are_checked() {
    assert_eq!(MatrixRef::from_strides(&[0_i32], usize::MAX, 2, 0, 0).unwrap_err(), Error::SizeOverflow);
    assert_eq!(MatrixRef::from_strides(&[0_i32], 2, 1, usize::MAX, 0).unwrap_err(), Error::SizeOverflow);
    assert_eq!(MatrixRef::from_strides(&[0_i32], 3, 1, usize::MAX, 0).unwrap_err(), Error::SizeOverflow);
    assert_eq!(MatrixRef::from_column_major(&[0_i32; 5], 2, 3).unwrap_err(), Error::BufferTooShort { required: 6, available: 5 });
    let data = [0_i32; 6];
    let view = MatrixRef::from_column_major(&data, 2, 3).unwrap();
    assert_eq!(view.submatrix(usize::MAX, 0, 1, 1).unwrap_err(), Error::SizeOverflow);
    assert_eq!(view.submatrix(2, 0, 1, 1).unwrap_err(), Error::BlockOutOfBounds);
    assert!(view.get(usize::MAX, usize::MAX).is_none());
    let mut mutable = [9_i32; 5];
    assert!(matches!(MatrixMut::from_column_major(&mut mutable, 2, 3), Err(Error::BufferTooShort { .. })));
    assert_eq!(mutable, [9; 5]);
    let mut view = MatrixMut::from_column_major(&mut mutable, 1, 5).unwrap();
    assert!(view.get_mut(usize::MAX, usize::MAX).is_none());
    assert_eq!(view.reborrow().submatrix(0, 5, 1, 1).unwrap_err(), Error::BlockOutOfBounds);
    assert_eq!(mutable, [9; 5]);
}

#[test]
fn growing_and_shrinking_active_views_reuse_caller_storage() {
    let mut storage = [11_i32; 20];
    for (rows, columns) in [(2, 3), (1, 1), (4, 4), (0, 4), (3, 2)] {
        let mut view = MatrixMut::from_strides(&mut storage, rows, columns, 1, 5).unwrap();
        for c in 0..columns {
            for r in 0..rows {
                view[(r, c)] = (10 * c + r) as i32;
            }
        }
        assert_eq!(view.as_ref().shape(), (rows, columns));
    }
    for column in 0..4 {
        assert_eq!(storage[column * 5 + 4], 11);
    }
}

#[test]
fn nonfinite_values_follow_scalar_arithmetic_not_admission_errors() {
    let a = [f64::NAN];
    let b = [1.0_f64];
    let mut output = [0.0];
    matmul_into(
        MatrixRef::from_column_major(&a, 1, 1).unwrap(),
        MatrixRef::from_column_major(&b, 1, 1).unwrap(),
        &mut MatrixMut::from_column_major(&mut output, 1, 1).unwrap(),
    ).unwrap();
    assert!(output[0].is_nan());
}
