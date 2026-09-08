use core::fmt::Debug;
use core::mem::MaybeUninit;
use stack_algebra::{StaticCscMatrix, StaticCscOrdering, StaticCscPermutation, Zero};
use std::panic::{catch_unwind, AssertUnwindSafe};

type Sparse = StaticCscMatrix<3, 3, 9, f64>;

fn source() -> Sparse {
    Sparse::from_pattern(
        &[10.0, 1.0, 20.0, 2.0, 30.0],
        &[0, 1, 1, 2, 2],
        &[0, 2, 4, 5],
    )
    .unwrap()
}

fn ordering() -> StaticCscOrdering<3> {
    StaticCscOrdering::from_permutation(&[2, 0, 1]).unwrap()
}

fn expected() -> Sparse {
    Sparse::from_pattern(
        &[30.0, 2.0, 10.0, 1.0, 20.0],
        &[0, 2, 1, 2, 2],
        &[0, 2, 4, 5],
    )
    .unwrap()
}

fn assert_active<const N: usize, const CAP: usize, T>(
    actual: &StaticCscMatrix<N, N, CAP, T>,
    expected: &StaticCscMatrix<N, N, CAP, T>,
) where
    T: Copy + Zero + PartialEq + Debug,
{
    assert_eq!(actual.pattern(), expected.pattern());
    assert_eq!(actual.values(), expected.values());
    for column in 0..N {
        for row in 0..N {
            assert_eq!(actual.get(row, column), expected.get(row, column));
        }
    }
}

#[test]
fn returning_permutation_matches_hand_written_coordinates() {
    let input = source();
    let map = ordering().permutation_for_pattern(input.pattern()).unwrap();
    assert_active(&map.apply(&input), &expected());
}

#[test]
fn apply_into_accepts_empty_destination() {
    let input = source();
    let map = ordering().permutation_for_pattern(input.pattern()).unwrap();
    let mut output = Sparse::new();
    map.apply_into(&input, &mut output);
    assert_active(&output, &expected());
}

#[test]
fn apply_into_replaces_equal_length_destination_pattern() {
    let input = source();
    let map = ordering().permutation_for_pattern(input.pattern()).unwrap();
    let mut output = source();
    output.set_values(&[-1.0; 5]).unwrap();
    assert_eq!(output.nnz(), map.pattern_ref().nnz());
    assert_ne!(output.pattern(), map.pattern_ref());
    map.apply_into(&input, &mut output);
    assert_active(&output, &expected());
}

#[test]
fn apply_into_replaces_larger_destination_pattern() {
    let input = source();
    let map = ordering().permutation_for_pattern(input.pattern()).unwrap();
    let mut output = Sparse::new();
    for column in 0..3 {
        for row in 0..3 {
            output.insert(row, column, -100.0).unwrap();
        }
    }
    map.apply_into(&input, &mut output);
    assert_active(&output, &expected());
}

#[test]
fn repeated_maps_can_grow_and_shrink_destination() {
    let input = source();
    let diagonal = Sparse::from_pattern(&[4.0, 5.0, 6.0], &[0, 1, 2], &[0, 1, 2, 3]).unwrap();
    let diagonal_map = ordering()
        .permutation_for_pattern(diagonal.pattern())
        .unwrap();
    let map = ordering().permutation_for_pattern(input.pattern()).unwrap();
    let mut output = diagonal_map.apply(&diagonal);
    map.apply_into(&input, &mut output);
    assert_active(&output, &expected());
    diagonal_map.apply_into(&diagonal, &mut output);
    let diagonal_expected =
        Sparse::from_pattern(&[6.0, 4.0, 5.0], &[0, 1, 2], &[0, 1, 2, 3]).unwrap();
    assert_active(&output, &diagonal_expected);
    map.apply_into(&input, &mut output);
    assert_active(&output, &expected());
}

#[test]
fn empty_maps_reset_previously_populated_destination() {
    let empty = Sparse::new();
    let analyzed = ordering().permutation_for_pattern(empty.pattern()).unwrap();
    for map in [analyzed, StaticCscPermutation::default()] {
        let mut output = source();
        map.apply_into(&empty, &mut output);
        assert_active(&output, &empty);
    }
}

#[test]
fn zero_dimensional_permutations_remain_valid() {
    let input = StaticCscMatrix::<0, 0, 4, f64>::new();
    let map = StaticCscOrdering::identity()
        .permutation_for_pattern(input.pattern())
        .unwrap();
    let mut output = input;
    map.apply_into(&input, &mut output);
    assert_active(&output, &input);
    let mut storage = MaybeUninit::<StaticCscPermutation<0, 0>>::uninit();
    StaticCscPermutation::new_into(&mut storage);
    // SAFETY: new_into initializes the entire empty permutation workspace.
    let empty_map = unsafe { storage.assume_init() };
    let empty = StaticCscMatrix::<0, 0, 0, i32>::new();
    assert_eq!(empty_map.apply(&empty), empty);
}

#[test]
fn matching_pattern_numeric_updates_preserve_reuse() {
    let mut input = source();
    let map = ordering().permutation_for_pattern(input.pattern()).unwrap();
    let mut output = Sparse::zero_with_pattern(map.pattern());
    for scale in [1.0, 2.0, -3.0] {
        input
            .set_values(&[10.0 * scale, scale, 20.0 * scale, 2.0 * scale, 30.0 * scale])
            .unwrap();
        map.apply_into(&input, &mut output);
        let mut result = expected();
        for value in result.values_mut() {
            *value *= scale;
        }
        assert_active(&output, &result);
    }
}

#[test]
fn full_symmetric_input_uses_cached_source_offsets() {
    let input = StaticCscMatrix::<2, 2, 4, f64>::from_pattern(
        &[4.0, 1.0, 1.0, 3.0],
        &[0, 1, 0, 1],
        &[0, 2, 4],
    )
    .unwrap();
    let map = StaticCscOrdering::identity()
        .permutation_for_pattern(input.pattern())
        .unwrap();
    let mut output = StaticCscMatrix::zero_with_pattern(map.pattern());
    map.apply_into(&input, &mut output);
    let result = StaticCscMatrix::from_pattern(&[4.0, 1.0, 3.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    assert_active(&output, &result);
}

#[test]
fn short_input_is_rejected_before_destination_changes() {
    let input = source();
    let map = StaticCscOrdering::identity()
        .permutation_for_pattern(input.pattern())
        .unwrap();
    let short = Sparse::from_pattern(&[8.0, 7.0, 6.0, 5.0], &[0, 1, 1, 2], &[0, 2, 4, 4]).unwrap();
    for replacement in [short, Sparse::new()] {
        let mut output = input;
        output.set_values(&[-1.0; 5]).unwrap();
        let before = output;
        let result = catch_unwind(AssertUnwindSafe(|| {
            map.apply_into(&replacement, &mut output);
        }));
        assert!(result.is_err());
        assert_eq!(output, before);
    }
}

#[test]
fn full_storage_offsets_are_checked_before_destination_changes() {
    let full = StaticCscMatrix::<2, 2, 4, f64>::from_pattern(
        &[4.0, 1.0, 1.0, 3.0],
        &[0, 1, 0, 1],
        &[0, 2, 4],
    )
    .unwrap();
    let map = StaticCscOrdering::identity()
        .permutation_for_pattern(full.pattern())
        .unwrap();
    let lower = StaticCscMatrix::from_pattern(&[4.0, 1.0, 3.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let mut output = StaticCscMatrix::zero_with_pattern(map.pattern());
    output.set_values(&[-1.0; 3]).unwrap();
    let before = output;
    let result = catch_unwind(AssertUnwindSafe(|| map.apply_into(&lower, &mut output)));
    assert!(result.is_err());
    assert_eq!(output, before);
}

#[test]
fn apply_into_supports_f32_and_integer_values() {
    let order = StaticCscOrdering::from_permutation(&[1, 0]).unwrap();
    let input =
        StaticCscMatrix::<2, 2, 3, f32>::from_pattern(&[4.0, 1.0, 3.0], &[0, 1, 1], &[0, 2, 3])
            .unwrap();
    let map = order.permutation_for_pattern(input.pattern()).unwrap();
    let mut output = StaticCscMatrix::new();
    map.apply_into(&input, &mut output);
    let result = StaticCscMatrix::from_pattern(&[3.0, 1.0, 4.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
    assert_active(&output, &result);

    let integers =
        StaticCscMatrix::<2, 2, 3, i32>::from_pattern(&[4, 1, 3], &[0, 1, 1], &[0, 2, 3]).unwrap();
    let mut integer_output = StaticCscMatrix::new();
    map.apply_into(&integers, &mut integer_output);
    let result = StaticCscMatrix::from_pattern(&[3, 1, 4], &[0, 1, 1], &[0, 2, 3]).unwrap();
    assert_active(&integer_output, &result);
}

fn small_matrix(mask: usize) -> StaticCscMatrix<2, 2, 4, i32> {
    let mut matrix = StaticCscMatrix::new();
    for column in 0..2 {
        for row in 0..2 {
            let index = column * 2 + row;
            if mask & (1 << index) != 0 {
                matrix.insert(row, column, index as i32 + 10).unwrap();
            }
        }
    }
    matrix
}

#[test]
fn exhaustive_2x2_destinations_match_coordinate_reference() {
    let mut cases = 0;
    for permutation in [[0, 1], [1, 0]] {
        let order = StaticCscOrdering::from_permutation(&permutation).unwrap();
        for source_mask in 0..16 {
            let input = small_matrix(source_mask);
            let map = order.permutation_for_pattern(input.pattern()).unwrap();
            let mut result = StaticCscMatrix::new();
            for column in 0..2 {
                for row in column..2 {
                    if let Some(&value) = input.get(row, column) {
                        let first = order.inverse()[row];
                        let second = order.inverse()[column];
                        result
                            .insert(first.max(second), first.min(second), value)
                            .unwrap();
                    }
                }
            }
            for destination_mask in 0..16 {
                let mut output = small_matrix(destination_mask);
                map.apply_into(&input, &mut output);
                assert_active(&output, &result);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 512);
}
