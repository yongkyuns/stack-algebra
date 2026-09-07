use stack_algebra::Matrix;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fixture<const M: usize, const N: usize>() -> Matrix<M, N, i32> {
    Matrix::from_fn(|row, column| (row * 17 + column) as i32)
}

fn rejects_without_mutation<const M: usize, const N: usize>(
    original: Matrix<M, N, i32>,
    operation: impl FnOnce(&mut Matrix<M, N, i32>),
    message: &str,
) {
    let mut actual = original;
    let panic = catch_unwind(AssertUnwindSafe(|| operation(&mut actual)))
        .expect_err("invalid swap indices must panic");
    let actual_message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    assert_eq!(actual_message, Some(message));
    assert_eq!(actual, original, "rejected swaps must not mutate storage");
}

#[test]
fn every_valid_row_pair_matches_a_permutation_and_round_trips() {
    let original = fixture::<3, 5>();
    for first in 0..3 {
        for second in 0..3 {
            let mut actual = original;
            actual.swap_rows(first, second);
            let expected = Matrix::<3, 5, i32>::from_fn(|row, column| {
                let source_row = if row == first {
                    second
                } else if row == second {
                    first
                } else {
                    row
                };
                original[(source_row, column)]
            });
            assert_eq!(actual, expected);
            actual.swap_rows(first, second);
            assert_eq!(actual, original);
        }
    }
}

#[test]
fn every_valid_column_pair_matches_a_permutation_and_round_trips() {
    let original = fixture::<3, 5>();
    for first in 0..5 {
        for second in 0..5 {
            let mut actual = original;
            actual.swap_columns(first, second);
            let expected = Matrix::<3, 5, i32>::from_fn(|row, column| {
                let source_column = if column == first {
                    second
                } else if column == second {
                    first
                } else {
                    column
                };
                original[(row, source_column)]
            });
            assert_eq!(actual, expected);
            actual.swap_columns(first, second);
            assert_eq!(actual, original);
        }
    }
}

#[test]
fn invalid_row_indices_in_either_position_panic_before_mutation() {
    for (first, second) in [
        (3, 0),
        (0, 3),
        (3, 3),
        (usize::MAX, 0),
        (0, usize::MAX),
        (usize::MAX, usize::MAX),
    ] {
        rejects_without_mutation(
            fixture::<3, 5>(),
            |matrix| matrix.swap_rows(first, second),
            "row index out of bounds",
        );
    }
}

#[test]
fn invalid_column_indices_in_either_position_panic_before_mutation() {
    for (first, second) in [
        (5, 0),
        (0, 5),
        (5, 5),
        (usize::MAX, 0),
        (0, usize::MAX),
        (usize::MAX, usize::MAX),
    ] {
        rejects_without_mutation(
            fixture::<3, 5>(),
            |matrix| matrix.swap_columns(first, second),
            "column index out of bounds",
        );
    }
}

#[test]
fn row_bounds_are_checked_even_when_there_are_no_columns() {
    let original = fixture::<3, 0>();
    for first in 0..3 {
        for second in 0..3 {
            let mut actual = original;
            actual.swap_rows(first, second);
            assert_eq!(actual, original);
        }
    }
    for (first, second) in [(3, 0), (0, 3), (usize::MAX, usize::MAX)] {
        rejects_without_mutation(
            original,
            |matrix| matrix.swap_rows(first, second),
            "row index out of bounds",
        );
    }
}

#[test]
fn column_bounds_are_checked_even_when_there_are_no_rows() {
    let original = fixture::<0, 5>();
    for first in 0..5 {
        for second in 0..5 {
            let mut actual = original;
            actual.swap_columns(first, second);
            assert_eq!(actual, original);
        }
    }
    for (first, second) in [(5, 0), (0, 5), (usize::MAX, usize::MAX)] {
        rejects_without_mutation(
            original,
            |matrix| matrix.swap_columns(first, second),
            "column index out of bounds",
        );
    }
}

#[test]
fn a_matrix_without_rows_rejects_every_row_index() {
    for (first, second) in [(0, 0), (usize::MAX, 0), (0, usize::MAX)] {
        rejects_without_mutation(
            fixture::<0, 3>(),
            |matrix| matrix.swap_rows(first, second),
            "row index out of bounds",
        );
    }
}

#[test]
fn a_matrix_without_columns_rejects_every_column_index() {
    for (first, second) in [(0, 0), (usize::MAX, 0), (0, usize::MAX)] {
        rejects_without_mutation(
            fixture::<3, 0>(),
            |matrix| matrix.swap_columns(first, second),
            "column index out of bounds",
        );
    }
}

#[test]
fn an_empty_matrix_rejects_both_swap_operations() {
    rejects_without_mutation(
        fixture::<0, 0>(),
        |matrix| matrix.swap_rows(0, 0),
        "row index out of bounds",
    );
    rejects_without_mutation(
        fixture::<0, 0>(),
        |matrix| matrix.swap_columns(0, 0),
        "column index out of bounds",
    );
}

#[test]
fn singleton_self_swaps_preserve_the_value() {
    let original = Matrix::<1, 1, i32>::from_rows([[42]]);
    let mut actual = original;
    actual.swap_rows(0, 0);
    actual.swap_columns(0, 0);
    assert_eq!(actual, original);
    rejects_without_mutation(
        original,
        |matrix| matrix.swap_rows(0, 1),
        "row index out of bounds",
    );
    rejects_without_mutation(
        original,
        |matrix| matrix.swap_columns(1, 0),
        "column index out of bounds",
    );
}
