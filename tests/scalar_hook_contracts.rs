use core::fmt::Debug;
use core::ops::{Div, Sub};
use stack_algebra::{Matrix, MatrixScalar};
use std::panic::{catch_unwind, AssertUnwindSafe};

trait TestScalar:
    MatrixScalar + From<i16> + Debug + PartialEq + Sub<Output = Self> + Div<Output = Self>
{
}

impl<T> TestScalar for T where
    T: MatrixScalar + From<i16> + Debug + PartialEq + Sub<Output = T> + Div<Output = T>
{
}

fn rejects_without_mutation<T: TestScalar>(original: &[T], operation: impl FnOnce(&mut [T])) {
    let mut actual = original.to_vec();
    assert!(catch_unwind(AssertUnwindSafe(|| operation(&mut actual))).is_err());
    assert_eq!(actual, original, "rejected hooks must preserve output");
}

fn mismatched_slices<T: TestScalar>() {
    for length in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
        for (first_len, second_len) in [(length, length + 1), (length + 1, length)] {
            let first = vec![T::from(2); first_len];
            let second = vec![T::from(3); second_len];
            rejects_without_mutation(&first, |target| {
                T::rank_update_sub(target, &second, T::from(1));
            });
            // Validate both source lengths before touching the target, including
            // the case where the first source is valid but the second is not.
            rejects_without_mutation(&first, |target| {
                T::rank_update_two_sub(target, &second, T::from(1), &first, T::from(1));
            });
            rejects_without_mutation(&first, |target| {
                T::rank_update_two_sub(target, &first, T::from(1), &second, T::from(1));
            });
            let mut left = first.clone();
            let mut right = second.clone();
            assert!(catch_unwind(AssertUnwindSafe(|| {
                T::rotate_columns(&mut left, &mut right, T::from(0), T::from(1));
            }))
            .is_err());
            assert_eq!(left, first);
            assert_eq!(right, second);
            assert!(catch_unwind(AssertUnwindSafe(|| {
                T::dot_accumulate(&first, &second, T::from(7))
            }))
            .is_err());
            assert!(catch_unwind(AssertUnwindSafe(|| T::symmetric_dot(&first, &second))).is_err());
        }
    }
}

fn valid_packet_boundaries<T: TestScalar>() {
    for length in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
        let first = vec![T::from(2); length];
        let second = vec![T::from(3); length];
        let count = T::from(length as i16);
        assert_eq!(
            T::dot_accumulate(&first, &second, T::from(7)),
            T::from(7) + T::from(6) * count
        );
        assert_eq!(
            T::symmetric_dot(&first, &second),
            (T::from(4) * count, T::from(9) * count, T::from(6) * count)
        );
        let mut target = vec![T::from(10); length];
        T::rank_update_sub(&mut target, &first, T::from(3));
        assert_eq!(target, vec![T::from(4); length]);
        target.fill(T::from(10));
        T::rank_update_two_sub(&mut target, &first, T::from(1), &second, T::from(2));
        assert_eq!(target, vec![T::from(2); length]);
        let mut left = first;
        let mut right = second;
        T::rotate_columns(&mut left, &mut right, T::from(0), T::from(1));
        assert_eq!(left, vec![T::from(-3); length]);
        assert_eq!(right, vec![T::from(2); length]);
    }
}

fn invalid_matrix_arguments<T: TestScalar>() {
    let original = Matrix::<8, 8, T>::from_fn(|row, column| {
        T::from(if row == column { 2 } else { 1 })
    });
    for (start, end) in [
        (0, 9),
        (9, 9),
        (4, 3),
        (0, usize::MAX),
        (usize::MAX, 0),
        (usize::MAX, usize::MAX),
    ] {
        let mut actual = original;
        assert!(catch_unwind(AssertUnwindSafe(|| {
            T::symmetric_rank_k_update(&mut actual, start, end);
        }))
        .is_err());
        assert_eq!(actual, original);
    }
    for column in [8, usize::MAX] {
        let mut actual = original;
        assert!(catch_unwind(AssertUnwindSafe(|| {
            T::cholesky_update_column(&mut actual, column, T::from(1));
        }))
        .is_err());
        assert_eq!(actual, original);
    }
    // Valid empty update ranges do not change the matrix.
    for (start, end) in [(0, 0), (4, 4), (8, 8), (0, 8)] {
        let mut actual = original;
        T::symmetric_rank_k_update(&mut actual, start, end);
        assert_eq!(actual, original);
    }
    let mut empty = Matrix::<0, 0, T>::zeros();
    T::symmetric_rank_k_update(&mut empty, 0, 0);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        T::cholesky_update_column(&mut empty, 0, T::from(1));
    }))
    .is_err());
}

macro_rules! scalar_contract_suite {
    ($module:ident, $scalar:ty) => {
        mod $module {
            #[test]
            fn rejects_mismatched_slices_before_mutation() {
                super::mismatched_slices::<$scalar>();
            }

            #[test]
            fn valid_empty_slices_and_packet_tails_preserve_results() {
                super::valid_packet_boundaries::<$scalar>();
            }

            #[test]
            fn rejects_invalid_matrix_arguments_before_mutation() {
                super::invalid_matrix_arguments::<$scalar>();
            }
        }
    };
}

scalar_contract_suite!(f32_hooks, f32);
scalar_contract_suite!(f64_hooks, f64);
// Integer scalars always use the portable trait defaults, even on SIMD hosts.
scalar_contract_suite!(portable_hooks, i32);
