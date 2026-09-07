use stack_algebra::Matrix;
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[test]
fn shared_slices_are_column_major_and_borrow_storage() {
    let matrix = Matrix::from_columns([[10, 11], [20, 21], [30, 31]]);
    let slice = matrix.as_slice();
    assert_eq!(slice, &[10, 11, 20, 21, 30, 31]);
    for column in 0..3 {
        for row in 0..2 {
            assert!(core::ptr::eq(
                &slice[column * 2 + row],
                &matrix[(row, column)]
            ));
        }
    }
}

#[test]
fn mutable_slices_update_matrix_and_iterators() {
    let mut matrix = Matrix::<2, 3, i32>::from_columns([[0; 2]; 3]);
    for (index, value) in matrix.as_mut_slice().iter_mut().enumerate() {
        *value = index as i32 + 1;
    }
    assert_eq!(matrix[(0, 0)], 1);
    assert_eq!(matrix[(1, 0)], 2);
    assert_eq!(matrix[(0, 1)], 3);
    assert_eq!(matrix[(1, 2)], 6);
    matrix[(1, 2)] = 99;
    assert_eq!(matrix.as_slice(), &[1, 2, 3, 4, 5, 99]);
    assert!(matrix.iter().copied().eq([1, 2, 3, 4, 5, 99]));
    for value in matrix.iter_mut() {
        *value += 1;
    }
    assert_eq!(matrix.as_slice(), &[2, 3, 4, 5, 6, 100]);
}

#[test]
fn slices_support_non_copy_values() {
    let mut matrix = Matrix::<2, 2, String>::from_fn(|row, column| format!("r{row}c{column}"));
    assert_eq!(matrix.as_slice()[1], "r1c0");
    matrix.as_mut_slice()[1].push('!');
    assert_eq!(matrix[(1, 0)], "r1c0!");
    let (first, second) = matrix.as_mut_slice().split_at_mut(2);
    core::mem::swap(&mut first[0], &mut second[1]);
    assert_eq!(matrix[(0, 0)], "r1c1");
    assert_eq!(matrix[(1, 1)], "r0c0");
}

fn check_empty_slices<const M: usize, const N: usize>() {
    let mut matrix = Matrix::<M, N, i32>::from_fn(|_, _| {
        panic!("an empty matrix must not construct an element")
    });
    assert!(matrix.as_slice().is_empty());
    assert!(matrix.as_mut_slice().is_empty());
    assert_eq!(matrix.iter().count(), 0);
    assert_eq!(matrix.iter_mut().count(), 0);
}

#[test]
fn empty_shapes_have_empty_slices_and_iterators() {
    check_empty_slices::<0, 0>();
    check_empty_slices::<0, 3>();
    check_empty_slices::<3, 0>();
}

#[repr(align(64))]
struct AlignedZst;

#[test]
fn zero_sized_values_keep_logical_length_and_alignment() {
    let mut matrix = Matrix::<2, 3, AlignedZst>::from_fn(|_, _| AlignedZst);
    assert_eq!(core::mem::size_of::<AlignedZst>(), 0);
    assert_eq!(matrix.as_slice().len(), 6);
    assert_eq!(core::mem::align_of_val(&matrix.as_slice()[0]), 64);
    assert_eq!(matrix.iter().count(), 6);
    let (first, second) = matrix.as_mut_slice().split_at_mut(2);
    assert_eq!(first.len(), 2);
    assert_eq!(second.len(), 4);
    first[1] = AlignedZst;
    second[3] = AlignedZst;
    assert_eq!(matrix.iter_mut().count(), 6);
}

#[test]
#[should_panic]
fn shared_slices_reject_overflowing_zero_sized_lengths() {
    let matrix = Matrix::<{ usize::MAX }, 2, ()>::from_columns([[(); usize::MAX]; 2]);
    let _ = matrix.as_slice();
}

#[test]
#[should_panic]
fn mutable_slices_reject_overflowing_zero_sized_lengths() {
    let mut matrix = Matrix::<{ usize::MAX }, 2, ()>::from_columns([[(); usize::MAX]; 2]);
    let _ = matrix.as_mut_slice();
}

struct Tracked<'a> {
    id: usize,
    drops: &'a [Cell<usize>; 6],
}

impl Drop for Tracked<'_> {
    fn drop(&mut self) {
        self.drops[self.id].set(self.drops[self.id].get() + 1);
    }
}

fn check_drops(drops: &[Cell<usize>; 6], expected: [usize; 6]) {
    for (actual, expected) in drops.iter().zip(expected) {
        assert_eq!(actual.get(), expected);
    }
}

#[test]
fn collecting_non_copy_values_preserves_order_and_drops_once() {
    let drops = core::array::from_fn(|_| Cell::new(0));
    let mut matrix: Matrix<2, 3, Tracked<'_>> =
        (0..6).map(|id| Tracked { id, drops: &drops }).collect();
    for (index, value) in matrix.as_slice().iter().enumerate() {
        assert_eq!(value.id, index);
    }
    assert_eq!(matrix[(1, 2)].id, 5);
    matrix.as_mut_slice().swap(0, 5);
    assert_eq!(matrix[(0, 0)].id, 5);
    assert_eq!(matrix[(1, 2)].id, 0);
    check_drops(&drops, [0; 6]);
    drop(matrix);
    check_drops(&drops, [1; 6]);
}

#[test]
fn short_iterators_drop_the_initialized_prefix() {
    let drops = core::array::from_fn(|_| Cell::new(0));
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _: Matrix<2, 3, Tracked<'_>> = (0..3).map(|id| Tracked { id, drops: &drops }).collect();
    }));
    assert!(result.is_err());
    check_drops(&drops, [1, 1, 1, 0, 0, 0]);
}

#[test]
fn panicking_iterators_drop_the_initialized_prefix() {
    let drops = core::array::from_fn(|_| Cell::new(0));
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _: Matrix<2, 3, Tracked<'_>> = (0..6)
            .map(|id| {
                assert!(id < 3, "iterator panic after three values");
                Tracked { id, drops: &drops }
            })
            .collect();
    }));
    assert!(result.is_err());
    check_drops(&drops, [1, 1, 1, 0, 0, 0]);
}

fn check_empty_collection<const M: usize, const N: usize>() {
    let mut matrix: Matrix<M, N, String> = core::iter::from_fn(|| -> Option<String> {
        panic!("an empty matrix must not poll its iterator")
    })
    .collect();
    assert!(matrix.as_slice().is_empty());
    assert!(matrix.as_mut_slice().is_empty());
}

#[test]
fn collecting_empty_shapes_never_polls_the_iterator() {
    check_empty_collection::<0, 0>();
    check_empty_collection::<0, 3>();
    check_empty_collection::<3, 0>();
}

#[test]
fn collecting_zero_sized_values_consumes_exactly_the_matrix_length() {
    let mut iter = (0..7).map(|_| AlignedZst);
    let matrix: Matrix<2, 3, AlignedZst> = iter.by_ref().collect();
    assert_eq!(matrix.as_slice().len(), 6);
    assert!(iter.next().is_some());
    assert!(iter.next().is_none());
}
