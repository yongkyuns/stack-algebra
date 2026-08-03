//! Row and column slices of a matrix.

use core::iter::Sum;
use core::ops::{Deref, DerefMut, Index, IndexMut, Mul};

use crate::Matrix;
use stride::Stride;

/// A fixed-size block view into a matrix.
pub struct Block<'a, const M: usize, const N: usize, const R: usize, const C: usize, T> {
    matrix: &'a Matrix<M, N, T>,
    row_offset: usize,
    column_offset: usize,
}

impl<'a, const M: usize, const N: usize, const R: usize, const C: usize, T>
    Block<'a, M, N, R, C, T>
{
    pub(crate) fn new(
        matrix: &'a Matrix<M, N, T>,
        row_offset: usize,
        column_offset: usize,
    ) -> Self {
        Self {
            matrix,
            row_offset,
            column_offset,
        }
    }

    /// Returns the element at block-local coordinates.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        if row < R && column < C {
            Some(&self.matrix[(self.row_offset + row, self.column_offset + column)])
        } else {
            None
        }
    }

    /// Copies this view into an owning fixed-size matrix.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<R, C, T>
    where
        T: Copy,
    {
        Matrix::from_fn(|row, column| *self.get(row, column).expect("block index is in bounds"))
    }
}

impl<const M: usize, const N: usize, const R: usize, const C: usize, T> Index<(usize, usize)>
    for Block<'_, M, N, R, C, T>
{
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("block index is out of bounds")
    }
}

/// A mutable fixed-size block view into a matrix.
pub struct BlockMut<'a, const M: usize, const N: usize, const R: usize, const C: usize, T> {
    matrix: &'a mut Matrix<M, N, T>,
    row_offset: usize,
    column_offset: usize,
}

impl<'a, const M: usize, const N: usize, const R: usize, const C: usize, T>
    BlockMut<'a, M, N, R, C, T>
{
    pub(crate) fn new(
        matrix: &'a mut Matrix<M, N, T>,
        row_offset: usize,
        column_offset: usize,
    ) -> Self {
        Self {
            matrix,
            row_offset,
            column_offset,
        }
    }

    /// Returns the element at block-local coordinates.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        if row < R && column < C {
            Some(&self.matrix[(self.row_offset + row, self.column_offset + column)])
        } else {
            None
        }
    }

    /// Returns the mutable element at block-local coordinates.
    #[inline]
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        if row < R && column < C {
            Some(&mut self.matrix[(self.row_offset + row, self.column_offset + column)])
        } else {
            None
        }
    }

    /// Copies this view into an owning fixed-size matrix.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<R, C, T>
    where
        T: Copy,
    {
        Matrix::from_fn(|row, column| *self.get(row, column).expect("block index is in bounds"))
    }
}

impl<const M: usize, const N: usize, const R: usize, const C: usize, T> Index<(usize, usize)>
    for BlockMut<'_, M, N, R, C, T>
{
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("block index is out of bounds")
    }
}

impl<const M: usize, const N: usize, const R: usize, const C: usize, T> IndexMut<(usize, usize)>
    for BlockMut<'_, M, N, R, C, T>
{
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1)
            .expect("block index is out of bounds")
    }
}

////////////////////////////////////////////////////////////////////////////////
// Row
////////////////////////////////////////////////////////////////////////////////

/// A row in a [`Matrix`][crate::Matrix].
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Row<const M: usize, const N: usize, T> {
    data: Stride<T, M>,
}

impl<T, const M: usize, const N: usize> Row<M, N, T> {
    pub(crate) fn new(data: &[T]) -> &Self {
        // SAFETY: `Row` and `Stride` are both repr(transparent)
        unsafe { &*(data as *const [T] as *const Self) }
    }

    pub(crate) fn new_mut(data: &mut [T]) -> &mut Self {
        // SAFETY: `Row` and `Stride` are both repr(transparent)
        unsafe { &mut *(data as *mut [T] as *mut Self) }
    }
}

impl<T, const M: usize, const N: usize> Deref for Row<M, N, T> {
    type Target = Stride<T, M>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, const M: usize, const N: usize> DerefMut for Row<M, N, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T, U, const M: usize, const N: usize, const S: usize> PartialEq<Stride<U, S>> for Row<M, N, T>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &Stride<U, S>) -> bool {
        self.data.eq(other)
    }
}

impl<T, U, const M: usize, const N: usize> PartialEq<[U]> for Row<M, N, T>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U]) -> bool {
        self.data.eq(other)
    }
}

impl<T, U, const M: usize, const N: usize, const P: usize> PartialEq<[U; P]> for Row<M, N, T>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U; P]) -> bool {
        self.data.eq(other)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Column
////////////////////////////////////////////////////////////////////////////////

/// A column in a [`Matrix`][crate::Matrix].
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Column<const M: usize, const N: usize, T> {
    data: Stride<T, 1>,
}

impl<T, const M: usize, const N: usize> Column<M, N, T> {
    pub(crate) fn new(data: &[T]) -> &Self {
        // SAFETY: `Column` and `Stride` are both repr(transparent)
        unsafe { &*(data as *const [T] as *const Self) }
    }

    pub(crate) fn new_mut(data: &mut [T]) -> &mut Self {
        // SAFETY: `Column` and `Stride` are both repr(transparent)
        unsafe { &mut *(data as *mut [T] as *mut Self) }
    }
}

impl<T, const M: usize, const N: usize> Deref for Column<M, N, T> {
    type Target = Stride<T, 1>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, const M: usize, const N: usize> DerefMut for Column<M, N, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T, U, const M: usize, const N: usize, const S: usize> PartialEq<Stride<U, S>>
    for Column<M, N, T>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &Stride<U, S>) -> bool {
        self.data.eq(other)
    }
}

impl<T, U, const M: usize, const N: usize> PartialEq<[U]> for Column<M, N, T>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U]) -> bool {
        self.data.eq(other)
    }
}

impl<T, U, const M: usize, const N: usize, const P: usize> PartialEq<[U; P]> for Column<M, N, T>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U; P]) -> bool {
        self.data.eq(other)
    }
}

////////////////////////////////////////////////////////////////////////////////
// General
////////////////////////////////////////////////////////////////////////////////

impl<T, const M: usize, const N: usize> Row<M, N, T> {
    #[inline]
    pub fn dot<const P: usize>(&self, other: &Column<N, P, T>) -> T
    where
        T: Copy + Mul<Output = T> + Sum,
    {
        (0..N).map(|i| self[i] * other[i]).sum()
    }

    /// Compute the dot product, but only with elements specified by the range
    #[inline]
    pub fn dot_partial<const P: usize>(
        &self,
        other: &Column<N, P, T>,
        range: core::ops::Range<usize>,
    ) -> T
    where
        T: Copy + Mul<Output = T> + Sum,
    {
        (0..N)
            .skip(range.start)
            .take(range.count())
            .map(|i| self[i] * other[i])
            .sum()
    }
}

#[test]
fn iter() {
    use super::*;
    let m = matrix![
        1.0, 2.0, 3.0, 4.0;
        5.0, 6.0, 7.0, 8.0;
    ];
    let mut r = m.row(1).get(1..3).unwrap().iter();
    assert_eq!(r.next(), Some(&6.0));
    assert_eq!(r.next(), Some(&7.0));
    assert_eq!(r.next(), None);

    let mut c = m.column(2).get(0..2).unwrap().iter();
    assert_eq!(c.next(), Some(&3.0));
    assert_eq!(c.next(), Some(&7.0));
    assert_eq!(c.next(), None);
}

#[test]
fn dot_partial() {
    use super::*;
    let m = matrix![
         1.0,  2.0,  3.0,  4.0;
         5.0,  6.0,  7.0,  8.0;
         9.0, 10.0, 12.0, 13.0;
        14.0, 15.0, 16.0, 17.0;
    ];
    let d = m.row(1).dot_partial(m.column(2), 1..3);
    assert_eq!(d, 126.0);
}

#[test]
fn block_views_read_and_write_column_major_storage() {
    use super::*;
    let mut matrix = matrix![
        1, 2, 3, 4;
        5, 6, 7, 8;
        9, 10, 11, 12;
    ];
    let block = matrix.block::<2, 2>(1, 1).expect("block is in bounds");
    assert_eq!(
        block.to_matrix(),
        Matrix::<2, 2, i32>::from_rows([[6, 7], [10, 11]])
    );
    assert!(matrix.block::<2, 2>(2, 3).is_none());

    let mut block = matrix.block_mut::<2, 2>(1, 1).expect("block is in bounds");
    block[(0, 1)] = 70;
    *block.get_mut(1, 0).expect("block index is in bounds") = 100;
    assert_eq!(matrix[(1, 2)], 70);
    assert_eq!(matrix[(2, 1)], 100);
}
