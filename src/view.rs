//! Row and column slices of a matrix.

use core::iter::Sum;
use core::ops::{Deref, DerefMut, Mul};

use stride::Stride;

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
    #[inline]
    pub fn dot_partial<const P: usize>(
        &self,
        other: &Column<N, P, T>,
        range: core::ops::Range<usize>,
    ) -> T
    where
        T: Copy + Mul<Output = T> + Sum,
    {
        // let f = range.start;
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
