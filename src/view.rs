//! Row and column slices of a matrix.

use core::iter::Sum;
use core::ops::{Deref, DerefMut, Index, IndexMut, Mul};

use crate::Matrix;
use stride::Stride;

/// Read-only access to a fixed-size matrix-shaped view.
pub trait MatrixRead<const M: usize, const N: usize, T> {
    /// Returns the element at `(row, column)`, or `None` when out of bounds.
    fn get(&self, row: usize, column: usize) -> Option<&T>;
}

/// Mutable access to a fixed-size matrix-shaped view.
pub trait MatrixWrite<const M: usize, const N: usize, T>: MatrixRead<M, N, T> {
    /// Returns the mutable element at `(row, column)`, or `None` when out of bounds.
    fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T>;
}

impl<const M: usize, const N: usize, T> MatrixRead<M, N, T> for Matrix<M, N, T> {
    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<&T> {
        if row < M && column < N {
            Some(&self[(row, column)])
        } else {
            None
        }
    }
}

impl<const M: usize, const N: usize, T> MatrixWrite<M, N, T> for Matrix<M, N, T> {
    #[inline]
    fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        if row < M && column < N {
            Some(&mut self[(row, column)])
        } else {
            None
        }
    }
}

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

impl<const M: usize, const N: usize, const R: usize, const C: usize, T> MatrixRead<R, C, T>
    for Block<'_, M, N, R, C, T>
{
    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<&T> {
        Block::get(self, row, column)
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

impl<const M: usize, const N: usize, const R: usize, const C: usize, T> MatrixRead<R, C, T>
    for BlockMut<'_, M, N, R, C, T>
{
    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<&T> {
        BlockMut::get(self, row, column)
    }
}

impl<const M: usize, const N: usize, const R: usize, const C: usize, T> MatrixWrite<R, C, T>
    for BlockMut<'_, M, N, R, C, T>
{
    #[inline]
    fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        BlockMut::get_mut(self, row, column)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Map
////////////////////////////////////////////////////////////////////////////////

/// A fixed-size, zero-copy view over an external column-major buffer.
pub struct Map<'a, const M: usize, const N: usize, T> {
    data: &'a [T],
}

impl<'a, const M: usize, const N: usize, T> Map<'a, M, N, T> {
    /// Maps the first `M * N` elements of `data`, or returns `None` when it is
    /// too short.
    #[inline]
    pub fn from_slice(data: &'a [T]) -> Option<Self> {
        Some(Self {
            data: data.get(..M * N)?,
        })
    }

    /// Returns the mapped column-major storage.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Returns the element at `(row, column)`, or `None` when out of bounds.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        if row < M && column < N {
            Some(&self.data[row + M * column])
        } else {
            None
        }
    }

    /// Copies the mapped data into an owning matrix.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<M, N, T>
    where
        T: Copy,
    {
        Matrix::from_columns(core::array::from_fn(|column| {
            core::array::from_fn(|row| self.data[row + M * column])
        }))
    }
}

impl<const M: usize, const N: usize, T> Index<usize> for Map<'_, M, N, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const M: usize, const N: usize, T> Index<(usize, usize)> for Map<'_, M, N, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("mapped matrix index is out of bounds")
    }
}

impl<const M: usize, const N: usize, T> MatrixRead<M, N, T> for Map<'_, M, N, T> {
    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<&T> {
        Map::get(self, row, column)
    }
}

/// A mutable fixed-size, zero-copy view over an external column-major buffer.
pub struct MapMut<'a, const M: usize, const N: usize, T> {
    data: &'a mut [T],
}

impl<'a, const M: usize, const N: usize, T> MapMut<'a, M, N, T> {
    /// Maps the first `M * N` elements of `data`, or returns `None` when it is
    /// too short.
    #[inline]
    pub fn from_slice(data: &'a mut [T]) -> Option<Self> {
        Some(Self {
            data: data.get_mut(..M * N)?,
        })
    }

    /// Returns the mapped column-major storage.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Returns the mapped column-major storage mutably.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// Reborrows the mutable mapping as an immutable mapping.
    #[inline]
    pub fn as_map(&self) -> Map<'_, M, N, T> {
        Map { data: self.data }
    }

    /// Returns the element at `(row, column)`, or `None` when out of bounds.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        if row < M && column < N {
            Some(&self.data[row + M * column])
        } else {
            None
        }
    }

    /// Returns the mutable element at `(row, column)`, or `None` when out of
    /// bounds.
    #[inline]
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        if row < M && column < N {
            Some(&mut self.data[row + M * column])
        } else {
            None
        }
    }

    /// Copies the mapped data into an owning matrix.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<M, N, T>
    where
        T: Copy,
    {
        self.as_map().to_matrix()
    }
}

impl<const M: usize, const N: usize, T> Index<usize> for MapMut<'_, M, N, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const M: usize, const N: usize, T> IndexMut<usize> for MapMut<'_, M, N, T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const M: usize, const N: usize, T> Index<(usize, usize)> for MapMut<'_, M, N, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("mapped matrix index is out of bounds")
    }
}

impl<const M: usize, const N: usize, T> IndexMut<(usize, usize)> for MapMut<'_, M, N, T> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1)
            .expect("mapped matrix index is out of bounds")
    }
}

impl<const M: usize, const N: usize, T> MatrixRead<M, N, T> for MapMut<'_, M, N, T> {
    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<&T> {
        MapMut::get(self, row, column)
    }
}

impl<const M: usize, const N: usize, T> MatrixWrite<M, N, T> for MapMut<'_, M, N, T> {
    #[inline]
    fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        MapMut::get_mut(self, row, column)
    }
}

/// A fixed-size, zero-copy view with runtime inner and outer strides.
///
/// `inner_stride` advances one row and `outer_stride` advances one column.
pub struct StridedMap<'a, const M: usize, const N: usize, T> {
    data: &'a [T],
    inner_stride: usize,
    outer_stride: usize,
}

impl<'a, const M: usize, const N: usize, T> StridedMap<'a, M, N, T> {
    /// Maps a strided buffer, or returns `None` when the layout exceeds it.
    #[inline]
    pub fn from_slice(data: &'a [T], inner_stride: usize, outer_stride: usize) -> Option<Self> {
        let last = last_strided_index::<M, N>(inner_stride, outer_stride)?;
        Some(Self {
            data: data.get(..last)?,
            inner_stride,
            outer_stride,
        })
    }

    /// Returns the mapped storage, including any padding between elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Returns the row stride.
    #[inline]
    pub fn inner_stride(&self) -> usize {
        self.inner_stride
    }

    /// Returns the column stride.
    #[inline]
    pub fn outer_stride(&self) -> usize {
        self.outer_stride
    }

    /// Returns the element at `(row, column)`, or `None` when out of bounds.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        if row < M && column < N {
            Some(&self.data[row * self.inner_stride + column * self.outer_stride])
        } else {
            None
        }
    }

    /// Copies the mapped data into an owning matrix.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<M, N, T>
    where
        T: Copy,
    {
        Matrix::from_fn(|row, column| {
            *self
                .get(row, column)
                .expect("strided map index is in bounds")
        })
    }
}

impl<const M: usize, const N: usize, T> Index<(usize, usize)> for StridedMap<'_, M, N, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("strided map index is out of bounds")
    }
}

impl<const M: usize, const N: usize, T> MatrixRead<M, N, T> for StridedMap<'_, M, N, T> {
    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<&T> {
        StridedMap::get(self, row, column)
    }
}

/// A mutable fixed-size, zero-copy view with runtime inner and outer strides.
pub struct StridedMapMut<'a, const M: usize, const N: usize, T> {
    data: &'a mut [T],
    inner_stride: usize,
    outer_stride: usize,
}

impl<'a, const M: usize, const N: usize, T> StridedMapMut<'a, M, N, T> {
    /// Maps a strided buffer, or returns `None` when the layout exceeds it.
    #[inline]
    pub fn from_slice(data: &'a mut [T], inner_stride: usize, outer_stride: usize) -> Option<Self> {
        let last = last_strided_index::<M, N>(inner_stride, outer_stride)?;
        Some(Self {
            data: data.get_mut(..last)?,
            inner_stride,
            outer_stride,
        })
    }

    /// Returns the mapped storage, including any padding between elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Returns the mapped storage mutably.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// Reborrows the mutable mapping as an immutable mapping.
    #[inline]
    pub fn as_map(&self) -> StridedMap<'_, M, N, T> {
        StridedMap {
            data: self.data,
            inner_stride: self.inner_stride,
            outer_stride: self.outer_stride,
        }
    }

    /// Returns the row stride.
    #[inline]
    pub fn inner_stride(&self) -> usize {
        self.inner_stride
    }

    /// Returns the column stride.
    #[inline]
    pub fn outer_stride(&self) -> usize {
        self.outer_stride
    }

    /// Returns the element at `(row, column)`, or `None` when out of bounds.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        if row < M && column < N {
            Some(&self.data[row * self.inner_stride + column * self.outer_stride])
        } else {
            None
        }
    }

    /// Returns the mutable element at `(row, column)`, or `None` when out of
    /// bounds.
    #[inline]
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        if row < M && column < N {
            Some(&mut self.data[row * self.inner_stride + column * self.outer_stride])
        } else {
            None
        }
    }

    /// Copies the mapped data into an owning matrix.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<M, N, T>
    where
        T: Copy,
    {
        self.as_map().to_matrix()
    }
}

impl<const M: usize, const N: usize, T> Index<(usize, usize)> for StridedMapMut<'_, M, N, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("strided map index is out of bounds")
    }
}

impl<const M: usize, const N: usize, T> IndexMut<(usize, usize)> for StridedMapMut<'_, M, N, T> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1)
            .expect("strided map index is out of bounds")
    }
}

impl<const M: usize, const N: usize, T> MatrixRead<M, N, T> for StridedMapMut<'_, M, N, T> {
    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<&T> {
        StridedMapMut::get(self, row, column)
    }
}

impl<const M: usize, const N: usize, T> MatrixWrite<M, N, T> for StridedMapMut<'_, M, N, T> {
    #[inline]
    fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        StridedMapMut::get_mut(self, row, column)
    }
}

#[inline]
fn last_strided_index<const M: usize, const N: usize>(
    inner_stride: usize,
    outer_stride: usize,
) -> Option<usize> {
    if M == 0 || N == 0 {
        return Some(0);
    }
    if inner_stride == 0 || outer_stride == 0 {
        return None;
    }
    let row_offset = (M - 1).checked_mul(inner_stride)?;
    let column_offset = (N - 1).checked_mul(outer_stride)?;
    row_offset.checked_add(column_offset)?.checked_add(1)
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

#[test]
fn map_views_read_and_write_external_column_major_storage() {
    use super::*;
    let mut storage = [1, 4, 2, 5, 3, 6, 99];
    let mapped = Map::<2, 3, _>::from_slice(&storage).expect("storage is large enough");
    assert_eq!(mapped[(0, 0)], 1);
    assert_eq!(mapped[(1, 2)], 6);
    assert_eq!(
        mapped.to_matrix(),
        Matrix::<2, 3, i32>::from_rows([[1, 2, 3], [4, 5, 6]])
    );
    assert!(Map::<3, 3, _>::from_slice(&storage).is_none());

    let mut mapped = MapMut::<2, 3, _>::from_slice(&mut storage).expect("storage is large enough");
    mapped[(1, 1)] = 50;
    *mapped.get_mut(0, 2).expect("mapped index is in bounds") = 30;
    assert_eq!(mapped.as_map().to_matrix(), matrix![1, 2, 30; 4, 50, 6]);
    assert_eq!(storage, [1, 4, 2, 50, 30, 6, 99]);
}

#[test]
fn strided_map_views_handle_padding_and_row_major_storage() {
    use super::*;
    let mut storage = [1, 2, 3, 99, 4, 5, 6, 88];
    let mapped = StridedMap::<2, 3, _>::from_slice(&storage, 4, 1)
        .expect("row-major padded storage is large enough");
    assert_eq!(mapped[(0, 0)], 1);
    assert_eq!(mapped[(0, 2)], 3);
    assert_eq!(mapped[(1, 0)], 4);
    assert_eq!(mapped.to_matrix(), matrix![1, 2, 3; 4, 5, 6]);
    assert_eq!(mapped.inner_stride(), 4);
    assert_eq!(mapped.outer_stride(), 1);
    assert!(StridedMap::<2, 3, _>::from_slice(&storage[..5], 4, 1).is_none());

    let mut mapped = StridedMapMut::<2, 3, _>::from_slice(&mut storage, 4, 1)
        .expect("row-major padded storage is large enough");
    mapped[(1, 2)] = 60;
    assert_eq!(storage, [1, 2, 3, 99, 4, 5, 60, 88]);
}

#[test]
fn matrix_read_and_write_traits_cover_views() {
    use super::*;
    let matrix = matrix![1, 2, 3; 4, 5, 6];
    let block = matrix.block::<2, 2>(0, 1).expect("block is in bounds");
    let copied = Matrix::<2, 2, i32>::from_view(&block);
    assert_eq!(copied, matrix![2, 3; 5, 6]);

    let mut storage = [7, 9, 8, 10];
    let mut mapped = MapMut::<2, 2, _>::from_slice(&mut storage).unwrap();
    let view: &dyn MatrixRead<2, 2, i32> = &mapped;
    assert_eq!(view.get(1, 0), Some(&9));
    *<MapMut<'_, 2, 2, i32> as MatrixWrite<2, 2, i32>>::get_mut(&mut mapped, 0, 1).unwrap() = 12;
    assert_eq!(mapped.to_matrix(), matrix![7, 12; 9, 10]);
}
