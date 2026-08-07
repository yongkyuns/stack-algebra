//! Borrowed, zero-copy views into fixed-size matrices and external buffers.
//!
//! A [`Matrix`](crate::Matrix) is column-major, so a contiguous buffer for an
//! `M`-by-`N` matrix stores `(row, column)` at `row + M * column`. [`Map`] and
//! [`MapMut`] expose that layout without copying. [`StridedMap`] is useful when
//! a producer adds padding or uses a different layout: its element address is
//! `row * inner_stride + column * outer_stride`.
//!
//! Views borrow their source for their lifetime and therefore do not allocate.
//! Use [`Block`] or [`BlockMut`] for a compile-time-sized region of an owning
//! matrix, and [`MatrixRead`] / [`MatrixWrite`] when an algorithm should accept
//! any of these representations.
//!
//! # Example
//!
//! ```
//! use stack_algebra::{matrix, matmul_view_into, Map, Matrix};
//!
//! let storage = [1, 3, 2, 4]; // [[1, 2], [3, 4]] in column-major order
//! let input = Map::<2, 2, _>::from_slice(&storage).unwrap();
//! let mut output = Matrix::<2, 2, i32>::zeros();
//! matmul_view_into(&input, &input, &mut output).unwrap();
//! assert_eq!(output, matrix![7, 10; 15, 22]);
//! ```

use core::iter::Sum;
use core::ops::{Deref, DerefMut, Index, IndexMut, Mul};

use crate::Matrix;
use stride::Stride;

/// Read-only access to a fixed-size matrix-shaped view.
///
/// Implement this trait for generic algorithms that should work with an
/// owning [`Matrix`](crate::Matrix), a block, or an externally mapped buffer.
/// Implementations must return `None` for coordinates outside the declared
/// `M`-by-`N` shape.
pub trait MatrixRead<const M: usize, const N: usize, T> {
    /// Returns the element at `(row, column)`, or `None` when out of bounds.
    fn get(&self, row: usize, column: usize) -> Option<&T>;
}

/// Mutable access to a fixed-size matrix-shaped view.
///
/// This trait extends [`MatrixRead`], allowing one algorithm to inspect and
/// update a view without taking ownership of its backing storage.
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
///
/// The block keeps a shared borrow of its source. Creating it checks the
/// offsets once; indexing the returned view uses block-local coordinates.
/// Call [`Matrix::block`](crate::Matrix::block) to construct one.
///
/// # Example
///
/// ```
/// use stack_algebra::{matrix, Matrix};
///
/// let matrix = matrix![1, 2, 3; 4, 5, 6; 7, 8, 9];
/// let block = matrix.block::<2, 2>(1, 1).unwrap();
/// assert_eq!(block[(0, 0)], 5);
/// assert_eq!(block.to_matrix(), Matrix::from_rows([[5, 6], [8, 9]]));
/// ```
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
///
/// A mutable block exclusively borrows its source until it is dropped. This
/// makes in-place updates safe while preserving the source matrix's storage
/// and avoiding an intermediate allocation.
///
/// # Example
///
/// ```
/// use stack_algebra::{matrix, Matrix};
///
/// let mut matrix = matrix![1, 2; 3, 4];
/// {
///     let mut block = matrix.block_mut::<1, 2>(1, 0).unwrap();
///     block[(0, 0)] = 30;
///     *block.get_mut(0, 1).unwrap() = 40;
/// }
/// assert_eq!(matrix, Matrix::from_rows([[1, 2], [30, 40]]));
/// ```
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
///
/// `Map` borrows exactly the first `M * N` elements of the supplied slice.
/// Extra elements remain available to the caller after the mapped view is
/// dropped. The source must use column-major order.
///
/// # Example
///
/// ```
/// use stack_algebra::{Map, Matrix};
///
/// let storage = [1.0_f32, 3.0, 2.0, 4.0];
/// let mapped = Map::<2, 2, _>::from_slice(&storage).unwrap();
/// assert_eq!(mapped[(0, 1)], 2.0);
/// assert_eq!(mapped.to_matrix(), Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]));
/// ```
pub struct Map<'a, const M: usize, const N: usize, T> {
    data: &'a [T],
}

impl<'a, const M: usize, const N: usize, T> Map<'a, M, N, T> {
    /// Maps the first `M * N` elements of `data`, or returns `None` when it is
    /// too short.
    #[inline]
    pub fn from_slice(data: &'a [T]) -> Option<Self> {
        let len = M.checked_mul(N)?;
        Some(Self {
            data: data.get(..len)?,
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
///
/// `MapMut` is the allocation-free way to let an algorithm update a caller's
/// buffer. Use [`MapMut::as_map`] to temporarily pass an immutable view while
/// retaining ownership of the mutable mapping.
///
/// # Example
///
/// ```
/// use stack_algebra::MapMut;
///
/// let mut storage = [1_i32, 3, 2, 4];
/// let mut mapped = MapMut::<2, 2, _>::from_slice(&mut storage).unwrap();
/// mapped[(1, 1)] = 40;
/// assert_eq!(mapped.as_map()[(1, 1)], 40);
/// drop(mapped);
/// assert_eq!(storage, [1, 3, 2, 40]);
/// ```
pub struct MapMut<'a, const M: usize, const N: usize, T> {
    data: &'a mut [T],
}

impl<'a, const M: usize, const N: usize, T> MapMut<'a, M, N, T> {
    /// Maps the first `M * N` elements of `data`, or returns `None` when it is
    /// too short.
    #[inline]
    pub fn from_slice(data: &'a mut [T]) -> Option<Self> {
        let len = M.checked_mul(N)?;
        Some(Self {
            data: data.get_mut(..len)?,
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
/// Strides are measured in elements, not bytes. This supports row-major
/// buffers (`inner_stride = columns`, `outer_stride = 1`) as well as padded
/// sensor/DMA buffers. Construction validates the last accessed element before
/// borrowing the slice.
///
/// # Example
///
/// ```
/// use stack_algebra::{matrix, StridedMap};
///
/// let padded = [1_i32, 2, 99, 3, 4, 5, 88, 6];
/// let view = StridedMap::<2, 3, _>::from_slice(&padded, 4, 1).unwrap();
/// assert_eq!(view.to_matrix(), matrix![1, 2, 99; 4, 5, 88]);
/// ```
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
///
/// This is the mutable counterpart to [`StridedMap`]. It is useful for
/// writing a fixed-size result directly into a padded or interleaved buffer;
/// no temporary matrix is needed.
///
/// # Example
///
/// ```
/// use stack_algebra::StridedMapMut;
///
/// let mut padded = [0_i32; 8];
/// let mut view = StridedMapMut::<2, 3, _>::from_slice(&mut padded, 4, 1).unwrap();
/// view[(1, 2)] = 7;
/// drop(view);
/// assert_eq!(padded[6], 7);
/// ```
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
///
/// Rows are strided because the matrix is column-major. They dereference to a
/// [`Stride`](stride::Stride), so ordinary slice-style `get`, `iter`, and
/// indexing operations remain available without copying.
///
/// # Example
///
/// ```
/// use stack_algebra::matrix;
///
/// let matrix = matrix![1, 2, 3; 4, 5, 6];
/// assert_eq!(matrix.row(1).get(0..2).unwrap(), &[4, 5]);
/// ```
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
///
/// Columns are contiguous in the column-major representation. A column view
/// dereferences to a [`Stride`](stride::Stride) with unit stride.
///
/// # Example
///
/// ```
/// use stack_algebra::matrix;
///
/// let matrix = matrix![1, 2, 3; 4, 5, 6];
/// assert_eq!(matrix.column(1).get(0..2).unwrap(), &[2, 5]);
/// ```
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
    /// Returns the dot product with a column of matching length.
    ///
    /// The row and column are borrowed views, so this operation does not copy
    /// either operand. The column's row count must equal this row's length.
    #[inline]
    pub fn dot<const P: usize>(&self, other: &Column<N, P, T>) -> T
    where
        T: Copy + Mul<Output = T> + Sum,
    {
        (0..N).map(|i| self[i] * other[i]).sum()
    }

    /// Computes a dot product over a selected range of entries.
    ///
    /// Indices outside `0..N` are ignored by the iterator; an empty range
    /// returns the additive identity supplied by `Sum`.
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
fn map_dimensions_reject_multiplication_overflow() {
    let storage: [u8; 0] = [];
    assert!(Map::<{ usize::MAX }, 2, _>::from_slice(&storage).is_none());

    let mut storage: [u8; 0] = [];
    assert!(MapMut::<{ usize::MAX }, 2, _>::from_slice(&mut storage).is_none());
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
