//! Runtime-shaped borrowed matrices, with storage owned by the caller.
//!
//! [`MatrixRef`] and [`MatrixMut`] describe an active rectangle in a slice.
//! They never allocate, resize storage, or pad a problem to a compile-time bound.
//! Strides are nonnegative element counts; transpose and submatrix operations
//! only change the view. Existing fixed-size types and solvers are unchanged.
//!
//! [`copy_into`], [`transpose_into`], and [`matmul_into`] validate all shapes
//! before writing. A returned error leaves the destination unchanged. Arithmetic
//! follows the scalar's ordinary operators: floating-point nonfinite values are
//! not rejected, and a scalar panic does not promise rollback. No runtime-sized
//! factorization or optimized runtime kernel dispatch is provided here.
//!
//! ```
//! use stack_algebra::runtime::{matmul_into, MatrixMut, MatrixRef};
//!
//! let lhs = [1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
//! let rhs = [2.0_f64, -1.0, 0.5];
//! let mut result = [0.0; 2];
//! matmul_into(
//!     MatrixRef::from_column_major(&lhs, 2, 3)?,
//!     MatrixRef::from_column_major(&rhs, 3, 1)?,
//!     &mut MatrixMut::from_column_major(&mut result, 2, 1)?,
//! )?;
//! assert_eq!(result, [1.5, 6.0]);
//! # Ok::<(), stack_algebra::runtime::Error>(())
//! ```
//!
//! Shared inputs may alias each other, but safe borrowing forbids an output from
//! aliasing a live input. Use disjoint slices for distinct input/output buffers.
//!
//! ```compile_fail
//! use stack_algebra::runtime::{copy_into, MatrixMut, MatrixRef};
//! let mut data = [1.0_f32; 4];
//! let input = MatrixRef::from_column_major(&data, 2, 2).unwrap();
//! let mut output = MatrixMut::from_column_major(&mut data, 2, 2).unwrap();
//! copy_into(input, &mut output).unwrap();
//! ```
#![forbid(unsafe_code)]

use core::ops::{Add, Index, IndexMut, Mul};

use crate::Zero;

/// Invalid runtime layout, submatrix, or operation shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A shape, extent, or offset calculation overflowed `usize`.
    SizeOverflow,
    /// The backing slice does not cover all addressed elements.
    BufferTooShort {
        /// Required slice length, including any addressed padding.
        required: usize,
        /// Supplied slice length.
        available: usize,
    },
    /// Two logical elements of a mutable view address the same slice element.
    OverlappingElements,
    /// A requested submatrix extends beyond its parent's active shape.
    BlockOutOfBounds,
    /// An operand or destination has the wrong active shape.
    ShapeMismatch {
        /// Required `(rows, columns)`.
        expected: (usize, usize),
        /// Supplied `(rows, columns)`.
        actual: (usize, usize),
    },
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    rows: usize,
    columns: usize,
    row_stride: usize,
    column_stride: usize,
}

impl Layout {
    fn is_empty(self) -> bool {
        self.rows == 0 || self.columns == 0
    }

    fn span(self) -> Result<usize, Error> {
        self.rows.checked_mul(self.columns).ok_or(Error::SizeOverflow)?;
        if self.is_empty() {
            return Ok(0);
        }
        (self.rows - 1)
            .checked_mul(self.row_stride)
            .and_then(|row| {
                (self.columns - 1)
                    .checked_mul(self.column_stride)
                    .and_then(|column| row.checked_add(column))
            })
            .and_then(|last| last.checked_add(1))
            .ok_or(Error::SizeOverflow)
    }

    fn validate(self, available: usize) -> Result<usize, Error> {
        let required = self.span()?;
        if required > available {
            return Err(Error::BufferTooShort { required, available });
        }
        Ok(required)
    }

    fn validate_mutable(self) -> Result<(), Error> {
        if self.is_empty() {
            return Ok(());
        }
        if (self.rows > 1 && self.row_stride == 0)
            || (self.columns > 1 && self.column_stride == 0)
        {
            return Err(Error::OverlappingElements);
        }
        if self.rows > 1 && self.columns > 1 {
            // A collision requires dr*row_stride = dc*column_stride. The
            // smallest positive pair is (column_stride/gcd, row_stride/gcd).
            let mut a = self.row_stride;
            let mut b = self.column_stride;
            while b != 0 {
                (a, b) = (b, a % b);
            }
            if self.rows > self.column_stride / a && self.columns > self.row_stride / a {
                return Err(Error::OverlappingElements);
            }
        }
        Ok(())
    }

    fn index(self, row: usize, column: usize) -> Option<usize> {
        if row < self.rows && column < self.columns {
            // Construction validated the maximum addressed offset.
            Some(row * self.row_stride + column * self.column_stride)
        } else {
            None
        }
    }

    fn transpose(self) -> Self {
        Self {
            rows: self.columns,
            columns: self.rows,
            row_stride: self.column_stride,
            column_stride: self.row_stride,
        }
    }

    fn submatrix(
        self,
        row: usize,
        column: usize,
        rows: usize,
        columns: usize,
    ) -> Result<(Self, usize), Error> {
        let row_end = row.checked_add(rows).ok_or(Error::SizeOverflow)?;
        let column_end = column.checked_add(columns).ok_or(Error::SizeOverflow)?;
        if row_end > self.rows || column_end > self.columns {
            return Err(Error::BlockOutOfBounds);
        }
        let layout = Self { rows, columns, ..self };
        // Empty edge blocks have no address. Do not calculate a potentially
        // overflowing or beyond-buffer origin for them.
        let offset = if layout.is_empty() {
            0
        } else {
            self.index(row, column).ok_or(Error::BlockOutOfBounds)?
        };
        Ok((layout, offset))
    }
}

/// Read-only runtime-shaped view, including strided or repeated input elements.
///
/// Copying this view copies only its borrow and layout, not the scalar values.
#[derive(Debug)]
pub struct MatrixRef<'a, T> {
    data: &'a [T],
    layout: Layout,
}

impl<T> Copy for MatrixRef<'_, T> {}

impl<T> Clone for MatrixRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> MatrixRef<'a, T> {
    /// Borrows packed column-major storage. Extra trailing elements are ignored.
    pub fn from_column_major(data: &'a [T], rows: usize, columns: usize) -> Result<Self, Error> {
        Self::from_strides(data, rows, columns, 1, rows)
    }

    /// Borrows packed row-major storage. Extra trailing elements are ignored.
    pub fn from_row_major(data: &'a [T], rows: usize, columns: usize) -> Result<Self, Error> {
        Self::from_strides(data, rows, columns, columns, 1)
    }

    /// Borrows a layout addressing `row * row_stride + column * column_stride`.
    ///
    /// Strides count elements, not bytes. Padding and repeated read-only elements
    /// (including zero strides) are permitted. Empty shapes address no elements
    /// and accept arbitrary strides. Shape and extent arithmetic is checked.
    pub fn from_strides(
        data: &'a [T],
        rows: usize,
        columns: usize,
        row_stride: usize,
        column_stride: usize,
    ) -> Result<Self, Error> {
        let layout = Layout { rows, columns, row_stride, column_stride };
        let span = layout.validate(data.len())?;
        Ok(Self { data: &data[..span], layout })
    }

    /// Returns the active `(rows, columns)`.
    pub fn shape(self) -> (usize, usize) {
        (self.layout.rows, self.layout.columns)
    }

    /// Returns `(row_stride, column_stride)` in elements.
    pub fn strides(self) -> (usize, usize) {
        (self.layout.row_stride, self.layout.column_stride)
    }

    /// Returns a scalar reference, or `None` outside the active rectangle.
    pub fn get(self, row: usize, column: usize) -> Option<&'a T> {
        self.layout.index(row, column).map(|index| &self.data[index])
    }

    /// Transposes the view without moving data.
    pub fn transpose(self) -> Self {
        Self { data: self.data, layout: self.layout.transpose() }
    }

    /// Borrows an active submatrix without copying, with block-local indexing.
    pub fn submatrix(
        self,
        row: usize,
        column: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Self, Error> {
        let (layout, offset) = self.layout.submatrix(row, column, rows, columns)?;
        let span = layout.span()?;
        Ok(Self { data: &self.data[offset..offset + span], layout })
    }
}

impl<T> Index<(usize, usize)> for MatrixRef<'_, T> {
    type Output = T;

    fn index(&self, (row, column): (usize, usize)) -> &T {
        self.get(row, column).expect("runtime matrix index out of bounds")
    }
}

/// Exclusive runtime-shaped view with nonoverlapping logical elements.
///
/// This view is not `Copy` or `Clone`. A transpose or submatrix consumes the
/// view; use [`Self::reborrow`] when the parent should remain usable afterward.
/// Any child borrow prevents simultaneous parent access.
///
/// ```compile_fail
/// use stack_algebra::runtime::MatrixMut;
/// let mut data = [0_i32; 4];
/// let mut parent = MatrixMut::from_column_major(&mut data, 2, 2).unwrap();
/// let mut child = parent.reborrow().submatrix(0, 0, 1, 1).unwrap();
/// parent[(0, 0)] = 1;
/// child[(0, 0)] = 2;
/// ```
///
/// A shared reborrow also prevents writes while it remains in use:
///
/// ```compile_fail
/// use stack_algebra::runtime::MatrixMut;
/// let mut data = [0_i32; 4];
/// let mut parent = MatrixMut::from_column_major(&mut data, 2, 2).unwrap();
/// let input = parent.as_ref();
/// parent[(0, 0)] = 1;
/// assert_eq!(input[(0, 0)], 0);
/// ```
#[derive(Debug)]
pub struct MatrixMut<'a, T> {
    data: &'a mut [T],
    layout: Layout,
}

impl<'a, T> MatrixMut<'a, T> {
    /// Borrows packed column-major storage. Extra trailing elements are untouched.
    pub fn from_column_major(data: &'a mut [T], rows: usize, columns: usize) -> Result<Self, Error> {
        Self::from_strides(data, rows, columns, 1, rows)
    }

    /// Borrows packed row-major storage. Extra trailing elements are untouched.
    pub fn from_row_major(data: &'a mut [T], rows: usize, columns: usize) -> Result<Self, Error> {
        Self::from_strides(data, rows, columns, columns, 1)
    }

    /// Borrows a nonoverlapping strided rectangle, validating before mutation.
    ///
    /// Uses the same addressing and empty-shape rules as [`MatrixRef::from_strides`],
    /// but rejects any layout mapping distinct logical elements to one scalar.
    pub fn from_strides(
        data: &'a mut [T],
        rows: usize,
        columns: usize,
        row_stride: usize,
        column_stride: usize,
    ) -> Result<Self, Error> {
        let layout = Layout { rows, columns, row_stride, column_stride };
        let span = layout.validate(data.len())?;
        layout.validate_mutable()?;
        Ok(Self { data: &mut data[..span], layout })
    }

    /// Returns the active `(rows, columns)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.layout.rows, self.layout.columns)
    }

    /// Returns `(row_stride, column_stride)` in elements.
    pub fn strides(&self) -> (usize, usize) {
        (self.layout.row_stride, self.layout.column_stride)
    }

    /// Creates a shared view limited to this borrow of `self`.
    pub fn as_ref(&self) -> MatrixRef<'_, T> {
        MatrixRef { data: self.data, layout: self.layout }
    }

    /// Creates an exclusive view limited to this borrow of `self`.
    pub fn reborrow(&mut self) -> MatrixMut<'_, T> {
        MatrixMut { data: self.data, layout: self.layout }
    }

    /// Returns a scalar reference, or `None` outside the active rectangle.
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.layout.index(row, column).map(|index| &self.data[index])
    }

    /// Returns an exclusive scalar reference, or `None` outside the active rectangle.
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        self.layout.index(row, column).map(|index| &mut self.data[index])
    }

    /// Transposes this exclusive view without moving data or extending its borrow.
    pub fn transpose(self) -> Self {
        Self { data: self.data, layout: self.layout.transpose() }
    }

    /// Restricts this exclusive view to a submatrix with block-local indexing.
    pub fn submatrix(
        self,
        row: usize,
        column: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Self, Error> {
        let (layout, offset) = self.layout.submatrix(row, column, rows, columns)?;
        let span = layout.span()?;
        Ok(Self { data: &mut self.data[offset..offset + span], layout })
    }
}

impl<T> Index<(usize, usize)> for MatrixMut<'_, T> {
    type Output = T;

    fn index(&self, (row, column): (usize, usize)) -> &T {
        self.get(row, column).expect("runtime matrix index out of bounds")
    }
}

impl<T> IndexMut<(usize, usize)> for MatrixMut<'_, T> {
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut T {
        self.get_mut(row, column).expect("runtime matrix index out of bounds")
    }
}

fn require_shape(expected: (usize, usize), actual: (usize, usize)) -> Result<(), Error> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::ShapeMismatch { expected, actual })
    }
}

/// Copies active elements without touching destination padding.
///
/// Shape rejection leaves the entire destination unchanged. No scratch buffer
/// or allocation is needed; the input and destination must be disjoint borrows.
pub fn copy_into<T: Copy>(input: MatrixRef<'_, T>, output: &mut MatrixMut<'_, T>) -> Result<(), Error> {
    require_shape(input.shape(), output.shape())?;
    if output.layout.is_empty() {
        return Ok(());
    }
    for column in 0..output.layout.columns {
        for row in 0..output.layout.rows {
            output[(row, column)] = input[(row, column)];
        }
    }
    Ok(())
}

/// Writes the transposed input to a disjoint destination without a temporary.
///
/// Shape rejection leaves the destination unchanged, including padding.
pub fn transpose_into<T: Copy>(
    input: MatrixRef<'_, T>,
    output: &mut MatrixMut<'_, T>,
) -> Result<(), Error> {
    copy_into(input.transpose(), output)
}

/// Computes `output = lhs * rhs` over active runtime dimensions.
///
/// Both the inner dimension and output shape are checked before any writes.
/// Empty outputs return immediately; a zero inner dimension writes zeros to
/// active output elements. Padding is untouched and existing output values are
/// not read. Shared inputs may alias each other but not the exclusive output.
///
/// Uses scalar multiplication/addition in increasing inner-index order with
/// constant auxiliary storage. This is not a performance-qualified SIMD path.
pub fn matmul_into<T>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    output: &mut MatrixMut<'_, T>,
) -> Result<(), Error>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    require_shape((lhs.layout.columns, rhs.layout.columns), rhs.shape())?;
    require_shape((lhs.layout.rows, rhs.layout.columns), output.shape())?;
    if output.layout.is_empty() {
        return Ok(());
    }
    for column in 0..output.layout.columns {
        for row in 0..output.layout.rows {
            let mut value = T::zero();
            for inner in 0..lhs.layout.columns {
                value = value + lhs[(row, inner)] * rhs[(inner, column)];
            }
            output[(row, column)] = value;
        }
    }
    Ok(())
}
