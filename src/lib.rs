// #![no_std]
#![allow(non_snake_case)]

mod algebra;
mod fmt;
mod index;
mod iter;
mod new;
mod num;
mod ops;
mod util;
mod view;

pub use core::slice;
pub use index::MatrixIndex;
// pub use iter::{IntoIter, IterColumns, IterColumnsMut, IterRows, IterRowsMut};
pub use num::{One, Zero};
pub use view::{Column, Row};

#[doc(hidden)]
pub use vectrix_macro as proc_macro;

/// Represents a matrix with constant `M` rows and constant `N` columns.
///
/// The underlying data is represented as an array and is always stored in
/// column-major order.
///
/// See the [crate root][crate] for usage examples.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Matrix<const M: usize, const N: usize, T = f32> {
    data: [[T; M]; N],
}

impl<const M: usize, const N: usize, T> Matrix<M, N, T> {
    /// Returns a raw pointer to the underlying data.
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr() as *const T
    }

    /// Returns an unsafe mutable pointer to the underlying data.
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr() as *mut T
    }

    /// Views the underlying data as a contiguous slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), M * N) }
    }

    /// Views the underlying data as a contiguous mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), M * N) }
    }

    /// Returns a reference to the `i`-th row of this matrix.
    #[inline]
    pub fn row(&self, i: usize) -> &Row<M, N, T> {
        Row::new(&self.as_slice()[i..])
    }

    /// Returns a mutable reference to the `i`-th row of this matrix.
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut Row<M, N, T> {
        Row::new_mut(&mut self.as_mut_slice()[i..])
    }

    /// Returns a reference to the `i`-th column of this matrix.
    #[inline]
    pub fn column(&self, i: usize) -> &Column<M, N, T> {
        Column::new(&self.data[i])
    }

    /// Returns a mutable reference to the `i`-th column of this matrix.
    #[inline]
    pub fn column_mut(&mut self, i: usize) -> &mut Column<M, N, T> {
        Column::new_mut(&mut self.data[i])
    }

    /// Returns a reference to an element in the matrix or `None` if out of
    /// bounds.
    #[inline]
    pub fn get<I>(&self, i: I) -> Option<&I::Output>
    where
        I: MatrixIndex<Self>,
    {
        i.get(self)
    }

    /// Returns a mutable reference to an element in the matrix or `None` if out
    /// of bounds.
    #[inline]
    pub fn get_mut<I>(&mut self, i: I) -> Option<&mut I::Output>
    where
        I: MatrixIndex<Self>,
    {
        i.get_mut(self)
    }

    /// Returns a reference to an element in the matrix without doing any bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// *[undefined behavior]* even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub unsafe fn get_unchecked<I>(&self, i: I) -> &I::Output
    where
        I: MatrixIndex<Self>,
    {
        unsafe { &*i.get_unchecked(self) }
    }

    /// Returns a mutable reference to an element in the matrix without doing
    /// any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// *[undefined behavior]* even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub unsafe fn get_unchecked_mut<I>(&mut self, i: I) -> &mut I::Output
    where
        I: MatrixIndex<Self>,
    {
        unsafe { &mut *i.get_unchecked_mut(self) }
    }

    /// Returns an iterator over the underlying data.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the underlying data.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// Swap the two given rows of this matrix
    #[inline]
    pub fn swap_rows(&mut self, r1: usize, r2: usize)
    where
        T: Copy,
    {
        if r1 < M && r2 < M {
            for i in 0..N {
                let tmp = self[(r1, i)];
                self[(r1, i)] = self[(r2, i)];
                self[(r2, i)] = tmp;
            }
        }
    }

    /// Swap the two given columns of this matrix
    #[inline]
    pub fn swap_columns(&mut self, c1: usize, c2: usize)
    where
        T: Copy,
    {
        if c1 < N && c2 < N {
            for i in 0..M {
                let tmp = self[(i, c1)];
                self[(i, c1)] = self[(i, c2)];
                self[(i, c2)] = tmp;
            }
        }
    }

    /// Clone the current matrix.
    #[inline]
    pub fn clone(&self) -> Matrix<N, M, T>
    where
        T: Copy + Zero,
    {
        let mut clone = zeros!(N, M, T);
        for c in 0..N {
            for r in 0..M {
                clone[(r, c)] = self[(r, c)];
            }
        }
        clone
    }

    /// Transpose of the current matrix.
    #[inline]
    pub fn transpose(&self) -> Matrix<N, M, T>
    where
        T: Copy + Zero,
    {
        let mut transpose = zeros!(N, M, T);
        for c in 0..N {
            for r in 0..M {
                transpose[(c, r)] = self[(r, c)];
            }
        }
        transpose
    }

    /// Transpose of the current matrix.
    #[inline]
    pub fn T(&self) -> Matrix<N, M, T>
    where
        T: Copy + Zero,
    {
        self.transpose()
    }

    // /// Returns an iterator over the rows in this matrix.
    // #[inline]
    // pub fn iter_rows(&self) -> IterRows<'_, T, M, N> {
    //     IterRows::new(self)
    // }

    // /// Returns a mutable iterator over the rows in this matrix.
    // #[inline]
    // pub fn iter_rows_mut(&mut self) -> IterRowsMut<'_, T, M, N> {
    //     IterRowsMut::new(self)
    // }

    // /// Returns an iterator over the columns in this matrix.
    // #[inline]
    // pub fn iter_columns(&self) -> IterColumns<'_, T, M, N> {
    //     IterColumns::new(self)
    // }

    // /// Returns a mutable iterator over the columns in this matrix.
    // #[inline]
    // pub fn iter_columns_mut(&mut self) -> IterColumnsMut<'_, T, M, N> {
    //     IterColumnsMut::new(self)
    // }

    // /// Returns a matrix of the same size as self, with function `f` applied to
    // /// each element in column-major order.
    // #[inline]
    // pub fn map<F, U>(self, f: F) -> Matrix<M, N, U>
    // where
    //     F: FnMut(T) -> U,
    // {
    //     // SAFETY: the iterator has the exact number of elements required.
    //     unsafe { new::collect_unchecked(self.into_iter().map(f)) }
    // }

    // /// Returns the L1 norm of the matrix.
    // ///
    // /// Also known as *Manhattan Distance* or *Taxicab norm*. L1 Norm is the sum
    // /// of the magnitudes of the vectors in a space.
    // pub fn l1_norm(&self) -> T
    // where
    //     T: Copy + Ord + Abs + Zero + Sum<T>,
    // {
    //     (0..N)
    //         .map(|i| self.data[i].iter().copied().map(Abs::abs).sum())
    //         .max()
    //         .unwrap_or_else(Zero::zero)
    // }
}

/// A matrix with one row and `N` columns.
pub type RowVector<const N: usize, T> = Matrix<1, N, T>;

/// A matrix with one column and `M` rows.
pub type Vector<const M: usize, T> = Matrix<M, 1, T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 2)], 6.0);

        let v = vector![1.0, 2.0, 3.0];
        assert_eq!(v[0], 1.0);
        assert_eq!(v[2], 3.0);

        let z = zeros!(2, 3);
        assert_eq!(z[(0, 0)], 0.0);
        assert_eq!(z[(1, 2)], 0.0);

        let z = zeros!(3);
        assert_eq!(z[(2, 2)], 0.0);

        let o = ones!(2, 3);
        assert_eq!(o[(0, 0)], 1.0);
        assert_eq!(o[(1, 2)], 1.0);

        let o = ones!(3);
        assert_eq!(o[(2, 2)], 1.0);
    }

    #[test]
    fn index() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        assert_eq!(m[1], 4.0);
        assert_eq!(m[(1, 2)], 6.0);

        let mut s = m.as_slice().iter();
        assert_eq!(s.next(), Some(&1.0));
        assert_eq!(s.next(), Some(&4.0));
        assert_eq!(s.next(), Some(&2.0));
        assert_eq!(s.next(), Some(&5.0));
        assert_eq!(s.next(), Some(&3.0));
        assert_eq!(s.next(), Some(&6.0));
        assert_eq!(s.next(), None);
    }
    #[test]
    fn swap() {
        let mut m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
        ];
        m.swap_rows(0, 2);
        let exp = matrix![
            7.0, 8.0, 9.0;
            4.0, 5.0, 6.0;
            1.0, 2.0, 3.0;
        ];
        assert_eq!(m, exp);
        m.swap_columns(0, 2);
        let exp = matrix![
            9.0, 8.0, 7.0;
            6.0, 5.0, 4.0;
            3.0, 2.0, 1.0;
        ];
        assert_eq!(m, exp);
    }
    #[test]
    fn transpose() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        let t = matrix![
            1.0, 4.0;
            2.0, 5.0;
            3.0, 6.0;
        ];
        assert_eq!(m.transpose(), t);
    }
}
