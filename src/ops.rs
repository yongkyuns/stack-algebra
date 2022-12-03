use core::ops::{Index, IndexMut};

use crate::index::MatrixIndex;
use crate::Matrix;

// ////////////////////////////////////////////////////////////////////////////////
// // Indexing
// ////////////////////////////////////////////////////////////////////////////////

// impl<T, const M: usize, const N: usize> Index<usize> for Matrix<M, N, T> {
//     type Output = Option<&[T; N]>;

//     #[inline]
//     fn index(&self, index: usize) -> &Self::Output {
//         // &self.data[index]
//         &self.data.iter().map(|col| col[index]).collect::<[T; N]>()
//     }
// }

////////////////////////////////////////////////////////////////////////////////
// Indexing
////////////////////////////////////////////////////////////////////////////////

impl<T, I, const M: usize, const N: usize> Index<I> for Matrix<M, N, T>
where
    I: MatrixIndex<Self>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, I, const M: usize, const N: usize> IndexMut<I> for Matrix<M, N, T>
where
    I: MatrixIndex<Self>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}
