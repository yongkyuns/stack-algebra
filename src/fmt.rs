use core::fmt;

use crate::Matrix;

////////////////////////////////////////////////////////////////////////////////
// Debug
////////////////////////////////////////////////////////////////////////////////

impl<T: fmt::Debug, const M: usize, const N: usize> fmt::Debug for Matrix<M, N, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if M == 1 || N == 1 {
            f.debug_list().entries(self.iter()).finish()
        } else {
            fmt::Debug::fmt(&self.data, f)
        }
    }
}

impl<T: fmt::Display, const M: usize, const N: usize> fmt::Display for Matrix<M, N, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\n")?;
        for r in 0..M {
            for c in 0..N {
                write!(f, "{:2.11} ", self[(r, c)])?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}
