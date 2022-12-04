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
