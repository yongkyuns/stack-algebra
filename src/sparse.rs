//! Fixed-capacity sparse matrices and factorizations.
//!
//! The sparse types use compile-time dimensions and inline arrays.  `MAX_NNZ`
//! is a storage bound, not a request to allocate that many entries at runtime;
//! constructors and insertions return an error when the bound is exceeded.
//! This makes the API suitable for `no_std` and embedded applications while
//! still supporting the symbolic/numeric split used by sparse solvers.
//!
//! A typical CSC workflow is to validate a pattern once, analyze it once, and
//! then refactorize new numeric values during each control or estimation step:
//!
//! ```
//! use stack_algebra::{Matrix, StaticCscCholeskyPattern, StaticCscMatrix};
//!
//! type Matrix3 = StaticCscMatrix<2, 2, 3, f64>;
//! let a = Matrix3::from_pattern(
//!     &[4.0, 1.0, 3.0],
//!     &[0, 1, 1],
//!     &[0, 2, 3],
//! ).unwrap();
//! let pattern = StaticCscCholeskyPattern::<2, 3>::analyze(&a).unwrap();
//! let mut factor = pattern.factor(&a).unwrap();
//!
//! let b = Matrix::<2, 1, f64>::from_rows([[1.0], [2.0]]);
//! let x = factor.solve(&b);
//! assert!((x[(0, 0)] - 1.0 / 11.0).abs() < 1.0e-12);
//!
//! // Keep `pattern` and `factor` and update only values on later iterations.
//! let next = Matrix3::from_pattern(&[5.0, 1.0, 4.0], &[0, 1, 1], &[0, 2, 3]).unwrap();
//! factor.recompute(&next).unwrap();
//! ```
//!
//! CSC arrays are column-major: row indices are strictly increasing within a
//! column and the pointer array has `COLS + 1` entries.  Symmetric
//! factorizations consume the lower triangle; store that triangle in the
//! pattern even when an application also keeps an upper-triangle view.

use crate::{Ldlt, Matrix, MatrixScalar, Real, Zero};

mod cholesky;
mod errors;
mod ldlt;
mod ordering;
mod storage;

pub use cholesky::{StaticCscCholesky, StaticCscCholeskyPattern};
pub(crate) use errors::map_ldlt_error;
pub use errors::{CscError, SparseCholeskyError};
pub use ldlt::{StaticCscLdlt, StaticCscLdltFactor};
pub use ordering::StaticCscOrdering;
pub use storage::{StaticCscMatrix, StaticCscPattern};

/// Symbolic CSC pattern shared by sparse LLT and no-pivot LDLᵀ.
///
/// This alias is useful when a generated solver may switch between Cholesky
/// and LDLᵀ while retaining the same fill pattern.
pub type StaticCscLdltPattern<const N: usize, const MAX_L_NNZ: usize> =
    StaticCscCholeskyPattern<N, MAX_L_NNZ>;

pub(crate) fn default_ldlt_threshold<const N: usize, const MAX_NNZ: usize, T: Real>(
    matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
) -> T {
    let mut scale = T::zero();
    for column in 0..N {
        let start = matrix.column_starts()[column];
        let end = matrix.column_end(column).unwrap_or(matrix.nnz());
        for index in start..end {
            if matrix.row_indices()[index] >= column {
                scale = scale.max(matrix.values()[index].abs());
            }
        }
    }
    T::epsilon() * scale
}

impl<const N: usize, const MAX_NNZ: usize, T> StaticCscMatrix<N, N, MAX_NNZ, T> {
    /// Factors with native sparse LDLT, tries bounded diagonal pivoting for a
    /// zero leading pivot, and falls back to dense global Bunch–Kaufman
    /// pivoting only when a sparse 2x2 pivot is required.
    #[inline]
    pub fn try_ldlt_with_dense_fallback<const MAX_L_NNZ: usize>(
        &self,
    ) -> Result<StaticCscLdltFactor<N, MAX_L_NNZ, T>, SparseCholeskyError>
    where
        T: Real + MatrixScalar,
    {
        self.try_ldlt_with_dense_fallback_threshold(default_ldlt_threshold(self))
    }

    /// Factors using sparse LDLT, bounded diagonal pivoting, and a dense
    /// Bunch–Kaufman fallback with an explicit absolute pivot threshold.
    #[inline]
    pub fn try_ldlt_with_dense_fallback_threshold<const MAX_L_NNZ: usize>(
        &self,
        threshold: T,
    ) -> Result<StaticCscLdltFactor<N, MAX_L_NNZ, T>, SparseCholeskyError>
    where
        T: Real + MatrixScalar,
    {
        match StaticCscLdlt::decompose(self) {
            Ok(factor) => Ok(StaticCscLdltFactor::Sparse(factor)),
            Err(SparseCholeskyError::ZeroPivot) => {
                match StaticCscLdlt::decompose_with_diagonal_pivoting(self, threshold) {
                    Ok(factor) => Ok(StaticCscLdltFactor::Sparse(factor)),
                    Err(SparseCholeskyError::ZeroPivot) => {
                        Ok(StaticCscLdltFactor::Dense(self.try_dense_ldlt()?))
                    }
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        }
    }

    /// Expands the stored lower triangle into fixed-size dense storage and
    /// factors it with global Bunch–Kaufman pivoting.
    ///
    /// This is an explicit fallback for sparse systems that require a scalar
    /// 2×2 pivot. It remains allocation-free, but uses `N * N` inline dense
    /// storage and should therefore be reserved for small bounded systems.
    #[inline]
    pub fn try_dense_ldlt(&self) -> Result<Ldlt<N, T>, SparseCholeskyError>
    where
        T: Real + MatrixScalar,
    {
        let mut dense = Matrix::<N, N, T>::zeros();
        for column in 0..N {
            let start = self.column_starts()[column];
            let end = self.column_end(column).unwrap_or(self.nnz());
            for index in start..end {
                let row = self.row_indices()[index];
                if row >= column {
                    dense[(row, column)] = self.values()[index];
                }
            }
        }
        Ldlt::try_decompose(&dense).map_err(map_ldlt_error)
    }
}

impl<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize, T> Default
    for StaticCscMatrix<ROWS, COLS, MAX_NNZ, T>
where
    T: Copy + Zero,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector;

    type Csc = StaticCscMatrix<3, 2, 4, f64>;

    #[test]
    fn insertion_keeps_canonical_csc_pattern() {
        let mut matrix = Csc::new();
        matrix.insert(2, 1, 5.0).unwrap();
        matrix.insert(0, 0, 1.0).unwrap();
        matrix.insert(1, 0, 2.0).unwrap();
        matrix.insert(0, 1, 3.0).unwrap();
        assert_eq!(matrix.values(), &[1.0, 2.0, 3.0, 5.0]);
        assert_eq!(matrix.row_indices(), &[0, 1, 0, 2]);
        assert_eq!(matrix.column_starts(), &[0, 2]);
        assert_eq!(matrix.column_end(0), Some(2));
        assert_eq!(matrix.column_end(1), Some(4));
    }

    #[test]
    fn matvec_and_value_updates_work_without_repacking() {
        let mut matrix = Csc::from_pattern(&[1.0, 2.0, 3.0], &[0, 2, 1], &[0, 2, 3]).unwrap();
        let vector = vector![4.0; 5.0];
        assert_eq!(matrix.matvec(&vector), vector![4.0; 15.0; 8.0]);
        matrix.set_value(2, 0, 7.0).unwrap();
        assert_eq!(matrix.matvec(&vector), vector![4.0; 15.0; 28.0]);
        matrix.set_values(&[2.0, 4.0, 6.0]).unwrap();
        assert_eq!(matrix.values(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn malformed_patterns_are_rejected() {
        assert_eq!(
            Csc::from_pattern(&[1.0], &[0], &[0, 1]),
            Err(CscError::LengthMismatch)
        );
        assert_eq!(
            Csc::from_pattern(&[1.0], &[3], &[0, 1, 1]),
            Err(CscError::InvalidRowIndices)
        );
        assert_eq!(
            Csc::from_pattern(&[1.0], &[0], &[1, 1, 1]),
            Err(CscError::InvalidColumnPointers)
        );
    }

    #[test]
    fn capacity_and_bounds_are_checked() {
        let mut matrix = StaticCscMatrix::<1, 1, 1, f64>::new();
        assert_eq!(matrix.insert(1, 0, 1.0), Err(CscError::IndexOutOfBounds));
        matrix.insert(0, 0, 1.0).unwrap();
        assert_eq!(matrix.insert(0, 0, 2.0), Ok(()));
        assert_eq!(matrix.get(0, 0), Some(&2.0));
        assert_eq!(matrix.set_value(0, 1, 1.0), Err(CscError::IndexOutOfBounds));
    }
}
