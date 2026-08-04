use crate::Zero;

mod cholesky;
mod errors;
mod ldlt;
mod ordering;
mod storage;

pub use cholesky::{StaticCscCholesky, StaticCscCholeskyPattern};
pub use errors::{CscError, SparseCholeskyError};
pub use ldlt::StaticCscLdlt;
pub use ordering::StaticCscOrdering;
pub use storage::{StaticCscMatrix, StaticCscPattern};

/// Symbolic CSC pattern shared by sparse LLT and no-pivot LDLᵀ.
pub type StaticCscLdltPattern<const N: usize, const MAX_L_NNZ: usize> =
    StaticCscCholeskyPattern<N, MAX_L_NNZ>;

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
