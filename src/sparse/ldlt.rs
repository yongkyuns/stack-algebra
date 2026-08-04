use crate::{Matrix, Real};

use super::cholesky::StaticCscCholeskyPattern;
use super::{SparseCholeskyError, StaticCscMatrix, StaticCscOrdering};

/// Numeric fixed-capacity simplicial sparse LDLᵀ factorization without
/// diagonal pivoting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticCscLdlt<const N: usize, const MAX_L_NNZ: usize, T = f32> {
    pub(crate) lower: StaticCscMatrix<N, N, MAX_L_NNZ, T>,
    pub(crate) diagonal: [T; N],
    pub(crate) ordering: StaticCscOrdering<N>,
}

impl<const N: usize, const MAX_L_NNZ: usize, T: Real> StaticCscLdlt<N, MAX_L_NNZ, T> {
    /// Returns the exact inline storage footprint of this numeric factor.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Performs symbolic analysis and numeric LDLᵀ factorization in one step.
    #[inline]
    pub fn decompose<const MAX_A_NNZ: usize>(
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<Self, SparseCholeskyError> {
        StaticCscCholeskyPattern::analyze(matrix)?.factor_ldlt(matrix)
    }

    /// Performs analysis-time diagonal pivoting followed by sparse LDLᵀ.
    ///
    /// This supports 1x1 symmetric diagonal pivots. Matrices requiring a
    /// 2x2 pivot block still return [`SparseCholeskyError::ZeroPivot`].
    #[inline]
    pub fn decompose_with_diagonal_pivoting<const MAX_A_NNZ: usize>(
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
        threshold: T,
    ) -> Result<Self, SparseCholeskyError> {
        StaticCscCholeskyPattern::analyze_with_diagonal_pivoting(matrix, threshold)?
            .factor_ldlt(matrix)
    }

    /// Returns the reusable symbolic factor pattern.
    #[inline]
    pub fn pattern(&self) -> StaticCscCholeskyPattern<N, MAX_L_NNZ> {
        StaticCscCholeskyPattern::from_lower_with_ordering(*self.lower.pattern(), self.ordering)
    }

    /// Recomputes numeric values using this factor's analyzed sparsity pattern.
    #[inline]
    pub fn recompute<const MAX_A_NNZ: usize>(
        &mut self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        let pattern = self.pattern();
        pattern.refactorize_ldlt(matrix, self)
    }

    /// Recomputes LDLᵀ values from coordinates already transformed by
    /// [`StaticCscCholeskyPattern::prepare_ordered`].
    #[inline]
    pub fn recompute_ordered<const MAX_A_NNZ: usize>(
        &mut self,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        let pattern = self.pattern();
        pattern.factorize_ldlt_ordered(matrix, self)
    }

    /// Returns the unit lower-triangular factor.
    #[inline]
    pub fn lower(&self) -> &StaticCscMatrix<N, N, MAX_L_NNZ, T> {
        &self.lower
    }

    /// Returns the diagonal factor `D`.
    #[inline]
    pub fn diagonal(&self) -> &[T; N] {
        &self.diagonal
    }

    /// Returns the ordering used by this factorization.
    #[inline]
    pub const fn ordering(&self) -> StaticCscOrdering<N> {
        self.ordering
    }

    /// Solves `A * X = B` using the sparse factor.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<N, P, T>) -> Matrix<N, P, T> {
        let mut output = *rhs;
        self.solve_in_place(&mut output);
        output
    }

    /// Solves `A * X = B` into caller-provided output storage.
    #[inline]
    pub fn solve_into<const P: usize>(&self, rhs: &Matrix<N, P, T>, output: &mut Matrix<N, P, T>) {
        *output = *rhs;
        self.solve_in_place(output);
    }

    /// Solves `A * X = B` in place using sparse LDLᵀ substitution.
    #[inline]
    pub fn solve_in_place<const P: usize>(&self, rhs: &mut Matrix<N, P, T>) {
        if !self.ordering.is_identity() {
            let mut permuted = Matrix::<N, P, T>::zeros();
            for ordered in 0..N {
                let original = self.ordering.permutation()[ordered];
                for column in 0..P {
                    permuted[(ordered, column)] = rhs[(original, column)];
                }
            }
            self.solve_natural_in_place(&mut permuted);
            for ordered in 0..N {
                let original = self.ordering.permutation()[ordered];
                for column in 0..P {
                    rhs[(original, column)] = permuted[(ordered, column)];
                }
            }
            return;
        }
        self.solve_natural_in_place(rhs);
    }

    fn solve_natural_in_place<const P: usize>(&self, rhs: &mut Matrix<N, P, T>) {
        for row in 0..N {
            let start = self.lower.column_starts()[row];
            let end = self.lower.column_end(row).unwrap_or(self.lower.nnz());
            for index in (start + 1)..end {
                let target = self.lower.row_indices()[index];
                let value = self.lower.values()[index];
                for column in 0..P {
                    rhs[(target, column)] = rhs[(target, column)] - value * rhs[(row, column)];
                }
            }
        }

        for row in 0..N {
            let diagonal = self.diagonal[row];
            for column in 0..P {
                rhs[(row, column)] = rhs[(row, column)] / diagonal;
            }
        }

        for row in (0..N).rev() {
            let start = self.lower.column_starts()[row];
            let end = self.lower.column_end(row).unwrap_or(self.lower.nnz());
            for index in (start + 1)..end {
                let source = self.lower.row_indices()[index];
                let value = self.lower.values()[index];
                for column in 0..P {
                    rhs[(row, column)] = rhs[(row, column)] - value * rhs[(source, column)];
                }
            }
        }
    }
}
