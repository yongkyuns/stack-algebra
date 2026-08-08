use crate::{Ldlt, Matrix, MatrixScalar, Real};

use super::cholesky::StaticCscCholeskyPattern;
use super::{default_ldlt_threshold, SparseCholeskyError, StaticCscMatrix, StaticCscOrdering};

/// Numeric fixed-capacity simplicial sparse LDLᵀ factorization without
/// diagonal pivoting.
///
/// `StaticCscLdlt` stores a unit lower factor and diagonal `D` such that
/// `A = L * D * Lᵀ`. Use [`Self::decompose_with_diagonal_pivoting`] when
/// diagonal scaling is needed; that mode selects a fixed ordering during
/// analysis and still reports [`SparseCholeskyError::ZeroPivot`] when a
/// 2-by-2 pivot is required. Use [`StaticCscMatrix::try_dense_ldlt`] for an
/// explicit bounded dense fallback when global scalar pivoting is required.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticCscLdlt<const N: usize, const MAX_L_NNZ: usize, T = f32> {
    pub(crate) lower: StaticCscMatrix<N, N, MAX_L_NNZ, T>,
    pub(crate) diagonal: [T; N],
    pub(crate) ordering: StaticCscOrdering<N>,
}

/// Sparse LDLᵀ with an explicit bounded dense fallback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticCscLdltFactor<const N: usize, const MAX_L_NNZ: usize, T = f32> {
    /// Native sparse factorization using the analyzed 1x1 pivot model.
    Sparse(StaticCscLdlt<N, MAX_L_NNZ, T>),
    /// Dense fixed-size Bunch–Kaufman fallback for global 2x2 pivots.
    Dense(Ldlt<N, T>),
}

impl<const N: usize, const MAX_L_NNZ: usize, T> StaticCscLdltFactor<N, MAX_L_NNZ, T>
where
    T: Real + MatrixScalar,
{
    /// Returns the exact inline storage footprint of this factor.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Returns whether the dense fallback is active.
    #[inline]
    pub const fn uses_dense_fallback(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// Solves `A * X = B` using the selected factorization.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<N, P, T>) -> Matrix<N, P, T> {
        match self {
            Self::Sparse(factor) => factor.solve(rhs),
            Self::Dense(factor) => factor.solve(rhs),
        }
    }

    /// Solves into caller-provided output storage.
    #[inline]
    pub fn solve_into<const P: usize>(&self, rhs: &Matrix<N, P, T>, output: &mut Matrix<N, P, T>) {
        match self {
            Self::Sparse(factor) => factor.solve_into(rhs, output),
            Self::Dense(factor) => factor.solve_into(rhs, output),
        }
    }

    /// Solves `A * X = B` in place using the selected factorization.
    #[inline]
    pub fn solve_in_place<const P: usize>(&self, rhs: &mut Matrix<N, P, T>) {
        match self {
            Self::Sparse(factor) => factor.solve_in_place(rhs),
            Self::Dense(factor) => factor.solve_in_place(rhs),
        }
    }

    /// Recomputes numeric values and switches to the dense fallback if a
    /// later sparse update requires a global 2×2 pivot.
    #[inline]
    pub fn recompute_with_dense_fallback<const MAX_A_NNZ: usize>(
        &mut self,
        matrix: &super::StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
        let result = match self {
            Self::Sparse(factor) => factor.recompute(matrix),
            Self::Dense(factor) => {
                *factor = matrix.try_dense_ldlt()?;
                return Ok(());
            }
        };

        match result {
            Ok(()) => Ok(()),
            Err(SparseCholeskyError::ZeroPivot) => {
                match StaticCscLdlt::decompose_with_diagonal_pivoting(
                    matrix,
                    default_ldlt_threshold(matrix),
                ) {
                    Ok(factor) => {
                        *self = Self::Sparse(factor);
                        Ok(())
                    }
                    Err(SparseCholeskyError::ZeroPivot) => {
                        *self = Self::Dense(matrix.try_dense_ldlt()?);
                        Ok(())
                    }
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        }
    }
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

    /// Performs minimum-degree symbolic analysis and numeric LDLᵀ factorization.
    #[inline]
    pub fn decompose_with_minimum_degree<const MAX_A_NNZ: usize>(
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<Self, SparseCholeskyError> {
        StaticCscCholeskyPattern::analyze_with_minimum_degree(matrix)?.factor_ldlt(matrix)
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

    /// Recomputes numeric values using caller-retained symbolic metadata.
    ///
    /// This avoids rebuilding the update schedule on every iteration. Retain
    /// the [`StaticCscCholeskyPattern`] returned by symbolic analysis when a
    /// factor is updated repeatedly.
    #[inline]
    pub fn recompute_with_pattern<const MAX_A_NNZ: usize>(
        &mut self,
        pattern: &StaticCscCholeskyPattern<N, MAX_L_NNZ>,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
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

    /// Recomputes ordered coordinates using caller-retained symbolic metadata.
    #[inline]
    pub fn recompute_ordered_with_pattern<const MAX_A_NNZ: usize>(
        &mut self,
        pattern: &StaticCscCholeskyPattern<N, MAX_L_NNZ>,
        matrix: &StaticCscMatrix<N, N, MAX_A_NNZ, T>,
    ) -> Result<(), SparseCholeskyError> {
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
