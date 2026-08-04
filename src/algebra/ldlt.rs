use crate::kernels::MatmulBackend;
use crate::{DecompositionError, Matrix, MatrixScalar, Real};

/// Symmetric LDLᵀ factorization with Eigen-compatible diagonal pivoting.
///
/// The factorization has the form `P * A * Pᵀ = L * D * Lᵀ`, where `D` is
/// diagonal, `L` is unit lower-triangular, and `P` is represented internally
/// by a fixed-size permutation index array. The factor matrix stores `L` below
/// the diagonal and `D` on the diagonal, matching Eigen's compact layout.
/// This is the diagonal-`D` variant; matrices that require a 2x2 pivot block
/// return [`DecompositionError::ZeroPivot`] rather than using a
/// Bunch–Kaufman block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ldlt<const D: usize, T> {
    factor: Matrix<D, D, T>,
    permutation: [usize; D],
}

impl<const D: usize, T: Real + MatrixScalar> Ldlt<D, T> {
    /// Recomputes this diagonal-pivot factorization in place.
    #[inline]
    pub fn try_compute(&mut self, matrix: &Matrix<D, D, T>) -> Result<(), DecompositionError> {
        Self::try_factorize_into(matrix, self)
    }

    /// Recomputes this no-pivot factorization in place.
    #[inline]
    pub fn try_compute_no_pivot(
        &mut self,
        matrix: &Matrix<D, D, T>,
    ) -> Result<(), DecompositionError> {
        Self::try_factorize_no_pivot_into(matrix, self)
    }

    /// Computes an Eigen-compatible symmetric diagonal-pivot LDLᵀ decomposition.
    ///
    /// Returns `None` for non-finite or singular input. The input is expected
    /// to be symmetric; only its lower triangle is read.
    #[inline]
    pub fn decompose(matrix: &Matrix<D, D, T>) -> Option<Self> {
        Self::try_decompose(matrix).ok()
    }

    /// Computes a symmetric diagonal-pivot LDLᵀ decomposition with a typed
    /// failure result.
    #[inline]
    pub fn try_decompose(matrix: &Matrix<D, D, T>) -> Result<Self, DecompositionError> {
        let mut output = Self {
            factor: *matrix,
            permutation: core::array::from_fn(|index| index),
        };
        Self::decompose_impl::<true>(&mut output)?;
        Ok(output)
    }

    /// Computes a symmetric diagonal-pivot LDLᵀ decomposition into caller-provided storage.
    /// On failure, `output` may contain a partial factorization.
    #[inline]
    fn try_factorize_into(
        matrix: &Matrix<D, D, T>,
        output: &mut Self,
    ) -> Result<(), DecompositionError> {
        output.factor = *matrix;
        output.permutation = core::array::from_fn(|index| index);
        Self::decompose_impl::<true>(output)
    }

    /// Computes an LDLᵀ decomposition without diagonal pivoting.
    ///
    /// This avoids pivot-search and swap overhead, but requires every
    /// leading diagonal value to remain finite and nonzero. Use `decompose`
    /// when the matrix may need pivoting for numerical stability.
    #[inline]
    pub fn decompose_no_pivot(matrix: &Matrix<D, D, T>) -> Option<Self> {
        Self::try_decompose_no_pivot(matrix).ok()
    }

    /// Computes an LDLᵀ decomposition without diagonal pivoting with a typed
    /// failure result.
    #[inline]
    pub fn try_decompose_no_pivot(matrix: &Matrix<D, D, T>) -> Result<Self, DecompositionError> {
        let mut output = Self {
            factor: *matrix,
            permutation: core::array::from_fn(|index| index),
        };
        Self::decompose_impl::<false>(&mut output)?;
        Ok(output)
    }

    /// Computes an LDLᵀ decomposition without pivoting into caller-provided storage.
    /// On failure, `output` may contain a partial factorization.
    #[inline]
    fn try_factorize_no_pivot_into(
        matrix: &Matrix<D, D, T>,
        output: &mut Self,
    ) -> Result<(), DecompositionError> {
        output.factor = *matrix;
        output.permutation = core::array::from_fn(|index| index);
        Self::decompose_impl::<false>(output)
    }

    #[inline]
    fn decompose_impl<const PIVOTING: bool>(output: &mut Self) -> Result<(), DecompositionError> {
        let factor = &mut output.factor;
        let permutation = &mut output.permutation;

        const BLOCK_SIZE: usize = 16;
        let mut block_start = 0;
        while block_start < D {
            let block_end = core::cmp::min(block_start + BLOCK_SIZE, D);
            for diagonal in block_start..block_end {
                let pivot = if PIVOTING {
                    Self::pivot_index(factor, diagonal)
                } else {
                    diagonal
                };
                let pivot_value = factor[(pivot, pivot)];
                if !pivot_value.is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
                if pivot_value == T::zero() {
                    return Err(DecompositionError::ZeroPivot);
                }
                if pivot != diagonal {
                    Self::swap_lower_rows(factor, diagonal, pivot, diagonal);
                    Self::swap_symmetric_lower(factor, diagonal, pivot, diagonal);
                    permutation.swap(diagonal, pivot);
                }

                let diagonal_value = factor[(diagonal, diagonal)];
                if !diagonal_value.is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
                if diagonal_value == T::zero() {
                    return Err(DecompositionError::ZeroPivot);
                }
                factor[(diagonal, diagonal)] = diagonal_value;

                {
                    let data = factor.as_mut_slice();
                    let column = &mut data[diagonal * D + diagonal + 1..diagonal * D + D];
                    T::Matmul::scale_divide(column, diagonal_value);
                    if column.iter().any(|value| !value.is_finite()) {
                        return Err(DecompositionError::NonFinite);
                    }
                }

                for column in (diagonal + 1)..block_end {
                    let scale = diagonal_value * factor[(column, diagonal)];
                    let data = factor.as_mut_slice();
                    let column_offset = column * D;
                    let (prefix, suffix) = data.split_at_mut(column_offset);
                    let source = &prefix[diagonal * D + column..diagonal * D + D];
                    let target = &mut suffix[column..D];
                    T::Matmul::rank_update_sub(target, source, scale);
                }
            }

            T::Matmul::symmetric_rank_k_update(factor, block_start, block_end);
            block_start = block_end;
        }

        Ok(())
    }

    #[inline]
    fn swap_lower_rows(matrix: &mut Matrix<D, D, T>, first: usize, second: usize, columns: usize) {
        for column in 0..columns {
            let value = matrix[(first, column)];
            matrix[(first, column)] = matrix[(second, column)];
            matrix[(second, column)] = value;
        }
    }

    #[inline]
    fn swap_symmetric_lower(
        matrix: &mut Matrix<D, D, T>,
        first: usize,
        second: usize,
        start: usize,
    ) {
        let first_diagonal = matrix[(first, first)];
        matrix[(first, first)] = matrix[(second, second)];
        matrix[(second, second)] = first_diagonal;
        let between = matrix[(first.max(second), first.min(second))];
        for index in start..D {
            if index == first || index == second {
                continue;
            }
            let first_value = matrix[(first.max(index), first.min(index))];
            let second_value = matrix[(second.max(index), second.min(index))];
            matrix[(first.max(index), first.min(index))] = second_value;
            matrix[(second.max(index), second.min(index))] = first_value;
        }
        matrix[(first.max(second), first.min(second))] = between;
    }

    /// Returns the unit lower-triangular factor `L`.
    #[inline]
    pub fn lower(&self) -> Matrix<D, D, T> {
        Matrix::from_fn(|row, column| {
            if row > column {
                self.factor[(row, column)]
            } else if row == column {
                T::one()
            } else {
                T::zero()
            }
        })
    }

    /// Returns the diagonal factor `D` as a column matrix.
    #[inline]
    pub fn diagonal(&self) -> Matrix<D, 1, T> {
        Matrix::from_fn(|row, _| self.factor[(row, row)])
    }

    /// Returns the diagonal factor `D` as a square matrix.
    #[inline]
    pub fn diagonal_matrix(&self) -> Matrix<D, D, T> {
        Matrix::from_fn(|row, column| {
            if row == column {
                self.factor[(row, row)]
            } else {
                T::zero()
            }
        })
    }

    /// Returns the permutation indices used by the factorization.
    #[inline]
    pub fn permutation_indices(&self) -> &[usize; D] {
        &self.permutation
    }

    /// Returns the permutation matrix `P` as a fixed-size value.
    #[inline]
    pub fn permutation(&self) -> Matrix<D, D, T> {
        Matrix::from_fn(|row, column| {
            if self.permutation[row] == column {
                T::one()
            } else {
                T::zero()
            }
        })
    }

    /// Solves `A * X = B` using this decomposition.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<D, P, T>) -> Matrix<D, P, T> {
        let mut solution = Matrix::<D, P, T>::zeros();
        self.solve_into(rhs, &mut solution);
        solution
    }

    /// Solves `A * X = B` into a caller-provided output matrix.
    #[inline]
    pub fn solve_into<const P: usize>(&self, rhs: &Matrix<D, P, T>, output: &mut Matrix<D, P, T>) {
        *output = *rhs;
        self.solve_in_place(output);
    }

    /// Solves `A * X = B` in place.
    #[inline]
    pub fn solve_in_place<const P: usize>(&self, rhs: &mut Matrix<D, P, T>) {
        let mut transformed: Matrix<D, P, T> = Matrix::zeros();
        for column in 0..P {
            for row in 0..D {
                transformed[(row, column)] = rhs[(self.permutation[row], column)];
            }
        }

        for column in 0..P {
            for row in 0..D {
                let mut value = transformed[(row, column)];
                for previous in 0..row {
                    value = value - self.factor[(row, previous)] * transformed[(previous, column)];
                }
                transformed[(row, column)] = value;
            }

            for row in 0..D {
                transformed[(row, column)] = transformed[(row, column)] / self.factor[(row, row)];
            }

            for row in (0..D).rev() {
                let mut value = transformed[(row, column)];
                for next in (row + 1)..D {
                    value = value - self.factor[(next, row)] * transformed[(next, column)];
                }
                transformed[(row, column)] = value;
            }
        }

        for column in 0..P {
            for row in 0..D {
                rhs[(self.permutation[row], column)] = transformed[(row, column)];
            }
        }
    }

    #[inline]
    fn pivot_index(matrix: &Matrix<D, D, T>, start: usize) -> usize {
        let data = matrix.as_slice();
        let mut pivot = start;
        let mut magnitude = data[start * D + start].abs();
        for index in (start + 1)..D {
            let candidate = data[index * D + index].abs();
            if candidate > magnitude {
                magnitude = candidate;
                pivot = index;
            }
        }
        pivot
    }
}

impl<const D: usize, T: Real + MatrixScalar> Matrix<D, D, T> {
    /// Computes a symmetric diagonal-pivot LDLᵀ factorization with a typed
    /// failure result.
    #[inline]
    pub fn try_ldlt(&self) -> Result<Ldlt<D, T>, DecompositionError> {
        Ldlt::try_decompose(self)
    }

    /// Computes a symmetric diagonal-pivot LDLᵀ factorization of this matrix.
    #[inline]
    pub fn ldlt(&self) -> Option<Ldlt<D, T>> {
        Ldlt::decompose(self)
    }

    /// Computes an LDLᵀ factorization without diagonal pivoting.
    #[inline]
    pub fn ldlt_no_pivot(&self) -> Option<Ldlt<D, T>> {
        Ldlt::decompose_no_pivot(self)
    }

    /// Computes an LDLᵀ factorization without diagonal pivoting with a typed
    /// failure result.
    #[inline]
    pub fn try_ldlt_no_pivot(&self) -> Result<Ldlt<D, T>, DecompositionError> {
        Ldlt::try_decompose_no_pivot(self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{matrix, DecompositionError, Matrix};

    #[test]
    fn typed_errors_distinguish_ldlt_failures() {
        assert_eq!(
            matrix![f64::NAN].try_ldlt(),
            Err(DecompositionError::NonFinite)
        );
        assert_eq!(
            matrix![0.0_f64].try_ldlt_no_pivot(),
            Err(DecompositionError::ZeroPivot)
        );
        assert!(matrix![0.0_f64].ldlt_no_pivot().is_none());
    }

    #[test]
    fn reconstructs_symmetric_indefinite_matrix() {
        let matrix = matrix![
            0.0_f64, 1.0, 2.0;
            1.0, 3.0, 4.0;
            2.0, 4.0, 5.0;
        ];
        let factor = matrix.ldlt().expect("matrix is nonsingular");
        let transformed = factor.permutation() * matrix * factor.permutation().transpose();
        assert_relative_eq!(
            transformed,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn solves_with_diagonal_pivoting() {
        let matrix = matrix![-1.0_f64, 2.0; 2.0, 3.0];
        let rhs = matrix![1.0_f64; 4.0];
        let factor = matrix.ldlt().expect("matrix is nonsingular");
        assert_relative_eq!(matrix * factor.solve(&rhs), rhs, max_relative = 1e-12);
    }

    #[test]
    fn reuses_caller_provided_factor_storage() {
        let first = matrix![4.0_f64, 1.0; 1.0, 3.0];
        let second = matrix![0.0_f64, 2.0; 2.0, 3.0];
        let mut factor = first.ldlt().expect("first matrix is nonsingular");
        factor
            .try_compute(&second)
            .expect("second matrix is nonsingular");
        let transformed = factor.permutation() * second * factor.permutation().transpose();
        assert_relative_eq!(
            transformed,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn no_pivot_reconstructs_stable_matrix() {
        let matrix = matrix![
            4.0_f64, 1.0, 0.5;
            1.0, 3.0, 0.25;
            0.5, 0.25, 2.0;
        ];
        let factor = matrix.ldlt_no_pivot().expect("matrix is nonsingular");
        assert_eq!(factor.permutation_indices(), &[0, 1, 2]);
        assert_relative_eq!(
            matrix,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn no_pivot_rejects_zero_leading_pivot() {
        let matrix = matrix![0.0_f64, 1.0; 1.0, 2.0];
        assert!(matrix.ldlt_no_pivot().is_none());
        assert!(matrix.ldlt().is_some());
    }

    #[test]
    fn diagonal_pivoting_rejects_matrix_requiring_two_by_two_pivot() {
        let matrix = matrix![0.0_f64, 1.0; 1.0, 0.0];
        assert_eq!(matrix.try_ldlt(), Err(DecompositionError::ZeroPivot));
        assert!(matrix.ldlt().is_none());
    }

    #[test]
    fn reconstructs_blocked_symmetric_indefinite_matrix() {
        let matrix = Matrix::<16, 16, f64>::from_fn(|row, column| {
            if row == column {
                if row % 2 == 0 {
                    -16.0
                } else {
                    17.0
                }
            } else {
                (row + column + 1) as f64 / 29.0
            }
        });
        let factor = matrix.ldlt().expect("matrix is nonsingular");
        let transformed = factor.permutation() * matrix * factor.permutation().transpose();
        assert_relative_eq!(
            transformed,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn rejects_singular_matrix() {
        assert!(matrix![1.0_f64, 2.0; 2.0, 4.0].ldlt().is_none());
    }
}
