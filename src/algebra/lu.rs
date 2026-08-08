use crate::view::MatrixRead;
use crate::{DecompositionError, Matrix, MatrixScalar, Real};

/// LU decomposition with partial row pivoting.
///
/// This factorization follows Eigen's `PartialPivLU`: it assumes the input is
/// square and invertible. Use a future rank-revealing decomposition when a
/// singularity decision is required.
///
/// # Examples
///
/// ```
/// use stack_algebra::{matrix, PartialPivLu};
///
/// let a = matrix![3.0_f64, 1.0; 1.0, 2.0];
/// let lu: PartialPivLu<2, f64> = a.partial_piv_lu();
/// let rhs = matrix![5.0_f64; 5.0];
/// let x = lu.solve(&rhs);
/// assert!((a * x - rhs).norm() < 1.0e-12);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartialPivLu<const D: usize, T> {
    lower: Matrix<D, D, T>,
    upper: Matrix<D, D, T>,
    permutation: [usize; D],
    row_swaps: usize,
}

impl<const D: usize, T: Real + MatrixScalar> PartialPivLu<D, T> {
    /// Recomputes this partial-pivot factorization in place.
    #[inline]
    pub fn compute(&mut self, matrix: &Matrix<D, D, T>) {
        Self::factorize_into(matrix, self);
    }

    /// Recomputes this factorization with typed failure reporting.
    ///
    /// Unlike [`Self::compute`], this checked path rejects non-finite input or
    /// an intermediate value that exceeds the scalar range. The factor is
    /// replaced only after the complete decomposition succeeds.
    #[inline]
    pub fn try_compute(&mut self, matrix: &Matrix<D, D, T>) -> Result<(), DecompositionError> {
        let factor = Self::try_decompose(matrix)?;
        *self = factor;
        Ok(())
    }

    /// Recomputes this factorization directly from a fixed-size matrix view.
    #[inline]
    pub fn try_compute_view<V>(&mut self, matrix: &V) -> Result<(), DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        *self = Self::try_decompose_view(matrix)?;
        Ok(())
    }

    /// Computes a partial-pivot LU decomposition of `matrix`.
    #[inline]
    pub fn decompose(matrix: &Matrix<D, D, T>) -> Self {
        let mut output = Self {
            lower: Matrix::eye(),
            upper: *matrix,
            permutation: core::array::from_fn(|index| index),
            row_swaps: 0,
        };
        Self::factorize(&mut output);
        output
    }

    /// Computes a partial-pivot LU factorization with typed failure reporting.
    ///
    /// The checked path rejects non-finite input or an intermediate value that
    /// exceeds the scalar range.
    #[inline]
    pub fn try_decompose(matrix: &Matrix<D, D, T>) -> Result<Self, DecompositionError> {
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(DecompositionError::NonFinite);
        }
        let output = Self::decompose(matrix);
        if !output.lower.iter().all(|value| value.is_finite())
            || !output.upper.iter().all(|value| value.is_finite())
        {
            return Err(DecompositionError::NonFinite);
        }
        Ok(output)
    }

    /// Computes a partial-pivot LU factorization directly from a fixed-size
    /// matrix view without materializing a separate owning input matrix.
    #[inline]
    pub fn try_decompose_view<V>(matrix: &V) -> Result<Self, DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        let mut output = Self {
            lower: Matrix::eye(),
            upper: Matrix::zeros(),
            permutation: core::array::from_fn(|index| index),
            row_swaps: 0,
        };
        for column in 0..D {
            for row in 0..D {
                output.upper[(row, column)] = *matrix
                    .get(row, column)
                    .ok_or(DecompositionError::InvalidView)?;
            }
        }
        if output.upper.iter().any(|value| !value.is_finite()) {
            return Err(DecompositionError::NonFinite);
        }
        Self::factorize(&mut output);
        if !output.lower.iter().all(|value| value.is_finite())
            || !output.upper.iter().all(|value| value.is_finite())
        {
            return Err(DecompositionError::NonFinite);
        }
        Ok(output)
    }

    /// Computes a partial-pivot LU decomposition into caller-provided storage.
    #[inline]
    fn factorize_into(matrix: &Matrix<D, D, T>, output: &mut Self) {
        output.lower = Matrix::eye();
        output.upper = *matrix;
        output.permutation = core::array::from_fn(|index| index);
        output.row_swaps = 0;
        Self::factorize(output);
    }

    #[inline]
    fn factorize(output: &mut Self) {
        for diagonal in 0..D {
            let pivot_row = Self::pivot_row(&output.upper, diagonal);
            if pivot_row != diagonal {
                output.upper.swap_rows(diagonal, pivot_row);
                output.permutation.swap(diagonal, pivot_row);
                for column in 0..diagonal {
                    let value = output.lower[(diagonal, column)];
                    output.lower[(diagonal, column)] = output.lower[(pivot_row, column)];
                    output.lower[(pivot_row, column)] = value;
                }
                output.row_swaps += 1;
            }

            let pivot = output.upper[(diagonal, diagonal)];
            if pivot == T::zero() {
                continue;
            }

            for row in (diagonal + 1)..D {
                let multiplier = output.upper[(row, diagonal)] / pivot;
                output.lower[(row, diagonal)] = multiplier;
                output.upper[(row, diagonal)] = T::zero();
            }
            let tail_len = D - diagonal - 1;
            let multiplier_start = diagonal * D + diagonal + 1;
            let multiplier_end = multiplier_start + tail_len;
            for column in (diagonal + 1)..D {
                let target_start = column * D + diagonal + 1;
                let scale = output.upper[(diagonal, column)];
                T::rank_update_sub(
                    &mut output.upper.as_mut_slice()[target_start..target_start + tail_len],
                    &output.lower.as_slice()[multiplier_start..multiplier_end],
                    scale,
                );
            }
        }
    }

    /// Returns the unit lower-triangular factor.
    #[inline]
    pub fn lower(&self) -> &Matrix<D, D, T> {
        &self.lower
    }

    /// Returns the upper-triangular factor.
    #[inline]
    pub fn upper(&self) -> &Matrix<D, D, T> {
        &self.upper
    }

    /// Returns the row-permutation matrix such that `P * A = L * U`.
    ///
    /// The matrix is materialized on demand; the factorization stores only the
    /// compact row-index permutation. Use [`Self::permutation_indices`] when
    /// the index representation is sufficient.
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

    /// Returns the compact row permutation as ordered-to-original indices.
    #[inline]
    pub fn permutation_indices(&self) -> &[usize; D] {
        &self.permutation
    }

    /// Returns the determinant using the recorded row-swap parity.
    #[inline]
    pub fn determinant(&self) -> T {
        let mut determinant = T::one();
        for index in 0..D {
            determinant = determinant * self.upper[(index, index)];
        }
        if self.row_swaps.is_multiple_of(2) {
            determinant
        } else {
            -determinant
        }
    }

    /// Solves `A * x = rhs` using this decomposition.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<D, P, T>) -> Matrix<D, P, T> {
        let mut solution = Matrix::<D, P, T>::zeros();
        self.solve_into(rhs, &mut solution);
        solution
    }

    /// Solves `A * X = B` into a caller-provided output matrix.
    #[inline]
    pub fn solve_into<const P: usize>(&self, rhs: &Matrix<D, P, T>, output: &mut Matrix<D, P, T>) {
        let mut intermediate: Matrix<D, P, T> = Matrix::zeros();
        for column in 0..P {
            for row in 0..D {
                let mut value = rhs[(self.permutation[row], column)];
                for previous in 0..row {
                    value = value - self.lower[(row, previous)] * intermediate[(previous, column)];
                }
                intermediate[(row, column)] = value;
            }
        }

        *output = Matrix::zeros();
        for column in 0..P {
            for row in (0..D).rev() {
                let mut value = intermediate[(row, column)];
                for next in (row + 1)..D {
                    value = value - self.upper[(row, next)] * output[(next, column)];
                }
                output[(row, column)] = value / self.upper[(row, row)];
            }
        }
    }

    /// Computes the inverse by solving against the identity matrix.
    #[inline]
    pub fn inverse(&self) -> Matrix<D, D, T> {
        self.solve(&Matrix::eye())
    }

    #[inline]
    fn pivot_row(upper: &Matrix<D, D, T>, diagonal: usize) -> usize {
        let mut pivot_row = diagonal;
        let mut pivot_abs = upper[(diagonal, diagonal)].abs();
        for row in (diagonal + 1)..D {
            let candidate = upper[(row, diagonal)].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        pivot_row
    }
}

impl<const D: usize, T: Real + MatrixScalar> Matrix<D, D, T> {
    /// Computes the partial-pivot LU factorization of this matrix.
    #[inline]
    pub fn partial_piv_lu(&self) -> PartialPivLu<D, T> {
        PartialPivLu::decompose(self)
    }

    /// Computes a checked partial-pivot LU factorization.
    #[inline]
    pub fn try_partial_piv_lu(&self) -> Result<PartialPivLu<D, T>, DecompositionError> {
        PartialPivLu::try_decompose(self)
    }

    /// Computes the determinant of this matrix.
    #[inline]
    pub fn determinant(&self) -> T {
        self.partial_piv_lu().determinant()
    }

    /// Computes the inverse of this matrix.
    ///
    /// The matrix must be invertible.
    #[inline]
    pub fn inverse(&self) -> Self {
        self.partial_piv_lu().inverse()
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    use crate::{eye, matrix, DecompositionError, Map, Matrix, PartialPivLu};

    #[test]
    fn factorization_reconstructs_matrix() {
        let matrix = matrix![
            1.0_f64, 3.0, 5.0;
            2.0, 4.0, 7.0;
            1.0, 1.0, 0.0;
        ];
        let factor = matrix.partial_piv_lu();
        assert_relative_eq!(
            factor.permutation() * matrix,
            factor.lower() * factor.upper(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn stores_compact_row_permutation() {
        let matrix = matrix![
            0.0_f64, 1.0, 2.0;
            3.0, 4.0, 5.0;
            6.0, 7.0, 8.0;
        ];
        let factor = matrix.partial_piv_lu();
        assert_eq!(factor.permutation_indices(), &[2, 0, 1]);
        assert_relative_eq!(
            factor.permutation() * matrix,
            factor.lower() * factor.upper(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn decomposes_map_and_block_views() {
        let matrix = matrix![
            1.0_f64, 3.0, 5.0;
            2.0, 4.0, 7.0;
            1.0, 1.0, 0.0;
        ];
        let mapped = Map::<3, 3, f64>::from_slice(matrix.as_slice()).unwrap();
        let mapped_factor = PartialPivLu::try_decompose_view(&mapped).unwrap();
        assert_relative_eq!(
            mapped_factor.permutation() * matrix,
            mapped_factor.lower() * mapped_factor.upper(),
            max_relative = 1e-12
        );

        let mut storage = Matrix::<4, 4, f64>::zeros();
        for row in 0..3 {
            for column in 0..3 {
                storage[(row + 1, column)] = matrix[(row, column)];
            }
        }
        let block = storage.block::<3, 3>(1, 0).unwrap();
        let mut reused = mapped_factor;
        reused.try_compute_view(&block).unwrap();
        assert_relative_eq!(
            reused.permutation() * matrix,
            reused.lower() * reused.upper(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn determinant_uses_actual_swap_parity() {
        assert_eq!(eye!(3, f64).determinant(), 1.0);

        let odd_swap = matrix![0.0_f64, 1.0; 1.0, 0.0];
        assert_eq!(odd_swap.determinant(), -1.0);
    }

    #[test]
    fn solve_and_inverse_match_known_values() {
        let matrix = matrix![
            6.0_f64, 2.0, 3.0;
            1.0, 1.0, 1.0;
            0.0, 4.0, 9.0;
        ];
        let rhs = matrix![1.0_f64; 2.0; 3.0];
        let solution = matrix.partial_piv_lu().solve(&rhs);
        assert_relative_eq!(matrix * solution, rhs, max_relative = 1e-12);

        let inverse = matrix.inverse();
        assert_relative_eq!(matrix * inverse, eye!(3, f64), max_relative = 1e-12);
        assert_abs_diff_eq!(matrix.determinant(), 24.0, epsilon = 1e-12);
    }

    #[test]
    fn checked_factorization_rejects_nonfinite_intermediates() {
        let matrix = matrix![
            1.0e308_f64, 1.0e308;
            -1.0e308, 1.0e308;
        ];
        assert_eq!(
            matrix.try_partial_piv_lu(),
            Err(DecompositionError::NonFinite)
        );
        assert_eq!(
            matrix![f64::NAN, 0.0; 0.0, 1.0].try_partial_piv_lu(),
            Err(DecompositionError::NonFinite)
        );

        let stable = matrix![2.0_f64, 1.0; 1.0, 3.0];
        let mut factor = stable.partial_piv_lu();
        let previous = factor;
        assert_eq!(
            factor.try_compute(&matrix),
            Err(DecompositionError::NonFinite)
        );
        assert_eq!(factor, previous);
    }

    #[test]
    fn reuses_caller_provided_factor_storage() {
        let first = matrix![2.0_f64, 1.0; 4.0, 3.0];
        let second = matrix![0.0_f64, 1.0; 2.0, 3.0];
        let mut factor = first.partial_piv_lu();
        factor.compute(&second);
        assert_relative_eq!(
            factor.permutation() * second,
            factor.lower() * factor.upper(),
            max_relative = 1e-12
        );
    }
}
