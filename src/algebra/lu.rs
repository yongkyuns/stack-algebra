use crate::{Matrix, MatrixScalar, Real};

/// LU decomposition with partial row pivoting.
///
/// This factorization follows Eigen's `PartialPivLU`: it assumes the input is
/// square and invertible. Use a future rank-revealing decomposition when a
/// singularity decision is required.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartialPivLu<const D: usize, T> {
    lower: Matrix<D, D, T>,
    upper: Matrix<D, D, T>,
    permutation: Matrix<D, D, T>,
    row_swaps: usize,
}

impl<const D: usize, T: Real + MatrixScalar> PartialPivLu<D, T> {
    /// Computes a partial-pivot LU decomposition of `matrix`.
    #[inline]
    pub fn decompose(matrix: &Matrix<D, D, T>) -> Self {
        let mut lower = Matrix::eye();
        let mut upper = *matrix;
        let mut permutation = Matrix::eye();
        let mut row_swaps = 0;

        for diagonal in 0..D {
            let pivot_row = Self::pivot_row(&upper, diagonal);
            if pivot_row != diagonal {
                upper.swap_rows(diagonal, pivot_row);
                permutation.swap_rows(diagonal, pivot_row);
                for column in 0..diagonal {
                    let value = lower[(diagonal, column)];
                    lower[(diagonal, column)] = lower[(pivot_row, column)];
                    lower[(pivot_row, column)] = value;
                }
                row_swaps += 1;
            }

            let pivot = upper[(diagonal, diagonal)];
            if pivot == T::zero() {
                continue;
            }

            for row in (diagonal + 1)..D {
                let multiplier = upper[(row, diagonal)] / pivot;
                lower[(row, diagonal)] = multiplier;
                upper[(row, diagonal)] = T::zero();
                for column in (diagonal + 1)..D {
                    upper[(row, column)] =
                        upper[(row, column)] - multiplier * upper[(diagonal, column)];
                }
            }
        }

        Self {
            lower,
            upper,
            permutation,
            row_swaps,
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
    #[inline]
    pub fn permutation(&self) -> &Matrix<D, D, T> {
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
        let mut permuted_rhs: Matrix<D, P, T> = Matrix::zeros();
        self.permutation.mul_into(rhs, &mut permuted_rhs);

        let mut intermediate: Matrix<D, P, T> = Matrix::zeros();
        for column in 0..P {
            for row in 0..D {
                let mut value = permuted_rhs[(row, column)];
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

    use crate::{eye, matrix};

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
}
