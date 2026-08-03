use crate::kernels::matmul;
use crate::{Matrix, MatrixScalar, Real};

/// Cholesky factorization of a symmetric positive-definite matrix.
///
/// The factorization stores the lower-triangular matrix `L` such that
/// `A = L * L.transpose()`. Only the lower triangle of the input is read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cholesky<const D: usize, T> {
    lower: Matrix<D, D, T>,
}

impl<const D: usize, T: Real + MatrixScalar> Cholesky<D, T> {
    /// Computes a Cholesky factorization, returning `None` when the input is
    /// not finite positive-definite.
    #[inline]
    pub fn decompose(matrix: &Matrix<D, D, T>) -> Option<Self> {
        const BLOCKED_THRESHOLD: usize = 64;
        if D < BLOCKED_THRESHOLD {
            return Self::decompose_unblocked(matrix);
        }

        Self::decompose_blocked(matrix)
    }

    #[inline]
    fn decompose_unblocked(matrix: &Matrix<D, D, T>) -> Option<Self> {
        let mut lower = Matrix::zeros();
        let input = matrix.as_slice();
        let output = lower.as_mut_slice();

        for column in 0..D {
            let column_offset = column * D;
            for row in column..D {
                let mut value = input[column_offset + row];
                for previous in 0..column {
                    let previous_offset = previous * D;
                    value =
                        value - output[previous_offset + row] * output[previous_offset + column];
                }

                if row == column {
                    if !value.is_finite() || value <= T::zero() {
                        return None;
                    }
                    output[column_offset + row] = value.sqrt();
                } else {
                    output[column_offset + row] = value / output[column_offset + column];
                }
            }
        }

        Some(Self { lower })
    }

    #[inline]
    fn decompose_blocked(matrix: &Matrix<D, D, T>) -> Option<Self> {
        const BLOCK_SIZE: usize = 8;
        let mut residual = *matrix;
        let mut lower = Matrix::zeros();

        let mut block_start = 0;
        while block_start < D {
            let block_end = core::cmp::min(block_start + BLOCK_SIZE, D);

            for column in block_start..block_end {
                for row in column..block_end {
                    let mut value = residual[(row, column)];
                    for previous in block_start..column {
                        value = value - lower[(row, previous)] * lower[(column, previous)];
                    }

                    if row == column {
                        if !value.is_finite() || value <= T::zero() {
                            return None;
                        }
                        lower[(row, column)] = value.sqrt();
                    } else {
                        let diagonal = lower[(column, column)];
                        let multiplier = value / diagonal;
                        if !multiplier.is_finite() {
                            return None;
                        }
                        lower[(row, column)] = multiplier;
                    }
                }
            }

            for row in block_end..D {
                for column in block_start..block_end {
                    let mut value = residual[(row, column)];
                    for previous in block_start..column {
                        value = value - lower[(row, previous)] * lower[(column, previous)];
                    }
                    let multiplier = value / lower[(column, column)];
                    if !multiplier.is_finite() {
                        return None;
                    }
                    lower[(row, column)] = multiplier;
                }
            }

            Self::update_trailing(&mut residual, &lower, block_start, block_end);

            block_start = block_end;
        }

        Some(Self { lower })
    }

    #[inline]
    fn update_trailing(
        residual: &mut Matrix<D, D, T>,
        lower: &Matrix<D, D, T>,
        block_start: usize,
        block_end: usize,
    ) {
        const TILE_SIZE: usize = 8;
        let mut row_start = block_end;
        while row_start < D {
            let row_end = core::cmp::min(row_start + TILE_SIZE, D);
            let mut column_start = block_end;
            while column_start <= row_start && column_start < D {
                let column_end = core::cmp::min(column_start + TILE_SIZE, D);
                let mut lhs = Matrix::<TILE_SIZE, TILE_SIZE, T>::zeros();
                let mut rhs = Matrix::<TILE_SIZE, TILE_SIZE, T>::zeros();

                for row in row_start..row_end {
                    for index in block_start..block_end {
                        lhs[(row - row_start, index - block_start)] = lower[(row, index)];
                    }
                }
                for column in column_start..column_end {
                    for index in block_start..block_end {
                        rhs[(index - block_start, column - column_start)] = lower[(column, index)];
                    }
                }

                let mut update = Matrix::<TILE_SIZE, TILE_SIZE, T>::zeros();
                matmul(&lhs, &rhs, &mut update);
                for row in row_start..row_end {
                    for column in column_start..column_end {
                        if row >= column {
                            residual[(row, column)] = residual[(row, column)]
                                - update[(row - row_start, column - column_start)];
                        }
                    }
                }
                column_start += TILE_SIZE;
            }
            row_start += TILE_SIZE;
        }
    }

    /// Returns the lower-triangular factor `L`.
    #[inline]
    pub fn lower(&self) -> &Matrix<D, D, T> {
        &self.lower
    }

    /// Solves `A * X = B` using the factorization.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<D, P, T>) -> Matrix<D, P, T> {
        let mut solution = *rhs;
        self.solve_in_place(&mut solution);
        solution
    }

    /// Solves `A * X = B` in place.
    #[inline]
    pub fn solve_in_place<const P: usize>(&self, rhs: &mut Matrix<D, P, T>) {
        let lower = self.lower.as_slice();
        let solution = rhs.as_mut_slice();
        for column in 0..P {
            let column_offset = column * D;
            for pivot in 0..D {
                let value = solution[column_offset + pivot] / lower[pivot * D + pivot];
                solution[column_offset + pivot] = value;
                let mut row = pivot + 1;
                while row + 4 <= D {
                    solution[column_offset + row] =
                        solution[column_offset + row] - lower[pivot * D + row] * value;
                    solution[column_offset + row + 1] =
                        solution[column_offset + row + 1] - lower[pivot * D + row + 1] * value;
                    solution[column_offset + row + 2] =
                        solution[column_offset + row + 2] - lower[pivot * D + row + 2] * value;
                    solution[column_offset + row + 3] =
                        solution[column_offset + row + 3] - lower[pivot * D + row + 3] * value;
                    row += 4;
                }
                while row < D {
                    solution[column_offset + row] =
                        solution[column_offset + row] - lower[pivot * D + row] * value;
                    row += 1;
                }
            }

            for row in (0..D).rev() {
                let mut next = row + 1;
                let mut sum0 = T::zero();
                let mut sum1 = T::zero();
                let mut sum2 = T::zero();
                let mut sum3 = T::zero();
                while next + 4 <= D {
                    sum0 = sum0 + lower[row * D + next] * solution[column_offset + next];
                    sum1 = sum1 + lower[row * D + next + 1] * solution[column_offset + next + 1];
                    sum2 = sum2 + lower[row * D + next + 2] * solution[column_offset + next + 2];
                    sum3 = sum3 + lower[row * D + next + 3] * solution[column_offset + next + 3];
                    next += 4;
                }
                let mut value = solution[column_offset + row] - sum0 - sum1 - sum2 - sum3;
                while next < D {
                    value = value - lower[row * D + next] * solution[column_offset + next];
                    next += 1;
                }
                solution[column_offset + row] = value / lower[row * D + row];
            }
        }
    }

    /// Computes the inverse by solving against the identity matrix.
    #[inline]
    pub fn inverse(&self) -> Matrix<D, D, T> {
        self.solve(&Matrix::eye())
    }
}

impl<const D: usize, T: Real + MatrixScalar> Matrix<D, D, T> {
    /// Computes a Cholesky factorization of this matrix.
    #[inline]
    pub fn cholesky(&self) -> Option<Cholesky<D, T>> {
        Cholesky::decompose(self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{eye, matrix, Matrix};

    #[test]
    fn reconstructs_spd_matrix() {
        let matrix = matrix![
            4.0_f64, 1.0, 1.0;
            1.0, 3.0, 0.0;
            1.0, 0.0, 2.0;
        ];
        let factor = matrix.cholesky().expect("matrix is positive-definite");

        assert_relative_eq!(
            factor.lower() * factor.lower().transpose(),
            matrix,
            max_relative = 1e-12
        );
    }

    #[test]
    fn solves_and_inverts() {
        let matrix = matrix![
            4.0_f64, 1.0, 1.0;
            1.0, 3.0, 0.0;
            1.0, 0.0, 2.0;
        ];
        let rhs = matrix![1.0_f64; 2.0; 3.0];
        let factor = matrix.cholesky().expect("matrix is positive-definite");

        assert_relative_eq!(matrix * factor.solve(&rhs), rhs, max_relative = 1e-12);

        let mut solution = rhs;
        factor.solve_in_place(&mut solution);
        assert_relative_eq!(matrix * solution, rhs, max_relative = 1e-12);
        assert_relative_eq!(
            matrix * factor.inverse(),
            eye!(3, f64),
            max_relative = 1e-12
        );
    }

    #[test]
    fn rejects_non_positive_definite_matrices() {
        assert!(matrix![1.0_f64, 2.0; 2.0, 1.0].cholesky().is_none());
        assert!(matrix![0.0_f64, 0.0; 0.0, 1.0].cholesky().is_none());
        assert!(matrix![f64::NAN, 0.0; 0.0, 1.0].cholesky().is_none());
    }

    #[test]
    fn blocked_factorization_reconstructs_large_spd_matrix() {
        let matrix = Matrix::<64, 64, f64>::from_fn(|row, column| {
            let mut value = if row == column { 64.0 } else { 0.0 };
            for index in 0..64 {
                let left = (row + index + 1) as f64 / 17.0;
                let right = (column + index + 1) as f64 / 17.0;
                value += left * right;
            }
            value
        });
        let factor = matrix.cholesky().expect("matrix is positive-definite");

        assert_relative_eq!(
            factor.lower() * factor.lower().transpose(),
            matrix,
            max_relative = 1e-10
        );
    }
}
