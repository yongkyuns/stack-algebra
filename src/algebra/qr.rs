use crate::{Matrix, MatrixScalar, Real, Vector};

/// Householder QR factorization of a fixed-size matrix.
///
/// The factorization stores the upper-triangular `R` entries in the upper
/// triangle and the Householder vectors below the diagonal. The vectors and
/// their coefficients are sufficient to apply `Qᵀ` without materializing `Q`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HouseholderQr<const M: usize, const N: usize, T> {
    factors: Matrix<M, N, T>,
    coefficients: Vector<N, T>,
}

impl<const M: usize, const N: usize, T: Real + MatrixScalar> HouseholderQr<M, N, T> {
    /// Computes a Householder QR factorization.
    #[inline]
    pub fn decompose(matrix: &Matrix<M, N, T>) -> Self {
        let mut factors = *matrix;
        let mut coefficients = Vector::<N, T>::zeros();
        let limit = core::cmp::min(M, N);

        for column in 0..limit {
            let mut norm = T::zero();
            for row in column..M {
                let value = factors[(row, column)];
                norm = norm.hypot(value);
            }
            if !norm.is_finite() || norm == T::zero() {
                continue;
            }

            let diagonal = factors[(column, column)];
            let beta = if diagonal >= T::zero() { -norm } else { norm };
            let first = diagonal - beta;
            if first == T::zero() || !first.is_finite() {
                continue;
            }

            factors[(column, column)] = beta;
            for row in (column + 1)..M {
                factors[(row, column)] = factors[(row, column)] / first;
            }

            let coefficient = (beta - diagonal) / beta;
            coefficients[column] = coefficient;
            for trailing_column in (column + 1)..N {
                let mut dot = factors[(column, trailing_column)];
                for row in (column + 1)..M {
                    dot = dot + factors[(row, column)] * factors[(row, trailing_column)];
                }
                let scale = coefficient * dot;
                factors[(column, trailing_column)] = factors[(column, trailing_column)] - scale;
                for row in (column + 1)..M {
                    factors[(row, trailing_column)] =
                        factors[(row, trailing_column)] - scale * factors[(row, column)];
                }
            }
        }

        Self {
            factors,
            coefficients,
        }
    }

    /// Returns the packed factor storage.
    #[inline]
    pub fn factors(&self) -> &Matrix<M, N, T> {
        &self.factors
    }

    /// Returns the upper-triangular `R` factor in the original matrix shape.
    #[inline]
    pub fn r(&self) -> Matrix<M, N, T> {
        let mut upper = Matrix::zeros();
        for column in 0..N {
            for row in 0..core::cmp::min(column + 1, M) {
                upper[(row, column)] = self.factors[(row, column)];
            }
        }
        upper
    }

    /// Applies `Qᵀ` to a matrix without materializing `Q`.
    #[inline]
    pub fn apply_q_transpose<const P: usize>(&self, rhs: &Matrix<M, P, T>) -> Matrix<M, P, T> {
        let mut transformed = *rhs;
        let limit = core::cmp::min(M, N);
        for column in 0..limit {
            let coefficient = self.coefficients[column];
            if coefficient == T::zero() {
                continue;
            }
            for rhs_column in 0..P {
                let mut dot = transformed[(column, rhs_column)];
                for row in (column + 1)..M {
                    dot = dot + self.factors[(row, column)] * transformed[(row, rhs_column)];
                }
                let scale = coefficient * dot;
                transformed[(column, rhs_column)] = transformed[(column, rhs_column)] - scale;
                for row in (column + 1)..M {
                    transformed[(row, rhs_column)] =
                        transformed[(row, rhs_column)] - scale * self.factors[(row, column)];
                }
            }
        }
        transformed
    }

    /// Solves the full-rank least-squares problem `min ||A X - B||₂`.
    ///
    /// This method supports square and overdetermined matrices (`M >= N`). It
    /// returns `None` when the leading `N x N` factor is singular or non-finite.
    #[inline]
    pub fn solve_least_squares<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
    ) -> Option<Matrix<N, P, T>> {
        if M < N {
            return None;
        }

        let transformed = self.apply_q_transpose(rhs);
        let mut solution = Matrix::<N, P, T>::zeros();
        let mut diagonal_scale = T::one();
        for row in 0..N {
            let diagonal = self.factors[(row, row)];
            if !diagonal.is_finite() {
                return None;
            }
            diagonal_scale = diagonal_scale.max(diagonal.abs());
        }
        let dimension = T::from(core::cmp::max(M, N)).unwrap_or(T::one());
        let tolerance = T::epsilon() * dimension * diagonal_scale;
        for rhs_column in 0..P {
            for row in (0..N).rev() {
                let diagonal = self.factors[(row, row)];
                if diagonal.abs() <= tolerance {
                    return None;
                }
                let mut value = transformed[(row, rhs_column)];
                for next in (row + 1)..N {
                    value = value - self.factors[(row, next)] * solution[(next, rhs_column)];
                }
                solution[(row, rhs_column)] = value / diagonal;
            }
        }
        Some(solution)
    }
}

impl<const M: usize, const N: usize, T: Real + MatrixScalar> Matrix<M, N, T> {
    /// Computes a Householder QR factorization of this matrix.
    #[inline]
    pub fn householder_qr(&self) -> HouseholderQr<M, N, T> {
        HouseholderQr::decompose(self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{matrix, Matrix};

    #[test]
    fn reconstructs_square_matrix() {
        let input = matrix![
            12.0_f64, -51.0, 4.0;
            6.0, 167.0, -68.0;
            -4.0, 24.0, -41.0;
        ];
        let qr = input.householder_qr();
        let transformed = qr.apply_q_transpose(&input);
        assert_relative_eq!(transformed, qr.r(), epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    fn solves_overdetermined_system() {
        let input = matrix![
            1.0_f64, 1.0;
            1.0, 2.0;
            1.0, 3.0;
            1.0, 4.0;
        ];
        let rhs = matrix![3.0_f64; 5.0; 7.0; 9.0];
        let solution = input
            .householder_qr()
            .solve_least_squares(&rhs)
            .expect("full-rank least-squares system");
        assert_relative_eq!(solution, matrix![1.0_f64; 2.0], max_relative = 1e-12);
    }

    #[test]
    fn rejects_rank_deficient_system() {
        let input = matrix![
            1.0_f64, 2.0;
            2.0, 4.0;
            3.0, 6.0;
        ];
        let rhs = Matrix::<3, 1, f64>::ones();
        assert!(input.householder_qr().solve_least_squares(&rhs).is_none());
    }
}
