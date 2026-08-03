use crate::{Matrix, MatrixScalar, Real, Vector};

/// Self-adjoint eigendecomposition of a fixed-size real symmetric matrix.
///
/// The decomposition satisfies `A = V * diag(λ) * Vᵀ`. Eigenvalues are sorted
/// in nondecreasing order and the columns of `V` are orthonormal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfAdjointEigen<const D: usize, T> {
    eigenvalues: Vector<D, T>,
    eigenvectors: Matrix<D, D, T>,
}

impl<const D: usize, T: Real + MatrixScalar> SelfAdjointEigen<D, T> {
    /// Computes the eigendecomposition of a symmetric matrix.
    ///
    /// The input is accepted when its symmetry error is within a scaled
    /// machine-precision tolerance. Non-finite or materially asymmetric input
    /// returns `None`.
    #[inline]
    pub fn decompose(matrix: &Matrix<D, D, T>) -> Option<Self> {
        let dimension = T::from(D).unwrap_or(T::one());
        let tolerance = T::epsilon() * dimension;
        for row in 0..D {
            for column in 0..row {
                let left = matrix[(row, column)];
                let right = matrix[(column, row)];
                if !left.is_finite() || !right.is_finite() {
                    return None;
                }
                let scale = T::one().max(left.abs()).max(right.abs());
                if (left - right).abs() > tolerance * scale {
                    return None;
                }
            }
            if !matrix[(row, row)].is_finite() {
                return None;
            }
        }

        let mut work = *matrix;
        let mut eigenvectors = Matrix::<D, D, T>::eye();
        const MAX_SWEEPS: usize = 64;

        for _ in 0..MAX_SWEEPS {
            let mut changed = false;
            for p in 0..D {
                for q in (p + 1)..D {
                    let diagonal_p = work[(p, p)];
                    let diagonal_q = work[(q, q)];
                    let off_diagonal = work[(p, q)];
                    let scale = T::one()
                        .max(diagonal_p.abs())
                        .max(diagonal_q.abs())
                        .max(off_diagonal.abs());
                    if off_diagonal.abs() <= tolerance * scale {
                        continue;
                    }

                    let tau = (diagonal_q - diagonal_p) / (off_diagonal + off_diagonal);
                    let root = (T::one() + tau * tau).sqrt();
                    let denominator = tau.abs() + root;
                    if !denominator.is_finite() || denominator == T::zero() {
                        return None;
                    }
                    let tangent = if tau >= T::zero() {
                        T::one() / denominator
                    } else {
                        -T::one() / denominator
                    };
                    let cosine = T::one() / (T::one() + tangent * tangent).sqrt();
                    let sine = tangent * cosine;
                    let cosine_squared = cosine * cosine;
                    let sine_squared = sine * sine;
                    let cross = (sine * cosine) * off_diagonal;

                    work[(p, p)] = cosine_squared * diagonal_p - (T::one() + T::one()) * cross
                        + sine_squared * diagonal_q;
                    work[(q, q)] = sine_squared * diagonal_p
                        + (T::one() + T::one()) * cross
                        + cosine_squared * diagonal_q;
                    work[(p, q)] = T::zero();
                    work[(q, p)] = T::zero();

                    for index in 0..D {
                        if index == p || index == q {
                            continue;
                        }
                        let value_p = work[(index, p)];
                        let value_q = work[(index, q)];
                        let rotated_p = cosine * value_p - sine * value_q;
                        let rotated_q = sine * value_p + cosine * value_q;
                        work[(index, p)] = rotated_p;
                        work[(p, index)] = rotated_p;
                        work[(index, q)] = rotated_q;
                        work[(q, index)] = rotated_q;
                    }

                    for row in 0..D {
                        let value_p = eigenvectors[(row, p)];
                        let value_q = eigenvectors[(row, q)];
                        eigenvectors[(row, p)] = cosine * value_p - sine * value_q;
                        eigenvectors[(row, q)] = sine * value_p + cosine * value_q;
                    }
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut eigenvalues = Vector::<D, T>::zeros();
        for index in 0..D {
            eigenvalues[index] = work[(index, index)];
        }
        for index in 0..D {
            let mut smallest = index;
            for candidate in (index + 1)..D {
                if eigenvalues[candidate] < eigenvalues[smallest] {
                    smallest = candidate;
                }
            }
            if smallest != index {
                eigenvalues.swap_rows(index, smallest);
                eigenvectors.swap_columns(index, smallest);
            }
        }

        Some(Self {
            eigenvalues,
            eigenvectors,
        })
    }

    /// Returns the eigenvalues in nondecreasing order.
    #[inline]
    pub fn eigenvalues(&self) -> &Vector<D, T> {
        &self.eigenvalues
    }

    /// Returns the orthonormal eigenvectors as columns.
    #[inline]
    pub fn eigenvectors(&self) -> &Matrix<D, D, T> {
        &self.eigenvectors
    }

    /// Reconstructs the original symmetric matrix from the factors.
    #[inline]
    pub fn reconstruct(&self) -> Matrix<D, D, T> {
        let mut matrix = Matrix::<D, D, T>::zeros();
        for row in 0..D {
            for column in 0..D {
                for eigenvalue in 0..D {
                    matrix[(row, column)] = matrix[(row, column)]
                        + self.eigenvectors[(row, eigenvalue)]
                            * self.eigenvalues[eigenvalue]
                            * self.eigenvectors[(column, eigenvalue)];
                }
            }
        }
        matrix
    }
}

impl<const D: usize, T: Real + MatrixScalar> Matrix<D, D, T> {
    /// Computes the eigendecomposition of a real symmetric matrix.
    #[inline]
    pub fn self_adjoint_eigen(&self) -> Option<SelfAdjointEigen<D, T>> {
        SelfAdjointEigen::decompose(self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{matrix, Matrix};

    #[test]
    fn reconstructs_symmetric_matrix() {
        let input = matrix![
            4.0_f64, 1.0, 2.0;
            1.0, 3.0, 0.5;
            2.0, 0.5, 5.0;
        ];
        let eigen = input.self_adjoint_eigen().expect("matrix is symmetric");
        assert_relative_eq!(
            eigen.reconstruct(),
            input,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
        for index in 1..3 {
            assert!(eigen.eigenvalues()[index - 1] <= eigen.eigenvalues()[index]);
        }
        for left in 0..3 {
            for right in 0..3 {
                let mut dot = 0.0;
                for row in 0..3 {
                    dot += eigen.eigenvectors()[(row, left)] * eigen.eigenvectors()[(row, right)];
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert_relative_eq!(dot, expected, epsilon = 1e-10, max_relative = 1e-10);
            }
        }
    }

    #[test]
    fn handles_repeated_eigenvalues() {
        let input =
            Matrix::<3, 3, f64>::from_rows([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]);
        let eigen = input.self_adjoint_eigen().expect("matrix is symmetric");
        assert_relative_eq!(
            eigen.reconstruct(),
            input,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
        assert_relative_eq!(eigen.eigenvalues()[0], -1.0, epsilon = 1e-12);
        assert_relative_eq!(eigen.eigenvalues()[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(eigen.eigenvalues()[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn rejects_asymmetric_matrix() {
        let input = matrix![1.0_f64, 2.0; 0.0, 1.0];
        assert!(input.self_adjoint_eigen().is_none());
    }

    #[test]
    fn rejects_non_finite_matrix() {
        let input = matrix![1.0_f64, f64::NAN; f64::NAN, 1.0];
        assert!(input.self_adjoint_eigen().is_none());
    }
}
