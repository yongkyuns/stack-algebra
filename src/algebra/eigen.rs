use crate::view::MatrixRead;
use crate::{DecompositionError, Matrix, MatrixScalar, Real, Vector};

/// Self-adjoint eigendecomposition of a fixed-size real symmetric matrix.
///
/// The decomposition satisfies `A = V * diag(λ) * Vᵀ`. Eigenvalues are sorted
/// in nondecreasing order and the columns of `V` are orthonormal.
///
/// The input must be finite and symmetric (within the implementation's
/// scaled tolerance). This decomposition is for real self-adjoint matrices;
/// use LU, QR, or SVD for a general non-symmetric matrix.
///
/// # Examples
///
/// ```
/// use stack_algebra::matrix;
///
/// let a = matrix![2.0_f64, 1.0; 1.0, 2.0];
/// let eigen = a.self_adjoint_eigen().expect("symmetric input");
/// assert!((eigen.eigenvalues()[0] - 1.0).abs() < 1.0e-12);
/// assert!((eigen.eigenvalues()[1] - 3.0).abs() < 1.0e-12);
/// assert!((eigen.reconstruct() - a).norm() < 1.0e-12);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfAdjointEigen<const D: usize, T> {
    eigenvalues: Vector<D, T>,
    eigenvectors: Matrix<D, D, T>,
}

/// Caller-owned scratch storage for self-adjoint eigendecomposition sweeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfAdjointEigenWorkspace<const D: usize, T> {
    work: Matrix<D, D, T>,
}

impl<const D: usize, T: Real> SelfAdjointEigenWorkspace<D, T> {
    /// Creates zeroed workspace storage.
    #[inline]
    pub fn new() -> Self {
        Self {
            work: Matrix::zeros(),
        }
    }
}

impl<const D: usize, T: Real> Default for SelfAdjointEigenWorkspace<D, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, T: Real + MatrixScalar> SelfAdjointEigen<D, T> {
    /// Recomputes this eigendecomposition in place with typed failure reporting.
    #[inline]
    pub fn try_compute(&mut self, matrix: &Matrix<D, D, T>) -> Result<(), DecompositionError> {
        let mut workspace = SelfAdjointEigenWorkspace::new();
        Self::try_factorize(matrix, self, &mut workspace.work)
    }

    /// Recomputes this eigendecomposition directly from a fixed-size matrix
    /// view with typed failure reporting.
    #[inline]
    pub fn try_compute_view<V>(&mut self, matrix: &V) -> Result<(), DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        let mut workspace = SelfAdjointEigenWorkspace::new();
        *self = Self::try_decompose_view_with_workspace(matrix, &mut workspace)?;
        Ok(())
    }

    /// Recomputes this eigendecomposition using caller-owned scratch storage.
    #[inline]
    pub fn try_compute_with_workspace(
        &mut self,
        matrix: &Matrix<D, D, T>,
        workspace: &mut SelfAdjointEigenWorkspace<D, T>,
    ) -> Result<(), DecompositionError> {
        Self::try_factorize(matrix, self, &mut workspace.work)
    }

    /// Recomputes this eigendecomposition from a view using caller-owned
    /// scratch storage.
    #[inline]
    pub fn try_compute_view_with_workspace<V>(
        &mut self,
        matrix: &V,
        workspace: &mut SelfAdjointEigenWorkspace<D, T>,
    ) -> Result<(), DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        *self = Self::try_decompose_view_with_workspace(matrix, workspace)?;
        Ok(())
    }

    /// Computes the eigendecomposition of a symmetric matrix.
    ///
    /// The input is accepted when its symmetry error is within a scaled
    /// machine-precision tolerance. Non-finite or materially asymmetric input
    /// returns `None`.
    #[inline]
    pub fn decompose(matrix: &Matrix<D, D, T>) -> Option<Self> {
        Self::try_decompose(matrix).ok()
    }

    /// Computes an eigendecomposition and reports numerical failures
    /// explicitly.
    #[inline]
    pub fn try_decompose(matrix: &Matrix<D, D, T>) -> Result<Self, DecompositionError> {
        let mut output = Self {
            eigenvalues: Vector::<D, T>::zeros(),
            eigenvectors: Matrix::<D, D, T>::eye(),
        };
        let mut workspace = SelfAdjointEigenWorkspace::new();
        Self::try_factorize(matrix, &mut output, &mut workspace.work)?;
        Ok(output)
    }

    /// Computes an eigendecomposition directly from a fixed-size matrix view
    /// without materializing a separate owning input matrix.
    #[inline]
    pub fn try_decompose_view<V>(matrix: &V) -> Result<Self, DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        let mut workspace = SelfAdjointEigenWorkspace::new();
        Self::try_decompose_view_with_workspace(matrix, &mut workspace)
    }

    /// Computes an eigendecomposition from a view using caller-owned scratch.
    #[inline]
    pub fn try_decompose_view_with_workspace<V>(
        matrix: &V,
        workspace: &mut SelfAdjointEigenWorkspace<D, T>,
    ) -> Result<Self, DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        let mut output = Self {
            eigenvalues: Vector::<D, T>::zeros(),
            eigenvectors: Matrix::<D, D, T>::eye(),
        };
        Self::try_factorize(matrix, &mut output, &mut workspace.work)?;
        Ok(output)
    }

    #[inline]
    fn try_factorize<V>(
        matrix: &V,
        output: &mut Self,
        work: &mut Matrix<D, D, T>,
    ) -> Result<(), DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        for column in 0..D {
            for row in 0..D {
                work[(row, column)] = *matrix
                    .get(row, column)
                    .ok_or(DecompositionError::InvalidView)?;
            }
        }
        let dimension = T::from(D).unwrap_or(T::one());
        let tolerance = T::epsilon() * dimension;
        for row in 0..D {
            for column in 0..row {
                let left = work[(row, column)];
                let right = work[(column, row)];
                if !left.is_finite() || !right.is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
                let scale = T::one().max(left.abs()).max(right.abs());
                if (left - right).abs() > tolerance * scale {
                    return Err(DecompositionError::NotSymmetric);
                }
            }
            if !work[(row, row)].is_finite() {
                return Err(DecompositionError::NonFinite);
            }
        }

        // Jacobi's intermediate products are quadratic in the working values.
        // Normalize first so finite inputs near the scalar limit do not overflow
        // during the rotations, then restore the eigenvalue scale at the end.
        let matrix_scale = work
            .iter()
            .copied()
            .fold(T::zero(), |scale, value| scale.max(value.abs()));
        if !matrix_scale.is_finite() {
            return Err(DecompositionError::NonFinite);
        }
        if matrix_scale != T::zero() {
            for value in work.iter_mut() {
                *value = *value / matrix_scale;
            }
        }

        output.eigenvectors = Matrix::<D, D, T>::eye();
        let eigenvectors = &mut output.eigenvectors;
        const MAX_SWEEPS: usize = 64;
        let mut converged = false;

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
                        return Err(DecompositionError::NonFinite);
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

                    Self::rotate_columns(work, p, q, cosine, sine);
                    work[(p, p)] = cosine_squared * diagonal_p - (T::one() + T::one()) * cross
                        + sine_squared * diagonal_q;
                    work[(q, q)] = sine_squared * diagonal_p
                        + (T::one() + T::one()) * cross
                        + cosine_squared * diagonal_q;
                    work[(p, q)] = T::zero();
                    work[(q, p)] = T::zero();
                    for index in 0..D {
                        if index != p && index != q {
                            work[(p, index)] = work[(index, p)];
                            work[(q, index)] = work[(index, q)];
                        }
                    }

                    Self::rotate_columns(eigenvectors, p, q, cosine, sine);
                    changed = true;
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(DecompositionError::NoConvergence);
        }

        let eigenvalues = &mut output.eigenvalues;
        for index in 0..D {
            eigenvalues[index] = work[(index, index)] * matrix_scale;
            if !eigenvalues[index].is_finite() {
                return Err(DecompositionError::NonFinite);
            }
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

        Ok(())
    }

    #[inline]
    fn rotate_columns(
        matrix: &mut Matrix<D, D, T>,
        first: usize,
        second: usize,
        cosine: T,
        sine: T,
    ) {
        let data = matrix.as_mut_slice();
        let (_, after_first) = data.split_at_mut(first * D);
        let (first_and_between, after_second) = after_first.split_at_mut((second - first) * D);
        let (first_column, _) = first_and_between.split_at_mut(D);
        let second_column = &mut after_second[..D];
        T::rotate_columns(first_column, second_column, cosine, sine);
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

    /// Computes the eigendecomposition with a typed failure result.
    #[inline]
    pub fn try_self_adjoint_eigen(&self) -> Result<SelfAdjointEigen<D, T>, DecompositionError> {
        SelfAdjointEigen::try_decompose(self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{matrix, DecompositionError, Map, Matrix, SelfAdjointEigen};

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
    fn reuses_caller_provided_factor_storage() {
        let first = matrix![4.0_f64, 1.0; 1.0, 3.0];
        let second = matrix![2.0_f64, -1.0; -1.0, 5.0];
        let mut factor = first
            .self_adjoint_eigen()
            .expect("first matrix is symmetric");
        factor
            .try_compute(&second)
            .expect("second matrix is symmetric");
        assert_relative_eq!(
            factor.reconstruct(),
            second,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    fn reuses_caller_provided_eigen_workspace() {
        let first = matrix![4.0_f64, 1.0; 1.0, 3.0];
        let second = matrix![2.0_f64, -1.0; -1.0, 5.0];
        let mut factor = first
            .self_adjoint_eigen()
            .expect("first matrix is symmetric");
        let mut workspace = super::SelfAdjointEigenWorkspace::new();
        factor
            .try_compute_with_workspace(&second, &mut workspace)
            .expect("second matrix is symmetric");
        assert_relative_eq!(
            factor.reconstruct(),
            second,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    fn decomposes_map_and_block_views() {
        let matrix = matrix![
            4.0_f64, 1.0, 0.0;
            1.0, 3.0, 1.0;
            0.0, 1.0, 2.0;
        ];
        let mapped = Map::<3, 3, f64>::from_slice(matrix.as_slice()).unwrap();
        let mapped_factor = SelfAdjointEigen::try_decompose_view(&mapped).unwrap();
        assert_relative_eq!(mapped_factor.reconstruct(), matrix, max_relative = 1e-10);

        let mut storage = Matrix::<4, 4, f64>::zeros();
        for row in 0..3 {
            for column in 0..3 {
                storage[(row + 1, column + 1)] = matrix[(row, column)];
            }
        }
        let block = storage.block::<3, 3>(1, 1).unwrap();
        let mut reused = mapped_factor;
        reused.try_compute_view(&block).unwrap();
        assert_relative_eq!(reused.reconstruct(), matrix, max_relative = 1e-10);
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
    fn handles_large_finite_values_without_jacobi_overflow() {
        let input = matrix![
            1.0e308_f64, 2.0e307;
            2.0e307, -1.0e308;
        ];
        let eigen = input
            .try_self_adjoint_eigen()
            .expect("finite symmetric matrix is supported");
        assert_relative_eq!(eigen.reconstruct(), input, max_relative = 1e-12);
        for value in eigen.eigenvalues().iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn rejects_asymmetric_matrix() {
        let input = matrix![1.0_f64, 2.0; 0.0, 1.0];
        assert!(input.self_adjoint_eigen().is_none());
        assert_eq!(
            input.try_self_adjoint_eigen(),
            Err(DecompositionError::NotSymmetric)
        );
    }

    #[test]
    fn rejects_non_finite_matrix() {
        let input = matrix![1.0_f64, f64::NAN; f64::NAN, 1.0];
        assert!(input.self_adjoint_eigen().is_none());
        assert_eq!(
            input.try_self_adjoint_eigen(),
            Err(DecompositionError::NonFinite)
        );
    }
}
