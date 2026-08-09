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

        // Normalize first so finite inputs near the scalar limit do not overflow
        // during tridiagonalization, then restore the eigenvalue scale at the end.
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
        let mut diagonal = Vector::<D, T>::zeros();
        let mut off_diagonal = Vector::<D, T>::zeros();
        let mut reflector = Vector::<D, T>::zeros();
        let mut update = Vector::<D, T>::zeros();

        for column in 0..D.saturating_sub(2) {
            let start = column + 1;
            let mut scale = T::zero();
            for row in start..D {
                scale = scale.max(work[(row, column)].abs());
            }
            if scale == T::zero() {
                off_diagonal[start] = T::zero();
                continue;
            }

            let mut norm_squared = T::zero();
            for row in start..D {
                let value = work[(row, column)] / scale;
                reflector[row] = value;
                norm_squared = norm_squared + value * value;
            }
            let norm = norm_squared.sqrt();
            let first = reflector[start];
            let alpha = if first >= T::zero() { -norm } else { norm };
            reflector[start] = first - alpha;
            let mut reflector_squared = T::zero();
            for row in start..D {
                reflector_squared = reflector_squared + reflector[row] * reflector[row];
            }
            if reflector_squared == T::zero() || !reflector_squared.is_finite() {
                return Err(DecompositionError::NonFinite);
            }
            let beta = (T::one() + T::one()) / reflector_squared;
            let subdiagonal = scale * alpha;
            if !beta.is_finite() || !subdiagonal.is_finite() {
                return Err(DecompositionError::NonFinite);
            }
            off_diagonal[start - 1] = subdiagonal;

            for row in start..D {
                update[row] = T::zero();
            }
            for column_index in start..D {
                let coefficient = reflector[column_index];
                let column_start = column_index * D + start;
                let column = &work.as_slice()[column_start..column_start + (D - start)];
                for (row_offset, value) in column.iter().copied().enumerate() {
                    update[start + row_offset] = update[start + row_offset] + value * coefficient;
                }
            }
            for row in start..D {
                update[row] = beta * update[row];
            }
            let mut correction = T::zero();
            for row in start..D {
                correction = correction + reflector[row] * update[row];
            }
            correction = correction * beta / (T::one() + T::one());
            for row in start..D {
                update[row] = update[row] - correction * reflector[row];
            }
            let reflector_tail = &reflector.as_slice()[start..D];
            let update_tail = &update.as_slice()[start..D];
            for column_index in start..D {
                let column_start = column_index * D + start;
                T::rank_update_two_sub(
                    &mut work.as_mut_slice()[column_start..column_start + (D - start)],
                    reflector_tail,
                    update[column_index],
                    update_tail,
                    reflector[column_index],
                );
            }

            for row in start..D {
                work[(row, column)] = T::zero();
                work[(column, row)] = T::zero();
            }
            work[(start, column)] = subdiagonal;
            work[(column, start)] = subdiagonal;

            for row in 0..D {
                update[row] = T::zero();
            }
            for column_index in start..D {
                let coefficient = reflector[column_index];
                let column_start = column_index * D;
                let column = &eigenvectors.as_slice()[column_start..column_start + D];
                for (row, value) in column.iter().copied().enumerate() {
                    update[row] = update[row] + value * coefficient;
                }
            }
            for column_index in start..D {
                let column_start = column_index * D;
                T::rank_update_sub(
                    &mut eigenvectors.as_mut_slice()[column_start..column_start + D],
                    update.as_slice(),
                    beta * reflector[column_index],
                );
            }
        }

        for index in 0..D {
            diagonal[index] = work[(index, index)];
            if !diagonal[index].is_finite() {
                return Err(DecompositionError::NonFinite);
            }
            if index + 1 < D {
                off_diagonal[index] = work[(index + 1, index)];
                if !off_diagonal[index].is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
            }
        }

        let mut iterations = 0;
        if D > 1 {
            let mut end = D - 1;
            while end > 0 {
                for index in 0..end {
                    let scale = T::one()
                        .max(diagonal[index].abs())
                        .max(diagonal[index + 1].abs());
                    if off_diagonal[index].abs() <= tolerance * scale {
                        off_diagonal[index] = T::zero();
                    }
                }
                while end > 0 && off_diagonal[end - 1] == T::zero() {
                    end -= 1;
                }
                if end == 0 {
                    break;
                }
                let mut start = end - 1;
                while start > 0 && off_diagonal[start - 1] != T::zero() {
                    start -= 1;
                }
                iterations += 1;
                if iterations > 64 * D {
                    return Err(DecompositionError::NoConvergence);
                }

                let half = T::one() / (T::one() + T::one());
                let delta = (diagonal[end - 1] - diagonal[end]) * half;
                let subdiagonal = off_diagonal[end - 1];
                let mut shift = diagonal[end];
                if delta == T::zero() {
                    shift = shift - subdiagonal.abs();
                } else if subdiagonal != T::zero() {
                    let root = delta.hypot(subdiagonal);
                    let denominator = delta + if delta >= T::zero() { root } else { -root };
                    if denominator == T::zero() || !denominator.is_finite() {
                        return Err(DecompositionError::NonFinite);
                    }
                    shift = shift - (subdiagonal * subdiagonal) / denominator;
                }

                let mut x = diagonal[start] - shift;
                let mut z = off_diagonal[start];
                for index in start..end {
                    let radius = x.hypot(z);
                    let (cosine, sine) = if radius == T::zero() {
                        (T::one(), T::zero())
                    } else {
                        (x / radius, -z / radius)
                    };
                    let first = diagonal[index];
                    let second = off_diagonal[index];
                    let next = diagonal[index + 1];
                    let upper = sine * first + cosine * second;
                    let lower = sine * second + cosine * next;
                    diagonal[index] = cosine * (cosine * first - sine * second)
                        - sine * (cosine * second - sine * next);
                    diagonal[index + 1] = sine * upper + cosine * lower;
                    off_diagonal[index] = cosine * upper - sine * lower;
                    if index > start {
                        off_diagonal[index - 1] = cosine * off_diagonal[index - 1] - sine * z;
                    }
                    x = off_diagonal[index];
                    if index + 1 < end {
                        z = -sine * off_diagonal[index + 1];
                        off_diagonal[index + 1] = cosine * off_diagonal[index + 1];
                    }
                    for row in 0..D {
                        let old = eigenvectors[(row, index + 1)];
                        eigenvectors[(row, index + 1)] =
                            sine * eigenvectors[(row, index)] + cosine * old;
                        eigenvectors[(row, index)] =
                            cosine * eigenvectors[(row, index)] - sine * old;
                    }
                }
            }
        }

        let eigenvalues = &mut output.eigenvalues;
        for index in 0..D {
            eigenvalues[index] = diagonal[index] * matrix_scale;
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
    fn larger_tridiagonal_qr_reconstructs_and_orthogonalizes() {
        let input = Matrix::<6, 6, f64>::from_fn(|row, column| {
            if row == column {
                (6 + row) as f64
            } else {
                (row + column + 1) as f64 / 17.0
            }
        });
        let eigen = input
            .self_adjoint_eigen()
            .expect("symmetric input is supported");
        assert_relative_eq!(
            eigen.reconstruct(),
            input,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
        for index in 1..6 {
            assert!(eigen.eigenvalues()[index - 1] <= eigen.eigenvalues()[index]);
        }
        for left in 0..6 {
            for right in 0..6 {
                let mut dot = 0.0;
                for row in 0..6 {
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
    fn handles_large_finite_values_without_tridiagonal_overflow() {
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
