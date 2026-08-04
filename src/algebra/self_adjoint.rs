use crate::view::MatrixRead;
use crate::{DecompositionError, Matrix, Real};

/// Read-only self-adjoint view backed by one triangle of a square matrix.
///
/// The selected triangle is authoritative and is mirrored on reads. This
/// matches Eigen's `SelfAdjointView`: values in the opposite triangle are not
/// consulted by the view.
///
/// # Examples
///
/// ```
/// use stack_algebra::matrix;
///
/// let storage = matrix![1.0_f64, 99.0; 2.0, 3.0];
/// let symmetric = storage.self_adjoint_lower().to_matrix();
/// assert_eq!(symmetric, matrix![1.0_f64, 2.0; 2.0, 3.0]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SelfAdjointView<'a, const D: usize, T, const LOWER: bool> {
    matrix: &'a Matrix<D, D, T>,
}

/// Lower-triangle self-adjoint view.
pub type SelfAdjointLower<'a, const D: usize, T> = SelfAdjointView<'a, D, T, true>;

/// Upper-triangle self-adjoint view.
pub type SelfAdjointUpper<'a, const D: usize, T> = SelfAdjointView<'a, D, T, false>;

impl<'a, const D: usize, T, const LOWER: bool> SelfAdjointView<'a, D, T, LOWER> {
    #[inline]
    const fn source_indices(row: usize, column: usize) -> (usize, usize) {
        if LOWER == (row >= column) {
            (row, column)
        } else {
            (column, row)
        }
    }

    /// Returns the element at `(row, column)` using the selected triangle.
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Option<&'a T> {
        if row >= D || column >= D {
            return None;
        }
        let (source_row, source_column) = Self::source_indices(row, column);
        Some(&self.matrix[(source_row, source_column)])
    }

    /// Returns the original matrix backing this view.
    #[inline]
    pub const fn matrix(&self) -> &'a Matrix<D, D, T> {
        self.matrix
    }

    /// Copies the mirrored self-adjoint values into an owned matrix.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<D, D, T>
    where
        T: Copy,
    {
        Matrix::from_fn(|row, column| *self.get(row, column).expect("view index is in bounds"))
    }
}

impl<const D: usize, T, const LOWER: bool> MatrixRead<D, D, T>
    for SelfAdjointView<'_, D, T, LOWER>
{
    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<&T> {
        SelfAdjointView::get(self, row, column)
    }
}

impl<const D: usize, T> Matrix<D, D, T> {
    /// Creates a view that reads and mirrors the lower triangle.
    #[inline]
    pub fn self_adjoint_lower(&self) -> SelfAdjointLower<'_, D, T> {
        SelfAdjointView { matrix: self }
    }

    /// Creates a view that reads and mirrors the upper triangle.
    #[inline]
    pub fn self_adjoint_upper(&self) -> SelfAdjointUpper<'_, D, T> {
        SelfAdjointView { matrix: self }
    }
}

impl<const D: usize, T: Real> Matrix<D, D, T> {
    /// Validates finite symmetry using a scaled absolute tolerance.
    #[inline]
    pub fn validate_symmetric(&self, tolerance: T) -> Result<(), DecompositionError> {
        if !tolerance.is_finite() || tolerance < T::zero() {
            return Err(DecompositionError::NotSymmetric);
        }
        for row in 0..D {
            if !self[(row, row)].is_finite() {
                return Err(DecompositionError::NonFinite);
            }
            for column in 0..row {
                let left = self[(row, column)];
                let right = self[(column, row)];
                if !left.is_finite() || !right.is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
                let scale = T::one().max(left.abs()).max(right.abs());
                if (left - right).abs() > tolerance * scale {
                    return Err(DecompositionError::NotSymmetric);
                }
            }
        }
        Ok(())
    }

    /// Returns whether the matrix is finite and symmetric within tolerance.
    #[inline]
    pub fn is_symmetric(&self, tolerance: T) -> bool {
        self.validate_symmetric(tolerance).is_ok()
    }

    /// Creates a checked lower-triangle self-adjoint view.
    #[inline]
    pub fn try_self_adjoint_lower(
        &self,
        tolerance: T,
    ) -> Result<SelfAdjointLower<'_, D, T>, DecompositionError> {
        self.validate_symmetric(tolerance)?;
        Ok(self.self_adjoint_lower())
    }

    /// Creates a checked upper-triangle self-adjoint view.
    #[inline]
    pub fn try_self_adjoint_upper(
        &self,
        tolerance: T,
    ) -> Result<SelfAdjointUpper<'_, D, T>, DecompositionError> {
        self.validate_symmetric(tolerance)?;
        Ok(self.self_adjoint_upper())
    }
}

#[cfg(test)]
mod tests {
    use crate::{matrix, matvec_view, DecompositionError, Matrix, MatrixRead};

    #[test]
    fn lower_view_mirrors_only_lower_triangle() {
        let matrix = matrix![1_i32, 20, 3; 2, 4, 30; 5, 6, 7];
        let view = matrix.self_adjoint_lower();
        assert_eq!(
            view.to_matrix(),
            Matrix::from_rows([[1, 2, 5], [2, 4, 6], [5, 6, 7]])
        );
        assert_eq!(MatrixRead::get(&view, 0, 2), Some(&5));
    }

    #[test]
    fn upper_view_mirrors_only_upper_triangle() {
        let matrix = matrix![1_i32, 20, 3; 2, 4, 30; 5, 6, 7];
        let view = matrix.self_adjoint_upper();
        assert_eq!(
            view.to_matrix(),
            Matrix::from_rows([[1, 20, 3], [20, 4, 30], [3, 30, 7]])
        );
        let vector = matrix![1_i32; 2; 3];
        assert_eq!(matvec_view(&view, &vector), Some(matrix![50_i32; 118; 84]));
    }

    #[test]
    fn checked_views_validate_scaled_symmetry() {
        let matrix = matrix![1.0_f64, 2.0; 2.0 + 1e-13, 3.0];
        assert!(matrix.is_symmetric(1e-12));
        assert!(matrix.try_self_adjoint_lower(1e-12).is_ok());
        assert!(matrix.try_self_adjoint_upper(1e-15).is_err());
    }

    #[test]
    fn symmetry_validation_reports_asymmetry_and_non_finite_values() {
        let asymmetric = matrix![1.0_f64, 2.0; 0.0, 1.0];
        assert_eq!(
            asymmetric.validate_symmetric(1e-12),
            Err(DecompositionError::NotSymmetric)
        );
        let non_finite = matrix![1.0_f64, f64::NAN; f64::NAN, 1.0];
        assert_eq!(
            non_finite.validate_symmetric(1e-12),
            Err(DecompositionError::NonFinite)
        );
    }
}
