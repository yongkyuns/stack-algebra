use crate::view::MatrixRead;
use crate::{DecompositionError, Matrix, MatrixScalar, Real, Vector};

#[inline]
fn column_norm<const M: usize, const N: usize, T: Real + MatrixScalar>(
    matrix: &Matrix<M, N, T>,
    column: usize,
) -> T {
    let values_start = column * M;
    let values_end = values_start + M;
    let values = &matrix.as_slice()[values_start..values_end];
    let sum = T::dot_accumulate(values, values, T::zero());
    if sum.is_finite() && sum != T::zero() {
        return sum.sqrt();
    }

    let mut max_abs = T::zero();
    for &value in values {
        max_abs = max_abs.max(value.abs());
    }
    if sum.is_finite() && max_abs == T::zero() {
        return sum.sqrt();
    }
    if max_abs.is_infinite() {
        return max_abs;
    }
    if !max_abs.is_finite() || max_abs == T::zero() {
        return sum.sqrt();
    }
    let mut scaled_sum = T::zero();
    for &value in values {
        let ratio = value.abs() / max_abs;
        scaled_sum = scaled_sum + ratio * ratio;
    }
    max_abs * scaled_sum.sqrt()
}

/// Thin singular value decomposition of a fixed-size matrix.
///
/// For an `M x N` matrix with `M >= N`, the decomposition satisfies
/// `A = U * diag(S) * Vᵀ`, where `U` is `M x N`, `S` has `N` entries, and `V`
/// is `N x N`. The implementation uses one-sided Jacobi rotations and does
/// not allocate from the heap.
///
/// Singular values are returned in descending order. A relative threshold
/// controls rank decisions; use [`Self::with_threshold`] when the default
/// machine-precision cutoff is not appropriate for the problem scale.
///
/// # Examples
///
/// ```
/// use stack_algebra::matrix;
///
/// let a = matrix![1.0_f64, 0.0; 0.0, 1.0; 1.0, 1.0];
/// let rhs = matrix![2.0_f64; 3.0; 5.0];
/// let svd = a.svd().expect("finite matrix");
/// let x = svd.solve(&rhs);
/// assert!((a * x - rhs).norm() < 1.0e-12);
/// assert_eq!(svd.rank(), 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Svd<const M: usize, const N: usize, T> {
    u: Matrix<M, N, T>,
    singular_values: Vector<N, T>,
    v: Matrix<N, N, T>,
    threshold: T,
}

impl<const M: usize, const N: usize, T: Real + MatrixScalar> Svd<M, N, T> {
    /// Recomputes this SVD in place with typed failure reporting.
    #[inline]
    pub fn try_compute(&mut self, matrix: &Matrix<M, N, T>) -> Result<(), DecompositionError> {
        Self::try_factorize(matrix, self)
    }

    /// Recomputes this SVD directly from a fixed-size matrix view.
    #[inline]
    pub fn try_compute_view<V>(&mut self, matrix: &V) -> Result<(), DecompositionError>
    where
        V: MatrixRead<M, N, T>,
    {
        *self = Self::try_decompose_view(matrix)?;
        Ok(())
    }

    /// Computes a fixed-size SVD for any matrix shape.
    ///
    /// For wide matrices (`M < N`), the final `N - M` singular values and
    /// corresponding columns of `U` are zero-padded.
    #[inline]
    pub fn decompose(matrix: &Matrix<M, N, T>) -> Option<Self> {
        Self::try_decompose(matrix).ok()
    }

    /// Computes a fixed-size SVD and reports numerical failures explicitly.
    #[inline]
    pub fn try_decompose(matrix: &Matrix<M, N, T>) -> Result<Self, DecompositionError> {
        let mut output = Self {
            u: *matrix,
            singular_values: Vector::<N, T>::zeros(),
            v: Matrix::<N, N, T>::eye(),
            threshold: T::zero(),
        };
        Self::try_factorize(matrix, &mut output)?;
        Ok(output)
    }

    /// Computes an SVD directly from a fixed-size matrix view without
    /// materializing a separate owning input matrix.
    #[inline]
    pub fn try_decompose_view<V>(matrix: &V) -> Result<Self, DecompositionError>
    where
        V: MatrixRead<M, N, T>,
    {
        let mut output = Self {
            u: Matrix::zeros(),
            singular_values: Vector::<N, T>::zeros(),
            v: Matrix::<N, N, T>::eye(),
            threshold: T::zero(),
        };
        for column in 0..N {
            for row in 0..M {
                output.u[(row, column)] = *matrix
                    .get(row, column)
                    .ok_or(DecompositionError::InvalidView)?;
            }
        }
        Self::try_factorize_storage(&mut output)?;
        Ok(output)
    }

    #[inline]
    fn try_factorize(
        matrix: &Matrix<M, N, T>,
        output: &mut Self,
    ) -> Result<(), DecompositionError> {
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(DecompositionError::NonFinite);
        }
        output.u = *matrix;
        Self::try_factorize_storage(output)
    }

    #[inline]
    fn try_factorize_storage(output: &mut Self) -> Result<(), DecompositionError> {
        output.v = Matrix::<N, N, T>::eye();
        output.singular_values = Vector::<N, T>::zeros();
        let u = &mut output.u;
        let v = &mut output.v;
        let singular_values = &mut output.singular_values;
        let dimension = T::from(core::cmp::max(M, N)).unwrap_or(T::one());
        let tolerance = T::epsilon() * dimension;
        const MAX_SWEEPS: usize = 32;
        let mut converged = false;

        for _ in 0..MAX_SWEEPS {
            let mut changed = false;
            for p in 0..N {
                for q in (p + 1)..N {
                    let mut scale = T::zero();
                    for row in 0..M {
                        scale = scale.max(u[(row, p)].abs()).max(u[(row, q)].abs());
                    }
                    if !scale.is_finite() {
                        return Err(DecompositionError::NonFinite);
                    }
                    if scale == T::zero() {
                        continue;
                    }
                    let left = &u.as_slice()[p * M..(p + 1) * M];
                    let right = &u.as_slice()[q * M..(q + 1) * M];
                    let scale_squared = scale * scale;
                    let (app, aqq, apq) = if scale_squared.is_finite() && scale_squared != T::zero()
                    {
                        let (raw_app, raw_aqq, raw_apq) = T::symmetric_dot(left, right);
                        if raw_app.is_finite() && raw_aqq.is_finite() && raw_apq.is_finite() {
                            (
                                raw_app / scale_squared,
                                raw_aqq / scale_squared,
                                raw_apq / scale_squared,
                            )
                        } else {
                            let mut app = T::zero();
                            let mut aqq = T::zero();
                            let mut apq = T::zero();
                            for (&left, &right) in left.iter().zip(right.iter()) {
                                let left = left / scale;
                                let right = right / scale;
                                app = app + left * left;
                                aqq = aqq + right * right;
                                apq = apq + left * right;
                            }
                            (app, aqq, apq)
                        }
                    } else {
                        let mut app = T::zero();
                        let mut aqq = T::zero();
                        let mut apq = T::zero();
                        for (&left, &right) in left.iter().zip(right.iter()) {
                            let left = left / scale;
                            let right = right / scale;
                            app = app + left * left;
                            aqq = aqq + right * right;
                            apq = apq + left * right;
                        }
                        (app, aqq, apq)
                    };
                    if !app.is_finite() || !aqq.is_finite() || !apq.is_finite() {
                        return Err(DecompositionError::NonFinite);
                    }
                    let gram_scale = app.sqrt() * aqq.sqrt();
                    if !gram_scale.is_finite() {
                        return Err(DecompositionError::NonFinite);
                    }
                    if gram_scale == T::zero() || apq.abs() <= tolerance * gram_scale {
                        continue;
                    }

                    let tau = (aqq - app) / (apq + apq);
                    let root = (T::one() + tau * tau).sqrt();
                    let denominator = tau.abs() + root;
                    if denominator == T::zero() || !denominator.is_finite() {
                        continue;
                    }
                    let tangent = if tau >= T::zero() {
                        T::one() / denominator
                    } else {
                        -T::one() / denominator
                    };
                    let cosine = T::one() / (T::one() + tangent * tangent).sqrt();
                    let sine = tangent * cosine;

                    let (u_prefix, u_suffix) = u.as_mut_slice().split_at_mut(q * M);
                    let u_first = &mut u_prefix[p * M..(p + 1) * M];
                    let u_second = &mut u_suffix[..M];
                    T::rotate_columns(u_first, u_second, cosine, sine);
                    let (v_prefix, v_suffix) = v.as_mut_slice().split_at_mut(q * N);
                    let v_first = &mut v_prefix[p * N..(p + 1) * N];
                    let v_second = &mut v_suffix[..N];
                    T::rotate_columns(v_first, v_second, cosine, sine);
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

        for column in 0..N {
            singular_values[column] = column_norm(u, column);
            if !singular_values[column].is_finite() {
                return Err(DecompositionError::NonFinite);
            }
        }

        for index in 0..N {
            let mut largest = index;
            for candidate in (index + 1)..N {
                if singular_values[candidate] > singular_values[largest] {
                    largest = candidate;
                }
            }
            if largest != index {
                singular_values.swap_rows(index, largest);
                u.swap_columns(index, largest);
                v.swap_columns(index, largest);
            }
        }

        let max_singular_value = singular_values.iter().copied().fold(T::zero(), T::max);
        if !max_singular_value.is_finite() {
            return Err(DecompositionError::NonFinite);
        }
        let cutoff = tolerance * max_singular_value;
        for column in 0..N {
            let singular_value = singular_values[column];
            if singular_value > cutoff && singular_value.is_finite() {
                for row in 0..M {
                    u[(row, column)] = u[(row, column)] / singular_value;
                }
            } else {
                singular_values[column] = T::zero();
                for row in 0..M {
                    u[(row, column)] = T::zero();
                }
            }
        }

        output.threshold = tolerance;
        Ok(())
    }

    /// Returns the thin left singular vectors.
    #[inline]
    pub fn u(&self) -> &Matrix<M, N, T> {
        &self.u
    }

    /// Returns the singular values in descending order.
    #[inline]
    pub fn singular_values(&self) -> &Vector<N, T> {
        &self.singular_values
    }

    /// Returns the right singular vectors.
    #[inline]
    pub fn v(&self) -> &Matrix<N, N, T> {
        &self.v
    }

    /// Returns the relative threshold used for rank decisions.
    #[inline]
    pub fn threshold(&self) -> T {
        self.threshold
    }

    /// Sets the relative rank threshold.
    #[inline]
    pub fn set_threshold(&mut self, threshold: T) {
        self.threshold = threshold;
    }

    /// Returns a copy using a custom relative rank threshold.
    #[inline]
    pub fn with_threshold(mut self, threshold: T) -> Self {
        self.set_threshold(threshold);
        self
    }

    /// Returns the numerical rank using the configured threshold.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank_with_threshold(self.threshold)
    }

    /// Returns the numerical rank using a custom relative threshold.
    #[inline]
    pub fn rank_with_threshold(&self, threshold: T) -> usize {
        let maximum = self.singular_values.iter().copied().fold(T::zero(), T::max);
        let cutoff = threshold * maximum;
        self.singular_values
            .iter()
            .filter(|&&value| value.is_finite() && value > cutoff)
            .count()
    }

    /// Computes the Moore-Penrose pseudoinverse using the configured threshold.
    #[inline]
    pub fn pseudo_inverse(&self) -> Matrix<N, M, T> {
        self.pseudo_inverse_with_threshold(self.threshold)
    }

    /// Computes the Moore-Penrose pseudoinverse using a custom threshold.
    #[inline]
    pub fn pseudo_inverse_with_threshold(&self, threshold: T) -> Matrix<N, M, T> {
        let maximum = self.singular_values.iter().copied().fold(T::zero(), T::max);
        let cutoff = threshold * maximum;
        let mut inverse = Matrix::<N, M, T>::zeros();
        for singular in 0..N {
            let value = self.singular_values[singular];
            if !value.is_finite() || value <= cutoff {
                continue;
            }
            let scale = T::one() / value;
            for row in 0..N {
                for column in 0..M {
                    inverse[(row, column)] = inverse[(row, column)]
                        + self.v[(row, singular)] * self.u[(column, singular)] * scale;
                }
            }
        }
        inverse
    }

    /// Solves a least-squares problem using the configured rank threshold.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<M, P, T>) -> Matrix<N, P, T> {
        let mut solution = Matrix::<N, P, T>::zeros();
        self.solve_with_threshold_into(rhs, self.threshold, &mut solution);
        solution
    }

    /// Solves a least-squares problem using a custom rank threshold.
    #[inline]
    pub fn solve_with_threshold<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
        threshold: T,
    ) -> Matrix<N, P, T> {
        let mut solution = Matrix::<N, P, T>::zeros();
        self.solve_with_threshold_into(rhs, threshold, &mut solution);
        solution
    }

    /// Solves a least-squares problem into a caller-provided output matrix.
    #[inline]
    pub fn solve_into<const P: usize>(&self, rhs: &Matrix<M, P, T>, output: &mut Matrix<N, P, T>) {
        self.solve_with_threshold_into(rhs, self.threshold, output);
    }

    /// Solves a least-squares problem with a custom threshold into an output.
    #[inline]
    pub fn solve_with_threshold_into<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
        threshold: T,
        output: &mut Matrix<N, P, T>,
    ) {
        let maximum = self.singular_values.iter().copied().fold(T::zero(), T::max);
        let cutoff = threshold * maximum;
        let mut coefficients = Matrix::<N, P, T>::zeros();
        for singular in 0..N {
            let value = self.singular_values[singular];
            if !value.is_finite() || value <= cutoff {
                continue;
            }
            let scale = T::one() / value;
            for rhs_column in 0..P {
                let mut projection = T::zero();
                for row in 0..M {
                    projection = projection + self.u[(row, singular)] * rhs[(row, rhs_column)];
                }
                coefficients[(singular, rhs_column)] = projection * scale;
            }
        }

        *output = Matrix::zeros();
        for row in 0..N {
            for rhs_column in 0..P {
                let mut value = T::zero();
                for singular in 0..N {
                    value = value + self.v[(row, singular)] * coefficients[(singular, rhs_column)];
                }
                output[(row, rhs_column)] = value;
            }
        }
    }
}

impl<const M: usize, const N: usize, T: Real + MatrixScalar> Matrix<M, N, T> {
    /// Computes a fixed-size SVD for any matrix shape.
    #[inline]
    pub fn svd(&self) -> Option<Svd<M, N, T>> {
        Svd::decompose(self)
    }

    /// Computes a fixed-size SVD with a typed failure result.
    #[inline]
    pub fn try_svd(&self) -> Result<Svd<M, N, T>, DecompositionError> {
        Svd::try_decompose(self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{matrix, DecompositionError, Map, Matrix, Svd};

    #[test]
    fn reconstructs_square_matrix() {
        let input = matrix![
            3.0_f64, 1.0, 0.0;
            0.0, 2.0, 1.0;
            1.0, 0.0, 2.0;
        ];
        let svd = input.svd().expect("square matrix is supported");
        let mut reconstructed = Matrix::<3, 3, f64>::zeros();
        for row in 0..3 {
            for column in 0..3 {
                for singular in 0..3 {
                    reconstructed[(row, column)] += svd.u()[(row, singular)]
                        * svd.singular_values()[singular]
                        * svd.v()[(column, singular)];
                }
            }
        }
        assert_relative_eq!(reconstructed, input, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn handles_large_finite_values_without_gram_overflow() {
        let input = matrix![
            1.0e200_f64, 2.0e200;
            3.0e200, 4.0e200;
        ];
        let svd = input.svd().expect("finite matrix is supported");
        let mut reconstructed = Matrix::<2, 2, f64>::zeros();
        for row in 0..2 {
            for column in 0..2 {
                for singular in 0..2 {
                    reconstructed[(row, column)] += svd.u()[(row, singular)]
                        * svd.singular_values()[singular]
                        * svd.v()[(column, singular)];
                }
            }
        }
        assert_relative_eq!(reconstructed, input, max_relative = 1e-10);
    }

    #[test]
    fn reuses_caller_provided_factor_storage() {
        let first = matrix![1.0_f64, 2.0; 3.0, 4.0; 5.0, 6.0];
        let second = matrix![2.0_f64, -1.0; 0.0, 3.0; 4.0, 5.0];
        let mut factor = first.svd().expect("first SVD converges");
        factor.try_compute(&second).expect("second SVD converges");
        let diagonal = Matrix::from_fn(|row, column| {
            if row == column {
                factor.singular_values()[row]
            } else {
                0.0
            }
        });
        assert_relative_eq!(
            *factor.u() * diagonal * factor.v().transpose(),
            second,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    fn decomposes_map_and_block_views() {
        let matrix = matrix![
            1.0_f64, 2.0;
            3.0, 4.0;
            5.0, 6.0;
        ];
        let mapped = Map::<3, 2, f64>::from_slice(matrix.as_slice()).unwrap();
        let mapped_factor = Svd::try_decompose_view(&mapped).unwrap();
        let diagonal = Matrix::from_fn(|row, column| {
            if row == column {
                mapped_factor.singular_values()[row]
            } else {
                0.0
            }
        });
        assert_relative_eq!(
            *mapped_factor.u() * diagonal * mapped_factor.v().transpose(),
            matrix,
            epsilon = 1e-10,
            max_relative = 1e-10
        );

        let mut storage = Matrix::<4, 4, f64>::zeros();
        for row in 0..3 {
            for column in 0..2 {
                storage[(row + 1, column + 1)] = matrix[(row, column)];
            }
        }
        let block = storage.block::<3, 2>(1, 1).unwrap();
        let mut reused = mapped_factor;
        reused.try_compute_view(&block).unwrap();
        let diagonal = Matrix::from_fn(|row, column| {
            if row == column {
                reused.singular_values()[row]
            } else {
                0.0
            }
        });
        assert_relative_eq!(
            *reused.u() * diagonal * reused.v().transpose(),
            matrix,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    fn solves_rank_deficient_system() {
        let input = matrix![
            1.0_f64, 2.0;
            2.0, 4.0;
            3.0, 6.0;
        ];
        let rhs = matrix![1.0_f64; 2.0; 3.0];
        let svd = input.svd().expect("overdetermined matrix is supported");
        assert_eq!(svd.rank(), 1);
        let solution = svd.solve(&rhs);
        assert_relative_eq!(input * solution, rhs, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn reconstructs_wide_matrix_with_zero_padding() {
        let input = matrix![
            1.0_f64, 2.0, 3.0;
            4.0, 5.0, 6.0;
        ];
        let svd = input.svd().expect("wide matrix is supported");
        let mut reconstructed = Matrix::<2, 3, f64>::zeros();
        for row in 0..2 {
            for column in 0..3 {
                for singular in 0..3 {
                    reconstructed[(row, column)] += svd.u()[(row, singular)]
                        * svd.singular_values()[singular]
                        * svd.v()[(column, singular)];
                }
            }
        }
        assert_relative_eq!(reconstructed, input, epsilon = 1e-10, max_relative = 1e-10);
        assert_eq!(svd.rank(), 2);
        assert_eq!(svd.singular_values()[2], 0.0);
    }

    #[test]
    fn threshold_controls_ill_conditioned_rank() {
        let input = Matrix::<3, 2, f64>::from_rows([[1.0, 0.0], [0.0, 1.0e-10], [0.0, 0.0]]);
        let svd = input.svd().expect("tall matrix is supported");
        assert_eq!(svd.rank(), 2);
        assert_eq!(svd.with_threshold(1.0e-8).rank(), 1);
    }

    #[test]
    fn typed_svd_reports_non_finite_input() {
        let input = matrix![1.0_f64, f64::NAN; 2.0, 3.0];
        assert_eq!(input.try_svd(), Err(DecompositionError::NonFinite));
        assert!(input.svd().is_none());
    }
}
