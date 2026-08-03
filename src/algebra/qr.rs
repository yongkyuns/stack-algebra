use crate::{MatmulBackend, Matrix, MatrixScalar, Real, Vector};

#[inline]
fn column_norm<const M: usize, const N: usize, T: Real>(
    matrix: &Matrix<M, N, T>,
    column: usize,
    start_row: usize,
) -> T {
    let mut sum = T::zero();
    let mut max_abs = T::zero();
    for row in start_row..M {
        let value = matrix[(row, column)];
        let absolute = value.abs();
        max_abs = max_abs.max(absolute);
        sum = sum + value * value;
    }

    if sum.is_finite() && (sum != T::zero() || max_abs == T::zero()) {
        return sum.sqrt();
    }
    if max_abs.is_infinite() {
        return max_abs;
    }
    if !max_abs.is_finite() || max_abs == T::zero() {
        return sum.sqrt();
    }

    let mut scaled_sum = T::zero();
    for row in start_row..M {
        let ratio = matrix[(row, column)].abs() / max_abs;
        scaled_sum = scaled_sum + ratio * ratio;
    }
    max_abs * scaled_sum.sqrt()
}

#[inline]
fn apply_q_transpose_in_place<const M: usize, const N: usize, const P: usize, T: Real>(
    factors: &Matrix<M, N, T>,
    coefficients: &Vector<N, T>,
    transformed: &mut Matrix<M, P, T>,
) {
    let limit = core::cmp::min(M, N);
    for column in 0..limit {
        let coefficient = coefficients[column];
        if coefficient == T::zero() {
            continue;
        }
        for rhs_column in 0..P {
            let mut dot = transformed[(column, rhs_column)];
            for row in (column + 1)..M {
                dot = dot + factors[(row, column)] * transformed[(row, rhs_column)];
            }
            let scale = coefficient * dot;
            transformed[(column, rhs_column)] = transformed[(column, rhs_column)] - scale;
            for row in (column + 1)..M {
                transformed[(row, rhs_column)] =
                    transformed[(row, rhs_column)] - scale * factors[(row, column)];
            }
        }
    }
}

#[inline]
fn apply_q_in_place<const M: usize, const N: usize, const P: usize, T: Real>(
    factors: &Matrix<M, N, T>,
    coefficients: &Vector<N, T>,
    transformed: &mut Matrix<M, P, T>,
) {
    let limit = core::cmp::min(M, N);
    for column in (0..limit).rev() {
        let coefficient = coefficients[column];
        if coefficient == T::zero() {
            continue;
        }
        for rhs_column in 0..P {
            let mut dot = transformed[(column, rhs_column)];
            for row in (column + 1)..M {
                dot = dot + factors[(row, column)] * transformed[(row, rhs_column)];
            }
            let scale = coefficient * dot;
            transformed[(column, rhs_column)] = transformed[(column, rhs_column)] - scale;
            for row in (column + 1)..M {
                transformed[(row, rhs_column)] =
                    transformed[(row, rhs_column)] - scale * factors[(row, column)];
            }
        }
    }
}

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
            let norm = column_norm(&factors, column, column);
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
            let source_start = column * M + column + 1;
            let source_end = (column + 1) * M;
            T::Matmul::scale_divide(&mut factors.as_mut_slice()[source_start..source_end], first);

            let coefficient = (beta - diagonal) / beta;
            coefficients[column] = coefficient;
            for trailing_column in (column + 1)..N {
                let initial = factors[(column, trailing_column)];
                let dot = if M <= 8 {
                    let mut dot = initial;
                    for row in (column + 1)..M {
                        dot = dot + factors[(row, column)] * factors[(row, trailing_column)];
                    }
                    dot
                } else {
                    let target_start = trailing_column * M + column + 1;
                    let source = &factors.as_slice()[source_start..source_end];
                    let target = &factors.as_slice()[target_start..target_start + source.len()];
                    T::Matmul::dot(source, target, initial)
                };
                let scale = coefficient * dot;
                factors[(column, trailing_column)] = factors[(column, trailing_column)] - scale;
                let target_start = trailing_column * M + column + 1;
                let (prefix, suffix) = factors.as_mut_slice().split_at_mut(source_end);
                let source = &prefix[source_start..source_end];
                let target_offset = target_start - source_end;
                T::Matmul::rank_update_sub(
                    &mut suffix[target_offset..target_offset + source.len()],
                    source,
                    scale,
                );
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
        self.apply_q_transpose_in_place(&mut transformed);
        transformed
    }

    /// Applies `Qᵀ` to a matrix in place without materializing `Q`.
    #[inline]
    pub fn apply_q_transpose_in_place<const P: usize>(&self, rhs: &mut Matrix<M, P, T>) {
        apply_q_transpose_in_place(&self.factors, &self.coefficients, rhs);
    }

    /// Applies `Q` to a matrix without materializing `Q`.
    #[inline]
    pub fn apply_q<const P: usize>(&self, rhs: &Matrix<M, P, T>) -> Matrix<M, P, T> {
        let mut transformed = *rhs;
        self.apply_q_in_place(&mut transformed);
        transformed
    }

    /// Applies `Q` to a matrix in place without materializing `Q`.
    #[inline]
    pub fn apply_q_in_place<const P: usize>(&self, rhs: &mut Matrix<M, P, T>) {
        apply_q_in_place(&self.factors, &self.coefficients, rhs);
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

    /// Computes a column-pivoted Householder QR factorization of this matrix.
    #[inline]
    pub fn col_piv_householder_qr(&self) -> ColPivHouseholderQr<M, N, T> {
        ColPivHouseholderQr::decompose(self)
    }
}

/// Column-pivoted Householder QR factorization of a fixed-size matrix.
///
/// The factorization satisfies `A * P = Q * R`, where `P` is represented by
/// [`Self::permutation`]. Column pivoting improves rank detection and the
/// numerical behavior of least-squares solves for ill-conditioned inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColPivHouseholderQr<const M: usize, const N: usize, T> {
    factors: Matrix<M, N, T>,
    coefficients: Vector<N, T>,
    permutation: [usize; N],
    max_pivot: T,
    threshold: T,
}

impl<const M: usize, const N: usize, T: Real + MatrixScalar> ColPivHouseholderQr<M, N, T> {
    /// Computes a column-pivoted Householder QR factorization.
    #[inline]
    pub fn decompose(matrix: &Matrix<M, N, T>) -> Self {
        let mut factors = *matrix;
        let mut coefficients = Vector::<N, T>::zeros();
        let mut permutation = core::array::from_fn(|index| index);
        let mut direct_norms: [T; N] =
            core::array::from_fn(|column| column_norm(&factors, column, 0));
        let mut updated_norms = direct_norms;
        let limit = core::cmp::min(M, N);

        for column in 0..limit {
            let mut pivot_column = column;
            let mut pivot_norm = updated_norms[column];
            for (candidate, &norm) in updated_norms.iter().enumerate().skip(column) {
                if norm > pivot_norm {
                    pivot_norm = norm;
                    pivot_column = candidate;
                }
            }
            if pivot_column != column {
                factors.swap_columns(column, pivot_column);
                permutation.swap(column, pivot_column);
                direct_norms.swap(column, pivot_column);
                updated_norms.swap(column, pivot_column);
            }

            if !pivot_norm.is_finite() || pivot_norm == T::zero() {
                continue;
            }

            let diagonal = factors[(column, column)];
            let beta = if diagonal >= T::zero() {
                -pivot_norm
            } else {
                pivot_norm
            };
            let first = diagonal - beta;
            if first == T::zero() || !first.is_finite() {
                continue;
            }

            factors[(column, column)] = beta;
            let source_start = column * M + column + 1;
            let source_end = (column + 1) * M;
            T::Matmul::scale_divide(&mut factors.as_mut_slice()[source_start..source_end], first);

            let coefficient = (beta - diagonal) / beta;
            coefficients[column] = coefficient;
            for trailing_column in (column + 1)..N {
                let initial = factors[(column, trailing_column)];
                let dot = if M <= 8 {
                    let mut dot = initial;
                    for row in (column + 1)..M {
                        dot = dot + factors[(row, column)] * factors[(row, trailing_column)];
                    }
                    dot
                } else {
                    let target_start = trailing_column * M + column + 1;
                    let source = &factors.as_slice()[source_start..source_end];
                    let target = &factors.as_slice()[target_start..target_start + source.len()];
                    T::Matmul::dot(source, target, initial)
                };
                let scale = coefficient * dot;
                factors[(column, trailing_column)] = factors[(column, trailing_column)] - scale;
                let target_start = trailing_column * M + column + 1;
                let (prefix, suffix) = factors.as_mut_slice().split_at_mut(source_end);
                let source = &prefix[source_start..source_end];
                let target_offset = target_start - source_end;
                T::Matmul::rank_update_sub(
                    &mut suffix[target_offset..target_offset + source.len()],
                    source,
                    scale,
                );

                let old_updated_norm = updated_norms[trailing_column];
                if old_updated_norm != T::zero() {
                    let ratio = factors[(column, trailing_column)].abs() / old_updated_norm;
                    let one = T::one();
                    let product = (one + ratio) * (one - ratio);
                    let product = product.max(T::zero());
                    let update_ratio = old_updated_norm / direct_norms[trailing_column];
                    let accuracy = product * update_ratio * update_ratio;
                    if accuracy <= T::epsilon().sqrt() {
                        let direct_norm = column_norm(&factors, trailing_column, column + 1);
                        direct_norms[trailing_column] = direct_norm;
                        updated_norms[trailing_column] = direct_norm;
                    } else {
                        updated_norms[trailing_column] = old_updated_norm * product.sqrt();
                    }
                }
            }
        }

        let mut max_pivot = T::zero();
        for index in 0..limit {
            max_pivot = max_pivot.max(factors[(index, index)].abs());
        }
        let dimension = T::from(core::cmp::max(M, N)).unwrap_or(T::one());

        Self {
            factors,
            coefficients,
            permutation,
            max_pivot,
            threshold: T::epsilon() * dimension,
        }
    }

    /// Returns the numerical rank detected from the triangular factor.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank_with_threshold(self.threshold)
    }

    /// Returns the maximum absolute diagonal pivot.
    #[inline]
    pub fn max_pivot(&self) -> T {
        self.max_pivot
    }

    /// Returns the relative threshold used for rank decisions.
    #[inline]
    pub fn threshold(&self) -> T {
        self.threshold
    }

    /// Sets the relative rank threshold and returns the updated factorization.
    ///
    /// A diagonal pivot is considered nonzero when its absolute value is
    /// greater than `threshold * max_pivot()`.
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

    /// Returns the numerical rank using a custom relative threshold.
    #[inline]
    pub fn rank_with_threshold(&self, threshold: T) -> usize {
        let cutoff = threshold * self.max_pivot;
        let limit = core::cmp::min(M, N);
        let mut rank = 0;
        for index in 0..limit {
            let diagonal = self.factors[(index, index)];
            if diagonal.is_finite() && diagonal.abs() > cutoff {
                rank += 1;
            }
        }
        rank
    }

    /// Returns the column permutation such that `A * P = Q * R`.
    ///
    /// Entry `permutation[k]` is the original column now stored at factor
    /// column `k`.
    #[inline]
    pub fn permutation(&self) -> &[usize; N] {
        &self.permutation
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
        self.apply_q_transpose_in_place(&mut transformed);
        transformed
    }

    /// Applies `Qᵀ` to a matrix in place without materializing `Q`.
    #[inline]
    pub fn apply_q_transpose_in_place<const P: usize>(&self, rhs: &mut Matrix<M, P, T>) {
        apply_q_transpose_in_place(&self.factors, &self.coefficients, rhs);
    }

    /// Applies `Q` to a matrix without materializing `Q`.
    #[inline]
    pub fn apply_q<const P: usize>(&self, rhs: &Matrix<M, P, T>) -> Matrix<M, P, T> {
        let mut transformed = *rhs;
        self.apply_q_in_place(&mut transformed);
        transformed
    }

    /// Applies `Q` to a matrix in place without materializing `Q`.
    #[inline]
    pub fn apply_q_in_place<const P: usize>(&self, rhs: &mut Matrix<M, P, T>) {
        apply_q_in_place(&self.factors, &self.coefficients, rhs);
    }

    /// Solves the full-rank least-squares problem `min ||A X - B||₂`.
    ///
    /// This method supports square and overdetermined matrices (`M >= N`). It
    /// returns `None` when the detected rank is less than `N`.
    #[inline]
    pub fn solve_least_squares<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
    ) -> Option<Matrix<N, P, T>> {
        if M < N || self.rank() < N {
            return None;
        }

        self.solve_least_squares_basic(rhs)
    }

    /// Solves a least-squares problem using the detected independent pivots.
    ///
    /// Dependent variables are set to zero in the permuted coordinate system,
    /// matching Eigen's basic rank-deficient `ColPivHouseholderQR` solve.
    /// Returns `None` for underdetermined matrices (`M < N`).
    #[inline]
    pub fn solve_least_squares_basic<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
    ) -> Option<Matrix<N, P, T>> {
        if M < N {
            return None;
        }

        let transformed = self.apply_q_transpose(rhs);
        let mut permuted_solution = Matrix::<N, P, T>::zeros();
        let rank = self.rank();
        for rhs_column in 0..P {
            for row in (0..rank).rev() {
                let diagonal = self.factors[(row, row)];
                let mut value = transformed[(row, rhs_column)];
                for next in (row + 1)..rank {
                    value =
                        value - self.factors[(row, next)] * permuted_solution[(next, rhs_column)];
                }
                permuted_solution[(row, rhs_column)] = value / diagonal;
            }
        }

        let mut solution = Matrix::<N, P, T>::zeros();
        for column in 0..N {
            for rhs_column in 0..P {
                solution[(self.permutation[column], rhs_column)] =
                    permuted_solution[(column, rhs_column)];
            }
        }
        Some(solution)
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
    fn applies_q_and_q_transpose_in_place() {
        let input = matrix![
            12.0_f64, -51.0, 4.0;
            6.0, 167.0, -68.0;
            -4.0, 24.0, -41.0;
        ];
        let qr = input.householder_qr();
        let r = qr.r();
        assert_relative_eq!(qr.apply_q(&r), input, epsilon = 1e-12, max_relative = 1e-12);

        let rhs = matrix![1.0_f64, 2.0; 3.0, 4.0; 5.0, 6.0];
        let expected = qr.apply_q(&rhs);
        let mut actual = rhs;
        qr.apply_q_in_place(&mut actual);
        assert_relative_eq!(actual, expected, epsilon = 1e-12, max_relative = 1e-12);
        qr.apply_q_transpose_in_place(&mut actual);
        assert_relative_eq!(actual, rhs, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    fn householder_norm_avoids_overflow_and_underflow() {
        let large = Matrix::<2, 1, f64>::from_rows([[1.0e308], [1.0e308]]);
        let small = Matrix::<2, 1, f64>::from_rows([[1.0e-300], [1.0e-300]]);
        let expected_large = 2.0_f64.sqrt() * 1.0e308;
        let expected_small = 2.0_f64.sqrt() * 1.0e-300;

        assert_relative_eq!(
            super::column_norm(&large, 0, 0),
            expected_large,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            super::column_norm(&small, 0, 0),
            expected_small,
            max_relative = 1e-14
        );
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

    #[test]
    fn pivots_columns_and_solves() {
        let input = matrix![
            0.0_f64, 1.0;
            1.0, 2.0;
            2.0, 3.0;
        ];
        let rhs = matrix![-1.0_f64; 0.0; 1.0];
        let factor = input.col_piv_householder_qr();
        assert_eq!(factor.permutation(), &[1, 0]);
        assert_eq!(factor.rank(), 2);
        let solution = factor
            .solve_least_squares(&rhs)
            .expect("pivoted system is full rank");
        assert_relative_eq!(solution, matrix![2.0_f64; -1.0], max_relative = 1e-12);
        assert_relative_eq!(input * solution, rhs, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    fn reconstructs_pivoted_matrix_with_q() {
        let input = matrix![
            0.0_f64, 1.0, 2.0;
            1.0, 2.0, 4.0;
            2.0, 3.0, 8.0;
        ];
        let factor = input.col_piv_householder_qr();
        let permuted = Matrix::from_fn(|row, column| input[(row, factor.permutation()[column])]);
        assert_relative_eq!(
            factor.apply_q(&factor.r()),
            permuted,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn reports_rank_deficiency_after_pivoting() {
        let input = matrix![
            1.0_f64, 2.0;
            2.0, 4.0;
            3.0, 6.0;
        ];
        let rhs = Matrix::<3, 1, f64>::ones();
        let factor = input.col_piv_householder_qr();
        assert_eq!(factor.rank(), 1);
        assert_eq!(factor.with_threshold(1.0).rank(), 0);
        assert!(factor.solve_least_squares(&rhs).is_none());

        let solution = factor
            .solve_least_squares_basic(&rhs)
            .expect("overdetermined rank-deficient solve");
        assert_relative_eq!(solution, matrix![0.0_f64; 3.0 / 14.0], max_relative = 1e-12);
    }
}
