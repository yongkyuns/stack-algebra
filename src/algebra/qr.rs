use crate::view::MatrixRead;
use crate::{DecompositionError, Matrix, MatrixScalar, Real, Vector};

#[inline]
fn column_norm<const M: usize, const N: usize, T: Real + MatrixScalar>(
    matrix: &Matrix<M, N, T>,
    column: usize,
    start_row: usize,
) -> T {
    let values_start = column * M + start_row;
    let values_end = (column + 1) * M;
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
    for row in start_row..M {
        let ratio = matrix[(row, column)].abs() / max_abs;
        scaled_sum = scaled_sum + ratio * ratio;
    }
    max_abs * scaled_sum.sqrt()
}

#[inline]
fn finite_matrix<const M: usize, const N: usize, T: Real>(matrix: &Matrix<M, N, T>) -> bool {
    matrix.as_slice().iter().all(|value| value.is_finite())
}

#[inline]
fn apply_reflector_in_place<
    const M: usize,
    const N: usize,
    const P: usize,
    T: Real + MatrixScalar,
>(
    factors: &Matrix<M, N, T>,
    column: usize,
    coefficient: T,
    transformed: &mut Matrix<M, P, T>,
    rhs_column: usize,
) {
    let tail_len = M - column - 1;
    let dot = if tail_len <= 8 {
        let mut dot = transformed[(column, rhs_column)];
        for row in (column + 1)..M {
            dot = dot + factors[(row, column)] * transformed[(row, rhs_column)];
        }
        dot
    } else {
        let source_start = column * M + column + 1;
        let source_end = source_start + tail_len;
        let target_start = rhs_column * M + column + 1;
        let target_end = target_start + tail_len;
        T::dot_accumulate(
            &factors.as_slice()[source_start..source_end],
            &transformed.as_slice()[target_start..target_end],
            transformed[(column, rhs_column)],
        )
    };
    let scale = coefficient * dot;
    if scale.is_finite() {
        transformed[(column, rhs_column)] = transformed[(column, rhs_column)] - scale;
        if tail_len <= 8 {
            for row in (column + 1)..M {
                transformed[(row, rhs_column)] =
                    transformed[(row, rhs_column)] - scale * factors[(row, column)];
            }
        } else {
            let source_start = column * M + column + 1;
            let target_start = rhs_column * M + column + 1;
            T::rank_update_sub(
                &mut transformed.as_mut_slice()[target_start..target_start + tail_len],
                &factors.as_slice()[source_start..source_start + tail_len],
                scale,
            );
        }
        return;
    }

    let mut normalization = T::zero();
    for row in column..M {
        normalization = normalization.max(transformed[(row, rhs_column)].abs());
    }
    if !normalization.is_finite() || normalization == T::zero() {
        transformed[(column, rhs_column)] = transformed[(column, rhs_column)] - scale;
        for row in (column + 1)..M {
            transformed[(row, rhs_column)] =
                transformed[(row, rhs_column)] - scale * factors[(row, column)];
        }
        return;
    }

    for row in column..M {
        transformed[(row, rhs_column)] = transformed[(row, rhs_column)] / normalization;
    }
    let mut normalized_dot = transformed[(column, rhs_column)];
    for row in (column + 1)..M {
        normalized_dot = normalized_dot + factors[(row, column)] * transformed[(row, rhs_column)];
    }
    let normalized_scale = coefficient * normalized_dot;
    transformed[(column, rhs_column)] = transformed[(column, rhs_column)] - normalized_scale;
    for row in (column + 1)..M {
        transformed[(row, rhs_column)] =
            transformed[(row, rhs_column)] - normalized_scale * factors[(row, column)];
    }
    for row in column..M {
        transformed[(row, rhs_column)] = transformed[(row, rhs_column)] * normalization;
    }
}

#[inline]
fn apply_q_transpose_in_place<
    const M: usize,
    const N: usize,
    const P: usize,
    T: Real + MatrixScalar,
>(
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
            apply_reflector_in_place(factors, column, coefficient, transformed, rhs_column);
        }
    }
}

#[inline]
fn apply_q_in_place<const M: usize, const N: usize, const P: usize, T: Real + MatrixScalar>(
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
            apply_reflector_in_place(factors, column, coefficient, transformed, rhs_column);
        }
    }
}

/// Householder QR factorization of a fixed-size matrix.
///
/// The factorization stores the upper-triangular `R` entries in the upper
/// triangle and the Householder vectors below the diagonal. The vectors and
/// their coefficients are sufficient to apply `Qᵀ` without materializing `Q`.
///
/// # Examples
///
/// ```
/// use stack_algebra::matrix;
///
/// let a = matrix![1.0_f64, 0.0; 0.0, 1.0; 1.0, 1.0];
/// let rhs = matrix![2.0_f64; 3.0; 5.0];
/// let qr = a.householder_qr();
/// let x = qr.solve_least_squares(&rhs).expect("full column rank");
/// assert!((a * x - rhs).norm() < 1.0e-12);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HouseholderQr<const M: usize, const N: usize, T> {
    factors: Matrix<M, N, T>,
    coefficients: Vector<N, T>,
}

impl<const M: usize, const N: usize, T: Real + MatrixScalar> HouseholderQr<M, N, T> {
    /// Recomputes this factorization in place.
    #[inline]
    pub fn compute(&mut self, matrix: &Matrix<M, N, T>) {
        Self::factorize_into(matrix, self);
    }

    /// Recomputes this factorization with typed failure reporting.
    #[inline]
    pub fn try_compute(&mut self, matrix: &Matrix<M, N, T>) -> Result<(), DecompositionError> {
        let factor = Self::try_decompose(matrix)?;
        *self = factor;
        Ok(())
    }

    /// Recomputes this factorization directly from a fixed-size matrix view.
    #[inline]
    pub fn try_compute_view<V>(&mut self, matrix: &V) -> Result<(), DecompositionError>
    where
        V: MatrixRead<M, N, T>,
    {
        *self = Self::try_decompose_view(matrix)?;
        Ok(())
    }

    /// Computes a Householder QR factorization.
    #[inline]
    pub fn decompose(matrix: &Matrix<M, N, T>) -> Self {
        let mut output = Self {
            factors: *matrix,
            coefficients: Vector::<N, T>::zeros(),
        };
        Self::factorize(&mut output);
        output
    }

    /// Computes a Householder QR factorization with typed failure reporting.
    #[inline]
    pub fn try_decompose(matrix: &Matrix<M, N, T>) -> Result<Self, DecompositionError> {
        if !finite_matrix(matrix) {
            return Err(DecompositionError::NonFinite);
        }
        let output = Self::decompose(matrix);
        if !finite_matrix(&output.factors)
            || !output
                .coefficients
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(DecompositionError::NonFinite);
        }
        Ok(output)
    }

    /// Computes a Householder QR factorization directly from a fixed-size
    /// matrix view without materializing a separate owning input matrix.
    #[inline]
    pub fn try_decompose_view<V>(matrix: &V) -> Result<Self, DecompositionError>
    where
        V: MatrixRead<M, N, T>,
    {
        let mut output = Self {
            factors: Matrix::zeros(),
            coefficients: Vector::<N, T>::zeros(),
        };
        for column in 0..N {
            for row in 0..M {
                output.factors[(row, column)] = *matrix
                    .get(row, column)
                    .ok_or(DecompositionError::InvalidView)?;
            }
        }
        if !finite_matrix(&output.factors) {
            return Err(DecompositionError::NonFinite);
        }
        Self::factorize(&mut output);
        if !finite_matrix(&output.factors)
            || !output
                .coefficients
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(DecompositionError::NonFinite);
        }
        Ok(output)
    }

    /// Computes a Householder QR factorization into caller-provided storage.
    #[inline]
    fn factorize_into(matrix: &Matrix<M, N, T>, output: &mut Self) {
        output.factors = *matrix;
        output.coefficients = Vector::<N, T>::zeros();
        Self::factorize(output);
    }

    #[inline]
    fn factorize(output: &mut Self) {
        let factors = &mut output.factors;
        let coefficients = &mut output.coefficients;
        let limit = core::cmp::min(M, N);
        for column in 0..limit {
            let norm = column_norm(factors, column, column);
            if !norm.is_finite() || norm == T::zero() {
                continue;
            }

            let diagonal = factors[(column, column)];
            let beta = if diagonal >= T::zero() { -norm } else { norm };
            let first = diagonal - beta;
            if first == T::zero() {
                continue;
            }
            let scaled = !first.is_finite();
            let (normalized_beta, normalized_diagonal, normalized_first) = if scaled {
                let normalized_diagonal = diagonal / norm;
                let normalized_beta = if diagonal >= T::zero() {
                    -T::one()
                } else {
                    T::one()
                };
                (
                    normalized_beta,
                    normalized_diagonal,
                    normalized_diagonal - normalized_beta,
                )
            } else {
                (T::zero(), T::zero(), T::zero())
            };
            if scaled && (normalized_first == T::zero() || !normalized_first.is_finite()) {
                continue;
            }

            factors[(column, column)] = beta;
            let source_start = column * M + column + 1;
            let source_end = (column + 1) * M;
            if !scaled {
                T::scale_divide(&mut factors.as_mut_slice()[source_start..source_end], first);
            } else {
                T::scale_divide(&mut factors.as_mut_slice()[source_start..source_end], norm);
                T::scale_divide(
                    &mut factors.as_mut_slice()[source_start..source_end],
                    normalized_first,
                );
            }

            let coefficient = if !scaled {
                (beta - diagonal) / beta
            } else {
                (normalized_beta - normalized_diagonal) / normalized_beta
            };
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
                    T::dot_accumulate(source, target, initial)
                };
                let scale = coefficient * dot;
                factors[(column, trailing_column)] = factors[(column, trailing_column)] - scale;
                let target_start = trailing_column * M + column + 1;
                let (prefix, suffix) = factors.as_mut_slice().split_at_mut(source_end);
                let source = &prefix[source_start..source_end];
                let target_offset = target_start - source_end;
                T::rank_update_sub(
                    &mut suffix[target_offset..target_offset + source.len()],
                    source,
                    scale,
                );
            }
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
        self.try_solve_least_squares(rhs).ok()
    }

    /// Solves a full-rank least-squares problem with a typed failure result.
    #[inline]
    pub fn try_solve_least_squares<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
    ) -> Result<Matrix<N, P, T>, DecompositionError> {
        let mut solution = Matrix::<N, P, T>::zeros();
        self.try_solve_least_squares_into(rhs, &mut solution)?;
        Ok(solution)
    }

    /// Solves a full-rank least-squares problem into a caller-provided output.
    #[inline]
    pub fn solve_least_squares_into<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
        output: &mut Matrix<N, P, T>,
    ) -> Option<()> {
        self.try_solve_least_squares_into(rhs, output).ok()
    }

    /// Solves a full-rank least-squares problem into caller-provided storage
    /// with a typed failure result.
    #[inline]
    pub fn try_solve_least_squares_into<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
        output: &mut Matrix<N, P, T>,
    ) -> Result<(), DecompositionError> {
        if M < N {
            return Err(DecompositionError::Singular);
        }
        if !finite_matrix(rhs) {
            return Err(DecompositionError::NonFinite);
        }

        let transformed = self.apply_q_transpose(rhs);
        if !finite_matrix(&transformed) {
            return Err(DecompositionError::NonFinite);
        }
        let mut diagonal_scale = T::one();
        for row in 0..N {
            let diagonal = self.factors[(row, row)];
            if !diagonal.is_finite() {
                return Err(DecompositionError::NonFinite);
            }
            diagonal_scale = diagonal_scale.max(diagonal.abs());
        }
        let dimension = T::from(core::cmp::max(M, N)).unwrap_or(T::one());
        let tolerance = T::epsilon() * dimension * diagonal_scale;
        for rhs_column in 0..P {
            for row in (0..N).rev() {
                let diagonal = self.factors[(row, row)];
                if diagonal.abs() <= tolerance {
                    return Err(DecompositionError::Singular);
                }
                let mut value = transformed[(row, rhs_column)];
                for next in (row + 1)..N {
                    value = value - self.factors[(row, next)] * output[(next, rhs_column)];
                }
                let result = value / diagonal;
                if !result.is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
                output[(row, rhs_column)] = result;
            }
        }
        Ok(())
    }
}

impl<const M: usize, const N: usize, T: Real + MatrixScalar> Matrix<M, N, T> {
    /// Computes a Householder QR factorization of this matrix.
    #[inline]
    pub fn householder_qr(&self) -> HouseholderQr<M, N, T> {
        HouseholderQr::decompose(self)
    }

    /// Computes a checked Householder QR factorization.
    #[inline]
    pub fn try_householder_qr(&self) -> Result<HouseholderQr<M, N, T>, DecompositionError> {
        HouseholderQr::try_decompose(self)
    }

    /// Computes a column-pivoted Householder QR factorization of this matrix.
    #[inline]
    pub fn col_piv_householder_qr(&self) -> ColPivHouseholderQr<M, N, T> {
        ColPivHouseholderQr::decompose(self)
    }

    /// Computes a checked column-pivoted Householder QR factorization.
    #[inline]
    pub fn try_col_piv_householder_qr(
        &self,
    ) -> Result<ColPivHouseholderQr<M, N, T>, DecompositionError> {
        ColPivHouseholderQr::try_decompose(self)
    }
}

/// Column-pivoted Householder QR factorization of a fixed-size matrix.
///
/// The factorization satisfies `A * P = Q * R`, where `P` is represented by
/// [`Self::permutation`]. Column pivoting improves rank detection and the
/// numerical behavior of least-squares solves for ill-conditioned inputs.
///
/// # Examples
///
/// ```
/// use stack_algebra::matrix;
///
/// let a = matrix![1.0_f64, 0.0; 0.0, 1.0; 1.0, 1.0];
/// let rhs = matrix![2.0_f64; 3.0; 5.0];
/// let qr = a.col_piv_householder_qr();
/// let x = qr.solve_least_squares(&rhs).expect("full column rank");
/// assert!((a * x - rhs).norm() < 1.0e-12);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColPivHouseholderQr<const M: usize, const N: usize, T> {
    factors: Matrix<M, N, T>,
    coefficients: Vector<N, T>,
    permutation: [usize; N],
    max_pivot: T,
    threshold: T,
}

impl<const M: usize, const N: usize, T: Real + MatrixScalar> ColPivHouseholderQr<M, N, T> {
    /// Recomputes this column-pivoted factorization in place.
    #[inline]
    pub fn compute(&mut self, matrix: &Matrix<M, N, T>) {
        Self::factorize_into(matrix, self);
    }

    /// Recomputes this factorization with typed failure reporting.
    #[inline]
    pub fn try_compute(&mut self, matrix: &Matrix<M, N, T>) -> Result<(), DecompositionError> {
        let factor = Self::try_decompose(matrix)?;
        *self = factor;
        Ok(())
    }

    /// Recomputes this factorization directly from a fixed-size matrix view.
    #[inline]
    pub fn try_compute_view<V>(&mut self, matrix: &V) -> Result<(), DecompositionError>
    where
        V: MatrixRead<M, N, T>,
    {
        *self = Self::try_decompose_view(matrix)?;
        Ok(())
    }

    /// Computes a column-pivoted Householder QR factorization.
    #[inline]
    pub fn decompose(matrix: &Matrix<M, N, T>) -> Self {
        let mut output = Self {
            factors: *matrix,
            coefficients: Vector::<N, T>::zeros(),
            permutation: core::array::from_fn(|index| index),
            max_pivot: T::zero(),
            threshold: T::zero(),
        };
        Self::factorize(&mut output);
        output
    }

    /// Computes a column-pivoted Householder QR factorization with typed
    /// failure reporting.
    #[inline]
    pub fn try_decompose(matrix: &Matrix<M, N, T>) -> Result<Self, DecompositionError> {
        if !finite_matrix(matrix) {
            return Err(DecompositionError::NonFinite);
        }
        let output = Self::decompose(matrix);
        if !finite_matrix(&output.factors)
            || !output
                .coefficients
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
            || !output.max_pivot.is_finite()
            || !output.threshold.is_finite()
        {
            return Err(DecompositionError::NonFinite);
        }
        Ok(output)
    }

    /// Computes a column-pivoted Householder QR factorization directly from a
    /// fixed-size matrix view without materializing a separate owning input
    /// matrix.
    #[inline]
    pub fn try_decompose_view<V>(matrix: &V) -> Result<Self, DecompositionError>
    where
        V: MatrixRead<M, N, T>,
    {
        let mut output = Self {
            factors: Matrix::zeros(),
            coefficients: Vector::<N, T>::zeros(),
            permutation: core::array::from_fn(|index| index),
            max_pivot: T::zero(),
            threshold: T::zero(),
        };
        for column in 0..N {
            for row in 0..M {
                output.factors[(row, column)] = *matrix
                    .get(row, column)
                    .ok_or(DecompositionError::InvalidView)?;
            }
        }
        if !finite_matrix(&output.factors) {
            return Err(DecompositionError::NonFinite);
        }
        Self::factorize(&mut output);
        if !finite_matrix(&output.factors)
            || !output
                .coefficients
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
            || !output.max_pivot.is_finite()
            || !output.threshold.is_finite()
        {
            return Err(DecompositionError::NonFinite);
        }
        Ok(output)
    }

    /// Computes a column-pivoted Householder QR factorization into caller-provided storage.
    #[inline]
    fn factorize_into(matrix: &Matrix<M, N, T>, output: &mut Self) {
        output.factors = *matrix;
        output.coefficients = Vector::<N, T>::zeros();
        output.permutation = core::array::from_fn(|index| index);
        Self::factorize(output);
    }

    #[inline]
    fn factorize(output: &mut Self) {
        let factors = &mut output.factors;
        let coefficients = &mut output.coefficients;
        let permutation = &mut output.permutation;
        let mut direct_norms: [T; N] =
            core::array::from_fn(|column| column_norm(factors, column, 0));
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
            if first == T::zero() {
                continue;
            }
            let scaled = !first.is_finite();
            let (normalized_beta, normalized_diagonal, normalized_first) = if scaled {
                let normalized_diagonal = diagonal / pivot_norm;
                let normalized_beta = if diagonal >= T::zero() {
                    -T::one()
                } else {
                    T::one()
                };
                (
                    normalized_beta,
                    normalized_diagonal,
                    normalized_diagonal - normalized_beta,
                )
            } else {
                (T::zero(), T::zero(), T::zero())
            };
            if scaled && (normalized_first == T::zero() || !normalized_first.is_finite()) {
                continue;
            }

            factors[(column, column)] = beta;
            let source_start = column * M + column + 1;
            let source_end = (column + 1) * M;
            if !scaled {
                T::scale_divide(&mut factors.as_mut_slice()[source_start..source_end], first);
            } else {
                T::scale_divide(
                    &mut factors.as_mut_slice()[source_start..source_end],
                    pivot_norm,
                );
                T::scale_divide(
                    &mut factors.as_mut_slice()[source_start..source_end],
                    normalized_first,
                );
            }

            let coefficient = if !scaled {
                (beta - diagonal) / beta
            } else {
                (normalized_beta - normalized_diagonal) / normalized_beta
            };
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
                    T::dot_accumulate(source, target, initial)
                };
                let scale = coefficient * dot;
                factors[(column, trailing_column)] = factors[(column, trailing_column)] - scale;
                let target_start = trailing_column * M + column + 1;
                let (prefix, suffix) = factors.as_mut_slice().split_at_mut(source_end);
                let source = &prefix[source_start..source_end];
                let target_offset = target_start - source_end;
                T::rank_update_sub(
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
                        let direct_norm = column_norm(factors, trailing_column, column + 1);
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

        output.max_pivot = max_pivot;
        output.threshold = T::epsilon() * dimension;
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
        let mut solution = Matrix::<N, P, T>::zeros();
        self.try_solve_least_squares_into(rhs, &mut solution).ok()?;
        Some(solution)
    }

    /// Solves a full-rank least-squares problem with a typed failure result.
    #[inline]
    pub fn try_solve_least_squares<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
    ) -> Result<Matrix<N, P, T>, DecompositionError> {
        let mut solution = Matrix::<N, P, T>::zeros();
        self.try_solve_least_squares_into(rhs, &mut solution)?;
        Ok(solution)
    }

    /// Solves a full-rank least-squares problem into a caller-provided output.
    #[inline]
    pub fn solve_least_squares_into<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
        output: &mut Matrix<N, P, T>,
    ) -> Option<()> {
        self.try_solve_least_squares_into(rhs, output).ok()
    }

    /// Solves a full-rank least-squares problem into caller-provided storage
    /// with a typed failure result.
    #[inline]
    pub fn try_solve_least_squares_into<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
        output: &mut Matrix<N, P, T>,
    ) -> Result<(), DecompositionError> {
        if M < N {
            return Err(DecompositionError::Singular);
        }
        if !finite_matrix(rhs) || !finite_matrix(&self.factors) {
            return Err(DecompositionError::NonFinite);
        }
        if !self.max_pivot.is_finite() || !self.threshold.is_finite() {
            return Err(DecompositionError::NonFinite);
        }
        if self.rank() < N {
            return Err(DecompositionError::Singular);
        }

        let transformed = self.apply_q_transpose(rhs);
        if !finite_matrix(&transformed) {
            return Err(DecompositionError::NonFinite);
        }
        let mut permuted_solution = Matrix::<N, P, T>::zeros();
        for rhs_column in 0..P {
            for row in (0..N).rev() {
                let diagonal = self.factors[(row, row)];
                if diagonal == T::zero() {
                    return Err(DecompositionError::Singular);
                }
                let mut value = transformed[(row, rhs_column)];
                for next in (row + 1)..N {
                    value =
                        value - self.factors[(row, next)] * permuted_solution[(next, rhs_column)];
                }
                let result = value / diagonal;
                if !result.is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
                permuted_solution[(row, rhs_column)] = result;
            }
        }

        *output = Matrix::zeros();
        for column in 0..N {
            for rhs_column in 0..P {
                output[(self.permutation[column], rhs_column)] =
                    permuted_solution[(column, rhs_column)];
            }
        }
        Ok(())
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
        let mut solution = Matrix::<N, P, T>::zeros();
        self.solve_least_squares_basic_into(rhs, &mut solution)?;
        Some(solution)
    }

    /// Solves a least-squares problem using detected pivots into a caller output.
    #[inline]
    pub fn solve_least_squares_basic_into<const P: usize>(
        &self,
        rhs: &Matrix<M, P, T>,
        output: &mut Matrix<N, P, T>,
    ) -> Option<()> {
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

        *output = Matrix::zeros();
        for column in 0..N {
            for rhs_column in 0..P {
                output[(self.permutation[column], rhs_column)] =
                    permuted_solution[(column, rhs_column)];
            }
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{matrix, ColPivHouseholderQr, DecompositionError, HouseholderQr, Map, Matrix};

    #[test]
    fn typed_householder_qr_errors_are_distinguishable() {
        let rank_deficient = matrix![
            1.0_f64, 2.0;
            2.0, 4.0;
            3.0, 6.0;
        ];
        let rhs = Matrix::<3, 1, f64>::ones();
        assert_eq!(
            rank_deficient
                .householder_qr()
                .try_solve_least_squares(&rhs),
            Err(DecompositionError::Singular)
        );
        assert!(rank_deficient
            .householder_qr()
            .solve_least_squares(&rhs)
            .is_none());

        let underdetermined = Matrix::<2, 3, f64>::ones();
        let rhs = Matrix::<2, 1, f64>::ones();
        assert_eq!(
            underdetermined
                .householder_qr()
                .try_solve_least_squares(&rhs),
            Err(DecompositionError::Singular)
        );

        let non_finite = matrix![f64::NAN; 1.0; 2.0];
        let rhs = Matrix::<3, 1, f64>::ones();
        assert_eq!(
            non_finite.try_householder_qr(),
            Err(DecompositionError::NonFinite)
        );
        assert_eq!(
            non_finite.householder_qr().try_solve_least_squares(&rhs),
            Err(DecompositionError::NonFinite)
        );
    }

    #[test]
    fn typed_col_pivoted_qr_errors_are_distinguishable() {
        let rank_deficient = matrix![
            1.0_f64, 2.0;
            2.0, 4.0;
            3.0, 6.0;
        ];
        let rhs = Matrix::<3, 1, f64>::ones();
        assert_eq!(
            rank_deficient
                .col_piv_householder_qr()
                .try_solve_least_squares(&rhs),
            Err(DecompositionError::Singular)
        );

        let underdetermined = Matrix::<2, 3, f64>::ones();
        let rhs = Matrix::<2, 1, f64>::ones();
        assert_eq!(
            underdetermined
                .col_piv_householder_qr()
                .try_solve_least_squares(&rhs),
            Err(DecompositionError::Singular)
        );

        let non_finite = matrix![f64::NAN; 1.0; 2.0];
        let rhs = Matrix::<3, 1, f64>::ones();
        assert_eq!(
            non_finite.try_col_piv_householder_qr(),
            Err(DecompositionError::NonFinite)
        );
        assert_eq!(
            non_finite
                .col_piv_householder_qr()
                .try_solve_least_squares(&rhs),
            Err(DecompositionError::NonFinite)
        );
    }

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
    fn reconstructs_packetized_square_matrix() {
        let input = Matrix::<16, 16, f64>::from_fn(|row, column| {
            if row == column {
                20.0 + row as f64
            } else {
                (row + 2 * column + 1) as f64 / 19.0
            }
        });
        let qr = input.householder_qr();
        assert_relative_eq!(
            qr.apply_q(&qr.r()),
            input,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
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
    fn reuses_caller_provided_householder_storage() {
        let first = matrix![1.0_f64, 2.0; 3.0, 4.0; 5.0, 7.0];
        let second = matrix![2.0_f64, -1.0; 0.0, 3.0; 4.0, 5.0];
        let mut factor = first.householder_qr();
        factor.compute(&second);
        assert_relative_eq!(
            factor.apply_q(&factor.r()),
            second,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn reuses_caller_provided_pivoted_householder_storage() {
        let first = matrix![1.0_f64, 2.0; 3.0, 4.0; 5.0, 7.0];
        let second = matrix![0.0_f64, 1.0; 1.0, 2.0; 2.0, 3.0];
        let mut factor = first.col_piv_householder_qr();
        factor.compute(&second);
        let permuted = Matrix::from_fn(|row, column| second[(row, factor.permutation()[column])]);
        assert_relative_eq!(
            factor.apply_q(&factor.r()),
            permuted,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn decomposes_householder_qr_map_and_block_views() {
        let matrix = matrix![
            1.0_f64, 2.0;
            3.0, 4.0;
            5.0, 7.0;
        ];
        let mapped = Map::<3, 2, f64>::from_slice(matrix.as_slice()).unwrap();
        let mapped_factor = HouseholderQr::try_decompose_view(&mapped).unwrap();
        assert_relative_eq!(
            mapped_factor.apply_q(&mapped_factor.r()),
            matrix,
            epsilon = 1e-12,
            max_relative = 1e-12
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
        assert_relative_eq!(
            reused.apply_q(&reused.r()),
            matrix,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn decomposes_pivoted_qr_map_and_block_views() {
        let matrix = matrix![
            1.0_f64, 2.0;
            3.0, 4.0;
            5.0, 7.0;
        ];
        let mapped = Map::<3, 2, f64>::from_slice(matrix.as_slice()).unwrap();
        let mapped_factor = ColPivHouseholderQr::try_decompose_view(&mapped).unwrap();
        let permuted =
            Matrix::from_fn(|row, column| matrix[(row, mapped_factor.permutation()[column])]);
        assert_relative_eq!(
            mapped_factor.apply_q(&mapped_factor.r()),
            permuted,
            epsilon = 1e-12,
            max_relative = 1e-12
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
        let permuted = Matrix::from_fn(|row, column| matrix[(row, reused.permutation()[column])]);
        assert_relative_eq!(
            reused.apply_q(&reused.r()),
            permuted,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
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
    fn householder_reflector_scales_near_f64_limit() {
        let input = matrix![
            -1.0e308_f64;
            1.0e308;
        ];
        let qr = input.householder_qr();
        assert!(qr
            .factors()
            .as_slice()
            .iter()
            .all(|value| value.is_finite()));
        assert_relative_eq!(qr.apply_q(&qr.r()), input, max_relative = 1e-12);
        let transformed = qr.apply_q_transpose(&input);
        assert_relative_eq!(transformed[(0, 0)], qr.r()[(0, 0)], max_relative = 1e-12);
        assert!(transformed[(1, 0)].abs() < 1.0e293);
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
