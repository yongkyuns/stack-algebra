use crate::kernels::MatmulBackend;
use crate::view::MatrixRead;
use crate::{DecompositionError, Matrix, MatrixScalar, Real};

/// Symmetric LDLᵀ factorization with Eigen-compatible Bunch–Kaufman pivoting.
///
/// The factorization has the form `P * A * Pᵀ = L * D * Lᵀ`, where `D` is
/// block diagonal, `L` is unit lower-triangular, and `P` is represented internally
/// by a fixed-size permutation index array. The factor matrix stores `L` below
/// the diagonal and scalar 1×1 or 2×2 `D` pivot blocks in compact form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ldlt<const D: usize, T> {
    factor: Matrix<D, D, T>,
    permutation: [usize; D],
    pivots: [u8; D],
}

impl<const D: usize, T: Real + MatrixScalar> Ldlt<D, T> {
    /// Recomputes this diagonal-pivot factorization in place.
    #[inline]
    pub fn try_compute(&mut self, matrix: &Matrix<D, D, T>) -> Result<(), DecompositionError> {
        Self::try_factorize_into(matrix, self)
    }

    /// Recomputes this no-pivot factorization in place.
    #[inline]
    pub fn try_compute_no_pivot(
        &mut self,
        matrix: &Matrix<D, D, T>,
    ) -> Result<(), DecompositionError> {
        Self::try_factorize_no_pivot_into(matrix, self)
    }

    /// Recomputes this diagonal-pivot factorization directly from a view.
    #[inline]
    pub fn try_compute_view<V>(&mut self, matrix: &V) -> Result<(), DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        *self = Self::try_decompose_view(matrix)?;
        Ok(())
    }

    /// Recomputes this no-pivot factorization directly from a view.
    #[inline]
    pub fn try_compute_no_pivot_view<V>(&mut self, matrix: &V) -> Result<(), DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        *self = Self::try_decompose_no_pivot_view(matrix)?;
        Ok(())
    }

    /// Computes an Eigen-compatible symmetric Bunch–Kaufman LDLᵀ decomposition.
    ///
    /// Returns `None` for non-finite or singular input. The input is expected
    /// to be symmetric; only its lower triangle is read.
    ///
    /// # Examples
    ///
    /// ```
    /// use stack_algebra::matrix;
    ///
    /// let a = matrix![4.0_f64, 2.0; 2.0, 3.0];
    /// let ldlt = a.try_ldlt().expect("nonsingular symmetric input");
    /// let rhs = matrix![1.0_f64; 0.0];
    /// let x = ldlt.solve(&rhs);
    /// assert!((a * x - rhs).norm() < 1.0e-12);
    /// ```
    #[inline]
    pub fn decompose(matrix: &Matrix<D, D, T>) -> Option<Self> {
        Self::try_decompose(matrix).ok()
    }

    /// Computes a symmetric Bunch–Kaufman LDLᵀ decomposition with a typed
    /// failure result.
    #[inline]
    pub fn try_decompose(matrix: &Matrix<D, D, T>) -> Result<Self, DecompositionError> {
        let mut output = Self {
            factor: *matrix,
            permutation: core::array::from_fn(|index| index),
            pivots: [1; D],
        };
        Self::decompose_impl::<true>(&mut output)?;
        Ok(output)
    }

    /// Computes a diagonal-pivot LDLᵀ factorization directly from a view.
    #[inline]
    pub fn try_decompose_view<V>(matrix: &V) -> Result<Self, DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        let mut output = Self::view_storage(matrix)?;
        Self::decompose_impl::<true>(&mut output)?;
        Ok(output)
    }

    /// Computes a symmetric diagonal-pivot LDLᵀ decomposition into caller-provided storage.
    /// On failure, `output` may contain a partial factorization.
    #[inline]
    fn try_factorize_into(
        matrix: &Matrix<D, D, T>,
        output: &mut Self,
    ) -> Result<(), DecompositionError> {
        output.factor = *matrix;
        output.permutation = core::array::from_fn(|index| index);
        output.pivots = [1; D];
        Self::decompose_impl::<true>(output)
    }

    /// Computes an LDLᵀ decomposition without diagonal pivoting.
    ///
    /// This avoids pivot-search and swap overhead, but requires every
    /// leading diagonal value to remain finite and nonzero. Use `decompose`
    /// when the matrix may need pivoting for numerical stability.
    #[inline]
    pub fn decompose_no_pivot(matrix: &Matrix<D, D, T>) -> Option<Self> {
        Self::try_decompose_no_pivot(matrix).ok()
    }

    /// Computes an LDLᵀ decomposition without diagonal pivoting with a typed
    /// failure result.
    #[inline]
    pub fn try_decompose_no_pivot(matrix: &Matrix<D, D, T>) -> Result<Self, DecompositionError> {
        let mut output = Self {
            factor: *matrix,
            permutation: core::array::from_fn(|index| index),
            pivots: [1; D],
        };
        Self::decompose_impl::<false>(&mut output)?;
        Ok(output)
    }

    /// Computes a no-pivot LDLᵀ factorization directly from a view.
    #[inline]
    pub fn try_decompose_no_pivot_view<V>(matrix: &V) -> Result<Self, DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        let mut output = Self::view_storage(matrix)?;
        Self::decompose_impl::<false>(&mut output)?;
        Ok(output)
    }

    fn view_storage<V>(matrix: &V) -> Result<Self, DecompositionError>
    where
        V: MatrixRead<D, D, T>,
    {
        let mut output = Self {
            factor: Matrix::zeros(),
            permutation: core::array::from_fn(|index| index),
            pivots: [1; D],
        };
        for column in 0..D {
            for row in column..D {
                output.factor[(row, column)] = *matrix
                    .get(row, column)
                    .ok_or(DecompositionError::InvalidView)?;
            }
        }
        Ok(output)
    }

    /// Computes an LDLᵀ decomposition without pivoting into caller-provided storage.
    /// On failure, `output` may contain a partial factorization.
    #[inline]
    fn try_factorize_no_pivot_into(
        matrix: &Matrix<D, D, T>,
        output: &mut Self,
    ) -> Result<(), DecompositionError> {
        output.factor = *matrix;
        output.permutation = core::array::from_fn(|index| index);
        output.pivots = [1; D];
        Self::decompose_impl::<false>(output)
    }

    #[inline]
    fn decompose_impl<const PIVOTING: bool>(output: &mut Self) -> Result<(), DecompositionError> {
        if PIVOTING {
            return Self::decompose_bunch_kaufman(output);
        }
        let factor = &mut output.factor;
        let permutation = &mut output.permutation;

        const BLOCK_SIZE: usize = 16;
        let mut block_start = 0;
        while block_start < D {
            let block_end = core::cmp::min(block_start + BLOCK_SIZE, D);
            for diagonal in block_start..block_end {
                let pivot = diagonal;
                let pivot_value = factor[(pivot, pivot)];
                if !pivot_value.is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
                if pivot_value == T::zero() {
                    return Err(DecompositionError::ZeroPivot);
                }
                if pivot != diagonal {
                    Self::swap_lower_rows(factor, diagonal, pivot, diagonal);
                    Self::swap_symmetric_lower(factor, diagonal, pivot, diagonal);
                    permutation.swap(diagonal, pivot);
                }

                let diagonal_value = factor[(diagonal, diagonal)];
                if !diagonal_value.is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
                if diagonal_value == T::zero() {
                    return Err(DecompositionError::ZeroPivot);
                }
                factor[(diagonal, diagonal)] = diagonal_value;

                {
                    let data = factor.as_mut_slice();
                    let column = &mut data[diagonal * D + diagonal + 1..diagonal * D + D];
                    T::Matmul::scale_divide(column, diagonal_value);
                    if column.iter().any(|value| !value.is_finite()) {
                        return Err(DecompositionError::NonFinite);
                    }
                }

                for column in (diagonal + 1)..block_end {
                    let scale = diagonal_value * factor[(column, diagonal)];
                    let data = factor.as_mut_slice();
                    let column_offset = column * D;
                    let (prefix, suffix) = data.split_at_mut(column_offset);
                    let source = &prefix[diagonal * D + column..diagonal * D + D];
                    let target = &mut suffix[column..D];
                    T::Matmul::rank_update_sub(target, source, scale);
                }
            }

            T::Matmul::symmetric_rank_k_update(factor, block_start, block_end);
            block_start = block_end;
        }

        Ok(())
    }

    fn decompose_bunch_kaufman(output: &mut Self) -> Result<(), DecompositionError> {
        let factor = &mut output.factor;
        let permutation = &mut output.permutation;
        output.pivots = [1; D];

        let alpha = T::from(0.6403882032022076).unwrap_or(T::one());
        let mut position = 0;
        while position < D {
            if position + 1 == D {
                Self::factor_one_pivot(factor, position, &mut output.pivots)?;
                break;
            }

            let abs_diagonal = factor[(position, position)].abs();
            let mut column_max = T::zero();
            let mut imax = position + 1;
            for row in (position + 1)..D {
                let magnitude = factor[(row, position)].abs();
                if magnitude > column_max {
                    column_max = magnitude;
                    imax = row;
                }
            }

            if !abs_diagonal.is_finite() || !column_max.is_finite() {
                return Err(DecompositionError::NonFinite);
            }
            if abs_diagonal >= alpha * column_max || column_max == T::zero() {
                Self::factor_one_pivot(factor, position, &mut output.pivots)?;
                position += 1;
                continue;
            }

            let mut row_max = T::zero();
            for column in position..D {
                if column != imax {
                    row_max = row_max.max(Self::lower_value(factor, imax, column).abs());
                }
            }
            let abs_imax_diagonal = factor[(imax, imax)].abs();
            if !row_max.is_finite() || !abs_imax_diagonal.is_finite() {
                return Err(DecompositionError::NonFinite);
            }
            if abs_imax_diagonal >= alpha * row_max {
                Self::swap_symmetric_lower_bunch(factor, position, imax);
                permutation.swap(position, imax);
                Self::factor_one_pivot(factor, position, &mut output.pivots)?;
                position += 1;
            } else {
                if imax != position + 1 {
                    Self::swap_symmetric_lower_bunch(factor, position + 1, imax);
                    permutation.swap(position + 1, imax);
                }
                Self::factor_two_pivot(factor, position, &mut output.pivots)?;
                position += 2;
            }
        }

        Ok(())
    }

    fn factor_one_pivot(
        factor: &mut Matrix<D, D, T>,
        position: usize,
        pivots: &mut [u8; D],
    ) -> Result<(), DecompositionError> {
        let diagonal = factor[(position, position)];
        if !diagonal.is_finite() {
            return Err(DecompositionError::NonFinite);
        }
        if diagonal == T::zero() {
            return Err(DecompositionError::ZeroPivot);
        }
        pivots[position] = 1;
        {
            let data = factor.as_mut_slice();
            let column = &mut data[position * D + position + 1..position * D + D];
            if column.len() >= 16 {
                T::Matmul::scale_divide(column, diagonal);
            } else {
                for value in column.iter_mut() {
                    *value = *value / diagonal;
                }
            }
            if column.iter().any(|value| !value.is_finite()) {
                return Err(DecompositionError::NonFinite);
            }
        }
        for column in (position + 1)..D {
            let scale = diagonal * factor[(column, position)];
            {
                let data = factor.as_mut_slice();
                let column_offset = column * D;
                let (prefix, suffix) = data.split_at_mut(column_offset);
                let source = &prefix[position * D + column..position * D + D];
                let target = &mut suffix[column..D];
                T::Matmul::rank_update_sub(target, source, scale);
            }
            for row in column..D {
                if !factor[(row, column)].is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
            }
        }
        Ok(())
    }

    fn factor_two_pivot(
        factor: &mut Matrix<D, D, T>,
        position: usize,
        pivots: &mut [u8; D],
    ) -> Result<(), DecompositionError> {
        let first = factor[(position, position)];
        let coupling = factor[(position + 1, position)];
        let second = factor[(position + 1, position + 1)];
        let determinant = first * second - coupling * coupling;
        if !first.is_finite()
            || !coupling.is_finite()
            || !second.is_finite()
            || !determinant.is_finite()
        {
            return Err(DecompositionError::NonFinite);
        }
        if determinant == T::zero() {
            return Err(DecompositionError::ZeroPivot);
        }
        pivots[position] = 2;
        pivots[position + 1] = 3;
        for row in (position + 2)..D {
            let first_value = factor[(row, position)];
            let second_value = factor[(row, position + 1)];
            let lower_first = (first_value * second - second_value * coupling) / determinant;
            let lower_second = (second_value * first - first_value * coupling) / determinant;
            if !lower_first.is_finite() || !lower_second.is_finite() {
                return Err(DecompositionError::NonFinite);
            }
            factor[(row, position)] = lower_first;
            factor[(row, position + 1)] = lower_second;
        }
        for column in (position + 2)..D {
            for row in column..D {
                let row_first = factor[(row, position)];
                let row_second = factor[(row, position + 1)];
                let column_first = factor[(column, position)];
                let column_second = factor[(column, position + 1)];
                let value = factor[(row, column)]
                    - row_first * first * column_first
                    - row_first * coupling * column_second
                    - row_second * coupling * column_first
                    - row_second * second * column_second;
                if !value.is_finite() {
                    return Err(DecompositionError::NonFinite);
                }
                factor[(row, column)] = value;
            }
        }
        Ok(())
    }

    #[inline]
    fn lower_value(matrix: &Matrix<D, D, T>, row: usize, column: usize) -> T {
        if row >= column {
            matrix[(row, column)]
        } else {
            matrix[(column, row)]
        }
    }

    fn swap_symmetric_lower_bunch(matrix: &mut Matrix<D, D, T>, first: usize, second: usize) {
        if first == second {
            return;
        }
        for column in 0..first {
            let value = matrix[(first, column)];
            matrix[(first, column)] = matrix[(second, column)];
            matrix[(second, column)] = value;
        }
        for index in (first + 1)..second {
            let value = matrix[(index, first)];
            matrix[(index, first)] = matrix[(second, index)];
            matrix[(second, index)] = value;
        }
        let diagonal = matrix[(first, first)];
        matrix[(first, first)] = matrix[(second, second)];
        matrix[(second, second)] = diagonal;
        for row in (second + 1)..D {
            let value = matrix[(row, first)];
            matrix[(row, first)] = matrix[(row, second)];
            matrix[(row, second)] = value;
        }
    }

    #[inline]
    fn swap_lower_rows(matrix: &mut Matrix<D, D, T>, first: usize, second: usize, columns: usize) {
        for column in 0..columns {
            let value = matrix[(first, column)];
            matrix[(first, column)] = matrix[(second, column)];
            matrix[(second, column)] = value;
        }
    }

    #[inline]
    fn swap_symmetric_lower(
        matrix: &mut Matrix<D, D, T>,
        first: usize,
        second: usize,
        start: usize,
    ) {
        let first_diagonal = matrix[(first, first)];
        matrix[(first, first)] = matrix[(second, second)];
        matrix[(second, second)] = first_diagonal;
        let between = matrix[(first.max(second), first.min(second))];
        for index in start..D {
            if index == first || index == second {
                continue;
            }
            let first_value = matrix[(first.max(index), first.min(index))];
            let second_value = matrix[(second.max(index), second.min(index))];
            matrix[(first.max(index), first.min(index))] = second_value;
            matrix[(second.max(index), second.min(index))] = first_value;
        }
        matrix[(first.max(second), first.min(second))] = between;
    }

    /// Returns the unit lower-triangular factor `L`.
    #[inline]
    pub fn lower(&self) -> Matrix<D, D, T> {
        Matrix::from_fn(|row, column| {
            if row > column {
                if self.pivots[column] == 2 && row == column + 1 {
                    T::zero()
                } else {
                    self.factor[(row, column)]
                }
            } else if row == column {
                T::one()
            } else {
                T::zero()
            }
        })
    }

    /// Returns the diagonal factor `D` as a column matrix.
    #[inline]
    pub fn diagonal(&self) -> Matrix<D, 1, T> {
        Matrix::from_fn(|row, _| self.factor[(row, row)])
    }

    /// Returns the compact pivot metadata: `1` for a 1×1 pivot, `2` for the
    /// first entry of a 2×2 pivot, and `3` for its second entry.
    #[inline]
    pub fn pivot_blocks(&self) -> &[u8; D] {
        &self.pivots
    }

    /// Returns the diagonal factor `D` as a square matrix.
    #[inline]
    pub fn diagonal_matrix(&self) -> Matrix<D, D, T> {
        Matrix::from_fn(|row, column| {
            if row == column {
                self.factor[(row, row)]
            } else if row == column + 1 && self.pivots[column] == 2 {
                self.factor[(row, column)]
            } else if column == row + 1 && self.pivots[row] == 2 {
                self.factor[(column, row)]
            } else {
                T::zero()
            }
        })
    }

    /// Returns the permutation indices used by the factorization.
    #[inline]
    pub fn permutation_indices(&self) -> &[usize; D] {
        &self.permutation
    }

    /// Returns the permutation matrix `P` as a fixed-size value.
    #[inline]
    pub fn permutation(&self) -> Matrix<D, D, T> {
        Matrix::from_fn(|row, column| {
            if self.permutation[row] == column {
                T::one()
            } else {
                T::zero()
            }
        })
    }

    /// Solves `A * X = B` using this decomposition.
    #[inline]
    pub fn solve<const P: usize>(&self, rhs: &Matrix<D, P, T>) -> Matrix<D, P, T> {
        let mut solution = Matrix::<D, P, T>::zeros();
        self.solve_into(rhs, &mut solution);
        solution
    }

    /// Solves `A * X = B` into a caller-provided output matrix.
    #[inline]
    pub fn solve_into<const P: usize>(&self, rhs: &Matrix<D, P, T>, output: &mut Matrix<D, P, T>) {
        *output = *rhs;
        self.solve_in_place(output);
    }

    /// Solves `A * X = B` in place.
    #[inline]
    pub fn solve_in_place<const P: usize>(&self, rhs: &mut Matrix<D, P, T>) {
        let mut transformed: Matrix<D, P, T> = Matrix::zeros();
        for column in 0..P {
            for row in 0..D {
                transformed[(row, column)] = rhs[(self.permutation[row], column)];
            }
        }

        for column in 0..P {
            let mut row = 0;
            while row < D {
                if self.pivots[row] == 2 {
                    for target in (row + 2)..D {
                        transformed[(target, column)] = transformed[(target, column)]
                            - self.factor[(target, row)] * transformed[(row, column)]
                            - self.factor[(target, row + 1)] * transformed[(row + 1, column)];
                    }
                    row += 2;
                } else {
                    for target in (row + 1)..D {
                        transformed[(target, column)] = transformed[(target, column)]
                            - self.factor[(target, row)] * transformed[(row, column)];
                    }
                    row += 1;
                }
            }

            let mut row = 0;
            while row < D {
                if self.pivots[row] == 2 {
                    let first = transformed[(row, column)];
                    let second = transformed[(row + 1, column)];
                    let d11 = self.factor[(row, row)];
                    let d12 = self.factor[(row + 1, row)];
                    let d22 = self.factor[(row + 1, row + 1)];
                    let determinant = d11 * d22 - d12 * d12;
                    transformed[(row, column)] = (first * d22 - second * d12) / determinant;
                    transformed[(row + 1, column)] = (second * d11 - first * d12) / determinant;
                    row += 2;
                } else {
                    transformed[(row, column)] =
                        transformed[(row, column)] / self.factor[(row, row)];
                    row += 1;
                }
            }

            let mut row = D;
            while row > 0 {
                if row >= 2 && self.pivots[row - 2] == 2 {
                    let first = row - 2;
                    let second = row - 1;
                    let mut first_value = transformed[(first, column)];
                    let mut second_value = transformed[(second, column)];
                    for next in row..D {
                        first_value =
                            first_value - self.factor[(next, first)] * transformed[(next, column)];
                        second_value = second_value
                            - self.factor[(next, second)] * transformed[(next, column)];
                    }
                    transformed[(first, column)] = first_value;
                    transformed[(second, column)] = second_value;
                    row -= 2;
                } else {
                    let current = row - 1;
                    let mut value = transformed[(current, column)];
                    for next in row..D {
                        value = value - self.factor[(next, current)] * transformed[(next, column)];
                    }
                    transformed[(current, column)] = value;
                    row -= 1;
                }
            }
        }

        for column in 0..P {
            for row in 0..D {
                rhs[(self.permutation[row], column)] = transformed[(row, column)];
            }
        }
    }
}

impl<const D: usize, T: Real + MatrixScalar> Matrix<D, D, T> {
    /// Computes a symmetric Bunch–Kaufman LDLᵀ factorization with a typed
    /// failure result.
    #[inline]
    pub fn try_ldlt(&self) -> Result<Ldlt<D, T>, DecompositionError> {
        Ldlt::try_decompose(self)
    }

    /// Computes a symmetric Bunch–Kaufman LDLᵀ factorization of this matrix.
    #[inline]
    pub fn ldlt(&self) -> Option<Ldlt<D, T>> {
        Ldlt::decompose(self)
    }

    /// Computes an LDLᵀ factorization without diagonal pivoting.
    #[inline]
    pub fn ldlt_no_pivot(&self) -> Option<Ldlt<D, T>> {
        Ldlt::decompose_no_pivot(self)
    }

    /// Computes an LDLᵀ factorization without diagonal pivoting with a typed
    /// failure result.
    #[inline]
    pub fn try_ldlt_no_pivot(&self) -> Result<Ldlt<D, T>, DecompositionError> {
        Ldlt::try_decompose_no_pivot(self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{matrix, DecompositionError, Ldlt, Map, Matrix};

    #[test]
    fn typed_errors_distinguish_ldlt_failures() {
        assert_eq!(
            matrix![f64::NAN].try_ldlt(),
            Err(DecompositionError::NonFinite)
        );
        assert_eq!(
            matrix![0.0_f64].try_ldlt_no_pivot(),
            Err(DecompositionError::ZeroPivot)
        );
        assert!(matrix![0.0_f64].ldlt_no_pivot().is_none());
    }

    #[test]
    fn reconstructs_symmetric_indefinite_matrix() {
        let matrix = matrix![
            0.0_f64, 1.0, 2.0;
            1.0, 3.0, 4.0;
            2.0, 4.0, 5.0;
        ];
        let factor = matrix.ldlt().expect("matrix is nonsingular");
        let transformed = factor.permutation() * matrix * factor.permutation().transpose();
        assert_relative_eq!(
            transformed,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn solves_with_diagonal_pivoting() {
        let matrix = matrix![-1.0_f64, 2.0; 2.0, 3.0];
        let rhs = matrix![1.0_f64; 4.0];
        let factor = matrix.ldlt().expect("matrix is nonsingular");
        assert_relative_eq!(matrix * factor.solve(&rhs), rhs, max_relative = 1e-12);
    }

    #[test]
    fn reuses_caller_provided_factor_storage() {
        let first = matrix![4.0_f64, 1.0; 1.0, 3.0];
        let second = matrix![0.0_f64, 2.0; 2.0, 3.0];
        let mut factor = first.ldlt().expect("first matrix is nonsingular");
        factor
            .try_compute(&second)
            .expect("second matrix is nonsingular");
        let transformed = factor.permutation() * second * factor.permutation().transpose();
        assert_relative_eq!(
            transformed,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn decomposes_map_and_block_views_without_input_copy() {
        let matrix = matrix![-1.0_f64, 2.0; 2.0, 3.0];
        let mapped = Map::<2, 2, f64>::from_slice(matrix.as_slice()).unwrap();
        let factor = Ldlt::try_decompose_view(&mapped).unwrap();
        let transformed = factor.permutation() * matrix * factor.permutation().transpose();
        assert_relative_eq!(
            transformed,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );

        let mut storage = Matrix::<3, 3, f64>::zeros();
        storage[(0, 0)] = -1.0;
        storage[(1, 0)] = 2.0;
        storage[(0, 1)] = 2.0;
        storage[(1, 1)] = 3.0;
        let block = storage.block::<2, 2>(0, 0).unwrap();
        let mut reused = factor;
        reused.try_compute_no_pivot_view(&block).unwrap();
        assert_eq!(reused.permutation_indices(), &[0, 1]);
        assert_relative_eq!(
            matrix,
            reused.lower() * reused.diagonal_matrix() * reused.lower().transpose(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn no_pivot_reconstructs_stable_matrix() {
        let matrix = matrix![
            4.0_f64, 1.0, 0.5;
            1.0, 3.0, 0.25;
            0.5, 0.25, 2.0;
        ];
        let factor = matrix.ldlt_no_pivot().expect("matrix is nonsingular");
        assert_eq!(factor.permutation_indices(), &[0, 1, 2]);
        assert_relative_eq!(
            matrix,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn diagonal_pivoting_reads_only_the_lower_triangle() {
        let matrix = matrix![4.0_f64, f64::NAN; 1.0, 3.0];
        let factor = matrix.ldlt().expect("finite lower triangle");
        let expected = matrix![4.0_f64, 1.0; 1.0, 3.0];
        let transformed = factor.permutation() * expected * factor.permutation().transpose();
        assert_relative_eq!(
            transformed,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn no_pivot_rejects_zero_leading_pivot() {
        let matrix = matrix![0.0_f64, 1.0; 1.0, 2.0];
        assert!(matrix.ldlt_no_pivot().is_none());
        assert!(matrix.ldlt().is_some());
    }

    #[test]
    fn diagonal_pivoting_handles_two_by_two_pivot() {
        let matrix = matrix![0.0_f64, 1.0; 1.0, 0.0];
        let factor = matrix.ldlt().expect("nonsingular 2x2 pivot");
        assert_eq!(factor.pivot_blocks(), &[2, 3]);
        let transformed = factor.permutation() * matrix * factor.permutation().transpose();
        assert_relative_eq!(
            transformed,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
        let rhs = matrix![2.0_f64; -3.0];
        assert_relative_eq!(matrix * factor.solve(&rhs), rhs, max_relative = 1e-12);
    }

    #[test]
    fn two_by_two_pivot_reconstructs_after_symmetric_exchange() {
        let matrix = matrix![
            0.0_f64, 0.0, 1.0;
            0.0, 2.0, 0.1;
            1.0, 0.1, 0.0;
        ];
        let factor = matrix.ldlt().expect("nonsingular permuted 2x2 pivot");
        assert_eq!(factor.pivot_blocks(), &[2, 3, 1]);
        let transformed = factor.permutation() * matrix * factor.permutation().transpose();
        assert_relative_eq!(
            transformed,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
        let rhs = matrix![1.0_f64; -2.0; 3.0];
        assert_relative_eq!(matrix * factor.solve(&rhs), rhs, max_relative = 1e-12);
    }

    #[test]
    fn reconstructs_blocked_symmetric_indefinite_matrix() {
        let matrix = Matrix::<16, 16, f64>::from_fn(|row, column| {
            if row == column {
                if row % 2 == 0 {
                    -16.0
                } else {
                    17.0
                }
            } else {
                (row + column + 1) as f64 / 29.0
            }
        });
        let factor = matrix.ldlt().expect("matrix is nonsingular");
        let transformed = factor.permutation() * matrix * factor.permutation().transpose();
        assert_relative_eq!(
            transformed,
            factor.lower() * factor.diagonal_matrix() * factor.lower().transpose(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn rejects_singular_matrix() {
        assert!(matrix![1.0_f64, 2.0; 2.0, 4.0].ldlt().is_none());
    }
}
