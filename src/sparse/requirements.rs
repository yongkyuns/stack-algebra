use super::{CscError, StaticCscOrdering, StaticCscPattern, StaticCscPermutation};

impl<const N: usize, const MAX_NNZ: usize> StaticCscPattern<N, N, MAX_NNZ> {
    /// Computes the exact natural-order sparse Cholesky factor capacity.
    ///
    /// The result is the number of entries required by the lower factor,
    /// including its diagonal. This is a structural preflight: it depends only
    /// on the validated CSC sparsity pattern, not on numeric values, and does
    /// not require selecting a tentative `MAX_L_NNZ` first.
    ///
    /// Upper-triangle entries, when present, are ignored because symmetric
    /// factorizations consume the lower triangle.
    #[inline]
    pub fn cholesky_required_factor_nnz(&self) -> usize {
        required_factor_nnz(self)
    }

    /// Computes the exact sparse Cholesky factor capacity after applying a
    /// caller-selected symmetric ordering.
    ///
    /// The permutation uses the existing fixed-capacity sparse permutation
    /// machinery and performs no heap allocation. This is useful for generated
    /// solvers that want to choose the smallest safe `MAX_L_NNZ` after an
    /// ordering has been fixed.
    #[inline]
    pub fn cholesky_required_factor_nnz_with_ordering(
        &self,
        ordering: StaticCscOrdering<N>,
    ) -> Result<usize, CscError> {
        if ordering.is_identity() {
            return Ok(self.cholesky_required_factor_nnz());
        }
        let permutation = StaticCscPermutation::from_ordering(self, ordering)?;
        Ok(required_factor_nnz(permutation.pattern_ref()))
    }
}

fn required_factor_nnz<const N: usize, const MAX_NNZ: usize>(
    pattern: &StaticCscPattern<N, N, MAX_NNZ>,
) -> usize {
    let mut row_counts = [0usize; N];
    for column in 0..N {
        let start = pattern.column_starts()[column] as usize;
        let end = pattern.column_end(column).unwrap_or(pattern.nnz());
        for index in start..end {
            let row = pattern.row_indices()[index] as usize;
            if row > column {
                row_counts[row] += 1;
            }
        }
    }

    let mut row_starts = [0usize; N];
    for row in 1..N {
        row_starts[row] = row_starts[row - 1] + row_counts[row - 1];
    }
    let mut row_cursor = row_starts;
    let mut upper_columns = [0usize; MAX_NNZ];
    for column in 0..N {
        let start = pattern.column_starts()[column] as usize;
        let end = pattern.column_end(column).unwrap_or(pattern.nnz());
        for index in start..end {
            let row = pattern.row_indices()[index] as usize;
            if row > column {
                upper_columns[row_cursor[row]] = column;
                row_cursor[row] += 1;
            }
        }
    }

    let mut visited = [usize::MAX; N];
    let mut parent = [usize::MAX; N];
    let mut column_counts = [1usize; N];
    for column in 0..N {
        visited[column] = column;
        let start = row_starts[column];
        let end = if column + 1 < N {
            row_starts[column + 1]
        } else {
            row_starts[column] + row_counts[column]
        };
        for &upper_column in &upper_columns[start..end] {
            let mut node = upper_column;
            while visited[node] != column {
                if parent[node] == usize::MAX {
                    parent[node] = column;
                }
                column_counts[node] += 1;
                visited[node] = column;
                node = parent[node];
            }
        }
    }

    column_counts.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StaticCscCholeskyPattern, StaticCscMatrix};

    #[test]
    fn preflight_matches_natural_symbolic_fill() {
        type Sparse = StaticCscMatrix<4, 4, 7, f64>;
        let matrix = Sparse::from_pattern(
            &[4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0],
            &[0, 1, 1, 2, 2, 3, 3],
            &[0, 2, 4, 6, 7],
        )
        .unwrap();
        let required = matrix.pattern().cholesky_required_factor_nnz();
        let analyzed = StaticCscCholeskyPattern::<4, 10>::analyze(&matrix).unwrap();
        assert_eq!(required, analyzed.lower().nnz());
    }

    #[test]
    fn preflight_reports_fill_beyond_input_nnz() {
        type Sparse = StaticCscMatrix<4, 4, 8, f64>;
        // A star centered on the last row creates fill among the first three
        // variables under natural elimination.
        let matrix = Sparse::from_pattern(
            &[4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0],
            &[0, 3, 1, 3, 2, 3, 3],
            &[0, 2, 4, 6, 7],
        )
        .unwrap();
        let required = matrix.pattern().cholesky_required_factor_nnz();
        assert!(required >= matrix.nnz());
    }

    #[test]
    fn ordered_preflight_matches_ordered_analysis() {
        type Sparse = StaticCscMatrix<4, 4, 8, f64>;
        let matrix = Sparse::from_pattern(
            &[4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0],
            &[0, 3, 1, 3, 2, 3, 3],
            &[0, 2, 4, 6, 7],
        )
        .unwrap();
        let ordering = StaticCscOrdering::minimum_degree(&matrix);
        let required = matrix
            .pattern()
            .cholesky_required_factor_nnz_with_ordering(ordering)
            .unwrap();
        let analyzed =
            StaticCscCholeskyPattern::<4, 10>::analyze_with_ordering(&matrix, ordering).unwrap();
        assert_eq!(required, analyzed.lower().nnz());
    }
}
