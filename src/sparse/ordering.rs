use crate::Zero;

use super::errors::CscError;
use super::storage::{StaticCscMatrix, StaticCscPattern};

/// A reusable sparse-coordinate permutation for a validated CSC pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticCscPermutation<const N: usize, const MAX_NNZ: usize> {
    pattern: StaticCscPattern<N, N, MAX_NNZ>,
    source_indices: [usize; MAX_NNZ],
    nnz: usize,
}

impl<const N: usize, const MAX_NNZ: usize> StaticCscPermutation<N, MAX_NNZ> {
    /// Builds a reusable map from an original lower CSC pattern.
    #[inline]
    pub fn from_ordering(
        matrix_pattern: &StaticCscPattern<N, N, MAX_NNZ>,
        ordering: StaticCscOrdering<N>,
    ) -> Result<Self, CscError> {
        let mut entries = [(0usize, 0usize, 0usize); MAX_NNZ];
        let mut entry_count = 0;
        for column in 0..N {
            let start = matrix_pattern.column_starts()[column];
            let end = matrix_pattern
                .column_end(column)
                .unwrap_or(matrix_pattern.nnz());
            for source_index in start..end {
                let row = matrix_pattern.row_indices()[source_index];
                if row < column {
                    continue;
                }
                let ordered_row = ordering.inverse()[row];
                let ordered_column = ordering.inverse()[column];
                let (lower_row, lower_column) = if ordered_row >= ordered_column {
                    (ordered_row, ordered_column)
                } else {
                    (ordered_column, ordered_row)
                };
                entries[entry_count] = (lower_column, lower_row, source_index);
                entry_count += 1;
            }
        }
        entries[..entry_count].sort_unstable_by_key(|&(column, row, _)| (column, row));

        let mut row_indices = [0usize; MAX_NNZ];
        let mut column_starts = [0usize; N];
        let mut source_indices = [0usize; MAX_NNZ];
        for (target_index, &(column, row, source_index)) in
            entries[..entry_count].iter().enumerate()
        {
            row_indices[target_index] = row;
            source_indices[target_index] = source_index;
            if column + 1 < N {
                column_starts[column + 1] += 1;
            }
        }
        for column in 0..N {
            if column + 1 < N {
                column_starts[column + 1] += column_starts[column];
            }
        }
        let pattern =
            StaticCscPattern::from_parts(&row_indices[..entry_count], &column_starts, entry_count)?;
        Ok(Self {
            pattern,
            source_indices,
            nnz: entry_count,
        })
    }

    /// Returns the ordered pattern.
    #[inline]
    pub const fn pattern(&self) -> StaticCscPattern<N, N, MAX_NNZ> {
        self.pattern
    }

    /// Applies the precomputed coordinate map to a matrix's numeric values.
    #[inline]
    pub fn apply<T: Copy + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
    ) -> StaticCscMatrix<N, N, MAX_NNZ, T> {
        let mut output = StaticCscMatrix::zero_with_pattern(self.pattern);
        self.apply_into(matrix, &mut output);
        output
    }

    /// Applies the precomputed coordinate map into caller-provided storage.
    #[inline]
    pub fn apply_into<T: Copy + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
        output: &mut StaticCscMatrix<N, N, MAX_NNZ, T>,
    ) {
        for target_index in 0..self.nnz {
            output.values_mut()[target_index] = matrix.values()[self.source_indices[target_index]];
        }
    }
}

/// A fixed-capacity symmetric permutation for sparse factorization.
///
/// The permutation is represented as an ordered-to-original map. Use
/// [`Self::minimum_degree`] for a deterministic fill-reducing heuristic, or
/// [`Self::from_permutation`] when an application already has an ordering.
/// The inverse map is retained so factor solves can move right-hand sides
/// back to the caller's original coordinates without allocating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticCscOrdering<const N: usize> {
    permutation: [usize; N],
    inverse: [usize; N],
}

impl<const N: usize> StaticCscOrdering<N> {
    /// Returns the exact inline storage footprint of this ordering.
    #[inline]
    pub const fn storage_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Returns the identity ordering.
    #[inline]
    pub const fn identity() -> Self {
        let mut permutation = [0; N];
        let mut index = 0;
        while index < N {
            permutation[index] = index;
            index += 1;
        }
        Self {
            permutation,
            inverse: permutation,
        }
    }

    /// Validates an ordering given as ordered-index to original-index maps.
    #[inline]
    pub fn from_permutation(permutation: &[usize]) -> Result<Self, CscError> {
        if permutation.len() != N {
            return Err(CscError::LengthMismatch);
        }
        let mut output = Self {
            permutation: [0; N],
            inverse: [0; N],
        };
        let mut seen = [false; N];
        for (ordered, &original) in permutation.iter().enumerate() {
            if original >= N || seen[original] {
                return Err(CscError::InvalidPermutation);
            }
            seen[original] = true;
            output.permutation[ordered] = original;
            output.inverse[original] = ordered;
        }
        Ok(output)
    }

    /// Computes a deterministic fixed-workspace minimum-degree ordering.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub fn minimum_degree<const MAX_NNZ: usize, T: Copy + Zero>(
        matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
    ) -> Self {
        Self::minimum_degree_from_pattern(matrix.pattern())
    }

    /// Computes minimum degree directly from a validated symmetric CSC pattern.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub fn minimum_degree_from_pattern<const MAX_NNZ: usize>(
        pattern: &StaticCscPattern<N, N, MAX_NNZ>,
    ) -> Self {
        let mut adjacency = [[false; N]; N];
        for column in 0..N {
            let start = pattern.column_starts()[column];
            let end = pattern.column_end(column).unwrap_or(pattern.nnz());
            for index in start..end {
                let row = pattern.row_indices()[index];
                if row != column {
                    adjacency[row][column] = true;
                    adjacency[column][row] = true;
                }
            }
        }

        Self::minimum_degree_from_adjacency(adjacency)
    }

    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn minimum_degree_from_adjacency(mut adjacency: [[bool; N]; N]) -> Self {
        let mut eliminated = [false; N];
        let mut permutation = [0; N];
        for slot in permutation.iter_mut() {
            let mut selected = 0;
            let mut selected_degree = usize::MAX;
            for candidate in 0..N {
                if !eliminated[candidate] {
                    let degree = (0..N)
                        .filter(|&neighbor| !eliminated[neighbor] && adjacency[candidate][neighbor])
                        .count();
                    if degree < selected_degree {
                        selected = candidate;
                        selected_degree = degree;
                    }
                }
            }
            *slot = selected;
            eliminated[selected] = true;

            let mut neighbors = [0; N];
            let mut neighbor_count = 0;
            for neighbor in 0..N {
                if !eliminated[neighbor] && adjacency[selected][neighbor] {
                    neighbors[neighbor_count] = neighbor;
                    neighbor_count += 1;
                }
            }
            for left in 0..neighbor_count {
                for right in (left + 1)..neighbor_count {
                    let first = neighbors[left];
                    let second = neighbors[right];
                    adjacency[first][second] = true;
                    adjacency[second][first] = true;
                }
            }
        }
        Self::from_permutation(&permutation).expect("minimum-degree ordering is a permutation")
    }

    /// Returns the ordered-index to original-index map.
    #[inline]
    pub const fn permutation(&self) -> &[usize; N] {
        &self.permutation
    }

    /// Returns the original-index to ordered-index map.
    #[inline]
    pub const fn inverse(&self) -> &[usize; N] {
        &self.inverse
    }

    /// Applies this symmetric ordering and returns a lower-triangular CSC
    /// matrix in ordered coordinates.
    #[inline]
    pub fn permute<const MAX_NNZ: usize, T: Copy + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
    ) -> Result<StaticCscMatrix<N, N, MAX_NNZ, T>, CscError> {
        Ok(self
            .permutation_for_pattern(matrix.pattern())?
            .apply(matrix))
    }

    /// Precomputes a reusable sparse-coordinate permutation for a CSC pattern.
    #[inline]
    pub fn permutation_for_pattern<const MAX_NNZ: usize>(
        &self,
        pattern: &StaticCscPattern<N, N, MAX_NNZ>,
    ) -> Result<StaticCscPermutation<N, MAX_NNZ>, CscError> {
        StaticCscPermutation::from_ordering(pattern, *self)
    }

    /// Returns whether this ordering leaves the scalar coordinates unchanged.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.permutation == Self::identity().permutation
    }
}
