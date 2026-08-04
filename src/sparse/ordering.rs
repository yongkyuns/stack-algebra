use crate::Zero;

use super::errors::CscError;
use super::storage::StaticCscMatrix;

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
        let mut adjacency = [[false; N]; N];
        for column in 0..N {
            let start = matrix.column_starts()[column];
            let end = matrix.column_end(column).unwrap_or(matrix.nnz());
            for index in start..end {
                let row = matrix.row_indices()[index];
                if row != column {
                    adjacency[row][column] = true;
                    adjacency[column][row] = true;
                }
            }
        }

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
        permute_matrix(matrix, *self)
    }

    #[inline]
    pub(crate) fn is_identity(&self) -> bool {
        self.permutation == Self::identity().permutation
    }
}

fn permute_matrix<const N: usize, const MAX_NNZ: usize, T: Copy + Zero>(
    matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
    ordering: StaticCscOrdering<N>,
) -> Result<StaticCscMatrix<N, N, MAX_NNZ, T>, CscError> {
    let mut present = [[false; N]; N];
    let mut mapped_values = [[T::zero(); N]; N];
    for column in 0..N {
        let start = matrix.column_starts()[column];
        let end = matrix.column_end(column).unwrap_or(matrix.nnz());
        for index in start..end {
            let row = matrix.row_indices()[index];
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
            if !present[lower_column][lower_row] {
                present[lower_column][lower_row] = true;
                mapped_values[lower_column][lower_row] = matrix.values()[index];
            }
        }
    }

    let mut output = StaticCscMatrix::new();
    let mut position = 0;
    for column in 0..N {
        output.pattern.column_starts[column] = position;
        for row in column..N {
            if present[column][row] {
                if position == MAX_NNZ {
                    return Err(CscError::CapacityExceeded);
                }
                output.pattern.row_indices[position] = row;
                output.values[position] = mapped_values[column][row];
                position += 1;
            }
        }
    }
    output.pattern.nnz = position;
    Ok(output)
}
