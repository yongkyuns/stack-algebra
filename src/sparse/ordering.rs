use core::{mem::MaybeUninit, ptr};

use crate::Zero;

use super::errors::CscError;
use super::storage::{StaticCscMatrix, StaticCscPattern};

/// Sorts two parallel arrays by the values in `keys` without allocating a
/// temporary array. This is used while building a fixed-capacity permutation,
/// where a temporary pair array would duplicate the sparse capacity on the
/// caller's stack.
#[inline]
fn sort_parallel_by_key(keys: &mut [u32], values: &mut [u32]) {
    debug_assert_eq!(keys.len(), values.len());
    if keys.len() < 2 {
        return;
    }

    let pivot = keys[keys.len() / 2];
    let mut left = 0isize;
    let mut right = keys.len() as isize - 1;
    while left <= right {
        while keys[left as usize] < pivot {
            left += 1;
        }
        while keys[right as usize] > pivot {
            right -= 1;
        }
        if left > right {
            break;
        }
        keys.swap(left as usize, right as usize);
        values.swap(left as usize, right as usize);
        left += 1;
        right -= 1;
    }

    if right >= 1 {
        sort_parallel_by_key(&mut keys[..=right as usize], &mut values[..=right as usize]);
    }
    if left < keys.len() as isize {
        sort_parallel_by_key(&mut keys[left as usize..], &mut values[left as usize..]);
    }
}

/// A reusable sparse-coordinate permutation for a validated CSC pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticCscPermutation<const N: usize, const MAX_NNZ: usize> {
    pattern: StaticCscPattern<N, N, MAX_NNZ>,
    source_indices: [u32; MAX_NNZ],
    // One past the largest active source index, or zero for an empty map.
    // The ordered entry count is already stored in `pattern.nnz`.
    required_source_len: usize,
}

impl<const N: usize, const MAX_NNZ: usize> StaticCscPermutation<N, MAX_NNZ> {
    /// Creates an empty reusable permutation workspace.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pattern: StaticCscPattern::new(),
            source_indices: [0; MAX_NNZ],
            required_source_len: 0,
        }
    }

    /// Initializes an empty permutation directly in caller-owned storage.
    pub fn new_into(output: &mut MaybeUninit<Self>) {
        // SAFETY: all-zero values are valid for the pattern metadata, source indices,
        // and required source length.
        unsafe {
            ptr::write_bytes(
                output.as_mut_ptr().cast::<u8>(),
                0,
                core::mem::size_of::<Self>(),
            );
        }
    }

    /// Builds a reusable map from an original lower CSC pattern.
    #[inline]
    pub fn from_ordering(
        matrix_pattern: &StaticCscPattern<N, N, MAX_NNZ>,
        ordering: StaticCscOrdering<N>,
    ) -> Result<Self, CscError> {
        let mut output = Self::new();
        output.from_ordering_into(matrix_pattern, ordering)?;
        Ok(output)
    }

    /// Builds a reusable map directly into an existing permutation object.
    #[inline]
    pub fn from_ordering_into(
        &mut self,
        matrix_pattern: &StaticCscPattern<N, N, MAX_NNZ>,
        ordering: StaticCscOrdering<N>,
    ) -> Result<(), CscError> {
        let mut column_counts = [0u32; N];
        let mut entry_count = 0;
        let mut required_source_len = 0;
        for column in 0..N {
            let start = matrix_pattern.column_starts()[column] as usize;
            let end = matrix_pattern
                .column_end(column)
                .unwrap_or(matrix_pattern.nnz());
            for source_index in start..end {
                let row = matrix_pattern.row_indices()[source_index] as usize;
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
                let lower_column =
                    u32::try_from(lower_column).map_err(|_| CscError::InvalidColumnPointers)?;
                u32::try_from(lower_row).map_err(|_| CscError::InvalidRowIndices)?;
                // Validate every fallible conversion before modifying the workspace.
                u32::try_from(source_index).map_err(|_| CscError::CapacityExceeded {
                    required: source_index.saturating_add(1),
                    capacity: u32::MAX as usize,
                })?;
                // Source indices are visited in increasing CSC order. The index
                // is below the active slice length, so adding one cannot overflow.
                required_source_len = source_index + 1;
                column_counts[lower_column as usize] += 1;
                entry_count += 1;
            }
        }
        // Clear the existing parallel arrays in place instead of materializing capacity-sized
        // array values during permutation setup.
        unsafe {
            ptr::write_bytes(
                (&mut self.pattern as *mut StaticCscPattern<N, N, MAX_NNZ>).cast::<u8>(),
                0,
                core::mem::size_of::<StaticCscPattern<N, N, MAX_NNZ>>(),
            );
            ptr::write_bytes(
                self.source_indices.as_mut_ptr().cast::<u8>(),
                0,
                core::mem::size_of_val(&self.source_indices),
            );
        }
        self.required_source_len = required_source_len;

        let mut column_starts = [0u32; N];
        for column in 1..N {
            column_starts[column] = column_starts[column - 1] + column_counts[column - 1];
        }
        self.pattern.column_starts = column_starts;
        self.pattern.nnz = entry_count;

        let mut cursors = column_starts;
        for column in 0..N {
            let start = matrix_pattern.column_starts()[column] as usize;
            let end = matrix_pattern
                .column_end(column)
                .unwrap_or(matrix_pattern.nnz());
            for source_index in start..end {
                let row = matrix_pattern.row_indices()[source_index] as usize;
                if row < column {
                    continue;
                }
                let ordered_row = ordering.inverse()[row];
                let ordered_column = ordering.inverse()[column];
                let lower_column = ordered_row.min(ordered_column);
                let lower_row = ordered_row.max(ordered_column);
                let target = cursors[lower_column] as usize;
                self.pattern.row_indices[target] = lower_row as u32;
                // The first pass checked this conversion before any mutation.
                self.source_indices[target] = source_index as u32;
                cursors[lower_column] += 1;
            }
        }
        for column in 0..N {
            let start = column_starts[column] as usize;
            let end = if column + 1 < N {
                column_starts[column + 1] as usize
            } else {
                entry_count
            };
            sort_parallel_by_key(
                &mut self.pattern.row_indices[start..end],
                &mut self.source_indices[start..end],
            );
        }
        Ok(())
    }

    /// Returns the ordered pattern.
    #[inline]
    pub const fn pattern(&self) -> StaticCscPattern<N, N, MAX_NNZ> {
        self.pattern
    }

    /// Borrows the ordered sparse pattern without copying its fixed-capacity storage.
    #[inline]
    pub const fn pattern_ref(&self) -> &StaticCscPattern<N, N, MAX_NNZ> {
        &self.pattern
    }

    /// Applies the precomputed coordinate map to a matrix's numeric values.
    ///
    /// The input must retain the CSC source pattern used to build this map,
    /// including any stored upper-triangle entries. Rebuild the map when that
    /// pattern changes; matching dimensions or entry counts are not sufficient.
    ///
    /// # Panics
    ///
    /// Panics if a cached source index exceeds the input's active values.
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
    ///
    /// Replaces the destination's pattern and active values, accepting an empty
    /// destination or any previous pattern of the same capacity. Inactive value
    /// storage is left untouched. No temporary capacity-sized matrix is created.
    ///
    /// As with [`Self::apply`], the input must retain the source pattern used to
    /// build this map. Cached-offset bounds are checked, but same-length source
    /// coordinate changes are not detected. Rebuild the map for a changed input
    /// pattern rather than reusing its old numeric offsets.
    ///
    /// # Panics
    ///
    /// Panics before changing the destination if any cached source index exceeds
    /// the input's active values.
    #[inline]
    pub fn apply_into<T: Copy + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
        output: &mut StaticCscMatrix<N, N, MAX_NNZ, T>,
    ) {
        let source_values = matrix.values();
        let source_indices = &self.source_indices[..self.pattern.nnz()];
        assert!(
            self.required_source_len <= source_values.len(),
            "sparse permutation source index out of bounds"
        );

        // Construction caches the exact maximum active source offset plus one;
        // sorting preserves it. This one comparison is equivalent to checking
        // every cached index, including for empty maps and full source storage.
        // All reads are in bounds before changing either destination field.
        // Install the ordered pattern before borrowing its active value slice.
        output.pattern = self.pattern;
        // The source-index slice already checked this exact active length
        // against MAX_NNZ. Do not reload the copied destination count here.
        let output_values = &mut output.values[..source_indices.len()];
        for (value, &source_index) in output_values.iter_mut().zip(source_indices) {
            *value = source_values[source_index as usize];
        }
    }
}

impl<const N: usize, const MAX_NNZ: usize> Default for StaticCscPermutation<N, MAX_NNZ> {
    fn default() -> Self {
        Self::new()
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
            let start = pattern.column_starts()[column] as usize;
            let end = pattern.column_end(column).unwrap_or(pattern.nnz());
            for index in start..end {
                let row = pattern.row_indices()[index] as usize;
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

#[cfg(test)]
mod permutation_extent_tests {
    use super::*;

    fn check_patterns<const N: usize, const CAP: usize>(permutations: &[[usize; N]]) {
        let mut reused = StaticCscPermutation::<N, CAP>::new();
        for permutation in permutations {
            let order = StaticCscOrdering::from_permutation(permutation).unwrap();
            for mask in 0..(1usize << (N * N)) {
                let mut input = StaticCscMatrix::<N, N, CAP, i32>::new();
                for column in 0..N {
                    for row in 0..N {
                        if mask & (1 << (column * N + row)) != 0 {
                            input.insert(row, column, 1).unwrap();
                        }
                    }
                }
                let map = order.permutation_for_pattern(input.pattern()).unwrap();
                reused.from_ordering_into(input.pattern(), order).unwrap();
                assert_eq!(reused, map);
                let mut expected_len = 0;
                for column in 0..N {
                    for row in column..N {
                        if let Some(index) = input.pattern().entry_index(row, column) {
                            expected_len = expected_len.max(index + 1);
                        }
                    }
                }
                assert_eq!(map.required_source_len, expected_len);
                for len in 0..=CAP {
                    let old_scan = map.source_indices[..map.pattern.nnz()]
                        .iter()
                        .all(|&index| (index as usize) < len);
                    assert_eq!(map.required_source_len <= len, old_scan);
                }
            }
        }
    }

    #[test]
    fn exhaustive_2x2_source_extent() {
        check_patterns::<2, 4>(&[[0, 1], [1, 0]]);
    }

    #[test]
    fn exhaustive_3x3_source_extent() {
        check_patterns::<3, 9>(&[
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ]);
    }

    #[test]
    fn trailing_upper_entries_and_empty_rebuilds_use_exact_extent() {
        let input = StaticCscMatrix::<3, 3, 9, i32>::from_pattern(&[7, 99], &[0, 0], &[0, 1, 2, 2])
            .unwrap();
        let mut map = StaticCscOrdering::identity()
            .permutation_for_pattern(input.pattern())
            .unwrap();
        assert_eq!(map.required_source_len, 1);
        assert_eq!(map.pattern.nnz(), 1);
        let empty = StaticCscPattern::new();
        map.from_ordering_into(&empty, StaticCscOrdering::identity())
            .unwrap();
        assert_eq!(map.required_source_len, 0);
        assert_eq!(map, StaticCscPermutation::new());
    }

    #[test]
    fn apply_preserves_inactive_values_and_installs_complete_pattern() {
        let input =
            StaticCscMatrix::<2, 2, 4, i32>::from_pattern(&[4, 3], &[0, 1], &[0, 1, 2]).unwrap();
        let map = StaticCscOrdering::identity()
            .permutation_for_pattern(input.pattern())
            .unwrap();
        let mut output =
            StaticCscMatrix::from_pattern(&[100, 101, 102, 103], &[0, 1, 0, 1], &[0, 2, 4])
                .unwrap();
        map.apply_into(&input, &mut output);
        assert_eq!(output.values, [4, 3, 102, 103]);
        assert_eq!(output.pattern, map.pattern);
        StaticCscPermutation::new().apply_into(&StaticCscMatrix::new(), &mut output);
        assert_eq!(output.values, [4, 3, 102, 103]);
        assert_eq!(output.pattern, StaticCscPattern::new());
    }

    #[test]
    fn workspace_footprint_matches_previous_fields() {
        #[allow(dead_code)]
        struct Previous<const N: usize, const CAP: usize> {
            pattern: StaticCscPattern<N, N, CAP>,
            source_indices: [u32; CAP],
            nnz: usize,
        }
        fn check<const N: usize, const CAP: usize>() {
            assert_eq!(
                core::mem::size_of::<StaticCscPermutation<N, CAP>>(),
                core::mem::size_of::<Previous<N, CAP>>()
            );
            assert_eq!(
                core::mem::align_of::<StaticCscPermutation<N, CAP>>(),
                core::mem::align_of::<Previous<N, CAP>>()
            );
        }
        check::<0, 0>();
        check::<0, 4>();
        check::<2, 3>();
        check::<3, 9>();
        check::<128, 2560>();
    }
}
