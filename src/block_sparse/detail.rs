//! Shared private helpers for block-sparse factorization.

use crate::StaticCscPattern;

#[inline]
pub(super) fn find_block_index<const ROWS: usize, const COLS: usize, const MAX_NNZ: usize>(
    pattern: &StaticCscPattern<ROWS, COLS, MAX_NNZ>,
    row: usize,
    column: usize,
) -> Option<usize> {
    let start = *pattern.column_starts().get(column)? as usize;
    let end = pattern.column_end(column)?;
    pattern.row_indices()[start..end]
        .iter()
        .position(|&candidate| candidate as usize == row)
        .map(|offset| start + offset)
}
