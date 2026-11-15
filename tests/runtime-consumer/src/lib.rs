#![no_std]
#![forbid(unsafe_code)]
//! Separate consumer to instantiate runtime operations without dev dependencies.

use stack_algebra::runtime::{matmul_into, transpose_into, Error, MatrixMut, MatrixRef};

/// Multiplies caller-owned f32 buffers with runtime dimensions.
pub fn multiply(
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) -> Result<(), Error> {
    matmul_into(
        MatrixRef::from_column_major(lhs, rows, inner)?,
        MatrixRef::from_column_major(rhs, inner, columns)?,
        &mut MatrixMut::from_column_major(output, rows, columns)?,
    )
}

/// Instantiates f64 copy/transpose operations without a temporary or allocator.
pub fn transpose(
    input: &[f64],
    output: &mut [f64],
    rows: usize,
    columns: usize,
) -> Result<(), Error> {
    transpose_into(
        MatrixRef::from_column_major(input, rows, columns)?,
        &mut MatrixMut::from_column_major(output, columns, rows)?,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn runtime_consumer_executes_both_scalar_paths() {
        let mut result = [0.0_f32; 2];
        super::multiply(
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            &[2.0, -1.0, 0.5],
            &mut result,
            2,
            3,
            1,
        )
        .unwrap();
        assert_eq!(result, [1.5, 6.0]);
        let mut transposed = [0.0_f64; 6];
        super::transpose(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &mut transposed, 2, 3).unwrap();
        assert_eq!(transposed, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
