use core::hint::black_box;
use core::mem::size_of_val;

use stack_algebra::{
    Matrix, StaticBlockCscMatrix, StaticCscCholesky, StaticCscMatrix, Vector,
};

#[inline(never)]
pub fn baseline() -> usize {
    black_box(0usize)
}

#[inline(never)]
pub fn dense3() -> usize {
    let matrix = Matrix::<3, 3, f32>::from_rows([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 0.25],
        [0.5, 0.25, 2.0],
    ]);
    let rhs = Vector::<3, f32>::from_columns([[1.0, 2.0, 3.0]]);
    let factor = matrix
        .try_partial_piv_lu()
        .expect("dense3 qualification matrix is finite");
    let solution = factor.solve(&rhs);
    let residual = matrix * solution - rhs;
    assert!(residual.norm() < 1.0e-4);

    black_box(&factor);
    black_box(&solution);
    size_of_val(&matrix) + size_of_val(&rhs) + size_of_val(&factor) + size_of_val(&solution)
}

#[inline(never)]
pub fn dense6() -> usize {
    let lhs = Matrix::<6, 6, f32>::from_fn(|row, column| {
        let base = ((row * 7 + column * 3 + 1) % 11) as f32 * 0.125;
        if row == column {
            base + 2.0
        } else {
            base
        }
    });
    let rhs = Matrix::<6, 6, f32>::from_fn(|row, column| {
        ((row * 5 + column * 11 + 2) % 13) as f32 * 0.0625 - 0.25
    });
    let mut product = Matrix::<6, 6, f32>::zeros();
    lhs.mul_into(&rhs, &mut product);
    let mut fused = Matrix::<6, 6, f32>::zeros();
    product.linear_combination_into(0.75, &lhs, 0.25, &mut fused);

    black_box(&fused);
    size_of_val(&lhs) + size_of_val(&rhs) + size_of_val(&product) + size_of_val(&fused)
}

#[inline(never)]
pub fn dense6_f64() -> usize {
    let lhs = Matrix::<6, 6, f64>::from_fn(|row, column| {
        let base = ((row * 7 + column * 3 + 1) % 11) as f64 * 0.125;
        if row == column {
            base + 2.0
        } else {
            base
        }
    });
    let rhs = Matrix::<6, 6, f64>::from_fn(|row, column| {
        ((row * 5 + column * 11 + 2) % 13) as f64 * 0.0625 - 0.25
    });
    let mut product = Matrix::<6, 6, f64>::zeros();
    lhs.mul_into(&rhs, &mut product);
    let mut fused = Matrix::<6, 6, f64>::zeros();
    product.linear_combination_into(0.75, &lhs, 0.25, &mut fused);

    black_box(&fused);
    size_of_val(&lhs) + size_of_val(&rhs) + size_of_val(&product) + size_of_val(&fused)
}

#[inline(never)]
pub fn dense15() -> usize {
    let lhs = Matrix::<15, 15, f32>::from_fn(|row, column| {
        let base = ((row * 17 + column * 7 + 3) % 23) as f32 * 0.03125;
        if row == column {
            base + 1.5
        } else {
            base - 0.25
        }
    });
    let rhs = Matrix::<15, 15, f32>::from_fn(|row, column| {
        ((row * 13 + column * 5 + 1) % 19) as f32 * 0.03125 - 0.25
    });
    let mut product = Matrix::<15, 15, f32>::zeros();
    lhs.mul_into(&rhs, &mut product);
    let mut fused = Matrix::<15, 15, f32>::zeros();
    product.axpy_into(0.5, &lhs, &mut fused);

    black_box(&fused);
    size_of_val(&lhs) + size_of_val(&rhs) + size_of_val(&product) + size_of_val(&fused)
}

#[inline(never)]
pub fn sparse() -> usize {
    type Sparse = StaticCscMatrix<4, 4, 10, f32>;

    let matrix = Sparse::from_pattern(
        &[4.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0],
        &[0, 1, 2, 3, 0, 1, 0, 2, 0, 3],
        &[0, 4, 6, 8, 10],
    )
    .expect("sparse qualification pattern is valid");
    let factor = StaticCscCholesky::<4, 10, f32>::decompose(&matrix)
        .expect("sparse qualification matrix is SPD");
    let rhs = Vector::<4, f32>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
    let solution = factor.solve(&rhs);
    let residual = matrix.matvec(&solution) - rhs;
    assert!(residual.norm() < 1.0e-4);

    black_box(&factor);
    black_box(&solution);
    size_of_val(&matrix) + size_of_val(&factor) + size_of_val(&rhs) + size_of_val(&solution)
}

#[inline(never)]
pub fn block_sparse() -> usize {
    type Blocks = StaticBlockCscMatrix<2, 2, 2, 2, 3, f32>;

    let values = [
        Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]),
        Matrix::from_rows([[0.5, 0.0], [0.0, 0.5]]),
        Matrix::from_rows([[2.0, 0.25], [0.25, 2.5]]),
    ];
    let matrix = Blocks::from_pattern(&values, &[0, 1, 1], &[0, 2, 3])
        .expect("block-sparse qualification pattern is valid");
    let rhs = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    matrix
        .matvec_into(&rhs, &mut output)
        .expect("block-sparse dimensions match");

    black_box(&output);
    size_of_val(&matrix) + size_of_val(&rhs) + size_of_val(&output)
}
