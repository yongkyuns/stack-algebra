use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use stack_algebra::{Matrix, StaticBlockCscLdlt, StaticBlockCscMatrix, StaticCscLdlt};

#[cfg(feature = "eigen-compare")]
unsafe extern "C" {
    fn sa_eigen_ldlt_solve_f64(
        input: *const f64,
        rhs: *const f64,
        dimension: usize,
        columns: usize,
        output: *mut f64,
    ) -> i32;
}

type Blocks = StaticBlockCscMatrix<2, 2, 2, 2, 4, f64>;
type NativeLdlt = StaticBlockCscLdlt<2, 2, 2, 2, 4, f64>;

fn block_matrix() -> Blocks {
    Blocks::from_pattern(
        &[
            Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]),
            Matrix::from_rows([[1.0, 0.0], [0.0, 0.5]]),
            Matrix::from_rows([[1.0, 0.0], [0.0, 0.5]]),
            Matrix::from_rows([[3.0, 0.0], [0.0, 2.0]]),
        ],
        &[0, 1, 0, 1],
        &[0, 2, 4],
    )
    .unwrap()
}

fn rhs() -> Matrix<4, 1, f64> {
    Matrix::from_columns([[1.0, 2.0, 3.0, 4.0]])
}

type CrossBlockMatrix = StaticBlockCscMatrix<1, 1, 2, 2, 3, f64>;

fn cross_block_matrix() -> CrossBlockMatrix {
    CrossBlockMatrix::from_pattern(
        &[
            Matrix::from_rows([[0.0]]),
            Matrix::from_rows([[1.0]]),
            Matrix::from_rows([[4.0]]),
        ],
        &[0, 1, 1],
        &[0, 2, 3],
    )
    .unwrap()
}

fn cross_dense_matrix() -> Matrix<2, 2, f64> {
    Matrix::from_rows([[0.0, 1.0], [1.0, 4.0]])
}

fn cross_faer_matrix() -> Mat<f64> {
    Mat::from_fn(2, 2, |row, column| {
        if row == column {
            if row == 0 {
                4.0
            } else {
                -3.0
            }
        } else {
            1.0
        }
    })
}

fn criterion_benchmark(criterion: &mut Criterion) {
    let matrix = block_matrix();
    let right_hand_side = rhs();
    let pattern =
        stack_algebra::StaticBlockCscCholeskyPattern::<2, 2, 2, 2, 4>::analyze(&matrix).unwrap();
    let mut native = pattern.factor(&matrix).unwrap();
    let mut scalar = matrix.cholesky::<4, 16, 16>().unwrap();
    let mut native_output = Matrix::<4, 1, f64>::zeros();
    let mut scalar_output = Matrix::<4, 1, f64>::zeros();

    let mut group = criterion.benchmark_group("block-sparse/cholesky/f64/2x2-blocks");
    group.bench_function("native-factor", |bench| {
        bench.iter(|| {
            black_box(pattern.factor(black_box(&matrix)).unwrap());
        });
    });
    group.bench_function("scalar-expanded-factor", |bench| {
        bench.iter(|| {
            black_box(black_box(&matrix).cholesky::<4, 16, 16>().unwrap());
        });
    });
    group.bench_function("native-refactor", |bench| {
        bench.iter(|| {
            native.recompute(black_box(&matrix)).unwrap();
            black_box(&native);
        });
    });
    group.bench_function("scalar-expanded-refactor", |bench| {
        bench.iter(|| {
            scalar = black_box(&matrix).cholesky::<4, 16, 16>().unwrap();
            black_box(&scalar);
        });
    });
    group.bench_function("native-solve", |bench| {
        bench.iter(|| {
            native
                .try_solve_into(black_box(&right_hand_side), black_box(&mut native_output))
                .unwrap();
            black_box(&native_output);
        });
    });
    group.bench_function("scalar-expanded-solve", |bench| {
        bench.iter(|| {
            scalar.solve_into(black_box(&right_hand_side), black_box(&mut scalar_output));
            black_box(&scalar_output);
        });
    });
    group.finish();

    let cross_blocks = cross_block_matrix();
    let cross_dense = cross_dense_matrix();
    let cross_rhs = Matrix::<2, 1, f64>::from_columns([[2.0, -3.0]]);
    let mut dense_factor = cross_blocks.try_dense_ldlt::<2>().unwrap();
    let mut dense_output = Matrix::<2, 1, f64>::zeros();
    let faer_input = cross_faer_matrix();
    let faer_factor = faer_input.ldlt(Side::Lower).unwrap();
    let faer_rhs = Mat::from_fn(2, 1, |row, _| cross_rhs[(row, 0)]);
    let mut faer_output = Mat::from_fn(2, 1, |row, column| faer_rhs[(row, column)]);

    let mut group = criterion.benchmark_group("block-sparse/ldlt/cross-block/f64/2x2");
    group.bench_function("stack-dense-factor", |bench| {
        bench.iter(|| {
            let factor = black_box(&cross_dense).ldlt().unwrap();
            black_box(factor);
        });
    });
    group.bench_function("stack-dense-fallback-factor", |bench| {
        bench.iter(|| {
            dense_factor = black_box(&cross_blocks).try_dense_ldlt::<2>().unwrap();
            black_box(&dense_factor);
        });
    });
    group.bench_function("faer-stable-factor", |bench| {
        bench.iter(|| {
            let factor = black_box(&faer_input).ldlt(Side::Lower).unwrap();
            black_box(factor);
        });
    });
    group.bench_function("stack-dense-fallback-solve", |bench| {
        bench.iter(|| {
            dense_factor.solve_into(black_box(&cross_rhs), black_box(&mut dense_output));
            black_box(&dense_output);
        });
    });
    group.bench_function("faer-stable-solve", |bench| {
        bench.iter(|| {
            faer_output.copy_from(&faer_rhs);
            faer_factor.solve_in_place(black_box(&mut faer_output));
            black_box(&faer_output);
        });
    });
    #[cfg(feature = "eigen-compare")]
    group.bench_function("eigen-solve", |bench| {
        let mut output = [0.0_f64; 2];
        bench.iter(|| {
            let status = unsafe {
                sa_eigen_ldlt_solve_f64(
                    cross_dense.as_slice().as_ptr(),
                    cross_rhs.as_slice().as_ptr(),
                    2,
                    1,
                    output.as_mut_ptr(),
                )
            };
            assert_eq!(status, 1);
            black_box(output);
        });
    });
    group.finish();

    let scalar_matrix = matrix.to_scalar_csc::<4, 4, 16>().unwrap();
    let ldlt_pattern =
        stack_algebra::StaticBlockCscLdltPattern::<2, 2, 2, 2, 4>::analyze(&matrix).unwrap();
    let mut native_ldlt: NativeLdlt = NativeLdlt::decompose(&matrix).unwrap();
    let mut scalar_ldlt = StaticCscLdlt::<4, 16, f64>::decompose(&scalar_matrix).unwrap();
    let mut native_ldlt_output = Matrix::<4, 1, f64>::zeros();
    let mut scalar_ldlt_output = Matrix::<4, 1, f64>::zeros();
    let mut group = criterion.benchmark_group("block-sparse/ldlt/f64/2x2-blocks");
    group.bench_function("native-factor", |bench| {
        bench.iter(|| {
            black_box(ldlt_pattern.factor_ldlt(black_box(&matrix)).unwrap());
        });
    });
    group.bench_function("scalar-expanded-factor", |bench| {
        bench.iter(|| {
            black_box(StaticCscLdlt::<4, 16, f64>::decompose(black_box(&scalar_matrix)).unwrap());
        });
    });
    group.bench_function("native-diagonal-pivot-factor", |bench| {
        bench.iter(|| {
            black_box(
                NativeLdlt::decompose_with_diagonal_pivoting(black_box(&matrix), 1e-12).unwrap(),
            );
        });
    });
    group.bench_function("scalar-expanded-diagonal-pivot-factor", |bench| {
        bench.iter(|| {
            black_box(
                StaticCscLdlt::<4, 16, f64>::decompose_with_diagonal_pivoting(
                    black_box(&scalar_matrix),
                    1e-12,
                )
                .unwrap(),
            );
        });
    });
    group.bench_function("native-refactor", |bench| {
        bench.iter(|| {
            native_ldlt.recompute(black_box(&matrix)).unwrap();
            black_box(&native_ldlt);
        });
    });
    group.bench_function("scalar-expanded-refactor", |bench| {
        bench.iter(|| {
            scalar_ldlt.recompute(black_box(&scalar_matrix)).unwrap();
            black_box(&scalar_ldlt);
        });
    });
    group.bench_function("native-solve", |bench| {
        bench.iter(|| {
            native_ldlt
                .try_solve_into(
                    black_box(&right_hand_side),
                    black_box(&mut native_ldlt_output),
                )
                .unwrap();
            black_box(&native_ldlt_output);
        });
    });
    group.bench_function("scalar-expanded-solve", |bench| {
        bench.iter(|| {
            scalar_ldlt.solve_into(
                black_box(&right_hand_side),
                black_box(&mut scalar_ldlt_output),
            );
            black_box(&scalar_ldlt_output);
        });
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
