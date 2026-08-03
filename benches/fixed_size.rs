use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::{Accum, Mat, Par};
use stack_algebra::Matrix;

const BATCH_SIZE: usize = 64;

fn stack_matrix<const D: usize>() -> Matrix<D, D, f64> {
    Matrix::from_fn(|row, column| (row * D + column + 1) as f64 / 17.0)
}

fn faer_matrix<const D: usize>() -> Mat<f64> {
    Mat::from_fn(D, D, |row, column| (row * D + column + 1) as f64 / 17.0)
}

fn stack_matrix_f32<const D: usize>() -> Matrix<D, D, f32> {
    Matrix::from_fn(|row, column| (row * D + column + 1) as f32 / 17.0)
}

fn faer_matrix_f32<const D: usize>() -> Mat<f32> {
    Mat::from_fn(D, D, |row, column| (row * D + column + 1) as f32 / 17.0)
}

fn bench_stack_algebra<const D: usize>(criterion: &mut Criterion) {
    let lhs = stack_matrix::<D>();
    let rhs: Matrix<D, D, f64> =
        Matrix::from_fn(|row, column| (row + 2 * column + 3) as f64 / 11.0);
    let mut output = Matrix::<D, D, f64>::zeros();
    let mut group = criterion.benchmark_group("matmul/f64");
    group.bench_with_input(BenchmarkId::new("stack-algebra", D), &D, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                black_box(&lhs).mul_into(black_box(&rhs), black_box(&mut output));
            }
            black_box(&output);
        });
    });
    group.finish();
}

fn bench_faer<const D: usize>(criterion: &mut Criterion) {
    let lhs = faer_matrix::<D>();
    let rhs = Mat::from_fn(D, D, |row, column| (row + 2 * column + 3) as f64 / 11.0);
    let mut output = Mat::<f64>::zeros(D, D);
    let mut group = criterion.benchmark_group("matmul/f64");
    group.bench_with_input(BenchmarkId::new("faer-dynamic", D), &D, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                faer::linalg::matmul::matmul(
                    black_box(&mut output),
                    Accum::Replace,
                    black_box(&lhs),
                    black_box(&rhs),
                    1.0,
                    Par::Seq,
                );
            }
            black_box(&output);
        });
    });
    group.finish();
}

fn bench_stack_algebra_f32<const D: usize>(criterion: &mut Criterion) {
    let lhs = stack_matrix_f32::<D>();
    let rhs: Matrix<D, D, f32> =
        Matrix::from_fn(|row, column| (row + 2 * column + 3) as f32 / 11.0);
    let mut output = Matrix::<D, D, f32>::zeros();
    let mut group = criterion.benchmark_group("matmul/f32");
    group.bench_with_input(BenchmarkId::new("stack-algebra", D), &D, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                black_box(&lhs).mul_into(black_box(&rhs), black_box(&mut output));
            }
            black_box(&output);
        });
    });
    group.finish();
}

fn bench_faer_f32<const D: usize>(criterion: &mut Criterion) {
    let lhs = faer_matrix_f32::<D>();
    let rhs = Mat::from_fn(D, D, |row, column| (row + 2 * column + 3) as f32 / 11.0);
    let mut output = Mat::<f32>::zeros(D, D);
    let mut group = criterion.benchmark_group("matmul/f32");
    group.bench_with_input(BenchmarkId::new("faer-dynamic", D), &D, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                faer::linalg::matmul::matmul(
                    black_box(&mut output),
                    Accum::Replace,
                    black_box(&lhs),
                    black_box(&rhs),
                    1.0,
                    Par::Seq,
                );
            }
            black_box(&output);
        });
    });
    group.finish();
}

fn bench_all(criterion: &mut Criterion) {
    bench_stack_algebra::<2>(criterion);
    bench_stack_algebra::<3>(criterion);
    bench_stack_algebra::<4>(criterion);
    bench_stack_algebra::<6>(criterion);
    bench_stack_algebra::<9>(criterion);
    bench_stack_algebra::<15>(criterion);
    bench_faer::<2>(criterion);
    bench_faer::<3>(criterion);
    bench_faer::<4>(criterion);
    bench_faer::<6>(criterion);
    bench_faer::<9>(criterion);
    bench_faer::<15>(criterion);
    bench_stack_algebra_f32::<2>(criterion);
    bench_stack_algebra_f32::<3>(criterion);
    bench_stack_algebra_f32::<4>(criterion);
    bench_stack_algebra_f32::<6>(criterion);
    bench_stack_algebra_f32::<9>(criterion);
    bench_stack_algebra_f32::<15>(criterion);
    bench_faer_f32::<2>(criterion);
    bench_faer_f32::<3>(criterion);
    bench_faer_f32::<4>(criterion);
    bench_faer_f32::<6>(criterion);
    bench_faer_f32::<9>(criterion);
    bench_faer_f32::<15>(criterion);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
