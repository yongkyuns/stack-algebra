//! Focused benchmarks for explicit fused updates and mapped-view kernel reuse.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stack_algebra::{matmul_view_into, matvec_view_into, Map, Matrix};

const BATCH: usize = 8;

fn dense<const R: usize, const C: usize>() -> Matrix<R, C, f32> {
    Matrix::from_fn(|row, column| (1 + row * C + 2 * column) as f32 / 17.0)
}

fn vector<const D: usize>() -> Matrix<D, 1, f32> {
    Matrix::from_fn(|row, _| (row + 3) as f32 / 11.0)
}

fn fused_updates<const D: usize>(criterion: &mut Criterion) {
    let x = dense::<D, D>();
    let y = Matrix::<D, D, f32>::from_fn(|row, column| {
        (2 + 3 * row + column) as f32 / 19.0
    });
    let alpha = 1.25_f32;
    let beta = -0.75_f32;
    let mut output = Matrix::<D, D, f32>::zeros();

    let mut group = criterion.benchmark_group("fused/axpy/f32");
    group.bench_function(BenchmarkId::new("expression", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                output = black_box(black_box(x) * black_box(alpha) + black_box(y));
            }
            black_box(&output);
        })
    });
    group.bench_function(BenchmarkId::new("axpy-into", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                black_box(&x).axpy_into(black_box(alpha), black_box(&y), black_box(&mut output));
            }
        })
    });
    group.finish();

    let mut group = criterion.benchmark_group("fused/linear-combination/f32");
    group.bench_function(BenchmarkId::new("expression", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                output = black_box(
                    black_box(x) * black_box(alpha) + black_box(y) * black_box(beta),
                );
            }
            black_box(&output);
        })
    });
    group.bench_function(BenchmarkId::new("linear-combination-into", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                black_box(&x).linear_combination_into(
                    black_box(alpha),
                    black_box(&y),
                    black_box(beta),
                    black_box(&mut output),
                );
            }
        })
    });
    group.finish();
}

fn mapped_views<const D: usize>(criterion: &mut Criterion) {
    let lhs_storage = dense::<D, D>();
    let rhs_storage = Matrix::<D, D, f32>::from_fn(|row, column| {
        (3 + row + 2 * column) as f32 / 23.0
    });
    let lhs = Map::<D, D, f32>::from_slice(lhs_storage.as_slice()).unwrap();
    let rhs = Map::<D, D, f32>::from_slice(rhs_storage.as_slice()).unwrap();
    let vector = vector::<D>();
    let mut matrix_output = Matrix::<D, D, f32>::zeros();
    let mut vector_output = Matrix::<D, 1, f32>::zeros();

    let mut group = criterion.benchmark_group("views/matmul-map/f32");
    group.bench_function(BenchmarkId::new("generic-view", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                matmul_view_into(
                    black_box(&lhs),
                    black_box(&rhs),
                    black_box(&mut matrix_output),
                );
            }
        })
    });
    group.bench_function(BenchmarkId::new("optimized-map", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                black_box(&lhs).mul_into(black_box(&rhs), black_box(&mut matrix_output));
            }
        })
    });
    group.finish();

    let mut group = criterion.benchmark_group("views/matvec-map/f32");
    group.bench_function(BenchmarkId::new("generic-view", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                matvec_view_into(
                    black_box(&lhs),
                    black_box(&vector),
                    black_box(&mut vector_output),
                );
            }
        })
    });
    group.bench_function(BenchmarkId::new("optimized-map", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                black_box(&lhs).matvec_into(black_box(&vector), black_box(&mut vector_output));
            }
        })
    });
    group.finish();
}

fn benchmark(criterion: &mut Criterion) {
    fused_updates::<6>(criterion);
    fused_updates::<15>(criterion);
    fused_updates::<32>(criterion);
    mapped_views::<6>(criterion);
    mapped_views::<15>(criterion);
    mapped_views::<32>(criterion);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
