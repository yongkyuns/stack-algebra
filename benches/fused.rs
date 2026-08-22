//! Focused benchmarks for explicit fused updates, GEMM accumulation probes,
//! and mapped-view kernel reuse.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stack_algebra::{matmul_view_into, matvec_view_into, Map, Matrix, MatrixRead, StridedMap};

const BATCH: usize = 8;
const PAD: usize = 3;

fn dense<const R: usize, const C: usize>() -> Matrix<R, C, f32> {
    Matrix::from_fn(|row, column| (1 + row * C + 2 * column) as f32 / 17.0)
}

fn vector<const D: usize>() -> Matrix<D, 1, f32> {
    Matrix::from_fn(|row, _| (row + 3) as f32 / 11.0)
}

fn prototype_mul_add_into<const D: usize>(
    lhs: &Matrix<D, D, f32>,
    rhs: &Matrix<D, D, f32>,
    addend: &Matrix<D, D, f32>,
    output: &mut Matrix<D, D, f32>,
) {
    for column in 0..D {
        for row in 0..D {
            let mut value = addend[(row, column)];
            for inner in 0..D {
                value += lhs[(row, inner)] * rhs[(inner, column)];
            }
            output[(row, column)] = value;
        }
    }
}

fn fused_updates<const D: usize>(criterion: &mut Criterion) {
    let x = dense::<D, D>();
    let y = Matrix::<D, D, f32>::from_fn(|row, column| (2 + 3 * row + column) as f32 / 19.0);
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
                output =
                    black_box(black_box(x) * black_box(alpha) + black_box(y) * black_box(beta));
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

fn gemm_accumulate_probe<const D: usize>(criterion: &mut Criterion) {
    let lhs = dense::<D, D>();
    let rhs = Matrix::<D, D, f32>::from_fn(|row, column| (3 + row + 2 * column) as f32 / 23.0);
    let addend = Matrix::<D, D, f32>::from_fn(|row, column| (5 + 2 * row + column) as f32 / 29.0);
    let mut output = Matrix::<D, D, f32>::zeros();

    let mut group = criterion.benchmark_group("decision/gemm-accumulate/f32");
    group.bench_function(BenchmarkId::new("expression", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                output = black_box(black_box(lhs) * black_box(rhs) + black_box(addend));
            }
            black_box(&output);
        })
    });
    group.bench_function(BenchmarkId::new("mul-then-axpy", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                black_box(&lhs).mul_into(black_box(&rhs), black_box(&mut output));
                black_box(&addend).axpy_in_place(black_box(1.0), black_box(&mut output));
            }
        })
    });
    group.bench_function(BenchmarkId::new("prototype-fused", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                prototype_mul_add_into(
                    black_box(&lhs),
                    black_box(&rhs),
                    black_box(&addend),
                    black_box(&mut output),
                );
            }
        })
    });
    group.finish();
}

fn mapped_views<const D: usize>(criterion: &mut Criterion) {
    let lhs_storage = dense::<D, D>();
    let rhs_storage =
        Matrix::<D, D, f32>::from_fn(|row, column| (3 + row + 2 * column) as f32 / 23.0);
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

fn padded_mapped_probe<const D: usize>(criterion: &mut Criterion) {
    let outer_stride = D + PAD;
    let storage_len = outer_stride * D;
    let mut lhs_storage = vec![0.0_f32; storage_len];
    let mut rhs_storage = vec![0.0_f32; storage_len];
    for column in 0..D {
        for row in 0..D {
            lhs_storage[row + column * outer_stride] = (1 + row * D + column) as f32 / 17.0;
            rhs_storage[row + column * outer_stride] = (3 + row + 2 * column) as f32 / 23.0;
        }
    }
    let lhs = StridedMap::<D, D, f32>::from_slice(&lhs_storage, 1, outer_stride).unwrap();
    let rhs = StridedMap::<D, D, f32>::from_slice(&rhs_storage, 1, outer_stride).unwrap();
    let mut output = Matrix::<D, D, f32>::zeros();

    let mut group = criterion.benchmark_group("decision/padded-leading-dimension/f32");
    group.bench_function(BenchmarkId::new("generic-zero-copy", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                matmul_view_into(black_box(&lhs), black_box(&rhs), black_box(&mut output));
            }
        })
    });
    group.bench_function(BenchmarkId::new("copy-then-optimized", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                let owned_lhs = Matrix::<D, D, f32>::from_view(black_box(&lhs));
                let owned_rhs = Matrix::<D, D, f32>::from_view(black_box(&rhs));
                black_box(&owned_lhs).mul_into(black_box(&owned_rhs), black_box(&mut output));
            }
        })
    });
    group.bench_function(BenchmarkId::new("copy-cost-only", D), |bench| {
        bench.iter(|| {
            for _ in 0..BATCH {
                let owned_lhs = Matrix::<D, D, f32>::from_fn(|row, column| {
                    *black_box(&lhs).get_in_bounds(row, column)
                });
                let owned_rhs = Matrix::<D, D, f32>::from_fn(|row, column| {
                    *black_box(&rhs).get_in_bounds(row, column)
                });
                black_box((owned_lhs, owned_rhs));
            }
        })
    });
    group.finish();
}

fn benchmark(criterion: &mut Criterion) {
    fused_updates::<6>(criterion);
    fused_updates::<15>(criterion);
    fused_updates::<32>(criterion);
    gemm_accumulate_probe::<6>(criterion);
    gemm_accumulate_probe::<15>(criterion);
    gemm_accumulate_probe::<32>(criterion);
    mapped_views::<6>(criterion);
    mapped_views::<15>(criterion);
    mapped_views::<32>(criterion);
    padded_mapped_probe::<6>(criterion);
    padded_mapped_probe::<15>(criterion);
    padded_mapped_probe::<32>(criterion);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
