use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::prelude::Solve;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::{Mat, Side};
use faer_traits::ComplexField;
use num_traits::FromPrimitive;
use stack_algebra::{
    Matrix, Real, StaticCscCholeskyPattern, StaticCscLdltPattern, StaticCscMatrix,
    StaticCscOrdering,
};

const MAX_NNZ: usize = 128;
const MAX_STAR_NNZ: usize = 1024;
const BATCH_SIZE: usize = 64;

fn cast<T: FromPrimitive>(value: f64) -> T {
    T::from_f64(value).unwrap()
}

fn stack_matrix<const N: usize, T: Real + FromPrimitive>() -> StaticCscMatrix<N, N, MAX_NNZ, T> {
    let mut matrix = StaticCscMatrix::new();
    for column in 0..N {
        matrix.insert(column, column, cast(4.0)).unwrap();
        if column + 1 < N {
            matrix.insert(column + 1, column, cast(1.0)).unwrap();
        }
    }
    matrix
}

fn stack_indefinite<const N: usize, T: Real + FromPrimitive>() -> StaticCscMatrix<N, N, MAX_NNZ, T>
{
    let mut matrix = StaticCscMatrix::new();
    for column in 0..N {
        matrix
            .insert(
                column,
                column,
                cast(if column % 2 == 0 { 4.0 } else { -3.0 }),
            )
            .unwrap();
        if column + 1 < N {
            matrix.insert(column + 1, column, cast(1.0)).unwrap();
        }
    }
    matrix
}

fn faer_matrix<const N: usize, T: ComplexField + FromPrimitive>() -> SparseColMat<usize, T> {
    let mut column_pointers = Vec::with_capacity(N + 1);
    let mut row_indices = Vec::with_capacity(2 * N - 1);
    let mut values = Vec::with_capacity(2 * N - 1);
    column_pointers.push(0);
    for column in 0..N {
        row_indices.push(column);
        values.push(cast(4.0));
        if column + 1 < N {
            row_indices.push(column + 1);
            values.push(cast(1.0));
        }
        column_pointers.push(row_indices.len());
    }
    let symbolic = SymbolicSparseColMat::new_checked(N, N, column_pointers, None, row_indices);
    SparseColMat::new(symbolic, values)
}

fn stack_banded<const N: usize, const BAND: usize, T: Real + FromPrimitive>(
) -> StaticCscMatrix<N, N, MAX_NNZ, T> {
    let mut matrix = StaticCscMatrix::new();
    for column in 0..N {
        let end = (column + BAND + 1).min(N);
        for row in column..end {
            matrix
                .insert(row, column, cast(if row == column { 4.0 } else { 1.0 }))
                .unwrap();
        }
    }
    matrix
}

fn faer_banded<const N: usize, const BAND: usize, T: ComplexField + FromPrimitive>(
) -> SparseColMat<usize, T> {
    let mut column_pointers = Vec::with_capacity(N + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    column_pointers.push(0);
    for column in 0..N {
        let end = (column + BAND + 1).min(N);
        for row in column..end {
            row_indices.push(row);
            values.push(cast(if row == column { 4.0 } else { 1.0 }));
        }
        column_pointers.push(row_indices.len());
    }
    let symbolic = SymbolicSparseColMat::new_checked(N, N, column_pointers, None, row_indices);
    SparseColMat::new(symbolic, values)
}

fn stack_star<const N: usize, T: Real + FromPrimitive>() -> StaticCscMatrix<N, N, MAX_STAR_NNZ, T> {
    let mut matrix = StaticCscMatrix::new();
    for column in 0..N {
        matrix.insert(column, column, cast(4.0)).unwrap();
        if column == 0 {
            for row in 1..N {
                matrix.insert(row, column, cast(1.0)).unwrap();
            }
        }
    }
    matrix
}

fn faer_star<const N: usize, T: ComplexField + FromPrimitive>() -> SparseColMat<usize, T> {
    let mut column_pointers = Vec::with_capacity(N + 1);
    let mut row_indices = Vec::with_capacity(2 * N - 1);
    let mut values = Vec::with_capacity(2 * N - 1);
    column_pointers.push(0);
    for column in 0..N {
        row_indices.push(column);
        values.push(cast(4.0));
        if column == 0 {
            for row in 1..N {
                row_indices.push(row);
                values.push(cast(1.0));
            }
        }
        column_pointers.push(row_indices.len());
    }
    let symbolic = SymbolicSparseColMat::new_checked(N, N, column_pointers, None, row_indices);
    SparseColMat::new(symbolic, values)
}

fn stack_rhs<const N: usize, T: Real + FromPrimitive>() -> Matrix<N, 1, T> {
    Matrix::from_fn(|row, _| cast((row + 1) as f64))
}

fn faer_rhs<const N: usize, T: ComplexField + FromPrimitive>() -> Mat<T> {
    Mat::from_fn(N, 1, |row, _| cast((row + 1) as f64))
}

fn bench_stack_matrix<const N: usize, const CAPACITY: usize, T: Real + FromPrimitive>(
    criterion: &mut Criterion,
    scalar_name: &str,
    pattern_name: &str,
    matrix: StaticCscMatrix<N, N, CAPACITY, T>,
) {
    let symbolic = StaticCscCholeskyPattern::<N, CAPACITY>::analyze(&matrix).unwrap();
    let mut factor = symbolic.factor(&matrix).unwrap();
    let rhs = stack_rhs::<N, T>();
    let mut solution = Matrix::<N, 1, T>::zeros();

    let mut group = criterion.benchmark_group(format!("sparse-llt/{scalar_name}/{pattern_name}"));
    group.bench_with_input(BenchmarkId::new("stack-analyze", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                black_box(StaticCscCholeskyPattern::<N, CAPACITY>::analyze(black_box(
                    &matrix,
                )))
                .unwrap();
            }
        });
    });
    group.bench_with_input(
        BenchmarkId::new("stack-factor-checked", N),
        &N,
        |bench, _| {
            bench.iter(|| {
                for _ in 0..BATCH_SIZE {
                    symbolic
                        .factor_into(black_box(&matrix), black_box(&mut factor))
                        .unwrap();
                }
                black_box(&factor);
            });
        },
    );
    group.bench_with_input(BenchmarkId::new("stack-factor-reuse", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                symbolic
                    .factor_reuse_into(black_box(&matrix), black_box(&mut factor))
                    .unwrap();
            }
            black_box(&factor);
        });
    });
    group.bench_with_input(BenchmarkId::new("stack-solve", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                factor.solve_into(black_box(&rhs), black_box(&mut solution));
            }
            black_box(&solution);
        });
    });
    group.finish();
}

fn bench_stack<const N: usize, T: Real + FromPrimitive>(
    criterion: &mut Criterion,
    scalar_name: &str,
) {
    bench_stack_matrix(criterion, scalar_name, "tridiag", stack_matrix::<N, T>());
}

fn bench_stack_ldlt<const N: usize, T: Real + FromPrimitive>(
    criterion: &mut Criterion,
    scalar_name: &str,
) {
    let matrix = stack_indefinite::<N, T>();
    let symbolic = StaticCscLdltPattern::<N, MAX_NNZ>::analyze(&matrix).unwrap();
    let mut factor = symbolic.factor_ldlt(&matrix).unwrap();
    let rhs = stack_rhs::<N, T>();
    let mut solution = Matrix::<N, 1, T>::zeros();
    let mut group = criterion.benchmark_group(format!("sparse-ldlt/{scalar_name}/tridiag"));
    group.bench_with_input(BenchmarkId::new("stack-analyze", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                black_box(StaticCscLdltPattern::<N, MAX_NNZ>::analyze(black_box(
                    &matrix,
                )))
                .unwrap();
            }
        });
    });
    group.bench_with_input(BenchmarkId::new("stack-factor-reuse", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                symbolic
                    .factor_ldlt_reuse_into(black_box(&matrix), black_box(&mut factor))
                    .unwrap();
            }
            black_box(&factor);
        });
    });
    group.bench_with_input(BenchmarkId::new("stack-solve", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                factor.solve_into(black_box(&rhs), black_box(&mut solution));
            }
            black_box(&solution);
        });
    });
    group.finish();
}

fn bench_faer_matrix<const N: usize, T: ComplexField + FromPrimitive>(
    criterion: &mut Criterion,
    scalar_name: &str,
    pattern_name: &str,
    matrix: SparseColMat<usize, T>,
) {
    let symbolic = SymbolicLlt::try_new(matrix.symbolic(), Side::Lower).unwrap();
    let factor =
        Llt::try_new_with_symbolic(symbolic.clone(), matrix.as_ref(), Side::Lower).unwrap();
    let rhs = faer_rhs::<N, T>();
    let mut solution = faer_rhs::<N, T>();

    let mut group = criterion.benchmark_group(format!("sparse-llt/{scalar_name}/{pattern_name}"));
    group.bench_with_input(BenchmarkId::new("faer-analyze", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                black_box(SymbolicLlt::try_new(
                    black_box(matrix.symbolic()),
                    Side::Lower,
                ))
                .unwrap();
            }
        });
    });
    group.bench_with_input(BenchmarkId::new("faer-factor-reuse", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                black_box(Llt::try_new_with_symbolic(
                    symbolic.clone(),
                    black_box(matrix.as_ref()),
                    Side::Lower,
                ))
                .unwrap();
            }
        });
    });
    group.bench_with_input(BenchmarkId::new("faer-solve", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                solution.copy_from(black_box(&rhs));
                factor.solve_in_place(black_box(&mut solution));
            }
            black_box(&solution);
        });
    });
    group.finish();
}

fn bench_faer<const N: usize, T: ComplexField + FromPrimitive>(
    criterion: &mut Criterion,
    scalar_name: &str,
) {
    bench_faer_matrix::<N, T>(criterion, scalar_name, "tridiag", faer_matrix::<N, T>());
}

fn bench_pattern<const N: usize, T: Real + ComplexField + FromPrimitive>(
    criterion: &mut Criterion,
    scalar_name: &str,
) {
    bench_stack_matrix(criterion, scalar_name, "band2", stack_banded::<N, 2, T>());
    bench_faer_matrix::<N, T>(criterion, scalar_name, "band2", faer_banded::<N, 2, T>());
    bench_stack_matrix(criterion, scalar_name, "star", stack_star::<N, T>());
    bench_faer_matrix::<N, T>(criterion, scalar_name, "star", faer_star::<N, T>());
    bench_stack_ordered_star::<N, T>(criterion, scalar_name);
}

fn bench_stack_ordered_star<const N: usize, T: Real + FromPrimitive>(
    criterion: &mut Criterion,
    scalar_name: &str,
) {
    let matrix = stack_star::<N, T>();
    let ordering = StaticCscOrdering::minimum_degree(&matrix);
    let symbolic =
        StaticCscCholeskyPattern::<N, MAX_STAR_NNZ>::analyze_with_ordering(&matrix, ordering)
            .unwrap();
    let ordered = symbolic.prepare_ordered(&matrix).unwrap();
    let mut factor = symbolic.factor_ordered(&ordered).unwrap();
    let mut factor_with_permutation = symbolic.factor(&matrix).unwrap();
    let rhs = stack_rhs::<N, T>();
    let mut solution = Matrix::<N, 1, T>::zeros();
    let mut group = criterion.benchmark_group(format!("sparse-llt/{scalar_name}/star-min-degree"));
    group.bench_with_input(BenchmarkId::new("stack-permute", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                black_box(ordering.permute(black_box(&matrix))).unwrap();
            }
        });
    });
    group.bench_with_input(
        BenchmarkId::new("stack-factor-with-permutation", N),
        &N,
        |bench, _| {
            bench.iter(|| {
                for _ in 0..BATCH_SIZE {
                    symbolic
                        .factor_into(black_box(&matrix), black_box(&mut factor_with_permutation))
                        .unwrap();
                }
                black_box(&factor_with_permutation);
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("stack-factor-reuse-with-permutation", N),
        &N,
        |bench, _| {
            bench.iter(|| {
                for _ in 0..BATCH_SIZE {
                    symbolic
                        .factor_reuse_into(
                            black_box(&matrix),
                            black_box(&mut factor_with_permutation),
                        )
                        .unwrap();
                }
                black_box(&factor_with_permutation);
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("stack-factor-ordered", N),
        &N,
        |bench, _| {
            bench.iter(|| {
                for _ in 0..BATCH_SIZE {
                    symbolic
                        .factor_ordered_into(black_box(&ordered), black_box(&mut factor))
                        .unwrap();
                }
                black_box(&factor);
            });
        },
    );
    group.bench_with_input(BenchmarkId::new("stack-solve", N), &N, |bench, _| {
        bench.iter(|| {
            for _ in 0..BATCH_SIZE {
                factor.solve_into(black_box(&rhs), black_box(&mut solution));
            }
            black_box(&solution);
        });
    });
    group.finish();
}

fn bench_all(criterion: &mut Criterion) {
    for scalar_name in ["f32", "f64"] {
        if scalar_name == "f32" {
            bench_stack::<3, f32>(criterion, scalar_name);
            bench_stack::<6, f32>(criterion, scalar_name);
            bench_stack::<15, f32>(criterion, scalar_name);
            bench_stack::<32, f32>(criterion, scalar_name);
            bench_stack_ldlt::<15, f32>(criterion, scalar_name);
            bench_faer::<3, f32>(criterion, scalar_name);
            bench_faer::<6, f32>(criterion, scalar_name);
            bench_faer::<15, f32>(criterion, scalar_name);
            bench_faer::<32, f32>(criterion, scalar_name);
            bench_pattern::<15, f32>(criterion, scalar_name);
        } else {
            bench_stack::<3, f64>(criterion, scalar_name);
            bench_stack::<6, f64>(criterion, scalar_name);
            bench_stack::<15, f64>(criterion, scalar_name);
            bench_stack::<32, f64>(criterion, scalar_name);
            bench_stack_ldlt::<15, f64>(criterion, scalar_name);
            bench_faer::<3, f64>(criterion, scalar_name);
            bench_faer::<6, f64>(criterion, scalar_name);
            bench_faer::<15, f64>(criterion, scalar_name);
            bench_faer::<32, f64>(criterion, scalar_name);
            bench_pattern::<15, f64>(criterion, scalar_name);
        }
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
