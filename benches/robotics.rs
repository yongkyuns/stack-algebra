use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dyn_stack::{MemBuffer, MemStack};
use faer::{Accum, Mat, Par};
use stack_algebra::Matrix;

const BATCH_SIZE: usize = 64;

macro_rules! scalar_benches {
    ($module:ident, $scalar:ty, $scalar_name:literal) => {
        mod $module {
            use super::*;

            fn stack_matrix<const ROWS: usize, const COLUMNS: usize>(
            ) -> Matrix<ROWS, COLUMNS, $scalar> {
                Matrix::from_fn(|row, column| (row * COLUMNS + column + 1) as $scalar / 17.0)
            }

            fn faer_matrix<const ROWS: usize, const COLUMNS: usize>() -> Mat<$scalar> {
                Mat::from_fn(ROWS, COLUMNS, |row, column| {
                    (row * COLUMNS + column + 1) as $scalar / 17.0
                })
            }

            fn stack_system<const DIMENSION: usize>() -> Matrix<DIMENSION, DIMENSION, $scalar> {
                Matrix::from_fn(|row, column| {
                    if row == column {
                        (DIMENSION + 1) as $scalar
                    } else {
                        (row + 2 * column + 1) as $scalar / 19.0
                    }
                })
            }

            fn faer_system<const DIMENSION: usize>() -> Mat<$scalar> {
                Mat::from_fn(DIMENSION, DIMENSION, |row, column| {
                    if row == column {
                        (DIMENSION + 1) as $scalar
                    } else {
                        (row + 2 * column + 1) as $scalar / 19.0
                    }
                })
            }

            fn bench_stack_product<const ROWS: usize, const SHARED: usize, const COLUMNS: usize>(
                criterion: &mut Criterion,
                group_name: &str,
            ) {
                let lhs = stack_matrix::<ROWS, SHARED>();
                let rhs = Matrix::<SHARED, COLUMNS, $scalar>::from_fn(|row, column| {
                    (row + 2 * column + 3) as $scalar / 11.0
                });
                let mut output = Matrix::<ROWS, COLUMNS, $scalar>::zeros();
                let mut group = criterion.benchmark_group(group_name);
                let size = format!("{ROWS}x{SHARED}x{COLUMNS}");
                group.bench_with_input(BenchmarkId::new("stack-algebra", size), &(), |bench, _| {
                    bench.iter(|| {
                        for _ in 0..BATCH_SIZE {
                            black_box(&lhs).mul_into(black_box(&rhs), black_box(&mut output));
                        }
                        black_box(&output);
                    });
                });
                group.finish();
            }

            fn bench_faer_product<const ROWS: usize, const SHARED: usize, const COLUMNS: usize>(
                criterion: &mut Criterion,
                group_name: &str,
            ) {
                let lhs = faer_matrix::<ROWS, SHARED>();
                let rhs = Mat::<$scalar>::from_fn(SHARED, COLUMNS, |row, column| {
                    (row + 2 * column + 3) as $scalar / 11.0
                });
                let mut output = Mat::<$scalar>::zeros(ROWS, COLUMNS);
                let mut group = criterion.benchmark_group(group_name);
                let size = format!("{ROWS}x{SHARED}x{COLUMNS}");
                group.bench_with_input(BenchmarkId::new("faer-dynamic", size), &(), |bench, _| {
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

            fn bench_stack_lu_factor<const DIMENSION: usize>(criterion: &mut Criterion) {
                let input = stack_system::<DIMENSION>();
                let mut factor = input.partial_piv_lu();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/lu-factor/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("stack-algebra", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                factor = black_box(&input).partial_piv_lu();
                            }
                            black_box(&factor);
                        });
                    },
                );
                group.finish();
            }

            fn bench_faer_lu_factor<const DIMENSION: usize>(criterion: &mut Criterion) {
                let input = faer_system::<DIMENSION>();
                let mut factor = input.partial_piv_lu();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/lu-factor/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("faer-dynamic", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                factor = black_box(&input).partial_piv_lu();
                            }
                            black_box(&factor);
                        });
                    },
                );
                group.finish();
            }

            fn bench_stack_lu_solve<const DIMENSION: usize>(criterion: &mut Criterion) {
                let factor = stack_system::<DIMENSION>().partial_piv_lu();
                let rhs = stack_matrix::<DIMENSION, 1>();
                let mut solution = Matrix::<DIMENSION, 1, $scalar>::zeros();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/lu-solve/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("stack-algebra", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                solution = black_box(&factor).solve(black_box(&rhs));
                            }
                            black_box(&solution);
                        });
                    },
                );
                group.finish();
            }

            fn bench_faer_lu_solve<const DIMENSION: usize>(criterion: &mut Criterion) {
                let factor = faer_system::<DIMENSION>().partial_piv_lu();
                let rhs = faer_matrix::<DIMENSION, 1>();
                let mut solution = Mat::<$scalar>::zeros(DIMENSION, 1);
                let scratch = faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<
                    usize,
                    $scalar,
                >(DIMENSION, 1, Par::Seq);
                let mut scratch = MemBuffer::new(scratch);
                let mut group =
                    criterion.benchmark_group(concat!("robotics/lu-solve/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("faer-dynamic", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                solution.copy_from(black_box(&rhs));
                                faer::linalg::lu::partial_pivoting::solve::solve_in_place(
                                    black_box(factor.L()),
                                    black_box(factor.U()),
                                    black_box(factor.P()),
                                    black_box(solution.as_mut()),
                                    Par::Seq,
                                    MemStack::new(&mut scratch),
                                );
                            }
                            black_box(&solution);
                        });
                    },
                );
                group.finish();
            }

            pub fn bench_all(criterion: &mut Criterion) {
                let matmul_group = concat!("robotics/matmul/", $scalar_name);
                let matvec_group = concat!("robotics/matvec/", $scalar_name);
                for_bench_product!(
                    criterion,
                    matmul_group,
                    bench_stack_product,
                    bench_faer_product,
                    (2, 3, 2),
                    (3, 6, 3),
                    (6, 15, 6)
                );
                for_bench_product!(
                    criterion,
                    matvec_group,
                    bench_stack_product,
                    bench_faer_product,
                    (3, 3, 1),
                    (6, 6, 1),
                    (15, 15, 1)
                );
                for_bench_dimension!(
                    criterion,
                    bench_stack_lu_factor,
                    bench_faer_lu_factor,
                    3,
                    6,
                    15
                );
                for_bench_dimension!(
                    criterion,
                    bench_stack_lu_solve,
                    bench_faer_lu_solve,
                    3,
                    6,
                    15
                );
            }
        }
    };
}

macro_rules! for_bench_product {
    ($criterion:expr, $group:expr, $stack:ident, $faer:ident, $(($rows:expr, $shared:expr, $columns:expr)),+ $(,)?) => {
        $(
            $stack::<$rows, $shared, $columns>($criterion, $group);
            $faer::<$rows, $shared, $columns>($criterion, $group);
        )+
    };
}

macro_rules! for_bench_dimension {
    ($criterion:expr, $stack:ident, $faer:ident, $($dimension:expr),+ $(,)?) => {
        $(
            $stack::<$dimension>($criterion);
            $faer::<$dimension>($criterion);
        )+
    };
}

scalar_benches!(f32_benches, f32, "f32");
scalar_benches!(f64_benches, f64, "f64");

fn bench_all(criterion: &mut Criterion) {
    f32_benches::bench_all(criterion);
    f64_benches::bench_all(criterion);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
