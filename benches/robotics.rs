use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::solvers::Solve;
use faer::{Accum, Mat, Par, Side};
use stack_algebra::{Matrix, Vector};

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

            fn stack_spd_system<const DIMENSION: usize>() -> Matrix<DIMENSION, DIMENSION, $scalar> {
                Matrix::from_fn(|row, column| {
                    let mut value = <$scalar>::default();
                    for shared in 0..DIMENSION {
                        let left = (shared + 3 * row + 1) as $scalar / 23.0;
                        let right = (shared + 3 * column + 1) as $scalar / 23.0;
                        value += left * right;
                    }
                    value
                        + if row == column {
                            DIMENSION as $scalar
                        } else {
                            0.0
                        }
                })
            }

            fn stack_ldlt_system<const DIMENSION: usize>() -> Matrix<DIMENSION, DIMENSION, $scalar>
            {
                Matrix::from_fn(|row, column| {
                    if row == column {
                        if row % 2 == 0 {
                            -(DIMENSION as $scalar)
                        } else {
                            (DIMENSION + 1) as $scalar
                        }
                    } else {
                        (row + column + 1) as $scalar / 29.0
                    }
                })
            }

            fn stack_rhs<const DIMENSION: usize>() -> Matrix<DIMENSION, 1, $scalar> {
                Matrix::from_fn(|row, column| (row + 2 * column + 3) as $scalar / 11.0)
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

            fn faer_spd_system<const DIMENSION: usize>() -> Mat<$scalar> {
                Mat::from_fn(DIMENSION, DIMENSION, |row, column| {
                    let mut value = <$scalar>::default();
                    for shared in 0..DIMENSION {
                        let left = (shared + 3 * row + 1) as $scalar / 23.0;
                        let right = (shared + 3 * column + 1) as $scalar / 23.0;
                        value += left * right;
                    }
                    value
                        + if row == column {
                            DIMENSION as $scalar
                        } else {
                            0.0
                        }
                })
            }

            fn faer_ldlt_system<const DIMENSION: usize>() -> Mat<$scalar> {
                Mat::from_fn(DIMENSION, DIMENSION, |row, column| {
                    if row == column {
                        if row % 2 == 0 {
                            -(DIMENSION as $scalar)
                        } else {
                            (DIMENSION + 1) as $scalar
                        }
                    } else {
                        (row + column + 1) as $scalar / 29.0
                    }
                })
            }

            fn faer_rhs<const DIMENSION: usize>() -> Mat<$scalar> {
                Mat::from_fn(DIMENSION, 1, |row, column| {
                    (row + 2 * column + 3) as $scalar / 11.0
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

            fn bench_stack_norm<const ROWS: usize, const COLUMNS: usize>(
                criterion: &mut Criterion,
                group_name: &str,
            ) {
                let input = stack_matrix::<ROWS, COLUMNS>();
                let mut output = <$scalar>::default();
                let mut group = criterion.benchmark_group(group_name);
                let size = format!("{ROWS}x{COLUMNS}");
                group.bench_with_input(BenchmarkId::new("stack-algebra", size), &(), |bench, _| {
                    bench.iter(|| {
                        for _ in 0..BATCH_SIZE {
                            output = black_box(&input).norm();
                        }
                        black_box(output);
                    });
                });
                group.finish();
            }

            fn bench_faer_norm<const ROWS: usize, const COLUMNS: usize>(
                criterion: &mut Criterion,
                group_name: &str,
            ) {
                let input = faer_matrix::<ROWS, COLUMNS>();
                let mut output = <$scalar>::default();
                let mut group = criterion.benchmark_group(group_name);
                let size = format!("{ROWS}x{COLUMNS}");
                group.bench_with_input(BenchmarkId::new("faer-dynamic", size), &(), |bench, _| {
                    bench.iter(|| {
                        for _ in 0..BATCH_SIZE {
                            output = black_box(&input).norm_l2();
                        }
                        black_box(output);
                    });
                });
                group.finish();
            }

            fn bench_stack_dot<const DIMENSION: usize>(criterion: &mut Criterion) {
                let lhs =
                    Vector::<DIMENSION, $scalar>::from_fn(|row, _| (row + 1) as $scalar / 13.0);
                let rhs =
                    Vector::<DIMENSION, $scalar>::from_fn(|row, _| (2 * row + 3) as $scalar / 7.0);
                let mut output = <$scalar>::default();
                let mut group = criterion.benchmark_group(concat!("robotics/dot/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("stack-algebra", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                output = black_box(&lhs).dot(black_box(&rhs));
                            }
                            black_box(output);
                        });
                    },
                );
                group.finish();
            }

            fn bench_faer_dot<const DIMENSION: usize>(criterion: &mut Criterion) {
                let lhs = Mat::<$scalar>::from_fn(1, DIMENSION, |_, column| {
                    (column + 1) as $scalar / 13.0
                });
                let rhs =
                    Mat::<$scalar>::from_fn(DIMENSION, 1, |row, _| (2 * row + 3) as $scalar / 7.0);
                let mut output = <$scalar>::default();
                let mut group = criterion.benchmark_group(concat!("robotics/dot/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("faer-dynamic", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                output = faer::linalg::matmul::dot::inner_prod(
                                    black_box(&lhs).row(0),
                                    faer::Conj::No,
                                    black_box(&rhs).col(0),
                                    faer::Conj::No,
                                );
                            }
                            black_box(output);
                        });
                    },
                );
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

            fn bench_stack_llt_factor<const DIMENSION: usize>(criterion: &mut Criterion) {
                let input = stack_spd_system::<DIMENSION>();
                let mut factor = input.cholesky();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/llt-factor/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("stack-algebra", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                factor = black_box(&input).cholesky();
                            }
                            black_box(&factor);
                        });
                    },
                );
                group.finish();
            }

            fn bench_stack_llt_solve<const DIMENSION: usize>(criterion: &mut Criterion) {
                let factor = stack_spd_system::<DIMENSION>()
                    .cholesky()
                    .expect("benchmark system is positive-definite");
                let rhs = stack_rhs::<DIMENSION>();
                let mut solution = Matrix::<DIMENSION, 1, $scalar>::zeros();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/llt-solve/", $scalar_name));
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

            fn bench_faer_llt_factor<const DIMENSION: usize>(criterion: &mut Criterion) {
                let input = faer_spd_system::<DIMENSION>();
                let mut factor = input.llt(Side::Lower);
                let mut group =
                    criterion.benchmark_group(concat!("robotics/llt-factor/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("faer-dynamic", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                factor = black_box(&input).llt(Side::Lower);
                            }
                            black_box(&factor);
                        });
                    },
                );
                group.finish();
            }

            fn bench_faer_llt_solve<const DIMENSION: usize>(criterion: &mut Criterion) {
                let input = faer_spd_system::<DIMENSION>();
                let factor = input
                    .llt(Side::Lower)
                    .expect("benchmark system is positive-definite");
                let rhs = faer_rhs::<DIMENSION>();
                let mut solution = faer_rhs::<DIMENSION>();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/llt-solve/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("faer-dynamic", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                solution.copy_from(black_box(&rhs));
                                black_box(&factor).solve_in_place(black_box(&mut solution));
                            }
                            black_box(&solution);
                        });
                    },
                );
                group.finish();
            }

            fn bench_stack_ldlt_factor<const DIMENSION: usize>(criterion: &mut Criterion) {
                let input = stack_ldlt_system::<DIMENSION>();
                let mut factor = input.ldlt();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/ldlt-factor/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("stack-algebra", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                factor = black_box(&input).ldlt();
                            }
                            black_box(&factor);
                        });
                    },
                );
                group.finish();
            }

            fn bench_stack_ldlt_no_pivot_factor<const DIMENSION: usize>(criterion: &mut Criterion) {
                let input = stack_ldlt_system::<DIMENSION>();
                let mut factor = input.ldlt_no_pivot();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/ldlt-factor/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("stack-algebra-no-pivot", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                factor = black_box(&input).ldlt_no_pivot();
                            }
                            black_box(&factor);
                        });
                    },
                );
                group.finish();
            }

            fn bench_stack_ldlt_solve<const DIMENSION: usize>(criterion: &mut Criterion) {
                let factor = stack_ldlt_system::<DIMENSION>()
                    .ldlt()
                    .expect("benchmark system is nonsingular");
                let rhs = stack_rhs::<DIMENSION>();
                let mut solution = Matrix::<DIMENSION, 1, $scalar>::zeros();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/ldlt-solve/", $scalar_name));
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

            fn bench_stack_ldlt_no_pivot_solve<const DIMENSION: usize>(criterion: &mut Criterion) {
                let factor = stack_ldlt_system::<DIMENSION>()
                    .ldlt_no_pivot()
                    .expect("benchmark system is nonsingular without pivoting");
                let rhs = stack_rhs::<DIMENSION>();
                let mut solution = Matrix::<DIMENSION, 1, $scalar>::zeros();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/ldlt-solve/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("stack-algebra-no-pivot", DIMENSION),
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

            fn bench_faer_ldlt_factor<const DIMENSION: usize>(criterion: &mut Criterion) {
                let input = faer_ldlt_system::<DIMENSION>();
                let mut factor = input.ldlt(Side::Lower);
                let mut group =
                    criterion.benchmark_group(concat!("robotics/ldlt-factor/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("faer-dynamic", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                factor = black_box(&input).ldlt(Side::Lower);
                            }
                            black_box(&factor);
                        });
                    },
                );
                group.finish();
            }

            fn bench_faer_ldlt_solve<const DIMENSION: usize>(criterion: &mut Criterion) {
                let input = faer_ldlt_system::<DIMENSION>();
                let factor = input
                    .ldlt(Side::Lower)
                    .expect("benchmark system is nonsingular");
                let rhs = faer_rhs::<DIMENSION>();
                let mut solution = faer_rhs::<DIMENSION>();
                let mut group =
                    criterion.benchmark_group(concat!("robotics/ldlt-solve/", $scalar_name));
                group.bench_with_input(
                    BenchmarkId::new("faer-dynamic", DIMENSION),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            for _ in 0..BATCH_SIZE {
                                solution.copy_from(black_box(&rhs));
                                black_box(&factor).solve_in_place(black_box(&mut solution));
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
                let norm_group = concat!("robotics/norm/", $scalar_name);
                for_bench_norm!(
                    criterion,
                    norm_group,
                    bench_stack_norm,
                    bench_faer_norm,
                    (3, 3),
                    (6, 6),
                    (15, 15),
                    (6, 15)
                );
                for_bench_dimension!(criterion, bench_stack_dot, bench_faer_dot, 3, 6, 15);
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
                for_stack_dimension!(criterion, bench_stack_llt_factor, 3, 6, 15, 32);
                for_stack_dimension!(criterion, bench_stack_llt_solve, 3, 6, 15, 32);
                for_stack_dimension!(criterion, bench_faer_llt_factor, 3, 6, 15, 32);
                for_stack_dimension!(criterion, bench_faer_llt_solve, 3, 6, 15, 32);
                for_stack_dimension!(criterion, bench_stack_ldlt_factor, 3, 6, 15, 32);
                for_stack_dimension!(criterion, bench_stack_ldlt_no_pivot_factor, 3, 6, 15, 32);
                for_stack_dimension!(criterion, bench_stack_ldlt_solve, 3, 6, 15, 32);
                for_stack_dimension!(criterion, bench_stack_ldlt_no_pivot_solve, 3, 6, 15, 32);
                for_stack_dimension!(criterion, bench_faer_ldlt_factor, 3, 6, 15, 32);
                for_stack_dimension!(criterion, bench_faer_ldlt_solve, 3, 6, 15, 32);
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

macro_rules! for_bench_norm {
    ($criterion:expr, $group:expr, $stack:ident, $faer:ident, $(($rows:expr, $columns:expr)),+ $(,)?) => {
        $(
            $stack::<$rows, $columns>($criterion, $group);
            $faer::<$rows, $columns>($criterion, $group);
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

macro_rules! for_stack_dimension {
    ($criterion:expr, $stack:ident, $($dimension:expr),+ $(,)?) => {
        $(
            $stack::<$dimension>($criterion);
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
