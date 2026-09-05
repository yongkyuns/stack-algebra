use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::linalg::solvers::Solve;
use faer::{Mat, Par, Side};
use stack_algebra::Matrix;

macro_rules! scalar_benches {
    ($module:ident, $scalar:ty, $scalar_name:literal, $tolerance:expr) => {
        mod $module {
            use super::*;

            fn general_matrix<const D: usize>() -> Matrix<D, D, $scalar> {
                Matrix::from_fn(|row, column| {
                    if row == column {
                        (D + 2) as $scalar
                    } else {
                        (row + 2 * column + 1) as $scalar / 23.0
                    }
                })
            }

            fn spd_matrix<const D: usize>() -> Matrix<D, D, $scalar> {
                Matrix::from_fn(|row, column| {
                    let mut value = 0.0 as $scalar;
                    for shared in 0..D {
                        let left = (shared + 3 * row + 1) as $scalar / 23.0;
                        let right = (shared + 3 * column + 1) as $scalar / 23.0;
                        value += left * right;
                    }
                    value
                        + if row == column {
                            D as $scalar
                        } else {
                            0.0 as $scalar
                        }
                })
            }

            fn indefinite_matrix<const D: usize>() -> Matrix<D, D, $scalar> {
                Matrix::from_fn(|row, column| {
                    if row == column {
                        if row % 2 == 0 {
                            -(D as $scalar)
                        } else {
                            (D + 1) as $scalar
                        }
                    } else {
                        (row + column + 1) as $scalar / 29.0
                    }
                })
            }

            fn lower_matrix<const D: usize>() -> Matrix<D, D, $scalar> {
                Matrix::from_fn(|row, column| {
                    if row < column {
                        0.0 as $scalar
                    } else if row == column {
                        (D + row + 2) as $scalar
                    } else {
                        (row + column + 1) as $scalar / 19.0
                    }
                })
            }

            fn upper_matrix<const D: usize>() -> Matrix<D, D, $scalar> {
                Matrix::from_fn(|row, column| {
                    if row > column {
                        0.0 as $scalar
                    } else if row == column {
                        (D + row + 2) as $scalar
                    } else {
                        (row + column + 1) as $scalar / 19.0
                    }
                })
            }

            fn rhs<const D: usize, const P: usize>() -> Matrix<D, P, $scalar> {
                Matrix::from_fn(|row, column| (row + 2 * column + 3) as $scalar / 11.0)
            }

            fn faer_matrix<const D: usize>(matrix: &Matrix<D, D, $scalar>) -> Mat<$scalar> {
                Mat::from_fn(D, D, |row, column| matrix[(row, column)])
            }

            fn faer_rhs<const D: usize, const P: usize>(
                rhs: &Matrix<D, P, $scalar>,
            ) -> Mat<$scalar> {
                Mat::from_fn(D, P, |row, column| rhs[(row, column)])
            }

            fn stack_from_faer<const D: usize, const P: usize>(
                matrix: &Mat<$scalar>,
            ) -> Matrix<D, P, $scalar> {
                Matrix::from_fn(|row, column| matrix[(row, column)])
            }

            fn assert_residual<const D: usize, const P: usize>(
                matrix: &Matrix<D, D, $scalar>,
                solution: &Matrix<D, P, $scalar>,
                rhs: &Matrix<D, P, $scalar>,
            ) {
                let residual = *matrix * *solution - *rhs;
                let mut residual_max = 0.0 as $scalar;
                let mut rhs_max = 1.0 as $scalar;
                for &value in residual.as_slice() {
                    residual_max = residual_max.max(value.abs());
                }
                for &value in rhs.as_slice() {
                    rhs_max = rhs_max.max(value.abs());
                }
                assert!(
                    residual_max <= ($tolerance as $scalar) * rhs_max,
                    "residual {residual_max:?} exceeds tolerance for D={D}, P={P}"
                );
            }

            fn bench_cholesky<const D: usize, const P: usize>(criterion: &mut Criterion) {
                let matrix = spd_matrix::<D>();
                let rhs = rhs::<D, P>();
                let stack_factor = matrix.cholesky().expect("benchmark matrix is SPD");
                let mut stack_output = Matrix::<D, P, $scalar>::zeros();
                stack_factor.solve_into(&rhs, &mut stack_output);
                assert_residual(&matrix, &stack_output, &rhs);

                let faer_matrix = faer_matrix(&matrix);
                let faer_rhs = faer_rhs(&rhs);
                let faer_factor = faer_matrix
                    .llt(Side::Lower)
                    .expect("benchmark matrix is SPD");
                let mut faer_output = faer_rhs.clone();
                faer_factor.solve_in_place(&mut faer_output);
                let faer_solution = stack_from_faer::<D, P>(&faer_output);
                assert_residual(&matrix, &faer_solution, &rhs);

                let mut group =
                    criterion.benchmark_group(concat!("dense-multi-rhs/llt/", $scalar_name));
                let shape = format!("{D}x{P}");
                group.bench_with_input(
                    BenchmarkId::new("stack-algebra", &shape),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            black_box(&stack_factor)
                                .solve_into(black_box(&rhs), black_box(&mut stack_output));
                            black_box(&stack_output);
                        });
                    },
                );
                group.bench_with_input(BenchmarkId::new("faer", &shape), &(), |bench, _| {
                    bench.iter(|| {
                        faer_output.copy_from(black_box(&faer_rhs));
                        black_box(&faer_factor).solve_in_place(black_box(&mut faer_output));
                        black_box(&faer_output);
                    });
                });
                group.finish();
            }

            fn bench_ldlt<const D: usize, const P: usize>(criterion: &mut Criterion) {
                let matrix = indefinite_matrix::<D>();
                let rhs = rhs::<D, P>();
                let stack_factor = matrix.ldlt().expect("benchmark matrix is nonsingular");
                let stack_no_pivot_factor = matrix
                    .ldlt_no_pivot()
                    .expect("benchmark matrix supports no-pivot LDLT");
                let mut stack_output = Matrix::<D, P, $scalar>::zeros();
                stack_factor.solve_into(&rhs, &mut stack_output);
                assert_residual(&matrix, &stack_output, &rhs);
                stack_no_pivot_factor.solve_into(&rhs, &mut stack_output);
                assert_residual(&matrix, &stack_output, &rhs);

                let faer_matrix = faer_matrix(&matrix);
                let faer_rhs = faer_rhs(&rhs);
                let faer_factor = faer_matrix
                    .ldlt(Side::Lower)
                    .expect("benchmark matrix is nonsingular");
                let mut faer_output = faer_rhs.clone();
                faer_factor.solve_in_place(&mut faer_output);
                let faer_solution = stack_from_faer::<D, P>(&faer_output);
                assert_residual(&matrix, &faer_solution, &rhs);

                let mut group =
                    criterion.benchmark_group(concat!("dense-multi-rhs/ldlt/", $scalar_name));
                let shape = format!("{D}x{P}");
                group.bench_with_input(
                    BenchmarkId::new("stack-algebra", &shape),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            black_box(&stack_factor)
                                .solve_into(black_box(&rhs), black_box(&mut stack_output));
                            black_box(&stack_output);
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("stack-no-pivot", &shape),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            black_box(&stack_no_pivot_factor)
                                .solve_into(black_box(&rhs), black_box(&mut stack_output));
                            black_box(&stack_output);
                        });
                    },
                );
                group.bench_with_input(BenchmarkId::new("faer", &shape), &(), |bench, _| {
                    bench.iter(|| {
                        faer_output.copy_from(black_box(&faer_rhs));
                        black_box(&faer_factor).solve_in_place(black_box(&mut faer_output));
                        black_box(&faer_output);
                    });
                });
                group.finish();
            }

            fn bench_lu<const D: usize, const P: usize>(criterion: &mut Criterion) {
                let matrix = general_matrix::<D>();
                let rhs = rhs::<D, P>();
                let stack_factor = matrix.partial_piv_lu();
                let mut stack_output = Matrix::<D, P, $scalar>::zeros();
                stack_factor.solve_into(&rhs, &mut stack_output);
                assert_residual(&matrix, &stack_output, &rhs);

                let faer_matrix = faer_matrix(&matrix);
                let faer_rhs = faer_rhs(&rhs);
                let faer_factor = faer_matrix.partial_piv_lu();
                let mut faer_output = faer_rhs.clone();
                faer_factor.solve_in_place(&mut faer_output);
                let faer_solution = stack_from_faer::<D, P>(&faer_output);
                assert_residual(&matrix, &faer_solution, &rhs);

                let mut group =
                    criterion.benchmark_group(concat!("dense-multi-rhs/lu/", $scalar_name));
                let shape = format!("{D}x{P}");
                group.bench_with_input(
                    BenchmarkId::new("stack-algebra", &shape),
                    &(),
                    |bench, _| {
                        bench.iter(|| {
                            black_box(&stack_factor)
                                .solve_into(black_box(&rhs), black_box(&mut stack_output));
                            black_box(&stack_output);
                        });
                    },
                );
                group.bench_with_input(BenchmarkId::new("faer", &shape), &(), |bench, _| {
                    bench.iter(|| {
                        faer_output.copy_from(black_box(&faer_rhs));
                        black_box(&faer_factor).solve_in_place(black_box(&mut faer_output));
                        black_box(&faer_output);
                    });
                });
                group.finish();
            }

            fn bench_triangular<const D: usize, const P: usize>(criterion: &mut Criterion) {
                let rhs = rhs::<D, P>();
                let lower = lower_matrix::<D>();
                let upper = upper_matrix::<D>();
                let lower_view = lower.lower_triangular();
                let upper_view = upper.upper_triangular();
                let mut stack_output = Matrix::<D, P, $scalar>::zeros();

                lower_view.solve_into(&rhs, &mut stack_output);
                assert_residual(&lower, &stack_output, &rhs);
                upper_view.solve_into(&rhs, &mut stack_output);
                assert_residual(&upper, &stack_output, &rhs);

                let faer_lower = faer_matrix(&lower);
                let faer_upper = faer_matrix(&upper);
                let faer_rhs = faer_rhs(&rhs);
                let mut faer_output = faer_rhs.clone();
                faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                    faer_lower.as_ref(),
                    faer_output.as_mut(),
                    Par::Seq,
                );
                let faer_solution = stack_from_faer::<D, P>(&faer_output);
                assert_residual(&lower, &faer_solution, &rhs);
                faer_output.copy_from(&faer_rhs);
                faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                    faer_upper.as_ref(),
                    faer_output.as_mut(),
                    Par::Seq,
                );
                let faer_solution = stack_from_faer::<D, P>(&faer_output);
                assert_residual(&upper, &faer_solution, &rhs);

                let mut group =
                    criterion.benchmark_group(concat!("dense-multi-rhs/triangular/", $scalar_name));
                let shape = format!("{D}x{P}");
                group.bench_with_input(BenchmarkId::new("stack-lower", &shape), &(), |bench, _| {
                    bench.iter(|| {
                        lower_view.solve_into(black_box(&rhs), black_box(&mut stack_output));
                        black_box(&stack_output);
                    });
                });
                group.bench_with_input(BenchmarkId::new("faer-lower", &shape), &(), |bench, _| {
                    bench.iter(|| {
                        faer_output.copy_from(black_box(&faer_rhs));
                        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                            black_box(faer_lower.as_ref()),
                            black_box(faer_output.as_mut()),
                            Par::Seq,
                        );
                        black_box(&faer_output);
                    });
                });
                group.bench_with_input(BenchmarkId::new("stack-upper", &shape), &(), |bench, _| {
                    bench.iter(|| {
                        upper_view.solve_into(black_box(&rhs), black_box(&mut stack_output));
                        black_box(&stack_output);
                    });
                });
                group.bench_with_input(BenchmarkId::new("faer-upper", &shape), &(), |bench, _| {
                    bench.iter(|| {
                        faer_output.copy_from(black_box(&faer_rhs));
                        faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                            black_box(faer_upper.as_ref()),
                            black_box(faer_output.as_mut()),
                            Par::Seq,
                        );
                        black_box(&faer_output);
                    });
                });
                group.finish();
            }

            fn bench_case<const D: usize, const P: usize>(criterion: &mut Criterion) {
                bench_cholesky::<D, P>(criterion);
                bench_ldlt::<D, P>(criterion);
                bench_lu::<D, P>(criterion);
                bench_triangular::<D, P>(criterion);
            }

            pub fn bench_all(criterion: &mut Criterion) {
                bench_case::<6, 1>(criterion);
                bench_case::<6, 2>(criterion);
                bench_case::<6, 4>(criterion);
                bench_case::<8, 1>(criterion);
                bench_case::<8, 2>(criterion);
                bench_case::<8, 4>(criterion);
                bench_case::<15, 1>(criterion);
                bench_case::<15, 2>(criterion);
                bench_case::<15, 4>(criterion);
                bench_case::<16, 1>(criterion);
                bench_case::<16, 2>(criterion);
                bench_case::<16, 4>(criterion);
                bench_case::<32, 1>(criterion);
                bench_case::<32, 2>(criterion);
                bench_case::<32, 4>(criterion);
            }
        }
    };
}

scalar_benches!(f32_benches, f32, "f32", 2.0e-4);
scalar_benches!(f64_benches, f64, "f64", 1.0e-10);

fn bench_all(criterion: &mut Criterion) {
    f32_benches::bench_all(criterion);
    f64_benches::bench_all(criterion);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
