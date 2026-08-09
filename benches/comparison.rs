//! Cross-library dense benchmark suite.
//!
//! Every group uses `comparison/<operation>/<scalar>` and every benchmark id is
//! `<implementation>/<shape>`. Inputs are deterministic and setup is kept out
//! of solve/reuse measurements. Faer is explicitly run with `Par::Seq`.

use std::{hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::linalg::solvers::{Solve, SolveLstsq};
use faer::{Accum, Mat, Par, Side};
use nalgebra::{DMatrix, SMatrix};
use stack_algebra::Matrix;

const BATCH: usize = 8;

macro_rules! scalar_benchmarks {
    ($module:ident, $t:ty, $name:literal) => {
        #[allow(unused_must_use)]
        mod $module {
            use super::*;

            type Stack<const R: usize, const C: usize> = Matrix<R, C, $t>;
            type Nalgebra<const R: usize, const C: usize> = SMatrix<$t, R, C>;

            fn dense<const R: usize, const C: usize>() -> Stack<R, C> {
                Stack::from_fn(|r, c| (1 + r * C + 2 * c) as $t / 17.0)
            }

            fn system<const D: usize>() -> Stack<D, D> {
                Stack::from_fn(|r, c| {
                    if r == c {
                        (D + r + 2) as $t
                    } else {
                        (r + 2 * c + 1) as $t / 19.0
                    }
                })
            }

            fn spd<const D: usize>() -> Stack<D, D> {
                Stack::from_fn(|r, c| {
                    let mut value = 0.0 as $t;
                    for k in 0..D {
                        value += ((k + 3 * r + 1) as $t / 23.0) * ((k + 3 * c + 1) as $t / 23.0);
                    }
                    value + if r == c { D as $t } else { 0.0 }
                })
            }

            fn tall<const R: usize, const C: usize>() -> Stack<R, C> {
                Stack::from_fn(|r, c| {
                    if r == c {
                        (R + 1) as $t
                    } else {
                        (r + 2 * c + 1) as $t / 19.0
                    }
                })
            }

            fn rhs<const D: usize>() -> Stack<D, 1> {
                Stack::from_fn(|r, _| (r + 3) as $t / 11.0)
            }

            fn faer<const R: usize, const C: usize>() -> Mat<$t> {
                let a = dense::<R, C>();
                Mat::from_fn(R, C, |r, c| a[(r, c)])
            }

            fn faer_system<const D: usize>() -> Mat<$t> {
                let a = system::<D>();
                Mat::from_fn(D, D, |r, c| a[(r, c)])
            }

            fn faer_spd<const D: usize>() -> Mat<$t> {
                let a = spd::<D>();
                Mat::from_fn(D, D, |r, c| a[(r, c)])
            }

            fn faer_rhs<const D: usize>() -> Mat<$t> {
                let a = rhs::<D>();
                Mat::from_fn(D, 1, |r, c| a[(r, c)])
            }

            fn nalgebra<const R: usize, const C: usize>() -> Nalgebra<R, C> {
                Nalgebra::from_fn(|r, c| dense::<R, C>()[(r, c)])
            }

            fn nalgebra_tall<const R: usize, const C: usize>() -> DMatrix<$t> {
                DMatrix::from_fn(R, C, |r, c| tall::<R, C>()[(r, c)])
            }

            fn nalgebra_system<const D: usize>() -> DMatrix<$t> {
                DMatrix::from_fn(D, D, |r, c| system::<D>()[(r, c)])
            }

            fn nalgebra_spd<const D: usize>() -> DMatrix<$t> {
                DMatrix::from_fn(D, D, |r, c| spd::<D>()[(r, c)])
            }

            fn nalgebra_rhs<const D: usize>() -> DMatrix<$t> {
                DMatrix::from_fn(D, 1, |r, c| rhs::<D>()[(r, c)])
            }

            fn group<'a>(
                criterion: &'a mut Criterion,
                operation: &str,
            ) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
                criterion.benchmark_group(format!("comparison/{operation}/{}", $name))
            }

            fn matmul<const D: usize>(criterion: &mut Criterion) {
                let a = dense::<D, D>();
                let rhs_mat = dense::<D, D>();
                let na = nalgebra::<D, D>();
                let nb = nalgebra::<D, D>();
                let fa = faer::<D, D>();
                let fb = faer::<D, D>();
                let mut fo = Mat::zeros(D, D);
                let mut so = Stack::<D, D>::zeros();
                let mut g = group(criterion, "matmul");
                let shape = format!("{D}x{D}");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            black_box(&a).mul_into(black_box(&rhs_mat), black_box(&mut so));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-static", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            black_box(black_box(&na) * black_box(&nb));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            faer::linalg::matmul::matmul(
                                black_box(&mut fo),
                                Accum::Replace,
                                black_box(&fa),
                                black_box(&fb),
                                1.0,
                                Par::Seq,
                            );
                        }
                    })
                });
                g.finish();
            }

            fn matvec<const D: usize>(criterion: &mut Criterion) {
                let a = dense::<D, D>();
                let x = rhs::<D>();
                let na = nalgebra::<D, D>();
                let nx = nalgebra_rhs::<D>();
                let fa = faer::<D, D>();
                let fx = faer_rhs::<D>();
                let mut fo = Mat::zeros(D, 1);
                let mut so = Stack::<D, 1>::zeros();
                let mut g = group(criterion, "matvec");
                let shape = format!("{D}x{D}");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            a.matvec_into(black_box(&x), black_box(&mut so));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            black_box(black_box(&na) * black_box(&nx));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            faer::linalg::matmul::matmul(
                                black_box(&mut fo),
                                Accum::Replace,
                                black_box(&fa),
                                black_box(&fx),
                                1.0,
                                Par::Seq,
                            );
                        }
                    })
                });
                g.finish();
            }

            fn reductions<const D: usize>(criterion: &mut Criterion) {
                let a = dense::<D, 1>();
                let b = dense::<D, 1>();
                let na = nalgebra::<D, 1>();
                let nb = nalgebra::<D, 1>();
                let fa = faer::<1, D>();
                let fb = faer::<D, 1>();
                let mut out = 0.0 as $t;
                let mut g = group(criterion, "norm");
                g.bench_function(BenchmarkId::new("stack-algebra", D), |bch| {
                    bch.iter(|| {
                        for _ in 0..BATCH {
                            out = black_box(&a).norm();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-static", D), |bch| {
                    bch.iter(|| {
                        for _ in 0..BATCH {
                            out = black_box(&na).norm();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", D), |bch| {
                    bch.iter(|| {
                        for _ in 0..BATCH {
                            out = black_box(&fa).norm_l2();
                        }
                    })
                });
                g.finish();
                let mut g = group(criterion, "dot");
                g.bench_function(BenchmarkId::new("stack-algebra", D), |bch| {
                    bch.iter(|| {
                        for _ in 0..BATCH {
                            out = black_box(&a).dot(black_box(&b));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-static", D), |bch| {
                    bch.iter(|| {
                        for _ in 0..BATCH {
                            out = black_box(&na).dot(black_box(&nb));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", D), |bch| {
                    bch.iter(|| {
                        for _ in 0..BATCH {
                            out = faer::linalg::matmul::dot::inner_prod(
                                black_box(fa.row(0)),
                                faer::Conj::No,
                                black_box(fb.col(0)),
                                faer::Conj::No,
                            );
                        }
                    })
                });
                g.finish();
                black_box(out);
            }

            fn lu<const D: usize>(criterion: &mut Criterion) {
                let a = system::<D>();
                let na = nalgebra_system::<D>();
                let fa = faer_system::<D>();
                let x = rhs::<D>();
                let nx = nalgebra_rhs::<D>();
                let fx = faer_rhs::<D>();
                let mut sf = a.partial_piv_lu();
                let mut nf = na.clone().lu();
                let ff = fa.partial_piv_lu();
                let mut so = Stack::<D, 1>::zeros();
                let mut fo = Mat::zeros(D, 1);
                let shape = D.to_string();
                let mut g = group(criterion, "lu-factor");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            sf = a.partial_piv_lu();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            nf = na.clone().lu();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            black_box(&fa).partial_piv_lu();
                        }
                    })
                });
                g.finish();
                let mut g = group(criterion, "lu-solve");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            sf.solve_into(black_box(&x), black_box(&mut so));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            black_box(nf.solve(black_box(&nx)));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fo.copy_from(&fx);
                            ff.solve_in_place(black_box(&mut fo));
                        }
                    })
                });
                g.finish();
            }

            fn llt<const D: usize>(criterion: &mut Criterion) {
                let a = spd::<D>();
                let na = nalgebra_spd::<D>();
                let fa = faer_spd::<D>();
                let x = rhs::<D>();
                let nx = nalgebra_rhs::<D>();
                let fx = faer_rhs::<D>();
                let mut sf = a.cholesky().unwrap();
                let mut nf = na.clone().cholesky().unwrap();
                let ff = fa.llt(Side::Lower).unwrap();
                let mut so = Stack::<D, 1>::zeros();
                let mut fo = Mat::zeros(D, 1);
                let shape = D.to_string();
                let mut g = group(criterion, "llt-factor");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            sf = a.cholesky().unwrap();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            nf = na.clone().cholesky().unwrap();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fa.llt(Side::Lower);
                        }
                    })
                });
                g.finish();
                let mut g = group(criterion, "llt-solve");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            sf.solve_into(black_box(&x), black_box(&mut so));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            black_box(nf.solve(black_box(&nx)));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fo.copy_from(&fx);
                            ff.solve_in_place(black_box(&mut fo));
                        }
                    })
                });
                g.finish();
            }

            fn ldlt<const D: usize>(criterion: &mut Criterion) {
                let a = system::<D>();
                let na = nalgebra_system::<D>();
                let fa = faer_system::<D>();
                let x = rhs::<D>();
                let nx = nalgebra_rhs::<D>();
                let fx = faer_rhs::<D>();
                let mut sf = a.ldlt().expect("benchmark matrix is nonsingular");
                let mut nf = na.clone().lu();
                let ff = fa.ldlt(Side::Lower).unwrap();
                let mut so = Stack::<D, 1>::zeros();
                let mut fo = Mat::zeros(D, 1);
                let shape = D.to_string();
                let mut g = group(criterion, "ldlt-factor");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            sf = a.ldlt().expect("benchmark matrix is nonsingular");
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-lu-fallback", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            nf = na.clone().lu();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fa.ldlt(Side::Lower)
                                .expect("benchmark matrix is nonsingular");
                        }
                    })
                });
                g.finish();
                let mut g = group(criterion, "ldlt-solve");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            sf.solve_into(black_box(&x), black_box(&mut so));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-lu-fallback", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            black_box(nf.solve(black_box(&nx)));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fo.copy_from(&fx);
                            ff.solve_in_place(black_box(&mut fo));
                        }
                    })
                });
                g.finish();
            }

            fn qr<const D: usize>(criterion: &mut Criterion) {
                let a = system::<D>();
                let na = nalgebra_system::<D>();
                let fa = faer_system::<D>();
                let mut sf = a.householder_qr();
                let mut sn = na.clone().qr();
                let mut ff = fa.qr();
                let shape = D.to_string();
                let mut g = group(criterion, "qr-factor");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            sf = a.householder_qr();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            sn = na.clone().qr();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            ff = fa.qr();
                        }
                    })
                });
                g.finish();
                let mut g = group(criterion, "col-piv-qr-factor");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            a.col_piv_householder_qr();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            na.clone().col_piv_qr();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fa.col_piv_qr();
                        }
                    })
                });
                g.finish();
                black_box((&mut sf, &mut sn, &mut ff));
                let x = rhs::<D>();
                let nx = nalgebra_rhs::<D>();
                let fx = faer_rhs::<D>();
                let mut so = Stack::<D, 1>::zeros();
                let mut no = nx.clone();
                let mut fo = fx.clone();
                let mut g = group(criterion, "qr-solve");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            sf.solve_least_squares_into(black_box(&x), black_box(&mut so));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            no = sn.solve(black_box(&nx)).unwrap();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fo.copy_from(&fx);
                            ff.solve_lstsq_in_place(black_box(&mut fo));
                        }
                    })
                });
                g.finish();
                black_box((&mut so, &mut no, &mut fo));
            }

            fn triangular<const D: usize>(criterion: &mut Criterion) {
                let a = spd::<D>();
                let na = nalgebra_spd::<D>();
                let fa = faer_spd::<D>();
                let x = rhs::<D>();
                let nx = nalgebra_rhs::<D>();
                let fx = faer_rhs::<D>();
                let mut so = Stack::<D, 1>::zeros();
                let mut no = nx.clone();
                let mut fo = fx.clone();
                let shape = D.to_string();
                let mut g = group(criterion, "triangular-solve");
                let lower = a.lower_triangular();
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            lower.solve_into(black_box(&x), black_box(&mut so));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            no = na.solve_lower_triangular(black_box(&nx)).unwrap();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fo.copy_from(&fx);
                            faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                                black_box(fa.as_ref()),
                                black_box(fo.as_mut()),
                                Par::Seq,
                            );
                        }
                    })
                });
                g.finish();
                let upper = a.upper_triangular();
                let mut g = group(criterion, "triangular-upper-solve");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            upper.solve_into(black_box(&x), black_box(&mut so));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            no = na.solve_upper_triangular(black_box(&nx)).unwrap();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fo.copy_from(&fx);
                            faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                                black_box(fa.as_ref()),
                                black_box(fo.as_mut()),
                                Par::Seq,
                            );
                        }
                    })
                });
                g.finish();
                black_box((&mut so, &mut no, &mut fo));
            }

            fn tall_ops<const R: usize, const C: usize>(criterion: &mut Criterion) {
                let a = tall::<R, C>();
                let na = nalgebra_tall::<R, C>();
                let fa = {
                    let x = a;
                    Mat::from_fn(R, C, |r, c| x[(r, c)])
                };
                let mut ss = a.svd().unwrap();
                let mut ns = na.clone().svd(true, true);
                let mut fs = fa.thin_svd().unwrap();
                let shape = format!("{R}x{C}");
                let mut g = group(criterion, "svd-factor");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            ss = a.svd().unwrap();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            ns = na.clone().svd(true, true);
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fs = fa.thin_svd().unwrap();
                        }
                    })
                });
                g.finish();
                black_box((&mut ss, &mut ns, &mut fs));
                let x = Mat::from_fn(R, 1, |r, _| (r + 3) as $t / 11.0);
                let nx = DMatrix::from_fn(R, 1, |r, _| (r + 3) as $t / 11.0);
                let sx = Stack::<R, 1>::from_fn(|r, _| (r + 3) as $t / 11.0);
                let mut so = Stack::<C, 1>::zeros();
                let mut no = DMatrix::zeros(C, 1);
                let mut fo = x.clone();
                let mut g = group(criterion, "svd-solve");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            ss.solve_into(black_box(&sx), black_box(&mut so));
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            no = ns.solve(black_box(&nx), <$t>::default()).unwrap();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fo.copy_from(&x);
                            fs.solve_lstsq_in_place(black_box(&mut fo));
                        }
                    })
                });
                g.finish();
                black_box((&mut so, &mut no, &mut fo));
            }

            fn eigen<const D: usize>(criterion: &mut Criterion) {
                let a = spd::<D>();
                let na = nalgebra_spd::<D>();
                let fa = faer_spd::<D>();
                let mut se = a.self_adjoint_eigen().unwrap();
                let mut ne = na.clone().symmetric_eigen();
                let mut fe = fa.self_adjoint_eigen(Side::Lower).unwrap();
                let shape = D.to_string();
                let mut g = group(criterion, "self-adjoint-eigen-factor");
                g.bench_function(BenchmarkId::new("stack-algebra", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            se = a.self_adjoint_eigen().unwrap();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("nalgebra-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            ne = na.clone().symmetric_eigen();
                        }
                    })
                });
                g.bench_function(BenchmarkId::new("faer-dynamic", &shape), |b| {
                    b.iter(|| {
                        for _ in 0..BATCH {
                            fe = fa.self_adjoint_eigen(Side::Lower).unwrap();
                        }
                    })
                });
                g.finish();
                black_box((&mut se, &mut ne, &mut fe));
            }

            pub fn run(c: &mut Criterion) {
                for d in [3usize, 6, 8, 15, 16, 32] {
                    match d {
                        3 => {
                            matmul::<3>(c);
                            matvec::<3>(c);
                            reductions::<3>(c);
                            lu::<3>(c);
                            llt::<3>(c);
                            ldlt::<3>(c);
                            qr::<3>(c);
                            triangular::<3>(c);
                            eigen::<3>(c);
                            tall_ops::<6, 3>(c);
                        }
                        6 => {
                            matmul::<6>(c);
                            matvec::<6>(c);
                            reductions::<6>(c);
                            lu::<6>(c);
                            llt::<6>(c);
                            ldlt::<6>(c);
                            qr::<6>(c);
                            triangular::<6>(c);
                            eigen::<6>(c);
                            tall_ops::<12, 6>(c);
                        }
                        8 => {
                            matmul::<8>(c);
                            matvec::<8>(c);
                            reductions::<8>(c);
                            lu::<8>(c);
                            llt::<8>(c);
                            ldlt::<8>(c);
                            qr::<8>(c);
                            triangular::<8>(c);
                            eigen::<8>(c);
                            tall_ops::<16, 8>(c);
                        }
                        16 => {
                            matmul::<16>(c);
                            matvec::<16>(c);
                            reductions::<16>(c);
                            lu::<16>(c);
                            llt::<16>(c);
                            ldlt::<16>(c);
                            qr::<16>(c);
                            triangular::<16>(c);
                            eigen::<16>(c);
                            tall_ops::<32, 16>(c);
                        }
                        15 => {
                            matmul::<15>(c);
                            matvec::<15>(c);
                            reductions::<15>(c);
                            lu::<15>(c);
                            llt::<15>(c);
                            ldlt::<15>(c);
                            qr::<15>(c);
                            triangular::<15>(c);
                            eigen::<15>(c);
                            tall_ops::<30, 15>(c);
                        }
                        32 => {
                            matmul::<32>(c);
                            matvec::<32>(c);
                            reductions::<32>(c);
                            lu::<32>(c);
                            llt::<32>(c);
                            ldlt::<32>(c);
                            qr::<32>(c);
                            triangular::<32>(c);
                            eigen::<32>(c);
                            tall_ops::<64, 32>(c);
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
    };
}

scalar_benchmarks!(f32_benchmarks, f32, "f32");
scalar_benchmarks!(f64_benchmarks, f64, "f64");

fn run(c: &mut Criterion) {
    f32_benchmarks::run(c);
    f64_benchmarks::run(c);
}

fn criterion_config() -> Criterion {
    if std::env::var_os("STACK_ALGEBRA_BENCH_FAST").is_some() {
        Criterion::default()
            .warm_up_time(Duration::from_millis(20))
            .measurement_time(Duration::from_millis(20))
            .sample_size(10)
    } else {
        Criterion::default()
            .warm_up_time(Duration::from_millis(200))
            .measurement_time(Duration::from_millis(300))
            .sample_size(20)
    }
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = run
}
criterion_main!(benches);
