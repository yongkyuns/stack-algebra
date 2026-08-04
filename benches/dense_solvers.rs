use std::hint::black_box;

#[cfg(feature = "eigen-compare")]
use std::{ffi::c_void, marker::PhantomData};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use stack_algebra::Matrix;

#[cfg(feature = "eigen-compare")]
unsafe extern "C" {
    fn sa_eigen_dense_ldlt_create_f32(input: *const f32, dimension: usize) -> *mut c_void;
    fn sa_eigen_dense_ldlt_factorize_f32(context: *mut c_void) -> i32;
    fn sa_eigen_dense_ldlt_solve_f32(
        context: *mut c_void,
        rhs: *const f32,
        columns: usize,
        output: *mut f32,
    ) -> i32;
    fn sa_eigen_dense_ldlt_destroy_f32(context: *mut c_void);
    fn sa_eigen_dense_ldlt_create_f64(input: *const f64, dimension: usize) -> *mut c_void;
    fn sa_eigen_dense_ldlt_factorize_f64(context: *mut c_void) -> i32;
    fn sa_eigen_dense_ldlt_solve_f64(
        context: *mut c_void,
        rhs: *const f64,
        columns: usize,
        output: *mut f64,
    ) -> i32;
    fn sa_eigen_dense_ldlt_destroy_f64(context: *mut c_void);
}

#[cfg(feature = "eigen-compare")]
trait EigenDenseLdltScalar: Copy {
    unsafe fn create(input: *const Self, dimension: usize) -> *mut c_void;
    unsafe fn factorize(context: *mut c_void) -> i32;
    unsafe fn solve(
        context: *mut c_void,
        rhs: *const Self,
        columns: usize,
        output: *mut Self,
    ) -> i32;
    unsafe fn destroy(context: *mut c_void);
}

#[cfg(feature = "eigen-compare")]
macro_rules! impl_eigen_dense_ldlt_scalar {
    ($scalar:ty, $create:ident, $factorize:ident, $solve:ident, $destroy:ident) => {
        impl EigenDenseLdltScalar for $scalar {
            unsafe fn create(input: *const Self, dimension: usize) -> *mut c_void {
                $create(input, dimension)
            }

            unsafe fn factorize(context: *mut c_void) -> i32 {
                $factorize(context)
            }

            unsafe fn solve(
                context: *mut c_void,
                rhs: *const Self,
                columns: usize,
                output: *mut Self,
            ) -> i32 {
                $solve(context, rhs, columns, output)
            }

            unsafe fn destroy(context: *mut c_void) {
                $destroy(context)
            }
        }
    };
}

#[cfg(feature = "eigen-compare")]
impl_eigen_dense_ldlt_scalar!(
    f32,
    sa_eigen_dense_ldlt_create_f32,
    sa_eigen_dense_ldlt_factorize_f32,
    sa_eigen_dense_ldlt_solve_f32,
    sa_eigen_dense_ldlt_destroy_f32
);
#[cfg(feature = "eigen-compare")]
impl_eigen_dense_ldlt_scalar!(
    f64,
    sa_eigen_dense_ldlt_create_f64,
    sa_eigen_dense_ldlt_factorize_f64,
    sa_eigen_dense_ldlt_solve_f64,
    sa_eigen_dense_ldlt_destroy_f64
);

#[cfg(feature = "eigen-compare")]
struct EigenDenseLdlt<T: EigenDenseLdltScalar> {
    context: *mut c_void,
    _scalar: PhantomData<T>,
}

#[cfg(feature = "eigen-compare")]
impl<T: EigenDenseLdltScalar> EigenDenseLdlt<T> {
    fn new<const D: usize>(matrix: &Matrix<D, D, T>) -> Self {
        let context = unsafe { T::create(matrix.as_slice().as_ptr(), D) };
        assert!(
            !context.is_null(),
            "Eigen dense LDLT context allocation failed"
        );
        Self {
            context,
            _scalar: PhantomData,
        }
    }

    fn factorize(&mut self) {
        assert_eq!(unsafe { T::factorize(self.context) }, 1);
    }

    fn solve<const D: usize>(&self, rhs: &Matrix<D, 1, T>, output: &mut Matrix<D, 1, T>) {
        assert_eq!(
            unsafe {
                T::solve(
                    self.context,
                    rhs.as_slice().as_ptr(),
                    1,
                    output.as_mut_slice().as_mut_ptr(),
                )
            },
            1
        );
    }
}

#[cfg(feature = "eigen-compare")]
impl<T: EigenDenseLdltScalar> Drop for EigenDenseLdlt<T> {
    fn drop(&mut self) {
        unsafe { T::destroy(self.context) }
    }
}

fn matrix_f64<const D: usize>() -> Matrix<D, D, f64> {
    Matrix::from_fn(|row, column| {
        if row == column {
            (D + row + 1) as f64
        } else {
            1.0 / (row + column + 2) as f64
        }
    })
}

fn matrix_f32<const D: usize>() -> Matrix<D, D, f32> {
    Matrix::from_fn(|row, column| {
        if row == column {
            (D + row + 1) as f32
        } else {
            1.0 / (row + column + 2) as f32
        }
    })
}

fn rhs_f64<const D: usize>() -> Matrix<D, 1, f64> {
    Matrix::from_fn(|row, _| (row + 1) as f64 / 3.0)
}

fn rhs_f32<const D: usize>() -> Matrix<D, 1, f32> {
    Matrix::from_fn(|row, _| (row + 1) as f32 / 3.0)
}

fn faer_matrix_f64<const D: usize>() -> Mat<f64> {
    let matrix = matrix_f64::<D>();
    Mat::from_fn(D, D, |row, column| matrix[(row, column)])
}

fn faer_matrix_f32<const D: usize>() -> Mat<f32> {
    let matrix = matrix_f32::<D>();
    Mat::from_fn(D, D, |row, column| matrix[(row, column)])
}

fn bench_f64<const D: usize>(criterion: &mut Criterion) {
    let matrix = matrix_f64::<D>();
    let rhs = rhs_f64::<D>();
    let faer_matrix = faer_matrix_f64::<D>();
    let faer_rhs = Mat::from_fn(D, 1, |row, _| rhs[(row, 0)]);
    let mut stack_factor = matrix.ldlt().unwrap();
    let mut stack_no_pivot_factor = matrix.ldlt_no_pivot().unwrap();
    let faer_factor = faer_matrix.ldlt(Side::Lower).unwrap();
    let mut stack_output = Matrix::<D, 1, f64>::zeros();
    let mut faer_output = Mat::zeros(D, 1);
    #[cfg(feature = "eigen-compare")]
    let mut eigen_factor = EigenDenseLdlt::new(&matrix);
    #[cfg(feature = "eigen-compare")]
    let mut eigen_output = Matrix::<D, 1, f64>::zeros();
    let mut group = criterion.benchmark_group("dense-ldlt/f64");

    group.bench_with_input(BenchmarkId::new("stack-factor-solve", D), &D, |bench, _| {
        bench.iter(|| {
            let factor = black_box(&matrix).ldlt().unwrap();
            factor.solve_into(black_box(&rhs), black_box(&mut stack_output));
            black_box(&stack_output);
        });
    });
    group.bench_with_input(
        BenchmarkId::new("stack-factorize-reuse", D),
        &D,
        |bench, _| {
            bench.iter(|| {
                stack_factor.try_compute(black_box(&matrix)).unwrap();
                black_box(&stack_factor);
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("stack-no-pivot-factorize-reuse", D),
        &D,
        |bench, _| {
            bench.iter(|| {
                stack_no_pivot_factor
                    .try_compute_no_pivot(black_box(&matrix))
                    .unwrap();
                black_box(&stack_no_pivot_factor);
            });
        },
    );
    group.bench_with_input(BenchmarkId::new("faer-factor-solve", D), &D, |bench, _| {
        bench.iter(|| {
            let factor = black_box(&faer_matrix).ldlt(Side::Lower).unwrap();
            faer_output.copy_from(&faer_rhs);
            factor.solve_in_place(black_box(&mut faer_output));
            black_box(&faer_output);
        });
    });
    #[cfg(feature = "eigen-compare")]
    group.bench_with_input(
        BenchmarkId::new("eigen-factorize-reuse", D),
        &D,
        |bench, _| {
            bench.iter(|| {
                eigen_factor.factorize();
                black_box(&eigen_factor);
            });
        },
    );
    #[cfg(feature = "eigen-compare")]
    group.bench_with_input(BenchmarkId::new("eigen-solve-reuse", D), &D, |bench, _| {
        bench.iter(|| {
            eigen_factor.solve(black_box(&rhs), black_box(&mut eigen_output));
            black_box(&eigen_output);
        });
    });
    group.bench_with_input(BenchmarkId::new("stack-solve-reuse", D), &D, |bench, _| {
        bench.iter(|| {
            stack_factor.solve_into(black_box(&rhs), black_box(&mut stack_output));
            black_box(&stack_output);
        });
    });
    group.bench_with_input(BenchmarkId::new("faer-solve-reuse", D), &D, |bench, _| {
        bench.iter(|| {
            faer_output.copy_from(&faer_rhs);
            faer_factor.solve_in_place(black_box(&mut faer_output));
            black_box(&faer_output);
        });
    });
    group.finish();
}

fn bench_f32<const D: usize>(criterion: &mut Criterion) {
    let matrix = matrix_f32::<D>();
    let rhs = rhs_f32::<D>();
    let faer_matrix = faer_matrix_f32::<D>();
    let faer_rhs = Mat::from_fn(D, 1, |row, _| rhs[(row, 0)]);
    let mut stack_factor = matrix.ldlt().unwrap();
    let mut stack_no_pivot_factor = matrix.ldlt_no_pivot().unwrap();
    let faer_factor = faer_matrix.ldlt(Side::Lower).unwrap();
    let mut stack_output = Matrix::<D, 1, f32>::zeros();
    let mut faer_output = Mat::zeros(D, 1);
    #[cfg(feature = "eigen-compare")]
    let mut eigen_factor = EigenDenseLdlt::new(&matrix);
    #[cfg(feature = "eigen-compare")]
    let mut eigen_output = Matrix::<D, 1, f32>::zeros();
    let mut group = criterion.benchmark_group("dense-ldlt/f32");

    group.bench_with_input(BenchmarkId::new("stack-factor-solve", D), &D, |bench, _| {
        bench.iter(|| {
            let factor = black_box(&matrix).ldlt().unwrap();
            factor.solve_into(black_box(&rhs), black_box(&mut stack_output));
            black_box(&stack_output);
        });
    });
    group.bench_with_input(
        BenchmarkId::new("stack-factorize-reuse", D),
        &D,
        |bench, _| {
            bench.iter(|| {
                stack_factor.try_compute(black_box(&matrix)).unwrap();
                black_box(&stack_factor);
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("stack-no-pivot-factorize-reuse", D),
        &D,
        |bench, _| {
            bench.iter(|| {
                stack_no_pivot_factor
                    .try_compute_no_pivot(black_box(&matrix))
                    .unwrap();
                black_box(&stack_no_pivot_factor);
            });
        },
    );
    group.bench_with_input(BenchmarkId::new("faer-factor-solve", D), &D, |bench, _| {
        bench.iter(|| {
            let factor = black_box(&faer_matrix).ldlt(Side::Lower).unwrap();
            faer_output.copy_from(&faer_rhs);
            factor.solve_in_place(black_box(&mut faer_output));
            black_box(&faer_output);
        });
    });
    #[cfg(feature = "eigen-compare")]
    group.bench_with_input(
        BenchmarkId::new("eigen-factorize-reuse", D),
        &D,
        |bench, _| {
            bench.iter(|| {
                eigen_factor.factorize();
                black_box(&eigen_factor);
            });
        },
    );
    #[cfg(feature = "eigen-compare")]
    group.bench_with_input(BenchmarkId::new("eigen-solve-reuse", D), &D, |bench, _| {
        bench.iter(|| {
            eigen_factor.solve(black_box(&rhs), black_box(&mut eigen_output));
            black_box(&eigen_output);
        });
    });
    group.bench_with_input(BenchmarkId::new("stack-solve-reuse", D), &D, |bench, _| {
        bench.iter(|| {
            stack_factor.solve_into(black_box(&rhs), black_box(&mut stack_output));
            black_box(&stack_output);
        });
    });
    group.bench_with_input(BenchmarkId::new("faer-solve-reuse", D), &D, |bench, _| {
        bench.iter(|| {
            faer_output.copy_from(&faer_rhs);
            faer_factor.solve_in_place(black_box(&mut faer_output));
            black_box(&faer_output);
        });
    });
    group.finish();
}

fn bench_all(criterion: &mut Criterion) {
    for dimension in [8, 16, 32] {
        match dimension {
            8 => {
                bench_f32::<8>(criterion);
                bench_f64::<8>(criterion);
            }
            16 => {
                bench_f32::<16>(criterion);
                bench_f64::<16>(criterion);
            }
            32 => {
                bench_f32::<32>(criterion);
                bench_f64::<32>(criterion);
            }
            _ => unreachable!(),
        }
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
