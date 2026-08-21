use core::sync::atomic::{AtomicUsize, Ordering};

use stack_algebra::{
    matrix, FactorizationScalar, Map, Matrix, MatrixScalar, ReductionScalar, StridedMap, Vector,
    Zero,
};

#[test]
fn fused_matrix_updates_match_equivalent_expressions() {
    let x = matrix![1.0_f64, 2.0; 3.0, 4.0];
    let y = matrix![5.0_f64, 6.0; 7.0, 8.0];

    let mut in_place = y;
    in_place.axpy_in_place(2.0, &x);
    assert_eq!(in_place, matrix![7.0, 10.0; 13.0, 16.0]);

    let mut axpy = Matrix::<2, 2, f64>::zeros();
    x.axpy_into(2.0, &y, &mut axpy);
    assert_eq!(axpy, 2.0 * x + y);

    let mut combination = Matrix::<2, 2, f64>::zeros();
    x.linear_combination_into(2.0, &y, -1.0, &mut combination);
    assert_eq!(combination, 2.0 * x - y);
}

static MATMUL_DISPATCHES: AtomicUsize = AtomicUsize::new(0);
static MATVEC_DISPATCHES: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Probe(i32);

impl core::ops::Add for Probe {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl core::ops::Mul for Probe {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Zero for Probe {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl FactorizationScalar for Probe {}

impl MatrixScalar for Probe {
    fn matmul<const M: usize, const N: usize, const P: usize>(
        lhs: &Matrix<M, N, Self>,
        rhs: &Matrix<N, P, Self>,
        output: &mut Matrix<M, P, Self>,
    ) {
        MATMUL_DISPATCHES.fetch_add(1, Ordering::SeqCst);
        for column in 0..P {
            for row in 0..M {
                let mut value = Self::zero();
                for shared in 0..N {
                    value = value + lhs[(row, shared)] * rhs[(shared, column)];
                }
                output[(row, column)] = value;
            }
        }
    }
}

impl ReductionScalar for Probe {
    fn matvec<const M: usize, const N: usize>(
        matrix: &Matrix<M, N, Self>,
        vector: &Vector<N, Self>,
        output: &mut Vector<M, Self>,
    ) {
        MATVEC_DISPATCHES.fetch_add(1, Ordering::SeqCst);
        for row in 0..M {
            let mut value = Self::zero();
            for column in 0..N {
                value = value + matrix[(row, column)] * vector[column];
            }
            output[row] = value;
        }
    }
}

#[test]
fn mapped_exact_column_major_layouts_reuse_scalar_kernel_dispatch() {
    MATMUL_DISPATCHES.store(0, Ordering::SeqCst);
    MATVEC_DISPATCHES.store(0, Ordering::SeqCst);

    let lhs_storage = [Probe(1), Probe(3), Probe(2), Probe(4)];
    let rhs_storage = [Probe(5), Probe(7), Probe(6), Probe(8)];
    let lhs = Map::<2, 2, _>::from_slice(&lhs_storage).unwrap();
    let rhs = Map::<2, 2, _>::from_slice(&rhs_storage).unwrap();
    let mut product = Matrix::<2, 2, Probe>::zeros();
    lhs.mul_into(&rhs, &mut product);
    assert_eq!(
        product,
        Matrix::from_rows([[Probe(19), Probe(22)], [Probe(43), Probe(50)]])
    );
    assert_eq!(MATMUL_DISPATCHES.load(Ordering::SeqCst), 1);

    let vector = Vector::<2, Probe>::from_rows([[Probe(5)], [Probe(6)]]);
    let mut output = Vector::<2, Probe>::zeros();
    lhs.matvec_into(&vector, &mut output);
    assert_eq!(output, Vector::from_rows([[Probe(17)], [Probe(39)]]));
    assert_eq!(MATVEC_DISPATCHES.load(Ordering::SeqCst), 1);

    let lhs = StridedMap::<2, 2, _>::from_slice(&lhs_storage, 1, 2).unwrap();
    let rhs = StridedMap::<2, 2, _>::from_slice(&rhs_storage, 1, 2).unwrap();
    lhs.mul_into(&rhs, &mut product);
    lhs.matvec_into(&vector, &mut output);
    assert_eq!(MATMUL_DISPATCHES.load(Ordering::SeqCst), 2);
    assert_eq!(MATVEC_DISPATCHES.load(Ordering::SeqCst), 2);
}

#[test]
fn arbitrary_strides_keep_zero_copy_scalar_fallback() {
    MATMUL_DISPATCHES.store(0, Ordering::SeqCst);
    MATVEC_DISPATCHES.store(0, Ordering::SeqCst);

    let lhs_storage = [Probe(1), Probe(2), Probe(3), Probe(4)];
    let rhs_storage = [Probe(5), Probe(6), Probe(7), Probe(8)];
    let lhs = StridedMap::<2, 2, _>::from_slice(&lhs_storage, 2, 1).unwrap();
    let rhs = StridedMap::<2, 2, _>::from_slice(&rhs_storage, 2, 1).unwrap();
    let mut product = Matrix::<2, 2, Probe>::zeros();
    lhs.mul_into(&rhs, &mut product);
    assert_eq!(
        product,
        Matrix::from_rows([[Probe(19), Probe(22)], [Probe(43), Probe(50)]])
    );
    assert_eq!(MATMUL_DISPATCHES.load(Ordering::SeqCst), 0);

    let vector = Vector::<2, Probe>::from_rows([[Probe(5)], [Probe(6)]]);
    let mut output = Vector::<2, Probe>::zeros();
    lhs.matvec_into(&vector, &mut output);
    assert_eq!(output, Vector::from_rows([[Probe(17)], [Probe(39)]]));
    assert_eq!(MATVEC_DISPATCHES.load(Ordering::SeqCst), 0);
}
