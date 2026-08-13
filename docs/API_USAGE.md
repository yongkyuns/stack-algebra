# API Usage Guide

This guide shows the intended call patterns. For the complete inventory and
limitations, see [`FEATURES.md`](FEATURES.md).

## 1. Choose storage

| Problem | Start with |
| --- | --- |
| Known dimensions (pose, IMU state, small Jacobian) | `Matrix<M, N, T>` |
| Runtime-active dimensions with a compile-time bound | `MatrixBuf<MAX_ROWS, MAX_COLS, T>` |
| Contiguous external column-major buffer | `Map` / `MapMut` |
| Padded or row-major external buffer | `StridedMap` / `StridedMapMut` |
| Fixed submatrix of an owned matrix | `Block` / `BlockMut` |
| Scalar sparse structure | `StaticCscPattern` + `StaticCscMatrix` |
| Block sparse structure | `StaticBlockCscMatrix` or `StaticBlockCsrMatrix` |

There is no hidden heap allocation or runtime matrix-size dispatch in the
fixed-size API.

## 2. Choose precision

Use the scalar parameter explicitly. Implicit `f32`/`f64` operations do not
compile; cast at a deliberate boundary:

```rust
use stack_algebra::{matrix, Matrix};

let state_f32: Matrix<3, 1, f32> = matrix![1.0; 2.0; 3.0];
let state_f64: Matrix<3, 1, f64> = state_f32.cast();
let covariance: Matrix<3, 3, f64> = Matrix::eye();
```

## 3. Stable magnitude reductions

Use `norm()` for a Frobenius magnitude when inputs may be very large or very
small. It scales the accumulation to avoid intermediate overflow/underflow;
`squared_norm()` intentionally exposes the raw sum-of-squares operation:

```rust
use stack_algebra::matrix;

let values = matrix![1.0e308_f64, 1.0e308];
assert!(values.norm().is_finite());
assert!(values.squared_norm().is_infinite());
```

## 4. Products and output reuse

Use operators for short expressions and `mul_into` in loops:

```rust
use stack_algebra::{matrix, Matrix};

let a = matrix![1.0_f32, 2.0; 3.0, 4.0];
let b = Matrix::<2, 2, f32>::eye();
let mut product = Matrix::<2, 2, f32>::zeros();
a.mul_into(&b, &mut product);
let state = a * matrix![1.0_f32; 2.0];
```

Use `mul_into`, `matvec_view_into`, and decomposition `*_into` methods when
output ownership and scratch reuse should be explicit.

## 5. External buffers and views

`Map` consumes contiguous column-major storage:

```rust
use stack_algebra::{HouseholderQr, Map};

let raw = [1.0_f64, 3.0, 5.0, 2.0, 4.0, 7.0];
let mapped = Map::<3, 2, f64>::from_slice(&raw).expect("six values");
let qr = HouseholderQr::try_decompose_view(&mapped).expect("QR succeeds");
```

Mapping errors are typed: match `ViewError::BufferTooShort`,
`ViewError::SizeOverflow`, or `ViewError::ZeroStride` when input validation
must be reported rather than handled with `expect`.

`StridedMap` uses `inner_stride` for one row and `outer_stride` for one column:

```rust
use stack_algebra::{Svd, StridedMap};

let raw = [1.0_f64, 2.0, 0.0, 3.0, 4.0, 0.0];
let mapped = StridedMap::<2, 2, f64>::from_slice(&raw, 1, 3)
    .expect("padded columns fit");
let svd = Svd::try_decompose_view(&mapped).expect("SVD succeeds");
```

`Block` exposes a fixed-size submatrix without copying:

```rust
use stack_algebra::{Cholesky, Matrix};

let storage = Matrix::<6, 6, f64>::eye();
let block = storage.block::<3, 3>(1, 2).expect("block is in bounds");
let factor = Cholesky::try_decompose_view(&block);
```

All dense decomposition view paths own only factor/workspace storage. Use
`Matrix::from_view` when an owned snapshot is intentionally required.

## 6. Dense solves and decompositions

Cholesky is for symmetric positive-definite systems:

```rust
use stack_algebra::{matrix, Cholesky};

let a = matrix![
    4.0_f64, 1.0, 1.0;
    1.0, 3.0, 0.0;
    1.0, 0.0, 2.0;
];
let rhs = matrix![1.0_f64; 2.0; 3.0];
let factor = Cholesky::try_decompose(&a).expect("SPD input");
let x = factor.solve(&rhs);
```

Use `SelfAdjointView`, `SelfAdjointLower`, or `SelfAdjointUpper` when only one
triangle stores a symmetric matrix. The view supplies the missing mirrored
entries to reductions and decompositions without building a second matrix.

`PartialPivLu` is the general square-system factorization. Use the checked
constructor when non-finite inputs or scalar-range overflow should be reported
instead of retained in the factor:

```rust
use stack_algebra::matrix;

let a = matrix![3.0_f64, 1.0; 1.0, 2.0];
let factor = a.try_partial_piv_lu().expect("finite LU factorization");
let x = factor.solve(&matrix![5.0_f64; 5.0]);
```

`Ldlt` is the diagonal-pivoted LDLT factorization for symmetric indefinite or
weakly ordered input:

```rust
use stack_algebra::matrix;

let a = matrix![0.0_f64, 2.0; 2.0, 3.0];
let factor = a.try_ldlt().expect("nonsingular symmetric input");
let x = factor.solve(&matrix![1.0_f64; 4.0]);
```

Dense LDLT uses bounded Bunch–Kaufman scalar pivots. Inspect
`factor.pivot_blocks()` when a KKT or contact system may need a 2×2 pivot:
`1` denotes a 1×1 block, while `2` and `3` denote the first and second entries
of a 2×2 block. `factor.diagonal_matrix()` reconstructs the full block-
diagonal `D` for diagnostics or verification.

Use `try_ldlt_no_pivot` only when pivot stability is already known.

`HouseholderQr` is the default for a known full-rank least-squares system:

```rust
use stack_algebra::matrix;

let design = matrix![
    1.0_f64, 1.0;
    1.0, 2.0;
    1.0, 3.0;
    1.0, 4.0;
];
let observations = matrix![3.0_f64; 5.0; 7.0; 9.0];
let coefficients = design
    .householder_qr()
    .solve_least_squares(&observations)
    .expect("full-rank design");
```

Use `try_householder_qr()` when non-finite input or scalar-range overflow must
be reported at factorization time:

```rust
let checked = design
    .try_householder_qr()
    .expect("finite QR factorization");
let coefficients = checked
    .try_solve_least_squares(&observations)
    .expect("full-rank design");
```

Use `ColPivHouseholderQr` when rank detection matters, and `Svd` when a
pseudoinverse or robust rank threshold is required:

```rust
let pivoted = design.col_piv_householder_qr();
let rank = pivoted.rank();
let basic_solution = pivoted.solve_least_squares_basic(&observations);

let svd = design.svd().expect("SVD converges");
let robust_solution = svd.solve(&observations);
let pinv = svd.pseudo_inverse();
```

`SelfAdjointEigen` returns sorted eigenvalues and orthonormal eigenvectors:

```rust
let eig = a.self_adjoint_eigen().expect("symmetric input");
let values = eig.eigenvalues();
let vectors = eig.eigenvectors();
let reconstructed = eig.reconstruct();
```

`LowerTriangular` and `UpperTriangular` provide borrowed triangular views for
specialized solves and products when the unused half of a matrix is not part
of the input contract.

## 7. Reuse factors and workspaces

Construct a factor once, then recompute it in a control or estimation loop:

```rust
let mut factor = matrix.cholesky().expect("initial SPD input");
for next_matrix in matrices {
    factor.try_compute(&next_matrix).expect("SPD input");
    factor.solve_in_place(&mut rhs);
}
```

Use `try_compute_view` for mapped or blocked inputs. For eigendecomposition,
make scratch ownership explicit:

```rust
use stack_algebra::SelfAdjointEigenWorkspace;

let mut workspace = SelfAdjointEigenWorkspace::<3, f64>::new();
let mut eigen = matrix.self_adjoint_eigen().expect("symmetric input");
eigen
    .try_compute_with_workspace(&next_matrix, &mut workspace)
    .expect("symmetric input");
```

Use `solve_into` or `solve_least_squares_into` when output is reused. Use
`solve_in_place` when the right-hand side may be overwritten.

## 8. Bounded runtime dimensions

`MatrixBuf` reserves a compile-time maximum while tracking active dimensions:

```rust
use stack_algebra::MatrixBuf;

let mut buffer = MatrixBuf::<16, 16, f32>::new(6, 6).expect("within capacity");
buffer[(0, 0)] = 1.0;
buffer.resize(8, 4).expect("within capacity");
const BYTES: usize = MatrixBuf::<16, 16, f32>::storage_bytes();
let view = buffer.as_view::<8, 4>().expect("active dimensions match");
```

`MatrixBuf` remains bounded storage rather than a dynamic decomposition
interface. When active dimensions match compile-time dimensions, pass
`view` to any `try_decompose_view` API without copying; use `to_matrix` when an
owned `Matrix<M, N, T>` is needed. Its construction, resize, conversion, and
fixed-view operations return `MatrixBufError` with capacity, length, or shape
details when they cannot satisfy the request.

`MatrixBufView` and `MatrixBufViewMut` are the corresponding fixed-size
read-only and mutable view types. Use `row` and `column` when an algorithm
needs a one-dimensional `Row` or `Column` view instead of an owned matrix.

## 9. Scalar sparse CSC

Build a canonical pattern once, update values repeatedly, and reuse the factor:

```rust
use stack_algebra::{Matrix, StaticCscCholesky, StaticCscMatrix};

type Sparse = StaticCscMatrix<3, 3, 5, f64>;
let mut a = Sparse::from_pattern(
    &[4.0, 1.0, 3.0, 1.0, 2.0],
    &[0, 1, 1, 2, 2],
    &[0, 2, 4, 5],
).expect("canonical CSC");
let mut factor = StaticCscCholesky::<3, 8, f64>::decompose(&a)
    .expect("SPD sparse input");
let rhs = Matrix::<3, 1, f64>::from_columns([[1.0, 2.0, 3.0]]);
let x = factor.solve(&rhs);

// Reuse a validated entry position when assembling the same pattern.
let entry = a.pattern().entry_index(1, 0).expect("stored entry");
let values = a.values_mut();
values[entry] += 0.5;
factor.recompute(&a).expect("same sparsity pattern");
```

Use `StaticCscCholeskyPattern::analyze` when symbolic analysis is shared by
multiple factors. Use `prepare_ordered` plus `recompute_ordered` when the same
permutation and coordinate transform are reused.
For fill-heavy patterns, use
`StaticCscCholeskyPattern::analyze_with_minimum_degree` to select the
deterministic fixed-workspace ordering during analysis; the block CSC pattern
provides the same method.

`StaticCscLdltPattern` is the matching symbolic alias when the numeric phase
uses sparse LDLT rather than Cholesky.

When a sparse system has a zero or poorly scaled leading diagonal, use
`StaticCscLdlt::decompose_with_diagonal_pivoting(&a, threshold)` to select a
bounded symmetric diagonal permutation. The threshold is an absolute pivot
cutoff; non-finite thresholds are rejected. This mode still reports
`ZeroPivot` when a global 2×2 pivot is required.

Sparse LDLT currently uses no-pivot or analysis-time 1x1 diagonal pivoting.
For a single ergonomic entry point, use
`try_ldlt_with_dense_fallback::<MAX_L_NNZ>`; it returns native sparse LDLT when
possible, tries bounded sparse diagonal pivoting for a zero leading pivot, and
automatically selects the dense Bunch–Kaufman path only when a global 2x2 pivot
is required:

The unified overload uses a scale-relative default threshold (`epsilon` times
the largest stored absolute value). Use
`try_ldlt_with_dense_fallback_threshold::<MAX_L_NNZ>(threshold)` when a
different numerical policy is required.

```rust
use stack_algebra::{Matrix, StaticCscMatrix};

type Sparse = StaticCscMatrix<2, 2, 3, f64>;
let a = Sparse::from_pattern(
    &[0.0, 1.0, 2.0],
    &[0, 1, 1],
    &[0, 2, 3],
).expect("canonical CSC");
let rhs = Matrix::<2, 1, f64>::from_columns([[3.0, 4.0]]);
let factor = a
    .try_ldlt_with_dense_fallback::<3>()
    .expect("nonsingular sparse system");
let solution = factor.solve(&rhs);
assert!(factor.uses_dense_fallback());
```

The fallback expands only the stored lower triangle into fixed-size stack
storage. Use `try_dense_ldlt` directly when the dense fallback is intentional;
otherwise the unified factor keeps the normal sparse path allocation-free. For
reused sparsity patterns, call `recompute_with_dense_fallback` to refactor new
values; it preserves the sparse factor when possible and transitions to dense
storage only when required. Use `solve_in_place` when the right-hand side can
be overwritten to avoid an additional fixed-size output copy.

## 10. Block sparse storage

Use block CSC/CSR for explicit block matvec. Native block Cholesky is available
for square blocks and grids; the scalar-expansion method remains a useful
reference path:

```rust
use stack_algebra::{Matrix, StaticBlockCscMatrix};

type Blocks = StaticBlockCscMatrix<2, 2, 2, 2, 4, f64>;
let blocks = Blocks::from_pattern(
    &[
        Matrix::from_rows([[4.0, 1.0], [1.0, 3.0]]),
        Matrix::from_rows([[1.0, 0.0], [0.0, 0.5]]),
        Matrix::from_rows([[1.0, 0.0], [0.0, 0.5]]),
        Matrix::from_rows([[3.0, 0.0], [0.0, 2.0]]),
    ],
    &[0, 1, 0, 1],
    &[0, 2, 4],
).expect("canonical block CSC");
let factor = blocks.cholesky::<4, 16, 16>().expect("SPD block input");
let rhs = Matrix::<4, 1, f64>::from_columns([[1.0, 2.0, 3.0, 4.0]]);
let native = stack_algebra::StaticBlockCscCholesky::<2, 2, 2, 2, 4, f64>::decompose(&blocks)
    .expect("SPD block input");
let step = native.try_solve::<4, 1>(&rhs).expect("matching scalar dimension");
```

The scalar-adapter capacities are scalar dimension, scalar input capacity, and
scalar factor capacity. Native block factor capacity is measured in blocks.
For indefinite systems, use `StaticBlockCscLdlt::decompose` (or
`decompose_with_ordering`) and the same `try_solve` methods; its diagonal
blocks use local Bunch–Kaufman compact `L·D` storage. If a leading block is singular,
`decompose_with_diagonal_pivoting(matrix, threshold)` performs bounded
analysis-time block permutations before local Bunch–Kaufman factorization.
Inspect `local_pivot_blocks()` and `local_permutations()` when diagnosing a
KKT block.

When a pivot must cross a block boundary, use the explicit fixed-size dense
fallback:

```rust
let factor = blocks.try_dense_ldlt::<4>().expect("nonsingular block system");
let solution = factor.solve(&rhs);
```

This path uses stack storage sized by `SCALAR_DIM`; native block LDLT remains
the performance path when pivots stay within blocks.

## 11. Build and validate

```sh
cargo test --all-features
cargo check --no-default-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTFLAGS="-C target-cpu=native" cargo bench --bench fixed_size --bench small_fixed
RUSTFLAGS="-C target-cpu=native" cargo bench --bench sparse
RUSTFLAGS="-C target-cpu=native" cargo bench --bench dense_solvers
qemu-tests/run_cortex_m.sh
qemu-tests/run_riscv32.sh
qemu-tests/run_aarch64.sh
```

Use `storage_bytes()` and the QEMU checks before moving a fixed-size algorithm
onto an MCU. QEMU validates software behavior and kernel paths, not peripheral
timing or hardware floating-point throughput.
