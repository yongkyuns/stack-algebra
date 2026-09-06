# Choosing the right API

`stack-algebra` has several storage and solver types, but most applications can
make the right choice from two questions:

1. **Where does the data live, and are its dimensions fixed or bounded?**
2. **What mathematical structure does the problem have?**

This page is a decision guide rather than an exhaustive API listing. Once you
have chosen a type, use the [generated API reference](api-reference.md) for
method signatures and trait bounds.

## 1. Choose the storage model

Start with the simplest representation that matches ownership and layout.

| Data shape / ownership | Start with | Why |
| --- | --- | --- |
| Fixed dimensions, owned values | `Matrix<M, N, T>` | Smallest and most direct API |
| Runtime-active dimensions with a compile-time maximum | `MatrixBuf<MAX_R, MAX_C, T>` | Bounded storage without a heap |
| Contiguous external column-major storage | `Map` / `MapMut` | Borrow existing data without copying |
| Padded, row-major, or custom-stride storage | `StridedMap` / `StridedMapMut` | Borrow non-contiguous layouts |
| Fixed submatrix of another matrix | `Block` / `BlockMut` | Zero-copy access to a region |
| Scalar sparse structure | `StaticCscPattern` + `StaticCscMatrix` | Reuse a bounded sparsity pattern |
| Repeated dense blocks in a sparse system | `StaticBlockCscMatrix` / `StaticBlockCsrMatrix` | Preserve block structure explicitly |

The fixed-size path does not hide a heap allocation or runtime matrix-size
dispatch. A value can still be placed wherever the owner chooses: a local, a
long-lived state object, static memory, or an external buffer exposed through a
view.

### Bounded runtime dimensions

`MatrixBuf` is useful when a maximum size is known but the active region changes
at runtime:

```rust
use stack_algebra::MatrixBuf;

let mut work = MatrixBuf::<16, 16, f32>::new(6, 6).expect("within capacity");
work.resize(8, 4).expect("within capacity");

let view = work.as_view::<8, 4>().expect("active shape matches");
```

`MatrixBufView` and `MatrixBufViewMut` are the corresponding fixed-shape borrowed
views. They let algorithms consume the active region without converting it to a
second owned matrix.

### External buffers and submatrices

Use `Map` when the data is already contiguous and column-major:

```rust
use stack_algebra::{HouseholderQr, Map};

let raw = [1.0_f64, 3.0, 5.0, 2.0, 4.0, 7.0];
let a = Map::<3, 2, f64>::from_slice(&raw).expect("six values");
let factor = HouseholderQr::try_decompose_view(&a).expect("finite input");
```

Use `StridedMap` for padding, row-major layout, interleaved data, or another
known stride. `MapMut` and `StridedMapMut` provide mutable access when the
borrowed storage may be changed.

Use `Block` or `BlockMut` when a fixed region inside a larger matrix should be
processed in place. `Row` and `Column` are the corresponding one-dimensional
borrowed views.

These view types are especially useful at FFI, generated-code, DMA, and
workspace boundaries because the library does not require the source data to be
repacked first.

## 2. Choose the factorization from the problem

A factorization should follow the matrix assumptions you already have.

| Problem | Type | Main reason to choose it |
| --- | --- | --- |
| Symmetric positive definite | `Cholesky` | Efficient solve with the strongest useful assumption |
| Symmetric, possibly indefinite | `Ldlt` | Symmetric factorization with diagonal pivoting |
| General square system | `PartialPivLu` | General-purpose direct solve |
| Full-rank square/tall least squares | `HouseholderQr` | Stable least-squares path without normal equations |
| Least squares where rank detection matters | `ColPivHouseholderQr` | Column pivoting exposes numerical rank |
| Rank-deficient / ill-conditioned problem | `Svd` | Singular values, pseudoinverse, and rank-aware solve |
| Symmetric eigenproblem | `SelfAdjointEigen` | Eigenvalues and orthonormal eigenvectors |

### Positive-definite systems

Use `Cholesky` when the SPD contract is part of the problem:

```rust
use stack_algebra::{matrix, Cholesky};

let a = matrix![
    4.0_f64, 1.0, 1.0;
    1.0,     3.0, 0.0;
    1.0,     0.0, 2.0;
];
let b = matrix![1.0_f64; 2.0; 3.0];

let factor = Cholesky::try_decompose(&a).expect("SPD matrix");
let x = factor.solve(&b);
```

If only one triangle is authoritative, `SelfAdjointView` can mirror it without
constructing another matrix. `LowerTriangular` and `UpperTriangular` similarly
provide borrowed triangular solve/product views when the unused half is not
part of the input.

### Symmetric indefinite systems

Use `Ldlt` for symmetric systems that are not guaranteed positive definite,
such as some KKT or contact systems:

```rust
use stack_algebra::matrix;

let a = matrix![0.0_f64, 2.0; 2.0, 3.0];
let factor = a.try_ldlt().expect("nonsingular symmetric input");
let x = factor.solve(&matrix![1.0_f64; 4.0]);
```

The dense factor uses bounded 1x1/2x2 diagonal pivots. If your algorithm already
knows that pivoting is unnecessary, the no-pivot variant avoids that search;
otherwise prefer the checked pivoted path.

### General square systems

`PartialPivLu` is the straightforward choice when no symmetry or positive-
definite structure can be relied on:

```rust
use stack_algebra::matrix;

let a = matrix![3.0_f64, 1.0; 1.0, 2.0];
let factor = a.try_partial_piv_lu().expect("finite nonsingular input");
let x = factor.solve(&matrix![5.0_f64; 5.0]);
```

### Least squares and rank

Use `HouseholderQr` for an expected full-rank least-squares problem:

```rust
use stack_algebra::matrix;

let a = matrix![
    1.0_f64, 1.0;
    1.0,     2.0;
    1.0,     3.0;
    1.0,     4.0;
];
let b = matrix![3.0_f64; 5.0; 7.0; 9.0];

let x = a
    .householder_qr()
    .solve_least_squares(&b)
    .expect("full-rank design");
```

Choose `ColPivHouseholderQr` when rank loss should be detected as part of the
normal workflow. Choose `Svd` when you need singular values, a pseudoinverse, or
more explicit behavior for rank-deficient and ill-conditioned problems.

`SelfAdjointEigen` is for symmetric eigenvalue problems rather than linear
solves; it returns sorted eigenvalues and orthonormal eigenvectors.

## 3. Reuse work in repeated loops

Many robotics, control, estimation, and simulation loops solve the same-shaped
problem many times. The useful optimization is usually to retain ownership of
factor/output storage rather than repeatedly rebuilding temporary objects.

```rust
let mut factor = matrix.cholesky().expect("initial SPD input");

for next_matrix in matrices {
    factor.try_compute(&next_matrix).expect("SPD input");
    factor.solve_in_place(&mut rhs);
}
```

The dense factor types expose recomputation and reusable-output variants such as
`solve_into`, `solve_least_squares_into`, or `solve_in_place` where appropriate.
Use `try_compute_view` when the next input is a `Map`, `StridedMap`, or `Block`.

The goal is not to avoid every copy at all costs; it is to make ownership and
scratch use explicit when a loop actually benefits from reuse.

## 4. Choose precision deliberately

`f32` and `f64` are separate matrix types. There is no implicit mixed-precision
expression:

```rust
use stack_algebra::{matrix, Matrix};

let state_f32: Matrix<3, 1, f32> = matrix![1.0; 2.0; 3.0];
let state_f64: Matrix<3, 1, f64> = state_f32.cast();
```

For embedded systems, `f32` is often the natural starting point because of
memory and hardware throughput. Use `f64` when conditioning, accumulated error,
or the target's floating-point capabilities justify it. Measure the complete
application rather than choosing precision from a microbenchmark alone.

For magnitude calculations over extreme floating-point ranges, `norm()` uses a
scale-stable reduction. `squared_norm()` intentionally exposes the raw sum of
squares and can overflow or underflow when the inputs are extreme.

## 5. Sparse systems: separate structure from values

Sparse workloads benefit when the symbolic structure changes less often than
the numeric values.

### Scalar CSC

`StaticCscPattern` validates and owns the bounded sparsity structure, while
`StaticCscMatrix` owns the numeric values. For SPD systems,
`StaticCscCholeskyPattern` can retain symbolic analysis and
`StaticCscCholesky` can be recomputed as values change.

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

let entry = a.pattern().entry_index(1, 0).expect("stored entry");
a.values_mut()[entry] += 0.5;
factor.recompute(&a).expect("same sparsity pattern");
```

Use `StaticCscLdltPattern` and `StaticCscLdlt` for symmetric sparse systems that
need LDLT rather than LLT. The sparse LDLT path supports bounded diagonal
pivoting and an explicit fixed-size dense fallback for systems that require a
global 2x2 pivot. Those details matter only when your matrix structure actually
needs them; the normal starting point is still to pick LLT vs LDLT from the
mathematics.

### Block sparse

Use `StaticBlockCscMatrix` or `StaticBlockCsrMatrix` when repeated dense blocks
are the natural structure, such as pose/state blocks in a fixed graph. Native
block Cholesky and `StaticBlockCscLdlt` retain bounded block-level factor
storage and reusable solves.

Block storage is not automatically better than scalar CSC. Prefer it when your
problem already has meaningful fixed-size blocks and you want to preserve that
structure through assembly, matvec, or factorization.

## 6. Handle failure as part of the numerical contract

The `try_*` constructors report invalid or unsupported numerical conditions
through typed errors instead of silently producing a factor. Examples include
non-finite input, singular systems, failed convergence, or a matrix that does
not satisfy a required positive-definite contract.

Use unchecked/convenience paths only when the surrounding algorithm already
establishes the preconditions. In application code, it is often clearer to
propagate a decomposition error than to substitute a different solver without
understanding why the original assumption failed.

## 7. Keep the boundary simple

A useful rule of thumb is:

- own a `Matrix` when your subsystem owns the values;
- borrow with a view when another subsystem owns the storage;
- use `MatrixBuf` when dimensions vary inside a known bound;
- preserve sparse structure when symbolic reuse is meaningful;
- choose the factorization from matrix assumptions;
- introduce reuse only where repeated execution makes it valuable.

For examples organized around complete workloads rather than API types, see
[Common use cases](use-cases.md).