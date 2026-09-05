# Capabilities and limits

This page is the compact reference for what `stack-algebra` currently supports.
For workflow-oriented guidance, start with [Choosing an API](api-usage.md) or
[Common use cases](use-cases.md).

## Core model

`stack-algebra` is built around fixed or bounded storage rather than a general
heap-backed dynamic matrix.

| Property | Current behavior |
| --- | --- |
| Dense dimensions | `Matrix<M, N, T>` uses compile-time rows and columns |
| Bounded dimensions | `MatrixBuf<MAX_ROWS, MAX_COLS, T>` tracks an active shape inside a fixed capacity |
| Layout | Dense matrices are column-major |
| Allocation | Core fixed/bounded operations do not require heap allocation |
| Scalar types | `f32` and `f64` are the primary numerical types; basic integer matrix operations are also available |
| Mixed precision | Explicit; use `.cast::<U>()` at the boundary |
| `no_std` | Supported by the default core |
| External storage | Borrowed through contiguous, strided, and block views |
| Reuse | Factors, workspaces, output buffers, and in-place solves are available where meaningful |
| Sparse storage | Fixed-capacity scalar and block CSC/CSR representations |

Inline storage describes representation, not required placement. A matrix can
be local, static, embedded in a long-lived state object, or backed by external
memory through a view.

## Dense matrices and views

The main dense types are:

- `Matrix<M, N, T>` and `Vector<M, T>`;
- common aliases such as `Matrix2f`, `Matrix3f`, `Matrix4f`, `Matrix2d`,
  `Matrix3d`, `Matrix4d`, `Vector3f`, and `Vector3d`;
- construction macros `matrix!`, `vector!`, `eye!`, `zeros!`, `ones!`, and
  `diag!`.

Core operations include indexing, addition/subtraction, scalar arithmetic,
matrix multiplication, matrix-vector multiplication, transpose, trace,
determinant, inverse, dot products, norms, and reusable-output multiplication.

Borrowed views include:

| Type | Use |
| --- | --- |
| `Map` / `MapMut` | Contiguous external column-major buffers |
| `StridedMap` / `StridedMapMut` | Padded, row-major, or custom-stride buffers |
| `Block` / `BlockMut` | Fixed-size submatrices |
| `Row` / `Column` | One-dimensional borrowed regions |
| `MatrixBufView` / `MatrixBufViewMut` | Fixed-shape view of an active bounded region |

Dense decompositions can consume compatible views directly, so a mapped or
blocked input does not need to be copied into a second owning `Matrix` first.

## Dense decompositions

| Type | Typical use | Notes |
| --- | --- | --- |
| `Cholesky` | Symmetric positive-definite solves | Lower-triangle input; reusable factor and in-place solve support |
| `Ldlt` | Symmetric indefinite solves | Bounded 1x1/2x2 diagonal pivot blocks |
| `PartialPivLu` | General square systems | Partial row pivoting |
| `HouseholderQr` | Full-rank square/tall least squares | Q application and reusable solve outputs |
| `ColPivHouseholderQr` | Rank-aware least squares | Column pivoting and rank reporting |
| `Svd` | Rank-deficient / ill-conditioned systems | Singular values, rank, solve, and pseudoinverse |
| `SelfAdjointEigen` | Symmetric eigenproblems | Sorted eigenvalues and orthonormal eigenvectors |

The factor types expose checked construction/recomputation paths when numerical
failure needs to be reported. Reusable-output and in-place methods are available
where they fit the operation.

`LowerTriangular` and `UpperTriangular` provide triangular views and solves.
`SelfAdjointLower`, `SelfAdjointUpper`, and `SelfAdjointView` let one triangle be
authoritative for a symmetric matrix without materializing the mirrored half.

## Geometry

The geometry module provides fixed-size 3D types:

- `Quaternion<T>`;
- `AngleAxis<T>`;
- `RotationMatrix<T>`;
- `Isometry<T>` for rigid transforms;
- `AffineTransform<T>` for general affine transforms.

These types use the same explicit scalar/storage model as the dense matrix API
and do not require a robotics framework.

## Bounded and sparse storage

### Scalar CSC

The scalar sparse path separates symbolic structure from numeric values:

- `StaticCscPattern<ROWS, COLS, MAX_NNZ>` — validated bounded CSC structure;
- `StaticCscMatrix<ROWS, COLS, MAX_NNZ, T>` — numeric values for a pattern;
- `StaticCscOrdering<N>` — fixed-size ordering/permutation information;
- `StaticCscCholeskyPattern` + `StaticCscCholesky` — reusable sparse LLT;
- `StaticCscLdltPattern` + `StaticCscLdlt` — reusable sparse LDLT;
- `StaticCscLdltFactor` — sparse-first LDLT factor that can represent the
  explicit fixed-size dense fallback when a global 2x2 pivot is required.

Symbolic analysis can be retained while numeric values change. A deterministic
minimum-degree ordering is available for patterns where reducing fill is useful.

### Block sparse

The block-sparse path keeps repeated dense blocks explicit:

- `StaticBlockCscMatrix` — fixed-capacity block CSC;
- `StaticBlockCsrMatrix` — fixed-capacity block CSR;
- `StaticBlockCscCholeskyPattern` + `StaticBlockCscCholesky` — reusable native
  block Cholesky;
- `StaticBlockCscLdltPattern` + `StaticBlockCscLdlt` — reusable native block
  LDLT with local dense pivot metadata.

Block storage is intended for problems that already have meaningful fixed-size
blocks. Scalar CSC remains the simpler representation when the structure is
naturally scalar.

## Numerical and storage behavior

A few behaviors are worth knowing before choosing the crate:

- `norm()` uses a scale-stable Frobenius reduction for extreme finite values;
  `squared_norm()` is the direct sum of squares and may overflow or underflow.
- Checked factorization APIs report numerical failure rather than silently
  changing the problem assumptions.
- Fixed-capacity storage types expose `storage_bytes()` so long-lived RAM costs
  can be estimated at compile time.
- Dense and sparse reuse APIs separate one-time structure/setup work from the
  numeric work that changes each iteration.

## Platforms and kernels

The portable scalar path is the baseline implementation and supports `no_std`
targets. Optimized x86-64 and AArch64 kernels are selected at compile time where
available; unsupported targets use the portable path.

Cross-compilation and QEMU smoke tests cover several embedded target classes,
but those checks are not substitutes for board-specific timing or stack
measurement. See [Platforms and embedded use](targets.md) for the current test
matrix.

## Where the current scope stops

The current core does **not** provide:

- a general heap-backed dynamic matrix whose dimensions can grow freely at
  runtime;
- runtime-sized dense decomposition APIs independent of a compile-time bound;
- implicit mixed-precision expressions;
- automatic framework/code-generator integration;
- globally pivoted native block-sparse LDLT across arbitrary block boundaries
  (an explicit fixed-size dense fallback is available when required).

These boundaries define the intended operating model. Workloads centered on
fixed, bounded, mapped, or fixed-capacity sparse data fit that model directly;
large unbounded runtime dimensions and freely growing storage do not.
