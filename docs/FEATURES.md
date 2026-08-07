# Feature Set

`stack-algebra` is a standalone, fixed-size Rust linear-algebra library for
native and embedded robotics workloads. It does not depend on SymForce, Eigen,
or faer at runtime. Eigen and faer are used only as external comparison
references in optional tests and benchmarks.

## Design contract

| Property | Current behavior |
| --- | --- |
| Allocation | Core operations use fixed-size inline storage and are allocation-free. Placement follows the owner (local, static, embedded field, arena, or caller-managed buffer). |
| Dimensions | `Matrix<M, N, T>` dimensions are compile-time constants. `MatrixBuf` adds bounded runtime-active dimensions. |
| Layout | Matrices and vectors use column-major storage. |
| Scalar types | `f32` and `f64` are the primary numerical types; integer scalar matrices support basic algebra. |
| Mixed precision | Implicit `f32`/`f64` operations are not supported; use `.cast::<U>()`. |
| Platform | `no_std`; portable scalar kernels plus compile-time-selected x86 and AArch64 packet kernels. |
| Failure model | Fallible decompositions return `Result<_, DecompositionError>`; convenience constructors return `Option<_>` where available. |
| Runtime dispatch | No runtime ISA dispatch is used in the embedded path. |

Every bounded storage type exposes `storage_bytes()` as a `const fn`, so an
application can budget RAM at compile time.

Heap-backed dynamic owning matrices are intentionally outside the current core
scope. They will only be considered for a concrete workload that cannot use
fixed-size, bounded, mapped, or sparse storage, and would be isolated behind an
optional `alloc`/`std` layer.

## Dense core

### Matrix and vector types

- `Matrix<M, N, T>`: fixed-size, column-major matrix.
- `Vector<M, T>`: alias for `Matrix<M, 1, T>`.
- `Matrix2f`, `Matrix3f`, `Matrix4f`: common `f32` aliases.
- `Matrix2d`, `Matrix3d`, `Matrix4d`: common `f64` aliases.
- `Vector3f`, `Vector3d`: common 3D vector aliases.
- Constructors/macros: `matrix!`, `vector!`, `eye!`, `zeros!`, `ones!`, `diag!`.
- Core operations: indexing, addition/subtraction, scalar arithmetic,
  multiplication, transpose, trace, determinant, inverse, norm, dot, matvec,
  and caller-provided `mul_into`. `norm()` uses a scale-stable Frobenius
  reduction; `squared_norm()` is the raw sum of squares and may overflow for
  extreme inputs.

The `T` parameter is explicit in the type. This is the normal way to choose
precision:

```rust
use stack_algebra::{matrix, Matrix};

let state_f32: Matrix<3, 1, f32> = matrix![1.0; 2.0; 3.0];
let state_f64: Matrix<3, 1, f64> = state_f32.cast();
```

### Views

All dense decomposition view APIs consume `MatrixRead<M, N, T>`:

| View | Purpose | Ownership |
| --- | --- | --- |
| `Map` / `MapMut` | Contiguous external column-major storage | Borrowed |
| `StridedMap` / `StridedMapMut` | Padded, row-major, or custom-stride storage | Borrowed |
| `Block` / `BlockMut` | Fixed-size submatrix of a `Matrix` | Borrowed |
| `Matrix` | Owning fixed-size matrix | Owned |

`MatrixBuf` has checked runtime dimensions. When the active dimensions are
known at a call site, `as_view::<M, N>()` and `as_view_mut::<M, N>()` expose the
active region through the same zero-copy `MatrixRead`/`MatrixWrite` view APIs;
use `to_matrix` only when an owned snapshot is required.

`Map`, `StridedMap`, and `MatrixBuf` constructors return `ViewError` or
`MatrixBufError`, including the required and available sizes or the mismatched
shape. View arithmetic uses the const-generic shape and writes directly to its
caller-provided output.

View-based factorization fills factor-owned storage directly. It does not first
create a second owning input matrix. Use `Matrix::from_view` when an owned
snapshot is explicitly desired.

## Dense decompositions

| Type | Input shape | Main constructor | Recompute | Solve / outputs | Notes |
| --- | --- | --- | --- | --- | --- |
| `PartialPivLu<D, T>` | `D x D` | `decompose` / `matrix.partial_piv_lu()`; checked: `try_decompose` / `matrix.try_partial_piv_lu()` | `compute`; checked: `try_compute` | `solve`, `solve_into`, `inverse`, `determinant` | Partial row pivoting; checked path rejects non-finite input/intermediates. |
| `Cholesky<D, T>` | SPD `D x D` | `try_decompose` / `matrix.cholesky()` | `try_compute` | `solve`, `solve_into`, `solve_in_place`, `inverse` | Reads the lower triangle; rejects non-SPD input. |
| `Ldlt<D, T>` | Symmetric `D x D` | `try_decompose` / `matrix.ldlt()` | `try_compute` | `solve`, `solve_into`, `solve_in_place`, `inverse` | Eigen-compatible Bunch–Kaufman 1x1/2x2 scalar pivoting; `pivot_blocks` exposes the compact block layout. |
| `HouseholderQr<M, N, T>` | `M x N` | `decompose` / `matrix.householder_qr()`; checked: `try_decompose` / `matrix.try_householder_qr()` | `compute`; checked: `try_compute` | Full-rank least squares, `apply_q`, `apply_q_transpose` | Best for full-rank square/tall systems. |
| `ColPivHouseholderQr<M, N, T>` | `M x N` | `decompose` / `matrix.col_piv_householder_qr()`; checked: `try_decompose` / `matrix.try_col_piv_householder_qr()` | `compute`; checked: `try_compute` | Rank, basic/rank-aware least squares, Q application | Column pivoting improves rank detection. |
| `Svd<M, N, T>` | Any fixed shape | `try_decompose` / `matrix.svd()` | `try_compute` | `solve`, rank, `pseudo_inverse`, `u`, `v`, singular values | Thin SVD; wide matrices use zero padding. |
| `SelfAdjointEigen<D, T>` | Symmetric `D x D` | `try_decompose` / `matrix.self_adjoint_eigen()` | `try_compute` | Eigenvalues, eigenvectors, reconstruction | Sorted eigenvalues and orthonormal eigenvectors. |

Every dense decomposition also exposes `try_decompose_view` and
`try_compute_view`, including `Map`, `StridedMap`, and `Block` inputs. Self-
adjoint eigendecomposition additionally supports caller-owned
`SelfAdjointEigenWorkspace`.

## Triangular and self-adjoint views

- `LowerTriangular` and `UpperTriangular` provide solve and multiplication
  operations without copying the triangular matrix.
- `SelfAdjointLower` and `SelfAdjointUpper` mirror one authoritative triangle
  without reading the other triangle.
- `try_self_adjoint_lower(tolerance)` and `try_self_adjoint_upper(tolerance)`
  validate scaled symmetry before creating a view.

## Sparse storage and factorization

### Scalar sparse CSC

- `StaticCscPattern<ROWS, COLS, MAX_NNZ>`: reusable validated symbolic CSC
  structure.
- `StaticCscMatrix<ROWS, COLS, MAX_NNZ, T>`: fixed-capacity numeric CSC values.
- `StaticCscOrdering<N>`: identity or deterministic minimum-degree ordering.
- `StaticCscCholeskyPattern` + `StaticCscCholesky`: reusable sparse LLT with
  bounded symbolic fill.
- `StaticCscLdlt`: sparse LDLT with no-pivot and analysis-time 1x1 diagonal
  pivoting. Matrices requiring 2x2 pivots return `ZeroPivot`.
- `StaticCscMatrix::try_ldlt_with_dense_fallback::<MAX_L_NNZ>`: unified factor
  that keeps native sparse LDLT when possible, tries bounded sparse diagonal
  pivoting for zero leading pivots, and dispatches to the explicit dense
  Bunch–Kaufman fallback only when a global 2x2 pivot is required.
- `StaticCscMatrix::try_ldlt_with_dense_fallback_threshold::<MAX_L_NNZ>`:
  same unified path with an explicit absolute pivot threshold.
- `StaticCscLdltFactor::recompute_with_dense_fallback`: reusable numeric update
  that transitions from sparse to dense storage only when needed.
- `StaticCscLdltFactor::solve_in_place`: allocation-free solve that overwrites
  the right-hand side for parity with dense factor APIs.
- `StaticCscMatrix::try_dense_ldlt`: explicit fixed-size dense fallback for
  callers that want to force dense Bunch–Kaufman factorization.

Sparse factor objects separate symbolic analysis from numeric refactorization:
call `recompute` for natural CSC coordinates and `recompute_ordered` after a
single `prepare_ordered` step.

### Block sparse

- `StaticBlockCscMatrix`: fixed-capacity block CSC storage and block matvec.
- `StaticBlockCsrMatrix`: fixed-capacity block CSR storage and block matvec.
- `StaticBlockCscMatrix::to_scalar_csc`: no-allocation expansion into bounded
  scalar CSC storage.
- `StaticBlockCscMatrix::try_dense_ldlt::<SCALAR_DIM>`: fixed-size dense
  Bunch–Kaufman fallback for scalar pivots that cross block boundaries.
- `StaticBlockCscCholeskyPattern` + `StaticBlockCscCholesky`: native dense-block
  Cholesky with symbolic block fill, reusable factors, and multi-RHS solves.
- `StaticBlockCscLdltPattern` + `StaticBlockCscLdlt`: native dense-block LDLᵀ
  with local Bunch–Kaufman 1x1/2x2 `L·D` diagonal blocks, local pivot metadata,
  block ordering, reusable factors, and multi-RHS solves.
- `StaticBlockCscMatrix::cholesky`: scalar-expanded sparse Cholesky reference
  adapter with explicit scalar input and factor capacities.

Native block Cholesky and LDLᵀ require square block and grid dimensions at
runtime. Native LDLᵀ retains local scalar permutations and 1x1/2x2
Bunch–Kaufman pivot metadata for each dense diagonal block. Use
`try_dense_ldlt` when a global scalar pivot is required.

## Geometry

- `Quaternion<T>` and `RotationMatrix<T>` for validated 3D rotations.
- `Isometry<T>` for rigid transforms and homogeneous conversion.
- `AngleAxis<T>` and quaternion spherical interpolation.
- `AffineTransform<T>` for validated homogeneous affine transforms.

These types use the same fixed-size matrix/vector storage and have no robotics
framework dependency.

## Kernel and target support

- Portable scalar multiplication/reduction kernels are the correctness
  reference and work on bare-metal targets.
- x86-64 uses compile-time SSE2/AVX2 packet implementations where available.
- AArch64 with NEON uses the ARM packet implementation.
- Unsupported targets use the scalar implementation.
- QEMU harnesses cover Cortex-M4, RISC-V/ESP32-C-class scalar paths, and
  AArch64/NEON smoke behavior.

Use `RUSTFLAGS="-C target-cpu=native"` only when deploying to a matching CPU
feature set. See [`API_USAGE.md`](API_USAGE.md) for build commands and
[`targets.md`](targets.md) for the evidence matrix. See
[`ROADMAP.md`](ROADMAP.md) for planned native block kernels and remaining
numerical extensions.

## Deliberate non-features

- No implicit mixed-precision arithmetic.
- No heap-backed dynamic matrix in the core `no_std` API.
- No runtime-sized decomposition API; use `MatrixBuf` as bounded storage and
  convert active regions explicitly.
- No SymForce/code-generation integration.
- Native block sparse LDLT does not yet support scalar pivots crossing block
  boundaries; block-level ordering remains the boundary between dense blocks.
