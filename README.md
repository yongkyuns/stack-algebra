# stack-algebra

[![Crates.io Version](https://img.shields.io/crates/v/stack-algebra.svg)](https://crates.io/crates/stack-algebra)
[![Docs.rs Latest](https://img.shields.io/badge/docs.rs-latest-blue.svg)](https://docs.rs/stack-algebra)
![Build Status](https://github.com/yongkyuns/stack-algebra/actions/workflows/ci.yml/badge.svg?branch=main)

A fixed-size, `no_std` linear algebra library with inline storage.

## Overview
This crate provides fixed-size matrices, vectors, views, factorizations, and
bounded sparse structures for Rust programs that benefit from compile-time
dimensions and predictable storage. It supports host applications as well as
`no_std` and embedded targets.

The design provides:
1. Compile-time matrix dimensions and scalar types (`f32` or `f64`)
2. Inline storage with no required heap allocation
3. Fixed-capacity bounded matrices and sparse structures
4. Dense, geometric, and sparse linear-algebra operations

The implementation roadmap and release gates are tracked in
[`docs/roadmap.md`](docs/roadmap.md).

For a consolidated capability matrix, see
[`docs/features.md`](docs/features.md). For copy-ready API patterns, see
[`docs/api-usage.md`](docs/api-usage.md); storage and algorithm recipes are in
[`docs/use-cases.md`](docs/use-cases.md).

The crate provides the matrix abstractions and algebra routines needed to build
numerical algorithms while keeping dimensions and storage explicit. It is based on
[`vectrix`][vectrix] for core implementations.

## Install
Use cargo to add to your project (or add manually to your `Cargo.toml`)
```sh
cargo add stack-algebra
```
Then import to your module by using
```rust
use stack_algebra::*; // or import just the items you need
```

## Usage

### Fixed-size types

`Matrix<R, C, T>` stores its data inline in column-major order. `T` defaults to
`f32`; use `.cast::<f64>()` or an explicitly typed matrix when changing precision.

```rust
use stack_algebra::{Matrix3d, Vector3f};

let rotation: Matrix3d = Matrix3d::eye();
let vector: Vector3f = Vector3f::from_columns([[1.0, 2.0, 3.0]]);
let vector64 = vector.cast::<f64>();
```

- `matrix!` macro can be used to create a new matrix
  ```rust
  // 2-by-3 matrix 
  let m = matrix![
      1.0, 2.0, 3.0;
      4.0, 5.0, 6.0; // Semicolon here is optional
  ]; 
  ```

- `vector!` macro can be used to create a row/column vector
  ```rust
  // 1-by-3 row vector
  let r = vector![1.0, 2.0, 3.0]; 

  // 3-by-1 column vector
  let c = vector![1.0; 2.0; 3.0]; 

  // Vector to tuple conversion (for 3 or 4 element vectors)
  let (x, y, z) = r.into();
  ```

- `eye!` for creating square identity matrix
  ```rust
  let m = eye!(2); 
  let exp = matrix![
    1.0, 0.0;
	0.0, 1.0
  ];
  assert_eq!(m, exp);
  ```

- `zeros!` for creating zero-valued matrix
  ```rust
  let m = zeros!(2); // Square 2-by-2 matrix
  let exp = matrix![
    0.0, 0.0;
	0.0, 0.0
  ];
  assert_eq!(m, exp);

  let m = zeros!(2,3); // 2-by-3 matrix
  let exp = matrix![
    0.0, 0.0, 0.0;
	0.0, 0.0, 0.0
  ];
  assert_eq!(m, exp);
  ```

- `ones!` for creating matrix with 1.0s (same as `zeros!` for usage)

- `diag!` for creating a diagonal matrix with given entries (up to 6-by-6 size)
  ```rust
  let m = diag!(1.0, 2.0, 3.0);
  let exp = matrix![
    1.0, 0.0, 0.0;
	0.0, 2.0, 0.0;
	0.0, 0.0, 3.0
  ];
  assert_eq!(m, exp);
  ```

- `[i]` or `[(r,c)]` to access individual elements
  ```rust
  let m = matrix![
      1.0, 2.0, 3.0;
      4.0, 5.0, 6.0
  ]; 

  assert_eq!(m[1], 4.0); // Using a single index assumes column-major order
  assert_eq!(m[(1,2)], 6.0);
  ```

- `*`, `/`, `+`, `-` for matrix arithmetic
  ```rust
  let m = matrix![
      1.0, 2.0;
	  3.0, 4.0
  ];

  let exp = matrix![
      2.0, 4.0;
	  6.0, 8.0
  ];

  assert_eq!(m + m, exp); // Add matrices
  
  let exp = matrix![
      2.0, 3.0;
	  4.0, 5.0
  ];

  assert_eq!(m + 1.0, exp); // Add scalar to matrix (note scalar has to be behind the operator)
  ```

- `.transpose()` for matrix transpose
  ```rust
  let m = matrix![
      1.0, 2.0;
	  3.0, 4.0
  ];

  let exp = matrix![
      1.0, 3.0;
	  2.0, 4.0
  ];

  assert_eq!(m.transpose(), exp);
  ```

- `.norm()` for computing the [`Frobenius norm`][frobenius]. The reduction is
  scale-stable for finite extreme values; use `.squared_norm()` when the raw
  sum-of-squares semantics are desired.
  ```rust
	let m = matrix![
	  1.0,-2.0;
	 -3.0, 6.0;
	];
	assert_relative_eq!(m.norm(), 7.0710678, max_relative = 1e-6);
  ```

- `.squared_norm()`, `.dot()`, and `.matvec()` for fixed-size reductions and
  matrix-vector products
  ```rust
  let lhs = vector![1.0; 2.0; 3.0];
  let rhs = vector![4.0; 5.0; 6.0];
  assert_eq!(lhs.dot(&rhs), 32.0);
  assert_eq!(lhs.squared_norm(), 14.0);

  let matrix = matrix![
      1.0, 2.0, 3.0;
      4.0, 5.0, 6.0;
  ];
  assert_eq!(matrix.matvec(&lhs), vector![14.0; 32.0]);
  ```

- `.trace()` for the sum of diagonal elements of a square matrix
  ```rust
	let m = matrix![
	  9.0, 8.0, 7.0;
	  6.0, 5.0, 4.0;
	  3.0, 2.0, 1.0;
	];
	assert_eq!(m.trace(), 15.0);
  ```

- `.determinant()` for determinant (only available for square matrix)
  ```rust
    let m = matrix![
	  3.0, 7.0;
	  1.0, -4.0;
	];
    assert_eq!(m.determinant(), -19.0);
  ```

- `.inverse()` for inverse of a square invertible matrix
  ```rust
	let m = matrix![
	  6.0, 2.0, 3.0;
	  1.0, 1.0, 1.0;
	  0.0, 4.0, 9.0;
	];
	let exp = matrix![
	  0.20833333, -0.25, -0.04166667;
	      -0.375,  2.25,      -0.125;
	  0.16666667,  -1.0,  0.16666667;
	];
	assert_relative_eq!(m.inverse(), exp, max_relative = 1e-6);
  ```

- `.cholesky()` for symmetric positive-definite systems
  ```rust
  let matrix = Matrix::<3, 3, f64>::eye();
  let factor = matrix.cholesky().expect("matrix is positive-definite");
  let solution = factor.solve(&vector![1.0; 2.0; 3.0]);
  ```

- `.ldlt()` for symmetric systems with Eigen-compatible diagonal pivoting
  ```rust
  let matrix = matrix![0.0_f64, 2.0; 2.0, 3.0];
  let factor = matrix.ldlt().expect("matrix is nonsingular");
  let solution = factor.solve(&vector![1.0; 4.0]);
  ```

- `.ldlt_no_pivot()` for stable systems when pivot-search overhead is unnecessary
  ```rust
  let factor = matrix.ldlt_no_pivot().expect("matrix is nonsingular");
  ```

The dense LDLT path stores compact 1x1 or 2x2 `D` pivot blocks using
Eigen-compatible Bunch–Kaufman selection. `pivot_blocks()` reports the block
layout (`1`, `2`, `3`), while `diagonal_matrix()` reconstructs the full `D`.
Native block sparse LDLT retains block ordering plus local Bunch–Kaufman
metadata inside each dense diagonal block. Use `try_dense_ldlt::<N>()` on a
block matrix when a global scalar pivot may cross block boundaries.

- `.householder_qr()` for full-rank square or overdetermined least-squares systems
  ```rust
  let design = matrix![
      1.0_f64, 1.0;
      1.0, 2.0;
      1.0, 3.0;
  ];
  let observations = vector![3.0; 5.0; 7.0];
  let coefficients = design
      .householder_qr()
      .solve_least_squares(&observations)
      .expect("design matrix is full rank");
  ```
  QR factors also expose `apply_q`, `apply_q_transpose`, and in-place variants
  for allocation-free orthogonal transforms.

- `.col_piv_householder_qr()` for rank-aware least-squares systems
  ```rust
  let factor = design.col_piv_householder_qr();
  assert_eq!(factor.rank(), 2);
  let coefficients = factor
      .solve_least_squares(&observations)
      .expect("design matrix is full rank");
  ```
  Use `.solve_least_squares_basic()` when dependent columns should be handled
  using Eigen-compatible basic rank-deficient semantics.

- `.svd()` for fixed-size SVD of square, tall, or wide matrices
  ```rust
  let svd = design.svd().expect("SVD decomposition succeeds");
  let rank = svd.rank();
  let coefficients = svd.solve(&observations);
  ```
  SVD also exposes `singular_values()`, `u()`, `v()`, and `pseudo_inverse()`.

All decompositions provide reusable-output solve variants: use `solve_into` for
LU, Cholesky, LDLT, and SVD, or `solve_least_squares_into` for QR. Cholesky and
LDLT also retain `solve_in_place` when the right-hand side itself can be reused.
All factor types support Eigen-style recomputation: use `try_compute` for
fallible factorizations and `compute` for infallible ones. Factor storage is
owned by the factor object and reused by these methods.
Self-adjoint eigendecomposition also accepts a reusable
`SelfAdjointEigenWorkspace` through `try_compute_with_workspace` when callers
need explicit stack/RAM control.

- `.self_adjoint_eigen()` for fixed-size symmetric eigendecomposition
  ```rust
  let eig = matrix.self_adjoint_eigen().expect("matrix is symmetric");
  let values = eig.eigenvalues();
  let vectors = eig.eigenvectors();
  ```
  Eigenvalues are sorted in ascending order and eigenvectors are orthonormal.

- `.self_adjoint_lower()` and `.self_adjoint_upper()` for zero-copy Eigen-style
  views that mirror one authoritative triangle without reading the other.
  Use `try_self_adjoint_lower(tolerance)` or
  `try_self_adjoint_upper(tolerance)` when symmetry validation is required;
  `validate_symmetric` and `is_symmetric` expose the same scaled check.

- `PartialPivLu` for allocation-free linear solves
  ```rust
  let matrix = Matrix::<3, 3, f64>::eye();
  let factor = matrix.partial_piv_lu();
  let solution = factor.solve(&vector![1.0; 2.0; 3.0]);
  ```

- `.mul_into()` for allocation-free matrix multiplication into a reusable output
  ```rust
  let matrix = Matrix::<3, 3, f64>::eye();
  let mut output = Matrix::<3, 3, f64>::zeros();
  matrix.mul_into(&matrix, &mut output);
  ```

- `block`, `row`, and `column` views for fixed-size submatrix access
  ```rust
  let block = matrix.block::<2, 2>(0, 0).expect("block is in bounds");
  let values = block.to_matrix();
  ```
  Mutable blocks are available through `block_mut`. Triangular views expose
  `lower_triangular()` and `upper_triangular()` with in-place solves and
  `mul_into` operations.

- `Map` and `MapMut` for zero-copy fixed-size views over external column-major
  buffers, with checked construction and mutable indexing. `StridedMap` and
  `StridedMapMut` cover padded or row-major buffers without repacking.
  `MatrixRead` and `MatrixWrite` provide a shared compile-time-dimension view
  interface, and all dense decompositions expose `try_decompose_view` plus
  `try_compute_view` variants for zero-copy factorization from compatible views.
  `Matrix::from_view` remains available when an owned snapshot is required.

- `MatrixBuf<MAX_ROWS, MAX_COLS, T>` for bounded runtime dimensions without
  heap allocation. It reserves a compile-time capacity, tracks active rows and
  columns, supports checked resizing and column access, and exposes matching
  active regions through zero-copy `as_view::<M, N>()`/
  `as_view_mut::<M, N>()` views. It can also round-trip to fixed-size `Matrix`
  values. `Matrix`, `MatrixBuf`, sparse patterns/factors, and block sparse
  matrices expose `storage_bytes()` for compile-time RAM budgeting on embedded
  targets.

- `StaticBlockCscMatrix` and `StaticBlockCsrMatrix` for fixed-capacity block
  sparse storage. Block patterns are validated once, dense blocks remain
  stack-owned, and block matvec writes into caller-provided scalar slices
  without allocation. Block CSC supports native block Cholesky plus a bounded
  scalar-CSC expansion reference path. Native block LDLT supports compact
  local Bunch–Kaufman diagonal blocks, fixed block orderings, and analysis-time
  block diagonal pivoting.

- `StaticCscPattern` and `StaticCscMatrix` for allocation-free, bounded sparse
  CSC storage. Symbolic structure can be validated once and numeric values
  updated repeatedly; fixed-size sparse matrix-vector products are supported.
  `StaticCscCholeskyPattern` and `StaticCscCholesky` add reusable symbolic and
  numeric sparse LLT factorization with bounded fill-in storage. LLT consumes
  the lower triangle (optional mirrored upper entries are checked for symmetry).

- `Quaternion<T>` and `RotationMatrix<T>` for generic 3D rotations
  ```rust
  let rotation = Quaternion::from_axis_angle(&axis, angle)
      .expect("axis is nonzero")
      .to_rotation_matrix()
      .expect("quaternion is nonzero");
  let rotated = rotation.apply(&point);
  ```
  These types use existing fixed-size `Matrix`/`Vector` storage and do not
  depend on robotics or code-generation crates.

- `Isometry<T>` for fixed-size rigid transforms, with point/direction
  application, composition, inverse, and homogeneous `4x4` conversion.

- `AngleAxis<T>` and quaternion `slerp` provide interpolation-friendly
  rotation APIs; `AffineTransform<T>` adds validated homogeneous affine
  transforms with composition, point/direction application, and inversion.

Fixed-size multiplication selects its kernel at compile time. On
x86-64, `f32` and `f64` use AVX2 when enabled by the target and otherwise use
an SSE2 kernel; AArch64 targets with NEON use the ARM NEON packet kernel;
reduction kernels use fused multiply-add when both AVX2 and FMA are enabled.
AVX2-only targets retain a non-FMA packet path, and targets without a packet
backend use the portable scalar fallback. Use
`RUSTFLAGS="-C target-cpu=native"` only when the resulting binary will run on
the same CPU feature set. Dot products, squared norms, and matrix-vector
products use the same compile-time scalar/packet dispatch.
Custom scalar types can implement `MatrixScalar` for multiplication and add
`ReductionScalar` when dot, norm, or matrix-vector kernels are needed.

## Eigen and faer comparison

The published library has no C++ or faer dependency. Local correctness and
performance comparisons are opt-in and require Eigen headers discoverable via
`pkg-config eigen3` or `EIGEN3_INCLUDE_DIR`:

```sh
cargo test --features eigen-compare
RUSTFLAGS="-C target-cpu=native" cargo bench --bench fixed_size --bench small_fixed
RUSTFLAGS="-C target-cpu=native" cargo bench --bench sparse
RUSTFLAGS="-C target-cpu=native" cargo bench --bench dense_solvers
# Focus on decomposition cases:
RUSTFLAGS="-C target-cpu=native" cargo bench --bench small_fixed -- 'llt|ldlt'
CXXFLAGS="-march=native" ./eigen/run_native_bench.sh f64 "Sparse LLT"
CXXFLAGS="-march=native" ./eigen/run_native_bench.sh f64 QR
CXXFLAGS="-march=native" ./eigen/run_native_bench.sh f64 SVD
CXXFLAGS="-march=native" ./eigen/run_native_bench.sh f64 "Self-adjoint eigen"
CXXFLAGS="-march=native" ./eigen/run_native_bench.sh f64 triangular
```

The parity suite compares elementary operations bit-for-bit and compares
floating-point reductions and decompositions with documented tolerances. The
Criterion reports include `stack-algebra` and reusable-buffer faer baselines
for `f32` and `f64` square products from 2-by-2 through 15-by-15, small fixed
shapes (`2x3 * 3x2`, `3x6 * 6x3`, and `6x15 * 15x6`), matrix-vector products,
dot products, norms, and partial-pivot LU factorization and one-right-hand-side solves. The native
Eigen runner uses the same static, column-major matrices, input values,
dimensions, and 64-operation batch. Pass `QR`, `LLT`, or another operation
filter as the optional second argument to run only matching native cases.
Compare its `ns/batch` median with
Criterion's reported time; its `ns/op` column is the batch time divided by 64.
It uses no Rust-to-C++ calls in the timed region. Criterion also includes faer
dynamic-matrix LLT and LDLT factor/solve baselines. Those faer cases measure
faer's normal heap-backed `Mat` API, while stack-algebra and Eigen use fixed-size
stack storage, so they are algorithm comparisons rather than identical memory
allocation models. Decomposition comparisons
include partial-pivot LU, Householder QR, Cholesky LLT on generated SPD systems, and pivoted
LDLT plus explicit no-pivot LDLT on generated symmetric-indefinite systems, each with factorization and
one-right-hand-side solve cases. LLT and LDLT include 3, 6, 15, and 32 square
dimensions to expose small-matrix overhead and larger fixed-size scaling.
QR additionally covers tall `6x3`, `15x6`, `32x8`, and `64x16` systems.
SVD benchmarks cover tall `6x3` and `15x6` systems.
Self-adjoint eigendecomposition benchmarks cover symmetric `3x3`, `6x6`,
`15x15`, and `32x32` systems.
Triangular solve benchmarks cover lower and upper `3x3`, `6x6`, and `15x15`
systems using the same static column-major inputs as Eigen.
Sparse LLT and no-pivot LDLT benchmarks use lower-triangular tridiagonal,
band-2, and star patterns at representative fixed sizes in both `f32` and `f64`; analysis,
numeric refactorization with a reused symbolic pattern, and solve are reported
separately for stack-algebra and faer. Native Eigen comparison covers the dense
benchmark suite plus sparse LLT/LDLT when `eigen-compare` is enabled. Fixed-capacity sparse Cholesky
also exposes deterministic minimum-degree ordering for reducing fill-in before
reusing a symbolic pattern. Stack-algebra factor benchmarks reuse the numeric
factor buffer, matching Eigen's in-place `factorize` model.
Repeated ordered refactorization can validate and prepare the ordered CSC
matrix once with `prepare_ordered`, then use `recompute_ordered` for numeric
updates without repeating permutation or structural checks. For natural CSC
coordinates, call `recompute` on the existing numeric factor.
Sparse LDLT provides a fast no-pivot path plus analysis-time 1x1 diagonal
pivoting; matrices requiring scalar 2x2 pivot blocks are reported as zero-pivot
failures. Dense fixed-size LDLT supports Bunch–Kaufman 1x1/2x2 blocks.
The block-sparse benchmark also reports native local-pivot LDLT, the global
`try_dense_ldlt` fallback, faer, and optional Eigen solve baselines for a
cross-block indefinite 2x2 case.
The dense-solver benchmark adds 8x8, 16x16, and 32x32 SPD LDLT factor-and-solve
and reused-solve measurements for stack-algebra and faer, with an optional
Eigen baseline. Matrix construction, factor setup, and faer/Eigen allocation are
outside reused-factor and reused-solve timing.
The same benchmark reports reusable stack LDLT with and without Bunch–Kaufman
pivoting; this separates the robust pivot-selection/update overhead from the
underlying no-pivot factorization before any architecture-specific tuning.

## QEMU target validation

The standalone Cortex-M harness builds the library for
`thumbv7em-none-eabihf` and runs deterministic multiplication, LU-solve, and
pivoted/no-pivot LDLT checks under QEMU's MPS2 Cortex-M4 machine. The harness
also reports a stack watermark from `cortex-m-rt`'s painted stack. The AArch64
and RISC-V harnesses report the same bounded 64 KiB watermark:

```sh
qemu-tests/run_cortex_m.sh
```

The RISC-V harness targets `riscv32imc-unknown-none-elf` and runs the same
checks on QEMU's generic `virt` machine. This validates the scalar kernel path
used by RISC-V microcontrollers such as ESP32-C-class devices:

```sh
rustup target add riscv32imc-unknown-none-elf
qemu-tests/run_riscv32.sh
```

The AArch64 harness targets `aarch64-unknown-none` and validates the NEON
packet kernels on QEMU's Cortex-A53 model:

```sh
rustup target add aarch64-unknown-none
qemu-tests/run_aarch64.sh
```

The scripts enforce the linker-provided stack budget. Set
`STACK_USAGE_LIMIT_BYTES` to apply a stricter regression threshold; CI uses an
8 KiB limit for all three targets.

## License

This project is distributed under the terms of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.

[vectrix]: https://docs.rs/vectrix/latest/vectrix/
[frobenius]: https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
