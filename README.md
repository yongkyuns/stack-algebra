# stack-algebra

[![Crates.io Version](https://img.shields.io/crates/v/stack-algebra.svg)](https://crates.io/crates/stack-algebra)
[![Docs.rs Latest](https://img.shields.io/badge/docs.rs-latest-blue.svg)](https://docs.rs/stack-algebra)
![Build Status](https://github.com/yongkyuns/stack-algebra/actions/workflows/ci.yml/badge.svg?branch=main)

A stack-allocated lightweight algebra library for bare-metal applications.

## Overview
This crate provides a stack-allocated matrix type with constant size determined at 
compile time. The primary goal for this library is to be useful in building robotics
applications in rust. This means several things:
1. Target platform is often bare-metal
2. Size of the matrices can usually be defined at compile-time
3. Problem solving does not require large matrices or heavy optimization
4. Users are not experts in rust but often familiar with scientific tools 
(e.g. python or matlab)

Implementing numerical algorithms in rust can be made much more productive and ergonomic
if simple abstractions and necessary algebra routines are available. This library is
a growing collection of addressing those needs. It is heavily based on 
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

`Matrix<R, C, T>` is always stack allocated and column major. `T` defaults to
`f32`; use explicit casts when changing precision.

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

- `*`, `/`, `+`, `-` for matrix arithmatics
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

- `.norm()` for computing the [`Frobenius norm`][frobenius]
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

- `.trace()` for sum of diagonal elements of a sqaure matrix
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

- `.ldlt()` for symmetric indefinite systems with pivoting
  ```rust
  let matrix = matrix![0.0_f64, 2.0; 2.0, 3.0];
  let factor = matrix.ldlt().expect("matrix is nonsingular");
  let solution = factor.solve(&vector![1.0; 4.0]);
  ```

- `.ldlt_no_pivot()` for stable systems when pivot-search overhead is unnecessary
  ```rust
  let factor = matrix.ldlt_no_pivot().expect("matrix is nonsingular");
  ```

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

Fixed-size multiplication selects its kernel at compile time. On
x86-64, `f32` and `f64` use AVX2 when enabled by the target and otherwise use
an SSE2 kernel; reduction kernels use fused multiply-add when both AVX2 and
FMA are enabled. AVX2-only targets retain a non-FMA packet path, and other
targets use the portable scalar fallback. Use
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
RUSTFLAGS="-C target-cpu=native" cargo bench --bench fixed_size --bench robotics
# Focus on decomposition cases:
RUSTFLAGS="-C target-cpu=native" cargo bench --bench robotics -- 'llt|ldlt'
CXXFLAGS="-march=native" ./eigen/run_native_bench.sh
```

The parity suite compares elementary operations bit-for-bit and compares
floating-point reductions and decompositions with documented tolerances. The
Criterion reports include `stack-algebra` and reusable-buffer faer baselines
for `f32` and `f64` square products from 2-by-2 through 15-by-15, robotics
shapes (`2x3 * 3x2`, `3x6 * 6x3`, and `6x15 * 15x6`), matrix-vector products,
dot products, norms, and partial-pivot LU factorization and one-right-hand-side solves. The native
Eigen runner uses the same static, column-major matrices, input values,
dimensions, and 64-operation batch. Compare its `ns/batch` median with
Criterion's reported time; its `ns/op` column is the batch time divided by 64.
It uses no Rust-to-C++ calls in the timed region. Criterion also includes faer
dynamic-matrix LLT and LDLT factor/solve baselines. Those faer cases measure
faer's normal heap-backed `Mat` API, while stack-algebra and Eigen use fixed-size
stack storage, so they are algorithm comparisons rather than identical memory
allocation models. Decomposition comparisons
include partial-pivot LU, Cholesky LLT on generated SPD systems, and pivoted
LDLT plus explicit no-pivot LDLT on generated symmetric-indefinite systems, each with factorization and
one-right-hand-side solve cases. LLT and LDLT include 3, 6, 15, and 32 square
dimensions to expose small-matrix overhead and larger fixed-size scaling.

## QEMU target validation

The standalone Cortex-M harness builds the library for
`thumbv7em-none-eabihf` and runs deterministic multiplication and LU-solve
checks under QEMU's MPS2 Cortex-M4 machine:

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

## License

This project is distributed under the terms of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.

[vectrix]: https://docs.rs/vectrix/latest/vectrix/
[frobenius]: https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
