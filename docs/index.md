# stack-algebra

`stack-algebra` is a fixed-size linear-algebra library for Rust applications that
care about **predictable memory use, compile-time dimensions, and `no_std`
support**. It is particularly useful for embedded systems, robotics, estimation,
control, and other workloads where matrix sizes are known or bounded ahead of
time.

The core types store their data inline and do not require a heap. When the data
already lives elsewhere, borrowed views let you operate on caller-owned,
strided, or submatrix storage without first repacking it into a new matrix.

## When it is a good fit

| You need | Start with |
| --- | --- |
| Small or medium matrices with known dimensions | `Matrix<M, N, T>` |
| Runtime-active dimensions with a fixed maximum | `MatrixBuf<MAX_ROWS, MAX_COLS, T>` |
| A DMA, generated, padded, or caller-owned buffer | `Map`, `StridedMap`, or `Block` |
| Dense linear solves or least squares | Cholesky, LDLT, LU, QR, or SVD |
| 3D rotations and rigid transforms | `Quaternion`, `RotationMatrix`, `Isometry` |
| A fixed or bounded sparse structure | Static CSC/CSR and block-sparse types |
| An embedded target without `alloc` | The default `no_std` core |

`stack-algebra` is less suitable when the primary requirement is very large,
fully dynamic, heap-backed matrices whose dimensions change freely at runtime.
In that case, a library centered on dynamic desktop/server workloads may be a
better match.

## A first solve

```rust
use stack_algebra::{matrix, Cholesky};

let a = matrix![
    4.0_f64, 1.0;
    1.0,     3.0;
];
let b = matrix![1.0_f64; 2.0];

let factor = Cholesky::try_decompose(&a).expect("A must be positive definite");
let x = factor.solve(&b);

assert!((a * x - b).norm() < 1.0e-12);
```

The dimensions are part of the Rust type, so shape mismatches are caught at
compile time. The scalar type is explicit as well; use `.cast()` when crossing
a deliberate precision boundary.

## Where to go next

- [Getting started](getting-started.md) introduces the matrix model, storage,
  precision, and solver choices with small examples.
- [Choosing an API](api-usage.md) is the practical decision guide for storage,
  views, decompositions, reuse, and sparse workflows.
- [Common use cases](use-cases.md) maps the library to estimation, calibration,
  control, geometry, sparse systems, and embedded loops.
- [Capabilities and limits](features.md) is the compact inventory of what the
  crate currently supports and where its boundaries are.
- [Platforms and embedded use](targets.md) explains `no_std` support, tested
  targets, memory placement, and what you should still validate on hardware.
- [Performance](benchmarking.md) gives measured context against familiar linear
  algebra libraries. The comparisons are informational rather than a claim
  that `stack-algebra` is intended to replace or outperform them.
- [API reference](api-reference.md) links to the rustdoc generated from the same
  revision as this guide.

## Design at a glance

The library favors explicitness over hidden policy:

- dimensions are compile-time constants for `Matrix`, or bounded for
  `MatrixBuf`;
- dense values are column-major;
- the fixed-size core does not require heap allocation;
- `f32` and `f64` precision is explicit rather than implicitly mixed;
- reusable factors, workspaces, and `*_into`/in-place operations are available
  for repeated loops;
- sparse symbolic structure can be retained while numeric values change;
- target-specific kernels are selected at compile time, with a portable scalar
  path available for unsupported targets.

Those choices are meant to make memory ownership, numerical assumptions, and
runtime cost easier to reason about in systems code.