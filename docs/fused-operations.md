# Fused operations and mapped-view fast paths

`stack-algebra` deliberately avoids a general expression-template system. For
hot estimation and control loops, use explicit output or in-place operations
when they remove a known temporary without hiding storage ownership.

## Scaled accumulation

Use `axpy_in_place` for the common update `y += alpha * x`:

```rust
use stack_algebra::matrix;

let x = matrix![1.0_f32, 2.0; 3.0, 4.0];
let mut y = matrix![5.0_f32, 6.0; 7.0, 8.0];
y.axpy_in_place(2.0, &x);
assert_eq!(y, matrix![7.0, 10.0; 13.0, 16.0]);
```

When the destination must be separate, `axpy_into` writes
`alpha * x + y` directly into caller-owned storage:

```rust
use stack_algebra::{matrix, Matrix};

let x = matrix![1.0_f32, 2.0; 3.0, 4.0];
let y = matrix![5.0_f32, 6.0; 7.0, 8.0];
let mut output = Matrix::<2, 2, f32>::zeros();
x.axpy_into(2.0, &y, &mut output);
assert_eq!(output, matrix![7.0, 10.0; 13.0, 16.0]);
```

For two scaled inputs, `linear_combination_into` computes
`alpha * x + beta * y` without building the two scaled temporary matrices:

```rust
use stack_algebra::{matrix, Matrix};

let x = matrix![1.0_f64, 2.0; 3.0, 4.0];
let y = matrix![5.0_f64, 6.0; 7.0, 8.0];
let mut output = Matrix::<2, 2, f64>::zeros();
x.linear_combination_into(2.0, &y, -1.0, &mut output);
assert_eq!(output, matrix![-3.0, -2.0; -1.0, 0.0]);
```

These operations use the scalar backend's `mul_add` operation. Built-in
floating-point scalars therefore retain the maintained fused multiply-add path
where the selected target supports it.

## Contiguous mapped input

A `Map` is exactly column-major contiguous. Its direct `mul_into` and
`matvec_into` methods reuse the same compile-time-selected kernels as an owned
`Matrix` without first copying the mapped buffer:

```rust
use stack_algebra::{matrix, Map, Matrix};

let lhs_storage = [1.0_f32, 3.0, 2.0, 4.0];
let rhs_storage = [5.0_f32, 7.0, 6.0, 8.0];
let lhs = Map::<2, 2, _>::from_slice(&lhs_storage).unwrap();
let rhs = Map::<2, 2, _>::from_slice(&rhs_storage).unwrap();
let mut output = Matrix::<2, 2, f32>::zeros();
lhs.mul_into(&rhs, &mut output);
assert_eq!(output, matrix![19.0, 22.0; 43.0, 50.0]);
```

Owned/mapped mixed forms are available as `Matrix::mul_map_into`,
`Map::mul_matrix_into`, and the corresponding strided methods.

## Strided input

`StridedMap` keeps its general zero-copy contract. When its runtime strides are
exactly the normal column-major layout (`inner_stride = 1` and
`outer_stride = rows`), its direct product methods reuse the optimized owned
kernel without repacking:

```rust
use stack_algebra::{matrix, Matrix, StridedMap};

let storage = [1.0_f32, 3.0, 2.0, 4.0];
let lhs = StridedMap::<2, 2, _>::from_slice(&storage, 1, 2).unwrap();
let rhs = StridedMap::<2, 2, _>::from_slice(&storage, 1, 2).unwrap();
let mut output = Matrix::<2, 2, f32>::zeros();
lhs.mul_into(&rhs, &mut output);
assert_eq!(output, matrix![7.0, 10.0; 15.0, 22.0]);
```

Padded column-major, row-major, and arbitrary-stride layouts still operate
without an owning input copy, but currently use the generic direct-read kernel.
This distinction is intentional: the current optimized kernels assume the exact
owned `Matrix` layout. A future leading-dimension kernel should be added only
with benchmark evidence rather than by silently materializing a temporary.

The free `matmul_view_into` and `matvec_view_into` functions remain the fully
generic `MatrixRead` interface. Use the direct `Map`/`StridedMap` methods when
the concrete mapped type is known and optimized routing matters.

## Benchmarking

The `fused` Criterion benchmark compares:

- expression temporaries against `axpy_into`;
- expression temporaries against `linear_combination_into`;
- generic `Map` matrix multiplication against `Map::mul_into`;
- generic `Map` matrix-vector multiplication against `Map::matvec_into`.

The nightly benchmark job runs these at representative 6x6, 15x15, and 32x32
sizes as regression triage. Those short hosted-runner measurements are not
release performance evidence; release claims still require the pinned-machine
methodology described in the benchmarking guide.

A GEMM-like `C = alpha * A * B + beta * C` primitive remains deferred until
these measurements show that a dedicated accumulate path would materially help
the target workloads.
