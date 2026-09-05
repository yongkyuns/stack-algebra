# Common use cases

`stack-algebra` is most useful when the numerical problem is structurally small,
fixed, or bounded and the application benefits from explicit memory ownership.
The examples below focus on how that shows up in real systems rather than on
individual API methods.

## Quick selection guide

| Workload | Useful starting point | Why it fits |
| --- | --- | --- |
| Rigid-body pose and frame transforms | `Quaternion`, `RotationMatrix`, `Isometry` | Small fixed geometry types with explicit scalar precision |
| IMU/GNSS or other state estimation | `Matrix`, Cholesky/LDLT, `Block` | Fixed state sizes, reusable factors, covariance sub-blocks |
| Calibration and linearized least squares | QR, pivoted QR, or SVD | Solver can follow rank/conditioning assumptions |
| Fixed-horizon control or optimization | `Matrix`, `MatrixBuf`, QR/Cholesky | Bounded problem sizes and reusable scratch |
| Fixed-topology sparse estimation | Static CSC + sparse LLT/LDLT | Symbolic structure can be retained across iterations |
| Block-structured graph or normal equations | Block CSC/CSR | Dense state/pose blocks remain explicit |
| Generated, FFI, or DMA-owned data | `Map`, `StridedMap`, `Block` | Work directly from external storage |
| MCU control/estimation loop | Fixed/bounded storage + `storage_bytes()` | Predictable memory and no required heap |

## Coordinate frames and rigid transforms

For pose, calibration, and frame conversion, the geometry types avoid treating
rotation as an unstructured matrix when the stronger representation is useful.

- `Quaternion<T>` represents a 3D rotation compactly.
- `AngleAxis<T>` is convenient at construction and conversion boundaries.
- `RotationMatrix<T>` exposes the familiar 3x3 representation.
- `Isometry<T>` combines rotation and translation for rigid transforms.
- `AffineTransform<T>` covers the broader affine case when rigidity is not a
  valid assumption.

```rust
use stack_algebra::{AngleAxis, Isometry, Vector3f};

let axis = Vector3f::from_columns([[0.0, 0.0, 1.0]]);
let rotation = AngleAxis::new(&axis, 0.25)
    .expect("nonzero axis")
    .to_rotation_matrix()
    .expect("valid axis-angle");

let transform = Isometry::from_parts(rotation, axis);
let point = Vector3f::from_columns([[1.0, 0.0, 0.0]]);
let transformed = transform.apply_point(&point);
```

A useful boundary is to keep a rotation-specific type while the value represents
a physical rotation, then convert to a dense matrix only when the surrounding
algebra actually requires matrix form.

## State estimation and covariance updates

Small fixed-state filters are a natural match for compile-time matrix sizes.
For example, an estimator can keep state, covariance, Jacobian, innovation, and
workspace dimensions explicit in their Rust types.

For a positive-definite innovation/covariance solve, start with Cholesky:

```rust
let factor = innovation
    .try_cholesky()
    .expect("innovation covariance should be positive definite");
let correction = factor.solve(&residual);
```

If a symmetric intermediate is legitimately indefinite, LDLT is a better model
of the mathematics than silently forcing a Cholesky path.

In a high-rate loop:

- retain factor and output storage when the shape is unchanged;
- call `try_compute` rather than rebuilding surrounding workspaces;
- use `Block` when the needed covariance/Jacobian region already lives inside a
  larger matrix;
- use `f32` or `f64` based on conditioning and target behavior, not habit alone.

This makes the memory cost of the filter easier to account for and keeps
failure of a numerical assumption visible to the caller.

## Sensor calibration and least squares

Calibration often produces a small tall Jacobian and a residual vector.
Householder QR is a good default when the design is expected to be full rank.
Column-pivoted QR is useful when rank loss should be detected, and SVD is useful
when singular values or a pseudoinverse are part of the workflow.

```rust
let qr = jacobian.col_piv_householder_qr();
let rank = qr.rank();

if rank == jacobian.cols() {
    let step = qr
        .solve_least_squares(&residual)
        .expect("full-rank design");
}
```

This pattern applies to sensor mounting calibration, local linear regression,
small bundle adjustments, and other problems where the number of parameters is
known even if measurements are assembled repeatedly.

When conditioning is uncertain, prefer QR/SVD over forming normal equations
just to reach a Cholesky solve. The stronger SPD assumption is valuable only
when the application can justify it.

## Fixed-horizon control and optimization

A fixed prediction horizon is often easier to express with bounded storage than
with a general dynamic-matrix abstraction.

A typical loop is:

1. assemble the current linearization into fixed or bounded matrices;
2. expose the active region through a view if `MatrixBuf` is used;
3. choose QR for a general least-squares step or Cholesky for a known SPD
   system;
4. reuse factor/output storage on the next iteration;
5. account for the complete workspace before placing it on an MCU stack.

`stack-algebra` deliberately does not provide an NMPC framework. The application
still owns horizon layout, constraints, linearization, globalization, and solver
policy. The library is the numerical building block underneath those choices.

## Sparse estimation and fixed-topology graphs

When a graph or normal-equation pattern is fixed for many iterations, symbolic
reuse can matter more than a one-shot factorization API.

With scalar CSC storage:

1. build and validate the sparse pattern once;
2. analyze the factor structure once;
3. update numeric values for each new linearization;
4. recompute the numeric factor;
5. solve one or more right-hand sides.

This fits fixed-topology pose graphs, repeated sparse normal equations, and
bounded batch estimators where the maximum structure is known in advance.

Use block CSC/CSR when the problem naturally consists of repeated dense state or
pose blocks. Block storage is most useful when those blocks are meaningful to
assembly and factorization—not simply as a way to make a scalar sparse matrix
look more complicated.

## External, generated, and DMA-owned buffers

Many embedded and robotics systems do not want the algebra library to own the
source data. A driver, code generator, FFI boundary, or shared workspace may
already define the layout.

For contiguous column-major data, borrow it with `Map`. For padded, interleaved,
or row-major data, use `StridedMap`. For a region inside an existing matrix,
use `Block`.

```rust
let jacobian = stack_algebra::Map::<6, 3, f32>::from_slice(jacobian_data)
    .expect("generated buffer has the expected size");

let factor = jacobian.col_piv_householder_qr();
let step = factor.solve_least_squares(&rhs);
```

The important property here is ownership: the source remains owned by the
surrounding subsystem, while the numerical operation borrows it for as long as
needed.

## Embedded control and estimation loops

For MCU code, the main advantage is not that every matrix value literally lives
on the stack. It is that storage is **fixed or bounded and visible in the type**.
The owner decides whether that value belongs in a local frame, static memory, a
long-lived application state object, or a caller-owned buffer.

A practical sequence is:

1. choose fixed `Matrix` dimensions where possible;
2. use `MatrixBuf` only where the active size genuinely varies;
3. start with `f32` unless the numerical problem or target justifies `f64`;
4. use `storage_bytes()` to estimate long-lived matrix/factor/workspace costs;
5. reuse `*_into` or in-place outputs in high-rate loops when useful;
6. test the complete workload on the actual MCU before making cycle or stack
   claims.

```rust
use stack_algebra::{Matrix, MatrixBuf};

const STATE_BYTES: usize = Matrix::<15, 1, f32>::storage_bytes();
const COVARIANCE_BYTES: usize = Matrix::<15, 15, f32>::storage_bytes();
const WORK_BYTES: usize = MatrixBuf::<32, 32, f32>::storage_bytes();
```

QEMU and cross-compilation are useful portability checks, but they do not model
your board's cache, FPU throughput, linker layout, interrupts, or real stack
budget. See [Platforms and embedded use](targets.md) for the current test matrix.

## When another library may be a better fit

`stack-algebra` is intentionally centered on fixed and bounded workloads. A
library designed around dynamic heap-backed matrices may be a better choice
when:

- matrix dimensions are large and genuinely unpredictable at runtime;
- the workload is dominated by large desktop/server dense algebra;
- you need a broad dynamic sparse ecosystem or backend integrations that are
  outside this crate's scope;
- allocating and resizing matrices freely is more important than static memory
  accounting.

Using another library in those cases is not a failure of `stack-algebra`; it is
simply a different design point. The [Performance](benchmarking.md) page uses
other libraries as familiar measurement references for the same reason: to give
context, not to position the project as a replacement for them.