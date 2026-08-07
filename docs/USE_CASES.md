# Robotics Use Cases

`stack-algebra` is a standalone numerical library. These examples describe how
robotics applications can use its APIs; they do not add a dependency on a
robotics framework or code generator.

## Quick selection guide

| Workload | Recommended starting point | Why |
| --- | --- | --- |
| Rigid-body pose and frame transforms | `Quaternion`, `RotationMatrix`, `Isometry` | Validated fixed-size geometry with no heap allocation |
| Small state propagation or covariance update | `Matrix`, `MatrixBuf`, Cholesky/LDLT | Predictable RAM and reusable factors |
| Linearized measurement update | QR or SVD | Least-squares and rank-aware solves |
| Dense short-horizon optimization | QR, pivoted QR, or Cholesky | Fixed horizon and explicit scratch |
| Sparse pose-graph or batch normal equations | CSC pattern + sparse LLT/LDLT | Symbolic reuse and bounded fill |
| Block-structured sparse matvec | Block CSC/CSR | Explicit block layout and no packing step |
| Existing generated or DMA buffer | `Map` / `StridedMap` / `Block` | Factor without an intermediate input matrix |
| MCU deployment | Fixed `Matrix`, `MatrixBuf`, `storage_bytes()` | Compile-time resource budgeting |

## Rigid transforms and kinematics

Use the geometry types for rotations and rigid transforms. Keep scalar precision
consistent across a subsystem:

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

Use `Matrix3f`/`Matrix3d` directly when a downstream algorithm needs a rotation
matrix. Use `storage_bytes()` when a transform is embedded in a larger state
workspace.

## State estimation and covariance solves

For a positive-definite covariance or innovation matrix, use Cholesky. For a
symmetric matrix that can be indefinite during debugging or poor conditioning,
use pivoted LDLT and inspect the typed error:

```rust
use stack_algebra::DecompositionError;

// Inside a function returning Result<_, DecompositionError>:
let step = match innovation.try_cholesky() {
    Ok(factor) => factor.solve(&residual),
    Err(DecompositionError::NotPositiveDefinite) => {
        innovation.try_ldlt()?.solve(&residual)
    }
    Err(error) => return Err(error),
}
```

In a repeated filter loop, retain the factor object and call `try_compute`.
When the covariance lives in a larger workspace, call `try_compute_view` on a
`Block` instead of copying the submatrix.

## Linearized least squares

Use Householder QR when the Jacobian is expected to be full rank. Use
column-pivoted QR when rank loss should be detected cheaply. Use SVD for a
pseudoinverse or a configurable rank threshold:

```rust
let qr = jacobian.col_piv_householder_qr();
if qr.rank() == jacobian.cols() {
    let step = qr.solve_least_squares(&residual).expect("full rank");
}

let svd = jacobian.svd().expect("SVD converges");
let damped_or_rank_aware_step = svd.solve(&residual);
```

The matrix dimensions remain compile-time constants. For a changing active
window, choose a maximum horizon and use `MatrixBuf` for assembly, then expose
matching active dimensions with `as_view::<M, N>()` for zero-copy view-based
factorization.

## NMPC and trajectory optimization

A fixed horizon is a natural fit for `Matrix<M, N, T>`:

1. Assemble the linearized dynamics and cost Jacobian into fixed-size storage.
2. Use QR for a general least-squares step, or Cholesky for a known SPD normal
   equation system.
3. Reuse the factor and output matrices at every iteration.
4. Use `storage_bytes()` to verify the state, Jacobian, factor, and RHS fit the
   target stack/RAM budget.

Prefer QR over explicitly forming normal equations when numerical conditioning
is uncertain. Prefer Cholesky when the SPD contract is guaranteed and the
extra speed/RAM savings matter.

There is no NMPC-specific type in the crate. The application owns the horizon,
state ordering, constraints, and iteration policy.

## SLAM and pose-graph systems

For a fixed graph topology, represent the scalar sparsity pattern with
`StaticCscPattern` and numeric values with `StaticCscMatrix`:

1. Build the symbolic pattern once.
2. Analyze LLT or LDLT once, optionally applying minimum-degree ordering.
3. Update numeric values for each linearization.
4. Call `recompute` or `recompute_ordered` and solve multiple RHS columns.

For block-structured Jacobian products, use `StaticBlockCscMatrix` or
`StaticBlockCsrMatrix`. Native block Cholesky and LDLᵀ are available for square
blocks and grids; `StaticBlockCscMatrix::cholesky` remains a scalar-expansion
reference path. Use native block factors when dense block arithmetic and
block-level fill control matter, and use block matvec directly when
factorization is not needed.

## Generated code and external buffers

Generated code can keep its own arrays and expose them through `Map` or
`StridedMap`:

```rust
let jacobian = stack_algebra::Map::<6, 3, f32>::from_slice(jacobian_data)
    .expect("generated buffer has the expected size");
let factor = jacobian
    .col_piv_householder_qr();
let step = factor.solve_least_squares(&rhs);
```

This is an ordinary Rust API boundary. The library does not know whether the
values came from generated code, a sensor driver, or a hand-written model.

## Embedded MCU deployment

Recommended sequence:

1. Select `f32` unless `f64` is available and numerically justified.
2. Keep dimensions fixed or bounded by `MatrixBuf`.
3. Query `storage_bytes()` for every long-lived factor and workspace.
4. Use `*_into` and `solve_in_place` to avoid temporary outputs in loops.
5. Run the scalar QEMU harness for Cortex-M and RISC-V targets.
6. Measure real hardware timing separately; QEMU does not model peripheral or
   MCU floating-point timing accurately.

Example resource declaration:

```rust
use stack_algebra::{Matrix, MatrixBuf};

const STATE_BYTES: usize = Matrix::<15, 1, f32>::storage_bytes();
const COVARIANCE_BYTES: usize = Matrix::<15, 15, f32>::storage_bytes();
const WORK_BYTES: usize = MatrixBuf::<32, 32, f32>::storage_bytes();
```

## What is not currently covered

- Heap-backed dynamic matrices and runtime-sized decompositions.
- Native scalar pivots that cross block boundaries; use the explicit
  `try_dense_ldlt` fallback for those cases.
- Scalar Bunch–Kaufman 1x1/2x2 LDLT pivots are supported for dense fixed-size
  systems and inside native block-sparse diagonal blocks.
- Automatic mixed-precision expressions.
- Framework-specific adapters or code-generation integrations.

These boundaries are intentional. They keep the current core standalone,
`no_std`, allocation-free, and suitable for compile-time resource analysis.
