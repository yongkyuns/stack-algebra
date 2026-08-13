# Tutorials

These tutorials are organized by the data layout and numerical operation that
drive a design. Start with the smallest matching path, then open the linked API
reference for method signatures and trait bounds.

## Fixed-size dense algebra

Use `Matrix<M, N, T>` when dimensions are known at compile time. The path is:

1. Construct a matrix with `matrix!`, `Matrix::zeros`, or `Matrix::eye`.
2. Compose products and reductions with the operators and `mul_into` methods.
3. Choose a decomposition from the matrix assumptions.

See [Getting started](getting-started.md), [API usage](api-usage.md), and the
[`Matrix` API](api-reference.md).

## Views and external buffers

Use `Map` for contiguous column-major storage and `StridedMap` when row or
column spacing is supplied by another system. Use `Block` for fixed-size
submatrices without copying. The view types borrow their source for the view's
lifetime; use `Matrix::from_view` only when an owned snapshot is intentional.

See [API usage — external buffers and views](api-usage.md) and the generated
[view APIs](api-reference.md).

## Dense factorizations

Select a factorization from the input assumptions:

- `Cholesky` for symmetric positive-definite systems.
- `Ldlt` for symmetric systems that may be indefinite.
- `PartialPivLu` for general square systems.
- `HouseholderQr` or `ColPivHouseholderQr` for least-squares systems.
- `Svd` when rank information or a robust pseudoinverse is required.

The [solver guide](api-usage.md) describes failure behavior, factor reuse, and
output-reuse methods.

## Geometry

Use `Quaternion`, `AngleAxis`, and `RotationMatrix` for rotations; use
`Isometry` for rigid transforms and `AffineTransform` for general affine
transforms. Keep the scalar type explicit and convert at boundaries with
`cast`. The [feature set](features.md) lists the available representations and
the [use-case guide](use-cases.md) shows how they compose with dense matrices.

## Sparse and block-sparse systems

Use `StaticCscPattern` and `StaticCscMatrix` when a scalar sparsity pattern is
known. Use block sparse storage when repeated fixed-size blocks describe the
problem more naturally. Build or reuse the symbolic pattern before numeric
factorization; see [API usage — sparse storage](api-usage.md) and [use cases —
sparse systems](use-cases.md).

## Embedded and bounded workflows

The fixed-size core is `no_std` and does not require a heap allocation. Use
bounded storage when active dimensions vary within a compile-time limit, and
map caller-owned memory when the buffer belongs to a device or driver. The
[feature set](features.md) and [use cases](use-cases.md) describe the supported
boundaries; target-specific validation remains separate from the API guide.
