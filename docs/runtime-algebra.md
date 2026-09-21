# Runtime-shaped borrowed algebra

Use `runtime::MatrixRef` and `runtime::MatrixMut` when dimensions arrive at runtime
and the caller already owns the storage. They borrow slices; they are not growing
heap-backed matrix owners. Allocation, storage growth, and workspace policy stay
with the application. The existing fixed-size APIs and `MatrixBuf` are unchanged.

## Layout and ownership

`from_column_major` and `from_row_major` borrow packed storage. `from_strides`
accepts nonnegative row and column strides, measured in elements, with address
`row * row_stride + column * column_stride`. This covers padded leading dimensions,
row-major input, and transposed layouts. Extra trailing capacity is ignored.

Read-only views can repeat elements. Mutable views reject repeated logical
addresses, including collisions between nonzero strides, not just zero strides.
Only safe slices and references are used; the runtime module forbids unsafe code.

`transpose()` swaps shape and stride metadata without copying. `submatrix()`
checks the requested rectangle and preserves the parent's strides. Mutable
transformations consume the view; `reborrow()` keeps a parent usable after the
child expires. `as_ref()` gives a read-only view tied to the current parent borrow.
Safe borrowing prevents an output from aliasing a live input or parent view.

An empty rectangle addresses no elements, even with very large inactive
strides. Empty submatrices at a valid end boundary do not compute nonexistent
addresses. Nonempty shape products, extents, and submatrix bounds are checked
for overflow. The `runtime::Error` variants distinguish overflow, short storage,
overlapping mutable elements, out-of-bounds blocks, and operation shape errors.

## Operations

`copy_into`, `transpose_into`, and `matmul_into` write only active elements of a
caller-provided destination. Padding and trailing capacity remain untouched.
All structural checks happen before writes: a returned error leaves the entire
destination unchanged. Matrix multiplication overwrites rather than accumulates;
a zero inner dimension produces zero active output, and an empty output returns
immediately. Input matrices may share storage with one another.

```rust
use stack_algebra::runtime::{matmul_into, MatrixMut, MatrixRef};

let a = [1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0];
let x = [2.0_f32, -1.0, 0.5];
let mut y = [0.0; 2];
matmul_into(
    MatrixRef::from_column_major(&a, 2, 3).unwrap(),
    MatrixRef::from_column_major(&x, 3, 1).unwrap(),
    &mut MatrixMut::from_column_major(&mut y, 2, 1).unwrap(),
).unwrap();
assert_eq!(y, [1.5, 6.0]);
```

Multiplication uses ordinary scalar operators in increasing inner-index order.
Nonfinite floating-point values propagate; there is no numerical-failure check
or rollback on a scalar panic. No expression templates, dynamic backend traits,
implicit temporary allocation, or SIMD performance claim is introduced.

## Qualification and scope

The runtime integration tests compare varying rectangular products with an
independent nalgebra reference and check padding canaries, transposed subblocks,
empty dimensions, input aliasing, failed admission, and caller-storage reuse.
An exhaustive small-layout oracle checks mutable admission against distinct
addresses. Rustdoc compile-fail cases cover input/output and parent/child borrows.
A separate no_std consumer instantiates f32 multiplication and f64 transpose on
Rust 1.87 and stable, including Cortex-M and WASM compilation. These are test
contracts; passing build results are recorded per pull-request revision.

Runtime partial-pivot LU, reusable factors, numerical failure/invalidation, and
multiple-RHS solves remain a separate next slice. This API does not yet replace
RustRobotics' dynamic SLAM matrices or remove its nalgebra dependency. It imposes
no new graph/landmark limit and does not substitute sequential EKF updates for
batch updates. A growing matrix owner and true sparse graph factorization are
not part of this slice.
