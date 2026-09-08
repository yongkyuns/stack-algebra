# Unsafe code policy

`stack-algebra` is primarily safe Rust, but a small amount of `unsafe` is used where fixed-layout matrices, zero-copy views, or architecture-specific kernels need lower-level operations.

## Policy

Every unsafe boundary in `src/` must have a local safety argument and must preserve the crate's public safe API invariants. New unsafe code is not accepted as an implementation detail that is invisible to review: a pull request that adds an `unsafe` token under `src/` must also update this document.

The required review questions are:

1. **Why is unsafe required?** Prefer a safe implementation unless the unsafe boundary provides a concrete layout, interoperability, or measured kernel benefit.
2. **What invariant makes the operation valid?** State alignment, initialization, lifetime, aliasing, bounds, target-feature, or representation assumptions next to the unsafe operation.
3. **Can safe callers violate the invariant?** If yes, the API is not sound. Unsafe preconditions must not leak through a safe public entry point.
4. **How is the boundary validated?** Extend focused tests and Miri coverage when the operation is visible to Miri. Architecture intrinsics that Miri cannot execute require target-specific tests or compile coverage.
5. **What happens on unsupported layouts or targets?** Prefer an explicit safe fallback rather than widening the unsafe precondition.

## Current boundary categories

### Fixed matrix representation and zero-copy reinterpretation

`Matrix` is `#[repr(C)]` over nested fixed-size arrays. A few internal paths reinterpret a correctly sized, initialized column-major slice as a fixed matrix reference so mapped buffers can reuse the same optimized kernels as owning matrices.

Safety depends on exact element count, compatible alignment, initialized storage, the `repr(C)` matrix representation, and the returned reference never outliving the borrowed slice. These paths must reject incompatible strided layouts rather than reinterpret them.

Owning `Matrix::as_slice` and `Matrix::as_mut_slice` use the safe
`as_flattened` / `as_flattened_mut` APIs on the nested arrays, rather than
constructing slices from raw pointers. These APIs are available below the
Rust 1.87 MSRV, preserve column-major borrowing without `Copy` or `Clone`
bounds, and reject overflowing zero-sized slice lengths in release builds too.

### Dense matrix initialization

`Matrix<MaybeUninit<T>>::uninit` uses the existing safe `Matrix::from_fn`
constructor, which builds nested arrays with `core::array::from_fn`. Creating
uninitialized slots no longer requires an `assume_init` boundary or any trait
bound on `T`.

The subsequent conversion from initialized slots into `Matrix<T>` remains an
unsafe operation: every slot must contain a valid `T`, and ownership must be
transferred exactly once. Iterator collection retains its initialized-prefix
guard so short or panicking iterators drop all produced values without reading
uninitialized slots or double-dropping them.

`tests/matrix_storage.rs` covers column-major shared/mutable borrowing, empty
shapes, non-`Copy` values, over-aligned zero-sized elements, slice-length
overflow, and collection/drop behavior on success, shortage, and panic. This
suite runs with the normal tests, in a focused release check, and under Miri.

### Unchecked indexing

Public `unsafe` unchecked-access methods expose the usual caller obligation that indices are in bounds. Internal unchecked indexing must only be used after bounds have been established by compile-time dimensions or an explicit preceding check.

### SIMD and target-specific kernels

Architecture-specific kernels may use unsafe intrinsics, pointer loads/stores, or target-feature entry points. Safety depends on valid matrix storage ranges, appropriate target-feature dispatch, and respecting alignment/load requirements of the selected intrinsic. Portable scalar fallbacks remain the reference behavior.

The safe, publicly callable `FactorizationScalar` and `MatrixScalar` hooks
validate dynamic slice lengths, block ranges, and column indices before entering
unchecked kernels. These checks are unconditional, including in release builds,
and reject invalid arguments before any output mutation. The portable defaults
use the same checks so behavior does not change with the selected ISA. Hiding a
method from rustdoc does not make it an unsafe API.

`tests/scalar_hook_contracts.rs` exercises these boundaries for f32, f64, and the
portable i32 defaults, including short/long/empty slices, packet tails, invalid
ranges, and `usize::MAX`. The x86 matrix runs this suite in debug and release;
Miri and native ARM64 include it alongside the existing numerical contracts.

### Sparse fixed-capacity storage initialization

Sparse storage occasionally uses lower-level initialization techniques to avoid heap allocation while constructing fixed-capacity buffers. Every element must be initialized before it is observed, and capacity/length bookkeeping must prevent reads beyond the initialized prefix. Sparse and mapped-view suites are included in Miri CI specifically to exercise these invariants.

### Cached sparse LDLT schedules

Symbolic analysis alone does not validate cached source indices for every
subsequent numeric matrix. The safe LDLT entry points accept independently
constructed canonical CSC matrices, potentially with different capacities,
column starts, upper-triangle storage, and lower coordinates. Factor-pattern
coverage is not proof that a cached numeric source offset remains valid.

Before entering the unchecked aggregate kernel, `matches_aggregate_input`
checks each cached diagonal and off-diagonal source index against the current
column range and row coordinate. It also compares the total lower-entry count.
Canonical coordinate uniqueness makes these exact checks sufficient to rule
out missing, moved, or added lower entries; no hash or pointer identity is used.
The check uses safe borrowed slices and adds no retained storage or allocation.

A changed source layout takes the existing checked left-looking algorithm,
which validates factor coverage before modifying reusable output. Ordered
LDLT recomputation installs the supplied symbolic lower pattern before taking
raw pointers to output values, even when the destination previously held a
different factor. Numeric-failure rollback is not promised by these in-place
APIs; structural mismatch is rejected before output mutation.

`tests/sparse_reuse_contracts.rs` covers empty/smaller input storage, changed
coordinates with equal nnz, full-versus-lower layouts, matching numeric updates,
ordered and reordered calls, initialized and uninitialized destinations, f32,
and structural-error output preservation. It runs in native tests, x86 release
checks, Miri, and the isolated extracted-package consumer. The regression-first
baseline demonstrated a Miri invalid read from an empty numeric slice.

## Validation

The repository currently uses several complementary checks rather than treating `unsafe` review as sufficient by itself:

- Miri runs matrix-storage, mapped-view, and sparse safety suites;
- x86/SSE2 and native ARM64 jobs exercise architecture-specific dispatch;
- Cortex-M, RISC-V, AArch64, and WASM `no_std` builds cover portability;
- QEMU executes representative embedded workloads;
- numerical and public-API contracts verify that optimized paths retain the same observable results as the safe reference behavior.

The `Unsafe audit` workflow enforces the review policy for new pull requests. If a PR adds an `unsafe` token to `src/**/*.rs`, it must also update this file so the new boundary and its validation can be reviewed explicitly.

## Exact-layout sparse reuse fast path

Fill-free lower input can be checked by exact CSC array equality instead of
repeating per-entry searches. Factor coverage is proved by equal active entry
counts, column starts, and row indices; unused capacity is not read.

For LDLT source-map reuse, equality with the factor alone is not sufficient.
The fast path also requires that the aggregate source count equals the factor
off-diagonal count and every cached source diagonal equals its factor column
start. All analyzed lower entries are contained in the factor; equal counts
therefore rule out fill. The diagonal checks establish that all diagonals exist
and that upper entries have not shifted source offsets. Under these conditions
the analyzed input is exactly the factor's lower CSC layout, so the direct
current-layout comparison proves every cached source coordinate and offset.
Missing diagonals, fill, and full-storage schedules retain the entry-wise check.
No hashes, pointer identity, new retained metadata, or unchecked loads are used.

The internal reuse-validation tests independently compare both validators with
coordinate-lookup references for 192 canonical 2x2 source/input pairs and
110,592 canonical 3x3 pairs, including missing diagonals and upper-only numeric
inputs. Native debug/release checks execute both; Miri executes the complete
2x2 enumeration alongside all existing public sparse reuse/storage suites.
