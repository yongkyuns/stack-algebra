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

### Unchecked indexing

Public `unsafe` unchecked-access methods expose the usual caller obligation that indices are in bounds. Internal unchecked indexing must only be used after bounds have been established by compile-time dimensions or an explicit preceding check.

### SIMD and target-specific kernels

Architecture-specific kernels may use unsafe intrinsics, pointer loads/stores, or target-feature entry points. Safety depends on valid matrix storage ranges, appropriate target-feature dispatch, and respecting alignment/load requirements of the selected intrinsic. Portable scalar fallbacks remain the reference behavior.

### Sparse fixed-capacity storage initialization

Sparse storage occasionally uses lower-level initialization techniques to avoid heap allocation while constructing fixed-capacity buffers. Every element must be initialized before it is observed, and capacity/length bookkeeping must prevent reads beyond the initialized prefix. Sparse and mapped-view suites are included in Miri CI specifically to exercise these invariants.

## Validation

The repository currently uses several complementary checks rather than treating `unsafe` review as sufficient by itself:

- Miri runs mapped-view and sparse safety suites;
- x86/SSE2 and native ARM64 jobs exercise architecture-specific dispatch;
- Cortex-M, RISC-V, AArch64, and WASM `no_std` builds cover portability;
- QEMU executes representative embedded workloads;
- numerical and public-API contracts verify that optimized paths retain the same observable results as the safe reference behavior.

The `Unsafe audit` workflow enforces the review policy for new pull requests. If a PR adds an `unsafe` token to `src/**/*.rs`, it must also update this file so the new boundary and its validation can be reviewed explicitly.
