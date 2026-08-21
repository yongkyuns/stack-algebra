# 0.3 stabilization plan

Status: active implementation plan

This plan turns the current breadth of `stack-algebra` into a tighter, measurable
contract before adding more major algorithms. The library already contains a
substantial dense, bounded, sparse, block-sparse, geometry, SIMD, and embedded
validation surface. The immediate risk is therefore stabilization debt rather
than missing feature count.

The target position is:

> Predictable linear algebra for embedded and robotics workloads: small-to-medium
> compile-time or tightly bounded matrices, explicit storage and workspace reuse,
> zero-copy caller buffers, and fixed-capacity sparse/block-sparse solvers that
> work in `no_std` environments.

The project should not try to become a general replacement for Eigen, nalgebra,
or faer across dynamic large dense and sparse workloads.

## Release objective

`0.3` is a stabilization release. Major algorithm expansion is paused until the
existing API, numerical behavior, memory/resource envelope, and performance
claims have release-quality evidence.

The release should establish four contracts:

1. **API contract** — normal usage has predictable naming and failure behavior,
   and accidental public API breakage is caught in CI.
2. **Numerical contract** — correctness is specified by reconstruction,
   residual, orthogonality, rank, pivot, and convergence invariants rather than
   element-for-element agreement with one reference implementation.
3. **Resource contract** — representative operations have measured storage,
   stack, code-size, and target evidence.
4. **Scope contract** — unsupported workloads are stated directly instead of
   being implied by broad Eigen/faer parity language.

## Priority 0 — freeze and measure the current surface

### API/release gates

- Add a pull-request semver check against the PR base revision.
- Keep compile-time rejection tests for incompatible dimensions and scalar types.
- Expand public API smoke coverage across dense, bounded, view, solver, geometry,
  and representative sparse entry points.
- Add an MSRV only after verifying the real dependency/toolchain floor on CI.
- Before the `0.3.0` release, capture a machine-readable public API snapshot or
  equivalent release artifact.

### Numerical specification

- Treat invariant tests as the primary numerical specification.
- Test dimensions around packet boundaries, not only 3x3 and 4x3 examples.
- Cover `f32` and `f64`, multiple scales, singular/rank-deficient cases,
  non-finite values, and multiple right-hand sides.
- Exercise owned, mapped, strided, block, and bounded-view paths with the same
  mathematical inputs.
- Keep Eigen as differential evidence; add a second independent reference where
  it materially improves confidence.

### Benchmark/release evidence

- Keep short GitHub-hosted nightly runs as regression triage rather than release
  performance evidence.
- Release benchmarks should run longer on a pinned machine and record CPU,
  compiler, dependency versions, ISA flags, allocation/setup semantics, sample
  configuration, and correctness prechecks.
- Report performance by operation/phase/shape/scalar/target; do not aggregate
  unlike operations into a general "faster than Eigen" conclusion.

## Priority 1 — clean the public contract

### Scalar/kernel separation

`MatrixScalar` and `ReductionScalar` currently serve both as public scalar bounds
and as dispatch hooks for optimized implementation details. Before `0.3` is
considered stable:

- define the supported external scalar contract explicitly;
- keep floating-point ISA dispatch private;
- separate matrix-product, reduction, triangular, and factor-update kernel
  families internally;
- allow custom scalars to use a portable path without requiring downstream code
  to understand SIMD/backend mechanics.

This is a breaking-API candidate and should be implemented deliberately with a
migration note rather than incrementally leaking more backend hooks.

### Failure and mutation conventions

Adopt predictable conventions:

- `Result` when the failure reason is actionable;
- `Option` only when absence has one obvious meaning;
- `_into` for caller-provided distinct output;
- `_in_place` when an input/output is overwritten;
- `compute`/`recompute` consistently for factor reuse;
- checked constructors by default for invariant-bearing objects.

Specific cleanup targets:

- add safer bounded resize forms (`resize_zeroed`, `resize_with`) before
  deprecating or renaming storage-preserving growth semantics;
- make sparse capacity errors report required and available capacity;
- make normalization and geometry validity failures explicit and scale-aware;
- avoid determinant-vs-machine-epsilon singularity tests when a factor-relative
  criterion is available.

## Priority 2 — improve real workload data flow

Do **not** build a general Eigen-style expression-template system.

Instead add a small set of fused operations that map directly to estimation and
control loops, such as:

- `axpy_into` / scaled accumulation;
- linear-combination output operations;
- GEMM-like `C = alpha * A * B + beta * C`;
- reusable multi-RHS solve/update paths where profiling shows material benefit.

These operations should preserve the library's explicit-memory model while
eliminating avoidable temporary matrices in hot loops.

## Priority 3 — make zero-copy views a performance path

Maps, blocks, and compatible strided views should be able to reach optimized
kernels without first materializing an owned matrix.

Introduce internal layout capabilities such as:

- contiguous column-major;
- contiguous with a leading dimension;
- arbitrary stride.

Safe public construction should establish bounds/aliasing invariants once;
inner loops should then use trusted in-bounds access. Owned and compatible mapped
inputs should converge on the same kernel families where layout permits it.

## Priority 4 — sparse/block-sparse ergonomics

The bounded sparse layer is most valuable when capacity planning is explicit and
diagnostic.

- Report `required` versus `capacity` for pattern/fill exhaustion.
- Keep symbolic analysis separate from numeric recomputation and solve timing.
- Document lower/upper triangle semantics and pivot models precisely.
- Test adversarial fill patterns and repeated symbolic/numeric reuse.
- Keep global/cross-block pivot fallback explicit; do not hide dense fallback
  behind an API that looks purely sparse.
- State when a runtime sparse solver from another library is the better tool.

## Priority 5 — prove the embedded advantage

QEMU and cross-compilation are necessary but not sufficient evidence for an
embedded-focused library.

Qualify at least:

- one Cortex-M FPU target (for example M4F/M7/H7 class); and
- one maintained RISC-V or ESP-class target if hardware is available.

For representative 3x3, 6x6, 15x15, and bounded sparse/block-sparse workloads,
record:

- cycles or wall time under a controlled clock;
- peak stack for the measured call path;
- factor/workspace/static storage size;
- `.text` and `.data/.bss` contribution;
- compile time/codegen impact;
- `f32`/`f64` behavior where hardware support differs.

Compare static nalgebra and CMSIS-DSP where the comparison is meaningful. This
is the evidence most likely to establish a durable project advantage.

## Deliberately deferred

Until a real workload demonstrates need, continue to defer:

- heap-owning fully dynamic matrices;
- general expression templates;
- general runtime sparse indefinite solving;
- GPU/accelerator backends;
- per-shape hand-written kernels;
- new ISA families without maintained hardware and benchmarks;
- broad geometry expansion unrelated to the linear-algebra core.

## Implementation sequence

### Slice A — release and contract gates

- [x] Document this stabilization plan.
- [x] Add a PR semver/API compatibility gate against the base revision.
- [x] Add invariant coverage around small-matrix/packet-boundary dimensions.
- [ ] Expand the curated public API smoke test to bounded, geometry, and sparse
      entry points.
- [ ] Add release benchmark metadata/version pinning beyond the nightly triage
      workflow.

### Slice B — bounded and sparse diagnostics

- [ ] Add `MatrixBuf::resize_zeroed`.
- [ ] Add `MatrixBuf::resize_with`.
- [ ] Preserve current `resize` semantics through `0.2.x`; decide the `0.3`
      naming/deprecation path explicitly.
- [ ] Add detailed sparse capacity exhaustion errors.
- [ ] Add tests that distinguish capacity exhaustion, unsupported pivot model,
      and numerical failure.

### Slice C — public scalar/kernel boundary

- [ ] Define the supported scalar extension contract.
- [ ] Move ISA/factor-update hooks behind private kernel families.
- [ ] Add compile tests for one representative external scalar implementation.
- [ ] Record the migration as an intentional `0.3` API change.

### Slice D — views and fused operations

- [ ] Add internal layout classification for owned/maps/strided views.
- [ ] Route compatible views through optimized kernels.
- [ ] Add `axpy_into`/linear-combination primitives.
- [ ] Add a GEMM-like accumulate primitive only after benchmark evidence.

### Slice E — target qualification

- [ ] Establish a repeatable real Cortex-M benchmark harness.
- [ ] Publish resource tables for representative dense and sparse workloads.
- [ ] Add a second real target when maintainable.

## Release gate for 0.3

Do not call `0.3` stable until:

- semver/API changes are intentional and documented;
- every public solver has invariant-based numerical coverage;
- failure semantics are predictable for the common APIs;
- bounded/sparse capacity failures are actionable;
- release benchmark methodology is reproducible;
- at least one real embedded target has resource measurements; and
- the README/docs describe the supported workload envelope without implying
  general Eigen/faer replacement.
