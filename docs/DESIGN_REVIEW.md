# Design Review and Improvement Plan

Status: design review for the `0.3` development cycle
Review date: 2026-08-07
Scope: architecture, numerical behavior, safety, performance, API design,
documentation, and suitability for robotics workloads on OS-hosted and MCU
targets.

This document reviews the current implementation rather than defining
compatibility with another library. Comparisons are used to identify useful
design choices and workload boundaries. They are not a requirement to copy
another API or implementation.

## 1. Executive assessment

The repository has a coherent fixed-size foundation:

- `Matrix<M, N, T>` has inline, column-major storage and compile-time shape.
- The crate is `no_std` by default and has a small runtime dependency graph.
- Dense factors, external-buffer views, bounded storage, fixed-capacity sparse
  storage, block-sparse storage, and 3D geometry use the same scalar and matrix
  model.
- Dense and sparse factors support reuse, which is important when a matrix
  pattern is fixed while values change.
- Portable scalar kernels, x86 packet kernels, AArch64 NEON kernels, Eigen
  differential tests, Miri jobs, cross-target builds, and QEMU smoke tests are
  already present.

The main issue is not a missing list of algorithms. It is that several design
goals are currently treated as universally beneficial when they are only
beneficial for a particular workload range.

The library is presently a reasonable fit for:

- small dense matrices whose dimensions are known at compile time;
- fixed-capacity systems whose active dimensions or sparsity remain within a
  known bound;
- reusable factorizations in deterministic control or estimation loops;
- generated numerical kernels that operate on caller-owned buffers; and
- `no_std` firmware where heap allocation is unavailable or intentionally not
  used.

It is not presently a general substitute for Eigen, nalgebra, faer, or a
production sparse solver across their full domains. In particular:

- fixed-size-only storage is a poor default for large matrices on an OS;
- fixed-capacity sparse types become difficult to size when graph topology and
  fill-in are not bounded tightly;
- large fixed factors can consume substantial task stack, static RAM, flash,
  and compile time;
- compile-time ISA selection is appropriate for firmware and locally tuned
  binaries, but not sufficient for one portable OS binary distributed across
  different CPUs; and
- the current performance report is broad but does not yet constitute a
  reproducible performance contract.

The recommended direction is therefore:

> Keep a small, fixed-size, inline-storage `no_std` core. Define its supported
> workload range precisely. Make safety invariants and numerical contracts
> explicit. Treat runtime-sized allocation, runtime CPU dispatch, and large
> sparse systems as optional layers or separate future work justified by real
> workloads, not as requirements for the core.

“Eigen parity” should not be used as a release criterion. Eigen itself uses
fixed and dynamic storage, expression templates, multiple allocation modes,
packetization, and a large sparse subsystem. The useful criterion is parity of
specified operations within a declared shape, scalar, target, and numerical
contract.

## 2. Evidence reviewed

The review is based on the current repository at commit `5cdd0c5`, including
the uncommitted nightly benchmark workflow changes present during the review.

Repository evidence:

- approximately 17,400 lines of Rust under `src/`;
- the largest implementation files are `block_sparse.rs` (about 2,300 lines),
  `algebra/qr.rs` (about 1,350), `kernels/x86/avx2_fma.rs` (about 1,080),
  `view.rs` (about 1,030), and `algebra/ldlt.rs` (about 940);
- six Criterion benchmark binaries plus a native Eigen runner;
- more than 1,800 lines of Eigen differential tests;
- cross-builds for Cortex-M, RISC-V, AArch64, and WASM;
- QEMU execution for Cortex-M4, RISC-V32, and AArch64; and
- targeted Miri execution for view and sparse integration tests.

External comparisons use official documentation:

- [Eigen matrix types and fixed/dynamic storage](https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html)
- [Eigen external-buffer maps and strides](https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html)
- [Eigen sparse storage](https://eigen.tuxfamily.org/dox/group__TutorialSparse.html)
- [nalgebra matrix storage and dimensions](https://www.nalgebra.rs/docs/user_guide/vectors_and_matrices/)
- [nalgebra embedded and `no_std` behavior](https://www.nalgebra.rs/docs/user_guide/wasm_and_embedded_targets/)
- [faer crate design and intended matrix sizes](https://docs.rs/faer/latest/faer/)
- [glam types and SIMD storage](https://docs.rs/crate/glam/latest)
- [CMSIS-DSP matrix functions](https://arm-software.github.io/CMSIS-DSP/main/group__groupMatrix.html)
- [sprs sparse matrix support](https://docs.rs/sprs/latest/sprs/)
- [nalgebra-sparse scope and limitations](https://docs.rs/nalgebra-sparse/latest/nalgebra_sparse/)
- [micromath embedded approximation tradeoffs](https://docs.rs/micromath/)

The external documentation represents current releases or current online
documentation, while this repository benchmarks specific dependency versions
from `Cargo.toml`. Version changes must be recorded with future benchmark
results.

## 3. Workload suitability

### 3.1 Current fit by workload

| Workload | Current fit | Reason |
| --- | --- | --- |
| MCU attitude, calibration, small filters, and control | Good for supported algorithms | Shapes are small and usually fixed; storage and execution can be budgeted. |
| OS-hosted small generated kernels | Good | Const dimensions, explicit output reuse, and external maps are useful. |
| Small dense least-squares or covariance solves | Good with validation | QR, pivoted QR, Cholesky, LDLT, LU, SVD, and self-adjoint eigen are available, but numerical contracts need stronger tests. |
| Medium or large dense matrices | Weak | Full unrolling/monomorphization and inline factors stop being clear advantages; faer or dynamic Eigen/nalgebra are more suitable. |
| Fixed-topology sparse estimation or optimization | Promising but specialized | Symbolic/numeric separation and fixed capacity are useful if fill bounds are known. API and capacity planning are complex. |
| Runtime pose graphs, mapping, or variable-horizon optimization | Weak | Dimensions, topology, and fill are commonly runtime values. Fixed capacities can waste memory or reject valid workloads. |
| General 3D application geometry | Adequate but narrow | Basic quaternion, rotation, isometry, angle-axis, and affine types exist. nalgebra has a much broader geometry/type ecosystem; glam is more specialized for small SIMD geometry. |
| Cortex-M fixed/floating-point DSP kernels | Mixed | Portable Rust is useful, but CMSIS-DSP has architecture-specific kernels and fixed-point formats that this crate does not match. |
| One portable x86 OS binary | Incomplete | Current optimized backend selection depends on compile-time target features; no runtime multiversion dispatch exists. |

### 3.2 OS-hosted robotics

On Linux, Windows, macOS, or a larger RTOS with allocation, deterministic
execution does not require every value to be an inline fixed-size type. A
common safe pattern is to allocate or reserve storage during initialization,
separate symbolic analysis from numeric updates, and reuse those buffers in
the real-time loop.

For these targets, the library has merit as a small-matrix kernel library and
as an external-buffer computation layer. It does not yet have a compelling
advantage for large dense matrices or runtime sparse graphs. A complete system
may reasonably use this crate for small fixed blocks and another library for a
large runtime-sized outer solve.

### 3.3 MCU robotics

On MCUs, inline storage and `no_std` support are directly useful, but
“allocation-free” is only one resource constraint. The relevant budgets are:

- peak stack per call and per task;
- total static RAM;
- flash/code size after monomorphization;
- cycles and worst-case execution time;
- floating-point support and instruction latency;
- alignment and DMA/external-buffer constraints; and
- whether `f64` is implemented efficiently in hardware.

The type `Matrix<M, N, T>` is inline, not inherently stack-resident. It may be
stored as a local, static, field, arena member, or heap allocation supplied by
the application. Documentation and naming should use “inline storage” when
describing the representation, and reserve “stack” for measured placement in
a particular program.

QEMU demonstrates that selected binaries execute, but it does not establish
real-device timing, peripheral interaction, cache behavior, FPU configuration,
or exact stack consumption for every algorithm. Hardware results should be
required before making target-specific performance claims.

## 4. Revisited design assumptions

| Assumption | Assessment | Decision |
| --- | --- | --- |
| Fixed size should remain the core matrix model. | Sound for the intended small-matrix core. | Keep. Define an intended size range and do not imply it is best for large matrices. |
| The crate should be allocation-free. | Sound as a core property, but not a universal robotics requirement. | Keep the core independent of `alloc`. Do not add low-value allocation policing. Permit caller placement and future opt-in layers. |
| Every matrix is “stack allocated.” | Inaccurate. Inline values are placed wherever their owner is placed. | Change terminology to “inline storage” or “fixed-capacity storage.” |
| Fixed capacity can replace runtime-sized matrices and sparse storage. | Only when maxima and fill are known tightly. | Keep `MatrixBuf` and static sparse types as bounded tools, not universal substitutes. |
| Compile-time SIMD selection is always sufficient. | Sufficient for firmware and target-specific native builds, not portable distributed binaries. | Keep it in `no_std`; evaluate optional `std` runtime dispatch with measured demand. |
| More algorithms move the library closer to stability. | Only if contracts, tests, API consistency, and maintenance scale with them. | Pause feature expansion until architecture and validation gates are complete. |
| Matching Eigen output proves correctness. | It is valuable differential evidence, not a specification or proof. Different stable algorithms may return equivalent factors with different signs, orderings, or pivots. | Define mathematical invariants and residual/error bounds first; use Eigen as one oracle. |
| Matching Eigen speed is the correct performance goal. | Too broad. Relative performance changes by operation, size, scalar, compiler, ISA, allocation model, and output semantics. | Define benchmark envelopes per workload and target. |
| A single matrix/view trait should be open to external implementations and fast in inner loops. | The current safe `Option` contract supports openness but adds checks and invalid-view branches to hot loops. | Split safe extensibility from trusted internal access. |
| Custom scalar support and architecture specialization should use the same public trait. | This leaks backend mechanics into the public API and makes custom scalar implementation unusually difficult. | Rework or seal scalar/backend traits in the next breaking release. |
| Fixed-capacity sparse LDLT should cover general sparse indefinite systems. | General pivoting may change structure and fill unpredictably. The existing dense fallback is explicit evidence of this boundary. | Document the supported pivot model; do not claim general sparse indefinite coverage. |

## 5. Architecture review

### 5.1 Current module structure

The major conceptual layers are:

1. `Matrix` storage, constructors, indexing, iteration, formatting, and dense
   arithmetic.
2. Borrowed views and bounded runtime-active storage.
3. Dense decompositions and structured dense views.
4. Scalar sparse and block-sparse storage, symbolic analysis, factorization,
   and solves.
5. Geometry types built on the dense core.
6. Portable and architecture-specific numerical kernels.
7. Differential tests, benchmarks, and target harnesses.

This layering is reasonable, but file boundaries do not consistently reflect
it. `block_sparse.rs` contains storage, symbolic analysis, numerical factors,
fallbacks, and tests in one large module. `lib.rs`, `view.rs`, `qr.rs`, and the
x86 FMA backend also carry several distinct responsibilities. `iter.rs`
contains a large amount of commented-out implementation, which obscures the
supported surface.

### 5.2 Recommended module boundaries

Keep one crate until there is a concrete reason to split packages, but organize
the implementation by invariant:

```text
src/
  matrix/
    mod.rs            owning inline matrix and aliases
    constructors.rs
    indexing.rs
    iter.rs
    ops.rs
    views.rs
    bounded.rs
  linalg/
    mod.rs
    error.rs
    triangular.rs
    self_adjoint.rs
    cholesky.rs
    ldlt.rs
    lu.rs
    qr.rs
    svd.rs
    eigen.rs
  sparse/
    mod.rs
    error.rs
    pattern.rs
    ordering.rs
    scalar/
      storage.rs
      cholesky.rs
      ldlt.rs
    block/
      storage.rs
      symbolic.rs
      cholesky.rs
      ldlt.rs
      fallback.rs
  geometry/
    mod.rs
    quaternion.rs
    rotation.rs
    transform.rs
  kernel/
    mod.rs
    scalar.rs
    x86/
    aarch64/
```

This is a refactor, not an API expansion. Move code only after tests identify
the invariant protected by each module. Avoid a single large move that makes
history and review difficult.

### 5.3 Public surface and kernel leakage

`MatrixScalar` and `ReductionScalar` are the public scalar bounds. Their
portable default methods allow a custom scalar implementation without naming a
kernel or ISA type. The concrete matrix and reduction backends are crate
private. Built-in `f32` and `f64` implementations override the hidden hooks to
select the target-specific kernels at monomorphization time.

Recommended direction for `0.3`:

1. Decide which scalar types are part of the supported public contract. At
   minimum this is `f32` and `f64`; integer matrix arithmetic may remain if its
   semantics are tested and documented.
2. Split generic matrix product kernels from factorization update kernels.
3. Keep portable generic arithmetic separate from optimized floating-point
   dispatch.
4. Add compile tests for supported external scalar implementations.

The goal is not one generic `KernelBackend`. It is a private dispatch layer
with cohesive kernel families and a public scalar API that does not expose ISA
selection.

### 5.4 Views and trusted access

`MatrixRead<M, N, T>` provides checked `get` for arbitrary coordinates and a
default `get_in_bounds` accessor for algorithms whose const-generic loop bounds
have already established a valid coordinate. Built-in views override the latter
with their direct layout calculation. Dense view arithmetic therefore has no
`Option` result or repeated optional access in its inner loops.

The default accessor keeps the trait safe and open: an external implementation
can use `get`, while an implementation with a direct layout can override the
in-bounds method. Safe `Map` and `StridedMap` constructors establish length,
stride, aliasing, and borrow-lifetime invariants before the view exists. This
avoids a public or crate-private unsafe extension trait solely for dense-view
performance.

Dense decompositions retain their typed `InvalidView` result path, because
their public checked constructors accept arbitrary `MatrixRead` implementations.
Property tests should continue to cover view construction, strided indexing,
and mutable aliasing cases.

### 5.5 Bounded storage

`MatrixBuf<MAX_ROWS, MAX_COLS, T>` is useful when active dimensions vary under
a known bound. It reserves the full rectangular capacity inline. That cost is
appropriate only when the capacity is reasonably dense and small.

Current `resize` does not clear newly exposed values. This is memory-safe but
easy to misuse numerically. Choose one explicit contract:

- `resize_zeroed`, which initializes newly active coordinates;
- `resize_with`, which takes an initializer; and
- optionally `set_shape_preserving_storage`, with a deliberately explicit name
  for the current stale-value behavior.

Returning `Option<()>` also loses the requested and maximum dimensions. A
small `ShapeError` should be shared by map and bounded constructors.

### 5.6 Sparse architecture

The scalar sparse layer has a good high-level separation between pattern,
ordering, symbolic factor pattern, numeric factor, recomputation, and solve.
That separation should become the model for the block-sparse module.

The fixed-capacity design requires the user to choose:

- input nonzero capacity;
- factor fill capacity;
- block nonzero capacity;
- scalar expansion capacity in fallback paths; and sometimes
- scalar dense dimensions for global pivot fallback.

This is unavoidable at some layer, but it should not dominate the ordinary
API. Add builders or analyzed capacity reports that produce actionable errors:

- required nonzeros versus provided capacity;
- required factor fill versus provided capacity;
- unsupported cross-block pivot versus capacity exhaustion; and
- recommended fallback options.

Do not hide a dense fallback behind an API that appears sparse. The current
explicit fallback naming is preferable. Document worst-case storage for every
factor type and measure it in examples.

## 6. Safety and soundness review

### 6.1 Current unsafe boundaries

The library's direct unsafe code is concentrated in four areas:

1. flattening `[[T; M]; N]` into contiguous slices;
2. uninitialized matrix construction with `MaybeUninit`;
3. converting slices into `Row`, `Column`, and `Stride`-based unsized views;
4. architecture intrinsics and target-feature functions; and
5. sealed unchecked indexing used by safe and unsafe matrix accessors.

The basic representations are defensible:

- nested arrays are contiguous and the matrix uses a stable C representation;
- `MaybeUninit<T>` has the required element layout;
- the `stride` crate's `Stride<T, S>` is explicitly `repr(transparent)` over
  `[T]`; and
- target-feature modules are selected by compile-time configuration.

However, soundness is distributed across files and external representation
assumptions. A future refactor could break an invariant without making the
unsafe boundary obvious.

### 6.2 Required safety work

Create a private `raw` or `representation` module containing the minimum
unsafe primitives:

- array flattening;
- initialized/uninitialized matrix conversion;
- slice-to-view casts; and
- target-specific load/store helpers.

Each primitive should state:

- layout assumptions;
- initialization requirements;
- aliasing requirements;
- target-feature requirements;
- valid length/stride formulas; and
- which safe constructor establishes them.

Then add the following gates:

- Miri tests for constructors, `from_fn`, iterator collection including panic
  cleanup, row/column views, maps, strided maps, mutable blocks, and unchecked
  indexing wrappers;
- property tests for every accepted and rejected stride/length combination;
- compile-time size/alignment assertions for matrix and view representations;
- sanitizer execution for host integration tests and C++ Eigen FFI tests; and
- scalar-reference differential tests for every SIMD kernel and tail length.

SIMD intrinsics cannot be validated by Miri. Their memory accesses still need
host sanitizer tests, while instruction selection needs separate compile and
execution jobs for SSE2, AVX2, AVX2+FMA, and NEON.

### 6.3 Unsafe code that should be removed

`Matrix<3, 1, T>::cross` uses uninitialized construction for only three output
elements. A safe `from_columns` or `from_fn` construction is clearer and should
compile equivalently.

Large commented-out unsafe iterator and cloning implementations should be
deleted or restored as tested code. Dead unsafe examples make the actual
safety surface harder to audit.

### 6.4 Semantic invariants

Memory safety is only one part of a numerical type's soundness. Types described
as validated must preserve their mathematical invariant.

Review in particular:

- `Quaternion::from_rotation_matrix` accepts a raw matrix and returns a
  normalized quaternion, while `RotationMatrix::from_matrix` performs a
  stronger round-trip validation. Prefer accepting `&RotationMatrix<T>` in the
  infallible conversion and give raw-matrix projection/validation a distinct,
  explicit API.
- `Matrix::normalize` divides by the norm without reporting zero or non-finite
  input. Add `try_normalize`/`normalized` semantics and decide whether the
  panicking or IEEE-propagating convenience method should remain.
- geometry and decomposition thresholds should be scale-aware and documented
  in terms of the invariant they check, not only machine epsilon constants.

## 7. Numerical design review

### 7.1 Positive aspects

- Stable scaling is already used in several norms and Householder paths.
- Dense LDLT includes Bunch-Kaufman-style 1x1/2x2 pivots.
- Pivoted and unpivoted algorithms are distinct rather than silently changing
  robustness.
- QR and SVD expose rank thresholds.
- iterative decompositions report non-convergence.
- sparse analysis and numeric recomputation are separable.

### 7.2 Missing contracts

Each solver needs a short numerical contract covering:

- accepted matrix shape and structure;
- which triangle is read;
- whether symmetry is assumed or checked;
- pivoting strategy and threshold interpretation;
- rank definition;
- convergence budget;
- NaN/infinity handling;
- ordering/sign ambiguity of factors and vectors;
- expected residual measure; and
- whether the algorithm is intended for `f32`, `f64`, or both at the tested
  dimension range.

Matching Eigen element-for-element is not always mathematically meaningful.
For QR, SVD, and eigendecomposition, correctness tests should prioritize:

- reconstruction error;
- orthogonality error;
- residual error;
- singular/eigenvalue ordering contract;
- rank under an explicit threshold; and
- agreement of solved systems.

Differential tests can then compare invariant quantities and only compare raw
factor storage where the storage convention is part of the public contract.

### 7.3 Validation matrix

For every dense factor, test a generated grid of:

- dimensions including zero where supported, one, packet-width boundaries,
  rectangular tall/wide cases, and the largest declared supported size;
- `f32` and `f64`;
- well-conditioned, ill-conditioned, rank-deficient, singular, and non-finite
  inputs;
- scales near minimum normal, maximum finite, and mixed magnitudes;
- one and multiple right-hand sides; and
- owned, mapped, strided, block, and bounded views.

Use deterministic property generation so failures are reproducible. Compare
against more than one independent implementation where practical. Eigen
parity alone can reproduce the same algorithmic convention or hide a shared
assumption.

## 8. Performance review

### 8.1 Current kernel architecture

The portable matrix product has an appropriate column-major loop order: reuse
one right-hand-side scalar while walking contiguous left-hand-side and output
columns. Architecture modules provide SSE2, AVX2, AVX2+FMA, and AArch64 NEON
implementations. Scalar tails make the packet kernels applicable beyond exact
packet multiples.

This is a reasonable fixed-size design. It should not be replaced with
per-shape hand-written kernels.

The concerns are:

- `MatmulBackend` also contains solver rank-update and scaling primitives;
- backend selection is one associated type per scalar/target, making
  operation- or shape-specific policy awkward;
- view algorithms bypass the optimized owned kernels and use checked access
  in inner loops;
- compile-time target features can produce a binary that fails on an older CPU
  if built with `target-cpu=native` and redistributed; and
- large const-generic algorithms can increase code size and compile time even
  when runtime is fast.

### 8.2 Current benchmark evidence

The local ignored `benchmark-report/results.csv` contains more than 1,300 rows
from Criterion and native Eigen. On the matched comparison rows in that local
snapshot, the median `stack-algebra / Eigen` latency ratio is approximately
1.00, but individual operation medians and dimensions vary widely. Examples
include faster QR/LU cases, slower LDLT and self-adjoint eigen cases, and
matrix-product ratios that change significantly with shape and scalar.

This is useful diagnostic evidence, not a release conclusion, because:

- the generated report is ignored and does not record sufficient machine,
  compiler, dependency, governor, and thermal metadata;
- Criterion and the native C++ runner use different harnesses and batch sizes;
- several comparison paths use different ownership or allocation models;
- very small nanosecond measurements are sensitive to inlining, constant
  propagation, batching, and black-box placement;
- the report combines many algorithm phases and fallback labels; and
- the nightly suite previously exceeded its intended wall-clock budget.

No library should be declared generally faster from the aggregate median.
Performance must be reported per operation, phase, size, scalar, and target.

### 8.3 Benchmark improvements

Define four benchmark tiers:

1. **Kernel microbenchmarks**: dot, norm, matvec, matmul, rank update, transpose,
   and triangular solve. Verify assembly/codegen for representative sizes.
2. **Factor benchmarks**: analysis, factorization, refactorization, solve, and
   factor-and-solve measured separately.
3. **Workload benchmarks**: representative fixed small systems, fixed block
   sparse systems, and runtime sparse systems only where the library supports
   them honestly.
4. **Target resource benchmarks**: wall time/cycles, peak stack, static RAM,
   binary text size, and compile time for selected MCU and OS profiles.

Every reported row must include:

- git commit and dirty state;
- Rust/C++ compiler version and flags;
- dependency versions;
- CPU model, enabled ISA, core affinity, and frequency policy;
- scalar, shape, layout, and alignment;
- allocation and setup included/excluded;
- algorithm and pivoting/ordering mode;
- correctness precheck status; and
- sample duration and confidence interval.

Add benchmark correctness prechecks that compare outputs before timing. Do not
perform FFI calls in a timed Eigen inner loop. Keep native Eigen compilation,
but generate the same deterministic inputs and validate the same residuals.

### 8.4 Performance strategy

Prioritize general improvements in this order:

1. remove repeated checked view access after safe construction;
2. improve data flow and output reuse in algorithms;
3. separate cohesive kernel families;
4. inspect generated assembly at packet boundaries;
5. tune blocking based on cache/packet properties, not named matrix sizes;
6. add optional runtime dispatch only for a demonstrated portable-OS need; and
7. add new ISAs only with representative hardware and maintenance capacity.

Do not keep a micro-optimization solely because it wins one benchmark size.
Require either a structural explanation or a consistent improvement across a
declared shape range without numerical regressions.

## 9. API ergonomics review

### 9.1 What currently works well

- Dimensions appear directly in `Matrix<M, N, T>`.
- `f32` and `f64` are explicit, and mixed precision requires an explicit cast.
- Literal macros are readable for small matrices and vectors.
- Owned, mapped, strided, block, and bounded storage share operations.
- `*_into`, in-place solve, factor recomputation, and workspace APIs make reuse
  possible without hidden allocation.
- structured triangular and self-adjoint views communicate intended access.

### 9.2 Main inconsistencies

The API mixes:

- `Option` constructors;
- typed `Result` constructors;
- pairs such as `decompose`/`try_decompose` where the shorter form returns
  `Option` rather than panicking;
- infallible-looking methods whose invalid input produces non-finite values;
- indexing that panics and `get` that returns `Option`; and
- sparse errors, dense errors, and shape failure encoded separately or lost.

The `try_` prefix is not itself the problem. The problem is that callers
cannot infer the failure model consistently from the name.

Recommended conventions for the next breaking release:

- checked construction is the default for objects with invariants;
- use `Result` when the reason changes recovery or diagnostics;
- use `Option` only when absence has one obvious meaning;
- reserve `try_` for an operation whose unprefixed form is genuinely
  infallible or intentionally panics;
- use `_into` for caller-provided distinct output;
- use `_in_place` when an input is overwritten;
- use `compute`/`recompute` consistently for factor reuse; and
- make unchecked/assumed-structure APIs explicit in the name or wrapper type.

### 9.3 Constructor and macro cleanup

- Replace arbitrary `diag!` arity limits with `Matrix::from_diagonal` accepting
  a fixed vector or array; retain a macro only as syntax sugar.
- Add typed errors for map length, stride overflow, and bounded shape.
- Decide whether default scalar `f32` improves common code enough to justify
  inference surprises. Keep it only if examples and compile tests show clear
  behavior.
- Reconsider the Eigen-style `T()` alias. `transpose()` is idiomatic Rust and
  unambiguous; an uppercase method adds surface without new capability.
- Prefer focused prelude examples over `use stack_algebra::*`.

### 9.4 Solver selection

Users should not need to infer solver choice from a feature list. Provide one
decision table based on matrix properties:

| Known property | Preferred method | Reason |
| --- | --- | --- |
| symmetric positive definite | Cholesky | Lower cost and clear failure condition. |
| symmetric indefinite | pivoted LDLT | Preserves symmetry and handles 1x1/2x2 pivots. |
| general square | partial-pivot LU | General direct solve. |
| full-rank tall least squares | Householder QR | Avoids normal-equation conditioning loss. |
| rank uncertainty | column-pivoted QR | Rank-revealing at lower cost than SVD in many cases. |
| minimum-norm/rank-deficient or diagnostic decomposition | SVD | More expensive but explicit singular spectrum. |
| repeated same-pattern sparse SPD | analyzed sparse Cholesky | Reuses symbolic structure. |

The table should state when the library does not cover the workload, especially
for general runtime sparse indefinite systems.

## 10. Documentation review

The repository already has a documentation site, feature summary, roadmap,
API guide, tutorials, use cases, rustdoc examples, and benchmark guide. The
problem is fragmentation and drift rather than absence.

Current issues include:

- README, feature summary, roadmap, and use-case guide repeat the same design
  claims with slightly different wording;
- `API_USAGE.md` has duplicate section number 6;
- “stack allocated” is used where “inline storage” is accurate;
- benchmark duration claims have not consistently matched observed runs;
- numerical contracts are spread across method docs and implementation
  thresholds;
- target support lists compilation and QEMU smoke evidence without a single
  table distinguishing compile, emulation, and hardware validation; and
- generated capability claims are manually maintained.

Recommended documentation structure:

1. **Home**: concise scope and supported workload range.
2. **Getting started**: construction, scalar choice, product, and one solve.
3. **Storage and views**: inline, bounded, mapped, strided, sparse, and their
   exact memory costs.
4. **Solver guide**: selection table and numerical contracts.
5. **Sparse guide**: pattern/factor reuse, capacities, and limitations.
6. **Geometry guide**: invariants and conversion behavior.
7. **Targets**: compile/emulator/hardware matrix with resource measurements.
8. **Validation**: correctness methodology and benchmark methodology.
9. **API reference**: rustdoc.

Generate the feature and target tables from a checked machine-readable manifest
where practical. A documentation test should fail when a public solver lacks a
guide entry or numerical contract.

## 11. Alternatives and actual merit

### 11.1 Comparison summary

| Library | Storage/shape model | Main merit | Main limitation relative to this crate |
| --- | --- | --- | --- |
| Eigen | Fixed, dynamic, bounded-dynamic, maps, expressions, dense and sparse | Very broad algorithms and mature optimization across sizes | C++, large surface, and dynamic paths may allocate unless controlled. |
| nalgebra | Static/dynamic dimensions with generic storage; broad geometry | Mature Rust API, strong geometry, static `no_std` matrices and decompositions | Larger abstraction/dependency surface; performance varies by operation and size. |
| faer | Dynamic owning matrices and lightweight views; dense/sparse high-performance algorithms | Strong medium/large dense and sparse performance, runtime CPU detection under `std` | Official docs state it is not aimed at mostly low-dimensional matrices; not an MCU fixed-inline design. |
| glam | Specialized small vectors, matrices, quaternions, and transforms | Compact ergonomic geometry API with SIMD storage | Not a general decomposition or sparse linear-algebra library. |
| CMSIS-DSP | Runtime dimensions over caller buffers; row-major; floating and fixed-point variants | Tuned Arm MCU kernels and fixed-point support | C API, Arm-specific, runtime shape errors, and less type-level dimension checking. |
| sprs | Dynamic compressed sparse matrices and vectors | Established Rust sparse storage and operations | Allocation-oriented and separate solver ecosystem; not a fixed-capacity `no_std` core. |
| nalgebra-sparse | Dynamic CSR/CSC/COO integrated with nalgebra | Clear pattern representation and ecosystem integration | Official docs describe limited solver availability and an early performance focus. |
| micromath | Small embedded vector/quaternion and approximate `f32` math | Small code and fast approximate functions | Precision is deliberately traded for speed; not a general matrix solver. |

### 11.2 Position by domain

For small fixed dense algebra on an MCU, the closest practical comparison is
nalgebra's static `no_std` matrices plus architecture-specific alternatives
such as CMSIS-DSP. The merit of this crate must be demonstrated through smaller
or clearer APIs, predictable storage, competitive cycles, or algorithms that
the alternatives do not provide under the same constraints. `no_std` and
fixed-size types alone are not differentiators because nalgebra also supports
them.

For small fixed dense algebra on an OS, Eigen and nalgebra are direct
alternatives. This crate can be useful when callers want a narrower API,
explicit buffer reuse, and the same code on host and firmware. It needs better
compile-time, code-size, and API evidence to establish that benefit.

For medium/large dense algebra, faer is the appropriate Rust performance
reference and Eigen's dynamic matrices are the C++ reference. This crate should
not expand large fixed kernels merely to compete in a domain where inline const
storage is no longer the right primary abstraction.

For large runtime sparse systems, Eigen Sparse, faer sparse, sprs, external
sparse solvers, or application-specific solvers remain more appropriate. The
fixed-capacity sparse implementation has merit for bounded patterns and
deterministic reuse, but its capacity and pivot limitations must be explicit.

For geometry-only use, nalgebra and glam are already strong alternatives. The
geometry module should remain small and invariant-focused unless a concrete
linear-algebra workflow requires more.

## 12. Recommended target architecture

### 12.1 Core contract

The default crate should provide:

- `no_std` operation without `alloc`;
- inline `Matrix<M, N, T>` and fixed-capacity storage;
- safe external-buffer maps and strided views;
- portable scalar algorithms;
- compile-time packet kernels for explicitly targeted binaries;
- deterministic factor/workspace reuse;
- supported `f32` and `f64` numerical contracts; and
- fixed-capacity scalar and block sparse types with explicit capacity errors.

### 12.2 Optional OS layer

Do not implement this until a real OS workload requires it. If required, an
optional `std` layer may provide:

- runtime CPU feature detection and multiversioned kernels;
- benchmark/system metadata collection; and
- adapters to caller-owned dynamically sized buffers.

A heap-owning dynamic matrix should remain out of scope until mapped buffers,
bounded matrices, and an external dynamic library have been shown insufficient
for a real integration. If introduced, it should use the same view and solver
interfaces rather than duplicating the core API.

### 12.3 Feature policy

Use features to remove meaningful code or platform dependencies, not to expose
arbitrary combinations that cannot be tested. Candidate features after
measurement:

- `geometry`;
- `sparse`;
- `advanced-decompositions` for SVD/eigen if they materially affect MCU flash;
- `std` for runtime dispatch and host integration; and
- test/benchmark-only Eigen comparison.

Do not feature-gate every individual solver. Measure binary dead-code
elimination first; unused generic code may already be removed by the linker.

## 13. Implementation plan

### Phase 0 — Freeze claims and establish baselines

Goal: make current behavior measurable before changing architecture.

Tasks:

1. Record current public API with `cargo public-api` or an equivalent checked
   artifact.
2. Define supported scalar, shape, target, and algorithm matrices.
3. Add machine-readable benchmark metadata and separate cold build time from
   execution time.
4. Add representative OS and MCU size/code/stack baselines.
5. Mark benchmark reports with commit, dirty state, dependency versions, and
   compiler flags.
6. Replace broad parity claims in README/docs with scoped statements.

Exit criteria:

- every public solver appears in the capability matrix;
- every benchmark row is reproducible from recorded metadata;
- the documentation distinguishes compile, QEMU, and hardware evidence; and
- no performance conclusion uses an aggregate across unlike operations.

### Phase 1 — Safety and invariant isolation

Goal: make every unsafe assumption local and testable.

Tasks:

1. Move representation unsafe code into a small private module.
2. Replace unnecessary unsafe construction in `cross` and delete dead unsafe
   commented code.
3. Add Miri suites for constructors, panic cleanup, row/column views, maps,
   strides, blocks, and mutable aliasing.
4. Add host sanitizer jobs for view and C++ FFI tests.
5. Add layout assertions and scalar-reference SIMD differential tests.
6. Audit geometry invariant constructors and normalization failure behavior.

Exit criteria:

- all unsafe blocks are in reviewed low-level modules or ISA modules;
- every safe API reaching unsafe code has a direct regression test;
- Miri covers all non-SIMD unsafe paths; and
- validated geometry types cannot be constructed through a weaker accidental
  path.

### Phase 2 — API cleanup for `0.3`

Goal: make the common API predictable without compatibility shims.

Tasks:

1. Define and apply naming rules for checked construction, output reuse,
   in-place mutation, and factor recomputation.
2. Add `ShapeError`, map/stride errors, and capacity diagnostics.
3. Replace arbitrary diagonal macro arity with a general constructor.
4. Resolve zero/non-finite normalization semantics.
5. Remove or deprecate redundant aliases and compatibility-style methods.
6. Seal or redesign scalar/backend traits so concrete kernel types disappear
   from the public root.
7. Add compile tests for intended ergonomic examples and rejected dimension or
   scalar combinations.

Exit criteria:

- a caller can predict `Option`, `Result`, panic, and in-place behavior from
  documented conventions;
- normal `f32`/`f64` use does not mention kernel backend types;
- all constructor errors provide actionable context; and
- rustdoc examples cover every matrix/view/factor type.

### Phase 3 — Internal architecture cleanup

Goal: align module boundaries with invariants while preserving behavior.

Tasks:

1. Split matrix, linalg, sparse scalar, sparse block, geometry, and kernel
   internals incrementally.
2. Use scalar sparse pattern/symbolic/numeric separation as the block-sparse
   structure.
3. Introduce private trusted read/write traits for built-in checked views.
4. Separate matrix product, reduction, triangular, and factor-update kernel
   families.
5. Keep portable scalar implementations as the reference for every family.

Exit criteria:

- no implementation module combines public storage, symbolic analysis,
  numeric factorization, fallback policy, and tests in one large file;
- view algorithms do not perform fallible coordinate lookup in their innermost
  loops after construction; and
- optimized backends remain replaceable without changing public scalar traits.

### Phase 4 — Numerical validation

Goal: define correctness independently of one reference implementation.

Tasks:

1. Write numerical contracts for every factor and geometry conversion.
2. Add deterministic property tests across dimensions, scales, rank, and
   conditioning.
3. Validate reconstruction, orthogonality, residual, and rank invariants.
4. Add Eigen and at least one independent Rust/reference comparison where
   meaningful.
5. Test owned and all supported view/storage paths with identical inputs.
6. Record convergence limits and threshold behavior in docs.

Exit criteria:

- every solver has invariant-based randomized tests;
- failures report the documented class;
- results meet scalar- and dimension-specific residual bounds; and
- differential tests do not require arbitrary factor signs/orderings to match.

### Phase 5 — Performance architecture

Goal: improve general data paths without shape-specific special cases.

Tasks:

1. Benchmark trusted view access versus owned access.
2. Measure packet-width boundaries and inspect generated assembly.
3. Tune loop/block policies using declared size ranges rather than named
   dimensions.
4. Measure code size and compile time for each added kernel family.
5. Test portable, SSE2, AVX2, AVX2+FMA, and NEON results against scalar
   references.
6. Decide whether optional `std` runtime dispatch is justified for portable OS
   binaries.

Exit criteria:

- no retained optimization is justified by one size only;
- the supported small-matrix envelope is competitive with fixed Eigen and
  static nalgebra within documented tolerances;
- medium/large comparisons are reported without claiming the fixed core should
  win; and
- MCU improvements do not exceed stack, flash, or numerical error budgets.

### Phase 6 — Sparse scope and ergonomics

Goal: make bounded sparse behavior understandable and diagnosable.

Tasks:

1. Split block storage, analysis, factor, solve, and fallback modules.
2. Add required-capacity diagnostics from symbolic analysis.
3. Document supported sparse symmetry and pivot models precisely.
4. Add generated sparse pattern tests including adversarial fill.
5. Add repeated-analysis/refactor/solve correctness and performance tests.
6. Define when users should use dense fallback or another library.

Exit criteria:

- capacity failures report required and available storage;
- cross-block/global pivot limitations are explicit;
- symbolic and numeric reuse have independent tests and benchmarks; and
- no sparse API silently changes to a dense allocation or hidden storage
  model.

### Phase 7 — Target qualification and stable release

Goal: convert broad target claims into maintained support tiers.

Tasks:

1. Define Tier 1 host, Tier 1 MCU, compile-only, and experimental targets.
2. Run real-device tests for at least one Cortex-M FPU target and one RISC-V or
   ESP-class target used by maintainers.
3. Record cycles, peak stack, static RAM, and binary size for representative
   operations.
4. Test portable OS binaries separately from `target-cpu=native` binaries.
5. Publish API, numerical, target, and benchmark compatibility reports.

Exit criteria:

- stable APIs have documented behavior and resource envelopes;
- Tier 1 targets execute on real hardware in CI or a repeatable release
  process;
- benchmark and correctness reports are attached to releases; and
- unsupported workload classes are stated directly.

## 14. Priority order

| Priority | Work | Impact | Risk if delayed |
| --- | --- | --- | --- |
| P0 | Freeze claims and benchmark metadata | High | Optimization and parity conclusions remain unreliable. |
| P0 | Isolate unsafe representation code and expand Miri | High | Refactors can invalidate distributed assumptions. |
| P0 | Define numerical contracts and invariant tests | High | Element parity can mask incorrect or brittle behavior. |
| P1 | Clean failure/naming conventions | High | Public API becomes harder to change as adoption grows. |
| P1 | Remove backend types from normal public API | High | Kernel implementation becomes a permanent user-facing constraint. |
| P1 | Trusted internal view access | Medium-high | External-buffer algorithms retain avoidable overhead. |
| P1 | Split block-sparse responsibilities | Medium-high | Sparse changes remain difficult to review and test. |
| P2 | Code-size/stack/compile-time benchmarks | Medium-high | MCU suitability is inferred from allocation behavior alone. |
| P2 | Optional runtime CPU dispatch study | Medium | Portable OS binaries may leave performance unused or become incompatible. |
| P3 | Additional ISAs or algorithms | Workload-dependent | Adds maintenance before the current surface is stable. |
| Deferred | Heap-owning dynamic matrices | Unknown | Complexity doubles without a demonstrated unmet workload. |
| Deferred | General expression-template system | High complexity | Large API/compiler cost; explicit reuse APIs already cover core loops. |

## 15. Decisions to keep, change, and defer

### Keep

- fixed-size `Matrix<M, N, T>` as the core;
- column-major default storage;
- explicit scalar conversion and no implicit mixed precision;
- `no_std` and no `alloc` requirement in the default core;
- caller-provided maps, output reuse, in-place solves, and reusable factors;
- portable scalar reference kernels;
- fixed-capacity sparse and block-sparse storage for bounded patterns; and
- Eigen/faer/nalgebra as external validation references.

### Change

- describe storage as inline rather than inherently stack allocated;
- replace broad Eigen parity language with scoped operation contracts;
- isolate unsafe code and trusted-view invariants;
- remove optimized backend mechanics from the normal public API;
- standardize constructor and solver failure semantics;
- split large modules by storage/symbolic/numeric/kernel responsibility;
- make sparse capacity errors actionable; and
- measure code size, compile time, and real-device resources alongside latency.

### Defer

- heap-owning dynamic matrices;
- a general expression-template system;
- general runtime sparse indefinite solving;
- GPU or accelerator backends;
- per-shape hand-written kernels;
- Arm32 NEON, RVV, or other new ISA modules without hardware ownership and
  benchmark evidence; and
- broad geometry expansion unrelated to the matrix/factor core.

## 16. First implementation slice

The first change set should be deliberately narrow and should not alter
algorithms:

1. add a generated capability/target manifest;
2. correct “stack” terminology in public docs;
3. add benchmark provenance metadata;
4. create the private representation safety module;
5. replace unsafe `cross` construction;
6. expand Miri to constructors and dense views;
7. add `ShapeError` and map/bounded constructor diagnostics behind the planned
   `0.3` breaking API; and
8. write one complete numerical contract and property suite for Cholesky as the
   template for other solvers.

This slice produces immediate clarity and safety evidence without committing
the repository to dynamic allocation, a new ISA, or a large API abstraction.
