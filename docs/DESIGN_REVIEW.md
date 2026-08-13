# Technical review: scope, merit, and comparison

Review date: 2026-08-14

Repository revision: [`cb242f9`](https://github.com/yongkyuns/stack-algebra/commit/cb242f98055886260a8148f38511ac231426c890)

Compared with: faer 0.24.4 and Eigen 5.0 documentation; the repository's
Eigen benchmark currently uses the distribution-provided Eigen headers rather
than recording an exact version.

## Executive verdict

`stack-algebra` has real merit, but it is narrower and less mature than its
surface area initially suggests.

Its strongest proposition is this combination:

- compile-time shapes and inline storage in safe Rust;
- a default `no_std`, no-`alloc` core;
- reusable dense factors and caller-owned outputs;
- mapped, strided, and bounded storage;
- fixed-capacity scalar and block-sparse storage; and
- a useful set of decompositions for small robotics, control, and estimation
  problems.

That is a credible niche. Neither faer's primary dynamic API nor Eigen's C++
API has the same Rust, heap-free, const-shaped contract.

The project is **not** currently a general alternative to faer or Eigen. It
lacks their dynamic-size breadth, scalar breadth, mature sparse solvers,
ecosystem, and validation history. More importantly, the current `main` branch
has a confirmed undefined-behavior failure in a safe sparse constructor. The
latest CI run passes normal tests, Eigen differential tests, docs, Clippy,
cross-compilation, QEMU, and native Arm64 tests, but fails Miri in
`StaticCscPattern::from_arrays`. This is a release blocker.

The right positioning is:

> A focused, allocation-free Rust algebra core for small fixed or tightly
> bounded systems—not “Eigen in Rust” and not a medium/large replacement for
> faer.

## Evidence and review limits

This review inspected the public API, implementation, tests, benchmarks,
workflows, documentation, repository history, and the latest CI and nightly
benchmark artifacts.

The strongest available evidence is:

- 223 Rust tests across unit and integration suites;
- Eigen differential tests for selected dense and sparse operations;
- portable, SSE2, AVX2/FMA, and AArch64 NEON implementations;
- compile checks for the documented embedded targets;
- QEMU smoke runs for Cortex-M4, RISC-V32, and AArch64;
- a native Arm64 test job; and
- nightly comparisons with faer, Eigen, and nalgebra.

Important limits:

- the current published crate is still `0.1.0`; `main` declares `0.2.0` but is
  unreleased;
- repository development and validation are effectively single-maintainer;
- there are no maintained real-board latency, peak-stack, flash, or RAM
  results;
- there is no independent numerical audit or fuzzing record;
- the Miri gate is currently red; and
- benchmark reports do not record the exact Eigen version, C++ compiler
  version, dependency lock, generated assembly, frequency policy, or runner
  isolation.

Those limits do not negate the implementation. They constrain what can be
claimed from it.

## Claim audit

| Claim or implication | Assessment | Required wording or action |
| --- | --- | --- |
| “Stack allocated” | Imprecise. `Matrix` owns inline storage, but a value can live on a stack, in static memory, in an arena, or inside another allocation. | Use **inline storage**. Say that placement follows the owner. |
| “No heap allocation” | Credible for the default core API and fixed-capacity paths, but not yet backed by an allocator-instrumented regression test. | Say **does not require `alloc`**. Add a no-allocation test for representative public paths. |
| “Bare-metal support” | Supported by `no_std` builds and QEMU smoke tests. It is not evidence of board peripherals, cycle counts, or production qualification. | Keep the evidence tiers in `targets.md`; add real-device results before naming a board as validated. |
| “Eigen-compatible Bunch–Kaufman” | Overstated. The code implements classical Bunch–Kaufman-style 1x1/2x2 pivot selection, but differential tests compare solutions with Eigen's ordinary `LDLT`; they do not establish identical pivot choices or factor layout. | Call it **Bunch–Kaufman-style pivoting** until pivot-by-pivot compatibility is tested against the intended Eigen decomposition and version. |
| “Eigen parity” | Too broad. Tests cover selected operations, shapes, scalars, and residual tolerances, not API or algorithm parity. | Use **Eigen differential tests for the enumerated cases**. |
| Eigen benchmark uses the “same ... input values” | False for several current cases. Rust `dense()` and native Eigen `make_matrix()` use different formulas, and their general-system diagonals also differ. | Generate shared input files/code or use one FFI benchmark process, then assert input hashes. |
| Competitive performance | True for parts of the intended small-matrix envelope, false as a blanket statement. The latest report contains meaningful wins and losses. | State results per operation, shape, scalar, phase, commit, and machine. Never aggregate unlike work into a marketing claim. |
| Miri safety validation | Currently false as a passing claim. The latest Miri job reports undefined behavior from uninitialized `u32` entries in `StaticCscPattern::from_arrays`. | Fix before release and keep the Miri badge/gate red until it passes. |
| Deterministic or predictable execution | Storage bounds are predictable; execution time has not been qualified as WCET. Pivoting, convergence iterations, cache state, and target math implementation affect timing. | Use **bounded storage** and **no hidden allocation**. Reserve “deterministic timing” for measured, target-specific evidence. |
| “Published library” describes current docs | Misleading. `cargo add stack-algebra` installs `0.1.0`, which does not contain most of the documented `main` API. | Clearly label the site as development documentation until `0.2.0` is published. |

## Feature-set comparison

The table compares practical public capability, not line counts or similarly
named methods.

| Area | stack-algebra `main` | faer 0.24.4 | Eigen 5.0 |
| --- | --- | --- | --- |
| Primary language and model | Rust; const-shaped, inline values | Rust; dynamic owning matrices plus views | Header-only C++; fixed, bounded-dynamic, and dynamic matrices plus expressions |
| Default memory model | `no_std`, no `alloc`; caller-selected placement | Heap-backed `Mat`; `no_std` is possible but the crate uses `alloc` | Fixed matrices can avoid allocation; dynamic matrices allocate unless controlled |
| Runtime-sized owning dense matrix | No | Yes, resizable | Yes |
| Bounded active dimensions | `MatrixBuf<MAX_R, MAX_C>` storage; decompositions still require a const-shaped view | Dynamic views and owning capacity | Fixed maximum rows/columns are part of `Matrix`'s type |
| External buffers/views | Contiguous, strided, mutable, and fixed blocks | Rich sliced/split `MatRef` and `MatMut` views | `Map`, `Ref`, blocks, strides, expressions |
| Dense scalar coverage | Numerical solvers target `f32`/`f64`; basic algebra is more generic | Real, complex, and extensible entity/scalar machinery | Standard numeric types, complex, and documented custom scalar extension |
| Dense direct solvers | Partial-pivot LU, LLT, pivoted LDLT, QR, column-pivot QR | LLT, block-pivoted LBLT, partial/full-pivot LU, QR, column-pivot QR | Broad LU, Cholesky, QR, and related decompositions |
| SVD/eigen | Thin SVD; self-adjoint eigen only | Full/thin SVD; self-adjoint and general eigen APIs | Jacobi/BDC SVD, self-adjoint/general/generalized eigen, Schur, Hessenberg, and more |
| Sparse storage | Fixed-capacity scalar CSC plus block CSC/CSR | Dynamic sparse matrices | Dynamic compressed sparse matrices |
| Sparse solvers | LLT; limited LDLT with diagonal pivots or bounded dense fallback | Sparse LLT, LU, and QR | Sparse LLT/LDLT, LU, QR, iterative solvers, and support modules |
| Geometry | Quaternion, angle-axis, rotation, isometry, affine transform | Not a primary feature | Broad geometry module and related unsupported modules |
| Lazy expressions/fusion | Mostly eager; explicit `*_into` reuse | High- and low-level operation APIs with reusable scratch | Extensive expression templates and lazy evaluation |
| Parallelism | None; predictable sequential core | Optional Rayon; `Par::Seq` or parallel execution | Threaded operations where enabled; external backends available |
| SIMD selection | Compile-time scalar/SSE2/AVX2/FMA/NEON | Runtime CPU dispatch under `std` through its kernel stack | Compile-time vectorization across a much wider ISA set |
| Current maturity evidence | Unreleased 0.2 development tree; one maintainer; current Miri failure | Published 0.24 series, broader user base and active upstream | Decades of production use and a very large test matrix |

Primary comparison sources:

- [faer 0.24.4 crate documentation](https://docs.rs/faer/0.24.4/faer/)
- [faer feature manifest](https://github.com/sarah-quinones/faer-rs/blob/main/faer/Cargo.toml)
- [Eigen overview and supported scope](https://libeigen.gitlab.io/)
- [Eigen fixed- and dynamic-size matrix guidance](https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialMatrixClass.html)

### Where stack-algebra is genuinely better

For its intended domain, stack-algebra can offer advantages that are not mere
feature-count differences:

1. **The memory bound is part of the type.** A factor, sparse pattern, and
   workspace can be budgeted with `size_of`/`storage_bytes` before deployment.
2. **No allocator is required.** This matters on bare metal and in control
   loops where an allocator is unavailable or prohibited.
3. **Shape mismatches are compile-time errors** in the fixed dense API.
4. **Factor and output reuse are explicit.** The API makes repeated estimation
   or control-loop work visible rather than relying on expression lifetime
   optimization.
5. **Fixed-capacity sparse and block-sparse support is unusually ambitious for
   a small `no_std` Rust crate.** It can suit generated normal equations whose
   topology and fill bounds are known.
6. **The same Rust API crosses host and firmware builds.** This can reduce the
   translation boundary between desktop validation and embedded deployment.

These are engineering merits. They should be supported with resource and
target evidence, not converted into a claim that the library is universally
faster or more capable.

### Where faer is the better choice

Choose faer when matrices are runtime-sized, medium or large; complex scalars
matter; sparse LU/QR is needed; parallelism is useful; or high-performance
dynamic algebra is the primary workload. Faer's own documentation says it is
focused on medium/large matrices and is not well suited to mostly
low-dimensional vector/matrix workloads. That makes faer an important
complement and performance boundary, not the product that stack-algebra needs
to imitate everywhere.

### Where Eigen is the better choice

Choose Eigen when C++ is acceptable and breadth, maturity, custom scalars,
dynamic/fixed interoperability, expression templates, geometry breadth,
specialized decompositions, or an established production history outweigh a
narrow Rust/no-allocator core. Eigen also optimizes fixed matrices and can
avoid dynamic allocation, so “fixed size” alone is not a differentiator.

## What the latest benchmark actually shows

The latest successful nightly artifact at the reviewed commit contains 1,344
measurements on an AMD EPYC 9V74 GitHub runner using Rust 1.97.1. The nightly
profile uses 20 ms Criterion warmup/measurement windows with 10 samples; the
native Eigen executable uses five samples of at least 5 ms.

For the `comparison/*` dense groups, matching by operation, shape, scalar, and
phase gives:

| Comparison | Matched cells | stack-algebra faster | Geometric mean of `stack / peer` time | Defensible conclusion |
| --- | ---: | ---: | ---: | --- |
| Eigen fixed-size native executable | 192 | 72 (38%) | 1.16x | Mixed. Competitive in selected small QR/LLT/eigen/reduction cases; slower overall in this snapshot. |
| faer dynamic API, sequential | 216 | 181 (84%) | 0.35x | Strong fixed-size/low-overhead result, but many cells include a storage/allocation-model advantage and are outside faer's stated sweet spot. |
| nalgebra static where used | 36 | 16 (44%) | 1.26x | Mixed; the small sample covers only reductions and multiplication. |

The aggregate is included to audit broad claims, not as a product score.
Representative results show why per-case reporting is necessary:

| Case | stack-algebra | Eigen | faer dynamic | Observation |
| --- | ---: | ---: | ---: | --- |
| `f32` 8x8 matrix multiply | 14.7 ns | 65.0 ns | 82.9 ns | Clear win in a relevant small fixed case. |
| `f32` 32x32 matrix multiply | 1,295 ns | 517 ns | 1,171 ns | Loses as size moves beyond the strongest fixed-kernel envelope. |
| `f64` 8x8 LLT factor | 180 ns | 129 ns | 377 ns | Between Eigen and faer. |
| `f64` 32x32 LLT factor | 5,652 ns | 1,900 ns | 1,882 ns | Roughly 3x slower. |
| `f64` 32x32 self-adjoint eigen with reused workspace | 15,804 ns | 24,105 ns | 46,353 ns | Strong result for this implementation and input. |
| `f64` 64x32 SVD factor | 283,411 ns | not measured in the native Eigen suite | 283,993 ns | Rough parity with faer, but nalgebra is substantially faster in this case. |

Do not promote these exact times as durable release claims. They come from a
short shared-cloud run, and the report's `git_dirty=true` is caused by the
report job downloading benchmark inputs into the checkout rather than by a
recorded source patch.

### Benchmark defects to fix

1. Use exactly the same input generator. At present, Rust and native Eigen
   differ for several dense and general-system inputs.
2. Run cross-library cases on the same runner and preferably in the same
   process or sequentially pinned environment.
3. Record exact Eigen, faer, nalgebra, Rust, C++, linker, CPU-feature, and
   dependency-lock versions.
4. Separate one-shot construction/allocation, factorization, refactorization,
   and solve in every library.
5. Validate residuals or output hashes immediately before timing every case.
6. Report code size, compile time, peak stack, and static RAM alongside
   latency; these are central to the library's stated merit.
7. Use longer, isolated release runs. Keep the short nightly profile only for
   regression triage.
8. Publish a versioned artifact per release instead of a hand-maintained table
   that drifts after each kernel change.

## Safety and numerical assessment

### Confirmed release blocker

The [latest build run](https://github.com/yongkyuns/stack-algebra/actions/runs/31647917174)
fails Miri because `StaticCscPattern::from_arrays_into` initializes only
`row_indices[..nnz]` inside an otherwise uninitialized struct, then
`assume_init()` creates a value whose remaining `u32` entries are
uninitialized. Constructing that safe Rust value is undefined behavior even if
the unused tail is never read.

The simplest robust fix is to initialize the complete struct to
`StaticCscPattern::new()` and then overwrite validated active entries. If the
direct-into API is retained for stack-pressure reasons, it must initialize all
array elements and fields before exposing `Self`. Add Miri tests for every
partial-capacity constructor and error/panic path.

### Numerical strengths

- Checked dense decomposition paths reject non-finite inputs/intermediates.
- Cholesky, QR, SVD, eigendecomposition, and LDLT have reconstruction or
  residual tests.
- Norm accumulation includes a scaled path for extreme finite magnitudes.
- Factor recomputation preserves the prior factor in several checked paths.
- Differential tests compare results rather than requiring bitwise equality
  for decompositions with non-unique factors.

### Numerical/API risks

- `try_partial_piv_lu` checks finiteness but does not report singularity; a zero
  pivot can survive construction and later produce non-finite solve results.
  Either return `DecompositionError::Singular` or make the assumption explicit
  in the checked method's name.
- `normalize()` has no zero/non-finite failure contract.
- Numerical tests cover a small deterministic set of dimensions and scales;
  they are not a conditioning or adversarial corpus.
- `Option` and `Result` constructors coexist inconsistently across factors,
  making recovery behavior difficult to infer.
- Dense Bunch–Kaufman, sparse diagonal pivoting, local block pivots, and dense
  fallback are materially different algorithms but are easy to conflate in
  high-level prose.
- Sparse capacity failures usually report only that capacity was exceeded,
  not the required symbolic/fill capacity.

## Prioritized improvements

### P0 — before publishing 0.2

1. Fix the `StaticCscPattern` undefined behavior and require a green Miri gate.
2. Audit every `MaybeUninit`, raw-slice, pointer, and SIMD safe-API boundary;
   run Miri over all non-SIMD unsafe paths.
3. Correct benchmark input equivalence and remove “same inputs”/broad parity
   wording until automated checks prove it.
4. Decide the release identity: publish `0.2.0` or label all `main`
   documentation as unreleased and provide a Git dependency example.
5. Add a release checklist that requires normal tests, Eigen differential
   tests, docs, Clippy, formatting, Miri, target builds, QEMU, and a reviewed
   benchmark artifact to pass on the same commit.

### P1 — prove the intended niche

1. Add real Cortex-M and RISC-V/ESP-class measurements for representative 3x3,
   6x6, and 15x15 operations: cycles, peak stack, static RAM, flash, and
   numerical residual.
2. Add allocator-instrumented tests proving no allocation for the documented
   fixed, mapped, sparse, and factor-reuse paths.
3. Add deterministic randomized/property tests across scale, conditioning,
   rank, pivot patterns, sparse fill, and malformed inputs.
4. Define one numerical contract per solver: accepted structure, triangle
   read, threshold, rank rule, convergence bound, failure class, and residual
   expectation.
5. Make sparse symbolic failures report required versus available capacity.
6. Standardize checked construction, `_into`, `_in_place`, and recomputation
   naming before the public API gains more users.

### P2 — improve integration without diluting the core

1. Add opt-in adapters to faer/nalgebra or generic slice/view conversion so an
   application can use stack-algebra on firmware-sized kernels and faer for
   larger host-side work.
2. Consider optional runtime CPU dispatch only in a `std` host layer; retain
   compile-time selection in the embedded core.
3. Feature-gate geometry, sparse, and advanced decompositions only if measured
   flash/compile-time savings justify the added test matrix.
4. Add compile-time and code-size regression reports for representative public
   API use, not just crate-wide builds.
5. Split large sparse modules around storage, symbolic analysis, numeric
   factorization, and solve invariants to make unsafe review tractable.

### Defer unless a concrete workload demands it

- a heap-owning dynamic matrix;
- general large sparse indefinite solving;
- GPU/accelerator backends;
- a general expression-template system;
- additional ISAs without owned hardware and sustained benchmarks; and
- geometry breadth unrelated to embedded estimation/control workloads.

For dynamic or medium/large work, integration with faer is more credible than
reimplementing faer inside this crate. For C++ consumers, Eigen should remain a
reference and interoperability target rather than a promised parity level.

## Release-readiness scorecard

| Area | Current state | 0.2 release gate |
| --- | --- | --- |
| Scope and positioning | Good niche, overstated in places | Adopt the focused positioning above. |
| Dense API breadth | Strong for small fixed real matrices | Freeze and document numerical contracts. |
| Sparse ambition | High and potentially differentiating | Fix UB; improve capacity and pivot-boundary diagnostics. |
| Correctness tests | Good start | Add adversarial/property coverage and output checks in benchmarks. |
| Memory safety | Not releasable today | Green Miri plus unsafe-boundary audit. |
| Embedded portability | Credible compile/QEMU evidence | Add at least one real-device resource baseline. |
| Performance | Competitive but mixed | Same-input, same-runner, versioned release artifact. |
| Documentation | Broad but previously repetitive/stale | Keep one feature inventory, one review, one roadmap, and generated benchmark artifacts. |
| Distribution | Main and crates.io diverge | Publish 0.2 or label development installs explicitly. |

The project should optimize for trust now, not additional feature count. A
smaller set of precisely specified, memory-safe, measured operations would be
more valuable than another decomposition or ISA backend.
