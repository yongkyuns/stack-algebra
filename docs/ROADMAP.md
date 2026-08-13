# Roadmap

This roadmap starts from the reviewed `main` branch at commit `cb242f9`. It is
ordered by risk reduction and evidence, not by feature count. Completed work
belongs in the [feature inventory](FEATURES.md); the roadmap records only work
that remains materially open.

## Product boundary

Keep the core focused on small fixed or tightly bounded systems that need
inline or caller-owned storage, compile-time shapes, and no dependency on
`alloc`. Runtime-sized medium/large algebra is better served through optional
interoperability with faer than by cloning faer's scope. Eigen remains a
correctness and C++ interoperability reference, not an API-parity target.

## P0 — release blockers for 0.2

- Fix the undefined behavior in `StaticCscPattern::from_arrays_into`. Require
  all fields and inactive capacity to be initialized before constructing
  `Self`.
- Audit every safe API backed by `MaybeUninit`, raw slices, pointer arithmetic,
  or SIMD. Run Miri over all applicable non-SIMD paths and keep the release
  blocked until it passes.
- Make checked LU construction reject a singular pivot or explicitly document
  a different contract. A checked factor must not defer division by zero to
  `solve` without a reported error.
- Correct the Rust/Eigen input mismatch in comparison benchmarks. Generate one
  input representation and verify hashes or values before timing.
- Label all docs as development-only until `0.2.0` is published. Do not point
  users to a crates.io version that does not contain the documented API.
- Add a release checklist requiring tests, Eigen differential cases, rustdoc,
  Clippy, formatting, Miri, target builds, QEMU, and a reviewed benchmark
  artifact to pass on the same commit.

## P1 — numerical and API contracts

- Define the accepted structure, authoritative triangle, threshold/rank rule,
  convergence limit, failure classes, and residual expectation for every
  solver.
- Rename claims of “Eigen compatibility” to the exact operation tested. Add
  pivot-by-pivot and factor-layout tests before claiming Bunch–Kaufman layout
  compatibility.
- Add deterministic property and adversarial tests across magnitude,
  conditioning, rank, pivot patterns, sparse fill, malformed inputs, zero
  vectors, and non-finite data.
- Standardize `Result`/`Option`, checked constructors, `compute`, `_into`, and
  `_in_place` naming. Preserve the previous valid factor consistently after a
  failed recomputation.
- Make sparse symbolic and numeric capacity errors report required and
  available capacity.
- Split sparse implementation modules around storage, symbolic analysis,
  ordering, numeric factorization, and solve invariants so unsafe review is
  tractable.

## P1 — prove the embedded merit

- Add allocator-instrumented tests proving zero allocation for the documented
  dense, view, sparse, and factor-reuse paths.
- Publish real-device measurements for representative 3x3, 6x6, and 15x15
  operations: cycles, peak stack, static RAM, flash, and residuals. Record the
  board, toolchain, flags, clock, memory placement, and commit.
- Add code-size and compile-time regression tracking for representative user
  programs, not only crate-wide builds.
- Treat QEMU as execution smoke evidence only. Do not infer peripheral,
  floating-point throughput, or worst-case timing claims from it.

## P1 — make performance evidence reproducible

- Run stack-algebra, Eigen, faer, and nalgebra sequentially on the same runner;
  use one FFI process where practical.
- Record exact dependency versions, lockfile, compilers, linker, CPU flags,
  runner image, and frequency policy.
- Validate outputs immediately before timing. Separate allocation/construction,
  symbolic analysis, factorization, refactorization, and solve for each peer.
- Retain short nightly runs for regression triage. Use longer isolated runs for
  release claims and publish immutable CSV/JSON artifacts per release.
- Add memory, code-size, and peak-stack results beside latency. Those metrics
  test the project's main proposition more directly than host nanoseconds do.

## P2 — integration and carefully justified scope

- Add opt-in adapters or view conversions for faer and nalgebra so host tools
  can use their broader dynamic APIs while firmware-sized kernels stay in
  stack-algebra.
- Consider runtime CPU dispatch only in an optional `std` host layer; retain
  compile-time selection in the embedded core.
- Feature-gate geometry, sparse, or advanced decompositions only after
  measuring meaningful flash or compile-time savings and accepting the added
  test matrix.
- Add new ISAs only with owned hardware, scalar-reference differential tests,
  and sustained benchmarks.

## Explicitly deferred

- a heap-owning dynamic matrix;
- a general expression-template system;
- general large sparse indefinite solvers;
- GPU or accelerator backends; and
- geometry breadth unrelated to embedded estimation and control workloads.

Revisit a deferred item only when a concrete workload cannot be served by
fixed, bounded, mapped, sparse, or interoperable storage. The next release will
gain more credibility from a smaller, safe, precisely specified surface than
from another decomposition or backend.
