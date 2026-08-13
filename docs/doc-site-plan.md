# stack-algebra Documentation Site Plan

This plan defines a standalone documentation site for `stack-algebra`. It
combines curated guides, runnable examples, generated Rust API documentation,
use cases, and validation reports.

## 1. Goals and constraints

### Goals

- Document installation and a working fixed-size matrix example.
- Describe storage, scalar types, views, dense solvers, geometry, sparse APIs,
  and embedded deployment in a consistent structure.
- Provide every public type with a runnable example and usage guidance.
- Publish the generated API reference, benchmark reports, and target results.
- Record allocation, layout, numerical assumptions, failure behavior, and
  measured performance methodology.

### Constraints

- Keep the project standalone and framework-independent.
- Keep the visual identity and terminology specific to `stack-algebra`.
- Do not manually duplicate the generated API reference in Markdown.
- Do not make the docs site a runtime dependency of the crate.
- Do not claim MCU hardware support based only on host or QEMU results.
- Do not require a JavaScript-heavy frontend for the core site.

## 2. Site architecture and navigation

Use two complementary outputs:

1. **Guide site:** `mdBook` built from Markdown in `docs/`.
2. **API reference:** `rustdoc` from
   `cargo doc --no-deps`.

Rustdoc remains authoritative for public signatures, trait bounds, examples,
and item-level documentation. The generated API output is copied below the
guide site without being committed:

```text
build/site/
├── index.html
├── guides/
├── use-cases/
├── validation/
└── api/stack_algebra/
```

The site should use a technical theme with a landing page, sidebar, local
search, readable code blocks, callouts, light/dark mode, and an API Reference
link. Use CSS and mdBook configuration before introducing a frontend
framework.

### Navigation

#### Overview

- Home
- Why fixed-size?
- Design contract
- Feature set

#### Getting started

- Installation
- First `Matrix<M, N, T>`
- `f32`/`f64` selection and casts
- Products, reductions, and output reuse
- Reading dimension and scalar compiler errors

#### Tutorials

- Fixed-size dense algebra
- Views and external buffers
- Dense factorizations
- Reusable factors and workspaces
- Geometry
- Sparse and block-sparse systems
- Embedded deployment

#### Use cases

- State estimation and covariance updates
- Linearized least squares
- NMPC-style fixed-horizon solves
- SLAM and pose-graph systems
- Generated numerical kernels
- MCU control loops

#### Reference and validation

- Generated Rust API reference
- Feature and storage references
- Numerical behavior
- Kernel and target support
- Eigen/faer/nalgebra methodology
- Benchmark reports
- QEMU and sanitizer status
- Roadmap, contributing, license, and releases

The landing page should include a short typed example, a summary of the design
constraints, links to the main sections, and links to getting started, the API
reference, and benchmarks. Avoid unsupported performance claims; link measured
results to their inputs, scalar type, target, and storage model.

## 3. Content and API coverage

Every public matrix class, view, geometry type, and solver must have three
layers of documentation:

1. **Rustdoc entry** for every public type and method.
2. **Runnable example** compiled by doctests or an executable example test.
3. **Guide/tutorial section** explaining the workflow and alternatives.

Each guide must answer:

- What problem does this type solve?
- When should it be chosen?
- What shape, scalar, storage, symmetry, rank, or conditioning assumptions
  apply?
- What is allocated, stored inline, or reusable?
- What is the smallest `f32`/`f64` example where applicable?
- Which `compute`, `try_compute`, `*_into`, and in-place APIs matter in loops?
- How are failures reported?
- Which alternative should be used for another problem shape or assumption?
- Where are the generated API reference and correctness tests?

### Matrix and storage coverage

| Type | Required guidance |
| --- | --- |
| `Matrix<M, N, T>` / `Vector<M, T>` | Fixed dimensions, column-major layout, scalar choice, operators, casts, output reuse |
| `MatrixBuf<MAX_ROWS, MAX_COLS, T>` | Bounded dimensions, RAM budgeting, resize semantics, fixed-size views |
| `Map` / `MapMut` | External column-major buffers, borrow lifetime, mutable zero-copy access |
| `StridedMap` / `StridedMapMut` | Padded, row-major, DMA, and interleaved layouts; stride validation |
| `Block` / `BlockMut` | Zero-copy submatrices, offsets, and decomposition from parent storage |
| Scalar CSC types | Patterns, capacity, symbolic reuse, construction, and solves |
| Block CSC/CSR types | Block dimensions, capacity, matvec, ordering, and factorization choices |

### Dense solver coverage

| Solver | Required “when to use” guidance |
| --- | --- |
| `Cholesky` | Symmetric positive-definite systems and lower-triangle semantics |
| `Ldlt` | Symmetric-indefinite systems, 1x1/2x2 pivots, thresholds, and reuse |
| `PartialPivLu` | General square systems, row pivoting, singularity, and conditioning |
| `HouseholderQr` | Full-rank square/tall least squares and Q application |
| `ColPivHouseholderQr` | Rank detection, column scaling, and pivot interpretation |
| `Svd` | Rank-deficient, ill-conditioned, and pseudo-inverse workflows |
| `SelfAdjointEigen` | Symmetric eigenproblems, sorted values, convergence, reconstruction |
| Triangular solvers | Solving from an existing factor and in-place RHS APIs |

Every solver page includes a minimal solve, a reusable-factor example, failure
notes, and a decision table for SPD, indefinite, general square, tall
least-squares, rank-deficient, and eigenvalue problems. One-shot factorization,
recomputation, and solve-only timing must be distinguished.

### Geometry, sparse, and use-case coverage

Apply the same standard to:

- Scalar sparse Cholesky and LDLT, including ordering, dense fallback,
  symbolic reuse, numeric recomputation, and multi-RHS solves.
- Block sparse Cholesky and LDLT, including block capacity, local pivots, and
  dense/scalar fallback boundaries.
- `Quaternion`, `AngleAxis`, and `RotationMatrix`, including construction,
  normalization, composition, conversion, and invalid-input behavior.
- `Isometry` and `AffineTransform`, including point/direction application,
  composition, inversion, and homogeneous conversion.
- Framework-neutral state estimation, least squares, NMPC, SLAM, generated
  kernels, and MCU examples.

Each page must state allocation behavior and provide at least one representative,
deterministic numerical workflow.

### Source migration

Use existing content as the starting source instead of rewriting everything:

| Existing file | Initial site destination |
| --- | --- |
| `README.md` | Home and getting-started material |
| `docs/features.md` | Overview and reference |
| `docs/api-usage.md` | Tutorials and API usage |
| `docs/use-cases.md` | Use-case pages |
| `docs/benchmarking.md` | Validation and benchmarks |
| `docs/roadmap.md` | Project information |

Start with `docs/` as the mdBook source root. Add `docs/SUMMARY.md` and
`docs/index.md`; split pages into subdirectories only when content size makes
that useful. Fix duplicate section numbering, normalize terminology, and keep
examples compilable as Rust doctests.

## 4. Generated docs, examples, and validation

Build the API reference with:

```sh
cargo doc --no-deps
```

Requirements:

- Keep `#![deny(missing_docs)]` enabled.
- Use intra-links for crate-local symbols.
- Publish API docs from the same commit as the guide site.
- Link from the guide to `/api/stack_algebra/` and back where possible.
- Keep guide search and rustdoc search as separate contexts.
- Exclude private implementation and benchmark-only modules from public docs.
- Keep a checked-in coverage manifest for user-facing matrix, view, geometry,
  sparse, and solver types; CI must verify each declaration has a guide page.
- Run a local-link check after the generated API is copied into the site so
  guide links to both Markdown and rustdoc targets are validated.

Examples must be validated by at least one of:

- `cargo test --doc`.
- An executable under `examples/` covered by `cargo test --examples`.
- A deterministic integration test for numerical output or storage layout.

Examples should cover both scalar types where relevant, avoid hidden
allocations, and state whether operations return `Result`/`Option`.

The validation area must explain Eigen/faer/nalgebra comparisons, benchmark
phases, storage/allocation models, QEMU limitations, stack-budget checks, and
Miri/sanitizer results. Benchmark reports should be generated separately from
the docs build and linked as artifacts or published reports.

## 5. Build, publishing, design, and versioning

Add `.github/workflows/docs.yml`.

### Pull requests

- Install a pinned mdBook version.
- Run formatting checks and `cargo test --doc`.
- Generate rustdoc and build mdBook.
- Verify internal links and the API link.
- Upload a preview artifact without deploying.

### Main branch

- Build guide and API docs together.
- Upload a Pages artifact and deploy it.
- Use a concurrency group so older deployments cannot overwrite newer ones.

### Nightly

Keep benchmarks in the existing nightly workflow. Publish reports separately;
docs deployment must not depend on long-running benchmark execution.

### Local development

```sh
mdbook serve docs
cargo doc --no-deps --open
```

The initial theme should provide responsive navigation, readable typography,
light/dark modes, visible keyboard focus, skip navigation, sufficient contrast,
and text alternatives for diagrams. Use stable relative URLs and defer version
switching until multiple releases are maintained.

## 6. Implementation phases

### Phase 0 — Scaffold

- Add `docs/SUMMARY.md`, `docs/index.md`, and `book.toml`.
- Add minimal custom CSS and local build instructions.
- Generate the API reference locally.
- Add a non-deploying docs CI check.

Exit criterion: `mdbook build docs` succeeds and the landing page links to all
canonical pages and the generated API reference.

### Phase 1 — Content normalization

- Organize navigation into overview, tutorials, use cases, reference, and
  validation.
- Normalize terminology and cross-links.
- Add the first matrix, views, solver, geometry, sparse, and embedded pages.
- Add runnable examples for each primary user journey.

Exit criterion: the guide contains documented paths for constructing a matrix,
performing a product, solving a small system, and locating the matching API
page.

### Phase 2 — Coverage and validation integration

- Complete the matrix/storage and solver coverage tables.
- Add all geometry and sparse API pages.
- Link benchmark, QEMU, and safety-validation results.
- Enforce doctest, API-generation, and link checks in CI.

Exit criterion: every public top-level type is discoverable through a guide or
the API reference, with an example and usage guidance.

### Phase 3 — Publishing and review

- Add opt-in GitHub Pages deployment and retain PR build artifacts.
- Complete responsive, mobile, and accessibility review.
- Add a custom domain only if needed.

Exit criterion: a main-branch push publishes the guide, API, and validation
links without manual steps.

### Phase 4 — Maintenance

- Require docs/API build success for releases.
- Add stale-link and missing-example checks.
- Add versioned docs only after multiple maintained releases.
- Document ownership and review expectations.

## 7. Acceptance checklist

- [ ] `mdbook build docs` succeeds on a clean checkout.
- [ ] `cargo test --doc` passes.
- [ ] `cargo doc --no-deps` passes.
- [ ] API coverage and generated-site link checks pass.
- [ ] Landing page links to getting started, tutorials, use cases, API,
      benchmarks, the feature set, and roadmap.
- [ ] API docs match the same revision as guide content.
- [ ] Every public matrix/storage type has a runnable example and usage guide.
- [ ] Every dense, sparse, and geometry solver/type has examples, assumptions,
      failure notes, and alternatives guidance.
- [ ] Allocation, layout, scalar, and failure semantics are documented.
- [ ] Native and embedded/no-std examples exist.
- [ ] Benchmark storage models and phases are distinguished.
- [ ] PR artifacts and main-branch deployment work.
- [ ] Keyboard navigation, dark mode, mobile layout, and links pass.

The first implementation should stop after Phase 0 until the end-to-end build
is green. Defer a custom frontend and broad page restructuring until that
build is working.
