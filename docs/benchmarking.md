# Performance

Benchmarks are included to give a practical sense of the runtime cost of
`stack-algebra` operations and where performance changes across matrix sizes.
They are most useful for orientation and regression tracking. For a product or
algorithm decision, measure the actual workload on the actual target.

## Accepted production improvements — September 6, 2026

These charts summarize **same-runner before/after validation** for performance
changes that are now part of `main`. They report relative improvement over the
immediately preceding production implementation, not cross-library latency.
Experimental candidates rejected because they regressed other workloads are
not included.

### Dense reusable solves

![Accepted dense-solve performance improvements](assets/accepted-performance-dense-solves.svg)

The dense chart covers reusable LDLT multi-RHS solves plus representative D=32
lower- and upper-triangular solves. `unchanged` means the accepted
implementation retained the previous fast path for that case rather than
trading it away for gains elsewhere.

### Small `f32` dot products

![Accepted small-f32 dot-product improvements](assets/accepted-performance-small-dot.svg)

The dot-product chart covers AVX2/FMA lengths validated in repeated hosted
runs. The implementation remains generic over vector length; the listed sizes
are measurement points, not dispatch cases.

The checked-in [accepted-change source data](assets/accepted-performance.csv)
includes merged-change provenance. Its SVGs are generated during the docs
build.

## Exhaustive benchmark reference — September 6, 2026

The full reference below is generated from one successful scheduled benchmark
snapshot rather than assembled from unrelated runs. The snapshot contains
**1,389 measurements across 111 operation groups** and covers the complete
scheduled benchmark matrix available at that commit.

| Reference section | Operation groups | What it covers |
| --- | ---: | --- |
| [Dense operations](generated/benchmark-dense.md) | 8 | matmul, matvec, dot, norm; `f32` and `f64` |
| [Decompositions and solves](generated/benchmark-solvers.md) | 30 | LU, LLT/Cholesky, LDLT, QR/CPQR, triangular solves, eigen, SVD |
| [Sparse operations](generated/benchmark-sparse.md) | 26 | matvec, symbolic analysis, assembly, factor/refactor, solve, multiple sparsity patterns |
| [Structured and specialized workloads](generated/benchmark-structured.md) | 47 | block sparse, dense LDLT stress cases, fused paths, mapped views, workload-decision cases |
| [All benchmark results](generated/benchmark-all.md) | 111 | every row in the source snapshot, including unmatched controls |

The snapshot was produced by the scheduled benchmark workflow from commit
`5fb2755e902da16873252cb7017c23c74364f3b8` on Linux/x86-64 using an AMD EPYC
9V74 runner and Rust 1.98.1. It therefore includes the merged LDLT multi-RHS
work but predates the later same-day small-dot and triangular-solve changes;
the accepted-change charts above cover those later production improvements.

The [snapshot CSV](generated/benchmark-snapshot.csv) and
[raw provenance](data/benchmark-snapshot-2026-09-06-provenance.txt) are checked
in as the source of the generated reference. The provenance records
`git_dirty=true` because the report job downloads generated benchmark inputs
into the checkout before it captures `git status`; the source revision remains
the recorded commit SHA.

The generated charts include only shapes measured by at least two
implementations. The tables retain every row, including single-implementation
controls and extra reference-library sizes, so missing measurements are never
silently presented as equivalent comparisons.

### Measurement scope

The scheduled snapshot uses short Criterion windows so a broad matrix can run
regularly. It is a useful comprehensive **hosted snapshot**, not a release
qualification or a guarantee for embedded targets. Matrix shape, scalar type,
compiler flags, target features, memory placement, and machine load can all
change results.

A separate release benchmark workflow exists for longer measurements on the
pinned benchmark host. That pinned run is the authority for a future release
snapshot once such a qualification has been executed; the docs generator can
consume the resulting CSV without redesigning these pages.

## Matching the measured phase matters

Factorization and reusable solve are deliberately shown separately. A solver
that factors once and solves repeatedly has a different performance profile
from a factor-and-solve call.

Sparse measurements likewise keep symbolic analysis, numeric assembly,
factorization, refactorization, and solve separate. Combining those phases
would hide the costs that matter in real application loops.

## Fresh nightly measurements

The repository runs the broad benchmark workflow nightly and publishes a
`nightly-benchmark-report` artifact containing the self-contained HTML report,
CSV measurements, raw inputs, and provenance. The nightly matrix also includes
the dedicated dense multi-RHS benchmark for future snapshots.

Open the [nightly benchmark workflow](https://github.com/yongkyuns/stack-algebra/actions/workflows/nightly-bench.yml),
choose a successful run, and download `nightly-benchmark-report` for the newest
measurements.

## Running focused benchmarks locally

For fixed-size dense comparisons:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --bench comparison
```

For reusable dense multi-RHS solves:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --bench dense_multi_rhs
```

For embedded work, host microbenchmarks are only a first check. The most useful
measurement is the complete operation on the target MCU or processor with the
same scalar type, dimensions, compiler flags, and memory placement used by the
application.
