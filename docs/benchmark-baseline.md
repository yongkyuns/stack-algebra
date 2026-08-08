# Native benchmark baseline

This baseline was generated on 2026-08-08 with the fast full sweep. It uses
the six Cargo benchmark targets, 10 ms warmup, 20 ms measurement, and 10
Criterion samples. Rust and Eigen were both compiled for the host CPU:

- Rust: `RUSTFLAGS="-C target-cpu=native"`
- Eigen: `CXXFLAGS="-march=native"`

The run completed in 166 seconds on a warm build and produced 1,931 report
rows. The generated report is `benchmark-report/index.html`; the source data
is `benchmark-report/results.csv`.

## Representative 32×32 results

Values are median nanoseconds per operation. Lower is better. Ratios are
stack-algebra divided by Eigen.

| Operation | Scalar | Stack-algebra | Eigen | Faer | Stack/Eigen |
| --- | --- | ---: | ---: | ---: | ---: |
| Matrix multiply | `f32` | 1,021 | 1,378 | 1,088 | 0.74× |
| Matrix multiply | `f64` | 1,637 | 2,493 | 1,827 | 0.66× |
| Matrix-vector multiply | `f32` | 33 | 52 | 174 | 0.64× |
| Matrix-vector multiply | `f64` | 68 | 61 | 163 | 1.11× |
| LLT factor | `f32` | 3,594 | 3,991 | 3,367 | 0.90× |
| LLT factor | `f64` | 3,294 | 3,847 | 2,869 | 0.86× |
| LDLT factor | `f32` | 4,793 | 4,953 | 2,783 | 0.97× |
| LDLT factor | `f64` | 5,244 | 6,245 | 3,519 | 0.84× |
| LU factor | `f32` | 3,671 | 4,757 | 7,576 | 0.77× |
| LU factor | `f64` | 5,689 | 7,693 | 6,960 | 0.74× |
| QR factor | `f32` | 12,643 | 11,422 | — | 1.11× |
| QR factor | `f64` | 9,499 | 12,938 | — | 0.73× |
| Column-pivoted QR factor | `f32` | 11,630 | 13,263 | — | 0.88× |
| Column-pivoted QR factor | `f64` | 11,851 | 14,051 | — | 0.84× |

Focused native SVD factorization measurements use the same tall shapes as the
Eigen reference:

| Shape | Scalar | Stack-algebra | Eigen | Stack/Eigen |
| --- | --- | ---: | ---: | ---: |
| 6×3 | `f32` | 740 | 930 | 0.80× |
| 6×3 | `f64` | 984 | 1,180 | 0.83× |
| 15×6 | `f32` | 4,400 | 4,449 | 0.99× |
| 15×6 | `f64` | 5,400 | 6,861 | 0.79× |

The fast profile is for triage. Use longer measurement windows before making
release decisions, especially for sub-microsecond operations. The LU values
above are from focused sequential measurements after the follow-up kernel
change; the parallel sweep is intentionally less stable because benchmark
processes share the CPU.

## Follow-up kernel changes

After the initial native baseline, three targeted changes were made:

- The AVX2/FMA backend now performs f32 and f64 matvec accumulation with FMA
  instead of delegating to the non-FMA reduction.
- LU factorization updates contiguous column tails through the SIMD rank-update
  kernel rather than scalar strided row updates.
- Pivoted QR column norms use the SIMD dot kernel while retaining the existing
  scaled overflow fallback; the max-absolute scan is now fallback-only for
  finite norms.
- Tall SVDs use a fixed-size Householder QR preconditioner before the reduced
  Jacobi iteration; the pairwise Jacobi statistics use one fused packet pass.
- The Jacobi fast path now uses raw finite dot products directly and retains
  scaled accumulation only for overflow, underflow, or zero-norm cases.
- Sparse CSC matvec now works through borrowed contiguous value, index, and
  vector/output slices instead of matrix indexing in the inner loop. In a
  longer isolated 32×32 banded `f32` run, this reduced stack-algebra from
  approximately 7.65 µs to 6.82 µs, matching the Eigen reference at about
  6.82 µs. The change is portable and does not add an ISA-specific path.
- `MatrixScalar::mul_add` now provides a compile-time fused-accumulation hook;
  x86 and Arm floating-point backends use the scalar FMA operation while the
  portable implementation retains `add + multiply`. The isolated 32×32
  banded `f64` path improved to approximately 7.0 µs from about 8.2 µs.

The comparison benchmark now calls `Matrix::matvec_into` for stack-algebra;
the previous version called the generic `mul_into` matrix-multiply path while
Eigen was measuring its dedicated matrix-vector product. Focused `32×32`
Repeated native sequential measurements after that correction put f32/f64
matvec at approximately 33/68 ns per operation. The f64 result is within about
15% of the Eigen reference; the earlier 1.6× result came from a parallel fast
sweep sharing the CPU and a stale non-native local benchmark artifact. LU and
pivoted QR are at or below the native Eigen reference in the focused native
measurements.

## Remaining priorities

1. Keep f64 matvec under regression monitoring; the focused native gap is now
   approximately 1.1× Eigen and does not justify a bespoke kernel.
2. Keep the larger tall SVD path under regression monitoring; the focused
   native measurements are now at or below the Eigen reference for both
   scalars at 15×6.
3. Validate the native kernels on non-x86 targets before adding more x86-only
   specialization.
