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
| Matrix-vector multiply | `f32` | 36 | 99 | 174 | 0.36× |
| Matrix-vector multiply | `f64` | 104 | 63 | 163 | 1.65× |
| LLT factor | `f32` | 3,594 | 3,991 | 3,367 | 0.90× |
| LLT factor | `f64` | 3,294 | 3,847 | 2,869 | 0.86× |
| LDLT factor | `f32` | 4,793 | 4,953 | 2,783 | 0.97× |
| LDLT factor | `f64` | 5,244 | 6,245 | 3,519 | 0.84× |
| LU factor | `f32` | 3,671 | 4,757 | 7,576 | 0.77× |
| LU factor | `f64` | 5,689 | 7,693 | 6,960 | 0.74× |
| QR factor | `f32` | 12,643 | 11,422 | — | 1.11× |
| QR factor | `f64` | 9,499 | 12,938 | — | 0.73× |
| Column-pivoted QR factor | `f32` | 16,890 | 14,043 | — | 1.20× |
| Column-pivoted QR factor | `f64` | 15,437 | 11,961 | — | 1.29× |

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
  scaled overflow fallback.

The comparison benchmark now calls `Matrix::matvec_into` for stack-algebra;
the previous version called the generic `mul_into` matrix-multiply path while
Eigen was measuring its dedicated matrix-vector product. Focused `32×32`
measurements after that correction put f32/f64 matvec at approximately 36/104
ns per operation. The remaining material gap is f64 matvec at about 1.6× the
short native Eigen sample; LU and pivoted QR are close to the native Eigen
reference.

## Remaining priorities

1. Improve f64 matvec, whose focused result remains about 1.6× Eigen.
2. Align SVD shape coverage across libraries before drawing performance
   conclusions from those rows.
3. Validate the native kernels on non-x86 targets before adding more x86-only
   specialization.
