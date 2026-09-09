# Tutorials

Learn `stack-algebra` through complete runnable examples, with figures and
terminal output generated from Rust execution during the documentation build.
Familiarity with Rust variables, functions, and arrays is useful; the guides
introduce the matrix and estimation concepts as they use them.

**Start with recursive curve fitting.** Learn a quadratic calibration from
one observation at a time using fixed-memory square-root RLS. The batch
companion covers caller-owned buffers and QR; the Kalman tutorial then follows
a changing physical state rather than fixed parameters.

| Walkthrough | The question you will answer |
| --- | --- |
| [Fit a curve online with RLS](tutorial-recursive-least-squares.md) | How can an embedded estimator learn from a stream without keeping every observation? |
| [Batch fitting from a borrowed buffer](tutorial-mapped-least-squares.md) | How can linear least squares fit a curved trend using caller-owned storage and QR? |
| [Follow a moving object](tutorial-kalman-1d.md) | How do a motion prediction and imperfect position readings estimate position and velocity? |

## Before you start

Run the host drivers on your computer. No sensors, special hardware, Python
setup, or external C++ library is needed to execute the Rust programs.

```sh
git clone https://github.com/yongkyuns/stack-algebra.git
cd stack-algebra
cargo run --example recursive_least_squares --no-default-features
cargo run --example mapped_least_squares --no-default-features
cargo run --example kalman_1d --no-default-features
```

Use an existing checkout instead of cloning again when appropriate. Run the
commands from the folder containing `Cargo.toml`. Cargo may download Rust
dependencies on the first build. The guide follows the development source,
which can be newer than the registry release. Record `git rev-parse HEAD`
when reporting a problem; see [Getting started](getting-started.md) to use the
library in another application.

Host reporting uses `std`, while the algebra runs with default features
disabled. The RLS estimator is a separate `std`/`alloc`-free module. The
example drivers themselves are not bare-metal executables or MCU benchmarks.

## Streaming calibration with recursive least squares

[Fit a curve online](tutorial-recursive-least-squares.md). Feed one `(x,y)`
pair at a time, retain a small triangular information factor and transformed
right-hand side, and obtain coefficients after every accepted observation.
The figures show progressive curve estimates, coefficient histories, online
prediction errors, and final residuals. Initial regularization and optional
forgetting are explicit, with prefix-by-prefix comparisons to matching batch QR.

The estimator's memory does not grow with the stream. History and plotting
snapshots belong only to the host driver. The tutorial distinguishes finite
coefficients from sufficient excitation and records memory sizes without
claiming measured hardware stack or timing performance.

## Batch fitting from a caller-owned buffer

[The batch companion](tutorial-mapped-least-squares.md) builds all rows of
`[x², x, 1]` in column-major storage, borrows them with `Map`, and solves with
column-pivoted QR. It uses the same noisy synthetic pairs as the RLS driver,
but its unregularized objective differs from the RLS example's explicit prior.

Batch solves remain useful on embedded systems when a finite calibration
window is already buffered. This companion specifically teaches views,
ownership, and factorization; RLS is the primary streaming example, not a
claim that all embedded least squares must be recursive.

## A two-state Kalman filter

[Follow the moving-object walkthrough](tutorial-kalman-1d.md). Start with a
cart on a track and imperfect position readings, then predict and correct
position and velocity. The guide explains the first correction, covariance,
and the full executed sequence. Unlike the fitting scenario, the fixed
Kalman inputs do not supply a ground-truth trajectory.

## Reproduce the figures

With Python 3.10 or newer installed:

```sh
python3 scripts/generate_tutorial_assets.py
python3 scripts/generate_rls_assets.py
python3 scripts/generate_tutorial_assets.py --check
python3 scripts/generate_rls_assets.py --check
```

Both commands execute Rust and reuse the same standard-library-only SVG
renderer. The [RLS walkthrough](tutorial-recursive-least-squares.md#reproduce-and-verify)
describes its exports; [Reproducing tutorial figures](tutorial-assets.md)
covers provenance and the full website build. `./scripts/build_docs.sh`
generates all tutorial assets before building the book.

## Continue to other workloads

[Getting started](getting-started.md) covers dense matrices and fixed dimensions.
[Choosing an API](api-usage.md) covers ownership, bounded storage, views,
Cholesky/LU/QR/SVD selection, failure handling, and reuse. The
[API reference](api-reference.md) gives exact method signatures.

For coordinate frames, rotations, fixed-topology sparse systems, and embedded
control loops, see [Common use cases](use-cases.md), [Capabilities and limits](features.md),
and [Platforms and embedded use](targets.md).
