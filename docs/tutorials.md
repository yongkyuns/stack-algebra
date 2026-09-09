# Tutorials

Learn `stack-algebra` through complete runnable examples, with figures and
terminal output generated from the code during the documentation build.
Familiarity with Rust variables, functions, and arrays is useful; the guides
introduce the matrix and estimation concepts as they use them.

**Start with fitting a curve.** Recover a quadratic trend from 40 visibly noisy
observations using a borrowed design matrix and QR. Then try the Kalman filter,
which updates position and velocity estimates as new readings arrive.

| Walkthrough | The question you will answer |
| --- | --- |
| [Fit a curve to noisy measurements](tutorial-mapped-least-squares.md) | How can linear least squares recover a curved trend, and what do the residuals tell us? |
| [Follow a moving object](tutorial-kalman-1d.md) | How can a prediction and an imperfect position reading work together to estimate position and velocity? |

## Before you start

Run these programs on your computer, not on a microcontroller. Inputs are
constructed entirely by the examples. No sensors, special hardware, Python
setup, or external C++ library is needed to run the Rust programs.

You need Git and a stable Rust installation with Cargo:

```sh
git clone https://github.com/yongkyuns/stack-algebra.git
cd stack-algebra
cargo run --example mapped_least_squares --no-default-features
cargo run --example kalman_1d --no-default-features
```

Use an existing checkout instead of cloning again when appropriate. Run the
commands from the folder containing `Cargo.toml`. Cargo may download Rust
dependencies during the first build. The first program prints three fitted
coefficients and 40 rows of predictions and residuals; the second prints ten
steps of position and velocity estimates.

The guide follows the development source, which can be newer than the registry
release. `cargo add` is not a substitute for checking out these examples.
Record `git rev-parse HEAD` when reporting a problem. See
[Getting started](getting-started.md) to add the library to another application.

These host programs use Rust's standard library for printing, while
`--no-default-features` exercises the algebra library's `no_std` core. This does
not turn a desktop example into a bare-metal executable or measure an MCU's
memory budget. Each walkthrough includes snippets and the complete listing
from its own source version, not a second implementation to assemble by hand.
GitHub source links point to `main`, which can move ahead of a built guide.

## A quadratic fit from a caller-owned buffer

[Fit a curve to noisy measurements](tutorial-mapped-least-squares.md). Build
rows of `[x², x, 1]` in column-major storage, borrow them with `Map`, and ask
column-pivoted QR for three coefficients. A curved model can still be linear in
its coefficients. The tutorial explains that distinction, shows a visibly
noisy dataset, and separates the fitted curve from its known synthetic reference.

The residual plot shows what remains after fitting. The smooth curves are
exported from a 201-point grid evaluated by Rust; the renderer does not
calculate another fit or polynomial. Try changing the noise scale, then test
why fewer than three distinct inputs cannot determine a unique quadratic.

The [example](https://github.com/yongkyuns/stack-algebra/blob/main/examples/mapped_least_squares.rs)
and [tests](https://github.com/yongkyuns/stack-algebra/blob/main/tests/mapped_least_squares.rs)
keep numerical work separate from reporting and verification. For other memory
layouts, see [external buffers and views](api-usage.md).

## A two-state Kalman filter

[Follow the moving-object walkthrough](tutorial-kalman-1d.md). Start with a
cart on a track and imperfect position readings, then predict and correct
position and velocity. The guide works through the first reading, explains
uncertainty, and shows the full executed sequence.

The [example](https://github.com/yongkyuns/stack-algebra/blob/main/examples/kalman_1d.rs)
and [tests](https://github.com/yongkyuns/stack-algebra/blob/main/tests/kalman_1d.rs)
use a small model to explain the library calls, not to implement a complete
tracking system. Unlike the quadratic's explicitly synthetic reference, the
fixed Kalman readings do not supply a ground-truth trajectory.

## Reproduce the figures

With Python 3.10 or newer installed:

```sh
python3 scripts/generate_tutorial_assets.py
python3 scripts/generate_tutorial_assets.py --check
```

Generation executes the Rust programs and requires no Python numerical or
plotting packages. [Reproducing tutorial figures](tutorial-assets.md) explains
the CSV schemas, source and dependency provenance, and full website build.

## Continue to other workloads

[Getting started](getting-started.md) covers dense matrices and fixed dimensions.
[Choosing an API](api-usage.md) covers ownership, bounded storage, views,
Cholesky/LU/QR/SVD selection, failure handling, and reuse. The
[API reference](api-reference.md) gives exact method signatures.

For coordinate frames and rotations, fixed-topology sparse or block-sparse
systems, and embedded control loops, start with [Common use cases](use-cases.md),
[Capabilities and limits](features.md), and [Platforms and embedded use](targets.md).
These are independent next steps, not prerequisites for the two walkthroughs.
