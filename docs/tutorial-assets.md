# How tutorial figures are built

Tutorial plots and displayed terminal output are build artifacts of the Rust
examples. They are not generated artwork, hand-maintained expected output, or
results of Python implementations of the solvers.

```text
Rust examples → full-precision exports + actual terminal output → SVG figures → mdBook
```

## Reproduce a documentation build

From a Git checkout, with Cargo, Python 3.10 or newer, and mdBook installed:

```sh
python3 scripts/generate_tutorial_assets.py
python3 scripts/generate_rls_assets.py
python3 scripts/generate_tutorial_assets.py --check
python3 scripts/generate_rls_assets.py --check
./scripts/build_docs.sh
```

The first generator executes batch fitting and the Kalman filter, writing
`docs/generated/tutorials/`. The second executes streaming RLS and writes
`docs/generated/rls/`. Each `--check` executes its programs again and requires
identical files, including provenance. The combined build runs both generators
before mdBook; `mdbook build docs` alone on a clean checkout is not sufficient.

Both generators share the standard-library-only SVG renderer. There are no
Python numerical or plotting dependencies. CI pins Rust and Python versions
and the runner image family. SVG text alternatives and marker/line styles
identify observations, estimates, and references without relying on color.
The white canvas keeps figures readable in the book's light and dark themes.

## Batch and Kalman exports

Normal example commands print human-readable output. Add `-- --csv` to
request numeric rows. The batch quadratic also accepts `-- --curve-csv`:

```sh
cargo run --quiet --no-default-features --example mapped_least_squares -- --csv
cargo run --quiet --no-default-features --example mapped_least_squares -- --curve-csv
cargo run --quiet --no-default-features --example kalman_1d -- --csv
```

The batch observation export reports `x`, `observed_y`, `reference_y`, injected
`noise`, `fitted_y`, `residual`, fitted coefficients `a,b,c`, reference
coefficients `reference_a,reference_b,reference_c`, `residual_norm`, and
`sample_count`. Coefficient order is `[x², x, 1]`. This synthetic regression
has no assigned physical units.

The batch curve export is `x,fitted_y,reference_y,sample_count`. Rust evaluates
both curves at 201 points, captured as `mapped_least_squares_curve.csv`.
Those are plotting points, not observations used in the fit. The batch/Kalman
provenance schema is version 2; the old straight-line schema is not accepted.

The Kalman export reports time and noise settings, initial conditions,
predicted and corrected state and covariance, innovation, innovation variance,
and gain entries. `update_position` returns the intermediate values it used;
the reporter does not calculate a second correction.

For Kalman data, `p00` and `p11` are position and velocity variances in m² and
(m/s)²; `p01` and `p10` are covariance in m²/s. Prefixes `initial_` and
`predicted_` distinguish snapshots; unprefixed state and covariance are
post-correction. Position gain is dimensionless; velocity gain has units s⁻¹.
Innovation and measurement variances are in m²; acceleration variance is in
(m/s²)². The fixed readings do not supply a true trajectory or measured velocity.

## Recursive least-squares exports

```sh
cargo run --quiet --no-default-features --example recursive_least_squares -- --csv
cargo run --quiet --no-default-features --example recursive_least_squares -- --curve-csv
```

All reporting modes execute the same estimator update path. Batch and RLS
share `examples/support/quadratic_data.rs`, which contains only deterministic
inputs, not a solver. RLS visits source row `(17 * step) % 40` for zero-based
`step`, preserving the same pairs while covering the input range early.

`trace.csv` contains one record per accepted observation:

| Fields | Meaning |
| --- | --- |
| `step`, `source_row`, `x`, `observed_y`, `reference_y` | One-based update count, zero-based fixture row, and actual input/reference values. |
| `prediction_before`, `innovation` | Prediction from the previous estimate, then observed minus that prediction. |
| `a,b,c` | Coefficients after this update, converted back to original x coordinates. |
| `final_fitted_y`, `final_residual` | This observation evaluated using the final model; not its online prediction error. |
| `reference_a,reference_b,reference_c` | Synthetic generating coefficients, never supplied to the estimator. |
| `input_scale`, `prior_precision`, `forgetting` | Normalization, prior precision in the normalized basis, and forgetting factor. The host scenario uses zero prior mean. |
| `sample_count`, `storage_bytes_f32`, `storage_bytes_f64` | Export row count and actual estimator object sizes for the execution target; not peak stack measurements. |

`curves.csv` contains `step,x,fitted_y,reference_y,a,b,c,sample_count`.
There are four snapshots after 5, 10, 20, and 40 accepted samples, each
containing 201 evaluations performed by Rust. Its `sample_count` is **804
export rows**, not 804 observations. Snapshot figures show only observations
received by the indicated step; no future observation is displayed early.

The RLS generator has its own provenance schema, version 1, identified by
`scenario = quadratic-square-root-rls`. Its files are
[trace data](generated/rls/trace.csv), [snapshot curves](generated/rls/curves.csv),
[terminal output](generated/rls/output.txt), [provenance](generated/rls/provenance.json),
and [resolved lockfile](generated/rls/Cargo.lock). See the
[RLS walkthrough](tutorial-recursive-least-squares.md) for the exact objective,
including how forgetting weights the prior.

## Precision and rendering boundaries

CSV uses 17 digits after the decimal point in scientific notation. `f32`
values are promoted exactly to `f64` for printing; this adds no computational
precision. The current RLS and batch figures use `f64`, while Kalman uses
`f32`. Parsers require exact headers, column order, finite values, and complete
row counts. Schema changes require explicit validator and test updates.

The renderer maps exported values to axes, joins points, formats labels, and
may take square roots of variances for standard-deviation bands. It does not
fit coefficients, evaluate a new polynomial grid, or propagate a Kalman/RLS
estimator. Consistency assertions check identities against the exports but
never substitute their calculations for the plotted data.

Synthetic reference curves are distinct from fitted estimates. Injected noise,
online pre-update error, and final-fit residuals are separate quantities.
The Kalman first-update diagram uses one shared position scale and identifies
its bands as assumed marginal uncertainty, not actual measured error.

## Validation and provenance

Generation validates schemas, finite values, counts, time/step semantics,
coefficient and residual consistency, dense curve grids, and correction
metadata. Actual human-readable stdout must match the formatted CSV values.
These are reporting checks, not independent solver validation.

CI separately runs numerical Rust tests. RLS is checked against augmented
batch QR at every prefix with matching normalization, prior and forgetting
weights in `f32` and `f64`. Tests cover early and late atomic rejection and
recovery, repeated inputs, sample ordering, and a longer stream. The same
estimator helper is compiled by a separate `no_std` Cortex-M consumer; that
is not physical-device execution or timing evidence.

Each generator records the executed Git revision/tree, dirty-worktree status,
source hashes including the library and shared renderer, compiler/Cargo/Python
versions, selected environment flags, commands, and generated-file hashes.
On PR builds, the recorded revision can be GitHub's test merge commit rather
than the branch head. The manifest has no timestamps or machine-local absolute
paths. Rendering font appearance can vary across viewers even when SVG bytes match.

The batch/Kalman [provenance](generated/tutorials/provenance.json),
[lockfile](generated/tutorials/Cargo.lock),
[batch observations](generated/tutorials/mapped_least_squares.csv),
[batch curves](generated/tutorials/mapped_least_squares_curve.csv), and
[Kalman data](generated/tutorials/kalman_1d.csv) are also published with the site.
Lockfiles are generated only when absent; subsequent execution uses `--locked`.
Replay an older build using its source revision, toolchain, and saved lockfile.
There is no cross-platform bitwise-identity or fully hermetic-build claim.

Normal generation deletes its previous output directory and stages new files
until all checks pass. Failure cannot silently reuse an old image. `--check`
is non-mutating and requires identical regenerated bytes. CI checks both sets
after the site build; only successful main-branch documentation runs publish.
Generated assets are ignored by Git.

For validation/rendering unit tests:

```sh
python3 -m unittest discover -s scripts -p 'test_tutorial_assets.py' -v
python3 -m unittest discover -s scripts -p 'test_rls_assets.py' -v
```

These tests use explicitly synthetic fixtures. Production generation has no
fixture mode and always executes Rust. Tests include corrupted curve points,
future-observation isolation in snapshots, stale-output failures, and provenance.
