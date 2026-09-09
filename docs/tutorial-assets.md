# How tutorial figures are built

Tutorial plots and displayed terminal output are build artifacts of the Rust
examples. They are not generated artwork, hand-maintained expected output, or
results of a Python implementation of the solver.

```text
Rust examples → full-precision CSV + actual terminal output → SVG figures → mdBook
```

## Reproduce a documentation build

From a Git checkout, with Cargo, Python 3.10 or newer, and mdBook installed:

```sh
python3 scripts/generate_tutorial_assets.py
python3 scripts/generate_tutorial_assets.py --check
./scripts/build_docs.sh
```

The first command executes both examples with the library's default features
disabled and writes `docs/generated/tutorials/`. The second executes them
again and requires identical files, including the provenance manifest. The
combined documentation build also runs generation before mdBook; running
`mdbook build docs` alone on a clean checkout is not sufficient.

The renderer uses only Python's standard library and writes SVG directly.
There are no numerical or plotting dependencies to install. The documentation
workflow pins its Rust and Python versions and runner image family. The SVGs
have text alternatives and distinguish observations, fits, and references by
marker shapes and line styles, not color alone. The white canvas keeps them
readable in the book's light and dark themes.

## Export contract

Normal example commands print human-readable output. Add `-- --csv` to
request a CSV header followed by numeric rows. The quadratic example also
accepts `-- --curve-csv` for its separate dense plotting grid:

```sh
cargo run --quiet --no-default-features --example mapped_least_squares -- --csv
cargo run --quiet --no-default-features --example mapped_least_squares -- --curve-csv
cargo run --quiet --no-default-features --example kalman_1d -- --csv
```

The quadratic example uses one calculation path for all three reporting modes.
Its observation export contains `x`, `observed_y`, `reference_y`, injected
`noise`, `fitted_y`, `residual`, fitted coefficients `a,b,c`, synthetic reference
coefficients `reference_a,reference_b,reference_c`, `residual_norm`, and
`sample_count`. Coefficient order is `[x², x, 1]`. Inputs and outputs have no
assigned physical units in this synthetic regression example.

The curve export contains `x,fitted_y,reference_y,sample_count`. Rust evaluates
both curves at 201 inputs across the observation interval; those are plotting
points, not additional measurements used in fitting. The generator captures
this stream as `mapped_least_squares_curve.csv`. The provenance schema is now
version 2; the older straight-line schema is deliberately not accepted.

The Kalman exporter reports time and noise settings, initial conditions,
predicted and corrected state and covariance, innovation, innovation variance,
and the two gain entries. `update_position` returns the intermediate values
it used; the reporter does not calculate another correction.

CSV values use 17 digits after the decimal point in scientific notation.
`f32` values are promoted exactly to `f64` for printing; this does not add
precision to the filter. Initial conditions and fixed settings are repeated
on each observation row. `sample_count` detects truncated exports and refers
to the number of rows in that particular stream (40 observations or 201 curve
points for the quadratic). The parser requires the exact header, column order,
finite values, and a complete set of rows; schema changes require explicit
renderer and test updates.

For the Kalman data, `p00` and `p11` are position and velocity variances, in m²
and (m/s)²; `p01` and `p10` are shared covariance in m²/s. Prefixes `initial_`
and `predicted_` distinguish snapshots; unprefixed covariance and
`position_m`/`velocity_m_s` are post-correction. Position gain is dimensionless;
`velocity_gain` has units s⁻¹. Innovation and measurement variances are in m²;
acceleration variance is in (m/s²)².

## What the renderer may do

The renderer maps exported values to axes, joins exported points, formats
labels, and takes square roots of variances to show standard-deviation bands.
It does not fit coefficients, solve a system, evaluate a new polynomial grid,
or propagate a Kalman filter. Its consistency validator checks polynomial
identities against the exports but never uses those checks as plotting data.

The quadratic reference is explicitly synthetic and generated in Rust. It is
not a measured trajectory or a fitted result. The residuals are observed minus
fitted, not the injected noise. The main and residual plots have distinct,
labelled axes. The Kalman example supplies no true trajectory or velocity
measurement, so none is invented. Its first-update diagram uses one shared
position scale and labels the bands as marginal uncertainty, not actual error.

## Validation and provenance

Generation checks schemas, finite values, counts, residual consistency,
polynomial coefficients and reference metadata, the dense curve's interval and
grid, correction diagnostics, and covariance validity. It compares actual
human-readable stdout with formatted CSV values. These consistency checks are
not a replacement for the independent Rust numerical tests, which the
documentation workflow also runs.

Each build records the executed Git revision and tree, dirty-worktree status,
source hashes (including the library implementation), compiler details, Cargo
and Python versions, selected environment flags, commands, and SHA-256 hashes
of the generated files. There are no timestamps or machine-local absolute paths
in the manifest. On a pull request, the revision is the actual checked-out
revision, which may be GitHub's test merge commit rather than the branch head.

The [provenance manifest](generated/tutorials/provenance.json),
[resolved Cargo.lock](generated/tutorials/Cargo.lock),
[quadratic observation data](generated/tutorials/mapped_least_squares.csv),
[quadratic curve data](generated/tutorials/mapped_least_squares_curve.csv), and
[Kalman data](generated/tutorials/kalman_1d.csv) are published with the site.
The lockfile is generated only when absent; execution then uses `--locked`.
To replay an older build, use its source revision, toolchain, and saved
lockfile. Without that saved lockfile, a later dependency resolution can differ.
No cross-platform bitwise-identity or fully hermetic-build claim is made.

A normal generation deletes its previous output before execution and stages
new assets until all checks pass. A failed run leaves no old tutorial assets
available to the site build. `--check` is non-mutating and fails when any
regenerated file differs. CI runs it after building the site, and only a
successful main-branch documentation workflow may publish. Generated assets
are ignored by Git; there is no stale committed-image fallback.

For renderer and validation unit tests, run:

```sh
python3 -m unittest discover -s scripts -p 'test_tutorial_assets.py' -v
```

Those tests use synthetic fixtures only. The production documentation command
has no fixture mode and always executes Rust. Tests also verify that changing
an exported interior grid point changes the drawing: the renderer cannot
silently substitute a curve calculated from its own coefficients.
