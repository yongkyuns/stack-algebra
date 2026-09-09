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
disabled and writes `docs/generated/tutorials/`. The second executes both
again and requires identical files, including the provenance manifest. The
combined documentation build also runs generation before mdBook; running
`mdbook build docs` alone on a clean checkout is not sufficient.

The renderer uses only Python's standard library and writes SVG directly.
There are no numerical or plotting dependencies to install. The documentation
workflow pins its Rust and Python versions and runner image family. The SVGs
have text alternatives and distinguish observations from estimates using
marker shapes, not color alone. The white canvas keeps them readable in the
book's light and dark themes.

## Export contract

Normal example commands retain their human-readable output. Add `-- --csv`
to request a CSV header followed by numeric rows:

```sh
cargo run --quiet --no-default-features --example mapped_least_squares -- --csv
cargo run --quiet --no-default-features --example kalman_1d -- --csv
```

Both formats use the same calculation path. The line-fit exporter reports
inputs, fitted values, residuals, coefficients, and the residual norm. The
Kalman exporter reports time and noise settings, initial conditions,
predicted and corrected state and covariance, innovation, innovation variance,
and the two gain entries. `update_position` returns the intermediate values
it used; the reporter does not calculate another correction.

CSV values use 17 digits after the decimal point in scientific notation.
`f32` values are promoted exactly to `f64` for printing; this does not add
precision to the filter. Initial conditions and fixed settings are repeated
on each row so each record is self-describing. `sample_count` detects
truncated exports. The parser requires the exact header, column order, finite
values, and a complete set of rows; changes to the schema require an explicit
renderer/test update.

`p00` and `p11` are the position and velocity variances, in m² and (m/s)²;
`p01` and `p10` are the shared covariance in m²/s. The prefixes `initial_`
and `predicted_` distinguish snapshots; unprefixed covariance entries and
`position_m`/`velocity_m_s` are post-correction. The position gain is
dimensionless; `velocity_gain` has units s⁻¹. `innovation_variance` and
`measurement_variance` are in m²; `acceleration_variance` is in (m/s²)².

## What the renderer may do

The renderer may map exported values to axes, connect samples, format labels,
and take the square root of a variance to show a standard-deviation interval.
It does not fit a line, solve a linear system, or propagate a Kalman filter.
The first-update diagram uses a shared position scale and labels its bands as
marginal uncertainty, not ground truth. No true trajectory or measured velocity
is invented for the current fixed-input example.

The main line-fit chart and the separate residual chart have distinct, labelled
axes. This makes small differences visible without exaggerating the main plot.
The worked arithmetic in the prose remains an explanation of the checked-in
teaching inputs; modifying a tutorial scenario also requires reviewing that
prose and its numerical tests.

## Validation and provenance

Generation checks schemas, finite values, sample counts, residual consistency,
correction diagnostics, and covariance validity. It compares the actual
human-readable stdout to the formatted CSV values. These consistency checks
are not a replacement for the existing independent Rust numerical tests, which
the documentation workflow also runs.

Each build records the executed Git revision and tree, dirty-worktree status,
source hashes (including the library implementation), compiler details, Cargo
and Python versions, selected environment flags, commands, and SHA-256 hashes
of the generated files. There are no timestamps or machine-local absolute paths
in the manifest. On a pull request, the revision is the actual checked-out
revision, which may be GitHub's test merge commit rather than the branch head.

The [provenance manifest](generated/tutorials/provenance.json),
[resolved Cargo.lock](generated/tutorials/Cargo.lock),
[line-fit data](generated/tutorials/mapped_least_squares.csv), and
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
has no fixture mode and always executes Rust.
