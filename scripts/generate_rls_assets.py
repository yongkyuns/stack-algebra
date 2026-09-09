#!/usr/bin/env python3
"""Execute square-root RLS; draw exported values with the existing SVG renderer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import generate_tutorial_assets as shared

TRACE_COLUMNS = (
    "step,source_row,x,observed_y,reference_y,prediction_before,innovation,a,b,c,"
    "final_fitted_y,final_residual,reference_a,reference_b,reference_c,input_scale,"
    "prior_precision,forgetting,sample_count,storage_bytes_f32,storage_bytes_f64"
).split(",")
CURVE_COLUMNS = "step,x,fitted_y,reference_y,a,b,c,sample_count".split(",")


def validate(trace: list[dict[str, float]], curves: list[dict[str, float]]) -> None:
    first, final = trace[0], trace[-1]
    constants = ["reference_a", "reference_b", "reference_c", "input_scale", "prior_precision",
                 "forgetting", "storage_bytes_f32", "storage_bytes_f64"]
    shared.require(first["input_scale"] > 0 and first["prior_precision"] > 0
                   and 0 < first["forgetting"] <= 1, "invalid RLS configuration")
    for key in ("storage_bytes_f32", "storage_bytes_f64"):
        shared.require(first[key] > 0 and first[key].is_integer(), "invalid storage size")
    shared.require(sorted(r["source_row"] for r in trace) == list(range(len(trace))),
                   "source rows must be a complete permutation")
    previous = {"a": 0.0, "b": 0.0, "c": 0.0}  # This host scenario has a zero prior mean.
    for step, row in enumerate(trace, 1):
        shared.require(row["step"] == step, "missing or reordered RLS step")
        shared.require(all(row[key] == first[key] for key in constants), "changing RLS configuration")
        x = row["x"]
        # These are export consistency assertions, never a replacement solver.
        shared.close(row["prediction_before"], (previous["a"] * x + previous["b"]) * x + previous["c"],
                     "pre-update prediction", 1e-9)
        shared.close(row["innovation"], row["observed_y"] - row["prediction_before"], "online error", 1e-9)
        shared.close(row["final_fitted_y"], (final["a"] * x + final["b"]) * x + final["c"],
                     "final prediction", 1e-9)
        shared.close(row["final_residual"], row["observed_y"] - row["final_fitted_y"], "final residual", 1e-9)
        shared.close(row["reference_y"], (first["reference_a"] * x + first["reference_b"]) * x + first["reference_c"],
                     "synthetic reference", 1e-9)
        previous = row
    steps = sorted(set(r["step"] for r in curves))
    shared.require(steps[-1] == len(trace), "missing final curve snapshot")
    grid = None
    for step in steps:
        shared.require(step.is_integer() and 1 <= step <= len(trace), "invalid snapshot step")
        rows = [r for r in curves if r["step"] == step]
        xs = [r["x"] for r in rows]
        shared.require(len(xs) >= 2 and all(a < b for a, b in zip(xs, xs[1:])), "unordered snapshot grid")
        if grid is None:
            grid = xs
        shared.require(xs == grid, "different snapshot grids")
        snapshot = trace[int(step) - 1]
        for row in rows:
            shared.require(all(row[k] == snapshot[k] for k in ("a", "b", "c")), "snapshot coefficients")
            x = row["x"]
            shared.close(row["fitted_y"], (row["a"] * x + row["b"]) * x + row["c"], "snapshot evaluation", 1e-9)
            shared.close(row["reference_y"], (first["reference_a"] * x + first["reference_b"]) * x + first["reference_c"],
                         "snapshot reference", 1e-9)


def terminal_text(trace: list[dict[str, float]]) -> str:
    first = trace[0]
    lines = [f"samples = {len(trace)}, forgetting = {first['forgetting']:.3f}, prior precision = {first['prior_precision']:.6f}",
             f"estimator bytes: f32 = {int(first['storage_bytes_f32'])}, f64 = {int(first['storage_bytes_f64'])}",
             "step x observed_y prediction_before innovation a b c"]
    lines += [f"{int(r['step'])} " + " ".join(f"{r[key]:.3f}" for key in
              ("x", "observed_y", "prediction_before", "innovation", "a", "b", "c")) for r in trace]
    return "\n".join(lines) + "\n"


def render(trace: list[dict[str, float]], curves: list[dict[str, float]]) -> dict[str, str]:
    figures = {}
    for step in sorted(set(int(r["step"]) for r in curves)):
        prefix = trace[:step]
        grid = [r for r in curves if r["step"] == step]
        figures[f"rls-after-{step}.svg"] = shared.chart(
            f"Learn the curve: after {step} observations",
            "Only observations received so far are shown. The dashed synthetic reference is not given to RLS.",
            prefix, "x", [("fitted_y", "Recursive estimate", False),
                          ("reference_y", "Synthetic reference", False), ("observed_y", "Received observations", True)],
            "Input x", "Observed / predicted y",
            f"a = {prefix[-1]['a']:.3f}, b = {prefix[-1]['b']:.3f}, c = {prefix[-1]['c']:.3f}. Curve points are evaluated in Rust.",
            curve=grid)
    for coefficient in ("a", "b", "c"):
        # Alias an exported reference coefficient for the shared dashed-line renderer.
        rows = [dict(row, reference_y=row["reference_" + coefficient]) for row in trace]
        figures[f"rls-coefficient-{coefficient}.svg"] = shared.chart(
            f"Coefficient {coefficient}: the estimate after each observation",
            "Original x coordinates. Synthetic reference and recursive estimate are distinct quantities.",
            rows, "step", [(coefficient, f"Estimated {coefficient}", False),
                           ("reference_y", "Synthetic reference", False)],
            "Accepted observation count", f"Coefficient {coefficient}",
            f"Final {coefficient} = {trace[-1][coefficient]:.6f}. Regularization is defined in the normalized basis.", curve=rows)
    figures["rls-online-error.svg"] = shared.chart(
        "Prediction error before learning from the new sample",
        "Innovation = observed y minus prediction BEFORE the update; this is not a final-fit residual.",
        trace, "step", [("innovation", "Pre-update prediction error", True)],
        "Accepted observation count", "Prediction error in y",
        "The first prediction uses the prior mean. The error sequence need not decrease monotonically.", zero=True)
    residuals = [dict(row, residual=row["final_residual"]) for row in trace]
    figures["rls-final-residuals.svg"] = shared.chart(
        "Final model: inspect the remaining residuals",
        "All observations evaluated with the final coefficients, not the model available at arrival time.",
        residuals, "x", [("residual", "Observed minus final fit", True)],
        "Input x", "Final residual in y",
        "A regularized RLS fit need not equal unregularized batch QR exactly.", zero=True)
    return figures


def generate(root: Path, check: bool = False) -> None:
    output = root / "docs/generated/rls"
    if not check and output.exists():
        shutil.rmtree(output)
    (root / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rls-assets-", dir=root / "build") as temp:
        stage = Path(temp) / "assets"
        stage.mkdir()
        if not (root / "Cargo.lock").exists():
            shared.capture(root, ["cargo", "generate-lockfile"])
        command = ["cargo", "run", "--locked", "--quiet", "--no-default-features", "--example", "recursive_least_squares"]
        executions = []
        for filename, arguments in (("trace.csv", ["--", "--csv"]), ("curves.csv", ["--", "--curve-csv"]), ("output.txt", [])):
            text = shared.capture(root, command + arguments)
            (stage / filename).write_text(text, encoding="utf-8")
            executions.append({"command": command + arguments, "exit_code": 0, "stdout": filename})
        trace = shared.read_csv((stage / "trace.csv").read_text(), TRACE_COLUMNS)
        curves = shared.read_csv((stage / "curves.csv").read_text(), CURVE_COLUMNS)
        validate(trace, curves)
        shared.require((stage / "output.txt").read_text() == terminal_text(trace), "RLS CSV and text disagree")
        figures = render(trace, curves)
        for filename, contents in figures.items():
            (stage / filename).write_text(contents, encoding="utf-8")
        shutil.copyfile(root / "Cargo.lock", stage / "Cargo.lock")
        paths = shared.capture(root, ["git", "ls-files", "-z", "src", "examples", "Cargo.toml",
                                      "scripts/generate_tutorial_assets.py", "scripts/generate_rls_assets.py"]).split("\0")
        manifest = {
            "schema_version": 1, "scenario": "quadratic-square-root-rls",
            "source_revision": shared.capture(root, ["git", "rev-parse", "HEAD"]).strip(),
            "source_tree": shared.capture(root, ["git", "rev-parse", "HEAD^{tree}"]).strip(),
            "worktree_dirty": bool(shared.capture(root, ["git", "status", "--porcelain", "--untracked-files=normal"]).strip()),
            "source_sha256": {p: shared.digest(root / p) for p in paths if p},
            "rustc": shared.capture(root, ["rustc", "--version", "--verbose"]).strip(),
            "cargo": shared.capture(root, ["cargo", "--version"]).strip(),
            "python": platform.python_version(), "platform": {"system": platform.system(), "machine": platform.machine()},
            "environment": {k: os.environ.get(k) for k in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "RUSTUP_TOOLCHAIN")},
            "features": [], "profile": "dev", "executions": executions,
            "outputs_sha256": {p.name: shared.digest(p) for p in sorted(stage.iterdir())},
        }
        (stage / "provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if check:
            shared.require(output.is_dir(), "generate RLS assets before --check")
            old = {p.name: shared.digest(p) for p in output.iterdir() if p.is_file()}
            new = {p.name: shared.digest(p) for p in stage.iterdir()}
            shared.require(old == new, "RLS assets stale or not reproducible")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            stage.replace(output)
    print(f"RLS assets {'verified by re-execution' if check else 'generated from execution'}: {len(trace)} updates, {len(curves)} curve points, {len(figures)} SVGs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        generate(args.root.resolve(), args.check)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"RLS generation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
