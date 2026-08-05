#!/usr/bin/env python3
"""Build a self-contained comparison report from Criterion and Eigen results.

Criterion stores one ``estimates.json`` file per benchmark case.  This script
walks those files (including arbitrarily nested group paths), combines them
with Eigen CSV or native-text output, and writes an HTML report containing
inline SVG charts.  It intentionally uses only the Python standard library so
that the nightly workflow does not need a virtual environment.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


LIBRARY_TOKENS = (
    "stack",
    "stack-algebra",
    "stack-factor-reuse",
    "stack-solve",
    "stack-no-pivot",
    "faer",
    "eigen",
    "nalgebra",
)
def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _first(row: dict[str, str], names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = row.get(_normalise_key(name), "")
        if value.strip():
            return value.strip()
    return default


def _float(value: str) -> float | None:
    try:
        number = float(value.strip())
    except (AttributeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _scalar(value: str) -> str:
    match = re.search(r"(?:^|[_/-])(f(?:32|64))(?:$|[_/-])", value.lower())
    return match.group(1) if match else "unknown"


def _phase(value: str) -> str:
    lowered = value.lower()
    for phase in ("refactor", "analyze", "factor", "solve"):
        if phase in lowered:
            return phase
    return "operation"


def _canonical_eigen_group(operation: str, scalar: str) -> str:
    """Map native Eigen labels onto the Criterion comparison groups."""

    lowered = operation.lower()
    sparse_pattern = "star" if "star" in lowered else "band2" if "band2" in lowered else "tridiag"
    if "sparse" in lowered and "llt" in lowered:
        return f"sparse-llt/{scalar}/{sparse_pattern}"
    if "sparse" in lowered and "ldlt" in lowered:
        return f"sparse-ldlt/{scalar}/tridiag"

    dense_operations = (
        ("colpiv qr", "col-piv-qr"),
        ("self-adjoint eigen", "self-adjoint-eigen"),
        ("matmul", "matmul"),
        ("matvec", "matvec"),
        ("norm", "norm"),
        ("dot", "dot"),
        ("lu", "lu"),
        ("llt", "llt"),
        ("ldlt", "ldlt"),
        ("qr", "qr"),
        ("svd", "svd"),
    )
    if "upper triangular solve" in lowered:
        return f"comparison/triangular-upper-solve/{scalar}"
    if "lower triangular solve" in lowered:
        return f"comparison/triangular-solve/{scalar}"
    for needle, group in dense_operations:
        if needle in lowered:
            phase = _phase(lowered)
            phase_suffix = "" if phase == "operation" else f"-{phase}"
            return f"comparison/{group}{phase_suffix}/{scalar}"

    return operation


def _path_fields(path: Path, root: Path) -> tuple[str, str, str, str, str]:
    """Infer operation/library/shape/scalar/phase from a Criterion path.

    Criterion turns slashes in group names into directories, so the parser
    does not assume a fixed number of path components.  Known implementation
    tokens identify the library; otherwise the final component is treated as
    a variant and the library is reported as ``unknown``.
    """

    relative = path.relative_to(root)
    parts = list(relative.parts)
    if parts and parts[-1] == "estimates.json":
        parts.pop()
    if parts and parts[-1] in {"new", "base", "change"}:
        parts.pop()
    if not parts:
        return "unknown", "unknown", "", "unknown", "operation"

    token_index = next(
        (
            index
            for index, part in enumerate(parts)
            if any(part.lower() == token or part.lower().startswith(token + "-") for token in LIBRARY_TOKENS)
        ),
        None,
    )
    if token_index is not None:
        operation = "/".join(parts[:token_index]) or parts[0]
        library = parts[token_index]
        suffix = parts[token_index + 1 :]
    elif len(parts) >= 3:
        operation = "/".join(parts[:-2])
        library = parts[-2]
        suffix = parts[-1:]
    elif len(parts) == 2:
        operation, library, suffix = parts[0], parts[1], []
    else:
        operation, library, suffix = parts[0], "unknown", []

    shape = "/".join(suffix)
    combined = "/".join(parts)
    scalar = _scalar(combined)
    return operation, library, shape, scalar, _phase(combined)


def _canonical_criterion_operation(operation: str, scalar: str) -> str:
    """Restore Criterion's underscore-escaped group names."""

    if operation.startswith("comparison_"):
        body = operation.removeprefix("comparison_")
        suffix = f"_{scalar}"
        if body.endswith(suffix):
            body = body[: -len(suffix)]
        return f"comparison/{body}/{scalar}"
    for prefix in ("sparse-llt", "sparse-ldlt"):
        marker = f"{prefix}_{scalar}_"
        if operation.startswith(marker):
            return f"{prefix}/{scalar}/{operation.removeprefix(marker)}"
    return operation


def parse_criterion(root: Path) -> list[dict[str, object]]:
    if not root.is_dir():
        raise ValueError(f"Criterion directory does not exist: {root}")
    rows: list[dict[str, object]] = []
    # ``new`` is the current measurement.  Ignore baseline/change files to
    # avoid duplicate rows when a previous report is retained in the tree.
    files = sorted(path for path in root.rglob("estimates.json") if path.parent.name == "new")
    if not files:
        raise ValueError(f"No Criterion estimates found below {root} (expected */new/estimates.json)")
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            median = payload["median"]
            value = float(median["point_estimate"])
            interval = median["confidence_interval"]
            lower = float(interval["lower_bound"])
            upper = float(interval["upper_bound"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid Criterion estimate {path}: {error}") from error
        if not math.isfinite(value) or value <= 0:
            continue
        operation, library, shape, scalar, phase = _path_fields(path, root)
        operation = _canonical_criterion_operation(operation, scalar)
        canonical_operation = operation.split("/")[-2] if "/" in operation else operation
        shape = _canonical_shape(canonical_operation, shape)
        if library == "nalgebra-static" and canonical_operation in {
            "lu-factor",
            "lu-solve",
            "llt-factor",
            "llt-solve",
            "ldlt-factor",
            "ldlt-solve",
            "qr-factor",
            "qr-solve",
            "col-piv-qr-factor",
            "self-adjoint-eigen-factor",
            "svd-factor",
            "matvec",
        }:
            library = "nalgebra-dynamic"
        batch_size = 8 if operation.startswith("comparison/") else 1
        rows.append(
            {
                "source": "criterion",
                "library": library,
                "operation": operation,
                "shape": shape,
                "scalar": scalar,
                "phase": phase,
                "median_ns": value / batch_size,
                "lower_ns": lower / batch_size,
                "upper_ns": upper / batch_size,
                "path": str(path.relative_to(root)),
            }
        )
    if not rows:
        raise ValueError(f"Criterion estimates below {root} contained no positive median values")
    return rows


def _csv_rows(path: Path) -> list[dict[str, object]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError("CSV has no header")
            for row in reader:
                normalised = {_normalise_key(key): value or "" for key, value in row.items()}
                operation = _first(normalised, ("operation", "benchmark", "name", "case"), "unknown")
                value = _float(_first(normalised, ("median_ns", "ns_per_op", "nanoseconds", "time_ns", "median", "ns_op")))
                if value is None:
                    continue
                scalar = _first(normalised, ("scalar", "dtype", "precision"), _scalar(operation))
                canonical = _canonical_eigen_group(operation, scalar or "unknown")
                canonical_operation = canonical.split("/")[-2] if "/" in canonical else canonical
                shape = _first(normalised, ("shape", "size", "dimension", "dimensions"), "")
                shape = _canonical_shape(canonical_operation, shape or _shape_from_text(operation))
                return_row = {
                    "source": "eigen",
                    "library": _first(normalised, ("library", "backend", "implementation"), "eigen"),
                    "operation": canonical,
                    "shape": shape,
                    "scalar": scalar or "unknown",
                    "phase": _first(normalised, ("phase", "stage"), _phase(operation)),
                    "median_ns": value,
                    "lower_ns": _float(_first(normalised, ("lower_ns", "ci_lower", "lower"))) or value,
                    "upper_ns": _float(_first(normalised, ("upper_ns", "ci_upper", "upper"))) or value,
                    "path": str(path),
                }
                yield return_row
    except OSError as error:
        raise ValueError(f"Cannot read Eigen CSV {path}: {error}") from error


def parse_eigen_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"Eigen CSV does not exist: {path}")
    rows = list(_csv_rows(path))
    if not rows:
        raise ValueError(f"Eigen CSV contains no rows with a positive timing: {path}")
    return rows


def parse_eigen_text(paths: Iterable[Path]) -> list[dict[str, object]]:
    """Parse the fixed-width output of ``eigen/run_native_bench.sh``."""

    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Eigen benchmark output does not exist: {path}")
        scalar = "unknown"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(f"Cannot read Eigen output {path}: {error}") from error
        for line in lines:
            section = re.search(r"\b(f32|f64)\s+fixed-size operations\b", line, re.IGNORECASE)
            if section:
                scalar = section.group(1).lower()
                continue
            match = re.match(r"^\s*(.*?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*$", line)
            if not match or match.group(1).lower() in {"operation", ""}:
                continue
            operation = match.group(1).strip()
            canonical = _canonical_eigen_group(operation, scalar)
            value = float(match.group(3))
            canonical_operation = canonical.split("/")[-2] if "/" in canonical else canonical
            rows.append(
                {
                    "source": "eigen",
                    "library": "eigen",
                    "operation": canonical,
                    "shape": _canonical_shape(canonical_operation, _shape_from_text(operation)),
                    "scalar": scalar,
                    "phase": _phase(operation),
                    "median_ns": value,
                    "lower_ns": value,
                    "upper_ns": value,
                    "path": str(path),
                }
            )
    if not rows:
        joined = ", ".join(str(path) for path in paths)
        raise ValueError(f"Eigen output contains no timing rows: {joined}")
    return rows


def _shape_from_text(value: str) -> str:
    match = re.search(r"\b\d+x\d+(?:x\d+)?\b", value)
    if match:
        return match.group(0)
    match = re.search(r"\b\d+\b", value)
    return match.group(0) if match else ""


def _canonical_shape(group: str, shape: str) -> str:
    """Use the same shape key for square Eigen labels and Criterion ids."""

    # Matmul and matvec benchmark ids retain their matrix shape (for example
    # ``3x3``), while scalar reductions and square decompositions use the
    # dimension alone in the Rust benchmark ids (for example ``3``).
    if group in {"matmul", "matvec"}:
        if re.fullmatch(r"\d+", shape):
            return f"{shape}x{shape}"
    else:
        match = re.fullmatch(r"(\d+)x\1", shape)
        if match:
            return match.group(1)
    return shape


def _svg_chart(rows: list[dict[str, object]]) -> str:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["operation"]), str(row["shape"]), str(row["scalar"]), str(row["phase"]))
        grouped[key].append(row)
    groups = sorted(grouped.items())
    width, left, right, row_height = 1000, 315, 30, 32
    height = max(96, 55 + len(groups) * row_height)
    maximum = max(float(row["median_ns"]) for row in rows)
    maximum = maximum or 1.0
    colors = {"stack-algebra": "#2563eb", "eigen": "#dc2626", "faer": "#059669", "nalgebra": "#d97706"}
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">', '<style>text{font:12px system-ui,sans-serif}.axis{fill:#475569}.grid{stroke:#e2e8f0}.bar{opacity:.9}</style>']
    pieces.append(f'<rect width="{width}" height="{height}" fill="white"/><text x="{left}" y="22" font-weight="bold">Median latency (ns/op; lower is better)</text>')
    for index, ((operation, shape, scalar, phase), entries) in enumerate(groups):
        y = 42 + index * row_height
        label = html.escape(" / ".join(part for part in (operation, shape, scalar, phase) if part and part != "unknown"))
        pieces.append(f'<text class="axis" x="5" y="{y + 13}">{label}</text>')
        entries = sorted(entries, key=lambda item: float(item["median_ns"]))
        slot = max(5, (row_height - 5) // max(1, len(entries)))
        for item_index, entry in enumerate(entries):
            value = float(entry["median_ns"])
            bar_width = max(1.0, (width - left - right) * value / maximum)
            bar_y = y + item_index * slot
            library = html.escape(str(entry["library"]))
            library_name = str(entry["library"]).lower()
            color = next((color for name, color in colors.items() if library_name.startswith(name)), "#64748b")
            pieces.append(f'<rect class="bar" x="{left}" y="{bar_y}" width="{bar_width:.1f}" height="{max(4, slot - 2)}" fill="{color}"/><text x="{left + bar_width + 5:.1f}" y="{bar_y + max(9, slot - 3)}">{library} {value:.2f}</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("source", "library", "operation", "shape", "scalar", "phase", "median_ns", "lower_ns", "upper_ns", "path")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def build_html(rows: list[dict[str, object]], title: str, metadata: str) -> str:
    chart = _svg_chart(rows)
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["operation"]), str(row["shape"]), str(row["scalar"]), str(row["phase"]))].append(row)
    body_rows: list[str] = []
    for key, entries in sorted(groups.items()):
        baseline = next((float(entry["median_ns"]) for entry in entries if str(entry["library"]) == "stack-algebra"), None)
        for entry in sorted(entries, key=lambda item: (str(item["library"]), float(item["median_ns"]))):
            value = float(entry["median_ns"])
            relative = f"{baseline / value:.2f}x" if baseline else "—"
            body_rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in (*key, str(entry["library"]), f"{value:.2f}", relative)) + "</tr>")
    generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{html.escape(title)}</title><style>body{{font:15px system-ui,sans-serif;line-height:1.45;color:#172033;max-width:1200px;margin:2rem auto;padding:0 1rem}}h1{{margin-bottom:.2rem}}.meta{{color:#475569}}svg{{width:100%;height:auto;border:1px solid #e2e8f0;margin:1rem 0 2rem}}table{{border-collapse:collapse;width:100%;font-size:.86rem}}th,td{{border-bottom:1px solid #e2e8f0;padding:.35rem .5rem;text-align:left}}th{{background:#f8fafc;position:sticky;top:0}}code{{background:#f1f5f9;padding:.1rem .25rem}}</style></head><body><h1>{html.escape(title)}</h1><p class="meta">Generated {generated}. {html.escape(metadata)}</p><p>This report compares median steady-state latency. Static/fixed-capacity stack-algebra cases are compared with each library's corresponding static or dynamic storage mode; allocation and setup are excluded where the benchmark labels a reusable factor. Factorization, symbolic analysis, refactorization, and solve phases are reported separately and must not be interpreted as interchangeable operations.</p>{chart}<h2>Measurements</h2><table><thead><tr><th>Operation</th><th>Shape</th><th>Scalar</th><th>Phase</th><th>Library</th><th>Median ns/op</th><th>vs stack-algebra</th></tr></thead><tbody>{''.join(body_rows)}</tbody></table></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criterion-dir", type=Path, default=Path("target/criterion"))
    parser.add_argument("--eigen-csv", type=Path, action="append", default=[])
    parser.add_argument("--eigen-text", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("benchmark-report/index.html"))
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--title", default="Stack Algebra nightly benchmark comparison")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--require-eigen", action="store_true")
    args = parser.parse_args()
    try:
        rows = parse_criterion(args.criterion_dir)
        for path in args.eigen_csv:
            rows.extend(parse_eigen_csv(path))
        if args.eigen_text:
            rows.extend(parse_eigen_text(args.eigen_text))
        if args.require_eigen and not any(row["source"] == "eigen" for row in rows):
            raise ValueError("--require-eigen was set but no Eigen measurements were parsed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(build_html(rows, args.title, args.metadata), encoding="utf-8")
        # Keep a standalone image next to the HTML as well as embedding it in
        # the document, so artifact consumers can link or post-process it.
        args.output.with_name("latency.svg").write_text(_svg_chart(rows), encoding="utf-8")
        csv_output = args.csv_output or args.output.with_name("results.csv")
        _write_csv(rows, csv_output)
        print(f"wrote {args.output} ({len(rows)} measurements) and {csv_output}")
    except (OSError, ValueError) as error:
        print(f"benchmark report: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
