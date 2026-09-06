#!/usr/bin/env python3
"""Generate exhaustive benchmark-reference pages from a checked-in CSV snapshot.

The source CSV is produced by scripts/generate_benchmark_report.py. This script
keeps mdBook presentation deterministic and uses only the Python standard
library so docs CI needs no extra plotting dependency.
"""
from __future__ import annotations

import csv
import gzip
import html
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "benchmark-snapshot-2026-09-06.csv.gz"
PROVENANCE = ROOT / "docs" / "data" / "benchmark-snapshot-2026-09-06-provenance.txt"
GENERATED = ROOT / "docs" / "generated"
ASSETS = ROOT / "docs" / "assets" / "benchmark-reference"

PALETTE = ["#2563eb", "#059669", "#dc2626", "#d97706", "#7c3aed", "#0891b2", "#be123c", "#4f46e5"]


def read_rows() -> list[dict[str, str]]:
    with gzip.open(DATA, mode="rt", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"library", "operation", "shape", "scalar", "phase", "median_ns"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"invalid benchmark snapshot: {DATA}")
    return rows


def display_library(name: str) -> str:
    aliases = {
        "faer-dynamic": "faer",
        "nalgebra-static": "nalgebra (static)",
        "nalgebra-dynamic": "nalgebra (dynamic)",
        "nalgebra-lu-fallback": "nalgebra (LU fallback)",
        "stack-algebra-workspace": "stack-algebra (workspace)",
        "eigen": "Eigen",
    }
    return aliases.get(name, name)


def operation_title(operation: str) -> str:
    if operation.startswith("comparison/"):
        parts = operation.split("/")
        return parts[1].replace("-", " ").title() + f" — {parts[2]}"
    return operation.replace("_", " / ").replace("-", " ").replace("/", " / ")


def numeric_shape_key(shape: str) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", shape or "")]
    padded = (nums + [10**9, 10**9, 10**9])[:3]
    return (*padded, shape or "")


def value(row: dict[str, str], key: str = "median_ns") -> float:
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return math.nan


def matched_labels(rows: list[dict[str, str]]) -> list[str]:
    by_shape: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        shape = row["shape"]
        if shape and shape.lower() != "nan":
            by_shape[shape].add(row["library"])
    return sorted(
        (shape for shape, libs in by_shape.items() if len(libs) >= 2),
        key=numeric_shape_key,
    )


def chartable(rows: list[dict[str, str]]) -> bool:
    labels = matched_labels(rows)
    libs = {r["library"] for r in rows if r["shape"] in labels}
    return len(labels) >= 2 and len(libs) >= 2


def svg_line_chart(operation: str, rows: list[dict[str, str]], out: Path) -> None:
    labels = matched_labels(rows)
    series: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["shape"] in labels and math.isfinite(value(row)) and value(row) > 0:
            series[display_library(row["library"])][row["shape"]] = value(row)
    series = {k: v for k, v in series.items() if v}
    if len(series) < 2:
        return

    width, height = 920, 430
    left, right, top, bottom = 90, 28, 52, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [v for mapping in series.values() for v in mapping.values()]
    lo, hi = min(values), max(values)
    if lo == hi:
        lo *= 0.8
        hi *= 1.2
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    pad = max(0.08, (log_hi - log_lo) * 0.08)
    log_lo -= pad
    log_hi += pad

    def x_at(i: int) -> float:
        return left + (plot_w / max(1, len(labels) - 1)) * i

    def y_at(v: float) -> float:
        return top + plot_h * (1 - (math.log10(v) - log_lo) / (log_hi - log_lo))

    esc_title = html.escape(operation_title(operation))
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc_title} benchmark chart">',
        '<style>text{font:12px system-ui,sans-serif;fill:#334155}.title{font-size:16px;font-weight:700;fill:#0f172a}.grid{stroke:#cbd5e1;stroke-width:1}.axis{stroke:#64748b;stroke-width:1}.legend{font-size:11px}</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{left}" y="26">{esc_title}</text>',
        f'<text x="{left}" y="43">Median latency, ns/op — logarithmic y-axis; lower is better</text>',
    ]

    for i in range(5):
        exponent = log_lo + (log_hi - log_lo) * i / 4
        tick = 10**exponent
        y = y_at(tick)
        pieces.append(
            f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        if tick >= 1000:
            label = f"{tick / 1000:.2g}k"
        elif tick >= 10:
            label = f"{tick:.0f}"
        else:
            label = f"{tick:.2g}"
        pieces.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{label}</text>'
        )

    pieces.append(
        f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>'
    )
    pieces.append(
        f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>'
    )
    for i, label in enumerate(labels):
        x = x_at(i)
        pieces.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle">{html.escape(label)}</text>'
        )
    pieces.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle">Matrix dimension / shape</text>'
    )

    legend_x, legend_y = left, height - 48
    for sidx, (name, points) in enumerate(sorted(series.items())):
        color = PALETTE[sidx % len(PALETTE)]
        coords = []
        for i, label in enumerate(labels):
            if label in points:
                coords.append((x_at(i), y_at(points[label])))
        if len(coords) >= 2:
            path = " ".join(
                ("M" if idx == 0 else "L") + f" {x:.1f} {y:.1f}"
                for idx, (x, y) in enumerate(coords)
            )
            pieces.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4"/>'
            )
        for x, y in coords:
            pieces.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>'
            )
        lx = legend_x + (sidx % 4) * 200
        ly = legend_y + (sidx // 4) * 17
        pieces.append(
            f'<line x1="{lx}" x2="{lx + 18}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="2.4"/>'
        )
        pieces.append(
            f'<text class="legend" x="{lx + 24}" y="{ly + 4}">{html.escape(name)}</text>'
        )

    pieces.append("</svg>")
    out.write_text("\n".join(pieces), encoding="utf-8")


def format_ns(v: float) -> str:
    if not math.isfinite(v):
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.3g} ms"
    if v >= 1_000:
        return f"{v / 1_000:.4g} µs"
    return f"{v:.4g} ns"


def operation_table(rows: list[dict[str, str]]) -> str:
    ordered = sorted(
        rows,
        key=lambda r: (
            numeric_shape_key(r["shape"]),
            r["phase"],
            display_library(r["library"]),
        ),
    )
    lines = [
        "<table>",
        "<thead><tr><th>Shape</th><th>Phase</th><th>Implementation</th><th>Median</th></tr></thead>",
        "<tbody>",
    ]
    for row in ordered:
        shape = (
            row["shape"]
            if row["shape"] and row["shape"].lower() != "nan"
            else "—"
        )
        med = value(row)
        lines.append(
            "<tr>"
            f'<td><code>{html.escape(shape)}</code></td>'
            f'<td>{html.escape(row["phase"])}</td>'
            f'<td>{html.escape(display_library(row["library"]))}</td>'
            f"<td>{format_ns(med)}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def classify(operation: str) -> str:
    if operation.startswith("comparison/"):
        family = operation.split("/")[1]
        if family in {"matmul", "matvec", "dot", "norm"}:
            return "dense"
        return "solvers"
    if operation.startswith("sparse-"):
        return "sparse"
    return "structured"


def generate_page(
    name: str,
    title: str,
    intro: str,
    groups: list[tuple[str, list[dict[str, str]]]],
) -> None:
    lines = [
        f"# {title}",
        "",
        intro,
        "",
        "All timings are medians from the same September 6 snapshot. Lower is better. Charts use a logarithmic latency axis and include only shapes measured by at least two implementations; the tables retain every reported row, including unmatched controls.",
        "",
    ]
    for operation, rows in groups:
        anchor_name = re.sub(r"[^a-z0-9]+", "-", operation.lower()).strip("-")
        lines += [f"## {operation_title(operation)}", ""]
        if chartable(rows):
            asset = f"{anchor_name}.svg"
            svg_line_chart(operation, rows, ASSETS / asset)
            lines += [
                f"![{operation_title(operation)} benchmark](../assets/benchmark-reference/{asset})",
                "",
            ]
        lines += [operation_table(rows), ""]
    (GENERATED / name).write_text("\n".join(lines), encoding="utf-8")


def full_results(rows: list[dict[str, str]]) -> None:
    lines = [
        "# All benchmark results",
        "",
        "This table contains every row in the checked-in September 6 benchmark snapshot. It is the audit/reference view behind the categorized charts. Values are median nanoseconds per operation as emitted by the benchmark report pipeline.",
        "",
        "<table>",
        "<thead><tr><th>Operation</th><th>Shape</th><th>Scalar</th><th>Phase</th><th>Implementation</th><th>Median</th></tr></thead>",
        "<tbody>",
    ]
    for row in sorted(
        rows,
        key=lambda r: (
            r["operation"],
            numeric_shape_key(r["shape"]),
            r["phase"],
            r["library"],
        ),
    ):
        shape = (
            row["shape"]
            if row["shape"] and row["shape"].lower() != "nan"
            else "—"
        )
        med = value(row)
        lines.append(
            "<tr>"
            f'<td><code>{html.escape(row["operation"])}</code></td>'
            f'<td><code>{html.escape(shape)}</code></td>'
            f'<td>{html.escape(row["scalar"])}</td>'
            f'<td>{html.escape(row["phase"])}</td>'
            f'<td>{html.escape(display_library(row["library"]))}</td>'
            f"<td>{format_ns(med)}</td>"
            "</tr>"
        )
    lines += ["</tbody>", "</table>", ""]
    (GENERATED / "benchmark-all.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = read_rows()
    GENERATED.mkdir(parents=True, exist_ok=True)
    with gzip.open(DATA, mode="rt", encoding="utf-8") as source:
        (GENERATED / "benchmark-snapshot.csv").write_text(
            source.read(), encoding="utf-8"
        )
    ASSETS.mkdir(parents=True, exist_ok=True)

    for path in ASSETS.glob("*.svg"):
        path.unlink()
    for path in GENERATED.glob("benchmark-*.md"):
        path.unlink()

    by_op: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_op[row["operation"]].append(row)

    buckets: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for operation in sorted(by_op):
        buckets[classify(operation)].append((operation, by_op[operation]))

    generate_page(
        "benchmark-dense.md",
        "Dense operations",
        "Cross-library fixed-size matrix multiply, matrix-vector multiply, dot product, and norm measurements across the dimensions present in the scheduled comparison suite.",
        buckets["dense"],
    )
    generate_page(
        "benchmark-solvers.md",
        "Decompositions and solves",
        "Cross-library factorization and reusable-solve measurements for LU, Cholesky/LLT, LDLT, QR/CPQR, triangular solves, self-adjoint eigen decomposition, and SVD.",
        buckets["solvers"],
    )
    generate_page(
        "benchmark-sparse.md",
        "Sparse operations",
        "Sparse matrix-vector, symbolic analysis, factorization/refactorization, assembly, and solve measurements. Patterns and phases remain separate so unlike work is not combined into one number.",
        buckets["sparse"],
    )
    generate_page(
        "benchmark-structured.md",
        "Structured and specialized workloads",
        "Block-sparse factorizations, dense LDLT stress cases, fused operations, mapped views, and workload-decision measurements from the scheduled suite.",
        buckets["structured"],
    )
    full_results(rows)

    print(
        f"Generated benchmark reference: {len(rows)} rows across {len(by_op)} operation groups"
    )


if __name__ == "__main__":
    main()
