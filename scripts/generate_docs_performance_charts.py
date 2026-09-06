#!/usr/bin/env python3
"""Generate documentation SVGs from accepted production benchmark evidence.

The input intentionally stores only same-runner before/after improvement
percentages for merged production changes. It does not mix experimental
candidates into user-facing charts or reinterpret cross-library snapshots.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "assets" / "accepted-performance.csv"
DENSE_OUT = ROOT / "docs" / "assets" / "accepted-performance-dense-solves.svg"
DOT_OUT = ROOT / "docs" / "assets" / "accepted-performance-small-dot.svg"


def load_rows() -> list[dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "date",
        "area",
        "benchmark",
        "scalar",
        "lower_improvement_pct",
        "upper_improvement_pct",
        "source_pr",
        "source_commit",
        "notes",
    }
    if not rows or set(rows[0]) != required:
        raise SystemExit(f"{DATA}: unexpected columns")
    for row in rows:
        low = float(row["lower_improvement_pct"])
        high = float(row["upper_improvement_pct"])
        if low < 0 or high < low:
            raise SystemExit(f"{DATA}: invalid improvement interval: {row}")
    return rows


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def svg_chart(
    title: str,
    subtitle: str,
    rows: list[dict[str, str]],
    maximum: float,
) -> str:
    width = 1120
    left = 385
    right = 105
    top = 92
    row_height = 34
    bottom = 48
    height = top + row_height * len(rows) + bottom
    plot_width = width - left - right

    def x(value: float) -> float:
        return left + plot_width * value / maximum

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{esc(title)}</title>",
        f"<desc id=\"desc\">{esc(subtitle)}</desc>",
        "<style>"
        "text{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}"
        ".title{font-size:20px;font-weight:700;fill:#111827}"
        ".subtitle{font-size:12px;fill:#475569}"
        ".label{font-size:12px;fill:#1f2937}"
        ".tick{font-size:11px;fill:#64748b}"
        ".grid{stroke:#e2e8f0;stroke-width:1}"
        ".track{stroke:#cbd5e1;stroke-width:2}"
        ".range{stroke:#2563eb;stroke-width:8;stroke-linecap:round}"
        ".point{fill:#2563eb}"
        ".zero{fill:#64748b}"
        ".value{font-size:11px;font-weight:600;fill:#334155}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="20" y="30">{esc(title)}</text>',
        f'<text class="subtitle" x="20" y="51">{esc(subtitle)}</text>',
    ]

    for tick in range(0, int(maximum) + 1, 10):
        xpos = x(tick)
        pieces.append(f'<line class="grid" x1="{xpos:.1f}" y1="{top - 16}" x2="{xpos:.1f}" y2="{height - bottom + 4}"/>')
        pieces.append(f'<text class="tick" x="{xpos:.1f}" y="{height - 16}" text-anchor="middle">{tick}%</text>')

    for index, row in enumerate(rows):
        y = top + index * row_height
        low = float(row["lower_improvement_pct"])
        high = float(row["upper_improvement_pct"])
        midpoint = (low + high) / 2.0
        label = f'{row["benchmark"]} · {row["scalar"]}'
        pieces.append(f'<text class="label" x="20" y="{y + 5}">{esc(label)}</text>')
        pieces.append(f'<line class="track" x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}"/>')
        if high == 0:
            pieces.append(f'<circle class="zero" cx="{left}" cy="{y}" r="4"/>')
            value_text = "unchanged"
        elif abs(high - low) < 1e-9:
            xpos = x(midpoint)
            pieces.append(f'<line class="range" x1="{left}" y1="{y}" x2="{xpos:.1f}" y2="{y}"/>')
            pieces.append(f'<circle class="point" cx="{xpos:.1f}" cy="{y}" r="5"/>')
            value_text = f"≈{midpoint:g}% faster"
        else:
            xlow, xhigh, xmid = x(low), x(high), x(midpoint)
            pieces.append(f'<line class="range" x1="{xlow:.1f}" y1="{y}" x2="{xhigh:.1f}" y2="{y}"/>')
            pieces.append(f'<circle class="point" cx="{xmid:.1f}" cy="{y}" r="5"/>')
            value_text = f"{low:g}–{high:g}% faster"
        pieces.append(f'<text class="value" x="{width - right + 12}" y="{y + 4}">{esc(value_text)}</text>')

    pieces.append("</svg>\n")
    return "".join(pieces)


def main() -> None:
    rows = load_rows()

    dense = [row for row in rows if row["area"] in {"ldlt-multi-rhs", "triangular-d32"}]
    dense.sort(key=lambda row: (
        0 if row["area"] == "ldlt-multi-rhs" else 1,
        row["benchmark"],
        row["scalar"],
    ))
    dot = [row for row in rows if row["area"] == "small-f32-dot"]
    dot.sort(key=lambda row: int(row["benchmark"].split("=")[1]))

    DENSE_OUT.write_text(
        svg_chart(
            "Accepted dense-solve improvements",
            "Same-runner before/after validation for production code merged on 2026-09-06; higher improvement is better.",
            dense,
            maximum=80,
        ),
        encoding="utf-8",
    )
    DOT_OUT.write_text(
        svg_chart(
            "Accepted small-f32 dot improvements",
            "Repeatable hosted AVX2/FMA before/after ranges for production code merged on 2026-09-06; higher improvement is better.",
            dot,
            maximum=50,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
