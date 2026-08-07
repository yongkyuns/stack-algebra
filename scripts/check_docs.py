#!/usr/bin/env python3
"""Check guide links and the documented public-type coverage manifest."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
DECLARATION_RE = re.compile(
    r"\bpub\s+(?:struct|enum|trait|type)\s+{symbol}\b"
)


def read_coverage(root: Path) -> list[tuple[str, Path, Path]]:
    manifest = root / "docs" / "api-coverage.tsv"
    entries: list[tuple[str, Path, Path]] = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 3 or any(not field for field in fields):
            raise ValueError(f"{manifest}:{line_number}: expected symbol|source|guide")
        symbol, source, guide = fields
        entries.append((symbol, root / source, root / guide))
    return entries


def check_coverage(root: Path) -> list[str]:
    errors: list[str] = []
    try:
        entries = read_coverage(root)
    except (OSError, ValueError) as error:
        return [str(error)]

    docs_text = {
        path: path.read_text()
        for path in (root / "docs").glob("*.md")
    }
    seen: set[str] = set()
    for symbol, source, guide in entries:
        if symbol in seen:
            errors.append(f"duplicate coverage entry: {symbol}")
        seen.add(symbol)
        if not source.is_file():
            errors.append(f"{symbol}: declaration source is missing: {source}")
            continue
        declaration = DECLARATION_RE.pattern.format(symbol=re.escape(symbol))
        if not re.search(declaration, source.read_text()):
            errors.append(f"{symbol}: public declaration not found in {source}")
        if not guide.is_file():
            errors.append(f"{symbol}: guide page is missing: {guide}")
            continue
        guide_text = docs_text.get(guide, guide.read_text())
        if not re.search(rf"\b{re.escape(symbol)}\b", guide_text):
            errors.append(f"{symbol}: guide page does not mention the type: {guide}")
    return errors


def check_links(root: Path, site: Path) -> list[str]:
    errors: list[str] = []
    docs_root = root / "docs"
    for document in sorted(docs_root.glob("*.md")):
        text = document.read_text()
        for target in LINK_RE.findall(text):
            target = target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path = target.split("#", 1)[0].split("?", 1)[0]
            if not path:
                continue
            if path.startswith("/"):
                candidate = site / path.lstrip("/")
            elif path.startswith("api/"):
                candidate = site / path
            else:
                candidate = document.parent / path
            if not candidate.is_file():
                errors.append(f"{document}: broken link target: {target}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--site", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    site = (args.site or root / "build" / "site").resolve()

    errors = check_coverage(root)
    if not site.is_dir():
        errors.append(f"generated site is missing: {site}")
    else:
        errors.extend(check_links(root, site))
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Documentation coverage and links are valid ({site})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
