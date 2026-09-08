#!/usr/bin/env python3
"""Measure sparse permutation and permutation-plus-factorization around #51."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile
import tempfile

BEFORE = "f4fd7fdd8944c95a2951ef654c465271d1e4297c"
REPAIRED = "65c9d37d598984034abc18e31f0ffabb12798854"
KEYS = "n,scalar,capacity,layout,ordering,operation".split(",")
FIELDS = ["sample"] + KEYS + "input_capacity,factor_capacity,input_nnz,ordered_nnz,factor_nnz,iterations,elapsed_ns".split(",")
EXPECTED = set(itertools.product(
    ("6", "15", "64", "128"), ("f32", "f64"), ("compact", "roomy"),
    ("lower", "full"), ("identity", "reverse", "rotate"),
    ("apply_into", "apply", "permute_cholesky", "permute_ldlt"),
))


def capture(args, cwd, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def factor_nnz(n, kind):
    order = list(range(n))
    if kind == "reverse":
        order.reverse()
    elif kind == "rotate":
        order = order[n // 2:] + order[:n // 2]
    inverse = {original: ordered for ordered, original in enumerate(order)}
    adjacency = [set() for _ in range(n)]
    for first in range(n):
        for second in range(first + 1, min(n, first + 3)):
            a, b = inverse[first], inverse[second]
            adjacency[a].add(b)
            adjacency[b].add(a)
    count = n
    for column in range(n):
        later = {row for row in adjacency[column] if row > column}
        count += len(later)
        for row in later:
            adjacency[row].update(later - {row})
    return count


COUNTS = {(n, kind): factor_nnz(int(n), kind)
          for n in ("6", "15", "64", "128") for kind in ("identity", "reverse", "rotate")}


def read_samples(text, sample, timed=True):
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != FIELDS:
        raise ValueError("unexpected measurement schema")
    rows = list(reader)
    keys = [tuple(row[k] for k in KEYS) for row in rows]
    if len(keys) != len(EXPECTED) or set(keys) != EXPECTED:
        raise ValueError("missing, duplicate or unexpected fixture")
    for row in rows:
        n = int(row["n"])
        expected = {
            "sample": sample, "input_capacity": n * (5 if row["capacity"] == "compact" else 20),
            "factor_capacity": n * 8, "input_nnz": 5 * n - 6 if row["layout"] == "full" else 3 * n - 3,
            "ordered_nnz": 3 * n - 3, "factor_nnz": COUNTS[row["n"], row["ordering"]],
        }
        if any(int(row[key]) != value for key, value in expected.items()):
            raise ValueError("incorrect sample or fixture metadata")
        if int(row["iterations"]) <= 0 or (timed and int(row["elapsed_ns"]) <= 0):
            raise ValueError("empty measurement")
        if not timed and (int(row["iterations"]) != 1 or int(row["elapsed_ns"]) != 0):
            raise ValueError("unexpected correctness-only result")
    return rows


def summarize(rows, rounds):
    cases = {}
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        versions = cases.setdefault(key, {})
        samples = versions.setdefault(row["revision"], {})
        sample = int(row["sample"])
        if sample in samples:
            raise ValueError("duplicate sample")
        samples[sample] = int(row["elapsed_ns"]) / int(row["iterations"])
    if set(cases) != EXPECTED:
        raise ValueError("incomplete fixture coverage")
    lines = ["# Sparse permutation repair measurements", "",
             "Nanoseconds per call, warm-cache synthetic banded fixtures. Before is #50; candidate includes #51.",
             "All source layouts and destination patterns are valid on both revisions.",
             "Ratios are paired-round medians with observed min/max, not confidence intervals or timing gates.",
             "Cached map construction, symbolic analysis, allocation and verification are outside timing.",
             "Pipeline timings include permutation plus numeric refactorization, but not the solve.", "",
             "| Case | Before ns | Candidate ns | Candidate / before |",
             "|---|---:|---:|---:|"]
    for key, versions in sorted(cases.items()):
        if set(versions) != {"before", "candidate"}:
            raise ValueError("missing revision")
        if any(set(samples) != set(range(rounds)) for samples in versions.values()):
            raise ValueError("missing round")
        a, b = versions["before"], versions["candidate"]
        ratios = [b[i] / a[i] for i in range(rounds)]
        lines.append(f"| {'/'.join(key)} | {statistics.median(a.values()):.1f} | "
                     f"{statistics.median(b.values()):.1f} | {statistics.median(ratios):.3f} "
                     f"[{min(ratios):.3f}, {max(ratios):.3f}] |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="HEAD")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 6 or args.rounds > 30 or args.rounds % 2:
        parser.error("rounds must be even, between 6 and 30")
    tool = Path(__file__).resolve().parent
    repo = tool.parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate = capture(["git", "rev-parse", "--verify", args.candidate + "^{commit}"], repo)
    subprocess.run(["git", "merge-base", "--is-ancestor", REPAIRED, candidate], cwd=repo, check=True)
    env = os.environ.copy()
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    env.update(CARGO_INCREMENTAL="0", RUSTFLAGS="-C target-cpu=native",
               CARGO_PROFILE_RELEASE_OPT_LEVEL="3", CARGO_PROFILE_RELEASE_CODEGEN_UNITS="1",
               CARGO_PROFILE_RELEASE_LTO="false")
    provenance = {
        "revisions": {"before": BEFORE, "candidate": candidate},
        "harness_checkout": capture(["git", "rev-parse", "HEAD"], repo),
        "rustc": capture(["rustc", "-Vv"], repo), "cargo": capture(["cargo", "-V"], repo),
        "rustflags": env["RUSTFLAGS"], "rounds": args.rounds,
        "release": {"opt_level": 3, "codegen_units": 1, "lto": False},
        "files": {}, "source_sha256": {}, "binaries": {}, "order": [],
    }
    for path, name in [(tool / "src/main.rs", "harness.rs"), (tool / "Cargo.toml", "manifest.toml"),
                       (Path(__file__), "driver.py"), (tool / "test_compare.py", "test_compare.py")]:
        data = path.read_bytes()
        (out / name).write_bytes(data)
        provenance["files"][name] = digest(data)
    (out / "cpu.txt").write_text(capture(["lscpu"], repo) + "\n")
    with tempfile.TemporaryDirectory(prefix="permutation-bench-") as tmp:
        binaries = {}
        lock = None
        for label, sha in provenance["revisions"].items():
            root = Path(tmp) / label
            root.mkdir()
            archive = subprocess.check_output(["git", "archive", sha], cwd=repo)
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(root, filter="data")
            provenance["source_sha256"][label] = {
                str(p.relative_to(root)): digest(p.read_bytes()) for p in sorted((root / "src").rglob("*.rs"))
            }
            dest = root / "tools/sparse-permutation-bench"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(tool, dest, ignore=shutil.ignore_patterns("target", "__pycache__", "Cargo.lock"))
            manifest = dest / "Cargo.toml"
            if lock is None:
                subprocess.run(["cargo", "generate-lockfile", "--manifest-path", str(manifest)], cwd=root, env=env, check=True)
                lock = (dest / "Cargo.lock").read_bytes()
                (out / "Cargo.lock").write_bytes(lock)
                provenance["lock_sha256"] = digest(lock)
            else:
                (dest / "Cargo.lock").write_bytes(lock)
            build_env = dict(env, CARGO_TARGET_DIR=str(root / "bench-target"))
            with (out / f"build-{label}.log").open("w") as log:
                subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", str(manifest)],
                               cwd=root, env=build_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            binary = root / "bench-target/release/sparse-permutation-bench"
            binaries[label] = binary
            provenance["binaries"][label] = digest(binary.read_bytes())
            checked = capture([str(binary), "--check"], repo, env)
            read_samples(checked, 0, timed=False)
            (out / f"check-{label}.csv").write_text(checked + "\n")
            print(f"Built and checked {label}: {sha}", flush=True)
        if hasattr(os, "sched_getaffinity"):
            allowed = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {allowed[0]})
            provenance["available_cpus"] = allowed
            provenance["measurement_cpu"] = allowed[0]
        rows = []
        with (out / "samples.csv").open("w", newline="") as raw:
            writer = csv.DictWriter(raw, fieldnames=["revision"] + FIELDS)
            writer.writeheader()
            for sample in range(args.rounds):
                order = ["before", "candidate"] if sample % 2 == 0 else ["candidate", "before"]
                provenance["order"].append(order)
                for label in order:
                    text = capture([str(binaries[label]), str(sample)], repo, env)
                    for row in read_samples(text, sample):
                        row["revision"] = label
                        writer.writerow(row)
                        rows.append(row)
                raw.flush()
                print(f"Completed paired round {sample + 1}/{args.rounds}", flush=True)
        (out / "summary.md").write_text(summarize(rows, args.rounds))
        provenance["sample_rows"] = len(rows)
        provenance["samples_sha256"] = digest((out / "samples.csv").read_bytes())
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
