#!/usr/bin/env python3
"""Compare the two sparse reuse repairs with identical tooling on one host."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tarfile
import tempfile

BEFORE = "9903a4cede6a8dcb4808c67dafdd4a03587bf39f"
AFTER_LDLT = "63d429f89691a2189088b0edcf38819aa9e9e3e5"
SAFE_BASE = "a5b4dbde9d3baec272b7ba1fef118d62b0c0b007"
FIELDS = "sample,shape,n,scalar,solver,entry,layout,iterations,elapsed_ns,input_nnz,factor_nnz".split(",")


def capture(args, cwd, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_samples(text, expected):
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != FIELDS:
        raise ValueError("unexpected benchmark CSV schema")
    rows = list(reader)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} cases, received {len(rows)}")
    for row in rows:
        if int(row["iterations"]) <= 0 or int(row["elapsed_ns"]) <= 0:
            raise ValueError("empty measurement")
    return rows


def summarize(rows, rounds):
    cases = {}
    for row in rows:
        key = tuple(row[k] for k in ("shape", "n", "scalar", "solver", "entry", "layout"))
        samples = cases.setdefault(key, {}).setdefault(row["revision"], {})
        sample = int(row["sample"])
        if sample in samples:
            raise ValueError("duplicate sample")
        samples[sample] = int(row["elapsed_ns"]) / int(row["iterations"])
    expected = set(range(rounds))
    for revisions in cases.values():
        for samples in revisions.values():
            if samples.keys() != expected:
                raise ValueError("missing samples")
    lines = [
        "# Sparse reuse measurements", "",
        "Warm-cache synthetic fixtures; nanoseconds per recomputation. Setup, symbolic analysis,",
        "allocation and residual checks are outside timing. Ratios are medians of paired rounds;",
        "the bracket is the observed min/max, not a confidence interval or a regression gate.", "",
        "## Matching source layouts", "",
        "| Case | Before ns | After LDLT ns | Candidate ns | Candidate / before |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, versions in sorted(cases.items()):
        if key[-1] != "matching":
            continue
        if set(versions) != {"before", "after_ldlt", "candidate"}:
            raise ValueError("incomplete matching-layout comparison")
        ns = [statistics.median(versions[v].values()) for v in ("before", "after_ldlt", "candidate")]
        ratios = [versions["candidate"][i] / versions["before"][i] for i in range(rounds)]
        lines.append(f"| {'/'.join(key[:-1])} | {ns[0]:.1f} | {ns[1]:.1f} | {ns[2]:.1f} | "
                     f"{statistics.median(ratios):.3f} [{min(ratios):.3f}, {max(ratios):.3f}] |")
    lines += ["", "## Candidate-only changed source layout", "",
              "Repacked uses a lower-input schedule with full symmetric numeric storage.",
              "Rebuilt uses a schedule analyzed from that same full numeric storage.",
              "Both solve the same system. Historical unsafe code is never run on changed layouts.", "",
              "| Case | Repacked ns | Rebuilt ns | Repacked / rebuilt |",
              "|---|---:|---:|---:|"]
    for key, versions in sorted(cases.items()):
        if key[-1] != "repacked":
            continue
        first = versions["candidate"]
        second = cases[key[:-1] + ("rebuilt",)]["candidate"]
        ratios = [first[i] / second[i] for i in range(rounds)]
        lines.append(f"| {'/'.join(key[:-1])} | {statistics.median(first.values()):.1f} | "
                     f"{statistics.median(second.values()):.1f} | {statistics.median(ratios):.3f} "
                     f"[{min(ratios):.3f}, {max(ratios):.3f}] |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="HEAD")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.rounds <= 30:
        parser.error("rounds must be between 3 and 30")
    tool = Path(__file__).resolve().parent
    repo = tool.parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidate = capture(["git", "rev-parse", "--verify", args.candidate + "^{commit}"], repo)
    # Never run repacked fixtures against an unfixed or arbitrary historical revision.
    subprocess.run(["git", "merge-base", "--is-ancestor", SAFE_BASE, candidate], cwd=repo, check=True)
    env = os.environ.copy()
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    env.update(CARGO_INCREMENTAL="0", RUSTFLAGS="-C target-cpu=native")
    revisions = {"before": BEFORE, "after_ldlt": AFTER_LDLT, "candidate": candidate}
    provenance = {
        "revisions": revisions, "harness_checkout": capture(["git", "rev-parse", "HEAD"], repo),
        "harness_sha256": digest((tool / "src/main.rs").read_bytes()),
        "manifest_sha256": digest((tool / "Cargo.toml").read_bytes()),
        "rustc": capture(["rustc", "-Vv"], repo), "cargo": capture(["cargo", "-V"], repo),
        "platform": platform.platform(), "rounds": args.rounds, "rustflags": env["RUSTFLAGS"],
        "release": {"opt_level": 3, "codegen_units": 1, "lto": False},
        "binaries": {}, "order": [],
    }
    (output / "cpu.txt").write_text(capture(["lscpu"], repo) + "\n")
    (output / "harness.rs").write_bytes((tool / "src/main.rs").read_bytes())
    with tempfile.TemporaryDirectory(prefix="sparse-reuse-") as temporary:
        binaries = {}
        lock = None
        for label, sha in revisions.items():
            root = Path(temporary) / label
            root.mkdir()
            archive = subprocess.check_output(["git", "archive", sha], cwd=repo)
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(root, filter="data")
            destination = root / "tools/sparse-reuse-bench"
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(tool, destination, ignore=shutil.ignore_patterns("target", "__pycache__", "Cargo.lock"))
            manifest = destination / "Cargo.toml"
            if lock is None:
                subprocess.run(["cargo", "generate-lockfile", "--manifest-path", str(manifest)],
                               cwd=root, env=env, check=True)
                lock = (destination / "Cargo.lock").read_bytes()
                (output / "Cargo.lock").write_bytes(lock)
                provenance["lock_sha256"] = digest(lock)
            else:
                (destination / "Cargo.lock").write_bytes(lock)
            # A separate target directory per revision prevents stale build reuse.
            build_env = dict(env, CARGO_TARGET_DIR=str(root / "bench-target"))
            with (output / f"build-{label}.log").open("w") as log:
                subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", str(manifest)],
                               cwd=root, env=build_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            binary = root / "bench-target/release/sparse-reuse-bench"
            binaries[label] = binary
            provenance["binaries"][label] = digest(binary.read_bytes())
            print(f"Built {label} {sha}", flush=True)
        if hasattr(os, "sched_getaffinity"):
            allowed = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {allowed[0]})
            provenance["available_cpus"] = allowed
            provenance["measurement_cpu"] = allowed[0]
        permutations = list(itertools.permutations(revisions))
        rows = []
        with (output / "samples.csv").open("w", newline="") as raw:
            writer = csv.DictWriter(raw, fieldnames=["revision"] + FIELDS)
            writer.writeheader()
            for sample in range(args.rounds):
                order = permutations[sample % len(permutations)]
                provenance["order"].append(list(order))
                for label in order:
                    text = capture([str(binaries[label]), "matching", str(sample)], repo, env)
                    batch = read_samples(text, 48)
                    for row in batch:
                        row["revision"] = label
                        writer.writerow(row)
                    rows.extend(batch)
                text = capture([str(binaries["candidate"]), "layouts", str(sample)], repo, env)
                batch = read_samples(text, 96)
                for row in batch:
                    row["revision"] = "candidate"
                    writer.writerow(row)
                rows.extend(batch)
                raw.flush()
                print(f"Completed round {sample + 1}/{args.rounds}", flush=True)
        (output / "summary.md").write_text(summarize(rows, args.rounds))
        provenance["sample_rows"] = len(rows)
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
