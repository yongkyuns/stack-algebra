#!/usr/bin/env python3
"""Diagnose N128 permutation code generation; not a performance acceptance gate."""
import argparse
import csv
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import compare as shared

BASELINE = "d1cd847ec4ef307dab7a91518d2bf1a2a11f90dc"
SUITE_START = "fn suite<T: Real + Debug>(sample: usize, timed: bool) {"
SUITE_END = "\nfn main() {"
FOCUSED_SUITE = """fn suite<T: Real + Debug>(sample: usize, timed: bool) {
    for layout in [\"lower\", \"full\"] {
        for kind in [\"identity\", \"reverse\", \"rotate\"] {
            run::<128, 640, 1024, T>(\"compact\", layout, kind, sample, timed);
            run::<128, 2560, 1024, T>(\"roomy\", layout, kind, sample, timed);
        }
    }
}
"""


def focused_harness(original):
    """Change only suite selection; keep run, timing and assertions verbatim."""
    if original.count(SUITE_START) != 1 or original.count(SUITE_END) != 1:
        raise ValueError("unexpected public harness structure")
    start = original.index(SUITE_START)
    end = original.index(SUITE_END, start)
    return original[:start] + FOCUSED_SUITE + original[end:]


def project_rows(rows, before, candidate):
    return [dict(row, revision="before" if row["revision"] == before else "candidate")
            for row in rows if row["revision"] in (before, candidate)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tool = Path(__file__).resolve().parent
    repo = tool.parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate = shared.capture(["git", "rev-parse", "--verify", args.candidate + "^{commit}"], repo)
    subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, candidate], cwd=repo, check=True)
    original = (tool / "src/main.rs").read_text()
    focused = focused_harness(original)
    # All dimensions other than N128 are omitted only in this diagnostic copy.
    # Preserve both scalar types, capacities, source layouts, orders and pipelines.
    shared.EXPECTED = {case for case in shared.EXPECTED if case[0] == "128"}
    assert len(shared.EXPECTED) == 96
    env = os.environ.copy()
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    env.update(CARGO_INCREMENTAL="0", RUSTFLAGS="-C target-cpu=native",
               CARGO_PROFILE_RELEASE_OPT_LEVEL="3", CARGO_PROFILE_RELEASE_CODEGEN_UNITS="1",
               CARGO_PROFILE_RELEASE_LTO="false")
    provenance = {
        "purpose": "Focused diagnostic only; unchanged full grid required for acceptance",
        "revisions": {"before": BASELINE, "candidate": candidate, "control": candidate},
        "harness_checkout": shared.capture(["git", "rev-parse", "HEAD"], repo),
        "rustc": shared.capture(["rustc", "-Vv"], repo),
        "cargo": shared.capture(["cargo", "-V"], repo),
        "rustflags": env["RUSTFLAGS"], "rounds": 12,
        "release": {"opt_level": 3, "codegen_units": 1, "lto": False},
        "files": {}, "source_sha256": {}, "binaries": {}, "diagnostics": {}, "order": [],
        "expected_cases": sorted(shared.EXPECTED),
    }
    for name, data in {
        "original-harness.rs": original.encode(), "focused-harness.rs": focused.encode(),
        "manifest.toml": (tool / "Cargo.toml").read_bytes(),
        "compare.py": (tool / "compare.py").read_bytes(),
        "diagnose.py": Path(__file__).read_bytes(),
        "test_compare.py": (tool / "test_compare.py").read_bytes(),
        "test_diagnose.py": (tool / "test_diagnose.py").read_bytes(),
    }.items():
        (out / name).write_bytes(data)
        provenance["files"][name] = shared.digest(data)
    (out / "cpu.txt").write_text(shared.capture(["lscpu"], repo) + "\n")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="permutation-focus-") as tmp:
        binaries = {}
        lock = None
        for label, sha in provenance["revisions"].items():
            root = Path(tmp) / label
            root.mkdir()
            archive = subprocess.check_output(["git", "archive", sha], cwd=repo)
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(root, filter="data")
            provenance["source_sha256"][label] = {
                str(p.relative_to(root)): shared.digest(p.read_bytes())
                for p in sorted((root / "src").rglob("*.rs"))
            }
            dest = root / "tools/sparse-permutation-bench"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(tool, dest, ignore=shutil.ignore_patterns("target", "__pycache__", "Cargo.lock"))
            (dest / "src/main.rs").write_text(focused)
            manifest = dest / "Cargo.toml"
            if lock is None:
                subprocess.run(["cargo", "generate-lockfile", "--manifest-path", str(manifest)],
                               cwd=root, env=env, check=True)
                lock = (dest / "Cargo.lock").read_bytes()
                (out / "Cargo.lock").write_bytes(lock)
                provenance["lock_sha256"] = shared.digest(lock)
            else:
                (dest / "Cargo.lock").write_bytes(lock)
            build_env = dict(env, CARGO_TARGET_DIR=str(root / "bench-target"))
            with (out / f"build-{label}.log").open("w") as log:
                subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", str(manifest)],
                               cwd=root, env=build_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            binary = root / "bench-target/release/sparse-permutation-bench"
            binaries[label] = binary
            provenance["binaries"][label] = shared.digest(binary.read_bytes())
            checked = shared.capture([str(binary), "--check"], repo, env)
            shared.read_samples(checked, 0, timed=False)
            (out / f"check-{label}.csv").write_text(checked + "\n")
            print(f"Built and checked focused {label}: {sha}", flush=True)
        if provenance["source_sha256"]["candidate"] != provenance["source_sha256"]["control"]:
            raise ValueError("same-source control differs from candidate")
        if hasattr(os, "sched_getaffinity"):
            allowed = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {allowed[0]})
            provenance.update(available_cpus=allowed, measurement_cpu=allowed[0])
        rows = []
        # Twelve rounds: all six execution orders twice, with operation ordering
        # still reversed on odd rounds by the unchanged public run function.
        orders = list(itertools.permutations(("before", "candidate", "control")))
        with (out / "samples.csv").open("w", newline="") as raw:
            writer = csv.DictWriter(raw, fieldnames=["revision"] + shared.FIELDS)
            writer.writeheader()
            for sample in range(12):
                order = orders[sample % len(orders)]
                provenance["order"].append(order)
                for label in order:
                    text = shared.capture([str(binaries[label]), str(sample)], repo, env)
                    for row in shared.read_samples(text, sample):
                        row["revision"] = label
                        writer.writerow(row)
                        rows.append(row)
                raw.flush()
                print(f"Focused paired round {sample + 1}/12", flush=True)
        for name, first, second, description in [
            ("summary.md", "before", "candidate", "Focused N128 diagnostic: repaired baseline vs candidate."),
            ("control-summary.md", "candidate", "control", "Focused N128 same-source independent-build control."),
        ]:
            (out / name).write_text(shared.summarize(project_rows(rows, first, second), 12, description))
        provenance["sample_rows"] = len(rows)
        provenance["samples_sha256"] = shared.digest((out / "samples.csv").read_bytes())
        # Post-timing diagnostics retain the actual executables, not rebuilds.
        for label, binary in binaries.items():
            retained = out / f"{label}.elf"
            shutil.copyfile(binary, retained)
            if shared.digest(retained.read_bytes()) != provenance["binaries"][label]:
                raise ValueError("retained executable differs from measured binary")
            for suffix, command in [
                ("objdump.txt", ["objdump", "-Cd", str(binary)]),
                ("symbols.txt", ["nm", "-S", "--size-sort", "--demangle", str(binary)]),
            ]:
                path = out / f"{label}.{suffix}"
                with path.open("w") as report:
                    subprocess.run(command, cwd=repo, stdout=report, check=True)
                provenance["diagnostics"][path.name] = shared.digest(path.read_bytes())
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
