#!/usr/bin/env python3
"""Profile the qualified safe baseline, then compare unchanged public fixtures."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile
import tempfile

from compare import FIELDS, capture, digest, read_samples

BASELINE = "3c97ceb89ead00eed0c754e9f6bc8be38ccba326"

# These wrappers are appended ONLY to an isolated profiling copy of the baseline.
# Neither the repository source nor the ordinary comparison builds are modified.
SHIM = r'''
impl<const N: usize, const L: usize> StaticCscCholeskyPattern<N, L> {
    /// Temporary component profiling entry point, not a library API.
    pub fn __profile_source<const A: usize, T: Copy + Zero>(
        &self, input: &StaticCscMatrix<N, N, A, T>,
    ) -> bool { self.matches_aggregate_input(input) }
    /// Temporary component profiling entry point, not a library API.
    pub fn __profile_coverage<const A: usize, T: Copy + Zero>(
        &self, input: &StaticCscMatrix<N, N, A, T>,
    ) -> bool { self.validate_factor_pattern(input).is_ok() }
    /// Temporary component profiling entry point, not a library API.
    pub fn __profile_copy(&self, output: &mut StaticCscPattern<N, N, L>, conditional: bool) {
        if !conditional || *output != self.lower { *output = self.lower; }
    }
}
'''

PROFILE = r'''
#[inline(never)]
fn run<const N: usize, const A: usize, const L: usize, T: Real + Debug>(
    star: bool, sample: usize,
) {
    let input = fixture::<N, A, T>(star, false);
    let pattern = StaticCscCholeskyPattern::<N, L>::analyze(&input).unwrap();
    let rhs = Matrix::<N, 1, T>::from_fn(|r, _| T::from(r % 5 + 1).unwrap());
    check_solution(&pattern.factor_ldlt(&input).unwrap().solve(&rhs), star);
    assert!(pattern.__profile_source(&input));
    assert!(pattern.__profile_coverage(&input));
    let mut output = *pattern.lower();
    let components = if sample.is_multiple_of(2) {
        ["source", "coverage", "copy", "conditional_copy"]
    } else { ["conditional_copy", "copy", "coverage", "source"] };
    for component in components {
        let (iterations, elapsed) = match component {
            "source" => measure(|| { black_box(black_box(&pattern).__profile_source(black_box(&input))); }, true),
            "coverage" => measure(|| { black_box(black_box(&pattern).__profile_coverage(black_box(&input))); }, true),
            _ => measure(|| {
                black_box(&pattern).__profile_copy(black_box(&mut output), component == "conditional_copy");
                black_box(&output);
            }, true),
        };
        assert_eq!(&output, pattern.lower());
        let shape = if star { "star" } else { "banded" };
        let scalar = core::any::type_name::<T>();
        println!("{sample},{shape},{N},{scalar},{component},{iterations},{elapsed}");
    }
}
fn suite<T: Real + Debug>(sample: usize) {
    run::<6, 30, 15, T>(false, sample);
    run::<15, 75, 42, T>(false, sample);
    run::<64, 320, 189, T>(false, sample);
    run::<128, 640, 381, T>(false, sample);
    run::<64, 256, 2080, T>(true, sample);
    run::<128, 512, 8256, T>(true, sample);
}
fn main() {
    let sample: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    println!("sample,shape,n,scalar,component,iterations,elapsed_ns");
    suite::<f32>(sample);
    suite::<f64>(sample);
}
'''


def summarize(rows, rounds):
    cases = {}
    for row in rows:
        key = tuple(row[k] for k in ("shape", "n", "scalar", "solver", "entry", "layout"))
        samples = cases.setdefault(key, {}).setdefault(row["revision"], {})
        sample = int(row["sample"])
        if sample in samples:
            raise ValueError("duplicate measurement")
        samples[sample] = int(row["elapsed_ns"]) / int(row["iterations"])
    if len(cases) != 144:
        raise ValueError("incomplete fixture coverage")
    lines = ["# Safe-baseline sparse reuse comparison", "",
             "Nanoseconds per whole recomputation, warm-cache hosted-runner fixtures.",
             "Ratios are paired-round medians with observed min/max, not confidence intervals.",
             "Setup, analysis, allocation and residual checks are excluded. Both revisions are safe.", "",
             "| Case | Baseline ns | Candidate ns | Candidate / baseline |",
             "|---|---:|---:|---:|"]
    for key, versions in sorted(cases.items()):
        if set(versions) != {"baseline", "candidate"}:
            raise ValueError("missing revision")
        for samples in versions.values():
            if set(samples) != set(range(rounds)):
                raise ValueError("missing round")
        a, b = versions["baseline"], versions["candidate"]
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
        parser.error("rounds must be even and between 6 and 30")
    tool = Path(__file__).resolve().parent
    repo = tool.parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate = capture(["git", "rev-parse", "--verify", args.candidate + "^{commit}"], repo)
    subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, candidate], cwd=repo, check=True)
    env = os.environ.copy()
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    env.update(CARGO_INCREMENTAL="0", RUSTFLAGS="-C target-cpu=native")
    harness = (tool / "src/main.rs").read_text()
    manifest_bytes = (tool / "Cargo.toml").read_bytes()
    provenance = {
        "revisions": {"baseline": BASELINE, "candidate": candidate},
        "harness_checkout": capture(["git", "rev-parse", "HEAD"], repo),
        "harness_sha256": digest(harness.encode()), "manifest_sha256": digest(manifest_bytes),
        "rustc": capture(["rustc", "-Vv"], repo), "cargo": capture(["cargo", "-V"], repo),
        "rustflags": env["RUSTFLAGS"], "rounds": args.rounds,
        "release": {"opt_level": 3, "codegen_units": 1, "lto": False},
        "binaries": {}, "source_sha256": {}, "order": [],
    }
    (out / "cpu.txt").write_text(capture(["lscpu"], repo) + "\n")
    (out / "harness.rs").write_text(harness)
    (out / "manifest.toml").write_bytes(manifest_bytes)
    (out / "driver.py").write_bytes(Path(__file__).read_bytes())
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        provenance["available_cpus"] = allowed
        provenance["measurement_cpu"] = allowed[0]
    with tempfile.TemporaryDirectory(prefix="sparse-optimization-") as tmp:
        binaries = {}
        lock = None
        for label, sha in [*provenance["revisions"].items(), ("profile", BASELINE)]:
            root = Path(tmp) / label
            root.mkdir()
            archive = subprocess.check_output(["git", "archive", sha], cwd=repo)
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(root, filter="data")
            dest = root / "tools/sparse-reuse-bench"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(tool, dest, ignore=shutil.ignore_patterns("target", "__pycache__", "Cargo.lock"))
            source_path = root / "src/sparse/cholesky.rs"
            if label == "profile":
                source_path.write_text(source_path.read_text() + SHIM)
                profile_harness = harness[:harness.index("#[inline(never)]")] + PROFILE
                (dest / "src/main.rs").write_text(profile_harness)
                (out / "profile-harness.rs").write_text(profile_harness)
                (out / "profile-shim.rs").write_text(SHIM)
            provenance["source_sha256"][label] = digest(source_path.read_bytes())
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
                subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", str(manifest)], cwd=root, env=build_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            binary = root / "bench-target/release/sparse-reuse-bench"
            binaries[label] = binary
            provenance["binaries"][label] = digest(binary.read_bytes())
            print(f"Built {label} {sha}", flush=True)
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {provenance["measurement_cpu"]})
        profiles = []
        for sample in range(6):
            text = capture([str(binaries["profile"]), str(sample)], repo, env)
            batch = list(csv.DictReader(io.StringIO(text)))
            if len(batch) != 48 or any(int(r["iterations"]) <= 0 or int(r["elapsed_ns"]) <= 0 for r in batch):
                raise ValueError("incomplete component profile")
            profiles.extend(batch)
        with (out / "components.csv").open("w", newline="") as raw:
            writer = csv.DictWriter(raw, fieldnames=list(profiles[0]))
            writer.writeheader()
            writer.writerows(profiles)
        rows = []
        with (out / "samples.csv").open("w", newline="") as raw:
            writer = csv.DictWriter(raw, fieldnames=["revision"] + FIELDS)
            writer.writeheader()
            for sample in range(args.rounds):
                order = ["baseline", "candidate"] if sample % 2 == 0 else ["candidate", "baseline"]
                provenance["order"].append(order)
                for label in order:
                    for mode, count in [("matching", 48), ("layouts", 96)]:
                        text = capture([str(binaries[label]), mode, str(sample)], repo, env)
                        batch = read_samples(text, count)
                        for row in batch:
                            row["revision"] = label
                            writer.writerow(row)
                        rows.extend(batch)
                raw.flush()
                print(f"Completed paired round {sample + 1}/{args.rounds}", flush=True)
        (out / "summary.md").write_text(summarize(rows, args.rounds))
        provenance["sample_rows"] = len(rows)
        provenance["profile_rows"] = len(profiles)
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
