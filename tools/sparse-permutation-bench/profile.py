#!/usr/bin/env python3
"""Profile #51's unchanged permutation checks and pattern-copy alternatives in isolation."""
import argparse
import csv
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

from compare import capture, digest

BASELINE = "d1cd847ec4ef307dab7a91518d2bf1a2a11f90dc"
WRAPPERS = r'''
    /// Isolated baseline bounds-scan profile; never included in published source.
    #[inline]
    pub fn __profile_bounds<T: Copy + Zero>(
        &self,
        matrix: &StaticCscMatrix<N, N, MAX_NNZ, T>,
    ) -> bool {
        self.source_indices[..self.nnz]
            .iter()
            .all(|&index| (index as usize) < matrix.values().len())
    }

    /// Isolated whole-pattern copy profile.
    #[inline]
    pub fn __profile_copy<T: Copy + Zero>(
        &self,
        output: &mut StaticCscMatrix<N, N, MAX_NNZ, T>,
    ) {
        output.pattern = self.pattern;
    }

    /// Isolated exact-equality guarded copy profile for matching destinations.
    #[inline]
    pub fn __profile_guarded_copy<T: Copy + Zero>(
        &self,
        output: &mut StaticCscMatrix<N, N, MAX_NNZ, T>,
    ) {
        if output.pattern != self.pattern {
            output.pattern = self.pattern;
        }
    }
'''
PROFILE_RUN = r'''
fn profile<const N: usize, const A: usize, T: Real + Debug>(
    capacity: &str, layout: &str, kind: &str, sample: usize,
) {
    let input = fixture::<N, A, T>(layout == "full");
    let map = ordering::<N>(kind).permutation_for_pattern(input.pattern()).unwrap();
    let mut output = StaticCscMatrix::zero_with_pattern(map.pattern());
    let required_len = input.nnz(); // These fixtures store the last diagonal.
    let mut components = ["bounds_scan", "bounds_length", "copy_pattern", "guarded_pattern"];
    if sample & 1 != 0 { components.reverse(); }
    for component in components {
        let (iterations, elapsed) = match component {
            "bounds_scan" => measure(|| {
                black_box(black_box(&map).__profile_bounds(black_box(&input)));
            }, true),
            "bounds_length" => measure(|| {
                black_box(black_box(required_len) <= black_box(&input).values().len());
            }, true),
            "copy_pattern" => measure(|| {
                black_box(&map).__profile_copy(black_box(&mut output));
                black_box(&output);
            }, true),
            "guarded_pattern" => measure(|| {
                black_box(&map).__profile_guarded_copy(black_box(&mut output));
                black_box(&output);
            }, true),
            _ => unreachable!(),
        };
        assert!(map.__profile_bounds(&input));
        assert_eq!(output.pattern(), map.pattern_ref());
        println!("{sample},{N},{},{capacity},{layout},{kind},{component},{},{iterations},{elapsed}",
                 core::any::type_name::<T>(), input.nnz());
    }
}

fn profiles<T: Real + Debug>(sample: usize) {
    for layout in ["lower", "full"] {
        for kind in ["identity", "reverse", "rotate"] {
            profile::<6, 30, T>("compact", layout, kind, sample);
            profile::<6, 120, T>("roomy", layout, kind, sample);
            profile::<15, 75, T>("compact", layout, kind, sample);
            profile::<15, 300, T>("roomy", layout, kind, sample);
            profile::<64, 320, T>("compact", layout, kind, sample);
            profile::<64, 1280, T>("roomy", layout, kind, sample);
            profile::<128, 640, T>("compact", layout, kind, sample);
            profile::<128, 2560, T>("roomy", layout, kind, sample);
        }
    }
}

fn main() {
    let sample = std::env::args().nth(1).unwrap().parse().unwrap();
    println!("sample,n,scalar,capacity,layout,ordering,component,nnz,iterations,elapsed_ns");
    profiles::<f32>(sample);
    profiles::<f64>(sample);
}
'''
KEYS = "n,scalar,capacity,layout,ordering,component".split(",")
FIELDS = ["sample"] + KEYS + ["nnz", "iterations", "elapsed_ns"]
EXPECTED = set(itertools.product(
    ("6", "15", "64", "128"), ("f32", "f64"), ("compact", "roomy"),
    ("lower", "full"), ("identity", "reverse", "rotate"),
    ("bounds_scan", "bounds_length", "copy_pattern", "guarded_pattern"),
))


def summarize(rows, rounds):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        samples = groups.setdefault(key, {})
        sample = int(row["sample"])
        if sample in samples or int(row["iterations"]) <= 0 or int(row["elapsed_ns"]) <= 0:
            raise ValueError("duplicate or empty component measurement")
        n = int(row["n"])
        if int(row["nnz"]) != (5 * n - 6 if row["layout"] == "full" else 3 * n - 3):
            raise ValueError("invalid component fixture")
        samples[sample] = int(row["elapsed_ns"]) / int(row["iterations"])
    if set(groups) != EXPECTED or any(set(v) != set(range(rounds)) for v in groups.values()):
        raise ValueError("incomplete component grid")
    lines = ["# Permutation components on unchanged repaired source", "",
             "Isolated warm-cache components are not additive whole-call decompositions.",
             "The scalar bound is a proposed check cost, not a validated replacement by itself.",
             "Guarded copying is measured with matching destinations only.", "",
             "| Case | Median ns | Min ns | Max ns |", "|---|---:|---:|---:|"]
    for key, samples in sorted(groups.items()):
        values = samples.values()
        lines.append(f"| {'/'.join(key)} | {statistics.median(values):.2f} | {min(values):.2f} | {max(values):.2f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if not 2 <= args.rounds <= 12:
        parser.error("rounds must be between 2 and 12")
    tool = Path(__file__).resolve().parent
    repo = tool.parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CARGO_INCREMENTAL="0", RUSTFLAGS="-C target-cpu=native",
               CARGO_PROFILE_RELEASE_OPT_LEVEL="3", CARGO_PROFILE_RELEASE_CODEGEN_UNITS="1",
               CARGO_PROFILE_RELEASE_LTO="false")
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    provenance = {"source": BASELINE, "harness_checkout": capture(["git", "rev-parse", "HEAD"], repo),
                  "rustc": capture(["rustc", "-Vv"], repo), "cargo": capture(["cargo", "-V"], repo),
                  "rustflags": env["RUSTFLAGS"], "rounds": args.rounds}
    with tempfile.TemporaryDirectory(prefix="permutation-components-") as tmp:
        root = Path(tmp)
        archive = subprocess.check_output(["git", "archive", BASELINE], cwd=repo)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root, filter="data")
        path = root / "src/sparse/ordering.rs"
        original = path.read_text()
        anchor = "    /// Returns the ordered pattern."
        if original.count(anchor) != 1:
            raise ValueError("baseline profile insertion point changed")
        path.write_text(original.replace(anchor, WRAPPERS + "\n" + anchor))
        provenance["original_ordering_sha256"] = digest(original.encode())
        provenance["profiled_ordering_sha256"] = digest(path.read_bytes())
        (out / "profiled-ordering.rs").write_bytes(path.read_bytes())
        dest = root / "tools/sparse-permutation-bench"
        shutil.rmtree(dest)
        (dest / "src").mkdir(parents=True)
        env["CARGO_TARGET_DIR"] = str(dest / "target")
        shutil.copyfile(tool / "Cargo.toml", dest / "Cargo.toml")
        prefix = (tool / "src/main.rs").read_text().split("#[inline(never)]\nfn run", 1)[0]
        harness = "#![allow(dead_code, unused_imports)]\n" + prefix + PROFILE_RUN
        (dest / "src/main.rs").write_text(harness)
        (out / "profile-harness.rs").write_text(harness)
        shutil.copyfile(__file__, out / "profile-driver.py")
        provenance["harness_sha256"] = digest(harness.encode())
        subprocess.run(["cargo", "generate-lockfile", "--manifest-path", str(dest / "Cargo.toml")],
                       cwd=root, env=env, check=True)
        shutil.copyfile(dest / "Cargo.lock", out / "Cargo.lock")
        provenance["lock_sha256"] = digest((out / "Cargo.lock").read_bytes())
        with (out / "build.log").open("w") as log:
            subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", str(dest / "Cargo.toml")],
                           cwd=root, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        binary = dest / "target/release/sparse-permutation-bench"
        provenance["binary_sha256"] = digest(binary.read_bytes())
        (out / "cpu.txt").write_text(capture(["lscpu"], repo) + "\n")
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {allowed[0]})
        provenance.update(available_cpus=allowed, measurement_cpu=allowed[0])
        rows = []
        for sample in range(args.rounds):
            reader = csv.DictReader(io.StringIO(capture([str(binary), str(sample)], repo, env)))
            if reader.fieldnames != FIELDS:
                raise ValueError("unexpected component schema")
            block = list(reader)
            if len(block) != len(EXPECTED) or any(int(r["sample"]) != sample for r in block):
                raise ValueError("invalid component round")
            rows.extend(block)
        summary = summarize(rows, args.rounds)
        with (out / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (out / "summary.md").write_text(summary)
        provenance.update(sample_rows=len(rows), samples_sha256=digest((out / "samples.csv").read_bytes()))
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
