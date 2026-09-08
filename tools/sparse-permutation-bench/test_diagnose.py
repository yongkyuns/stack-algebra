import io
import csv
from pathlib import Path
import unittest

import compare
import diagnose


class FocusedDiagnosticTests(unittest.TestCase):
    def test_only_suite_selection_changes(self):
        original = (Path(__file__).parent / "src/main.rs").read_text()
        result = diagnose.focused_harness(original)
        self.assertEqual(original.split(diagnose.SUITE_START)[0], result.split(diagnose.SUITE_START)[0])
        self.assertEqual(original.split(diagnose.SUITE_END)[1], result.split(diagnose.SUITE_END)[1])
        suite = result.split(diagnose.SUITE_START)[1].split(diagnose.SUITE_END)[0]
        self.assertEqual(suite.count("run::<"), 2)
        self.assertIn("run::<128, 640, 1024, T>", suite)
        self.assertIn("run::<128, 2560, 1024, T>", suite)

    def test_rejects_unexpected_harness(self):
        with self.assertRaises(ValueError):
            diagnose.focused_harness("fn main() {}")

    def test_control_projection_is_explicit_and_does_not_mutate(self):
        rows = [{"revision": label, "sample": "0"} for label in ("before", "candidate", "control")]
        result = diagnose.project_rows(rows, "candidate", "control")
        self.assertEqual([r["revision"] for r in result], ["before", "candidate"])
        self.assertEqual([r["revision"] for r in rows], ["before", "candidate", "control"])

    def test_exact_focused_grid_and_rejects_other_dimensions(self):
        old = compare.EXPECTED
        try:
            compare.EXPECTED = {case for case in old if case[0] == "128"}
            self.assertEqual(len(compare.EXPECTED), 96)
            rows = []
            for case in sorted(compare.EXPECTED):
                row = dict(zip(compare.KEYS, case))
                n = int(row["n"])
                row.update(sample=0, input_capacity=n * (5 if row["capacity"] == "compact" else 20),
                           factor_capacity=n * 8, input_nnz=5*n-6 if row["layout"] == "full" else 3*n-3,
                           ordered_nnz=3*n-3, factor_nnz=compare.COUNTS[row["n"], row["ordering"]],
                           iterations=1, elapsed_ns=1)
                rows.append(row)
            def encode(rows):
                text = io.StringIO()
                writer = csv.DictWriter(text, fieldnames=compare.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                return text.getvalue()
            self.assertEqual(len(compare.read_samples(encode(rows), 0)), 96)
            rows[0]["n"] = "64"
            with self.assertRaises(ValueError):
                compare.read_samples(encode(rows), 0)
        finally:
            compare.EXPECTED = old


if __name__ == "__main__":
    unittest.main()
