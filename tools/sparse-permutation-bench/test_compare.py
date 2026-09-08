import csv
import io
import unittest

import compare


def sample_text():
    rows = []
    for n, scalar, capacity, layout, order, operation in sorted(compare.EXPECTED):
        size = int(n)
        rows.append([0, n, scalar, capacity, layout, order, operation,
                     size * (5 if capacity == "compact" else 20), size * 8,
                     5 * size - 6 if layout == "full" else 3 * size - 3,
                     3 * size - 3, compare.COUNTS[n, order], 64, 1000])
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(compare.FIELDS)
    writer.writerows(rows)
    return text.getvalue()


class SampleValidation(unittest.TestCase):
    def test_accepts_exact_fixture_grid(self):
        self.assertEqual(len(compare.read_samples(sample_text(), 0)), 384)

    def test_rejects_duplicate_and_missing_cases(self):
        lines = sample_text().splitlines()
        lines[-1] = lines[-2]
        with self.assertRaises(ValueError):
            compare.read_samples("\n".join(lines), 0)

    def test_rejects_wrong_round(self):
        with self.assertRaises(ValueError):
            compare.read_samples(sample_text(), 1)

    def test_rejects_incorrect_metadata_and_zero_timings(self):
        text = sample_text()
        for corrupt in [text.replace(",64,1000", ",64,0", 1), text.replace(",42,42,", ",41,42,", 1)]:
            self.assertNotEqual(corrupt, text)
            with self.assertRaises(ValueError):
                compare.read_samples(corrupt, 0)

    def test_summary_requires_paired_rounds(self):
        rows = compare.read_samples(sample_text(), 0)
        rows = [dict(row, revision="before") for row in rows]
        with self.assertRaises(ValueError):
            compare.summarize(rows, 12)


if __name__ == "__main__":
    unittest.main()
