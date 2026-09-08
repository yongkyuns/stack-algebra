import unittest
import profile as components


class ComponentValidation(unittest.TestCase):
    def rows(self):
        rows = []
        for sample in range(2):
            for key in sorted(components.EXPECTED):
                row = dict(zip(components.KEYS, key))
                n = int(row["n"])
                row.update(sample=str(sample), iterations="10", elapsed_ns="100",
                           nnz=str(5 * n - 6 if row["layout"] == "full" else 3 * n - 3))
                rows.append(row)
        return rows

    def test_complete_grid(self):
        self.assertIn("Median ns", components.summarize(self.rows(), 2))

    def test_missing_duplicate_and_wrong_round(self):
        rows = self.rows()
        for bad in [rows[:-1], rows + [rows[0]], [dict(r, sample="9") for r in rows]]:
            with self.assertRaises(ValueError):
                components.summarize(bad, 2)

    def test_bad_metadata_and_empty_measurements(self):
        for field, value in [("nnz", "0"), ("iterations", "0"), ("elapsed_ns", "0")]:
            rows = self.rows()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                components.summarize(rows, 2)


if __name__ == "__main__":
    unittest.main()
