"""Synthetic renderer fixtures only; documentation generation always executes Rust."""
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import generate_rls_assets as rls


def fixture():
    # Two deliberately simple reporting records, not a second RLS implementation.
    trace = []
    for step in (1, 2):
        row = dict.fromkeys(rls.TRACE_COLUMNS, 0.0)
        row.update(step=float(step), source_row=float(step - 1), x=float(step - 1),
                   observed_y=1.0, reference_y=1.0, prediction_before=float(step - 1),
                   innovation=float(2 - step), c=1.0, final_fitted_y=1.0,
                   reference_c=1.0, input_scale=3.0, prior_precision=0.01, forgetting=1.0,
                   sample_count=2.0, storage_bytes_f32=64.0, storage_bytes_f64=120.0)
        trace.append(row)
    curves = [dict(step=float(step), x=float(x), fitted_y=1.0, reference_y=1.0,
                   a=0.0, b=0.0, c=1.0, sample_count=4.0) for step in (1, 2) for x in (0, 1)]
    return trace, curves


def csv_text(rows, columns):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


class RlsAssetsTests(unittest.TestCase):
    def test_valid_export_and_deterministic_accessible_render(self):
        trace, curves = fixture()
        before = copy.deepcopy((trace, curves))
        rls.validate(trace, curves)
        result = rls.render(trace, curves)
        self.assertEqual(result, rls.render(trace, curves))
        self.assertEqual(before, (trace, curves))
        self.assertEqual(len(result), 7)  # Two fixture snapshots + five diagnostic figures.
        for svg in result.values():
            element = ET.fromstring(svg)
            self.assertEqual(element.attrib['role'], 'img')
            self.assertIsNotNone(element.find('{http://www.w3.org/2000/svg}desc'))
        self.assertIn('BEFORE', result['rls-online-error.svg'])

    def test_rejects_wrong_time_semantics_and_nonfinite_truncated_exports(self):
        for key in ('prediction_before', 'innovation', 'final_fitted_y', 'final_residual', 'step', 'source_row'):
            trace, curves = fixture()
            trace[0][key] += 2
            with self.subTest(key=key), self.assertRaises(ValueError):
                rls.validate(trace, curves)
        trace, _ = fixture()
        for invalid in (float('nan'), float('inf')):
            trace[0]['a'] = invalid
            with self.assertRaises(ValueError):
                rls.shared.read_csv(csv_text(trace, rls.TRACE_COLUMNS), rls.TRACE_COLUMNS)
        with self.assertRaises(ValueError):
            rls.shared.read_csv(csv_text(fixture()[0][:1], rls.TRACE_COLUMNS), rls.TRACE_COLUMNS)

    def test_rejects_snapshot_corruption(self):
        for key in ('step', 'a', 'fitted_y', 'reference_y', 'x'):
            trace, curves = fixture()
            curves[0][key] += 5
            with self.subTest(key=key), self.assertRaises(ValueError):
                rls.validate(trace, curves)

    def test_snapshot_does_not_show_future_observations(self):
        trace, curves = fixture()
        before = rls.render(trace, curves)
        trace[-1]['observed_y'] = 100.0
        after = rls.render(trace, curves)
        self.assertEqual(before['rls-after-1.svg'], after['rls-after-1.svg'])
        self.assertNotEqual(before['rls-after-2.svg'], after['rls-after-2.svg'])

    def test_no_stale_fallback_on_execution_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'docs/generated/rls'
            output.mkdir(parents=True)
            (output / 'old.svg').write_text('stale')
            with patch.object(rls.shared, 'capture', side_effect=subprocess.CalledProcessError(1, ['cargo'])):
                with self.assertRaises(subprocess.CalledProcessError):
                    rls.generate(root)
            self.assertFalse(output.exists())

    def test_generation_reexecutes_and_hashes_provenance(self):
        trace, curves = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'Cargo.toml').write_text('fixture source')
            (root / 'Cargo.lock').write_text('fixture lock')
            calls = []

            def execute(directory, command):
                self.assertEqual(directory, root)
                calls.append(command)
                if command[:2] == ['cargo', 'run']:
                    self.assertIn('--locked', command)
                    if command[-1] == '--curve-csv':
                        return csv_text(curves, rls.CURVE_COLUMNS)
                    if command[-1] == '--csv':
                        return csv_text(trace, rls.TRACE_COLUMNS)
                    return rls.terminal_text(trace)
                if command[:2] == ['git', 'ls-files']:
                    return 'Cargo.toml\0'
                if command[:2] == ['git', 'status']:
                    return ''
                return 'fixture\n'

            with patch.object(rls.shared, 'capture', side_effect=execute):
                rls.generate(root)
                rls.generate(root, check=True)
                self.assertEqual(sum(c[:2] == ['cargo', 'run'] for c in calls), 6)
                output = root / 'docs/generated/rls'
                manifest = json.loads((output / 'provenance.json').read_text())
                for name, sha in manifest['outputs_sha256'].items():
                    self.assertEqual(rls.shared.digest(output / name), sha)
                (output / 'rls-online-error.svg').write_text('stale')
                with self.assertRaisesRegex(ValueError, 'not reproducible'):
                    rls.generate(root, check=True)
                self.assertEqual((output / 'rls-online-error.svg').read_text(), 'stale')


if __name__ == '__main__':
    unittest.main()
