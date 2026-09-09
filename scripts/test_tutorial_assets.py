#!/usr/bin/env python3
"""Synthetic fixtures test plumbing only; production generation always runs Rust."""
import copy
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import generate_tutorial_assets as assets


def csv_text(rows, columns):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def polynomial_fixture():
    # Deliberately not the documentation's 40 noisy observations.
    return [dict(x=x, observed_y=x*x+2*x+3, reference_y=x*x+2*x+3, noise=0,
                 fitted_y=x*x+2*x+3, residual=0, a=1, b=2, c=3,
                 reference_a=1, reference_b=2, reference_c=3,
                 residual_norm=0, sample_count=3) for x in (-1, 0, 1)]


def curve_fixture():
    return [dict(x=x, fitted_y=x*x+2*x+3, reference_y=x*x+2*x+3, sample_count=5)
            for x in (-1, -0.5, 0, 0.5, 1)]


def kalman_fixture():
    row = dict.fromkeys(assets.KALMAN_COLUMNS, 0.0)
    row.update(time_s=0.5, measured_m=3, dt_s=0.5, measurement_variance=1,
               initial_position_m=1.5, initial_velocity_m_s=1,
               initial_p00=1, initial_p11=1,
               predicted_position_m=2, predicted_velocity_m_s=1,
               predicted_p00=4, predicted_p01=2, predicted_p10=2, predicted_p11=3,
               position_m=2.8, velocity_m_s=1.4, p00=0.8, p01=0.4, p10=0.4, p11=2.2,
               innovation_m=1, innovation_variance=5, position_gain=0.8, velocity_gain=0.4,
               sample_count=1)
    return [row]


class ExportTests(unittest.TestCase):
    def test_valid_synthetic_exports(self):
        rows = assets.read_csv(csv_text(polynomial_fixture(), assets.POLYNOMIAL_COLUMNS), assets.POLYNOMIAL_COLUMNS)
        curve = assets.read_csv(csv_text(curve_fixture(), assets.CURVE_COLUMNS), assets.CURVE_COLUMNS)
        kalman = assets.read_csv(csv_text(kalman_fixture(), assets.KALMAN_COLUMNS), assets.KALMAN_COLUMNS)
        assets.validate_polynomial(rows, curve)
        assets.validate_kalman(kalman)

    def test_rejects_missing_reordered_and_duplicate_columns(self):
        for expected in (assets.POLYNOMIAL_COLUMNS, assets.CURVE_COLUMNS, assets.KALMAN_COLUMNS):
            for columns in (expected[:-1], list(reversed(expected)), expected + [expected[0]]):
                with self.subTest(columns=columns), self.assertRaises(ValueError):
                    assets.read_csv(','.join(columns) + '\n', expected)

    def test_rejects_empty_truncated_or_nonfinite_data(self):
        for fixture, columns in ((polynomial_fixture, assets.POLYNOMIAL_COLUMNS), (curve_fixture, assets.CURVE_COLUMNS)):
            text = csv_text(fixture(), columns)
            for malformed in (text.splitlines()[0] + '\n', '\n'.join(text.splitlines()[:-1]),
                              text + '1,2\n', text + ','.join(['1'] * (len(columns)+1)) + '\n'):
                with self.subTest(text=malformed), self.assertRaises(ValueError):
                    assets.read_csv(malformed, columns)
            for invalid in (float('nan'), float('inf'), -float('inf')):
                rows = fixture()
                rows[0]['x'] = invalid
                with self.assertRaises(ValueError):
                    assets.read_csv(csv_text(rows, columns), columns)

    def test_rejects_residual_sign_and_inconsistent_polynomial(self):
        for field, value in (('residual', 0.1), ('a', 9), ('b', 8), ('c', 6), ('residual_norm', 1),
                             ('fitted_y', 5), ('noise', 1), ('reference_y', 10), ('reference_a', 2)):
            rows = polynomial_fixture()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                assets.validate_polynomial(rows, curve_fixture())

    def test_rejects_invalid_curve_values_or_grid(self):
        for index, field, value in ((1, 'fitted_y', 10), (1, 'reference_y', 10),
                                    (0, 'x', -2), (4, 'x', 2), (2, 'x', -0.5)):
            curve = curve_fixture()
            curve[index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                assets.validate_polynomial(polynomial_fixture(), curve)
        with self.assertRaises(ValueError):
            assets.validate_polynomial(polynomial_fixture(), [])
        rows = polynomial_fixture()
        rows[0]['x'] = 0
        with self.assertRaises(ValueError):
            assets.validate_polynomial(rows, curve_fixture())

    def test_rejects_invalid_kalman_diagnostics(self):
        for field, value in (('dt_s', 0), ('measurement_variance', 0), ('acceleration_variance', -1),
                             ('time_s', 0), ('innovation_m', -1), ('innovation_variance', 0),
                             ('position_m', 3), ('velocity_m_s', 7), ('position_gain', 0),
                             ('velocity_gain', 0), ('p00', -1), ('p01', 10), ('p11', 0)):
            rows = kalman_fixture()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                assets.validate_kalman(rows)

    def test_text_formatting_is_only_presentation(self):
        self.assertIn('a = 1.000, b = 2.000, c = 3.000', assets.polynomial_text(polynomial_fixture()))
        self.assertIn('3.000 2.800 1.400', assets.kalman_text(kalman_fixture()))


class RenderingTests(unittest.TestCase):
    def test_deterministic_accessible_svg(self):
        inputs = polynomial_fixture(), kalman_fixture(), curve_fixture()
        before = copy.deepcopy(inputs)
        first = assets.render(*inputs)
        self.assertEqual(first, assets.render(*inputs))
        self.assertEqual(before, inputs)
        self.assertEqual(len(first), 5)
        for name, svg in first.items():
            element = ET.fromstring(svg)
            self.assertEqual(element.attrib['role'], 'img', name)
            self.assertIsNotNone(element.find('{http://www.w3.org/2000/svg}title'))
            self.assertIsNotNone(element.find('{http://www.w3.org/2000/svg}desc'))
            self.assertNotIn('nan', svg.lower())
            self.assertNotIn('Infinity', svg)

    def test_changed_execution_values_change_figures_and_labels(self):
        rows, kalman, curve = polynomial_fixture(), kalman_fixture(), curve_fixture()
        before = assets.render(rows, kalman, curve)
        for row in rows:
            row.update(a=2, fitted_y=2*row['x']**2+2*row['x']+3,
                       observed_y=2*row['x']**2+2*row['x']+3, noise=row['x']**2)
        for row in curve:
            row['fitted_y'] = 2*row['x']**2+2*row['x']+3
        assets.validate_polynomial(rows, curve)
        after = assets.render(rows, kalman, curve)
        self.assertNotEqual(before['polynomial-fit.svg'], after['polynomial-fit.svg'])
        self.assertIn('y = 2.000x²', after['polynomial-fit.svg'])
        self.assertIn('t = 0.5 s', after['kalman-first-update.svg'])
        self.assertIn('0.800000', after['kalman-first-update.svg'])
        self.assertNotIn('1.067', after['kalman-first-update.svg'])

    def test_renderer_draws_exported_grid_not_recomputed_model(self):
        rows, kalman, curve = polynomial_fixture(), kalman_fixture(), curve_fixture()
        before = assets.render(rows, kalman, curve)['polynomial-fit.svg']
        # Intentionally inconsistent for a renderer unit test; production validation rejects this.
        # An interior point exists only in the dense grid, not among observations.
        curve[1]['fitted_y'] += 0.4
        after = assets.render(rows, kalman, curve)['polynomial-fit.svg']
        self.assertNotEqual(before, after)
        element = ET.fromstring(after)
        ns = {'s': 'http://www.w3.org/2000/svg'}
        for key in ('fitted_y', 'reference_y'):
            path = element.find(f'.//s:g[@data-series="{key}"]/s:polyline', ns)
            self.assertIsNotNone(path)
            self.assertEqual(len(path.attrib['points'].split()), len(curve))
        self.assertIn('stroke-dasharray="7 5"', after)

    def test_constant_or_negative_axis_data_has_nonzero_extent(self):
        for values in ([0], [1, 1], [-1, -1], [-4, -2], [-0.001, 0.001]):
            lo, hi, ticks = assets.limits(values)
            self.assertLess(lo, min(values))
            self.assertGreater(hi, max(values))
            self.assertTrue(ticks)

    def test_failed_execution_removes_previous_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'docs/generated/tutorials'
            output.mkdir(parents=True)
            (output / 'old.svg').write_text('must not be reused')
            with patch.object(assets, 'capture', side_effect=subprocess.CalledProcessError(1, ['cargo'])):
                with self.assertRaises(subprocess.CalledProcessError):
                    assets.generate(root)
            self.assertFalse(output.exists())
            self.assertEqual(list((root / 'build').iterdir()), [])

    def test_generation_manifest_and_reexecution_check(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            (root / 'Cargo.toml').write_text('synthetic test manifest')

            def fake_execution(directory, command):
                self.assertEqual(directory, root)
                if command == ['cargo', 'generate-lockfile']:
                    (root / 'Cargo.lock').write_text('synthetic resolved dependencies')
                    return ''
                if command[:2] == ['cargo', 'run']:
                    self.assertIn('--locked', command)
                    self.assertIn('--no-default-features', command)
                    name = command[command.index('--example') + 1]
                    if command[-1] == '--curve-csv':
                        self.assertEqual(name, 'mapped_least_squares')
                        return csv_text(curve_fixture(), assets.CURVE_COLUMNS)
                    polynomial = name == 'mapped_least_squares'
                    rows = polynomial_fixture() if polynomial else kalman_fixture()
                    if command[-1] == '--csv':
                        return csv_text(rows, assets.EXAMPLES[name])
                    return assets.polynomial_text(rows) if polynomial else assets.kalman_text(rows)
                if command[:2] == ['git', 'ls-files']:
                    return 'Cargo.toml\0'
                if command[:2] == ['git', 'rev-parse']:
                    return '0' * 40 + '\n'
                if command[:2] == ['git', 'status']:
                    return ''
                if command[0] in ('cargo', 'rustc'):
                    return 'synthetic toolchain for unit testing\n'
                self.fail(f'unexpected command: {command}')

            with patch.object(assets, 'capture', side_effect=fake_execution):
                assets.generate(root)
                output = root / 'docs/generated/tutorials'
                manifest = json.loads((output / 'provenance.json').read_text())
                self.assertEqual(manifest['schema_version'], 2)
                self.assertEqual(len(manifest['executions']), 5)
                self.assertFalse(manifest['worktree_dirty'])
                self.assertIn(assets.CURVE_FILE, manifest['outputs_sha256'])
                for filename, sha in manifest['outputs_sha256'].items():
                    self.assertEqual(sha, assets.digest(output / filename))
                assets.generate(root, check=True)
                (output / 'polynomial-fit.svg').write_text('stale fixture')
                with self.assertRaisesRegex(ValueError, 'not reproducible'):
                    assets.generate(root, check=True)
                self.assertEqual((output / 'polynomial-fit.svg').read_text(), 'stale fixture')
                assets.generate(root)
                (root / 'Cargo.toml').write_text('changed source')
                with self.assertRaisesRegex(ValueError, 'not reproducible'):
                    assets.generate(root, check=True)

    def test_tutorials_use_existing_named_source_anchors(self):
        root = Path(__file__).resolve().parents[1]
        documents = list((root / 'docs').glob('tutorial-*.md'))
        self.assertGreaterEqual(len(documents), 2)
        for doc in documents:
            for filename, anchor in re.findall(r'\{\{#include ../examples/([^:}]+):([^}]+)\}\}', doc.read_text()):
                self.assertFalse(anchor[0].isdigit(), f'{doc}: fragile numeric include')
                source = (root / 'examples' / filename).read_text()
                self.assertEqual(source.count('ANCHOR: ' + anchor + '\n'), 1)
                self.assertEqual(source.count('ANCHOR_END: ' + anchor + '\n'), 1)


if __name__ == '__main__':
    unittest.main()
