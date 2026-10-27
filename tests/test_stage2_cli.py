"""Run the standalone command, including failure exit codes and real files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ENV_DIR = Path(__file__).resolve().parents[1] / 'embodied/envs'
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))
import scenario_data as data
from test_world_model_adapter import episode_and_window

SCRIPT = ENV_DIR / 'evaluate_scenario_risk.py'


class Stage2CliTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), 'stage 2 CLI is missing')

    def run_cli(self, source, output, extra=()):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run([sys.executable, str(SCRIPT), '--input', str(source), '--output', str(output),
                               '--history', '5', '--future', '30', '--seed', '7'] + list(extra),
                              env=env, capture_output=True, text=True)

    def test_native_report_determinism_and_unavailable_model_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _ = episode_and_window()
            source = data.write_episode(episode, root/'input')
            for output in (root/'one', root/'two'):
                result = self.run_cli(source, output)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual((root/'one/windows.jsonl').read_bytes(), (root/'two/windows.jsonl').read_bytes())
            summary = json.loads((root/'one/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['window_count'], 31)
            self.assertEqual(summary['valid_rule_window_count'], 31)
            self.assertFalse(summary['model']['inference_verified'])
            self.assertAlmostEqual(summary['dt'], .2)
            self.assertEqual(len((root/'one/observed_frames.jsonl').read_text().splitlines()), 65)

    def test_requested_missing_checkpoint_fails_but_preserves_rule_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _ = episode_and_window()
            source = data.write_episode(episode, root/'input')
            result = self.run_cli(source, root/'report', ['--checkpoint', str(root/'missing.ckpt')])
            self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
            summary = json.loads((root/'report/summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['model']['requested'])
            self.assertFalse(summary['model']['checkpoint_loaded'])
            self.assertIn('FileNotFoundError', summary['model']['error'])
            self.assertEqual(summary['valid_rule_window_count'], 31)

    def test_no_windows_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, _ = episode_and_window()
            source = data.write_episode(episode, root/'input')
            result = self.run_cli(source, root/'report', ['--history', '1000'])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('No complete windows', result.stderr)


if __name__ == '__main__':
    unittest.main()
