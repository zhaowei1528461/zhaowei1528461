"""Physical regressions for NPCs that approach on a different heading."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'embodied/envs'))
from test_scenario_risk import vehicle, window
from scene_parameters import DynamicsLimits, SceneParameterSpec, SceneGenerator
from scene_road import RoadEnvelope
from scene_search import SceneEvaluator


def angled_window():
    # Ego moves east at 3 m/s; NPC moves at 3 m/s on a 60-degree heading.
    w = window(vehicle(vx=3.), vehicle(x=15., y=3.5, vx=1.5,
               vy=1.5*math.sqrt(3.), yaw=math.pi/3))
    w.source_type = 'synthetic'
    w.source_path = None
    return w


def steady_parameters():
    # Ego projection on the target heading is 1.5 m/s; NPC initial speed is 2.
    return dict(initial_gap=10., relative_speed=.5, trigger_time=.5,
                longitudinal_accel=0., target_offset=0., lane_change_duration=4.)


class ReferenceHeadingTest(unittest.TestCase):
    def make_generator(self, w=None, heading='target'):
        w = angled_window() if w is None else w
        limits = DynamicsLimits(4., .2, 4.)
        spec = SceneParameterSpec.from_window(w, 0, limits, reference_heading=heading)
        return SceneGenerator(w, 0, spec)

    def test_keep_lane_preserves_target_heading_instead_of_turning_toward_ego(self):
        generator = self.make_generator()
        candidate = generator.generate(steady_parameters(), 'keep_lane')
        expected_velocity = np.tile([1., math.sqrt(3.)], (31, 1))
        np.testing.assert_allclose(candidate.states[:, 2:4], expected_velocity, atol=1e-10)
        np.testing.assert_allclose(candidate.states[:, 4], math.pi/3, atol=1e-10)
        np.testing.assert_allclose(np.diff(candidate.states[:, :2], axis=0),
                                   expected_velocity[:-1]*.2, atol=1e-10)
        np.testing.assert_allclose(candidate.accelerations, 0., atol=1e-10)
        evaluator = SceneEvaluator(generator.spec.limits, RoadEnvelope.rectangle([-100., -100., 100., 100.]))
        result = evaluator.evaluate(candidate)
        self.assertTrue(result.valid, result.failures)

    def test_speed_bounds_and_generator_use_the_same_target_axis(self):
        generator = self.make_generator()
        low, high = generator.spec.bounds['relative_speed']
        self.assertAlmostEqual(generator.ego_speed, 1.5)
        self.assertAlmostEqual(low, -1.3)
        self.assertAlmostEqual(high, 2.)
        p = steady_parameters()
        p['relative_speed'] = low
        candidate = generator.generate(p, 'keep_lane')
        np.testing.assert_allclose(np.linalg.norm(candidate.states[:, 2:4], axis=-1), .2, atol=1e-10)

    def test_frame_selection_does_not_rewrite_observed_ego_or_source_npcs(self):
        w = angled_window()
        original = copy.deepcopy(w)
        candidate = self.make_generator(w).generate(steady_parameters(), 'keep_lane')
        for name in ('history_ego', 'future_ego', 'history_npcs', 'future_npcs', 'npc_mask'):
            np.testing.assert_array_equal(getattr(w, name), getattr(original, name))
        np.testing.assert_array_equal(candidate.context.history_ego, original.history_ego)
        np.testing.assert_array_equal(candidate.context.future_ego, original.future_ego)
        self.assertTrue(candidate.context.metadata['generated_target_history'])

    def test_target_mode_is_invariant_to_a_rigid_world_rotation(self):
        w = angled_window()
        first = self.make_generator(w).generate(steady_parameters(), 'keep_lane')
        transformed = copy.deepcopy(w)
        angle = .7
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        for name in ('history_ego', 'future_ego', 'history_npcs', 'future_npcs'):
            states = getattr(transformed, name)
            states[..., :2] = states[..., :2] @ rotation.T + [50., -20.]
            states[..., 2:4] = states[..., 2:4] @ rotation.T
            states[..., 4] += angle
        second = self.make_generator(transformed).generate(steady_parameters(), 'keep_lane')
        np.testing.assert_allclose(second.states[:, :2], first.states[:, :2] @ rotation.T + [50., -20.], atol=1e-10)
        np.testing.assert_allclose(second.states[:, 2:4], first.states[:, 2:4] @ rotation.T, atol=1e-10)

    def test_default_mode_preserves_explicit_legacy_ego_trajectories(self):
        w = angled_window()
        limits = DynamicsLimits(10., 2.6, 4.5)
        default = SceneGenerator(w, 0, SceneParameterSpec.from_window(w, 0, limits))
        explicit = SceneGenerator(w, 0, SceneParameterSpec.from_window(w, 0, limits, reference_heading='ego'))
        p = steady_parameters()
        first, second = default.generate(p, 'keep_lane'), explicit.generate(p, 'keep_lane')
        np.testing.assert_array_equal(first.states, second.states)
        self.assertEqual(first.candidate_id, second.candidate_id)

    def test_unknown_heading_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reference_heading'):
            self.make_generator(heading='nearest_road_guess')

    def test_generator_rejects_nonworld_input_before_producing_world_provenance(self):
        w = angled_window()
        w.coordinate_frame = 'ego-centric'
        with self.assertRaisesRegex(ValueError, 'world'):
            self.make_generator(w)

    def test_acceleration_failure_reports_actual_extrema(self):
        from test_scene_generation import make_generator, parameters
        generator, road = make_generator()
        p = parameters()
        p['longitudinal_accel'] = .5
        candidate = generator.generate(p, 'keep_lane')
        limits = DynamicsLimits(10., .2, 4.5)
        result = SceneEvaluator(limits, road).evaluate(candidate)
        self.assertIn('longitudinal_acceleration_limit', result.failures)
        self.assertIn('max_longitudinal_accel', result.metrics)
        self.assertAlmostEqual(result.metrics['max_longitudinal_accel'], .5)
        self.assertAlmostEqual(result.metrics['min_longitudinal_accel'], 0.)

    def test_cli_records_reference_frame_in_summary_and_exported_json_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'target_mode'
            result = subprocess.run([sys.executable, str(ROOT/'embodied/envs/run_scene_search.py'),
                '--demo', '--reference-heading', 'target', '--budget', '32', '--seed', '0',
                '--top-k', '2', '--output', str(output)], cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['parameter_frame']['heading_source'], 'target')
            for method in ('random', 'cem'):
                self.assertEqual(summary['searches'][method]['evaluations'], 32)
                scene = json.loads((output/method/'top_001.json').read_text())
                self.assertEqual(scene['parameter_frame']['heading_source'], 'target')
                self.assertFalse(scene['simulation_verified'])
                with np.load(output/method/'top_001.npz', allow_pickle=False) as archive:
                    metadata = json.loads(str(archive['metadata_json'].item()))
                    self.assertEqual(metadata['parameter_frame'], scene['parameter_frame'])


if __name__ == '__main__':
    unittest.main()
