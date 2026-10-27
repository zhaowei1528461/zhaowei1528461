"""Hand-derived geometry cases; independent from JAX and a running SUMO."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ENV_DIR = Path(__file__).resolve().parents[1] / 'embodied/envs'
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))
try:
    import scenario_risk as risk
except ImportError:
    risk = None


def vehicle(x=0., y=0., vx=0., vy=0., yaw=0., length=4., width=2.):
    return np.array([x, y, vx, vy, yaw, 0., length, width])


def window(ego=None, npc=None, steps=31, dt=.2, reference='center'):
    ego = vehicle(vx=2.) if ego is None else ego
    t = np.arange(steps) * dt
    egos = np.tile(ego, (steps, 1))
    egos[:, :2] += t[:, None] * ego[2:4]
    npcs = np.empty((steps, 0, 8)) if npc is None else np.tile(npc, (steps, 1, 1))
    if npc is not None:
        npcs[:, :, :2] += t[:, None, None] * npc[2:4]
    return SimpleNamespace(history_ego=egos[:1], future_ego=egos[1:],
        history_npcs=npcs[:1], future_npcs=npcs[1:], timestamps=t,
        anchor_timestamp=0., dt=dt, ego_mask=np.ones(steps, bool),
        npc_mask=np.ones(npcs.shape[:2], bool), npc_ids=() if npc is None else ('npc',),
        normalizer_id=None, coordinate_frame='world', validity_flags={},
        metadata={'position_reference': reference}, event_labels=(),
        scenario_id='s', episode_id='e', seed=1)


class ScenarioRiskTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(risk, 'stage 2 rule module is missing')

    def test_footprint_distance_and_closing_time(self):
        # Centers 10 m apart, each body length 4: clearance 6 m.
        a, b = vehicle(vx=2.), vehicle(x=10.)
        self.assertAlmostEqual(risk.obb_distance(a, b, 'center'), 6.)
        self.assertAlmostEqual(risk.obb_ttc(a, b, 'center', horizon=5.), 3.)

    def test_front_bumper_center_is_converted_per_vehicle_length(self):
        # Bodies [-4,0] and [4,10]: clearance is 4 m, not the center gap 9 m.
        a, b = vehicle(), vehicle(x=10., length=6.)
        self.assertAlmostEqual(risk.obb_distance(a, b, 'front_bumper_center'), 4.)

    def test_adjacent_lane_and_receding_have_no_ttc(self):
        for npc in (vehicle(x=10., y=4.), vehicle(x=10., vx=3.)):
            self.assertIsNone(risk.obb_ttc(vehicle(vx=2.), npc, 'center', horizon=20.))

    def test_crossing_fixed_heading_footprints_can_conflict(self):
        # Ego translates east, vertical NPC translates south, contact at 3.5 s.
        self.assertAlmostEqual(risk.obb_ttc(vehicle(x=-10., vx=2.),
            vehicle(y=10., vy=-2., yaw=-math.pi/2), 'center', horizon=10.), 3.5)

    def test_static_and_overlap_are_stable(self):
        self.assertIsNone(risk.obb_ttc(vehicle(), vehicle(x=10.), 'center'))
        self.assertEqual(risk.obb_ttc(vehicle(), vehicle(x=1.), 'center'), 0.)
        report = risk.RiskEvaluator().evaluate_risk(window(vehicle(), vehicle(x=1.)))
        self.assertEqual(report.min_distance, 0.)
        self.assertTrue(report.validity_flags['geometric_overlap'])
        self.assertIsNone(report.predicted_collision_probability)
        self.assertIsNone(report.uncertainty)
        json.dumps(report.to_dict(), allow_nan=False)

    def test_rigid_rotation_preserves_distance_and_ttc(self):
        a, b = vehicle(vx=2.), vehicle(x=10.)
        angle = .7
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        for state in (a, b):
            state[:2] = rotation.dot(state[:2]) + [200., -50.]
            state[2:4] = rotation.dot(state[2:4])
            state[4] += angle
        self.assertAlmostEqual(risk.obb_distance(a, b, 'center'), 6.)
        self.assertAlmostEqual(risk.obb_ttc(a, b, 'center'), 3.)

    def test_no_npc_is_distinct_from_missing_npc(self):
        report = risk.RiskEvaluator().evaluate_risk(window())
        self.assertTrue(report.validity_flags['valid'])
        self.assertEqual(report.score, 0.)
        self.assertIsNone(report.min_ttc)
        w = window(npc=vehicle(x=10.))
        w.npc_mask[:] = False
        report = risk.RiskEvaluator().evaluate_risk(w)
        self.assertFalse(report.validity_flags['valid'])
        self.assertIsNone(report.score)

    def test_anchor_only_npc_coverage_cannot_certify_future_safe(self):
        w = window(npc=vehicle(x=50., vx=2.))
        w.npc_mask[1:] = False
        report = risk.RiskEvaluator().evaluate_risk(w)
        self.assertFalse(report.validity_flags['valid'])
        self.assertIsNone(report.score)
        self.assertFalse(report.validity_flags['npc_observation_each_frame'])

    def test_candidate_cannot_hide_observed_vehicles_with_mask(self):
        w = window(npc=vehicle(x=1.))
        candidate = risk.CandidateTrajectory(w.future_npcs.copy(), np.zeros((30, 1), bool),
                                             w.npc_ids, w.dt, w.coordinate_frame)
        with self.assertRaises(ValueError):
            risk.RiskEvaluator().evaluate_risk(w, candidate)

    def test_internal_npc_capture_gap_is_not_hidden_by_another_visible_npc(self):
        w = window(npc=vehicle(x=50., vx=2.))
        w.npc_ids = ('one', 'two')
        w.history_npcs = np.repeat(w.history_npcs, 2, axis=1)
        w.future_npcs = np.repeat(w.future_npcs, 2, axis=1)
        w.npc_mask = np.ones((31, 2), bool)
        w.npc_mask[1:-1, 1] = False
        report = risk.RiskEvaluator().evaluate_risk(w)
        self.assertFalse(report.validity_flags['valid'])
        self.assertIsNone(report.score)
        self.assertTrue(report.validity_flags['npc_internal_observation_gap'])

    def test_unknown_reference_and_normalized_states_cannot_be_ranked(self):
        for w in (window(reference=None), window()):
            if w.metadata['position_reference'] is not None:
                w.normalizer_id = 'normalizer-1'
            report = risk.RiskEvaluator().evaluate_risk(w)
            self.assertFalse(report.validity_flags['valid'])
            self.assertIsNone(report.score)

    def test_valid_mask_with_nonfinite_value_is_flagged(self):
        w = window(npc=vehicle(x=10.))
        w.future_npcs[0, 0, 0] = np.nan
        report = risk.RiskEvaluator().evaluate_risk(w)
        self.assertFalse(report.validity_flags['valid'])
        self.assertIsNone(report.score)

    def test_candidate_is_fixed_ego_and_must_match_window(self):
        w = window(npc=vehicle(x=50.))
        candidate = risk.CandidateTrajectory(np.tile(vehicle(x=1.), (30, 1, 1)),
            np.ones((30, 1), bool), w.npc_ids, w.dt, w.coordinate_frame)
        report = risk.RiskEvaluator().evaluate_risk(w, candidate)
        self.assertEqual(report.prediction_source, 'fixed_ego_candidate_rule')
        self.assertFalse(report.validity_flags['counterfactual_closed_loop'])
        self.assertTrue(report.validity_flags['geometric_overlap'])
        candidate.dt = .1
        with self.assertRaises(ValueError):
            risk.RiskEvaluator().evaluate_risk(w, candidate)

    def test_progress_requires_duration_and_uses_center_path(self):
        evaluator = risk.RiskEvaluator()
        self.assertIsNone(evaluator.evaluate_risk(window(vehicle(), steps=5)).progress_penalty)
        self.assertEqual(evaluator.evaluate_risk(window(vehicle())).progress_penalty, 1.)
        self.assertEqual(evaluator.evaluate_risk(window()).progress_penalty, 0.)

    def test_weights_thresholds_and_shapes_are_validated(self):
        for kwargs in ({'ttc_horizon': 0.}, {'weights': {'ttc': -1.}}, {'near_distance': float('nan')}):
            with self.assertRaises(ValueError):
                risk.RiskConfig(**kwargs)
        w = window(npc=vehicle(x=10.))
        w.npc_mask = np.ones((2, 1), bool)
        with self.assertRaises(ValueError):
            risk.RiskEvaluator().evaluate_risk(w)


if __name__ == '__main__':
    unittest.main()
