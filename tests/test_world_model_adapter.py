"""CPU contracts; the small test backend is not an RSSM validation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ENV_DIR = Path(__file__).resolve().parents[1] / 'embodied/envs'
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))
import scenario_data as data
try:
    import world_model_adapter as adapter
    import rssm_inference as inference
except ImportError:
    adapter = inference = None


def episode_and_window():
    frames = []
    for i in range(65):
        frame = {'sim_time': .2 * i, 'state': [float(i)] * 31, 'cost': 0.,
                 'vehicles': [{'id': 'Auto', 'pos': [i*.4, 0.], 'yaw': 0.,
                               'v': 2., 'a': 0., 'length': 4., 'width': 2.}]}
        if i:
            frame['action'] = [-.5 + i * .001]
        frames.append(frame)
    episode = data.episode_from_record({'episode_id': 'e', 'scenario_id': 's',
        'metadata': {'position_reference': 'center'}, 'replay_trajectory': {'frames': frames}})
    episode.metadata['position_reference'] = 'center'
    return episode, data.make_windows(episode, history=5, future=30, dt=.2)[0]


class CPUSequenceBackend:
    """Test-only observation sequence; predictions are hand-controlled."""
    model_version = 'test_backend_not_rssm'

    def encode(self, observations, incoming_actions, seed):
        return {'last': observations['state'][0, -1].copy(),
                'incoming': incoming_actions.copy()}

    def rollout(self, latent, actions, seed):
        length = actions.shape[1]
        return {'state': latent['last'][None] + np.arange(1, length+1)[:, None],
                'cost': np.maximum(actions[0, :, 0], 0), 'reward': actions[0, :, 0].copy()}

    def teacher_force(self, latent, observations, incoming_actions, seed):
        return {'state': observations['state'][0].copy(), 'cost': np.zeros(incoming_actions.shape[1])}


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(adapter, 'stage 2 model adapter is missing')
        self.episode, self.window = episode_and_window()

    def test_no_backend_is_explicit_and_rule_fallback_still_works(self):
        model = adapter.WorldModelAdapter(self.episode)
        self.assertFalse(model.encode_history(self.window)['supported'])
        self.assertFalse(model.predict_future(self.window)['supported'])
        report = model.evaluate_risk(self.window)
        self.assertEqual(report.prediction_source, 'measured_trajectory_rule')
        self.assertIsNone(report.predicted_collision_probability)

    def test_npc_or_policy_condition_is_rejected_before_inference(self):
        model = adapter.WorldModelAdapter(self.episode, CPUSequenceBackend())
        result = model.predict_future(self.window, npc_plan=np.zeros((30, 0, 8)))
        self.assertFalse(result['supported'])
        self.assertEqual(result['reason'], 'npc_plan_not_supported')
        self.assertFalse(model.predict_future(self.window, ego_policy=lambda x: 0.)['supported'])

    def test_native_grid_and_incoming_outgoing_actions_align(self):
        model = adapter.WorldModelAdapter(self.episode, CPUSequenceBackend(), seed=7)
        encoded = model.encode_history(self.window)
        self.assertTrue(encoded['supported'])
        np.testing.assert_allclose(encoded['latent']['incoming'][0, :, 0],
                                   [0., -.499, -.498, -.497, -.496])
        result = model.predict_future(self.window, horizon=5)
        self.assertTrue(result['supported'])
        # Anchor index 4; incoming action of frame 5 executes [4,5].
        np.testing.assert_allclose(result['free_rollout']['reward'], [-.495, -.494, -.493, -.492, -.491])
        np.testing.assert_allclose(result['free_rollout']['state'][:, 0], [5., 6., 7., 8., 9.])
        self.assertEqual(result['metrics']['free_rollout']['steps']['5']['state_rmse_at_step'], 0.)
        self.assertIsNone(result['metrics']['free_rollout']['ADE'])
        self.assertIsNone(result['metrics']['free_rollout']['risk_recall'])
        self.assertIsNone(result['uncertainty'])

    def test_resampled_neighbor_slots_are_not_fed_to_model(self):
        resampled = data.make_windows(self.episode, history=5, future=30, dt=.1)[0]
        result = adapter.WorldModelAdapter(self.episode, CPUSequenceBackend()).predict_future(resampled)
        self.assertFalse(result['supported'])
        self.assertEqual(result['reason'], 'model_dt_mismatch')

    def test_raw_state_and_action_missing_are_not_imputed(self):
        model = adapter.WorldModelAdapter(self.episode, CPUSequenceBackend())
        self.episode.raw_state_mask[2] = False
        self.assertFalse(model.encode_history(self.window)['supported'])
        self.episode.raw_state_mask[2] = True
        self.episode.ego_action_mask[4] = False
        self.assertFalse(model.predict_future(self.window)['supported'])

    def test_window_identity_and_history_cold_start(self):
        later = data.make_windows(self.episode, history=5, future=30, dt=.2)[10]
        result = adapter.WorldModelAdapter(self.episode, CPUSequenceBackend()).predict_future(later, horizon=1)
        self.assertTrue(result['supported'])
        self.assertTrue(result['cold_start'])
        self.assertEqual(result['free_rollout']['state'][0, 0], 15.)
        other = copy.copy(self.window)
        other.episode_id = 'other'
        self.assertFalse(adapter.WorldModelAdapter(self.episode, CPUSequenceBackend()).encode_history(other)['supported'])

    def test_known_terminal_alignment_is_excluded_from_ground_truth(self):
        self.episode.metadata['state_alignment'] = 'legacy_observation; terminal_state_may_precede_snapshot'
        final = data.make_windows(self.episode, history=5, future=30, dt=.2)[-1]
        result = adapter.WorldModelAdapter(self.episode, CPUSequenceBackend()).predict_future(final)
        self.assertFalse(result['ground_truth_mask'][-1])
        self.assertIsNone(result['metrics']['free_rollout']['steps']['30']['state_rmse_at_step'])

    def test_no_decoder_reports_unavailable_without_claiming_positions(self):
        class CostOnly(CPUSequenceBackend):
            def rollout(self, latent, actions, seed):
                return {'cost': np.zeros(actions.shape[1])}
            def teacher_force(self, latent, observations, incoming_actions, seed):
                return {'cost': np.zeros(incoming_actions.shape[1])}
        result = adapter.WorldModelAdapter(self.episode, CostOnly()).predict_future(self.window)
        self.assertTrue(result['supported'])
        self.assertFalse(result['capabilities']['world_vehicle_decoder'])
        self.assertIsNone(result['metrics']['free_rollout']['steps']['1']['state_rmse_at_step'])

    def test_prediction_metrics_error_and_cost_recall_are_not_collision_probability(self):
        outputs = {'state': np.ones((30, 31)) * 2, 'cost': np.r_[1., np.zeros(29)]}
        report = adapter.prediction_metrics(outputs, np.zeros((30, 31)), np.ones(30, bool),
                                             recorded_cost=np.r_[1., 1., np.zeros(28)])
        for step in ('1', '5', '10', '30'):
            self.assertEqual(report['steps'][step]['state_rmse_at_step'], 2.)
        self.assertEqual(report['risk_recall'], .5)
        self.assertIn('recorded_cost', report['risk_recall_definition'])
        self.assertIsNone(report['FDE'])
        json.dumps(report, allow_nan=False)

    def test_seeded_predictions_reproduce_without_altering_episode(self):
        before = self.episode.raw_states.copy()
        model = adapter.WorldModelAdapter(self.episode, CPUSequenceBackend(), seed=123)
        for _ in range(2):
            prediction = model.predict_future(self.window)
            np.testing.assert_array_equal(prediction['free_rollout']['state'], self.episode.raw_states[5:35])
        np.testing.assert_array_equal(before, self.episode.raw_states)

    def test_nonfinite_or_truncated_model_outputs_do_not_count_as_supported(self):
        for outputs in ({'state': np.full((30, 31), np.nan)},
                        {'cost': np.full(30, np.inf)},
                        {'state': np.zeros((1, 31))}, {}):
            class Invalid(CPUSequenceBackend):
                def rollout(self, latent, actions, seed):
                    return outputs
            result = adapter.WorldModelAdapter(self.episode, Invalid()).predict_future(self.window)
            self.assertFalse(result['supported'], 'Invalid model outputs cannot verify inference')

    def test_nonfinite_posterior_cannot_start_rollout(self):
        class InvalidPosterior(CPUSequenceBackend):
            def encode(self, observations, incoming_actions, seed):
                return {'deter': np.full((1, 10), np.nan)}
        result = adapter.WorldModelAdapter(self.episode, InvalidPosterior()).encode_history(self.window)
        self.assertFalse(result['supported'])

    def test_original_checkpoint_container_reads_only_agent(self):
        import embodied
        class Entry:
            def __init__(self, value): self.value = value
            def save(self): return self.value
            def load(self, value): self.value = value
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'checkpoint.ckpt'
            original = embodied.Checkpoint(str(filename), log=False, parallel=False)
            original.agent = Entry({'agent/wm/enc/weight': np.arange(3)})
            original.step = Entry(400)
            original.replay = Entry('must not restore replay')
            original.save()
            target = Entry(None)
            inference.load_original_checkpoint(target, filename)
            np.testing.assert_array_equal(target.value['agent/wm/enc/weight'], [0, 1, 2])

    def test_missing_checkpoint_fails_before_importing_jax(self):
        with self.assertRaises(FileNotFoundError):
            inference.NinjaxRSSMBackend.from_checkpoint('not_present/checkpoint.ckpt')


if __name__ == '__main__':
    unittest.main()
