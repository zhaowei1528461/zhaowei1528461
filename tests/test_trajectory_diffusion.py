"""Real CPU tensor tests for direct-trajectory denoising and sampling."""
import importlib
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'embodied/envs'))
try:
    import torch
except ImportError:
    torch = None
try:
    subject = importlib.import_module('trajectory_diffusion')
except ImportError:
    subject = None


def normalizer_fixture():
    return {'mean': [0.]*8, 'std': [1.]*8, 'coordinate_frame': 'ego-centric',
            'normalizer_id': 'synthetic-identity', 'training_episode_ids': ['synthetic-train'], 'count': 1}


def config_fixture():
    return {'history': 3, 'future': 30, 'dt': .2, 'base_dim': 8, 'dim_mults': [1, 2],
            'condition_dim': 16, 'diffusion_steps': 8, 'beta_schedule': 'cosine'}


def condition_fixture(batch=2, history=3, neighbors=2):
    return {'ego_history': torch.randn(batch, history, 8), 'ego_mask': torch.ones(batch, history, dtype=torch.bool),
            'target_history': torch.randn(batch, history, 8), 'target_mask': torch.ones(batch, history, dtype=torch.bool),
            'neighbor_history': torch.randn(batch, history, neighbors, 8),
            'neighbor_mask': torch.ones(batch, history, neighbors, dtype=torch.bool),
            'map_segments': torch.zeros(batch, 4, 6), 'map_mask': torch.zeros(batch, 4, dtype=torch.bool),
            'scene_type': torch.zeros(batch, dtype=torch.long)}


@unittest.skipIf(torch is None, 'PyTorch runtime required; NumPy data tests remain available')
class TrajectoryDiffusionTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(subject, 'Direct trajectory diffusion module is missing')
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_thirty_frame_forward_backward_and_real_optimizer_update(self):
        model = subject.make_model(config_fixture(), normalizer_fixture())
        condition = condition_fixture()
        future = torch.randn(2, 30, 2)
        prediction = model.denoiser(torch.randn_like(future), torch.tensor([0, 7]), condition)
        self.assertEqual(tuple(prediction.shape), (2, 30, 2))
        loss = model.training_loss(condition, future)
        self.assertTrue(torch.isfinite(loss))
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        optimizer.step()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_masked_nan_neighbors_and_missing_maps_do_not_contaminate_output(self):
        model = subject.make_model(config_fixture(), normalizer_fixture())
        condition = condition_fixture()
        condition['neighbor_mask'][:] = False
        condition['neighbor_history'][:] = float('nan')
        condition['map_segments'][:] = float('nan')
        result = model.denoiser(torch.zeros(2, 30, 2), torch.zeros(2, dtype=torch.long), condition)
        self.assertTrue(torch.isfinite(result).all())
        condition['neighbor_history'] = torch.zeros(2, 3, 0, 8)
        condition['neighbor_mask'] = torch.zeros(2, 3, 0, dtype=torch.bool)
        self.assertTrue(torch.isfinite(model.training_loss(condition, torch.zeros(2, 30, 2))))

    def test_q_sample_and_posterior_use_the_same_zero_based_schedule(self):
        config = dict(config_fixture(), diffusion_steps=4, beta_schedule='linear', beta_start=.1, beta_end=.4)
        model = subject.make_model(config, normalizer_fixture())
        x = torch.ones(2, 30, 2)
        noisy = model.q_sample(x, torch.tensor([0, 3]), noise=torch.zeros_like(x))
        self.assertAlmostEqual(noisy[0, 0, 0].item(), math.sqrt(.9), places=6)
        self.assertAlmostEqual(noisy[1, 0, 0].item(), math.sqrt(.3024), places=6)
        mean, variance = model.q_posterior(x, torch.zeros_like(x), torch.tensor([0, 1]))
        self.assertAlmostEqual(mean[0, 0, 0].item(), 1., places=6)
        self.assertAlmostEqual(variance[0, 0, 0].item(), 0., places=7)
        self.assertAlmostEqual(mean[1, 0, 0].item(), .6776309271789387, places=6)
        self.assertAlmostEqual(variance[1, 0, 0].item(), .07142857142857144, places=6)

    def test_seeded_ddpm_and_actual_ddim_keep_batch_sample_and_time_axes(self):
        model = subject.make_model(config_fixture(), normalizer_fixture())
        condition = condition_fixture()
        for method, steps in (('ddpm', 8), ('ddim', 4), ('ddim', 1)):
            a = model.sample(condition, num_samples=3, method=method, steps=steps, seed=12)
            b = model.sample(condition, num_samples=3, method=method, steps=steps, seed=12)
            self.assertEqual(tuple(a.shape), (2, 3, 30, 2))
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(torch.equal(a[:, 0], a[:, 1]))
        self.assertTrue(model.training)

    def test_masked_future_nan_values_do_not_enter_noising_or_loss(self):
        model = subject.make_model(config_fixture(), normalizer_fixture())
        condition = condition_fixture()
        labels = torch.zeros(2, 30, 2)
        mask = torch.ones(2, 30, dtype=torch.bool)
        mask[:, -1] = False
        t, noise = torch.tensor([1, 4]), torch.zeros_like(labels)
        first = model.training_loss(condition, labels, mask, t=t, noise=noise)
        labels[:, -1] = float('nan')
        second = model.training_loss(condition, labels, mask, t=t, noise=noise)
        torch.testing.assert_close(first, second)
        mask[:] = False
        with self.assertRaisesRegex(ValueError, 'valid|mask'):
            model.training_loss(condition, labels, mask)

    def test_invalid_sampling_and_active_nonfinite_conditions_are_explicit_errors(self):
        model = subject.make_model(config_fixture(), normalizer_fixture())
        condition = condition_fixture()
        with self.assertRaises(ValueError):
            model.sample(condition, num_samples=0)
        with self.assertRaises(ValueError):
            model.sample(condition, method='ddpm', steps=4)
        condition['target_history'][0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite|Nonfinite'):
            model.training_loss(condition, torch.zeros(2, 30, 2))

    def test_model_state_round_trip_reproduces_seeded_samples(self):
        model = subject.make_model(config_fixture(), normalizer_fixture())
        clone = subject.make_model(config_fixture(), normalizer_fixture())
        clone.load_state_dict(model.state_dict(), strict=True)
        condition = condition_fixture()
        torch.testing.assert_close(model.sample(condition, seed=3, steps=4),
                                   clone.sample(condition, seed=3, steps=4), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
