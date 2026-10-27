from . import generic
from . import selectors
from . import limiters

import random
import numpy as np


class Uniform(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, online=False, chunks=1024,
      min_size=1, samples_per_insert=None, tolerance=1e4, seed=0):
    if samples_per_insert:
      limiter = limiters.SamplesPerInsert(
          samples_per_insert, tolerance, min_size)
    else:
      limiter = limiters.MinSize(min_size)
    assert not capacity or min_size <= capacity
    super().__init__(
        length=length,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiter,
        directory=directory,
        online=online,
        chunks=chunks,
    )


class Danger(Uniform):
  """Uniform replay with a controlled fraction of high-cost sequences.

  It is intentionally episode-agnostic: a sequence is hard when any of its
  transitions contains a cost above ``cost_threshold``.  This lets dense
  near-miss risk and sparse collision costs both receive more world-model and
  actor-critic updates without changing their targets or reward definitions.
  """

  def __init__(
      self, length, capacity=None, directory=None, online=False, chunks=1024,
      min_size=1, samples_per_insert=None, tolerance=1e4, seed=0,
      danger_ratio=0.5, cost_threshold=0.1):
    self.danger_ratio = float(np.clip(danger_ratio, 0.0, 1.0))
    self.cost_threshold = float(cost_threshold)
    self._danger_keys = set()
    self._danger_rng = random.Random(seed)
    self.danger_sampling_enabled = False
    self._danger_sample_calls = 0
    self._danger_samples = 0
    super().__init__(
        length=length,
        capacity=capacity,
        directory=directory,
        online=online,
        chunks=chunks,
        min_size=min_size,
        samples_per_insert=samples_per_insert,
        tolerance=tolerance,
        seed=seed,
    )

  @property
  def stats(self):
    stats = super().stats
    stats.update({
        'danger_sequences': len(self._danger_keys),
        'danger_sample_fraction': (
            self._danger_samples / self._danger_sample_calls
            if self._danger_sample_calls else 0.0),
    })
    self._danger_sample_calls = 0
    self._danger_samples = 0
    return stats

  def set_danger_sampling(self, active):
    """Switch hard-sequence oversampling on only for post-training."""
    self.danger_sampling_enabled = bool(active)
    return self.danger_sampling_enabled

  def _on_insert(self, key, seq):
    costs = []
    for step in seq:
      if 'cost' not in step:
        continue
      value = np.asarray(step['cost'], dtype=np.float32)
      if value.size:
        costs.append(float(np.nanmax(value)))
    if costs and max(costs) >= self.cost_threshold:
      self._danger_keys.add(key)

  def _on_remove(self, key, seq):
    self._danger_keys.discard(key)

  def _sample_sequence(self):
    self._danger_sample_calls += 1
    # Keep online samples fresh; the hard-sample mixture only applies to the
    # persistent table after online replay has no ready sequence.
    if self.danger_sampling_enabled and not self.online and self._danger_keys and (
        self._danger_rng.random() < self.danger_ratio):
      key = self._danger_rng.choice(tuple(self._danger_keys))
      self._danger_samples += 1
      return self.table[key]
    return super()._sample_sequence()
