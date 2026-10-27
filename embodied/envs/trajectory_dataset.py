"""NumPy-only frozen-window loader; condition dictionaries contain history only."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

try:
    from .scenario_data import ScenarioNormalizer, VEHICLE_FIELDS, _resample_states, to_ego_centric
    from .prepare_trajectory_windows import FIT_SCOPE, _hash, _read_manifest, _load_checked, _check_unchanged
    from .trajectory_maps import SumoMapStore
except ImportError:
    from scenario_data import ScenarioNormalizer, VEHICLE_FIELDS, _resample_states, to_ego_centric
    from prepare_trajectory_windows import FIT_SCOPE, _hash, _read_manifest, _load_checked, _check_unchanged
    from trajectory_maps import SumoMapStore


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class TrajectoryDataset:
    def __init__(self, dataset_manifest, split, input_root=None, split_manifest=None,
                 max_neighbors=8, cache_size=2, limit=None, seed=0, map_root=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown dataset split')
        if max_neighbors < 0 or cache_size < 1 or (limit is not None and limit < 1):
            raise ValueError('Invalid neighbors/cache/limit')
        self.manifest_path = Path(dataset_manifest).resolve()
        self.manifest = _json(self.manifest_path)
        payload = self.manifest
        if (payload.get('artifact_type') != 'trajectory_window_dataset' or payload.get('schema_version') != 1
                or payload.get('preparation_exit_code') != 0 or payload.get('coordinate_frame') != 'ego-centric'
                or tuple(payload.get('vehicle_fields', ())) != VEHICLE_FIELDS):
            raise ValueError('Expected a successful ego-centric trajectory window dataset')
        self.config = payload['window_config']
        for key in ('history', 'future', 'stride'):
            if isinstance(self.config[key], bool) or not isinstance(self.config[key], int) or self.config[key] < 1:
                raise ValueError('Invalid window configuration')
        if not np.isfinite(self.config['dt']) or self.config['dt'] <= 0:
            raise ValueError('Invalid dt')
        frozen_path = Path(split_manifest or payload['source_manifest_path']).resolve()
        if _hash(frozen_path) != payload['source_manifest_sha256']:
            raise ValueError('Frozen split manifest hash mismatch; supply the original manifest')
        _, entries, root, _, self.audit = _read_manifest(frozen_path, input_root)
        self.entries = {e['episode_id']: e for e in entries}
        self.input_root = root
        self.index_path = self.manifest_path.parent/'window_index.jsonl'
        if not self.index_path.is_file():
            self.index_path = Path(payload['index_path'])
        if _hash(self.index_path) != payload['index_sha256']:
            raise ValueError('Window index hash mismatch')
        index_stat = self.index_path.stat()
        self._index_signature = (index_stat.st_size, index_stat.st_mtime_ns, index_stat.st_ino)
        normalizer_path = self.manifest_path.parent/'normalizer.json'
        if not normalizer_path.is_file():
            normalizer_path = Path(payload['normalizer']['path'])
        self.normalizer = ScenarioNormalizer.load(normalizer_path)
        normalizer = self.normalizer
        train_ids = set(e['episode_id'] for e in entries if e['split'] == 'train')
        if (not payload['normalizer']['fitted'] or normalizer.coordinate_frame != 'ego-centric'
                or normalizer.normalizer_id != payload['normalizer']['normalizer_id']
                or set(normalizer.training_episode_ids) != set(payload['normalizer']['training_episode_ids'])
                or not set(normalizer.training_episode_ids) <= train_ids
                or normalizer.count != payload['normalizer']['training_state_count']):
            raise ValueError('Normalizer identity/training population mismatch')
        identity = json.dumps({'config': self.config, 'coordinate_frame': 'ego-centric', 'fit_scope': FIT_SCOPE,
                   'mean': normalizer.mean.tolist(), 'std': normalizer.std.tolist(), 'count': normalizer.count,
                   'training_sources': sorted((i, self.entries[i]['input_sha256']) for i in normalizer.training_episode_ids)},
                   sort_keys=True, allow_nan=False)
        expected = 'normalizer-'+hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]
        if normalizer.normalizer_id != expected:
            raise ValueError('Normalizer parameters do not match their frozen identity')
        self.split, self.max_neighbors, self.cache_size = split, int(max_neighbors), int(cache_size)
        self.seed, self.epoch = int(seed), 0
        self._cache, self._loads = OrderedDict(), 0
        self.maps = SumoMapStore(map_root)
        self.offsets, self.episode_indices = [], OrderedDict()
        observed = dict.fromkeys(('train', 'val', 'test'), 0)
        with self.index_path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                self._validate_row(row)
                observed[row['split']] += 1
                if row['split'] == split and (limit is None or len(self.offsets) < limit):
                    self.episode_indices.setdefault(row['episode_id'], []).append(len(self.offsets))
                    self.offsets.append(offset)
        if observed != payload['split_window_counts']:
            raise ValueError('Window index counts contradict its manifest')
        self._check_index()
        if _hash(self.index_path) != payload['index_sha256']:
            raise ValueError('Window index changed during construction')
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        self.episode_indices = {k: np.asarray(v, dtype=np.int64) for k, v in self.episode_indices.items()}
        self.fingerprint = {'window_manifest_sha256': _hash(self.manifest_path),
                            'split_manifest_sha256': payload['source_manifest_sha256'],
                            'index_sha256': payload['index_sha256'], 'normalizer_id': normalizer.normalizer_id,
                            'window_config': dict(self.config), 'coordinate_frame': 'ego-centric',
                            'map_binding_sha256': self.maps.binding_fingerprint(e.get('source_cfg') for e in entries)}

    def _validate_row(self, row):
        entry = self.entries.get(row.get('episode_id'))
        if entry is None:
            raise ValueError('Window index contains an unknown episode')
        for key in ('split', 'scenario_id', 'parent_group_id', 'relative_path', 'input_sha256'):
            if row.get(key) != entry[key]:
                raise ValueError('Window member contradicts frozen split: '+key)
        for key in ('window_index', 'start_grid_index'):
            if isinstance(row.get(key), bool) or not isinstance(row.get(key), int) or row[key] < 0:
                raise ValueError('Invalid window grid index')
        if row['start_grid_index'] != row['window_index']*self.config['stride']:
            raise ValueError('Window stride/grid identity mismatch')
        ids = row.get('eligible_target_ids')
        if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids) or not np.isfinite(row['anchor_timestamp'])):
            raise ValueError('Invalid complete target list or anchor')

    def __len__(self):
        return len(self.offsets)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache'] = OrderedDict()
        state['_loads'] = 0
        return state

    def cache_info(self):
        return {'episodes': len(self._cache), 'capacity': self.cache_size, 'loads': self._loads}

    def row(self, index):
        self._check_index()
        with self.index_path.open('rb') as stream:
            stream.seek(int(self.offsets[index]))
            row = json.loads(stream.readline())
        self._check_index()
        self._validate_row(row)
        if row['split'] != self.split:
            raise ValueError('Selected index row belongs to another split')
        return row

    def _check_index(self):
        stat = self.index_path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != self._index_signature:
            raise ValueError('Window index changed after its hash was verified')

    def _grid(self, episode_id):
        entry = self.entries[episode_id]
        if episode_id in self._cache:
            grid = self._cache.pop(episode_id)
            stat = Path(entry['resolved_source']).stat()
            if (stat.st_size, stat.st_mtime_ns) != grid.fingerprint:
                raise ValueError('Source changed after episode was cached')
            self._cache[episode_id] = grid
            return grid
        episode, fingerprint = _load_checked(entry)
        dt = self.config['dt']
        times = episode.timestamps
        if len(times) < 2:
            raise ValueError('Window source needs at least two timestamps')
        count = int(np.floor((times[-1]-times[0])/dt+1e-8))+1
        target = times[0]+np.arange(count)*dt
        max_gap = 1.5*float(np.median(np.diff(times)))
        ego, emask = _resample_states(times, episode.ego_state, episode.ego_valid_mask, target, max_gap)
        npcs, nmask = _resample_states(times, episode.npc_states, episode.npc_valid_mask, target, max_gap)
        grid = SimpleNamespace(times=target, ego=ego, ego_mask=emask, npcs=npcs, npc_mask=nmask,
                    npc_ids=episode.npc_ids, metadata=episode.metadata, source_cfg=episode.source_cfg,
                    fingerprint=fingerprint, scenario_id=episode.scenario_id)
        _check_unchanged(entry, fingerprint)
        self._cache[episode_id] = grid
        self._loads += 1
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return grid

    def __getitem__(self, index):
        row = self.row(index)
        grid = self._grid(row['episode_id'])
        h, f = self.config['history'], self.config['future']
        start, stop = row['start_grid_index'], row['start_grid_index']+h+f
        anchor_index = start+h-1
        if stop > len(grid.times) or not np.isclose(grid.times[anchor_index], row['anchor_timestamp'], rtol=0, atol=1e-7):
            raise ValueError('Window index/timestamp no longer matches source')
        anchor = grid.ego[anchor_index].copy()
        ego_history = to_ego_centric(grid.ego[start:start+h], anchor)
        states = to_ego_centric(grid.npcs[start:stop], anchor)
        masks = grid.npc_mask[start:stop] & np.isfinite(states).all(axis=-1)
        emask = grid.ego_mask[start:start+h] & np.isfinite(ego_history).all(axis=-1)
        complete = [grid.npc_ids[i] for i in np.flatnonzero(masks.all(axis=0))]
        if not emask.all() or set(complete) != set(row['eligible_target_ids']):
            raise ValueError('Frozen target/ego eligibility contradicts loaded data')
        eligible = [grid.npc_ids.index(i) for i in row['eligible_target_ids']]
        if self.split == 'train':
            token = '%d:%d:%d:%s' % (self.seed, self.epoch, int(self.offsets[index]), row['episode_id'])
            choice = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little') % len(eligible)
            selected = eligible[choice]
        else:
            selected = min(eligible, key=lambda i: (float(np.linalg.norm(states[h-1, i, :2])), grid.npc_ids[i]))
        # Other vehicles are selected using anchor/history only, never future.
        neighbors = [i for i in range(len(grid.npc_ids)) if i != selected and masks[h-1, i]]
        neighbors.sort(key=lambda i: (float(np.linalg.norm(states[h-1, i, :2])), grid.npc_ids[i]))
        neighbors = neighbors[:self.max_neighbors]
        neighbor_values = np.zeros((h, self.max_neighbors, 8), dtype=np.float32)
        neighbor_mask = np.zeros((h, self.max_neighbors), dtype=bool)
        if neighbors:
            neighbor_mask[:, :len(neighbors)] = masks[:h, neighbors]
            normalized = self.normalizer.transform(states[:h, neighbors], masks[:h, neighbors])
            neighbor_values[:, :len(neighbors)] = np.where(masks[:h, neighbors, None], normalized, 0.)
        map_values, map_mask, map_source = self.maps.features(grid.source_cfg, anchor)
        scene_type = {'merge': 1, 'cut_in': 2, 'keep_lane': 3}.get(grid.metadata.get('scenario_type'), 0)
        condition = {'ego_history': self.normalizer.transform(ego_history).astype(np.float32),
                     'ego_mask': emask.copy(),
                     'target_history': self.normalizer.transform(states[:h, selected]).astype(np.float32),
                     'target_mask': masks[:h, selected].copy(),
                     'neighbor_history': neighbor_values, 'neighbor_mask': neighbor_mask,
                     'map_segments': map_values, 'map_mask': map_mask,
                     'scene_type': np.asarray(scene_type, dtype=np.int64)}
        label = (states[h:, selected, :2]-self.normalizer.mean[:2])/self.normalizer.std[:2]
        metadata = {'dataset_index': int(index), 'episode_id': row['episode_id'], 'scenario_id': row['scenario_id'],
                    'parent_group_id': row['parent_group_id'], 'target_id': grid.npc_ids[selected],
                    'source_sha256': row['input_sha256'], 'source_path': self.entries[row['episode_id']]['resolved_source'],
                    'window_index': row['window_index'], 'anchor_timestamp': row['anchor_timestamp'],
                    'world_anchor': anchor, 'target_anchor_world': grid.npcs[anchor_index, selected].copy(),
                    'target_anchor_local': states[h-1, selected].copy(),
                    'target_history_world': grid.npcs[start:start+h, selected].copy(),
                    'ego_history_world': grid.ego[start:start+h].copy(),
                    'position_reference': grid.metadata.get('position_reference'),
                    'source_cfg': grid.source_cfg, 'map_source': map_source, 'split': self.split}
        return {'condition': condition, 'future_xy': label.astype(np.float32),
                'future_mask': masks[h:, selected].copy(), 'meta': metadata}

    def xy_to_world(self, normalized, meta):
        local = np.asarray(normalized)*self.normalizer.std[:2]+self.normalizer.mean[:2]
        anchor = meta['world_anchor']
        c, s = np.cos(anchor[4]), np.sin(anchor[4])
        return local@np.array([[c, s], [-s, c]])+anchor[:2]


class EpisodeSampler:
    """Shuffle within episodes and between episodes while retaining cache locality."""
    def __init__(self, dataset, seed=0):
        self.dataset, self.seed = dataset, int(seed)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = np.random.RandomState((self.seed+self.dataset.epoch) % (2**32))
        groups = list(self.dataset.episode_indices.values())
        for i in rng.permutation(len(groups)):
            for index in rng.permutation(groups[i]):
                yield int(index)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    parser.add_argument('--input-root')
    parser.add_argument('--split-manifest')
    parser.add_argument('--map-root')
    parser.add_argument('--inspect', type=int, default=16)
    parser.add_argument('--max-neighbors', type=int, default=8)
    args = parser.parse_args(argv)
    try:
        if args.inspect < 1:
            raise ValueError('--inspect must be positive')
        dataset = TrajectoryDataset(args.dataset_manifest, args.split, input_root=args.input_root,
                    split_manifest=args.split_manifest, max_neighbors=args.max_neighbors, map_root=args.map_root)
        if not len(dataset):
            raise ValueError('Requested split has no windows')
        count = min(args.inspect, len(dataset))
        samples = [dataset[i] for i in range(count)]
        error = max(float(np.max(np.abs(dataset.xy_to_world(s['future_xy'], s['meta'])-
                    (dataset._grid(s['meta']['episode_id']).npcs[
                     dataset.row(i)['start_grid_index']+dataset.config['history']:
                     dataset.row(i)['start_grid_index']+dataset.config['history']+dataset.config['future'],
                     dataset._grid(s['meta']['episode_id']).npc_ids.index(s['meta']['target_id']), :2]))))
                    for i, s in enumerate(samples))
        report = {'exit_code': 0, 'split': args.split, 'split_windows': len(dataset), 'inspected_windows': count,
                  'window_config': dataset.config, 'label_shape': list(samples[0]['future_xy'].shape),
                  'condition_shapes': {k: list(v.shape) for k, v in samples[0]['condition'].items()},
                  'finite_conditions': all(np.isfinite(v).all() for s in samples for v in s['condition'].values()),
                  'max_world_roundtrip_error_m': error, 'cache': dataset.cache_info(),
                  'fingerprint': dataset.fingerprint, 'training_performed': False}
    except (ValueError, TypeError, KeyError, OSError) as error:
        report = {'exit_code': 1, 'error': str(error)}
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
