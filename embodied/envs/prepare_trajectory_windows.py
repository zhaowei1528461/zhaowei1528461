"""Audit frozen-split NPC supervision and fit train-only ego-centric statistics.

Retains one episode and a window at a time. Original NPZs and the split manifest
remain read-only; this writes indexes/statistics, not tensors or model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import zlib

import numpy as np

try:
    from .scenario_data import SCHEMA_VERSION, VEHICLE_FIELDS, ScenarioNormalizer, group_episodes, iter_windows, load_episode
except ImportError:
    from scenario_data import SCHEMA_VERSION, VEHICLE_FIELDS, ScenarioNormalizer, group_episodes, iter_windows, load_episode


SPLITS = ('train', 'val', 'test')
INPUT_ERRORS = (ValueError, TypeError, KeyError, OSError, EOFError, AttributeError,
                IndexError, OverflowError, zipfile.BadZipFile, zlib.error)
FIT_SCOPE = ('Eligible train windows only; ego history plus complete target NPC history/future; '
             'overlapping windows contribute repeatedly; future ego is excluded')


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


class _Moments:
    """Stable parallel population moments; no state rows retained."""
    def __init__(self, width=8):
        self.count = 0
        self.mean = np.zeros(width)
        self.m2 = np.zeros(width)
        self.sin_yaw = self.cos_yaw = 0.

    def merge(self, other):
        if not other.count:
            return
        count = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta**2 * (self.count*other.count/count)
        self.mean += delta*(other.count/count)
        self.count = count
        self.sin_yaw += other.sin_yaw
        self.cos_yaw += other.cos_yaw

    def update(self, values):
        if not len(values):
            return
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite value in masked training population')
        part = _Moments(values.shape[1])
        part.count = len(values)
        part.mean = np.mean(values, axis=0)
        part.m2 = np.sum((values-part.mean)**2, axis=0)
        if values.shape[1] == 8:
            part.sin_yaw = float(np.sin(values[:, 4]).sum())
            part.cos_yaw = float(np.cos(values[:, 4]).sum())
        self.merge(part)


def _read_manifest(path, input_root):
    manifest_hash = _hash(path)
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(payload, dict) or payload.get('artifact_type') != 'scenario_split_manifest'
            or payload.get('schema_version') != SCHEMA_VERSION):
        raise ValueError('Expected a canonical frozen scenario split manifest')
    if payload.get('preparation_exit_code') != 0 or payload.get('all_requested_splits_nonempty') is not True:
        raise ValueError('Frozen split preparation must succeed before preparing windows')
    ratios = payload.get('ratios')
    if (not isinstance(ratios, list) or len(ratios) != 3
            or any(not isinstance(r, (int, float)) or not math.isfinite(r) or r < 0 for r in ratios)
            or not np.isclose(sum(ratios), 1.)):
        raise ValueError('Invalid frozen split ratios')
    if not isinstance(payload.get('splits'), dict) or set(payload['splits']) != set(SPLITS):
        raise ValueError('Expected train/val/test frozen memberships')
    root = Path(input_root if input_root is not None else payload['input_root']).resolve()
    entries, ids = [], set()
    for name in SPLITS:
        rows = payload['splits'][name]
        if not isinstance(rows, list):
            raise ValueError('Split members must be a list')
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Split member must be an object')
            for key in ('episode_id', 'scenario_id', 'relative_path', 'input_sha256'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError('Missing frozen member field: '+key)
            for key in ('parent_scenario_id', 'map_id'):
                if row.get(key) is not None and not isinstance(row[key], str):
                    raise ValueError('Invalid ancestry identity: '+key)
            if row['episode_id'] in ids:
                raise ValueError('Duplicate episode membership: '+row['episode_id'])
            ids.add(row['episode_id'])
            relative = Path(row['relative_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Member relative path escapes input root')
            source = (root/relative).resolve()
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise ValueError('Member relative path escapes input root') from exc
            digest = row['input_sha256']
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid source SHA256')
            entries.append(dict(row, split=name, resolved_source=str(source)))
    if not entries:
        raise ValueError('Frozen manifest has no episodes')
    groups = group_episodes([SimpleNamespace(**e) for e in entries])
    group_ids = {e.episode_id: key for key, rows in groups.items() for e in rows}
    identities, tokens, ancestry = {}, {}, {}
    for name in SPLITS:
        rows = [e for e in entries if e['split'] == name]
        identities[name] = {e['episode_id'] for e in rows}
        tokens[name] = {token for e in rows for token in (e['scenario_id'], e.get('parent_scenario_id')) if token}
        ancestry[name] = {group_ids[e['episode_id']] for e in rows}
    overlap = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i+1:]:
            overlap[left+'_'+right] = {'episode_ids': len(identities[left]&identities[right]),
                                      'lineage_tokens': len(tokens[left]&tokens[right]),
                                      'parent_groups': len(ancestry[left]&ancestry[right])}
    if any(any(row.values()) for row in overlap.values()):
        raise ValueError('Frozen manifest contains cross-split ancestry overlap/leakage')
    for e in entries:
        if e.get('parent_group_id') != group_ids[e['episode_id']]:
            raise ValueError('Frozen ancestry group identity contradicts memberships')
    return payload, entries, root, manifest_hash, {'passed': True, 'overlap_counts': overlap}


def _load_checked(entry):
    path = Path(entry['resolved_source'])
    before = path.stat()
    if _hash(path) != entry['input_sha256']:
        raise ValueError('Source hash does not match frozen manifest')
    episode = load_episode(path)
    for key in ('episode_id', 'scenario_id', 'parent_scenario_id', 'map_id'):
        if getattr(episode, key) != entry.get(key):
            raise ValueError('Loaded episode identity mismatch: '+key)
    if episode.coordinate_frame != 'world':
        raise ValueError('Ego-centric preparation requires canonical world episodes')
    return episode, (before.st_size, before.st_mtime_ns)


def _check_unchanged(entry, fingerprint):
    path = Path(entry['resolved_source'])
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != fingerprint or _hash(path) != entry['input_sha256']:
        raise ValueError('Source file changed during window preparation')


def _counters(history, future):
    return dict(episodes=0, windows_scanned=0, eligible_windows=0, complete_target_trajectories=0,
                anchor_npc_samples=0, complete_history_targets=0, complete_future_targets=0,
                ego_history_incomplete_windows=0, ego_future_complete_windows=0,
                no_complete_target_windows=0, episodes_without_eligible_windows=0,
                contiguous_history_frames={str(i): 0 for i in range(history+1)},
                contiguous_future_frames={str(i): 0 for i in range(future+1)})


def _merge_counts(total, part):
    for key, value in part.items():
        if isinstance(value, dict):
            for k, v in value.items():
                total[key][k] += v
        else:
            total[key] += value


def _coverage(window, history):
    ego = np.concatenate((window.history_ego, window.future_ego))
    npcs = np.concatenate((window.history_npcs, window.future_npcs))
    ego_mask = window.ego_mask & np.isfinite(ego).all(axis=-1)
    npc_mask = window.npc_mask & np.isfinite(npcs).all(axis=-1)
    anchor = npc_mask[history-1]
    hist = np.cumprod(npc_mask[:history][::-1], axis=0).sum(axis=0)
    fut = np.cumprod(npc_mask[history:], axis=0).sum(axis=0)
    targets = np.flatnonzero(npc_mask.all(axis=0))
    return ego_mask, anchor, hist, fut, targets


def _training_values(window, targets):
    # All values are already in the fixed ego-anchor frame; future ego excluded.
    return np.concatenate((window.history_ego,
                           window.history_npcs[:, targets].reshape(-1, 8),
                           window.future_npcs[:, targets].reshape(-1, 8)))


def _scan_episode(entry, config, stream):
    episode, fingerprint = _load_checked(entry)
    history, future = config['history'], config['future']
    counts, moments = _counters(history, future), _Moments()
    counts['episodes'] = 1
    for ordinal, window in enumerate(iter_windows(episode, coordinate_frame='ego-centric', **config)):
        counts['windows_scanned'] += 1
        ego, anchor, hist, fut, targets = _coverage(window, history)
        counts['anchor_npc_samples'] += int(anchor.sum())
        counts['complete_history_targets'] += int(np.count_nonzero((hist == history)&anchor))
        counts['complete_future_targets'] += int(np.count_nonzero((fut == future)&anchor))
        for value, count in Counter(int(v) for v in hist[anchor]).items():
            counts['contiguous_history_frames'][str(value)] += count
        for value, count in Counter(int(v) for v in fut[anchor]).items():
            counts['contiguous_future_frames'][str(value)] += count
        counts['ego_future_complete_windows'] += int(ego[history:].all())
        if not ego[:history].all():
            counts['ego_history_incomplete_windows'] += 1
            continue
        if not len(targets):
            counts['no_complete_target_windows'] += 1
            continue
        counts['eligible_windows'] += 1
        counts['complete_target_trajectories'] += len(targets)
        row = {'split': entry['split'], 'episode_id': episode.episode_id,
               'scenario_id': episode.scenario_id, 'parent_group_id': entry['parent_group_id'],
               'input_source': entry['resolved_source'], 'relative_path': entry['relative_path'],
               'input_sha256': entry['input_sha256'], 'window_index': ordinal,
               'start_grid_index': ordinal*config['stride'], 'anchor_timestamp': window.anchor_timestamp,
               'eligible_target_ids': [window.npc_ids[i] for i in targets],
               'ego_future_complete': bool(ego[history:].all())}
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        if entry['split'] == 'train':
            moments.update(_training_values(window, targets))
    counts['episodes_without_eligible_windows'] = int(not counts['eligible_windows'])
    _check_unchanged(entry, fingerprint)
    provenance = {key: episode.metadata.get(key) for key in
                  ('prediction_source', 'position_reference', 'state_alignment', 'event_verification')}
    return counts, moments, provenance


def _yaw_episode(entry, config, mean_yaw):
    episode, fingerprint = _load_checked(entry)
    moments = _Moments(1)
    for window in iter_windows(episode, coordinate_frame='ego-centric', **config):
        ego, _, _, _, targets = _coverage(window, config['history'])
        if not ego[:config['history']].all() or not len(targets):
            continue
        yaw = _training_values(window, targets)[:, 4] - mean_yaw
        moments.update(np.arctan2(np.sin(yaw), np.cos(yaw))[:, None])
    _check_unchanged(entry, fingerprint)
    return moments


def prepare_windows(manifest_path, output, history=11, future=30, dt=.2, stride=1,
                    input_root=None, progress=None):
    for key, value in (('history', history), ('future', future), ('stride', stride)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(key+' must be a positive integer')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    config = {'history': int(history), 'future': int(future), 'dt': float(dt), 'stride': int(stride)}
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output directory must be new or empty')
    manifest_path = Path(manifest_path).resolve()
    frozen, entries, root, manifest_hash, audit = _read_manifest(manifest_path, input_root)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output/'window_index.jsonl'
    totals = {name: _counters(history, future) for name in SPLITS}
    episodes, rejected, fitted_entries = [], [], []
    moments = _Moments()
    if progress:
        progress('windows', 0, len(entries), '', 'start')
    with index_path.open('w', encoding='utf-8', newline='\n') as stream:
        for number, entry in enumerate(entries, 1):
            checkpoint = stream.tell()
            try:
                counts, part, provenance = _scan_episode(entry, config, stream)
                _merge_counts(totals[entry['split']], counts)
                episodes.append({'episode_id': entry['episode_id'], 'split': entry['split'],
                                 **counts, 'source_provenance': provenance})
                if entry['split'] == 'train' and part.count:
                    moments.merge(part)
                    fitted_entries.append(entry)
                status = 'accepted'
            except INPUT_ERRORS as error:
                stream.seek(checkpoint)
                stream.truncate()
                rejected.append({'episode_id': entry['episode_id'], 'phase': 'windows',
                                 'source': entry['resolved_source'], 'error': type(error).__name__+': '+str(error)})
                status = 'rejected'
            if progress:
                progress('windows', number, len(entries), entry['resolved_source'], status)
    nonempty = all(not ratio or totals[name]['eligible_windows'] > 0
                   for name, ratio in zip(SPLITS, frozen['ratios']))
    normalizer = None
    if not rejected and nonempty and moments.count:
        mean_yaw = math.atan2(moments.sin_yaw, moments.cos_yaw)
        residual = _Moments(1)
        if progress:
            progress('normalizer', 0, len(fitted_entries), '', 'start')
        for number, entry in enumerate(fitted_entries, 1):
            try:
                residual.merge(_yaw_episode(entry, config, mean_yaw))
                status = 'accepted'
            except INPUT_ERRORS as error:
                rejected.append({'episode_id': entry['episode_id'], 'phase': 'normalizer',
                                 'source': entry['resolved_source'], 'error': type(error).__name__+': '+str(error)})
                status = 'rejected'
            if progress:
                progress('normalizer', number, len(fitted_entries), entry['resolved_source'], status)
        if not rejected:
            if residual.count != moments.count:
                raise ValueError('Normalizer passes disagree on the fitting population')
            mean = moments.mean.copy()
            mean[4] = mean_yaw
            std = np.sqrt(np.maximum(moments.m2/moments.count, 0.))
            std[4] = math.sqrt(max(float(residual.m2[0]/residual.count), 0.))
            std = np.where(std < 1e-8, 1., std)
            training_ids = tuple(sorted(e['episode_id'] for e in fitted_entries))
            identity = json.dumps({'config': config, 'coordinate_frame': 'ego-centric', 'fit_scope': FIT_SCOPE,
                                   'mean': mean.tolist(), 'std': std.tolist(), 'count': moments.count,
                                   'training_sources': sorted((e['episode_id'], e['input_sha256']) for e in fitted_entries)},
                                  sort_keys=True, allow_nan=False)
            normalizer = ScenarioNormalizer(mean, std, moments.count, training_ids,
                         'normalizer-'+hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20], 'ego-centric')
    if not rejected and normalizer is not None:
        # Held-out files have no second fit pass. Verify the complete snapshot
        # again before publishing, including train files whose fit pass ended.
        if progress:
            progress('source_check', 0, len(entries), '', 'start')
        for number, entry in enumerate(entries, 1):
            try:
                if _hash(entry['resolved_source']) != entry['input_sha256']:
                    raise ValueError('Source hash changed after its processing pass')
                status = 'accepted'
            except INPUT_ERRORS as error:
                rejected.append({'episode_id': entry['episode_id'], 'phase': 'source_check',
                                 'source': entry['resolved_source'], 'error': type(error).__name__+': '+str(error)})
                status = 'rejected'
            if progress:
                progress('source_check', number, len(entries), entry['resolved_source'], status)
    if _hash(manifest_path) != manifest_hash:
        rejected.append({'phase': 'finalize', 'source': str(manifest_path),
                         'error': 'Frozen split manifest changed during preparation'})
        normalizer = None
    exit_code = 1 if rejected else (0 if nonempty and normalizer is not None else 2)
    if exit_code == 0:
        normalizer.save(output/'normalizer.json')
    normalizer_info = {'fitted': exit_code == 0, 'coordinate_frame': 'ego-centric', 'fit_scope': FIT_SCOPE,
                       'normalizer_id': normalizer.normalizer_id if exit_code == 0 else None,
                       'path': str(output/'normalizer.json') if exit_code == 0 else None,
                       'training_state_count': moments.count,
                       'training_episode_ids': sorted(e['episode_id'] for e in fitted_entries)}
    dataset = {'artifact_type': 'trajectory_window_dataset', 'schema_version': SCHEMA_VERSION,
               'preparation_exit_code': exit_code, 'source_manifest_path': str(manifest_path),
               'source_manifest_sha256': manifest_hash, 'input_root': str(root),
               'window_config': config, 'coordinate_frame': 'ego-centric', 'vehicle_fields': list(VEHICLE_FIELDS),
               'position_reference': 'as_recorded; see per-episode source provenance',
               'target_rule': 'All eight fields finite and masked throughout history and future',
               'ego_conditioning': 'History only; future ego excluded from normalizer',
               'index_path': str(index_path), 'index_sha256': _hash(index_path),
               'normalizer': normalizer_info, 'leakage_check': audit,
               'split_window_counts': {name: totals[name]['eligible_windows'] for name in SPLITS},
               'tensors_materialized': False}
    _write_json(output/'window_dataset_manifest.json', dataset)
    report = {'artifact_type': 'trajectory_window_report', 'schema_version': SCHEMA_VERSION,
              'exit_code': exit_code, 'window_config': config,
              'history_seconds': (history-1)*float(dt), 'future_seconds': future*float(dt),
              'coordinate_frame': 'ego-centric', 'source_manifest_path': str(manifest_path),
              'source_manifest_sha256': manifest_hash, 'manifest_episode_count': len(entries),
              'splits': totals, 'leakage_check': audit, 'all_requested_splits_have_eligible_windows': nonempty,
              'normalizer': normalizer_info, 'rejected_episodes': rejected, 'episodes': episodes,
              'validation_level': 'Recorded mask/finite coverage only; not physical or simulator validity',
              'index_path': str(index_path), 'dataset_manifest_path': str(output/'window_dataset_manifest.json'),
              'report_path': str(output/'window_report.json')}
    _write_json(output/'window_report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--input-root', help='Relocated pool root; frozen relative paths and hashes are preserved')
    parser.add_argument('--history', type=int, default=11)
    parser.add_argument('--future', type=int, default=30)
    parser.add_argument('--dt', type=float, default=.2)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--progress-every', type=int, default=25)
    args = parser.parse_args(argv)
    if args.progress_every < 1:
        parser.error('--progress-every must be positive')
    def progress(phase, number, total, path, status):
        if number in (0, 1, total) or number % args.progress_every == 0 or status == 'rejected':
            print('[%s %d/%d] %s %s' % (phase, number, total, status, Path(path).name if path else ''),
                  file=sys.stderr, flush=True)
    try:
        report = prepare_windows(args.manifest, args.output, history=args.history, future=args.future,
                                 dt=args.dt, stride=args.stride, input_root=args.input_root, progress=progress)
        # Detailed histograms and per-episode provenance stay in the saved report.
        summary = {key: value for key, value in report.items() if key not in ('episodes', 'splits', 'normalizer')}
        summary['splits'] = {name: {key: value for key, value in counts.items() if not isinstance(value, dict)}
                             for name, counts in report['splits'].items()}
        summary['normalizer'] = {key: value for key, value in report['normalizer'].items()
                                 if key != 'training_episode_ids'}
        summary['normalizer']['training_episode_count'] = len(report['normalizer']['training_episode_ids'])
    except INPUT_ERRORS as error:
        summary = {'exit_code': 1, 'error': type(error).__name__+': '+str(error)}
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return summary['exit_code']


if __name__ == '__main__':
    sys.exit(main())
