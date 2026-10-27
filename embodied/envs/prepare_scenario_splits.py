"""Index recorded SUMO episodes one at a time and freeze ancestry-safe splits.

No windows, training tensors or normalizer are created. Full trajectories and
raw frames are released after each canonical episode has been inspected.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional
import zipfile
import zlib

import numpy as np

try:
    from .scenario_data import SCHEMA_VERSION, group_episodes, load_episode, split_episodes
except ImportError:
    from scenario_data import SCHEMA_VERSION, group_episodes, load_episode, split_episodes


RECORDED_SOURCE_TYPES = {'sumo_original', 'sumo_copy', 'sumo_attack', 'sumo_replay'}


class ExcludedEpisode(ValueError):
    pass


@dataclass
class EpisodeIndex:
    episode_id: str
    scenario_id: str
    parent_scenario_id: Optional[str]
    map_id: Optional[str]
    input_source: str
    relative_path: str
    input_sha256: str
    source_type: str
    prediction_source: Optional[str]
    source_cfg: Optional[str]
    coordinate_frame: str
    frame_count: int
    npc_count: int
    dt: Optional[float]
    collection_complete: Optional[bool]


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def _read_index(path, input_root):
    before = path.stat()
    # Stage 3 candidate NPZs deliberately have a different contract. Identify
    # them before the canonical loader reports missing recorded-episode arrays.
    with np.load(path, allow_pickle=False) as archive:
        if 'metadata_json' in archive.files:
            header = json.loads(str(archive['metadata_json'].item()))
            if not isinstance(header, dict):
                raise ValueError('metadata_json must contain a JSON object')
            if header.get('metadata') is not None and not isinstance(header['metadata'], dict):
                raise ValueError('Episode metadata must be a JSON object or null')
            if 'raw_frames_json' not in archive.files and header.get('format') == 'long_tail_candidate_scene':
                raise ExcludedEpisode('Generated Stage 3 candidate, not a recorded episode')
    episode = load_episode(path)
    if episode.source_type not in RECORDED_SOURCE_TYPES:
        raise ExcludedEpisode('Not a recorded SUMO source_type: %s' % episode.source_type)
    prediction_source = episode.metadata.get('prediction_source')
    if prediction_source not in (None, 'sumo_execution'):
        raise ExcludedEpisode('Non-execution prediction_source: %s' % prediction_source)
    for name in ('episode_id', 'scenario_id'):
        value = getattr(episode, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Nonempty string %s required' % name)
    for name in ('parent_scenario_id', 'map_id'):
        value = getattr(episode, name)
        if value is not None and not isinstance(value, str):
            raise ValueError('%s must be a string or None' % name)
    if len(episode.timestamps) < 2:
        raise ValueError('Recorded episode needs at least two timestamps')
    if episode.dt is not None:
        if not np.isfinite(episode.dt) or episode.dt <= 0:
            raise ValueError('Invalid native episode dt')
        if not np.allclose(np.diff(episode.timestamps), episode.dt, rtol=1e-6, atol=1e-7):
            raise ValueError('Recorded dt contradicts the native timestamp intervals')
    index = EpisodeIndex(
        episode.episode_id, episode.scenario_id, episode.parent_scenario_id,
        episode.map_id, str(path), path.relative_to(input_root).as_posix(), _sha256(path),
        episode.source_type, prediction_source, episode.source_cfg, episode.coordinate_frame,
        len(episode.timestamps), len(episode.npc_ids), episode.dt,
        episode.metadata.get('collection_complete'))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Input file changed while scanning; use a stable pool snapshot')
    return index


def scan_pool(input_path, progress=None):
    source = Path(input_path).resolve()
    if not source.exists():
        raise FileNotFoundError('Episode input does not exist: ' + str(source))
    input_root = source if source.is_dir() else source.parent
    paths = sorted(p for p in source.rglob('*.npz') if p.is_file()) if source.is_dir() else [source]
    if not paths or any(p.suffix.lower() != '.npz' for p in paths):
        raise ValueError('Expected recorded episode NPZ files')
    issues = {'input_root': str(input_root), 'discovered_file_count': len(paths),
              'rejected_files': [], 'excluded_files': [], 'duplicate_files': []}
    indexes, conflicts = {}, set()
    if progress:
        progress(0, len(paths), '', 'start')
    for number, path in enumerate(paths, 1):
        status = 'accepted'
        try:
            index = _read_index(path, input_root)
            if index.episode_id in conflicts:
                raise ValueError('Conflicting episode_id: ' + index.episode_id)
            previous = indexes.get(index.episode_id)
            if previous is not None:
                if previous.input_sha256 == index.input_sha256:
                    issues['duplicate_files'].append({'source': str(path), 'episode_id': index.episode_id,
                                                     'canonical_source': previous.input_source})
                    status = 'duplicate'
                else:
                    del indexes[index.episode_id]
                    conflicts.add(index.episode_id)
                    issues['rejected_files'].append({'source': previous.input_source,
                                                     'error': 'Conflicting episode_id: '+index.episode_id})
                    raise ValueError('Conflicting episode_id: ' + index.episode_id)
            else:
                indexes[index.episode_id] = index
        except ExcludedEpisode as error:
            issues['excluded_files'].append({'source': str(path), 'reason': str(error)})
            status = 'excluded'
        except (ValueError, TypeError, KeyError, OSError, EOFError, zipfile.BadZipFile, zlib.error) as error:
            issues['rejected_files'].append({'source': str(path), 'error': str(error)})
            status = 'rejected'
        if progress:
            progress(number, len(paths), str(path), status)
    return sorted(indexes.values(), key=lambda i: i.episode_id), issues


def _audit(splits, group_ids):
    identities, tokens, groups = {}, {}, {}
    for name, rows in splits.items():
        identities[name] = {e.episode_id for e in rows}
        tokens[name] = {value for e in rows for value in (e.scenario_id, e.parent_scenario_id) if value}
        groups[name] = {group_ids[e.episode_id] for e in rows}
    overlap = {}
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            overlap[left+'_'+right] = {
                'episode_ids': len(identities[left] & identities[right]),
                'lineage_tokens': len(tokens[left] & tokens[right]),
                'parent_groups': len(groups[left] & groups[right])}
    passed = all(not any(counts.values()) for counts in overlap.values())
    if not passed:
        raise ValueError('Cross-split episode or ancestry overlap detected')
    return {'passed': passed, 'overlap_counts': overlap}


def prepare_pool(input_path, output, ratios=(.8, .1, .1), seed=0, progress=None):
    # Reuse legacy validation and deterministic allocation, including small pools.
    split_episodes([], ratios=ratios, seed=seed)
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output directory must be new or empty; preserve frozen splits')
    indexes, issues = scan_pool(input_path, progress=progress)
    groups = group_episodes(indexes)
    group_ids = {episode.episode_id: key for key, rows in groups.items() for episode in rows}
    splits = split_episodes(indexes, ratios=ratios, seed=seed)
    audit = _audit(splits, group_ids)
    split_counts = {name: {'episodes': len(rows),
                          'parent_groups': len({group_ids[e.episode_id] for e in rows})}
                    for name, rows in splits.items()}
    nonempty = all(not ratio or split_counts[name]['episodes'] > 0
                   for name, ratio in zip(('train', 'val', 'test'), ratios))
    exit_code = 1 if issues['rejected_files'] or not indexes else (0 if nonempty else 2)
    manifest = {'artifact_type': 'scenario_split_manifest', 'schema_version': SCHEMA_VERSION,
                'seed': seed, 'ratios': list(ratios), 'group_by': 'scenario',
                'input_root': issues['input_root'], 'normalizer_id': None, 'normalizer_path': None,
                'leakage_check': audit, 'all_requested_splits_nonempty': nonempty,
                'preparation_exit_code': exit_code,
                'splits': {name: [{**asdict(e), 'parent_group_id': group_ids[e.episode_id]} for e in rows]
                           for name, rows in splits.items()}}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, index_path, report_path = (output/name for name in
                                            ('split_manifest.json', 'episode_index.jsonl', 'dataset_report.json'))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    destinations = {e.episode_id: name for name, rows in splits.items() for e in rows}
    with index_path.open('w', encoding='utf-8', newline='\n') as stream:
        for e in indexes:
            row = {**asdict(e), 'parent_group_id': group_ids[e.episode_id], 'split': destinations[e.episode_id]}
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
    completeness = Counter('true' if e.collection_complete is True else
                           'false' if e.collection_complete is False else 'unknown' for e in indexes)
    report = {'artifact_type': 'recorded_pool_split_report', 'schema_version': SCHEMA_VERSION,
              'exit_code': exit_code, 'seed': seed, 'ratios': list(ratios),
              'recorded_episode_count': len(indexes), 'independent_parent_group_count': len(groups),
              'distinct_scenario_id_count': len({e.scenario_id for e in indexes}),
              'declared_parent_id_count': len({e.parent_scenario_id for e in indexes if e.parent_scenario_id}),
              'source_type_counts': dict(Counter(e.source_type for e in indexes)),
              'collection_completeness_counts': dict(completeness),
              'native_dt_values': sorted({e.dt for e in indexes if e.dt is not None}),
              'nonuniform_episode_count': sum(e.dt is None for e in indexes),
              'source_type_only_provenance_count': sum(e.prediction_source is None for e in indexes),
              'zero_npc_episode_count': sum(e.npc_count == 0 for e in indexes),
              'splits': split_counts, 'leakage_check': audit,
              'all_requested_splits_nonempty': nonempty, 'windows_materialized': False,
              'window_validity_checked': False, 'normalizer_fitted': False,
              'validation_level': 'canonical NPZ schema, array shapes and timestamps; no per-window eligibility checks',
              'snapshot_rule': 'Only NPZ paths discovered at scan start are included; source file hashes are recorded.',
              'manifest_path': str(manifest_path), 'index_path': str(index_path), 'report_path': str(report_path),
              'rejected_file_count': len(issues['rejected_files']), 'excluded_file_count': len(issues['excluded_files']),
              'duplicate_file_count': len(issues['duplicate_files']), **issues}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Recorded NPZ pool or one canonical episode NPZ')
    parser.add_argument('--output', required=True, help='New directory for frozen index/manifest/report')
    parser.add_argument('--ratios', nargs=3, type=float, default=(.8, .1, .1))
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--progress-every', type=int, default=25)
    args = parser.parse_args(argv)
    if args.progress_every < 1:
        parser.error('--progress-every must be positive')
    def progress(number, total, path, status):
        if number in (0, 1, total) or number % args.progress_every == 0 or status in ('rejected', 'excluded'):
            print('[scan %d/%d] %s %s' % (number, total, status, Path(path).name if path else ''),
                  file=sys.stderr, flush=True)
    try:
        report = prepare_pool(args.input, args.output, ratios=tuple(args.ratios), seed=args.split_seed, progress=progress)
    except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile) as error:
        report = {'exit_code': 1, 'error': str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
