"""Standalone direct diffusion training/evaluation; no original agent imports."""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .scenario_data import ScenarioNormalizer, _json_safe
    from .trajectory_dataset import TrajectoryDataset, EpisodeSampler
    from .trajectory_diffusion import make_model, collate_samples, to_device
    from .trajectory_metrics import displacement_metrics, constant_velocity, states_from_xy, motion_valid, PROVISIONAL_LIMITS
    from .diffusion_candidate_adapter import make_candidate, save_candidate
    from .prepare_trajectory_windows import _hash
except ImportError:
    from scenario_data import ScenarioNormalizer, _json_safe
    from trajectory_dataset import TrajectoryDataset, EpisodeSampler
    from trajectory_diffusion import make_model, collate_samples, to_device
    from trajectory_metrics import displacement_metrics, constant_velocity, states_from_xy, motion_valid, PROVISIONAL_LIMITS
    from diffusion_candidate_adapter import make_candidate, save_candidate
    from prepare_trajectory_windows import _hash


def write_json(path, payload):
    Path(path).write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def normalizer_payload(normalizer):
    return {'mean': normalizer.mean.tolist(), 'std': normalizer.std.tolist(), 'count': int(normalizer.count),
            'normalizer_id': normalizer.normalizer_id, 'coordinate_frame': normalizer.coordinate_frame,
            'training_episode_ids': list(normalizer.training_episode_ids), 'yaw_mode': 'circular_mean_wrapped_residual'}


class SyntheticTrajectoryDataset:
    """Explicit kinematic format smoke data; never a recorded-episode substitute."""
    def __init__(self, split='train', limit=None, seed=0, max_neighbors=8):
        self.split, self.seed, self.epoch = split, int(seed), 0
        self.config = {'history': 11, 'future': 30, 'dt': .2, 'stride': 1}
        self.available_windows = 16 if split == 'train' else 4
        self.count = min(limit, self.available_windows) if limit is not None else self.available_windows
        self.max_neighbors = max_neighbors
        self.normalizer = ScenarioNormalizer(np.zeros(8), np.array([20., 10., 10., 10., 1., 4., 5., 2.]),
                       0, tuple('synthetic-train-%d' % i for i in range(16)), 'synthetic-fixed-scales-v1', 'ego-centric')
        self.fingerprint = {'dataset_source': 'synthetic_kinematics', 'generator_version': 1,
                            'seed': self.seed, 'window_config': dict(self.config),
                            'normalizer_id': self.normalizer.normalizer_id}
        self.episode_indices = {'synthetic-%d' % i: np.array([i]) for i in range(self.count)}
        self.manifest = {'split_window_counts': {'train': 16, 'val': 4, 'test': 4}}
        self.maps = None

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        h, f, dt = self.config['history'], self.config['future'], self.config['dt']
        offset = 0 if self.split == 'train' else 100 if self.split == 'val' else 200
        i = int(index)+offset
        past, future = np.arange(-(h-1), 1)*dt, np.arange(1, f+1)*dt
        position, speed = np.array([10.+(i % 5), 3.+.2*(i % 3)]), 3.+.2*(i % 4)
        ego = np.tile([0., 0., 5., 0., 0., 0., 4.5, 1.8], (h, 1))
        ego[:, 0] = 5*past
        target = np.tile([position[0], position[1], speed, 0., 0., 0., 4.5, 1.8], (h, 1))
        target[:, 0] += speed*past
        label = position+np.stack((speed*future, .2*np.sin(future*.5)*(i % 3-1)), axis=-1)
        condition = {'ego_history': self.normalizer.transform(ego).astype(np.float32), 'ego_mask': np.ones(h, bool),
                     'target_history': self.normalizer.transform(target).astype(np.float32), 'target_mask': np.ones(h, bool),
                     'neighbor_history': np.zeros((h, self.max_neighbors, 8), np.float32),
                     'neighbor_mask': np.zeros((h, self.max_neighbors), bool),
                     'map_segments': np.zeros((64, 6), np.float32), 'map_mask': np.zeros(64, bool),
                     'scene_type': np.asarray(0, dtype=np.int64)}
        origin = np.array([100., 200.])
        target_world, ego_world = target.copy(), ego.copy()
        target_world[:, :2] += origin
        ego_world[:, :2] += origin
        metadata = {'dataset_index': int(index), 'episode_id': 'synthetic-%s-%d' % (self.split, i),
                    'scenario_id': 'synthetic-%s-%d' % (self.split, i), 'parent_group_id': 'synthetic-%s-%d' % (self.split, i),
                    'target_id': 'target', 'source_path': 'synthetic_kinematics', 'source_sha256': 'not_a_recorded_npz',
                    'window_index': 0, 'anchor_timestamp': 2., 'world_anchor': ego_world[-1],
                    'target_anchor_world': target_world[-1], 'target_anchor_local': target[-1],
                    'target_history_world': target_world, 'ego_history_world': ego_world,
                    'position_reference': 'center', 'map_source': {'available': False}, 'source_cfg': None, 'split': self.split}
        return {'condition': condition, 'future_xy': ((label-self.normalizer.mean[:2])/self.normalizer.std[:2]).astype(np.float32),
                'future_mask': np.ones(f, bool), 'meta': metadata}

    def xy_to_world(self, values, metadata):
        return TrajectoryDataset.xy_to_world(self, values, metadata)


def new_output(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output directory must be new or empty')
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_device(value):
    if value == 'auto':
        value = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable')
    return device


def make_dataset(args, split, limit=None):
    if args.synthetic_smoke:
        return SyntheticTrajectoryDataset(split, limit=limit, seed=args.seed, max_neighbors=args.max_neighbors)
    if not args.dataset_manifest:
        raise ValueError('--dataset-manifest is required unless --synthetic-smoke is used')
    return TrajectoryDataset(args.dataset_manifest, split, input_root=args.input_root,
                split_manifest=args.split_manifest, max_neighbors=args.max_neighbors,
                cache_size=args.cache_size, limit=limit, seed=args.seed, map_root=args.map_root)


def save_checkpoint(path, model, optimizer, config, normalizer, fingerprint, step, epoch, args):
    torch.save({'format': 'sumo_direct_trajectory_diffusion', 'schema_version': 1,
                'model_config': dict(config), 'normalizer': normalizer, 'dataset_fingerprint': fingerprint,
                'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'optimization_steps': step, 'epoch': epoch, 'seed': int(args.seed),
                'dataset_source': 'synthetic_kinematics' if args.synthetic_smoke else 'recorded_sumo',
                'training_options': dict(vars(args)), 'torch_version': str(torch.__version__)}, path)


def load_checkpoint(path, device='cpu', expected_fingerprint=None):
    payload = torch.load(path, map_location=device, weights_only=True)
    if (payload.get('format') != 'sumo_direct_trajectory_diffusion' or payload.get('schema_version') != 1):
        raise ValueError('Unsupported trajectory checkpoint')
    if expected_fingerprint is not None and payload['dataset_fingerprint'] != expected_fingerprint:
        raise ValueError('Checkpoint dataset fingerprint mismatch')
    if payload['normalizer']['normalizer_id'] != payload['dataset_fingerprint']['normalizer_id']:
        raise ValueError('Checkpoint normalizer identity mismatch')
    model = make_model(payload['model_config'], payload['normalizer']).to(device)
    model.load_state_dict(payload['model_state'], strict=True)
    return model, payload


def _plot(path, predictions, truth, metadata):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    history = metadata['target_history_world'][:, :2]
    ax.plot(history[:, 0], history[:, 1], 'k.-', label='Recorded history')
    ax.plot(truth[:, 0], truth[:, 1], 'g.-', label='Recorded future')
    for i, candidate in enumerate(predictions):
        ax.plot(candidate[:, 0], candidate[:, 1], alpha=.6, label='Generated samples' if i == 0 else None)
    ax.set(xlabel='World x [m]', ylabel='World y [m]', title='Offline generated NPC references; not simulation verified')
    ax.axis('equal'); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return str(path)


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=32, num_samples=4, sampling_steps=20,
             seed=0, method='ddim', num_workers=0, output=None, export_limit=0, generation=None):
    if not len(dataset):
        raise ValueError('Evaluation split is empty')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate_samples, generator=torch.Generator().manual_seed(seed))
    sums, count, duration = defaultdict(float), 0, 0.
    motion_ok = road_ok = road_count = map_windows = 0
    exported, plots = [], []
    previous_training = model.training
    model.eval()
    try:
        for number, batch in enumerate(loader):
            batch = to_device(batch, device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            sampled = model.sample(batch['condition'], num_samples=num_samples, method=method,
                                   steps=sampling_steps, seed=seed+number)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            duration += time.perf_counter()-started
            generator = torch.Generator(device=device).manual_seed(seed+100000+number)
            t = torch.randint(model.num_steps, (len(sampled),), device=device, generator=generator)
            noise = torch.randn(batch['future_xy'].shape, device=device, generator=generator)
            loss = model.training_loss(batch['condition'], batch['future_xy'], batch['future_mask'], t=t, noise=noise)
            normalizer = dataset.normalizer
            prediction = sampled.cpu().numpy()*normalizer.std[:2]+normalizer.mean[:2]
            truth = batch['future_xy'].cpu().numpy()*normalizer.std[:2]+normalizer.mean[:2]
            result = displacement_metrics(prediction, truth)
            baseline = constant_velocity(batch['meta'], model.future, dataset.config['dt'])
            cv = displacement_metrics(baseline[:, None], truth)
            result.update(constant_velocity_ADE_m=cv['mean_ADE_m'], constant_velocity_FDE_m=cv['mean_FDE_m'],
                          noise_prediction_loss=float(loss))
            n = len(prediction)
            for key, value in result.items():
                sums[key] += n*value
            for i, metadata in enumerate(batch['meta']):
                world = dataset.xy_to_world(sampled[i].cpu().numpy(), metadata)
                world_truth = dataset.xy_to_world(batch['future_xy'][i].cpu().numpy(), metadata)
                loaded_map = dataset.maps.load(metadata['source_cfg']) if dataset.maps is not None else None
                if loaded_map is not None:
                    map_windows += 1
                for trajectory in world:
                    states = states_from_xy(trajectory, metadata['target_anchor_world'], dataset.config['dt'])
                    motion_ok += int(motion_valid(states, dataset.config['dt']))
                    if loaded_map is not None and metadata['position_reference'] in ('center', 'front_bumper_center'):
                        center = states.copy()
                        if metadata['position_reference'] == 'front_bumper_center':
                            center[:, :2] -= .5*center[:, 6:7]*np.stack((np.cos(center[:, 4]), np.sin(center[:, 4])), axis=-1)
                        road_count += 1
                        road_ok += int(loaded_map[0].footprint_valid(center))
                if output is not None and len(exported) < export_limit:
                    # Export the first sample, never an oracle selected by future labels.
                    candidate_generation = dict(generation or {}, sampling_context={
                        'batch_size': n, 'batch_row': i, 'num_samples': num_samples,
                        'batch_dataset_indices': [m['dataset_index'] for m in batch['meta']]})
                    candidate = make_candidate(metadata, world[0], dataset.config['dt'], candidate_generation,
                                               seed=seed+number, sample_index=0)
                    prefix = Path(output)/('candidate_%04d' % len(exported))
                    exported.append(save_candidate(candidate, prefix))
                    plot = _plot(prefix.with_suffix('.png'), world, world_truth, metadata)
                    if plot:
                        plots.append(plot)
            count += n
        report = {key: value/count for key, value in sums.items()}
        report.update(evaluated_windows=count, evaluation_split=dataset.split, samples_per_window=num_samples,
                      sampling_method=method, sampling_steps=sampling_steps,
                      limited_evaluation=count < dataset.manifest['split_window_counts'][dataset.split],
                      sampling_seconds=duration, seconds_per_window=duration/count,
                      motion_valid_rate=motion_ok/(count*num_samples), motion_limits=PROVISIONAL_LIMITS,
                      motion_validity_definition='sampled kinematics under provisional configured thresholds; not calibrated vehicle validity',
                      road_valid_rate=road_ok/road_count if road_count else None,
                      road_checked_samples=road_count, map_missing_window_rate=1-map_windows/count,
                      road_validity_definition='sampled footprint in current SUMO lane buffers; not simulator verification',
                      oracle_metric_definition='minADE and minFDE independently select the best of K against recorded labels',
                      exported_candidates=exported, plots=plots, simulation_verified=False)
        return report
    finally:
        model.train(previous_training)


def run_training(args):
    if args.epochs < 1 or args.batch_size < 1 or args.samples < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError('Training counts must be positive')
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    train = make_dataset(args, 'train', args.train_limit or None)
    val = make_dataset(args, 'val', args.val_limit or None)
    if not len(train) or (not len(val) and not args.allow_train_only_smoke):
        raise ValueError('Nonempty train/val required; use --allow-train-only-smoke only for explicit smoke checks')
    config = dict(history=train.config['history'], future=train.config['future'], dt=train.config['dt'],
                  base_dim=args.base_dim, dim_mults=args.dim_mults, condition_dim=args.condition_dim,
                  diffusion_steps=args.diffusion_steps, beta_schedule=args.beta_schedule)
    if args.sampling_steps > args.diffusion_steps:
        raise ValueError('Sampling steps cannot exceed diffusion steps')
    output = new_output(args.output)
    write_json(output/'config.json', {'model': config, 'training': vars(args), 'dataset_fingerprint': train.fingerprint})
    normalizer = normalizer_payload(train.normalizer)
    model = make_model(config, normalizer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler = EpisodeSampler(train, seed=args.seed)
    step, best, epochs, last_validation = 0, float('inf'), [], None
    for epoch in range(args.epochs):
        train.set_epoch(epoch)
        loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                            collate_fn=collate_samples, generator=torch.Generator().manual_seed(args.seed+epoch))
        model.train()
        loss_sum = trained = 0
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(batch['condition'], batch['future_xy'], batch['future_mask'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient_norm):
                raise ValueError('Nonfinite training gradients')
            optimizer.step()
            step += 1
            loss_sum += float(loss.detach())*len(batch['future_xy'])
            trained += len(batch['future_xy'])
            if step == 1 or step % args.log_every == 0:
                print('[train epoch=%d step=%d] loss=%.6f' % (epoch+1, step, float(loss.detach())), file=__import__('sys').stderr, flush=True)
            if args.max_steps is not None and step >= args.max_steps:
                break
        if len(val):
            last_validation = evaluate(model, val, device, batch_size=args.batch_size, num_samples=args.samples,
                                     sampling_steps=args.sampling_steps, seed=args.seed, num_workers=args.num_workers)
            score = last_validation['minADE_m']
        else:
            score = loss_sum/max(trained, 1)
        row = {'epoch': epoch+1, 'optimization_steps': step, 'train_loss': loss_sum/max(trained, 1),
               'trained_window_visits': trained, 'validation': last_validation}
        epochs.append(row)
        with (output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        save_checkpoint(output/'latest.pt', model, optimizer, config, normalizer, train.fingerprint, step, epoch+1, args)
        if score < best:
            best = score
            save_checkpoint(output/'best.pt', model, optimizer, config, normalizer, train.fingerprint, step, epoch+1, args)
        if args.max_steps is not None and step >= args.max_steps:
            break
    restored, _ = load_checkpoint(output/'latest.pt', device, expected_fingerprint=train.fingerprint)
    probe = to_device(collate_samples([train[0]]), device)
    a = model.sample(probe['condition'], method='ddim', steps=args.sampling_steps, seed=args.seed)
    b = restored.sample(probe['condition'], method='ddim', steps=args.sampling_steps, seed=args.seed)
    if not torch.equal(a, b):
        raise ValueError('Saved checkpoint does not reproduce seeded samples')
    report = {'exit_code': 0, 'dataset_source': 'synthetic_kinematics' if args.synthetic_smoke else 'recorded_sumo',
              'device': str(device), 'torch_version': str(torch.__version__), 'model_config': config,
              'dataset_fingerprint': train.fingerprint, 'normalizer_id': train.normalizer.normalizer_id,
              'optimization_steps': step, 'epochs_completed': len(epochs), 'train_windows_selected': len(train),
              'validation_windows_selected': len(val), 'validation': last_validation,
              'train_only_smoke': not len(val), 'checkpoint_roundtrip_verified': True,
              'best_checkpoint_selection': 'val minADE over K samples' if len(val) else 'train loss for smoke only',
              'latest_checkpoint': str(output/'latest.pt'), 'best_checkpoint': str(output/'best.pt'),
              'training_report_path': str(output/'training_report.json'), 'simulation_verified': False}
    write_json(output/'training_report.json', report)
    return report
