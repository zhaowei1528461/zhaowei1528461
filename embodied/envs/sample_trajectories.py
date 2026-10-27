"""Evaluate unguided samples; export a bounded number of unverified references."""
import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-manifest')
    parser.add_argument('--synthetic-smoke', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--input-root')
    parser.add_argument('--split-manifest')
    parser.add_argument('--map-root')
    parser.add_argument('--limit', type=int, default=128, help='0 evaluates the whole explicitly requested split')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-neighbors', type=int)
    parser.add_argument('--cache-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--sampling-steps', type=int, default=20)
    parser.add_argument('--method', choices=('ddpm', 'ddim'), default='ddim')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--export-limit', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if (args.limit < 0 or args.export_limit < 0 or args.samples < 1 or args.batch_size < 1
                or args.torch_threads < 1 or args.num_workers < 0):
            raise ValueError('Invalid evaluation count/worker settings')
        try:
            from .trajectory_training import load_checkpoint, make_dataset, choose_device, new_output, evaluate, write_json, _hash
        except ImportError:
            from trajectory_training import load_checkpoint, make_dataset, choose_device, new_output, evaluate, write_json, _hash
        import torch
        torch.set_num_threads(args.torch_threads)
        device = choose_device(args.device)
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        if (checkpoint['dataset_source'] == 'synthetic_kinematics') != args.synthetic_smoke:
            raise ValueError('Synthetic and recorded dataset sources cannot be mixed')
        if args.max_neighbors is None:
            args.max_neighbors = checkpoint['training_options']['max_neighbors']
        sampling_seed = args.seed
        args.seed = checkpoint['seed']
        dataset = make_dataset(args, args.split, args.limit or None)
        args.seed = sampling_seed
        if dataset.fingerprint != checkpoint['dataset_fingerprint']:
            raise ValueError('Evaluation dataset fingerprint differs from checkpoint')
        output = new_output(args.output)
        generation = {'checkpoint_sha256': _hash(args.checkpoint),
                      'normalizer_id': checkpoint['normalizer']['normalizer_id'],
                      'dataset_fingerprint': dataset.fingerprint, 'model_config': checkpoint['model_config'],
                      'sampling_method': args.method, 'sampling_steps': args.sampling_steps}
        report = evaluate(model, dataset, device, batch_size=args.batch_size, num_samples=args.samples,
                          sampling_steps=args.sampling_steps, method=args.method, seed=args.seed,
                          num_workers=args.num_workers, output=output, export_limit=args.export_limit,
                          generation=generation)
        report.update(exit_code=0, dataset_source=checkpoint['dataset_source'], generation=generation,
                      report_path=str(output/'evaluation.json'))
        write_json(output/'evaluation.json', report)
    except (ImportError, ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        report = {'exit_code': 1, 'error': type(error).__name__+': '+str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
