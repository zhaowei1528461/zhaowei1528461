"""Train plain single-NPC x/y diffusion using the already-frozen window index."""
import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest')
    parser.add_argument('--synthetic-smoke', action='store_true')
    parser.add_argument('--allow-train-only-smoke', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--input-root')
    parser.add_argument('--split-manifest')
    parser.add_argument('--map-root')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--train-limit', type=int, default=0, help='0 uses all train windows')
    parser.add_argument('--val-limit', type=int, default=512, help='0 uses all val windows; default explicitly monitors a subset')
    parser.add_argument('--max-neighbors', type=int, default=8)
    parser.add_argument('--cache-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base-dim', type=int, default=32)
    parser.add_argument('--dim-mults', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--condition-dim', type=int, default=64)
    parser.add_argument('--diffusion-steps', type=int, default=100)
    parser.add_argument('--beta-schedule', choices=('cosine', 'linear'), default='cosine')
    parser.add_argument('--sampling-steps', type=int, default=20)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=100)
    args = parser.parse_args(argv)
    try:
        if (args.lr <= 0 or args.train_limit < 0 or args.val_limit < 0 or args.num_workers < 0
                or args.torch_threads < 1 or args.log_every < 1):
            raise ValueError('Invalid learning rate/count/worker settings')
        try:
            from .trajectory_training import run_training
        except ImportError:
            from trajectory_training import run_training
        report = run_training(args)
    except (ImportError, ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        report = {'exit_code': 1, 'error': type(error).__name__+': '+str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
