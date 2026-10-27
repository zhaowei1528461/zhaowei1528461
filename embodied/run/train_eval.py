import json
import os
import re

import embodied
import numpy as np


class CostEma:

    def __init__(self, initial=0):
        self.value = initial


class Arrive:

    def __init__(self):
        self.value = []


def _enable_danger_posttraining(train_env, freeze_library):
    """Share first/resumed phase controls without changing episode collection."""
    # Freeze controls the old library/logs only. None keeps each worker's
    # independently configured episode collection and online update settings.
    return train_env.call(
        'set_danger_mining', True,
        record_events=not freeze_library, collect_episodes=None,
        update_library=False if freeze_library else None)


def train_eval(
        agent, train_env, eval_env, train_replay, eval_replay, logger, args, lag):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)    # 模型储存地址
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)   # 定期写日志
    should_save = embodied.when.Clock(args.save_every)     # 定期存储
    should_eval = embodied.when.Every(args.eval_every, args.eval_initial)   # 定期评估
    should_sync = embodied.when.Every(args.sync_every)    # 定期同步参数到推理设备上
    step = logger.step
    global_episode = 0
    posttrain_cfg = getattr(args, 'danger_posttrain', {})

    def pget(name, default):
        if isinstance(posttrain_cfg, dict):
            return posttrain_cfg.get(name, default)
        return getattr(posttrain_cfg, name, default)

    posttrain_enabled = bool(pget('enabled', False))
    bootstrap_episodes = int(pget('bootstrap_episodes', 6000))
    freeze_danger_library_after_sweep = bool(
        pget('freeze_danger_library_after_sweep', False))
    posttrain_pending = [False]
    posttrain_started = [False]
    eval_mode = ['eval']
    sweep_results_path = logdir / 'bootstrap_sweep_results.jsonl'
    sweep_result_count = [0]
    sweep_attempt_count = [0]
    sweep_seen = [set()]
    cost_ema = CostEma(0.0)
    train_arrive_num = Arrive()
    eval_arrive_num = Arrive()
    updates = embodied.Counter()     # 更新step回合
    metrics = embodied.Metrics()    # 统计训练参数方便写数据
    print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
    print('Action space:', embodied.format(train_env.act_space), sep='\n')

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
    timer.wrap('env', train_env, ['step'])
    if hasattr(train_replay, '_sample'):
        timer.wrap('replay', train_replay, ['_sample'])

    nonzeros = set()
    dual_gate_status = {
        'fea': 0.0,
        'fea_ema': 0.0,
        'thresh': 0.0,
        'enabled': False,
        'ratio': 0.0,
        'real_steps': 0.0,
    }

    def current_train_phase():
        if not posttrain_enabled:
            return 'train'
        return 'posttrain' if posttrain_started[0] else 'bootstrap'

    def json_ready(value):
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_ready(v) for v in value]
        return value

    def get_eval_scenario(worker):
        try:
            scenarios = eval_env.call('get_current_scenario')
        except Exception as exc:
            return {'scenario_lookup_error': str(exc)}
        if worker is not None and 0 <= int(worker) < len(scenarios):
            scenario = scenarios[int(worker)]
        else:
            scenario = scenarios[0] if scenarios else {}
        if isinstance(scenario, dict):
            return dict(scenario)
        return {
            'scenario_path': str(scenario),
            'scenario_name': os.path.basename(str(scenario)),
        }

    def write_sweep_result(ep, worker, length, score, cost, behavior, dual_steps):
        scenario = get_eval_scenario(worker)
        scenario_key = (
            scenario.get('scenario_path') or
            scenario.get('scenario_name') or
            f'worker_{worker}_result_{sweep_attempt_count[0]}')
        sweep_attempt_count[0] += 1
        if scenario_key in sweep_seen[0]:
            print(
                '[danger posttrain] duplicate sweep result ignored for '
                f'{scenario_key}')
            return
        sweep_seen[0].add(scenario_key)
        record = {
            'index': sweep_result_count[0],
            'unique_scenarios': len(sweep_seen[0]),
            'global_step': int(step),
            'bootstrap_episode': int(global_episode),
            'worker': int(worker) if worker is not None else -1,
            'length': int(length),
            'score': float(score),
            'cost': float(cost) if cost is not None else 0.0,
            'max_cost': (
                float(np.asarray(ep['cost'], dtype=np.float64).max())
                if 'cost' in ep else 0.0),
            'behavior': behavior,
            'dual_steps': int(dual_steps),
            'is_dual': float(behavior == 'dual'),
        }
        if 'arrive_dest' in ep:
            record['arrive_dest'] = int(np.asarray(ep['arrive_dest'])[-1])
        record.update(scenario)
        sweep_results_path.write(
            json.dumps(json_ready(record), ensure_ascii=False, sort_keys=True) + '\n',
            mode='a')
        sweep_result_count[0] += 1
        print(
            '[danger posttrain] sweep result '
            f"{record.get('scenario_name', record.get('scenario_path', 'unknown'))} "
            f"score={score:.1f} cost={record['cost']:.1f} length={length}")

    def per_episode(ep, mode, worker=None):
        nonlocal global_episode    # ⭐ 加这一行

        if mode == 'train':
            global_episode += 1
            if posttrain_enabled and global_episode >= bootstrap_episodes:
                # The step callback already trained on this terminal
                # transition. Remaining collection steps in the current block
                # stay frozen until the all-scene sweep has completed.
                posttrain_pending[0] = True
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        dual_steps = 0
        dual_episode = False
        if 'log_use_dual' in ep:
            dual_values = ep['log_use_dual'].astype(np.float32)
            dual_steps = int((dual_values > 0.5).sum())
            dual_episode = dual_steps > 0
        behavior = 'dual' if dual_episode else 'main'
        phase = current_train_phase() if mode == 'train' else mode
        phase_id = {
            'bootstrap': 0.0,
            'posttrain': 1.0,
            'train': 0.0,
            'eval': -1.0,
            'bootstrap_sweep': 2.0,
        }.get(phase, -1.0)
        base_prefix = 'episode' if mode == 'train' else f'{mode}_episode'
        behavior_prefix = (
            f'{behavior}_episode' if mode == 'train'
            else f'{mode}_{behavior}_episode')
        logger.add({
            'length': length,
            'score': score,
            'dual_steps': dual_steps,
            'is_dual': float(dual_episode),
            'phase_id': phase_id,
        }, prefix=base_prefix)
        logger.add({
            'length': length,
            'score': score,
            'dual_steps': dual_steps,
            'phase_id': phase_id,
        }, prefix=behavior_prefix)
        print(
            f'[phase {phase}] {behavior} episode has {length} steps, '
            f'{dual_steps} dual steps, and return {score:.1f}.')
        cost = None
        if 'cost' in ep.keys():
            cost = float(ep['cost'].astype(np.float64).sum())
            logger.add({
                'cost': cost,
            }, prefix=base_prefix)
            logger.add({
                'cost': cost,
            }, prefix=behavior_prefix)
            print(
                f'[phase {phase}] {behavior} episode has {length} steps '
                f'and cost {cost:.1f}.')
            # lag.add_cost(cost)
            if mode == 'train':
                cost_ema.value = cost_ema.value * 0.99 + cost * 0.01
                logger.add({
                    'cost_ema': cost_ema.value,
                }, prefix=base_prefix)
                if step > 5000:
                    lag.pid_update(cost_ema.value, step)    #   更新pid的参数。
        if 'arrive_dest' in ep.keys():
            if mode == 'train':
                train_arrive_num.value.append(int(ep['arrive_dest'][-1]))
                if len(train_arrive_num.value) == 10:
                    arrive_rate = sum(train_arrive_num.value) / 10
                    train_arrive_num.value = []
                    logger.add({
                        'arrive_rate': arrive_rate,
                    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
                    print(f'train 10 episodes has average arrive rate {arrive_rate:.2f}.')

            else:
                eval_arrive_num.value.append(int(ep['arrive_dest'][-1]))
                if len(eval_arrive_num.value) == 10:
                    arrive_rate = sum(eval_arrive_num.value) / 10
                    eval_arrive_num.value = []
                    logger.add({
                        'arrive_rate': arrive_rate,
                    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
                    print(f'eval 10 episodes has average arrive rate {arrive_rate:.2f}.')

        if mode == 'bootstrap_sweep':
            write_sweep_result(ep, worker, length, score, cost, behavior, dual_steps)

        if mode == 'train':
            print(
                f'[phase {phase}] [dual gate] '
                f'step={int(step)} '
                f'fea={dual_gate_status["fea"]:.4f} '
                f'fea_ema={dual_gate_status["fea_ema"]:.4f} '
                f'thresh={dual_gate_status["thresh"]:.4f} '
                f'enabled={dual_gate_status["enabled"]} '
                f'ratio={dual_gate_status["ratio"]:.3f} '
                f'real_steps={dual_gate_status["real_steps"]:.0f}')

        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f'policy_{key}'] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f'sum_{key}'] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f'mean_{key}'] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f'max_{key}'] = ep[key].max(0).mean()
        metrics.add(stats, prefix=f'{mode}_stats')

    driver_train = embodied.Driver(train_env)   # 训练环境的采样驱动器
    driver_train.on_episode(
        lambda ep, worker: per_episode(ep, mode='train', worker=worker))  #  收尾统计
    driver_train.on_step(lambda tran, _: step.increment())     # 每步增加全局step
    driver_train.on_step(train_replay.add)       # 每一步把transition加入训练经验池

    # 这四个都拿出来看

    driver_eval = embodied.Driver(eval_env)
    driver_eval.on_step(eval_replay.add)
    driver_eval.on_episode(
        lambda ep, worker: per_episode(ep, mode=eval_mode[0], worker=worker))

    if posttrain_enabled:
        # Bootstrap phase: normal data collection and policy updates only.
        train_env.call('set_danger_mining', False)
        eval_env.call('set_danger_mining', False)
        if hasattr(train_replay, 'set_danger_sampling'):
            train_replay.set_danger_sampling(False)
        print(
            '[danger posttrain] '
            f'bootstrap active for {bootstrap_episodes} training episodes')

    random_agent = embodied.RandomAgent(train_env.act_space)
    print('Prefill train dataset.')
    while len(train_replay) < max(args.batch_steps, args.train_fill):
        driver_train(random_agent.policy, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i,
                     lag_d=lag.pid_d)
    print('Prefill eval dataset.')
    while len(eval_replay) < max(args.batch_steps, args.eval_fill):
        driver_eval(random_agent.policy, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i,
                    lag_d=lag.pid_d)
    # Random prefill is not part of the requested 10k policy-training budget.
    global_episode = 0
    posttrain_pending[0] = False
    logger.add(metrics.result())
    logger.write()

    dataset_train = agent.dataset(train_replay.dataset)    # 用 train_replay 作为数据源，生成一个 训练用的批次流。
    dataset_eval = agent.dataset(eval_replay.dataset)
    state = [None]  # To be writable from train step function below.
    batch = [None]
    dual_sample_enabled = [False]
    dual_sample_ratio = [0.0]
    dual_sample_blend = [1.0]
    dual_feasible_ema = [0.0]
    dual_current_fea = [0.0]
    dual_real_steps = [0.0]

    dual_cfg = getattr(args, 'dual', {})

    def dget(name, default):
        if isinstance(dual_cfg, dict):
            return dual_cfg.get(name, default)
        return getattr(dual_cfg, name, default)

    def update_dual_sampling_gate(mets):
        enable_thresh = float(dget('enable_thresh', 0.95))
        probe_ratio = float(dget('probe_ratio', 0.05))
        full_ratio = float(dget('dual_mix_ratio', 0.3))

        if 'dual_feasible_ema' in mets:
            dual_feasible_ema[0] = float(np.asarray(mets['dual_feasible_ema']))
        if 'main_actor_fea_ratio' in mets:
            decay = float(dget('enable_decay', 0.99))
            dual_current_fea[0] = float(np.asarray(mets['main_actor_fea_ratio']))
            if 'dual_feasible_ema' not in mets:
                dual_feasible_ema[0] = (
                    decay * dual_feasible_ema[0] +
                    (1.0 - decay) * dual_current_fea[0])

        if 'dual_real_steps' in mets:
            dual_real_steps[0] = float(np.asarray(mets['dual_real_steps']))

        if dual_feasible_ema[0] > enable_thresh:
            dual_sample_enabled[0] = True
            dual_sample_ratio[0] = full_ratio if full_ratio > 0 else probe_ratio
        else:
            dual_sample_enabled[0] = False
            dual_sample_ratio[0] = 0.0

        dual_sample_blend[0] = 1.0

        dual_gate_status.update({
            'fea': float(dual_current_fea[0]),
            'fea_ema': float(dual_feasible_ema[0]),
            'thresh': enable_thresh,
            'enabled': bool(dual_sample_enabled[0]),
            'ratio': float(dual_sample_ratio[0]),
            'real_steps': float(dual_real_steps[0]),
        })



    def train_step(tran, worker):
        if posttrain_pending[0] and not posttrain_started[0]:
            return
        for _ in range(should_train(step)):
            with timer.scope('dataset_train'):
                batch[0] = next(dataset_train)            # 取数据
            outs, state[0], mets = agent.train(batch[0], state[0])
            update_dual_sampling_gate(mets)
            metrics.add(mets, prefix='train')
            if 'priority' in outs:
                train_replay.prioritize(outs['key'], outs['priority'])
                print('update priority')
            updates.increment()
        if should_sync(updates):
            agent.sync()
        if should_log(step):
            logger.add(metrics.result())
            logger.add(agent.report(batch[0]), prefix='report')
            with timer.scope('dataset_eval'):
                eval_batch = next(dataset_eval)
            logger.add(agent.report(eval_batch), prefix='eval')
            logger.add(train_replay.stats, prefix='replay')
            logger.add(eval_replay.stats, prefix='eval_replay')
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)

    driver_train.on_step(train_step)       # 每step一步都调用这个部分进行训练。

    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.train_replay = train_replay
    checkpoint.eval_replay = eval_replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    print('Start training loop.')
    policy_train = lambda *args: agent.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    policy_eval = lambda *args: agent.policy(*args, mode='eval')
    phase_marker = logdir / 'danger_posttrain_started.txt'

    def start_danger_posttraining():
        """Freeze, evaluate every base scenario, then enable the rare loop."""
        if posttrain_started[0]:
            return
        print(
            '[danger posttrain] bootstrap complete; freezing reference policy')
        freeze_mets = agent.freeze_reference_policy()
        logger.add(freeze_mets, prefix='danger_posttrain')
        checkpoint.save(logdir / f'bootstrap_{global_episode}.ckpt')

        if bool(pget('evaluate_all_scenarios', True)):
            counts = eval_env.call('get_scenario_count')
            scene_count = max((int(count) for count in counts), default=0)
            if scene_count > 0:
                sweep_result_count[0] = 0
                sweep_attempt_count[0] = 0
                sweep_seen[0].clear()
                sweep_results_path.write('', mode='w')
                print(
                    '[danger posttrain] '
                    f'evaluating frozen policy on all {scene_count} scenarios; '
                    f'results -> {sweep_results_path}')
                eval_mode[0] = 'bootstrap_sweep'
                eval_env.call('set_scenario_mode', 'sweep')
                # Mining is enabled only for this frozen-policy evaluation;
                # generated traffic remains disabled in the evaluation env.
                eval_stats = eval_env.call(
                    'set_danger_mining', True, record_events=True,
                    collect_episodes=False)
                print(f'[danger posttrain] eval mining stats: {eval_stats}')
                driver_eval.reset()
                max_attempts = scene_count * max(len(eval_env), 1) * 10
                while (
                        len(sweep_seen[0]) < scene_count and
                        sweep_attempt_count[0] < max_attempts):
                    before_attempts = sweep_attempt_count[0]
                    driver_eval(
                        policy_eval,
                        episodes=1,
                        lag=lag.lagrange_penalty,
                        lag_p=lag.delta_p,
                        lag_i=lag.pid_i,
                        lag_d=lag.pid_d)
                    if sweep_attempt_count[0] == before_attempts:
                        print(
                            '[danger posttrain] sweep made no progress; '
                            'stopping early')
                        break
                eval_env.call('set_danger_mining', False)
                eval_env.call('set_scenario_mode', 'normal')
                eval_mode[0] = 'eval'
                print(
                    '[danger posttrain] all-scene evaluation finished; '
                    f'saved {sweep_result_count[0]}/{scene_count} '
                    'unique results')
            else:
                print('[danger posttrain] no .sumocfg scenes found; skip sweep')

        # Each training worker imports the sweep JSONL into its local bounded
        # library, then starts Scenario Copy / Intent Attack and danger replay.
        train_stats = _enable_danger_posttraining(
            train_env, freeze_danger_library_after_sweep)
        print(f'[danger posttrain] train mining stats: {train_stats}')
        if hasattr(train_replay, 'set_danger_sampling'):
            train_replay.set_danger_sampling(True)
        posttrain_started[0] = True
        posttrain_pending[0] = False
        phase_marker.write(str(global_episode))
        logger.add({
            'bootstrap_episodes': global_episode,
            'all_scene_evaluation': float(pget('evaluate_all_scenarios', True)),
            'freeze_danger_library_after_sweep': float(
                freeze_danger_library_after_sweep),
        }, prefix='danger_posttrain')
        logger.write()
        print('[phase posttrain] second-round training active')

    if posttrain_enabled and phase_marker.exists():
        # A resumed post-training run keeps the reference actor embedded in
        # the checkpoint and only needs to reactivate its environment phase.
        train_stats = _enable_danger_posttraining(
            train_env, freeze_danger_library_after_sweep)
        print(f'[danger posttrain] train mining stats: {train_stats}')
        if hasattr(train_replay, 'set_danger_sampling'):
            train_replay.set_danger_sampling(True)
        posttrain_started[0] = True
        print('[danger posttrain] resumed active phase')

    while step < args.steps:

        if should_eval(step):
            print('Starting evaluation at step', int(step))
            driver_eval.reset()
            driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps), lag=lag.lagrange_penalty,
                        lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
        driver_train(
            policy_train,
            steps=100,
            lag=lag.lagrange_penalty,
            lag_p=lag.delta_p,
            lag_i=lag.pid_i,
            lag_d=lag.pid_d,
            dual_enabled=float(dual_sample_enabled[0]),
            dual_ratio=float(dual_sample_ratio[0]),
            dual_blend=float(dual_sample_blend[0]),
            dual_beta=float(dget('beta', 0.5)),
        )

        if posttrain_enabled and posttrain_pending[0]:
            start_danger_posttraining()

        if global_episode and global_episode %2000==0 :
            checkpoint.save(logdir / f'checkpoint_{int(global_episode)}.ckpt')
        if should_save(step):
            checkpoint.save()
    logger.write()
    # logger.write()
