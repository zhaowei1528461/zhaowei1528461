import multiprocessing as mp
import os
import time
from collections import deque
from multiprocessing import Manager
import datetime
from typing import Tuple
from utils.opendrive2discretenet import parse_opendrive 
import pandas as pd
from numpy import ndarray
from tools1 import convert_spaces
import collections
from DreamerV3 import tools
import jax.numpy as jnp
import sys
manager = Manager()
shared_road_info = manager.dict()
import embodied
from embodied.core.basics import convert
# from embodied.run.train_eval import per_episode
# 在主程序中创建共享对象
import numpy as np
import embodied.envs.utils as u

print(f"ID of 'u' in main script: {id(u)}")
from onsite_dreamer.planner.CVaR.parameters import agent_par
import pathlib

from pathlib import Path
from . import agent as agt  # 这里是加载agent

road_info_dict_pingjie1 = {}
ref_pt1 = [] # 模块级变量
ref_pianyi = []
import jax
ROAD_INFO_FILE = Path("temp_road_info.pkl")
ZUOBIAO_INFO_FILE = Path("temp_zuobiao_info.pkl")
PIANYI_INFO_FILE = Path("temp_pianyi_info.pkl")


class AttrDict(dict):
    def __getattr__(self, item):
        v = self[item]
        if isinstance(v, dict) and not isinstance(v, AttrDict):
            v = AttrDict(v)
            self[item] = v
        return v

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]




def adapt_observation(
    state, 
    reward=0, 
    cost=0, 
    is_first=False, 
    is_terminal=False, 
    lag_p=0.0,   # lag 的比例项（可选，默认0）
    lag_i=0.0,   # lag 的积分项（可选，默认0）
    lag_d=0.0    # lag 的微分项（可选，默认0）
):
    return {
        # 状态向量：扩展批次维度（单环境 → 形状 (1, 特征数)）
        'state': np.expand_dims(np.array(state, dtype=np.float32), 0),
        # 回合生命周期标志：布尔数组，形状 (1,)
        'is_first': np.array([is_first], dtype=bool),
        'is_terminal': np.array([is_terminal], dtype=bool),
        # 奖励与成本：float32 数组，形状 (1,)
        'reward': np.array([reward], dtype=np.float32),
        'cost': np.array([cost], dtype=np.float32),
        # Lagrangian 相关参数（用于约束优化等场景）：float32 数组，形状 (1,)
        'lagrange_p': np.array([lag_p], dtype=np.float32),
        'lagrange_i': np.array([lag_i], dtype=np.float32),
        'lagrange_d': np.array([lag_d], dtype=np.float32),
        'lagrange_penalty': np.array([lag_d], dtype=np.float32),
    }





# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class PIDLagrangian():  # noqa: B024 output 为self._cost_penalty, self._pid_d, self._pid_i, self._delta_p
    """PID version of Lagrangian.

    Similar to the :class:`Lagrange` module, this module implements the PID version of the
    lagrangian method.

    .. note::
        The PID-Lagrange is more general than the Lagrange, and can be used in any policy gradient
        algorithm. As PID_Lagrange use the PID controller to control the lagrangian multiplier, it
        is more stable than the naive Lagrange.

    Args:
        pid_kp (float): The proportional gain of the PID controller.
        pid_ki (float): The integral gain of the PID controller.
        pid_kd (float): The derivative gain of the PID controller.
        pid_d_delay (int): The delay of the derivative term.
        pid_delta_p_ema_alpha (float): The exponential moving average alpha of the delta_p.
        pid_delta_d_ema_alpha (float): The exponential moving average alpha of the delta_d.
        sum_norm (bool): Whether to use the sum norm.
        diff_norm (bool): Whether to use the diff norm.
        penalty_max (int): The maximum penalty.
        lagrangian_multiplier_init (float): The initial value of the lagrangian multiplier.
        cost_limit (float): The cost limit.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        - URL: `PID Lagrange <https://arxiv.org/abs/2007.03964>`_
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
            self,
            config,
    ) -> None:
        """Initialize an instance of :class:`PIDLagrangian`."""
        self._pid_kp: float = config.pid.kp
        self._pid_ki: float = config.pid.ki
        self._pid_kd: float = config.pid.kd
        self._pid_d_delay = config.pid.d_delay
        self._pid_delta_p_ema_alpha: float = config.pid.delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = config.pid.delta_d_ema_alpha
        self._penalty_max: int = config.pid.penalty_max
        self._sum_norm: bool = config.pid.sum_norm
        self._diff_norm: bool = config.pid.diff_norm
        self._pid_i: float = config.pid.lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._pid_d: float = 0.0
        self._cost_limit: float = config.cost_limit
        self._cost_penalty: float = config.pid.init_penalty
        self._use_cost_decay: bool = config.pid.use_cost_decay
        self._current_cost_limit: float = config.pid.init_cost_limit
        if self._use_cost_decay:
            self._steps = [config.pid.decay_time_step * (i + 1) for i in range(config.pid.decay_num)]
            self._limits = [max(config.pid.init_cost_limit - i * config.pid.decay_limit_step, config.cost_limit) for i
                            in range(config.pid.decay_num)]

    @property
    def lagrange_penalty(self) -> float:
        """The lagrangian multiplier."""
        return self._cost_penalty

    @property
    def delta_p(self) -> float:
        """The lagrangian multiplier p."""
        return self._delta_p

    @property
    def pid_i(self) -> float:
        """The lagrangian multiplier i."""
        return self._pid_i

    @property
    def pid_d(self) -> float:
        """The lagrangian multiplier d."""
        return self._pid_d

    def pid_update(self, epcost, step) -> Tuple[ndarray, float, float, float]:
        r"""Update the PID controller.

        PID controller update the lagrangian multiplier following the next equation:

        .. math::

            \lambda_{t+1} = \lambda_t + (K_p e_p + K_i \int e_p dt + K_d \frac{d e_p}{d t}) \eta

        where :math:`e_p` is the error between the current episode cost and the cost limit,
        :math:`K_p`, :math:`K_i`, :math:`K_d` are the PID parameters, and :math:`\eta` is the
        learning rate.

        Args:
            ep_cost_avg (float): The average cost of the current episode.
        """
        ep_cost_avg = epcost
        if self._use_cost_decay:
            for i, threshold in enumerate(self._steps):
                if step < threshold:
                    self._current_cost_limit = self._limits[i]
                    break
            else:
                self._current_cost_limit = self._cost_limit
        else:
            self._current_cost_limit = self._cost_limit

        delta = float(ep_cost_avg - self._current_cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        self._pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * self._pid_d
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)
        self._cost_penalty = np.clip(self._cost_penalty, 0.0, self._penalty_max)
        return self._cost_penalty, self._pid_d, self._pid_i, self._delta_p


def make_logger(logdir, step, config):
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score|episode/cost'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ], multiplier)
    return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False, **kwargs):
    print('config.replay',config.replay)
    assert config.replay == 'uniform' or not rate_limit
    length = config.batch_length
    size = config.replay_size // 10 if is_eval else config.replay_size
    if config.replay == 'uniform' or is_eval:
        kw = {'online': config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
            kw['tolerance'] = 10 * config.batch_size
            kw['min_size'] = config.batch_size
        replay = embodied.replay.Uniform(length, size, directory, **kw)
    elif config.replay == 'reverb':
        replay = embodied.replay.Reverb(length, size, directory)
    elif config.replay == 'chunks':
        replay = embodied.replay.NaiveChunks(length, size, directory)
    else:
        raise NotImplementedError(config.replay)
    return replay


def load_config(argv=None):
    """
    加载和解析 SafeDreamer 的配置文件，并生成最终 args。

    参数：
        argv: list，可选，命令行参数（默认 None）

    返回：
        config: embodied.Config 对象，包含算法和环境配置
        args:   embodied.Config 对象，包含运行时参数（logdir 等）

        调用的方法：config, args = load_config(argv)
    """
    INTERNAL_DEBUG_ARGS = [
    "--configs", "osrp_onsite_lag_vector",
    "--method", "osrp_onsite_lag_vector",
    "--run.script", "train_eval",
    # "--run.from_checkpoint", "/home/hmx/ZW/onsite_dreamer_zw_62/onsite_dreamer/log_huandao_orsp/2025_10_8_1_cost_dense/checkpoint.ckpt",
    "--task", "onsite_None",
    "--jax.logical_gpus", "0",
    # "--jax.platform cpu"  ,
    "--run.steps", "10000",]
    # 可选：调试更容易断点
    # "--debug.no_jit",

    if argv is None:
        argv = sys.argv[1:]
    if  len(argv) == 0:
        argv = INTERNAL_DEBUG_ARGS[:]


    # 设置渲染后端（可选）
    os.environ["MUJOCO_GL"] = "egl"

    # 1. 解析命令行参数
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)

    # 2. 加载默认配置
    config = embodied.Config(agt.Agent.configs['defaults'])

    # 3. 根据命令行 --configs 指定的配置名叠加更新
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])

    # 4. 再解析命令行其他参数，覆盖前面配置
    config = embodied.Flags(config).parse(other)

    # 5. 生成带时间戳的 logdir
    now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir_algo = (
            config.logdir
            + now_time
            + '_' + str(config.method)
            + '_' + str(config.task)
            + '_' + str(config.seed)
    )

    logdir_algo = "/home/hmx/ZW/onsite_dreamer_zw_59/onsite_dreamer/log_huandao_orsp/2025_10_8_1_cost_dense_reward_simple_58"
    # logdir_algo = "/home/hmx/ZW/onsite_dreamer_zw_62/onsite_dreamer/log_test/2025_10_5_2_cost_dense_reward_simple"
    print('logdir_algo 222',logdir_algo)
    # 6. 封装运行参数
    args = embodied.Config(
        **config.run,
        logdir=logdir_algo,
        use_cost=config.use_cost,
        batch_steps=config.batch_size * config.batch_length,
    )
    # print('config666',config)
    return config, args, logdir_algo


class Worker_paral(mp.Process):
    def __init__(self, input_queue, output_queue, mode=None):
        mp.Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shared_road_info = shared_road_info  # 共享的字典
        self._initialize_resources()
        self.mode = mode
        print("Worker_paral __init__")

    def _initialize_resources(self):  # todo 这里需要完成config文件的加载。以及
        """一次性初始化资源密集型对象"""
        #   先加载config文件
        self.config, self.args, self.logdir_algo = load_config()   
        # print('self.config',self.config)
        # print('self.args',self.args)
        # print('self.logdir_algo',self.logdir_algo)
        self.logdir = embodied.Path(self.logdir_algo)
        logdir = embodied.Path(self.args.logdir)
        logdir.mkdirs()
        self.config.save(logdir / 'config.yaml')
        print('Logdir', logdir)    # 模型储存地址
        self.should_expl = embodied.when.Until(self.args.expl_until)
        self.should_train = embodied.when.Ratio(self.args.train_ratio / self.args.batch_steps)
        self.should_log = embodied.when.Clock(self.args.log_every)   # 定期写日志
        self.should_save = embodied.when.Clock(self.args.save_every)     # 定期存储
        self.should_eval = embodied.when.Every(self.args.eval_every, self.args.eval_initial)   # 定期评估
        self.should_sync = embodied.when.Every(self.args.sync_every)    # 定期同步参数到推理设备上
        self.step_counter = embodied.Counter() 
        self.metrics = embodied.Metrics()
        self.timer = embodied.Timer()
        # 占位符，会在init场景时填充
        self.Env = None
        self.env = None
        self.observation_space = None
        self.action_space = None
        self._acts = None 
        self.global_step=0
        self.cost_ema = 0.0          # 指数滑动平均的初值
        self.cost_alpha = 0.01
        #  Dreamer Agent (延迟到场景初始化时创建)
        self.dreamer_agent = None
        self.checkpoint = embodied.Checkpoint(self.logdir / 'checkpoint.ckpt')
        self.checkpoint.step = self.step_counter
        # 数据集
        train_eps = tools.load_episodes(self.logdir / "train_eps", 
                                    limit=self.config.get('dataset_size', 1000000))
        generator = tools.sample_episodes(train_eps, self.config.batch_length)
        self.dataset = tools.from_generator(generator, self.config.batch_size)
        self._episode_cache = train_eps  # 添加这一行
        # log
        self.logger = make_logger(self.logdir, self.step_counter , self.config)
        #   创建pidlag
        self.lag = PIDLagrangian(self.config)




    def run(self):
        # save path
        localtime = time.strftime("%Y-%m-%d_%H%M", time.localtime())
        ModelPath = os.path.dirname(os.path.realpath(
            __file__)) + '/logs/' + str(localtime) + '/model/'
        ModelPath_temp = os.path.dirname(os.path.realpath(
            __file__)) + '/logs/' + str(localtime) + '/model_temp/'
        logpath = os.path.dirname(os.path.realpath(
            __file__)) + '/logs/' + str(localtime) + '/log/'

        # self.logger = logger(path=logpath)

        self.episode = 0
        self.global_step = 0
        self.average_episode_reward = deque(maxlen=100)
        self.average_episode_cost = deque(maxlen=100)
        print("Worker_paral run")

        while True:
            k = self.input_queue.get()
            # print("get success!")
            if agent_par['train']:
                if k[0] == 'init':
                    self.init(k[1])
                    self.output_queue.put(['init success!'])
                if k[0] == 'act':
                    # print(k[1])
                    control = self.act(k[1])
                    # print('control: ',control)  #control:  [-2.083, -0.008301100884099522] control = [acc, rot]
                    self.output_queue.put(['act success!', control])
                if k[0] == 'end':
                    self.act_end()
                    self.output_queue.put(['act_end success!'])
                if k[0] == 'end_no_csv':
                    self.act_end_no_csv()
                    self.output_queue.put(['act_end_no_csv success!'])
                if k[0] == 'end work':
                    break

    def init(self, scenario_dict):
        print('scenario init')
        # 重置状态变量


        self.current_s = 0
        self.scenario_dict = scenario_dict
        self.episode_trans = []
        self.totalReward, self.LossArray = 0, []
        self.totalcost = 0
        self.step = 0
        self.state_temp = None
        self.action_temp = None
        self.episode_buffer = []
        self.episode_count = 0
        self.state = None
        self.database_train = None 
        # 创建/重置环境
        # if self.Env is None:
        from SafeDreamer.embodied.envs.utils import Sumoenv  # todo
        self.Env = Sumoenv(state_dim=agent_par['state_dim'], action_dim=agent_par['action_dim'],
                           cnn=agent_par['cnn'], safe_rule=agent_par['safe_rule'], cnn_state=agent_par['cnn_state'],
                           use_spare_reward=agent_par['use_spare_reward'], sparse_episode=agent_par['change_episode'],
                           nb_ego_states=agent_par['nb_ego_states'],
                           nb_states_per_vehicle=agent_par['nb_states_per_vehicle'],
                           sensor_nb_vehicles=agent_par['sensor_nb_vehicles'])

        self.env = self.Env
        self.env.reset(scenario_dict, parse_opendrive(scenario_dict['source_file']['xodr']))
        self.merged = getattr(self.env, 'road_info_dict_pingjie1', None)
        self.zuobiao = getattr(self.env, 'ref_pt1', None)
        self.pianyi = getattr(self.env, 'ref_pianyi', None)

        self._state = None
        self._eps = [collections.defaultdict(list)]  # 包含一个默认字典的列表


        # 保存到文件
        if self.merged:
            import pickle
            try:
                with open(ROAD_INFO_FILE, 'wb') as f:
                    pickle.dump(self.merged, f)
                # print(f"保存道路信息到文件: {ROAD_INFO_FILE}")
                # 验证文件是否正确保存
                with open(ROAD_INFO_FILE, 'rb') as f:
                    test_data = pickle.load(f)
                # print(f"验证文件保存成功，数据长度: {len(test_data)}")

            except Exception as e:
                print(f"保存文件失败: {e}")
        else:
            print("警告: merged数据为空")
        # 初始化观测空间
        if self.observation_space is None:
            self.observation_space = self.Env.observation_space
            self.action_space = self.Env.action_space
            # print('1231324action',self.action_space )
            # print('1231324obs',self.observation_space )
            # self.new_obs,self.new_action =convert_spaces(self.observation_space,self.action_space
            if self.dreamer_agent is None:

                # 填充数据的random-agent以及agent
                random_agent = embodied.RandomAgent(self.action_space )  # todo
                self.random_policy = random_agent.policy
                # 这个部分的作用调用内部policy来从state中获取动作，然后调用train（传入经验进行训练）
                
                self.agent = agt.Agent(self.observation_space, self.action_space , self.step, self.config)  
                self.policy_train = lambda *args: self.agent.policy(
                *args, mode='explore' if self.should_expl(self.global_step) else 'train')

                self.policy_eval = lambda *args: self.agent.policy(*args, mode='eval')
                self.checkpoint.agent = self.agent  # 绑定到 checkpoint
                # 创建储存经验地址                         

            #首次加载储存经验地址
                if getattr(self.args, 'from_checkpoint', None):
                    self.checkpoint.load(self.args.from_checkpoint)

            self.checkpoint.load_or_save()
            # self.should_save(self.global_step)



    def act(self, observation):
        prefill=False
        paction = None
        should_train=True
        state = self.Env.get_features(observation)
        # if self.step == 0:
        #     is_first=True
        #     self.initial_state = state
        #     obs0 = adapt_observation(self.initial_state, is_first=is_first, is_terminal=False)
        #     dummy_action = {
        #     'action': np.zeros((2,), dtype=np.float32),      # 连续动作2维：全0占位
        #     'reset':  obs0['is_terminal'] ,      # 占位的对数概率
        #     }
        #     transition0 = {
        #         'state':    obs0['state'],                     # (1, feat_dim)
        #         'is_first':  obs0['is_first'],                   # (1,) = True
        #         'is_terminal': obs0['is_terminal'],              # (1,) = False
        #         'action':    dummy_action,                       # dict( action: (2,), logprob: () )
        #         'reward':    0.0,    
        #         'cost':    0.0,                                  # 起始帧奖励固定为0
        #     }
        #     self.episode_buffer.append(transition0)
        #     self.step += 1
        # else:
        #     is_first= False

        current_s = state[0] * 100
        current_d = state[1]
        self.zuobiao = getattr(u, 'ref_pt1', None)
        if self.step==0:
            is_first=True
        else:
            is_first =False

        is_terminal = False  # 由环境 step 后判断
        obs = adapt_observation(state, is_first = is_first, is_terminal = False)
        if prefill:       # 随机时候产生的动作
             if paction is None:
                # 生成随机动作
                import random
                # raw_action = self.action_space.sample()
                a = random.uniform(-1, 1)
                b = random.uniform(-1, 1)
                raw_action = [a, b] # d v
                # 连续动作空间：假设是均匀分布
                logprob = np.log(1.0 / (1 + 1))  
                # 构建复合动作结构
                action = {
                    'action': raw_action,
                    'reset': is_terminal
                }
        else:
             # To be writable from train step function below.

            if should_train:

                # acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}  # 先获取_acts 剃出log记录部分   似乎没用？？？
                acts, self._state = self.policy_train(obs, self._state, **self._kwargs)
                # print('acts',acts['action'])
                acts['action']=np.clip(acts['action'],-1.0,1.0)
                # print('555555555555555555555555555555')

                action = {
                    'action': acts["action"][0],  # 移除batch维度
                    'reset': is_terminal  # 移除batch维度
                 }
                #从这个部分进入训练的循环
                num_train_steps = self.should_train(self.global_step) # 只调用一次
                # print('需要训练次数：', num_train_steps)
                # print('self.global_step', self.global_step)
                batch = [None]
                lan_state = [None] 
                for _ in range(num_train_steps): # 使用存储的结果
                    batch[0] = next(self.dataset)
                    # print('buffer_batch:',batch[0])
                    outs, lan_state[0], mets = self.agent.train(batch[0], lan_state[0])

                    self.metrics.add(mets, prefix='train')
                    # print('mets',mets)
                    # print('999999999999999999999999999999999999999')
                if self.global_step % 5000==0:
                    self.logger.add(self.metrics.result())
                    self.logger.add(self.agent.report(batch[0]), prefix='report')
                    self.logger.add(self.timer.stats(), prefix='timer')
                    self.logger.write(fps=True)

                if self.global_step%10000==0:
                    #保存模型
                    # ckpt_path = self.logdir / 'checkpoint.ckpt'
                    # size_before = os.path.getsize(ckpt_path) if ckpt_path.exists() else 0
                    # print(f"Checkpoint size before save: {size_before} bytes")
                    self.checkpoint.save()
                    # size_after = os.path.getsize(ckpt_path)
                    # print(f"Checkpoint size after save: {size_after} bytes")
            else:
                acts, self._state = self.policy_eval(obs, self._state, **self._kwargs)
                acts['action']=np.clip(acts['action'],-1.0,1.0)
                action = {
                    'action': acts["action"][0],  # 移除batch维度
                    'reset': is_terminal  # 移除batch维度
                 }
        trajectory, dikaer, vd, _ = self.Env.plan_trajectory_polynomial(action['action'])
        if self.current_s == 0:
            delta_s = 0
        else:
            delta_s = current_s - self.current_s
        self.current_s = current_s
        # reward,cost, done, info = self.Env.step(action['action'],vd, self.episode)#提取关键字传递得到奖励等信息。
        reward, cost,done, info = self.Env.step(action['action'],vd, self.episode, self.current_s)
        self.action_temp = action  # self.action_temp主要传递给后续act_end使用
        # self._acts = acts
        obs = adapt_observation(state,reward,cost, is_first=is_first, is_terminal=is_terminal)   #first不就一直为False了吗？
        obs['lagrange_penalty'] = self.lag.lagrange_penalty    # todo 重新取用即可
        obs['lagrange_p'] = self.lag.delta_p 
        obs['lagrange_i'] = self.lag.pid_i 
        obs['lagrange_d'] = self.lag.pid_d 
        obs = {k: convert(v) for k, v in obs.items()}  #转换数据结构
            

        #     # print('acts',acts)  #得到动作acts
        # acts = {k: convert(v) for k, v in acts.items()}     #转换格式
        #     #将当前动作给到下一step使用
        # acts['reset'] = obs['is_terminal'].copy()
        # print('222222222222222222222222222222222222',action)

        transition = {
                'state': obs['state'],
                'is_first': obs['is_first'],
                'is_terminal': obs['is_terminal'],
                'action': action,
                'reward': reward,
                'cost':  cost,
                'discount': np.array([0.0 if done else 1.0], dtype=np.float32)
        }
            # print('transition',transition)

        self.episode_buffer.append(transition)

        self.step += 1
        self.global_step+=1
        self.totalReward += reward
        self.totalcost+=cost
        self.step_counter.increment() 

            # acc = control_commands[0]
            # rot = control_commands[1]
            # return [acc, rot]  # 返回控制指令，[acceleration, rotation]
        # print(action)
        
        # print('trajectory',trajectory)
        control_commands = self.Env.execute_trajectory_control(trajectory)

        if dikaer is not None:
            # print('trajectory', trajectory)
            import pickle
            try:
                with open(PIANYI_INFO_FILE, 'wb') as f:
                    pickle.dump(dikaer, f)
                # 验证文件是否正确保存              
            except Exception as e:
                print(f"保存文件失败: {e}")
        

        acc = control_commands[0]
        rot = control_commands[1]
        # print('control:', [acc, rot])



        return [acc, rot]  # 返回控制指令，[acceleration, rotation]


    def save_episode(self, episode_buffer, episode_id):
        # 计算当前episode的step数（即episode_buffer的长度）
        episode_length = len(episode_buffer)
        if episode_length < 15:
            print(f"[skip] episode_{episode_id} length {episode_length} < 25, not saved.")
            return
        # 构建文件名，格式示例: episode_1_len55.npz
        filename = f"episode_{episode_id}_{episode_length}.npz"
        # print('save')
        # episode = {
        #     'vector': np.stack([t['vector'] for t in episode_buffer]).squeeze(1),
        #     'is_first': np.stack([t['is_first'] for t in episode_buffer]).squeeze(-1),
        #     'is_terminal': np.stack([t['is_terminal'] for t in episode_buffer]).squeeze(-1),
        #     'action': np.stack([t['action'] for t in episode_buffer]).squeeze(-1),
        #     'reward': np.stack([t['reward'] for t in episode_buffer]),
        # }
        episode = {
            'state': np.stack([t['state'] for t in episode_buffer]).squeeze(1),       # (T, 53)
            'is_first': np.stack([t['is_first'] for t in episode_buffer]).squeeze(-1),   # (T,)
            'is_terminal': np.stack([t['is_terminal'] for t in episode_buffer]).squeeze(-1), # (T,)
            'action': np.stack([t['action']['action'] for t in episode_buffer]), # (T, action_dim)
            # 'action_logprob': np.stack([t['action']['logprob'] for t in episode_buffer]), # (T,)
            'reward': np.stack([t['reward'] for t in episode_buffer]) ,       # (T,)
            'cost': np.stack([t['cost'] for t in episode_buffer]) , 
            # 'discount': np.stack([t['discount'] for t in episode_buffer]).squeeze(-1)
        }
        # for key, value in episode.items():
        #     print(f"{key} shape:", value.shape)
            # vector shape: (51, 1, 53)
            # is_first shape: (51, 1)
            # is_terminal shape: (51, 1)
            # action shape: (51, 1)
            # action_logprob shape: (51,)
            # reward shape: (51,)
        save_dir = self.logdir / "train_eps"
        # print('save_dir',save_dir)
        save_dir_str = str(save_dir)
        Path(save_dir_str).mkdir(parents=True, exist_ok=True)
        # save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_dir / filename, **episode)
        # 保存后强制刷新 dataset
        # train_eps = tools.load_episodes(self.logdir / "train_eps", limit=self.config.get('dataset_size', 1000000))
        # generator = tools.sample_episodes(train_eps, self.config.batch_length)
        # self.dataset = tools.from_generator(generator, self.config.batch_size)
        # if self.dreamer_agent is not None:
        #     self.dreamer_agent._dataset = self.dataset
            # print("刷新了 agent 的 dataset，当前 episode 文件数：", len(train_eps))
            # 增量更新数据集，而不是完全重新加载
        if hasattr(self, '_episode_cache') and self._episode_cache is not None:
            # 将新episode添加到缓存中
            episode_key = f"episode_{self.episode}_{self.step}"
            # 将新episode添加到缓存字典中
            self._episode_cache[episode_key] = episode
            # self._episode_cache.append(episode)
            print('nonono')
            # 如果缓存过大，移除最旧的episode
            if len(self._episode_cache) > self.config.get('dataset_size', 1000000):
                self._episode_cache.pop(0)
            
            # 更新数据集
            generator = tools.sample_episodes(self._episode_cache, self.config.batch_length)
            self.dataset = tools.from_generator(generator, self.config.batch_size)
            
            if self.dreamer_agent is not None:
                self.dreamer_agent._dataset = self.dataset
        else:
            # 首次加载或缓存丢失时，完整加载
            self._refresh_dataset()


    def _refresh_dataset(self):
        """完整刷新数据集，仅在必要时调用"""
        train_eps = tools.load_episodes(self.logdir / "train_eps", limit=self.config.get('dataset_size', 1000000))
        self._episode_cache = train_eps  # 保存到缓存
        generator = tools.sample_episodes(train_eps, self.config.batch_length)
        self.dataset = tools.from_generator(generator, self.config.batch_size)
        if self.dreamer_agent is not None:
            self.dreamer_agent._dataset = self.dataset


    def act_end(self):
        # print(self.Env.scenario_output_path)
        observation = self.csv_observation_end(self.Env.scenario_output_path)
        state = self.Env._get_features_end(observation)
        if self.action_temp is None:
            self.action_temp = {
                        'action': [0, 0],  # 移除batch维度
                        'reset': True # 移除batch维度
                        }
        # print(self.action_temp)
        trajectory, dikaer, vd, _ = self.Env.plan_trajectory_polynomial(self.action_temp['action'])
        reward, cost,done, info = self.Env.step(self.action_temp['action'], vd,self.episode,self.current_s)
        is_first = False
        is_terminal = True
        obs = adapt_observation(state, reward,cost,is_first, is_terminal)
        action = self.action_temp

        last_transition = {
            'state': obs['state'],
            'is_first': obs['is_first'],
            'is_terminal': obs['is_terminal'],
            'action': action,
            'reward': reward,
            'cost': cost,
        }


        done = 1
        done_store = 0
        if done:
            done_store = 1
        info_store = 0
        if info == "collision":
            info_store = 3
        if info == "goal":
            info_store = 1
        if info == "map_out_range":
            info_store = 4
        if info == "max_t":
            info_store = 2
        self.state_temp = state
        self.totalReward += reward
        self.totalcost+=cost
        self.step += 1
        self.global_step += 1
        self.step_counter.increment() 
        self.average_episode_reward.append(self.totalReward)
        self.average_episode_cost.append(self.totalcost)
            # 更新 EMA
        alpha = 0.01
        self.cost_ema = (1 - alpha) * self.cost_ema + alpha * self.totalcost
        # print('totoal_cost:',self.totalcost)
        # print('ema_cost',self.cost_ema)
        print('Episode: {} | reward: {:.2f}|  cost: {:.2f}  | time: {} | info: {} '.format(self.episode, self.totalReward,self.totalcost, self.step,
                                                                           info))
        # ji lu log
        # print('self.step_counter',self.step_counter)
        if self.global_step > 5000:  # 训练一定步数后启动
            # print('in step:{}in pid_update cost={} ',self.step_counter,self.cost_ema)
            self.lag.pid_update(self.cost_ema, self.step_counter)
        #回合分数cost的记录
        self.logger.add({
        'reward': self.totalReward,      # 或者写成 'score'，按你习惯
        'cost': self.totalcost,
        'cost_ema': self.cost_ema,
        }, prefix='episode')

        self.del_files_csv("./outputs")
        self.episode_buffer.append(last_transition)
        # print('self.episode_buffer',self.episode_buffer)
         # 保存 episode


        self.save_episode(self.episode_buffer, self.episode)
        # print('self.episode_buffer',self.episode_buffer)
        self.episode_buffer = []
        self.episode_count += 1
        self.episode += 1


    def del_files_csv(self, path_file):
        ls = os.listdir(path_file)
        for i in ls:
            if str(i.split('.')[-1]) == 'csv':
                f_path = os.path.join(path_file, i)
                # 判断是否是一个目录,若是,则递归删除
                if os.path.isdir(f_path):
                    self.del_files_csv(f_path)
                else:
                    os.remove(f_path)


    def csv_observation_end(self, path_file):
        df = pd.read_csv(path_file, encoding="utf-8")
        # 获取最后一行数据
        last_row = df.tail(1)
        # 获取所有列索引
        column_indexes = df.columns
        object_len_total = (len(column_indexes) - 3 - 8 - 1) / 7  # npc数量

        end = last_row.iloc[0, -1]
        t = last_row.iloc[0, 0]
        # acc = last_row.iloc[0, 1]
        # rot = last_row.iloc[0, 2]
        # x_ego, y_ego, v_ego, a_eg, yaw_ego, rot_ego, width_ego, length_ego = last_row.iloc[0, 3], last_row.iloc[0, 4], last_row.iloc[0, 5], last_row.iloc[0, 6], last_row.iloc[0, 7], last_row.iloc[0, 8], last_row.iloc[0, 9], last_row.iloc[0, 10]
        # object_need = []
        # object_need_name = []
        # for i in range(int(object_len_total)):
        #     temp = []
        #     for j in range(7):
        #         temp.append(last_row.iloc[0, 11+7*i+j])
        #     if str(temp[0]) != 'nan':
        #         object_need.append(temp)
        #         object_need_name.append(column_indexes[11+7*i+0].split('_')[-1])

        observation = Observation_need()
        for i in range(8):
            observation.ego_info[str(str(column_indexes[3 + i]).split('_')[0])] = last_row.iloc[0, 3 + i]

        for i in range(int(object_len_total)):
            temp = {}
            if str(last_row.iloc[0, 11 + 7 * i + 0]) != 'nan':
                for j in range(7):
                    temp[str(str(column_indexes[11 + 7 * i + j]).split('_')[0])] = last_row.iloc[0, 11 + 7 * i + j]
            if len(temp) != 0:
                observation.object_info[str(str(column_indexes[11 + 7 * i + j]).split('_')[-1])] = temp

        observation.test_info['t'] = t
        observation.test_info['end'] = end
        return observation



def get_road_info_dict():
    """从文件读取道路信息"""
    # print(f"get_road_info_dict 调用，进程ID: {os.getpid()}")
    import pickle
    try:
        # print('ROAD_INFO_FILE',ROAD_INFO_FILE)
        if ROAD_INFO_FILE.exists():
            with open(ROAD_INFO_FILE, 'rb') as f:
                data = pickle.load(f)
            # print(f"1491, 从文件读取数据长度: {len(data)}")
            return data
        else:
            print(f"1491, 文件不存在: {ROAD_INFO_FILE}")
            return {}
    except Exception as e:
        print(f"1491, 读取文件失败: {e}")
        return {}
def get_zuobiao_info_dict():
    """从文件读取道路信息"""
    # print(f"get_road_info_dict 调用，进程ID: {os.getpid()}")
    import pickle
    try:
        if ZUOBIAO_INFO_FILE.exists():
            with open(ZUOBIAO_INFO_FILE, 'rb') as f:
                data1 = pickle.load(f)
            # print(f"542, 从文件读取数据长度: {len(data1)}")
            return data1
        else:
            print(f"1491, 文件不存在: {ROAD_INFO_FILE}")
            return {}
    except Exception as e:
        print(f"1491, 读取文件失败: {e}")
        return {}
def get_pianyi_info_dict():
    """从文件读取道路信息"""
    # print(f"get_road_info_dict 调用，进程ID: {os.getpid()}")
    import pickle
    try:
        if PIANYI_INFO_FILE.exists():
            with open(PIANYI_INFO_FILE, 'rb') as f:
                data2 = pickle.load(f)
            # print(f"542, 从文件读取数据长度: {len(data1)}")
            return data2
        else:
            print(f"1491, 文件不存在: {ROAD_INFO_FILE}")
            return {}
    except Exception as e:
        print(f"1491, 读取文件失败: {e}")
        return {}


class Observation_need:
    def __init__(self):
        self.ego_info = {}
        self.object_info = {}
        self.light_info = {}
        self.test_info = {}
