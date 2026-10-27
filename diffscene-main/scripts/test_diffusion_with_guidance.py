import os
import time
import sys
import gc

sys.path.append('../')

import numpy as np
import torch
import tqdm
import torch.distributed as dist

from torch_geometric.data import Batch as GraphBatch

import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from guidefn.Guidance import Sample_fn
from models.diffusion import VarianceSchedule, DiffusionLatent
from models.temporal import TemporalUnet
from utils.torch import get_device, count_params, load_state, save_state
from utils.logger import Logger, throw_err

from datasets import nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from nuscenes.nuscenes import NuScenes
from torch_geometric.data import DataLoader as GraphDataLoader
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.map_env import NuScenesMapEnv
from utils.common import dict2obj, mkdir
from utils.config import get_parser, add_base_args
from utils.scenario_gen import viz_optim_results, prepare_output_dict, determine_feasibility_nusc
from utils.eval import get_att_agt, compute_adv_gen_success

import argparse
import configargparse

import matplotlib
matplotlib.use('Agg')

METRIC_KEYS = (
    'scene_total',
    'ego_attack_success',
    'background_collision',
    'offroad',
)


def _get_distributed_info():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        return {
            'enabled': False,
            'rank': 0,
            'world_size': 1,
            'local_rank': 0,
        }

    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method='env://')
    return {
        'enabled': True,
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
    }


def _cleanup_distributed(dist_info):
    if dist_info['enabled'] and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(dist_info):
    return (not dist_info['enabled']) or dist_info['rank'] == 0


def _resolve_device(dist_info):
    if dist_info['enabled']:
        if torch.cuda.is_available():
            return torch.device('cuda', dist_info['local_rank'])
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return get_device()


def _reduce_metric_counts(metric_cnt, device, dist_info):
    if not dist_info['enabled']:
        return metric_cnt

    metric_tensor = torch.tensor(
        [metric_cnt[k] for k in METRIC_KEYS],
        device=device,
        dtype=torch.long,
    )
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
    return {
        key: int(metric_tensor[idx].item())
        for idx, key in enumerate(METRIC_KEYS)
    }


def _get_rank_output_path(out_path, dist_info):
    if not dist_info['enabled']:
        return out_path
    return os.path.join(out_path, 'rank_%02d' % dist_info['rank'])


def _make_scene_tag(sample_idx, scene_idx, dist_info):
    if dist_info['enabled']:
        return 'rank%02d_iter%06d_scene%02d' % (dist_info['rank'], int(sample_idx), int(scene_idx))
    return 'iter%06d_scene%02d' % (int(sample_idx), int(scene_idx))

def traffic_model_parse_cfg(description):
    '''
    Parse given config file into a config object.

    Returns: config object and config dict
    '''
    # parser = get_parser('traffic model')
    parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                                      config_file_parser_class=configargparse.YAMLConfigFileParser,
                                      description=description)

    parser.add_argument('-t_c', '--traffic_cfg',
                        required=True, is_config_file=True,
                        help='yaml config file path')

    parser.add_argument('-d_c', '--diffusion_cfg',
                        required=False, is_config_file=False,
                        help='yaml config file path')

    parser = add_base_args(parser, model_type='traffic_model')

    # additional dataset options
    parser.add_argument('--split', type=str, default='val',
                        choices=['test', 'val', 'train'],
                        help='Which split of the dataset to find scnarios in')
    parser.add_argument('--val_size', type=int, default=400,
                        help='The size of the validation set used to split the trainval version of the dataset.')
    parser.add_argument('--seq_interval', type=int, default=10,
                        help='skips ahead this many steps from start of current sequence to get the next sequence.')
    parser.add_argument('--shuffle_test', dest='shuffle_test', action='store_true',
                        help="Shuffle data")
    parser.set_defaults(shuffle_test=False)

    parser.add_argument('--adv_attack_with', type=str, default=None,
                        choices=['pedestrian', 'cyclist', 'motorcycle', 'car', 'truck'],
                        help='what to attack with (optional - by default will use any kind of agent)')

    # which planner to attack
    parser.add_argument('--planner', type=str, default='ego',
                        choices=['ego', 'hardcode'],
                        help='Which planner to attack. ego is will use ego motion from nuscenes dataset (i.e. the replay planner).')
    parser.add_argument('--planner_cfg', type=str, default='default',
                        help='hyperparameter configuration to use for the planner (if relevant)')

    # determining feasibility
    parser.add_argument('--feasibility_thresh', type=float, default=10.0,
                        help='Future samples for target must be within this many meters from another agent for the initialization scenario to be feasible.')
    parser.add_argument('--feasibility_time', type=int, default=4,
                        help='For feasibility, only consider timesteps >= feasibility_time, i.e., do not try to crash at timestep 0.')
    parser.add_argument('--feasibility_vel', type=float, default=0.5,
                        help='maximum velocity (delta position of one timestep) of sampled trajectory for an agent must be >= this thresh to be considered feasible')
    parser.add_argument('--feasibility_infront_min', type=float, default=0.0,
                        help='threshold for how in-front-of the ego vehicle the attacker is (measured by cosine similarity).')
    parser.add_argument('--feasibility_check_sep', dest='feasibility_check_sep', action='store_true',
                        help="If given, ensures attacker and target on not separated by non-drivable area.")
    parser.set_defaults(feasibility_check_sep=False)

    # optimizer & losses
    # initialization optimization
    parser.add_argument('--init_loss_match_ext', type=float, default=10.0,
                        help='Match initial trajectory from nuScenes data.')
    parser.add_argument('--init_loss_motion_prior_ext', type=float, default=0.1,
                        help='Keep latent z likely under the traffic model prior.')
    # adversarial optimization
    parser.add_argument('--loss_coll_veh', type=float, default=20.0,
                        help='Loss to avoid vehicle-vehicle collisions between non-planner agents.')
    parser.add_argument('--loss_coll_veh_plan', type=float, default=20.0,
                        help='Loss to avoid collisions between the planner and unlikely adversaries.')
    parser.add_argument('--loss_coll_env', type=float, default=20.0,
                        help='Loss to avoid vehicle-environment collisions for non-planner agents.')
    parser.add_argument('--loss_init_z', type=float, default=0.5,
                        help='Loss to keep latent z near init for unlikely adversaries (i.e. the MAX weight of init loss).')
    parser.add_argument('--loss_init_z_atk', type=float, default=0.05,
                        help='Loss to keep latent z near init for likely adversaries (i.e. the MIN weight of init loss).')
    parser.add_argument('--loss_motion_prior', type=float, default=1.0,
                        help='Loss to keep latent z likely under motion prior for unlikely adversaries (i.e. the MAX weight of prior loss).')
    parser.add_argument('--loss_motion_prior_atk', type=float, default=0.005,
                        help='Loss to keep latent z likely under motion prior for likely adversaries (i.e. the MIN weight of prior loss).')
    parser.add_argument('--loss_motion_prior_ext', type=float, default=0.0001,
                        help='Loss to keep latent z likely under motion prior for the planner.')
    parser.add_argument('--loss_match_ext', type=float, default=10.0,
                        help='Match predicted planner trajectory to true planner rollout.')
    parser.add_argument('--loss_adv_crash', type=float, default=2.0,
                        help='Minimize distance between planner and adversaries.')
    # solution optimization
    parser.add_argument('--sol_future_len', type=int, default=16,
                        help='The number of timesteps to roll out to compute collision losses for solution. If > model/data future len, will avoid irrecoverable final states for solution.')
    parser.add_argument('--sol_loss_coll_veh', type=float, default=10.0,
                        help='Loss to avoid planner-vehicle collisions.')
    parser.add_argument('--sol_loss_coll_env', type=float, default=10.0,
                        help='Loss to avoid planner-environment collisions.')
    parser.add_argument('--sol_loss_motion_prior', type=float, default=0.005,
                        help='Loss to keep planner z likely under the motion prior.')
    parser.add_argument('--sol_loss_init_z', type=float, default=0.0,
                        help='Loss to keep planner z near output of adv optim.')
    parser.add_argument('--sol_loss_motion_prior_ext', type=float, default=0.001,
                        help='Loss to keep non-planner z near output of adv optim.')
    parser.add_argument('--sol_loss_match_ext', type=float, default=10.0,
                        help='Match trajectories from output of adv optim for non-planner agents.')

    parser.add_argument('--num_iters', type=int, default=300, help='Number of optimization iterations.')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate for adam.')
    parser.add_argument('--guidance_scale', type=float, default=0.2,
                        help='Guidance scale applied in diffusion sampling.')
    parser.add_argument('--adv_reward_weight', type=float, default=2.0,
                        help='Weight of adversarial reward term that attracts other vehicles to ego.')
    parser.add_argument('--adv_attack_topk', type=int, default=0,
                        help='If > 0, only top-k closest non-ego agents receive adversarial attraction reward.')

    parser.add_argument('--viz', dest='viz', action='store_true',
                        help="If given, saves low-quality visualization before and after optimization.")
    parser.set_defaults(viz=False)
    parser.add_argument('--save', dest='save', action='store_true',
                        help="If given, saves the scenarios as json so they can be used later.")
    parser.set_defaults(save=False)

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)

    return config, config_dict


    # Losses
    # parser.add_argument('--loss_kl', type=float, default=0.004, help='KL loss weight')
    # parser.add_argument('--kl_anneal_end', type=int, default=20,
    #                     help='If given, uses KL loss annealing and will reach full weight at this epoch.')
    # parser.add_argument('--loss_recon', type=float, default=1.0, help='Reconstruction loss weight')
    # parser.add_argument('--loss_veh_coll_prior', type=float, default=0.05,
    #                     help='Vehicle collision loss weight for sample from prior')
    # parser.add_argument('--loss_env_coll_prior', type=float, default=0.1,
    #                     help='Map collision loss weight for sample from prior')


def diffusion_parse_cfg(description):
    # parser = get_parser('Train diffusion model')

    parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                                      config_file_parser_class=configargparse.YAMLConfigFileParser,
                                      description=description)

    parser.add_argument('-t_c', '--traffic_cfg',
                        required=False, is_config_file=False,
                        help='yaml config file path')

    parser.add_argument('-d_c', '--diffusion_cfg',
                        required=False, is_config_file=True,
                        help='yaml config file path')

    parser = add_base_args(parser, model_type='diffusion_model')

    # Training options
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--val_interval', type=int, default=3, help='Number of epochs between validations.')
    parser.add_argument('--save_interval', type=int, default=3, help='Number of epochs between saving model checkpoint.')
    parser.add_argument('--print_interval', type=int, default=10, help='Number of batches between printing stats.')

    # Optimizer options
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for ADAM')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay on params.')

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)

    return config, config_dict

def _compute_pairwise_collision_mask(traj, lw, num_circ=5):
    """
    Approximate box collision using multiple circle centers along vehicle length.
    :param traj: (N, T, 4) unnormalized (x, y, hx, hy)
    :param lw: (N, 2) unnormalized (length, width)
    :return: (T, N, N) bool collision mask
    """
    if traj.dim() == 4:
        traj = traj[:, 0]
    N, T, _ = traj.size()
    if N == 0:
        return torch.zeros((T, N, N), dtype=torch.bool, device=traj.device)

    veh_rad = lw[:, 1] / 2.0
    cent_min = -(lw[:, 0] / 2.0) + veh_rad
    cent_max = (lw[:, 0] / 2.0) - veh_rad
    cent_x = torch.stack([
        torch.linspace(cent_min[idx].item(), cent_max[idx].item(), num_circ, device=traj.device)
        for idx in range(N)
    ], dim=0)  # (N, C)

    hx = traj[:, :, 2].unsqueeze(-1)  # (N, T, 1)
    hy = traj[:, :, 3].unsqueeze(-1)  # (N, T, 1)
    px = traj[:, :, 0].unsqueeze(-1)
    py = traj[:, :, 1].unsqueeze(-1)
    lx = cent_x.unsqueeze(1)  # (N, 1, C)
    wx = lx * hx + px
    wy = lx * hy + py
    world_cent = torch.stack([wx, wy], dim=-1).permute(1, 0, 2, 3).contiguous()  # (T, N, C, 2)

    c1 = world_cent.view(T, N, 1, num_circ, 2).expand(T, N, N, num_circ, 2).reshape(T * N * N, num_circ, 2)
    c2 = world_cent.view(T, 1, N, num_circ, 2).expand(T, N, N, num_circ, 2).reshape(T * N * N, num_circ, 2)
    pair_dists = torch.cdist(c1, c2).view(T, N, N, num_circ, num_circ)
    min_pair_dists = pair_dists.amin(dim=(-1, -2))

    penalty_dists = veh_rad.view(1, N, 1) + veh_rad.view(1, 1, N)
    coll_mask = min_pair_dists <= penalty_dists

    valid = ~torch.isnan(traj[:, :, :4]).any(dim=-1)  # (N, T)
    valid_pair = valid.transpose(0, 1).unsqueeze(2) & valid.transpose(0, 1).unsqueeze(1)  # (T, N, N)
    coll_mask = coll_mask & valid_pair
    coll_mask[:, torch.arange(N), torch.arange(N)] = False
    return coll_mask

def _compute_batch_metrics(fut_traj_norm, scene_graph, map_idx, map_env, model):
    state_normalizer = model.get_normalizer()
    att_normalizer = model.get_att_normalizer()
    fut_traj = state_normalizer.unnormalize(fut_traj_norm.detach())
    if fut_traj.dim() == 4:
        fut_traj = fut_traj[:, 0]
    lw = att_normalizer.unnormalize(scene_graph.lw.detach())

    ptr = scene_graph.ptr
    scene_cnt = int(ptr.numel() - 1)
    ego_attack_success = 0
    background_collision = 0
    offroad = 0
    eps = 1e-6

    for b in range(scene_cnt):
        s = int(ptr[b].item())
        e = int(ptr[b + 1].item())
        traj_b = fut_traj[s:e]
        lw_b = lw[s:e]
        n = e - s
        t_steps = traj_b.size(1)

        coll_mask = _compute_pairwise_collision_mask(traj_b, lw_b)

        ego_hit = (n > 1) and bool(coll_mask[:, 0, 1:].any().item())
        ego_attack_success += int(ego_hit)

        bg_hit = False
        if n > 2:
            bg_mask = coll_mask[:, 1:, 1:]
            off_diag = ~torch.eye(n - 1, dtype=torch.bool, device=traj_b.device).unsqueeze(0)
            bg_hit = bool((bg_mask & off_diag).any().item())
        background_collision += int(bg_hit)

        cars = traj_b[:, :, :4].reshape(n * t_steps, 4)
        lw_expand = lw_b[:, None, :].expand(n, t_steps, 2).reshape(n * t_steps, 2)
        valid = ~torch.isnan(cars).any(dim=-1)
        cur_offroad = False
        if valid.any():
            mapixes = map_idx[b].view(1).expand(int(valid.sum().item()))
            driv_frac = nutils.check_on_layer(map_env.nusc_raster[:, 0], map_env.nusc_dx,
                                              cars[valid], lw_expand[valid], mapixes)
            cur_offroad = bool((driv_frac < (1.0 - eps)).any().item())
        offroad += int(cur_offroad)

    return {
        'scene_total': scene_cnt,
        'ego_attack_success': ego_attack_success,
        'background_collision': background_collision,
        'offroad': offroad,
    }


def _select_best_traj_by_adv_realism(fut_traj_norm, scene_graph, map_idx, map_env, model,
                                     adv_weight=100.0,
                                     bg_coll_weight=20.0,
                                     bg_offroad_weight=20.0,
                                     dist_weight=0.1):
    """
    Select one best sample per scene from [NA, NS, T, 4] to [NA, 1, T, 4].
    Priority:
    1) Adversariality (ego collides with non-ego)
    2) Realism (non-ego agents avoid collisions and offroad)
    3) Tie-break by closer ego distance (more adversarial potential)
    """
    if fut_traj_norm.dim() == 3:
        return fut_traj_norm.unsqueeze(1), [0] * int(scene_graph.ptr.numel() - 1)
    if fut_traj_norm.dim() != 4 or fut_traj_norm.size(1) <= 1:
        return fut_traj_norm, [0] * int(scene_graph.ptr.numel() - 1)

    state_normalizer = model.get_normalizer()
    att_normalizer = model.get_att_normalizer()
    fut_traj = state_normalizer.unnormalize(fut_traj_norm.detach())
    lw = att_normalizer.unnormalize(scene_graph.lw.detach())

    ptr = scene_graph.ptr
    scene_cnt = int(ptr.numel() - 1)
    ns = int(fut_traj_norm.size(1))
    eps = 1e-6
    best_idx = [0] * scene_cnt

    for b in range(scene_cnt):
        s = int(ptr[b].item())
        e = int(ptr[b + 1].item())
        n = e - s
        traj_scene = fut_traj[s:e]  # (n, NS, T, 4)
        lw_b = lw[s:e]
        t_steps = traj_scene.size(2)

        best_score = -float('inf')
        best_j = 0

        for j in range(ns):
            traj_b = traj_scene[:, j]  # (n, T, 4)
            coll_mask = _compute_pairwise_collision_mask(traj_b, lw_b)

            ego_hit = (n > 1) and bool(coll_mask[:, 0, 1:].any().item())

            bg_hit = False
            if n > 2:
                bg_mask = coll_mask[:, 1:, 1:]
                off_diag = ~torch.eye(n - 1, dtype=torch.bool, device=traj_b.device).unsqueeze(0)
                bg_hit = bool((bg_mask & off_diag).any().item())

            bg_offroad = False
            if n > 1:
                bg_traj = traj_b[1:]
                bg_lw = lw_b[1:]
                cars = bg_traj[:, :, :4].reshape((n - 1) * t_steps, 4)
                lw_expand = bg_lw[:, None, :].expand(n - 1, t_steps, 2).reshape((n - 1) * t_steps, 2)
                valid = ~torch.isnan(cars).any(dim=-1)
                if valid.any():
                    mapixes = map_idx[b].view(1).expand(int(valid.sum().item()))
                    driv_frac = nutils.check_on_layer(map_env.nusc_raster[:, 0], map_env.nusc_dx,
                                                      cars[valid], lw_expand[valid], mapixes)
                    bg_offroad = bool((driv_frac < (1.0 - eps)).any().item())

            ego_min_dist = float('inf')
            if n > 1:
                ego_xy = traj_b[0:1, :, :2]
                oth_xy = traj_b[1:, :, :2]
                dmat = torch.norm(oth_xy - ego_xy, dim=-1)
                finite_mask = torch.isfinite(dmat)
                if finite_mask.any():
                    ego_min_dist = float(dmat[finite_mask].amin().item())

            score = (
                adv_weight * float(ego_hit)
                - bg_coll_weight * float(bg_hit)
                - bg_offroad_weight * float(bg_offroad)
                - dist_weight * (ego_min_dist if np.isfinite(ego_min_dist) else 1e3)
            )

            if score > best_score:
                best_score = score
                best_j = j

        best_idx[b] = best_j

    selected = fut_traj_norm[:, :1].clone()
    for b in range(scene_cnt):
        s = int(ptr[b].item())
        e = int(ptr[b + 1].item())
        selected[s:e, 0] = fut_traj_norm[s:e, best_idx[b]]

    return selected, best_idx



def run_one_epoch(data_loader, batch_size, diffusion_model, model_latent_space, map_env, device, out_path,
                  guide_sample_fn = None,
                  planner_name=None,
                  guidance_scale = 0.05,
                  result_dir='with_guidance',
                  feasibility_thresh=10.0,
                  feasibility_time=4,
                  feasibility_vel=0.5,
                  feasibility_infront_min=0.0,
                  feasibility_check_sep=True,
                  viz = True,
                  save = True,
                  adv_attack_with=None,
                  dist_info=None):
    if dist_info is None:
        dist_info = {
            'enabled': False,
            'rank': 0,
            'world_size': 1,
            'local_rank': 0,
        }

    def rank_log(msg):
        if _is_main_process(dist_info):
            Logger.log(msg)

    pbar = tqdm.tqdm(data_loader, disable=not _is_main_process(dist_info))
    metric_cnt = {
        'scene_total': 0,
        'ego_attack_success': 0,
        'background_collision': 0,
        'offroad': 0,
    }

    gen_out_path = out_path
    mkdir(gen_out_path)
    if viz:
        gen_out_path_viz = os.path.join(gen_out_path, 'viz_results')
        mkdir(gen_out_path_viz)
    if save:
        gen_out_path_scenes = os.path.join(gen_out_path, 'scenario_results')
        mkdir(gen_out_path_scenes)

    empty_cache = False
    batch_i = []
    batch_scene_graph = []
    batch_map_idx = []
    batch_total_NA = 0
    for i, data in enumerate(pbar):
        sample_pred = None
        scene_graph, map_idx = data
        if empty_cache:
            empty_cache = False
            gc.collect()
            torch.cuda.empty_cache()

        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)
        is_last_batch = i == (len(data_loader)-1)
        with torch.no_grad():
            sample_pred = model_latent_space.sample_batched(scene_graph, map_idx, map_env, 20, include_mean=True)

        empty_cache = True

        feasible, feasible_time, feasible_dist = determine_feasibility_nusc(sample_pred['future_pred'],
                                                                            model_latent_space.get_normalizer(),
                                                                            feasibility_thresh,
                                                                            feasibility_time,
                                                                            0.0,
                                                                            feasibility_infront_min=feasibility_infront_min,
                                                                            check_non_drivable_separation=feasibility_check_sep,
                                                                            map_env=map_env,
                                                                            map_idx=map_idx)

        if planner_name == 'ego':
            ego_gt = model_latent_space.get_normalizer().unnormalize(scene_graph.future_gt[0])
            ego_vels = torch.norm(ego_gt[1:, :2] - ego_gt[:-1, :2], dim=-1)
            max_vel = torch.max(ego_vels).cpu().item()
            if max_vel < feasibility_vel:
                rank_log('Ego vehicle not moving more than velocity threshold, skipping...')
                if not is_last_batch:
                    continue
        elif planner_name == 'hardcode':
            ego_samps = model_latent_space.get_normalizer().unnormalize(sample_pred['future_pred'][0].detach())
            ego_vels = torch.norm(ego_samps[:, 1:, :2] - ego_samps[:, :-1, :2], dim=-1)
            max_vel = torch.max(ego_vels).cpu().item()
            if max_vel < feasibility_vel:
                rank_log('Ego samples not moving more than velocity threshold, skipping...')
                if not is_last_batch:
                    continue

        if feasible is None:
            rank_log('Only ego vehicle in scene, skipping...')
            if not is_last_batch:
                continue
        elif torch.sum(feasible).item() == 0:
            rank_log('Infeasible, no vehicles near ego, skipping...')
            if not is_last_batch:
                continue

        is_feas = False
        if feasible is not None and torch.sum(feasible).item() > 0:
            is_feas = True
            if adv_attack_with is not None:
                feas_sem = scene_graph.sem[1:]
                veclist = [tuple(feas_sem[aidx].to(int).cpu().numpy().tolist()) for aidx in range(feas_sem.size(0))]
                is_adv_atk = [data_loader.dataset.vec2cat[curvec] == adv_attack_with for curvec in veclist]
                adv_atk_feas = torch.zeros_like(feasible)
                adv_atk_feas[is_adv_atk] = True
                feasible = torch.logical_and(feasible, adv_atk_feas)
                if torch.sum(feasible) == 0:
                    rank_log('No feasible attackers of requested category, skipping...')
                    is_feas = False

            if is_feas:
                feasible_dist[~feasible] = float('inf')
                temp_attack_agt = torch.min(feasible_dist, dim=0)[1] + 1
                rank_log('Heuristic attack agt is %d' % (temp_attack_agt))
                rank_log('Heuristic attack time is %d' % (feasible_time[temp_attack_agt-1]))

        if is_feas:
            rank_log('Feasible. Adding to batch...')
            batch_scene_graph += scene_graph.to_data_list()
            batch_map_idx.append(map_idx)
            batch_i.append(i)
            batch_total_NA += scene_graph.future_gt.size(0)
            rank_log('Current batch NA: %d' % (batch_total_NA))

        if batch_total_NA < batch_size and not is_last_batch:
            continue
        else:
            if len(batch_scene_graph) == 0:
                continue
            scene_graph = GraphBatch.from_data_list(batch_scene_graph)
            map_idx = torch.cat(batch_map_idx, dim=0)
            cur_batch_i = batch_i

            rank_log('Formed batch! Starting optimization...')
            rank_log(scene_graph)

            batch_scene_graph = []
            batch_map_idx = []
            batch_i = []
            batch_total_NA = 0

        B = map_idx.size(0)
        NA = scene_graph.past.size(0)
        ego_inds = scene_graph.ptr[:-1]
        ego_mask = torch.zeros((NA), dtype=torch.bool, device=device)
        ego_mask[ego_inds] = True

        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = model_latent_space.encode_map(scene_graph, map_idx, map_env)
        past_feat = model_latent_space.encode_past(scene_graph)
        future_feat = model_latent_space.encode_future(scene_graph)

        prior_mu, prior_var = model_latent_space.prior(scene_graph, map_feat, past_feat)
        post_mu_init, post_var_init = model_latent_space.encoder(scene_graph, map_feat, past_feat, future_feat)

        post_mu_init = post_mu_init.to(device)
        prior_mu = prior_mu.to(device)

        shape = post_mu_init.shape
        info = {
            'scene_graph': scene_graph,
            'map_feat': map_feat,
            'past_feat': past_feat,
            'map_idx': map_idx,
            'map_env': map_env,
        }

        post_mu_pred = diffusion_model.p_sample_loop(shape,
                                                     cond=prior_mu,
                                                     guide_sample_fn=guide_sample_fn,
                                                     guidance_scale=guidance_scale,
                                                     guide_start_step=0,
                                                     info=info)

        traj_out_init = model_latent_space.decoder(scene_graph, map_feat, past_feat, post_mu_init, map_idx, map_env)
        traj_out = model_latent_space.decoder(scene_graph, map_feat, past_feat, post_mu_pred, map_idx, map_env)

        if planner_name == 'ego':
            traj_out[ego_inds, torch.zeros_like(ego_inds)] = scene_graph.future_gt[ego_mask][:, :, :4]

        traj_out, selected_sample_idx = _select_best_traj_by_adv_realism(
            traj_out,
            scene_graph,
            map_idx,
            map_env,
            model_latent_space,
        )
        rank_log('Selected trajectory sample per scene: %s' % (selected_sample_idx))

        tgt_traj = model_latent_space.get_normalizer().unnormalize(scene_graph.future_gt[0])
        future_pred = model_latent_space.get_normalizer().unnormalize(traj_out[:, 0])
        attack_agt, attack_t = get_att_agt(future_pred, tgt_traj, scene_graph, attack_agt_idx=None, crash_min_t=feasibility_time, crash_min_infront=feasibility_infront_min)

        adv_succeeded = []
        for b in range(B):
            cur_adv_succeeded = compute_adv_gen_success(traj_out[scene_graph.ptr[b]:scene_graph.ptr[b+1]],
                                                        model_latent_space,
                                                        GraphBatch.from_data_list([scene_graph.to_data_list()[b]]),
                                                        attack_agt[b] - scene_graph.ptr[b].item())
            adv_succeeded.append(cur_adv_succeeded)

        cur_sidx = 0
        scene_graph_list = scene_graph.to_data_list()
        for b in range(B):
            save_dir = result_dir + '_sol_failed' if adv_succeeded[b] else result_dir + '_adv_failed'
            sample_idx = cur_batch_i[b] if b < len(cur_batch_i) else (i * max(B, 1) + b)
            scene_tag = _make_scene_tag(sample_idx, b, dist_info)
            cur_attack_agt = attack_agt[b] - scene_graph.ptr[b].item()
            s = int(scene_graph.ptr[b].item())
            e = int(scene_graph.ptr[b + 1].item())

            if save:
                import json
                cur_scene_out_path = os.path.join(gen_out_path_scenes, save_dir)
                mkdir(cur_scene_out_path)

                scene_out_dict = prepare_output_dict(scene_graph_list[b], map_idx[b].item(), map_env,
                                                    data_loader.dataset.dt, model_latent_space,
                                                    traj_out_init[s:e],
                                                    traj_out[s:e],
                                                    adv_z=post_mu_pred[s:e],
                                                    prior_distrib=(prior_mu[s:e], prior_var[s:e]),
                                                    attack_agt=attack_agt[b],
                                                    internal_ego_traj=traj_out[s].detach())

                fout_path = os.path.join(cur_scene_out_path, '%s.json' % scene_tag)
                rank_log('Saving scene to %s' % (fout_path))
                with open(fout_path, 'w') as writer:
                    json.dump(scene_out_dict, writer)

            if viz:
                cur_viz_out_path = os.path.join(gen_out_path_viz, save_dir)
                mkdir(cur_viz_out_path)

                cur_crop_t = None
                pred_prefix = '%s_after' % scene_tag
                pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
                viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model_latent_space,
                                  traj_out, 'planner_name', cur_attack_agt,
                                  cur_crop_t,
                                  bidx=b,
                                  show_gt=True,
                                  ow_gt=traj_out.squeeze(1))

                if adv_succeeded[b]:
                    cur_sidx += 1

        batch_metric = _compute_batch_metrics(traj_out, scene_graph, map_idx, map_env, model_latent_space)
        for k in metric_cnt:
            metric_cnt[k] += batch_metric[k]

    metric_cnt = _reduce_metric_counts(metric_cnt, device, dist_info)
    if _is_main_process(dist_info) and metric_cnt['scene_total'] > 0:
        total = float(metric_cnt['scene_total'])
        Logger.log('[%s] total_scenes = %d' % (result_dir, metric_cnt['scene_total']))
        Logger.log('[%s] ego_attack_success_rate = %.6f' % (result_dir, metric_cnt['ego_attack_success'] / total))
        Logger.log('[%s] background_collision_rate = %.6f' % (result_dir, metric_cnt['background_collision'] / total))
        Logger.log('[%s] offroad_rate = %.6f' % (result_dir, metric_cnt['offroad'] / total))

    return metric_cnt


def main():
    dist_info = _get_distributed_info()

    try:
        traffic_model_cfg, traffic_model_cfg_dict = traffic_model_parse_cfg('traffic_model')
        diffusion_model_cfg, diffusion_model_cfg_dict = diffusion_parse_cfg('train_diffusion_model')

        output_path = _get_rank_output_path(traffic_model_cfg.out, dist_info)
        mkdir(output_path)
        Logger.init(os.path.join(output_path, 'test_diffusion_with_guidance.log'))

        device = _resolve_device(dist_info)
        Logger.log('Using device %s...' % (str(device)))
        if dist_info['enabled']:
            Logger.log('Distributed inference enabled: rank %d / %d' % (dist_info['rank'], dist_info['world_size']))

        test_dataset = map_env = None
        data_path = os.path.join(traffic_model_cfg.data_dir, traffic_model_cfg.data_version)
        map_env = NuScenesMapEnv(data_path,
                                 bounds=traffic_model_cfg.map_obs_bounds,
                                 L=traffic_model_cfg.map_obs_size_pix,
                                 W=traffic_model_cfg.map_obs_size_pix,
                                 layers=traffic_model_cfg.map_layers,
                                 device=device)

        nusc_obj = NuScenes(version='v1.0-{}'.format(traffic_model_cfg.data_version),
                            dataroot=data_path,
                            verbose=False)

        test_dataset = NuScenesDataset(data_path, map_env,
                                version=traffic_model_cfg.data_version,
                                split=traffic_model_cfg.split,
                                categories=traffic_model_cfg.agent_types,
                                npast=traffic_model_cfg.past_len,
                                nfuture=traffic_model_cfg.future_len,
                                reduce_cats=traffic_model_cfg.reduce_cats,
                                use_challenge_splits=traffic_model_cfg.use_challenge_splits,
                                seq_interval=traffic_model_cfg.seq_interval,
                                randomize_val=True,
                                val_size=traffic_model_cfg.val_size)
        sampler = None
        if dist_info['enabled']:
            sampler = DistributedSampler(
                test_dataset,
                num_replicas=dist_info['world_size'],
                rank=dist_info['rank'],
                shuffle=traffic_model_cfg.shuffle_test,
                drop_last=False,
            )

        test_loader = GraphDataLoader(test_dataset,
                                      batch_size=traffic_model_cfg.batch_size,
                                      shuffle=(traffic_model_cfg.shuffle_test if sampler is None else False),
                                      sampler=sampler,
                                      num_workers=traffic_model_cfg.num_workers,
                                      pin_memory=torch.cuda.is_available(),
                                      worker_init_fn=lambda _: np.random.seed())

        model_latent_space = TrafficModel(traffic_model_cfg.past_len, traffic_model_cfg.future_len, traffic_model_cfg.map_obs_size_pix, len(test_dataset.categories),
                                          map_feat_size=traffic_model_cfg.map_feat_size,
                                          past_feat_size=traffic_model_cfg.past_feat_size,
                                          future_feat_size=traffic_model_cfg.future_feat_size,
                                          latent_size=traffic_model_cfg.latent_size,
                                          output_bicycle=traffic_model_cfg.model_output_bicycle,
                                          conv_channel_in=map_env.num_layers,
                                          conv_kernel_list=traffic_model_cfg.conv_kernel_list,
                                          conv_stride_list=traffic_model_cfg.conv_stride_list,
                                          conv_filter_list=traffic_model_cfg.conv_filter_list).to(device)

        if traffic_model_cfg.ckpt is not None:
            traffic_model_ckpt_epoch, _ = load_state(traffic_model_cfg.ckpt, model_latent_space, map_location=device)
            Logger.log('Loaded checkpoint from epoch %d...' % (traffic_model_ckpt_epoch))
        else:
            throw_err('Must pass in model weights to evaluate a trained model!')

        model_latent_space.set_normalizer(test_dataset.get_state_normalizer())
        model_latent_space.set_att_normalizer(test_dataset.get_att_normalizer())
        if traffic_model_cfg.model_output_bicycle:
            from datasets.utils import NUSC_BIKE_PARAMS
            model_latent_space.set_bicycle_params(NUSC_BIKE_PARAMS)

        net = TemporalUnet(
            horizon=diffusion_model_cfg.horizon,
            transition_dim=diffusion_model_cfg.transition_dim,
            cond_dim=diffusion_model_cfg.cond_dim,
            dim=diffusion_model_cfg.dim,
            dim_mults=diffusion_model_cfg.dim_mults,
            attention=diffusion_model_cfg.attention)
        model = DiffusionLatent(net=net,
                                var_sched=VarianceSchedule(
                                    num_steps=diffusion_model_cfg.num_steps,
                                    mode=diffusion_model_cfg.Var_schedule,
                                    beta_1=diffusion_model_cfg.beta_1,
                                    beta_T=diffusion_model_cfg.beta_T,
                                ),
                                observation_dim=32,
                                action_dim=32,
                                predict_epsilon=False).to(device)

        if diffusion_model_cfg.ckpt is not None:
            diffusion_model_ckpt_epoch, _ = load_state(diffusion_model_cfg.ckpt, model, map_location=device)
            Logger.log('Loaded diffusion checkpoint from epoch %d...' % (diffusion_model_ckpt_epoch))
        else:
            throw_err('Must pass in model weights to evaluate a trained model!')

        attack_topk = traffic_model_cfg.adv_attack_topk if traffic_model_cfg.adv_attack_topk > 0 else None
        guide_sample_fn = Sample_fn(model_latent_space,
                                    test_dataset.get_state_normalizer(),
                                    test_dataset.get_att_normalizer(),
                                    adv_reward_weight=traffic_model_cfg.adv_reward_weight,
                                    attack_topk=attack_topk)

        model.eval()
        model_latent_space.eval()
        if sampler is not None:
            sampler.set_epoch(0)
        with torch.no_grad():
            start_t = time.time()
            run_one_epoch(test_loader, traffic_model_cfg.batch_size, model, model_latent_space, map_env, device, output_path,
                          guide_sample_fn=guide_sample_fn,
                          guidance_scale=traffic_model_cfg.guidance_scale,
                          planner_name=traffic_model_cfg.planner,
                          result_dir='with_guidance',
                          feasibility_thresh=traffic_model_cfg.feasibility_thresh,
                          feasibility_time=traffic_model_cfg.feasibility_time,
                          feasibility_vel=traffic_model_cfg.feasibility_vel,
                          feasibility_infront_min=traffic_model_cfg.feasibility_infront_min,
                          feasibility_check_sep=traffic_model_cfg.feasibility_check_sep,
                          viz=traffic_model_cfg.viz,
                          save=traffic_model_cfg.save,
                          adv_attack_with=traffic_model_cfg.adv_attack_with,
                          dist_info=dist_info)
            if dist_info['enabled']:
                dist.barrier()
            if _is_main_process(dist_info):
                Logger.log('Test time: %f s' % (time.time() - start_t))
    finally:
        _cleanup_distributed(dist_info)


if __name__ == '__main__':
    main()
