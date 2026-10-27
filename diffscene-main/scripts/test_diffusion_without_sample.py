import os
import time

import numpy as np
import torch
import tqdm
import torch.optim as optim
from prometheus_client.samples import Sample

from guidefn.GuideFnGeneration import Sample_fn
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
from utils.scenario_gen import viz_optim_results, prepare_output_dict

import argparse
import configargparse
import matplotlib
matplotlib.use('Agg')

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

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)

    return config, config_dict

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

def run_one_epoch(data_loader, diffusion_model, model_latent_space, map_env, device, out_path,
                  guide_sample_fn = None,
                  result_dir='without_guidance',
                  viz = True,
                  save = True):
    pbar = tqdm.tqdm(data_loader)
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
    for i, data in enumerate(pbar):
        scene_graph, map_idx = data
        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)
        # print(scene_graph)
        # print(map_idx)
        B = map_idx.size(0)
        NA = scene_graph.past.size(0)
        ego_inds = scene_graph.ptr[:-1]
        ego_mask = torch.zeros((NA), dtype=torch.bool)
        ego_mask[ego_inds] = True

        scene_graph.pos = scene_graph.past[:, -1, :4]

        # extract map feature for each agent based on last frame of past
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = model_latent_space.encode_map(scene_graph, map_idx, map_env)  # NA x map_feat

        # extract past motion feature for each agent
        past_feat = model_latent_space.encode_past(scene_graph)

        # extract future motion for each agent
        future_feat = model_latent_space.encode_future(scene_graph)

        # PRIOR
        prior_mu, prior_var = model_latent_space.prior(scene_graph, map_feat, past_feat)  # cond

        # POSTERIOR
        post_mu_init, post_var_init = model_latent_space.encoder(scene_graph, map_feat, past_feat, future_feat)  # z_start

        post_mu_init = post_mu_init.to(device)
        prior_mu = prior_mu.to(device)

        shape = post_mu_init.shape
        info = {
            "scene_graph": scene_graph,
            "map_feat": map_feat,
            "past_feat": past_feat,
            "map_idx": map_idx,
            "map_env": map_env
        }

        # post_mu_pred = diffusion_model.p_sample_loop(shape,
        #                                              cond = prior_mu,
        #                                              guide_sample_fn = None,
        #                                              guidance_scale = None,
        #                                              guide_start_step = 0,
        #                                              info = info)  #without guidance


        post_mu_pred, Z = diffusion_model.p_sample_loop(shape,
                                                     cond = prior_mu,
                                                     guide_sample_fn = guide_sample_fn,
                                                     guidance_scale = None,
                                                     guide_start_step = 0,
                                                     info = info)

        traj_out_init = model_latent_space.decoder(scene_graph, map_feat, past_feat, post_mu_init, map_idx, map_env)


        traj_out =model_latent_space.decoder(scene_graph, map_feat, past_feat, post_mu_pred, map_idx, map_env)

        # output scenario and viz
        cur_sidx = 0
        scene_graph_list = scene_graph.to_data_list()

        if result_dir == 'no_diffusion':
            record_traj = traj_out_init
        elif result_dir == 'without_guidance':
            record_traj = traj_out
        else:
            record_traj = None

        # make index local to each batch idx
        if save:
            import json
            # save scenario
            cur_scene_out_path = os.path.join(gen_out_path_scenes, result_dir)
            mkdir(cur_scene_out_path)

            # prepare_output_dict(scene_graph, map_idx, map_env, dt, model,
            #                     init_fut_traj,
            #                     adv_fut_traj,
            #                     sol_fut_traj=None,
            #                     attack_agt=None,
            #                     attack_t=None,
            #                     adv_z=None,
            #                     sol_z=None,
            #                     prior_distrib=None,
            #                     attack_bike_params=None,
            #                     internal_ego_traj=None)
            scene_out_dict = prepare_output_dict(scene_graph_list[0], map_idx[0].item(), map_env,
                                                 data_loader.dataset.dt, model_latent_space,
                                                 traj_out_init,
                                                 record_traj,
                                                 adv_z=post_mu_pred,
                                                 internal_ego_traj=traj_out[0].detach())

            fout_path = os.path.join(cur_scene_out_path, 'scene_%04d.json' % i)
            Logger.log('Saving scene to %s' % (fout_path))
            with open(fout_path, 'w') as writer:
                json.dump(scene_out_dict, writer)

        if viz:
            cur_viz_out_path = os.path.join(gen_out_path_viz, result_dir)
            mkdir(cur_viz_out_path)

            # viz_optim_results(out_path, scene_graph, map_idx, map_env,
            #                       model, future_pred, planner_name, attack_agt, crop_t,
            #                       viz_bounds=[-60.0, -60.0, 60.0, 60.0],
            #                       bidx=0,
            #                       ow_gt=None,
            #                       show_gt=False,
            #                       show_gt_idx=None):

            # save before viz
            cur_crop_t = None
            cur_attack_agt = 0
            init_traj = traj_out_init
            pred_prefix = 'test_sample_%d_before' % i
            pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
            viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model_latent_space,
                              record_traj, "planner_name", cur_attack_agt,
                              cur_crop_t,
                              bidx=0,
                              show_gt=True,  # show entire nuscenes scene
                              ow_gt=None)

        batch_metric = _compute_batch_metrics(record_traj, scene_graph, map_idx, map_env, model_latent_space)
        for k in metric_cnt:
            metric_cnt[k] += batch_metric[k]

        #         # save after optimization viz
        #         pred_prefix = 'test_sample_%d_after' % (cur_batch_i[b])
        #         pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
        #         viz_optim_results(pred_out_path, scene_graph, map_idx, map_env, model,
        #                           final_result_traj, planner_name, cur_attack_agt, cur_crop_t,
        #                           bidx=b,
        #                           show_gt_idx=0,
        #                           ow_gt=final_decoder_out[
        #                               'future_pred'].clone().detach())  # show our internal pred of planner as "gt" since final_result_traj is actual planner traj
        #
        #         if adv_succeeded[b]:
        #             pred_prefix = 'test_sample_%d_sol' % (cur_batch_i[b])
        #             pred_out_path = os.path.join(cur_viz_out_path, pred_prefix)
        #             viz_optim_results(pred_out_path, sol_scene_graph, sol_map_idx, map_env, model,
        #                               sol_result_traj, planner_name, cur_attack_agt, cur_crop_t,
        #                               bidx=cur_sidx,
        #                               show_gt_idx=0,
        #                               ow_gt=final_result_traj[sol_amask][:,
        #                                     0])  # show the failed planner path pre-sol as "GT"
        #
        #     if adv_succeeded[b]:
        #         cur_sidx += 1
        #
        # # store loss value
        # loss = diffusion_model.loss(post_mu, prior_mu)
        # total_loss.append(loss)
        #
        #
        #
        #
        # if train:
        #     # training step for diffusion model
        #     loss = loss / accumulation_steps
        #     loss.backward()
        #
        #     if (i + 1) % accumulation_steps == 0:
        #         optimizer.step()
        #         optimizer.zero_grad()
        #
        #     # optimizer.zero_grad()
        #     # loss.backward()
        #     # optimizer.step()
    #
    if metric_cnt['scene_total'] > 0:
        total = float(metric_cnt['scene_total'])
        Logger.log('[%s] total_scenes = %d' % (result_dir, metric_cnt['scene_total']))
        Logger.log('[%s] ego_attack_success_rate = %.6f' % (result_dir, metric_cnt['ego_attack_success'] / total))
        Logger.log('[%s] background_collision_rate = %.6f' % (result_dir, metric_cnt['background_collision'] / total))
        Logger.log('[%s] offroad_rate = %.6f' % (result_dir, metric_cnt['offroad'] / total))


def main():

    # load config

    traffic_model_cfg, traffic_model_cfg_dict = traffic_model_parse_cfg("traffic_model")

    diffusion_model_cfg, diffusion_model_cfg_dict = diffusion_parse_cfg("train_diffusion_model")


    # device setup
    device = get_device()
    Logger.log('Using device %s...' % (str(device)))

    # load dataset
    test_dataset = map_env = None
    # first create map environment
    data_path = os.path.join(traffic_model_cfg.data_dir, traffic_model_cfg.data_version)
    map_env = NuScenesMapEnv(data_path,
                             bounds=traffic_model_cfg.map_obs_bounds,
                             L=traffic_model_cfg.map_obs_size_pix,
                             W=traffic_model_cfg.map_obs_size_pix,
                             layers=traffic_model_cfg.map_layers,
                             device=device,
                             )

    # train_dataset = NuScenesDataset(data_path, map_env,
    #                                 version=traffic_model_cfg.data_version,
    #                                 split='test' if not traffic_model_cfg.test_on_val else 'val',
    #                                 categories=traffic_model_cfg.agent_types,
    #                                 npast=traffic_model_cfg.past_len,
    #                                 nfuture=traffic_model_cfg.future_len,
    #                                 reduce_cats=traffic_model_cfg.reduce_cats,
    #                                 use_challenge_splits=traffic_model_cfg.use_challenge_splits
    #                                 )
    nusc_obj = NuScenes(version='v1.0-{}'.format(traffic_model_cfg.data_version),
                        dataroot=data_path,
                        verbose=False)

    # test_dataset = NuScenesDataset(data_path, map_env,
    #                         version=traffic_model_cfg.data_version,
    #                         split='train',
    #                         categories=traffic_model_cfg.agent_types,
    #                         npast=traffic_model_cfg.past_len,
    #                         nfuture=traffic_model_cfg.future_len,
    #                         nusc=nusc_obj,
    #                         noise_std=traffic_model_cfg.data_noise_std,
    #                         scenario_path=traffic_model_cfg.scenario_dir,
    #                         use_challenge_splits=traffic_model_cfg.use_challenge_splits,
    #                         reduce_cats=traffic_model_cfg.reduce_cats
    #                         )


    test_dataset = NuScenesDataset(data_path, map_env,
                            version=traffic_model_cfg.data_version,
                            split='test',
                            categories=traffic_model_cfg.agent_types,
                            npast=traffic_model_cfg.past_len,
                            nfuture=traffic_model_cfg.future_len,
                            reduce_cats=traffic_model_cfg.reduce_cats,
                            use_challenge_splits=traffic_model_cfg.use_challenge_splits,
                            seq_interval=traffic_model_cfg.seq_interval
                            )
    # create loaders
    test_loader = GraphDataLoader(test_dataset,
                                   batch_size=traffic_model_cfg.batch_size,
                                   shuffle=traffic_model_cfg.shuffle_test,
                                   num_workers=traffic_model_cfg.num_workers,
                                   pin_memory=False,
                                   worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug


    # create latent space model
    model_latent_space = TrafficModel(traffic_model_cfg.past_len, traffic_model_cfg.future_len, traffic_model_cfg.map_obs_size_pix, len(test_dataset.categories),
                                      map_feat_size=traffic_model_cfg.map_feat_size,
                                      past_feat_size=traffic_model_cfg.past_feat_size,
                                      future_feat_size=traffic_model_cfg.future_feat_size,
                                      latent_size=traffic_model_cfg.latent_size,
                                      output_bicycle=traffic_model_cfg.model_output_bicycle,
                                      conv_channel_in=map_env.num_layers,
                                      conv_kernel_list=traffic_model_cfg.conv_kernel_list,
                                      conv_stride_list=traffic_model_cfg.conv_stride_list,
                                      conv_filter_list=traffic_model_cfg.conv_filter_list
                                      ).to(device)

    # load pretrained latent space model
    if traffic_model_cfg.ckpt is not None:
        traffic_model_ckpt_epoch, _ = load_state(traffic_model_cfg.ckpt, model_latent_space, map_location=device)
        Logger.log('Loaded checkpoint from epoch %d...' % (traffic_model_ckpt_epoch))
    else:
        throw_err('Must pass in model weights to evaluate a trained model!')

    # so can unnormalize as needed
    model_latent_space.set_normalizer(test_dataset.get_state_normalizer())
    model_latent_space.set_att_normalizer(test_dataset.get_att_normalizer())
    if traffic_model_cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model_latent_space.set_bicycle_params(NUSC_BIKE_PARAMS)

    # create diffusion model
    net = TemporalUnet(
        horizon=diffusion_model_cfg.horizon,
        transition_dim=diffusion_model_cfg.transition_dim,
        cond_dim=diffusion_model_cfg.cond_dim,
        dim=diffusion_model_cfg.dim,
        dim_mults=diffusion_model_cfg.dim_mults,
        attention=diffusion_model_cfg.attention
        )
    model = DiffusionLatent(net=net,
                            var_sched=VarianceSchedule(
                                num_steps=diffusion_model_cfg.num_steps,
                                mode=diffusion_model_cfg.Var_schedule,
                                beta_1=diffusion_model_cfg.beta_1,
                                beta_T=diffusion_model_cfg.beta_T,
                                ),
                            observation_dim=32,
                            action_dim=32,
                            predict_epsilon=False).to(device)#observation_dim, action_dim`

    if diffusion_model_cfg.ckpt is not None:
        diffusion_model_ckpt_epoch, _ = load_state(diffusion_model_cfg.ckpt, model_latent_space, map_location=device)
        Logger.log('Loaded diffusion checkpoint from epoch %d...' % (diffusion_model_ckpt_epoch))
    else:
        throw_err('Must pass in model weights to evaluate a trained model!')

    # guide_sample_fn = Sample_fn(model_latent_space, test_dataset.get_state_normalizer(), test_dataset.get_state_normalizer())

    # run evaluations on test data
    model.eval()
    with torch.no_grad():
        start_t = time.time()
        run_one_epoch(test_loader, model, model_latent_space, map_env, device, traffic_model_cfg.out
                    , result_dir='without_guidance',
                    viz=False)
        Logger.log('Test time: %f s' % (time.time() - start_t))


if __name__ == '__main__':
    main()
