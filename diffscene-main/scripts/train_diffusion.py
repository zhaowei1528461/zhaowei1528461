import csv
import os
import numpy as np
import torch
import tqdm
import torch.optim as optim
from urllib3.filepost import writer

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

import argparse
import configargparse


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
    parser.add_argument('--scenario_dir', type=str, default=None,
                        help='additional adv gen scenarios to train on')

    parser.add_argument('--data_noise_std', type=float, default=0.0, help='std of noise to add to model input data.')
    # additional data args
    parser.add_argument('--test_on_val', dest='test_on_val', action='store_true',
                        help="If given, uses the validation dataset rather than test set for evaluation.")
    parser.set_defaults(test_on_val=False)
    parser.add_argument('--shuffle_test', dest='shuffle_test', action='store_true',
                        help="If given, shuffles test dataset.")
    parser.set_defaults(shuffle_test=False)

    #
    # test options
    #

    # reconstruct (use posterior)
    parser.add_argument('--test_recon_viz_multi', dest='test_recon_viz_multi', action='store_true',
                        help="Save all-agent visualization for reconstructing all test sequences.")
    parser.set_defaults(test_recon_viz_multi=False)
    parser.add_argument('--test_recon_coll_rate', dest='test_recon_coll_rate', action='store_true',
                        help="Computes collision rate of reconstructed test trajectories")
    parser.set_defaults(test_recon_coll_rate=False)
    # sample (use prior)
    parser.add_argument('--test_sample_viz_multi', dest='test_sample_viz_multi', action='store_true',
                        help="Save all-agent visualization for samplin all test sequences.")
    parser.set_defaults(test_sample_viz_multi=False)
    parser.add_argument('--test_sample_viz_rollout', dest='test_sample_viz_rollout', action='store_true',
                        help="Create videos of multiple sampled futures individually.")
    parser.set_defaults(test_sample_viz_rollout=False)

    parser.add_argument('--test_sample_disp_err', dest='test_sample_disp_err', action='store_true',
                        help="Computes min displacement errors (ADE, FDE, and angle-based version) based on multiple samples.")
    parser.set_defaults(test_sample_disp_err=False)
    parser.add_argument('--test_sample_coll_rate', dest='test_sample_coll_rate', action='store_true',
                        help="Computes collision rate of N random samples.")
    parser.set_defaults(test_sample_coll_rate=False)

    parser.add_argument('--test_sample_num', type=int, default=3, help='Number of future traj to sample')
    parser.add_argument('--test_sample_future_len', type=int, default=None,
                        help='If not None, samples this many steps into the future rather than future_len')

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


def run_one_epoch(data_loader, diffusion_model, model_latent_space, map_env, device, train=True, optimizer=None):
    total_loss = []
    pbar = tqdm.tqdm(data_loader)
    # 模拟大批量训练
    accumulation_steps = 4  # 累积4个batch
    for i, data in enumerate(pbar):
        scene_graph, map_idx = data
        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)
        # print(scene_graph)
        # print(map_idx)
        B = map_idx.size(0)
        NA = scene_graph.past.size(0)

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
        post_mu, post_var = model_latent_space.encoder(scene_graph, map_feat, past_feat, future_feat)  # z_start

        post_mu = post_mu.to(device)
        prior_mu = prior_mu.to(device)

        # store loss value
        loss = diffusion_model.loss(post_mu, prior_mu)
        total_loss.append(loss)

        if train:
            # training step for diffusion model
            loss = loss / accumulation_steps
            diffusion_model.train_log.record_batch(loss)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

    diffusion_model.train_log._save_to_csv()
    mean_epoch_loss = np.mean([loss.item() if hasattr(loss, 'item') else loss for loss in total_loss])
    return mean_epoch_loss


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

    train_dataset = NuScenesDataset(data_path, map_env,
                            version=traffic_model_cfg.data_version,
                            split='train',
                            categories=traffic_model_cfg.agent_types,
                            npast=traffic_model_cfg.past_len,
                            nfuture=traffic_model_cfg.future_len,
                            nusc=nusc_obj,
                            noise_std=traffic_model_cfg.data_noise_std,
                            scenario_path=traffic_model_cfg.scenario_dir,
                            use_challenge_splits=traffic_model_cfg.use_challenge_splits,
                            reduce_cats=traffic_model_cfg.reduce_cats
                            )

    val_dataset = NuScenesDataset(data_path, map_env,
                            version=traffic_model_cfg.data_version,
                            split='val',
                            categories=traffic_model_cfg.agent_types,
                            npast=traffic_model_cfg.past_len,
                            nfuture=traffic_model_cfg.future_len,
                            nusc=nusc_obj,
                            use_challenge_splits=traffic_model_cfg.use_challenge_splits,
                            reduce_cats=traffic_model_cfg.reduce_cats
                            )

    # create loaders
    train_loader = GraphDataLoader(train_dataset,
                                   batch_size=traffic_model_cfg.batch_size,
                                   shuffle=traffic_model_cfg.shuffle_test,
                                   num_workers=traffic_model_cfg.num_workers,
                                   pin_memory=False,
                                   worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

    val_loader = GraphDataLoader(val_dataset,
                                    batch_size=traffic_model_cfg.batch_size,
                                    shuffle=False,
                                    num_workers=traffic_model_cfg.num_workers,
                                    pin_memory=False,
                                    worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug


    # create latent space model
    model_latent_sapce = TrafficModel(traffic_model_cfg.past_len, traffic_model_cfg.future_len, traffic_model_cfg.map_obs_size_pix, len(train_dataset.categories),
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
        traffic_model_ckpt_epoch, _ = load_state(traffic_model_cfg.ckpt, model_latent_sapce, map_location=device)
        Logger.log('Loaded checkpoint from epoch %d...' % (traffic_model_ckpt_epoch))
    else:
        throw_err('Must pass in model weights to evaluate a trained model!')

    # so can unnormalize as needed
    model_latent_sapce.set_normalizer(train_dataset.get_state_normalizer())
    model_latent_sapce.set_att_normalizer(train_dataset.get_att_normalizer())
    if traffic_model_cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model_latent_sapce.set_bicycle_params(NUSC_BIKE_PARAMS)

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

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=diffusion_model_cfg.lr, weight_decay=diffusion_model_cfg.weight_decay)

    # run training
    ckpts_path = os.path.join(diffusion_model_cfg.out, 'checkpoints')
    mkdir(ckpts_path)
    ckpt_epoch = 0
    ckpt_eval_loss = float('inf')
    # load model weights
    if diffusion_model_cfg.ckpt is not None:
        ckpt_epoch, ckpt_eval_loss = load_state(diffusion_model_cfg.ckpt, model,
                                                optimizer=optimizer,
                                                map_location=device)
        Logger.log('Loaded checkpoint from epoch %d with validation loss %f...' % (ckpt_epoch, ckpt_eval_loss))

    min_eval_loss = ckpt_eval_loss
    for epoch in range(ckpt_epoch, diffusion_model_cfg.epochs):
        Logger.log('Starting epoch %d...' % (epoch))

        model.train()
        _ = run_one_epoch(train_loader, model, model_latent_sapce, map_env, device, train=True, optimizer=optimizer)

        # validate if desired
        if epoch % diffusion_model_cfg.val_interval == 0:
            Logger.log("Validating...")
            with torch.no_grad():
                model.eval()
                mean_eval_loss = run_one_epoch(val_loader, model, model_latent_sapce, map_env, device, train=False)

                if mean_eval_loss < min_eval_loss:
                    Logger.log('Lowest eval loss so far! Saving checkpoint...')
                    min_eval_loss = mean_eval_loss
                    save_file = os.path.join(ckpts_path, 'best_eval_model.pth')
                    save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)
                torch.cuda.empty_cache()

        # save checkpoint if desired
        if epoch % diffusion_model_cfg.save_interval == 0:
            Logger.log("Saving checkpoint...")
            save_file = os.path.join(ckpts_path, 'epoch_%08d_model.pth' % (epoch))
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)
            save_file = os.path.join(ckpts_path, 'latest_model.pth')
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)

if __name__ == '__main__':
    main()


