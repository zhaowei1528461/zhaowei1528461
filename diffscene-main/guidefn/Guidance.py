import torch
import torch.autograd
from torch import nn

from .helperfunction import EnvCollLoss, BvCollLoss, VehCollLoss


class Sample_fn:
    def __init__(self, traffic_model,
                 state_normalizer=None,
                 att_normalizer=None,
                 adv_reward_weight=20.0,
                 veh_coll_weight=2.0,
                 env_coll_weight=5.0,
                 eps=1e-6,
                 attack_topk=None):
        self.traffic_model = traffic_model
        self.state_normalizer = state_normalizer
        self.att_normalizer = att_normalizer
        self.adv_reward_weight = adv_reward_weight
        self.veh_coll_weight = veh_coll_weight
        self.env_coll_weight = env_coll_weight
        self.eps = eps
        self.attack_topk = attack_topk

    def _compute_adv_reward(self, future_pred, scene_graph, map_idx):
        """Reward is higher when non-ego agents stay closer to ego."""
        NA = future_pred.size(0)
        B = map_idx.size(0)
        adv_crash_loss = min_dist = dist_traj = None

        # these are only computed on non-planner vehicles
        ptr = scene_graph.ptr
        graph_sizes = ptr[1:] - ptr[:-1]
        ego_mask = torch.zeros((NA), dtype=torch.bool)
        ego_mask[ptr[:-1]] = True
        # vehicle collision will only be on non-ego, so need to adjust ptr accordingly
        nonego_ptr = ptr - torch.arange(len(ptr)).to(ptr)

        # minimize POSITIONAL distance
        attacker_pred = future_pred[~ego_mask][:, 0:, :] # (NA-B, T, 4)
        # tgt_pred = future_pred[ego_mask][:, 0:, :]
        tgt_pred = self.state_normalizer.unnormalize(scene_graph.future_gt[0])
        if tgt_pred.dim() == 2:
            tgt_pred = tgt_pred.unsqueeze(0)  # [1, T, features]
        else:
            tgt_pred = tgt_pred
        
        tgt_expanded = torch.cat([tgt_pred[b:b+1, :, :].expand(graph_sizes[b]-1, -1, -1) for b in range(B)], dim=0)                
        dist_traj = torch.norm(attacker_pred[:,:,:2] - tgt_expanded[:,:,:2], dim=-1) # (NA-B, T)
        min_dist_in = dist_traj

        # soft min over all timesteps and agents
        NT = future_pred.size(1) - 0
        min_dist = [nn.functional.softmin(min_dist_in[nonego_ptr[b]:nonego_ptr[b+1]].view(-1), dim=0) for b in range(B)]
        # handle case where all frames for all agents are behind and therefore softmin is nan. (set to all 0 prob)
        min_dist = [bdist if torch.isnan(bdist[0]).item() == False else torch.zeros_like(bdist) for bdist in min_dist]

        min_dist = torch.cat(min_dist, dim=0)
        dist_traj = dist_traj.view(-1)**2 # (NA-B*T)
        weighted_adv_crash = min_dist * dist_traj # (NA-B*T)
        adv_crash_loss = [torch.sum(weighted_adv_crash[(nonego_ptr[b]*NT):(nonego_ptr[b+1]*NT)]) for b in range(B)]
        adv_crash_loss = torch.stack(adv_crash_loss)
        return adv_crash_loss

        # scene_rewards = []

        # for b in range(ptr.size(0) - 1):
        #     s = int(ptr[b].item())
        #     e = int(ptr[b + 1].item())
        #     if e - s <= 1:
        #         continue

        #     ego_xy = traj[s:s + 1, :, :2]      # 1 x T x 2
        #     other_xy = traj[s + 1:e, :, :2]    # (N-1) x T x 2
        #     dists = torch.norm(other_xy - ego_xy, dim=-1)  # (N-1) x T
        #     dists = torch.clamp(dists, min=self.eps)

        #     if self.attack_topk is not None and self.attack_topk > 0 and self.attack_topk < dists.size(0):
        #         nearest_per_agent = dists.min(dim=1)[0]
        #         keep_idx = torch.topk(nearest_per_agent, k=self.attack_topk, largest=False).indices
        #         dists = dists[keep_idx]

        #     # Smaller distance => larger reward.
        #     scene_rewards.append(-dists.mean())

        # if len(scene_rewards) == 0:
        #     return traj.new_tensor(0.0)
        # return torch.stack(scene_rewards).mean()

    def calculate_loss(self, z, scene_graph, map_feat, past_feat,
                       map_idx=None,
                       map_env=None):
        # z: (NA, sample_size, feat)
        if self.state_normalizer is None or self.att_normalizer is None:
            raise RuntimeError('Must provide state and attribute normalizers to compute guidance loss.')

        prev_training = self.traffic_model.training
        try:
            # Guidance needs gradients through the autoregressive decoder GRU.
            # CuDNN RNN backward is unavailable in eval mode, so temporarily switch
            # the traffic model to train mode while keeping inference behavior otherwise unchanged.
            self.traffic_model.train()
            with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
                full_grad = torch.zeros_like(z)
                veh_att = self.att_normalizer.unnormalize(scene_graph.lw)
                z_samples = torch.unbind(z, dim=1)
                loss = []
                B = map_idx.size(0)
                NA = scene_graph.past.size(0)

                for i, z_i in enumerate(z_samples):
                    traj = self.traffic_model.decoder(scene_graph, map_feat, past_feat, z_i, map_idx, map_env)
                    traj = self.state_normalizer.unnormalize(traj).contiguous()

                    bv_coll_loss = VehCollLoss(veh_att, buffer_dist=0.1, ptr=scene_graph.ptr)
                    env_coll_loss = EnvCollLoss(veh_att, map_idx, map_env, traj.size(1))

                    veh_coll_pens, veh_coll_mask = bv_coll_loss(traj, return_raw=True)

                    non_ego_coll_mask = torch.ones((1, NA, NA), dtype=torch.bool).to(veh_coll_mask.device) # NA x NA
                    for b in range(B):
                        non_ego_coll_mask[0, scene_graph.ptr[b], :] = False
                        non_ego_coll_mask[0, :, scene_graph.ptr[b]] = False
                    non_ego_coll_mask = torch.logical_and(veh_coll_mask, non_ego_coll_mask.expand_as(veh_coll_mask))
                    if torch.sum(non_ego_coll_mask) == 0:
                        veh_coll_loss_val = torch.Tensor([0.0]).to(veh_coll_pens.device)
                    else:
                        veh_coll_loss_val = veh_coll_pens[non_ego_coll_mask]

                    env_penalties = env_coll_loss(traj)

                    adv_reward = self._compute_adv_reward(traj, scene_graph, map_idx)
                    # Optimize by minimizing loss; subtract reward to encourage attacks.
                    total_loss = - self.veh_coll_weight * veh_coll_loss_val.mean() - self.env_coll_weight * env_penalties.mean() - self.adv_reward_weight * adv_reward.mean()

                    loss.append(total_loss)
                    grad_i = torch.autograd.grad(total_loss, z_i, retain_graph=True)[0]
                    full_grad[:, i, :] = grad_i

            return loss, full_grad
        finally:
            self.traffic_model.train(prev_training)

    def calculate_grad(self, loss, z):
        grad = torch.autograd.grad(loss, z, retain_graph=True)
        return grad
