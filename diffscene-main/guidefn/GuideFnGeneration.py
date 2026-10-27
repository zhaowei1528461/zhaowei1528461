import torch.autograd

from .helperfunction import EnvCollLoss, BvCollLoss

class Sample_fn:
    def __init__(self, traffic_model,
                 state_normalizer=None,
                 att_normalizer=None):
        self.traffic_model = traffic_model
        self.state_normalizer = state_normalizer
        self.att_normalizer = att_normalizer
        self.adv_reward_weight = 2.0
        self.eps = 1e-6
        self.attack_topk = None

    def _compute_adv_reward(self, traj, ptr):
        scene_rewards = []
        for b in range(ptr.size(0) - 1):
            s = int(ptr[b].item())
            e = int(ptr[b + 1].item())
            if e - s <= 1:
                continue

            ego_xy = traj[s:s + 1, :, :2]
            other_xy = traj[s + 1:e, :, :2]
            dists = torch.norm(other_xy - ego_xy, dim=-1)
            dists = torch.clamp(dists, min=self.eps)

            if self.attack_topk is not None and self.attack_topk > 0 and self.attack_topk < dists.size(0):
                nearest_per_agent = dists.min(dim=1)[0]
                keep_idx = torch.topk(nearest_per_agent, k=self.attack_topk, largest=False).indices
                dists = dists[keep_idx]

            scene_rewards.append(-dists.mean())

        if len(scene_rewards) == 0:
            return traj.new_tensor(0.0)
        return torch.stack(scene_rewards).mean()

    def calculate_loss(self, z, scene_graph, map_feat, past_feat,
                       map_idx=None,
                       map_env=None):
        # compute veh2veh collision
        # z: (sample_size, NA, feat)
        if self.state_normalizer is None or self.att_normalizer is None:
            print('Must have normalizers to compute collisison loss!')
            exit()
        # unnormalize
        with torch.enable_grad():
            full_grad = torch.zeros_like(z)
            veh_att = self.att_normalizer.unnormalize(scene_graph.lw)
            z = torch.unbind(z, dim=1)
            loss = []
            for i, z_i in enumerate(z):
                traj = self.traffic_model.decoder(scene_graph, map_feat, past_feat, z_i, map_idx, map_env)
                traj = self.state_normalizer.unnormalize(traj)

                # build the loss function
                #Initialize loss function
                bv_coll_loss = BvCollLoss(veh_att, scene_graph.batch, scene_graph.ptr)
                env_coll_loss = EnvCollLoss(veh_att, map_idx, map_env, traj.size(1))

                traj = traj.contiguous()
                bv_penalties, num_pairs = bv_coll_loss(traj)
                env_penalties = env_coll_loss(traj)
                adv_reward = self._compute_adv_reward(traj, scene_graph.ptr)

                total_loss = bv_penalties.sum() + env_penalties.sum() - self.adv_reward_weight * adv_reward

                if not total_loss.requires_grad:
                    total_loss = total_loss.requires_grad_(True)

                loss.append(total_loss)

                grad_i = torch.autograd.grad(total_loss, z_i, retain_graph=True)
                full_grad[:, i, :] = grad_i[0]

        return loss, full_grad

    def calculate_grad(self, loss, z):
        grad = torch.autograd.grad(loss, z, retain_graph=True)
        return grad



