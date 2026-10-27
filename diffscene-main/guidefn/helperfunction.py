import sys
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加上一级目录（项目根目录）到 sys.path
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
import torch
from torch import nn

import torch.nn.functional as F
from datasets.nuscenes_dataset import nutils
from utils.transforms import transform2frame

class VehCollLoss(nn.Module):
    '''
    Penalizes collision between vehicles with circle approximation.
    '''
    def __init__(self, veh_att,
                       num_circ=5,
                       buffer_dist=0.0,
                       single_veh_idx=None,
                       ptr=None
                       ):
        '''
        :param veh_att: UNNORMALIZED lw for the vehicles that will be computing loss for (NA x 2)
        :param num_circ: number of circles used to approximate each vehicle.
        :param buffer: extra buffer distance that circles must be apart to avoid being penalized
        :param single_veh_idx: only computes loss w.r.t a single vehicle at the given index within each scene graph.
                                If given must also include.ptr from scene graph as input even if only a single batch.
        :param ptr: if given, treats the traj input as a batch of scene graphs and only computes collisions between
                    vehicles in the same batch. if None, assumes a single batch
        '''
        super(VehCollLoss, self).__init__()
        self.veh_att = veh_att
        self.buffer_dist = buffer_dist
        self.single_veh_idx = single_veh_idx
        self.ptr = ptr

        NA = self.veh_att.size(0)
        # construct centroids circles of each agent
        self.veh_rad = self.veh_att[:, 1] / 2. # radius of the discs for each vehicle assuming length > width
        cent_min = -(self.veh_att[:, 0] / 2.) + self.veh_rad
        cent_max = (self.veh_att[:, 0] / 2.) - self.veh_rad
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_circ) for vidx in range(NA)], dim=0).to(veh_att.device)
        # create dummy states for centroids with y=0 and hx,hy=1,0 so can transform later
        self.centroids = torch.stack([cent_x, torch.zeros_like(cent_x), torch.ones_like(cent_x), torch.zeros_like(cent_x)], dim=2)
        self.num_circ = num_circ
        # minimum distance that two vehicle circle centers can be apart without collision
        self.penalty_dists = self.veh_rad.view(NA, 1).expand(NA, NA) + self.veh_rad.view(1, NA).expand(NA, NA) + self.buffer_dist
        # need a mask to ignore self-collisions when computing
        self.off_diag_mask = ~torch.eye(NA, dtype=torch.bool).to(self.veh_att.device)
        if self.ptr is None:
            # assume single batch
            self.ptr = torch.Tensor([0, NA]).to(self.veh_att.device).to(torch.long)
        # ignore collisiosn with other scene graphs in the batch
        batch_mask = torch.zeros((NA, NA), dtype=torch.bool).to(self.veh_att.device)
        for b in range(1, len(self.ptr)):
            # only the block corresponding to pairs of vehicles in the same scene graph matter
            batch_mask[self.ptr[b-1]:self.ptr[b], self.ptr[b-1]:self.ptr[b]] = True
        self.off_diag_mask = torch.logical_and(self.off_diag_mask, batch_mask)

        if self.single_veh_idx is not None:
            # then only want to use penalties associated with a single agent
            single_mask = torch.zeros((NA), dtype=torch.bool).to(self.off_diag_mask.device)
            single_inds = self.ptr[:-1] + self.single_veh_idx
            single_mask[single_inds] = True # target is always at index 0 of each scene graph
            single_mask_row = single_mask.view(1, NA).expand(NA, NA)
            single_mask_col = single_mask.view(NA, 1).expand(NA, NA)
            single_mask = torch.logical_or(single_mask_row, single_mask_col)
            self.off_diag_mask = torch.logical_and(self.off_diag_mask, single_mask)


    def forward(self, traj, att_inds=None, return_raw=False):
        '''
        :param traj: (N x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :param att_inds: list [N] if not using the full list of agents that was used to init the loss, which indices are you using
        :param return_raw: if True, returns the (T x N x N) penalty cost matrix and (T x N x N) mask where each True entry is a
                            colliding pair that is valid (according to settings)
        :return: loss
        '''
        NA, T, _ = traj.size()

        traj = traj[:, :, :4].view(NA*T, 4)
        cur_centroids = self.centroids
        if att_inds is not None:
            cur_centroids = cur_centroids[att_inds]
        cur_cent = cur_centroids.view(NA, 1, self.num_circ, 4).expand(NA, T, self.num_circ, 4).reshape(NA*T, self.num_circ, 4)
        # centroids are in local, need to transform to global based on current traj
        world_cent = transform2frame(traj, cur_cent, inverse=True).view(NA, T, self.num_circ, 4)[:, :, :, :2] # only need centers
        
        world_cent = world_cent.transpose(0, 1) # T x NA X C x 2
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = world_cent.view(T, NA, 1, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        cur_cent2 = world_cent.view(T, 1, NA, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*NA*NA, self.num_circ*self.num_circ)

        # get minimum distance overall all circle pairs between each pair
        min_pair_dists = torch.min(pair_dists, 1)[0].view(T, NA, NA)   
        cur_penalty_dists = self.penalty_dists
        if att_inds is not None:
            cur_penalty_dists = cur_penalty_dists[att_inds][:, att_inds]
        cur_penalty_dists = cur_penalty_dists.view(1, NA, NA)
        is_colliding_mask = min_pair_dists <= cur_penalty_dists
        # diagonals are self collisions so ignore them
        cur_off_diag_mask = self.off_diag_mask
        if att_inds is not None:
            cur_off_diag_mask = cur_off_diag_mask[att_inds][:, att_inds]
            # print(cur_off_diag_mask)
        is_colliding_mask = torch.logical_and(is_colliding_mask, cur_off_diag_mask.view(1, NA, NA))
        if not return_raw and torch.sum(is_colliding_mask) == 0:
            return torch.Tensor([0.0]).to(traj.device)
        # compute penalties
        # penalty is inverse normalized distance apart for those already colliding
        cur_penalties = 1.0 - (min_pair_dists / cur_penalty_dists)

        if return_raw:
            return cur_penalties, is_colliding_mask
        else:
            cur_penalties = cur_penalties[is_colliding_mask]
            return cur_penalties    

class BvCollLoss(nn.Module):
    '''
    penalizes collision between background vehicles
    '''
    def __init__(self, veh_att, batch, ptr, num_circ=5, buffer_dist=0.0):
        super(BvCollLoss, self).__init__()
        self.veh_att = veh_att
        self.buffer_dist = buffer_dist
        self.batch = batch
        self.ptr = ptr

        self.graph_sizes = self.ptr[1:] - self.ptr[:-1]
        self.num_pairs = torch.sum(self.graph_sizes*self.graph_sizes - self.graph_sizes)

        NA = self.veh_att.size(0)

        # construct centroids circles of each agent
        self.veh_rad = self.veh_att[:, 1] / 2.
        cent_min = -(self.veh_att[:, 0] / 2.) + self.veh_rad
        cent_max = (self.veh_att[:, 0] / 2.) - self.veh_rad
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_circ) for vidx in range(NA)], dim=0).to(veh_att.device)
        # create dummy states for centroids with y=0 and hx,hy=1,0 so can transform later
        self.centroids = torch.stack([cent_x, torch.zeros_like(cent_x), torch.ones_like(cent_x), torch.zeros_like(cent_x)], dim=2)
        self.num_circ = num_circ
        # minimum distance that two vehicle circle centers can be apart without collision
        self.penalty_dists = self.veh_rad.view(NA, 1).expand(NA, NA) + self.veh_rad.view(1, NA).expand(NA, NA) + self.buffer_dist # r_i + r_j + d_safe
        # need a mask to ignore "self" collisions and "collisions" from other scene graphs in the batch
        off_diag_mask = ~torch.eye(NA, dtype=torch.bool).to(self.veh_att.device)
        batch_mask = torch.zeros((NA, NA), dtype=torch.bool).to(self.veh_att.device)
        for b in range(1, len(self.ptr)):
            # only the block corresponding to pairs of vehicles in the same scene graph matter
            batch_mask[self.ptr[b-1]:self.ptr[b], self.ptr[b-1]:self.ptr[b]] = True
        self.valid_mask = torch.logical_and(off_diag_mask, batch_mask)

    def forward(self, traj):
        '''
        :param traj: (NA x T x 4) trajectories (x, y, hx, hy) if not use bicycle model else trajectories (x, y, heading, speed)
        '''
        NA, T, _ = traj.size()
        cur_valid_mask = self.valid_mask.view(1, NA, NA).expand(T, NA, NA)

        traj = traj[:, :, :4].view(NA*T, 4)
        cur_cent = self.centroids.view(NA, 1, self.num_circ, 4).expand(NA, T, self.num_circ, 4).reshape(NA*T, self.num_circ, 4)
        # centroids are in local, need to transform to global based on current traj
        world_cent = transform2frame(traj, cur_cent, inverse=True).view(NA, T, self.num_circ, 4)[:, :, :, :2] #only need centers
        world_cent = world_cent.transpose(0, 1) # T x NA x num_circ x 2
        # distance between all pairs of circles between all pairs of agents
        cur_cent1 = world_cent.view(T, NA, 1, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        cur_cent2 = world_cent.view(T, 1, NA, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*NA*NA, self.num_circ*self.num_circ)

        # get minimum distance overall all circle pairs between each pair
        min_pair_dists = torch.min(pair_dists, 1)[0].view(T, NA, NA)
        cur_penalty_dists = self.penalty_dists.view(1, NA, NA)
        is_colliding_mask = min_pair_dists <= cur_penalty_dists
        # diagonals are self collisions so ignore them
        is_colliding_mask = torch.logical_and(is_colliding_mask, cur_valid_mask)
        # compute penalties
        cur_penalties = torch.where(is_colliding_mask, 1.0 - (min_pair_dists / cur_penalty_dists), torch.zeros_like(cur_penalty_dists))
        cur_penalties = cur_penalties[cur_valid_mask]

        return cur_penalties, self.num_pairs

class EnvCollLoss(nn.Module):
    '''
    penalizes overlap with non-drivarble area
    '''
    def __init__(self, veh_att, mapixes, map_env, T):
        '''
        :param veh_att: (NA, 2) Unnormalized
        :param mapixes: (NA, )
        '''
        super(EnvCollLoss, self).__init__()
        self.map_env = map_env
        # loss will be applied on all timesteps, so update info accordingly
        NA = veh_att.size(0)
        if mapixes.size(0) == 1 and NA > 1:
            mapixes = mapixes.expand(NA)
        assert(NA == mapixes.size(0))
        self.mapixes = mapixes.view(NA, 1).expand(NA, T).reshape(NA*T)
        self.penalty_dists = torch.sqrt((veh_att[:, 0]**2 / 4.0) + (veh_att[:, 1]**2 / 4.0)) # max dist from center to corner of each vehicle
        self.penalty_dists = self.penalty_dists.view(NA, 1).expand(NA, T).reshape(NA*T)
        self.veh_att = veh_att.view(NA, 1, 2).expand(NA, T, 2).reshape(NA*T, 2)
        self.T = T

    def forward(self, traj):
        '''
        :param traj: (NA x T x 4) trajectories (x, y, hx, hy) if not use bicycle model else trajectories (x, y, heading, speed). should be unnormalized
        '''
        NA = traj.size(0)
        T = self.T
        assert(T == traj.size(1))
        assert(NA*T == self.veh_att.size(0))
        traj = traj.view(NA*T, 4)

        all_penalties = torch.zeros((NA*T)).to(traj.device)
        # get collisions w / non-drivable (first layer)
        drivable_raster = self.map_env.nusc_raster[:, 0]
        coll_pt = nutils.get_coll_point(drivable_raster,
                                        self.map_env.nusc_dx,
                                        traj.detach(),
                                        self.veh_att,
                                        self.mapixes)
        valid = ~torch.isnan(torch.sum(coll_pt, axis=1))
        if torch.sum(valid) == 0:
            return all_penalties.view(NA, T)

        # compute penalties
        traj_cent = traj[:,:2][valid]
        cur_dists = torch.norm(traj_cent - coll_pt[valid], dim=1)
        cur_pen_dists = self.penalty_dists[valid]
        val_penalties = 1.0 - (cur_dists / cur_pen_dists)
        # return a penalty for every time step, if not colliding is just 0
        all_penalties[valid] = val_penalties

        return all_penalties.view(NA, T)


def select_adv_agent(traj, ego_idx=0, collision_type='any', use_bicycle_model=False,
                     scene_mask=None, agent_pos=None, agent_yaw=None, world_from_agent=None,
                     return_probs=True, epsilon=1e-8):
    '''
    :param traj: (NA x T x 4) trajectories (x, y, hx, hy) if not use bicycle model else trajectories (x, y, heading, speed). should be unnormalized
    :param ego_idx: index of ego vehicle
    :param collision_type: specific collision type to target
    :param use_bicycle_model: whether using bicycle model
    :param scene_mask: mask indicating which agents are in the same scene
    :param agent_pos: current positions of all agents (NA, 2)
    :param agent_yaw: current yaw angles of all agents (NA,)
    :param world_from_agent: transformation matrices
    :param return_probs: whether to return probability distribution
    :param epsilon: small value for numerical stability

    :return: if return_probs=True: probability distribution over all agents
             else: index of the vehicle most likely to engage in a collision
    '''
    NA, T, _ = traj.shape
    device = traj.device

    # If scene_mask not provided, assume all agents are in the same scene
    if scene_mask is None:
        scene_mask = torch.ones(NA, dtype=torch.bool, device=device)

    # Extract positions from trajectories
    if use_bicycle_model:
        # For bicycle model: (x, y, heading, speed)
        positions = traj[..., :2]  # (NA, T, 2)
        headings = traj[..., 2]  # (NA, T)
    else:
        # For kinematic model: (x, y, hx, hy)
        positions = traj[..., :2]  # (NA, T, 2)
        headings = torch.atan2(traj[..., 3], traj[..., 2])  # (NA, T)

    # If agent_pos not provided, use current positions
    if agent_pos is None:
        agent_pos = positions[:, 0]  # Use first timestep positions

    # If agent_yaw not provided, use current headings
    if agent_yaw is None:
        agent_yaw = headings[:, 0]  # Use first timestep headings

    # Step 1: Compute pairwise distances (inspired by Strive paper)
    # Expand dimensions for broadcasting: (NA, 1, T, 2) and (1, NA, T, 2)
    pos_expanded_i = positions.unsqueeze(1)  # (NA, 1, T, 2)
    pos_expanded_j = positions.unsqueeze(0)  # (1, NA, T, 2)

    # Compute squared distances
    distances_sq = torch.sum((pos_expanded_i - pos_expanded_j) ** 2, dim=-1)  # (NA, NA, T)

    # Step 2: Apply scene mask and self-collision mask
    scene_mask_expanded = scene_mask.unsqueeze(1) & scene_mask.unsqueeze(0)  # (NA, NA)
    self_mask = ~torch.eye(NA, dtype=torch.bool, device=device)  # Exclude self-distances
    valid_mask = scene_mask_expanded & self_mask

    # Apply mask to distances
    masked_distances_sq = distances_sq.clone()
    masked_distances_sq[~valid_mask] = float('inf')

    # Step 3: Compute exponential negative distances (Strive method)
    exp_neg_distances = torch.exp(-masked_distances_sq + epsilon)  # (NA, NA, T)

    # Step 5: Normalize across agents (delta_t in Strive paper)
    delta_t = exp_neg_distances / (exp_neg_distances.sum(dim=1, keepdim=True) + epsilon)  # (NA, NA, T)

    # Step 6: Compute adversarial term (adv_term in Strive paper)
    min_distances_sq = masked_distances_sq.min(dim=-1, keepdim=True)[0]  # (NA, NA, 1)
    adv_term = (masked_distances_sq + min_distances_sq * 0.1) * delta_t  # (NA, NA, T)

    # Step 7: Aggregate across time and other agents
    collision_scores = adv_term.sum(dim=(1, 2))  # (NA,) - sum over other agents and time

    # Step 8: Apply collision type specific weighting
    type_weights = compute_collision_type_weights(traj, ego_idx, collision_type, use_bicycle_model)
    weighted_scores = collision_scores * type_weights

    # Step 9: Convert to probability distribution
    if return_probs:
        # Softmax to get probability distribution
        probs = F.softmax(weighted_scores, dim=0)
        return probs
    else:
        # Return the index with highest score (excluding ego if specified)
        if ego_idx is not None:
            weighted_scores[ego_idx] = -float('inf')  # Don't select ego as adversary
        adv_idx = torch.argmax(weighted_scores).item()
        return adv_idx


def compute_collision_type_weights(traj, ego_idx, collision_type, use_bicycle_model):
    """
    Compute weights based on desired collision type
    """
    NA, T, _ = traj.shape
    device = traj.device

    # Default weights (all ones)
    weights = torch.ones(NA, device=device)

    if ego_idx is None or collision_type == 'any':
        return weights

    # Extract ego and other trajectories
    ego_traj = traj[ego_idx]  # (T, 4)

    if use_bicycle_model:
        ego_heading = ego_traj[:, 2]  # (T,)
    else:
        ego_heading = torch.atan2(ego_traj[:, 3], ego_traj[:, 2])  # (T,)

    for i in range(NA):
        if i == ego_idx:
            weights[i] = 0.0  # Don't select ego
            continue

        other_traj = traj[i]

        if use_bicycle_model:
            other_heading = other_traj[:, 2]  # (T,)
        else:
            other_heading = torch.atan2(other_traj[:, 3], other_traj[:, 2])  # (T,)

        # Compute relative information
        rel_pos = other_traj[:, :2] - ego_traj[:, :2]  # (T, 2)
        rel_angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])  # (T,)

        # Angle differences
        ego_to_other_angle = rel_angle - ego_heading
        other_to_ego_angle = (rel_angle + torch.pi) - other_heading

        # Normalize angles to [-pi, pi]
        ego_to_other_angle = (ego_to_other_angle + torch.pi) % (2 * torch.pi) - torch.pi
        other_to_ego_angle = (other_to_ego_angle + torch.pi) % (2 * torch.pi) - torch.pi

        if collision_type == 'rear_end':
            # Prefer vehicles with similar heading behind ego
            heading_similarity = 1.0 - torch.abs(ego_to_other_angle).mean() / torch.pi
            is_behind = torch.abs(ego_to_other_angle).mean() > torch.pi / 2
            weights[i] = heading_similarity * (1.0 if is_behind else 0.1)

        elif collision_type == 'head_on':
            # Prefer vehicles with opposite heading
            heading_diff = torch.abs(ego_heading - other_heading).mean()
            opposition = 1.0 - torch.abs(heading_diff - torch.pi) / torch.pi
            weights[i] = opposition

        elif collision_type == 'side_impact':
            # Prefer vehicles with perpendicular headings
            heading_diff = torch.abs(ego_heading - other_heading).mean()
            perpendicularity = 1.0 - torch.abs(heading_diff - torch.pi / 2) / (torch.pi / 2)
            weights[i] = perpendicularity

        elif collision_type == 'angle':
            # Prefer specific angle collisions (30-60 degrees)
            heading_diff = torch.abs(ego_heading - other_heading).mean()
            target_angle = torch.pi / 3  # 60 degrees
            angle_match = 1.0 - torch.abs(heading_diff - target_angle) / target_angle
            weights[i] = torch.clamp(angle_match, 0, 1)

    return weights

def get_agent_idx_velocity(traj, idx, use_bicycle_model=False, dt=0.5):
    '''
    :param traj: (NA x T x 4) trajectories (x, y, hx, hy) if not use bicycle model else trajectories (x, y, heading, speed). should be unnormalized
    :param idx: vehicle position index in traj

    :return speed: the speed of the vehicle idx (T, )
    '''
    vehicle_traj = traj[idx]  # T x 4

    if use_bicycle_model:
        speed = vehicle_traj[:, 3]  # T
    else:
        positions = vehicle_traj[:, :2]  # T x 2

        displacements = positions[1:] - positions[:-1]  # (T-1) x 2
        speeds = torch.norm(displacements, dim=1) / dt  # (T-1)

        speed = torch.cat([speeds[0:1], speeds])  # T

    return speed


def get_agent_idx_acceleration(traj, idx, use_bicycle_model=False, dt=0.5):
    '''
    :param traj: (NA x T x 4) trajectories (x, y, hx, hy) if not use bicycle model else trajectories (x, y, heading, speed). should be unnormalized
    :param idx: vehicle position index in traj

    :return acceleration: the acceleration of the vehicle idx (T, )
    '''

    speed = get_agent_idx_velocity(traj, idx, use_bicycle_model, dt)  # T

    if use_bicycle_model:
        acceleration = (speed[1:] - speed[:-1]) / dt  # (T-1)
        acceleration = torch.cat([acceleration[0:1], acceleration])  # T
    else:
        positions = traj[idx, :, :2]  # T x 2

        velocity = (positions[1:] - positions[:-1]) / dt  # (T-1) x 2
        acceleration_vec = (velocity[1:] - velocity[:-1]) / dt  # (T-2) x 2
        acceleration_magnitude = torch.norm(acceleration_vec, dim=1)  # (T-2)

        acceleration = torch.zeros_like(speed)
        if len(acceleration_magnitude) > 0:
            acceleration[1:-1] = acceleration_magnitude
            acceleration[0] = acceleration_magnitude[0] if len(acceleration_magnitude) > 0 else 0
            acceleration[-1] = acceleration_magnitude[-1] if len(acceleration_magnitude) > 0 else 0

    return acceleration

def get_agent_idx_heading_angle(traj, idx, use_bicycle_model=False, dt=0.5):
    '''
    :param traj: (NA x T x 4) trajectories (x, y, hx, hy) if not use bicycle model else trajectories (x, y, heading, speed). should be unnormalized
    :param idx: vehicle position index in traj

    :return heading_angle: the heading angle of the vehicle idx (T, )
    '''
    # Extract the specific vehicle's trajectory
    vehicle_traj = traj[idx]  # (T x 4)

    if use_bicycle_model:
        # For bicycle model: (x, y, heading, speed)
        # Heading is directly provided in the trajectory
        heading_angle = vehicle_traj[:, 2]  # (T,)

    else:
        # For kinematic model: (x, y, hx, hy)
        hx, hy = vehicle_traj[:, 2], vehicle_traj[:, 3]  # (T,), (T,)
        heading_angle = torch.atan2(hy, hx)  # (T,)

    # Ensure heading angle is in the range [-pi, pi]
    heading_angle = (heading_angle + torch.pi) % (2 * torch.pi) - torch.pi

    return heading_angle


def compute_relative_position_i_j(traj, ego_idx, adv_idx, use_bicycle_model=False):
    '''
    :param traj: (NA x T x 4) trajectories (x, y, hx, hy) if not use bicycle model else trajectories (x, y, heading, speed). should be unnormalized
    :param ego_idx: the index of ego vehicle
    :param adv_idx: the index of adversarial vehicle

    :return relative_info: relative information between the ego and the adversarial
    {
        'position': relative position (x, y) (T, 2)
        'distance': Euclidean distance (T, )
        'heading_diff': heading difference (T, )
        'relative_angle': angle of relative position vector (T, )
        'closing_speed': relative speed along line of sight  (T, )
    }
    '''
    NA, T, _ = traj.size()

    # Extract trajectories for ego and adversarial vehicles
    ego_traj = traj[ego_idx]  # (T x 4)
    adv_traj = traj[adv_idx]  # (T x 4)

    # Initialize return dictionary
    relative_info = {
        'position': torch.zeros(T, 2),  # relative position (x, y)
        'distance': torch.zeros(T),  # Euclidean distance
        'heading_diff': torch.zeros(T),  # heading difference
        'relative_angle': torch.zeros(T),  # angle of relative position vector
        'closing_speed': torch.zeros(T),  # relative speed along line of sight
    }

    device = traj.device

    # Compute relative position
    relative_pos = adv_traj[:, :2] - ego_traj[:, :2]  # (T x 2)
    relative_info['position'] = relative_pos
    relative_info['distance'] = torch.norm(relative_pos, dim=1)

    # Compute heading information
    if use_bicycle_model:
        # For bicycle model: (x, y, heading, speed)
        ego_heading = ego_traj[:, 2]  # (T,)
        adv_heading = adv_traj[:, 2]  # (T,)

        # Extract speeds
        ego_speed = ego_traj[:, 3]  # (T,)
        adv_speed = adv_traj[:, 3]  # (T,)
    else:
        # For kinematic model: (x, y, hx, hy)
        ego_heading = torch.atan2(ego_traj[:, 3], ego_traj[:, 2])  # (T,)
        adv_heading = torch.atan2(adv_traj[:, 3], adv_traj[:, 2])  # (T,)

        # Compute speeds from direction vectors
        ego_speed = torch.norm(ego_traj[:, 2:4], dim=1)  # (T,)
        adv_speed = torch.norm(adv_traj[:, 2:4], dim=1)  # (T,)

    # Compute heading difference (normalized to [-pi, pi])
    heading_diff = adv_heading - ego_heading
    heading_diff = (heading_diff + torch.pi) % (2 * torch.pi) - torch.pi
    relative_info['heading_diff'] = heading_diff

    # Compute relative angle (angle of relative position vector)
    relative_angle = torch.atan2(relative_pos[:, 1], relative_pos[:, 0])
    relative_info['relative_angle'] = relative_angle

    # Compute velocity vectors
    if use_bicycle_model:
        ego_velocity = torch.stack([
            ego_speed * torch.cos(ego_heading),
            ego_speed * torch.sin(ego_heading)
        ], dim=1)  # (T x 2)

        adv_velocity = torch.stack([
            adv_speed * torch.cos(adv_heading),
            adv_speed * torch.sin(adv_heading)
        ], dim=1)  # (T x 2)
    else:
        ego_velocity = ego_traj[:, 2:4]  # (T x 2)
        adv_velocity = adv_traj[:, 2:4]  # (T x 2)

    # Compute relative velocity
    relative_velocity = adv_velocity - ego_velocity  # (T x 2)

    # Compute closing speed (relative speed along line of sight)
    relative_pos_norm = relative_pos / (relative_info['distance'].unsqueeze(1) + 1e-6)
    closing_speed = torch.sum(relative_velocity * relative_pos_norm, dim=1)
    relative_info['closing_speed'] = closing_speed

    return relative_info