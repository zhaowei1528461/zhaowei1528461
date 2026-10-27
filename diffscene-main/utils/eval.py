import numpy as np
import torch
from torch import nn

ENV_COLL_THRESH = 0.05 # up to 5% of vehicle can be off the road
VEH_COLL_THRESH = 0.02 # IoU must be over this to count as a collision for metric (not loss)

def interp_traj(future_pred, scale_factor=3):
    '''
    :param future_pred (NA x T x 4) or (NA x NS x T x 4)
    '''
    mult_samp = False
    if len(future_pred.size()) == 4:
        mult_samp = True
        NA, NS, T, _ = future_pred.size()
        future_pred = future_pred.reshape(NA*NS, T, 4)
    future_pred_interp = nn.functional.interpolate(future_pred.transpose(1,2),
                                                        scale_factor=scale_factor,
                                                        mode='linear').transpose(1,2)
    # normalize heading (NOTE: interp hx and hy is not exactly correct, but close)
    interp_pos = future_pred_interp[:, :, :2]
    interp_h = future_pred_interp[:, :, 2:4]
    interp_h = interp_h / torch.norm(interp_h, dim=-1, keepdim=True)
    future_pred_interp = torch.cat([interp_pos, interp_h], dim=-1)
    if mult_samp:
        future_pred_interp = future_pred_interp.reshape(NA, NS, future_pred_interp.size(1), 4)
    return future_pred_interp

def check_single_veh_coll(traj_tgt, lw_tgt, traj_others, lw_others):
    '''
    Checks if the target trajectory collides with each of the given other trajectories.

    Assumes all trajectories and attributes are UNNORMALIZED. Handles nan frames in traj_others by simply skipping.

    :param traj_tgt: (T x 4)
    :param lw_tgt: (2, )
    :param traj_others: (N x T x 4)
    :param lw_others: (N x 2)

    :returns veh_coll: (N)
    :returns coll_time: (N)
    '''
    import datasets.nuscenes_utils as nutils
    from shapely.geometry import Polygon

    NA, FT, _ = traj_others.size()
    traj_tgt = traj_tgt.cpu().numpy()
    lw_tgt = lw_tgt.cpu().numpy()
    traj_others = traj_others.cpu().numpy()
    lw_others = lw_others.cpu().numpy()

    veh_coll = np.zeros((NA), dtype=bool)
    coll_time = np.ones((NA), dtype=int)*FT
    poly_cache = dict() # for the tgt polygons since used many times
    for aj in range(NA):
        for t in range(FT):
            # compute iou
            if t not in poly_cache:
                ai_state = traj_tgt[t, :]
                ai_corners = nutils.get_corners(ai_state, lw_tgt)
                ai_poly = Polygon(ai_corners)
                poly_cache[t] = ai_poly
            else:
                ai_poly = poly_cache[t]

            aj_state = traj_others[aj, t, :]
            if np.sum(np.isnan(aj_state)) > 0:
                continue
            aj_corners = nutils.get_corners(aj_state, lw_others[aj])
            aj_poly = Polygon(aj_corners)
            cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
            if cur_iou > VEH_COLL_THRESH:
                veh_coll[aj] = True
                coll_time[aj] = t
                break # don't need to check rest of sequence

    return veh_coll, coll_time


def compute_adv_gen_success(final_result_traj, model, scene_graph, attack_agt):
    '''
    Computes whether scenario is successful in colliding w/ planner
    All inputs assumed NORMALIZED.
    :param final_result_traj: (NA, 1, FT, 4) the final scenario where the agent at idx=0 is the true planner
                                            reaction to the scenario (NOT the model's prediction of the planner)
    '''

    # final result trajectories
    planner_gt_fut = model.get_normalizer().unnormalize(final_result_traj[0, 0])
    other_fut = model.get_normalizer().unnormalize(final_result_traj[1:, 0])
    # agent attribs
    planner_lw = model.get_att_normalizer().unnormalize(scene_graph.lw[0])
    other_lw = model.get_att_normalizer().unnormalize(scene_graph.lw[1:])

    # Collision with planner target
    planner_coll_all, planner_coll_time = check_single_veh_coll(planner_gt_fut, planner_lw, other_fut, other_lw)
    attack_coll = planner_coll_all[attack_agt-1]
    adv_success = bool(attack_coll)

    return adv_success


def check_behind(attacker_fut, tgt_fut, ptr, crash_min_infront):
    '''
    checks if each attacker is behind the target at each time step.

    :param attacker_fut: future for all attackers (NA-B, T, 4)
    :param tgt_fut: future for targets (B, T, 4)
    :param ptr: graph ptr to start of each batch
    :param crash_min_infront: threshold to determine if behind

    :return behind_steps: (NA-B, T) True if attacker currently behind tgt
    '''
    graph_sizes = ptr[1:] - ptr[:-1]
    B = graph_sizes.size(0)
    attacker_pred = attacker_fut
    tgt_expanded = torch.cat([tgt_fut[b:b+1, :, :].expand(attacker_pred.size(0), -1, -1) for b in range(B)], dim=0)

    tgt_h = tgt_expanded[:, :, 2:4]
    tgt_pos = tgt_expanded[:, :, :2]
    atk_pos = attacker_pred[:, :, :2]

    tgt2atk = atk_pos - tgt_pos
    tgt2atk = tgt2atk / torch.norm(tgt2atk, dim=-1, keepdim=True)
    cossim = torch.sum(tgt2atk * tgt_h, dim=-1) # (NA-B, T)

    # determine for each agent if:
    #   - behind and stays behind
    behind_steps = cossim < crash_min_infront
    return behind_steps


def get_att_agt(future_pred, tgt_traj, scene_graph, attack_agt_idx=None, crash_min_t=0, crash_min_infront=None):
        ''' 
        :param future_pred: NA x T x 4 UNNORMALIZED output of the motion model where idx=0 is modeling the planner
        :param tgt_traj: B x T x 4 UNNORMALIZED planner trajectory to attack.
        '''

        if tgt_traj.dim() == 2:
            tgt_traj = tgt_traj.unsqueeze(0)  # [1, T, features]
        else:
            tgt_traj = tgt_traj
                
        NA = future_pred.size(0)
        B = tgt_traj.size(0)
        min_dist = dist_traj = None
        cur_min_agt = cur_min_t = None

        # these are only computed on non-planner vehicles
        ptr = scene_graph.ptr
        graph_sizes = ptr[1:] - ptr[:-1]
        ego_mask = torch.zeros((NA), dtype=torch.bool)
        ego_mask[ptr[:-1]] = True
        # vehicle collision will only be on non-ego, so need to adjust ptr accordingly
        nonego_ptr = ptr - torch.arange(len(ptr)).to(ptr)

        # minimize POSITIONAL distance
        attacker_pred = future_pred[~ego_mask][:, crash_min_t:, :] # (NA-B, T, 4)
        tgt_pred = tgt_traj[:, crash_min_t:, :4]
        tgt_expanded = torch.cat([tgt_pred[b:b+1, :, :].expand(graph_sizes[b]-1, -1, -1) for b in range(B)], dim=0)                
        dist_traj = torch.norm(attacker_pred[:,:,:2] - tgt_expanded[:,:,:2], dim=-1) # (NA-B, T)
        min_dist_in = dist_traj

        if crash_min_infront is not None:
            behind_steps = check_behind(attacker_pred.detach(), tgt_pred.detach(), ptr, crash_min_infront)
            behind_traj = torch.sum(behind_steps, dim=1, keepdim=True) == behind_steps.size(1)
            behind_traj = behind_traj.expand_as(behind_steps)
            if torch.sum(behind_traj.reshape((-1))) == behind_steps.size(0)*behind_traj.size(1):
                # every agent is behind... just optim normally for now
                behind_traj = torch.zeros_like(behind_traj).to(torch.bool)
            min_dist_in = torch.where(behind_traj, float('inf')*torch.ones_like(min_dist_in), min_dist_in) # set to 0 weight for agent behind the target

        if attack_agt_idx is not None:
            attack_agt_mask = torch.zeros(future_pred.size(0), dtype=torch.bool).to(attack_agt_idx.device)
            attack_agt_mask[attack_agt_idx] = True
            attack_agt_mask = attack_agt_mask[~ego_mask].unsqueeze(1).expand_as(min_dist_in)
            min_dist_in = torch.where(~attack_agt_mask, float('inf')*torch.ones_like(min_dist_in), min_dist_in) # set to 0 weight for agent behind the target                

        # soft min over all timesteps and agents
        NT = future_pred.size(1) - crash_min_t

        min_dist = [nn.functional.softmin(min_dist_in[nonego_ptr[b]:nonego_ptr[b+1]].view(-1), dim=0) for b in range(B)]

        # handle case where all frames for all agents are behind and therefore softmin is nan. (set to all 0 prob)
        min_dist = [bdist if torch.isnan(bdist[0]).item() == False else torch.zeros_like(bdist) for bdist in min_dist]

        cur_min_agt = [(torch.max(min_dist_b, dim=0)[1].item() // NT) + 1 for min_dist_b in min_dist]
        cur_min_t = [(torch.max(min_dist_b, dim=0)[1].item() % NT) + crash_min_t for min_dist_b in min_dist]

        return cur_min_agt, cur_min_t