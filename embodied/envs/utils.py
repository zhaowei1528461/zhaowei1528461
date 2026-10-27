import os
import sys
import math
import copy
import numpy as np
import pandas as pd
import math
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from .tool import build_obs_around_ego ,ego_from_self_info
from .predictor_future import predictor_future_only
from scipy.stats import chi2
from copy import deepcopy
from utils.observation import Observation
import xml.etree.ElementTree as ET
# from OnSiteReplay.ReplayParser import ReplayParser
from .guikong import kongzhi, Poly_planner_onsite, Obstacle
from .jiaohe import jiaohe_yundong,jiaohe_dongli_1,jiaohe_dongli_2,jiaohe_dongli_3,jiaohe_dongli_8,jiaohe_dongli_8_new
import embodied
from .risk_cost import TrajectoryRiskEvaluator
# from planner.DreamerV3.dreamer_muti import _shared_scenario, _shared_laneinfo, _shared_obs
from .simple_reward import simple_rewards,check_comfort_number
road_info_dict_pingjie1 = {}  # 模块级变量
ref_pt1 = [] # 模块级变量
ref_pianyi = []
class TrajectoryController:
    """轨迹跟踪控制器"""
    
    def __init__(self):
        # 纵向PID控制器参数
        self.longitudinal_pid = PIDController(
            kp=0.8,    # 位置比例增益
            ki=0.1,    # 积分增益
            kd=0.2     # 微分增益
        )
        
        # Stanley横向控制器参数
        self.stanley_k = 0.5      # Stanley增益
        self.stanley_ksoft = 0.1  # 软化参数
        self.k_ld0 = 5
        self.k_ld1 = 0.5
        # 控制输出限制
        self.max_acceleration = 5.0      # 最大加速度 m/s²
        self.max_deceleration = -5.0     # 最大减速度 m/s²
        self.max_steering_angle = 0.5    # 最大转向角 rad
        
        # 轨迹跟踪状态
        self.last_target_index = 0
        self.lookahead_distance = 10.0   # 前瞻距离 m

class PIDController:
    """PID控制器"""
    
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0
        self.dt = 0.1  # 控制周期
        
    def update(self, error, pid_dt):
        """更新PID控制器"""
        self.dt = pid_dt
        # print('pid_dt', self.dt)
        self.integral += error * self.dt
        derivative = (error - self.last_error) / self.dt
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.last_error = error
        return output
    
    def reset(self):
        """重置PID控制器"""
        self.integral = 0.0
        self.last_error = 0.0

class Sumoenv():
    def __init__(
        self,
        state_dim=56,
        action_dim=4,
        cnn = False,
        safe_rule = False,
        cnn_state = False,
        use_spare_reward = False,
        sparse_episode = 5000,
        nb_ego_states = 11,
        nb_states_per_vehicle = 7,
        sensor_nb_vehicles = 6,
        frenet_state = True,
        # input_queue=None,
    ):
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.safe_rule = safe_rule
        self.cnn_state = cnn_state
        self.cnn = cnn
        self.frenet_state = frenet_state

        self.nb_ego_states = nb_ego_states
        self.nb_states_per_vehicle = nb_states_per_vehicle
        self.sensor_nb_vehicles = sensor_nb_vehicles
        if self.cnn:
            self.state_dim = self.nb_ego_states + self.nb_states_per_vehicle * self.sensor_nb_vehicles
        
        self.use_spare_reward = use_spare_reward
        self.sparse_episode = sparse_episode

        self.sensor_range = 200
        self.x_goal_min = 99999
        self.x_goal_max = 99999
        self.y_goal_min = 99999
        self.y_goal_max = 99999

        self.observation = None
            # 动作范围定义
        self.action_ranges = {
            'planning_horizon': (0.0, 2.0),    # 规划窗口时间 [0s, 2s]
            'lateral_offset': (-2.5, 2.5),     
            'target_velocity': (0.0, 30.0),    
                    }
        # === 纯跟踪控制所需的内部状态 ===
        self._u_prev = np.zeros(2, dtype=np.float32)   # 上一拍 [acc, rot]，用于简单滤波
        self._last_vref = 0.0                          # 记录当拍 v_ref（便于后续奖励对齐）
        self._last_dref = 0.0                          # 记录当拍 d_ref
        self.ego_s_m = 0
        self.v_phys_max = 50.0                         # 物理最大速度（m/s）
        self.d_phys_max_default = 1.75                 # 默认车道半幅（m），若取不到车道宽
        self.s_merge = None
        # 添加轨迹跟踪控制器参数
        self.trajectory_controller = TrajectoryController()
        self.huandao_flag = False
        # 当前规划的轨迹
        self.current_trajectory = None
        self.trajectory_start_time = 0.0
        # 速度环PID（纵向），单位一致：误差=m/s
        self.velocity_pid = PIDController(kp=1, ki=0.05, kd=0.1)
        self.k_pos_assist = 0.05  # 位置辅助项系数（m）
        self.merge_x = None
        self.merge_y = None
        self.risk_evaluator = TrajectoryRiskEvaluator()

        # ji suan acc de chu shi zhi
        self.old_ego_a = 0
        self.old_ego_yaw = 0
    @property
    def action_space(self):
        return {
            'action': embodied.Space(np.float32, (2,), -1.0, 1.0),
            'reset': embodied.Space(bool),
        }

    
    @property
    def observation_space(self):
        return {
            'state': embodied.Space(np.float32, (58,)),
            'reward': embodied.Space(np.float32),
            'cost': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
            'reset': embodied.Space(bool),
        }
    
    def _get_lane_arrays(self, lane: int):
        lane = str(lane)
        ref = self.road_info_dict_pingjie[lane]
        return ref['center_vertices'], ref['phi_road'], ref['curvature'], ref['station']

    def _s_from_idx(self, station: np.ndarray, idx: int) -> float:
        idx = int(np.clip(idx, 0, len(station)-1))
        return float(station[idx])

    def _idx_from_s(self, station: np.ndarray, s_m: float) -> int:
        i = int(np.searchsorted(station, s_m, side='left'))
        return int(np.clip(i, 0, len(station)-2))
    
    def convert_frenet_to_cartesian(self, s_m, d, ref_path, ref_phi, station):
        """
        Frenet(s_m, d) -> (x, y)，s_m 为米制弧长；沿中心线弧长插值，再按左法线偏移 d。
        """
        # 定位 s_m 所在的弧长区间
        idx = self._idx_from_s(station, float(s_m))
        idx_next = min(idx + 1, len(ref_path) - 1)
        s0 = float(station[idx])
        s1 = float(station[idx_next])
        denom = max(s1 - s0, 1e-8)
        alpha = float(np.clip((float(s_m) - s0) / denom, 0.0, 1.0))

        # 空间位置线性插值
        p0 = ref_path[idx]
        p1 = ref_path[idx_next]
        p = (1.0 - alpha) * p0 + alpha * p1

        # 航向用单位向量插值避免角度跳变
        h0 = np.array([np.cos(ref_phi[idx]), np.sin(ref_phi[idx])], dtype=float)
        h1 = np.array([np.cos(ref_phi[idx_next]), np.sin(ref_phi[idx_next])], dtype=float)
        h = (1.0 - alpha) * h0 + alpha * h1
        h /= (np.linalg.norm(h) + 1e-8)

        # 左法线
        n = np.array([-h[1], h[0]], dtype=float)

        # 按 d 偏移
        x_cartesian = float(p[0] + d * n[0])
        y_cartesian = float(p[1] + d * n[1])
        return x_cartesian, y_cartesian
    
    def get_features(self, observation: Observation):
        v_max_temp = 0
        self.huandao_flag = False
        self.observation = observation
        # print(observation)
        ego_x, ego_y, ego_v, ego_a, ego_yaw, ego_width, ego_length = observation.ego_info.x, observation.ego_info.y,  abs(observation.ego_info.v), observation.ego_info.a, observation.ego_info.yaw, observation.ego_info.width, observation.ego_info.length
        if len(self.ego_info) != 0:
            if ego_x == 'nan' or ego_x == 'inf':
                ego_x = self.ego_info.loc['ego','x']
            if ego_y == 'nan' or ego_y == 'inf':
                ego_y = self.ego_info.loc['ego','y']
            if ego_v == 'nan' or ego_v == 'inf':
                ego_v = self.ego_info.loc['ego','v']
            if ego_a == 'nan' or ego_a == 'inf':
                ego_a = self.ego_info.loc['ego','a']
            if ego_yaw == 'nan' or ego_yaw == 'inf':
                ego_yaw = self.ego_info.loc['ego','yaw']
            if ego_width == 'nan' or ego_width == 'inf':
                ego_width = self.ego_info.loc['ego','width']
            if ego_length == 'nan' or ego_length == 'inf':
                ego_length = self.ego_info.loc['ego','length']
        self.ego_rot = observation.ego_info.rot
        ego_station = np.array([ego_x,ego_y])

        frame = pd.DataFrame()
        ego_list = {'ego': {'x':ego_x,'y':ego_y,'v':ego_v,'a':ego_a,'yaw':ego_yaw,'length':ego_length,'width':ego_width}}
        for key,value in ego_list.items():
            sub_frame = pd.DataFrame(value,columns=['x', 'y', 'v', 'a', 'yaw', 'length', 'width', 'lane', 'lane_id', 'rel_pos_ind', 'rel_des', 'ind_node_x', 'ind_node_y', 'dis_ind', 'zuoyou'],index=[key])
            car_station = np.array([sub_frame.loc[key, 'x'],sub_frame.loc[key, 'y']])
            sub_frame['lane'] = int('-1')
            sub_frame['lane_id'] = '-1.-1.-1.-1'
            sub_frame['rel_pos_ind'] = 0
            sub_frame['rel_des'] = abs(np.linalg.norm(ego_station-car_station))
            sub_frame['ind_node_x'] = 0
            sub_frame['ind_node_y'] = 0
            sub_frame['dis_ind'] = 0
            sub_frame['zuoyou'] = 0
            sub_frame['rel_pos_ind_houlun'] = 0
            car_station_houlun = np.array([sub_frame.loc[key, 'x']-sub_frame.loc[key, 'length']/1.7/2*math.cos(sub_frame.loc[key, 'yaw']),sub_frame.loc[key, 'y']-sub_frame.loc[key, 'length']/1.7/2*math.sin(sub_frame.loc[key, 'yaw'])])
            
            m = 999999
            for k1,v1 in self.road_info_dict_pingjie.items():
                refer_tree = KDTree(v1['center_vertices'])
                distance, ind = refer_tree.query(car_station)
                distance_houlun, ind_houlun = refer_tree.query(car_station_houlun)
                if distance <= m:
                    sub_frame.loc[key,'lane'] = int(k1)
                    sub_frame.loc[key,'rel_pos_ind'] = ind
                    sub_frame.loc[key,'rel_pos_ind_houlun'] = ind_houlun
                    sub_frame.loc[key,'ind_node_x'] = v1['center_vertices'][ind,0]
                    sub_frame.loc[key,'ind_node_y'] = v1['center_vertices'][ind,1]
                    m = distance
                    sub_frame.loc[key,'dis_ind'] = m
                    if ind < len(v1['center_vertices']) - 2:
                        xiangliang_car = car_station - v1['center_vertices'][ind,:]
                        xiangliang_line = v1['center_vertices'][ind+1,:] - v1['center_vertices'][ind,:]
                        if xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] > 0:
                            sub_frame.loc[key,'zuoyou'] = 1 #车在道路中心线右侧
                        elif xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] < 0:
                            sub_frame.loc[key,'zuoyou'] = 2 #车在道路中心线左侧
                        else:
                            sub_frame.loc[key,'zuoyou'] = 0 #车在道路中心线
                    else:
                        sub_frame.loc[key,'zuoyou'] = 0 #车在道路外前方
            # ——在 KDTree 得到最近车道后，追加同向性与“冻结”逻辑——
            # if key == 'ego':
            #     # 1) 若已有上一帧的车道，则以“同向”为硬约束，防止切到逆向
            #     if len(self.ego_info) != 0:
            #         prev_lane = int(self.ego_info.loc['ego', 'lane'])
            #         prev_idx  = int(self.ego_info.loc['ego', 'rel_pos_ind'])
            #     else:
            #         prev_lane = int(sub_frame.loc[key, 'lane'])
            #         prev_idx  = int(sub_frame.loc[key, 'rel_pos_ind'])

            #     cand_lane = int(sub_frame.loc[key, 'lane'])
            #     cand_idx  = int(sub_frame.loc[key, 'rel_pos_ind'])
            #     ego_yaw_now = float(sub_frame.loc[key, 'yaw'])

            #     # 若候选最近车道与自车朝向非同向（逆向），则回退到上一帧的本向车道
            #     if not self._same_direction_yaw(cand_lane, cand_idx, ego_yaw_now, cos_thresh=0.5):
            #         sub_frame.loc[key, 'lane']        = prev_lane
            #         sub_frame.loc[key, 'rel_pos_ind'] = prev_idx

            #     # 2) 可选：进入逆向时“冻结”车道，直到回正再释放
            #     if not hasattr(self, "_frozen_lane"):
            #         self._frozen_lane = None

            #     # 若检测到最近车道逆向，则冻结为上一帧本向车道
            #     if not self._same_direction_yaw(cand_lane, cand_idx, ego_yaw_now, cos_thresh=0.5):
            #         self._frozen_lane = prev_lane

            #     # 若已冻结，则强制使用冻结车道；当重新和该车道同向且横向回到半幅内再解冻
            #     if self._frozen_lane is not None:
            #         sub_frame.loc[key, 'lane'] = int(self._frozen_lane)
            #         # 用当前车道的半幅判断是否“回到本向车道内”
            #         lane_widths = self.road_info_dict_pingjie[str(int(self._frozen_lane))]["width"]
            #         idx_safe = int(np.clip(int(sub_frame.loc[key, 'rel_pos_ind']), 0, len(lane_widths)-1))
            #         half_w   = float(lane_widths[idx_safe]) / 2.0 if len(lane_widths) else 1.75
            #         # 计算相对于冻结车道的带符号横距
            #         d_prev, _ = self._signed_d_to_lane(int(self._frozen_lane), np.array([sub_frame.loc[key,'x'], sub_frame.loc[key,'y']], dtype=float))
            #         # 回正判据：同向 + |d| 回到半幅内（阈值可调 0.45~0.65）
            #         if self._same_direction_yaw(int(self._frozen_lane), idx_safe, ego_yaw_now, cos_thresh=0.5) and abs(d_prev) < 0.5 * half_w:
            #             self._frozen_lane = None

            if key == 'ego':
                self.ego_info = sub_frame
            frame = pd.concat([frame, sub_frame])

        ego_lane = self.ego_info.loc['ego','lane']
        ego_lane_width = self.road_info_dict_pingjie[str(ego_lane)]['width'][self.ego_info.loc['ego','rel_pos_ind']]
        refer_tree = KDTree(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])
        for obj_type in observation.object_info:
            for key, value in observation.object_info[obj_type].items():
                sub_frame = pd.DataFrame(vars(value),columns=['x', 'y', 'v', 'a', 'yaw', 'length', 'width', 'lane', 'lane_id', 'rel_pos_ind', 'rel_des', 'ind_node_x', 'ind_node_y', 'dis_ind', 'zuoyou', 'rel_pos_ind_houlun'],index=[key])
                car_station = np.array([sub_frame.loc[key, 'x'],sub_frame.loc[key, 'y']])
                sub_frame['lane'] = int('-1')
                sub_frame['lane_id'] = '-1.-1.-1.-1'
                sub_frame['rel_pos_ind'] = 0
                sub_frame['rel_des'] = abs(np.linalg.norm(ego_station-car_station))
                sub_frame['ind_node_x'] = 0
                sub_frame['ind_node_y'] = 0
                sub_frame['dis_ind'] = 0
                sub_frame['zuoyou'] = 0
                sub_frame['rel_pos_ind_houlun'] = 0

                if sub_frame.loc[key,'v'] > v_max_temp:
                    v_max_temp = sub_frame.loc[key,'v']

                distance, ind = refer_tree.query(car_station)
                if distance <= self.road_info_dict_pingjie[str(ego_lane)]['width'][ind]/2 + 1e-4:
                    sub_frame.loc[key,'lane'] = ego_lane
                sub_frame.loc[key,'rel_pos_ind'] = ind
                sub_frame.loc[key,'ind_node_x'] = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,0]
                sub_frame.loc[key,'ind_node_y'] = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,1]
                sub_frame.loc[key,'dis_ind'] = distance
                if ind < len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']) - 2:
                    xiangliang_car = car_station - self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,:]
                    xiangliang_line = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind+1,:] - self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,:]
                    if xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] > 0:
                        sub_frame.loc[key,'zuoyou'] = 1 #车在道路中心线右侧
                    elif xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] < 0:
                        sub_frame.loc[key,'zuoyou'] = 2 #车在道路中心线左侧
                    else:
                        sub_frame.loc[key,'zuoyou'] = 0 #车在道路中心线
                else:
                    sub_frame.loc[key,'zuoyou'] = 0 #车在道路外前方

                if key == 'ego':
                    self.ego_info = sub_frame
                frame = pd.concat([frame, sub_frame])
        
        #速度限制，跟随道路曲率
        v_max_des = 0
        ind = self.ego_info.loc['ego','rel_pos_ind'] + 2
        # ind = self.ego_info.loc['ego','rel_pos_ind'] + 10
        total = 0
        for i in range(5):
            ind += i
            ind = self.limit(ind,0,len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])-2)
            total += abs(self.road_info_dict_pingjie[str(ego_lane)]['curvature'][ind])
        k_lane = total / 5
        
        if k_lane <= 0.001:
            v_max_des = 50
        elif k_lane > 0.03:
            v_max_des = 10
        else:
            # v_max_des = -1600 * k_lane + 50
            v_max_des = -40 / ((math.pow(2.7,0.03) - math.pow(2.7,0.001)) + 1e-6) * k_lane + 50
        
        self.v_max = self.limit(v_max_des,6,50)
        # print(self.v_max)
        
        # print('k_lane: ', k_lane, 'self.v_max: ', self.v_max)

        frame = frame.drop(['ego'])
        same_temp = frame.loc[frame['lane'] == int(self.ego_info.loc['ego','lane'])]
        front_same_temp = same_temp.loc[same_temp['rel_pos_ind'] > self.ego_info.loc['ego','rel_pos_ind']]
        #速度限制，跟随周围车辆
        ind_lim = self.ego_info.loc['ego','rel_pos_ind'] + 15
        total_lim = 0
        for i in range(5):
            ind_lim += i
            ind_lim = self.limit(ind_lim,0,len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])-2)
            total_lim += abs(self.road_info_dict_pingjie[str(ego_lane)]['curvature'][ind_lim])
        k_lane_lim = total_lim / 5
        # if v_max_temp > 1 and not front_same_temp.empty:
        if v_max_temp > 1 and (not front_same_temp.empty or k_lane_lim > 0.02):
            self.v_max = self.limit(self.v_max,0,v_max_temp)


        same = frame.loc[frame['lane'] == int(self.ego_info.loc['ego','lane'])]
        self.front = same.loc[same['rel_pos_ind'] > self.ego_info.loc['ego','rel_pos_ind']]
        self.behind = same.loc[same['rel_pos_ind'] <= self.ego_info.loc['ego','rel_pos_ind']]
        if not self.frenet_state:
            x1, x2, x3, x4, x5, x6 = (-1 for i in range(6))
            y1, y2, y3, y4, y5, y6 = (-1 for i in range(6))
            a1, a2, a3, a4, a5, a6 = (-1 for i in range(6))
            yaw1, yaw2, yaw3, yaw4, yaw5, yaw6 = (-1 for i in range(6))
            v1, v2, v3, v4, v5, v6 = (-1 for i in range(6))
            length1, length2, length3, length4, length5, length6 = (-1 for i in range(6))
            width1, width2, width3, width4, width5, width6 = (-1 for i in range(6))

            same = frame.loc[frame['lane'] == int(self.ego_info.loc['ego','lane'])]
            other = frame.loc[frame['lane'] != int(self.ego_info.loc['ego','lane'])]
            right = other.loc[other['zuoyou'] == 1]
            left = other.loc[other['zuoyou'] ==2]

            front_same = same.loc[same['rel_pos_ind'] > self.ego_info.loc['ego','rel_pos_ind']]
            behind_same = same.loc[same['rel_pos_ind'] <= self.ego_info.loc['ego','rel_pos_ind']]
            front_left = left.loc[left['rel_pos_ind'] > self.ego_info.loc['ego','rel_pos_ind']]
            behind_left = left.loc[left['rel_pos_ind'] <= self.ego_info.loc['ego','rel_pos_ind']]
            front_right = right.loc[right['rel_pos_ind'] > self.ego_info.loc['ego','rel_pos_ind']]
            behind_right = right.loc[right['rel_pos_ind'] <= self.ego_info.loc['ego','rel_pos_ind']]

            if not front_same.empty:
                index2 = front_same['rel_pos_ind'].idxmin()
                x2 = abs(front_same['x'][index2] - ego_x)
                y2 = abs(front_same['y'][index2] - ego_y)
                a2 = abs(front_same['a'][index2])
                yaw2 = front_same['yaw'][index2]
                v2 = front_same['v'][index2]
                length1 = front_same['length'][index2]
                width1 = front_same['width'][index2]

            if not behind_same.empty:
                index5 = behind_same['rel_pos_ind'].idxmax()
                x5 = abs(behind_same['x'][index5] - ego_x) 
                y5 = abs(behind_same['y'][index5] - ego_y)
                a5 = abs(behind_same['a'][index5])
                yaw5 = behind_same['yaw'][index5]
                v5 = behind_same['v'][index5]
                length5 = behind_same['length'][index5]
                width5 = behind_same['width'][index5]

            if not front_left.empty:
                index1 = front_left['rel_pos_ind'].idxmin()
                x1 = abs(front_left['x'][index1] - ego_x)
                y1 = abs(front_left['y'][index1] - ego_y)
                a1 = abs(front_left['a'][index1])
                yaw1 = front_left['yaw'][index1]
                v1 = front_left['v'][index1]
                length1 = front_left['length'][index1]
                width1 = front_left['width'][index1]

            if not behind_left.empty:
                index4 = behind_left['rel_pos_ind'].idxmax()
                x4 = abs(behind_left['x'][index4] - ego_x)
                y4 = abs(behind_left['y'][index4] - ego_y)
                a4 = abs(behind_left['a'][index4])
                yaw4 = behind_left['yaw'][index4]
                v4 = behind_left['v'][index4]
                length4 = behind_left['length'][index4]
                width4 = behind_left['width'][index4]

            if not front_right.empty:
                index3 = front_right['rel_pos_ind'].idxmin()
                x3 = abs(front_right['x'][index3] - ego_x)
                y3 = abs(front_right['y'][index3] - ego_y)
                a3 = abs(front_right['a'][index3])
                yaw3 = front_right['yaw'][index3]
                v3 = front_right['v'][index3]
                length3 = front_right['length'][index3]
                width3 = front_right['width'][index3]

            if not behind_right.empty:
                index6 = behind_right['rel_pos_ind'].idxmax()
                x6 = abs(behind_right['x'][index6] - ego_x)
                y6 = abs(behind_right['y'][index6] - ego_y)
                a6 = abs(behind_right['a'][index6])
                yaw6 = behind_right['yaw'][index6]
                v6 = behind_right['v'][index6]
                length6 = behind_right['length'][index6]
                width6 = behind_right['width'][index6]

            
            state = [ego_x/1000,ego_y/1000,ego_v/50,ego_a/15,ego_yaw/(math.pi*2),ego_length/10,ego_width/5,\
                    x1/self.sensor_range,y1/self.sensor_range,v1/50,a1/15,yaw1/(math.pi*2),length1/10,width1/5,\
                    x2/self.sensor_range,y2/self.sensor_range,v2/50,a2/15,yaw2/(math.pi*2),length2/10,width2/5,\
                    x3/self.sensor_range,y3/self.sensor_range,v3/50,a3/15,yaw3/(math.pi*2),length3/10,width3/5,\
                    x4/self.sensor_range,y4/self.sensor_range,v4/50,a4/15,yaw4/(math.pi*2),length4/10,width4/5,\
                    x5/self.sensor_range,y5/self.sensor_range,v5/50,a5/15,yaw5/(math.pi*2),length5/10,width5/5,\
                    x6/self.sensor_range,y6/self.sensor_range,v6/50,a6/15,yaw6/(math.pi*2),length6/10,width6/5,\
                    self.x_goal_min/1000,self.x_goal_max/1000,self.y_goal_min/1000,self.y_goal_max/1000]
            state = np.array(state)
            # for v in state:
            #     if v > 1:
            #         print(state)
            #     if v < 0.01:
            #         print(state)
            ego_x = self.ego_info.loc['ego', 'x']
            ego_y = self.ego_info.loc['ego', 'y']
            ego_v = self.ego_info.loc['ego', 'v']
            ego_phi = self.ego_info.loc['ego', 'yaw'] * 180 / math.pi  # 转为角度
            ego_lane = self.ego_info.loc['ego', 'lane']
            rel_pos_ind = self.ego_info.loc['ego', 'rel_pos_ind']
            ego_d = self.ego_info.loc['ego', 'dis_ind'] 
            # print('334rel_pos_ind', rel_pos_ind)
            # print('334ego_d', ego_d)
            ref_path = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']
            # print('ref_path', ref_path)
            ref_phi = self.road_info_dict_pingjie[str(ego_lane)]['phi_road']
            # print('ref_phi', ref_phi)
            ref_v = self.v_max  # 目标速度，已经动态调整

            # 当前目标点
            ref_pt = ref_path[rel_pos_ind]
            global ref_pt1
            ref_pt1 = ref_pt
            # print('ref_pt1', ref_pt1)
            ref_angle = ref_phi[rel_pos_ind] * 180 / math.pi  # 转为角度

            # delta_y: 横向距离
            delta_y = math.sqrt((ego_x - ref_pt[0])**2 + (ego_y - ref_pt[1])**2)
            # print(delta_y)
            # delta_phi: 朝向角误差，注意 wrap 到 [-180, 180]
            delta_phi = ego_phi - ref_angle
            if delta_phi > 180:
                delta_phi -= 360
            elif delta_phi < -180:
                delta_phi += 360

            # delta_v: 速度偏差
            delta_v = ego_v - ref_v

            # 归一化并拼接
            tracking_error = np.array([delta_y / self.sensor_range,
                                    delta_phi / 180,
                                    delta_v / 50])
            # print('state tracking_error', tracking_error)
            # print('tracking_error', tracking_error.shape)
            # print('state', state.shape)
            # state = np.concatenate([state, tracking_error])
            # print('state1', state.shape)
            # 判断数组中是否存在 NaN 或 Inf
            has_nan_or_inf = np.isnan(state) | np.isinf(state)
            # 将 NaN 和 Inf 替换成 -1
            state[has_nan_or_inf] = -1
        else:  
            # 1. 自车Frenet信息（统一米制）
            # ego_x = self.ego_info.loc['ego', 'x'] / 1000             # 自车位置x
            # ego_y = self.ego_info.loc['ego', 'y'] / 1000          # 自车位置y
            goal_x = self.goal_center_x / 1000
            goal_y = self.goal_center_y / 1000
            ego_v = self.ego_info.loc['ego', 'v']            # 自车速度
            ego_a = self.ego_info.loc['ego', 'a']            # 自车加速度
            ego_pos = np.array([ego_x, ego_y])
            ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)

            # 找到最近的参考点
            distances = np.linalg.norm(ref_center - ego_pos, axis=1)
            nearest_idx = np.argmin(distances)

            # 纵向弧长（m）
            ego_s_m = self._s_from_idx(station, int(nearest_idx))
            # print('ego_s_m_before', ego_s_m)
            # 切线与法线
            ref_tangent = np.array([np.cos(ref_phi[nearest_idx]), np.sin(ref_phi[nearest_idx])])
            ref_normal = np.array([-np.sin(ref_phi[nearest_idx]), np.cos(ref_phi[nearest_idx])])

            # 带符号横向偏移（m）
            to_ego = ego_pos - ref_center[nearest_idx]
            ego_d_frenet = float(np.dot(to_ego, ref_normal))
            # print('ego_d_frenet', ego_d_frenet)
            # 速度在 Frenet 坐标系的分解
            ego_vs = float(ego_v * np.dot([np.cos(ego_yaw), np.sin(ego_yaw)], ref_tangent))
            ego_vd = float(ego_v * np.dot([np.cos(ego_yaw), np.sin(ego_yaw)], ref_normal))
            ego_s_m_new, ego_d_frenet_new, ego_seg_idx, ego_alpha, ego_h_unit = self._project_point_to_centerline(
                ego_pos, ref_center, ref_phi, station, nearest_idx
            )
            self.ego_s_m = ego_s_m_new
            # 速度在 Frenet 中的分解（基于插值切向/法向）
            ego_n_unit = np.array([-ego_h_unit[1], ego_h_unit[0]], dtype=float)
            ego_vs_new = float(ego_v * np.dot([np.cos(ego_yaw), np.sin(ego_yaw)], ego_h_unit))
            ego_vd_new = float(ego_v * np.dot([np.cos(ego_yaw), np.sin(ego_yaw)], ego_n_unit))
            # print('ego_s_m_after', ego_s_m_new)
            # 航向角误差
            ego_heading_error = ego_yaw - ref_phi[nearest_idx]
            ego_heading_error = np.arctan2(np.sin(ego_heading_error), np.cos(ego_heading_error))

            # 车道边界距离（m）
            lane_width = self.road_info_dict_pingjie[str(ego_lane)]['width'][nearest_idx]
            ego_left_boundary_dist = lane_width/2 + ego_d_frenet
            ego_right_boundary_dist = lane_width/2 - ego_d_frenet

            # 2. 旁车 Frenet 信息
            other_vehicles_frenet = []

            # 使用 frame 中的所有车辆（包含相邻车道），注意：frame['rel_pos_ind'] 已是对“自车当前车道中心线”的最近点索引
            all_cars = frame.copy()
            # print('all_cars', all_cars)
            def _idx_to_s_m(idx):
                return self._s_from_idx(station, int(idx))
            cand = []
            k = 6
            d_max = []
            for name, other_car in all_cars.iterrows():
                # 跳过与自车重合的记录_ind
                if (abs(other_car['x'] - ego_x) < 0.1 and abs(other_car['y'] - ego_y) < 0.1):
                    continue
                # print('other_car', other_car)
                other_idx = int(other_car['rel_pos_ind'])  # 这是在“自车车道中心线”上的索引
                other_pos = np.array([other_car['x'], other_car['y']], dtype=float)
                other_s_m = self._s_from_idx(station, other_idx)

                # 参考切向/法向（仍以“自车车道中心线”为准）
                t_hat = np.array([np.cos(ref_phi[other_idx]), np.sin(ref_phi[other_idx])], dtype=float)
                n_hat = np.array([-np.sin(ref_phi[other_idx]), np.cos(ref_phi[other_idx])], dtype=float)

                # 该车的带符号横向偏移 d（对自车车道中心线）
                to_other = other_pos - ref_center[other_idx]
                other_d = float(np.dot(to_other, n_hat))
                # 该车在 Frenet 下的速度分解
                other_v = float(other_car['v'])
                other_yaw = float(other_car['yaw'])
                other_vs = float(other_v * math.cos(other_yaw - ref_phi[other_idx]))
                other_vd = float(other_v * math.sin(other_yaw - ref_phi[other_idx]))
                other_s_m_new, other_d_new, other_seg_idx, other_alpha, other_h_unit = self._project_point_to_centerline(
                    other_pos, ref_center, ref_phi, station, other_idx
                )
                # 相对 Frenet 距离
                s_diff = float(other_s_m - ego_s_m)
                # print('s_diff', s_diff)
                s_diff_new = float(other_s_m_new - ego_s_m_new)
                # print('s_diff_new', s_diff_new)
                d_diff = float(other_d - ego_d_frenet)
                d_diff_new = float(other_d_new - ego_d_frenet_new)
                # 该车在 Frenet 下的速度分解（基于插值切向/法向）
                other_vs_new = float(other_v * (np.cos(other_yaw) * other_h_unit[0] + np.sin(other_yaw) * other_h_unit[1]))
                other_n_unit = np.array([-other_h_unit[1], other_h_unit[0]], dtype=float)
                other_vd_new = float(other_v * (np.cos(other_yaw) * other_n_unit[0] + np.sin(other_yaw) * other_n_unit[1]))
                dtheta = self._wrap_to_pi(other_yaw - ego_yaw)
                # print('dtheta', abs(dtheta))
                if 2.79 <= abs(dtheta) <= 3.49:
                    continue
                if abs(s_diff_new) <= 30 and abs(d_diff_new) <= 10.0:
                    d_max.append(abs(d_diff_new))
                    # 二级排序键：先 |s_diff|，再 |d_diff|（可选，保证并列时更近车道者优先）
                    # cand.append((abs(s_diff), abs(d_diff), s_diff, d_diff, (other_vs - ego_vs), (other_vd - ego_vd)))
                    cand.append((abs(s_diff_new), abs(d_diff_new), s_diff_new, d_diff_new, (other_vs_new - ego_vs_new), (other_vd_new - ego_vd_new)))
            cand.sort(key=lambda x: (x[0], x[1]))
            out = []
            mask = []
            for _, _, s_diff, d_diff, dvs, dvd in cand[:k]:
                out.extend([s_diff/200.0, d_diff/5.0, dvs/30.0, dvd/30.0])
                mask.append(1.0)
            # 不足 k 辆补零
            while len(out) < 4 * k:
                out.extend([0.0, 0.0, 0.0, 0.0])
                mask.append(0.0)
            try:
                other_vehicles_frenet = out
                other_vehicles_mask = mask
            except UnboundLocalError:
                other_vehicles_frenet = [0.0] * (4 * k) 
                other_vehicles_mask = [0.0] * k
            # print('other_vehicles_frenet', other_vehicles_frenet)
            # print('other_vehicles_mask', other_vehicles_mask)
            # 3. 未来路点（按米推进）
            future_waypoints = []
            for i in range(5):
                future_s_m = float(ego_s_m + (i+1) * 3.0)  # +10m, +20m
                future_idx = self._idx_from_s(station, future_s_m)
                future_idx = min(future_idx, len(ref_center) - 1)

                s_to_future = float(future_s_m - ego_s_m)
                d_to_future = float(0.0 - ego_d_frenet)  # 中心线 d=0

                future_curvature = self.road_info_dict_pingjie[str(ego_lane)]['curvature'][future_idx]
                future_heading = ref_phi[future_idx]

                future_waypoints.extend([
                    s_to_future/200,
                    d_to_future/5,
                    future_curvature/0.02,
                    future_heading/(2*math.pi)
                ])
            
            # 读 reset 预计算的左右最大偏移；若缺省则退化为本车道半宽
            def _pick(arr, i):
                return float(arr[min(max(i,0), len(arr)-1)])
            ego_idx  = int(self.ego_info.loc['ego','rel_pos_ind'])
            if hasattr(self, "lateral_limits") and ego_lane in self.lateral_limits:
                left_max  = _pick(self.lateral_limits[ego_lane]["left_max"],  ego_idx)
                right_max = _pick(self.lateral_limits[ego_lane]["right_max"], ego_idx)
            else:
                w = self.road_info_dict_pingjie[str(ego_lane)]["width"]
                half = (_pick(w, ego_idx)/2.0) if len(w) else 1.75
                left_max = right_max = half

            # 二次保护（避免过大/过小；按你原参数）
            left_max  = max(0.8, min(left_max,  4.0))
            right_max = max(0.8, min(right_max, 4.0))

            d_min, ttc_min = self._nearest_dist_and_ttc_xy(ego_x, ego_y, ego_v, ego_yaw)
            # print('d_min, ttc_min', d_min, ttc_min)
            goal_xy = np.array([self.goal_center_x, self.goal_center_y], dtype=float)
            # 最近中心线索引
            goal_idx = int(np.argmin(np.linalg.norm(ref_center - goal_xy, axis=1)))
            s_goal = self._s_from_idx(station, goal_idx)
            # 在你的 get_features 方法中找到这部分代码：
            goal_xy = np.array([self.goal_center_x, self.goal_center_y], dtype=float)
            # 最近中心线索引
            goal_idx = int(np.argmin(np.linalg.norm(ref_center - goal_xy, axis=1)))
            s_goal = self._s_from_idx(station, goal_idx)

            # ============ 新增：计算 d_goal ============ 
            # 1. 找到目标点所在的车道中心线最近点
            goal_on_center = ref_center[goal_idx]
            # 2. 计算该点处的法向量
            phi_goal = ref_phi[goal_idx]
            n_hat_goal = np.array([-np.sin(phi_goal), np.cos(phi_goal)])
            # 3. 计算目标点相对于中心线的横向偏移
            vector_to_goal = goal_xy - goal_on_center
            d_goal_absolute = np.dot(vector_to_goal, n_hat_goal)

            # 4. 计算相对于自车的横向偏移（关键！）
            d_goal_relative = d_goal_absolute - ego_d_frenet

            # 5. 归一化（可选，但推荐）
            d_goal_normalized = d_goal_relative / 5.0  # 假设横向最大偏移为5米
            # print('s_goal', s_goal)
            phi_merge = None
            # if self.s_merge is None:
            #     # self.s_merge = self.compute_merge_s(
            #     #                                     ego_lane=ego_lane,
            #     #                                     ego_idx=ego_idx,
            #     #                                     kappa_thr=0.02,
            #     #                                     consec=6,
            #     #                                     back_offset_m=3,
            #     #                                     search_ahead_pts=200
            #     #                                 )
            #     self.s_merge = self.compute_merge_s_precise(
            #                                         ego_lane=ego_lane,
            #                                         ego_idx=ego_idx,
            #                                         kappa_thr=0.02,
            #                                         consec=8,
            #                                         min_back_m = 2.0,
            #                                         max_back_m = 10.0,
            #                                         t_buffer = 0.6,        # 速度相关回退（秒）
            #                                         search_ahead_pts = 300,
            #                                         smooth_win = 9, 
            #                                     )
            # # print(702, max(d_max))
            # # print(703, ego_s_m)
            # # print(716, self.s_merge)
            # # print('s_merge', self.s_merge)
            #     merge_x, merge_y = self.convert_frenet_to_cartesian(
            #                                                         self.s_merge,
            #                                                         0.0,
            #                                                         ref_center,
            #                                                         ref_phi,
            #                                                         station
            #                                                     )
            #     self.merge_x, self.merge_y = merge_x, merge_y
            # print('merge_x, merge_y', self.merge_x, self.merge_y)
            # 2) 计算自车到达时间（笛卡尔）
            # t_ego_to_merge = self._time_to_point_cartesian(
            #     x=ego_x, y=ego_y, v=ego_v, yaw=ego_yaw, x_tar=self.merge_x, y_tar=self.merge_y
            # )
            # self.t_to_merge_ego = float(t_ego_to_merge)
            # print(805, self.t_to_merge_ego)
            # print(806, t_o_to_merge)
            # 组装状态
            # print(ego_d_frenet)
            frenet_state = [
                ego_s_m/200,                    # 纵向位置（m）
                ego_d_frenet/5,                 # 横向偏移（m）
                ego_vs/30,                      # 纵向速度
                ego_a/5,
                ego_heading_error/math.pi,      # 航向角误差
                *other_vehicles_frenet,         # 旁车
                *other_vehicles_mask,           # 旁车掩码
                *future_waypoints,
                left_max/5,                     # 最大左偏移
                right_max/5,                     # 最大右偏移   
                # min(d_min / 10.0, 1.0),          # 归一化最小距离（d_safe=5.0）
                # ttc_min,        # 归一化最小TTC（t_safe=2.5）        
                s_goal/200,                       # 目标点纵向位置（m） 
                # d_goal_normalized,               # 目标点横向位置（m
                # ego_x / 1000,
                # ego_y / 1000,
                # goal_x,
                # goal_y,
            ]
            # print('future_waypoints',future_waypoints)
            frenet_state = np.array(frenet_state, dtype=np.float32)
            has_nan_or_inf = np.isnan(frenet_state) | np.isinf(frenet_state)
            frenet_state[has_nan_or_inf] = -1
            state = frenet_state
            # print('front', self.front)
            # print('ego_info', self.ego_info)
            # print('frenet_state', frenet_state[-4:])
            # print('frenet_bucket_names', self.frenet_bucket_names)
            # import time
            # time.sleep(0.5)
        # self._update_ego_lane_with_hysteresis()
        return state
    
    def _get_features_end(self, observation):
        self.observation = observation
        ego_x, ego_y, ego_v, ego_a, ego_yaw, ego_width, ego_length = [float(observation.ego_info[key]) for key in ['x', 'y', 'v', 'a', 'yaw', 'width', 'length']]
        ego_station = np.array([ego_x,ego_y])
        frame = pd.DataFrame()
        ego_list = {'ego': {'x':ego_x,'y':ego_y,'v':ego_v,'a':ego_a,'yaw':ego_yaw,'length':ego_length,'width':ego_width}}
        for key,value in ego_list.items():
            sub_frame = pd.DataFrame(value ,columns=['x', 'y', 'v', 'a', 'yaw', 'length', 'width', 'lane', 'lane_id', 'rel_pos_ind', 'rel_des', 'ind_node_x', 'ind_node_y', 'dis_ind', 'zuoyou', 'rel_pos_ind_houlun'],index=[key])
            car_station = np.array([sub_frame.loc[key, 'x'],sub_frame.loc[key, 'y']])
            sub_frame['lane'] = int('-1')
            sub_frame['lane_id'] = '-1.-1.-1.-1'
            sub_frame['rel_pos_ind'] = 0
            sub_frame['rel_des'] = abs(np.linalg.norm(ego_station-car_station))
            sub_frame['ind_node_x'] = 0
            sub_frame['ind_node_y'] = 0
            sub_frame['dis_ind'] = 0
            sub_frame['zuoyou'] = 0
            sub_frame['rel_pos_ind_houlun'] = 0
            car_station_houlun = np.array([sub_frame.loc[key, 'x']-sub_frame.loc[key, 'length']/1.7/2*math.cos(sub_frame.loc[key, 'yaw']),sub_frame.loc[key, 'y']-sub_frame.loc[key, 'length']/1.7/2*math.sin(sub_frame.loc[key, 'yaw'])])
            
            m = 999999
 

            for k1,v1 in self.road_info_dict_pingjie.items():
                refer_tree = KDTree(v1['center_vertices'])
                distance, ind = refer_tree.query(car_station)
                distance_houlun, ind_houlun = refer_tree.query(car_station_houlun)
                if distance <= m:
                    sub_frame.loc[key,'lane'] = int(k1)
                    sub_frame.loc[key,'rel_pos_ind'] = ind
                    sub_frame.loc[key,'rel_pos_ind_houlun'] = ind_houlun
                    sub_frame.loc[key,'ind_node_x'] = v1['center_vertices'][ind,0]
                    sub_frame.loc[key,'ind_node_y'] = v1['center_vertices'][ind,1]
                    m = distance
                    sub_frame.loc[key,'dis_ind'] = m
                    if ind < len(v1['center_vertices']) - 2:
                        xiangliang_car = car_station - v1['center_vertices'][ind,:]
                        xiangliang_line = v1['center_vertices'][ind+1,:] - v1['center_vertices'][ind,:]
                        if xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] > 0:
                            sub_frame.loc[key,'zuoyou'] = 1 #车在道路中心线右侧
                        elif xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] < 0:
                            sub_frame.loc[key,'zuoyou'] = 2 #车在道路中心线左侧
                        else:
                            sub_frame.loc[key,'zuoyou'] = 0 #车在道路中心线
                    else:
                        sub_frame.loc[key,'zuoyou'] = 0 #车在道路外前方
            if key == 'ego':
                self.ego_info = sub_frame
            frame = pd.concat([frame, sub_frame])

        ego_lane = self.ego_info.loc['ego','lane']
        ego_lane_width = self.road_info_dict_pingjie[str(ego_lane)]['width'][self.ego_info.loc['ego','rel_pos_ind']]
        refer_tree = KDTree(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'])
        for key, value in observation.object_info.items():
            sub_frame = pd.DataFrame(value,columns=['x', 'y', 'v', 'a', 'yaw', 'length', 'width', 'lane', 'lane_id', 'rel_pos_ind', 'rel_des', 'ind_node_x', 'ind_node_y', 'dis_ind', 'zuoyou', 'rel_pos_ind_houlun'],index=[key])
            car_station = np.array([sub_frame.loc[key, 'x'],sub_frame.loc[key, 'y']])
            sub_frame['lane'] = int('-1')
            sub_frame['lane_id'] = '-1.-1.-1.-1'
            sub_frame['rel_pos_ind'] = 0
            sub_frame['rel_des'] = abs(np.linalg.norm(ego_station-car_station))
            sub_frame['ind_node_x'] = 0
            sub_frame['ind_node_y'] = 0
            sub_frame['dis_ind'] = 0
            sub_frame['zuoyou'] = 0
            sub_frame['rel_pos_ind_houlun'] = 0

            distance, ind = refer_tree.query(car_station)
            if distance <= self.road_info_dict_pingjie[str(ego_lane)]['width'][ind]/2 + 1e-4:
                sub_frame.loc[key,'lane'] = ego_lane
            sub_frame.loc[key,'rel_pos_ind'] = ind
            sub_frame.loc[key,'ind_node_x'] = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,0]
            sub_frame.loc[key,'ind_node_y'] = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,0]
            sub_frame.loc[key,'dis_ind'] = distance
            if ind < len(self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']) - 2:
                xiangliang_car = car_station - self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,:]
                xiangliang_line = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind+1,:] - self.road_info_dict_pingjie[str(ego_lane)]['center_vertices'][ind,:]
                if xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] > 0:
                    sub_frame.loc[key,'zuoyou'] = 1 #车在道路中心线右侧
                elif xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] < 0:
                    sub_frame.loc[key,'zuoyou'] = 2 #车在道路中心线左侧
                else:
                    sub_frame.loc[key,'zuoyou'] = 0 #车在道路中心线
            else:
                sub_frame.loc[key,'zuoyou'] = 0 #车在道路外前方


            if key == 'ego':
                self.ego_info = sub_frame
            frame = pd.concat([frame, sub_frame])
        
        frame = frame.drop(['ego'])
        
        same = frame.loc[frame['lane'] == int(self.ego_info.loc['ego','lane'])]
        self.front = same.loc[same['rel_pos_ind'] > self.ego_info.loc['ego','rel_pos_ind']]
        self.behind = same.loc[same['rel_pos_ind'] <= self.ego_info.loc['ego','rel_pos_ind']]
        if not self.frenet_state:
            x1, x2, x3, x4, x5, x6 = (-1 for i in range(6))
            y1, y2, y3, y4, y5, y6 = (-1 for i in range(6))
            a1, a2, a3, a4, a5, a6 = (-1 for i in range(6))
            yaw1, yaw2, yaw3, yaw4, yaw5, yaw6 = (-1 for i in range(6))
            v1, v2, v3, v4, v5, v6 = (-1 for i in range(6))
            length1, length2, length3, length4, length5, length6 = (-1 for i in range(6))
            width1, width2, width3, width4, width5, width6 = (-1 for i in range(6))

            same = frame.loc[frame['lane'] == int(self.ego_info.loc['ego','lane'])]
            other = frame.loc[frame['lane'] != int(self.ego_info.loc['ego','lane'])]
            right = other.loc[other['zuoyou'] == 1]
            left = other.loc[other['zuoyou'] == 2]

            front_same = same.loc[same['rel_pos_ind'] > self.ego_info.loc['ego','rel_pos_ind']]
            behind_same = same.loc[same['rel_pos_ind'] <= self.ego_info.loc['ego','rel_pos_ind']]
            front_left = left.loc[left['rel_pos_ind'] > self.ego_info.loc['ego','rel_pos_ind']]
            behind_left = left.loc[left['rel_pos_ind'] <= self.ego_info.loc['ego','rel_pos_ind']]
            front_right = right.loc[right['rel_pos_ind'] > self.ego_info.loc['ego','rel_pos_ind']]
            behind_right = right.loc[right['rel_pos_ind'] <= self.ego_info.loc['ego','rel_pos_ind']]

            if not front_same.empty:
                index2 = front_same['rel_pos_ind'].idxmin()
                x2 = abs(front_same['x'][index2] - ego_x)
                y2 = abs(front_same['y'][index2] - ego_y)
                a2 = abs(front_same['a'][index2])
                yaw2 = front_same['yaw'][index2]
                v2 = front_same['v'][index2]
                length1 = front_same['length'][index2]
                width1 = front_same['width'][index2]
            if not behind_same.empty:
                index5 = behind_same['rel_pos_ind'].idxmax()
                x5 = abs(behind_same['x'][index5] - ego_x) 
                y5 = abs(behind_same['y'][index5] - ego_y)
                a5 = abs(behind_same['a'][index5])
                yaw5 = behind_same['yaw'][index5]
                v5 = behind_same['v'][index5]
                length5 = behind_same['length'][index5]
                width5 = behind_same['width'][index5]
            if not front_left.empty:
                index1 = front_left['rel_pos_ind'].idxmin()
                x1 = abs(front_left['x'][index1] - ego_x)
                y1 = abs(front_left['y'][index1] - ego_y)
                a1 = abs(front_left['a'][index1])
                yaw1 = front_left['yaw'][index1]
                v1 = front_left['v'][index1]
                length1 = front_left['length'][index1]
                width1 = front_left['width'][index1]
            if not behind_left.empty:
                index4 = behind_left['rel_pos_ind'].idxmax()
                x4 = abs(behind_left['x'][index4] - ego_x)
                y4 = abs(behind_left['y'][index4] - ego_y)
                a4 = abs(behind_left['a'][index4])
                yaw4 = behind_left['yaw'][index4]
                v4 = behind_left['v'][index4]
                length4 = behind_left['length'][index4]
                width4 = behind_left['width'][index4]
            if not front_right.empty:
                index3 = front_right['rel_pos_ind'].idxmin()
                x3 = abs(front_right['x'][index3] - ego_x)
                y3 = abs(front_right['y'][index3] - ego_y)
                a3 = abs(front_right['a'][index3])
                yaw3 = front_right['yaw'][index3]
                v3 = front_right['v'][index3]
                length3 = front_right['length'][index3]
                width3 = front_right['width'][index3]
            if not behind_right.empty:
                index6 = behind_right['rel_pos_ind'].idxmax()
                x6 = abs(behind_right['x'][index6] - ego_x)
                y6 = abs(behind_right['y'][index6] - ego_y)
                a6 = abs(behind_right['a'][index6])
                yaw6 = behind_right['yaw'][index6]
                v6 = behind_right['v'][index6]
                length6 = behind_right['length'][index6]
                width6 = behind_right['width'][index6]

            
            state = [ego_x/10000,ego_y/10000,ego_v/50,ego_a/15,ego_yaw/(math.pi*2),ego_length/10,ego_width/5,\
                    x1/self.sensor_range,y1/self.sensor_range,v1/50,a1/15,yaw1/(math.pi*2),length1/10,width1/5,\
                    x2/self.sensor_range,y2/self.sensor_range,v2/50,a2/15,yaw2/(math.pi*2),length2/10,width2/5,\
                    x3/self.sensor_range,y3/self.sensor_range,v3/50,a3/15,yaw3/(math.pi*2),length3/10,width3/5,\
                    x4/self.sensor_range,y4/self.sensor_range,v4/50,a4/15,yaw4/(math.pi*2),length4/10,width4/5,\
                    x5/self.sensor_range,y5/self.sensor_range,v5/50,a5/15,yaw5/(math.pi*2),length5/10,width5/5,\
                    x6/self.sensor_range,y6/self.sensor_range,v6/50,a6/15,yaw6/(math.pi*2),length6/10,width6/5,\
                    self.x_goal_min/10000,self.x_goal_max/10000,self.y_goal_min/10000,self.y_goal_max/10000]
            state = np.array(state)
            
            # print(state)
            ego_x = self.ego_info.loc['ego', 'x']
            ego_y = self.ego_info.loc['ego', 'y']
            ego_v = self.ego_info.loc['ego', 'v']
            ego_phi = self.ego_info.loc['ego', 'yaw'] * 180 / math.pi  # 转为角度
            ego_lane = self.ego_info.loc['ego', 'lane']
            rel_pos_ind = self.ego_info.loc['ego', 'rel_pos_ind']
            # print('rel_pos_ind')
            ref_path = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']
            # print('ref_path', ref_path)
            ref_phi = self.road_info_dict_pingjie[str(ego_lane)]['phi_road']
            # print('ref_phi', ref_phi)
            ref_v = self.v_max  # 目标速度，已经动态调整

            # 当前目标点
            ref_pt = ref_path[rel_pos_ind]
            global ref_pt1
            ref_pt1 = ref_pt
            # print('ref_pt1', ref_pt1)
            ref_angle = ref_phi[rel_pos_ind] * 180 / math.pi  # 转为角度

            # delta_y: 横向距离
            delta_y = math.sqrt((ego_x - ref_pt[0])**2 + (ego_y - ref_pt[1])**2)
            # print(delta_y)
            # delta_phi: 朝向角误差，注意 wrap 到 [-180, 180]
            delta_phi = ego_phi - ref_angle
            if delta_phi > 180:
                delta_phi -= 360
            elif delta_phi < -180:
                delta_phi += 360

            # delta_v: 速度偏差
            delta_v = ego_v - ref_v
            # print('delta_v', delta_v)
            # 归一化并拼接
            tracking_error = np.array([delta_y / self.sensor_range,
                                    delta_phi / 180,
                                    delta_v / 50])
            # state = np.concatenate([state, tracking_error])
            # print('state2', state.shape)
            # 判断数组中是否存在 NaN 或 Inf
            has_nan_or_inf = np.isnan(state) | np.isinf(state)
            # 将 NaN 和 Inf 替换成 -1
            state[has_nan_or_inf] = -1
        else:  
            # 1. 自车Frenet信息（统一米制）
            # ego_x = self.ego_info.loc['ego', 'x'] / 1000             # 自车位置x
            # ego_y = self.ego_info.loc['ego', 'y'] / 1000          # 自车位置y
            goal_x = self.goal_center_x / 1000
            goal_y = self.goal_center_y / 1000
            ego_v = self.ego_info.loc['ego', 'v']            # 自车速度
            ego_a = self.ego_info.loc['ego', 'a']            # 自车加速度
            ego_pos = np.array([ego_x, ego_y])
            ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)

            # 找到最近的参考点
            distances = np.linalg.norm(ref_center - ego_pos, axis=1)
            nearest_idx = np.argmin(distances)

            # 纵向弧长（m）
            ego_s_m = self._s_from_idx(station, int(nearest_idx))
            # print('ego_s_m_before', ego_s_m)
            # 切线与法线
            ref_tangent = np.array([np.cos(ref_phi[nearest_idx]), np.sin(ref_phi[nearest_idx])])
            ref_normal = np.array([-np.sin(ref_phi[nearest_idx]), np.cos(ref_phi[nearest_idx])])

            # 带符号横向偏移（m）
            to_ego = ego_pos - ref_center[nearest_idx]
            ego_d_frenet = float(np.dot(to_ego, ref_normal))
            # print('ego_d_frenet', ego_d_frenet)
            # 速度在 Frenet 坐标系的分解
            ego_vs = float(ego_v * np.dot([np.cos(ego_yaw), np.sin(ego_yaw)], ref_tangent))
            ego_vd = float(ego_v * np.dot([np.cos(ego_yaw), np.sin(ego_yaw)], ref_normal))
            ego_s_m_new, ego_d_frenet_new, ego_seg_idx, ego_alpha, ego_h_unit = self._project_point_to_centerline(
                ego_pos, ref_center, ref_phi, station, nearest_idx
            )
            # 速度在 Frenet 中的分解（基于插值切向/法向）
            ego_n_unit = np.array([-ego_h_unit[1], ego_h_unit[0]], dtype=float)
            ego_vs_new = float(ego_v * np.dot([np.cos(ego_yaw), np.sin(ego_yaw)], ego_h_unit))
            ego_vd_new = float(ego_v * np.dot([np.cos(ego_yaw), np.sin(ego_yaw)], ego_n_unit))
            # print('ego_s_m_after', ego_s_m_new)
            # 航向角误差
            ego_heading_error = ego_yaw - ref_phi[nearest_idx]
            ego_heading_error = np.arctan2(np.sin(ego_heading_error), np.cos(ego_heading_error))

            # 车道边界距离（m）
            lane_width = self.road_info_dict_pingjie[str(ego_lane)]['width'][nearest_idx]
            ego_left_boundary_dist = lane_width/2 + ego_d_frenet
            ego_right_boundary_dist = lane_width/2 - ego_d_frenet

            # 2. 旁车 Frenet 信息
            other_vehicles_frenet = []

            # 使用 frame 中的所有车辆（包含相邻车道），注意：frame['rel_pos_ind'] 已是对“自车当前车道中心线”的最近点索引
            all_cars = frame.copy()
            # print('all_cars', all_cars)
            def _idx_to_s_m(idx):
                return self._s_from_idx(station, int(idx))
            cand = []
            k = 6
            d_max = []
            for name, other_car in all_cars.iterrows():
                # 跳过与自车重合的记录_ind
                if (abs(other_car['x'] - ego_x) < 0.1 and abs(other_car['y'] - ego_y) < 0.1):
                    continue
                # print('other_car', other_car)
                other_idx = int(other_car['rel_pos_ind'])  # 这是在“自车车道中心线”上的索引
                other_pos = np.array([other_car['x'], other_car['y']], dtype=float)
                other_s_m = self._s_from_idx(station, other_idx)

                # 参考切向/法向（仍以“自车车道中心线”为准）
                t_hat = np.array([np.cos(ref_phi[other_idx]), np.sin(ref_phi[other_idx])], dtype=float)
                n_hat = np.array([-np.sin(ref_phi[other_idx]), np.cos(ref_phi[other_idx])], dtype=float)

                # 该车的带符号横向偏移 d（对自车车道中心线）
                to_other = other_pos - ref_center[other_idx]
                other_d = float(np.dot(to_other, n_hat))
                # 该车在 Frenet 下的速度分解
                other_v = float(other_car['v'])
                other_yaw = float(other_car['yaw'])
                other_vs = float(other_v * math.cos(other_yaw - ref_phi[other_idx]))
                other_vd = float(other_v * math.sin(other_yaw - ref_phi[other_idx]))
                other_s_m_new, other_d_new, other_seg_idx, other_alpha, other_h_unit = self._project_point_to_centerline(
                    other_pos, ref_center, ref_phi, station, other_idx
                )
                # 相对 Frenet 距离
                s_diff = float(other_s_m - ego_s_m)
                # print('s_diff', s_diff)
                s_diff_new = float(other_s_m_new - ego_s_m_new)
                # print('s_diff_new', s_diff_new)
                d_diff = float(other_d - ego_d_frenet)
                d_diff_new = float(other_d_new - ego_d_frenet_new)
                # 该车在 Frenet 下的速度分解（基于插值切向/法向）
                other_vs_new = float(other_v * (np.cos(other_yaw) * other_h_unit[0] + np.sin(other_yaw) * other_h_unit[1]))
                other_n_unit = np.array([-other_h_unit[1], other_h_unit[0]], dtype=float)
                other_vd_new = float(other_v * (np.cos(other_yaw) * other_n_unit[0] + np.sin(other_yaw) * other_n_unit[1]))
                dtheta = self._wrap_to_pi(other_yaw - ego_yaw)
                # print('dtheta', abs(dtheta))
                if 2.79 <= abs(dtheta) <= 3.49:
                    continue
                if abs(s_diff_new) <= 30 and abs(d_diff_new) <= 10.0:
                    d_max.append(abs(d_diff_new))
                    # 二级排序键：先 |s_diff|，再 |d_diff|（可选，保证并列时更近车道者优先）
                    # cand.append((abs(s_diff), abs(d_diff), s_diff, d_diff, (other_vs - ego_vs), (other_vd - ego_vd)))
                    cand.append((abs(s_diff_new), abs(d_diff_new), s_diff_new, d_diff_new, (other_vs_new - ego_vs_new), (other_vd_new - ego_vd_new)))
            cand.sort(key=lambda x: (x[0], x[1]))
            out = []
            mask = []
            for _, _, s_diff, d_diff, dvs, dvd in cand[:k]:
                out.extend([s_diff/200.0, d_diff/5.0, dvs/30.0, dvd/30.0])
                mask.append(1.0)
            # 不足 k 辆补零
            while len(out) < 4 * k:
                out.extend([0.0, 0.0, 0.0, 0.0])
                mask.append(0.0)
            try:
                other_vehicles_frenet = out
                other_vehicles_mask = mask
            except UnboundLocalError:
                other_vehicles_frenet = [0.0] * (4 * k) 
                other_vehicles_mask = [0.0] * k
            # print('other_vehicles_frenet', other_vehicles_frenet)
            # print('other_vehicles_mask', other_vehicles_mask)
            # 3. 未来路点（按米推进）
            future_waypoints = []
            for i in range(5):
                future_s_m = float(ego_s_m + (i+1) * 3.0)  # +10m, +20m
                future_idx = self._idx_from_s(station, future_s_m)
                future_idx = min(future_idx, len(ref_center) - 1)

                s_to_future = float(future_s_m - ego_s_m)
                d_to_future = float(0.0 - ego_d_frenet)  # 中心线 d=0

                future_curvature = self.road_info_dict_pingjie[str(ego_lane)]['curvature'][future_idx]
                future_heading = ref_phi[future_idx]

                future_waypoints.extend([
                    s_to_future/200,
                    d_to_future/5,
                    future_curvature/0.02,
                    future_heading/(2*math.pi)
                ])
            
            # 读 reset 预计算的左右最大偏移；若缺省则退化为本车道半宽
            def _pick(arr, i):
                return float(arr[min(max(i,0), len(arr)-1)])
            ego_idx  = int(self.ego_info.loc['ego','rel_pos_ind'])
            if hasattr(self, "lateral_limits") and ego_lane in self.lateral_limits:
                left_max  = _pick(self.lateral_limits[ego_lane]["left_max"],  ego_idx)
                right_max = _pick(self.lateral_limits[ego_lane]["right_max"], ego_idx)
            else:
                w = self.road_info_dict_pingjie[str(ego_lane)]["width"]
                half = (_pick(w, ego_idx)/2.0) if len(w) else 1.75
                left_max = right_max = half

            # 二次保护（避免过大/过小；按你原参数）
            left_max  = max(0.8, min(left_max,  4.0))
            right_max = max(0.8, min(right_max, 4.0))

            # d_min, ttc_min = self._nearest_dist_and_ttc_xy(ego_x, ego_y, ego_v, ego_yaw)
            # print('d_min, ttc_min', d_min, ttc_min)
            goal_xy = np.array([self.goal_center_x, self.goal_center_y], dtype=float)
            # 最近中心线索引
            goal_idx = int(np.argmin(np.linalg.norm(ref_center - goal_xy, axis=1)))
            s_goal = self._s_from_idx(station, goal_idx)
            # 在你的 get_features 方法中找到这部分代码：
            goal_xy = np.array([self.goal_center_x, self.goal_center_y], dtype=float)
            # 最近中心线索引
            goal_idx = int(np.argmin(np.linalg.norm(ref_center - goal_xy, axis=1)))
            s_goal = self._s_from_idx(station, goal_idx)

            # ============ 新增：计算 d_goal ============ 
            # 1. 找到目标点所在的车道中心线最近点
            goal_on_center = ref_center[goal_idx]
            # 2. 计算该点处的法向量
            phi_goal = ref_phi[goal_idx]
            n_hat_goal = np.array([-np.sin(phi_goal), np.cos(phi_goal)])
            # 3. 计算目标点相对于中心线的横向偏移
            vector_to_goal = goal_xy - goal_on_center
            d_goal_absolute = np.dot(vector_to_goal, n_hat_goal)

            # 4. 计算相对于自车的横向偏移（关键！）
            d_goal_relative = d_goal_absolute - ego_d_frenet

            # 5. 归一化（可选，但推荐）
            d_goal_normalized = d_goal_relative / 5.0  # 假设横向最大偏移为5米
            # print('s_goal', s_goal)
            phi_merge = None
            # if self.s_merge is None:
            #     # self.s_merge = self.compute_merge_s(
            #     #                                     ego_lane=ego_lane,
            #     #                                     ego_idx=ego_idx,
            #     #                                     kappa_thr=0.02,
            #     #                                     consec=6,
            #     #                                     back_offset_m=3,
            #     #                                     search_ahead_pts=200
            #     #                                 )
            #     self.s_merge = self.compute_merge_s_precise(
            #                                         ego_lane=ego_lane,
            #                                         ego_idx=ego_idx,
            #                                         kappa_thr=0.02,
            #                                         consec=8,
            #                                         min_back_m = 2.0,
            #                                         max_back_m = 10.0,
            #                                         t_buffer = 0.6,        # 速度相关回退（秒）
            #                                         search_ahead_pts = 300,
            #                                         smooth_win = 9, 
            #                                     )
            # # print(702, max(d_max))
            # # print(703, ego_s_m)
            # # print(716, self.s_merge)
            # # print('s_merge', self.s_merge)
            #     merge_x, merge_y = self.convert_frenet_to_cartesian(
            #                                                         self.s_merge,
            #                                                         0.0,
            #                                                         ref_center,
            #                                                         ref_phi,
            #                                                         station
            #                                                     )
            #     self.merge_x, self.merge_y = merge_x, merge_y
            # 2) 计算自车到达时间（笛卡尔）
            # t_ego_to_merge = self._time_to_point_cartesian(
            #     x=ego_x, y=ego_y, v=ego_v, yaw=ego_yaw, x_tar=self.merge_x, y_tar=self.merge_y
            # )
            # self.t_to_merge_ego = float(t_ego_to_merge)
            # print(805, self.t_to_merge_ego)
            # print(806, t_o_to_merge)
            # 组装状态
            # print(ego_d_frenet)
            frenet_state = [
                ego_s_m/200,                # 纵向位置（m）
                ego_d_frenet/5,                 # 横向偏移（m）
                ego_vs/30,                      # 纵向速度
                ego_a/5,
                ego_heading_error/math.pi,      # 航向角误差
                *other_vehicles_frenet,         # 旁车
                *other_vehicles_mask,           # 旁车掩码
                *future_waypoints,
                left_max/5,                     # 最大左偏移
                right_max/5,                     # 最大右偏移   
                # min(d_min / 10.0, 1.0),          # 归一化最小距离（d_safe=5.0）
                # ttc_min,        # 归一化最小TTC（t_safe=2.5）  e      
                s_goal/200,                       # 目标点纵向位置（m） 
                # d_goal_normalized,               # 目标点横向位置（m
                # ego_x / 1000,
                # ego_y / 1000,
                # goal_x,
                # goal_y,
            ]

            frenet_state = np.array(frenet_state, dtype=np.float32)
            has_nan_or_inf = np.isnan(frenet_state) | np.isinf(frenet_state)
            frenet_state[has_nan_or_inf] = -1
            state = frenet_state
            # print('front', self.front)
            # print('ego_info', self.ego_info)
            # print('frenet_state', frenet_state[-4:])
            # print('frenet_bucket_names', self.frenet_bucket_names)
            # import time
            # time.sleep(0.5)
        # self._update_ego_lane_with_hysteresis()
        return state
    
    def _to_obs(self, state):
        # image: dummy array
        # image = np.zeros((64, 64, 3), dtype=np.uint8)
        # velocity: flatten entire state vector
        velocity = np.array(state, dtype=np.float32)
        return {
            # 'image':    image,
            'velocity': velocity,
        }
    
    def _build_offset_ref_traj(self, ego_lane: int, start_idx: int, d_ref: float, horizon: int = 20, step_stride: int = 2):
        """沿当前车道中心线取 N 个点，并按左法线整体平移 d_ref，生成 (x,y,psi) 参考轨迹。"""
        # print('pianyi', d_ref)
        ref = self.road_info_dict_pingjie[str(ego_lane)]
        C   = ref['center_vertices']
        Phi = ref['phi_road']
        idxs = [min(start_idx + i*step_stride, len(C)-1) for i in range(horizon)]
        pts  = C[idxs]
        phs  = Phi[idxs]
        nx = -np.sin(phs); ny = np.cos(phs)       # 左法线
        pts_off = pts + np.stack([nx, ny], axis=1) * d_ref
        # print('pts_off', pts_off)
        ref_traj = np.column_stack([pts_off[:,0], pts_off[:,1], phs])
        global ref_pianyi
        ref_pianyi = pts_off
        return ref_traj

    def _signed_lateral_error(self, pt_xy: np.ndarray, ref_pts: np.ndarray, idx: int):
        """以参考线左法线为正方向，计算带符号横向误差。"""
        idx2 = min(idx+1, len(ref_pts)-1)
        t = ref_pts[idx2] - ref_pts[idx]
        n = np.array([-t[1], t[0]])
        n = n / (np.linalg.norm(n) + 1e-8)
        e = pt_xy - ref_pts[idx]
        return float(np.dot(e, n))

    def _find_lookahead_target(self, pos_xy: np.ndarray, ref_traj_xypsi: np.ndarray, Ld: float):
        """在离当前位置累计弧长约 Ld 的参考点作为目标点（逐段累加近似）。"""
        acc_len = 0.0
        for i in range(len(ref_traj_xypsi)-1):
            p0 = ref_traj_xypsi[i, :2]
            p1 = ref_traj_xypsi[i+1, :2]
            seg = np.linalg.norm(p1 - p0)
            acc_len += seg
            if acc_len >= Ld:
                return ref_traj_xypsi[i+1, :]
        return ref_traj_xypsi[-1, :]
    
    def cal_control_track(self, action_vec):
        """
        纯跟踪控制：RL 动作为 [v_ref_norm, d_ref_norm] ∈ [-1,1]^2。
        1) 将动作映射到物理量 v_ref ∈ [0, v_max]、d_ref ∈ [-d_max, d_max]
        2) 依据 d_ref 生成偏移参考轨迹
        3) 纵向：PID 跟踪 v_ref -> acc
           横向：Pure-Pursuit/Stanley 混合 -> rot（航向角速度）
        4) 经 jiaohe_* 模块做限幅/光滑，返回 [acc, rot]
        """
        assert len(self.ego_info) != 0, "get_features() 需先执行以更新 ego_info"
        ego_lane = int(self.ego_info.loc['ego','lane'])
        rel_idx  = int(self.ego_info.loc['ego','rel_pos_ind'])

        # === 1) 动作映射 ===
        v_max = getattr(self, 'v_max', self.v_phys_max)
        v_max = max(2.0, min(float(v_max), self.v_phys_max))   # 保护
        v_max = 30.0
        # print(action_vec[0])
        v_ref = float((action_vec[0] + 1.0)/2.0) * v_max
        # print('v_ref', v_ref)
        width_arr = self.road_info_dict_pingjie[str(ego_lane)]['width']
        half_w = float(width_arr[min(rel_idx, len(width_arr)-1)]/2.0) if len(width_arr) else self.d_phys_max_default
        d_max = max(0.8, min(half_w, 3.0)) #* 3 / 2
        d_ref = float(action_vec[1]) * d_max

        self._last_vref, self._last_dref = v_ref, d_ref  # 记录给奖励/诊断

        # === 2) 偏移参考轨迹 ===
        N  = 20
        ref_traj = self._build_offset_ref_traj(ego_lane, rel_idx, d_ref, horizon=N, step_stride=2)

        # === 3) 纵向 PID + 横向纯跟踪 ===
        x  = float(self.ego_info.loc['ego','x'])
        y  = float(self.ego_info.loc['ego','y'])
        v  = float(self.ego_info.loc['ego','v'])
        a0 = float(self.ego_info.loc['ego','a'])
        psi= float(self.ego_info.loc['ego','yaw'])
        rot_now = float(self.ego_rot)

        # 纵向：a_cmd = Kp*(v_ref - v) + Ki*∫ + Kd*(0 - a_now)（简化用 PD）
        ev = v_ref - v
        a_cmd = 0.6*ev - 0.05*a0
        a_cmd = float(np.clip(a_cmd, -3.0, 2.0))

        # 横向：目标点 Pure-Pursuit + Stanley 误差项
        pos = np.array([x, y], dtype=float)
        Ld  = np.clip(0.8*v + 3.0, 3.0, 15.0)                # 自适应前视
        xt  = self._find_lookahead_target(pos, ref_traj, Ld)
        vec_target = xt[:2] - pos
        heading_err = np.arctan2(vec_target[1], vec_target[0]) - psi
        # 归一化到 [-pi, pi]
        heading_err = (heading_err + np.pi)%(2*np.pi) - np.pi

        # 带符号横向误差（当前索引处）
        ref_center = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']
        e_lat = self._signed_lateral_error(pos, ref_center, rel_idx) - d_ref

        # Stanley/Pure-Pursuit 混合（输出航向角速度 rot_cmd）
        rot_cmd = 1.2*heading_err + np.arctan2(0.6*e_lat, (v + 0.1))
        rot_cmd = float(np.clip(rot_cmd, -0.5, 0.5))

        # 一阶滤波（防尖峰）
        u = np.array([a_cmd, rot_cmd], dtype=float)
        u = 0.7*self._u_prev + 0.3*u
        self._u_prev = u.copy()
        a_cmd, rot_cmd = float(u[0]), float(u[1])

        # === 4) 用你已有的“交合”模块做物理约束/平滑（若已在 reset() 初始化）===
        if self.ego_info.loc['ego','v'] >= 100:
            # 顺序与原 cal_control 一致：yundong -> dongli_1 -> dongli_2 -> dongli_8_new
            a_cmd, rot1 = self.jiaohe_yun.update(v_now=v,a_now=a0,yaw_now=psi,rot_now=rot_now,
                                                 a_chushi=a_cmd, rot_chushi=rot_cmd, ego_info=self.ego_info)
            a_cmd, rot2 = self.jiaohe_dong_1.update(v_now=v,a_now=a0,yaw_now=psi,rot_now=rot_now,
                                                    a_chushi=a_cmd, rot_chushi=rot_cmd, ego_info=self.ego_info)
            a_cmd, rot3 = self.jiaohe_dong_2.update(v_now=v,a_now=a0,yaw_now=psi,rot_now=rot_now,
                                                    a_chushi=a_cmd, rot_chushi=rot_cmd, ego_info=self.ego_info)
            rot_cmd = min([rot1,rot2,rot3], key=lambda rr: abs(rr))
            a_cmd, rot_cmd = self.jiaohe_dong_8_new.update(v_now=v,a_now=a0,yaw_now=psi,rot_now=rot_now,
                                                           a_chushi=a_cmd, rot_chushi=rot_cmd, ego_info=self.ego_info)

        return [a_cmd, rot_cmd]
    
    def plan_trajectory_segmented(self, actions):
        """
        分段规划策略，避免倒退回转
        """
        planning_horizon = float(actions[0]) + 1.01
        target_velocity = (float(actions[2]) + 1) * 15 + 0.01
        
        ego_v = float(self.ego_info.loc['ego','v'])
        ego_lane = int(self.ego_info.loc['ego','lane'])
        ego_idx = int(self.ego_info.loc['ego','rel_pos_ind'])
        
        ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)
        s0_m = self._s_from_idx(station, ego_idx)
        
        # 分段规划：先加速/减速，再匀速
        if abs(target_velocity - ego_v) > 1.0:  # 需要速度变化
            # 第一段：速度调整
            t1 = min(1.0, planning_horizon * 0.6)  # 速度调整时间
            s1 = s0_m + (ego_v + target_velocity) * t1 / 2  # 中间位置
            
            # 第二段：匀速
            t2 = planning_horizon - t1
            s2 = s1 + target_velocity * t2
            
            # 分段求解
            coeffs1 = self._solve_longitudinal_polynomial(
                t0=0, s0=s0_m, v0=ego_v, a0=float(self.ego_info.loc['ego','a']),
                tf=t1, sf=s1, vf=target_velocity, af=0
            )
            
            coeffs2 = self._solve_longitudinal_polynomial(
                t0=t1, s0=s1, v0=target_velocity, a0=0,
                tf=planning_horizon, sf=s2, vf=target_velocity, af=0
            )
            
            # 生成分段轨迹
            trajectory = []
            dt = 0.1
            t = 0.0
            
            while t <= planning_horizon + 1e-6:
                if t <= t1:
                    s_m = self._evaluate_polynomial(coeffs1, t)
                else:
                    s_m = self._evaluate_polynomial(coeffs2, t)
                
                # 确保单调性
                if len(trajectory) > 0 and s_m < trajectory[-1][0]:
                    s_m = trajectory[-1][0] + 0.01  # 强制递增
                
                trajectory.append((s_m, 0, t))  # 横向偏移设为0
                t += dt
                
            return trajectory, []
        else:
            # 速度接近目标值，直接匀速规划
            return self.plan_trajectory_polynomial(actions)
    
    def plan_trajectory_polynomial(self, actions):
        """
        基于三维动作求解纵向四次多项式和横向五次多项式规划路径
        
        Args:
            actions: 2维动作 [lateral_offset, target_velocity]
        
        Returns:
            trajectory: 规划路径点列表 [(s, d, t), ...]
        """
        def _d_dt(eval_fn, coeffs, t, h):
        # f'(t) ~= (f(t+h) - f(t-h)) / (2h)
            return (float(eval_fn(coeffs, t + h)) - float(eval_fn(coeffs, t - h))) / (2.0 * h)
        # planning_horizon = (float(actions[0]) + 1.0) * 3 + 0.1
        ego_lane = self.ego_info.loc['ego','lane']
        # print('ego_lane', ego_lane)
        # ——确保绑定到最近的“同向”车道（若最近车道逆向，会切到同向那条）——
        self._ensure_forward_lane_and_index(cos_thresh=0.5)

        # ——确保 ego_lane/rel_pos_ind 已更新（若你在别处已调用可省略）——
        if hasattr(self, "_update_ego_lane_with_hysteresis"):
            self._update_ego_lane_with_hysteresis()
            # print('1330ego_lane', ego_lane)

        # 当前车道与索引
        ego_idx  = int(self.ego_info.loc['ego','rel_pos_ind'])

        # 读 reset 预计算的左右最大偏移；若缺省则退化为本车道半宽
        def _pick(arr, i):
            return float(arr[min(max(i,0), len(arr)-1)])

        if hasattr(self, "lateral_limits") and ego_lane in self.lateral_limits:
            left_max  = _pick(self.lateral_limits[ego_lane]["left_max"],  ego_idx)
            right_max = _pick(self.lateral_limits[ego_lane]["right_max"], ego_idx)
        else:
            w = self.road_info_dict_pingjie[str(ego_lane)]["width"]
            half = (_pick(w, ego_idx)/2.0) if len(w) else 1.75
            left_max = right_max = half

        # 二次保护（避免过大/过小；按你原参数）
        left_max  = max(0.8, min(left_max,  4.0))
        right_max = max(0.8, min(right_max, 4.0))
        # print('left_max', left_max)
        # print('right_max', right_max)
        # 动作 → 不对称 lateral_offset（左正右负）
        u = float(np.clip(actions[0], -1.0, 1.0))
        lateral_offset = u * (left_max if u >= 0.0 else right_max)
        # print(lateral_offset)
        # lateral_offset = 0
        target_velocity = (float(actions[1]) + 1 ) * 6 + 0.01
        # planning_horizon = self._compute_planning_horizon_1(target_velocity, lateral_offset)
        planning_horizon = 1.5
        # print('target_velocity', target_velocity)
        # target_velocity = 10
        # ego_lane = int(self.ego_info.loc['ego','lane'])
        ego_v    = float(self.ego_info.loc['ego','v'])
        ego_yaw  = float(self.ego_info.loc['ego','yaw'])
        delta_v = abs(target_velocity - ego_v)
        ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)
        delta    = ego_yaw - ref_phi[ego_idx]          # 航向相对中心线
        phi_dot  = ego_v * float(ref_kappa[ego_idx])   # 中心线角速度
        yaw_rate = float(getattr(self, 'ego_rot', phi_dot)) 
        # s 用米制弧长
        s0_m = self._s_from_idx(station, ego_idx)
        # print('s0_m', s0_m)
        current_acceleration = float(self.ego_info.loc['ego','a'])
        cal_acc = (target_velocity - ego_v) / ( planning_horizon ) 
        # print('cal_acc', cal_acc)
        sf_m = s0_m + ego_v * planning_horizon + 0.5 * cal_acc * planning_horizon**2
        # 纵向四次多项式（米）
        s_coeffs = self._solve_longitudinal_polynomial_monotonic(
            t0=0, s0=s0_m, v0=ego_v, a0=current_acceleration,
            tf=planning_horizon, sf=sf_m, vf=target_velocity, af=0
        )
        # 当前横向有符号偏移（米）
        ego_pos = np.array([self.ego_info.loc['ego','x'], self.ego_info.loc['ego','y']], dtype=float)
        ego_d_signed, _ = self._calculate_signed_lateral_offset(ego_pos, ego_idx, ref_center, ref_phi)
        ego_a    = float(self.ego_info.loc['ego','a'])
        ddot0  = ego_a * math.sin(delta) + ego_v * math.cos(delta) * (yaw_rate - phi_dot)
        # 横向五次多项式（米）
        d_coeffs = self._solve_lateral_polynomial(
            t0=0, d0=ego_d_signed, dd0=ego_v * math.sin(ego_yaw - ref_phi[ego_idx]), ddot0=ddot0,
            tf=planning_horizon, df=lateral_offset, ddf=0, ddotf=0
        )
        # 生成轨迹：(s_m, d, t)
        trajectory = []
        yaw_list, v_list = [], []
        # 车体尺寸（按你的项目来源设置
        ego_length = float(getattr(self, "ego_length", 4.7))
        ego_width = float(getattr(self, "ego_width", 1.8))
        dt = self.scenario_dt
        # print('dt', dt)
        h = min(1e-3, dt * 0.25)        
        t = 0.0
        s_last = station[-1]
        last_s = s0_m
        while t <= planning_horizon + 1e-6:
            s_m = float(self._evaluate_polynomial(s_coeffs, t))
            # print(1, s_m)
            # print(3, planning_horizon)
            d   = float(self._evaluate_polynomial(d_coeffs, t))
            # 强制单调性：s不能减小
            if s_m < last_s:
                s_m = last_s + 0.001  # 最小增量
            s_m = float(np.clip(s_m, station[0], s_last))
            # print(2, station[0])
            trajectory.append((s_m, d, t))
            last_s = s_m
            
            # —— 速度与航向 —— #
            idx = int(np.clip(np.searchsorted(station, s_m, side='left'), 0, len(station)-1))
            phi   = float(ref_phi[idx])    # 中心线航向
            kappa = float(ref_kappa[idx])  # 中心线曲率

            # 2) 一阶导（中心差分近似）
            sdot = _d_dt(self._evaluate_polynomial, s_coeffs, t, h)
            ddot = _d_dt(self._evaluate_polynomial, d_coeffs, t, h)

                    # 3) 速度与航向（Frenet → 世界坐标）
                 # v = sqrt( (s_dot*(1 - kappa*d))^2 + (d_dot)^2 )
                # yaw = phi + atan2(d_dot, s_dot*(1 - kappa*d))
            one_minus_kd = (1.0 - kappa * d)
            v = math.sqrt((sdot * one_minus_kd) ** 2 + (ddot ** 2))
            yaw = phi + math.atan2(ddot, sdot * one_minus_kd)

            v_list.append(float(v))
            yaw_list.append(float(yaw))
            t += dt
        xy = [point[:2] for point in trajectory]
        # print(xy)
        ref = []
        for s_m, d, _ in trajectory:
            x, y = self.convert_frenet_to_cartesian(s_m, d, ref_center, ref_phi, station)
            ref.append((x, y))
        vref = target_velocity
        dref = lateral_offset
        vd = [vref, dref]
        cost_trajectory = {
            "pos_list": np.array(ref, dtype=float),  # (T,2) -> 下游几何/碰撞使用
            "orientation_list": np.array(yaw_list, dtype=float),  # (T,)
            "v_list": np.array(v_list, dtype=float),  # (T,)
            "shape": {"length": ego_length, "width": ego_width},
        }
        return trajectory, ref, vd, cost_trajectory
    
    def _calculate_signed_lateral_offset(self, ego_pos, ego_s, ref_path, ref_phi):
        """统一的横向偏移计算函数"""
        idx = int(np.clip(ego_s, 0, len(ref_path)-2))
        
        # 计算路径切线方向
        t = ref_path[idx+1] - ref_path[idx]
        t_norm = np.linalg.norm(t)
        if t_norm > 1e-8:
            t = t / t_norm
        
        # 计算法线方向（左法线为正）
        n = np.array([-t[1], t[0]])
        
        # 计算有符号横向偏移
        to_ego = ego_pos - ref_path[idx]
        lateral_offset = np.dot(to_ego, n)
        
        return lateral_offset, idx
    
    def _solve_longitudinal_polynomial(self, t0, s0, v0, a0, tf, sf, vf, af ):
        """
        求解纵向四次多项式 s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
        
        边界条件：
        s(t0) = s0, s'(t0) = v0, s''(t0) = a0
        s(tf) = sf, s'(tf) = vf, s''(tf) = af
        """
        # 构建系数矩阵
        A = np.array([
            [1, t0, t0**2, t0**3, t0**4],
            [0, 1, 2*t0, 3*t0**2, 4*t0**3],
            [0, 0, 2, 6*t0, 12*t0**2],
            [1, tf, tf**2, tf**3, tf**4],
            [0, 1, 2*tf, 3*tf**2, 4*tf**3],
            # [0, 0, 2, 6*tf, 12*tf**2]
        ])
        
        # b = np.array([s0, v0, a0, vf, af])
        b = np.array([s0, v0, a0, sf, vf])
        
        # 求解系数
        try:
            coeffs = np.linalg.solve(A, b)
            return coeffs
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            print('juzhenqiyi')
            coeffs = np.linalg.pinv(A) @ b
            return coeffs
    
    def _solve_longitudinal_polynomial_monotonic(self, t0, s0, v0, a0, tf, sf, vf, af):
        """
        求解单调递增的纵向四次多项式
        """
        # 首先尝试标准解法
        coeffs = self._solve_longitudinal_polynomial(t0, s0, v0, a0, tf, sf, vf, af)
        
        # 检查单调性
        dt = 0.01
        t_check = np.arange(t0, tf, dt)
        s_values = [self._evaluate_polynomial(coeffs, t) for t in t_check]
        
        # 检查是否有递减
        for i in range(1, len(s_values)):
            if s_values[i] < s_values[i-1]:
                # 如果发现递减，使用更保守的规划
                return self._solve_conservative_polynomial(t0, s0, v0, a0, tf, sf, vf, af)
        
        return coeffs

    def _solve_conservative_polynomial(self, t0, s0, v0, a0, tf, sf, vf, af):
        """
        保守的规划策略：使用三次多项式或分段规划
        """
        # 使用三次多项式，减少振荡
        A = np.array([
            [1, t0, t0**2, t0**3],
            [0, 1, 2*t0, 3*t0**2],
            [1, tf, tf**2, tf**3],
            [0, 1, 2*tf, 3*tf**2]
        ])
        
        b = np.array([s0, v0, sf, vf])
        
        try:
            coeffs = np.linalg.solve(A, b)
            # 补零到5个系数
            return np.append(coeffs, [0, 0])
        except np.linalg.LinAlgError:
            coeffs = np.linalg.pinv(A) @ b
            return np.append(coeffs, [0, 0])
        
    def _solve_lateral_polynomial(self, t0, d0, dd0, ddot0, tf, df, ddf, ddotf):
        """
        求解横向五次多项式 d(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        
        边界条件：
        d(t0) = d0, d'(t0) = dd0, d''(t0) = ddot0
        d(tf) = df, d'(tf) = ddf, d''(tf) = ddotf
        """
        # 构建系数矩阵
        A = np.array([
            [1, t0, t0**2, t0**3, t0**4, t0**5],
            [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
            [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
            [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
        ])
        
        b = np.array([d0, dd0, ddot0, df, ddf, ddotf])
        
        # 求解系数
        try:
            coeffs = np.linalg.solve(A, b)
            return coeffs
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            # print('juzhenqiyi')
            coeffs = np.linalg.pinv(A) @ b
            return coeffs

    def _evaluate_polynomial(self, coeffs, t):
        """计算多项式在时间t处的值"""
        result = 0
        for i, coeff in enumerate(coeffs):
            result += coeff * (t ** i)
        return result

    def _denormalize_action(self, normalized_action, action_type):
        """将归一化的动作反归一化到实际范围"""
        min_val, max_val = self.action_ranges[action_type]
        return min_val + (normalized_action + 1) * (max_val - min_val) / 2

    def _normalize_action(self, action, action_type):
        """将实际动作归一化到[-1, 1]范围"""
        min_val, max_val = self.action_ranges[action_type]
        return 2 * (action - min_val) / (max_val - min_val) - 1
    
    def execute_trajectory_control(self, trajectory):
        """
        执行轨迹跟踪控制
        
        Args:
            trajectory: 规划轨迹 [(s, d, t), ...]
        
        Returns:
            control_commands: [acceleration, steering]
        """
        if trajectory is None or len(trajectory) < 2:
            return np.array([0.0, 0.0])
        
        # 更新当前轨迹
        self.current_trajectory = trajectory
        if self.trajectory_start_time == 0.0:
            self.trajectory_start_time = self.observation.time if hasattr(self.observation, 'time') else 0.0
        
        # 获取当前车辆状态
        ego_s = self.ego_info.loc['ego', 'rel_pos_ind']
        ego_d = self.ego_info.loc['ego', 'dis_ind']
        ego_v = self.ego_info.loc['ego', 'v']
        ego_yaw = self.ego_info.loc['ego', 'yaw']
        ego_length = self.ego_info.loc['ego', 'length']
        # 获取道路信息
        ego_lane = self.ego_info.loc['ego', 'lane']
        ref_path = self.road_info_dict_pingjie[str(ego_lane)]['center_vertices']
        ref_phi = self.road_info_dict_pingjie[str(ego_lane)]['phi_road']
        
        # 1. 纵向PID控制
        acceleration = self._longitudinal_pid_control(trajectory, ego_s, ego_v)
        
        # 2. 横向Stanley控制
        steering = self._pp_follow_quintic(trajectory)
        
        # 3. 应用控制限制
        acceleration = np.clip(acceleration, self.trajectory_controller.max_deceleration, 
                            self.trajectory_controller.max_acceleration)
        steering = np.clip(steering, -self.trajectory_controller.max_steering_angle, 
                        self.trajectory_controller.max_steering_angle)
        # v = float(self.Env.Env.ego_info.loc['ego','v'] if hasattr(self, 'Env') and hasattr(self.Env, 'Env') else self.Env.ego_info.loc['ego','v'])
        # L = float(self.Env.Env.ego_info.loc['ego','length'] if hasattr(self, 'Env') and hasattr(self.Env, 'Env') else self.Env.ego_info.loc['ego','length']) / 1.7
        # rot = ego_v / max(ego_length, 1e-3) * np.tan(steering)

        return np.array([acceleration, steering])
 

    def _longitudinal_pid_control(self, trajectory, ego_s_idx, ego_v):
        """纵向控制：速度环PID + 少量位置辅助（全部米/秒单位）"""
        if not trajectory or len(trajectory) < 2:
            return 0.0

        ego_lane = int(self.ego_info.loc['ego','lane'])
        _, _, _, station = self._get_lane_arrays(ego_lane)
        ego_s_m = self._s_from_idx(station, int(ego_s_idx))
        ego_s_m = self.ego_s_m
        # 计算 ego 带符号 d（米）供最近点权重（保持原逻辑）
        ref_center, ref_phi, _, _ = self._get_lane_arrays(ego_lane)
        ego_pos = np.array([self.ego_info.loc['ego','x'], self.ego_info.loc['ego','y']], dtype=float)
        ego_d_signed, _ = self._calculate_signed_lateral_offset(ego_pos, int(ego_s_idx), ref_center, ref_phi)

        target_index = self._find_nearest_trajectory_point(trajectory, ego_s_m, ego_d_signed)
        if target_index >= len(trajectory):
            return 0.0

        target_s_m, target_d, target_t = trajectory[target_index]

        # ======= 方案A：按“时间预瞄 tau”来计算 v_ref（关键改动）=======
        # 1) 计算当前轨迹相对时间 t_now
        #    execute_trajectory_control() 首次执行时会设置 self.trajectory_start_time = observation.time
        #    若拿不到，则回退到“就地相邻点差分”的保底方案
        try:
            t_obs = float(getattr(self.observation, 'time', 0.0))
            t_now = float(t_obs - getattr(self, 'trajectory_start_time', 0.0))
            if not np.isfinite(t_now):
                t_now = trajectory[target_index][2]
        except Exception:
            t_now = trajectory[target_index][2]

        # 2) 选择预瞄时间（可按需要调大/调小）
        if self.ego_info.loc['ego','v'] >= 0 and self.ego_info.loc['ego','v'] < 5:
            tau = 2 
        elif self.ego_info.loc['ego','v'] >= 5 and self.ego_info.loc['ego','v'] < 10:
            tau = 1.5
        elif self.ego_info.loc['ego','v'] >= 10 and self.ego_info.loc['ego','v'] < 15:
            tau = 1
        else:
            tau = 1
        # tau = 1.0  # 0.3~0.6 都可以，越大越“前瞻”，刹车更果断
        # print('t_now', t_now)
        # 3) 在轨迹里找到 t >= t_now + tau 的未来点索引
        target_t_future = t_now + tau
        i_future = target_index
        N = len(trajectory)
        while i_future < N - 1 and trajectory[i_future][2] < target_t_future:
            i_future += 1

        # 4) 用未来两点的 (s,t) 做差分，得到 v_ref（严格基于时间）
        i1 = max(0, min(i_future,     N - 2))
        i2 = min(i_future+1, N - 1)
        s1, _, t1 = trajectory[i1]
        s2, _, t2 = trajectory[i2]
        dt = max(t2 - t1, 1e-3)
        v_ref = (s2 - s1) / dt
        # ======= 方案A 结束 =======

        # 速度环 PID（保持原逻辑）
        v_err = v_ref - float(ego_v)
        pid_dt = self.scenario_dt
        a_cmd = float(self.velocity_pid.update(v_err, pid_dt))

        # 少量位置辅助项（m -> m/s^2）（保持原逻辑）
        pos_err_m = target_s_m - ego_s_m
        # a_cmd += self.k_pos_assist * pos_err_m

        # 限幅（保持原逻辑）
        a_cmd = float(np.clip(a_cmd, self.trajectory_controller.max_deceleration,
                                        self.trajectory_controller.max_acceleration))
        return a_cmd
    
    def _wrap_to_pi(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi
    
    def _lateral_stanley_control(self, trajectory, ego_s_idx, ego_d_uns, ego_v, ego_yaw, ref_phi_unused):
        """横向Stanley：按米制 s 取参考，d_s 用米制导数，psi_traj 推导有效"""
        if not trajectory or len(trajectory) < 2:
            return 0.0

        ego_lane = int(self.ego_info.loc['ego','lane'])
        ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)

        # ego s 米制与带符号 d
        ego_s_m = self._s_from_idx(station, int(ego_s_idx))
        ego_pos = np.array([self.ego_info.loc['ego','x'], self.ego_info.loc['ego','y']], dtype=float)
        ego_d_signed, _ = self._calculate_signed_lateral_offset(ego_pos, int(ego_s_idx), ref_center, ref_phi)

        # 目标（s_m, d）
        target_index = self._find_nearest_trajectory_point(trajectory, ego_s_m, ego_d_signed)
        if target_index >= len(trajectory):
            return 0.0
        target_s_m, target_d, _ = trajectory[target_index]

        # 找到对应中心线段索引
        idx = self._idx_from_s(station, target_s_m)

        # 切向/法向
        t_vec = ref_center[idx+1] - ref_center[idx]
        t_hat = t_vec / (np.linalg.norm(t_vec) + 1e-8)
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)

        # 偏移后的参考点位置
        P_ref  = ref_center[idx]
        P_traj = P_ref + target_d * n_hat

        # d_s 近似：(dd/dt)/(ds/dt)，用邻点（米/秒）
        if target_index < len(trajectory) - 1:
            s1, d1, t1 = trajectory[target_index]
            s2, d2, t2 = trajectory[target_index + 1]
        else:
            s2, d2, t2 = trajectory[target_index]
            s1, d1, t1 = trajectory[target_index - 1]
        dt = max(t2 - t1, 1e-3)
        dsdt = (s2 - s1) / dt
        ddt  = (d2 - d1) / dt
        d_s  = ddt / max(dsdt, 1e-3)

        kappa = float(ref_kappa[idx])
        psi_traj = float(ref_phi[idx]) + math.atan2(d_s, 1.0 - kappa * target_d)

        # 误差
        lateral_error = float(np.dot(ego_pos - P_traj, n_hat))
        heading_error = self._wrap_to_pi(psi_traj - float(ego_yaw))

        # Stanley 控制律
        k = self.trajectory_controller.stanley_k
        ksoft = self.trajectory_controller.stanley_ksoft
        v_term = max(float(ego_v), 0.1)
        steering_angle = heading_error + math.atan2(k * lateral_error, v_term + ksoft)
        steering_angle = float(np.clip(steering_angle, -self.trajectory_controller.max_steering_angle,
                                                        self.trajectory_controller.max_steering_angle))
        return steering_angle
    def _pp_follow_quintic(self, trajectory):
        if not trajectory or len(trajectory) < 2:
            return 0.0, 0.0  # acc, steering/rot

        ego_lane = int(self.ego_info.loc['ego','lane'])
        ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)

        # 把 (s,d) -> XY
        pts_xy = []
        for (s_m, d_m, _) in trajectory:
            idx = self._idx_from_s(station, s_m)
            idx = min(max(idx, 0), len(ref_center)-2)
            t_vec = ref_center[idx+1] - ref_center[idx]
            t_hat = t_vec / (np.linalg.norm(t_vec) + 1e-8)
            n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
            P = ref_center[idx] + float(d_m) * n_hat
            pts_xy.append(P)
        ref_traj_xy = np.array(pts_xy, dtype=float)  # (N, 2)

        # 纵向：简单速度PD到目标速度（可从轨迹推/或保持现有PID）
        v = float(self.ego_info.loc['ego','v'])
        a0 = float(self.ego_info.loc['ego','a'])
        v_ref = max(v, 0.0)  # 若无v_ref，可先用当前v（或你已有速度规划）
        a_cmd = 0.6 * (v_ref - v) - 0.05 * a0
        a_cmd = float(np.clip(a_cmd, self.trajectory_controller.max_deceleration,
                                    self.trajectory_controller.max_acceleration))

        # 横向：Pure Pursuit with speed-adaptive lookahead
        x = float(self.ego_info.loc['ego','x']); y = float(self.ego_info.loc['ego','y']); psi = float(self.ego_info.loc['ego','yaw'])
        pos = np.array([x, y], dtype=float)
        k_ld0 = float(getattr(self.trajectory_controller, 'k_ld0', 5.0))
        k_ld1 = float(getattr(self.trajectory_controller, 'k_ld1', 0.5))
        Ld = max(1.0, k_ld0 + k_ld1 * v)

        # 找到预瞄点（沿折线累计弧长≈Ld）
        acc_len = 0.0
        look = ref_traj_xy[-1]
        for i in range(len(ref_traj_xy)-1):
            seg = np.linalg.norm(ref_traj_xy[i+1] - ref_traj_xy[i])
            acc_len += seg
            if acc_len >= Ld:
                look = ref_traj_xy[i+1]
                break

        alpha = self._wrap_to_pi(math.atan2(look[1]-pos[1], look[0]-pos[0]) - psi)
        L = float(self.ego_info.loc['ego','length']) / 1.7
        steering = math.atan2(2.0 * L * math.sin(alpha), max(Ld, 1e-3))
        steering = float(np.clip(steering, -self.trajectory_controller.max_steering_angle,
                                            self.trajectory_controller.max_steering_angle))

        # 若 act 期望 rot（yaw_rate），做 δ→˙ψ
        rot = v / max(L, 1e-3) * math.tan(steering)

        return rot  # 或返回 steering，视你的上层接口而定
    def _find_nearest_trajectory_point(self, trajectory, ego_s_m: float, ego_d_m: float, w_d: float = 1.3):
        """按 |s差| + w_d*|d差| 选择最近点；s 为米制，d 为米."""
        if not trajectory:
            return 0
        best_i, best_cost = 0, float('inf')
        for i, (s_m, d, t) in enumerate(trajectory):
            cost = abs(s_m - ego_s_m) + w_d * abs(d - ego_d_m)
            if cost < best_cost:
                best_cost = cost
                best_i = i
        return best_i

    def _get_trajectory_lookahead_point(self, trajectory, ego_s, lookahead_distance=10.0):
        """获取前瞻点（用于更平滑的控制）"""
        if not trajectory:
            return None
        
        # 找到前瞻距离对应的轨迹点
        for i, (s, d, t) in enumerate(trajectory):
            if s - ego_s >= lookahead_distance:
                return i, (s, d, t)
        
        # 如果没有找到前瞻点，返回轨迹末端
        return len(trajectory) - 1, trajectory[-1]
    
    def step(self, act,vd, episode, egos):
        # print(action)
        self.step_count += 1
        # goal_lane = self._get_goal_lane()
        # print('goal_lane', goal_lane)
        # ego_lane = int(self.ego_info.loc['ego','lane'])
        # print('ego_lane', ego_lane)
        # state = self._get_features(self.observation)
        # obs = self._to_obs(state)
        # obs['is_first'] = False
        # obs['is_terminal'] = done
        reward, done, info = 0, False, None
        cost = 0
        if self.observation.test_info['end'] == 1:
            info = 'goal'
            done = True
        elif self.observation.test_info['end'] == 2:
            info = 'max_t'
            done = True
        elif self.observation.test_info['end'] == 3:
            info = 'collision'
            done = True
        elif self.observation.test_info['end'] == 4:
            info = 'map_out_range'
            done = True
        else:
            info = None
            done = False
        if self.use_spare_reward and episode >= self.sparse_episode:
            reward = self.get_reward_sparse(info=info, done=done)
        else:
            reward = self.get_reward_dense(info=info, done=done, egos = egos)
            
            reward_new = self.get_simple_reward(info=info)

            # print('act',act)

            cost = self.cost_function(self.observation, act,info=info)
            # print('cost2250',cost)
            # reward = reward - cost
        info_out = {'discount': np.array(0.0 if done else 1.0, dtype=np.float32)}
        return reward_new, cost,done, info

    
    def get_simple_reward(self,info):
        ego_x = float(self.ego_info.loc['ego', 'x'])
        ego_y = float(self.ego_info.loc['ego', 'y'])
        ego_s = self.ego_info.loc['ego', 'rel_pos_ind']
        ego_d = self.ego_info.loc['ego', 'dis_ind']
        ego_v = self.ego_info.loc['ego', 'v']
        ego_yaw = self.ego_info.loc['ego', 'yaw']
        ego_a = self.ego_info.loc['ego', 'a']
        ego_pos = np.array([self.ego_info.loc['ego', 'x'], self.ego_info.loc['ego', 'y']], dtype=float)
        ego_lane = int(self.ego_info.loc['ego', 'lane'])
        #道路中心线的x,y坐标以及弧长
        ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)
        goal_center = np.array([self.goal_center_x, self.goal_center_y])
                    # 找到最近参考点
        distances = np.linalg.norm(ref_center - ego_pos, axis=1)
        nearest_idx = np.argmin(distances)

        # 纵向弧长（m）
        ego_s_m = self._s_from_idx(station, int(nearest_idx))
        comfort_number =  check_comfort_number(self.old_ego_a,ego_a,self.old_ego_yaw,ego_yaw,dt=0.1) #计算违反舒适性的次数
        self.old_ego_a = ego_a
        self.old_ego_yaw = ego_yaw
        d_min, ttc_min = self._nearest_dist_and_ttc_xy(ego_x, ego_y, ego_v, ego_yaw)
        total_reward ,RCt,penalties  = simple_rewards(ref_center,station,comfort_number,goal_center,ego_s_m,ttc_min,ego_d,info,ego_v)
        # print('total_reward : {} ,RCt : {},penalties: {}',total_reward ,RCt,penalties)
        return  total_reward


    def get_reward_dense(self, info=None, done=0, egos=None):
        """
        论文即时奖励 R_i（式(16)-(18)）的实现，并保留你现有的终止加分/扣分逻辑。
        R_i = R_iE + R_iS
        R_iE = k1 * V + k2 * L_goal
        R_iS = k3 * min(d_safe - d_min, 0) + k4 * max(t_safe - TTC_min, 0)
        """
        # === 读取自车与目标信息（你已有的字段/写法） ===
        ego_v = float(self.ego_info.loc['ego', 'v'])
        ego_x = float(self.ego_info.loc['ego', 'x'])
        ego_y = float(self.ego_info.loc['ego', 'y'])
        ego_yaw = float(self.ego_info.loc['ego', 'yaw'])  # radians
        # 目标中心，已有赋值（reset 时计算）与归一化距离 start_dis 
        L_goal = -math.sqrt((ego_x - self.goal_center_x) ** 2 + (ego_y - self.goal_center_y) ** 2) / max(self.start_dis, 1e-6)

        # === 计算与邻车的 d_min、TTC_min（来自你在 get_features 里保存的 self.front/self.behind） ===
        d_min, ttc_min = self._nearest_dist_and_ttc_xy(ego_x, ego_y, ego_v, ego_yaw)
        if 0.03 < ttc_min < 0.53:
            r_ttc = 0.1 / ttc_min
        elif ttc_min >= 0.53:
            r_ttc = 0
        elif 0 < ttc_min <= 0.03:
            r_ttc = 0.2 / ttc_min
        else:
            r_ttc = -0.2 / ttc_min
        # print('ttc_min', ttc_min) 
        # === 论文即时奖励（式(16)-(18)） ===  
        # 可按需要暴露为成员变量；这里给一组稳健的初值
        k1, k2, k3, k4 = 0.05, 0.0, 0.0, 0.0
        d_safe, t_safe = 10.0, 2.5  # m, s
        # print(L_goal)
        RiE = k1 * ego_v + k2 * L_goal
        # print('rie', L_goal)
        # print('ttc min', ttc_min)
        # print('d min', d_min)
        RiS = k3 * min(d_min - d_safe, 0.0) + k4 * min(ttc_min - t_safe, 0.0)
        # print('ris', RiS)
        # print('r_ttc', r_ttc)
        Ri  = RiE + RiS - r_ttc
        r_merge = 0
        if self.s_merge is not None:
            ego_s = egos
            Win = 8.0
            in_window = abs(2 * ego_s - self.s_merge) <=Win
            # print(1969, in_window)
            # print(1964, self.huandao_flag)
            if in_window and self.huandao_flag:
                r_merge = -1 * (ego_v)
            if in_window and not self.huandao_flag:
                r_merge = 1 * (ego_v)
            # print('merge', r_merge)
        # === 终止奖励（论文 R_t；保留你原来的 info 终止加分/扣分以兼容现有训练脚本） ===  
        R_term = 0.0
        if info == 'goal':
            R_term = 100.0
        if info == 'collision':
            R_term = 0
        if info == 'map_out_range':
            R_term = 0
        if info == 'max_t':
            R_term = 30.0
        # R_term = 0.0
        # === 总即时奖励（本函数每步返回）===
        reward = float(Ri + R_term + r_merge)
        return reward


    def _nearest_dist_and_ttc_0(self, ego_x, ego_y, ego_v, ego_yaw, t_horizon: float = 8.0):
        """
        更准确的 d_min & TTC 计算（常速直线假设）：
        - d_min_now: 当前时刻自车与所有邻车的中心到中心最小欧氏距离（m）
        - ttc_min:   在 [0, t_horizon] 内，若最近接距离进入安全阈值（考虑车体尺寸），则返回该时刻，否则为 +inf

        做法：
        1) 相对位置 r = p_other - p_ego
        2) 相对速度 v = v_other - v_ego
        3) 最近接时刻 t* = clip( -<r,v> / ||v||^2, 0, t_horizon )
        4) 最近接距离 d* = || r + v t* ||
        5) 若 d* <= d_safe（由两车外接圆半径和 + 预留边界），则 TTC = t*；否则 = +inf
        """
        import numpy as np, math, pandas as pd

        # 汇总候选邻车
        cand = []
        if isinstance(self.front, pd.DataFrame) and not self.front.empty:
            cand.append(self.front)
        if isinstance(self.behind, pd.DataFrame) and not self.behind.empty:
            cand.append(self.behind)
        if not cand:
            return 1e9, 1e9

        neigh = pd.concat(cand, axis=0, ignore_index=True)

        # 自车速度向量
        dir_ego   = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=float)
        v_ego_vec = dir_ego * float(ego_v)

        # 自车外接圆半径（长宽对角线的一半；缺省 4.5x1.8）
        ego_L = float(getattr(self, "ego_info", {}).loc['ego', 'length']) if hasattr(self, "ego_info") and len(self.ego_info) else 4.5
        ego_W = float(getattr(self, "ego_info", {}).loc['ego', 'width'])  if hasattr(self, "ego_info") and len(self.ego_info) else 1.8
        R_ego = 0.5 * math.hypot(ego_L, ego_W)

        d_min_now = 1e9
        ttc_min   = 1e9
        eps       = 1e-9
        # 可选：为安全留一点边界（m）
        safety_margin = 0.2

        for _, row in neigh.iterrows():
            # 提取并做容错
            try:
                ox   = float(row.get('x', 0.0))
                oy   = float(row.get('y', 0.0))
                ov   = float(row.get('v', 0.0))
                oyaw = float(row.get('yaw', 0.0))
                oL   = float(row.get('length', 4.5))
                oW   = float(row.get('width', 1.8))
            except Exception:
                continue

            # 当前位置最近距离（中心到中心）
            r    = np.array([ox - ego_x, oy - ego_y], dtype=float)
            dist = float(np.linalg.norm(r))
            d_min_now = min(d_min_now, dist)

            # 相对速度与最近接时刻
            v_other_vec = np.array([math.cos(oyaw), math.sin(oyaw)], dtype=float) * ov
            v_rel   = v_other_vec - v_ego_vec
            v_rel2  = float(np.dot(v_rel, v_rel))

            if v_rel2 <= eps:
                # 相对静止：最近接就是现在
                t_star = 0.0
                d_star = dist
            else:
                t_star = - float(np.dot(r, v_rel)) / v_rel2
                # 只考虑未来且限制预测时域
                if t_star < 0.0:
                    t_star = 0.0
                elif t_star > t_horizon:
                    t_star = t_horizon
                d_star = float(np.linalg.norm(r + v_rel * t_star))

            # 车辆尺寸 → 外接圆半径和
            R_other = 0.5 * math.hypot(oL, oW)
            d_safe  = R_ego + R_other + safety_margin

            # 若未来最近接会进入危险距离，认为存在有限 TTC
            if d_star <= d_safe:
                ttc = t_star
            else:
                ttc = 10

            if ttc < ttc_min:
                ttc_min = ttc

        return d_min_now, ttc_min


   
    def reset(self, scenario_dict=None, lane_info=None):
        # print(scenario_dict)

        # 重置所有PID控制器
        self.trajectory_controller.longitudinal_pid.reset()
        self.velocity_pid.reset()
        scenario_num = scenario_dict['num']
        scenario_name = scenario_dict['name']
        self.scenario_type = scenario_dict['type']
        self.scenario_source_file = scenario_dict['source_file']
        self.scenario_output_path = scenario_dict['output_path']
        self.scenario_startPos = scenario_dict['task_info']['startPos']
        self.scenario_targetPos = scenario_dict['task_info']['targetPos']
        self.scenario_waypoints = scenario_dict['task_info']['waypoints']
        self.scenario_dt = scenario_dict['task_info']['dt']
        print(scenario_name)
        # 解析环岛半径与圆心
        # xodr_path = os.path.join(
        #     "/home/hmx/Hmx/onsite_dreamertdv/scenario/replay",
        #     scenario_name,
        #     f"{scenario_name}.xodr"
        # )
        # if os.path.exists(xodr_path):
        #     cx, cy, R = self.extract_roundabout_from_xodr(xodr_path)
        #     self.roundabout_center_x = cx
        #     self.roundabout_center_y = cy
        #     self.roundabout_radius = R
        #     print(f"[INFO] Roundabout parsed: center=({cx:.2f},{cy:.2f}), R={R:.2f} m")
        # else:
        #     self.roundabout_center_x = None
        #     self.roundabout_center_y = None
        #     self.roundabout_radius = None
        #     print(f"[WARN] XODR file not found: {xodr_path}")
        self.goal_center_x = (self.scenario_targetPos[0][0] + self.scenario_targetPos[1][0]) / 2
        self.goal_center_y = (self.scenario_targetPos[0][1] + self.scenario_targetPos[1][1]) / 2

        self.x_goal_min = min(self.scenario_targetPos[0][0],self.scenario_targetPos[1][0])
        self.x_goal_max = max(self.scenario_targetPos[0][0],self.scenario_targetPos[1][0])
        self.y_goal_min = min(self.scenario_targetPos[0][1],self.scenario_targetPos[1][1])
        self.y_goal_max = max(self.scenario_targetPos[0][1],self.scenario_targetPos[1][1])
        # print('goal_center', self.goal_center_x, self.goal_center_y)
        # print('x_goal', self.x_goal_min, self.x_goal_max, 'y_goal', self.y_goal_min, self.y_goal_max)
        self.start_dis = math.sqrt(math.pow(self.scenario_startPos[0]-self.goal_center_x,2) + math.pow(self.scenario_startPos[1]-self.goal_center_y,2))
        # print(1175, self.start_dis)
        self.v_max = 9

        #换道连续
        self.step_pre = 0
        self.action_pre = 0
        self.v_des_pre = 0
        self.v_des_steer_pre = 0
        self.v_des = 0
        self.target_lane_pre = 0
        self.action_now = 0
        self.front = None
        self.behind = None
        self.observation = None
        self.ego_info = pd.DataFrame()

        self.t_need = []
        self.rot_need = []
        self.v_need = []
        self.a_need = []

        self.test = []

        # 场景全解析，违规，不使用
        # reply = Reply()
        # reply.source_file["tess"] = scenario_source_file["tess"]
        # reply.source_file["xodr"] = scenario_source_file["xodr"]
        # reply.source_file["xosc"] = scenario_source_file["xosc"]
        # reply.source_file["json"] = scenario_source_file["json"]
        # parser = ReplayParser()
        # replay_info = parser.parse(reply)


        #道路拼接

        if len(self.scenario_waypoints) == 0:
            #道路信息
            lanes = lane_info.discretelanes
            # for i in lanes:
            #     print(i.lane_id, i.center_vertices[0,:], i.center_vertices[-1,:])
            #计算目标点与初始点的角度
            # angle_start, xiangxian_start = cal_angle([self.scenario_startPos[0],self.scenario_startPos[1]], [self.goal_center_x,self.goal_center_y])
            #需要的车道以及原id
            self.road_info_dict = {}#{id:[(,),...]}
            self.road_info_index = {}#{id:车道编号，初始点，末尾点,角度，k，b}
            self.road_info_pingjie = {}#{车道编号:初始点，末尾点，id}

            #查找自车初始车道和目标所在车道
            robot_goal = np.array([self.goal_center_x,self.goal_center_y])
            robot_ego = np.array((self.scenario_startPos[0],self.scenario_startPos[1]))
            dis_goal = 999999
            dis_ego = 999999
            id_goal = '-1.-1.-1.-1'
            id_ego = '-1.-1.-1.-1'

            for i in lanes:
                # print("id: ", i.lane_id, "start: ", i.center_vertices[0,:], "end: ", i.center_vertices[-1,:])
                refer_tree = KDTree(i.center_vertices)
                distance_goal, ind_goal = refer_tree.query(robot_goal)
                distance_ego, ind_ego = refer_tree.query(robot_ego)
                if distance_goal < dis_goal:
                    id_goal = i.lane_id
                    dis_goal = distance_goal
                if distance_ego < dis_ego:
                    id_ego = i.lane_id
                    dis_ego = distance_ego
            
            # print("id_goal: ", id_goal, "id_ego: ", id_ego)
            #取出goal和ego所在的道路信息
            temp_dict_goal = {}
            temp_id_goal = []
            temp_dict_ego = {}
            temp_id_ego = []
            for i in lanes:
                dict_test_goal = {}
                dict_test_ego = {}
                if int(i.lane_id.split('.')[0]) == int(id_goal.split('.')[0]) and int(i.lane_id.split('.')[1]) == int(id_goal.split('.')[1]):
                    dict_test_goal['center_vertices'] = copy(i.center_vertices)
                    dict_test_goal['node_start'] = np.array(i.center_vertices[0,:])
                    dict_test_goal['node_end'] = np.array(i.center_vertices[-1,:])
                    left = np.array(copy(i.left_vertices))
                    right = np.array(copy(i.right_vertices))
                    k = (left - right) ** 2
                    dict_test_goal['width'] = np.sqrt(k[:,0] + k[:,1])
                    # dict_test_goal['width'] = math.sqrt((i.left_vertices[0,0] - i.right_vertices[0,0])**2 + (i.left_vertices[0,1] - i.right_vertices[0,1])**2)
                    temp_dict_goal[i.lane_id] = dict_test_goal
                    temp_id_goal.append(i.lane_id)
                if int(i.lane_id.split('.')[0]) == int(id_ego.split('.')[0]) and int(i.lane_id.split('.')[1]) == int(id_ego.split('.')[1]):
                    dict_test_ego['center_vertices'] = copy(i.center_vertices)
                    dict_test_ego['node_start'] = np.array(i.center_vertices[0,:])
                    dict_test_ego['node_end'] = np.array(i.center_vertices[-1,:])
                    left = np.array(copy(i.left_vertices))
                    right = np.array(copy(i.right_vertices))
                    k = (left - right) ** 2
                    dict_test_ego['width'] = np.sqrt(k[:,0] + k[:,1])
                    # dict_test_ego['width'] = math.sqrt((i.left_vertices[0,0] - i.right_vertices[0,0])**2 + (i.left_vertices[0,1] - i.right_vertices[0,1])**2)
                    temp_dict_ego[i.lane_id] = dict_test_ego
                    temp_id_ego.append(i.lane_id)
            #取出所有车道与车道id（除开goal或者ego）
            temp_dict_g = {}
            temp_id_g = []
            temp_dict_e = {}
            temp_id_e = []
            for i in lanes:
                dict_test_g = {}
                dict_test_e = {}
                if i.lane_id not in temp_id_goal:
                    dict_test_g['center_vertices'] = copy(i.center_vertices)
                    dict_test_g['node_start'] = np.array(i.center_vertices[0,:])
                    dict_test_g['node_end'] = np.array(i.center_vertices[-1,:])
                    left = np.array(copy(i.left_vertices))
                    right = np.array(copy(i.right_vertices))
                    k = (left - right) ** 2
                    dict_test_g['width'] = np.sqrt(k[:,0] + k[:,1])
                    # dict_test_g['width'] = math.sqrt((i.left_vertices[0,0] - i.right_vertices[0,0])**2 + (i.left_vertices[0,1] - i.right_vertices[0,1])**2)
                    temp_dict_g[i.lane_id] = dict_test_g
                    temp_id_g.append(i.lane_id)
                if i.lane_id not in temp_id_ego:
                    dict_test_e['center_vertices'] = copy(i.center_vertices)
                    dict_test_e['node_start'] = np.array(i.center_vertices[0,:])
                    dict_test_e['node_end'] = np.array(i.center_vertices[-1,:])
                    left = np.array(copy(i.left_vertices))
                    right = np.array(copy(i.right_vertices))
                    k = (left - right) ** 2
                    dict_test_e['width'] = np.sqrt(k[:,0] + k[:,1])
                    # dict_test_e['width'] = math.sqrt((i.left_vertices[0,0] - i.right_vertices[0,0])**2 + (i.left_vertices[0,1] - i.right_vertices[0,1])**2)
                    temp_dict_e[i.lane_id] = dict_test_e
                    temp_id_e.append(i.lane_id)
            
            road_info_dict_temp = {}
            road_info_id_temp = {}
            
            #车道拼接
            error = 0.5
            m = 0
            l_t=[]
            #车道拼接(根据ego)
            for x_ego,y_ego in temp_dict_ego.items():
                while True:
                    k = {}
                    l_test = []
                    k['center_vertices'] = y_ego['center_vertices']
                    k['width'] = y_ego['width']
                    l_test.append(x_ego)
                    while True:
                        i = 0
                        k1 = 0
                        num = 0
                        # angle_temp = math.pi * 4
                        dis_temp = 999
                        id_temp = '-1.-1.-1.-1'
                        for x,y in temp_dict_e.items():
                            if abs(np.linalg.norm(k['center_vertices'][-1,:]-y['center_vertices'][0,:]))<=error:
                                # angle_, _ = cal_angle(y['center_vertices'][-1,:], [self.goal_center_x,self.goal_center_y])
                                dis_temp_temp = 9999999
                                for x_temp, y_temp in temp_dict_goal.items():
                                    dis = abs(np.linalg.norm(y['center_vertices'][-1,:]-y_temp['center_vertices'][0,:]))
                                    if dis_temp_temp > dis:
                                        dis_temp_temp = dis
                                dis = dis_temp_temp
                                # dis = abs(np.linalg.norm(y['center_vertices'][-1,:]-[self.goal_center_x,self.goal_center_y]))
                                # if abs(angle_-angle_start) < angle_temp:
                                #     angle_temp = abs(angle_-angle_start)
                                #     id_temp = x
                                if dis_temp > dis:
                                    dis_temp = dis
                                    if id_temp != id_goal:
                                        id_temp = x
                                
                                if x == id_goal:
                                    id_temp = id_goal
                                num += 1

                        for x,y in temp_dict_e.items():
                            if abs(np.linalg.norm(k['center_vertices'][0,:]-y['center_vertices'][-1,:]))<=error:
                                if x in l_test:
                                    break
                                k['center_vertices'] = np.vstack((y['center_vertices'],k['center_vertices']))
                                # k['width'] = np.vstack((y['width'],k['width']))
                                k['width'] = np.concatenate((y['width'],k['width']))
                                l_test.insert(0,x)
                                # i += 1
                                k1 += 1
                                # if k['width'] > y['width']:
                                #     k['width'] = y['width']
                            if abs(np.linalg.norm(k['center_vertices'][-1,:]-y['center_vertices'][0,:]))<=error:
                                if x in l_test:
                                    break
                                if num <= 1:
                                    k['center_vertices'] = np.vstack((k['center_vertices'],y['center_vertices']))
                                    # k['width'] = np.vstack((k['width'],y['width']))
                                    k['width'] = np.concatenate((k['width'],y['width']))
                                    l_test.append(x)
                                    i += 1
                                    # if k['width'] > y['width']:
                                    #     k['width'] = y['width']
                                else:
                                    if x == id_temp:
                                        k['center_vertices'] = np.vstack((k['center_vertices'],y['center_vertices']))
                                        # k['width'] = np.vstack((k['width'],y['width']))
                                        k['width'] = np.concatenate((k['width'],y['width']))
                                        l_test.append(x)
                                        i += 1
                                        # if k['width'] > y['width']:
                                        #     k['width'] = y['width']
                            if i == 1:
                                break
                        if i == 0 and k1 == 0:
                            break
                        # if x in temp_id_goal:
                        #     break
                    if l_test not in l_t:
                        road_info_dict_temp[str(m)] = k
                        road_info_id_temp[str(m)] = l_test
                        m += 1
                        l_t.append(l_test)
                    break
            
            #车道拼接(根据goal)
            for x_goal,y_goal in temp_dict_goal.items():
                while True:
                    k = {}
                    l_test = []
                    k['center_vertices'] = y_goal['center_vertices']
                    k['width'] = y_goal['width']
                    l_test.append(x_goal)
                    while True:
                        i = 0
                        k1 = 0
                        num = 0
                        # angle_temp = math.pi * 4
                        dis_temp = 99999
                        id_temp = '-1.-1.-1.-1'
                        for x,y in temp_dict_g.items():
                            if abs(np.linalg.norm(k['center_vertices'][0,:]-y['center_vertices'][-1,:]))<=error:
                                # angle_, _ = cal_angle(y['center_vertices'][-1,:], [self.goal_center_x,self.goal_center_y])
                                # if abs(angle_-angle_start) < angle_temp:
                                #     angle_temp = abs(angle_-angle_start)
                                #     id_temp = x
                                # dis = abs(np.linalg.norm(y['center_vertices'][0,:]-[self.goal_center_x,self.goal_center_y]))
                                dis_temp_temp = 9999999
                                for x_temp, y_temp in temp_dict_ego.items():
                                    dis = abs(np.linalg.norm(y['center_vertices'][0,:]-y_temp['center_vertices'][-1,:]))
                                    if dis_temp_temp > dis:
                                        dis_temp_temp = dis
                                dis = dis_temp_temp
                                
                                if dis_temp > dis:
                                    dis_temp = dis
                                    if id_temp != id_ego:
                                        id_temp = x
                                
                                if x == id_ego:
                                    id_temp = id_ego
                                num += 1
                        
                        for x,y in temp_dict_g.items():
                            if abs(np.linalg.norm(k['center_vertices'][0,:]-y['center_vertices'][-1,:]))<=error:
                                if x in l_test:
                                    break
                                if num <= 1:
                                    k['center_vertices'] = np.vstack((y['center_vertices'],k['center_vertices']))
                                    # k['width'] = np.vstack((y['width'],k['width']))
                                    k['width'] = np.concatenate((y['width'],k['width']))
                                    l_test.insert(0,x)
                                    i += 1
                                    # if k['width'] > y['width']:
                                    #     k['width'] = y['width']
                                else:
                                    if x == id_temp:
                                        k['center_vertices'] = np.vstack((y['center_vertices'],k['center_vertices']))
                                        # k['width'] = np.vstack((y['width'],k['width']))
                                        k['width'] = np.concatenate((y['width'],k['width']))
                                        l_test.insert(0,x)
                                        i += 1
                                        # if k['width'] > y['width']:
                                        #     k['width'] = y['width']
                            if abs(np.linalg.norm(k['center_vertices'][-1,:]-y['center_vertices'][0,:]))<=error:
                                if x in l_test:
                                    break
                                k['center_vertices'] = np.vstack((k['center_vertices'],y['center_vertices']))
                                # k['width'] = np.vstack((k['width'],y['width']))
                                k['width'] = np.concatenate((k['width'],y['width']))
                                l_test.append(x)
                                # i += 1
                                k1 += 1
                                # if k['width'] > y['width']:
                                #     k['width'] = y['width']
                            if i == 1:
                                break
                        if i == 0 and k1 == 0:
                            break
                        # if x in temp_id_ego:
                        #     break
                    if l_test not in l_t:
                        road_info_dict_temp[str(m)] = k
                        road_info_id_temp[str(m)] = l_test
                        m += 1
                        l_t.append(l_test)
                    break
            
            #拼接好的道路后处理（去除包含车道)
            del_num = []
            for i in range(len(road_info_id_temp)):
                j = i+1
                while j < len(road_info_id_temp):
                    if self.check_containment(road_info_id_temp[str(i)],road_info_id_temp[str(j)]):
                        print(road_info_id_temp[str(j)])
                        if len(road_info_id_temp[str(i)]) < len(road_info_id_temp[str(j)]):
                            del_num.append(str(i))
                        else:
                            del_num.append(str(j))
                    j += 1
            
            del_num = list(set(del_num))

            for key in del_num:
                if key in road_info_dict_temp:
                    del road_info_dict_temp[key]
                if key in road_info_id_temp:
                    del road_info_id_temp[key]
            
            #复制
            road_info_dict_temp_temp_temp = {}
            road_info_id_temp_temp_temp = {}
            for x,y in road_info_id_temp.items():
                road_info_id_temp_temp_temp[str(x)] = y
            for x,y in road_info_dict_temp.items():
                road_info_dict_temp_temp_temp[str(x)] = y
            #拼接好的道路后处理（留下包含ego和goal的车道)
            del_num = []
            for x,y in road_info_id_temp.items():
                if not (self.check_common_elements(y,temp_id_goal) and self.check_common_elements(y,temp_id_ego)):
                    del_num.append(x)
            
            for key in del_num:
                if key in road_info_dict_temp:
                    del road_info_dict_temp[key]
                if key in road_info_id_temp:
                    del road_info_id_temp[key]
            
            road_info_dict_temp_temp = {}
            road_info_id_temp_temp = {}
            i = 0
            for x,y in road_info_id_temp.items():
                road_info_id_temp_temp[str(i)] = y
                i += 1
            i = 0
            for x,y in road_info_dict_temp.items():
                road_info_dict_temp_temp[str(i)] = y
                i += 1
            
            #不同时包含goal和ego
            if len(road_info_dict_temp_temp) == 0:
                road_info_dict_temp_temp = road_info_dict_temp_temp_temp
                road_info_id_temp_temp = road_info_id_temp_temp_temp

            road_info_dict_temp = {}
            road_info_id_temp = {}
            i = 0
            for x,y in road_info_id_temp_temp.items():
                road_info_id_temp[str(i)] = y
                i += 1
            i = 0
            
            for x,y in road_info_dict_temp_temp.items():
                road_info_dict_temp[str(i)] = y
                i += 1
            
            # 仅保留包含自车 reset 车道的拼接轨迹；若有多条，选与自车初始位置最近的一条ego one
            # ego_candidates = [k for k, ids in road_info_id_temp.items() if id_ego in ids]
            # if len(ego_candidates) >= 1:
            #     if len(ego_candidates) > 1:
            #         # 选距离自车初始位置最近的一条
            #         best_k = min(
            #             ego_candidates,
            #             key=lambda k: KDTree(road_info_dict_temp[k]['center_vertices']).query(robot_ego)[0]
            #         )
            #     else:
            #         best_k = ego_candidates[0]
            #     road_info_dict_temp = {best_k: road_info_dict_temp[best_k]}
            #     road_info_id_temp   = {best_k: road_info_id_temp[best_k]}
            #车道id赋值，按照自车位置，在自车左方的最大
            for i in range(len(road_info_dict_temp)):
                for j in range(0, len(road_info_dict_temp)-i-1):
                    a = road_info_dict_temp[str(j)]['center_vertices'][1,:] - road_info_dict_temp[str(j)]['center_vertices'][0,:]
                    b = road_info_dict_temp[str(j+1)]['center_vertices'][0,:] - road_info_dict_temp[str(j)]['center_vertices'][0,:]
                    if b[0] == 0 and b[1] == 0:
                        a = road_info_dict_temp[str(j)]['center_vertices'][-1,:] - road_info_dict_temp[str(j)]['center_vertices'][-2,:]
                        b = road_info_dict_temp[str(j+1)]['center_vertices'][-2,:] - road_info_dict_temp[str(j)]['center_vertices'][-2,:]
                    if a[0] * b[1] - a[1] * b[0] < 0:
                        road_info_dict_temp[str(j)], road_info_dict_temp[str(j+1)] = road_info_dict_temp[str(j+1)], road_info_dict_temp[str(j)]
                        road_info_id_temp[str(j)], road_info_id_temp[str(j+1)] = road_info_id_temp[str(j+1)], road_info_id_temp[str(j)]

            #得到self.road_info_dict和self.road_info_index
            # k = []
            # for x,y in road_info_id_temp.items():
            #     for i in y:
            #         if i not in k:
            #             k.append(i)
            # for i in lanes:
            #     dict_test = {}
            #     if i.lane_id in k:
            #         angle_lane, xiangxian_lane = cal_angle(i.center_vertices[0,:],i.center_vertices[-1,:])
            #         dict_test['center_vertices'] = copy(i.center_vertices)
            #         dict_test['node_start'] = np.array(i.center_vertices[0,:])
            #         dict_test['node_end'] = np.array(i.center_vertices[-1,:])
            #         node_start_angle, _ = cal_angle(i.center_vertices[0,:],i.center_vertices[1,:])
            #         node_end_angle, _ = cal_angle(i.center_vertices[-2,:],i.center_vertices[-1,:])
            #         dict_test['node_start_angle'] = node_start_angle
            #         dict_test['node_end_angle'] = node_end_angle
            #         dict_test['lane_angle'] = angle_lane
            #         dict_test['lane_xiangxian'] = xiangxian_lane
            #         dict_test['k'], dict_test['b'], dict_test['c'] = cal_k_b_c(i.center_vertices[0,:],i.center_vertices[-1,:])
            #         dict_test['width'] = math.sqrt((i.left_vertices[0,0] - i.right_vertices[0,0])**2 + (i.left_vertices[0,1] - i.right_vertices[0,1])**2)
            #         self.road_info_dict[i.lane_id] = dict_test
            
            # for x,y in self.road_info_dict.items():
            #     dict_test = {}
            #     dict_test['node_start'] = copy(y['node_start'])
            #     dict_test['node_end'] = copy(y['node_end'])
            #     dict_test['lane_angle'] = copy(y['lane_angle'])
            #     dict_test['lane_xiangxian'] = copy(y['lane_xiangxian'])
            #     dict_test['k'] = copy(y['k'])
            #     dict_test['b'] = copy(y['b'])
            #     dict_test['c'] = copy(y['c'])
            #     dict_test['node_start_angle'] = copy(y['node_start_angle'])
            #     dict_test['node_end_angle'] = copy(y['node_end_angle'])
            #     lane_id_temp = []
            #     dict_test['lane_id'] = [str(-1)]
            #     dict_test['width'] = copy(y['width'])
            #     for i, j in road_info_id_temp.items():
            #         if x in j:
            #             lane_id_temp.append(i)
            #     if len(lane_id_temp) != 0:
            #         dict_test['lane_id'] = lane_id_temp
            #     self.road_info_index[str(x)] = dict_test


            #得到self.road_info_pingjie
            for x,y in road_info_dict_temp.items():
                dict_test = {}
                dict_test['node_start'] = np.array(y['center_vertices'][0,:])
                dict_test['node_end'] = np.array(y['center_vertices'][-1,:])
                # dict_test['width'] = copy(y['width'])
                for i,j in road_info_id_temp.items():
                    if i == x:
                        dict_test['id'] = j
                self.road_info_pingjie[x] = dict_test
            
            #路点稀疏化
            # 计算自车整体行驶方向向量
            ego_start = np.array(self.scenario_startPos)
            goal_center = np.array([self.goal_center_x, self.goal_center_y])
            ego_dir = goal_center - ego_start
            ego_dir_unit = ego_dir / (np.linalg.norm(ego_dir) + 1e-8)

            # 创建一个新的字典保存正向车道
            filtered_road_info = {}


            for key, val in road_info_dict_temp.items():
                # print(len(road_info_dict_temp))
                vertices = val['center_vertices']
                # print(len(vertices))
                if len(vertices) <= 2:
                    continue
                lane_dir = vertices[-1] - vertices[0]
                lane_dir_unit = lane_dir / (np.linalg.norm(lane_dir) + 1e-8)
                cos_angle = np.dot(ego_dir_unit, lane_dir_unit)
                # print(cos_angle)
                
                if cos_angle > 0:  # 如果cos夹角大于0说明大致同方向
                    filtered_road_info[key] = val
            if len(road_info_dict_temp) > 2:
                self.road_info_dict_pingjie = filtered_road_info
            else:
                self.road_info_dict_pingjie = road_info_dict_temp
            
            # self.road_info_dict_pingjie = road_info_dict_temp
            # for x, y in self.road_info_dict.items():
            #     self.road_info_dict[x]['center_vertices'] = np.vstack((np.vstack((y['node_start'], y['center_vertices'][::4])),y['node_end']))
            for x, y in self.road_info_dict_pingjie.items():
                self.road_info_dict_pingjie[x]['center_vertices'] = y['center_vertices'][::4]
                self.road_info_dict_pingjie[x]['width'] = y['width'][::4]
            
            #计算phi_road, curvature, station
            for x, y in self.road_info_dict_pingjie.items():
                self.road_info_dict_pingjie[x]['phi_road'], self.road_info_dict_pingjie[x]['curvature'], self.road_info_dict_pingjie[x]['station'] = self.refer_cacul(y['center_vertices'][:,0],y['center_vertices'][:,1])
        else:
            self.road_info_dict_pingjie = {}
            dict_test_list = []
            for x,y in self.scenario_waypoints.items():
                dict_test_list.append(y)
            dict_test_list = np.array(dict_test_list)
            dict_test = {}
            dict_test['center_vertices'] = dict_test_list
            # print(dict_test_list)
            self.road_info_dict_pingjie[str(0)] = dict_test
            for x, y in self.road_info_dict_pingjie.items():
                self.road_info_dict_pingjie[x]['phi_road'], self.road_info_dict_pingjie[x]['curvature'], self.road_info_dict_pingjie[x]['station'] = self.refer_cacul(y['center_vertices'][:,0],y['center_vertices'][:,1])
                self.road_info_dict_pingjie[x]['width'] = np.array([3.5]*len(self.road_info_dict_pingjie[x]['center_vertices']))
        
        ludian_len = 2
        for road in self.road_info_dict_pingjie.values():
            if 'station' in road and len(road['station']) >= 2:
                ludian_len = road['station'][1] - road['station'][0]
                break
        # if len(self.road_info_dict_pingjie) != 0:
        #     if len(self.road_info_dict_pingjie[str(0)]['station']) >=2:
        #         ludian_len = self.road_info_dict_pingjie[str(0)]['station'][1] - self.road_info_dict_pingjie[str(0)]['station'][0]
        self.guikong = kongzhi(self.road_info_dict_pingjie,self.scenario_dt,ludian_len)
        self.jiaohe_yun = jiaohe_yundong(self.scenario_dt)
        self.jiaohe_dong_1 = jiaohe_dongli_1(self.scenario_dt)
        self.jiaohe_dong_2 = jiaohe_dongli_2(self.scenario_dt)
        self.jiaohe_dong_3 = jiaohe_dongli_3(self.scenario_dt)
        self.jiaohe_dong_8 = jiaohe_dongli_8(self.scenario_dt)
        self.jiaohe_dong_8_new = jiaohe_dongli_8_new(self.scenario_dt)
        global road_info_dict_pingjie1
        road_info_dict_pingjie1 = self.road_info_dict_pingjie
        # print('road_info_dict_pingjie1',road_info_dict_pingjie1)

        self.road_info_dict_pingjie1 = self.road_info_dict_pingjie
        
        # 预计算每条车道的左右可行驶限幅
        self._precompute_lateral_limits(margin=0.5, cap=5.0)
        self.step_count = 0
        self._lane_change_lock_until = 0

    def diff(self,val):
        differences = val[1:] - val[:-1]
        return differences


    def sign(self,para):
        if para > 0:
            return 1
        elif para < 0:
            return -1
        else:
            return 0

    def refer_cacul(self, x, y):
        x = np.array(x)
        y = np.array(y)
        dx = self.diff(x)
        dy = self.diff(y)

        dx_pre = deepcopy(dx)
        dx_after = deepcopy(dx)
        dx_pre = np.insert(dx_pre, 0, dx[0])
        dx_after = np.append(dx_after,dx[-1])
        dy_pre = deepcopy(dy)
        dy_after = deepcopy(dy)
        dy_pre = np.insert(dy_pre,0, dy[0])
        dy_after = np.append(dy_after, dy[-1])

        s_r = []
        s_r.append(0.0)
        dx_final = (dx_pre + dx_after) / 2.0
        dy_final = (dy_pre + dy_after) / 2.0
        ds = np.sqrt(dx_final**2 + dy_final**2)
        h_r = np.arctan2(dy_final, dx_final)
        for ii in range(len(dx_pre) - 1):
            s_r.append(s_r[ii] + math.sqrt(math.pow(dx[ii], 2) + math.pow(dy[ii], 2)))
        
        for ii in range(1,len(h_r)):
            if abs(h_r[ii] - h_r[ii - 1]) > math.pi:
                h_r[ii] = h_r[ii] - 2 * math.pi * self.sign(h_r[ii] - h_r[ii - 1])

        dh = self.diff(h_r)
        dh_p = deepcopy(dh)
        dh_a = deepcopy(dh)
        dh_p = np.insert(dh_p,0,dh[0])
        dh_a = np.append(dh_a, dh[-1])
        dh_f = (dh_p + dh_a) / 2.0
        k_r = np.sin(dh_f) / ds
        
        return h_r, k_r, s_r
    
    #检查两个数组中是否有相同元素
    def check_common_elements(self, arr1, arr2):
        set1 = set(arr1)
        set2 = set(arr2)
        common_elements = set1.intersection(set2)
        if common_elements:
            return True
        else:
            return False

    #检查两个数组中是否有包含关系
    def check_containment(self, arr1, arr2):
        set1 = set(arr1)
        set2 = set(arr2)
        if set1.issubset(set2) or set2.issubset(set1):
            return True
        else:
            return False
        
    def limit(self,para,low,up):
        if para >= up:
            return up
        elif para <= low:
            return low
        else:
            return para

    def _nearest_lane_id_and_idx(self, pos_xy):
        """
        返回与 pos_xy 最近的车道 (lane_id, dist, idx_on_that_lane)。
        dist 为欧氏距离（米），无符号；横向符号需要另算。
        """
        import numpy as np
        best = (None, float('inf'), 0)
        px, py = float(pos_xy[0]), float(pos_xy[1])
        for k, v in self.road_info_dict_pingjie.items():
            center = v["center_vertices"]
            # 线性扫描足够快；也可换 KDTree
            dists = ((center[:,0]-px)**2 + (center[:,1]-py)**2)**0.5
            idx = int(np.argmin(dists))
            d  = float(dists[idx])
            if d < best[1]:
                best = (int(k), d, idx)
        return best
    
    def _precompute_lateral_limits(self, margin=0.5, cap=5.0):
        """
        在 reset 之后，基于 self.road_info_dict_pingjie 预计算每条路径每个采样点的左右可行驶偏移上限。
        结果保存在：
            self.lateral_limits = {
                lane_id: {"left_max": np.ndarray, "right_max": np.ndarray}
            }
        约定：左为正、右为负；left_max/right_max 全为非负数（单位米）。
        """
        import numpy as np

        self.lateral_limits = {}

        # 收集各车道数组（提升访问效率）
        lanes = {}
        for k, v in self.road_info_dict_pingjie.items():
            lanes[int(k)] = {
                "center":  np.asarray(v["center_vertices"], dtype=float),
                "phi":     np.asarray(v["phi_road"], dtype=float),
                "station": np.asarray(v["station"], dtype=float),
                "width":   np.asarray(v["width"], dtype=float)
            }

        for e, E in lanes.items():
            center_e, phi_e, station_e, width_e = E["center"], E["phi"], E["station"], E["width"]
            N = len(center_e)
            if N == 0:
                continue

            half_e_arr = (width_e/2.0) if len(width_e) else np.full(N, 1.75)
            left_max  = half_e_arr.copy()
            right_max = half_e_arr.copy()

            for i in range(max(0, N-1)):  # 末点镜像上一点
                P_e   = center_e[i]
                phi0  = float(phi_e[i])
                n_hat = np.array([-np.sin(phi0), np.cos(phi0)], dtype=float)
                s_e   = float(station_e[i])
                half_e= float(half_e_arr[i])

                for k, K in lanes.items():
                    if k == e:
                        continue
                    center_k, station_k, width_k = K["center"], K["station"], K["width"]
                    # s 对齐：将 s_e 投到车道 k
                    idx_k = int(np.clip(self._idx_from_s(station_k, s_e), 0, len(center_k)-1))
                    P_k   = center_k[idx_k]
                    half_k= float((width_k[idx_k]/2.0) if len(width_k) else 1.75)

                    # 左为正的横向距离
                    d_k = float(np.dot(P_k - P_e, n_hat))
                    # 相邻判定（中心线横距足够近）
                    if abs(d_k) <= (half_e + half_k + margin):
                        # ——同向判定：在各自中心线“就近点”处比较切向——
                        # 自车道的就近索引就是 i；目标车道 K 的就近索引是 idx_k（上面已算）
                        if self._same_direction(e, i, k, idx_k, cos_thresh=0.866):
                            # 同向 → 允许放宽
                            if d_k > 0.0:  # 左邻道
                                left_max[i] = max(left_max[i],  min(d_k + half_k, cap))
                            else:          # 右邻道
                                right_max[i]= max(right_max[i], min(abs(d_k) + half_k, cap))
                        else:
                            # 逆向 → 不放宽（保持默认 half_e）
                            pass

            # 末点镜像
            if N >= 2:
                left_max[-1]  = left_max[-2]
                right_max[-1] = right_max[-2]

            self.lateral_limits[e] = {"left_max": left_max, "right_max": right_max}
    
    def _signed_d_to_lane(self, lane_id, pos_xy):
        """计算 pos_xy 相对于 lane_id 的带符号横向距离（左为正）。"""
        import numpy as np
        center, phi, _, station = self._get_lane_arrays(int(lane_id))
        # 用最近中心点的法向近似
        lane_id_cand, _, idx = self._nearest_lane_id_and_idx(pos_xy) if lane_id is None else (lane_id, 0.0, 0)
        idx = int(np.clip(idx, 0, len(center)-1))
        n_hat = np.array([-np.sin(float(phi[idx])), np.cos(float(phi[idx]))], dtype=float)
        return float(np.dot(np.asarray(pos_xy, dtype=float) - center[idx], n_hat)), idx
    
    def _update_ego_lane_with_hysteresis(self):
        """
        每步调用：根据自车位置更新 ego_lane（带滞回，防抖），并刷新 rel_pos_ind。
        不依赖 _s_from_pos；仅使用最近中心点索引。
        """
        import numpy as np

        # 当前位置
        if 'x' in self.ego_info.loc['ego'] and 'y' in self.ego_info.loc['ego']:
            pos_xy = np.array([float(self.ego_info.loc['ego','x']), float(self.ego_info.loc['ego','y'])], dtype=float)
        else:
            pos_xy = np.asarray(self.ego_pos_xy, dtype=float)

        # 候选最近车道（基于所有车道的最近中心点）
        lane_cand, dist_cand, idx_cand = self._nearest_lane_id_and_idx(pos_xy)

        # 取当前车道
        prev_lane = int(self.ego_info.loc['ego','lane'])
        prev_width = self.road_info_dict_pingjie[str(prev_lane)]["width"]
        prev_idx   = int(self.ego_info.loc['ego','rel_pos_ind'])
        half_prev  = float(prev_width[min(prev_idx, len(prev_width)-1)]/2.0) if len(prev_width) else 1.75

        # 相对“当前车道”的带符号横距（左正右负），基于最近索引
        d_prev, _ = self._signed_d_to_lane(prev_lane, pos_xy)

        # 滞回阈值
        enter_th = 0.65 * half_prev   # 进入新车道阈值（偏严）
        # exit_th  = 0.45 * half_prev # 如需回切阈值可启用

        # ——可选锁存：基于步数（如果你已在 reset/step 里维护 step_count 与 scenario_dt，见下文“第4步”）——
        use_lock = hasattr(self, "step_count") and hasattr(self, "scenario_dt")
        # 基于“当前车道”的最近索引（用于同向判断的 i）
        _, idx_prev_local = self._signed_d_to_lane(prev_lane, pos_xy)

        # 基于“候选车道”的最近索引（用我们在 _nearest_lane_id_and_idx() 返回的 idx_cand）
        idx_cand_local = int(idx_cand)

        def _allow_switch():
            # 只有当相对于当前车道“越过进入阈值” 且 与候选车道“同向”时，才允许切换
            if abs(d_prev) <= enter_th:
                return False            
            return self._same_direction(prev_lane, idx_prev_local, lane_cand, idx_cand_local, cos_thresh=0.866)

        if use_lock:
            if getattr(self, "_lane_change_lock_until", 0) > self.step_count:
                ego_lane_new = prev_lane
            else:
                if _allow_switch():
                    ego_lane_new = lane_cand
                    lock_steps = int(0.5 / self.scenario_dt)
                    self._lane_change_lock_until = self.step_count + lock_steps
                else:
                    ego_lane_new = prev_lane
        else:
            ego_lane_new = lane_cand if _allow_switch() else prev_lane


        # ——用“新车道的最近点索引”刷新 rel_pos_ind（不需要 _s_from_pos）——
        center_new = self.road_info_dict_pingjie[str(int(ego_lane_new))]["center_vertices"]
        dists_new  = ((center_new[:,0]-pos_xy[0])**2 + (center_new[:,1]-pos_xy[1])**2)**0.5
        new_idx    = int(np.argmin(dists_new))

        # 写回
        self.ego_info.loc['ego','lane'] = int(ego_lane_new)
        self.ego_info.loc['ego','rel_pos_ind'] = int(new_idx)

    def _same_direction(self, lane_a: int, idx_a: int, lane_b: int, idx_b: int, cos_thresh: float = 0.866):
        """
        判断在各自中心线上就近采样点处，两条车道是否“同向”。
        使用切向 t̂ = [cos(phi), sin(phi)]，两向量点积 > cos_thresh 即认为同向。
        默认 cos_thresh=0.866 ≈ cos(30°)，可按需要放宽/收紧（例如 0.707=45°）。
        """
        phi_a = float(self.road_info_dict_pingjie[str(int(lane_a))]["phi_road"][int(np.clip(idx_a, 0, len(self.road_info_dict_pingjie[str(int(lane_a))]["phi_road"])-1))])
        phi_b = float(self.road_info_dict_pingjie[str(int(lane_b))]["phi_road"][int(np.clip(idx_b, 0, len(self.road_info_dict_pingjie[str(int(lane_b))]["phi_road"])-1))])
        ta = np.array([np.cos(phi_a), np.sin(phi_a)], dtype=float)
        tb = np.array([np.cos(phi_b), np.sin(phi_b)], dtype=float)
        return float(np.dot(ta, tb)) >= float(cos_thresh)

    def _same_direction_yaw(self, lane_id: int, idx: int, ego_yaw: float, cos_thresh: float = 0.5):
        """
        判断某车道在 idx 处的切向与自车朝向是否“同向”：
        cos(ego_yaw - phi[idx]) >= cos_thresh 即视为同向。
        默认 cos_thresh=0.5（≈60°内），可按需调严到 0.866（≈30°）。
        """
        import numpy as np, math
        lane_id = int(lane_id)
        phi_arr = self.road_info_dict_pingjie[str(lane_id)]["phi_road"]
        j = int(np.clip(idx, 0, len(phi_arr)-1))
        return math.cos(float(ego_yaw) - float(phi_arr[j])) >= float(cos_thresh)


    def _ensure_forward_lane_and_index(self, cos_thresh: float = 0.5):
        """
        以“当前位置与朝向”为基准，确保自车绑定到最近的**同向**车道。
        若最近车道逆向，则搜索所有车道，选择：
            - 与自车朝向同向
            - 最近中心点距离最小
        找到就写回 ego_info.lane / rel_pos_ind；找不到就维持原值。
        """
        import numpy as np, math
        # 当前位置与朝向
        px = float(self.ego_info.loc['ego','x']); py = float(self.ego_info.loc['ego','y'])
        yaw = float(self.ego_info.loc['ego','yaw'])
        pos = np.array([px, py], dtype=float)

        # 先基于距离取候选
        lane_cand, dist_cand, idx_cand = self._nearest_lane_id_and_idx(pos)
        # 若同向，直接用
        if self._same_direction_yaw(lane_cand, idx_cand, yaw, cos_thresh=cos_thresh):
            if int(self.ego_info.loc['ego','lane']) != int(lane_cand):
                self.ego_info.loc['ego','lane'] = int(lane_cand)
            self.ego_info.loc['ego','rel_pos_ind'] = int(idx_cand)
            return

        # 否则在所有车道里找“同向且最近”的
        best = None
        for k, v in self.road_info_dict_pingjie.items():
            center = v["center_vertices"]; phi = v["phi_road"]
            dists = np.linalg.norm(center - pos, axis=1)
            j = int(np.argmin(dists))
            if len(phi) == 0: 
                continue
            if math.cos(yaw - float(phi[j])) >= cos_thresh:
                d = float(dists[j])
                if (best is None) or (d < best[1]):
                    best = (int(k), d, j)

        if best is not None:
            self.ego_info.loc['ego','lane'] = int(best[0])
            self.ego_info.loc['ego','rel_pos_ind'] = int(best[2])
        # 如果完全找不到同向车道，就维持原值；后续仍会强制 s 前行，避免“倒着走”

#===============================================================================================
    def cost_function(self, trajectory, actions,info):
        c_term = 0
        _, _,_,ego_all_trajectory = self.plan_trajectory_polynomial(actions)
        # print('自车预测轨迹:',ego_all_trajectory)
        # print(len(ego_all_trajectory['orientation_list']))
        ego_one_traj = ego_from_self_info(self.ego_info)
        # print('ziche info',ego_one_traj)
        obs_only_trajectory = build_obs_around_ego(        # 不包含自车
            trajectory,
            ego_pos=ego_one_traj['pos'],
            side_types=('vehicle','bicycle','pedestrian'),
            max_dist=30.0,
            exclude_ids=('ego',)  # 以防 observation 里恰好也有 ego
                    )
        # print('obs_only_trajectory:',obs_only_trajectory)     

        #到这里得到的只是当前位置的信息不包含预测信息。

        planning_horizon = 1.5
        # print('planning_horizon:',planning_horizon)
        # print('预测前：',obs_only_trajectory)
        # todo 需要对其obj轨迹所需的list类型的各个部分的key包含pos，速度，航向角，obs的类型，车，行人还是自行车。
        obj_predictions = predictor_future_only(
            obs_only_trajectory,
            planning_horizon,
            dt=self.scenario_dt,
            vmax_by_type={"bicycle": 8.0, "vehicle": 30.0, "pedestrian": 5})
             # 根据旁车当前x,y,raw,预测未来时间的轨迹。
        # print('预测后',obj_predictions)
        risk_cost = self.total_costs(ego_all_trajectory, obj_predictions)
        if info == 'goal':
            c_term = 0
        if info == 'collision':
            c_term = 1
        if info == 'map_out_range':
            c_term = 1
        if info == 'max_t':
            c_term = 0.1

        total_cost = 0.2*risk_cost + c_term
        return total_cost


    def total_costs(self, ego_traj, obj_predictions):
        return self.risk_evaluator.total_costs(ego_traj, obj_predictions)

    def get_bayesian_costs(self, ego_risk_max, obj_risk_max, boundary_harm=0):



    # # 打印相关变量的详细信息
    #     print(f"类型: {type(ego_risk_max)}")
    #     print("\n===== 调试信息：ego_risk_max =====")
    #     print(f"是否为字典: {isinstance(ego_risk_max, dict)}")
    #     print(f"长度: {len(ego_risk_max)}")
    #     print(f"键: {list(ego_risk_max.keys())}")
    #     print(f"值: {list(ego_risk_max.values())}")
    #     print(f"值的类型: {[type(v) for v in ego_risk_max.values()]}")
    
    #     print("\n===== 调试信息：obj_risk_max =====")
    #     print(f"类型: {type(obj_risk_max)}")
    #     print(f"是否为字典: {isinstance(obj_risk_max, dict)}")
    #     print(f"长度: {len(obj_risk_max)}")
    #     print(f"键: {list(obj_risk_max.keys())}")
    #     print(f"值: {list(obj_risk_max.values())}")
    #     print(f"值的类型: {[type(v) for v in obj_risk_max.values()]}")


        """
        Bayesian Principle: The risk of all detected road users are cumulated and normalized.
        Minimizing the overall Risk

        Calculate the risk cost via the Bayesian Principle for the given
        trajectory.

        Args:
            ego_harm_traj (Dict): Dictionary with collision data for all
                obstacles and all time steps.
            obj_risk_max (Dict): Dictionary with collision data for all
                obstacles and all time steps.

        Returns:
            Dict: Risk costs for the considered trajectory according to the
                Bayesian Principle
        """
        if len(ego_risk_max) == 0:
            return 0

        # sum_ego_risk1 = sum(ego_risk_max.values())
        sum_ego_risk = float(np.sum([np.max(v) for v in ego_risk_max.values()]))
        sum_obj_risk = float(np.sum([np.max(v) for v in obj_risk_max.values()]))
        return (sum_ego_risk +  sum_obj_risk+boundary_harm)/(
                len(ego_risk_max) * 2)
        # return (sum(ego_risk_max.values()) + sum(obj_risk_max.values()) + boundary_harm) / (
        #         len(ego_risk_max) * 2
        # )

    def get_equality_costs(self, ego_risk_max, obj_risk_max):
        """
        Equality Principle: distribute risks equally to all road users

        Calculate the risk cost via the Equality Principle for the given
        trajectory.

        Args:
            ego_harm_traj (Dict): Dictionary with collision data for all
                obstacles and all time steps.
            obj_risk_max (Dict): Dictionary with collision data for all
                obstacles and all time steps.

        Returns:
            float: Risk costs for the considered trajectory according to the
                Equality Principle
                :param ego_risk_max:
        """
        if len(ego_risk_max) == 0:
            return 0

        return sum(
            [abs(ego_risk_max[key] - obj_risk_max[key]) for key in ego_risk_max]
        ) / len(ego_risk_max)

    def get_maximin_costs(self, ego_risk_max, obj_risk_max, ego_harm_max, obj_harm_max, boundary_harm=0, eps=10e-10,
                          scale_factor=100):
        """
        Maximin Principle: Minimizing the greatest possible harm regardless of probablity

        Calculate the risk cost via the Maximin principle for the given
        trajectory.

        Args:
            ego_harm_traj (Dict): Dictionary with collision data for all
                obstacles and all time steps.
            obst_harm_traj (Dict): Dictionary with collision data for all
                obstacles and all time steps.
            timestep (Int): Currently considered time step.
            maximin_mode (Str): Select between normalized ego risk
                ("normalized") and partial risks ("partial").

        Returns:
            float: Risk costs for the considered trajectory according to the
                Maximum Principle
        """
        if len(ego_harm_max) == 0:
            return 0

        # Set maximin to 0 if probability (or risk) is 0
        maximin_ego = [a * int(b < eps) for a, b in zip(ego_harm_max.values(), ego_risk_max.values())]
        maximin_obj = [a * int(bool(b < eps)) for a, b in zip(obj_harm_max.values(), obj_risk_max.values())]

        return max(maximin_ego + maximin_obj + [boundary_harm]) ** scale_factor

    def get_ego_costs(self, ego_risk_max, boundary_harm=0):
        """
        Calculate the utilitarian ego cost for the given trajectory.

        Args:
            ego_harm_traj (Dict): Dictionary with collision data for all
                obstacles and all time steps.
            timestep (Int): Currently considered time step.

        Returns:
            Dict: Utilitarian ego risk cost
        """
        if len(ego_risk_max) == 0:
            return 0
        sum_ego_risk = float(np.sum([np.max(v) for v in ego_risk_max.values()]))
        return sum_ego_risk + boundary_harm

    def cal_risk(self, ego_traj, obj_traj):
        """返回值为：ego_risk_max, obj_risk_max, ego_harm_max, obj_harm_max"""

        coll_prob_dict = self.get_collision_probability(  # 计算碰撞概率
            traj=ego_traj,
            obj_prediction=obj_traj,
        )

        # print('coll_prob_dict',coll_prob_dict)
        ego_harm_traj, obj_harm_traj = self.cal_harm(  # 计算伤害值
            ego_traj=ego_traj,
            obj_prediction=obj_traj,
        )

        if ego_harm_traj is None or obj_harm_traj is None:
            return None, None, None, None
        ego_risk_traj = {}
        obj_risk_traj = {}
        ego_risk_max = {}
        obj_risk_max = {}
        ego_harm_max = {}
        obj_harm_max = {}
        high_collisions = {} 
        # Calculate risk out of harm and collision probability
        risk_cost = 0
        for key in ego_harm_traj:
            ego_risk_traj[key] = [
                ego_harm_traj[key][t] * coll_prob_dict[key][t]
                for t in range(len(ego_harm_traj[key]))
            ]

            obj_risk_traj[key] = [
                obj_harm_traj[key][t] * coll_prob_dict[key][t]
                for t in range(len(obj_harm_traj[key]))
            ]

            # Take max as representative for the whole trajectory 遍历所有旁车计算自车最大风险最大harm，旁车最大风险最大harm
            ego_risk_max[key] = max(ego_risk_traj[key])
            obj_risk_max[key] = max(obj_risk_traj[key])
            ego_harm_max[key] = max(ego_harm_traj[key])
            obj_harm_max[key] = max(obj_harm_traj[key])




        return ego_risk_max, obj_risk_max, ego_harm_max, obj_harm_max,risk_cost

    def cal_harm(self, ego_traj, obj_prediction):
        """计算自车轨迹以及旁车轨迹的harm然后对于harm的计算需要根据周围障碍物的类型进行计算。"""

        # 初始化自车参数
        ego_mass = 1500   # kg
        ego_pos = np.array(ego_traj['pos_list'], dtype=float)
        ego_v = np.array(ego_traj['v_list'], dtype=float)  # 怎么还需要车速？那么我还需要把每一步车速提取出来
        ego_yaw = np.array(ego_traj['orientation_list'], dtype=float)
        ego_harm_traj = {}
        obj_harm_traj = {}

        for obj in obj_prediction:
            # 按循环提取目标信息
            obj_type = obj['type']
            pred_path = np.array(obj['pos_list'], dtype=float)
            pred_length = min(len(ego_pos), len(pred_path))  # todo
            pred_v = np.array(obj['v_list'], dtype=float)[:pred_length]
            pred_yaw = np.array(obj['orientation_list'], dtype=float)[:pred_length]
            pred_size = obj['shape']['length'] * obj['shape']['width']
            obstacle_mass = self.get_obstacle_mass(obj_type)  # todo 根据传入障碍物的类型，计算m障碍物的大小
            ego_harm_fun, obstacle_harm_fun = self.get_harm_model(obj_type=obj_type)
            # 传入的obj_type最好是：{"vehicle": , "pedestrian": }
            # Initialize harm lists
            ego_harm_obst = np.zeros(pred_length)
            obst_harm_obst = np.zeros(pred_length)

            pdof_array = pred_yaw - ego_yaw[:pred_length]  # 自车与旁车航向差
            rel_angle_array = np.arctan2(pred_path[:, 1] - ego_pos[:pred_length, 1],
                                         pred_path[:, 0] - ego_pos[:pred_length, 0])
            ego_angle_array = rel_angle_array - ego_yaw[:pred_length]
            obs_angle_array = rel_angle_array - pred_yaw

            delta_v_array = np.sqrt(
                ego_v[:pred_length] ** 2 +
                pred_v ** 2 +
                2 * ego_v[:pred_length] * pred_v * np.cos(pdof_array)
            )

            ego_delta_v = (obstacle_mass / (ego_mass + obstacle_mass)) * delta_v_array
            obstacle_delta_v = (ego_mass / (ego_mass + obstacle_mass)) * delta_v_array

            ego_harm_obst = ego_harm_fun(  
                velocity=ego_delta_v,
            )

            obst_harm_obst = obstacle_harm_fun(  
                velocity=obstacle_delta_v,
            )

            ego_harm_traj[obj['id']] = ego_harm_obst
            obj_harm_traj[obj['id']] = obst_harm_obst

        return ego_harm_traj, obj_harm_traj

    def get_harm_model(self, obj_type):  # 速度没有传入
        obstacle_protection = {
            "vehicle": True,  # 车内乘员有车身保护
            "pedestrian": False,  # 行人无保护
            "bicycle": False  # 自行车无保护
        }

        def get_obstacle_protection(obj_type: str) -> bool:
            """
            根据对象类型返回是否有保护
            - vehicle: True
            - pedestrian: False
            其他未知类型 → None
            """
            return obstacle_protection.get(obj_type, None)

        protection_flag = get_obstacle_protection(obj_type)
        if protection_flag is True:  # 对于车与车的碰撞
            ego_harm = lambda velocity: 1 / (  # noqa E731
                    1 + np.exp(5.820 - 0.292 * np.maximum(10,velocity))
            )
            obstacle_harm = lambda velocity: 1 / (  # noqa E731
                    1
                    + np.exp(5.820 - 0.292 * np.maximum(10,velocity))
            )
        elif protection_flag is False:  # 车与行人的碰撞
            ego_harm = lambda velocity: 1 / (  # noqa E731
                    1 + np.exp(3.164 - 0.292 * velocity)
            )

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity: 1 / (1 + np.exp(1.786 - 0.259 * velocity))
        else:
            ego_harm = lambda velocity: 1  # noqa E731  #出现未知类型直接输出最大伤害
            obstacle_harm = lambda velocity: 1  # noqa E731
        # print('obstacle_harm', ego_harm)

        return ego_harm, obstacle_harm

    def get_obstacle_mass(self, type):
        if type == 'car':
            return 1000
        elif type == 'bicycle':
            return 85
        else:
            return 70

    def get_collision_probability(self, traj, obj_prediction):
        collision_prob_dict = {}
        #   得到碰撞概率
        """自车traj为字典类型包含：位置pos，朝向ori
        旁车obj_prediction为列表，每个对象至少包含目标位置pos，朝向ori"""
        # 自车轨迹信息从键名提取：
        # print(traj)
        ego_traj_pose = traj['pos_list']
        pred_length = len(ego_traj_pose)
        traj_x = ego_traj_pose[:, 0][0:pred_length]
        traj_y = ego_traj_pose[:, 1][0:pred_length]
        ego_length = traj['shape']['length']
        ego_width = traj['shape']['width']
        # print('ego_traj_pose',ego_traj_pose)
        # print('other_obj',obj_prediction)
        # 只保留有航向角的旁车
        valid_obj_predictions = [
            obj for obj in obj_prediction
            if 'orientation_list' in obj
        ]

        # 根据旁车数量做循环：
        for obj in valid_obj_predictions:
            obj_id = obj['id']
            obj_mean_list = obj['pos_list']
            # obj_cov_list = obj['cov_list']
            obj_yaw_list = obj['orientation_list']
            # print('obj_yaw_list',obj_yaw_list)
            obj_length = obj['shape']['length']
            obj_width = obj['shape']['width']
            obj_probs = []
            # 如果预测包含预测的协方差则这里加入逆协方差。

            for i in range(len(traj_x)):  # 根据自车预测的x的长度来遍历旁车所需长度
                if i >= len(obj_mean_list):
                    break

                # get the current position of the ego vehicle
                ego_pos = [traj_x[i], traj_y[i]]
                ego_yaw = traj['orientation_list'][i]

                # get the mean and the covariance of the prediction
                mean = obj_mean_list[i]
                obj_yaw = obj_yaw_list[i]
                # iv = obj_inv_cov_list[i]

                # Check for overlap using SAT (Separating Axis Theorem)
                ego_corners = self.get_four_corners(  # todo
                    position=ego_pos,
                    length=ego_length,
                    width=ego_width,
                    angle=ego_yaw
                )
                obj_corners = self.get_four_corners(  # todo
                    position=mean,
                    length=obj_length,
                    width=obj_width,
                    angle=obj_yaw
                )
                if self.separating_axis_theorem(ego_corners, obj_corners):
                    # If there's a potential overlap, calculate the collision probability based on uncertainty

                    prob = self.prob_with_mahalanobis_chi_squared(
                        ego_pose=ego_pos,
                        obj_pose=mean
                    )
                else:
                    prob = 0  # No overlap, no collision probability

                obj_probs.append(prob)
            collision_prob_dict[obj_id] = obj_probs

        return collision_prob_dict

    def get_four_corners(self, position, length, width, angle):
        """
        Get the corners of a rectangle given its position, size, and orientation.

        Parameters:
        position (np.array): The center position of the rectangle (x, y).
        length (float): The length of the rectangle.
        width (float): The width of the rectangle.
        angle (float): The orientation angle of the rectangle (in radians).

        Returns:
        np.array: The four corners of the rectangle.
        """

        def rotate_point(point, angle):
            """Rotate a 2D point by an angle (in radians)."""
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
            return np.dot(rotation_matrix, point)


        half_length = length / 2.0
        half_width = width / 2.0
        corners = np.array([[-half_length, -half_width],
                            [half_length, -half_width],
                            [half_length, half_width],
                            [-half_length, half_width]])
        rotated_corners = np.array([rotate_point(corner, angle) for corner in corners])

        return rotated_corners + np.array(position)

    def separating_axis_theorem(self, corners1, corners2, tolerance=0.1, early_exit=True):
        """
        Apply the Separating Axis Theorem (SAT) to check if two rectangles collide.

        Parameters:
        corners1 (np.array): The corners of the first rectangle (4x2 array).
        corners2 (np.array): The corners of the second rectangle (4x2 array).
        tolerance (float): The tolerance value for near-collision detection.
        early_exit : IF we do early exit, first simple calculate the distances

        Returns:
        bool: True if the rectangles overlap (collide), False otherwise.
        """
        if early_exit:
            # Calculate bounding box centers
            center1 = np.mean(corners1, axis=0)
            center2 = np.mean(corners2, axis=0)

            # Calculate bounding box half-diagonals
            half_diagonal1 = np.sqrt(np.sum((corners1[0] - corners1[2]) ** 2)) / 2
            half_diagonal2 = np.sqrt(np.sum((corners2[0] - corners2[2]) ** 2)) / 2

            # Calculate distance between centers
            center_distance = np.linalg.norm(center1 - center2)

            # Early exit: if center distance is greater than the sum of the half-diagonals
            if center_distance > (half_diagonal1 + half_diagonal2 + tolerance):
                return False  # No collision possible, exit early

        axes = np.array([
            corners1[1] - corners1[0],  # Edge 1 of rect1
            corners1[3] - corners1[0],  # Edge 2 of rect1
            corners2[1] - corners2[0],  # Edge 1 of rect2
            corners2[3] - corners2[0]  # Edge 2 of rect2
        ])

        axes = np.array([axis / np.linalg.norm(axis) for axis in axes])

        for axis in axes:
            projections1 = np.dot(corners1, axis)
            projections2 = np.dot(corners2, axis)
            if max(projections1) < min(projections2) - tolerance or max(projections2) < min(projections1) - tolerance:
                return False  # Separating axis found, no collision
        return True  # No separating axis found, collision occurs

    def prob_with_mahalanobis_chi_squared(self, ego_pose, obj_pose, collision_threshold=0.0):
        """
        Calculate the collision probability using the Mahalanobis distance
        and the chi-squared distribution.

        Parameters:
        r (np.array): The relative position vector (x, y) between two vehicles.
        cov (np.array): The combined covariance matrix (2x2).
        collision_threshold (float): The threshold for collision detection (e.g., size of bounding boxes).

        Returns:
        float: Collision probability.
        """
        # Compute the Mahalanobis distance
        # d_m = mahalanobis(ego_pose, obj_pose, cov_mat)     #若后续预测有协方差的情况
        d_m = np.linalg.norm(np.array(ego_pose) - np.array(obj_pose))

        prob = 1-chi2.cdf(d_m ** 2, df=2)     #做了修改 原为chi2

        # Compute the collision probability using a chi-squared distribution
        return min(prob,1)

    def _compute_vref_from_scene(self):
        # === 0) 读取状态（你已有） ===
        ego_v = float(self.ego_info.loc['ego','v'])
        ego_lane = int(self.ego_info.loc['ego','lane'])
        idx = int(self.ego_info.loc['ego','rel_pos_ind'])
        ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)

        # === 1) 曲率候选 ===
        # 取前方窗口平均曲率，和你 get_features 里做法一致
        ahead = [min(idx+i, len(ref_kappa)-2) for i in (2,3,4,5,6)]
        k_bar = float(np.mean([abs(ref_kappa[i]) for i in ahead])) if ahead else abs(ref_kappa[idx])
        ay_max = 2.0  # 可调：1.8~2.5 m/s^2
        eps = 1e-4
        v_kappa = math.sqrt(ay_max / max(k_bar, eps))

        # === 2) 物理/限速候选 ===
        v_phys = 30

        # === 3) 前车候选（简式） ===
        v_lead = float('inf')
        if isinstance(self.front, pd.DataFrame) and not self.front.empty:
            j = self.front['rel_pos_ind'].idxmin()
            d = float(self.front['rel_des'][j])
            v_front = float(self.front['v'][j])
            L_front = float(self.front['length'][j])
            d0, tau = 3.0, 1.5
            d_safe = d0 + tau*max(ego_v,0.0) + 0.5*L_front
            if d < 50.0:  # 只在近距离内生效，防抖
                kd = 0.6
                v_lead = max(0.0, v_front + kd*(d - d_safe))

        # === 7) 合成（取最小，再乘抑制） ===
        v_limit = min(v_kappa, v_lead, v_phys)
        v_ref = v_limit
        # === 8) 平滑 & 限幅 ===
        if not hasattr(self, "_vref_filt"):
            self._vref_filt = v_ref
        alpha = 0.3
        self._vref_filt = (1-alpha)*self._vref_filt + alpha*v_ref
        return float(np.clip(self._vref_filt, 0.0, v_phys))
    
    def _compute_planning_horizon_1(self, target_velocity, lateral_offset):
        """
        基于优化问题计算规划时间T
        :param target_velocity: 期望车速
        :param lateral_offset: 横向偏移
        :return: 规划时间T
        """
        # 获取当前状态
        ego_v = float(self.ego_info.loc['ego', 'v'])
        ego_a = float(self.ego_info.loc['ego', 'a'])
        ego_lane = int(self.ego_info.loc['ego', 'lane'])
        idx = int(self.ego_info.loc['ego', 'rel_pos_ind'])
        
        # 曲率约束（避免过大的横向加速度）
        ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)
        kappa = ref_kappa[idx]
        max_lateral_acceleration = 2.0  # m/s^2
        if abs(kappa) > 1e-5:
            v_curve = math.sqrt(max_lateral_acceleration / abs(kappa))
        else:
            v_curve = float('inf')
        
        # 速度变化所需时间
        if abs(target_velocity - ego_v) < 0.1:
            t_velocity = 0.0
        else:
            max_acceleration = 5.0  # m/s^2
            t_velocity = abs(target_velocity - ego_v) / max_acceleration
        
        # 横向移动所需时间（假设横向速度最大1m/s）
        max_lateral_speed = 1.0  # m/s
        current_d, _ = self._calculate_signed_lateral_offset(
            np.array([self.ego_info.loc['ego', 'x'], self.ego_info.loc['ego', 'y']]),
            idx, ref_center, ref_phi
        )
        lateral_distance = abs(lateral_offset - current_d)
        if lateral_distance > 0.1:
            t_lateral = 0
        else:
            t_lateral = 0.0
        # print('t_velocity:',t_velocity,'t_lateral:',t_lateral)
        # 规划时间T取最大值，并加上缓冲时间
        T = max(t_velocity, t_lateral) + 0.1  # 缓冲1秒
        
        # 约束在合理范围内
        T = np.clip(T, 0.0, 5.0)
        
        return T

    def _compute_planning_horizon(self, target_velocity: float, lateral_offset: float) -> float:
        """
        计算安全的规划时间 T（s）：
        - 纵向：满足速度变化与加速度/加加速度（jerk）约束
        - 横向：满足偏移位移所需时间（v_d,max, a_d,max）
        - 曲率：限制侧向加速度 a_y = v^2 * |kappa|
        - 前车/障碍：满足最小 TTC 缓冲
        - 道路：不超过可用弧长与急弯/车道终点的安全停车需求
        """
        import math
        import numpy as np

        # -------- 当前状态 --------
        ego_v   = float(self.ego_info.loc['ego', 'v'])
        ego_a   = float(self.ego_info.loc['ego', 'a'])
        ego_lane = int(self.ego_info.loc['ego', 'lane'])
        idx     = int(self.ego_info.loc['ego', 'rel_pos_ind'])
        ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)

        # ===== 1) 曲率与侧向加速度约束 =====
        ay_max = 2.0  # m/s^2（与你 _compute_vref_from_scene 一致/略保守）
        kappa_here = float(ref_kappa[int(np.clip(idx, 0, len(ref_kappa)-1))])
        if abs(kappa_here) > 1e-6:
            v_curve = math.sqrt(ay_max / max(abs(kappa_here), 1e-6))
        else:
            v_curve = float('inf')
        v_allowed = min(target_velocity, v_curve)  # 保守处理：目标车速不得超过曲率上限

        # ===== 2) 速度变化所需时间（考虑 a_max 与 jerk 上限）=====
        a_max = 3.0   # m/s^2（可按车型调参）
        j_max = 3.0   # m/s^3（平滑度/舒适性）
        dv = abs(v_allowed - ego_v)
        # 采用简单 S-curve 近似：若受 jerk 限制，t >= 2*sqrt(dv/j_max)；再与 dv/a_max 取最大
        t_vel = max(dv / max(a_max, 1e-6), 2.0 * math.sqrt(max(dv, 0.0) / max(j_max, 1e-6)))

        # ===== 3) 横向位移所需时间（修复你代码中的 t_lateral 逻辑）=====
        # 先算当前带符号 d，再算与 lateral_offset 的差
        ego_pos = np.array([self.ego_info.loc['ego', 'x'], self.ego_info.loc['ego', 'y']], dtype=float)
        current_d, _ = self._calculate_signed_lateral_offset(ego_pos, idx, ref_center, ref_phi)
        d_req = abs(float(lateral_offset) - float(current_d))
        vd_max = 1.2   # m/s  横向速度上限（可与车辆动力学/舒适度匹配）
        ad_max = 1.5   # m/s^2 横向加速度上限
        # 梯形速度轮廓时间估计：加速段+匀速段+减速段
        if d_req <= (vd_max**2 / ad_max):  # 达不到匀速平台的三角形轮廓
            t_lat = 2.0 * math.sqrt(d_req / max(ad_max, 1e-6))
        else:
            t_lat = 2.0 * (vd_max / ad_max) + (d_req - vd_max**2 / ad_max) / vd_max

        # ===== 4) 前方可用距离与停车可行性 =====
        s0 = float(self._s_from_idx(station, idx))
        s_last = float(station[-1]) if len(station) else s0 + 1.0
        # 可用弧长（到当前车道末端/路段末尾）
        ds_free = max(s_last - s0, 0.0)
        # 以当前/目标速度、最大减速度估算“安全停车距离”，并反推时间上界
        a_brake = 4.0  # m/s^2（紧急制动能力）
        v_probe = max(ego_v, v_allowed)
        d_stop = (v_probe**2) / (2.0 * max(a_brake, 1e-6)) + 3.0  # +3m 缓冲
        # 若 ds_free 不足以再跑很久，限制 T 不能过大（粗估以保守常速 v_probe）
        t_range = ds_free / max(v_probe, 0.5)
        # 若必须具备“随时停住”的能力，则 T 不能超过“到必须决策点”的时间
        t_endcap = max(1.0, min(t_range, 6.0))  # 保守把硬上限卡在 6s 以内

        # ===== 5) 与前车/障碍的 TTC 约束（简式）=====
        ttc_min = 2.0  # s  期望的最小时距
        t_ttc = 0.0
        if hasattr(self, "front") and isinstance(self.front, pd.DataFrame) and not self.front.empty:
            j = self.front['rel_pos_ind'].idxmin()
            # 只做粗约：以相对速度和距离估 TTC（更精细可用你已有 predictor）
            rel_v = float(self.front['v'][j]) - ego_v
            rel_s = float(self.front['rel_des'][j])
            if rel_v < 0.0 and rel_s > 1.0:  # 我更快才有追尾风险
                ttc = rel_s / max(-rel_v, 1e-3)
                if ttc < ttc_min:
                    # 若 TTC 过小，则 T 至少覆盖 ttc_min 以便采取保守动作
                    t_ttc = ttc_min

        # ===== 6) 汇总取最大 + 缓冲，并限幅/平滑 =====
        T_candidates = [
            1.0,          # 基本下限
            t_vel,        # 纵向速度变化需要
            t_lat,        # 横向偏移需要
            t_ttc         # 碰撞时距缓冲
        ]
        T = max(T_candidates) + 0.5  # +0.5s 缓冲
        # 全局限幅（与你 action range 对齐更安全）：[1.0, 2.0] 或按项目需求放宽
        T_min, T_max = 1.0, 8.0
        T = float(np.clip(T, T_min, min(T_max, t_endcap)))

        # 简单一阶滤波防抖（可选）
        if not hasattr(self, "_T_filt"):
            self._T_filt = T
        alpha = 0.3
        self._T_filt = (1 - alpha) * self._T_filt + alpha * T
        return float(self._T_filt)

    def _get_goal_lane(self):
        """
        返回目标区域所在的车道ID
        """
        goal_pos = np.array([self.goal_center_x, self.goal_center_y])
        best_lane = None
        min_dist = float('inf')
        
        for lane_id, road_info in self.road_info_dict_pingjie.items():
            center_points = road_info['center_vertices']
            distances = np.linalg.norm(center_points - goal_pos, axis=1)
            current_min = np.min(distances)
            if current_min < min_dist:
                min_dist = current_min
                best_lane = int(lane_id)
        
        return best_lane
    
    def _nearest_dist_and_ttc_xy(self, ego_x, ego_y, ego_v, ego_yaw, t_horizon: float = 20.0, ttc_max: float = 20.0):
        """
        更准确的 d_min & TTC 计算（常速直线假设）：
        - d_min_now: 当前时刻自车与所有邻车的中心到中心最小欧氏距离（m）
        - ttc_min:   在 [0, t_horizon] 内，若最近接距离进入安全阈值（考虑车体尺寸），则返回该时刻，否则为 +inf

        做法：
        1) 相对位置 r = p_other - p_ego
        2) 相对速度 v = v_other - v_ego
        3) 使用二次方程计算碰撞时间 t*，并判断是否发生碰撞
        4) 将TTC转换为对数尺度并归一化
        """
        import numpy as np, math, pandas as pd

        # 汇总候选邻车
        cand = []
        if isinstance(self.front, pd.DataFrame) and not self.front.empty:
            cand.append(self.front)
            # print("front:",self.front)
        if isinstance(self.behind, pd.DataFrame) and not self.behind.empty:
            cand.append(self.behind)
        if not cand:
            return 1e9, 1e9

        neigh = pd.concat(cand, axis=0, ignore_index=True)

        # 自车速度向量
        dir_ego = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=float)
        v_ego_vec = dir_ego * float(ego_v)

        # 自车外接圆半径（长宽对角线的一半；缺省 4.5x1.8）
        ego_L = float(getattr(self, "ego_info", {}).loc['ego', 'length']) if hasattr(self, "ego_info") and len(self.ego_info) else 4.5
        ego_W = float(getattr(self, "ego_info", {}).loc['ego', 'width']) if hasattr(self, "ego_info") and len(self.ego_info) else 1.8
        R_ego = 0.5 * math.hypot(ego_L, ego_W)

        d_min_now = 1e9
        ttc_min = 1e9
        eps = 1e-2
        safety_margin = 0.2  # 可选：为安全留一点边界（m）

        # 存储每辆车的TTC
        ttc_info = {}
        # for _, row in neigh.iterrows():
        #     # 提取并做容错
        #     try:
        #         ox = float(row.get('x', 0.0))
        #         oy = float(row.get('y', 0.0))
        #         ov = float(row.get('v', 0.0))
        #         oyaw = float(row.get('yaw', 0.0))
        #         oL = float(row.get('length', 4.5))
        #         oW = float(row.get('width', 1.8))
        #         other_r =  0.5 * math.hypot(oL, oW)
        #         # print('other_r', other_r)
        #         name = row.get('name', str(row.get('x', 0)))  # 通过 'name' 字段获取车辆标识符，若没有则用位置坐标
        #     except Exception:
        #         continue
        obs = getattr(self, "observation", None)
        for obj_type, group in obs.object_info.items():
            if not isinstance(group, dict):
                continue
            for key, value in group.items():

                # 如果数据结构是对象（ExternObject），用 vars(value)；如果本身就是 dict 就直接用
                if isinstance(value, dict):
                    d = value
                else:
                    try:
                        d = vars(value)
                    except Exception:
                        continue  # 非 dict/无 __dict__ 的，跳过

                # 取字段（缺失用兜底）—— 注意不要用位置索引
                try:
                    ox   = float(d.get('x', 0.0))
                    oy   = float(d.get('y', 0.0))
                    ov   = float(d.get('v', 0.0))
                    oyaw = float(d.get('yaw', 0.0))
                    oL   = float(d.get('length', 4.5))
                    oW   = float(d.get('width', 1.8))
                    other_r =  0.5 * math.hypot(oL, oW)
                    # print('other_r', other_r)
                    name = d.get('name', str(d.get('x', 0)))
                except Exception:
                    continue
                # 当前位置最近距离（中心到中心）
                r = np.array([ox - ego_x, oy - ego_y], dtype=float)
                dist = float(np.linalg.norm(r))
                d_min_now = min(d_min_now, dist)

                # 相对速度与最近接时刻
                v_other_vec = np.array([math.cos(oyaw), math.sin(oyaw)], dtype=float) * ov
                v_rel = v_other_vec - v_ego_vec
                v_rel2 = float(np.dot(v_rel, v_rel))

                if v_rel2 <= eps:
                    d_star = dist
                    # 是否已经在阈值内
                    R_tot = R_ego + other_r - safety_margin
                    if dist <= R_tot:
                        # print('no safe')
                        t_star = 0.0          # 现在就不安全
                    else:
                        t_star = 20 # 永远不会进入阈值
                else:
                    # 解二次方程
                    A = v_rel2
                    B = 2 * np.dot(r, v_rel)
                    # C = np.dot(r, r) - (R_ego + R_ego)**2
                    # print('rel, r_ego, other_r', np.dot(r, r), R_ego, other_r)
                    C = np.dot(r, r) - (R_ego + other_r - 0.2)**2
                    # 解方程
                    discriminant = B**2 - 4 * A * C
                    if discriminant < 0.00001:
                        # print('wujie')
                        t_star = 20  # 无解，表示没有碰撞
                    else:
                        t_star = (-B - math.sqrt(discriminant)) / (2 * A)
                        # print('t_star_raw:', t_star)
                        # print('t_another:', (-B + math.sqrt(discriminant)) / (2 * A))
                        if t_star < 0.0:
                            t_star = 20  # 过去的碰撞不考虑
                        t_star = max(0.0, min(t_star, t_horizon))  # 限制在[0, t_horizon]范围内
                    d_star = float(np.linalg.norm(r + v_rel * t_star))

                # 计算TTC的对数尺度
                if t_star != float('inf'):
                    log_ttc = math.log((t_star + 0.0001) / ttc_max)  # 将TTC值转为对数尺度
                    # 归一化到 [0, 1] 区间
                    normalized_log_ttc = (log_ttc - math.log(1e-3)) / (math.log(ttc_max) - math.log(1e-3))
                    ttc = normalized_log_ttc
                else:
                    ttc = float('inf')

                # 记录当前车辆的TTC
                ttc_info[name] = t_star

                # 更新最小TTC
                # print('min', ttc_min, 'this', ttc)
                if ttc < ttc_min:
                    ttc_min = ttc

        # 输出每辆车的TTC
        # print("车辆 TTC 信息: ", ttc_info)
        # print(ttc_min)
        return d_min_now, ttc_min

    # ===== (1) 函数：计算汇入点 s_merge =====
    def compute_merge_s(self,
                        ego_lane: int,
                        ego_idx: int,
                        kappa_thr: float = 0.02,
                        consec: int = 8,
                        back_offset_m: float = 4.0,
                        search_ahead_pts: int = 200) -> float:
        """
        返回：汇入点的 s 坐标（沿当前参考路径）
        依赖：_get_lane_arrays, _s_from_idx
        """
        import numpy as np

        ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)
        kappa_abs = np.abs(ref_kappa)

        # 找到连续 consec 个 |kappa| >= kappa_thr 的起始索引
        ring_start_idx, run = None, 0
        for i in range(int(ego_idx), len(kappa_abs)):
            if kappa_abs[i] >= kappa_thr:
                run += 1
                if run >= consec:
                    ring_start_idx = i - consec + 1
                    break
            else:
                run = 0

        # 兜底：未来窗口内取最大曲率点
        if ring_start_idx is None:
            window_end = min(int(ego_idx) + int(search_ahead_pts), len(kappa_abs) - 1)
            ring_start_idx = int(np.argmax(kappa_abs[int(ego_idx):window_end]) + int(ego_idx))

        s_ring_start = float(self._s_from_idx(station, ring_start_idx))
        s_merge = max(float(station[0]), s_ring_start - float(back_offset_m))
        return s_merge

    # ======================= (A) s_merge 精准计算 =======================
    def compute_merge_s_precise(
        self,
        ego_lane: int,
        ego_idx: int,
        kappa_thr: float = 0.02,
        consec: int = 8,
        min_back_m: float = 2.0,
        max_back_m: float = 10.0,
        t_buffer: float = 0.6,        # 速度相关回退（秒）
        search_ahead_pts: int = 300,
        smooth_win: int = 9,          # 曲率平滑窗口（奇数）
    ) -> float:
        """
        更精准地计算汇入点 s_merge：
        1) |kappa| 平滑；2) 阈值穿越插值定位起点；3) 兜底：峰值 + 二次细化；
        4) 回退距离 back_offset = clip(v_now * t_buffer, [min_back_m, max_back_m]).
        依赖: self._get_lane_arrays(lane) 提供 (ref_center, ref_phi, ref_kappa, station)
        """
        import numpy as np
        ref_center, ref_phi, ref_kappa, station = self._get_lane_arrays(ego_lane)
        kappa = np.asarray(ref_kappa, dtype=float)
        S = np.asarray(station, dtype=float)
        N = len(kappa)
        ego_idx = int(np.clip(ego_idx, 0, N-2))

        # 1) 平滑 |kappa|
        kappa_abs = np.abs(kappa)
        if smooth_win < 3: smooth_win = 3
        if smooth_win % 2 == 0: smooth_win += 1
        pad = smooth_win // 2
        k_pad = np.pad(kappa_abs, (pad, pad), mode="edge")
        kernel = np.ones(smooth_win, dtype=float) / smooth_win
        kappa_s = np.convolve(k_pad, kernel, mode="valid")

        # 2) 连续门限 + 穿越插值
        ring_start_idx, run = None, 0
        for i in range(ego_idx, N):
            if kappa_s[i] >= kappa_thr:
                run += 1
                if run >= consec:
                    ring_start_idx = i - consec + 1
                    break
            else:
                run = 0

        def s_from_idx_frac(idx_float: float) -> float:
            i0 = int(np.clip(np.floor(idx_float), 0, N-2))
            i1 = i0 + 1
            a = float(np.clip(idx_float - i0, 0.0, 1.0))
            return float(S[i0] + a * (S[i1] - S[i0]))

        if ring_start_idx is not None:
            i0 = max(ego_idx, ring_start_idx - 1)
            while i0 > ego_idx and kappa_s[i0] >= kappa_thr:
                i0 -= 1
            i1 = i0 + 1
            y0, y1 = kappa_s[i0], kappa_s[i1]
            if y1 != y0:
                frac = (kappa_thr - y0) / (y1 - y0)
                idx_frac = i0 + float(np.clip(frac, 0.0, 1.0))
            else:
                idx_frac = float(ring_start_idx)
            s_ring_start = s_from_idx_frac(idx_frac)
        else:
            # 3) 峰值兜底 + 二次细化
            end = min(ego_idx + int(search_ahead_pts), N-1)
            win = kappa_s[ego_idx:end]
            peak_i = int(np.argmax(win)) + ego_idx
            iL, iC, iR = max(peak_i-1, 0), peak_i, min(peak_i+1, N-1)
            x = np.array([iL, iC, iR], dtype=float)
            y = kappa_s[x.astype(int)]
            A = np.vstack([x**2, x, np.ones_like(x)]).T
            try:
                a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
                x_peak = float(-b / (2*a)) if a != 0 else float(iC)
                x_peak = float(np.clip(x_peak, iL, iR))
            except np.linalg.LinAlgError:
                x_peak = float(iC)
            s_ring_start = s_from_idx_frac(x_peak)

        # 4) 速度自适应回退
        try:
            v_now = float(self.ego_info.loc['ego', 'v'])
        except Exception:
            v_now = 0.0
        back_offset = float(np.clip(v_now * float(t_buffer), float(min_back_m), float(max_back_m)))

        s0 = float(S[0])
        s_merge = max(s0, float(s_ring_start) - back_offset)
        return float(s_merge)

    # ==== 放在 Sumoenv 类里（建议放到其它工具函数附近）====
    def _time_to_point_cartesian(self, x, y, v, yaw, x_tar, y_tar, eps=1e-3):
        """用笛卡尔坐标估算到达指定点的时间（常速直线近似）。不朝向目标则返回 +inf。"""
        import numpy as np, math
        p = np.array([float(x), float(y)], dtype=float)
        v_vec = np.array([math.cos(float(yaw)), math.sin(float(yaw))], dtype=float) * float(v)
        d = np.array([float(x_tar), float(y_tar)], dtype=float) - p
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return 0.0
        d_hat = d / dist
        proj_speed = float(np.dot(v_vec, d_hat))  # 速度沿目标方向的分量
        if proj_speed <= eps:
            return float('inf')
        return dist / max(proj_speed, eps)
    
    def extract_roundabout_from_xodr(self, xodr_path, tol_center=1.0, tol_radius=1.0, min_arc_length=5.0):
        """解析XODR文件，返回主环岛的圆心和半径"""
        tree = ET.parse(xodr_path)
        root = tree.getroot()

        def center_from_arc(x, y, hdg, curvature):
            R = 1.0 / abs(curvature)
            sign = 1.0 if curvature > 0 else -1.0
            cx = x - sign * R * math.sin(hdg)
            cy = y + sign * R * math.cos(hdg)
            return cx, cy, R, sign

        arcs = []
        for road in root.findall(".//road"):
            for g in road.findall("./planView/geometry"):
                arc = g.find("arc")
                if arc is None:
                    continue
                length = float(g.attrib.get("length", "0"))
                if length < min_arc_length:
                    continue
                k = float(arc.attrib.get("curvature", "0"))
                if abs(k) < 1e-9:
                    continue
                x, y, hdg = map(float, (g.attrib["x"], g.attrib["y"], g.attrib["hdg"]))
                cx, cy, R, _ = center_from_arc(x, y, hdg, k)
                arcs.append((cx, cy, R))

        # 聚类
        clusters = {}
        for cx, cy, R in arcs:
            key = (round(cx / tol_center), round(cy / tol_center), round(R / tol_radius))
            clusters.setdefault(key, []).append((cx, cy, R))

        if not clusters:
            return None, None, None

        best = max(clusters.values(), key=len)
        mean_cx = sum(c[0] for c in best) / len(best)
        mean_cy = sum(c[1] for c in best) / len(best)
        mean_R = sum(c[2] for c in best) / len(best)
        return mean_cx, mean_cy, mean_R
    
    def _in_roundabout_xy(self, x: float, y: float, lane_width: float = 7.0, tol: float = 0.5) -> bool:
        """
        用圆心+半径+车道宽度判断 (x,y) 是否在环岛道路环带内。
        - lane_width: 环岛车道宽度（默认 7m）
        - tol: 允许的数值公差（米），避免边界抖动
        """
        # 未解析到环岛参数时，直接返回 False（或按需返回 True/跳过判断）
        if getattr(self, "roundabout_radius", None) in (None, float("inf")):
            return False

        cx = float(self.roundabout_center_x)
        cy = float(self.roundabout_center_y)
        R  = float(self.roundabout_radius)

        r    = ((x - cx)**2 + (y - cy)**2) ** 0.5
        r_in = max(R - lane_width / 2.0, 0.0)
        r_out= R + lane_width / 2.0
        return (r_in - tol) <= r <= (r_out + tol)

    def _interp_heading_unit(self, phi0, phi1, alpha):
        """单位航向向量插值，避免角度跳变"""
        import numpy as np, math
        h0 = np.array([math.cos(float(phi0)), math.sin(float(phi0))], dtype=float)
        h1 = np.array([math.cos(float(phi1)), math.sin(float(phi1))], dtype=float)
        h  = (1.0 - float(alpha)) * h0 + float(alpha) * h1
        nrm = float(np.linalg.norm(h))
        return h0 if nrm < 1e-8 else (h / nrm)

    def _segment_projection_alpha(self, P, C0, C1):
        """点 P 到线段 C0->C1 的投影参数 α∈[0,1]"""
        import numpy as np
        e = C1 - C0
        denom = float(np.dot(e, e))
        if denom <= 1e-12:
            return 0.0
        return float(np.clip(np.dot(P - C0, e) / denom, 0.0, 1.0))

    def _project_point_to_centerline(self, P, ref_center, ref_phi, station, guess_idx):
        """
        将笛卡尔点 P=[x,y] 正交投影到中心线（折线），返回：
        s_m_cont: 连续弧长（m）
        d_signed: 带符号横向偏移（左正）
        idx_used: 使用的段起点索引
        alpha_used: 段内插值参数 α
        h_unit: 插值后的单位切向
        """
        import numpy as np, math
        N = len(ref_center)
        if N < 2:
            idx = int(np.clip(int(guess_idx), 0, max(0, N-2)))
            C = ref_center[idx]
            phi = ref_phi[idx]
            h_unit = np.array([math.cos(float(phi)), math.sin(float(phi))], dtype=float)
            n_unit = np.array([-h_unit[1], h_unit[0]], dtype=float)
            s_m = float(station[int(idx)])
            d_signed = float(np.dot(P - C, n_unit))
            return s_m, d_signed, int(idx), 0.0, h_unit

        # 仅在 guess_idx 周围两段试投影： [i-1,i], [i,i+1]
        candidates = []
        for base in [int(guess_idx)-1, int(guess_idx)]:
            idx0 = int(np.clip(base, 0, N-2))
            C0 = ref_center[idx0]
            C1 = ref_center[idx0+1]
            alpha = self._segment_projection_alpha(P, C0, C1)
            Q = (1.0 - alpha) * C0 + alpha * C1
            dist2 = float(np.dot(P - Q, P - Q))
            candidates.append((dist2, idx0, alpha))
        candidates.sort(key=lambda t: t[0])
        _, idx_used, alpha_used = candidates[0]

        # 连续 s
        s0 = float(station[idx_used]); s1 = float(station[idx_used+1])
        s_m_cont = s0 + float(alpha_used) * max(s1 - s0, 1e-8)

        # 插值切向 & 左法向
        phi0 = float(ref_phi[idx_used])
        phi1 = float(ref_phi[min(idx_used+1, len(ref_phi)-1)])
        h_unit = self._interp_heading_unit(phi0, phi1, alpha_used)
        n_unit = np.array([-h_unit[1], h_unit[0]], dtype=float)

        # 段内参考点 Q，用于 d
        C0 = ref_center[idx_used]; C1 = ref_center[idx_used+1]
        Q = (1.0 - alpha_used) * C0 + alpha_used * C1
        d_signed = float(np.dot(P - Q, n_unit))

        return float(s_m_cont), float(d_signed), int(idx_used), float(alpha_used), h_unit


class Reply():
    def __init__(self):
        self.source_file = {"tess": "", "xodr": "", "xosc": "", "json": ""}
        
   
