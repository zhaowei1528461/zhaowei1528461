import os
import math
# from saferule import safe
os.environ["PATH"] += os.pathsep + "/usr/bin/graphviz"
import sys
import random
from scipy.spatial import KDTree
# choose which device to use
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__) , '..' , '..' , '..' , '..' , "tools"))
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

from sumolib import checkBinary
import traci.constants as tc
import traci
from gym import spaces
import numpy as np
import pandas as pd
import math
from .smarts_math import wrap_value, position_to_ego_frame
from .risk_cost import TrajectoryRiskEvaluator
from .predictor_future import predictor_future_only
from .danger_scenarios import StructuredDangerScenarioMiner
from collections import deque, defaultdict
import warnings
warnings.filterwarnings("ignore")
class Sumoenv():
    _shared_cfgs = {}
    _shared_rng = random.Random(0)

    def __init__(
            self,
            path="/home/ubuntu/zw/safedreamer_fpi_danger_huifang/embodied/generated_scenarios_merge_7/scenario_000_easy_veh000.sumocfg",
            state_dim=31,
            action_dim=1,
            worker_index=0,
            env_amount=1,
            env_mode='train',
            danger_config=None,
            danger_output_dir=''):
        self.path = path
        self._sumo_gui = False
        self._sumo_started = False
        self.worker_index = int(worker_index)
        self.env_amount = max(1, int(env_amount))
        self.env_mode = str(env_mode)
        self.scenario_group_size = 2 if self.env_amount >= 2 else 1
        self._scenario_mode = 'normal'
        self._scenario_sweep_index = self.worker_index
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.egoid = 'Auto'
        self.last_acc = 0
        self.last_action = 0
        self.cfg_dir = "/home/ubuntu/zw/safedreamer_fpi_danger_huifang/embodied/generated_scenarios_merge_7"  # 新增：cfg 所在目录
        self.episode = 0
        self.episode = 0
        self.desSpeed = 12 # desired speed
        self.i = 0
        self.max_x = 100
        self.max_y = 100
        self.other_speed_max = 10
        self.ego_speed_max = 10
        self.last_road_index = '110'
        self.pid_lon = PID_posi(target=0,upper=2,lower=-2,k=[0.8,0,0])
        self.last_action = 0
        self.risk_evaluator = TrajectoryRiskEvaluator()
        self.risk_horizon = 1.5
        self.risk_dt = 0.1
        self.risk_range = 100.0
        self.risk_top_k = 6
        self.risk_cost_weight = 0.5
        self.last_risk_cost = 0.0
        self.danger_miner = StructuredDangerScenarioMiner(
            danger_config, danger_output_dir, worker_index=self.worker_index)
        self._episode_active = False
        self._episode_steps = 0
        self._episode_total_cost = 0.0
        self._episode_max_cost = 0.0
        self._episode_min_ttc = math.inf
        self._episode_terminal_reason = None
        self._episode_critical_state = None
        self._episode_critical_index = 0
        self._episode_replay_frames = []
        self._episode_variant = None
        self._episode_initial_data_frame = None
        self._episode_data_last_time = None
        self._episode_collected = False
        self._last_replay_error = {}
        self._playback_replay_frames = []
        self._playback_replay_index = 0
        self._playback_replay_active = False
        self._playback_replay_ended = False
        self._runtime_episode_steps = 0
        self._sumo_connection_lost = False
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'velocity':     spaces.Box(-np.inf, np.inf, (31,), dtype=np.float32),
            'is_first':     spaces.Box(0, 1, (), dtype=bool),
            'is_terminal':  spaces.Box(0, 1, (), dtype=bool),
        })
        self._brake_streak_s = {}
        self._brake_excluded = set()
        self._brake_random_exempt = set()
        self._brake_seen_npcs = set()
        self.npc_emergency_decel = 1.0
        self.npc_brake_exempt_ratio = 0.35
        self._low_speed_pair_streak_s = {}
        # 连续制动累计时长（秒）
        self._brake_streak_s = {}   # vid -> float

        # 已剔除车辆集合（以后不再制动）
        self._brake_excluded = set()


    def _get_features(self, egoid=None, Range=1000, last_state=None):
        '''
        Calculate features of surrounding vehicle together with self-features.
        args:
            egoid - feature of vehicle (str) ,defalut = None
            Range - perception range (int), default = 50009
         returen:
            Arry ,shape=(state_dim,)
        '''
        def get_dushu(x, y):
            point0 = [122.10, 326.37]   # todo 从此来得到正北方向。
            point1 = [120.83, 327.92]     #对规定正北方向
            A = point0
            B = point1
            C = point0
            D = [x, y]

            AB = (B[0] - A[0], B[1] - A[1])
            CD = (D[0] - C[0], D[1] - C[1])
            
            # 计算点积 
            dot_product = AB[0] * CD[0] + AB[1] * CD[1]
            
            # 计算向量的模
            length_AB = math.sqrt(AB[0]**2 + AB[1]**2)
            length_CD = math.sqrt(CD[0]**2 + CD[1]**2)
            
            # 计算夹角的余弦值
            cos_theta = dot_product / (length_AB * length_CD)
            
            # 确保 cos_theta 的值在 -1 和 1 之间
            cos_theta = max(-1, min(1, cos_theta))
            
            # 计算夹角（以弧度为单位）
            angle_rad = math.acos(cos_theta)
            
            # 使用叉积判断角度方向
            cross_product = AB[0] * CD[1] - AB[1] * CD[0]
            
            # 如果叉积为负，角度应该大于180度
            if cross_product < 0:
                angle_rad = 2 * math.pi - angle_rad
            
            # 将角度转换为度
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg
        
        if egoid == None:
            egoid = self.egoid
        try:
            ego_road = traci.vehicle.getRoadID(egoid)
            ego_lane = traci.vehicle.getLaneIndex(egoid)
            ego_speed = traci.vehicle.getSpeed(egoid)
            ego_angle = traci.vehicle.getAngle(egoid)
            ego_positionx = traci.vehicle.getPosition(egoid)[0]
            ego_positiony = traci.vehicle.getPosition(egoid)[1]
        except:
            ego_positionx = 0
            ego_positiony = 0
            ego_speed = 0
            ego_angle = 0
        self.x = ego_positionx
        self.y = ego_positiony
        self.jiaodu = get_dushu(ego_positionx, ego_positiony)
        # print('self.jiaodu',self.jiaodu)
        self.head = self.adjust_angle(ego_angle)
        # print('self.head',self.head)
        temp_data = pd.DataFrame(
            columns=['veh_id', 'positionx', 'positiony', 'velocity', 'road', 'lane', 'distance',
                     'priority','velocity_y','angle', 'jiaodu'])

        i = 0
        VehicleIds = traci.vehicle.getIDList()

        def calculate_angle(point1,point2):
            x1 = point1[0]
            y1 = point1[1]
            x2 = point2[0]
            y2 = point2[1]
            angle = math.atan2(y2 - y1, x2 - x1)
            
            return angle
        
        

        for vehId in VehicleIds:
            info = {}
            if vehId != egoid:  # 排除自动驾驶车辆id
                traci.vehicle.subscribe(
                    vehId, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ROAD_ID, tc.VAR_LANE_INDEX,
                            tc.VAR_SPEED_LAT, tc.VAR_ROUTE_ID, tc.VAR_ANGLE))
                info = traci.vehicle.getAllSubscriptionResults()[vehId]
                distance = math.sqrt((ego_positionx - info[tc.VAR_POSITION][0]) ** 2 + (
                                ego_positiony - info[tc.VAR_POSITION][1]) ** 2)
                if distance < 100:  # 只读取在range范围内车辆信息
                    # distance = math.sqrt((ego_positionx - info[tc.VAR_POSITION][0]) ** 2 + (
                    #             ego_positiony - info[tc.VAR_POSITION][1]) ** 2)
                    priority = False
                    temp_data.loc[i] = [
                        vehId , info[tc.VAR_POSITION][0] , info[tc.VAR_POSITION][1] , info[tc.VAR_SPEED] ,
                        info[tc.VAR_ROAD_ID] , info[tc.VAR_LANE_INDEX] , distance , priority ,
                        info[tc.VAR_SPEED_LAT], info[tc.VAR_ANGLE], get_dushu(info[tc.VAR_POSITION][0] , info[tc.VAR_POSITION][1]) - self.jiaodu]
                    i += 1

        library = temp_data.loc[temp_data['distance'] >= 0]
        same = temp_data.loc[~temp_data['road'].isin(['-E1', 'E2', '-E2', 'E3', '-E3', 'E4', '-E4'])]
        front_same = same.loc[same['jiaodu'] > 0]
        if not front_same.empty:
            front_same = front_same.loc[front_same['jiaodu'].idxmin()]
            self.front_same = pd.DataFrame([front_same])
        else:
            self.front_same = front_same

        x_1 , x_2 , x_3 , x_4 , x_5 , x_6 = (100 for i in range(6))
        y_1 , y_2 , y_3 , y_4 , y_5 , y_6 = (100 for i in range(6))
        vx_1 , vx_2 , vx_3 , vx_4 , vx_5 , vx_6 = (10 for i in range(6))
        vy_1 , vy_2 , vy_3 , vy_4 , vy_5 , vy_6 = (10 for i in range(6))
        angle_1, angle_2, angle_3, angle_4, angle_5, angle_6 = (0 for i in range(6))
        p_1 , p_2 , p_3 , p_4 , p_5, p_6 = (False for i in range(6))


        if not library.empty:
            if len(library['distance']) < 6:
                index = library['distance'].sort_values().index
            else:
                index = library['distance'].nsmallest(6).index

            n = len(index)
            if n >= 1:
                x_1 = library['positionx'][index[0]]
                y_1 = library['positiony'][index[0]]
                # vx_1 = library['velocity'][index[0]]
                # vy_1 = library['velocity_y'][index[0]]
                angle_1 = library['angle'][index[0]]
                p_1 = library['priority'][index[0]]

                vehicle_pos = self.transform(np.append([x_1,y_1], [0]))[:2]
                x_1 = vehicle_pos[0]
                y_1 = vehicle_pos[1]
                head = self.adjust_angle(angle_1)
                angle_1 = self.adjust_heading(head)
                vx_1 = library['velocity'][index[0]]*np.cos(self.adjust_heading(head))
                vy_1 = library['velocity'][index[0]]*np.sin(self.adjust_heading(head))
            if n >= 2:
                x_2 = library['positionx'][index[1]]
                y_2 = library['positiony'][index[1]]
                # vx_2 = library['velocity'][index[1]]
                # vy_2 = library['velocity_y'][index[1]]
                angle_2 = library['angle'][index[1]]
                p_2 = library['priority'][index[1]]
                # print('转换前的旁车2的坐标为:',x_2,y_2)
                # print('转换前的自车的坐标为:',self.x,self.y)
                # print('转换前的自车的航向角为:',self.head)
                vehicle_pos = self.transform(np.append([x_2,y_2], [0]))[:2]
                x_2 = vehicle_pos[0]
                y_2 = vehicle_pos[1]
                # print('自车坐标系下的旁车位置信息：:',x_2,y_2)
                head = self.adjust_angle(angle_2)
                angle_2 = self.adjust_heading(head)
                vx_2 = library['velocity'][index[1]]*np.cos(self.adjust_heading(head))
                vy_2 = library['velocity'][index[1]]*np.sin(self.adjust_heading(head))
            if n >= 3:
                x_3 = library['positionx'][index[2]]
                y_3 = library['positiony'][index[2]]
                # vx_3 = library['velocity'][index[2]]
                # vy_3 = library['velocity_y'][index[2]]
                angle_3 = library['angle'][index[2]]
                p_3 = library['priority'][index[2]]

                vehicle_pos = self.transform(np.append([x_3,y_3], [0]))[:2]
                x_3 = vehicle_pos[0]
                y_3 = vehicle_pos[1]
                head = self.adjust_angle(angle_3)
                angle_3 = self.adjust_heading(head)
                vx_3 = library['velocity'][index[2]]*np.cos(self.adjust_heading(head))
                vy_3 = library['velocity'][index[2]]*np.sin(self.adjust_heading(head))
            if n >= 4:
                x_4 = library['positionx'][index[3]]
                y_4 = library['positiony'][index[3]]
                # vx_4 = library['velocity'][index[3]]
                # vy_4 = library['velocity_y'][index[3]]
                angle_4 = library['angle'][index[3]]
                p_4 = library['priority'][index[3]]

                vehicle_pos = self.transform(np.append([x_4,y_4], [0]))[:2]
                x_4 = vehicle_pos[0]
                y_4 = vehicle_pos[1]
                head = self.adjust_angle(angle_4)
                angle_4 = self.adjust_heading(head)
                vx_4 = library['velocity'][index[3]]*np.cos(self.adjust_heading(head))
                vy_4 = library['velocity'][index[3]]*np.sin(self.adjust_heading(head))
            if n >= 5:
                x_5 = library['positionx'][index[4]]
                y_5 = library['positiony'][index[4]]
                # vx_5 = library['velocity'][index[4]]
                # vy_5 = library['velocity_y'][index[4]]
                angle_5 = library['angle'][index[4]]
                p_5 = library['priority'][index[4]]

                vehicle_pos = self.transform(np.append([x_5,y_5], [0]))[:2]
                x_5 = vehicle_pos[0]
                y_5 = vehicle_pos[1]
                head = self.adjust_angle(angle_5)
                angle_5 = self.adjust_heading(head)
                vx_5 = library['velocity'][index[4]]*np.cos(self.adjust_heading(head))
                vy_5 = library['velocity'][index[4]]*np.sin(self.adjust_heading(head))
            if n >= 6:
                x_6 = library['positionx'][index[5]]
                y_6 = library['positiony'][index[5]]
                # vx_6 = library['velocity'][index[5]]
                # vy_6 = library['velocity_y'][index[5]]
                angle_6 = library['angle'][index[5]]
                p_6 = library['priority'][index[5]]

                vehicle_pos = self.transform(np.append([x_6,y_6], [0]))[:2]
                x_6 = vehicle_pos[0]
                y_6 = vehicle_pos[1]
                head = self.adjust_angle(angle_6)
                angle_6 = self.adjust_heading(head)
                vx_6 = library['velocity'][index[5]]*np.cos(self.adjust_heading(head))
                vy_6 = library['velocity'][index[5]]*np.sin(self.adjust_heading(head))
        state = [ego_speed/self.ego_speed_max,x_1/self.max_x, y_1/self.max_y, vx_1/self.other_speed_max, vy_1/self.other_speed_max, angle_1/math.pi, 
                 x_2/self.max_x, y_2/self.max_y, vx_2/self.other_speed_max, vy_2/self.other_speed_max, angle_2/math.pi,
                 x_3/self.max_x, y_3/self.max_y, vx_3/self.other_speed_max, vy_3/self.other_speed_max, angle_3/math.pi,
                 x_4/self.max_x, y_4/self.max_y, vx_4/self.other_speed_max, vy_4/self.other_speed_max, angle_4/math.pi,
                 x_5/self.max_x, y_5/self.max_y, vx_5/self.other_speed_max, vy_5/self.other_speed_max, angle_5/math.pi,
                 x_6/self.max_x, y_6/self.max_y, vx_6/self.other_speed_max, vy_6/self.other_speed_max, angle_6/math.pi,]

        state = np.array(state)

        # if self.episode >2000:
        #     noise = np.zeros_like(state)

        #     # 结构： [ego_v, 车1(x,y,vx,vy,angle), 车2(...), ...]
        #     # 你可以按索引段来分配噪声
        #     ego_v_idx = 0
        #     other_start = 1            # 后面是 6 辆车，每辆 5 维

        #     # 自车速度
        #     noise[ego_v_idx] = np.random.normal(0, 0.01)

        #     # 每辆车：x, y, vx, vy, angle
        #     for k in range(6):
        #         base = other_start + k * 5
        #         # 位置噪声大一点
        #         noise[base + 0] = np.random.normal(0, 0.02)  # x
        #         noise[base + 1] = np.random.normal(0, 0.02)  # y
        #         # 速度噪声中等
        #         noise[base + 2] = np.random.normal(0, 0.015) # vx
        #         noise[base + 3] = np.random.normal(0, 0.015) # vy
        #         # 角度（已经 /pi），噪声小一点
        #         noise[base + 4] = np.random.normal(0, 0.01)  # angle/pi

        #     state = state + noise

        return state
    


    def transform(self, v):
        return position_to_ego_frame(v, [self.x, self.y, 0], self.head)

    def adjust_heading(self, h):
        return wrap_value(h - self.head, -math.pi, math.pi)
    
    def adjust_angle(self, angle):
        angle -= 90
        if angle != 180 and angle != 0:
            angle = angle * -1
        return (angle*math.pi)/180




    def _to_obs(self, state):
        # image: dummy array
        # image = np.zeros((64, 64, 3), dtype=np.uint8)
        # velocity: flatten entire state vector
        velocity = np.array(state, dtype=np.float32)
        return {
            # 'image':    image,
            'velocity': velocity,
        }

    def _get_vehicle_snapshot(self, veh_id):
        x, y = traci.vehicle.getPosition(veh_id)
        speed = float(traci.vehicle.getSpeed(veh_id))
        accel = float(traci.vehicle.getAcceleration(veh_id))
        raw_angle = float(traci.vehicle.getAngle(veh_id))
        yaw = self.adjust_angle(raw_angle)
        length = float(traci.vehicle.getLength(veh_id))
        width = float(traci.vehicle.getWidth(veh_id))
        try:
            road_id = str(traci.vehicle.getRoadID(veh_id))
        except Exception:
            road_id = ""
        try:
            lane_id = str(traci.vehicle.getLaneID(veh_id))
        except Exception:
            lane_id = ""
        try:
            lane_index = int(traci.vehicle.getLaneIndex(veh_id))
        except Exception:
            lane_index = -1
        try:
            route_id = str(traci.vehicle.getRouteID(veh_id))
        except Exception:
            route_id = ""
        try:
            route_edges = [str(edge) for edge in traci.vehicle.getRoute(veh_id)]
        except Exception:
            route_edges = []
        try:
            type_id = str(traci.vehicle.getTypeID(veh_id))
        except Exception:
            type_id = ""
        try:
            lane_position = float(traci.vehicle.getLanePosition(veh_id))
        except Exception:
            lane_position = 0.0
        return {
            "id": str(veh_id),
            "type": "vehicle",
            "type_id": type_id,
            "pos": (float(x), float(y)),
            "v": speed,
            "a": accel,
            "yaw": yaw,
            "angle": raw_angle,
            "road_id": road_id,
            "lane_id": lane_id,
            "lane_index": lane_index,
            "lane_position": lane_position,
            "route_id": route_id,
            "route_edges": route_edges,
            "length": length,
            "width": width,
        }

    def _get_risk_candidate_vehicles(self, egoid=None):
        if egoid is None:
            egoid = self.egoid
        try:
            ego_positionx, ego_positiony = traci.vehicle.getPosition(egoid)
        except Exception:
            return []

        temp_data = pd.DataFrame(
            columns=['veh_id', 'positionx', 'positiony', 'velocity', 'road', 'lane', 'distance',
                     'priority', 'velocity_y', 'angle', 'jiaodu'])
        i = 0
        for veh_id in traci.vehicle.getIDList():
            if veh_id == egoid:
                continue
            try:
                traci.vehicle.subscribe(
                    veh_id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ROAD_ID, tc.VAR_LANE_INDEX,
                             tc.VAR_SPEED_LAT, tc.VAR_ROUTE_ID, tc.VAR_ANGLE))
                info = traci.vehicle.getAllSubscriptionResults()[veh_id]
                distance = math.sqrt((ego_positionx - info[tc.VAR_POSITION][0]) ** 2 +
                                     (ego_positiony - info[tc.VAR_POSITION][1]) ** 2)
            except Exception:
                continue

            if distance < self.risk_range:
                temp_data.loc[i] = [
                    veh_id, info[tc.VAR_POSITION][0], info[tc.VAR_POSITION][1], info[tc.VAR_SPEED],
                    info[tc.VAR_ROAD_ID], info[tc.VAR_LANE_INDEX], distance, False,
                    info[tc.VAR_SPEED_LAT], info[tc.VAR_ANGLE], 0.0
                ]
                i += 1

        library = temp_data.loc[temp_data['distance'] >= 0]
        if library.empty:
            return []
        if len(library['distance']) < self.risk_top_k:
            index = library['distance'].sort_values().index
        else:
            index = library['distance'].nsmallest(self.risk_top_k).index
        return [str(library.loc[idx, 'veh_id']) for idx in index]

    def _build_ego_traj_for_risk(self, ego_snapshot, ego_acc):
        steps = int(round(self.risk_horizon / self.risk_dt))
        x0, y0 = ego_snapshot["pos"]
        yaw = float(ego_snapshot["yaw"])
        v0 = float(ego_snapshot["v"])
        pos_list = np.zeros((steps, 2), dtype=float)
        v_list = np.zeros((steps,), dtype=float)
        for i in range(steps):
            t = (i + 1) * self.risk_dt
            s = v0 * t + 0.5 * float(ego_acc) * t * t
            v_t = max(0.0, v0 + float(ego_acc) * t)
            pos_list[i, 0] = x0 + s * math.cos(yaw)
            pos_list[i, 1] = y0 + s * math.sin(yaw)
            v_list[i] = v_t
        return {
            "id": ego_snapshot["id"],
            "type": "vehicle",
            "pos_list": pos_list,
            "v_list": v_list,
            "orientation_list": np.full((steps,), yaw, dtype=float),
            "shape": {"length": float(ego_snapshot["length"]), "width": float(ego_snapshot["width"])},
            "dt": float(self.risk_dt),
        }

    def _build_obj_predictions_for_risk(self):
        obs_trajectory = []
        for veh_id in self._get_risk_candidate_vehicles(self.egoid):
            try:
                obs_trajectory.append(self._get_vehicle_snapshot(veh_id))
            except Exception:
                continue
        return predictor_future_only(
            obs_trajectory,
            self.risk_horizon,
            dt=self.risk_dt,
            vmax_by_type={"vehicle": 30.0, "bicycle": 8.0, "pedestrian": 5.0},
        )

    def _compute_risk_cost(self, ego_acc):
        try:
            ego_snapshot = self._get_vehicle_snapshot(self.egoid)
        except Exception:
            return 0.0
        ego_traj = self._build_ego_traj_for_risk(ego_snapshot, ego_acc)
        obj_predictions = self._build_obj_predictions_for_risk()
        if not obj_predictions:
            return 0.0
        return float(self.risk_evaluator.total_costs(ego_traj, obj_predictions))

    # ------------------------------------------------------------------
    # World-Engine-style mining and structured-scenario generation.
    # ------------------------------------------------------------------
    def _begin_episode_stats(self, variant):
        self._episode_active = (
            self.danger_miner.recording or self.danger_miner.collecting)
        self._episode_steps = 0
        self._episode_total_cost = 0.0
        self._episode_max_cost = 0.0
        self._episode_min_ttc = math.inf
        self._episode_terminal_reason = None
        self._episode_critical_state = None
        self._episode_critical_index = 0
        self._episode_replay_frames = []
        self._episode_variant = variant
        self._episode_initial_data_frame = None
        self._episode_data_last_time = None
        self._episode_collected = False

    def _capture_initial_episode_frame(self, state):
        """Keep reset truth for action alignment in the independent dataset."""
        if not self.danger_miner.collecting:
            return
        frame = self._capture_replay_frame(state, 0.0, math.inf)
        frame['simulation_advanced'] = False
        frame['is_initial'] = True
        self._episode_initial_data_frame = frame
        self._episode_data_last_time = frame['sim_time']

    def _capture_replay_frame(
            self, state, cost, ttc, terminal_reason=None, action=None, acc=None):
        vehicles = []
        if not self._sumo_connection_lost:
            try:
                vehicle_ids = traci.vehicle.getIDList()
            except Exception as exc:
                if type(exc).__name__ == "FatalTraCIError":
                    self._mark_sumo_connection_lost(
                        "capture_replay_frame", exc)
                vehicle_ids = []
            for veh_id in vehicle_ids:
                try:
                    vehicles.append(self._get_vehicle_snapshot(veh_id))
                except Exception:
                    continue
        time_available = True
        try:
            sim_time = float(traci.simulation.getTime())
        except Exception:
            sim_time = 0.0
            time_available = False
        frame = {
            "step": int(self._episode_steps),
            "sim_time": sim_time,
            "state": np.asarray(state, dtype=np.float32).reshape(-1).tolist(),
            "cost": float(cost),
            "ttc": float(ttc) if math.isfinite(float(ttc)) else None,
            "terminal_reason": terminal_reason or "",
            "vehicles": vehicles,
        }
        if self.danger_miner.collecting:
            frame['capture_failed'] = (
                not time_available or self._sumo_connection_lost)
        if action is not None:
            frame["action"] = np.asarray(action, dtype=np.float32).reshape(-1).tolist()
        if acc is not None:
            frame["acc"] = float(np.asarray(acc).reshape(()))
        return frame

    def _build_replay_trajectory(self):
        frames = list(self._episode_replay_frames)
        if not frames:
            return {}
        critical_index = int(np.clip(
            self._episode_critical_index, 0, max(len(frames) - 1, 0)))
        max_steps = max(
            1, int(self.danger_miner.config.get("max_replay_steps", len(frames))))
        if len(frames) > max_steps:
            start = min(
                max(0, critical_index - max_steps + 1),
                len(frames) - max_steps)
            frames = frames[start:start + max_steps]
            critical_index -= start
        background_counts = [
            sum(
                1 for snapshot in (frame.get("vehicles") or [])
                if str(snapshot.get("id", "")) != self.egoid)
            for frame in frames]
        return {
            "version": 1,
            "source_cfg": str(self.path),
            "critical_index": int(critical_index),
            "background_vehicle_frames": int(
                sum(1 for count in background_counts if count > 0)),
            "max_background_vehicles": int(max(background_counts or [0])),
            "frames": frames,
        }

    def _record_episode_step(
            self, state, cost, terminal_reason=None, action=None, acc=None):
        if not self._episode_active:
            return
        self._episode_steps += 1
        cost = float(np.asarray(cost).reshape(()))
        self._episode_total_cost += cost
        self._episode_max_cost = max(self._episode_max_cost, cost)
        try:
            ttc = float(self.ttc_compute(state)[0])
        except Exception:
            ttc = math.inf
        is_more_critical = (
            cost >= self._episode_max_cost or
            ttc <= self._episode_min_ttc or
            terminal_reason == 'collision')
        self._episode_min_ttc = min(self._episode_min_ttc, ttc)
        frame = self._capture_replay_frame(
            state, cost, ttc, terminal_reason=terminal_reason,
            action=action, acc=acc)
        if self.danger_miner.collecting:
            last_time = self._episode_data_last_time
            frame['simulation_advanced'] = (
                last_time is None or frame['sim_time'] > last_time + 1e-9)
            self._episode_data_last_time = frame['sim_time']
        self._episode_replay_frames.append(frame)
        if is_more_critical:
            self._episode_critical_state = np.asarray(state, dtype=np.float32).copy()
            self._episode_critical_index = len(self._episode_replay_frames) - 1
        if terminal_reason:
            self._episode_terminal_reason = str(terminal_reason)

    def _finish_active_episode(self):
        if not self._episode_active:
            return
        if self.danger_miner.collecting and not self._episode_collected:
            self._write_collected_episode()
            self._episode_collected = True
        mined = self.danger_miner.record_episode(
            source_cfg=self.path,
            length=self._episode_steps,
            total_cost=self._episode_total_cost,
            max_cost=self._episode_max_cost,
            min_ttc=self._episode_min_ttc,
            terminal_reason=self._episode_terminal_reason,
            critical_state=self._episode_critical_state,
            replay_trajectory=self._build_replay_trajectory(),
            generated_variant=self._episode_variant)
        if mined:
            print(
                '[danger mining] '
                f"type={mined['event_type']} severity={mined['severity']:.2f} "
                f"library={len(self.danger_miner.library)}")
        self._episode_active = False

    def _write_collected_episode(self):
        """Export actual execution separately from the frozen prototype library."""
        import hashlib
        import json
        from pathlib import Path
        from .scenario_data import episode_from_record, write_episode

        output_dir = self.danger_miner.output_dir
        if not output_dir:
            print('[episode collection skipped] danger_output_dir is not configured', flush=True)
            return
        frames = list(self._episode_replay_frames)
        if self._episode_initial_data_frame is not None:
            frames.insert(0, self._episode_initial_data_frame)
        if len(frames) < 2:
            return
        terminal_reason = self._episode_terminal_reason or ''
        dropped_connection_frame = False
        dropped_capture_frames = []
        capture_failed = any(frame.get('capture_failed', False) for frame in frames)
        if capture_failed:
            first_failed = next(
                index for index, frame in enumerate(frames)
                if frame.get('capture_failed', False))
            dropped_capture_frames = frames[first_failed:]
            frames = frames[:first_failed]
            dropped_connection_frame = True
        # A disconnected TraCI session may report fallback time=0. Preserve
        # the last known real prefix, rather than inventing a terminal pose.
        if terminal_reason == 'sumo_closed' or self._sumo_connection_lost:
            while len(frames) > 1 and (
                    frames[-1]['sim_time'] <= frames[-2]['sim_time']):
                dropped_capture_frames.insert(0, frames.pop())
                dropped_connection_frame = True
        if len(frames) < 2:
            print('[episode collection skipped] fewer than two valid time samples', flush=True)
            return
        variant = self._episode_variant or {}
        mode = variant.get('used_danger_mode', variant.get('mode', variant.get('kind', 'off')))
        source_type = {
            'scenario_copy': 'sumo_copy', 'copy': 'sumo_copy',
            'intent_attack': 'sumo_attack', 'attack': 'sumo_attack',
            'replay': 'sumo_replay',
        }.get(mode, 'sumo_original')
        source_name = str(self.path).replace('\\', '/').rsplit('/', 1)[-1]
        parent_id = 'sumo:' + source_name
        # Collection labels must remain available when legacy event logging is
        # frozen; the miner's classifier deliberately gates on recording.
        if terminal_reason == 'collision':
            event_type = 'collision'
        elif terminal_reason in ('max_time', 'max_steps'):
            event_type = 'low_progress'
        elif self._episode_min_ttc < float(self.danger_miner.config['near_miss_ttc']):
            event_type = 'near_miss'
        elif self._episode_max_cost >= float(self.danger_miner.config['risk_cost_threshold']):
            event_type = 'high_risk_cost'
        else:
            event_type = 'safe'
        if terminal_reason == 'sumo_closed' or capture_failed or self._sumo_connection_lost:
            event_type = 'execution_error'
        record = {
            'event': 'episode', 'worker': self.worker_index,
            'episode': self.episode, 'scenario_id': parent_id,
            'parent_scenario_id': parent_id, 'source_cfg': str(self.path),
            'map_id': None, 'seed': None, 'source_type': source_type,
            'length': self._episode_steps,
            'total_cost': self._episode_total_cost,
            'max_cost': self._episode_max_cost,
            'min_ttc': self._episode_min_ttc,
            'terminal_reason': terminal_reason, 'event_type': event_type,
            'generated_variant': variant,
            'metadata': {
                'phase': 'posttrain', 'env_mode': self.env_mode,
                'ego_id': self.egoid, 'prediction_source': 'sumo_execution',
                'execution_valid': None, 'model_version': None,
                'simulator_seed': None,
                'scenario_sampling_seed': self.danger_miner.config.get('seed'),
                'action_alignment': 'transition_ending_at_frame',
                'state_alignment': 'legacy_observation; terminal_state_may_precede_snapshot',
                'generated_variant': variant,
                'terminal_reason': terminal_reason,
                'event_type': event_type,
                'episode_metrics': {
                    'steps': self._episode_steps,
                    'total_cost': self._episode_total_cost,
                    'max_cost': self._episode_max_cost,
                    'min_ttc': self._episode_min_ttc,
                },
                'episode_label_thresholds': {
                    'near_miss_ttc': self.danger_miner.config['near_miss_ttc'],
                    'risk_cost_threshold': self.danger_miner.config['risk_cost_threshold'],
                },
                'collection_complete': event_type != 'execution_error',
                'dropped_connection_terminal_frame': dropped_connection_frame,
                'dropped_capture_frames': dropped_capture_frames,
            },
            'replay_trajectory': {
                'version': 1, 'source_cfg': str(self.path),
                'critical_index': self._episode_critical_index + int(
                    self._episode_initial_data_frame is not None),
                'frames': frames,
            },
        }
        record = self.danger_miner._json_safe(record)
        record['episode_id'] = hashlib.sha256(
            json.dumps(record, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:24]
        if variant:
            record['scenario_id'] = 'sumo_variant:' + record['episode_id']
        try:
            episode = episode_from_record(record, ego_id=self.egoid)
            path = write_episode(episode, Path(output_dir).parent / 'posttrain_episode_pool')
        except (OSError, ValueError) as exc:
            print(
                f'[episode collection failed] worker={self.worker_index} '
                f'episode={self.episode} id={record["episode_id"]} error={exc}',
                flush=True)
            return
        print(f'[episode collection] worker={self.worker_index} episode={self.episode} path={path}')

    def _sumo_command(self, path):
        sumo_binary = checkBinary('sumo-gui') if self._sumo_gui else checkBinary('sumo')
        command = [sumo_binary, '-c', path]
        if self._sumo_gui:
            command += ['--start', '--quit-on-end']
        return command

    def _restart_sumo(self, path):
        try:
            traci.close()
        except Exception:
            pass
        traci.start(self._sumo_command(path))
        self._sumo_started = True
        self._sumo_connection_lost = False

    def _load_scene(self, path):
        """Load one SUMO config and restore the control modes used by training."""
        last_error = None
        for attempt in range(2):
            try:
                if not self._sumo_started:
                    self._restart_sumo(path)
                else:
                    traci.load(['-c', path])
                traci.simulationStep()
                for veh_id in traci.vehicle.getIDList():
                    traci.vehicle.setLaneChangeMode(veh_id, 0)
                    traci.vehicle.setSpeedMode(veh_id, 0)
                self._sumo_connection_lost = False
                return True
            except Exception as exc:
                last_error = exc
                self._sumo_started = False
                print(
                    '[sumo load retry] '
                    f'worker={self.worker_index} path={os.path.basename(str(path))} '
                    f'attempt={attempt + 1} error={type(exc).__name__}: {exc}')
                try:
                    traci.close()
                except Exception:
                    pass
        print(
            '[sumo load failed] '
            f'worker={self.worker_index} path={path} error={last_error}')
        return False

    def _candidate_scene_paths(self):
        if not os.path.isdir(self.cfg_dir):
            return []
        paths = [
            os.path.join(self.cfg_dir, name)
            for name in os.listdir(self.cfg_dir)
            if name.lower().endswith('.sumocfg')]
        random.shuffle(paths)
        return paths

    def _load_scene_with_fallbacks(self, preferred_path, fallback_path):
        candidates = []
        for path in (preferred_path, fallback_path):
            if path and path not in candidates:
                candidates.append(path)
        for path in self._candidate_scene_paths():
            if path not in candidates:
                candidates.append(path)
            if len(candidates) >= 6:
                break
        for path in candidates:
            if self._load_scene(path):
                self.path = path
                return True
        return False

    def _legacy_apply_replay_frame(self, frame):
        vehicles = frame.get("vehicles") or []
        if not vehicles:
            return False
        restored = 0
        current_ids = set(traci.vehicle.getIDList())
        for snapshot in vehicles:
            veh_id = str(snapshot.get("id", ""))
            if not veh_id or veh_id not in current_ids:
                continue
            try:
                x, y = snapshot.get("pos", (None, None))
                if x is None or y is None:
                    continue
                edge_id = str(snapshot.get("road_id", ""))
                lane_index = int(snapshot.get("lane_index", -1))
                angle = float(snapshot.get("angle", traci.vehicle.getAngle(veh_id)))
                traci.vehicle.moveToXY(
                    veh_id, edge_id, lane_index, float(x), float(y), angle, 2)
                traci.vehicle.setSpeed(veh_id, max(0.0, float(snapshot.get("v", 0.0))))
                traci.vehicle.setLaneChangeMode(veh_id, 0)
                traci.vehicle.setSpeedMode(veh_id, 0)
                restored += 1
            except Exception:
                continue
        return restored > 0 and self.egoid in traci.vehicle.getIDList()

    def _legacy_apply_replay_trajectory(self, trajectory):
        frames = (trajectory or {}).get("frames") or []
        if not frames:
            return False
        critical_index = int(np.clip(
            int(trajectory.get("critical_index", len(frames) - 1)),
            0, len(frames) - 1))
        pre_margin = max(
            0, int(self.danger_miner.config.get("replay_pre_margin", 3)))
        frame_index = max(0, critical_index - pre_margin)
        frame = frames[frame_index]
        target_time = frame.get("sim_time")
        if target_time is not None:
            max_seek_steps = max(
                0, int(self.danger_miner.config.get("max_replay_seek_steps", 400)))
            seek_steps = 0
            while (
                    traci.simulation.getTime() + 1e-6 < float(target_time) and
                    seek_steps < max_seek_steps):
                # traci.simulationStep()
                seek_steps += 1
        return self._apply_replay_frame(frame)

    def _set_replay_error(self, reason, **info):
        self._last_replay_error = {"reason": str(reason)}
        self._last_replay_error.update(info)

    def _format_replay_error(self):
        if not self._last_replay_error:
            return ""
        parts = [
            f"{key}={value}"
            for key, value in self._last_replay_error.items()
            if key != "reason"]
        return " ".join(parts)

    def _route_exists(self, route_id):
        if not route_id:
            return False
        try:
            return route_id in set(traci.route.getIDList())
        except Exception:
            return False

    def _replay_context(self):
        return (
            f"worker={self.worker_index} pid={os.getpid()} "
            f"episode={self.episode} env_step={self._runtime_episode_steps} "
            f"source={os.path.basename(str(self.path))}")

    def _should_log_replay_detail(self, frame_index=None):
        interval = int(self.danger_miner.config.get(
            "replay_debug_interval", 0))
        if interval <= 0:
            return False
        if frame_index is None:
            current = int(self._playback_replay_index)
        else:
            current = int(frame_index) + 1
        return current <= 3 or current % interval == 0

    def _log_replay_detail(self, event, **fields):
        parts = [f"[danger replay {event}]", self._replay_context()]
        parts.extend(f"{key}={value}" for key, value in fields.items())
        print(" ".join(parts), flush=True)

    def _rgba_config(self, key, default):
        color = self.danger_miner.config.get(key, default)
        try:
            values = [int(np.clip(int(component), 0, 255)) for component in color]
        except Exception:
            values = list(default)
        if len(values) == 3:
            values.append(255)
        if len(values) != 4:
            values = list(default)
        return tuple(values)

    def _mark_sumo_connection_lost(self, where, exc):
        self._sumo_connection_lost = True
        self._sumo_started = False
        self._playback_replay_active = False
        self._set_replay_error(
            "sumo_connection_lost", where=where,
            error=type(exc).__name__)
        self._log_replay_detail(
            "connection-lost", where=where,
            error=type(exc).__name__)

    def _ensure_replay_route(self, snapshot):
        route_id = str(snapshot.get("route_id", "") or "")
        route_edges = [str(edge) for edge in (snapshot.get("route_edges") or [])]
        if route_id and self._route_exists(route_id):
            return route_id
        if not route_edges:
            road_id = str(snapshot.get("road_id", "") or "")
            if road_id and not road_id.startswith(":"):
                route_edges = [road_id]
        if not route_edges:
            return route_id if route_id and self._route_exists(route_id) else ""
        if not route_id:
            route_id = f"replay_route_{snapshot.get('id', 'vehicle')}"
        try:
            traci.route.add(route_id, route_edges)
        except Exception as exc:
            if self._playback_replay_active:
                self._log_replay_detail(
                    "route-add-failed", route=route_id,
                    error=type(exc).__name__)
        return route_id if self._route_exists(route_id) else ""

    def _insert_replay_vehicle(self, snapshot):
        if not bool(self.danger_miner.config.get("replay_insert_missing", True)):
            return False
        veh_id = str(snapshot.get("id", "") or "")
        if not veh_id:
            return False
        try:
            if veh_id in traci.vehicle.getIDList():
                return True
        except Exception as exc:
            if self._playback_replay_active:
                self._log_replay_detail(
                    "insert-id-check-failed", vehicle=veh_id,
                    error=type(exc).__name__)
            return False
        route_id = self._ensure_replay_route(snapshot)
        if not route_id:
            if self._playback_replay_active:
                self._log_replay_detail(
                    "insert-no-route", vehicle=veh_id)
            return False
        type_id = str(snapshot.get("type_id", "") or "DEFAULT_VEHTYPE")
        speed = max(0.0, float(snapshot.get("v", 0.0)))
        last_error = None
        for depart_lane, depart_pos in (
                ("free", "random_free"),
                ("first", "base"),
                ("0", "0")):
            try:
                traci.vehicle.add(
                    veh_id, routeID=route_id, typeID=type_id,
                    depart="now", departLane=depart_lane,
                    departPos=depart_pos, departSpeed=str(speed))
                return True
            except Exception as exc:
                last_error = exc
                continue
        if self._playback_replay_active:
            self._log_replay_detail(
                "insert-failed", vehicle=veh_id, route=route_id,
                error=type(last_error).__name__ if last_error else "unknown")
        return False

    def _move_replay_vehicle(self, snapshot):
        veh_id = str(snapshot.get("id", "") or "")
        if not veh_id:
            return False
        try:
            if veh_id not in traci.vehicle.getIDList():
                if self._playback_replay_active:
                    self._log_replay_detail(
                        "move-missing-vehicle", vehicle=veh_id)
                return False
            x, y = snapshot.get("pos", (None, None))
            if x is None or y is None:
                if self._playback_replay_active:
                    self._log_replay_detail(
                        "move-missing-position", vehicle=veh_id)
                return False
            angle = float(snapshot.get("angle", traci.vehicle.getAngle(veh_id)))
        except Exception as exc:
            if self._playback_replay_active:
                self._log_replay_detail(
                    "move-read-failed", vehicle=veh_id,
                    error=type(exc).__name__)
            return False

        road_id = str(snapshot.get("road_id", "") or "")
        lane_index = int(snapshot.get("lane_index", -1))
        lane_id = str(snapshot.get("lane_id", "") or "")
        lane_edge = lane_id.rsplit("_", 1)[0] if "_" in lane_id else ""
        candidates = []
        last_error = None
        for edge_id, lane in (
                (road_id, lane_index),
                (lane_edge, lane_index),
                ("", -1)):
            key = (edge_id, lane)
            if key not in candidates:
                candidates.append(key)
        for edge_id, lane in candidates:
            try:
                traci.vehicle.moveToXY(
                    veh_id, edge_id, lane, float(x), float(y),
                    angle, 2, matchThreshold=1000)
                traci.vehicle.setSpeed(
                    veh_id, max(0.0, float(snapshot.get("v", 0.0))))
                traci.vehicle.setLaneChangeMode(veh_id, 0)
                traci.vehicle.setSpeedMode(veh_id, 0)
                return True
            except Exception as exc:
                last_error = exc
                continue
        if self._playback_replay_active:
            self._log_replay_detail(
                "move-failed", vehicle=veh_id, road=road_id,
                lane=lane_index,
                error=type(last_error).__name__ if last_error else "unknown")
        return False

    def _replay_target_snapshots(self, frame, include_ego):
        vehicles = frame.get("vehicles") or []
        if include_ego:
            return vehicles
        return [
            snapshot for snapshot in vehicles
            if str(snapshot.get("id", "")) != self.egoid]

    def _replay_has_targets(self, frames, include_ego):
        return any(
            self._replay_target_snapshots(frame, include_ego)
            for frame in frames)

    def _apply_replay_frame(self, frame, frame_index=None, include_ego=True):
        vehicles = frame.get("vehicles") or []
        log_detail = (
            self._playback_replay_active and
            self._should_log_replay_detail(frame_index))
        if not vehicles:
            self._set_replay_error("empty_frame", frame=frame_index)
            return False
        targets = self._replay_target_snapshots(frame, include_ego)
        if log_detail:
            self._log_replay_detail(
                "frame-begin", frame=frame_index,
                vehicles=len(vehicles), targets=len(targets),
                replay_ego=int(bool(include_ego)))
        if not targets:
            self._set_replay_error(
                "no_replay_targets_in_frame", frame=frame_index,
                vehicles=len(vehicles), replay_ego=int(bool(include_ego)))
            return True
        ego_snapshot = None
        for snapshot in vehicles:
            if str(snapshot.get("id", "")) == self.egoid:
                ego_snapshot = snapshot
                break
        if include_ego and ego_snapshot is None:
            self._set_replay_error(
                "ego_missing_in_frame", frame=frame_index,
                vehicles=len(vehicles))
            return False

        inserted = 0
        skipped_missing = 0
        inserted_not_ready = 0
        movable_ids = set()
        insert_during_playback = bool(self.danger_miner.config.get(
            "replay_insert_during_playback", False))
        for snapshot in targets:
            veh_id = str(snapshot.get("id", "") or "")
            if not veh_id:
                continue
            try:
                exists = veh_id in traci.vehicle.getIDList()
            except Exception:
                exists = False
            if not exists:
                if self._playback_replay_active and not insert_during_playback:
                    skipped_missing += 1
                    if log_detail:
                        self._log_replay_detail(
                            "insert-skip-playback", frame=frame_index,
                            vehicle=veh_id)
                    continue
                if self._insert_replay_vehicle(snapshot):
                    inserted += 1
                    try:
                        inserted_ready = veh_id in traci.vehicle.getIDList()
                    except Exception:
                        inserted_ready = False
                    if inserted_ready:
                        movable_ids.add(veh_id)
                    else:
                        inserted_not_ready += 1
                else:
                    skipped_missing += 1
            else:
                movable_ids.add(veh_id)

        if inserted and log_detail:
            self._log_replay_detail(
                "insert-no-extra-step", frame=frame_index,
                inserted=inserted)

        restored = 0
        ego_restored = False
        for snapshot in targets:
            veh_id = str(snapshot.get("id", "") or "")
            if veh_id not in movable_ids:
                continue
            if not self._move_replay_vehicle(snapshot):
                continue
            self._highlight_replay_vehicle(veh_id)
            restored += 1
            if str(snapshot.get("id", "")) == self.egoid:
                ego_restored = True
        if restored <= 0:
            self._set_replay_error(
                "no_replay_targets_restored", frame=frame_index,
                vehicles=len(vehicles), targets=len(targets),
                inserted=inserted, skipped_missing=skipped_missing,
                inserted_not_ready=inserted_not_ready,
                restored=restored, replay_ego=int(bool(include_ego)))
            if log_detail:
                self._log_replay_detail(
                    "frame-end", frame=frame_index, inserted=inserted,
                    skipped_missing=skipped_missing,
                    inserted_not_ready=inserted_not_ready,
                    restored=restored)
            return False
        if include_ego and not ego_restored:
            try:
                present = self.egoid in traci.vehicle.getIDList()
            except Exception:
                present = False
            self._set_replay_error(
                "ego_restore_failed", frame=frame_index,
                vehicles=len(vehicles), inserted=inserted,
                inserted_not_ready=inserted_not_ready,
                restored=restored, ego_present=int(bool(present)))
            return False
        self._set_replay_error(
            "ok", frame=frame_index, vehicles=len(vehicles),
            targets=len(targets), inserted=inserted,
            skipped_missing=skipped_missing, restored=restored,
            inserted_not_ready=inserted_not_ready,
            replay_ego=int(bool(include_ego)))
        if log_detail:
            self._log_replay_detail(
                "frame-end", frame=frame_index, inserted=inserted,
                skipped_missing=skipped_missing,
                inserted_not_ready=inserted_not_ready,
                restored=restored)
        return restored > 0

    def _apply_replay_trajectory(self, trajectory):
        self._last_replay_error = {}
        self._playback_replay_frames = []
        self._playback_replay_index = 0
        self._playback_replay_active = False
        self._playback_replay_ended = False
        frames = (trajectory or {}).get("frames") or []
        if not frames:
            self._set_replay_error("no_frames")
            return False
        include_ego = bool(self.danger_miner.config.get("replay_ego", False))
        if not self._replay_has_targets(frames, include_ego):
            self._set_replay_error(
                "no_background_replay_targets",
                frames=len(frames), replay_ego=int(bool(include_ego)))
            return False
        initial_restored = self._apply_replay_frame(
            frames[0], frame_index=0, include_ego=include_ego)
        if self._sumo_connection_lost:
            return False
        initial_error = dict(self._last_replay_error)
        if not initial_restored:
            self._set_replay_error(
                "initial_replay_restore_failed",
                frames=len(frames),
                initial_reason=initial_error.get("reason", "unknown"),
                replay_ego=int(bool(include_ego)))
            return False
        self._playback_replay_frames = list(frames)
        self._playback_replay_index = 1
        self._playback_replay_active = True
        self._set_replay_error(
            "started", frames=len(frames),
            initial_restored=int(bool(initial_restored)),
            initial_reason=initial_error.get("reason", "unknown"),
            replay_ego=int(bool(include_ego)))
        return True

    def _clear_playback_replay(self):
        self._playback_replay_frames = []
        self._playback_replay_index = 0
        self._playback_replay_active = False
        self._playback_replay_ended = False

    def _advance_playback_replay(self):
        if not self._playback_replay_active:
            return
        if self._playback_replay_index >= len(self._playback_replay_frames):
            self._playback_replay_active = False
            self._playback_replay_ended = True
            return
        include_ego = bool(self.danger_miner.config.get("replay_ego", False))
        frame_index = self._playback_replay_index
        frame = self._playback_replay_frames[frame_index]
        if self._should_log_replay_detail(frame_index):
            self._log_replay_detail(
                "advance-before", frame=frame_index,
                total=len(self._playback_replay_frames))
        ok = self._apply_replay_frame(
            frame, frame_index=frame_index, include_ego=include_ego)
        if self._sumo_connection_lost:
            self._clear_playback_replay()
            return
        if ok:
            self._playback_replay_index += 1
            self._log_playback_replay_progress(frame_index)
            return
        detail = self._format_replay_error()
        print(
            '[danger replay skipped frame] '
            f"{self._replay_context()} {detail}",
            flush=True)
        self._playback_replay_index += 1
        self._log_playback_replay_progress(frame_index)

    def _log_playback_replay_progress(self, frame_index):
        interval = max(
            0, int(self.danger_miner.config.get(
                "replay_progress_interval", 0)))
        total = len(self._playback_replay_frames)
        current = frame_index + 1
        if current >= total:
            self._playback_replay_active = False
            self._playback_replay_ended = True
            print(
                '[danger replay finished] '
                f"{self._replay_context()} frames={total}",
                flush=True)
        elif interval and current % interval == 0:
            print(
                '[danger replay progress] '
                f"{self._replay_context()} frame={current}/{total}",
                flush=True)

    def _danger_candidates(self):
        """Return nearby vehicles, preferring those on ego's current road."""
        try:
            ego_pos = traci.vehicle.getPosition(self.egoid)
            ego_road = traci.vehicle.getRoadID(self.egoid)
        except Exception:
            return []
        candidates = []
        for veh_id in traci.vehicle.getIDList():
            if veh_id == self.egoid:
                continue
            try:
                pos = traci.vehicle.getPosition(veh_id)
                same_road = traci.vehicle.getRoadID(veh_id) == ego_road
            except Exception:
                continue
            distance = math.hypot(pos[0] - ego_pos[0], pos[1] - ego_pos[1])
            candidates.append((not same_road, distance, veh_id))
        return sorted(candidates)

    def _valid_danger_initial_state(self):
        """Reject scenarios that are already impossible before ego can react."""
        if self.egoid not in traci.vehicle.getIDList():
            return False
        try:
            if traci.simulation.getCollidingVehiclesNumber() > 0:
                return False
            if self.check_collision(self.egoid, xishu=0.7):
                return False
        except Exception:
            return False
        candidates = self._danger_candidates()
        if candidates:
            min_distance = float(self.danger_miner.config['min_initial_distance'])
            if candidates[0][1] < min_distance:
                return False
        return True

    def _danger_target_color(self):
        return self._rgba_config("danger_target_color", [255, 255, 0, 255])

    def _highlight_danger_target(self, target_id):
        if not bool(self.danger_miner.config.get("highlight_danger_target", True)):
            return None
        color = self._danger_target_color()
        try:
            traci.vehicle.setColor(str(target_id), color)
            return color
        except Exception:
            return None

    def _replay_vehicle_color(self):
        return self._rgba_config("replay_vehicle_color", [0, 0, 255, 255])

    def _highlight_replay_vehicle(self, veh_id):
        if str(veh_id) == self.egoid:
            return None
        if not bool(self.danger_miner.config.get("highlight_replay_vehicles", True)):
            return None
        color = self._replay_vehicle_color()
        try:
            traci.vehicle.setColor(str(veh_id), color)
            return color
        except Exception:
            return None

    def _highlight_replay_background_vehicles(self):
        if not self._playback_replay_active:
            return 0
        try:
            vehicle_ids = traci.vehicle.getIDList()
        except Exception:
            return 0
        highlighted = 0
        for veh_id in vehicle_ids:
            if str(veh_id) == self.egoid:
                continue
            if self._highlight_replay_vehicle(veh_id) is not None:
                highlighted += 1
        return highlighted

    def _wait_for_attack_candidates(self, variant):
        max_steps = max(0, int(self.danger_miner.config.get(
            "intent_attack_wait_steps", 30)))
        if max_steps <= 0:
            return []
        for step in range(1, max_steps + 1):
            try:
                traci.simulationStep()
                for veh_id in traci.vehicle.getIDList():
                    traci.vehicle.setLaneChangeMode(veh_id, 0)
                    traci.vehicle.setSpeedMode(veh_id, 0)
            except Exception as exc:
                if type(exc).__name__ == "FatalTraCIError":
                    self._mark_sumo_connection_lost(
                        "intent_attack_wait", exc)
                    variant['apply_error'] = 'sumo_connection_lost'
                    return []
                variant['apply_error'] = f"attack_wait_{type(exc).__name__}"
                return []
            candidates = self._danger_candidates()
            if candidates:
                variant['attack_wait_steps'] = int(step)
                return candidates
        return []

    def _bootstrap_attack_from_replay(self, variant):
        if not bool(self.danger_miner.config.get(
                "intent_attack_replay_bootstrap", True)):
            return False
        replay = variant.get("replay_trajectory") or {}
        frames = replay.get("frames") or []
        if not frames:
            variant['apply_error'] = 'no_attack_candidates_no_replay'
            return False
        max_frames = max(1, int(self.danger_miner.config.get(
            "intent_attack_bootstrap_frames", 8)))
        include_ego = bool(self.danger_miner.config.get("replay_ego", False))
        for frame_index, frame in enumerate(frames[:max_frames]):
            restored = self._apply_replay_frame(
                frame, frame_index=frame_index, include_ego=include_ego)
            if self._sumo_connection_lost:
                variant['apply_error'] = 'sumo_connection_lost'
                return False
            if restored and self._danger_candidates():
                variant['attack_bootstrap_frame'] = int(frame_index)
                return True
        last_error = dict(self._last_replay_error)
        variant['apply_error'] = (
            'replay_bootstrap_failed_'
            f"{last_error.get('reason', 'unknown')}")
        return False

    def _apply_danger_variant(self, variant):
        """Apply a small, executable traffic-intent perturbation in SUMO.

        ``scenario_copy`` keeps the selected logged configuration unchanged.
        ``intent_attack`` asks a nearby same-road agent to enter ego's lane,
        when that lane exists, and applies a speed perturbation.
        """
        if variant.get('kind') == 'scenario_copy':
            return True
        candidates = self._danger_candidates()
        variant['attack_candidate_count'] = int(len(candidates))
        if not candidates:
            candidates = self._wait_for_attack_candidates(variant)
            variant['attack_candidate_count'] = int(len(candidates))
        if not candidates:
            variant.setdefault('apply_error', 'no_attack_candidates_after_wait')
            return False
        _, _, target_id = candidates[0]
        try:
            target_speed = float(traci.vehicle.getSpeed(target_id))
            ego_speed = float(traci.vehicle.getSpeed(self.egoid))
            max_speed = float(self.danger_miner.config['max_target_speed'])
            desired_speed = max(
                target_speed,
                ego_speed + float(variant['attack_speed_delta']))
            ego_lane = int(traci.vehicle.getLaneIndex(self.egoid))
            target_lane = int(traci.vehicle.getLaneIndex(target_id))
            if target_lane != ego_lane:
                # SUMO rejects infeasible lane changes; the speed
                # perturbation below remains a useful valid variant.
                try:
                    traci.vehicle.changeLane(target_id, ego_lane, 2.0)
                except Exception:
                    pass
            traci.vehicle.setSpeed(target_id, np.clip(desired_speed, 0.0, max_speed))
            target_color = self._highlight_danger_target(target_id)
            variant['target_id'] = str(target_id)
            variant['initial_distance'] = float(candidates[0][1])
            if target_color is not None:
                variant['target_color'] = list(target_color)
            return True
        except Exception as exc:
            variant['apply_error'] = type(exc).__name__
            variant['target_id'] = str(target_id)
            return False

    def set_danger_mining(
            self, active, record_events=None, collect_episodes=None,
            update_library=None):
        """Called by the training loop to delimit bootstrap and post-training."""
        stats = self.danger_miner.set_active(
            active, record_events=record_events,
            collect_episodes=collect_episodes, update_library=update_library)
        print(
            '[danger mining phase] '
            f"worker={self.worker_index} active={self.danger_miner.active} "
            f"record_events={self.danger_miner.record_events} "
            f"collect_episodes={self.danger_miner.collecting} "
            f"update_library={self.danger_miner.update_library} "
            f"library={int(stats['library_size'])}")
        return stats

    def set_scenario_mode(self, mode):
        if mode not in ('normal', 'sweep'):
            raise ValueError(f'Unknown scenario mode: {mode}')
        self._scenario_mode = mode
        if mode == 'sweep':
            # Each batched worker starts at a different scene and strides by
            # env_amount, so one evaluation pass covers the full config set.
            self._scenario_sweep_index = self.worker_index
        return mode

    def get_scenario_count(self):
        if not os.path.isdir(self.cfg_dir):
            return 0
        return sum(
            name.lower().endswith('.sumocfg')
            for name in os.listdir(self.cfg_dir))

    def get_current_scenario(self):
        return {
            'scenario_path': str(self.path),
            'scenario_name': os.path.basename(str(self.path)),
            'scenario_mode': self._scenario_mode,
            'env_mode': self.env_mode,
            'worker': int(self.worker_index),
            'env_episode': int(self.episode),
        }

    def _danger_generation_active(self):
        return (
            bool(getattr(self.danger_miner, "active", False)) and
            bool(self.danger_miner.config.get("generation_enabled", True)))
    
    def _step(self, action):
        try:
            return self._step_impl(action)
        except Exception as exc:
            if type(exc).__name__ == "FatalTraCIError":
                self._mark_sumo_connection_lost("step_outer", exc)
                fallback_state = np.zeros(self.state_dim, dtype=np.float32)
                return fallback_state, True, "sumo_closed", 0.0, False
            raise

    def _step_impl(self, action):
        '''
        update ego vehicle state
        input:
            action - Int ，{ 0：-2m/s^2,
                            1: -1m/s^2,
                            2:  0m/s^2,
                            3:  1m/s^2,
                            4:  2m/s^2,
                             }
        output:
            state - array
            done - bool , True if collision or offroad
        '''
        self._runtime_episode_steps += 1
        replay_step_log = (
            self._playback_replay_active and
            self._should_log_replay_detail())
        if replay_step_log:
            self._log_replay_detail(
                "step-begin", next_frame=self._playback_replay_index,
                frames=len(self._playback_replay_frames))
        target_v = 5

        done , info = False , None
        try:
            road_index = traci.vehicle.getRoadID('Auto')
            self.last_road_index=road_index

        except Exception as exc:
            road_index = self.last_road_index
            if replay_step_log:
                self._log_replay_detail(
                    "road-read-failed", error=type(exc).__name__)
            else:
                print('no road 325')
        # last state, backup if egocar is removrd when collision.
        state = self._get_features()
        if replay_step_log:
            self._log_replay_detail(
                "features-ready", road=road_index,
                ego_speed=f"{float(state[0] * self.ego_speed_max):.3f}")
        # print('state', state)
        ego_speed = state[0] * self.ego_speed_max
        ego_positionx = state[0] * self.max_x
        ego_positiony = state[1] * self.max_y
        amax = 2
        stop_line = 7.2
        acc = 0

        collision = traci.simulation.getCollidingVehiclesNumber() > 0
        if collision:
            collision_id = traci.simulation.getCollidingVehiclesIDList()
            if self.egoid in collision_id:
                print(7777777777)
                done = True
                info = 'collision'

        p_unsafe = False

# 直接使用输入的期望车速（范围[-1,1]）
        action = np.clip(action,-1,1)
        normalized_speed = action  # 假设action是单个值，范围[-1,1]
        target_v = ((normalized_speed + 1) / 2) * 6  # 0-6m/s
# 保留跟车逻辑（根据前车距离调整期望速度）
        # print('target_v',target_v)
        
        acc = self.pid_lon.cal_output(ego_speed - target_v)
        try:
            self.last_risk_cost = float(self._compute_risk_cost(acc))
        except Exception as exc:
            self.last_risk_cost = 0.0
            if replay_step_log:
                self._log_replay_detail(
                    "risk-cost-failed", error=type(exc).__name__)


        if road_index == "78" :
            done = True
            info = "goal"
            print('GOAL!!!')
        self.dt = traci.simulation.getDeltaT()
        ego_speed += acc * self.dt
        ego_speed = np.clip(ego_speed , 0 , 6.0)
        if not done:
            traci.vehicle.setSpeed(self.egoid , ego_speed)
            self._highlight_replay_background_vehicles()
            # Upadate all the vehicles. Collision will happen.
            if not self._playback_replay_active:
                random_horizon = random.uniform(1.5,2.0)
                self.npc_collision_avoidance_ctra(
                    horizon=random_horizon, dt_pred=0.2, k_neighbors=3,
                    decel=self.npc_emergency_decel, ego_id="Auto",
                    safety_scale=0.9)
            if replay_step_log:
                self._log_replay_detail(
                    "sumo-step-before", next_frame=self._playback_replay_index)
            try:
                traci.simulationStep()
            except Exception as exc:
                if type(exc).__name__ == "FatalTraCIError":
                    self._mark_sumo_connection_lost("env_simulation_step", exc)
                    return state, True, "sumo_closed", acc, p_unsafe
                raise
            self._highlight_replay_background_vehicles()
            if replay_step_log:
                self._log_replay_detail(
                    "sumo-step-after",
                    sim_time=f"{traci.simulation.getTime():.2f}",
                    next_frame=self._playback_replay_index)
            self._advance_playback_replay()
            if self._sumo_connection_lost:
                return state, True, "sumo_closed", acc, p_unsafe
            if replay_step_log:
                self._log_replay_detail(
                    "advance-after", next_frame=self._playback_replay_index,
                    active=int(bool(self._playback_replay_active)))
            vid = traci.vehicle.getIDList()
            if self.egoid not in vid:
                # print('555555555555555555555555555555555555555555555555555555')
                done = True  # Collision happened!
                info = 'collision'

            elif self.check_collision('Auto',xishu=0.7):
                # print('6666666666666666666666666666666666666666666666666666')
                done = True
                info ='collision'

        next_state = state if done else self._get_features()
        time = traci.simulation.getTime()
        if time >= 200.0:
            done = True
            info = 'max_time'
            print('max_time')
        max_episode_steps = int(self.danger_miner.config.get(
            "max_episode_steps", 2500))
        if not done and max_episode_steps > 0 and (
                self._runtime_episode_steps >= max_episode_steps):
            done = True
            info = 'max_steps'
            print(
                '[danger episode cutoff] '
                f"source={os.path.basename(str(self.path))} "
                f"steps={self._runtime_episode_steps} sim_time={time:.1f}")
        # print('state',state)
        return next_state, done, info, acc, p_unsafe

    def _get_reward(self, state, step_over1):

        ego_position_y = state[1] * self.max_y
        # p_1 = state[8]
        # p_2 = state[14]
        # p_3 = state[20]
        # p_4 = state[26]
        # p_5 = state[32]
        # p_6 = state[38]

        r_efficiency = -0.01
        r_unsafe, r_rule_vi, r_rule_co = (0 for i in range(3))
        ttc = self.ttc_compute(state)[0]

        if 1.5 < ttc <3:
            # print('ttc!!!!')
            r_ttc = -1
        elif ttc < 1.5:
            r_ttc = -2
        else:
            r_ttc = 0
        # print('ttc',ttc)
        return r_efficiency, r_unsafe, r_rule_vi, r_rule_co,r_ttc


    def ttc_compute(self, state, d_safe=0.0, ttc_cap=10.0):
        # print('STATE',state)
        """
        根据state向量计算TTC（取最小纵向TTC）。
        参数:
            state: np.array, 结构与_get_features返回一致
            d_safe: 安全距离（m）
            ttc_cap: 上限秒数，用于归一化
        返回:
            min_ttc, min_ttc_norm
        """
        d_safe = 4.36
        # 取自车速度（反归一化）
        ego_speed = state[0] * self.ego_speed_max

        min_ttc = math.inf

        # 遍历6辆车，每辆占5维特征
        for i in range(6):
            base = 1 + i * 5
            x_rel = state[base] * self.max_x
            y_rel = state[base + 1] * self.max_y
            vx_i = state[base + 2] * self.other_speed_max
            vy_i = state[base + 3] * self.other_speed_max
            # angle 不参与 TTC 直接计算

            # 相对纵向速度
            v_rel_x = vx_i - ego_speed

            # 只计算前方车辆
            if x_rel > 0 and abs(y_rel)<2:
                if v_rel_x < 0:
                    ttc = (x_rel - d_safe) / (-v_rel_x)
                    if ttc < 0:
                        ttc = 0.0
                else:
                    ttc = math.inf
                min_ttc = min(min_ttc, ttc)

        # 归一化
        if math.isinf(min_ttc):
            min_ttc_norm = 1.0
        else:
            min_ttc_norm = max(0.0, min(1.0, min_ttc / ttc_cap))

        return min_ttc, min_ttc_norm


    def step(self, action):
        # print('action', action)
        state, done, info, acc, p_unsafe = self._step(action)
        # print('acc',acc)
        obs = self._to_obs(state)
        obs['is_first'] = False
        obs['is_terminal'] = done
        ego_speed = state[0] * self.ego_speed_max
        r_collision = 0
        r_goal = 0
        r_time = 0
        r_speed = 0
        r_speed = ego_speed*0.02
        cost_coll = 0
        if done == True:
            if info == "collision":
                r_collision = -100 - 2*ego_speed
                cost_coll = 1
                print('LOSE!!!')
            if info == "goal":
                r_goal = 140
                # print('GOAL!!!')
            if info in ('max_time', 'max_steps', 'sumo_closed'):
                r_time = -40
        r_jerk = -abs((self.last_acc- acc) / 0.15)*0.15
        lambda_action = 0.1  # 惩罚系数，可自己调
        r_action_inconsistency = -abs(action - self.last_action) * lambda_action
        # print(r_jerk)
        self.last_acc = acc
        self.last_action =action
        r_efficiency, r_unsafe, r_rule_vi, r_rule_co,r_ttc = self._get_reward(state, 1)


        reward = r_efficiency + r_collision + r_unsafe*0 + r_goal  + r_rule_vi + r_rule_co +r_time + r_jerk+r_ttc+r_speed +0*r_action_inconsistency
        # print('reward',reward)
        try:
            reward= reward[0]
        except:
            reward = reward
        cost = cost_coll
        cost = float(cost + self.risk_cost_weight * self.last_risk_cost)
        
        if cost<0.10:
            cost=0.0
        elif cost>0.10 and cost<0.5:
            cost=cost*0.5
        elif cost>=0.5:
            print('cost',cost)
            cost=1.0

        if self.danger_miner.active and self._episode_active:
            self._record_episode_step(
                state, cost, info if done else None, action=action, acc=acc)
            if done:
                # Persist terminal failures immediately. Evaluation sweeps stop
                # directly after their final episode and therefore do not
                # necessarily issue one more reset.
                self._finish_active_episode()

        info_out = {'discount': np.array(0.0 if done else 1.0, dtype=np.float32)}

        return state, reward,cost, done, info_out

    def set_speed(self, **kwargs):
        for (k, v) in kwargs.items():
            traci.vehicle.setSpeed(k, v)

    def _select_cfg_for_current_episode(self):
            if not os.path.isdir(self.cfg_dir):
                raise FileNotFoundError(f"cfg directory does not exist: {self.cfg_dir}")

            all_cfg_files = [
                f for f in os.listdir(self.cfg_dir)
                if f.lower().endswith(".sumocfg")
            ]
            if not all_cfg_files:
                raise FileNotFoundError(f"no .sumocfg files found in {self.cfg_dir}")

            if self._scenario_mode == 'sweep':
                all_cfg_files = sorted(all_cfg_files)
                chosen = all_cfg_files[
                    self._scenario_sweep_index % len(all_cfg_files)]
                self._scenario_sweep_index += self.env_amount
                full_path = os.path.join(self.cfg_dir, chosen)
                if not self._danger_generation_active():
                    print(
                        f"[worker {self.worker_index}] all-scene sweep: {chosen}")
                return full_path

            pair_index = self.worker_index // self.scenario_group_size
            pair_episode = (self.episode - 1) // self.scenario_group_size + 1
            levels = ("easy", "medium", "hard")

            def match_level(fname):
                name_low = fname.lower()
                for lv in levels:
                    if lv in name_low:
                        return lv
                return None

            files_by_level = {level: [] for level in levels}
            unmatched_cfgs = []
            for fname in all_cfg_files:
                level = match_level(fname)
                if level is None:
                    unmatched_cfgs.append(fname)
                else:
                    files_by_level[level].append(fname)

            if pair_episode >= 500:
                level_weights = {"easy": 0.20, "medium": 0.50, "hard": 0.30}
            else:
                progress = max(0.0, min(1.0, (pair_episode - 1) / 499.0))
                level_weights = {
                    "easy": 1.00 + (0.20 - 1.00) * progress,
                    "medium": 0.50 * progress,
                    "hard": 0.30 * progress,
                }

            schedule_key = (self.env_mode, self.cfg_dir, pair_index, pair_episode)
            if schedule_key not in Sumoenv._shared_cfgs:
                available_levels = [
                    level for level in levels
                    if files_by_level[level] and level_weights.get(level, 0.0) > 0.0]
                if available_levels:
                    weights = [level_weights[level] for level in available_levels]
                    selected_level = Sumoenv._shared_rng.choices(
                        available_levels, weights=weights, k=1)[0]
                    candidate_cfgs = files_by_level[selected_level]
                else:
                    candidate_cfgs = unmatched_cfgs or all_cfg_files
                Sumoenv._shared_cfgs[schedule_key] = Sumoenv._shared_rng.choice(candidate_cfgs)
            chosen = Sumoenv._shared_cfgs[schedule_key]
            selected_level = match_level(chosen) or "all"
            full_path = os.path.join(self.cfg_dir, chosen)
            if not self._danger_generation_active():
                print(
                    f"[worker {self.worker_index} episode {self.episode} "
                    f"pair_episode {pair_episode}] scenario cfg: {chosen} "
                    f"level={selected_level}")
            return full_path


    def add_vehicle(self, random_initial=True):

        n = np.random.randint(7, 15)
        m = np.random.randint(3,6)

        # n = np.random.randint(4, 15)
        # m = np.random.randint(3,6)

        if n > m:
            h = n-m
        else:
            h = 0
        for i in range(m):
            position, speed = 0.5,0.5
            position += np.random.randint(0,5)
            speed += np.random.uniform(0,1)
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 10)
            vehicle_id = 'Vm' + str(i)
            vehicle_idb = 'Vmb' + str(i)
            time = str(1)
            if a == 1:
                # traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan11route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan12route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
                if b ==1:
                    traci.vehicle.add(vehicle_idb, typeID='SlowCar' , routeID='huan2route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')

            elif a == 2:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan11route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
                if b ==2:
                    traci.vehicle.add(vehicle_idb, typeID='SlowCar' , routeID='huan2route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')

            elif a == 3:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan1route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
                if b ==3:
                    traci.vehicle.add(vehicle_idb, typeID='SlowCar' , routeID='huan2route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
            elif a == 4:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan10route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
                if b ==4:
                    traci.vehicle.add(vehicle_idb, typeID='SlowCar' , routeID='huan3route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')

            elif a == 5:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan2route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='1')
                if b ==5:
                    traci.vehicle.add(vehicle_idb, typeID='SlowCar' , routeID='huan3route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
            elif a == 6:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan15route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
                if b ==6:
                    traci.vehicle.add(vehicle_idb, typeID='SlowCar' , routeID='huan_route1' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')

            elif a == 7:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan2route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='1')
                if b ==7:
                    traci.vehicle.add(vehicle_idb, typeID='SlowCar' , routeID='huan_route1' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')

            elif a == 8:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan1route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
                if b ==8:
                    traci.vehicle.add(vehicle_idb, typeID='SlowCar1' , routeID='huan_route' , depart=time, departPos=str(0*position) , departSpeed=str(0*speed) , departLane='0')

            else:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan2route' , depart=time, departPos=str(position) , departSpeed=str(0.5*speed) , departLane='0')
                traci.vehicle.add(vehicle_idb, typeID='SlowCar' , routeID='huan_route2' , depart=time, departPos=str(0*position) , departSpeed=str(0*speed) , departLane='0')




        for i in range(h):
            position, speed = 0,0.5
            position += np.random.randint(0,10)
            speed += np.random.uniform(0,1)
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 10)
            vehicle_id = 'Vh' + str(i)
            vehicle_idb = 'Vhb' + str(i)
            time = str(np.random.randint(5, 40))
            if a == 1:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan15route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
                if b ==1:
                    traci.vehicle.add(vehicle_idb, typeID='SlowCar' , routeID='huan15route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
            elif a == 2:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan10route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
            elif a == 3:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan15route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
            elif a == 4:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan2route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='1')
            elif a == 5:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan15route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
            elif a == 6:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan1route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='1')
            elif a == 7:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan10route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
            elif a == 8:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan15route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')
            else:
                traci.vehicle.add(vehicle_id, typeID='SlowCar' , routeID='huan11route' , depart=time, departPos=str(position) , departSpeed=str(speed) , departLane='0')

    def reset(self, initial=True):
        self.pid_lon.reset()
        # zw = np.random.randint(0, 3)
        # # zw = 1
        # if zw ==0:
        #     path = self.path
        #     traci.load(["-c", path]) #加载并连接sumo，self.path:路径
        #     # self.add_vehicle(initial)
        #     traci.simulationStep()
        # elif zw == 1:
        #     path ='/home/sugon/zw/huandao_orsplag_safe/embodied/huandao_map/huandao_1.sumocfg'
        #     traci.load(["-c", path]) #加载并连接sumo，self.path:路径
        #     # self.add_vehicle(initial)
        #     traci.simulationStep()
        #     self.add_vehicle(initial)        
        # elif zw == 2 :
        #     path = '/home/sugon/zw/huandao_orsplag_safe/embodied/huandao_map/huandao_2.sumocfg'
        self._finish_active_episode()
        self._clear_playback_replay()
        self._runtime_episode_steps = 0
        self.episode += 1
        fallback_path = self._select_cfg_for_current_episode()
        variant = None
        danger_attempted = False
        self._brake_streak_s = {}

            # —— 制动剔除相关 ——
        self._brake_streak_s = {}
        self._brake_excluded = set()
        self._brake_random_exempt = set()
        self._brake_seen_npcs = set()
        self._low_speed_pair_streak_s = {}
        max_attempts = max(1, int(self.danger_miner.config.get(
            "max_danger_generation_attempts", 8)))
        fallback_to_normal = bool(self.danger_miner.config.get(
            "fallback_to_normal_on_danger_failure", False))

        for attempt in range(1, max_attempts + 1):
            candidate = self.danger_miner.sample_variant(fallback_path)
            if not candidate:
                break
            danger_attempted = True
            self._clear_playback_replay()
            mode = str(candidate.get("mode", candidate.get("kind", "scenario_copy")))
            self.path = candidate['source_cfg']
            if not self._load_scene(self.path):
                print(
                    '[danger generation skipped] '
                    f"worker={self.worker_index} pid={os.getpid()} "
                    f"attempt={attempt}/{max_attempts} mode={mode} "
                    f"reason=source_load_failed source={os.path.basename(str(self.path))}",
                    flush=True)
                continue

            replay = candidate.get("replay_trajectory") or {}
            replayed = False
            applied = False
            if mode == "replay":
                replayed = self._apply_replay_trajectory(replay)
                candidate["used_replay_trajectory"] = float(replayed)
                if not replayed:
                    detail = self._format_replay_error()
                    print(
                        '[danger generation skipped] '
                        f"worker={self.worker_index} pid={os.getpid()} "
                        f"attempt={attempt}/{max_attempts} mode=replay "
                        f"reason=replay_restore_failed source={os.path.basename(str(self.path))} "
                        f"{detail}",
                        flush=True)
                    continue
                require_valid = bool(self.danger_miner.config.get(
                    "require_valid_replay_state", False))
                if require_valid and not self._valid_danger_initial_state():
                    print(
                        '[danger generation skipped] '
                        f"worker={self.worker_index} pid={os.getpid()} "
                        f"attempt={attempt}/{max_attempts} mode=replay "
                        f"reason=invalid_initial_state source={os.path.basename(str(self.path))}",
                        flush=True)
                    continue
            else:
                if not self._valid_danger_initial_state():
                    print(
                        '[danger generation skipped] '
                        f"worker={self.worker_index} pid={os.getpid()} "
                        f"attempt={attempt}/{max_attempts} mode={mode} "
                        f"reason=invalid_initial_state source={os.path.basename(str(self.path))}",
                        flush=True)
                    continue
                if mode == "scenario_copy":
                    applied = True
                else:
                    applied = self._apply_danger_variant(candidate)
                    if not applied:
                        apply_error = candidate.get('apply_error', 'unknown')
                        print(
                            '[danger generation skipped] '
                            f"worker={self.worker_index} pid={os.getpid()} "
                            f"attempt={attempt}/{max_attempts} mode={mode} "
                            f"reason=variant_apply_failed apply_error={apply_error} "
                            f"candidates={candidate.get('attack_candidate_count', 'unknown')} "
                            f"target={candidate.get('target_id', 'none')} "
                            f"wait={candidate.get('attack_wait_steps', 'none')} "
                            f"bootstrap={candidate.get('attack_bootstrap_frame', 'none')} "
                            f"source={os.path.basename(str(self.path))}",
                            flush=True)
                        continue

            candidate["used_danger_mode"] = mode
            variant = candidate
            print(
                '[danger generation] '
                f"worker={self.worker_index} pid={os.getpid()} "
                f"episode={self.episode} mode={mode} kind={candidate.get('kind', mode)} "
                f"source={os.path.basename(str(self.path))} "
                f"target={candidate.get('target_id', 'none')} "
                f"replay={int(bool(replayed))} "
                f"copy={int(mode == 'scenario_copy')} "
                f"attack={int(mode == 'intent_attack')} "
                f"attack_wait={candidate.get('attack_wait_steps', 'none')} "
                f"attack_bootstrap={candidate.get('attack_bootstrap_frame', 'none')} "
                f"frames={len(replay.get('frames') or []) if replayed else 0} "
                f"replay_ego={int(bool(self.danger_miner.config.get('replay_ego', False)))}",
                flush=True)
            break

        if variant is None:
            if danger_attempted and not fallback_to_normal:
                raise RuntimeError(
                    'Danger scenario generation failed and normal-scene fallback '
                    f'is disabled after {max_attempts} attempts.')
            self.path = fallback_path
            if not self._load_scene_with_fallbacks(self.path, fallback_path):
                raise RuntimeError(
                    f'Could not load any SUMO scenario from {self.cfg_dir}')
        self._begin_episode_stats(variant)
        state = self._get_features()
        self._capture_initial_episode_frame(state)
        obs = self._to_obs(state)
        obs['is_first'] = True
        obs['is_terminal'] = False
        return state



    def is_point_in_polygon(self,point, polygon):
        """判断点是否在多边形内部"""
        x, y = point
        n = len(polygon)
        inside = False
        for i in range(n):
            j = (i + 1) % n
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            # 检查点是否在边上
            if ((yi > y) != (yj > y)):
                x_intersect = (y - yi) * (xj - xi) / (yj - yi + 1e-8) + xi
                if x <= x_intersect:
                    inside = not inside
        return inside

    def get_edge_normal(self,p1, p2):
        """计算边的法向量（单位向量）"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        # 垂直方向向量（法向量）
        normal = (-dy, dx)
        # 归一化
        length = math.hypot(normal[0], normal[1])
        if length < 1e-8:
            return (0, 0)
        return (normal[0]/length, normal[1]/length)

    def project_polygon(self,polygon, axis):
        """将多边形投影到轴上，返回最小值和最大值"""
        min_val = float('inf')
        max_val = -float('inf')
        for (x, y) in polygon:
            proj = x * axis[0] + y * axis[1]
            if proj < min_val:
                min_val = proj
            if proj > max_val:
                max_val = proj
        return (min_val, max_val)

    def intervals_overlap(self,a_min, a_max, b_min, b_max, tolerance=0):
        """判断两个区间是否重叠，可设置容忍度"""
        return not (a_max + tolerance < b_min or b_max + tolerance < a_min)

    def is_rectangle_overlap(self, x1, y1, angle1, width1, height1, x2, y2, angle2, width2, height2, tolerance=0):
        """
        判断两个矩形是否重叠，增加tolerance参数控制宽松度
        tolerance > 0 时检测更宽松（允许一定距离内判定为重叠）
        tolerance = 0 时为严格检测
        """
        # 将角度转换为弧度
        angle1_rad = math.radians(angle1)
        angle2_rad = math.radians(angle2)

        # 计算第一个矩形的四个顶点坐标
        w1_half = width1 / 2
        h1_half = height1 / 2
        cos1 = math.cos(angle1_rad)
        sin1 = math.sin(angle1_rad)
        
        p1_1 = (x1 - w1_half * cos1 - h1_half * sin1, y1 - w1_half * sin1 + h1_half * cos1)
        p2_1 = (x1 + w1_half * cos1 - h1_half * sin1, y1 + w1_half * sin1 + h1_half * cos1)
        p3_1 = (x1 + w1_half * cos1 + h1_half * sin1, y1 + w1_half * sin1 - h1_half * cos1)
        p4_1 = (x1 - w1_half * cos1 + h1_half * sin1, y1 - w1_half * sin1 - h1_half * cos1)
        rect1 = [p1_1, p2_1, p3_1, p4_1]

        # 计算第二个矩形的四个顶点坐标
        w2_half = width2 / 2
        h2_half = height2 / 2
        cos2 = math.cos(angle2_rad)
        sin2 = math.sin(angle2_rad)
        
        p1_2 = (x2 - w2_half * cos2 - h2_half * sin2, y2 - w2_half * sin2 + h2_half * cos2)
        p2_2 = (x2 + w2_half * cos2 - h2_half * sin2, y2 + w2_half * sin2 + h2_half * cos2)
        p3_2 = (x2 + w2_half * cos2 + h2_half * sin2, y2 + w2_half * sin2 - h2_half * cos2)
        p4_2 = (x2 - w2_half * cos2 + h2_half * sin2, y2 - w2_half * sin2 - h2_half * cos2)
        rect2 = [p1_2, p2_2, p3_2, p4_2]

        # 检查是否有顶点在另一个矩形内部
        for p in rect1:
            if self.is_point_in_polygon(p, rect2):
                return True
        for p in rect2:
            if self.is_point_in_polygon(p, rect1):
                return True

        # 使用分离轴定理(SAT)检查边是否分离
        axes = []
        # 获取第一个矩形的四条边的法向量作为潜在分离轴
        for i in range(4):
            p1 = rect1[i]
            p2 = rect1[(i+1)%4]
            normal = self.get_edge_normal(p1, p2)
            if normal != (0, 0):
                axes.append(normal)
        
        # 获取第二个矩形的四条边的法向量作为潜在分离轴
        for i in range(4):
            p1 = rect2[i]
            p2 = rect2[(i+1)%4]
            normal = self.get_edge_normal(p1, p2)
            if normal != (0, 0):
                axes.append(normal)
        
        # 检查所有潜在分离轴
        for axis in axes:
            proj1 = self.project_polygon(rect1, axis)
            proj2 = self.project_polygon(rect2, axis)
            if not self.intervals_overlap(proj1[0], proj1[1], proj2[0], proj2[1], tolerance):
                return False  # 找到分离轴，不重叠
        
        return True  # 没有找到分离轴，重叠

    def check_collision(self, vehicle_id, xishu=1, tolerance=0):
        """
        检查车辆碰撞风险
        xishu: 车辆尺寸缩放系数
        tolerance: 碰撞检测容忍度，值越大检测越宽松
        """
        vehicle_length = traci.vehicle.getLength(vehicle_id) * xishu
        vehicle_width = traci.vehicle.getWidth(vehicle_id) * xishu 

        # 获取当前车辆的位置和方向
        pos_x, pos_y = traci.vehicle.getPosition(vehicle_id)
        angle = traci.vehicle.getAngle(vehicle_id)
        angle_rad = math.radians(angle)
        # 调整位置（根据原始代码的偏移逻辑）
        pos_x = pos_x - math.sin(angle_rad) * 2.18
        pos_y = pos_y - math.cos(angle_rad) * 2.18

        # 获取周围车辆信息
        surrounding_vehicles = traci.vehicle.getIDList()

        for other_vehicle_id in surrounding_vehicles:
            if other_vehicle_id == vehicle_id:
                continue  # 跳过自身车辆
                
            # 获取其他车辆的位置、方向和尺寸
            other_pos_x, other_pos_y = traci.vehicle.getPosition(other_vehicle_id)
            other_angle = traci.vehicle.getAngle(other_vehicle_id)
            other_vehicle_length = traci.vehicle.getLength(other_vehicle_id) * xishu
            other_vehicle_width = traci.vehicle.getWidth(other_vehicle_id) * xishu
            
            # 调整其他车辆位置
            other_angle_rad = math.radians(other_angle)
            other_pos_x = other_pos_x - math.sin(other_angle_rad) * 2.18
            other_pos_y = other_pos_y - math.cos(other_angle_rad) * 2.18

            # 判断是否存在碰撞风险
            if self.is_rectangle_overlap(
                pos_x, pos_y, angle, vehicle_width, vehicle_length,
                other_pos_x, other_pos_y, other_angle, other_vehicle_width, other_vehicle_length,
                tolerance
            ):
                # print('存在碰撞风险')
                return True
        return False


    def start(self, gui=False):
        config_path = self.path
        self._sumo_gui = bool(gui)
        self._restart_sumo(config_path)
        for _ in range(3):
            traci.simulationStep()
        disable_tls=True

    
        # if disable_tls:
        #     # 将所有信号灯置为“全绿”以失效信号控制
        #     for tl in traci.trafficlight.getIDList():
        #         # 用第一相位长度来构造同长度的全绿状态
        #         logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0]
        #         # 取相位0的state长度；不同路口灯位数不同，需动态适配
        #         state_len = len(logic.phases[0].state)
        #         traci.trafficlight.setRedYellowGreenState(tl, 'G' * state_len)
        
    def close(self):
        try:
            traci.close()
        finally:
            self._sumo_started = False


    def npc_collision_avoidance_ctra(
            self,
            horizon=3.0,          # 预测时长（秒）
            dt_pred=0.1,          # 预测步长（秒）
            k_neighbors=6,        # 每车只检测最近k辆
            decel=2.0,            # 触发时给 -decel 的减速度
            ego_id="Auto",
            safety_scale=1.0,     # 碰撞半径放大系数（可调，比如 1.1 更保守）
            min_speed=0.0,
        ):
        """
        对除 ego_id 外的车辆做基于CTRA预测的碰撞检测，检测到碰撞的车才下发减速度。
        （已从圆碰撞改为：带航向的矩形OBB碰撞检测）
        """

        # ---------- 工具：角度unwrap ----------
        def unwrap_deg(prev, curr):
            d = (curr - prev + 180.0) % 360.0 - 180.0
            return prev + d

        # ---------- 初始化历史缓存 ----------
        if not hasattr(self, "_last_angle_deg"):
            self._last_angle_deg = {}
        if not hasattr(self, "_last_time_s"):
            self._last_time_s = None
# ---------- GUI 可视化：紧急制动车辆颜色 ----------
        if not hasattr(self, "_brake_vis_original_color"):
            self._brake_vis_original_color = {}

        if not hasattr(self, "_last_brake_vis_set"):
            self._last_brake_vis_set = set()
            
        # ---------- 当前仿真时刻 / dt ----------
        try:
            t_now = float(traci.simulation.getTime())
        except Exception:
            t_now = None

        dt_sim = float(getattr(traci.simulation, "getDeltaT", lambda: 0.1)())
        if dt_sim <= 1e-6:
            dt_sim = 0.1

        if self._last_time_s is None:
            self._last_time_s = t_now if t_now is not None else 0.0

        dt_for_yaw = (t_now - self._last_time_s) if (t_now is not None and self._last_time_s is not None) else dt_sim
        if dt_for_yaw <= 1e-6:
            dt_for_yaw = dt_sim

        # ---------- OBB 碰撞检测：SAT（分离轴定理） ----------
        def obb_intersect(x1, y1, psi1, hl1, hw1,
                        x2, y2, psi2, hl2, hw2, eps=1e-9):
            """
            两个带朝向矩形（中心 x,y；航向 psi；半长 hl；半宽 hw）是否相交。
            2D OBB SAT 检测：检查 4 个轴（A 的2轴 + B 的2轴）上投影是否重叠。
            """

            c1, s1 = math.cos(psi1), math.sin(psi1)
            c2, s2 = math.cos(psi2), math.sin(psi2)

            # A 的局部轴（单位向量）
            ax1 = (c1, s1)      # 前向
            ay1 = (-s1, c1)     # 左向
            # B 的局部轴
            ax2 = (c2, s2)
            ay2 = (-s2, c2)

            # A->B 的平移向量（世界坐标）
            tx = x2 - x1
            ty = y2 - y1

            # 在 A 的坐标系里表示平移（可减少计算）
            # tA = [dot(t, ax1), dot(t, ay1)]
            tA0 = tx * ax1[0] + ty * ax1[1]
            tA1 = tx * ay1[0] + ty * ay1[1]

            # 旋转矩阵 R = A^T * B（把B轴投影到A轴）
            R00 = ax1[0] * ax2[0] + ax1[1] * ax2[1]
            R01 = ax1[0] * ay2[0] + ax1[1] * ay2[1]
            R10 = ay1[0] * ax2[0] + ay1[1] * ax2[1]
            R11 = ay1[0] * ay2[0] + ay1[1] * ay2[1]

            # absR 加上小量避免接近平行时数值问题
            absR00 = abs(R00) + eps
            absR01 = abs(R01) + eps
            absR10 = abs(R10) + eps
            absR11 = abs(R11) + eps

            # 1) 轴 = A.x
            ra = hl1
            rb = hl2 * absR00 + hw2 * absR01
            if abs(tA0) > ra + rb:
                return False

            # 2) 轴 = A.y
            ra = hw1
            rb = hl2 * absR10 + hw2 * absR11
            if abs(tA1) > ra + rb:
                return False

            # 在 B 的坐标系里表示平移 tB = R^T * tA
            # tB0 = dot(t, ax2) = tA0*R00 + tA1*R10
            # tB1 = dot(t, ay2) = tA0*R01 + tA1*R11
            tB0 = tA0 * R00 + tA1 * R10
            tB1 = tA0 * R01 + tA1 * R11

            # 3) 轴 = B.x
            ra = hl1 * absR00 + hw1 * absR10
            rb = hl2
            if abs(tB0) > ra + rb:
                return False

            # 4) 轴 = B.y
            ra = hl1 * absR01 + hw1 * absR11
            rb = hw2
            if abs(tB1) > ra + rb:
                return False

            return True

        # ---------- 收集车辆状态（位置/速度/加速度/角度/尺寸） ----------
        veh_ids = traci.vehicle.getIDList()
        if not veh_ids:
            self._last_time_s = t_now if t_now is not None else self._last_time_s
            return []

        states = {}
        positions = []
        ids_for_tree = []

        for vid in veh_ids:
            try:
                x, y = traci.vehicle.getPosition(vid)
                v = float(traci.vehicle.getSpeed(vid))
                a = float(traci.vehicle.getAcceleration(vid))
                ang = float(traci.vehicle.getAngle(vid))  # deg
                L = float(traci.vehicle.getLength(vid))
                W = float(traci.vehicle.getWidth(vid))
            except Exception:
                continue

            # 估算 yaw_rate（rad/s）
            if vid in self._last_angle_deg:
                prev = self._last_angle_deg[vid]
                ang_unwrap = unwrap_deg(prev, ang)
                yaw_rate = math.radians((ang_unwrap - prev) / dt_for_yaw)
                self._last_angle_deg[vid] = ang_unwrap
            else:
                self._last_angle_deg[vid] = ang
                yaw_rate = 0.0

            psi = self.adjust_angle(ang)  # rad

            # 矩形半尺寸，乘 safety_scale（等价于放大车身）
            hl = 0.5 * L * safety_scale
            hw = 0.5 * W * safety_scale

            states[vid] = dict(x=x, y=y, v=v, a=a, psi=psi, yaw_rate=yaw_rate, hl=hl, hw=hw, L=L, W=W)

            positions.append([x, y])
            ids_for_tree.append(vid)

        if len(ids_for_tree) <= 1:
            self._last_time_s = t_now if t_now is not None else self._last_time_s
            return []

        current_npcs = [vid for vid in ids_for_tree if vid != ego_id]
        if not hasattr(self, "_brake_seen_npcs"):
            self._brake_seen_npcs = set()
        exempt_ratio = float(np.clip(
            getattr(self, "npc_brake_exempt_ratio", 0.35), 0.0, 1.0))
        for vid in current_npcs:
            if vid in self._brake_seen_npcs:
                continue
            self._brake_seen_npcs.add(vid)
            if np.random.random() < exempt_ratio:
                self._brake_random_exempt.add(vid)

        dt_step = dt_sim
        low_speed_threshold = 0.1
        low_speed_exclude_after_s = 3.0
        ego_state = states.get(ego_id)
        active_ids = set(ids_for_tree)
        for vid in list(self._low_speed_pair_streak_s):
            if vid not in active_ids:
                self._low_speed_pair_streak_s.pop(vid, None)
        if ego_state is not None:
            ego_stopped = ego_state["v"] <= low_speed_threshold
            for vid in current_npcs:
                st = states.get(vid)
                if st is None:
                    continue
                if ego_stopped and st["v"] <= low_speed_threshold:
                    streak = self._low_speed_pair_streak_s.get(vid, 0.0) + dt_step
                    self._low_speed_pair_streak_s[vid] = streak
                    if streak >= low_speed_exclude_after_s:
                        self._brake_excluded.add(vid)
                else:
                    self._low_speed_pair_streak_s[vid] = 0.0

        # ---------- KDTree ----------
        tree = KDTree(np.array(positions))
        k_query = min(k_neighbors + 1, len(ids_for_tree))

        # ---------- CTRA离散预测：输出 (x,y,psi) ----------
        def integrate_ctra(x, y, v, a, psi, yaw_rate, horizon, dt):
            """
            返回 shape=(N,3): [x, y, psi] 的预测轨迹
            """
            n = int(max(1, horizon / dt))
            traj = np.zeros((n, 3), dtype=np.float32)
            xx, yy, vv, pp = x, y, v, psi
            for i in range(n):
                vv = max(min_speed, vv + a * dt)
                pp = pp + yaw_rate * dt
                xx = xx + vv * math.cos(pp) * dt
                yy = yy + vv * math.sin(pp) * dt
                traj[i, 0] = xx
                traj[i, 1] = yy
                traj[i, 2] = pp
            return traj

        will_brake = set()

        # 预先算轨迹
        traj_cache = {}
        for vid, st in states.items():
            traj_cache[vid] = integrate_ctra(
                st["x"], st["y"], st["v"], st["a"], st["psi"], st["yaw_rate"],
                horizon=horizon, dt=dt_pred
            )

        # 主循环：对每辆旁车（非Auto）做预测碰撞
        for i, vid in enumerate(ids_for_tree):
            if vid == ego_id:
                continue
            if vid in self._brake_random_exempt:
                continue
            if vid in self._brake_excluded:
                continue
            st_i = states.get(vid)
            if st_i is None:
                continue

            st_auto = states.get(ego_id)
            if st_auto is None:
                continue

            # ===== 你原来的逻辑：Auto 在后方则跳过 =====
            psi_i = st_i["psi"]
            L_i = st_i["L"]
            rear_offset = 0.5 * L_i

            rear_x = st_i["x"] - rear_offset * math.cos(psi_i)
            rear_y = st_i["y"] - rear_offset * math.sin(psi_i)

            dx = st_auto["x"] - rear_x
            dy = st_auto["y"] - rear_y
            s = dx * math.cos(psi_i) + dy * math.sin(psi_i)

            if s <= 0.0:
                continue
            # ===== 新增结束 =====

            # 找最近邻（你现在固定只测 Auto，因此这里保留原结构）
            dists, idxs = tree.query(np.array([st_i["x"], st_i["y"]]), k=k_query)
            if np.isscalar(idxs):
                continue

            neighbor_ids = ["Auto"]

            traj_i = traj_cache[vid]
            hl_i, hw_i = st_i["hl"], st_i["hw"]

            # 逐步预测：任一步矩形相交就触发
            for other_id in neighbor_ids:
                st_j = states.get(other_id)
                if st_j is None:
                    continue

                traj_j = traj_cache[other_id]
                hl_j, hw_j = st_j["hl"], st_j["hw"]

                n = min(len(traj_i), len(traj_j))
                collide = False
                for k in range(n):
                    xi, yi, psii = float(traj_i[k, 0]), float(traj_i[k, 1]), float(traj_i[k, 2])
                    xj, yj, psij = float(traj_j[k, 0]), float(traj_j[k, 1]), float(traj_j[k, 2])

                    if obb_intersect(xi, yi, psii, hl_i, hw_i,
                                    xj, yj, psij, hl_j, hw_j):
                        collide = True
                        break

                if collide:
                    will_brake.add(vid)
                    break


        # ---------- 连续制动剔除机制 ----------
        exclude_after_s = 5.0  # 连续触发 5 秒则剔除
        dt_step = dt_sim       # 用仿真步长累计（如果你用 t_now 更精确也可以）

        # 先把这一步没触发制动的车 streak 清零
        # （只对目前还在场景里、且未剔除的车）
        for vid in ids_for_tree:
            if vid == ego_id or vid in self._brake_excluded:
                continue
            if vid not in will_brake:
                self._brake_streak_s[vid] = 0.0

        # 对触发制动的车累计 streak，达到阈值则剔除
        for vid in will_brake:
            if vid in self._brake_excluded:
                continue
            self._brake_streak_s[vid] = self._brake_streak_s.get(vid, 0.0) + dt_step
            if self._brake_streak_s[vid] >= exclude_after_s:
                self._brake_excluded.add(vid)

        # ---------- GUI 可视化：恢复上一帧不再制动的车辆颜色 ----------
        current_ids = set(traci.vehicle.getIDList())

        for vid in list(self._last_brake_vis_set):
            if vid not in will_brake and vid in current_ids:
                try:
                    old_color = self._brake_vis_original_color.get(vid, (255, 255, 255, 255))
                    traci.vehicle.setColor(vid, old_color)
                except Exception:
                    pass

        # ---------- GUI 可视化：触发紧急制动的车辆标红 ----------
        for vid in will_brake:
            try:
                if vid in current_ids:
                    if vid not in self._brake_vis_original_color:
                        self._brake_vis_original_color[vid] = traci.vehicle.getColor(vid)

                    # 红色：正在紧急制动
                    traci.vehicle.setColor(vid, (255, 0, 0, 255))
            except Exception:
                pass

        # ---------- GUI 可视化：被剔除车辆标灰 ----------
        for vid in self._brake_excluded:
            try:
                if vid in current_ids:
                    # 灰色：连续制动超过阈值，已剔除
                    traci.vehicle.setColor(vid, (128, 128, 128, 255))
            except Exception:
                pass

        self._last_brake_vis_set = set(will_brake)

        # ---------- 下发控制 ----------
        for vid in will_brake:
            try:
                traci.vehicle.setAcceleration(vid, -float(decel), 1.0)
            except Exception:
                pass

        # 更新时间戳
        self._last_time_s = t_now if t_now is not None else (self._last_time_s + dt_sim)

        return list(will_brake)





class PID_posi:
    """位置式实现
    """
    def __init__(self, target, upper, lower, k=[1., 0., 0.]):
        self.kp, self.ki, self.kd = k

        self.e = 0  # error
        self.pre_e = 0  # previous error
        self.sum_e = 0  # sum of error

        self.target = target  # target
        self.upper_bound = upper    # upper bound of output
        self.lower_bound = lower    # lower bound of output

    def set_target(self, target):
        self.target = target

    def set_k(self, k):
        self.kp, self.ki, self.kd = k

    def set_bound(self, upper, lower):
        self.upper_bound = upper
        self.lower_bound = lower

    def cal_output(self, obs):   # calculate output
        self.e = self.target - obs

        u = self.e * self.kp + self.sum_e * \
            self.ki + (self.e - self.pre_e) * self.kd
        if u < self.lower_bound:
            u = self.lower_bound
        elif u > self.upper_bound:
            u = self.upper_bound

        self.pre_e = self.e
        self.sum_e += self.e
        # print(self.sum_e)
        return u

    def reset(self):
        # self.kp = 0
        # self.ki = 0
        # self.kd = 0

        self.e = 0
        self.pre_e = 0
        self.sum_e = 0
        # self.target = 0

    def set_sum_e(self, sum_e):
        self.sum_e = sum_e
