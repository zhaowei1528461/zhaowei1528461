import os
import time
import random

os.environ["PATH"] += os.pathsep + "/usr/bin/graphviz"
import sys

# choose which device to use
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

from sumolib import checkBinary
import traci.constants as tc
import traci

import numpy as np
import pandas as pd
import math
from .saferule import safe

class Sumoenv:
    def __init__(self,path="/home/hmx/ZW/onsite_dreamer_zw1/onsite_dreamer/planner/SafeDreamer/embodied/map/map/shizilukou.sumocfg"):

        self.path = path
        self.action_dim = 1
        self.state_dim = 39
        self.egoid = 'Auto'
        self.desSpeed = 7  # desired speed
        self.i = 0
        self.max_x = 50.0
        self.max_y = 50.0
        self.other_speed_max = 13.0
        self.ego_speed_max = 15.0

    def _get_features(self, egoid=None, Range=1000, last_state=None):
        '''
        Calculate features of surrounding vehicle together with self-features.
        args:
            egoid - feature of vehicle (str) ,defalut = None
            Range - perception range (int), default = 50009
         returen:
            Arry ,shape=(state_dim,)
        '''
        if egoid == None:
            egoid = self.egoid
        try:
            ego_road = traci.vehicle.getRoadID(egoid)
            ego_lane = traci.vehicle.getLaneIndex(egoid)
            ego_speed = traci.vehicle.getSpeed(egoid)
            ego_positionx = traci.vehicle.getPosition(egoid)[0]
            ego_positiony = traci.vehicle.getPosition(egoid)[1]
        except:
            ego_positionx = last_state[0] * self.max_x
            ego_positiony = last_state[1] * self.max_y
            ego_speed = last_state[2] * self.ego_speed_max
        temp_data = pd.DataFrame(
            columns=['veh_id', 'positionx', 'positiony', 'velocity', 'road', 'lane', 'distance',
                     'priority', 'velocity_y', 'angle'])

        i = 0
        VehicleIds = traci.vehicle.getIDList()
        for vehId in VehicleIds:
            info = {}
            if vehId != egoid:  # 排除�?动驾驶车辆id
                traci.vehicle.subscribe(
                    vehId, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ROAD_ID, tc.VAR_LANE_INDEX,
                            tc.VAR_SPEED_LAT, tc.VAR_ROUTE_ID, tc.VAR_ANGLE))
                info = traci.vehicle.getAllSubscriptionResults()[vehId]
                if abs(info[tc.VAR_POSITION][0] - ego_positionx) < 150:  # �?读取在range范围内车辆信�?
                    if info[tc.VAR_ROUTE_ID] in {'Straight31', 'Straight32', 'Straight34'}:
                        distance1 = info[tc.VAR_POSITION][0] - 7.2
                        distance = math.sqrt((ego_positionx - info[tc.VAR_POSITION][0]) ** 2 + (
                                ego_positiony - info[tc.VAR_POSITION][1]) ** 2)
                        priority = False
                        if distance1 > 0:
                            if info[tc.VAR_SPEED] != 0:
                                if distance1 <= 30 or abs(distance1) / info[tc.VAR_SPEED] <= 3:
                                    priority = True
                                else:
                                    priority = False
                            else:
                                if distance1 <= 30:
                                    priority = True
                                else:
                                    priority = False
                        temp_data.loc[i] = [
                            vehId, info[tc.VAR_POSITION][0], info[tc.VAR_POSITION][1], info[tc.VAR_SPEED],
                            info[tc.VAR_ROAD_ID], info[tc.VAR_LANE_INDEX], distance, priority,
                            info[tc.VAR_SPEED_LAT], info[tc.VAR_ANGLE]]
                        i += 1

                    else:
                        distance = math.sqrt((ego_positionx - info[tc.VAR_POSITION][0]) ** 2 + (
                                ego_positiony - info[tc.VAR_POSITION][1]) ** 2)
                        priority = False
                        temp_data.loc[i] = [
                            vehId, info[tc.VAR_POSITION][0], info[tc.VAR_POSITION][1], info[tc.VAR_SPEED],
                            info[tc.VAR_ROAD_ID], info[tc.VAR_LANE_INDEX], distance, priority,
                            info[tc.VAR_SPEED_LAT], info[tc.VAR_ANGLE]]
                        i += 1

        library = temp_data.loc[temp_data['distance'] >= 0]
        x_1, x_2, x_3, x_4, x_5, x_6 = (150 for i in range(6))
        y_1, y_2, y_3, y_4, y_5, y_6 = (150 for i in range(6))
        vx_1, vx_2, vx_3, vx_4, vx_5, vx_6 = (-1 for i in range(6))
        vy_1, vy_2, vy_3, vy_4, vy_5, vy_6 = (-1 for i in range(6))
        angle_1, angle_2, angle_3, angle_4, angle_5, angle_6 = (0 for i in range(6))
        p_1, p_2, p_3, p_4, p_5, p_6 = (False for i in range(6))

        if not library.empty:
            if len(library['distance']) < 6:
                index = library['distance'].sort_values().index
            else:
                index = library['distance'].nsmallest(5).index

            n = len(index)
            if n >= 1:
                x_1 = library['positionx'][index[0]]
                y_1 = library['positiony'][index[0]]
                vx_1 = library['velocity'][index[0]]
                vy_1 = library['velocity_y'][index[0]]
                angle_1 = library['angle'][index[0]]
                p_1 = library['priority'][index[0]]
            if n >= 2:
                x_2 = library['positionx'][index[1]]
                y_2 = library['positiony'][index[1]]
                vx_2 = library['velocity'][index[1]]
                vy_2 = library['velocity_y'][index[1]]
                angle_2 = library['angle'][index[1]]
                p_2 = library['priority'][index[1]]
            if n >= 3:
                x_3 = library['positionx'][index[2]]
                y_3 = library['positiony'][index[2]]
                vx_3 = library['velocity'][index[2]]
                vy_3 = library['velocity_y'][index[2]]
                angle_3 = library['angle'][index[2]]
                p_3 = library['priority'][index[2]]
            if n >= 4:
                x_4 = library['positionx'][index[3]]
                y_4 = library['positiony'][index[3]]
                vx_4 = library['velocity'][index[3]]
                vy_4 = library['velocity_y'][index[3]]
                angle_4 = library['angle'][index[3]]
                p_4 = library['priority'][index[3]]
            if n >= 5:
                x_5 = library['positionx'][index[4]]
                y_5 = library['positiony'][index[4]]
                vx_5 = library['velocity'][index[4]]
                vy_5 = library['velocity_y'][index[4]]
                angle_5 = library['angle'][index[4]]
                p_5 = library['priority'][index[4]]
            if n >= 6:
                x_6 = library['positionx'][index[5]]
                y_6 = library['positiony'][index[5]]
                vx_6 = library['velocity'][index[5]]
                vy_6 = library['velocity_y'][index[5]]
                angle_6 = library['angle'][index[5]]
                p_6 = library['priority'][index[5]]

        state = [ego_positionx / self.max_x, ego_positiony / self.max_y, ego_speed / self.ego_speed_max,
                 x_1 / self.max_x, y_1 / self.max_y, vx_1 / self.other_speed_max, vy_1 / self.other_speed_max,
                 angle_1 / 360.0, p_1,
                 x_2 / self.max_x, y_2 / self.max_y, vx_2 / self.other_speed_max, vy_2 / self.other_speed_max,
                 angle_2 / 360.0, p_2,
                 x_3 / self.max_x, y_3 / self.max_y, vx_3 / self.other_speed_max, vy_3 / self.other_speed_max,
                 angle_3 / 360.0, p_3,
                 x_4 / self.max_x, y_4 / self.max_y, vx_4 / self.other_speed_max, vy_4 / self.other_speed_max,
                 angle_4 / 360.0, p_4,
                 x_5 / self.max_x, y_5 / self.max_y, vx_5 / self.other_speed_max, vy_5 / self.other_speed_max,
                 angle_5 / 360.0, p_5,
                 x_6 / self.max_x, y_6 / self.max_y, vx_6 / self.other_speed_max, vy_6 / self.other_speed_max,
                 angle_6 / 360.0, p_6]
        state = np.array(state)
        return state

    def _step(self, action):

        '''
        update ego vehicle state
        input:
            action - Int ，{ 0�?-2m/s^2,
                            1: -1m/s^2,
                            2:  0m/s^2,
                            3:  1m/s^2,
                            4:  2m/s^2,
                             }
                             修改为连续动作-
        output:-2到2m/s^2
            state - array
            done - bool , True if collision or offroad
        '''
        done, info = False, None
        road_index = traci.vehicle.getRoadID('Auto')

        # last state, backup if egocar is removrd when collision.
        state = self._get_features()
        ego_speed = state[2] * self.ego_speed_max
        ego_positionx = state[0] * self.max_x
        ego_positiony = state[1] * self.max_y
        amax = 2
        stop_line = 7.2
        acc = 0

        collision = traci.simulation.getCollidingVehiclesNumber() > 0
        if collision:
            collision_id = traci.simulation.getCollidingVehiclesIDList()
            if self.egoid in collision_id:
                done = True
                info = 'collision'

        p_unsafe = False
        check_safe = safe('Auto', amax, stop_line)
        # check = check_safe.check()

        # if action == 0:
        #     acc = -2
        # elif action == 1:
        #     acc = -1
        # elif action == 2:
        #     acc = 0
        # elif action == 3:
        #     acc = 1
        # elif action == 4:
        #     acc = 2
        acc = np.clip(2*action, -2.0, 2.0)  # 修改为连续动作空间

        if road_index == "-E2" and ego_positiony >= 20:
            done = True
            info = "goal"

        self.dt = traci.simulation.getDeltaT()
        ego_speed += acc * self.dt
        ego_speed = np.clip(ego_speed, 0, 15)
        if not done:
            traci.vehicle.setSpeed(self.egoid, ego_speed)
            # Upadate all the vehicles. Collision will happen.
            traci.simulationStep()
            vid = traci.vehicle.getIDList()
            if self.egoid not in vid:
                done = True  # Collision happened!
                info = 'collision'

            elif self.check_collision('Auto'):
                done = True
                info = 'collision'
        next_state = state if done else self._get_features()
        time = traci.simulation.getTime()
        if time >= 100.0:
            done = True

        return next_state, done, info, acc, p_unsafe, time

    def _get_reward(self, state, step_over1):
        ego_position_y = state[1] * self.max_y
        ego_v = state[2]
        p_1 = state[8]
        p_2 = state[14]
        p_3 = state[20]
        p_4 = state[26]
        p_5 = state[32]
        p_6 = state[38]

        r_efficiency = ego_v - 0.2  # 修改加入对于车速的考虑
        r_unsafe, r_rule_vi, r_rule_co = (0 for i in range(3))

        if (
                p_1 or p_2 or p_3 or p_4 or p_5 or p_6) and -7.2 <= ego_position_y < -2.84 and step_over1 == 0:
            r_rule_vi = -20
        if (p_1 or p_2 or p_3 or p_4 or p_5 or p_6) and -17.2 < ego_position_y < -7.2:
            r_rule_co = 0.4

        return r_efficiency, r_unsafe, r_rule_vi, r_rule_co

    def step(self, action, step, step_over1, last_acc):
        max_steps = 2e4
        state, done, info, acc, p_unsafe, time = self._step(action)
        ego_speed = state[2] * self.ego_speed_max
        ego_y = state[1] * self.max_y
        r_position = ego_y
        r_collision = 0
        r_goal = 0
        cost1=0
        cost2=0
        cost_risk=0
        if done == True:
            if info == "collision":
                progress_ratio = step / max_steps  # 计算回合进度 (0~1)
                r_collision = - 80 - 2 * ego_speed
                cost1 = 1

            if info == "goal":
                r_goal = 100
        r_jerk = -(abs(last_acc - acc) / 0.2) * 0.1

        r_efficiency, r_unsafe, r_rule_vi, r_rule_co = self._get_reward(state, step_over1)
        if p_unsafe:
            r_unsafe = -0.1

        cost2 = self.cost(state)
        per_agent, cost_risk = self.eval_state_PET_cost_intersection_ego_north_polar(
            state, self.max_x, self.max_y, self.ego_speed_max, self.other_speed_max,
            half_width_a=1.5, half_height_b=3,
            Th=6.0, tau=1.5, k=0.3, tau_max=6.0)
        cost = 0.1*cost2+cost1+0.05*cost_risk

        reward = r_efficiency + r_collision + r_unsafe + \
                 r_goal + r_jerk + r_rule_vi + r_rule_co + 0.01 * r_position
        # print(r_efficiency, r_collision, r_goal, r_jerk, r_rule_vi, r_rule_co)

        # 代价部分的书写

        return state, reward,cost ,done, info

    # --- 工具：按“北零顺时针”把 (v, heading_deg) 分解成 (vx, vy) ---
    def v_heading_to_vxy_north_cw(self, v, heading_deg):
        rad = math.radians(heading_deg)
        vx = v * math.sin(rad)  # 0°→北(+y), 90°→东(+x)
        vy = v * math.cos(rad)
        return np.array([vx, vy], dtype=float)

    # --- 轨迹-矩形占用 与 PET（与你之前的相同，可直接复用） ---
    def line_rect_occupancy_times(self, x0, v, center, a, b, Th=6.0, eps=1e-8):
        x0r = x0 - center
        vx, vy = v[0], v[1]
        t0, t1 = 0.0, Th

        def clip(p, q, t0, t1):
            if abs(p) < eps: return (t0, t1) if q >= 0 else (None, None)
            t = q / p
            if p < 0:  # t >= t
                if t > t1: return None, None
                return max(t0, t), t1
            else:  # t <= t
                if t < t0: return None, None
                return (t0, min(t1, t))

        for p, q in [(+vx, a - x0r[0]), (-vx, a + x0r[0]),
                     (+vy, b - x0r[1]), (-vy, b + x0r[1])]:
            t0, t1 = clip(p, q, t0, t1)
            if t0 is None: return []
        if t1 < t0: return []
        return [t0, t1]

    def pet_at_intersection_rect(self, re0, ve, ro0, vo, center, a, b, Th=6.0):
        def heading_into(x0, v):
            return np.dot(center - x0, v) > 0

        Te = self.line_rect_occupancy_times(re0, ve, center, a, b, Th)
        To = self.line_rect_occupancy_times(ro0, vo, center, a, b, Th)
        closing = heading_into(re0, ve) or heading_into(ro0, vo)
        if len(Te) == 0 or len(To) == 0:
            return np.inf, {"ego": Te, "oth": To}, closing
        ego_enter, ego_exit = Te
        oth_enter, oth_exit = To
        if ego_enter <= oth_enter:
            first_exit, second_enter = ego_exit, oth_enter
        else:
            first_exit, second_enter = oth_exit, ego_enter
        PET = second_enter - first_exit
        if PET < 0: PET = 0.0
        return PET, {"ego": Te, "oth": To}, closing

    def pet_cost(self, PET, closing, tau=1.5, k=0.3, tau_max=6.0):
        if not np.isfinite(PET) or PET >= tau_max or not closing: return 0.0
        x = (tau - PET) / k
        return 1.0 / (1.0 + math.exp(-x))

    # --- ★ 主函数：自车固定朝北；旁车用 (v, angle) 分解 ---
    def eval_state_PET_cost_intersection_ego_north_polar(self,
                                                         state, max_x, max_y, ego_speed_max, other_speed_max,
                                                         half_width_a, half_height_b,
                                                         Th=6.0, tau=1.5, k=0.4, tau_max=6.0
                                                         ):
        s = np.array(state, dtype=float)

        # 自车：朝北 -> vx=0, vy=ego_v
        ego_x = s[0] * max_x
        ego_y = s[1] * max_y
        ego_v = s[2] * ego_speed_max
        re0 = np.array([ego_x, ego_y], dtype=float)
        ve = np.array([0.0, ego_v], dtype=float)
        center_xy = (1.5, 0)
        center = np.array(center_xy, dtype=float)
        per_agent, total_cost = [], 0.0

        base, stride = 3, 6
        for i in range(6):
            xi = s[base + i * stride + 0] * max_x
            yi = s[base + i * stride + 1] * max_y

            # 读取速度“标量”与角度（度）；忽略 vy 槽位（作为占位/无效）
            v_norm = s[base + i * stride + 2]  # 原 vx 槽位：速度标量/other_speed_max
            ang_deg = s[base + i * stride + 4] * 360.0  # 航向角（0°=北, 顺时针）
            v_i = v_norm * other_speed_max
            vo = self.v_heading_to_vxy_north_cw(v_i,ang_deg)

            ro0 = np.array([xi, yi], dtype=float)

            PET, intervals, closing = self.pet_at_intersection_rect(
                re0, ve, ro0, vo,
                center=center, a=half_width_a, b=half_height_b, Th=Th
            )
            cost = self.pet_cost(PET, closing, tau=tau, k=k, tau_max=tau_max)
            total_cost += cost
            per_agent.append({
                "i": i + 1, "PET": float(PET), "cost": float(cost),
                "closing": bool(closing), "intervals": intervals
            })

        return per_agent, float(total_cost)



    def cost(self, state):  # 此处的输入考虑一下state输入是否正确
        ego_v = state[2] * 15
        # 绘制双圈判断条件
        ego_y = state[1] * self.max_y
        if -7.2 < ego_y < 8:
            ego_v = state[2] * 15
            # 自车双圈模型
            ego_x = state[0] * self.max_x
            ego_y = state[1] * self.max_x
            ego_y_1 = state[1] * self.max_y - 1
            ego_y_2 = state[1] * self.max_y - 3.36
            # 取出各车辆航向角并转为弧度
            angle_1 = np.radians(state[7] * 360)
            angle_2 = np.radians(state[13] * 360)
            angle_3 = np.radians(state[19] * 360)
            angle_4 = np.radians(state[25] * 360)
            angle_5 = np.radians(state[31] * 360)
            angle_6 = np.radians(state[37] * 360)

            # 周围六个车辆双圈模型
            x1_1 = state[3] * self.max_x - np.sin(angle_1)
            y1_1 = state[4] * self.max_y - np.cos(angle_1)

            x1_2 = state[3] * self.max_x - 3.36 * np.sin(angle_1)
            y1_2 = state[4] * self.max_y - 3.36 * np.cos(angle_1)

            x2_1 = state[9] * self.max_x - np.sin(angle_2)
            y2_1 = state[10] * self.max_y - np.cos(angle_2)

            x2_2 = state[9] * self.max_x - 3.36 * np.sin(angle_2)
            y2_2 = state[10] * self.max_y - 3.36 * np.cos(angle_2)

            x3_1 = state[15] * self.max_x - np.sin(angle_3)
            y3_1 = state[16] * self.max_y - np.cos(angle_3)

            x3_2 = state[15] * self.max_x - 3.36 * np.sin(angle_3)
            y3_2 = state[16] * self.max_y - 3.36 * np.cos(angle_3)

            x4_1 = state[21] * self.max_x - np.sin(angle_4)
            y4_1 = state[22] * self.max_y - np.cos(angle_4)

            x4_2 = state[21] * self.max_x - 3.36 * np.sin(angle_4)
            y4_2 = state[22] * self.max_y - 3.36 * np.cos(angle_4)

            x5_1 = state[27] * self.max_x - np.sin(angle_5)
            y5_1 = state[28] * self.max_y - np.cos(angle_5)

            x5_2 = state[27] * self.max_x - 3.36 * np.sin(angle_5)
            y5_2 = state[28] * self.max_y - 3.36 * np.cos(angle_5)

            x6_1 = state[33] * self.max_x - np.sin(angle_6)
            y6_1 = state[34] * self.max_y - np.cos(angle_6)

            x6_2 = state[33] * self.max_x - 3.36 * np.sin(angle_6)
            y6_2 = state[34] * self.max_y - 3.36 * np.cos(angle_6)

            # 计算欧氏距离平方
            def euclidean_distance_squared(x1, y1, x2, y2):
                return (x1 - x2) ** 2 + (y1 - y2) ** 2

            # 周围六个车辆到前后坐标的距离平方
            dist1_front = euclidean_distance_squared(ego_x, ego_y_1, x1_1, y1_1)
            dist1_rear = euclidean_distance_squared(ego_x, ego_y_2, x1_2, y1_2)

            dist2_front = euclidean_distance_squared(ego_x, ego_y_1, x2_1, y2_1)
            dist2_rear = euclidean_distance_squared(ego_x, ego_y_2, x2_2, y2_2)

            dist3_front = euclidean_distance_squared(ego_x, ego_y_1, x3_1, y3_1)
            dist3_rear = euclidean_distance_squared(ego_x, ego_y_2, x3_2, y3_2)

            dist4_front = euclidean_distance_squared(ego_x, ego_y_1, x4_1, y4_1)
            dist4_rear = euclidean_distance_squared(ego_x, ego_y_2, x4_2, y4_2)

            dist5_front = euclidean_distance_squared(ego_x, ego_y_1, x5_1, y5_1)
            dist5_rear = euclidean_distance_squared(ego_x, ego_y_2, x5_2, y5_2)

            dist6_front = euclidean_distance_squared(ego_x, ego_y_1, x6_1, y6_1)
            dist6_rear = euclidean_distance_squared(ego_x, ego_y_2, x6_2, y6_2)

            # 检查是否有欧氏距离平方小于7
            all_distances = [
                dist1_front, dist1_rear,
                dist2_front, dist2_rear,
                dist3_front, dist3_rear,
                dist4_front, dist4_rear,
                dist5_front, dist5_rear,
                dist6_front, dist6_rear
            ]

            # 如果任一距离平方小于9.5，则cost = 1，否则cost = 0
            cost = 1 if any(d < 9.5 for d in all_distances) else 0
        else:
            cost = 0


        # 返回计算结果，可以根据需要做进一步处理
        return cost

    def set_speed(self, **kwargs):
        for (k, v) in kwargs.items():
            traci.vehicle.setSpeed(k, v)

    def add_vehicle(self, random_initial=True):

        n = np.random.randint(3, 18)
        m = np.random.randint(1, 4)
        if n > m:
            h = n - m
        else:
            h = 0
        for i in range(m):
            position, speed = 0, 4
            position += np.random.randint(0, 40)
            speed += np.random.randint(0, 3)
            a = np.random.randint(1, 10)
            vehicle_id = 'Vm' + str(i)
            time = str(0)
            if a == 1:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight12', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 2:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight13', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 3:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight14', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 4:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight21', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 5:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight23', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 6:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight24', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 7:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight31', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 8:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight32', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            else:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight34', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')

        for i in range(h):
            position, speed = 0, 4
            # position += np.random.randint(0,0)
            speed += np.random.randint(0, 3)
            a = np.random.randint(1, 10)
            vehicle_id = 'Vh' + str(i)
            time = str(np.random.randint(1, 60))
            if a == 1:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight12', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 2:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight13', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 3:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight14', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 4:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight21', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 5:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight23', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 6:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight24', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 7:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight31', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            elif a == 8:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight32', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')
            else:
                traci.vehicle.add(vehicle_id, typeID='SlowCar', routeID='Straight34', depart=time,
                                  departPos=str(position), departSpeed=str(speed), departLane='0')

    def reset(self, initial=True):
        traci.load(["-c", self.path])  # 加载并连�?sumo，self.path:�?�?
        self.add_vehicle(initial)
        traci.simulationStep()
        VehicleIds = traci.vehicle.getIDList()
        for vehId in VehicleIds:
            traci.vehicle.setLaneChangeMode(vehId, 0)  # 设置换道模式（无�?
            traci.vehicle.setSpeedMode(vehId, 0)  # 设置跟车模式(�?)

        state = self._get_features()
        return state

    def cross_product(self, p1, p2, p3):
        # 计算向量p2-p1和向量p3-p1的叉�?  (x1,y1)=p2-p1  (x2,y2)=p3-p1  cp=x1*y2-x2*y1
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

    def is_intersected(self, p1, p2, p3, p4):
        # 判断线�?�p1p2和线段p3p4�?否相�?
        if self.cross_product(p1, p2, p3) * self.cross_product(p1, p2, p4) < 0 and self.cross_product(p3, p4,
                                                                                                      p1) * self.cross_product(
            p3, p4, p2) < 0:
            return True
        else:
            return False

    def is_rectangle_overlap(self, x1, y1, angle1, width1, height1, x2, y2, angle2, width2,
                             height2):  # 判断两个矩形�?否重�?
        """
        判断两个矩形�?否重�?

         x1: �?丢��?矩形的中心点x坐标
         y1: �?丢��?矩形的中心点y坐标
         angle1: �?丢��?矩形的�?�度（以逆时针方向为正）
         width1: �?丢��?矩形的�?�度
         height1: �?丢��?矩形的高�?
         x2: �?二个矩形的中心点x坐标
         y2: �?二个矩形的中心点y坐标
         angle2: �?二个矩形的�?�度（以逆时针方向为正）
         width2: �?二个矩形的�?�度
         height2: �?二个矩形的高�?
         return: True（重叠）�? False（不重叠�?
        """

        # 将�?�度�?�?为弧�?
        angle1 = math.radians(angle1)
        angle2 = math.radians(angle2)

        # 计算�?丢��?矩形的四�?顶点坐标
        x1_1 = x1 - width1 / 2 * math.cos(angle1) + height1 / 2 * math.sin(angle1)
        y1_1 = y1 - width1 / 2 * math.sin(angle1) - height1 / 2 * math.cos(angle1)
        x2_1 = x1 + width1 / 2 * math.cos(angle1) + height1 / 2 * math.sin(angle1)
        y2_1 = y1 + width1 / 2 * math.sin(angle1) - height1 / 2 * math.cos(angle1)
        x3_1 = x1 + width1 / 2 * math.cos(angle1) - height1 / 2 * math.sin(angle1)
        y3_1 = y1 + width1 / 2 * math.sin(angle1) + height1 / 2 * math.cos(angle1)
        x4_1 = x1 - width1 / 2 * math.cos(angle1) - height1 / 2 * math.sin(angle1)
        y4_1 = y1 - width1 / 2 * math.sin(angle1) + height1 / 2 * math.cos(angle1)

        p1_1 = (x1_1, y1_1)
        p2_1 = (x2_1, y2_1)
        p3_1 = (x3_1, y3_1)
        p4_1 = (x4_1, y4_1)
        S1 = (p1_1, p2_1, p3_1, p4_1, p1_1)

        # 计算�?二个矩形的四�?顶点坐标
        x1_2 = x2 - width2 / 2 * math.cos(angle2) + height2 / 2 * math.sin(angle2)
        y1_2 = y2 - width2 / 2 * math.sin(angle2) - height2 / 2 * math.cos(angle2)
        x2_2 = x2 + width2 / 2 * math.cos(angle2) + height2 / 2 * math.sin(angle2)
        y2_2 = y2 + width2 / 2 * math.sin(angle2) - height2 / 2 * math.cos(angle2)
        x3_2 = x2 + width2 / 2 * math.cos(angle2) - height2 / 2 * math.sin(angle2)
        y3_2 = y2 + width2 / 2 * math.sin(angle2) + height2 / 2 * math.cos(angle2)
        x4_2 = x2 - width2 / 2 * math.cos(angle2) - height2 / 2 * math.sin(angle2)
        y4_2 = y2 - width2 / 2 * math.sin(angle2) + height2 / 2 * math.cos(angle2)

        p1_2 = (x1_2, y1_2)
        p2_2 = (x2_2, y2_2)
        p3_2 = (x3_2, y3_2)
        p4_2 = (x4_2, y4_2)
        S2 = (p1_2, p2_2, p3_2, p4_2, p1_2)
        n = 0
        for i in range(4):
            p1 = S1[i]
            p2 = S1[i + 1]
            for i in range(4):
                p3 = S2[i]
                p4 = S2[i + 1]
                if self.is_intersected(p1, p2, p3, p4):
                    n += 1

        if n == 0:
            return False
        else:
            return True

    def check_collision(self, vehicle_id):
        vehicle_length = traci.vehicle.getLength(vehicle_id) * 0.78
        vehicle_width = traci.vehicle.getWidth(vehicle_id) * 0.78

        # 获取当前车辆的位�?和方�?
        pos_x, pos_y = traci.vehicle.getPosition(vehicle_id)
        angle = traci.vehicle.getAngle(vehicle_id)
        angle1 = math.radians(angle)
        pos_x = pos_x - math.sin(angle1) * 2.18
        pos_y = pos_y - math.cos(angle1) * 2.18

        # 获取当前车辆的��度和加速度
        velocity = traci.vehicle.getSpeed(vehicle_id)
        acceleration = traci.vehicle.getAcceleration(vehicle_id)

        # 获取与当前车辆相交的其他车辆信息
        surrounding_vehicles = traci.vehicle.getIDList()

        for other_vehicle_id in surrounding_vehicles:
            if other_vehicle_id == vehicle_id:
                continue  # 跳过�?�?车辆
            # 获取其他车辆的位�?、方向和长�??
            other_pos_x, other_pos_y = traci.vehicle.getPosition(other_vehicle_id)
            other_angle = traci.vehicle.getAngle(other_vehicle_id)
            other_vehicle_length = traci.vehicle.getLength(other_vehicle_id) * 0.78
            other_vehicle_width = traci.vehicle.getWidth(other_vehicle_id) * 0.78
            other_angle1 = math.radians(other_angle)
            other_pos_x = other_pos_x - math.sin(other_angle1) * 2.18
            other_pos_y = other_pos_y - math.cos(other_angle1) * 2.18

            # 判断�?否存在�?�撞风险
            if self.is_rectangle_overlap(pos_x, pos_y, angle, vehicle_width, vehicle_length, other_pos_x, other_pos_y,
                                         other_angle, other_vehicle_width, other_vehicle_length):
                # print(traci.vehicle.getRouteID(other_vehicle_id))
                return True
        return False

    def start(self, gui=False):
        config_path = self.path
        sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        sumoCmd = [sumoBinary, "-c", config_path]

    # ✅ 使用唯一标签避免并发冲突
        # unique_label = f"conn_{int(time.time() * 1000)}_{random.randint(0, 9999)}"

        if gui:
            sumoCmd += ["--quit-on-end", "--start"]
            print("[DEBUG] Launching SUMO with GUI:", " ".join(sumoCmd))




        traci.start(sumoCmd)

    def close(self):
        traci.close()
