import numpy as np
from scipy.stats import chi2


class TrajectoryRiskEvaluator:
    def __init__(self, bayes_weight=10, equality_weight=10, maximin_weight=10, ego_weight=15):
        self.bayes_weight = float(bayes_weight)
        self.equality_weight = float(equality_weight)
        self.maximin_weight = float(maximin_weight)
        self.ego_weight = float(ego_weight)

    def total_costs(self, ego_traj, obj_predictions):
        ego_risk_max, obj_risk_max, ego_harm_max, obj_harm_max, top_risk_cost = self.cal_risk(
            ego_traj=ego_traj,
            obj_traj=obj_predictions,
        )
        if ego_risk_max is None or obj_predictions is None:
            return 1e-6

        bayes_cost = self.get_bayesian_costs(ego_risk_max, obj_risk_max)
        equality_cost = self.get_equality_costs(ego_risk_max, obj_risk_max)
        maximin_cost = self.get_maximin_costs(
            ego_risk_max,
            obj_risk_max,
            ego_harm_max,
            obj_harm_max,
        )
        ego_cost = self.get_ego_costs(ego_risk_max)
        total_risk_cost = (
            self.bayes_weight * bayes_cost
            + self.equality_weight * equality_cost
            + self.maximin_weight * maximin_cost
            + self.ego_weight * ego_cost
            + top_risk_cost
        )
        return float(np.clip(total_risk_cost, 0, 2))

    def get_bayesian_costs(self, ego_risk_max, obj_risk_max, boundary_harm=0):
        if len(ego_risk_max) == 0:
            return 0.0
        sum_ego_risk = float(np.sum([np.max(v) for v in ego_risk_max.values()]))
        sum_obj_risk = float(np.sum([np.max(v) for v in obj_risk_max.values()]))
        return (sum_ego_risk + sum_obj_risk + boundary_harm) / (len(ego_risk_max) * 2)

    def get_equality_costs(self, ego_risk_max, obj_risk_max):
        if len(ego_risk_max) == 0:
            return 0.0
        return float(
            sum(abs(float(ego_risk_max[key]) - float(obj_risk_max[key])) for key in ego_risk_max)
            / len(ego_risk_max)
        )

    def get_maximin_costs(
        self,
        ego_risk_max,
        obj_risk_max,
        ego_harm_max,
        obj_harm_max,
        boundary_harm=0,
        eps=1e-9,
        scale_factor=100,
    ):
        if len(ego_harm_max) == 0:
            return 0.0
        maximin_ego = [a * int(b < eps) for a, b in zip(ego_harm_max.values(), ego_risk_max.values())]
        maximin_obj = [a * int(b < eps) for a, b in zip(obj_harm_max.values(), obj_risk_max.values())]
        return float(max(maximin_ego + maximin_obj + [boundary_harm]) ** scale_factor)

    def get_ego_costs(self, ego_risk_max, boundary_harm=0):
        if len(ego_risk_max) == 0:
            return 0.0
        sum_ego_risk = float(np.sum([np.max(v) for v in ego_risk_max.values()]))
        return sum_ego_risk + boundary_harm

    def cal_risk(self, ego_traj, obj_traj):
        coll_prob_dict = self.get_collision_probability(traj=ego_traj, obj_prediction=obj_traj)
        ego_harm_traj, obj_harm_traj = self.cal_harm(ego_traj=ego_traj, obj_prediction=obj_traj)
        if ego_harm_traj is None or obj_harm_traj is None:
            return None, None, None, None, 0.0

        ego_risk_max = {}
        obj_risk_max = {}
        ego_harm_max = {}
        obj_harm_max = {}
        risk_cost = 0.0

        for key in ego_harm_traj:
            ego_risk_traj = [
                ego_harm_traj[key][t] * coll_prob_dict[key][t]
                for t in range(len(ego_harm_traj[key]))
            ]
            obj_risk_traj = [
                obj_harm_traj[key][t] * coll_prob_dict[key][t]
                for t in range(len(obj_harm_traj[key]))
            ]
            ego_risk_max[key] = float(max(ego_risk_traj)) if ego_risk_traj else 0.0
            obj_risk_max[key] = float(max(obj_risk_traj)) if obj_risk_traj else 0.0
            ego_harm_max[key] = float(max(ego_harm_traj[key])) if len(ego_harm_traj[key]) else 0.0
            obj_harm_max[key] = float(max(obj_harm_traj[key])) if len(obj_harm_traj[key]) else 0.0

        return ego_risk_max, obj_risk_max, ego_harm_max, obj_harm_max, risk_cost

    def cal_harm(self, ego_traj, obj_prediction):
        ego_mass = 1500.0
        ego_pos = np.array(ego_traj["pos_list"], dtype=float)
        ego_v = np.array(ego_traj["v_list"], dtype=float)
        ego_yaw = np.array(ego_traj["orientation_list"], dtype=float)
        ego_harm_traj = {}
        obj_harm_traj = {}

        for obj in obj_prediction or []:
            obj_type = obj["type"]
            pred_path = np.array(obj["pos_list"], dtype=float)
            pred_length = min(len(ego_pos), len(pred_path))
            pred_v = np.array(obj["v_list"], dtype=float)[:pred_length]
            pred_yaw = np.array(obj["orientation_list"], dtype=float)[:pred_length]
            obstacle_mass = self.get_obstacle_mass(obj_type)
            ego_harm_fun, obstacle_harm_fun = self.get_harm_model(obj_type=obj_type)

            pdof_array = pred_yaw - ego_yaw[:pred_length]
            delta_v_array = np.sqrt(
                ego_v[:pred_length] ** 2
                + pred_v ** 2
                + 2 * ego_v[:pred_length] * pred_v * np.cos(pdof_array)
            )
            ego_delta_v = (obstacle_mass / (ego_mass + obstacle_mass)) * delta_v_array
            obstacle_delta_v = (ego_mass / (ego_mass + obstacle_mass)) * delta_v_array

            ego_harm_traj[obj["id"]] = ego_harm_fun(velocity=ego_delta_v)
            obj_harm_traj[obj["id"]] = obstacle_harm_fun(velocity=obstacle_delta_v)

        return ego_harm_traj, obj_harm_traj

    def get_harm_model(self, obj_type):
        obstacle_protection = {
            "vehicle": True,
            "car": True,
            "pedestrian": False,
            "bicycle": False,
        }
        protection_flag = obstacle_protection.get(obj_type, None)
        if protection_flag is True:
            ego_harm = lambda velocity: 1 / (1 + np.exp(5.820 - 0.292 * np.maximum(10, velocity)))
            obstacle_harm = lambda velocity: 1 / (1 + np.exp(5.820 - 0.292 * np.maximum(10, velocity)))
        elif protection_flag is False:
            ego_harm = lambda velocity: 1 / (1 + np.exp(3.164 - 0.292 * velocity))
            obstacle_harm = lambda velocity: 1 / (1 + np.exp(1.786 - 0.259 * velocity))
        else:
            ego_harm = lambda velocity: 1.0
            obstacle_harm = lambda velocity: 1.0
        return ego_harm, obstacle_harm

    def get_obstacle_mass(self, obstacle_type):
        if obstacle_type in ("vehicle", "car"):
            return 1000.0
        if obstacle_type == "bicycle":
            return 85.0
        return 70.0

    def get_collision_probability(self, traj, obj_prediction):
        collision_prob_dict = {}
        ego_traj_pose = np.array(traj["pos_list"], dtype=float)
        pred_length = len(ego_traj_pose)
        traj_x = ego_traj_pose[:, 0][:pred_length]
        traj_y = ego_traj_pose[:, 1][:pred_length]
        ego_length = traj["shape"]["length"]
        ego_width = traj["shape"]["width"]

        valid_obj_predictions = [obj for obj in (obj_prediction or []) if "orientation_list" in obj]
        for obj in valid_obj_predictions:
            obj_id = obj["id"]
            obj_mean_list = obj["pos_list"]
            obj_yaw_list = obj["orientation_list"]
            obj_length = obj["shape"]["length"]
            obj_width = obj["shape"]["width"]
            obj_probs = []

            for i in range(len(traj_x)):
                if i >= len(obj_mean_list):
                    break
                ego_pos = [traj_x[i], traj_y[i]]
                ego_yaw = traj["orientation_list"][i]
                mean = obj_mean_list[i]
                obj_yaw = obj_yaw_list[i]

                ego_corners = self.get_four_corners(ego_pos, ego_length, ego_width, ego_yaw)
                obj_corners = self.get_four_corners(mean, obj_length, obj_width, obj_yaw)
                if self.separating_axis_theorem(ego_corners, obj_corners):
                    prob = self.prob_with_mahalanobis_chi_squared(ego_pose=ego_pos, obj_pose=mean)
                else:
                    prob = 0.0
                obj_probs.append(prob)
            collision_prob_dict[obj_id] = obj_probs

        return collision_prob_dict

    def get_four_corners(self, position, length, width, angle):
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=float,
        )
        half_length = length / 2.0
        half_width = width / 2.0
        corners = np.array(
            [
                [-half_length, -half_width],
                [half_length, -half_width],
                [half_length, half_width],
                [-half_length, half_width],
            ],
            dtype=float,
        )
        rotated_corners = np.array([rotation_matrix.dot(corner) for corner in corners], dtype=float)
        return rotated_corners + np.array(position, dtype=float)

    def separating_axis_theorem(self, corners1, corners2, tolerance=0.1, early_exit=True):
        if early_exit:
            center1 = np.mean(corners1, axis=0)
            center2 = np.mean(corners2, axis=0)
            half_diagonal1 = np.sqrt(np.sum((corners1[0] - corners1[2]) ** 2)) / 2
            half_diagonal2 = np.sqrt(np.sum((corners2[0] - corners2[2]) ** 2)) / 2
            center_distance = np.linalg.norm(center1 - center2)
            if center_distance > (half_diagonal1 + half_diagonal2 + tolerance):
                return False

        axes = np.array(
            [
                corners1[1] - corners1[0],
                corners1[3] - corners1[0],
                corners2[1] - corners2[0],
                corners2[3] - corners2[0],
            ],
            dtype=float,
        )
        axes = np.array([axis / np.linalg.norm(axis) for axis in axes], dtype=float)
        for axis in axes:
            projections1 = np.dot(corners1, axis)
            projections2 = np.dot(corners2, axis)
            if max(projections1) < min(projections2) - tolerance or max(projections2) < min(projections1) - tolerance:
                return False
        return True

    def prob_with_mahalanobis_chi_squared(self, ego_pose, obj_pose, collision_threshold=0.0):
        del collision_threshold
        d_m = np.linalg.norm(np.array(ego_pose, dtype=float) - np.array(obj_pose, dtype=float))
        prob = 1 - chi2.cdf(d_m ** 2, df=2)
        return float(min(prob, 1.0))
