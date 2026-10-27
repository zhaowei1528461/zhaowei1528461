import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from guidefn.helperfunction import select_adv_agent


def _try_imports():
    try:
        import gymnasium as gym  # noqa: F401
        from metadrive.envs.metadrive_env import MetaDriveEnv  # noqa: F401
        from stable_baselines3 import PPO  # noqa: F401
        from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing runtime dependencies. Please install: metadrive, gymnasium, stable-baselines3"
        ) from exc


@dataclass
class KinematicState:
    x: float
    y: float
    yaw: float
    speed: float


def _safe_get(obj, names: Iterable[str], default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _extract_xy(vehicle) -> Optional[Tuple[float, float]]:
    pos = _safe_get(vehicle, ["position", "pos", "last_position"])
    if pos is None:
        return None
    if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
        return float(pos[0]), float(pos[1])
    return None


def _extract_yaw(vehicle) -> float:
    yaw = _safe_get(vehicle, ["heading_theta", "yaw", "heading"])
    if isinstance(yaw, (float, int, np.floating)):
        return float(yaw)
    if hasattr(yaw, "theta"):
        return float(yaw.theta)
    return 0.0


def _extract_speed(vehicle) -> float:
    spd = _safe_get(vehicle, ["speed_km_h", "speed"])
    if isinstance(spd, (float, int, np.floating)):
        if hasattr(vehicle, "speed_km_h"):
            return float(spd) / 3.6
        return float(spd)
    vel = _safe_get(vehicle, ["velocity"])
    if isinstance(vel, (list, tuple, np.ndarray)) and len(vel) >= 2:
        return float(np.linalg.norm(np.array(vel[:2], dtype=np.float32)))
    return 0.0


def _iter_traffic_vehicles(env) -> List[object]:
    engine = _safe_get(env, ["engine"])
    if engine is None:
        return []
    tm = _safe_get(engine, ["traffic_manager"])
    if tm is None:
        return []

    cand = _safe_get(tm, ["traffic_vehicles", "vehicles"], default=[])
    if isinstance(cand, dict):
        return list(cand.values())
    if isinstance(cand, (list, tuple)):
        return list(cand)
    return []


def _apply_action_to_vehicle(vehicle, steering: float, throttle: float) -> None:
    action = np.array([steering, throttle], dtype=np.float32)
    if hasattr(vehicle, "set_action"):
        vehicle.set_action(action)
        return
    if hasattr(vehicle, "before_step"):
        vehicle.before_step(action)
        return
    if hasattr(vehicle, "set_control"):
        vehicle.set_control(action)
        return
    if hasattr(vehicle, "set_steering") and hasattr(vehicle, "set_throttle_brake"):
        vehicle.set_steering(float(steering))
        vehicle.set_throttle_brake(float(throttle))


def _rollout_constant_velocity(state: KinematicState, horizon: int, dt: float) -> np.ndarray:
    traj = np.zeros((horizon, 4), dtype=np.float32)
    x, y, yaw, v = state.x, state.y, state.yaw, state.speed
    for t in range(horizon):
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        traj[t] = np.array([x, y, math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    return traj


class AdversarialTrafficController:
    """Closed-loop adversarial controller for MetaDrive background traffic."""

    def __init__(
        self,
        horizon: int = 12,
        dt: float = 0.1,
        throttle: float = 0.6,
        steering_gain: float = 1.2,
        max_attackers: int = 1,
        collision_type: str = "any",
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.throttle = throttle
        self.steering_gain = steering_gain
        self.max_attackers = max_attackers
        self.collision_type = collision_type

    def _build_states(self, env) -> Tuple[Optional[KinematicState], List[Tuple[object, KinematicState]]]:
        ego = _safe_get(env, ["vehicle"])
        if ego is None:
            return None, []
        ego_xy = _extract_xy(ego)
        if ego_xy is None:
            return None, []
        ego_state = KinematicState(
            x=ego_xy[0],
            y=ego_xy[1],
            yaw=_extract_yaw(ego),
            speed=_extract_speed(ego),
        )

        states = []
        for veh in _iter_traffic_vehicles(env):
            xy = _extract_xy(veh)
            if xy is None:
                continue
            states.append(
                (
                    veh,
                    KinematicState(
                        x=xy[0],
                        y=xy[1],
                        yaw=_extract_yaw(veh),
                        speed=_extract_speed(veh),
                    ),
                )
            )
        return ego_state, states

    def _select_attackers(self, ego_state: KinematicState, others: List[Tuple[object, KinematicState]]) -> List[int]:
        if not others:
            return []

        # traj shape: NA x T x 4, where idx=0 is ego
        all_traj = [_rollout_constant_velocity(ego_state, self.horizon, self.dt)]
        for _, st in others:
            all_traj.append(_rollout_constant_velocity(st, self.horizon, self.dt))
        traj = torch.tensor(np.stack(all_traj, axis=0), dtype=torch.float32)

        attacker_idx = select_adv_agent(
            traj=traj,
            ego_idx=0,
            collision_type=self.collision_type,
            use_bicycle_model=False,
            return_probs=False,
        )
        if attacker_idx <= 0:
            return []

        # Map from global idx in traj to idx in others list.
        primary = attacker_idx - 1
        if self.max_attackers <= 1:
            return [primary]

        # Add nearest vehicles as secondary attackers.
        dists = []
        for i, (_, st) in enumerate(others):
            d = math.hypot(st.x - ego_state.x, st.y - ego_state.y)
            dists.append((d, i))
        dists.sort(key=lambda x: x[0])
        chosen = [primary]
        for _, idx in dists:
            if idx == primary:
                continue
            chosen.append(idx)
            if len(chosen) >= self.max_attackers:
                break
        return chosen

    def _compute_control(self, attacker: KinematicState, ego: KinematicState) -> Tuple[float, float]:
        dx = ego.x - attacker.x
        dy = ego.y - attacker.y
        target_yaw = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(target_yaw - attacker.yaw), math.cos(target_yaw - attacker.yaw))
        steering = float(np.clip(self.steering_gain * yaw_err, -1.0, 1.0))
        throttle = float(np.clip(self.throttle, -1.0, 1.0))
        return steering, throttle

    def apply(self, env) -> Dict[str, float]:
        ego_state, traffic_states = self._build_states(env)
        if ego_state is None or len(traffic_states) == 0:
            return {"attackers": 0.0}

        attacker_indices = self._select_attackers(ego_state, traffic_states)
        if len(attacker_indices) == 0:
            return {"attackers": 0.0}

        for idx in attacker_indices:
            veh, st = traffic_states[idx]
            steer, thr = self._compute_control(st, ego_state)
            _apply_action_to_vehicle(veh, steer, thr)

        return {"attackers": float(len(attacker_indices))}


class AdversarialMetaDriveWrapper:
    def __init__(self, env, controller: AdversarialTrafficController, adv_penalty: float = 0.0):
        self.env = env
        self.controller = controller
        self.adv_penalty = adv_penalty
        self._last_attackers = 0.0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action):
        info_adv = self.controller.apply(self.env)
        self._last_attackers = info_adv.get("attackers", 0.0)
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError("Unexpected step output format from MetaDrive env.")
        reward = reward - self.adv_penalty * self._last_attackers
        info["adv_attackers"] = self._last_attackers
        return obs, reward, terminated, truncated, info

    def __getattr__(self, item):
        return getattr(self.env, item)


def parse_args():
    parser = argparse.ArgumentParser("MetaDrive closed-loop adversarial RL training")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="./out/metadrive_closed_loop")
    parser.add_argument("--model-name", type=str, default="ppo_metadrive_adv")

    parser.add_argument("--traffic-density", type=float, default=0.2)
    parser.add_argument("--num-scenarios", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=1000)

    parser.add_argument("--attack-horizon", type=int, default=12)
    parser.add_argument("--attack-dt", type=float, default=0.1)
    parser.add_argument("--attack-throttle", type=float, default=0.6)
    parser.add_argument("--attack-steering-gain", type=float, default=1.2)
    parser.add_argument("--attack-max-vehicles", type=int, default=1)
    parser.add_argument("--collision-type", type=str, default="any", choices=["any", "rear_end", "head_on", "side"])
    parser.add_argument("--adv-penalty", type=float, default=0.0)
    return parser.parse_args()


def build_env_fn(rank: int, args):
    def _make():
        from metadrive.envs.metadrive_env import MetaDriveEnv

        cfg = {
            "use_render": False,
            "manual_control": False,
            "traffic_density": args.traffic_density,
            "num_scenarios": args.num_scenarios,
            "horizon": args.horizon,
            "random_agent_model": False,
            "start_seed": args.seed + rank * 10_000,
        }
        base_env = MetaDriveEnv(config=cfg)
        controller = AdversarialTrafficController(
            horizon=args.attack_horizon,
            dt=args.attack_dt,
            throttle=args.attack_throttle,
            steering_gain=args.attack_steering_gain,
            max_attackers=args.attack_max_vehicles,
            collision_type=args.collision_type,
        )
        return AdversarialMetaDriveWrapper(base_env, controller, adv_penalty=args.adv_penalty)

    return _make


def main():
    _try_imports()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = DummyVecEnv([build_env_fn(i, args) for i in range(args.num_envs)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(args.save_dir, "tb"),
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    save_path = os.path.join(args.save_dir, f"{args.model_name}.zip")
    model.save(save_path)
    print(f"[done] model saved to: {save_path}")
    env.close()


if __name__ == "__main__":
    main()
