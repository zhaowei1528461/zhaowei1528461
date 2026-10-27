import math
import numpy as np

def _default_shape_from_obj(obj):
    if "shape" in obj and isinstance(obj["shape"], dict):
        L = float(obj["shape"].get("length", 4.5))
        W = float(obj["shape"].get("width", 1.9))
    else:
        L = float(obj.get("length", 4.5))
        W = float(obj.get("width", 1.9))
    return {"length": L, "width": W}

def _clip(v, lo, hi):
    return float(min(max(v, lo), hi))

def predictor_future_only(obs_trajectory, planning_horizon, *, dt=0.1, vmax_by_type=None):
    """
    恒加速度 + 恒航向角预测（仅未来，不含 t=0）。
    - planning_horizon=3.0, dt=0.1 -> 严格返回 30 步（0.1..3.0）
    - 输出字段：id,type,pos_list,orientation_list,v_list,shape,dt
    """
    # 严格按 dt 刻度构造未来时间网格（不含 0）
    steps = int(round(planning_horizon / dt))          # 3.0/0.1 加1 = 31
    t_future = dt * np.arange(1, steps + 1, dtype=float)  # [0.1..3.0] 共30个

    results = []
    for obj in obs_trajectory:
        oid   = str(obj.get("id", "unknown"))
        otype = str(obj.get("type", "vehicle"))
        x0, y0 = map(float, obj["pos"])
        v0  = float(obj.get("v", 0.0))
        a0  = float(obj.get("a", 0.0))
        yaw = float(obj.get("yaw", 0.0))   # 弧度
        shape = _default_shape_from_obj(obj)

        # 简单类型限速（可按需调整）
        vmax = (vmax_by_type or {}).get(otype)
        if vmax is None:
            vmax = 8.0 if otype in ("bicycle", "pedestrian") else 30.0

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        Tf = len(t_future)                 # 未来步数
        pos_list = np.zeros((Tf, 2), dtype=float)
        v_list   = np.zeros((Tf,), dtype=float)
        yaw_list = np.full((Tf,), yaw, dtype=float)


        for i, t in enumerate(t_future):
    # 1) 停车时间（仅减速时有效）
            if a0 < 0 and v0 > 0:
                t_stop = v0 / (-a0)
            else:
                t_stop = float("inf")

            # 有效时间：若减速到0则不再前进
            t_eff = min(t, t_stop)

            # 2) 达到限速的时间（仅加速且 v0<vmax 时有效）
            if a0 > 0 and v0 < vmax:
                t_sat = (vmax - v0) / a0
            else:
                t_sat = float("inf")

             # 3) 位移：分段（加速段 + 匀速段）
            t1 = min(t_eff, t_sat)                    # 仍处于加速/减速段的时间
            s  = v0 * t1 + 0.5 * a0 * t1 * t1
            if t_eff > t_sat:                         # 已经达到限速后，剩余时间匀速
                s += vmax * (t_eff - t_sat)

            # 4) 速度：分段（先 v0+a0*t，再饱和/停车）
            v = v0 + a0 * t_eff
            if t > t_sat:                             # 达到限速后保持 vmax
                 v = vmax
            if t > t_stop:                            # 停车后速度为 0
                v = 0.0

    # 5) 投影到全局坐标 & 赋值
            x = x0 + s * cos_y
            y = y0 + s * sin_y

            pos_list[i] = (x, y)
            v_list[i]   = _clip(v, 0.0, vmax)

        results.append({
            "id": oid,
            "type": otype,
            "pos_list": pos_list,              # 形状: (30, 2)
            "orientation_list": yaw_list,      # 形状: (30,)
            "v_list": v_list,                  # 形状: (30,)
            "shape": shape,
            "dt": float(dt),
        })

    return results
