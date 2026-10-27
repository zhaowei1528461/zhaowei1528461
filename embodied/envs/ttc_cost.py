# ttc_cost.py
# -*- coding: utf-8 -*-
import math
from typing import List, Dict, Any, Tuple, Optional

def _wrap_pi(a: float) -> float:
    # wrap to (-pi, pi]
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _circumradius(obj: Dict[str, Any]) -> float:
    """由长宽计算外接圆半径 R = sqrt((L/2)^2 + (W/2)^2)。缺省按类型给默认。"""
    default_by_type = {
        "vehicle": (4.7, 1.8),
        "bicycle": (1.6, 0.6),
        "pedestrian": (0.6, 0.6),
    }
    L = obj.get("length"); W = obj.get("width")
    if L is None or W is None:
        Ld, Wd = default_by_type.get(obj.get("type", "vehicle"), (4.5, 1.8))
        L = L if L is not None else Ld
        W = W if W is not None else Wd
    return math.hypot(float(L) * 0.5, float(W) * 0.5)

def _vxvy(obj: Dict[str, Any]) -> Tuple[float, float]:
    """标量速度 + 航向角(弧度) -> (vx, vy)（全局坐标）。"""
    v = float(obj.get("v", 0.0))
    yaw = float(obj.get("yaw", 0.0))
    return v * math.cos(yaw), v * math.sin(yaw)

class TTCRisk:
    """
    TTC 动态风险计算器（圆-圆碰撞 + 航向门控）。
    - TTC：论文式(15) 的常速圆-圆解析；可选最坏加速度离散逼近。
    - 风险：式(16) 的对数风险；normalize=True 时除以 ln(10)，使 -log(0.1) 对应 1。
    - 门控 gate（纯 2D）:
        gate_mode=None                 -> 不门控（全部算 TTC）
        gate_mode='heading_only'       -> ① 只看航向差
        gate_mode='heading_lateral'    -> ② 航向差 + 横向间距
        gate_mode='heading_lat_closure'-> ③ 航向差 + 横向间距 + 接近性（默认，最稳健）
    """

    def __init__(
        self,
        ttc_max: float = 8.0,
        use_worst_accel: bool = False,
        normalize: bool = True,
        # ---------- 门控参数 ----------
        gate_mode: Optional[str] = "heading_lat_closure",     # ③（默认）；可改为 'heading_only' / 'heading_lateral' / None
        lane_width: float = 3.6,
        margin: float = 1.0,
        oncoming_deg: float = 140.0,  # 判为“明显对向”的阈值（角度越大越容易剔除）
        vclose_min: float = 0.5,      # m/s，闭合速度过小视为不接近
        ttc_keep: float = 6.0,        # s，粗略TTC小于此则强制保留做精确计算
        # ---------- 最坏加速度离散 ----------
        dt: float = 0.1,
        horizon: float = 1,
        amax_map: Optional[Dict[str, float]] = None,
    ):
        self.ttc_max = float(ttc_max)
        self.use_worst_accel = bool(use_worst_accel)
        self.normalize = bool(normalize)

        self.gate_mode = gate_mode
        self.lane_width = float(lane_width)
        self.margin = float(margin)
        self.oncoming_deg = float(oncoming_deg)
        self.vclose_min = float(vclose_min)
        self.ttc_keep = float(ttc_keep)

        self.dt = float(dt)
        self.horizon = float(horizon)
        self.amax_map = amax_map or {"vehicle": 3.0, "bicycle": 1.5, "pedestrian": 0.8}

    # ================== 对外接口 ==================

    def penalty(self, ego_one_traj: Dict[str, Any], obs_only_trajectory: List[Dict[str, Any]]) -> float:
        """返回动态风险惩罚和（可直接作为 cost 的一部分）。"""
        report = self.compute(ego_one_traj, obs_only_trajectory)
        return report["sum_dyn_risk"]

    def compute(self, ego_one_traj: Dict[str, Any], obs_only_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """返回明细 + 汇总，便于调试/可视化。"""
        items = []
        for obs in obs_only_trajectory:
            gated, gate_tag = self._gate(ego_one_traj, obs)
            if gated:
                # 被门控剔除：当作无风险
                items.append({
                    "id": obs.get("id"), "type": obs.get("type"),
                    "ttc": float("inf"), "risk": 0.0, "method": gate_tag,
                    "dist_to_ego": float(obs.get("dist_to_ego", -1.0)),
                    "pos": obs.get("pos"), "v": obs.get("v"), "a": obs.get("a"),
                    "yaw": obs.get("yaw"), "width": obs.get("width"), "length": obs.get("length"),
                })
                continue

            if self.use_worst_accel:
                ttc_val = self._ttc_worst_accel_discrete(ego_one_traj, obs)
                method = "worst_accel"
            else:
                ttc_val = self._ttc_circle_cv(ego_one_traj, obs)
                method = "cv"

            risk = self._dyn_risk_from_ttc(ttc_val)

            items.append({
                "id": obs.get("id"), "type": obs.get("type"),
                "ttc": float(ttc_val), "risk": float(risk), "method": method,
                "dist_to_ego": float(obs.get("dist_to_ego", -1.0)),
                "pos": obs.get("pos"), "v": obs.get("v"), "a": obs.get("a"),
                "yaw": obs.get("yaw"), "width": obs.get("width"), "length": obs.get("length"),
            })

        # 排序与汇总
        items.sort(key=lambda x: (math.inf if not math.isfinite(x["ttc"]) else x["ttc"]))
        min_item = items[0] if items else None
        sum_dyn_risk = float(sum(it["risk"] for it in items))

        return {
            "items": items,
            "min_ttc": (min_item["ttc"] if min_item else math.inf),
            "sum_dyn_risk": sum_dyn_risk,
            "worst_case": min_item,
        }

    # ================== 门控（只用 2D） ==================

    def _gate(self, ego: Dict[str, Any], obs: Dict[str, Any]) -> Tuple[bool, str]:
        """
        返回 (should_gate_out, tag)
        True  -> 剔除，不做 TTC；tag 记录剔除原因
        False -> 不剔除，进入 TTC
        """
        mode = (self.gate_mode or "").lower()
        if mode == "" or mode == "none" or mode is None:
            return False, "no_gate"

        # 航向差（abs in [0, pi]）
        dpsi = abs(_wrap_pi(float(obs.get("yaw", 0.0)) - float(ego.get("yaw", 0.0))))
        oncoming = dpsi >= math.radians(self.oncoming_deg)

        if mode == "heading_only":
            # ① 只看航向差：明显对向 -> 剔除
            return (True, "gated_heading") if oncoming else (False, "pass_heading")

        # 自车局部坐标系（纵向 e，横向 l）
        ex, ey = math.cos(ego["yaw"]), math.sin(ego["yaw"])
        lx, ly = -ey, ex
        dx = obs["pos"][0] - ego["pos"][0]
        dy = obs["pos"][1] - ego["pos"][1]
        # 纵向/横向偏移
        x_lon = dx * ex + dy * ey
        y_lat = dx * lx + dy * ly

        # 行驶走廊半宽（车道半宽 + 外接圆和 + 余量）
        Re = _circumradius(ego)
        Ro = _circumradius(obs)
        half_corridor = self.lane_width * 0.5 + (Re + Ro) + self.margin
        far_side = abs(y_lat) > half_corridor  # 在对向车道/隔离外侧

        if mode == "heading_lateral":
            # ② 航向差 + 横向间距：对向且离走廊很远 -> 剔除
            if oncoming and far_side:
                return True, "gated_heading_lateral"
            return False, "pass_heading_lateral"

        # ③ 航向差 + 横向间距 + 接近性（稳健）
        if oncoming and far_side:
            # 先看是否在接近
            vxi, vyi = _vxvy(ego); vxj, vyj = _vxvy(obs)
            vdx, vdy = vxi - vxj, vyi - vyj
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                return False, "pass_closure_dist0"

            # 径向闭合速度：>0 表示在接近
            v_close = -(dx / dist) * vdx - (dy / dist) * vdy
            if v_close < self.vclose_min:
                return True, "gated_no_closure"

            # 粗略 TTC（线性近似），若很大则剔除
            Rsum = Re + Ro
            ttc_lin = (max(dist - Rsum, 0.0)) / max(v_close, 1e-6)
            if ttc_lin > self.ttc_keep:
                return True, "gated_ttc_lin_large"

        return False, "pass_heading_lat_closure"

    # ================== TTC 实现 ==================

    def _ttc_circle_cv(self, i: Dict[str, Any], j: Dict[str, Any], eps: float = 1e-9) -> float:
        """圆-圆碰撞时间解析解（常速，对应式(15)）：取最小非负根；无解 -> +inf。"""
        xi, yi = i["pos"]; xj, yj = j["pos"]
        vxi, vyi = _vxvy(i); vxj, vyj = _vxvy(j)
        dx, dy = (xi - xj), (yi - yj)
        vdx, vdy = (vxi - vxj), (vyi - vyj)
        Rsum = _circumradius(i) + _circumradius(j)

        a = vdx * vdx + vdy * vdy
        b = 2.0 * (dx * vdx + dy * vdy)
        c = dx * dx + dy * dy - Rsum * Rsum

        if c <= 0.0:
            return 0.0
        if a <= eps:
            return math.inf

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return math.inf

        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        candidates = [t for t in (t1, t2) if t >= 0.0]
        return min(candidates) if candidates else math.inf

    def _ttc_worst_accel_discrete(self, i: Dict[str, Any], j: Dict[str, Any]) -> float:
        """最坏加速度离散逼近：双方沿“朝对方”方向以各自 a_max 加速；首次相交时刻。"""
        xi, yi = map(float, i["pos"]); vxi, vyi = _vxvy(i)
        xj, yj = map(float, j["pos"]); vxj, vyj = _vxvy(j)
        Rsum = _circumradius(i) + _circumradius(j)

        ai = float(self.amax_map.get(i.get("type", "vehicle"), 2.5))
        aj = float(self.amax_map.get(j.get("type", "vehicle"), 2.5))

        t = 0.0
        while t <= self.horizon:
            dx, dy = (xi - xj), (yi - yj)
            if dx * dx + dy * dy <= Rsum * Rsum:
                return t

            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                ax_i = -ai * (dx / dist); ay_i = -ai * (dy / dist)
                ax_j =  aj * (dx / dist); ay_j =  aj * (dy / dist)
            else:
                ax_i = ay_i = ax_j = ay_j = 0.0

            vxi += ax_i * self.dt; vyi += ay_i * self.dt
            vxj += ax_j * self.dt; vyj += ay_j * self.dt
            xi  += vxi * self.dt;  yi  += vyi * self.dt
            xj  += vxj * self.dt;  yj  += vyj * self.dt
            t   += self.dt

        return math.inf

    def _dyn_risk_from_ttc(self, ttc: float) -> float:
        """动态风险：-log( max(0.1, min(TTC/TTCmax, 1)) )；normalize=True 时再 /ln(10)。"""
        if not math.isfinite(ttc):
            return 0.0
        x = max(0.1, min(ttc / max(self.ttc_max, 1e-6), 1.0))
        val = -math.log(x)
        return float(val / math.log(10.0)) if self.normalize else float(val)
# ttc_cost.py
# -*- coding: utf-8 -*-
import math
from typing import List, Dict, Any, Tuple, Optional

def _wrap_pi(a: float) -> float:
    # wrap to (-pi, pi]
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _circumradius(obj: Dict[str, Any]) -> float:
    """由长宽计算外接圆半径 R = sqrt((L/2)^2 + (W/2)^2)。缺省按类型给默认。"""
    default_by_type = {
        "vehicle": (4.7, 1.8),
        "bicycle": (1.6, 0.6),
        "pedestrian": (0.6, 0.6),
    }
    L = obj.get("length"); W = obj.get("width")
    if L is None or W is None:
        Ld, Wd = default_by_type.get(obj.get("type", "vehicle"), (4.5, 1.8))
        L = L if L is not None else Ld
        W = W if W is not None else Wd
    return math.hypot(float(L) * 0.5, float(W) * 0.5)

def _vxvy(obj: Dict[str, Any]) -> Tuple[float, float]:
    """标量速度 + 航向角(弧度) -> (vx, vy)（全局坐标）。"""
    v = float(obj.get("v", 0.0))
    yaw = float(obj.get("yaw", 0.0))
    return v * math.cos(yaw), v * math.sin(yaw)

class TTCRisk:
    """
    TTC 动态风险计算器（圆-圆碰撞 + 航向门控）。
    - TTC：论文式(15) 的常速圆-圆解析；可选最坏加速度离散逼近。
    - 风险：式(16) 的对数风险；normalize=True 时除以 ln(10)，使 -log(0.1) 对应 1。
    - 门控 gate（纯 2D）:
        gate_mode=None                 -> 不门控（全部算 TTC）
        gate_mode='heading_only'       -> ① 只看航向差
        gate_mode='heading_lateral'    -> ② 航向差 + 横向间距
        gate_mode='heading_lat_closure'-> ③ 航向差 + 横向间距 + 接近性（默认，最稳健）
    """

    def __init__(
        self,
        ttc_max: float = 8.0,
        use_worst_accel: bool = False,
        normalize: bool = True,
        # ---------- 门控参数 ----------
        gate_mode: Optional[str] = "heading_only",    # 这个版本只使用航向角剔除对向车辆
        lane_width: float = 3.6,
        margin: float = 1.0,
        oncoming_deg: float = 160.0,  # 判为“明显对向”的阈值（角度越大越容易剔除）
        vclose_min: float = 0.5,      # m/s，闭合速度过小视为不接近
        ttc_keep: float = 6.0,        # s，粗略TTC小于此则强制保留做精确计算
        # ---------- 最坏加速度离散 ----------
        dt: float = 0.05,
        horizon: float = 10.0,
        amax_map: Optional[Dict[str, float]] = None,
    ):
        self.ttc_max = float(ttc_max)
        self.use_worst_accel = bool(use_worst_accel)
        self.normalize = bool(normalize)

        self.gate_mode = gate_mode
        self.lane_width = float(lane_width)
        self.margin = float(margin)
        self.oncoming_deg = float(oncoming_deg)
        self.vclose_min = float(vclose_min)
        self.ttc_keep = float(ttc_keep)

        self.dt = float(dt)
        self.horizon = float(horizon)
        self.amax_map = amax_map or {"vehicle": 3.0, "bicycle": 1.5, "pedestrian": 0.8}

    # ================== 对外接口 ==================

    def penalty(self, ego_one_traj: Dict[str, Any], obs_only_trajectory: List[Dict[str, Any]]) -> float:
        """返回动态风险惩罚和（可直接作为 cost 的一部分）。"""
        report = self.compute(ego_one_traj, obs_only_trajectory)
        # print('report:::',report)
        return report["sum_dyn_risk"]

    def compute(self, ego_one_traj: Dict[str, Any], obs_only_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """返回明细 + 汇总，便于调试/可视化。"""
        items = []
        for obs in obs_only_trajectory:
            gated, gate_tag = self._gate(ego_one_traj, obs)
            if gated:
                # 被门控剔除：当作无风险
                items.append({
                    "id": obs.get("id"), "type": obs.get("type"),
                    "ttc": float("inf"), "risk": 0.0, "method": gate_tag,
                    "dist_to_ego": float(obs.get("dist_to_ego", -1.0)),
                    "pos": obs.get("pos"), "v": obs.get("v"), "a": obs.get("a"),
                    "yaw": obs.get("yaw"), "width": obs.get("width"), "length": obs.get("length"),
                })
                continue

            if self.use_worst_accel:
                ttc_val = self._ttc_worst_accel_discrete(ego_one_traj, obs)
                method = "worst_accel"
            else:
                ttc_val = self._ttc_circle_cv(ego_one_traj, obs)
                method = "cv"

            risk = self._dyn_risk_from_ttc(ttc_val)

            items.append({
                "id": obs.get("id"), "type": obs.get("type"),
                "ttc": float(ttc_val), "risk": float(risk), "method": method,
                "dist_to_ego": float(obs.get("dist_to_ego", -1.0)),
                "pos": obs.get("pos"), "v": obs.get("v"), "a": obs.get("a"),
                "yaw": obs.get("yaw"), "width": obs.get("width"), "length": obs.get("length"),
            })

        # 排序与汇总
        items.sort(key=lambda x: (math.inf if not math.isfinite(x["ttc"]) else x["ttc"]))
        min_item = items[0] if items else None
        sum_dyn_risk = float(sum(it["risk"] for it in items))

        return {
            "items": items,
            "min_ttc": (min_item["ttc"] if min_item else math.inf),
            "sum_dyn_risk": sum_dyn_risk,
            "worst_case": min_item,
        }

    # ================== 门控（只用 2D） ==================

    def _gate(self, ego: Dict[str, Any], obs: Dict[str, Any]) -> Tuple[bool, str]:
        """
        返回 (should_gate_out, tag)
        True  -> 剔除，不做 TTC；tag 记录剔除原因
        False -> 不剔除，进入 TTC
        """
        mode = (self.gate_mode or "").lower()
        if mode == "" or mode == "none" or mode is None:
            return False, "no_gate"

        # 航向差（abs in [0, pi]）
        dpsi = abs(_wrap_pi(float(obs.get("yaw", 0.0)) - float(ego.get("yaw", 0.0))))
        oncoming = dpsi >= math.radians(self.oncoming_deg)

        if mode == "heading_only":
            # ① 只看航向差：明显对向 -> 剔除
            return (True, "gated_heading") if oncoming else (False, "pass_heading")

        # 自车局部坐标系（纵向 e，横向 l）
        ex, ey = math.cos(ego["yaw"]), math.sin(ego["yaw"])
        lx, ly = -ey, ex
        dx = obs["pos"][0] - ego["pos"][0]
        dy = obs["pos"][1] - ego["pos"][1]
        # 纵向/横向偏移
        x_lon = dx * ex + dy * ey
        y_lat = dx * lx + dy * ly

        # 行驶走廊半宽（车道半宽 + 外接圆和 + 余量）
        Re = _circumradius(ego)
        Ro = _circumradius(obs)
        half_corridor = self.lane_width * 0.5 + (Re + Ro) + self.margin
        far_side = abs(y_lat) > half_corridor  # 在对向车道/隔离外侧

        if mode == "heading_lateral":
            # ② 航向差 + 横向间距：对向且离走廊很远 -> 剔除
            if oncoming and far_side:
                return True, "gated_heading_lateral"
            return False, "pass_heading_lateral"

        # ③ 航向差 + 横向间距 + 接近性（稳健）
        if oncoming and far_side:
            # 先看是否在接近
            vxi, vyi = _vxvy(ego); vxj, vyj = _vxvy(obs)
            vdx, vdy = vxi - vxj, vyi - vyj
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                return False, "pass_closure_dist0"

            # 径向闭合速度：>0 表示在接近
            v_close = -(dx / dist) * vdx - (dy / dist) * vdy
            if v_close < self.vclose_min:
                return True, "gated_no_closure"

            # 粗略 TTC（线性近似），若很大则剔除
            Rsum = Re + Ro
            ttc_lin = (max(dist - Rsum, 0.0)) / max(v_close, 1e-6)
            if ttc_lin > self.ttc_keep:
                return True, "gated_ttc_lin_large"

        return False, "pass_heading_lat_closure"

    # ================== TTC 实现 ==================

    def _ttc_circle_cv(self, i: Dict[str, Any], j: Dict[str, Any], eps: float = 1e-9) -> float:
        """圆-圆碰撞时间解析解（常速，对应式(15)）：取最小非负根；无解 -> +inf。"""
        xi, yi = i["pos"]; xj, yj = j["pos"]
        vxi, vyi = _vxvy(i); vxj, vyj = _vxvy(j)
        dx, dy = (xi - xj), (yi - yj)
        vdx, vdy = (vxi - vxj), (vyi - vyj)
        Rsum = _circumradius(i) + _circumradius(j)

        a = vdx * vdx + vdy * vdy
        b = 2.0 * (dx * vdx + dy * vdy)
        c = dx * dx + dy * dy - Rsum * Rsum

        if c <= 0.0:
            return 0.0
        if a <= eps:
            return math.inf

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return math.inf

        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        candidates = [t for t in (t1, t2) if t >= 0.0]
        return min(candidates) if candidates else math.inf

    def _ttc_worst_accel_discrete(self, i: Dict[str, Any], j: Dict[str, Any]) -> float:
        """最坏加速度离散逼近：双方沿“朝对方”方向以各自 a_max 加速；首次相交时刻。"""
        xi, yi = map(float, i["pos"]); vxi, vyi = _vxvy(i)
        xj, yj = map(float, j["pos"]); vxj, vyj = _vxvy(j)
        Rsum = _circumradius(i) + _circumradius(j)

        ai = float(self.amax_map.get(i.get("type", "vehicle"), 2.5))
        aj = float(self.amax_map.get(j.get("type", "vehicle"), 2.5))

        t = 0.0
        while t <= self.horizon:
            dx, dy = (xi - xj), (yi - yj)
            if dx * dx + dy * dy <= Rsum * Rsum:
                return t

            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                ax_i = -ai * (dx / dist); ay_i = -ai * (dy / dist)
                ax_j =  aj * (dx / dist); ay_j =  aj * (dy / dist)
            else:
                ax_i = ay_i = ax_j = ay_j = 0.0

            vxi += ax_i * self.dt; vyi += ay_i * self.dt
            vxj += ax_j * self.dt; vyj += ay_j * self.dt
            xi  += vxi * self.dt;  yi  += vyi * self.dt
            xj  += vxj * self.dt;  yj  += vyj * self.dt
            t   += self.dt

        return math.inf

    def _dyn_risk_from_ttc(self, ttc: float) -> float:
        """动态风险：-log( max(0.1, min(TTC/TTCmax, 1)) )；normalize=True 时再 /ln(10)。"""
        if not math.isfinite(ttc):
            return 0.0
        x = max(0.1, min(ttc / max(self.ttc_max, 1e-6), 1.0))
        val = -math.log(x)
        return float(val / math.log(10.0)) if self.normalize else float(val)
