import math
from typing import Any, Dict, List, Tuple, Optional, Iterable

def _get(entry: Any, key: str, default=None):
    """兼容 dict / 对象 取属性。"""
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)

def ego_from_self_info(ego_info_df) -> Dict[str, Any]:
    """
    从 self.ego_info（单行 DataFrame）构造 ego_traj 字典。
    需要列: x, y, v, a, yaw, width, length
    """
    row = ego_info_df.iloc[0]
    return {
        'id': 'ego',
        'type': 'vehicle',  # 如需保留具体类型可自行修改
        'pos': (float(row['x']), float(row['y'])),
        'v': float(row['v']),
        'a': float(row['a']),
        'yaw': float(row['yaw']),
        'width': float(row['width']),
        'length': float(row['length']),
    }

def build_obs_around_ego(observation,
                         ego_pos: Tuple[float, float],
                         side_types: Iterable[str] = ('vehicle','bicycle','motorcycle','bus','truck','pedestrian'),
                         max_dist: float = 30.0,
                         exclude_ids: Iterable[str] = ('ego',)) -> List[Dict[str, Any]]:
    """
    仅根据 ego 的 (x,y) 从 observation.object_info 里提取旁车。
    不再依赖 observation 里是否存在 'ego'。
    """
    import numpy as np
    def is_valid_number(x):
        """检查是否为有效的数字（不是NaN也不是无穷大）"""
        return isinstance(x, (int, float)) and not math.isinf(x) and not math.isnan(x)




    # print('all_observation',observation)
    ex_ids = set(exclude_ids or [])
    ex_ids.add('ego')  # 防御性排除
    ex_ids = {str(x) for x in ex_ids}

    ego_x, ego_y = ego_pos
    obs_list: List[Dict[str, Any]] = []

    for obj_type, objs in observation.object_info.items():
        if side_types is not None and obj_type not in side_types:
            continue
        for obj_id, entry in objs.items():
            if str(obj_id) in ex_ids:
                continue
            
            # 获取坐标
            x = _get(entry, 'x', None)
            y = _get(entry, 'y', None)
            if x is None or y is None:
                continue

            # 计算距离
            dist = math.hypot(float(x) - ego_x, float(y) - ego_y)
            if dist > max_dist:
                continue

            # 获取需要检查的数值字段
            v = _get(entry, 'v', 0.0)
            a = _get(entry, 'a', 0.0)
            yaw = _get(entry, 'yaw', 0.0)

            # 检查是否有无效值
            if not (is_valid_number(v) and is_valid_number(a) and is_valid_number(yaw)):
                # 有任意一个是 inf/-inf/NaN，就跳过这辆车
                continue

            # 全部正常，加入结果列表
            obs_list.append({
                'id': obj_id,
                'type': obj_type,
                'pos': (float(x), float(y)),
                'v': v,
                'a': np.clip(a,-8,6),
                'yaw': yaw,
                'width': _get(entry, 'width', None),
                'length': _get(entry, 'length', None),
                'dist_to_ego': dist,
            })

    return obs_list