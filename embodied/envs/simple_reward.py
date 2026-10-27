import numpy as np

def calculate_arc_length_to_target(center_vertices: np.ndarray, station: np.ndarray, target_xy: np.ndarray) -> float:

    """
    计算从车道起点到目标点的弧长（单位：米）
    
    参数：
    - center_vertices: 车道中心线坐标，(N, 2) 数组，表示车道的 N 个点
    - station: 累计弧长，(N,) 数组，表示从起点到车道各个点的弧长（单位：米）
    - target_xy: 目标点的 (x, y) 坐标，(2,) 数组
    
    返回值：
    - target_s: 从车道起点到目标点的弧长（单位：米）
    """
    # 1. 计算目标点与车道上所有点的欧氏距离，找到最近点的索引
    distances = np.linalg.norm(center_vertices - target_xy, axis=1)
    nearest_idx = np.argmin(distances)
    
    # 2. 目标点如果与某个点重合，直接返回该点的累计弧长
    if distances[nearest_idx] < 1e-6:
        return station[nearest_idx]

    # 3. 目标点在线段内（通过线性插值）
    # 目标点所在的段：center_vertices[nearest_idx] -> center_vertices[nearest_idx + 1]
    if nearest_idx < len(center_vertices) - 1:
        start_point = center_vertices[nearest_idx]
        end_point = center_vertices[nearest_idx + 1]
        start_station = station[nearest_idx]
        end_station = station[nearest_idx + 1]

        # 计算目标点在该段上的比例 t (0 <= t <= 1)
        segment_vector = end_point - start_point
        target_vector = target_xy - start_point
        t = np.dot(target_vector, segment_vector) / np.dot(segment_vector, segment_vector)

        # 保证 t 的值在 [0, 1] 之间
        t = max(0.0, min(t, 1.0))

        # 计算目标点的弧长坐标
        target_s = start_station + t * (end_station - start_station)
        return target_s
    else:
        # 如果目标点在最后一个点之后，直接返回最后一个点的弧长
        return station[-1]
    

def calculate_car_progress(s0_m: float, target_s: float) -> float:
        """
        计算自车的完成度，即自车当前弧长 s0_m 相对于目标点的弧长 target_s。
        
        参数：
        - s0_m: 当前自车的弧长坐标，(单位：米)
        - target_s: 目标点的弧长坐标，(单位：米)
        返回值：
        - car_completion: 自车的完成度，范围 [0, 1]，表示自车当前位置与目标点之间的进度。
        """
        if target_s > 0:
            car_completion = min(s0_m / target_s, 1.0)  # 完成度不能超过 1
        else:
            car_completion = 0  # 如果目标点的弧长是 0，完成度为 0
        return car_completion




def calculate_penalties(ttc: float, d: float, speed: float, 
                      comfort_violation_count: int,
                      max_speed: float = 20.0, ttc_threshold: float = 2.0):
    """
    计算四个惩罚项
    最高车速根据所在道路类型来确定。
    参数：
    - ttc: 碰撞时间 (s)
    - d: 车辆到车道中心的横向距离 (m)
    - speed: 车辆当前速度 (m/s)
    - comfort_violation_count: 违反舒适度的次数
    - lane_width: 车道宽度 (m)，默认3.7m
    - max_speed: 最大允许速度 (m/s)，默认20m/s
    - ttc_threshold: TTC阈值 (s)，默认2s
    
    返回：
    - p_outside_lane: 车辆偏离车道惩罚 [0,1]
    - p_lane_center: 车辆不在车道中心惩罚 [0,1]
    - p_speed: 速度惩罚 [0,1]
    - p_ttc: TTC惩罚 [0,1]
    """
    # 1. 车辆偏离车道惩罚
    # 如果车辆完全离开车道，p就为0；否则为1

    p_outside_lane = 0 if abs(d) > 3 else 1
    
    # 2. 车辆不在车道中心权重 
    # 离中心越远p越大，使用二次函数映射到[0,1]
    normalized_d = abs(d)/2.5  # 归一化到[0,1]
    p_lane_center = max(0.0, min(1.0 - normalized_d ** 2, 1.0))
    
    # 3. 速度惩罚
    # 超过最大速度则惩罚为1；低于限速则按比例惩罚
    if speed >  max_speed :
        p_speed = 0.8
    elif speed <1:
        p_speed=0.8
    else:
        p_speed = 1
        
    # 4. TTC惩罚
    # TTC小于阈值p为0.5，否则p为1
    if ttc <= ttc_threshold:
        p_ttc = 0.5
    else:
        p_ttc = 1.0
        
    # 5. 舒适度惩罚
    # 违反舒适度的次数越多，惩罚越大
    p_comfort = 1-0.5*comfort_violation_count  #违反舒适性次数越多，p越小
    
    return p_outside_lane, p_lane_center, p_speed, p_ttc, p_comfort





def check_comfort_number(old_a, a, old_raw, raw, dt):
    """
    计算加加速度和角速度，并统计超过阈值的次数
    :param old_a: 上一时刻线加速度 (m/s^2)
    :param a: 当前线加速度 (m/s^2)
    :param old_raw: 上一时刻角速度 (rad/s)
    :param raw: 当前角速度 (rad/s)
    :param dt: 仿真时间步长 (s)
    :return: 违反舒适性的次数 (int)
    """
    # 舒适性阈值（示例值，可根据标准调整）
    JERK_THRESHOLD = 5   # m/s^3
    ANG_ACC_THRESHOLD = 4 # rad/s^2
    # 计算加加速度 (jerk)
    jerk = (a - old_a) / dt
    #为了避免初始化时候的时候角速度较大。
    if old_raw==0:
        old_raw=raw
    
    # 计算角速度
    ang_acc = (raw - old_raw) / dt

    # 统计超过阈值的次数
    violation_count = 0
    if abs(jerk) > JERK_THRESHOLD:
        violation_count += 1
    if abs(ang_acc) > ANG_ACC_THRESHOLD:
        violation_count += 1

    return violation_count




# ---------------------- 新增：综合奖励计算函数（实现 rt = RCt * 各项惩罚项乘积）- T    ----------------------
def simple_rewards(center_vertices: np.ndarray, station: np.ndarray, comfort_violation_count,
                          target_xy: np.ndarray, s0_m: float, ttc: float, 
                          d: float,info: str, speed: float,
                          ) -> tuple:
    """
    计算综合奖励 rt,公式:rt = RCt * (p_outside_lane * p_lane_center * p_speed * p_ttc * p_comfort)
    
    参数：
    - center_vertices: 车道中心线坐标 (N, 2)
    - station: 车道各点累计弧长 (N,)
    - target_xy: 目标点坐标 (2,)
    - s0_m: 自车当前弧长（米）
    - ttc: 碰撞时间s
    - d: 自车到车道中心的横向距离m
    - speed: 自车当前速度m/s

    
    返回：
    - total_reward (rt): 综合奖励
    - RCt: 自车完成度（鼓励向目标点行驶）
    - penalties: 各项惩罚项元组p_outside_lane, p_lane_center, p_speed, p_ttc, p_comfort
    """
    max_speed=6
    ttc_threshold=4.0
    if info == 'goal':
        # 到达目标场景：T设为0（按你的需求定义）
        T = -50
    elif info == 'collision':
        # 发生碰撞场景：T设为1（可根据实际需求调整，如设为-1表示负向终止）
        # 若碰撞时需额外惩罚，可在此处补充（如降低综合奖励）
        T = 5
    elif info =='max_t':
        T = 20
    else:
        T = 0
    # 计算目标点弧长
    target_s = calculate_arc_length_to_target(center_vertices, station, target_xy)
    
    #  计算自车完成度 RCt
    RCt = calculate_car_progress(s0_m, target_s)
    #计算舒适性的完成度

    comfort_violation_count  = comfort_violation_count
    #  计算各项惩罚项
    p_outside_lane, p_lane_center, p_speed, p_ttc, p_comfort = calculate_penalties(
        ttc=ttc, d=d, speed=speed, comfort_violation_count=comfort_violation_count,
        max_speed=max_speed, ttc_threshold=ttc_threshold
    )
    penalties = (p_outside_lane, p_lane_center, p_speed, p_ttc, p_comfort)
    # print(penalties)
    # 4. 计算综合奖励：RCt * 各项惩罚项的乘积-T
    total_reward = RCt * (p_outside_lane * p_lane_center * p_speed * p_ttc * p_comfort)-T
    
    return total_reward, RCt, penalties