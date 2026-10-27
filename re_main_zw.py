import math
import numpy as np
import pandas as pd
import threading
import time
from collections import deque
import functools
from typing import Tuple
# 只使用ROS2
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Imu
# Gym相关导入
from gym import spaces
import embodied
from ruamel.yaml import YAML
import pathlib
from . import agent as agt  # 这里是加载agent
from numpy import ndarray
# 数学工具函数
def wrap_value(value, min_val, max_val):
    """将值包装在指定范围内"""
    range_val = max_val - min_val
    while value < min_val:
        value += range_val
    while value >= max_val:
        value -= range_val
    return value

def position_to_ego_frame(position, ego_position, ego_heading):
    """将位置转换到自车坐标系"""
    x_global, y_global, z_global = position
    x_ego, y_ego, z_ego = ego_position
    heading = ego_heading
    
    # 计算相对位置
    dx = x_global - x_ego
    dy = y_global - y_ego
    
    # 旋转到自车坐标系
    cos_heading = math.cos(heading)
    sin_heading = math.sin(heading)
    
    x_relative = dx * cos_heading + dy * sin_heading
    y_relative = -dx * sin_heading + dy * cos_heading
    
    return np.array([x_relative, y_relative, z_global])
import math
from pyproj import CRS, Transformer

class CoordinateTransformer:
    def __init__(self, reference_lat, reference_lon, net_offset=(0, 0)):
        """
        初始化坐标转换器，与 SUMO 的投影保持一致
        Args:
            reference_lat: 参考点纬度 (度) -> 对应 projParameter 中的 +lat_0
            reference_lon: 参考点经度 (度) -> 对应 projParameter 中的 +lon_0
            net_offset: (x_offset, y_offset)，来自 .net.xml 的 netOffset
        """
        self.ref_lat = reference_lat
        self.ref_lon = reference_lon
        self.net_offset = net_offset

        # 构造与 SUMO 一致的投影定义
        proj4_str = f"+proj=tmerc +lon_0={reference_lon} +lat_0={reference_lat} +ellps=WGS84 +k=1 +x_0=0 +y_0=0"
        crs_geo = CRS.from_epsg(4326)
        crs_map = CRS.from_proj4(proj4_str)

        # 定义 pyproj 转换器
        self.to_map = Transformer.from_crs(crs_geo, crs_map, always_xy=True)
        self.to_geo = Transformer.from_crs(crs_map, crs_geo, always_xy=True)

    def latlon_to_xy(self, lat, lon):
        """经纬度 -> SUMO 仿真坐标"""
        x, y = self.to_map.transform(lon, lat)
        # 加上 netOffset
        x += self.net_offset[0]
        y += self.net_offset[1]
        return x, y

    def xy_to_latlon(self, x, y):
        """SUMO 仿真坐标 -> 经纬度"""
        # 减去 netOffset
        x -= self.net_offset[0]
        y -= self.net_offset[1]
        lon, lat = self.to_geo.transform(x, y)
        return lat, lon
    
    def xy_to_latlon(self, x, y):
        """
        将局部XY坐标转换回经纬度
        """
        ref_lat_rad = math.radians(self.ref_lat)
        ref_lon_rad = math.radians(self.ref_lon)
        
        lon_rad = ref_lon_rad + x / (self.earth_radius * math.cos(ref_lat_rad))
        lat_rad = ref_lat_rad + y / self.earth_radius
        
        return math.degrees(lat_rad), math.degrees(lon_rad)

class RealWorldEnv(Node):
    def __init__(self, state_dim=31, action_dim=1):
        # 初始化ROS2节点
        super().__init__('real_world_env_node')
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.egoid = 'Auto'
        self.max_x = 210  # 坐标范围，单位：米
        self.max_y = 520  # 坐标范围，单位：米
        self.other_speed_max = 10
        self.ego_speed_max = 10
        
        # 坐标转换器
        self.coord_transformer = None
        self.reference_set = False
        
        # 动作空间和观测空间
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'velocity': spaces.Box(-np.inf, np.inf, (31,), dtype=np.float32),
            'is_first': spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': spaces.Box(0, 1, (), dtype=bool),
        })
        
        # ROS2订阅器和发布器
        self.obs_info_sub = None
        self.ego_info_sub = None
        self.ctrl_publisher = None
        
        # 数据存储
        self.latest_obstacles = None
        self.latest_ego_info = None
        self.obstacles_lock = threading.Lock()
        self.ego_info_lock = threading.Lock()
        
        # 状态跟踪
        self.x = 0.0
        self.y = 0.0
        self.head = 0.0
        self.last_valid_ego = None
        
        # 初始化默认参考点
        self.set_reference_point(31.290823210868965, 121.20585769414902)
        
        # print('128')
        # 初始化ROS2组件
        self.init_ros2()
        # print('131')
        
    def init_ros2(self):
        """完全重写的ROS2初始化"""
        # print('init')
        try:
            print("🚀 Initializing ROS2 components...")
            
            # 首先检查ROS2上下文是否有效
            if not rclpy.ok():
                print("❌ ROS2 context not valid!")
                return False
            
            # 创建订阅器 - 使用更可靠的qos配置
            qos_profile = rclpy.qos.QoSProfile(
                depth=10,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                durability=rclpy.qos.DurabilityPolicy.VOLATILE
            )
            
            # 创建障碍物信息订阅器
            self.obs_info_sub = self.create_subscription(
                MarkerArray,
                '/obs_info',
                self.obstacle_callback,
                qos_profile
            )
            # print("✅ Created subscriber for /obs_info")
            
            # 创建自车信息订阅器 - 使用更宽松的配置
            self.ego_info_sub = self.create_subscription(
                Imu,
                '/global_info', 
                self.ego_info_callback,
                qos_profile
            )
            # print("✅ Created subscriber for /global_info")
            
            # 立即检查订阅器状态
            time.sleep(0.5)  # 给ROS2一些时间
            
            # 检查订阅器计数
            # print(f"📊 Subscriber count for /global_info: {self.count_subscribers('/global_info')}")
            # print(f"📊 Subscriber count for /obs_info: {self.count_subscribers('/obs_info')}")
            
            # 创建发布器
            self.ctrl_publisher = self.create_publisher(
                Imu,
                '/RL_ctrl_info',
                10
            )
            # print("✅ Created publisher for RL_ctrl_info")
            
            # print("🎉 ROS2 initialization completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ ROS2 initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def set_reference_point(self, lat, lon):
        """设置坐标转换的参考点"""
        self.coord_transformer = CoordinateTransformer(lat, lon, net_offset=(434.76, 265.57))
        self.reference_set = True
        # print(f"Reference point set: lat={lat:.6f}, lon={lon:.6f}")

    def obstacle_callback(self, msg):
        """ROS2障碍物信息回调函数"""
        obstacles = []
        for marker in msg.markers:
            obstacle = {
                'id': marker.id,
                'positionx': marker.pose.position.x,  # 经度
                'positiony': marker.pose.position.y,  # 纬度
                'positionz': marker.pose.position.z,
                'velocity': marker.pose.position.z,  # 如果没有速度信息，设为0
                'angle': marker.pose.orientation.x,
                'scale_x': marker.scale.x,
                'scale_y': marker.scale.y,
                'timestamp': time.time()
            }
            obstacles.append(obstacle)
        
        with self.obstacles_lock:
            self.latest_obstacles = obstacles

    def ego_info_callback(self, msg):
        """ROS2自车信息回调函数"""
        # print(f"Ego info callback received!")  # 调试信息
        # print(f"Message type: {type(msg)}")
        
        with self.ego_info_lock:
            self.latest_ego_info = {
                'msg': msg,
                'timestamp': time.time()
            }
        # print(f"Ego info updated at {self.latest_ego_info['timestamp']}")

    def get_current_obstacles(self):
        """获取当前障碍物信息"""
        with self.obstacles_lock:
            if self.latest_obstacles is not None:
                return self.latest_obstacles.copy()
            return []

    def get_current_ego_info(self):
        """获取当前自车信息 - 修复版本"""
        with self.ego_info_lock:
            if self.latest_ego_info is not None:
                return self.latest_ego_info['msg'], self.latest_ego_info['timestamp']
        
        # 如果没有收到自车信息，返回默认值
        # print("⚠️ No ego info received yet, using default values")
        return None, 0

    def get_dushu(self, x, y):
        """计算相对于正北方向的角度（度）"""
        point0 = [122.10, 326.37]   # 参考点1
        point1 = [120.83, 327.92]   # 参考点2（正北方向）
        A = point0
        B = point1
        C = point0
        D = [x, y]

        AB = (B[0] - A[0], B[1] - A[1])
        CD = (D[0] - C[0], D[1] - C[1])
        
        dot_product = AB[0] * CD[0] + AB[1] * CD[1]
        
        length_AB = math.sqrt(AB[0]**2 + AB[1]**2)
        length_CD = math.sqrt(CD[0]**2 + CD[1]**2)
        
        cos_theta = dot_product / (length_AB * length_CD)
        cos_theta = max(-1, min(1, cos_theta))
        
        angle_rad = math.acos(cos_theta)
        
        cross_product = AB[0] * CD[1] - AB[1] * CD[0]
        
        if cross_product < 0:
            angle_rad = 2 * math.pi - angle_rad
        
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg

    def get_state(self):
        """
        获取当前状态（使用经纬度坐标转换）
        返回格式与训练时相同的状态向量
        """
        # 获取最新的自车信息
        ego_msg, ego_timestamp = self.get_current_ego_info()

        # 使用默认值或上一次的有效值
        if ego_msg is None:
            print("⚠️ Using default ego info")
            if self.last_valid_ego is not None:
                ego_x, ego_y, ego_speed, ego_angle = self.last_valid_ego
            else:
                # 默认值
                ego_x, ego_y = 0, 0
                ego_speed = 2.0
                ego_angle = 234.37
        else:
            # print('11111111111111')
            # 从global_info消息中提取自车信息（经纬度格式）
            ego_lat = ego_msg.orientation_covariance[0]  # 30.51472787
            ego_lon = ego_msg.orientation_covariance[1]  # 114.33284932000001
            ego_angle = ego_msg.orientation_covariance[2]  # 234.37
            ego_speed = ego_msg.orientation.w  # 从角速度估算速度
            # print('ego_speed', ego_speed)
            
            # 将经纬度转换为XY坐标（米）
            ego_x, ego_y = self.coord_transformer.latlon_to_xy(ego_lat, ego_lon)
            
            # 检查数据有效性
            if math.isnan(ego_x) or math.isnan(ego_y):
                if self.last_valid_ego is not None:
                    ego_x, ego_y, ego_speed, ego_angle = self.last_valid_ego
                else:
                    ego_x, ego_y = 0, 0
                    ego_speed = 2.0
                    ego_angle = 234.37
        
        # 保存有效值
        self.last_valid_ego = (ego_x, ego_y, ego_speed, ego_angle)
        
        self.x = ego_x
        self.y = ego_y
        self.jiaodu = self.get_dushu(ego_x, ego_y)
        self.head = self.adjust_angle(ego_angle)
        
        # 获取最新的障碍物信息
        current_obstacles = self.get_current_obstacles()
        # print('current_obstacles', current_obstacles)
        
        # 创建临时数据容器
        temp_data = pd.DataFrame(
            columns=['veh_id', 'positionx', 'positiony', 'velocity', 'distance', 'angle', 'jiaodu'])
        
        # 处理实际障碍物信息（经纬度转XY）
        i = 0
        for obstacle in current_obstacles:
            # 障碍物的经纬度坐标
            obs_lon = obstacle['positionx']  # 经度
            obs_lat = obstacle['positiony']  # 纬度
            
            # 转换为XY坐标
            obs_x, obs_y = self.coord_transformer.latlon_to_xy(obs_lat, obs_lon)
            # print(obs_x, obs_y)
            
            # 计算障碍物与自车的距离
            distance = math.sqrt((ego_x - obs_x) ** 2 + (ego_y - obs_y) ** 2)
            
            # 只考虑在感知范围内的障碍物（150米）
            if distance < 150:
                temp_data.loc[i] = [
                    obstacle['id'],
                    obs_x,  # 使用转换后的X坐标
                    obs_y,  # 使用转换后的Y坐标
                    obstacle['velocity'],
                    distance,
                    obstacle['angle'],
                    self.get_dushu(obs_x, obs_y) - self.jiaodu
                ]
                i += 1
        
        # 初始化状态变量
        x_1, x_2, x_3, x_4, x_5, x_6 = (0 for i in range(6))
        y_1, y_2, y_3, y_4, y_5, y_6 = (0 for i in range(6))
        vx_1, vx_2, vx_3, vx_4, vx_5, vx_6 = (0 for i in range(6))
        vy_1, vy_2, vy_3, vy_4, vy_5, vy_6 = (0 for i in range(6))
        angle_1, angle_2, angle_3, angle_4, angle_5, angle_6 = (0 for i in range(6))
        
        # 选择最近的6个障碍物
        if not temp_data.empty:
            if len(temp_data['distance']) < 6:
                index = temp_data['distance'].sort_values().index
            else:
                index = temp_data['distance'].nsmallest(6).index
            
            n = len(index)
            
            # 处理每个障碍物的坐标转换
            for i in range(min(n, 6)):
                x = temp_data['positionx'][index[i]]  # 已经是转换后的X坐标
                y = temp_data['positiony'][index[i]]  # 已经是转换后的Y坐标
                velocity = temp_data['velocity'][index[i]]
                angle = temp_data['angle'][index[i]]
                
                # 坐标转换到自车坐标系（相对坐标）
                vehicle_pos = self.transform(np.append([x, y], [0]))[:2]
                
                # 航向角调整
                head = self.adjust_angle(angle)
                adjusted_heading = self.adjust_heading(head)
                
                # 速度分解
                vx = velocity * np.cos(adjusted_heading)
                vy = velocity * np.sin(adjusted_heading)
                
                # 赋值给对应的状态变量
                if i == 0:
                    x_1, y_1, vx_1, vy_1, angle_1 = vehicle_pos[0], vehicle_pos[1], vx, vy, adjusted_heading
                elif i == 1:
                    x_2, y_2, vx_2, vy_2, angle_2 = vehicle_pos[0], vehicle_pos[1], vx, vy, adjusted_heading
                elif i == 2:
                    x_3, y_3, vx_3, vy_3, angle_3 = vehicle_pos[0], vehicle_pos[1], vx, vy, adjusted_heading
                elif i == 3:
                    x_4, y_4, vx_4, vy_4, angle_4 = vehicle_pos[0], vehicle_pos[1], vx, vy, adjusted_heading
                elif i == 4:
                    x_5, y_5, vx_5, vy_5, angle_5 = vehicle_pos[0], vehicle_pos[1], vx, vy, adjusted_heading
                elif i == 5:
                    x_6, y_6, vx_6, vy_6, angle_6 = vehicle_pos[0], vehicle_pos[1], vx, vy, adjusted_heading
        
        # 构建状态向量（与训练时完全一致）
        state = [ego_speed/self.ego_speed_max,
                 x_1/self.max_x, y_1/self.max_y, vx_1/self.other_speed_max, vy_1/self.other_speed_max, angle_1/math.pi, 
                 x_2/self.max_x, y_2/self.max_y, vx_2/self.other_speed_max, vy_2/self.other_speed_max, angle_2/math.pi,
                 x_3/self.max_x, y_3/self.max_y, vx_3/self.other_speed_max, vy_3/self.other_speed_max, angle_3/math.pi,
                 x_4/self.max_x, y_4/self.max_y, vx_4/self.other_speed_max, vy_4/self.other_speed_max, angle_4/math.pi,
                 x_5/self.max_x, y_5/self.max_y, vx_5/self.other_speed_max, vy_5/self.other_speed_max, angle_5/math.pi,
                 x_6/self.max_x, y_6/self.max_y, vx_6/self.other_speed_max, vy_6/self.other_speed_max, angle_6/math.pi,]

        state = np.array(state, dtype=np.float32)
        # print('state', state)
        state1 = [ego_x/self.max_x, ego_y/self.max_y]
        
        return state, state1

    def transform(self, v):
        """坐标转换到自车坐标系（相对坐标）"""
        return position_to_ego_frame(v, [self.x, self.y, 0], self.head)

    def adjust_heading(self, h):
        """调整航向角"""
        return wrap_value(h - self.head, -math.pi, math.pi)
    
    def adjust_angle(self, angle):
        """调整角度格式"""
        angle -= 90
        if angle != 180 and angle != 0:
            angle = angle * -1
        return (angle*math.pi)/180

    def to_obs(self, state):
        """转换为观测格式"""
        return {
            'velocity': np.expand_dims(np.array(state, dtype=np.float32), 0),
            'is_first': np.array([False]),
            'is_terminal': np.array([False])
        }
    
    def publish_action(self, action):
        """发布action到RL_ctrl_info话题（Imu版）"""
        if self.ctrl_publisher is not None:
            # try:
                # 兼容不同容器：把 action 转成 list
                if isinstance(action, np.ndarray):
                    action_list = action.tolist()
                elif hasattr(action, '__len__'):
                    action_list = list(action)
                else:
                    action_list = [float(action)]

                # 保护：至少要有 action[1]
                if len(action_list) < 1:
                    raise ValueError("action 维度不足，至少需要1维（包含 action[1]）")

                # 构造 Imu 消息
                msg = Imu()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "base_link"
                action_list = [float(x) for x in action_list]
                print('action_list', action_list)
                # 根据你的需求：同时把 action[1] 写入 orientation.x 和 orientation.y
                target_v = ((action_list[0] + 1) / 2.0) * 5.0
                lane_change_cmd = 0.0
                msg.orientation.x = target_v
                msg.orientation.y = lane_change_cmd
                # 其余字段若无需求，填 0
                msg.orientation.z = 1.0
                msg.orientation.w = 0.0
                msg.angular_velocity.x = 0.0
                msg.angular_velocity.y = 0.0
                msg.angular_velocity.z = 0.0
                msg.linear_acceleration.x = 0.0
                msg.linear_acceleration.y = 0.0
                msg.linear_acceleration.z = 0.0

                # 可选：设置协方差为未知（-1）或 0
                msg.orientation_covariance[0] = -1.0
                msg.angular_velocity_covariance[0] = -1.0
                msg.linear_acceleration_covariance[0] = -1.0

                # 发布
                self.ctrl_publisher.publish(msg)
                # print(f"Published Imu: orientation.x/y set to {action}")

            # except Exception as e:
            #     print(f"Error publishing Imu action: {e}")
        else:
            print("ROS2 publisher not available")

    # def execute_action(self, action):
    #     """
    #     执行动作并发布到控制话题
    #     """
    #     # 发布action到ROS2话题
    #     print('action', action)
    #     self.publish_action(action)
        
    #     # 解析动作信息
    #     target_v = ((action[0] + 1) / 2) * 3  # 0-3m/s
    #     lane_change_cmd = action[1]
        
    #     control_info = {
    #         'target_speed': f"{target_v:.2f}m/s",
    #         'lane_change': 'left' if lane_change_cmd < -0.8 else 
    #                       'right' if lane_change_cmd > 0.8 else 'keep',
    #         'raw_action': action.tolist() if isinstance(action, np.ndarray) else list(action)
    #     }
    #     print(f"Control command executed: {control_info}")

    def execute_action(self, action):
        """
        执行动作并发布到控制话题
        """
        # 发布action到ROS2话题
        
        # print('476action', action)
        # 解析动作信息
        "预留动作的滤波部分 action = cbf(actiosn)"
        target_v = ((action[0] + 1) / 2) * 5  # 0-3m/s
        lane_change_cmd = 0.0
        self.publish_action(action)
        control_info = {
            'target_speed': f"{target_v:.2f}m/s",
            'lane_change': 'left' if lane_change_cmd < -0.8 else 
                          'right' if lane_change_cmd > 0.8 else 'keep',
            'raw_action': action.tolist() if isinstance(action, np.ndarray) else list(action)
        }
        
        # print(f"Control command executed: {control_info}")
    def run_inference_loop(self, model, frequency=10):
        """
        运行推理循环
        Args:
            model: 训练好的强化学习模型
            frequency: 推理频率 (Hz)
        """
        # print(f"Starting inference loop at {frequency}Hz...")
        
        # 创建定时器
        timer_period = 1.0 / frequency
        self.timer = self.create_timer(timer_period, self.inference_step)
        
        self.model = model
        self.step_count = 0
        
        # print("Inference loop started. Press Ctrl+C to stop.")

    def inference_step(self):
        """推理步骤回调"""
        try:
            self.step_count += 1
            # print(f"\n--- Step {self.step_count} ---")
            
            # 1. 获取当前状态（包含坐标转换）
            state, state1 = self.get_state()
            print(569, state) #meiwwnti
            obs = self.to_obs(state)
            # print(571, obs) #nengdayin
            # 2. 模型推理
            action = self.model.predict(obs)
            # print(574, action)
            # 3. 执行动作
            self.execute_action(action[0])
            
            # 4. 显示状态信息
            ego_speed = state[0] * self.ego_speed_max
            # print(f"Ego speed: {ego_speed:.2f}m/s, State shape: {state.shape}")
            
        except Exception as e:
            print(f"Error in inference step: {e}")

    def get_single_state(self):
        """获取单次状态（用于测试）"""
        state, state1 = self.get_state()
        return state

    def close(self):
        """关闭系统"""
        # print("Closing real world environment...")
        if hasattr(self, 'timer'):
            self.destroy_timer(self.timer)
        self.destroy_node()
        rclpy.shutdown()






# 测试用的随机模型

class RandomModel:
    def __init__(self,observation_space,action_space, action_dim=1):
        self.action_dim = action_dim
        self.observation_space = observation_space
        self.action_space = action_space
        self.config, self.args, self.logdir_algo = load_config()   
        # print('self.config',self.config)
        # print('self.args',self.args)
        # print('self.logdir_algo',self.logdir_algo)
        self.logdir = embodied.Path(self.logdir_algo)
        logdir = embodied.Path(self.args.logdir)
        logdir.mkdirs()
        self.config.save(logdir / 'config.yaml')
        self.step_counter = embodied.Counter()
        # self.logger = 0      #这两项应该不重要？
        self.dataset = 0     #这两项应该不重要？
        self.step = 0

        #加载模型
        self.checkpoint = embodied.Checkpoint(self.logdir / 'checkpoint.ckpt')
        self.checkpoint.step = self.step_counter
            #首次加载储存经验地址
        if getattr(self.args, 'from_checkpoint', None):
            self.checkpoint.load(self.args.from_checkpoint)
        self.checkpoint.load_or_save()
        # self.logger = make_logger(self.logdir, self.step_counter , self.config)
        #   创建pidlag
        self.lag = PIDLagrangian(self.config)
       
        # 调用agent训练
        self.agent = agt.Agent(self.observation_space, self.action_space , self.step, self.config)  
        self.policy_eval = lambda *args: self.agent.policy(*args, mode='eval')
        self.state = None

    def predict(self, obs):
        is_first = False
        """生成随机动作用于测试"""
        # 详细计时
        start_time = time.time()
        
        # 模型前向传播
        model_start = time.time()
        acts, self._state = self.policy_eval(obs, self._state, **self._kwargs)
        # print('out', out)
        model_time = time.time() - model_start
        
        # 动作提取
        action_start = time.time()
        action = acts['action']
        action_time = time.time() - action_start
        
        total_time = time.time() - start_time
        
        print(f"模型前向传播: {model_time*1000:.2f}ms, 动作处理: {action_time*1000:.2f}ms, 总计: {total_time*1000:.2f}ms")
        return action


def load_config(argv=None):
    """
    加载和解析 SafeDreamer 的配置文件，并生成最终 args。

    参数：
        argv: list，可选，命令行参数（默认 None）

    返回：
        config: embodied.Config 对象，包含算法和环境配置
        args:   embodied.Config 对象，包含运行时参数（logdir 等）

        调用的方法：config, args = load_config(argv)
    """
    INTERNAL_DEBUG_ARGS = [
    "--configs", "osrp_onsite_lag_vector",
    "--method", "osrp_onsite_lag_vector",
    "--run.script", "eval",
    # "--run.from_checkpoint", "/home/hmx/ZW/onsite_dreamer_zw_62/onsite_dreamer/log_huandao_orsp/2025_10_8_1_cost_dense/checkpoint.ckpt",
    "--task", "onsite_None",
    "--jax.platform cpu"  ,
    "--run.steps", "10000",]
    # 可选：调试更容易断点
    # "--debug.no_jit",

    if argv is None:
        argv = sys.argv[1:]
    if  len(argv) == 0:
        argv = INTERNAL_DEBUG_ARGS[:]


    # 设置渲染后端（可选）
    os.environ["MUJOCO_GL"] = "egl"

    # 1. 解析命令行参数
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)

    # 2. 加载默认配置
    config = embodied.Config(agt.Agent.configs['defaults'])

    # 3. 根据命令行 --configs 指定的配置名叠加更新
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])

    # 4. 再解析命令行其他参数，覆盖前面配置
    config = embodied.Flags(config).parse(other)

    # 5. 生成带时间戳的 logdir
    now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir_algo = (
            config.logdir
            + now_time
            + '_' + str(config.method)
            + '_' + str(config.task)
            + '_' + str(config.seed)
    )

    logdir_algo = "/home/hmx/ZW/onsite_dreamer_zw_59/onsite_dreamer/log_huandao_orsp/2025_10_8_1_cost_dense_reward_simple_58"
    # logdir_algo = "/home/hmx/ZW/onsite_dreamer_zw_62/onsite_dreamer/log_test/2025_10_5_2_cost_dense_reward_simple"
    print('logdir_algo 222',logdir_algo)
    # 6. 封装运行参数
    args = embodied.Config(
        **config.run,
        logdir=logdir_algo,
        use_cost=config.use_cost,
        batch_steps=config.batch_size * config.batch_length,
    )
    # print('config666',config)
    return config, args, logdir_algo


# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class PIDLagrangian():  # noqa: B024 output 为self._cost_penalty, self._pid_d, self._pid_i, self._delta_p
    """PID version of Lagrangian.

    Similar to the :class:`Lagrange` module, this module implements the PID version of the
    lagrangian method.

    .. note::
        The PID-Lagrange is more general than the Lagrange, and can be used in any policy gradient
        algorithm. As PID_Lagrange use the PID controller to control the lagrangian multiplier, it
        is more stable than the naive Lagrange.

    Args:
        pid_kp (float): The proportional gain of the PID controller.
        pid_ki (float): The integral gain of the PID controller.
        pid_kd (float): The derivative gain of the PID controller.
        pid_d_delay (int): The delay of the derivative term.
        pid_delta_p_ema_alpha (float): The exponential moving average alpha of the delta_p.
        pid_delta_d_ema_alpha (float): The exponential moving average alpha of the delta_d.
        sum_norm (bool): Whether to use the sum norm.
        diff_norm (bool): Whether to use the diff norm.
        penalty_max (int): The maximum penalty.
        lagrangian_multiplier_init (float): The initial value of the lagrangian multiplier.
        cost_limit (float): The cost limit.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        - URL: `PID Lagrange <https://arxiv.org/abs/2007.03964>`_
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
            self,
            config,
    ) -> None:
        """Initialize an instance of :class:`PIDLagrangian`."""
        self._pid_kp: float = config.pid.kp
        self._pid_ki: float = config.pid.ki
        self._pid_kd: float = config.pid.kd
        self._pid_d_delay = config.pid.d_delay
        self._pid_delta_p_ema_alpha: float = config.pid.delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = config.pid.delta_d_ema_alpha
        self._penalty_max: int = config.pid.penalty_max
        self._sum_norm: bool = config.pid.sum_norm
        self._diff_norm: bool = config.pid.diff_norm
        self._pid_i: float = config.pid.lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._pid_d: float = 0.0
        self._cost_limit: float = config.cost_limit
        self._cost_penalty: float = config.pid.init_penalty
        self._use_cost_decay: bool = config.pid.use_cost_decay
        self._current_cost_limit: float = config.pid.init_cost_limit
        if self._use_cost_decay:
            self._steps = [config.pid.decay_time_step * (i + 1) for i in range(config.pid.decay_num)]
            self._limits = [max(config.pid.init_cost_limit - i * config.pid.decay_limit_step, config.cost_limit) for i
                            in range(config.pid.decay_num)]

    @property
    def lagrange_penalty(self) -> float:
        """The lagrangian multiplier."""
        return self._cost_penalty

    @property
    def delta_p(self) -> float:
        """The lagrangian multiplier p."""
        return self._delta_p

    @property
    def pid_i(self) -> float:
        """The lagrangian multiplier i."""
        return self._pid_i

    @property
    def pid_d(self) -> float:
        """The lagrangian multiplier d."""
        return self._pid_d

    def pid_update(self, epcost, step) -> Tuple[ndarray, float, float, float]:
        r"""Update the PID controller.

        PID controller update the lagrangian multiplier following the next equation:

        .. math::

            \lambda_{t+1} = \lambda_t + (K_p e_p + K_i \int e_p dt + K_d \frac{d e_p}{d t}) \eta

        where :math:`e_p` is the error between the current episode cost and the cost limit,
        :math:`K_p`, :math:`K_i`, :math:`K_d` are the PID parameters, and :math:`\eta` is the
        learning rate.

        Args:
            ep_cost_avg (float): The average cost of the current episode.
        """
        ep_cost_avg = epcost
        if self._use_cost_decay:
            for i, threshold in enumerate(self._steps):
                if step < threshold:
                    self._current_cost_limit = self._limits[i]
                    break
            else:
                self._current_cost_limit = self._cost_limit
        else:
            self._current_cost_limit = self._cost_limit

        delta = float(ep_cost_avg - self._current_cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        self._pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * self._pid_d
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)
        self._cost_penalty = np.clip(self._cost_penalty, 0.0, self._penalty_max)
        return self._cost_penalty, self._pid_d, self._pid_i, self._delta_p


def make_logger(logdir, step, config):
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score|episode/cost'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ], multiplier)
    return logger


# 主函数
def main():
    # 初始化ROS2
    rclpy.init()
    # 创建真实世界环境
    env = RealWorldEnv()
    obs_space=env.observation_space
    action_space = env.action_space
    # 创建随机模型用于测试
    model = RandomModel(obs_space,action_space,action_dim=1)
    
    #接入真实模型
    # try:
        # 测试单次状态获取
    # print("Testing state retrieval...")
    state = env.get_single_state()
    # print(f"State shape: {state.shape}")
    
    # 运行推理循环
    # print("\nStarting inference loop...")
    env.run_inference_loop(model, frequency=5)  # 5Hz用于测试
    
    # 保持运行
    rclpy.spin(env)
        
    # except KeyboardInterrupt:e
    #     print("\nInference loop interrupted by user")
    # except Exception as e:
    #     print(f"Error in main: {e}")
    # finally:
        # env.close()

if __name__ == "__main__":
    main()
