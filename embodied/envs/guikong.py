import math
import numpy as np
from scipy.spatial import KDTree
from copy import deepcopy



class kongzhi():
    def __init__(self, road_info_dict_pingjie, dt, ludian_len):
        self.road_info_dict_pingjie = road_info_dict_pingjie
        self.dt = dt
        self.pid_lat = PID_posi(target=0,upper=math.pi/9*2,lower=-math.pi/9*2,k=[0.075,0,0])
        self.pid_lon = PID_posi(target=0,upper=3,lower=-3,k=[0.5,0,0])
        # self.pid_lon = PID_inc(k=[0.1,0.02,0],target=0,upper=3,lower=-3)
        self.pid_lon_brake = PID_inc(k=[0.1,0.02,0],target=0,upper=3,lower=-3)
        self.v_des_temp = 0
        self.bizhang = False
        self.ludian_len = ludian_len

    def Pure_Pursuit(self,ego_info,front_car_info,behind_car_info,v_des,target_lane,action,v_max,rot_now,ind_houlun_changgelane=0):
        #参数
        if action == 0 or action == 1:
            if ego_info.loc['ego','v'] >= 0 and ego_info.loc['ego','v'] < 5:
                yumiao_time=2
            elif ego_info.loc['ego','v'] >= 5 and ego_info.loc['ego','v'] < 10:
                yumiao_time=1.5
            elif ego_info.loc['ego','v'] >= 10 and ego_info.loc['ego','v'] < 15:
                yumiao_time=1
            else:
                yumiao_time = 1
        else:
            if ego_info.loc['ego','v'] >= 0 and ego_info.loc['ego','v'] < 5:
                yumiao_time=2
            elif ego_info.loc['ego','v'] >= 5 and ego_info.loc['ego','v'] < 10:
                yumiao_time=1.5
            elif ego_info.loc['ego','v'] >= 10 and ego_info.loc['ego','v'] < 15:
                yumiao_time=1
            else:
                yumiao_time = 1

        d_bianyi = 0

        ego_station = np.array([ego_info.loc['ego','x'],ego_info.loc['ego','y']])
        ego_v = ego_info.loc['ego','v']
        ego_yaw = ego_info.loc['ego','yaw']
        ego_rel_ind = ego_info.loc['ego','rel_pos_ind']
        ego_rel_ind_houlun = ego_info.loc['ego','rel_pos_ind_houlun']
        ego_length = ego_info.loc['ego','length']
        ego_x_houlun = ego_station[0] - ego_length/1.7/2*math.cos(ego_yaw)
        ego_y_houlun = ego_station[1] - ego_length/1.7/2*math.sin(ego_yaw)
        if ego_yaw > math.pi and ego_yaw <= 2*math.pi:
            ego_yaw = ego_yaw - 2*math.pi
        
        # if ego_info.loc['ego','lane'] == target_lane:#直行
        if action == 0 or action == 1:#直行
            zuoyou, d_bianyi = self.panduan_bizhang(ego_info,front_car_info,behind_car_info,target_lane)
            self.ludian_bizhang(ego_info,target_lane)

            path = deepcopy(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'][ego_rel_ind_houlun:,:])
            path = path[(path[:,0]-ego_x_houlun)**2 + (path[:,1]-ego_y_houlun)**2 >= self.limit((ego_v*yumiao_time)**2,(ego_length+3)**2,1000**2)]
            # path = path[(path[:,0]-ego_x_houlun)**2 + (path[:,1]-ego_y_houlun)**2 >= self.limit(0,ego_length+3,1000)]
            if len(path) != 0:
                if not self.bizhang:
                    # a1 = math.atan2(path[0, 1]-ego_y_houlun, path[0, 0]-ego_x_houlun) - (ego_yaw)
                    # a1 = a1 % (2*math.pi)
                    # if a1 > math.pi and a1 <= 2*math.pi:
                    #     a1 = a1 - 2*math.pi
                    # if a1 > math.pi /2:
                    #     a1 = math.pi - a1
                    # if a1 < -math.pi /2:
                    #     a1 = -math.pi - a1
                    # a1 = a1 / 2
                    a1 = math.atan2(path[0, 1]-ego_y_houlun, path[0, 0]-ego_x_houlun) - math.atan2(ego_info.loc['ego','y']-ego_y_houlun, ego_info.loc['ego','x']-ego_x_houlun)
                    ld = math.sqrt((path[0, 1]-ego_y_houlun)**2+(path[0, 0]-ego_x_houlun)**2)
                    a2 = math.atan2(2*ego_length/1.7*math.sin(a1),ld)
                    a2 = math.atan2(a2*ego_length,self.dt*1.7*ego_v)
                    rot_1 = a2

                    alpha = math.atan2(path[0, 1]-ego_y_houlun, path[0, 0]-ego_x_houlun)
                    theta_e = alpha-ego_yaw
                    e_y = -ld*math.sin(theta_e)
                    rot_2 = self.pid_lat.cal_output(e_y)
                    
                    rot = 0.43*rot_2 + 0.57*rot_1
                    self.pid_lat.reset()
                else:
                    # refer_tree = KDTree(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'])
                    # dis, ind = refer_tree.query(path[0,:])
                    ind = ego_rel_ind_houlun + self.limit(int(ego_v*yumiao_time/self.ludian_len),2,200)
                    ind = self.limit(ind,0,len(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'])-2)
                    x_temp = deepcopy(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'][ind,0])
                    y_temp = deepcopy(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'][ind,1])
                    yaw_temp = deepcopy(self.road_info_dict_pingjie[str(target_lane)]['phi_road'][ind])

                    d_bianyi += 0.5
                    # zuoyou = abs(front_car_info['zuoyou'][index])#1右2左0中心
                    if zuoyou == 1:
                        x_now = x_temp - d_bianyi*math.sin(yaw_temp)
                        y_now = y_temp + d_bianyi*math.cos(yaw_temp)
                    if zuoyou == 2:
                        x_now = x_temp + d_bianyi*math.sin(yaw_temp)
                        y_now = y_temp - d_bianyi*math.cos(yaw_temp)
                    if zuoyou == 0:
                        x_now = x_temp
                        y_now = y_temp

                    # a1 = math.atan2(y_now-ego_y_houlun, x_now-ego_x_houlun) - (ego_yaw)
                    # a1 = a1 % (2*math.pi)
                    # if a1 > math.pi and a1 <= 2*math.pi:
                    #     a1 = a1 - 2*math.pi
                    # if a1 > math.pi /2:
                    #     a1 = math.pi - a1
                    # if a1 < -math.pi /2:
                    #     a1 = -math.pi - a1
                    # a1 = a1 / 2
                    a1 = math.atan2(y_now-ego_y_houlun, x_now-ego_x_houlun) - math.atan2(ego_info.loc['ego','y']-ego_y_houlun, ego_info.loc['ego','x']-ego_x_houlun)
                    ld = math.sqrt((y_now-ego_y_houlun)**2+(x_now-ego_x_houlun)**2)
                    a2 = math.atan2(2*ego_length/1.7*math.sin(a1),ld)
                    a2 = math.atan2(a2*ego_length,self.dt*1.7*ego_v)
                    rot_1 = a2

                    alpha = math.atan2(y_now-ego_y_houlun, x_now-ego_x_houlun)
                    theta_e = alpha-ego_yaw
                    e_y = -ld*math.sin(theta_e)
                    rot_2 = self.pid_lat.cal_output(e_y)
                    
                    rot = 0.43*rot_2 + 0.57*rot_1
                    self.pid_lat.reset()
            else:
                return 0, 0
        else:#换道
            path = deepcopy(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'][ind_houlun_changgelane:,:])
            path = path[(path[:,0]-ego_x_houlun)**2 + (path[:,1]-ego_y_houlun)**2 >= self.limit((ego_v*yumiao_time)**2,(ego_length+3)**2,1000**2)]
            if len(path) != 0:
                # a1 = math.atan2(path[0, 1]-ego_y_houlun, path[0, 0]-ego_x_houlun) - (ego_yaw)
                # a1 = a1 % (2*math.pi)
                # if a1 > math.pi and a1 <= 2*math.pi:
                #     a1 = a1 - 2*math.pi
                # if a1 > math.pi /2:
                #     a1 = math.pi - a1
                # if a1 < -math.pi /2:
                #     a1 = -math.pi - a1
                # a1 = a1 / 2
                a1 = math.atan2(path[0, 1]-ego_y_houlun, path[0, 0]-ego_x_houlun) - math.atan2(ego_info.loc['ego','y']-ego_y_houlun, ego_info.loc['ego','x']-ego_x_houlun)
                ld = math.sqrt((path[0, 1]-ego_y_houlun)**2+(path[0, 0]-ego_x_houlun)**2)
                a2 = math.atan2(2*ego_length/1.7*math.sin(a1),ld)
                a2 = math.atan2(a2*ego_length,self.dt*1.7*ego_v)
                rot_1 = a2
                # rot = rot_1

                alpha = math.atan2(path[0, 1]-ego_y_houlun, path[0, 0]-ego_x_houlun)
                theta_e = alpha-ego_yaw
                e_y = -ld*math.sin(theta_e)
                rot_2 = self.pid_lat.cal_output(e_y)

                rot = 0.43*rot_2 + 0.57*rot_1
            else:
                return 0, 0
        
        acc = 0
        if action == 0 or action == 1:#直行
            if front_car_info.empty:
                v_des = v_max
            else:
                index = front_car_info['rel_pos_ind'].idxmin()
                rel_des_front = abs(front_car_info['rel_des'][index])
                v_front = abs(front_car_info['v'][index])
                # print("front_index: ", index, "front_v: ", v_front, "rel_des_front: ", rel_des_front)
                if self.bizhang:
                    v_des = v_front + 10
                    v_des = self.limit(v_des,0,50)
                else:
                    if rel_des_front > front_car_info['v'][index]*1.5 + 10:
                        v_des = v_max
                    else:
                        v_des = v_front
            if v_des != self.v_des_temp:
                self.pid_lon.reset()
            acc = self.pid_lon.cal_output(ego_v - v_des)
            # print('期望: ', v_des, ego_v, v_des - ego_v)
            # if action == 0:
            #     acc = self.pid_lon.cal_output(ego_v - v_des)
            # else:
            #     acc = self.pid_lon.cal_output(ego_v - 0)

            # #IDM计算加速度
            # if front_car_info.empty:
            #     v_des = v_max
            #     acc = self.idm(ego_info.loc['ego','v'],-1,-1,v_des)
            # else:
            #     index = front_car_info['rel_pos_ind'].idxmin()
            #     rel_des_front = abs(front_car_info['rel_des'][index])
            #     v_front = abs(front_car_info['v'][index])
            #     if self.bizhang:
            #         v_des = v_front + 10
            #         v_des = self.limit(v_des,0,50)
            #         # acc = self.idm(ego_info.loc['ego','v'],-1,-1,v_des)
            #         if v_des != self.v_des_temp:
            #             self.pid_lon.reset()
            #         acc = self.pid_lon.cal_output(ego_v - v_des)
            #     else:
            #         if rel_des_front > self.limit(front_car_info['v'][index]*1.5 + 10,abs(front_car_info['v'][index]-ego_info.loc['ego','v'])*1.5+10,1000):
            #             v_des = v_max
            #             acc = self.idm(ego_info.loc['ego','v'],-1,-1,v_des)
            #         else:
            #             v_des = v_front
            #             acc = self.idm(ego_info.loc['ego','v'],v_front,rel_des_front,v_des)
            
            self.v_des_temp = v_des
        
            if not front_car_info.empty and not self.bizhang:
                index = front_car_info['rel_pos_ind'].idxmin()
                rel_des_front = abs(front_car_info['rel_des'][index])
                v_front = abs(front_car_info['v'][index])
                ttc = (rel_des_front - (ego_length + front_car_info['length'][index] ) / 2 - 2) /(abs(ego_v - v_front) + 1e-6)#2
                if ego_v > v_front:
                    if rel_des_front < 30 and ttc < 3:
                        # print("启用ttc")
                        acc = -3
                
                # if ego_v < 4 and rel_des_front < ((ego_length + front_car_info['length'][index] ) / 2 + 10) and v_front < 4:
                #     # acc = -0.5*(ego_v - 0)
                #     acc = self.pid_lon.cal_output(ego_v - 0)
                #         # acc = self.pid_lon.cal_output(ego_v - 0)
            # print('TTC: ',ttc)

        #紧急制动
        if action == 1:
            acc = self.pid_lon.cal_output(ego_v - 0)
        
        if action == 0:
            if ego_info.loc['ego','rel_pos_ind'] >= (len(self.road_info_dict_pingjie[str(target_lane)]['center_vertices']) - 2):
                acc = self.pid_lon.cal_output(ego_v - 0)

        # print("避障：", self.bizhang, "v_des: ", v_des)
        return acc, rot
    
    def ludian_bizhang(self,ego_info,target_lane):
        ind = ego_info.loc['ego','rel_pos_ind'] + 2
        total = 0
        for i in range(5):
            ind += i
            ind = self.limit(ind,0,len(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'])-2)
            total += abs(self.road_info_dict_pingjie[str(target_lane)]['curvature'][ind])
        k_lane = total / 5
        if k_lane >= 0.01:
            self.bizhang = False

    def panduan_bizhang(self,ego_info,front_car_info,behind_car_info,target_lane):
        zuoyou = 0 #1右2左0中心
        dis_bizhang = 0
        #有前车，有后车
        if (not front_car_info.empty) and (not behind_car_info.empty):
            index = front_car_info['rel_pos_ind'].idxmin()
            rel_des_front = abs(front_car_info['rel_des'][index])
            index_b = behind_car_info['rel_pos_ind'].idxmax()
            rel_des_behind = abs(behind_car_info['rel_des'][index_b])
            #超过后车
            if self.panduan_over_houche(ego_info,behind_car_info,target_lane):
                if rel_des_front > 30:
                    self.bizhang = False
                else:
                    dis_need, zuoyou_need = self.panduan_bizhang_dis_zuoyou(front_car_info,index,target_lane)
                    if ego_info.loc['ego','width'] > dis_need:
                        self.bizhang = False
                    else:
                        self.bizhang = True
                        zuoyou = zuoyou_need
                        dis_bizhang = abs(self.road_info_dict_pingjie[str(target_lane)]['width'][front_car_info['rel_pos_ind'][index]]/2 - dis_need/2)
            else:
                self.bizhang = True
                zuoyou_b = behind_car_info['zuoyou'][index_b]
                zuoyou_f = front_car_info['zuoyou'][index]
                dis_need_f, zuoyou_need_f = self.panduan_bizhang_dis_zuoyou(front_car_info,index,target_lane)
                dis_need_b, zuoyou_need_b = self.panduan_bizhang_dis_zuoyou(behind_car_info,index_b,target_lane)
                zuoyou = zuoyou_need_b
                if zuoyou_need_b == zuoyou_need_f:
                    dis_bizhang = max(abs(self.road_info_dict_pingjie[str(target_lane)]['width'][behind_car_info['rel_pos_ind'][index_b]]/2 - dis_need_b/2),abs(self.road_info_dict_pingjie[str(target_lane)]['width'][front_car_info['rel_pos_ind'][index]]/2 - dis_need_f/2))
                else:
                    dis_bizhang = abs(self.road_info_dict_pingjie[str(target_lane)]['width'][behind_car_info['rel_pos_ind'][index_b]]/2 - dis_need_b/2)

        #有前车，无后车
        if (not front_car_info.empty) and (behind_car_info.empty):
            index = front_car_info['rel_pos_ind'].idxmin()
            rel_des_front = abs(front_car_info['rel_des'][index])
            if rel_des_front > 30:
                self.bizhang = False
            else:
                dis_need, zuoyou_need = self.panduan_bizhang_dis_zuoyou(front_car_info,index,target_lane)
                if ego_info.loc['ego','width'] > dis_need:
                    self.bizhang = False
                else:
                    self.bizhang = True
                    zuoyou = zuoyou_need
                    dis_bizhang = abs(self.road_info_dict_pingjie[str(target_lane)]['width'][front_car_info['rel_pos_ind'][index]]/2 - dis_need/2)
        #无前车，有后车
        if (front_car_info.empty) and (not behind_car_info.empty):
            index_b = behind_car_info['rel_pos_ind'].idxmax()
            rel_des_behind = abs(behind_car_info['rel_des'][index_b])
            #超过后车
            if self.panduan_over_houche(ego_info,behind_car_info,target_lane):
                self.bizhang = False
            else:
                dis_need, zuoyou_need = self.panduan_bizhang_dis_zuoyou(behind_car_info,index_b,target_lane)
                self.bizhang = True
                zuoyou = behind_car_info['zuoyou'][index_b]
                dis_bizhang = abs(self.road_info_dict_pingjie[str(target_lane)]['width'][behind_car_info['rel_pos_ind'][index_b]]/2 - dis_need/2)
        #无前车，无后车
        if (front_car_info.empty) and (behind_car_info.empty):
            self.bizhang = False

        return zuoyou, dis_bizhang

    def panduan_over_houche(self,ego_info,behind_car_info,target_lane):
        index = behind_car_info['rel_pos_ind'].idxmax()
        npc_ind = behind_car_info['rel_pos_ind'][index]
        ego_ind = ego_info.loc['ego','rel_pos_ind']
        low = self.limit(npc_ind-int(30/self.ludian_len),0,len(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'])-2)
        high = self.limit(ego_ind+int(30/self.ludian_len),0,len(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'])-2)
        #索引车道缩减
        path = deepcopy(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'][low:high,:])
        # print("panduan_over_houche: ",len(path))
        refer_tree = KDTree(path)
        #找到自车最后点的索引
        ego_x = ego_info.loc['ego','x']
        ego_y = ego_info.loc['ego','y']
        ego_yaw = ego_info.loc['ego','yaw']
        ego_height = ego_info.loc['ego','length']
        ego_width = ego_info.loc['ego','width']
        ind_ego_hou = 99999999
        x1_ego = ego_x + ego_height / 2 * math.cos(ego_yaw) - ego_width / 2 * math.sin(ego_yaw)
        y1_ego = ego_y + ego_height / 2 * math.sin(ego_yaw) + ego_width / 2 * math.cos(ego_yaw)
        x2_ego = ego_x - ego_height / 2 * math.cos(ego_yaw) + ego_width / 2 * math.sin(ego_yaw)
        y2_ego = ego_y - ego_height / 2 * math.sin(ego_yaw) - ego_width / 2 * math.cos(ego_yaw)
        x3_ego = ego_x + ego_width / 2 * math.sin(ego_yaw) + ego_height / 2 * math.cos(ego_yaw)
        y3_ego = ego_y - (ego_width / 2 - ego_height / 2 * math.tan(ego_yaw)) * math.cos(ego_yaw)
        x4_ego = ego_x - ego_width / 2 * math.sin(ego_yaw) - ego_height / 2 * math.cos(ego_yaw)
        y4_ego = ego_y + (ego_width / 2 - ego_height / 2 * math.tan(ego_yaw)) * math.cos(ego_yaw)
        ego_array = np.array([[x1_ego,y1_ego],[x2_ego,y2_ego],[x3_ego,y3_ego],[x4_ego,y4_ego]])
        for i in ego_array:
            distance, ind = refer_tree.query(i)
            if ind < ind_ego_hou:
                ind_ego_hou = ind
        #找到后车最前点的索引
        npc_x = behind_car_info['x'][index]
        npc_y = behind_car_info['y'][index]
        npc_height = behind_car_info['length'][index]
        npc_width = behind_car_info['width'][index]
        npc_yaw = behind_car_info['yaw'][index]
        ind_npc_qian = 0
        x1_npc = npc_x + npc_height / 2 * math.cos(npc_yaw) - npc_width / 2 * math.sin(npc_yaw)
        y1_npc = npc_y + npc_height / 2 * math.sin(npc_yaw) + npc_width / 2 * math.cos(npc_yaw)
        x2_npc = npc_x - npc_height / 2 * math.cos(npc_yaw) + npc_width / 2 * math.sin(npc_yaw)
        y2_npc = npc_y - npc_height / 2 * math.sin(npc_yaw) - npc_width / 2 * math.cos(npc_yaw)
        x3_npc = npc_x + npc_width / 2 * math.sin(npc_yaw) + npc_height / 2 * math.cos(npc_yaw)
        y3_npc = npc_y - (npc_width / 2 - npc_height / 2 * math.tan(npc_yaw)) * math.cos(npc_yaw)
        x4_npc = npc_x - npc_width / 2 * math.sin(npc_yaw) - npc_height / 2 * math.cos(npc_yaw)
        y4_npc = npc_y + (npc_width / 2 - npc_height / 2 * math.tan(npc_yaw)) * math.cos(npc_yaw)
        npc_array = np.array([[x1_npc,y1_npc],[x2_npc,y2_npc],[x3_npc,y3_npc],[x4_npc,y4_npc]])
        for i in npc_array:
            distance, ind = refer_tree.query(i)
            if ind > ind_npc_qian:
                ind_npc_qian = ind
        
        if ind_ego_hou > ind_npc_qian:
            return True
        else:
            return False

    def panduan_bizhang_dis_zuoyou(self,car_info,index,target_lane):
        x1 = car_info['x'][index]
        y1 = car_info['y'][index]
        height1 = car_info['length'][index]
        width1 = car_info['width'][index]
        angle1 = car_info['yaw'][index]
        rel_pos_ind = car_info['rel_pos_ind'][index]
        ind_detal = self.limit(int(height1/self.ludian_len) + int(10/self.ludian_len),1,1000)

        lane_width = self.road_info_dict_pingjie[str(target_lane)]['width'][rel_pos_ind]

        #计算四个顶点坐标
        x1_1 = x1 + height1 / 2 * math.cos(angle1) - width1 / 2 * math.sin(angle1)
        y1_1 = y1 + height1 / 2 * math.sin(angle1) + width1 / 2 * math.cos(angle1)
        x2_1 = x1 - height1 / 2 * math.cos(angle1) + width1 / 2 * math.sin(angle1)
        y2_1 = y1 - height1 / 2 * math.sin(angle1) - width1 / 2 * math.cos(angle1)
        x3_1 = x1 + width1 / 2 * math.sin(angle1) + height1 / 2 * math.cos(angle1)
        y3_1 = y1 - (width1 / 2 - height1 / 2 * math.tan(angle1)) * math.cos(angle1)
        x4_1 = x1 - width1 / 2 * math.sin(angle1) - height1 / 2 * math.cos(angle1)
        y4_1 = y1 + (width1 / 2 - height1 / 2 * math.tan(angle1)) * math.cos(angle1)

        zuobiao = np.array([[x1_1,y1_1],[x2_1,y2_1],[x3_1,y3_1],[x4_1,y4_1]])
        dis_zuobiao = []
        zuoyou_zuobiao = []

        ind_hou = self.limit(rel_pos_ind - ind_detal,0,len(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'])-2)
        ind_qian = self.limit(rel_pos_ind + ind_detal,0,len(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'])-2)
        path = deepcopy(self.road_info_dict_pingjie[str(target_lane)]['center_vertices'][ind_hou:ind_qian,:])
        # print("panduan_bizhang_dis_zuoyou: ",len(path))
        refer_tree = KDTree(path)
        for i in zuobiao:
            distance, ind = refer_tree.query(i)
            dis_zuobiao.append(distance)
            if ind < len(path) - 2:
                xiangliang_car = i - path[ind,:]
                xiangliang_line = path[ind+1,:] - path[ind,:]
                if xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] > 0:
                    zuoyou_i = 1 #车在道路中心线右侧
                elif xiangliang_car[0]*xiangliang_line[1] - xiangliang_car[1]*xiangliang_line[0] < 0:
                    zuoyou_i = 2 #车在道路中心线左侧
                else:
                    zuoyou_i = 0 #车在道路中心线
            else:
                zuoyou_i = 0 #车在道路外前方
            
            zuoyou_zuobiao.append(zuoyou_i)
        
        dis_need = 9999
        zuoyou_need = 0

        dis_zuo_temp = 0
        dis_you_temp = 0
        zuoyou_zuo_temp = 0
        zuoyou_you_temp = 0
        if (1 not in zuoyou_zuobiao) or (2 not in zuoyou_zuobiao):
            for i in range(len(dis_zuobiao)):
                if dis_zuobiao[i] < dis_need:
                    dis_need = dis_zuobiao[i]
                    zuoyou_need = zuoyou_zuobiao[i]
            dis_need = lane_width/2 + dis_need
        else:
            for i in range(len(zuoyou_zuobiao)):
                if zuoyou_zuobiao[i] == 1:
                    if dis_zuobiao[i] > dis_you_temp:
                        dis_you_temp = dis_zuobiao[i]
                        zuoyou_you_temp = zuoyou_zuobiao[i]
                if zuoyou_zuobiao[i] == 2:
                    if dis_zuobiao[i] > dis_zuo_temp:
                        dis_zuo_temp = dis_zuobiao[i]
                        zuoyou_zuo_temp = zuoyou_zuobiao[i]
            
            if dis_zuo_temp > dis_you_temp:
                dis_need = dis_you_temp
                zuoyou_need = zuoyou_zuo_temp
            else:
                dis_need = dis_zuo_temp
                zuoyou_need = zuoyou_you_temp
            
            dis_need = lane_width/2 - dis_need

        return dis_need, zuoyou_need


    def limit(self, value, min_num, max_num):
        if (value < min_num):
            return min_num
        elif (value > max_num):
            return max_num
        else:
            return value
    
    def idm(self,ego_v, f_v, dis, exv=50):
        """跟idm模型有关的模型参数
        :param a_bound: 本车加速度绝对值的上下界
        :param exv: 期望速度
        :param t: 反应时间
        :param a: 起步加速度
        :param b: 舒适减速度
        :param gama: 加速度指数
        :param s0: 静止安全距离
        :param s1: 与速度有关的安全距离选择参数
        """
        a_bound=3.0
        exv=exv
        t=1.2
        a=2.22
        b=2.4
        gama=4
        s0=1.0
        s1=2.0
        s_ = 0
        k = 1

        v, fv, dis_gap = ego_v, f_v, dis
        if dis_gap < 0:
            a_idm = a * (1 - (v / exv) ** gama)
        else:
            # 求解本车与前车的期望距离
            self.s_ = s0 + s1 * (v / exv) ** 0.5 + t * v + v * (v - fv) / 2 / (a * b) ** 0.5
            # 求解本车加速度
            a_idm = a * (1 - (v / exv) ** gama - ((s_ / (dis_gap+1e-6)) ** 2))
        # 对加速度进行约束
        a_idm = np.clip(a_idm, - a_bound, a_bound)

        return a_idm



class IDM():
    def __init__(self, a_bound=3.0, exv=50, t=1.2, a=2.22, b=2.4, gama=4, s0=1.0, s1=2.0):
        """跟idm模型有关的模型参数
        :param a_bound: 本车加速度绝对值的上下界
        :param exv: 期望速度
        :param t: 反应时间
        :param a: 起步加速度
        :param b: 舒适减速度
        :param gama: 加速度指数
        :param s0: 静止安全距离
        :param s1: 与速度有关的安全距离选择参数
        """
        self.a_bound = a_bound
        self.exv = exv
        self.t = t
        self.a = a
        self.b = b
        self.gama = gama
        self.s0 = s0
        self.s1 = s1
        self.s_ = 0
        self.k = 1

    def act(self, ego_v, f_v, dis):
        acc = 0
        acc = self.deside_acc(ego_v, f_v, dis)
        return acc

    def deside_acc(self, ego_v, f_v, dis):
        v, fv, dis_gap = ego_v, f_v, dis
        if dis_gap < 0:
            a_idm = self.a * (1 - (v / self.exv) ** self.gama)
        else:
            # 求解本车与前车的期望距离
            self.s_ = self.s0 + self.s1 * (v / self.exv) ** 0.5 + self.t * v + v * (v - fv) / 2 / (self.a * self.b) ** 0.5
            # 求解本车加速度
            a_idm = self.a * (1 - (v / self.exv) ** self.gama - ((self.s_ / (dis_gap+1e-6)) ** 2))
        # 对加速度进行约束
        a_idm = np.clip(a_idm, -self.a_bound, self.a_bound)
        # print(v,fv,dis_gap,a_idm,self.s_)
        # print(state,v,fv,dis_gap,a_idm)
        return a_idm

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


class PID_inc:
    """增量式实现
    """
    def __init__(self, k, target, upper=1., lower=-1.):
        self.kp, self.ki, self.kd = k   
        self.err = 0
        self.err_last = 0
        self.err_ll = 0
        self.target = target
        self.upper = upper
        self.lower = lower
        self.value = 0
        self.inc = 0

    def cal_output(self, state):
        self.err = self.target - state
        self.inc = self.kp * (self.err - self.err_last) + self.ki * self.err + self.kd * (
            self.err - 2 * self.err_last + self.err_ll)
        self._update()
        return self.value

    def _update(self):
        self.err_ll = self.err_last
        self.err_last = self.err
        self.value = self.value + self.inc
        if self.value > self.upper:
            self.value = self.upper
        elif self.value < self.lower:
            self.value = self.lower

    def set_target(self, target):
        self.target = target

    def set_k(self, k):
        self.kp, self.ki, self.kd = k

    def set_bound(self, upper, lower):
        self.upper_bound = upper
        self.lower_bound = lower

class Obstacle:
    def __init__(self):
        self.type_ob = 1#1:car  2:pedestrian  3:cyclist  4:others
        self.m_ind = 0
        self.m_lane = 0
        self.is_static = True
        self.L1 = 4#Half of length
        self.L2 = 2#Half of width
        self.L1_sd = 0#Safe distance of longitude direction
        self.L2_sd = 0#Safe distance of lateral direction
        self.mx = 0
        self.my = 0
        self.ms = 0#Station on Frenet
        self.ml = 0# Lateral error on Frenet
        self.mv = 0
        self.mh = 0
        self.ma = 0
        self.mdh = 0
        self.m_lane_width = 3.5#Width of each lane
        self.x_ob = []
        self.y_ob = []
        self.v_ob = []
        self.h_ob = []
    
    def ObstacleLaneCalcu(self, si, li, vi, ri):
        self.ms = si
        self.ml = li
        self.mv = vi
        self.L2 = ri
        if (self.ml >= 0.5 * self.m_lane_width and self.ml <= 1.5 * self.m_lane_width):
            self.m_lane = 1
        elif (self.ml >= -0.5 * self.m_lane_width and self.ml <= 0.5 * self.m_lane_width):
            self.m_lane = 0
        elif (self.ml >= -1.5 * self.m_lane_width and self.ml <= -0.5 * self.m_lane_width):
            self.m_lane = -1
        elif (self.ml >= -2.5 * self.m_lane_width and self.ml <= -1.5 * self.m_lane_width):
            self.m_lane = -2
        elif (self.ml >= -3.5 * self.m_lane_width and self.ml <= -2.5 * self.m_lane_width):
            self.m_lane = -3
        else: 
            self.m_lane = -4#Do not on the road


        

class Poly_planner_onsite:
    def __init__(self, road_info_dict_pingjie):
        #Basic parameters of ego car 
        self.m = 1845#汽车质量
        self.R = 0.281#轮胎半径
        self.Iz = 2488#转动惯量4095
        self.L = 2.875#轴距
        self.lf = self.L/2#前轴距1.18
        self.lr = self.L/2#后轴距1.77
        self.Cf = 164370#前轮侧偏刚度
        self.Cr = 124597#后轮侧偏刚度
        self.g = 9.7885#重力加速度
        self.i_ste = 11.0#转向系统传动比
        self.i_tra = 10.6#传动系统传动比

        self.smk, self.smc, self.smd = 50, 300, 0.1
        self.pre_n = 1
        self.k_ff, self.k_fb = 1.0, 1.0

        self.is_first_run = True
        self.edge_left, self.edge_right, self.pf_precision, self.T_plan, self.safe_dis, self.Rv, self.t_pre = 1.5, -1.5, 0.1, 0.1, 0.5, 1.0, 4.0
        self.lane_width = 3.5
        self.steer_FF, self.steer_FB, self.steer = 0.0, 0.0, 0.0
        self.Xo, self.Yo, self.Vx, self.Vy, self.Ax, self.Ay, self.Yaw, self.Yawrate, self.Station, self.Lat_veh = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.x_proj, self.y_proj, self.h_proj, self.k_proj = 0, 0, 0, 0
        self.Yaw0, self.Ts = math.pi, 0.01
        self.s_f, self.a_0, self.b_0, self.c_0, self.d_0, self.e_0, self.f_0 = 0, 0, 0, 0, 0, 0, 0 #五次多项式参数信息
        self.edL, self.dedL, self.ed0, self.ded0, self.ephi0, self.dephi0, self.row0, self.rowL, self.es0, self.esL = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.edx, self.dedx, self.ephix, self.dephix, self.esx, self.rowx = [0]*5, [0]*5, [0]*5, [0]*5, [0]*5, [0]*5
        self.G0, self.G1, self.G2, self.G3, self.G4 = 0.2, 0.3, 0.3, 0.1, 0.1 #预瞄点权重
        self.cof_obd, self.cof_sm, self.cof_cen, self.cof_con = 8.0, 0.1, 1.9, 0.1 #路径规划,安全性、平滑性、中心性、一致性的权重系数
        self.im, self.im_pre = 0, 1 #Index of match point
        self.lane_ego = 0; #Id of current lane
        self.num_plan_point = 51
        self.ax_desire = 0
        self.spd_des = 4.0
        self.spd_max_direct = 30

        self.road_x = []
        self.road_y = []
        self.road_heading = []
        self.road_kappa = []
        self.road_station = []
        self.path_x, self.path_y, self.path_h, self.path_k = [], [], [], []#规划的轨迹

        self.index_s = []
        self.LQR_K1 = [40.033562628470648,0.235507608343038,0.413467482284626,-0.024525390067828 ]#LQR控制系数 1m/s
        self.LQR_K2 = [36.751187573624073,0.463866597546888,0.386577747671965,-0.034443819863887]#LQR控制系数 2m/s
        self.LQR_K3 = [34.235465748523545,0.622826221243402,0.366751706868862,-0.033533834539390]
        self.LQR_K4 = [32.297372887760226,0.715412326313747,0.352449359343296,-0.028568422245594]
        self.LQR_K5 = [30.755201060760438,0.760813896674884,0.341959185626009,-0.022784796160408]

        self.vector_n, self.ob_flag = [], []
        self.count_p = 10
        self.no_lane_free = False

        self.t_cost = 0
        self.t_all = 0

        self.ob_car_detect = []
        self.road_info_dict_pingjie = road_info_dict_pingjie

    def limit(self, value, min_num, max_num):
        if (value < min_num):
            return min_num
        elif (value > max_num):
            return max_num
        else:
            return value
    
    def sign(self, val):
        if (val > 0):
            return 1
        elif (val < 0):
            return -1
        else:
            return 0
    
    #五次多项式路径规划函数
    def path_plan(self, s_0, p_0, vx):
        s, s0, sp, sf, dp_0, ddp_0 = 0, 0, 0, 0, 0, 0
        p, dp, ddp = 0, 0, 0
        s_num, j, n_s = 0, 0, 0 #规划路径的数量以及可能用到的临时变量
        a, b, c, d, e, f = [], [], [], [], [], []
        f_obd_data, f_sm_data, f_con_data, f_cen_data, pf, f_cost = [], [], [], [], [], []
        ds = []

        s0 = s_0
        dp_0 = 5.0 * self.a_0 * pow(s_0, 4) + 4.0 * self.b_0 * pow(s_0, 3) + 3.0 * self.c_0 * pow(s_0, 2) + 2.0 * self.d_0 * s_0 + self.e_0
        ddp_0 = 20.0 * self.a_0 * pow(s_0, 3) + 12.0 * self.b_0 * pow(s_0, 2) + 6.0 * self.c_0 * s_0 + 2.0 * self.d_0

        self.ob_flag = []
        s_num = int((self.edge_left - self.edge_right) / self.pf_precision + 1)
        a, b, c, d, e, f = [0]*s_num, [0]*s_num, [0]*s_num, [0]*s_num, [0]*s_num, [0]*s_num
        f_obd_data, f_sm_data, f_con_data, f_cen_data, pf, f_cost = [0]*s_num, [0]*s_num, [0]*s_num, [0]*s_num, [0]*s_num, [0]*s_num
        self.ob_flag = [0]*s_num
        ds = [0]*len(self.ob_car_detect)

        sp = self.t_pre * vx
        if (sp <= 10.0):
            sp = 10.0
        if (sp >= 40):
            sp = 40
        sf = s_0 + sp#规划终点的纵坐标
        self.s_f = sf

        self.vector_n = []
        for i in range(s_num):
            pf[i] = self.edge_right + self.pf_precision * i
            X = np.array([[1.0, s0, pow(s0, 2), pow(s0, 3), pow(s0, 4), pow(s0, 5)],
                         [1.0, sf, pow(sf, 2), pow(sf, 3), pow(sf, 4), pow(sf, 5)],
                         [0.0, 1.0, 2.0 * s0, 3.0 * pow(s0, 2), 4.0 * pow(s0, 3), 5.0 * pow(s0, 4)],
                         [0.0, 1.0, 2.0 * sf, 3.0 * pow(sf, 2), 4.0 * pow(sf, 3), 5.0 * pow(sf, 4)],
                         [0.0, 0.0, 2.0, 6.0 * s0, 12.0 * pow(s0, 2), 20.0 * pow(s0, 3)],
                         [0.0, 0.0, 2.0, 6.0 * sf, 12.0 * pow(sf, 2), 20.0 * pow(sf, 3)]])
            Y = np.array([p_0, pf[i], dp_0, 0, ddp_0, 0])
            Y = np.resize(Y, (6,1))
            A_Matri = np.linalg.solve(X,Y)

            a[i] =  A_Matri[5,0]
            b[i] =  A_Matri[4,0]
            c[i] =  A_Matri[3,0]
            d[i] =  A_Matri[2,0]
            e[i] =  A_Matri[1,0]
            f[i] =  A_Matri[0,0]
        
        have_obs = []
        for i in range(s_num):
            s = s_0
            while s <= sf:
                p = a[i] * pow(s, 5) + b[i] * pow(s, 4) + c[i] * pow(s, 3) + d[i] * pow(s, 2) + e[i] * s + f[i]
                have_obs = []
                for nm in range(len(self.ob_car_detect)):
                    if s_0 < self.ob_car_detect[nm].ms and self.ob_car_detect[nm].ms <= sf:
                        ds[nm] = math.sqrt(pow(s - self.ob_car_detect[nm].ms - 2.0, 2) + pow(p - self.ob_car_detect[nm].ml, 2))
                    else:
                        ds[nm] = 99999
                    if ds[nm] - self.ob_car_detect[nm].L2 - self.Rv <= self.safe_dis:
                        have_obs.append(nm)
                if len(have_obs) != 0:
                    self.ob_flag[i] = 1
                    break
                s += 1.0
            if self.ob_flag[i] == 0:
                self.vector_n.append(i)
        
        #If all of these paths have obstacle, choose the nearest lane to follow and slow down
        if len(self.vector_n) == 0:
            self.no_lane_free = True
            for i in range(s_num):
                self.vector_n.append(i)
        else:
            self.no_lane_free = False
        
        mincost = 1e8#记录最小代价函数
        for i in range(len(self.vector_n)):
            if len(self.vector_n) < 1:
                break
            j = self.vector_n[i]
            for nm in range(len(self.ob_car_detect)):
                if (self.ob_car_detect[nm].ms >= (s_0 - 2.0)) and (self.ob_car_detect[nm].ms - s_0 <= 15):
                    s = s0
                    while s < sf:
                        p = a[j] * pow(s, 5) + b[j] * pow(s, 4) + c[j] * pow(s, 3) + d[j] * pow(s, 2) + e[j] * s + f[j]
                        ds[nm] = math.sqrt(pow(s - self.ob_car_detect[nm].ms - 2.0, 2) + pow(p - self.ob_car_detect[nm].ml, 2)) - self.Rv - self.ob_car_detect[nm].L2
                        f_obd_data[i] = 1.0 / ds[nm] + f_obd_data[i]
                        s += 1
            
            s = s0
            while s < sf:
                #平滑性，防止曲率过大
                p = a[j] * pow(s, 5) + b[j] * pow(s, 4) + c[j] * pow(s, 3) + d[j] * pow(s, 2) + e[j] * s + f[j]
                dp = 5.0 * a[j] * pow(s, 4) + 4.0 * b[j] * pow(s, 3) + 3.0 * c[j] * pow(s, 2) + 2.0 * d[j] * s + e[j]
                ddp = 20.0 * a[j] * pow(s, 3) + 12.0 * b[j] * pow(s, 2) + 6.0 * c[j] * s + 2.0 * d[j]

                k1 = 0
                S1, Q, k = 0, 0, 0 #计算曲率的中间参数
                S1 = math.sqrt(pow(dp,2) + pow((1.0 - p * k1),2))
                if ((1.0 - p * k1) == 0):
                    Q = 0.0
                elif(1.0 - p * k1 < 0):
                    Q = -1.0
                else:
                    Q = 1.0
                k = abs(Q / S1 * (k1 + ((1.0 - p * k1) * ddp + k1 * dp * dp) / (S1 * S1)))
                f_sm_data[i] = f_sm_data[i] + k

                #一致性，防止前后两条规划的路径相差较大而导致突变
                y_0, y_1 = 0, 0
                y_0 = self.a_0 * pow(s - vx * self.T_plan, 5) + self.b_0 * pow(s - vx * self.T_plan, 4) + self.c_0 * pow(s - vx * self.T_plan, 3) + self.d_0 * pow(s - vx * self.T_plan, 2) + self.e_0 * (s - vx * self.T_plan) + self.f_0
                y_1 = a[j] * pow(s, 5) + b[j] * pow(s, 4) + c[j] * pow(s, 3) + d[j] * pow(s, 2) + e[j] * s + f[j]
                f_con_data[i] = abs(y_0 - y_1) * 1.0 + f_con_data[i]#上一次规划的路径点与本次规划的路径点距离之间的差值累加

                s += 1

            #中心性，确保车辆在道路中心线附近行驶
            s = (s_0 + sp * 0.25)
            while s < sf:
                f_cen1 = 0
                f_cen1 = abs((a[j] * pow(s, 5) + b[j] * pow(s, 4) + c[j] * pow(s, 3) + d[j] * pow(s, 2) + e[j] * s + f[j]))
                f_cen_data[i] = 1.0 * f_cen1 + f_cen_data[i]
                s += 1
            
            f_cost[i] = self.cof_obd * f_obd_data[i] + self.cof_sm * f_sm_data[i] + self.cof_con * f_con_data[i] + self.cof_cen * abs(f_cen_data[i])
            if (f_cost[i] < mincost):
                n_s = j#代价函数最小路径的编号
                mincost = f_cost[i]

        self.a_0 = a[n_s]
        self.b_0 = b[n_s]
        self.c_0 = c[n_s]
        self.d_0 = d[n_s]
        self.e_0 = e[n_s]
        self.f_0 = f[n_s]
    
    def index2s(self, length, origin_x, origin_y, match_index):
        self.index_s = [0]*length
        for i in range(1,length):
            self.index_s[i] = math.sqrt(pow(self.road_x[i] - self.road_x[i - 1], 2) + pow(self.road_y[i] - self.road_y[i - 1], 2)) + self.index_s[i - 1]
        s_temp = self.index_s[match_index]
        ss, vec_product = 0, 0
        vector_match_2_origin, vector_match_2_match_next = [], []
        vector_match_2_origin.append(origin_x - self.road_x[match_index])
        vector_match_2_origin.append(origin_y - self.road_y[match_index])
        vector_match_2_match_next.append(self.road_x[match_index + 1] - self.road_x[match_index])
        vector_match_2_match_next.append(self.road_y[match_index + 1] - self.road_y[match_index])
        vec_product = vector_match_2_origin[0]*vector_match_2_match_next[0] + vector_match_2_origin[1]*vector_match_2_match_next[1]
        if (vec_product > 0.0):
            ss = s_temp + math.sqrt(vector_match_2_origin[0]**2 + vector_match_2_origin[1]**2)
        else:
            ss = s_temp - math.sqrt(vector_match_2_origin[0]**2 + vector_match_2_origin[1]**2)
        for i in range(len(self.index_s)):
            self.index_s[i] -= ss

    #该函数将计算在frenet坐标系下，点(s,l)在frenet坐标轴的投影的直角坐标(proj_x,proj_y,proj_heading,proj_kappa)
    def CalcProjPoint(self, s):
        match_i = 0
        ds = 0
        while (self.index_s[match_i] < s + 0.00000001):
            match_i += 1
        ds = s - self.index_s[match_i]
        p_x = self.road_x[match_i] + ds * math.cos(self.road_heading[match_i])
        p_y = self.road_y[match_i] + ds * math.sin(self.road_heading[match_i])
        p_h = self.road_heading[match_i] + ds * self.road_kappa[match_i]
        p_k = self.road_kappa[match_i]
        return p_x, p_y, p_h, p_k

    
    #Frenet坐标转笛卡尔坐标
    #已知S(五次多项式规划的起点0和终点中的各点数值)及其l、dl、ddl，全局路径的X、Y及其对应的参考航向角和曲率
    def Frenet2Cartesian(self, s_0, sf):
        sx = (sf - s_0) / (self.num_plan_point - 1)
        l_temp, dl_temp, ddl_temp, pro_x, pro_y, pro_h, pro_k = 0, 0, 0, 0, 0, 0, 0
        s_set, l_set, dl_set, ddl_set = [], [], [], []
        ss = s_0
        while ss < sf + 1e-5:
            s_set.append(ss-s_0)
            l_temp = self.a_0 * pow(ss, 5) + self.b_0 * pow(ss, 4) + self.c_0 * pow(ss, 3) + self.d_0 * pow(ss, 2) + self.e_0 * ss + self.f_0
            dl_temp = 5.0 * self.a_0 * pow(ss, 4) + 4.0 * self.b_0 * pow(ss, 3) + 3.0 * self.c_0 * pow(ss, 2) + 2.0 * self.d_0 * ss + self.e_0
            ddl_temp = 20.0 * self.a_0 * pow(ss, 3) + 12.0 * self.b_0 * pow(ss, 2) + 6.0 * self.c_0 * ss + 2.0 * self.d_0
            l_set.append(l_temp)
            dl_set.append(dl_temp)
            ddl_set.append(ddl_temp)
            ss += sx
        
        self.path_x = [0]*len(s_set)
        self.path_y = [0]*len(s_set)
        self.path_h = [0]*len(s_set)
        self.path_k = [0]*len(s_set)
        for i in range(len(s_set)):
            pro_x, pro_y, pro_h, pro_k = self.CalcProjPoint(s_set[i])
            self.path_x[i] = pro_x - math.sin(pro_h) * l_set[i]
            self.path_y[i] = pro_y + math.cos(pro_h) * l_set[i]
            self.path_h[i] = pro_h + math.atan(dl_set[i] / (1.0 - pro_k * l_set[i]))
            self.path_k[i] = ((ddl_set[i] + pro_k * dl_set[i] * math.tan(self.path_h[i] - pro_h)) * pow(math.cos(self.path_h[i] - pro_h), 2) / (1.0 - pro_k * l_set[i]) + pro_k) *math.cos(self.path_h[i] - pro_h) / (1.0 - pro_k * l_set[i])

    #误差计算模块
    def err_cal_module(self, xo, yo, phi, vx, vy, phi_dot, xr, yr, hr, kr):
        proj_hr, s_dot, ss_dot = 0, 0, 0
        self.ed0 = -math.sin(hr) * (xo - xr) + math.cos(hr) * (yo - yr)
        self.es0 = math.cos(hr) * (xo - xr) + math.sin(hr) * (yo - yr)
        proj_hr = hr + kr * self.es0
        self.ded0 = vy * math.cos(phi - proj_hr) + vx * math.sin(phi - proj_hr)
        self.ephi0 = phi - proj_hr
        ss_dot = vx * math.cos(phi - proj_hr) - vy * math.sin(phi - proj_hr)
        s_dot = ss_dot / (1.0 - kr * self.ed0)
        self.dephi0 = phi_dot - kr * s_dot
    
    def err_cal_module_return(self, xo, yo, phi, vx, vy, phi_dot, xr, yr, hr, kr):
        proj_hr, s_dot, ss_dot = 0, 0, 0
        ed0 = -math.sin(hr) * (xo - xr) + math.cos(hr) * (yo - yr)
        es0 = math.cos(hr) * (xo - xr) + math.sin(hr) * (yo - yr)
        proj_hr = hr + kr * es0
        ded0 = vy * math.cos(phi - proj_hr) + vx * math.sin(phi - proj_hr)
        ephi0 = phi - proj_hr
        ss_dot = vx * math.cos(phi - proj_hr) - vy * math.sin(phi - proj_hr)
        s_dot = ss_dot / (1.0 - kr * ed0)
        dephi0 = phi_dot - kr * s_dot
        return ed0, ded0, ephi0, dephi0, es0
    
    #前馈控制前轮转角计算模块,输入单位为rad，输出单位为度
    def control_FF(self, vx, row):
        self.steer_FF = row * (self.Cf * self.Cr * (self.lf + self.lr) * (self.lf + self.lr) + (self.Cr * self.lr - self.Cf * self.lf) * self.m * vx * vx) / (self.Cf * self.Cr * (self.lf + self.lr))
        self.steer_FF = self.steer_FF * 180.0 / math.pi
    
    #反馈控制前轮转角计算模块，输出单位为度
    def control_FB(self, ed, ed_dot, ephi, ephi_dot, es, vx, kp, cp, dp):
        a11, a12, a21, a22, b11, b21 = 0, 0, 0, 0, 0, 0
        a1, a2, a3, a4, s, s_sign, H, sat, L, T = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        if (vx <= 0.0):
            vx += 0.0000001
        a11 = -(self.Cf + self.Cr) / (self.m * vx)
        a12 = -vx + (self.Cr * self.lr - self.Cf * self.lf) / (self.m * vx)
        a21 = (self.Cr * self.lr - self.Cf * self.lf) / (self.Iz * vx)
        a22 = -(self.Cr * self.lr * self.lr + self.Cf * self.lf * self.lf) / (self.Iz * vx)
        b11 = self.Cf / self.m
        b21 = (self.Cf * self.lf) / self.Iz
        L = abs(es)#预瞄距离
        T = 0.1#采样时间 0.1
        a1 = a11 + L * a21
        a2 = b11 + L * b21
        a3 = -(a11 + L * a21) * vx
        a4 = a12 + L * a22 - L * a11 - L * L * a21 + vx
        H = a3 * ephi + a4 * ephi_dot
        s = cp * ed + ed_dot
        s_sign = self.sign(s)
        if (abs(s) > dp):
            sat = s_sign
        else:
            sat = s / dp
        self.steer_FB = -((cp + a1) * ed_dot + kp * s * T + 0.5 * T * abs(s) * sat + H) / a2
    
    #反馈LQR控制,直接输出反馈前轮转角，单位为度deg，还需根据速度查LQR系数
    def control_LQR(self, ed, ed_dot, ephi, ephi_dot):
        ste = -(ephi * self.LQR_K1[0] + ephi_dot * self.LQR_K1[1] + ed * 10 * self.LQR_K1[2] + ed_dot * 10 * self.LQR_K1[3])
    
    def control_spd(self, vx, v_lim):
        self.ax_desire = 0,
        brk_des = 0
        self.ax_desire = (v_lim - vx) * 0.5
        self.ax_desire = self.limit(self.ax_desire, -3.0, 3.0)
        brk_des = self.limit(brk_des, 0.0, 3.0)

        return brk_des
    
    def SpeedSetWhileNoLaneFree(self):
        host_lane_ob_id = []
        nearest_ob = 99999
        ob_speed_delect = 1000
        # print("No Free Lane!")
        for obi in range(len(self.ob_car_detect)):
            if ((self.ob_car_detect[obi].m_lane == self.lane_ego) and (self.ob_car_detect[obi].ms > self.Station)):
                host_lane_ob_id.append(obi)
        for obi in range(len(host_lane_ob_id)):
            if (self.ob_car_detect[obi].ms - self.Station <= nearest_ob):
                nearest_ob = abs(self.ob_car_detect[obi].ms - self.Station)
                ob_speed_delect = abs(self.ob_car_detect[obi].mv)
        self.spd_max_direct = ob_speed_delect
    
    def Poly_proc(self,):
        if self.Vx < 20:
            self.smk = 20
            self.smc = 300
        else:
            self.smk = 100
            self.smc = 1000

        if self.is_first_run:
            self.is_first_run = False

        if (self.Lat_veh >= 0.5 * self.lane_width and self.Lat_veh <= 1.5 * self.lane_width):
            self.lane_ego = 1
        elif (self.Lat_veh >= -0.5 * self.lane_width and self.Lat_veh <= 0.5 * self.lane_width):
            self.lane_ego = 0
        elif (self.Lat_veh >= -1.5 * self.lane_width and self.Lat_veh <= -0.5 * self.lane_width):
            self.lane_ego = -1
        elif (self.Lat_veh >= -2.5 * self.lane_width and self.Lat_veh <= -1.5 * self.lane_width):
            self.lane_ego = -2
        elif (self.Lat_veh >= -3.5 * self.lane_width and self.Lat_veh <= -2.5 * self.lane_width):
            self.lane_ego = -3
        else:
            self.lane_ego = -4

        self.index2s(len(self.road_x),self.x_proj,self.y_proj,self.im)
        if (self.Vx == 0.0):
            self.Vx += 0.0000001
        if (self.count_p == 10):
            self.path_plan(self.Station, self.Lat_veh, self.Vx)
            self.count_p = 0
        self.count_p+=1
        if (self.no_lane_free):
            self.SpeedSetWhileNoLaneFree()

        self.Frenet2Cartesian(self.Station,self.s_f)
        self.err_cal_module(self.Xo, self.Yo, self.Yaw, self.Vx, self.Vy, self.Yawrate, self.path_x[0], self.path_y[0], self.path_h[0], self.path_k[0])#质心处偏差计算

        for i in range(5):
            ir = (i + 1) * self.pre_n
            self.edx[i], self.dedx[i], self.ephix[i], self.dephix[i], self.esx[i] = self.err_cal_module_return(self.Xo, self.Yo, self.Yaw, self.Vx, self.Vy, self.Yawrate, self.path_x[ir], self.path_y[ir], self.path_h[ir], self.path_k[ir])
            self.rowx[i] = self.path_k[ir]

        self.rowL = self.G0 * self.rowx[0] + self.G1 * self.rowx[1] + self.G2 * self.rowx[2] + self.G3 * self.rowx[3] + self.G4 * self.rowx[4]
        self.edL = self.G0 * self.edx[0] + self.G1 * self.edx[1] + self.G2 * self.edx[2] + self.G3 * self.edx[3] + self.G4 * self.edx[4]
        self.dedL = self.G0 * self.dedx[0] + self.G1 * self.dedx[1] + self.G2 * self.dedx[2] + self.G3 * self.dedx[3] + self.G4 * self.dedx[4]
        self.esL = self.G0 * self.esx[0] + self.G1 * self.esx[1] + self.G2 * self.esx[2] + self.G3 * self.esx[3] + self.G4 * self.esx[4]
        self.esL = abs(self.esL)

        self.control_FF(self.Vx, self.rowL)
        self.control_FB(self.edL, self.dedL, self.ephi0, self.dephi0, self.esL, self.Vx, self.smk, self.smc, self.smd)


        self.steer = self.k_ff * self.steer_FF + self.k_fb * self.steer_FB
        self.steer = self.limit(self.steer, -40.0, 40.0)

        brk_temp = 0
        brk_temp = self.control_spd(self.Vx, self.spd_des)
    
    def Poly_road_init(self,lane):
        self.road_x = deepcopy(np.array(self.road_info_dict_pingjie[str(lane)]['center_vertices'])[:,0])
        self.road_y = deepcopy(np.array(self.road_info_dict_pingjie[str(lane)]['center_vertices'])[:,1])
        self.road_heading = deepcopy(np.array(self.road_info_dict_pingjie[str(lane)]['phi_road']))
        self.road_kappa = deepcopy(np.array(self.road_info_dict_pingjie[str(lane)]['curvature']))
        self.road_station = deepcopy(np.array(self.road_info_dict_pingjie[str(lane)]['station']))

        #更新路点后需要重新初始化
        self.a_0 = 0
        self.b_0 = 0
        self.c_0 = 0
        self.d_0 = 0
        self.e_0 = 0
        self.f_0 = 0
        self.im_pre = 0
        self.lf = self.L/2
        self.lr = self.L/2
        self.lane_width = 3.5
        self.edge_left, self.edge_right = self.lane_width/2, -self.lane_width/2
        
    








    



        


        
    
    





