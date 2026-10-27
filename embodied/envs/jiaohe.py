import numpy as np
import math
import pandas as pd
from collections import deque
from copy import deepcopy
from scipy.linalg import expm
import os

class jiaohe_yundong():
    def __init__(self,dt):
        self.dt = dt
        self.rot_temp = None
        self.v_temp = None
        self.a_temp = None
        self.yaw_yemp = None

        self.m = 1134#整车质量 kg
        self.ms = 1008.1#簧上质量 kg
        self.muf = 71.4#前轴簧下质量 kg
        self.mur = 54.5#后轴簧下质量 kg
        self.lz = 1343.1#车辆横摆转动惯量 kg·m2
        self.lx = 440.6#车辆侧倾转动惯量 kg·m2
        self.lf = 1.04#车辆质心到前轴的距离 m
        self.lr = 1.56#车辆质心到后轴的距离 m
        self.L = 2.6#轴距 m
        self.B = 1.48#轮距 m
        self.hcg = 0.54#车辆质心离地高度 m
        self.hr = 0.165#侧倾中心离地距离 m
        self.kphif = 22002#车辆前悬架侧倾角刚度 N·m/rad
        self.kphir = 14381#车辆前后架侧倾角刚度 N·m/rad
        self.bphif = 2000#车辆前悬架侧倾角阻尼 N·m·s/rad
        self.bphir = 2000#车辆前后架侧倾角阻尼 N·m·s/rad
        self.f = 0.008#车轮滚动阻力系数 
        self.CD = 0.3#空气阻力系数
        self.ru = 1.206#空气密度 N·s2/m-4
        self.A = 1.6#汽车正面迎风面积 m2
        self.lw = 0.8#车轮转动惯量 kg·m2
        self.Rw = 0.287#车轮滚动半径 m
        self.g = 9.8#重力加速度 kg·m/s-2
        self.Cyf = 43160#前轮侧偏刚度 N/rad
        self.Cyr = 29210#后轮侧偏刚度 N/rad
        self.Cxf = 89520#前轮纵向刚度 N/rad
        self.Cxr = 60740#后轮纵向刚度 N/rad

        # self.rot_pre = deque(maxlen=10)

    def update(self, v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        # v_now = v_now + a_now * self.dt
        #初始校核，不扣分
        # print('1: ',rot_chushi)
        if abs(rot_chushi) >= 0.7:
            if rot_chushi >= 0:
                rot_chushi = 0.7
            else:
                rot_chushi = -0.7
        # print('2: ',rot_chushi)
        # if self.rot_temp != None:
        if abs((rot_chushi - rot_now) / self.dt) >= 1.4:
            if (rot_chushi - rot_now) <= 0:
                rot_chushi = -1.4 * self.dt + rot_now
            else:
                rot_chushi = 1.4 * self.dt + rot_now
        
        # if abs(a_chushi - a_now) >= 49*self.dt:
        #     if a_chushi - a_now > 0:
        #         a_chushi = a_now + 49*self.dt
        #     else:
        #         a_chushi = a_now - 49*self.dt
        # print('3: ',rot_chushi)
        #第二次校核，不通过0分
        a_chushi, rot_chushi = self.update_end(v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info)

        self.rot_temp = rot_now
        self.v_temp = v_now
        self.a_temp = a_now
        # if self.yaw_yemp != None:
        #     if abs(yaw_now - self.yaw_yemp) > math.pi:
        #         yaw_now = yaw_now - 2*math.pi*self.sign((yaw_now - self.yaw_yemp))
        self.yaw_yemp = yaw_now
        return a_chushi, rot_chushi

    def update_end(self,v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        beta_max = abs(math.atan2(0.02*0.85*9.8/1,1))*0.9#质心侧偏角
        r_max = abs(9.8*0.85/((v_now)*1+1e-6))*0.9#横摆角速度
        ceqing_max = abs(1/30*math.pi/1)*0.9#车身侧倾角

        rot_chushi_1 = 99999
        rot_chushi_2 = 99999
        
        k1 = math.atan2(self.lr*math.tan(rot_chushi)/self.L,1)
        k2 = v_now*math.cos(math.atan2(self.lr*math.tan(rot_chushi)/self.L,1))*math.tan(math.atan2(self.lr*math.tan(rot_chushi)/self.L,1))/self.lr

        i = 0
        while True:
            if abs(k1) < beta_max and abs(k2) < r_max:
                break
            rot_chushi = rot_chushi * 0.9
            k1 = math.atan2(self.lr*math.tan(rot_chushi)/self.L,1)
            k2 = v_now*math.cos(math.atan2(self.lr*math.tan(rot_chushi)/self.L,1))*math.tan(math.atan2(self.lr*math.tan(rot_chushi)/self.L,1))/self.lr
            if i >= 50:
                break
            i += 1


        # if abs(math.atan2(self.lr*math.tan(rot_chushi)/self.L,1)) > beta_max:
        #     rot_chushi_1 = abs(math.atan2(math.tan(beta_max)*self.L/self.lr,1))

        # if abs(v_now*math.cos(math.atan2(self.lr*math.tan(rot_chushi)/self.L,1))*math.tan(math.atan2(self.lr*math.tan(rot_chushi)/self.L,1))/self.lr) >= r_max:
        #     rot_chushi_2 = abs(math.atan2(math.tan(math.asin(r_max*self.lr/v_now))*self.L/self.lr,1))

        # if math.sin(2*math.atan2(self.lr*math.tan(rot_chushi)/self.L,1)) >= 1.7*9.8*0.9*self.lr/(v_now**2+1e-6):
        #     if math.sin(2*math.atan2(self.lr*math.tan(rot_chushi)/self.L,1)) > 0:
        #         rot_chushi_2 = math.atan2(math.tan(math.asin(1.7*9.8*0.9*self.lr/(v_now**2+1e-6))/2)/self.lr*self.L,1)
        #     else:
        #         rot_chushi_2 = math.atan2(math.tan(math.asin(-1.7*9.8*0.9*self.lr/(v_now**2+1e-6))/2)/self.lr*self.L,1)
        
        # if rot_chushi_1 != 99999 and rot_chushi_1 != 99999:
        #     if abs(rot_chushi_1) > abs(rot_chushi_2):
        #         rot_chushi = rot_chushi_2*self.sign(rot_chushi)
        #     else:
        #         rot_chushi = rot_chushi_1*self.sign(rot_chushi)
        
        rot_chushi = rot_chushi
        # self.rot_pre.append(rot_chushi)
        # total = 0
        # for i in range(len(self.rot_pre)):
        #     total += self.rot_pre[i]*(0.9**(len(self.rot_pre)-i))
        # rot_chushi = total / len(self.rot_pre)

        # print("rot_chushi_1: ",rot_chushi_1,"rot_chushi_2: ",rot_chushi_2)

        return a_chushi, rot_chushi

    def sign(self,x):
        if x > 0:
            return 1
        if x == 0:
            return 0
        if x < 0:
            return -1


#二自由度
class jiaohe_dongli_1():
    def __init__(self,dt):
        self.dt = dt
        self.rot_temp = None
        self.v_temp = None
        self.a_temp = None
        self.yaw_yemp = None

        self.m = 1134#整车质量 kg
        self.ms = 1008.1#簧上质量 kg
        self.muf = 71.4#前轴簧下质量 kg
        self.mur = 54.5#后轴簧下质量 kg
        self.lz = 1343.1#车辆横摆转动惯量 kg·m2
        self.lx = 440.6#车辆侧倾转动惯量 kg·m2
        self.lf = 1.04#车辆质心到前轴的距离 m
        self.lr = 1.56#车辆质心到后轴的距离 m
        self.L = 2.6#轴距 m
        self.B = 1.48#轮距 m
        self.hcg = 0.54#车辆质心离地高度 m
        self.hr = 0.165#侧倾中心离地距离 m
        self.kphif = 22002#车辆前悬架侧倾角刚度 N·m/rad
        self.kphir = 14381#车辆前后架侧倾角刚度 N·m/rad
        self.bphif = 2000#车辆前悬架侧倾角阻尼 N·m·s/rad
        self.bphir = 2000#车辆前后架侧倾角阻尼 N·m·s/rad
        self.f = 0.008#车轮滚动阻力系数 
        self.CD = 0.3#空气阻力系数
        self.ru = 1.206#空气密度 N·s2/m-4
        self.A = 1.6#汽车正面迎风面积 m2
        self.lw = 0.8#车轮转动惯量 kg·m2
        self.Rw = 0.287#车轮滚动半径 m
        self.g = 9.8#重力加速度 kg·m/s-2
        self.Cyf = 43160#前轮侧偏刚度 N/rad
        self.Cyr = 29210#后轮侧偏刚度 N/rad
        self.Cxf = 89520#前轮纵向刚度 N/rad
        self.Cxr = 60740#后轮纵向刚度 N/rad

        self.x_0 = np.array([[0],[0]])
        self.x_1 = np.array([[0],[0]])
        self.step = 1

    def update(self, v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        # v_now = v_now + a_now * self.dt
        #初始校核，不扣分
        # print('1: ',rot_chushi)
        if abs(rot_chushi) >= 0.7:
            if rot_chushi >= 0:
                rot_chushi = 0.7
            else:
                rot_chushi = -0.7
        # print('2: ',rot_chushi)
        # if self.rot_temp != None:
        if abs((rot_chushi - rot_now) / self.dt) >= 1.4:
            if (rot_chushi - rot_now) <= 0:
                rot_chushi = -1.4 * self.dt + rot_now
            else:
                rot_chushi = 1.4 * self.dt + rot_now
        
        # if abs(a_chushi - a_now) >= 49*self.dt:
        #     if a_chushi - a_now > 0:
        #         a_chushi = a_now + 49*self.dt
        #     else:
        #         a_chushi = a_now - 49*self.dt
        # print('3: ',rot_chushi)
        #第二次校核，不通过0分
        a_chushi, rot_chushi = self.update_end(v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info)

        self.rot_temp = rot_now
        self.v_temp = v_now
        self.a_temp = a_now
        # if self.yaw_yemp != None:
        #     if abs(yaw_now - self.yaw_yemp) > math.pi:
        #         yaw_now = yaw_now - 2*math.pi*self.sign((yaw_now - self.yaw_yemp))
        self.yaw_yemp = yaw_now
        return a_chushi, rot_chushi

    def update_end(self,v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        # k_rot = self.sign(rot_chushi)
        # print("input: ",rot_chushi)
        # print("step: ", self.step*self.dt)
        # self.step += 1
        # print('开始：',self.x_0,self.x_1)
        beta_max = abs(math.atan2(0.02*0.85*9.8/1,1))*0.8#质心侧偏角
        r_max = abs(9.8*0.85/(v_now*1+1e-6))*0.8#横摆角速度
        ceqing_max = abs(1/30*math.pi/1)*0.8#车身侧倾角

        rot_chushi_1 = 99999
        rot_chushi_2 = 99999

        M = np.array([[self.lz,0],[0,self.m*(v_now+1e-6)]])
        c11 = 2*(self.Cyf*(self.lf**2) + self.Cyr*(self.lr**2)) / (v_now + 1e-6)
        c12 = 2*(self.Cyf*self.lf - self.Cyr*self.lr)
        c21 = self.m*v_now + 2*(self.Cyf*self.lf - self.Cyr*self.lr) / (v_now + 1e-6)
        c22 = 2*(self.Cyf + self.Cyr)
        C = np.array([[c11,c12],[c21,c22]])
        N = np.array([[2*self.Cyf*self.lf],[2*self.Cyr*self.lr]])
        A = -np.linalg.solve(M,C)
        B = np.linalg.solve(M,N)
        # A = np.array([[(self.Cyf + self.Cyr)/(self.m*v_now+1e-6),(self.Cyf*self.lf - self.Cyr*self.lr)/(self.m*v_now*v_now+1e-6)-1],[(self.Cyf*self.lf - self.Cyr*self.lr)/self.lz,(self.Cyf*(self.lf**2) + self.Cyr*(self.lr**2))/self.lz/(v_now+1e-6)]])
        E = np.array([[1,0],[0,1]])
        # A = E + self.dt * A
        # B = self.dt * B
        F = expm(A*self.dt)
        G = np.dot(np.linalg.solve(A,(F-E)),B)
        x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)
        beta = x[1][0]
        r = x[0][0]

        # print('开始进入迭代')
        i = 0
        while True:
            if abs(beta) < beta_max and abs(r) < r_max:
                break
            # print("2rot_chushi: ", rot_chushi)
            rot_chushi = 0.9*rot_chushi
            x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)
            beta = x[1][0]
            r = x[0][0]
            i += 1
            if i >= 100:
                break
        #     print("beta: ", x[1][0], "beta_max: ", beta_max, "r: ", x[0][0], "r_max: ", r_max)
        # print('退出迭代')


        # if abs(beta) >= beta_max or abs(r) >= r_max:
        #     x = np.array([[beta_max],[r_max]])
        #     k = x - np.dot(F,self.x_1)
        #     rot_chushi_1 = abs(k[0][0] / (G[0][0] + 1e-6))
        #     rot_chushi_2 = abs(k[1][0] / (G[1][0] + 1e-6))
        
        # if rot_chushi_1 != 99999 and rot_chushi_1 != 99999:
        #     if abs(rot_chushi_1) > abs(rot_chushi_2):
        #         rot_chushi = rot_chushi_2*self.sign(rot_chushi)
        #         # x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)
        #     else:
        #         rot_chushi = rot_chushi_1*self.sign(rot_chushi)
        #         # x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)

        # print("self.x_1: ", self.x_1[0][0], self.x_1[1][0])
        # self.x_0 = deepcopy(self.x_1)
        # self.x_1 = deepcopy(x)
        
        # if x[0][0] > beta_max or x[1][0] > r_max:
        #     print('over !')
        # print("beta: ", x[0][0], "beta_max: ", beta_max, "r: ", x[1][0], "r_max: ", r_max)

        # rot_chushi = rot_chushi * 0.9

        rot_chushi = rot_chushi
        return a_chushi, rot_chushi
    
    def update_temp(self, v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        M = np.array([[self.lz,0],[0,self.m*(v_now+1e-6)]])
        c11 = 2*(self.Cyf*(self.lf**2) + self.Cyr*(self.lr**2)) / (v_now + 1e-6)
        c12 = 2*(self.Cyf*self.lf - self.Cyr*self.lr)
        c21 = self.m*v_now + 2*(self.Cyf*self.lf - self.Cyr*self.lr) / (v_now + 1e-6)
        c22 = 2*(self.Cyf + self.Cyr)
        C = np.array([[c11,c12],[c21,c22]])
        N = np.array([[2*self.Cyf*self.lf],[2*self.Cyr*self.lr]])
        A = -np.linalg.solve(M,C)
        B = np.linalg.solve(M,N)
        E = np.array([[1,0],[0,1]])
        # A = E + self.dt * A
        # B = self.dt * B
        F = expm(A*self.dt)
        G = np.dot(np.linalg.solve(A,(F-E)),B)
        x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)

        self.x_0 = deepcopy(self.x_1)
        self.x_1 = deepcopy(x)


    def sign(self,x):
        if x > 0:
            return 1
        if x == 0:
            return 0
        if x < 0:
            return -1


#三自由度
class jiaohe_dongli_2():
    def __init__(self,dt):
        self.dt = dt
        self.rot_temp = None
        self.v_temp = None
        self.a_temp = None
        self.yaw_yemp = None

        self.m = 1134#整车质量 kg
        self.ms = 1008.1#簧上质量 kg
        self.muf = 71.4#前轴簧下质量 kg
        self.mur = 54.5#后轴簧下质量 kg
        self.lz = 1343.1#车辆横摆转动惯量 kg·m2
        self.lx = 440.6#车辆侧倾转动惯量 kg·m2
        self.lf = 1.04#车辆质心到前轴的距离 m
        self.lr = 1.56#车辆质心到后轴的距离 m
        self.L = 2.6#轴距 m
        self.B = 1.48#轮距 m
        self.hcg = 0.54#车辆质心离地高度 m
        self.hr = 0.165#侧倾中心离地距离 m
        self.kphif = 22002#车辆前悬架侧倾角刚度 N·m/rad
        self.kphir = 14381#车辆前后架侧倾角刚度 N·m/rad
        self.bphif = 2000#车辆前悬架侧倾角阻尼 N·m·s/rad
        self.bphir = 2000#车辆前后架侧倾角阻尼 N·m·s/rad
        self.f = 0.008#车轮滚动阻力系数 
        self.CD = 0.3#空气阻力系数
        self.ru = 1.206#空气密度 N·s2/m-4
        self.A = 1.6#汽车正面迎风面积 m2
        self.lw = 0.8#车轮转动惯量 kg·m2
        self.Rw = 0.287#车轮滚动半径 m
        self.g = 9.8#重力加速度 kg·m/s-2
        self.Cyf = 43160#前轮侧偏刚度 N/rad
        self.Cyr = 29210#后轮侧偏刚度 N/rad
        self.Cxf = 89520#前轮纵向刚度 N/rad
        self.Cxr = 60740#后轮纵向刚度 N/rad

        self.x_0 = np.array([[0],[0],[0],[0]])
        self.x_1 = np.array([[0],[0],[0],[0]])
        self.step = 1

        self.s_fl = 0.15#初始滑移率
        self.s_fr = 0.14
        self.s_rl = 0.13
        self.s_rr = 0.12
        self.w_fl = 0
        self.w_fr = 0
        self.w_rl = 0
        self.w_rr = 0


    def update(self, v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        # v_now = v_now + a_now * self.dt
        #初始校核，不扣分
        # print('1: ',rot_chushi)
        if abs(rot_chushi) >= 0.7:
            if rot_chushi >= 0:
                rot_chushi = 0.7
            else:
                rot_chushi = -0.7
        # print('2: ',rot_chushi)
        # if self.rot_temp != None:
        if abs((rot_chushi - rot_now) / self.dt) >= 1.4:
            if (rot_chushi - rot_now) <= 0:
                rot_chushi = -1.4 * self.dt + rot_now
            else:
                rot_chushi = 1.4 * self.dt + rot_now
        
        # if abs(a_chushi - a_now) >= 49*self.dt:
        #     if a_chushi - a_now > 0:
        #         a_chushi = a_now + 49*self.dt
        #     else:
        #         a_chushi = a_now - 49*self.dt

        # print('3: ',rot_chushi)
        #第二次校核，不通过0分
        a_chushi, rot_chushi = self.update_end(v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info)

        self.rot_temp = rot_now
        self.v_temp = v_now
        self.a_temp = a_now
        # if self.yaw_yemp != None:
        #     if abs(yaw_now - self.yaw_yemp) > math.pi:
        #         yaw_now = yaw_now - 2*math.pi*self.sign((yaw_now - self.yaw_yemp))
        self.yaw_yemp = yaw_now
        return a_chushi, rot_chushi

    def update_end(self,v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        # k_rot = self.sign(rot_chushi)
        # print("input: ",v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi)
        # print("step: ", self.step*self.dt)
        # self.step += 1
        # print('开始：',self.x_0,self.x_1)
        beta_max = abs(math.atan2(0.02*0.85*9.8/1,1))*0.8#质心侧偏角
        r_max = abs(9.8*0.85/(v_now*1+1e-6))*0.8#横摆角速度
        ceqing_max = abs(1/30*math.pi/1)*0.8#车身侧倾角

        rot_chushi_1 = 99999
        rot_chushi_2 = 99999

        M = np.array([[self.lz,0,0,0],
                      [0,self.m*(v_now+1e-6),-self.ms*self.hr,0],
                      [0,-self.ms*self.hr*(v_now+1e-6),self.lx,0],
                      [0,0,0,1]])
        C = np.array([[2*(self.Cyf*(self.lf**2) + self.Cyr*(self.lr**2)) / (v_now + 1e-6),2*(self.Cyf*self.lf - self.Cyr*self.lr),0,-2*(self.Cyf*self.lf*(-0.05) - self.Cyr*self.lr*(0.05))],
                      [self.m*(v_now+1e-6)+2*(self.Cyf*self.lf - self.Cyr*self.lr) / (v_now + 1e-6),2*(self.Cyf+self.Cyr),0,-2*(self.Cyf*(-0.05) + self.Cyr*(0.05))],
                      [-self.ms*self.hr*(v_now+1e-6),0,self.bphif+self.bphir,self.kphif+self.kphir-self.ms*self.hr*self.g],
                      [0,0,-1,0]])
        N = np.array([[2*self.lf*self.Cyf,-self.B/2,-self.B/2],[2*self.lf,0,0],[0,0,0],[0,0,0]])
        A = -np.linalg.solve(M,C)
        B = np.linalg.solve(M,N)
        E = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        # A = E + self.dt * A
        # B = self.dt * B
        F = expm(A*self.dt)
        G = np.dot(np.linalg.solve(A,(F-E)),B)

        #轮胎纵向力计算
        qianlun_cepian_angle_fl = rot_now - math.atan2((v_now*math.sin(self.x_1[0][0])+self.Cyf*self.x_1[1][0])/(v_now*math.cos(self.x_1[0][0])+1e-6-self.x_1[0][0]*self.x_1[1][0]/2),1)
        qianlun_cepian_angle_fr = rot_now - math.atan2((v_now*math.sin(self.x_1[0][0])+self.Cyf*self.x_1[1][0])/(v_now*math.cos(self.x_1[0][0])+1e-6+self.x_1[0][0]*self.x_1[1][0]/2),1)
        houlun_cepian_angle_rl = - math.atan2((v_now*math.sin(0)-self.Cyr*self.x_1[1][0])/(v_now*math.cos(0)+1e-6-self.x_1[0][0]*self.x_1[1][0]/2),1)
        houlun_cepian_angle_rr = - math.atan2((v_now*math.sin(0)-self.Cyr*self.x_1[1][0])/(v_now*math.cos(0)+1e-6+self.x_1[0][0]*self.x_1[1][0]/2),1)
        Td_f = self.m*(a_now*math.cos(self.x_1[0][0])-v_now*math.cos(self.x_1[0][0])*self.x_1[0][0]*self.x_1[1][0])*self.Rw/4
        Td_r = Td_f

        Fzfl = self.m*self.g*self.lr/(2*self.L)-self.m*(a_now*math.cos(self.x_1[0][0])-self.x_1[1][0]*v_now*math.sin(self.x_1[0][0]))*self.hcg/(2*self.L)-self.m*(a_now*math.sin(self.x_1[0][0])+self.x_1[1][0]*v_now*math.cos(self.x_1[0][0]))*self.lr*self.hcg/(self.L*self.B)
        Fzfr = self.m*self.g*self.lr/(2*self.L)-self.m*(a_now*math.cos(self.x_1[0][0])-self.x_1[1][0]*v_now*math.sin(self.x_1[0][0]))*self.hcg/(2*self.L)+self.m*(a_now*math.sin(self.x_1[0][0])+self.x_1[1][0]*v_now*math.cos(self.x_1[0][0]))*self.lr*self.hcg/(self.L*self.B)
        Fzrl = self.m*self.g*self.lf/(2*self.L)+self.m*(a_now*math.cos(self.x_1[0][0])-self.x_1[1][0]*v_now*math.sin(self.x_1[0][0]))*self.hcg/(2*self.L)-self.m*(a_now*math.sin(self.x_1[0][0])+self.x_1[1][0]*v_now*math.cos(self.x_1[0][0]))*self.lf*self.hcg/(self.L*self.B)
        Fzrr = self.m*self.g*self.lf/(2*self.L)+self.m*(a_now*math.cos(self.x_1[0][0])-self.x_1[1][0]*v_now*math.sin(self.x_1[0][0]))*self.hcg/(2*self.L)+self.m*(a_now*math.sin(self.x_1[0][0])+self.x_1[1][0]*v_now*math.cos(self.x_1[0][0]))*self.lf*self.hcg/(self.L*self.B)
        fz_fl = (0.85*Fzfl*(1-self.s_fl)/(2*math.pow(math.pow(self.Cxf*self.s_fl,2)+math.pow(self.Cyf*math.tan(qianlun_cepian_angle_fl),2),0.5)+1e-6))
        if fz_fl < 1:
            fz_fl = (2-fz_fl)*fz_fl
        else:
            fz_fl = 1
        fx_fl = self.Cxf*self.s_fl/(1+self.s_fl+1e-6)*fz_fl
        fz_fr = (0.85*Fzfr*(1-self.s_fr)/(2*math.pow(math.pow(self.Cxf*self.s_fr,2)+math.pow(self.Cyf*math.tan(qianlun_cepian_angle_fr),2),0.5)+1e-6))
        if fz_fr < 1:
            fz_fr = (2-fz_fr)*fz_fr
        else:
            fz_fr = 1
        fx_fr = self.Cxf*self.s_fr/(1+self.s_fr+1e-6)*fz_fr
        fz_rl = (0.85*Fzrl*(1-self.s_rl)/(2*math.pow(math.pow(self.Cxr*self.s_rl,2)+math.pow(self.Cyr*math.tan(houlun_cepian_angle_rl),2),0.5)+1e-6))
        if fz_rl < 1:
            fz_rl = (2-fz_rl)*fz_rl
        else:
            fz_rl = 1
        fx_rl = self.Cxr*self.s_rl/(1+self.s_rl+1e-6)*fz_rl
        fz_rr = (0.85*Fzrr*(1-self.s_rr)/(2*math.pow(math.pow(self.Cxr*self.s_rr,2)+math.pow(self.Cyr*math.tan(houlun_cepian_angle_rr),2),0.5)+1e-6))
        if fz_rr < 1:
            fz_rr = (2-fz_rr)*fz_rr
        else:
            fz_rr = 1
        fx_rr = self.Cxr*self.s_rr/(1+self.s_rr+1e-6)*fz_rr

        w_fl1 = (Td_f - fx_fl*self.Rw)/self.lw + self.w_fl
        self.s_fl = (w_fl1*self.Rw - ((v_now*math.cos(self.x_1[0][0])-self.B*self.x_1[1][0]/2)*math.cos(rot_now)+(v_now*math.sin(self.x_1[0][0])+self.lf*self.x_1[1][0])*math.sin(rot_now)))/((v_now*math.cos(self.x_1[0][0])-self.B*self.x_1[1][0]/2)*math.cos(rot_now)+(v_now*math.sin(self.x_1[0][0])+self.lf*self.x_1[1][0])*math.sin(rot_now) + 1e-6)
        self.w_fl = w_fl1
        w_fr1 = (Td_f - fx_fr*self.Rw)/self.lw + self.w_fr
        self.s_fr = (w_fr1*self.Rw - ((v_now*math.cos(self.x_1[0][0])-self.B*self.x_1[1][0]/2)*math.cos(rot_now)+(v_now*math.sin(self.x_1[0][0])+self.lf*self.x_1[1][0])*math.sin(rot_now)))/((v_now*math.cos(self.x_1[0][0])-self.B*self.x_1[1][0]/2)*math.cos(rot_now)+(v_now*math.sin(self.x_1[0][0])+self.lf*self.x_1[1][0])*math.sin(rot_now) + 1e-6)
        self.w_fr = w_fr1
        w_rl1 = (Td_r - fx_rl*self.Rw)/self.lw + self.w_rl
        self.s_rl = (w_rl1*self.Rw - (v_now - self.B*self.x_1[1][0]/2))/(v_now - self.B*self.x_1[1][0]/2 + 1e-6)
        self.w_rl = w_rl1
        w_rr1 = (Td_r - fx_rr*self.Rw)/self.lw + self.w_rr
        self.s_rr = (w_rr1*self.Rw - (v_now - self.B*self.x_1[1][0]/2))/(v_now - self.B*self.x_1[1][0]/2 + 1e-6)
        self.w_rr = w_rr1

        fx1 = (fx_fl - fx_fr)
        fx2 = (fx_rl - fx_rr)
        U = np.array([[rot_chushi],[fx1],[fx2]])
        x = np.dot(F,self.x_1) + np.dot(G,U)
        beta = x[1][0]
        r = x[0][0]
        ceqing = x[3][0]

        i = 0
        while True:
            if abs(beta) <= beta_max and abs(r) <= r_max and abs(ceqing) <= ceqing_max:
                break
            # print("3rot_chushi: ", rot_chushi)
            rot_chushi = 0.9 * rot_chushi
            x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)
            beta = x[1][0]
            r = x[0][0]
            ceqing = x[3][0]
            i += 1
            if i >= 100:
                break
        
        # if rot_chushi_1 != 99999 and rot_chushi_1 != 99999:
        #     if abs(rot_chushi_1) > abs(rot_chushi_2):
        #         rot_chushi = rot_chushi_2*self.sign(rot_chushi)
        #         x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)
        #     else:
        #         rot_chushi = rot_chushi_1*self.sign(rot_chushi)
        #         x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)

        # print("self.x_1: ", self.x_1[0][0], self.x_1[1][0])
        # self.x_0 = deepcopy(self.x_1)
        # self.x_1 = deepcopy(x)
        
        # if x[1][0] > beta_max or x[0][0] > r_max or x[3][0] > ceqing_max:
        #     print('over !')
        # print("beta: ", x[1][0], "beta_max: ", beta_max, "r: ", x[0][0], "r_max: ", r_max, "ceqing: ", x[3][0], "ceqing_max: ", ceqing_max)
        # rot_chushi = rot_chushi * 0.9
        rot_chushi = rot_chushi

        return a_chushi, rot_chushi
    
    def update_temp(self, v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        M = np.array([[self.lz,0,0,0],
                      [0,self.m*(v_now+1e-6),-self.ms*self.hr,0],
                      [0,-self.ms*self.hr*(v_now+1e-6),self.lx,0],
                      [0,0,0,1]])
        C = np.array([[2*(self.Cyf*(self.lf**2) + self.Cyr*(self.lr**2)) / (v_now + 1e-6),2*(self.Cyf*self.lf - self.Cyr*self.lr),0,-2*(self.Cyf*self.lf*(-0.05) - self.Cyr*self.lr*(0.05))],
                      [self.m*(v_now+1e-6)+2*(self.Cyf*self.lf - self.Cyr*self.lr) / (v_now + 1e-6),2*(self.Cyf+self.Cyr),0,-2*(self.Cyf*(-0.05) + self.Cyr*(0.05))],
                      [-self.ms*self.hr*(v_now+1e-6),0,self.bphif+self.bphir,self.kphif+self.kphir-self.ms*self.hr*self.g],
                      [0,0,-1,0]])
        N = np.array([[2*self.lf*self.Cyf,-self.B/2,-self.B/2],[2*self.lf,0,0],[0,0,0],[0,0,0]])
        A = -np.linalg.solve(M,C)
        B = np.linalg.solve(M,N)
        E = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        # A = E + self.dt * A
        # B = self.dt * B
        F = expm(A*self.dt)
        G = np.dot(np.linalg.solve(A,(F-E)),B)

        x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)
        self.x_0 = deepcopy(self.x_1)
        self.x_1 = deepcopy(x)

         

    def sign(self,x):
        if x > 0:
            return 1
        if x == 0:
            return 0
        if x < 0:
            return -1

#四自由度
class jiaohe_dongli_3():
    def __init__(self,dt):
        self.dt = dt
        self.rot_temp = None
        self.v_temp = None
        self.a_temp = None
        self.yaw_yemp = None

        self.m = 1134#整车质量 kg
        self.ms = 1008.1#簧上质量 kg
        self.muf = 71.4#前轴簧下质量 kg
        self.mur = 54.5#后轴簧下质量 kg
        self.lz = 1343.1#车辆横摆转动惯量 kg·m2
        self.lx = 440.6#车辆侧倾转动惯量 kg·m2
        self.lf = 1.04#车辆质心到前轴的距离 m
        self.lr = 1.56#车辆质心到后轴的距离 m
        self.L = 2.6#轴距 m
        self.B = 1.48#轮距 m
        self.hcg = 0.54#车辆质心离地高度 m
        self.hr = 0.165#侧倾中心离地距离 m
        self.kphif = 22002#车辆前悬架侧倾角刚度 N·m/rad
        self.kphir = 14381#车辆前后架侧倾角刚度 N·m/rad
        self.bphif = 2000#车辆前悬架侧倾角阻尼 N·m·s/rad
        self.bphir = 2000#车辆前后架侧倾角阻尼 N·m·s/rad
        self.f = 0.008#车轮滚动阻力系数 
        self.CD = 0.3#空气阻力系数
        self.ru = 1.206#空气密度 N·s2/m-4
        self.A = 1.6#汽车正面迎风面积 m2
        self.lw = 0.8#车轮转动惯量 kg·m2
        self.Rw = 0.287#车轮滚动半径 m
        self.g = 9.8#重力加速度 kg·m/s-2
        self.Cyf = 43160#前轮侧偏刚度 N/rad
        self.Cyr = 29210#后轮侧偏刚度 N/rad
        self.Cxf = 89520#前轮纵向刚度 N/rad
        self.Cxr = 60740#后轮纵向刚度 N/rad

        self.x_0 = np.array([[0],[0],[0],[0]])
        self.x_1 = np.array([[0],[0],[0],[0]])
        self.step = 1

    def update(self, v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        v_now = v_now + a_now * self.dt
        #初始校核，不扣分
        # print('1: ',rot_chushi)
        if abs(rot_chushi) >= 0.7:
            if rot_chushi >= 0:
                rot_chushi = 0.7
            else:
                rot_chushi = -0.7
        # print('2: ',rot_chushi)
        # if self.rot_temp != None:
        if abs((rot_chushi - rot_now) / self.dt) >= 1.4:
            if (rot_chushi - rot_now) <= 0:
                rot_chushi = -1.4 * self.dt + rot_now
            else:
                rot_chushi = 1.4 * self.dt + rot_now

        if abs(a_chushi - a_now) >= 49*self.dt:
            if a_chushi - a_now > 0:
                a_chushi = a_now + 49*self.dt
            else:
                a_chushi = a_now - 49*self.dt
        # print('3: ',rot_chushi)
        #第二次校核，不通过0分
        a_chushi, rot_chushi = self.update_end(v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info)

        self.rot_temp = rot_now
        self.v_temp = v_now
        self.a_temp = a_now
        # if self.yaw_yemp != None:
        #     if abs(yaw_now - self.yaw_yemp) > math.pi:
        #         yaw_now = yaw_now - 2*math.pi*self.sign((yaw_now - self.yaw_yemp))
        self.yaw_yemp = yaw_now
        return a_chushi, rot_chushi

    def update_end(self,v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        # print("input: ",rot_chushi)
        # print("step: ", self.step*self.dt)
        # self.step += 1
        # print('开始：',self.x_0,self.x_1)
        beta_max = abs(math.atan2(0.02*0.85*9.8/1,1))*0.95#质心侧偏角
        r_max = abs(9.8*0.85/(v_now*1+1e-6))*0.95#横摆角速度
        ceqing_max = abs(1/30*math.pi/1)*0.95#车身侧倾角

        rot_chushi_1 = 99999
        rot_chushi_2 = 99999

        # M = np.array([[self.lz,0],[0,self.m*(v_now+1e-6)]])
        # c11 = 2*(self.Cyf*(self.lf**2) + self.Cyr*(self.lr**2)) / (v_now + 1e-6)
        # c12 = 2*(self.Cyf*self.lf - self.Cyr*self.lr)
        # c21 = self.m*v_now + 2*(self.Cyf*self.lf - self.Cyr*self.lr) / (v_now + 1e-6)
        # c22 = 2*(self.Cyf + self.Cyr)
        # C = np.array([[c11,c12],[c21,c22]])
        # N = np.array([[2*self.Cyf*self.lf],[2*self.Cyr*self.lr]])
        # A = -np.linalg.solve(M,C)
        # B = np.linalg.solve(M,N)
        A = np.array([[(self.Cyf+self.Cyr)/(self.m*(v_now+1e-6)),(self.Cyf*self.lf-self.Cyr*self.lr)/(self.m*v_now*v_now+1e-6),0,0],
                     [(self.Cyf*self.lf-self.Cyr*self.lr)/self.lz,(self.Cyf*(self.lf**2)-self.Cyr*(self.lr**2))/(self.lz*v_now*v_now+1e-6),0,0],
                     [0,0,0,1],
                     [self.ms*self.hcg*(self.Cyf+self.Cyr)*math.cos(self.x_1[3][0])/(self.lx*self.m),self.ms*self.hcg*(self.Cyf*self.lf-self.Cyr*self.lr)/(self.lx*self.m*v_now+1e-6),-(self.kphif+self.kphir)/2/self.lx,-self.bphir/self.lx]])
        B = np.array([[-self.Cyf/(self.m*v_now+1e-6)],[-self.lf*self.Cyf/self.lz],[0],[self.ms*self.hcg*self.Cyf/(self.lx*self.m)]])
        # A = np.array([[(self.Cyf + self.Cyr)/(self.m*v_now+1e-6),(self.Cyf*self.lf - self.Cyr*self.lr)/(self.m*v_now*v_now+1e-6)-1],[(self.Cyf*self.lf - self.Cyr*self.lr)/self.lz,(self.Cyf*(self.lf**2) + self.Cyr*(self.lr**2))/self.lz/(v_now+1e-6)]])
        E = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
 
        # A = E + self.dt * A
        # B = self.dt * B
        F = expm(A*self.dt)
        G = np.dot(np.linalg.solve(A,(F-E)),B)
        x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)
        beta = x[1][0]
        r = x[0][0]
        # print('chushi: ',beta, r)

        # print('开始进入迭代')
        i = 0
        while True:
            if abs(beta) < beta_max and abs(r) < r_max:
                break
            # print("rot_chushi: ", rot_chushi)
            rot_chushi = 0.95*rot_chushi
            x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)
            beta = x[1][0]
            r = x[0][0]
            i += 1
            if i >= 50:
                break
        #     print("beta: ", x[1][0], "beta_max: ", beta_max, "r: ", x[0][0], "r_max: ", r_max)
        # print('退出迭代')


        # if abs(beta) >= beta_max or abs(r) >= r_max:
        #     x = np.array([[beta_max],[r_max]])
        #     k = x - np.dot(F,self.x_1)
        #     rot_chushi_1 = abs(k[0][0] / (G[0][0] + 1e-6))
        #     rot_chushi_2 = abs(k[1][0] / (G[1][0] + 1e-6))
        
        # if rot_chushi_1 != 99999 and rot_chushi_1 != 99999:
        #     if abs(rot_chushi_1) > abs(rot_chushi_2):
        #         rot_chushi = rot_chushi_2*self.sign(rot_chushi)
        #         # x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)
        #     else:
        #         rot_chushi = rot_chushi_1*self.sign(rot_chushi)
        #         # x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)

        # print("self.x_1: ", self.x_1[0][0], self.x_1[1][0])
        # self.x_0 = deepcopy(self.x_1)
        # self.x_1 = deepcopy(x)
        
        # if x[0][0] > beta_max or x[1][0] > r_max:
        #     print('over !')
        # print("beta: ", x[0][0], "beta_max: ", beta_max, "r: ", x[1][0], "r_max: ", r_max)

        # rot_chushi = rot_chushi * 0.9

        return a_chushi, rot_chushi
    
    def update_temp(self, v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        A = np.array([[(self.Cyf+self.Cyr)/(self.m*(v_now+1e-6)),(self.Cyf*self.lf-self.Cyr*self.lr)/(self.m*v_now*v_now+1e-6),0,0],
                     [(self.Cyf*self.lf-self.Cyr*self.lr)/self.lz,(self.Cyf*(self.lf**2)-self.Cyr*(self.lr**2))/(self.lz*v_now*v_now+1e-6),0,0],
                     [0,0,0,1],
                     [self.ms*self.hcg*(self.Cyf+self.Cyr)*math.cos(self.x_1[3][0])/(self.lx*self.m),self.ms*self.hcg*(self.Cyf*self.lf-self.Cyr*self.lr)/(self.lx*self.m*v_now+1e-6),-(self.kphif+self.kphir)/2/self.lx,-self.bphir/self.lx]])
        B = np.array([[-self.Cyf/(self.m*v_now+1e-6)],[-self.lf*self.Cyf/self.lz],[0],[self.ms*self.hcg*self.Cyf/(self.lx*self.m)]])
        E = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        # A = E + self.dt * A
        # B = self.dt * B
        F = expm(A*self.dt)
        G = np.dot(np.linalg.solve(A,(F-E)),B)
        x = np.dot(F,self.x_1) + np.dot(G,rot_chushi)

        self.x_0 = deepcopy(self.x_1)
        self.x_1 = deepcopy(x)


    def sign(self,x):
        if x > 0:
            return 1
        if x == 0:
            return 0
        if x < 0:
            return -1

#八自由度_new
class jiaohe_dongli_8_new():
    def __init__(self,dt):
        self.dt = dt
        
        self.m = 1134#整车质量 kg
        self.ms = 1008.1#簧上质量 kg
        self.muf = 71.4#前轴簧下质量 kg
        self.mur = 54.5#后轴簧下质量 kg
        self.lz = 1343.1#车辆横摆转动惯量 kg·m2
        self.lx = 440.6#车辆侧倾转动惯量 kg·m2
        self.lf = 1.04#车辆质心到前轴的距离 m
        self.lr = 1.56#车辆质心到后轴的距离 m
        self.L = 2.6#轴距 m
        self.B = 1.48#轮距 m
        self.hcg = 0.54#车辆质心离地高度 m
        self.hr = 0.165#侧倾中心离地距离 m
        self.kphif = 22002#车辆前悬架侧倾角刚度 N·m/rad
        self.kphir = 14381#车辆前后架侧倾角刚度 N·m/rad
        self.bphif = 2000#车辆前悬架侧倾角阻尼 N·m·s/rad
        self.bphir = 2000#车辆前后架侧倾角阻尼 N·m·s/rad
        self.f = 0.008#车轮滚动阻力系数 
        self.CD = 0.3#空气阻力系数
        self.ru = 1.206#空气密度 N·s2/m-4
        self.A = 1.6#汽车正面迎风面积 m2
        self.lw = 0.8#车轮转动惯量 kg·m2
        self.Rw = 0.287#车轮滚动半径 m
        self.g = 9.8#重力加速度 kg·m/s-2
        self.Cyf = 43160*2#前轮侧偏刚度 N/rad
        self.Cyr = 29210*2#后轮侧偏刚度 N/rad
        self.Cxf = 89520*2#前轮纵向刚度 N/rad
        self.Cxr = 60740*2#后轮纵向刚度 N/rad

        self.step = 1

        self.sample_time = 0.01
        self.d_vy1 = 0
        self.d_d_phi1 = 0
        self.s0 = 0.01
        self.Rw0 = 0.28415841584158413
        self.x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.n = round(self.dt / self.sample_time) + 1


    def update(self, v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        #初始校核，不扣分
        # print('1: ',rot_chushi)
        if abs(rot_chushi) >= 0.7:
            if rot_chushi >= 0:
                rot_chushi = 0.7
            else:
                rot_chushi = -0.7
        # print('2: ',rot_chushi)
        # if self.rot_temp != None:
        if abs((rot_chushi - rot_now) / self.dt) >= 1.4:
            if (rot_chushi - rot_now) <= 0:
                rot_chushi = -1.4 * self.dt + rot_now
            else:
                rot_chushi = 1.4 * self.dt + rot_now

        if abs(a_chushi - a_now) >= 49*self.dt:
            if a_chushi - a_now > 0:
                a_chushi = a_now + 49*self.dt
            else:
                a_chushi = a_now - 49*self.dt
        
        if v_now + a_chushi * self.dt <= 0:
            a_chushi = -v_now / self.dt
        # print('3: ',rot_chushi)
        #第二次校核，不通过0分
        a_chushi, rot_chushi = self.update_end(v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info)

        return a_chushi, rot_chushi

    def update_end(self,v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        if self.step == 1:
            omega0 = (v_now + a_chushi*self.dt) / ((1 - self.s0) * self.Rw0)
            self.x = np.array([0, 0, 0, 0, omega0, omega0, omega0, omega0])
        
        beta_max = abs(math.atan2(0.02*0.85*9.8/1,1))*0.95#质心侧偏角
        r_max = abs(9.8*0.85/(v_now*1+1e-6))*0.95#横摆角速度
        ceqing_max = abs(1/30*math.pi/1)*0.95#车身侧倾角

        k = self.cal_need(v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info)

        i = 0
        while True:
            if k == False:
                # print("退出迭代")
                break
            rot_chushi = rot_chushi * 0.9
            k = self.cal_need(v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info)
            if i >= 50:
                # print("退出迭代，未成功", '时刻： ', self.step * self.dt)
                break
            i += 1

        return a_chushi, rot_chushi
    
    def update_temp(self, v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        beta_max = abs(math.atan2(0.02*0.85*9.8/1,1))*0.95#质心侧偏角
        r_max = abs(9.8*0.85/(v_now*1+1e-6))*0.95#横摆角速度
        ceqing_max = abs(1/30*math.pi/1)*0.95#车身侧倾角

        vx = np.array([v_now, v_now + a_chushi * self.dt])
        ax = np.array([a_now, a_chushi])
        deltaf = np.array([rot_now, rot_chushi])
        vx = np.where(vx <= 0.0001, 0.0001, vx)
        t = np.array([0, self.dt])
        u = np.zeros(3)
        ys = np.zeros((self.n, 3))

        if self.step == 1:
            ys = np.zeros((self.n, 3))
        else:
            ys = np.zeros((self.n-1, 3))

        if self.step == 1:
            d_vy1 = deepcopy(self.d_vy1)
            d_d_phi1 = deepcopy(self.d_d_phi1)
            x = deepcopy(self.x)
            u[0],u[1],u[2]=rot_chushi,a_chushi,v_now+a_chushi*self.dt
            dx = self.Devs(x, u, d_vy1, d_d_phi1)
            d_vy1 = dx[0]
            d_d_phi1 = dx[2]
            x = x + dx * self.sample_time
            beta = np.arctan(x[0] / u[2])
            y = [beta, x[1], x[3]]
            ys[0, :] = y
            self.d_vy1 = d_vy1
            self.d_d_phi1 = d_d_phi1
            self.x = x

        for i in range(1, self.n):
            if i == 1:
                d_vy1 = deepcopy(self.d_vy1)
                d_d_phi1 = deepcopy(self.d_d_phi1)
                x = deepcopy(self.x)
            if self.step == 1:
                u[0],u[1],u[2]=rot_chushi,a_chushi,v_now+a_chushi*self.dt
            else:
                u[0] = np.interp(self.sample_time * (i), t, deltaf)
                u[1] = np.interp(self.sample_time * (i), t, ax)
                u[2] = np.interp(self.sample_time * (i), t, vx)
            dx = self.Devs(x, u, d_vy1, d_d_phi1)
            d_vy1 = dx[0]
            d_d_phi1 = dx[2]
            x = x + dx * self.sample_time
            beta = np.arctan(x[0] / u[2])
            y = [beta, x[1], x[3]]
            if self.step == 1:
                ys[i, :] = y
            else:
                ys[i-1, :] = y
        self.d_vy1 = d_vy1
        self.d_d_phi1 = d_d_phi1
        self.x = x

        self.step += 1
    
    def sign(self,x):
        if x > 0:
            return 1
        if x == 0:
            return 0
        if x < 0:
            return -1
    
    def cal_need(self, v_now, a_now, yaw_now, rot_now, a_chushi, rot_chushi, ego_info):
        beta_max = abs(math.atan2(0.02*0.85*9.8/1,1))*0.95#质心侧偏角
        r_max = abs(9.8*0.85/(v_now*1+1e-6))*0.95#横摆角速度
        ceqing_max = abs(1/30*math.pi/1)*0.95#车身侧倾角

        vx = np.array([v_now, v_now + a_chushi * self.dt])
        ax = np.array([a_now, a_chushi])
        deltaf = np.array([rot_now, rot_chushi])
        vx = np.where(vx <= 0.0001, 0.0001, vx)
        t = np.array([0, self.dt])
        u = np.zeros(3)
        ys = np.zeros((self.n, 3))

        if self.step == 1:
            ys = np.zeros((self.n, 3))
        else:
            ys = np.zeros((self.n-1, 3))

        if self.step == 1:
            d_vy1 = deepcopy(self.d_vy1)
            d_d_phi1 = deepcopy(self.d_d_phi1)
            x = deepcopy(self.x)
            u[0],u[1],u[2]=rot_chushi,a_chushi,v_now+a_chushi*self.dt
            dx = self.Devs(x, u, d_vy1, d_d_phi1)
            d_vy1 = dx[0]
            d_d_phi1 = dx[2]
            x = x + dx * self.sample_time
            beta = np.arctan(x[0] / u[2])
            y = [beta, x[1], x[3]]
            ys[0, :] = y
            self.d_vy1 = d_vy1
            self.d_d_phi1 = d_d_phi1
            self.x = x



        for i in range(1, self.n):
            if i == 1:
                d_vy1 = deepcopy(self.d_vy1)
                d_d_phi1 = deepcopy(self.d_d_phi1)
                x = deepcopy(self.x)
            if self.step == 1:
                u[0],u[1],u[2]=rot_chushi,a_chushi,v_now+a_chushi*self.dt
            else:
                u[0] = np.interp(self.sample_time * (i), t, deltaf)
                u[1] = np.interp(self.sample_time * (i), t, ax)
                u[2] = np.interp(self.sample_time * (i), t, vx)
            dx = self.Devs(x, u, d_vy1, d_d_phi1)
            d_vy1 = dx[0]
            d_d_phi1 = dx[2]
            x = x + dx * self.sample_time
            beta = np.arctan(x[0] / u[2])
            y = [beta, x[1], x[3]]
            if self.step == 1:
                ys[i, :] = y
            else:
                ys[i-1, :] = y
        
        miu = 0.85
        g = 9.8
        beta_lim = np.arctan(0.02 * miu * g / 1) * 0.9
        miu_y = 0.85
        if self.step == 1:
            vx1 = np.array([v_now+a_chushi*self.dt]*self.n)
        else:
            vx1 = np.interp(self.sample_time * np.arange(1, self.n), t, vx)
        # print(vx1)
        r_lim = g * miu_y / (vx1 * 1) * 0.9
        phi_lim = 0.03333333333333333 * np.pi / 1 * 0.9
        if np.sum(np.abs(ys[:, 0]) > beta_lim) > 0:
            return True
        if np.sum(np.abs(ys[:, 1]) > r_lim) > 0:
            return True
        if np.sum(np.abs(ys[:, 2]) > phi_lim) > 0:
            return True
        return False


    def Devs(self, x, u, d_vy1, d_d_phi1):
        m = 1134
        mu = 125.9
        ms = m - mu
        Iz = 1343.1
        Ix = 440.6
        Ixz = 0
        lf = 1.04
        L = 2.6
        lr = L - lf
        B = 1.48
        hcg = 0.54
        h = 0.23
        kf = 86320.0
        kr = 58420.0
        Cf = 179040.0
        Cr = 121480.0
        kphif = 69120 / np.pi
        kphir = 45180 / np.pi
        bphif = 2000
        bphir = 2000
        f = 0.008
        CD = 0.3
        ru = 1.206
        A = 1.6
        Iw = 0.8
        Rw = 0.287
        lfs = lf
        lrs = lr
        hrf = 0.165
        hrr = 0.165
        muf = 71.4
        mur = 54.5
        huf = Rw
        hur = Rw
        g = 9.8
        vy = x[0]
        r = x[1]
        d_phi = x[2]
        phi = x[3]
        Wfl = x[4]
        Wfr = x[5]
        Wrl = x[6]
        Wrr = x[7]
        deltaf_out = u[0]
        ax_out = u[1]
        vx_out = u[2]
        vx = vx_out
        deltarl = 0
        deltarr = 0
        Tbfl = 0
        Tbfr = 0
        Tbrl = 0
        Tbrr = 0
        miu = 0.85
        deltafl = np.arctan(np.tan(deltaf_out) / (1 - B / L / 2 * np.tan(deltaf_out)))
        deltafr = np.arctan(np.tan(deltaf_out) / (1 + B / L / 2 * np.tan(deltaf_out)))
        Fx = m * ax_out + f * m * g + CD * A * ru * vx_out ** 2 / 2
        Tall = Fx * Rw
        Delta_T = Tall / 2 * (1 - np.cos(deltaf_out))
        Tdfl = (Tall + Delta_T) / 4
        Tdfr = Tdfl
        Tdrl = Tdfl
        Tdrr = Tdfl
        ufl = (vx - 0.5 * B * r) * np.cos(deltafl) + (vy + lf * r) * np.sin(deltafl)
        ufr = (vx + 0.5 * B * r) * np.cos(deltafr) + (vy + lf * r) * np.sin(deltafr)
        url = (vx - 0.5 * B * r) * np.cos(deltarl) + (vy - lr * r) * np.sin(deltarl)
        urr = (vx + 0.5 * B * r) * np.cos(deltarr) + (vy - lr * r) * np.sin(deltarr)
        alfafl = deltafl - np.arctan((vy + lf * r) / (vx - 0.5 * B * r))
        alfafr = deltafr - np.arctan((vy + lf * r) / (vx + 0.5 * B * r))
        alfarl = deltarl - np.arctan((vy - lr * r) / (vx - 0.5 * B * r))
        alfarr = deltarr - np.arctan((vy - lr * r) / (vx + 0.5 * B * r))
        if ufl < Rw * Wfl:
            sfl = 1 - ufl / (Rw * Wfl)
        else:
            sfl = Rw * Wfl / ufl - 1
        if ufr < Rw * Wfr:
            sfr = 1 - ufr / (Rw * Wfr)
        else:
            sfr = Rw * Wfr / ufr - 1
        if url < Rw * Wrl:
            srl = 1 - url / (Rw * Wrl)
        else:
            srl = Rw * Wrl / url - 1
        if urr < Rw * Wrr:
            srr = 1 - urr / (Rw * Wrr)
        else:
            srr = Rw * Wrr / urr - 1
        ax = ax_out - vy * r
        ay = d_vy1 + vx * r
        Fzfl = m * g * lr / 2 / L - m * ax * hcg / 2 / L - ay / B * (ms * hrf * lrs / L + muf * huf) - 1 / B * (kphif * phi + bphif * d_phi)
        Fzfr = m * g * lr / 2 / L - m * ax * hcg / 2 / L + ay / B * (ms * hrf * lrs / L + muf * huf) + 1 / B * (kphif * phi + bphif * d_phi)
        Fzrl = m * g * lf / 2 / L + m * ax * hcg / 2 / L - ay / B * (ms * hrr * lfs / L + mur * hur) - 1 / B * (kphir * phi + bphir * d_phi)
        Fzrr = m * g * lf / 2 / L + m * ax * hcg / 2 / L + ay / B * (ms * hrr * lfs / L + mur * hur) + 1 / B * (kphir * phi + bphir * d_phi)
        lambdafl = miu * Fzfl * (1 - sfl) / 2 / np.sqrt(Cf ** 2 * sfl ** 2 + kf ** 2 * np.tan(alfafl) ** 2)
        if lambdafl <= 1:
            flambdafl = (2 - lambdafl) * lambdafl
        else:
            flambdafl = 1
        Fxfl = Cf * sfl * flambdafl / (1 - sfl)
        Fyfl = kf * np.tan(alfafl) * flambdafl / (1 - sfl)
        lambdafr = miu * Fzfr * (1 - sfr) / 2 / np.sqrt(Cf ** 2 * sfr ** 2 + kf ** 2 * np.tan(alfafr) ** 2)
        if lambdafr <= 1:
            flambdafr = (2 - lambdafr) * lambdafr
        else:
            flambdafr = 1
        Fxfr = Cf * sfr * flambdafr / (1 - sfr)
        Fyfr = kf * np.tan(alfafr) * flambdafr / (1 - sfr)
        lambdarl = miu * Fzrl * (1 - srl) / 2 / np.sqrt(Cr ** 2 * srl ** 2 + kr ** 2 * np.tan(alfarl) ** 2)
        if lambdarl <= 1:
            flambdarl = (2 - lambdarl) * lambdarl
        else:
            flambdarl = 1
        Fxrl = Cr * srl * flambdarl / (1 - srl)
        Fyrl = kr * np.tan(alfarl) * flambdarl / (1 - srl)
        lambdarr = miu * Fzrr * (1 - srr) / 2 / np.sqrt(Cr ** 2 * srr ** 2 + kr ** 2 * np.tan(alfarr) ** 2)
        if lambdarr <= 1:
            flambdarr = (2 - lambdarr) * lambdarr
        else:
            flambdarr = 1
        Fxrr = Cr * srr * flambdarr / (1 - srr)
        Fyrr = kr * np.tan(alfarr) * flambdarr / (1 - srr)
        d_vy = ms / m * h * d_d_phi1 - vx * r + 1 / m * (Fxfl * np.sin(deltafl) + Fxfr * np.sin(deltafr) + Fxrl * np.sin(deltarl) + Fxrr * np.sin(deltarr) + Fyfl * np.cos(deltafl) + Fyfr * np.cos(deltafr) + Fyrl * np.cos(deltarl) + Fyrr * np.cos(deltarr))
        d_r = 1 / Iz * Ixz * d_d_phi1 + 1 / Iz * (lf * (Fyfl * np.cos(deltafl) + Fyfr * np.cos(deltafr) + Fxfl * np.sin(deltafl) + Fxfr * np.sin(deltafr)) - lr * (Fyrl * np.cos(deltarl) + Fyrr * np.cos(deltarr) + Fxrl * np.sin(deltarl) + Fxrr * np.sin(deltarr)) + B / 2 * (Fyfl * np.sin(deltafl) + Fyrl * np.sin(deltarl) - Fyfr * np.sin(deltafr) - Fyrr * np.sin(deltarr)) + B / 2 * (-Fxfl * np.cos(deltafl) + Fxfr * np.cos(deltafr) - Fxrl * np.cos(deltarl) + Fxrr * np.cos(deltarr)))
        d_d_phi = 1 / Ix * (ms * g * h * phi - (bphif + bphir) * d_phi - (kphif + kphir) * phi + ms * h * (d_vy1 + vx * r) + Ixz * d_r)
        d_Wfl = 1 / Iw * (Tdfl - Fxfl * Rw - Tbfl)
        d_Wfr = 1 / Iw * (Tdfr - Fxfr * Rw - Tbfr)
        d_Wrl = 1 / Iw * (Tdrl - Fxrl * Rw - Tbrl)
        d_Wrr = 1 / Iw * (Tdrr - Fxrr * Rw - Tbrr)
        dx = np.array([d_vy, d_r, d_d_phi, d_phi, d_Wfl, d_Wfr, d_Wrl, d_Wrr])
        return dx
    

        
#八自由度
class jiaohe_dongli_8():
    def __init__(self,dt):
        self.dt = dt

    def update(self, a_chushi, rot_chushi, t_arr, rot_arr, v_arr, a_arr):
        # print(t_arr,rot_arr,v_arr,a_arr)
        t_now = t_arr[-1]
        rot_now = rot_arr[-1]
        a_now = a_arr[-1]
        v_now = v_arr[-1]
        #初始校核，不扣分
        # print('1: ',rot_chushi)
        if abs(rot_chushi) >= 0.7:
            if rot_chushi >= 0:
                rot_chushi = 0.7
            else:
                rot_chushi = -0.7
        # print('2: ',rot_chushi)
        # if self.rot_temp != None:
        if abs((rot_chushi - rot_now) / self.dt) >= 1.4:
            if (rot_chushi - rot_now) <= 0:
                rot_chushi = -1.4 * self.dt + rot_now
            else:
                rot_chushi = 1.4 * self.dt + rot_now

        if abs(a_chushi - a_now) >= 49*self.dt:
            if a_chushi - a_now > 0:
                a_chushi = a_now + 49*self.dt
            else:
                a_chushi = a_now - 49*self.dt
        
        if v_now + a_chushi * self.dt <= 0:
            a_chushi = -v_now / self.dt
        # print('3: ',rot_chushi)
        #第二次校核，不通过0分
        a_chushi, rot_chushi = self.update_end(a_chushi, rot_chushi, t_arr, rot_arr, v_arr, a_arr)

        return a_chushi, rot_chushi
    

    def update_end(self, a_chushi, rot_chushi, t_arr, rot_arr, v_arr, a_arr):
        # print(t_arr,rot_arr,v_arr,a_arr)
        t_now = t_arr[-1]
        rot_now = rot_arr[-1]
        a_now = a_arr[-1]
        v_now = v_arr[-1]

        t_arr.append(t_now+self.dt)
        rot_arr.append(rot_chushi)
        a_arr.append(a_chushi)
        v_arr.append(v_now+a_chushi*self.dt)
        # print(t_arr,rot_arr,v_arr,a_arr)
        t_arr = np.array(t_arr)
        rot_arr = np.array(rot_arr)
        a_arr = np.array(a_arr)
        v_arr = np.array(v_arr)
        assert len(t_arr) == len(rot_arr)
        assert len(t_arr) == len(a_arr)
        assert len(t_arr) == len(v_arr)

        k1, k2 = self.dynamic_check(t_arr,rot_arr,v_arr,a_arr)
        

        i = 0
        while True:
            if k1 == 1:
                print("校核成功退出")
                break
            rot_chushi = rot_chushi * 0.9
            rot_arr[-1] = rot_chushi
            # print(rot_arr)
            assert len(t_arr) == len(rot_arr)
            k1, k2 = self.dynamic_check(t_arr,rot_arr,v_arr,a_arr)
            i += 1
            if i >= 50:
                print("校核失败退出")
                print(v_now, a_now, rot_chushi, a_chushi)
                break
        # if k1 != 1:
        #     a_chushi = 0
        # print(k1, k2)

        return a_chushi, rot_chushi
    
    def dynamic_check(self, t, deltaf, vx, ax, vx_lim=None, ax_lim=None, jx_lim=None, deltaf_lim=None, omegaf_lim=None, k1=None, k2=None, k3=None):
        t = t[1:]
        deltaf = deltaf[1:]
        vx = vx[1:]
        ax = ax[1:]
        if vx_lim is None:
            vx_lim = 55.55555555555556
        if ax_lim is None:
            ax_lim = 9.8
        if jx_lim is None:
            jx_lim = 49.0
        if deltaf_lim is None:
            deltaf_lim = 40 * np.pi / 180
        if omegaf_lim is None:
            omegaf_lim = 80 * np.pi / 180
        if k1 is None:
            k1 = 1
        if k2 is None:
            k2 = 1
        if k3 is None:
            k3 = 1
        flag = 1
        reasons = ""
        jx = np.diff(ax) / np.diff(t)
        omegaf = np.diff(deltaf) / np.diff(t)
        if np.sum(np.abs(vx) > vx_lim) > 0:
            flag = 0
            reasons += "Exceeding speed limit."
        else:
            sample_time = 0.01
            d_vy1 = 0
            d_d_phi1 = 0
            v0 = vx[0]
            Rw = 0.28415841584158413
            s0 = 0.01
            omega0 = v0 / ((1 - s0) * Rw)
            x = np.array([0, 0, 0, 0, omega0, omega0, omega0, omega0])
            Tend = t[-1]
            n = round(Tend / sample_time)
            dx = np.zeros_like(x)
            u = np.zeros(3)
            ys = np.zeros((n, 3))
            vx = np.array(vx)
            vx = np.where(vx <= 0.0001, 0.0001, vx)
            for i in range(n):
                u[0] = np.interp(sample_time * (i - 1), t, deltaf)
                u[1] = np.interp(sample_time * (i - 1), t, ax)
                u[2] = np.interp(sample_time * (i - 1), t, vx)
                dx = self.Devs(x, u, d_vy1, d_d_phi1)
                d_vy1 = dx[0]
                d_d_phi1 = dx[2]
                x = x + dx * sample_time
                beta = np.arctan(x[0] / u[2])
                y = [beta, x[1], x[3]]
                ys[i, :] = y

            miu = 0.85
            g = 9.8
            beta_lim = np.arctan(0.02 * miu * g / k1)
            miu_y = 0.85
            vx1 = np.interp(sample_time * np.arange(1, n + 1), t, vx)
            # print(vx1)
            r_lim = g * miu_y / (vx1 * k2)
            phi_lim = 0.03333333333333333 * np.pi / k3
            if np.sum(np.abs(ys[:, 0]) > beta_lim) > 0:
                flag = 0
                reasons += "Exceeding sideslip-angle limit."
            if np.sum(np.abs(ys[:, 1]) > r_lim) > 0:
                flag = 0
                reasons += "Exceeding yaw-rate limit."
            if np.sum(np.abs(ys[:, 2]) > phi_lim) > 0:
                flag = 0
                reasons += "Exceeding roll-angle limit."
            if flag == 1:
                reasons += "Dynamics verification passed!"
            else:
                reasons = reasons
        return (flag, reasons)


    def Devs(self, x, u, d_vy1, d_d_phi1):
        m = 1134
        mu = 125.9
        ms = m - mu
        Iz = 1343.1
        Ix = 440.6
        Ixz = 0
        lf = 1.04
        L = 2.6
        lr = L - lf
        B = 1.48
        hcg = 0.54
        h = 0.23
        kf = 86320.0
        kr = 58420.0
        Cf = 179040.0
        Cr = 121480.0
        kphif = 69120 / np.pi
        kphir = 45180 / np.pi
        bphif = 2000
        bphir = 2000
        f = 0.008
        CD = 0.3
        ru = 1.206
        A = 1.6
        Iw = 0.8
        Rw = 0.287
        lfs = lf
        lrs = lr
        hrf = 0.165
        hrr = 0.165
        muf = 71.4
        mur = 54.5
        huf = Rw
        hur = Rw
        g = 9.8
        vy = x[0]
        r = x[1]
        d_phi = x[2]
        phi = x[3]
        Wfl = x[4]
        Wfr = x[5]
        Wrl = x[6]
        Wrr = x[7]
        deltaf_out = u[0]
        ax_out = u[1]
        vx_out = u[2]
        vx = vx_out
        deltarl = 0
        deltarr = 0
        Tbfl = 0
        Tbfr = 0
        Tbrl = 0
        Tbrr = 0
        miu = 0.85
        deltafl = np.arctan(np.tan(deltaf_out) / (1 - B / L / 2 * np.tan(deltaf_out)))
        deltafr = np.arctan(np.tan(deltaf_out) / (1 + B / L / 2 * np.tan(deltaf_out)))
        Fx = m * ax_out + f * m * g + CD * A * ru * vx_out ** 2 / 2
        Tall = Fx * Rw
        Delta_T = Tall / 2 * (1 - np.cos(deltaf_out))
        Tdfl = (Tall + Delta_T) / 4
        Tdfr = Tdfl
        Tdrl = Tdfl
        Tdrr = Tdfl
        ufl = (vx - 0.5 * B * r) * np.cos(deltafl) + (vy + lf * r) * np.sin(deltafl)
        ufr = (vx + 0.5 * B * r) * np.cos(deltafr) + (vy + lf * r) * np.sin(deltafr)
        url = (vx - 0.5 * B * r) * np.cos(deltarl) + (vy - lr * r) * np.sin(deltarl)
        urr = (vx + 0.5 * B * r) * np.cos(deltarr) + (vy - lr * r) * np.sin(deltarr)
        alfafl = deltafl - np.arctan((vy + lf * r) / (vx - 0.5 * B * r))
        alfafr = deltafr - np.arctan((vy + lf * r) / (vx + 0.5 * B * r))
        alfarl = deltarl - np.arctan((vy - lr * r) / (vx - 0.5 * B * r))
        alfarr = deltarr - np.arctan((vy - lr * r) / (vx + 0.5 * B * r))
        if ufl < Rw * Wfl:
            sfl = 1 - ufl / (Rw * Wfl)
        else:
            sfl = Rw * Wfl / ufl - 1
        if ufr < Rw * Wfr:
            sfr = 1 - ufr / (Rw * Wfr)
        else:
            sfr = Rw * Wfr / ufr - 1
        if url < Rw * Wrl:
            srl = 1 - url / (Rw * Wrl)
        else:
            srl = Rw * Wrl / url - 1
        if urr < Rw * Wrr:
            srr = 1 - urr / (Rw * Wrr)
        else:
            srr = Rw * Wrr / urr - 1
        ax = ax_out - vy * r
        ay = d_vy1 + vx * r
        Fzfl = m * g * lr / 2 / L - m * ax * hcg / 2 / L - ay / B * (ms * hrf * lrs / L + muf * huf) - 1 / B * (kphif * phi + bphif * d_phi)
        Fzfr = m * g * lr / 2 / L - m * ax * hcg / 2 / L + ay / B * (ms * hrf * lrs / L + muf * huf) + 1 / B * (kphif * phi + bphif * d_phi)
        Fzrl = m * g * lf / 2 / L + m * ax * hcg / 2 / L - ay / B * (ms * hrr * lfs / L + mur * hur) - 1 / B * (kphir * phi + bphir * d_phi)
        Fzrr = m * g * lf / 2 / L + m * ax * hcg / 2 / L + ay / B * (ms * hrr * lfs / L + mur * hur) + 1 / B * (kphir * phi + bphir * d_phi)
        lambdafl = miu * Fzfl * (1 - sfl) / 2 / np.sqrt(Cf ** 2 * sfl ** 2 + kf ** 2 * np.tan(alfafl) ** 2)
        if lambdafl <= 1:
            flambdafl = (2 - lambdafl) * lambdafl
        else:
            flambdafl = 1
        Fxfl = Cf * sfl * flambdafl / (1 - sfl)
        Fyfl = kf * np.tan(alfafl) * flambdafl / (1 - sfl)
        lambdafr = miu * Fzfr * (1 - sfr) / 2 / np.sqrt(Cf ** 2 * sfr ** 2 + kf ** 2 * np.tan(alfafr) ** 2)
        if lambdafr <= 1:
            flambdafr = (2 - lambdafr) * lambdafr
        else:
            flambdafr = 1
        Fxfr = Cf * sfr * flambdafr / (1 - sfr)
        Fyfr = kf * np.tan(alfafr) * flambdafr / (1 - sfr)
        lambdarl = miu * Fzrl * (1 - srl) / 2 / np.sqrt(Cr ** 2 * srl ** 2 + kr ** 2 * np.tan(alfarl) ** 2)
        if lambdarl <= 1:
            flambdarl = (2 - lambdarl) * lambdarl
        else:
            flambdarl = 1
        Fxrl = Cr * srl * flambdarl / (1 - srl)
        Fyrl = kr * np.tan(alfarl) * flambdarl / (1 - srl)
        lambdarr = miu * Fzrr * (1 - srr) / 2 / np.sqrt(Cr ** 2 * srr ** 2 + kr ** 2 * np.tan(alfarr) ** 2)
        if lambdarr <= 1:
            flambdarr = (2 - lambdarr) * lambdarr
        else:
            flambdarr = 1
        Fxrr = Cr * srr * flambdarr / (1 - srr)
        Fyrr = kr * np.tan(alfarr) * flambdarr / (1 - srr)
        d_vy = ms / m * h * d_d_phi1 - vx * r + 1 / m * (Fxfl * np.sin(deltafl) + Fxfr * np.sin(deltafr) + Fxrl * np.sin(deltarl) + Fxrr * np.sin(deltarr) + Fyfl * np.cos(deltafl) + Fyfr * np.cos(deltafr) + Fyrl * np.cos(deltarl) + Fyrr * np.cos(deltarr))
        d_r = 1 / Iz * Ixz * d_d_phi1 + 1 / Iz * (lf * (Fyfl * np.cos(deltafl) + Fyfr * np.cos(deltafr) + Fxfl * np.sin(deltafl) + Fxfr * np.sin(deltafr)) - lr * (Fyrl * np.cos(deltarl) + Fyrr * np.cos(deltarr) + Fxrl * np.sin(deltarl) + Fxrr * np.sin(deltarr)) + B / 2 * (Fyfl * np.sin(deltafl) + Fyrl * np.sin(deltarl) - Fyfr * np.sin(deltafr) - Fyrr * np.sin(deltarr)) + B / 2 * (-Fxfl * np.cos(deltafl) + Fxfr * np.cos(deltafr) - Fxrl * np.cos(deltarl) + Fxrr * np.cos(deltarr)))
        d_d_phi = 1 / Ix * (ms * g * h * phi - (bphif + bphir) * d_phi - (kphif + kphir) * phi + ms * h * (d_vy1 + vx * r) + Ixz * d_r)
        d_Wfl = 1 / Iw * (Tdfl - Fxfl * Rw - Tbfl)
        d_Wfr = 1 / Iw * (Tdfr - Fxfr * Rw - Tbfr)
        d_Wrl = 1 / Iw * (Tdrl - Fxrl * Rw - Tbrl)
        d_Wrr = 1 / Iw * (Tdrr - Fxrr * Rw - Tbrr)
        dx = np.array([d_vy, d_r, d_d_phi, d_phi, d_Wfl, d_Wfr, d_Wrl, d_Wrr])
        return dx


# x = jiaohe_dongli_8_new(dt=0.04)

# folder_path = 'E:/第二届Onsite比赛/code/code_train_demo_test5.0/outputs/'
# if os.path.exists(str(folder_path)):
#     file_names = []  # 用于保存文件名
#     file_paths = []  # 用于保存文件路径
#     # 遍历文件夹中的每一个文件
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             file_name = file  # 文件名（包含后缀）
#             file_path = os.path.join(root, file)  # 文件绝对路径
#             file_names.append(file_name)
#             file_paths.append(file_path)
    
#     for j in range(len(file_paths)):
#         data = np.genfromtxt(str(file_paths[j]), delimiter=',', skip_header=1)
#         t = np.array(data[:,0])
#         deltaf = np.array(data[:,8])
#         vx = np.array(data[:,5])
#         ax = np.array(data[:,6])
#         for i in range(len(t)-1):
#             a,r = x.update(v_now=vx[i],a_now=ax[i],yaw_now=0,rot_now=deltaf[i],a_chushi=ax[i+1],rot_chushi=deltaf[i+1],ego_info=0)
#             x.update_temp(v_now=vx[i],a_now=ax[i],yaw_now=0,rot_now=deltaf[i],a_chushi=ax[i+1],rot_chushi=deltaf[i+1],ego_info=0)
