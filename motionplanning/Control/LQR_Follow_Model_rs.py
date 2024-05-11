"""
LQR Follow Reeds Shepp controller
author: huaiyang zhu
"""


import os
import sys
import math
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import motionplanning.Control.draw_lqr as draw
from motionplanning.Control.config_control import *
import motionplanning.CurvesGenerator.reeds_shepp as rs
import motionplanning.CurvesGenerator.dubins_path as dp
from motionplanning.Control.Obj_Generotor import *
from motionplanning.Control import gif
import time
import random
# class C:
#     # PID config
#     Kp = 1.0
#
#     # System config
#     dt = 0.1
#     dist_stop = 0.5
#     Q = np.eye(4)
#     R = np.eye(1)
#
#     # vehicle config
#     RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
#     RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
#     W = 2.4  # [m] width of vehicle
#     WD = 0.7 * W  # [m] distance between left-right wheels
#     WB = 2.5  # [m] Wheel base
#     TR = 0.44  # [m] Tyre radius
#     TW = 0.7  # [m] Tyre width
#     MAX_STEER = 0.30


# Controller Config
ts = 0.1  # [s]
l_f = 1.165  # [m]
l_r = 1.165  # [m]
max_iteration = 150
eps = 0.01

matrix_q = [0.5, 0.0, 1.0, 0.0]
matrix_r = [1.0]

state_size = 4

max_acceleration = 5.0  # [m / s^2] 
max_steer_angle = np.deg2rad(40)  # [rad] 最大转向角度
max_speed = 35 / 3.6  # [m / s]


class Gear(Enum):
    GEAR_DRIVE = 1
    GEAR_REVERSE = 2

class ObjectState:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, gear=Gear.GEAR_DRIVE, maxx=50, maxy=50):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.gear = gear
        self.maxx = maxx
        self.maxy = maxy

    def UpdateObjectState(self, yaw, v=0.1, gear=Gear.GEAR_DRIVE):

        if gear == Gear.GEAR_DRIVE:
            direction = 1.0
        else:
            direction = -1.0

        vx = direction * v * np.cos(yaw)
        vy = direction * v * np.sin(yaw)

        self.x += vx * ts
        self.y += vy * ts

        self.yaw = yaw
        self.v = v
        self.gear = gear

class VehicleState:  # 车辆状态
    def __init__(self, x=0.0, y=0.0, yaw=0.0,
                 v=0.0, gear=Gear.GEAR_DRIVE):
        self.x = x  # 位置
        self.y = y
        self.yaw = yaw  # 整车转向
        self.v = v  # 速度
        self.e_cg = 0.0  # 偏向距离
        self.theta_e = 0.0  # 偏航角误差

        self.gear = gear  # 挡位  前进和后退
        self.steer = 0.0  # 前车轮转向

    def UpdateVehicleState(self, delta, a, e_cg, theta_e,
                           gear=Gear.GEAR_DRIVE):  # 转角，加速度，横向偏差，偏航角
        """
        update states of vehicle
        :param theta_e: yaw error to ref trajectory
        :param e_cg: lateral error to ref trajectory
        :param delta: steering angle [rad]
        :param a: acceleration [m / s^2]
        :param gear: gear mode [GEAR_DRIVE / GEAR/REVERSE]
        """

        # wheelbase_ = l_r + l_f
        delta, a = self.RegulateInput(delta, a)

        self.gear = gear
        self.steer = delta
        self.x += self.v * math.cos(self.yaw) * ts
        self.y += self.v * math.sin(self.yaw) * ts
        self.yaw += self.v / chassis_d * math.tan(delta) * ts  # v/tan(delta)->v_w  v_w/L -> w
        self.e_cg = e_cg
        self.theta_e = theta_e

        if gear == Gear.GEAR_DRIVE:
            self.v += a * ts
        else:
            self.v += -1.0 * a * ts

        self.v = self.RegulateOutput(self.v)

    @staticmethod
    def RegulateInput(delta, a):  # 限制车辆的转向角和加速度输入
        """
        regulate delta to : - max_steer_angle ~ max_steer_angle
        regulate a to : - max_acceleration ~ max_acceleration
        :param delta: steering angle [rad]
        :param a: acceleration [m / s^2]
        :return: regulated delta and acceleration
        """

        if delta < -1.0 * max_steer_angle:
            delta = -1.0 * max_steer_angle

        if delta > 1.0 * max_steer_angle:
            delta = 1.0 * max_steer_angle

        if a < -1.0 * max_acceleration:
            a = -1.0 * max_acceleration

        if a > 1.0 * max_acceleration:
            a = 1.0 * max_acceleration

        return delta, a

    @staticmethod
    def RegulateOutput(v):  # 限制测量的输入
        """
        regulate v to : -max_speed ~ max_speed
        :param v: calculated speed [m / s]
        :return: regulated speed
        """

        max_speed_ = max_speed

        if v < -1.0 * max_speed_:
            v = -1.0 * max_speed_

        if v > 1.0 * max_speed_:
            v = 1.0 * max_speed_

        return v

class TrajectoryAnalyzer:  # 车辆相对于参考轨迹的位置和方向
    def __init__(self, x, y, yaw, k):  # 跟踪轨迹，x坐标，y坐标，航向角，曲率
        self.x_ = x
        self.y_ = y
        self.yaw_ = yaw
        self.k_ = k

        self.ind_old = 0
        self.ind_end = len(x)

    def ToTrajectoryFrame(self, vehicle_state):
        """
        errors to trajectory frame
        theta_e = yaw_vehicle - yaw_ref_path
        e_cg = lateral distance of center of gravity (cg) in frenet frame
        :param vehicle_state: vehicle state (class VehicleState)
        :return: theta_e, e_cg, yaw_ref, k_ref
        """

        x_cg = vehicle_state.x
        y_cg = vehicle_state.y
        yaw = vehicle_state.yaw

        # calc nearest point in ref path  计算车辆到跟踪轨迹上所有点的横纵坐标值
        dx = [x_cg - ix for ix in self.x_[self.ind_old: self.ind_end]]  # 车辆 - 轨迹点
        dy = [y_cg - iy for iy in self.y_[self.ind_old: self.ind_end]]

        ind_add = int(np.argmin(np.hypot(dx, dy)))  # 在轨迹种找到离车辆位置最接近的索引
        dist = math.hypot(dx[ind_add], dy[ind_add])  # 车辆与下一个跟踪轨迹点的欧几里得距离

        # calc lateral relative position of vehicle to ref path 
        vec_axle_rot_90 = np.array([[math.cos(yaw + math.pi / 2.0)],
                                    [math.sin(yaw + math.pi / 2.0)]])

        vec_path_2_cg = np.array([[dx[ind_add]],
                                  [dy[ind_add]]])

        if np.dot(vec_axle_rot_90.T, vec_path_2_cg) > 0.0:
            e_cg = 1.0 * dist  # vehicle on the right of ref path  右侧 + 逆时针 夹角一定小于90
        else:
            e_cg = -1.0 * dist  # vehicle on the left of ref path

        # calc yaw error: theta_e = yaw_vehicle - yaw_ref
        self.ind_old += ind_add
        yaw_ref = self.yaw_[self.ind_old]  # 轨迹偏差
        theta_e = pi_2_pi(yaw - yaw_ref)  # 车辆偏航与轨迹偏航的偏差

        # calc ref curvature
        k_ref = self.k_[self.ind_old]  # 轨迹曲率

        return theta_e, e_cg, yaw_ref, k_ref


class LatController:  # 横向控制器，最优转向控制指令
    """
    Lateral Controller using LQR
    """

    def ComputeControlCommand(self, vehicle_state, ref_trajectory):
        """
        calc lateral control command.
        :param vehicle_state: vehicle state
        :param ref_trajectory: reference trajectory (analyzer)
        :return: steering angle (optimal u), theta_e, e_cg
        """

        ts_ = ts
        e_cg_old = vehicle_state.e_cg
        theta_e_old = vehicle_state.theta_e

        theta_e, e_cg, yaw_ref, k_ref = \
            ref_trajectory.ToTrajectoryFrame(vehicle_state)  # 车辆与轨迹偏航角偏差，距离偏差，轨迹偏航角，轨迹曲率

        matrix_ad_, matrix_bd_ = self.UpdateMatrix(vehicle_state)  # 系统矩阵A，输入矩阵B

        matrix_state_ = np.zeros((state_size, 1))  # [d, dt, e, de]
        matrix_r_ = np.diag(matrix_r)
        matrix_q_ = np.diag(matrix_q)

        matrix_k_ = self.SolveLQRProblem(matrix_ad_, matrix_bd_, matrix_q_,
                                         matrix_r_, eps, max_iteration)  # 反馈增益矩阵求解

        matrix_state_[0][0] = e_cg
        matrix_state_[1][0] = (e_cg - e_cg_old) / ts_
        matrix_state_[2][0] = theta_e
        matrix_state_[3][0] = (theta_e - theta_e_old) / ts_

        steer_angle_feedback = -(matrix_k_ @ matrix_state_)[0][0]  # u = -Kx  反馈控制

        steer_angle_feedforward = self.ComputeFeedForward(k_ref)  # 前馈控制

        steer_angle = steer_angle_feedback + steer_angle_feedforward

        return steer_angle, theta_e, e_cg

    @staticmethod
    def ComputeFeedForward(ref_curvature):
        """
        calc feedforward control term to decrease the steady error.
        :param ref_curvature: curvature of the target point in ref trajectory
        :return: feedforward term
        """

        # wheelbase_ = l_f + l_r

        steer_angle_feedforward = chassis_d * ref_curvature  # angle = arctan(L*k)  转角，轴距，曲率

        return steer_angle_feedforward

    @staticmethod
    def SolveLQRProblem(A, B, Q, R, tolerance, max_num_iteration):  # 求解LQR最优反馈矩阵
        """
        iteratively calculating feedback matrix K
        :param A: matrix_a_
        :param B: matrix_b_
        :param Q: matrix_q_
        :param R: matrix_r_
        :param tolerance: lqr_eps
        :param max_num_iteration: max_iteration
        :return: feedback matrix K
        """

        assert np.size(A, 0) == np.size(A, 1) and \
               np.size(B, 0) == np.size(A, 0) and \
               np.size(Q, 0) == np.size(Q, 1) and \
               np.size(Q, 0) == np.size(A, 1) and \
               np.size(R, 0) == np.size(R, 1) and \
               np.size(R, 0) == np.size(B, 1), \
            "LQR solver: one or more matrices have incompatible dimensions."

        M = np.zeros((np.size(Q, 0), np.size(R, 1)))

        AT = A.T
        BT = B.T
        MT = M.T

        P = Q
        num_iteration = 0
        diff = math.inf

        while num_iteration < max_num_iteration and diff > tolerance:
            num_iteration += 1
            P_next = AT @ P @ A - (AT @ P @ B + M) @ \
                     np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT) + Q

            # check the difference between P and P_next
            diff = (abs(P_next - P)).max()
            P = P_next

        if num_iteration >= max_num_iteration:
            print("LQR solver cannot converge to a solution",
                  "last consecutive result diff is: ", diff)

        K = np.linalg.inv(BT @ P @ B + R) @ (BT @ P @ A + MT)

        return K

    @staticmethod
    def UpdateMatrix(vehicle_state):  # 更新线性化系统矩阵
        """
        calc A and b matrices of linearized, discrete system.
        :return: A, b
        """

        ts_ = ts
        # wheelbase_ = l_f + l_r

        v = vehicle_state.v
        
        matrix_ad_ = np.zeros((state_size, state_size))  # time discrete A matrix

        matrix_ad_[0][0] = 1.0
        matrix_ad_[0][1] = ts_
        matrix_ad_[1][2] = v
        matrix_ad_[2][2] = 1.0
        matrix_ad_[2][3] = ts_

        # b = [0.0, 0.0, 0.0, v / L].T
        matrix_bd_ = np.zeros((state_size, 1))  # time discrete b matrix
        matrix_bd_[3][0] = v / chassis_d

        return matrix_ad_, matrix_bd_


class LonController:  # 纵向控制器，速度PID
    """
    Longitudinal Controller using PID.
    """

    @staticmethod
    def ComputeControlCommand(target_speed, vehicle_state, dist):  # 目标速度，车辆状态，到目标的距离
        """
        calc acceleration command using PID.
        :param target_speed: target speed [m / s]
        :param vehicle_state: vehicle state
        :param dist: distance to goal [m]
        :return: control command (acceleration) [m / s^2]
        """

        if vehicle_state.gear == Gear.GEAR_DRIVE:
            direct = 1.0
        else:
            direct = -1.0

        a = 0.6 * (target_speed - direct * vehicle_state.v)

        if dist < 10.0:
            if vehicle_state.v > 2.0:
                a = -3.0
            elif vehicle_state.v < -2:
                a = -1.0

        return a

    @staticmethod
    def ControlSpeedByDistance(target_speed, vehicle_state, dist):
        """
        Control vehicle speed based on the distance to the target.
        :param target_speed: target speed [m / s]
        :param vehicle_state: vehicle state
        :param dist: distance to goal [m]
        :return: control command (acceleration) [m / s^2]
        """
        # 设置一些基本参数
        safe_distance = 5.0  # 安全距离
        max_deceleration = -5.0  # 最大减速度
        min_deceleration = -1.0  # 最小减速度
        deceleration_zone = 20.0  # 开始减速的区域

        # 如果距离目标较远，尝试加速到目标速度
        if dist > deceleration_zone:
            # 计算加速指令，可以使用 PID 控制器，或者简单的比例控制
            acceleration = 0.5 * (target_speed - vehicle_state.v)
        # 如果在减速区域内
        elif safe_distance < dist <= deceleration_zone:
            # 根据距离线性减小加速度
            accel_ratio = (dist - safe_distance) / (deceleration_zone - safe_distance)
            acceleration = max_deceleration * (1 - accel_ratio)
        # 如果小于安全距离，进行强制减速
        else:
            acceleration = min_deceleration

        # 根据当前档位调整加速度方向
        if vehicle_state.gear != Gear.GEAR_DRIVE:
            acceleration *= -1

        return acceleration
    

def pi_2_pi(angle):
    """
    regulate theta to -pi ~ pi.
    :param angle: input angle
    :return: regulated angle
    """

    M_PI = math.pi

    if angle > M_PI:
        return angle - 2.0 * M_PI

    if angle < -M_PI:
        return angle + 2.0 * M_PI

    return angle



def generate_path(s):
    """
    design path using reeds-shepp path generator.
    divide paths into sections, in each section the direction is the same.
    :param s: objective positions and directions.
    :return: paths
    """
    # wheelbase_ = l_f + l_r

    max_c = math.tan(0.5 * max_steer_angle) / chassis_d  # 最大曲率  tan(angle) = L*K -> L/R
    path_x, path_y, yaw, direct, rc = [], [], [], [], []
    x_rec, y_rec, yaw_rec, direct_rec, rc_rec = [], [], [], [], []
    direct_flag = 1.0

    for i in range(len(s) - 1):  # 多点
        # s_x, s_y, s_yaw = s[i][0], s[i][1], np.deg2rad(s[i][2])
        # g_x, g_y, g_yaw = s[i + 1][0], s[i + 1][1], np.deg2rad(s[i + 1][2])

        s_x, s_y, s_yaw = s[i][0], s[i][1], s[i][2]
        g_x, g_y, g_yaw = s[i + 1][0], s[i + 1][1], s[i + 1][2]
        path_i = rs.calc_optimal_path(s_x, s_y, s_yaw,
                                      g_x, g_y, g_yaw, max_c)  # 两点之间选择最优的路径

        irc, rds = rs.calc_curvature(path_i.x, path_i.y, path_i.yaw, path_i.directions)  # irc曲率  rds弧长

        ix = path_i.x
        iy = path_i.y
        iyaw = path_i.yaw
        idirect = path_i.directions

        for j in range(len(ix)):
            if idirect[j] == direct_flag:
                x_rec.append(ix[j])
                y_rec.append(iy[j])
                yaw_rec.append(iyaw[j])
                direct_rec.append(idirect[j])
                rc_rec.append(irc[j])
            else:
                if len(x_rec) == 0 or direct_rec[0] != direct_flag:  # C->S->C转接处
                    direct_flag = idirect[j]
                    continue

                path_x.append(x_rec)
                path_y.append(y_rec)
                yaw.append(yaw_rec)
                direct.append(direct_rec)
                rc.append(rc_rec)
                x_rec, y_rec, yaw_rec, direct_rec, rc_rec = \
                    [x_rec[-1]], [y_rec[-1]], [yaw_rec[-1]], [-direct_rec[-1]], [rc_rec[-1]]

    path_x.append(x_rec)
    path_y.append(y_rec)
    yaw.append(yaw_rec)
    direct.append(direct_rec)
    rc.append(rc_rec)

    x_all, y_all, yaw_all, dic_all, rc_all = [], [], [], [], []
    for ix, iy, iyaw, idic, irc in zip(path_x, path_y, yaw, direct, rc):
        x_all += ix
        y_all += iy
        yaw_all += iyaw
        dic_all += idic
        rc_all += irc

    return path_x, path_y, yaw, direct, rc, x_all, y_all, yaw_all, dic_all, rc_all

def Waveform_test():
    obj_state = ObjectState(0, 0, 0, 0.1, Gear.GEAR_DRIVE)
    valgenerator = SinGenerator()
    x_run, y_run = [], []

    @gif.frame
    def plott():
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        ax.set_xlim(-20, 80)
        ax.set_ylim(-50, 50)
        plt.plot(x_run, y_run, linewidth=2.0, color='darkviolet')
        draw.draw_person(obj_state.x, obj_state.y, obj_state.yaw)
        time.sleep(ts)

    time.sleep(ts)
    fs = []
    while True:
        now_yaw = valgenerator.update()
        obj_state.UpdateObjectState(now_yaw, 10)
        cnt = 0

        print(f"x: {obj_state.x}, y: {obj_state.y}, yaw: {now_yaw}")
        x_run.append(obj_state.x)
        y_run.append(obj_state.y)
        f = plott()
        fs.append(f)
        if len(fs) > 150:
            gif.save(fs, "output/Generator/SinGenerator.gif", duration=50)
            break

def main2():
    obj_state = ObjectState(0, 0, 0, 0.1, Gear.GEAR_DRIVE)
    vehicle_state = VehicleState(0, 0, 0, 0.1, Gear.GEAR_DRIVE)  # 初始化车辆状态

    lat_controller = LatController()  # 横向角度控制器LQR
    lon_controller = LonController()  # 纵向距离控制器PID
    valgenerator = EightShapeGenerator()
    fs = []
    x_objs, y_objs, x_vehicles, y_vehicles = [], [], [], []

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-20, 180)
    follow_dist_threshold = 5
    while True:
        now_yaw = valgenerator.update()
        obj_state.UpdateObjectState(now_yaw, 4)
        dist = math.hypot(vehicle_state.x - obj_state.x, vehicle_state.y - obj_state.y)

        @gif.frame
        def plott(x_all, y_all):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            plt.title("The distance: {:.2f} m".format(dist))
            plt.plot(x_all, y_all, color='gray', linewidth=2.0)  # ref
            # plt.plot(x_rec, y_rec, linewidth=2.0, color='darkviolet')  # 机器人形势路线
            draw.draw_person(obj_state.x, obj_state.y, obj_state.yaw)
            draw.draw_omni_wheel_vehicle(vehicle_state.x, vehicle_state.y, vehicle_state.yaw)
            time.sleep(0.01)

        path_cnt_choose, path_cnt = 3, 0  # 两点间选择的跟踪点数设置，当前执行的跟踪点数
        ref_trajectory = None
        while dist > follow_dist_threshold:  # 安全跟踪距离阈值
            print("idx: {}, cnt: {}, follow dist: {:.2f}".format(len(fs), path_cnt, dist))
            print("obj_x: {:.2f}, obj_y: {:.2f}, obj_yaw: {:.2f}".format(obj_state.x, obj_state.y, obj_state.yaw))
            print("vehicle_x: {:.2f}, vehicle_y: {:.2f}, vehicle_yaw: {:.2f}".format(vehicle_state.x, vehicle_state.y, vehicle_state.yaw))

            if len(fs) >= 77:
                a = 1

            if path_cnt == 0:
                x_ref, y_ref, yaw_ref, direct, curv, x_all, y_all, yaw_all, dic_all, rc_all = generate_path([
                    [vehicle_state.x, vehicle_state.y, vehicle_state.yaw],  # 起始点
                    [obj_state.x, obj_state.y, obj_state.yaw]  # 末端点
                ])  # 生成目标与机器人之间的路径
                ref_trajectory = TrajectoryAnalyzer(x_all[:path_cnt_choose], y_all[:path_cnt_choose], yaw_all[:path_cnt_choose], rc_all[:path_cnt_choose])  # 初始化当前两点跟踪轨迹
    
            if dic_all[path_cnt] > 0:
                target_speed = 25.0 / 3.6
                direct_s = Gear.GEAR_DRIVE
            else:
                target_speed = 15.0 / 3.6
                direct_s = Gear.GEAR_REVERSE

            delta_opt, theta_e, e_cg = \
                lat_controller.ComputeControlCommand(vehicle_state, ref_trajectory)  # 车辆纵向位置调节，[车辆转向控制角度，偏航角误差，横向偏移]
            a_opt = lon_controller.ComputeControlCommand(target_speed, vehicle_state, dist)  # 横向角度调节
            print("delta_opt: {:.2f}, a_opt: {:.2f}, e_cg: {:.2f}, theta_e: {:.2f}, direct_s: {}".format(delta_opt, a_opt, e_cg, theta_e, direct_s))
            print("\n")
            vehicle_state.UpdateVehicleState(delta_opt, a_opt, e_cg, theta_e, direct_s)  # 更新车辆位置信息

            path_cnt = (path_cnt + 1) % path_cnt_choose
            now_yaw = valgenerator.update()
            obj_state.UpdateObjectState(now_yaw, 4)
            dist = math.hypot(vehicle_state.x - obj_state.x, vehicle_state.y - obj_state.y)

            f = plott(x_all, y_all)
            fs.append(f)
            if len(fs) > 200:
                break

        print("idx: {}, no follow dist: {:.2f}".format(len(fs), dist))

        f = plott([], [])
        fs.append(f)
        if len(fs) > 200:
            gif.save(fs, "output/Robot_Follow_EightShape_RS.gif", duration=50)
            break

        # draw.draw_person(obj_state.x, obj_state.y, obj_state.yaw)
        # plt.pause(0.01)

def main():
    # generate path
    # states = [(0, 0, 0), (20, 15, 0), (35, 20, 90), (40, 0, 180),
    #           (20, 0, 120), (5, -10, 180), (15, 5, 30)]
    #
    # states = [(-3, 3, 120), (10, -7, 30), (10, 13, 30), (20, 5, -25),
    #           (35, 10, 180), (30, -10, 160), (5, -12, 90)]

    # states = [(0, 0, 0), (20, 20, 45), (40, 0, 135), (0, 0, 180)]
    states = [(0, 0, np.deg2rad(0)), (40, 15, np.deg2rad(45))]


    # start_time = time.time()
    # x坐标，y坐标，路径角度，方向，曲率
    x_ref, y_ref, yaw_ref, direct, curv, x_all, y_all, _, _, _ = generate_path(states)  # 路径生成

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"The code block took {elapsed_time} seconds to execute.")

    # wheelbase_ = l_f + l_r

    maxTime = 100.0
    yaw_old = 0.0
    x0, y0, yaw0, direct0 = \
        x_ref[0][0], y_ref[0][0], yaw_ref[0][0], direct[0][0]

    x_rec, y_rec, yaw_rec, direct_rec = [], [], [], []

    lat_controller = LatController()  # 横向角度控制器LQR
    lon_controller = LonController()  # 纵向距离控制器PID
    fs = []
    for x, y, yaw, gear, k in zip(x_ref, y_ref, yaw_ref, direct, curv):

        t = 0.0

        if gear[0] == 1.0:
            direct_s = Gear.GEAR_DRIVE
        else:
            direct_s = Gear.GEAR_REVERSE

        ref_trajectory = TrajectoryAnalyzer(x, y, yaw, k)  # 初始化当前两点跟踪轨迹

        vehicle_state = VehicleState(x=x0, y=y0, yaw=yaw0, v=0.1, gear=direct)  # 初始化车辆状态

        while t < maxTime:

            dist = math.hypot(vehicle_state.x - x[-1], vehicle_state.y - y[-1])  # 目标距离

            if gear[0] > 0:
                target_speed = 25.0 / 3.6
            else:
                target_speed = 15.0 / 3.6

            delta_opt, theta_e, e_cg = \
                lat_controller.ComputeControlCommand(vehicle_state, ref_trajectory)  # 车辆纵向位置调节，[车辆转向控制角度，偏航角误差，横向偏移]

            a_opt = lon_controller.ComputeControlCommand(target_speed, vehicle_state, dist)  # 车辆加速度调节，[加速度]
            print("delta_opt: {:.2f}, a_opt: {:.2f}, e_cg: {:.2f}, theta_e: {:.2f}, direct_s: {}".format(delta_opt, a_opt, e_cg, theta_e, direct_s))
            # a_opt = lon_controller.ControlSpeedByDistance(target_speed, vehicle_state, dist + 10)
            vehicle_state.UpdateVehicleState(delta_opt, a_opt, e_cg, theta_e, direct)  # 更新车辆位置信息

            t += ts

            if dist <= 0.5:
                break

            x_rec.append(vehicle_state.x)
            y_rec.append(vehicle_state.y)
            yaw_rec.append(vehicle_state.yaw)

            dy = (vehicle_state.yaw - yaw_old) / (vehicle_state.v * ts)
            # steer = rs.pi_2_pi(-math.atan(wheelbase_ * dy))

            yaw_old = vehicle_state.yaw
            x0 = x_rec[-1]
            y0 = y_rec[-1]
            yaw0 = yaw_rec[-1]

            @gif.frame
            def plott(x, y, yaw):
                # plt.cla()
                fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
                ax.set_xlim(-50, 50)
                ax.set_ylim(-50, 50)
                plt.plot(x_all, y_all, color='gray', linewidth=2.0)  # ref
                plt.plot(x_rec, y_rec, linewidth=2.0, color='darkviolet')  # 机器人形势路线
                # plt.plot(x[ind], y[ind], '.r')
                draw.draw_omni_wheel_vehicle(x, y, yaw)
                # plt.axis("equal")
                # plt.title("LQR: v=" + str(vehicle_state.v * 3.6)[:4] + "km/h")
                # plt.gcf().canvas.mpl_connect('key_release_event',
                #                             lambda event:
                #                             [exit(0) if event.key == 'escape' else None])
                # plt.pause(0.001)
                time.sleep(0.001)

            f = plott(x0, y0, yaw0)
            fs.append(f)

    gif.save(fs, "output/LQR.gif", duration=50)
            
if __name__ == '__main__':
    main2()
