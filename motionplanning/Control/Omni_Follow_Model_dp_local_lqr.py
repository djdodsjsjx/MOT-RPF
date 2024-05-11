"""
Omni Follow Dubins Local Coor Controller
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
from motionplanning.Control.eval import draw_control_err, draw_follow_show
import time
import random
from pathlib import Path

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

class Gear(Enum):
    GEAR_DRIVE = 1
    GEAR_REVERSE = 2

class ObjectState:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, gear=Gear.GEAR_DRIVE, maxx=50, maxy=50):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.lastx = x
        self.lasty = y
        # self.yaw = yaw
        # self.v = v
        # self.gear = gear
        # self.maxx = maxx
        # self.maxy = maxy

    def UpdateObjectState(self, x, y, yaw, gear=Gear.GEAR_DRIVE):
        
        self.lastx = self.x
        self.lasty = self.y
        self.x = x
        self.y = y
        self.yaw = yaw
        # if gear == Gear.GEAR_DRIVE:
        #     direction = 1.0
        # else:
        #     direction = -1.0

        # vx = direction * v * np.cos(yaw)
        # vy = direction * v * np.sin(yaw)

        # self.x += vx * ts
        # self.y += vy * ts

        # self.yaw = yaw
        # self.v = v
        # self.gear = gear

class VehicleState:  # 车辆状态, 默认\beta = 0
    def __init__(self, x=0.0, y=0.0, yaw=0.0,
                 v=0.0, vy=0.0, w=0.0, gear=Gear.GEAR_DRIVE):
        self.x = x  # 位置
        self.y = y
        self.yaw = yaw  # 整车转向
        self.v = v  # 速度
        self.vy = vy
        self.w = w
        self.e_cg = 0.0  # 偏向距离
        self.theta_e = 0.0  # 偏航角误差

        self.gear = gear  # 挡位  前进和后退
        
        self.UpdateLocalCoor()

    def UpdateLocalCoor(self):
        self.local_x = 0
        self.local_y = 0
        self.local_yaw = 0

        self.sx = self.x
        self.sy = self.y
        self.syaw = self.yaw

    def UpdateVehicleState(self, vy, w, v, e_cg, theta_e, dts,
                           gear=Gear.GEAR_DRIVE):  # 车辆运动学模型

        # wheelbase_ = l_r + l_f
        self.v = self.RegulateOutput(v, max_speed_x)
        self.vy = self.RegulateOutput(vy, max_speed_y)
        self.w = self.RegulateOutput(w, max_steer_w)

        self.gear = gear
        self.x += (self.v * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dts
        self.y += (self.v * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * dts
        self.yaw = modpi(self.yaw + self.w * dts)

        dyaw = self.yaw - self.syaw
        self.local_x += (self.v * math.cos(dyaw) - self.vy * math.sin(dyaw)) * dts
        self.local_y += (self.v * math.sin(dyaw) + self.vy * math.cos(dyaw)) * dts
        self.local_yaw += self.w * dts

        self.e_cg = e_cg
        self.theta_e = theta_e

        # if gear == Gear.GEAR_DRIVE:
        #     self.v += ax * dts
        # else:
        #     self.v += -1.0 * ax * dts
        # self.vy += ay * dts
        # self.w += alpha * dts


    def GetOmniVel(self):
        return self.v, self.y, self.w

    @staticmethod
    def RegulateInput(ay, alpha, ax):  # 加速度限制
        """
        regulate delta to : - max_steer_angle ~ max_steer_angle
        regulate a to : - max_acceleration ~ max_acceleration
        :param delta: steering angle [rad]
        :param a: acceleration [m / s^2]
        :return: regulated delta and acceleration
        """

        if alpha < -1.0 * max_steer_alpha:
            alpha = -1.0 * max_steer_alpha

        if alpha > 1.0 * max_steer_alpha:
            alpha = 1.0 * max_steer_alpha

        if ay < -1.0 * max_acceleration:
            ay = -1.0 * max_acceleration

        if ay > 1.0 * max_acceleration:
            ay = 1.0 * max_acceleration

        if ax < -1.0 * max_acceleration:
            ax = -1.0 * max_acceleration

        if ax > 1.0 * max_acceleration:
            ax = 1.0 * max_acceleration

        return ay, alpha, ax

    @staticmethod
    def RegulateOutput(v, maxv):  # 限制测量的输入
        """
        regulate v to : -max_speed ~ max_speed
        :param v: calculated speed [m / s]
        :return: regulated speed
        """

        max_speed_ = maxv

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

        x_cg = vehicle_state.local_x
        y_cg = vehicle_state.local_y
        yaw = vehicle_state.local_yaw

        # x_cg = vehicle_state.x
        # y_cg = vehicle_state.y
        # yaw = vehicle_state.yaw

        # calc nearest point in ref path  计算车辆到跟踪轨迹上所有点的横纵坐标值
        dx = [x_cg - ix for ix in self.x_[self.ind_old: self.ind_end]]  # 车辆 - 离散轨迹点
        dy = [y_cg - iy for iy in self.y_[self.ind_old: self.ind_end]]

        ind_add = int(np.argmin(np.hypot(dx, dy)))  # 在轨迹种找到离车辆位置最接近的索引  匹配点
        dist = math.hypot(dx[ind_add], dy[ind_add])  # 车辆与匹配轨迹点的距离

        # calc lateral relative position of vehicle to ref path 
        vec_axle_rot_90 = np.array([[math.cos(yaw + math.pi / 2.0)],
                                    [math.sin(yaw + math.pi / 2.0)]])  # 法线

        vec_path_2_cg = np.array([[dx[ind_add]],
                                  [dy[ind_add]]])

        if np.dot(vec_axle_rot_90.T, vec_path_2_cg) > 0.0:  # 夹角小于 90, 位置在frenet右侧
            e_cg = 1.0 * dist  # vehicle on the right of ref path
        else:
            e_cg = -1.0 * dist  # vehicle on the left of ref path

        # calc yaw error: theta_e = yaw_vehicle - yaw_ref
        self.ind_old += ind_add
        yaw_ref = self.yaw_[self.ind_old]  # 航向参考
        theta_e = pi_2_pi(yaw - yaw_ref)  # 车辆偏航与轨迹偏航的偏差

        # calc ref curvature
        k_ref = self.k_[self.ind_old]  # 轨迹曲率

        return theta_e, e_cg, yaw_ref, k_ref

class LatController:  # 横向控制器，调节前轮转角使得 err 最小化
    """
    Lateral Controller using LQR
    """

    def ComputeControlCommand(self, vehicle_state, ref_trajectory, dts):
        """
        calc lateral control command.
        :param vehicle_state: vehicle state
        :param ref_trajectory: reference trajectory (analyzer)
        :return: steering angle (optimal u), theta_e, e_cg
        """

        ts_ = dts
        e_cg_old = vehicle_state.e_cg
        theta_e_old = vehicle_state.theta_e

        theta_e, e_cg, yaw_ref, k_ref = \
            ref_trajectory.ToTrajectoryFrame(vehicle_state)  # 车辆与轨迹偏航角偏差， 横向偏差，轨迹偏航角，轨迹曲率

        matrix_ad_, matrix_bd_ = self.UpdateMatrix(vehicle_state, dts)  # 系统矩阵A，输入矩阵B

        matrix_state_ = np.zeros((state_size, 1))  # [[d, dt, e, de]]
        matrix_r_ = np.diag(matrix_r)
        matrix_q_ = np.diag(matrix_q)

        matrix_k_ = self.SolveLQRProblem(matrix_ad_, matrix_bd_, matrix_q_,
                                         matrix_r_, eps, max_iteration)  # 反馈增益矩阵求解

        matrix_state_[0][0] = e_cg
        matrix_state_[1][0] = (e_cg - e_cg_old) / ts_
        matrix_state_[2][0] = theta_e
        matrix_state_[3][0] = (theta_e - theta_e_old) / ts_

        matrix_u_ = -matrix_k_ @ matrix_state_

        vy = matrix_u_[0][0]  # LQR
        omega = matrix_u_[1][0]

        # ay = -0.5 * e_cg
        # alpha = -0.5 * theta_e
        # return ay, alpha, theta_e, e_cg
        return vy, omega, theta_e, e_cg

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
    def UpdateMatrix(vehicle_state, dts):  # 更新线性化系统矩阵
        """
        calc A and b matrices of linearized, discrete system.
        :return: A, b
        """

        ts_ = dts

        v = vehicle_state.v
        
        matrix_ad_ = np.zeros((state_size, state_size))  # time discrete A matrix

        matrix_ad_[0][0] = 1.0
        matrix_ad_[0][1] = ts_
        matrix_ad_[1][2] = v
        matrix_ad_[2][2] = 1.0
        matrix_ad_[2][3] = ts_

        # matrix_bd_ = np.eye(state_size)  # time discrete b matrix
        matrix_bd_ = np.zeros((state_size, 2))
        matrix_bd_[1][0] = 1
        matrix_bd_[3][1] = 1

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

        # a = 0.6 * (target_speed - direct * vehicle_state.v)

        a = 0.6 * (target_speed - direct * vehicle_state.v)

        return a

    @staticmethod
    def ControlSpeedByDistance(target_dist, dist, max_theta, theta, vehicle_state, speed_x):
        
        # spe_x = speed_x
        # if theta > max_theta or dist < target_dist:
        #     spe_x *= lon_theta_p
        spe_x = lon_p * (dist - target_dist)
        # print(f"vx: {spe_x:.2f}")
        if theta > max_theta:
            spe_x *= lon_theta_p

        return spe_x, 0.6 * (spe_x - vehicle_state.v)


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


def modpi(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi

def generate_path(s, maxc=1, step=0.1):
    """
    design path using reeds-shepp path generator.
    divide paths into sections, in each section the direction is the same.
    :param s: objective positions and directions.
    :return: paths
    """
    # wheelbase_ = l_f + l_r

    path_x, path_y, yaw, rc = [], [], [], []
    x_rec, y_rec, yaw_rec, rc_rec = [], [], [], []

    for i in range(len(s) - 1):  # 多点
        # s_x, s_y, s_yaw = s[i][0], s[i][1], np.deg2rad(s[i][2])
        # g_x, g_y, g_yaw = s[i + 1][0], s[i + 1][1], np.deg2rad(s[i + 1][2])

        s_x, s_y, s_yaw = s[i][0], s[i][1], s[i][2]
        g_x, g_y, g_yaw = s[i + 1][0], s[i + 1][1], s[i + 1][2]
        path_i = dp.calc_dubins_path(s_x, s_y, s_yaw,
                                      g_x, g_y, g_yaw, maxc, step)  # 两点之间选择最优的路径

        irc, _ = dp.calc_curvature(path_i.x, path_i.y, path_i.yaw)  # irc曲率  rds弧长

        ix = path_i.x
        iy = path_i.y
        iyaw = path_i.yaw

        for j in range(len(ix)):
                x_rec.append(ix[j])
                y_rec.append(iy[j])
                yaw_rec.append(iyaw[j])
                rc_rec.append(irc[j])

    path_x.append(x_rec)
    path_y.append(y_rec)
    yaw.append(yaw_rec)
    rc.append(rc_rec)

    x_all, y_all, yaw_all, rc_all = [], [], [], []
    for ix, iy, iyaw, irc in zip(path_x, path_y, yaw, rc):
        x_all += ix
        y_all += iy
        yaw_all += iyaw
        rc_all += irc

    return path_x, path_y, yaw, rc, x_all, y_all, yaw_all, rc_all


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



def main2(filename):
    # obj_plan_points = [[0, 0, 0], [4, 4, 0], [6, 2, -10], [7, -2, -50], [-3, -1, -100], [-4, 2, 120]]
    obj_plan_points = [[0, 0, 0], [4, 0.5, 0]]
    # obj_plan_points = [[0, 0, 0], [4, 0.5, 0], [6, -1, -20], [3, -3, -160], [0, 0, 160]]
    obj_state = ObjectState(0, 0, 0, 0.1, Gear.GEAR_DRIVE)
    vehicle_state = VehicleState(0, 0, 0, 0.1, gear=Gear.GEAR_DRIVE)  # 初始化车辆状态
    err_files = []
    @gif.frame
    def plott(x_all, y_all):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        plt.title("The distance: {:.2f} m, idx: {}".format(dist, len(fs)))
        plt.plot(x_all, y_all, color='gray', linewidth=2.0)  # ref
        # plt.plot(x_rec, y_rec, linewidth=2.0, color='darkviolet')  # 机器人形势路线
        draw.draw_person(obj_state.x, obj_state.y, obj_state.yaw)
        draw.draw_omni_wheel_vehicle(vehicle_state.x, vehicle_state.y, vehicle_state.yaw)
        time.sleep(ts)

    lat_controller = LatController()  # 横向角度控制器LQR
    lon_controller = LonController()  # 纵向距离控制器PID
    valgenerator = PlanGenerator(obj_plan_points, obj_maxc, obj_step)
    # valgenerator = CircleGenerator()
    fs = []
    x_objs, y_objs, yaw_objs, x_vehicles, y_vehicles, yaw_vehicles = [], [], [], [], [], []
    err_ds, err_thetas, err_dists, obj_thetas = [], [], [], []
    is_follow = True
    path_cnt = 0  # 当前执行的跟踪点数
    ref_trajectory = None
    ay_opt, alpha_opt, a_opt, e_cg, theta_e = 0.0, 0.0, 0.0, 0.0, 0.0
    while True and is_follow:
        is_follow, obj_nx, obj_ny, obj_nyaw = valgenerator.update()
        obj_state.UpdateObjectState(obj_nx, obj_ny, obj_nyaw)
        x_objs.append(obj_state.x)
        y_objs.append(obj_state.y)
        yaw_objs.append(obj_state.yaw)
        # dist和theta信息都是相对机器人本身，在仿真时通过全局坐标获取，在实战中通过传感器获取
        dist = math.hypot(vehicle_state.x - obj_state.x, vehicle_state.y - obj_state.y)
        obj_theta = math.atan2(obj_state.y - vehicle_state.y, obj_state.x - vehicle_state.x) - vehicle_state.syaw

        vehicle_state.UpdateVehicleState(ay_opt, alpha_opt, a_opt, e_cg, theta_e, 0.1)  # 更新车辆位置信息
        err_ds.append(e_cg)
        err_thetas.append(theta_e)
        err_dists.append(dist)
        obj_thetas.append(obj_theta)
        err_files.append([abs(e_cg), abs(theta_e), abs(dist - follow_dist_threshold)])
        x_vehicles.append(vehicle_state.x)
        y_vehicles.append(vehicle_state.y)
        yaw_vehicles.append(vehicle_state.yaw)

        print("obj_x: {:.2f}, obj_y: {:.2f}, obj_yaw: {:.2f}".format(obj_state.x, obj_state.y, obj_state.yaw))
        print("vehicle_x: {:.2f}, vehicle_y: {:.2f}, vehicle_yaw: {:.2f}, vx: {:.2f}, vy: {:.2f}, w: {:.2f}".format(vehicle_state.x, vehicle_state.y, vehicle_state.yaw, vehicle_state.v, vehicle_state.vy, vehicle_state.w))
        print("a_opt: {:.2f}, ay_opt: {:.2f}, alpha_opt: {:.2f}, e_cg: {:.2f}, theta_e: {:.2f}".format(a_opt, ay_opt, alpha_opt, e_cg, theta_e))
        print("\n")

        if dist < 0.1:
            continue

        if path_cnt == 0:
            vehicle_state.UpdateLocalCoor()
            obj_theta = math.atan2(obj_state.y - vehicle_state.y, obj_state.x - vehicle_state.x) - vehicle_state.syaw
            obj_x, obj_y = dist * math.cos(obj_theta), dist * math.sin(obj_theta)
            x_ref, y_ref, yaw_ref, curv, x_all, y_all, yaw_all, rc_all = generate_path([
                [vehicle_state.local_x, vehicle_state.local_y, vehicle_state.local_yaw],  # 起始点
                [obj_x, obj_y, obj_theta]  # 末端点
                # [vehicle_state.x, vehicle_state.y, vehicle_state.yaw],  # 起始点
                # [obj_state.x, obj_state.y, obj_state.yaw]  # 末端点
            ], plan_maxc, plan_step)  # 生成目标与机器人之间的路径
            ref_trajectory = TrajectoryAnalyzer(x_all[:path_cnt_choose], y_all[:path_cnt_choose], yaw_all[:path_cnt_choose], rc_all[:path_cnt_choose])  # 初始化当前两点跟踪轨迹

        path_cnt = (path_cnt + 1) % path_cnt_choose
        print("idx: {}, cnt: {}, follow dist: {:.2f}, theta: {:.2f}, x_size: {:d}".format(len(fs), path_cnt, dist, obj_theta, len(x_all)))
        ay_opt, alpha_opt, theta_e, e_cg = \
            lat_controller.ComputeControlCommand(vehicle_state, ref_trajectory, 0.1)  # 车辆横向位置调节，[车辆转向控制角度，偏航角误差，横向偏移]
        # a_opt = lon_controller.ControlSpeedByDistance(follow_dist_threshold, vehicle_state, dist)  # 距离调节加速度
        speed_x = const_speed_x
        # if path_cnt < path_cnt_choose * turn_choose_p:
        #     speed_x *= turn_speed_p

        if abs(obj_theta) > np.deg2rad(20) or dist < follow_dist_threshold:
            speed_x *= 0.01

        a_opt = lon_controller.ComputeControlCommand(speed_x, vehicle_state, dist)  # 恒速调节
        f = plott(x_all, y_all)
        fs.append(f)

        # is_follow, obj_nx, obj_ny, obj_nyaw = valgenerator.update()
        # obj_state.UpdateObjectState(obj_nx, obj_ny, obj_nyaw)
        # dist = math.hypot(vehicle_state.x - obj_state.x, vehicle_state.y - obj_state.y)
        # x_objs.append(obj_state.x)
        # y_objs.append(obj_state.y)
        # yaw_objs.append(obj_state.yaw)


    err_files_np = np.array(err_files)
    err_means = err_files_np.mean(axis=0)
    err_files_np = np.vstack((err_files_np, err_means))

    # np.savetxt(filename, err_files_np, fmt="%.2f %.2f %.2f")

    # vis_gif = str(increment_path("output/eval/visual/gif/Omni_Follow_Plan_DP_whole.gif", exist_ok=False))
    vis_plt = str(increment_path("output/eval/visual/plt/Omni_Follow_Plan_DP_whole.jpg", exist_ok=False))
    err_plt = str(increment_path("output/eval/err/Omni_Follow_Plan_DP_whole.jpg", exist_ok=False))
    # gif.save(fs, vis_gif, duration=50)
    draw_follow_show(x_objs, y_objs, yaw_objs, x_vehicles, y_vehicles, yaw_vehicles, vis_plt)
    draw_control_err(err_ds, err_thetas, err_dists, obj_thetas, err_plt)

    return err_means
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
    x_ref, y_ref, yaw_ref, curv, x_all, y_all, _, _ = generate_path(states)  # 路径生成

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"The code block took {elapsed_time} seconds to execute.")

    # wheelbase_ = l_f + l_r

    maxTime = 100.0
    yaw_old = 0.0
    x0, y0, yaw0 = \
        x_ref[0][0], y_ref[0][0], yaw_ref[0][0]

    x_rec, y_rec, yaw_rec = [], [], []

    lat_controller = LatController()  # 横向角度控制器LQR
    lon_controller = LonController()  # 纵向距离控制器PID
    fs = []
    for x, y, yaw, k in zip(x_ref, y_ref, yaw_ref, curv):

        t = 0.0

        ref_trajectory = TrajectoryAnalyzer(x, y, yaw, k)  # 初始化当前两点跟踪轨迹

        vehicle_state = VehicleState(x=x0, y=y0, yaw=yaw0, v=0.1)  # 初始化车辆状态

        while t < maxTime:

            dist = math.hypot(vehicle_state.x - x[-1], vehicle_state.y - y[-1])  # 目标距离

            target_speed = 25.0 / 3.6

            delta_opt, theta_e, e_cg = \
                lat_controller.ComputeControlCommand(vehicle_state, ref_trajectory)  # 车辆纵向位置调节，[车辆转向控制角度，偏航角误差，横向偏移]

            a_opt = lon_controller.ComputeControlCommand(target_speed, vehicle_state, dist)  # 车辆加速度调节，[加速度]
            print("delta_opt: {:.2f}, a_opt: {:.2f}, e_cg: {:.2f}, theta_e: {:.2f}".format(delta_opt, a_opt, e_cg, theta_e))
            # a_opt = lon_controller.ControlSpeedByDistance(target_speed, vehicle_state, dist + 10)
            vehicle_state.UpdateVehicleState(delta_opt, a_opt, e_cg, theta_e)  # 更新车辆位置信息

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

    main2(None)