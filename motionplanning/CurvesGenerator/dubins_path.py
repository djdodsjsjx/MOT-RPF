"""
Dubins Path
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")
from motionplanning.CurvesGenerator import draw
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']


# class for PATH element
class PATH:
    def __init__(self, L, mode, x, y, yaw):
        self.L = L  # total path length [float]
        self.mode = mode  # type of each part of the path [string]
        self.x = x  # final x positions [m]
        self.y = y  # final y positions [m]
        self.yaw = yaw  # final yaw angles [rad]


# utils
def pi_2_pi(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi

    while theta < -math.pi:
        theta += 2.0 * math.pi

    return theta


def mod2pi(theta):
    return theta - 2.0 * math.pi * math.floor(theta / math.pi / 2.0)


def LSL(alpha, beta, dist):  # (0, 0, \alpha), (d, 0, \beta)
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    p_lsl = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_a - sin_b)

    if p_lsl < 0:
        return None, None, None, ["WB", "S", "WB"]
    else:
        p_lsl = math.sqrt(p_lsl)

    denominate = dist + sin_a - sin_b
    t_lsl = mod2pi(-alpha + math.atan2(cos_b - cos_a, denominate))
    q_lsl = mod2pi(beta - math.atan2(cos_b - cos_a, denominate))

    return t_lsl, p_lsl, q_lsl, ["WB", "S", "WB"]  # l, p, q 长度


def RSR(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    p_rsr = 2 + dist ** 2 - 2 * cos_a_b + 2 * dist * (sin_b - sin_a)

    if p_rsr < 0:
        return None, None, None, ["R", "S", "R"]
    else:
        p_rsr = math.sqrt(p_rsr)

    denominate = dist - sin_a + sin_b
    t_rsr = mod2pi(alpha - math.atan2(cos_a - cos_b, denominate))
    q_rsr = mod2pi(-beta + math.atan2(cos_a - cos_b, denominate))

    return t_rsr, p_rsr, q_rsr, ["R", "S", "R"]


def LSR(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    p_lsr = -2 + dist ** 2 + 2 * cos_a_b + 2 * dist * (sin_a + sin_b)

    if p_lsr < 0:
        return None, None, None, ["WB", "S", "R"]
    else:
        p_lsr = math.sqrt(p_lsr)

    rec = math.atan2(-cos_a - cos_b, dist + sin_a + sin_b) - math.atan2(-2.0, p_lsr)
    t_lsr = mod2pi(-alpha + rec)
    q_lsr = mod2pi(-mod2pi(beta) + rec)

    return t_lsr, p_lsr, q_lsr, ["WB", "S", "R"]


def RSL(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    p_rsl = -2 + dist ** 2 + 2 * cos_a_b - 2 * dist * (sin_a + sin_b)

    if p_rsl < 0:
        return None, None, None, ["R", "S", "WB"]
    else:
        p_rsl = math.sqrt(p_rsl)

    rec = math.atan2(cos_a + cos_b, dist - sin_a - sin_b) - math.atan2(2.0, p_rsl)
    t_rsl = mod2pi(alpha - rec)
    q_rsl = mod2pi(beta - rec)

    return t_rsl, p_rsl, q_rsl, ["R", "S", "WB"]


def RLR(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    rec = (6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_a - sin_b)) / 8.0

    if abs(rec) > 1.0:
        return None, None, None, ["R", "WB", "R"]

    p_rlr = mod2pi(2 * math.pi - math.acos(rec))
    t_rlr = mod2pi(alpha - math.atan2(cos_a - cos_b, dist - sin_a + sin_b) + mod2pi(p_rlr / 2.0))
    q_rlr = mod2pi(alpha - beta - t_rlr + mod2pi(p_rlr))

    return t_rlr, p_rlr, q_rlr, ["R", "WB", "R"]


def LRL(alpha, beta, dist):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_a_b = math.cos(alpha - beta)

    rec = (6.0 - dist ** 2 + 2.0 * cos_a_b + 2.0 * dist * (sin_b - sin_a)) / 8.0

    if abs(rec) > 1.0:
        return None, None, None, ["WB", "R", "WB"]

    p_lrl = mod2pi(2 * math.pi - math.acos(rec))
    t_lrl = mod2pi(-alpha - math.atan2(cos_a - cos_b, dist + sin_a - sin_b) + p_lrl / 2.0)
    q_lrl = mod2pi(mod2pi(beta) - alpha - t_lrl + mod2pi(p_lrl))

    return t_lrl, p_lrl, q_lrl, ["WB", "R", "WB"]


def interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions):  # 到ind需要走的弧度
    if m == "S":  # 直线
        px[ind] = ox + l / maxc * math.cos(oyaw)
        py[ind] = oy + l / maxc * math.sin(oyaw)
        pyaw[ind] = oyaw
    else:
        ldx = math.sin(l) / maxc  # \theta = l
        if m == "WB":
            ldy = (1.0 - math.cos(l)) / maxc  # r - r*cos(\theta)
        elif m == "R":
            ldy = (1.0 - math.cos(l)) / (-maxc)
        
        gdx = math.cos(-oyaw) * ldx + math.sin(-oyaw) * ldy  # 自然坐标系 -> 车身坐标系(起点坐标系)
        gdy = -math.sin(-oyaw) * ldx + math.cos(-oyaw) * ldy

        px[ind] = ox + gdx
        py[ind] = oy + gdy

    if m == "WB":
        pyaw[ind] = oyaw + l
    elif m == "R":
        pyaw[ind] = oyaw - l

    if l > 0.0:
        directions[ind] = 1
    else:
        directions[ind] = -1
    return px, py, pyaw, directions


def generate_local_course(L, lengths, mode, maxc, step_size):
    point_num = int(L / step_size) + len(lengths) + 3

    px = [0.0 for _ in range(point_num)]
    py = [0.0 for _ in range(point_num)]
    pyaw = [0.0 for _ in range(point_num)]
    directions = [0 for _ in range(point_num)]
    ind = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    if lengths[0] > 0.0:
        d = step_size
    else:
        d = -step_size

    ll = 0.0

    for m, l, i in zip(mode, lengths, range(len(mode))):
        if l > 0.0:
            d = step_size
        else:
            d = -step_size

        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]  # 每一段小圆弧起点位置(车身坐标系)

        ind -= 1  # 不用长一条轨迹的最后一步
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = -d - ll
        else:
            pd = d - ll  # -ll 为上一周期多走的部分

        while abs(pd) <= abs(l):  # pd已经走过的距离
            ind += 1
            px, py, pyaw, directions = \
                interpolate(ind, pd, m, maxc, ox, oy, oyaw, px, py, pyaw, directions)  # 在更新px, py, pyaw中ind位置， 到ind需要走的长度
            pd += d
            # print("ind: {}, pd: {:.2f}, maxc: {:.2f}, ox: {:.2f}, oy:{:.2f}, oyaw: {:.2f}, nx: {:.2f}, ny: {:.2f}, nyaw: {:.2f}".format(ind, pd, maxc, ox, oy, oyaw, px[ind], py[ind], pyaw[ind]))

        ll = pd - l - d  # 此刻 pd > l, 但是没有进行插值
        # ll = l - pd - d  # 源代码

        ind += 1
        px, py, pyaw, directions = \
            interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions)
        # print("ind, {}, nx: {:.2f}, ny: {:.2f}, nyaw: {:.2f}".format(ind, px[ind], py[ind], pyaw[ind]))

    if len(px) <= 1:
        return [], [], [], []

    # remove unused data
    while len(px) >= 1 and px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()

    return px, py, pyaw, directions


def planning_from_origin(gx, gy, gyaw, curv, step_size):
    D = math.hypot(gx, gy)
    d = D * curv
    theta = mod2pi(math.atan2(gy, gx))  # arctan(gy/gx)
    alpha = mod2pi(-theta)  # 2*pi - \theta
    beta = mod2pi(gyaw - theta)

    planners = [LSL, RSR, LSR, RSL, RLR, LRL]  # 可选路径

    best_cost = float("inf")
    bt, bp, bq, best_mode = None, None, None, None

    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)  # 将d轴旋转至x轴，并归一化  (0, 0, 0), (gx, gy, gyaw) -> (0, 0, alpha), (d, 0, beta)

        if t is None:
            continue

        cost = (abs(t) + abs(p) + abs(q))   # 三段距离和
        if best_cost > cost:
            bt, bp, bq, best_mode = t, p, q, mode
            best_cost = cost
    lengths = [bt, bp, bq]  # 选择最优路劲

    x_list, y_list, yaw_list, directions = generate_local_course(
        sum(lengths), lengths, best_mode, curv, step_size)  # 输出为 起点坐标系

    return x_list, y_list, yaw_list, best_mode, best_cost

def planning_from_origin_plt(gx, gy, gyaw, curv, step_size):
    D = math.hypot(gx, gy)
    d = D * curv
    theta = mod2pi(math.atan2(gy, gx))  # arctan(gy/gx)
    alpha = mod2pi(-theta)  # 2*pi - \theta
    beta = mod2pi(gyaw - theta)

    planners = [LSL, RSR, LSR, RSL, RLR, LRL]  # 可选路径

    best_cost = float("inf")
    best_idx = -1
    
    x_lists, y_lists, yaw_lists, modes, cost_lists = [], [], [], [], []
    for i, planner in enumerate(planners):
        t, p, q, mode = planner(alpha, beta, d)  # 将d轴旋转至x轴，并归一化  (0, 0, 0), (gx, gy, gyaw) -> (0, 0, alpha), (d, 0, beta)

        if t is None:
            continue
        cost = (abs(t) + abs(p) + abs(q))
        lengths = [t, p, q]
        x_list, y_list, yaw_list, _ = generate_local_course(
        sum(lengths), lengths, mode, curv, step_size)  # 输出为 起点坐标系

        x_lists.append(x_list)
        y_lists.append(y_list)
        yaw_lists.append(yaw_list)
        modes.append(mode)
        cost_lists.append(cost)

        if best_cost > cost:
            best_idx = i
            best_cost = cost

    return best_idx, x_lists, y_lists, yaw_lists, modes, cost_lists

def calc_dubins_path_plt(sx, sy, syaw, gx, gy, gyaw, curv, step_size=0.1):
    gx = gx - sx
    gy = gy - sy

    l_rot = Rot.from_euler('z', syaw).as_matrix()[0:2, 0:2]  
    le_xy = np.stack([gx, gy]).T @ l_rot  # -> 起点坐标系(机器人当前位置状态)
    le_yaw = gyaw - syaw

    best_idx, lp_xs, lp_ys, lp_yaws, modes, lengthss = planning_from_origin_plt(
        le_xy[0], le_xy[1], le_yaw, curv, step_size)

    rot = Rot.from_euler('z', -syaw).as_matrix()[0:2, 0:2]

    paths = []
    for lp_x, lp_y, lp_yaw, mode, lengths in zip(lp_xs, lp_ys, lp_yaws, modes, lengthss):
        converted_xy = np.stack([lp_x, lp_y]).T @ rot  # 转换到大地坐标系
        x_list = converted_xy[:, 0] + sx
        y_list = converted_xy[:, 1] + sy
        yaw_list = [pi_2_pi(i_yaw + syaw) for i_yaw in lp_yaw]
        paths.append(PATH(lengths, mode, x_list, y_list, yaw_list))

    return best_idx, paths

def calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, curv, step_size=0.1):
    gx = gx - sx
    gy = gy - sy

    l_rot = Rot.from_euler('z', syaw).as_matrix()[0:2, 0:2]  
    le_xy = np.stack([gx, gy]).T @ l_rot  # -> 起点坐标系(机器人当前位置状态)
    le_yaw = gyaw - syaw

    lp_x, lp_y, lp_yaw, mode, lengths = planning_from_origin(
        le_xy[0], le_xy[1], le_yaw, curv, step_size)

    rot = Rot.from_euler('z', -syaw).as_matrix()[0:2, 0:2]
    converted_xy = np.stack([lp_x, lp_y]).T @ rot  # 转换到大地坐标系
    x_list = converted_xy[:, 0] + sx
    y_list = converted_xy[:, 1] + sy
    yaw_list = [pi_2_pi(i_yaw + syaw) for i_yaw in lp_yaw]

    return PATH(lengths, mode, x_list, y_list, yaw_list)


def calc_curvature(x, y, yaw):
    c, ds = [], []

    for i in range(1, len(x) - 1):
        dxn = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        dyn = y[i] - y[i - 1]
        dyp = y[i + 1] - y[i]
        dn = math.hypot(dxn, dyn)
        dp = math.hypot(dxp, dyp)
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp * dyp)
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)
        curvature = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        d = (dn + dp) / 2.0

        if np.isnan(curvature):
            curvature = 0.0
        if len(c) == 0:
            ds.append(d)
            c.append(curvature)

        ds.append(d)
        c.append(curvature)

    ds.append(ds[-1])
    c.append(c[-1])

    return c, ds


def main():
    # choose states pairs: (x, y, yaw)
    # simulation-1
    states = [(0, 0, 0), (10, 10, -90), (20, 5, 60), (30, 10, 120),
              (35, -5, 30), (25, -10, -120), (15, -15, 100), (0, -10, -90)]

    # simulation-2
    # states = [(-3, 3, 120), (10, -7, 30), (10, 13, 30), (20, 5, -25),
    #           (35, 10, 180), (32, -10, 180), (5, -12, 90)]
    
    # states = [(0, 0, 0), (10, 10, -90)]
    # max_steer_angle = np.deg2rad(150)
    # max_c = math.tan(0.5 * max_steer_angle) / chassis_d
    max_c = 0.5  # max curvature  最大曲率半径
    plt.title("Dubins Path", fontsize=20)
    for i in range(len(states) - 1):
        s_x = states[i][0]
        s_y = states[i][1]
        s_yaw = np.deg2rad(states[i][2])
        g_x = states[i + 1][0]
        g_y = states[i + 1][1]
        g_yaw = np.deg2rad(states[i + 1][2])

        best_idx, paths = calc_dubins_path_plt(s_x, s_y, s_yaw, g_x, g_y, g_yaw, max_c)

        for i, path_i in enumerate(paths):
            path_x, path_y, yaw = [], [], []
            for x, y, iyaw in zip(path_i.x, path_i.y, path_i.yaw):
                path_x.append(x)
                path_y.append(y)
                yaw.append(iyaw)
            
            if i == best_idx:
                plt.plot(path_x, path_y, linewidth=2, color='red')
            else:
                plt.plot(path_x, path_y, linewidth=1, color='gray')

    
    for i, (x, y, theta) in enumerate(states):
        draw.Arrow(x, y, np.deg2rad(theta), 1.5, 'blueviolet')
        plt.plot(x, y, 'o', markersize=5, color='green')
    # plt.show()
    plt.savefig("output/eval/path/dubins_path.pdf", format='pdf')

    ## animation
    # plt.ion()
    # plt.figure(1)

    # for i in range(len(path_x)):
    #     plt.clf()
    #     plt.plot(path_x, path_y, linewidth=1, color='gray')

    #     for x, y, theta in states:
    #         draw.Arrow(x, y, np.deg2rad(theta), 2, 'blueviolet')

    #     draw.Car(path_x[i], path_y[i], yaw[i], 1.5, 3)

    #     plt.axis("equal")
    #     plt.title("Simulation of Dubins Path")
    #     plt.axis([-10, 42, -20, 20])
    #     plt.draw()
    #     plt.pause(0.01)
    # plt.pause(1)


if __name__ == '__main__':
    main()
