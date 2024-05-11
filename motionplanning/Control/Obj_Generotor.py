import math
import time
import motionplanning.CurvesGenerator.dubins_path as dp
import numpy as np

class SinGenerator:
    def __init__(self, period=40, A=5):
        self.period = period
        self.A = A
        self.start_time = time.time()

    def update(self):
        current_time = time.time()
        elapsed_time = (current_time - self.start_time) % self.period
        x = (elapsed_time / self.period) * 2 * math.pi
        y = self.A * math.sin(x)
        tangent_angle = math.atan(self.A * math.cos(x))  # tan(yaw) = (y/x)'
        return tangent_angle

class EllipseGenerator:
    def __init__(self, period=40, a=10, b=5):
        """
        :param period: 周期时间
        :param a: 椭圆长轴的一半，对应x轴的幅度
        :param b: 椭圆短轴的一半，对应y轴的幅度
        """
        self.period = period
        self.a = a  # x轴幅度
        self.b = b  # y轴幅度
        self.start_time = time.time()

    def update(self):
        current_time = time.time()
        elapsed_time = (current_time - self.start_time) % self.period
        angle = (elapsed_time / self.period) * 2 * math.pi
        x = self.a * math.cos(angle)
        y = self.b * math.sin(angle)
        tangent_angle = math.atan(-self.b * math.cos(angle) / (self.a * math.sin(angle)))  # tan(yaw) = (y/x)'
        
        # if x > 0 and y > 0:
        #     tangent_angle = math.pi + tangent_angle
        # elif x < 0 and y > 0:
        #     tangent_angle = tangent_angle - math.pi
        if y > 0:
            tangent_angle += math.pi
        return tangent_angle

class EightShapeGenerator:
    def __init__(self, period=40, a=10):
        """
        :param period: 周期时间
        :param a: 形状大小参数
        """
        self.period = period
        self.a = a
        self.start_time = time.time()

    def update(self):
        current_time = time.time()
        elapsed_time = (current_time - self.start_time) % self.period
        angle = (elapsed_time / self.period) * 4 * math.pi  # 八字形是两圆相交组成的
        x = self.a * math.sin(angle)
        y = self.a * math.sin(2 * angle)  # 在八字形中，y随x做正弦变化
        tangent_angle = math.atan(2 * self.a * math.cos(2 * angle) / (self.a * math.cos(angle)))  # tan(yaw) = (y/x)'
        
        if x * y < 0:
            tangent_angle += math.pi
        return tangent_angle

class CircleGenerator:
    def __init__(self, period=40, r=5):
        """
        :param period: 周期时间
        :param r: 圆的半径
        """
        self.period = period
        self.r = r
        self.start_time = time.time()

    def update(self):
        current_time = time.time()
        elapsed_time = (current_time - self.start_time) % self.period
        angle = (elapsed_time / self.period) * 2 * math.pi + 1.5 * math.pi
        # angle = (elapsed_time / self.period) * 2 * math.pi
        x = self.r * math.cos(angle)
        y = self.r * math.sin(angle)
        tangent_angle = math.atan(-self.r * math.cos(angle) / (self.r * math.sin(angle)))  # 切线与x轴夹角
        if y > 0:
            tangent_angle += math.pi
        return tangent_angle


class PlanGenerator:
    def __init__(self, s, maxc=10, step=0.1):  # maxc: 运动曲率  step: ts秒运动的距离
        _, _, _, _, self.x, self.y, self.yaw, _ = self.generate_path(s, maxc, step)
        self.n = len(self.x)
        self.idx = 0
    def generate_path(self, s, maxc, step):

        path_x, path_y, yaw, rc = [], [], [], []
        x_rec, y_rec, yaw_rec, rc_rec = [], [], [], []

        for i in range(len(s) - 1):  # 多点
            s_x, s_y, s_yaw = s[i][0], s[i][1], np.deg2rad(s[i][2])
            g_x, g_y, g_yaw = s[i + 1][0], s[i + 1][1], np.deg2rad(s[i + 1][2])

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


    def update(self):
        if self.idx >= self.n:
            return False, self.x[-1], self.y[-1], self.yaw[self.idx - 1]
        self.idx += 1
        return True, self.x[self.idx - 1], self.y[self.idx - 1], self.yaw[self.idx - 1]

