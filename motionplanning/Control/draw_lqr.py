import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")
from motionplanning.Control.config_control import *

PI = np.pi


class Arrow:
    def __init__(self, x, y, theta, L, c):
        angle = np.deg2rad(30)
        d = 0.3 * L
        w = 2

        x_start = x
        y_start = y
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)

        theta_hat_L = theta + PI - angle
        theta_hat_R = theta + PI + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        plt.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=c, linewidth=w)


def draw_car(x, y, yaw, steer, color='black'):
    car = np.array([[-r_b, -r_b, r_f, r_f, -r_b],
                    [v_w / 2, -v_w / 2, -v_w / 2, v_w / 2, v_w / 2]])

    wheel = np.array([[-t_r, -t_r, t_r, t_r, -t_r],
                      [t_w / 2, -t_w / 2, -t_w / 2, t_w / 2, t_w / 2]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                     [np.sin(yaw), np.cos(yaw)]])

    Rot2 = np.array([[np.cos(steer), np.sin(steer)],
                     [-np.sin(steer), np.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[wheelbase], [-wheeldist / 2]])
    flWheel += np.array([[wheelbase], [wheeldist / 2]])
    rrWheel[1, :] -= wheeldist / 2
    rlWheel[1, :] += wheeldist / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    Arrow(x, y, yaw, 0.8 * wheelbase, color)
    plt.axis("equal")
    # plt.show()

def draw_omni_wheel_vehicle(x, y, yaw, color='black'):
    # 创建车身轮廓
    angles = np.linspace(0, 2 * np.pi, 100)
    chassis_x = chassis_radius * np.cos(angles) + x
    chassis_y = chassis_radius * np.sin(angles) + y

    # 为每个轮子创建轮廓
    for i in range(3):
        wheel_angle = (yaw + np.deg2rad(120) * i) % (2 * np.pi)
        wheel_center_x = 0.8 * chassis_radius * np.cos(wheel_angle) + x
        wheel_center_y = 0.8 * chassis_radius * np.sin(wheel_angle) + y

        wheel_outline = np.array([[np.cos(wheel_angle), -np.sin(wheel_angle)],
                                  [np.sin(wheel_angle), np.cos(wheel_angle)]]) @ \
                        np.array([[wheel_width / 2, -wheel_width / 2, -wheel_width / 2, wheel_width / 2, wheel_width / 2],
                                  [-wheel_radius, -wheel_radius, wheel_radius, wheel_radius, -wheel_radius]])
        wheel_outline[0, :] += wheel_center_x
        wheel_outline[1, :] += wheel_center_y

        plt.plot(wheel_outline[0, :], wheel_outline[1, :], color)

    plt.plot(chassis_x, chassis_y, color=color) # 绘制车身
    Arrow(x, y, yaw, 0.8 * chassis_radius, color)

    # plt.axis('equal')
    # plt.show()

def draw_person(x, y, yaw, color='blue'):
    angles = np.linspace(0, 2 * np.pi, 100)
    chassis_x = 0.25 * np.cos(angles) + x
    chassis_y = 0.25 * np.sin(angles) + y
    plt.plot(chassis_x, chassis_y, color=color)
    Arrow(x, y, yaw, 0.8 * 0.25, color)
    # plt.axis('equal')
    # plt.show()




if __name__ == '__main__':
    # draw_car(0, 0, 0, 0.2)
    draw_omni_wheel_vehicle(0, 0, 0)
    draw_person(20, 20, 0)
    # plt.show()