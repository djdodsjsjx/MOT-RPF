import numpy as np

# Controller Config
ts = 0.10  # [s]  采样时间
max_iteration = 150  # 最大迭代次数
eps = 0.01  # 迭代收敛值

# 恒定距离   恒定速度
# 权重越大，控制越保守，权重越大，响应越快
matrix_q = [1, 0, 1, 0]  # [1, 0, 1, 0]
matrix_r = [0.01, 0.2]  # ay, alpha [0.01, 0.2]
# matrix_q = [1, 0, 1, 0]  # [1, 0, 1, 0]
# matrix_r = [0.75, 0.5]  # ay, alpha  [1, 0.12]  [1, 0.06]
state_size = 4  # 状态维数


# 全向轮
wheel_radius = 0.07  # 轮子半径 [m]
wheel_width = 0.02   # 轮子宽度 [m]
chassis_radius = 0.2 # 车身半径 [m]
chassis_d = 0.4
follow_dist_threshold = 1  # 跟踪距离 [m] 1
lon_p = 2  # 0.3
lon_theta_p = 0.3

max_acceleration = 0.4  # [m / s^2]  x, y最大加速度  0.4
max_speed_x = 0.6  # [m / s]  0.8
max_speed_y = 0.1  # [m / s]  0.1
const_speed_x = 1.2   # [m / s]  0  0.3
turn_speed_p = 0.05  # 单线程 0.5   多线程0.05
max_steer_w = np.deg2rad(180)  # [rad/s]   90
max_steer_alpha = np.deg2rad(180)  # [rad/s^2]  50  

# 运动目标
obj_maxc = 10  # 最大曲率   10
obj_step = 0.3 # ts秒运动的距离  0.3

# 规划轨迹
plan_maxc = 100  # 最大曲率  1 / R  4
plan_step = obj_step  # ts秒运动的距离
path_cnt_choose = 5  # 两点间选择的跟踪点数设置  10  5
turn_choose_p = 0.5
