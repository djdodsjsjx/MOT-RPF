import threading
import time
import sys
sys.path.insert(0, 'F:/github/MOT-RPF')
from motionplanning.Control.Omni_Follow_Model_dp_local_lqr import *
from motionplanning.Control.config_control import *
from motionplanning.Control import gif
import motionplanning.Control.draw_lqr as draw
import matplotlib.pyplot as plt
import queue

def mod2pi(theta):
    return theta - 2.0 * math.pi * math.floor(theta / math.pi / 2.0)


class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff + 0.001
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.01

class MotionPlanning:
    def __init__(self, jetu=None):
        self.thread = None
        self.stop_event = threading.Event()

        self.lat_controller = LatController()  # 横向角度控制器LQR 
        self.lon_controller = LonController()  # 纵向速度控制器PID
        self.vehicle_state = VehicleState(0, 0, 0, 0.1)  # 初始化车辆状态

        self.queue = queue.Queue()
        self.objqueue = queue.Queue()
        self.obj_distance, self.obj_theta, self.obj_x, self.obj_y = 0, 0, 0, 0
        self.isupdate = False
        self.timer = Timer()
        self.err_ds, self.err_thetas, self.err_dists, self.obj_err_thetas = [], [], [], []
        self.jetu = jetu
        self.frame_id = 0
        self.logs = []

        self.obj_whole_x, self.obj_whole_y, self.obj_whole_yaw = 0, 0, 0
        self.obj_whole_xs, self.obj_whole_ys, self.obj_whole_yaws = [], [], []
        self.x_vehicles, self.y_vehicles, self.yaw_vehicles = [], [], []
        self.x_alls, self.y_alls = [], []

    def robot_control(self):
        path_cnt = 0
        ref_trajectory = None
        ay_opt, alpha_opt, a_opt, e_cg, theta_e = 0.0, 0.0, 0.0, 0.0, 0.0
        vy_opt, w_opt, sed_x = 0.0, 0.0, 0.0
        self.timer.tic()
        x_all, y_all = [], []
        while not self.stop_event.is_set():
            self.timer.toc(False)
            self.timer.tic()


            self.vehicle_state.UpdateVehicleState(vy_opt, w_opt, sed_x, e_cg, theta_e, self.timer.duration)  # 速度控制
            # self.obj_distance = math.hypot(self.vehicle_state.local_x - self.obj_x, self.vehicle_state.local_y - self.obj_y)
            # self.obj_theta = modpi(math.atan2(self.obj_y - self.vehicle_state.local_y, self.obj_x - self.vehicle_state.local_x) - self.vehicle_state.syaw)
            # self.logs.append([self.frame_id, self.obj_x, self.vehicle_state.v, sed_x, self.obj_y, self.vehicle_state.w, w_opt])

            if self.jetu:
                jetu_vx = int(self.vehicle_state.v * 1000) if True else 0
                jetu_vy = int(self.vehicle_state.vy * 1000) if True else 0
                jetu_w = int(self.vehicle_state.w * 1000) if True else 0
                self.logs.append([self.frame_id, self.obj_x, jetu_vx, sed_x, self.obj_y, jetu_w, w_opt])
                self.jetu.send_to_stm([jetu_vx, jetu_vy, jetu_w])

            time.sleep(0.04)  # 绘图用

            self.err_ds.append(e_cg)
            self.err_thetas.append(theta_e)
            self.err_dists.append(self.obj_distance)
            self.obj_err_thetas.append(self.obj_theta)
            self.x_vehicles.append(self.vehicle_state.x)
            self.y_vehicles.append(self.vehicle_state.y)
            self.yaw_vehicles.append(self.vehicle_state.yaw)
            self.obj_whole_xs.append(self.obj_whole_x)
            self.obj_whole_ys.append(self.obj_whole_y)
            self.obj_whole_yaws.append(self.obj_whole_yaw)
            self.x_alls.append(x_all)
            self.y_alls.append(y_all)

            if not self.queue.empty():
                self.frame_id, self.obj_distance, self.obj_theta = self.queue.get()
                self.isupdate = True
                self.queue.task_done()

            if not self.objqueue.empty():
                self.obj_whole_x, self.obj_whole_y, self.obj_whole_yaw = self.objqueue.get()
                self.objqueue.task_done()

            if self.obj_distance < 0.5:
                continue 

            print("dist: {:.2f}, theta: {:.2f}, time: {:.4f} ms".format(self.obj_distance, self.obj_theta, self.timer.duration * 1000))

            if self.isupdate:
                self.isupdate = False
                path_cnt = 0
                self.vehicle_state.UpdateLocalCoor()
                self.obj_x, self.obj_y = self.obj_distance * math.cos(self.obj_theta), self.obj_distance * math.sin(self.obj_theta)
                x_ref, y_ref, yaw_ref, curv, x_all, y_all, yaw_all, rc_all = generate_path([
                    [self.vehicle_state.local_x, self.vehicle_state.local_y, self.vehicle_state.local_yaw],  # 起始点 [0, 0, 0]
                    [self.obj_x, self.obj_y, self.obj_theta]  # 末端点  local
                    # [self.vehicle_state.x, self.vehicle_state.y, self.vehicle_state.yaw],  # 起始点
                    # [self.obj_whole_x, self.obj_whole_y, self.obj_whole_yaw]  # 末端点  whole
                ], plan_maxc, plan_step)  # 生成目标与机器人之间的路径
                ref_trajectory = TrajectoryAnalyzer(x_all, y_all, yaw_all, rc_all)  # 初始化当前两点跟踪轨迹

            if ref_trajectory is None:
                continue

            vy_opt, w_opt, theta_e, e_cg = \
            self.lat_controller.ComputeControlCommand(self.vehicle_state, ref_trajectory, self.timer.duration)  # 车辆横向位置调节，[车辆转向控制角度，偏航角误差，横向偏移]
            sed_x, a_opt = self.lon_controller.ControlSpeedByDistance(follow_dist_threshold, self.obj_distance, np.deg2rad(15), abs(self.obj_theta), self.vehicle_state, const_speed_x)

            path_cnt += 1
        if self.jetu:
            self.jetu.send_to_stm([0, 0, 0])

    def run(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.robot_control)
            self.thread.daemon = True
            self.thread.start()

    def UpdateDisTheta(self, frame_id, distance, theta):
        self.queue.put((frame_id, distance, theta))

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.thread = None

    def Updateobjwhole(self, x, y, yaw):
        self.objqueue.put((x, y, yaw))

    def GetRobotPosition(self):  # 仿真用
        return self.vehicle_state.x, self.vehicle_state.y, self.vehicle_state.yaw


def main(filename=None, vis_filename=None, dataset=None):
    mp = MotionPlanning()
    mp.run()
    obj_state = ObjectState(0, 0, 0, 0.1)
    is_follow = True
    obj_distance = 0
    if dataset is None:
        obj_sites = np.loadtxt("datasets/pf/Rectangle.txt", delimiter=' ')
    else:
        obj_sites = np.loadtxt(f"datasets/pf/{dataset}.txt", delimiter=' ')
    i = 0
    while is_follow or obj_distance > 1.001:
        is_follow, obj_nx, obj_ny, obj_nyaw = obj_sites[i if i < len(obj_sites) else -1]
        obj_state.UpdateObjectState(obj_nx, obj_ny, obj_nyaw)
        print("main obj_x: {:.2f}, obj_y: {:.2f}, obj_yaw: {:.2f}".format(obj_state.x, obj_state.y, obj_state.yaw))

        robot_x, robot_y, robot_yaw = mp.GetRobotPosition()  # 全局信息，仿真用
        print(f"main robot_x: {robot_x:.2f}, robot_y: {robot_y:.2f}, robot_yaw: {robot_yaw:.2f}")
        obj_distance = math.hypot(robot_x - obj_state.x, robot_y - obj_state.y)
        if abs(obj_state.x - robot_x) < 1e-3:
            obj_theta = 0.0
        else:
            obj_theta = math.atan2(obj_state.y - robot_y, obj_state.x - robot_x)
        print(obj_theta)
        obj_theta = modpi(obj_theta - robot_yaw)
        print(f"main obj_distance: {obj_distance:.2f}, obj_theta: {obj_theta:.2f}")
        mp.Updateobjwhole(obj_state.x, obj_state.y, obj_state.yaw)
        mp.UpdateDisTheta(i, obj_distance, obj_theta)
        time.sleep(0.2)
        i += 1

    mp.stop()

    # @gif.frame
    def plott(i):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        ax.set_xlim(min(min(mp.obj_whole_xs), min(mp.x_vehicles))-0.5, max(max(mp.obj_whole_xs), max(mp.x_vehicles))+0.5)
        ax.set_ylim(min(min(mp.obj_whole_ys), min(mp.y_vehicles))-0.5, max(max(mp.obj_whole_ys), max(mp.y_vehicles))+0.5)
        plt.title("distance: {:.2f} m, theta: {:.2f} rad".format(mp.err_dists[i], mp.obj_err_thetas[i]))
        ax.plot(mp.x_alls[i], mp.y_alls[i], color='gray', linewidth=2.0)  # ref
        # plt.plot(x_rec, y_rec, linewidth=2.0, color='darkviolet')  # 机器人形势路线
        draw.draw_person(mp.obj_whole_xs[i], mp.obj_whole_ys[i], mp.obj_whole_yaws[i])
        draw.draw_omni_wheel_vehicle(mp.x_vehicles[i], mp.y_vehicles[i], mp.yaw_vehicles[i])
    fs = []
    for i in range(len(mp.obj_whole_xs)):
        f = plott(i)
        fs.append(f)
    vis_gif = str(increment_path(f"evaldata/control/visual/gif/{dataset}.gif", exist_ok=False))
    gif.save(fs, vis_gif, duration=50)

    if len(mp.logs) != 0:
        logtxt = str(increment_path(f"evaldata/control/logs/{dataset}.txt", exist_ok=False))
        np.savetxt(logtxt, mp.logs, fmt='%d %.3f %.3f %.3f %.3f %.3f %.3f')
    vis_plt = str(increment_path(f"evaldata/control/visual/plt/{dataset}.pdf", exist_ok=False))
    err_plt = str(increment_path(f"evaldata/control/err/{dataset}.pdf", exist_ok=False))
    draw_follow_show(mp.obj_whole_xs, mp.obj_whole_ys, mp.obj_whole_yaws, mp.x_vehicles, mp.y_vehicles, mp.yaw_vehicles, vis_plt)
    draw_control_err(mp.err_ds, mp.err_thetas, mp.err_dists, mp.obj_err_thetas, err_plt)


if __name__ == '__main__':

    plans = ["Rectangle", "Ellipse", "Fluid", "Stop-Go"]
    for plan in plans:
        main(dataset=plan)
