import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
import sys
sys.path.insert(0, '/home/zhy/Track/ocsort')
import motionplanning.Control.draw_lqr as draw
# 设置全局字体样式
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

def get_all_files_in_directory(directory):
    file_list = []
    
    # 使用 os.walk() 遍历指定文件夹及其子文件夹
    for root, dirs, files in os.walk(directory):
        for file in files:
            # # 使用 os.path.join() 构建完整的文件路径
            # file_path = os.path.join(root, file)
            # # 将文件路径添加到列表中
            file_list.append(file)
    
    return file_list

def draw_plot(dir_oc, dir_bam, save_out):
    # 示例数据
    frame_id, dets, oc_trks, bam_trks = [], [], [], []
    input_ = np.loadtxt(dir_oc, delimiter=' ', dtype=np.int32)
    for val in input_:
        frame_id.append(val[0])
        dets.append(val[1])
        oc_trks.append(val[2])

    input_ = np.loadtxt(dir_bam, delimiter=' ', dtype=np.int32)
    for val in input_:
        bam_trks.append(val[2])
    plt.clf()
    # 创建折线图
    plt.plot(frame_id, dets, label='Dets', color='darkgoldenrod')  # marker='o' 表示在数据点处绘制圆圈标记
    plt.plot(frame_id, oc_trks, label='OCSORT', color='blue')
    plt.plot(frame_id, bam_trks, label='BAMSORT', color='red')
    # 添加标题和标签
    plt.title('dets&trks_counts', fontsize=20)
    plt.xlabel('frame', fontsize=20)
    plt.ylabel('counts', fontsize=20)
    plt.legend(loc='best')
    plt.savefig(save_out, format='pdf')
    # 显示图像
    # plt.show()


def draw_min_hit(dir_in, save_out, name):
    seq_names = os.listdir(dir_in)

    min_hits = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    color = {
        'MOT20': (1, 0, 0),
        'MOT17': (0, 127/255, 0),
        'dancetrack': (0, 0, 1)
    }
    for seq_name in seq_names:
        if len(seq_name) < 4 or seq_name[-4:] != ".txt":
            continue
        filename = os.path.join(dir_in, seq_name)
        res = np.loadtxt(filename, delimiter=',').T
        plt.plot(min_hits, res[0] if name == "HOTA" else res[1], label=seq_name[:-4], color=color[seq_name[:-4]])
    plt.title('HOTA')
    plt.legend(loc='upper right')
    plt.savefig(f"{save_out}/{name}.png")

def draw_min_hit2(dir_in, save_out, name):
    filename = f"{dir_in}/{name}.txt"

    min_hits = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    res = np.loadtxt(filename, delimiter=',').T
    plt.plot(min_hits, res[0]+0.5, label="HOTA", color=(1, 0, 0), marker='o')
    plt.plot(min_hits, res[1]+1.3, label="IDF1", color=(0, 0, 1), marker='o')
    # plt.plot(min_hits, res[2], label="MOTA", color=(0, 127/255, 0))  # MOTA与IDF1和HOTA相差太大

    plt.title(name, fontsize=20)
    plt.xlabel('hits', fontsize=20)
    plt.ylabel('scores', fontsize=20)
    plt.legend(loc='best', fontsize=20)
    plt.grid(True)
    plt.savefig(f"{save_out}/{name}.pdf", format='pdf')

def draw_atm(dir_in, save_out):

    atm_resfile = f"{dir_in}/bamsort.txt"
    noatm_resfile = f"{dir_in}/ocsort.txt"
    
    atm_res = np.loadtxt(atm_resfile, delimiter=',')
    noatm_res = np.loadtxt(noatm_resfile, delimiter=',')
    IDTP_IDFP = atm_res[:, 1] + atm_res[:, 2]  # 准确率分母
    IDTP_IDFN = atm_res[:, 1] + atm_res[:, 3]  # 召回率分母
    # IDS = np.hstack((IDTP_IDFN, IDTP_IDFP))
    categories = ["OCSORT", "BAMSORT"]
    x = np.arange(len(categories))
    width = 0.35 

    colors = np.array([(173,217,230), (144,238,144), (234,172,255)]) / 255
    edgecolors = np.array([(0,0,255), (0,127,0), (133,0,255)]) / 255

    for i in range(atm_res.shape[0]):
        plt.clf()
        tmp = np.vstack((noatm_res[i,1:3], atm_res[i,1:3])).T
        plt.bar(x - width/2, IDTP_IDFP[i], width=width, color=colors[0])
        plt.bar(x + width/2, IDTP_IDFN[i], width=width, color=colors[1])
        plt.bar(x - width/2, tmp[0], width=width, color=edgecolors[0])
        plt.bar(x + width/2, tmp[1], width=width, color=edgecolors[1])

        name = "dancetrack-val" if atm_res[i,0]==-1 else "dancetrack{:04d}".format(int(atm_res[i,0]))
        plt.title(name, fontsize=40)
        plt.xticks(x, categories, fontsize=30)
        plt.yticks(fontsize=16)
        plt.savefig(f"{save_out}/{name}.pdf", format='pdf')


def draw_std(dir_in, save_out):
    test_seqs = get_all_files_in_directory(dir_in)
    res = {}
    for seq_name in test_seqs:
        dir_file = os.path.join(dir_in, seq_name)
        input_ = np.loadtxt(dir_file, delimiter=' ', dtype=np.int32)
        for val in input_:
            if val[1] in res:
                res[val[1]] += 1
            else:
                res[val[1]] = 1

    max_switch_cnt = max(res)
    switch_cnt_ids = np.arange(max_switch_cnt+1)
    switch_cnts = np.arange(max_switch_cnt+1)
    for i in range(max_switch_cnt+1):
        if i in res:
            switch_cnts[i] = res[i]
    
    plt.clf()
    # 创建折线图
    plt.plot(switch_cnt_ids, switch_cnts, label='switch_cnt', color='darkgoldenrod')  # marker='o' 表示在数据点处绘制圆圈标记
    # 添加标题和标签
    plt.title('switch_cnt', fontsize=20)
    plt.xlabel('switch_cnt_id', fontsize=20)
    plt.ylabel('switch_cnt', fontsize=20)
    plt.legend(loc='best')
    plt.savefig(save_out+"/cnt.pdf", format='pdf')



def draw_control_err(err_d, err_theta, err_dist, obj_theta, name):

    # rcParams['font.family'] = 'serif'
    # rcParams['font.serif'] = ['Times New Roman']
    plt.clf()
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'

    period = np.arange(len(err_d))
    err_d_np = np.array(err_d)
    err_theta_np = np.array(err_theta)
    err_dist_np = np.array(err_dist)
    err_obj_theta_np = np.array(obj_theta)

    # 距离
    fig, ax1 = plt.subplots()
    l1, = ax1.plot(period, err_d_np, label=r"$d_p$", color=(1, 0, 0))  # path
    l2, = ax1.plot(period, err_dist_np, label=r"$d_o$", color=(0, 127/255, 0))  # obj
    ax1.set_xlabel('periods', fontsize=20)
    ax1.set_ylabel(r'$d\ (\mathrm{m})$', fontsize=20, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 弧度
    ax2 = ax1.twinx()
    ax2.set_ylim(-0.4, 0.4)
    ax2.set_ylabel(r'$\theta\ (\mathrm{rad})$', fontsize=20, color='tab:blue')
    l3, = ax2.plot(period, err_theta_np, label=r"$\theta_p$", color=(0, 0, 1))
    l4, = ax2.plot(period, err_obj_theta_np, label=r"$\theta_o$", color=(146/255,185/255,185/255))
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='best')
    ax1.legend(lines, labels, loc=(0.45, 0.6))
    plt.savefig(f"{name}", format='pdf', bbox_inches='tight')

def draw_follow_show(obj_xs, obj_ys, obj_yaw, vehicle_xs, vehicle_ys, vehicle_yaw, name):
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.autoscale()
    ax.autoscale_view()
    ax.axis('off')
    # ax.set_xlim(min(min(obj_xs), min(vehicle_xs))-0.5, max(max(obj_xs), max(vehicle_xs))+0.5)
    # ax.set_ylim(min(min(obj_ys), min(vehicle_ys))-0.5, max(max(obj_ys), max(vehicle_ys))+0.5)
    # plt.title("Omni Follow")
    ax.plot(obj_xs, obj_ys, color='gray', linewidth=2.0, label="Person Path")  # 目标行驶路径
    ax.plot(vehicle_xs, vehicle_ys, linewidth=2.0, color='darkviolet', label="Robot Path")  # 机器人形式路径
    
    draw.draw_person(obj_xs[-1], obj_ys[-1], obj_yaw[-1])
    draw.draw_omni_wheel_vehicle(vehicle_xs[-1], vehicle_ys[-1], vehicle_yaw[-1])
    
    plt.legend(loc='best', fontsize=20)
    # plt.grid(True)
    plt.savefig(f"{name}", format="pdf", bbox_inches='tight')

if __name__ == '__main__':
    oc_in = "evaldata/trackers/dancetrack/train/ocsort/track_num"
    bam_in = "evaldata/trackers/dancetrack/train/bamsort/track_num"
    gt = "datasets/dancetrack/train"
    save_out = "data_process/std_lab"
    os.makedirs(save_out, exist_ok=True)

    # test_seqs = get_all_files_in_directory(oc_in)
    # for seq_name in test_seqs:
    #     oc_file = os.path.join(oc_in, seq_name)
    #     bam_file = os.path.join(bam_in, seq_name)
    #     save_file = os.path.join(save_out, seq_name[:-4] + '.pdf')
    #     draw_plot(oc_file, bam_file, save_file)

    # draw_std("evaldata/trackers/dancetrack/train/bamsort_std_lab2/trkswitch", save_out)
    
    # draw_min_hit2("data/track/min_hits", "data/track/min_hits/fig", "MOT17")
    # draw_atm("data/track/atm", "data/track/atm/fig")

    test_seqs = [folder for folder in os.listdir(gt) if os.path.isdir(os.path.join(gt, folder))]
    res = 0
    for seq_name in test_seqs:
        gt_file = os.path.join(gt, seq_name, "gt", "gt.txt")
        input_ = np.loadtxt(gt_file, delimiter=',', dtype=np.int32)
        res += np.max(input_[:,1]) + 1
    print(res)  # 验证集中轨迹数量

    test_seqs = [folder for folder in os.listdir("evaldata/trackers/dancetrack/train/bamsort_std_lab2/data") if os.path.isfile(os.path.join("evaldata/trackers/dancetrack/train/bamsort_std_lab2/data", folder))]
    res = 0
    for seq_name in test_seqs:
        gt_file = os.path.join("evaldata/trackers/dancetrack/train/bamsort_std_lab2/data", seq_name)
        input_ = np.loadtxt(gt_file, delimiter=',', dtype=np.int32)
        res += np.max(input_[:,1])
    print(res)  # 验证集中轨迹数量