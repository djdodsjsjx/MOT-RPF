import numpy as np
import os
import glob
import motmetrics as mm
import sys 
sys.path.insert(0, 'D:/Code/python/DeepLearning/track/OC_SORT/')
from yolox.evaluators.evaluation import Evaluator
from loguru import logger

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def eval_mota(data_root, txt_path):
    accs = []
    seqs = sorted([s for s in os.listdir(data_root) if s.endswith('FRCNN')])  # MOT17  [MOT17-02-FRCNN, ...]
    for seq in seqs:
        video_out_path = os.path.join(txt_path, "data", seq + '.txt')  # 轨迹文件
        evaluator = Evaluator(data_root, seq, 'MOT17', anno="gt_val_half.txt")  # gt文件
        accs.append(evaluator.eval_file(video_out_path))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    # print(strsummary)
    logger.info(strsummary)

def get_mota(data_root, txt_path):
    accs = []
    seqs = sorted([s for s in os.listdir(data_root) if s.endswith('FRCNN')])
    for seq in seqs:
        video_out_path = os.path.join(txt_path, seq + '.txt')
        evaluator = Evaluator(data_root, seq, 'mot')
        accs.append(evaluator.eval_file(video_out_path))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    mota = float(strsummary.split(' ')[-6][:-1])
    return mota


def write_results_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for i in range(results.shape[0]):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[2:6]
            score = frame_data[6]
            line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
            f.write(line)

# n_min: 插值轨迹存在的帧数需要大于n_min， n_dti: 连续插值帧数最大为n_dti
def dti(txt_path, save_path, n_min=25, n_dti=20):
    seq_txts = sorted([f for f in glob.glob(os.path.join(txt_path, '*.txt')) if "detections" not in f])
    for seq_txt in seq_txts:
        seq_name = seq_txt.split('\\')[-1]
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')  # [fid, tid, x, y, w, h]
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        for track_id in range(min_id, max_id + 1):  # 枚举每一个轨迹
            index = (seq_data[:, 1] == track_id)
            tracklet = seq_data[index]  # 找到所有track_id轨迹
            tracklet_dti = tracklet  
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]  # 计算当前轨迹存在的帧总数
            n_conf = np.sum(tracklet[:, 6] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]  # 得到离散帧
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)  # 插值次数
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                        (right_frame - left_frame) + left_bbox  # [x0, x2] => x1, x1 = (t1 - t0) * (x_t2 - x_t0) / (t2 - t0) + x0
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())  # 需要插值的帧数
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))  # 将插值的轨迹放入到原有的轨迹中
            seq_results = np.vstack((seq_results, tracklet_dti))  # 插值后的所有轨迹数组
        save_seq_txt = os.path.join(save_path, seq_name)
        # os.makedirs(save_seq_txt, exist_ok=True)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]  # 将所有轨迹按照帧数递增排序
        write_results_score(save_seq_txt, seq_results)


def dti_kitti(txt_path, save_path, n_min=30, n_dti=20):
    seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.txt')))
    for seq_txt in seq_txts:
        seq_name = seq_txt.split('/')[-1]
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        for track_id in range(min_id, max_id + 1):
            index = (seq_data[:, 1] == track_id)
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]
            n_conf = np.sum(tracklet[:, 6] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                        (right_frame - left_frame) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        write_results_score(save_seq_txt, seq_results)


if __name__ == '__main__':
    # txt_path, save_path = sys.argv[1], sys.argv[2]
    txt_path = "evaldata/trackers/mot_challenge/MOT17-train/yolox_x_mot17_train_results2/data"
    save_path = "evaldata/trackers/mot_challenge/MOT17-train/yolox_x_mot17_train_LD_results2"
    data_root = 'datasets/MOT17/train'
    mkdir_if_missing(save_path)
    dti(txt_path, save_path, n_min=30, n_dti=20)  # 线性插值
    # print('Before DTI: ')
    # eval_mota(data_root, txt_path)
    # print('After DTI:')
    # eval_mota(data_root, save_path)
