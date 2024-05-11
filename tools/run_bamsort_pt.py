'''
    This script makes tracking over the results of existing
    tracking algorithms. Namely, we run OC-SORT over theirdetections.
    Output in such a way is not strictly accurate because
    building tracks from existing tracking results causes loss
    of detections (usually initializing tracks requires a few
    continuous observations which are not recorded in the output
    tracking results by other methods). But this quick adaptation
    can provide a rough idea about OC-SORT's performance on
    more datasets. For more strict study, we encourage to implement 
    a specific detector on the target dataset and then run OC-SORT 
    over the raw detection results.
    NOTE: this script is not for the reported tracking with public
    detection on MOT17/MOT20 which requires the detection filtering
    following previous practice. See an example from centertrack for
    example: https://github.com/xingyizhou/CenterTrack/blob/d3d52145b71cb9797da2bfb78f0f1e88b286c871/src/lib/utils/tracker.py#L83
'''

from loguru import logger
import time

import sys
sys.path.insert(0, 'F:/github/MOT-RPF/')
from trackers.bamsort_tracker.bamsort_pt import OCSort
from utils.utils import write_results, write_results_no_score, write_det_results
from yolox.utils import setup_logger
from utils.args import make_parser
from tools.mota import eval, eval_hota
import os
import motmetrics as mm
import numpy as np
from pathlib import Path
from yolox.utils.visualize import plot_tracking, vis_notag
import cv2
from tools.eval_icvs_pt import eval_pt
from trackers.tracking_utils.timer import Timer
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

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(args):
    results_folder = os.path.join(args.out_path, args.dataset, args.dataset_type, args.expn)
    results_folder = str(increment_path(results_folder, exist_ok=False))
    results_data = os.path.join(results_folder, "data")
    os.makedirs(results_data, exist_ok=True)
    results_folder_tracker_num = os.path.join(results_folder, "track_num")
    os.makedirs(results_folder_tracker_num, exist_ok=True)
    results_trkswitch = os.path.join(results_folder, "trkswitch")
    os.makedirs(results_trkswitch, exist_ok=True)
    results_folder_fig = os.path.join(results_folder, "fig")
    os.makedirs(results_folder_fig, exist_ok=True)
    results_folder_conf = os.path.join(results_folder, "conf")
    os.makedirs(results_folder_conf, exist_ok=True)
    results_folder_res = os.path.join(results_folder, "res")
    os.makedirs(results_folder_res, exist_ok=True)

    setup_logger(results_folder, distributed_rank=args.local_rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    raw_path = "{}/{}/{}/{}".format(args.raw_results_path, args.dataset, args.det_type, args.dataset_type)  # 检测路径
    dataset = args.dataset
    if args.dataset == "dancetrack":
        pic_raw_path = "datasets/{}/{}".format(args.dataset, args.dataset_type)
    elif args.dataset == "pt":
        pic_raw_path = "datasets/{}".format(args.dataset)
    else:
        pic_raw_path = "datasets/{}/{}".format(args.dataset, "test" if args.dataset_type == "test" else "train")

    test_seqs = get_all_files_in_directory(raw_path)

    for seq_name in test_seqs:
        # if seq_name[:-4] != "MOT17-03-FRCNN" and seq_name[:-4] != "MOT17-12-FRCNN":
        #     continue
        print("starting seq {}".format(seq_name))

        # if seq_name[:-4] != "hallway_2" and seq_name[:-4] != "multi_crossings":
        #     continue
        tracker = OCSort(args=args, det_thresh = args.track_thresh, iou_threshold=args.iou_thresh, asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia, use_byte=args.use_byte, min_hits=args.min_hits)  # BAM-SORT

        # tracker = OCSort(det_thresh = args.track_thresh, iou_threshold=args.iou_thresh,
        #     asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia)  # OC-SORT

        results_filename = os.path.join(results_data, seq_name)
        results_filename_tracker_num = os.path.join(results_folder_tracker_num, seq_name)
        results_filename_switch = os.path.join(results_trkswitch, seq_name)

        rffig_seq = os.path.join(results_folder_fig, seq_name[:-4])
        os.makedirs(rffig_seq, exist_ok=True)
        rfconf_seq = os.path.join(results_folder_conf, seq_name[:-4])
        os.makedirs(rfconf_seq, exist_ok=True)
        pic_seq_path = os.path.join(pic_raw_path, seq_name[:-4], "left")
        depth_seq_path = os.path.join(pic_raw_path, seq_name[:-4], "depth")
        img_cnt = len([f for f in os.listdir(pic_seq_path) if os.path.isfile(os.path.join(pic_seq_path, f))])
        start_img_idx = 0
        if (args.dataset == "MOT20" or args.dataset == "MOT17") and args.dataset_type == "val":
            start_img_idx = img_cnt // 2

        seq_file = os.path.join(raw_path, seq_name)
        seq_trks = np.loadtxt(seq_file, delimiter=',')
        min_frame = seq_trks[:,0].min()
        max_frame = seq_trks[:,0].max()
        results = []
        timer = Timer()
        start_row = 0
        results_tracker_num = []
        fid, unfollow_cnt = -1, float('inf')
        for frame_ind in range(int(min_frame), int(max_frame)+1):
            if args.dataset == "dancetrack":
                cur_img = cv2.imread(os.path.join(pic_seq_path, '{:08d}.jpg'.format(frame_ind + start_img_idx)))
            elif args.dataset == "pt":
                cur_img = cv2.imread(os.path.join(pic_seq_path, 'left{:08d}.jpg'.format(frame_ind + start_img_idx - 1)))
                depth_img = cv2.imread(os.path.join(depth_seq_path, 'depth{:08d}.jpg'.format(frame_ind + start_img_idx - 1)), cv2.IMREAD_UNCHANGED)
            else:
                cur_img = cv2.imread(os.path.join(pic_seq_path, '{:06d}.jpg'.format(frame_ind + start_img_idx)))

            dets = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2:6]
            scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,6]
            cur_dets = np.concatenate((dets, scores.reshape(-1, 1)), axis=1)

            # dists = np.zeros((dets.shape[0], 1))
            # for i in range(dets.shape[0]):
            #     cx, cy = int((dets[i][0]+dets[i][2]) // 2), int((dets[i][1]+dets[i][3]) // 2)
            #     cx, cy = max(min(cx, cur_img.shape[1] - 1), 0), max(min(cy, cur_img.shape[0] - 1), 0)
            #     dists[i][0] = depth_img[cy][cx]
            # cur_dets = np.concatenate((dets, dists, scores.reshape(-1, 1)), axis=1)

            timer.tic()
            online_targets = tracker.update(cur_dets)
            timer.toc()
            online_tlwhs = []
            online_ids = []
            # online_scores = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                # score = t[5]
                # tlwh = t.tlwh
                # tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    # online_scores.append(score)

            if unfollow_cnt > 20:
                fid = tracker.GetFollowId(cur_img.shape[1])
            isfollow = False
            for id, tlwh in zip(online_ids, online_tlwhs):
                if id == fid:
                    results.append([frame_ind, int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])])
                    unfollow_cnt = 0
                    isfollow = True
                    break
            if not isfollow:
                results.append([frame_ind, 0, 0, 0, 0])
                unfollow_cnt += 1

            # results_tracker_num.append(tracker.save_info(cur_dets))
            results_tracker_num.append(tracker.save_info(cur_dets))

            if args.save_datasets_pic:  # 保存图片
                online_im = plot_tracking(
                    cur_img, online_tlwhs, online_ids, frame_id=frame_ind, fps=1. / max(1e-5, timer.average_time), follow_id=fid
                )
                cv2.imwrite(rffig_seq + f'/{frame_ind}.jpg', online_im)

                # online_im_conf = vis_notag(cur_img, dets, scores, frame_ind)
                # cv2.imwrite(rfconf_seq + f'/{frame_ind}.jpg', online_im_conf)
        np.savetxt(results_filename, results, fmt="%d %d %d %d %d")
        # np.savetxt(results_filename_tracker_num, results_tracker_num, fmt="%d %d %d")
        # np.savetxt(results_filename_switch, tracker.get_switch_cnt(), fmt="%d %d")

    # eval_hota(results_data, args.dataset, "train")
    eval_pt(results_data, results_folder_res)
    logger.info('Completed')
    # print("Running over {} frames takes {}s. FPS={}".format(total_frame, total_time, total_frame / total_time))
    return 


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.dataset = "pt"
    args.dataset_type = "test"
    args.det_type = "yolox_x"
    args.w_bec = 0.4
    args.bec_num = 4
    args.min_hits = 7 
    args.tau_s = 0.4
    args.std_switch_cnt = 12
    args.sort_with_tau = False
    args.sort_with_std = True
    args.expn = "mot-rpf-pt"
    main(args)