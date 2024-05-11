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
from trackers.bamsort_tracker.bamsort import OCSort
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
    results_folder_maxid = os.path.join(results_folder, "maxid")
    os.makedirs(results_folder_maxid, exist_ok=True)

    setup_logger(results_folder, distributed_rank=args.local_rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    raw_path = "{}/{}/{}/{}".format(args.raw_results_path, args.dataset, args.det_type, args.dataset_type)  # 检测路径
    dataset = args.dataset
    if args.dataset == "dancetrack" or args.dataset == "myset":
        pic_raw_path = "datasets/{}/{}".format(args.dataset, args.dataset_type)
    else:
        pic_raw_path = "datasets/{}/{}".format(args.dataset, "test" if args.dataset_type == "test" else "train")

    total_time = 0
    total_frame = 0

    test_seqs = get_all_files_in_directory(raw_path)

    for seq_name in test_seqs:
        # if seq_name[:-4] != "MOT17-03-FRCNN" and seq_name[:-4] != "MOT17-12-FRCNN":
        #     continue
        print("starting seq {}".format(seq_name))

        video_name = seq_name[:8]
        ori_thresh = args.track_thresh
        if video_name == 'MOT17-01':  # ByteTrack
            args.track_thresh = 0.65
        elif video_name == 'MOT17-06':
            args.track_thresh = 0.65
        elif video_name == 'MOT17-12':
            args.track_thresh = 0.7
        elif video_name == 'MOT17-14':
            args.track_thresh = 0.67
        else:
            args.track_thresh = ori_thresh

        if video_name == 'MOT20-06' or video_name == 'MOT20-08':
            args.track_thresh = 0.3
        else:
            args.track_thresh = ori_thresh

        tracker = OCSort(args=args, det_thresh = args.track_thresh, iou_threshold=args.iou_thresh, asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia, use_byte=args.use_byte, min_hits=args.min_hits)

        results_filename = os.path.join(results_data, seq_name)
        results_filename_tracker_num = os.path.join(results_folder_tracker_num, seq_name)
        results_filename_switch = os.path.join(results_trkswitch, seq_name)
        results_filename_tracker_maxid = os.path.join(results_folder_maxid, seq_name)

        rffig_seq = os.path.join(results_folder_fig, seq_name[:-4])
        os.makedirs(rffig_seq, exist_ok=True)
        rfconf_seq = os.path.join(results_folder_conf, seq_name[:-4])
        os.makedirs(rfconf_seq, exist_ok=True)
        pic_seq_path = os.path.join(pic_raw_path, seq_name[:-4], "img1")
        if args.save_datasets_pic:
            img_cnt = len([f for f in os.listdir(pic_seq_path) if os.path.isfile(os.path.join(pic_seq_path, f))])
            start_img_idx = 0
            if (args.dataset == "MOT20" or args.dataset == "MOT17") and args.dataset_type == "val":
                start_img_idx = img_cnt // 2

        seq_file = os.path.join(raw_path, seq_name)
        seq_trks = np.loadtxt(seq_file, delimiter=',')
        if args.det_type == "public":
            seq_trks[:,4] += seq_trks[:,2]
            seq_trks[:,5] += seq_trks[:,3]
        min_frame = seq_trks[:,0].min()
        max_frame = seq_trks[:,0].max()
        results = []
        start_row = 0
        results_tracker_num = []
        results_tracker_maxid = []
        for frame_ind in range(int(min_frame), int(max_frame)+1):
            dets = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2:6]  # (x1,y1,x2,y2)
            scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,6]
            cur_dets = np.concatenate((dets, scores.reshape(-1, 1)), axis=1)

            if args.save_datasets_pic:
                if args.dataset == "dancetrack":
                    cur_img = cv2.imread(os.path.join(pic_seq_path, '{:08d}.jpg'.format(frame_ind + start_img_idx)))
                elif args.dataset == "myset":
                    cur_img = cv2.imread(os.path.join(pic_seq_path, '{}.jpg'.format(frame_ind + start_img_idx)))
                else:
                    cur_img = cv2.imread(os.path.join(pic_seq_path, '{:06d}.jpg'.format(frame_ind + start_img_idx)))

            t0 = time.time()
            if args.det_type == "public":
                online_targets = tracker.update_mot_public(cur_dets)
            else:
                online_targets = tracker.update(cur_dets)
            t1 = time.time()
            total_frame += 1
            total_time += t1 - t0

            online_tlwhs = []
            online_ids = []
            maxid = 0
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
                    maxid = max(maxid, tid)
                    # online_scores.append(score)
            # save results
            results.append((frame_ind, online_tlwhs, online_ids))  # 每一帧跟踪器信息: fid, x, y, w, h, tid
            results_tracker_num.append(tracker.save_info(cur_dets))
            results_tracker_maxid.append((frame_ind, maxid))

            if args.save_datasets_pic:  # 保存图片
                online_im = plot_tracking(
                    cur_img, online_tlwhs, online_ids, frame_id=frame_ind, fps=0, distance=0
                )
                cv2.imwrite(rffig_seq + f'/{frame_ind}.jpg', online_im)

                # online_im_conf = vis_notag(cur_img, dets, scores, frame_ind)
                # cv2.imwrite(rfconf_seq + f'/{frame_ind}.jpg', online_im_conf)

        write_results_no_score(results_filename, results)  # 将results写入到result_filename, fid, tid, x, y, w, h
        np.savetxt(results_filename_tracker_num, results_tracker_num, fmt="%d %d %d")
        np.savetxt(results_filename_switch, tracker.get_switch_cnt(), fmt="%d %d")
        np.savetxt(results_filename_tracker_maxid, results_tracker_maxid, fmt="%d %d")
    track_time = 1000 * total_time / total_frame
    logger.info('track_fps: {} '.format(1000 / track_time))

    if args.dataset_type == "test" or args.dataset == "myset":
        return 
    eval_hota(results_data, args.dataset, "val")
    logger.info('Completed')
    return 

def main_ablation(args):

    # tmp = args.tau_s
    # tau_ss = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for tau_s in tau_ss:      
    #     args.expn = "ablation_{}_tau_s".format(tau_s)
    #     args.tau_s = tau_s
    #     main(args)
    # args.tau_s = tmp

    # tmp = args.bec_num
    # bec_nums = [3, 4, 5, 6]
    # for bec_num in bec_nums:      
    #     args.expn = "ablation_{}_bec_num_NOSMD".format(bec_num)
    #     args.bec_num = bec_num
    #     main(args)
    # args.bec_num = tmp

    # tmp = args.w_bec
    # w_becs = [0.1, 0.2, 0.3, 0.4, 0.5]
    # for w_bec in w_becs:      
    #     filename = "{}_w_bec".format(w_bec)
    #     tmp_out_path = args.out_path
    #     args.out_path = os.path.join(args.out_path, filename)
    #     args.w_bec = w_bec
    #     main(args)
    #     args.out_path = tmp_out_path
    # args.w_bec = tmp

    # tmp = args.min_hits
    # min_hits = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # for min_hit in min_hits:      
    #     filename = "{}_min_hit".format(min_hit)
    #     tmp_out_path = args.out_path
    #     args.out_path = os.path.join(args.out_path, filename)
    #     args.min_hits = min_hit
    #     main(args)
    #     args.out_path = tmp_out_path
    # args.min_hits = tmp


    # std_time_since_updates = [5, 10, 15, 20, 25]
    # std_switch_cnts = [0]
    # for std_time_since_update in std_time_since_updates:
    #     for std_switch_cnt in std_switch_cnts:      
    #         filename = "{}_{}_time_switch_cnt".format(std_time_since_update, std_switch_cnt)
    #         tmp_tn_out_path = args.out_path
    #         args.out_path = os.path.join(args.out_path, filename)
    #         args.std_time_since_update = std_time_since_update
    #         args.std_switch_cnt = std_switch_cnt
    #         main(args)
    #         args.out_path = tmp_out_path

    # tmp1 = args.min_hits
    # tmp2 = args.tau_s
    # min_hits = [1, 3, 5, 7, 9]
    # tau_ss = [0.4, 0.5, 0.6, 0.7, 0.8]
    # for min_hit in min_hits: 
    #     args.min_hits = min_hit
    #     for taus in tau_ss:
    #         args.expn = "ablation_{}_min_hit_{}_taus".format(min_hit, taus)
    #         args.tau_s = taus
    #         main(args)
    # args.min_hits = tmp1
    # args.tau_s = tmp2

    # tmp = args.sort_with_std
    # sort_with_stds = [False, True]
    # for sort_with_std in sort_with_stds:      
    #     args.expn = "ablation_{}_sort_with_std".format(1 if sort_with_std else 0)
    #     args.sort_with_std = sort_with_std
    #     main(args)
    # args.sort_with_std = tmp
    tmp = args.asso
    assos = ["iou", "giou", "diou", "ciou"]
    for asso in assos:
        args.expn = "ablation_{}_assos".format(asso)
        args.asso = asso
        main(args)
    args.asso = tmp 

if __name__ == "__main__":
    args = make_parser().parse_args()
    args.dataset = "dancetrack"
    args.dataset_type = "test"
    args.det_type = "yolox_x"
    args.w_bec = 0.3  # bec
    args.bec_num = 4
    args.min_hits = 7  # atm
    args.tau_s = 0.4
    args.std_switch_cnt = 12  # tss
    args.sort_with_tau = False  # False
    args.sort_with_std = True  # True
    args.expn = "mot-rpf-dancetrack"
    main(args)

