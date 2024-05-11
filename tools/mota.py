from loguru import logger
import sys
sys.path.insert(0, 'F:/BAM-SORT/')
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

import os
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path

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

def eval(results_folder, gt_root, gt_type):
    # evaluate MOTA
    # results_folder = 'YOLOX_outputs/yolox_x_ablation/track_results'  # 需要评估的轨迹路径
    mm.lap.default_solver = 'lap'

    # gt_type = '_val_half'
    #gt_type = ''
    # print('gt_type', gt_type)
    gtfiles = glob.glob(
        os.path.join(gt_root, '*/gt/gt{}.txt'.format(gt_type)))
    # gtfiles = gtfiles[0:1]
    # print('gt_files', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval') and "detections" not in f]
    # tsfiles = tsfiles[0:1]
    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1.0)) for f in tsfiles])    
    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)

    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
                'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
                'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters, 
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                        'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    logger.info('\n' + mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    logger.info('\n' + mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')

def eval_hota(results_folder, dataset, dataset_type):
    if dataset == "dancetrack":
        # python TrackEval/scripts/run_mot_challenge.py --BENCHMARK dancetrack --SPLIT_TO_EVAL val --TRACKERS_TO_EVAL '' --METRICS HOTA CLEAR Identity --TIME_PROGRESS False --TRACKER_SUB_FOLDER '' --GT_FOLDER datasets/dancetrack/ --USE_PARALLEL False --NUM_PARALLEL_CORES 8 --TRACKERS_FOLDER evaldata/trackers/DanceTrack/improve/val/baseline+bec+act --GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       f"--BENCHMARK dancetrack " \
                       f"--SPLIT_TO_EVAL {dataset_type}  " \
                       "--METRICS HOTA CLEAR Identity " \
                       "--TRACKERS_TO_EVAL eval " \
                       "--TIME_PROGRESS False " \
                       "--TRACKER_SUB_FOLDER ''  " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 8 " \
                       "--GT_FOLDER datasets/dancetrack/ " \
                       "--TRACKERS_FOLDER " + results_folder + " "\
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
    elif dataset == "MOT17":
        # python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL '' --METRICS HOTA CLEAR Identity VACE --TIME_PROGRESS False --GT_FOLDER datasets/MOT17/ --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --TRACKERS_FOLDER evaldata/trackers/MOT17/improve/val-half/baseline+bec+act+new --GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       "--BENCHMARK MOT17 " \
                       "--SPLIT_TO_EVAL train " \
                       "--TRACKERS_TO_EVAL eval " \
                       "--METRICS HOTA CLEAR Identity VACE " \
                       "--TIME_PROGRESS False " \
                       "--TRACKER_SUB_FOLDER ''  " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 1  " \
                       "--GT_FOLDER datasets/MOT17/ " \
                       "--TRACKERS_FOLDER " + results_folder + " " \
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_" + "{}_half.txt".format(dataset_type)
                    #    "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_" + "{}_half.txt".format(dataset_type) if dataset_type == "val" else "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
    elif dataset == "MOT20":
        # python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT20 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL '' --METRICS HOTA CLEAR Identity VACE --TIME_PROGRESS False --GT_FOLDER datasets/MOT20/ --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --TRACKERS_FOLDER evaldata/trackers/MOT20/improve/val-half/baseline+bec+act --GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       "--BENCHMARK MOT20 " \
                       "--SPLIT_TO_EVAL train " \
                       "--TRACKERS_TO_EVAL eval " \
                       "--METRICS HOTA CLEAR Identity VACE " \
                       "--TRACKER_SUB_FOLDER ''  " \
                       "--TIME_PROGRESS False " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 1  " \
                       "--GT_FOLDER datasets/MOT20/ " \
                       "--TRACKERS_FOLDER " + results_folder + " " \
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_" + "{}_half.txt".format(dataset_type)
                    #    "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_" + "{}_half.txt".format(dataset_type) if dataset_type == "val" else "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
                    #    "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_" + "{}_half.txt".format(dataset_type)
    elif dataset == "myset":
        # python TrackEval/scripts/run_mot_challenge.py --BENCHMARK myset --SPLIT_TO_EVAL val --TRACKERS_TO_EVAL '' --METRICS HOTA CLEAR Identity --TIME_PROGRESS False --TRACKER_SUB_FOLDER '' --GT_FOLDER datasets/dancetrack/ --USE_PARALLEL False --NUM_PARALLEL_CORES 8 --TRACKERS_FOLDER evaldata/trackers/DanceTrack/improve/val/baseline+bec+act --GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       f"--BENCHMARK myset " \
                       f"--SPLIT_TO_EVAL {dataset_type}  " \
                       "--METRICS HOTA CLEAR Identity " \
                       "--TRACKERS_TO_EVAL eval " \
                       "--TIME_PROGRESS False " \
                       "--TRACKER_SUB_FOLDER ''  " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 8 " \
                       "--GT_FOLDER datasets/myset/ " \
                       "--TRACKERS_FOLDER " + results_folder + " "\
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
    else:
        assert dataset in ["dancetrack", "MOT17", "MOT20", "myset"]
    os.system(hota_command)

    logger.info('Completed')
    # print("Running over {} frames takes {}s. FPS={}".format(total_frame, total_time, total_frame / total_time))
    return 

def eval_hota_all(results_folder, dataset, dataset_type):
    if dataset == "dancetrack":
        # python TrackEval/scripts/run_mot_challenge.py --BENCHMARK dancetrack --SPLIT_TO_EVAL val --TRACKERS_TO_EVAL '' --METRICS HOTA CLEAR Identity --TIME_PROGRESS False --TRACKER_SUB_FOLDER '' --GT_FOLDER datasets/dancetrack/ --USE_PARALLEL False --NUM_PARALLEL_CORES 8 --TRACKERS_FOLDER evaldata/trackers/DanceTrack/improve/val/baseline+bec+act --GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       f"--BENCHMARK dancetrack " \
                       f"--SPLIT_TO_EVAL {dataset_type}  " \
                       "--METRICS HOTA CLEAR Identity " \
                       "--TRACKERS_TO_EVAL eval " \
                       "--TIME_PROGRESS False " \
                       "--TRACKER_SUB_FOLDER ''  " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 8 " \
                       "--GT_FOLDER datasets/dancetrack/ " \
                       "--TRACKERS_FOLDER " + results_folder + " "\
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
    elif dataset == "MOT17":
        # python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL '' --METRICS HOTA CLEAR Identity VACE --TIME_PROGRESS False --GT_FOLDER datasets/MOT17/ --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --TRACKERS_FOLDER evaldata/trackers/MOT17/improve/val-half/baseline+bec+act+new --GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       "--BENCHMARK MOT17 " \
                       "--SPLIT_TO_EVAL train " \
                       "--TRACKERS_TO_EVAL eval " \
                       "--METRICS HOTA CLEAR Identity VACE " \
                       "--TIME_PROGRESS False " \
                       "--TRACKER_SUB_FOLDER ''  " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 1  " \
                       "--GT_FOLDER datasets/MOT17/ " \
                       "--TRACKERS_FOLDER " + results_folder + " " \
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
                    #    "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_" + "{}_half.txt".format(dataset_type) if dataset_type == "val" else "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
    elif dataset == "MOT20":
        # python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT20 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL '' --METRICS HOTA CLEAR Identity VACE --TIME_PROGRESS False --GT_FOLDER datasets/MOT20/ --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --TRACKERS_FOLDER evaldata/trackers/MOT20/improve/val-half/baseline+bec+act --GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       "--BENCHMARK MOT20 " \
                       "--SPLIT_TO_EVAL train " \
                       "--TRACKERS_TO_EVAL eval " \
                       "--METRICS HOTA CLEAR Identity VACE " \
                       "--TRACKER_SUB_FOLDER ''  " \
                       "--TIME_PROGRESS False " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 1  " \
                       "--GT_FOLDER datasets/MOT20/ " \
                       "--TRACKERS_FOLDER " + results_folder + " " \
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
                    #    "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_" + "{}_half.txt".format(dataset_type) if dataset_type == "val" else "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
                    #    "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_" + "{}_half.txt".format(dataset_type)
    else:
        assert dataset in ["dancetrack", "MOT17", "MOT20"]
    os.system(hota_command)

    logger.info('Completed')
    # print("Running over {} frames takes {}s. FPS={}".format(total_frame, total_time, total_frame / total_time))
    return 

if __name__ == '__main__':

    # results_folder = 'evaldata/trackers/DanceTrack/improve/train/baseline'  # 需要评估的轨迹路径
    # gt_root = "datasets/dancetrack/train"
    # gt_type = ''  # '_train_half', '_val_half', ''
    # eval(results_folder, gt_root, gt_type)

    dataset_type = "val"
    dataset = "myset"
    results_folder = "evaldata/trackers/{}/val/bamsort/data".format(dataset)
    eval_hota(results_folder, dataset, dataset_type)
