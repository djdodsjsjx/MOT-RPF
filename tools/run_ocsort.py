from loguru import logger
import sys
sys.path.insert(0, '/home/zhy/Track/ocsort')
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator, MOTEvaluatorPublic
from utils.args import make_parser

import os
import random
import warnings
import glob
import motmetrics as mmp
from collections import OrderedDict
from pathlib import Path

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

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:       
            print(k)     
            logger.info('Comparing {}...'.format(k))
            os.makedirs("results_log", exist_ok=True)
            vflag = open("results_log/eval_{}.txt".format(k), 'w')
            accs.append(mmp.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5, vflag=vflag))
            names.append(k)
            vflag.close()
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1
    cudnn.benchmark = True

    rank = args.local_rank
    """
        This is for MOT17/MOT20 data configuration
    """
    if exp.val_ann == 'val_half.json':
        gt_type = '_val_half'
        seqs = "{}-val-half".format(exp.dataset_type)
    elif exp.val_ann == "train_half.json":
        gt_type = '_train_half'
        seqs = "{}-train-half".format(exp.dataset_type)
    elif exp.val_ann == "test.json" or "train.json": 
        gt_type = ''
        seqs = "{}-{}".format(exp.dataset_type, "test" if args.test else "train")
    else:
        assert 0
    

    result_folder = "{}_test_results".format(args.expn) if args.test else "{}_results".format(args.expn)  # 评估的文件名
    file_name = os.path.join(exp.output_dir, seqs, result_folder)  # evaldata/trackers/mot_challenge/MOT20-val-half/yolox_x_mix_mot20_ch_results
    file_name = str(increment_path(file_name, exist_ok=False))  # 对已有的文件进行评估，需要注释
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()  # yolox_base.py
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)  # 评估集加载

    if not args.public:
        evaluator = MOTEvaluator(  # 评估器初始化
            args=args,
            dataloader=val_loader,
            img_size=exp.test_size,
            confthre=exp.test_conf,
            nmsthre=exp.nmsthre,
            num_classes=exp.num_classes,
            )
    else:
        evaluator = MOTEvaluatorPublic(
            args=args,
            dataloader=val_loader,
            img_size=exp.test_size,
            confthre=exp.test_conf,
            nmsthre=exp.nmsthre,
            num_classes=exp.num_classes,
            )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)  # .pth.tar模型文件加载
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    results_folder = os.path.join(file_name, "data")
    os.makedirs(results_folder, exist_ok=True)

    # start evaluate  评估  跟踪算法加载 + YOLOX模型的评估 + OC-SORT 跟踪轨迹写入到 results_folder 文件夹中
    # 如果已有跟踪轨迹data文件，可注释，直接运行结果
    *_, summary = evaluator.evaluate_ocsort(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )
    logger.info("\n" + summary)
    if args.test:
        # we skip evaluation for inference on test set
        return 
        
    # evaluate MOTA
    mmp.lap.default_solver = 'lap'
    print('gt_type', gt_type)
    gtfiles = glob.glob(
      os.path.join('datasets/{}/train'.format(exp.dataset_type), '*/gt/gt{}.txt'.format(gt_type)))
    print('gt_files', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mmp.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mmp.lap.default_solver))
    logger.info('Loading files.')
    
    gt = OrderedDict([(Path(f).parts[-3], mmp.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mmp.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles if "detections" not in f])    
    
    mh = mmp.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
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
    # print(mmp.io.render_summary(summary, formatters=fmt, namemap=mmp.io.motchallenge_metric_names))
    logger.info('\n' + mmp.io.render_summary(summary, formatters=fmt, namemap=mmp.io.motchallenge_metric_names))

    metrics = mmp.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # print(mmp.io.render_summary(summary, formatters=mh.formatters, namemap=mmp.io.motchallenge_metric_names))
    logger.info('\n' + mmp.io.render_summary(summary, formatters=mh.formatters, namemap=mmp.io.motchallenge_metric_names))
    logger.info('Completed')

'''
python tools/run_ocsort.py -f exps/example/mot/yolox_s_mot20.py -c pretrained/ocsort_yolox_s_mot20.pth.tar -b 1 -d 1 --fp16 --fuse

python tools/run_ocsort.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -c pretrained/ocsort_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse
'''
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    exp.output_dir = args.output_dir

    if not args.expn:
        args.expn = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
