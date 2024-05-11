import argparse

def make_parser():
    parser = argparse.ArgumentParser("BAM-SORT parameters")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--output_dir", type=str, default="evaldata/trackers/mot_challenge")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")

    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

    parser.add_argument(
        "-f", "--exp_file",
        default="exps/example/mot/yolox_nano_rpf.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16", dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn for testing.",)
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    
    # det args
    parser.add_argument("-c", "--ckpt", default="pretrained/yolox_nano.pth.tar", type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.8, help="detection confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")  # mot20:0.4
    # parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
    parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
    parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--public", action="store_true", help="use public detection")
    parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
    parser.add_argument("--use_byte", dest="use_byte", default=True, action="store_true", help="use byte in tracking.")

    # for kitti/bdd100k inference with public detections
    parser.add_argument('--raw_results_path', type=str, default="exps",
        help="path to the raw tracking results from other tracks")
    parser.add_argument('--raw_results_public_path', type=str, default="datasets",
        help="path to the raw tracking results from other tracks")
    # parser.add_argument('--out_path', type=str, default="evaldata/trackers/MOT20/improve/train-half/baseline+now", help="path to save output results")
    parser.add_argument('--out_path', type=str, default="evaldata/trackers", help="path to save output results") 
    parser.add_argument("--expn", type=str, default="demo_track0502")
    # parser.add_argument('--tn_out_path', type=str, default="evaldata/trackers/MOT20/tracker_num/val-half/baseline", help="path to save output results")
    parser.add_argument('--det_type', type=str, default="yolox_nano", help="yolox_x | ablation")
    parser.add_argument("--dataset", type=str, default="myset", help="MOT17| MOT20 | dancetrack | kitti | bdd")
    parser.add_argument("--dataset_type", type=str, default="all", help="train | val | test | all")
    # parser.add_argument("--gt_type", type=str, default="", help="| _val_half | _train_half")
    parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
            initializing the tracks (offline).")

    # for demo video
    parser.add_argument("--demo_type", type=str, default="webcam", help="demo type, eg. image, video and webcam")
    parser.add_argument("--demo_out", default="evaldata/trackers/myset", type=str)
    parser.add_argument( "--path", default="dancetrack0005", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    # gesture
    parser.add_argument("-gt", "--sort_with_gesture", default=False, action="store_true")
    parser.add_argument("-gp", "--gesture_path_to_config", default="hagrid/configs/SSDLiteMobileNetV3Small.yaml", type=str, help="Path to config")
    parser.add_argument("-glm", "--glandmarks", default=False, action="store_true", help="Use landmarks")

    parser.add_argument("--sort_with_bec", default=True, action="store_true")
    parser.add_argument("--w_bec", default=0.3, type=float)  # dancetrack&MOT20: 0.3  MOT17&PT: 0.4
    parser.add_argument("--bec_num", default=4, type=int, help=" >= 3")  # dancetrack: 4   MOT17: 4  MOT20: 6
    parser.add_argument("--min_hits", type=int, default=7, help="min hits to create track in SORT")   # dancetrack&PT: 7  MOT17&MOT20: 1
    parser.add_argument("--beta", type=int, default=0.95, help="action tracker factor")
    parser.add_argument("--tau_s", type=int, default=0.7, help="inaction tracker in SORT")
    parser.add_argument("--sort_with_tau", default=False, action="store_true")
    parser.add_argument("--sort_with_std", default=True, action="store_true")
    parser.add_argument("--std_time_since_update", type=int, default=5)  # dancetrack: 5  MOT17: 15   MOT20: 20
    parser.add_argument("--std_switch_cnt", type=int, default=12)  # dancetrack & MOT17&MOT20: 1
    parser.add_argument("--std_max_hits", type=int, default=50)
    
    parser.add_argument("--sort_with_reid", default=False, action="store_true", help="use ReID model for BAM-SORT.")
    parser.add_argument("--w_emb", type=float, default=0.75, help="Combine weight for emb cost")
    parser.add_argument("--alpha_fixed_emb", type=float, default=0.95)
    parser.add_argument("--fast_reid_config", dest="fast_reid_config", default=r"external/fast_reid/configs/CUHKSYSU_DanceTrack/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast_reid_weights", dest="fast_reid_weights", default=r"pretrained/dancetrack_sbs_S50.pth", type=str, help="reid weight path")

    parser.add_argument("--save_datasets_pic", default=False, action="store_true")  # 是否保存数据集运行后的图片，实验分析用
    return parser


