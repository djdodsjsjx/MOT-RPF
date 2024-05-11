import cv2
import numpy as np
from openni import openni2
from typing import Iterator
import argparse
import os
import os.path as osp
import time
import cv2
import torch
import sys
sys.path.insert(0, '/home/zhy/Track/ocsort')
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.timer import Timer
from pathlib import Path
from tools.robot.pid import PID
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser

import serial
import struct
import queue
import math
from tools import gif
import matplotlib.pyplot as plt
import matplotlib

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


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):  # 检测
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            timer.toc()
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, "person")
        return vis_res


class ColorImage:
    def __init__(self, device_index=0):
        self.cap = cv2.VideoCapture(device_index)
        
    def __iter__(self) -> Iterator:
        return self
        
    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return frame
        
    def release(self):
        self.cap.release()
    def get(self, x):
        return self.cap.get(x)


class DepthImage:
    def __init__(self, w, h):
        openni2.initialize()
        self.w = w
        self.h = h
        self.dev = openni2.Device.open_any()
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.set_video_mode(
            openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM, resolutionX=w, resolutionY=h, fps=30))
        self.depth_stream.start()
        self.depth_data = None
    def __iter__(self) -> Iterator:
        return self
        
    def __next__(self):
        frame = self.depth_stream.read_frame()
        depth_data = np.array(frame.get_buffer_as_uint16()).reshape(self.h, self.w)
        depth_map_scaled = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        self.depth_data = depth_data
        return depth_map_scaled
    def getdistance(self, x, y):
        print(f"x: {x}, y: {y}")
        return self.depth_data[min(y, 479), min(x, 639)]
    def release(self):
        self.depth_stream.stop()
        openni2.unload()

class DistanceFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.average = 0
        self.q = queue.Queue()
        self.sum = 0
    def update(self, distance):
        if distance != 0:
            self.q.put(distance)
            self.sum += distance 
            if self.q.qsize() > self.window_size:
                self.sum -= self.q.get()
        if self.q.qsize() == 0: return 0
        self.average = self.sum / self.q.qsize()
        return self.average

class JetsonUart:
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        self.a = [0x7B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D]

    def send_to_stm(self, vs):
        cnt = 0
        mask = 0
        send_to_stm = bytearray(100)
        for i, v in enumerate(self.a):
            if i < 3: send_to_stm[cnt] = v
            elif i < 9: 
                if i & 1: send_to_stm[cnt] = (vs[(i - 3) // 2] >> 8) & 0xFF
                else: send_to_stm[cnt] = vs[(i - 3) // 2] & 0xFF
            elif i < 10: send_to_stm[cnt] = mask
            else: send_to_stm[cnt] = v
            mask ^= send_to_stm[cnt]
            cnt += 1
        # print(f"send : {send_to_stm[:cnt]}, cnt: {cnt}")
        # print(send_to_stm[:cnt])

        self.ser.write(send_to_stm[:cnt])

    def receive_from_jet(self):
        buf = bytearray(1024)
        num_bytes_read = self.ser.readinto(buf)
        if num_bytes_read > 0 and buf[0] == 0x78 and buf[23] == 0x7D:
            mask = 0
            for i in range(22):
                mask ^= buf[i]
            if mask != buf[22]: return None
            
            # Extract data fields
            Flag_Stop = buf[1]
            X_speed = struct.unpack('>h', buf[2:4])[0]  # Use big-endian byte order
            Y_speed = struct.unpack('>h', buf[4:6])[0]
            Z_speed = struct.unpack('>h', buf[6:8])[0]
            accel_X = struct.unpack('>h', buf[8:10])[0]
            accel_Y = struct.unpack('>h', buf[10:12])[0]
            accel_Z = struct.unpack('>h', buf[12:14])[0]
            gyro_X = struct.unpack('>h', buf[14:16])[0]
            gyro_Y = struct.unpack('>h', buf[16:18])[0]
            gyro_Z = struct.unpack('>h', buf[18:20])[0]
            Power_Voltage = struct.unpack('>H', buf[20:22])[0]
            return {
                'Flag_Stop': Flag_Stop,
                'X_speed': X_speed,
                'Y_speed': Y_speed,
                'Z_speed': Z_speed,
                'accel_X': accel_X,
                'accel_Y': accel_Y,
                'accel_Z': accel_Z,
                'gyro_X': gyro_X,
                'gyro_Y': gyro_Y,
                'gyro_Z': gyro_Z,
                'Power_Voltage': Power_Voltage / 1000.0  # Convert back to floating point number
            }
        else:
            return None


def imageflow_demo(predictor, vis_folder, current_time, args):

    fs = []
    distances = []

    @gif.frame
    def plott(bind):  # 没有历史轨迹
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # 轨迹图初始化
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 3000)
        for i in range(bind+1):
            pv = distances[i - 1]
            v = distances[i]
            l = matplotlib.lines.Line2D([i, i + 1], [pv, v], color="blue")
            ax.add_line(l)


    # cap = cv2.VideoCapture(args.path if args.demo_type == "video" else args.camid)
    color_image = ColorImage()
    width = int(color_image.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(color_image.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    fps = color_image.get(cv2.CAP_PROP_FPS)
    # print(f"width: {width}, height: {height}")
    # ju = JetsonUart()
    depth_image = DepthImage(width, height)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo_type == "video":
        save_path = args.out_path
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
    timer_det, timer_track = Timer(), Timer()
    frame_id = 0
    results = []
    dfd = DistanceFilter(10)
    dfdx = DistanceFilter(10)
    dpid = PID(-0.6, 0, 0, setpoint=1500, sample_time=0)
    dxpid = PID(-0.6, 0, 0, setpoint=0, sample_time=0)
    
    input_ = np.loadtxt("YOLOX_outputs/output.txt", delimiter=',')

    for color_frame, depth_frame in zip(color_image, depth_image):
        if frame_id % 20 == 0:
            logger.info('frame: {:d} det: {:.2f} ms, track: {:.2f} ms'.format(frame_id, timer_det.average_time * 1000, timer_track.average_time * 1000))  # timer.average_time
        outputs, img_info = predictor.inference(color_frame, timer_det)
        # timer_det.toc()
        # result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
        if outputs[0] is not None:
            timer_track.tic()
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                    )
            timer_track.toc()


            # 获取距离
            distance = 0
            dx = 0
            for tlwhs, ids in zip(online_tlwhs, online_ids):
                # if ids == 1:
                x1, y1, w, h = tlwhs
                cx, cy = min(int(x1 + w // 2), width - 1), min(int(y1 + h // 2), height - 1)
                distance = depth_image.getdistance(width - cx, cy)
                dx = width // 2 - cx
                break
            distance = int(dfd.update(distance))
            dx = int(dfdx.update(dx))

            vx = int(dpid(distance))
            vy = int(dxpid(dx))
            vx = min(300, max(-300, vx))
            vy = min(300, max(-300, vy))
            vx = 0 if vx >= -50 and vx <= 50 else vx
            vy = 0 if vy >= -50 and vy <= 50 else vy
            print(f"distance: {distance}, vx: {vx}, dx: {dx}, vy: {vy}")
            # ju.send_to_stm([vx, 0, 0])
            distances.append(distance)
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / (timer_det.average_time + timer_track.average_time), distance=distance
            )
            # logger.info("fps: %.2f" %(1. / timer.average_time))
        else:
            # timer.toc()
            online_im = img_info['raw_img']
        if args.save_result:
            # vid_writer_notrack.write(result_frame)
            vid_writer.write(online_im)
            cv2.imshow("frame", online_im)
            # cv2.imshow("Depth Image", depth_frame)

            # f = plott(frame_id)
            # fs.append(f)
            # if frame_id % 50 == 0:
            #     gif.save(fs, "hh.gif", duration=int((timer_det.average_time + timer_track.average_time) * 1000))
            if frame_id % 50 == 0:
                np.savetxt("YOLOX_outputs/output.txt", distances, fmt='%d')
            ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

def main(exp, args):
    if not args.expn:
        args.expn = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        vis_folder = str(increment_path(vis_folder, exist_ok=False))
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)  # yolox模型加载
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo_type == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo_type == "video" or args.demo_type == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
