"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
from .association import *
# from external.fast_reid.fast_reid_interfece import FastReIDInterface
# [cur_age - k, cur_age) || [0, cur_age)
def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]

# [x1, y1, x2, y2] => [x, y, s, r]
def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

# [x, y, s, r] => [x1, y1, x2, y2]
def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

# 计算两个检测框之间的运动速度
def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, id, bbox, score, delta_t=3, start_frame=0, max_hits=50, emb=None, orig=False, beta=0.95):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
          from .kalmanfilter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                            0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)  # 位置和速度
        self.time_since_update = 0  # 到当前帧，目标的丢失帧数
        # self.id = KalmanBoxTracker.count  # 轨迹线ID号
        self.id = id
        KalmanBoxTracker.count += 1
        self.history = []  # 跟踪器 仅存放丢失预测状态
        self.hits = 0  # 到当前帧，总匹配的帧数
        self.hit_streak = 0  # 到当前帧，连续匹配帧数
        self.hit_switch_cnt = 0  # 连续匹配到非匹配的切换次数
        self.hit_max_switch_cnt = 0  # 实验用，在非稳态状态下，最大切换次数
        self.start_frame = start_frame
        self.std_max_hits = max_hits
        self.is_activation = False
        self.age = 0  # 预测总帧数
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # 上一次观测状态
        self.observations = dict()  # 每一帧对应检测框
        self.history_observations = []  # 跟踪器 历史观测状态
        self.velocity = None  # 跟踪器的速度
        self.delta_t = delta_t  # 时间间隔
        
        self.emb = emb
        self.score = score
        self.active_score = 1
        self.beta = beta
    def updateId(self, id):
        self.id = id
    def update(self, bbox, score=0):
        """
        Updates the state vector with observed bbox.
        """
        x = 1.0 if bbox is not None else 0.0
        self.active_score = self.active_score * self.beta + (1 - self.beta) * x
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # 寻找上一帧检测框
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)  # 历史观测框(previous_box)与当前观测框(box)之间的速度, 用于下一时刻速度关联矩阵求解
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            if self.hit_streak > self.std_max_hits:  # 连续跟踪std_max_hits次
                self.hit_max_switch_cnt = max(self.hit_switch_cnt, self.hit_max_switch_cnt)
                self.hit_switch_cnt = 0
            self.score = score
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def update_emb(self, emb, alpha=0.9):  # DA
        self.emb = alpha * self.emb + (1 - alpha) * emb
        # self.emb = 0.9 * self.emb + 0.1 * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):  # 目标存在丢失
            if self.hit_streak > 0:  # 丢失第一帧
                self.hit_switch_cnt += 1
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))  # [x1, y1, x2, y2]
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}


class OCSort(object):
    def __init__(self, args, det_thresh, max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age  # 最大存活帧数
        self.min_hits = min_hits  # 最小观测次数
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.trackers_activation = []
        self.trackers_id = set()
        self.frame_count = 0  # 当前帧数
        self.det_thresh = det_thresh  # 高分阈值
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        self.args = args
        if self.args.sort_with_reid:
            self.embedder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, 'cuda')
        KalmanBoxTracker.count = 0
    
    def GetTrackersId(self, sid):
        resid = -1
        for i in range(sid, 100000, 1):
            if i not in self.trackers_id:
                resid = i
                break
        return resid

    def save_info(self, dets):
        res_id = []
        for trt in self.trackers:
            res_id.append(trt.id)
        if dets is None:
            return [self.frame_count, 0, len(res_id)]
        # dets = dets.cpu().numpy()
        return [self.frame_count, len(dets), len(res_id)]

    def get_switch_cnt(self):
        res = []
        for trk in self.trackers:
            res.append([trk.id, max(trk.hit_max_switch_cnt, trk.hit_switch_cnt)])
        return res
    def GetTrackernum(self):
        return len(self.trackers)
    # def update(self, cur_img, output_results, img_info, img_size):
    # def update(self, cur_img, dets):
    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # if output_results is None:
        #     return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        # if output_results.shape[1] == 5:
        #     scores = output_results[:, 4]
        #     bboxes = output_results[:, :4]
        # else:
        #     output_results = output_results.cpu().numpy()
        #     scores = output_results[:, 4] * output_results[:, 5]
        #     bboxes = output_results[:, :4]  # x1y1x2y2
        # img_h, img_w = img_info[0], img_info[1]
        # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        # bboxes /= scale
        # dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        scores = dets[:, 4]

        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        dets_embs = np.ones((dets.shape[0], 1))
        if self.args.sort_with_reid and dets.shape[0] != 0:
            dets_embs = self.embedder.inference(cur_img, dets[:, :4])

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        trks_activation = np.zeros((len(self.trackers_activation), 5))
        taid_to_tid = {}
        to_del = []
        ret = []
        k = 0
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            if self.trackers[t].is_activation:
                trks_activation[k, :] = [pos[0], pos[1], pos[2], pos[3], 0]
                taid_to_tid[k] = t
                k += 1
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        trks_activation = np.ma.compress_rows(np.ma.masked_invalid(trks_activation))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers_activation])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])  # 当前跟踪器上一时刻的检测框
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers_activation])  # [cur_age - k, cur_age) || [0, cur_age)

        """
            First round of association
            matched: 轨迹线与检测框匹配
            unmatched_dets: 该检测框，没有与之匹配的跟踪线
            unmatched_trks: 该跟踪线，没有与之匹配的检测框
        """
        matched, unmatched_dets, unmatched_trks_activation = associate(  # 第一次关联
            dets, trks_activation, self.iou_threshold, velocities, k_observations,
            self.inertia, det_embs=None, trk_embs=None, 
            sort_with_reid=self.args.sort_with_reid, sort_with_bec=self.args.sort_with_bec,
            w_emb=self.args.w_emb, w_bec=self.args.w_bec, bec_num=self.args.bec_num)
        for m in matched:  # 更新轨迹线
            self.trackers[taid_to_tid[m[1]]].update(dets[m[0], :], scores[m[0]])
            self.trackers[taid_to_tid[m[1]]].update_emb(dets_embs[m[0]])

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks_activation.shape[0] > 0:
            u_trks = trks_activation[unmatched_trks_activation]
            # if not self.args.sort_with_eiou:
            #     iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            # else:
            #     iou_left = expend_iou_batch(dets_second, u_trks, self.args.extendr_eiou)
            iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)

            bec_matrix = None
            # if self.args.dataset == "myset":
            if False:
                bec_matrix = bec_score(dets_second, u_trks, self.args.bec_num)
            else:
                bec_matrix = 0

            if iou_left.max() > self.iou_threshold:  # 可改进，被遮挡后人物运动无法观察，IoU不能作为唯一的决定权
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left-bec_matrix*self.args.w_bec)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks_activation[m[1]]
                    dt = 0.2 if self.args.dataset == "dancetrack" else 0
                    if iou_left[m[0], m[1]] < self.iou_threshold - dt:
                        continue
                    # self.trackers_activation[trk_ind].update(dets_second[det_ind, :], scores[det_ind])
                    self.trackers[taid_to_tid[trk_ind]].update(dets_second[det_ind, :], scores[det_ind])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks_activation = np.setdiff1d(unmatched_trks_activation, np.array(to_remove_trk_indices))
        
        all_trks_activation_idx = np.arange(len(self.trackers_activation))
        all_trks_idx = np.arange(len(self.trackers))
        matched_trks_activation = np.setdiff1d(all_trks_activation_idx, unmatched_trks_activation)
        matched_trks = [taid_to_tid[x] for x in matched_trks_activation]
        unmatched_trks = np.setdiff1d(all_trks_idx, matched_trks)

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:  
            left_dets = dets[unmatched_dets]
            # left_trks = last_boxes[unmatched_trks]
            left_trks = trks[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)  # [undet, untrack]
            iou_left = np.array(iou_left)

            bec_matrix = None
            if True:
                bec_matrix = bec_score(left_dets, left_trks, self.args.bec_num)
            else:
                bec_matrix = 0

            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left-bec_matrix*self.args.w_bec)  # asso_func匹配
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    dt = 0.2 if self.args.dataset == "dancetrack" else 0
                    if iou_left[m[0], m[1]] < self.iou_threshold - dt:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :], scores[det_ind])
                    self.trackers[trk_ind].update_emb(dets_embs[det_ind])
                    # 从待删除轨迹中去除
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:  
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)  # [undet, untrack]
            iou_left = np.array(iou_left)

            bec_matrix = None
            if False:
                bec_matrix = bec_score(left_dets, left_trks, self.args.bec_num)
            else:
                bec_matrix = 0

            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left-bec_matrix*self.args.w_bec)  # asso_func匹配
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    dt = 0.2 if self.args.dataset == "dancetrack" else 0
                    if iou_left[m[0], m[1]] < self.iou_threshold - dt:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :], scores[det_ind])
                    self.trackers[trk_ind].update_emb(dets_embs[det_ind])
                    # 从待删除轨迹中去除
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # 创建新轨迹
            # if self.args.sort_with_bgtf and self.frame_count > 20 and (dets[i, 0] > self.args.pix_edge_bgtf or dets[i, 2] + self.args.pix_edge_bgtf < img_w):  # 检测框不在边缘处，则过滤。值越小，可创建的轨迹越少
            #     continue
            if self.args.sort_with_std and (self.args.dataset == "dancetrack" or self.args.dataset == "myset"):
                trkid = self.GetTrackersId(0 if len(dets) <= 20 else KalmanBoxTracker.count)  # dancetrack
            else:
                trkid = KalmanBoxTracker.count  # MOT
            self.trackers_id.add(trkid)
            trk = KalmanBoxTracker(trkid, dets[i, :], scores[i], delta_t=self.delta_t, emb=dets_embs[i], start_frame=self.frame_count, max_hits=self.args.std_max_hits, beta=self.args.beta)
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  # 轨迹激活
                # +1 as MOT benchmark requires positive
                if not trk.is_activation:
                    self.trackers_activation.append(trk)
                    trk.is_activation = True
                    self.active_score = 1.0
                    # trk.updateId(KalmanBoxTracker.count)
                ret.append(np.concatenate((d, [trk.id+1], [trk.score])).reshape(1, -1))  # 返回满足要求的跟踪框

            if self.args.sort_with_tau and trk.is_activation and trk.active_score < self.args.tau_s:  # 非激活轨迹
                self.hit_streak = 0
                self.trackers_activation.remove(trk)
                trk.is_activation = False
            i -= 1
            # remove dead tracklet
            # if (trk.time_since_update > self.max_age) or (self.args.sort_with_std and trk.time_since_update > self.args.std_time_since_update and (trk.hit_switch_cnt > self.args.std_switch_cnt)) :  # mot17: 5, 1  dancetrack: 10, 0
            if trk.time_since_update > self.max_age or (self.args.sort_with_std and trk.hit_switch_cnt > self.args.std_switch_cnt):  
                self.trackers.pop(i)
                trk.is_activation = False
                self.trackers_id.remove(trk.id)

        i = len(self.trackers_activation) - 1
        for trk in reversed(self.trackers_activation):
            if not trk.is_activation:
                self.trackers_activation.pop(i)
            i -= 1
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_mot_public(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # if output_results is None:
        #     return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        # if output_results.shape[1] == 5:
        #     scores = output_results[:, 4]
        #     bboxes = output_results[:, :4]
        # else:
        #     output_results = output_results.cpu().numpy()
        #     scores = output_results[:, 4] * output_results[:, 5]
        #     bboxes = output_results[:, :4]  # x1y1x2y2
        # img_h, img_w = img_info[0], img_info[1]
        # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        # bboxes /= scale
        # dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        
        scores = dets[:, 4]

        # inds_low = scores > 0.1
        # inds_high = scores < self.det_thresh
        # inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        # dets_second = dets[inds_second]  # detections for second matching
        # remain_inds = scores > self.det_thresh
        # dets = dets[remain_inds]

        dets_embs = np.ones((dets.shape[0], 1))
        if self.args.sort_with_reid and dets.shape[0] != 0:
            dets_embs = self.embedder.inference(cur_img, dets[:, :4])

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        trks_activation = np.zeros((len(self.trackers_activation), 5))
        taid_to_tid = {}
        to_del = []
        ret = []
        k = 0
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            if self.trackers[t].is_activation:
                trks_activation[k, :] = [pos[0], pos[1], pos[2], pos[3], 0]
                taid_to_tid[k] = t
                k += 1
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        trks_activation = np.ma.compress_rows(np.ma.masked_invalid(trks_activation))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers_activation])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])  # 当前跟踪器上一时刻的检测框
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers_activation])  # [cur_age - k, cur_age) || [0, cur_age)

        """
            First round of association
            matched: 轨迹线与检测框匹配
            unmatched_dets: 该检测框，没有与之匹配的跟踪线
            unmatched_trks: 该跟踪线，没有与之匹配的检测框
        """
        matched, unmatched_dets, unmatched_trks_activation = associate(  # 第一次关联
            dets, trks_activation, self.iou_threshold, velocities, k_observations,
            self.inertia, det_embs=None, trk_embs=None, 
            sort_with_reid=self.args.sort_with_reid, sort_with_bec=self.args.sort_with_bec,
            w_emb=self.args.w_emb, w_bec=self.args.w_bec, bec_num=self.args.bec_num)
        for m in matched:  # 更新轨迹线
            self.trackers[taid_to_tid[m[1]]].update(dets[m[0], :], scores[m[0]])
            self.trackers[taid_to_tid[m[1]]].update_emb(dets_embs[m[0]])

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks_activation.shape[0] > 0:
            u_trks = trks_activation[unmatched_trks_activation]
            # if not self.args.sort_with_eiou:
            #     iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            # else:
            #     iou_left = expend_iou_batch(dets_second, u_trks, self.args.extendr_eiou)
            iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)

            bec_matrix = None
            if False:
                bec_matrix = bec_score(dets_second, u_trks, self.args.bec_num)
            else:
                bec_matrix = 0

            if iou_left.max() > self.iou_threshold:  # 可改进，被遮挡后人物运动无法观察，IoU不能作为唯一的决定权
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left-bec_matrix*self.args.w_bec)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks_activation[m[1]]
                    dt = 0.2 if self.args.dataset == "dancetrack" else 0
                    if iou_left[m[0], m[1]] < self.iou_threshold - dt:
                        continue
                    # self.trackers_activation[trk_ind].update(dets_second[det_ind, :], scores[det_ind])
                    self.trackers[taid_to_tid[trk_ind]].update(dets_second[det_ind, :], scores[det_ind])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks_activation = np.setdiff1d(unmatched_trks_activation, np.array(to_remove_trk_indices))
        
        all_trks_activation_idx = np.arange(len(self.trackers_activation))
        all_trks_idx = np.arange(len(self.trackers))
        matched_trks_activation = np.setdiff1d(all_trks_activation_idx, unmatched_trks_activation)
        matched_trks = [taid_to_tid[x] for x in matched_trks_activation]
        unmatched_trks = np.setdiff1d(all_trks_idx, matched_trks)

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:  
            left_dets = dets[unmatched_dets]
            # left_trks = last_boxes[unmatched_trks]
            left_trks = trks[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)  # [undet, untrack]
            iou_left = np.array(iou_left)

            bec_matrix = None
            if False:
                bec_matrix = bec_score(left_dets, left_trks, self.args.bec_num)
            else:
                bec_matrix = 0

            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left-bec_matrix*self.args.w_bec)  # asso_func匹配
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    dt = 0.2 if self.args.dataset == "dancetrack" else 0
                    if iou_left[m[0], m[1]] < self.iou_threshold - dt:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :], scores[det_ind])
                    self.trackers[trk_ind].update_emb(dets_embs[det_ind])
                    # 从待删除轨迹中去除
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:  
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)  # [undet, untrack]
            iou_left = np.array(iou_left)

            bec_matrix = None
            if False:
                bec_matrix = bec_score(left_dets, left_trks, self.args.bec_num)
            else:
                bec_matrix = 0

            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left-bec_matrix*self.args.w_bec)  # asso_func匹配
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    dt = 0.2 if self.args.dataset == "dancetrack" else 0
                    if iou_left[m[0], m[1]] < self.iou_threshold - dt:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :], scores[det_ind])
                    self.trackers[trk_ind].update_emb(dets_embs[det_ind])
                    # 从待删除轨迹中去除
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # 创建新轨迹
            # if self.args.sort_with_bgtf and self.frame_count > 20 and (dets[i, 0] > self.args.pix_edge_bgtf or dets[i, 2] + self.args.pix_edge_bgtf < img_w):  # 检测框不在边缘处，则过滤。值越小，可创建的轨迹越少
            #     continue
            if self.args.sort_with_std and self.args.dataset == "dancetrack":
                trkid = self.GetTrackersId(0 if len(dets) <= 20 else KalmanBoxTracker.count)  # dancetrack
            else:
                trkid = KalmanBoxTracker.count  # MOT
            self.trackers_id.add(trkid)
            trk = KalmanBoxTracker(trkid, dets[i, :], scores[i], delta_t=self.delta_t, emb=dets_embs[i], start_frame=self.frame_count, max_hits=self.args.std_max_hits, beta=self.args.beta)
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  # 轨迹激活
                # +1 as MOT benchmark requires positive
                if not trk.is_activation:
                    self.trackers_activation.append(trk)
                    trk.is_activation = True
                    self.active_score = 1.0
                    # trk.updateId(KalmanBoxTracker.count)
                ret.append(np.concatenate((d, [trk.id+1], [trk.score])).reshape(1, -1))  # 返回满足要求的跟踪框

            if self.args.sort_with_tau and trk.is_activation and trk.active_score < self.args.tau_s:  # 非激活轨迹
                self.hit_streak = 0
                self.trackers_activation.remove(trk)
                trk.is_activation = False
            i -= 1
            # remove dead tracklet
            # if (trk.time_since_update > self.max_age) or (self.args.sort_with_std and trk.time_since_update > self.args.std_time_since_update and (trk.hit_switch_cnt > self.args.std_switch_cnt)) :  # mot17: 5, 1  dancetrack: 10, 0
            if trk.time_since_update > self.max_age or (self.args.sort_with_std and trk.hit_switch_cnt > self.args.std_switch_cnt):  
                self.trackers.pop(i)
                trk.is_activation = False
                self.trackers_id.remove(trk.id)

        i = len(self.trackers_activation) - 1
        for trk in reversed(self.trackers_activation):
            if not trk.is_activation:
                self.trackers_activation.pop(i)
            i -= 1
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh

        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        cates_second = cates[inds_second]

        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        cates = cates[remain_inds]


        # remain_inds = scores > self.det_thresh
        
        # cates = cates[remain_inds]
        # dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        trks_activation = np.zeros((len(self.trackers_activation), 5))
        taid_to_tid = {}
        to_del = []
        ret = []
        k = 0
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            if self.trackers[t].is_activation:
                trks_activation[k, :] = [pos[0], pos[1], pos[2], pos[3], cat]
                taid_to_tid[k] = t
                k += 1

            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        trks_activation = np.ma.compress_rows(np.ma.masked_invalid(trks_activation))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers_activation])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers_activation])

        matched, unmatched_dets, unmatched_trks_activation = associate_kitti\
              (dets, trks_activation, cates, self.iou_threshold, velocities, k_observations, self.inertia,
              sort_with_bec=self.args.sort_with_bec, w_bec=self.args.w_bec)

        for m in matched:
            self.trackers[taid_to_tid[m[1]]].update(dets[m[0], :])

        if self.use_byte and len(dets_second) > 0 and unmatched_trks_activation.shape[0] > 0:
            u_trks = trks_activation[unmatched_trks_activation]
            iou_left = self.asso_func(dets_second, u_trks)
            iou_left = np.array(iou_left)
            # det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks_activation[unmatched_trks_activation][:,4]
            # num_dets = unmatched_dets.shape[0]
            num_dets = cates_second.shape[0]
            num_trks_activation = unmatched_trks_activation.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks_activation))
            for i in range(num_dets):
                for j in range(num_trks_activation):
                    if cates_second[i] != trk_cates_left[j]:
                        cate_matrix = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold:
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks_activation[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[taid_to_tid[trk_ind]].update(dets_second[det_ind, :])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks_activation = np.setdiff1d(unmatched_trks_activation, np.array(to_remove_trk_indices))

        all_trks_activation_idx = np.arange(len(self.trackers_activation))
        all_trks_idx = np.arange(len(self.trackers))
        matched_trks_activation = np.setdiff1d(all_trks_activation_idx, unmatched_trks_activation)
        matched_trks = [taid_to_tid[x] for x in matched_trks_activation]
        unmatched_trks = np.setdiff1d(all_trks_idx, matched_trks)


        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = trks[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
        
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trkid = KalmanBoxTracker.count
            trk = KalmanBoxTracker(trkid, dets[i, :], scores[i])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    if not trk.is_activation:
                        self.trackers_activation.append(trk)
                        trk.is_activation = True
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age) or (len(self.trackers) > 20 and trk.time_since_update > 5 and trk.hit_switch_cnt > 1):
                  self.trackers.pop(i)
                  trk.is_activation = False

        i = len(self.trackers_activation) - 1
        for trk in reversed(self.trackers_activation):
            if not trk.is_activation:
                self.trackers_activation.pop(i)
            i -= 1
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))


