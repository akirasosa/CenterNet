from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# noinspection PyUnresolvedReferences
import _init_paths
from demo import time_stats
from detectors.base_detector import BaseDetector
from utils.post_process import multi_pose_post_process

# from detectors.multi_pose import MultiPoseDetector

# %%
img = cv2.imread('/home/akirasosa/.ghq/github.com/akirasosa/CenterNet/images/16004479832_a748d55f21_k.jpg')
# img = cv2.imread('../images/16004479832_a748d55f21_k.jpg')
img.shape


# %%
@dataclass
class Option:
    # load_model: str = Path.home() / 'tmp' / 'res18_no-head_trt.pth'
    load_model: str = Path.home() / 'data' / 'model_best.pth'
    fix_res: bool = True
    input_h: int = 512
    input_w: int = 512
    flip_test: bool = False
    down_ratio: int = 4
    num_classes: int = 1
    hm_hp: bool = True
    reg_offset: bool = True
    reg_hp_offset: bool = True
    mse_loss: bool = False
    K: int = 100
    nms: bool = False
    vis_thresh: float = 0.3
    dataset: str = 'coco_hp'
    debug: int = 1
    debugger_theme: str = 'white'

    arch: str = 'mobilev2'

    heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
    head_conv = 64
    gpus = [0]
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]
    test_scales = [1.0]
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind, shape):
    ind = ind.unsqueeze(2).expand(*shape)
    feat = feat.gather(1, ind)
    return feat


def _tranpose_and_gather_feat(feat, ind, shape):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(shape[0], -1, shape[2])
    feat = _gather_feat(feat, ind, shape)
    return feat


class PostProcess(nn.Module):
    def __init__(self, batch=1, height=128, width=128, hm=1, wh=2, hps=34, reg=2, hm_hp=17, hp_offset=2, K=100):
        super().__init__()
        self.batch = batch
        self.height = height
        self.width = width
        self.hm = hm
        self.wh = wh
        self.hps = hps
        self.reg = reg
        self.hm_hp = hm_hp
        self.hp_offset = hp_offset
        self.K = K

    def _topk(self, scores):
        batch, cat, height, width, K = self.batch, self.hm, self.height, self.width, self.K

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        shape = (batch, K, 1)
        topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind, shape).view(batch, K)
        topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind, shape).view(batch, K)
        topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind, shape).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _topk_channel(self, scores):
        batch, cat, height, width, K = self.batch, self.hm_hp, self.height, self.height, self.K

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs

    def decode(self, heat, wh, kps, reg, hm_hp, hp_offset, K=100):
        batch, cat, height, width = self.batch, self.hm, self.height, self.width
        num_joints = kps.shape[1] // 2
        heat = _nms(heat)
        scores, inds, clses, ys, xs = self._topk(heat)

        kps = _tranpose_and_gather_feat(kps, inds, (batch, K, self.hps))
        kps = kps.view(batch, K, num_joints * 2)
        kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
        kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
        reg = _tranpose_and_gather_feat(reg, inds, (batch, K, self.reg))
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        wh = _tranpose_and_gather_feat(wh, inds, (batch, K, self.wh))
        wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = self._topk_channel(hm_hp)  # b x J x K
        hp_offset = _tranpose_and_gather_feat(hp_offset, hm_inds.view(batch, -1),
                                              (batch, K * self.hm_hp, self.hp_offset))
        hp_offset = hp_offset.view(batch, num_joints, K, 2)
        hm_xs = hm_xs + hp_offset[:, :, :, 0]
        hm_ys = hm_ys + hp_offset[:, :, :, 1]

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1) \
            .unsqueeze(2) \
            .expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1) \
            .expand(batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3) \
            .contiguous() \
            .view(batch, K, num_joints * 2)
        detections = torch.cat([bboxes, scores, kps, clses], dim=2)

        return detections

    def forward(self, x):
        x = {
            'hm': x[0].sigmoid_(),
            'wh': x[1],
            'hps': x[2],
            'reg': x[3],
            'hm_hp': x[4].sigmoid_(),
            'hp_offset': x[5],
        }
        # return list(output.values())
        detections = self.decode(x['hm'], x['wh'], x['hps'], x['reg'], x['hm_hp'], x['hp_offset'])
        return list(x.values()), detections


class Net(nn.Module):
    def __init__(self, pose_net):
        super().__init__()
        self.pose_net = pose_net
        self.post_process = PostProcess()

    def forward(self, x):
        x = self.pose_net(x)
        x = self.post_process(x)
        return x


class MultiPoseDetector(BaseDetector):
    def __init__(self, opt):
        super(MultiPoseDetector, self).__init__(opt)
        self.model = Net(self.model)
        self.model.eval().cuda()

    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()

            output_tmp, dets = self.model(images)
            output = {
                'hm': output_tmp[0],
                'wh': output_tmp[1],
                'hps': output_tmp[2],
                'reg': output_tmp[3],
                'hm_hp': output_tmp[4],
                'hp_offset': output_tmp[5],
            }

        if return_time:
            return output, dets, 0
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = multi_pose_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
            # import pdb; pdb.set_trace()
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='multi_pose')
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
                debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
        # debugger.show_all_imgs(pause=self.pause)
        debugger.save_all_imgs()


# %%
opt = Option()
det = MultiPoseDetector(opt)

# %%
images, _ = det.pre_process(img, scale=1)

# %%
det.process(images.cuda())
pass

# %%
ret = det.run(img)
time_str = ''
for stat in time_stats:
    time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
print(time_str)

# %%
# times = []
#
# for _ in range(100):
#     t0 = time.time()
#     det.pre_process(img, 1)
#     times.append(time.time() - t0)
#
# # %%
# np.mean(times[:5]), np.std(times[:5])
