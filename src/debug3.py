from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# noinspection PyUnresolvedReferences
from torch2trt import torch2trt

import _init_paths
from detectors.base_detector import BaseDetector
from models.model import load_model
from models.networks import mobilenetv2_centernet, mobilenetv3_centernet
from utils.post_process import multi_pose_post_process

# from detectors.multi_pose import MultiPoseDetector

# %%
img = cv2.imread('/home/akirasosa/.ghq/github.com/akirasosa/CenterNet/images/16004479832_a748d55f21_k.jpg')
# img = cv2.imread('../images/16004479832_a748d55f21_k.jpg')
img.shape


# %%
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
    feat = feat.permute(0, 2, 3, 1)
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
        topk_ys = (topk_inds.float() / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind.float() / K).int()
        shape = (batch, K, 1)
        topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind, shape).view(batch, K)
        topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind, shape).view(batch, K)
        topk_inds = _gather_feat(topk_inds.float().view(batch, -1, 1), topk_ind, shape).view(batch, K).long()

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
        num_joints = self.hps // 2
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
        x[0] = x[0].sigmoid_()  # hm
        x[4] = x[4].sigmoid_()  # hm_hp
        detections = self.decode(*x)
        return x + [detections]


class PreProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.408, 0.447, 0.47]).reshape(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.289, 0.274, 0.278]).reshape(1, 3, 1, 1).cuda()

    def forward(self, x):
        # Channel first
        x = x.permute(0, 3, 1, 2)
        # BGR to RGB
        x = x[:, [2, 1, 0]]
        # Standarize
        x = x / 255.
        x = (x - self.mean) / self.std
        return x


class FullNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_process = PreProcess()

        heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
        # pose_net = mobilenetv2_centernet.get_pose_net(0, heads, head_conv=64)
        pose_net = mobilenetv3_centernet.get_pose_net(0, heads, head_conv=64)
        # pose_net = load_model(pose_net, Path.home() / 'data' / 'model_best.pth')
        self.pose_net = pose_net

        self.post_process = PostProcess()

    def forward(self, x):
        x = self.pre_process(x)
        x = self.pose_net(x)
        x = self.post_process(x)

        return x


# %%
net = FullNet().eval().cuda()
x = torch.ones((1, 512, 512, 3)).cuda()
out = net(x)
print(len(out))
# %%
torch.onnx.export(net, x, Path.home() / 'tmp' / 'mobilenetv2_centernet_prepro_postpro_trained.onnx',
                  # verbose=True,
                  input_names=['input0'],
                  output_names=[f'output{n}' for n in range(len(out))],
                  )

# %%
net_trt = torch2trt(net, [x], max_workspace_size=1 << 25)
