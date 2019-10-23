import time
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
from models.decode import multi_pose_decode
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


class PostProcess(nn.Module):
    def forward(self, x):
        output = {
            'hm': x[0].sigmoid_(),
            'wh': x[1],
            'hps': x[2],
            'reg': x[3],
            'hm_hp': x[4].sigmoid_(),
            'hp_offset': x[5],
        }
        return list(output.values())


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
            # output = self.model(images)[-1]

            output_tmp = self.model(images)
            output = {
                'hm': output_tmp[0],
                'wh': output_tmp[1],
                'hps': output_tmp[2],
                'reg': output_tmp[3],
                'hm_hp': output_tmp[4],
                'hp_offset': output_tmp[5],
            }

            # output['hm'] = output['hm'].sigmoid_()
            # output['hm_hp'] = output['hm_hp'].sigmoid_()
            reg = output['reg']
            hm_hp = output['hm_hp']
            hp_offset = output['hp_offset']
            torch.cuda.synchronize()
            forward_time = time.time()

            dets = multi_pose_decode(
                output['hm'], output['wh'], output['hps'],
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
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
