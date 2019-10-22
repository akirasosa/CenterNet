import time
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

import cv2
import numpy as np

# noinspection PyUnresolvedReferences
import _init_paths
from demo import time_stats
from detectors.base_detector import BaseDetector
from detectors.multi_pose import MultiPoseDetector

# %%
img = cv2.imread('/home/akirasosa/.ghq/github.com/akirasosa/CenterNet/images/16004479832_a748d55f21_k.jpg')
# img = cv2.imread('../images/16004479832_a748d55f21_k.jpg')
img.shape


# %%
@dataclass
class Option:
    # load_model: str = Path.home() / 'tmp' / 'res18_no-head_trt.pth'
    load_model: str = '/tmp'
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


# %%
opt = Option()
det = MultiPoseDetector(opt)

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
