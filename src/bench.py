from pathlib import Path
from time import time
import numpy as np

# noinspection PyUnresolvedReferences
import _init_paths
import torch
# from torch2trt import TRTModule

from models.networks import msra_resnet, mobilenet_dcn


def load_res18_trt():
    dict_path = Path.home() / 'tmp' / 'res18_no-head_trt.pth'

    model = TRTModule()
    model.load_state_dict(torch.load(dict_path))
    model = model.eval().cuda()
    return model


def load_mobilev3_trt():
    dict_path = Path.home() / 'tmp' / 'mobilenet_dcn_no-head_trt.pth'

    model = TRTModule()
    model.load_state_dict(torch.load(dict_path))
    model = model.eval().cuda()
    return model


def load_res18_pth():
    dict_path = Path.home() / 'tmp' / 'res18_no-head_torch.pth'
    heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}

    model = msra_resnet.get_pose_net(18, heads, 64)
    model.load_state_dict(torch.load(dict_path))
    model = model.eval().cuda()
    return model


def load_mobilev3_pth():
    dict_path = Path.home() / 'tmp' / 'mobilenet_dcn_no-head_torch.pth'
    heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}

    model = mobilenet_dcn.get_pose_net(0, heads, 64)
    model.load_state_dict(torch.load(dict_path))
    model = model.eval().cuda()
    return model


def bench(model, n_repeat=10):
    inputs = torch.ones((1, 3, 512, 512)).cuda()
    results = []

    for n in range(n_repeat):
        start = time()
        model(inputs)
        results.append(time() - start)

    return np.array(results)


if __name__ == '__main__':
    # model = load_res18_trt()
    # results = bench(model, 20)
    # print(results[2:].mean())

    # model = load_mobilev3_trt()
    # results = bench(model, 20)
    # print(results[2:].mean())

    model = load_res18_pth()
    results = bench(model, 20)
    print(results[2:].mean())

    model = load_mobilev3_pth()
    results = bench(model, 20)
    print(results[2:].mean())
