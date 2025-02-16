from pathlib import Path
from time import time

import numpy as np
import torch
from torch2trt import TRTModule

# noinspection PyUnresolvedReferences
import _init_paths
from models.networks import msra_resnet, mobilenetv3_centernet, efficientnet_centernet


def load_res18_trt(is_half=False):
    if is_half:
        dict_path = Path.home() / 'tmp' / 'res18_no-head_trt_half.pth'
    else:
        dict_path = Path.home() / 'tmp' / 'res18_no-head_trt.pth'

    model = TRTModule()
    model.load_state_dict(torch.load(dict_path))
    model = model.eval().cuda()
    return model


def load_mobilev3_trt(is_half=False):
    if is_half:
        dict_path = Path.home() / 'tmp' / 'mobilenet_centernet_trt_half.pth'
    else:
        dict_path = Path.home() / 'tmp' / 'mobilenet_centernet_trt.pth'

    model = TRTModule()
    model.load_state_dict(torch.load(dict_path))
    model = model.eval().cuda()
    return model


def load_efficient_trt():
    dict_path = Path.home() / 'tmp' / 'efficient_no-head_trt.pth'

    model = TRTModule()
    model.load_state_dict(torch.load(dict_path))
    # model = model.eval().cuda()
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

    model = mobilenetv3_centernet.get_pose_net(0, heads, 64)
    model.load_state_dict(torch.load(dict_path))
    model = model.eval().cuda()
    return model


def load_efficient_pth():
    dict_path = Path.home() / 'tmp' / 'efficient_no-head_torch.pth'
    heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}

    model = efficientnet_centernet.get_pose_net(0, heads, 64)
    model.load_state_dict(torch.load(dict_path))
    model = model.eval().cuda()
    return model


def bench(model, n_repeat=100, shape=(1, 3, 512, 512)):
    inputs = torch.ones(shape).cuda()
    results = []

    with torch.no_grad():
        for n in range(n_repeat):
            start = time()
            model(inputs)
            torch.cuda.synchronize()
            results.append(time() - start)

    return np.array(results)


if __name__ == '__main__':
    # model = load_res18_trt(is_half=True)
    # results = bench(model, 40)
    # print('res18_trt_half', results[2:].mean())

    # model = load_mobilev3_trt(is_half=True)
    # results = bench(model, 40)
    # print('mobilev3_trt_half', results[2:].mean())

    # model = load_res18_trt()
    # results = bench(model, 40)
    # print('res18_trt', results[2:].mean())

    model = load_mobilev3_trt()
    results = bench(model, 40)
    print('mobilev3_trt', results[2:].mean())

    # model = load_efficient_trt()
    # results = bench(model, 20)
    # print(results[2:].mean())
    # print('efficient_trt', '---')

    # model = load_res18_pth()
    # results = bench(model, 40)
    # print('res18_pth', results[2:].mean())

    # model = load_mobilev3_pth()
    # results = bench(model, 40)
    # print('mobilev3_pth', results[2:].mean())

    # model = load_efficient_pth()
    # results = bench(model, 20)
    # print('efficient_pth', results[2:].mean())
