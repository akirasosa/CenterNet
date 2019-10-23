from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch2trt import torch2trt

# noinspection PyUnresolvedReferences
import _init_paths
from bench import bench
from models.model import load_model
from models.networks import mobilenetv2_centernet


# %%
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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_process = PreProcess()

        heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
        pose_net = mobilenetv2_centernet.get_pose_net(0, heads, head_conv=64)
        pose_net = load_model(pose_net, Path.home() / 'data' / 'model_best.pth')
        self.pose_net = pose_net

        # self.pose_net = mobilenetv3_centernet.get_pose_net(0, heads, head_conv=64)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.pose_net(x)

        return x


# %%
net = Net().eval().cuda()
x = torch.ones((1, 512, 512, 3)).cuda()
# net_trt = torch2trt(net, [x], max_workspace_size=1 << 25)

# %%
torch.onnx.export(net, x, Path.home() / 'tmp' / 'mobilenetv2_centernet_prepro_trained.onnx',
                  # verbose=True,
                  input_names=['input0'],
                  output_names=[f'output{n}' for n in range(6)],
                  )
# torch.save(net.state_dict(), Path.home() / 'tmp' / 'mobilenet_dcn_no-head_torch.pth')
# torch.save(net_trt.state_dict(), Path.home() / 'tmp' / 'mobilenetv2_centernet_prepro_trt.pth')

# %%
img = cv2.imread('/tmp/tmp.jpg')
img = torch.from_numpy(img.astype(np.float32)).cuda()
img_batch = img.unsqueeze(0)
with torch.no_grad():
    out = net(img_batch)

# %%
out[0].reshape(-1)[:8]

# %%
times = bench(net_trt, n_repeat=100, shape=(1, 512, 512, 3))[5:]
# %%
print(times.mean() * 1000, times.std() * 1000)

# %%
# net = msra_resnet.get_pose_net(18, heads, head_conv=64).eval().cuda()
# net = load_model(net, Path.home() / 'data' / 'model_best.pth')
# net = load_model(net, Path.home() / 'tmp' / 'res18_torch_best.pth')
# x = torch.ones((1, 3, 512, 512)).cuda()
# net_trt = torch2trt(net, [x], max_workspace_size=1 << 25)

# %%
# torch.onnx.export(net, x, Path.home() / 'tmp' / 'resnet_centernet.onnx')
# torch.save(net.state_dict(), Path.home() / 'tmp' / 'res18_no-head_torch.pth')
# torch.save(net_trt.state_dict(), Path.home() / 'tmp' / 'res18_no-head_trt.pth')

# net = net.half()
# x = x.half()
# net_trt_half = torch2trt(net, [x], max_workspace_size=1 << 25, fp16_mode=True)
#
# torch.save(net_trt_half.state_dict(), Path.home() / 'tmp' / 'res18_no-head_trt_half.pth')

# %%
# net = efficientnet_centernet.get_pose_net(0, heads, head_conv=64)
# net = load_model(net, Path.home() / 'data' / 'model_best.pth')
# net = net.eval().cuda()
# x = torch.ones((1, 3, 512, 512)).cuda()
# net_trt = torch2trt(net, [x], max_workspace_size=1 << 25)

# %%
# torch.save(net.state_dict(), Path.home() / 'tmp' / 'efficient_no-head_torch.pth')
# torch.save(net_trt.state_dict(), Path.home() / 'tmp' / 'efficient_no-head_trt.pth')

# %%
# out_tr = net_trt(torch.ones((2, 3, 512, 512)).cuda())
# for o in out_tr:
#     print(o.shape)

# # %%
# out_pth = net(torch.ones((2, 3, 512, 512)).cuda())
# for o in out_pth:
#     print(o.shape)
