from pathlib import Path

import torch
import torch.nn as nn
from torch2trt import torch2trt

# noinspection PyUnresolvedReferences
import _init_paths
from bench import bench
from models.model import load_model
from models.networks import efficientnet_centernet, msra_resnet, mobilenetv3_centernet, mobilenetv2_centernet


# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_net = mobilenetv2_centernet.get_pose_net(0, heads, head_conv=64)
        # self.pose_net = mobilenetv3_centernet.get_pose_net(0, heads, head_conv=64)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).cuda()

    def forward(self, x):
        # Channel first
        x = x.permute(0, 3, 1, 2)
        # BGR to RGB
        x = x[:, [2, 1, 0]]
        # Standarize
        x = x / 255.
        x = (x - self.mean) / self.std
        x = self.pose_net(x)

        return x


# %%
heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}

# %%
# net = mobilenet_centernet.get_pose_net(0, heads, head_conv=64).eval().cuda()
net = Net().eval().cuda()
# x = torch.ones((1, 3, 512, 512)).cuda()
x = torch.ones((1, 512, 512, 3)).cuda()
net_trt = torch2trt(net, [x], max_workspace_size=1 << 25)

# %%
torch.onnx.export(net, x, Path.home() / 'tmp' / 'mobilenetv2_centernet_prepro.onnx',
                  # verbose=True,
                  input_names=['input0'],
                  output_names=[f'output{n}' for n in range(6)],
                  )
# torch.save(net.state_dict(), Path.home() / 'tmp' / 'mobilenet_dcn_no-head_torch.pth')
torch.save(net_trt.state_dict(), Path.home() / 'tmp' / 'mobilenetv2_centernet_prepro_trt.pth')
#
# net = net.half()
# x = x.half()
# net_trt_half = torch2trt(net, [x], max_workspace_size=1 << 25, fp16_mode=True)
#
# torch.save(net_trt_half.state_dict(), Path.home() / 'tmp' / 'mobilenet_dcn_no-head_trt_half.pth')

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
