from pathlib import Path

import torch
from torch2trt import torch2trt

# noinspection PyUnresolvedReferences
import _init_paths
from models.model import load_model
from models.networks import efficientnet_centernet, msra_resnet, mobilenet_dcn

# %%
heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}

# %%
# net = mobilenet_dcn.get_pose_net(0, heads, head_conv=64).eval().cuda()
# x = torch.ones((1, 3, 512, 512)).cuda()
# net_trt = torch2trt(net, [x], max_workspace_size=1 << 25)

# %%
# torch.save(net.state_dict(), Path.home() / 'tmp' / 'mobilenet_dcn_no-head_torch.pth')
# torch.save(net_trt.state_dict(), Path.home() / 'tmp' / 'mobilenet_dcn_no-head_trt.pth')
#
# net = net.half()
# x = x.half()
# net_trt_half = torch2trt(net, [x], max_workspace_size=1 << 25, fp16_mode=True)
#
# torch.save(net_trt_half.state_dict(), Path.home() / 'tmp' / 'mobilenet_dcn_no-head_trt_half.pth')

# %%
net = msra_resnet.get_pose_net(18, heads, head_conv=64).eval().cuda()
# net = load_model(net, Path.home() / 'data' / 'model_best.pth')
# net = load_model(net, Path.home() / 'tmp' / 'res18_torch_best.pth')
x = torch.ones((1, 3, 512, 512)).cuda()
# net_trt = torch2trt(net, [x], max_workspace_size=1 << 25)

# %%
# torch.save(net.state_dict(), Path.home() / 'tmp' / 'res18_no-head_torch.pth')
# torch.save(net_trt.state_dict(), Path.home() / 'tmp' / 'res18_no-head_trt.pth')

net = net.half()
x = x.half()
net_trt_half = torch2trt(net, [x], max_workspace_size=1 << 25, fp16_mode=True)

torch.save(net_trt_half.state_dict(), Path.home() / 'tmp' / 'res18_no-head_trt_half.pth')

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
