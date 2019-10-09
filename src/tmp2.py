from pathlib import Path

import torch
from torch2trt import torch2trt
from torchvision.models import resnet18

from models.networks import mobilenet_dcn, msra_resnet

# %%

net = resnet18().eval().cuda()
x = torch.ones((1, 3, 224, 224)).cuda()

# %%
model_trt = torch2trt(net, [x])

# %%
model_trt(x)

# %%
heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}

# %%
net = mobilenet_dcn.get_pose_net(0, heads, head_conv=64).eval().cuda()
x = torch.ones((1, 3, 512, 512)).cuda()
net_trt = torch2trt(net, [x], max_workspace_size=1 << 25)

# %%
torch.save(net.state_dict(), Path.home() / 'tmp' / 'mobilenet_dcn_no-head_torch.pth')
torch.save(net_trt.state_dict(), Path.home() / 'tmp' / 'mobilenet_dcn_no-head_trt.pth')

# %%
net = msra_resnet.get_pose_net(18, heads, head_conv=64).eval().cuda()
x = torch.ones((1, 3, 512, 512)).cuda()
net_trt = torch2trt(net, [x], max_workspace_size=1 << 25)

# %%
torch.save(net.state_dict(), Path.home() / 'tmp' / 'res18_no-head_torch.pth')
torch.save(net_trt.state_dict(), Path.home() / 'tmp' / 'res18_no-head_trt.pth')
