import torch
from torch2trt import torch2trt

# noinspection PyUnresolvedReferences
import _init_paths
from models.networks import efficientnet_centernet, msra_resnet

# %%
heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
# net = efficientnet_centernet.get_pose_net(0, heads, head_conv=64).eval().cuda()
net = msra_resnet.get_pose_net(18, heads, head_conv=64).eval().cuda()
x = torch.ones((1, 3, 512, 512)).cuda()
# print(net(x).shape)
net_trt = torch2trt(net, [x], max_workspace_size=1 << 25)

# %%
out_tr = net_trt(torch.ones((2, 3, 512, 512)).cuda())
# print(out_tr.shape)
for o in out_tr:
    print(o.shape)

# %%
out_pth = net.cuda()(torch.ones((2, 3, 512, 512)).cuda())
# print(out_pth.shape)
for o in out_pth:
    print(o.shape)
