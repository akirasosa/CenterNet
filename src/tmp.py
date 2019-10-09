import numpy as np
import onnx
import onnx_tensorrt.backend as backend
import torch
from torchvision.models import ResNet, resnet18

from models.networks.mobilenet_dcn import get_pose_net

# %%
heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
# net = get_pose_net(0, heads).to('cuda')
net = resnet18().cuda()
# save_model('/tmp/model.pth', 1, net)

x = torch.ones((1, 3, 512, 512)).cuda()

torch.onnx.export(net, x, "/tmp/tmp.onnx", verbose=True)

# %%
model = onnx.load("/tmp/tmp.onnx")
engine = backend.prepare(model, device='CUDA:0')
input_data = np.random.random(size=(1, 3, 512, 512)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
