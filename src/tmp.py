import cv2
import numpy as np
import torch
import urllib
import torch.nn as nn

# %%
url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

# %%
img = cv2.imread(filename)
h, w, _ = img.shape
img.shape

# %%
img = cv2.resize(img, (224, 224))
# img = cv2.resize(img, (256 * w // h, 256))
# h, w, _ = img.shape
img.shape

# %%
# offset_h = (h - 224) // 2
# offset_w = (w - 224) // 2
# img = img[offset_h:offset_h + 224, offset_w:offset_w + 224, :]


# cv2.imwrite('/tmp/tmp.jpg', img)

# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
        # self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).cuda()

    def forward(self, x):
        # Channel first
        # x = x.permute((0, 3, 1, 2))
        # BGR to RGB
        # x = x[:, [2, 1, 0]]
        # Standarize
        # x = x / 255.
        # x = (x - self.mean) / self.std
        x = self.model(x)

        return x


model = Net()
model.eval()

# %%
batch = np.expand_dims(img, axis=0).astype(np.float32)
batch = torch.from_numpy(batch)

# %%
if torch.cuda.is_available():
    batch = batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(batch)

# %%
cls_id = torch.nn.functional.softmax(output[0], dim=0).argmax()
prob = torch.nn.functional.softmax(output[0], dim=0).max()
print(cls_id, prob)

# %%
# dummy_input = torch.randn(1, 224, 224, 3, device='cuda')
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
torch.onnx.export(model, dummy_input, '/tmp/resnet_with_prepro.onnx', input_names=['input0'],
                  output_names=['output0'], verbose=True)
