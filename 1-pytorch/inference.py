import torch
import cv2
import torchvision.models as models


img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
img = torch.from_numpy(img).unsqueeze(0).float() / 255.

model = models.resnet18(pretrained=True)
model.eval()
with torch.no_grad():
    out = model(img)
    print(torch.argmax(out)) # 574