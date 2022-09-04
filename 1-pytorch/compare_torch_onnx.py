import torch
import torchvision.models as models
import onnxruntime
import numpy as np
import cv2

img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
img = img[None, ...].astype(np.float32) / 255.


torch_model = models.resnet18(pretrained=True)
torch_model.eval()
with torch.no_grad():
    result_torch = torch_model(torch.from_numpy(img))

session = onnxruntime.InferenceSession('resnet18.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
io_binding = session.io_binding()
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([1, 1000], np.float32, 'cuda', 0)
io_binding.bind_output(name='output', device_type=Y_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=Y_ortvalue.shape(), buffer_ptr=Y_ortvalue.data_ptr())
session.run_with_iobinding(io_binding)
result_onnx = io_binding.copy_outputs_to_cpu()[0]

print(result_torch.cpu().numpy())
print('||' * 50)
print(result_onnx)