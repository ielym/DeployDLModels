import torch
import torchvision.models as models

img = torch.randn(1, 3, 224, 224, requires_grad=True)
model = models.resnet18(pretrained=True)
model.eval()

torch.onnx.export(model,               # model being run
                  img,                         # model input (or a tuple for multiple inputs)
                  "resnet18.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['inputtttt'],   # the model's input names
                  output_names = ['outputtttt'], # the model's output names
                  # dynamic_axes={
                  #     'inputtttt': {0: 'batch_size'},  # variable length axes
                  #     'outputtttt': {0: 'batch_size'},  # variable length axes
                  # }
                  )