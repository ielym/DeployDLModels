from onnx import load_model, save_model
from onnxmltools.utils import float16_converter

onnx_model = load_model('resnet18.onnx')
trans_model = float16_converter.convert_float_to_float16(onnx_model,keep_io_types=True)
save_model(trans_model, "resnet18_fp16.onnx")