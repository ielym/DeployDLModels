import tensorrt as trt
import torch
import cv2
import numpy as np

class TensorRTModel():
    def __init__(self, engine=None, input_names=None, output_names=None):
        '''
        :param engine:
        :param input_names: 可以是None
        :param output_names: 可以是None
        '''
        self.engine = engine
        assert self.engine, "engine cannot be None, you can deserialize an engine using runtime.deserialize_cuda_engine()"

        if self.engine is not None:
            self.context = self.engine.create_execution_context()

        if input_names == None or output_names == None:
            input_names = []
            output_names = []
            for idx in range(engine.num_bindings):
                name = engine.get_binding_name(idx)
                op_type = engine.get_binding_dtype(idx)
                shape = engine.get_binding_shape(idx)

                is_input = engine.binding_is_input(idx)

                if is_input:
                    input_names.append(name)
                else:
                    output_names.append(name)
        self.input_names = input_names
        self.output_names = output_names

    def torch_info_from_trt(self, bind_idx):
        trt_shape = self.context.get_binding_shape(bind_idx)
        torch_shape = tuple(trt_shape)

        trt_device = self.engine.get_location(bind_idx)
        if trt_device == trt.TensorLocation.DEVICE:
            torch_device = torch.device("cuda")
        elif trt_device == trt.TensorLocation.HOST:
            torch_device = torch.device("cpu")
        else:
            raise TypeError(f"Unknow trt_device : {trt_device}")

        trt_dtype = self.engine.get_binding_dtype(bind_idx)
        if trt_dtype == trt.int8:
            torch_dtype = torch.int8
        elif trt.__version__ >= '7.0' and trt_dtype == trt.bool:
            torch_dtype = torch.bool
        elif trt_dtype == trt.int32:
            torch_dtype = torch.int32
        elif trt_dtype == trt.float16:
            torch_dtype = torch.float16
        elif trt_dtype == trt.float32:
            torch_dtype = torch.float32
        else:
            raise TypeError(f"Unknow trt_dtype : {trt_dtype}")

        return torch_shape, torch_device, torch_dtype

    def __call__(self, *inputs):

        buffers = [None] * (len(self.input_names) + len(self.output_names))

        for input_idx, input_name in enumerate(self.input_names):
            bind_idx = self.engine.get_binding_index(input_name)
            self.context.set_binding_shape(bind_idx, tuple(inputs[input_idx].shape))
            buffers[bind_idx] = inputs[input_idx].contiguous().data_ptr()

        # output buffers
        outputs = [None] * len(self.output_names)
        for output_idx, output_name in enumerate(self.output_names):
            bind_idx = self.engine.get_binding_index(output_name)
            torch_shape, torch_device, torch_dtype = self.torch_info_from_trt(bind_idx)
            output = torch.empty(size=torch_shape, dtype=torch_dtype, device=torch_device)

            outputs[output_idx] = output
            buffers[bind_idx] = output.data_ptr()

        stream = torch.cuda.current_stream()
        self.context.execute_async_v2(buffers, stream.cuda_stream)
        stream.synchronize()

        return outputs

def build_runtime(serialized_engine=None, serialized_engine_path=''):
    '''
    serialized_engine 和 serialized_engine_path 只需要传一个即可。如果两个都传，用 serialized_engine
    :param serialized_engine: 是一个buffer, 如：serialized_engine = builder.build_serialized_network(network, config)
    :param serialized_engine_path: 是 .trt 或 .engine 的文件
    :return:
    '''
    logger = trt.Logger(trt.Logger.WARNING)

    runtime = trt.Runtime(logger)

    if serialized_engine:
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    elif serialized_engine_path:
        with open(serialized_engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        raise Exception("You must define one of serialized_engine or serialized_engine_path")

    return engine

img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
img = img[None, ...].astype(np.float32) / 255.
img = torch.from_numpy(img).cuda()

engine = build_runtime(serialized_engine_path="resnet18.engine")
model = TensorRTModel(engine=engine)
outputs = model(img)
output = outputs[0]
print(torch.argmax(output[0]))