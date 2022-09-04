import tensorrt as trt

# 创建记录器 Logger
logger = trt.Logger(trt.Logger.WARNING)

# 创建构造器 Builder
builder = trt.Builder(logger)
# 创建网络定义
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# 创建 ONNX 解析器
parser = trt.OnnxParser(network, logger)
# 读取 ONNX 模型文件，并处理可能出现的错误
parser.parse_from_file('./resnet18.onnx')

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 22)
serialized_engine = builder.build_serialized_network(network, config)
with open("resnet18.engine", "wb") as f:
    f.write(serialized_engine)