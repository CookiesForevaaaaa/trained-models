import tensorrt as trt
import jetson_utils  # Updated import as per the deprecation warning
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA driver
import numpy as np
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB workspace size
        
        # Parse the ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('Failed to parse ONNX model')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Build the engine
        engine = builder.build_engine(network, config)
        
        if engine:
            # Serialize and save the engine to a file
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())
            print(f"Engine saved to: {engine_file_path}")
        else:
            print("Failed to build the engine.")
        
        return engine

# Paths
import os
onnx_model_path = os.path.expanduser('~/magisterska/jetson-inference/python/training/detection/ssd/models/fruit/ssd-mobilenet.onnx')
engine_file_path = os.path.expanduser('~/magisterska/jetson-inference/python/training/detection/ssd/models/fruit/ssd-mobilenet.trt')

# Build and save the engine
engine = build_engine(onnx_model_path, engine_file_path)

