# tensorrt-lib

import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import common
from calibrator import Calibrator
from torch.autograd import Variable
import torch
import numpy as np
import time

# add verbose
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


# create tensorrt-engine
# fixed and dynamic
def build_engine(onnx_file_path, engine_file_path, input_shape, **kwargs):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        config.max_workspace_size = 1 << 28  # 256MiB
        builder.max_batch_size = 1
        int8_mode = kwargs['int8_mode']
        if int8_mode:
            # calibrator = kwargs['calibrator_stream']
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = Calibrator(kwargs["calibrator_stream"], kwargs["cache_file"])
            print('Int8 mode enabled')

        # Parse model file
        if not os.path.exists(onnx_file_path):
            print(
                "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
            )
            exit(0)
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
        network.get_input(0).shape = input_shape
        print("Completed parsing of ONNX file")
        print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine


def get_engine(onnx_file_path, input_shape, engine_file_path, **kwargs):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    return build_engine(onnx_file_path, engine_file_path, input_shape, **kwargs)
