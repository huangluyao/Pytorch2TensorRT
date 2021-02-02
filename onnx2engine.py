import os, time
import argparse
import tensorrt as trt
from utils.calibrator import YOLOEntropyCalibrator
MAX_BATCH_SIZE = 1

def use_offices_trtexec():
    trtexec_path = "/home/hly/workspace/library/TensorRT-7.2.1.6/bin/trtexec"
    onnx_file = "weights/yolov4.onnx"
    save_path = "weights/yolov4_fp32.engine"
    workspace =120

    run = "%s --onnx=%s --explicitBatch --saveEngine=%s --workspace=%s " %(trtexec_path,
                                                                                 onnx_file,
                                                                                 save_path,
                                                                                 workspace)
    print(os.system(run))

    print("create engine file done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '--width', type=int, default=416,
        help='the width of image')
    parser.add_argument(
        '--height', type=int, default=416,
        help='the height of image')
    parser.add_argument(
        '-m', '--model', type=str, default="weights/yolov4.onnx",
        help=('.onnx file'))
    parser.add_argument(
        '--int8', action='store_true',
        help='build INT8 TensorRT engine')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging)')

    args = parser.parse_args()
    return args


def set_net_batch(network, batch_size):
    """Set network input batch size.

    The ONNX file might have been generated with a different batch size,
    say, 64.
    """
    if trt.__version__[0] >= '7':
        shape = list(network.get_input(0).shape)
        shape[0] = batch_size
        network.get_input(0).shape = shape
    return network


def build_engine(model_name, width, height, do_int8, verbose=False):
    with open(model_name, 'rb') as f:
        onnx_data = f.read()

    """1. Create logger"""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()

    """2. Specify that the network should be created with an explicit batch dimension."""
    EXPLICIT_BATCH = [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]

    """3. Create Builder & network & parser"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        if do_int8 and not builder.platform_has_fast_int8:
            raise RuntimeError('INT8 not supported on this platform')

        if not parser.parse(onnx_data):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

        network = set_net_batch(network, MAX_BATCH_SIZE)

        builder.max_batch_size = MAX_BATCH_SIZE

        """Create a builder configuration object."""
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30                 # The maximum GPU temporary memory which the engine can use at execution time.
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)       # Enable layers marked to execute on GPU if layer cannot execute on DLA
        config.set_flag(trt.BuilderFlag.FP16)               # Enable FP16 layer selection

        """Create a new optimization profile."""
        profile = builder.create_optimization_profile()
        profile.set_shape(
            'input',  # input tensor name
            (MAX_BATCH_SIZE, 3, height, width),  # min shape
            (MAX_BATCH_SIZE, 3, height, width),  # opt shape
            (MAX_BATCH_SIZE, 3, height, width))  # max shape
        config.add_optimization_profile(profile)

        if do_int8:
            print("YOLOEntropyCalibrator to use int8 ...")
            config.set_flag(trt.BuilderFlag.INT8)
            calib_name = 'calib_%s.bin'%model_name.split('.')[0]
            config.int8_calibrator = YOLOEntropyCalibrator(
                'calib_images', (height,width), calib_name)

            config.set_calibration_profile(profile)

        engine = builder.build_engine(network, config)

        if engine is not None:
            print('Completed creating engine.')
        return engine


if __name__=="__main__":

    args = parse_args()

    engine = build_engine(
        args.model, args.width, args.height, args.int8, args.verbose)

    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    engine_path = '%s.trt' % args.model.split('.')[0]

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    print("Down")