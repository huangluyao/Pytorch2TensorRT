import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.tools import load_class_names, plot_boxes_cv2
from utils.detect_tools import *
import cv2
import numpy as np



TRT_LOGGER = trt.Logger()


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)

        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def gen_engine(engine_path):
    print("Reading engine form file %s ..." % (engine_path))
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

@time_warp
def trt_infer(context, bindings, stream):
    context.execute_async(bindings=bindings, stream_handle=stream.handle)

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    # context.execute_async(bindings=bindings, stream_handle=stream.handle)
    trt_infer(context, bindings, stream)

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def trt_detect(context, buffers, img, img_size, num_classes):
    input_h, input_w = img_size
    resized = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    inputs, outputs, bindings, stream = buffers
    inputs[0].host = img_in
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)

    boxes = post_processing( 0.4, 0.6, trt_outputs)
    return boxes


if __name__ == "__main__":
    engine_path = "weights/yolov4.trt"
    image_path = "./data/dog.jpg"
    image_size = (416, 416)
    num_classes = 80
    namesfile = 'data/coco.names'
    class_names = load_class_names(namesfile)

    with gen_engine(engine_path) as engine, engine.create_execution_context() as context:
        buffers = allocate_buffers(engine, 1)
        input_h , input_w = image_size
        context.set_binding_shape(0, (1, 3, input_h, input_w))

        cap = cv2.VideoCapture(0)
        ret = cap.isOpened()
        for i in range(1000):
            ret, image = cap.read()
            boxes = trt_detect(context, buffers, image, image_size, num_classes)
            image = plot_boxes_cv2(image, boxes[0], class_names=class_names)
            cv2.imshow("src", image)
            cv2.waitKey(10)

        print("avg_inger_time:", np.mean(np.array(infer_times[1:])))
