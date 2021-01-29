import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.tools import load_class_names, plot_boxes_cv2
from utils.detect_tools import post_processing, time_warp
import cv2
import numpy as np
from utils.detect_tools import pytorch_detect
from models import Yolov4
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
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


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def trt_detect(context, buffers, img, img_size, num_classes):
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

def get_trt_output(context, buffers, img, img_size, num_classes):

    inputs, outputs, bindings, stream = buffers
    inputs[0].host = img
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)
    return trt_outputs


if __name__ == "__main__":
    engine_path = "weights/yolov4_fp32.engine"
    image_path = "./data/dog.jpg"
    image_size = (416, 416)
    num_classes = 80
    namesfile = 'data/coco.names'
    class_names = load_class_names(namesfile)

    model = Yolov4(inference=True).to(device).eval()
    static_dict = torch.load("weights/yolov4.pth")
    model.load_state_dict(static_dict)

    with gen_engine(engine_path) as engine, engine.create_execution_context() as context:
        buffers = allocate_buffers(engine, 1)
        input_h , input_w = image_size
        context.set_binding_shape(0, (1, 3, input_h, input_w))

        cap = cv2.VideoCapture(0)
        ret = cap.isOpened()

        boxes_max_errors = []
        conf_max_errors = []
        top_100_errors = []
        for i in range(1000):
            ret, image = cap.read()
            resized = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
            img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
            img_in = np.expand_dims(img_in, axis=0)
            img_in /= 255.0
            img_in = np.ascontiguousarray(img_in)

            image_tensor = img_in.copy()
            image_tensor = torch.from_numpy(image_tensor).to(device)
            torch_output = model(image_tensor)
            trt_output = get_trt_output(context, buffers, img_in, image_size, num_classes)

            torch_output[0] = torch_output[0].cpu().detach().numpy()
            torch_output[1] = torch_output[1].cpu().detach().numpy()

            torch_id = torch_output[1].argmax(-1)
            trt_id = trt_output[1].argmax(-1)
            torch_top_100 = torch_output[1].max(-1)
            trt_top_100 = trt_output[1].max(-1)
            torch_top_100 = np.sort(torch_top_100)[0,-100:]
            trt_top_100 = np.sort(trt_top_100)[0,-100:]

            top_100_error = np.max(np.abs(torch_top_100 - trt_top_100))
            all_boxes_max_error = np.max(np.abs(torch_output[0]-trt_output[0]))
            all_conf_max_error = np.max(np.abs(torch_output[1]-trt_output[1]))

            print("top_100_error", top_100_error)
            print("boxes_max_error", all_boxes_max_error)
            print("conf_max_error", all_conf_max_error)

            boxes_max_errors.append(all_boxes_max_error)
            conf_max_errors.append(all_conf_max_error)
            top_100_errors.append(top_100_error)


        avg_box_error = np.mean(np.array(boxes_max_errors))
        avg_conf_error = np.mean(np.array(conf_max_errors))
        avg_top_100_errors = np.mean(np.array(top_100_errors))
        print("avg_box_error" , avg_box_error)
        print("avg_conf_error", avg_conf_error)
        print("avg_top_100_errors", avg_top_100_errors)
