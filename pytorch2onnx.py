import torch
from models import Yolov4
import cv2
import onnxruntime


device = "cuda" if torch.cuda.is_available() else "cpu"


def transform_to_onnx(weight_file, onnx_file_name, batch_size, n_classes, input_h, input_w):
    model = Yolov4(n_classes = n_classes, inference=True).to(device)
    pretrained_dict = torch.load(weight_file)
    model.load_state_dict(pretrained_dict)

    input_names = ["input"]
    output_names = ["boxes", "confs"]
    x = torch.randn(batch_size, 3, input_h, input_w).to(device)
    print('Export the onnx model ...')
    torch.onnx.export(model, x,
                      onnx_file_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=None)
    print('Onnx model exporting done')


if __name__ == "__main__":
    weight_file = "weights/yolov4.pth"
    image_path = "./data/dog.jpg"
    onnx_file_name = "weights/yolov4.onnx"
    batch_size = 1
    n_classes = 80
    IN_IMAGE_H = 416
    IN_IMAGE_W = 416

    transform_to_onnx(weight_file, onnx_file_name, 1, 80, 416, 416)

    session = onnxruntime.InferenceSession(onnx_file_name)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

