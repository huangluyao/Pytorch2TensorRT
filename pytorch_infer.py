from models.yolov4 import Yolov4
import cv2
import torch
from utils.tools import load_class_names, plot_boxes_cv2
from utils.detect_tools import *

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    width = 416
    height = 416
    namesfile = 'data/coco.names'

    model = Yolov4(inference=True).to(device)
    static_dict = torch.load("weights/yolov4.pth")
    model.load_state_dict(static_dict)

    cap = cv2.VideoCapture(0)
    ret, image = cap.read()

    class_names = load_class_names(namesfile)

    for i in range(1000):
        ret, img = cap.read()
        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        boxes = pytorch_detect(model, sized, 0.4, 0.6, device)
        image = plot_boxes_cv2(img, boxes[0], class_names=class_names)
        cv2.imshow("result", image)
        cv2.waitKey(10)

    print(np.mean(np.array(infer_times[1:])))
