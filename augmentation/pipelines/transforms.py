from augmentation.third_party import albumentations as albu
import cv2


class BuildTransforms():
    def __init__(self):
        self.TRANSFORMS = {}

    def register(self, name, cls):
        self.TRANSFORMS[name] = cls

    def __call__(self, *args, **kwargs):
        if "type" in args[0].keys():
            type = args[0]["type"]
            cls = self.TRANSFORMS[type]
            return cls(*args)


# 创建全局类对象
build_transform = BuildTransforms()
# 注册class
from augmentation.pipelines import augmentations

BboxParams = albu.BboxParams
"""
Parameters of bounding boxes
Args:
    format (str): format of bounding boxes. Should be 'coco', 'pascal_voc', 'albumentations' or 'yolo'.

        The `coco` format
            `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
        The `pascal_voc` format
            `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
        The `albumentations` format
            is like `pascal_voc`, but normalized,
            in other words: [x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
        The `yolo` format
            `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
            `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
    label_fields (list): list of fields that are joined with boxes, e.g labels.
        Should be same type as boxes.
    min_area (float): minimum area of a bounding box. All bounding boxes whose
        visible area in pixels is less than this value will be removed. Default: 0.0.
    min_visibility (float): minimum fraction of area for a bounding box
        to remain this box in list. Default: 0.0.
    check_each_transform (bool): if `True`, then bboxes will be checked after each dual transform.
        Default: `True`
"""
