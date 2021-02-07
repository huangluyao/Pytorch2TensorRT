from augmentation.pipelines.transforms import build_transform, BboxParams
from augmentation.third_party.albumentations.augmentations.bbox_utils import BboxProcessor

# RandomCrop 5


class Compose(object):
    """Compose transforms and handle all transformations regrading bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
    """

    def __init__(self, transforms, bbox_params=None, additional_targets=None):
        self.transforms = []
        transforms = transforms["pipeline"]

        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_transform(transform)
                if transform != None:
                    self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

        self.processors = {}
        if bbox_params:
            if isinstance(bbox_params, dict):
                params = BboxParams(**bbox_params)
            else:
                raise ValueError("unknown format of bbox_params, please use `dict`")
            self.processors["bboxes"] = BboxProcessor(params, additional_targets)

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

    def add_targets(self, additional_targets):
        if additional_targets:
            for t in self.transforms:
                t.add_targets(additional_targets)


    def __call__(self, data):
        type = data["type"]
        for t in self.transforms:
            if type == "segmentation" and t is not None:
                 data = t(**data)
            elif type == "classification" and t is not None:
                data = t(image=data['image'])
            elif type == "object_detection" and t is not None:
                for p in self.processors.values():
                    p.preprocess(data)
                data = t(**data)
                for p in self.processors.values():
                    p.postprocess(data)
            else:
                return None
            if data is None:
                return None
        return data



