from augmentation.third_party import albumentations as albu
import cv2
from augmentation.pipelines.transforms import build_transform
from augmentation.third_party.WBAugmenter import WBEmulator as wbAug
import random


class Rotate(albu.Rotate):
    """Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """
    def __init__(self, *args):

        limit = int(args[0]["limit"]) if "limit" in args[0].keys() else 90  # 范围为（-distort_limit，distort_limit）
        value = float(args[0]["value"]) if "value" in args[0].keys() else None  # padding值如果采用 cv2.BORDER_CONSTANT.
        mask_value = float(args[0]["mask_value"]) if "mask_value" in args[
            0].keys() else None  # padding值如果掩码采用的是 cv2.BORDER_CONSTANT.
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5

        if "interpolation" in args[0].keys():  # 对指定像素进行内插法
            if args[0]["interpolation"] == "nearest":
                interpolation = cv2.INTER_NEAREST
            elif args[0]["interpolation"] == "linear":
                interpolation = cv2.INTER_LINEAR
            elif args[0]["interpolation"] == "cubic":
                interpolation = cv2.INTER_CUBIC
            elif args[0]["interpolation"] == "area":
                interpolation = cv2.INTER_AREA
            elif args[0]["interpolation"] == "lanczos4":
                interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = 1

        if "border_mode" in args[0].keys():  # 对指定像素的外插法
            if args[0]["border_mode"] == "constant":
                border_mode = cv2.BORDER_CONSTANT
            elif args[0]["border_mode"] == "replicate":
                border_mode = cv2.BORDER_REPLICATE
            elif args[0]["border_mode"] == "reflect":
                border_mode = cv2.BORDER_REFLECT
            elif args[0]["border_mode"] == "wrap":
                border_mode = cv2.BORDER_WRAP
            elif args[0]["border_mode"] == "reflect_101":
                border_mode = cv2.BORDER_REFLECT_101
        else:
            border_mode = 4

        super(Rotate, self).__init__(limit, interpolation, border_mode, value, mask_value, always_apply, p)


class RandomFlip(albu.Flip):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """
    def __init__(self, *args):
        super(RandomFlip, self).__init__(p=args[0]["prob"])


class CenterCrop(albu.CenterCrop):
    """Crop the central part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        It is recommended to use uint8 images as input.
        Otherwise the operation will require internal conversion
        float32 -> uint8 -> float32 that causes worse performance.
    """
    def __init__(self, *args):
        width, height = args[0]["size"]
        p = float(args[0]["prob"] if "prob" in args[0].keys() else 1.)
        super(CenterCrop, self).__init__(height, width, always_apply=False, p=p)


class RandomCrop(albu.RandomCrop):
    """Crop a random part of the input.
    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """
    def __init__(self, *args):
        width, height = args[0]["size"]
        p = float(args[0]["prob"] if "prob" in args[0].keys() else 1.)
        super(RandomCrop, self).__init__(height, width, always_apply=False, p=p)


class CoarseDropout(albu.CoarseDropout):
    """CoarseDropout of the rectangular regions in the image.

    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int): Maximum height of the hole.
        max_width (int): Maximum width of the hole.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
        min_width (int): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, lisf of int, list of float): fill value for dropped pixels
            in mask. If None - mask is not affected.

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """
    def __init__(self, *args):
        max_holes = float(args[0]["max_holes"]) if "max_holes" in args[0].keys() else 8  # 最大遮挡物体的数量
        max_height = float(args[0]["max_height"]) if "max_height" in args[0].keys() else 8  # 遮挡物体的尺寸
        max_width = float(args[0]["max_width"]) if "max_width" in args[0].keys() else 8
        min_holes = float(args[0]["min_holes"]) if "min_holes" in args[0].keys() else None  # 最小遮挡物体的数量
        min_height = float(args[0]["min_height"]) if "min_height" in args[0].keys() else None  # 最小遮挡物体的尺寸
        min_width = float(args[0]["min_width"]) if "min_width" in args[0].keys() else None
        fill_value = float(args[0]["fill_value"]) if "fill_value" in args[0].keys() else 0  # 被遮挡的像素值
        mask_fill_value = float(args[0]["mask_fill_value"]) if "mask_fill_value" in args[
            0].keys() else None  # 掩码是否要被遮挡，要遮挡设置像素值
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"] if "prob" in args[0].keys() else 0.5)
        super(CoarseDropout, self).__init__(max_holes, max_height, max_width, min_holes,
                             min_height, min_width, fill_value, mask_fill_value, always_apply, p)


class GridDistortion(albu.GridDistortion):

    """
Args:
    num_steps (int): count of grid cells on each side.
    distort_limit (float, (float, float)): If distort_limit is a single float, the range
        will be (-distort_limit, distort_limit). Default: (-0.03, 0.03).
    interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
        cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
        Default: cv2.INTER_LINEAR.
    border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
        cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
        Default: cv2.BORDER_REFLECT_101
    value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
    mask_value (int, float,
                list of ints,
                list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.

Targets:
    image, mask

Image types:
    uint8, float32
"""
    def __init__(self, *args):
        num_steps = int(args[0]["num_steps"]) if "num_steps" in args[0].keys() else 5  # 畸变的单元格数
        distort_limit = float(args[0]["distort_limit"]) if "distort_limit" in args[
            0].keys() else 0.3  # 范围为（-distort_limit，distort_limit）
        value = float(args[0]["value"]) if "value" in args[0].keys() else None  # padding值如果采用 cv2.BORDER_CONSTANT.
        mask_value = float(args[0]["mask_value"]) if "mask_value" in args[
            0].keys() else None  # padding值如果掩码采用的是 cv2.BORDER_CONSTANT.
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5

        if "interpolation" in args[0].keys():  # 对指定像素进行内插法
            if args[0]["interpolation"] == "nearest":
                interpolation = cv2.INTER_NEAREST
            elif args[0]["interpolation"] == "linear":
                interpolation = cv2.INTER_LINEAR
            elif args[0]["interpolation"] == "cubic":
                interpolation = cv2.INTER_CUBIC
            elif args[0]["interpolation"] == "area":
                interpolation = cv2.INTER_AREA
            elif args[0]["interpolation"] == "lanczos4":
                interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = 1

        if "border_mode" in args[0].keys():  # 对指定像素的外插法
            if args[0]["border_mode"] == "constant":
                border_mode = cv2.BORDER_CONSTANT
            elif args[0]["border_mode"] == "replicate":
                border_mode = cv2.BORDER_REPLICATE
            elif args[0]["border_mode"] == "reflect":
                border_mode = cv2.BORDER_REFLECT
            elif args[0]["border_mode"] == "wrap":
                border_mode = cv2.BORDER_WRAP
            elif args[0]["border_mode"] == "reflect_101":
                border_mode = cv2.BORDER_REFLECT_101
        else:
            border_mode = 4
        super(GridDistortion, self).__init__(num_steps, distort_limit, interpolation, border_mode, value, mask_value, always_apply, p)


class Resize(albu.Resize):
    """Resize the input to the given height and width.

Args:
    height (int): desired height of the output.
    width (int): desired width of the output.
    interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
        cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
        Default: cv2.INTER_LINEAR.
    p (float): probability of applying the transform. Default: 1.

Targets:
    image, mask, bboxes, keypoints

Image types:
    uint8, float32
"""
    def __init__(self, *args):
        width, height = args[0]["size"]
        super(Resize, self).__init__(width, height)


class ChannelDropout(albu.ChannelDropout):
    """Randomly Drop Channels in the input Image.

Args:
    fill_value (int, float): pixel value for the dropped channel.
    p (float): probability of applying the transform. Default: 0.5.

Targets:
    image

Image types:
    uint8, uint16, unit32, float32
"""
    def __init__(self, *args):
        fill_value = float(args[0]["fill_value"]) if "fill_value" in args[0].keys() else 0  # 删除通道的像素值
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(ChannelDropout, self).__init__(fill_value=fill_value, p=p)


class ChannelShuffle(albu.ChannelShuffle):
    """Randomly rearrange channels of the input RGB image.

Args:
    p (float): probability of applying the transform. Default: 0.5.

Targets:
    image

Image types:
    uint8, float32
"""
    def __init__(self, *args):
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(ChannelShuffle, self).__init__(p=p)


class CLAHE(albu.CLAHE):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

Args:
    clip_limit (float or (float, float)): upper threshold value for contrast limiting.
        If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
    tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
    p (float): probability of applying the transform. Default: 0.5.

Targets:
    image

Image types:
    uint8
"""
    def __init__(self, *args):
        clip_limit = float(args[0]["clip_limit"]) if "clip_limit" in args[0].keys() else 4.0  # 对比度限制的上阈值
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        if "tile_grid_size" in args[0].keys():  # 直方图均衡化的网格大小
            width, height = args[0]["tile_grid_size"]
            tile_grid_size = (width, height)
        else:
            tile_grid_size = (8, 8)

        super(CLAHE, self).__init__(clip_limit, tile_grid_size, always_apply, p)


class ColorJitter(albu.ColorJitter):
    """Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision,
this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in
Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8
overflow, but we use value saturation.

Args:
    brightness (float or tuple of float (min, max)): How much to jitter brightness.
        brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
        or the given [min, max]. Should be non negative numbers.
    contrast (float or tuple of float (min, max)): How much to jitter contrast.
        contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
        or the given [min, max]. Should be non negative numbers.
    saturation (float or tuple of float (min, max)): How much to jitter saturation.
        saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
        or the given [min, max]. Should be non negative numbers.
    hue (float or tuple of float (min, max)): How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
"""
    def __init__(self, *args):
        brightness = float(args[0]["brightness"]) if "brightness" in args[0].keys() else 0.2  # 亮度抖动的范围
        contrast = float(args[0]["contrast"]) if "contrast" in args[0].keys() else 0.2  # 对比度抖动的范围
        saturation = float(args[0]["saturation"]) if "saturation" in args[0].keys() else 0.2  # 饱和度抖动的范围
        hue = float(args[0]["hue"]) if "hue" in args[0].keys() else 0.2  # 色调抖动的范围
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(ColorJitter, self).__init__(brightness, contrast, saturation, hue, always_apply, p)


class CropNonEmptyMaskIfExists(albu.CropNonEmptyMaskIfExists):
    """Crop area with mask if mask is non-empty, else make random crop.

Args:
    height (int): vertical size of crop in pixels
    width (int): horizontal size of crop in pixels
    ignore_values (list of int): values to ignore in mask, `0` values are always ignored
        (e.g. if background value is 5 set `ignore_values=[5]` to ignore)
    ignore_channels (list of int): channels to ignore in mask
        (e.g. if background is a first channel set `ignore_channels=[0]` to ignore)
    p (float): probability of applying the transform. Default: 1.0.

Targets:
    image, mask, bboxes, keypoints

Image types:
    uint8, float32
"""
    def __init__(self, *args):
        height, width = args[0]["size"]  # 长宽
        ignore_values = int(args[0]["ignore_values"]) if "ignore_values" in args[0].keys() else None  # 掩码被忽略的像素值
        ignore_channels = int(args[0]["ignore_channels"]) if "ignore_channels" in args[0].keys() else None  # 掩码被忽略的通道
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 1.0
        super(CropNonEmptyMaskIfExists, self).__init__(height, width ,ignore_values, always_apply, p)


class Downscale(albu.Downscale):
    """Decreases image quality by downscaling and upscaling back.

    Args:
        scale_min (float): lower bound on the image scale. Should be < 1.
        scale_max (float):  lower bound on the image scale. Should be .
        interpolation: cv2 interpolation method. cv2.INTER_NEAREST by default

    Targets:
        image

    Image types:
        uint8, float32
    """
    def __init__(self, *args):
        cale_min = float(args[0]["cale_min"]) if "cale_min" in args[0].keys() else 0.25  # 尺寸应该要小于1
        scale_max = float(args[0]["scale_max"]) if "scale_max" in args[0].keys() else 0.25
        interpolation = float(args[0]["interpolation"]) if "interpolation" in args[0].keys() else 0
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(Downscale, self).__init__(cale_min, scale_max, interpolation, always_apply, p)


class ElasticTransform(albu.ElasticTransform):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Args:
        alpha (float):
        sigma (float): Gaussian filter parameter.
        alpha_affine (float): The range will be (-alpha_affine, alpha_affine)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        approximate (boolean): Whether to smooth displacement map with fixed kernel size.
                               Enabling this option gives ~2X speedup on large images.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """
    def __init__(self, *args):
        alpha = float(args[0]["alpha"]) if "alpha" in args[0].keys() else 1.
        sigma = float(args[0]["sigma"]) if "sigma" in args[0].keys() else 50.  # 高斯滤波参数
        alpha_affine = float(args[0]["alpha_affine"]) if "alpha_affine" in args[
            0].keys() else 50  # 范围是（-alpha_affine，alpha_affine）
        value = float(args[0]["value"]) if "value" in args[0].keys() else None  # padding值如果采用 cv2.BORDER_CONSTANT.
        mask_value = float(args[0]["mask_value"]) if "mask_value" in args[0].keys() else None  # padding值如果掩码采用的是 cv2.BORDER_CONSTANT.
        always_apply = float(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        approximate = float(args[0]["approximate"]) if "approximate" in args[
            0].keys() else False  # 是否通过固定尺寸的内核平滑图像，采用这种方法，大图速度可以提高2倍
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5

        if "interpolation" in args[0].keys():  # 对指定像素进行内插法
            if args[0]["interpolation"] == "nearest":
                interpolation = cv2.INTER_NEAREST
            elif args[0]["interpolation"] == "linear":
                interpolation = cv2.INTER_LINEAR
            elif args[0]["interpolation"] == "cubic":
                interpolation = cv2.INTER_CUBIC
            elif args[0]["interpolation"] == "area":
                interpolation = cv2.INTER_AREA
            elif args[0]["interpolation"] == "lanczos4":
                interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = 1

        if "border_mode" in args[0].keys():  # 对指定像素的外插法
            if args[0]["border_mode"] == "constant":
                border_mode = cv2.BORDER_CONSTANT
            elif args[0]["border_mode"] == "replicate":
                border_mode = cv2.BORDER_REPLICATE
            elif args[0]["border_mode"] == "reflect":
                border_mode = cv2.BORDER_REFLECT
            elif args[0]["border_mode"] == "wrap":
                border_mode = cv2.BORDER_WRAP
            elif args[0]["border_mode"] == "reflect_101":
                border_mode = cv2.BORDER_REFLECT_101
        else:
            border_mode = 4
        super(ElasticTransform, self).__init__(alpha, sigma, alpha_affine, interpolation, border_mode, value, mask_value,
                                     always_apply, approximate, p)


class Equalize(albu.Equalize):
    """Equalize the image histogram.

    Args:
        mode (str): {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
        by_channels (bool): If True, use equalization by channels separately,
            else convert image to YCbCr representation and use equalization by `Y` channel.
        mask (np.ndarray, callable): If given, only the pixels selected by
            the mask are included in the analysis. Maybe 1 channel or 3 channel array or callable.
            Function signature must include `image` argument.
        mask_params (list of str): Params for mask function.

    Targets:
        image

    Image types:
        uint8
    """
    def __init__(self, *args):
        if "mode" in args[0].keys():  # 模型有 OpenCV or Pillow 均衡化
            mode = args[0]["mode"] if args[0]["mode"] == 'pil' else 'cv'
        else:
            mode = 'cv'
        by_channels = bool(args[0]["by_channels"]) if "by_channels" in args[
            0].keys() else True  # 如果为True，则分别使用通道均衡，否则将图像转换为YCbCr表示并使用Y通道均衡。
        mask = float(args[0]["mask"]) if "mask" in args[0].keys() else None
        mask_params = float(args[0]["mask_params"]) if "mask_params" in args[0].keys() else ()
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(Equalize,self).__init__(mode, by_channels, mask, mask_params, always_apply, p)


class FancyPCA(albu.FancyPCA):
    """Augment RGB image using FancyPCA from Krizhevsky's paper
    "ImageNet Classification with Deep Convolutional Neural Networks"

    Args:
        alpha (float):  how much to perturb/scale the eigen vecs and vals.
            scale is samples from gaussian distribution (mu=0, sigma=alpha)

    Targets:
        image

    Image types:
        3-channel uint8 images only

    Credit:
        http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
        https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
    """
    def __init__(self, *args):
        alpha = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.1  # 扰动/缩放本征vecs和val的数量。标度是高斯分布的样本
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(FancyPCA, self).__init__(alpha, always_apply, p)


class FromFloat(albu.FromFloat):
    """Take an input array where all values should lie in the range [0, 1.0], multiply them by `max_value` and then
    cast the resulted value to a type specified by `dtype`. If `max_value` is None the transform will try to infer
    the maximum value for the data type from the `dtype` argument.

    This is the inverse transform for :class:`~albumentations.augmentations.transforms.ToFloat`.

    Args:
        max_value (float): maximum possible input value. Default: None.
        dtype (string or numpy data type): data type of the output. See the `'Data types' page from the NumPy docs`_.
            Default: 'uint16'.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        float32

    .. _'Data types' page from the NumPy docs:
       https://docs.scipy.org/doc/numpy/user/basics.types.html
    """
    def __init__(self, *args):
        max_value = float(args[0]["max_value"]) if "max_value" in args[0].keys() else None  # 扰动的最大值
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 1.0
        super(FromFloat, self).__init__(max_value=max_value,always_apply=always_apply, p=p)


class GaussianBlur(albu.GaussianBlur):
    """Blur the input image using using a Gaussian filter with a random kernel size.

    Args:
        blur_limit (int, (int, int)): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be greater in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """
    def __init__(self, *args):
        blur_limit = args[0]["blur_limit"] if "blur_limit" in args[0].keys() else (3, 7)  # 高斯滤波最大尺寸核
        sigma_limit = args[0]["blur_limit"] if "blur_limit" in args[0].keys() else 0  # 高斯标准偏差
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(GaussianBlur, self).__init__(blur_limit, sigma_limit, always_apply, p)


class GaussNoise(albu.GaussNoise):
    """Apply gaussian noise to the input image.

    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """
    def __init__(self, *args):
        var_limit = float(args[0]["var_limit"]) if "var_limit" in args[0].keys() else (10.0, 50.0)  # 噪声方差范围
        mean = float(args[0]["mean"]) if "mean" in args[0].keys() else 0  # 噪声平均值
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(GaussNoise, self).__init__(var_limit, mean, always_apply, p)


class RandomSizedBBoxSafeCrop(albu.RandomSizedBBoxSafeCrop):
    def __init__(self, *args):
        height, width, = width, height = args[0]["size"]
        erosion_rate = float(args[0]["mean"]) if "mean" in args[0].keys() else 0.0
        always_apply = bool(args[0]["always_apply"]) if "always_apply" in args[0].keys() else False
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 1

        if "interpolation" in args[0].keys():  # 对指定像素进行内插法
            if args[0]["interpolation"] == "nearest":
                interpolation = cv2.INTER_NEAREST
            elif args[0]["interpolation"] == "linear":
                interpolation = cv2.INTER_LINEAR
            elif args[0]["interpolation"] == "cubic":
                interpolation = cv2.INTER_CUBIC
            elif args[0]["interpolation"] == "area":
                interpolation = cv2.INTER_AREA
            elif args[0]["interpolation"] == "lanczos4":
                interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_LINEAR
        super(RandomSizedBBoxSafeCrop, self).__init__(height, width, erosion_rate, interpolation, always_apply, p)


class Normalize(albu.Normalize):
    def __init__(self, *args):
        mean = args[0]["mean"] if "mean" in args[0].keys() else (0.485, 0.456, 0.406)
        std = args[0]["std"] if "std" in args[0].keys() else (0.229, 0.224, 0.225)
        max_pixel_value = args[0]["max_pixel_value"] if "max_pixel_value" in args[0].keys() else 255.
        super(Normalize, self).__init__(mean=mean, std=std, max_pixel_value=max_pixel_value)


class Blur(albu.Blur):
    def __init__(self, *args):
        blur_limit = args[0]["blur_limit"] if "blur_limit" in args[0].keys() else (3,7)
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(Blur, self).__init__(blur_limit=blur_limit, p=p)


class GlassBlur(albu.GlassBlur):
    def __init__(self, *args):
        sigma = float(args[0]["sigma"]) if "sigma" in args[0].keys() else 0.7
        max_delta = int(args[0]["max_delta"]) if "max_delta" in args[0].keys() else 4
        iterations = int(args[0]["max_delta"]) if "max_delta" in args[0].keys() else 2
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(GlassBlur, self).__init__(sigma=sigma, max_delta=max_delta, iterations=iterations, p=p)


class HueSaturationValue(albu.HueSaturationValue):
    def __init__(self, *args):
        hue_shift_limit = int(args[0]["hue_shift_limit"]) if "hue_shift_limit" in args[0].keys() else 20
        sat_shift_limit = int(args[0]["sat_shift_limit"]) if "sat_shift_limit" in args[0].keys() else 30
        val_shift_limit = int(args[0]["val_shift_limit"]) if "val_shift_limit" in args[0].keys() else 20
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(HueSaturationValue,self).__init__(hue_shift_limit=hue_shift_limit, sat_shift_limit=sat_shift_limit, val_shift_limit=val_shift_limit,p=p)


class MultiplicativeNoise(albu.MultiplicativeNoise):
    def __init__(self, *args):
        multiplier = args[0]["multiplier"] if "multiplier" in args[0].keys() else [0.9, 1.1]
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(MultiplicativeNoise, self).__init__(multiplier=multiplier, p=p)


class RandomGridShuffle(albu.RandomGridShuffle):
    def __init__(self, *args):
        grid =  args[0]["grid"] if "grid" in args[0].keys() else [3, 3]
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(RandomGridShuffle, self).__init__(grid=grid, p=p)


class RandomBrightnessContrast(albu.RandomBrightnessContrast):
    def __init__(self, *args):
        brightness_limit = float(args[0]["brightness_limit"]) if "brightness_limit" in args[0].keys() else 0.5
        contrast_limit = float(args[0]["contrast_limit"]) if "contrast_limit" in args[0].keys() else 0.5
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(RandomBrightnessContrast, self).__init__(brightness_limit, contrast_limit, p=p)


class HorizontalFlip(albu.HorizontalFlip):
    def __init__(self, *args):
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(HorizontalFlip, self).__init__(p=p)


class WhiteBalance(wbAug.WBEmulator):
    def __init__(self, *args):
        self.p = float(args[0]["prob"]) if "prob" in args[0].keys() else 0.5
        super(WhiteBalance, self).__init__()

    def __call__(self, *args, **data):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")

        if random.random() < self.p:
            data["image"], _ = self.generateWbsRGB(data["image"], 1)
            data["image"] = data["image"][..., 0]*255
        return data


class ToFloat(albu.ToFloat):
    def __init__(self, *args):
        p = float(args[0]["prob"]) if "prob" in args[0].keys() else 1
        max_value = args[0]["max_value"] if "max_value" in args[0].keys() else None
        super(ToFloat, self).__init__(max_value, p=p)


build_transform.register("ToFloat", ToFloat)
build_transform.register("WhiteBalance", WhiteBalance)
build_transform.register("HorizontalFlip", HorizontalFlip)      # 随机水平翻转
build_transform.register("RandomBrightnessContrast", RandomBrightnessContrast)  # 随机改变亮度和对比度
build_transform.register("RandomGridShuffle", RandomGridShuffle)    # 将图片分成多个网格
build_transform.register("MultiplicativeNoise", MultiplicativeNoise)  # 对图像乘以一个随机数 扰动
build_transform.register("HueSaturationValue", HueSaturationValue)  # 随机改变色调、饱和度
build_transform.register("GlassBlur", GlassBlur)                # 应用 glass噪声     (不是很好用)
build_transform.register("Blur", Blur)                          # 滤波
build_transform.register("Normalize", Normalize)                # 标准化数据
build_transform.register("CenterCrop", CenterCrop)              # 中心区域裁剪
build_transform.register("RandomFlip", RandomFlip)              # 随机翻转
build_transform.register("Rotate", Rotate)                      # 随机旋转
build_transform.register("RandomCrop", RandomCrop)              # 随机裁剪
build_transform.register("CoarseDropout", CoarseDropout)        # 随机遮挡
build_transform.register("GridDistortion", GridDistortion)      # 网格畸变
build_transform.register("Resize", Resize)                      # 重调尺寸
build_transform.register("ChannelDropout", ChannelDropout)      # 随机通道填充一个值
build_transform.register("ChannelShuffle", ChannelShuffle)      # 随机交换通道数据
build_transform.register("CLAHE", CLAHE)                        # 对比度有上限的适应直方图均衡化
build_transform.register("ColorJitter", ColorJitter)            # 随机更改图像的亮度，对比度和饱和度
build_transform.register("CropNonEmptyMaskIfExists", CropNonEmptyMaskIfExists)   # 裁剪有掩码区域
build_transform.register("Downscale", Downscale)                # 下采样降低图像质量
build_transform.register("ElasticTransform", ElasticTransform)  # 弹性变形
build_transform.register("Equalize", Equalize)                  # 直方图均衡化
build_transform.register("FancyPCA", FancyPCA)                  # PCA增强
build_transform.register("FromFloat", FromFloat)                # 对图像数据进行放大
build_transform.register("GaussianBlur", GaussianBlur)          # 高斯滤波
build_transform.register("GaussNoise", GaussNoise)              # 高斯噪声
build_transform.register("RandomSizedBBoxSafeCrop", RandomSizedBBoxSafeCrop)    # 裁剪随机部分，重新缩放大小，不会丢失box信息

