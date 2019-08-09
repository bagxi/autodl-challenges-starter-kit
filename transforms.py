from typing import Callable, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


class ImgToTensor(A.core.transforms_interface.DualTransform):
    def __init__(self):
        super().__init__(always_apply=True, p=1.0)

    def __call__(self, force_apply=True, **kwargs):
        kwargs.update({'image': tensor_from_rgb_image(kwargs['image'])})
        return kwargs


class ImgFromTF(A.core.transforms_interface.DualTransform):
    def __init__(
        self,
        image_size: Tuple[int, int] = None,
        min_max_size: Tuple[int, int] = (28, 224),
        default_size: int = 224,
        gray_to_rgb: bool = True
    ):
        super().__init__(always_apply=True, p=1.0)

        min_size, max_size = min_max_size

        def preproc(dim):
            return default_size if dim == -1 else max(min(dim, max_size), min_size)

        self.height, self.width = (preproc(dim) for dim in image_size[:2])

        self.gray_to_rgb = gray_to_rgb

    def __call__(self, force_apply=True, **kwargs):
        image = kwargs['image']

        # drop time dim
        image = image[0]

        # denorm image (e.g. scale to [0, 255])
        if image.max() <= 1.0:
            image = (image * 255).round().astype(np.uint8)

        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if self.gray_to_rgb and image.shape[-1] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        kwargs.update({'image': image})
        return kwargs


def _resolve_transform_fn(name):
    d = {
        'stack': A.Compose,
        'one_of': A.OneOf,
        'nil': A.NoOp,
        'resize': A.Resize,
        'crop': A.RandomSizedCrop,
        'flip_h': A.HorizontalFlip,
        'brightness_contrast': A.RandomBrightnessContrast,
        'gamma': A.RandomGamma,
        'hsv_shift': A.HueSaturationValue,
        'clahe': A.CLAHE,
        'jpeg': A.JpegCompression,
        'normalize': A.Normalize,
        'img_to_tensor': ImgToTensor,
        'img_from_tf': ImgFromTF,
    }
    return d[name]


def get_transform_function(config: List[Dict], **kwargs) -> Callable:
    """

    Args:
        config: list of augs with params

    Thanks to @arsenyinfo

    """
    transforms = []
    for transform_params in config:
        name = transform_params.pop('name')
        cls = _resolve_transform_fn(name)
        transform_params = {k: (v if v != 'forward' else kwargs[k]) for k, v in transform_params.items()}
        transforms.append(cls(**transform_params))

    transform = A.Compose(transforms)

    def process(image):
        image = transform(image=image)['image']
        return image

    return process
