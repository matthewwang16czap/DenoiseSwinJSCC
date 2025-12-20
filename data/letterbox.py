import cv2
import numpy as np
from typing import Any


class LetterBox:
    """
    This class resizes and pads images to a specified shape while preserving aspect ratio.
    """

    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """Initialize LetterBox object for resizing and padding images.

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.

        Args:
            new_shape (tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scale_fill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).
            padding_value (int): Value for padding the image. Default is 114.
            interpolation (int): Interpolation method for resizing. Default is cv2.INTER_LINEAR.
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(self, image: np.ndarray = None):
        """Resize and pad an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        Args:
            image (np.ndarray | None): The input image as a numpy array

        Returns:
            image (np.ndarray): The resized and padded image.
            valid (np.ndarray): The valid regions (1 = valid, 0 = padded).
        """
        shape = image.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (
            self.new_shape[1] - new_unpad[0],
            self.new_shape[0] - new_unpad[1],
        )  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (self.new_shape[1], self.new_shape[0])

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=self.interpolation)
            if image.ndim == 2:
                image = image[..., None]

        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)
        h, w, c = image.shape
        if c == 3:
            image = cv2.copyMakeBorder(
                image,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(self.padding_value,) * 3,
            )
        else:  # multispectral
            pad_image = np.full(
                (h + top + bottom, w + left + right, c),
                fill_value=self.padding_value,
                dtype=image.dtype,
            )
            pad_image[top : top + h, left : left + w] = image
            image = pad_image

        valid = np.ones(self.new_shape, dtype=np.uint8)
        valid[top : top + h, left : left + w] = 0

        return image, valid
