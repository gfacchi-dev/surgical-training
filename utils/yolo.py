import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image


class NeedleDriverDetector:
    def __init__(self, model_path="yolov8x-seg.pt", confidence=0.85):
        self._model = YOLO(model_path)
        self._class_names = self._model.names
        self._conf = confidence
        self._model.fuse()

    def predict_on_image(self, rgb_image):
        result = self._model(rgb_image, conf=self._conf)[0]
        # detection
        # result.boxes.xyxy   # box with xyxy format, (N, 4)
        cls = result.boxes.cls.cpu().numpy()  # cls, (N, 1)
        probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
        boxes = result.boxes.xyxy.cpu().numpy()  # box with xyxy format, (N, 4)

        # segmentation
        masks = result.masks.data.cpu().numpy()  # masks, (N, H, W)
        masks = np.moveaxis(masks, 0, -1)  # masks, (H, W, N)
        # rescale masks to original image

        masks = scale_image(masks, result.masks.orig_shape)
        masks = np.moveaxis(masks, -1, 0)  # masks, (N, H, W)

        # filter boxes, probs, cls, masks with class_name == "scissosr"
        class_name = "scissors"
        cls_index = [index for index, name in self._class_names.items() if name == class_name][0]
        cls_filter_index = np.argwhere(cls == cls_index).flatten()
        boxes = boxes[cls_filter_index]
        masks = masks[cls_filter_index]
        cls = cls[cls_filter_index]
        probs = probs[cls_filter_index]

        return boxes, masks, cls, probs

    def draw_result_on_image(self, rgb_image, masks, color, alpha, resize=None):
        rgb_image = np.copy(rgb_image)
        for mask in masks:
            color = color[::-1]
            colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
            colored_mask = np.moveaxis(colored_mask, 0, -1)
            masked = np.ma.MaskedArray(rgb_image, mask=colored_mask, fill_value=color)
            image_overlay = masked.filled()

            if resize is not None:
                rgb_image = cv2.resize(rgb_image.transpose(1, 2, 0), resize)
                image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

            rgb_image = cv2.addWeighted(rgb_image, 1 - alpha, image_overlay, alpha, 0)

        return rgb_image
