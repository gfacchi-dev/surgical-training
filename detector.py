import os
import cv2
import mediapipe as mp
from utils.mediapipe import HandDetector
from utils.yolo import NeedleDriverDetector


class Detector:
    def __init__(self) -> None:
        self.hand_detector = HandDetector()
        self.needle_driver_detector = NeedleDriverDetector()

    def detect(self, video_path):
        capture = cv2.VideoCapture(video_path)
        with self.hand_detector.landmarker as landmarker:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                hand_detection_results = landmarker.detect_for_video(frame_mp, ts)
                _, masks, _, _ = self.needle_driver_detector._predict_on_image(frame)
                yield frame, hand_detection_results, masks

    def detect_and_draw(self, video_path, interactive=True, save_path=None):
        capture = cv2.VideoCapture(video_path)
        with self.hand_detector.landmarker as landmarker:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ts = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                hand_detection_results = landmarker.detect_for_video(frame_mp, ts)
                img_hand = self.hand_detector.draw_result_on_image(frame, hand_detection_results)
                boxes, masks, _, _ = self.needle_driver_detector.predict_on_image(frame)
                frame = self.needle_driver_detector.draw_result_on_image(img_hand, masks, (0, 255, 0), 0.5)
                if save_path and len(boxes) > 0:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path, f"{frame_number}.jpg"), frame)
                if interactive:
                    cv2.imshow("Image", frame)
                    cv2.waitKey(1) & 0xFF
