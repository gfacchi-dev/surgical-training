import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class DrawingOptions:
    def __init__(self, margin=10, font_size=1, font_thickness=1, handedness_text_color=(88, 205, 54)):
        self.MARGIN = margin
        self.FONT_SIZE = font_size
        self.FONT_THICKNESS = font_thickness
        self.HANDEDNESS_TEXT_COLOR = handedness_text_color


class HandDetector:
    def __init__(self, model_path="hand_landmarker.task", running_mode=VisionRunningMode.VIDEO, num_hands=2, min_hand_presence_confidence=0.1, drawing_options: DrawingOptions = DrawingOptions()):
        assert running_mode in [VisionRunningMode.VIDEO, VisionRunningMode.IMAGE], "Invalid running mode. Choose from VisionRunningMode.VIDEO or VisionRunningMode.IMAGE"
        assert num_hands in [1, 2], "Invalid number of hands. Choose from 1 or 2"
        assert 0.0 <= min_hand_presence_confidence <= 1.0, "Invalid min_hand_presence_confidence. Choose from 0.0 to 1.0"
        assert isinstance(drawing_options, DrawingOptions), "Invalid drawing options. Should be an instance of DrawingOptions"
        self._options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path), running_mode=running_mode, num_hands=num_hands, min_hand_presence_confidence=min_hand_presence_confidence
        )
        self._drawing_options = drawing_options
        self.landmarker = HandLandmarker.create_from_options(self._options)

    def draw_result_on_image(self, rgb_image, detection_result):
        assert isinstance(rgb_image, np.ndarray), "Invalid image. Should be a numpy array"
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self._drawing_options.MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                self._drawing_options.FONT_SIZE,
                self._drawing_options.HANDEDNESS_TEXT_COLOR,
                self._drawing_options.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image

    # def detect_for_video(self, video_path):
    #     cap = cv2.VideoCapture(video_path)
    #     with self.landmarker as landmarker:
    #         while cap.isOpened():
    #             success, image = cap.read()
    #             if not success:
    #                 break

    #             # Convert the BGR image to RGB and process it with MediaPipe Hands.
    #             results = landmarker.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #             # Draw the hand landmarks on the image.
    #             if results.multi_hand_landmarks:
    #                 image = self._draw_landmarks_on_image(image, results)

    #             cv2.imshow("MediaPipe Hands", image)
    #             if cv2.waitKey(5) & 0xFF == 27:
    #                 break
    #         cap.release()
    #         cv2.destroyAllWindows()
