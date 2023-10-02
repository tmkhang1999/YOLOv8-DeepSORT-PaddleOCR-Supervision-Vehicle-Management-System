from typing import Union
import cv2
import numpy as np
from supervision.draw.color import Color, ColorPalette


class PlateRecognizer:
    """
    A class for license plate recognition.

    Args:
        license_plate_detector: A license plate detection model.
        ocr_model: An Optical Character Recognition (OCR) model.
        plate_conf (float, optional): Confidence threshold for license plate detection.
            Defaults to 0.6.
        color (Union[Color, ColorPalette], optional): Color or ColorPalette for annotation.
            Defaults to ColorPalette.default().
        text_scale (float, optional): The scale of the annotated text. Defaults to 1.5.
        text_thickness (int, optional): The thickness of the annotated text. Defaults to 5.
        text_padding (int, optional): The padding around the annotated text. Defaults to 10.
    """

    def __init__(
            self,
            license_plate_detector,
            ocr_model,
            plate_conf: float = 0.6,
            color: Union[Color, ColorPalette] = ColorPalette(),
            text_scale: float = 1.5,
            text_thickness: int = 5,
            text_padding: int = 10
    ):
        self.license_plate_detector = license_plate_detector
        self.ocr_model = ocr_model
        self.plate_conf: float = plate_conf
        self.color: Union[Color, ColorPalette] = color
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.text_padding = text_padding

    def detect(
            self,
            frame: np.ndarray,
            xyxy: np.ndarray
    ) -> Union[None, tuple[str, float]]:
        """
        Detect a license plate in a given frame.

        Args:
            frame(np.ndarray): The input image frame to detect.
            xyxy (np.ndarray): A NumPy array containing the coordinates [x1, y1, x2, y2] of the region of interest (ROI)
                where the car and license plate are located.

        Returns:
            Union[None, tuple[str, float]]: A tuple containing the recognized plate text and confidence score,
            or None if no plate is detected.
        """
        x1, y1, x2, y2 = [int(i) for i in xyxy]
        car_frame = frame[int(y1):int(y2), int(x1):int(x2)]
        plates = self.license_plate_detector(car_frame, conf=self.plate_conf)

        if plates[0]:
            x1, y1, x2, y2 = plates[0].boxes.xyxy.cpu().numpy()[0]
            plate_frame = car_frame[int(y1):int(y2), int(x1):int(x2)]

            result = self.ocr_model.ocr(plate_frame, cls=True)
            if result and result[0]:
                info = result[0][0][-1]
                return info[0], info[1]

        return None, None

    def annotate(
            self,
            frame: np.ndarray,
            xyxy: np.ndarray,
            plate: str) -> np.ndarray:
        """
        Annotate the frame with a license plate.

        Args:
            frame (np.ndarray): The input image frame to annotate.
            xyxy (np.ndarray): A NumPy array containing the coordinates [x1, y1, x2, y2] of the license plate region.
            plate (str): The recognized license plate text.

        Returns:
            np.ndarray: The annotated image frame.
        """
        x1, y1, x2, y2 = xyxy.astype(int)

        text_width, text_height = cv2.getTextSize(
            text=plate,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=self.text_scale,
            thickness=self.text_thickness,
        )[0]

        text_x = x1 + self.text_padding
        text_y = y2 + self.text_padding + text_height

        text_background_x1 = x1
        text_background_y1 = y2

        text_background_x2 = x1 + 2 * self.text_padding + text_width
        text_background_y2 = y2 + 2 * self.text_padding + text_height

        cv2.rectangle(
            img=frame,
            pt1=(text_background_x1, text_background_y1),
            pt2=(text_background_x2, text_background_y2),
            color=(255, 255, 255),
            thickness=cv2.FILLED,
        )

        cv2.putText(
            img=frame,
            text=plate,
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=self.text_scale,
            color=(0, 0, 0),
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )

        return frame
