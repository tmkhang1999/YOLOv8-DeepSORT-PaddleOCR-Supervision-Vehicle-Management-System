from collections import deque
from typing import Union

import cv2
import numpy as np
from supervision.draw.color import Color
from supervision.draw.color import ColorPalette


def find_longest_value(color_dict):
    """
    Find the length of the longest color name in the color dictionary.

    Args:
        color_dict (dict): A dictionary mapping color IDs to color names.

    Returns:
        int: The length of the longest color name.
    """
    return max(len(name) for name in color_dict.values())


class NoteAnnotator:
    """
    A class for annotating an image with colored boxes and labels.

    Args:
        color_dict (dict): A dictionary mapping color IDs to color names.
        color (Union[Color, ColorPalette], optional): The color palette to use. Defaults to ColorPalette().
        font (int, optional): The font type to use. Defaults to cv2.FONT_HERSHEY_TRIPLEX.
        text_scale (float, optional): The scale of the text. Defaults to 2.
        text_thickness (int, optional): The thickness of the text. Defaults to 5.
    """

    def __init__(
            self,
            color_dict: dict,
            color: Union[Color, ColorPalette] = ColorPalette(),
            font=cv2.FONT_HERSHEY_TRIPLEX,
            text_scale=2,
            text_thickness=5):

        self.color_dict = color_dict
        self.color = color
        self.font = font
        self.text_scale = text_scale
        self.text_thickness = text_thickness

    def annotate(self, frame):
        """
        Draw colored boxes and labels on the image (frame) based on the color dictionary.

        Args:
            frame (numpy.ndarray): The input image (frame) to annotate.

        Returns:
            numpy.ndarray: The annotated image (frame).
        """
        padding = int(frame.shape[0] / 72)
        box_size = padding * 4

        x1, y1 = padding, padding
        x2 = box_size + padding * 3 + int(padding * 4 / 3) * find_longest_value(self.color_dict)
        y2 = len(self.color_dict) * (padding + box_size) + padding * 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), cv2.FILLED)

        x1_box = x1 + padding
        x2_box = x1 + padding + box_size
        y1_box = y1 + padding
        y2_box = y1 + padding + box_size

        x_text = x1 + padding * 2 + box_size

        # Iterate through the color dictionary and draw square boxes
        for i, key in enumerate(self.color_dict):
            if i != 0:
                y1_box += (padding + box_size)
                y2_box += (padding + box_size)

            color = self.color.by_idx(key)
            cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), color.as_bgr(), cv2.FILLED)

            y_text = y2_box - padding

            cv2.putText(
                img=frame,
                text=self.color_dict[key].capitalize(),
                org=(x_text, y_text),
                fontFace=self.font,
                fontScale=self.text_scale,
                color=(0, 0, 0),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )

        return frame


class TraceAnnotator:
    """
    A class for annotating a frame with a tracking trail.

    Args:
        color (Union[Color, ColorPalette], optional): Color or ColorPalette for annotation.
            Defaults to ColorPalette.default().
        text_scale (float, optional): The scale of the annotated text. Defaults to 1.5.
        text_thickness (int, optional): The thickness of the annotated text. Defaults to 5.
        text_padding (int, optional): The padding around the annotated text. Defaults to 10.
    """

    def __init__(
            self,
            color: Union[Color, ColorPalette] = ColorPalette(),
            text_scale: float = 1.5,
            text_thickness: int = 5,
            text_padding: int = 10
    ):
        self.color = color
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.text_padding = text_padding

    def annotate(
            self,
            frame,
            data_deque: deque,
            cls: int
    ) -> None:
        """
        Annotate the frame with a tracking trail.

        Args:
            frame: The input image frame to annotate.
            data_deque (deque): A deque containing tracking data points.
            cls: The class index for selecting a color for the tracking trail.
        """
        # Define color of the tracking trail
        color = self.color.by_idx(cls)

        for i in range(1, len(data_deque)):
            # Check if one of the buffer values is None
            if data_deque[i - 1] is None or data_deque[i] is None:
                continue

            # Generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 4)

            # Draw trails
            cv2.line(frame, data_deque[i - 1], data_deque[i], color.as_bgr(), thickness)

