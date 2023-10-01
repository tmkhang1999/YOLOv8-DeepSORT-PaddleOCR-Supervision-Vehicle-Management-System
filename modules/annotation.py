from supervision.draw.color import ColorPalette
import numpy as np
import cv2


def draw_tracking_line(frame, data_deque, cls):
    # remove tracked point from buffer if object is lost
    # for key in list(data_deque):
    #     if key not in list(detections.tracker_id):
    #         data_deque.pop(key)

    # draw trail
    color = ColorPalette().by_idx(cls).as_bgr()

    for i in range(1, len(data_deque)):
        # check if on buffer value is none
        if data_deque[i - 1] is None or data_deque[i] is None:
            continue

        # generate dynamic thickness of trails
        thickness = int(np.sqrt(64 / float(i + i)) * 4)

        # draw trails
        cv2.line(frame, data_deque[i - 1], data_deque[i], color, thickness)


def draw_plate_box(frame, xyxy, plate):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 1.5
    text_thickness = 5
    text_padding = 10

    x1, y1, x2, y2 = xyxy.astype(int)

    text_width, text_height = cv2.getTextSize(
        text=plate,
        fontFace=font,
        fontScale=text_scale,
        thickness=text_thickness,
    )[0]

    text_x = x1 + text_padding
    text_y = y2 + text_padding + text_height

    text_background_x1 = x1
    text_background_y1 = y2

    text_background_x2 = x1 + 2 * text_padding + text_width
    text_background_y2 = y2 + 2 * text_padding + text_height

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
        fontFace=font,
        fontScale=text_scale,
        color=(0, 0, 0),
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )

    return frame


def find_longest_value(my_dict):
    longest_length = 0

    for val in my_dict.values():
        if isinstance(val, str) and len(val) > longest_length:
            longest_length = len(val)

    return longest_length


def draw_note_box(image, color_dict):
    padding = int(image.shape[0] / 72)
    box_size = padding * 4

    font = cv2.FONT_HERSHEY_TRIPLEX
    text_scale = 2
    text_thickness = 5

    x1, y1 = padding, padding
    x2 = box_size + padding * 3 + int(padding * 4 / 3) * find_longest_value(color_dict)
    y2 = len(color_dict) * (padding + box_size) + padding * 2

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), cv2.FILLED)

    x1_box = x1 + padding
    x2_box = x1 + padding + box_size
    y1_box = y1 + padding
    y2_box = y1 + padding + box_size

    x_text = x1 + padding * 2 + box_size

    # Iterate through the color dictionary and draw square boxes
    for i, key in enumerate(color_dict):
        if i != 0:
            y1_box += (padding + box_size)
            y2_box += (padding + box_size)

        color_box = ColorPalette().by_idx(key).as_bgr()
        cv2.rectangle(image, (x1_box, y1_box), (x2_box, y2_box), color_box, cv2.FILLED)

        y_text = y2_box - padding

        cv2.putText(
            img=image,
            text=color_dict[key].capitalize(),
            org=(x_text, y_text),
            fontFace=font,
            fontScale=text_scale,
            color=(0, 0, 0),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    return image
