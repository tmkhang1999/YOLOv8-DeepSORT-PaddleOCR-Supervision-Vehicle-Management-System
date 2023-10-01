from tqdm.notebook import tqdm
import numpy as np
from collections import deque
from supervision.video.sink import VideoSink
from deep_sort_realtime.deepsort_tracker import DeepSort
from paddleocr import PaddleOCR
from ultralytics import YOLO

from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.geometry.dataclasses import Point
from supervision.draw.color import ColorPalette

from modules.annotation import draw_note_box, draw_plate_box, draw_tracking_line
from modules.plate_recognition import recognize_plate
from modules.speed_estimation import compute_vel


class VideoProcessor:
    def __init__(self, source_video_path, target_video_path, cfg):
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.data_tracker = {}
        self.CLASS_ID = [2, 3, 5, 7]

        # Get the camera's information
        self.s_x = cfg['s_x']
        self.s_y = cfg['s_y']
        self.f1 = cfg['f1']
        self.f2 = cfg['f2']
        self.l0 = cfg['L0']

        # Specify H matrix to rectify videos of each location
        self.H = [[self.l0, -self.l0 * (self.f1 / self.f2), 0.0], [0.0, 1.0, 0.0], [0.0, -(1 / self.f2), 1.0]]
        self.y_min = cfg['y_min']
        self.y_max = cfg['y_max']
        self.V0 = cfg['V0']

    def setup(self):
        # Initialize the vehicle detection
        self.vehicle_detector = YOLO("../data/models/yolov8x.pt")
        self.CLASS_DICT = {}
        for id in self.CLASS_ID:
            self.CLASS_DICT[id] = self.vehicle_detector.model.names[id]

        # Initialize the tracker
        self.object_tracker = DeepSort(max_age=20,
                                       n_init=2,
                                       nms_max_overlap=1.0,
                                       max_cosine_distance=0.3,
                                       nn_budget=None,
                                       override_track_class=None,
                                       embedder="mobilenet",
                                       half=True,
                                       bgr=True,
                                       embedder_gpu=True,
                                       embedder_model_name=None,
                                       embedder_wts=None,
                                       polygon=False,
                                       today=None)

        # Initialize the license_plate_detector
        self.plate_detector = YOLO("../data/models/license_plate_detector.pt")

        # Initialize the Paddle OCR
        self.ocr_model = PaddleOCR(lang='en', show_log=False, use_angle_cls=True, use_gpu=False)

        # create VideoInfo instance
        self.video_info = VideoInfo.from_video_path(self.source_video_path)
        # create frame generator
        self.generator = get_video_frames_generator(self.source_video_path)

        # create LineCounter instance
        LINE_START = Point(50, 1300)
        LINE_END = Point(3840 - 50, 1300)
        self.line_counter = LineCounter(start=LINE_START, end=LINE_END)
        # create instance of BoxAnnotator and LineCounterAnnotator
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=1)
        self.line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

    def process_video(self):
        with VideoSink(self.target_video_path, self.video_info) as sink:
            frame_count = 0
            for frame in tqdm(self.generator, total=self.video_info.total_frames):
                draw_note_box(frame, self.CLASS_DICT)
                # detect vehicle on a single frame
                results = self.vehicle_detector(frame, conf=0.6)

                # tracking detections
                dets = []
                for x1, y1, x2, y2, conf, id in results[0].boxes.data.cpu().numpy():
                    if id in self.CLASS_ID:
                        dets.append(([x1, y1, int(x2 - x1), int(y2 - y1)], conf, id))

                if dets:
                    tracks = self.object_tracker.update_tracks(dets, frame=frame)
                    track_info = [(track.to_tlbr(), track.get_det_conf(), track.get_det_class(), track.track_id)
                                  for track in tracks]
                    track_dets, track_confs, track_classes, track_ids = zip(*track_info)
                    mask = np.array([conf is not None for conf in track_confs])

                    if len(track_dets) > 0:
                        detections = Detections(
                            xyxy=np.array(track_dets),
                            confidence=np.array(track_confs),
                            class_id=np.array(track_classes).astype(int),
                            tracker_id=np.array(track_ids).astype(int)
                        )

                        # filter not confirmed
                        detections.filter(mask=mask, inplace=True)

                        labels = []
                        for xyxy, confidence, class_id, tracker_id in detections:
                            if tracker_id not in self.data_tracker:
                                # Initialize the tracker state if it doesn't exist
                                self.data_tracker[tracker_id] = [self.V0, 0, frame_count, deque(maxlen=64), None,
                                                                 0]  # [speed, step, frame_count, center_point, plate, ocr_conf]

                            # add center to buffer
                            x1, y1, x2, y2 = [int(i) for i in xyxy]
                            center = (int((x2 + x1) / 2), int((y1 + y2) / 2))
                            self.data_tracker[tracker_id][3].appendleft(center)

                            speed_label = ""
                            # Check y threshold
                            if xyxy[1] > self.y_min:
                                tracker_state = self.data_tracker[tracker_id]
                                prev_speed, step, frame_num = tracker_state[0], tracker_state[1], frame_count - \
                                                              tracker_state[2]

                                # Estimate the speed of vehicle
                                if step == 0:
                                    tracker_state[1] += 1
                                    tracker_state[2] = frame_count
                                else:
                                    # Calculate the box velocities
                                    data_deque = self.data_tracker[tracker_id][3]
                                    box_vel = np.array(data_deque[0]) - np.array(data_deque[1])

                                    # Calculate the new speed
                                    speed = compute_vel(xyxy, box_vel, step, frame_num, prev_speed,
                                                        self.s_x, self.s_y, self.y_min, self.y_max,self.H,
                                                        self.video_info.fps, scale=3)

                                    # Update the tracker
                                    tracker_state[0] = speed
                                    tracker_state[1] += 1
                                    tracker_state[2] = frame_count

                                    # Update speed label
                                    speed_label = f" | Speed: {tracker_state[0]:0.2f} mph"

                                # Recognize plate
                                plate, ocr_conf = recognize_plate(frame, xyxy)
                                if plate is not None and ocr_conf > tracker_state[-1]:
                                    tracker_state[-2], tracker_state[-1] = plate, ocr_conf

                                # Visualize the plate
                                if tracker_state[-2] is not None:
                                    text = f"[{tracker_state[-2]}] Conf: {tracker_state[-1]:0.2f}"
                                    frame = draw_plate_box(frame, xyxy, text)

                            # Update labels
                            labels.append(
                                f"[{tracker_id}] {self.CLASS_DICT[class_id].capitalize()} | Conf: {confidence:0.2f}" + speed_label)

                            # draw the tracking line
                            draw_tracking_line(frame, self.data_tracker[tracker_id][3], class_id)

                        # updating line counter
                        self.line_counter.update(detections=detections)

                # annotate and display frame
                frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                frame_count += 1

                self.line_annotator.annotate(frame=frame, line_counter=self.line_counter)
                sink.write_frame(frame)

        print("DONE")