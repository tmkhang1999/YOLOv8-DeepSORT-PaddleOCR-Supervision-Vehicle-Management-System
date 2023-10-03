import logging
import math

from tqdm import tqdm
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

from modules.plate_recognition import PlateRecognizer
from modules.speed_estimation import SpeedEstimator
from modules.annotation import NoteAnnotator, TraceAnnotator

log = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, source_video_path, target_video_path, cfg):
        """
        Initialize the VideoProcessor.

        Args:
            source_video_path (str): Path to the source video.
            target_video_path (str): Path to the target video.
            cfg (dict): Configuration dictionary.
        """
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.config = cfg
        self.V0 = cfg['camera']['V0']
        self.y_min = cfg['camera']['y_min']
        self.data_tracker = {}
        self.CLASS_ID = [2, 3, 5, 7]

    def setup(self):
        """
           Set up the video processing environment and initialize necessary components.
        """
        # Initialize the vehicle detection/ plate detection/ paddleOCR models
        self.vehicle_detector = YOLO(self.config['models']['vehicle_detector_path'])
        self.plate_detector = YOLO(self.config['models']['plate_detector_path'])
        self.ocr_model = PaddleOCR(lang='en', show_log=False, use_angle_cls=True, use_gpu=False)

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

        # Create an ID dictionary for each vehicle type
        self.CLASS_DICT = {}
        for id in self.CLASS_ID:
            self.CLASS_DICT[id] = self.vehicle_detector.model.names[id]

        # Create VideoInfo instance, frame generator
        self.video_info = VideoInfo.from_video_path(self.source_video_path)
        self.generator = get_video_frames_generator(self.source_video_path)

        # Create LineCounter instance
        width, height = self.video_info.resolution
        LINE_START = Point(50, int(height * 0.6))
        LINE_END = Point(width - 50, int(height * 0.6))
        self.line_counter = LineCounter(start=LINE_START, end=LINE_END)

        # Create instance of BoxAnnotator and LineCounterAnnotator
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=1)
        self.line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

        # Create instance of PlateRecognizer and SpeedEstimator
        self.plate_recognizer = PlateRecognizer(license_plate_detector=self.plate_detector, ocr_model=self.ocr_model)
        self.speed_estimator = SpeedEstimator(cfg=self.config['camera'], fps=self.video_info.fps)

        # Create instance of NoteAnnotator and NoteAnnotator
        self.note_annotator = NoteAnnotator(color_dict=self.CLASS_DICT)
        self.trace_annotator = TraceAnnotator()

        # Calculate the scale of our own frame compared to 720p (1280 x 720px)
        self.scale = math.sqrt(width * height / 1280 / 720)

    def process_video(self):
        """
           Process the video, including detection, tracking, speed estimation, vehicle counting and annotation.
        """
        with VideoSink(self.target_video_path, self.video_info) as sink:
            frame_count = 0
            for frame in tqdm(self.generator, total=self.video_info.total_frames):
                # Draw notes in the upper left of the frame
                self.note_annotator.annotate(frame)

                # Detect vehicle on a single frame
                results = self.vehicle_detector(frame, conf=0.6)

                # Tracking detections
                dets = []
                for x1, y1, x2, y2, conf, id in results[0].boxes.data.cpu().numpy():
                    if id in self.CLASS_ID:
                        dets.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], conf, id))

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

                        # Filter not confirmed
                        detections.filter(mask=mask, inplace=True)

                        labels = []
                        for xyxy, confidence, class_id, tracker_id in detections:
                            if tracker_id not in self.data_tracker:
                                # Initialize the tracker state
                                # if it doesn't exist [speed, step, frame_count, center_points_buffer, plate, ocr_conf]
                                self.data_tracker[tracker_id] = [self.V0, 0, frame_count, deque(maxlen=64), None, 0]

                            # Add center_point to buffer
                            x1, y1, x2, y2 = [int(i) for i in xyxy]
                            center_point = (int((x2 + x1) / 2), int((y1 + y2) / 2))
                            self.data_tracker[tracker_id][3].appendleft(center_point)

                            speed_label = ""
                            tracker_state = self.data_tracker[tracker_id]
                            # Check y threshold for speed estimation and plate detection
                            if xyxy[1] > self.y_min:
                                prev_speed = tracker_state[0]
                                step = tracker_state[1]
                                frame_num = frame_count - tracker_state[2]

                                # Estimate the speed of vehicle
                                if step == 0:
                                    tracker_state[1] += 1
                                    tracker_state[2] = frame_count
                                else:
                                    # Calculate the box velocities
                                    data_deque = self.data_tracker[tracker_id][3]
                                    box_vel = np.array(data_deque[0]) - np.array(data_deque[1])

                                    # Calculate the new speed
                                    speed = self.speed_estimator.compute_vel(xyxy, box_vel, step, frame_num, prev_speed,
                                                                             self.scale)

                                    # Update the tracker_state
                                    tracker_state[0] = speed
                                    tracker_state[1] += 1
                                    tracker_state[2] = frame_count

                                    # Update speed label
                                    speed_label = f" | Speed: {tracker_state[0]:0.2f} mph"

                                # Recognize plate
                                plate, ocr_conf = self.plate_recognizer.detect(frame, xyxy)
                                if plate is not None and ocr_conf > tracker_state[-1]:
                                    tracker_state[-2], tracker_state[-1] = plate, ocr_conf

                                # Visualize plate
                                if tracker_state[-2] is not None:
                                    text = f"[{tracker_state[-2]}] Conf: {tracker_state[-1]:0.2f}"
                                    frame = self.plate_recognizer.annotate(frame, xyxy, text)

                            # Update labels
                            labels.append(
                                f"[{tracker_id}] {self.CLASS_DICT[class_id].capitalize()} "
                                f"| Conf: {confidence:0.2f}" + speed_label)

                            # Annotate tracking trail
                            self.trace_annotator.annotate(frame, tracker_state[3], class_id)

                        # Update line counter
                        self.line_counter.update(detections=detections)

                # Annotate and display frame
                frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                self.line_annotator.annotate(frame=frame, line_counter=self.line_counter)

                frame_count += 1
                sink.write_frame(frame)

        log.info("___________________________DONE___________________________")
