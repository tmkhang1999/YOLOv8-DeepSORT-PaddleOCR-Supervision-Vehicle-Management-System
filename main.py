from utils.video_processor import VideoProcessor

if __name__ == "__main__":
    cfg = {'V0': 75.0, 's_x': 3.2 / 5, 's_y': 3, 'y_min': 750.0, 'y_max': 2500.0,
           'L0': 0.2, 'f1': 928.7943, 'f2': -67.777}
    processor = VideoProcessor("./data/videos/vehicle-counting.mp4", "./data/videos/output.mp4", cfg)
    processor.setup()
    processor.process_video()
