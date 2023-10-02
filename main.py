import argparse
from modules import VideoProcessor
from utils.config import ConfigManager


def main(args):
    """
    Process a video using the specified configuration.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Load the configuration
    config_manager = ConfigManager(args.config_path)
    config = config_manager.load_config()

    # Initialize and setup the video processor
    processor = VideoProcessor(source_video_path=args.source_video,
                               target_video_path=args.target_video,
                               cfg=config)
    processor.setup()

    # Process the video
    processor.process_video()


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process a video with vehicle tracking and counting.")

    # Define command-line arguments with default values
    parser.add_argument("--config-path", default="./utils/config.yml",
                        help="Path to the configuration file (config.yml).")
    parser.add_argument("--source-video", default="./data/videos/vehicle-counting.mp4",
                        help="Path to the source video file.")
    parser.add_argument("--target-video", default="./data/videos/output.mp4",
                        help="Path to the target video file for the output.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
