import numpy as np


class SpeedEstimator:
    """
    A class for estimating the speed of an object based on 2D images.

    Args:
        cfg (dict): Configuration parameters containing camera and transformation details.
        fps (int): Frames per second of the video stream.
    """

    def __init__(
            self,
            cfg: dict,
            fps: int):
        # Get the camera's information
        self.s_x = cfg['s_x']
        self.s_y = cfg['s_y']
        self.f1 = cfg['f1']
        self.f2 = cfg['f2']
        self.l0 = cfg['L0']

        # Specify H matrix to rectify videos of each location
        self.H = [
            [self.l0, -self.l0 * (self.f1 / self.f2), 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -(1 / self.f2), 1.0]
        ]
        self.y_min = cfg['y_min']
        self.y_max = cfg['y_max']
        self.V0 = cfg['V0']
        self.fps = fps

    def compute_vel(
            self,
            det: np.ndarray,
            box_vel: np.ndarray,
            step: int,
            frame_num: int,
            prev_speed: float,
            scale: float
    ) -> float:
        """
        Compute the estimated velocity of an object based on detected velocities.

        Args:
            det (np.ndarray): Bounding box coordinates [x1, y1, x2, y2].
            box_vel (np.ndarray): Velocity vector of bottom center.
            step (int): The number of times the object's speed is calculated.
            frame_num (int): Total number of frames between 2 calculations.
            prev_speed (float): Previous speed estimate.
            scale (float): Scaling factor compared to 720p (1280 x 720px).

        Returns:
            float: Estimated speed in miles per hour.
        """
        bottom_center = np.array([(det[0] + det[2]) / 2.0, (det[3] - self.y_min)]) / scale
        box_vel = box_vel / scale

        # Transform the Velocities using the computed camera to real world equations
        v_x_trans = (((self.H[0][0] * box_vel[0] + self.H[0][1] * box_vel[1]) *
                      (self.H[2][0] * bottom_center[0] + self.H[2][1] * bottom_center[1] + self.H[2][2]) -
                      (self.H[0][0] * bottom_center[0] + self.H[0][1] * bottom_center[1] + self.H[0][2]) *
                      (self.H[2][0] * box_vel[0] + self.H[2][1] * box_vel[1])) /
                     ((self.H[2][0] * bottom_center[0] + self.H[2][1] * bottom_center[1] + self.H[2][2]) ** 2))

        v_y_trans = (((self.H[1][0] * box_vel[0] + self.H[1][1] * box_vel[1]) *
                      (self.H[2][0] * bottom_center[0] + self.H[2][1] * bottom_center[1] + self.H[2][2]) -
                      (self.H[1][0] * bottom_center[0] + self.H[1][1] * bottom_center[1] + self.H[1][2]) *
                      (self.H[2][0] * box_vel[0] + self.H[2][1] * box_vel[1])) /
                     ((self.H[2][0] * bottom_center[0] + self.H[2][1] * bottom_center[1] + self.H[2][2]) ** 2))

        # Scale Recovery
        s1, s2 = 1.1, 0.9
        a, b = (s2 - s1) / (self.y_max - self.y_min) * scale, s1
        s = a * bottom_center[1] + b

        # Velocity calculation
        instant_speed = np.sqrt(sum([(v_x_trans * self.s_x) ** 2, (v_y_trans * self.s_y) ** 2]))

        # Translating the speed to miles per hour
        vi = instant_speed / frame_num * self.fps * 9 / 4

        # Suppress noise
        if vi <= 3.0:
            vi = 0.0

        # Speed can be calculated using moving average which is more robust to noisy jitters in instant velocity
        # due to non-ideal detection and tracking
        speed_estimate = (prev_speed * step + vi * s) / (step + 1)

        return speed_estimate
