import numpy as np

def compute_vel(det, box_vel, step, frame_num, prev_speed, s_x, s_y, y_min, y_max, H, fps, scale):
    bottom_center = np.array([(det[0] + det[2]) / 2.0, (det[3] - y_min)]) / scale
    box_vel = box_vel / scale

    # Transform the Velocities using the computed camera to real world equations in the paper
    v_x_trans = (((H[0][0] * box_vel[0] + H[0][1] * box_vel[1]) * (
                H[2][0] * bottom_center[0] + H[2][1] * bottom_center[1] + H[2][2]) -
                  (H[0][0] * bottom_center[0] + H[0][1] * bottom_center[1] + H[0][2]) * (
                              H[2][0] * box_vel[0] + H[2][1] * box_vel[1])) /
                 ((H[2][0] * bottom_center[0] + H[2][1] * bottom_center[1] + H[2][2]) ** 2))

    v_y_trans = (((H[1][0] * box_vel[0] + H[1][1] * box_vel[1]) * (
                H[2][0] * bottom_center[0] + H[2][1] * bottom_center[1] + H[2][2]) -
                  (H[1][0] * bottom_center[0] + H[1][1] * bottom_center[1] + H[1][2]) * (
                              H[2][0] * box_vel[0] + H[2][1] * box_vel[1])) /
                 ((H[2][0] * bottom_center[0] + H[2][1] * bottom_center[1] + H[2][2]) ** 2))

    # Scale Recovery
    s1, s2 = 1.1, 0.9
    a, b = (s2 - s1) / (y_max - y_min) * scale, s1
    s = a * bottom_center[1] + b

    # Velocity calculation
    instant_speed = np.sqrt(sum([(v_x_trans * s_x) ** 2, (v_y_trans * s_y) ** 2]))

    # translating the speed to miles/hour
    vi = instant_speed / frame_num * fps * 9 / 4

    # Suppress noise
    if vi <= 3.0:
        vi = 0.0

    # Speed can be calculated using moving average which is more robust to noisy jitters in instant velocity
    # due to non-ideal detection and tracking
    speed_estimate = (prev_speed * step + vi * s) / (step + 1)

    return speed_estimate
