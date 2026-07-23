"""
Borrows the pattern from Liam's main.py, but we use 3D depth for the Torso instead of 2D
"""

import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


POSE_MODEL_PATH = "yolo11n-pose.pt"
TORSO_FALLEN_DEG = 65.0     # torso angle-from-vertical considered "down"
SUSTAIN_FRAMES = 10          # consecutive down-frames before confirming a fall
STANDING_GRACE = 5           # consecutive up-frames to reset the counter (avoids
                              # a single good frame mid-fall resetting everything)
ALERT_COOLDOWN_SECONDS = 60  # matches Liam's alert cooldown
ANGLE_SMOOTH_WINDOW = 5      # rolling mean window; raw per-frame angle is noisy

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def start_realsense(width=1280, height=720, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_intrinsics = (
        profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    )
    return pipeline, align, depth_intrinsics


def deproject_pixel(depth_frame, intrinsics, x, y, patch=2):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    xi, yi = int(round(x)), int(round(y))
    samples = []
    for dx in range(-patch, patch + 1):
        for dy in range(-patch, patch + 1):
            px, py = xi + dx, yi + dy
            if 0 <= px < w and 0 <= py < h:
                d = depth_frame.get_distance(px, py)
                if d > 0:
                    samples.append(d)
    if not samples:
        return None
    depth_m = float(np.median(samples))
    return rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_m)


def torso_angle_from_vertical(kp_xy, kp_conf, depth_frame, intrinsics, conf_thresh=0.3):
  
    angles = []
    for side in ("left", "right"):
        sh_i = KP_INDEX[f"{side}_shoulder"]
        hip_i = KP_INDEX[f"{side}_hip"]
        if kp_conf[sh_i] < conf_thresh or kp_conf[hip_i] < conf_thresh:
            continue

        sh_3d = deproject_pixel(depth_frame, intrinsics, *kp_xy[sh_i])
        hip_3d = deproject_pixel(depth_frame, intrinsics, *kp_xy[hip_i])
        if sh_3d is None or hip_3d is None:
            continue

        vec = np.array(hip_3d) - np.array(sh_3d)
        length = np.linalg.norm(vec)
        if length == 0:
            continue

        vertical = np.array([0.0, 1.0, 0.0])
        cos_a = np.dot(vec, vertical) / (length * np.linalg.norm(vertical))
        angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    return float(np.mean(angles)) if angles else None


def send_alert(message):
   
    ts = time.strftime("%H:%M:%S")
    print(f"[ALERT {ts}] {message}")


def main():
    model = YOLO(POSE_MODEL_PATH)
    pipeline, align, depth_intrinsics = start_realsense()

    down_frames = 0
    up_frames = 0
    fall_confirmed = False
    last_alert_time = 0.0
    angle_history = deque(maxlen=ANGLE_SMOOTH_WINDOW)

    print("Live fall detection running. Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            results = model(color_image, verbose=False)[0]
            annotated = results.plot()

            state_this_frame = "STANDING"

            if results.keypoints is not None and len(results.keypoints) > 0:
                kps = results.keypoints[0]  # single-person for this simple version
                kp_xy = kps.xy[0].cpu().numpy()
                kp_conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(len(kp_xy))

                angle = torso_angle_from_vertical(kp_xy, kp_conf, depth_frame, depth_intrinsics)
                if angle is not None:
                    angle_history.append(angle)
                smoothed_angle = float(np.mean(angle_history)) if angle_history else None

                if smoothed_angle is not None and smoothed_angle >= TORSO_FALLEN_DEG:
                    down_frames += 1
                    up_frames = 0
                else:
                    up_frames += 1
                    if up_frames > STANDING_GRACE:
                        down_frames = 0

            if down_frames >= SUSTAIN_FRAMES:
                state_this_frame = "FALL DETECTED"
                if not fall_confirmed:
                    fall_confirmed = True
                    now = time.time()
                    if now - last_alert_time > ALERT_COOLDOWN_SECONDS:
                        last_alert_time = now
                        send_alert("Fall detected -- person on ground.")
            else:
                fall_confirmed = False

          
            color = (0, 0, 255) if state_this_frame == "FALL DETECTED" else (0, 200, 0)
            cv2.putText(annotated, state_this_frame, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            if down_frames > 0:
                cv2.putText(annotated, f"({down_frames}/{SUSTAIN_FRAMES})", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("live_fall_alert", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
