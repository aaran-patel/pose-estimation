"""
pose_link_logger.py

Pose estimation experiment: runs YOLO11-pose on live RealSense D415 color
frames, deprojects each keypoint into 3D camera-space using the aligned
depth stream, and logs per-link (bone) geometry to a JSONL file.

Deliberately does NOT classify, threshold, or interpret anything (no fall
logic, no risk scoring). The goal is to leave raw link geometry (vector,
length, angle, confidence) available for whatever analysis comes later.

Requirements (already in ~/venvs/aura per prior setup):
    pip install ultralytics opencv-python numpy pyrealsense2

Run:
    source ~/venvs/aura/bin/activate
    python pose_link_logger.py --output session_001.jsonl --show

First run will auto-download yolo11n-pose.pt (~6MB) via ultralytics.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Keypoint + link topology (COCO-17, same order Ultralytics/YOLO-pose emits)
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Named links (bones). Kept as named pairs, not a bare index list, so
# downstream analysis can address e.g. "left_shin" directly.
LINKS = {
    "shoulders":        ("left_shoulder", "right_shoulder"),
    "hips":             ("left_hip", "right_hip"),
    "left_torso":       ("left_shoulder", "left_hip"),
    "right_torso":      ("right_shoulder", "right_hip"),
    "left_upper_arm":   ("left_shoulder", "left_elbow"),
    "left_forearm":     ("left_elbow", "left_wrist"),
    "right_upper_arm":  ("right_shoulder", "right_elbow"),
    "right_forearm":    ("right_elbow", "right_wrist"),
    "left_thigh":       ("left_hip", "left_knee"),
    "left_shin":        ("left_knee", "left_ankle"),
    "right_thigh":      ("right_hip", "right_knee"),
    "right_shin":       ("right_knee", "right_ankle"),
}

KP_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


# ---------------------------------------------------------------------------
# RealSense setup
# ---------------------------------------------------------------------------

def start_realsense(width=1280, height=720, fps=30):
    """Start aligned color+depth streams and return (pipeline, align, intrinsics)."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)  # align depth -> color pixel space

    depth_intrinsics = (
        profile.get_stream(rs.stream.depth)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    return pipeline, align, depth_intrinsics, depth_scale


def deproject_pixel(depth_frame, intrinsics, depth_scale, x, y, patch=2):
    """
    Convert a 2D pixel (x, y) to a 3D camera-space point in meters, using a
    small median-filtered patch around the pixel to reduce depth noise/holes.
    Returns None if no valid depth is found in the patch.
    """
    h, w = depth_frame.get_height(), depth_frame.get_width()
    xi, yi = int(round(x)), int(round(y))

    samples = []
    for dx in range(-patch, patch + 1):
        for dy in range(-patch, patch + 1):
            px, py = xi + dx, yi + dy
            if 0 <= px < w and 0 <= py < h:
                d = depth_frame.get_distance(px, py)  # already in meters
                if d > 0:
                    samples.append(d)

    if not samples:
        return None

    depth_m = float(np.median(samples))
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_m)
    return point  # [X, Y, Z] in meters, camera-space


# ---------------------------------------------------------------------------
# Link geometry computation
# ---------------------------------------------------------------------------

def compute_link_geometry(kp3d, kp_conf, a_name, b_name, conf_thresh=0.3):
    """
    Given a dict of 3D keypoint positions (or None) and confidences, compute
    the 3D vector, length (m), and angle-from-vertical (deg) for one link.
    Returns None if either endpoint is missing or below confidence threshold.
    """
    pa, pb = kp3d.get(a_name), kp3d.get(b_name)
    ca, cb = kp_conf.get(a_name, 0.0), kp_conf.get(b_name, 0.0)

    if pa is None or pb is None or ca < conf_thresh or cb < conf_thresh:
        return None

    vec = np.array(pb) - np.array(pa)  # meters, camera-space
    length = float(np.linalg.norm(vec))
    if length == 0:
        return None

    # Angle from vertical (camera Y-axis). 0 deg = vertical bone,
    # 90 deg = horizontal bone. Useful later for torso-collapse type
    # analysis, but computed here as raw geometry only -- no threshold.
    vertical = np.array([0.0, 1.0, 0.0])
    cos_angle = np.dot(vec, vertical) / (length * np.linalg.norm(vertical))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    return {
        "vector_3d_m": vec.tolist(),
        "length_m": length,
        "angle_from_vertical_deg": angle_deg,
        "confidence": float(min(ca, cb)),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n-pose.pt",
                         help="Ultralytics pose model (nano by default for real-time CPU use)")
    parser.add_argument("--output", default="pose_links.jsonl",
                         help="Path to append JSONL frame records to")
    parser.add_argument("--conf", type=float, default=0.3,
                         help="Per-keypoint confidence threshold for link computation")
    parser.add_argument("--show", action="store_true",
                         help="Display annotated color feed while logging")
    args = parser.parse_args()

    model = YOLO(args.model)
    pipeline, align, depth_intrinsics, depth_scale = start_realsense()

    out_path = Path(args.output)
    frame_id = 0

    print(f"Logging link geometry to {out_path.resolve()}  (Ctrl+C to stop)")

    try:
        with out_path.open("a") as f:
            while True:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                results = model(color_image, verbose=False)[0]
                timestamp = time.time()

                if results.keypoints is None or len(results.keypoints) == 0:
                    frame_id += 1
                    continue

                # Process each detected person separately
                for person_idx, kps in enumerate(results.keypoints):
                    xy = kps.xy[0].cpu().numpy()      # (17, 2) pixel coords
                    conf = kps.conf[0].cpu().numpy() if kps.conf is not None \
                        else np.ones(len(xy))

                    kp2d, kp3d, kp_conf = {}, {}, {}
                    for i, name in enumerate(KEYPOINT_NAMES):
                        x, y = xy[i]
                        c = float(conf[i])
                        kp_conf[name] = c
                        kp2d[name] = [float(x), float(y), c]

                        if c >= args.conf and x > 0 and y > 0:
                            point = deproject_pixel(
                                depth_frame, depth_intrinsics, depth_scale, x, y
                            )
                            kp3d[name] = point
                        else:
                            kp3d[name] = None

                    link_geometry = {}
                    for link_name, (a, b) in LINKS.items():
                        geom = compute_link_geometry(
                            kp3d, kp_conf, a, b, conf_thresh=args.conf
                        )
                        link_geometry[link_name] = geom  # None if unavailable

                    record = {
                        "timestamp": timestamp,
                        "frame_id": frame_id,
                        "person_index": person_idx,
                        "keypoints_2d": kp2d,     # pixel space, per name: [x, y, conf]
                        "keypoints_3d_m": kp3d,   # camera space, per name: [X, Y, Z] or None
                        "links": link_geometry,   # per link name: geometry dict or None
                    }
                    f.write(json.dumps(record) + "\n")

                f.flush()

                if args.show:
                    annotated = results.plot()
                    cv2.imshow("pose_link_logger", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_id += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pipeline.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
