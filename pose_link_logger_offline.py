"""
pose_link_logger_offline.py

Offline counterpart to pose_link_logger.py: runs YOLO11-pose on a pre-recorded
RGB+depth PNG sequence (built for the UR Fall Detection Dataset / URFD file
layout: "<prefix>-rgb-NNN.png" and "<prefix>-d-NNN.png", zero-padded, frame
numbers matching 1:1 between the two folders) instead of a live RealSense
camera, and logs the same per-link geometry schema.

Depth conversion follows URFD's documented formula (depth in mm =
depth_scale * raw_pixel_value / 65535, where depth_scale is a per-camera,
per-sequence-type constant given in the dataset docs -- e.g. 6000 for fall
sequences on camera 0, 7000 for ADL sequences on camera 0). Deprojection to
3D uses APPROXIMATE Kinect v1 default intrinsics (fx=fy=525, cx=319.5,
cy=239.5 @ 640x480), since no exact calibration exists for that capture rig.
Treat absolute lengths from this script as approximate -- angle *patterns*
are the more trustworthy signal.

Deliberately does NOT classify, threshold, or interpret anything (no fall
logic, no risk scoring). --label just records the dataset's ground-truth
activity/session tag as metadata for downstream comparison; it is not
derived from the geometry.

Run:
    source ~/venvs/aura/bin/activate
    python pose_link_logger_offline.py \
        --input-dir ~/aura_pose_project/datasets/urfd_fall01 \
        --depth-scale 6000 --label fall \
        --output ~/aura_pose_project/pose_sessions/session_urfd_fall01.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Keypoint + link topology (COCO-17, same order Ultralytics/YOLO-pose emits)
# -- identical to pose_link_logger.py so output schemas match.
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

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

# Approximate Kinect v1 default intrinsics @ 640x480 -- see module docstring.
KINECT_FX = 525.0
KINECT_FY = 525.0
KINECT_CX = 319.5
KINECT_CY = 239.5

FRAME_NUM_RE = re.compile(r"(\d+)\.\w+$")


# ---------------------------------------------------------------------------
# Frame pairing
# ---------------------------------------------------------------------------

def index_frames(directory):
    """Map zero-padded frame-number string -> file path for one folder."""
    out = {}
    for p in sorted(Path(directory).iterdir()):
        if not p.is_file():
            continue
        m = FRAME_NUM_RE.search(p.name)
        if not m:
            continue
        out[m.group(1)] = p
    return out


def pair_frames(rgb_dir, depth_dir):
    """
    Match rgb/ and depth/ files by frame-number stem. Refuses to silently
    drop frames: if either side has a frame number the other lacks, this
    is treated as a data problem and raises rather than guessing.
    """
    rgb_frames = index_frames(rgb_dir)
    depth_frames = index_frames(depth_dir)

    rgb_only = sorted(set(rgb_frames) - set(depth_frames), key=int)
    depth_only = sorted(set(depth_frames) - set(rgb_frames), key=int)
    if rgb_only or depth_only:
        raise SystemExit(
            f"Frame mismatch between {rgb_dir} and {depth_dir}: "
            f"{len(rgb_only)} rgb-only frame(s) {rgb_only[:10]}, "
            f"{len(depth_only)} depth-only frame(s) {depth_only[:10]}"
        )

    common = sorted(rgb_frames.keys(), key=int)
    return [(n, rgb_frames[n], depth_frames[n]) for n in common]


# ---------------------------------------------------------------------------
# Depth conversion + deprojection
# ---------------------------------------------------------------------------

def deproject_pixel_offline(depth_raw, depth_scale, x, y, patch=2):
    """
    Convert a 2D pixel (x, y) to a 3D camera-space point in meters, using a
    small median-filtered patch around the pixel (URFD depth PNGs have
    holes/zeros like any structured-light sensor).
    Returns None if no valid depth is found in the patch.
    """
    h, w = depth_raw.shape[:2]
    xi, yi = int(round(x)), int(round(y))

    x0, x1 = max(0, xi - patch), min(w, xi + patch + 1)
    y0, y1 = max(0, yi - patch), min(h, yi + patch + 1)
    window = depth_raw[y0:y1, x0:x1]
    valid = window[window > 0]
    if valid.size == 0:
        return None

    raw_median = float(np.median(valid))
    depth_mm = depth_scale * raw_median / 65535.0
    depth_m = depth_mm / 1000.0
    if depth_m <= 0:
        return None

    X = (x - KINECT_CX) * depth_m / KINECT_FX
    Y = (y - KINECT_CY) * depth_m / KINECT_FY
    Z = depth_m
    return [float(X), float(Y), float(Z)]


# ---------------------------------------------------------------------------
# Link geometry computation (identical to pose_link_logger.py)
# ---------------------------------------------------------------------------

def compute_link_geometry(kp3d, kp_conf, a_name, b_name, conf_thresh=0.3):
    pa, pb = kp3d.get(a_name), kp3d.get(b_name)
    ca, cb = kp_conf.get(a_name, 0.0), kp_conf.get(b_name, 0.0)

    if pa is None or pb is None or ca < conf_thresh or cb < conf_thresh:
        return None

    vec = np.array(pb) - np.array(pa)
    length = float(np.linalg.norm(vec))
    if length == 0:
        return None

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
    parser.add_argument("--input-dir", required=True,
                         help="Sequence directory containing rgb/ and depth/ subfolders")
    parser.add_argument("--depth-scale", type=float, required=True,
                         help="URFD per-camera depth scale constant "
                              "(depth_mm = depth_scale * raw / 65535)")
    parser.add_argument("--label", required=True,
                         help="Ground-truth activity/session tag recorded as metadata "
                              "(not derived from geometry)")
    parser.add_argument("--model", default="yolo11n-pose.pt",
                         help="Ultralytics pose model")
    parser.add_argument("--output", required=True,
                         help="Path to append JSONL frame records to")
    parser.add_argument("--conf", type=float, default=0.3,
                         help="Per-keypoint confidence threshold for link computation")
    parser.add_argument("--fps", type=float, default=30.0,
                         help="Nominal capture fps, used only to synthesize a timestamp")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    rgb_dir = input_dir / "rgb"
    depth_dir = input_dir / "depth"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise SystemExit(f"Expected {rgb_dir} and {depth_dir} to both exist")

    pairs = pair_frames(rgb_dir, depth_dir)
    print(f"Matched {len(pairs)} rgb/depth frame pairs in {input_dir}")
    if not pairs:
        raise SystemExit("No matching frame pairs found -- aborting")

    model = YOLO(args.model)

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames_with_detections = 0
    records_written = 0

    with out_path.open("a") as f:
        for frame_id, (frame_num, rgb_path, depth_path) in enumerate(pairs):
            color_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if color_image is None or depth_raw is None:
                print(f"WARNING: failed to read frame {frame_num} "
                      f"({rgb_path.name} / {depth_path.name}) -- skipping")
                continue

            results = model(color_image, verbose=False)[0]
            timestamp = frame_id / args.fps

            if results.keypoints is None or len(results.keypoints) == 0:
                continue

            frames_with_detections += 1

            for person_idx, kps in enumerate(results.keypoints):
                xy = kps.xy[0].cpu().numpy()
                conf = kps.conf[0].cpu().numpy() if kps.conf is not None \
                    else np.ones(len(xy))

                kp2d, kp3d, kp_conf = {}, {}, {}
                for i, name in enumerate(KEYPOINT_NAMES):
                    x, y = xy[i]
                    c = float(conf[i])
                    kp_conf[name] = c
                    kp2d[name] = [float(x), float(y), c]

                    if c >= args.conf and x > 0 and y > 0:
                        point = deproject_pixel_offline(depth_raw, args.depth_scale, x, y)
                        kp3d[name] = point
                    else:
                        kp3d[name] = None

                link_geometry = {}
                for link_name, (a, b) in LINKS.items():
                    geom = compute_link_geometry(
                        kp3d, kp_conf, a, b, conf_thresh=args.conf
                    )
                    link_geometry[link_name] = geom

                record = {
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "source_frame": frame_num,
                    "label": args.label,
                    "person_index": person_idx,
                    "keypoints_2d": kp2d,
                    "keypoints_3d_m": kp3d,
                    "links": link_geometry,
                }
                f.write(json.dumps(record) + "\n")
                records_written += 1

    print(f"Frames processed: {len(pairs)}")
    print(f"Frames with >=1 person detected: {frames_with_detections}")
    print(f"Records written: {records_written}")
    if frames_with_detections == 0:
        print("WARNING: zero frames had any detections -- check input path/model, "
              "this likely means every frame was silently skipped.", file=sys.stderr)


if __name__ == "__main__":
    main()
