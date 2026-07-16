# Project: AURA Greenhouse Pose Estimation — Fall Detection Link Geometry

## Overview
Vision-based fall/hazard detection for elderly farmers in a greenhouse (AURA project,
Maynooth). Current phase: run pose estimation on live + recorded RGB-D data, extract
per-frame skeleton "link" (bone) geometry, and leave it unclassified/unthresholded for
later analysis rather than building fall-detection logic directly on top yet.

## Environment
- Machine: Dell OptiPlex 5000, Ubuntu 24.04.4 LTS (noble), kernel 6.17.0-40-generic
- ROS2 Jazzy installed (`ros-jazzy-desktop`, `ros-dev-tools`) — Jazzy, not Humble, since
  this is 24.04. Humble would be needed separately only for Jetson/JetPack 6 deployment.
- Python venv at `~/venvs/aura`, created with `--system-site-packages` so it can still
  see ROS2's `rclpy`. Activate with:
  ```
  source ~/venvs/aura/bin/activate
  ```
- Installed in venv: `ultralytics`, `opencv-python`, `numpy`, `pandas`, `matplotlib`,
  `pyrealsense2`.
- Camera: Intel RealSense D415, serial 220222066652, connected via USB 3.2. Confirmed
  working via `rs-enumerate-devices`. Streams: color up to 1920x1080@30fps, depth up to
  1280x720@30fps (Z16), IR1/IR2 up to 1920x1080.
- RealSense ROS2 wrapper (`realsense-ros`) NOT yet installed — still pending. Check
  whether it has a released Jazzy binary or needs building from source before assuming
  Humble instructions apply directly.

## Known gotchas (already solved once — don't re-debug from scratch)
- **Secure Boot must stay disabled**, or the DKMS-patched `uvcvideo` kernel module fails
  to load and RealSense shows 0 devices with no obvious error pointing at Secure Boot as
  the cause. If a future BIOS update re-enables it, this will break silently.
- `uvcvideo` may not auto-load on a fresh reboot yet — if `rs-enumerate-devices` comes up
  empty after a reboot, add `uvcvideo` to `/etc/modules` or `/etc/modules-load.d/`.
- **NumPy 2.x / matplotlib ABI mismatch**: the venv uses `--system-site-packages`, so
  ultralytics's matplotlib dependency can fall back to the apt-installed
  `/usr/lib/python3/dist-packages/matplotlib`, which is compiled against NumPy 1.x and
  crashes on import. Fixed by `pip install --upgrade matplotlib` *inside* the venv so it
  shadows the system copy. If this resurfaces, check `scipy` too — same failure mode.
- Editor of choice is **vim**, not nano.

## Project files (all currently in `~`, should probably move into a proper project dir)
- `pose_link_logger.py` — live RealSense capture + YOLO11-pose + per-link 3D geometry,
  logs to JSONL. Deliberately does NOT do fall classification — only raw geometry
  (vector, length, angle-from-vertical, confidence) per named link (COCO-17 topology,
  12 named bones e.g. `left_forearm`, `left_thigh`). This is the core design constraint:
  keep geometry extraction and interpretation/classification decoupled.
- `pose_link_logger_offline.py` — same output schema, but reads pre-recorded RGB+depth
  PNG sequences (built for the URFD dataset format) instead of a live camera. Depth
  scale conversion per URFD's documented formula; uses APPROXIMATE Kinect v1 default
  intrinsics (fx=fy=525, cx=319.5, cy=239.5 @ 640x480) since no exact calibration exists
  for that capture rig — treat absolute lengths from this script as approximate, angle
  *patterns* as the more trustworthy signal.
- `inspect_links.py` — loads one or more session JSONL files, reports per-link length
  stability (mean/std — high std flags noisy links), plots torso angle-from-vertical
  across sessions for comparison.

## Data collected so far
Sessions in `~/pose_sessions/`: `session_standing`, `session_walking`, `session_bend`,
`session_sitting`, `session_fall_sim` (own D415 recordings, single subject, one room).

## Findings from first analysis pass (see chat log for full detail — not yet in a file)
- **`hips` link is unreliable everywhere** — std exceeds the mean in `session_walking`.
  Left/right hip keypoints are close together in-frame, so small pixel errors become
  large relative errors in the vector between them. Treat as low-confidence / exclude
  from downstream analysis, or weight it down.
- **`session_fall_sim` is the noisiest session across nearly every link.** Likely cause:
  RealSense depth precision is much better across the image plane (X/Y) than along the
  depth axis (Z); the recorded fall was toward/away from the camera, putting most body
  length along the noisy Z-axis. UNTESTED HYPOTHESIS — next step is to redo the fall
  capture with the subject falling sideways across the frame and compare noise.
- **`session_standing` had far fewer thigh/knee observations than shoulder/arm
  observations** (~30 vs ~1000), despite presumably similar frame counts. Cause not yet
  isolated — could be confidence threshold (0.3) rejecting hip/knee keypoints when legs
  are straight, or the lower body being partially out of frame. Next step: rerun with
  `--conf` lowered (~0.2) to see if counts recover (confirms threshold vs framing).
- Have not yet visually reviewed `torso_angle_comparison.png` output — need to confirm
  whether bend/fall_sim visibly separate from standing/walking in torso angle, or
  whether it's muddier than the length-noise findings suggest it might be.

## Next steps (in rough order)
1. Investigate the two open findings above (fall_sim orientation-vs-noise hypothesis,
   standing thigh/knee dropout cause).
2. Download UR Fall Detection Dataset (URFD) — https://fenix.ur.edu.pl/~mkepski/ds/uf.html
   — free, RGB+depth, includes labeled fall and ADL (walking/sitting/crouching/lying)
   sequences. Run `pose_link_logger_offline.py` on a few sequences per class.
3. Merge own D415 sessions with URFD output through `inspect_links.py` and see whether
   the angle-based separation between fall/non-fall holds up on external data, not just
   the 5 self-recorded sessions.
4. Once geometry is validated as trustworthy: consider wrapping `pose_link_logger.py` as
   a ROS2 node (publishing keypoints/links as a message) instead of JSONL, to integrate
   with the rest of the AURA pipeline — this depends on the still-pending realsense-ros
   wrapper install (see Environment section).
5. Longer-term/separate track: reimplementing RTMPose (SimCC-based, Apache-2.0) was
   researched and selected as a good target for understanding modern pose estimation
   theory more deeply — lower priority than the above, not blocking current work.

## Design principle to preserve
Keep geometry extraction (pose model -> keypoints -> link vectors/lengths/angles) fully
decoupled from interpretation (fall classification, thresholds, risk scoring). Any fall
logic should sit downstream of this data, consuming it, not embedded inside the logger
scripts themselves.
