"""
pose_estimation_node.py

ROS2 port of pose_link_logger.py. Publishes raw pose + link geometry to a
topic instead of writing JSONL to disk. Deliberately contains NO
classification logic -- same decoupling principle as the rest of this
project, now enforced structurally by ROS2's pub/sub model rather than
just by convention (separate files).

Publishes:
    /pose/frame  (aura_pose_msgs/PoseFrame)

Camera input: uses pyrealsense2 directly rather than subscribing to the
realsense-ros wrapper's topics. This is a deliberate first-pass choice --
see the module-level note near start_realsense() for the tradeoff and when
it's worth switching to the wrapper instead.
"""

import time

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from ultralytics import YOLO

from aura_pose_msgs.msg import Keypoint3D, Link, PersonPose, PoseFrame
from aura_pose_estimation.floor_plane import sample_point_cloud, fit_plane_ransac, height_above_floor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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


def start_realsense(width=1280, height=720, fps=30):
    """
    NOTE ON DESIGN CHOICE: this opens the RealSense pipeline directly via
    pyrealsense2, the same as pose_link_logger.py. That's fine as long as
    this is the ONLY node touching the camera. The moment a second node
    (e.g. a future hazard-detection node) also needs the color/depth
    stream, direct SDK access from two processes will conflict -- that's
    exactly the problem the realsense-ros wrapper solves, by having one
    driver node own the camera and publish topics multiple nodes can
    subscribe to. Worth switching to the wrapper once you have more than
    one node that needs camera frames, not before.
    """
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


def compute_link(kp3d, kp_conf, a_name, b_name, conf_thresh):
    pa, pb = kp3d.get(a_name), kp3d.get(b_name)
    ca, cb = kp_conf.get(a_name, 0.0), kp_conf.get(b_name, 0.0)

    link = Link()
    link.name = ""  # set by caller
    if pa is None or pb is None or ca < conf_thresh or cb < conf_thresh:
        link.valid = False
        return link

    vec = np.array(pb) - np.array(pa)
    length = float(np.linalg.norm(vec))
    if length == 0:
        link.valid = False
        return link

    vertical = np.array([0.0, 1.0, 0.0])
    cos_angle = np.dot(vec, vertical) / (length * np.linalg.norm(vertical))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    link.vector_3d_m = vec.tolist()
    link.length_m = length
    link.angle_from_vertical_deg = angle_deg
    link.confidence = float(min(ca, cb))
    link.valid = True
    return link


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")

        self.declare_parameter("pose_model_path", "yolo11n-pose.pt")
        self.declare_parameter("conf_threshold", 0.3)
        self.declare_parameter("publish_rate_hz", 15.0)

        model_path = self.get_parameter("pose_model_path").value
        self.conf_threshold = self.get_parameter("conf_threshold").value
        publish_rate = self.get_parameter("publish_rate_hz").value

        self.get_logger().info(f"Loading pose model: {model_path}")
        self.model = YOLO(model_path)

        self.get_logger().info("Starting RealSense pipeline...")
        self.pipeline, self.align, self.depth_intrinsics = start_realsense()

        self.get_logger().info(
            "Calibrating floor plane -- keep the floor visible and mostly "
            "clear of people/objects for a moment..."
        )
        self.floor_plane = self._calibrate_floor_plane()
        if self.floor_plane is None:
            self.get_logger().warn(
                "Floor plane calibration failed (not enough of a flat "
                "surface visible). hip_height_above_floor_m will be "
                "reported as invalid until recalibrated."
            )
        else:
            self.get_logger().info("Floor plane calibrated successfully.")

        self.publisher = self.create_publisher(PoseFrame, "/pose/frame", 10)
        self.image_publisher = self.create_publisher(Image, "/pose/debug_image", 10)
        self.cv_bridge = CvBridge()
        self.frame_id = 0

        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.process_frame)

        self.get_logger().info("pose_estimation_node ready, publishing to /pose/frame")

    def _calibrate_floor_plane(self, n_frames=5):
        """
        Averages the plane fit over a few frames at startup for a more
        stable result than a single frame. Assumes the camera is static
        after this point -- if the camera moves, this needs rerunning.
        """
        planes = []
        for _ in range(n_frames):
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            if not depth_frame:
                continue
            points = sample_point_cloud(depth_frame, self.depth_intrinsics)
            plane = fit_plane_ransac(points)
            if plane is not None:
                planes.append(plane)

        if not planes:
            return None

        # Average normals and offsets across successful fits.
        normals = np.array([p[0] for p in planes])
        ds = np.array([p[1] for p in planes])
        avg_normal = normals.mean(axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        avg_d = float(ds.mean())
        return avg_normal, avg_d

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        results = self.model(color_image, verbose=False)[0]

        pose_frame_msg = PoseFrame()
        pose_frame_msg.header = Header()
        pose_frame_msg.header.stamp = self.get_clock().now().to_msg()
        pose_frame_msg.header.frame_id = "camera_color_optical_frame"

        if results.keypoints is not None and len(results.keypoints) > 0:
            for person_idx, kps in enumerate(results.keypoints):
                xy = kps.xy[0].cpu().numpy()
                conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(len(xy))

                kp3d, kp_conf = {}, {}
                keypoint_msgs = []

                for i, name in enumerate(KEYPOINT_NAMES):
                    x, y = xy[i]
                    c = float(conf[i])
                    kp_conf[name] = c

                    kp_msg = Keypoint3D()
                    kp_msg.name = name
                    kp_msg.confidence = c

                    if c >= self.conf_threshold and x > 0 and y > 0:
                        point = deproject_pixel(depth_frame, self.depth_intrinsics, x, y)
                        if point is not None:
                            kp_msg.x, kp_msg.y, kp_msg.z = point
                            kp_msg.valid = True
                            kp3d[name] = point
                        else:
                            kp_msg.valid = False
                            kp3d[name] = None
                    else:
                        kp_msg.valid = False
                        kp3d[name] = None

                    keypoint_msgs.append(kp_msg)

                link_msgs = []
                for link_name, (a, b) in LINKS.items():
                    link_msg = compute_link(kp3d, kp_conf, a, b, self.conf_threshold)
                    link_msg.name = link_name
                    link_msgs.append(link_msg)

                person_msg = PersonPose()
                person_msg.person_index = person_idx
                person_msg.keypoints = keypoint_msgs
                person_msg.links = link_msgs

                left_hip, right_hip = kp3d.get("left_hip"), kp3d.get("right_hip")
                valid_hips = [h for h in (left_hip, right_hip) if h is not None]
                if valid_hips and self.floor_plane is not None:
                    hip_mid = np.mean(valid_hips, axis=0)
                    height = height_above_floor(hip_mid, self.floor_plane)
                    person_msg.hip_height_above_floor_m = height
                    person_msg.hip_height_valid = True
                else:
                    person_msg.hip_height_above_floor_m = -1.0
                    person_msg.hip_height_valid = False

                pose_frame_msg.people.append(person_msg)

        self.publisher.publish(pose_frame_msg)

        annotated = results.plot()  # draws YOLO's skeleton overlay
        img_msg = self.cv_bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        img_msg.header = pose_frame_msg.header
        self.image_publisher.publish(img_msg)

        self.frame_id += 1

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
