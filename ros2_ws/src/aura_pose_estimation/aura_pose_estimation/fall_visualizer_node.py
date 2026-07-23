"""
fall_visualizer_node.py

Subscribes to the debug image from pose_estimation_node and the state
from fall_detection_node, overlays the state as on-screen text, and
displays it in a live window. Knows nothing about pose extraction or
fall logic itself -- just displays whatever the other two nodes publish,
same decoupling principle as everything else here.

Subscribes:
    /pose/debug_image  (sensor_msgs/Image)
    /pose/fall_state   (aura_pose_msgs/FallState)

Run:
    ros2 run aura_pose_estimation fall_visualizer_node
Press 'q' in the video window to quit.
"""

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from aura_pose_msgs.msg import FallState


class FallVisualizerNode(Node):
    def __init__(self):
        super().__init__("fall_visualizer_node")

        self.bridge = CvBridge()
        self.latest_state = {}  # person_index -> FallState

        self.image_sub = self.create_subscription(
            Image, "/pose/debug_image", self.on_image, 10
        )
        self.state_sub = self.create_subscription(
            FallState, "/pose/fall_state", self.on_state, 10
        )

        self.get_logger().info("fall_visualizer_node ready -- press 'q' in the window to quit")

    def on_state(self, msg: FallState):
        self.latest_state[msg.person_index] = msg

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if not self.latest_state:
            cv2.putText(frame, "NO PERSON DETECTED", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 3)
        else:
            y_offset = 50
            for person_index, state in self.latest_state.items():
                is_fallen = state.state == "FALL_DETECTED"
                color = (0, 0, 255) if is_fallen else (0, 200, 0)
                label = f"P{person_index}: {state.state}"
                cv2.putText(frame, label, (30, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

                detail = (
                    f"  angle={state.smoothed_torso_angle_deg:.1f}deg  "
                    f"sustained={state.sustained_frames}"
                )
                cv2.putText(frame, detail, (30, y_offset + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 70

        cv2.imshow("fall_visualizer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FallVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
