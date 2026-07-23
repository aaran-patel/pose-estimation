"""
fall_detection_node.py

Subscribes to /pose/frame, classifies fall state using TWO independent
signals combined, not angle alone:

  1. PRIMARY: hip height above the floor plane (from pose_estimation_node's
     RANSAC floor calibration). Directly measures "is this person actually
     near the ground" -- the strongest, least ambiguous signal available.
  2. CONFIRMATION: smoothed torso angle from vertical. Distinguishes an
     actual fall (low height + flat torso) from someone sitting/crouching
     on the ground deliberately (low height + upright torso) -- height
     alone can't tell those apart, since both put the hip near the floor.

Both are required together for a FALLEN classification. This is a
meaningfully stronger check than angle alone, at the cost of needing the
floor plane calibration to have succeeded at startup.

Subscribes:
    /pose/frame       (aura_pose_msgs/PoseFrame)
Publishes:
    /pose/fall_state  (aura_pose_msgs/FallState)
"""

from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from aura_pose_msgs.msg import PoseFrame, FallState


class FallDetectionNode(Node):
    def __init__(self):
        super().__init__("fall_detection_node")

        self.declare_parameter("hip_height_fallen_m", 0.35)
        self.declare_parameter("torso_confirm_deg", 45.0)  # lower than the
        # angle-only version's 65 deg -- height is now doing most of the
        # work, angle just needs to rule out "sitting upright on the ground"
        self.declare_parameter("sustain_frames", 10)
        self.declare_parameter("standing_grace", 5)
        self.declare_parameter("angle_smooth_window", 5)
        self.declare_parameter("height_smooth_window", 5)

        self.hip_height_fallen_m = self.get_parameter("hip_height_fallen_m").value
        self.torso_confirm_deg = self.get_parameter("torso_confirm_deg").value
        self.sustain_frames = self.get_parameter("sustain_frames").value
        self.standing_grace = self.get_parameter("standing_grace").value
        self.angle_smooth_window = self.get_parameter("angle_smooth_window").value
        self.height_smooth_window = self.get_parameter("height_smooth_window").value

        self.person_state = {}

        self.subscription = self.create_subscription(
            PoseFrame, "/pose/frame", self.on_pose_frame, 10
        )
        self.publisher = self.create_publisher(FallState, "/pose/fall_state", 10)

        self.get_logger().info(
            f"fall_detection_node ready "
            f"(height<{self.hip_height_fallen_m}m AND angle>{self.torso_confirm_deg}deg, "
            f"sustained {self.sustain_frames} frames)"
        )

    def get_state(self, person_index):
        if person_index not in self.person_state:
            self.person_state[person_index] = {
                "angle_history": deque(maxlen=self.angle_smooth_window),
                "height_history": deque(maxlen=self.height_smooth_window),
                "down_frames": 0,
                "up_frames": 0,
                "fall_confirmed": False,
            }
        return self.person_state[person_index]

    def on_pose_frame(self, msg: PoseFrame):
        for person in msg.people:
            state = self.get_state(person.person_index)

            # --- Angle signal (confirmation) ---
            torso_links = [
                link for link in person.links
                if link.name in ("left_torso", "right_torso") and link.valid
            ]
            angle = None
            if torso_links:
                angle = sum(l.angle_from_vertical_deg for l in torso_links) / len(torso_links)
                state["angle_history"].append(angle)
            smoothed_angle = (
                sum(state["angle_history"]) / len(state["angle_history"])
                if state["angle_history"] else None
            )

            # --- Height signal (primary) ---
            if person.hip_height_valid:
                state["height_history"].append(person.hip_height_above_floor_m)
            smoothed_height = (
                sum(state["height_history"]) / len(state["height_history"])
                if state["height_history"] else None
            )

            # --- Combined check: BOTH must agree ---
            height_says_down = (
                smoothed_height is not None and smoothed_height <= self.hip_height_fallen_m
            )
            angle_confirms = (
                smoothed_angle is not None and smoothed_angle >= self.torso_confirm_deg
            )
            is_down_this_frame = height_says_down and angle_confirms

            if is_down_this_frame:
                state["down_frames"] += 1
                state["up_frames"] = 0
            else:
                state["up_frames"] += 1
                if state["up_frames"] > self.standing_grace:
                    state["down_frames"] = 0

            newly_confirmed = False
            if state["down_frames"] >= self.sustain_frames:
                fall_state = "FALL_DETECTED"
                if not state["fall_confirmed"]:
                    state["fall_confirmed"] = True
                    newly_confirmed = True
            else:
                fall_state = "STANDING"
                state["fall_confirmed"] = False

            out_msg = FallState()
            out_msg.header = Header()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.person_index = person.person_index
            out_msg.state = fall_state
            out_msg.smoothed_torso_angle_deg = smoothed_angle if smoothed_angle is not None else -1.0
            out_msg.sustained_frames = state["down_frames"]
            out_msg.newly_confirmed = newly_confirmed

            self.publisher.publish(out_msg)

            if newly_confirmed:
                self.get_logger().warn(
                    f"FALL DETECTED -- person_index={person.person_index}, "
                    f"height={smoothed_height:.2f}m, angle={smoothed_angle:.1f}deg"
                )


def main(args=None):
    rclpy.init(args=args)
    node = FallDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
