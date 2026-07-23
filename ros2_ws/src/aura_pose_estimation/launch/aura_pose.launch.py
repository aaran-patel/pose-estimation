from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pose_model_path_arg = DeclareLaunchArgument(
        "pose_model_path",
        default_value="/home/aaran/aura_pose_project/yolo11n-pose.pt",
        description="Path to the YOLO pose model weights file.",
    )

    pose_estimation_node = Node(
        package="aura_pose_estimation",
        executable="pose_estimation_node",
        name="pose_estimation_node",
        output="screen",
        parameters=[{
            "pose_model_path": LaunchConfiguration("pose_model_path"),
        }],
    )

    fall_detection_node = Node(
        package="aura_pose_estimation",
        executable="fall_detection_node",
        name="fall_detection_node",
        output="screen",
    )

    fall_visualizer_node = Node(
        package="aura_pose_estimation",
        executable="fall_visualizer_node",
        name="fall_visualizer_node",
        output="screen",
    )

    return LaunchDescription([
        pose_model_path_arg,
        pose_estimation_node,
        fall_detection_node,
        fall_visualizer_node,
    ])
