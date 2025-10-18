#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    # Get package share directory and set default weights path
    package_share_directory = get_package_share_directory("vision_ros2")
    default_config_path = os.path.join(
        package_share_directory, "resource", "GroundingDINO_SwinT_OGC.py"
    )

    namespace_arg = DeclareLaunchArgument(
        "namespace", default_value="perception", description="Node namespace"
    )

    # Declare launch arguments
    weights_arg = DeclareLaunchArgument(
        "weights",
        default_value="/home/dcist/data/weights/groundingdino_swint_ogc.pth",
        description="Path to the GroundingDINO model weights file",
    )

    confidence_arg = DeclareLaunchArgument(
        "confidence",
        default_value="0.3", # 0.4 with larger model
        description="Confidence threshold for detection (default: 0.4)",
    )

    tracker_n_dets_arg = DeclareLaunchArgument(
        "tracker_n_dets",
        default_value="5", # 8 also works for small; 5 for larger model
        description="number of detections for a track",
    )

    track_distance_thresh_arg = DeclareLaunchArgument(
        "track_distance_thresh",
        default_value="5.0",
        description="clustering distance"
    )

    config_arg = DeclareLaunchArgument(
        "config", default_value=default_config_path, description="config path"
    )

    input_rgb_topic_arg = DeclareLaunchArgument(
        "input_rgb_topic",
        default_value="zed/zed_node/left/image_rect_color",
        description="Input image topic",
    )

    input_depth_topic_arg = DeclareLaunchArgument(
        "input_depth_topic",
        default_value="zed/zed_node/depth/depth_registered",
        description="depth",
    )

    input_camera_info_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="zed/zed_node/depth/camera_info",
        description="camera info",
    )

    camera_frame_arg = DeclareLaunchArgument(
        "camera_frame",
        default_value="zed_left_camera_optical_frame",
        description="camera frame",
    )
    flip_img_arg = DeclareLaunchArgument(
        "flip_img", default_value="False", description="config path"
    )

    flip_img = LaunchConfiguration("flip_img")

    vision_pkg = get_package_share_directory("vision_ros2")
    static_transforms_launch = PathJoinSubstitution(
        [vision_pkg, "launch", "vision_static_transforms.launch.py"]
    )

    vision_static_ld = IncludeLaunchDescription(static_transforms_launch)

    # Node configuration
    grounding_dino_node = Node(
        package="vision_ros2",
        executable="grounding_dino_node",
        name="grounding_dino_node",
        output="screen",
        parameters=[
            {
                "weights": LaunchConfiguration("weights"),
                "confidence": LaunchConfiguration("confidence"),
                "config": LaunchConfiguration("config"),
                "labels": "",
                "camera_frame": LaunchConfiguration("camera_frame"),
                "tracker_n_dets": LaunchConfiguration("tracker_n_dets"),
                "track_distance_thresh": LaunchConfiguration("track_distance_thresh"),
                "flip_img": flip_img
            }
        ],
        remappings=[
            # Add any topic remappings here if needed
            ("image_raw", LaunchConfiguration("input_rgb_topic")),
            ("depth_raw", LaunchConfiguration("input_depth_topic")),
            ("camera_info", LaunchConfiguration("camera_info_topic")),
        ],
    )

    vlm_node = Node(
        package="vision_ros2",
        executable="vlm_node",
        name="vlm_node",
        output="screen",
        parameters=[{"flip_img": flip_img}],
        remappings=[("~/image_raw", LaunchConfiguration("input_rgb_topic"))],
    )

    return LaunchDescription(
        [
            namespace_arg,
            weights_arg,
            confidence_arg,
            flip_img_arg,
            config_arg,
            input_rgb_topic_arg,
            input_depth_topic_arg,
            input_camera_info_arg,
            camera_frame_arg,
            tracker_n_dets_arg,
            track_distance_thresh_arg,
            vision_static_ld,
            grounding_dino_node,
            vlm_node,
        ]
    )
