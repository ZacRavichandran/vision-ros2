#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_share_directory = get_package_share_directory("vision_ros2")
    default_config_path = os.path.join(
        package_share_directory, "resource", "GroundingDINO_SwinT_OGC.py"
    )

    namespace_arg = DeclareLaunchArgument(
        "namespace", default_value="perception", description="Node namespace"
    )

    weights_arg = DeclareLaunchArgument(
        "weights",
        default_value="/home/dcist/data/weights/groundingdino_swint_ogc.pth",
        description="Path to the GroundingDINO model weights file",
    )

    confidence_arg = DeclareLaunchArgument(
        "confidence",
        default_value="0.5",
        description="Confidence threshold for detection (default: 0.4)",
    )

    tracker_n_dets_arg = DeclareLaunchArgument(
        "tracker_n_dets",
        default_value="5",
        description="number of detections for a track",
    )

    track_distance_thresh_arg = DeclareLaunchArgument(
        "track_distance_thresh",
        default_value="5.0",
        description="clustering distance",
    )

    config_arg = DeclareLaunchArgument(
        "config", default_value=default_config_path, description="config path"
    )

    scale_depth_arg = DeclareLaunchArgument(
        "scale_depth", default_value="True", description="depth scaling factor"
    )

    input_rgb_topic_arg = DeclareLaunchArgument(
        "input_rgb_topic",
        default_value="/prometheus/frontleft/color/image_raw",
        description="Input image topic",
    )

    input_depth_topic_arg = DeclareLaunchArgument(
        "input_depth_topic",
        default_value="/prometheus/frontleft/depth/image_rect",
        description="depth",
    )

    input_camera_info_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/prometheus/frontleft/color/camera_info",
        description="camera info",
    )

    camera_frame_arg = DeclareLaunchArgument(
        "camera_frame",
        default_value="prometheus/frontleft",
        description="camera frame",
    )

    flip_img_arg = DeclareLaunchArgument(
        "flip_img", default_value="False", description="config path"
    )

    scale_depth = LaunchConfiguration("scale_depth")

    flip_img = LaunchConfiguration("flip_img")

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
                "labels": "desk,chair",
                "camera_frame": LaunchConfiguration("camera_frame"),
                "tracker_n_dets": LaunchConfiguration("tracker_n_dets"),
                "track_distance_thresh": LaunchConfiguration("track_distance_thresh"),
                "scale_depth": scale_depth,
                "flip_img": flip_img,
            }
        ],
        remappings=[
            ("image_raw", LaunchConfiguration("input_rgb_topic")),
            ("depth_raw", LaunchConfiguration("input_depth_topic")),
            ("camera_info", LaunchConfiguration("camera_info_topic")),
            ("odom", "spot/odom"),
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
            scale_depth_arg,
            flip_img_arg,
            config_arg,
            input_rgb_topic_arg,
            input_depth_topic_arg,
            input_camera_info_arg,
            camera_frame_arg,
            tracker_n_dets_arg,
            track_distance_thresh_arg,
            grounding_dino_node,
            vlm_node,
        ]
    )
