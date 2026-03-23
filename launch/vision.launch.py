#!/usr/bin/env python3

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    config_file = LaunchConfiguration("config_file").perform(context)

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    detector_node = Node(
        package="vision_ros2",
        executable="detector_node",
        name="detector_node",
        namespace=cfg.get("namespace", ""),
        output="screen",
        parameters=[config_file],
    )

    vlm_node = Node(
        package="vision_ros2",
        executable="vlm_node",
        name="vlm_node",
        namespace=cfg.get("namespace", ""),
        output="screen",
        parameters=[config_file],
    )

    return [detector_node, vlm_node]


def generate_launch_description():
    package_share_directory = get_package_share_directory("vision_ros2")
    default_config = os.path.join(
        package_share_directory, "resource", "vision_ros2_params.yaml"
    )

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Path to the YAML parameter config file",
    )

    return LaunchDescription(
        [
            config_file_arg,
            OpaqueFunction(function=launch_setup),
        ]
    )
