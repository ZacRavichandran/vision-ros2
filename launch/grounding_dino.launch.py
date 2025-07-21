#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory and set default weights path
    package_share_directory = get_package_share_directory('vision_ros2')
    default_weights_path = os.path.join(package_share_directory, 'weights', 'groundingdino_swint_ogc.pth')
    default_config_path = os.path.join(package_share_directory, 'resource', 'GroundingDINO_SwinT_OGC.py')


    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='perception',
        description='Node namespace'
    )
    
    # Declare launch arguments
    weights_arg = DeclareLaunchArgument(
        'weights',
        default_value=default_weights_path,
        description='Path to the GroundingDINO model weights file'
    )
    
    confidence_arg = DeclareLaunchArgument(
        'confidence',

        default_value='0.3',
        description='Confidence threshold for detection (default: 0.3)'
    )

    config_arg = DeclareLaunchArgument(
        'config',
        default_value=default_config_path,
        description='config path'
    )


    input_rgb_topic_arg = DeclareLaunchArgument(
        'input_rgb_topic',
        default_value='/zed2i/zed_node/rgb/image_rect_color',
        description='Input image topic'
    )
    
    input_depth_topic_arg = DeclareLaunchArgument(
        'input_depth_topic', 
        default_value='/zed2i/zed_node/depth/depth_registered',
        description='depth'
    )

    input_camera_info_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/zed2i/zed_node/depth/camera_info',
        description='camera info'
    )

    # Node configuration
    grounding_dino_node = Node(
        package='vision_ros2', 
        executable='grounding_dino_node',  
        name='grounding_dino_node',
        output='screen',
        namespace=LaunchConfiguration('namespace'),
        parameters=[
            {
                'weights': LaunchConfiguration('weights'),
                'confidence': LaunchConfiguration('confidence'),
                'config': LaunchConfiguration('config'),
            }
        ],
        remappings=[
            # Add any topic remappings here if needed
            ('image_raw', LaunchConfiguration('input_rgb_topic')),
            ('depth_raw', LaunchConfiguration('input_depth_topic')),
            ('camera_info', LaunchConfiguration('camera_info_topic'))
        ]
    )
    
    return LaunchDescription([
        namespace_arg,
        weights_arg,
        confidence_arg,
        config_arg,
        input_rgb_topic_arg,
        input_depth_topic_arg,
        input_camera_info_arg,
        grounding_dino_node,
    ])