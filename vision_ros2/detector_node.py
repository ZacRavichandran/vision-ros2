#!/usr/bin/env python3

from typing import List

import rclpy
from rclpy.node import Node

from vision_ros2.detector import CameraChannel, CameraChannelConfig, DetectionComponenet

# from vision_ros2.grounding_dino import GroundingDinoInfer
from vision_ros2.sam3 import SAM3Infer


class DetectorNode(Node):
    def __init__(self):
        super().__init__(
            "detector_node",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        weights = self.get_parameter("weights").get_parameter_value().string_value
        confidence = self.get_parameter("confidence").get_parameter_value().double_value
        save_sam3_results = (
            self.get_parameter("save_sam3_results").get_parameter_value().bool_value
        )
        print_sam3_stats = (
            self.get_parameter("print_sam3_stats").get_parameter_value().bool_value
        )

        channels = self.parse_camera_channels()
        self.get_logger().info(f"Got camera: {channels}")

        self._detector = SAM3Infer(
            confidence=0.3,
            ckpt_path=weights,
            save_sam3_results=save_sam3_results,
            print_stats=print_sam3_stats,
        )

        self._detection_componenet = DetectionComponenet(
            self, detector=self._detector, camera_channels=channels
        )

        # TODO not the cleanest way to do this
        self._detection_componenet._data_config.detector_confidence = confidence

    def parse_camera_channels(self: Node) -> List[CameraChannel]:
        # get_parameters_by_prefix returns all params under the prefix as a dict
        # keys are relative: "front_left.color_sub_topic", etc.
        camera_params = self.get_parameters_by_prefix("cameras")

        if not camera_params:
            self.get_logger().warn(
                "No camera channels found under 'cameras.*' namespace!"
            )
            return []

        # Extract unique camera names from keys like "front_left.color_sub_topic"
        camera_names = set()
        for key in camera_params:
            parts = key.split(".")
            if len(parts) >= 2:
                camera_names.add(parts[0])

        channels = []
        for cam_name in sorted(camera_names):
            config = CameraChannelConfig(
                name=cam_name,
                color_sub_topic=camera_params[f"{cam_name}.color_sub_topic"]
                .get_parameter_value()
                .string_value,
                depth_sub_topic=camera_params[f"{cam_name}.depth_sub_topic"]
                .get_parameter_value()
                .string_value,
                depth_info_sub_topic=camera_params[f"{cam_name}.depth_info_sub_topic"]
                .get_parameter_value()
                .string_value,
                camera_frame=camera_params[f"{cam_name}.camera_frame"]
                .get_parameter_value()
                .string_value,
                camera_to_body_rotation=list(
                    camera_params[f"{cam_name}.camera_to_body_rotation"]
                    .get_parameter_value()
                    .double_array_value
                ),
            )

            channels.append(CameraChannel(config))
            self.get_logger().info(f"Loaded camera channel: {channels[-1]}")

        return channels


def main():
    rclpy.init()
    node = DetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
