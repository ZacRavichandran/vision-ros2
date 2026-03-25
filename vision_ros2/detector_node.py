#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from vision_ros2.detector import DetectionComponenet

# from vision_ros2.grounding_dino import GroundingDinoInfer
from vision_ros2.sam3 import SAM3Infer


class GroundingDinoNode(Node):
    def __init__(self):
        super().__init__("detector_node")

        self.declare_parameter("weights", "")
        self.declare_parameter("confidence", 0.3)
        self.declare_parameter("config", "")
        self.declare_parameter("save_sam3_results", False)
        self.declare_parameter("print_sam3_stats", False)

        weights = self.get_parameter("weights").get_parameter_value().string_value
        confidence = self.get_parameter("confidence").get_parameter_value().double_value
        config = self.get_parameter("config").get_parameter_value().string_value
        save_sam3_results = self.get_parameter("save_sam3_results").get_parameter_value().bool_value
        print_sam3_stats = self.get_parameter("print_sam3_stats").get_parameter_value().bool_value


        print(weights)
        print(config)

        # self._gd_infer = GroundingDinoInfer(
        #     ckpt_path=weights,
        #     confidence=confidence,
        #     classes=["chair", "desk"],
        #     config_path=config,
        # )

        self._gd_infer = SAM3Infer(
            confidence=0.3,
            ckpt_path=weights,
            save_sam3_results=save_sam3_results,
            print_stats=print_sam3_stats,
        )

        self._detection_componenet = DetectionComponenet(self, detector=self._gd_infer)

        # TODO not the cleanest way to do this
        self._detection_componenet._data_config.detector_confidence = confidence


def main():
    rclpy.init()
    node = GroundingDinoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
