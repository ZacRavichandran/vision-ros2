#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from vision_ros2.detector import DetectionComponenet
# from vision_ros2.grounding_dino import GroundingDinoInfer
from vision_ros2.florence_model import FlorenceModel


class GroundingDinoNode(Node):
    def __init__(self):
        super().__init__("grounding_dino_node")

        # self.declare_parameter("weights", "")
        self.declare_parameter("confidence", 0.3)
        # self.declare_parameter("config", "")

        # weights = self.get_parameter("weights").get_parameter_value().string_value
        confidence = self.get_parameter("confidence").get_parameter_value().double_value
        # config = self.get_parameter("config").get_parameter_value().string_value

        # print(weights)
        # print(config)

        # self._gd_infer = GroundingDinoInfer(
        #     ckpt_path=weights,
        #     confidence=confidence,
        #     classes=["chair", "desk"],
        #     config_path=config,
        # )

        self._florence_infer = FlorenceModel(model_id="microsoft/Florence-2-base", detection_conf=confidence) # TODO(Ankit): Make model_id a parameter

        self._detection_componenet = DetectionComponenet(self, detector=self._florence_infer)

        # TODO not the cleanest way to do this
        # self._detection_componenet._data_config.detector_confidence = confidence


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
