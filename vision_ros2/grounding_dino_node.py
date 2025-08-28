#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from vision_ros2.detector_shared import DetectionComponenet
from vision_ros2.grounding_dino import GroundingDinoInfer


class GroundingDinoNode(Node):
    def __init__(self):
        super().__init__("grounding_dino_node")
        self.get_logger().info("GroundingDinoNode constructor started. Deferring heavy initialization.")

        # --- Stage 1: Fast Initialization ---
        # Initialize members to None. Do NOT load models or create components here.
        self._gd_infer = None
        self._detection_componenet = None

        # Declare parameters with default values.
        self.declare_parameter("weights", "")
        self.declare_parameter("confidence", 0.3)
        self.declare_parameter("config", "")

        # Create a one-shot timer to handle the slow setup tasks.
        # The callback will run after the node has started spinning.
        self._init_timer = self.create_timer(0.1, self.deferred_init)
        
        self.get_logger().info("GroundingDinoNode constructor finished. Waiting for deferred init.")

    def deferred_init(self):
        """
        This callback executes after the node is spinning.
        All heavy initialization happens here.
        """
        self.get_logger().info("Deferred initialization started...")
        
        # Cancel the timer so it only runs once.
        if self._init_timer:
            self._init_timer.cancel()

        try:
            # --- Stage 2: Slow Initialization ---
            # 1. Get parameters now that the node is live.
            weights = self.get_parameter("weights").get_parameter_value().string_value
            confidence = self.get_parameter("confidence").get_parameter_value().double_value
            config = self.get_parameter("config").get_parameter_value().string_value

            self.get_logger().info(f"Weights path: {weights}")
            self.get_logger().info(f"Config path: {config}")
            self.get_logger().info("Loading model weights. This may take a moment...")

            # 2. Perform the heavy model loading. This is the blocking call.
            self._gd_infer = GroundingDinoInfer(
                ckpt_path=weights,
                confidence=confidence,
                classes=None, # This can be updated later via a service or topic
                config_path=config,
            )
            self.get_logger().info("Model weights loaded successfully.")

            # 3. Now that the model is loaded, create the component that uses it.
            self._detection_componenet = DetectionComponenet(self, detector=self._gd_infer)
            
            # This will trigger the secondary deferred init inside DetectionComponenet
            # for the RobotBookKeepers, which is the correct pattern.

            # TODO: This is still not ideal. It's better to pass the confidence
            # value to the DetectionComponent's constructor.
            self._detection_componenet._data_config.detector_confidence = confidence
            
            self.get_logger().info("DetectionComponenet created. Initialization complete.")

        except Exception as e:
            self.get_logger().fatal(f"Failed during deferred initialization: {e}",)
            # You might want to shut down if initialization fails
            # rclpy.shutdown()


def main():
    rclpy.init()
    
    try:
        node = GroundingDinoNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        # You should see this message almost instantly now.
        print("Executor is now spinning the GroundingDinoNode...")
        executor.spin()

    except KeyboardInterrupt:
        print("Keyboard interrupt, shutting down.")
    finally:
        if 'executor' in locals() and executor:
            executor.shutdown()
        if 'node' in locals() and node:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
