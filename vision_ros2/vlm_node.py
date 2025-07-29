#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from teaming_msgs.srv import Query

from vision_ros2.utils import decode_img_msg
from vision_ros2.vlm.vlm import VLMWrapper
from rclpy.callback_groups import ReentrantCallbackGroup


class VLMInferNode(Node):
    def __init__(self):
        super().__init__("vlm_node")

        self.declare_parameter("vlm_model", "llava-hf/vip-llava-7b-hf")
        model = self.get_parameter("vlm_model").get_parameter_value().string_value

        self._vlm = VLMWrapper(model)

        self._latest_img = None

        sub_cbk = ReentrantCallbackGroup()
        self._img_sub = self.create_subscription(Image, "~/image_raw", self._img_cbk, 1,
                                                 callback_group=sub_cbk)

        self._query_scene = self.create_service(
            Query, "~/query_scene", self._query_scene
        )

    def _img_cbk(self, img: Image) -> None:
        self._latest_img = decode_img_msg(img)

    def _query_scene(self, query_request, query_response):
        if self._latest_img is None:
            query_response.success = False
            query_response.answer = "unknown"
            return query_response

        query = f"{query_request.query}. And why? Provide a brief explaination with details in 25 words or less."
        self.get_logger().info(f"sending query: {query}")

        msg = self._vlm.open_query(prompt=query, image=self._latest_img)

        parsed = msg.split(query)[-1].strip()
        self.get_logger().info(f"vlm response: {[parsed]}")

        query_response.success = True
        query_response.answer = parsed
        return query_response


def main(args=None):
    rclpy.init(args=args)

    node = VLMInferNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
