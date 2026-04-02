#!/usr/bin/env python3

from typing import Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from teaming_msgs.srv import Query

from vision_ros2.utils import decode_img_msg
from vision_ros2.vlm.vlm import VLMWrapper


class VLMInferNode(Node):
    def __init__(self):
        super().__init__("vlm_node")

        self.declare_parameter("vlm_model", "llava-hf/vip-llava-7b-hf")
        self.declare_parameter("flip_img", False)
        self.declare_parameter("color_sub_topic", "image_raw")
        self.declare_parameter("hand_sub_topic", "hand_img_raw")
        self.declare_parameter("postpend","And why? Provide a brief explanation with details in 25 words or less.")
        model = self.get_parameter("vlm_model").get_parameter_value().string_value
        self._flip_img = self.get_parameter("flip_img").get_parameter_value().bool_value
        self._postpend = self.get_parameter("postpend").get_parameter_value().string_value
        img_sub = (
            self.get_parameter("color_sub_topic").get_parameter_value().string_value
        )

        # anything with `hand` is intended for spot hand camera use.
        # This logic is reduntant and temporary
        hand_camera_sub = (
            self.get_parameter("hand_sub_topic").get_parameter_value().string_value
        )

        self._vlm = VLMWrapper(model)

        self._latest_img = None
        self._latest_hand_img = None

        sub_cbk = ReentrantCallbackGroup()
        self._img_sub = self.create_subscription(
            Image, img_sub, self._img_cbk, 1, callback_group=sub_cbk
        )

        self._query_scene_main = self.create_service(
            Query, "~/query_scene", self._query_scene_main
        )

        self._hand_img_sub = self.create_subscription(
            Image, hand_camera_sub, self._hand_img_cbk, 1, callback_group=sub_cbk
        )
        self._query_scene_main = self.create_service(
            Query, "~/query_scene" + hand_camera_sub, self._query_scene_hand
        )

    def _hand_img_cbk(self, img: Image) -> None:
        self._latest_hand_img = decode_img_msg(img)

    def _img_cbk(self, img: Image) -> None:
        self._latest_img = decode_img_msg(img)

        if self._flip_img:
            self._latest_img = self._latest_img[::-1]

    def _query_scene(
        self,
        query_request: Query.Request,
        query_response: Query.Response,
        img_msg: Image,
        postpend: Optional[bool] = True,
    ) -> Query.Response:
        if img_msg is None:
            query_response.success = False
            query_response.answer = "VLM could not recieve image. Response is unknown"
            return query_response

        query = f"{query_request.query}. {self._postpend}"
        self.get_logger().info(f"sending query: {query}")

        msg = self._vlm.open_query(prompt=query, image=img_msg)

        parsed = msg.split(query)[-1].strip()
        self.get_logger().info(f"vlm response: {[parsed]}")

        query_response.success = True
        query_response.answer = parsed
        return query_response

    def _query_scene_main(
        self, query_request: Query.Request, query_response: Query.Response
    ) -> Query.Response:
        return self._query_scene(query_request, query_response, self._latest_img)

    def _query_scene_hand(
        self, query_request: Query.Request, query_response: Query.Response
    ) -> Query.Response:
        return self._query_scene(
            query_request, query_response, self._latest_hand_img, postpend=False
        )


def main(args=None):
    rclpy.init(args=args)

    node = VLMInferNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
