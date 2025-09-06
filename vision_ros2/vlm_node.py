#!/usr/bin/env python3

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from teaming_msgs.srv import Query

from vision_ros2.utils import decode_img_msg
from vision_ros2.vlm.vlm import VLMWrapper

from functools import partial

class RobotBookKeeper:

    def __init__(self, parent_vlm_node: Node, robot_id: int, robot_name: str):

        self._parent_vlm_node = parent_vlm_node
        self._robot_id = robot_id
        self._robot_name = robot_name
        self._latest_img = None

        # Subscriber
        book_keeper_cbk = ReentrantCallbackGroup()
        _img_cbk_partial = partial(self._parent_vlm_node._img_cbk, robot_id=self._robot_id)
        self._img_sub = self._parent_vlm_node.create_subscription(
            Image, f"/{self._robot_name}/{self._parent_vlm_node._color_sub_topic}", _img_cbk_partial, 1, callback_group=book_keeper_cbk
        )
        # for _ in range(5):
        #     self._parent_vlm_node.get_logger().info(f"Subscribed to /{self._robot_name}/{self._parent_vlm_node._color_sub_topic}")

        self._parent_vlm_node.get_logger().info(f"RobotBookKeeper for robot {self._robot_name} initialized with robot_id {self._robot_id} for {self._parent_vlm_node._vlm_node_name}")


class VLMInferNode(Node):
    def __init__(self):
        super().__init__("vlm_node")

        self.declare_parameter("vlm_model", "llava-hf/vip-llava-7b-hf")
        model = self.get_parameter("vlm_model").get_parameter_value().string_value

        self.declare_parameter("names_of_robots", ["warty", "wanda", "wilbur", "wendy"])
        robot_names = self.get_parameter("names_of_robots").value

        self.declare_parameter("vlm_color_sub_topic", "multisense_front/color/image_raw")
        self._color_sub_topic = self.get_parameter("vlm_color_sub_topic").value

        self.declare_parameter("vlm_node_name", "vlm_node")
        self._vlm_node_name = self.get_parameter("vlm_node_name").value

        self._vlm = VLMWrapper(model)

        self._robot_book_keepers = [RobotBookKeeper(parent_vlm_node=self, robot_id=robot_id, robot_name=robot_name) for robot_id, robot_name in enumerate(robot_names)]

        # self._latest_img = None

        # sub_cbk = ReentrantCallbackGroup()
        # self._img_sub = self.create_subscription(
        #     Image, "~/image_raw", self._img_cbk, 1, callback_group=sub_cbk
        # )

        query_service_cbk = ReentrantCallbackGroup()

        self._query_scene = self.create_service(
            Query, f"/{self._vlm_node_name}/query_scene", self._query_scene, callback_group=query_service_cbk
        )

        self.get_logger().info(f"{self._vlm_node_name} initialized with service /{self._vlm_node_name}/query_scene")

    def _img_cbk(self, img: Image, robot_id: int) -> None:
        # self._latest_img = decode_img_msg(img)
        self._robot_book_keepers[robot_id]._latest_img = decode_img_msg(img)

    def _query_scene(
        self, query_request: Query.Request, query_response: Query.Response
    ) -> Query.Response:
        
        # if self._latest_img is None:
        #     query_response.success = False
        #     query_response.answer = "VLM could not recieve image. Response is unknown"
        #     return query_response

        # query = f"{query_request.query}. And why? Provide a brief explaination with details in 25 words or less."
        # self.get_logger().info(f"sending query: {query}")

        # msg = self._vlm.open_query(prompt=query, image=self._latest_img)

        # parsed = msg.split(query)[-1].strip()
        # self.get_logger().info(f"vlm response: {[parsed]}")

        # query_response.success = True
        # query_response.answer = parsed
        # return query_response

        query_robot_name = query_request.robot_name.strip("'\"")

        for robot_book in self._robot_book_keepers:
            if query_robot_name == robot_book._robot_name:
                if robot_book._latest_img is None:
                    query_response.success = False
                    query_response.answer = f"VLM could not recieve image from robot {robot_book._robot_name}. Response is unknown"
                    return query_response

                query = f"{query_request.query}. And why? Provide a brief explaination with details in 25 words or less."
                self.get_logger().info(f"sending query: {query} for robot {robot_book._robot_name}")

                msg = self._vlm.open_query(prompt=query, image=robot_book._latest_img)

                parsed = msg.split(query)[-1].strip()
                self.get_logger().info(f"vlm response from robot {robot_book._robot_name}: {[parsed]}")

                query_response.success = True
                query_response.answer = parsed
                return query_response
        
        # Handle case where robot_name is not found
        self.get_logger().info(
            f"Robot name '{query_robot_name}' in request not found in VLM node's list of robots."
        )
        query_response.success = False
        query_response.answer = f"Robot '{query_robot_name}' not recognized by VLM node."
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
