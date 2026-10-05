#!/usr/bin/env python3

import rclpy
import gc
import time
import json
import torch
from pathlib import Path
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from vision_ros2.inspection_capture import save_capture, finish_capture
from threading import Lock
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
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
        self._parent_vlm_node.get_logger().info(f"RobotBookKeeper for robot {self._robot_name} initialized with robot_id {self._robot_id} for {self._parent_vlm_node._vlm_node_name}")


class VLMInferNode(Node):
    def __init__(self):
        super().__init__("vlm_node")

        self.declare_parameter('inspection_capture_dir', str(Path.home() / 'data/vlm-inspections'))
        self._capture_dir = self.get_parameter('inspection_capture_dir').value
        self._capture_tf = Buffer()
        self._capture_listener = TransformListener(self._capture_tf, self)
        self.declare_parameter("vlm_model", "llava-hf/vip-llava-7b-hf")
        model = self.get_parameter("vlm_model").get_parameter_value().string_value

        self.declare_parameter("vlm_quantization", "4bit")
        quantization = self.get_parameter("vlm_quantization").value

        self.declare_parameter("vlm_device", "cuda")
        device = self.get_parameter("vlm_device").value
        self.declare_parameter('vlm_gpu_memory_fraction', 1.0)
        fraction = float(self.get_parameter('vlm_gpu_memory_fraction').value)
        if not 0 < fraction <= 1:
            raise ValueError('vlm_gpu_memory_fraction must be in (0, 1]')
        if device != 'cpu':
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        self.declare_parameter('vlm_unload_after_query', False)
        self._unload_after_query = self.get_parameter('vlm_unload_after_query').value

        self.declare_parameter("vlm_load_on_demand", True)
        load_on_demand = self.get_parameter("vlm_load_on_demand").value
        self.declare_parameter("vlm_cpu_fallback_on_oom", True)
        self._cpu_fallback_on_oom = self.get_parameter("vlm_cpu_fallback_on_oom").value
        self.declare_parameter("vlm_cpu_threads", 8)
        self._cpu_threads = int(self.get_parameter("vlm_cpu_threads").value)
        if self._cpu_threads < 1:
            raise ValueError("vlm_cpu_threads must be positive")

        self.declare_parameter("names_of_robots", ["warty", "wanda", "wilbur", "wendy"]) #TODO(Ankit): This needs to be param set in launch file
        robot_names = self.get_parameter("names_of_robots").value

        self.declare_parameter("vlm_color_sub_topic", "multisense_front/color/image_raw")
        self._color_sub_topic = self.get_parameter("vlm_color_sub_topic").value

        self.declare_parameter("vlm_node_name", "vlm_node")
        self._vlm_node_name = self.get_parameter("vlm_node_name").value

        self.declare_parameter("vlm_profile", False)
        self._profile = self.get_parameter("vlm_profile").value
        self._vlm_config = (model, quantization, device)
        self._vlm_lock = Lock()
        self._inference_lock = Lock()
        self._vlm = None
        if not load_on_demand:
            self._ensure_vlm()

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
        self._robot_book_keepers[robot_id]._latest_frame = (
            self._robot_book_keepers[robot_id]._latest_img,
            img.header.stamp.sec * 1000000000 + img.header.stamp.nanosec, time.monotonic(), img.header.frame_id)

    def _ensure_vlm(self) -> VLMWrapper:
        with self._vlm_lock:
            if self._vlm is None:
                model, quantization, device = self._vlm_config
                if device == 'cpu':
                    torch.set_num_threads(self._cpu_threads)
                started = time.monotonic()
                self.get_logger().info(
                    "Loading terminal-verification VLM "
                    f"model={model} quantization={quantization} device={device}"
                )
                self._vlm = VLMWrapper(
                    model, quantization=quantization, device=device
                )
                self._vlm.model.profile = self._profile
                self.get_logger().info(f'VLM ready: load_s={time.monotonic()-started:.2f} device={device} cpu_threads={self._cpu_threads}')
        return self._vlm

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
        # self.get_logger().info("----------------------------------------------------------------------")
        # self.get_logger().info("----------------------------------------------------------------------")
        # self.get_logger().info(f"Received query request from robot: {query_request.robot_name} with query: {query_request.query}")
        # self.get_logger().info("----------------------------------------------------------------------")
        # self.get_logger().info("----------------------------------------------------------------------")
        
        if query_request.query.startswith(('SCENE_OBSERVATION_', 'LOCALIZE_CUE_')):
            query_response.success = False
            query_response.answer = 'VLM exploration was removed; use RADIO bearings. Target verification remains available.'
            return query_response
        query_robot_name = query_request.robot_name.strip("'\"")

        for robot_book in self._robot_book_keepers:
            if query_robot_name == robot_book._robot_name:
                if robot_book._latest_img is None:
                    query_response.success = False
                    query_response.answer = f"VLM could not recieve image from robot {robot_book._robot_name}. Response is unknown"
                    return query_response

                capture = None
                image = robot_book._latest_img
                if query_request.query.startswith('MISSION_VERIFICATION_V1\n'):
                    frame = getattr(robot_book, '_latest_frame', None)
                    if (frame is None or time.monotonic() - frame[2] > 2.
                            or abs(self.get_clock().now().nanoseconds - frame[1]) > 2_000_000_000):
                        query_response.success = False
                        query_response.answer = 'Mission verification unavailable: stale image'
                        return query_response
                    image = frame[0]
                    self.get_logger().info('VLM capture ' + json.dumps(dict(
                        image_stamp_ns=frame[1], camera_frame=frame[3] if len(frame) > 3 else None,
                        age_s=time.monotonic()-frame[2])))
                    metadata = dict(image_stamp_ns=frame[1], camera_frame=frame[3],
                                    robot=robot_book._robot_name, query=query_request.query,
                                    target_requirements=query_request.query.split('Verify only these visible target requirements: ')[-1].split('\n')[0],
                                    model=self._vlm_config[0])
                    try:
                        tf = self._capture_tf.lookup_transform(f'{robot_book._robot_name}/odom',
                            f'{robot_book._robot_name}/base_link', Time(nanoseconds=frame[1]))
                        p,q=tf.transform.translation,tf.transform.rotation
                        metadata['capture_pose'] = dict(frame_id=tf.header.frame_id,
                            position=[p.x,p.y,p.z], quaternion_xyzw=[q.x,q.y,q.z,q.w])
                    except Exception as exc:
                        metadata['capture_pose'] = None
                        metadata['pose_error'] = str(exc)
                    try:
                        capture = save_capture(self._capture_dir, image, metadata)
                        self.get_logger().info(f'Inspection image saved: {capture[0]}.png')
                    except Exception as exc:
                        query_response.success = False
                        query_response.answer = f'Inspection capture could not be recorded: {exc}'
                        return query_response
                query = (query_request.query if query_request.query.startswith('MISSION_VERIFICATION_V1\n')
                         else f"{query_request.query}. And why? Provide a brief explaination with details in 25 words or less.")
                self.get_logger().info(f"sending query: {query} for robot {robot_book._robot_name}")

                try:
                    msg = self._infer_serialized(query, image)
                except Exception as exc:
                    if capture:
                        try:
                            finish_capture(capture, error=str(exc))
                        except OSError as error:
                            self.get_logger().error(f'Inspection record failed: {error}')
                    self.get_logger().error(f"VLM inference failed: {exc}")
                    query_response.success = False
                    query_response.answer = f"VLM inference failed: {exc}"
                    return query_response

                parsed = (msg.strip() if query.startswith('MISSION_VERIFICATION_V1\n')
                          else msg.split(query)[-1].strip())
                self.get_logger().info(f"vlm response from robot {robot_book._robot_name}: {[parsed]}")

                if capture:
                    try:
                        finish_capture(capture, answer=parsed)
                    except OSError as exc:
                        query_response.success = False
                        query_response.answer = f'Inspection answer could not be recorded: {exc}'
                        return query_response
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

    def _infer_serialized(self, prompt, image):
        queued = time.monotonic()
        with self._inference_lock:
            self.get_logger().info(f'VLM queue_wait_s={time.monotonic()-queued:.3f}')
            try:
                return self._infer_with_fallback(prompt, image)
            finally:
                if getattr(self, '_unload_after_query', False) and self._vlm_config[2] != 'cpu':
                    self._vlm = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.get_logger().info('Released VLM GPU model after query')

    def _infer_with_fallback(self, prompt, image):
        # Serialize loading AND generation across all robot service callbacks.
        with torch.inference_mode():
            try:
                model = self._ensure_vlm()
                started = time.monotonic()
                self.get_logger().info('VLM generation started')
                answer = model.open_query(prompt=prompt, image=image)
                self.get_logger().info(f'VLM generation finished: generation_s={time.monotonic()-started:.2f}')
                return answer
            except torch.cuda.OutOfMemoryError:
                if not self._cpu_fallback_on_oom or self._vlm_config[2] == "cpu":
                    raise
                self.get_logger().warning(
                    "CUDA OOM: releasing VLM GPU weights and retrying on CPU. "
                    "CPU inference is slower; subsequent queries stay on CPU."
                )
            # Outside the except block so the traceback no longer retains
            # partially loaded weights or generation tensors.
            self._vlm = None
            gc.collect()
            torch.cuda.empty_cache()
            model, quantization, _ = self._vlm_config
            self._vlm_config = (model, quantization, "cpu")
            return self._ensure_vlm().open_query(prompt=prompt, image=image)

def main(args=None):
    rclpy.init(args=args)

    node = VLMInferNode()
    # Inference is already serial. MultiThreadedExecutor's spin/worker contention
    # inflated a 16-token GPU request from 1.84 s to 70.65 s on the simulator host.
    # Subscription depth=1 retains the latest pending image during generation.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
