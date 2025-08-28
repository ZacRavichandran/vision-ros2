import math
import queue
from dataclasses import dataclass
from typing import List, Tuple

import cv_bridge
import numpy as np
import tf2_ros
from geometry_msgs.msg import Point, Quaternion
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Header
from teaming_msgs.msg import Detection, Track
from teaming_msgs.srv import GetLabels, SetLabels
from vision_msgs.msg import ObjectHypothesisWithPose
from visualization_msgs.msg import Marker

from vision_ros2.tracker import Hypothesis, Tracker, header_from_track, to_track_msg
from vision_ros2.utils import create_marker_msg, decode_img_msg

from functools import partial


class Detector:
    def __init__(self):
        pass

    def set_labels(self) -> bool:
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


@dataclass
class DetectionConfig:
    # Topics
    color_sub_topic: str = "image_raw"
    depth_sub_topic: str = "depth_raw"
    depth_info_sub_topic: str = "camera_info"
    detection_topic: str = "detections"
    track_topic: str = "tracks"

    detection_viz_image: str = "detection_img"
    detection_viz_3d: str = "detections_marker"
    track_viz_topic: str = "track_markers"

    # Behavior
    drop_old_msg: bool = True
    debug: bool = True
    target_frame: str = "map"

    camera_frame: str = "camera_optical_frame"

    # Detection params
    labels: str = ""
    detector_confidence: float = 0.5
    detection_depth_threshold: float = 7.5
    detection_depth_scale: int = 1000
    detection_publish_deprojection: bool = True
    detection_max_marker_count: int = 1000

    # tracker
    track_distance_thresh: float = 2
    tracker_n_dets: int = 10

    # TODO(Ankit): Changed this from 1e-3 to 1e-2
    detect_period: float = 1e-3

    main_node_name: str = "grounding_dino_node"


class RobotBookKeeper:

    def __init__(self, parent_node: Node, robot_id: int, robot_name: str):

        self._robot_id = robot_id
        self._robot_name = robot_name
        self._detection_parent_node = parent_node
        self._img_queue = queue.Queue(maxsize=2)
        self._last_depth = None
        self._intrinsics = None

        mutex_group = MutuallyExclusiveCallbackGroup()

        self.tracker = Tracker(distance_threshold=self._detection_parent_node._data_config.track_distance_thresh,
                               n_track_thresh=self._detection_parent_node._data_config.tracker_n_dets)

        # publishers
        self._detection_viz_pub = self._detection_parent_node._parent_node.create_publisher(
            Marker, f"/{self._robot_name}/{self._detection_parent_node._data_config.main_node_name}/{self._detection_parent_node._data_config.detection_viz_3d}", self._detection_parent_node._qos_profile
        )
        self._annotation_pub = self._detection_parent_node._parent_node.create_publisher(
            Image, f"/{self._robot_name}/{self._detection_parent_node._data_config.main_node_name}/{self._detection_parent_node._data_config.detection_viz_image}", self._detection_parent_node._qos_profile
        )
        self._detection_pub = self._detection_parent_node._parent_node.create_publisher(
            Detection, f"/{self._robot_name}/{self._detection_parent_node._data_config.main_node_name}/{self._detection_parent_node._data_config.detection_topic}", self._detection_parent_node._qos_profile
        )
        self._track_pub = self._detection_parent_node._parent_node.create_publisher(
            Track, f"/{self._robot_name}/{self._detection_parent_node._data_config.main_node_name}/{self._detection_parent_node._data_config.track_topic}", self._detection_parent_node._qos_profile
        )
        self._track_viz_pub = self._detection_parent_node._parent_node.create_publisher(
            Marker, f"/{self._robot_name}/{self._detection_parent_node._data_config.main_node_name}/{self._detection_parent_node._data_config.track_viz_topic}", self._detection_parent_node._qos_profile
        )

        # subscriptions
        _img_cbk_partial = partial(self._detection_parent_node._img_cbk, robot_id=self._robot_id)
        self._rgb_sub = self._detection_parent_node._parent_node.create_subscription(
            Image, f"/{self._robot_name}/{self._detection_parent_node._data_config.color_sub_topic}", _img_cbk_partial, self._detection_parent_node._qos_profile, callback_group=mutex_group
        )

        _depth_cbk_partial = partial(self._detection_parent_node._depth_cbk, robot_id=self._robot_id)
        self._depth_sub = self._detection_parent_node._parent_node.create_subscription(
            Image, f"/{self._robot_name}/{self._detection_parent_node._data_config.depth_sub_topic}", _depth_cbk_partial, self._detection_parent_node._qos_profile, callback_group=mutex_group
        )

        _depth_info_cbk_partial = partial(self._detection_parent_node._depth_info_cbk, robot_id=self._robot_id)
        self._depth_info_sub = self._detection_parent_node._parent_node.create_subscription(
            CameraInfo, f"/{self._robot_name}/{self._detection_parent_node._data_config.depth_info_sub_topic}", _depth_info_cbk_partial, self._detection_parent_node._qos_profile, callback_group=mutex_group
        )

        self._detection_parent_node._parent_node.get_logger().info(f"Initialized book keeper for robot: {self._robot_name} with id: {self._robot_id}")


class DetectionComponenet:
    def __init__(self, parent_node: Node, detector: Detector, labels: List[str] = ""):
        # --- FAST INITIALIZATION ---
        # Only setup non-ROS objects here.
        self._detector = detector
        # self._img_queue = queue.Queue(maxsize=2)
        self._bridge = cv_bridge.CvBridge()
        self._parent_node = parent_node
        self._data_config = self._load_config()

        # Initialize all ROS-related members to None
        self._tf_buffer = None
        self._tf_listener = None
        self._set_labels_service = None
        self._get_labels_service = None
        self._detection_timer = None
        self._init_bookkeepers_timer = None

        self._labels = self._parse_labels()
        self._label_set = {}
        self.setting_labels = False
        self._marker_count = 0
        self._robot_book_keepers = []

        self._qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )

        # Create a one-shot timer to perform all slow/blocking initializations
        self._init_timer = self._parent_node.create_timer(0.1, self.deferred_init)

    def deferred_init(self):
        """
        This callback runs once after the node is spinning.
        All ROS communications objects are created here.
        """
        if self._init_timer:
            self._init_timer.cancel()

        self._parent_node.get_logger().info("DetectionComponenet deferred_init started...")

        # 1. Initialize TF Listener (this is a blocking call)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer, self._parent_node
        )
        self._parent_node.get_logger().info("TF Listener initialized.")

        # 2. Create services
        self._set_labels_service = self._parent_node.create_service(
            SetLabels,
            f"/{self._data_config.main_node_name}/detector/set_labels",
            self._set_labels_callback,
        )
        self._get_labels_service = self._parent_node.create_service(
            GetLabels,
            f"/{self._data_config.main_node_name}/detector/get_labels",
            self._get_labels_callback,
        )
        self._parent_node.get_logger().info("Services initialized.")

        # 3. Create the main processing timer
        processing_cb_group = MutuallyExclusiveCallbackGroup()
        self._detection_timer = self._parent_node.create_timer(
            self._data_config.detect_period, self._process_queue, callback_group=processing_cb_group
        )
        self._parent_node.get_logger().info("Detection timer initialized.")

        # 4. Create the timer to initialize the bookkeepers
        self._init_bookkeepers_timer = self._parent_node.create_timer(0.1, self._init_robot_book_keepers)
        self._parent_node.get_logger().info("DetectionComponenet deferred_init finished.")

    def _init_robot_book_keepers(self):
        if self._init_bookkeepers_timer:
            self._init_bookkeepers_timer.cancel()

        self._parent_node.declare_parameter("names_of_robots", ["warty", "wanda"])
        robot_names = self._parent_node.get_parameter("names_of_robots").value
        self._parent_node.get_logger().info(f"Initializing book keepers for robots: {robot_names}")
        
        self._robot_book_keepers = [RobotBookKeeper(self, robot_id=idx, robot_name=robot_name) for idx, robot_name in enumerate(robot_names)]
        self._parent_node.get_logger().info("RobotBookKeepers initialized.")

    def _get_class_id_from_label(self, label: str) -> int:
        if label not in self._label_set:
            self._label_set[label] = len(self._label_set)

        return self._label_set[label]

    def _process_queue(self) -> None:
        """Read from image queue and run detection.

        ROS will drop incoming messages if the subscriber queue is full.
        We want to drop old messages in favor of incoming ones, hence the
        queue logic.
        """
        # if not self._img_queue.empty():
        #     if self._img_queue.qsize():
        #         # print(f"[detector] processing queue with size: {self._img_queue.qsize()}", flush=True)
        #         img = self._img_queue.get(block=True)

        #         if len(self._labels):
        #             # print(f"[detector] running detection with labels: {self._labels}", flush=True)
        #             self.detect(img)

        # tracks = self._tracker.get_tracks()
        # self._pub_tracks(tracks)

        # self._parent_node.get_logger().info(f"[detector] processing queues for {len(self._robot_book_keepers)} robots", throttle_duration_sec=1.0)

        for robot_id, robot_book in enumerate(self._robot_book_keepers):
            if not robot_book._img_queue.empty():
                if robot_book._img_queue.qsize():
                    # print(f"[detector] processing queue with size: {robot_book._img_queue.qsize()} for robot id: {robot_id}", flush=True)
                    img = robot_book._img_queue.get(block=True)

                    if len(self._labels):
                        # print(f"[detector] running detection with labels: {self._labels}", flush=True)
                        self._parent_node.get_logger().info(f"Trying to detect for robot : {robot_book._robot_name}", throttle_duration_sec=10.0)
                        self.detect(img, robot_book)
                    else:
                        self._parent_node.get_logger().info(f"[detector] WARNING: no labels set for robot: {robot_book._robot_name}, skipping detection", throttle_duration_sec=3.0)
            
            if hasattr(robot_book, 'tracker'):
                tracks = robot_book.tracker.get_tracks()
                self._pub_tracks(tracks, robot_book)

    def _img_cbk(self, img_msg: Image, robot_id: int) -> None:
        """If `self.drop_old_msg` is true, empty the queue before
        placing image so that we segment the most recent one / don't lag.

        Parameters
        ----------
        img_msg : Image
            Input image
        """
        # self._parent_node.get_logger().info(f"[detector] received image for robot: {robot_id}", throttle_duration_sec=1.0)
        # self._parent_node.get_logger().info(f"robot_id inside img_cbk: {robot_id}")
        while self._data_config.drop_old_msg and not self._robot_book_keepers[robot_id]._img_queue.empty():
            self._robot_book_keepers[robot_id]._img_queue.get(block=False)

        self._robot_book_keepers[robot_id]._img_queue.put(img_msg)

    def _depth_cbk(self, depth_msg: Image, robot_id: int) -> None:
        # self._parent_node.get_logger().info(f"[detector] received depth for robot: {robot_id}", throttle_duration_sec=1.0)
        self._robot_book_keepers[robot_id]._last_depth = depth_msg

    def _depth_info_cbk(self, camera_info: CameraInfo, robot_id: int) -> None:
        # self._parent_node.get_logger().info(f"[detector] received camera info for robot: {robot_id}", throttle_duration_sec=1.0)
        # self._parent_node.get_logger().info(f"robot_id inside depth_info_cbk: {robot_id}")
        self._robot_book_keepers[robot_id]._intrinsics = camera_info

    def _parse_labels(self) -> List[str]:
        if self._data_config.labels != "":
            labels = self._data_config.labels.split(",")
            labels = [l.strip() for l in labels]
        else:
            labels = []
        return labels

    def _load_config(self) -> DetectionConfig:

        config = DetectionConfig()

        # Declare and get all parameters in one clean loop
        for field_name, field_type in DetectionConfig.__annotations__.items():
            default_value = getattr(config, field_name)
            self._parent_node.declare_parameter(field_name, default_value)

            param_value = self._parent_node.get_parameter(field_name)
            setattr(config, field_name, param_value.value)

        return config

    def _set_labels_callback(
        self, request: SetLabels.Request, response: SetLabels.Response
    ) -> SetLabels.Response:
        try:
            self._labels = [l.strip().replace("'", '').replace('"', '')
                            for l in request.labels.split(",")]
            self.setting_labels = True
            self._detector.set_labels(self._labels)
            self.setting_labels = False
            self._parent_node.get_logger().info(
                f"setting labels to: {self._labels}")
            response.success = True
        except Exception as e:
            self._parent_node.get_logger().error(
                f"Failed to set labels: {str(e)}")
            response.success = False

        return response

    def _get_labels_callback(
        self, request: GetLabels.Request, response: GetLabels.Response
    ) -> GetLabels.Response:
        response.labels = str(self._labels)
        return response

    def _publish_detection_marker(self, header: Header, position: np.ndarray, robot_book: RobotBookKeeper) -> None:
        marker_msg = create_marker_msg(
            id=self._marker_count,
            header=header,
            position=(position[0], position[1], position[2]),
            scale=0.25,
            color=ColorRGBA(r=0.75, g=0.75, b=0.75, a=1.0),
        )

        robot_book._detection_viz_pub.publish(marker_msg)

        self._marker_count += 1
        if self._marker_count > self._data_config.detection_max_marker_count:
            self._marker_count = 0

    def _pub_tracks(self, tracks: List[Hypothesis], robot_book: RobotBookKeeper) -> None:
        for track in tracks:
            track_msg = create_marker_msg(
                id=track.class_id,
                header=header_from_track(track),
                position=track.pose,
                color=ColorRGBA(r=1.0, g=0.75, b=0.0, a=1.0),
            )
            robot_book._track_viz_pub.publish(track_msg)
            robot_book._track_pub.publish(to_track_msg(track))

    def _publish_detection_msg(
        self,
        pose: Tuple[float, float, float],
        header: Header,
        label: str,
        robot_book: RobotBookKeeper
    ) -> None:
        # self._parent_node.get_logger().info(f"publishing detection: {pose[0]}, {label}")
        object_msgs = []
        object_msg = ObjectHypothesisWithPose()
        object_msg.pose.pose.position = Point(x=pose[0], y=pose[1], z=pose[2])
        object_msg.pose.pose.orientation = Quaternion(
            x=0.0, y=0.0, z=0.0, w=1.0)
        object_msgs.append(object_msg)
        detection_msg = Detection()
        detection_msg.header = header
        detection_msg.header.frame_id = self._data_config.target_frame
        detection_msg.results = object_msgs
        detection_msg.labels.append(label)

        # self._detection_pub.publish(detection_msg)
        robot_book._detection_pub.publish(detection_msg)

    def _unnormalize_coords(
        self, norm_x: float, norm_y: float, norm_w: float, norm_h: float, robot_book: RobotBookKeeper
    ) -> Tuple[int, int, int, int]:
        # Get image dimensions
        img_width = robot_book._intrinsics.width  # 640
        img_height = robot_book._intrinsics.height  # 360

        # Convert normalized to pixel coordinates
        pixel_x = int(norm_x * img_width)
        pixel_y = int(norm_y * img_height)
        pixel_width = int(norm_w * img_width)
        pixel_height = int(norm_h * img_height)

        return pixel_x, pixel_y, pixel_width, pixel_height

    def _deproject_detections(
        self, x: np.ndarray, y: np.ndarray, w: np.ndarray, h: np.ndarray, time, robot_book: RobotBookKeeper
    ) -> Tuple[Tuple[float, float, float], float]:
        """Convert pixel coordinates to 3D point using camera intrinsics"""
        if robot_book._intrinsics is None or robot_book._last_depth is None:
            self._parent_node.get_logger().info(
                f"[detector] WARNING: no camera info or depth image inside deproject function for robot: {robot_book._robot_name}", throttle_duration_sec=2.0
            )
            return (0, 0, 0), 0
        # else:
        #     self._parent_node.get_logger().info(
        #         f"[detector] deprojecting detections for robot: {robot_book._robot_name}"
        #     )

        fx = robot_book._intrinsics.k[0]
        fy = robot_book._intrinsics.k[4]
        cx = robot_book._intrinsics.k[2]
        cy = robot_book._intrinsics.k[5]

        # depth is given in mm. We convert that to meters
        depth_img = cv_bridge.CvBridge().imgmsg_to_cv2(robot_book._last_depth, "passthrough")
        depth_img = np.array(depth_img, dtype=np.float32)
        # depth_img = decode_img_msg(self._last_depth)

        x, y, w, h = self._unnormalize_coords(x, y, w, h, robot_book)

        # self._parent_node.get_logger().info(f"deproject with wh: {w}, {h}")

        x_min = round(max(0, x - w / 2))
        x_max = round(min(robot_book._intrinsics.width - 1, x + w // 2))
        y_min = round(max(0, y - h / 2))
        y_max = round(min(robot_book._intrinsics.height - 1, y + h // 2))

        # self._parent_node.get_logger().info(f"bounds: {x_min}, {x_max}, {y_min}, {y_max}")

        # Extract the region
        region = depth_img[y_min: y_max + 1, x_min: x_max + 1]

        # Get all finite (non-NaN, non-inf) values
        valid_pixels = region[np.isfinite(region)]

        if len(valid_pixels) == 0:
            # self._parent_node.get_logger().info(
            #     f"[detector] WARNING: no valid depth pixels found inside deproject function for robot: {robot_book._robot_name}", throttle_duration_sec=2.0
            # )
            return (0, 0, 0), 0

        depth_value = np.mean(valid_pixels)

        # self._parent_node.get_logger().info(f"depth stats: mean: {depth_value}, min: {valid_pixels.min()}, max: {valid_pixels.max()}")

        # Convert to 3D coordinates
        X = (x - cx) * depth_value / fx
        Y = (y - cy) * depth_value / fy
        Z = depth_value

        result_camera_coords = np.array([X, Y, Z])

        try:
            transform_msg = self._tf_buffer.lookup_transform(
                self._data_config.target_frame,
                f"{robot_book._robot_name}/{self._data_config.camera_frame}",
                Time(),
                timeout=Duration(seconds=1),
            )
        except Exception as ex:  # TODO not good
            self._parent_node.get_logger().info(
                f"[detector] ERROR cannot lookup transform between: {self._data_config.target_frame} and {robot_book._robot_name}/{self._data_config.camera_frame}"
            )
            return (0, 0, 0), 0

        transform = transform_msg.transform

        rot = Rotation.from_quat(
            [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ]
        )
        trans = np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z]
        )

        result_map = rot.as_matrix() @ result_camera_coords + trans

        x = result_map[0]
        y = result_map[1]
        z = result_map[2]

        return (x, y, z), depth_value

    def set_labels(self):
        pass

    def detect(self, img_msg: Image, robot_book: RobotBookKeeper) -> None:
        """Run inference and publish
        - detections
        - visualizations (if requested)

        Parameters
        ----------
        img_msg : Image
            Incoming image message.
        """
        if self.setting_labels:
            return

        # pred_labels = self._labels.copy()

        img = decode_img_msg(img_msg)

        pred_color, classes, boxes, confidences = self._detector.predict(
            img, plot_output=self._data_config.debug
        )

        # self._parent_node.get_logger().info(
        #    f"running dets: with labels: {pred_labels}: {classes}, {confidences}"
        # )

        if self._data_config.debug:
            # pred_color = pred[0].plot()
            color_msg = self._bridge.cv2_to_imgmsg(
                np.array(pred_color), encoding="passthrough"
            )
            color_msg.header = img_msg.header  # TODO do we want this?
            color_msg.encoding = "rgb8"

            # self._annotation_pub.publish(color_msg)
            robot_book._annotation_pub.publish(color_msg)

        for box, label, conf in zip(boxes, classes, confidences):

            if conf < self._data_config.detector_confidence:
                continue

            box = box.cpu().numpy()

            self._parent_node.get_logger().info(
                f"got box: {box} for robot: {robot_book._robot_name}", throttle_duration_sec=10.0)

            (x, y, z), depth_point = self._deproject_detections(
                box[0],
                box[1],
                w=box[2],
                h=box[3],
                # box[::2].mean(),
                # box[1::2].mean(),
                # w=box[0] - box[2],
                # h=box[1] - box[3],
                time=img_msg.header.stamp,
                robot_book=robot_book,
            )

            # self._parent_node.get_logger().info(f"deproject depth: {depth_point}")

            # dont publish if 0
            if depth_point == 0:
                continue

            # don't publish detections far from camera
            if depth_point > self._data_config.detection_depth_threshold:
                continue

            msg_stamp = img_msg.header.stamp

            # self._tracker.add_detection(
            #     time=msg_stamp.sec + msg_stamp.nanosec // 1e9,
            #     class_id=self._get_class_id_from_label(label),
            #     score=conf,
            #     pose=np.array([x, y, z]),
            #     label=label,
            #     frame=self._data_config.target_frame,
            # )

            robot_book.tracker.add_detection(
                time=msg_stamp.sec + msg_stamp.nanosec // 1e9,
                class_id=self._get_class_id_from_label(label),
                score=conf,
                pose=np.array([x, y, z]),
                label=label,
                frame=self._data_config.target_frame,
            )

            self._publish_detection_msg(
                pose=(x, y, z),
                header=img_msg.header,
                label=label,
                robot_book=robot_book
            )
            self._publish_detection_marker(
                header=img_msg.header, position=(x, y, z), robot_book=robot_book)
