import math
import queue
from dataclasses import dataclass
from typing import List, Tuple, Union

import cv2
import cv_bridge
import numpy as np
import pyrealsense2 as rs2
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
from geometry_msgs.msg import Point, Quaternion
from PIL import Image as PILImage
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from std_msgs.msg import ColorRGBA, Header
from teaming_msgs.msg import Detection, Track
from teaming_msgs.srv import GetLabels, SetLabels
from tf2_ros import tf2
from vision_msgs.msg import ObjectHypothesisWithPose
from visualization_msgs.msg import Marker

from vision_ros2.tracker import Hypothesis, Tracker, header_from_track, to_track_msg
from vision_ros2.utils import create_marker_msg, decode_img_msg


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

    # TODO  this  should be programatically set
    camera_frame: str = "zed_left_camera_optical_frame"

    # Detection params
    labels: str = ""
    detection_confidence_thresh: float = 0.3
    detection_depth_threshold: float = 7.5
    detection_depth_scale: int = 1000
    detection_publish_deprojection: bool = True
    detection_max_marker_count: int = 1000

    # tracker
    track_distance_thresh: float = 2
    tracker_n_dets: int = 10

    detect_period: float = 1e-3


class DetectionComponenet:
    def __init__(self, parent_node: Node, detector: Detector, labels: List[str] = ""):
        self._detector = detector
        self._img_queue = queue.Queue(maxsize=2)
        self._bridge = cv_bridge.CvBridge()
        self._parent_node = parent_node
        self._data_config = self._load_config()

        self._intrinsics = None
        self._last_depth = None
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer, self._parent_node
        )
        self._labels = self._parse_labels()
        self._label_set = {}

        self.setting_labels = False
        self._marker_count = 0

        self._tracker = Tracker(
            distance_threshold=self._data_config.track_distance_thresh,
            n_track_thresh=self._data_config.tracker_n_dets,
        )

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            # history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # pub / sub / service
        self._detection_viz_pub = self._parent_node.create_publisher(
            Marker, f"~/{self._data_config.detection_viz_3d}", qos_profile
        )
        self._annotation_pub = self._parent_node.create_publisher(
            Image, f"~/{self._data_config.detection_viz_image}", qos_profile
        )
        self._detection_pub = self._parent_node.create_publisher(
            Detection, f"~/{self._data_config.detection_topic}", qos_profile
        )
        self._track_pub = self._parent_node.create_publisher(
            Track, f"~/{self._data_config.track_topic}", qos_profile
        )
        self._track_viz_pub = self._parent_node.create_publisher(
            Marker, f"~/{self._data_config.track_viz_topic}", qos_profile
        )
        self._rgb_sub = self._parent_node.create_subscription(
            Image, self._data_config.color_sub_topic, self._img_cbk, qos_profile
        )
        self._depth_sub = self._parent_node.create_subscription(
            Image, self._data_config.depth_sub_topic, self._depth_cbk, qos_profile
        )
        self._depth_info_sub = self._parent_node.create_subscription(
            CameraInfo,
            self._data_config.depth_info_sub_topic,
            self._depth_info_cbk,
            qos_profile,
        )
        self._set_labels_service = self._parent_node.create_service(
            SetLabels,
            "set_labels",
            self._set_labels_callback,
        )
        self._get_labels_service = self._parent_node.create_service(
            GetLabels,
            "get_labels",
            self._get_labels_callback,
        )

        self._detection_timer = self._parent_node.create_timer(
            self._data_config.detect_period, self._process_queue
        )

    def _get_class_id_from_label(self, label: str):
        if label not in self._label_set:
            self._label_set[label] = len(self._label_set)

        return self._label_set[label]

    def _process_queue(self):
        """Read from image queue and run detection.

        ROS will drop incoming messages if the subscriber queue is full.
        We want to drop old messages in favor of incoming ones, hence the
        queue logic.
        """
        if not self._img_queue.empty():
            if self._img_queue.qsize():
                img = self._img_queue.get(block=True)

                if len(self._labels):
                    self.detect(img)

        tracks = self._tracker.get_tracks()
        self._pub_tracks(tracks)

    def _img_cbk(self, img_msg: Image) -> None:
        """If `self.drop_old_msg` is true, empty the queue before
        placing image so that we segment the most recent one / don't lag.

        Parameters
        ----------
        img_msg : Image
            Input image
        """
        while self._data_config.drop_old_msg and not self._img_queue.empty():
            self._img_queue.get(block=False)

        self._img_queue.put(img_msg)

    def _depth_cbk(self, depth_msg: Image) -> None:
        self._last_depth = depth_msg

    def _depth_info_cbk(self, camera_info: CameraInfo) -> None:
        # from rs2 / show_center_depth.py
        self._intrinsics = camera_info
        # try:
        #     if self._intrinsics:
        #         return
        #     self._intrinsics = rs2.intrinsics()
        #     self._intrinsics.width = camera_info.width
        #     self._intrinsics.height = camera_info.height
        #     self._intrinsics.ppx = camera_info.k[2]
        #     self._intrinsics.ppy = camera_info.k[5]
        #     self._intrinsics.fx = camera_info.k[0]
        #     self._intrinsics.fy = camera_info.k[4]
        #     if camera_info.distortion_model == "plumb_bob":
        #         self._intrinsics.model = rs2.distortion.brown_conrady
        #     elif camera_info.distortion_model == "equidistant":
        #         self._intrinsics.model = rs2.distortion.kannala_brandt4
        #     self._intrinsics.coeffs = [i for i in camera_info.d]

        # except cv_bridge.CvBridgeError as e:
        #     print(e)
        #     return

    def _parse_labels(self):
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
            self._labels = [l.strip() for l in request.labels.split(",")]
            self.setting_labels = True
            self._detector.set_labels(self._labels)
            self.setting_labels = False
            self._parent_node.get_logger().info(f"setting labels to: {self._labels}")
            response.success = True
        except Exception as e:
            self._parent_node.get_logger().error(f"Failed to set labels: {str(e)}")
            response.success = False

        return response

    def _get_labels_callback(
        self, request: GetLabels.Request, response: GetLabels.Response
    ) -> GetLabels.Response:
        response.labels = str(self._labels)
        return response

    def _publish_detection_marker(self, header: Header, position: np.ndarray) -> None:
        marker_msg = create_marker_msg(
            id=self._marker_count,
            header=header,
            position=(position[0], position[1], position[2]),
            scale=0.25,
            color=ColorRGBA(r=0.75, g=0.75, b=0.75, a=1.0),
        )

        self._detection_viz_pub.publish(marker_msg)

        self._marker_count += 1
        if self._marker_count > self._data_config.detection_max_marker_count:
            self._marker_count = 1000

    def _pub_tracks(self, tracks: List[Hypothesis]) -> None:
        for track in tracks:
            track_msg = create_marker_msg(
                id=track.class_id,
                header=header_from_track(track),
                position=track.pose,
                color=ColorRGBA(r=1.0, g=0.75, b=0.0, a=1.0),
            )
            self._track_viz_pub.publish(track_msg)
            self._track_pub.publish(to_track_msg(track))

    def _publish_detection_msg(
        self,
        class_id: int,
        confidence: float,
        pose: Tuple[float, float, float],
        header: Header,
        label: str,
    ) -> None:
        self._parent_node.get_logger().info(f"publishing detection: {pose[0]}, {label}")
        object_msgs = []
        object_msg = ObjectHypothesisWithPose()
        object_msg.pose.pose.position = Point(x=pose[0], y=pose[1], z=pose[2])
        object_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        object_msgs.append(object_msg)
        detection_msg = Detection()
        detection_msg.header = header
        detection_msg.header.frame_id = self._data_config.target_frame
        detection_msg.results = object_msgs
        detection_msg.labels.append(label)

        self._detection_pub.publish(detection_msg)

    def _unnormalize_coords(self, norm_x, norm_y, norm_w, norm_h):
        """
        Convert normalized coordinates (0-1) back to pixel coordinates.

        Args:
            norm_x, norm_y: Normalized center coordinates (0-1)
            norm_w, norm_h: Normalized width/height (0-1)

        Returns:
            tuple: (pixel_x, pixel_y, pixel_width, pixel_height)
        """
        # Get image dimensions
        img_width = self._intrinsics.width  # 640
        img_height = self._intrinsics.height  # 360

        # Convert normalized to pixel coordinates
        pixel_x = int(norm_x * img_width)
        pixel_y = int(norm_y * img_height)
        pixel_width = int(norm_w * img_width)
        pixel_height = int(norm_h * img_height)

        return pixel_x, pixel_y, pixel_width, pixel_height

    def _deproject_detections(
        self, x: np.ndarray, y: np.ndarray, w: np.ndarray, h: np.ndarray, time
    ):
        """Convert pixel coordinates to 3D point using camera intrinsics"""
        if self._intrinsics is None or self._last_depth is None:
            return (0, 0, 0), 0

        fx = self._intrinsics.k[0]
        fy = self._intrinsics.k[4]
        cx = self._intrinsics.k[2]
        cy = self._intrinsics.k[5]

        # depth is given in mm. We convert that to meters
        depth_img = cv_bridge.CvBridge().imgmsg_to_cv2(self._last_depth, "passthrough")
        depth_img = np.array(depth_img, dtype=np.float32)
        # depth_img = decode_img_msg(self._last_depth)

        x, y, h, w = self._unnormalize_coords(x, y, w, h)

        x_min = round(max(0, x - w / 2))
        x_max = round(min(self._intrinsics.width - 1, x + w // 2))
        y_min = round(max(0, y - h / 2))
        y_max = round(min(self._intrinsics.height - 1, y + h // 2))

        # Extract the region
        region = depth_img[y_min : y_max + 1, x_min : x_max + 1]

        # Get all finite (non-NaN, non-inf) values
        valid_pixels = region[np.isfinite(region)]

        if len(valid_pixels) == 0:
            return (0, 0, 0), 0

        depth_value = np.mean(valid_pixels)
        

        # Convert to 3D coordinates
        X = (x - cx) * depth_value / fx
        Y = (y - cy) * depth_value / fy
        Z = depth_value

        result_camera_coords = np.array([X, Y, Z])

        try:
            transform_msg = self._tf_buffer.lookup_transform(
                self._data_config.target_frame,
                self._data_config.camera_frame,
                Time(),
                #self._last_depth.header.stamp,
                timeout=Duration(seconds=1),
            )
        except Exception as ex:  # TODO not good
            self._parent_node.get_logger().info(f"[detector] ERROR cannot lookup transform between: {self._data_config.target_frame} and {self._data_config.camera_frame}")
            return (0, 0, 0), 0
            transform_msg = self._tf_buffer.lookup_transform(
                self._data_config.target_frame,
                self._data_config.camera_frame,
                Time(seconds=0, nanoseconds=0),
            )

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

    def _deprecated_deproject_detections(
        self, x: float, y: float, w: float, h: float, time: float
    ) -> Tuple[Tuple[float, float, float], float]:
        """Get 3D location of a 2D detection from depth

        Parameters
        ----------
        x : float
            X coordinate (center, image space)
        y : float
            Y coordinate (center, image space)
        w : float
            Detection width
        h : float
            Detection height
        time : float
            Time of detection

        Returns
        -------
        Tuple[Tuple[float, float, float], Float]
            - (x, y, z) location in world coordinates
            - value of corresponding depth image
        """

        if self._last_depth == None or self._intrinsics == None:
            return (0, 0, 0), 0

        # depth is given in mm. We convert that to meters
        depth_img = decode_img_msg(self._last_depth)
        depth_img = depth_img / self._data_config.detection_depth_scale

        int_x, int_y = np.int16(x), np.int16(y)

        # take 10% crop around box to reduce noise
        w = np.maximum(w * 0.1, 2).astype(np.int32)
        h = np.maximum(h * 0.1, 2).astype(np.int32)

        depth_point = depth_img[
            int_y - h // 2 : int_y + h // 2, int_x - w // 2 : int_x + w // 2
        ]
        depth_point = depth_point.mean()

        result_camera_coords = rs2.rs2_deproject_pixel_to_point(
            self._intrinsics, (int_x, int_y), depth_point
        )

        result_camera_coords = np.array(result_camera_coords)

        transform_msg = self._tf_buffer.lookup_transform(
            self._data_config.target_frame,
            self._data_config.camera_frame,
            self._parent_node.get_clock().now().to_msg(),
        )

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

        return (x, y, z), depth_point

    def set_labels(self):
        pass

    def detect(self, img_msg: Image) -> None:
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

        pred_labels = self._labels.copy()

        img = decode_img_msg(img_msg)

        pred_color, classes, boxes, confidences = self._detector.predict(
            img, plot_output=self._data_config.debug
        )

        #self._parent_node.get_logger().info(
        #    f"running dets: with labels: {pred_labels}: {classes}, {confidences}"
        #)

        if self._data_config.debug:
            # pred_color = pred[0].plot()
            color_msg = self._bridge.cv2_to_imgmsg(np.array(pred_color), encoding="passthrough")
            color_msg.header = img_msg.header  # TODO do we want this?
            color_msg.encoding = "rgb8"

            self._annotation_pub.publish(color_msg)

        for box, label, conf in zip(boxes, classes, confidences):

            if conf < self._data_config.detection_confidence_thresh:
                continue

            box = box.cpu().numpy()

            (x, y, z), depth_point = self._deproject_detections(
                box[::2].mean(),
                box[1::2].mean(),
                w=box[0] - box[2],
                h=box[1] - box[3],
                time=img_msg.header.stamp,
            )

            self._parent_node.get_logger().info(f"deproject depth: {depth_point}")

            # dont publish if 0
            if depth_point == 0:
                continue

            # don't publish detections far from camera
            if depth_point > self._data_config.detection_depth_threshold:
                continue

            msg_stamp = img_msg.header.stamp

            self._tracker.add_detection(
                time=msg_stamp.sec + msg_stamp.nanosec // 1e9,
                class_id=self._get_class_id_from_label(label),
                score=conf,
                pose=np.array([x, y, z]),
                label=label,
                frame=self._data_config.target_frame,
            )

            self._publish_detection_msg(
                class_id=self._get_class_id_from_label(label),
                confidence=conf,
                pose=(x, y, z),
                header=img_msg.header,
                label=label,
            )
            self._publish_detection_marker(header=img_msg.header, position=(x, y, z))
