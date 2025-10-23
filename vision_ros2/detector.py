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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Header, String
from teaming_msgs.msg import Detection, Track
from teaming_msgs.srv import GetLabels, SetLabels
from vision_msgs.msg import ObjectHypothesisWithPose
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry

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

    camera_frame: str = "camera_optical_frame"

    camera_transform : str = "spot_camera"

    # Detection params
    labels: str = ""
    detector_confidence: float = 0.5
    detection_depth_threshold: float = 15
    detection_depth_scale: float = 1000.0
    detection_publish_deprojection: bool = True
    detection_max_marker_count: int = 1000

    # tracker
    track_distance_thresh: float = 5.0
    tracker_n_dets: int = 10

    detect_period: float = 0.05


    flip_img: bool = False

    scale_depth: bool = False


class DetectionComponenet:
    def __init__(self, parent_node: Node, detector: Detector, labels: List[str] = ""):
        self._detector = detector
        self._img_queue = queue.Queue(maxsize=2)
        self._bridge = cv_bridge.CvBridge()
        self._parent_node = parent_node
        self._data_config = self._load_config()

        self._intrinsics = None
        self._last_depth = None
        self._last_odom = None
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer, self._parent_node
        )
        self._labels = self._parse_labels()
        self._label_set = {}

        self.setting_labels = False
        self._marker_count = 0
        self._total_track_count = 0

        self._tracker = Tracker(
            distance_threshold=self._data_config.track_distance_thresh,
            n_track_thresh=self._data_config.tracker_n_dets,
        )

        self._parent_node.get_logger().info(f"config tracker with thresh: {self._data_config.track_distance_thresh}")
        self._parent_node.get_logger().info(f"Camera Transform is : {self._data_config.camera_transform}")
        self._parent_node.get_logger().info(f"Scale depth is : {self._data_config.scale_depth}")
        self._parent_node.get_logger().info(f"Labels: {self._labels}")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_ALL,
        )

        img_sub_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        # pub / sub / service

        self._info_topic_pub = self._parent_node.create_publisher(
            String, f"~/info", qos_profile
        )

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
            Image, self._data_config.color_sub_topic, self._img_cbk, img_sub_profile
        )
        self._depth_sub = self._parent_node.create_subscription(
            Image, self._data_config.depth_sub_topic, self._depth_cbk, img_sub_profile
        )
        self._depth_info_sub = self._parent_node.create_subscription(
            CameraInfo,
            self._data_config.depth_info_sub_topic,
            self._depth_info_cbk,
            img_sub_profile
        )
        self._odom_sub = self._parent_node.create_subscription(
            Odometry, "/odom", self._odom_cbk, img_sub_profile
        )
        self._set_labels_service = self._parent_node.create_service(
            SetLabels,
            "detector/set_labels",
            self._set_labels_callback,
        )
        self._get_labels_service = self._parent_node.create_service(
            GetLabels,
            "detector/get_labels",
            self._get_labels_callback,
        )

        self._detection_timer = self._parent_node.create_timer(
            self._data_config.detect_period, self._process_queue
        )


    def _get_class_id_from_label(self, label: str) -> int:
        if label not in self._label_set:
            self._label_set[label] = len(self._label_set)

        return self._label_set[label]
    
    def _odom_cbk(self, odom_msg: Odometry) -> None:
        self._last_odom = odom_msg

    def _process_queue(self) -> None:
        """Read from image queue and run detection.

        ROS will drop incoming messages if the subscriber queue is full.
        We want to drop old messages in favor of incoming ones, hence the
        queue logic.
        """
        if not self._img_queue.empty():
            if self._img_queue.qsize():
                img = self._img_queue.get(block=True)
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
        # self._parent_node.get_logger().info("Received image msg!")
        while self._data_config.drop_old_msg and not self._img_queue.empty():
            self._img_queue.get(block=False)
 
        self._img_queue.put(img_msg)

    def _depth_cbk(self, depth_msg: Image) -> None:
        # self._parent_node.get_logger().info("Received depth msg!")
        self._last_depth = depth_msg

    def _depth_info_cbk(self, camera_info: CameraInfo) -> None:
        # self._parent_node.get_logger().info("Received depth info msg!")
        self._intrinsics = camera_info

    def _parse_labels(self) -> List[str]:
        if self._data_config.labels != "":
            labels = self._data_config.labels.split(",")
            labels = [l.strip() for l in labels]
            self.setting_labels = True
            self._detector.set_labels(labels)
            self.setting_labels = False
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
            self._labels = [l.strip().replace("'", '').replace('"', '') for l in request.labels.split(",")]
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
            self._marker_count = 0

    def _pub_tracks(self, tracks: List[Hypothesis]) -> None:
        # delete all previous tracks
        delete_all_marker = Marker()
        delete_all_marker.action = Marker.DELETEALL
        self._track_viz_pub.publish(delete_all_marker)
        self._total_track_count = 0

        for track in tracks:
            track_msg, text_marker = create_marker_msg(
                id=self._total_track_count,
                header=header_from_track(track),
                position=track.pose,
                color=ColorRGBA(r=1.0, g=0.75, b=0.0, a=1.0),
                class_name=track.label,
                publish_text=True,
                scale=1.0,
            )
            self._track_viz_pub.publish(track_msg)
            self._track_viz_pub.publish(text_marker)
            self._total_track_count += 2
            
            if track.is_published == False:
                self._track_pub.publish(to_track_msg(track))
                track.is_published = True

    def _publish_detection_msg(
        self,
        pose: Tuple[float, float, float],
        header: Header,
        label: str,
    ) -> None:
        # self._parent_node.get_logger().info(f"publishing detection: {pose[0]}, {label}")
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

    def _unnormalize_coords(
        self, norm_x: float, norm_y: float, norm_w: float, norm_h: float
    ) -> Tuple[int, int, int, int]:
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
    ) -> Tuple[Tuple[float, float, float], float]:
        """Convert pixel coordinates to 3D point using camera intrinsics"""
        if self._intrinsics is None or self._last_depth is None:
            self._parent_node.get_logger().info(f"Error intrinsics: {self._intrinsics is None}, depth: {self._last_depth is None}")
            return (0, 0, 0), 0

        fx = self._intrinsics.k[0]
        fy = self._intrinsics.k[4]
        cx = self._intrinsics.k[2]
        cy = self._intrinsics.k[5]

        if self._data_config.camera_transform == "spot_camera":
            depth_img = cv_bridge.CvBridge().imgmsg_to_cv2(self._last_depth, "16UC1")
        else:
            depth_img = cv_bridge.CvBridge().imgmsg_to_cv2(self._last_depth, "32FC1")
        
        if self._data_config.scale_depth:
            depth_img = depth_img.astype(np.float32) / self._data_config.detection_depth_scale

        x, y, w, h = self._unnormalize_coords(x, y, w, h)
        
        # self._parent_node.get_logger().info(f"deproject with wh: {w}, {h}")

        x_min = round(max(0, x - w / 2))
        x_max = round(min(self._intrinsics.width - 1, x + w // 2))
        y_min = round(max(0, y - h / 2))
        y_max = round(min(self._intrinsics.height - 1, y + h // 2))

        #self._parent_node.get_logger().info(f"bounds: {x_min}, {x_max}, {y_min}, {y_max}")

        # Extract the region
        region = depth_img[y_min : y_max + 1, x_min : x_max + 1]

        # Get all finite (non-NaN, non-inf) values
        valid_pixels = region[np.isfinite(region)]

        if len(valid_pixels) == 0:
            return (0, 0, 0), 0


        if self._data_config.camera_transform == "spot_camera":
            depth_value = np.percentile(valid_pixels, 96)
        else:
            depth_value = np.mean(valid_pixels)

        #self._parent_node.get_logger().info(f"depth stats: mean: {depth_value}, min: {valid_pixels.min()}, max: {valid_pixels.max()}")

        # Convert to 3D coordinates
        X = (x - cx) * depth_value / fx
        Y = (y - cy) * depth_value / fy
        Z = depth_value

        result_camera_coords = np.array([X, Y, Z])

        if self._last_odom is None:
            self._parent_node.get_logger().error("Latest odometry or transform is not available in deproject detections!!")
            return (0, 0, 0), 0

        # try:
        #     transform_msg = self._tf_buffer.lookup_transform(
        #         self._data_config.target_frame,
        #         self._data_config.camera_frame,
        #         Time(),
        #         timeout=Duration(seconds=1),
        #     )
        # except Exception as ex:  # TODO not good
        #     self._parent_node.get_logger().info(
        #         f"[detector] ERROR cannot lookup transform between: {self._data_config.target_frame} and {self._data_config.camera_frame}"
        #     )
        #     return (0, 0, 0), 0

        # transform = transform_msg.transform

        if self._data_config.camera_transform == "spot_camera":
            # self._parent_node.get_logger().info("Using spot camera transform for deprojection.")
            zed_camera_rot = Rotation.from_quat([
                0.143, 0.812, -0.229, 0.518
            ])
        else:
            # self._parent_node.get_logger().info("Using default camera transform for deprojection.")
            zed_camera_rot = Rotation.from_quat([
                -0.5, 0.5, -0.5, 0.5
            ])

        rot = Rotation.from_quat(
            [
                self._last_odom.pose.pose.orientation.x,
                self._last_odom.pose.pose.orientation.y,
                self._last_odom.pose.pose.orientation.z,
                self._last_odom.pose.pose.orientation.w,
            ]
        )
        trans = np.array(
            [self._last_odom.pose.pose.position.x, 
             self._last_odom.pose.pose.position.y, 
             self._last_odom.pose.pose.position.z]
        )

        result_map = (rot.as_matrix() @ zed_camera_rot.as_matrix() @ result_camera_coords) + trans


        x = result_map[0]
        y = result_map[1]
        z = result_map[2]

        return (x, y, z), depth_value

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

        if self._data_config.flip_img:
            img = img[::-1, ::-1]

        pred_color, classes, boxes, confidences = self._detector.predict(
            img, plot_output=self._data_config.debug
        )

        #self._parent_node.get_logger().info(
        #    f"running dets: with labels: {pred_labels}: {classes}, {confidences}"
        #)

        if self._data_config.debug:
            # pred_color = pred[0].plot()
            color_msg = self._bridge.cv2_to_imgmsg(
                np.array(pred_color), encoding="passthrough"
            )
            color_msg.header = img_msg.header  # TODO do we want this?
            color_msg.encoding = "rgb8"

            self._annotation_pub.publish(color_msg)

        for box, label, conf in zip(boxes, classes, confidences):

            if conf < self._data_config.detector_confidence:
                continue

            box = box.cpu().numpy()

            # self._parent_node.get_logger().info(f"got box: {box}")

            (x, y, z), depth_point = self._deproject_detections(
                box[0],
                box[1],
                w=box[2],
                h=box[3],
                #box[::2].mean(),
                #box[1::2].mean(),
                #w=box[0] - box[2],
                #h=box[1] - box[3],
                time=img_msg.header.stamp,
            )

            # self._parent_node.get_logger().info(f"deproject depth: {depth_point}")

            # dont publish if 0
            if depth_point == 0:
                info_msg = f"Skipping detection: {label}: ({x}, {y}, {z}) with conf: {conf:0.2f}. " \
                    f"{depth_point} is 0"
                msg = String()
                msg.data = info_msg
                self._info_topic_pub.publish(msg)
                continue

            # don't publish detections far from camera
            if depth_point > self._data_config.detection_depth_threshold:
                info_msg = f"Skipping detection: {label}: ({x}, {y}, {z}) with conf: {conf:0.2f}. " \
                    f"{depth_point} out of range {self._data_config.detection_depth_threshold}"
                msg = String()
                msg.data = info_msg
                self._info_topic_pub.publish(msg)
                continue

            msg_stamp = img_msg.header.stamp

            self._tracker.add_detection(
                time=msg_stamp.sec + msg_stamp.nanosec / 1e9,
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
            )
            self._publish_detection_marker(header=img_msg.header, position=(x, y, z))

            self._parent_node.get_logger().info(
                f"Published detection: {label}: ({x:0.2f}, {y:0.2f}, {z:0.2f}) with conf: {conf:0.2f}", throttle_duration_sec=4.0)
