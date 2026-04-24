#!/usr/bin/env python3
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image


def compute_homography(K_left, K_right, q_left_to_body, q_right_to_body):
    R_left = Rotation.from_quat(q_left_to_body).as_matrix()
    R_right = Rotation.from_quat(q_right_to_body).as_matrix()
    R_rel = R_left.T @ R_right
    H = K_left @ R_rel @ np.linalg.inv(K_right)
    H = H / H[2, 2]  # normalize
    return H


class FrontImageStitcherNode(Node):
    def __init__(self):
        super().__init__("front_image_stitcher_node")
        self.declare_parameter("camera_topics", [""])
        self.declare_parameter("output_topic", "front_stitched/image_raw")

        self.declare_parameter("front_left_intrinsics", [0.0] * 9)  # 3x3 K row-major
        self.declare_parameter("front_right_intrinsics", [0.0] * 9)
        self.declare_parameter("front_left_rotation", [0.0] * 4)  # [x, y, z, w]
        self.declare_parameter("front_right_rotation", [0.0] * 4)

        K_left = np.array(
            self.get_parameter("front_left_intrinsics")
            .get_parameter_value()
            .double_array_value
        ).reshape(3, 3)
        K_right = np.array(
            self.get_parameter("front_right_intrinsics")
            .get_parameter_value()
            .double_array_value
        ).reshape(3, 3)
        q_left = (
            self.get_parameter("front_left_rotation")
            .get_parameter_value()
            .double_array_value
        )
        q_right = (
            self.get_parameter("front_right_rotation")
            .get_parameter_value()
            .double_array_value
        )

        self._H = compute_homography(K_left, K_right, q_left, q_right)

        camera_topics = (
            self.get_parameter("camera_topics").get_parameter_value().string_array_value
        )
        output_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )

        if not camera_topics or camera_topics == [""]:
            self.get_logger().error("No camera_topics provided. Shutting down.")
            raise RuntimeError("camera_topics must be set.")

        self.get_logger().info(
            f"Stitching {len(camera_topics)} cameras: {list(camera_topics)}"
        )

        self._bridge = CvBridge()
        self._pub = self.create_publisher(Image, output_topic, 1)

        subs = [message_filters.Subscriber(self, Image, t) for t in camera_topics]

        if len(subs) == 1:
            # Passthrough — no stitching needed, just republish
            self.create_subscription(Image, camera_topics[0], self._passthrough_cbk, 1)
        else:
            self._sync = message_filters.ApproximateTimeSynchronizer(
                subs, queue_size=10, slop=0.5
            )
            self._sync.registerCallback(self._stitch_cbk)

    def _stitch_cbk(self, *msgs: Image) -> None:
        frames = [self._bridge.imgmsg_to_cv2(m, desired_encoding="rgb8") for m in msgs]
        h = min(f.shape[0] for f in frames)
        frames = [f[:h] for f in frames]
        stitched = np.hstack(frames)
        out_msg = self._bridge.cv2_to_imgmsg(stitched, encoding="rgb8")
        out_msg.header = msgs[0].header
        self._pub.publish(out_msg)

    def _passthrough_cbk(self, msg: Image) -> None:
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrontImageStitcherNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
