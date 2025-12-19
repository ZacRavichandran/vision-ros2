import dataclasses
from typing import Sequence, Union

import cv2
import cv_bridge
import numpy as np
import supervision as sv
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import ColorRGBA, Header
from supervision.draw.color import DEFAULT_COLOR_PALETTE, Color, ColorPalette
from visualization_msgs.msg import Marker

IDENTITY_QUATERNION = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
BASE_MARKER = Marker()


def decode_img_msg(msg: Union[Image, CompressedImage]) -> np.ndarray:
    """Decode ROS image message.

    This implements functionality of cv_bridge (was having issues with some
    dependencies. This was the simplest solution).

    Parameters
    ----------
    msg : Union[ImageMsg, CompressedImgMsg]
        Incoming ROS image message.

    Returns
    -------
    np.ndarray
        Decoded image as numpy array.

    Raises
    ------
    ValueError
        Raises error if image encoding is not implemented.
    """
    if "Compressed" in str(type(msg)):  # better way?
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    else:
        if msg.encoding == "rgb8":
            img = np.copy(
                np.ndarray(
                    shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data
                )
            )
        elif msg.encoding == "bgra8":
            img = np.copy(
                np.ndarray(
                    shape=(msg.height, msg.width, 4), dtype=np.uint8, buffer=msg.data
                )
            )[:, :, :3]
            img = img[..., ::-1]  # swap r and b channels
        elif msg.encoding == "rgba8":
            img = np.copy(
                np.ndarray(
                    shape=(msg.height, msg.width, 4), dtype=np.uint8, buffer=msg.data
                )
            )
            img = img[..., :3]
        elif msg.encoding == "32FC1":
            dtype = np.dtype("float32")
            dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
            img = np.ndarray(
                shape=(msg.height, msg.width), dtype=dtype, buffer=msg.data
            ).copy()
        elif msg.encoding == "16UC1":
            dtype = np.dtype("uint16")
            dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
            img = np.ndarray(
                shape=(msg.height, msg.width), dtype=dtype, buffer=msg.data
            )

        else:
            raise ValueError(f"{msg.encoding} not supported")
    return img


# should go in utils
def vis_result_fast(
    image: np.ndarray,
    detections: sv.Detections,
    classes: list[str],
    color: Union[Color, ColorPalette] = DEFAULT_COLOR_PALETTE,
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> np.ndarray:
    """
    Annotate the image with the detection results.
    This is fast but of the same resolution of the input image, thus can be blurry.

    Taken from conceptgraphs code.
    """
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        color=color,
    )
    label_annotator = sv.LabelAnnotator(text_scale=0.3)
    mask_annotator = sv.MaskAnnotator(color=color)

    if hasattr(detections, "confidence") and hasattr(detections, "class_id"):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None:
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for confidence, class_id in zip(confidences, class_ids)
            ]
        else:
            labels = [f"{classes[class_id]}" for class_id in class_ids]
    else:
        print(
            "Detections object does not have 'confidence' or 'class_id' attributes or one of them is missing."
        )

    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(
        annotated_image, detections=detections, labels=labels
    )

    if draw_bbox:
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections
        )
    return annotated_image, labels


def debug_marker_types(marker):
    """Debug ALL fields that need to be integers"""
    print(f"=== Marker ID {marker.id} Debug ===")

    # Basic fields (you already check these)
    print(f"marker.id: {type(marker.id)} = {marker.id}")
    print(f"marker.type: {type(marker.type)} = {marker.type}")
    print(f"marker.action: {type(marker.action)} = {marker.action}")

    # Timestamp fields
    print(f"stamp.sec: {type(marker.header.stamp.sec)} = {marker.header.stamp.sec}")
    print(
        f"stamp.nanosec: {type(marker.header.stamp.nanosec)} = {marker.header.stamp.nanosec}"
    )

    # Lifetime fields
    print(f"lifetime.sec: {type(marker.lifetime.sec)} = {marker.lifetime.sec}")
    print(
        f"lifetime.nanosec: {type(marker.lifetime.nanosec)} = {marker.lifetime.nanosec}"
    )

    # Scale fields (these are often the culprit!)
    print(f"scale.x: {type(marker.scale.x)} = {marker.scale.x}")
    print(f"scale.y: {type(marker.scale.y)} = {marker.scale.y}")
    print(f"scale.z: {type(marker.scale.z)} = {marker.scale.z}")

    # Pose position
    print(f"pose.position.x: {type(marker.pose.position.x)} = {marker.pose.position.x}")
    print(f"pose.position.y: {type(marker.pose.position.y)} = {marker.pose.position.y}")
    print(f"pose.position.z: {type(marker.pose.position.z)} = {marker.pose.position.z}")

    # Pose orientation
    print(
        f"pose.orientation.x: {type(marker.pose.orientation.x)} = {marker.pose.orientation.x}"
    )
    print(
        f"pose.orientation.y: {type(marker.pose.orientation.y)} = {marker.pose.orientation.y}"
    )
    print(
        f"pose.orientation.z: {type(marker.pose.orientation.z)} = {marker.pose.orientation.z}"
    )
    print(
        f"pose.orientation.w: {type(marker.pose.orientation.w)} = {marker.pose.orientation.w}"
    )

    # Color fields
    print(f"color.r: {type(marker.color.r)} = {marker.color.r}")
    print(f"color.g: {type(marker.color.g)} = {marker.color.g}")
    print(f"color.b: {type(marker.color.b)} = {marker.color.b}")
    print(f"color.a: {type(marker.color.a)} = {marker.color.a}")

    print("=== End Debug ===\n")


def create_marker_msg(
    *,
    id: int,
    header: Header,
    position: Sequence[float],
    color: ColorRGBA,
    orientation: Quaternion = IDENTITY_QUATERNION,
    scale: float = 0.25,
    marker_type=BASE_MARKER.SPHERE,
    publish_text: bool = False,
    class_name: str = ""
) -> Marker:
    marker_msg = Marker()
    marker_msg.id = int(id)
    marker_msg.header = header
    marker_msg.pose.position = Point(
        x=float(position[0]), y=float(position[1]), z=float(position[2])
    )

    # getting typing issues
    marker_msg.pose.orientation.x = float(orientation.x)
    marker_msg.pose.orientation.y = float(orientation.y)
    marker_msg.pose.orientation.z = float(orientation.z)
    marker_msg.pose.orientation.w = float(orientation.w)

    marker_msg.color = color
    marker_msg.scale.x = scale
    marker_msg.scale.y = scale
    marker_msg.scale.z = scale
    marker_msg.action = marker_msg.ADD
    marker_msg.type = marker_type

    if publish_text and class_name:
        text_marker = Marker()
        text_marker.id = int(id) + 1
        text_marker.header = header
        text_marker.text = class_name
        text_marker.pose.position = Point(
            x=float(position[0]), y=float(position[1]), z=float(position[2]) + 1.0
        )
        text_marker.pose.orientation = marker_msg.pose.orientation
        text_marker.color = color
        text_marker.scale.z = scale
        text_marker.action = marker_msg.ADD
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.lifetime.sec = 0

        return marker_msg, text_marker
    else:
        return marker_msg
