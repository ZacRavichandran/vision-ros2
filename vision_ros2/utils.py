import dataclasses
from typing import Sequence, Union

import cv2
import numpy as np
import supervision as sv
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import ColorRGBA, Header
from supervision.draw.color import DEFAULT_COLOR_PALETTE, Color, ColorPalette
from visualization_msgs.msg import Marker

IDENTITY_QUATERNION = Quaternion(x=0, y=0, z=0, w=1)
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
    annotated_image = label_annotator.annotate(annotated_image, detections=detections, labels=labels)

    if draw_bbox:
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections
        )
    return annotated_image, labels


def create_marker_msg(
    *,
    id: int,
    header: Header,
    position: Sequence[float],
    color: ColorRGBA,
    orientation: Quaternion = IDENTITY_QUATERNION,
    scale: float = 0.25,
    marker_type=BASE_MARKER.SPHERE,
) -> Marker:
    marker_msg = Marker()
    marker_msg.id = id
    marker_msg.header = header
    marker_msg.pose.position = Point(x=position[0], y=position[1], z=position[2])
    marker_msg.pose.orientation = orientation

    marker_msg.color = color
    marker_msg.scale.x = scale
    marker_msg.scale.y = scale
    marker_msg.scale.z = scale
    marker_msg.action = marker_msg.ADD
    marker_msg.type = marker_type

    return marker_msg
