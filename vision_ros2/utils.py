from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Quaternion

from typing import Sequence

from typing import Union

import dataclasses

import numpy as np
import supervision as sv
from supervision.draw.color import Color, ColorPalette, DEFAULT_COLOR_PALETTE


IDENTITY_QUATERNION = Quaternion(x=0, y=0, z=0, w=1)
BASE_MARKER = Marker()


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
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
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

    if draw_bbox:
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
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