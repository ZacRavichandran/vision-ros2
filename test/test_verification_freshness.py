import time
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from vision_ros2.vlm_node import VLMInferNode


def test_terminal_verification_rejects_stale_frame_before_inference():
    image=np.zeros((2,2,3),dtype=np.uint8)
    book=SimpleNamespace(_robot_name='warty',_latest_img=image,
                         _latest_frame=(image,1_000_000_000,time.monotonic()-5.,'camera'))
    node=SimpleNamespace(_robot_book_keepers=[book],_infer_serialized=Mock(),
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=6_000_000_000)))
    request=SimpleNamespace(robot_name='warty',query='MISSION_VERIFICATION_V1\nFind a boat')
    response=VLMInferNode._query_scene(node,request,SimpleNamespace())
    assert response.success is False
    assert 'stale image' in response.answer
    node._infer_serialized.assert_not_called()
