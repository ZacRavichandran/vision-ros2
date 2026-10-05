import json
import numpy as np
from PIL import Image
from vision_ros2.inspection_capture import save_capture, finish_capture


def test_exact_rgb_and_evidence_roundtrip(tmp_path):
    pixels=np.array([[[255,0,0],[0,0,255]]],dtype=np.uint8)
    record=save_capture(tmp_path,pixels,dict(image_stamp_ns=123,query='boat',capture_pose=None))
    finish_capture(record,answer='unknown')
    assert np.array_equal(np.asarray(Image.open(record[0].with_suffix('.png'))),pixels)
    metadata=json.loads(record[0].with_suffix('.json').read_text())
    assert metadata['answer']=='unknown' and metadata['capture_pose'] is None
