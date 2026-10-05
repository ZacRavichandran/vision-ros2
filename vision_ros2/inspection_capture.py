"""Persist the exact RGB input and associated inspection evidence."""
import json
from pathlib import Path
from uuid import uuid4
from PIL import Image
import numpy as np


def save_capture(directory, image, metadata):
    folder=Path(directory).expanduser();folder.mkdir(parents=True,exist_ok=True)
    base=folder/f"{metadata['image_stamp_ns']}_{uuid4().hex}"
    rgb=np.asarray(image)
    Image.fromarray(rgb).save(base.with_suffix('.png'))
    record=dict(metadata,image_path=str(base.with_suffix('.png')),status='captured')
    base.with_suffix('.json').write_text(json.dumps(record,indent=2))
    return base,record


def finish_capture(capture, answer=None, error=None):
    base,record=capture
    record.update(answer=answer,error=error,status='failed' if error else 'answered')
    base.with_suffix('.json').write_text(json.dumps(record,indent=2))
