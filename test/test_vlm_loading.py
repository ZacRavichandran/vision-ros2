from unittest.mock import patch

import torch

from vision_ros2.vlm.vlm import _model_load_kwargs


def test_four_bit_loading_is_memory_safe():
    with patch("torch.cuda.is_available", return_value=True):
        kwargs = _model_load_kwargs("4bit", "cuda")

    assert kwargs["device_map"] == {"": 0}
    assert kwargs["torch_dtype"] == torch.float16
    assert kwargs["quantization_config"].load_in_4bit


def test_cpu_loading_does_not_use_bitsandbytes():
    kwargs = _model_load_kwargs("4bit", "cpu")

    assert kwargs == {"device_map": "cpu", "torch_dtype": torch.float32}
