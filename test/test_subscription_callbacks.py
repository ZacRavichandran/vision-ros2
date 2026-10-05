"""Match rclpy's callback signature check without starting a model or DDS."""
import inspect
from types import SimpleNamespace
from unittest.mock import Mock
from vision_ros2.vlm_node import RobotBookKeeper


def test_rgb_and_calibration_callbacks_are_inspectable():
    parent = Mock()
    parent._color_sub_topic = 'color'
    parent._vlm_node_name = 'test'
    parent.get_parameter.return_value = SimpleNamespace(value='test_topic')
    callbacks = []

    def subscribe(typ, topic, callback, *args, **kwargs):
        inspect.signature(callback).bind(object())
        callbacks.append(callback)
        return Mock()

    parent.create_subscription.side_effect = subscribe
    book = RobotBookKeeper(parent, 0, 'warty')
    assert len(callbacks) == 2
    info = object()
    callbacks[1](info)
    assert book._camera_info is info
