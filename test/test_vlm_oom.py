from threading import Lock
from unittest.mock import Mock, patch
import torch
from vision_ros2.vlm_node import VLMInferNode


def _node():
    node = VLMInferNode.__new__(VLMInferNode)
    node._inference_lock = Lock()
    node._cpu_fallback_on_oom = True
    node._vlm_config = ('model', '4bit', 'cuda')
    node._vlm = Mock()
    node.get_logger = Mock(return_value=Mock())
    return node


def test_oom_retries_on_cpu_and_stays_there():
    node = _node()
    gpu, cpu = Mock(), Mock()
    gpu.open_query.side_effect = torch.cuda.OutOfMemoryError('full')
    cpu.open_query.return_value = 'Yes'
    node._ensure_vlm = Mock(side_effect=[gpu, cpu])
    with patch('torch.cuda.empty_cache') as clear:
        assert node._infer_serialized('query', None) == 'Yes'
    assert node._vlm_config[2] == 'cpu'
    clear.assert_called_once()


def test_non_oom_failure_is_not_retried():
    node = _node()
    node._ensure_vlm = Mock(side_effect=ValueError('bad image'))
    try:
        node._infer_serialized('query', None)
    except ValueError:
        pass
    else:
        raise AssertionError('failure hidden')
    node._ensure_vlm.assert_called_once()


def test_inference_mode_enabled():
    node = _node()
    wrapper = Mock()
    wrapper.open_query.side_effect = lambda **kw: torch.is_inference_mode_enabled()
    node._ensure_vlm = Mock(return_value=wrapper)
    assert node._infer_serialized('query', None)


def test_gpu_weights_released_after_successful_query():
    node = _node()
    node._unload_after_query = True
    node._ensure_vlm = Mock(return_value=Mock(open_query=Mock(return_value='Yes')))
    with patch('torch.cuda.empty_cache') as clear:
        assert node._infer_serialized('query', None) == 'Yes'
    assert node._vlm is None
    clear.assert_called_once()


def test_gpu_weights_released_after_failed_query():
    node = _node()
    node._unload_after_query = True
    node._ensure_vlm = Mock(side_effect=ValueError('bad image'))
    with patch('torch.cuda.empty_cache') as clear:
        try:node._infer_serialized('query', None)
        except ValueError:pass
        else:raise AssertionError('failure swallowed')
    assert node._vlm is None
    clear.assert_called_once()


def test_structured_query_and_answer_preserved():
    import time
    from types import SimpleNamespace
    node = _node()
    query = 'MISSION_VERIFICATION_V1\nFind a blue truck, no other colors.'
    answer = '{"verdict":"unknown","evidence":"red, blue panels"}'
    frame = object()
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)))
    node._robot_book_keepers = [SimpleNamespace(_robot_name='warty', _latest_img=frame,
        _latest_frame=(frame, 10_000_000_000, time.monotonic()))]
    node._infer_serialized = Mock(return_value=answer)
    response = node._query_scene(SimpleNamespace(query=query, robot_name='warty'), SimpleNamespace())
    assert node._infer_serialized.call_args.args[0] == query
    assert response.answer == answer and response.success


def test_verification_requires_fresh_image_and_preserves_contract():
    import time
    from types import SimpleNamespace as S
    node = _node()
    node.get_clock = Mock(return_value=S(now=lambda:S(nanoseconds=10_000_000_000)))
    frame = object()
    book = S(_robot_name='warty', _latest_img=frame,
             _latest_frame=(frame, 1_000_000_000, time.monotonic()))
    node._robot_book_keepers = [book]
    node._infer_serialized = Mock(return_value='{"visibility":"unknown"}')
    query='MISSION_VERIFICATION_V1\nReturn JSON.'
    result=node._query_scene(S(query=query,robot_name='warty'), S())
    assert not result.success
    node._infer_serialized.assert_not_called()
    book._latest_frame=(frame,10_000_000_000,time.monotonic())
    assert node._query_scene(S(query=query,robot_name='warty'), S()).success
    assert node._infer_serialized.call_args.args==(query,frame)


def test_qwen_token_limit_is_unknown_unless_eos():
    from vision_ros2.vlm.vlm import Qwen25VL
    from types import SimpleNamespace as S
    adapter = Qwen25VL.__new__(Qwen25VL)
    adapter.device, adapter.profile = 'cpu', False
    adapter.processor = Mock(return_value={'input_ids': torch.tensor([[1, 2]])})
    adapter.processor.batch_decode.return_value = ['complete']
    adapter.model = Mock(generation_config=S(eos_token_id=[99]))
    with patch('vision_ros2.vlm.vlm._move_inputs', side_effect=lambda inputs, device: inputs):
        adapter.model.generate.return_value = torch.ones((1, 202), dtype=torch.long)
        try:
            adapter.infer('MISSION_VERIFICATION_V1\n', None)
        except ValueError as exc:
            assert 'unknown' in str(exc)
        else:
            raise AssertionError('Truncated generation accepted')
        adapter.model.generate.return_value[0, -1] = 99
        assert adapter.infer('MISSION_VERIFICATION_V1\n', None) == 'complete'


def test_removed_exploration_requests_do_not_invoke_vlm():
    from types import SimpleNamespace as S
    node = _node()
    node._infer_serialized = Mock()
    for query in ['SCENE_OBSERVATION_V1\nReturn JSON.', 'LOCALIZE_CUE_V1\n{}']:
        response = node._query_scene(S(query=query, robot_name='warty'), S())
        assert not response.success
        assert 'removed' in response.answer
    node._infer_serialized.assert_not_called()
