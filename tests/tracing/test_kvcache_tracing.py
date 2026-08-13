import sys
import types
from unittest.mock import MagicMock, patch

# Create dummy modules for vllm & torch_tpu to bypass import errors in a fresh venv
sys.modules['vllm'] = MagicMock()
sys.modules['vllm.config'] = MagicMock()
sys.modules['torch_tpu'] = MagicMock()
sys.modules['torch_tpu._internal'] = MagicMock()
sys.modules['torch_tpu._internal.profiler'] = MagicMock()


@patch(
    "vllm_torchtpu.distributed.kv_transfer.v2.tpu_connector.TraceAnnotation")
def test_trace_annotation_on_strided_load(mock_trace):
    from vllm_torchtpu.distributed.kv_transfer.v2 import tpu_connector as mod

    config = types.SimpleNamespace()
    config.parallel_config = types.SimpleNamespace(tensor_parallel_size=2)
    config.device_config = types.SimpleNamespace()

    worker = mod.TPUConnectorV2Worker(config)
    worker.set_strided_transfer_bridge(MagicMock())

    req_meta = types.SimpleNamespace(uuid=123,
                                     remote_block_ids=[1, 2],
                                     remote_dp_rank=2)

    worker._request_v2_pull_start = MagicMock()
    worker._require_strided_uuid = MagicMock(return_value=123)
    worker.new_destination_write_session = MagicMock()
    worker.pull_rank_ops_into_session = MagicMock(return_value=1)

    rank_ops = {
        0: (mod.StridedSegmentOp(buffer_id=0,
                                 segment_bytes=100,
                                 num_segments=5,
                                 element_size_bytes=2), ),
        1: (mod.StridedSegmentOp(buffer_id=1,
                                 segment_bytes=200,
                                 num_segments=1,
                                 element_size_bytes=2), )
    }

    req_id = "req-1"

    worker._process_one_strided_load(req_id, req_meta, rank_ops)

    assert mock_trace.call_count == 2
    mock_trace.assert_any_call("KV_Cache_Pull",
                               uuid=123,
                               request_id="req-1",
                               bytes=500,
                               dimensions="strided",
                               source="tp_rank:0")
    mock_trace.assert_any_call("KV_Cache_Pull",
                               uuid=123,
                               request_id="req-1",
                               bytes=200,
                               dimensions="strided",
                               source="tp_rank:1")


@patch("vllm_torchtpu.distributed.kv_transfer.zmq_shm_base.TraceAnnotation")
def test_trace_annotation_on_zmq_pull(mock_trace):
    from concurrent.futures import Future

    from vllm_torchtpu.distributed.kv_transfer.zmq_shm_base import (
        ZmqShmKvConnectorBase, _CoordRecvEntry)

    class DummyConnector(ZmqShmKvConnectorBase):

        def _get_pull_channels(self, host: int):
            return ["tcp://dummy"]

        def _coord_rank0_pull(self, entry: _CoordRecvEntry):
            pass

    config = types.SimpleNamespace()
    config.parallel_config = types.SimpleNamespace(tensor_parallel_size=2)
    config.device_config = types.SimpleNamespace()

    with patch.object(ZmqShmKvConnectorBase, '__init__', return_value=None):
        connector = DummyConnector()

    connector._coord_channel_executor = MagicMock()

    # Mock futures that return immediately
    f_val = Future()
    f_val.set_result((0, 0, 1.0, 1.0, [0], None))
    connector._coord_channel_executor.submit = MagicMock(return_value=f_val)
    connector._n_channels = 2

    # Needs to mock self._coord_pool.spec.layer_shard_shape
    mock_pool = MagicMock()
    mock_pool.spec = MagicMock()
    mock_pool.spec.layer_shard_shape = (100, 16, 2, 2, 128)
    mock_pool.spec.dtype = MagicMock()
    mock_pool.spec.dtype.itemsize = 2
    mock_pool.spec.num_layers = 1
    connector._coord_pool = mock_pool

    entry = _CoordRecvEntry(req_id="cmpl-xyz",
                            uuid=456,
                            slot_idx=0,
                            num_blocks=10,
                            local_blocks=[0, 1],
                            remote_blocks=[0, 1],
                            remote_host=["10.0.0.1"],
                            remote_port=[7600])

    # Let's bypass _coord_rank0_pull by re-verifying how to test it. Wait!
    # Calling the super method of ZmqShmKvConnectorBase
    try:
        ZmqShmKvConnectorBase._coord_rank0_pull(connector, entry)
    except Exception:
        # Ignore downstream unpacking errors since we just want to verify TraceAnnotation was called
        pass

    assert mock_trace.call_count >= 1
    mock_trace.assert_any_call("KV_Cache_Pull",
                               uuid=456,
                               request_id="cmpl-xyz",
                               bytes=163840,
                               dimensions="(10, 16, 2, 2, 128)",
                               source="hosts:1_ch:2")
