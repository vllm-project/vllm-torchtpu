from typing import types
from unittest.mock import MagicMock, patch


from tests.distributed.test_tpu_connector_v2_runtime import (RecordingBridge,
                                                             _stub_vllm_config)
from vllm_torchtpu.distributed.kv_transfer.v2 import tpu_connector as mod


@patch(
    "vllm_torchtpu.distributed.kv_transfer.v2.tpu_connector.TraceAnnotation")
def test_trace_annotation_on_strided_load(mock_trace):
    worker = mod.TPUConnectorV2Worker(_stub_vllm_config())
    bridge = RecordingBridge()
    worker.set_strided_transfer_bridge(bridge)

    req_meta = types.SimpleNamespace(uuid=123,
                                     remote_block_ids=[1, 2],
                                     remote_dp_rank=2)

    # Disabling the actual RPC pull start so we don't need real sockets
    worker._request_v2_pull_start = MagicMock()
    worker._require_strided_uuid = MagicMock(return_value=123)

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

    # We don't actually process it, we just want to execute the logic in _process_one_strided_load
    try:
        worker._process_one_strided_load(req_id, req_meta, rank_ops)
    except Exception:
        # Not fully setting up the environment, but that's fine if the trace annotation was called
        pass

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
