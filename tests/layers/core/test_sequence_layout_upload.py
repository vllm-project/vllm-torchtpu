"""localize_token_tensor_and_gather_indices with an upload callable: same result as the
default path, and the host index arrays never go through the synchronous torch.tensor copy."""
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.core import sequence_layout as sl
from vllm_torchtpu.layers.core.pcp_sequence_layout import PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR


def _plan():
    # 2 PCP ranks, interleave 4, one request of 12 tokens padded to 16 globally; this is rank 1's slice.
    order = np.array([0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, -1, -1, -1, -1], dtype=np.int64)  # packed -> request-major
    inv = np.full(12, -1, dtype=np.int64)
    for p, r in enumerate(order):
        if r >= 0:
            inv[r] = p
    return sl.SequenceLayoutPlan(
        descriptor=PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
        token_slice=slice(8, 16), global_num_tokens=12, global_padded_num_tokens=16,
        local_num_tokens=4, local_padded_num_tokens=8,
        _packed_to_request_major_token_indices=order, _request_major_to_packed_token_indices=inv)


def _inputs():
    tokens = torch.arange(100, 112, dtype=torch.long)
    gathers = torch.tensor([11, 3, 0, 0], dtype=torch.long)
    return tokens, gathers


def test_upload_matches_default_and_bypasses_sync_torch_tensor(monkeypatch):
    plan = _plan()
    ref_tok, ref_g = plan.localize_token_tensor_and_gather_indices(*_inputs(), num_valid_gathers=2)
    calls = []

    def upload(host):
        assert isinstance(host, np.ndarray) and host.dtype == np.int64 and host.flags["C_CONTIGUOUS"]
        calls.append(host.copy())
        return torch.from_numpy(host)

    real_tensor = torch.tensor

    def no_sync_tensor(data, *a, **k):
        if isinstance(data, np.ndarray):
            raise AssertionError("host index array copied with torch.tensor despite upload=")
        return real_tensor(data, *a, **k)

    monkeypatch.setattr(sl.torch, "tensor", no_sync_tensor)
    tok, g = plan.localize_token_tensor_and_gather_indices(*_inputs(), num_valid_gathers=2, upload=upload)
    assert len(calls) == 2
    assert torch.equal(tok, ref_tok) and torch.equal(g, ref_g)
    # rank 1 holds packed rows 8..15 -> request-major tokens 4..7 then padding
    assert tok.tolist()[:4] == [104, 105, 106, 107]


def test_default_path_still_works_without_upload():
    tok, g = _plan().localize_token_tensor_and_gather_indices(*_inputs(), num_valid_gathers=2)
    assert tok.shape[0] == 8 and g.shape == (4,)
