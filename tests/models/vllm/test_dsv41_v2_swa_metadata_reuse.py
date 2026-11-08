# SPDX-License-Identifier: Apache-2.0
"""CPU contracts: reused SWA metadata equals what each consumer computes alone.

Covers the decoder's replanned uploads (`_upload`), the CSA2 branch's
stage-wide token requests and write mask, and the DSpark draft layers' shared
context-KV addresses.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import csa2_adapter
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import dspark as v2_dspark
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import _upload

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# A: decoder bounds and replay starts
# ---------------------------------------------------------------------------


def test_upload_reuses_equal_values_and_reuploads_on_change():
    reuse: dict = {}
    first = _upload((0, 3, 7), torch.int32, torch.device("cpu"), reuse)
    assert first.tolist() == [0, 3, 7] and first.dtype == torch.int32
    assert _upload((0, 3, 7), torch.int32, torch.device("cpu"), reuse) is first
    changed = _upload((0, 3, 8), torch.int32, torch.device("cpu"), reuse)
    assert changed is not first and changed.tolist() == [0, 3, 8]
    # Same values in another dtype are another upload.
    wide = _upload((0, 3, 8), torch.int64, torch.device("cpu"), reuse)
    assert wide is not changed and wide.dtype == torch.int64
    # Without a reuse dict every call uploads.
    assert _upload((1,), torch.int32, torch.device("cpu")) is not _upload(
        (1,), torch.int32, torch.device("cpu")
    )


# ---------------------------------------------------------------------------
# B: CSA2 token requests and write mask, once per stage
# ---------------------------------------------------------------------------


class _Stop(Exception):
    """Raised at the SWA publication, after the metadata branch ran."""


def _csa2_attn():
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.window_size = 128
    attn._ced_decoder = True
    attn._csa2 = True
    attn._fused_preprocess = False
    attn._local_q_attn = False
    attn.compress_ratio = 0
    attn.n_local_heads = 1
    attn.head_dim = 512
    attn._rotate = lambda x, positions: x
    attn.wq_b = lambda qr: qr
    cache = MagicMock(spec=PagedCacheRows)
    cache.numel.return_value = 1024
    cache.shape = (64, 64, 512)
    cache.page_stride = 64
    cache.row_offset = 0
    attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=cache)
    return attn


@pytest.fixture
def csa2_calls(monkeypatch):
    positions = torch.tensor([10, 11, 40, 41], dtype=torch.int32)
    swa = SimpleNamespace(
        seq_lens=torch.tensor([12, 42], dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    context = SimpleNamespace(
        attn_metadata={"swa": swa},
        additional_kwargs={
            "deepseek_v41_replay_starts": torch.zeros(2, dtype=torch.int32)
        },
    )
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: context)
    calls = SimpleNamespace(prepare=0, requests=[], published=[], swa=swa)
    original_requests = v2_attn._token_requests

    def token_requests(metadata, positions, out_int32=False):
        result = original_requests(metadata, positions, out_int32)
        calls.requests.append(result)
        return result

    def prepare_swa(table, starts, positions, minimum, cache, non_causal):
        calls.prepare += 1
        records = (table[:, :1].repeat_interleave(2) * 64 + positions % 64).int()
        records[3] = -1
        return torch.zeros((2, 4), dtype=torch.int32), torch.zeros((2, 2)), records

    def publish_swa(cache, records, values):
        calls.published.append(records)
        raise _Stop

    monkeypatch.setattr(v2_attn, "_token_requests", token_requests)
    monkeypatch.setattr(csa2_adapter, "prepare_swa", prepare_swa)
    monkeypatch.setattr(csa2_adapter, "publish_swa", publish_swa)

    def run(attn):
        out = torch.zeros((4, 1, 512), dtype=torch.bfloat16)
        x = torch.ones((4, 512), dtype=torch.bfloat16)
        with pytest.raises(_Stop):
            attn._prepare_tpu(None, x, x, None, None, None, positions, out)

    calls.run = run
    return calls


def test_csa2_requests_computed_once_per_stage(csa2_calls):
    layers = [_csa2_attn() for _ in range(3)]
    with v2_attn.stage_attention_cache_scope():
        for attn in layers:
            csa2_calls.run(attn)
    assert csa2_calls.prepare == 1
    assert len(csa2_calls.requests) == 1
    assert csa2_calls.requests[0].tolist() == [0, 0, 1, 1]
    assert all(r is csa2_calls.published[0] for r in csa2_calls.published)


def test_csa2_requests_without_stage_cache_match(csa2_calls):
    with v2_attn.stage_attention_cache_scope():
        csa2_calls.run(_csa2_attn())
    cached = csa2_calls.requests[0]
    csa2_calls.requests.clear()
    for _ in range(2):
        csa2_calls.run(_csa2_attn())
    assert len(csa2_calls.requests) == 2
    assert all(torch.equal(r, cached) for r in csa2_calls.requests)


def test_csa2_new_block_tables_recompute(csa2_calls):
    attn = _csa2_attn()
    with v2_attn.stage_attention_cache_scope():
        csa2_calls.run(attn)
        csa2_calls.swa.block_tables = csa2_calls.swa.block_tables + 10
        csa2_calls.run(attn)
    assert csa2_calls.prepare == 2
    assert len(csa2_calls.requests) == 2
    assert not torch.equal(csa2_calls.published[0], csa2_calls.published[1])


# ---------------------------------------------------------------------------
# C: DSpark draft layers share their context-KV addresses
# ---------------------------------------------------------------------------


def _draft_attn():
    return SimpleNamespace(
        window_size=4,
        swa_cache_layer=SimpleNamespace(kv_cache=torch.zeros((8, 4, 1, 8))),
    )


def _draft_metadata():
    return SimpleNamespace(
        block_tables=torch.tensor([[1, 2], [5, 6]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
        seq_lens=torch.tensor([6, 7], dtype=torch.int32),
    )


@pytest.fixture
def inserted(monkeypatch):
    found: list[tuple[torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(
        v2_dspark,
        "_insert",
        lambda cache, slots, kv, valid: found.append((slots, valid)),
    )
    return found


def test_draft_layers_share_addresses(monkeypatch, inserted):
    calls = []
    original = v2_dspark._context_kv_addresses

    def counted(attn, positions, metadata):
        calls.append(metadata)
        return original(attn, positions, metadata)

    monkeypatch.setattr(v2_dspark, "_context_kv_addresses", counted)
    # Position 1 of request 0 (length 6) lies before its 4-row window.
    positions = torch.tensor([1, 4, 5, 5, 6], dtype=torch.int64)
    kv = torch.ones((5, 8))
    metadata = _draft_metadata()
    layers = [_draft_attn() for _ in range(3)]

    for attn in layers:
        v2_dspark._insert_context_kv(attn, kv, positions, metadata)
    alone = list(inserted)
    inserted.clear()
    calls.clear()

    addresses: dict = {}
    for attn in layers:
        v2_dspark._insert_context_kv(attn, kv, positions, metadata, addresses)
    assert len(calls) == 1
    assert len(inserted) == 3
    for (slots, valid), (ref_slots, ref_valid) in zip(inserted, alone):
        assert torch.equal(slots, ref_slots) and torch.equal(valid, ref_valid)
    assert inserted[0][1].tolist() == [False, True, True, True, True]


def test_draft_layers_with_other_tables_recompute(monkeypatch, inserted):
    calls = []
    original = v2_dspark._context_kv_addresses

    def counted(attn, positions, metadata):
        calls.append(metadata)
        return original(attn, positions, metadata)

    monkeypatch.setattr(v2_dspark, "_context_kv_addresses", counted)
    positions = torch.tensor([3, 4, 5, 5, 6], dtype=torch.int64)
    kv = torch.ones((5, 8))
    first, second = _draft_metadata(), _draft_metadata()
    second.block_tables = second.block_tables + 1
    addresses: dict = {}
    v2_dspark._insert_context_kv(_draft_attn(), kv, positions, first, addresses)
    v2_dspark._insert_context_kv(_draft_attn(), kv, positions, second, addresses)
    v2_dspark._insert_context_kv(_draft_attn(), kv, positions, first, addresses)
    assert len(calls) == 2
    assert not torch.equal(inserted[0][0], inserted[1][0])
    assert torch.equal(inserted[0][0], inserted[2][0])


def test_draft_slot_tensor_metadata_shared(inserted):
    slots = torch.tensor([2, -1, 5], dtype=torch.long)
    addresses: dict = {}
    kv = torch.ones((3, 8))
    positions = torch.tensor([0, 1, 2])
    for _ in range(3):
        v2_dspark._insert_context_kv(_draft_attn(), kv, positions, slots, addresses)
    assert all(found[0] is slots for found in inserted)
    assert inserted[0][1].tolist() == [True, False, True]
