# SPDX-License-Identifier: Apache-2.0
"""Hash-unit prefix hits under model-internal (QSA main-KV) DCP.

See `_patch_vllm_qsa_dcp_fine_prefix_hits`. With VLLM_TPU_QSA_MAIN_KV_DCP=D
the main-KV group's block is D*B tokens and the Mamba "align" group's is B,
so vLLM schedules in D*B blocks and hashes in B. These tests drive the
scheduler-side KV cache manager the way the scheduler does: without the flag
a shared prefix hits only at D*B multiples; with it the hit lands on the
prompt-tail hash boundary inside a D*B block and the shared block is
copy-on-written.
"""

import inspect

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_coordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu import _patch_vllm_qsa_dcp_fine_prefix_hits

B = 16


@pytest.fixture
def patched(monkeypatch):
    cls = kv_cache_coordinator.HybridKVCacheCoordinator
    monkeypatch.setattr(cls, "__init__", inspect.unwrap(cls.__init__))
    monkeypatch.setattr(cls, "_tpu_qsa_dcp_fine_prefix_patch", False, raising=False)
    monkeypatch.setattr(
        KVCacheManager, "free", inspect.unwrap(KVCacheManager.free)
    )
    _patch_vllm_qsa_dcp_fine_prefix_hits()


def _flags(monkeypatch, *, dcp: int, fine: bool, seq_tail: bool = False) -> None:
    monkeypatch.setenv("VLLM_TPU_QSA_MAIN_KV_DCP", str(dcp))
    monkeypatch.setenv("VLLM_TPU_QSA_DCP_FINE_PREFIX", "1" if fine else "0")
    monkeypatch.setenv("VLLM_TPU_QSA_DCP_FINE_PREFIX_SEQ_TAIL", "1" if seq_tail else "0")


def _manager(dcp: int, eagle: bool = False) -> KVCacheManager:
    """GDN-like state group at B, QSA main-KV group at dcp*B (same page), and
    with ``eagle`` an MTP-draft main-KV group at B."""
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=dcp * B, num_kv_heads=1, head_size=64, dtype=torch.float32
    )
    state = MambaSpec(
        block_size=B,
        shapes=((1, 64),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        page_size_padded=attention.page_size_bytes,
    )
    groups = [
        KVCacheGroupSpec(["state"], state),
        KVCacheGroupSpec(["attention"], attention),
    ]
    if eagle:
        draft = FullAttentionSpec(
            block_size=B,
            num_kv_heads=1,
            head_size=64,
            dtype=torch.float32,
            page_size_padded=attention.page_size_bytes,
        )
        groups.append(KVCacheGroupSpec(["draft"], draft, is_eagle_group=True))
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        config,
        max_model_len=1024,
        scheduler_block_size=dcp * B,
        hash_block_size=B,
        enable_caching=True,
        use_eagle=eagle,
    )


def _request(request_id: str, tokens: list[int]) -> Request:
    return Request(
        request_id,
        tokens,
        SamplingParams(max_tokens=200),
        None,
        block_hasher=get_request_block_hasher(B, sha256),
    )


def _run(
    manager: KVCacheManager, request: Request, eagle: bool = False, decode: int = 1
) -> tuple[int, list]:
    """Admit, prefill stopping at the prompt's last hash boundary (the align
    split's checkpoint stop; one unit earlier with EAGLE), then decode.
    Returns the hit and the CoW pairs."""
    manager.new_step_starts()
    blocks, hit = manager.get_computed_blocks(request)[:2]
    prompt = request.num_tokens
    checkpoint = prompt // B * B - (B if eagle else 0)
    stops = sorted({checkpoint, prompt} - {0})
    start, first = hit, True
    copies = []
    for end in stops:
        if end <= start:
            continue
        manager.new_step_starts()
        assert (
            manager.allocate_slots(
                request,
                end - start,
                hit if first else 0,
                blocks if first else None,
            )
            is not None
        )
        copies += manager.take_kv_cache_block_copies() or []
        first = False
        request.num_computed_tokens = start = end
    for i in range(decode):
        request.append_output_token_ids(10_000 + i)
        manager.new_step_starts()
        assert manager.allocate_slots(request, 1) is not None
        request.num_computed_tokens += 1
    return hit, copies


def _shared_then_diverge(shared: int, total: int) -> list[int]:
    return list(range(shared)) + [50_000 + i for i in range(total - shared)]


@pytest.mark.parametrize("dcp", [2, 4])
def test_flag_off_rounds_hits_to_dcp_block(patched, monkeypatch, dcp):
    _flags(monkeypatch, dcp=dcp, fine=False)
    manager = _manager(dcp)
    assert not manager.coordinator.enable_partial_hash_hits
    prompt_a = list(range(dcp * B + 3 * B + 8))
    _run(manager, _request("a", prompt_a))
    hit, copies = _run(manager, _request("b", prompt_a[: dcp * B + 3 * B + 2] + [9] * 20))
    assert hit == dcp * B
    assert copies == []


@pytest.mark.parametrize("dcp", [2, 4])
def test_prompt_tail_hit_inside_dcp_block_with_cow(patched, monkeypatch, dcp):
    _flags(monkeypatch, dcp=dcp, fine=True)
    manager = _manager(dcp)
    assert manager.coordinator.enable_partial_hash_hits
    # A's prompt ends inside its second D*B block; its last hash boundary,
    # (D + D - 1) * B, is the prompt-tail partial entry (an odd B multiple).
    boundary = (2 * dcp - 1) * B
    prompt_a = list(range(boundary + 8))
    _run(manager, _request("a", prompt_a))
    attention_a = manager.coordinator.single_type_managers[1].req_to_blocks["a"]
    # B shares A past that boundary, then diverges.
    hit, copies = _run(manager, _request("b", _shared_then_diverge(boundary + 4, boundary + 40)))
    assert hit == boundary
    assert hit % (dcp * B) != 0
    # Exactly one CoW: A's partially hit main-KV block into a private block.
    assert len(copies) == 1
    assert copies[0].src_block_id == attention_a[1].block_id
    attention_b = manager.coordinator.single_type_managers[1].req_to_blocks["b"]
    assert copies[0].dst_block_id == attention_b[1].block_id
    assert attention_b[0].block_id == attention_a[0].block_id
    assert attention_b[1].block_id != attention_a[1].block_id


@pytest.mark.parametrize("fine", [True, False])
def test_eagle_drop_is_one_hash_unit(patched, monkeypatch, fine):
    # D=4: A's prompt-tail entry at 7B; EAGLE drops one unit (B) to 6B, which
    # is inside the second 4B block. Without the flag the hit rounds to 4B.
    dcp = 4
    _flags(monkeypatch, dcp=dcp, fine=fine)
    manager = _manager(dcp, eagle=True)
    boundary = 7 * B
    prompt_a = list(range(boundary + 8))
    _run(manager, _request("a", prompt_a), eagle=True)
    hit, copies = _run(
        manager,
        _request("b", _shared_then_diverge(boundary + 4, boundary + 40)),
        eagle=True,
    )
    assert hit == (6 * B if fine else 4 * B)
    assert len(copies) == (1 if fine else 0)


@pytest.mark.parametrize("seq_tail", [True, False])
def test_sequence_tail_entry_after_free(patched, monkeypatch, seq_tail):
    # A: prompt 2B+8, then decode to 3B+5 computed tokens. The next turn
    # shares A's whole sequence; only the sequence-tail entry (at 3B, inside
    # A's second 2B block) lets it hit past 2B.
    dcp = 2
    _flags(monkeypatch, dcp=dcp, fine=True, seq_tail=seq_tail)
    manager = _manager(dcp)
    a = _request("a", list(range(2 * B + 8)))
    _run(manager, a, decode=B - 3)
    assert a.num_computed_tokens == 3 * B + 5
    manager.free(a)
    # The main-KV entry at 3B exists only with the sequence-tail registration.
    entry = manager.block_pool.get_cached_block(a.block_hashes[2], [1])
    assert (entry is not None) == seq_tail
    shared = a.all_token_ids[: 3 * B + 5]
    hit, copies = _run(manager, _request("b", list(shared) + [7] * 30))
    # The Mamba checkpoint at 3B comes from decode crossing it; without one
    # the hybrid hit stays at 2B either way.
    state_cached = manager.block_pool.get_cached_block(a.block_hashes[2], [0])
    if seq_tail and state_cached:
        assert hit == 3 * B
        assert len(copies) == 1
    else:
        assert hit == 2 * B


def test_not_applied_without_main_kv_dcp(patched, monkeypatch):
    _flags(monkeypatch, dcp=0, fine=True)
    assert not _manager(2).coordinator.enable_partial_hash_hits


def test_patch_is_idempotent(patched):
    cls = kv_cache_coordinator.HybridKVCacheCoordinator
    init = cls.__init__
    _patch_vllm_qsa_dcp_fine_prefix_hits()
    assert cls.__init__ is init
