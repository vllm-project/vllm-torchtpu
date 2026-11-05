# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
    is_kv_cache_spec_uniform,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import (
    RunLocalStateSpec,
    VllmCompressorStateCache,
    VllmDeepseekCompressor,
)


def _state_cache(compress_ratio: int, head_dim: int = 512):
    """A state cache with the attributes its geometry depends on.

    The base `__init__` registers into the vLLM forward context, which needs a
    live config; the geometry under test does not. `state_dim` follows vLLM's
    own `2 * coff * head_dim` (kv_state + score_state).
    """
    cache = VllmCompressorStateCache.__new__(VllmCompressorStateCache)
    coff = 1 + (compress_ratio == 4)
    cache.prefix = "model.layers.0.self_attn.compressor.state_cache"
    cache.compress_ratio = compress_ratio
    cache.head_dim = head_dim
    cache.state_dim = 2 * coff * head_dim
    cache.sliding_window = coff * compress_ratio
    cache.block_size = 0
    return cache


@pytest.mark.parametrize(
    "compress_ratio, head_dim, cache_block_size, expected",
    [
        # CSA: its state overlays its own compressed-KV page.
        (4, 512, 1024, 16),
        # HCA state rows are hosted on a *CSA* page, not on HCA's own (which
        # holds two token states), so the page is far larger than HCA's
        # compression ratio alone would suggest.
        (128, 512, 1024, 32),
        (128, 512, 512, 16),
        # The lightning indexer: 33 SEQ_ALONG_LANE rows of 1024 bytes hold 16
        # token states (2 rows each), matching CSA's 16 so the two share a
        # `(block_size, sliding_window)` cache group.
        (4, 128, 1024, 16),
    ],
)
def test_state_block_size_follows_the_host_page(
    compress_ratio, head_dim, cache_block_size, expected
):
    """Block size must come from the kernel's own layout model.

    It is not a closed form -- HCA writes two rows per record, the indexer
    array is 256 lanes wide, HCA's state is hosted on a CSA page, and CSA and
    the indexer are floored to a shared value -- so this pins the values
    `compress_and_store.config` actually produces, which is what the
    compressor kernel indexes the block table with.
    """
    cache = _state_cache(compress_ratio, head_dim)
    assert cache._derive_block_size(cache_block_size) == expected


def test_state_block_size_refuses_a_page_holding_no_row():
    """A page too small to hold one compressed row must not be emitted."""
    cache = _state_cache(128)
    with pytest.raises(ValueError, match="cannot be paged"):
        cache._derive_block_size(64)


@pytest.mark.parametrize("compress_ratio, head_dim", [(4, 512), (4, 128)])
def test_state_block_size_with_seq_along_lane_indexer(compress_ratio, head_dim):
    """The SEQ_ALONG_LANE indexer requires a 1024-token block (256 compressed tokens).

    The SEQ_ALONG_LANE indexer page holds 16 compressor states, the same as a
    CSA page, so CSA and the indexer can stay in the same KV cache group. The
    layout only works with 256 compressed tokens per page (a 1024-token
    block). Any other block size raises an error when the block size is
    computed, instead of failing later inside the kernel.
    """
    cache = _state_cache(compress_ratio, head_dim)
    assert cache._derive_block_size(1024) == 16
    with pytest.raises(ValueError, match="cannot be paged"):
        cache._derive_block_size(512)


def test_state_cache_spec_is_raw_uint8_at_the_host_page_size():
    """The spec must survive `kv_cache_spec_normalizer` unretyped.

    Any dtype but uint8 is rewritten to the model's KV dtype, which truncates
    the raw f32 state; the block size must come from the final cache config,
    not the base class's CUDA constant.
    """
    cache = _state_cache(4)
    assert cache.block_size == 0, "fixture starts unset"
    spec = cache.get_kv_cache_spec(
        SimpleNamespace(cache_config=SimpleNamespace(block_size=1024))
    )

    assert spec.dtype == torch.uint8
    assert spec.block_size == 16
    # `_build_compressor_op` bakes this in, so the object must be updated too.
    assert cache.block_size == 16
    # 2048 f32 words = 8192 B, already a multiple of 128.
    assert spec.head_size == 8192
    assert spec.num_kv_heads == 1
    assert spec.sliding_window == 8
    # vLLM's DSv4 grouping is isinstance-based, so the base type is load-bearing.
    assert isinstance(spec, SlidingWindowMLASpec)


@pytest.mark.parametrize(
    "compress_ratio, expected",
    [
        (128, RunLocalStateSpec),
        # The 4-ratio window overlaps the previous run, so a hit still reads it.
        (4, SlidingWindowMLASpec),
    ],
)
def test_only_the_128_ratio_state_leaves_prefix_caching(compress_ratio, expected):
    spec = _state_cache(compress_ratio).get_kv_cache_spec(
        SimpleNamespace(cache_config=SimpleNamespace(block_size=1024))
    )
    assert type(spec) is expected
    assert spec.prefix_cacheable == (expected is SlidingWindowMLASpec)


def test_run_local_state_refuses_a_block_size_off_the_run_grid():
    # 1088 = 8.5 runs: a hit at a block boundary would land mid-run.
    with pytest.raises(ValueError, match="not a multiple of the compress ratio"):
        _state_cache(128).get_kv_cache_spec(
            SimpleNamespace(cache_config=SimpleNamespace(block_size=1088))
        )


def test_run_local_state_refuses_an_overlapping_kernel():
    # An overlapping window also reads the previous run, which a prefix hit
    # leaves as uncached null blocks under RunLocalStateSpec.
    compressor = VllmDeepseekCompressor.__new__(VllmDeepseekCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.prefix = "model.layers.0.self_attn.compressor"
    compressor.state_cache = _state_cache(128)
    compressor.overlap = True
    with pytest.raises(RuntimeError, match="overlapping compressor kernel"):
        compressor._build_compressor_op()


# RunLocalStateSpec on vLLM's KVCacheManager with DeepSeek-V4-shaped groups at
# --block-size 1024: hits are unchanged, the HCA state group caches nothing and
# is left out of the KV offload groups.
BLOCK = 1024
SWA_BLOCK = 128
HCA_STATE = 4
_ids = iter(range(1, 1 << 30))
# 40,333-token first turn: replay boundary 39,936.
TURN1 = list(range(1, 40_334))
# The next turn shares all but the last two prompt tokens (the generation
# prompt), as a chat template renders it.
TURN2 = TURN1[:-2] + list(range(50_000_000, 50_003_000))


def _swa(block_size, window, head_size, cls=SlidingWindowMLASpec):
    return cls(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        sliding_window=window,
    )


def test_run_local_state_does_not_group_with_a_cacheable_window():
    run_local = _swa(SWA_BLOCK, 128, 1024, cls=RunLocalStateSpec)
    cacheable = _swa(SWA_BLOCK, 128, 1024)
    with pytest.raises(AssertionError, match="cannot share a KV cache group"):
        RunLocalStateSpec.merge([run_local, cacheable])
    # vLLM's uniformity probe catches only AssertionError from `merge`.
    assert not is_kv_cache_spec_uniform({"a": run_local, "b": cacheable})
    assert not run_local.is_uniform_with_collection({"a": run_local, "b": cacheable})
    merged = RunLocalStateSpec.merge([run_local, run_local])
    assert type(merged) is RunLocalStateSpec and not merged.prefix_cacheable
    assert run_local.is_uniform_with_collection({"a": run_local, "b": run_local})


@pytest.fixture
def upstream_block_pool(monkeypatch):
    """Undo the plugin's same-step prefix-hit patch, if installed.

    Once the platform activates (any earlier test in the process can trigger
    it), every block hashed since the last `Scheduler.schedule()` is a miss.
    These tests drive `KVCacheManager` without a scheduler, so all hits would
    be 0.
    """
    for name in ("_insert_block_hash", "get_cached_block"):
        monkeypatch.setattr(BlockPool, name, inspect.unwrap(getattr(BlockPool, name)))


def _manager(num_blocks, run_local=True):
    init_none_hash(sha256)
    csa_state = _state_cache(4)._derive_block_size(BLOCK)
    hca_state = _state_cache(128)._derive_block_size(BLOCK)
    hca_cls = RunLocalStateSpec if run_local else SlidingWindowMLASpec
    groups = [
        KVCacheGroupSpec(["model.layers.0.attn.swa_cache"], _swa(SWA_BLOCK, 128, 512)),
        KVCacheGroupSpec(["mtp.0.attn.swa_cache"], _swa(SWA_BLOCK, 128, 512)),
        KVCacheGroupSpec(
            ["model.layers.0.attn", "model.layers.2.attn"],
            MLAAttentionSpec(
                block_size=BLOCK,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.uint8,
                tokens_per_state=128,
            ),
        ),
        KVCacheGroupSpec(
            [
                "model.layers.2.attn.compressor.state_cache",
                "model.layers.2.attn.indexer.compressor.state_cache",
            ],
            _swa(csa_state, 8, 4096),
        ),
        KVCacheGroupSpec(
            ["model.layers.0.attn.compressor.state_cache"],
            _swa(hca_state, 128, 2048, hca_cls),
        ),
    ]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=32768,
    )
    mgr = KVCacheManager(
        config,
        max_model_len=1 << 20,
        scheduler_block_size=BLOCK,
        hash_block_size=csa_state,
        enable_caching=True,
        use_eagle=False,
    )
    mgr._test_hasher = get_request_block_hasher(csa_state, sha256)
    mgr._test_hca_state_block = hca_state
    return mgr


def _request(mgr, tokens, max_tokens):
    return Request(
        f"r{next(_ids)}",
        list(tokens),
        SamplingParams(max_tokens=max_tokens),
        None,
        block_hasher=mgr._test_hasher,
    )


def _prefill(mgr, req, step=128):
    blocks, hit, _ = mgr.get_computed_blocks(req)
    first = True
    while req.num_computed_tokens < req.num_prompt_tokens:
        local = hit if first else 0
        n = min(step, req.num_prompt_tokens - req.num_computed_tokens - local)
        out = mgr.allocate_slots(
            req,
            n,
            num_new_computed_tokens=local,
            new_computed_blocks=blocks if first else None,
        )
        assert out is not None
        req.num_computed_tokens += n + local
        first = False
    return hit


def _decode(mgr, req, steps, accept=4):
    for _ in range(steps):
        for _ in range(accept):
            req.append_output_token_ids(7_000_000 + len(req.output_token_ids))
        assert mgr.allocate_slots(req, accept, num_lookahead_tokens=6) is not None
        req.num_computed_tokens += accept


def _run(mgr, tokens, decode_steps=50):
    req = _request(mgr, tokens, decode_steps * 4)
    hit = _prefill(mgr, req)
    _decode(mgr, req, decode_steps)
    mgr.free(req)
    return hit


def _cached_blocks(mgr, group):
    return sum(
        b.block_hash is not None and get_group_id(b.block_hash) == group
        for b in mgr.block_pool.blocks
    )


def _scenario(mgr):
    """Hits of: next turn, a truncated history, and a next turn after churn."""
    hits = []
    _run(mgr, TURN1, decode_steps=300)
    hits.append(mgr.get_computed_blocks(_request(mgr, TURN2, 4))[1])
    truncated = TURN1[:40_100] + [3] * 900
    hits.append(mgr.get_computed_blocks(_request(mgr, truncated, 4))[1])
    req = _request(mgr, TURN2, 1300)
    hits.append(_prefill(mgr, req))
    for step in range(300):
        _decode(mgr, req, 1)
        if step % 10 == 0:
            base = 10_000_000 + step * 100_000
            _run(mgr, range(base, base + 3000), decode_steps=5)
    mgr.free(req)
    turn3 = TURN2[:-2] + list(range(60_000_000, 60_002_000))
    hits.append(mgr.get_computed_blocks(_request(mgr, turn3, 10))[1])
    return hits


def test_run_local_state_leaves_prefix_hits_unchanged(upstream_block_pool):
    expected = _scenario(_manager(130, run_local=False))
    assert _scenario(_manager(130)) == expected
    assert expected[0] == 39_936

    fresh = _manager(130)
    _run(fresh, TURN1)
    blocks, hit, _ = fresh.get_computed_blocks(_request(fresh, TURN2, 4))
    assert hit == expected[0]
    assert not blocks.blocks[HCA_STATE]
    assert _cached_blocks(fresh, HCA_STATE) == 0


def test_run_local_state_caches_and_offloads_nothing(upstream_block_pool):
    base = _manager(400, run_local=False)
    _run(base, TURN1)
    lean = _manager(400)
    _run(lean, TURN1)
    # Two checkpoints (32k and the replay boundary), each a 128-token window.
    hca_blocks_per_checkpoint = 128 // lean._test_hca_state_block
    assert _cached_blocks(base, HCA_STATE) == 2 * hca_blocks_per_checkpoint
    assert _cached_blocks(lean, HCA_STATE) == 0
    for group in range(HCA_STATE):
        assert _cached_blocks(lean, group) == _cached_blocks(base, group)

    assert get_offloading_group_ids(base.kv_cache_config) == (0, 1, 2, 3, HCA_STATE)
    assert get_offloading_group_ids(lean.kv_cache_config) == (0, 1, 2, 3)


def test_compressor_op_is_built_lazily_and_cached():
    """Building in `__init__` bakes the pre-KV-init `state_block_size`.

    That constant is the base class's CUDA value until `get_kv_cache_spec`
    replaces it, so an eagerly built op traces against the wrong page geometry.
    """
    assert isinstance(VllmDeepseekCompressor.__dict__["compressor_op"], property), (
        "compressor_op must stay a property, not an __init__ field"
    )

    compressor = VllmDeepseekCompressor.__new__(VllmDeepseekCompressor)
    sentinel = object()
    with patch.object(
        VllmDeepseekCompressor, "_build_compressor_op", return_value=sentinel
    ) as build:
        assert compressor.compressor_op is sentinel
        assert compressor.compressor_op is sentinel
    build.assert_called_once()


def test_transpose_wkv_wgate_allocates_fresh_buffer():
    """The transposed weight gets its own buffer even if `.contiguous()` is a no-op.

    On CPU `.contiguous()` already copies, so it is patched to return `self`,
    mimicking the torch_tpu view chain that the fresh buffer + `copy_` breaks.
    """
    compressor = VllmDeepseekCompressor.__new__(VllmDeepseekCompressor)
    torch.nn.Module.__init__(compressor)
    orig = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    orig_ptr = orig.untyped_storage().data_ptr()
    compressor.fused_wkv_wgate = SimpleNamespace(
        weight=torch.nn.Parameter(orig, requires_grad=False)
    )
    compressor._wkv_wgate_transposed = False

    with patch.object(torch.Tensor, "contiguous", lambda self, *a, **k: self):
        compressor.transpose_wkv_wgate()

    w = compressor.fused_wkv_wgate.weight
    assert compressor._wkv_wgate_transposed is True
    assert w.shape == (4, 3)
    assert w.is_contiguous()
    assert w.untyped_storage().data_ptr() != orig_ptr
    assert torch.equal(w, orig.t())

    compressor.transpose_wkv_wgate()  # A second call must not flip it back.
    assert compressor.fused_wkv_wgate.weight is w
