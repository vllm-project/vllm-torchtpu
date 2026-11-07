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
"""MiniMax M3 DSpark under DCP, on CPU.

The target's MSA caches are DCP-sharded, so the KV manager's block is
``128 * dcp`` tokens. The drafter keeps its K/V whole on every rank: its spec
keeps the 128-token block (one group with the target) but budgets ``dcp``
times the page bytes, it is allocated with ``128 * dcp``-token pages that
the logical block table addresses directly, and its attention runs the
non-DCP kernel over the full context.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    dynamic_validate_inputs,
)
from vllm_torchtpu.layers.adapter import attention as attention_adapter
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasAttentionBackendImpl,
)
from vllm_torchtpu.models.minimax_m3.attention import MiniMaxM3Cache
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.runner import kv_cache_manager
from vllm_torchtpu.spec_decode.dflash import DFlashProposer

MAX_MODEL_LEN = 262144
BLOCK = 128
HEAD_DIM = 128
WINDOW = 4096
K = 7
DTYPE = torch.float8_e4m3fn


def _parallel(dcp: int) -> SimpleNamespace:
    return SimpleNamespace(
        decode_context_parallel_size=dcp, prefill_context_parallel_size=1
    )


def _impl(replicated: bool) -> PallasAttentionBackendImpl:
    impl = PallasAttentionBackendImpl(
        num_heads=16,
        head_size=HEAD_DIM,
        scale=HEAD_DIM**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=WINDOW,
        kv_cache_dtype="fp8",
    )
    impl.kv_replicated_over_dcp = replicated
    return impl


def _draft_layer(replicated: bool) -> Attention:
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.num_kv_heads = 1
    layer.head_size = HEAD_DIM
    layer.sliding_window = WINDOW
    layer.attn_type = AttentionType.DECODER
    layer.kv_sharing_target_layer_name = None
    layer.impl = SimpleNamespace(kv_replicated_over_dcp=replicated)
    return layer


@pytest.mark.parametrize(
    "dcp,replicated,tokens", [(1, True, BLOCK), (2, False, BLOCK), (2, True, 256)]
)
def test_page_tokens(dcp, replicated, tokens):
    vllm_config = SimpleNamespace(parallel_config=_parallel(dcp))
    layer = _draft_layer(replicated)
    page_tokens = kv_cache_manager.dcp_replicated_page_tokens
    assert page_tokens(layer, BLOCK, vllm_config) == tokens
    # Layers with no impl (owned caches) keep the block.
    assert page_tokens(object(), BLOCK, vllm_config) == BLOCK


@pytest.mark.parametrize("replicated", [False, True])
def test_replicated_impl_skips_the_dcp_kernels(monkeypatch, replicated):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(
        attention_adapter,
        "_get_dcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    monkeypatch.setattr(
        attention_adapter, "_build_dcp_kernels", lambda **kwargs: ("cache", "new")
    )
    with set_current_vllm_config(VllmConfig()):
        impl = _impl(replicated)
    monkeypatch.setattr(impl, "_build_rpa_kernel", lambda *a, **k: "rpa")
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    vllm_config = SimpleNamespace(parallel_config=_parallel(2))
    with set_vllm_model_wrapper_context(mesh=None, vllm_config=vllm_config):
        impl.initialize_kernel(layer)
    if replicated:
        assert (impl.dcp_world_size, impl.rpa_kernel) == (1, "rpa")
        assert impl.rpa_dcp_cache_kernel is None
    else:
        assert impl.dcp_world_size == 2
        assert impl.rpa_dcp_cache_kernel == "cache"


def _runner(dcp: int, layers: dict, hma_disabled: bool = True) -> SimpleNamespace:
    vllm_config = SimpleNamespace(
        parallel_config=_parallel(dcp),
        cache_config=SimpleNamespace(block_size=BLOCK, cache_dtype="fp8"),
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )
    return SimpleNamespace(
        vllm_config=vllm_config,
        kv_cache_dtype=DTYPE,
        _unified_kv_layout=True,  # skips the page padding pass
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=hma_disabled),
        _hybrid_uniform_page_size_bytes=None,
        shared_kv_cache_layers={},
        _is_async_drafter=False,
    )


def _owned(width: int, prefix: str) -> MiniMaxM3Cache:
    cache = MiniMaxM3Cache.__new__(MiniMaxM3Cache)
    torch.nn.Module.__init__(cache)
    cache.row_width, cache.dtype, cache.prefix = width, DTYPE, prefix
    cache.token_shards, cache.block_size = 1, BLOCK
    cache.scales_in_page = False
    return cache


def _specs(monkeypatch, dcp: int, replicated: bool, hma_disabled: bool = True):
    layers = {
        "model.layers.0.self_attn.attn": _owned(2 * HEAD_DIM, "kv"),
        "model.layers.0.self_attn.attn.index_cache": _owned(HEAD_DIM, "index"),
        "draft_model.model.layers.0.self_attn.attn": _draft_layer(replicated),
    }
    runner = _runner(dcp, layers, hma_disabled)
    monkeypatch.setattr(
        kv_cache_manager, "get_layers_from_vllm_config", lambda *args: layers
    )
    monkeypatch.setattr(
        kv_cache_manager,
        "dense_attention_backend",
        lambda vllm_config: PallasAttentionBackend,
    )
    manager = kv_cache_manager.KVCacheManager.__new__(kv_cache_manager.KVCacheManager)
    manager.runner = runner
    return manager.get_kv_cache_spec()


@pytest.mark.parametrize("dcp", [2, 4])
def test_draft_spec_groups_with_the_sharded_target(monkeypatch, dcp):
    specs = _specs(monkeypatch, dcp, replicated=True)
    draft = specs["draft_model.model.layers.0.self_attn.attn"]
    plain = _specs(monkeypatch, 1, replicated=False)[
        "draft_model.model.layers.0.self_attn.attn"
    ]
    # Same block, same (DCP-sharded) flag as the target: one KV cache group
    # whose manager block is 128 * dcp.
    assert type(draft) is FullAttentionSpec
    assert draft.sliding_window == WINDOW
    assert draft.block_size == BLOCK and draft.dcp_sharded
    assert all(
        isinstance(spec, MLAAttentionSpec) and spec.block_size == BLOCK
        for name, spec in specs.items()
        if not name.startswith("draft")
    )
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    group = UniformTypeKVCacheSpecs.from_specs(specs)
    assert group is not None and group.dcp_sharded and group.block_size == BLOCK
    # A whole logical block per page: dcp times the 128-token page bytes.
    assert draft.page_size_bytes == dcp * plain.page_size_bytes
    # The allocation that budget pays for: 128 * dcp-token pages.
    shape = PallasAttentionBackend.get_kv_cache_shape(
        1, BLOCK * dcp, 1, HEAD_DIM, DTYPE
    )
    assert shape[1] == BLOCK * dcp
    assert int(np.prod(shape)) * DTYPE.itemsize == draft.page_size_bytes
    # Full-context cost per request: pages of the logical block table.
    blocks = draft.max_num_blocks_per_req(
        SimpleNamespace(parallel_config=_parallel(dcp)), MAX_MODEL_LEN
    )
    assert blocks * draft.page_size_bytes == (
        cdiv(MAX_MODEL_LEN, BLOCK) * plain.page_size_bytes
    )


def test_replicated_draft_needs_the_unified_group(monkeypatch):
    with pytest.raises(NotImplementedError, match="hybrid"):
        _specs(monkeypatch, 2, replicated=True, hma_disabled=False)


def _validate(page_size: int, pages_per_seq: int, seq_lens: torch.Tensor):
    num_reqs = seq_lens.shape[0]
    rng = np.random.default_rng(0)
    num_pages = 64
    tokens = num_reqs * K
    q = jnp.zeros((tokens, 16, HEAD_DIM), jnp.bfloat16)
    kv = jnp.zeros((tokens, 1, HEAD_DIM), jnp.float8_e4m3fn)
    kv_cache = jnp.zeros((num_pages, page_size, 1, 4, HEAD_DIM), jnp.float8_e4m3fn)
    dynamic_validate_inputs(
        q,
        kv,
        kv,
        kv_cache,
        jnp.asarray(seq_lens.numpy(), jnp.int32),
        jnp.asarray(rng.integers(1, num_pages, num_reqs * pages_per_seq), jnp.int32),
        jnp.arange(num_reqs + 1, dtype=jnp.int32) * K,
        jnp.asarray([0, 0, num_reqs], jnp.int32),
        use_causal_mask=False,
        sliding_window=WINDOW,
    )


@pytest.mark.parametrize("dcp", [2, 4])
def test_long_context_draft_block_fits_the_logical_table(dcp):
    """The draft forward at long anchors indexes inside the DCP block table
    (cdiv(max_model_len, 128 * dcp) entries) with 128 * dcp-token pages; with
    128-token pages it would need dcp times the entries."""
    anchors = [1000, 70000, 200000, MAX_MODEL_LEN - 3]
    stub = SimpleNamespace(_max_model_len=MAX_MODEL_LEN)
    n = len(anchors)
    _, _, seq_lens = DFlashProposer._layout_draft_block(
        stub,
        torch.zeros(n, dtype=torch.int32),
        torch.tensor(anchors, dtype=torch.int32).unsqueeze(1),
        K,
        n * K,
        200058,
    )
    pages_per_seq = cdiv(MAX_MODEL_LEN, BLOCK * dcp)
    _validate(BLOCK * dcp, pages_per_seq, seq_lens)
    with pytest.raises(ValueError, match="page_cnt"):
        _validate(BLOCK, pages_per_seq, seq_lens)
