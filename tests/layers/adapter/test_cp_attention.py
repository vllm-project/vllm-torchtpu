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
"""CPU tests for DCP layout propagation and kernel registry isolation.

Real TPU pass correctness is covered by test_attention_dcp_hnd.py. Here the
kernel builder and collectives are replaced so both adapter entry points and
the registry can be checked together, including reuse across layers/layouts.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import (
    DeviceConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.kernels.experimental.batched_rpa.configs import (
    AttentionScope,
    KVLayout,
)
from vllm_torchtpu.layers.adapter import attention, cp_attention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def cpu_vllm_config_context():
    # CPU CI does not register torch's TPU device type. These tests only
    # exercise layout propagation with CPU tensors and mocked kernels.
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _impl(cls=attention.PallasBatchedRPAAttentionBackendImpl, head_size=256):
    return cls(
        num_heads=8,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )


def _config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=2,
        )
    )


@pytest.mark.parametrize("decode_block", [None, "8192,4,2"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("compiled", [False, True])
def test_dcp_layout_survives_prebuild_forward_and_registry_reuse(
    monkeypatch, rank, compiled, decode_block, cpu_vllm_config_context
):
    if decode_block is None:
        monkeypatch.delenv("TPU_BATCHED_RPA_DECODE_BLOCK", raising=False)
    else:
        monkeypatch.setenv("TPU_BATCHED_RPA_DECODE_BLOCK", decode_block)
    # Each case compiles the same forward code for three layers; start from an
    # empty Dynamo cache so the cases stay under its recompile limit.
    torch._dynamo.reset()
    built, calls = [], []

    def build_op(_name, fn):
        built.append(fn)

        def kernel(_cache, query, *args):
            scope = fn.keywords["attention_scope"]
            calls.append((fn.keywords["kv_layout"], scope, query.shape))
            value = 1 if scope == AttentionScope.CACHE_ONLY else 3
            return torch.full_like(query, value), torch.zeros(query.shape[:2])

        return kernel

    group = SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        all_gather=lambda tensor, dim: torch.cat([tensor, tensor], dim=dim),
    )
    monkeypatch.setattr(attention, "_get_dcp_group", lambda: group)
    monkeypatch.setattr(cp_attention, "get_dcp_group", lambda: group)
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", build_op)
    monkeypatch.setattr(cp_attention, "synchronize_tensors", lambda *a, **k: None)
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    implementations = []
    # The third layer must reuse NHD's ops after HND has populated the registry.
    for layout in ("LBNHC", "LBHNC", "LBNHC"):
        get_current_vllm_config().cache_config.kv_cache_layout = layout
        impl = _impl()
        with set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()):
            impl.initialize_kernel(layer)
        implementations.append(impl)

    assert len(built) == 4  # Two layouts, two scopes; shared by matching layers.
    assert (
        implementations[0].rpa_dcp_cache_kernel
        is implementations[2].rpa_dcp_cache_kernel
    )
    assert (
        implementations[0].rpa_dcp_new_kernel is implementations[2].rpa_dcp_new_kernel
    )
    assert (
        implementations[0].rpa_dcp_cache_kernel
        is not implementations[1].rpa_dcp_cache_kernel
    )
    assert (
        implementations[0].rpa_dcp_new_kernel
        is not implementations[1].rpa_dcp_new_kernel
    )

    metadata = SimpleNamespace(
        seq_lens=torch.tensor([2, 3, 4], dtype=torch.int32),
        block_tables=torch.zeros(3, 2, dtype=torch.int32),
        query_start_loc=torch.arange(4, dtype=torch.int32),
        request_distribution=torch.tensor([3, 3, 3], dtype=torch.int32),
    )
    query = torch.ones(3, 8, 256)
    key = torch.ones(3, 1, 256)
    for impl, expected_layout in zip(
        implementations,
        (
            KVLayout.HEAD_ALONG_SUBLANE,
            KVLayout.SEQ_ALONG_LANE,
            KVLayout.HEAD_ALONG_SUBLANE,
        ),
    ):
        # The per-layer layout must survive even if the current config changes.
        get_current_vllm_config().cache_config.kv_cache_layout = "LBHNC"
        forward = impl._run_dcp_forward
        if compiled:
            forward = torch.compile(forward, backend="eager", fullgraph=True)
        output = forward(layer, query, key, key, torch.empty(1), metadata)
        assert calls[-2:] == [
            (expected_layout, AttentionScope.CACHE_ONLY, (3, 16, 256)),
            (expected_layout, AttentionScope.NEW_TOKENS_ONLY, (3, 8, 256)),
        ]
        # Two cached partials (value=1, weight=1 each) and one new partial
        # (value=3, weight=1), including both production LSE merges.
        torch.testing.assert_close(output, torch.full_like(query, 5 / 3))
    assert len(built) == 4  # Forward must reuse the prebuilt ops for each layout.
    assert all(type(fn.keywords["kv_layout"]) is KVLayout for fn in built)
    cache_blocks = [
        fn.keywords["decode_block_sizes"]
        for fn in built
        if fn.keywords["attention_scope"] == AttentionScope.CACHE_ONLY
    ]
    assert all((b is None) == (decode_block is None) for b in cache_blocks)


def test_dcp_kernel_default_layout_reuses_explicit_nhd(monkeypatch):
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", lambda name, fn: fn)
    kwargs = dict(
        sliding_window=None,
        sm_scale=0.0625,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=2,
        cp_rank=0,
    )
    implicit = cp_attention.build_dcp_kernels(**kwargs)
    explicit = cp_attention.build_dcp_kernels(
        **kwargs, kv_layout=KVLayout.HEAD_ALONG_SUBLANE
    )
    assert all(a is b for a, b in zip(implicit, explicit))


@pytest.mark.parametrize(
    "cls,head_size",
    [
        (attention.PallasAttentionBackendImpl, 256),
        (attention.PallasBatchedRPAAttentionBackendImpl, 64),
    ],
)
def test_dcp_hnd_rejects_backends_that_allocate_nhd(
    cls, head_size, cpu_vllm_config_context
):
    get_current_vllm_config().cache_config.kv_cache_layout = "LBHNC"
    impl = _impl(cls, head_size)
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    with (
        set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()),
        pytest.raises(NotImplementedError, match="DCP with HND requires"),
    ):
        impl.initialize_kernel(layer)


@pytest.mark.parametrize(
    "page_size,pages_per_seq,decode_query_size,columns",
    [
        (128, 4104, 1, 4096),  # 1M tokens, DCP2: 1152-token blocks round up
        (128, 4104, 4, 4104),  # verify window may reach column 4096: keep all
        (256, 2048, 1, 2048),  # already exact
    ],
)
def test_dcp_block_table_keeps_the_rank_local_columns(
    page_size, pages_per_seq, decode_query_size, columns
):
    import jax.numpy as jnp

    num_seqs = 3
    table = jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32)
    local = cp_attention._rank_local_block_tables(
        table,
        num_seqs,
        page_size,
        max_model_len=1 << 20,
        decode_query_size=decode_query_size,
        cp_group_size=2,
    )
    expected = table.reshape(num_seqs, pages_per_seq)[:, :columns].reshape(-1)
    assert local.shape == (num_seqs * columns,)
    assert bool(jnp.all(local == expected))


@pytest.mark.parametrize(
    "scope,layout,decode_query_size,bkv",
    [
        (AttentionScope.NEW_TOKENS_ONLY, KVLayout.SEQ_ALONG_LANE, 1, 256),
        (AttentionScope.NEW_TOKENS_ONLY, KVLayout.SEQ_ALONG_LANE, 4, None),
        (AttentionScope.NEW_TOKENS_ONLY, KVLayout.HEAD_ALONG_SUBLANE, 1, None),
        (AttentionScope.CACHE_ONLY, KVLayout.SEQ_ALONG_LANE, 1, None),
    ],
)
def test_dcp_new_token_pass_uses_one_mxu_wide_decode_block(
    monkeypatch, scope, layout, decode_query_size, bkv
):
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.experimental.batched_rpa.configs import BlockSizes

    seen = {}

    def fake_rpa(**kwargs):
        seen.update(kwargs)
        return kwargs["queries"], kwargs["kv_cache"], kwargs["queries"][..., 0]

    monkeypatch.setattr(
        cp_attention._batched_rpa_wrapper, "ragged_paged_attention", fake_rpa
    )
    monkeypatch.setattr(
        cp_attention.pltpu, "get_tpu_info", lambda: SimpleNamespace(mxu_column_size=256)
    )
    kv_cache = jnp.zeros(
        (4, 2, 1, 4, 128) if layout == KVLayout.SEQ_ALONG_LANE else (4, 128, 1, 4, 256)
    )
    query = jnp.zeros((2, 8, 256))
    cp_attention._pallas_rpa_kernel_dcp(
        kv_cache,
        query,
        query[:, :1],
        query[:, :1],
        jnp.array([5, 6], dtype=jnp.int32),
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.array([0, 1, 2], dtype=jnp.int32),
        jnp.array([2, 2, 2], dtype=jnp.int32),
        None,
        1.0,
        1.0,
        sliding_window=None,
        cp_group_size=2,
        cp_rank_val=0,
        attention_scope=scope,
        kv_layout=layout,
        decode_query_size=decode_query_size,
    )
    expected = (
        None
        if bkv is None
        else BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=bkv, batch_size=8, n_buffer=2)
    )
    assert seen["decode_block_sizes"] == expected
    assert seen["attention_scope"] == scope


def test_dcp_cache_pass_takes_the_decode_block_env(monkeypatch):
    from vllm_torchtpu.kernels.experimental.batched_rpa.configs import BlockSizes

    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", lambda name, fn: fn)
    kwargs = dict(
        sliding_window=None,
        sm_scale=0.0625,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=2,
        cp_rank=0,
        kv_layout=KVLayout.SEQ_ALONG_LANE,
    )
    monkeypatch.delenv("TPU_BATCHED_RPA_DECODE_BLOCK", raising=False)
    assert cp_attention.dcp_cache_decode_block(1) is None
    cache, new = cp_attention.build_dcp_kernels(**kwargs)
    assert cache.keywords["decode_block_sizes"] is None
    assert "decode_block_sizes" not in new.keywords

    monkeypatch.setenv("TPU_BATCHED_RPA_DECODE_BLOCK", "8192,4,2")
    block = BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=8192, batch_size=4, n_buffer=2)
    assert cp_attention.dcp_cache_decode_block(1) == block
    # A multi-token verify pass keeps the kernel's own block.
    assert cp_attention.dcp_cache_decode_block(4) is None
    cache, new = cp_attention.build_dcp_kernels(**kwargs, cache_decode_block=block)
    assert cache.keywords["decode_block_sizes"] == block
    assert "decode_block_sizes" not in new.keywords
