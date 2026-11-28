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

import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
import torch
from jax import export
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    config as compressor_config,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    config as streamidx_config,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
    deepseek_v4_indexer as indexer_mod,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer import (
    VllmDeepseekV4Indexer,
    VllmDeepseekV4IndexerCache,
    _indexer_jax,
    static_verify_q_len,
)


@pytest.mark.parametrize("block_size, expected_ratio", [(1024, 4), (2, 2)])
def test_indexer_cache_spec_is_raw_uint8(block_size, expected_ratio):
    """The indexer cache is a packed byte record, not KV-dtype features.

    Any dtype but uint8 is retyped by `kv_cache_spec_normalizer`; the compress
    ratio is clamped so it can never exceed the page it is packed into.
    """
    cache = VllmDeepseekV4IndexerCache.__new__(VllmDeepseekV4IndexerCache)
    cache.cache_config = SimpleNamespace(block_size=block_size)
    cache.compress_ratio = 4

    spec = cache.get_kv_cache_spec(SimpleNamespace())

    assert isinstance(spec, MLAAttentionSpec)
    assert spec.dtype == torch.uint8
    assert spec.block_size == block_size
    assert spec.num_kv_heads == 1
    # 32 fp8 sublane groups + 1 scale group = 33 * 4 = 132 bytes per token.
    assert spec.head_size == 132
    assert spec.tokens_per_state == expected_ratio


def test_indexer_refuses_a_cache_of_the_other_layout():
    """The indexer raises if the cache does not have the 33 SEQ_ALONG_LANE rows.

    Without this check it would read the key bytes in the wrong order and
    pick top-k from meaningless scores, with no error.
    """
    zeros = jnp.zeros((1,), jnp.int32)
    with pytest.raises(ValueError, match="SEQ_ALONG_LANE layout"):
        _indexer_jax(
            jnp.zeros((1, 1, 128), jnp.bfloat16),
            jnp.zeros((1, 64), jnp.float32),
            jnp.zeros((1, 1), jnp.float32),
            jnp.zeros((1, 64, 4, 256), jnp.uint8),
            zeros,
            zeros,
            jnp.zeros((2,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            k=1,
            compression_ratio=4,
            softmax_scale=1.0,
            n_head=1,
        )


def _config(model_type, num_speculative_tokens, target_model_type=None):
    """A VllmConfig stand-in: model type, K, and optionally the target's type."""
    hf = lambda t: SimpleNamespace(hf_config=SimpleNamespace(model_type=t))  # noqa: E731
    spec = None
    if num_speculative_tokens is not None:
        spec = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            target_model_config=None
            if target_model_type is None
            else hf(target_model_type),
        )
    return SimpleNamespace(speculative_config=spec, model_config=hf(model_type))


@pytest.mark.parametrize("k, expected", [(5, 6), (6, 7)])
def test_deepseek_v4_gets_a_full_window_length(k, expected):
    assert static_verify_q_len(_config("deepseek_v4", k)) == expected


def test_no_speculative_config_means_no_static_length():
    assert static_verify_q_len(_config("deepseek_v4", None)) is None
    assert static_verify_q_len(None) is None


def test_other_models_keep_the_mixed_pass():
    """Their kernels read the middle segment as the prefill-only bucket and
    skip it, so they must never be handed a static verify length."""
    assert static_verify_q_len(_config("qwen3_5", 5)) is None


def test_draft_layers_follow_the_target_model():
    """Draft layers are built under the target's config copy; even if the
    model being built is not DSv4, the target decides."""
    assert (
        static_verify_q_len(_config("other", 5, target_model_type="deepseek_v4")) == 6
    )
    assert (
        static_verify_q_len(_config("deepseek_v4", 5, target_model_type="other"))
        is None
    )


def test_knob_turns_the_static_pass_off_but_never_on(monkeypatch):
    """TPU_SPEC_STATIC_VERIFY=0 keeps every window on the mixed pass; the
    default cannot enable the pass for a model whose ops lack it."""
    monkeypatch.setattr(envs, "TPU_SPEC_STATIC_VERIFY", False)
    assert static_verify_q_len(_config("deepseek_v4", 5)) is None
    monkeypatch.setattr(envs, "TPU_SPEC_STATIC_VERIFY", True)
    assert static_verify_q_len(_config("deepseek_v4", 5)) == 6
    assert static_verify_q_len(_config("qwen3_5", 5)) is None


class _FakeJaxOp:
    def __init__(self, name, fn, **kwargs):
        self.name, self.fn = name, fn

    def register_fake(self, fn):
        pass


def _build_op(monkeypatch, layer, chunk_tokens):
    """`_build_indexer_op` of the CSA layer `layer`, with `pallas.jax_op` faked."""
    monkeypatch.setattr(envs, "TPU_STREAMIDX_CHUNK_TOKENS", chunk_tokens)
    monkeypatch.setattr(indexer_mod, "_indexer_op_cache", {})
    monkeypatch.setattr(indexer_mod.pallas, "jax_op", _FakeJaxOp)
    monkeypatch.setattr(indexer_mod, "get_pallas_op_mesh", lambda: None)
    monkeypatch.setattr(indexer_mod, "get_current_vllm_config", lambda: None)
    indexer = VllmDeepseekV4Indexer.__new__(VllmDeepseekV4Indexer)
    object.__setattr__(indexer, "topk_tokens", 512)
    object.__setattr__(indexer, "compress_ratio", 4)
    object.__setattr__(indexer, "softmax_scale", 0.088)
    object.__setattr__(indexer, "n_head", 64)
    object.__setattr__(
        indexer,
        "k_cache",
        SimpleNamespace(prefix=f"model.layers.{layer}.attn.indexer.k_cache"),
    )
    return indexer._build_indexer_op


def test_unchunked_layers_share_one_indexer_op(monkeypatch):
    build_first, build_second = (_build_op(monkeypatch, i, None) for i in (2, 4))
    first = build_first()
    assert build_second() is first
    assert first.fn.keywords["scheduling_group_slot"] is None


def _group_ids(fn):
    """Scheduling group ids in `fn`, lowered for TPU on an abstract mesh."""
    t, h, pages = 128, 4, 8
    args = (
        jax.ShapeDtypeStruct((t, h, 128), jnp.bfloat16),  # q
        jax.ShapeDtypeStruct((t, 64), jnp.float32),  # cos_sin
        jax.ShapeDtypeStruct((t, h), jnp.float32),  # indexer_weights
        jax.ShapeDtypeStruct(  # SEQ_ALONG_LANE cache
            (pages, compressor_config.SAL_SUBLANE_GROUPS, 4, 256), jnp.uint8
        ),
        jax.ShapeDtypeStruct((2,), jnp.int32),  # seq_lens
        jax.ShapeDtypeStruct((2, pages // 2), jnp.int32),  # page_indices
        jax.ShapeDtypeStruct((3,), jnp.int32),  # cu_q_lens
        jax.ShapeDtypeStruct((3,), jnp.int32),  # distribution
    )
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        text = export.export(jax.jit(fn), platforms=("tpu",))(*args).mlir_module()
    return set(map(int, re.findall(r'_scheduling_group_id\s*=\s*"?(\d+)', text)))


def test_chunked_layers_get_their_own_op_and_groups(monkeypatch):
    """A shared op would put one layer's scheduling groups in every layer."""
    first = _build_op(monkeypatch, 2, 32)()
    second = _build_op(monkeypatch, 4, 32)()
    assert first is not second and first.name != second.name
    streamidx_config.reset_scheduling_group_ids()
    ids = [_group_ids(op.fn) for op in (first, second)]
    # 128 tokens in chunks of 32: 4 chunks, 3 groups per layer.
    assert [len(layer) for layer in ids] == [3, 3], ids
    assert not ids[0] & ids[1], ids


def test_indexer_forward_looks_up_cos_sin():
    """VllmDeepseekV4Indexer.forward looks up per-token cos_sin at positions."""
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.forward_context import set_forward_context

    num_tokens, rotary_dim = 6, 8
    positions = torch.tensor([0, 3, 4, 7, 11, 15], dtype=torch.int32)
    table = torch.randn(32, rotary_dim)

    indexer = VllmDeepseekV4Indexer.__new__(VllmDeepseekV4Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.n_head = 2
    indexer.head_dim = 16
    indexer.topk_tokens = 4
    indexer.wq_b = lambda q: (torch.zeros(num_tokens, 2, 16), None)
    indexer.compressor = lambda *_a: None
    indexer.k_cache = SimpleNamespace(
        prefix="idx", kv_cache=torch.ones(1, dtype=torch.uint8)
    )
    captured: list[torch.Tensor] = []
    object.__setattr__(
        indexer,
        "indexer_op",
        lambda *args: captured.append(args[1]) or torch.zeros(num_tokens, 4),
    )

    attn_meta = {
        "idx": SimpleNamespace(
            seq_lens=torch.ones(1, dtype=torch.int32),
            block_tables=torch.zeros(1, 2, dtype=torch.int32),
            query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        )
    }
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    zeros = torch.zeros(num_tokens, 4)

    with set_forward_context(attn_meta, cfg):
        indexer.forward(
            zeros,
            zeros,
            torch.zeros(num_tokens, 2),
            positions,
            SimpleNamespace(cos_sin_cache=table),
        )

    assert torch.equal(captured[0], table.index_select(0, positions))
