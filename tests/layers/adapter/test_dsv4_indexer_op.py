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

from types import SimpleNamespace

import jax.numpy as jnp
import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer import (
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
            zeros,
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
