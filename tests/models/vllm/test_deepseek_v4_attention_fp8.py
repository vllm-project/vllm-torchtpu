# SPDX-License-Identifier: Apache-2.0
"""FP8 CSA must register separately from the original numerical path."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
    VllmDeepseekV4MLAAttention,
    _pallas_op_cache,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def build_op(enabled, compress_ratio=4):
    # Registration needs runtime geometry, not a checkpoint or device tensors.
    layer = SimpleNamespace(
        compress_ratio=compress_ratio,
        custom_prefix="test_fp8_attention",
        window_size=128,
        mla_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(block_size=32, kv_cache=None),
            kv_cache=None,
            scale=512**-0.5,
            n_local_heads=64,
            head_dim=512,
        ),
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(head_dim=512)),
        additional_config={"tpu_csa_fp8_compute": enabled},
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("model",))
    with set_vllm_model_wrapper_context(mesh=mesh, vllm_config=config):
        return VllmDeepseekV4MLAAttention._build_attn_op(layer)[0]


def test_fp8_csa_has_distinct_registered_op_and_reuses_own_cache():
    baseline = build_op(False)
    fp8 = build_op(True)
    assert fp8 is not baseline
    assert build_op(True) is fp8
    assert build_op(False) is baseline
    keys = [k for k, v in _pallas_op_cache.items() if v is fp8]
    assert len(keys) == 1 and "_fp8_qk_pv" in keys[0]


@pytest.mark.parametrize("compress_ratio", [1, 128])
def test_csa_fp8_option_does_not_change_other_attention(compress_ratio):
    assert build_op(False, compress_ratio) is build_op(True, compress_ratio)


def test_csa_fp8_rejects_ambiguous_configuration():
    with pytest.raises(ValueError, match="must be a boolean"):
        build_op("false")


@pytest.mark.parametrize("name", ["hca_preprocess_attn_op", "hca_compressor_attn_op"])
@pytest.mark.parametrize("real_cache", [False, True])
def test_hca_op_getter_reuses_profile_and_real_registration(name, real_cache):
    import torch

    calls = []
    registered = object()
    layer = SimpleNamespace(
        mla_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(
                kv_cache=torch.empty(1) if real_cache else None
            )
        )
    )

    def build(**kwargs):
        calls.append(kwargs)
        return registered, real_cache

    layer._build_attn_op = build
    if real_cache:
        # A profiling op must not be reused after binding real caches.
        setattr(layer, "_" + name + "_prof_instance", object())
    getter = getattr(VllmDeepseekV4MLAAttention, name).fget
    assert getter(layer) is registered
    assert getter(layer) is registered
    assert len(calls) == 1
    expected = {"preprocess": True}
    if name == "hca_compressor_attn_op":
        expected["compress"] = True
    assert calls[0] == expected
