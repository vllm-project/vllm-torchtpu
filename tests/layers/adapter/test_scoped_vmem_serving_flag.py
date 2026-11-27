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
"""Serving constructors carry TPU_ATTN_SCOPED_VMEM_SIZING to each kernel launch.

The kernel suites call the Pallas entry points directly, so a flag dropped by
a serving adapter would leave them green. These tests build each op through
its serving constructor. Paths that lower on an abstract TPU are checked on
the lowered launches; pooled and PCP paths, which need a real pool or
collectives, hand their kernel arguments to a recording stand-in.
"""

import functools
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, AxisType, Mesh, use_abstract_mesh
from kernels import test_kimi_pooled_kda as pooled_kda
from kernels.scoped_vmem_test_utils import (
    _abstract_tpu,
    _extract_tpu_custom_calls,
    _scoped_vmem_bytes,
    _sds,
)
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import config as gdn_config
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_wrapper
from vllm_torchtpu.layers.adapter.custom_ops import (
    gdn_attention_op,
    kda_attention_op,
    mamba_op_utils,
)
from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention,
)
from vllm_torchtpu.layers.core import gdn_attention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

_FLAG = "TPU_ATTN_SCOPED_VMEM_SIZING"
_BF16, _F32, _I32 = jnp.bfloat16, jnp.float32, jnp.int32
on_off = pytest.mark.parametrize("enabled", [False, True], ids=["off", "on"])


def _override_scoped_flag(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    # A module-dict entry shadows envs' lazy __getattr__, including under its
    # env cache. setitem deletes the entry on undo; setattr would restore the
    # looked-up value as a real attribute and freeze the flag afterwards.
    monkeypatch.setitem(vars(envs), _FLAG, enabled)


@pytest.fixture
def scoped_flag(monkeypatch):
    """Set TPU_ATTN_SCOPED_VMEM_SIZING for one test."""
    return functools.partial(_override_scoped_flag, monkeypatch)


def test_scoped_flag_override_leaves_env_lookup_live(monkeypatch):
    monkeypatch.delitem(vars(envs), _FLAG, raising=False)
    monkeypatch.setenv(_FLAG, "0")
    with pytest.MonkeyPatch.context() as mp:
        _override_scoped_flag(mp, True)
        assert envs.TPU_ATTN_SCOPED_VMEM_SIZING is True
    assert _FLAG not in vars(envs)
    assert envs.TPU_ATTN_SCOPED_VMEM_SIZING is False
    monkeypatch.setenv(_FLAG, "1")
    assert envs.TPU_ATTN_SCOPED_VMEM_SIZING is True


@pytest.fixture
def cpu_vllm_config_context():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _capture_jax_op(captured: dict):
    def fake_jax_op(_name, wrapped_fn, **_kwargs):
        captured["fn"] = wrapped_fn
        return MagicMock()

    return fake_jax_op


def _record_kernel_flags(seen: list):
    """Stand-in for fused_conv1d_gdn that records the two launch options."""

    def fake_kernel(*args, **kwargs):
        seen.append((kwargs["use_scoped_vmem_limit"], kwargs["use_cost_estimate"]))
        if args:
            num_tokens, num_heads = args[0].shape[0], args[1].shape[-1]
            return args[3], jnp.zeros((num_tokens, num_heads * 4), _F32)
        num_tokens = kwargs["qkv"].shape[0]
        width = kwargs["states"][1].shape[1] * kwargs["states"][1].shape[3]
        return kwargs["states"], jnp.zeros((num_tokens, width), kwargs["qkv"].dtype)

    return fake_kernel


def _assert_launches(calls, enabled: bool):
    """Flag on: estimated limits (verify windows keep the default) and costs."""
    assert calls
    for name, cfg in calls:
        match = re.search(r"_w(\d+)", name)
        window = int(match.group(1)) if match else 1
        default = gdn_config.GDNConfig.get_vmem_limit_bytes(window)
        limit = _scoped_vmem_bytes(cfg)
        assert ("cost_estimate" in cfg["custom_call_config"]) == enabled, name
        if enabled and window == 1:
            assert limit < default, name
        else:
            assert limit == default, name


def _vllm_config(*, pcp_size: int = 1, block_size: int = 16):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            tensor_parallel_size=1,
            cp_kv_cache_interleave_size=16,
            data_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            mamba_block_size=4096,
            mamba_page_size_padded=None,
            mamba_cache_mode=None,
            use_kda_recoverssm=False,
            use_replayssm=False,
        ),
        speculative_config=None,
        num_speculative_tokens=0,
    )


def _gdn_attn(*, n_kq: int = 2, n_v: int = 8, head_dim: int = 128, num_spec=0):
    attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
    attn.prefix = "language_model.model.layers.0.linear_attn"
    attn.num_k_heads = n_kq
    attn.num_v_heads = n_v
    attn.gqa_interleaved_layout = False
    attn.tp_size = 1
    attn.head_k_dim = head_dim
    attn.head_v_dim = head_dim
    attn.conv_kernel_size = 4
    attn.num_spec = num_spec
    attn.conv1d = SimpleNamespace(bias=None)
    attn.model_config = SimpleNamespace(
        dtype=torch.bfloat16, architecture="Qwen3_5MoeForCausalLM"
    )
    attn.cache_config = SimpleNamespace(
        mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="float32"
    )
    return attn


@on_off
def test_gdn_op_flag_reaches_lowered_launches(scoped_flag, enabled):
    scoped_flag(enabled)
    mesh = AbstractMesh(
        (1,),
        ("model",),
        (AxisType.Auto,),
        abstract_device=AbstractDevice("TPU v6 lite", 1, "tpu"),
    )
    captured = {}
    with (
        set_vllm_model_wrapper_context(mesh=mesh, vllm_config=_vllm_config()),
        patch.object(
            gdn_attention_op.pallas, "jax_op", side_effect=_capture_jax_op(captured)
        ),
    ):
        _gdn_attn(num_spec=3)._build_gdn_op()

    tokens, seqs, slots, dim = 32, 8, 33, (2 * 2 + 8) * 128
    args = (
        _sds((tokens, dim), _BF16),
        _sds((tokens, 8), _BF16),
        _sds((tokens, 8), _BF16),
        _sds((slots, 3, dim), _F32),
        _sds((slots, 8, 128, 128), _F32),
        _sds((dim, 1, 4), _BF16),
        None,
        _sds((8,), _F32),
        _sds((8,), _F32),
        _sds((seqs,), _I32),
        _sds((seqs + 1,), _I32),
        _sds((3,), _I32),
        _sds((seqs,), _I32),
        _sds((slots,), _I32),
    )
    with use_abstract_mesh(mesh):
        calls = _extract_tpu_custom_calls(captured["fn"], args)
        # The verify window and the mixed prefill segment.
        assert len(calls) == 2
        _assert_launches(calls, enabled)


@on_off
def test_gdn_pooled_op_flag_reaches_kernel(
    monkeypatch, scoped_flag, cpu_vllm_config_context, enabled
):
    scoped_flag(enabled)
    seen = []
    monkeypatch.setattr(gdn_wrapper, "fused_conv1d_gdn", _record_kernel_flags(seen))
    monkeypatch.setattr(mamba_op_utils, "_shared_mamba_ops", {})
    captured = {}
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("model",))
    with (
        set_vllm_model_wrapper_context(
            mesh=mesh, vllm_config=_vllm_config(block_size=4)
        ),
        patch.object(
            gdn_attention_op.pallas, "jax_op", side_effect=_capture_jax_op(captured)
        ),
    ):
        _gdn_attn(n_kq=2, n_v=2, head_dim=4)._build_pooled_gdn_op()

    tokens = 8
    captured["fn"](
        jnp.zeros((tokens, 16), _BF16),
        jnp.zeros((tokens, 2), _BF16),
        jnp.zeros((tokens, 2), _BF16),
        jnp.zeros((6, 4, 2, 8), _BF16),
        jnp.zeros((16, 1, 4), _BF16),
        None,
        jnp.zeros((2,), _F32),
        jnp.zeros((2,), _F32),
        jnp.array([1, 2, 3, 0], _I32),
        jnp.array([0, 1, 2, 8, 8], _I32),
        jnp.array([2, 2, 3], _I32),
        jnp.array([10, 20, 8191, 0], _I32),
    )
    assert seen == [(enabled, enabled)]


@on_off
def test_gdn_pcp_op_flag_reaches_kernel(monkeypatch, scoped_flag, enabled):
    scoped_flag(enabled)
    seen = []
    monkeypatch.setattr(gdn_wrapper, "fused_conv1d_gdn", _record_kernel_flags(seen))
    # The ragged-exchange layout check reads the chip; present a validated one.
    monkeypatch.setattr(
        gdn_attention.pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("pcp", "tp"))
    captured = {}
    with (
        set_vllm_model_wrapper_context(
            mesh=MagicMock(), vllm_config=_vllm_config(pcp_size=1)
        ),
        patch.object(gdn_attention_op, "get_pcp_world_size", return_value=1),
        patch.object(gdn_attention_op, "get_or_create_pcp_mesh", return_value=mesh),
        patch.object(
            gdn_attention_op.pallas, "jax_op", side_effect=_capture_jax_op(captured)
        ),
    ):
        _gdn_attn(n_kq=2, n_v=2, head_dim=4)._build_gdn_op(pcp_streaming=True)

    tokens, dim = 16, (2 * 2 + 2) * 4
    # Trace only: XLA:CPU cannot run the PCP ragged all-to-all.
    jax.make_jaxpr(captured["fn"])(
        jnp.zeros((tokens, dim), _F32),
        jnp.zeros((tokens, 2), _F32),
        jnp.zeros((tokens, 2), _F32),
        jnp.zeros((5, 3, dim), _F32),
        jnp.zeros((5, 2, 4, 4), _F32),
        jnp.zeros((dim, 1, 4), _F32),
        None,
        jnp.zeros((2,), _F32),
        jnp.zeros((2,), _F32),
        jnp.array([1, 2, 0, 0], _I32),
        jnp.array([0, 9, 16, 16, 16], _I32),
        jnp.array([0, 0, 2], _I32),
        jnp.array([9, 7, 0, 0], _I32),
    )
    assert seen == [(enabled, enabled)]


def _kda_dense_args(num_spec_tokens: int, heads: int = 8, head_dim: int = 128):
    tokens, seqs = 32, 8
    slots = seqs * (num_spec_tokens + 1) + 1
    dim = 3 * heads * head_dim
    return (
        _sds((tokens, dim), _BF16),
        _sds((tokens, heads * head_dim), _BF16),
        _sds((tokens, heads), _BF16),
        _sds((tokens, heads * head_dim), _BF16),
        _sds((slots, 3, dim), _F32),
        _sds((slots, heads, head_dim, head_dim), _F32),
        _sds((4, 3, heads, head_dim), _BF16),
        _sds((heads,), _F32),
        _sds((heads * head_dim,), _F32),
        _sds((head_dim,), _BF16),
        _sds((seqs + 1,), _I32),
        _sds((seqs,), _I32),
        _sds((seqs,), _I32),
        _sds((3,), _I32),
        _sds((3,), _I32),
        _sds((slots,), _I32),
    )


@on_off
@pytest.mark.parametrize("num_spec_tokens", [0, 3])
def test_kda_dispatched_op_flag_reaches_lowered_launches(
    scoped_flag, enabled, num_spec_tokens
):
    scoped_flag(enabled)
    captured = {}
    with patch.object(
        kda_attention_op.pallas, "jax_op", side_effect=_capture_jax_op(captured)
    ):
        kda_attention_op.build_kimi_dispatched_kda_op(
            "model.layers.0.self_attn",
            lower_bound=None,
            eps=1e-5,
            state_dim_first=False,
            num_spec_tokens=num_spec_tokens,
        )

    with _abstract_tpu("TPU v6 lite"):
        calls = _extract_tpu_custom_calls(
            captured["fn"], _kda_dense_args(num_spec_tokens)
        )
        # Decode or verify window, then the mixed prefill segment.
        assert len(calls) == 2
        _assert_launches(calls, enabled)


@on_off
def test_kda_pooled_op_flag_reaches_kernel(monkeypatch, scoped_flag, enabled):
    scoped_flag(enabled)
    seen = []
    monkeypatch.setattr(gdn_wrapper, "fused_conv1d_gdn", _record_kernel_flags(seen))
    captured = {}
    block_tokens = pooled_kda.BLOCK_SIZE * pooled_kda.PACK
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=block_tokens))
    with patch.object(
        kda_attention_op.pallas, "jax_op", side_effect=_capture_jax_op(captured)
    ):
        kda_attention_op.build_kimi_pooled_kda_op(
            "model.layers.0.self_attn",
            lower_bound=None,
            eps=1e-5,
            vllm_config=vllm_config,
        )

    rng = np.random.default_rng(0)
    inputs = pooled_kda._inputs(rng, [1, 5, 3], [9, 5, 11])
    block_ids = [1, 2, 3]
    pool = _sds(
        (
            pooled_kda.NUM_BLOCKS,
            pooled_kda.BLOCK_SIZE,
            pooled_kda.PACK,
            pooled_kda.LANES,
        ),
        _BF16,
    )
    # Trace only: the pool region gathers and scatters are TPU Pallas kernels.
    jax.make_jaxpr(captured["fn"])(
        inputs["mixed_qkv"],
        inputs["raw_gate"],
        inputs["beta"],
        inputs["output_gate"],
        pool,
        inputs["conv_weight"],
        inputs["a_log"],
        inputs["dt_bias"],
        inputs["norm_weight"],
        inputs["query_start_loc"],
        jnp.asarray(block_ids, _I32),
        inputs["seq_lens"],
        jnp.asarray([1, 3, 3], _I32),
    )
    assert seen == [(enabled, enabled)]
