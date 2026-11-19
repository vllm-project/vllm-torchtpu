# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL GDN (rule 12): the VLLM_TPU_GDN_LEAN_GLUE3 pooled GDN jax op is built by the REAL torch_tpu jax_op.

The flag gives the pooled GDN op a second jax function (``wrapped_fn_lean``: today's 14 operands plus the stacked
state indices and the layer's row). torch_tpu's pallas.jax_op checks the function's signature when the op is built
(``inspect.signature(fn, eval_str=True)`` + ``_verify_signature``: G1 died at engine init on a missing annotation,
T94c). Here the image's own jax_op builds the flag-off and the flag-on op (template: tests/q38_l1g1/
test_g1_jax_op_signature.py, L1-G1FIX; credit):
  J1 both ops build through the real jax_op (a spy records each call and calls the real one);
  J2 their torch schemas: flag off 14 operands ending in ckpt_indices (today's), flag on 16 ending in
     stacked_state_indices, layer_row; two results; donate the pool (operand 3);
  J3 each recorded function passes the real ``_verify_signature`` directly;
  J0 control: the real check rejects an unannotated twin of the lean signature.
Run on a CDK CPU pod (torch_tpu installed); elsewhere the module skips, and the proof harness fails on any skip.
"""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("torch_tpu")

import torch  # noqa: E402
from torch_tpu._internal import pallas as tt_pallas  # noqa: E402
from torch_tpu._internal.pallas import pallas as tt_impl  # noqa: E402
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op as op  # noqa: E402
from vllm_torchtpu.models.vllm.qwen4_exp.gdn import Qwen4ExpGatedDeltaNetAttention  # noqa: E402
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context  # noqa: E402

MISSING = "Missing argument type annotation for JAX function"
_REAL_JAX_OP = tt_pallas.jax_op  # captured once: a spy must never wrap another spy
TODAY = ["mixed_qkv", "b", "a", "recurrent_state", "conv_weight", "conv_bias", "A_log", "dt_bias", "state_indices",
         "query_start_loc", "distribution", "seq_lens", "slot_read_offsets", "ckpt_indices"]


def _layer(prefix):
    attn = Qwen4ExpGatedDeltaNetAttention.__new__(Qwen4ExpGatedDeltaNetAttention)
    attn.prefix = prefix
    attn.num_k_heads, attn.num_v_heads, attn.tp_size = 2, 6, 1
    attn.head_k_dim = attn.head_v_dim = 128
    attn.conv_kernel_size, attn.num_spec = 4, 3
    attn.gqa_interleaved_layout = False
    attn.conv1d = SimpleNamespace(bias=None)
    attn.model_config = SimpleNamespace(dtype=torch.bfloat16, architecture="Qwen3_5MoeForCausalLM")
    attn.cache_config = SimpleNamespace(mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="float32")
    attn._pooled_conv_state_dtype = torch.float32
    return attn


def _build(monkeypatch, lean: bool):
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_LEAN_GLUE3", lean)
    monkeypatch.setattr(op, "_POOLED_GDN_PREFIXES", set())
    calls = []

    def spy(name, fn=None, /, **kw):
        built = _REAL_JAX_OP(name, fn, **kw)
        calls.append((name, fn, kw))
        return built

    monkeypatch.setattr(tt_pallas, "jax_op", spy)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, tensor_parallel_size=1,
                                        cp_kv_cache_interleave_size=16, data_parallel_size=1),
        cache_config=SimpleNamespace(block_size=1664, mamba_block_size=4096, mamba_page_size_padded=None,
                                     mamba_cache_mode=None, use_kda_recoverssm=False, use_replayssm=False),
        speculative_config=None, num_speculative_tokens=3,
    )
    attn = _layer("model.layers.0.linear_attn")
    with (
        set_current_vllm_config(config),
        mock.patch.dict(op._shared_gdn_ops, {}, clear=True),
        set_vllm_model_wrapper_context(mesh=SimpleNamespace(shape={"model": 1}), vllm_config=vllm_config),
    ):
        attn._build_pooled_gdn_op()
    assert attn._lean_glue3 == lean
    assert len(calls) == 1, [n for n, *_ in calls]
    return calls[0]


def _schema(name: str):
    ns, opname = name.split("::")
    return getattr(getattr(torch.ops, ns), opname).default._schema


@pytest.mark.parametrize("lean", [False, True])
def test_j1_j2_j3_pooled_op_real_jax_op(monkeypatch, lean):
    name, fn, kw = _build(monkeypatch, lean)
    s = _schema(name)
    args = [a.name for a in s.arguments]
    want = TODAY + (["stacked_state_indices", "layer_row"] if lean else [])
    print(f"PBL_GDN_J2 lean {lean} {name}: {len(args)} args -> {len(s.returns)} results; donate "
          f"{kw.get('donate_argnums')}; types {sorted({str(a.type) for a in s.arguments})}")
    assert re.fullmatch(r"pallas::gdn_attention_pooled%s_\d+" % ("_lg3" if lean else ""), name), name
    assert args == want, args
    assert len(s.returns) == 2
    assert kw.get("donate_argnums") == (3,)
    sig = inspect.signature(inspect.unwrap(fn), eval_str=True)
    tt_impl._verify_signature(sig, bound_names=tt_impl._get_bound_names(fn))
    print(f"PBL_GDN_J3 lean {lean}: real _verify_signature OK ({len(sig.parameters)} parameters)")


def test_j0_the_real_check_rejects_an_unannotated_twin():
    def lean_twin(mixed_qkv, b, a, recurrent_state, conv_weight, conv_bias, A_log, dt_bias, state_indices,
                  query_start_loc, distribution, seq_lens, slot_read_offsets=None, ckpt_indices=None,
                  stacked_state_indices=None, layer_row=None):
        return recurrent_state, mixed_qkv

    with pytest.raises(ValueError, match=re.escape(MISSING)):
        tt_pallas.jax_op("pallas::q38_portbl_gdn_unannotated_twin", lean_twin, donate_argnums=(3,))
    print("PBL_GDN_J0 real jax_op rejects the unannotated lean twin")
