# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_GDN_STEP_KERNEL_METADATA under vLLM-style dynamic token shapes.

vLLM marks the token dim of the model inputs dynamic and compiles ranges of
it, so the GDN ops traced inside the forward see a symbolic token count. Their
fakes must not specialize on it: a fake that turns the token count into a
Python int guards on it, and dynamo raises ConstraintViolationError for the
dynamic dim (c32 startup died this way at compile range 8192). CPU only:
dynamo traces through the ops' fakes; the backend never runs the graph.
"""

from types import SimpleNamespace as NS

import jax
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.qwen4_exp import gdn
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

KV, VH, D, NSPEC, SEQS, BLOCKS = 2, 6, 128, 3, 24, 4
CONV = 2 * KV * D + VH * D


def _ops(monkeypatch):
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu import envs

    # The pre-fix fake ran the metadata in jax.eval_shape, which sizes tiles
    # from the TPU info; fake it so that path is traced on CPU too.
    info = NS(num_lanes=128, num_sublanes=8, vmem_capacity_bytes=64 * 1024**2)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *a, **k: info)

    for name in (
        "TPU_GDN_RAW_QKV",
        "TPU_GDN_RAW_BA",
        "VLLM_TPU_GDN_PREFILL_RAW_QKV",
        "VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR",
    ):
        monkeypatch.setattr(envs, name, True)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    config = NS(
        cache_config=NS(block_size=512),
        scheduler_config=NS(max_num_batched_tokens=8192, max_num_seqs=SEQS),
    )
    with set_vllm_model_wrapper_context(mesh=mesh, vllm_config=config):
        cls = gdn.Qwen4ExpGatedDeltaNetAttention
        attn = cls.__new__(cls)
        attn.num_k_heads, attn.num_v_heads, attn.tp_size = KV, VH, 1
        attn.gqa_interleaved_layout = False
        attn.head_k_dim = attn.head_v_dim = D
        attn.conv_kernel_size, attn.num_spec = 4, NSPEC
        attn._pooled_conv_state_dtype = torch.bfloat16
        attn.model_config = NS(
            dtype=torch.bfloat16, architecture="Qwen4ExpForCausalLM"
        )
        attn.cache_config = NS(
            mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="float32"
        )
        step = attn._build_pooled_gdn_op(kernel_metadata="step")
        layer = attn._build_pooled_gdn_op(kernel_metadata="layer")
    return step, layer


def test_step_kernel_metadata_traces_with_dynamic_tokens(monkeypatch):
    step, layer = _ops(monkeypatch)
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        (output,) = [n for n in gm.graph.nodes if n.op == "output"]
        count = len(output.args[0])
        # Trace only: the TPU ops never run on CPU.
        return lambda *args: tuple(torch.empty(0) for _ in range(count))

    def forward(input_ids, embed, b, a, pool, conv_w, a_log, dt_bias, *md):
        qkv = embed[input_ids]  # token dim follows the dynamic input_ids
        bt, at = b[input_ids], a[input_ids]
        packed = step(qkv, bt, at, pool, conv_w, None, a_log, dt_bias, *md)
        out0 = layer(qkv, bt, at, pool, conv_w, None, a_log, dt_bias, *md, packed)
        out1 = layer(qkv, bt, at, pool, conv_w, None, a_log, dt_bias, *md, packed)
        return out0 + out1

    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32)
    md = (
        i32(np.zeros(SEQS)),
        i32(np.zeros(SEQS + 1)),
        i32([0, 1, 1]),
        i32(np.zeros(SEQS)),
        i32(np.zeros(BLOCKS)),
        i32(np.zeros((SEQS, NSPEC + 1))),
    )
    weights = (
        torch.zeros((64, CONV), dtype=torch.bfloat16),
        torch.zeros((64, VH), dtype=torch.bfloat16),
        torch.zeros((64, VH), dtype=torch.bfloat16),
        torch.zeros((BLOCKS, 512, 1, 4, 256), dtype=torch.float8_e4m3fn),
        torch.zeros((4, 1, CONV), dtype=torch.float32),
        torch.zeros((VH,), dtype=torch.float32),
        torch.zeros((VH,), dtype=torch.float32),
    )
    for tokens in (8192, 4096):
        input_ids = torch.zeros((tokens,), dtype=torch.int64)
        # As vLLM's support_torch_compile does, over a compile range covering 8192.
        torch._dynamo.mark_dynamic(input_ids, 0, min=2, max=16384)
        compiled(input_ids, *weights, *md)
    assert graphs, "dynamo produced no graph"
    # One graph for both token counts: nothing specialized on the token dim.
    assert len(graphs) == 1
    placeholders = [
        n.meta.get("example_value") for n in graphs[0].graph.nodes if n.op == "placeholder"
    ]
    assert any(
        isinstance(v, torch.Tensor) and isinstance(v.shape[0], torch.SymInt)
        for v in placeholders
    )


@pytest.mark.parametrize("max_tokens,max_seqs", [(8192, 24), (16384, 48)])
def test_packed_size_is_static(max_tokens, max_seqs):
    from vllm_torchtpu.kernels.gdn.v3 import wrapper

    a = wrapper.packed_kernel_metadata_size(max_tokens, max_seqs, NSPEC)
    assert a == wrapper.packed_kernel_metadata_size(max_tokens, max_seqs, NSPEC)
    assert a > 2 * 6 * max_tokens
