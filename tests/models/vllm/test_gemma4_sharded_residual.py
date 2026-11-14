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
"""Unit tests for Gemma 4 token-sharded residual and split FP8 linear ops."""

from __future__ import annotations

import sys
import threading
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.layers.adapter import prequantized_linear as pq
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.models.vllm import gemma4_patch as gp

HIDDEN = 24
HEADS = 8
HEAD_DIM = 4
INTERMEDIATE = 32
PLE_DIM = 8
NUM_LAYERS = 3


# ---------------------------------------------------------------------------
# Split FP8 ops (pure JAX + adapter checks)
# ---------------------------------------------------------------------------


def _fp8_inputs(m: int = 16, k: int = 48, n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((m, k)) * 3.0, dtype=jnp.bfloat16)
    x = x.at[3].set(0)  # all-zero row exercises the scale == 0 path
    w = jnp.asarray(rng.standard_normal((k, n)), dtype=jnp.float8_e4m3fn)
    w_scale = jnp.asarray(rng.uniform(0.01, 0.1, (n,)), dtype=jnp.float32)
    return x, w, w_scale


def test_split_ops_match_fused_fp8_matmul_bitwise():
    x, w, w_scale = _fp8_inputs()
    fused = xla_quantized_matmul(x, w, w_scale)
    x_q, x_scale = pq.quantize_rows_jax(x)
    split = pq.prequantized_matmul_jax(x_q, x_scale, w, w_scale)
    assert x_q.dtype == jnp.uint8 and x_scale.shape == (x.shape[0], 1)
    assert split.dtype == fused.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(split.astype(jnp.float32)), np.asarray(fused.astype(jnp.float32))
    )


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_quantizing_row_shards_matches_quantizing_all_rows(tp: int):
    x, w, w_scale = _fp8_inputs()
    full_q, full_scale = pq.quantize_rows_jax(x)
    shards = [pq.quantize_rows_jax(part) for part in jnp.split(x, tp, axis=0)]
    gathered_q = jnp.concatenate([q for q, _ in shards], axis=0)
    gathered_scale = jnp.concatenate([s for _, s in shards], axis=0)
    np.testing.assert_array_equal(np.asarray(gathered_q), np.asarray(full_q))
    np.testing.assert_array_equal(np.asarray(gathered_scale), np.asarray(full_scale))
    out = pq.prequantized_matmul_jax(gathered_q, gathered_scale, w, w_scale)
    fused = xla_quantized_matmul(x, w, w_scale)
    np.testing.assert_array_equal(
        np.asarray(out.astype(jnp.float32)), np.asarray(fused.astype(jnp.float32))
    )


def test_split_ops_jit():
    x, w, w_scale = _fp8_inputs(seed=1)

    @jax.jit
    def run(x_in, w_in, s_in):
        x_q, x_scale = pq.quantize_rows_jax(x_in)
        return pq.prequantized_matmul_jax(x_q, x_scale, w_in, s_in)

    np.testing.assert_array_equal(
        np.asarray(run(x, w, w_scale).astype(jnp.float32)),
        np.asarray(xla_quantized_matmul(x, w, w_scale).astype(jnp.float32)),
    )


def test_check_prequantized_linear_and_quantize_rows_guards():
    def valid_layer() -> torch.nn.Module:
        mod = torch.nn.Module()
        mod.bias = None
        mod.weight = torch.nn.Parameter(
            torch.zeros(16, 32, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        mod.register_buffer("weight_scale", torch.ones(32, dtype=torch.float32))
        setattr(mod, WEIGHT_FLIPPED_ATTR, True)
        mod.quant_method = SimpleNamespace(w8a16_max_rows=0)
        return mod

    pq.check_prequantized_linear(valid_layer())

    biased = valid_layer()
    biased.bias = torch.zeros(32)
    with pytest.raises(NotImplementedError, match="bias"):
        pq.check_prequantized_linear(biased)

    bf16_w = valid_layer()
    bf16_w.weight = torch.nn.Parameter(
        torch.zeros(16, 32, dtype=torch.bfloat16), requires_grad=False
    )
    with pytest.raises(NotImplementedError, match="float8_e4m3fn"):
        pq.check_prequantized_linear(bf16_w)

    block_s = valid_layer()
    block_s.weight_scale = torch.ones(1, 1, 1, 32, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="1-D"):
        pq.check_prequantized_linear(block_s)

    unflipped = valid_layer()
    setattr(unflipped, WEIGHT_FLIPPED_ATTR, False)
    with pytest.raises(NotImplementedError, match="flipped"):
        pq.check_prequantized_linear(unflipped)

    kept_layout = valid_layer()
    setattr(kept_layout, KEEP_VLLM_LAYOUT_ATTR, True)
    with pytest.raises(NotImplementedError, match="flipped"):
        pq.check_prequantized_linear(kept_layout)

    w8a16 = valid_layer()
    w8a16.quant_method = SimpleNamespace(w8a16_max_rows=64)
    with pytest.raises(NotImplementedError, match="TPU_FP8_LINEAR_W8A16_MAX_ROWS"):
        pq.check_prequantized_linear(w8a16)

    pq.ensure_ops()
    with pytest.raises(ValueError, match="2-D"):
        pq.quantize_rows(torch.zeros(2, 4, 8, dtype=torch.bfloat16))
    with pytest.raises(NotImplementedError, match="bfloat16"):
        pq.quantize_rows(torch.zeros(4, 8, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Layer data flow against simulated TP ranks
# ---------------------------------------------------------------------------


class _TpSim:
    """Barrier-synchronized collectives across ``n`` simulated TP rank threads."""

    def __init__(self, n: int):
        self.n = n
        self.local = threading.local()
        self._submitted = threading.Barrier(n)
        self._consumed = threading.Barrier(n)
        self._slots: list[torch.Tensor | None] = [None] * n
        self.calls = {"all_reduce": 0, "all_gather": 0, "reduce_scatter": 0}

    @property
    def rank(self) -> int:
        return self.local.rank

    def _exchange(self, x: torch.Tensor, name: str) -> list[torch.Tensor]:
        if self.rank == 0:
            self.calls[name] += 1
        self._slots[self.rank] = x
        self._submitted.wait(timeout=30)
        parts = [slot for slot in self._slots if slot is not None]
        self._consumed.wait(timeout=30)
        return parts

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(self._exchange(x, "all_reduce")).sum(0)

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.cat(self._exchange(x, "all_gather"), dim=dim)

    def reduce_scatter(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0
        total = torch.stack(self._exchange(x, "reduce_scatter")).sum(0)
        rows = total.shape[0] // self.n
        return total[self.rank * rows : (self.rank + 1) * rows]


class _Linear:
    """Stand-in for a vLLM parallel linear returning ``(out, None)``."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.bias = None
        self.reduce_results = True

    def __call__(self, x: torch.Tensor):
        return x @ self.weight, None


def _rms_norm(weight: torch.Tensor):
    def norm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight

    return norm


def _make_weights(seed: int):
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, dtype=torch.float64) * 0.2

    return {
        "qkv": rand(HIDDEN, 3 * HEADS * HEAD_DIM),
        "o": rand(HEADS * HEAD_DIM, HIDDEN),
        "gate_up": rand(HIDDEN, 2 * INTERMEDIATE),
        "down": rand(INTERMEDIATE, HIDDEN),
        "norms": [1.0 + rand(HIDDEN) for _ in range(5)],
        "ple_gate": rand(HIDDEN, PLE_DIM),
        "ple_proj": rand(PLE_DIM, HIDDEN),
    }


def _make_layer(
    weights: dict, layer_idx: int, rank: int, tp: int, *, with_ple: bool = False
):
    heads = HEADS // tp
    q_size = heads * HEAD_DIM

    def head_cols(block: int) -> slice:
        start = block * HEADS * HEAD_DIM + rank * q_size
        return slice(start, start + q_size)

    qkv_w = torch.cat([weights["qkv"][:, head_cols(b)] for b in range(3)], dim=1)
    inter = INTERMEDIATE // tp
    gate_cols = slice(rank * inter, (rank + 1) * inter)
    up_cols = slice(INTERMEDIATE + rank * inter, INTERMEDIATE + (rank + 1) * inter)
    gate_up_w = torch.cat(
        [weights["gate_up"][:, gate_cols], weights["gate_up"][:, up_cols]], dim=1
    )

    def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.tanh(q) + torch.cumsum(k * v, dim=0) * 0.05

    def act_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.gelu(gate, approximate="tanh") * up

    attn = SimpleNamespace(
        is_kv_shared_layer=False,
        qkv_proj=_Linear(qkv_w),
        o_proj=_Linear(weights["o"][rank * q_size : (rank + 1) * q_size]),
        q_size=q_size,
        kv_size=q_size,
        num_heads=heads,
        num_kv_heads=heads,
        head_dim=HEAD_DIM,
        q_norm=lambda x: x,
        k_norm=lambda x: x,
        v_norm=lambda x: x,
        rotary_emb=lambda positions, q, k: (q, k),
        attn=attend,
    )
    mlp = SimpleNamespace(
        gate_up_proj=_Linear(gate_up_w),
        down_proj=_Linear(weights["down"][rank * inter : (rank + 1) * inter]),
        act_fn=act_and_mul,
    )
    norms = [_rms_norm(w) for w in weights["norms"]]
    return SimpleNamespace(
        layer_idx=layer_idx,
        enable_moe_block=False,
        self_attn=attn,
        mlp=mlp,
        input_layernorm=norms[0],
        post_attention_layernorm=norms[1],
        pre_feedforward_layernorm=norms[2],
        post_feedforward_layernorm=norms[3],
        per_layer_input_gate=(lambda x: x @ weights["ple_gate"]) if with_ple else None,
        per_layer_projection=(lambda x: x @ weights["ple_proj"]) if with_ple else None,
        post_per_layer_input_norm=norms[4] if with_ple else None,
        layer_scalar=torch.tensor([0.75], dtype=torch.float64),
    )


def _run_stack(
    monkeypatch: pytest.MonkeyPatch,
    tp: int,
    num_tokens: int,
    mode: str,
    min_rows: int,
    *,
    with_ple: bool = False,
):
    sim = _TpSim(tp)
    monkeypatch.setattr(gp, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(gp, "get_tensor_model_parallel_rank", lambda: sim.rank)
    monkeypatch.setattr(gp, "_get_pp_world_size", lambda: 1)
    monkeypatch.setattr(gp, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(gp, "get_dcp_world_size", lambda: 1)
    monkeypatch.setattr(gp, "tensor_model_parallel_all_reduce", sim.all_reduce)
    monkeypatch.setattr(gp, "tensor_model_parallel_all_gather", sim.all_gather)
    monkeypatch.setattr(gp, "tensor_model_parallel_reduce_scatter", sim.reduce_scatter)
    monkeypatch.setattr(pq, "ensure_ops", lambda: None)
    monkeypatch.setattr(pq, "check_prequantized_linear", lambda _layer: None)

    def quantize_rows(x: torch.Tensor):
        scale = x.abs().amax(-1, keepdim=True)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return x / scale, scale

    monkeypatch.setattr(pq, "quantize_rows", quantize_rows)
    monkeypatch.setattr(
        pq, "prequantized_linear", lambda layer, x_q, x_scale: layer(x_q * x_scale)[0]
    )

    weights = [_make_weights(seed) for seed in range(NUM_LAYERS)]
    g = torch.Generator().manual_seed(123)
    embedding = torch.randn(num_tokens, HIDDEN, generator=g, dtype=torch.float64)
    ple_inputs = [
        torch.randn(num_tokens, PLE_DIM, generator=g, dtype=torch.float64)
        if with_ple
        else None
        for _ in range(NUM_LAYERS)
    ]
    positions = torch.arange(num_tokens)
    outputs: list[torch.Tensor | None] = [None] * tp
    shard_rows: list[int | None] = [None] * tp
    errors: list[BaseException] = []

    def rank_main(rank: int) -> None:
        try:
            sim.local.rank = rank
            layers = [
                _make_layer(weights[i], i, rank, tp, with_ple=with_ple)
                for i in range(NUM_LAYERS)
            ]
            for layer in layers:
                gp.configure_layer(layer, NUM_LAYERS, mode, min_rows)
                assert layer.self_attn.o_proj.reduce_results is False
                assert layer.mlp.down_proj.reduce_results is False
            hidden = embedding.clone()
            for idx, layer in enumerate(layers):
                ple = ple_inputs[idx]
                hidden, residual = gp.sharded_residual_forward(
                    layer,
                    positions,
                    hidden,
                    None,
                    per_layer_input=ple.clone() if ple is not None else None,
                )
                assert residual is None
                if layer.layer_idx == 0:
                    shard_rows[rank] = hidden.shape[0]
            outputs[rank] = hidden
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            sim._submitted.abort()
            sim._consumed.abort()

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(tp)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    if errors:
        raise errors[0]
    return outputs, shard_rows, sim


_NEVER = 1 << 30


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("mode", [gp.MODE_BF16, gp.MODE_FP8])
@pytest.mark.parametrize("with_ple", [False, True])
def test_sharded_residual_matches_replicated_residual(
    monkeypatch: pytest.MonkeyPatch, tp: int, mode: str, with_ple: bool
):
    num_tokens = 16
    reference, ref_rows, ref_sim = _run_stack(
        monkeypatch, tp, num_tokens, mode, _NEVER, with_ple=with_ple
    )
    sharded, rows, sim = _run_stack(
        monkeypatch, tp, num_tokens, mode, 0, with_ple=with_ple
    )

    assert ref_rows == [num_tokens] * tp
    assert ref_sim.calls == {
        "all_reduce": 2 * NUM_LAYERS,
        "all_gather": 0,
        "reduce_scatter": 0,
    }
    assert rows == [num_tokens // tp] * tp
    gathers_per_matmul = 2 if mode == gp.MODE_FP8 else 1
    assert sim.calls == {
        "all_reduce": 0,
        "all_gather": 2 * NUM_LAYERS * gathers_per_matmul + 1,
        "reduce_scatter": 2 * NUM_LAYERS,
    }
    for rank in range(tp):
        out_r = sharded[rank]
        ref_r = reference[rank]
        ref_0 = reference[0]
        assert out_r is not None and ref_r is not None and ref_0 is not None
        assert out_r.shape == (num_tokens, HIDDEN)
        torch.testing.assert_close(ref_r, ref_0, rtol=0, atol=1e-12)
        torch.testing.assert_close(out_r, ref_0, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("num_tokens", "expect_sharded"),
    [
        (16, False),  # below min_rows
        (32, True),  # aligned at min_rows
        (34, False),  # >= min_rows, but not divisible by tp=4
        (64, True),  # aligned above min_rows
    ],
)
def test_min_rows_and_alignment_gate_by_token_bucket(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, expect_sharded: bool
):
    tp = 4
    reference, _, _ = _run_stack(monkeypatch, tp, num_tokens, gp.MODE_BF16, _NEVER)
    gated, rows, sim = _run_stack(monkeypatch, tp, num_tokens, gp.MODE_BF16, 32)
    assert (sim.calls["reduce_scatter"] > 0) == expect_sharded
    assert (sim.calls["all_reduce"] == 0) == expect_sharded
    assert rows == [num_tokens // tp if expect_sharded else num_tokens] * tp
    for rank in range(tp):
        out_r = gated[rank]
        ref_0 = reference[0]
        assert out_r is not None and ref_0 is not None
        torch.testing.assert_close(out_r, ref_0, rtol=1e-10, atol=1e-10)


def test_configure_layer_rejects_unsupported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gp, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(gp, "_get_pp_world_size", lambda: 1)
    monkeypatch.setattr(gp, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(gp, "get_dcp_world_size", lambda: 1)
    layer = _make_layer(_make_weights(0), 0, 0, 2)

    monkeypatch.setattr(gp, "get_tensor_model_parallel_world_size", lambda: 2)
    with pytest.raises(ValueError, match="Unsupported TPU_SHARDED_RESIDUAL"):
        gp.configure_layer(layer, NUM_LAYERS, "int8", 0)

    monkeypatch.setattr(gp, "get_tensor_model_parallel_world_size", lambda: 1)
    with pytest.raises(ValueError, match="TP > 1"):
        gp.configure_layer(layer, NUM_LAYERS, gp.MODE_BF16, 0)

    monkeypatch.setattr(gp, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(gp, "_get_pp_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="PP > 1"):
        gp.configure_layer(layer, NUM_LAYERS, gp.MODE_BF16, 0)

    monkeypatch.setattr(gp, "_get_pp_world_size", lambda: 1)
    monkeypatch.setattr(gp, "get_pcp_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="PCP/DCP > 1"):
        gp.configure_layer(layer, NUM_LAYERS, gp.MODE_BF16, 0)

    monkeypatch.setattr(gp, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(gp, "get_dcp_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="PCP/DCP > 1"):
        gp.configure_layer(layer, NUM_LAYERS, gp.MODE_BF16, 0)

    monkeypatch.setattr(gp, "get_dcp_world_size", lambda: 1)
    layer.enable_moe_block = True
    with pytest.raises(NotImplementedError, match="MoE"):
        gp.configure_layer(layer, NUM_LAYERS, gp.MODE_BF16, 0)

    layer.enable_moe_block = False
    layer.self_attn.is_kv_shared_layer = True
    with pytest.raises(NotImplementedError, match="KV-shared"):
        gp.configure_layer(layer, NUM_LAYERS, gp.MODE_BF16, 0)

    layer.self_attn.is_kv_shared_layer = False
    layer.mlp.down_proj.bias = torch.zeros(HIDDEN)
    with pytest.raises(NotImplementedError, match="bias"):
        gp.configure_layer(layer, NUM_LAYERS, gp.MODE_BF16, 0)


def test_maybe_patch_gemma4_lifecycle_and_idempotence(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TPU_SHARDED_RESIDUAL", raising=False)
    assert gp.maybe_patch_gemma4_sharded_residual(None) is None

    monkeypatch.setenv("TPU_SHARDED_RESIDUAL", "invalid")
    with pytest.raises(ValueError, match="TPU_SHARDED_RESIDUAL"):
        _ = envs.TPU_SHARDED_RESIDUAL

    monkeypatch.setenv("TPU_SHARDED_RESIDUAL", "FP8")
    assert envs.TPU_SHARDED_RESIDUAL == "FP8"
    assert gp.maybe_patch_gemma4_sharded_residual(None) is False

    non_gemma = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_text", num_kv_shared_layers=0)
    )
    assert gp.maybe_patch_gemma4_sharded_residual(non_gemma) is False

    kv_shared = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="gemma4_text", num_kv_shared_layers=4)
    )
    with pytest.raises(NotImplementedError, match="KV-shared"):
        gp.maybe_patch_gemma4_sharded_residual(kv_shared)

    init_calls: list[tuple] = []
    configure_calls: list[tuple] = []

    class _FakeGemma4DecoderLayer:
        def __init__(self, config):
            init_calls.append((self, config))

        def forward(self, *_args, **_kwargs):
            return None

    fake_mod = ModuleType("vllm.model_executor.models.gemma4")
    fake_mod.Gemma4DecoderLayer = _FakeGemma4DecoderLayer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.gemma4", fake_mod)
    monkeypatch.setattr(
        gp,
        "configure_layer",
        lambda layer, n_layers, mode, min_rows: configure_calls.append(
            (layer, n_layers, mode, min_rows)
        ),
    )

    gemma_cfg = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="gemma4_text", num_kv_shared_layers=0)
    )
    assert gp.maybe_patch_gemma4_sharded_residual(gemma_cfg) is None
    patched_init = _FakeGemma4DecoderLayer.__init__
    assert _FakeGemma4DecoderLayer.forward is gp.sharded_residual_forward

    # Calling the patch callback a second time must be a no-op (no double wrap).
    assert gp.maybe_patch_gemma4_sharded_residual(gemma_cfg) is None
    assert _FakeGemma4DecoderLayer.__init__ is patched_init

    layer_inst = _FakeGemma4DecoderLayer(SimpleNamespace(num_hidden_layers=62))
    assert len(init_calls) == 1
    assert configure_calls == [(layer_inst, 62, "fp8", gp._MIN_SHARDED_TOKENS)]


@pytest.mark.parametrize("mode", [gp.MODE_BF16, gp.MODE_FP8])
def test_min_rows_dynamo_graph_signatures_match(
    monkeypatch: pytest.MonkeyPatch, mode: str
):
    """Below-threshold, aligned, and unaligned buckets must lift identical FX
    placeholder signatures when input_ids and positions are marked dynamic."""
    from vllm_torchtpu.compilation import shape_variants

    tp = 4
    monkeypatch.setattr(gp, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(gp, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(gp, "_get_pp_world_size", lambda: 1)
    monkeypatch.setattr(gp, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(gp, "get_dcp_world_size", lambda: 1)
    monkeypatch.setattr(gp, "tensor_model_parallel_all_reduce", lambda x: x)
    monkeypatch.setattr(
        gp, "tensor_model_parallel_all_gather", lambda x, dim=0: x.repeat(tp, 1)
    )
    monkeypatch.setattr(
        gp,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: x.narrow(0, 0, x.shape[0] // tp),
    )
    pq.ensure_ops()
    monkeypatch.setattr(pq, "check_prequantized_linear", lambda _layer: None)

    w_dtype = torch.float8_e4m3fn if mode == gp.MODE_FP8 else torch.bfloat16

    class _Fp8OrFloatLinear(torch.nn.Module):
        def __init__(self, in_f: int, out_f: int):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros(in_f, out_f, dtype=w_dtype), requires_grad=False
            )
            self.register_buffer("weight_scale", torch.ones(out_f, dtype=torch.float32))
            self.bias = None
            self.reduce_results = True

        def forward(self, x: torch.Tensor):
            return ((x @ self.weight.to(torch.bfloat16)) * self.weight_scale).to(
                torch.bfloat16
            ), None

    class _ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(128, HIDDEN, dtype=torch.bfloat16)
            w = _make_weights(0)
            self.layers = []
            for idx in range(2):
                lyr = _make_layer(w, idx, 0, tp)
                lyr.self_attn.qkv_proj = _Fp8OrFloatLinear(
                    HIDDEN, 3 * (HEADS // tp) * HEAD_DIM
                )
                lyr.self_attn.o_proj = _Fp8OrFloatLinear(
                    (HEADS // tp) * HEAD_DIM, HIDDEN
                )
                lyr.mlp.gate_up_proj = _Fp8OrFloatLinear(
                    HIDDEN, 2 * (INTERMEDIATE // tp)
                )
                lyr.mlp.down_proj = _Fp8OrFloatLinear(INTERMEDIATE // tp, HIDDEN)
                lyr.input_layernorm = _rms_norm(
                    torch.nn.Parameter(
                        w["norms"][0].to(torch.bfloat16), requires_grad=False
                    )
                )
                lyr.post_attention_layernorm = _rms_norm(
                    torch.nn.Parameter(
                        w["norms"][1].to(torch.bfloat16), requires_grad=False
                    )
                )
                lyr.pre_feedforward_layernorm = _rms_norm(
                    torch.nn.Parameter(
                        w["norms"][2].to(torch.bfloat16), requires_grad=False
                    )
                )
                lyr.post_feedforward_layernorm = _rms_norm(
                    torch.nn.Parameter(
                        w["norms"][3].to(torch.bfloat16), requires_grad=False
                    )
                )
                lyr.layer_scalar = torch.nn.Parameter(
                    torch.tensor([0.75], dtype=torch.bfloat16), requires_grad=False
                )
                gp.configure_layer(lyr, 2, mode, min_rows=32)
                setattr(self, f"qkv_{idx}", lyr.self_attn.qkv_proj)
                setattr(self, f"o_{idx}", lyr.self_attn.o_proj)
                setattr(self, f"gu_{idx}", lyr.mlp.gate_up_proj)
                setattr(self, f"dn_{idx}", lyr.mlp.down_proj)
                self.layers.append(lyr)

        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
            h = self.embed(input_ids)
            for lyr in self.layers:
                h, _ = gp.sharded_residual_forward(lyr, positions, h, None)
            return h + positions.unsqueeze(-1).to(h.dtype) * 0.0

    sigs = []

    def capture_backend(gm, _example_inputs):
        sigs.append(shape_variants.graph_signature(gm))
        return lambda *_args: torch.zeros(1, HIDDEN, dtype=torch.bfloat16)

    model = _ToyModel()
    with shape_variants.warmup():
        for n in (16, 64, 66):
            torch._dynamo.reset()
            compiled = torch.compile(model, backend=capture_backend, fullgraph=True)
            ids = torch.zeros(n, dtype=torch.int32)
            pos = torch.arange(n, dtype=torch.int32)
            torch._dynamo.mark_dynamic(ids, 0)
            torch._dynamo.mark_dynamic(pos, 0)
            compiled(ids, pos)

    assert len(sigs) == 3
    assert sigs[0] == sigs[1] == sigs[2]
