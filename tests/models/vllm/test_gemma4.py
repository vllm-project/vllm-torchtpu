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
"""Unit, Pallas interpret, and TPU tests for Gemma 4 routing, layers, and CP prefill."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.models import gemma4 as vllm_gemma4
from vllm.model_executor.models.gemma4 import Gemma4Attention, Gemma4MLP

from vllm_torchtpu.kernels.experimental.batched_rpa import configs
from vllm_torchtpu.layers.adapter.attention import PallasBatchedRPAAttentionBackendImpl
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.models.vllm import gemma4 as gemma4_patch
from vllm_torchtpu.models.vllm.gemma4 import (
    cp_prefill,
)
from vllm_torchtpu.models.vllm.gemma4 import (
    layers as gemma4_layers,
)
from vllm_torchtpu.models.vllm.gemma4 import (
    routing as gemma4_routing,
)

FP8 = jnp.float8_e4m3fn
PAGE = 128
INTERPRET = pltpu.InterpretParams(detect_races=True)
LAYOUTS = (configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE)
_UPSTREAM_ROUTING = vllm_gemma4.gemma4_routing_function_torch


def _bytes(x: Any) -> np.ndarray:
    return np.asarray(x).view(np.uint8)


def _kv(
    num_tokens: int, kv_heads: int, head_dim: int, seed: int
) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((num_tokens, kv_heads, head_dim), np.float32)
    v = rng.standard_normal((num_tokens, kv_heads, head_dim), np.float32)
    return jnp.asarray(k, FP8), jnp.asarray(v, FP8)


def _rank_rows(x: jax.Array, plan: cp_prefill.ZigzagPlan, rank: int) -> jax.Array:
    return jnp.concatenate([x[s:e] for s, e in plan.rank_slices(rank)])


def _reference_write_pages(
    init_cache: jax.Array,
    src_pages_data: jax.Array,
    src_idx: jax.Array,
    dst_idx: jax.Array,
    count: int,
) -> np.ndarray:
    out = np.array(_bytes(init_cache), copy=True)
    src_bytes = _bytes(src_pages_data)
    for i in range(count):
        out[int(dst_idx[i])] = src_bytes[int(src_idx[i])]
    return out


def test_zigzag_plan_balances_causal_pairs_and_inverts_page_tables():
    plan = cp_prefill.make_zigzag_plan(2 * 4 * 2 * PAGE, world=4, page_size=PAGE)
    assert plan.piece_tokens == 2 * PAGE
    covered = sorted(s for rank in range(plan.world) for s in plan.rank_slices(rank))
    assert covered == [(i * 2 * PAGE, (i + 1) * 2 * PAGE) for i in range(8)]

    causal_pairs = [
        sum((s + e + 1) * (e - s) // 2 for s, e in plan.rank_slices(rank))
        for rank in range(plan.world)
    ]
    assert max(causal_pairs) == min(causal_pairs)

    table = plan.gathered_page_table()
    inv = plan.gathered_position_pages()
    np.testing.assert_array_equal(table[inv], np.arange(plan.num_pages, dtype=np.int32))
    np.testing.assert_array_equal(inv[table], np.arange(plan.num_pages, dtype=np.int32))

    k, v = _kv(plan.num_tokens, 4, 128, seed=0)
    gathered = jnp.concatenate(
        [
            cp_prefill.kv_pages_from_rows(
                cp_prefill.pack_kv_rows(_rank_rows(k, plan, r), _rank_rows(v, plan, r)),
                128,
                PAGE,
            )
            for r in range(plan.world)
        ]
    )
    in_order = cp_prefill.kv_pages_from_rows(cp_prefill.pack_kv_rows(k, v), 128, PAGE)
    np.testing.assert_array_equal(_bytes(gathered[table]), _bytes(in_order))


def test_zigzag_plan_rejects_unaligned_tokens_and_invalid_halos():
    with pytest.raises(ValueError, match="multiple of"):
        cp_prefill.make_zigzag_plan(2 * 4 * PAGE + 64, world=4, page_size=PAGE)
    plan = cp_prefill.make_zigzag_plan(2 * 4 * 2 * PAGE, world=4, page_size=PAGE)
    with pytest.raises(ValueError, match="Halo"):
        plan.sliding_page_tables(0, halo_pages=0)
    with pytest.raises(ValueError, match="Halo"):
        plan.sliding_page_tables(0, halo_pages=plan.piece_pages + 1)


def test_sliding_tables_neighbours_and_tp_kv_heads():
    plan = cp_prefill.make_zigzag_plan(2 * 4 * 3 * PAGE, world=4, page_size=PAGE)
    for rank in range(plan.world):
        first_src, second_src = cp_prefill.halo_neighbours(rank, plan.world)
        p0, p1 = plan.rank_pieces(rank)
        if rank > 0:
            assert plan.piece_owner(p0 - 1) == (first_src, 0)
        if rank < plan.world - 1:
            assert plan.piece_owner(p1 - 1) == (second_src, 1)
        tables = np.asarray(plan.sliding_page_tables(rank, halo_pages=2))
        assert tables.shape == (2, plan.num_pages)

    assert [cp_prefill.tp_kv_heads(8, 4, r) for r in range(4)] == [
        (0, 2),
        (2, 2),
        (4, 2),
        (6, 2),
    ]
    assert [cp_prefill.tp_kv_heads(2, 8, r) for r in range(8)] == [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ]


def test_window_page_helpers_assemble_decode_window_slots():
    plan = cp_prefill.make_zigzag_plan(2 * 4 * 4 * PAGE, world=4, page_size=PAGE)
    seq_len = 9 * PAGE + 37
    sliding_window = 512
    k, v = _kv(plan.num_tokens, 4, 128, seed=3)
    pos_pages, count = cp_prefill.window_position_pages(
        jnp.int32(seq_len), sliding_window, PAGE
    )
    owners = cp_prefill.window_page_owners(plan, pos_pages)

    window_rows = []
    for r in range(plan.world):
        rk, rv = _rank_rows(k, plan, r), _rank_rows(v, plan, r)
        rows = cp_prefill.pack_kv_rows(rk, rv).reshape(2 * plan.piece_pages, PAGE, -1)
        local_idx = cp_prefill._window_local_pages(plan, r, pos_pages)
        window_rows.append(rows[local_idx])
    gathered_window = jnp.stack(window_rows, axis=0)

    for r in range(plan.world):
        first, num_heads = cp_prefill.tp_kv_heads(4, plan.world, r)
        slots = cp_prefill.window_slots_from_rows(
            jax.lax.bitcast_convert_type(gathered_window, jnp.int8),
            owners,
            first,
            num_heads,
            128,
        )
        slots_fp8 = jax.lax.bitcast_convert_type(slots, FP8)
        valid = np.asarray(pos_pages)[: int(count)]
        want = jnp.stack(
            [
                k.reshape(plan.num_pages, PAGE, 4, 128)[
                    valid, :, first : first + num_heads
                ],
                v.reshape(plan.num_pages, PAGE, 4, 128)[
                    valid, :, first : first + num_heads
                ],
            ],
            axis=3,
        ).reshape(len(valid), PAGE, 2 * num_heads, 128)
        np.testing.assert_array_equal(_bytes(slots_fp8[: int(count)]), _bytes(want))


@pytest.mark.parametrize(
    ("num_kv_heads", "layout"),
    [
        (4, configs.KVLayout.HEAD_ALONG_SUBLANE),
        (8, configs.KVLayout.HEAD_ALONG_SUBLANE),
        (4, configs.KVLayout.SEQ_ALONG_LANE),
    ],
)
def test_write_kv_head_pages_interpret_matches_reference(num_kv_heads, layout):
    plan = cp_prefill.make_zigzag_plan(2 * 4 * 2 * PAGE, world=4, page_size=PAGE)
    num_cache_pages = 24
    seq_len = plan.num_tokens - 2 * PAGE - 19  # partial write: 14 of 16 pages
    with jax.default_device(jax.devices("cpu")[0]):
        k, v = _kv(plan.num_tokens, num_kv_heads, 128, seed=11)
        gathered_k = jnp.concatenate(
            [_rank_rows(k, plan, r) for r in range(plan.world)]
        )
        gathered_v = jnp.concatenate(
            [_rank_rows(v, plan, r) for r in range(plan.world)]
        )
        gathered_pages = cp_prefill.kv_pages_from_rows(
            cp_prefill.pack_kv_rows(gathered_k, gathered_v), 128, PAGE
        )

        rng = np.random.default_rng(17)
        block_table = jnp.asarray(
            rng.choice(num_cache_pages, size=plan.num_pages, replace=False), jnp.int32
        )
        src, dst, count = cp_prefill.full_attention_page_writes(
            plan, block_table, jnp.int32(seq_len)
        )
        assert int(count) == 14 and dst.shape[0] == 16

        gk_pages = gathered_k.reshape(plan.num_pages, PAGE, num_kv_heads, 128)
        gv_pages = gathered_v.reshape(plan.num_pages, PAGE, num_kv_heads, 128)
        for rank in range(plan.world):
            first, num_heads = cp_prefill.tp_kv_heads(num_kv_heads, plan.world, rank)
            slots = jnp.stack(
                [
                    gk_pages[:, :, first : first + num_heads],
                    gv_pages[:, :, first : first + num_heads],
                ],
                axis=3,
            ).reshape(plan.num_pages, PAGE, 2 * num_heads, 128)
            expected_pages = cp_prefill.cache_pages_from_slots(slots, layout)
            init_cache = jnp.full(
                (num_cache_pages, *expected_pages.shape[1:]), 7.0, FP8
            )

            want_bytes = _reference_write_pages(
                init_cache, expected_pages, src, dst, int(count)
            )
            got_full = cp_prefill.write_kv_pages(
                init_cache, expected_pages, src, dst, count, interpret=INTERPRET
            )
            got_head = cp_prefill.write_kv_head_pages(
                init_cache,
                gathered_pages,
                jnp.int32(first),
                src,
                dst,
                count,
                num_heads=num_heads,
                kv_layout=layout,
                interpret=INTERPRET,
            )
            np.testing.assert_array_equal(_bytes(got_full), want_bytes)
            np.testing.assert_array_equal(_bytes(got_head), want_bytes)


def test_write_kv_head_pages_lowers_for_tpu7x():
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    pages = jax.ShapeDtypeStruct((8, PAGE, 2, 4, 128), FP8)
    idx = jax.ShapeDtypeStruct((8,), jnp.int32)
    scalar = jax.ShapeDtypeStruct((), jnp.int32)
    for layout, cache_tail in (
        (configs.KVLayout.HEAD_ALONG_SUBLANE, (PAGE, 1, 4, 128)),
        (configs.KVLayout.SEQ_ALONG_LANE, (2, 32, 4, PAGE)),
    ):
        cache = jax.ShapeDtypeStruct((16, *cache_tail), FP8)
        with jax.sharding.use_abstract_mesh(mesh):
            hlo = (
                jax.jit(
                    lambda c, p, h, s, d, n, ly=layout: cp_prefill.write_kv_head_pages(
                        c, p, h, s, d, n, num_heads=1, kv_layout=ly
                    )
                )
                .trace(cache, pages, scalar, idx, idx, scalar)
                .lower(lowering_platforms=("tpu",))
                .as_text()
            )
        assert "tpu_custom_call" in hlo or "custom_call" in hlo


@pytest.mark.tpu_test
@pytest.mark.parametrize("layout", LAYOUTS)
def test_write_kv_head_pages_on_tpu(layout):
    plan = cp_prefill.make_zigzag_plan(2 * 4 * PAGE, world=4, page_size=PAGE)
    k, v = _kv(plan.num_tokens, 4, 128, seed=23)
    pages = cp_prefill.kv_pages_from_rows(cp_prefill.pack_kv_rows(k, v), 128, PAGE)
    src = jnp.arange(plan.num_pages, dtype=jnp.int32)
    dst = jnp.arange(plan.num_pages, dtype=jnp.int32)[::-1]
    count = jnp.int32(plan.num_pages - 1)

    slots = jnp.stack([k[:, 1:2], v[:, 1:2]], axis=2).reshape(
        plan.num_pages, PAGE, 2, 128
    )
    expected_pages = cp_prefill.cache_pages_from_slots(slots, layout)
    init_cache = jnp.full((plan.num_pages + 2, *expected_pages.shape[1:]), 5.0, FP8)
    got = cp_prefill.write_kv_head_pages(
        init_cache,
        pages,
        jnp.int32(1),
        src,
        dst,
        count,
        num_heads=1,
        kv_layout=layout,
        interpret=False,
    )
    want_bytes = _reference_write_pages(
        init_cache, expected_pages, src, dst, int(count)
    )
    np.testing.assert_array_equal(_bytes(got), want_bytes)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 16, 1000])
def test_gemma4_topk_softmax_routing_matches_upstream(dtype, num_tokens):
    gen = torch.Generator().manual_seed(num_tokens)
    logits = (torch.randn(num_tokens, 128, generator=gen) * 3.0).to(dtype)
    scale = (torch.rand(128, generator=gen) + 0.5).to(torch.bfloat16)

    ref_w, ref_ids = _UPSTREAM_ROUTING(logits, 8, scale)
    got_w, got_ids = gemma4_routing.gemma4_topk_softmax_routing(logits, 8, scale)

    assert got_ids.dtype == ref_ids.dtype == torch.int32
    assert got_w.dtype == ref_w.dtype == torch.float32
    torch.testing.assert_close(got_ids, ref_ids, rtol=0, atol=0)
    tol = 1e-6 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(got_w, ref_w, rtol=tol, atol=tol)


class _StubFp8Method(VllmFp8LinearMethodTPU):
    def __init__(self, max_w8a16_rows: int = 16) -> None:
        self.max_w8a16_rows = max_w8a16_rows
        self.quantized_calls: list[int] = []

    def supports_quantized_input(self, layer: Any, rows: int) -> bool:
        del layer
        return rows > self.max_w8a16_rows

    def apply_quantized_input(
        self,
        layer: Any,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, bias
        self.quantized_calls.append(x_q.shape[0])
        return x_q.to(torch.bfloat16) * x_scale.to(torch.bfloat16)


class _StubRpaImpl(PallasBatchedRPAAttentionBackendImpl):
    def __init__(self, prologue_eps: float | None = 1e-6) -> None:
        self.qkv_prologue_eps = prologue_eps
        self.qkv_v_rms_norm = True
        self.qkv_output_quant_dtype: torch.dtype | None = torch.float8_e4m3fn
        self.seen_quantize_output: list[bool] = []

    def forward_from_qkv(
        self,
        layer: Any,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        eps: float,
        kv_cache: Any,
        attn_metadata: Any,
        *,
        quantize_output: bool = False,
    ) -> Any:
        del layer, cos, sin, q_norm_weight, k_norm_weight, eps, kv_cache, attn_metadata
        self.seen_quantize_output.append(quantize_output)
        rows = qkv.shape[0]
        if quantize_output:
            return (
                torch.ones((rows, 8), dtype=torch.float8_e4m3fn),
                torch.ones((rows, 1), dtype=torch.float32),
            )
        return torch.ones((rows, 8), dtype=torch.bfloat16)


class _StubCallableLinear:
    def __init__(self, calls: list[int], **attrs: Any) -> None:
        self._calls = calls
        for k, v in attrs.items():
            setattr(self, k, v)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self._calls.append(x.shape[0])
        return x, None


def test_mlp_and_attention_layers_respect_quantized_row_thresholds(monkeypatch):
    method = _StubFp8Method(max_w8a16_rows=16)
    fallback_calls: list[int] = []
    down = _StubCallableLinear(
        fallback_calls,
        quant_method=method,
        bias=None,
        input_size_per_partition=128,
        weight=torch.zeros((8, 128), dtype=torch.float8_e4m3fn),
        reduce_results=False,
        tp_size=1,
    )
    with set_current_vllm_config(VllmConfig()):
        act_fn = GeluAndMul(approximate="tanh")
    mlp = SimpleNamespace(act_fn=act_fn, down_proj=down)
    monkeypatch.setattr(
        gemma4_layers,
        "gelu_and_mul_quantize_op",
        lambda gu: (
            torch.ones((gu.shape[0], 128), dtype=torch.float8_e4m3fn),
            torch.ones((gu.shape[0], 1), dtype=torch.float32),
        ),
    )

    gemma4_layers.down_proj_from_gate_up(
        mlp, torch.ones((8, 256), dtype=torch.bfloat16)
    )
    gemma4_layers.down_proj_from_gate_up(
        mlp, torch.ones((32, 256), dtype=torch.bfloat16)
    )
    assert fallback_calls == [8]
    assert method.quantized_calls == [32]

    impl = _StubRpaImpl(prologue_eps=1e-6)
    o_method = _StubFp8Method(max_w8a16_rows=16)
    o_fallback: list[int] = []
    o_proj = _StubCallableLinear(
        o_fallback,
        quant_method=o_method,
        reduce_results=False,
        tp_size=1,
    )
    rope = SimpleNamespace(
        _match_cos_sin_cache_dtype=lambda qkv: torch.ones((64, 8), dtype=qkv.dtype)
    )
    attn = SimpleNamespace(
        attn=SimpleNamespace(impl=impl, layer_name="layers.0.self_attn"),
        rotary_emb=rope,
        o_proj=o_proj,
        q_norm=SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.ones(4)),
    )
    monkeypatch.setattr(
        gemma4_layers,
        "get_attention_context",
        lambda _name: ("md", "layer", "kv", None),
    )

    out_decode = gemma4_layers.attention_from_qkv(
        attn, torch.arange(8), torch.ones((8, 16), dtype=torch.bfloat16)
    )
    gemma4_layers.o_projection(attn, out_decode)

    out_prefill = gemma4_layers.attention_from_qkv(
        attn, torch.arange(32), torch.ones((32, 16), dtype=torch.bfloat16)
    )
    gemma4_layers.o_projection(attn, out_prefill)

    assert impl.seen_quantize_output == [False, True]
    assert o_fallback == [8]
    assert o_method.quantized_calls == [32]


def test_maybe_patch_gemma4_filters_model_type_and_is_idempotent(monkeypatch):
    fake_mod = SimpleNamespace(gemma4_routing_function_torch=_UPSTREAM_ROUTING)
    assert gemma4_routing.apply_gemma4_routing_patch(fake_mod) is True
    first_fn = fake_mod.gemma4_routing_function_torch
    assert gemma4_routing.apply_gemma4_routing_patch(fake_mod) is False
    assert fake_mod.gemma4_routing_function_torch is first_fn

    assert gemma4_patch.maybe_patch_gemma4(None) is False
    llama_cfg = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="llama"))
    assert gemma4_patch.maybe_patch_gemma4(llama_cfg) is False

    gemma_cfg = SimpleNamespace(
        architecture="Gemma4ForCausalLM",
        hf_text_config=SimpleNamespace(model_type="gemma4_text"),
    )
    monkeypatch.setattr(gemma4_patch, "gemma4_mod", fake_mod)
    # Keep the sequence-parallel patch out of the shared vLLM classes.
    monkeypatch.setenv("TPU_GEMMA4_SEQUENCE_PARALLEL", "0")
    assert gemma4_patch.maybe_patch_gemma4(gemma_cfg) is None
    mlp_fwd = Gemma4MLP.forward
    attn_fwd = Gemma4Attention.forward
    assert gemma4_patch.maybe_patch_gemma4(gemma_cfg) is None
    assert Gemma4MLP.forward is mlp_fwd
    assert Gemma4Attention.forward is attn_fwd
