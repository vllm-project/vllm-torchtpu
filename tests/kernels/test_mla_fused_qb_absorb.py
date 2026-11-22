# SPDX-License-Identifier: Apache-2.0
"""Tests for fused VMEM q_b_proj + NeoX q_pe RoPE + absorb_q_split."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.mla.absorb_q import absorb_q_split
from vllm_torchtpu.kernels.mla.fused_qb_absorb import (
    fused_qb_rope_absorb_split,
    fused_qb_rope_absorb_split_ref,
    select_tail_chunk,
    supports_fused_qb_rope_absorb_split,
)
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)
from vllm_torchtpu.layers.adapter import attention as attn_mod
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op_mod
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op_mod
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_WEIGHT,
    PREQUANT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)


def _require_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel requires TPU backend")


def _make_jax_inputs(
    total_tokens: int,
    *,
    heads: int = 16,
    k_dim: int = 256,
    d_nope: int = 192,
    d_pe: int = 64,
    rank: int = 256,
    seed: int = 17,
):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q_c = jax.random.normal(k1, (total_tokens, k_dim), dtype=jnp.bfloat16) * 0.4
    w_qb_f = jax.random.normal(k2, (k_dim, heads * (d_nope + d_pe)), dtype=jnp.float32)
    w_qb, w_s = quantize_tensor(w_qb_f * 0.05, jnp.float8_e4m3fn, dim=0)
    w_s = w_s.reshape(-1)
    w_uk_t = jax.random.normal(k3, (heads, d_nope, rank), dtype=jnp.bfloat16) * 0.05
    # Asymmetric non-degenerate angles across rows and dims.
    angles = jax.random.uniform(k4, (total_tokens, d_pe // 2), dtype=jnp.float32) * 6.0
    cos_sin = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    return q_c, w_qb, w_s, w_uk_t, cos_sin


def test_should_select_largest_supported_tail_chunk():
    assert select_tail_chunk(2048) == 512
    assert select_tail_chunk(1024) == 1024
    assert select_tail_chunk(512) == 512
    assert select_tail_chunk(256) == 256
    assert select_tail_chunk(128) == 128
    assert select_tail_chunk(64) == 64
    assert select_tail_chunk(192) == 64
    assert select_tail_chunk(32) is None
    assert select_tail_chunk(0) is None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"q_c_shape": (64, 2048)}, False),
        ({"q_c_dtype": torch.float16}, False),
        ({"w_qb_dtype": torch.bfloat16}, False),
        ({"w_s_shape": (1, 16, 1, 16384)}, False),
        (
            {
                "w_uk_t_shape": (60, 192, 512),
                "w_qb_shape": (2048, 60 * 256),
                "w_s_shape": (60 * 256,),
            },
            False,
        ),
        ({"cos_sin_shape": (1088, 32)}, False),
        ({"split_rows": 60}, False),
        ({"split_rows": 0}, False),
        ({"q_c_shape": (1088, 192), "w_qb_shape": (192, 16384)}, False),
        (
            {"q_c_shape": (100, 2048), "cos_sin_shape": (100, 64), "split_rows": 64},
            False,
        ),
    ],
)
def test_should_validate_fused_qb_contract_one_factor_at_a_time(overrides, expected):
    base = {
        "q_c_shape": (1088, 2048),
        "q_c_dtype": torch.bfloat16,
        "w_qb_shape": (2048, 16384),
        "w_qb_dtype": torch.float8_e4m3fn,
        "w_s_shape": (16384,),
        "w_uk_t_shape": (64, 192, 512),
        "cos_sin_shape": (1088, 64),
        "split_rows": 64,
        "bfloat16_dtype": torch.bfloat16,
        "fp8_dtype": torch.float8_e4m3fn,
    }
    base.update(overrides)
    assert supports_fused_qb_rope_absorb_split(**base) is expected


def test_should_match_independent_unfused_oracle_in_reference_chain():
    q_c, w_qb, w_s, w_uk_t, cos_sin = _make_jax_inputs(192, heads=8, k_dim=256)
    split_rows = 64
    got_h, got_t, got_pe = fused_qb_rope_absorb_split_ref(
        q_c, w_qb, w_s, w_uk_t, cos_sin, split_rows
    )

    q_full = xla_quantized_matmul(q_c, w_qb, w_s).reshape(192, 8, 256)
    q_nope, q_pe_raw = q_full[:, :, :192], q_full[:, :, 192:]
    exp_ql = jnp.einsum(
        "thd,hdr->thr",
        q_nope.astype(jnp.float32),
        w_uk_t.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    c = cos_sin[:, None, :32]
    s = cos_sin[:, None, 32:]
    q_pe_f = q_pe_raw.astype(jnp.float32)
    exp_pe = jnp.concatenate(
        [
            q_pe_f[..., :32] * c - q_pe_f[..., 32:] * s,
            q_pe_f[..., 32:] * c + q_pe_f[..., :32] * s,
        ],
        axis=-1,
    ).astype(jnp.bfloat16)

    np.testing.assert_array_equal(
        np.asarray(got_h).view(np.uint16),
        np.asarray(exp_ql[:split_rows]).view(np.uint16),
    )
    np.testing.assert_array_equal(
        np.asarray(got_t).view(np.uint16),
        np.asarray(exp_ql[split_rows:]).view(np.uint16),
    )
    np.testing.assert_array_equal(
        np.asarray(got_pe).view(np.uint16),
        np.asarray(exp_pe).view(np.uint16),
    )


@pytest.mark.parametrize(
    ("total_tokens", "split_rows", "heads", "k_dim", "rank"),
    [
        (192, 64, 16, 256, 256),
        (576, 64, 16, 512, 512),
        (1088, 64, 64, 2048, 512),
    ],
)
def test_should_match_ckpt4_unfused_chain_bit_exact_on_tpu(
    total_tokens: int,
    split_rows: int,
    heads: int,
    k_dim: int,
    rank: int,
):
    _require_tpu()
    q_c, w_qb, w_s, w_uk_t, cos_sin = _make_jax_inputs(
        total_tokens, heads=heads, k_dim=k_dim, rank=rank, seed=total_tokens + heads
    )
    got_h, got_t, got_pe = jax.jit(fused_qb_rope_absorb_split, static_argnums=(5,))(
        q_c, w_qb, w_s, w_uk_t, cos_sin, split_rows
    )

    q_full = jax.jit(xla_quantized_matmul)(q_c, w_qb, w_s).reshape(
        total_tokens, heads, 256
    )
    exp_h, exp_t = jax.jit(absorb_q_split, static_argnums=(2,))(
        q_full[:, :, :192], w_uk_t, split_rows
    )
    _, _, exp_pe = jax.jit(fused_qb_rope_absorb_split_ref, static_argnums=(5,))(
        q_c, w_qb, w_s, w_uk_t, cos_sin, split_rows
    )

    np.testing.assert_array_equal(
        np.asarray(got_h).view(np.uint16), np.asarray(exp_h).view(np.uint16)
    )
    np.testing.assert_array_equal(
        np.asarray(got_t).view(np.uint16), np.asarray(exp_t).view(np.uint16)
    )
    np.testing.assert_array_equal(
        np.asarray(got_pe).view(np.uint16), np.asarray(exp_pe).view(np.uint16)
    )


def test_should_reject_invalid_shapes_in_fused_qb_kernel():
    q_c, w_qb, w_s, w_uk_t, cos_sin = _make_jax_inputs(100, heads=8, k_dim=256)
    with pytest.raises(ValueError, match="Unsupported inputs"):
        fused_qb_rope_absorb_split(q_c, w_qb, w_s, w_uk_t, cos_sin, 64)


def test_should_guard_fused_qb_operands_against_padded_and_prequant_layers():
    class _Linear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros(128, 256, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            self.weight_scale = torch.nn.Parameter(
                torch.ones(256, dtype=torch.float32), requires_grad=False
            )
            self.bias = None
            self.gather_output = False
            setattr(self, WEIGHT_FLIPPED_ATTR, True)

    lin = _Linear()
    assert mla_op_mod._fused_qb_operands(lin) == (lin.weight, lin.weight_scale)  # noqa: SLF001 -- exercises operand eligibility guard directly

    setattr(lin, FP8_PALLAS_WEIGHT, torch.zeros(128, 256, dtype=torch.float8_e4m3fn))
    assert mla_op_mod._fused_qb_operands(lin) is None  # noqa: SLF001 -- verifies padded copy disables fused op
    delattr(lin, FP8_PALLAS_WEIGHT)

    setattr(lin, PREQUANT_ATTR, SimpleNamespace(value=None))
    assert mla_op_mod._fused_qb_operands(lin) is None  # noqa: SLF001 -- verifies prequant slot disables fused op


def test_should_route_preabsorbed_split_q_through_attention_backend_without_reabsorb():
    backend = object.__new__(attn_mod.PallasMLAttentionBackendImpl)
    backend.dcp_size = 4

    heads, kv_rank, rope_dim = 8, 128, 64
    split_rows, tail_rows = 8, 16
    num_tokens = 32
    ql_head = torch.randn(split_rows, heads, kv_rank, dtype=torch.bfloat16)
    ql_tail = torch.randn(tail_rows, heads, kv_rank, dtype=torch.bfloat16)
    q_pe = torch.randn(split_rows + tail_rows, heads, rope_dim, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, kv_rank, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, 1, rope_dim, dtype=torch.bfloat16)
    topk = torch.zeros(num_tokens, 4, dtype=torch.int32)
    kv_cache = (torch.zeros(4, 16, kv_rank), torch.zeros(4, 16, rope_dim))

    tp_op = SimpleNamespace(
        layout=SimpleNamespace(split_rows=split_rows),
        split_q_inputs=True,
    )
    layer = SimpleNamespace(
        num_heads=heads // 4,
        kv_lora_rank=kv_rank,
        qk_rope_head_dim=rope_dim,
        v_head_dim=64,
        W_UK_T_dcp_qrep=torch.zeros(heads, 192, kv_rank, dtype=torch.bfloat16),
        W_UK_T_dcp_qrep_scale=None,
        W_UV_dcp_qrep=None,
        kv_cache_quantized_dtype=None,
        sparse_mla_tp_op=tp_op,
        _q_scale_float=1.0,
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )
    captured: dict[str, object] = {}

    def _fake_tp_attention(
        _layer, _kv_cache, ql_arg, qpe_arg, *_rest, **_kw
    ) -> torch.Tensor:
        captured["ql_arg"] = ql_arg
        captured["qpe_arg"] = qpe_arg
        return torch.zeros(
            num_tokens, layer.num_heads, layer.v_head_dim, dtype=torch.bfloat16
        )

    with (
        patch.object(tp_op_mod, "applies", return_value=True),
        patch.object(tp_op_mod, "attention", side_effect=_fake_tp_attention),
        patch.object(
            attn_mod, "_absorb_q_split", side_effect=AssertionError("must skip absorb")
        ),
        patch.object(
            attn_mod, "_absorb_q", side_effect=AssertionError("must skip absorb")
        ),
        patch.object(backend, "_get_kv_scales", return_value=(None, None, None)),
    ):
        out = backend.forward(
            layer=layer,
            q=(ql_head[:, :2, :64], q_pe[:, :2, :]),
            kv_c_normed=kv_c,
            k_pe=k_pe,
            kv_cache=kv_cache,
            attn_metadata=SimpleNamespace(),
            topk_indices=topk,
            q_dcp_replicated=((ql_head, ql_tail), q_pe),
        )

    assert tuple(out.shape) == (num_tokens, layer.num_heads * layer.v_head_dim)
    assert isinstance(captured["ql_arg"], tuple)
    torch.testing.assert_close(captured["ql_arg"][0], ql_head, rtol=0, atol=0)
    torch.testing.assert_close(captured["ql_arg"][1], ql_tail, rtol=0, atol=0)
    torch.testing.assert_close(captured["qpe_arg"], q_pe, rtol=0, atol=0)


def test_should_match_dynamo_fx_placeholders_for_decode_and_fused_prefill():
    heads, k_dim, d_nope, d_pe, rank = 8, 128, 192, 64, 128
    split_rows = 8

    class _FakeQProj(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(k_dim, heads * (d_nope + d_pe), dtype=torch.bfloat16).to(
                    torch.float8_e4m3fn
                ),
                requires_grad=False,
            )
            self.weight_scale = torch.nn.Parameter(
                torch.ones(heads * (d_nope + d_pe), dtype=torch.float32),
                requires_grad=False,
            )
            self.bias = None
            self.gather_output = False
            setattr(self, WEIGHT_FLIPPED_ATTR, True)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            y = x.to(torch.float32) @ self.weight.to(torch.float32)
            return (y * self.weight_scale).to(x.dtype), None

    class _FakeRotary(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "cos_sin_cache", torch.ones(256, d_pe, dtype=torch.float32)
            )
            self.rotary_dim = d_pe
            self.head_size = d_pe

        def _match_cos_sin_cache_dtype(self, x: torch.Tensor) -> torch.Tensor:
            return self.cos_sin_cache.to(x.dtype)

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_b_proj = _FakeQProj()
            self.rotary_emb = _FakeRotary()
            self.register_buffer(
                "W_UK_T_dcp_qrep",
                torch.ones(heads, d_nope, rank, dtype=torch.bfloat16),
            )

        def forward(self, q_c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            if q_c.shape[0] >= 128:
                # Fused prefill path reads exact same 4 tensors via module attributes.
                w_qb, s_qb = mla_op_mod._fused_qb_operands(self.q_b_proj)  # noqa: SLF001 -- exercises real operand getter under Dynamo
                cs = self.rotary_emb.cos_sin_cache.index_select(0, positions)
                q_full = (
                    (q_c.to(torch.float32) @ w_qb.to(torch.float32) * s_qb)
                    .to(q_c.dtype)
                    .view(-1, heads, d_nope + d_pe)
                )
                ql = torch.einsum(
                    "thd,hdr->thr", q_full[:, :, :d_nope], self.W_UK_T_dcp_qrep
                )
                return (
                    ql[:split_rows].sum()
                    + ql[split_rows:].sum()
                    + cs.sum().to(q_c.dtype)
                )
            # Decode path reads the exact same 4 tensors.
            q_full = self.q_b_proj(q_c)[0].view(-1, heads, d_nope + d_pe)
            cs = self.rotary_emb.cos_sin_cache.index_select(0, positions)
            ql = torch.einsum(
                "thd,hdr->thr", q_full[:, :, :d_nope], self.W_UK_T_dcp_qrep
            )
            return ql.sum() + cs.sum().to(q_c.dtype)

    model = _FakeModel()
    seen: dict[str, set[str]] = {}
    for label, n in (("decode", 4), ("prefill", 192)):
        captured: list[set[str]] = []

        def _backend(gm, _ex, _cap=captured):
            _cap.append({nd.name for nd in gm.graph.nodes if nd.op == "placeholder"})
            return gm.forward

        torch._dynamo.reset()  # noqa: SLF001 -- isolates per-bucket Dynamo compile trace
        compiled = torch.compile(model, backend=_backend, fullgraph=True)
        compiled(
            torch.randn(n, k_dim, dtype=torch.bfloat16),
            torch.arange(n, dtype=torch.int32),
        )
        assert len(captured) == 1
        seen[label] = captured[0]

    assert seen["prefill"] == seen["decode"]
    assert envs.TPU_MLA_FUSED_QB_ROPE_ABSORB is True


def test_should_match_wrapper_forward_between_fused_and_unfused_prefill():
    torch.manual_seed(11)
    dcp, total_tokens, split_rows, tail_rows = 4, 256, 64, 64
    q_total = split_rows + tail_rows
    heads_all, heads_local = 8, 2
    q_lora, kv_rank, d_nope, d_pe, v_dim = 128, 128, 192, 64, 64
    d_q = d_nope + d_pe

    q_c_all = torch.randn(total_tokens, q_lora, dtype=torch.bfloat16) * 0.4
    kv_lora_all = torch.randn(total_tokens, kv_rank + d_pe, dtype=torch.bfloat16) * 0.3
    positions = torch.arange(total_tokens, dtype=torch.int32)
    q_rows = torch.cat(
        [
            torch.arange(split_rows, dtype=torch.int32),
            torch.arange(split_rows, split_rows + tail_rows, dtype=torch.int32),
        ]
    )

    w_qb = (torch.randn(q_lora, heads_all * d_q, dtype=torch.float32) * 0.05).to(
        torch.float8_e4m3fn
    )
    s_qb = torch.rand(heads_all * d_q, dtype=torch.float32) * 0.1 + 0.01
    w_uk_t = torch.randn(heads_all, d_nope, kv_rank, dtype=torch.bfloat16) * 0.05
    angles = torch.rand(total_tokens, d_pe // 2, dtype=torch.float32) * 6.0
    cos_sin_cache = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def _run_fused_ref(qc, wq, ws, wuk, cs, s_rows):
        q_full = (
            ((qc.to(torch.float32) @ wq.to(torch.float32)) * ws[None, :])
            .to(qc.dtype)
            .view(-1, heads_all, d_q)
        )
        qn, qp = q_full[:, :, :d_nope], q_full[:, :, d_nope:]
        ql = torch.einsum(
            "thd,hdr->thr", qn.to(torch.float32), wuk.to(torch.float32)
        ).to(qc.dtype)
        c, s = cs.to(torch.float32)[:, None, :32], cs.to(torch.float32)[:, None, 32:]
        qf = qp.to(torch.float32)
        qpe_rot = torch.cat(
            [qf[..., :32] * c - qf[..., 32:] * s, qf[..., 32:] * c + qf[..., :32] * s],
            dim=-1,
        ).to(qc.dtype)
        return ql[:s_rows], ql[s_rows:], qpe_rot

    class _FakeQProj(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(w_qb, requires_grad=False)
            self.weight_scale = torch.nn.Parameter(s_qb, requires_grad=False)
            self.bias = None
            self.gather_output = False
            self.group_size = dcp
            setattr(self, WEIGHT_FLIPPED_ATTR, True)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            y = (
                x.to(torch.float32) @ self.weight.to(torch.float32)
            ) * self.weight_scale[None, :]
            return y.to(x.dtype), None

    class _FakeRotary(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("cos_sin_cache", cos_sin_cache)
            self.rotary_dim = d_pe
            self.head_size = d_pe

        def _match_cos_sin_cache_dtype(self, x: torch.Tensor) -> torch.Tensor:
            return self.cos_sin_cache.to(x.dtype)

    def _build_wrapper():
        w = object.__new__(mla_op_mod.VllmTPUMultiHeadLatentAttentionWrapper)
        torch.nn.Module.__init__(w)
        w.hidden_size = 64
        w.num_heads = heads_local
        w.dcp_group_size = dcp
        w.dcp_q_replicate = True
        w.is_sparse = True
        w.skip_topk = True
        w.q_lora_rank = q_lora
        w.kv_lora_rank = kv_rank
        w.qk_nope_head_dim = d_nope
        w.qk_rope_head_dim = d_pe
        w.qk_head_dim = d_q
        w.v_head_dim = v_dim
        w.gate_is_fused = False
        w.kimi_preprocess = None
        w.g_proj = None
        w.indexer = None
        w.indexer_rope_emb = None
        w.qb_overlap = False
        w.decode_prologue = False
        w._rope_neox_permuted = True  # noqa: SLF001 -- mimics load-time NeoX weight permutation
        w.fused_qkv_a_proj = lambda x: (torch.cat([q_c_all, kv_lora_all], dim=-1), None)
        w.q_a_layernorm = lambda x: x
        w.kv_a_layernorm = lambda x: x
        w.q_b_proj = _FakeQProj()
        w.rotary_emb = _FakeRotary()
        w.o_proj = lambda x: (x, None)
        w.topk_indices_buffer = torch.zeros(total_tokens, 4, dtype=torch.int32)
        captured: dict[str, object] = {}

        def _fake_mla_attn(q_arg, kv_c, k_pe_arg, **kwargs):
            captured["q_arg"] = q_arg
            captured["kv_c"] = kv_c
            captured["k_pe"] = k_pe_arg
            captured["q_dcp_replicated"] = kwargs["q_dcp_replicated"]
            return torch.zeros(total_tokens, heads_local * v_dim, dtype=torch.bfloat16)

        tp_op = SimpleNamespace(
            q_replicated=True,
            split_q_inputs=True,
            serves_oproj=False,
            layout=SimpleNamespace(split_rows=split_rows),
            rank=torch.zeros(1, dtype=torch.int32),
        )
        _fake_mla_attn.sparse_mla_tp_op = tp_op  # type: ignore[attr-defined]
        _fake_mla_attn.W_UK_T_dcp_qrep = w_uk_t  # type: ignore[attr-defined]
        _fake_mla_attn.W_UK_T_dcp_qrep_scale = None  # type: ignore[attr-defined]
        w.mla_attn = _fake_mla_attn
        return w, captured

    def _fake_rope_half_split(rope, pos, q_in, k_in):
        cs = rope.cos_sin_cache.to(q_in.dtype).index_select(0, pos).to(torch.float32)
        c, s = cs[:, None, :32], cs[:, None, 32:]

        def _rot(t):
            tf = t.to(torch.float32)
            return torch.cat(
                [
                    tf[..., :32] * c - tf[..., 32:] * s,
                    tf[..., 32:] * c + tf[..., :32] * s,
                ],
                dim=-1,
            )

        return _rot(q_in), _rot(k_in)

    w_ref, cap_ref = _build_wrapper()
    w_fused, cap_fused = _build_wrapper()
    dummy_hidden = torch.zeros(total_tokens, 64, dtype=torch.bfloat16)
    fused_calls: list[int] = []

    def _spy_fused_op(*args):
        fused_calls.append(args[0].shape[0])
        return _run_fused_ref(*args)

    with (
        patch.object(tp_op_mod, "applies", return_value=True),
        patch.object(tp_op_mod, "q_rows", return_value=q_rows),
        patch.object(mla_op_mod, "rope_half_split", side_effect=_fake_rope_half_split),
        patch.object(
            mla_op_mod, "_FUSED_QB_ROPE_ABSORB_SPLIT_OP", side_effect=_spy_fused_op
        ),
        patch.object(mla_op_mod, "_retain_buffers", side_effect=lambda x, *_rest: x),
    ):
        with patch.object(envs, "TPU_MLA_FUSED_QB_ROPE_ABSORB", False):
            w_ref.forward(positions, dummy_hidden)

        # Simulate TPU device gate on CPU for wrapper contract verification.
        orig_operands = (
            mla_op_mod.VllmTPUMultiHeadLatentAttentionWrapper._fused_qb_rope_absorb_operands  # noqa: SLF001 -- exercises wrapper seam on CPU
        )

        def _cpu_tpu_operands(self, q_c, **kw):
            with patch.object(
                mla_op_mod,
                "supports_fused_qb_rope_absorb_split",
                wraps=supports_fused_qb_rope_absorb_split,
            ):
                fake_qc = SimpleNamespace(
                    shape=q_c.shape, dtype=q_c.dtype, device=SimpleNamespace(type="tpu")
                )
                return orig_operands(self, fake_qc, **kw)  # type: ignore[arg-type]

        with (
            patch.object(envs, "TPU_MLA_FUSED_QB_ROPE_ABSORB", True),
            patch.object(
                mla_op_mod.VllmTPUMultiHeadLatentAttentionWrapper,
                "_fused_qb_rope_absorb_operands",
                _cpu_tpu_operands,
            ),
        ):
            w_fused.forward(positions, dummy_hidden)

    assert fused_calls == [q_total]
    ref_qn, ref_qpe = cap_ref["q_dcp_replicated"]
    fused_ql_split, fused_qpe = cap_fused["q_dcp_replicated"]
    exp_ql = torch.einsum(
        "thd,hdr->thr", ref_qn.to(torch.float32), w_uk_t.to(torch.float32)
    ).to(torch.bfloat16)
    torch.testing.assert_close(fused_ql_split[0], exp_ql[:split_rows], rtol=0, atol=0)
    torch.testing.assert_close(fused_ql_split[1], exp_ql[split_rows:], rtol=0, atol=0)
    torch.testing.assert_close(fused_qpe, ref_qpe, rtol=0, atol=0)
    torch.testing.assert_close(cap_fused["k_pe"], cap_ref["k_pe"], rtol=0, atol=0)
