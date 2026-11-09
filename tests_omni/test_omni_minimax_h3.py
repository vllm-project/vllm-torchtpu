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
"""Unit tests for MiniMax-H3 Pallas kernels, TPU compile hook, and denoise patches."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.omni


def test_diffusion_adaln_pallas_interpret_parity():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from vllm_torchtpu.kernels.diffusion_adaln.kernel import (
        indexed_gate_pallas,
        indexed_gate_rms_norm_scale_shift_pallas,
        indexed_scale_shift_pallas,
        rms_norm_indexed_scale_shift_pallas,
    )

    eps = 1e-6
    rng = np.random.default_rng(0)
    # Exercise non-multiple-of-8 R (e.g. 6 and 9) and non-power-of-2 S (e.g. 17).
    for s, r, d in ((17, 6, 256), (256, 9, 256)):
        x_np = rng.standard_normal((s, d)).astype(np.float32)
        br_np = rng.standard_normal((s, d)).astype(np.float32)
        w_np = rng.standard_normal((d,)).astype(np.float32)
        shift_np = rng.standard_normal((r, d)).astype(np.float32)
        scale_np = rng.standard_normal((r, d)).astype(np.float32)
        gate_np = rng.standard_normal((r, d)).astype(np.float32)
        idx_np = rng.integers(0, r, size=(s,), dtype=np.int32)

        x_t = torch.from_numpy(x_np).to(torch.bfloat16)
        br_t = torch.from_numpy(br_np).to(torch.bfloat16)
        w_t = torch.from_numpy(w_np).to(torch.bfloat16)
        shift_t = torch.from_numpy(shift_np).to(torch.bfloat16)
        scale_t = torch.from_numpy(scale_np).to(torch.bfloat16)
        gate_t = torch.from_numpy(gate_np).to(torch.bfloat16)
        idx_t = torch.from_numpy(idx_np).to(torch.long)

        x_j = jnp.asarray(x_t.float().numpy(), dtype=jnp.bfloat16)
        br_j = jnp.asarray(br_t.float().numpy(), dtype=jnp.bfloat16)
        w_j = jnp.asarray(w_t.float().numpy(), dtype=jnp.bfloat16)
        shift_j = jnp.asarray(shift_t.float().numpy(), dtype=jnp.bfloat16)
        scale_j = jnp.asarray(scale_t.float().numpy(), dtype=jnp.bfloat16)
        gate_j = jnp.asarray(gate_t.float().numpy(), dtype=jnp.bfloat16)
        idx_j = jnp.asarray(idx_np, dtype=jnp.int32)

        # 1. rms_norm_indexed_scale_shift
        got_rms = rms_norm_indexed_scale_shift_pallas(
            x_j, w_j, shift_j, scale_j, idx_j, eps=eps, interpret=True
        )
        got_rms_t = torch.from_numpy(np.asarray(got_rms, dtype=np.float32)).to(
            torch.bfloat16
        )
        norm_ref = (
            x_t.float()
            * torch.rsqrt(x_t.float().pow(2).mean(-1, keepdim=True) + eps)
            * w_t.float()
        )
        want_rms = (
            norm_ref * (1.0 + scale_t[idx_t].float()) + shift_t[idx_t].float()
        ).to(torch.bfloat16)
        torch.testing.assert_close(got_rms_t, want_rms, rtol=1e-2, atol=1e-2)

        # 2. indexed_gate_rms_norm_scale_shift
        got_res, got_mod = indexed_gate_rms_norm_scale_shift_pallas(
            x_j, gate_j, br_j, w_j, shift_j, scale_j, idx_j, eps=eps, interpret=True
        )
        got_res_t = torch.from_numpy(np.asarray(got_res, dtype=np.float32)).to(
            torch.bfloat16
        )
        got_mod_t = torch.from_numpy(np.asarray(got_mod, dtype=np.float32)).to(
            torch.bfloat16
        )
        want_res = (x_t.float() + gate_t[idx_t].float() * br_t.float()).to(
            torch.bfloat16
        )
        upd = want_res.float()
        want_mod = (
            upd
            * torch.rsqrt(upd.pow(2).mean(-1, keepdim=True) + eps)
            * w_t.float()
            * (1.0 + scale_t[idx_t].float())
            + shift_t[idx_t].float()
        ).to(torch.bfloat16)
        torch.testing.assert_close(got_res_t, want_res, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(got_mod_t, want_mod, rtol=1e-2, atol=1e-2)

        # 3. indexed_gate & indexed_scale_shift
        got_gate = indexed_gate_pallas(x_j, gate_j, br_j, idx_j, interpret=True)
        got_gate_t = torch.from_numpy(np.asarray(got_gate, dtype=np.float32)).to(
            torch.bfloat16
        )
        torch.testing.assert_close(got_gate_t, want_res, rtol=1e-2, atol=1e-2)

        got_ss = indexed_scale_shift_pallas(
            x_j, shift_j, scale_j, idx_j, interpret=True
        )
        got_ss_t = torch.from_numpy(np.asarray(got_ss, dtype=np.float32)).to(
            torch.bfloat16
        )
        want_ss = (
            x_t.float() * (1.0 + scale_t[idx_t].float()) + shift_t[idx_t].float()
        ).to(torch.bfloat16)
        torch.testing.assert_close(got_ss_t, want_ss, rtol=1e-2, atol=1e-2)
    del jax


def test_minimax_qk_norm_rope_pallas_interpret_parity():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import _eager_qk_norm_rope

    from vllm_torchtpu.kernels.diffusion_attention.minimax_qk_norm_rope import (
        fused_minimax_qk_norm_rope,
    )

    eps = 1e-6
    rng = np.random.default_rng(42)
    # Test non-multiple-of-8 heads (e.g. H=7, H=56) and unaligned token counts (T=17).
    for t, h in ((17, 7), (32, 14)):
        q_t = torch.from_numpy(rng.standard_normal((t, h, 128)).astype(np.float32)).to(
            torch.bfloat16
        )
        k_t = torch.from_numpy(rng.standard_normal((t, h, 128)).astype(np.float32)).to(
            torch.bfloat16
        )
        qw_t = torch.from_numpy(rng.standard_normal((128,)).astype(np.float32)).to(
            torch.bfloat16
        )
        kw_t = torch.from_numpy(rng.standard_normal((128,)).astype(np.float32)).to(
            torch.bfloat16
        )
        angles = torch.from_numpy(rng.standard_normal((t, 48)).astype(np.float32))
        rope_t = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).to(
            torch.bfloat16
        )

        want_q, want_k = _eager_qk_norm_rope(
            q_t, k_t, qw_t, kw_t, rope_t, eps, 128, 96, interleaved=False
        )

        qo_j, ko_j = fused_minimax_qk_norm_rope(
            jnp.asarray(q_t.float().numpy(), dtype=jnp.bfloat16),
            jnp.asarray(k_t.float().numpy(), dtype=jnp.bfloat16),
            jnp.asarray(qw_t.float().numpy(), dtype=jnp.bfloat16),
            jnp.asarray(kw_t.float().numpy(), dtype=jnp.bfloat16),
            jnp.asarray(rope_t.float().numpy(), dtype=jnp.bfloat16),
            rot_dim=96,
            eps=eps,
            interpret=True,
        )
        got_q = torch.from_numpy(np.asarray(qo_j, dtype=np.float32)).to(torch.bfloat16)
        got_k = torch.from_numpy(np.asarray(ko_j, dtype=np.float32)).to(torch.bfloat16)
        torch.testing.assert_close(got_q, want_q, rtol=1.6e-2, atol=1.6e-2)
        torch.testing.assert_close(got_k, want_k, rtol=1.6e-2, atol=1.6e-2)


def test_diffusion_tpu_compile_patch(monkeypatch):
    from vllm_omni.diffusion.worker import diffusion_model_runner as dmr

    from vllm_torchtpu.omni import patches

    compiled_calls: list[tuple[Any, dict]] = []

    def fake_regionally_compile(model, *args, **kwargs):
        compiled_calls.append((model, kwargs))
        return model

    monkeypatch.setattr(
        "vllm_omni.diffusion.compile.regionally_compile", fake_regionally_compile
    )

    class DummyBlock(nn.Module):
        def forward(self, x):
            return x

    class DummyDiT(nn.Module):
        _repeated_blocks = ["DummyBlock"]

        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([DummyBlock()])

    orig_load = lambda self, *a, **kw: None  # noqa: E731
    orig_comp = lambda self, attr: compiled_calls.append(("orig", {"attr": attr}))  # noqa: E731
    monkeypatch.setattr(dmr.DiffusionModelRunner, "load_model", orig_load)
    monkeypatch.setattr(dmr.DiffusionModelRunner, "_compile_transformer", orig_comp)

    patches._patch_diffusion_tpu_compile()

    dit = DummyDiT()
    runner = SimpleNamespace(
        device=SimpleNamespace(type="tpu"),
        pipeline=SimpleNamespace(_dit_modules=("transformer",), transformer=dit),
        od_config=SimpleNamespace(
            enforce_eager=False,
            diffusion_compile_dynamic=True,
            diffusion_compile_granularity="regional",
        ),
    )
    runner._compile_transformer = dmr.DiffusionModelRunner._compile_transformer.__get__(
        runner
    )
    dmr.DiffusionModelRunner.load_model(runner)
    assert len(compiled_calls) == 1
    assert compiled_calls[0][0] is dit
    assert compiled_calls[0][1] == {"backend": "tpu", "dynamic": False}


def test_minimax_h3_static_timestep_classes_equivalence():
    """Verify the static 4-class unique_timesteps + inverse_indices maps every
    packed token to the exact same timestep value as fill_timesteps + torch.unique,
    both when step 0 has t_video == t_audio == 0 and when step > 0 differs."""
    seq_len = 24
    img_pos = torch.tensor([4, 5, 6, 7, 8, 9], dtype=torch.long)
    upd_mask = torch.tensor([False, False, True, True, True, True], dtype=torch.bool)
    aud_pos = torch.tensor([10, 11, 12, 13], dtype=torch.long)
    aud_upd_mask = torch.tensor([True, True, True, False], dtype=torch.bool)

    cls = torch.zeros(seq_len, dtype=torch.long)
    cls[img_pos[upd_mask]] = 0
    cls[img_pos[~upd_mask]] = 1
    cls[aud_pos[aud_upd_mask]] = 2
    cls[aud_pos[~aud_upd_mask]] = 3
    present_classes, static_inv = torch.unique(cls, sorted=True, return_inverse=True)
    present_tuple = tuple(int(c) for c in present_classes.tolist())
    assert present_tuple == (0, 1, 2, 3)

    for t_v, t_a, cond_v, cond_a in (
        (0.0, 0.0, 0.05, 0.02),  # Step 0 where t_v == t_a == 0.0!
        (0.25, 0.60, 0.25, 0.60),  # Step > 0 where cond == target!
        (0.12, 0.45, 0.20, 0.50),  # All 4 distinct!
    ):
        # Reference upstream fill_timesteps + torch.unique
        ref_ts = torch.full((seq_len,), float(t_v), dtype=torch.float32)
        ref_ts[img_pos[upd_mask]] = float(t_v)
        ref_ts[img_pos[~upd_mask]] = float(cond_v)
        ref_ts[aud_pos[aud_upd_mask]] = float(t_a)
        ref_ts[aud_pos[~aud_upd_mask]] = float(cond_a)
        ref_u, ref_inv = torch.unique(ref_ts, sorted=True, return_inverse=True)

        # Static fixed-M path
        vals = (float(t_v), float(cond_v), float(t_a), float(cond_a))
        stat_u = torch.tensor([vals[c] for c in present_tuple], dtype=torch.float32)

        assert stat_u.shape == (4,)  # Constant M across all steps!
        torch.testing.assert_close(stat_u[static_inv], ref_u[ref_inv], rtol=0, atol=0)


def test_ulysses_batched_qkv_a2a_returns_5_tuple(monkeypatch):
    from vllm_omni.diffusion.attention.parallel import ulysses as uly_mod
    from vllm_omni.diffusion.distributed import comm as comm_mod

    from vllm_torchtpu.omni import patches

    patches._patch_ulysses_batched_qkv_a2a()
    a2a_calls: list[tuple[int, ...]] = []

    def fake_apply(pg, inp, scatter_idx, gather_idx, use_sync):
        a2a_calls.append(tuple(inp.shape))
        # Simulate P=2 S<->H transpose on (B, S_local, H, D) -> (B, S_local*2, H//2, D)
        b, s, h, d = inp.shape
        return (
            inp.reshape(b, s, 2, h // 2, d)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, s * 2, h // 2, d)
        )

    monkeypatch.setattr(comm_mod.SeqAllToAll4D, "apply", staticmethod(fake_apply))

    class FakeTPUTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="tpu")

    q = torch.randn(1, 5, 4, 8).as_subclass(FakeTPUTensor)
    k = torch.randn(1, 5, 4, 8).as_subclass(FakeTPUTensor)
    v = torch.randn(1, 5, 4, 8).as_subclass(FakeTPUTensor)
    meta = SimpleNamespace(
        joint_query=None,
        joint_key=None,
        joint_value=None,
        attn_mask=None,
        joint_attn_mask=None,
        extra={},
    )
    inst = object.__new__(uly_mod.UlyssesParallelAttention)
    inst._ulysses_pg = object()
    inst._sp_group = SimpleNamespace(ulysses_world_size=2)
    inst._ulysses_a2a_permute = False
    inst._scatter_idx = 2
    inst._gather_idx = 1
    inst._use_sync = False

    ret = inst.pre_attention(q, k, v, meta)
    assert len(ret) == 5
    q_ex, k_ex, v_ex, meta_out, ctx = ret
    assert meta_out is meta
    assert len(a2a_calls) == 1
    assert a2a_calls[0] == (3, 5, 4, 8)
    assert q_ex.shape == k_ex.shape == v_ex.shape == (1, 10, 2, 8)
    assert ctx.scatter_idx == 2 and ctx.gather_idx == 1


def test_block_sparse_attention_pallas_interpret_parity():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from vllm_torchtpu.kernels.diffusion_attention.block_sparse import (
        block_sparse_attention_bshd,
        sol_routing_stats,
    )

    rng = np.random.default_rng(7)
    bsz, seq_len, heads, dim = 1, 512, 2, 128
    valid_len = 480
    prefix_tokens = 140
    block = 128
    scale = dim**-0.5

    q_t = torch.from_numpy(
        rng.standard_normal((bsz, seq_len, heads, dim)).astype(np.float32) * 0.5
    ).to(torch.bfloat16)
    k_t = torch.from_numpy(
        rng.standard_normal((bsz, seq_len, heads, dim)).astype(np.float32) * 0.5
    ).to(torch.bfloat16)
    v_t = torch.from_numpy(
        rng.standard_normal((bsz, seq_len, heads, dim)).astype(np.float32) * 0.5
    ).to(torch.bfloat16)

    # PyTorch exact masked reference over valid_len tokens
    qf = q_t.float().permute(0, 2, 1, 3)  # [B, H, S, D]
    kf = k_t.float().permute(0, 2, 1, 3)
    vf = v_t.float().permute(0, 2, 1, 3)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    valid_k = torch.arange(seq_len) < valid_len
    scores = scores.masked_fill(~valid_k[None, None, None, :], -1e30)
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(probs, vf).permute(0, 2, 1, 3).to(torch.bfloat16)

    q_j = jnp.asarray(q_t.float().numpy(), dtype=jnp.bfloat16)
    k_j = jnp.asarray(k_t.float().numpy(), dtype=jnp.bfloat16)
    v_j = jnp.asarray(v_t.float().numpy(), dtype=jnp.bfloat16)

    # 1. Dense parity (max_density=1.0) against exact masked softmax over valid_len
    got_dense_j = block_sparse_attention_bshd(
        q_j,
        k_j,
        v_j,
        scale=scale,
        valid_len=valid_len,
        prefix_tokens=prefix_tokens,
        block=block,
        max_density=1.0,
        sink_stride=4,
        interpret=True,
    )
    got_dense = torch.from_numpy(np.asarray(got_dense_j, dtype=np.float32)).to(
        torch.bfloat16
    )
    torch.testing.assert_close(
        got_dense[:, :valid_len],
        ref_out[:, :valid_len],
        rtol=1.6e-2,
        atol=1.6e-2,
    )

    # 2. Sparse routing (max_density=0.5, prefix=140 -> 2 prefix blocks = 256 rows):
    #    Prefix query blocks [0:256] MUST match exact dense attention!
    got_sparse_j = block_sparse_attention_bshd(
        q_j,
        k_j,
        v_j,
        scale=scale,
        valid_len=valid_len,
        prefix_tokens=prefix_tokens,
        block=block,
        max_density=0.5,
        sink_stride=4,
        interpret=True,
    )
    got_sparse = torch.from_numpy(np.asarray(got_sparse_j, dtype=np.float32)).to(
        torch.bfloat16
    )
    assert torch.isfinite(got_sparse.float()).all()
    torch.testing.assert_close(
        got_sparse[:, :256],
        ref_out[:, :256],
        rtol=1.6e-2,
        atol=1.6e-2,
    )

    # 3. Verify Customer C1 routing density at S=37,760 with block=1024, sink_stride=8
    seg_c1 = (jnp.arange(37760, dtype=jnp.int32) >= 37719).astype(jnp.int32)
    q_c1 = jnp.ones((1, 37760, 128), dtype=jnp.bfloat16) * 0.1
    stats = sol_routing_stats(
        q_c1,
        q_c1,
        seg_c1,
        block=1024,
        prefix_tokens=412,
        max_density=0.25,
        sink_stride=8,
    )
    assert stats["n_b"] == 37
    assert stats["k_cap"] == 10
    assert 0.20 <= float(stats["density"]) <= 0.30


def test_tpu_omni_fp8_config_layer_policy():
    from vllm.model_executor.layers.linear import LinearBase
    from vllm_omni.quantization.factory import build_quant_config

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedLinearMethod,
    )
    from vllm_torchtpu.omni.quantization import (
        TpuOmniFp8Config,
        register_omni_tpu_quantization,
    )

    register_omni_tpu_quantization()
    cfg = build_quant_config(
        {
            "method": "fp8",
            "ignored_layers": ["blocks.0.*", "blocks.1.*", "blocks.49.*"],
        }
    )
    assert isinstance(cfg, TpuOmniFp8Config)
    assert not cfg.is_checkpoint_fp8_serialized

    dummy = LinearBase(128, 128, params_dtype=torch.bfloat16, tp_rank=0, tp_size=1)
    for skipped_prefix in (
        "blocks.0.attn.qkv_proj",
        "blocks.1.mlp.fc1",
        "blocks.49.attn.out_proj",
        "blocks.5.adaln_proj.linear",
        "condition_proj",
        "token_refiner.blocks.0.attn.qkv_proj",
        "final_layer.adaln_proj.linear",
        "text_encoder.model.layers.0.self_attn.qkv_proj",
    ):
        m = cfg.get_quant_method(dummy, skipped_prefix)
        assert isinstance(m, VllmUnquantizedLinearMethod), skipped_prefix

    for quant_prefix in (
        "blocks.2.attn.qkv_proj",
        "blocks.2.attn.out_proj",
        "blocks.25.mlp.fc1",
        "blocks.46.mlp.fc2",
    ):
        m = cfg.get_quant_method(dummy, quant_prefix)
        assert isinstance(m, VllmFp8LinearMethodTPU), quant_prefix


def test_tpu_block_sparse_step_gate_and_skip_layers(monkeypatch):
    from vllm_omni.diffusion import forward_context as fc
    from vllm_omni.diffusion.data import AttentionSpec
    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        override_forward_context,
    )

    from vllm_torchtpu.omni import patches
    from vllm_torchtpu.omni.attention import (
        TpuBlockSparseAttentionBackend,
        TpuBlockSparseAttentionImpl,
    )
    from vllm_torchtpu.omni.platform import OmniTpuPlatform

    monkeypatch.setattr(patches, "apply_omni_model_specific_patches", lambda: None)
    patches._patch_block_sparse_attention_registry()
    for name in ("RAINFUSION_ATTN", "FASTVIDEO_VSA", "TPU_BLOCK_SPARSE"):
        assert (
            OmniTpuPlatform.get_diffusion_attn_backend_cls(name, 128)
            == "vllm_torchtpu.omni.attention.TpuBlockSparseAttentionBackend"
        )
    assert TpuBlockSparseAttentionBackend.get_name() == "TPU_BLOCK_SPARSE"

    spec = AttentionSpec(
        backend="FASTVIDEO_VSA",
        block_sparse={"sparsity": 0.75, "start_step": 10, "skip_layers": "0-1"},
    )
    bkw = spec.backend_kwargs()
    assert bkw is not None

    imp_skip = TpuBlockSparseAttentionImpl(
        7, 128, 128**-0.5, prefix="blocks.0.attn", backend_kwargs=bkw, role="self"
    )
    imp_refiner = TpuBlockSparseAttentionImpl(
        7,
        128,
        128**-0.5,
        prefix="token_refiner.blocks.0.attn",
        backend_kwargs=bkw,
        role="minimax_h3.token_refiner",
    )
    imp_active = TpuBlockSparseAttentionImpl(
        7, 128, 128**-0.5, prefix="blocks.2.attn", backend_kwargs=bkw, role="self"
    )
    assert not imp_skip._sparse_step_active
    assert not imp_refiner._sparse_step_active
    assert not imp_active._sparse_step_active  # start_step=10 -> starts False

    with override_forward_context(ForwardContext()):
        fc.set_forward_context_denoise_total_steps(50)
        fc.set_forward_context_denoise_step_idx(9)
        assert not imp_active._sparse_step_active
        fc.set_forward_context_denoise_step_idx(10)
        assert imp_active._sparse_step_active
        assert not imp_skip._sparse_step_active
        assert not imp_refiner._sparse_step_active
        fc.set_forward_context_denoise_step_idx(None)
        assert not imp_active._sparse_step_active

    # Also verify step-gate updates work when ForwardContext is NOT active
    # (matching MiniMaxH3Pipeline's minimax_h3_publish_denoise_progress) AND
    # that TorchDynamo guards on _sparse_step_active across dense->sparse transition.
    fc.set_forward_context_denoise_total_steps(50)
    fc.set_forward_context_denoise_step_idx(9)
    assert not imp_active._sparse_step_active
    fc.set_forward_context_denoise_step_idx(10)
    assert imp_active._sparse_step_active

    calls: list[str] = []
    imp_dyn = TpuBlockSparseAttentionImpl(
        2,
        128,
        128**-0.5,
        prefix="blocks.5.attn",
        backend_kwargs={**bkw, "block_size": 128},
        role="self",
    )
    imp_dyn._op = lambda q, k, v, sc: (calls.append("dense"), q)[1]
    imp_dyn._sparse_op = lambda q, k, v, *a: (calls.append("sparse"), q * 2)[1]

    class Wrap(nn.Module):
        def __init__(self, impl):
            super().__init__()
            self.impl = impl

        def forward(self, q):
            return self.impl.forward(q, q, q, None)

    c_wrap = torch.compile(Wrap(imp_dyn), backend="eager", dynamic=False)
    q_in = torch.ones(1, 512, 2, 128, dtype=torch.bfloat16)
    fc.set_forward_context_denoise_step_idx(2)
    out_d = c_wrap(q_in)
    fc.set_forward_context_denoise_step_idx(12)
    out_s = c_wrap(q_in)
    fc.set_forward_context_denoise_step_idx(None)
    assert calls == ["dense", "sparse"]
    assert float(out_d[0, 0, 0, 0]) == 1.0
    assert float(out_s[0, 0, 0, 0]) == 2.0


def test_tpu_fp16_sdpa_upcast_patch(monkeypatch):
    import torch.nn.functional as F

    from vllm_torchtpu.omni import patches

    seen_dtypes: list[torch.dtype] = []

    def fake_sdpa(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        seen_dtypes.append(q.dtype)
        return q.clone()

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)
    patches._patch_tpu_fp16_sdpa()

    class FakeTPUTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="tpu")

    q = torch.randn(1, 4, 16, 64, dtype=torch.float16).as_subclass(FakeTPUTensor)
    out = F.scaled_dot_product_attention(q, q, q)
    assert seen_dtypes == [torch.bfloat16]
    assert out.dtype == torch.float16


def test_minimax_h3_tpu_helpers_and_lora_compile_ordering(monkeypatch):
    from vllm_omni.diffusion.worker import diffusion_model_runner as dmr

    from vllm_torchtpu.omni import patches
    from vllm_torchtpu.omni.minimax_h3 import (
        disable_tpu_adaln_cache,
        tpu_rf_v_to_x0_nosync,
        tpu_solver_step_nosync,
        wrap_condition_proj_cache,
    )

    # 1. disable_tpu_adaln_cache clears adaln_cache across transformer and submodules
    cache_obj = SimpleNamespace(sidecar=None)
    b0 = SimpleNamespace(_adaln_cache=cache_obj)
    tr = SimpleNamespace(
        adaln_cache=cache_obj,
        _adaln_offload=False,
        modules=lambda: [b0],
    )
    disable_tpu_adaln_cache(tr)
    assert tr.adaln_cache is None
    assert b0._adaln_cache is None

    # 2. wrap_condition_proj_cache caches across steps and invalidates on new req
    proj_calls = 0

    class FakeTPUTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="tpu")

    class FakeCondProj:
        def forward(self, x):
            nonlocal proj_calls
            proj_calls += 1
            return x * 2.0

    class FakeDiT:
        def __init__(self):
            self.condition_proj = FakeCondProj()
            self._tpu_active_denoise_req_id = "req-1"

    tr_cond = FakeDiT()
    wrap_condition_proj_cache(tr_cond)
    cond1 = torch.ones(1, 8, 16).as_subclass(FakeTPUTensor)
    out1 = tr_cond.condition_proj.forward(cond1)
    out1_repeat = tr_cond.condition_proj.forward(cond1)
    tr_cond._tpu_active_denoise_req_id = "req-2"
    out2 = tr_cond.condition_proj.forward(cond1)
    assert proj_calls == 2
    torch.testing.assert_close(out1, out1_repeat)
    torch.testing.assert_close(out1, out2)

    # 3. tpu_rf_v_to_x0_nosync & tpu_solver_step_nosync match exact scalar equations
    xt = torch.tensor([[2.0, 4.0]], dtype=torch.bfloat16)
    v = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16)
    x0 = tpu_rf_v_to_x0_nosync(xt, v, 0.25)
    # xt + (1 - 0.25) * v = [2 + 0.75, 4 - 0.75] = [2.75, 3.25]
    torch.testing.assert_close(
        x0, torch.tensor([[2.75, 3.25]], dtype=torch.bfloat16), rtol=1e-3, atol=1e-3
    )
    solver = SimpleNamespace(
        name="euler",
        step_index=0,
        coeffs=[None],
        sigmas=[0.75, 0.50],
        old_denoised=None,
    )
    x_next = tpu_solver_step_nosync(solver, xt, x0, 0)
    # sigma_ratio = 0.50 / 0.75 = 2/3
    want_next = torch.tensor(
        [
            [
                2.0 * (2.0 / 3.0) + 2.75 * (1.0 / 3.0),
                4.0 * (2.0 / 3.0) + 3.25 * (1.0 / 3.0),
            ]
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(x_next, want_next, rtol=1e-2, atol=1e-2)
    assert solver.step_index == 1

    # 4. _tpu_load_model calls initialize_lora before _compile_transformer
    order: list[str] = []
    monkeypatch.setattr(
        dmr.DiffusionModelRunner,
        "load_model",
        lambda self, *a, **kw: order.append("orig_load"),
    )
    patches._patch_diffusion_tpu_compile()

    class DummyDiT(nn.Module):
        _repeated_blocks = ["DummyBlock"]

    runner = SimpleNamespace(
        device=SimpleNamespace(type="tpu"),
        pipeline=SimpleNamespace(_dit_modules=("transformer",), transformer=DummyDiT()),
        od_config=SimpleNamespace(
            enforce_eager=False,
            lora_path="/tmp/fake_lora.safetensors",
            diffusion_compile_dynamic=False,
            diffusion_compile_granularity="regional",
        ),
        _compile_transformer=lambda attr: order.append(f"compile:{attr}"),
    )
    dmr.DiffusionModelRunner.load_model(
        runner,
        load_format="default",
        custom_pipeline_name=None,
        initialize_lora=lambda: order.append("init_lora"),
    )
    assert order == ["orig_load", "init_lora", "compile:transformer"]


def test_extract_prefix_tokens_ref2va_text_cap_and_sdpa_chunking(monkeypatch):
    from dataclasses import dataclass

    import torch.nn.functional as F

    from vllm_torchtpu.omni import minimax_h3 as mh3
    from vllm_torchtpu.omni import patches
    from vllm_torchtpu.omni.minimax_h3 import _extract_prefix_tokens

    layout_fl2va = SimpleNamespace(
        video_spans=(SimpleNamespace(role="target", start=4602),),
        _tpu_text_len=1528,
    )
    assert _extract_prefix_tokens(layout_fl2va, valid_kv_length=105984) == 4602

    @dataclass(frozen=True, slots=True)
    class _FrozenLayout:
        video_spans: tuple[Any, ...]

    monkeypatch.setattr(mh3, "_LAST_BRANCH_TEXT_LEN", 43306)
    layout_ref2va = _FrozenLayout(
        video_spans=(
            SimpleNamespace(role="reference", start=44458),
            SimpleNamespace(role="target", start=177130),
        ),
    )
    assert _extract_prefix_tokens(layout_ref2va, valid_kv_length=314218) == 43306

    seen_shapes: list[tuple[int, ...]] = []

    def fake_sdpa(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        seen_shapes.append(tuple(q.shape))
        return q.clone()

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)
    patches._patch_tpu_fp16_sdpa()

    class FakeTPUTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="tpu")

    q_causal = torch.ones(1, 2, 300, 128, dtype=torch.bfloat16).as_subclass(
        FakeTPUTensor
    )
    out_causal = F.scaled_dot_product_attention(
        q_causal, q_causal, q_causal, is_causal=True
    )
    assert seen_shapes == [(1, 2, 384, 128)]
    assert tuple(out_causal.shape) == (1, 2, 300, 128)

    seen_shapes.clear()
    q_vit = torch.ones(1, 2, 2500, 72, dtype=torch.bfloat16).as_subclass(FakeTPUTensor)
    out_vit = F.scaled_dot_product_attention(q_vit, q_vit, q_vit, is_causal=False)
    assert seen_shapes == [(1, 2, 2048, 72), (1, 2, 452, 72)]
    assert tuple(out_vit.shape) == (1, 2, 2500, 72)
