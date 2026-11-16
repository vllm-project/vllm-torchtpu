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

from tests_omni.omni_test_utils import assert_parity

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
        assert_parity(
            got_rms_t,
            want_rms,
            name=f"adaln_rms_s{s}_r{r}",
            max_rel_l2=5e-3,
            max_abs=2e-2,
        )

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
        assert_parity(
            got_res_t,
            want_res,
            name=f"adaln_res_s{s}_r{r}",
            max_rel_l2=5e-3,
            max_abs=2e-2,
        )
        assert_parity(
            got_mod_t,
            want_mod,
            name=f"adaln_mod_s{s}_r{r}",
            max_rel_l2=5e-3,
            max_abs=2e-2,
        )

        # 3. indexed_gate & indexed_scale_shift
        got_gate = indexed_gate_pallas(x_j, gate_j, br_j, idx_j, interpret=True)
        got_gate_t = torch.from_numpy(np.asarray(got_gate, dtype=np.float32)).to(
            torch.bfloat16
        )
        assert_parity(
            got_gate_t,
            want_res,
            name=f"adaln_gate_s{s}_r{r}",
            max_rel_l2=5e-3,
            max_abs=2e-2,
        )

        got_ss = indexed_scale_shift_pallas(
            x_j, shift_j, scale_j, idx_j, interpret=True
        )
        got_ss_t = torch.from_numpy(np.asarray(got_ss, dtype=np.float32)).to(
            torch.bfloat16
        )
        want_ss = (
            x_t.float() * (1.0 + scale_t[idx_t].float()) + shift_t[idx_t].float()
        ).to(torch.bfloat16)
        assert_parity(
            got_ss_t,
            want_ss,
            name=f"adaln_ss_s{s}_r{r}",
            max_rel_l2=5e-3,
            max_abs=2e-2,
        )
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
        # Products of unit-normal activations and weights reach |x| ~ 4-6 where
        # 1 BF16 ULP is 0.03125; keep rel_l2 tight at 5e-3 and allow <= 2 ULPs.
        assert_parity(
            got_q,
            want_q,
            name=f"qknorm_rope_q_t{t}_h{h}",
            max_rel_l2=5e-3,
            max_abs=6.5e-2,
        )
        assert_parity(
            got_k,
            want_k,
            name=f"qknorm_rope_k_t{t}_h{h}",
            max_rel_l2=5e-3,
            max_abs=6.5e-2,
        )


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

    # Explicit _tpu_text_len=0 must NOT fall back to stale _LAST_BRANCH_TEXT_LEN.
    layout_zero_text = SimpleNamespace(
        video_spans=(SimpleNamespace(role="target", start=177130),),
        _tpu_text_len=0,
    )
    assert _extract_prefix_tokens(layout_zero_text, valid_kv_length=314218) == 177130

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

    # Non-causal d=128 with 128-aligned KV pads Q in one call (no chunk loop).
    seen_shapes.clear()
    q_pad_only = torch.ones(1, 2, 2500, 128, dtype=torch.bfloat16).as_subclass(
        FakeTPUTensor
    )
    kv_aligned = torch.ones(1, 2, 2560, 128, dtype=torch.bfloat16).as_subclass(
        FakeTPUTensor
    )
    out_pad_only = F.scaled_dot_product_attention(
        q_pad_only, kv_aligned, kv_aligned, is_causal=False
    )
    assert seen_shapes == [(1, 2, 2560, 128)]
    assert tuple(out_pad_only.shape) == (1, 2, 2500, 128)

    # Non-causal d=72 (non-128-multiple) chunks Q to avoid MATH SDPA HBM OOM.
    seen_shapes.clear()
    q_vit = torch.ones(1, 2, 2500, 72, dtype=torch.bfloat16).as_subclass(FakeTPUTensor)
    out_vit = F.scaled_dot_product_attention(q_vit, q_vit, q_vit, is_causal=False)
    assert seen_shapes == [(1, 2, 2048, 72), (1, 2, 452, 72)]
    assert tuple(out_vit.shape) == (1, 2, 2500, 72)


@pytest.mark.tpu_test
def test_diffusion_adaln_pallas_tpu_hardware_parity():
    from vllm_torchtpu.omni.adaln import (
        MIN_PALLAS_ADALN_TOKENS,
        register_adaln_pallas_ops,
        tpu_indexed_gate,
        tpu_indexed_gate_rms_norm_scale_shift,
        tpu_indexed_scale_shift_,
        tpu_rms_norm_indexed_scale_shift,
    )

    register_adaln_pallas_ops()
    dev = torch.device("tpu")
    eps = 1e-6
    s, r, d = MIN_PALLAS_ADALN_TOKENS, 4, 256
    torch.manual_seed(0)
    x_cpu = torch.randn(s, d, dtype=torch.bfloat16)
    br_cpu = torch.randn(s, d, dtype=torch.bfloat16)
    w_cpu = torch.randn(d, dtype=torch.bfloat16)
    shift_cpu = torch.randn(r, d, dtype=torch.bfloat16)
    scale_cpu = torch.randn(r, d, dtype=torch.bfloat16)
    gate_cpu = torch.randn(r, d, dtype=torch.bfloat16)
    idx_cpu = torch.randint(0, r, (s,), dtype=torch.int32)

    x_tpu = x_cpu.to(dev)
    br_tpu = br_cpu.to(dev)
    w_tpu = w_cpu.to(dev)
    shift_tpu = shift_cpu.to(dev)
    scale_tpu = scale_cpu.to(dev)
    gate_tpu = gate_cpu.to(dev)
    idx_tpu = idx_cpu.to(dev)

    got_rms = tpu_rms_norm_indexed_scale_shift(
        x_tpu, w_tpu, shift_tpu, scale_tpu, idx_tpu, eps
    )
    got_res, got_mod = tpu_indexed_gate_rms_norm_scale_shift(
        x_tpu, gate_tpu, br_tpu, w_tpu, shift_tpu, scale_tpu, idx_tpu, eps
    )
    got_gate = tpu_indexed_gate(x_tpu, gate_tpu, br_tpu, idx_tpu)
    got_ss = tpu_indexed_scale_shift_(x_tpu.clone(), shift_tpu, scale_tpu, idx_tpu)
    torch.accelerator.synchronize()

    idx_l = idx_cpu.long()
    norm_ref = (
        x_cpu.float()
        * torch.rsqrt(x_cpu.float().pow(2).mean(-1, keepdim=True) + eps)
        * w_cpu.float()
    )
    want_rms = (
        norm_ref * (1.0 + scale_cpu[idx_l].float()) + shift_cpu[idx_l].float()
    ).to(torch.bfloat16)
    want_res = (x_cpu.float() + gate_cpu[idx_l].float() * br_cpu.float()).to(
        torch.bfloat16
    )
    upd = want_res.float()
    want_mod = (
        upd
        * torch.rsqrt(upd.pow(2).mean(-1, keepdim=True) + eps)
        * w_cpu.float()
        * (1.0 + scale_cpu[idx_l].float())
        + shift_cpu[idx_l].float()
    ).to(torch.bfloat16)
    want_ss = (
        x_cpu.float() * (1.0 + scale_cpu[idx_l].float()) + shift_cpu[idx_l].float()
    ).to(torch.bfloat16)

    # Over 4.2M normal-product elements (S=16,384, D=256), tail magnitudes reach
    # ~16 where 1 BF16 ULP is 0.125; keep global rel_l2 tight at 5e-3.
    assert_parity(
        got_rms, want_rms, name="tpu_adaln_rms", max_rel_l2=5e-3, max_abs=1.5e-1
    )
    assert_parity(
        got_res, want_res, name="tpu_adaln_res", max_rel_l2=5e-3, max_abs=1.5e-1
    )
    assert_parity(
        got_mod, want_mod, name="tpu_adaln_mod", max_rel_l2=5e-3, max_abs=1.5e-1
    )
    assert_parity(
        got_gate, want_res, name="tpu_adaln_gate", max_rel_l2=5e-3, max_abs=1.5e-1
    )
    assert_parity(got_ss, want_ss, name="tpu_adaln_ss", max_rel_l2=5e-3, max_abs=1.5e-1)


@pytest.mark.tpu_test
def test_minimax_qk_norm_rope_pallas_tpu_hardware_parity():
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import _eager_qk_norm_rope

    from vllm_torchtpu.omni.rope import (
        minimax_fused_qk_norm_rope,
        register_minimax_qk_norm_rope_op,
    )

    register_minimax_qk_norm_rope_op()
    dev = torch.device("tpu")
    eps = 1e-6
    t, h = 128, 7
    torch.manual_seed(42)
    q_cpu = torch.randn(t, h, 128, dtype=torch.bfloat16)
    k_cpu = torch.randn(t, h, 128, dtype=torch.bfloat16)
    qw_cpu = torch.randn(128, dtype=torch.bfloat16)
    kw_cpu = torch.randn(128, dtype=torch.bfloat16)
    angles = torch.randn(t, 48, dtype=torch.float32)
    rope_cpu = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).to(
        torch.bfloat16
    )

    want_q, want_k = _eager_qk_norm_rope(
        q_cpu, k_cpu, qw_cpu, kw_cpu, rope_cpu, eps, 128, 96, interleaved=False
    )
    got_q, got_k = minimax_fused_qk_norm_rope(
        q_cpu.to(dev),
        k_cpu.to(dev),
        qw_cpu.to(dev),
        kw_cpu.to(dev),
        rope_cpu.to(dev),
        eps=eps,
        rotary_dim=96,
    )
    torch.accelerator.synchronize()

    assert_parity(
        got_q, want_q, name="tpu_qknorm_rope_q", max_rel_l2=5e-3, max_abs=6.5e-2
    )
    assert_parity(
        got_k, want_k, name="tpu_qknorm_rope_k", max_rel_l2=5e-3, max_abs=6.5e-2
    )
