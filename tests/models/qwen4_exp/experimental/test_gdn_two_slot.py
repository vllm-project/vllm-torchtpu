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
"""Unit tests for 2-slot ping-pong GDN speculative verification runtime."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilderContext,
)
from vllm_torchtpu.models.qwen4_exp.experimental import config as qwen_cfg
from vllm_torchtpu.models.qwen4_exp.experimental.kernels.gdn_v3 import (
    config as gdn_cfg,
)
from vllm_torchtpu.models.qwen4_exp.experimental.kernels.gdn_v3 import (
    tiling as gdn_tiling,
)
from vllm_torchtpu.models.qwen4_exp.experimental.kernels.gdn_v3 import (
    wrapper as gdn_wrapper,
)
from vllm_torchtpu.models.qwen4_exp.experimental.runner import metadata as exp_md
from vllm_torchtpu.models.qwen4_exp.experimental.runner import two_slot

_FAKE_TPU_INFO = SimpleNamespace(
    num_lanes=128,
    vmem_capacity_bytes=32 * 1024 * 1024,
)


def _mock_cfg(
    num_spec: int = 3,
    *,
    additional_config: dict | None = None,
    architecture: str | None = "Qwen4ExpForCausalLM",
    max_num_seqs: int = 32,
    gpu_memory_utilization: float = 0.92,
) -> SimpleNamespace:
    spec = None if num_spec <= 0 else SimpleNamespace(num_speculative_tokens=num_spec)
    return SimpleNamespace(
        speculative_config=spec,
        additional_config=additional_config,
        model_config=SimpleNamespace(architecture=architecture, hf_config=None),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        cache_config=SimpleNamespace(gpu_memory_utilization=gpu_memory_utilization),
    )


def _mamba_spec(
    *,
    num_speculative_blocks: int,
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.GDN_ATTN,
    block_size: int = 512,
) -> MambaSpec:
    return MambaSpec(
        shapes=((3, 1, 128), (2, 128, 128)),
        dtypes=(torch.float32, torch.bfloat16),
        block_size=block_size,
        mamba_type=mamba_type,
        num_speculative_blocks=num_speculative_blocks,
    )


# ---------------------------------------------------------------------------
# Config policy & -50% scratch sizing
# ---------------------------------------------------------------------------


def test_mtp_gdn_two_slot_config_defaults_and_overrides(monkeypatch):
    monkeypatch.setattr(qwen_cfg, "use_unified_pool", lambda: True)
    # Default profile (agentx-mixed-256k) preserves per-token GDN checkpoints.
    assert qwen_cfg.mtp_gdn_two_slot_enabled(None) is False
    assert qwen_cfg.mtp_gdn_two_slot_enabled(_mock_cfg(0)) is False
    assert qwen_cfg.mtp_gdn_two_slot_enabled(_mock_cfg(1)) is False
    assert qwen_cfg.mtp_gdn_two_slot_enabled(_mock_cfg(2)) is False
    assert qwen_cfg.mtp_gdn_two_slot_enabled(_mock_cfg(3)) is False

    assert qwen_cfg.gdn_num_speculative_blocks(_mock_cfg(0)) == 0
    assert qwen_cfg.gdn_num_speculative_blocks(_mock_cfg(1)) == 1
    assert qwen_cfg.gdn_num_speculative_blocks(_mock_cfg(3)) == 3

    # Opt-in via model_plan profile agentx-mixed-256k-two-slot.
    two_slot_prof = {"model_plan": {"profile": "agentx-mixed-256k-two-slot"}}
    # Requesting two-slot without MTP fails loudly instead of silently falling back.
    with pytest.raises(ValueError, match="speculative decoding"):
        qwen_cfg.mtp_gdn_two_slot_enabled(_mock_cfg(0, additional_config=two_slot_prof))
    with pytest.raises(ValueError, match="speculative decoding"):
        qwen_cfg.mtp_gdn_two_slot_enabled(
            _mock_cfg(0, additional_config={"qwen4_exp": {"mtp_gdn_two_slot": True}})
        )
    for k in (1, 2, 3):
        prof_cfg = _mock_cfg(k, additional_config=two_slot_prof)
        assert qwen_cfg.mtp_gdn_two_slot_enabled(prof_cfg) is True
        assert qwen_cfg.gdn_num_speculative_blocks(prof_cfg) == 1

    # Explicit bool override works on top of any profile (disable or K>=1 opt-in).
    off_cfg = _mock_cfg(
        3,
        additional_config={
            **two_slot_prof,
            "qwen4_exp": {"mtp_gdn_two_slot": False},
        },
    )
    assert qwen_cfg.mtp_gdn_two_slot_enabled(off_cfg) is False
    assert qwen_cfg.gdn_num_speculative_blocks(off_cfg) == 3
    # Explicit False override also suppresses the profile even when MTP is off.
    off_no_mtp = _mock_cfg(
        0,
        additional_config={
            **two_slot_prof,
            "qwen4_exp": {"mtp_gdn_two_slot": False},
        },
    )
    assert qwen_cfg.mtp_gdn_two_slot_enabled(off_no_mtp) is False

    on_k1 = _mock_cfg(1, additional_config={"qwen4_exp": {"mtp_gdn_two_slot": True}})
    assert qwen_cfg.mtp_gdn_two_slot_enabled(on_k1) is True
    assert qwen_cfg.gdn_num_speculative_blocks(on_k1) == 1

    # Strict bool validation rejects non-bool values even when MTP is off.
    for num_spec in (0, 3):
        for bad in (1, 0, "true", None):
            bad_cfg = _mock_cfg(
                num_spec, additional_config={"qwen4_exp": {"mtp_gdn_two_slot": bad}}
            )
            with pytest.raises(ValueError, match="must be bool"):
                qwen_cfg.mtp_gdn_two_slot_enabled(bad_cfg)

    # Requesting two-slot when unified pool is off fails loudly.
    monkeypatch.setattr(qwen_cfg, "use_unified_pool", lambda: False)
    with pytest.raises(ValueError, match="unified block pool"):
        qwen_cfg.mtp_gdn_two_slot_enabled(_mock_cfg(3, additional_config=two_slot_prof))
    with pytest.raises(ValueError, match="unified block pool"):
        qwen_cfg.mtp_gdn_two_slot_enabled(
            _mock_cfg(3, additional_config={"qwen4_exp": {"mtp_gdn_two_slot": True}})
        )
    # Default profile with unified pool off stays False (no error).
    assert qwen_cfg.mtp_gdn_two_slot_enabled(_mock_cfg(3)) is False


def test_pooled_cache_dummy_pages_halves_gdn_spec_blocks(monkeypatch):
    """Real Qwen4ExpPooledCache dummy_pages & dummy_metadata for 36 GDN + 12 PLE."""
    import sys
    import types

    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

    try:
        from vllm_torchtpu.models.qwen4_exp.experimental.pooled_cache import (
            Qwen4ExpPooledCache,
        )
    except ModuleNotFoundError:
        kv_stub = types.ModuleType("vllm.v1.core.kv_cache_planning")
        kv_stub.get_kv_cache_groups = lambda *a, **kw: None  # type: ignore[attr-defined]
        kv_stub.get_kv_cache_config_from_groups = lambda *a, **kw: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_planning", kv_stub)
        from vllm_torchtpu.models.qwen4_exp.experimental.pooled_cache import (
            Qwen4ExpPooledCache,
        )

    active_rows = 16
    rows = 16
    seq_lens = [128] * active_rows

    def _make_cache(gdn_spec_blocks: int) -> Qwen4ExpPooledCache:
        groups = [
            SimpleNamespace(
                kv_cache_spec=_mamba_spec(num_speculative_blocks=gdn_spec_blocks),
                layer_names=[f"model.layers.{i}.linear_attn"],
            )
            for i in range(36)
        ] + [
            SimpleNamespace(
                kv_cache_spec=_mamba_spec(
                    num_speculative_blocks=3,
                    mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
                ),
                layer_names=[f"model.layers.{36 + i}.ple"],
            )
            for i in range(12)
        ]
        cache = object.__new__(Qwen4ExpPooledCache)
        cache.runner = SimpleNamespace(
            vllm_config=_mock_cfg(3),
            max_model_len=9216,
            device=torch.device("cpu"),
        )
        cache.groups = groups
        cache.pages = 0
        return cache

    per_token_cache = _make_cache(3)
    two_slot_cache = _make_cache(1)
    full_4slot = per_token_cache.dummy_pages(seq_lens, active_rows)
    two_slot_pages = two_slot_cache.dummy_pages(seq_lens, active_rows)
    assert full_4slot - two_slot_pages == 36 * 2 * active_rows  # 1,152 blocks
    assert two_slot_pages - 1 == active_rows * (36 * 2 + 12 * 4)

    # Verify dummy_metadata produces 2-col GDN and 4-col PLE checkpoint tables
    # and consumes exactly dummy_pages() pages.
    base_md = AttentionMetadata(
        input_positions=torch.arange(rows, dtype=torch.int32),
        query_start_loc=torch.arange(rows + 1, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        request_distribution=torch.tensor([rows, rows, rows], dtype=torch.int32),
    )
    two_slot_cache.pages = two_slot_pages
    md = two_slot_cache.dummy_metadata(base_md, active_rows=active_rows)
    for i in range(36):
        gdn_md = md[f"model.layers.{i}.linear_attn"]
        assert gdn_md.mamba_ckpt_indices is not None
        assert tuple(gdn_md.mamba_ckpt_indices.shape) == (rows, 2)
    for i in range(12):
        ple_md = md[f"model.layers.{36 + i}.ple"]
        assert ple_md.mamba_ckpt_indices is not None
        assert tuple(ple_md.mamba_ckpt_indices.shape) == (rows, 4)

    # One fewer page trips the page-budget assert, confirming dummy_pages and
    # dummy_metadata agree tightly.
    two_slot_cache.pages = two_slot_pages - 1
    with pytest.raises(AssertionError, match="dummy owner pages exceed"):
        two_slot_cache.dummy_metadata(base_md, active_rows=active_rows)


# ---------------------------------------------------------------------------
# gdn_v3 tiling, config, and race-detected Pallas interpret tests
# ---------------------------------------------------------------------------


def test_gdn_config_and_tiling_single_checkpoint():
    with mock.patch.object(pltpu, "get_tpu_info", return_value=_FAKE_TPU_INFO):
        c_full = gdn_cfg.GDNConfig(
            mode=gdn_cfg.GDNMode.BATCHED,
            dtypes=gdn_cfg.Dtypes(
                jnp.float32, jnp.bfloat16, jnp.float32, jnp.float32, jnp.float32
            ),
            batch_size=16,
            dim_size=512,
            kernel_size=4,
            tile_size=4,
            num_kq_heads=1,
            num_v_heads=3,
            kq_head_dim=128,
            v_head_dim=128,
            window_size=4,
            single_checkpoint=False,
        )
        c_s1 = gdn_cfg.GDNConfig(
            mode=gdn_cfg.GDNMode.BATCHED,
            dtypes=c_full.dtypes,
            batch_size=16,
            dim_size=512,
            kernel_size=4,
            tile_size=4,
            num_kq_heads=1,
            num_v_heads=3,
            kq_head_dim=128,
            v_head_dim=128,
            window_size=4,
            single_checkpoint=True,
        )
        assert c_full.num_state_checkpoints == 4
        assert c_s1.num_state_checkpoints == 1
        assert "_w4_s1" in c_s1.get_kernel_name()
        assert "_s1" not in c_full.get_kernel_name()
        assert gdn_cfg.GDNConfig.get_vmem_limit_bytes(4, single_checkpoint=True) < (
            gdn_cfg.GDNConfig.get_vmem_limit_bytes(4, single_checkpoint=False)
        )

        # VMEM estimate with single_checkpoint uses 2 slots (1 in + 1 out) instead of 5.
        est_full = gdn_tiling.get_vmem_estimate_bytes(
            1, 4, 4, 12, 128, 128, 4, 2, 2, 128, 2560, is_decode=True, window_size=4
        )
        est_s1 = gdn_tiling.get_vmem_estimate_bytes(
            1,
            4,
            4,
            12,
            128,
            128,
            4,
            2,
            2,
            128,
            2560,
            is_decode=True,
            window_size=4,
            single_checkpoint=True,
        )
        assert est_s1 < est_full

        # Default path (single_checkpoint=False) preserves T^2 * n_v <= 16 * num_seqs
        # unchanged: n_v=12, num_seqs=16, bf16 stays T=4, whereas single_checkpoint=True
        # uses eff_n_v=8 and 2x bf16 scale -> T=8.
        assert (
            gdn_tiling.decode_tile_target(
                16, 12, window_size=1, rec_state_bytes=2, single_checkpoint=False
            )
            == 4
        )
        t_fp32 = gdn_tiling.decode_tile_target(
            16, 12, window_size=1, rec_state_bytes=4, single_checkpoint=True
        )
        t_bf16 = gdn_tiling.decode_tile_target(
            16, 12, window_size=1, rec_state_bytes=2, single_checkpoint=True
        )
        assert t_bf16 == 8 and t_bf16 > t_fp32 >= 1

        # Explicit tile clamp against num_seqs is gated to single_checkpoint=True;
        # default path preserves the padded_batch_size clamp.
        dt_s1, mt_s1 = gdn_tiling.get_tile_sizes(
            batch_size=32,
            num_seqs=3,
            padded_batch_size=32,
            n_kq=4,
            n_v=12,
            d_k=128,
            d_v=128,
            kernel_size=4,
            conv_state_dim_size=2560,
            act_out_dtype=jnp.bfloat16,
            recurrent_state_dtype=jnp.bfloat16,
            num_lanes=128,
            decode_vmem_limit_bytes=int(0.8 * 32 * 1024 * 1024),
            mixed_vmem_limit_bytes=int(0.8 * 32 * 1024 * 1024),
            window_size=4,
            decode_tile_size=16,
            single_checkpoint=True,
        )
        assert dt_s1 <= 3
        assert mt_s1 >= 1

        dt_default, _ = gdn_tiling.get_tile_sizes(
            batch_size=32,
            num_seqs=3,
            padded_batch_size=32,
            n_kq=4,
            n_v=12,
            d_k=128,
            d_v=128,
            kernel_size=4,
            conv_state_dim_size=2560,
            act_out_dtype=jnp.bfloat16,
            recurrent_state_dtype=jnp.bfloat16,
            num_lanes=128,
            decode_vmem_limit_bytes=int(0.8 * 32 * 1024 * 1024),
            mixed_vmem_limit_bytes=int(0.8 * 32 * 1024 * 1024),
            window_size=4,
            decode_tile_size=16,
            single_checkpoint=False,
        )
        assert dt_default == 16


def _run_gdn_dense_race_checked(
    n_kq: int, n_v: int, num_seqs: int, mode: gdn_cfg.AttentionMode
):
    """Compare single_checkpoint 2-slot ping-pong against full 4-slot reference."""
    d_k = d_v = 128
    k_sz = 4
    dim = 2 * n_kq * d_k + n_v * d_v
    q_lens = [4] * num_seqs
    batch = sum(q_lens)
    qsl = jnp.array([4 * i for i in range(num_seqs + 1)], dtype=jnp.int32)
    dist = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    slens = jnp.array([10 + i for i in range(num_seqs)], dtype=jnp.int32)
    is_kda = mode == gdn_cfg.AttentionMode.KDA

    k0, k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.PRNGKey(42), 7)
    qkv = jax.random.normal(k0, (batch, dim), dtype=jnp.float32) * 0.08
    b = jax.random.normal(k1, (batch, n_v), dtype=jnp.float32) * 0.1
    a_shape = (batch, n_v * d_k) if is_kda else (batch, n_v)
    a = jax.random.normal(k2, a_shape, dtype=jnp.float32) * 0.1
    cw = jax.random.normal(k3, (dim, 1, k_sz), dtype=jnp.float32) * 0.05
    cb = jax.random.normal(k4, (dim,), dtype=jnp.float32) * 0.02
    a_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v, d_k) if is_kda else (n_v,), dtype=jnp.float32)

    num_slots = num_seqs * 4 + 2
    seed_conv = (
        jax.random.normal(k5, (num_seqs, k_sz - 1, dim), dtype=jnp.float32) * 0.05
    )
    seed_rec = (
        jax.random.normal(k6, (num_seqs, n_v, d_k, d_v), dtype=jnp.float32) * 0.05
    )

    conv_ref = jnp.zeros((num_slots, k_sz - 1, dim), dtype=jnp.float32)
    rec_ref = jnp.zeros((num_slots, n_v, d_k, d_v), dtype=jnp.float32)
    ckpt_4_rows = []
    ref_s_idx = []
    for i in range(num_seqs):
        base = 1 + 4 * i
        ref_s_idx.append(base)
        ckpt_4_rows.append([base, base + 1, base + 2, base + 3])
        conv_ref = conv_ref.at[base].set(seed_conv[i])
        rec_ref = rec_ref.at[base].set(seed_rec[i])
    ckpt_4 = jnp.array(ckpt_4_rows, dtype=jnp.int32)

    ro_vals = [i % 2 for i in range(num_seqs)]
    ckpt_2_rows = [[1 + 2 * i, 2 + 2 * i] for i in range(num_seqs)]
    conv_2s = jnp.zeros((num_slots, k_sz - 1, dim), dtype=jnp.float32)
    rec_2s = jnp.zeros((num_slots, n_v, d_k, d_v), dtype=jnp.float32)
    for i in range(num_seqs):
        r_slot = ckpt_2_rows[i][ro_vals[i]]
        conv_2s = conv_2s.at[r_slot].set(seed_conv[i])
        rec_2s = rec_2s.at[r_slot].set(seed_rec[i])

    # Cover partial (1, 2, 3), skipped (0), and full (4) commits.
    commit_vals = [2, 0, 4, 1, 3][:num_seqs]
    interp = pltpu.InterpretParams(detect_races=True)
    with mock.patch.object(pltpu, "get_tpu_info", return_value=_FAKE_TPU_INFO):
        (c_ref, r_ref), out_ref = gdn_wrapper.fused_conv1d_gdn(
            qkv,
            b,
            a,
            conv_ref,
            rec_ref,
            cw,
            cb,
            a_log,
            dt_bias,
            qsl,
            jnp.array(ref_s_idx, dtype=jnp.int32),
            dist,
            slens,
            read_offsets=jnp.zeros((num_seqs,), dtype=jnp.int32),
            ckpt_indices=ckpt_4,
            attention_mode=mode,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=k_sz,
            num_spec_tokens=3,
            batched_only=True,
            interpret=interp,
        )
        (c_out, r_out), out_2s = gdn_wrapper.fused_conv1d_gdn(
            qkv,
            b,
            a,
            conv_2s,
            rec_2s,
            cw,
            cb,
            a_log,
            dt_bias,
            qsl,
            jnp.array([r[0] for r in ckpt_2_rows], dtype=jnp.int32),
            dist,
            slens,
            read_offsets=jnp.array(ro_vals, dtype=jnp.int32),
            ckpt_indices=jnp.array(ckpt_2_rows, dtype=jnp.int32),
            commit_lens=jnp.array(commit_vals, dtype=jnp.int32),
            attention_mode=mode,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=k_sz,
            num_spec_tokens=3,
            single_checkpoint=True,
            batched_only=True,
            interpret=interp,
        )

    assert float(jnp.max(jnp.abs(out_2s - out_ref))) == 0.0
    for i in range(num_seqs):
        r_slot = ckpt_2_rows[i][ro_vals[i]]
        w_slot = ckpt_2_rows[i][1 - ro_vals[i]]
        # Read slot stays untouched after single_checkpoint verify/replay.
        assert float(jnp.max(jnp.abs(c_out[r_slot] - seed_conv[i]))) == 0.0
        assert float(jnp.max(jnp.abs(r_out[r_slot] - seed_rec[i]))) == 0.0
        cl = commit_vals[i]
        if cl == 0:
            # commit_lens == 0 skips HBM write to the write slot.
            assert float(jnp.max(jnp.abs(c_out[w_slot]))) == 0.0
            assert float(jnp.max(jnp.abs(r_out[w_slot]))) == 0.0
        else:
            ref_slot = ckpt_4_rows[i][cl - 1]
            assert float(jnp.max(jnp.abs(c_out[w_slot] - c_ref[ref_slot]))) == 0.0
            assert float(jnp.max(jnp.abs(r_out[w_slot] - r_ref[ref_slot]))) == 0.0


def test_gdn_single_checkpoint_ping_pong_odd_batch_detect_races():
    # Non-power-of-two heads (n_v=3 ~ 48v ratio) and odd num_seqs=5 to exercise
    # partial-tail SMEM padding and zero-DMA guards under detect_races=True.
    _run_gdn_dense_race_checked(1, 3, 5, gdn_cfg.AttentionMode.GDN)


def test_kda_single_checkpoint_ping_pong_detect_races():
    _run_gdn_dense_race_checked(1, 1, 3, gdn_cfg.AttentionMode.KDA)


def test_gdn_single_checkpoint_mixed_verify_and_prefill_detect_races():
    """Mixed [BATCHED verify + PER_SEQ prefill] with 2-col ckpt_indices."""
    n_kq, n_v, d_k, d_v, k_sz = 1, 2, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    # 2 verify seqs (len 4 each) + 1 prefill seq (len 8) = 16 tokens.
    qsl = jnp.array([0, 4, 8, 16], dtype=jnp.int32)
    dist = jnp.array([2, 2, 3], dtype=jnp.int32)
    slens = jnp.array([12, 20, 8], dtype=jnp.int32)
    batch = 16
    k0, k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 4)
    qkv = jax.random.normal(k0, (batch, dim), dtype=jnp.float32) * 0.05
    b = jax.random.normal(k1, (batch, n_v), dtype=jnp.float32) * 0.05
    a = jax.random.normal(k2, (batch, n_v), dtype=jnp.float32) * 0.05
    cw = jax.random.normal(k3, (dim, 1, k_sz), dtype=jnp.float32) * 0.05
    a_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)

    conv_ref = jnp.zeros((16, k_sz - 1, dim), dtype=jnp.float32)
    rec_ref = jnp.zeros((16, n_v, d_k, d_v), dtype=jnp.float32)
    ckpt_4 = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=jnp.int32)
    ckpt_2 = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.int32)
    # Exercise both BATCHED (2 verify windows) and PER_SEQ (1 prefill seq) with
    # `detect_races=True`. Force mixed_tile_size=1 so JAX 0.12's CPU Mosaic
    # interpreter runs without hitting its ReshapeTransform limitation in
    # `load_strided_heads` (`ref.reshape`).
    interp = pltpu.InterpretParams(detect_races=True)
    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=_FAKE_TPU_INFO),
        mock.patch.object(gdn_tiling, "get_tile_sizes", return_value=(2, 1)),
    ):
        (c_ref, r_ref), out_ref = gdn_wrapper.fused_conv1d_gdn(
            qkv,
            b,
            a,
            conv_ref,
            rec_ref,
            cw,
            None,
            a_log,
            dt_bias,
            qsl,
            jnp.array([1, 5, 9], jnp.int32),
            dist,
            slens,
            read_offsets=jnp.array([0, 0, 0], jnp.int32),
            ckpt_indices=ckpt_4,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=k_sz,
            num_spec_tokens=3,
            interpret=interp,
        )
        (c_2s, r_2s), out_2s = gdn_wrapper.fused_conv1d_gdn(
            qkv,
            b,
            a,
            jnp.zeros((16, k_sz - 1, dim), dtype=jnp.float32),
            jnp.zeros((16, n_v, d_k, d_v), dtype=jnp.float32),
            cw,
            None,
            a_log,
            dt_bias,
            qsl,
            jnp.array([1, 3, 5], jnp.int32),
            dist,
            slens,
            read_offsets=jnp.array([0, 1, 0], jnp.int32),
            ckpt_indices=ckpt_2,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=k_sz,
            num_spec_tokens=3,
            single_checkpoint=True,
            interpret=interp,
        )
    assert float(jnp.max(jnp.abs(out_2s - out_ref))) == 0.0
    # Verify seq 0 wrote to col 1 (slot 2), seq 1 wrote to col 0 (slot 3).
    assert float(jnp.max(jnp.abs(r_2s[2] - r_ref[4]))) == 0.0
    assert float(jnp.max(jnp.abs(r_2s[3] - r_ref[8]))) == 0.0
    # Prefill seq 2 wrote final state to col 0 (slot 5) vs ref slot 9.
    assert float(jnp.max(jnp.abs(r_2s[5] - r_ref[9]))) == 0.0
    assert float(jnp.max(jnp.abs(c_2s[5] - c_ref[9]))) == 0.0


def test_gdn_single_checkpoint_validation_errors():
    n_kq, n_v, d_k, d_v, k_sz = 1, 2, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    qkv = jnp.zeros((4, dim), jnp.float32)
    b = jnp.zeros((4, n_v), jnp.float32)
    a = jnp.zeros((4, n_v), jnp.float32)
    cs = jnp.zeros((8, k_sz - 1, dim), jnp.float32)
    rs = jnp.zeros((8, n_v, d_k, d_v), jnp.float32)
    cw = jnp.zeros((dim, 1, k_sz), jnp.float32)
    al = jnp.zeros((n_v,), jnp.float32)
    db = jnp.zeros((n_v,), jnp.float32)
    qsl = jnp.array([0, 4], jnp.int32)
    sid = jnp.array([1], jnp.int32)
    dist = jnp.array([1, 1, 1], jnp.int32)
    sl = jnp.array([4], jnp.int32)
    ro = jnp.array([0], jnp.int32)

    with mock.patch.object(pltpu, "get_tpu_info", return_value=_FAKE_TPU_INFO):
        with pytest.raises(ValueError, match="commit_lens requires single_checkpoint"):
            gdn_wrapper.fused_conv1d_gdn(
                qkv,
                b,
                a,
                cs,
                rs,
                cw,
                None,
                al,
                db,
                qsl,
                sid,
                dist,
                sl,
                read_offsets=ro,
                commit_lens=jnp.array([1], jnp.int32),
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=k_sz,
                num_spec_tokens=3,
                single_checkpoint=False,
            )
        with pytest.raises(ValueError, match="prefill_only"):
            gdn_wrapper.fused_conv1d_gdn(
                qkv,
                b,
                a,
                cs,
                rs,
                cw,
                None,
                al,
                db,
                qsl,
                sid,
                dist,
                sl,
                read_offsets=ro,
                commit_lens=jnp.array([1], jnp.int32),
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=k_sz,
                num_spec_tokens=3,
                single_checkpoint=True,
                prefill_only=True,
            )
        with pytest.raises(ValueError, match="ckpt_indices must have shape"):
            gdn_wrapper.fused_conv1d_gdn(
                qkv,
                b,
                a,
                cs,
                rs,
                cw,
                None,
                al,
                db,
                qsl,
                sid,
                dist,
                sl,
                read_offsets=ro,
                ckpt_indices=jnp.zeros((1, 3), jnp.int32),
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=k_sz,
                num_spec_tokens=3,
                single_checkpoint=True,
            )


def test_gdn_pooled_external_state_source_single_checkpoint_detect_races():
    """Verify ExternalStateIn/OutBufferedRef with 2-col ping-pong & commit_lens=0."""
    from vllm_torchtpu.models.qwen4_exp.experimental import gdn_attention as exp_ga

    n_kq, n_v, d_k, d_v, k_sz = 1, 1, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v  # 384
    # Build a synthetic pool layout large enough for 1 block per slot (stride 1).
    # Recurrent: 1 * 128 * 128 * 4B = 64 KB; Conv: 3 * 384 * 4B = 4.5 KB.
    # Use pool shape (10, 128, 2, 128) uint16 / bfloat16? Use float32 directly
    # with explicit StateSourcePlan matching the block shape:
    # Block payload: (200, 1, 128) float32 -> 200 rows of 128 lanes.
    # rec rows_used = 128 (0..128), conv rows_used = 3 * 3 = 9 (128..137).
    pool_shape = (10, 144, 128)
    pool = jnp.zeros(pool_shape, dtype=jnp.float32)
    plan = gdn_cfg.StateSourcePlan(
        stride=1,
        recurrent=gdn_cfg.StateRegion(
            kb0=0,
            nblocks=1,
            row0=0,
            nrows=128,
            view_dtype=jnp.float32,
            lane_split=1,
            rows_used=128,
        ),
        conv=gdn_cfg.StateRegion(
            kb0=0,
            nblocks=1,
            row0=128,
            nrows=16,
            view_dtype=jnp.float32,
            lane_split=1,
            rows_used=9,
        ),
        whole_block_dma=False,
    )
    del exp_ga  # importability sanity check
    qkv = jax.random.normal(jax.random.PRNGKey(10), (8, dim), jnp.float32) * 0.1
    b = jax.random.normal(jax.random.PRNGKey(11), (8, n_v), jnp.float32) * 0.1
    a = jax.random.normal(jax.random.PRNGKey(12), (8, n_v), jnp.float32) * 0.1
    cw = jax.random.normal(jax.random.PRNGKey(13), (dim, 1, k_sz), jnp.float32) * 0.1
    al = jnp.zeros((n_v,), jnp.float32)
    db = jnp.zeros((n_v,), jnp.float32)
    qsl = jnp.array([0, 4, 8], jnp.int32)
    dist = jnp.array([2, 2, 2], jnp.int32)
    slens = jnp.array([8, 8], jnp.int32)
    # Seed seq 0 read slot 1 (ro=0), seq 1 read slot 4 (ro=1) — both recurrent
    # and conv regions so parity checks validate both state regions.
    seed0 = jax.random.normal(jax.random.PRNGKey(14), (128, 128), jnp.float32) * 0.1
    seed1 = jax.random.normal(jax.random.PRNGKey(15), (128, 128), jnp.float32) * 0.1
    conv_seed0 = jax.random.normal(jax.random.PRNGKey(16), (3, dim), jnp.float32) * 0.1
    conv_seed1 = jax.random.normal(jax.random.PRNGKey(17), (3, dim), jnp.float32) * 0.1
    pool = (
        pool.at[1, :128]
        .set(seed0)
        .at[1, 128:137]
        .set(conv_seed0.reshape(9, 128))
        .at[4, :128]
        .set(seed1)
        .at[4, 128:137]
        .set(conv_seed1.reshape(9, 128))
    )
    ckpt_2 = jnp.array([[1, 2], [3, 4]], jnp.int32)
    ro = jnp.array([0, 1], jnp.int32)

    # Dense 4-slot reference with identical initial conv & recurrent states.
    conv_ref = jnp.zeros((10, k_sz - 1, dim), jnp.float32)
    rec_ref = jnp.zeros((10, n_v, d_k, d_v), jnp.float32)
    conv_ref = conv_ref.at[1].set(conv_seed0).at[5].set(conv_seed1)
    rec_ref = rec_ref.at[1, 0].set(seed0).at[5, 0].set(seed1)
    ckpt_4 = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], jnp.int32)

    from vllm_torchtpu.kernels import typed_ldst

    def _cpu_load_typed(ref, *, view_dtype, lane_split=1):
        del lane_split
        return ref[...].astype(view_dtype)

    def _cpu_store_typed(ref, arr, *, lane_split=1):
        del lane_split
        ref[...] = arr.astype(ref.dtype)

    interp = pltpu.InterpretParams(detect_races=True)
    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=_FAKE_TPU_INFO),
        mock.patch.object(typed_ldst, "load_typed", side_effect=_cpu_load_typed),
        mock.patch.object(typed_ldst, "store_typed", side_effect=_cpu_store_typed),
    ):
        (c_ref, r_ref), out_ref = gdn_wrapper.fused_conv1d_gdn(
            qkv,
            b,
            a,
            conv_ref,
            rec_ref,
            cw,
            None,
            al,
            db,
            qsl,
            jnp.array([1, 5], jnp.int32),
            dist,
            slens,
            read_offsets=jnp.zeros((2,), jnp.int32),
            ckpt_indices=ckpt_4,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=k_sz,
            num_spec_tokens=3,
            single_checkpoint=False,
            batched_only=True,
            interpret=interp,
        )
        out_pool, out_act = gdn_wrapper.fused_conv1d_gdn(
            qkv,
            b,
            a,
            None,
            None,
            cw,
            None,
            al,
            db,
            qsl,
            jnp.array([1, 3], jnp.int32),
            dist,
            slens,
            read_offsets=ro,
            ckpt_indices=ckpt_2,
            commit_lens=jnp.array([2, 0], jnp.int32),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=k_sz,
            state_source=pool,
            state_plan=plan,
            num_spec_tokens=3,
            single_checkpoint=True,
            batched_only=True,
            interpret=interp,
        )
    assert out_act.shape == (8, n_v * d_v)
    assert float(jnp.max(jnp.abs(out_act - out_ref))) < 1e-6
    # seq 0 (commit_lens=2): read slot 1 preserved; write slot 2 matches exact
    # reference checkpoint index (commit_lens - 1 = 1 -> slot 2) for both
    # recurrent and conv regions, and differs from the full 4-token state.
    assert float(jnp.max(jnp.abs(out_pool[1, :128] - seed0))) == 0.0
    assert (
        float(jnp.max(jnp.abs(out_pool[1, 128:137] - conv_seed0.reshape(9, 128))))
        == 0.0
    )
    assert float(jnp.max(jnp.abs(out_pool[2, :128] - r_ref[2, 0]))) < 1e-6
    assert float(jnp.max(jnp.abs(out_pool[2, :128] - r_ref[4, 0]))) > 1e-3
    assert (
        float(jnp.max(jnp.abs(out_pool[2, 128:137] - c_ref[2].reshape(9, 128)))) < 1e-6
    )
    assert (
        float(jnp.max(jnp.abs(out_pool[2, 128:137] - c_ref[4].reshape(9, 128)))) > 1e-3
    )
    # seq 1 commit_lens=0 -> read slot 4 preserved, write slot 3 untouched (0).
    assert float(jnp.max(jnp.abs(out_pool[4, :128] - seed1))) == 0.0
    assert (
        float(jnp.max(jnp.abs(out_pool[4, 128:137] - conv_seed1.reshape(9, 128))))
        == 0.0
    )
    assert float(jnp.max(jnp.abs(out_pool[3, :128]))) == 0.0
    assert float(jnp.max(jnp.abs(out_pool[3, 128:137]))) == 0.0


@pytest.mark.parametrize("with_prefill_ckpt", [False, True])
def test_run_jax_gdn_attention_pooled_commit_lens_and_read_offset_indices(
    monkeypatch, with_prefill_ckpt
):
    """Test shard_map pop order and block-major read_offset_indices gather."""
    import numpy as np

    from vllm_torchtpu.models.qwen4_exp.experimental import gdn_attention as exp_ga

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    captured: dict[str, object] = {}

    def _fake_local(
        j_mixed_qkv,
        j_b,
        j_a,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        read_offsets=None,
        ckpt_indices=None,
        per_seq_inputs=None,
        *,
        commit_lens=None,
        prefill_ckpt=None,
        **kwargs,
    ):
        del (
            j_b,
            j_a,
            j_conv_weight,
            j_conv_bias,
            j_A_log,
            j_dt_bias,
            query_start_loc,
            distribution,
            seq_lens,
            per_seq_inputs,
        )
        # Encode read_offsets, commit_lens, and optional prefill_ckpt into
        # output rows so values survive shard_map tracing.
        captured["single_checkpoint"] = kwargs["single_checkpoint"]
        captured["batched_only"] = kwargs["batched_only"]
        captured["has_commit_lens"] = commit_lens is not None
        captured["has_prefill_ckpt"] = prefill_ckpt is not None
        captured["ckpt_shape"] = None if ckpt_indices is None else ckpt_indices.shape
        out = jnp.zeros((j_mixed_qkv.shape[0], kwargs["n_v"] * kwargs["d_v"]))
        out = out.at[0, : read_offsets.shape[0]].set(read_offsets.astype(jnp.float32))
        if commit_lens is not None:
            out = out.at[1, : commit_lens.shape[0]].set(commit_lens.astype(jnp.float32))
        if prefill_ckpt is not None:
            out = out.at[2, : prefill_ckpt[0].shape[0]].set(
                prefill_ckpt[0].astype(jnp.float32)
            )
            out = out.at[3, : prefill_ckpt[1].shape[0]].set(
                prefill_ckpt[1].astype(jnp.float32)
            )
        return recurrent_state, out

    monkeypatch.setattr(exp_ga, "run_jax_gdn_attention_pooled_local", _fake_local)

    n_kq, n_v, d_k, d_v, k_sz = 1, 2, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    # Simulate block-major merged pool (num_pools=4, pool_index=2):
    # manager block IDs (read_offset_indices) = [1, 2] -> physical IDs = [6, 10].
    # Distinct values at [1, 2] vs [6, 10] make the old state_indices gather fail.
    slot_read_offsets = jnp.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=jnp.int32)
    raw_ids = jnp.array([1, 2], dtype=jnp.int32)
    phys_ids = raw_ids * 4 + 2  # [6, 10]
    ckpt_ids = jnp.array([[6, 14], [10, 18]], dtype=jnp.int32)
    commit_lens = jnp.array([2, 0], dtype=jnp.int32)
    prefill_ckpt = (
        (
            jnp.array([3, 7], dtype=jnp.int32),
            jnp.array([11, 15], dtype=jnp.int32),
        )
        if with_prefill_ckpt
        else None
    )

    _, out = exp_ga.run_jax_gdn_attention_pooled(
        jnp.zeros((8, dim), jnp.float32),
        jnp.zeros((8, n_v), jnp.float32),
        jnp.zeros((8, n_v), jnp.float32),
        jnp.zeros((16, 8, 8), jnp.float32),
        jnp.zeros((dim, 1, k_sz), jnp.float32),
        None,
        jnp.zeros((n_v,), jnp.float32),
        jnp.zeros((n_v,), jnp.float32),
        phys_ids,
        jnp.array([0, 4, 8], jnp.int32),
        jnp.array([2, 2, 2], jnp.int32),
        jnp.array([10, 12], jnp.int32),
        slot_read_offsets,
        ckpt_ids,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=k_sz,
        pool_block_tokens=128,
        mesh=mesh,
        num_spec_tokens=3,
        single_checkpoint=True,
        batched_only=True,
        commit_lens=commit_lens,
        read_offset_indices=raw_ids,
        prefill_ckpt=prefill_ckpt,
    )
    assert captured["single_checkpoint"] is True
    assert captured["batched_only"] is True
    assert captured["has_commit_lens"] is True
    assert captured["has_prefill_ckpt"] is with_prefill_ckpt
    assert captured["ckpt_shape"] == (2, 2)
    # Gathered from read_offset_indices=[1, 2] -> [1, 0], NOT phys_ids=[6, 10].
    assert out[0, :2].tolist() == [1.0, 0.0]
    assert out[1, :2].tolist() == [2.0, 0.0]
    if with_prefill_ckpt:
        assert out[2, :2].tolist() == [3.0, 7.0]
        assert out[3, :2].tolist() == [11.0, 15.0]


def test_wrapper_pooled_tiler_dtype_gated_to_single_checkpoint():
    """Default pooled path keeps fp32 tiler dtype; single_checkpoint uses bf16."""
    n_kq, n_v, d_k, d_v, k_sz = 1, 2, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    pool = jnp.zeros((6, 144, 128), dtype=jnp.uint32)
    plan = gdn_cfg.StateSourcePlan(
        stride=1,
        recurrent=gdn_cfg.StateRegion(
            kb0=0,
            nblocks=1,
            row0=0,
            nrows=128,
            view_dtype=jnp.uint32,
            lane_split=1,
            rows_used=128,
        ),
        conv=gdn_cfg.StateRegion(
            kb0=0,
            nblocks=1,
            row0=128,
            nrows=16,
            view_dtype=jnp.float32,
            lane_split=1,
            rows_used=9,
        ),
        whole_block_dma=False,
    )
    seen_dtypes: list[jnp.dtype] = []

    class _StopAfterTiling(Exception):
        pass

    def _spy_get_tile_sizes(**kwargs):
        seen_dtypes.append(kwargs["recurrent_state_dtype"])
        raise _StopAfterTiling()

    for s1, expected_dtype in ((False, jnp.float32), (True, jnp.bfloat16)):
        seen_dtypes.clear()
        with (
            mock.patch.object(pltpu, "get_tpu_info", return_value=_FAKE_TPU_INFO),
            mock.patch.object(
                gdn_tiling, "get_tile_sizes", side_effect=_spy_get_tile_sizes
            ),
            pytest.raises(_StopAfterTiling),
        ):
            gdn_wrapper.fused_conv1d_gdn(
                jnp.zeros((4, dim), jnp.float32),
                jnp.zeros((4, n_v), jnp.float32),
                jnp.zeros((4, n_v), jnp.float32),
                None,
                None,
                jnp.zeros((dim, 1, k_sz), jnp.float32),
                None,
                jnp.zeros((n_v,), jnp.float32),
                jnp.zeros((n_v,), jnp.float32),
                jnp.array([0, 4], jnp.int32),
                jnp.array([1], jnp.int32),
                jnp.array([1, 1, 1], jnp.int32),
                jnp.array([4], jnp.int32),
                read_offsets=jnp.array([0], jnp.int32),
                ckpt_indices=jnp.zeros((1, 2 if s1 else 4), jnp.int32),
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=k_sz,
                state_source=pool,
                state_plan=plan,
                num_spec_tokens=3,
                single_checkpoint=s1,
                batched_only=True,
            )
        assert seen_dtypes == [expected_dtype]


# ---------------------------------------------------------------------------
# Heterogeneous 2-slot GDN / 4-slot PLE metadata partitioning
# ---------------------------------------------------------------------------


def test_is_ping_pong_mamba_spec_k_ge_2_and_k1_opt_in(monkeypatch):
    monkeypatch.setattr(qwen_cfg, "use_unified_pool", lambda: True)
    gdn_2slot = _mamba_spec(num_speculative_blocks=1)
    ple_4slot = _mamba_spec(
        num_speculative_blocks=3, mamba_type=MambaAttentionBackendEnum.SHORT_CONV
    )
    ple_1slot = _mamba_spec(
        num_speculative_blocks=1, mamba_type=MambaAttentionBackendEnum.SHORT_CONV
    )
    cfg_k3 = _mock_cfg(3)
    assert exp_md.is_ping_pong_mamba_spec(gdn_2slot, cfg_k3) is True
    assert exp_md.is_ping_pong_mamba_spec(ple_4slot, cfg_k3) is False

    # K=1 default off -> False; K=1 opt-in -> True only for GDN_ATTN.
    cfg_k1_off = _mock_cfg(1)
    assert exp_md.is_ping_pong_mamba_spec(gdn_2slot, cfg_k1_off) is False
    for opt_in in (
        {"model_plan": {"profile": "agentx-mixed-256k-two-slot"}},
        {"qwen4_exp": {"mtp_gdn_two_slot": True}},
    ):
        cfg_k1_on = _mock_cfg(1, additional_config=opt_in)
        assert exp_md.is_ping_pong_mamba_spec(gdn_2slot, cfg_k1_on) is True
        assert exp_md.is_ping_pong_mamba_spec(ple_1slot, cfg_k1_on) is False


def test_staged_walk_products_heterogeneous_gdn_and_ple():
    """6-tuple mamba_geom separates 2-col GDN stacks and 4-col PLE stack in order."""
    # Two GDN groups (t=6 cols) interleaved with two PLE groups (t=8 cols), 2 reqs.
    n = 2
    g0 = torch.tensor(
        [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]], dtype=torch.int32
    )
    p0 = torch.tensor(
        [
            [100, 101, 102, 103, 104, 105, 106, 107],
            [200, 201, 202, 203, 204, 205, 206, 207],
        ],
        dtype=torch.int32,
    )
    g1 = g0 + 30
    p1 = p0 + 300
    tables = [g0.reshape(-1), p0.reshape(-1), g1.reshape(-1), p1.reshape(-1)]
    staged = torch.cat(tables)
    table_geom = []
    off = 0
    for ti, t in ((0, 6), (1, 8), (2, 6), (3, 8)):
        table_geom.append((off, n, t))
        off += n * t
    mamba_geom = (
        (0, 6, 512, None, 2, 1),  # GDN 0 (2-slot ping-pong)
        (1, 8, 512, None, 4, 0),  # PLE 0 (4-slot std)
        (2, 6, 512, None, 2, 1),  # GDN 1 (2-slot ping-pong)
        (3, 8, 512, None, 4, 0),  # PLE 1 (4-slot std)
    )
    seq_lens = torch.tensor([513, 10], dtype=torch.int32)  # state_off = [1, 0]
    out = exp_md._staged_walk_products(
        staged, seq_lens, tuple(table_geom), mamba_geom, ckpt_window=4
    )
    # 4 tables + 4 states + 4 ckpts + 3 tail stacks (pp_s, pp_c, ple_s) = 15
    assert len(out) == 15
    states = out[4:8]
    ckpts = out[8:12]
    pp_s, pp_c, ple_s = out[12], out[13], out[14]

    assert tuple(ckpts[0].shape) == (2, 2)
    assert tuple(ckpts[1].shape) == (2, 4)
    assert tuple(ckpts[2].shape) == (2, 2)
    assert tuple(ckpts[3].shape) == (2, 4)
    assert torch.equal(ckpts[0], torch.tensor([[11, 12], [20, 21]], dtype=torch.int32))
    assert torch.equal(
        ckpts[1],
        torch.tensor([[101, 102, 103, 104], [200, 201, 202, 203]], dtype=torch.int32),
    )
    assert tuple(pp_s.shape) == (2, 2)
    assert tuple(pp_c.shape) == (2, 2, 2)
    assert tuple(ple_s.shape) == (2, 2)
    assert torch.equal(pp_s[0], states[0])
    assert torch.equal(pp_s[1], states[2])
    assert torch.equal(pp_c[0], ckpts[0])
    assert torch.equal(pp_c[1], ckpts[2])
    assert torch.equal(ple_s[0], states[1])
    assert torch.equal(ple_s[1], states[3])


def test_build_group_partitions_ping_pong_vs_ple_state_indices():
    runner = SimpleNamespace(
        vllm_config=_mock_cfg(3),
        _unified_kv_layout=True,
        pcp_manager=None,
        position_ids=torch.zeros(4, dtype=torch.int32),
    )
    ctx = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=False,
        seq_lens=torch.tensor([10, 20], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        mamba_ckpt_window=4,
        unified_mamba_state_indices=[],
        staged_block_tables={
            0: torch.zeros(4, dtype=torch.int32),
            1: torch.zeros(8, dtype=torch.int32),
        },
        staged_mamba_products={
            0: (
                torch.tensor([11, 21], dtype=torch.int32),
                torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
                None,
            ),
            1: (
                torch.tensor([31, 41], dtype=torch.int32),
                torch.tensor([[31, 32, 33, 34], [41, 42, 43, 44]], dtype=torch.int32),
                None,
            ),
        },
    )
    runner._attn_metadata_builder_ctx = ctx
    common = SimpleNamespace(num_reqs=2)
    gdn_b = SimpleNamespace(
        runner=runner,
        kv_cache_group_id=0,
        is_mamba_group=True,
        kv_cache_spec=_mamba_spec(num_speculative_blocks=1),
        has_prefill_ckpt_blocks=False,
        prefill_ckpt_spec_blocks=None,
    )
    ple_b = SimpleNamespace(
        runner=runner,
        kv_cache_group_id=1,
        is_mamba_group=True,
        kv_cache_spec=_mamba_spec(
            num_speculative_blocks=3, mamba_type=MambaAttentionBackendEnum.SHORT_CONV
        ),
        has_prefill_ckpt_blocks=False,
        prefill_ckpt_spec_blocks=None,
    )
    md_gdn = exp_md._build_group(gdn_b, 0, common)
    md_ple = exp_md._build_group(ple_b, 0, common)
    assert (
        md_gdn.mamba_ckpt_indices is not None
        and md_gdn.mamba_ckpt_indices.shape == (2, 2)
    )
    assert (
        md_ple.mamba_ckpt_indices is not None
        and md_ple.mamba_ckpt_indices.shape == (2, 4)
    )
    assert len(ctx.unified_mamba_state_indices) == 1
    assert torch.equal(
        ctx.unified_mamba_state_indices[0], torch.tensor([31, 41], dtype=torch.int32)
    )
    pp_list = ctx.unified_ping_pong_mamba_state_indices
    assert len(pp_list) == 1
    assert torch.equal(pp_list[0], torch.tensor([11, 21], dtype=torch.int32))


# ---------------------------------------------------------------------------
# two_slot.py runtime helpers & 512-token boundary copy / interior export
# ---------------------------------------------------------------------------


def test_update_ping_pong_read_offsets_flip_reset_and_guards():
    ro = torch.tensor([0, 0, 1, 1, 0, 1, 0], dtype=torch.int32)
    # Row 0: windowed block 1 (0 -> 1)
    # Row 1: windowed block 2 (1 -> 0)
    # Row 2: non-windowed prefill block 3 (1 -> 0 reset)
    # Row 3: padded/OOB/sentinel block (999) ignored
    groups = [torch.tensor([1, 2, 3, 999], dtype=torch.int32)]
    qsl = torch.tensor([0, 4, 8, 20, 20], dtype=torch.int32)
    two_slot.update_ping_pong_read_offsets_impl(
        ro, groups, qsl, torch.tensor(2), torch.tensor(3)
    )
    assert ro[1].item() == 1
    assert ro[2].item() == 0
    assert ro[3].item() == 0
    assert ro[5].item() == 1  # untouched unrelated slot


def test_compute_replay_commit_lens_cases():
    qsl = torch.tensor([0, 4, 8, 12, 20], dtype=torch.int32)
    nt = torch.tensor(
        [
            [10, 11, 12, 13],  # 4 valid == q_len -> all accepted (0 replay)
            [10, 11, -1, -1],  # 2 valid < q_len -> replay 2
            [10, -1, -1, -1],  # 1 valid < q_len -> replay 1
            [10, -1, -1, -1],  # prefill row (outside num_windowed=3) -> 0
        ],
        dtype=torch.int64,
    )
    cl = two_slot.compute_replay_commit_lens(qsl, nt, num_windowed=3)
    assert cl.tolist() == [0, 2, 1, 0]


def test_boundary_copy_exact_and_interior_and_dummy_sink():
    """Verify exact 511->512, interior 510->514, PLE group, and OOB guards."""
    block_size = 512
    width = 5  # _mamba_ckpt_window (4) + 1
    # 2 groups: gid 0 = ping-pong GDN, gid 1 = standard 4-slot PLE.
    is_pp_mask = torch.tensor([True, False], dtype=torch.bool)
    layer_to_plan_idx = torch.tensor([0], dtype=torch.int64)
    # Candidate entries (padded with a bogus row=0 inactive entry AND negative row=-1
    # to confirm dummy max_seqs+1 sink never overwrites row 0!).
    entries = torch.tensor(
        [
            # row 0: interior crossing (start=510, sched=4, boundary=512, k=2)
            [0, 512, 4, 10, 11, 12, 0, 0, 20, 21, 22, 23, 24],
            # row 1: exact crossing at c_read=0 (start=511, sched=1, boundary=512)
            [1, 512, 1, 30, 31, 0, 0, 0, 40, 41, 42, 43, 44],
            # row 2: exact crossing at c_read=1 (already in b0=35)
            [2, 512, 1, 35, 36, 0, 0, 0, 45, 46, 47, 48, 49],
            # row 0 padding entry (boundary=0 -> must not clobber row 0)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # negative row & OOB row sentinel guard
            [-1, 512, 4, 10, 11, 12, 0, 0, 20, 21, 22, 23, 24],
            [99, 512, 4, 10, 11, 12, 0, 0, 20, 21, 22, 23, 24],
        ],
        dtype=torch.int32,
    )
    chunk_seq_lens = torch.tensor([514, 512, 512], dtype=torch.int32)
    next_tokens = torch.tensor(
        [
            [1, 2, 3, 4],  # row 0: 4 accepted -> 510+4=514 > 512 (interior!)
            [5, -1, -1, -1],  # row 1: 1 accepted -> 511+1=512 (exact!)
            [6, -1, -1, -1],  # row 2: 1 accepted -> 511+1=512 (exact, c_read=1)
        ],
        dtype=torch.int64,
    )
    offsets = torch.zeros(64, dtype=torch.int32)
    offsets[11] = 0  # row 0: c_read=0 -> read_blk=11, dst_b=10
    offsets[30] = 0  # row 1: c_read=0 -> write_blk=31 != dst_b=30 -> copy 31->30
    offsets[35] = 1  # row 2: c_read=1 -> write_blk=35 == dst_b=35 -> no-op copy

    out_bucket = 16
    src, dst, s_int, ckpt_int, cl_int = (
        two_slot.compute_boundary_copy_and_interior_impl(
            entries,
            chunk_seq_lens,
            next_tokens,
            offsets,
            torch.tensor(3, dtype=torch.int64),
            is_pp_mask,
            layer_to_plan_idx,
            block_size=block_size,
            width=width,
            split=1,
            out_copy_bucket=out_bucket,
        )
    )
    pairs = set(zip(src.tolist(), dst.tolist()))
    # GDN row 1 exact hit copies 31 -> 30; row 2 already in 35 (not copied).
    assert (31, 30) in pairs
    assert (36, 35) not in pairs
    # PLE row 0 interior hit: col = (513//512) + (512-1-510) - 0 = 2 -> block 22 -> 20.
    assert (22, 20) in pairs
    # Row 0 interior export must not be overwritten by zero-padded row 0.
    assert s_int[0, 0].item() == 11
    assert ckpt_int[0, 0].tolist() == [11, 10]
    assert cl_int[0, 0].item() == 2
    assert cl_int[0, 1].item() == 0
    assert cl_int[0, 2].item() == 0

    # Non-verify 1-token boundary copy with split=2 masks zero pairs to (0, 0).
    src_nv, dst_nv = two_slot.compute_boundary_copy_no_verify_impl(
        entries,
        chunk_seq_lens,
        offsets,
        torch.tensor(3, dtype=torch.int64),
        is_pp_mask,
        True,
        block_size=block_size,
        width=width,
        split=2,
        out_copy_bucket=32,
    )
    nv_pairs = list(zip(src_nv.tolist(), dst_nv.tolist()))
    assert (62, 60) in nv_pairs and (63, 61) in nv_pairs
    # Inactive slots stay exact (0, 0), never (1, 1)!
    assert (1, 1) not in nv_pairs


def test_bm_ckpt_ids_and_no_class_buffer_shadow():
    from vllm_torchtpu.models.qwen4_exp.experimental.ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    # No class-level `= None` shadow for attributes passed to register_buffer.
    for buf_name in ("_verify_qkv_buf", "_verify_b_buf", "_verify_a_buf"):
        assert buf_name not in VllmGatedDeltaNetAttention.__dict__

    stub = SimpleNamespace(
        _bm_kernel=True,
        _bm_region=(2, SimpleNamespace(num_pools=4)),
    )
    raw = torch.tensor([[0, 5], [3, 0]], dtype=torch.int32)
    mapped = VllmGatedDeltaNetAttention._bm_ckpt_ids(stub, raw)
    assert mapped.tolist() == [[0, 5 * 4 + 2], [3 * 4 + 2, 0]]


def test_snapshot_verify_inputs_fixed_shape_and_replay_guards():
    """Multi-chunk snapshot keeps fixed win_cap rows; replay fails on bad state."""
    from vllm_torchtpu.models.qwen4_exp.experimental.ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    layer = object.__new__(VllmGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.linear_attn"
    layer._two_slot_mtp = True
    layer._bm_kernel = True
    layer._bm_region = (2, SimpleNamespace(num_pools=4))
    layer.num_spec = 3
    layer.tp_size = 1
    layer.num_k_heads = 1
    layer.num_v_heads = 2
    layer.head_k_dim = 4
    layer.head_v_dim = 4
    layer.model_config = SimpleNamespace(dtype=torch.float32)
    layer.A_log = torch.zeros(2)
    layer.dt_bias = torch.zeros(2)
    layer.conv1d = SimpleNamespace(weight=torch.zeros(16, 1, 4), bias=None)
    layer._pooled_conv_weight = lambda: layer.conv1d.weight
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64)
    )
    layer._init_verify_input_buffers(cfg)
    assert layer._verify_win_cap == 16  # 4 * (3 + 1)

    layer._verify_qkv_buf[:5].fill_(7.0)
    snap = layer.snapshot_verify_inputs(max_rows=5)
    assert snap is not None
    assert tuple(snap[0].shape) == (16, 16)
    assert tuple(snap[1].shape) == (16, 2)
    assert tuple(snap[2].shape) == (16, 2)
    # Mutating the live buffer does not alter the cloned snapshot.
    layer._verify_qkv_buf[:5].fill_(99.0)
    assert torch.all(snap[0][:5] == 7.0)
    assert layer.snapshot_verify_inputs(max_rows=0) is None

    # Unallocated kv_cache returns early.
    layer.kv_cache = None
    layer.gdn_pooled_replay_op = None
    common_kw = dict(
        state_indices=torch.tensor([1, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        seq_lens=torch.tensor([10, 12], dtype=torch.int32),
        slot_read_offsets=torch.zeros(16, dtype=torch.int32),
        ckpt_indices=torch.tensor([[1, 2], [3, 0]], dtype=torch.int32),
        commit_lens=torch.tensor([2, 0], dtype=torch.int32),
    )
    layer.replay_two_slot_state(**common_kw)

    # Allocated kv_cache with missing replay op raises loudly.
    layer.kv_cache = (torch.zeros((32, 4), dtype=torch.float32),)
    with pytest.raises(RuntimeError, match="no gdn_pooled_replay_op"):
        layer.replay_two_slot_state(**common_kw)

    # Recorded call receives fixed-shape snap, mapped bm ids (preserving null 0),
    # and unmapped raw_state_indices for read_offset_indices.
    recorded: list[tuple] = []
    layer.gdn_pooled_replay_op = lambda *a: recorded.append(a)
    layer.replay_two_slot_state(**common_kw, staged_inputs=snap)
    assert len(recorded) == 1
    args = recorded[0]
    assert args[0] is snap[0] and args[0].shape[0] == 16
    assert args[8].tolist() == [6, 10]  # phys_state
    assert args[13].tolist() == [[6, 10], [14, 0]]  # phys_ckpt preserving 0
    assert args[14].tolist() == [2, 0]  # commit_lens
    assert args[15].tolist() == [1, 2]  # raw_state for read_offset_indices

    # Missing staged verify inputs raises loudly.
    layer.register_buffer("_verify_qkv_buf", None, persistent=False)
    with pytest.raises(RuntimeError, match="no staged verify inputs"):
        layer.replay_two_slot_state(**common_kw)


def test_gdn_forward_bm_read_offset_indices_gated_to_two_slot(monkeypatch):
    """Block-major read_offset_indices & _bm_ckpt_ids apply only under two-slot."""
    from vllm_torchtpu.models.qwen4_exp.experimental.ops import (
        gdn_attention_op as gdn_op,
    )

    def _run_forward(*, two_slot_mtp: bool):
        layer = object.__new__(gdn_op.VllmGatedDeltaNetAttention)
        torch.nn.Module.__init__(layer)
        layer.prefix = "model.layers.0.linear_attn"
        layer._pcp_streaming_configured = False
        layer._two_slot_mtp = two_slot_mtp
        layer._bm_kernel = True
        layer._bm_region = (2, SimpleNamespace(num_pools=4))
        layer.num_spec = 3
        layer.tp_size = 1
        layer.num_k_heads = 1
        layer.num_v_heads = 2
        layer.head_k_dim = 4
        layer.head_v_dim = 4
        layer.value_dim = 8
        layer.model_config = SimpleNamespace(dtype=torch.float32)
        layer.A_log = torch.zeros(2)
        layer.dt_bias = torch.zeros(2)
        layer.conv1d = SimpleNamespace(weight=torch.zeros(16, 1, 4), bias=None)
        layer._pooled_conv_weight = lambda: layer.conv1d.weight
        layer.kv_cache = (torch.zeros((32, 4), dtype=torch.float32),)
        layer.gdn_pooled_hoisted_op = None
        layer.gdn_step_metadata_op = None

        qkvz_dim = 16 + 8
        layer.in_proj_qkvz = lambda h: (torch.zeros((h.shape[0], qkvz_dim)), None)
        layer.in_proj_ba = lambda h: (torch.zeros((h.shape[0], 4)), None)
        layer.norm = lambda x, _z: x
        layer.out_proj = lambda x: (x, None)

        cfg = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64)
        )
        layer._init_verify_input_buffers(cfg)

        calls: list[tuple[str, tuple]] = []
        layer.gdn_pooled_op = lambda *a: (
            calls.append(("per_token", a)) or torch.zeros((a[0].shape[0], 2, 4))
        )
        layer.gdn_pooled_single_ckpt_op = (
            (
                lambda *a: (
                    calls.append(("two_slot", a)) or torch.zeros((a[0].shape[0], 2, 4))
                )
            )
            if two_slot_mtp
            else None
        )

        ckpt = (
            torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
            if two_slot_mtp
            else torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
        )
        md = SimpleNamespace(
            mamba_state_indices=torch.tensor([1, 3], dtype=torch.int32),
            mamba_ckpt_indices=ckpt,
            mamba_slot_read_offsets=torch.zeros(16, dtype=torch.int32),
            mamba_request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
            seq_lens=torch.tensor([10, 12], dtype=torch.int32),
        )
        monkeypatch.setattr(
            gdn_op,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={layer.prefix: md}),
        )
        layer.forward(torch.zeros((8, 8)))
        return calls

    pt_calls = _run_forward(two_slot_mtp=False)
    assert len(pt_calls) == 1 and pt_calls[0][0] == "per_token"
    # Default per_token path receives 14 args (no extra read_offset_indices) and
    # leaves ckpt_indices unmapped (unchanged from base).
    assert len(pt_calls[0][1]) == 14
    assert pt_calls[0][1][13].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]

    ts_calls = _run_forward(two_slot_mtp=True)
    assert len(ts_calls) == 1 and ts_calls[0][0] == "two_slot"
    # Two-slot path maps ckpt_indices (preserving 0) and passes raw_state_indices.
    assert len(ts_calls[0][1]) == 16
    assert ts_calls[0][1][13].tolist() == [[6, 10], [14, 0]]
    assert ts_calls[0][1][14] is None
    assert ts_calls[0][1][15].tolist() == [1, 3]


def test_qwen38_runner_override_signatures_match_base_ast():
    """Static check duplicating test_runner.py's override-signature guard."""
    repo_root = Path(__file__).resolve().parents[4]
    base_src = (repo_root / "src/vllm_torchtpu/runner/tpu_runner.py").read_text()
    sub_src = (
        repo_root / "src/vllm_torchtpu/models/qwen4_exp/experimental/runner/runner.py"
    ).read_text()
    test_src = (
        repo_root / "tests/models/qwen4_exp/experimental/runner/test_runner.py"
    ).read_text()

    def _class_methods(src: str, cls_name: str) -> dict[str, list[str]]:
        mod = ast.parse(src)
        for node in mod.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                return {
                    item.name: [a.arg for a in item.args.posonlyargs + item.args.args]
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                    and not item.name.startswith("__")
                }
        raise AssertionError(f"class {cls_name} not found")

    base_m = _class_methods(base_src, "TPUModelRunner")
    sub_m = _class_methods(sub_src, "Qwen38Runner")
    mod_t = ast.parse(test_src)
    new_methods: set[str] = set()
    for node in mod_t.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "NEW_METHODS":
                    new_methods = ast.literal_eval(node.value)
    assert set(sub_m) >= new_methods
    for name, params in sub_m.items():
        if name in new_methods:
            assert name not in base_m, f"TPUModelRunner already defines {name}"
        else:
            assert name in base_m, f"Unknown override {name} not in TPUModelRunner"
            assert params == base_m[name], (
                f"Signature mismatch on {name}: {params} vs {base_m[name]}"
            )
