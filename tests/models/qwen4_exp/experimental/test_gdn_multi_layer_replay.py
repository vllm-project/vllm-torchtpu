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
"""Unit and TPU tests for single-launch multi-layer GDN replay, dual-layout
verify staging, quiescent precompile, and fused GDN+PLE read-offset updates."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from torch import nn

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.models.qwen4_exp.experimental import model as exp_model
from vllm_torchtpu.models.qwen4_exp.experimental.kernels.gdn_v3 import (
    config as gdn_cfg,
)
from vllm_torchtpu.models.qwen4_exp.experimental.kernels.gdn_v3 import (
    wrapper as gdn_wrapper,
)
from vllm_torchtpu.models.qwen4_exp.experimental.ops.gdn_attention_op import (
    MultiLayerPooledReplayRunner,
    VllmGatedDeltaNetAttention,
    compute_fused_flip_and_ple_offsets_jax,
)
from vllm_torchtpu.models.qwen4_exp.experimental.runner import two_slot

_FAKE_TPU_INFO = SimpleNamespace(
    num_lanes=128,
    vmem_capacity_bytes=32 * 1024 * 1024,
)


def _cpu_load_typed(ref, *, view_dtype, lane_split=1):
    del lane_split
    return ref[...].astype(view_dtype)


def _cpu_store_typed(ref, arr, *, lane_split=1):
    del lane_split
    ref[...] = arr.astype(ref.dtype)


def _make_pool_plan(n_v: int, dim: int, k_sz: int):
    rec_rows = n_v * 128
    conv_rows = (k_sz - 1) * (dim // 128)
    conv_nrows = ((conv_rows + 15) // 16) * 16
    total_rows = rec_rows + conv_nrows
    plan = gdn_cfg.StateSourcePlan(
        stride=1,
        recurrent=gdn_cfg.StateRegion(
            kb0=0,
            nblocks=1,
            row0=0,
            nrows=rec_rows,
            view_dtype=jnp.float32,
            lane_split=1,
            rows_used=rec_rows,
        ),
        conv=gdn_cfg.StateRegion(
            kb0=0,
            nblocks=1,
            row0=rec_rows,
            nrows=conv_nrows,
            view_dtype=jnp.float32,
            lane_split=1,
            rows_used=conv_rows,
        ),
        whole_block_dma=False,
    )
    return plan, total_rows, rec_rows, conv_rows


def _run_multi_layer_parity(
    *,
    num_layers: int,
    n_kq: int,
    n_v: int,
    max_seqs: int,
    commit_lens_stack: jax.Array,
    with_bias: bool,
    kmajor: bool,
    slens_list: list[int],
    seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    d_k = d_v = 128
    k_sz = 4
    dim = 2 * n_kq * d_k + n_v * d_v
    win_rows = 4 * max_seqs
    plan, total_rows, rec_rows, conv_rows = _make_pool_plan(n_v, dim, k_sz)
    pool_blocks = 2 + num_layers * max_seqs * 2
    pool0 = jnp.zeros((pool_blocks, total_rows, 128), dtype=jnp.float32)

    keys = jax.random.split(jax.random.PRNGKey(seed), 7)
    qkv_3d = jax.random.normal(keys[0], (num_layers, win_rows, dim), jnp.float32) * 0.06
    b_3d = jax.random.normal(keys[1], (num_layers, win_rows, n_v), jnp.float32) * 0.08
    a_3d = jax.random.normal(keys[2], (num_layers, win_rows, n_v), jnp.float32) * 0.08
    cw_raw = jax.random.normal(keys[3], (num_layers, dim, 1, k_sz), jnp.float32) * 0.05
    cw = jnp.swapaxes(cw_raw, 1, 3) if kmajor else cw_raw
    cb = (
        jax.random.normal(keys[4], (num_layers, dim), jnp.float32) * 0.02
        if with_bias
        else None
    )
    al = jax.random.normal(keys[5], (num_layers, n_v), jnp.float32) * 0.1
    db = jax.random.normal(keys[6], (num_layers, n_v), jnp.float32) * 0.1

    qsl = jnp.array([4 * i for i in range(max_seqs + 1)], jnp.int32)
    dist = jnp.array([max_seqs, max_seqs, max_seqs], jnp.int32)
    slens = jnp.array(slens_list, jnp.int32)

    s_idx = jnp.zeros((num_layers, max_seqs), jnp.int32)
    ck_idx = jnp.zeros((num_layers, max_seqs, 2), jnp.int32)
    ro = jnp.zeros((num_layers, max_seqs), jnp.int32)
    nxt = 1
    for li in range(num_layers):
        for s in range(max_seqs):
            s_idx = s_idx.at[li, s].set(nxt)
            ck_idx = ck_idx.at[li, s, 0].set(nxt)
            ck_idx = ck_idx.at[li, s, 1].set(nxt + 1)
            ro_v = (li + s) % 2
            ro = ro.at[li, s].set(ro_v)
            read_slot = nxt + ro_v
            rk, ck = jax.random.split(jax.random.PRNGKey(seed * 100 + li * 20 + s))
            pool0 = pool0.at[read_slot, :rec_rows].set(
                jax.random.normal(rk, (rec_rows, 128), jnp.float32) * 0.04
            )
            pool0 = pool0.at[read_slot, rec_rows : rec_rows + conv_rows].set(
                jax.random.normal(ck, (conv_rows, 128), jnp.float32) * 0.04
            )
            nxt += 2
    snap = pool0.copy()

    interp = pltpu.InterpretParams(detect_races=True)
    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=_FAKE_TPU_INFO),
        mock.patch.object(typed_ldst, "load_typed", side_effect=_cpu_load_typed),
        mock.patch.object(typed_ldst, "store_typed", side_effect=_cpu_store_typed),
    ):
        ref_pool = snap.copy()
        for li in range(num_layers):
            cl_l = commit_lens_stack[0 if commit_lens_stack.shape[0] == 1 else li]
            ref_pool, _ = gdn_wrapper.fused_conv1d_gdn(
                qkv_3d[li],
                b_3d[li],
                a_3d[li],
                None,
                None,
                cw[li],
                None if cb is None else cb[li],
                al[li],
                db[li],
                qsl,
                s_idx[li],
                dist,
                slens,
                read_offsets=ro[li],
                ckpt_indices=ck_idx[li],
                commit_lens=cl_l,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=k_sz,
                state_source=ref_pool,
                state_plan=plan,
                num_spec_tokens=3,
                conv_weight_kmajor=kmajor,
                single_checkpoint=True,
                batched_only=True,
                interpret=interp,
            )
        ml_pool = gdn_wrapper.run_multi_layer_batched_replay_pooled(
            qkv_3d,
            b_3d,
            a_3d,
            snap.copy(),
            cw,
            cb,
            al,
            db,
            s_idx,
            ro,
            ck_idx,
            commit_lens_stack,
            qsl,
            dist,
            slens,
            state_plan=plan,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=k_sz,
            num_spec_tokens=3,
            conv_weight_kmajor=kmajor,
            interpret=interp,
        )
    return snap, ref_pool, ml_pool


# ---------------------------------------------------------------------------
# Single-launch L-layer Pallas replay bit-exactness tests
# ---------------------------------------------------------------------------


def test_multi_layer_replay_1d_compaction_bit_exact_with_bias():
    """cl_rows == 1 triggers 1D jnp.nonzero compaction, L=3, odd max_seqs=5."""
    cl_1d = jnp.array([[2, 0, 3, 1, 0]], dtype=jnp.int32)
    snap, ref_pool, ml_pool = _run_multi_layer_parity(
        num_layers=3,
        n_kq=1,
        n_v=1,
        max_seqs=5,
        commit_lens_stack=cl_1d,
        with_bias=True,
        kmajor=False,
        slens_list=[10, 12, 4, 20, 16],  # seq 2 is fresh (no init state)
        seed=11,
    )
    assert float(jnp.max(jnp.abs(ml_pool - ref_pool))) == 0.0
    # Pool actually mutated for active rejected slots.
    assert float(jnp.max(jnp.abs(ml_pool - snap))) > 0.0


def test_multi_layer_replay_per_layer_heterogeneous_kmajor_nobias():
    """cl_rows == L exercises heterogeneous per-layer compaction & kmajor weights."""
    cl_per_layer = jnp.array([[0, 2, 0], [3, 0, 1]], dtype=jnp.int32)
    snap, ref_pool, ml_pool = _run_multi_layer_parity(
        num_layers=2,
        n_kq=1,
        n_v=1,
        max_seqs=3,
        commit_lens_stack=cl_per_layer,
        with_bias=False,
        kmajor=True,
        slens_list=[10, 12, 14],
        seed=23,
    )
    assert float(jnp.max(jnp.abs(ml_pool - ref_pool))) == 0.0
    assert float(jnp.max(jnp.abs(ml_pool - snap))) > 0.0


def test_multi_layer_replay_zero_work_fast_path():
    """All-accepted batch (commit_lens == 0) launches 0 tiles & leaves pool intact."""
    cl_zero = jnp.zeros((1, 3), dtype=jnp.int32)
    snap, ref_pool, ml_pool = _run_multi_layer_parity(
        num_layers=4,
        n_kq=1,
        n_v=1,
        max_seqs=3,
        commit_lens_stack=cl_zero,
        with_bias=True,
        kmajor=False,
        slens_list=[10, 12, 14],
        seed=31,
    )
    assert float(jnp.max(jnp.abs(ref_pool - snap))) == 0.0
    assert float(jnp.max(jnp.abs(ml_pool - snap))) == 0.0


def test_multi_layer_replay_non_pow2_heads_detect_races():
    """Non-power-of-two head count (n_v=3 ~ 48v head ratio) stays race-free."""
    cl_1d = jnp.array([[2, 1, 0]], dtype=jnp.int32)
    _, ref_pool, ml_pool = _run_multi_layer_parity(
        num_layers=2,
        n_kq=1,
        n_v=3,
        max_seqs=3,
        commit_lens_stack=cl_1d,
        with_bias=False,
        kmajor=False,
        slens_list=[12, 4, 16],
        seed=47,
    )
    assert float(jnp.max(jnp.abs(ml_pool - ref_pool))) < 1e-6


def test_multi_layer_replay_validation_errors():
    plan, total_rows, _, _ = _make_pool_plan(1, 384, 4)
    pool = jnp.zeros((6, total_rows, 128), jnp.float32)
    with mock.patch.object(pltpu, "get_tpu_info", return_value=_FAKE_TPU_INFO):
        with pytest.raises(ValueError, match="state_indices_stack"):
            gdn_wrapper.run_multi_layer_batched_replay_pooled(
                jnp.zeros((2, 8, 384), jnp.float32),
                jnp.zeros((2, 8, 1), jnp.float32),
                jnp.zeros((2, 8, 1), jnp.float32),
                pool,
                jnp.zeros((2, 384, 1, 4), jnp.float32),
                None,
                jnp.zeros((2, 1), jnp.float32),
                jnp.zeros((2, 1), jnp.float32),
                jnp.zeros((3, 2), jnp.int32),  # wrong L
                jnp.zeros((2, 2), jnp.int32),
                jnp.zeros((2, 2, 2), jnp.int32),
                jnp.zeros((1, 2), jnp.int32),
                jnp.array([0, 4, 8], jnp.int32),
                jnp.array([2, 2, 2], jnp.int32),
                jnp.array([8, 8], jnp.int32),
                state_plan=plan,
                n_kq=1,
                n_v=1,
                d_k=128,
                d_v=128,
                kernel_size=4,
                num_spec_tokens=3,
            )
        with pytest.raises(ValueError, match="commit_lens_stack"):
            gdn_wrapper.run_multi_layer_batched_replay_pooled(
                jnp.zeros((2, 8, 384), jnp.float32),
                jnp.zeros((2, 8, 1), jnp.float32),
                jnp.zeros((2, 8, 1), jnp.float32),
                pool,
                jnp.zeros((2, 384, 1, 4), jnp.float32),
                None,
                jnp.zeros((2, 1), jnp.float32),
                jnp.zeros((2, 1), jnp.float32),
                jnp.zeros((2, 2), jnp.int32),
                jnp.zeros((2, 2), jnp.int32),
                jnp.zeros((2, 2, 2), jnp.int32),
                jnp.zeros((3, 2), jnp.int32),  # wrong rows
                jnp.array([0, 4, 8], jnp.int32),
                jnp.array([2, 2, 2], jnp.int32),
                jnp.array([8, 8], jnp.int32),
                state_plan=plan,
                n_kq=1,
                n_v=1,
                d_k=128,
                d_v=128,
                kernel_size=4,
                num_spec_tokens=3,
            )


def test_multi_layer_replay_rejects_non_divisible_tp_heads():
    """Head counts not divisible by tp_size fail loudly even when // tp_size > 0."""
    import numpy as np

    from vllm_torchtpu.models.qwen4_exp.experimental import (
        gdn_attention as exp_gdn_attn,
    )

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    with (
        mock.patch.object(exp_gdn_attn, "get_mesh_shape_product", return_value=2),
        pytest.raises(ValueError, match="divisible by tp_size=2"),
    ):
        exp_gdn_attn.run_jax_gdn_multi_layer_replay_pooled(
            jnp.zeros((2, 8, 384), jnp.float32),
            jnp.zeros((2, 8, 3), jnp.float32),
            jnp.zeros((2, 8, 3), jnp.float32),
            jnp.zeros((4, 16, 1, 4, 256), jnp.float8_e4m3fn),
            jnp.zeros((2, 384, 1, 4), jnp.float32),
            None,
            jnp.zeros((2, 3), jnp.float32),
            jnp.zeros((2, 3), jnp.float32),
            jnp.zeros((2, 2), jnp.int32),
            jnp.zeros((2, 2), jnp.int32),
            jnp.zeros((2, 2, 2), jnp.int32),
            jnp.zeros((1, 2), jnp.int32),
            jnp.array([0, 4, 8], jnp.int32),
            jnp.array([2, 2, 2], jnp.int32),
            jnp.array([8, 8], jnp.int32),
            n_kq=3,
            n_v=3,
            d_k=128,
            d_v=128,
            kernel_size=4,
            pool_block_tokens=16,
            mesh=mesh,
        )


# ---------------------------------------------------------------------------
# Dual-layout shared verify staging + QSA live-selection summary coexistence
# ---------------------------------------------------------------------------


class _FakeGdnLayer(nn.Module):
    _shared_verify_flat_tensors = None
    _shared_verify_sp_flat_tensors = None
    _shared_verify_backbone_ref = None

    def __init__(self, win_cap: int, qkv_dim: int, local_n_v: int):
        super().__init__()
        self._two_slot_mtp = True
        self._verify_win_cap = win_cap
        self._verify_qkv_dim = qkv_dim
        self._verify_local_n_v = local_n_v
        self._verify_act_dtype = torch.float32
        self._shared_verify_flat_info = None
        self.register_buffer("A_log", torch.zeros(local_n_v))

    bind_shared_verify_flat = VllmGatedDeltaNetAttention.bind_shared_verify_flat
    get_active_verify_flat_tensors = (
        VllmGatedDeltaNetAttention.get_active_verify_flat_tensors
    )
    get_verify_inputs = VllmGatedDeltaNetAttention.get_verify_inputs
    snapshot_verify_inputs = VllmGatedDeltaNetAttention.snapshot_verify_inputs


class _FakeDecoderLayer(nn.Module):
    def __init__(self, layer_type: str, win_cap: int, qkv_dim: int, local_n_v: int):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = _FakeGdnLayer(win_cap, qkv_dim, local_n_v)
        self._qkv_offset = 0.0

    def forward(
        self,
        hidden,
        *args,
        gdn_verify_collector=None,
        qsa_sel_rows=None,
        **kwargs,
    ):
        del args, kwargs
        if self.layer_type == "linear_attention" and gdn_verify_collector is not None:
            n_r = hidden.shape[0]
            qkv = torch.full(
                (n_r, self.linear_attn._verify_qkv_dim),
                self._qkv_offset,
                dtype=torch.float32,
            )
            b = torch.full(
                (n_r, self.linear_attn._verify_local_n_v),
                self._qkv_offset + 10.0,
                dtype=torch.float32,
            )
            a = torch.full(
                (n_r, self.linear_attn._verify_local_n_v),
                self._qkv_offset + 20.0,
                dtype=torch.float32,
            )
            gdn_verify_collector.append((qkv, b, a))
        elif self.layer_type == "full_attention" and qsa_sel_rows is not None:
            qsa_sel_rows.append(
                torch.tensor([int(self._qkv_offset)], dtype=torch.int32)
            )
        return hidden


def test_dual_layout_verify_staging_and_qsa_sel_coexistence():
    """SP (`dim=-1`) staging allocates lazily only when Qwen4ExpBackboneSP exists."""
    from vllm.config.compilation import CompilationMode

    win_cap = 5  # padded up to win_stride = 8
    qkv_dim = 6
    local_n_v = 2
    bb = object.__new__(exp_model.Qwen4ExpBackbone)
    nn.Module.__init__(bb)
    bb.host_ple_index = None
    bb.hc_count = 1
    bb.emit_mtp_hidden = False
    bb.embed_tokens = nn.Embedding(16, 4)
    l0 = _FakeDecoderLayer("linear_attention", win_cap, qkv_dim, local_n_v)
    l0._qkv_offset = 1.0
    l1 = _FakeDecoderLayer("full_attention", win_cap, qkv_dim, local_n_v)
    l1._qkv_offset = 7.0
    l2 = _FakeDecoderLayer("linear_attention", win_cap, qkv_dim, local_n_v)
    l2._qkv_offset = 2.0
    bb.layers = nn.ModuleList([l0, l1, l2])
    bb.hyper_connection_mixer = SimpleNamespace(mix=lambda x: (x,))
    bb._init_shared_gdn_verify_flat_buffers()

    assert tuple(bb._buffers["_gdn_verify_qkv_flat"].shape) == (2 * 8, qkv_dim)
    assert tuple(bb._buffers["_gdn_verify_b_flat"].shape) == (2 * 8, local_n_v)
    # Without Qwen4ExpBackboneSP, SP staging buffers remain unallocated (None).
    assert bb._buffers["_gdn_verify_qkv_sp_flat"] is None
    assert bb._buffers["_gdn_verify_b_sp_flat"] is None
    assert bb._buffers["_gdn_verify_a_sp_flat"] is None
    assert l0.linear_attn._shared_verify_sp_flat_tensors is None
    bb._apply(lambda t: t.clone())
    assert bb._buffers["_gdn_verify_qkv_sp_flat"] is None

    # Constructing Qwen4ExpBackboneSP lazily allocates and binds SP buffers.
    no_compile_cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )
    bb_sp = exp_model.Qwen4ExpBackboneSP(vllm_config=no_compile_cfg, backbone=bb)
    assert bb_sp._has_shared_gdn_verify is True
    assert tuple(bb._buffers["_gdn_verify_qkv_sp_flat"].shape) == (
        win_cap,
        2 * qkv_dim,
    )
    assert tuple(bb._buffers["_gdn_verify_b_sp_flat"].shape) == (
        win_cap,
        2 * local_n_v,
    )
    assert (
        bb_sp._buffers["_gdn_verify_qkv_flat"] is bb._buffers["_gdn_verify_qkv_sp_flat"]
    )
    assert l0.linear_attn._shared_verify_sp_flat_tensors is not None
    assert bb_sp._qsa_sel_buf is not None
    assert tuple(bb_sp._qsa_sel_buf.shape) == (1, 1)

    # Simulate non-SP forward (`dim=0` staging, 3 rows < win_stride=8).
    dummy_ids = torch.zeros(3, dtype=torch.int64)
    dummy_pos = torch.zeros(3, dtype=torch.int64)
    bb.forward(
        dummy_ids,
        dummy_pos,
        inputs_embeds=torch.zeros(3, 4),
        query_start_loc=torch.tensor([0, 3]),
        ngram_context=None,
        state_slots=None,
        context_lens=None,
        valid_tokens=None,
        main_block_table=None,
        compressed_block_table=None,
    )
    assert bb._last_verify_flat_from_sp is False
    # SP buffer stayed zero, non-SP buffer got both layers padded to win_stride=8.
    assert float(bb._buffers["_gdn_verify_qkv_sp_flat"].abs().max()) == 0.0
    q0, b0, a0 = l0.linear_attn.get_verify_inputs()
    q2, b2, a2 = l2.linear_attn.get_verify_inputs()
    assert tuple(q0.shape) == (win_cap, qkv_dim)
    assert torch.all(q0[:3] == 1.0) and torch.all(q0[3:] == 0.0)
    assert torch.all(b0[:3] == 11.0) and torch.all(a0[:3] == 21.0)
    assert torch.all(q2[:3] == 2.0) and torch.all(b2[:3] == 12.0)

    # Multi-chunk snapshot preserves fixed `(win_cap, ...)` shape and isolates values.
    snap_l0 = l0.linear_attn.snapshot_verify_inputs(3)
    assert snap_l0 is not None
    assert tuple(snap_l0[0].shape) == (win_cap, qkv_dim)
    assert torch.all(snap_l0[0][:3] == 1.0)

    # Simulate SP forward (`dim=-1` staging + `_qsa_sel_buf` populated together).
    l0._qkv_offset = 4.0
    l2._qkv_offset = 5.0
    with (
        mock.patch.object(exp_model.sp, "check_divisible", lambda n: None),
        mock.patch.object(exp_model.sp, "shard", lambda x: x),
        mock.patch.object(exp_model.sp, "gather", lambda x, n: x),
    ):
        bb_sp.forward(
            torch.zeros(4, dtype=torch.int64),
            torch.zeros(4, dtype=torch.int64),
            inputs_embeds=torch.zeros(4, 4),
            query_start_loc=torch.tensor([0, 4]),
            ngram_context=None,
            state_slots=None,
            context_lens=None,
            valid_tokens=None,
            main_block_table=None,
            compressed_block_table=None,
        )
    assert bb._last_verify_flat_from_sp is True
    sp_q0, sp_b0, _ = l0.linear_attn.get_verify_inputs()
    sp_q2, _, _ = l2.linear_attn.get_verify_inputs()
    assert torch.all(sp_q0[:4] == 4.0) and torch.all(sp_b0[:4] == 14.0)
    assert torch.all(sp_q2[:4] == 5.0)
    # Non-SP flat buffer wasn't overwritten by SP pass, and snapshot stayed 1.0.
    assert torch.all(bb._buffers["_gdn_verify_qkv_flat"][:3] == 1.0)
    assert torch.all(snap_l0[0][:3] == 1.0)
    # Live QSA selection summary buffer remained intact & updated (int32).
    assert bb_sp._qsa_sel_buf.dtype == torch.int32
    assert int(bb_sp._qsa_sel_buf[0, 0].item()) == 7

    # `_apply()` preserves dual bindings when buffers are replaced/moved.
    old_flat = bb._buffers["_gdn_verify_qkv_flat"]
    old_sp_flat = bb._buffers["_gdn_verify_qkv_sp_flat"]
    bb._apply(lambda t: t.clone())
    bb_sp._apply(lambda t: t)
    assert bb._buffers["_gdn_verify_qkv_flat"] is not old_flat
    assert bb._buffers["_gdn_verify_qkv_sp_flat"] is not old_sp_flat
    assert (
        l0.linear_attn.get_active_verify_flat_tensors(force_sp=False)[0]
        is bb._buffers["_gdn_verify_qkv_flat"]
    )
    assert (
        l0.linear_attn.get_active_verify_flat_tensors(force_sp=True)[0]
        is bb_sp._buffers["_gdn_verify_qkv_flat"]
    )


# ---------------------------------------------------------------------------
# Fused 48-group (36 GDN + 12 PLE) offset update & weight-cache tests
# ---------------------------------------------------------------------------


def test_fused_48_group_flip_and_ple_offsets_jax():
    """Fused update matches separate 36 GDN flip + 12 PLE offset scatter."""
    num_gdn, num_ple = 36, 12
    num_blocks = 512
    max_seqs = 8
    num_win = 4
    num_reqs = 6  # rows 0..3 verify, rows 4..5 prefill, rows 6..7 padded

    ro_init = torch.zeros(num_blocks, dtype=torch.int32)
    gdn_ids = torch.zeros((num_gdn, max_seqs), dtype=torch.int32)
    ple_ids = torch.zeros((num_ple, max_seqs), dtype=torch.int32)
    nxt = 1
    for g in range(num_gdn):
        for r in range(num_reqs):
            gdn_ids[g, r] = nxt
            ro_init[nxt] = (g + r) % 2
            nxt += 1
    for g in range(num_ple):
        for r in range(num_reqs):
            ple_ids[g, r] = nxt
            ro_init[nxt] = 3
            nxt += 1
    # Inject out-of-range sentinel into padded column to ensure clamp safety.
    gdn_ids[0, 7] = num_blocks + 50
    ple_ids[0, 7] = -1

    qsl = torch.tensor([0, 4, 8, 12, 16, 32, 48, 48, 48], dtype=torch.int32)
    ple_off_1d = torch.tensor([0, 1, 2, 3, 0, 0, 0, 0], dtype=torch.int32)

    out_jax = compute_fused_flip_and_ple_offsets_jax(
        jnp.asarray(ro_init.numpy()),
        jnp.asarray(gdn_ids.numpy()),
        jnp.asarray(qsl.numpy()),
        jnp.array([num_win, num_reqs], dtype=jnp.int32),
        ple_state_stack=jnp.asarray(ple_ids.numpy()),
        ple_offsets_1d=jnp.asarray(ple_off_1d.numpy()),
    )
    ref = ro_init.clone()
    two_slot.update_ping_pong_read_offsets_impl(
        ref, [gdn_ids], qsl, torch.tensor(num_win), torch.tensor(num_reqs)
    )
    for g in range(num_ple):
        for r in range(num_reqs):
            ref[int(ple_ids[g, r])] = ple_off_1d[r]
    assert torch.equal(torch.from_numpy(out_jax.__array__().copy()), ref)

    # Also verify GDN-only (ple_state_stack=None) path.
    out_gdn_only = compute_fused_flip_and_ple_offsets_jax(
        jnp.asarray(ro_init.numpy()),
        jnp.asarray(gdn_ids.numpy()),
        jnp.asarray(qsl.numpy()),
        jnp.array([num_win, num_reqs], dtype=jnp.int32),
    )
    ref_gdn = ro_init.clone()
    two_slot.update_ping_pong_read_offsets_impl(
        ref_gdn, [gdn_ids], qsl, torch.tensor(num_win), torch.tensor(num_reqs)
    )
    assert torch.equal(torch.from_numpy(out_gdn_only.__array__().copy()), ref_gdn)


def test_weight_stacks_cache_refresh_on_mutation():
    runner = object.__new__(MultiLayerPooledReplayRunner)
    layers = []
    for i in range(3):
        cw = torch.full((4, 1, 8), float(i + 1))
        cb = torch.full((8,), float(i + 10))
        al = torch.full((2,), float(i + 20))
        db = torch.full((2,), float(i + 30))
        layers.append(
            SimpleNamespace(
                _pooled_conv_weight=lambda cw=cw: cw,
                conv1d=SimpleNamespace(bias=cb),
                A_log=al,
                dt_bias=db,
            )
        )
    runner.layers = tuple(layers)
    runner._has_conv_bias = True
    runner._weight_key = None
    runner._cached_weight_stacks = None

    s1 = runner._weight_stacks()
    s2 = runner._weight_stacks()
    assert s1 is s2  # cache hit
    assert tuple(s1[0].shape) == (3, 4, 1, 8)
    # In-place mutation increments `_version` and refreshes CPU cache.
    layers[0].A_log.add_(5.0)
    s3 = runner._weight_stacks()
    assert s3 is not s1
    assert float(s3[2][0, 0].item()) == 25.0


def test_count_mamba_group_classes_handles_mamba_cache_mode_none():
    """Precompile sizes (G_pp, req_pad) and (G_std, req_pad) even without align plan."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_torchtpu.models.qwen4_exp.experimental.runner.metadata import (
        is_ping_pong_mamba_spec,
    )

    gdn_spec = MambaSpec(
        shapes=((3, 1, 128), (2, 128, 128)),
        dtypes=(torch.float32, torch.bfloat16),
        block_size=512,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        num_speculative_blocks=1,
    )
    ple_spec = MambaSpec(
        shapes=((2, 128),),
        dtypes=(torch.bfloat16,),
        block_size=512,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        num_speculative_blocks=3,
    )
    vllm_cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        model_config=SimpleNamespace(
            architecture="Qwen4ExpForCausalLM", hf_config=None
        ),
    )
    groups = (
        [
            [SimpleNamespace(kv_cache_group_id=i, kv_cache_spec=gdn_spec)]
            for i in range(36)
        ]
        + [
            [SimpleNamespace(kv_cache_group_id=36 + j, kv_cache_spec=ple_spec)]
            for j in range(12)
        ]
        + [[SimpleNamespace(kv_cache_group_id=48, kv_cache_spec=object())]]
    )
    # mamba_cache_mode="none": _mamba_copy_plan and _mamba_copy_ping_pong_gids empty!
    runner_none = SimpleNamespace(
        attn_groups=groups,
        vllm_config=vllm_cfg,
        _mamba_copy_plan=[],
        _mamba_copy_ping_pong_gids=set(),
    )
    assert two_slot.count_mamba_group_classes(
        runner_none, 36, is_ping_pong_spec=is_ping_pong_mamba_spec
    ) == (36, 12)

    # Without attn_groups, fall back to copy plan when present or num_gdn_layers.
    runner_fallback = SimpleNamespace(
        attn_groups=None,
        vllm_config=vllm_cfg,
        _mamba_copy_plan=[],
        _mamba_copy_ping_pong_gids=set(),
    )
    assert two_slot.count_mamba_group_classes(
        runner_fallback, 69, is_ping_pong_spec=is_ping_pong_mamba_spec
    ) == (69, 0)


def test_two_slot_multistep_verify_replay_parity_no_drift():
    """6-step alternating verify + rejection replay stays bit-exact vs 4-slot ref."""
    n_kq, n_v, d_k, d_v, k_sz = 1, 1, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    num_seqs = 3
    w = 4
    plan, total_rows, rec_rows, conv_rows = _make_pool_plan(n_v, dim, k_sz)
    pool_ref = jnp.zeros((2 + num_seqs * w, total_rows, 128), dtype=jnp.float32)
    pool_2s = jnp.zeros((2 + num_seqs * 2, total_rows, 128), dtype=jnp.float32)
    ckpt_ref = jnp.array(
        [[1 + s * w + t for t in range(w)] for s in range(num_seqs)], dtype=jnp.int32
    )
    ckpt_2s = jnp.array(
        [[1 + s * 2 + t for t in range(2)] for s in range(num_seqs)], dtype=jnp.int32
    )
    for s in range(num_seqs):
        rk, ck = jax.random.split(jax.random.PRNGKey(900 + s))
        sr = jax.random.normal(rk, (rec_rows, 128), jnp.float32) * 0.05
        sc = jax.random.normal(ck, (conv_rows, 128), jnp.float32) * 0.05
        pool_ref = pool_ref.at[1 + s * w, :rec_rows].set(sr)
        pool_ref = pool_ref.at[1 + s * w, rec_rows : rec_rows + conv_rows].set(sc)
        pool_2s = pool_2s.at[1 + s * 2, :rec_rows].set(sr)
        pool_2s = pool_2s.at[1 + s * 2, rec_rows : rec_rows + conv_rows].set(sc)

    kw, kb, ka, kd = jax.random.split(jax.random.PRNGKey(77), 4)
    cw = jax.random.normal(kw, (dim, 1, k_sz), jnp.float32) * 0.05
    cb = jax.random.normal(kb, (dim,), jnp.float32) * 0.02
    al = jax.random.normal(ka, (n_v,), jnp.float32) * 0.1
    db = jax.random.normal(kd, (n_v,), jnp.float32) * 0.1
    qsl = jnp.array([4 * s for s in range(num_seqs + 1)], dtype=jnp.int32)
    dist = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    ro_ref = jnp.zeros((num_seqs,), dtype=jnp.int32)
    ro_2s = jnp.zeros((num_seqs,), dtype=jnp.int32)
    accepted_sched = [
        [4, 4, 4],  # all accepted -> 0 replay work
        [1, 2, 4],
        [3, 1, 2],
        [4, 4, 4],
        [2, 3, 1],
        [4, 1, 3],
    ]
    cur_lens = [32, 32, 32]
    interp = pltpu.InterpretParams(detect_races=True)
    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=_FAKE_TPU_INFO),
        mock.patch.object(typed_ldst, "load_typed", side_effect=_cpu_load_typed),
        mock.patch.object(typed_ldst, "store_typed", side_effect=_cpu_store_typed),
    ):
        for step, acc in enumerate(accepted_sched):
            seq_lens = jnp.array([c + w for c in cur_lens], dtype=jnp.int32)
            k0, k1, k2 = jax.random.split(jax.random.PRNGKey(500 + step), 3)
            qkv = jax.random.normal(k0, (num_seqs * w, dim), jnp.float32) * 0.06
            b_t = jax.random.normal(k1, (num_seqs * w, n_v), jnp.float32) * 0.08
            a_t = jax.random.normal(k2, (num_seqs * w, n_v), jnp.float32) * 0.08
            pool_ref, out_ref = gdn_wrapper.fused_conv1d_gdn(
                qkv,
                b_t,
                a_t,
                None,
                None,
                cw,
                cb,
                al,
                db,
                qsl,
                ckpt_ref[:, 0],
                dist,
                seq_lens,
                read_offsets=ro_ref,
                ckpt_indices=ckpt_ref,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=k_sz,
                state_source=pool_ref,
                state_plan=plan,
                num_spec_tokens=3,
                batched_only=True,
                interpret=interp,
            )
            pool_2s, out_2s = gdn_wrapper.fused_conv1d_gdn(
                qkv,
                b_t,
                a_t,
                None,
                None,
                cw,
                cb,
                al,
                db,
                qsl,
                ckpt_2s[:, 0],
                dist,
                seq_lens,
                read_offsets=ro_2s,
                ckpt_indices=ckpt_2s,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=k_sz,
                state_source=pool_2s,
                state_plan=plan,
                num_spec_tokens=3,
                single_checkpoint=True,
                batched_only=True,
                interpret=interp,
            )
            assert float(jnp.max(jnp.abs(out_2s - out_ref))) < 1e-6
            commit = [a if a < w else 0 for a in acc]
            if any(commit):
                pool_2s = gdn_wrapper.run_multi_layer_batched_replay_pooled(
                    qkv[None],
                    b_t[None],
                    a_t[None],
                    pool_2s,
                    cw[None],
                    cb[None],
                    al[None],
                    db[None],
                    ckpt_2s[:, 0][None],
                    ro_2s[None],
                    ckpt_2s[None],
                    jnp.array([commit], dtype=jnp.int32),
                    qsl,
                    dist,
                    seq_lens,
                    state_plan=plan,
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=k_sz,
                    num_spec_tokens=3,
                    interpret=interp,
                )
            ro_ref = jnp.array([a - 1 for a in acc], dtype=jnp.int32)
            ro_2s = 1 - ro_2s
            for s in range(num_seqs):
                ref_blk = int(ckpt_ref[s, int(ro_ref[s])])
                two_blk = int(ckpt_2s[s, int(ro_2s[s])])
                assert (
                    float(jnp.max(jnp.abs(pool_2s[two_blk] - pool_ref[ref_blk]))) < 1e-6
                )
            cur_lens = [c + a for c, a in zip(cur_lens, acc)]


def test_backbone_forward_allows_stub_without_buffers(monkeypatch):
    """Unbound `Qwen4ExpBackbone.forward(SimpleNamespace(...), ...)` stays valid."""
    from types import SimpleNamespace

    from vllm_torchtpu.models.qwen4_exp.experimental.model import Qwen4ExpBackbone

    seen: dict[str, object] = {}

    def _fake_impl(self, *args, **kwargs):
        del args
        seen["flat"] = kwargs.get("gdn_verify_flat_bufs")
        return "ok"

    monkeypatch.setattr(Qwen4ExpBackbone, "_forward_impl", _fake_impl)
    stub = SimpleNamespace()
    out = Qwen4ExpBackbone.forward(
        stub,
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        query_start_loc=None,
        ngram_context=None,
        state_slots=None,
        context_lens=None,
        valid_tokens=None,
        main_block_table=None,
        compressed_block_table=None,
    )
    assert out == "ok"
    assert seen["flat"] is None


def _ensure_kv_cache_planning_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub `vllm.v1.core.kv_cache_planning` only in venvs that predate it."""
    import importlib.util
    import sys
    import types

    if (
        "vllm.v1.core.kv_cache_planning" not in sys.modules
        and importlib.util.find_spec("vllm.v1.core.kv_cache_planning") is None
    ):
        mod = types.ModuleType("vllm.v1.core.kv_cache_planning")
        mod.get_kv_cache_config_from_groups = lambda *a, **kw: None  # type: ignore[attr-defined]
        mod.get_kv_cache_groups = lambda *a, **kw: None  # type: ignore[attr-defined]
        mod.unify_kv_cache_spec_page_size = lambda *a, **kw: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_planning", mod)


class _FakeTpuTensor(torch.Tensor):
    """CPU-backed tensor reporting `device.type == 'tpu'` for routing tests."""

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        return torch.Tensor._make_subclass(cls, data, False)

    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu")


@pytest.mark.parametrize("has_sp_flat", [False, True])
def test_precompile_two_slot_gdn_replay_shapes_dtypes_and_sp_modes(
    monkeypatch, has_sp_flat: bool
):
    """Run `_precompile_two_slot_gdn_replay` and verify every warmed shape/dtype."""
    _ensure_kv_cache_planning_stub(monkeypatch)
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_torchtpu.models.qwen4_exp.experimental import (
        pooled_cache as exp_pooled_cache,
    )
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    fake_tpu_dev = SimpleNamespace(type="tpu")
    orig_tensor, orig_zeros, orig_full = torch.tensor, torch.zeros, torch.full

    def _redirect(fn):
        def _wrapped(*args, **kwargs):
            if kwargs.get("device") is fake_tpu_dev:
                kwargs = {**kwargs, "device": "cpu"}
            return fn(*args, **kwargs)

        return _wrapped

    monkeypatch.setattr(torch, "tensor", _redirect(orig_tensor))
    monkeypatch.setattr(torch, "zeros", _redirect(orig_zeros))
    monkeypatch.setattr(torch, "full", _redirect(orig_full))
    monkeypatch.setattr(exp_pooled_cache, "synchronize_tensors", lambda *a, **kw: None)

    num_gdn = 2
    num_ple = 1
    replay_flip_calls: list[dict[str, object]] = []
    plain_replay_calls: list[dict[str, object]] = []
    boundary_v_calls: list[dict[str, object]] = []
    boundary_nv_calls: list[dict[str, object]] = []
    copy_block_calls: list[tuple[int, int]] = []
    flip_offset_calls: list[tuple[int, tuple[int, ...], bool]] = []
    per_layer_calls: list[int] = []

    class _SpyMultiRunner:
        def can_replay_and_flip_fast(self) -> bool:
            return True

        def replay_and_flip(self, **kw) -> None:
            replay_flip_calls.append(
                {
                    "target_reqs": int(kw["seq_lens"].shape[0]),
                    "qsl_len": int(kw["query_start_loc"].shape[0]),
                    "pp_s_shape": tuple(kw["state_stack"].shape),
                    "pp_c_shape": tuple(kw["ckpt_stack"].shape),
                    "nt_shape": tuple(kw["next_tokens"].shape),
                    "nt_dtype": kw["next_tokens"].dtype,
                    "has_ple": kw["ple_state_stack"] is not None,
                    "ple_shape": (
                        tuple(kw["ple_state_stack"].shape)
                        if kw["ple_state_stack"] is not None
                        else None
                    ),
                    "force_sp": kw["force_sp"],
                }
            )

        def __call__(self, **kw) -> None:
            plain_replay_calls.append(
                {
                    "target_reqs": int(kw["seq_lens"].shape[0]),
                    "cl_shape": tuple(kw["commit_lens_stack"].shape),
                    "s_shape": tuple(kw["raw_state_indices_stack"].shape),
                    "c_shape": tuple(kw["raw_ckpt_indices_stack"].shape),
                    "force_sp": kw["force_sp"],
                }
            )

    def _spy_boundary_v(*args):
        entries, slens, nt = args[0], args[1], args[2]
        boundary_v_calls.append(
            {
                "entries_rows": int(entries.shape[0]),
                "slens_len": int(slens.shape[0]),
                "nt_rows": int(nt.shape[0]),
                "nt_dtype": nt.dtype,
            }
        )
        return two_slot.compute_boundary_copy_and_interior_impl(*args)

    def _spy_boundary_nv(*args):
        entries, slens, _, _, _, has_win = args[:6]
        boundary_nv_calls.append(
            {
                "entries_rows": int(entries.shape[0]),
                "slens_len": int(slens.shape[0]),
                "has_win": bool(has_win),
            }
        )
        return two_slot.compute_boundary_copy_no_verify_impl(*args)

    def _spy_update_pp(offsets, groups, qsl, w0, w1):
        del offsets, qsl, w0, w1
        distinct = len({id(g) for g in groups}) == len(groups)
        flip_offset_calls.append((len(groups), tuple(groups[0].shape), distinct))

    monkeypatch.setattr(
        two_slot,
        "get_or_build_multi_layer_two_slot_replay",
        lambda _runner, _layers: _SpyMultiRunner(),
    )
    monkeypatch.setattr(
        two_slot, "compute_boundary_copy_and_interior_compiled", _spy_boundary_v
    )
    monkeypatch.setattr(
        two_slot, "compute_boundary_copy_no_verify_compiled", _spy_boundary_nv
    )
    monkeypatch.setattr(
        two_slot, "update_ping_pong_read_offsets_compiled", _spy_update_pp
    )

    sp_sentinel = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    gdn_layers = []
    for i in range(num_gdn):
        lin = SimpleNamespace(
            prefix=f"model.layers.{i}.linear_attn",
            _two_slot_mtp=True,
            _shared_verify_sp_flat_tensors=sp_sentinel if has_sp_flat else None,
            replay_two_slot_state=lambda **kw: per_layer_calls.append(
                int(kw["seq_lens"].shape[0])
            ),
        )
        gdn_layers.append(
            SimpleNamespace(layer_type="linear_attention", linear_attn=lin)
        )

    gdn_spec = MambaSpec(
        shapes=((3, 1, 128), (2, 128, 128)),
        dtypes=(torch.float32, torch.bfloat16),
        block_size=512,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        num_speculative_blocks=1,
    )
    ple_spec = MambaSpec(
        shapes=((2, 128),),
        dtypes=(torch.bfloat16,),
        block_size=512,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        num_speculative_blocks=3,
    )
    attn_groups = [
        [SimpleNamespace(kv_cache_group_id=i, kv_cache_spec=gdn_spec)]
        for i in range(num_gdn)
    ] + [[SimpleNamespace(kv_cache_group_id=num_gdn, kv_cache_spec=ple_spec)]]
    vllm_cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        model_config=SimpleNamespace(
            architecture="Qwen4ExpForCausalLM", hf_config=None
        ),
    )
    raw0 = torch.zeros(4, dtype=torch.float32)
    ro = torch.zeros(64, dtype=torch.int32)
    zero_ro = torch.zeros_like(ro)
    is_pp_mask = torch.tensor([True] * num_gdn + [False] * num_ple, dtype=torch.bool)
    layer_to_plan = torch.arange(num_gdn, dtype=torch.int64)
    runner = SimpleNamespace(
        enforce_eager=False,
        device=fake_tpu_dev,
        mesh=None,
        vllm_config=vllm_cfg,
        attn_groups=attn_groups,
        max_num_reqs=64,
        num_reqs_paddings=[8, 16, 32, 64],
        num_reqs_max_model_len=32,
        num_reqs_most_model_len=64,
        mamba_slot_read_offsets=ro,
        _mamba_copy_plan=[(g, [raw0]) for g in range(num_gdn + num_ple)],
        _mamba_ckpt_window=4,
        _pool_block_split=1,
        _mamba_state_block_size=512,
        _mamba_column_stride=lambda: 512,
        _bucket_len=TPUModelRunner._bucket_len,
        _get_compiled_boundary_meta=lambda _ls, _dev: (
            [raw0],
            is_pp_mask,
            layer_to_plan,
            zero_ro,
        ),
        _copy_mamba_state_blocks=lambda _raws, s, d: copy_block_calls.append(
            (int(s.shape[0]), int(d.shape[0]))
        ),
    )
    cache = object.__new__(exp_pooled_cache.Qwen4ExpPooledCache)
    cache.model = SimpleNamespace(model=SimpleNamespace(layers=gdn_layers))
    materialized = SimpleNamespace(raw_tensors=[raw0])

    cache._precompile_two_slot_gdn_replay(runner, materialized)

    expected_pairs = {
        (32, 8),
        (32, 16),
        (32, 32),
        (64, 8),
        (64, 16),
        (64, 32),
        (64, 64),
    }
    expected_modes = {False, True} if has_sp_flat else {False}

    # Every next_tokens reaching replay_and_flip and boundary copy is int32.
    assert replay_flip_calls
    assert all(c["nt_dtype"] == torch.int32 for c in replay_flip_calls)
    assert all(c["nt_dtype"] == torch.int32 for c in boundary_v_calls)
    # Exact (target_num_reqs, nt_pad) pairs warmed including mismatched pairs!
    assert {(c["target_reqs"], c["nt_shape"][0]) for c in replay_flip_calls} == (
        expected_pairs
    )
    assert {(c["slens_len"], c["nt_rows"]) for c in boundary_v_calls} == expected_pairs
    assert {c["force_sp"] for c in replay_flip_calls} == expected_modes
    assert {c["has_ple"] for c in replay_flip_calls} == {True, False}
    assert all(
        c["qsl_len"] == c["target_reqs"] + 1
        and c["pp_s_shape"] == (num_gdn, c["target_reqs"])
        and c["pp_c_shape"] == (num_gdn, c["target_reqs"], 2)
        for c in replay_flip_calls
    )

    # Plain multi_runner(...) warmed in EVERY active staging layout for BOTH
    # interior export (cl_rows == L) and 1D replay fallback (cl_rows == 1).
    assert {
        (c["target_reqs"], c["cl_shape"][0], c["force_sp"]) for c in plain_replay_calls
    } == {
        (t, cl_r, m) for t in (32, 64) for cl_r in (1, num_gdn) for m in expected_modes
    }
    assert {c["has_win"] for c in boundary_nv_calls} == {True, False}
    assert copy_block_calls
    # Both [G, target] stacked and distinct list-of-G forms warmed.
    assert any(n == 1 and shp == (num_gdn, 32) for n, shp, _ in flip_offset_calls)
    assert any(
        n == num_gdn and shp == (32,) and distinct
        for n, shp, distinct in flip_offset_calls
    )
    assert set(per_layer_calls) == {32, 64}


# ---------------------------------------------------------------------------
# Runner routing tests (CPU + TPU-gated)
# ---------------------------------------------------------------------------


def _build_cpu_pooled_gdn_layers(
    *,
    num_layers: int,
    n_kq: int,
    n_v: int,
    max_seqs: int,
    win_cap: int,
    seed: int,
    with_bias: bool = True,
):
    """Build real `VllmGatedDeltaNetAttention` instances sharing one CPU pool."""
    import numpy as np

    d_k = d_v = 128
    k_sz = 4
    dim = 2 * n_kq * d_k + n_v * d_v
    plan, total_rows, rec_rows, conv_rows = _make_pool_plan(n_v, dim, k_sz)
    pool_blocks = 2 + num_layers * max_seqs * 2
    pool_jax = jnp.zeros((pool_blocks, total_rows, 128), dtype=jnp.float32)

    s_idx = torch.zeros((num_layers, max_seqs), dtype=torch.int32)
    ck_idx = torch.zeros((num_layers, max_seqs, 2), dtype=torch.int32)
    ro_buf = torch.zeros((pool_blocks,), dtype=torch.int32)
    nxt = 1
    for li in range(num_layers):
        for s in range(max_seqs):
            s_idx[li, s] = nxt
            ck_idx[li, s, 0] = nxt
            ck_idx[li, s, 1] = nxt + 1
            ro_v = (li + s) % 2
            ro_buf[nxt] = ro_v
            read_slot = nxt + ro_v
            rk, ck = jax.random.split(jax.random.PRNGKey(seed * 100 + li * 20 + s))
            pool_jax = pool_jax.at[read_slot, :rec_rows].set(
                jax.random.normal(rk, (rec_rows, 128), jnp.float32) * 0.04
            )
            pool_jax = pool_jax.at[read_slot, rec_rows : rec_rows + conv_rows].set(
                jax.random.normal(ck, (conv_rows, 128), jnp.float32) * 0.04
            )
            nxt += 2
    pool_torch = torch.from_numpy(np.asarray(pool_jax).copy())

    keys = jax.random.split(jax.random.PRNGKey(seed), 7)
    qkv_3d = torch.from_numpy(
        np.asarray(
            jax.random.normal(keys[0], (num_layers, win_cap, dim), jnp.float32) * 0.06
        ).copy()
    )
    b_3d = torch.from_numpy(
        np.asarray(
            jax.random.normal(keys[1], (num_layers, win_cap, n_v), jnp.float32) * 0.08
        ).copy()
    )
    a_3d = torch.from_numpy(
        np.asarray(
            jax.random.normal(keys[2], (num_layers, win_cap, n_v), jnp.float32) * 0.08
        ).copy()
    )
    cw_3d = torch.from_numpy(
        np.asarray(
            jax.random.normal(keys[3], (num_layers, dim, 1, k_sz), jnp.float32) * 0.05
        ).copy()
    )
    cb_3d = (
        torch.from_numpy(
            np.asarray(
                jax.random.normal(keys[4], (num_layers, dim), jnp.float32) * 0.02
            ).copy()
        )
        if with_bias
        else None
    )
    al_2d = torch.from_numpy(
        np.asarray(
            jax.random.normal(keys[5], (num_layers, n_v), jnp.float32) * 0.1
        ).copy()
    )
    db_2d = torch.from_numpy(
        np.asarray(
            jax.random.normal(keys[6], (num_layers, n_v), jnp.float32) * 0.1
        ).copy()
    )

    interp = pltpu.InterpretParams(detect_races=True)

    def _make_per_layer_replay_op():
        def _op(
            qkv_in,
            b_in,
            a_in,
            rec_state,
            cw,
            cb,
            al,
            dtb,
            s_ids,
            qsl,
            dist,
            slens,
            slot_ro,
            ckpt_ids,
            commit_lens,
            ro_ids,
        ):
            idx_for_ro = s_ids if ro_ids is None else ro_ids
            ro_vals = slot_ro[idx_for_ro.clamp(0, slot_ro.shape[0] - 1).long()]
            new_pool, _ = gdn_wrapper.fused_conv1d_gdn(
                jnp.asarray(qkv_in.numpy()),
                jnp.asarray(b_in.numpy()),
                jnp.asarray(a_in.numpy()),
                None,
                None,
                jnp.asarray(cw.numpy()),
                None if cb is None else jnp.asarray(cb.numpy()),
                jnp.asarray(al.numpy()),
                jnp.asarray(dtb.numpy()),
                jnp.asarray(qsl.numpy()),
                jnp.asarray(s_ids.numpy()),
                jnp.asarray(dist.numpy()),
                jnp.asarray(slens.numpy()),
                read_offsets=jnp.asarray(ro_vals.numpy()),
                ckpt_indices=jnp.asarray(ckpt_ids.numpy()),
                commit_lens=jnp.asarray(commit_lens.numpy()),
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=k_sz,
                state_source=jnp.asarray(rec_state.numpy()),
                state_plan=plan,
                num_spec_tokens=3,
                single_checkpoint=True,
                batched_only=True,
                interpret=interp,
            )
            rec_state.copy_(torch.from_numpy(np.asarray(new_pool).copy()))

        return _op

    layers: list[VllmGatedDeltaNetAttention] = []
    win_stride = ((win_cap + 7) // 8) * 8
    qkv_flat = torch.zeros((num_layers * win_stride, dim), dtype=torch.float32)
    b_flat = torch.zeros((num_layers * win_stride, n_v), dtype=torch.float32)
    a_flat = torch.zeros((num_layers * win_stride, n_v), dtype=torch.float32)
    bb_ref = SimpleNamespace(_last_verify_flat_from_sp=False)
    for li in range(num_layers):
        qkv_flat[li * win_stride : li * win_stride + win_cap].copy_(qkv_3d[li])
        b_flat[li * win_stride : li * win_stride + win_cap].copy_(b_3d[li])
        a_flat[li * win_stride : li * win_stride + win_cap].copy_(a_3d[li])
        layer = object.__new__(VllmGatedDeltaNetAttention)
        nn.Module.__init__(layer)
        layer.prefix = f"model.layers.{li}.linear_attn"
        layer.num_k_heads = n_kq
        layer.num_v_heads = n_v
        layer.tp_size = 1
        layer.head_k_dim = d_k
        layer.head_v_dim = d_v
        layer.conv_kernel_size = k_sz
        layer.num_spec = 3
        layer._two_slot_mtp = True
        layer._bm_kernel = False
        layer._bm_region = None
        layer._pooled_conv_state_dtype = torch.float32
        layer._verify_win_cap = win_cap
        layer._verify_qkv_dim = dim
        layer._verify_local_n_v = n_v
        layer._verify_act_dtype = torch.float32
        layer.kv_cache = (pool_torch,)
        layer.conv1d = SimpleNamespace(
            weight=cw_3d[li],
            bias=None if cb_3d is None else cb_3d[li],
        )
        layer.A_log = al_2d[li]
        layer.dt_bias = db_2d[li]
        layer._pooled_conv_weight = lambda cw=cw_3d[li]: cw  # type: ignore[method-assign]
        layer.get_state_dtype = lambda: (torch.float32, torch.float32)  # type: ignore[method-assign]
        layer.gdn_pooled_replay_op = _make_per_layer_replay_op()
        layer.bind_shared_verify_flat(
            qkv_flat,
            b_flat,
            a_flat,
            li,
            win_stride=win_stride,
            sp_flat_tensors=None,
            backbone_ref=bb_ref,
        )
        layers.append(layer)
    return layers, pool_torch, ro_buf, s_idx, ck_idx, plan, (qkv_3d, b_3d, a_3d)


def test_runner_replay_and_interior_export_match_per_layer_on_cpu(monkeypatch):
    """Runner replay and interior boundary export match per-layer replay."""
    _ensure_kv_cache_planning_stub(monkeypatch)
    import numpy as np

    from vllm_torchtpu.models.qwen4_exp.experimental import (
        gdn_attention as exp_gdn_attn,
    )
    from vllm_torchtpu.models.qwen4_exp.experimental.ops import (
        gdn_attention_op as exp_gdn_op,
    )
    from vllm_torchtpu.models.qwen4_exp.experimental.runner.runner import Qwen38Runner

    num_layers, max_seqs, win_cap = 2, 3, 12
    layers, pool_ml, ro_buf, s_idx, ck_idx, plan, (qkv_3d, b_3d, a_3d) = (
        _build_cpu_pooled_gdn_layers(
            num_layers=num_layers,
            n_kq=1,
            n_v=1,
            max_seqs=max_seqs,
            win_cap=win_cap,
            seed=53,
        )
    )
    pool_snap = pool_ml.clone()
    qsl = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    slens = torch.tensor([12, 16, 20], dtype=torch.int32)
    # seq 0: 2 accepted (replay), seq 1: 4 accepted (no replay), seq 2: 1 accepted.
    next_tokens = torch.tensor(
        [[10, 11, -1, -1], [20, 21, 22, 23], [30, -1, -1, -1]], dtype=torch.int32
    )
    md = SimpleNamespace(draft_lengths_cpu=np.array([3, 3, 3], dtype=np.int32))
    attn_md = {
        layers[li].prefix: SimpleNamespace(
            mamba_state_indices=s_idx[li],
            mamba_ckpt_indices=ck_idx[li],
        )
        for li in range(num_layers)
    }
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    vllm_cfg = SimpleNamespace(cache_config=SimpleNamespace(block_size=512))
    interp = pltpu.InterpretParams(detect_races=True)

    orig_multi_kernel = gdn_wrapper.run_multi_layer_batched_replay_pooled

    def _cpu_interpret_multi_kernel(*args, **kwargs):
        kwargs["interpret"] = interp
        return orig_multi_kernel(*args, **kwargs)

    runner = object.__new__(Qwen38Runner)
    runner.vllm_config = vllm_cfg
    runner.mesh = mesh
    runner.device = torch.device("cpu")
    runner.mamba_slot_read_offsets = ro_buf
    runner._get_two_slot_gdn_layers = lambda: tuple(layers)  # type: ignore[method-assign]

    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=_FAKE_TPU_INFO),
        mock.patch.object(typed_ldst, "load_typed", side_effect=_cpu_load_typed),
        mock.patch.object(typed_ldst, "store_typed", side_effect=_cpu_store_typed),
        mock.patch.object(exp_gdn_attn, "_build_v3_pool_state_plan", return_value=plan),
        mock.patch.object(
            gdn_wrapper,
            "run_multi_layer_batched_replay_pooled",
            side_effect=_cpu_interpret_multi_kernel,
        ),
    ):
        # 1a. Multi-layer replay via shared flat buffers (staged_verify_inputs=None).
        chunk_flat = two_slot.PingPongMambaChunkState(
            state_indices=[s_idx[li] for li in range(num_layers)],
            attn_metadata=attn_md,
            chunk_num_windowed=max_seqs,
            num_reqs=max_seqs,
            query_start_loc=qsl,
            seq_lens=slens,
            staged_verify_inputs=None,
        )
        runner._replay_two_slot_gdn_chunk(chunk_flat, md, next_tokens)
        out_ml_flat = pool_ml.clone()
        assert float((out_ml_flat - pool_snap).abs().max()) > 0.0

        # 1b. Multi-layer replay via explicit staged_verify_inputs (multi-chunk).
        pool_ml.copy_(pool_snap)
        chunk_staged = two_slot.PingPongMambaChunkState(
            state_indices=[s_idx[li] for li in range(num_layers)],
            attn_metadata=attn_md,
            chunk_num_windowed=max_seqs,
            num_reqs=max_seqs,
            query_start_loc=qsl,
            seq_lens=slens,
            staged_verify_inputs={
                layers[li].prefix: (qkv_3d[li], b_3d[li], a_3d[li])
                for li in range(num_layers)
            },
        )
        runner._replay_two_slot_gdn_chunk(chunk_staged, md, next_tokens)
        out_ml_staged = pool_ml.clone()

        # 1c. Per-layer fallback reference (`multi_runner=None`).
        pool_ml.copy_(pool_snap)
        with mock.patch.object(
            exp_gdn_op, "build_multi_layer_pooled_replay_runner", return_value=None
        ):
            runner._two_slot_multi_replay_cache = None
            runner._replay_two_slot_gdn_chunk(chunk_flat, md, next_tokens)
        out_ref = pool_ml.clone()
        assert torch.equal(out_ml_flat, out_ref)
        assert torch.equal(out_ml_staged, out_ref)

        # 2. Interior boundary export with heterogeneous per-layer commit_lens.
        runner._two_slot_multi_replay_cache = None
        zero_ro = torch.zeros_like(ro_buf)
        s_int = s_idx.clone()
        ckpt_int = ck_idx.clone()
        cl_int = torch.tensor([[0, 2, 0], [3, 0, 1]], dtype=torch.int32)
        pool_ml.copy_(pool_snap)
        runner._export_two_slot_interior_boundaries(
            s_int, ckpt_int, cl_int, chunk_flat, zero_ro
        )
        out_int_ml = pool_ml.clone()
        assert float((out_int_ml - pool_snap).abs().max()) > 0.0

        pool_ml.copy_(pool_snap)
        with mock.patch.object(
            exp_gdn_op, "build_multi_layer_pooled_replay_runner", return_value=None
        ):
            runner._two_slot_multi_replay_cache = None
            runner._export_two_slot_interior_boundaries(
                s_int, ckpt_int, cl_int, chunk_flat, zero_ro
            )
        out_int_ref = pool_ml.clone()
        assert torch.equal(out_int_ml, out_int_ref)


@pytest.mark.parametrize("with_boundary_plan", [False, True])
@pytest.mark.parametrize("ple_case", ["stacked", "missing_stack", "no_mamba_indices"])
def test_post_verify_two_slot_gdn_chunk_routing_and_ple_done(
    monkeypatch, with_boundary_plan: bool, ple_case: str
):
    """Verify `replay_and_flip` routing, `ple_done`, boundary order, and offsets."""
    _ensure_kv_cache_planning_stub(monkeypatch)
    import numpy as np

    from vllm_torchtpu.models.qwen4_exp.experimental.runner import runner as qr
    from vllm_torchtpu.models.qwen4_exp.experimental.runner.runner import Qwen38Runner
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    num_gdn, num_ple, num_blocks = 2, 2, 64
    num_win, num_reqs = 2, 3
    qsl = _FakeTpuTensor(torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32))
    slens = _FakeTpuTensor(torch.tensor([514, 20, 32, 0], dtype=torch.int32))
    next_tokens = torch.tensor(
        [[1, 2, 3, -1], [4, 5, -1, -1], [6, -1, -1, -1], [-1, -1, -1, -1]],
        dtype=torch.int32,
    )
    gdn_s = torch.tensor([[1, 3, 5, 0], [7, 9, 11, 0]], dtype=torch.int32)
    gdn_c = torch.stack([gdn_s, torch.where(gdn_s > 0, gdn_s + 1, 0)], dim=-1)
    ple_s = torch.tensor([[20, 21, 22, 0], [30, 31, 32, 0]], dtype=torch.int32)
    ro_init = torch.zeros(num_blocks, dtype=torch.int32)
    ro_init[1] = 0
    ro_init[3] = 1
    ro_init[5] = 1
    ro_init[7] = 1
    ro_init[9] = 0
    ro_init[11] = 1

    call_log: list[str] = []
    boundary_seen_offsets: list[torch.Tensor] = []
    interior_calls: list[dict[str, object]] = []

    class _FakeMultiRunner:
        def can_replay_and_flip_fast(self) -> bool:
            return True

        def replay_and_flip(self, **kw) -> None:
            call_log.append("replay_and_flip")
            qsl_cpu = kw["query_start_loc"].as_subclass(torch.Tensor)
            nt_cpu = kw["next_tokens"]
            wr_cpu = kw["win_and_reqs"]
            q_lens = qsl_cpu[1:] - qsl_cpu[:-1]
            raw_v = (nt_cpu != two_slot.INVALID_TOKEN_ID).sum(dim=1).to(torch.int32)
            ple_off = (raw_v[: q_lens.shape[0]] - 1).clamp(min=0)
            ple_st = kw["ple_state_stack"]
            new_ro = compute_fused_flip_and_ple_offsets_jax(
                jnp.asarray(kw["slot_read_offsets"].numpy()),
                jnp.asarray(kw["state_stack"].numpy()),
                jnp.asarray(qsl_cpu.numpy()),
                jnp.asarray(wr_cpu.numpy()),
                ple_state_stack=(
                    None if ple_st is None else jnp.asarray(ple_st.numpy())
                ),
                ple_offsets_1d=None if ple_st is None else jnp.asarray(ple_off.numpy()),
            )
            kw["slot_read_offsets"].copy_(torch.from_numpy(np.asarray(new_ro).copy()))

        def __call__(self, **kw) -> None:
            call_log.append("interior_replay")
            interior_calls.append(kw)

    def _fake_boundary_compiled(*args):
        call_log.append("boundary_compute")
        boundary_seen_offsets.append(args[3].clone())
        return two_slot.compute_boundary_copy_and_interior_impl(
            args[0],
            args[1].as_subclass(torch.Tensor),
            *args[2:],
        )

    monkeypatch.setattr(
        qr,
        "get_or_build_multi_layer_two_slot_replay",
        lambda _r, _ls: _FakeMultiRunner(),
    )
    monkeypatch.setattr(
        qr, "compute_boundary_copy_and_interior_compiled", _fake_boundary_compiled
    )

    layers = tuple(
        SimpleNamespace(prefix=f"model.layers.{i}.linear_attn", _two_slot_mtp=True)
        for i in range(num_gdn)
    )
    runner = object.__new__(Qwen38Runner)
    runner.device = torch.device("cpu")
    runner.mamba_slot_read_offsets = ro_init.clone()
    runner._mamba_ckpt_window = 4
    runner._pool_block_split = 1
    runner._mamba_copy_plan = [(g, [torch.zeros(4)]) for g in range(num_gdn)]
    runner._mamba_copy_ping_pong_gids = set(range(num_gdn))
    runner._bucket_len = TPUModelRunner._bucket_len
    runner._get_two_slot_gdn_layers = lambda: layers  # type: ignore[method-assign]
    runner._two_slot_dist3 = (  # type: ignore[method-assign]
        lambda w, _d: torch.tensor([w, w, w], dtype=torch.int32)
    )
    runner._two_slot_win_and_reqs = (  # type: ignore[method-assign]
        lambda w, n, _d: torch.tensor([w, n], dtype=torch.int32)
    )
    zero_offsets = torch.zeros_like(ro_init)
    runner._get_compiled_boundary_meta = lambda _ls, _d: (  # type: ignore[method-assign]
        [torch.zeros(4)],
        torch.ones(num_gdn, dtype=torch.bool),
        torch.arange(num_gdn, dtype=torch.int64),
        zero_offsets,
    )
    runner._copy_mamba_state_blocks = lambda _r, _s, _d: call_log.append("copy_blocks")  # type: ignore[method-assign]

    chunk = two_slot.PingPongMambaChunkState(
        state_indices=[gdn_s[i] for i in range(num_gdn)],
        attn_metadata={},
        chunk_num_windowed=num_win,
        num_reqs=num_reqs,
        query_start_loc=qsl,
        seq_lens=slens,
        staged_verify_inputs=None,
        staged_ping_pong_stacks=(gdn_s, gdn_c),
        staged_std_mamba_stack=ple_s if ple_case == "stacked" else None,
    )
    mamba_indices = (
        None if ple_case == "no_mamba_indices" else [ple_s[i] for i in range(num_ple)]
    )
    md = SimpleNamespace(draft_lengths_cpu=np.array([3, 3, 0, 0], dtype=np.int32))
    boundary_plan = None
    if with_boundary_plan:
        entries = torch.zeros((8, 3 + num_gdn * 5), dtype=torch.int32)
        entries[0, :3] = torch.tensor([0, 512, 4], dtype=torch.int32)
        entries[0, 3 : 3 + 5] = torch.tensor([40, 1, 2, 0, 0], dtype=torch.int32)
        entries[0, 8 : 8 + 5] = torch.tensor([41, 7, 8, 0, 0], dtype=torch.int32)
        boundary_plan = two_slot.SpecBoundaryPlan(
            entries=entries,
            rows=None,
            seq_lens=None,
            block_size=512,
            host_entries=entries,
            chunk_seq_lens=slens,
        )

    ple_done = runner._post_verify_two_slot_gdn_chunk(
        chunk, md, next_tokens, boundary_plan, mamba_indices
    )
    expected_ple_done = ple_case in ("stacked", "no_mamba_indices")
    assert ple_done is expected_ple_done
    if not ple_done:
        runner._update_mamba_slot_read_offsets(mamba_indices, next_tokens, num_reqs)

    # Compare against the old separate GDN-flip + standard-Mamba-offset path.
    ref_runner = object.__new__(Qwen38Runner)
    ref_runner.device = torch.device("cpu")
    ref_runner.mamba_slot_read_offsets = ro_init.clone()
    ref_chunk = chunk._replace(query_start_loc=qsl.as_subclass(torch.Tensor))
    ref_runner._update_ping_pong_mamba_slot_read_offsets(ref_chunk)
    ref_runner._update_mamba_slot_read_offsets(mamba_indices, next_tokens, num_reqs)
    assert torch.equal(
        runner.mamba_slot_read_offsets, ref_runner.mamba_slot_read_offsets
    )

    if with_boundary_plan:
        assert call_log == [
            "boundary_compute",
            "replay_and_flip",
            "copy_blocks",
            "interior_replay",
        ]
        # Boundary compute saw the pre-flip read offsets.
        assert torch.equal(boundary_seen_offsets[0], ro_init)
        assert interior_calls[0]["slot_read_offsets"] is zero_offsets
    else:
        assert call_log == ["replay_and_flip"]


def test_ping_pong_layers_match_group_order_guard():
    """Layer-to-group ordering guard detects reversed KV cache group order."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_torchtpu.models.qwen4_exp.experimental.runner.metadata import (
        is_ping_pong_mamba_spec,
    )

    gdn_spec = MambaSpec(
        shapes=((3, 1, 128), (2, 128, 128)),
        dtypes=(torch.float32, torch.bfloat16),
        block_size=512,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        num_speculative_blocks=1,
    )
    vllm_cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        model_config=SimpleNamespace(
            architecture="Qwen4ExpForCausalLM", hf_config=None
        ),
    )
    l0 = SimpleNamespace(prefix="model.layers.0.linear_attn")
    l1 = SimpleNamespace(prefix="model.layers.1.linear_attn")
    ordered_runner = SimpleNamespace(
        vllm_config=vllm_cfg,
        attn_groups=[
            [
                SimpleNamespace(
                    kv_cache_group_id=0,
                    kv_cache_spec=gdn_spec,
                    layer_names=[l0.prefix],
                )
            ],
            [
                SimpleNamespace(
                    kv_cache_group_id=1,
                    kv_cache_spec=gdn_spec,
                    layer_names=[l1.prefix],
                )
            ],
        ],
    )
    reversed_runner = SimpleNamespace(
        vllm_config=vllm_cfg,
        attn_groups=list(reversed(ordered_runner.attn_groups)),
    )
    assert (
        two_slot.ping_pong_layers_match_group_order(
            ordered_runner, (l0, l1), is_ping_pong_spec=is_ping_pong_mamba_spec
        )
        is True
    )
    assert (
        two_slot.ping_pong_layers_match_group_order(
            reversed_runner, (l0, l1), is_ping_pong_spec=is_ping_pong_mamba_spec
        )
        is False
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("with_bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("commit_mode", ["1d", "per_layer", "zero"])
def test_multi_layer_batched_replay_pooled_tpu(with_bias: bool, commit_mode: str):
    """Real Mosaic Pallas TPU test for `run_multi_layer_batched_replay_pooled`.

    Exercises real hardware DMAs, `OutBufferedRef(skip_out_write=True)`, and
    dynamic VMEM indexing `weights_ref...[l_id]` against dense `fused_conv1d_gdn`.
    """
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU hardware")
    import numpy as np

    from vllm_torchtpu.kernels import pool_adapters
    from vllm_torchtpu.models.qwen4_exp.experimental.gdn_attention import (
        run_jax_gdn_multi_layer_replay_pooled,
    )

    num_layers = 3
    n_kq, n_v, d_k, d_v, k_sz = 2, 6, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v  # 1280
    max_seqs = 5
    win_rows = 4 * max_seqs
    page_tokens = 256  # ssm(bf16)=192 tokens + conv(fp32)=16 tokens <= 256
    ssm_ntok, conv_ntok = 192, 16
    pool_blocks = 2 + num_layers * max_seqs * 2
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))

    rng = np.random.default_rng(20261126 + int(with_bias) * 7)

    def _rand(shape, dtype=jnp.float32, scale=0.05):
        return jnp.asarray(rng.normal(size=shape) * scale, dtype)

    qkv_3d = _rand((num_layers, win_rows, dim), jnp.bfloat16)
    b_3d = _rand((num_layers, win_rows, n_v), jnp.bfloat16)
    a_3d = _rand((num_layers, win_rows, n_v), jnp.bfloat16)
    cw = _rand((num_layers, dim, 1, k_sz), jnp.bfloat16)
    cb = _rand((num_layers, dim), jnp.bfloat16, 0.02) if with_bias else None
    al = _rand((num_layers, n_v), jnp.float32, 0.1)
    db = _rand((num_layers, n_v), jnp.float32, 0.1) - 1.0

    qsl = jnp.array([4 * i for i in range(max_seqs + 1)], jnp.int32)
    dist = jnp.array([max_seqs, max_seqs, max_seqs], jnp.int32)
    slens = jnp.array([12, 16, 4, 24, 20], jnp.int32)

    if commit_mode == "1d":
        commit_lens_stack = jnp.array([[2, 0, 3, 1, 0]], jnp.int32)
    elif commit_mode == "per_layer":
        commit_lens_stack = jnp.array(
            [[2, 0, 1, 0, 3], [0, 3, 2, 1, 0], [1, 0, 0, 2, 1]], jnp.int32
        )
    else:
        commit_lens_stack = jnp.zeros((1, max_seqs), jnp.int32)

    pool = jnp.zeros((pool_blocks, page_tokens, 1, 4, 256), jnp.float8_e4m3fn)
    seed_ssm = _rand((num_layers, max_seqs, n_v, d_k, d_v), jnp.bfloat16, 0.04)
    seed_conv = _rand((num_layers, max_seqs, k_sz - 1, dim), jnp.float32, 0.04)
    s_idx_np = np.zeros((num_layers, max_seqs), np.int32)
    ck_idx_np = np.zeros((num_layers, max_seqs, 2), np.int32)
    ro_np = np.zeros((num_layers, max_seqs), np.int32)
    nxt = 1
    for li in range(num_layers):
        for s in range(max_seqs):
            s_idx_np[li, s] = nxt
            ck_idx_np[li, s, 0] = nxt
            ck_idx_np[li, s, 1] = nxt + 1
            ro_v = (li + s) % 2
            ro_np[li, s] = ro_v
            read_slot = jnp.array([nxt + ro_v], jnp.int32)
            pool = pool_adapters.scatter_region(
                pool,
                seed_ssm[li, s : s + 1].reshape(1, -1, 256),
                read_slot,
                tok0=0,
                ntok=ssm_ntok,
                split=1,
            )
            conv_flat = seed_conv[li, s : s + 1].reshape(1, 15, 256)
            pool = pool_adapters.scatter_region(
                pool,
                jnp.pad(conv_flat, ((0, 0), (0, 1), (0, 0))),
                read_slot,
                tok0=ssm_ntok,
                ntok=conv_ntok,
                split=1,
            )
            nxt += 2
    s_idx = jnp.asarray(s_idx_np)
    ck_idx = jnp.asarray(ck_idx_np)
    ro = jnp.asarray(ro_np)
    pool_snap = pool.copy()

    ml_pool = jax.jit(
        lambda p: run_jax_gdn_multi_layer_replay_pooled(
            qkv_3d,
            b_3d,
            a_3d,
            p,
            cw,
            cb,
            al,
            db,
            s_idx,
            ro,
            ck_idx,
            commit_lens_stack,
            qsl,
            dist,
            slens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=k_sz,
            pool_block_tokens=page_tokens,
            mesh=mesh,
            recurrent_state_dtype=jnp.bfloat16,
            conv_state_dtype=jnp.float32,
            num_spec_tokens=3,
        )
    )(pool)

    if commit_mode == "zero":
        np.testing.assert_array_equal(
            np.asarray(ml_pool.view(jnp.uint8)), np.asarray(pool_snap.view(jnp.uint8))
        )
        return

    # Independent dense reference per layer via single-layer fused_conv1d_gdn.
    def _run_dense_layer(qkv_l, b_l, a_l, cr, sr, cw_l, cb_l, al_l, db_l, ckpt_4_arr):
        return gdn_wrapper.fused_conv1d_gdn(
            qkv_l,
            b_l,
            a_l,
            cr,
            sr,
            cw_l,
            cb_l,
            al_l,
            db_l,
            qsl,
            ckpt_4_arr[:, 0],
            dist,
            slens,
            read_offsets=jnp.zeros((max_seqs,), jnp.int32),
            ckpt_indices=ckpt_4_arr,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=k_sz,
            num_spec_tokens=3,
            batched_only=True,
        )

    run_dense_jit = jax.jit(_run_dense_layer)
    for li in range(num_layers):
        cl_l = commit_lens_stack[0 if commit_lens_stack.shape[0] == 1 else li]
        dense_slots = 1 + max_seqs * 4
        conv_ref = jnp.zeros((dense_slots, k_sz - 1, dim), jnp.float32)
        ssm_ref = jnp.zeros((dense_slots, n_v, d_k, d_v), jnp.bfloat16)
        ckpt_4 = np.zeros((max_seqs, 4), np.int32)
        for s in range(max_seqs):
            base = 1 + 4 * s
            ckpt_4[s] = [base, base + 1, base + 2, base + 3]
            conv_ref = conv_ref.at[base].set(seed_conv[li, s])
            ssm_ref = ssm_ref.at[base].set(seed_ssm[li, s])
        (c_ref, s_ref), _ = run_dense_jit(
            qkv_3d[li],
            b_3d[li],
            a_3d[li],
            conv_ref,
            ssm_ref,
            cw[li],
            None if cb is None else cb[li],
            al[li],
            db[li],
            jnp.asarray(ckpt_4),
        )

        for s in range(max_seqs):
            cl = int(cl_l[s])
            r_slot = int(ck_idx_np[li, s, int(ro_np[li, s])])
            w_slot = int(ck_idx_np[li, s, 1 - int(ro_np[li, s])])
            # Read slot stays untouched.
            np.testing.assert_array_equal(
                np.asarray(ml_pool[r_slot].view(jnp.uint8)),
                np.asarray(pool_snap[r_slot].view(jnp.uint8)),
            )
            if cl == 0:
                np.testing.assert_array_equal(
                    np.asarray(ml_pool[w_slot].view(jnp.uint8)),
                    np.asarray(pool_snap[w_slot].view(jnp.uint8)),
                )
            else:
                w_idx = jnp.array([w_slot], jnp.int32)
                got_ssm = pool_adapters.gather_region(
                    ml_pool,
                    w_idx,
                    tok0=0,
                    ntok=ssm_ntok,
                    split=1,
                    out_dtype=jnp.bfloat16,
                    out_lanes=256,
                ).reshape(n_v, d_k, d_v)
                got_conv = pool_adapters.gather_region(
                    ml_pool,
                    w_idx,
                    tok0=ssm_ntok,
                    ntok=conv_ntok,
                    split=1,
                    out_dtype=jnp.float32,
                )[0, :15].reshape(k_sz - 1, dim)
                ref_slot = int(ckpt_4[s, cl - 1])
                np.testing.assert_allclose(
                    np.asarray(got_ssm, np.float32),
                    np.asarray(s_ref[ref_slot], np.float32),
                    rtol=0.02,
                    atol=0.02,
                )
                np.testing.assert_allclose(
                    np.asarray(got_conv, np.float32),
                    np.asarray(c_ref[ref_slot], np.float32),
                    rtol=0.02,
                    atol=0.02,
                )


@pytest.mark.tpu_test
@pytest.mark.parametrize("with_boundary_plan", [False, True])
@pytest.mark.parametrize("with_ple_stack", [False, True])
def test_post_verify_two_slot_gdn_chunk_tpu(
    with_boundary_plan: bool, with_ple_stack: bool
):
    """Real TPU test for `Qwen38Runner._post_verify_two_slot_gdn_chunk` fast path."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU hardware")
    import numpy as np
    from torch_tpu._internal import sync

    from vllm_torchtpu.models.qwen4_exp.experimental.runner.runner import Qwen38Runner
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    tpu = torch.device("tpu")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    num_layers = 2
    n_kq, n_v, d_k, d_v, k_sz = 2, 6, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    num_win, num_reqs = 2, 3
    win_cap, win_stride = 16, 16
    page_tokens = 256
    pool_blocks = 40
    vllm_cfg = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=page_tokens),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, architecture="Qwen4ExpForCausalLM", hf_config=None
        ),
    )

    rng = torch.Generator(device="cpu").manual_seed(2026)
    pool_init = torch.zeros(
        (pool_blocks, page_tokens, 1, 4, 256), dtype=torch.float8_e4m3fn
    ).to(tpu)
    qkv_flat = (
        (torch.randn((num_layers * win_stride, dim), generator=rng) * 0.05)
        .to(torch.bfloat16)
        .to(tpu)
    )
    b_flat = (
        (torch.randn((num_layers * win_stride, n_v), generator=rng) * 0.05)
        .to(torch.bfloat16)
        .to(tpu)
    )
    a_flat = (
        (torch.randn((num_layers * win_stride, n_v), generator=rng) * 0.05)
        .to(torch.bfloat16)
        .to(tpu)
    )
    bb_ref = SimpleNamespace(_last_verify_flat_from_sp=False)

    layers = []
    with set_vllm_model_wrapper_context(vllm_config=vllm_cfg, mesh=mesh):
        for li in range(num_layers):
            layer = object.__new__(VllmGatedDeltaNetAttention)
            nn.Module.__init__(layer)
            layer.prefix = f"model.layers.{li}.linear_attn"
            layer.num_k_heads = n_kq
            layer.num_v_heads = n_v
            layer.tp_size = 1
            layer.head_k_dim = d_k
            layer.head_v_dim = d_v
            layer.conv_kernel_size = k_sz
            layer.num_spec = 3
            layer._two_slot_mtp = True
            layer._bm_kernel = False
            layer._bm_region = None
            layer._pooled_conv_state_dtype = torch.float32
            layer._verify_win_cap = win_cap
            layer._verify_qkv_dim = dim
            layer._verify_local_n_v = n_v
            layer._verify_act_dtype = torch.bfloat16
            layer.model_config = vllm_cfg.model_config
            layer.cache_config = SimpleNamespace(
                mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="bfloat16"
            )
            layer.kv_cache = (pool_init,)
            cw = (
                (torch.randn((dim, 1, k_sz), generator=rng) * 0.05)
                .to(torch.bfloat16)
                .to(tpu)
            )
            layer.conv1d = SimpleNamespace(weight=cw, bias=None)
            layer.A_log = (torch.randn((n_v,), generator=rng) * 0.1).to(tpu)
            layer.dt_bias = (torch.randn((n_v,), generator=rng) * 0.1 - 1.0).to(tpu)
            layer._pooled_conv_weight = lambda cw=cw: cw  # type: ignore[method-assign]
            layer.bind_shared_verify_flat(
                qkv_flat,
                b_flat,
                a_flat,
                li,
                win_stride=win_stride,
                sp_flat_tensors=None,
                backbone_ref=bb_ref,
            )
            layers.append(layer)

    ro_cpu = torch.zeros(pool_blocks, dtype=torch.int32)
    ro_cpu[1] = 0
    ro_cpu[3] = 1
    ro_cpu[7] = 1
    ro_cpu[9] = 0
    gdn_s_cpu = torch.tensor([[1, 3, 5, 0], [7, 9, 11, 0]], dtype=torch.int32)
    gdn_c_cpu = torch.stack(
        [gdn_s_cpu, torch.where(gdn_s_cpu > 0, gdn_s_cpu + 1, 0)], dim=-1
    )
    ple_s_cpu = torch.tensor([[20, 21, 22, 0]], dtype=torch.int32)
    qsl_cpu = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32)
    slens_cpu = torch.tensor([514, 20, 32, 0], dtype=torch.int32)
    nt_cpu = torch.tensor(
        [[1, 2, 3, -1], [4, 5, -1, -1], [6, -1, -1, -1], [-1, -1, -1, -1]],
        dtype=torch.int32,
    )

    runner = object.__new__(Qwen38Runner)
    runner.device = tpu
    runner.vllm_config = vllm_cfg
    runner.mesh = mesh
    runner.mamba_slot_read_offsets = ro_cpu.to(tpu)
    runner._mamba_ckpt_window = 4
    runner._pool_block_split = 1
    runner._mamba_copy_plan = [(g, [pool_init]) for g in range(num_layers)]
    runner._mamba_copy_ping_pong_gids = set(range(num_layers))
    runner._bucket_len = TPUModelRunner._bucket_len
    runner._get_two_slot_gdn_layers = lambda: tuple(layers)  # type: ignore[method-assign]
    runner._copy_mamba_state_blocks = lambda _raws, _src, _dst: None  # type: ignore[method-assign]
    zero_ro = torch.zeros_like(runner.mamba_slot_read_offsets)
    runner._get_compiled_boundary_meta = lambda _ls, _d: (  # type: ignore[method-assign]
        [pool_init],
        torch.ones(num_layers, dtype=torch.bool, device=tpu),
        torch.arange(num_layers, dtype=torch.int64, device=tpu),
        zero_ro,
    )

    gdn_s_tpu = gdn_s_cpu.to(tpu)
    gdn_c_tpu = gdn_c_cpu.to(tpu)
    ple_s_tpu = ple_s_cpu.to(tpu)
    chunk = two_slot.PingPongMambaChunkState(
        state_indices=[gdn_s_tpu[i] for i in range(num_layers)],
        attn_metadata={},
        chunk_num_windowed=num_win,
        num_reqs=num_reqs,
        query_start_loc=qsl_cpu.to(tpu),
        seq_lens=slens_cpu.to(tpu),
        staged_verify_inputs=None,
        staged_ping_pong_stacks=(gdn_s_tpu, gdn_c_tpu),
        staged_std_mamba_stack=ple_s_tpu if with_ple_stack else None,
    )
    mamba_indices = [ple_s_tpu[0]]
    md = SimpleNamespace(draft_lengths_cpu=np.array([3, 3, 0, 0], dtype=np.int32))
    boundary_plan = None
    if with_boundary_plan:
        entries_cpu = torch.zeros((8, 3 + num_layers * 5), dtype=torch.int32)
        entries_cpu[0, :3] = torch.tensor([0, 512, 4], dtype=torch.int32)
        entries_cpu[0, 3 : 3 + 5] = torch.tensor([30, 1, 2, 0, 0], dtype=torch.int32)
        entries_cpu[0, 8 : 8 + 5] = torch.tensor([31, 7, 8, 0, 0], dtype=torch.int32)
        boundary_plan = two_slot.SpecBoundaryPlan(
            entries=entries_cpu.to(tpu),
            rows=None,
            seq_lens=None,
            block_size=512,
            host_entries=entries_cpu,
            chunk_seq_lens=chunk.seq_lens,
        )

    nt_tpu = nt_cpu.to(tpu)
    ple_done = runner._post_verify_two_slot_gdn_chunk(
        chunk, md, nt_tpu, boundary_plan, mamba_indices
    )
    assert ple_done is with_ple_stack
    if not ple_done:
        runner._update_mamba_slot_read_offsets(mamba_indices, nt_tpu, num_reqs)
    sync.synchronize([pool_init, runner.mamba_slot_read_offsets], wait=True)

    # Compare device read offsets against separate CPU reference updates.
    ref_runner = object.__new__(Qwen38Runner)
    ref_runner.device = torch.device("cpu")
    ref_runner.mamba_slot_read_offsets = ro_cpu.clone()
    ref_chunk = two_slot.PingPongMambaChunkState(
        state_indices=[gdn_s_cpu[i] for i in range(num_layers)],
        attn_metadata={},
        chunk_num_windowed=num_win,
        num_reqs=num_reqs,
        query_start_loc=qsl_cpu,
        seq_lens=slens_cpu,
        staged_ping_pong_stacks=(gdn_s_cpu, gdn_c_cpu),
    )
    ref_runner._update_ping_pong_mamba_slot_read_offsets(ref_chunk)
    ref_runner._update_mamba_slot_read_offsets([ple_s_cpu[0]], nt_cpu, num_reqs)
    assert torch.equal(
        runner.mamba_slot_read_offsets.cpu(), ref_runner.mamba_slot_read_offsets
    )
