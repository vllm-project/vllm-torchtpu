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

import contextlib
import io
import os
import subprocess
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax._src.pallas.mosaic.interpret import interpret_pallas_call
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2.host import WeightFormat
from vllm_torchtpu.kernels.fused_moe.v2.kernel import build_combine_kernel
from vllm_torchtpu.kernels.fused_moe.v2.layer import (
    fused_ep_moe_v2,
    rebalance_ep_tp_pair_experts,
)


def _build_routing_inputs(ids: np.ndarray, *, ep: int, e_total: int, n_blocks: int = 4):
    t_total, topk = ids.shape
    t_local = t_total // ep
    pairs_g = jnp.asarray(ids.reshape(ep, t_local * topk), dtype=jnp.int32)
    rows_by_dest = jnp.stack(
        [
            jnp.bincount(
                jnp.where(
                    (pairs_g[r] >= 0) & (pairs_g[r] < e_total),
                    pairs_g[r],
                    jnp.int32(e_total),
                ),
                length=e_total + 1,
            )[:e_total]
            for r in range(ep)
        ],
        axis=0,
    ).astype(jnp.int32)
    return pairs_g, rows_by_dest, t_local, topk, n_blocks


def _assert_histograms_match_remapped(
    new_pairs: np.ndarray,
    new_local: np.ndarray,
    new_bh: np.ndarray,
    new_rbd: np.ndarray,
    *,
    e_plan: int,
):
    ep = new_pairs.shape[0]
    exp_rbd = np.stack(
        [
            np.bincount(
                np.where(
                    (new_pairs[r] >= 0) & (new_pairs[r] < e_plan),
                    new_pairs[r],
                    e_plan,
                ),
                minlength=e_plan + 1,
            )[:e_plan]
            for r in range(ep)
        ],
        axis=0,
    )
    exp_bh = np.stack(
        [
            np.bincount(
                np.where(
                    (new_local[b] >= 0) & (new_local[b] < e_plan),
                    new_local[b],
                    e_plan,
                ),
                minlength=e_plan + 1,
            )[:e_plan]
            for b in range(new_local.shape[0])
        ],
        axis=0,
    )
    np.testing.assert_array_equal(new_rbd, exp_rbd)
    np.testing.assert_array_equal(new_bh, exp_bh)


def test_should_keep_balanced_routing_inactive_when_z_clone_enabled():
    ep, g_local, k_split, topk, t_local = 8, 4, 2, 2, 64
    e_total = ep * g_local
    rng = np.random.default_rng(10)
    ids = rng.integers(0, e_total, size=(ep * t_local, topk), dtype=np.int32)
    pairs_g, rbd, t_local, topk, n_blocks = _build_routing_inputs(
        ids, ep=ep, e_total=e_total
    )
    local_blocks = pairs_g[0].reshape(n_blocks, -1)
    block_hist = jnp.stack(
        [jnp.bincount(local_blocks[b], length=e_total) for b in range(n_blocks)], axis=0
    ).astype(jnp.int32)

    new_pairs, new_local, new_bh, new_rbd, tp_phys, z_push = (
        rebalance_ep_tp_pair_experts(
            pairs_g,
            local_blocks,
            block_hist,
            rbd,
            ep=ep,
            ep_tp_size=2,
            g_local=g_local,
            e_total=e_total,
            k_split=k_split,
            capacity=16,
            me=jnp.int32(0),
            t_local=t_local,
            topk=topk,
            z_clone=True,
            z_min_total=512,
            z_min_remote=256,
        )
    )

    g_plan = g_local + k_split + 1
    e_plan = ep * g_plan
    assert np.all(np.asarray(z_push) == -1)
    slot_z_counts = np.asarray(new_rbd).reshape(ep, ep, g_plan)[:, :, g_plan - 1]
    assert int(slot_z_counts.sum()) == 0
    _assert_histograms_match_remapped(
        np.asarray(new_pairs),
        np.asarray(new_local),
        np.asarray(new_bh),
        np.asarray(new_rbd),
        e_plan=e_plan,
    )
    assert tp_phys.shape == (ep // 2, 2 * g_plan)


def test_should_offload_only_remote_host_tokens_to_z_partner_when_expert_hot():
    ep, g_local, k_split, topk, t_local = 16, 4, 2, 2, 32
    e_total = ep * g_local
    rng = np.random.default_rng(11)
    ids = rng.integers(0, e_total, size=(ep * t_local, topk), dtype=np.int32)
    # Force chip 0 physical expert 3 to own every col-0 token on both hosts, plus
    # local-host load on chip 0 expert 1 so full 8-rank remote offload is optimal.
    ids[:, 0] = 3
    ids[: (ep // 2) * t_local, 1] = 1
    pairs_g, rbd, t_local, topk, n_blocks = _build_routing_inputs(
        ids, ep=ep, e_total=e_total
    )
    g_plan = g_local + k_split + 1
    e_plan = ep * g_plan
    slot_z = g_plan - 1

    for me_val in (0, 1, 8, 9):
        local_blocks = pairs_g[me_val].reshape(n_blocks, -1)
        block_hist = jnp.stack(
            [jnp.bincount(local_blocks[b], length=e_total) for b in range(n_blocks)],
            axis=0,
        ).astype(jnp.int32)
        new_pairs, new_local, new_bh, new_rbd, tp_phys, z_push = (
            rebalance_ep_tp_pair_experts(
                pairs_g,
                local_blocks,
                block_hist,
                rbd,
                ep=ep,
                ep_tp_size=2,
                g_local=g_local,
                e_total=e_total,
                k_split=k_split,
                capacity=16,
                me=jnp.int32(me_val),
                t_local=t_local,
                topk=topk,
                z_clone=True,
                z_min_total=128,
                z_min_remote=64,
            )
        )
        zp = np.asarray(z_push)
        assert zp[0] == 3
        assert np.all(zp[1:] == -1)
        _assert_histograms_match_remapped(
            np.asarray(new_pairs),
            np.asarray(new_local),
            np.asarray(new_bh),
            np.asarray(new_rbd),
            e_plan=e_plan,
        )
        rbd_3d = np.asarray(new_rbd).reshape(ep, ep, g_plan)
        # Host 0 ranks (0..7) must send zero rows to slot_z; Host 1 ranks (8..15)
        # must route all expert-3 tokens strictly to chip 4 (cores 8 and 9) slot_z.
        assert int(rbd_3d[:8, :, slot_z].sum()) == 0
        assert int(rbd_3d[8:, 8, slot_z].sum()) > 0
        assert int(rbd_3d[8:, 9, slot_z].sum()) > 0
        assert int(rbd_3d[8:, 8, slot_z].sum() + rbd_3d[8:, 9, slot_z].sum()) == int(
            rbd[8:, 3].sum()
        )
        assert np.all(np.asarray(tp_phys)[:, slot_z] == 2 * g_local)
        assert np.all(np.asarray(tp_phys)[:, g_plan + slot_z] == 2 * g_local)


def test_should_support_bidirectional_z_clone_and_zero_local_host_tokens():
    ep, g_local, k_split, topk, t_local = 16, 4, 2, 2, 32
    e_total = ep * g_local
    rng = np.random.default_rng(12)
    ids = rng.integers(0, e_total, size=(ep * t_local, topk), dtype=np.int32)
    # Chip 0 expert 2 has ZERO Host-0 tokens and 100% Host-1 tokens (tests
    # _subtract_z_orig_local when owner chip has zero local tokens to split).
    ids[: (ep // 2) * t_local, :] = np.where(
        ids[: (ep // 2) * t_local, :] == 2, 10, ids[: (ep // 2) * t_local, :]
    )
    ids[(ep // 2) * t_local :, 0] = 2
    # Simultaneously chip 4 (Z-partner of chip 0) expert 35 (chip 4 phys 3) is
    # hot across both hosts -> tests z_waterfill=False bidirectional safety.
    ids[:, 1] = 4 * (2 * g_local) + 3
    pairs_g, rbd, t_local, topk, n_blocks = _build_routing_inputs(
        ids, ep=ep, e_total=e_total
    )
    g_plan = g_local + k_split + 1
    e_plan = ep * g_plan

    for me_val in (0, 8):
        local_blocks = pairs_g[me_val].reshape(n_blocks, -1)
        block_hist = jnp.stack(
            [jnp.bincount(local_blocks[b], length=e_total) for b in range(n_blocks)],
            axis=0,
        ).astype(jnp.int32)
        new_pairs, new_local, new_bh, new_rbd, _tp_phys, z_push = (
            rebalance_ep_tp_pair_experts(
                pairs_g,
                local_blocks,
                block_hist,
                rbd,
                ep=ep,
                ep_tp_size=2,
                g_local=g_local,
                e_total=e_total,
                k_split=k_split,
                capacity=16,
                me=jnp.int32(me_val),
                t_local=t_local,
                topk=topk,
                z_clone=True,
                z_min_total=128,
                z_min_remote=64,
                z_waterfill=False,
            )
        )
        zp = np.asarray(z_push)
        assert zp[0] == 2
        assert zp[4] == 3
        assert np.all(np.asarray(new_rbd) >= 0)
        assert np.all(np.asarray(new_bh) >= 0)
        _assert_histograms_match_remapped(
            np.asarray(new_pairs),
            np.asarray(new_local),
            np.asarray(new_bh),
            np.asarray(new_rbd),
            e_plan=e_plan,
        )


@pytest.mark.parametrize(
    ("partner_rows", "expected_chip0_push"),
    [
        (512, -1),
        (97, -1),
        (96, 2),
    ],
)
def test_should_gate_z_push_on_partner_load_gap_boundary(
    partner_rows: int,
    expected_chip0_push: int,
):
    ep, g_local, k_split, topk, t_local = 16, 4, 2, 2, 32
    e_total = ep * g_local
    rng = np.random.default_rng(21)
    ids = rng.integers(8, 24, size=(ep * t_local, topk), dtype=np.int32)
    ids[: 2 * t_local, 0] = 2
    ids[8 * t_local : 13 * t_local, 0] = 2
    partner_exp = 4 * (2 * g_local) + 3
    if partner_rows == 512:
        ids[:, 1] = partner_exp
    else:
        ids[8 * t_local : 8 * t_local + partner_rows, 1] = partner_exp
    pairs_g, rbd, t_local, topk, n_blocks = _build_routing_inputs(
        ids, ep=ep, e_total=e_total
    )
    g_plan = g_local + k_split + 1
    e_plan = ep * g_plan

    for me_val in (0, 8, 12):
        local_blocks = pairs_g[me_val].reshape(n_blocks, -1)
        block_hist = jnp.stack(
            [jnp.bincount(local_blocks[b], length=e_total) for b in range(n_blocks)],
            axis=0,
        ).astype(jnp.int32)
        new_pairs, new_local, new_bh, new_rbd, _tp_phys, z_push = (
            rebalance_ep_tp_pair_experts(
                pairs_g,
                local_blocks,
                block_hist,
                rbd,
                ep=ep,
                ep_tp_size=2,
                g_local=g_local,
                e_total=e_total,
                k_split=k_split,
                capacity=16,
                me=jnp.int32(me_val),
                t_local=t_local,
                topk=topk,
                z_clone=True,
                z_min_total=128,
                z_min_remote=64,
            )
        )
        zp = np.asarray(z_push)
        assert zp[0] == expected_chip0_push
        if partner_rows == 512:
            assert zp[4] == 3
        _assert_histograms_match_remapped(
            np.asarray(new_pairs),
            np.asarray(new_local),
            np.asarray(new_bh),
            np.asarray(new_rbd),
            e_plan=e_plan,
        )


def test_should_partially_waterfill_remote_ranks_when_hot_tokens_all_remote():
    ep, g_local, k_split, topk, t_local = 16, 4, 2, 2, 32
    e_total = ep * g_local
    rng = np.random.default_rng(22)
    ids = rng.integers(8, 24, size=(ep * t_local, topk), dtype=np.int32)
    ids[(ep // 2) * t_local :, 0] = 5
    pairs_g, rbd, t_local, topk, n_blocks = _build_routing_inputs(
        ids, ep=ep, e_total=e_total
    )
    g_plan = g_local + k_split + 1
    e_plan = ep * g_plan
    slot_z = g_plan - 1
    total_remote = int(rbd[8:, 5].sum())

    for me_val in (0, 8, 11, 15):
        local_blocks = pairs_g[me_val].reshape(n_blocks, -1)
        block_hist = jnp.stack(
            [jnp.bincount(local_blocks[b], length=e_total) for b in range(n_blocks)],
            axis=0,
        ).astype(jnp.int32)
        new_pairs, new_local, new_bh, new_rbd, _tp_phys, z_push = (
            rebalance_ep_tp_pair_experts(
                pairs_g,
                local_blocks,
                block_hist,
                rbd,
                ep=ep,
                ep_tp_size=2,
                g_local=g_local,
                e_total=e_total,
                k_split=k_split,
                capacity=16,
                me=jnp.int32(me_val),
                t_local=t_local,
                topk=topk,
                z_clone=True,
                z_min_total=64,
                z_min_remote=32,
            )
        )
        zp = np.asarray(z_push)
        assert zp[0] == 5
        assert zp[4] == -1
        rbd_3d = np.asarray(new_rbd).reshape(ep, ep, g_plan)
        moved = int(rbd_3d[8:, 8:10, slot_z].sum())
        kept = int(rbd_3d[:, 0:2, :].sum())
        assert 64 <= moved < total_remote
        assert kept + moved == total_remote
        assert max(kept, moved) < total_remote
        _assert_histograms_match_remapped(
            np.asarray(new_pairs),
            np.asarray(new_local),
            np.asarray(new_bh),
            np.asarray(new_rbd),
            e_plan=e_plan,
        )


@pytest.mark.parametrize("ep", [12, 32, 68])
def test_should_conserve_waterfill_histograms_for_non_pow2_and_wide_ep(ep: int):
    g_local, k_split, topk, t_local = 4, 2, 2, 16
    e_total = ep * g_local
    rng = np.random.default_rng(30 + ep)
    ids = rng.integers(8, 24, size=(ep * t_local, topk), dtype=np.int32)
    ids[(ep // 2) * t_local :, 0] = 3
    pairs_g, rbd, t_local, topk, n_blocks = _build_routing_inputs(
        ids, ep=ep, e_total=e_total
    )
    g_plan = g_local + k_split + 1
    e_plan = ep * g_plan
    me_val = ep // 2
    local_blocks = pairs_g[me_val].reshape(n_blocks, -1)
    block_hist = jnp.stack(
        [jnp.bincount(local_blocks[b], length=e_total) for b in range(n_blocks)],
        axis=0,
    ).astype(jnp.int32)

    new_pairs, new_local, new_bh, new_rbd, _tp_phys, z_push = (
        rebalance_ep_tp_pair_experts(
            pairs_g,
            local_blocks,
            block_hist,
            rbd,
            ep=ep,
            ep_tp_size=2,
            g_local=g_local,
            e_total=e_total,
            k_split=k_split,
            capacity=8,
            me=jnp.int32(me_val),
            t_local=t_local,
            topk=topk,
            z_clone=True,
            z_min_total=32,
            z_min_remote=16,
        )
    )
    assert int(np.asarray(z_push)[0]) == 3
    _assert_histograms_match_remapped(
        np.asarray(new_pairs),
        np.asarray(new_local),
        np.asarray(new_bh),
        np.asarray(new_rbd),
        e_plan=e_plan,
    )


@pytest.mark.parametrize("k_split", [0, 2])
def test_should_preserve_sentinels_with_z_clone_across_k_split(k_split: int):
    ep, g_local, topk, t_local = 8, 4, 2, 32
    e_total = ep * g_local
    rng = np.random.default_rng(13 + k_split)
    ids = rng.integers(0, e_total, size=(ep * t_local, topk), dtype=np.int32)
    ids[:, 0] = 1
    # Mark masked sentinel rows at e_total (non-routed padding tokens).
    ids[0, :] = e_total
    ids[-1, :] = e_total
    pairs_g, rbd, t_local, topk, n_blocks = _build_routing_inputs(
        ids, ep=ep, e_total=e_total
    )
    g_plan = g_local + k_split + 1
    e_plan = ep * g_plan
    local_blocks = pairs_g[0].reshape(n_blocks, -1)
    block_hist = jnp.stack(
        [
            jnp.bincount(
                jnp.where(local_blocks[b] < e_total, local_blocks[b], e_total),
                length=e_total + 1,
            )[:e_total]
            for b in range(n_blocks)
        ],
        axis=0,
    ).astype(jnp.int32)

    new_pairs, new_local, new_bh, new_rbd, _tp_phys, z_push = (
        rebalance_ep_tp_pair_experts(
            pairs_g,
            local_blocks,
            block_hist,
            rbd,
            ep=ep,
            ep_tp_size=2,
            g_local=g_local,
            e_total=e_total,
            k_split=k_split,
            capacity=16,
            me=jnp.int32(0),
            t_local=t_local,
            topk=topk,
            z_clone=True,
            z_min_total=64,
            z_min_remote=32,
        )
    )

    assert int(z_push[0]) == 1
    np_pairs = np.asarray(new_pairs)
    assert np.all(np_pairs[0, :topk] == e_plan)
    assert np.all(np_pairs[-1, -topk:] == e_plan)
    _assert_histograms_match_remapped(
        np_pairs,
        np.asarray(new_local),
        np.asarray(new_bh),
        np.asarray(new_rbd),
        e_plan=e_plan,
    )


@pytest.mark.parametrize("z_clone", [True, False], ids=["z_on", "z_off"])
def test_should_lower_fused_ep_moe_v2_z_clone_to_tpu7x_mlir(z_clone: bool):
    fake_info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    mesh = AbstractMesh((16,), ("ep",))
    sh = NamedSharding(mesh, P("ep"))

    def trace_fn(x, w1, w2, w1s, w2s, g, r):
        return fused_ep_moe_v2(
            x,
            w1,
            w2,
            w1s,
            w2s,
            g,
            topk=8,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            weight_format=WeightFormat.FP8,
            rank=r,
            sharded_plan=True,
            scoring_fn="sigmoid",
            ep_tp_size=2,
            ep_tp_row_splits=4,
            ep_tp_z_clone=z_clone,
            ep_tp_z_clone_delay=4,
        )

    def sds(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sh)

    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=fake_info),
        mock.patch.object(tpu_info, "get_tpu_info", return_value=fake_info),
    ):
        exported = jax.export.export(jax.jit(trace_fn), platforms=("tpu",))(
            sds((16384, 6144), jnp.bfloat16),
            sds((512, 6144, 2048), jnp.float8_e4m3fn),
            sds((512, 1024, 6144), jnp.float8_e4m3fn),
            sds((512, 2048), jnp.float32),
            sds((512, 6144), jnp.float32),
            sds((16384, 256), jnp.float32),
            sds((16, 1), jnp.int32),
        )
    mlir = exported.mlir_module()
    assert ("_zclone" in mlir) is z_clone
    assert "partition_id" not in mlir


@pytest.mark.parametrize("z_waterfill", [True, False], ids=["waterfill", "legacy"])
def test_should_keep_z_clone_inactive_when_remote_rows_zero_and_min_remote_zero(
    z_waterfill: bool,
):
    ep, g_local, k_split, topk, t_local = 8, 2, 1, 2, 32
    e_total = ep * g_local
    ids = np.zeros((ep * t_local, topk), dtype=np.int32)
    # Host 0 tokens (ranks 0..3) hit chip 0 expert 0; Host 1 tokens hit chip 2 expert 8.
    ids[: (ep // 2) * t_local, :] = 0
    ids[(ep // 2) * t_local :, :] = 8
    pairs_g, rbd, t_local, topk, n_blocks = _build_routing_inputs(
        ids, ep=ep, e_total=e_total
    )
    local_blocks = pairs_g[0].reshape(n_blocks, -1)
    block_hist = jnp.stack(
        [jnp.bincount(local_blocks[b], length=e_total) for b in range(n_blocks)], axis=0
    ).astype(jnp.int32)

    _, _, _, new_rbd, _, z_push = rebalance_ep_tp_pair_experts(
        pairs_g,
        local_blocks,
        block_hist,
        rbd,
        ep=ep,
        ep_tp_size=2,
        g_local=g_local,
        e_total=e_total,
        k_split=k_split,
        capacity=16,
        me=jnp.int32(0),
        t_local=t_local,
        topk=topk,
        z_clone=True,
        z_min_total=16,
        z_min_remote=0,
        z_waterfill=z_waterfill,
    )
    g_plan = g_local + k_split + 1
    assert np.all(np.asarray(z_push) == -1)
    assert int(np.asarray(new_rbd).reshape(ep, ep, g_plan)[:, :, g_plan - 1].sum()) == 0


def test_should_produce_identical_output_when_combine_kernel_receives_1d_or_2d_tables():
    fake_info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    t_local, topk, lane_blocks, lanes = 256, 2, 1, 128
    hidden = lane_blocks * lanes
    rng = np.random.default_rng(77)
    arr_f32 = rng.standard_normal((512, lane_blocks, lanes)).astype(np.float32) * 0.5
    pos_2d = jnp.asarray(rng.integers(0, 512, size=(t_local, topk), dtype=np.int32))
    coef_2d = jnp.asarray(
        (rng.uniform(0.1, 1.3, size=(t_local, topk))).astype(np.float32)
    )
    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=fake_info),
        mock.patch.object(tpu_info, "get_tpu_info", return_value=fake_info),
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)),
    ):
        arr = jnp.asarray(arr_f32).astype(jnp.float8_e4m3fn)
        combine = build_combine_kernel(
            t_local=t_local,
            topk=topk,
            hidden=hidden,
            lane_blocks=lane_blocks,
            wire_dtype=jnp.float8_e4m3fn,
            out_dtype=jnp.bfloat16,
        )
        with pytest.raises(ValueError, match="token-major"):
            combine(arr, coef_2d[:, :1], pos_2d)
        pltpu.reset_tpu_interpret_mode_state()
        with contextlib.redirect_stdout(io.StringIO()):
            out_2d = np.asarray(combine(arr, coef_2d, pos_2d)).view(np.uint16)
        assert not interpret_pallas_call.races.races_found
        pltpu.reset_tpu_interpret_mode_state()
        with contextlib.redirect_stdout(io.StringIO()):
            out_1d = np.asarray(
                combine(arr, coef_2d.reshape(-1), pos_2d.reshape(-1))
            ).view(np.uint16)
        assert not interpret_pallas_call.races.races_found
    np.testing.assert_array_equal(out_1d, out_2d)


def _run_two_layer_z_clone_race_and_parity_check() -> None:
    fake_info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    devs = sorted(jax.devices("cpu"), key=lambda d: d.id)[:8]
    assert len(devs) == 8
    mesh = Mesh(np.asarray(devs), ("d",))
    shd = NamedSharding(mesh, P("d"))
    t_tot, hid, inter_sl, g_loc = 8 * 16, 128, 128, 2
    e_tot = 8 * g_loc
    w_exp = 2 * e_tot

    x_raw = jax.random.normal(jax.random.PRNGKey(1), (t_tot, hid), jnp.bfloat16) * 0.3
    x_raw = x_raw.at[0, :].set(jnp.bfloat16(jnp.nan))
    x_m = jax.device_put(x_raw, shd)
    w1_m = jax.device_put(
        jax.random.normal(
            jax.random.PRNGKey(2), (w_exp, hid, 2 * inter_sl), jnp.bfloat16
        ).astype(jnp.float8_e4m3fn),
        shd,
    )
    w2_m = jax.device_put(
        jax.random.normal(
            jax.random.PRNGKey(3), (w_exp, inter_sl, hid), jnp.bfloat16
        ).astype(jnp.float8_e4m3fn),
        shd,
    )
    s1_m = jax.device_put(
        jnp.abs(jax.random.normal(jax.random.PRNGKey(4), (w_exp, 2 * inter_sl))) * 0.02
        + 1e-4,
        shd,
    )
    s2_m = jax.device_put(
        jnp.abs(jax.random.normal(jax.random.PRNGKey(5), (w_exp, hid))) * 0.02 + 1e-4,
        shd,
    )
    skew1 = jnp.zeros((e_tot,), dtype=jnp.float32).at[0].set(9.0).at[8].set(-20.0)
    skew2 = jnp.zeros((e_tot,), dtype=jnp.float32).at[12].set(9.0).at[4].set(-20.0)
    gt1 = jax.device_put(
        (
            jax.random.normal(jax.random.PRNGKey(6), (t_tot, e_tot), jnp.float32) * 0.2
            + skew1[None, :]
        )
        .at[0, :]
        .set(-jnp.inf),
        shd,
    )
    gt2 = jax.device_put(
        (
            jax.random.normal(jax.random.PRNGKey(7), (t_tot, e_tot), jnp.float32) * 0.2
            + skew2[None, :]
        )
        .at[0, :]
        .set(-jnp.inf),
        shd,
    )
    rk8 = jax.device_put(jnp.arange(8, dtype=jnp.int32).reshape(8, 1), shd)

    def _run_pair(z_clone: int) -> np.ndarray:
        def _fn(x_in):
            y1 = fused_ep_moe_v2(
                x_in,
                w1_m,
                w2_m,
                s1_m,
                s2_m,
                gt1,
                topk=4,
                renormalize=True,
                mesh=mesh,
                capacity=16,
                rank=rk8,
                weight_format=WeightFormat.FP8,
                sharded_plan=True,
                scoring_fn="sigmoid",
                ep_tp_size=2,
                ep_tp_row_splits=2,
                ep_tp_z_clone=z_clone,
                ep_tp_z_clone_min_total=16,
                ep_tp_z_clone_min_remote=8,
            )
            x2 = jnp.where(jnp.isfinite(x_in), x_in + y1, jnp.bfloat16(0.0)).astype(
                jnp.bfloat16
            )
            return fused_ep_moe_v2(
                x2,
                w1_m,
                w2_m,
                s1_m,
                s2_m,
                gt2,
                topk=4,
                renormalize=True,
                mesh=mesh,
                capacity=16,
                rank=rk8,
                weight_format=WeightFormat.FP8,
                sharded_plan=True,
                scoring_fn="sigmoid",
                ep_tp_size=2,
                ep_tp_row_splits=2,
                ep_tp_z_clone=z_clone,
                ep_tp_z_clone_min_total=16,
                ep_tp_z_clone_min_remote=8,
            )

        with (
            mock.patch.object(pltpu, "get_tpu_info", return_value=fake_info),
            mock.patch.object(tpu_info, "get_tpu_info", return_value=fake_info),
            pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)),
        ):
            pltpu.reset_tpu_interpret_mode_state()
            with contextlib.redirect_stdout(io.StringIO()):
                out = np.asarray(jax.jit(_fn)(x_m))
            assert not interpret_pallas_call.races.races_found
        assert np.all(np.isfinite(out[1:].astype(np.float32)))
        assert np.all(out[0].astype(np.float32) == 0.0)
        return out.view(np.uint16)

    np.testing.assert_array_equal(_run_pair(1), _run_pair(0))


def test_should_execute_two_back_to_back_z_clone_layers_without_races_or_drift():
    if len(jax.devices("cpu")) >= 8:
        _run_two_layer_z_clone_race_and_parity_check()
        return
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count=8",
    )
    cmd = (
        "from tests.kernels.fused_moe.test_fused_ep_moe_v2_z_clone import "
        "_run_two_layer_z_clone_race_and_parity_check; "
        "_run_two_layer_z_clone_race_and_parity_check()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", cmd],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
