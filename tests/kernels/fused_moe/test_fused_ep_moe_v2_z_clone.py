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

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractMesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2.host import WeightFormat
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


@pytest.mark.parametrize("ep", [12, 32])
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


def test_should_lower_fused_ep_moe_v2_z_clone_to_tpu7x_mlir():
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
            ep_tp_z_clone=True,
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
    assert "_zclone" in mlir
    assert "partition_id" not in mlir
