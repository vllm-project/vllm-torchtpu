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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as kernels,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import (
    fused,
    routed_16k,
)


class _ArrayRef:
    def __init__(self, array: np.ndarray, row_slice: slice | None = None):
        self._base = array
        self._slice = slice(None) if row_slice is None else row_slice

    @property
    def shape(self) -> tuple[int, ...]:
        return self._base[self._slice].shape

    def at(self, row_ds, _col=None):
        start = 0 if self._slice.start is None else self._slice.start
        sub_start = start + int(row_ds.start)
        sub_end = sub_start + int(row_ds.size)
        return _ArrayRef(self._base, slice(sub_start, sub_end))

    def __getitem__(self, item):
        if item is Ellipsis:
            return jnp.asarray(self._base[self._slice])
        if isinstance(item, tuple) and isinstance(item[0], pl.Slice):
            start = 0 if self._slice.start is None else self._slice.start
            s0 = start + int(item[0].start)
            return jnp.asarray(self._base[s0 : s0 + int(item[0].size), item[1]])
        return jnp.asarray(self._base[self._slice][item])

    def __setitem__(self, item, value):
        arr = np.asarray(value)
        if item is Ellipsis:
            self._base[self._slice] = arr
            return
        if isinstance(item, tuple) and isinstance(item[0], pl.Slice):
            start = 0 if self._slice.start is None else self._slice.start
            s0 = start + int(item[0].start)
            self._base[s0 : s0 + int(item[0].size), item[1]] = arr
            return
        self._base[self._slice][item] = arr


class _SliceableArrayRef(_ArrayRef):
    @property
    def at(self):  # type: ignore[override] -- matches Pallas Ref .at[...] indexing
        return _AtIndexer(self._base)


class _AtIndexer:
    def __init__(self, base: np.ndarray):
        self._base = base

    def __getitem__(self, item):
        row_ds = item[0] if isinstance(item, tuple) else item
        s0 = int(row_ds.start)
        return _SliceableArrayRef(self._base, slice(s0, s0 + int(row_ds.size)))


def test_should_keep_compound_16k_when_env_disabled(monkeypatch):
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_MOE_16K_MONOLITHIC", False)
    cfg = KernelConfig()

    assert cfg.prefill_moe_16k_monolithic is False
    assert cfg.key() != cfg.replace(prefill_moe_16k_monolithic=True).key()


def test_should_enable_monolithic_16k_when_env_set(monkeypatch):
    monkeypatch.setattr(k3_config, "DEFAULT_PREFILL_MOE_16K_MONOLITHIC", True)

    assert KernelConfig().prefill_moe_16k_monolithic is True


@pytest.mark.parametrize("with_mixture", [False, True])
@pytest.mark.parametrize(
    ("window", "monolithic", "expect_compound"),
    [
        (16384, False, True),
        (16384, True, False),
        (14336, True, True),
        (12288, True, True),
    ],
)
def test_should_route_only_16k_to_monolithic_when_flag_enabled(
    monkeypatch, with_mixture, window, monolithic, expect_compound
):
    cfg = KernelConfig(
        prefill_expert_parallel=8,
        prefill_moe_16k_monolithic=monolithic,
    )
    monkeypatch.setattr(k3_config, "active", lambda: cfg)
    recorded: list[tuple[str, int]] = []

    def fake_compound(_mesh, seq_len, parts, *, with_mixture=False):
        recorded.append(("compound", seq_len + len(parts) + int(with_mixture)))
        return ("compound", seq_len, parts, with_mixture)

    monkeypatch.setattr(fused, "_make_moe_compound", fake_compound)
    one_dev_mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))

    if expect_compound:
        out = (
            fused.make_indexed_mixture_moe_sp_ring(one_dev_mesh, seq_len=window)
            if with_mixture
            else fused.make_indexed_moe_sp_ring(one_dev_mesh, seq_len=window)
        )
        assert out[0] == "compound"
        assert out[1] == window
        assert len(recorded) == 1
    else:
        with pytest.raises(ValueError, match="flat 32-rank k3_sp mesh"):
            if with_mixture:
                fused.make_indexed_mixture_moe_sp_ring(one_dev_mesh, seq_len=window)
            else:
                fused.make_indexed_moe_sp_ring(one_dev_mesh, seq_len=window)
        assert recorded == []


@pytest.mark.parametrize("local_experts", [112, 56])
def test_should_sort_all_262144_routes_and_track_skew_peak_at_16k(local_experts):
    tokens = 16384
    rows = tokens // 32
    rng = np.random.default_rng(20261112 + local_experts)

    ids_by_rank = np.empty((32, rows, 16), dtype=np.int32)
    probs_by_rank = rng.uniform(0.02, 0.2, size=(32, rows, 16)).astype(np.float32)
    for r in range(32):
        for row_i in range(rows):
            if r < 22:
                ids_by_rank[r, row_i] = rng.choice(
                    local_experts, size=16, replace=False
                )
            else:
                ids_by_rank[r, row_i] = rng.choice(896, size=16, replace=False)
    probs_by_rank[31, -16:, :] = 0.0
    ids_by_rank[31, -16:, :] = 0

    wire_chunks = []
    for r in range(32):
        wire_chunks.append(ids_by_rank[r].reshape(rows // 8, 128))
        wire_chunks.append(probs_by_rank[r].view(np.int32).reshape(rows // 8, 128))
    wire_np = np.concatenate(wire_chunks, axis=0)

    ids_all = ids_by_rank.reshape(tokens, 16)
    probs_all = probs_by_rank.reshape(tokens, 16)
    live_mask = probs_all != 0
    hist_896 = np.bincount(ids_all[live_mask], minlength=896)
    expected_max = max(
        int(hist_896[g * local_experts : (g + 1) * local_experts].sum())
        for g in range(896 // local_experts)
    )
    assert expected_max > 98304

    keys_buf = np.zeros((rows * 4, 128), dtype=np.int32)
    probs_buf = np.zeros((tokens // 8, 128), dtype=np.float32)
    tids_buf = np.zeros((tokens // 8, 128), dtype=np.int32)
    counts_buf = np.zeros((local_experts,), dtype=np.int32)
    max_buf = np.zeros((1,), dtype=np.int32)
    for _ in fused._metadata_steps(  # noqa: SLF001 -- direct Pallas generator spec
        _SliceableArrayRef(wire_np),
        _SliceableArrayRef(keys_buf),
        _SliceableArrayRef(probs_buf),
        _SliceableArrayRef(tids_buf),
        _SliceableArrayRef(counts_buf),
        _SliceableArrayRef(max_buf),
        tokens=tokens,
        rank=jnp.int32(0),
        counts_ready=lambda: None,
        local_experts=local_experts,
    ):
        pass

    assert int(max_buf[0]) == expected_max
    np.testing.assert_array_equal(counts_buf, hist_896[:local_experts])
    total_local = int(hist_896[:local_experts].sum())
    assert total_local > 98304
    expected_tids = []
    expected_probs = []
    for expert in range(local_experts):
        tok_idx, slot_idx = np.where((ids_all == expert) & live_mask)
        order = np.lexsort((slot_idx, tok_idx))
        expected_tids.append(tok_idx[order])
        expected_probs.append(probs_all[tok_idx[order], slot_idx[order]])
    np.testing.assert_array_equal(
        tids_buf.reshape(-1)[:total_local], np.concatenate(expected_tids)
    )
    np.testing.assert_array_equal(
        probs_buf.reshape(-1)[:total_local], np.concatenate(expected_probs)
    )


def test_should_cover_all_512_rows_and_32_ranks_uniquely_in_16k_half_ring():
    layout = fused._latent_half_stream_layout(512, 32)  # noqa: SLF001 -- layout spec
    assert len(layout) == 6
    covered_rows: list[int] = []
    for family in range(3):
        pos_s, neg_s = layout[2 * family], layout[2 * family + 1]
        assert pos_s[2] == neg_s[2] and pos_s[3] == neg_s[3]
        covered_rows.extend(range(pos_s[2], pos_s[2] + pos_s[3]))
        cyc = [chip * 2 + core for chip in kernels.CYCLES[family] for core in range(2)]
        pos_of = {r: idx for idx, r in enumerate(cyc)}
        for rank in range(32):
            pos = pos_of[rank]
            received = [
                rank,
                *[cyc[(pos - step) & 31] for step in range(1, pos_s[4] + 1)],
                *[cyc[(pos + step) & 31] for step in range(1, neg_s[4] + 1)],
            ]
            assert len(received) == 32 and set(received) == set(range(32))
    assert covered_rows == list(range(512))


def test_should_partition_overflow_routes_bijectively_across_3_chunks():
    rng = np.random.default_rng(77)
    capacity = 98304
    chunk_rows = capacity // 128
    seq_len = 16384
    local_experts = 112
    counts = np.zeros((local_experts,), dtype=np.int32)
    counts[0] = 60000
    counts[1] = 60000
    counts[2] = 85000
    counts[3] = 40000
    total_live = int(counts.sum())

    ids_flat = np.zeros((seq_len * 16,), dtype=np.int32)
    probs_flat = np.zeros((seq_len * 16,), dtype=np.float32)
    ids_flat[:total_live] = rng.integers(0, seq_len, size=(total_live,), dtype=np.int32)
    probs_flat[:total_live] = rng.uniform(0.01, 0.5, size=(total_live,)).astype(
        np.float32
    )
    ids_flat[total_live:] = 16383
    probs_flat[total_live:] = 999.0

    ids_2d = ids_flat.reshape(seq_len // 8, 128)
    probs_2d = probs_flat.reshape(seq_len // 8, 128)
    max_chunks = (seq_len * 16 + capacity - 1) // capacity
    pad_rows = max_chunks * chunk_rows - ids_2d.shape[0]
    ids_pad = np.pad(ids_2d, ((0, pad_rows), (0, 0)))
    probs_pad = np.pad(probs_2d, ((0, pad_rows), (0, 0)))

    ends = np.cumsum(counts, dtype=np.int32)
    starts = ends - counts
    reconstructed: list[tuple[int, int, float]] = []
    for q in range(max_chunks):
        low, high = q * capacity, (q + 1) * capacity
        chunk_counts = np.maximum(0, np.minimum(ends, high) - np.maximum(starts, low))
        c_ids = ids_pad[q * chunk_rows : (q + 1) * chunk_rows].reshape(-1)
        c_probs = probs_pad[q * chunk_rows : (q + 1) * chunk_rows].reshape(-1)
        off = 0
        for e in range(local_experts):
            cnt = int(chunk_counts[e])
            for k_i in range(cnt):
                reconstructed.append(
                    (e, int(c_ids[off + k_i]), float(c_probs[off + k_i]))
                )
            off += cnt

    expected: list[tuple[int, int, float]] = []
    off = 0
    for e in range(local_experts):
        cnt = int(counts[e])
        for k_i in range(cnt):
            expected.append((e, int(ids_flat[off + k_i]), float(probs_flat[off + k_i])))
        off += cnt
    assert reconstructed == expected


def test_should_match_vmem_epilogue_bitwise_at_512_rows_with_masked_tail():
    rows, width, hidden = 512, 3584, 7168
    live_tokens = 14336 + 19
    rng = np.random.default_rng(20261124)
    reduced_np = rng.standard_normal((rows, width)).astype(np.float32) * 0.2
    shared_np = rng.standard_normal((rows, hidden)).astype(np.float32) * 0.1
    reduced_np[-45:, :] = 77.0
    shared_np[-45:, :] = -888.0
    reduced_bf16 = jnp.asarray(reduced_np, jnp.bfloat16)
    shared_bf16 = jnp.asarray(shared_np, jnp.bfloat16)
    norm_f32 = jnp.asarray(0.8 + 0.4 * rng.random((1, 1, width)), jnp.float32)
    up_bf16 = jnp.asarray(rng.standard_normal((1, width, hidden)) * 0.02, jnp.bfloat16)
    prefix_bf16 = jnp.asarray(rng.standard_normal((rows, hidden)) * 0.3, jnp.bfloat16)
    control = jnp.array([live_tokens, 0], dtype=jnp.int32)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    rep = jax.sharding.PartitionSpec()
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)

    def make_epilogue_call(*, staged_hbm: bool):
        def body(ctrl_r, red_r, shr_r, nrm_r, up_r, pfx_r, out_r):
            fused._epilogue(  # noqa: SLF001 -- direct Pallas helper spec
                ctrl_r, red_r, shr_r, nrm_r, up_r, pfx_r, out_r, staged_hbm=staged_hbm
            )

        rs_spec = hbm if staged_hbm else vmem
        return pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=(rs_spec, rs_spec, hbm, hbm, hbm),
                out_specs=hbm,
            ),
            interpret=True,
        )

    @jax.shard_map(
        mesh=mesh, in_specs=(rep,) * 6, out_specs=(rep, rep), check_vma=False
    )
    def run_pair(ctrl, red, shr, nrm, up, pfx):
        return (
            make_epilogue_call(staged_hbm=False)(ctrl, red, shr, nrm, up, pfx),
            make_epilogue_call(staged_hbm=True)(ctrl, red, shr, nrm, up, pfx),
        )

    out_vmem, out_hbm = run_pair(
        control, reduced_bf16, shared_bf16, norm_f32, up_bf16, prefix_bf16
    )
    np.testing.assert_array_equal(np.asarray(out_hbm), np.asarray(out_vmem))
    np.testing.assert_array_equal(
        np.asarray(out_hbm)[467:], np.asarray(prefix_bf16)[467:]
    )
    assert (
        np.max(
            np.abs(
                np.asarray(out_hbm, np.float32)[:467]
                - np.asarray(prefix_bf16, np.float32)[:467]
            )
        )
        > 0.05
    )


def test_should_select_lowest_expert_id_on_ties_in_top16_and_zero_invalid_rows():
    rng = np.random.default_rng(20261125)
    logits = rng.standard_normal((256, 896)).astype(np.float32) * 0.5
    bias = rng.uniform(-0.05, 0.05, size=(1, 896)).astype(np.float32)
    tie_experts = (3, 19, 400, 895)
    for exp_id in tie_experts:
        logits[0, exp_id] = 4.0
        bias[0, exp_id] = 0.25
    valid = np.arange(256)[:, None] < 240

    selector = fused._Top16(jnp.asarray(logits), jnp.asarray(bias))  # noqa: SLF001 -- Top16 spec
    selected, probs = selector.finish(jnp.asarray(valid))
    sel_np = np.asarray(selected)
    probs_np = np.asarray(probs)

    assert tuple(sel_np[0, :4]) == tie_experts
    np.testing.assert_allclose(probs_np[:240].sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(probs_np[240:], 0.0)


def test_should_tile_512_row_latent_half_pair_panels_identically_to_single_matmul():
    assert fused._compact_capacity(512) == 8192  # noqa: SLF001 -- capacity spec
    assert fused._compact_capacity(8192) == 65536  # noqa: SLF001 -- capacity spec
    assert fused._compact_capacity(16384) == 98304  # noqa: SLF001 -- capacity spec

    rng = np.random.default_rng(20261126)
    lhs = jnp.asarray(rng.standard_normal((512, 7168)) * 0.1, jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((7168, 3584)) * 0.05, jnp.bfloat16)
    expected = jnp.matmul(lhs, weights, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pltpu.VMEM
    dma = pltpu.SemaphoreType.DMA

    def body(lhs_r, w_r, out_r):
        def run(bufs, sems):
            for col, prod in fused._latent_half_pair_panels(  # noqa: SLF001 -- panel spec
                lhs_r, w_r, bufs, sems
            ):
                out_r[:, pl.ds(col, 256)] = prod

        pl.run_scoped(
            run,
            tuple(vmem((7168, 256), jnp.bfloat16) for _ in range(4)),
            tuple(dma for _ in range(4)),
        )

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    call = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((512, 3584), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, in_specs=(hbm, hbm), out_specs=hbm
        ),
        interpret=True,
    )

    @jax.shard_map(mesh=mesh, in_specs=(P(), P()), out_specs=P(), check_vma=False)
    def run_single(x, w):
        return call(x, w)

    actual = run_single(lhs, weights)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_should_match_prestarted_latent_half_pair_panels_bitwise():
    rng = np.random.default_rng(20261127)
    lhs = jnp.asarray(rng.standard_normal((512, 7168)) * 0.1, jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((7168, 3584)) * 0.05, jnp.bfloat16)
    expected = jnp.matmul(lhs, weights, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pltpu.VMEM
    dma = pltpu.SemaphoreType.DMA

    def body(lhs_r, w_r, out_r):
        def run(bufs, sems):
            pltpu.make_async_copy(w_r.at[:, pl.ds(0, 256)], bufs[0], sems[0]).start()
            pltpu.make_async_copy(w_r.at[:, pl.ds(1792, 256)], bufs[1], sems[1]).start()
            for col, prod in fused._latent_half_pair_panels(  # noqa: SLF001 -- panel spec
                lhs_r, w_r, bufs, sems, prestarted=True
            ):
                out_r[:, pl.ds(col, 256)] = prod

        pl.run_scoped(
            run,
            tuple(vmem((7168, 256), jnp.bfloat16) for _ in range(4)),
            tuple(dma for _ in range(4)),
        )

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    call = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((512, 3584), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, in_specs=(hbm, hbm), out_specs=hbm
        ),
        interpret=True,
    )

    @jax.shard_map(mesh=mesh, in_specs=(P(), P()), out_specs=P(), check_vma=False)
    def run_single(x, w):
        return call(x, w)

    actual = run_single(lhs, weights)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_should_alternate_two_phase_ring_semaphores_cleanly_across_eight_split_epochs():
    regions_h0 = kernels._regions(512, 512)  # noqa: SLF001 -- ring region spec
    regions_h1 = kernels._regions(512, 384)  # noqa: SLF001 -- ring region spec
    fam0_h0 = (regions_h0[0][2], regions_h0[0][2] + regions_h0[0][3])
    fam1_h1 = (regions_h1[2][2], regions_h1[2][2] + regions_h1[2][3])
    assert fam0_h0 == (0, 256)
    assert fam1_h1 == (128, 256)
    assert max(fam0_h0[0], fam1_h1[0]) < min(fam0_h0[1], fam1_h1[1])

    peers_by_rank: list[list[int]] = [[] for _ in range(32)]
    for cycle in kernels.CYCLES:
        ring32 = [chip * 2 + core for chip in cycle for core in range(2)]
        pos_of = {r: idx for idx, r in enumerate(ring32)}
        for r in range(32):
            p = pos_of[r]
            peers_by_rank[r].append(ring32[(p - 1) & 31])
            peers_by_rank[r].append(ring32[(p + 1) & 31])

    def has_stale_epoch_collision(num_semaphores: int) -> bool:
        signaled_epoch0 = {r for r in range(32) if r != 0}
        completed_epoch0 = {
            r
            for r in signaled_epoch0
            if all(nb in signaled_epoch0 for nb in peers_by_rank[r])
        }
        assert completed_epoch0
        for fast_r in completed_epoch0:
            for nb in peers_by_rank[fast_r]:
                if nb not in completed_epoch0 and (1 % num_semaphores) == (
                    0 % num_semaphores
                ):
                    return True
        return False

    assert has_stale_epoch_collision(1) is True
    assert has_stale_epoch_collision(2) is False

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def body(out_r):
        def run(phase_sems, phase_smem):
            phase_smem[0] = 0
            trace = jnp.zeros((8,), jnp.int32)
            for epoch in range(8):
                sem = phase_sems.at[phase_smem[0] & 1]
                for _ in range(6):
                    pl.semaphore_signal(sem, 1)
                slot = phase_smem[0]
                pl.semaphore_wait(phase_sems.at[slot & 1], 6)
                trace = trace.at[epoch].set(slot & 1)
                phase_smem[0] = slot + 1
            out_r[:, 0] = trace
            out_r[0, 1] = phase_smem[0]

        pl.run_scoped(
            run,
            pltpu.SemaphoreType.REGULAR((2,)),
            pltpu.SMEM((1,), jnp.int32),
        )

    call = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(),
            out_specs=hbm,
        ),
        interpret=True,
    )
    res = np.asarray(call())
    np.testing.assert_array_equal(
        res[:, 0], np.array([0, 1, 0, 1, 0, 1, 0, 1], np.int32)
    )
    assert int(res[0, 1]) == 8


def test_should_double_sparse_poll_threshold_for_split_rings_to_prevent_starvation():
    unsplit = fused.sparse_poll_threshold(split_ring=False)
    split = fused.sparse_poll_threshold(split_ring=True)

    assert unsplit == 72
    assert split == 144
    assert (unsplit // 2) >= 32
    assert ((split // 2) // 2) >= 32
    assert ((unsplit // 2) // 2) < 31


def test_should_cover_all_32_ranks_bijectively_in_16k_half_ring_rs():
    for cycle in kernels.CYCLES:
        ring32 = [chip * 2 + core for chip in cycle for core in range(2)]
        pos_of = {r: idx for idx, r in enumerate(ring32)}
        for owner_rank in range(32):
            q = pos_of[owner_rank]
            cw_senders: list[int] = []
            for step in range(routed_16k.CW_MAX_HOPS + 1):
                sender_pos = (q - routed_16k.CW_MAX_HOPS + step) & 31
                read_pos = routed_16k.half_ring_owner_positions(
                    sender_pos, direction=0, n_ranks=32
                )[step]
                assert read_pos == q
                cw_senders.append(ring32[sender_pos])

            ccw_senders: list[int] = []
            for step in range(routed_16k.CCW_MAX_HOPS):
                sender_pos = (q + routed_16k.CCW_MAX_HOPS - step) & 31
                read_pos = routed_16k.half_ring_owner_positions(
                    sender_pos, direction=1, n_ranks=32
                )[step]
                assert read_pos == q
                ccw_senders.append(ring32[sender_pos])

            assert len(cw_senders) == 17
            assert len(ccw_senders) == 15
            assert set(cw_senders).isdisjoint(ccw_senders)
            assert sorted(cw_senders + ccw_senders) == list(range(32))

    with pytest.raises(ValueError, match="even n_ranks >= 4"):
        routed_16k.half_ring_owner_positions(0, direction=0, n_ranks=3)
    with pytest.raises(ValueError, match="direction"):
        routed_16k.half_ring_owner_positions(0, direction=2, n_ranks=32)


def test_should_partition_3pass_1536_1280_768_columns_and_col_specs_bijectively():
    specs = routed_16k.DOWN_PASS_SPECS_16K
    assert len(specs) == 3
    assert [s.width for s in specs] == [
        routed_16k.SUBTOTAL_16K_COLS,
        routed_16k.PASS1_16K_COLS,
        routed_16k.LAST_PASS_16K_COLS,
    ]
    assert [s.width for s in specs] == [1536, 1280, 768]

    full_cov = np.zeros((routed_16k.OUTPUT_WIDTH_16K,), dtype=np.int32)
    cursor = 0
    for spec in specs:
        assert spec.col_start == cursor
        full_cov[spec.col_start : spec.col_start + spec.width] += 1
        cursor += spec.width
    assert cursor == routed_16k.OUTPUT_WIDTH_16K == 3584
    assert np.all(full_cov == 1)

    for total_cols, col_spec in (
        (routed_16k.SUBTOTAL_16K_COLS, routed_16k.HALF_RING_1536_COL_SPEC),
        (routed_16k.PASS1_16K_COLS, routed_16k.HALF_RING_1280_COL_SPEC),
        (routed_16k.LAST_PASS_16K_COLS, routed_16k.HALF_RING_768_COL_SPEC),
    ):
        assert len(col_spec) == 3
        cov = np.zeros((total_cols,), dtype=np.int32)
        for col_start, col_width in col_spec:
            assert col_start % 128 == 0 and col_width % 128 == 0
            cov[col_start : col_start + col_width] += 1
        assert np.all(cov == 1)

    assert max(cw for _, cw in routed_16k.HALF_RING_768_COL_SPEC) == 256

    rows = 32
    rng = np.random.default_rng(2410)
    sums_full = np.full((rows + 16, 2, 384), 999.0, dtype=np.float32)
    sums_full[:rows] = rng.standard_normal((rows, 2, 384)).astype(np.float32)
    expected_768 = sums_full[:rows].reshape(rows, 768)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def read_body(sums_r, out_r):
        sr = pl.ds(0, rows)
        for c0, cw in routed_16k.HALF_RING_768_COL_SPEC:
            out_r[:, pl.ds(c0, cw)] = routed_16k._read_banked_sums_cols(  # noqa: SLF001 -- banked slice spec
                sums_r, sr, col_start=c0, col_width=cw
            )

    call = pl.pallas_call(
        read_body,
        out_shape=jax.ShapeDtypeStruct((rows, 768), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, in_specs=(hbm,), out_specs=hbm
        ),
        interpret=True,
    )
    actual_768 = np.asarray(call(jnp.asarray(sums_full, jnp.bfloat16)))
    np.testing.assert_array_equal(
        actual_768, np.asarray(jnp.asarray(expected_768, jnp.bfloat16))
    )
    assert not np.any(actual_768 == np.asarray(jnp.bfloat16(999.0)))


def test_should_stage_2d_subtotals_for_full_and_narrow_widths_in_pallas_interpret():
    tokens = 2048
    chunk_rows = 512
    rng = np.random.default_rng(24101)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    for cols, bank_cols in ((1536, 768), (1280, 640)):
        sums_np = np.full((tokens + 16, 2, bank_cols), 999.0, dtype=np.float32)
        sums_np[:tokens] = rng.standard_normal((tokens, 2, bank_cols)).astype(
            np.float32
        )
        sums_bf16 = jnp.asarray(sums_np, jnp.bfloat16)
        init_hbm = jnp.full((tokens, 1536), -777.0, jnp.bfloat16)

        def emit_body(sums_r, init_r, out_r, *, cols=cols):
            def run(sums_v, stage_sems):
                pltpu.sync_copy(init_r, out_r)
                pltpu.sync_copy(sums_r, sums_v)
                routed_16k._emit_subtotals_to_hbm(  # noqa: SLF001 -- emit spec
                    sums_v,
                    out_r,
                    stage_sems,
                    tokens=tokens,
                    cols=cols,
                    chunk_rows=chunk_rows,
                )

            pl.run_scoped(
                run,
                pltpu.VMEM(sums_r.shape, jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
            )

        call = pl.pallas_call(
            emit_body,
            out_shape=jax.ShapeDtypeStruct((tokens, 1536), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0, in_specs=(hbm, hbm), out_specs=hbm
            ),
            interpret=pltpu.InterpretParams(detect_races=True),
        )
        out = np.asarray(call(sums_bf16, init_hbm))
        expected_written = np.asarray(sums_bf16[:tokens].reshape(tokens, cols))
        np.testing.assert_array_equal(out[:, :cols], expected_written)
        assert not np.any(out[:, :cols] == np.asarray(jnp.bfloat16(999.0)))
        if cols < 1536:
            np.testing.assert_array_equal(
                out[:, cols:], np.full((tokens, 1536 - cols), -777.0, dtype=out.dtype)
            )


def test_should_keep_16k_decoupled_tile_and_panel_invariants_for_ep16_and_ep8():
    tm = routed_16k.TILE_M_16K
    bucket = routed_16k.BUCKET_BASE_16K
    gu_pw = routed_16k.GU_PANEL_WIDTH_16K

    assert tm == 384 and tm % bucket == 0 and bucket % 8 == 0
    assert 49152 % tm == 0
    for gu_width in (3072, 1536):
        assert gu_width % gu_pw == 0
        assert gu_pw % 256 == 0
    assert routed_16k.GU_ACT_BANKS_16K == 3
    assert 0 < routed_16k.DOWN_OVERLAP_UNROLL_16K < routed_16k.DOWN_TAIL_UNROLL_16K
    assert routed_16k.DOWN_OVERLAP_UNROLL_16K % 2 == 0
    assert routed_16k.DOWN_TAIL_UNROLL_16K % 2 == 0

    for spec in routed_16k.DOWN_PASS_SPECS_16K:
        assert spec.sub_banks == 2
        assert spec.bank_cols % 128 == 0
        assert spec.width % spec.panel_width == 0
        assert spec.panel_width % spec.bank_cols == 0
        assert spec.weight_buffers == 3
    assert routed_16k.TOKENS_16K % routed_16k.EMIT_CHUNK_PASS0_16K == 0
    assert routed_16k.TOKENS_16K % routed_16k.EMIT_CHUNK_PASS1_16K == 0


def test_should_bound_16k_3pass_fixed_ring_scoped_vmem_within_budget():
    vmem_budget_bytes = 64 * 1024 * 1024
    tm = routed_16k.TILE_M_16K
    local_rows = routed_16k.TOKENS_16K // 32
    p0, p1, p2 = routed_16k.DOWN_PASS_SPECS_16K

    acts_bytes = routed_16k.GU_ACT_BANKS_16K * tm * routed_16k.OUTPUT_WIDTH_16K * 2
    gu_weights_buf3 = 3 * (routed_16k.OUTPUT_WIDTH_16K * 3072 // 2)
    gu_out_buf2 = 2 * (tm * 1536 * 2)
    scope_gu_bytes = acts_bytes + gu_weights_buf3 + gu_out_buf2
    assert scope_gu_bytes < 32 * 1024 * 1024

    sums0_bytes = (routed_16k.TOKENS_16K + 16) * p0.width * 2
    packed0_bytes = 2 * tm * p0.width * 2
    dn_lhs_buf2 = 2 * (tm * 1536 * 2)
    dn_rhs0_buf3 = p0.weight_buffers * (1536 * p0.width // 2)
    scope_p0_bytes = sums0_bytes + packed0_bytes + dn_lhs_buf2 + dn_rhs0_buf3
    assert scope_p0_bytes < vmem_budget_bytes

    emit0_stage_bytes = 2 * routed_16k.EMIT_CHUNK_PASS0_16K * p0.width * 2
    assert sums0_bytes + emit0_stage_bytes < vmem_budget_bytes

    ring_bytes = (2 + 4 + 4) * local_rows * routed_16k.SUBTOTAL_16K_COLS * 2
    sums1_bytes = (routed_16k.TOKENS_16K + 16) * p1.width * 2
    packed1_bytes = 1 * tm * p1.width * 2
    dn_rhs1_buf3 = p1.weight_buffers * (1536 * p1.width // 2)
    scope_p1_bytes = (
        ring_bytes + sums1_bytes + packed1_bytes + dn_lhs_buf2 + dn_rhs1_buf3
    )
    assert scope_p1_bytes < vmem_budget_bytes

    emit1_stage_bytes = 2 * routed_16k.EMIT_CHUNK_PASS1_16K * p1.width * 2
    assert ring_bytes + sums1_bytes + emit1_stage_bytes < vmem_budget_bytes

    sums2_bytes = (routed_16k.TOKENS_16K + 16) * p2.width * 2
    packed2_bytes = 1 * tm * p2.width * 2
    dn_rhs2_buf3 = p2.weight_buffers * (1536 * p2.width // 2)
    scope_p2_bytes = (
        ring_bytes + sums2_bytes + packed2_bytes + dn_lhs_buf2 + dn_rhs2_buf3
    )
    assert scope_p2_bytes < vmem_budget_bytes

    assert scope_p0_bytes + ring_bytes > vmem_budget_bytes
