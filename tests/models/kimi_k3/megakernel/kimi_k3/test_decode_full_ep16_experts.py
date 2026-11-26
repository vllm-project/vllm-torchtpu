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

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.models.kimi_k3.megakernel.common import pool_alias
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    decode_megakernel as dm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    unpack_mxfp4,
)


def _reference_gated_activation(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    val = (
        4.0
        * np.tanh(gate / 4.0)
        * (1.0 / (1.0 + np.exp(-gate)))
        * 25.0
        * np.tanh(up / 25.0)
    )
    return jnp.asarray(val, jnp.bfloat16)


def _reference_ep16_split_half_moe(
    source_bf16: jnp.ndarray,
    gu_packed: jnp.ndarray,
    gu_scales: jnp.ndarray,
    dn_packed: jnp.ndarray,
    dn_scales: jnp.ndarray,
    selected_experts: np.ndarray,
    selected_probs: np.ndarray,
    local_route_counts: np.ndarray,
    prob_totals: np.ndarray,
) -> np.ndarray:
    batch_size, input_dim = source_bf16.shape
    expert_width = gu_packed.shape[2] // 2
    half_width = expert_width // 2
    output_dim = dn_packed.shape[2]
    src_f32 = np.asarray(source_bf16.astype(jnp.float32))
    out = np.zeros((batch_size, output_dim), dtype=np.float32)

    for b in range(batch_size):
        for r in range(int(local_route_counts[b])):
            e = int(selected_experts[b, r])
            w_p = float(selected_probs[b, r]) / float(prob_totals[b])
            w_gu = np.asarray(
                unpack_mxfp4(gu_packed[e], gu_scales[e]).astype(jnp.float32)
            )
            w_dn = np.asarray(
                unpack_mxfp4(dn_packed[e], dn_scales[e]).astype(jnp.float32)
            )
            x_b = src_f32[b : b + 1]
            gu_bf = np.asarray(
                jnp.asarray(x_b @ w_gu, jnp.bfloat16).astype(jnp.float32)
            )
            gate = gu_bf[:, :expert_width]
            up = gu_bf[:, expert_width:]
            mid = np.asarray(_reference_gated_activation(gate, up).astype(jnp.float32))
            for half in range(2):
                h_slice = slice(half * half_width, (half + 1) * half_width)
                dn_h = mid[:, h_slice] @ w_dn[h_slice, :]
                dn_bf = np.asarray(jnp.asarray(dn_h, jnp.bfloat16).astype(jnp.float32))
                out[b : b + 1] += dn_bf * w_p
    return out


def _run_stream_interpreted(
    source_bf16: jnp.ndarray,
    gu_hbm: jnp.ndarray,
    gus_hbm: jnp.ndarray,
    dn_hbm: jnp.ndarray,
    dns_hbm: jnp.ndarray,
    selected_experts: jnp.ndarray,
    selected_probs: jnp.ndarray,
    local_route_counts: jnp.ndarray,
    prob_totals: jnp.ndarray,
    *,
    group_pending_routes: bool,
) -> np.ndarray:
    batch_size, num_routes = selected_experts.shape
    input_dim = gu_hbm.shape[1] * 8
    expert_width = gu_hbm.shape[2] // 2
    output_dim = dn_hbm.shape[2]

    def body(
        src_ref,
        sel_exp_ref,
        sel_prob_ref,
        cnt_ref,
        tot_ref,
        gu_ref,
        gus_ref,
        dn_ref,
        dns_ref,
        out_ref,
        gu_vmem,
        gus_vmem,
        dn_vmem,
        dns_vmem,
        gu_acc,
        proc_vmem,
        sems,
    ):
        def copy_expert(expert, slot, *, gate_up: bool = True, down: bool = True):
            return SimpleNamespace(
                gate_up=(
                    tpu.make_async_copy(
                        gu_ref.at[expert], gu_vmem.at[slot], sems.at[slot, 0]
                    )
                    if gate_up
                    else None
                ),
                gate_up_scales=(
                    tpu.make_async_copy(
                        gus_ref.at[expert], gus_vmem.at[slot], sems.at[slot, 1]
                    )
                    if gate_up
                    else None
                ),
                down=(
                    tpu.make_async_copy(
                        dn_ref.at[expert], dn_vmem.at[slot], sems.at[slot, 2]
                    )
                    if down
                    else None
                ),
                down_scales=(
                    tpu.make_async_copy(
                        dns_ref.at[expert], dns_vmem.at[slot], sems.at[slot, 3]
                    )
                    if down
                    else None
                ),
            )

        sel_exp = sel_exp_ref[...]
        sel_prob = sel_prob_ref[...]
        counts = cnt_ref[:, 0]
        totals = tot_ref[:, 0]

        route_slots = jnp.arange(num_routes)[None, :]
        processed = jnp.zeros((batch_size, num_routes), jnp.bool_)
        for slot in range(2):
            pending = (route_slots < counts[:, None]) & ~processed
            if group_pending_routes:
                min_e = jnp.min(
                    jnp.where(pending, sel_exp.astype(jnp.float32), jnp.inf)
                )
                has_e = jnp.any(pending)
                exp = jnp.where(has_e, min_e, 0).astype(jnp.int32)
                consumed = pending & (sel_exp == exp)
            else:
                rows_w = jnp.any(pending, axis=1)
                cur_s = jnp.sum(processed.astype(jnp.int32), axis=1)
                cur_m = route_slots == cur_s[:, None]
                cur_e = jnp.sum(jnp.where(cur_m, sel_exp, 0), axis=1)
                min_e = jnp.min(jnp.where(rows_w, cur_e.astype(jnp.float32), jnp.inf))
                has_e = jnp.any(rows_w)
                exp = jnp.where(has_e, min_e, 0).astype(jnp.int32)
                consumed = (rows_w & (cur_e == exp))[:, None] & cur_m

            @pl.when(has_e)
            def _start(exp=exp, slot=slot):
                c = copy_expert(exp, slot)
                c.gate_up.start()
                c.gate_up_scales.start()
                c.down.start()
                c.down_scales.start()

            processed |= consumed

        dm.run_full_ep16_expert_stream(
            src_ref,
            sel_exp,
            sel_prob,
            counts,
            totals,
            gu_vmem,
            gus_vmem,
            dn_vmem,
            dns_vmem,
            out_ref,
            gu_acc,
            proc_vmem,
            copy_expert,
            expert_input_size=input_dim,
            expert_width=expert_width,
            expert_output_size=output_dim,
            group_pending_routes=group_pending_routes,
        )

    out = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((batch_size, output_dim), jnp.float32),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=tpu.VMEM),
                pl.BlockSpec(memory_space=tpu.VMEM),
                pl.BlockSpec(memory_space=tpu.VMEM),
                pl.BlockSpec(memory_space=tpu.VMEM),
                pl.BlockSpec(memory_space=tpu.VMEM),
                pl.BlockSpec(memory_space=tpu.HBM),
                pl.BlockSpec(memory_space=tpu.HBM),
                pl.BlockSpec(memory_space=tpu.HBM),
                pl.BlockSpec(memory_space=tpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=tpu.VMEM),
            scratch_shapes=(
                tpu.VMEM((2, *gu_hbm.shape[1:]), jnp.uint32),
                tpu.VMEM((2, *gus_hbm.shape[1:]), jnp.uint8),
                tpu.VMEM((2, *dn_hbm.shape[1:]), jnp.uint32),
                tpu.VMEM((2, *dns_hbm.shape[1:]), jnp.uint8),
                tpu.VMEM((batch_size, 2 * expert_width), jnp.float32),
                tpu.VMEM((batch_size, 128), jnp.int32),
                tpu.SemaphoreType.DMA((2, 4)),
            ),
        ),
        interpret=True,
    )(
        source_bf16,
        selected_experts,
        selected_probs,
        jnp.broadcast_to(local_route_counts[:, None], (batch_size, 128)),
        jnp.broadcast_to(prob_totals[:, None], (batch_size, 128)),
        gu_hbm,
        gus_hbm,
        dn_hbm,
        dns_hbm,
    )
    return np.asarray(out)


def _make_mxfp4_weights(
    seed: int,
    num_experts: int,
    input_dim: int = 512,
    expert_width: int = 512,
    output_dim: int = 256,
):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    gu = jax.random.randint(
        k1, (num_experts, input_dim // 8, 2 * expert_width), 0, 2**31 - 1, jnp.uint32
    )
    gus = jax.random.randint(
        k2, (num_experts, input_dim // 32, 2 * expert_width), 122, 126, jnp.uint8
    )
    dn = jax.random.randint(
        k3, (num_experts, expert_width // 8, output_dim), 0, 2**31 - 1, jnp.uint32
    )
    dns = jax.random.randint(
        k4, (num_experts, expert_width // 32, output_dim), 122, 126, jnp.uint8
    )
    return gu, gus, dn, dns


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_should_enable_full_ep16_experts_for_b1_to_b8_under_ep16(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    monkeypatch.setattr(
        k3_config, "_ACTIVE", k3_config.KernelConfig(decode_expert_parallel=16)
    )
    assert dm.use_full_ep16_experts(batch_size)


@pytest.mark.parametrize(
    ("decode_ep", "batch_size"),
    [(16, 12), (16, 16), (16, 32), (16, 64), (16, 128), (8, 1), (8, 4), (8, 8)],
)
def test_should_disable_full_ep16_experts_above_b8_or_under_ep8(
    monkeypatch: pytest.MonkeyPatch, decode_ep: int, batch_size: int
) -> None:
    monkeypatch.setattr(
        k3_config, "_ACTIVE", k3_config.KernelConfig(decode_expert_parallel=decode_ep)
    )
    assert not dm.use_full_ep16_experts(batch_size)


def test_should_match_half_slot_vmem_footprint_with_zero_byte_delta() -> None:
    half_slots = (
        ((4, 448, 1536), jnp.dtype(jnp.uint32)),
        ((4, 112, 1536), jnp.dtype(jnp.uint8)),
        ((4, 96, 3584), jnp.dtype(jnp.uint32)),
        ((4, 24, 3584), jnp.dtype(jnp.uint8)),
    )
    full_slots = (
        ((2, 448, 3072), jnp.dtype(jnp.uint32)),
        ((2, 112, 3072), jnp.dtype(jnp.uint8)),
        ((2, 192, 3584), jnp.dtype(jnp.uint32)),
        ((2, 48, 3584), jnp.dtype(jnp.uint8)),
    )
    half_bytes = sum(pool_alias.storage_size(s, d) for s, d in half_slots)
    full_bytes = sum(pool_alias.storage_size(s, d) for s, d in full_slots)
    assert full_bytes == half_bytes


def test_should_issue_four_contiguous_dmas_for_full_ep16_and_pairs_for_half_ep16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        k3_config, "_ACTIVE", k3_config.KernelConfig(decode_expert_parallel=16)
    )
    recorded_copies: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class _FakeRef:
        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape

        @property
        def at(self):
            return self

        def __getitem__(self, idx):
            if not isinstance(idx, tuple):
                idx = (idx,)
            out = []
            for dim, s in enumerate(self.shape):
                if dim >= len(idx):
                    out.append(s)
                else:
                    sel = idx[dim]
                    if isinstance(sel, slice):
                        out.append(s)
                    elif hasattr(sel, "size"):
                        out.append(int(sel.size))
            return _FakeRef(tuple(out))

    def fake_make_async_copy(src, dst, sem):
        del sem
        recorded_copies.append((src.shape, dst.shape))
        return SimpleNamespace(start=lambda: None, wait=lambda: None)

    monkeypatch.setattr(tpu, "make_async_copy", fake_make_async_copy)

    full_w = dm.MixtureOfExpertsWeights(
        routing_projection=None,
        routing_projection_vmem=None,
        routing_copy_semaphores=None,
        routing_bias=None,
        latent_input_projection=None,
        latent_output_projection=None,
        latent_output_projection_vmem=None,
        latent_output_copy_semaphore=None,
        latent_normalization=None,
        shared_gate_up_projection=None,
        shared_down_projection=None,
        expert_gate_up_projection=_FakeRef((56, 448, 3072)),
        expert_gate_up_projection_vmem=_FakeRef((2, 448, 3072)),
        expert_gate_up_scales=_FakeRef((56, 112, 3072)),
        expert_gate_up_scales_vmem=_FakeRef((2, 112, 3072)),
        expert_down_projection=_FakeRef((56, 192, 3584)),
        expert_down_projection_vmem=_FakeRef((2, 192, 3584)),
        expert_down_scales=_FakeRef((56, 48, 3584)),
        expert_down_scales_vmem=_FakeRef((2, 48, 3584)),
        expert_copy_semaphores=_FakeRef((2, 4)),
    )
    full_w.make_expert_copies(3, 1)
    assert recorded_copies == [
        ((448, 3072), (448, 3072)),
        ((112, 3072), (112, 3072)),
        ((192, 3584), (192, 3584)),
        ((48, 3584), (48, 3584)),
    ]

    recorded_copies.clear()
    half_w = dm.MixtureOfExpertsWeights(
        routing_projection=None,
        routing_projection_vmem=None,
        routing_copy_semaphores=None,
        routing_bias=None,
        latent_input_projection=None,
        latent_output_projection=None,
        latent_output_projection_vmem=None,
        latent_output_copy_semaphore=None,
        latent_normalization=None,
        shared_gate_up_projection=None,
        shared_down_projection=None,
        expert_gate_up_projection=_FakeRef((56, 448, 3072)),
        expert_gate_up_projection_vmem=_FakeRef((4, 448, 1536)),
        expert_gate_up_scales=_FakeRef((56, 112, 3072)),
        expert_gate_up_scales_vmem=_FakeRef((4, 112, 1536)),
        expert_down_projection=_FakeRef((56, 192, 3584)),
        expert_down_projection_vmem=_FakeRef((4, 96, 3584)),
        expert_down_scales=_FakeRef((56, 48, 3584)),
        expert_down_scales_vmem=_FakeRef((4, 24, 3584)),
        expert_copy_semaphores=_FakeRef((4, 4)),
    )
    half_w.make_expert_copies(7, 2)
    assert len(recorded_copies) == 6
    assert recorded_copies[0] == ((448, 768), (448, 768))
    assert recorded_copies[1] == ((448, 768), (448, 768))


def test_should_match_independent_oracle_across_all_b8_tiers_and_nan_inactive_row() -> (
    None
):
    gu, gus, dn, dns = _make_mxfp4_weights(seed=11, num_experts=6)
    key = jax.random.PRNGKey(99)
    src = (jax.random.normal(key, (8, 512), jnp.float32) * 0.25).astype(jnp.bfloat16)
    src_with_nan = src.at[7].set(jnp.bfloat16(jnp.nan))

    routes = [
        [2, 3, 4],
        [1, 4],
        [3, 4],
        [2, 4],
        [3, 4],
        [0, 2],
        [1, 3, 4],
        [],
    ]
    max_k = 4
    sel_exp = np.zeros((8, max_k), dtype=np.int32)
    sel_prob = np.zeros((8, max_k), dtype=np.float32)
    counts = np.zeros((8,), dtype=np.int32)
    for b, rs in enumerate(routes):
        counts[b] = len(rs)
        for j, e in enumerate(rs):
            sel_exp[b, j] = e
            sel_prob[b, j] = 0.13 * (b + 1) + 0.07 * (j + 1)
    totals = np.array([1.1, 1.3, 1.7, 1.9, 2.3, 2.9, 3.1, 1.0], dtype=np.float32)

    actual = _run_stream_interpreted(
        src_with_nan,
        gu,
        gus,
        dn,
        dns,
        jnp.asarray(sel_exp),
        jnp.asarray(sel_prob),
        jnp.asarray(counts),
        jnp.asarray(totals),
        group_pending_routes=True,
    )
    expected = _reference_ep16_split_half_moe(
        src, gu, gus, dn, dns, sel_exp, sel_prob, counts, totals
    )

    assert np.all(np.isfinite(actual[:7]))
    np.testing.assert_array_equal(actual[7], np.zeros((256,), dtype=np.float32))
    rel_l2 = np.linalg.norm(actual[:7] - expected[:7]) / np.linalg.norm(expected[:7])
    assert rel_l2 < 1e-3
    assert np.median(np.abs(actual[:7] - expected[:7])) < 1e-4

    gu_swapped = jnp.concatenate(
        [gu[:, :, :256], gu[:, :, 512:768], gu[:, :, 256:512], gu[:, :, 768:]], axis=2
    )
    wrong = _reference_ep16_split_half_moe(
        src, gu_swapped, gus, dn, dns, sel_exp, sel_prob, counts, totals
    )
    rel_wrong = np.linalg.norm(actual[:7] - wrong[:7]) / np.linalg.norm(expected[:7])
    assert rel_wrong > 0.2


def test_should_return_zero_when_b8_rank_has_zero_active_experts() -> None:
    gu, gus, dn, dns = _make_mxfp4_weights(seed=7, num_experts=4)
    src = jnp.ones((8, 512), jnp.bfloat16)
    actual = _run_stream_interpreted(
        src,
        gu,
        gus,
        dn,
        dns,
        jnp.zeros((8, 4), jnp.int32),
        jnp.zeros((8, 4), jnp.float32),
        jnp.zeros((8,), jnp.int32),
        jnp.ones((8,), jnp.float32),
        group_pending_routes=True,
    )
    np.testing.assert_array_equal(actual, np.zeros((8, 256), dtype=np.float32))


@pytest.mark.parametrize(
    ("batch_size", "routes"),
    [
        (1, [[3, 0, 2, 1]]),
        (1, [[2]]),
        (1, [[]]),
        (2, [[2, 0], [0, 2]]),
        (2, [[1, 3], [1, 3]]),
        (4, [[0, 2, 3], [1, 2], [2, 3], [0, 3]]),
    ],
)
def test_should_match_independent_oracle_for_b1_b2_b4_ordered_and_shared_routes(
    batch_size: int, routes: list[list[int]]
) -> None:
    gu, gus, dn, dns = _make_mxfp4_weights(seed=23 + batch_size, num_experts=4)
    src = (
        jax.random.normal(
            jax.random.PRNGKey(batch_size), (batch_size, 512), jnp.float32
        )
        * 0.25
    ).astype(jnp.bfloat16)
    max_k = 4
    sel_exp = np.zeros((batch_size, max_k), dtype=np.int32)
    sel_prob = np.zeros((batch_size, max_k), dtype=np.float32)
    counts = np.zeros((batch_size,), dtype=np.int32)
    for b, rs in enumerate(routes):
        counts[b] = len(rs)
        for j, e in enumerate(rs):
            sel_exp[b, j] = e
            sel_prob[b, j] = 0.21 * (b + 1) + 0.09 * (j + 1)
    totals = np.linspace(1.2, 2.7, batch_size, dtype=np.float32)

    actual = _run_stream_interpreted(
        src,
        gu,
        gus,
        dn,
        dns,
        jnp.asarray(sel_exp),
        jnp.asarray(sel_prob),
        jnp.asarray(counts),
        jnp.asarray(totals),
        group_pending_routes=False,
    )
    expected = _reference_ep16_split_half_moe(
        src, gu, gus, dn, dns, sel_exp, sel_prob, counts, totals
    )

    if np.linalg.norm(expected) == 0.0:
        np.testing.assert_array_equal(actual, expected)
    else:
        rel_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert rel_l2 < 1e-3
        assert np.median(np.abs(actual - expected)) < 1e-4


@pytest.mark.parametrize(
    ("batch_size", "output_dim"),
    [(8, 1024), (8, 1792), (16, 1792)],
)
def test_should_match_independent_oracle_when_gu_and_down_run_in_rolled_loops(
    batch_size: int,
    output_dim: int,
) -> None:
    input_dim = 2560
    expert_width = 512
    gu, gus, dn, dns = _make_mxfp4_weights(
        seed=41 + output_dim + batch_size,
        num_experts=6,
        input_dim=input_dim,
        expert_width=expert_width,
        output_dim=output_dim,
    )
    src = (
        jax.random.normal(
            jax.random.PRNGKey(1337 + output_dim + batch_size),
            (batch_size, input_dim),
            jnp.float32,
        )
        * 0.15
    ).astype(jnp.bfloat16)
    dead_row = batch_size - 1
    src_with_nan = src.at[dead_row].set(jnp.bfloat16(jnp.nan))

    base_routes = [
        [2, 3, 4],
        [1, 4],
        [3, 4],
        [2, 4],
        [3, 4],
        [0, 2],
        [1, 3, 4],
        [],
    ]
    routes = (
        base_routes
        if batch_size == 8
        else base_routes[:7] + [[4], [4], [], [], [], [], [], [], []]
    )
    max_k = 4
    sel_exp = np.zeros((batch_size, max_k), dtype=np.int32)
    sel_prob = np.zeros((batch_size, max_k), dtype=np.float32)
    counts = np.zeros((batch_size,), dtype=np.int32)
    for b, rs in enumerate(routes):
        counts[b] = len(rs)
        for j, e in enumerate(rs):
            sel_exp[b, j] = e
            sel_prob[b, j] = 0.17 * (b + 1) + 0.05 * (j + 1)
    totals = np.linspace(1.2, 3.2, batch_size, dtype=np.float32)

    actual = _run_stream_interpreted(
        src_with_nan,
        gu,
        gus,
        dn,
        dns,
        jnp.asarray(sel_exp),
        jnp.asarray(sel_prob),
        jnp.asarray(counts),
        jnp.asarray(totals),
        group_pending_routes=True,
    )
    expected = _reference_ep16_split_half_moe(
        src, gu, gus, dn, dns, sel_exp, sel_prob, counts, totals
    )

    assert np.all(np.isfinite(actual[:dead_row]))
    np.testing.assert_array_equal(
        actual[dead_row], np.zeros((output_dim,), dtype=np.float32)
    )
    rel_l2 = np.linalg.norm(actual[:dead_row] - expected[:dead_row]) / np.linalg.norm(
        expected[:dead_row]
    )
    assert rel_l2 < 2.5e-3
    assert (
        np.median(np.abs(actual[:dead_row] - expected[:dead_row]))
        / np.median(np.abs(expected[:dead_row]))
        < 2.5e-3
    )

    src_no_tail = src.at[:, 2048:].set(jnp.bfloat16(0.0))
    wrong = _reference_ep16_split_half_moe(
        src_no_tail, gu, gus, dn, dns, sel_exp, sel_prob, counts, totals
    )
    rel_wrong = np.linalg.norm(actual[:dead_row] - wrong[:dead_row]) / np.linalg.norm(
        expected[:dead_row]
    )
    assert rel_wrong > 0.1


def _count_dot_general_in_jaxpr(jaxpr) -> int:
    total = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            total += 1
        for val in eqn.params.values():
            if hasattr(val, "eqns"):
                total += _count_dot_general_in_jaxpr(val)
            elif hasattr(val, "jaxpr"):
                total += _count_dot_general_in_jaxpr(val.jaxpr)
            elif isinstance(val, (tuple, list)):
                for item in val:
                    if hasattr(item, "eqns"):
                        total += _count_dot_general_in_jaxpr(item)
                    elif hasattr(item, "jaxpr"):
                        total += _count_dot_general_in_jaxpr(item.jaxpr)
    return total


@pytest.mark.parametrize(
    ("batch_size", "lo_bound", "hi_bound"),
    [(8, 50, 70), (16, 70, 90)],
)
def test_should_bound_traced_dot_general_count_when_gu_and_down_loops_are_rolled(
    batch_size: int,
    lo_bound: int,
    hi_bound: int,
) -> None:
    input_dim, expert_width, output_dim = 3584, 1536, 3584
    noop_copy = SimpleNamespace(
        gate_up=SimpleNamespace(start=lambda: None, wait=lambda: None),
        gate_up_scales=SimpleNamespace(start=lambda: None, wait=lambda: None),
        down=SimpleNamespace(start=lambda: None, wait=lambda: None),
        down_scales=SimpleNamespace(start=lambda: None, wait=lambda: None),
    )

    def body(
        src_ref,
        sel_exp_ref,
        sel_prob_ref,
        cnt_ref,
        tot_ref,
        out_ref,
        gu_vmem,
        gus_vmem,
        dn_vmem,
        dns_vmem,
        gu_acc,
        proc_vmem,
    ):
        dm.run_full_ep16_expert_stream(
            src_ref,
            sel_exp_ref[...],
            sel_prob_ref[...],
            cnt_ref[:, 0],
            tot_ref[:, 0],
            gu_vmem,
            gus_vmem,
            dn_vmem,
            dns_vmem,
            out_ref,
            gu_acc,
            proc_vmem,
            lambda _expert, _slot, **_kw: noop_copy,
            expert_input_size=input_dim,
            expert_width=expert_width,
            expert_output_size=output_dim,
            group_pending_routes=True,
        )

    def trace_fn(src, se, sp, cnt, tot):
        return pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((batch_size, output_dim), jnp.float32),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec(memory_space=tpu.VMEM)] * 5,
                out_specs=pl.BlockSpec(memory_space=tpu.VMEM),
                scratch_shapes=(
                    tpu.VMEM((2, input_dim // 8, 2 * expert_width), jnp.uint32),
                    tpu.VMEM((2, input_dim // 32, 2 * expert_width), jnp.uint8),
                    tpu.VMEM((2, expert_width // 8, output_dim), jnp.uint32),
                    tpu.VMEM((2, expert_width // 32, output_dim), jnp.uint8),
                    tpu.VMEM((batch_size, 2 * expert_width), jnp.float32),
                    tpu.VMEM((batch_size, 128), jnp.int32),
                ),
            ),
        )(src, se, sp, cnt, tot)

    closed = jax.make_jaxpr(trace_fn)(
        jax.ShapeDtypeStruct((batch_size, input_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((batch_size, 16), jnp.int32),
        jax.ShapeDtypeStruct((batch_size, 16), jnp.float32),
        jax.ShapeDtypeStruct((batch_size, 128), jnp.int32),
        jax.ShapeDtypeStruct((batch_size, 128), jnp.float32),
    )
    dot_count = _count_dot_general_in_jaxpr(closed.jaxpr)
    assert lo_bound <= dot_count <= hi_bound
