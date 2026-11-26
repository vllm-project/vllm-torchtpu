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
"""TPU Mosaic hardware tests for decode_megakernel single-VREG row staging."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    decode_megakernel as dm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    unpack_mxfp4,
)


def _reference_gated_activation(gate: np.ndarray, up: np.ndarray) -> jax.Array:
    val = (
        4.0
        * np.tanh(gate / 4.0)
        * (1.0 / (1.0 + np.exp(-gate)))
        * 25.0
        * np.tanh(up / 25.0)
    )
    return jnp.asarray(val, jnp.bfloat16)


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


def _reference_moe(
    source_bf16: jax.Array,
    gu_packed: jax.Array,
    gu_scales: jax.Array,
    dn_packed: jax.Array,
    dn_scales: jax.Array,
    selected_experts: np.ndarray,
    selected_probs: np.ndarray,
    local_route_counts: np.ndarray,
    prob_totals: np.ndarray,
    *,
    split_half_down: bool,
) -> np.ndarray:
    batch_size = source_bf16.shape[0]
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
            if split_half_down:
                for half in range(2):
                    h_slice = slice(half * half_width, (half + 1) * half_width)
                    dn_h = mid[:, h_slice] @ w_dn[h_slice, :]
                    dn_bf = np.asarray(
                        jnp.asarray(dn_h, jnp.bfloat16).astype(jnp.float32)
                    )
                    out[b : b + 1] += dn_bf * w_p
            else:
                dn_full = mid @ w_dn
                dn_bf = np.asarray(
                    jnp.asarray(dn_full, jnp.bfloat16).astype(jnp.float32)
                )
                out[b : b + 1] += dn_bf * w_p
    return out


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("batch_size", "compact_capacity", "expert_input_size"),
    [
        (32, 8, 3584),
        (32, 16, 3584),
        (16, 8, 3584),
        (32, 8, 7168),
    ],
)
def test_should_gather_planned_rows_bitwise_exact_on_tpu_mosaic(
    batch_size: int,
    compact_capacity: int,
    expert_input_size: int,
) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend for Mosaic lowering")
    rng = np.random.default_rng(batch_size * 100 + compact_capacity + expert_input_size)
    src_np = rng.standard_normal((batch_size, expert_input_size), dtype=np.float32)
    src = jnp.asarray(src_np).astype(jnp.bfloat16)
    idx_np = rng.integers(0, batch_size, size=(compact_capacity,), dtype=np.int32)
    val_np = rng.integers(0, 2, size=(compact_capacity,), dtype=bool)
    val_np[0] = True

    def kernel(source, idx_ref, val_ref, destination):
        indices = idx_ref[:compact_capacity, 0]
        valid = val_ref[:compact_capacity, 0] > 0
        destination[...] = jnp.zeros(destination.shape, destination.dtype)
        if batch_size == 32:

            @pl.loop(0, batch_size // 16)
            def stage_block(block):
                in_block = valid & (indices // 16 == block)

                @pl.when(jnp.any(in_block))
                def copy_block():
                    rows = source[
                        pl.ds(pl.multiple_of(block * 16, 16), 16), :expert_input_size
                    ]
                    in_upper = (indices % 16) >= 8
                    gather_indices = jnp.broadcast_to(
                        (indices % 8)[:, None],
                        (compact_capacity, expert_input_size),
                    )
                    selected_lower = jnp.take_along_axis(
                        rows[:8].astype(jnp.float32),
                        gather_indices,
                        axis=0,
                        mode="promise_in_bounds",
                    )
                    selected_upper = jnp.take_along_axis(
                        rows[8:].astype(jnp.float32),
                        gather_indices,
                        axis=0,
                        mode="promise_in_bounds",
                    )
                    selected_rows = jnp.where(
                        in_upper[:, None], selected_upper, selected_lower
                    )
                    destination[...] = jnp.where(
                        in_block[:, None], selected_rows, destination[...]
                    )
        else:

            @pl.loop(0, jnp.sum(valid.astype(jnp.float32)).astype(jnp.int32))
            def copy_row(position):
                index = jnp.sum(
                    jnp.where(
                        jnp.arange(compact_capacity) == position,
                        indices.astype(jnp.float32),
                        0.0,
                    )
                ).astype(jnp.int32)
                rows = source[
                    pl.ds(pl.multiple_of(index // 16 * 16, 16), 16), :expert_input_size
                ]
                half_rows = jnp.where(index % 16 >= 8, rows[8:], rows[:8]).astype(
                    jnp.float32
                )
                gather_indices = jnp.full((1, expert_input_size), index % 8, jnp.int32)
                selected_row = jnp.take_along_axis(
                    half_rows,
                    gather_indices,
                    axis=0,
                    mode="promise_in_bounds",
                )
                destination[pl.ds(position, 1), :] = selected_row

    idx_pad = np.zeros((max(8, compact_capacity), 128), dtype=np.int32)
    idx_pad[:compact_capacity, :] = idx_np[:, None]
    val_pad = np.zeros((max(8, compact_capacity), 128), dtype=np.int32)
    val_pad[:compact_capacity, :] = val_np.astype(np.int32)[:, None]
    got = np.asarray(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (compact_capacity, expert_input_size), jnp.float32
            ),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec(memory_space=tpu.VMEM)] * 3,
                out_specs=pl.BlockSpec(memory_space=tpu.VMEM),
            ),
            interpret=False,
        )(src, jnp.asarray(idx_pad), jnp.asarray(val_pad))
    )

    src_bf_f32 = np.asarray(src.astype(jnp.float32))
    if batch_size == 32:
        want = np.where(val_np[:, None], src_bf_f32[idx_np], 0.0)
    else:
        want = np.zeros((compact_capacity, expert_input_size), dtype=np.float32)
        n_v = int(val_np.sum())
        for pos in range(n_v):
            want[pos] = src_bf_f32[idx_np[pos]]
    np.testing.assert_array_equal(got, want)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("batch_size", "compact_capacity"),
    [
        (16, 8),
        (32, 8),
        (32, 16),
        (64, 8),
    ],
)
def test_should_compile_and_match_routed_stream_stage_planned_rows_on_tpu(
    batch_size: int,
    compact_capacity: int,
) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend for Mosaic lowering")
    gu, gus, dn, dns = _make_mxfp4_weights(
        seed=31 + batch_size + compact_capacity, num_experts=8
    )
    rng = np.random.default_rng(200 + batch_size + compact_capacity)
    src = jnp.asarray(rng.standard_normal((batch_size, 512), dtype=np.float32)).astype(
        jnp.bfloat16
    )
    num_routes = 8
    sel_exp = jnp.asarray(
        rng.integers(0, 8, size=(batch_size, num_routes), dtype=np.int32)
    )
    sel_prob = jnp.asarray(
        rng.uniform(0.05, 1.0, size=(batch_size, num_routes)).astype(np.float32)
    )
    counts = jnp.asarray(rng.integers(1, 5, size=(batch_size,), dtype=np.int32))
    totals = jnp.maximum(jnp.sum(sel_prob, axis=1), 1.0)

    def body_routed(
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
        dn_acc,
        proc_vmem,
        c1_vmem,
        c2_vmem,
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

        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
        se = sel_exp_ref[...]
        sp = sel_prob_ref[...]
        cnt = cnt_ref[:, 0]
        tot = tot_ref[:, 0]
        rslots = jnp.arange(num_routes)[None, :]
        proc = jnp.zeros((batch_size, num_routes), jnp.bool_)
        for slot in range(4):
            pend = (rslots < cnt[:, None]) & ~proc
            min_e = jnp.min(jnp.where(pend, se.astype(jnp.float32), jnp.inf))
            has_e = jnp.any(pend)
            exp = jnp.where(has_e, min_e, 0).astype(jnp.int32)
            cons = pend & (se == exp)

            @pl.when(has_e)
            def _start(exp=exp, slot=slot):
                c = copy_expert(exp, slot)
                c.gate_up.start()
                c.gate_up_scales.start()
                c.down.start()
                c.down_scales.start()

            proc |= cons

        dm._run_routed_expert_stream(
            src_ref,
            se,
            sp,
            cnt,
            tot,
            gu_vmem,
            gus_vmem,
            dn_vmem,
            dns_vmem,
            out_ref,
            gu_acc,
            dn_acc,
            proc_vmem,
            c1_vmem,
            c2_vmem,
            copy_expert,
            expert_input_size=512,
            expert_width=512,
            expert_output_size=256,
            group_pending_routes=True,
        )

    actual = np.asarray(
        pl.pallas_call(
            body_routed,
            out_shape=jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec(memory_space=tpu.VMEM)] * 5
                + [pl.BlockSpec(memory_space=tpu.HBM)] * 4,
                out_specs=pl.BlockSpec(memory_space=tpu.VMEM),
                scratch_shapes=(
                    tpu.VMEM((4, *gu.shape[1:]), jnp.uint32),
                    tpu.VMEM((4, *gus.shape[1:]), jnp.uint8),
                    tpu.VMEM((4, *dn.shape[1:]), jnp.uint32),
                    tpu.VMEM((4, *dns.shape[1:]), jnp.uint8),
                    tpu.VMEM((batch_size, 2048), jnp.float32),
                    tpu.VMEM((batch_size, 256), jnp.float32),
                    tpu.VMEM((batch_size, 128), jnp.int32),
                    tpu.VMEM((compact_capacity, 512), jnp.float32),
                    tpu.VMEM((compact_capacity, 512), jnp.float32),
                    tpu.SemaphoreType.DMA((4, 4)),
                ),
            ),
            interpret=False,
        )(
            src,
            sel_exp,
            sel_prob,
            jnp.broadcast_to(counts[:, None], (batch_size, 128)),
            jnp.broadcast_to(totals[:, None], (batch_size, 128)),
            gu,
            gus,
            dn,
            dns,
        )
    )
    expected = _reference_moe(
        src,
        gu,
        gus,
        dn,
        dns,
        np.asarray(sel_exp),
        np.asarray(sel_prob),
        np.asarray(counts),
        np.asarray(totals),
        split_half_down=False,
    )
    rel_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert rel_l2 < 5e-3


@pytest.mark.tpu_test
@pytest.mark.parametrize("batch_size", [8, 16, 32, 64])
def test_should_compile_and_match_oracle_on_tpu_across_row_staging_regimes(
    batch_size: int,
) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend for Mosaic lowering")
    gu, gus, dn, dns = _make_mxfp4_weights(seed=17 + batch_size, num_experts=6)
    rng = np.random.default_rng(100 + batch_size)
    src = jnp.asarray(rng.standard_normal((batch_size, 512), dtype=np.float32)).astype(
        jnp.bfloat16
    )
    num_routes = 8
    sel_exp = jnp.asarray(
        rng.integers(0, 6, size=(batch_size, num_routes), dtype=np.int32)
    )
    sel_prob = jnp.asarray(
        rng.uniform(0.05, 1.0, size=(batch_size, num_routes)).astype(np.float32)
    )
    counts = jnp.asarray(rng.integers(0, 5, size=(batch_size,), dtype=np.int32))
    totals = jnp.maximum(jnp.sum(sel_prob, axis=1), 1.0)

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

        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
        se = sel_exp_ref[...]
        sp = sel_prob_ref[...]
        cnt = cnt_ref[:, 0]
        tot = tot_ref[:, 0]
        rslots = jnp.arange(num_routes)[None, :]
        proc = jnp.zeros((batch_size, num_routes), jnp.bool_)
        for slot in range(2):
            pend = (rslots < cnt[:, None]) & ~proc
            min_e = jnp.min(jnp.where(pend, se.astype(jnp.float32), jnp.inf))
            has_e = jnp.any(pend)
            exp = jnp.where(has_e, min_e, 0).astype(jnp.int32)
            cons = pend & (se == exp)

            @pl.when(has_e)
            def _start(exp=exp, slot=slot):
                c = copy_expert(exp, slot)
                c.gate_up.start()
                c.gate_up_scales.start()
                c.down.start()
                c.down_scales.start()

            proc |= cons

        dm.run_full_ep16_expert_stream(
            src_ref,
            se,
            sp,
            cnt,
            tot,
            gu_vmem,
            gus_vmem,
            dn_vmem,
            dns_vmem,
            out_ref,
            gu_acc,
            proc_vmem,
            copy_expert,
            expert_input_size=512,
            expert_width=512,
            expert_output_size=256,
            group_pending_routes=True,
        )

    actual = np.asarray(
        pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec(memory_space=tpu.VMEM)] * 5
                + [pl.BlockSpec(memory_space=tpu.HBM)] * 4,
                out_specs=pl.BlockSpec(memory_space=tpu.VMEM),
                scratch_shapes=(
                    tpu.VMEM((2, *gu.shape[1:]), jnp.uint32),
                    tpu.VMEM((2, *gus.shape[1:]), jnp.uint8),
                    tpu.VMEM((2, *dn.shape[1:]), jnp.uint32),
                    tpu.VMEM((2, *dns.shape[1:]), jnp.uint8),
                    tpu.VMEM((batch_size, 1024), jnp.float32),
                    tpu.VMEM((batch_size, 128), jnp.int32),
                    tpu.SemaphoreType.DMA((2, 4)),
                ),
            ),
            interpret=False,
        )(
            src,
            sel_exp,
            sel_prob,
            jnp.broadcast_to(counts[:, None], (batch_size, 128)),
            jnp.broadcast_to(totals[:, None], (batch_size, 128)),
            gu,
            gus,
            dn,
            dns,
        )
    )
    expected = _reference_moe(
        src,
        gu,
        gus,
        dn,
        dns,
        np.asarray(sel_exp),
        np.asarray(sel_prob),
        np.asarray(counts),
        np.asarray(totals),
        split_half_down=True,
    )
    rel_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert rel_l2 < 5e-3
