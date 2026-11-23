# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# Source provenance: inferact/k3-megakernel@30c9ba966b
#   tests/megakernel/kimi_k3/test_dspark_attention_dp.py
"""Tests for DSpark attention-DP schedules and TP32 fused draft AOT lowering."""

from __future__ import annotations

import base64
import gc
import re
import types
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import topologies
from jax.experimental.mosaic.dialects import tpu as mosaic_tpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxlib.mlir import ir as mlir_ir
from jaxlib.xla_client import hlo as xla_hlo

from vllm_torchtpu.models.kimi_k3.megakernel.decode import KimiDecoder
from vllm_torchtpu.models.kimi_k3.megakernel.decode_ops import (
    Control,
    dp_group_rows,
)
from vllm_torchtpu.models.kimi_k3.megakernel.draft import DsparkDraft, _batched_fn
from vllm_torchtpu.models.kimi_k3.megakernel.draft_mla import MlaDsparkDraft
from vllm_torchtpu.models.kimi_k3.megakernel.draft_mla import (
    _batched_fn as _mla_batched_fn,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    dspark,
    dspark_mla,
    dspark_mla_kernel,
)
from vllm_torchtpu.models.kimi_k3.megakernel.program import spec_tokens_per_batch
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)

_MAX_PART_REQUESTS = 8


def _assert_structural_hlo(
    lowered: jax.stages.Lowered,
    *,
    expected_custom_calls: int,
    expected_out_sharding: str,
) -> None:
    """Assert StableHLO sharding, custom-call count, MLIR parse, no partition-id."""
    stablehlo_text = lowered.as_text()
    assert expected_out_sharding in stablehlo_text, (
        f"Missing expected output sharding {expected_out_sharding!r}"
    )
    assert "AllocateBuffer" not in stablehlo_text
    custom_call_count = stablehlo_text.count("tpu_custom_call")
    assert custom_call_count == expected_custom_calls, (
        f"Expected {expected_custom_calls} tpu_custom_call op(s), "
        f"got {custom_call_count}"
    )

    hlo_comp = lowered.compiler_ir(dialect="hlo")
    assert hlo_comp is not None
    opts = xla_hlo.HloPrintOptions.canonical()
    opts.print_backend_config = True
    opts.print_metadata = False
    canonical_hlo = hlo_comp.as_hlo_module().to_string(opts)
    assert "tpu_custom_call" in canonical_hlo
    assert "partition-id" not in canonical_hlo

    parsed_bodies = 0
    for match in re.finditer(r'"body"\s*:\s*"([^"]+)"', canonical_hlo):
        raw_bytes = base64.b64decode(match.group(1))
        with mlir_ir.Context() as ctx:
            mosaic_tpu.register_dialect(ctx)
            ctx.allow_unregistered_dialects = True
            module = mlir_ir.Module.parse(raw_bytes)
            asm_text = str(module.operation.get_asm(enable_debug_info=False))
            assert "func.func" in asm_text
        parsed_bodies += 1
    assert parsed_bodies >= 1


@pytest.mark.parametrize("batch", [8, 16, 32, 64, 96, 128])
def test_verify_rows_are_the_rows_each_group_attends(batch: int) -> None:
    """Verify dp_group_rows matches KimiDecoder._group_rows for each DP group."""
    attention_dp = 2
    rows = dp_group_rows(batch, attention_dp)
    half = batch // attention_dp

    assert rows.shape == (batch,)
    assert rows.dtype == np.int32
    group_zero = rows[:half]
    group_one = rows[half:]
    assert set(group_zero.tolist()).isdisjoint(group_one.tolist())
    assert sorted(rows.tolist()) == list(range(batch))

    decoder = cast("KimiDecoder", types.SimpleNamespace(groups=attention_dp))
    group_rows_fn = KimiDecoder.__dict__["_group_rows"]
    expected = [group_rows_fn(decoder, batch, group) for group in range(attention_dp)]
    assert group_zero.tolist() == expected[0]
    assert group_one.tolist() == expected[1]
    assert rows.tolist() == expected[0] + expected[1]


def _stub_fused(
    _weights: object,
    step: dspark.FusedDraftStepInputs,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Deterministic stub for DSpark fused_draft in multi-part DP tests."""
    assert step.forced_anchor is not None
    out = jnp.zeros((step.position.shape[0], 128), jnp.int32)
    out = (
        out.at[:, dspark.OUT_COUNT]
        .set(step.forced_count)
        .at[:, dspark.OUT_BONUS]
        .set(step.forced_anchor)
    )
    # Outputs that depend on each request's own verify rows.
    out = out.at[:, 40].set(step.aux[0, :: dspark.ROWS, 0].astype(jnp.int32))
    out = out.at[:, 41].set(step.logits[0, :, 0, 0].astype(jnp.int32))
    return out, step.cache + 1, step.positions + step.position[None, :, None]


@pytest.mark.parametrize("verified", [24, 32])
def test_drafting_in_parts_matches_one_call(verified: int) -> None:
    """Verify multi-part DSpark drafting matches a single full-batch call."""
    rng = np.random.default_rng(verified)
    rows_per, pages = 4, 3
    control = Control(slots=65, rows=64, pages=pages)
    slots = rng.permutation(np.arange(1, 65))[:verified]
    slots[[3, verified - 2]] = 0  # padding requests share the null slot
    entries = [
        (
            int(slot_id),
            idx % 3 == 0,
            int(rng.integers(5, 999)),
            int(rng.integers(9, 9000)),
            (0.0, 1.0, 0, 0),
        )
        for idx, slot_id in enumerate(slots)
    ]
    plan = control.plan(entries, [np.arange(pages, dtype=np.int32)] * verified)
    state = np.concatenate([rng.integers(0, 50, 3 * control.slots), plan]).astype(
        np.int32
    )
    total = verified * rows_per
    ids = jnp.asarray(rng.integers(0, 1000, 4 * total + verified * pages), jnp.int32)
    aux = jnp.asarray(rng.integers(0, 99, (5, total, 16)), jnp.bfloat16)
    logits = jnp.asarray(rng.integers(0, 99, (32, total, 32)), jnp.float32)
    decisions = jnp.asarray(
        np.stack(
            [rng.integers(0, rows_per, verified), rng.integers(0, 999, verified)], 1
        ),
        jnp.int32,
    )

    def draft(requests: int, first: int, state_arr: jax.Array) -> tuple[jax.Array, ...]:
        """Run one _batched_fn slice over [first, first + requests)."""
        fn = _batched_fn(
            dspark, _stub_fused, control, (), requests, rows_per, first, verified
        )
        return fn(
            jnp.zeros((1, requests, 2, 4, 8), jnp.bfloat16),
            jnp.zeros((1, requests, 4), jnp.int32),
            jnp.zeros((1, requests, 128), jnp.int32),
            aux,
            logits,
            ids,
            state_arr,
            decisions,
        )

    whole = draft(verified, 0, jnp.asarray(state))
    parts, part_state = [], jnp.asarray(state)
    for first, requests in DsparkDraft.parts(verified, rows_per):
        out = draft(requests, first, part_state)
        parts.append(out)
        part_state = out[4]
    np.testing.assert_array_equal(
        np.concatenate([p[0] for p in parts], axis=1), whole[0]
    )
    np.testing.assert_array_equal(np.concatenate([p[3] for p in parts]), whole[3])
    np.testing.assert_array_equal(part_state, whole[4])


def _stub_mla_fused(
    _weights: object,
    step: dspark_mla_kernel.MlaDraftStepInputs,
) -> tuple[jax.Array, jax.Array]:
    """Deterministic stub for MLA fused_draft_local in multi-part DP tests."""
    out = jnp.zeros((step.position.shape[0], 128), jnp.int32)
    out = (
        out.at[:, dspark.OUT_COUNT]
        .set(step.forced_count)
        .at[:, dspark.OUT_BONUS]
        .set(step.forced_anchor)
    )
    # Outputs that depend on each request's own operands.
    out = out.at[:, 40].set(step.aux[0, :: step.logits.shape[1], 0].astype(jnp.int32))
    out = out.at[:, 41].set(step.logits[:, 0, 0].astype(jnp.int32))
    out = out.at[:, 42].set(step.tables[:, 0]).at[:, 43].set(step.previous[:, 0])
    rows = step.aux[0, :, :1].astype(jnp.float32)
    return out, jnp.stack([rows, rows + 1])  # [layers, rows, 1] context latents


def _stub_commit(
    pools: tuple[jax.Array, ...],
    latents: jax.Array,
    _tables: object,
    positions: jax.Array,
) -> tuple[jax.Array, ...]:
    """Deterministic stub for MLA latent pool commit."""
    return tuple(
        pool.at[positions].set(latents[layer, :, 0]) for layer, pool in enumerate(pools)
    )


@pytest.mark.parametrize(("verified", "dp"), [(16, 1), (16, 2), (32, 2)])
def test_mla_drafting_in_parts_matches_one_call(verified: int, dp: int) -> None:
    """Verify multi-part MLA drafting matches single-call MLA drafting."""
    rng = np.random.default_rng(verified + dp)
    rows_per, pages = 4, 3
    control = Control(slots=65, rows=64, pages=pages)
    slots = rng.permutation(np.arange(1, 65))[:verified]
    entries = [
        (int(slot_id), idx % 3 == 0, int(rng.integers(5, 999)), 0, (0.0, 1.0, 0, 0))
        for idx, slot_id in enumerate(slots)
    ]
    plan = control.plan(entries, [np.arange(pages, dtype=np.int32)] * verified)
    state = jnp.asarray(
        np.concatenate([rng.integers(0, 50, 3 * control.slots), plan]), jnp.int32
    )
    total = verified * rows_per
    bases = 100 * rng.permutation(verified)  # distinct positions per request
    positions = (bases[:, None] + np.arange(rows_per)[None, :]).reshape(-1)
    ids = np.concatenate(
        [
            rng.integers(0, 1000, total),
            positions,
            rng.integers(0, 1000, 2 * total),
            rng.integers(0, 1000, verified * pages),
        ]
    )
    ids_arr = jnp.asarray(ids, jnp.int32)
    aux = jnp.asarray(rng.integers(0, 99, (5, total, 16)), jnp.bfloat16)
    logits = jnp.asarray(rng.integers(0, 99, (1, total, 32)), jnp.float32)
    decisions = jnp.asarray(
        np.stack(
            [rng.integers(0, rows_per, verified), rng.integers(0, 999, verified)], 1
        ),
        jnp.int32,
    )
    previous = jnp.asarray(rng.integers(0, 999, (verified, 128)), jnp.int32)
    pools = (jnp.zeros(100 * verified, jnp.float32),) * 2

    def draft(
        index: tuple[int, ...], state_arr: jax.Array, pools_tuple: tuple[jax.Array, ...]
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], jax.Array, jax.Array]:
        """Run one _mla_batched_fn slice over the selected request indices."""
        fn = _mla_batched_fn(
            dspark,
            _stub_mla_fused,
            _stub_commit,
            control,
            (),
            2,
            index,
            verified,
            rows_per,
        )
        num_active = len(index)
        out = fn(
            *pools_tuple,
            *(previous[j : j + 1] for j in index),
            aux,
            logits,
            ids_arr,
            state_arr,
            decisions,
        )
        return (
            out[:num_active],
            out[num_active : num_active + 2],
            out[num_active + 2],
            out[num_active + 3],
        )

    whole = draft(tuple(range(verified)), state, pools)
    vectors: list[jax.Array | None] = [None] * verified
    emitted: list[jax.Array] = []
    order: list[int] = []
    part_state, part_pools = state, pools
    for index in MlaDsparkDraft.parts(verified, dp):
        # At most eight a call; under DP as many of each group's, group-major.
        assert len(index) <= _MAX_PART_REQUESTS
        assert [j // (verified // dp) for j in index] == sorted(
            list(range(dp)) * (len(index) // dp)
        )
        part_vectors, part_pools, part_emitted, part_state = draft(
            index, part_state, part_pools
        )
        for j, vector in zip(index, part_vectors, strict=True):
            vectors[j] = vector
        emitted.append(part_emitted)
        order.extend(index)
    assert sorted(order) == list(range(verified))
    filled_vectors = [v for v in vectors if v is not None]
    assert len(filled_vectors) == verified
    np.testing.assert_array_equal(
        np.concatenate(filled_vectors), np.concatenate(whole[0])
    )
    for part_pool, whole_pool in zip(part_pools, whole[1], strict=True):
        np.testing.assert_array_equal(part_pool, whole_pool)
    np.testing.assert_array_equal(np.concatenate(emitted)[np.argsort(order)], whole[2])
    np.testing.assert_array_equal(part_state, whole[3])


def _config(
    attention_dp: int,
    schedule: tuple[tuple[int, int, int], ...],
    max_num_seqs: int = 32,
) -> MegakernelEngineConfig:
    """Build a test MegakernelEngineConfig for speculative schedule checks."""
    return MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32 // attention_dp,
        block_size=512,
        kv_cache_dtype="fp8",
        max_num_seqs=max_num_seqs,
        max_model_len=1 << 17,
        max_num_batched_tokens=512,
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=schedule,
        attention_dp=attention_dp,
    )


def test_attention_dp2_speculates_up_to_32_requests() -> None:
    """Verify spec_tokens_per_batch schedule resolution under attention_dp=2."""
    schedule = ((1, 2, 7), (3, 32, 3))
    dense = spec_tokens_per_batch(_config(2, schedule))
    assert dense is not None
    assert dense[1:3] == (7, 7)
    assert set(dense[3:]) == {3}
    with pytest.raises(NotImplementedError):
        spec_tokens_per_batch(_config(1, schedule))
    with pytest.raises(NotImplementedError):
        spec_tokens_per_batch(_config(2, ((1, 16, 7),)))


def test_dspark_mla_verify8_lowers_cleanly() -> None:
    """Verify DSpark MLA verify8 lowers cleanly with expected HLO structure."""
    topo = topologies.get_topology_desc(topology_name="tpu7x_2x2x4", platform="tpu")
    assert topo is not None
    mesh_tp = Mesh(topo.devices, (dspark.AXIS_TP,))
    c_draft = dspark_mla.MlaDraftConfig()
    num_requests = 2
    rows = 8
    pages = 64
    block = 7
    attention_dp = 1
    total_rows = num_requests * rows

    cfg = dspark_mla_kernel.MlaDraftLocalConfig(
        pages=pages,
        requests=num_requests,
        block=block,
        rows=rows,
        attention_dp=attention_dp,
    )
    fused_draft = dspark_mla_kernel.fused_draft_local(c_draft, cfg)
    weights = {
        "ln1": jax.ShapeDtypeStruct((c_draft.layers, 1, c_draft.hidden), jnp.bfloat16),
        "ln2": jax.ShapeDtypeStruct((c_draft.layers, 1, c_draft.hidden), jnp.bfloat16),
        "q_norm": jax.ShapeDtypeStruct(
            (c_draft.layers, 1, c_draft.q_lora), jnp.bfloat16
        ),
        "kv_norm": jax.ShapeDtypeStruct((c_draft.layers, 1, 640), jnp.bfloat16),
        "context_norm": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "final_norm": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "mask_embed": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "rot": jax.ShapeDtypeStruct((c_draft.rope, c_draft.rope), jnp.bfloat16),
        "qkv_tile": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.hidden, 128), jnp.bfloat16
        ),
        "context_tile": jax.ShapeDtypeStruct((c_draft.hidden, 128), jnp.bfloat16),
        "q_b": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.q_lora, 512), jnp.bfloat16
        ),
        "w_uk": jax.ShapeDtypeStruct(
            (c_draft.layers, 2, c_draft.nope, c_draft.kv_lora), jnp.bfloat16
        ),
        "w_uv": jax.ShapeDtypeStruct(
            (c_draft.layers, 2, c_draft.kv_lora, c_draft.v_head), jnp.bfloat16
        ),
        "o": jax.ShapeDtypeStruct(
            (c_draft.layers, 2 * c_draft.v_head, c_draft.hidden), jnp.bfloat16
        ),
        "gate": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.hidden, 512), jnp.bfloat16
        ),
        "up": jax.ShapeDtypeStruct((c_draft.layers, c_draft.hidden, 512), jnp.bfloat16),
        "down": jax.ShapeDtypeStruct(
            (c_draft.layers, 512, c_draft.hidden), jnp.bfloat16
        ),
        "fc": jax.ShapeDtypeStruct((8960, 896), jnp.bfloat16),
        "w2t": jax.ShapeDtypeStruct((256, c_draft.vocab // 32), jnp.bfloat16),
        "w1": jax.ShapeDtypeStruct((c_draft.vocab, 256), jnp.bfloat16),
        "rope": jax.ShapeDtypeStruct((512, 2 * c_draft.rope), jnp.float32),
        "lm_head": jax.ShapeDtypeStruct(
            (c_draft.hidden, c_draft.vocab // 32), jnp.bfloat16
        ),
        "embed": jax.ShapeDtypeStruct(
            (c_draft.vocab // 32, c_draft.hidden), jnp.bfloat16
        ),
    }
    pools = tuple(
        jax.ShapeDtypeStruct((16, 128, 4, 640), jnp.float8_e4m3fn)
        for _ in range(c_draft.layers)
    )
    aux = jax.ShapeDtypeStruct(
        (len(c_draft.aux_layers), total_rows, c_draft.hidden), jnp.bfloat16
    )
    logits = jax.ShapeDtypeStruct(
        (num_requests, rows, c_draft.vocab // 32), jnp.float32
    )
    previous = jax.ShapeDtypeStruct((num_requests, 128), jnp.int32)
    position = jax.ShapeDtypeStruct((num_requests,), jnp.int32)
    forced_count = jax.ShapeDtypeStruct((num_requests,), jnp.int32)
    forced_anchor = jax.ShapeDtypeStruct((num_requests,), jnp.int32)
    tables = jax.ShapeDtypeStruct((num_requests, pages), jnp.int32)

    step = dspark_mla_kernel.MlaDraftStepInputs(
        pools=pools,
        aux=aux,
        logits=logits,
        previous=previous,
        position=position,
        forced_count=forced_count,
        forced_anchor=forced_anchor,
        tables=tables,
    )
    try:
        part_a_sharded = jax.shard_map(
            fused_draft,
            mesh=mesh_tp,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
        lowered_draft = jax.jit(part_a_sharded).lower(weights, step)
        out_tokens, out_latents = lowered_draft.out_info
        assert (out_tokens.shape, out_tokens.dtype) == (
            (num_requests, 128),
            jnp.dtype(jnp.int32),
        )
        assert (out_latents.shape, out_latents.dtype) == (
            (c_draft.layers, total_rows, dspark_mla.LANES),
            jnp.dtype(jnp.bfloat16),
        )
        _assert_structural_hlo(
            lowered_draft,
            expected_custom_calls=1,
            expected_out_sharding=(
                "out_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}, {}]>]"
            ),
        )
    finally:
        jax.clear_caches()
        gc.collect()


def test_dspark_make_fused_draft_lowers_cleanly() -> None:
    """Verify TP32 DSpark make_fused_draft lowers with expected HLO structure."""
    topo = topologies.get_topology_desc(topology_name="tpu7x_2x2x4", platform="tpu")
    assert topo is not None
    mesh_tp = Mesh(topo.devices, (dspark.AXIS_TP,))
    cfg = dspark.DraftConfig()
    rope_len = 8192 + 32
    block = 8
    requests = 2
    tp_sharding = NamedSharding(mesh_tp, P(dspark.AXIS_TP))
    rep_sharding = NamedSharding(mesh_tp, P())

    weight_specs = {
        "ln1": ((32, 5, 1, 7168), jnp.bfloat16),
        "ln2": ((32, 5, 1, 7168), jnp.bfloat16),
        "qnorm": ((32, 5, 1, 192), jnp.bfloat16),
        "kvnorm": ((32, 5, 1, 128), jnp.bfloat16),
        "hidden_norm": ((32, 1, 7168), jnp.bfloat16),
        "final_norm": ((32, 1, 7168), jnp.bfloat16),
        "mask_embed": ((32, 1, 7168), jnp.bfloat16),
        "rot_q": ((32, 192, 192), jnp.bfloat16),
        "rot_kv": ((32, 128, 128), jnp.bfloat16),
        "head_ones": ((32, 192, 192), jnp.bfloat16),
        "kv_ones": ((32, 128, 128), jnp.bfloat16),
        "head_select": ((32, 3, 192, 128), jnp.bfloat16),
        "head_place": ((32, 3, 128, 192), jnp.bfloat16),
        "q": ((32, 5, 7168, 192), jnp.bfloat16),
        "kv": ((32, 5, 7168, 128), jnp.bfloat16),
        "o": ((32, 5, 192, 7168), jnp.bfloat16),
        "gate": ((32, 5, 7168, 512), jnp.bfloat16),
        "up": ((32, 5, 7168, 512), jnp.bfloat16),
        "down": ((32, 5, 512, 7168), jnp.bfloat16),
        "fc": ((32, 8960, 896), jnp.bfloat16),
        "lm_head": ((32, 10, 7168, 512), jnp.bfloat16),
        "w2t": ((32, 256, 5120), jnp.bfloat16),
        "w1": ((32, 163840, 256), jnp.bfloat16),
        "embed": ((32, 5120, 7168), jnp.bfloat16),
        "rope_q": ((32, rope_len, 384), jnp.float32),
        "rope_kv": ((32, rope_len, 256), jnp.float32),
    }
    weights = {
        name: jax.ShapeDtypeStruct(shape, dtype, sharding=tp_sharding)
        for name, (shape, dtype) in weight_specs.items()
    }
    step = dspark.FusedDraftStepInputs(
        cache=jax.ShapeDtypeStruct(
            (32, 2, 5, 2048, 128), jnp.bfloat16, sharding=tp_sharding
        ),
        positions=jax.ShapeDtypeStruct((1, 2, 2048), jnp.int32, sharding=rep_sharding),
        aux=jax.ShapeDtypeStruct((5, 16, 7168), jnp.bfloat16, sharding=rep_sharding),
        logits=jax.ShapeDtypeStruct(
            (32, 2, 8, 5120), jnp.float32, sharding=tp_sharding
        ),
        previous=jax.ShapeDtypeStruct((1, 2, 128), jnp.int32, sharding=rep_sharding),
        position=jax.ShapeDtypeStruct((2,), jnp.int32, sharding=rep_sharding),
        forced_count=jax.ShapeDtypeStruct((2,), jnp.int32, sharding=rep_sharding),
        forced_anchor=jax.ShapeDtypeStruct((2,), jnp.int32, sharding=rep_sharding),
    )
    fused = dspark.make_fused_draft(cfg, mesh_tp, block, requests=requests)
    try:
        lowered = jax.jit(fused).lower(weights, step)
        out_tokens, out_cache, out_positions = lowered.out_info
        assert (out_tokens.shape, out_tokens.dtype) == (
            (requests, 128),
            jnp.dtype(jnp.int32),
        )
        assert (out_cache.shape, out_cache.dtype) == (
            (32, requests, cfg.layers, cfg.window, 2 * cfg.head_dim),
            jnp.dtype(jnp.bfloat16),
        )
        assert (out_positions.shape, out_positions.dtype) == (
            (1, requests, cfg.window),
            jnp.dtype(jnp.int32),
        )
        _assert_structural_hlo(
            lowered,
            expected_custom_calls=1,
            expected_out_sharding=(
                "out_shardings=[<@mesh, [{}, {}]>, "
                f'<@mesh, [{{"{dspark.AXIS_TP}"}}, {{}}, {{}}, {{}}, {{}}]>, '
                "<@mesh, [{}, {}, {}]>]"
            ),
        )
    finally:
        jax.clear_caches()
        gc.collect()


def test_dspark_prefill_ring_update_lowers_without_partition_id() -> None:
    """Verify prefill_ring_update lowers via mesh_axis_index with zero partition-id."""
    topo = topologies.get_topology_desc(topology_name="tpu7x_2x2x4", platform="tpu")
    assert topo is not None
    mesh_tp = Mesh(topo.devices, (dspark.AXIS_TP,))
    cfg = dspark.DraftConfig()
    rows_per_rank = 4
    slots = 2
    window_rows = dspark.TP_SIZE * rows_per_rank
    tp_sharding = NamedSharding(mesh_tp, P(dspark.AXIS_TP))
    rep_sharding = NamedSharding(mesh_tp, P())

    shards_struct = dspark.DsparkRingWeightShards(
        fc=jax.ShapeDtypeStruct(
            (dspark.TP_SIZE, dspark.FC_IN, dspark.CHUNK),
            jnp.bfloat16,
            sharding=tp_sharding,
        ),
        kv=jax.ShapeDtypeStruct(
            (dspark.TP_SIZE, cfg.layers, cfg.hidden, 2 * cfg.head_dim),
            jnp.bfloat16,
            sharding=tp_sharding,
        ),
        kvnorm=jax.ShapeDtypeStruct(
            (dspark.TP_SIZE, cfg.layers, 1, 2 * cfg.head_dim),
            jnp.bfloat16,
            sharding=tp_sharding,
        ),
        hidden_norm=jax.ShapeDtypeStruct(
            (dspark.TP_SIZE, 1, cfg.hidden), jnp.bfloat16, sharding=tp_sharding
        ),
    )
    spec_struct = dspark.DsparkRingUpdateSpec(
        aux=jax.ShapeDtypeStruct(
            (len(cfg.aux_layers), window_rows, cfg.hidden),
            jnp.bfloat16,
            sharding=NamedSharding(mesh_tp, P(None, dspark.AXIS_TP)),
        ),
        positions=jax.ShapeDtypeStruct(
            (window_rows,), jnp.int32, sharding=rep_sharding
        ),
        sequences=jax.ShapeDtypeStruct(
            (window_rows,), jnp.int32, sharding=rep_sharding
        ),
        rings=jax.ShapeDtypeStruct(
            (dspark.TP_SIZE, slots, cfg.layers, cfg.window, 2 * cfg.head_dim),
            jnp.bfloat16,
            sharding=tp_sharding,
        ),
        ends=jax.ShapeDtypeStruct((slots,), jnp.int32, sharding=rep_sharding),
        stops=jax.ShapeDtypeStruct((slots,), jnp.int32, sharding=rep_sharding),
    )

    def _run_ring_update(
        shards_in: dspark.DsparkRingWeightShards,
        spec_in: dspark.DsparkRingUpdateSpec,
    ) -> tuple[jax.Array, jax.Array]:
        """Trace ring weight assembly and prefill_ring_update inside shard_map."""
        local_shards = dspark.DsparkRingWeightShards(
            fc=shards_in.fc[0],
            kv=shards_in.kv[0],
            kvnorm=shards_in.kvnorm[0],
            hidden_norm=shards_in.hidden_norm[0],
        )
        ring_weights = dspark.ring_weights_from_shards(
            cfg, local_shards, axis=dspark.AXIS_TP
        )
        local_spec = dspark.DsparkRingUpdateSpec(
            aux=spec_in.aux,
            positions=spec_in.positions,
            sequences=spec_in.sequences,
            rings=spec_in.rings[0],
            ends=spec_in.ends,
            stops=spec_in.stops,
        )
        after, at = dspark.prefill_ring_update(
            cfg, ring_weights, local_spec, axis=dspark.AXIS_TP
        )
        return after[None], at[None]

    spec_in_sharding = dspark.DsparkRingUpdateSpec(
        aux=P(None, dspark.AXIS_TP),
        positions=P(),
        sequences=P(),
        rings=P(dspark.AXIS_TP),
        ends=P(),
        stops=P(),
    )
    sharded_update = jax.shard_map(
        _run_ring_update,
        mesh=mesh_tp,
        in_specs=(P(dspark.AXIS_TP), spec_in_sharding),
        out_specs=(P(dspark.AXIS_TP), P(dspark.AXIS_TP)),
        check_vma=False,
    )
    try:
        lowered = jax.jit(sharded_update).lower(shards_struct, spec_struct)
        out_after, out_at = lowered.out_info
        expected_shape = (
            dspark.TP_SIZE,
            slots,
            cfg.layers,
            cfg.window,
            2 * cfg.head_dim,
        )
        assert (out_after.shape, out_after.dtype) == (
            expected_shape,
            jnp.dtype(jnp.bfloat16),
        )
        assert (out_at.shape, out_at.dtype) == (
            expected_shape,
            jnp.dtype(jnp.bfloat16),
        )
        ring_sharding = f'<@mesh, [{{"{dspark.AXIS_TP}"}}, {{}}, {{}}, {{}}, {{}}]>'
        _assert_structural_hlo(
            lowered,
            expected_custom_calls=1,
            expected_out_sharding=f"out_shardings=[{ring_sharding}, {ring_sharding}]",
        )
    finally:
        jax.clear_caches()
        gc.collect()
