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
"""Admission rules of the fused MLA DSpark draft kernel."""

from __future__ import annotations

import dataclasses
import gc
import json
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import topologies
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    dspark,
    dspark_mla,
    dspark_mla_kernel,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.dspark_mla import MlaDraftConfig

if TYPE_CHECKING:
    from pathlib import Path

PAGES = 16


@pytest.mark.parametrize(
    ("geometry", "error"),
    [
        ({"requests": 1}, None),
        ({"requests": 8}, None),
        ({"requests": 16, "rows": 4, "block": 4}, None),
        ({"requests": 1, "rows": 6}, "4 or 8 rows"),
        ({"requests": 1, "block": 1}, "block must be"),
        ({"requests": 9}, "up to 8 requests"),
        ({"requests": 3, "attention_dp": 2}, "attention DP2"),
    ],
)
def test_fused_draft_admission(geometry: dict[str, int], error: str | None) -> None:
    """Supported geometries build a kernel; others raise naming the limit."""
    cfg = dspark_mla_kernel.MlaDraftLocalConfig(
        pages=PAGES,
        requests=geometry.get("requests", 1),
        block=geometry.get("block"),
        rows=geometry.get("rows", dspark_mla_kernel.ROWS),
        attention_dp=geometry.get("attention_dp", 1),
    )
    if error is None:
        kernel = dspark_mla_kernel.fused_draft_local(MlaDraftConfig(), cfg)
        assert callable(kernel)
    else:
        with pytest.raises(ValueError, match=error):
            dspark_mla_kernel.fused_draft_local(MlaDraftConfig(), cfg)


_RING_BLOCKS = 15
_RING_TOTAL_LENGTH = 20
_RING_FIRST_VALID = 12
_SMALL_RING_WINDOW = 16
_SMALL_RING_LAYERS = 2
_SMALL_RING_HEAD_DIM = 8


def test_dspark_ring_pack_unpack_and_positions_round_trip() -> None:
    """Verify DSpark ring packing, magic validation, slot positions, and overflow."""
    cfg = dspark.DraftConfig(
        layers=_SMALL_RING_LAYERS,
        window=_SMALL_RING_WINDOW,
        head_dim=_SMALL_RING_HEAD_DIM,
    )
    ring_shape = (1, cfg.layers, cfg.window, 2 * cfg.head_dim)
    ring = jnp.arange(
        cfg.layers * cfg.window * 2 * cfg.head_dim, dtype=jnp.bfloat16
    ).reshape(ring_shape)
    meta = jnp.array([[_RING_TOTAL_LENGTH, _RING_FIRST_VALID]], dtype=jnp.int32)

    packed = dspark.pack_ring(ring, meta, blocks=_RING_BLOCKS)
    unpacked_ring, unpacked_meta = dspark.unpack_ring(
        cfg, packed[0], length=_RING_TOTAL_LENGTH
    )
    assert jnp.array_equal(unpacked_ring, ring[0])
    assert tuple(int(x) for x in unpacked_meta) == (
        _RING_TOTAL_LENGTH,
        _RING_FIRST_VALID,
    )

    _, mismatch_meta = dspark.unpack_ring(cfg, packed[0], length=_RING_TOTAL_LENGTH + 1)
    assert tuple(int(x) for x in mismatch_meta) == (
        _RING_TOTAL_LENGTH + 1,
        _RING_TOTAL_LENGTH + 1,
    )

    positions = dspark.ring_positions(cfg, unpacked_meta)
    valid_positions = [int(p) for p in positions if int(p) >= 0]
    assert sorted(valid_positions) == list(range(_RING_FIRST_VALID, _RING_TOTAL_LENGTH))

    full_cfg = dspark.DraftConfig()
    huge_ring = jnp.zeros(
        (1, full_cfg.layers, full_cfg.window, 2 * full_cfg.head_dim),
        dtype=jnp.bfloat16,
    )
    with pytest.raises(ValueError, match="a ring needs"):
        dspark.pack_ring(huge_ring, meta, blocks=1)


def _verify_nucleus_accept_contract() -> None:
    """Verify frozen NucleusAcceptInputs and scalar/vmap nucleus_accept outputs."""
    zero_count = jnp.zeros((), jnp.int32)
    probs = np.array(
        [
            [0.3, 0.6, 0.1],
            [0.4, 0.5, 0.1],
            [0.7, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    values = jnp.asarray(np.log(probs), jnp.float32)
    ids = jnp.asarray([[11, 10, 12], [21, 20, 22], [30, 31, 32]], jnp.int32)
    lse = jnp.zeros((probs.shape[0],), jnp.float32)
    drafts = jnp.asarray([10, 20, -1], jnp.int32)
    uniforms = jnp.asarray([[0.2, 0.1], [0.9, 0.1], [0.5, 0.5]], jnp.float32)
    top_p = jnp.asarray(0.85, jnp.float32)

    reject_inputs = dspark.NucleusAcceptInputs(
        values=values,
        ids=ids,
        lse=lse,
        drafts=drafts,
        forced_count=jnp.asarray(-1, jnp.int32),
        top_p=top_p,
        uniforms=uniforms,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        reject_inputs.forced_count = zero_count

    count_reject, bonus_reject = dspark.nucleus_accept(reject_inputs)
    np.testing.assert_array_equal(count_reject, np.asarray(1, dtype=np.int32))
    np.testing.assert_array_equal(bonus_reject, np.asarray(21, dtype=np.int32))

    forced_inputs = dspark.NucleusAcceptInputs(
        values=values,
        ids=ids,
        lse=lse,
        drafts=drafts,
        forced_count=zero_count,
        top_p=top_p,
        uniforms=uniforms,
    )
    count_forced, bonus_forced = dspark.nucleus_accept(forced_inputs)
    np.testing.assert_array_equal(count_forced, np.asarray(0, dtype=np.int32))
    np.testing.assert_array_equal(bonus_forced, np.asarray(10, dtype=np.int32))

    batched_inputs = jax.tree.map(
        lambda first_item, second_item: jnp.stack([first_item, second_item]),
        reject_inputs,
        forced_inputs,
    )
    batched_counts, batched_bonuses = jax.vmap(dspark.nucleus_accept)(batched_inputs)
    np.testing.assert_array_equal(batched_counts, np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(batched_bonuses, np.asarray([21, 10], dtype=np.int32))


def _verify_ring_shards_contract() -> None:
    """Verify frozen ring shard specs and TP32 ring assembly/prefill output shapes."""
    zero_count = jnp.zeros((), jnp.int32)
    zero_bf16 = jnp.zeros((), jnp.bfloat16)
    cfg = dspark.DraftConfig()
    rows_per_rank = 4
    slots = 2
    window_rows = dspark.TP_SIZE * rows_per_rank
    shards_struct = dspark.DsparkRingWeightShards(
        fc=jax.ShapeDtypeStruct(
            (dspark.TP_SIZE, dspark.FC_IN, dspark.CHUNK), jnp.bfloat16
        ),
        kv=jax.ShapeDtypeStruct(
            (dspark.TP_SIZE, cfg.layers, cfg.hidden, 2 * cfg.head_dim), jnp.bfloat16
        ),
        kvnorm=jax.ShapeDtypeStruct(
            (dspark.TP_SIZE, cfg.layers, 1, 2 * cfg.head_dim), jnp.bfloat16
        ),
        hidden_norm=jax.ShapeDtypeStruct((dspark.TP_SIZE, 1, cfg.hidden), jnp.bfloat16),
    )
    spec_struct = dspark.DsparkRingUpdateSpec(
        aux=jax.ShapeDtypeStruct(
            (len(cfg.aux_layers), window_rows, cfg.hidden), jnp.bfloat16
        ),
        positions=jax.ShapeDtypeStruct((window_rows,), jnp.int32),
        sequences=jax.ShapeDtypeStruct((window_rows,), jnp.int32),
        rings=jax.ShapeDtypeStruct(
            (dspark.TP_SIZE, slots, cfg.layers, cfg.window, 2 * cfg.head_dim),
            jnp.bfloat16,
        ),
        ends=jax.ShapeDtypeStruct((slots,), jnp.int32),
        stops=jax.ShapeDtypeStruct((slots,), jnp.int32),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        shards_struct.fc = zero_bf16
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec_struct.ends = zero_count

    topo = topologies.get_topology_desc(topology_name="tpu7x_2x2x4", platform="tpu")
    assert topo is not None
    mesh_tp = Mesh(topo.devices, (dspark.AXIS_TP,))

    def _eval_ring_contracts(
        shards_in: dspark.DsparkRingWeightShards,
        spec_in: dspark.DsparkRingUpdateSpec,
    ) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
        """Trace ring weight assembly and prefill ring update on TP32."""
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
        return ring_weights, after[None], at[None]

    spec_in_sharding = dspark.DsparkRingUpdateSpec(
        aux=P(None, dspark.AXIS_TP),
        positions=P(),
        sequences=P(),
        rings=P(dspark.AXIS_TP),
        ends=P(),
        stops=P(),
    )
    sharded_eval = jax.shard_map(
        _eval_ring_contracts,
        mesh=mesh_tp,
        in_specs=(P(dspark.AXIS_TP), spec_in_sharding),
        out_specs=(P(), P(dspark.AXIS_TP), P(dspark.AXIS_TP)),
        check_vma=False,
    )
    try:
        weights_out, after_out, at_out = jax.eval_shape(
            sharded_eval, shards_struct, spec_struct
        )
    finally:
        jax.clear_caches()
        gc.collect()

    expected_weight_specs = {
        "fc": ((cfg.context_width, cfg.hidden), jnp.dtype(jnp.bfloat16)),
        "kv": (
            (cfg.layers, cfg.hidden, cfg.kv_heads, 2 * cfg.head_dim),
            jnp.dtype(jnp.bfloat16),
        ),
        "k_norm": ((cfg.layers, cfg.head_dim), jnp.dtype(jnp.bfloat16)),
        "hidden_norm": ((cfg.hidden,), jnp.dtype(jnp.bfloat16)),
    }
    actual_weight_specs = {
        name: (arr.shape, arr.dtype) for name, arr in weights_out.items()
    }
    assert actual_weight_specs == expected_weight_specs
    expected_ring_shape = (
        dspark.TP_SIZE,
        slots,
        cfg.layers,
        cfg.window,
        2 * cfg.head_dim,
    )
    assert (after_out.shape, after_out.dtype) == (
        expected_ring_shape,
        jnp.dtype(jnp.bfloat16),
    )
    assert (at_out.shape, at_out.dtype) == (
        expected_ring_shape,
        jnp.dtype(jnp.bfloat16),
    )


def test_dspark_nucleus_accept_inputs_and_ring_shards_contracts() -> None:
    """Verify frozen DSpark input specs, nucleus sampling, and ring shape contracts."""
    _verify_nucleus_accept_contract()
    _verify_ring_shards_contract()


def test_dspark_kernel_weight_shards_validation_and_shard_dir_round_trip(
    tmp_path: Path,
) -> None:
    """Verify kernel weight shard validation and binary rank-shard round-trip."""
    num_ranks = dspark.TP_SIZE
    zero_ranks = 0
    single_rank = 1
    vocab_offset = 1
    cols = 2
    first_rank = 0
    last_rank = num_ranks - single_rank
    scale_step = 0.25

    default_cfg = dspark.DraftConfig()
    non_divisible_cfg = dspark.DraftConfig(vocab=default_cfg.vocab + vocab_offset)
    single_rank_cfg = dspark.DraftConfig(vocab=dspark.LM_CHUNKS * dspark.LM_CHUNK)
    empty_weights: dict[str, np.ndarray] = {}
    with pytest.raises(ValueError, match="divide rank count"):
        dspark.kernel_weight_shards(default_cfg, empty_weights, ranks=zero_ranks)
    with pytest.raises(ValueError, match="divide rank count"):
        dspark.kernel_weight_shards(non_divisible_cfg, empty_weights, ranks=num_ranks)
    with pytest.raises(ValueError, match="divide rank count"):
        dspark.kernel_weight_shards(single_rank_cfg, empty_weights, ranks=single_rank)

    total_elems = num_ranks * cols
    f32_shard = np.arange(total_elems, dtype=np.float32).reshape(num_ranks, cols)
    bf16_shard = np.asarray(jnp.asarray(f32_shard * scale_step, dtype=jnp.bfloat16))
    shards = {"proj": bf16_shard, "scale": f32_shard}

    dspark_dir = tmp_path / "dspark_shards"
    dspark.save_kernel_shards(shards, dspark_dir, ranks=num_ranks)
    dspark_index = json.loads((dspark_dir / "index.json").read_text())
    assert "kind" not in dspark_index
    assert dspark_index["ranks"] == num_ranks

    mla_dir = tmp_path / "mla_shards"
    dspark_mla.save_kernel_shards(shards, mla_dir, ranks=num_ranks)
    mla_index = json.loads((mla_dir / "index.json").read_text())
    assert mla_index["kind"] == dspark_mla.DRAFT_KIND
    assert mla_index["ranks"] == num_ranks

    for rank in (first_rank, last_rank):
        loaded = dspark_mla.load_rank_shards(mla_dir, rank)
        assert set(loaded) == set(shards)
        for name, expected in shards.items():
            np.testing.assert_array_equal(loaded[name], expected[rank])
