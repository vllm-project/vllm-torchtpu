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
"""Tests for DSpark admission, ring packing, nucleus sampling, and shard-dir I/O."""

from __future__ import annotations

import dataclasses
import gc
import json
from pathlib import Path
from typing import IO, cast

import jax
import jax.numpy as jnp
import ml_dtypes
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
    cols = 2
    scale_step = 0.25

    default_cfg = dspark.DraftConfig()
    non_divisible_cfg = dspark.DraftConfig(vocab=default_cfg.vocab + 1)
    single_rank_cfg = dspark.DraftConfig(vocab=dspark.LM_CHUNKS * dspark.LM_CHUNK)
    empty_weights: dict[str, np.ndarray] = {}
    with pytest.raises(ValueError, match=r"divide rank count.*ranks=0"):
        dspark.kernel_weight_shards(default_cfg, empty_weights, ranks=0)
    with pytest.raises(
        ValueError,
        match=rf"divide rank count.*vocab={non_divisible_cfg.vocab}",
    ):
        dspark.kernel_weight_shards(non_divisible_cfg, empty_weights, ranks=num_ranks)
    with pytest.raises(ValueError, match=r"divide rank count.*ranks=1"):
        dspark.kernel_weight_shards(single_rank_cfg, empty_weights, ranks=1)

    total_elems = num_ranks * cols
    f32_shard = np.arange(total_elems, dtype=np.float32).reshape(num_ranks, cols)
    bf16_shard = np.asarray(jnp.asarray(f32_shard * scale_step, dtype=jnp.bfloat16))
    shards = {"proj": bf16_shard, "scale": f32_shard}

    dspark_dir = tmp_path / "dspark_shards"
    dspark.save_kernel_shards(shards, dspark_dir, ranks=num_ranks)
    dspark_index = json.loads((dspark_dir / "index.json").read_text(encoding="utf-8"))
    assert "kind" not in dspark_index
    assert dspark_index["ranks"] == num_ranks
    for rank in (0, num_ranks - 1):
        raw_bytes = (dspark_dir / f"rank{rank:02d}.bin").read_bytes()
        assert len(raw_bytes) == dspark_index["bytes_per_rank"]
        raw = np.frombuffer(raw_bytes, dtype=np.uint8)
        for name, expected in shards.items():
            spec = dspark_index["arrays"][name]
            dtype = (
                np.dtype(ml_dtypes.bfloat16)
                if spec["dtype"] == "bfloat16"
                else np.dtype(spec["dtype"])
            )
            nbytes = int(np.prod(spec["shape"])) * dtype.itemsize
            loaded_arr = (
                raw[spec["offset"] : spec["offset"] + nbytes]
                .view(dtype)
                .reshape(spec["shape"])
            )
            np.testing.assert_array_equal(loaded_arr, expected[rank])

    mla_dir = tmp_path / "mla_shards"
    dspark_mla.save_kernel_shards(shards, mla_dir, ranks=num_ranks)
    mla_index = json.loads((mla_dir / "index.json").read_text(encoding="utf-8"))
    assert mla_index["kind"] == dspark_mla.DRAFT_KIND
    assert mla_index["ranks"] == num_ranks

    for rank in (0, num_ranks - 1):
        loaded = dspark_mla.load_rank_shards(mla_dir, rank)
        assert set(loaded) == set(shards)
        for name, expected in shards.items():
            np.testing.assert_array_equal(loaded[name], expected[rank])


_SECOND_RENAME_CALL = 2


def _assert_dspark_uri_and_rank_validation(
    tmp_path: Path, shards: dict[str, np.ndarray]
) -> None:
    """Assert dspark loaders/writers reject remote URIs and invalid rank counts."""
    for uri in (
        "s3://bucket/dspark-draft",
        "s3:/bucket/dspark-draft",
        Path("s3://bucket/dspark-draft"),
    ):
        with pytest.raises(ValueError, match="local filesystem path"):
            dspark.DraftConfig.from_checkpoint(uri)
        with pytest.raises(ValueError, match="local filesystem path"):
            dspark.load_checkpoint_weights(uri)
        with pytest.raises(ValueError, match="local filesystem path"):
            dspark.write_rank_shard_dir(shards, uri, ranks=2)

    invalid_dir = tmp_path / "never_created"
    for bad_ranks in (0, -1):
        with pytest.raises(ValueError, match="ranks"):
            dspark.write_rank_shard_dir(shards, invalid_dir, ranks=bad_ranks)
        assert not invalid_dir.exists()

    for bad_leading in (1, 3):
        bad_shards = {"proj": np.zeros((bad_leading, 2), dtype=np.float32)}
        with pytest.raises(ValueError, match="ranks"):
            dspark.write_rank_shard_dir(bad_shards, invalid_dir, ranks=2)
        assert not invalid_dir.exists()


def _assert_dspark_atomic_replace_rollback(
    existing_root: Path,
    existing_dir: Path,
    shards_v2: dict[str, np.ndarray],
    orig_snapshot: tuple[bytes, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert failed staging-to-destination rename rolls back an existing shard dir."""
    orig_index, orig_rank0, orig_rank1 = orig_snapshot
    orig_rename = Path.rename
    rename_calls = 0

    def _fail_on_second_rename(self: Path, target: Path | str) -> Path:
        """Inject an OSError during the staging-to-destination rename."""
        nonlocal rename_calls
        rename_calls += 1
        if rename_calls == _SECOND_RENAME_CALL:
            msg = "simulated replace failure"
            raise OSError(msg)
        return orig_rename(self, target)

    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(Path, "rename", _fail_on_second_rename)
        with pytest.raises(OSError, match="simulated replace failure"):
            dspark.write_rank_shard_dir(shards_v2, existing_dir, ranks=2)
    assert (existing_dir / "index.json").read_bytes() == orig_index
    assert (existing_dir / "rank00.bin").read_bytes() == orig_rank0
    assert (existing_dir / "rank01.bin").read_bytes() == orig_rank1
    assert [entry.name for entry in existing_root.iterdir()] == ["shards"]


def test_dspark_write_rank_shard_dir_atomic_cleanup_and_uri_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify atomic shard-dir replacement/cleanup, rank checks, and URI rejection."""
    shards_v1 = {"proj": np.arange(4, dtype=np.float32).reshape(2, 2)}
    shards_v2 = {"proj": np.arange(10, 14, dtype=np.float32).reshape(2, 2)}
    _assert_dspark_uri_and_rank_validation(tmp_path, shards_v1)

    fresh_root = tmp_path / "fresh_case"
    fresh_root.mkdir()
    fresh_dir = fresh_root / "shards"
    orig_open = Path.open

    def _fail_on_rank1(
        self: Path,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> IO[bytes] | IO[str]:
        """Inject an OSError when opening rank01.bin for writing."""
        if self.name == "rank01.bin" and "w" in mode:
            msg = "simulated write failure"
            raise OSError(msg)
        return cast("IO[bytes] | IO[str]", orig_open(self, mode, *args, **kwargs))

    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(Path, "open", _fail_on_rank1)
        with pytest.raises(OSError, match="simulated write failure"):
            dspark.write_rank_shard_dir(shards_v1, fresh_dir, ranks=2)
    assert not fresh_dir.exists()
    assert list(fresh_root.iterdir()) == []

    existing_root = tmp_path / "existing_case"
    existing_root.mkdir()
    existing_dir = existing_root / "shards"
    dspark.write_rank_shard_dir(shards_v1, existing_dir, ranks=2)
    orig_snapshot = (
        (existing_dir / "index.json").read_bytes(),
        (existing_dir / "rank00.bin").read_bytes(),
        (existing_dir / "rank01.bin").read_bytes(),
    )

    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(Path, "open", _fail_on_rank1)
        with pytest.raises(OSError, match="simulated write failure"):
            dspark.write_rank_shard_dir(shards_v2, existing_dir, ranks=2)
    assert (existing_dir / "index.json").read_bytes() == orig_snapshot[0]
    assert (existing_dir / "rank00.bin").read_bytes() == orig_snapshot[1]
    assert (existing_dir / "rank01.bin").read_bytes() == orig_snapshot[2]
    assert [entry.name for entry in existing_root.iterdir()] == ["shards"]

    _assert_dspark_atomic_replace_rollback(
        existing_root, existing_dir, shards_v2, orig_snapshot, monkeypatch
    )

    dspark.write_rank_shard_dir(shards_v2, existing_dir, ranks=2)
    assert (existing_dir / "rank00.bin").read_bytes() == shards_v2["proj"][0].tobytes()
    assert (existing_dir / "rank01.bin").read_bytes() == shards_v2["proj"][1].tobytes()
    assert [entry.name for entry in existing_root.iterdir()] == ["shards"]


def test_dspark_mla_rejects_nonpositive_ranks_and_uris() -> None:
    """Verify dspark_mla rejects non-positive ranks and non-local URIs."""
    empty_weights: dict[str, np.ndarray] = {}
    for bad_ranks in (0, -1):
        with pytest.raises(ValueError, match="ranks"):
            dspark_mla.kernel_weight_shards(
                MlaDraftConfig(), empty_weights, ranks=bad_ranks
            )
        with pytest.raises(ValueError, match="ranks"):
            dspark_mla.kernel_weight_shards(
                MlaDraftConfig(heads=0), empty_weights, ranks=bad_ranks
            )

    for uri in (
        "s3://bucket/mla-draft",
        "s3:/bucket/mla-draft",
        Path("s3://bucket/mla-draft"),
    ):
        with pytest.raises(ValueError, match="local filesystem path"):
            MlaDraftConfig.from_checkpoint(uri)
        with pytest.raises(ValueError, match="local filesystem path"):
            dspark_mla.is_mla_draft(uri)
        with pytest.raises(ValueError, match="local filesystem path"):
            dspark_mla.load_checkpoint_weights(uri)
        with pytest.raises(ValueError, match="local filesystem path"):
            dspark_mla.load_rank_shards(uri, 0)
