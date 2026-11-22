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

from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, host, kernel
from vllm_torchtpu.kernels.fused_moe.v2.layer import fused_ep_moe_v2


def test_should_enable_height_rungs_for_prefill_when_ep_tp_size_exceeds_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", raising=False)

    rungs_ep1 = kernel.tile_height_rungs(128, 131584, 21, 16, ep_tp_size=1)
    rungs_ep2 = kernel.tile_height_rungs(128, 131584, 21, 16, ep_tp_size=2)

    assert rungs_ep1 == (128,)
    assert rungs_ep2 == (32, 64, 96, 128)

    monkeypatch.setenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", "1")
    assert kernel.tile_height_rungs(128, 131584, 21, 16, ep_tp_size=2) == (128,)


def test_should_form_latin_square_with_cross_host_half_first_when_ep_is_pow2() -> None:
    ep = 16
    matrix = np.array(
        [
            [int(kernel.rotated_push_peer(jnp.int32(me), ep, s)) for s in range(ep)]
            for me in range(ep)
        ]
    )

    for me in range(ep):
        assert sorted(matrix[me].tolist()) == list(range(ep))
        if me < ep // 2:
            assert set(matrix[me, : ep // 2].tolist()) == set(range(ep // 2, ep))
        else:
            assert set(matrix[me, : ep // 2].tolist()) == set(range(0, ep // 2))
    for s in range(ep):
        assert sorted(matrix[:, s].tolist()) == list(range(ep))


@pytest.mark.parametrize("ep", [4, 6])
def test_should_preserve_sequential_peer_order_when_ep_below_eight_or_non_pow2(
    ep: int,
) -> None:
    peers = [int(kernel.rotated_push_peer(jnp.int32(1), ep, s)) for s in range(ep)]
    assert peers == list(range(ep))


@pytest.mark.parametrize(
    ("total_rows", "runs"),
    [
        (0, [(0, 0, 0), (0, 0, 0)]),
        (384, [(0, 80, 153), (80, 112, 219)]),
        (2048, [(0, 320, 631), (320, 384, 0), (704, 320, 640)]),
        (2432, [(0, 200, 397), (200, 600, 1189), (800, 416, 821)]),
    ],
)
def test_should_partition_run_true_rows_without_loss_or_overlap_across_chunks(
    total_rows: int, runs: list[tuple[int, int, int]]
) -> None:
    tile_m = 128
    tile_blocks = tile_m // host.ROWBLK
    chunk_tiles = kernel.CHUNK_PUSH_TILES
    n_tiles = -(-total_rows // tile_m)
    n_interior = max(-(-n_tiles // chunk_tiles) - 1, 0)
    tail_lo = int(
        kernel.tail_chunk_start_blocks(
            jnp.int32(total_rows), tile_m, chunk_tiles, tile_blocks
        )
    )
    assert tail_lo == n_interior * chunk_tiles * tile_blocks

    slices = [
        (c * chunk_tiles * tile_blocks, (c + 1) * chunk_tiles * tile_blocks)
        for c in range(n_interior)
    ] + [(tail_lo, total_rows // host.ROWBLK)]

    for run_start, run_len, true_rows in runs:
        covered_intervals: list[tuple[int, int]] = []
        for lo_b, hi_b in slices:
            into_r, here_r = kernel.slice_run_true_rows(
                jnp.int32(run_start),
                jnp.int32(run_len),
                jnp.int32(true_rows),
                jnp.int32(lo_b),
                jnp.int32(hi_b),
            )
            into_i, here_i = int(into_r), int(here_r)
            assert here_i >= 0
            if here_i > 0:
                covered_intervals.append((into_i, into_i + here_i))

        if true_rows == 0:
            assert covered_intervals == []
        else:
            assert covered_intervals[0][0] == 0
            for prev_iv, next_iv in zip(covered_intervals, covered_intervals[1:]):
                assert prev_iv[1] == next_iv[0]
            assert covered_intervals[-1][1] == true_rows


@pytest.mark.parametrize(
    ("tokens", "inter", "z_delay"),
    [
        (16384, 2048, 2),
        (512, 1024, 0),
    ],
)
def test_should_export_ep_tp_z_clone_schedules_to_tpu7x_mlir(
    tokens: int, inter: int, z_delay: int
) -> None:
    fake_info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    mesh = AbstractMesh((16,), ("ep",))
    sh = NamedSharding(mesh, P("ep"))
    inter_slice = inter // 2

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
            ep_tp_z_clone_delay=z_delay,
        )

    def sds(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sh)

    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=fake_info),
        mock.patch.object(tpu_info, "get_tpu_info", return_value=fake_info),
    ):
        exported = jax.export.export(jax.jit(trace_fn), platforms=("tpu",))(
            sds((tokens, 6144), jnp.bfloat16),
            sds((512, 6144, 2 * inter_slice), jnp.float8_e4m3fn),
            sds((512, inter_slice, 6144), jnp.float8_e4m3fn),
            sds((512, 2 * inter_slice), jnp.float32),
            sds((512, 6144), jnp.float32),
            sds((tokens, 256), jnp.float32),
            sds((16, 1), jnp.int32),
        )
    mlir = exported.mlir_module()
    assert "_zclone" in mlir
    assert "partition_id" not in mlir
