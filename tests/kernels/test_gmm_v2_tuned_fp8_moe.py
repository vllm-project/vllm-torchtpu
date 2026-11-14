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
"""The grouped matmul's tuned FP8 MoE tiling and rhs-scale hoist are off
unless a model rule sets gmm_v2.TUNED_FP8_MOE: other models keep main's tiles
and main's order of scaling and accumulating.

The expected tiles are what calculate_tiling returns on TPU7x: main's for the
flag off, the tuned schedule for the flag on."""

import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gmm_v2

FP8 = jnp.float8_e4m3fn
VMEM = int(64 * 1024 * 1024 * 0.9)


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(gmm_v2.pltpu, "get_tpu_info", lambda: info)


def _tiles(rows, k, n, groups, fuse_act):
    """An FP8 MoE projection with per-channel weight scales and an fp8 lhs."""
    lhs = gmm_v2.InputConfigs(quant_dtype=FP8, quant_block_size=512, dtype=jnp.bfloat16)
    rhs = gmm_v2.InputConfigs(
        quant_dtype=FP8, quant_block_size=k, dtype=FP8, has_scale=True
    )
    dims = gmm_v2.Dimensions(
        size_m=rows,
        size_k=k,
        size_n=n,
        size_group=groups,
        size_lhs_group=groups,
        size_lhs_sublane=16,
    )
    t = gmm_v2.calculate_tiling(dims, lhs, rhs, VMEM, fuse_act)
    cfgs = gmm_v2.GmmConfigs(
        tiles=t,
        dims=dims,
        lhs_cfgs=lhs,
        rhs_cfgs=rhs,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.bfloat16,
        zero_init=False,
        fuse_act=fuse_act,
    )
    return (t.tile_m, t.tile_k, t.tile_n, t.bucket_base), gmm_v2.hoist_rhs_scale(cfgs)


# rows, (k, n), groups, fuse_act, main's tiles, tuned tiles. Qwen3-30B-A3B's
# gate/up projection at TP8 (128 experts, 2 x 128 padded columns) for 16 and
# 64 tokens; Qwen3.5's gate/up at 80 expert slots for 8 tokens (a table entry).
CASES = [
    (128, (2048, 256), 128, "silu", (128, 2048, 128, 128), (128, 2048, 128, 64)),
    (512, (2048, 256), 128, "silu", (512, 2048, 128, 128), (128, 2048, 128, 64)),
    (80, (4096, 2048), 80, "silu", (80, 4096, 1024, 80), (64, 4096, 1024, 16)),
]


def test_the_default_is_mains_tiling_and_order():
    assert gmm_v2.TUNED_FP8_MOE is False


@pytest.mark.parametrize("rows,kn,groups,fuse,main,_tuned", CASES)
def test_off_keeps_mains_tiles_and_scales_each_block(
    monkeypatch, rows, kn, groups, fuse, main, _tuned
):
    monkeypatch.setattr(gmm_v2, "TUNED_FP8_MOE", False)
    tiles, hoist = _tiles(rows, *kn, groups, fuse)
    assert tiles == main
    assert hoist is False


@pytest.mark.parametrize("rows,kn,groups,fuse,_main,tuned", CASES)
def test_on_takes_the_tuned_tiles_and_hoists_the_scale(
    monkeypatch, rows, kn, groups, fuse, _main, tuned
):
    monkeypatch.setattr(gmm_v2, "TUNED_FP8_MOE", True)
    tiles, hoist = _tiles(rows, *kn, groups, fuse)
    assert tiles == tuned
    # One per-channel rhs block spans the k tile, which holds several 512-wide
    # lhs blocks: the case where the hoist changes the bf16 rounding.
    assert hoist is True


def test_set_tuned_fp8_moe_switches_it(monkeypatch):
    monkeypatch.setattr(gmm_v2, "TUNED_FP8_MOE", False)
    gmm_v2.set_tuned_fp8_moe(True)
    assert gmm_v2.TUNED_FP8_MOE is True
    gmm_v2.set_tuned_fp8_moe(False)
    assert gmm_v2.TUNED_FP8_MOE is False
