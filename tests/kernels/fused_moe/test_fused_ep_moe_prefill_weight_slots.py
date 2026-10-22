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
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.fused_moe.v2_prefill import host, kernel


class _FakeChip:
    """The fields the VMEM accounting reads off a device record.

    The sublane tiling is the 32-byte rule the real chips report -- f32 tiles
    to 8 sublanes, bf16 to 16, eight-bit to 32 -- so the byte figures here are
    the shape of the real ones, not just self-consistent.
    """

    num_lanes = 128
    generation = host.MIN_GENERATION

    def __init__(self, vmem_capacity_bytes=128 * 2**20):
        self.vmem_capacity_bytes = vmem_capacity_bytes

    def get_sublane_tiling(self, dtype):
        return 32 // jnp.dtype(dtype).itemsize


# --------------------------------------------------------------------------- #
# Tile height 256.
#
# The four-bit FFN widens a k-block one FP4_OUTPUT_CHUNK of output columns at a
# time, so that chunk, not the full matmul width, is what the widened-block
# term has to hold. Sized at the full width the term alone was 5 MiB at
# DeepSeek-V4.1's shape and pushed tile 256's two-slot build over the budget,
# which leaves it on one slot -- a schedule with no weight prefetch at all.
# --------------------------------------------------------------------------- #

DSV41 = dict(g_local=48, hidden=5120, inter=2560)


def _widened(capacity):
    body = host.vmem_tile_body_arrays(
        capacity, DSV41["hidden"], DSV41["inter"], weight_format=host.WeightFormat.FP4
    )
    (shape,) = [s for n, s, _ in body if n == "widened_weight_block"]
    return shape


def test_fp4_widened_block_is_one_output_chunk_wide():
    assert _widened(128) == (
        host.WIDENED_BLOCK_BUFFERS,
        host.QB4,
        host.FP4_OUTPUT_CHUNK,
    )
    assert kernel.FP4_OUTPUT_CHUNK == host.FP4_OUTPUT_CHUNK


@pytest.mark.parametrize("capacity", [128, 256])
def test_dsv41_fp4_fits_two_weight_slots(capacity):
    """DeepSeek-V4.1 FP4 gets the one-expert-ahead prefetch at both heights.

    The 64 MiB chip is the v7x's. Measured post-RA VMEM of these builds is
    45.3 MiB (128) and 52.2 MiB (256), under the 62.7 MiB budget; the estimate
    stays an upper bound on both.
    """
    slots = host.fit_weight_slots(
        **DSV41,
        capacity=capacity,
        weight_format=host.WeightFormat.FP4,
        info=_FakeChip(64 * 2**20),
    )
    assert (slots.distance, slots.nbuf) == (1, 2)
    assert slots.vmem_bytes > {128: 45.3, 256: 52.2}[capacity] * 2**20


@pytest.mark.parametrize("tile_m,want", [(32, 1), (128, 1), (256, 2), (512, 4)])
def test_mirror_rows_per_tile(tile_m, want):
    """A tile's scales take whole 128-lane mirror rows above 128 rows."""
    assert host.mirror_rows_per_tile(tile_m) == want


def test_mirror_rows_per_tile_refuses_a_ragged_split():
    with pytest.raises(ValueError, match="whole 128-lane"):
        host.mirror_rows_per_tile(192)


@pytest.mark.parametrize("capacity", [128, 256])
def test_scale_staging_is_128_lanes_wide(capacity):
    arrays = host.vmem_scratch_arrays(
        DSV41["g_local"],
        capacity,
        DSV41["hidden"],
        DSV41["inter"],
        nbuf=2,
        weight_format=host.WeightFormat.FP4,
    )
    (shape,) = [s for n, s, _ in arrays if n == "oscl_vm"]
    rows = host.mirror_rows_per_tile(capacity)
    assert shape == (host.OUT_PARITIES, rows, capacity // rows)
    assert shape[-1] <= host.MIRROR_LANES
