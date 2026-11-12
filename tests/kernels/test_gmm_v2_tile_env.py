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
"""CPU tests of the opt-in gmm_v2 tile overrides (additional_config keys
gmm_v2_prefill_tiles / gmm_v2_decode_w13, `gmm_v2.set_tile_overrides`)."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.megablox import gmm_v2 as G

# MiniMax-M3.1-Flash per-core MoE shapes (EP8, 16 local experts, NVFP4 block 128).
H, FE, LOCAL = 6144, 3072, 16
VMEM = int(64 * 1024 * 1024 * 0.9)


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    monkeypatch.setattr(
        G.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            num_lanes=128,
            mxu_column_size=256,
            vmem_capacity_bytes=64 * 1024 * 1024,
            is_matmul_supported=lambda a, b: True,
            get_sublane_tiling=lambda dtype: 16,
        ),
    )
    G.set_tile_overrides()
    yield
    G.set_tile_overrides()


def _cfgs(size_k):
    lhs = G.InputConfigs(quant_dtype=None, quant_block_size=size_k, dtype=jnp.bfloat16)
    rhs = G.InputConfigs(
        quant_dtype=jnp.float4_e2m1fn,
        quant_block_size=128,
        dtype=jnp.float4_e2m1fn,
        has_scale=True,
    )
    return lhs, rhs


def _tiles(size_m, gate_up):
    k, n = (H, 2 * FE) if gate_up else (FE, H)
    dims = G.Dimensions(
        size_m=size_m, size_k=k, size_n=n, size_group=LOCAL,
        size_lhs_group=LOCAL, size_lhs_sublane=16,
    )
    lhs, rhs = _cfgs(k)
    return G.calculate_tiling(dims, lhs, rhs, VMEM, "swigluoai" if gate_up else None)


@pytest.mark.parametrize("size_m", [128, 2048, 4096, 32768])
@pytest.mark.parametrize("gate_up", [True, False])
def test_unset_keys_keep_default_tiles(size_m, gate_up):
    assert _tiles(size_m, gate_up) == G._calculate_tiling(
        *_args(size_m, gate_up)
    )


def _args(size_m, gate_up):
    k, n = (H, 2 * FE) if gate_up else (FE, H)
    dims = G.Dimensions(
        size_m=size_m, size_k=k, size_n=n, size_group=LOCAL,
        size_lhs_group=LOCAL, size_lhs_sublane=16,
    )
    lhs, rhs = _cfgs(k)
    return dims, lhs, rhs, VMEM, "swigluoai" if gate_up else None


@pytest.mark.parametrize("gate_up", [True, False])
def test_prefill_tiles_apply_above_decode_max(gate_up):
    G.set_tile_overrides(prefill_tiles="256:128/256")
    for size_m in (4096, 32768):
        t = _tiles(size_m, gate_up)
        assert (t.tile_m, t.bucket_base, t.bucket_sizes) == (256, 128, (128, 256))
        default = G._calculate_tiling(*_args(size_m, gate_up))
        assert (t.tile_k, t.tile_n, t.rhs_buffer_count) == (
            default.tile_k, default.tile_n, default.rhs_buffer_count)
    # decode-sized buffers keep their tiles
    assert _tiles(2048, gate_up) == G._calculate_tiling(*_args(2048, gate_up))


def test_prefill_tiles_max_size_m():
    G.set_tile_overrides(prefill_tiles="256:128/256@12288")
    assert _tiles(12288, True).bucket_sizes == (128, 256)
    assert _tiles(16384, True) == G._calculate_tiling(*_args(16384, True))


def test_prefill_tiles_bad_spec():
    with pytest.raises(ValueError, match=G.PREFILL_TILES_KEY):
        G.set_tile_overrides(prefill_tiles="256:128")


def test_decode_w13_small_and_large_decode():
    G.set_tile_overrides(decode_w13="3072:2:2")
    for size_m in (32, 512):
        t = _tiles(size_m, True)
        d = G._calculate_tiling(*_args(size_m, True))
        assert (t.tile_n, t.rhs_buffer_count) == (3072, 2)
        assert (t.tile_m, t.bucket_sizes) == (d.tile_m, d.bucket_sizes)
        t2 = _tiles(size_m, False)
        assert t2 == G.dataclasses.replace(
            G._calculate_tiling(*_args(size_m, False)), rhs_buffer_count=2)
    for size_m in (1536, 2048):
        t = _tiles(size_m, True)
        assert t == G.dataclasses.replace(
            G._calculate_tiling(*_args(size_m, True)), rhs_buffer_count=2)
        assert _tiles(size_m, False) == G._calculate_tiling(*_args(size_m, False))
    assert _tiles(4096, True) == G._calculate_tiling(*_args(4096, True))


def test_decode_w13_default_down_depth():
    G.set_tile_overrides(decode_w13="3072:2")
    assert _tiles(512, False) == G._calculate_tiling(*_args(512, False))
    assert _tiles(512, True).tile_n == 3072


def test_decode_w13_over_vmem_falls_back():
    # three 18.9 MB gate_up blocks do not fit
    G.set_tile_overrides(decode_w13="3072:3")
    assert _tiles(512, True) == G._calculate_tiling(*_args(512, True))


def test_vmem_estimate_counts_rhs_buffers():
    lhs, rhs = _cfgs(H)
    e3 = G.gmm_vmem_estimate(32, 3072, H, lhs_cfgs=lhs, rhs_cfgs=rhs, fuse_act="swigluoai")
    e2 = G.gmm_vmem_estimate(
        32, 3072, H, lhs_cfgs=lhs, rhs_cfgs=rhs, fuse_act="swigluoai", rhs_buffer_count=2
    )
    assert e3 - e2 == 2 * G.rhs_weight_block_bytes(H, 3072, rhs)


def test_comma_buckets_and_reset():
    G.set_tile_overrides(prefill_tiles="256:128,256")
    assert G.tile_overrides_active()
    assert _tiles(4096, True).bucket_sizes == (128, 256)
    G.set_tile_overrides()
    assert not G.tile_overrides_active()
    assert _tiles(4096, True) == G._calculate_tiling(*_args(4096, True))
