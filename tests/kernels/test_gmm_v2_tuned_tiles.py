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
"""TPU_GMM_V2_TUNED_TILES tile choice at the MoE shapes (CPU, tpu7x info)."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.megablox import gmm_v2 as G

H, I, E = 6144, 3072, 16


@pytest.fixture
def tpu7x(monkeypatch):
    def sublane(dtype):
        return {2: 16, 4: 8}.get(jnp.dtype(dtype).itemsize, 32)

    info = SimpleNamespace(
        mxu_column_size=256,
        num_lanes=128,
        vmem_capacity_bytes=64 * 2**20,
        is_matmul_supported=lambda a, b: True,
        fp8_ops_per_second=1,
        int8_ops_per_second=0,
        get_sublane_tiling=sublane,
    )
    monkeypatch.setattr(G.pltpu, "get_tpu_info", lambda: info)
    logged = []
    monkeypatch.setattr(
        G, "logger", SimpleNamespace(info_once=lambda *a, **k: logged.append(a))
    )
    path = G.skinny_gmm_path()
    G.set_skinny_gmm_path("fp8_feed_wide")
    yield
    G.set_skinny_gmm_path(path)


def _tiles(size_m, tuned, monkeypatch):
    monkeypatch.setattr(envs, "TPU_GMM_V2_TUNED_TILES", tuned, raising=False)
    sds = jax.ShapeDtypeStruct
    lhs = sds((size_m, H), jnp.bfloat16)
    w13 = sds((E, H, 2 * I), jnp.float4_e2m1fn)
    w2 = sds((E, I, H), jnp.float4_e2m1fn)
    s13 = sds((E, H // 128, 1, 2 * I), jnp.float32)
    s2 = sds((E, I // 128, 1, H), jnp.float32)
    gs, go = sds((E,), jnp.int32), sds((1,), jnp.int32)
    vmem = int(64 * 2**20 * 0.9)
    common = dict(
        tile_info=G.calculate_tiling,
        vmem_limit_bytes=vmem,
        acc_dtype=None,
        maybe_quantize_lhs=False,
        zero_initialize=False,
    )
    c1 = G.make_gmm_configs(
        lhs, w13, s13, None, gs, go, out_dtype=None, fuse_act="swigluoai", **common
    )
    h = sds((size_m, c1.out_size_n), c1.out_dtype)
    c2 = G.make_gmm_configs(h, w2, s2, None, gs, go, out_dtype=None, **common)
    return c1.tiles, c2.tiles


@pytest.mark.parametrize("size_m", [32, 128, 256, 2048, 4096, 8192, 32768])
def test_flag_off_keeps_default_tiles(tpu7x, monkeypatch, size_m):
    t1, t2 = _tiles(size_m, False, monkeypatch)
    assert t1.rhs_buffer_count == 3 and t2.rhs_buffer_count == 3
    assert t1.tile_n == 1536 and t2.tile_n == H


@pytest.mark.parametrize("size_m", [32, 64, 128, 192])
def test_small_decode_gate_up_two_deep(tpu7x, monkeypatch, size_m):
    base1, base2 = _tiles(size_m, False, monkeypatch)
    t1, t2 = _tiles(size_m, True, monkeypatch)
    assert t1 == G.dataclasses.replace(base1, rhs_buffer_count=2)
    assert t2 == base2


@pytest.mark.parametrize("size_m", [256, 1024, 2048])
def test_decode_gate_up_full_width(tpu7x, monkeypatch, size_m):
    base1, base2 = _tiles(size_m, False, monkeypatch)
    t1, t2 = _tiles(size_m, True, monkeypatch)
    assert (t1.tile_m, t1.bucket_base, t1.tile_k) == (32, 16, H)
    assert t1.tile_n == I and t1.rhs_buffer_count == 2
    assert t2 == base2


@pytest.mark.parametrize("size_m", [4096, 8192, 32768])
def test_prefill_buckets_128(tpu7x, monkeypatch, size_m):
    base1, base2 = _tiles(size_m, False, monkeypatch)
    assert (base1.tile_m, base1.bucket_base) == (256, 64)
    t1, t2 = _tiles(size_m, True, monkeypatch)
    for base, t in ((base1, t1), (base2, t2)):
        assert t == G.dataclasses.replace(base, bucket_base=128)
        assert t.bucket_sizes == (128, 256)


@pytest.mark.parametrize("path", ["bf16_after", "fp8_feed"])
def test_other_skinny_paths_untouched(tpu7x, monkeypatch, path):
    G.set_skinny_gmm_path(path)
    for size_m in (128, 1024, 8192):
        assert _tiles(size_m, True, monkeypatch) == _tiles(size_m, False, monkeypatch)
