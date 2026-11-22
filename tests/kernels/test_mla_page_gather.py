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
"""Contract tests for SparseCore VMEM-staged paged KV cache gather."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.mla import page_gather


def _require_tpu_with_sparsecore() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU SparseCore backend")
    if pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("SparseCore is not available on this TPU")


@pytest.mark.parametrize(
    ("num_pages", "rows", "row_width"),
    [
        (1, 256, 128),
        (7, 256, 128),
        (8, 256, 128),
        (12, 256, 128),
        (20, 256, 128),
        (28, 256, 128),
        (65, 256, 128),
        (257, 256, 128),
        (20, 256, 16),
    ],
)
def test_should_match_cache_indexing_when_gathering_pages_on_sparsecore(
    num_pages: int, rows: int, row_width: int
):
    _require_tpu_with_sparsecore()
    total_pages = max(96, num_pages + 17)
    rng = np.random.default_rng(100 + num_pages + row_width)
    cache_np = rng.integers(
        0, 0x7FFFFFFF, size=(total_pages, rows, row_width), dtype=np.uint32
    )
    page_ids_np = rng.choice(total_pages, size=num_pages, replace=False).astype(
        np.int32
    )

    got = np.asarray(
        jax.jit(lambda c, p: page_gather.gather_pages(c, p, row_width))(
            jnp.asarray(cache_np), jnp.asarray(page_ids_np)
        )
    )
    expected = cache_np[page_ids_np].reshape(num_pages * rows, row_width)
    np.testing.assert_array_equal(got, expected)


def test_should_reject_non_uint32_and_indivisible_row_width():
    cache_f32 = jnp.zeros((4, 16, 128), dtype=jnp.float32)
    pids = jnp.arange(2, dtype=jnp.int32)
    with pytest.raises(ValueError, match="expected uint32"):
        page_gather.gather_pages(cache_f32, pids, 128)

    cache_u32 = jnp.zeros((4, 15, 3), dtype=jnp.uint32)
    with pytest.raises(ValueError, match="expected uint32"):
        page_gather.gather_pages(cache_u32, pids, 16)


def test_should_reject_missing_sparsecore_and_excess_pages(monkeypatch):
    cache_u32 = jnp.zeros((4, 256, 128), dtype=jnp.uint32)
    pids = jnp.arange(2, dtype=jnp.int32)
    monkeypatch.setattr(
        pltpu, "get_tpu_info", lambda: SimpleNamespace(sparse_core=None)
    )
    with pytest.raises(NotImplementedError, match="SparseCore is not available"):
        page_gather.gather_pages(cache_u32, pids, 128)

    fake_sc = SimpleNamespace(num_cores=2, num_subcores=16, num_lanes=16)
    monkeypatch.setattr(
        pltpu, "get_tpu_info", lambda: SimpleNamespace(sparse_core=fake_sc)
    )
    too_many = jnp.zeros((513,), dtype=jnp.int32)
    with pytest.raises(ValueError, match="exceed one page per lane per subcore"):
        page_gather.gather_pages(cache_u32, too_many, 128)


@pytest.mark.parametrize(
    ("num_pages", "rows", "row_width", "expected_buffers"),
    [
        (8, 256, 128, 1),
        (20, 256, 128, 2),
        (65, 256, 128, 2),
        (20, 256, 16, 2),
        (4, 32, 128, 1),
    ],
)
def test_should_produce_valid_vmem_plan_and_per_subcore_table_across_regimes(
    monkeypatch, num_pages: int, rows: int, row_width: int, expected_buffers: int
):
    fake_sc = SimpleNamespace(num_cores=2, num_subcores=16, num_lanes=16)
    monkeypatch.setattr(
        pltpu, "get_tpu_info", lambda: SimpleNamespace(sparse_core=fake_sc)
    )
    monkeypatch.setattr(plsc, "VectorSubcoreMesh", lambda **kw: SimpleNamespace(**kw))
    captured = {}

    def fake_kernel(body, *, out_type, scratch_types, compiler_params, mesh, name):
        captured["out_shape"] = out_type.shape
        captured["scratch"] = scratch_types
        captured["name"] = name
        captured["kw"] = body.keywords

        def run(ids_table, flat_cache):
            captured["ids_table"] = np.asarray(ids_table)
            return jnp.zeros(out_type.shape, out_type.dtype)

        return run

    monkeypatch.setattr(jax.experimental.pallas, "kernel", fake_kernel)

    cache = jnp.zeros((num_pages + 5, rows, row_width), dtype=jnp.uint32)
    page_ids = jnp.asarray(np.arange(num_pages, dtype=np.int32) + 3)
    out = page_gather.gather_pages(cache, page_ids, row_width)

    assert out.shape == (num_pages * rows, row_width)
    kw = captured["kw"]
    assert kw["num_buffers"] == expected_buffers
    assert rows % kw["stripe_rows"] == 0
    assert kw["num_buffers"] * kw["stripe_rows"] * row_width * 4 <= 128 * 1024
    tbl = captured["ids_table"].reshape(kw["total_subcores"], kw["slots"])
    for stripe_idx in range(kw["total_stripes"]):
        sc_id = stripe_idx % kw["total_subcores"]
        slot_k = stripe_idx // kw["total_subcores"]
        expected_page = int(page_ids[stripe_idx // kw["stripes_per_page"]])
        assert int(tbl[sc_id, slot_k]) == expected_page
