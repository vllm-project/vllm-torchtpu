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
# Under tests/kernels, not tests/distributed: in-process JAX holds the TPU, and
# tests/conftest.py must run it after the engine-spawning tests/entrypoints.

import numpy as np
import pytest

from vllm_torchtpu.distributed import hier_reduce_scatter as hrs

_TPU7X_RANK_AT_MESH_INDEX = (0, 1, 6, 7, 2, 3, 4, 5)


def _snr_db(ref: np.ndarray, got: np.ndarray) -> float:
    noise = np.sum((got - ref) ** 2)
    return float("inf") if noise == 0 else 10 * np.log10(np.sum(ref**2) / noise)


@pytest.mark.tpu_test
@pytest.mark.multichip
@pytest.mark.parametrize("order", [tuple(range(8)), _TPU7X_RANK_AT_MESH_INDEX])
@pytest.mark.parametrize(
    "stream, stream_min_bytes",
    [
        pytest.param(False, 0, id="micro_batched"),
        pytest.param(True, 0, id="row_streaming"),
    ],
)
@pytest.mark.parametrize("rows_per_rank", [1024, 2048, 4096, 8192, 16384])
def test_matches_rank_ordered_reduce_scatter(
    order, stream, stream_min_bytes, rows_per_rank
):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    if jax.device_count() < 8:
        pytest.skip("needs 8 TPU devices")
    devices = sorted(jax.devices(), key=lambda d: d.id)[:8]
    mesh = Mesh(np.asarray(devices), ("d",))
    fn = hrs.make_reduce_scatter_fn(
        mesh,
        order,
        axis_name="d",
        stream=stream,
        stream_min_bytes=stream_min_bytes,
    )

    hidden, n = 4096, 8
    block = rows_per_rank // n
    # Fresh per case: a stale-buffer bug looks correct on a repeated input.
    rng = np.random.default_rng(rows_per_rank + 31 * len(set(order)))
    per_rank = [
        rng.standard_normal((rows_per_rank, hidden)).astype(np.float32)
        for _ in range(n)
    ]
    per_rank = [jnp.asarray(r, jnp.bfloat16) for r in per_rank]
    global_x = jnp.concatenate([per_rank[r] for r in order], axis=0)
    global_x = jax.device_put(global_x, NamedSharding(mesh, P("d", None)))

    out = np.asarray(jax.jit(fn)(global_x), np.float32)

    summed = sum(np.asarray(r, np.float32) for r in per_rank)
    expected = np.concatenate(
        [summed[r * block : (r + 1) * block] for r in order], axis=0
    )
    snr = _snr_db(expected, out)
    assert snr > 40.0, f"SNR {snr:.1f} dB <= 40 dB"
    for i, r in enumerate(order):
        got = out[i * block : (i + 1) * block]
        errs = [
            np.mean((got - summed[k * block : (k + 1) * block]) ** 2) for k in range(n)
        ]
        assert int(np.argmin(errs)) == r, (
            f"mesh index {i} got rank {np.argmin(errs)}'s block, want {r}"
        )
