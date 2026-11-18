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
"""Sequence-row o_proj of the token-parallel split vs o_proj + reduce-scatter.

`dcp_token_parallel.project_sequence_rows` against the head-sharded o_proj and
reduce-scatter it replaces, on a host-device mesh
(XLA_FLAGS=--xla_force_host_platform_device_count=16).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp

AXIS = "dcp"
PAD_SENTINEL = 1.0e4


def _mesh(dcp: int) -> Mesh:
    if len(jax.devices()) < dcp:
        pytest.skip(f"needs {dcp} devices")
    return Mesh(np.array(jax.devices()[:dcp]), (AXIS,))


def _layout(dcp: int, split_rows: int) -> tp.TokenParallelLayout:
    return tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=1,
        page_size=dcp,
        split_rows=split_rows,
        max_pages=1,
    )


def _sharded(mesh: Mesh, x: np.ndarray) -> jax.Array:
    return jax.device_put(jnp.asarray(x), NamedSharding(mesh, P(AXIS)))


def _own_rows_global(step_rows: np.ndarray, dcp: int, split_rows: int) -> np.ndarray:
    """Rank j's token-parallel rows `split_rows + j * R + i`, stacked.

    Rows past T carry PAD_SENTINEL."""
    num_tokens = step_rows.shape[0]
    rows = num_tokens // dcp
    padded = np.concatenate(
        [
            step_rows,
            np.full((split_rows, *step_rows.shape[1:]), PAD_SENTINEL, np.float32),
        ]
    )
    return np.concatenate(
        [
            padded[split_rows + j * rows : split_rows + (j + 1) * rows]
            for j in range(dcp)
        ]
    )


@pytest.mark.parametrize(("dcp", "split_rows"), [(4, 2), (16, 16)])
def test_should_land_on_contiguous_sequence_rows_when_token_rows_are_shifted(
    dcp, split_rows
):
    mesh = _mesh(dcp)
    layout = _layout(dcp, split_rows)
    num_tokens, width = dcp * 2 * split_rows, 8
    step_rows = np.arange(num_tokens * width, dtype=np.float32).reshape(
        num_tokens, width
    )
    own = _sharded(mesh, _own_rows_global(step_rows, dcp, split_rows))
    lead = jnp.asarray(step_rows[:split_rows])
    rank_ids = _sharded(mesh, np.arange(dcp, dtype=np.int32))

    out = jax.shard_map(
        lambda o, lr, r: tp.token_rows_to_sequence_rows(layout, o, lr, r[0], AXIS),
        mesh=mesh,
        in_specs=(P(AXIS), P(), P(AXIS)),
        out_specs=P(AXIS),
        check_vma=False,
    )(own, lead, rank_ids)

    np.testing.assert_array_equal(np.asarray(out), step_rows)


def _random_fp8_row_shards(
    rng, groups: int, k: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row-parallel FP8 o_proj: `[groups * k, n]` and `[groups, n]` scales.

    Each `k`-row shard is quantized per output channel on its own."""
    w = rng.standard_normal((groups, k, n)).astype(np.float32)
    w[1] *= 7.0  # shards with very different ranges get different scales
    scale = np.abs(w).max(axis=1) / 448.0
    w_q = np.asarray(jnp.asarray(w / scale[:, None, :]).astype(jnp.float8_e4m3fn))
    return w_q.reshape(groups * k, n), scale.astype(np.float32)


@pytest.mark.parametrize("dcp", [4, 16])
def test_should_match_head_sharded_oproj_reduce_scatter_when_projecting_split_step(dcp):
    rng = np.random.default_rng(dcp)
    mesh = _mesh(dcp)
    split_rows, heads_per_rank, latent, v_dim, hidden = 8, 2, 16, 8, 24
    layout = _layout(dcp, split_rows)
    heads = dcp * heads_per_rank
    num_tokens = dcp * 2 * split_rows

    lead_partial = rng.standard_normal((dcp, split_rows, heads, latent)).astype(
        np.float32
    )
    lead_lse = rng.uniform(-3.0, 3.0, (dcp, split_rows, heads)).astype(np.float32)
    lead_lse[0, 0, :] = -np.inf  # a rank that owns none of a row's keys
    w_uv = rng.standard_normal((heads, latent, v_dim)).astype(np.float32)
    tail_v = rng.standard_normal((num_tokens - split_rows, heads, v_dim)).astype(
        np.float32
    )
    w_q, w_s = _random_fp8_row_shards(rng, dcp, heads_per_rank * v_dim, hidden)

    # Reference: the exact LSE merge, W_UV, then o_proj over every row at once
    # (head-sharded partial sums + reduce-scatter compute the same rows).
    m = np.logaddexp.reduce(lead_lse, axis=0)
    weights = np.exp(lead_lse - m[None])
    lead_latent = (lead_partial * weights[..., None]).sum(axis=0)
    lead_v = np.einsum("rhl,hlv->rhv", lead_latent, w_uv)
    step_v = np.concatenate([lead_v, tail_v]).reshape(num_tokens, heads * v_dim)
    w_deq = (
        w_q.astype(np.float32).reshape(dcp, heads_per_rank * v_dim, hidden)
        * w_s[:, None, :]
    ).reshape(heads * v_dim, hidden)
    expected = step_v @ w_deq

    own_out = _own_rows_global(
        np.concatenate([lead_v, tail_v]), dcp, split_rows
    )  # [dcp * R, H, V]
    out = tp.project_sequence_rows(
        layout,
        mesh,
        AXIS,
        _sharded(mesh, lead_partial.reshape(dcp * split_rows, heads, latent)),
        _sharded(mesh, lead_lse.reshape(dcp * split_rows, heads)),
        _sharded(mesh, own_out),
        _sharded(mesh, np.arange(dcp, dtype=np.int32)),
        _sharded(mesh, w_uv),
        _sharded(mesh, w_q),
        _sharded(mesh, w_s.reshape(-1)),
    )

    assert out.shape == (num_tokens, hidden)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=2e-4, atol=2e-3)


def test_should_reject_lead_rows_not_matching_split_rows():
    layout = _layout(4, 2)
    with pytest.raises(ValueError, match="split_rows"):
        tp.token_rows_to_sequence_rows(
            layout, jnp.zeros((4, 8)), jnp.zeros((3, 8)), jnp.int32(0), AXIS
        )
