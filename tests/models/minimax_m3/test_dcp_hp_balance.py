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
"""Head-parallel MSA DCP prefill balance and glue (interpret mode, CPU):
TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT (the replicas of a KV head split its rows by
a tile cost model), TPU_MINIMAX_M3_DCP_HP_ROUND_CAPACITY (blocks per page
exchange round) and TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR (the group's
K/V/index queries handed to the scoring op head-major, bit for bit)."""

import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3 import test_msa_index_reuse as reuse_t
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_dcp_heads


# ------------------------------------------------------------------ envs
def test_env_defaults(monkeypatch):
    for env in (msa_dcp_heads.ROW_SPLIT_ENV, msa_dcp_heads.ROUND_CAPACITY_ENV):
        monkeypatch.delenv(env, raising=False)
    assert msa_dcp_heads.hp_row_split() is False
    assert msa_dcp_heads.hp_round_capacity() == 1


def test_env_parse(monkeypatch):
    monkeypatch.setenv(msa_dcp_heads.ROW_SPLIT_ENV, "1")
    assert msa_dcp_heads.hp_row_split() is True
    monkeypatch.setenv(msa_dcp_heads.ROW_SPLIT_ENV, "0")
    assert msa_dcp_heads.hp_row_split() is False
    monkeypatch.setenv(msa_dcp_heads.ROW_SPLIT_ENV, "2")
    with pytest.raises(ValueError):
        msa_dcp_heads.hp_row_split()
    monkeypatch.setenv(msa_dcp_heads.ROUND_CAPACITY_ENV, "4")
    assert msa_dcp_heads.hp_round_capacity() == 4
    monkeypatch.setenv(msa_dcp_heads.ROUND_CAPACITY_ENV, "0")
    with pytest.raises(ValueError):
        msa_dcp_heads.hp_round_capacity()


# ------------------------------------------------------------- cost model
def test_split_window():
    assert msa_dcp_heads.split_window(8192, 16) == 8192 - 3072
    assert msa_dcp_heads.split_window(8192, 8) == 8192 - 3072
    assert msa_dcp_heads.split_window(96, 8) == 64
    assert msa_dcp_heads.split_window(100, 8) == 0  # not tile-aligned
    assert msa_dcp_heads.split_window(16, 8) == 0  # too short
    for t, b in ((8192, 16), (8192, 8), (96, 8), (2048, 32)):
        w = msa_dcp_heads.split_window(t, b)
        assert (t - w) % b == 0 and w >= t // 2


def test_tile_costs_count_distinct_pages():
    # Tile 0: both rows select pages {1, 2, 3} (+ local 1): 3 distinct.
    # Tile 1: rows select {4, 5, 6} and {7, 8, -1}, local 9, 10: 7 distinct.
    keys = jnp.asarray([[1, 2, 3], [3, 2, 1], [4, 5, 6], [7, 8, -1]], jnp.int32)
    local = jnp.asarray([1, -1, 9, 10], jnp.int32)
    got = np.asarray(msa_dcp_heads.tile_costs(keys, local, 2, 2))
    empty = msa_dcp_heads._EMPTY_TILE_COST
    np.testing.assert_allclose(got, [empty + 2, empty + 4])
    none = np.asarray(
        msa_dcp_heads.tile_costs(
            -jnp.ones((4, 3), jnp.int32), -jnp.ones(4, jnp.int32), 2, 2
        )
    )
    np.testing.assert_allclose(none, [empty, empty])


def test_split_point_balances_and_clamps():
    block_q, tiles = 8, 32
    num_tokens = tiles * block_q
    window = msa_dcp_heads.split_window(num_tokens, block_q)
    flat = jnp.ones(tiles, jnp.float32)
    assert int(msa_dcp_heads.split_point(flat, block_q, window)) == num_tokens // 2
    # Work in the second half: the split moves right, up to the window.
    late = jnp.concatenate([jnp.ones(16), 1.5 * jnp.ones(16)]).astype(jnp.float32)
    s = int(msa_dcp_heads.split_point(late, block_q, window))
    assert s % block_q == 0 and num_tokens // 2 < s <= window
    costs = np.asarray(late)
    left, right = costs[: s // block_q].sum(), costs[s // block_q :].sum()
    assert left >= right  # the boundary tile goes left
    # All work in the last tiles: clamped at the window.
    tail = jnp.concatenate([jnp.zeros(30), jnp.ones(2)]).astype(jnp.float32)
    assert int(msa_dcp_heads.split_point(tail, block_q, window)) == window
    head = jnp.concatenate([jnp.ones(2), jnp.zeros(30)]).astype(jnp.float32)
    assert int(msa_dcp_heads.split_point(head, block_q, window)) == num_tokens - window


# --------------------------------------- head-major K/V gather (bitwise)
class _RecordingGroup:
    """Two-phase stand-in for the DCP group's all-gather: phase one records
    each rank's input, phase two returns the ranks' inputs concatenated."""

    def __init__(self, inputs):
        self.inputs = inputs
        self.rank = 0

    def all_gather(self, x, dim):
        import torch

        if self.inputs is None:
            self.recorded = x.clone()
            return torch.cat([x] * self.world, dim=dim)
        return torch.cat(self.inputs, dim=dim)


def _gathered(fn, ks, vs, iqs, dcp, reps):
    import torch

    recorded = []
    for r in range(dcp):
        group = _RecordingGroup(None)
        group.world = dcp
        fn(ks[r], vs[r], iqs[r], group, dcp, reps, torch.tensor([r]))
        recorded.append(group.recorded)
    group = _RecordingGroup(recorded)
    return [
        fn(ks[r], vs[r], iqs[r], group, dcp, reps, torch.tensor([r]))
        for r in range(dcp)
    ]


@pytest.mark.parametrize("dcp,reps", [(8, 2), (4, 2), (4, 1)])
def test_head_major_gather_is_bitwise(dcp, reps):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    num_tokens, d = 24, 128
    gen = torch.Generator().manual_seed(dcp * 10 + reps)

    def rows():
        # Rank r holds KV head r // reps; its replicas hold the same rows.
        heads = [
            torch.randn(num_tokens, 1, d, generator=gen).bfloat16()
            for _ in range(dcp // reps)
        ]
        return [heads[r // reps] for r in range(dcp)]

    ks, vs, iqs = rows(), rows(), rows()
    old = _gathered(msa_ops.gather_group_kv, ks, vs, iqs, dcp, reps)
    new = _gathered(msa_ops.gather_group_kv_head_major, ks, vs, iqs, dcp, reps)
    for r in range(dcp):
        assert (
            new[r].shape == (dcp // reps, num_tokens, 3 * d) and new[r].is_contiguous()
        )
        split = msa_ops.split_head_major_kv(jnp.asarray(new[r].float().numpy()))
        for want, got in zip(old[r], split):
            np.testing.assert_array_equal(want.float().numpy(), np.asarray(got))
    # Tokens that do not split over the replicas keep the old path.
    if reps > 1:
        odd = [x[:-1] for x in ks]
        assert (
            msa_ops.gather_group_kv_head_major(
                odd[0], odd[0], odd[0], None, dcp, reps, torch.tensor([0])
            )
            is None
        )


@pytest.mark.parametrize("dcp,g", [(4, 2), (8, 4)])
def test_score_from_head_major_rows_is_bitwise(dcp, g):
    """The scoring program over `split_head_major_kv` of the head-major rows
    gives the program over token-major K/V/index queries byte for byte."""
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    rng, num_tokens, meta = reuse_t._setup(30 + dcp, dcp)
    layer = reuse_t._layer(rng, num_tokens, dcp, g, "fp4")
    kvq = jnp.concatenate([layer["k"], layer["v"], layer["iq"]], axis=-1).transpose(
        1, 0, 2
    )
    k, v, iq = msa_ops.split_head_major_kv(kvq)
    for want, got in ((layer["k"], k), (layer["v"], v), (layer["iq"], iq)):
        np.testing.assert_array_equal(np.asarray(want), np.asarray(got))
    distribution = jnp.asarray([0, 0, len(reuse_t.Q_LENS)], jnp.int32)
    rank = 1
    common = dict(
        topk=reuse_t.TOPK,
        rank=rank,
        dcp=dcp,
        scale_cache=layer["scale"][rank],
        interpret=True,
        **reuse_t.CONFIG,
    )
    rest = (layer["ik"], layer["kv"][rank], layer["index"][rank], *meta, distribution)
    want = msa_dcp.msa_write_and_candidates_dcp(
        layer["k"], layer["v"], layer["iq"], *rest, **common
    )
    got = msa_dcp.msa_write_and_candidates_dcp(k, v, iq, *rest, **common)
    for w, x in zip(want, got):
        w, x = np.asarray(w), np.asarray(x)
        if w.dtype.itemsize == 1:
            w, x = w.view(np.uint8), x.view(np.uint8)
        np.testing.assert_array_equal(w, x)


# ------------------------------------- head-parallel SPMD program (4 CPUs)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))


def _spmd(dcp, g, variants, hp_config):
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={dcp}",
        PYTHONPATH=f"{_ROOT}:{_ROOT}/src",
    )
    res = subprocess.run(
        [
            sys.executable,
            os.path.join(_HERE, "dcp_hp_balance_spmd.py"),
            str(dcp),
            str(g),
            variants,
            hp_config,
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=3000,
    )
    assert res.returncode == 0 and "OK" in res.stdout, (
        res.stdout[-3000:] + res.stderr[-4000:]
    )
    return res.stdout


@pytest.mark.parametrize(
    "dcp,g,hp_config",
    [
        (4, 2, "8:2::0"),  # c64 DCP4 layout, two pages per step (multi-step tiles)
        (8, 4, "16:16::1024"),  # c64 DCP8 layout, its served config (page cache)
    ],
)
def test_head_parallel_row_split_and_round_capacity(dcp, g, hp_config):
    """DCP over KV heads of two replicas each (c64 layouts), two prefill
    requests that need two exchange rounds at capacity 1 and one at capacity
    2, one decode request. Row split = the head-split call bit for bit at
    either capacity; capacity 2 = capacity 1 up to bf16 rounding (tiles
    group rows differently); the decode rows never change."""
    out = _spmd(dcp, g, "off:1,split:1,off:2,split:2", hp_config)
    lines = {ln.split()[0]: ln for ln in out.splitlines() if " vs " in ln}
    for v in ("split:1", "off:2", "split:2"):
        assert "decode bitwise=True" in lines[v], out
        assert "pad zero=True" in lines[v], out
        rel = float(lines[v].split("hp rel=")[1].split()[0])
        assert rel < 2e-2, out
    assert "PAIR off:2 split:2 bitwise=True" in out, out
    assert "PAIR off:1 split:1 bitwise=True" in out, out
