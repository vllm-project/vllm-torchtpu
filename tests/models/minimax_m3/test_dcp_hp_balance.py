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
"""TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR (interpret mode, CPU): the group's
K/V/index queries handed to the scoring op head-major give the token-major
gather and scoring program bit for bit. The head-parallel row split and round
buffer size are covered by test_dcp_hp_row_split.py / test_dcp_hp_config.py."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3 import test_msa_index_reuse as reuse_t
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp


def test_env_default_off(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR", raising=False)
    assert envs.TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR is False
    monkeypatch.setenv("TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR", "1")
    assert envs.TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR is True


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
