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
"""The fused small all-reduce + mHC seam kernel (``ar_seam_kernel``) on the 8
cores of one host: every output bit-identical to the unfused pair it replaces
(``small_all_reduce`` -> ``mhc_fused_post_pre`` with in-kernel gates, fused
norm and the stacked fn), for every all-reduce form, over subgroups named by a
peer table, at decode bucket sizes; and a broken variant must differ."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.mhc import ar_seam_kernel as KS
from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel as K
from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar

pytestmark = pytest.mark.multichip

N, AXIS, H, M = 8, "d", 7168, 4
M3 = M * (M + 2)
EPS, POST_MULT, SINKHORN = 1e-6, 2.0, 20


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("needs the 8 TPU cores of one host")
    return Mesh(np.array(jax.devices()[:N]), (AXIS,))


def _tables(groups):
    rank = np.zeros((N, 1), np.int32)
    peers = np.zeros((N, len(groups[0])), np.int32)
    for g in groups:
        for pos, core in enumerate(g):
            rank[core], peers[core] = pos, g
    return rank, peers


def _operands(groups, T, seed):
    rng = np.random.default_rng(seed)

    def act(shape, scale=1.0):
        x = rng.standard_normal(shape).astype(np.float32)
        x = np.where(rng.random(shape) < 1e-3, x * 30.0, x)
        return (x * scale).astype(jnp.bfloat16)

    x_p = act((N, T, H), 1.0 / np.sqrt(len(groups[0])))
    res = np.zeros((N, T, M * H), jnp.bfloat16)
    post = np.zeros((N, T, M), np.float32)
    comb = np.zeros((N, T, M * M), np.float32)
    for g in groups:
        r = act((T, M * H))
        p = (2.0 / (1.0 + np.exp(-rng.standard_normal((T, M))))).astype(np.float32)
        c = np.exp(rng.standard_normal((T, M, M)))
        for _ in range(20):
            c = c / c.sum(-1, keepdims=True)
            c = c / c.sum(-2, keepdims=True)
        for core in g:
            res[core], post[core], comb[core] = r, p, c.reshape(T, M * M)
    fn32 = (rng.standard_normal((M3, M * H)) * 0.02).astype(np.float32)
    fn_cat = np.asarray(utils.cat_fn3(jnp.asarray(fn32)).cat)
    sc = (1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32)
    hb = (0.5 * rng.standard_normal(M3)).astype(np.float32)
    nw = rng.uniform(0.5, 1.5, (1, H)).astype(jnp.bfloat16)
    rank, peers = _tables(groups)
    tile = lambda a: np.broadcast_to(a, (N,) + a.shape)  # noqa: E731
    return [x_p, res, post, comb, tile(fn_cat), tile(sc[None]), tile(hb[None]), tile(nw), rank, peers]


def _run(mesh, body, ops, n_out):
    spec = NamedSharding(mesh, P(AXIS))
    args = [jax.device_put(np.asarray(o).reshape((-1,) + np.asarray(o).shape[2:]), spec) for o in ops]
    fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(AXIS),) * len(args),
                               out_specs=(P(AXIS),) * n_out, check_vma=False))
    return [np.asarray(o) for o in fn(*args)]


CASES = [
    # (groups, T, form, chunks, fn_chunks, small_block)
    ((tuple(range(8)),), 8, None, 1, 2, 8),  # c4: two-shot TP8, 2 fn chunks
    ((tuple(range(8)),), 16, None, 1, 3, 16),  # c8
    ((tuple(range(8)),), 32, None, 1, 3, 16),
    ((tuple(range(8)),), 1, None, 1, 2, 8),  # one-shot
    ((tuple(range(8)),), 24, None, 1, 2, 8),  # c4 extra bucket 24 (TPU_TOKEN_BUCKET_EXTRA)
    ((tuple(range(8)),), 16, None, 1, 2, 8),  # c4 T16
    (((0, 1, 2, 3), (4, 5, 6, 7)), 8, "pairs", 2, 3, 16),  # c16 T8
    (((6, 0, 2, 4, 3, 5, 7, 1),), 16, None, 1, 3, 16),  # permuted peer table
    (((0, 1, 2, 3), (4, 5, 6, 7)), 16, "pairs", 2, 3, 16),  # c16: pairs TP4 x2
    (((0, 1, 2, 3), (4, 5, 6, 7)), 32, "pairs", 2, 3, 16),
    (((0, 1, 2, 3), (4, 5, 6, 7)), 4, "pairs", 2, 3, 16),
    (((6, 0, 2, 4), (3, 5, 7, 1)), 16, "twoshot", 1, 3, 16),  # TP4 subgroups, two-shot
]


@pytest.mark.parametrize(
    "in_mode,out_mode,fn_mode", [("vmem", "vmem", "vmem"), ("vmem", "vmem", "pin"), ("any", "dma", "any")]
)
@pytest.mark.parametrize("groups,T,form,chunks,fn_chunks,small_block", CASES)
def test_bit_identical_to_unfused_pair(
    mesh, groups, T, form, chunks, fn_chunks, small_block, in_mode, out_mode, fn_mode
):
    n = len(groups[0])
    gates = utils.GateConsts(EPS, POST_MULT, SINKHORN)
    ops = _operands(groups, T, seed=T * 31 + n)

    def unfused(x_p, res, post, comb, fn_cat, sc, hb, nw, r, p):
        x = ar.small_all_reduce(x_p, num_cores=n, axis=AXIS, rank=r, peers=p, form=form, chunks=chunks)
        out = K.fused_post_pre_mixes(
            x, res, post, comb, utils.CatFn(fn_cat), sc[0], hb[0], EPS, EPS, token_block_size=128,
            gates=gates, norm_weight=nw, norm_eps=EPS, small_block=small_block, fn_chunks=fn_chunks)
        return (*out, x)

    def fused(x_p, res, post, comb, fn_cat, sc, hb, nw, r, p, fault=None, fn_mode=fn_mode):
        return KS.ar_seam(
            x_p, res, post, comb, fn_cat, sc[0], hb[0], nw, r, p, num_cores=n, axis=AXIS, form=form,
            chunks=chunks, rms_eps=EPS, hc_pre_eps=EPS, gates=gates, norm_eps=EPS, fn_chunks=fn_chunks,
            small_block=small_block, in_mode=in_mode, out_mode=out_mode, fn_mode=fn_mode,
            emit_x=True, fault=fault)

    want = _run(mesh, unfused, ops, 5)
    got = _run(mesh, fused, ops, 5)
    for name, w, g in zip(("new_res", "post_mix", "comb_mix", "layer", "x"), want, got):
        assert w.shape == g.shape, name
        assert np.array_equal(w.view(np.uint8), g.view(np.uint8)), (
            name, int((w != g).sum()), w.size)
    # The check can fail: a short fn read (hi chunk only) changes the outputs.
    bad = _run(mesh, lambda *a: fused(*a, fault="fn_short", fn_mode="pin"), ops, 5)
    assert not np.array_equal(want[3].view(np.uint16), bad[3].view(np.uint16))
