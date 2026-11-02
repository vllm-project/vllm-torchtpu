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
"""Static EPLB: replica table, logical->physical remap, and an end-to-end
dense reference showing that routing through replica slots reproduces the
logical-expert MoE output. CPU only (JAX_PLATFORMS=cpu)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.adapter import eplb_static

E, EP, R, K, H, INTER = 32, 4, 2, 4, 16, 8


def _spec():
    # Hot experts 0 (2 replicas), 1, 9 (1 replica each); rank 3 has an idle slot.
    reps = [(0, 1), (0, 2), (1, 3), (9, 0), (1, 2), (9, 1), (0, 0)]
    return eplb_static.build_spec(E, EP, R, reps, layer=5)


def _phys_to_log(spec):
    g = spec.g
    out = {}
    for r in range(spec.ep):
        for j in range(g):
            out[r * (g + spec.r) + j] = r * g + j
        for s, e in enumerate(spec.local_sources(r)):
            out[r * (g + spec.r) + g + s] = e
    return out


def test_table_layout():
    spec = _spec()
    g = E // EP
    assert spec.num_physical == E + EP * R
    reps = spec.replicas()
    for e in range(E):
        assert reps[e][0] == (e // g) * (g + R) + e % g
        assert len(set(reps[e])) == len(reps[e])
    assert len(reps[0]) == 4 and len(reps[1]) == 3 and len(reps[9]) == 3
    assert len(reps[5]) == 1
    t = spec.table()
    assert t.shape == (E, EP) and t.dtype == np.int32
    # Every entry is a copy of the right logical expert.
    p2l = _phys_to_log(spec)
    assert len(p2l) == spec.num_physical
    for e in range(E):
        for src in range(EP):
            assert p2l[int(t[e, src])] == e
    # The idle slot (rank 3, slot 1) is never a routing target.
    assert spec.slots[3][1] == -1
    idle = 3 * (g + R) + g + 1
    assert idle not in set(t.ravel().tolist())


def test_too_many_replicas_rejected():
    with pytest.raises(ValueError):
        eplb_static.build_spec(E, EP, 1, [(0, 1), (1, 1)])


def test_remap_locality_and_passthrough():
    spec = _spec()
    g_local = E // EP + R
    t = jnp.asarray(spec.table())
    ids = jnp.asarray([[0, 1, 5, E]] * 8, jnp.int32)  # E = sentinel
    for src in range(EP):
        out = np.asarray(eplb_static.remap_topk_ids(ids, t, jnp.int32(src)))
        assert (out[:, 3] == E).all()
        # Expert 0 has copies on ranks 0 (home + replica), 1, 2: local there;
        # rank 3 has none and uses copy 3 % 4.
        if src < 3:
            assert (out[:, 0] // g_local == src).all()
        else:
            assert (out[:, 0] == spec.replicas()[0][3]).all()
        # Non-replicated expert keeps its home id.
        assert (out[:, 2] == spec.home_phys(5)).all()
    # Expert 1 lives on ranks 0 (home), 3 and 2: ranks 0/2/3 stay local,
    # rank 1 uses copy 1 % 3.
    got = {
        src: int(eplb_static.remap_topk_ids(ids, t, jnp.int32(src))[0, 1])
        for src in range(EP)
    }
    for src in (0, 2, 3):
        assert got[src] // g_local == src
    assert got[1] == spec.replicas()[1][1 % 3]


def _dense_moe(x, ids, w, w1, w2):
    """sum_k w[t,k] * FFN_{ids[t,k]}(x[t]) with a SiLU-gated FFN."""
    out = 0
    for k in range(ids.shape[1]):
        a = jnp.einsum("th,thi->ti", x, w1[ids[:, k]])
        gate, up = jnp.split(a, 2, axis=-1)
        y = jnp.einsum("ti,tih->th", jax.nn.silu(gate) * up, w2[ids[:, k]])
        out = out + w[:, k : k + 1] * y
    return out


def test_dense_reference_matches_logical():
    spec = _spec()
    g = E // EP
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w1 = jax.random.normal(k1, (E, H, 2 * INTER), jnp.float32)
    w2 = jax.random.normal(k2, (E, INTER, H), jnp.float32)
    x = jax.random.normal(k3, (48, H), jnp.float32)
    logits = jax.random.normal(k4, (48, E), jnp.float32)
    # Concentrate routing on the hot experts.
    logits = logits.at[:, 0].add(5.0).at[:, 1].add(4.0).at[:, 9].add(3.0)
    wts, ids = jax.lax.top_k(jax.nn.softmax(logits), K)
    ref = _dense_moe(x, ids, wts, w1, w2)

    # Physical weights: per rank, home block then replica sources, exactly what
    # the loader writes into the [G + R] local slots.
    order = []
    for r in range(EP):
        order += list(range(r * g, (r + 1) * g)) + spec.local_sources(r)
    order = jnp.asarray(order)
    pw1, pw2 = w1[order], w2[order]
    # Rows are spread over the source ranks the way the kernel shards them.
    t = jnp.asarray(spec.table())
    rows = ids.shape[0] // EP
    pids = jnp.concatenate(
        [
            eplb_static.remap_topk_ids(ids[s * rows : (s + 1) * rows], t, jnp.int32(s))
            for s in range(EP)
        ]
    )
    assert int(pids.max()) < spec.num_physical
    # Some rows really do go to replica slots.
    homes = jnp.asarray([spec.home_phys(int(e)) for e in np.asarray(ids).ravel()])
    assert int((pids.ravel() != homes).sum()) > 0
    out = _dense_moe(x, pids, wts, pw1, pw2)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_weight_loader_wrapper_writes_replica_slots():
    spec = _spec()
    g = E // EP
    rank = 2

    class FakeLayer:
        def __init__(self):
            self.data = np.full((g + R,), -1)

        def _map_global_expert_id_to_local_expert_id(self, e):
            return e - rank * g if rank * g <= e < (rank + 1) * g else -1

        def weight_loader(
            self, param, w, name, shard_id=None, expert_id=None, return_success=False
        ):
            local = self._map_global_expert_id_to_local_expert_id(expert_id)
            if local == -1:
                return False if return_success else None
            self.data[local] = w
            return True if return_success else None

    layer = FakeLayer()
    loader = eplb_static.wrap_weight_loader(layer, layer.weight_loader, spec, rank)
    seen = []
    for e in range(E):
        ok = loader(None, e, "w", shard_id="w1", expert_id=e, return_success=True)
        seen.append(ok)
    assert layer.data[:g].tolist() == list(range(rank * g, (rank + 1) * g))
    assert layer.data[g:].tolist() == spec.local_sources(rank) == [0, 1]
    assert seen[0] and seen[1] and not seen[9]
    # The per-instance override is removed again.
    assert "_map_global_expert_id_to_local_expert_id" not in vars(layer)
