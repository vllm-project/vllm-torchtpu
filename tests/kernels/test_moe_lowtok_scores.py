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
"""The few-token kernel's scores entry: the router and the holder choice in
the kernel, with and without copy slots.

Runs on the host in Pallas interpret mode, so it needs no TPU. The
expert-parallel tests need 8 devices; on a CPU host run them with
XLA_FLAGS=--xla_force_host_platform_device_count=8.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.moe_lowtok.expert_parallel import moe_lowtok_scores_ep
from vllm_torchtpu.kernels.moe_lowtok.kernel import (
    LANES,
    hot_onehot,
    hot_tables,
    moe_lowtok,
    moe_lowtok_reference,
    moe_lowtok_scores,
    pack_tables,
    router_select_reference,
)
from vllm_torchtpu.layers.adapter import moe_lowtok_scores as scores_entry

FP8 = jnp.float8_e4m3fn
EP, LOCAL, COPIES, HIDDEN, INTER, TOPK = 8, LANES // 8, 4, 256, 256, 4


def _interpret():
    return jax.default_backend() != "tpu"


def _weights(rng, experts):
    w13 = jnp.asarray(
        np.clip(rng.standard_normal((experts, HIDDEN, 2 * INTER)) * 40, -448, 448), FP8
    )
    w2 = jnp.asarray(
        np.clip(rng.standard_normal((experts, INTER, HIDDEN)) * 40, -448, 448), FP8
    )
    s13 = jnp.asarray(
        rng.random((experts, 2 * INTER)).astype(np.float32) / (40 * np.sqrt(HIDDEN))
    )
    s2 = jnp.asarray(
        rng.random((experts, HIDDEN)).astype(np.float32) / (40 * np.sqrt(INTER))
    )
    return w13, s13, w2, s2


def _contiguous_tables(rank, local):
    gid = np.arange(rank * local, (rank + 1) * local)
    primary = np.arange(LANES) // local
    copy = np.full(LANES, -1)
    hot, hot_primary, hot_copy = hot_tables(primary, copy, [])
    return (
        pack_tables(rank, gid, hot, hot_primary, hot_copy),
        primary,
        copy,
        hot_onehot(hot),
    )


@pytest.mark.parametrize("whole_expert", [False, True])
@pytest.mark.parametrize("tokens", [1, 7, 33])
def test_scores_entry_matches_router_then_ids_entry(whole_expert, tokens):
    """With no copies, routing in the kernel equals the router's selection
    fed to the ids entry."""
    rank, local = 3, 8
    rng = np.random.default_rng(tokens)
    x = jnp.asarray(
        rng.standard_normal((tokens, HIDDEN)).astype(np.float32), jnp.bfloat16
    )
    logits = rng.standard_normal((tokens, LANES)).astype(np.float32)
    logits[:, rank * local : (rank + 1) * local] += 3.0  # some pairs land here
    scores = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    w13, s13, w2, s2 = _weights(rng, local)
    weights, ids = router_select_reference(scores, TOPK)
    local_ids = jnp.where(ids // local == rank, ids - rank * local, -1)
    want = np.asarray(
        moe_lowtok(
            x,
            local_ids,
            weights,
            w13,
            s13,
            w2,
            s2,
            whole_expert=whole_expert,
            interpret=_interpret(),
        ),
        np.float32,
    )
    tables, primary, copy, onehot = _contiguous_tables(rank, local)
    got = moe_lowtok_scores(
        x,
        scores,
        tables,
        primary,
        copy,
        onehot,
        w13,
        s13,
        w2,
        s2,
        topk=TOPK,
        ranks=LANES // local,
        has_copies=False,
        whole_expert=whole_expert,
        interpret=_interpret(),
    )
    assert np.abs(want).max() > 0
    np.testing.assert_allclose(np.asarray(got, np.float32), want, atol=1e-6, rtol=0)


def _copies_table(tmp_path, seed=0):
    """8 ranks, contiguous primaries (virtual id == expert id), and COPIES
    copy slots per rank holding experts of the next rank's block, listed
    hottest first in a shuffled order."""
    rng = np.random.default_rng(seed)
    cores = [list(range(r * LOCAL, (r + 1) * LOCAL)) for r in range(EP)]
    declared = {}
    for r in range(EP):
        block = ((r + 1) % EP) * LOCAL
        for e in rng.choice(LOCAL, size=COPIES, replace=False) + block:
            cores[r].append(int(e))
            declared[str(int(e))] = r
    hot = list(declared)
    rng.shuffle(hot)
    table = {
        "shards": EP,
        "experts": LANES,
        "owned_per_core": LOCAL,
        "slots_per_core": LOCAL + COPIES,
        "copies": True,
        "layers": {"0": {"cores": cores, "copies": {e: declared[e] for e in hot}}},
    }
    path = tmp_path / "copies.json"
    path.write_text(json.dumps(table))
    return str(path), cores, np.array([int(e) for e in declared])


def _ep_mesh():
    devices = jax.devices()
    if len(devices) < EP:
        pytest.skip(
            f"needs {EP} devices; on a CPU host set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=8"
        )
    return jax.sharding.Mesh(np.array(devices[:EP]), ("ep",))


def _dense_reference(x, scores, w13, s13, w2, s2):
    """Every row through its selected experts once (only the hit experts are
    computed): what the expert-parallel layer must return."""
    weights, ids = router_select_reference(scores, TOPK)
    hit = np.unique(np.asarray(ids))
    compact = jnp.asarray(np.searchsorted(hit, np.asarray(ids)).astype(np.int32))
    return moe_lowtok_reference(
        x, compact, weights, w13[hit], s13[hit], w2[hit], s2[hit]
    )


@pytest.mark.multichip
@pytest.mark.parametrize("whole_expert", [False, True])
@pytest.mark.parametrize("with_copies", [False, True])
def test_expert_parallel_scores_entry_matches_dense_layer(
    tmp_path, monkeypatch, whole_expert, with_copies
):
    """Each hit expert is served by exactly one holder: the ranks' summed
    outputs equal the dense layer, and padded rows come back zero."""
    mesh = _ep_mesh()
    rows_per_rank = 2
    rows = EP * rows_per_rank
    rng = np.random.default_rng(int(with_copies))
    x = jnp.asarray(
        rng.standard_normal((rows, HIDDEN)).astype(np.float32), jnp.bfloat16
    )
    logits = rng.standard_normal((rows, LANES)).astype(np.float32)
    if with_copies:
        path, cores, copied = _copies_table(tmp_path)
        monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
        # Route most pairs to copied experts so the holder choice decides.
        logits[:, copied] += 4.0
    else:
        monkeypatch.delenv("TPU_EXPERT_PLACEMENT", raising=False)
        cores = [list(range(r * LOCAL, (r + 1) * LOCAL)) for r in range(EP)]
    padding = np.zeros(rows, np.int32)
    padding[rows_per_rank - 1 :: rows_per_rank] = 1  # each rank's last row
    # Inject NaN into a padded row to verify NaN values do not poison outputs.
    x = x.at[rows_per_rank - 1].set(jnp.nan)
    w13, s13, w2, s2 = _weights(rng, LANES)

    per_rank = [
        scores_entry.build_scores_tables("model.layers.0.mlp.experts", r, EP)
        for r in range(EP)
    ]
    assert all(t.has_copies == with_copies for t in per_rank)
    primaries = np.concatenate([c[:LOCAL] for c in cores])
    copy_weights = None
    if with_copies:
        copies = np.concatenate([c[LOCAL:] for c in cores])
        copy_weights = (w13[copies], w2[copies], s13[copies], s2[copies])
    got = moe_lowtok_scores_ep(
        x,
        jnp.asarray(logits),
        jnp.asarray(padding),
        jnp.asarray(np.stack([t.tables for t in per_rank])),
        jnp.asarray(np.stack([t.primary for t in per_rank])),
        jnp.asarray(np.stack([t.copy for t in per_rank])),
        jnp.asarray(np.concatenate([t.hot_onehot for t in per_rank])),
        w13[primaries],
        w2[primaries],
        s13[primaries],
        s2[primaries],
        copy_weights,
        mesh=mesh,
        topk=TOPK,
        whole_expert=whole_expert,
        interpret=_interpret(),
    )
    got = np.asarray(got, np.float32)
    scores = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    want = np.asarray(_dense_reference(x, scores, w13, s13, w2, s2), np.float32)
    valid = padding == 0
    assert np.all(got[~valid] == 0)
    scale = np.abs(want[valid]).max()
    np.testing.assert_allclose(
        got[valid] / scale, want[valid] / scale, atol=3e-2, rtol=0
    )


@pytest.mark.parametrize("n_fixed_hits,n_copy_hits", [(0, 0), (0, 3), (1, 2)])
def test_scores_whole_expert_copied_only_late_prime(
    tmp_path, monkeypatch, n_fixed_hits, n_copy_hits
):
    """Cover holder-choice late priming and ring wraparound: empty (`0, 0`),
    copied-only wraparound (`0, 3`), and mixed primary + copied wraparound
    (`1, 2`)."""
    rng = np.random.default_rng(77 + n_fixed_hits * 7 + n_copy_hits)
    path, cores, _ = _copies_table(tmp_path, seed=11)
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", path)
    rank0_copies = cores[0][LOCAL:]
    rank7_copies = set(cores[EP - 1][LOCAL:])
    rank0_unshared = [e for e in cores[0][:LOCAL] if e not in rank7_copies]
    rank1_fixed = [e for e in cores[1][:LOCAL] if e not in rank0_copies][:TOPK]
    rank2_fixed = cores[2][:TOPK]

    tokens = 4
    x = jnp.asarray(
        rng.standard_normal((tokens, HIDDEN)).astype(np.float32), jnp.bfloat16
    )
    logits = np.full((tokens, LANES), -30.0, dtype=np.float32)
    if n_fixed_hits == 0 and n_copy_hits == 0:
        logits[:, rank2_fixed] = 10.0
    else:
        # Load rank 1 with 4 fixed hits so rank 0 wins holder choice on its
        # copy slots while staying strictly below rank 1's load.
        logits[0, rank1_fixed] = 10.0
        selected = (
            rank0_unshared[:n_fixed_hits]
            + rank0_copies[:n_copy_hits]
            + rank2_fixed[: TOPK - n_fixed_hits - n_copy_hits]
        )
        for r in range(1, tokens):
            logits[r, selected] = 10.0
    scores = jax.nn.softmax(jnp.asarray(logits), axis=-1)

    t0 = scores_entry.build_scores_tables("model.layers.0.mlp.experts", 0, EP)
    w13, s13, w2, s2 = _weights(rng, LANES)
    prim0 = np.asarray(cores[0][:LOCAL])
    copy0 = np.asarray(cores[0][LOCAL:])
    copy_weights = (w13[copy0], s13[copy0], w2[copy0], s2[copy0])

    split_out = np.asarray(
        moe_lowtok_scores(
            x,
            scores,
            t0.tables,
            t0.primary,
            t0.copy,
            t0.hot_onehot,
            w13[prim0],
            s13[prim0],
            w2[prim0],
            s2[prim0],
            copy_weights,
            topk=TOPK,
            ranks=EP,
            has_copies=True,
            whole_expert=False,
            interpret=_interpret(),
        ),
        np.float32,
    )
    whole_out = np.asarray(
        moe_lowtok_scores(
            x,
            scores,
            t0.tables,
            t0.primary,
            t0.copy,
            t0.hot_onehot,
            w13[prim0],
            s13[prim0],
            w2[prim0],
            s2[prim0],
            copy_weights,
            topk=TOPK,
            ranks=EP,
            has_copies=True,
            whole_expert=True,
            interpret=_interpret(),
        ),
        np.float32,
    )
    if n_fixed_hits == 0 and n_copy_hits == 0:
        assert np.all(split_out == 0)
        assert np.all(whole_out == 0)
        return
    assert np.abs(split_out).max() > 0
    scale = float(np.abs(split_out).max())
    tol = 1e-6 if _interpret() else 5e-3
    np.testing.assert_allclose(whole_out / scale, split_out / scale, atol=tol, rtol=0)


@pytest.mark.tpu_test
@pytest.mark.parametrize("whole_expert", [False, True])
def test_tpu_scores_entry_whole_expert(whole_expert):
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    rank, local, tokens = 3, 8, 8
    rng = np.random.default_rng(53)
    x = jnp.asarray(
        rng.standard_normal((tokens, HIDDEN)).astype(np.float32), jnp.bfloat16
    )
    logits = rng.standard_normal((tokens, LANES)).astype(np.float32)
    logits[:, rank * local : (rank + 1) * local] += 3.0
    scores = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    w13, s13, w2, s2 = _weights(rng, local)
    weights, ids = router_select_reference(scores, TOPK)
    local_ids = jnp.where(ids // local == rank, ids - rank * local, -1)
    want = np.asarray(
        moe_lowtok(
            x,
            local_ids,
            weights,
            w13,
            s13,
            w2,
            s2,
            whole_expert=whole_expert,
            interpret=False,
        ),
        np.float32,
    )
    tables, primary, copy, onehot = _contiguous_tables(rank, local)
    got = np.asarray(
        moe_lowtok_scores(
            x,
            scores,
            tables,
            primary,
            copy,
            onehot,
            w13,
            s13,
            w2,
            s2,
            topk=TOPK,
            ranks=LANES // local,
            has_copies=False,
            whole_expert=whole_expert,
            interpret=False,
        ),
        np.float32,
    )
    scale = float(np.abs(want).max())
    np.testing.assert_allclose(got / scale, want / scale, atol=1e-5, rtol=0)


@pytest.mark.tpu_test
@pytest.mark.parametrize("n_fixed_hits,n_copy_hits", [(0, 0), (0, 3), (1, 2)])
def test_tpu_scores_whole_expert_copied_hits(
    tmp_path, monkeypatch, n_fixed_hits, n_copy_hits
):
    """Compiles and runs whole_expert=True with has_copies=True on TPU across
    0-hit, copied-only late-prime, and mixed fixed+copy hit counts."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    test_scores_whole_expert_copied_only_late_prime(
        tmp_path, monkeypatch, n_fixed_hits, n_copy_hits
    )


def test_hot_tables_must_list_every_copy():
    primary = np.arange(LANES) // LOCAL
    copy = np.full(LANES, -1)
    copy[[5, 70]] = [1, 0]
    hot, hot_primary, hot_copy = hot_tables(primary, copy, [70, 5])
    assert hot[:3].tolist() == [70, 5, -1]
    assert hot_primary[:2].tolist() == [1, 0] and hot_copy[:2].tolist() == [0, 1]
    with pytest.raises(ValueError):
        hot_tables(primary, copy, [5])


@pytest.mark.multichip
@pytest.mark.tpu_test
def test_tpu_multichip_scores_ep_threshold_straddling_parity():
    """On an 8-core TPU mesh, buckets below and at whole_expert_min_tokens
    select split vs whole-expert streaming and match each other numerically."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from vllm_torchtpu.kernels.moe_lowtok.expert_parallel import (
        moe_lowtok_scores_ep,
        resolve_whole_expert_ep,
    )

    devices = jax.devices()
    if jax.default_backend() != "tpu" or len(devices) < EP:
        pytest.skip("requires >= 8 TPU devices")
    mesh = Mesh(np.array(devices[:EP]), ("ep",))
    sharding = NamedSharding(mesh, PartitionSpec("ep"))
    rng = np.random.default_rng(91)
    w13, s13, w2, s2 = _weights(rng, EP * LOCAL)
    w13_d = jax.device_put(w13, sharding)
    w2_d = jax.device_put(w2, sharding)
    s13_d = jax.device_put(s13.reshape(EP * LOCAL, -1), sharding)
    s2_d = jax.device_put(s2.reshape(EP * LOCAL, -1), sharding)
    tabs, prims, copies, onehots = zip(
        *(_contiguous_tables(r, LOCAL) for r in range(EP)), strict=True
    )
    tables_d = jax.device_put(np.stack(tabs), sharding)
    prim_d = jax.device_put(np.stack(prims), sharding)
    copy_d = jax.device_put(np.stack(copies), sharding)
    hot_d = jax.device_put(np.stack(onehots), sharding)

    min_tokens = 64
    traced_modes: list[tuple[int, bool]] = []

    @jax.jit
    def gated_ep(x_in, g_in, p_in):
        use_whole = resolve_whole_expert_ep(
            x_in,
            w13_d,
            w2_d,
            s13_d,
            s2_d,
            ep=EP,
            activation="silu",
            whole_expert=True,
            whole_expert_min_tokens=min_tokens,
            scores=True,
        )
        traced_modes.append((x_in.shape[0], use_whole))
        return moe_lowtok_scores_ep(
            x_in,
            g_in,
            p_in,
            tables_d,
            prim_d,
            copy_d,
            hot_d,
            w13_d,
            w2_d,
            s13_d,
            s2_d,
            mesh=mesh,
            topk=TOPK,
            whole_expert=use_whole,
        )

    for local_rows, want_whole in ((4, False), (8, True)):
        total_rows = EP * local_rows
        x_np = rng.standard_normal((total_rows, HIDDEN)).astype(np.float32)
        g_np = rng.standard_normal((total_rows, LANES)).astype(np.float32)
        x_d = jax.device_put(jnp.asarray(x_np, jnp.bfloat16), sharding)
        gate_d = jax.device_put(jnp.asarray(g_np, jnp.bfloat16), sharding)
        pad_d = jax.device_put(np.zeros((total_rows,), dtype=np.int32), sharding)
        got = np.asarray(gated_ep(x_d, gate_d, pad_d), np.float32)
        assert traced_modes[-1] == (total_rows, want_whole)

        # Independent cross-mode check: compare against the opposite streaming
        # mode across the EP mesh so neither bucket compares a mode to itself.
        opposite = np.asarray(
            jax.jit(
                lambda x_in, g_in, p_in, opp=(not want_whole): moe_lowtok_scores_ep(
                    x_in,
                    g_in,
                    p_in,
                    tables_d,
                    prim_d,
                    copy_d,
                    hot_d,
                    w13_d,
                    w2_d,
                    s13_d,
                    s2_d,
                    mesh=mesh,
                    topk=TOPK,
                    whole_expert=opp,
                )
            )(x_d, gate_d, pad_d),
            np.float32,
        )
        scale = max(float(np.abs(opposite).max()), 1e-6)
        np.testing.assert_allclose(got / scale, opposite / scale, atol=5e-3, rtol=0)
