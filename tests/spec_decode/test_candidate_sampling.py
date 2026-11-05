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
"""Candidate verify sampling against the full-vocabulary torch sampler.

Both paths get the same uniform per (row, token): the candidate path reads the
full-vocabulary draws at its candidate ids. The vocab-parallel ranks are
emulated with ``jax.vmap`` over a named axis, so the gather and the in-program
fallback run as on the TP group. Bonus tokens and the rejection sampler's
output must match bitwise.
"""

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.layers.core import candidate_sampling as cs

K = 3


def _eager(fn):
    return getattr(fn, "_torchdynamo_orig_callable", fn)


def _reference_bonus(logits, temps, u, top_k, top_p):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = types.SimpleNamespace(
        _apply_temperature=functools.partial(TPUModelRunner._apply_temperature, None)
    )
    out = _eager(TPUModelRunner.sample_from_logits)(
        runner, logits, temps, u, top_k, top_p, all_greedy=False
    )
    return out.view(-1)


def _to_jax(t):
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _case(
    seed,
    *,
    tp=4,
    shard=512,
    num_reqs=6,
    temp=0.7,
    top_k=20,
    top_p=0.8,
    greedy_frac=0.0,
    ties=0,
    synthetic=False,
    force_full=False,
):
    rng = np.random.default_rng(seed)
    vocab = tp * shard
    rows = num_reqs * (K + 1) + 2  # padded logits length
    logits = torch.from_numpy(
        (rng.normal(size=(rows, vocab)) * 3).astype(np.float32)
    ).to(torch.bfloat16)
    if ties:
        logits[:, :ties] = float(logits.max()) - 0.5
    bonus_idx = torch.arange(num_reqs, dtype=torch.int32) * (K + 1) + K
    target_idx = torch.arange(rows, dtype=torch.int32)
    draft = torch.from_numpy(rng.integers(0, vocab, rows).astype(np.int32))
    greedy_ids = logits.argmax(-1).to(torch.int32)
    draft = torch.where(torch.from_numpy(rng.random(rows) < 0.5), greedy_ids, draft)
    draft[-1] = -1
    temps = torch.full((num_reqs, 1), temp, dtype=torch.bfloat16)
    temps[torch.from_numpy(rng.random(num_reqs) < greedy_frac)] = 0.0
    top_k_r = torch.full((num_reqs, 1), top_k, dtype=torch.int32)
    top_p_r = torch.full((num_reqs, 1), top_p, dtype=torch.float32)
    segment = torch.cat(
        [
            torch.repeat_interleave(torch.arange(num_reqs), K),
            torch.full((rows - num_reqs * K,), num_reqs),
        ]
    )
    seg = segment.clamp(max=num_reqs - 1)
    group = torch.cat(
        [
            torch.arange(K, dtype=torch.int32).repeat(num_reqs),
            torch.zeros(rows - num_reqs * K, dtype=torch.int32),
        ]
    )
    num_draft = torch.full((num_reqs,), K, dtype=torch.int32)
    gen = torch.Generator().manual_seed(seed)
    u_b = torch.rand((num_reqs, vocab), generator=gen).to(torch.bfloat16)
    u_r = torch.rand((rows, vocab), generator=gen)
    accept_u = torch.rand((rows,), generator=gen)
    rates = torch.tensor([0.832, 0.692 / 0.832, 0.576 / 0.692])
    rates_t = rates[group.long()]

    ref_bonus = _reference_bonus(logits[bonus_idx], temps, u_b, top_k_r, top_p_r)
    ref_out = _eager(rs._random_rejection_sample_with_segment)(
        draft,
        logits[target_idx],
        num_draft,
        ref_bonus,
        segment,
        group,
        K,
        temps[seg],
        top_k_r[seg],
        top_p_r[seg],
        accept_u,
        u_r,
        rates if synthetic else None,
        synthetic,
    )

    kc = 64
    width = kc * tp

    def shards(rows):
        return _to_jax(rows).reshape(rows.shape[0], tp, shard).transpose(1, 0, 2)

    bounds = jnp.asarray([[r * shard, (r + 1) * shard] for r in range(tp)], jnp.int32)
    local = functools.partial(
        cs.verify_local,
        axis="tp",
        kc=kc,
        vocab_size=vocab,
        synthetic=synthetic,
        force_full=force_full,
        full_uniforms=(_to_jax(u_b), _to_jax(u_r)),
    )
    run = jax.jit(jax.vmap(local, axis_name="tp"))

    def call(bonus_u, recover_u):
        replicated = [
            draft,
            temps,
            top_k_r,
            top_p_r,
            temps[seg],
            top_k_r[seg],
            top_p_r[seg],
            bonus_u,
            accept_u,
            recover_u,
            rates_t,
            torch.zeros(2, dtype=torch.int32),
        ]
        operands = [shards(logits[bonus_idx]), shards(logits[target_idx]), bounds]
        operands += [
            jnp.broadcast_to(_to_jax(t)[None], (tp, *t.shape)) for t in replicated
        ]
        return run(*operands)

    # The candidate ids come first; then every candidate gets the uniform the
    # full-vocabulary path draws for its token.
    *_, ids_b, ids_t = call(
        torch.zeros((num_reqs, width), dtype=torch.bfloat16), torch.zeros((rows, width))
    )
    ids_b = torch.from_numpy(np.asarray(ids_b[0]).astype(np.int64))
    ids_t = torch.from_numpy(np.asarray(ids_t[0]).astype(np.int64))
    bonus, accepted, recovered, exact, _, _ = call(
        torch.gather(u_b, 1, ids_b), torch.gather(u_r, 1, ids_t)
    )
    for r in range(1, tp):  # every rank decides the same
        assert np.array_equal(np.asarray(bonus[r]), np.asarray(bonus[0]))
        assert np.array_equal(np.asarray(recovered[r]), np.asarray(recovered[0]))
    bonus = torch.from_numpy(np.asarray(bonus[0]).copy())
    out = rs.assemble_rejection_output(
        torch.from_numpy(np.asarray(accepted[0]).copy()) != 0,
        torch.from_numpy(np.asarray(recovered[0]).copy()),
        draft,
        num_draft,
        bonus,
        segment,
        group,
        K,
    )
    return ref_bonus, ref_out, bonus, out, int(exact[0])


@pytest.mark.cpu_test
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "kwargs,exact",
    [
        (dict(), 1),  # temperature 0.7, top_p 0.8, top_k 20
        (dict(temp=0.6, top_p=0.95), 1),
        (dict(top_p=1.0), 1),
        (dict(top_k=1), 1),
        (dict(top_k=48), 1),
        (dict(temp=0.05, top_p=0.5, top_k=5), 1),
        (dict(greedy_frac=0.5), 1),  # mixed greedy / sampled requests
        (dict(synthetic=True), 1),
        (dict(ties=30), 1),  # ties at the cutoff that fit the candidates
        (dict(ties=70), 0),  # ties that fill shard 0's candidates
        (dict(top_k=0), 0),  # no top-k: the full-vocabulary path
        (dict(force_full=True, temp=0.6, top_p=0.95), 0),
        (dict(force_full=True, temp=1.0, top_p=1.0, top_k=0), 0),
    ],
)
def test_candidates_match_full_vocabulary_sampler(seed, kwargs, exact):
    ref_bonus, ref_out, bonus, out, got_exact = _case(seed, **kwargs)
    assert got_exact == exact
    torch.testing.assert_close(bonus.long(), ref_bonus.long(), rtol=0, atol=0)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


@pytest.mark.cpu_test
def test_rows_exact_needs_every_shard_boundary_below_cutoff():
    cutoff = jnp.asarray([1.0, 1.0, 1.0, 1.0])
    scaled_boundary = jnp.asarray(
        [[0.5, 0.9], [1.0, 0.5], [0.5, 0.5], [2.0, 2.0]], jnp.float32
    )
    is_greedy = jnp.asarray([False, False, False, True])
    top_k = jnp.asarray([[20], [20], [0], [20]], jnp.int32)
    got = cs.rows_exact(scaled_boundary, is_greedy, top_k, cutoff)
    assert np.asarray(got).tolist() == [True, False, False, True]


@pytest.mark.cpu_test
@pytest.mark.parametrize("kind", ["normal", "integers", "flat"])
def test_shard_candidates_are_the_shard_top_kc(kind):
    """The candidates are kc of the shard's largest logits (as a multiset),
    no logit left out exceeds the smallest one, and ids point at them."""
    rng = np.random.default_rng(7)
    rows, width, kc, start = 6, 128 * 40, 64, 1000
    if kind == "normal":
        x = rng.normal(size=(rows, width)) * 3
    elif kind == "integers":  # heavy ties everywhere
        x = rng.integers(-20, 20, size=(rows, width)).astype(np.float64)
    else:  # one value
        x = np.zeros((rows, width))
    logits = jnp.asarray(x, jnp.float32).astype(jnp.bfloat16)
    bounds = jnp.asarray([start, start + width - 5], jnp.int32)  # 5 padding columns
    _, vals, ids, boundary, top, first = jax.jit(
        functools.partial(cs.shard_candidates, kc=kc)
    )(logits, bounds)
    xs = np.asarray(logits.astype(jnp.float32))[:, : width - 5]
    vals = np.asarray(vals.astype(jnp.float32))
    ids = np.asarray(ids)
    for r in range(rows):
        want = np.sort(xs[r])[::-1][:kc]
        np.testing.assert_array_equal(np.sort(vals[r])[::-1], want)
        assert len(set(ids[r].tolist())) == kc
        np.testing.assert_array_equal(xs[r, ids[r] - start], vals[r])
        rest = np.delete(xs[r], ids[r] - start)
        assert rest.max() <= vals[r].min() == float(boundary[r])
        assert int(first[r]) - start == int(np.argmax(xs[r]))
        assert float(top[r]) == xs[r].max()
