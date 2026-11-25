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
    ref_out = rs._random_rejection_sample_impl(
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


@pytest.mark.cpu_test
@pytest.mark.parametrize("value,enabled", [(None, False), ("1", True), ("0", False)])
def test_candidate_sampling_is_off_unless_switched_on(monkeypatch, value, enabled):
    from vllm_torchtpu import envs

    if value is None:
        monkeypatch.delenv("VLLM_TPU_CANDIDATE_SAMPLING", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_CANDIDATE_SAMPLING", value)
    assert envs.environment_variables["VLLM_TPU_CANDIDATE_SAMPLING"]() is enabled


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "change,expected",
    [
        (None, True),  # every request keeps top_k 20
        ("mixed_greedy", True),  # greedy requests need no top_k
        ("top_k_max", True),
        ("no_bounds", False),  # switched off or no vocab-parallel head
        ("all_greedy", False),
        ("no_top_k", False),
        ("top_k_above_max", False),
        ("logprobs", False),
        ("structured", False),
    ],
)
@pytest.mark.parametrize("program", [False, True])
def test_candidates_only_when_every_request_keeps_few_tokens(
    monkeypatch, change, expected, program
):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter.sample.candidate_sampling import MAX_TOP_K
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    temps = torch.tensor([0.7, 0.7, 0.7, 0.0])  # the last slot is unused
    top_k = torch.tensor([20, 20, 20, 0], dtype=torch.int32)
    if change == "mixed_greedy":
        temps[1], top_k[1] = 0.0, 0
    elif change == "top_k_max":
        top_k[2] = MAX_TOP_K
    elif change == "no_top_k":  # the batch stores the vocabulary size
        top_k[2] = 151936
    elif change == "top_k_above_max":
        top_k[2] = MAX_TOP_K + 1
    runner = types.SimpleNamespace(
        _candidate_bounds=None if change == "no_bounds" else torch.tensor([0, 64]),
        input_batch=types.SimpleNamespace(
            all_greedy=change == "all_greedy",
            max_num_logprobs=5 if change == "logprobs" else None,
            num_reqs=3,
            temperature_cpu_tensor=temps,
            top_k_cpu_tensor=top_k,
            temperature_cpu=temps.numpy(),  # the batch's numpy views
            top_k_cpu=top_k.numpy(),
        ),
    )
    # VLLM_TPU_CANDIDATE_VERIFY_PROGRAM reads the numpy views; same answer.
    monkeypatch.setattr(
        envs, "VLLM_TPU_CANDIDATE_VERIFY_PROGRAM", program, raising=False
    )
    scheduler_output = types.SimpleNamespace(
        has_structured_output_requests=change == "structured"
    )
    assert TPUModelRunner._samples_candidates(runner, scheduler_output) is expected


# --- VLLM_TPU_CANDIDATE_VERIFY_PROGRAM: the verify chunk as one program ------


@functools.cache
def _vmapped_verify(tp, shard, synthetic):
    """`verify_local` over a `tp`-rank axis, jitted once per geometry."""
    local = functools.partial(
        cs.verify_local, axis="tp", kc=64, vocab_size=tp * shard, synthetic=synthetic
    )
    return jax.jit(jax.vmap(local, axis_name="tp"))


def _emulated_op(bonus_rows, target_rows, tp, shard, synthetic):
    """The candidate op of rank 0 of a `tp`-rank group, emulated with
    `jax.vmap` over the TP axis: the program hands it rank 0's [1, ...]
    operands; the other ranks' logits shards and vocab bounds come from the
    full rows (`bonus_rows` / `target_rows`, gathered at the same indices)."""
    run = _vmapped_verify(tp, shard, synthetic)

    def shards(rows):
        return _to_jax(rows).reshape(rows.shape[0], tp, shard).transpose(1, 0, 2)

    bounds = jnp.asarray([[r * shard, (r + 1) * shard] for r in range(tp)], jnp.int32)

    def op(*operands):
        assert len(operands) == 15 and all(t.shape[0] == 1 for t in operands)
        stacked = [shards(bonus_rows), shards(target_rows), bounds]
        stacked += [
            jnp.broadcast_to(_to_jax(t[0])[None], (tp, *t.shape[1:]))
            for t in operands[3:]
        ]
        outs = run(*stacked)
        return tuple(torch.from_numpy(np.asarray(o[:1]).copy()) for o in outs)

    return op


def _chunk(seed, num_reqs, *, tp=2, shard=128, temps, top_k, top_p, peaked=True):
    """One verify chunk laid out as get_spec_decode_metadata lays it out:
    num_reqs x (K drafts + bonus) logits rows over a tp x shard vocabulary,
    every request's sampling params tiled from `temps` / `top_k` / `top_p`
    (one entry per request type). Drafts mix the argmax and the runner-up."""
    rng = np.random.default_rng(seed)
    types_ = len(temps)
    vocab = tp * shard
    rows = num_reqs * (K + 1)
    base = rng.normal(size=(types_ * (K + 1), vocab)) * (1.5 if peaked else 3)
    if peaked:  # a few dozen strong tokens over a flat bulk, LM-like
        for r in range(base.shape[0]):
            base[r, rng.choice(vocab, 24, replace=False)] += rng.uniform(2, 6, 24)
    reps = num_reqs // types_
    logits = torch.from_numpy(np.tile(base, (reps, 1)).astype(np.float32)).to(
        torch.bfloat16
    )
    reqs = torch.arange(num_reqs, dtype=torch.int32)
    bonus_idx = reqs * (K + 1) + K
    target_idx = (reqs[:, None] * (K + 1) + torch.arange(K, dtype=torch.int32)).view(-1)
    order = logits.float().argsort(dim=-1, descending=True)
    pick = torch.from_numpy(np.tile(rng.integers(0, 2, types_ * (K + 1)), reps))
    draft = order[torch.arange(rows), pick].to(torch.int32)[target_idx.long()]
    segment = torch.repeat_interleave(torch.arange(num_reqs), K)
    group = torch.arange(K, dtype=torch.int32).repeat(num_reqs)
    num_draft = torch.full((num_reqs,), K, dtype=torch.int32)
    tile = functools.partial(np.tile, reps=reps)
    temps_r = torch.from_numpy(tile(np.asarray(temps, np.float32)))[:, None].to(
        torch.bfloat16
    )
    top_k_r = torch.from_numpy(tile(np.asarray(top_k, np.int32)))[:, None]
    top_p_r = torch.from_numpy(tile(np.asarray(top_p, np.float32)))[:, None]
    return types.SimpleNamespace(
        tp=tp,
        shard=shard,
        vocab=vocab,
        logits=logits,
        bonus_idx=bonus_idx,
        target_idx=target_idx,
        draft=draft,
        segment=segment,
        group=group,
        num_draft=num_draft,
        temps=temps_r,
        top_k=top_k_r,
        top_p=top_p_r,
        types=types_,
    )


def _program_out(c, u, synthetic=False, rates_table=None):
    """The runner's one-program candidate verify body
    (`_candidate_verify_body`) on chunk `c` and flat draw `u`, with the
    candidate op emulated on CPU: returns (bonus ids, rejection output)."""
    from vllm_torchtpu.runner import tpu_runner as tr

    width = 64 * c.tp
    seen = {}
    op = _emulated_op(
        c.logits[c.bonus_idx.long()],
        c.logits[c.target_idx.long()],
        c.tp,
        c.shard,
        synthetic,
    )

    def recording_op(*operands):
        seen["out"] = op(*operands)
        return seen["out"]

    out = tr._candidate_verify_body(
        recording_op,
        c.logits[:, : c.shard],  # rank 0's logits shard
        c.bonus_idx,
        c.target_idx,
        torch.tensor([0, c.shard], dtype=torch.int32),
        c.draft,
        c.temps,
        c.top_k,
        c.top_p,
        c.segment,
        c.group,
        rates_table,
        u,
        width,
        c.num_draft,
        K,
    )
    return seen["out"][0][0], out


def _gathered_out(c, gen, bonus_dtype=torch.bfloat16):
    """The gathered-logits verify on the full rows (`sample_from_logits` and
    `_random_rejection_sample_impl`) with its own full-vocabulary draws."""
    num_reqs = c.bonus_idx.shape[0]
    u_b = torch.rand((num_reqs, c.vocab), generator=gen).to(bonus_dtype)
    accept_u = torch.rand(c.draft.shape, generator=gen)
    u_r = torch.rand((c.target_idx.shape[0], c.vocab), generator=gen)
    bonus = _reference_bonus(
        c.logits[c.bonus_idx.long()], c.temps, u_b, c.top_k, c.top_p
    )
    seg = c.segment
    out = rs._random_rejection_sample_impl(
        c.draft,
        c.logits[c.target_idx.long()],
        c.num_draft,
        bonus,
        c.segment,
        c.group,
        K,
        c.temps[seg],
        c.top_k[seg],
        c.top_p[seg],
        accept_u,
        u_r,
        None,
        False,
    )
    return bonus, out


def _decompose(c, bonus, out):
    """Per request: request type, bonus id, accepted length (0..K) and the
    token recovered at the first rejected slot (-1 when all were accepted)."""
    num_reqs = c.bonus_idx.shape[0]
    draft = c.draft.view(num_reqs, K)
    match = (out[:, :K] == draft).int()
    accepted = match.cumprod(dim=1).sum(dim=1)
    rows = torch.arange(num_reqs)
    recovered = torch.where(
        accepted < K,
        out[rows, accepted.clamp(max=K - 1)],
        torch.full_like(accepted, -1),
    )
    req_type = rows % c.types
    return req_type, bonus.view(-1).long(), accepted, recovered


def _chi2_p(a, b, min_expected=5.0):
    """Two-sample chi-square homogeneity p-value of integer samples a, b;
    categories expected below `min_expected` in either sample are pooled."""
    from scipy.stats import chi2_contingency

    cats = np.union1d(a, b)
    table = np.stack([[np.sum(a == v) for v in cats], [np.sum(b == v) for v in cats]])
    expected = table.sum(0) * min(table.sum(1)) / table.sum()
    keep = expected >= min_expected
    pooled = table[:, ~keep].sum(1, keepdims=True)
    table = np.concatenate([table[:, keep], pooled], axis=1)
    table = table[:, table.sum(0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(chi2_contingency(table)[1])


def _histograms(c, draws, sampler):
    """`draws` verify chunks of `sampler` (seeded per draw) → per-type arrays
    of bonus ids, accepted lengths and first-rejection recovered ids."""
    acc = {k: [[] for _ in range(c.types)] for k in ("bonus", "accepted", "recovered")}
    for d in range(draws):
        req_type, bonus, accepted, recovered = _decompose(c, *sampler(d))
        for t in range(c.types):
            m = req_type == t
            acc["bonus"][t].append(bonus[m].numpy())
            acc["accepted"][t].append(accepted[m].numpy())
            first = m & (accepted == 0)  # recovered at slot 0: one target row
            acc["recovered"][t].append(recovered[first].numpy())
    return {k: [np.concatenate(v) for v in vals] for k, vals in acc.items()}


DIST_TEMPS = [0.8, 0.6, 1.0, 0.7]
DIST_TOP_K = [5, 20, 48, 1]
DIST_TOP_P = [0.9, 0.95, 1.0, 0.8]


@pytest.mark.cpu_test
def test_program_samples_the_gathered_distribution():
    """Flag on vs the gathered-logits verify: chi-square homogeneity of the
    bonus-token, accepted-length and recovered-token histograms, per request
    type (temperature > 0, top_k in [1, 48], top_p), over many seeds. A
    temperature raised by 0.4 must fail the same bound (the test can
    fail)."""
    from vllm_torchtpu.runner import tpu_runner as tr

    num_reqs, draws = 4 * 256, 8
    c = _chunk(0, num_reqs, temps=DIST_TEMPS, top_k=DIST_TOP_K, top_p=DIST_TOP_P)
    width = 64 * c.tp
    n = tr._candidate_uniforms_count(num_reqs, c.draft.numel(), num_reqs * K, width)

    def program(d):
        gen = torch.Generator().manual_seed(1000 + d)
        return _program_out(c, torch.rand(n, generator=gen))

    def gathered(d):
        return _gathered_out(c, torch.Generator().manual_seed(2000 + d))

    got = _histograms(c, draws, program)
    want = _histograms(c, draws, gathered)
    pvals = {
        (k, t): _chi2_p(got[k][t], want[k][t])
        for k in got
        for t in range(c.types)
        if not (k != "bonus" and DIST_TOP_K[t] == 1)  # top_k 1: one token
    }
    print("chi-square p-values (path, request type):", pvals)
    assert min(pvals.values()) > 1e-3, pvals
    # Each path draws every category it should: rejections happen.
    assert all(len(got["recovered"][t]) > 50 for t in range(3))

    hot = _chunk(
        0,
        num_reqs,
        temps=[t + 0.4 for t in DIST_TEMPS],
        top_k=DIST_TOP_K,
        top_p=DIST_TOP_P,
    )
    shifted = _histograms(
        hot,
        draws,
        lambda d: _program_out(
            hot, torch.rand(n, generator=torch.Generator().manual_seed(1000 + d))
        ),
    )
    control = {k: _chi2_p(shifted[k][0], want[k][0]) for k in ("bonus", "accepted")}
    print("negative control p-values (temperature + 0.4):", control)
    assert max(control.values()) < 1e-3, control


@pytest.mark.cpu_test
def test_program_greedy_rows_are_bitwise_the_gathered_ones():
    """Temperature-0 requests ignore the draws: their rows of the program's
    output equal the gathered verify's bitwise, whatever the uniforms."""
    from vllm_torchtpu.runner import tpu_runner as tr

    num_reqs = 8
    c = _chunk(
        3, num_reqs, temps=[0.0, 0.8, 0.0, 0.7], top_k=[20, 20, 0, 48], top_p=[1.0] * 4
    )
    width = 64 * c.tp
    n = tr._candidate_uniforms_count(num_reqs, c.draft.numel(), num_reqs * K, width)
    greedy = (c.temps[:, 0] == 0).nonzero().view(-1)
    for seed in range(3):
        bonus, out = _program_out(
            c, torch.rand(n, generator=torch.Generator().manual_seed(seed))
        )
        ref_bonus, ref_out = _gathered_out(c, torch.Generator().manual_seed(50 + seed))
        assert torch.equal(bonus.long()[greedy], ref_bonus.long()[greedy])
        assert torch.equal(out[greedy], ref_out[greedy])


@pytest.mark.cpu_test
@pytest.mark.parametrize("synthetic", [False, True])
def test_program_operands_are_the_separate_draws(synthetic):
    """`_candidate_op_operands` cuts the flat draw into the bonus (rounded to
    the logits dtype), accept, recover and key draws and gathers the rows and
    per-slot params as the separate programs do."""
    from vllm_torchtpu.layers.adapter.sample.candidate_sampling import NUM_OPERANDS
    from vllm_torchtpu.runner import tpu_runner as tr

    g = torch.Generator().manual_seed(4)
    num_reqs, shard, width = 4, 96, 128
    rows = num_reqs * (K + 1)
    logits = torch.randn(rows, shard, generator=g).to(torch.bfloat16)
    bonus_idx = torch.arange(num_reqs, dtype=torch.int32) * (K + 1) + K
    target_idx = (torch.arange(num_reqs)[:, None] * (K + 1) + torch.arange(K)).view(-1)
    draft = torch.randint(0, 2 * shard, (num_reqs * K,), generator=g, dtype=torch.int32)
    segment = torch.arange(num_reqs).repeat_interleave(K)
    group = torch.arange(K, dtype=torch.int32).repeat(num_reqs)
    temps = torch.rand(num_reqs, 1, generator=g).to(torch.bfloat16)
    top_k = torch.randint(1, 48, (num_reqs, 1), generator=g, dtype=torch.int32)
    top_p = torch.rand(num_reqs, 1, generator=g)
    bounds = torch.tensor([shard, 2 * shard], dtype=torch.int32)
    table = torch.rand(K, generator=g) if synthetic else None
    n = tr._candidate_uniforms_count(num_reqs, draft.numel(), target_idx.numel(), width)
    assert n == (num_reqs + num_reqs * K) * width + num_reqs * K + 2
    u = torch.rand(n, generator=g)
    got = tr._candidate_op_operands(
        logits, bonus_idx, target_idx, bounds, draft, temps, top_k, top_p,
        segment, group, table, u, width,
    )  # fmt: skip
    nb = num_reqs * width
    na = nb + draft.numel()
    nr = na + target_idx.numel() * width
    want = (
        logits[bonus_idx.long()],
        logits[target_idx],
        bounds,
        draft,
        temps,
        top_k,
        top_p,
        temps[segment],
        top_k[segment],
        top_p[segment],
        u[:nb].view(num_reqs, width).to(torch.bfloat16),
        u[nb:na],
        u[na:nr].view(-1, width),
        table[group.long()] if synthetic else u[nb:na],
        (u[nr:] * (1 << 24)).to(torch.int32),
    )
    assert len(got) == len(want) == NUM_OPERANDS
    for a, b in zip(got, want):
        assert a.shape == (1, *b.shape) and a.dtype == b.dtype
        assert torch.equal(a[0], b)

    decisions = (
        torch.randint(0, 2 * shard, (1, num_reqs), generator=g, dtype=torch.int32),
        torch.randint(0, 2, (1, draft.numel()), generator=g, dtype=torch.int32),
        torch.randint(0, 2 * shard, (1, draft.numel()), generator=g, dtype=torch.int32),
    )
    num_draft = torch.full((num_reqs,), K, dtype=torch.int32)
    out = tr._candidate_op_tokens(decisions, draft, num_draft, segment, group, K)
    ref = _eager(rs.candidate_rejection_output)(
        decisions[1][0], decisions[2][0], draft, num_draft, decisions[0][0],
        segment, group, K,
    )  # fmt: skip
    assert torch.equal(out, ref)


def _fake_runner(monkeypatch, *, program, compiling=False, warmed=(), fold=None):
    """A runner stub for `_sample_spec_verify_candidates` on CPU tensors:
    records the torch.rand calls and the separate programs it reaches."""
    from vllm_torchtpu.runner import tpu_runner as tr

    calls = {"rand": 0, "decisions": 0, "rejection": 0, "gather": 0, "program": []}
    real_rand = torch.rand

    def rand(*a, **kw):
        calls["rand"] += 1
        kw.pop("device", None)
        return real_rand(*a, **kw)

    monkeypatch.setattr(tr.torch, "rand", rand)

    def decisions(bonus_rows, target_rows, *a, **kw):
        calls["decisions"] += 1
        n_b, n_t = bonus_rows.shape[0], target_rows.shape[0]
        z = torch.zeros
        return z(n_b, dtype=torch.int32), z(n_t, dtype=torch.int32), z(n_t), z(())

    def rejection(*a):
        calls["rejection"] += 1
        return "separate"

    monkeypatch.setattr(tr, "candidate_verify_decisions", decisions)
    monkeypatch.setattr(tr, "candidate_rejection_output", rejection)

    def compiled(vocab, synthetic, fold_ro):
        def run(*args):
            calls["program"].append((fold_ro, len(args)))
            return "program"

        return run

    monkeypatch.setattr(tr, "_candidate_verify_program", compiled)
    num_reqs, rows, shard = 2, 2 * (K + 1), 64
    md = tr.SpecDecodeMetadata(
        draft_token_ids=torch.zeros(num_reqs * K, dtype=torch.int32),
        draft_lengths=torch.full((num_reqs,), K, dtype=torch.int32),
        draft_lengths_cpu=np.full(num_reqs, K, dtype=np.int32),
        target_logits_indices=torch.zeros(num_reqs * K, dtype=torch.int32),
        bonus_logits_indices=torch.zeros(num_reqs, dtype=torch.int32),
        final_logits_indices=torch.zeros(num_reqs * K, dtype=torch.int32),
        segment_ids=torch.zeros(num_reqs * K, dtype=torch.int64),
        group_indices=torch.zeros(num_reqs * K, dtype=torch.int32),
    )

    def gather(local, b, t):
        calls["gather"] += 1
        return local[: b.shape[0]], local[: t.shape[0]]

    runner = types.SimpleNamespace(
        rejection_sampler=types.SimpleNamespace(
            synthetic_mode=False, synthetic_conditional_rates=None
        ),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=K),
        _candidate_bounds=torch.tensor([0, shard], dtype=torch.int32),
        _candidate_head=types.SimpleNamespace(tp_size=2, org_vocab_size=2 * shard),
        _padded_sampling_params=lambda s, e, n, dt, dev, pad_top_k=0: (
            torch.ones(n, 1, dtype=dt),
            torch.full((n, 1), 20, dtype=torch.int32),
            torch.ones(n, 1),
        ),
        spec_gather_bonus_and_target_logits=gather,
        _num_reqs_tensor=lambda n: torch.tensor(n),
        _fused_read_offsets_applied=False,
        _cvp_compiling=compiling,
    )
    runner._candidate_verify_one_program = functools.partial(
        tr.TPUModelRunner._candidate_verify_one_program, runner
    )
    runner._candidate_ro_fold = lambda: fold
    local = torch.zeros(rows, shard, dtype=torch.bfloat16)
    runner._cvp_warmed = {
        tr._candidate_program_key(local, md, False, g) for g in warmed
    }
    return runner, md, local, calls


@pytest.mark.cpu_test
def test_flag_off_keeps_the_separate_programs(monkeypatch):
    """VLLM_TPU_CANDIDATE_VERIFY_PROGRAM off: four draws, the row gather,
    the separate op call and rejection output, as before."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setattr(envs, "VLLM_TPU_CANDIDATE_VERIFY_PROGRAM", False, raising=False)
    runner, md, local, calls = _fake_runner(monkeypatch, program=False, compiling=True)
    out = TPUModelRunner._sample_spec_verify_candidates(runner, local, md, 0, 2, None)
    assert out == "separate"
    assert calls["rand"] == 4 and calls["gather"] == 1
    assert calls["decisions"] == calls["rejection"] == 1 and not calls["program"]


@pytest.mark.cpu_test
@pytest.mark.parametrize("folded", [False, True])
def test_flag_on_is_one_draw_and_one_program(monkeypatch, folded):
    """Flag on at a warmed key: one torch.rand and one compiled program call,
    none of the separate programs; with the read-offset fold the program
    takes (buffer, groups, rows) first and marks the offsets applied."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setattr(envs, "VLLM_TPU_CANDIDATE_VERIFY_PROGRAM", True, raising=False)
    groups = [torch.zeros(4, dtype=torch.int32), torch.zeros(4, dtype=torch.int32)]
    fold = (torch.zeros(16, dtype=torch.int32), groups, 2) if folded else None
    runner, md, local, calls = _fake_runner(
        monkeypatch, program=True, warmed=[groups if folded else None], fold=fold
    )
    out = TPUModelRunner._sample_spec_verify_candidates(runner, local, md, 0, 2, None)
    assert out == "program"
    assert calls["rand"] == 1 and calls["program"] == [(folded, 15 + 3 * folded)]
    assert calls["decisions"] == calls["rejection"] == calls["gather"] == 0
    assert runner._fused_read_offsets_applied is folded


@pytest.mark.cpu_test
def test_flag_on_unwarmed_keys_fall_back_with_one_warning(monkeypatch, caplog):
    """Outside warmup an unwarmed fold key keeps the separate scatter (the
    plain program runs), and an unwarmed plain key keeps the separate
    programs; each key warns once."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner import tpu_runner as tr
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setattr(envs, "VLLM_TPU_CANDIDATE_VERIFY_PROGRAM", True, raising=False)
    warnings = []
    monkeypatch.setattr(tr.logger, "warning", lambda *a: warnings.append(a))
    groups = [torch.zeros(8, dtype=torch.int32)]
    fold = (torch.zeros(16, dtype=torch.int32), groups, 2)
    runner, md, local, calls = _fake_runner(
        monkeypatch, program=True, warmed=[None], fold=fold
    )
    for _ in range(3):
        out = TPUModelRunner._sample_spec_verify_candidates(
            runner, local, md, 0, 2, None
        )
        assert out == "program" and not runner._fused_read_offsets_applied
    assert calls["program"] == [(False, 15)] * 3 and len(warnings) == 1

    runner, md, local, calls = _fake_runner(monkeypatch, program=True, warmed=[])
    for _ in range(2):
        out = TPUModelRunner._sample_spec_verify_candidates(
            runner, local, md, 0, 2, None
        )
        assert out == "separate"
    assert not calls["program"] and calls["rand"] == 8 and len(warnings) == 2


@pytest.mark.cpu_test
def test_program_with_fold_leaves_the_separate_read_offsets(monkeypatch):
    """The fold program returns the verify output and leaves the read offsets
    exactly as `_update_read_offsets_impl` after it does."""
    from vllm_torchtpu.runner import tpu_runner as tr

    next_tokens = torch.tensor(
        [[5, 6, -1, -1], [7, -1, -1, -1], [1, 2, 3, 4]], dtype=torch.int32
    )
    monkeypatch.setattr(tr, "_tpu_compile", lambda fn: fn)
    monkeypatch.setattr(tr, "candidate_verify_op", lambda *a: None)
    monkeypatch.setattr(tr, "_candidate_verify_body", lambda op, *a: next_tokens)
    tr._candidate_verify_program.cache_clear()
    try:
        for groups in (
            [torch.tensor([3, 1, 0, 0], dtype=torch.int32),
             torch.tensor([6, 2, 0, 0], dtype=torch.int32)],
            [torch.tensor([[3, 1, 0, 0], [6, 2, 0, 0]], dtype=torch.int32)],
        ):  # fmt: skip
            got = torch.full((8,), 9, dtype=torch.int32)
            want = got.clone()
            num_reqs = torch.tensor(2)
            out = tr._candidate_verify_program(100, False, True)(
                got, groups, num_reqs, *([None] * 15)
            )
            assert out is next_tokens
            tr._update_read_offsets_impl(want, groups, next_tokens, num_reqs)
            assert torch.equal(got, want)
            assert got.tolist() == [0, 0, 0, 1, 9, 9, 1, 9]
    finally:
        tr._candidate_verify_program.cache_clear()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "flag,ro,update,on_tpu,groups,expected",
    [
        (True, False, True, True, True, False),
        (True, True, False, True, True, False),
        (True, True, True, False, True, False),
        (True, True, True, True, False, False),
        (True, True, True, True, True, True),
    ],
)
def test_chunk_folds_read_offsets_only_when_switched_on(
    monkeypatch, flag, ro, update, on_tpu, groups, expected
):
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner import tpu_runner as tr
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setattr(envs, "VLLM_TPU_CANDIDATE_VERIFY_PROGRAM", flag, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_CANDIDATE_VERIFY_RO", ro, raising=False)
    monkeypatch.setattr(
        envs, "VLLM_TPU_FUSED_READ_OFFSET_UPDATE", update, raising=False
    )
    monkeypatch.setattr(tr, "_on_tpu", lambda t: on_tpu)
    buffer = torch.zeros(8, dtype=torch.int32)
    slot_groups = [torch.zeros(4, dtype=torch.int32)] if groups else None
    runner = types.SimpleNamespace(
        mamba_slot_read_offsets=buffer, _fused_read_offsets=(slot_groups, 2)
    )
    got = TPUModelRunner._candidate_ro_fold(runner)
    if expected:
        assert got[0] is buffer and got[1] == slot_groups and got[2] == 2
    else:
        assert got is None


@pytest.mark.cpu_test
def test_ro_layouts_recorded_until_warmed_then_compiled_at_every_width(monkeypatch):
    """The fold warmup compiles every recorded slot-group layout (aliases
    kept) at every width a chunk can have, for every warmed bucket, and
    stops recording layouts."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner import tpu_runner as tr
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    for name in (
        "VLLM_TPU_CANDIDATE_VERIFY_PROGRAM",
        "VLLM_TPU_CANDIDATE_VERIFY_RO",
        "VLLM_TPU_FUSED_READ_OFFSET_UPDATE",
    ):
        monkeypatch.setattr(envs, name, True, raising=False)
    monkeypatch.setattr(tr, "_on_tpu", lambda t: True)
    monkeypatch.setattr(tr, "synchronize_tensors", lambda *a: None)
    shared = torch.zeros(16, dtype=torch.int32)
    runner = types.SimpleNamespace(
        _candidate_bounds=torch.tensor([0, 64]),
        _candidate_head=types.SimpleNamespace(num_embeddings_per_partition=64),
        mamba_slot_read_offsets=torch.zeros(32, dtype=torch.int32),
        num_reqs_max_model_len=16,
        num_reqs_most_model_len=8,
        _hidden_states_dtype=torch.bfloat16,
        device="cpu",
        _get_sampling_generator=lambda: None,
        _precompile_timed=lambda what: __import__("contextlib").nullcontext(),
    )
    TPUModelRunner._record_candidate_ro_layout(runner, [shared, shared])
    TPUModelRunner._record_candidate_ro_layout(
        runner, [torch.zeros(4, dtype=torch.int32), torch.zeros(4, dtype=torch.int32)]
    )
    shapes = ((2,), torch.int32), ((2,), torch.int32), ((6,), torch.int32)
    shapes += ((2,), torch.int32), ((6,), torch.int64), ((6,), torch.int32)
    runner._cvro_buckets = [(8, 2, *shapes), (8, 2, *shapes)]
    seen = []

    def verify(local, md, s, e, gen, ro_fold):
        assert runner._cvp_compiling
        buf, groups, rows = ro_fold
        seen.append((local.shape, len(groups), groups[0] is groups[1],
                     tuple(groups[0].shape)))  # fmt: skip

    runner._sample_spec_verify_candidates = verify
    TPUModelRunner._precompile_candidate_verify_ro(runner)
    assert not runner._cvp_compiling
    # One bucket (deduplicated) x 2 layouts (aliased / distinct) x widths
    # {4, 8, 16}.
    assert sorted(seen, key=repr) == sorted(
        [((8, 64), 2, alias, (w,)) for alias in (True, False) for w in (4, 8, 16)],
        key=repr,
    )
    TPUModelRunner._record_candidate_ro_layout(
        runner, [torch.zeros(5, dtype=torch.int32)]
    )
    assert 5 not in runner._cvro_widths


@pytest.mark.cpu_test
@pytest.mark.parametrize("flag", [False, True])
def test_candidate_sampling_params_memo_keys_pad_top_k(monkeypatch, flag):
    """With the flag the candidate chunk's padded params (pad_top_k = 1) are
    memoized under their own key; without it they are rebuilt every call,
    as before. Either way the values are the uncached ones."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setattr(envs, "VLLM_TPU_CANDIDATE_VERIFY_PROGRAM", flag, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_SAMPLING_PARAMS_MEMO", True, raising=False)
    built = []

    def uncached(self, s, e, n, dtype, device, pad_top_k=0):
        built.append(pad_top_k)
        return ("params", pad_top_k)

    monkeypatch.setattr(TPUModelRunner, "_padded_sampling_params_uncached", uncached)
    ib = types.SimpleNamespace(
        temperature_cpu=np.full(4, 0.7, np.float32),
        top_k_cpu=np.full(4, 20, np.int32),
        top_p_cpu=np.full(4, 0.9, np.float32),
    )
    runner = types.SimpleNamespace(input_batch=ib)
    call = functools.partial(
        TPUModelRunner._padded_sampling_params, runner, 0, 3, 4, torch.bfloat16, "cpu"
    )
    for _ in range(3):
        assert call(pad_top_k=1) == ("params", 1)
        assert call() == ("params", 0)
    assert built == ([1, 0] if flag else [1, 0, 1, 1])


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "name", ["VLLM_TPU_CANDIDATE_VERIFY_PROGRAM", "VLLM_TPU_CANDIDATE_VERIFY_RO"]
)
def test_candidate_verify_program_flags_default_off(monkeypatch, name):
    from vllm_torchtpu import envs

    monkeypatch.delenv(name, raising=False)
    assert envs.environment_variables[name]() is False
    monkeypatch.setenv(name, "1")
    assert envs.environment_variables[name]() is True
