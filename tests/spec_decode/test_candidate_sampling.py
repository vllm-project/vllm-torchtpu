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
@pytest.mark.parametrize("kind", ["uniform", "ties", "extremes"])
def test_select_top_picks_the_k_largest_keys_in_position_order(kind):
    """The digit search finds the k-th largest key, and the one-hot holds the
    k largest keys (ties to the lowest positions) in position order."""
    rng = np.random.default_rng(3)
    for n in (7, 243, 1024):
        if kind == "uniform":
            keys = rng.integers(-(1 << 15), 1 << 15, (4, n))
        elif kind == "ties":
            keys = rng.integers(-3, 3, (4, n))
        else:
            keys = rng.choice([-(1 << 15), (1 << 15) - 1, 0, -1], (4, n))
        for k in sorted({1, min(64, n), n}):
            got = np.asarray(cs._kth_largest_key(jnp.asarray(keys, jnp.int32), k))
            np.testing.assert_array_equal(got, -np.sort(-keys, axis=1)[:, k - 1])
            onehot = np.asarray(cs._select_top(jnp.asarray(keys, jnp.int32), k))
            for r in range(keys.shape[0]):
                chosen = np.sort(np.argsort(-keys[r], kind="stable")[:k])
                want = np.zeros((k, n), bool)
                want[np.arange(k), chosen] = True
                np.testing.assert_array_equal(onehot[r], want)


@pytest.mark.cpu_test
def test_gather_candidates_matches_one_gather_per_array():
    tp, rows, kc = 4, 5, 8
    rng = np.random.default_rng(5)
    vals = jnp.asarray(rng.normal(size=(tp, rows, kc)), jnp.bfloat16)
    vals = vals.at[0, 0, 0].set(-0.0)  # a bit pattern a float compare would miss
    ids = jnp.asarray(rng.integers(0, 1 << 20, (tp, rows, kc)), jnp.int32)
    boundary = jnp.asarray(rng.normal(size=(tp, rows)), jnp.bfloat16)
    top = jnp.asarray(rng.normal(size=(tp, rows)), jnp.bfloat16)
    first = jnp.asarray(rng.integers(0, 1 << 20, (tp, rows)), jnp.int32)

    def separate(v, i, b, t, f):
        def gather(a):
            return jax.lax.all_gather(a, "tp", axis=1, tiled=True)

        return (
            gather(v),
            gather(i),
            gather(b[:, None]),
            gather(t[:, None]),
            gather(f[:, None]),
        )

    def packed(v, i, b, t, f):
        return cs._gather_candidates(v, i, b, t, f, "tp")

    args = (vals, ids, boundary, top, first)
    want = jax.vmap(separate, axis_name="tp")(*args)
    got = jax.vmap(packed, axis_name="tp")(*args)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape
        if g.dtype == jnp.bfloat16:
            g, w = (jax.lax.bitcast_convert_type(a, jnp.int16) for a in (g, w))
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


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
def test_candidates_only_when_every_request_keeps_few_tokens(change, expected):
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
            temperature_cpu=temps.numpy(),  # InputBatch's numpy views
            top_k_cpu=top_k.numpy(),
        ),
    )
    scheduler_output = types.SimpleNamespace(
        has_structured_output_requests=change == "structured"
    )
    assert TPUModelRunner._samples_candidates(runner, scheduler_output) is expected


@pytest.mark.parametrize("synthetic", [False, True])
def test_runner_operands_and_output_match_separate_ops(synthetic):
    """The runner's candidate verify program builds the op's operands and the
    rejection output (_candidate_verify_operands / _candidate_verify_tokens)
    exactly as the separate draws, gathers and candidate_rejection_output do,
    from the same flat uniform draw."""
    from vllm_torchtpu.layers.adapter.sample.candidate_sampling import NUM_OPERANDS
    from vllm_torchtpu.runner import tpu_runner as tr

    g = torch.Generator().manual_seed(0)
    num_reqs, k, shard, width = 4, 3, 96, 64 * 2
    rows = num_reqs * (k + 1)
    logits = torch.randn(rows, shard, generator=g).to(torch.bfloat16)
    bonus_idx = torch.arange(num_reqs) * (k + 1) + k
    target_idx = (torch.arange(num_reqs)[:, None] * (k + 1) + torch.arange(k)).flatten()
    draft = torch.randint(0, 2 * shard, (num_reqs * k,), generator=g, dtype=torch.int32)
    segment = torch.arange(num_reqs).repeat_interleave(k)
    group = torch.arange(k, dtype=torch.int32).repeat(num_reqs)
    temps = torch.rand(num_reqs, 1, generator=g).to(torch.bfloat16)
    top_k = torch.randint(1, 48, (num_reqs, 1), generator=g, dtype=torch.int32)
    top_p = torch.rand(num_reqs, 1, generator=g)
    bounds = torch.tensor([shard, 2 * shard], dtype=torch.int32)
    rates_table = torch.rand(k, generator=g) if synthetic else None
    n = tr._candidate_uniforms_count(num_reqs, draft.numel(), target_idx.numel(), width)
    u = torch.rand(n, generator=g)

    ops = tr._candidate_verify_operands(
        logits,
        bonus_idx,
        target_idx,
        bounds,
        draft,
        temps,
        top_k,
        top_p,
        segment,
        group,
        rates_table,
        u,
        width,
    )
    nb, na = num_reqs * width, num_reqs * width + draft.numel()
    nr = na + target_idx.numel() * width
    accept_u = u[nb:na]
    expected = (
        logits[bonus_idx],
        logits[target_idx],
        bounds,
        draft,
        temps,
        top_k,
        top_p,
        temps[segment],
        top_k[segment],
        top_p[segment],
        u[:nb].reshape(num_reqs, width).to(torch.bfloat16),
        accept_u,
        u[na:nr].reshape(-1, width),
        rates_table[group.long()] if synthetic else accept_u,
        (u[nr:] * (1 << 24)).to(torch.int32),
    )
    assert len(ops) == len(expected) == NUM_OPERANDS
    for got, want in zip(ops, expected):
        assert got.shape == (1, *want.shape) and got.dtype == want.dtype
        assert torch.equal(got[0], want)

    accepted = torch.randint(0, 2, (1, draft.numel()), generator=g, dtype=torch.int32)
    recovered = torch.randint(
        0, 2 * shard, (1, draft.numel()), generator=g, dtype=torch.int32
    )
    bonus = torch.randint(0, 2 * shard, (1, num_reqs), generator=g, dtype=torch.int32)
    num_draft = torch.full((num_reqs,), k, dtype=torch.int32)
    out = tr._candidate_verify_tokens(
        (bonus, accepted, recovered), draft, num_draft, segment, group, k
    )
    ref = _eager(rs.candidate_rejection_output)(
        accepted[0], recovered[0], draft, num_draft, bonus[0], segment, group, k
    )
    assert torch.equal(out, ref)


def test_candidate_verify_fused_program_folds_neighbours(monkeypatch):
    """The fused program runs the verify on the LM head projection of
    `hidden[indices]` (VLLM_TPU_CANDIDATE_VERIFY_LOGITS) and leaves the read
    offsets as `_update_read_offsets_impl` does (VLLM_TPU_CANDIDATE_VERIFY_RO)."""
    from vllm_torchtpu.runner import tpu_runner as tr

    next_tokens = torch.tensor(
        [[5, 6, -1, -1], [7, -1, -1, -1], [1, 2, 3, 4]], dtype=torch.int32
    )
    seen = []

    def tokens(outs, *rest):
        return next_tokens

    monkeypatch.setattr(tr, "_tpu_compile", lambda fn: fn)
    monkeypatch.setattr(tr, "candidate_verify_op", lambda *a: lambda *ops: ops)
    monkeypatch.setattr(
        tr, "_candidate_verify_operands", lambda logits, *a: seen.append(logits) or ()
    )
    monkeypatch.setattr(tr, "_candidate_verify_tokens", tokens)
    tr._candidate_verify_fn.cache_clear()
    tr._candidate_verify_fused_program.cache_clear()
    try:
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        head = types.SimpleNamespace(
            quant_method=types.SimpleNamespace(
                apply=lambda layer, x, bias=None: x @ weight.T
            )
        )
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        indices = torch.tensor([4, 0, 2], dtype=torch.int32)
        groups = [
            torch.tensor([3, 1, 0, 0], dtype=torch.int32),
            torch.tensor([6, 2, 0, 0], dtype=torch.int32),
        ]
        got = torch.full((8,), 9, dtype=torch.int32)
        want = got.clone()
        num_reqs = torch.tensor(2)
        args = (None,) * 13 + (2, 3)
        program = tr._candidate_verify_fused_program(100, False)
        no_seed = (None,) * 6
        out = program(got, groups, num_reqs, head, hidden, indices, *no_seed, *args)
        assert out is next_tokens
        assert torch.equal(seen[-1], hidden[indices.long()] @ weight.T)
        tr._update_read_offsets_impl(want, groups, next_tokens, num_reqs)
        assert torch.equal(got, want)
        assert got.tolist() == [0, 0, 0, 1, 9, 9, 1, 9]
        local = torch.ones(3, 3)
        none3 = (None,) * 3
        assert program(*none3, *none3, *no_seed, local, *args[1:]) is next_tokens
        assert seen[-1] is local
        # VLLM_TPU_CANDIDATE_VERIFY_SEED: the drafter's fused seed on the
        # verify output, reading the verify's draft lengths.
        from vllm_torchtpu.spec_decode.eagle3 import _async_seed_inputs_impl

        draft_lengths = torch.tensor([3, 3, 3], dtype=torch.int32)
        seed = (
            torch.tensor([0, 4, 8, 12], dtype=torch.int32),
            torch.tensor(2),
            torch.arange(12, dtype=torch.int32),
            4,
            0,
            None,
        )
        verify_args = (local, *args[1:5], draft_lengths, *args[6:])
        out = program(*none3, *none3, *seed, *verify_args)
        want_seed = _async_seed_inputs_impl(next_tokens, draft_lengths, *seed)
        assert out[0] is next_tokens and len(out) == 1 + len(want_seed)
        for a, b in zip(out[1:], want_seed):
            assert torch.equal(a, b)
    finally:
        tr._candidate_verify_fn.cache_clear()
        tr._candidate_verify_fused_program.cache_clear()


def test_candidate_chunk_folds_read_offsets_only_when_switched_on(monkeypatch):
    """The verify chunk hands the read offsets to the candidate program (and
    marks them applied) only with VLLM_TPU_CANDIDATE_VERIFY_RO, a device
    offset buffer and slot groups; otherwise the separate scatter runs."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter.sample.candidate_sampling import (
        VocabShardLogits,
    )
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    tpu_buffer = types.SimpleNamespace(device=types.SimpleNamespace(type="tpu"))
    cpu_buffer = types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"))
    passed = []
    md = types.SimpleNamespace(draft_token_ids=object())
    groups_ = [torch.zeros(4, dtype=torch.int32)]
    cases = [
        (False, tpu_buffer, groups_, None),
        (True, cpu_buffer, groups_, None),
        (True, tpu_buffer, None, None),
        (True, tpu_buffer, groups_, (tpu_buffer, groups_, 2)),
    ]
    monkeypatch.setattr(envs, "VLLM_TPU_FUSED_READ_OFFSET_UPDATE", True, raising=False)
    for flag, buffer, groups, expected in cases:
        monkeypatch.setattr(envs, "VLLM_TPU_CANDIDATE_VERIFY_RO", flag, raising=False)
        runner = types.SimpleNamespace(
            mamba_slot_read_offsets=buffer,
            _fused_read_offsets=(groups, 2),
            _fused_read_offsets_applied=False,
        )
        runner._candidate_read_offsets = functools.partial(
            TPUModelRunner._candidate_read_offsets, runner
        )
        runner._sample_spec_verify_candidates = (
            lambda *a: passed.append(a[5]) or "tokens"
        )
        out = TPUModelRunner._sample_spec_verify_chunk(
            runner, VocabShardLogits(local="L"), md, None, None, 0, 2, False, None
        )
        assert out == "tokens"
        assert passed[-1] == expected
        assert runner._fused_read_offsets_applied is (expected is not None)


def test_candidate_chunk_hands_hidden_rows_when_logits_deferred():
    """A VocabShardLogits without `local` reaches the verify as
    (hidden, indices) for the in-program projection."""
    from vllm_torchtpu.layers.adapter.sample.candidate_sampling import (
        VocabShardLogits,
    )
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    passed = []
    runner = types.SimpleNamespace(
        _candidate_read_offsets=lambda: None, _candidate_seed_args=lambda *a: None
    )
    runner._sample_spec_verify_candidates = lambda *a: passed.append(a) or "tokens"
    md = types.SimpleNamespace(draft_token_ids=object())
    hidden, indices = object(), object()
    deferred = VocabShardLogits(None, hidden, indices)
    TPUModelRunner._sample_spec_verify_chunk(
        runner, deferred, md, None, None, 0, 2, False, None
    )
    assert passed[-1][0] is None and passed[-1][6] == (hidden, indices)
    TPUModelRunner._sample_spec_verify_chunk(
        runner, VocabShardLogits("L"), md, None, None, 0, 2, False, None
    )
    assert passed[-1][0] == "L" and passed[-1][6] is None


@pytest.mark.parametrize(
    "flag, rows, device, expected",
    [
        (False, 8, "tpu", False),
        (True, 8, "tpu", True),
        (True, 8192, "tpu", False),
        (True, 8, "cpu", False),
    ],
)
def test_verify_projects_logits_only_on_verify_shaped_chunks(
    monkeypatch, flag, rows, device, expected
):
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setattr(envs, "VLLM_TPU_CANDIDATE_VERIFY_LOGITS", flag, raising=False)
    hidden = types.SimpleNamespace(
        shape=(rows, 2560), device=types.SimpleNamespace(type=device)
    )
    indices = types.SimpleNamespace(shape=(8,))
    assert TPUModelRunner._verify_projects_logits(hidden, indices) is expected


@pytest.mark.tpu_test
@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_tpu_projection_in_verify_program_is_bitwise(rows):
    """VLLM_TPU_CANDIDATE_VERIFY_LOGITS: the LM head projection compiled into
    the program that builds the candidate operands gives the operands the
    separate projection program gives, bit for bit (served shard shapes:
    2560 hidden, 248320 / 8 vocab rows, bf16)."""
    import torch_tpu  # noqa: F401

    from vllm_torchtpu.runner import tpu_runner as tr

    dev = torch.device("tpu")
    g = torch.Generator().manual_seed(rows)
    k, width = 3, tr.candidate_width(8)
    num_reqs = rows // (k + 1)
    hidden = (torch.randn(rows, 2560, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    weight = (torch.randn(31040, 2560, generator=g) * 0.02).to(torch.bfloat16).to(dev)
    indices = torch.randperm(rows, generator=g).to(torch.int32).to(dev)
    bonus_idx = (torch.arange(num_reqs) * (k + 1) + k).to(torch.int32).to(dev)
    target_idx = (
        (torch.arange(num_reqs)[:, None] * (k + 1) + torch.arange(k))
        .flatten()
        .to(torch.int32)
        .to(dev)
    )
    draft = torch.randint(0, 62080, (num_reqs * k,), generator=g, dtype=torch.int32)
    segment = torch.arange(num_reqs).repeat_interleave(k)
    group = torch.arange(k, dtype=torch.int32).repeat(num_reqs)
    temps = torch.rand(num_reqs, 1, generator=g).to(torch.bfloat16)
    top_k = torch.randint(1, 48, (num_reqs, 1), generator=g, dtype=torch.int32)
    top_p = torch.rand(num_reqs, 1, generator=g)
    bounds = torch.tensor([31040, 62080], dtype=torch.int32)
    n = tr._candidate_uniforms_count(num_reqs, draft.numel(), target_idx.numel(), width)
    u = torch.rand(n, generator=g)
    rest = [t.to(dev) for t in (bounds, draft, temps, top_k, top_p, segment, group)]
    u = u.to(dev)

    def operands(local):
        return tr._candidate_verify_operands(
            local, bonus_idx, target_idx, *rest, None, u, width
        )

    def project(hidden, indices, weight):
        selected = torch.index_select(hidden, 0, indices)
        return torch.nn.functional.linear(selected, weight)

    compile_ = functools.partial(
        torch.compile, backend="tpu", fullgraph=True, dynamic=False
    )
    separate = compile_(operands)(compile_(project)(hidden, indices, weight))
    fused = compile_(lambda h, i, w: operands(project(h, i, w)))(
        hidden, indices, weight
    )
    for a, b in zip(separate, fused):
        assert a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())


def test_candidate_chunk_stashes_the_folded_seed_for_the_drafter(monkeypatch):
    """With seed arguments the verify returns (tokens, *seed outputs); the
    chunk returns the tokens and hands the seed to the drafter for the chunk
    whose draft inputs sample_tokens named."""
    from vllm_torchtpu.layers.adapter.sample.candidate_sampling import (
        VocabShardLogits,
    )
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    stashed = []
    chunk = object()
    runner = types.SimpleNamespace(
        _candidate_read_offsets=lambda: None,
        _candidate_seed_args=lambda md, so: ("seed args",),
        _verify_seed_chunk=chunk,
        _async_drafter=types.SimpleNamespace(
            stash_verify_seed=lambda *a: stashed.append(a)
        ),
    )
    runner._sample_spec_verify_candidates = lambda *a: ("tokens", "ids", "gather")
    md = types.SimpleNamespace(draft_token_ids=object())
    out = TPUModelRunner._sample_spec_verify_chunk(
        runner,
        VocabShardLogits(None, "H", "I"),
        md,
        None,
        None,
        0,
        2,
        False,
        None,
    )
    assert out == "tokens"
    assert stashed == [(chunk, "tokens", ("ids", "gather"))]


@pytest.mark.parametrize(
    "fused, partial_flag, discard, expected",
    [
        (False, False, [], None),
        (True, False, [], ("args", "lengths", None)),
        (True, False, [1], None),
        (True, False, [5], ("args", "lengths", None)),
        (True, True, [1], ("args", "lengths", ("override", (0,)))),
    ],
)
def test_verify_seed_args_follow_the_drafter_seed_rules(
    monkeypatch, fused, partial_flag, discard, expected
):
    """The drafter hands the verify the fused seed arguments only where
    `_prepare_draft_inputs` takes the fused seed: both seed flags on, and no
    partial-prefill row in the chunk unless VLLM_TPU_FUSED_DRAFT_SEED_PARTIAL
    supplies the override."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    monkeypatch.setattr(envs, "VLLM_TPU_FUSED_ASYNC_DRAFT_SEED", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_SEED_FUSED", fused, raising=False)
    monkeypatch.setattr(
        envs, "VLLM_TPU_FUSED_DRAFT_SEED_PARTIAL", partial_flag, raising=False
    )
    proposer = types.SimpleNamespace(
        runner=types.SimpleNamespace(
            input_batch=types.SimpleNamespace(req_ids=["a", "b", "c", "d"])
        ),
        _fused_seed_args=lambda chunk, num_draft, override: (
            "args",
            num_draft,
            override,
        ),
        _seed_override=lambda ov_i, req_ids, so, width: ("override", tuple(ov_i)),
    )
    chunk = types.SimpleNamespace(draft_lengths="lengths", num_reqs=2, start_index=1)
    got = Eagle3Proposer.verify_seed_args(proposer, chunk, 2, discard, None)
    assert got == expected


def test_verify_seed_stash_keeps_only_this_steps_chunks():
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    old, cur = object(), object()
    proposer = types.SimpleNamespace(
        _verify_seeds={id(old): (old, "nt0", ())}, draft_chunks=[cur]
    )
    Eagle3Proposer.stash_verify_seed(proposer, cur, "nt1", ("seed",))
    assert proposer._verify_seeds == {id(cur): (cur, "nt1", ("seed",))}


@pytest.mark.parametrize("stacked", [False, True])
def test_ro_slot_group_shapes_cover_each_capacity(monkeypatch, stacked):
    """The read-offset fold is warmed at one [capacity] slot group per mamba
    KV group (deduplicated), for both request capacities, plus any recorded
    set; the stacked walk hands stacks instead, so only recorded sets."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setattr(envs, "VLLM_TPU_STAGED_WALK_STACKED", stacked, raising=False)

    def group(gid, mamba):
        builder = types.SimpleNamespace(kv_cache_group_id=gid, is_mamba_group=mamba)
        return types.SimpleNamespace(metadata_builders=[builder])

    runner = types.SimpleNamespace(
        _unified_kv_layout=True,
        attn_groups=[
            [group(0, False)],
            [group(1, True), group(1, True)],
            [group(2, True)],
        ],
        num_reqs_max_model_len=16,
        num_reqs_most_model_len=8,
        _ro_group_shapes={16: {((16,), (16,), (16,))}},
    )
    shapes = TPUModelRunner._ro_slot_group_shapes(runner)
    expected = {((16,), (16,), (16,))}
    if not stacked:
        expected |= {((16,), (16,)), ((8,), (8,))}
    assert shapes == expected
    runner.attn_groups = [[group(0, False)]]
    runner._ro_group_shapes = {}
    assert TPUModelRunner._ro_slot_group_shapes(runner) == set()
