# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_VERIFY_SHARD_TOPK: the spec verify on vocabulary-shard candidates
against the full-logits fused verify, on fake tensor-parallel shards whose
all-gather is played back from every rank."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.model_executor.layers import logits_processor as lp_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    MASKED_LOGIT_VALUE,
    apply_top_k_top_p,
)
from vllm_torchtpu.runner import tpu_runner, vocab_shard_verify

VOCAB = 16384
TP = 8
WIDTH = 32
K = 3


class _HeadShard(torch.nn.Module):
    """One rank's slice of a vocab-parallel LM head (rows [start, start+n))."""

    def __init__(self, weight, start, tp_size):
        super().__init__()
        self.weight = weight
        self.tp_size = tp_size
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start, num_org_vocab_padding=0
        )
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias=None: x @ layer.weight.T
        )


def _owner(head):
    with patch.object(
        lp_module,
        "get_current_vllm_config",
        return_value=SimpleNamespace(model_config=None),
    ):
        processor = LogitsProcessor(VOCAB)
    return SimpleNamespace(lm_head=head, logits_processor=processor)


def _candidates_by_play_back(weight, hidden_states, monkeypatch, width=WIDTH):
    """shard_candidates on every rank: the all-gather answers with every
    rank's pairs. Returns rank 0's result and asserts all ranks agree."""
    per_rank = weight.shape[0] // TP
    shards = [
        _HeadShard(weight[r * per_rank : (r + 1) * per_rank], r * per_rank, TP)
        for r in range(TP)
    ]
    pairs = []

    def record(pair, dim):
        pairs.append(pair)
        return pair.repeat(1, TP)

    monkeypatch.setattr(vocab_shard_verify, "tensor_model_parallel_all_gather", record)
    for shard in shards:
        vocab_shard_verify.shard_candidates(_owner(shard), hidden_states, width)
    monkeypatch.setattr(
        vocab_shard_verify,
        "tensor_model_parallel_all_gather",
        lambda pair, dim: torch.cat(pairs, dim=-1),
    )
    results = [
        vocab_shard_verify.shard_candidates(_owner(s), hidden_states, width)
        for s in shards
    ]
    for r in results[1:]:
        assert torch.equal(r, results[0])
    return results[0]


def _logits_and_candidates(monkeypatch, rows, seed, width=WIDTH, weight=None):
    gen = torch.Generator().manual_seed(seed)
    hidden = 64
    if weight is None:
        weight = torch.randn(VOCAB, hidden, generator=gen).to(torch.bfloat16)
    hidden_states = (torch.randn(rows, hidden, generator=gen) * 0.4).to(torch.bfloat16)
    logits = hidden_states @ weight.T
    packed = _candidates_by_play_back(weight, hidden_states, monkeypatch, width)
    return logits, packed


def test_candidates_are_each_rows_largest_logits(monkeypatch):
    logits, packed = _logits_and_candidates(monkeypatch, rows=24, seed=0)
    values, ids = vocab_shard_verify.merge(packed, WIDTH)
    want = torch.topk(logits.float(), WIDTH, dim=-1, sorted=True).values
    assert torch.equal(values, want)
    assert torch.equal(torch.gather(logits.float(), 1, ids), values)
    assert torch.equal(
        vocab_shard_verify.candidate_argmax(values, ids),
        torch.argmax(logits, dim=-1).to(torch.int32),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_k,top_p", [(20, 0.95), (20, 1.0), (5, 0.5), (31, 0.99)])
def test_keep_mask_matches_full_row_masks(monkeypatch, dtype, top_k, top_p):
    logits, packed = _logits_and_candidates(monkeypatch, rows=40, seed=1)
    values, ids = vocab_shard_verify.merge(packed, WIDTH)
    k = torch.full((40, 1), top_k, dtype=torch.int32)
    p = torch.full((40, 1), top_p, dtype=torch.float32)
    dense = apply_top_k_top_p(logits.to(dtype), k, p, 64) > MASKED_LOGIT_VALUE
    kept = vocab_shard_verify.keep_mask(values.to(dtype), k, p)
    scattered = torch.zeros_like(dense).scatter(1, ids, kept)
    flags = vocab_shard_verify.tie_flags(values, k, torch.ones(40, dtype=torch.bool))
    mismatch = (scattered != dense).any(dim=-1)
    # Only a row whose bf16 ties run past the candidates may differ, and every
    # such row is flagged. With top_k 20 of 32 none of these rows are.
    assert not (mismatch & ~flags).any()
    assert flags.any() == (top_k == 31)


def _spec_batch(num_reqs, padded_num_reqs, padded_len, draft_lengths):
    """get_spec_decode_metadata's padded indices for `draft_lengths`."""
    sampled = [d + 1 for d in draft_lengths]
    starts = [sum(sampled[:i]) for i in range(num_reqs)]
    bonus = torch.zeros(padded_num_reqs, dtype=torch.int32)
    bonus[:num_reqs] = torch.tensor([s + n - 1 for s, n in zip(starts, sampled)])
    target = [s + j for s, d in zip(starts, draft_lengths) for j in range(d)]
    seg = [i for i, d in enumerate(draft_lengths) for _ in range(d)]
    grp = [j for d in draft_lengths for j in range(d)]
    t = torch.zeros(padded_len, dtype=torch.int32)
    t[: len(target)] = torch.tensor(target, dtype=torch.int32)
    segment_ids = torch.full((padded_len,), num_reqs, dtype=torch.int64)
    segment_ids[: len(seg)] = torch.tensor(seg)
    group_indices = torch.zeros(padded_len, dtype=torch.int32)
    group_indices[: len(grp)] = torch.tensor(grp, dtype=torch.int32)
    lengths = torch.zeros(padded_num_reqs, dtype=torch.int32)
    lengths[:num_reqs] = torch.tensor(draft_lengths, dtype=torch.int32)
    return bonus, t, segment_ids, group_indices, lengths


def _params(padded_num_reqs, rows):
    """`rows`: (temperature, top_k, top_p) per real request; padding as
    _padded_sampling_params pads (1.0, 0, 1.0)."""
    temps = torch.ones(padded_num_reqs, 1, dtype=torch.bfloat16)
    top_k = torch.zeros(padded_num_reqs, 1, dtype=torch.int32)
    top_p = torch.ones(padded_num_reqs, 1, dtype=torch.float32)
    for i, (t, k, p) in enumerate(rows):
        temps[i] = t
        top_k[i] = k
        top_p[i] = p
    return temps, top_k, top_p


def _dense_u(u_cand, ids, rows):
    dense = torch.full((rows, VOCAB), 0.5)
    return dense.scatter(1, ids, u_cand)


@pytest.mark.parametrize("synthetic", [True, False])
@pytest.mark.parametrize("seed", [2, 3, 4])
def test_candidate_verify_matches_fused_verify(monkeypatch, synthetic, seed):
    num_reqs, padded_num_reqs, padded_len = 6, 8, 32
    draft_lengths = [3, 3, 3, 0, 3, 2]
    logits, packed = _logits_and_candidates(monkeypatch, rows=padded_len, seed=seed)
    bonus_idx, target_idx, seg, grp, lengths = _spec_batch(
        num_reqs, padded_num_reqs, padded_len, draft_lengths
    )
    temps, top_k, top_p = _params(
        padded_num_reqs,
        [
            (1.0, 20, 0.95),
            (1.0, 20, 0.95),
            (0.0, 0, 1.0),
            (1.0, 20, 0.95),
            (0.7, 10, 0.8),
            (1.0, 20, 0.95),
        ],
    )
    gen = torch.Generator().manual_seed(100 + seed)
    # Drafts: the target argmax for some slots, a candidate, or a far token.
    argmax = torch.argmax(logits, dim=-1).to(torch.int32)
    draft = torch.randint(0, VOCAB, (padded_len,), generator=gen, dtype=torch.int32)
    draft[::3] = argmax[torch.clamp(target_idx[::3].long() + 1, max=padded_len - 1)]
    _, ids = vocab_shard_verify.merge(packed, WIDTH)
    draft[1::3] = ids[target_idx[1::3].long(), 3].to(torch.int32)
    bonus_uc = torch.rand(padded_num_reqs, WIDTH, generator=gen)
    accept_u = torch.rand(padded_len, generator=gen)
    recover_uc = torch.rand(padded_len, WIDTH, generator=gen)
    rates = (
        torch.tensor(unconditional_to_conditional_rates([0.8, 0.5, 0.3]))
        if synthetic
        else None
    )

    want = tpu_runner._verify_fused_impl(
        logits,
        bonus_idx,
        target_idx,
        temps,
        top_k,
        top_p,
        _dense_u(bonus_uc, ids[bonus_idx.long()], padded_num_reqs),
        accept_u,
        _dense_u(recover_uc, ids[target_idx.long()], padded_len),
        draft,
        lengths,
        seg,
        grp,
        rates,
        K,
        synthetic,
        64,
    )
    got, counts = vocab_shard_verify.verify_candidates_impl(
        packed,
        bonus_idx,
        target_idx,
        temps,
        top_k,
        top_p,
        bonus_uc,
        accept_u,
        recover_uc,
        draft,
        lengths,
        seg,
        grp,
        rates,
        torch.zeros(2, dtype=torch.int32),
        K,
        synthetic,
        WIDTH,
    )
    assert torch.equal(got, want)
    # Masked rows: 5 sampled bonus rows, and the target rows of the sampled
    # requests' declared drafts (3 + 3 + 3 + 2).
    assert counts.tolist() == [0, 5 + 11]


def test_ties_beyond_the_candidates_are_flagged(monkeypatch):
    # Row 0's logits: one large value, then 40 ties inside rank 0's shard, so
    # the 20th largest is tied with more than the candidates hold.
    hidden = 4
    weight = torch.zeros(VOCAB, hidden, dtype=torch.bfloat16)
    weight[7, 0] = 4.0
    weight[100:140, 0] = 1.0
    hidden_states = torch.zeros(2, hidden, dtype=torch.bfloat16)
    hidden_states[0, 0] = 1.0
    hidden_states[1, 1] = 1.0
    weight[200:260:3, 1] = torch.linspace(1, 2, 20).to(torch.bfloat16)
    packed = _candidates_by_play_back(weight, hidden_states, monkeypatch)
    values, _ = vocab_shard_verify.merge(packed, WIDTH)
    k = torch.full((2, 1), 20, dtype=torch.int32)
    flags = vocab_shard_verify.tie_flags(values, k, torch.ones(2, dtype=torch.bool))
    assert flags.tolist() == [True, False]


def _check_case(monkeypatch, seed, top_p=0.95):
    """A verify chunk, its candidate verify output and uniforms, for the check."""
    num_reqs, padded_num_reqs, padded_len = 6, 8, 32
    logits, packed = _logits_and_candidates(monkeypatch, rows=padded_len, seed=seed)
    bonus_idx, target_idx, seg, grp, lengths = _spec_batch(
        num_reqs, padded_num_reqs, padded_len, [3, 3, 3, 3, 3, 3]
    )
    temps, top_k, top_p_t = _params(padded_num_reqs, [(1.0, 20, top_p)] * num_reqs)
    gen = torch.Generator().manual_seed(200 + seed)
    draft = torch.randint(0, VOCAB, (padded_len,), generator=gen, dtype=torch.int32)
    case = dict(
        logits=logits,
        bonus_idx=bonus_idx,
        target_idx=target_idx,
        temps=temps,
        top_k=top_k,
        top_p=top_p_t,
        bonus_u=torch.rand(padded_num_reqs, WIDTH, generator=gen),
        accept_u=torch.rand(padded_len, generator=gen),
        recover_u=torch.rand(padded_len, WIDTH, generator=gen),
        draft=draft,
        lengths=lengths,
        seg=seg,
        grp=grp,
        num_reqs=num_reqs,
    )
    return case, packed


def _run_check(case, pairs, verify_pairs=None):
    """The candidate verify on `verify_pairs` (default `pairs`), then the check
    of `pairs` against the full logits."""
    rates = torch.tensor(unconditional_to_conditional_rates([0.8, 0.5, 0.3]))
    selected, _ = vocab_shard_verify.verify_candidates_impl(
        verify_pairs if verify_pairs is not None else pairs,
        case["bonus_idx"],
        case["target_idx"],
        case["temps"],
        case["top_k"],
        case["top_p"],
        case["bonus_u"],
        case["accept_u"],
        case["recover_u"],
        case["draft"],
        case["lengths"],
        case["seg"],
        case["grp"],
        rates,
        torch.zeros(2, dtype=torch.int32),
        K,
        True,
        WIDTH,
    )
    rows, emitted = vocab_shard_verify.check_candidates_impl(
        case["logits"],
        pairs,
        selected,
        case["bonus_idx"],
        case["target_idx"],
        case["temps"],
        case["top_k"],
        case["top_p"],
        case["bonus_u"],
        case["accept_u"],
        case["recover_u"],
        case["draft"],
        case["lengths"],
        case["seg"],
        case["grp"],
        rates,
        torch.tensor(case["num_reqs"]),
        K,
        True,
        64,
        WIDTH,
    )
    col = {n: rows[:, i] for i, n in enumerate(vocab_shard_verify.CHECK_COLUMNS)}
    return col, emitted


def _classes(col):
    return {
        c: int((col["cls"] == c).sum())
        for c in (
            vocab_shard_verify.CLS_TARGET,
            vocab_shard_verify.CLS_BONUS_BOUNDARY,
            vocab_shard_verify.CLS_BONUS_OTHER,
            vocab_shard_verify.CLS_DEGENERATE,
        )
    }


def test_check_classes_nothing_on_matching_candidates(monkeypatch):
    case, pairs = _check_case(monkeypatch, seed=9)
    col, emitted = _run_check(case, pairs)
    assert int(col["valid"].sum()) == case["num_reqs"] + 18
    assert set(_classes(col).values()) == {0}
    assert int(col["kept_mismatch"].sum()) == 0
    assert int(col["argmax_mismatch"].sum()) == 0
    assert int(emitted[:, 0].sum()) == 0 and int(emitted[:, 1].sum()) == 0
    assert int(col["logit_mismatch"].sum()) == 0
    assert int(col["logit_max_steps"].max()) == 0
    assert int(col["diff_token"].max()) == -1
    valid = col["valid"].bool()
    assert torch.equal(col["kept_full"][valid], col["kept_candidates"][valid])


def test_check_classes_wrong_candidates_as_a_b_and_d(monkeypatch):
    case, pairs = _check_case(monkeypatch, seed=10)
    broken = pairs.reshape(pairs.shape[0], TP, 2, WIDTH).clone()
    broken[:, :, 1, :] = torch.roll(broken[:, :, 1, :], 1, dims=0)
    broken = broken.reshape(pairs.shape[0], -1)
    col, emitted = _run_check(case, broken)
    classes = _classes(col)
    assert classes[vocab_shard_verify.CLS_TARGET] > 0
    assert classes[vocab_shard_verify.CLS_BONUS_OTHER] > 0
    assert classes[vocab_shard_verify.CLS_BONUS_BOUNDARY] == 0
    assert int(emitted[:, 0].sum()) > 0
    assert int(col["logit_mismatch"].sum()) > 0


def _drop_bonus_row0_token(monkeypatch, which):
    """Makes the candidate side's bf16 (bonus) keep mask drop row 0's last kept
    token ("last") or its top token ("top"): a stand-in for a rounding flip."""
    original = vocab_shard_verify.keep_mask

    def keep_mask(values, top_k, top_p):
        kept = original(values, top_k, top_p)
        if values.dtype != torch.bfloat16:
            return kept
        kept = kept.clone()
        col = int(kept[0].nonzero().max()) if which == "last" else 0
        kept[0, col] = False
        return kept

    monkeypatch.setattr(vocab_shard_verify, "keep_mask", keep_mask)


def _boundary_case(monkeypatch, seed):
    """Row 0's top_p sits 0.64 bf16 steps above the mass in front of a token j
    whose front mass is in [0.5, 0.85), so the full-logits mask keeps exactly
    the top j tokens. Returns (case, pairs, j)."""
    case, pairs = _check_case(monkeypatch, seed=seed)
    row = int(case["bonus_idx"][0])
    values, _ = vocab_shard_verify.merge(pairs, WIDTH)
    v = values[row, :21]
    probs = torch.softmax(v[:20].double(), dim=-1)
    front = torch.cumsum(probs, 0) - probs
    # bf16 logits tie often; take a j whose value differs from both neighbours.
    untied = torch.cat([torch.tensor([True]), v[1:20] != v[:19]]) & (v[:20] != v[1:21])
    j = int(((front >= 0.5) & (front < 0.85) & (probs > 0.02) & untied).nonzero()[0])
    case["top_p"][0] = float(front[j]) + 0.64 * 2**-8
    return case, pairs, j + 1


def test_check_classes_a_top_p_boundary_flip_as_c(monkeypatch):
    case, pairs, kept = _boundary_case(monkeypatch, seed=11)
    _drop_bonus_row0_token(monkeypatch, "last")
    col, _ = _run_check(case, pairs)
    assert int(col["kept_full"][0]) == kept
    assert int(col["kept_candidates"][0]) == kept - 1
    assert int(col["cls"][0]) == vocab_shard_verify.CLS_BONUS_BOUNDARY
    assert 0 <= int(col["band_err_ppm"][0]) <= 3906
    assert _classes(col)[vocab_shard_verify.CLS_BONUS_OTHER] == 0
    # The dropped token is reported, kept on the full logits only, with the
    # same probability on both paths (the logits agree).
    _, ids = vocab_shard_verify.merge(pairs, WIDTH)
    row = int(case["bonus_idx"][0])
    assert int(col["diff_token"][0]) == int(ids[row, kept - 1])
    assert int(col["diff_kept_full"][0]) == 1 and int(col["diff_kept_cand"][0]) == 0
    assert int(col["diff_p_full_ppm"][0]) == int(col["diff_p_cand_ppm"][0]) > 0


def test_check_classes_a_flip_away_from_the_boundary_as_d(monkeypatch):
    case, pairs, _ = _boundary_case(monkeypatch, seed=11)
    _drop_bonus_row0_token(monkeypatch, "top")
    col, _ = _run_check(case, pairs)
    assert int(col["cls"][0]) == vocab_shard_verify.CLS_BONUS_OTHER
    assert int(col["band_err_ppm"][0]) > 3906


def test_check_sets_non_finite_rows_aside(monkeypatch):
    case, pairs = _check_case(monkeypatch, seed=12)
    row = int(case["bonus_idx"][1])
    case["logits"] = case["logits"].clone()
    case["logits"][row] = float("nan")
    col, emitted = _run_check(case, pairs)
    assert int(col["cls"][1]) == vocab_shard_verify.CLS_DEGENERATE
    assert int(emitted[1, 1]) == 1 and int(emitted[1, 0]) == 0
    assert _classes(col)[vocab_shard_verify.CLS_BONUS_OTHER] == 0


def test_check_measures_a_one_ulp_logit_difference(monkeypatch):
    case, pairs = _check_case(monkeypatch, seed=13)
    row = int(case["target_idx"][2])
    values, ids = vocab_shard_verify.merge(pairs, WIDTH)
    # The 25th candidate: outside top-k 20, so no mask changes; bump it one
    # bf16 step down where it sits in its rank's list.
    token = int(ids[row, 24])
    lists = pairs.reshape(pairs.shape[0], TP, 2, WIDTH).clone()
    rank, slot = (lists[row, :, 1, :] == token).nonzero()[0].tolist()
    v = lists[row, rank, 0, slot].to(torch.bfloat16)
    lists[row, rank, 0, slot] = torch.nextafter(
        v, torch.tensor(-1e9, dtype=torch.bfloat16)
    ).float()
    col, _ = _run_check(case, lists.reshape(pairs.shape[0], -1))
    target_row = case["bonus_idx"].shape[0] + 2
    assert int(col["logit_mismatch"][target_row]) == 1
    assert int(col["logit_mismatch"].sum()) == 1
    assert int(col["logit_max_steps"][target_row]) == 1
    assert int(col["kept_mismatch"].sum()) == 0


def test_bf16_steps():
    a = torch.tensor([1.0, 12.0, -3.0])
    b = torch.tensor([1.0078125, 12.0625, -3.015625])
    assert vocab_shard_verify.bf16_steps(a, b).tolist() == [1.0, 1.0, 1.0]


def test_boundary_band_step_is_one_bf16_ulp_at_top_p():
    values = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    probs = torch.softmax(values, dim=-1)
    near = float(probs[0, :2].sum()) + 0.003
    far = float(probs[0, :2].sum()) + 0.005
    for top_p, inside in ((near, True), (far, False)):
        err, in_band = vocab_shard_verify.boundary_band(
            values,
            torch.ones(1, 1),
            torch.full((1, 1), 4, dtype=torch.int32),
            torch.full((1, 1), top_p),
        )
        assert bool(in_band[0, 2]) is inside
        assert abs(float(err[0, 2]) - abs(top_p - float(probs[0, :2].sum()))) < 1e-6


def _batch(*params, max_num_logprobs=None, vocab=VOCAB):
    requests = {
        f"r{i}": SimpleNamespace(sampling_params=p) for i, p in enumerate(params)
    }
    temps = torch.tensor([p.temperature for p in params])
    top_k = torch.tensor([p.top_k if 0 < p.top_k < vocab else vocab for p in params])
    input_batch = SimpleNamespace(
        all_greedy=all(p.temperature == 0 for p in params),
        max_num_logprobs=max_num_logprobs,
        num_reqs=len(params),
        req_ids=[*requests, None],
        temperature_cpu_tensor=torch.cat([temps, torch.zeros(1)]),
        top_k_cpu_tensor=torch.cat([top_k, torch.zeros(1, dtype=top_k.dtype)]),
    )
    return input_batch, requests


def test_step_eligible():
    served = SamplingParams(temperature=1.0, top_k=20, top_p=0.95)
    greedy = SamplingParams(temperature=0)
    eligible = vocab_shard_verify.step_eligible
    assert eligible(*_batch(served, served), WIDTH)
    assert eligible(*_batch(served, greedy), WIDTH)
    assert not eligible(*_batch(greedy, greedy), WIDTH)
    assert not eligible(*_batch(served, SamplingParams(temperature=1.0)), WIDTH)
    assert not eligible(
        *_batch(served, SamplingParams(temperature=1.0, top_k=32)), WIDTH
    )
    assert eligible(*_batch(served, SamplingParams(temperature=1.0, top_k=31)), WIDTH)
    assert not eligible(*_batch(served, served, max_num_logprobs=1), WIDTH)
    grammar = SamplingParams(
        temperature=1.0,
        top_k=20,
        structured_outputs=StructuredOutputsParams(regex="[ab]+"),
    )
    assert not eligible(*_batch(served, grammar), WIDTH)


def test_warmup_runs_the_candidate_path_when_enabled():
    assert tpu_runner._warmup_sampling_kwargs(False) == [
        {"greedy": True},
        {"greedy": False},
    ]
    assert tpu_runner._warmup_sampling_kwargs(True)[-1] == {
        "greedy": False,
        "candidates": True,
    }
    params = tpu_runner._warmup_sampling_params(False, WIDTH - 1)
    assert params.temperature == 1.0 and params.top_k == WIDTH - 1
    assert tpu_runner._warmup_sampling_params(False, 0).top_k in (0, -1)
    assert tpu_runner._warmup_sampling_params(True, WIDTH - 1).temperature == 0.0
