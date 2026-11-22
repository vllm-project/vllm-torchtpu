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
"""TPU_SAMPLER_VOCAB_PARALLEL_TARGET: the non-greedy speculative verify on
each rank's vocabulary shard of the target logits.

Tensor parallelism is simulated on CPU: one thread per rank, with vLLM's TP
all-reduce (a sum over ranks in rank order) and all-gather (concatenation in
rank order) as barrier exchanges, so a function with any number of dependent
collectives runs as it would on the ranks of a TP group.

Checks:
1. The top-p threshold equals the 1-bit bisection's when masses are summed
   in shard order, for every radix width; against `topp_mask`'s threshold
   (full-row sums) it differs only where a candidate's mass straddles top_p
   within rounding.
2. With the same uniforms, the race picks the id of a single-device race on
   the same probabilities: recovered rows with a valid draft, with a
   placeholder draft, bonus rows, ties, padding columns.
3. Real acceptance: the [requests, K + 1] output equals the rejection
   sampler's fed the same accept_u and recover_u (and bonus tokens from the
   f32 reference race), on rows away from the top-p boundary.
4. Synthetic acceptance: the same; through the runner, the draws match the
   default path's (same generator end state) and so do the non-bonus tokens.
5. A greedy request and a request whose top-p set is only its draft take
   the raw argmax.
6. 200,000 draws on a 64-token vocabulary in 4 shards pass a chi-square test
   against the exact distribution; a slicing that gives every shard the same
   uniform columns fails it.
7. A request with logprobs, a grammar, top_k, min_p, a penalty or a logit
   bias sends the step to the default path.
8. The precompile sweep is not rotated and covers every runtime shape.
"""

import contextlib
import functools
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm.distributed as vd
from vllm.sampling_params import SamplingParams

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.layers.adapter.sample import top_k_top_p as tkp
from vllm_torchtpu.layers.adapter.sample import vocab_parallel_verify as vpv
from vllm_torchtpu.runner import tpu_runner, vocab_parallel_target
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner import SAMPLING_EPS, TPUModelRunner

K = 3
RATES = [0.8, 0.66, 0.53]
TOKEN_PADDINGS = [16, 32, 64, 128]
F32 = torch.float32
TINY = torch.finfo(F32).tiny


class _ThreadTP:
    """Runs one function per simulated TP rank, each in its own thread, with
    the TP collectives as barrier exchanges. The all-reduce sums in rank
    order in f32 and rounds once to the input dtype; the all-gather
    concatenates in rank order. Records each collective's kind and input
    shape (rank 0)."""

    def __init__(self, tp: int):
        self.tp = tp
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self._barrier = threading.Barrier(tp, timeout=300)
        self._slots: list[torch.Tensor | None] = [None] * tp
        self._local = threading.local()

    def _exchange(self, kind: str, x: torch.Tensor) -> list[torch.Tensor]:
        rank = self._local.rank
        self._slots[rank] = x
        self._barrier.wait()
        parts = list(self._slots)
        if rank == 0:
            self.calls.append((kind, tuple(x.shape)))
        self._barrier.wait()
        return parts  # type: ignore[return-value]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        parts = self._exchange("all_reduce", x)
        acc = torch.zeros_like(x, dtype=F32)
        for part in parts:
            acc = acc + part.to(F32)
        return acc.to(x.dtype)

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.cat(self._exchange("all_gather", x), dim=dim)

    def run(self, fn):
        results: list = [None] * self.tp
        errors: list[BaseException] = []

        def target(rank):
            self._local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                self._barrier.abort()

        threads = [threading.Thread(target=target, args=(r,)) for r in range(self.tp)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results


@pytest.fixture
def tp_group(monkeypatch):
    def make(tp: int) -> _ThreadTP:
        sim = _ThreadTP(tp)
        monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce", sim.all_reduce)
        monkeypatch.setattr(vd, "tensor_model_parallel_all_gather", sim.all_gather)
        return sim

    return make


@pytest.fixture(autouse=True)
def _one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _layout(vocab: int, tp: int) -> tuple[int, list[int], list[int]]:
    """Shard width (padded) and each rank's start and real column count."""
    width = -(-vocab // tp)
    starts = [r * width for r in range(tp)]
    real = [max(0, min(width, vocab - s)) for s in starts]
    return width, starts, real


def _split(full: torch.Tensor, tp: int, pad_value: float = 0.0) -> list[torch.Tensor]:
    width, _, _ = _layout(full.shape[-1], tp)
    padded = torch.nn.functional.pad(
        full, (0, tp * width - full.shape[-1]), value=pad_value
    )
    return list(padded.split(width, dim=-1))


def _start(rank: int, width: int) -> torch.Tensor:
    return torch.tensor(rank * width, dtype=torch.int32)


def _ref_race(probs, kept, excluded, u):
    """The rejection sampler's recovered-token race on full rows."""
    column = torch.arange(probs.shape[-1]).unsqueeze(0)
    residual = torch.where(kept & (column != excluded.unsqueeze(-1)), probs, 0.0)
    u = torch.clamp(u, min=TINY, max=1.0 - torch.finfo(F32).eps)
    values = residual / (-torch.log(u) + TINY)
    best = torch.argmax(values, dim=-1)
    return values.gather(-1, best.unsqueeze(-1)).squeeze(-1), best.to(torch.int32)


def _probs(rows: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scale = torch.rand(rows, 1, generator=g) * 6.0 + 0.5
    return torch.softmax(torch.randn(rows, vocab, generator=g) * scale, dim=-1)


def _shard_order_mass(shards, threshold):
    acc = torch.zeros(threshold.shape, dtype=F32)
    for shard in shards:
        acc = acc + torch.sum(
            torch.where(shard >= threshold.unsqueeze(-1), shard, 0.0), dim=-1
        )
    return acc


# --- 1. threshold ----------------------------------------------------------


@pytest.mark.parametrize("radix_bits", [1, 2, 3, 4])
@pytest.mark.parametrize("tp", [4, 16])
def test_threshold_equals_shard_order_bisection(tp_group, tp, radix_bits):
    """Bit for bit the 1-bit bisection (sign and all 31 bits evaluated) whose
    masses are summed in shard order, as the simulated all-reduce sums."""
    rows, width = 96, 64
    probs = _probs(rows, tp * width, seed=tp + radix_bits)
    g = torch.Generator().manual_seed(7)
    top_p = torch.rand(rows, generator=g) * 0.49 + 0.5
    top_p[::7] = 0.999
    shards = _split(probs, tp)
    sim = tp_group(tp)
    out = sim.run(
        lambda r: tkp.vocab_parallel_topp_threshold(shards[r], top_p, radix_bits)
    )
    for t in out[1:]:
        assert torch.equal(t.view(torch.int32), out[0].view(torch.int32))
    ref = tkp._float32_bsearch(
        (rows,), lambda t: _shard_order_mass(shards, t) < top_p, probs.device, 1
    )
    assert torch.equal(out[0].view(torch.int32), ref.view(torch.int32))
    # ceil(30 / radix_bits) rounds, one all-reduce of [2**b - 1, rows] each.
    rounds = -(-30 // radix_bits)
    reduces = [shape for kind, shape in sim.calls if kind == "all_reduce"]
    assert len(reduces) == rounds
    if radix_bits > 1:
        assert reduces[0] == ((1 << radix_bits) - 1, rows)


@pytest.mark.parametrize("tp", [4, 16])
def test_threshold_matches_topp_mask_off_the_rounding_boundary(tp_group, tp):
    """Against `topp_mask`'s threshold (the default path, masses summed over
    the full row in one reduction) the thresholds differ only where the two
    summation orders put a candidate's mass on opposite sides of top_p."""
    rows, width = 2048, 64
    vocab = tp * width
    probs = _probs(rows, vocab, seed=11 * tp)
    top_p = torch.full((rows,), 0.95)
    shards = _split(probs, tp)
    sim = tp_group(tp)
    out = sim.run(lambda r: tkp.vocab_parallel_topp_threshold(shards[r], top_p, 3))[0]
    ref = tkp._topp_bisect_threshold(probs, top_p, 1, fast=True)
    differ = (out != ref).nonzero().flatten().tolist()
    assert len(differ) <= rows // 100, differ
    ulp = float(np.spacing(np.float32(0.95)))
    for row in differ:
        hi = torch.maximum(out[row], ref[row])
        full_mass = torch.sum(torch.where(probs[row] >= hi, probs[row], 0.0))
        shard_mass = _shard_order_mass([s[row : row + 1] for s in shards], hi.view(1))
        exact = probs[row].double()
        exact_mass = float(torch.sum(torch.where(exact >= float(hi), exact, 0.0)))
        # One order reaches top_p at that candidate and the other does not.
        assert (float(full_mass) >= 0.95) != (float(shard_mass) >= 0.95), row
        assert abs(exact_mass - 0.95) <= 4 * ulp, (row, exact_mass)


# --- 2. draw ---------------------------------------------------------------


@pytest.mark.parametrize("tp,vocab", [(4, 4 * 37 - 5), (16, 16 * 64)])
def test_race_matches_single_device_race(tp_group, tp, vocab):
    """Same probabilities, same uniforms: the same winner and value. Rows:
    recovered with a valid draft (sometimes the argmax), recovered with a
    placeholder draft, bonus rows (nothing excluded), a tie across shards and
    a row with no mass left."""
    rows = 48
    g = torch.Generator().manual_seed(vocab)
    probs = _probs(rows, vocab, seed=vocab)
    threshold = torch.quantile(probs, 0.9, dim=-1)
    kept = probs >= threshold.unsqueeze(-1)
    kept[:8] = True  # top_p >= 1
    u = torch.rand(rows, vocab, generator=g)
    excluded = torch.randint(0, vocab, (rows,), generator=g)
    excluded[:6] = probs[:6].argmax(dim=-1)
    excluded[16:32] = -1  # placeholder drafts and bonus rows
    width, starts, real = _layout(vocab, tp)
    # Tie across shards: equal probability and uniform at ids 1 and width + 1.
    probs[40, 1] = probs[40, width + 1] = 0.5
    kept[40, 1] = kept[40, width + 1] = True
    u[40, 1] = u[40, width + 1] = 0.9999
    excluded[40] = -1
    # No mass left: the top-p set is only the excluded draft.
    kept[41] = False
    kept[41, 3] = True
    excluded[41] = 3
    probs_s, kept_s = _split(probs, tp), _split(kept, tp, pad_value=True)
    sim = tp_group(tp)
    out = sim.run(
        lambda r: vpv.race_on_shard(
            probs_s[r],
            kept_s[r],
            excluded,
            vpv.shard_columns(u, _start(r, width), width, tp),
            _start(r, width),
            real[r],
            tp,
        )
    )
    ref_max, ref_id = _ref_race(probs, kept, excluded, u)
    for value, token in out:
        assert torch.equal(token, ref_id)
        assert torch.equal(value, ref_max)
    assert int(ref_id[40]) == 1
    assert float(ref_max[41]) == 0.0


@pytest.mark.parametrize("minor_slice", [False, True], ids=["view", "slice"])
def test_shard_columns_is_the_rank_slice(minor_slice):
    tp, vocab = 4, 4 * 37 - 5
    width, starts, _ = _layout(vocab, tp)
    u = torch.rand(3, vocab)
    for r in range(tp):
        got = vpv.shard_columns(u, _start(r, width), width, tp, minor_slice)
        n = min(width, vocab - starts[r])
        assert torch.equal(got[:, :n], u[:, starts[r] : starts[r] + n])
        assert bool((got[:, n:] == 0.5).all())


@pytest.mark.parametrize("tp,vocab", [(16, 16 * 9680), (4, 4 * 37 - 5), (8, 8 * 128)])
def test_minor_slice_equals_the_view(tp, vocab):
    """TPU_SAMPLER_VP_UNIFORM_SLICE takes the same uniforms, bit for bit, on
    every rank, the served layout (16 x 9680) included."""
    width, _, _ = _layout(vocab, tp)
    u = torch.rand(5, vocab, generator=torch.Generator().manual_seed(vocab))
    for r in range(tp):
        view = vpv.shard_columns(u, _start(r, width), width, tp)
        sliced = vpv.shard_columns(u, _start(r, width), width, tp, minor_slice=True)
        assert sliced.shape == view.shape == (5, width)
        assert torch.equal(sliced, view)


# --- 3/4/5. the verify against the rejection sampler ------------------------


def _boundary_distance(probs_f32: torch.Tensor, top_p: float) -> torch.Tensor:
    """Per row, the smallest |cumulative mass - top_p| over the sorted
    probabilities (f64): how far the top-p cut is from flipping."""
    p = probs_f32.double().sort(dim=-1, descending=True).values
    return (p.cumsum(dim=-1) - top_p).abs().min(dim=-1).values


def _chunk(case, seed, vocab, padded_num_reqs=32, greedy_req=None, temps=None):
    """Logits rows (away from every top-p boundary), metadata and per-request
    params of one verify chunk with MTP k=3."""
    g = torch.Generator().manual_seed(seed)
    rows = int(np.sum(np.asarray(case) + 1))
    padded = next(p for p in TOKEN_PADDINGS if p >= rows)
    req_temps = torch.ones((padded_num_reqs, 1), dtype=torch.bfloat16)
    if temps is not None:
        req_temps[: len(temps), 0] = torch.tensor(temps, dtype=torch.bfloat16)
    if greedy_req is not None:
        req_temps[greedy_req] = 0.0
    top_p = torch.ones((padded_num_reqs, 1), dtype=F32)
    top_p[: len(case)] = 0.95
    top_p[1] = 1.0
    scale = torch.rand(padded, 1, generator=g) * 7.0 + 1.0
    logits = (torch.randn(padded, vocab, generator=g) * scale).to(torch.bfloat16)
    # Re-draw rows whose top-p cut sits within 1e-4 of flipping, at both
    # temperatures a row can be scaled by.
    for _ in range(50):
        near = torch.zeros(padded, dtype=torch.bool)
        for t in {float(x) for x in req_temps.flatten().tolist()} - {0.0}:
            p = torch.softmax(logits.float() / t, dim=-1)
            near |= _boundary_distance(p, 0.95) < 1e-4
        if not bool(near.any()):
            break
        logits[near] = (
            torch.randn(int(near.sum()), vocab, generator=g) * scale[near]
        ).to(torch.bfloat16)
    input_ids = torch.randint(0, vocab, (4096,), dtype=torch.int32, generator=g)
    top = logits.float().argmax(dim=-1)
    pos = 0
    for n in case:
        for j in range(n):
            if torch.rand(1, generator=g).item() < 0.7:
                input_ids[pos + j + 1] = int(top[pos + j])
        pos += n + 1
    nd = np.asarray(case, dtype=np.int32)
    runner = SimpleNamespace(
        input_ids_cpu=input_ids,
        arange_np=np.arange(4096, dtype=np.int64),
        num_tokens_paddings=TOKEN_PADDINGS,
        device=torch.device("cpu"),
    )
    md = SpeculativeDecodingManager(runner).get_spec_decode_metadata(
        nd, np.cumsum(nd + 1).astype(np.int32), padded_num_reqs
    )
    return logits, md, req_temps, top_p


def _uniforms(md, vocab, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    num_reqs = md.bonus_logits_indices.shape[0]
    bonus_u = torch.rand((num_reqs, vocab), generator=g)
    accept_u = torch.rand(md.draft_token_ids.shape, generator=g)
    recover_u = torch.rand((md.target_logits_indices.shape[0], vocab), generator=g)
    return accept_u, recover_u, bonus_u


def _ref_bonus(logits, temps, top_p, bonus_u):
    """Bonus tokens by the f32 top-p and exponential race the shard path
    uses (the default path draws them by bf16 Gumbel-max)."""
    t = temps.squeeze(-1)
    safe = torch.where(t == 0, torch.ones_like(t), t).float()
    probs = torch.softmax(logits.float() / safe.unsqueeze(-1), dim=-1)
    threshold = tkp._topp_bisect_threshold(probs, top_p.squeeze(-1), 1, fast=True)
    kept = (probs >= threshold.unsqueeze(-1)) | (top_p >= 1.0)
    excluded = torch.full((logits.shape[0],), -1)
    _, token = _ref_race(probs, kept, excluded, bonus_u)
    greedy = t <= SAMPLING_EPS
    return torch.where(greedy, logits.argmax(dim=-1).to(torch.int32), token)


def _reference(logits, md, temps, top_p, uniforms, synthetic, verify_rows=0):
    accept_u, recover_u, bonus_u = uniforms
    sampler = rs.RejectionSampler(
        SimpleNamespace(
            rejection_sample_method="synthetic", synthetic_acceptance_rates=RATES
        )
        if synthetic
        else None
    )
    num_reqs = md.bonus_logits_indices.shape[0]
    bonus = _ref_bonus(logits[md.bonus_logits_indices.long()], temps, top_p, bonus_u)
    seg = md.segment_ids.long().clamp(max=num_reqs - 1)
    return rs._random_rejection_sample_with_segment_impl(
        md.draft_token_ids,
        logits[md.target_logits_indices.long()],
        md.draft_lengths,
        bonus,
        md.segment_ids,
        md.group_indices,
        K,
        temps[seg],
        torch.zeros((seg.shape[0], 1), dtype=torch.int32),
        top_p[seg],
        accept_u,
        recover_u,
        sampler.synthetic_conditional_rates,
        sampler.synthetic_mode,
        need_top_k=False,
        fast_top_p=True,
        verify_rows=verify_rows,
    )


def _on_shards(
    sim,
    logits,
    md,
    temps,
    top_p,
    uniforms,
    synthetic,
    verify_rows=0,
    uniform_slice=False,
):
    accept_u, recover_u, bonus_u = uniforms
    vocab = logits.shape[-1]
    width, _, real = _layout(vocab, sim.tp)
    shards = _split(logits, sim.tp, pad_value=100.0)
    rates = torch.tensor(rs.unconditional_to_conditional_rates(RATES), dtype=F32)
    outs = sim.run(
        lambda r: vpv.vocab_parallel_spec_verify(
            shards[r],
            md.target_logits_indices,
            md.bonus_logits_indices,
            md.draft_token_ids,
            md.draft_lengths,
            md.segment_ids,
            md.group_indices,
            temps,
            top_p,
            accept_u,
            recover_u,
            bonus_u,
            _start(r, width),
            real[r],
            sim.tp,
            K,
            rates if synthetic else None,
            synthetic,
            3,
            SAMPLING_EPS,
            verify_rows,
            uniform_slice,
        )
    )
    for out in outs[1:]:
        assert torch.equal(out, outs[0])
    return outs[0]


CASES = [[K] * 9, [K] * 13, [K] * 10 + [1, 0], [2] * 16, [K] * 16]


@pytest.mark.parametrize("uniform_slice", [False, True], ids=["view", "slice"])
@pytest.mark.parametrize("synthetic", [False, True], ids=["real", "synthetic"])
@pytest.mark.parametrize("tp,vocab", [(4, 4 * 128 - 5), (16, 16 * 128)])
def test_verify_matches_rejection_sampler(
    tp_group, synthetic, tp, vocab, uniform_slice
):
    """The whole [requests, K + 1] output, recovered tokens, accept decisions
    and bonus placement included, equals the rejection sampler's on the same
    uniforms; a greedy request and a top_p = 1 request ride along. Both ways
    of taking the shard's uniform columns (TPU_SAMPLER_VP_UNIFORM_SLICE)."""
    sim = tp_group(tp)
    rejected = all_accepted = 0
    for i, case in enumerate(CASES):
        for seed in range(2):
            logits, md, temps, top_p = _chunk(
                case, seed + 10 * i, vocab, greedy_req=2, temps=[1.0, 1.0, 1.0, 0.7]
            )
            uniforms = _uniforms(md, vocab, seed)
            ref = _reference(logits, md, temps, top_p, uniforms, synthetic)
            got = _on_shards(
                sim,
                logits,
                md,
                temps,
                top_p,
                uniforms,
                synthetic,
                uniform_slice=uniform_slice,
            )
            assert torch.equal(got, ref), (case, seed)
            filled = (ref[: len(case)] != rs.PLACEHOLDER_TOKEN_ID).sum(dim=1)
            rejected += int((filled <= torch.tensor(case)).sum())
            all_accepted += int((filled == torch.tensor(case) + 1).sum())
    assert rejected > 0 and all_accepted > 0


@pytest.mark.parametrize("uniform_slice", [False, True], ids=["view", "slice"])
@pytest.mark.parametrize("synthetic", [False, True], ids=["real", "synthetic"])
def test_verify_rows_trim_is_exact(tp_group, synthetic, uniform_slice):
    """TPU_SAMPLER_VERIFY_TRIM_ROWS: a bucket that holds every real draft
    slot leaves the output unchanged."""
    sim = tp_group(4)
    vocab = 4 * 96
    for case in ([K] * 9, [K] * 10 + [1, 0]):
        logits, md, temps, top_p = _chunk(case, 5, vocab)
        uniforms = _uniforms(md, vocab, 5)
        full = _on_shards(sim, logits, md, temps, top_p, uniforms, synthetic)
        trimmed = _on_shards(
            sim, logits, md, temps, top_p, uniforms, synthetic, 40, uniform_slice
        )
        assert torch.equal(full, trimmed)
        assert torch.equal(
            trimmed, _reference(logits, md, temps, top_p, uniforms, synthetic, 40)
        )


def test_real_acceptance_runs_one_more_exchange(tp_group):
    """Collectives of one 160-token-like chunk: the statistics all-gather,
    ten radix rounds at 3 bits, the draft-probability all-reduce (real
    acceptance only) and the pair all-gather."""
    vocab = 4 * 64
    logits, md, temps, top_p = _chunk([K] * 9, 1, vocab)
    uniforms = _uniforms(md, vocab, 1)
    rows = md.target_logits_indices.shape[0] + md.bonus_logits_indices.shape[0]
    slots = md.target_logits_indices.shape[0]
    for synthetic in (True, False):
        sim = tp_group(4)
        _on_shards(sim, logits, md, temps, top_p, uniforms, synthetic)
        expected = [("all_gather", (rows, 4))] + [("all_reduce", (7, rows))] * 10
        if not synthetic:
            expected.append(("all_reduce", (2, slots)))
        expected.append(("all_gather", (rows, 2)))
        assert sim.calls == expected


def test_greedy_row_and_draft_only_top_p_set_take_the_raw_argmax(tp_group):
    """Request 0 is greedy; request 1's first row is so peaked that its top-p
    set is only its draft, which synthetic acceptance rejects: both recover
    the raw argmax, as in the rejection sampler."""
    vocab = 4 * 64
    logits, md, temps, top_p = _chunk([K, K, K], 3, vocab, greedy_req=0)
    top_p[1] = 0.95
    row = int(md.target_logits_indices[K])  # request 1, slot 0
    peak = int(logits[row].float().argmax())
    logits[row] = -20.0
    logits[row, peak] = 20.0
    md.draft_token_ids[K] = peak
    accept_u, recover_u, bonus_u = _uniforms(md, vocab, 3)
    accept_u[K] = 0.99  # rate 0.8 at slot 0: rejected
    uniforms = (accept_u, recover_u, bonus_u)
    sim = tp_group(4)
    got = _on_shards(sim, logits, md, temps, top_p, uniforms, synthetic=True)
    ref = _reference(logits, md, temps, top_p, uniforms, synthetic=True)
    assert torch.equal(got, ref)
    assert int(got[1, 0]) == peak
    greedy_rows = md.target_logits_indices[:K].long()
    first_rejected = int((got[0] == rs.PLACEHOLDER_TOKEN_ID).nonzero()[0]) - 1
    if first_rejected < K:
        assert int(got[0, first_rejected]) == int(
            logits[greedy_rows[first_rejected]].float().argmax()
        )


# --- 4 (runner). draws and dispatch through the runner ----------------------


def _runner(
    sim_tp, vocab, synthetic, graph_rng, trim_rows=(64, 96), uniform_slice=False
):
    width, _, real = _layout(vocab, sim_tp)
    sampler = rs.RejectionSampler(
        SimpleNamespace(
            rejection_sample_method="synthetic", synthetic_acceptance_rates=RATES
        )
        if synthetic
        else None
    )
    r = SimpleNamespace(
        rejection_sampler=sampler,
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        _sampler_fused_chain=False,
        _sampler_graph_rng=graph_rng,
        _sampler_verify_trim=False,
        _sampler_verify_trim_rows=tuple(trim_rows),
        _sampler_light=True,
        _sampler_need_top_k=False,
        _sampler_need_top_p=True,
        _sampler_radix_bits=1,
        _sampler_fast_top_p=True,
        _sampler_topk_candidates=lambda *args: 0,
        _vp_radix_bits=3,
        _vp_uniform_slice=uniform_slice,
        _vp_head=SimpleNamespace(
            vocab_start=None,
            num_org_local=None,
            width=width,
            tp_size=sim_tp,
            vocab_size=vocab,
        ),
    )
    for name in (
        "_sampler_kwargs",
        "_verify_trim_buckets",
        "_verify_trim_rows",
        "_spec_verify_on_shard_kwargs",
    ):
        setattr(r, name, functools.partial(getattr(TPUModelRunner, name), r))
    for name in (
        "sample_from_logits",
        "spec_gather_bonus_and_target_logits",
        "spec_verify_on_shard",
    ):
        fn = getattr(TPUModelRunner, name)._torchdynamo_orig_callable
        setattr(r, name, functools.partial(fn, r))
    r.sample_from_logits_func = r.sample_from_logits
    r._get_sampling_generator = lambda: None
    return r, real


@pytest.mark.parametrize("uniform_slice", [False, True], ids=["view", "slice"])
@pytest.mark.parametrize("graph_rng", [False, True])
@pytest.mark.parametrize("synthetic", [True, False], ids=["synthetic", "real"])
def test_runner_draws_match_the_default_path(
    tp_group, monkeypatch, synthetic, graph_rng, uniform_slice
):
    """_sample_spec_verify_chunk_on_shard draws bonus_u, accept_u and
    recover_u at the default path's shapes and in its order, from the
    sampling generator (or in-graph from the default one): the generator ends
    where the default path leaves it, every token that is not a bonus token
    equals the default verify's, and the whole output equals the reference on
    those uniforms. The target rows are trimmed to a bucket."""
    monkeypatch.setattr(
        rs,
        "_random_rejection_sample_with_segment",
        rs._random_rejection_sample_with_segment_impl,
    )
    tp, vocab, seed = 4, 4 * 96, 21
    case = [K] * 9
    logits, md, temps, top_p = _chunk(case, 4, vocab)
    params = (temps, torch.zeros_like(temps, dtype=torch.int32), top_p)

    def draw_all():
        gen = torch.Generator().manual_seed(seed)
        return (
            torch.rand((md.bonus_logits_indices.shape[0], vocab), generator=gen),
            torch.rand(md.draft_token_ids.shape, generator=gen),
            torch.rand((md.target_logits_indices.shape[0], vocab), generator=gen),
            gen.get_state(),
        )

    bonus_u, accept_u, recover_u, end_state = draw_all()

    base, _ = _runner(tp, vocab, synthetic, graph_rng)
    base._build_padded_sampling_params = lambda *a, **kw: params
    if graph_rng:
        torch.manual_seed(seed)
        gen = None
    else:
        gen = torch.Generator().manual_seed(seed)
    default_out = TPUModelRunner._sample_spec_verify_chunk(
        base, logits, md, None, SimpleNamespace(), 0, len(case), False, gen
    )
    assert torch.equal(
        torch.get_rng_state() if graph_rng else gen.get_state(), end_state
    )

    sim = tp_group(tp)
    shards = _split(logits, tp)
    width, _, real = _layout(vocab, tp)
    gens = [torch.Generator().manual_seed(seed) for _ in range(tp)]
    if graph_rng:
        # One process-wide default generator: give every rank thread the
        # stream the default path drew, draw by draw.
        draws = (bonus_u, accept_u, recover_u)
        seen: dict[int, int] = {}

        def replay_rand(size, *args, **kwargs):
            rank = sim._local.rank
            i = seen.get(rank, 0)
            seen[rank] = i + 1
            assert tuple(draws[i].shape) == tuple(size)
            return draws[i].clone()

        monkeypatch.setattr(torch, "rand", replay_rand)

    def rank_fn(rank):
        r, _ = _runner(
            tp,
            vocab,
            synthetic,
            graph_rng,
            trim_rows=(40, 48),
            uniform_slice=uniform_slice,
        )
        r._vp_head.vocab_start = _start(rank, width)
        r._vp_head.num_org_local = real[rank]
        r._build_padded_sampling_params = lambda *a, **kw: params
        return TPUModelRunner._sample_spec_verify_chunk_on_shard(
            r, shards[rank], md, 0, len(case), None if graph_rng else gens[rank]
        )

    outs = sim.run(rank_fn)
    if graph_rng:
        assert seen == dict.fromkeys(range(tp), 3)
    else:
        for g in gens:
            assert torch.equal(g.get_state(), end_state)
    got = outs[0]
    for out in outs[1:]:
        assert torch.equal(out, got)
    ref = _reference(
        logits, md, temps, top_p, (accept_u, recover_u, bonus_u), synthetic
    )
    assert torch.equal(got, ref)
    n, nd = len(case), torch.tensor(case)
    filled = got[:n] != rs.PLACEHOLDER_TOKEN_ID
    full = filled.sum(dim=1) == nd + 1
    bonus_slot = torch.zeros_like(filled)
    bonus_slot[full, nd[full]] = True
    keep = filled & ~bonus_slot
    assert torch.equal(got[:n][keep], default_out[:n][keep])
    assert bool(keep.any())


# --- 6. statistics ---------------------------------------------------------


def _chi_square_p(counts: np.ndarray, expected: np.ndarray) -> float:
    from scipy.stats import chisquare

    keep = expected >= 5
    obs = np.append(counts[keep], counts[~keep].sum())
    exp = np.append(expected[keep], expected[~keep].sum())
    if exp[-1] == 0:
        obs, exp = obs[:-1], exp[:-1]
    return float(chisquare(obs, exp).pvalue)


def _draw_counts(tp_group, draws=200_000, batch=25_000, seed=0):
    tp, vocab = 4, 64
    width, _, real = _layout(vocab, tp)
    g = torch.Generator().manual_seed(seed)
    row = (torch.randn(vocab, generator=g) * 1.5).to(torch.bfloat16)
    draft = int(row.float().argmax())
    top_p = 0.9
    probs = torch.softmax(row.double(), dim=-1)
    # The top-p set in f64, checked to be far from flipping.
    assert float(_boundary_distance(probs.float().unsqueeze(0), top_p)) > 1e-3
    order = probs.sort(descending=True)
    cum = order.values.cumsum(0)
    n_kept = int((cum < top_p).sum()) + 1
    kept = torch.zeros(vocab, dtype=torch.bool)
    kept[order.indices[:n_kept]] = True
    target = torch.where(kept, probs, 0.0)
    target[draft] = 0.0
    expected_p = (target / target.sum()).numpy()
    shards = _split(row.unsqueeze(0), tp)
    counts = np.zeros(vocab, dtype=np.int64)
    sim = tp_group(tp)
    for _ in range(draws // batch):
        u = torch.rand((batch, vocab), generator=g)
        excluded = torch.full((batch,), draft)

        def rank_fn(r, u=u, excluded=excluded):
            tokens, _ = vpv.sample_on_shard(
                shards[r].expand(batch, width),
                torch.ones(batch),
                torch.full((batch,), top_p),
                excluded,
                torch.zeros(batch, dtype=torch.bool),
                (u,),
                _start(r, width),
                real[r],
                tp,
                3,
            )
            return tokens

        tokens = sim.run(rank_fn)[0]
        counts += np.bincount(tokens.numpy(), minlength=vocab)
    return counts, expected_p * draws


def test_draws_follow_the_exact_distribution(tp_group):
    counts, expected = _draw_counts(tp_group)
    assert counts[expected == 0].sum() == 0
    assert _chi_square_p(counts, expected) > 1e-3


def test_positive_control_shared_uniform_columns_fail(tp_group, monkeypatch):
    """Every shard racing on the same uniform columns (the first width of
    the draw) is a biased sampler; the chi-square test must reject it."""
    monkeypatch.setattr(
        vpv, "shard_columns", lambda u, start, width, tp: u[:, :width].contiguous()
    )
    counts, expected = _draw_counts(tp_group)
    assert _chi_square_p(counts, expected) < 1e-3


# --- 7. gating -------------------------------------------------------------


def _batch(params_list, all_greedy=False, max_num_logprobs=None):
    req_ids = [f"r{i}" for i in range(len(params_list))] + [None]
    requests = {
        f"r{i}": SimpleNamespace(sampling_params=p) for i, p in enumerate(params_list)
    }
    batch = SimpleNamespace(
        all_greedy=all_greedy, max_num_logprobs=max_num_logprobs, req_ids=req_ids
    )
    return batch, requests


SERVED = {"temperature": 1.0, "top_p": 0.95}


@pytest.mark.parametrize(
    "extra",
    [
        {"top_k": 50},
        {"min_p": 0.05},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.5},
        {"repetition_penalty": 1.1},
        {"logit_bias": {7: 2.0}},
    ],
    ids=lambda d: next(iter(d)),
)
def test_step_with_an_unsupported_field_takes_the_default_path(extra):
    batch, requests = _batch([SamplingParams(**SERVED), SamplingParams(**SERVED)])
    assert vocab_parallel_target.step_eligible(batch, requests)
    batch, requests = _batch(
        [SamplingParams(**SERVED), SamplingParams(**SERVED, **extra)]
    )
    assert not vocab_parallel_target.step_eligible(batch, requests)


def test_logprobs_grammar_and_all_greedy_take_the_default_path():
    from vllm.sampling_params import StructuredOutputsParams

    batch, requests = _batch([SamplingParams(**SERVED)], max_num_logprobs=5)
    assert not vocab_parallel_target.step_eligible(batch, requests)
    batch, requests = _batch([SamplingParams(**SERVED)], all_greedy=True)
    assert not vocab_parallel_target.step_eligible(batch, requests)
    grammar = SamplingParams(
        **SERVED, structured_outputs=StructuredOutputsParams(regex="[ab]+")
    )
    batch, requests = _batch([SamplingParams(**SERVED), grammar])
    assert not vocab_parallel_target.step_eligible(batch, requests)
    # top_k = -1 and 0 both mean unset.
    batch, requests = _batch([SamplingParams(**SERVED, top_k=-1)])
    assert vocab_parallel_target.step_eligible(batch, requests)


@pytest.mark.parametrize(
    "flag,head,lockstep,eligible",
    [
        (True, True, False, True),
        (False, True, False, False),  # turned off on a running engine
        (True, False, False, False),  # env unset at load: nothing warmed
        (True, True, True, False),  # EP-DP lockstep
    ],
)
def test_runner_step_gate(flag, head, lockstep, eligible):
    batch, requests = _batch([SamplingParams(**SERVED)])
    r = SimpleNamespace(
        _vp_target=flag,
        _vp_head=object() if head else None,
        rejection_sampler=object(),
        _dp_lockstep_enabled=lambda: lockstep,
        input_batch=batch,
        requests=requests,
    )
    assert TPUModelRunner._vp_target_this_step(r) is eligible


# --- 8. precompile ---------------------------------------------------------


def _sweep_runner(trim_rows=(64, 96, 128, 160), graph_rng=False):
    calls: list[tuple] = []
    heads: list[tuple] = []

    def shard_head(hidden, indices):
        heads.append((hidden.shape[0], indices.shape[0]))
        return torch.zeros((indices.shape[0], 8), dtype=torch.bfloat16)

    def verify(shard, target_idx, bonus_idx, draft, lengths, seg, group, *rest, **kw):
        assert target_idx is not draft and draft is not group  # no aliasing
        calls.append((shard.shape[0], bonus_idx.shape[0], kw["verify_rows"]))
        return torch.zeros(1)

    runner = SimpleNamespace(
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        rejection_sampler=SimpleNamespace(
            synthetic_mode=True, synthetic_conditional_rates=torch.ones(K)
        ),
        num_tokens_paddings=[16, 32, 64, 96, 128, 160, 192, 256],
        num_reqs_paddings=[8, 16, 32, 48],
        max_num_reqs=48,
        model_config=SimpleNamespace(get_hidden_size=lambda: 4),
        _hidden_states_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        _precompile_timed=lambda name: contextlib.nullcontext(),
        _sampler_graph_rng=graph_rng,
        _sampler_verify_trim=False,
        _sampler_verify_trim_rows=tuple(trim_rows),
        _vp_radix_bits=3,
        _vp_uniform_slice=False,
        _vp_head=SimpleNamespace(
            vocab_start=torch.tensor(0, dtype=torch.int32),
            num_org_local=8,
            width=8,
            tp_size=2,
            vocab_size=16,
        ),
        sequence_layout_planner=SimpleNamespace(
            uses_selected_logits_hidden_states=False
        ),
        compute_selected_shard_logits=shard_head,
        spec_verify_on_shard=verify,
    )
    for name in (
        "_verify_trim_buckets",
        "_verify_trim_rows",
        "_spec_verify_on_shard_kwargs",
        "_spec_verify_bucket_reachable",
    ):
        setattr(runner, name, functools.partial(getattr(TPUModelRunner, name), runner))
    return runner, calls, heads


@pytest.mark.parametrize("flag", [None, "1"])
@pytest.mark.parametrize("rank", [0, 3, 7])
def test_precompile_sweep_is_not_rotated(monkeypatch, flag, rank):
    """The shard-verify programs hold collectives: every rank warms them in
    the same order, TPU_PARALLEL_PRECOMPILE or not."""
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    if flag is None:
        monkeypatch.delenv("TPU_PARALLEL_PRECOMPILE", raising=False)
    else:
        monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", flag)
    runner, calls, _ = _sweep_runner()
    TPUModelRunner._precompile_vocab_parallel_target(runner)
    pairs = tpu_runner._spec_verify_warmup_pairs(runner)
    assert [c[:2] for c in calls if c[2] == 0] == pairs
    reference, ref_calls, _ = _sweep_runner()
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("TPU_PARALLEL_PRECOMPILE", raising=False)
    TPUModelRunner._precompile_vocab_parallel_target(reference)
    assert calls == ref_calls


# The served decode paddings at batch cap 48, and at cap 56 with a 224-token
# bucket (8, 16, 32, 56 requests) and a target-row bucket for its 168 drafts.
CAP48 = ([16, 32, 64, 96, 128, 160, 192, 256], [8, 16, 32, 48], 48)
CAP56 = ([16, 32, 64, 96, 128, 160, 192, 224, 256], [8, 16, 32, 56], 56)


@pytest.mark.parametrize(
    "trim_rows,paddings",
    [
        ((), CAP48),
        ((64, 96, 128, 160), CAP48),
        ((64, 96, 128, 160), CAP56),
        ((64, 96, 128, 160, 176), CAP56),
    ],
)
def test_precompile_covers_every_runtime_shape(monkeypatch, trim_rows, paddings):
    """Every (padded_logits_length, padded_num_reqs, verify_rows) the runtime
    can pick, over all draft counts, is warmed; so is the shard head at each
    verify length."""
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    runner, calls, heads = _sweep_runner(trim_rows)
    runner.num_tokens_paddings, runner.num_reqs_paddings, runner.max_num_reqs = paddings
    TPUModelRunner._precompile_vocab_parallel_target(runner)
    warmed = set(calls)
    assert len(warmed) == len(calls)  # one program a row bucket
    for num_tokens, num_reqs in tpu_runner._spec_verify_warmup_pairs(runner):
        assert (num_tokens, num_tokens) in heads
        for drafts in range(0, min(num_tokens, num_reqs * K) + 1):
            lengths = np.zeros(num_reqs, dtype=np.int32)
            lengths[: -(-drafts // K)] = K
            if drafts % K:
                lengths[drafts // K] = drafts % K
            rows = runner._verify_trim_rows(num_tokens, lengths) if trim_rows else 0
            assert (num_tokens, num_reqs, rows) in warmed
    if paddings is CAP56:
        # A full cap-56 step: 224 tokens, 168 drafts, trimmed only by the 176 bucket.
        full = runner._verify_trim_rows(224, np.full(56, K, dtype=np.int32))
        assert full == (176 if 176 in trim_rows else 0)
        assert (224, 56, full) in warmed


def test_precompile_restores_the_default_generator(monkeypatch):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    # The device's default generator, as torch.tpu.default_generators has it.
    monkeypatch.setattr(
        torch.cpu, "default_generators", (torch.default_generator,), raising=False
    )
    runner, _, _ = _sweep_runner(graph_rng=True)
    real_verify = runner.spec_verify_on_shard

    def drawing_verify(*args, **kwargs):
        torch.rand(5)  # an in-graph draw moves the default generator
        return real_verify(*args, **kwargs)

    runner.spec_verify_on_shard = drawing_verify
    torch.manual_seed(5)
    before = torch.get_rng_state()
    TPUModelRunner._precompile_vocab_parallel_target(runner)
    assert torch.equal(torch.get_rng_state(), before)


def test_rejection_sampler_sweep_never_warms_the_shard_verify(monkeypatch):
    """The shard verify stays out of the rotated rejection-sampler sweep,
    whose ranks reach each shape at different times."""
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("shard verify in the rotated sweep")

    class _Sampler:
        synthetic_mode = True

        def __call__(self, *, draft_token_ids, **kwargs):
            return draft_token_ids

    runner, _, _ = _sweep_runner()
    runner.spec_verify_on_shard = forbidden
    runner.compute_selected_shard_logits = forbidden
    runner.rejection_sampler = _Sampler()
    runner.vocab_size = 16
    runner.spec_bonus_and_target_logits = lambda d, b, t: (b, d)
    runner.spec_gather_bonus_and_target_logits = lambda d, b, t, **kw: (b, d)
    runner._greedy_head_owner = None
    runner._topk_candidate_blocks = lambda: 0
    runner._sampler_light_variants = lambda: []
    runner._sampler_fused_chain = False
    runner._sampler_verify_trim_rows = ()
    TPUModelRunner._precompile_rejection_sampler(runner)


def test_math_helpers_are_consistent():
    """The statistical test's own reference: the chi-square p-value is high
    for exact counts and tiny for a shifted sample."""
    expected = np.full(10, 1000.0)
    assert _chi_square_p(expected.astype(np.int64), expected) > 0.99
    shifted = expected.copy()
    shifted[0] += 300
    shifted[1] -= 300
    assert _chi_square_p(shifted.astype(np.int64), expected) < 1e-6
    assert math.isfinite(_chi_square_p(expected.astype(np.int64), expected))
