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
"""TPU_SAMPLER_VERIFY_VOCAB_PARALLEL: the vocab-parallel verify
(vocab_parallel.spec_verify) against the usual verify on the full logits
(tpu_runner._sample_from_logits_impl for the bonus token,
rejection_sampler._random_rejection_sample_with_segment_impl for the drafts).

The TP ranks are simulated on CPU: the vocabulary is split into ``TP``
shards of ``WIDTH`` columns, the last shard ending in padding columns (set to
a large logit, so a padding column that leaks into a result shows up), and
each rank runs ``spec_verify`` in its own thread. The collectives are
thread-barrier sums and stacks of the ranks' tensors.

What is checked:

- Greedy rows give exactly the usual tokens, with ties inside and across
  shards (logits on a coarse integer grid).
- Sampling: over many draws, the bonus token, the first emitted token of a
  request with drafts (accept or recovered: by construction of speculative
  sampling it follows the target distribution), and the recovered token alone
  (synthetic rejection) follow the expected distribution, computed from the
  usual top-k / top-p mask on the full logits. The usual sampler passes the
  same test, as a control of the test itself.
- Positive control: noise that repeats across shards (each rank hashing its
  local column index instead of the global token id) must fail the test.
- The top-k cutoff and the top-p threshold searches keep the same entries as
  top_k_top_p.topk_mask and topp_mask.
"""

import math
import threading

import torch

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.layers.adapter.sample import top_k_top_p
from vllm_torchtpu.layers.adapter.sample import vocab_parallel as vp

TP = 4
WIDTH = 10
VOCAB = 37  # 3 padding columns on the last shard
PAD_LOGIT = 50.0
K = 3


class _ThreadComm:
    """All-reduce (sum) and all-gather (stack by rank) between TP threads."""

    def __init__(self, tp: int):
        self.tp = tp
        self.barrier = threading.Barrier(tp)
        self.slots: list[torch.Tensor | None] = [None] * tp

    def _exchange(self, rank: int, x: torch.Tensor) -> torch.Tensor:
        self.slots[rank] = x
        self.barrier.wait()
        out = torch.stack([s for s in self.slots if s is not None])
        self.barrier.wait()
        return out

    def for_rank(self, rank: int):
        return (
            lambda x: self._exchange(rank, x).sum(dim=0),
            lambda x: self._exchange(rank, x),
        )


def _run_ranks(fn):
    """Runs fn(rank, all_reduce, all_gather) on TP threads; returns the
    results by rank, re-raising the first failure."""
    comm = _ThreadComm(TP)
    results: list = [None] * TP
    errors: list = []

    def target(rank):
        try:
            results[rank] = fn(rank, *comm.for_rank(rank))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            comm.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(TP)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def _shard(full: torch.Tensor, rank: int) -> torch.Tensor:
    """Rank's [rows, WIDTH] logits shard of full [rows, VOCAB] logits."""
    padded = torch.full(
        (full.shape[0], TP * WIDTH), PAD_LOGIT, dtype=full.dtype, device=full.device
    )
    padded[:, :VOCAB] = full
    return padded[:, rank * WIDTH : (rank + 1) * WIDTH].contiguous()


def _layout(draft_counts: list[int]):
    """Rows of a verify chunk, laid out request-major: each request's draft
    rows, then its bonus row (as get_spec_decode_metadata lays them out)."""
    target, bonus, segment, group = [], [], [], []
    row = 0
    for b, n in enumerate(draft_counts):
        for j in range(n):
            target.append(row)
            segment.append(b)
            group.append(j)
            row += 1
        bonus.append(row)
        row += 1
    return {
        "rows": row,
        "target": torch.tensor(target, dtype=torch.int32),
        "bonus": torch.tensor(bonus, dtype=torch.int32),
        "segment": torch.tensor(segment, dtype=torch.int64),
        "group": torch.tensor(group, dtype=torch.int32),
        "num_draft": torch.tensor(draft_counts, dtype=torch.int32),
    }


def _vocab_parallel(
    logits,
    lay,
    drafts,
    temps,
    top_k,
    top_p,
    accept_u,
    seeds,
    rates=None,
    need_top_k=True,
    need_top_p=True,
):
    def fn(rank, all_reduce, all_gather):
        start = rank * WIDTH
        valid = max(0, min(WIDTH, VOCAB - start))
        return vp.spec_verify(
            _shard(logits, rank),
            torch.tensor(start, dtype=torch.int32),
            torch.tensor(valid, dtype=torch.int32),
            lay["bonus"],
            lay["target"],
            temps,
            top_k,
            top_p,
            drafts,
            lay["num_draft"],
            lay["segment"],
            lay["group"],
            accept_u,
            seeds,
            K,
            rates,
            rates is not None,
            all_reduce,
            all_gather,
            need_top_k=need_top_k,
            need_top_p=need_top_p,
        )

    out = _run_ranks(fn)
    for other in out[1:]:
        assert torch.equal(other, out[0]), "ranks disagree"
    return out[0]


def _usual(logits, lay, drafts, temps, top_k, top_p, generator, rates=None):
    from vllm_torchtpu.runner.tpu_runner import _sample_from_logits_impl

    bonus_logits = logits[lay["bonus"].long()]
    target_logits = logits[lay["target"].long()]
    bonus_u = torch.rand(bonus_logits.shape, generator=generator)
    accept_u = torch.rand(drafts.shape, generator=generator)
    recover_u = torch.rand(target_logits.shape, generator=generator)
    bonus = _sample_from_logits_impl(
        bonus_logits, temps, bonus_u, top_k, top_p, fast_top_p=True
    ).view(-1)
    seg = lay["segment"]
    return rs._random_rejection_sample_with_segment_impl(
        drafts,
        target_logits,
        lay["num_draft"],
        bonus,
        seg,
        lay["group"],
        K,
        temps[seg],
        top_k[seg],
        top_p[seg],
        accept_u,
        recover_u,
        rates,
        rates is not None,
        fast_top_p=True,
    )


def _params(n, temp, k, p, dtype=torch.bfloat16):
    return (
        torch.full((n, 1), temp, dtype=dtype),
        torch.full((n, 1), k, dtype=torch.int32),
        torch.full((n, 1), p, dtype=torch.float32),
    )


def _expected(row_logits, temp, k, p, exclude=None):
    """Expected distribution of one row: the usual top-k / top-p mask on the
    full logits (verify-row arithmetic: float32, temperature + eps), softmax,
    optionally with one token removed and renormalized."""
    x = row_logits.float().unsqueeze(0) / (temp + rs.TEMPERATURE_EPS)
    masked = top_k_top_p.apply_top_k_top_p(
        x,
        torch.tensor([[k]], dtype=torch.int32),
        torch.tensor([[p]]),
        fast_top_p=True,
    )
    probs = torch.softmax(masked, dim=-1).squeeze(0).double()
    if exclude is not None:
        probs[exclude] = 0.0
    return probs / probs.sum()


def _chi_square_ok(counts: torch.Tensor, probs: torch.Tensor) -> bool:
    """Goodness of fit of counts to probs at significance 1e-4. Cells with an
    expected count below 5 are pooled; a count in a zero-probability cell
    fails outright."""
    n = int(counts.sum())
    if bool(((probs == 0) & (counts > 0)).any()):
        return False
    expected = probs * n
    big = expected >= 5
    obs = torch.cat([counts[big].double(), counts[~big].double().sum().view(1)])
    exp = torch.cat([expected[big], expected[~big].sum().view(1)])
    keep = exp > 0
    stat = float((((obs - exp) ** 2)[keep] / exp[keep]).sum())
    df = int(keep.sum()) - 1
    z = 3.719  # one-sided normal quantile at 1e-4
    critical = df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3
    return stat < critical


def _random_logits(rows, generator, dtype=torch.bfloat16):
    return (torch.randn((rows, VOCAB), generator=generator) * 2.0).to(dtype)


def test_greedy_rows_match_exactly():
    """Greedy requests (temperature 0) emit the usual tokens, ties included,
    alongside sampled requests in the same chunk."""
    g = torch.Generator().manual_seed(0)
    draft_counts = [3, 3, 2, 0, 3, 1, 3, 3]
    lay = _layout(draft_counts)
    greedy = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0], dtype=torch.bool)
    for trial in range(20):
        # Coarse grid: many equal maxima, inside and across shards.
        logits = torch.randint(-3, 4, (lay["rows"], VOCAB), generator=g).to(
            torch.bfloat16
        )
        drafts = torch.randint(0, VOCAB, (lay["target"].shape[0],), generator=g)
        drafts = drafts.to(torch.int32)
        # Some drafts are the argmax, so greedy requests accept a few.
        argmax = torch.argmax(logits[lay["target"].long()], dim=-1).to(torch.int32)
        drafts = torch.where(
            torch.rand(drafts.shape, generator=g) < 0.5, argmax, drafts
        )
        drafts[0] = -1  # a placeholder draft
        temps, top_k, top_p = _params(len(draft_counts), 0.7, 0, 0.9)
        temps[greedy] = 0.0
        seeds = torch.randint(0, 1 << 24, (2,), generator=g, dtype=torch.int32)
        accept_u = torch.rand(drafts.shape, generator=g)
        got = _vocab_parallel(logits, lay, drafts, temps, top_k, top_p, accept_u, seeds)
        want = _usual(logits, lay, drafts, temps, top_k, top_p, g)
        assert torch.equal(got[greedy], want[greedy]), f"trial {trial}"


def _bonus_counts(sampler, logits_row, temp, k, p, draws, g):
    n = 64
    lay = _layout([0] * n)
    logits = logits_row.unsqueeze(0).expand(n, VOCAB).contiguous()
    temps, top_k, top_p = _params(n, temp, k, p)
    drafts = torch.zeros((0,), dtype=torch.int32)
    counts = torch.zeros(VOCAB, dtype=torch.long)
    for _ in range(draws // n):
        if sampler == "usual":
            out = _usual(logits, lay, drafts, temps, top_k, top_p, g)
        else:
            seeds = torch.randint(0, 1 << 24, (2,), generator=g, dtype=torch.int32)
            out = _vocab_parallel(
                logits, lay, drafts, temps, top_k, top_p, torch.zeros(0), seeds
            )
        counts += torch.bincount(out[:, 0].long(), minlength=VOCAB)
    return counts


def _bonus_expected(logits_row, temp, k, p):
    """The usual bonus arithmetic: scaling and probabilities in bfloat16."""
    x = logits_row.unsqueeze(0) / torch.tensor([[temp]], dtype=torch.bfloat16)
    masked = top_k_top_p.apply_top_k_top_p(
        x,
        torch.tensor([[k]], dtype=torch.int32),
        torch.tensor([[p]]),
        fast_top_p=True,
    )
    probs = torch.softmax(masked.double(), dim=-1).squeeze(0)
    return probs / probs.sum()


def test_bonus_token_distribution():
    g = torch.Generator().manual_seed(1)
    row = _random_logits(1, g)[0]
    for temp, k, p in [(1.0, 0, 0.95), (0.8, 6, 1.0), (0.6, 8, 0.8)]:
        want = _bonus_expected(row, temp, k, p)
        for sampler in ("usual", "vocab_parallel"):
            counts = _bonus_counts(sampler, row, temp, k, p, 6400, g)
            assert _chi_square_ok(counts, want), (sampler, temp, k, p, counts)


def _first_token_counts(row, draft, temp, k, p, draws, g, rates=None):
    n = 32
    lay = _layout([K] * n)
    logits = row.unsqueeze(0).expand(lay["rows"], VOCAB).contiguous()
    temps, top_k, top_p = _params(n, temp, k, p)
    drafts = torch.full((n * K,), draft, dtype=torch.int32)
    counts = {"usual": torch.zeros(VOCAB, dtype=torch.long)}
    counts["vocab_parallel"] = torch.zeros(VOCAB, dtype=torch.long)
    for _ in range(draws // n):
        out = _usual(logits, lay, drafts, temps, top_k, top_p, g, rates)
        counts["usual"] += torch.bincount(out[:, 0].long(), minlength=VOCAB)
        seeds = torch.randint(0, 1 << 24, (2,), generator=g, dtype=torch.int32)
        accept_u = torch.rand(drafts.shape, generator=g)
        out = _vocab_parallel(
            logits, lay, drafts, temps, top_k, top_p, accept_u, seeds, rates
        )
        counts["vocab_parallel"] += torch.bincount(out[:, 0].long(), minlength=VOCAB)
    return counts


def test_first_token_follows_target():
    """Accept a draft with probability p(draft), else recover from the
    residual: the first emitted token then follows p."""
    g = torch.Generator().manual_seed(2)
    row = _random_logits(1, g)[0]
    order = torch.argsort(row.float(), descending=True)
    for temp, k, p, draft in [
        (1.0, 0, 0.95, int(order[0])),
        (1.0, 0, 0.9, int(order[3])),
        (0.7, 5, 0.9, int(order[1])),
    ]:
        want = _expected(row, temp, k, p)
        counts = _first_token_counts(row, draft, temp, k, p, 6400, g)
        for sampler, c in counts.items():
            assert _chi_square_ok(c, want), (sampler, temp, k, p, draft, c)


def test_recovered_token_distribution():
    """Synthetic rejection at the first slot (rate 0): the first emitted token
    is always the recovered token, drawn from p without the draft."""
    g = torch.Generator().manual_seed(3)
    row = _random_logits(1, g)[0]
    draft = int(torch.argmax(row.float()))
    rates = torch.tensor([0.0, 0.5, 0.5])
    want = _expected(row, 1.0, 0, 0.95, exclude=draft)
    counts = _first_token_counts(row, draft, 1.0, 0, 0.95, 6400, g, rates)
    for sampler, c in counts.items():
        assert _chi_square_ok(c, want), (sampler, c)


def test_positive_control_repeated_noise_fails():
    """Two equal tokens at the same local column of two shards: with
    independent noise each wins half the draws; with noise that repeats
    across shards the lower shard always wins, and the test must see it."""
    row = torch.full((VOCAB,), -30.0, dtype=torch.bfloat16)
    row[2] = 1.0
    row[WIDTH + 2] = 1.0
    want = _bonus_expected(row, 1.0, 0, 1.0)
    g = torch.Generator().manual_seed(4)
    assert _chi_square_ok(_bonus_counts("vp", row, 1.0, 0, 1.0, 1280, g), want)

    original = vp.hashed_uniform
    vp.hashed_uniform = lambda seeds, rows, ids: original(seeds, rows, ids - ids[0])
    try:
        counts = _bonus_counts("vp", row, 1.0, 0, 1.0, 1280, g)
    finally:
        vp.hashed_uniform = original
    assert not _chi_square_ok(counts, want), counts


def test_threshold_searches_keep_the_same_entries():
    g = torch.Generator().manual_seed(5)
    rows = 24
    x = torch.randint(-6, 7, (rows, VOCAB), generator=g).float() * 0.5
    k = torch.randint(1, VOCAB + 3, (rows,), generator=g, dtype=torch.int32)
    want_k = top_k_top_p.topk_mask(x, k.unsqueeze(-1), float("-inf")) > float("-inf")
    probs = torch.softmax(torch.randn((rows, VOCAB), generator=g) * 2, dim=-1)
    p = torch.rand((rows,), generator=g) * 0.9 + 0.05
    threshold = top_k_top_p._topp_bisect_threshold(probs, p, 1)
    want_p = probs >= threshold.unsqueeze(-1)

    def fn(rank, all_reduce, all_gather):
        cols = slice(rank * WIDTH, min(VOCAB, (rank + 1) * WIDTH))
        cutoff = vp._top_k_cutoff(x[:, cols], k, all_reduce, 1)
        t, _ = vp._top_p_threshold(
            probs[:, cols],
            p,
            torch.zeros(rows, dtype=torch.bool),
            torch.float32,
            all_reduce,
            5,
        )
        return x[:, cols] >= cutoff.unsqueeze(-1), probs[:, cols] >= t.unsqueeze(-1)

    out = _run_ranks(fn)
    assert torch.equal(torch.cat([o[0] for o in out], dim=-1), want_k)
    # The sharded masses are summed in another order than the full-row ones;
    # with random p no boundary mass lies within rounding of p.
    assert torch.equal(torch.cat([o[1] for o in out], dim=-1), want_p)


def test_hashed_uniform_is_uniform_and_seeded():
    ids = torch.arange(1000, dtype=torch.int32)
    seeds = torch.tensor([123, 456], dtype=torch.int32)
    u = vp.hashed_uniform(seeds, 64, ids)
    assert float(u.min()) > 0.0 and float(u.max()) < 1.0
    assert bool(torch.isfinite(-torch.log(-torch.log(u))).all())
    assert abs(float(u.mean()) - 0.5) < 0.005
    assert abs(float(u.var()) - 1 / 12) < 0.002
    assert torch.equal(u, vp.hashed_uniform(seeds, 64, ids))
    other = vp.hashed_uniform(torch.tensor([124, 456], dtype=torch.int32), 64, ids)
    assert float((u == other).float().mean()) < 0.001
