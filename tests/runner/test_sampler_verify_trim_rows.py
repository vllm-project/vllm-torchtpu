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
"""TPU_SAMPLER_VERIFY_TRIM_ROWS: the non-greedy spec verify on the 32-48
target rows that hold a decode step's real drafts, not the 64-row bucket.

Exactness: on the same logits and the same seeded generator, the selected
tokens with the trim equal the untrimmed tokens bit for bit, and the
generator ends in the same state. This holds for the fused chain (real
acceptance, host-drawn and in-graph uniforms) and for the unfused path
(synthetic acceptance, and real acceptance with the fused chain off), at
9-16 requests with MTP k=3 and with requests that carry fewer drafts. The
metadata comes from the real get_spec_decode_metadata, so the "real slots
first" layout the trim relies on is the production one.

Positive controls (each must change some output, else the equality proves
nothing): a bucket below the real draft count, and recover_u drawn at the
trimmed shape.

Coverage: the trimmed calls see the expected row counts, and every trimmed
call the runtime makes has a warmed shape in the precompile sweep.

Runs compiled on a TPU when one is attached, else eager on CPU.
"""

import contextlib
import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch._dynamo

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

K = 3
# The served padded request count at 9-16 requests. The exactness cases pad to
# one bucket more: a padding slot's segment id is the request count, which at
# 16 requests indexes past 16 per-request rows; XLA clamps that gather, eager
# CPU raises.
PADDED_NUM_REQS = 16
EXACT_PADDED_NUM_REQS = 32
TOKEN_PADDINGS = [16, 32, 64, 128]
TRIM_ROWS = (32, 40, 48)
RATES = [0.8, 0.66, 0.53]
# TPU_SAMPLER_VOCAB_PARALLEL + FAST_TOP_P (the served stack), and flags off.
VARIANTS = {"served": True, "default": False}
# Draft counts per request at a padded logits length of 64.
CASES = [[K] * n for n in range(9, 17)] + [
    [K] * 10 + [1, 0],
    [K] * 13 + [2, 2],
    [2] * 16,
]
SEEDS = range(3)


@pytest.fixture
def dev():
    try:
        import torch_tpu  # noqa: F401

        d = torch.device("tpu")
        torch.zeros(1, device=d)
        return d
    except Exception:
        return torch.device("cpu")


@pytest.fixture(autouse=True)
def _eager_off_tpu(dev, monkeypatch):
    if dev.type != "tpu":
        monkeypatch.setattr(
            rs,
            "_random_rejection_sample_with_segment",
            rs._random_rejection_sample_with_segment_impl,
        )


@pytest.fixture(autouse=True)
def _raise_dynamo_recompile_limit():
    cfg = torch._dynamo.config
    names = [n for n in ("recompile_limit", "cache_size_limit") if hasattr(cfg, n)]
    saved = {n: getattr(cfg, n) for n in names}
    for n in names:
        setattr(cfg, n, 1024)
    yield
    for n, v in saved.items():
        setattr(cfg, n, v)


def _method(name, device):
    fn = getattr(TPUModelRunner, name)
    if device.type != "tpu":
        fn = getattr(fn, "_torchdynamo_orig_callable", fn)
    return fn


class _RecordingSampler:
    """The real rejection sampler, recording the target rows and verify_rows
    of each non-greedy call."""

    def __init__(self, synthetic, device):
        spec = (
            SimpleNamespace(
                rejection_sample_method="synthetic", synthetic_acceptance_rates=RATES
            )
            if synthetic
            else None
        )
        self.real = rs.RejectionSampler(spec, device)
        self.synthetic_mode = self.real.synthetic_mode
        self.synthetic_conditional_rates = self.real.synthetic_conditional_rates
        self.calls: list[tuple[int, int]] = []

    def __call__(self, **kwargs):
        self.calls.append(
            (kwargs["target_logits"].shape[0], kwargs.get("verify_rows", 0))
        )
        return self.real(**kwargs)


def _case(case, seed, vocab, device, padded_num_reqs=EXACT_PADDED_NUM_REQS):
    """Logits, metadata and per-request sampling params of one verify chunk."""
    gen = torch.Generator().manual_seed(seed)
    rows = int(np.sum(np.asarray(case) + 1))
    padded = next(p for p in TOKEN_PADDINGS if p >= rows)
    # Peaked rows (scale up to 8) so drafts are often accepted and the later
    # slots matter; flat rows so top-p keeps many tokens and recovery happens.
    scale = torch.rand(padded, 1, generator=gen) * 8.0
    logits = (torch.randn(padded, vocab, generator=gen) * scale).to(torch.bfloat16)
    input_ids = torch.randint(0, vocab, (4096,), dtype=torch.int32, generator=gen)
    top = logits.float().argmax(dim=-1)
    pos = 0
    for n in case:
        for j in range(n):
            if torch.rand(1, generator=gen).item() < 0.7:
                input_ids[pos + j + 1] = int(top[pos + j])
        pos += n + 1
    nd = np.asarray(case, dtype=np.int32)
    runner = SimpleNamespace(
        input_ids_cpu=input_ids,
        arange_np=np.arange(4096, dtype=np.int64),
        num_tokens_paddings=TOKEN_PADDINGS,
        device=device,
    )
    md = SpeculativeDecodingManager(runner).get_spec_decode_metadata(
        nd, np.cumsum(nd + 1).astype(np.int32), padded_num_reqs
    )
    assert md.target_logits_indices.shape[0] == padded
    temps = torch.ones((padded_num_reqs, 1), dtype=logits.dtype)
    top_k = torch.zeros((padded_num_reqs, 1), dtype=torch.int32)
    top_p = torch.ones((padded_num_reqs, 1), dtype=torch.float32)
    top_p[: len(case)] = 0.95
    # One greedy request and one top-k request beside the served top-p rows.
    temps[1] = 0.0
    top_k[2] = 50
    params = tuple(t.to(device) for t in (temps, top_k, top_p))
    return logits.to(device), md, params


def _runner(device, synthetic, trim_rows, variant, fused=False, graph_rng=False):
    light = VARIANTS[variant]
    r = SimpleNamespace(
        rejection_sampler=_RecordingSampler(synthetic, device),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        _sampler_fused_chain=fused,
        _sampler_graph_rng=graph_rng,
        _sampler_verify_trim=True,
        _sampler_verify_trim_rows=tuple(trim_rows),
        _sampler_light=light,
        _sampler_need_top_k=True,
        _sampler_need_top_p=True,
        _sampler_radix_bits=1,
        _sampler_fast_top_p=light,
        _sampler_topk_candidates=lambda *args: 0,
        fused_rows=[],
    )
    for name in ("_sampler_kwargs", "_verify_trim_buckets", "_verify_trim_rows"):
        setattr(r, name, functools.partial(getattr(TPUModelRunner, name), r))
    r.sample_from_logits_func = functools.partial(
        _method("sample_from_logits", device), r
    )
    r.spec_gather_bonus_and_target_logits = functools.partial(
        _method("spec_gather_bonus_and_target_logits", device), r
    )
    fused_fn = functools.partial(_method("spec_sample_verify_fused", device), r)

    def recording_fused(*args, **kwargs):
        r.fused_rows.append(kwargs.get("verify_rows", 0))
        return fused_fn(*args, **kwargs)

    r.spec_sample_verify_fused = recording_fused
    r._get_sampling_generator = lambda: None
    return r


def _verify(r, logits, md, params, seed, device, graph_rng=False):
    """One verify chunk through _sample_spec_verify_chunk; the tokens and the
    generator state after it."""
    temps, top_k, top_p = params
    r._build_padded_sampling_params = lambda start, end, logits, **kw: params
    if graph_rng:
        torch.manual_seed(seed)
        gen = None
    else:
        gen = torch.Generator(device=device).manual_seed(seed)
    out = TPUModelRunner._sample_spec_verify_chunk(
        r,
        logits,
        md,
        None,
        SimpleNamespace(),
        0,
        int(md.draft_lengths_cpu.shape[0]),
        False,
        gen,
    )
    state = torch.get_rng_state() if graph_rng else gen.get_state()
    return out.cpu(), state


def _vocab(device):
    # A short vocabulary off TPU: the trim is per row, independent of its width.
    return 154880 if device.type == "tpu" else 128 * 16


PATHS = {
    "synthetic-unfused": {"synthetic": True},
    "real-unfused": {"synthetic": False},
    "real-fused": {"synthetic": False, "fused": True},
    "real-fused-graph-rng": {"synthetic": False, "fused": True, "graph_rng": True},
}


def _expected_rows(case):
    drafts = sum(case)
    return next(r for r in TRIM_ROWS if drafts <= r)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("case", CASES, ids=[str(sum(c)) for c in CASES])
def test_trim_is_exact(dev, case, path, variant):
    vocab = _vocab(dev)
    kw = PATHS[path]
    graph_rng = kw.get("graph_rng", False)
    for seed in SEEDS:
        logits, md, params = _case(case, seed, vocab, dev)
        off = _runner(dev, trim_rows=(), variant=variant, **kw)
        on = _runner(dev, trim_rows=TRIM_ROWS, variant=variant, **kw)
        tokens_off, state_off = _verify(off, logits, md, params, seed, dev, graph_rng)
        tokens_on, state_on = _verify(on, logits, md, params, seed, dev, graph_rng)
        assert torch.equal(tokens_off, tokens_on), (case, seed, path)
        assert torch.equal(state_off, state_on), (case, seed, path)
        rows = _expected_rows(case)
        if kw.get("fused"):
            assert off.fused_rows == [0] and on.fused_rows == [rows]
        else:
            assert off.rejection_sampler.calls == [(64, 0)]
            assert on.rejection_sampler.calls == [(rows, rows)]


@pytest.mark.parametrize("path", list(PATHS))
def test_tokens_cover_rejection_and_bonus(dev, path):
    """The cases reach rejected slots and fully accepted requests, so the
    equality above compares recovered and bonus tokens, not only drafts."""
    vocab, kw = _vocab(dev), PATHS[path]
    rejected = bonus = 0
    for case in CASES[:8]:
        for seed in SEEDS:
            logits, md, params = _case(case, seed, vocab, dev)
            r = _runner(dev, trim_rows=TRIM_ROWS, variant="served", **kw)
            tokens, _ = _verify(r, logits, md, params, seed, dev, kw.get("graph_rng"))
            filled = (tokens[: len(case)] != rs.PLACEHOLDER_TOKEN_ID).sum(dim=1)
            rejected += int((filled <= K).sum())
            bonus += int((filled == K + 1).sum())
    assert rejected > 0 and bonus > 0


def test_positive_control_bucket_below_the_drafts_is_caught(dev):
    """verify_rows below the real slot count drops real draft slots."""
    vocab, differs = _vocab(dev), 0
    case = [K] * 12
    for seed in SEEDS:
        logits, md, params = _case(case, seed, vocab, dev)
        good = _runner(dev, synthetic=True, trim_rows=TRIM_ROWS, variant="served")
        bad = _runner(dev, synthetic=True, trim_rows=TRIM_ROWS, variant="served")
        bad._verify_trim_rows = lambda n, d: 32
        tokens, _ = _verify(good, logits, md, params, seed, dev)
        wrong, _ = _verify(bad, logits, md, params, seed, dev)
        differs += int(not torch.equal(tokens, wrong))
    assert differs > 0


def test_positive_control_trimmed_draw_shape_is_caught(dev, monkeypatch):
    """recover_u drawn at the trimmed shape leaves the generator elsewhere,
    so every later draw would differ: why the unfused path draws at the full
    shape. The end-state comparison above must see it."""
    vocab, differs = _vocab(dev), 0
    case = [K] * 9
    real_rand = torch.rand

    def rand_at_trimmed_rows(size, *args, **kwargs):
        if len(size) == 2 and size[0] == 64:
            size = (32, size[1])
        return real_rand(size, *args, **kwargs)

    for seed in SEEDS:
        logits, md, params = _case(case, seed, vocab, dev)
        r = _runner(dev, synthetic=False, trim_rows=TRIM_ROWS, variant="served")
        _, state = _verify(r, logits, md, params, seed, dev)
        with monkeypatch.context() as m:
            m.setattr(torch, "rand", rand_at_trimmed_rows)
            r = _runner(dev, synthetic=False, trim_rows=TRIM_ROWS, variant="served")
            _, wrong = _verify(r, logits, md, params, seed, dev)
        differs += int(not torch.equal(state, wrong))
    assert differs > 0


@pytest.mark.parametrize(
    "padded,drafts,flag,rows,expect",
    [
        (64, [3] * 9, True, TRIM_ROWS, 32),
        (64, [3] * 10, False, TRIM_ROWS, 32),
        (64, [3] * 11, True, TRIM_ROWS, 40),
        (64, [3] * 13, True, TRIM_ROWS, 40),
        (64, [3] * 14, True, TRIM_ROWS, 48),
        (64, [3] * 16, True, TRIM_ROWS, 48),
        (64, [3] * 16 + [1], True, TRIM_ROWS, 0),  # 49 real slots
        (64, [1], True, TRIM_ROWS, 32),
        (32, [3] * 8, True, TRIM_ROWS, 0),  # no bucket below 32 holds 24
        (32, [2, 2], True, TRIM_ROWS, 8),
        (16, [3], True, TRIM_ROWS, 8),
        (64, [1], True, (), 0),  # switch off: today's gate
        (64, [3] * 9, False, (), 0),
    ],
)
def test_verify_trim_rows_gate(padded, drafts, flag, rows, expect):
    r = SimpleNamespace(_sampler_verify_trim=flag, _sampler_verify_trim_rows=rows)
    r._verify_trim_buckets = functools.partial(TPUModelRunner._verify_trim_buckets, r)
    assert TPUModelRunner._verify_trim_rows(r, padded, np.asarray(drafts)) == expect


@pytest.mark.parametrize(
    "value,expect", [(None, []), ("", []), ("48, 32,40", [32, 40, 48]), ("40,40", [40])]
)
def test_env_parses_buckets(monkeypatch, value, expect):
    if value is None:
        monkeypatch.delenv("TPU_SAMPLER_VERIFY_TRIM_ROWS", raising=False)
    else:
        monkeypatch.setenv("TPU_SAMPLER_VERIFY_TRIM_ROWS", value)
    assert envs.environment_variables["TPU_SAMPLER_VERIFY_TRIM_ROWS"]() == expect


class _SignatureSampler:
    """Records each non-greedy rejection-sampler call by what keys its
    compiled graph: tensor shapes and the static kwargs."""

    synthetic_mode = True
    synthetic_conditional_rates = None

    def __init__(self):
        self.signatures: set[tuple] = set()

    def __call__(self, **kwargs):
        if kwargs.get("do_sampling"):
            shapes = tuple(
                (k, tuple(v.shape))
                for k, v in sorted(kwargs.items())
                if isinstance(v, torch.Tensor)
            )
            statics = tuple(
                (k, v)
                for k, v in sorted(kwargs.items())
                if not isinstance(v, torch.Tensor) and k != "topk_candidates"
            )
            self.signatures.add(shapes + statics)
        return torch.zeros(
            (kwargs["num_draft_tokens"].shape[0], K + 1), dtype=torch.int32
        )


def _gather_shapes(calls):
    def gather(logits, bonus_logits_indices, target_logits_indices, verify_rows=0):
        calls.add((logits.shape[0], target_logits_indices.shape[0], verify_rows))
        rows = verify_rows or target_logits_indices.shape[0]
        return logits[: bonus_logits_indices.shape[0]], logits[:rows]

    return gather


def _sweep_runner(trim_rows, sampler, gathers, fused_rows):
    vocab = 8
    r = SimpleNamespace(
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        rejection_sampler=sampler,
        num_tokens_paddings=TOKEN_PADDINGS[:3],
        num_reqs_paddings=[4, 8, 16],
        max_num_reqs=PADDED_NUM_REQS,
        vocab_size=vocab,
        _hidden_states_dtype=torch.float32,
        device="cpu",
        _precompile_timed=lambda name: contextlib.nullcontext(),
        _greedy_head_owner=None,
        _topk_candidate_blocks=lambda: 0,
        _sampler_light=True,
        _sampler_fast_top_p=True,
        _sampler_radix_bits=1,
        _sampler_need_top_k=False,
        _sampler_need_top_p=True,
        _sampler_fused_chain=True,
        _sampler_graph_rng=False,
        _sampler_verify_trim=True,
        _sampler_verify_trim_rows=tuple(trim_rows),
        spec_bonus_and_target_logits=lambda logits, b, t: (b, logits),
        spec_gather_bonus_and_target_logits=_gather_shapes(gathers),
        spec_sample_verify_fused=lambda *a, **kw: fused_rows.add(
            (a[2].shape[0], kw.get("verify_rows", 0))
        ),
        _sampler_topk_candidates=lambda *args: 0,
    )
    for name in (
        "_sampler_light_variants",
        "_sampler_kwargs",
        "_verify_trim_buckets",
        "_verify_trim_rows",
        "_spec_verify_bucket_reachable",
        "_precompile_spec_sample_verify_fused",
        "_precompile_trimmed_rejection_sampler",
    ):
        setattr(r, name, functools.partial(getattr(TPUModelRunner, name), r))
    return r


@pytest.mark.parametrize("trim_rows", [(), TRIM_ROWS])
def test_runtime_trimmed_calls_are_precompiled(monkeypatch, trim_rows):
    """Every unfused rejection-sampler call and target gather of a synthetic
    decode step at 9-16 requests has a warmed shape, and the fused chain is
    warmed at every bucket. With the switch off nothing trimmed is warmed."""
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    monkeypatch.delenv("TPU_PARALLEL_PRECOMPILE", raising=False)
    warm_sampler, warm_gathers, warm_fused = _SignatureSampler(), set(), set()
    warm = _sweep_runner(trim_rows, warm_sampler, warm_gathers, warm_fused)
    TPUModelRunner._precompile_rejection_sampler(warm)  # type: ignore[arg-type]
    trimmed = {rows for _, _, rows in warm_gathers if rows}
    assert trimmed == ({8, *trim_rows} if trim_rows else set())
    assert {rows for _, rows in warm_fused if rows} == {8, *trim_rows}

    run_sampler, run_gathers = _SignatureSampler(), set()
    run = _sweep_runner(trim_rows, run_sampler, run_gathers, set())
    run._sampler_fused_chain = False
    vocab = run.vocab_size
    # One row more than the padded requests: eager CPU raises on the padding
    # slots' sentinel segment, which XLA clamps.
    params = (
        torch.ones((PADDED_NUM_REQS + 1, 1)),
        torch.zeros((PADDED_NUM_REQS + 1, 1), dtype=torch.int32),
        torch.ones((PADDED_NUM_REQS + 1, 1)),
    )
    run._build_padded_sampling_params = lambda start, end, logits, **kw: params
    run.sample_from_logits_func = lambda logits, *a, **kw: torch.zeros(
        (logits.shape[0], 1), dtype=torch.int64
    )
    for case in CASES:
        _, md, _ = _case(case, 0, vocab, torch.device("cpu"), PADDED_NUM_REQS)
        logits = torch.zeros((md.target_logits_indices.shape[0], vocab))
        TPUModelRunner._sample_spec_verify_chunk(
            run,
            logits,
            md,
            None,
            SimpleNamespace(),
            0,
            len(case),
            False,
            torch.Generator().manual_seed(0),
        )
    assert run_gathers <= warm_gathers
    assert run_sampler.signatures <= warm_sampler.signatures
    expect = {(64, 64, 0)} if not trim_rows else {(64, 64, r) for r in TRIM_ROWS}
    assert run_gathers == expect
