# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_SPEC_TAIL_COMPILED, ported from dongruizi's 997662eac + 23f7f74bb (q38 DEC-FB-3 P-b'), CPU only.

The port's own test (tests/runner/test_mtp_async_contracts.py) runs on TPU only, so this file checks the port on CPU:
- the one-chunk spec tail equals the separate assembly plus sample_tokens' bound pad, bit for bit, over every request
  count 1..64 (exact eagle3 drafts, and a bucketed drafter whose draft rows differ from the sampler's);
- `_precompile_spec_tail` compiles every shape key the call site produces for those counts;
- flag off, the assembler is fb1f4ef44's (verbatim copy in postfwd_ref_fb1f4ef44.py);
- the compiled tail is wrapped with the owned-alias stamp strip (dongruizi's 1e044e7d7), not eagle3's view.
"""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import postfwd_ref_fb1f4ef44 as ref
from vllm_torchtpu.runner import owned_args, tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _get_padded_num_reqs_with_upper_limit,
    _get_padded_token_len,
)
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID

K = 3
STRIDE = 1 + K
MAX_REQS = 64
BOUND = MAX_REQS * STRIDE
PADDINGS = [16, 32, 64, 128, 256, 512]


def rows_of(R):
    return _get_padded_num_reqs_with_upper_limit(R, MAX_REQS)


def _flag(monkeypatch, on):
    if on:
        monkeypatch.setenv("VLLM_TPU_SPEC_TAIL_COMPILED", "1")
    else:
        monkeypatch.delenv("VLLM_TPU_SPEC_TAIL_COMPILED", raising=False)


def _runner(bucketed):
    r = object.__new__(TPUModelRunner)
    r.drafter = SimpleNamespace(bucketed_drafts=bucketed)
    r.speculative_config = SimpleNamespace(num_speculative_tokens=K)
    r.scheduler_config = SimpleNamespace(async_scheduling=True)
    r.max_num_reqs = MAX_REQS
    r.device = torch.device("cpu")
    return r


def _chunk(g, R, bucketed, md_none=False):
    """One verify chunk: the sampler's [rows, 1+K] output (INVALID-padded, filler in the padded rows), the drafts
    (exact [R, K] for eagle3; [p, K] at the draft-loop bucket for a bucketed drafter, p != rows for small R) and
    the spec metadata."""
    rows = rows_of(R)
    nt = np.full((rows, STRIDE), INVALID_TOKEN_ID, dtype=np.int32)
    acc = g.integers(0, K + 1, R)
    for i in range(R):
        nt[i, : acc[i] + 1] = g.integers(1, 150000, acc[i] + 1)
    nt[R:] = g.integers(1, 150000, (rows - R, STRIDE))
    p = _get_padded_token_len(PADDINGS, R) if bucketed else R
    drafts = torch.from_numpy(g.integers(1, 150000, (p, K)).astype(np.int32))
    dl = np.zeros(rows, dtype=np.int32)
    dl[:R] = g.choice([K, K, K, 0, 1, 2], R)
    md = None if md_none else SimpleNamespace(draft_lengths=torch.from_numpy(dl), draft_lengths_cpu=dl[:R].copy())
    return torch.from_numpy(nt), drafts, md


def _python_tail(*args):
    """The compiled program's body (`_spec_tail`), run eagerly on CPU."""
    return tpu_runner._spec_tail(*args)


def _assemble(runner, drafts, nts, mds, Rs, old=False):
    state = SimpleNamespace(num_reqs_list=list(Rs), spec_decode_metadata_list=list(mds))
    fn = ref.old_assemble_async_spec_substitution if old else TPUModelRunner._assemble_async_spec_substitution
    chunks, idx, nr, ndpr = fn(runner, drafts, list(nts), state)
    return ref.old_bounded_async_source(runner, chunks, nr), idx, nr, ndpr


def _same(a, b, what):
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    assert torch.equal(a, b), what


@pytest.mark.parametrize("bucketed", [False, True])
def test_spec_tail_equals_separate_assembly_for_every_count(monkeypatch, bucketed):
    monkeypatch.setattr(tpu_runner, "_spec_tail_compiled", _python_tail)
    g = np.random.default_rng(17 + bucketed)
    for R in range(1, MAX_REQS + 1):
        nt, d, md = _chunk(g, R, bucketed)
        drafts = [d] if bucketed else d
        _flag(monkeypatch, False)
        sep = _assemble(_runner(bucketed), drafts, [nt], [md], (R,))
        _flag(monkeypatch, True)
        tail = _assemble(_runner(bucketed), drafts, [nt], [md], (R,))
        _same(tail[0], sep[0], f"bounded source R={R}")
        _same(tail[2], sep[2], f"num_rejected R={R}")
        assert tail[1] == sep[1] and tail[3] == sep[3], R


def test_spec_tail_only_for_one_verified_chunk(monkeypatch):
    """Several chunks, or a chunk without spec metadata, keep the separate calls."""
    calls = []
    monkeypatch.setattr(tpu_runner, "_spec_tail_compiled", lambda *a: calls.append(a) or _python_tail(*a))
    _flag(monkeypatch, True)
    g = np.random.default_rng(5)
    nt1, d1, md1 = _chunk(g, 3, False)
    nt2, d2, md2 = _chunk(g, 5, False)
    _assemble(_runner(False), torch.cat([d1, d2]), [nt1, nt2], [md1, md2], (3, 5))
    nt3, d3, _ = _chunk(g, 4, False)
    _assemble(_runner(False), d3, [nt3], [None], (4,))
    assert calls == []


@pytest.mark.parametrize("bucketed", [False, True])
def test_precompile_covers_every_call_site_shape(monkeypatch, bucketed):
    def key(args):
        return tuple((tuple(a.shape), a.dtype) if isinstance(a, torch.Tensor) else a for a in args)

    seen = []
    monkeypatch.setattr(tpu_runner, "_spec_tail_compiled", lambda *a: seen.append(key(a)) or _python_tail(*a))
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: None)
    _flag(monkeypatch, True)
    runner = _runner(bucketed)
    runner._precompile_timed = lambda name: contextlib.nullcontext()
    TPUModelRunner._precompile_spec_tail(runner)
    warmed = set(seen)
    seen.clear()
    g = np.random.default_rng(29)
    for R in range(1, MAX_REQS + 1):
        nt, d, md = _chunk(g, R, bucketed=False)
        if bucketed:
            d = torch.from_numpy(g.integers(1, 150000, (rows_of(R), K)).astype(np.int32))
        _assemble(runner, [d] if bucketed else d, [nt], [md], (R,))
    served = set(seen)
    assert len(served) == (MAX_REQS if not bucketed else len({rows_of(R) for R in range(1, MAX_REQS + 1)}))
    missing = served - warmed
    assert not missing, f"call-site shapes the precompile does not compile: {sorted(missing)[:3]}"


@pytest.mark.parametrize("bucketed", [False, True])
@pytest.mark.parametrize("Rs", [(1,), (5,), (16,), (33,), (64,), (3, 5), (17, 9), (4, 6)])
def test_flag_off_assembler_is_fb1f4ef44(monkeypatch, bucketed, Rs):
    _flag(monkeypatch, False)

    def must_not_run(*args):
        raise AssertionError("the spec tail ran with VLLM_TPU_SPEC_TAIL_COMPILED unset")

    monkeypatch.setattr(tpu_runner, "_spec_tail_compiled", must_not_run)
    g = np.random.default_rng(sum(Rs) * 3 + bucketed)
    data = [_chunk(g, n, bucketed, md_none=(Rs == (4, 6) and i == 1)) for i, n in enumerate(Rs)]
    nts, ds, mds = zip(*data)
    drafts = list(ds) if bucketed else torch.cat(ds)
    new = _assemble(_runner(bucketed), drafts, nts, mds, Rs)
    old = _assemble(_runner(bucketed), drafts, nts, mds, Rs, old=True)
    _same(new[0], old[0], "bounded source")
    _same(new[2], old[2], "num_rejected")
    assert new[1] == old[1] and new[3] == old[3]


def test_compiled_tail_uses_the_owned_alias_strip():
    """B (dongruizi's 1e044e7d7 fix): the compiled tail is wrapped by owned_args.static_shape_args_owned, not by
    eagle3's `a[:]` view wrapper, and the wrapped callable is the compiled `_spec_tail`."""
    w = tpu_runner._spec_tail_compiled
    assert w.__code__.co_filename.endswith("owned_args.py"), w.__code__.co_filename
    inner = w.__wrapped__
    orig = getattr(inner, "_torchdynamo_orig_callable", None) or getattr(inner, "__wrapped__", None)
    assert orig is tpu_runner._spec_tail


def test_owned_alias_strip_passes_no_stamp_and_no_view():
    seen = []

    def probe(*args):
        seen.append([(owned_args.is_stamped(a), owned_args.is_owned(a)) for a in args if isinstance(a, torch.Tensor)])
        return tpu_runner._spec_tail(*args)

    g = np.random.default_rng(1)
    nt, d, md = _chunk(g, 5, False)
    big = torch.cat([nt, nt])
    view = big[: nt.shape[0]]
    torch._dynamo.mark_dynamic(view, 0)
    torch._dynamo.mark_dynamic(d, 0)
    out = owned_args.static_shape_args_owned(probe)(view, d, md.draft_lengths, 5, 5, BOUND)
    assert seen == [[(False, True)] * 3]
    ref_out = tpu_runner._spec_tail(nt, d, md.draft_lengths, 5, 5, BOUND)
    for a, b in zip(out, ref_out):
        _same(a, b, "tail through the strip")
