# SPDX-License-Identifier: Apache-2.0
"""q38 L6-H2 lever toklists: TOKEN-LISTS-ONCE, the async-output part of IX a46d91d20
(Jordan Dotzel), ported onto q38-l6-v4-eos. Host-only; runs in a CDK CPU pod.

The sampled-token copy converts its unpadded rows to Python lists once
(AsyncTPUCopyState.sampled_token_ids_list); get_output on the output thread and
_modify_prev_results on the main thread both filter those rows instead of masking,
summing, boolean-indexing and splitting the tensor. Every test compares the lever-on
result with the stock tensor path from the same inputs, list for list.
"""

from __future__ import annotations

import os
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import l6_carry_harness as H
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import tpu_runner_async_output as tao
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)

ENV = "VLLM_TPU_L6_TOKEN_LISTS"
CHECK = "VLLM_TPU_L6_TOKEN_LISTS_CHECK"


class _Event:
    def __init__(self):
        self.synced = 0

    def record(self):
        pass

    def synchronize(self):
        self.synced += 1


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(torch, "tpu", NS(Event=_Event), raising=False)
    tao._L6_TL.update(check=None, n=0, out_seen=0)
    tao._L6_OUT_LISTS[0] = 0
    l6_ab._resolve()
    yield monkeypatch
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    tao._L6_TL.update(check=None, n=0, out_seen=0)
    l6_ab._resolve()


def _lever(mp, value="1", **extra):
    mp.setenv(ENV, value)
    for k, v in extra.items():
        mp.setenv(k, v)
    tao._L6_TL.update(check=None, n=0)
    l6_ab._resolve()
    l6_ab.begin_step()


# ------------------------------------------------------------------ random served-shaped streams

# 25 streams: (seed, number of running requests, columns = 1 + K, chunks).
STREAMS = [
    (s, n, cols, chunks)
    for s, (n, cols, chunks) in enumerate(
        [(n, c, ch) for n in (1, 4, 9, 11, 25) for c, ch in ((4, 1), (4, 2), (1, 1), (2, 1), (4, 3))]
    )
]
assert len(STREAMS) == 25


def _rows(rng: random.Random, n: int, cols: int) -> torch.Tensor:
    """[n, cols] sampled rows shaped like the rejection sampler's output: an accepted prefix plus
    the bonus, then INVALID_TOKEN_ID padding; occasionally an INVALID inside a row (the filter
    must drop every INVALID entry, wherever it is)."""
    t = torch.full((n, cols), INVALID_TOKEN_ID, dtype=torch.int32)
    for i in range(n):
        k = rng.randint(1, cols)
        t[i, :k] = torch.tensor([rng.randint(0, 151000) for _ in range(k)], dtype=torch.int32)
        if cols > 2 and rng.random() < 0.05:
            t[i, rng.randrange(cols)] = INVALID_TOKEN_ID
    return t


def _state(rows: torch.Tensor, chunks: int, rng: random.Random, lists: bool) -> AsyncTPUCopyState:
    """A copy state holding `rows` split into `chunks` bucket-padded chunks (as from_device_chunks
    leaves them after the copy)."""
    n = rows.shape[0]
    cuts = sorted(rng.sample(range(1, n), min(chunks - 1, n - 1))) if n > 1 else []
    parts, start = [], 0
    for c in cuts + [n]:
        part = rows[start:c]
        pad = torch.full((rng.randint(0, 5), rows.shape[1]), 7, dtype=rows.dtype)  # garbage pad rows
        parts.append((torch.cat([part, pad]), c - start))
        start = c
    return AsyncTPUCopyState(
        chunks_cpu=[p for p, _ in parts],
        chunk_real_lens=[k for _, k in parts],
        copy_ready_event=_Event(),
        l6_lists=lists,
    )


def _get_output(state: AsyncTPUCopyState, discard: list[int]):
    mro = NS(sampled_token_ids=None, logprobs=None)
    return AsyncTPUModelRunnerOutput(mro, state, list(discard)).get_output().sampled_token_ids


@pytest.mark.parametrize("stream", STREAMS, ids=[f"s{s}-n{n}-c{c}-k{k}" for s, n, c, k in STREAMS])
def test_get_output_lists_equal_tensor_path(stream):
    seed, n, cols, chunks = stream
    rng = random.Random(seed)
    steps = 40
    for _ in range(steps):
        rows = _rows(rng, n, cols)
        discard = sorted(rng.sample(range(n), rng.randint(0, min(2, n))))
        r2 = random.Random(rng.random())
        stock = _get_output(_state(rows, chunks, random.Random(r2.random()), False), discard)
        st = _state(rows, chunks, random.Random(r2.random()), True)
        got = _get_output(st, discard)
        assert got == stock, (got, stock)
        # The cached rows are never edited by a consumer (discards clear copies).
        assert st._sampled_token_ids_list == st.sampled_token_ids_cpu.reshape(n, -1).tolist()
    assert tao._L6_OUT_LISTS[0] == steps


def test_spec_logprobs_gen_lens_equal(clean):
    """With spec logprobs the lists path hands spec_logprobs_lists the same per-row counts."""
    seen = []
    clean.setattr(tao, "spec_logprobs_lists", lambda chunks, gen_lens: seen.append(list(gen_lens)) or "LP")
    rng = random.Random(3)
    for _ in range(30):
        rows = _rows(rng, 9, 4)
        for lists in (False, True):
            st = _state(rows, 1, random.Random(5), lists)
            st.spec_logprobs = ["x"]
            mro = NS(sampled_token_ids=None, logprobs=None)
            AsyncTPUModelRunnerOutput(mro, st, [2]).get_output()
            assert mro.logprobs == "LP"
        assert seen[-1] == seen[-2], seen[-2:]


def test_copy_state_takes_the_issuing_steps_lever_state(clean):
    """from_device_chunks fixes l6_lists from the lever on the main thread at copy time."""
    chunk = torch.zeros((4, 4), dtype=torch.int32)
    assert AsyncTPUCopyState.from_device_chunks([chunk], [4]).l6_lists is False
    _lever(clean, "1")
    assert AsyncTPUCopyState.from_device_chunks([chunk], [4]).l6_lists is True
    clean.setenv(ENV, "0")
    l6_ab._resolve()
    l6_ab.begin_step()
    assert AsyncTPUCopyState.from_device_chunks([chunk], [4]).l6_lists is False


def test_lever_off_never_builds_lists(clean):
    calls = []
    clean.setattr(tao, "l6_valid_rows", lambda *a: calls.append(a) or [])
    rng = random.Random(1)
    rows = _rows(rng, 9, 4)
    st = _state(rows, 1, rng, False)
    _get_output(st, [])
    assert calls == [] and st._sampled_token_ids_list is None


# ------------------------------------------------------------------ the main thread (_modify_prev_results)


def _swap_copy_states(lists: bool):
    """on_step hook: replace the harness's stand-in copy state of the parked results with a real
    AsyncTPUCopyState holding the same rows (l6_lists = `lists`), 1-3 bucket-padded chunks."""

    def hook(sim):
        pre = sim.r._pre_async_results
        if pre is None or pre.copy_state is None:
            return
        rows = pre.copy_state.sampled_token_ids_cpu
        rng = random.Random(sim.n * 7919 + 13)
        st = _state(rows, rng.randint(1, 3), rng, lists)
        hook.states.append(st)
        pre.copy_state = st

    hook.states = []
    return hook


SHAPES = {
    "b1_mamba_comb": dict(nreq=1, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b3_padded": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b8_full_nocomb": dict(nreq=8, max_num_reqs=8, paddings=(32, 64), mamba=True, comb=False),
}
EVENTS = [{}, {6: ("join",), 9: ("discard",), 12: ("leave",), 15: ("swap",), 18: ("remap",)}]


def _final(sim):
    r = sim.r
    out = {rid: list(req.output_token_ids) for rid, req in r.requests.items()}
    n = r.input_batch.num_reqs
    return out, r.input_batch.token_ids_cpu[:n].copy(), r.input_batch.num_tokens_no_spec[:n].copy()


@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("ev", range(len(EVENTS)))
def test_modify_prev_results_lists_equal_tensor_path(clean, name, ev):
    """The real _modify_prev_results over the carry harness's async-spec streams: the rows it
    commits (output_token_ids, token_ids_cpu, num_tokens_no_spec) and every prepared input are
    equal with the copy states on the lists path and on the tensor path."""
    clean.setenv("VLLM_TPU_POSTFWD_FUSED", "1")
    undo = H.install_eager_programs()
    try:
        shape = SHAPES[name]
        # a full batch (nreq == max_num_reqs) has no free row to join
        events = {s: e for s, e in EVENTS[ev].items()
                  if not (e == ("join",) and shape["nreq"] >= shape["max_num_reqs"])}
        hook0 = _swap_copy_states(False)
        base, bsim = H.run(shape, 24, seed=31 + ev, events=events, on_step=hook0)
        _lever(clean, "1")
        main0 = l6_ab._state["counts"].get("toklists.main", 0)
        hook1 = _swap_copy_states(True)
        got, gsim = H.run(shape, 24, seed=31 + ev, events=events, on_step=hook1)
    finally:
        undo()
    assert H.first_difference(base, got) is None
    fb, fg = _final(bsim), _final(gsim)
    assert fb[0] == fg[0]
    assert np.array_equal(fb[1], fg[1]) and np.array_equal(fb[2], fg[2])
    # Every lists-path copy state was converted once and consumed by the main thread.
    assert hook1.states and all(s._sampled_token_ids_list is not None for s in hook1.states)
    assert all(s._sampled_token_ids_list is None for s in hook0.states)
    assert l6_ab._state["counts"].get("toklists.main", 0) - main0 == len(hook1.states)


def test_one_conversion_shared_by_both_threads(clean):
    """get_output (output thread) and the main thread's rows read the same cached list; the
    output thread's discard clears its own copy, not the main thread's rows."""
    _lever(clean, "1")
    rng = random.Random(4)
    rows = _rows(rng, 9, 4)
    st = _state(rows, 2, rng, True)
    calls = {"n": 0}
    real = AsyncTPUCopyState.sampled_token_ids_cpu.fget

    def counted(self):
        calls["n"] += 1
        return real(self)

    clean.setattr(AsyncTPUCopyState, "sampled_token_ids_cpu", property(counted))
    out = _get_output(st, [0, 3])
    cached = st._sampled_token_ids_list
    main = tao.l6_main_rows(st, 9, 4, rows)
    assert st._sampled_token_ids_list is cached
    assert out[0] == [] and out[3] == [] and main[0] != [] and main[3] != []
    stock, _ = tao.tensor_valid_rows(rows)
    assert main == stock
    assert calls["n"] == 2  # get_output reads the tensor once for its shape, the list builds once


def test_engagement_counts(clean):
    _lever(clean, "1")
    rng = random.Random(2)
    rows = _rows(rng, 4, 4)
    for _ in range(3):
        _get_output(_state(rows, 1, rng, True), [])
    tao.l6_main_rows(_state(rows, 1, rng, True), 4, 4, rows)
    c = l6_ab._state["counts"]
    assert c.get("toklists.main") == 1 and c.get("toklists.out") == 3, c


def test_check_mode_compares_and_catches_a_wrong_filter(clean):
    _lever(clean, "1", **{CHECK: "2"})
    rng = random.Random(9)
    rows = _rows(rng, 9, 4)
    for _ in range(4):
        assert tao.l6_main_rows(_state(rows, 1, rng, True), 9, 4, rows) == tao.tensor_valid_rows(rows)[0]
    c = l6_ab._state["counts"]
    assert c.get("toklists.checked") == 2 and c.get("toklists.mismatch", 0) == 0, c
    # A filter that keeps the INVALID entries: the CHECK turn reports it and returns the tensor path.
    clean.setattr(tao, "l6_valid_rows", lambda raw, g: [r[:] for r in raw])
    got = [tao.l6_main_rows(_state(rows, 1, rng, True), 9, 4, rows) for _ in range(2)]
    assert got[1] == tao.tensor_valid_rows(rows)[0]
    assert l6_ab._state["counts"].get("toklists.mismatch") == 1


def test_check_steps_bound_the_check(clean):
    _lever(clean, "1", **{CHECK: "1", "VLLM_TPU_L6_CHECK_STEPS": "2"})
    rng = random.Random(9)
    rows = _rows(rng, 4, 4)
    for _ in range(4):
        tao.l6_main_rows(_state(rows, 1, rng, True), 4, 4, rows)
        l6_ab.begin_step()
    assert l6_ab._state["counts"].get("toklists.checked", 0) == 2, l6_ab._state["counts"]


# ------------------------------------------------------------------ registration and import origin


def test_lever_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert l6_ab.LEVERS["toklists"] == ENV
    ignored = tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert ENV in ignored and CHECK in ignored
    assert hasattr(envs, ENV) and hasattr(envs, CHECK)


def test_runner_call_site_is_guarded():
    """_modify_prev_results takes the lists path only for a copy state that carries the lever."""
    import inspect

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    src = inspect.getsource(TPUModelRunner._modify_prev_results)
    assert 'pre_copy_state = getattr(self._pre_async_results, "copy_state", None)' in src
    assert "isinstance(pre_copy_state, AsyncTPUCopyState)" in src and "pre_copy_state.l6_lists" in src
    assert "pre_next_tokens_cpu.dim() == 2" in src and "l6_main_rows(" in src


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
