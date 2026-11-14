# SPDX-License-Identifier: Apache-2.0
"""Host-path flags for the per-decode-step host time (default off): each must
leave every output the same as the path it replaces.

- VLLM_TPU_HOST_TOKEN_LISTS: the previous step's sampled tokens become host
  lists in plain Python (``_modify_prev_results``, ``get_output``).
- VLLM_TPU_PREFILL_CKPT_DECODE_SKIP: no prefill checkpoint arithmetic when no
  request schedules more than one fold unit.
- VLLM_TPU_ENVS_FREEZE: env reads are kept as module attributes after warmup.
"""

import os
import random
import types

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.core import mamba_tail_fold
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncPreResults,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
    SpecLogprobsChunk,
    valid_token_lists,
)


def _torch_lists(t: torch.Tensor) -> list[list[int]]:
    """The torch path both call sites ran before the flag."""
    mask = t != INVALID_TOKEN_ID
    gen_lens = mask.sum(dim=1).tolist()
    return [seq.tolist() for seq in t[mask].split(gen_lens)]


def _random_tokens(rng: random.Random, rows: int, cols: int, dtype) -> torch.Tensor:
    t = torch.tensor(
        [[rng.randrange(0, 151_000) for _ in range(cols)] for _ in range(rows)],
        dtype=dtype,
    )
    for r in range(rows):
        # Usually an accepted prefix, sometimes holes anywhere.
        if rng.random() < 0.8:
            t[r, rng.randrange(1, cols + 1) :] = INVALID_TOKEN_ID
        else:
            for c in range(cols):
                if rng.random() < 0.4:
                    t[r, c] = INVALID_TOKEN_ID
    return t


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_valid_token_lists_matches_torch_path(dtype):
    rng = random.Random(0)
    for _ in range(500):
        t = _random_tokens(rng, rng.randrange(1, 33), rng.randrange(2, 6), dtype)
        got = valid_token_lists(t)
        assert got == _torch_lists(t)
        assert all(type(x) is int for row in got for x in row)


class _DoneEvent:
    def synchronize(self):
        pass


def _copy_state(tokens: torch.Tensor, real: int) -> AsyncTPUCopyState:
    return AsyncTPUCopyState([tokens], [real], _DoneEvent())


def _get_output(tokens, real, discard, monkeypatch, on, logprobs=False):
    monkeypatch.setattr(envs, "VLLM_TPU_HOST_TOKEN_LISTS", on, raising=False)
    from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

    state = _copy_state(tokens.clone(), real)
    if logprobs:
        n_rows = int(tokens.shape[0] * tokens.shape[1])
        g = torch.Generator().manual_seed(1)
        state.spec_logprobs = [
            SpecLogprobsChunk(
                LogprobsTensors(
                    torch.randint(0, 1000, (n_rows, 3), generator=g),
                    torch.randn((n_rows, 3), generator=g),
                    torch.randint(0, 50, (n_rows,), generator=g),
                ),
                np.arange(real, dtype=np.int64) * tokens.shape[1],
            )
        ]
    out = ModelRunnerOutput(
        req_ids=[f"r{i}" for i in range(real)],
        req_id_to_index={f"r{i}": i for i in range(real)},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    res = AsyncTPUModelRunnerOutput(out, state, list(discard)).get_output()
    lp = res.logprobs
    return res.sampled_token_ids, (
        None
        if lp is None
        else (
            lp.logprob_token_ids.tolist(),
            lp.logprobs.tolist(),
            lp.sampled_token_ranks.tolist(),
            list(lp.cu_num_generated_tokens),
        )
    )


@pytest.mark.parametrize("logprobs", [False, True])
def test_get_output_same_with_host_token_lists(monkeypatch, logprobs):
    rng = random.Random(1)
    for _ in range(100):
        real = rng.randrange(1, 17)
        padded = real + rng.randrange(0, 8)
        # Logprobs rows are only defined for an accepted prefix.
        t = _random_tokens(random.Random(rng.random()), padded, 4, torch.int32)
        if logprobs:
            for r in range(padded):
                row = t[r].tolist()
                n = next((i for i, x in enumerate(row) if x == INVALID_TOKEN_ID), 4)
                t[r, n:] = INVALID_TOKEN_ID
        discard = rng.sample(range(real), rng.randrange(0, min(3, real) + 1))
        off = _get_output(t, real, discard, monkeypatch, False, logprobs)
        on = _get_output(t, real, discard, monkeypatch, True, logprobs)
        assert on == off


def _prev_results_runner(rng: random.Random, tokens: torch.Tensor, real: int):
    """A minimal `self` for TPUModelRunner._modify_prev_results: requests
    whose last step appended 1 + num_draft optimistic placeholders."""
    max_len = 64
    num_batch = real + 2
    req_ids = [f"r{i}" for i in range(real)]
    token_ids_cpu = np.zeros((num_batch, max_len), dtype=np.int32)
    num_tokens_no_spec = np.zeros(num_batch, dtype=np.int32)
    requests, seq_lens, num_draft = {}, [], {}
    batch_rows = rng.sample(range(num_batch), real)
    for i, rid in enumerate(req_ids):
        drafts = int(tokens.shape[1]) - 1
        num_draft[i] = drafts
        out = [rng.randrange(1000) for _ in range(rng.randrange(0, 5))]
        out += [-7] * (1 + drafts)  # placeholders
        state = types.SimpleNamespace(output_token_ids=out)
        requests[rid] = state
        seq_lens.append((i, state, 0, rid))
        row = batch_rows[i]
        num_tokens_no_spec[row] = rng.randrange(1 + drafts, max_len - 8)
        token_ids_cpu[row, : num_tokens_no_spec[row]] = rng.randrange(1000)
    discard = rng.sample(range(real), rng.randrange(0, min(2, real) + 1))
    batch = types.SimpleNamespace(
        req_id_to_index={rid: batch_rows[i] for i, rid in enumerate(req_ids)},
        num_tokens_no_spec=num_tokens_no_spec,
        token_ids_cpu=token_ids_cpu,
    )
    # One request finished since (no longer in the persistent batch).
    if real > 1 and rng.random() < 0.3:
        batch.req_id_to_index.pop(req_ids[-1])
    pre = AsyncPreResults(
        req_ids=req_ids,
        next_tokens_tpu=torch.zeros(1),
        request_seq_lens=seq_lens,
        discard_sampled_tokens_req_indices=discard,
        req_id_to_index_copy={},
        copy_state=_copy_state(tokens.clone(), real),
        num_draft_per_req=num_draft,
    )
    return types.SimpleNamespace(_pre_async_results=pre, input_batch=batch)


def _runner_state(runner):
    pre = runner._pre_async_results
    return (
        [list(rec[1].output_token_ids) for rec in pre.request_seq_lens],
        runner.input_batch.num_tokens_no_spec.tolist(),
        runner.input_batch.token_ids_cpu.tolist(),
    )


def test_modify_prev_results_same_with_host_token_lists(monkeypatch):
    modify = tpu_runner.TPUModelRunner._modify_prev_results
    rng = random.Random(2)
    for case in range(200):
        real = rng.randrange(1, 17)
        t = _random_tokens(rng, real + rng.randrange(0, 8), rng.randrange(2, 5), torch.int32)
        # An accepted prefix per row (what the rejection sampler emits).
        for r in range(t.shape[0]):
            row = t[r].tolist()
            n = next((i for i, x in enumerate(row) if x == INVALID_TOKEN_ID), len(row))
            t[r, n:] = INVALID_TOKEN_ID
        results = []
        for on in (False, True):
            monkeypatch.setattr(envs, "VLLM_TPU_HOST_TOKEN_LISTS", on, raising=False)
            # Same seed: the same requests, rows and placeholders in both arms.
            runner = _prev_results_runner(random.Random(case), t, real)
            modify(runner)
            results.append(_runner_state(runner))
        assert results[0] == results[1]


@pytest.mark.parametrize("nobound", [False, True])
@pytest.mark.parametrize("unit,block", [(128, 128), (256, 256), (256, 1024)])
def test_no_checkpoint_at_or_below_one_unit(monkeypatch, nobound, unit, block):
    monkeypatch.setattr(
        envs, "VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY", nobound, raising=False
    )
    rng = np.random.default_rng(3)
    for _ in range(2000):
        n = int(rng.integers(1, 33))
        computed = rng.integers(0, 200_000, n)
        # Aligned starts too, the only ones that can checkpoint.
        computed[rng.random(n) < 0.5] //= unit
        computed[rng.random(n) < 0.5] *= unit
        scheduled = rng.integers(0, unit + 1, n)
        offsets, columns = mamba_tail_fold.prefill_checkpoints(
            computed, scheduled, block, unit, num_speculative_blocks=2
        )
        assert not offsets.any() and not columns.any()
    # Positive control: one unit + 1 at an aligned start does checkpoint.
    offsets, _ = mamba_tail_fold.prefill_checkpoints(
        np.array([4 * block]), np.array([2 * block + 1]), block, unit
    )
    assert offsets.any()


def _ckpt_runner(num_computed: np.ndarray):
    return types.SimpleNamespace(
        _mamba_tail_fold=True,
        _mamba_state_block_size=128,
        _mamba_spec_blocks=0,
        _mamba_prefill_ckpt_zeros={},
        device=torch.device("cpu"),
        input_batch=types.SimpleNamespace(num_computed_tokens_cpu=num_computed),
        _mamba_prefill_ckpt_zero=lambda n, z={}: z.setdefault(
            n, torch.zeros((2, n), dtype=torch.int32)
        ),
    )


def test_prefill_ckpt_same_with_decode_skip(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", 0, raising=False)
    monkeypatch.setattr(tpu_runner, "stage_h2d_upload", lambda runner, host: host)
    counter = mamba_tail_fold._Counter("exported", ("checkpoints", "prefill rows"))
    monkeypatch.setattr(tpu_runner, "WORKER_COUNTER", counter)
    fn = tpu_runner.TPUModelRunner._mamba_prefill_ckpt
    rng = np.random.default_rng(4)
    for case in range(400):
        n = int(rng.integers(1, 17))
        computed = rng.integers(0, 50_000, n + 3)
        computed[::2] = computed[::2] // 128 * 128
        # Mostly decode steps (4 rows per request), sometimes a prefill chunk.
        scheduled = np.full(n, 4, dtype=np.int32)
        if case % 3 == 0:
            scheduled[int(rng.integers(0, n))] = int(rng.integers(5, 2000))
        start = int(rng.integers(0, 3))
        target = n + int(rng.integers(0, 8))
        outs = []
        for on in (False, True):
            monkeypatch.setattr(
                envs, "VLLM_TPU_PREFILL_CKPT_DECODE_SKIP", on, raising=False
            )
            before = (counter.hits, counter.total)
            out = fn(_ckpt_runner(computed), start, n, scheduled, target)
            outs.append((out.tolist(), counter.hits - before[0], counter.total - before[1]))
        assert outs[0] == outs[1], case


def test_freeze_reads_keeps_values_and_errors(monkeypatch):
    names = [
        "VLLM_TPU_VERIFY_FUSED",
        "VLLM_TPU_CPU_PIN",
        "VLLM_TPU_HOST_PHASES",
        "VLLM_TPU_MK_TRYFWD_LEAN",
    ]
    monkeypatch.setenv("VLLM_TPU_VERIFY_FUSED", "1")
    monkeypatch.setenv("VLLM_TPU_CPU_PIN", "numa")
    monkeypatch.setenv("VLLM_TPU_HOST_PHASES", "500")
    monkeypatch.delenv("VLLM_TPU_MK_TRYFWD_LEAN", raising=False)
    monkeypatch.setenv("VLLM_TPU_HOST_TOKEN_LISTS", "notabool")
    original = envs.__dict__["__getattr__"]
    stale = {n: envs.__dict__.pop(n) for n in names + ["VLLM_TPU_HOST_TOKEN_LISTS"] if n in envs.__dict__}
    try:
        before = {n: getattr(envs, n) for n in names}
        envs.freeze_reads()
        envs.freeze_reads()  # idempotent
        after = {n: getattr(envs, n) for n in names}
        assert after == before
        assert all(n in envs.__dict__ for n in names)
        # Kept: a later env change is not seen (nothing changes it after warmup).
        os.environ["VLLM_TPU_VERIFY_FUSED"] = "0"
        assert envs.VLLM_TPU_VERIFY_FUSED is True
        # A malformed value still raises on its first read, not at freeze.
        with pytest.raises(Exception):
            _ = envs.VLLM_TPU_HOST_TOKEN_LISTS
        assert "VLLM_TPU_HOST_TOKEN_LISTS" not in envs.__dict__
        with pytest.raises(AttributeError):
            _ = envs.NOT_AN_ENV_VARIABLE
    finally:
        envs.__dict__["__getattr__"] = original
        for n in names + ["VLLM_TPU_HOST_TOKEN_LISTS"]:
            envs.__dict__.pop(n, None)
        envs.__dict__.update(stale)
