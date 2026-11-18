# SPDX-License-Identifier: Apache-2.0
"""c32 dispatch-order flags (default off). Each one only moves when work is
enqueued; the programs, their inputs and their outputs stay the same.

- VLLM_TPU_SEED_COPIES_LATE: the mamba seed copies, read-offset resets and the
  input-staging fence are issued after the substitution, the draft gather and
  the adapter's input program, right before the forward.
- VLLM_TPU_EXTRACT_AFTER_FORWARD: the verified-draft gather is enqueued after
  the forward launch.
- VLLM_TPU_TOKEN_COPY_EARLY_FLUSH: the sampled-token chunks are materialized
  (no host wait) before their non-blocking D2H copy is issued.
"""

import ast
import inspect
import random
import textwrap
import types

import pytest
import torch

import vllm_torchtpu.utils as tpu_utils
from vllm_torchtpu import envs
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner import tpu_runner_async_output as async_output
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncTPUCopyState,
    extract_draft_token_ids,
)

TPUModelRunner = tpu_runner.TPUModelRunner


# --------------------------------------------------------------------------
# VLLM_TPU_TOKEN_COPY_EARLY_FLUSH
# --------------------------------------------------------------------------


class _Log:
    def __init__(self):
        self.events = []


def _fake_tpu(monkeypatch, log):
    class _Event:
        def record(self):
            log.events.append(("record",))

        def synchronize(self):
            log.events.append(("event_sync",))

    monkeypatch.setattr(torch, "tpu", types.SimpleNamespace(Event=_Event), raising=False)

    def _sync(tensors=None, wait=True):
        log.events.append(("flush", tuple(id(t) for t in tensors), wait))

    monkeypatch.setattr(tpu_utils, "synchronize_tensors", _sync)


def _chunks(rng, n_chunks):
    chunks, lens = [], []
    for _ in range(n_chunks):
        rows = rng.choice([4, 8, 16, 32])
        t = torch.randint(0, 151_000, (rows, 4), dtype=torch.int32)
        t[t % 3 == 0] = INVALID_TOKEN_ID
        chunks.append(t)
        lens.append(rng.randrange(1, rows + 1))
    return chunks, lens


@pytest.mark.parametrize("on", [False, True])
def test_token_copy_early_flush_order_and_values(monkeypatch, on):
    rng = random.Random(1)
    monkeypatch.setattr(envs, "VLLM_TPU_TOKEN_COPY_EARLY_FLUSH", on, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_PINNED_TOKEN_COPY", False, raising=False)
    for _ in range(50):
        log = _Log()
        _fake_tpu(monkeypatch, log)
        chunks, lens = _chunks(rng, rng.randrange(1, 4))
        state = AsyncTPUCopyState.from_device_chunks(chunks, lens)
        flushes = [e for e in log.events if e[0] == "flush"]
        if on:
            # One flush of exactly the copied chunks, no host wait, before
            # the ready event snapshots the copies.
            assert flushes == [("flush", tuple(id(t) for t in chunks), False)]
            assert log.events.index(flushes[0]) < log.events.index(("record",))
        else:
            assert flushes == []
        state.wait()
        want = torch.cat([t[:n] for t, n in zip(chunks, lens)], dim=0)
        assert torch.equal(state.sampled_token_ids_cpu, want)


def test_token_copy_early_flush_fault_control(monkeypatch):
    """Positive control: a flush that waits (wait=True) is caught."""
    monkeypatch.setattr(envs, "VLLM_TPU_TOKEN_COPY_EARLY_FLUSH", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_PINNED_TOKEN_COPY", False, raising=False)
    log = _Log()
    _fake_tpu(monkeypatch, log)
    real = tpu_utils.synchronize_tensors
    monkeypatch.setattr(
        tpu_utils, "synchronize_tensors", lambda tensors=None, wait=True: real(tensors)
    )
    chunks, lens = _chunks(random.Random(2), 1)
    AsyncTPUCopyState.from_device_chunks(chunks, lens)
    assert [e for e in log.events if e[0] == "flush"][0][2] is True


# --------------------------------------------------------------------------
# VLLM_TPU_SEED_COPIES_LATE
# --------------------------------------------------------------------------


def _call_names(fn) -> list[tuple[int, str]]:
    """(line, dotted callee) of every call in `fn`'s source, in source order."""
    src = textwrap.dedent(inspect.getsource(fn))
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            out.append((node.lineno, ast.unparse(node.func)))
    return sorted(out)


def _first_line(calls, name):
    lines = [ln for ln, c in calls if c == name]
    assert lines, name
    return lines[0]


def test_stage_seed_copies_late_keeps_relative_order():
    log = []
    fake = types.SimpleNamespace(
        _collect_mamba_state_seed_copies=lambda so, s, n: log.append(("collect", so, s, n)),
        _flush_mamba_state_seed_copies=lambda: log.append(("flush",)),
        _reset_read_offsets_for_new_requests=lambda s, n: log.append(("reset", s, n)),
        _record_input_staging_fence=lambda: log.append(("fence",)),
    )
    so = object()
    TPUModelRunner._stage_seed_copies_late(fake, so, 3, 5)
    assert log == [("collect", so, 3, 5), ("flush",), ("reset", 3, 5), ("fence",)]


def test_seed_copies_late_execute_model_order():
    """Flag on: substitution, draft gather and adapter program come before the
    late seed block, which comes before the forward. Flag off: the old order
    (flush/reset right after _prepare_inputs, fence before the adapter)."""
    calls = _call_names(TPUModelRunner.execute_model)
    prep = _first_line(calls, "self._prepare_inputs")
    flush = _first_line(calls, "self._flush_mamba_state_seed_copies")
    reset = _first_line(calls, "self._reset_read_offsets_for_new_requests")
    subst = _first_line(calls, "self._apply_async_token_substitution")
    fence = _first_line(calls, "self._record_input_staging_fence")
    adapter = _first_line(calls, "self.model_adapter.prepare_inputs")
    late = _first_line(calls, "self._stage_seed_copies_late")
    fwd = _first_line(calls, "self.forward_model")
    assert prep < flush < reset < subst < fence < adapter < late < fwd
    src = inspect.getsource(TPUModelRunner.execute_model)
    assert src.count("if not seed_late:") == 2
    assert src.count("if seed_late:") == 1
    # The collector in _prepare_inputs is skipped under the flag.
    psrc = inspect.getsource(TPUModelRunner._prepare_inputs)
    i = psrc.index("if not envs.VLLM_TPU_SEED_COPIES_LATE:")
    assert psrc.index("self._collect_mamba_state_seed_copies(", i) - i < 120


def _moved_writes() -> set[str]:
    """`self.<attr>` assignment targets in the code that SEED_COPIES_LATE now
    runs before the collector."""
    out = set()
    for fn in (
        TPUModelRunner._apply_async_token_substitution,
        TPUModelRunner._get_model_inputs,
        TPUModelRunner._token_padding_update,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                while isinstance(t, (ast.Subscript, ast.Attribute)):
                    if (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                    ):
                        out.add(t.attr)
                        break
                    t = t.value
    return out


def _self_reads(fn) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "self"
    }


def test_moved_code_does_not_write_what_the_seed_block_reads():
    reads = (
        _self_reads(TPUModelRunner._collect_mamba_state_seed_copies)
        | _self_reads(TPUModelRunner._flush_mamba_state_seed_copies)
        | _self_reads(TPUModelRunner._reset_read_offsets_for_new_requests)
        | _self_reads(TPUModelRunner._spec_seed_copies)
    )
    writes = _moved_writes()
    assert not (reads & writes), sorted(reads & writes)
    # Positive control: the collector does read state the runner writes.
    assert "_mamba_state_pos" in reads and "_pre_async_results" in reads


# --------------------------------------------------------------------------
# VLLM_TPU_EXTRACT_AFTER_FORWARD
# --------------------------------------------------------------------------


def test_extract_after_forward_order():
    calls = _call_names(TPUModelRunner.execute_model)
    fwd = _first_line(calls, "self.forward_model")
    # Two call sites of the selected gather: before the forward (flag off)
    # and right after it (flag on); nothing between the forward and the
    # deferred gather.
    lines = [ln for ln, c in calls if c == "extract"]
    assert len(lines) == 2 and lines[0] < fwd < lines[1]
    between = [c for ln, c in calls if fwd < ln < lines[1]]
    assert set(between) <= {"host_phases.mark"}, between
    src = inspect.getsource(TPUModelRunner.execute_model)
    assert "if envs.VLLM_TPU_EXTRACT_AFTER_FORWARD:" in src


def test_extract_after_forward_same_drafts():
    """The gather reads only input_ids and the logits indices; a forward that
    leaves input_ids alone gives the same drafts before or after it."""
    rng = random.Random(3)
    for _ in range(200):
        n_req = rng.randrange(1, 33)
        k = 3
        num_tokens = n_req * (k + 1)
        padded = num_tokens + rng.randrange(0, 64)
        ids = torch.randint(0, 151_000, (padded,), dtype=torch.int32)
        final_logits = torch.arange(num_tokens, dtype=torch.int32)
        target = torch.tensor(
            [r * (k + 1) + j for r in range(n_req) for j in range(k)], dtype=torch.int32
        )
        before = extract_draft_token_ids(ids, final_logits, target)
        snapshot = ids.clone()
        _ = (ids.to(torch.float32) * 2).sum()  # a forward that reads ids
        after = extract_draft_token_ids(ids, final_logits, target)
        assert torch.equal(ids, snapshot)
        assert torch.equal(before, after)
    # Positive control: a forward that wrote input_ids would be caught.
    ids[target[0] + 1] += 1
    assert not torch.equal(extract_draft_token_ids(ids, final_logits, target), after)


def test_flags_registered_default_off(monkeypatch):
    for name in (
        "VLLM_TPU_SEED_COPIES_LATE",
        "VLLM_TPU_EXTRACT_AFTER_FORWARD",
        "VLLM_TPU_TOKEN_COPY_EARLY_FLUSH",
    ):
        monkeypatch.delenv(name, raising=False)
        assert name in envs.environment_variables
        assert envs.environment_variables[name]() is False
        monkeypatch.setenv(name, "1")
        assert envs.environment_variables[name]() is True
    assert async_output.envs is envs
