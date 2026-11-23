# SPDX-License-Identifier: Apache-2.0
"""Exact host-side cuts in `_prepare_inputs`' steady MK decode path.

Each cut is a default-off flag; with it on, every device input the step
produces (the folded prep program's operands and outputs, input ids,
positions, seq_lens, query_start_loc, both request distributions and the
whole attention metadata) must be bitwise the flag-off step's, across steps
with churn (add / finish / reorder), prefill pieces, block growth, folded and
unfolded steps. Each cut also has an invalidation test and a negative
control: a deliberately broken cut that the same comparison catches.

The runner is `steady_decode_fakes`' CPU shell, driven through the real
`_prepare_inputs` up to the attention-metadata build, with
VLLM_TPU_PREFWD_PREP_FOLD on and the folded program replaced by a recorder
(its compiled form needs a TPU; tests/runner/test_prefwd_prep_fold.py covers
the program itself).
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch

from tests.runner.steady_decode_fakes import Workload, _Stop, env, make_runner
from tests.runner.test_steady_decode_metadata import _assert_metadata_equal
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.runner import tpu_runner

K = 3
FOLD_ENV = {
    "VLLM_TPU_PREFWD_PREP_FOLD": "1",
    "VLLM_TPU_PREFWD_PREP_FOLD_CHECK": "",
    "VLLM_TPU_SUBTRACT_REJECTED_PAD_ROWS": "0",
    "VLLM_TPU_SUBTRACT_REJECTED_COMPILED": "0",
}
_WARM = ("warm",)


class _FoldAdapter:
    """The model-adapter surface `_prep_fold_precheck` / `_try_prep_fold`
    read; its fold operands are the step's query_start_loc."""

    def prep_fold_supported(self):
        return True

    def prep_fold_args(self, start_index, num_reqs, query_start_loc, msi):
        return (query_start_loc,), {"window": False}


def _fake_fold(*args, num_tokens, ids_dtype, **static):
    """Stands in for `_prep_fold_compiled`: records its operands and returns
    outputs that depend on every one of them."""
    seq_lens, positions, rejected, seq_idx, pos_idx, src, final, target = args[:8]
    rest = args[8:]
    if len(rest) == 4:  # (qsl, input_ids, cur, pre): the scatter
        qsl, ids_in, cur, pre = rest
        ids = ids_in.clone()
        ids[cur.long()] = src[pre.long()].to(ids_dtype)
    else:
        (qsl,) = rest
        ids = src[:num_tokens].to(ids_dtype)
    _fake_fold.calls.append(
        (tuple(a.clone() for a in args), dict(num_tokens=num_tokens, **static))
    )
    return (
        seq_lens - 1,
        positions + 1,
        ids,
        ids[target.long()],
        qsl * 2,
    )


_fake_fold.calls = []


def make_fold_runner(max_model_len=8192):
    r = make_runner(max_model_len=max_model_len)
    r._fast_token_substitution = True
    r._pp_wave = None
    r.supports_mm_inputs = False
    r.model_adapter = _FoldAdapter()
    r._prep_fold_keys = {_WARM}
    return r


def apply(wl: Workload, runner, step: int) -> None:
    wl.apply(runner)
    pre = runner._pre_async_results
    if step % 5 == 2:
        # The requests sat in other rows of the previous step's results (same
        # batch, so a steady hit): the scatter substitution.
        rows = pre.req_id_to_index_copy
        n = len(wl.reqs)
        pre.req_id_to_index_copy = {r: (j + 1) % n for r, j in rows.items()}
    width = K + 1
    runner._pre_async_results.next_tokens_tpu = (
        torch.arange(runner.max_num_reqs * width, dtype=torch.int64) * 13 + step
    ) % 50000


_TO = torch.Tensor.to


def _h2d(self, *args, **kwargs):
    """`.to(device, non_blocking=True)` as an H2D: a new tensor holding the
    source's values at the time of the call (on CPU `.to` returns `self`,
    which would let later host writes show through "device" tensors)."""
    out = _TO(self, *args, **kwargs)
    if kwargs.get("non_blocking") and out is self:
        out = self.clone()
    return out


def run_step(runner, scheduler_output, num_decode, num_windowed, warm=True):
    """`_prepare_inputs` up to the metadata build; returns every device input
    it produced."""
    with ExitStack() as stack:
        stack.callback(setattr, torch.Tensor, "to", _TO)
        torch.Tensor.to = _h2d
        return _run_step(runner, scheduler_output, num_decode, num_windowed, warm)


def _run_step(runner, scheduler_output, num_decode, num_windowed, warm):
    captured = {}
    build = runner._build_attention_metadata

    def capture_build(*args, **kwargs):
        captured["metadata"] = build(*args, **kwargs)[0]
        captured["ctx"] = runner._attn_metadata_builder_ctx
        raise _Stop

    runner._build_attention_metadata = capture_build
    runner._prep_fold_keys = {_WARM} if warm else set()
    runner.__dict__.pop("_prep_fold_memo", None)
    _fake_fold.calls = []
    try:
        runner._prepare_inputs(scheduler_output, 0, num_decode, num_windowed)
    except _Stop:
        pass
    finally:
        del runner._build_attention_metadata
    folded = runner.__dict__.get("_prep_fold_out")
    captured["folded"] = folded
    captured["fold_calls"] = list(_fake_fold.calls)
    # execute_model reads runner.input_ids only on an unfolded step.
    captured["input_ids"] = runner.input_ids if folded is None else None
    captured["position_ids"] = runner.position_ids
    captured["qsl_np"] = runner.query_start_loc_np.copy()
    captured["seq_lens_np"] = runner.seq_lens_np.copy()
    captured["comb_cpu"] = runner._combined_request_distribution_cpu.clone()
    return captured


def _same(a, b, what):
    if a is None or b is None:
        assert a is None and b is None, what
        return
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, what
        assert torch.equal(a, b), what
        return
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b), what
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{what}[{i}]")
        return
    if isinstance(a, dict):
        assert a.keys() == b.keys(), what
        for key in a:
            _same(a[key], b[key], f"{what}.{key}")
        return
    assert a == b, what


def assert_step_equal(off, on, step):
    _assert_metadata_equal(off, on, step)
    for name in ("folded", "fold_calls", "comb_cpu"):
        _same(off[name], on[name], f"step {step}: {name}")
    for name in ("mamba_request_distribution", "mamba_state_indices"):
        _same(
            getattr(off["ctx"], name),
            getattr(on["ctx"], name),
            f"step {step}: ctx.{name}",
        )


def run_pair(
    monkeypatch,
    flags: dict,
    seed: int,
    steps: int = 60,
    stats=None,
    cold_every: int = 7,
):
    """Flag off vs `flags` on over one random workload. Returns step stats."""
    monkeypatch.setattr(tpu_runner, "_prep_fold_compiled", _fake_fold)
    monkeypatch.setattr(tpu_runner, "_prep_fold_key", lambda args, static: _WARM)
    wl = Workload(seed, 4, k=K, max_model_len=8192, churn=0.08, prefill=0.05)
    off = make_fold_runner()
    on = make_fold_runner()
    stats = {"folded": 0, "one_to_one": 0, "scatter": 0, "unfolded": 0}
    for step in range(steps):
        if step:
            wl.step()
        num_decode, num_windowed = wl.segments()
        apply(wl, off, step)
        apply(wl, on, step)
        so = wl.scheduler_output()
        # Some steps' fold keys are "not warmed": the unfolded fallback.
        warm = bool(cold_every) and step % cold_every != cold_every - 1
        with env(steady=True, extra=FOLD_ENV):
            out_off = run_step(off, so, num_decode, num_windowed, warm)
        with env(steady=True, extra={**FOLD_ENV, **flags}):
            out_on = run_step(on, so, num_decode, num_windowed, warm)
        try:
            assert_step_equal(out_off, out_on, step)
        except AssertionError as e:
            raise AssertionError(f"MISMATCH at step {step}: {e}") from e
        if out_off["folded"] is None:
            stats["unfolded"] += 1
        else:
            stats["folded"] += 1
            calls = out_off["fold_calls"]
            stats["one_to_one" if len(calls[0][0]) == 9 else "scatter"] += 1
    return stats


def _check_coverage(stats):
    # The workload reaches every kind of step the cuts distinguish.
    assert stats["one_to_one"] > 5, stats
    assert stats["scatter"] > 2, stats
    assert stats["unfolded"] > 5, stats


def test_registered_and_out_of_compile_key(monkeypatch):
    for name in (
        "VLLM_TPU_MAMBA_RD_REUSE",
        "VLLM_TPU_KV_CONNECTOR_SKIP_NOOP",
        "VLLM_TPU_PREP_FOLD_PRECHECK_MEMO",
        "VLLM_TPU_PREP_LAZY_INPUT_IDS",
    ):
        assert name in envs.environment_variables
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
        monkeypatch.delenv(name, raising=False)
        assert envs.environment_variables[name]() is False
        monkeypatch.setenv(name, "1")
        assert envs.environment_variables[name]() is True


def test_harness_baseline_is_deterministic(monkeypatch):
    """Off vs off: the comparison is exact by itself (no flag)."""
    stats = run_pair(monkeypatch, {}, seed=3)
    _check_coverage(stats)
    print("step mix", stats)


# --------------------------------------------------------------------------
# D. VLLM_TPU_MAMBA_RD_REUSE
# --------------------------------------------------------------------------

D_FLAG = {"VLLM_TPU_MAMBA_RD_REUSE": "1"}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mamba_rd_reuse_bitwise(monkeypatch, seed):
    _check_coverage(run_pair(monkeypatch, D_FLAG, seed))


def test_mamba_rd_reuse_hits_and_invalidates(monkeypatch):
    """Hits reuse the very same view objects; a new key (other num_reqs or a
    prefill piece) re-slices a new upload; the views follow the cache."""
    monkeypatch.setattr(tpu_runner, "_prep_fold_compiled", _fake_fold)
    monkeypatch.setattr(tpu_runner, "_prep_fold_key", lambda args, static: _WARM)
    r = make_fold_runner()
    wl = Workload(5, 4, k=K, max_model_len=8192, churn=0.0, prefill=0.0)
    seen = []
    for step in range(6):
        if step:
            wl.step()
        if step == 3:
            wl._add()  # a new request: num_reqs 4 -> 5
        num_decode, num_windowed = wl.segments()
        apply(wl, r, step)
        with env(steady=True, extra={**FOLD_ENV, **D_FLAG}):
            out = run_step(r, wl.scheduler_output(), num_decode, num_windowed)
        seen.append(
            (
                out["ctx"].request_distribution,
                out["ctx"].mamba_request_distribution,
                r._cached_comb_dev,
            )
        )
    for step in (1, 2):
        assert seen[step][0] is seen[0][0] and seen[step][1] is seen[0][1]
    assert seen[3][2] is not seen[0][2]  # re-keyed: a new upload
    assert seen[3][0] is not seen[0][0]
    assert seen[3][0]._base is seen[3][2] or seen[3][0].data_ptr() == (
        seen[3][2].data_ptr()
    )
    assert seen[4][0] is seen[3][0] and seen[5][0] is seen[3][0]
    assert seen[5][0].tolist() == [5, 5, 5] and seen[5][1].tolist() == [5, 5, 5]
    # A replaced device copy (any other writer of _cached_comb_dev) drops them.
    r._cached_comb_dev = r._cached_comb_dev.clone()
    assert r._reused_comb_views(r._cached_comb_key) is None


def test_mamba_rd_reuse_negative_control(monkeypatch):
    """A broken reuse that ignores the key is caught by the comparison."""

    def broken(self, key):
        views = self.__dict__.get("_cached_comb_views")
        return None if views is None else (views[1], views[2])

    monkeypatch.setattr(tpu_runner.TPUModelRunner, "_reused_comb_views", broken)
    with pytest.raises(AssertionError, match="MISMATCH"):
        run_pair(monkeypatch, D_FLAG, seed=0)


# --------------------------------------------------------------------------
# E. VLLM_TPU_KV_CONNECTOR_SKIP_NOOP
# --------------------------------------------------------------------------


def _kv_runner():
    r = object.__new__(tpu_runner.TPUModelRunner)
    r.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1, use_sequence_parallel_moe=False, is_moe_model=True
        ),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False, static_forward_context={}
        ),
    )
    return r


def _kv_step(monkeypatch, runner, flag: bool, connector):
    """One execute_model KV setup; returns what it observably did."""
    from vllm import forward_context as fc

    monkeypatch.setenv("VLLM_TPU_KV_CONNECTOR_SKIP_NOOP", "1" if flag else "0")
    monkeypatch.setattr(
        tpu_runner, "has_kv_transfer_group", lambda: connector is not None
    )
    monkeypatch.setattr(tpu_runner, "get_kv_transfer_group", lambda: connector)
    seen = []
    if connector is not None:
        connector.bind_connector_metadata.side_effect = lambda md: seen.append(
            ("bind", md, fc._forward_context)
        )
    before = (fc._forward_context, dict(fc.batchsize_forward_time))
    so = SimpleNamespace(kv_connector_metadata=object())
    runner._setup_kv_connector_for_step(so)
    after = (fc._forward_context, dict(fc.batchsize_forward_time))
    return before, after, seen, so


@pytest.mark.parametrize("flag", [False, True])
def test_kv_connector_skip_no_group_is_noop(monkeypatch, flag):
    """No KV transfer group: flag on skips the block; off runs it. Either way
    the forward context and batch-size stats are as before and nothing else
    is touched (connector getters would raise)."""
    from unittest.mock import Mock

    r = _kv_runner()

    def boom():
        raise AssertionError("connector looked up without a group")

    monkeypatch.setattr(tpu_runner, "get_kv_transfer_group", boom)
    entered = Mock(wraps=tpu_runner.set_forward_context)
    monkeypatch.setattr(tpu_runner, "set_forward_context", entered)
    monkeypatch.setenv("VLLM_TPU_KV_CONNECTOR_SKIP_NOOP", "1" if flag else "0")
    monkeypatch.setattr(tpu_runner, "has_kv_transfer_group", lambda: False)
    from vllm import forward_context as fc

    before = (fc._forward_context, dict(fc.batchsize_forward_time))
    r._setup_kv_connector_for_step(SimpleNamespace(kv_connector_metadata=None))
    assert (fc._forward_context, dict(fc.batchsize_forward_time)) == before
    # Patched set_forward_context (as tests do) is not the pinned one: the
    # skip only applies to vLLM's own function.
    assert entered.call_count == 1
    monkeypatch.setattr(tpu_runner, "set_forward_context", fc.set_forward_context)
    r.__dict__.pop("_kv_setup_noop_static", None)
    calls = []
    real = fc.set_forward_context

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(fc, "set_forward_context", spy)
    monkeypatch.setattr(tpu_runner, "set_forward_context", spy)
    monkeypatch.setattr(tpu_runner, "_forward_ctx_source_matches", lambda: True)
    r._setup_kv_connector_for_step(SimpleNamespace(kv_connector_metadata=None))
    assert (fc._forward_context, dict(fc.batchsize_forward_time)) == before
    assert len(calls) == (0 if flag else 1)


def test_kv_connector_skip_real_source_pinned():
    # The installed vLLM is the source the exactness argument was made on.
    assert tpu_runner._forward_ctx_source_matches()
    r = _kv_runner()
    assert r._kv_connector_setup_is_noop() is False  # flag off by default


@pytest.mark.parametrize("flag", [False, True])
def test_kv_connector_skip_keeps_connector_path(monkeypatch, flag):
    """With a KV transfer group the block runs as before: the connector binds
    this step's metadata under a forward context, which is restored after."""
    from unittest.mock import MagicMock

    r = _kv_runner()
    connector = MagicMock()
    monkeypatch.setattr(tpu_runner.dist_utils, "get_raiden_inline_load", lambda: False)
    before, after, seen, so = _kv_step(monkeypatch, r, flag, connector)
    assert after == before
    assert len(seen) == 1 and seen[0][1] is so.kv_connector_metadata
    assert seen[0][2] is not None and seen[0][2] is not before[0]
    connector.handle_preemptions.assert_called_once_with(so.kv_connector_metadata)
    connector.start_load_kv.assert_called_once_with(None)


def test_kv_connector_skip_negative_control(monkeypatch):
    """A broken skip that ignores the transfer group drops the connector's
    metadata bind, which the connector test catches."""
    monkeypatch.setattr(
        tpu_runner.TPUModelRunner,
        "_kv_connector_setup_is_noop",
        lambda self: envs.VLLM_TPU_KV_CONNECTOR_SKIP_NOOP,
    )
    with pytest.raises(AssertionError):
        test_kv_connector_skip_keeps_connector_path(monkeypatch, True)


# --------------------------------------------------------------------------
# F. VLLM_TPU_PREP_FOLD_PRECHECK_MEMO
# --------------------------------------------------------------------------

F_FLAG = {"VLLM_TPU_PREP_FOLD_PRECHECK_MEMO": "1"}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fold_precheck_memo_bitwise(monkeypatch, seed):
    _check_coverage(run_pair(monkeypatch, F_FLAG, seed))


def test_fold_precheck_memo_hits_and_invalidates(monkeypatch):
    """Steady hits with the same rows reuse the indices (not recomputed);
    moved rows, a new batch or a new token slice recompute them."""
    monkeypatch.setattr(tpu_runner, "_prep_fold_compiled", _fake_fold)
    monkeypatch.setattr(tpu_runner, "_prep_fold_key", lambda args, static: _WARM)
    r = make_fold_runner()
    computed = []
    real = tpu_runner.TPUModelRunner._prepare_async_token_substitution_indices

    def spy(self, *a):
        out = real(self, *a)
        computed.append(out)
        return out

    monkeypatch.setattr(
        tpu_runner.TPUModelRunner, "_prepare_async_token_substitution_indices", spy
    )
    wl = Workload(7, 4, k=K, max_model_len=8192, churn=0.0, prefill=0.0)
    log = []
    for step in range(10):
        if step:
            wl.step()
        if step == 5:
            wl._add()
        num_decode, num_windowed = wl.segments()
        apply(wl, r, step)  # steps 2 and 7: rows rotated
        computed.clear()
        with env(steady=True, extra={**FOLD_ENV, **F_FLAG}):
            out = run_step(r, wl.scheduler_output(), num_decode, num_windowed)
        log.append(
            (len(computed), out["fold_calls"][0][0] if out["fold_calls"] else None)
        )
    recomputed = [n for n, _ in log]
    # 0: all requests new (no previous rows, no precheck). 1: first steady
    # step (miss, seeds the entry). 2: rows moved. 3: moved back. 4: memo.
    # 5: a new request (no steady key). 6: steady miss. 7: moved. 8: moved
    # back. 9: memo.
    assert recomputed == [0, 1, 1, 1, 0, 1, 1, 1, 1, 0], recomputed
    assert log[4][1] is not None and log[9][1] is not None  # memo steps folded
    entry = r._steady_decode_cache.entry
    assert entry.fold_subst is not None
    assert not entry.fold_subst[0].flags.writeable


def test_fold_precheck_memo_negative_control(monkeypatch):
    """A broken memo key without the previous-step rows is caught."""

    def broken(self, steady, subtract_key, layout_plan):
        if steady is None or subtract_key is None:
            return None
        return steady, (subtract_key[0],)

    monkeypatch.setattr(tpu_runner.TPUModelRunner, "_fold_subst_memo", broken)
    with pytest.raises(AssertionError, match="MISMATCH"):
        run_pair(monkeypatch, F_FLAG, seed=0)


# --------------------------------------------------------------------------
# C. VLLM_TPU_PREP_LAZY_INPUT_IDS
# --------------------------------------------------------------------------

C_FLAG = {"VLLM_TPU_PREP_LAZY_INPUT_IDS": "1"}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lazy_input_ids_bitwise(monkeypatch, seed):
    _check_coverage(run_pair(monkeypatch, C_FLAG, seed))


def test_lazy_input_ids_uploads_only_when_read(monkeypatch):
    """1:1 folds upload no input ids (runner.input_ids stays unset: nothing
    reads it); the scatter fold and every unfolded step upload exactly once,
    the same values the flag-off step uploads."""
    monkeypatch.setattr(tpu_runner, "_prep_fold_compiled", _fake_fold)
    monkeypatch.setattr(tpu_runner, "_prep_fold_key", lambda args, static: _WARM)
    uploads = []
    real = tpu_runner.TPUModelRunner._materialize_input_ids

    def spy(self):
        before = self.input_ids
        out = real(self)
        uploads.append(before is None)
        return out

    monkeypatch.setattr(tpu_runner.TPUModelRunner, "_materialize_input_ids", spy)
    wl = Workload(11, 4, k=K, max_model_len=8192, churn=0.1, prefill=0.08)
    on = make_fold_runner()
    off = make_fold_runner()
    kinds = set()
    for step in range(50):
        if step:
            wl.step()
        num_decode, num_windowed = wl.segments()
        apply(wl, on, step)
        apply(wl, off, step)
        so = wl.scheduler_output()
        warm = step % 6 != 5
        uploads.clear()
        with env(steady=True, extra=FOLD_ENV):
            ref = run_step(off, so, num_decode, num_windowed, warm)
        with env(steady=True, extra={**FOLD_ENV, **C_FLAG}):
            out = run_step(on, so, num_decode, num_windowed, warm)
        assert_step_equal(ref, out, step)
        if out["folded"] is None:
            kind = "unfolded"
            assert torch.equal(on.input_ids, off.input_ids), step
        elif len(out["fold_calls"][0][0]) == 9:
            kind = "one_to_one"
            assert on.input_ids is None, step
        else:
            kind = "scatter"
            assert torch.equal(out["fold_calls"][0][0][9], off.input_ids), step
        kinds.add(kind)
        assert uploads == ([] if kind == "one_to_one" else [True]), (step, kind)
    assert kinds == {"unfolded", "one_to_one", "scatter"}


def test_lazy_input_ids_off_paths_unchanged(monkeypatch):
    """Without PREFWD_PREP_FOLD the flag defers nothing."""
    monkeypatch.setattr(tpu_runner, "_prep_fold_compiled", _fake_fold)
    r = make_fold_runner()
    wl = Workload(2, 4, k=K, max_model_len=8192)
    for step in range(3):
        if step:
            wl.step()
        num_decode, num_windowed = wl.segments()
        apply(wl, r, step)
        extra = {**FOLD_ENV, **C_FLAG, "VLLM_TPU_PREFWD_PREP_FOLD": "0"}
        with env(steady=True, extra=extra):
            run_step(r, wl.scheduler_output(), num_decode, num_windowed)
        assert isinstance(r.input_ids, torch.Tensor)


def test_lazy_input_ids_negative_control(monkeypatch):
    """A broken deferred upload (reading the host buffer at the wrong place)
    is caught."""

    def broken(self):
        if self.input_ids is None:
            sl = self._deferred_input_ids_slice
            self.input_ids = self.input_ids_cpu[sl.start + 1 : sl.stop + 1].clone()
        return self.input_ids

    monkeypatch.setattr(tpu_runner.TPUModelRunner, "_materialize_input_ids", broken)
    with pytest.raises(AssertionError, match="MISMATCH"):
        run_pair(monkeypatch, C_FLAG, seed=0)


# --------------------------------------------------------------------------
# All four together.
# --------------------------------------------------------------------------

ALL_FLAGS = {**C_FLAG, **D_FLAG, **F_FLAG}


@pytest.mark.parametrize("seed", [0, 1])
def test_all_cuts_stacked_bitwise(monkeypatch, seed):
    _check_coverage(run_pair(monkeypatch, ALL_FLAGS, seed, steps=80))
