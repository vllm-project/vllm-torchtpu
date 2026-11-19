# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST lever ``adcarry`` (VLLM_TPU_L6_ADAPT_CARRY): H0-1 ADAPT-CARRY (a)-(c)
and POSTFWD-MERGE, one change to the step-inputs programs.

Today an async-spec decode step runs two small compiled programs before the
forward: the post-forward subtract / substitute / extract program in
``_prepare_inputs`` (``tpu_runner._postfwd_subst_fused``, or on a carried step
``decode_carry.carry_fused``), then the Qwen3.8 adapter's ``fused_step_inputs``
in ``execute_model``. The second reads only the first's outputs, the step's
``query_start_loc`` and state indices (both built before the first), its own
n-gram record and two small uploads (the ``[row_start, num_requests, previous
rows...]`` scalars and, with VLLM_TPU_QWEN4_NGRAM_WINDOW=1, the host n-gram
window). With the lever on:

* POSTFWD-MERGE (non-carried fused steps) and H0-1 (b) (carried steps): one
  compiled program, ``merged_postfwd`` / ``merged_carry``, computes both; the
  adapter's ``prepare_inputs`` then returns its parked outputs. The adapter half
  is ``fused_step_inputs`` itself in its functional form (the record comes back
  as outputs and is installed when the adapter consumes the result, so nothing
  changes if it does not). Integer ops only, the same functions as the two
  programs, so the outputs are the same bits.
* H0-1 (c): when the adapter's last call was the previous step's single full
  chunk over the same requests in the same order (the dict ``begin_step``
  swapped in is exactly the one that call filled), every previous row is the
  row index: reuse that row map and a cached ``[0, n, 0..n-1, -1...]`` scalars
  upload instead of the per-request loop.
* H0-1 (a): on a carried step (lever carry: same requests, order and scheduled
  tokens as the previous step, every request mapped), with (c)'s row map, every
  live row scheduling at least ``spec_window`` tokens and every host context at
  least ``width = spec_window - 1 + ngram - 1`` (the window rule), the record the
  previous step wrote covers every n-gram column of every live row
  (``merge_recent_ngram_context``: the device context advanced by
  ``spec_window - rejected``, rejected in [0, K], and the record holds the
  previous context plus ``spec_window`` chunk tokens), so the window's live rows
  are never read. The previous device window is fed instead of a new upload; its
  padded rows are zero (its live-row count is at most this step's), as a fresh
  window's are.

Both merged programs are compiled at warm-up for every shape the post-forward
program warmed (``precompile``); serving never compiles them (a shape that is
not warm takes today's two programs). Before ``warmup_done`` the lever stands
down so the warm-up runs (and warms) today's programs. No existing program
changes, so the flag is compile-ignored.

``VLLM_TPU_L6_ADAPT_CARRY_CHECK=N`` (bounded by VLLM_TPU_L6_CHECK_STEPS) runs
today's path as a reference every N engaged steps and compares on the host (it
synchronizes); a mismatch logs an ERROR, uses the reference values and drops
the lever's caches. Engagement (L6AB line): ``adcarry.merged_{post,carry}``,
``adcarry.separate`` with ``adcarry.why_<reason>``, ``adcarry.taken`` /
``adcarry.dropped``, ``adcarry.rows_{reused,built}``,
``adcarry.win_{reused,uploaded}``, ``adcarry.checked`` / ``adcarry.mismatch``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.decode_carry import carry_subtract_substitute_extract
from vllm_torchtpu.runner.owned_args import static_shape_args_owned
from vllm_torchtpu.runner.tpu_runner_async_output import subtract_substitute_extract

logger = init_logger(__name__)

LEVER = "adcarry"
_N_POST = 11  # subtract_substitute_extract's tensor arguments


# ---------------------------------------------------------------- the merged programs


def _inputs():
    """The Qwen3.8 adapter's module, imported on first use (the runner imports
    this module at start-up; the model package stays import-free until the
    adapter loads)."""
    from vllm_torchtpu.models.vllm.qwen4_exp import inputs

    return inputs


_BODIES: dict[str, Any] = {}


def _bodies() -> dict[str, Any]:
    """The two merged functions, built once ``fused_step_inputs`` is importable."""
    if _BODIES:
        return _BODIES
    fused_step_inputs = _inputs().fused_step_inputs

    def adapter_tail(post, starts, state, store, scalars, p0, p1, p2, c0, c1, c2, *, window, pooled,
                     spec_window, eos_token_id, ngram_size):
        seq_lens, positions, input_ids, drafts = post
        context, history, slots, valid_tokens, record = fused_step_inputs(
            starts,
            seq_lens,
            state,
            store,
            scalars,
            (p0, p1, p2),
            (c0, c1, c2),
            input_ids,
            window=window,
            pooled=pooled,
            spec_window=spec_window,
            eos_token_id=eos_token_id,
            ngram_size=ngram_size,
            functional=True,
        )
        return (seq_lens, positions, input_ids, drafts, context, history, slots, valid_tokens) + tuple(record)

    def postfwd(*args, window, pooled, spec_window, eos_token_id, ngram_size):
        post = subtract_substitute_extract(*args[:_N_POST])
        return adapter_tail(post, *args[_N_POST:], window=window, pooled=pooled, spec_window=spec_window,
                            eos_token_id=eos_token_id, ngram_size=ngram_size)

    def carry(step_delta, *args, window, pooled, spec_window, eos_token_id, ngram_size):
        post = carry_subtract_substitute_extract(step_delta, *args[:_N_POST])
        return adapter_tail(post, *args[_N_POST:], window=window, pooled=pooled, spec_window=spec_window,
                            eos_token_id=eos_token_id, ngram_size=ngram_size)

    _BODIES.update(postfwd=postfwd, carry=carry)
    return _BODIES


def merged_postfwd(*args, **statics):
    """POSTFWD-MERGE: ``subtract_substitute_extract`` (11 tensors), then the
    adapter's ``fused_step_inputs`` on its outputs (starts, state indices,
    store, scalars, previous record x3, current record x3), functional.
    Returns the post-forward 4-tuple, (context, history, slots, valid_tokens)
    and the new record (3 tensors)."""
    return _bodies()["postfwd"](*args, **statics)


def merged_carry(step_delta, *args, **statics):
    """H0-1 (b): ``carry_subtract_substitute_extract``, then ``fused_step_inputs``."""
    return _bodies()["carry"](step_delta, *args, **statics)


_PROGRAMS: dict[str, Any] = {}


def _compiled(kind: str):
    fn = _PROGRAMS.get(kind)
    if fn is None:
        fn = _PROGRAMS[kind] = torch.compile(_bodies()[kind], backend="tpu", fullgraph=True, dynamic=False)
    return fn


def program_postfwd(*args, **statics):
    """The compiled POSTFWD-MERGE program (dynamo's mark_dynamic stamps
    stripped from its arguments, as ``tpu_runner._postfwd_subst_fused``)."""
    return static_shape_args_owned(_compiled("postfwd"))(*args, **statics)


def program_carry(step_delta, *args, **statics):
    return static_shape_args_owned(_compiled("carry"))(step_delta, *args, **statics)


# ---------------------------------------------------------------- switch and counts


def flag_set() -> bool:
    """VLLM_TPU_L6_ADAPT_CARRY is 1 or ab:<classes>: the warm-up compiles the merged programs."""
    cfg = _l6_ab._cfg.get(LEVER)
    return cfg is not None and (cfg[0] or cfg[1] is not None)


def on() -> bool:
    return _l6_ab.ACTIVE and _l6_ab.on(LEVER)


def _note(key: str) -> None:
    _l6_ab.note(LEVER, key)


def _check_every() -> int:
    raw = os.environ.get("VLLM_TPU_L6_ADAPT_CARRY_CHECK", "").strip()
    if not raw:
        return 0
    # VLLM_TPU_L6_CHECK_STEPS bounds every L6 CHECK mode to the first N worker steps.
    return int(raw) if _l6_ab.check_allowed() else 0


def _check_turn(runner: Any) -> bool:
    every = _check_every()
    if not every:
        return False
    n = runner.__dict__.get("_adc_n", 0) + 1
    runner._adc_n = n
    return n % every == 0


def _statics(adapter: Any, window: bool) -> dict:
    return dict(
        window=window,
        pooled=bool(adapter.pooled),
        spec_window=int(adapter.spec_window),
        eos_token_id=adapter.config.eos_token_id,
        ngram_size=int(adapter.config.ngram_size),
    )


def _width(adapter: Any) -> int:
    return max(adapter.spec_window - 1, 0) + adapter.config.ngram_size - 1


# ---------------------------------------------------------------- the adapter half: (a) and (c)


def rows_reusable(rec: Any, ids: tuple, previous_rows: dict, row_start: int, full: bool) -> bool:
    """(c): the previous adapter call was a single full chunk over `ids` and the
    dict ``begin_step`` swapped in as ``_previous_rows`` is the one it filled
    (so it holds exactly {ids[i]: i})."""
    return full and row_start == 0 and rec is not None and rec[1] == ids and previous_rows is rec[2]


def window_reusable(win: Any, rows: int, width: int, n: int, ctx: np.ndarray, nsched: Any, spec_window: int,
                    carried: bool, rows_ok: bool) -> bool:
    """(a): the cached device window (shape, live rows, tensor) may stand in for
    a fresh upload on this step (see the module docstring)."""
    return (
        carried
        and rows_ok
        and win is not None
        and win[0] == (rows, width + 1)
        and win[1] <= n
        and nsched is not None
        and bool(np.all(np.asarray(nsched[:n]) >= spec_window))
        and bool(np.all(ctx >= width))
    )


class Staged:
    """One chunk's adapter-program inputs and its pending row map."""

    __slots__ = ("scalars", "store", "previous", "current", "pending", "ids", "rows_ok", "win_reused")


def _identity_scalars(adapter: Any, n: int, rows: int) -> torch.Tensor:
    cache = adapter.__dict__.setdefault("_adc_idscalars", {})
    key = (n, rows)
    dev = cache.get(key)
    if dev is None:
        host = np.full(2 + rows, -1, dtype=np.int64)
        host[0] = 0
        host[1] = n
        host[2 : 2 + n] = np.arange(n, dtype=np.int64)
        dev = cache[key] = adapter._scalars_upload(host)
    return dev


def _loop_scalars(adapter: Any, req_ids: list, row_start: int, n: int, rows: int) -> tuple[np.ndarray, dict]:
    """Today's scalars loop, without touching the adapter's dicts."""
    host = np.full(2 + rows, -1, dtype=np.int64)
    host[0] = row_start
    host[1] = n
    prev = adapter._previous_rows
    pending = {}
    for i in range(n):
        rid = req_ids[row_start + i]
        host[2 + i] = prev.get(rid, -1)
        pending[rid] = row_start + i
    return host, pending


def _fresh_window(adapter: Any, runner: Any, row_start: int, n: int, rows: int) -> np.ndarray:
    return _inputs().ngram_window_from_host(
        runner.input_batch.token_ids_cpu_tensor.numpy(),
        runner.input_batch.num_computed_tokens_cpu,
        row_start,
        n,
        rows,
        _width(adapter),
        adapter.config.eos_token_id,
    )


def stage(adapter: Any, runner: Any, *, row_start: int, n: int, rows: int, window: bool, carried: bool,
          nsched: Any, token_store: Any) -> Staged:
    """The adapter program's inputs for this chunk with (c) and, on a carried
    step, (a). Side-effect free apart from value caches; the row map is
    applied by ``commit``."""
    req_ids = runner.input_batch.req_ids
    ids = tuple(req_ids[row_start : row_start + n])
    full = row_start == 0 and n == runner.input_batch.num_reqs
    st = Staged()
    st.ids = ids
    st.rows_ok = rows_reusable(adapter.__dict__.get("_adc_rows"), ids, adapter._previous_rows, row_start, full)
    if st.rows_ok:
        st.scalars = _identity_scalars(adapter, n, rows)
        st.pending = adapter._previous_rows
        _note("rows_reused")
    else:
        host, st.pending = _loop_scalars(adapter, req_ids, row_start, n, rows)
        st.scalars = adapter._scalars_upload(host)
        _note("rows_built")
    st.previous, st.current = adapter._recent_buffers(runner.device)
    st.win_reused = False
    if window:
        width = _width(adapter)
        win = adapter.__dict__.get("_adc_win")
        ctx = np.asarray(runner.input_batch.num_computed_tokens_cpu[row_start : row_start + n])
        if window_reusable(win, rows, width, n, ctx, nsched, adapter.spec_window, carried, st.rows_ok):
            st.store = win[2]
            st.win_reused = True
            _note("win_reused")
        else:
            st.store = adapter._host_upload(_fresh_window(adapter, runner, row_start, n, rows))
            adapter._adc_win = ((rows, width + 1), n, st.store)
            _note("win_uploaded")
    else:
        st.store = token_store
    return st


def commit(adapter: Any, runner: Any, st: Staged, row_start: int, n: int) -> None:
    """Apply the chunk's row map (today's loop writes it into _current_rows) and
    remember the call for the next step's (c)."""
    adapter._current_rows.update(st.pending)
    if row_start == 0 and n == runner.input_batch.num_reqs:
        adapter._adc_rows = (_l6_ab.step_seq(), st.ids, adapter._current_rows)
    else:
        adapter.__dict__.pop("_adc_rows", None)


def drop_caches(adapter: Any) -> None:
    for k in ("_adc_rows", "_adc_win", "_adc_idscalars", "_adc_parked"):
        adapter.__dict__.pop(k, None)


def _check_rows(adapter: Any, runner: Any, st: Staged, row_start: int, n: int, rows: int) -> Staged:
    """CHECK on a (c) step of the unmerged path: today's loop must give the same
    scalars and row map."""
    host, pending = _loop_scalars(adapter, runner.input_batch.req_ids, row_start, n, rows)
    _note("checked")
    if np.array_equal(st.scalars.cpu().numpy(), host) and pending == dict(st.pending):
        return st
    _note("mismatch")
    logger.error("q38 L6 ADAPT_CARRY CHECK MISMATCH step=%d fields=scalars,rows; this step uses the loop's "
                 "values and the lever's caches restart", _l6_ab.step_seq())
    drop_caches(adapter)
    st.scalars = adapter._scalars_upload(host)
    st.pending = pending
    st.rows_ok = False
    return st


# ---------------------------------------------------------------- the runner half


class Ctx:
    """This chunk's merge context, built by ``begin`` in ``_prepare_inputs``."""

    __slots__ = ("num_reqs", "nsched", "starts", "state")

    def __init__(self, num_reqs, nsched, starts, state):
        self.num_reqs = num_reqs
        self.nsched = nsched
        self.starts = starts
        self.state = state


def _refusal(runner: Any, start_index: int, num_reqs: int) -> str:
    if not getattr(runner, "warmup_done", False):
        return "warmup"
    adapter = getattr(runner, "model_adapter", None)
    if adapter is None or not hasattr(adapter, "_recent_buffers") or not getattr(adapter, "spec_window", 0):
        return "adapter"
    if (
        not envs.VLLM_TPU_QWEN4_FUSED_PREP
        or envs.VLLM_TPU_QWEN4_NGRAM_WINDOW == "check"
        or envs.VLLM_TPU_QWEN4_FUSED_PREP_CHECK
        or getattr(runner, "supports_mm_inputs", False)
        or getattr(runner, "uses_mrope", False)
        or getattr(runner, "_pcp_enabled", False)
    ):
        return "config"
    if start_index != 0 or num_reqs != runner.input_batch.num_reqs:
        return "chunk"
    return ""


def begin(runner: Any, *, start_index: int, num_reqs: int, nsched: np.ndarray, query_start_loc: Any,
          state_indices: Any) -> Ctx | None:
    """Whether this chunk may run the merged program; counts a refusal."""
    adapter = getattr(runner, "model_adapter", None)
    if adapter is not None:
        adapter.__dict__.pop("_adc_parked", None)  # a result lives from here to its prepare_inputs
    why = _refusal(runner, start_index, num_reqs)
    if why:
        _note("separate")
        _note("why_" + why)
        return None
    return Ctx(num_reqs, nsched, query_start_loc, state_indices)


def _refuse(why: str) -> None:
    _note("separate")
    _note("why_" + why)
    return None


def warm_key(caller_key: tuple, window: bool) -> tuple:
    return (caller_key, window)


class Parked:
    __slots__ = ("step", "n", "ids", "seq_lens", "starts", "state", "store", "window", "outs", "record",
                 "staged")


def run_merged(runner: Any, adc: Ctx, kind: str, args: tuple, caller_key: tuple, step_delta: int = 0):
    """The merged program in place of the post-forward (``kind="postfwd"``) or
    carry (``"carry"``) program: returns their 4-tuple and parks the adapter's
    outputs, or None (nothing ran) to take today's program."""
    adapter = runner.model_adapter
    rows = int(args[0].shape[0])
    starts = adc.starts
    if starts is None or starts.dim() != 1 or int(starts.shape[0]) != rows + 1:
        return _refuse("shape")
    state = starts if adapter.pooled else adc.state
    if state is None:
        return _refuse("state")
    window = envs.VLLM_TPU_QWEN4_NGRAM_WINDOW == "1"
    if warm_key(caller_key, window) not in runner.__dict__.get("_adc_warm_keys", ()):
        return _refuse("cold")  # serving never compiles a merged shape
    st = stage(adapter, runner, row_start=0, n=adc.num_reqs, rows=rows, window=window, carried=kind == "carry",
               nsched=adc.nsched, token_store=adapter.token_store)
    full = tuple(args) + (starts, state, st.store, st.scalars) + tuple(st.previous) + tuple(st.current)
    statics = _statics(adapter, window)
    if kind == "carry":
        out = program_carry(step_delta, *full, **statics)
    else:
        out = program_postfwd(*full, **statics)
    if _check_turn(runner):
        out, st = _check_merged(runner, adapter, kind, args, step_delta, out, st, starts, state, window, rows)
    p = Parked()
    p.step = _l6_ab.step_seq()
    p.n = adc.num_reqs
    p.ids = out[2]
    p.seq_lens = out[0]
    p.starts = starts
    p.state = state
    p.store = st.store
    p.window = window
    p.outs = tuple(out[4:8])
    p.record = tuple(out[8:11])
    p.staged = st
    adapter._adc_parked = p
    _note("merged_" + ("carry" if kind == "carry" else "post"))
    return tuple(out[:4])


def discard(runner: Any) -> None:
    """Drop this chunk's parked adapter outputs (the post-forward values were
    replaced, e.g. by the carry CHECK's reference)."""
    adapter = getattr(runner, "model_adapter", None)
    if adapter is not None and adapter.__dict__.pop("_adc_parked", None) is not None:
        _note("dropped")


def _reference(runner, adapter, kind, args, step_delta, starts, state, window, rows, n):
    """Today's path on the same inputs: the post-forward / carry program, the
    loop's scalars, a fresh window and the adapter's own program."""
    from vllm_torchtpu.runner import decode_carry, tpu_runner

    if kind == "carry":
        post = decode_carry.carry_fused(step_delta, *args)
    else:
        post = tpu_runner._postfwd_subst_fused(*args)
    host, pending = _loop_scalars(adapter, runner.input_batch.req_ids, 0, n, rows)
    scalars = adapter._host_upload(host)
    store = adapter._host_upload(_fresh_window(adapter, runner, 0, n, rows)) if window else adapter.token_store
    previous, current = adapter._recent_buffers(runner.device)
    functional = envs.VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL
    cur_in = current if functional else tuple(t.clone() for t in current)
    ad = adapter._fused_fn()(
        starts, post[0], state, store, scalars, previous, cur_in, post[2], **_statics(adapter, window),
        functional=functional,
    )
    record = tuple(ad[4]) if functional else cur_in
    return tuple(post) + tuple(ad[:4]) + record, host, pending, store


def _check_merged(runner, adapter, kind, args, step_delta, out, st, starts, state, window, rows):
    n = len(st.ids)
    ref, host, pending, store = _reference(runner, adapter, kind, args, step_delta, starts, state, window, rows, n)
    names = ("seq_lens", "positions", "input_ids", "draft_token_ids", "context", "history", "slots",
             "valid_tokens", "rec_tokens", "rec_context", "rec_lengths")
    bad = [nm for nm, a, b in zip(names, out, ref) if not torch.equal(a.cpu(), b.cpu())]
    if not np.array_equal(st.scalars.cpu().numpy(), host):
        bad.append("scalars")
    if dict(st.pending) != pending:
        bad.append("rows")
    _note("checked")
    if not bad:
        return out, st
    _note("mismatch")
    logger.error(
        "q38 L6 ADAPT_CARRY CHECK MISMATCH step=%d kind=%s fields=%s; this step uses today's values and the "
        "lever's caches restart", _l6_ab.step_seq(), kind, ",".join(bad),
    )
    drop_caches(adapter)
    st.scalars = adapter._host_upload(host)
    st.pending = pending
    st.store = store
    st.rows_ok = False
    return ref, st


def parked_matches(p: Any, adapter: Any, snapshot: Any, ctx: Any, window: bool) -> bool:
    """The parked result is this chunk's: same step, chunk, input ids and the
    exact metadata tensors the merged program read."""
    return (
        p.step == _l6_ab.step_seq()
        and snapshot.row_start == 0
        and snapshot.num_requests == p.n
        and snapshot.input_ids is p.ids
        and snapshot.num_tokens == int(p.ids.shape[0])
        and ctx.seq_lens is p.seq_lens
        and ctx.query_start_loc is p.starts
        and (adapter.pooled or ctx.mamba_state_indices is p.state)
        and p.window == window
        and (window or snapshot.token_store is p.store)
    )


def adapter_prepare(adapter: Any, snapshot: Any):
    """``Qwen4ExpAdapter._prepare_inputs_fused`` with the lever on: the parked
    merged result when it is this chunk's, else today's program with (c).
    None (nothing done) leaves the call to today's path."""
    if envs.VLLM_TPU_QWEN4_FUSED_PREP_CHECK or not adapter.spec_window:
        adapter.__dict__.pop("_adc_parked", None)
        return None
    runner = adapter.runner
    ctx = runner._attn_metadata_builder_ctx
    starts = ctx.query_start_loc
    rows = starts.shape[0] - 1
    window = envs.VLLM_TPU_QWEN4_NGRAM_WINDOW == "1"
    p = adapter.__dict__.pop("_adc_parked", None)
    if p is not None:
        if parked_matches(p, adapter, snapshot, ctx, window):
            # Install this step's record (the merged program returned it functionally).
            adapter._recent[1] = p.record
            if envs.VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL:
                adapter._q38_count_functional()
            commit(adapter, runner, p.staged, 0, p.n)
            _note("taken")
            context, history, slots, valid_tokens = p.outs
            return _kwargs(adapter, snapshot, rows, starts, context, history, slots, valid_tokens)
        _note("dropped")
    st = stage(adapter, runner, row_start=snapshot.row_start, n=snapshot.num_requests, rows=rows,
               window=window, carried=False, nsched=None, token_store=snapshot.token_store)
    if st.rows_ok and _check_turn(runner):
        st = _check_rows(adapter, runner, st, snapshot.row_start, snapshot.num_requests, rows)
    functional = envs.VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL
    out = adapter._fused_fn()(
        starts,
        ctx.seq_lens,
        starts if adapter.pooled else ctx.mamba_state_indices,
        st.store,
        st.scalars,
        st.previous,
        st.current,
        snapshot.input_ids,
        **_statics(adapter, window),
        functional=functional,
    )
    context, history, slots, valid_tokens = out[:4]
    if functional:
        adapter._recent[1] = out[4]
        adapter._q38_count_functional()
    commit(adapter, runner, st, snapshot.row_start, snapshot.num_requests)
    return _kwargs(adapter, snapshot, rows, starts, context, history, slots, valid_tokens)


def _kwargs(adapter, snapshot, rows, starts, context, history, slots, valid_tokens) -> dict:
    tables = adapter._block_tables(snapshot, rows)
    return dict(
        query_start_loc=starts,
        ngram_context=history,
        state_slots=slots,
        context_lens=context,
        valid_tokens=valid_tokens,
        main_block_table=tables[0],
        compressed_block_table=tables[1],
    )


# ---------------------------------------------------------------- warm-up


def precompile(runner: Any) -> int:
    """Compile both merged programs for every shape the post-forward program
    warmed (the carry form with the carry's step delta), so an A/B window never
    compiles while serving. Returns the number of programs compiled."""
    adapter = getattr(runner, "model_adapter", None)
    keys = sorted(runner.__dict__.get("_postfwd_warm_keys", ()), key=repr)
    if (
        not keys
        or adapter is None
        or not hasattr(adapter, "_recent_buffers")
        or not getattr(adapter, "spec_window", 0)
        or runner.speculative_config is None
    ):
        return 0
    device = runner.device
    window = envs.VLLM_TPU_QWEN4_NGRAM_WINDOW == "1"
    statics = _statics(adapter, window)
    step_delta = 1 + int(runner.speculative_config.num_speculative_tokens)
    qsl_dtype = runner.query_start_loc_cpu.dtype
    state_cpu = getattr(runner, "mamba_state_indices_cpu", None)
    state_dtype = state_cpu.dtype if state_cpu is not None else torch.int32
    previous, current = adapter._recent_buffers(device)
    warm = runner.__dict__.setdefault("_adc_warm_keys", set())
    n = 0
    with torch.no_grad():
        for key in keys:
            rows = int(key[0][0][0])
            post = tuple(torch.zeros(shape, dtype=dtype).to(device) for shape, dtype in key)
            starts = torch.zeros(rows + 1, dtype=qsl_dtype).to(device)
            state = starts if adapter.pooled else torch.zeros(rows, dtype=state_dtype).to(device)
            if window:
                store = torch.zeros((rows, _width(adapter) + 1), dtype=torch.int64).to(device)
            else:
                t = runner.input_batch.token_ids_cpu_tensor
                store = torch.zeros(tuple(t.shape), dtype=t.dtype).to(device)
            host = np.full(2 + rows, -1, dtype=np.int64)
            host[:2] = 0
            scalars = torch.from_numpy(host).to(device)
            tail = (starts, state, store, scalars) + tuple(previous) + tuple(current)
            for kind, caller_key in (("postfwd", tuple(key)), ("carry", (step_delta,) + tuple(key))):
                try:
                    if kind == "carry":
                        out = program_carry(step_delta, *post, *tail, **statics)
                    else:
                        out = program_postfwd(*post, *tail, **statics)
                    if device.type != "cpu":
                        from vllm_torchtpu.utils import synchronize_tensors

                        synchronize_tensors(list(out))
                except Exception:
                    # The shape stays cold: serving takes today's two programs for it.
                    logger.warning("q38 L6 ADAPT_CARRY: %s program compile failed for %s", kind, key,
                                   exc_info=True)
                    continue
                warm.add(warm_key(caller_key, window))
                n += 1
    logger.info(
        "q38 L6 ADAPT_CARRY: precompiled %d merged programs for %d post-forward shapes (window=%s pooled=%s)",
        n, len(keys), window, adapter.pooled,
    )
    return n
