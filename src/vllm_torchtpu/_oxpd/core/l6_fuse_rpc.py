# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever ``fuserpc`` (VLLM_TPU_L6_FUSE_RPC): execute_model + sample_tokens as one RPC.

host_levers_h1.md §4.2 FUSE-RPC (hunt_0 v1 HT-4). With the batch queue, the engine core sends
``execute_model`` and then, whenever no structured-output tokens are pending, ``sample_tokens``
back to back (vLLM 0.31.0 ``EngineCore.step_with_batch_queue``, core.py:697-717). Each RPC costs
the engine a broadcast enqueue and the workers a dequeue, a decode and a dispatch, and (with a KV
connector, where every rank replies) eight ``None`` replies to ``execute_model`` that the engine
dequeues and unpickles. With the lever on, a step that would send both with ``grammar_output`` None
sends one ``execute_and_sample`` RPC; the worker runs exactly ``execute_model`` then
``sample_tokens(None)`` on its main thread and replies once, and one future serves both batch-queue
slots.

When it applies (all read after ``schedule()``, per engine-core step): the lever is on for this
step (``l6_ab.on`` on the engine core; ``begin_step('engine_core')`` runs in ``schedule()``), the
model is not a pooling model, the step scheduled tokens (``model_executed``), and the step has no
structured-output request (``has_structured_output_requests`` False, so
``get_grammar_bitmask`` is None, scheduler.py:1948, and nothing is pending). The grammar path, the
deferred path, pooling, empty steps and every other executor keep vLLM's code unchanged.

Installation (engine-core process, before ``EngineCore.__init__`` binds ``step_fn``): only when the
flag is set, only on a single pipeline stage, and only when ``EngineCore.step_with_batch_queue`` and
``MultiprocExecutor.execute_model`` / ``sample_tokens`` / ``collective_rpc`` are vLLM 0.31.0's own
(source hashes below, read on the served image by the L6-MK probe job j-3be93ddd-938e-47d8-853b):
the patched step is a copy of vLLM's method with one added branch, and a copy of anything else
would not be exact. Otherwise the lever stays inert with one ERROR. The worker side
(``WorkerWrapperBase.execute_and_sample``) is installed at worker init when the flag is set.

Exactness of the worker method: ``execute_model`` returns None on every step this lever fuses
(tpu_runner.py:4857-4864 return an output only when no tokens are scheduled; :5395-5414 return
None except for pooling models), and the engine discards that reply's value (vLLM aggregates it to
None: ``KVOutputAggregator.aggregate`` returns None when the output rank's reply is None). If
``execute_model`` raises, stock still runs the worker's ``sample_tokens`` (the next RPC) and the
engine raises the ``execute_model`` error when ``sample_tokens`` returned None; the fused method does
the same: it runs ``sample_tokens`` and re-raises the ``execute_model`` error if that returned None,
so the engine sees the same ``RuntimeError("Worker failed with error '<msg>' ...")``.

Engagement: ``fuserpc.fused`` / ``fuserpc.split`` (an eligible-by-config step the rule kept split,
counted by reason) on the engine core's L6AB window line; one INFO line at install.

Lever ``e2kvonly`` (VLLM_TPU_L6_OUT_KVONLY; host_levers_h1.md §4.4 E2 OUT-KVONLY) rides the fused RPC.
With a KV connector every rank replies and vLLM's ``KVOutputAggregator`` reads only the
``kv_connector_output`` of the ranks other than the output rank (utils.py:66-165; the EC aggregator
reads ``ec_connector_output``), yet each of them waits for its token copy and pickles a full output
(``AsyncTPUModelRunnerOutput.get_output`` on the worker's output thread). On a fused step with the
lever on (read by the engine core, so every rank follows one decision) and a KV aggregator, the RPC
carries the executor's output rank; the other ranks reply ``EMPTY_MODEL_RUNNER_OUTPUT`` carrying their
own ``kv_connector_output`` / ``ec_connector_output`` objects (always the objects, never dropped when
empty: ``KVConnectorOutput.is_empty`` ignores ``expected_finished_count``), built from the runner's
output without the wait. The output rank replies as before. Non-fused steps are unchanged. It needs
``fuserpc`` on.

CHECK (VLLM_TPU_L6_OUT_KVONLY_CHECK=N > 0, bounded by VLLM_TPU_L6_CHECK_STEPS): every Nth kv-only
step is sent with full replies instead, and the engine aggregates them twice, the real aggregator on
the full replies and a deep copy of it on the kv-only form of the same replies; the aggregated output
and both aggregators' pending counts must be equal, else e2kvonly is off for the process (one ERROR,
``e2kvonly.mismatch``). Engagement: ``e2kvonly.sent`` / ``e2kvonly.check_ok`` (engine) and
``e2kvonly.reply`` (each non-output worker).
"""

from __future__ import annotations

import hashlib
import inspect
import os
from typing import Any, cast

from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LEVER = "fuserpc"
ENV = "VLLM_TPU_L6_FUSE_RPC"
KVONLY = "e2kvonly"
KVONLY_CHECK_ENV = "VLLM_TPU_L6_OUT_KVONLY_CHECK"
METHOD = "execute_and_sample"

# sha256[:16] of inspect.getsource on the served image (vLLM 0.31.0).
VLLM_SOURCES = {
    "EngineCore.step_with_batch_queue": "65c2c7342d075e70",
    "MultiprocExecutor.execute_model": "cc58c9531e92455b",
    "MultiprocExecutor.sample_tokens": "f3962f97e3c83a0b",
    "MultiprocExecutor.collective_rpc": "936c16152d712061",
}

_INSTALLED = {"engine": False, "worker": False}


def configured() -> bool:
    return _l6_ab._cfg.get(LEVER, (False, None)) != (False, None)


def _src_hash(fn: Any) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


def vllm_source_hashes() -> dict[str, str]:
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    return {
        "EngineCore.step_with_batch_queue": _src_hash(EngineCore.__dict__["step_with_batch_queue"]),
        "MultiprocExecutor.execute_model": _src_hash(MultiprocExecutor.__dict__["execute_model"]),
        "MultiprocExecutor.sample_tokens": _src_hash(MultiprocExecutor.__dict__["sample_tokens"]),
        "MultiprocExecutor.collective_rpc": _src_hash(MultiprocExecutor.__dict__["collective_rpc"]),
    }


# ------------------------------------------------------------------------------------------ worker
def kv_only(out: Any) -> Any:
    """e2kvonly: an otherwise-empty output carrying `out`'s own KV / EC connector outputs (no wait for
    the token copy); `out` itself for any type it does not know (the full reply, as stock)."""
    import copy

    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

    from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput

    if type(out) is AsyncTPUModelRunnerOutput:
        mro = out._model_runner_output
    elif type(out) is ModelRunnerOutput:
        mro = out
    else:
        return out
    o = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
    o.kv_connector_output = mro.kv_connector_output
    o.ec_connector_output = mro.ec_connector_output
    return o


def execute_and_sample(self: Any, scheduler_output: Any, grammar_output: Any = None, kvonly_rank: Any = None) -> Any:
    """``WorkerWrapperBase.execute_and_sample``: the two RPCs' worker calls, in order, on this
    thread (``self.execute_model`` is the wrapper's, as the ``execute_model`` RPC calls it). With
    `kvonly_rank` (e2kvonly), a rank other than it replies with its KV output only."""
    exc = None
    try:
        out = self.execute_model(scheduler_output)
    except Exception as e:  # stock: this RPC fails, the sample_tokens RPC still runs
        exc = e
        out = None
        logger.exception("q38 L6 FUSE-RPC: execute_model failed on a fused step")
    if out is not None:
        # Not reachable on a fused step (module docstring); stock would have replied it to the
        # engine, which discards it.
        logger.error("q38 L6 FUSE-RPC: execute_model returned %s on a fused step", type(out).__name__)
    sampled = self.sample_tokens(grammar_output)
    if exc is not None and sampled is None:
        raise exc
    if kvonly_rank is not None and sampled is not None and getattr(self, "global_rank", kvonly_rank) != kvonly_rank:
        _l6_ab.note(KVONLY, "reply")
        return kv_only(sampled)
    return sampled


def install_worker() -> bool:
    """Add ``execute_and_sample`` to vLLM's WorkerWrapperBase (idempotent; flag set only)."""
    if _INSTALLED["worker"]:
        return True
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    if not hasattr(WorkerWrapperBase, METHOD):
        setattr(WorkerWrapperBase, METHOD, execute_and_sample)
    _INSTALLED["worker"] = True
    logger.info("q38 L6 FUSE-RPC: worker method %s installed", METHOD)
    return True


# ------------------------------------------------------------------------------------------ engine
def executor_execute_and_sample(self: Any, scheduler_output: Any, grammar_output: Any, non_block: bool = False,
                                kvonly_rank: Any = None, kv_aggregator: Any = None):
    """``MultiprocExecutor.execute_and_sample``: one RPC with execute_model's and sample_tokens'
    reply rule, timeout and aggregators (identical in the two, multiproc_executor.py:340-364).
    `kvonly_rank` / `kv_aggregator`: e2kvonly's output rank and its CHECK aggregator."""
    import vllm.envs as envs

    args = (scheduler_output, grammar_output) if kvonly_rank is None else (scheduler_output, grammar_output, kvonly_rank)
    return self.collective_rpc(
        METHOD,
        args=args,
        unique_reply_rank=self.output_rank,
        non_block=non_block,
        timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        kv_output_aggregator=self.kv_output_aggregator if kv_aggregator is None else kv_aggregator,
        ec_output_aggregator=self.ec_output_aggregator,
    )


_KV = {"check": None, "turn": 0}


def _kv_check_every() -> int:
    v = _KV["check"]
    if v is None:
        raw = os.environ.get(KVONLY_CHECK_ENV, "").strip()
        v = int(raw) if raw else 0
        if v < 0:
            raise ValueError(f"{KVONLY_CHECK_ENV} must be >= 0, got {v}")
        _KV["check"] = v
    return v


def _out_key(o: Any) -> Any:
    kvo = getattr(o, "kv_connector_output", None)
    kv = None if kvo is None else (
        sorted(kvo.finished_sending or ()), sorted(kvo.finished_recving or ()), repr(kvo.kv_connector_stats),
        repr(kvo.kv_cache_events), repr(kvo.kv_connector_worker_meta), sorted(kvo.invalid_block_ids or ()),
        sorted(kvo.failed_recving or ()), kvo.expected_finished_count)
    return (list(getattr(o, "req_ids", []) or []), repr(getattr(o, "sampled_token_ids", None)), kv,
            repr(getattr(o, "ec_connector_output", None)))


class KvOnlyCheckAggregator:
    """e2kvonly CHECK: the real KV aggregator on the full replies, and a deep copy of it on their kv-only
    form; returns the real result and turns the lever off if the two differ."""

    def __init__(self, core: Any, real: Any):
        self.core, self.real = core, real

    def aggregate(self, outputs: list, output_rank: int = 0) -> Any:
        import copy

        shadow = copy.deepcopy(self.real)
        kvo = [copy.deepcopy(o) if (i == output_rank or o is None) else kv_only(copy.deepcopy(o))
               for i, o in enumerate(outputs)]
        got = shadow.aggregate(kvo, output_rank=output_rank)
        want = self.real.aggregate(outputs, output_rank=output_rank)
        same = (want is None) == (got is None) and (want is None or _out_key(want) == _out_key(got))
        same = same and all(getattr(shadow, a, None) == getattr(self.real, a, None) for a in (
            "_recv_remaining_count", "_send_remaining_count", "_failed_recving_pending", "_expected_finished_count"))
        if same:
            _l6_ab.note(KVONLY, "check_ok")
        else:
            self.core._l6_kvonly_off = True
            _l6_ab.note(KVONLY, "mismatch")
            logger.error("q38 L6 OUT-KVONLY: CHECK mismatch between the full and the kv-only aggregation; "
                         "e2kvonly is off for the rest of this process")
        return want


def kvonly_plan(core: Any) -> tuple[Any, Any]:
    """(output rank to send, CHECK aggregator) for this fused step; (None, None) = full replies."""
    ex = core.model_executor
    real = getattr(ex, "kv_output_aggregator", None)
    if real is None or not _l6_ab.on(KVONLY) or core.__dict__.get("_l6_kvonly_off", False):
        return None, None
    every = _kv_check_every()
    if every > 0:
        _KV["turn"] += 1
        if _KV["turn"] >= every:
            _KV["turn"] = 0
            if _l6_ab.check_allowed():
                return None, KvOnlyCheckAggregator(core, real)
    _l6_ab.note(KVONLY, "sent")
    return ex.output_rank, None


def _fuse_ok(core: Any) -> bool:
    """Per-instance, fixed at the first step: a multiproc executor of vLLM's own RPC methods."""
    ok = core.__dict__.get("_l6_fuse_ok")
    if ok is None:
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        ex = core.model_executor
        ok = (
            isinstance(ex, MultiprocExecutor)
            and type(ex).execute_model is MultiprocExecutor.execute_model
            and type(ex).sample_tokens is MultiprocExecutor.sample_tokens
            and type(ex).collective_rpc is MultiprocExecutor.collective_rpc
            and hasattr(ex, "output_rank")
        )
        core._l6_fuse_ok = ok
        logger.info("q38 L6 FUSE-RPC: engine core executor %s fused RPCs %s", type(ex).__name__,
                    "enabled" if ok else "refused (not vLLM's MultiprocExecutor RPC methods)")
    return ok


def step_with_batch_queue(self) -> tuple[dict[int, Any] | None, bool]:
    """vLLM 0.31.0 ``EngineCore.step_with_batch_queue`` (core.py:671-783, copied unchanged) with one
    added branch, marked ``q38 L6 FUSE-RPC``: send ``execute_and_sample`` instead of
    ``execute_model`` + ``sample_tokens(None)`` when the module docstring's rule holds."""
    from concurrent.futures import Future

    from vllm.v1.outputs import ModelRunnerOutput

    batch_queue = self.batch_queue
    assert batch_queue is not None

    # Try to schedule a new batch if the batch queue is not full, but
    # the scheduler may return an empty batch if all requests are scheduled.
    # Note that this is not blocking.
    assert len(batch_queue) < self.batch_queue_size

    model_executed = False
    deferred_scheduler_output = None
    if self.scheduler.has_requests():
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        # q38 L6 FUSE-RPC begin: the rule, read after schedule() began this engine-core step.
        fused = (
            _l6_ab.on(LEVER)
            and not self.is_pooling_model
            and not self.is_mm_encoder_only
            and scheduler_output.total_num_scheduled_tokens > 0
            and not scheduler_output.pending_structured_output_tokens
            and not scheduler_output.has_structured_output_requests
            and _fuse_ok(self)
        )
        if fused:
            # q38 L6 OUT-KVONLY (e2kvonly) rides the fused RPC: the output rank to send, or a CHECK.
            kv_rank, kv_agg = kvonly_plan(self)
            with self.log_error_detail(scheduler_output):
                exec_future = executor_execute_and_sample(
                    self.model_executor, scheduler_output, None, non_block=True,
                    kvonly_rank=kv_rank, kv_aggregator=kv_agg,
                )
            _l6_ab.note(LEVER, "fused")
        else:
            if _l6_ab.ACTIVE and _l6_ab.on(LEVER):
                _l6_ab.note(LEVER, "split")
            # q38 L6 FUSE-RPC end (the stock call follows, unchanged).
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
        if not self.is_mm_encoder_only:
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

        if self.is_pooling_model or not model_executed:
            # No sampling required (no requests scheduled).
            future = cast(Future[ModelRunnerOutput], exec_future)
        elif fused:  # q38 L6 FUSE-RPC: the one future serves both slots
            future = cast(Future[ModelRunnerOutput], exec_future)
        else:
            if not scheduler_output.pending_structured_output_tokens:
                # We aren't waiting for any tokens, get any grammar output
                # and sample immediately.
                grammar_output = self.scheduler.get_grammar_bitmask(
                    scheduler_output
                )
                future = self.model_executor.sample_tokens(
                    grammar_output, non_block=True
                )
            else:
                # We need to defer sampling until we have processed the model output
                # from the prior step.
                deferred_scheduler_output = scheduler_output

        if not deferred_scheduler_output:
            # Add this step's future to the queue.
            batch_queue.appendleft((future, scheduler_output, exec_future))
            if len(batch_queue) < self.batch_queue_size and (
                model_executed or self.scheduler.has_requests()
            ):
                # Don't block on next worker response unless the queue is full
                # or there are no more requests to schedule.
                return None, model_executed

    elif not batch_queue:
        # Queue is empty. We should not reach here since this method should
        # only be called when the scheduler contains requests or the queue
        # is non-empty.
        return None, False

    # Block until the next result is available.
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    with (
        self.capture_iteration_details(scheduler_output) as iteration_details,
        self.log_error_detail(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            # None from sample_tokens() implies that the original execute_model()
            # call failed - raise that exception.
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

    # Before processing the model output, process any aborts that happened
    # during the model execution.
    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    self._attach_iteration_details(engine_core_outputs, iteration_details)

    # NOTE(nick): We can either handle the deferred tasks here or save
    # in a field and do it immediately once step_with_batch_queue is
    # re-called. The latter slightly favors TTFT over TPOT/throughput.
    if deferred_scheduler_output:
        # When draft tokens are used with structured output, validate them
        # before computing the grammar bitmask for the deferred request.
        if self.check_for_draft_tokens:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                # Update the draft token ids in the scheduler output to
                # filter out the invalid spec tokens, which will be padded
                # with -1 and skipped by the grammar bitmask computation.
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
        # We now have the tokens needed to compute the bitmask for the
        # deferred request. Get the bitmask and call sample tokens.
        grammar_output = self.scheduler.get_grammar_bitmask(
            deferred_scheduler_output
        )
        future = self.model_executor.sample_tokens(grammar_output, non_block=True)
        batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

    return engine_core_outputs, model_executed


step_with_batch_queue._l6_fuse = True  # type: ignore[attr-defined]


def install_engine(vllm_config: Any = None) -> bool:
    """Engine-core process, before EngineCore.__init__: route step_with_batch_queue through the copy
    above (idempotent; flag set only; one pipeline stage; vLLM's own sources only)."""
    if _INSTALLED["engine"]:
        return True
    if not configured():
        return False
    pp = getattr(getattr(vllm_config, "parallel_config", None), "pipeline_parallel_size", 1) or 1
    if pp > 1:
        logger.error("q38 L6 FUSE-RPC: NOT installed (pipeline_parallel_size=%d); the lever stays inert", pp)
        return False
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    if getattr(EngineCore.__dict__.get("step_with_batch_queue"), "_l6_fuse", False):
        _INSTALLED["engine"] = True
        return True
    try:
        got = vllm_source_hashes()
    except Exception as exc:  # noqa: BLE001 - no sources: stay inert
        logger.error("q38 L6 FUSE-RPC: NOT installed (%s); the lever stays inert", exc)
        return False
    bad = {k: v for k, v in got.items() if VLLM_SOURCES[k] != v}
    if bad:
        logger.error("q38 L6 FUSE-RPC: NOT installed: vLLM sources differ from the verified ones %s; "
                     "the lever stays inert", bad)
        return False
    EngineCore.step_with_batch_queue = step_with_batch_queue
    if not hasattr(MultiprocExecutor, METHOD):
        MultiprocExecutor.execute_and_sample = executor_execute_and_sample
    # begin_step('engine_core') per schedule(): engalloc's hook when it is set; installed here
    # otherwise (core/l6_engine_alloc.py install_engine_core, idempotent per scheduler).
    from vllm_torchtpu.core import l6_engine_alloc as _ea

    if not _ea.configured() and not getattr(EngineCore, "_q38_l6_fuse_hook", False):
        orig_init = EngineCore.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            try:
                _ea.install_engine_core(self)
            except Exception:  # noqa: BLE001
                logger.exception("q38 L6 FUSE-RPC: engine-core step hook failed; the lever stays off")

        EngineCore.__init__ = __init__
        EngineCore._q38_l6_fuse_hook = True
    _INSTALLED["engine"] = True
    logger.info("q38 L6 FUSE-RPC: installed (%s=%s); EngineCore.step_with_batch_queue routes through "
                "the fused copy", ENV, os.environ.get(ENV, ""))
    return True
