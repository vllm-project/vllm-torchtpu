# SPDX-License-Identifier: Apache-2.0
"""Reads the non-output ranks' step replies one call late in the engine core (VLLM_TPU_FANIN_LATE).

With a KV connector, vLLM's ``MultiprocExecutor.collective_rpc`` sends ``execute_model`` and ``sample_tokens`` with
``output_rank=None``: every worker replies, and the call's ``get_response`` dequeues all ranks, in rank order, before
the ``KVOutputAggregator`` merges their ``kv_connector_output`` onto the output rank's reply
(vllm/v1/executor/multiproc_executor.py:399-444). On plain B1 decode steps the engine then spends about 0.79 ms
waiting for ranks 1-7 after rank 0's reply has arrived, plus about 0.45 ms unpickling their replies (q38 SV-EC, T64n
probe, descriptive), on the path from a step's tokens to the next dispatch.

With the key on, a step RPC's ``get_response`` reads only the output rank's reply and returns it. The other ranks'
replies of that call are read at the start of the next ``get_response`` (any RPC), before anything else is read, so each
rank's FIFO reply queue stays in order. In the batch-queue engine that next read starts after the next step has been
dispatched, while the main thread waits for the device. The late replies' ``kv_connector_output`` goes to the
aggregator with the next step RPC whose output-rank reply is not empty.

What stays the same:
- The tokens the scheduler sees are the output rank's reply object, as in stock vLLM (multiproc_executor.py:404-411;
  kv_connector/utils.py:162-176 returns ``outputs[output_rank]`` and replaces only its ``kv_connector_output``).
- Each rank's report is counted once by the aggregator, whose per-request remaining counts persist across calls
  (kv_connector/utils.py:56-118).
- Late replies of a call whose output-rank reply is empty are dropped, as stock drops them (the aggregator returns
  early on an empty output-rank reply).

What moves: a KV transfer reported finished by the last of the non-output ranks reaches the scheduler one step RPC later,
so a landed request can be admitted one step later (schedule-only). A worker error from a non-output rank surfaces one
call later.

``VLLM_TPU_FANIN_LATE=1`` turns it on (default off: nothing is patched). ``VLLM_TPU_FANIN_LATE_AB_MS=<P>`` (0 = off):
in-process A/B in wall-clock windows, late reads in even ``time_ns // P ms`` windows and stock reads in odd ones; P must
divide 86,400,000 with an even quotient. Both keys are host-only and excluded from the compile key. Engine-core process
only. The patch is refused (logged, nothing installed) if ``MultiprocExecutor.collective_rpc`` is not vLLM's own
function or its source differs from the one this copy was made from.
"""

import collections
import hashlib
import inspect
import pickle
import time

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DAY_MS = 86_400_000
LATE_METHODS = ("execute_model", "sample_tokens")
# sha256 of vLLM 0.31's MultiprocExecutor.collective_rpc source, blank lines dropped and lines right-stripped.
SOURCE_SHA256 = "e24890821c673f3e62b74979f59e267acc4c71a1ea057f010c2a034d0441c4ab"
LOG_EVERY = 5000
_PATCH_FLAG = "_tpu_fanin_late_patch"

ENABLED = False
AB_MS = 0
_state = {"ab_window": None}
stats = {"late": 0, "full": 0, "drained": 0, "carried": 0, "dropped": 0}


def _load() -> None:
    global ENABLED, AB_MS
    ENABLED = envs.VLLM_TPU_FANIN_LATE
    AB_MS = envs.VLLM_TPU_FANIN_LATE_AB_MS
    if AB_MS and (DAY_MS % AB_MS or (DAY_MS // AB_MS) % 2):
        raise ValueError(
            f"Invalid value '{AB_MS}' for VLLM_TPU_FANIN_LATE_AB_MS: must be 0 or divide {DAY_MS} with an even quotient."
        )


_load()


def enabled() -> bool:
    return bool(ENABLED)


def _now_ns() -> int:
    return time.time_ns()


def ab_on() -> bool:
    """Whether late reads are on in the current A/B window (always, without AB)."""
    if AB_MS <= 0:
        return True
    k = _now_ns() // (AB_MS * 1_000_000)
    if k != _state["ab_window"]:
        _state["ab_window"] = k
        logger.info(
            "FANIN_LATE AB window=%d on=%d late=%d full=%d drained=%d carried=%d dropped=%d",
            k, int(k % 2 == 0), stats["late"], stats["full"], stats["drained"], stats["carried"], stats["dropped"],
        )
    return k % 2 == 0


def source_sha256(fn) -> str:
    src = inspect.getsource(fn)
    norm = "\n".join(line.rstrip() for line in src.splitlines() if line.strip())
    return hashlib.sha256(norm.encode()).hexdigest()


def _st(executor) -> dict:
    st = executor.__dict__.get("_tpu_fanin_late")
    if st is None:
        # pending: (ranks still to read, keep their replies) per earlier call, in call order; carry: late replies not yet
        # aggregated.
        st = {"pending": collections.deque(), "carry": []}
        executor.__dict__["_tpu_fanin_late"] = st
    return st


def make_collective_rpc():
    """vLLM 0.31 ``MultiprocExecutor.collective_rpc`` with late reads of the non-output ranks' step replies."""
    import cloudpickle
    from vllm.v1.executor.multiproc_executor import FutureWrapper, WorkerProc

    def collective_rpc(  # type: ignore[override]
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block=False,
        unique_reply_rank=None,
        kv_output_aggregator=None,
        ec_output_aggregator=None,
    ):
        assert self.rpc_broadcast_mq is not None, "collective_rpc should not be called on follower node"
        if self.is_failed:
            raise RuntimeError("Executor failed.")
        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}
        st = _st(self)
        aggregators = [a for a in (kv_output_aggregator, ec_output_aggregator) if a]
        rank = unique_reply_rank or 0
        output_rank = None if aggregators else unique_reply_rank
        # Pipeline stages read replies outside get_response (core/pp_lazy_step.py): stock reads there.
        pp = getattr(getattr(getattr(self, "vllm_config", None), "parallel_config", None), "pipeline_parallel_size", 1)
        late_ok = bool(aggregators) and method in LATE_METHODS and pp == 1
        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
        mqs = self.response_mqs

        def read(r, check=True):
            dequeue_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                status, result = mqs[r].dequeue(timeout=dequeue_timeout)
            except TimeoutError as e:
                raise TimeoutError(f"RPC call to {method} timed out.") from e
            if check and status != WorkerProc.ResponseStatus.SUCCESS:
                raise RuntimeError(
                    f"Worker failed with error '{result}', please check the stack trace above for the root cause"
                )
            return result

        def drain():
            """Earlier calls' unread replies, oldest call first (each rank's queue is FIFO)."""
            while st["pending"]:
                ranks, keep = st["pending"].popleft()
                for r in ranks:
                    result = read(r)
                    stats["drained"] += 1
                    if keep and result:
                        st["carry"].append(result)
                        stats["carried"] += 1
                    elif result:
                        stats["dropped"] += 1

        def get_response():
            drain()
            if output_rank is not None:
                return read(output_rank)
            if late_ok and ab_on():
                result = read(rank)
                st["pending"].append(([r for r in range(len(mqs)) if r != rank], bool(result)))
                stats["late"] += 1
                if stats["late"] % LOG_EVERY == 0:
                    logger.info(
                        "FANIN_LATE late=%d full=%d drained=%d carried=%d dropped=%d",
                        stats["late"], stats["full"], stats["drained"], stats["carried"], stats["dropped"],
                    )
                return [result]
            if late_ok:
                stats["full"] += 1
            return [read(r) for r in range(len(mqs))]

        if aggregators:

            def aggregate(outputs):
                idx = rank if len(outputs) == len(mqs) else 0
                if not outputs[idx]:
                    return outputs[idx]  # as stock: an empty output-rank reply skips the merge; the carry waits
                merged = list(outputs) + st["carry"]
                st["carry"] = []
                result = outputs[idx]
                for a in aggregators:
                    result = a.aggregate(merged, output_rank=idx)
                return result

        else:

            def aggregate(outputs):
                return outputs

        future = FutureWrapper(self.futures_queue, get_response=get_response, aggregate=aggregate)
        return future if non_block else future.result()

    setattr(collective_rpc, _PATCH_FLAG, True)
    return collective_rpc


def patch_fanin_late() -> bool:
    """engine_core stage; a no-op (nothing imported, nothing patched) unless VLLM_TPU_FANIN_LATE is set."""
    if not enabled():
        return False
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    orig = MultiprocExecutor.collective_rpc
    if getattr(orig, _PATCH_FLAG, False):
        return True
    if getattr(orig, "__module__", "") != "vllm.v1.executor.multiproc_executor" or hasattr(orig, "__wrapped__"):
        logger.warning(
            "VLLM_TPU_FANIN_LATE refused: MultiprocExecutor.collective_rpc is already wrapped; replies are read as before"
        )
        return False
    sha = source_sha256(orig)
    if sha != SOURCE_SHA256:
        logger.warning(
            "VLLM_TPU_FANIN_LATE refused: MultiprocExecutor.collective_rpc source %s differs from %s; "
            "replies are read as before",
            sha[:12], SOURCE_SHA256[:12],
        )
        return False
    fn = make_collective_rpc()
    fn.__wrapped__ = orig
    MultiprocExecutor.collective_rpc = fn
    _patch_engine_core_proc()
    logger.info(
        "VLLM_TPU_FANIN_LATE=1: step replies of the non-output ranks are read one call late (AB %d ms)", AB_MS
    )
    return True


def engaged(executor) -> str:
    """'' when late reads apply to this executor's step RPCs, else why not."""
    if executor is None:
        return "no model executor"
    if not getattr(getattr(type(executor), "collective_rpc", None), _PATCH_FLAG, False):
        return "executor %s does not use the patched MultiprocExecutor.collective_rpc" % type(executor).__name__
    if getattr(executor, "kv_output_aggregator", None) is None:
        return "no KV connector aggregator (every step RPC has a single reply)"
    pp = getattr(getattr(getattr(executor, "vllm_config", None), "parallel_config", None), "pipeline_parallel_size", 1)
    if pp != 1:
        return "pipeline parallel size %s" % pp
    return ""


def _patch_engine_core_proc() -> None:
    """The engagement line: logged by the engine core process once its engine is built (patch-time lines are logged
    before its logging reaches the served log; SV-EC DEFER-FIN 1a676ef719 pattern)."""
    from vllm.v1.engine.core import EngineCoreProc

    init = EngineCoreProc.__init__
    if getattr(init, _PATCH_FLAG, False):
        return

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        ex = getattr(self, "model_executor", None)
        why = engaged(ex)
        if why:
            logger.warning("FANIN_LATE not engaged: %s; replies are read as before", why)
        else:
            logger.info(
                "FANIN_LATE engaged: non-output ranks' step replies read one call late "
                "(%d workers, output rank %s, %s; AB %d ms)",
                len(ex.response_mqs), getattr(ex, "output_rank", 0), type(ex).__name__, AB_MS,
            )

    setattr(__init__, _PATCH_FLAG, True)
    __init__.__wrapped__ = init
    EngineCoreProc.__init__ = __init__
