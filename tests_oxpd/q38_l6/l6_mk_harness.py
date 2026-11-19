# SPDX-License-Identifier: Apache-2.0
"""q38 L6-MK test harness: random served-shaped streams for the mkctx / mkthin / fuserpc levers.

Importable on the base tree too (it never imports runner/l6_mk.py or core/l6_fuse_rpc.py), so the
CPU harness can run the same stream on base b488e1cb0 and on head with the flag off and on and
compare digests (``python3 l6_mk_harness.py <stream> <seed>`` prints ``L6MK_TRACE ...``).

* mkctx_stream: a minimal Qwen38SegmentDriver (real try_forward, real _contexts and vLLM context
  managers) over random decode steps: request counts, buckets, context lengths (some past the
  whole-step cap, some rows above the whole-step bucket), warm-up / served traffic, shadow-check
  and wholecheck budgets. The fake programs record the model tag and the config they run under.
* mkthin_stream: real torch_tpu ``_SplitCompiledExecutable`` / ``_WrapperModule`` objects around
  split fx graphs (split_module, as torch_tpu builds them) whose partition is a
  ``TorchTpuCompiledExecutable`` with a recording CPU fast runner; random boxed / unboxed calls.
* RpcRig: vLLM's real MultiprocExecutor RPC path (MessageQueue shm rings, FutureWrapper, the KV
  output aggregator) to forked processes running vLLM's real WorkerProc busy loop (async output
  thread) over a WorkerWrapperBase around a fake worker; the engine side runs an
  ``EngineCore.step_with_batch_queue`` implementation on a fake engine core with a scripted
  scheduler.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
import sys
from collections import deque
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import torch


def digest(obj: Any) -> str:
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:16]


def _l6_begin() -> None:
    from vllm_torchtpu.core import l6_ab

    l6_ab.begin_step()


# ------------------------------------------------------------------------------------------ mkctx
BUCKETS = (8, 16, 24, 32, 48, 64, 96)


class _FakeProg:
    def __init__(self, name: str, rec: list):
        self.name, self.rec = name, rec

    def __call__(self, input_ids=None, positions=None, **kw):
        from vllm.compilation import backends
        from vllm.config import vllm as vcfg

        cfg = vcfg._current_vllm_config
        self.rec.append((self.name, backends.model_tag, getattr(cfg, "label", repr(cfg)),
                         int(input_ids.sum()), tuple(sorted(kw))))
        return input_ids.float() * 2 + positions.float()


class _FakeWholeModule:
    def __init__(self, rec: list):
        self.rec = rec

    def check_gdn_meta_shared(self):
        self.rec.append(("gdn_meta_check",))

    def log_union_stats(self, n):
        self.rec.append(("union_stats", n))


def make_driver(drv: Any, rec: list, check_left: int = 0, wholecheck_left: int = 0, noaot: bool = False):
    d = drv.Qwen38SegmentDriver.__new__(drv.Qwen38SegmentDriver)
    d.whole = True
    d.buckets = BUCKETS
    d.whole_program = _FakeProg("whole", rec)
    d.program = _FakeProg("seg", rec)
    d.whole_above, d.whole_dry = "stock", False
    d.whole_max_rows, d.whole_cap = 64, 4096
    d.stats, d.counters, d.seq, d.mk_calls, d.whole_calls = {}, {}, 0, 0, 0
    d.sab = None
    d.segx_left = 0
    d.h0check_left = 0
    d.warm = False
    d.sync = False
    d.noaot = noaot
    d._whole_checked = True
    d.cfg = NS(label="seg_cfg")
    d.whole_cfg = NS(label="whole_cfg")
    d.whole_module = _FakeWholeModule(rec)
    d._log_gdn_meta = lambda: None
    d.check_left = check_left
    d.shadow = object() if check_left else None
    d.check_rows = (16, 32)
    d.check_warm = set()
    d.wholecheck_left = wholecheck_left
    d.whole_shadow = object() if wholecheck_left else None
    d.wholecheck_rows = (24, 48)
    d.wholecheck_warm = set()

    def shadow_step(runner, input_ids, positions, kw, check_real, n_req):
        d.check_left -= 1
        rec.append(("shadow", bool(check_real), n_req, int(input_ids.shape[0])))
        return input_ids.float() - 1

    def whole_shadow_step(runner, input_ids, positions, kw, rows, real, n_req):
        d.wholecheck_left -= 1
        rec.append(("wshadow", rows, bool(real), n_req))
        return input_ids.float() + 1

    d._shadow_step = shadow_step
    d._whole_shadow_step = whole_shadow_step
    return d


def mkctx_stream(seed: int, steps: int = 400, check_left: int = 3, wholecheck_left: int = 3,
                 noaot: bool = False, l6_steps: bool = True, counts: dict | None = None):
    """Run a random stream through the driver; returns (trace, rec, final state) for a digest.
    `counts` (if given) receives the number of _real_traffic / _whole_eligible evaluations."""
    from vllm_torchtpu.megakernel_runtime import driver as base_drv
    from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv

    rng = random.Random(seed)
    rec: list = []
    d = make_driver(drv, rec, check_left, wholecheck_left, noaot)
    cell = {"n": 1}
    orig_inspect = base_drv.MegakernelDriver.__dict__["_inspect_step_detail"]
    orig_rt = drv._real_traffic
    n_rt, n_we = [0], [0]

    def inspect_stub(*a, **k):
        return True, cell["n"], drv.T_VERIFY, ""

    def rt(*a, **k):
        n_rt[0] += 1
        return orig_rt(*a, **k)

    cls_we = drv.Qwen38SegmentDriver._whole_eligible

    def we(runner, n_req, rows):
        n_we[0] += 1
        return cls_we(d, runner, n_req, rows)

    d._whole_eligible = we
    base_drv.MegakernelDriver._inspect_step_detail = staticmethod(inspect_stub)
    drv._real_traffic = rt
    trace = []
    try:
        runner = NS(input_batch=NS(num_computed_tokens_cpu=np.zeros(128, dtype=np.int32)),
                    warmup_done=False, vllm_config=NS(label="runner_cfg"))
        for step in range(steps):
            if l6_steps:
                _l6_begin()
            n = rng.randint(1, 26)
            cell["n"] = n
            need = n * drv.T_VERIFY
            r = rng.random()
            if r < 0.08:
                rows = rng.choice((20, 40, 100))  # not a bucket
            else:
                rows = next((b for b in BUCKETS if b >= need), 128)
            ctx = np.array([rng.randint(1, 3000) for _ in range(n)], dtype=np.int32)
            if rng.random() < 0.12:
                ctx[rng.randrange(n)] = rng.randint(4090, 5000)  # past the whole-step cap
            if rng.random() < 0.1:
                ctx[:] = rng.randint(1, 64)  # warm-up-like traffic (not real)
            runner.input_batch.num_computed_tokens_cpu[:] = 0
            runner.input_batch.num_computed_tokens_cpu[:n] = ctx
            runner.warmup_done = step >= 6
            ids = torch.arange(rows, dtype=torch.int32) + step
            pos = torch.arange(rows, dtype=torch.int32) * 3
            kw = {"context_lens": torch.tensor(ctx, dtype=torch.int32)} if rng.random() < 0.5 else None
            ok, out = d.try_forward(runner, ids, pos, None, None, kw)
            trace.append((step, n, rows, bool(ok), None if out is None else out.tolist()))
    finally:
        base_drv.MegakernelDriver._inspect_step_detail = orig_inspect
        drv._real_traffic = orig_rt
    if counts is not None:
        counts["real_traffic"] = n_rt[0]
        counts["whole_eligible"] = n_we[0]
    state = (sorted(d.stats.items()), sorted(d.counters.items()), d.whole_calls, d.mk_calls, d.seq,
             d.check_left, d.wholecheck_left, sorted(d.check_warm), sorted(d.wholecheck_warm))
    return trace, rec, state


# ----------------------------------------------------------------------------------------- mkthin
class RunnerLog:
    """Records each fast-runner call: which caller objects it received (by identity)."""

    def __init__(self):
        self.calls: list = []


def _make_exe(sub: torch.fx.GraphModule, log: RunnerLog, name: str, as_list: bool):
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import TorchTpuCompiledExecutable

    exe = TorchTpuCompiledExecutable.__new__(TorchTpuCompiledExecutable)

    def fast_runner(args):
        log.calls.append((name, args))
        out = sub(*args)
        out = list(out) if isinstance(out, (tuple, list)) else [out]
        return out if as_list else tuple(out)

    exe.__dict__["_fast_runner"] = fast_runner
    exe.__dict__["_executable"] = None
    return exe


def build_split_exe(fn: Any, log: RunnerLog, name: str, partitions: int = 1, as_list: bool = False):
    """A torch_tpu _SplitCompiledExecutable over split_module(fn) (one or more TPU partitions),
    stitched as torch_tpu's _SubmodCompiler does (compiled_<submod> = _WrapperModule(exe, unwrap))."""
    from torch.fx.passes.split_module import split_module
    from torch_tpu._internal.compile import split_compiler as sc

    gm = torch.fx.symbolic_trace(fn)
    ops = [n for n in gm.graph.nodes if n.op not in ("placeholder", "output")]
    cut = {n: (0 if i < max(1, len(ops) // partitions) else 1) for i, n in enumerate(ops)}
    split = split_module(gm, None, lambda n: cut.get(n, 0) if partitions > 1 else 0)
    if hasattr(sc, "_call_partitions_in_order"):
        sc._call_partitions_in_order(split)
    for n in list(split.graph.nodes):
        if n.op == "call_module":
            sub = split.get_submodule(str(n.target))
            single = not any(u.target is __import__("operator").getitem for u in n.users)
            exe = _make_exe(sub, log, f"{name}:{n.target}", as_list)
            split.delete_submodule(str(n.target))
            n.target = "compiled_" + str(n.target)
            split.add_submodule(str(n.target), sc._WrapperModule(exe, single))
    split.recompile()
    return sc._SplitCompiledExecutable(split)


def thin_graphs():
    """(name, fn, input shapes, partitions, runner returns a list) — served-shaped small graphs."""

    def g_two_out(a, b, c):  # arguments used out of order
        return c * b + a, a - c

    def g_single(x, w):
        return (x @ w).relu()

    def g_passthrough(x, w):  # an input returned as is, next to a computed value
        return x, x + w

    def g_unused(a, b, c, d):  # b unused: the partition takes a subset
        return a * d, c + d, d - a

    def g_two_part(a, b):  # two partitions: stays on torch_tpu's path
        y = a + b
        z = y * 2
        return z - a, y

    return [
        ("two_out", g_two_out, [(8, 16), (8, 16), (8, 16)], 1, False),
        ("single", g_single, [(16, 32), (32, 8)], 1, True),
        ("passthrough", g_passthrough, [(4, 8), (4, 8)], 1, False),
        ("unused", g_unused, [(6,), (6,), (6,), (6,)], 1, True),
        ("two_part", g_two_part, [(5, 5), (5, 5)], 2, False),
    ]


def mkthin_stream(seed: int, calls: int = 300, l6_steps: bool = True):
    """Random calls on the thin_graphs executables. Returns (trace, runner-arg identity pattern)."""
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    log = RunnerLog()
    exes = [(name, build_split_exe(fn, log, name, parts, as_list), shapes)
            for name, fn, shapes, parts, as_list in thin_graphs()]
    trace, pattern = [], []
    for i in range(calls):
        if l6_steps and i % 3 == 0:
            _l6_begin()
        name, exe, shapes = exes[rng.randrange(len(exes))]
        args = [torch.randn(s, generator=gen) for s in shapes]
        boxed = rng.random() < 0.7
        n0 = len(log.calls)
        out = exe(list(args)) if boxed else exe(*args)
        got = log.calls[n0:]
        ids = {id(a): k for k, a in enumerate(args)}
        pattern.append((name, [(c[0], [ids.get(id(x), "?") for x in c[1]], type(c[1]).__name__) for c in got]))
        # Outputs: structure types and values; an input returned as is must be that same object.
        if isinstance(out, tuple):
            o = [("in", ids[id(t)]) if id(t) in ids else ("t", t.tolist()) for t in out]
            trace.append((i, name, "tuple", o))
        else:
            trace.append((i, name, type(out).__name__, ("in", ids[id(out)]) if id(out) in ids else out.tolist()))
    return trace, pattern


# ---------------------------------------------------------------------------------------- fuserpc
class FakeScheduler:
    """Scripted scheduler: one SchedulerOutput-shaped object per step, recorded outputs."""

    def __init__(self, plan: list[dict], n_req: int = 4):
        self.plan = list(plan)
        self.i = 0
        self.n_req = n_req
        self.updates: list = []
        self.grammar_calls: list = []

    def has_requests(self) -> bool:
        return self.i < len(self.plan)

    def schedule(self, throttle=False):
        p = self.plan[self.i]
        so = NS(
            step=self.i,
            total_num_scheduled_tokens=p.get("tokens", 4 * self.n_req),
            num_scheduled_tokens={f"r{k}": 4 for k in range(self.n_req)},
            pending_structured_output_tokens=p.get("pending", False),
            has_structured_output_requests=p.get("structured", False),
        )
        self.i += 1
        return so

    def get_grammar_bitmask(self, so):
        self.grammar_calls.append(so.step)
        if not so.has_structured_output_requests:
            return None
        return NS(grammar_for=so.step)

    def update_from_output(self, so, model_output):
        mo = model_output
        kvo = getattr(mo, "kv_connector_output", None)
        self.updates.append((so.step, list(mo.req_ids), [list(t) for t in mo.sampled_token_ids],
                             None if kvo is None else (sorted(kvo.finished_sending or ()),
                                                       sorted(kvo.finished_recving or ()))))
        return {0: ("out", so.step)}

    def update_draft_token_ids_in_output(self, *a):
        raise AssertionError("check_for_draft_tokens is False in the rig")


class FakeCore:
    """The attributes vLLM 0.31.0's step_with_batch_queue reads, on a fake engine core."""

    def __init__(self, executor, scheduler, pooling=False):
        self.model_executor = executor
        self.scheduler = scheduler
        self.batch_queue_size = 2
        self.batch_queue = deque(maxlen=2)
        self.is_mm_encoder_only = False
        self.is_pooling_model = pooling
        self.check_for_draft_tokens = False

    def _should_throttle_prefills(self):
        return False

    @contextlib.contextmanager
    def log_error_detail(self, so):
        yield

    @contextlib.contextmanager
    def capture_iteration_details(self, so):
        yield None

    def _process_aborts_queue(self):
        pass

    def _attach_iteration_details(self, outputs, details):
        pass


class FakeWorker:
    """The worker behind WorkerWrapperBase in a rig process: execute_model / sample_tokens like the
    TPU runner's contract (state between them), logging every call to a per-rank file."""

    def __init__(self, rank, logpath, fail_exec_at=None, async_out=False, n_req=4, tpu_async=False, delay_ms=0.0):
        self.rank, self.logpath, self.fail_exec_at, self.async_out, self.n_req = (
            rank, logpath, fail_exec_at, async_out, n_req)
        self.tpu_async, self.delay_ms = tpu_async, delay_ms
        self.pending = None

    def _log(self, *what):
        with open(self.logpath, "a") as f:
            f.write(repr(what) + "\n")

    def l6_rig_exit(self):
        raise SystemExit(0)

    def l6_rig_ping(self):
        return self.rank

    def execute_model(self, so):
        self._log("exec", so.step, so.total_num_scheduled_tokens)
        if self.fail_exec_at is not None and so.step == self.fail_exec_at:
            raise ValueError(f"injected execute_model failure at step {so.step}")
        if so.total_num_scheduled_tokens == 0:  # as the TPU runner (tpu_runner.py:4857-4864)
            import copy

            from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

            return copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        self.pending = so.step
        self.n_req = len(so.num_scheduled_tokens)  # the output carries one row per scheduled request
        return None

    def sample_tokens(self, grammar_output):
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput, KVConnectorOutput
        import copy

        step = self.pending
        self._log("sample", step, None if grammar_output is None else grammar_output.grammar_for)
        if step is None:
            return None
        self.pending = None
        out = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        out.req_ids = [f"r{k}" for k in range(self.n_req)]
        out.req_id_to_index = {r: k for k, r in enumerate(out.req_ids)}
        out.sampled_token_ids = [[(step * 7 + k + self.rank * 0) % 50] for k in range(self.n_req)]
        # Rank r reports step s's transfers at step s + r: the KV aggregator finishes a request only at the
        # last rank's report, so an aggregation that drops or skips ranks is visible in the scheduler updates.
        s0 = step - self.rank
        out.kv_connector_output = KVConnectorOutput(
            finished_sending={f"s{s0}"} if s0 >= 0 and s0 % 3 == 0 else None,
            finished_recving={f"v{s0}"} if s0 >= 0 and s0 % 4 == 1 else None,
        )
        if self.tpu_async:  # the TPU runner's async output; its copy wait is logged and staggered by rank
            from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput

            toks = [t for t in out.sampled_token_ids]
            out.sampled_token_ids = []
            return AsyncTPUModelRunnerOutput(model_runner_output=out, copy_state=_FakeCopy(self, step, toks),
                                             discard_sampled_tokens_req_indices=[])
        if self.async_out:
            class _A(AsyncModelRunnerOutput):
                def __init__(self, o):
                    self.o = o

                def get_output(self):
                    return self.o

            return _A(out)
        return out


class _FakeCopy:
    """Duck-typed AsyncTPUCopyState: wait() logs and sleeps a rank-staggered copy-ready time."""

    spec_logprobs = None
    # q38 L6-V6 compose: AsyncTPUCopyState carries L6-H2 TOKEN-LISTS-ONCE's issuing-step state (default False, the
    # stock tensor path), which get_output reads.
    l6_lists = False

    def __init__(self, worker, step, toks):
        self.worker, self.step, self.toks = worker, step, toks

    def wait(self):
        import time

        self.worker._log("wait", self.step)
        if self.worker.delay_ms:
            time.sleep(self.worker.delay_ms * (1 + self.worker.rank) / 1000.0)

    @property
    def sampled_token_ids_cpu(self):
        return torch.tensor(self.toks, dtype=torch.int64)


# Every rig's MessageQueues (and, for thread workers, their WorkerProc) stay referenced for the life of
# the process: a collected MessageQueue's zmq Context runs Context.__del__ -> term(), which blocks while any
# socket of it is still open, and it did block a rig worker's acquire_read (L6-MK p1f job j-38708ad7,
# faulthandler dump). vLLM's own executor keeps its queues for the process lifetime too.
_KEEPALIVE: list = []


def _worker_main(rank, handle, conn, logpath, fail_exec_at, async_out, n_req, install_fuse, tpu_async=False,
                 delay_ms=0.0, mode="fork"):
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    import queue
    import threading

    import faulthandler

    if mode != "thread":  # process-wide timer: only in a worker process
        stack = open(logpath + ".stack", "w")
        faulthandler.dump_traceback_later(60, repeat=True, file=stack)  # a hung rig worker shows its stacks
    try:
        if install_fuse:
            from vllm_torchtpu.core import l6_fuse_rpc

            l6_fuse_rpc.install_worker()
        wp = WorkerProc.__new__(WorkerProc)
        _KEEPALIVE.append(wp)
        wp.rank = rank
        wrapper = WorkerWrapperBase.__new__(WorkerWrapperBase)
        wrapper.__dict__["worker"] = FakeWorker(rank, logpath, fail_exec_at, async_out, n_req, tpu_async, delay_ms)
        wrapper.__dict__["mm_receiver_cache"] = None
        wrapper.__dict__["rpc_rank"] = rank
        wrapper.__dict__["global_rank"] = rank
        wp.worker = wrapper
        wp.rpc_broadcast_mq = MessageQueue.create_from_handle(handle, rank)
        wp.worker_response_mq = MessageQueue(1, 1)
        wp.use_async_scheduling = True
        wp.async_output_queue = queue.Queue()
        threading.Thread(target=wp.async_output_busy_loop, daemon=True).start()
        conn.send(wp.worker_response_mq.export_handle())
        wp.rpc_broadcast_mq.wait_until_ready()
        wp.worker_response_mq.wait_until_ready()
        wp.worker_busy_loop()
    except BaseException as e:  # noqa: BLE001
        with open(logpath + ".exit", "a") as f:
            f.write(repr(("worker_exit", type(e).__name__, str(e))) + "\n")
        if mode != "thread":
            os._exit(3)


def rig_mode() -> str:
    """Q38_L6_RIG_MODE: thread (default: rig workers are threads of this process; the same MessageQueue
    transport, pickling, WorkerProc loop and async output thread), spawn or fork (worker processes)."""
    m = os.environ.get("Q38_L6_RIG_MODE", "thread")
    assert m in ("thread", "spawn", "fork"), m
    return m


class RpcRig:
    """A real MultiprocExecutor RPC path to `world` rig workers (threads, or spawned / forked processes)."""

    def __init__(self, tmpdir, world=2, kv_agg=True, fail_exec_at=None, async_out=True, n_req=4,
                 install_fuse=True, tpu_async=False, delay_ms=0.0, mode=None):
        import multiprocessing as mp

        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
        from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        self.world = world
        self.logs = [os.path.join(tmpdir, f"w{r}.log") for r in range(world)]
        for p in self.logs:
            open(p, "w").close()
        mq = MessageQueue(world, world)
        _KEEPALIVE.append(mq)
        handle = mq.export_handle()
        self.mode = mode or rig_mode()
        self.procs, conns = [], []
        import threading

        for r in range(world):
            args = (r, handle, None, self.logs[r], fail_exec_at, async_out, n_req, install_fuse, tpu_async, delay_ms,
                    self.mode)
            if self.mode == "thread":
                a, b = mp.Pipe()
                p = threading.Thread(target=_worker_main, args=args[:2] + (b,) + args[3:], daemon=True)
            else:
                ctx = mp.get_context(self.mode)
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker_main, args=args[:2] + (b,) + args[3:], daemon=True)
            p.start()
            self.procs.append(p)
            conns.append(a)
        handles = []
        for c in conns:
            if not c.poll(120):
                raise RuntimeError("rig worker did not report its response queue")
            handles.append(c.recv())
        ex = MultiprocExecutor.__new__(MultiprocExecutor)
        ex.rpc_broadcast_mq = mq
        ex.response_mqs = [MessageQueue.create_from_handle(h, 0) for h in handles]
        _KEEPALIVE.append(ex.response_mqs)
        import threading

        def ready():
            mq.wait_until_ready()
            for rmq in ex.response_mqs:
                rmq.wait_until_ready()

        t = threading.Thread(target=ready, daemon=True)
        t.start()
        t.join(120)
        if t.is_alive():
            self.close()
            raise RuntimeError(f"rig message queues not ready after 120 s; worker logs {self.worker_logs()}")
        ex.futures_queue = deque()
        ex.is_failed = False
        ex.output_rank = 0
        ex.kv_output_aggregator = KVOutputAggregator(world) if kv_agg else None
        ex.ec_output_aggregator = None
        # Every rank answers one RPC before the stream (a rig that cannot is reported, not run).
        try:
            got = ex.collective_rpc("l6_rig_ping", timeout=120)
        except Exception as e:  # noqa: BLE001
            self.executor = ex
            diag = self.diagnostics()
            self.close()
            raise RuntimeError(f"rig ping failed ({type(e).__name__}: {e}); {diag}") from e
        if list(got) != list(range(world)):
            raise RuntimeError(f"rig ping answered {got}")
        self.enqueued: list = []
        orig_enqueue = mq.enqueue

        def enqueue(obj, timeout=None):
            self.enqueued.append(obj[0] if isinstance(obj[0], str) else "<fn>")
            return orig_enqueue(obj, timeout=timeout) if timeout is not None else orig_enqueue(obj)

        mq.enqueue = enqueue
        self.executor = ex

    def sync(self):
        """Wait until every rank has run every RPC sent so far (ranks that do not reply to the stream's
        RPCs can lag behind the engine): one RPC every rank answers, behind them in each rank's queue."""
        got = self.executor.collective_rpc("l6_rig_ping", timeout=120)
        assert list(got) == list(range(self.world)), got

    def worker_logs(self):
        out = []
        for p in self.logs:
            with open(p) as f:
                out.append([ln.strip() for ln in f if ln.strip()])
        return out

    def stacks(self) -> str:
        return self.diagnostics()

    def diagnostics(self) -> str:
        """Per rank: the last worker-log lines, the exit record and (worker processes) the stack dumps."""
        out = []
        for r, p in enumerate(self.logs):
            parts = []
            for suffix, n in (("", 6), (".exit", 4), (".stack", 60)):
                try:
                    with open(p + suffix) as f:
                        lines = f.read().splitlines()
                except OSError:
                    continue
                if lines:
                    parts.append(f"[{suffix or 'log'}] " + " | ".join(lines[-n:]))
            alive = self.procs[r].is_alive() if r < len(self.procs) else None
            out.append(f"rank {r} alive={alive}: " + " ".join(parts))
        return "\n".join(out)

    def close(self):
        if self.mode == "thread":  # end the worker threads' busy loops (SystemExit from the fake worker)
            try:
                self.executor.collective_rpc("l6_rig_exit", non_block=True, unique_reply_rank=0)
            except Exception:  # noqa: BLE001
                pass
            for p in self.procs:
                p.join(10)
            return
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(10)


def rpc_plan(seed: int, steps: int, structured: float = 0.1, pending: float = 0.05, empty: float = 0.0):
    rng = random.Random(seed)
    plan = []
    for _ in range(steps):
        r = rng.random()
        p: dict = {}
        if r < pending:
            p.update(structured=True, pending=True)
        elif r < pending + structured:
            p.update(structured=True)
        if empty and rng.random() < empty:
            p["tokens"] = 0
        plan.append(p)
    return plan


def run_rpc_stream(rig: RpcRig, step_fn, plan, n_req=4, l6_steps=True, pooling=False):
    """Drive step_fn (an EngineCore.step_with_batch_queue implementation) until the scripted
    scheduler is empty and the batch queue drained. Returns the engine-side trace."""
    sched = FakeScheduler(plan, n_req)
    if l6_steps:
        from vllm_torchtpu.core import l6_ab

        orig = sched.schedule

        def schedule(*a, **k):
            l6_ab.begin_step("engine_core")
            return orig(*a, **k)

        sched.schedule = schedule
    core = FakeCore(rig.executor, sched, pooling=pooling)
    results = []
    guard = 0
    while sched.has_requests() or core.batch_queue:
        guard += 1
        assert guard < 10 * len(plan) + 50, "stream did not drain"
        outs, executed = step_fn(core)
        results.append((outs, executed))
    return {"results": results, "updates": sched.updates, "grammar": sched.grammar_calls,
            "kvonly_off": core.__dict__.get("_l6_kvonly_off", False)}


if __name__ == "__main__":
    which, seed = sys.argv[1], int(sys.argv[2])
    if which == "mkctx":
        t, r, s = mkctx_stream(seed)
        print(f"L6MK_TRACE mkctx seed={seed} digest={digest((t, r, s))} steps={len(t)} rec={len(r)} "
              f"whole_calls={s[2]} stats={len(s[0])}")
    elif which == "mkthin":
        try:  # head: install the lever's class patch when its flag is set (worker init does this)
            from vllm_torchtpu.runner import l6_mk

            l6_mk.install_worker_patches()
        except ImportError:  # base: no lever
            pass
        t, p = mkthin_stream(seed)
        print(f"L6MK_TRACE mkthin seed={seed} digest={digest(t)} pattern={digest(p)} calls={len(t)}")
    else:
        raise SystemExit(f"unknown stream {which}")
