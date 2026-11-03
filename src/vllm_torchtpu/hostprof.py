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
"""Per-request host-side stage timing logs, opt-in with ``VLLM_TPU_HOSTPROF=1``.

Diagnostics only. Wraps API-server, engine-core and rank-0 runner methods with
wall-clock timers and emits one ``HOSTPROF <kind> k=v ...`` line per request
stage or engine step. Timestamps are epoch seconds; durations are ms.

Lines:
  api        API server, one per request: body read, handler entry, render
             (chat template + tokenize), AsyncLLM.add_request, first output, end.
  ec_add     engine core input thread: EngineCoreRequest -> Request (block hashes).
  ec_sched   engine core, a request's first schedule: queue wait and its
             prefix-cache / connector / allocation time so far.
  ec_first   engine core, a request's first output token, with totals.
  ec_step    engine core, per schedule() and update_from_output() call.
  wk         rank-0 runner, per execute_model / sample_tokens.

Patches install when the target module is imported, so each process only pays
for what it loads.
"""

import contextvars
import importlib.abc
import logging
import os
import sys
import time
from collections import defaultdict
from functools import wraps

logger = logging.getLogger("vllm." + __name__)

_CUR: contextvars.ContextVar = contextvars.ContextVar("hostprof_cur", default=None)
_API: dict = {}  # external request id -> api stage dict
_REQ: dict = defaultdict(lambda: defaultdict(float))  # engine request id -> stats
_STEP: dict = defaultdict(float)  # sub-call ms since the last ec_step line


def _now() -> float:
    return time.time()


def _fmt(d: dict) -> str:
    out = []
    for k, v in d.items():
        if isinstance(v, float):
            v = f"{v:.6f}" if k == "t0" or k.startswith("t_") else f"{v:.3f}"
        out.append(f"{k}={v}")
    return " ".join(out)


def _emit(kind: str, d: dict) -> None:
    logger.info("HOSTPROF %s %s", kind, _fmt(d))


def _ntok(prompt) -> int:
    try:
        if isinstance(prompt, dict):
            ids = prompt.get("prompt_token_ids")
            if ids is None and isinstance(prompt.get("decoder"), dict):
                ids = prompt["decoder"].get("prompt_token_ids")
            return len(ids) if ids is not None else -1
        ids = getattr(prompt, "prompt_token_ids", None)
        return len(ids) if ids is not None else -1
    except Exception:
        return -1


def _wrap(cls, name, make):
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_hostprof", False):
        return
    new = make(orig)
    new._hostprof = True
    setattr(cls, name, new)


def _acc(cls, name, key, rid_of=None):
    """Accumulate a method's ms into the step and, if rid_of, per request."""

    def make(orig):
        @wraps(orig)
        def f(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return orig(self, *a, **k)
            finally:
                ms = (time.perf_counter() - t0) * 1e3
                _STEP[key] += ms
                if rid_of is not None:
                    try:
                        r = _REQ[rid_of(a, k).request_id]
                        r[key] += ms
                        r["n_" + key] += 1
                    except Exception:
                        pass

        return f

    _wrap(cls, name, make)


def _first_arg(a, k):
    return a[0] if a else k.get("request")


# --- API server -------------------------------------------------------------


def _patch_starlette(mod) -> None:
    def make(orig):
        @wraps(orig)
        async def body(self):
            s = self.scope
            if "hp_b0" not in s:
                s["hp_b0"] = _now()
            try:
                return await orig(self)
            finally:
                s.setdefault("hp_b1", _now())

        return body

    _wrap(mod.Request, "body", make)


def _patch_serving_chat(mod) -> None:
    cls = mod.OpenAIServingChat

    def make_create(orig):
        @wraps(orig)
        async def f(self, request, raw_request=None, *a, **k):
            d = {"t_entry": _now()}
            if raw_request is not None:
                d["t_b0"] = raw_request.scope.get("hp_b0", -1.0)
                d["t_b1"] = raw_request.scope.get("hp_b1", -1.0)
            tok = _CUR.set(d)
            try:
                return await orig(self, request, raw_request, *a, **k)
            finally:
                _CUR.reset(tok)

        return f

    def make_render(orig):
        @wraps(orig)
        async def f(self, *a, **k):
            d = _CUR.get()
            t0 = _now()
            try:
                return await orig(self, *a, **k)
            finally:
                if d is not None:
                    d["t_render0"] = t0
                    d["render_ms"] = (_now() - t0) * 1e3

        return f

    _wrap(cls, "_create_chat_completion", make_create)
    _wrap(cls, "render_chat_request", make_render)


def _patch_async_llm(mod) -> None:
    cls = mod.AsyncLLM

    def make_generate(orig):
        @wraps(orig)
        def generate(self, prompt, sampling_params, request_id, *a, **k):
            d = dict(_CUR.get() or {})
            d["t_gen"] = _now()
            d["ntok"] = _ntok(prompt)
            _API[request_id] = d
            gen = orig(self, prompt, sampling_params, request_id, *a, **k)

            async def wrapped():
                d["t_iter"] = _now()
                n = 0
                try:
                    async for out in gen:
                        if n == 0:
                            d["t_first"] = _now()
                        n += 1
                        yield out
                finally:
                    d["t_end"] = _now()
                    d["n_out"] = n
                    _API.pop(request_id, None)
                    _emit("api", {"rid": request_id, **d})

            return wrapped()

        return generate

    def make_add(orig):
        @wraps(orig)
        async def add_request(self, request_id, *a, **k):
            t0 = _now()
            try:
                return await orig(self, request_id, *a, **k)
            finally:
                d = _API.get(request_id)
                if d is not None:
                    d["t_add0"] = t0
                    d["add_ms"] = (_now() - t0) * 1e3

        return add_request

    _wrap(cls, "generate", make_generate)
    _wrap(cls, "add_request", make_add)


# --- Engine core ------------------------------------------------------------


def _patch_engine_core(mod) -> None:
    def make(orig):
        @wraps(orig)
        def f(self, request, *a, **k):
            t0 = _now()
            p0 = time.perf_counter()
            res = orig(self, request, *a, **k)
            ms = (time.perf_counter() - p0) * 1e3
            try:
                req = res[0]
                r = _REQ[req.request_id]
                r["t_add"] = t0
                r["add_ms"] = ms
                r["ntok"] = req.num_prompt_tokens
                r["nblk_hash"] = len(req.block_hashes)
                _emit(
                    "ec_add",
                    {
                        "rid": req.request_id,
                        "t_add": t0,
                        "add_ms": ms,
                        "ntok": req.num_prompt_tokens,
                        "nblk_hash": len(req.block_hashes),
                    },
                )
            except Exception:
                pass
            return res

        return f

    _wrap(mod.EngineCore, "preprocess_add_request", make)


def _patch_scheduler(mod) -> None:
    cls = mod.Scheduler

    def make_schedule(orig):
        @wraps(orig)
        def schedule(self, *a, **k):
            t0 = _now()
            p0 = time.perf_counter()
            out = orig(self, *a, **k)
            ms = (time.perf_counter() - p0) * 1e3
            try:
                new = out.scheduled_new_reqs
                step = {
                    "phase": "sched",
                    "t0": t0,
                    "ms": ms,
                    "n_new": len(new),
                    "n_run": len(out.num_scheduled_tokens) - len(new),
                    "tok": out.total_num_scheduled_tokens,
                    "n_wait": len(self.waiting),
                    **dict(_STEP),
                }
                _STEP.clear()
                _emit("ec_step", step)
                for nr in new:
                    r = _REQ.get(nr.req_id)
                    if r is None:
                        continue
                    r["t_sched"] = t0
                    _emit(
                        "ec_sched",
                        {
                            "rid": nr.req_id,
                            "t_sched": t0,
                            "wait_ms": (t0 - r.get("t_add", t0)) * 1e3,
                            "cached": nr.num_computed_tokens,
                            "sched_tok": out.num_scheduled_tokens.get(nr.req_id, 0),
                            **{k2: v for k2, v in r.items() if k2 != "t_add"},
                        },
                    )
            except Exception:
                logger.debug("hostprof schedule log failed", exc_info=True)
            return out

        return schedule

    def make_update(orig):
        @wraps(orig)
        def update_from_output(self, scheduler_output, *a, **k):
            pre = []
            try:
                for rid in scheduler_output.num_scheduled_tokens:
                    req = self.requests.get(rid)
                    if req is not None and req.num_output_tokens == 0:
                        pre.append(req)
            except Exception:
                pass
            t0 = _now()
            p0 = time.perf_counter()
            res = orig(self, scheduler_output, *a, **k)
            ms = (time.perf_counter() - p0) * 1e3
            try:
                step = {"phase": "update", "t0": t0, "ms": ms, **dict(_STEP)}
                _STEP.clear()
                _emit("ec_step", step)
                t1 = _now()
                for req in pre:
                    if req.num_output_tokens > 0:
                        r = _REQ.pop(req.request_id, {})
                        _emit(
                            "ec_first",
                            {
                                "rid": req.request_id,
                                "t_first": t1,
                                "since_add_ms": (t1 - r.get("t_add", t1)) * 1e3,
                                **{k2: v for k2, v in r.items()},
                            },
                        )
            except Exception:
                logger.debug("hostprof update log failed", exc_info=True)
            return res

        return update_from_output

    _wrap(cls, "schedule", make_schedule)
    _wrap(cls, "update_from_output", make_update)


def _patch_kv_cache_manager(mod) -> None:
    cls = mod.KVCacheManager
    _acc(cls, "get_computed_blocks", "gcb_ms", _first_arg)
    _acc(cls, "allocate_slots", "alloc_ms", _first_arg)


def _patch_prefix_aware(mod) -> None:
    _acc(mod.PrefixAwareScheduler, "_promote_warmest", "promote_ms")


def _patch_raiden(mod) -> None:
    cls = mod.TPURaidenOffloadingConnector
    _acc(cls, "get_num_new_matched_tokens", "conn_match_ms", _first_arg)
    _acc(cls, "update_state_after_alloc", "conn_alloc_ms", _first_arg)
    _acc(cls, "build_connector_meta", "conn_meta_ms")
    _acc(cls, "update_connector_output", "conn_out_ms")
    _acc(cls, "request_finished", "conn_fin_ms", _first_arg)


def _patch_executor(mod) -> None:
    _acc(mod.MultiprocExecutor, "execute_model", "dispatch_ms")
    _acc(mod.MultiprocExecutor, "sample_tokens", "dispatch_sample_ms")


# --- Rank-0 runner ----------------------------------------------------------


def _is_rank0(self) -> bool:
    v = getattr(self, "_hp_rank0", None)
    if v is None:
        try:
            from vllm.distributed import parallel_state

            v = parallel_state.get_world_group().rank_in_group == 0
        except Exception:
            v = True
        self._hp_rank0 = v
    return v


def _patch_runner(mod) -> None:
    cls = mod.TPUModelRunner

    def sub(name, key):
        def make(orig):
            @wraps(orig)
            def f(self, *a, **k):
                if not _is_rank0(self):
                    return orig(self, *a, **k)
                p0 = time.perf_counter()
                try:
                    return orig(self, *a, **k)
                finally:
                    w = getattr(self, "_hp_w", None)
                    if w is not None:
                        w[key] = w.get(key, 0.0) + (time.perf_counter() - p0) * 1e3

            return f

        _wrap(cls, name, make)

    def top(name, phase):
        def make(orig):
            @wraps(orig)
            def f(self, *a, **k):
                if not _is_rank0(self):
                    return orig(self, *a, **k)
                self._hp_w = w = {}
                t0 = _now()
                p0 = time.perf_counter()
                try:
                    return orig(self, *a, **k)
                finally:
                    d = {"phase": phase, "t0": t0}
                    d["ms"] = (time.perf_counter() - p0) * 1e3
                    d.update(w)
                    so = a[0] if a and phase == "exec" else None
                    if so is not None:
                        try:
                            d["tok"] = so.total_num_scheduled_tokens
                            d["n_new"] = len(so.scheduled_new_reqs)
                            d["max_new_len"] = max(
                                (len(r.prompt_token_ids or ()) for r in so.scheduled_new_reqs),
                                default=0,
                            )
                        except Exception:
                            pass
                    self._hp_w = None
                    _emit("wk", d)

            return f

        _wrap(cls, name, make)

    sub("_update_states", "upd_ms")
    sub("_prepare_inputs", "prep_ms")
    top("execute_model", "exec")
    top("sample_tokens", "sample")


# --- Installation -----------------------------------------------------------

_TARGETS = {
    "starlette.requests": _patch_starlette,
    "vllm.entrypoints.openai.chat_completion.serving": _patch_serving_chat,
    "vllm.v1.engine.async_llm": _patch_async_llm,
    "vllm.v1.engine.core": _patch_engine_core,
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
    "vllm.v1.core.kv_cache_manager": _patch_kv_cache_manager,
    "vllm_torchtpu.core.prefix_aware_scheduler": _patch_prefix_aware,
    "vllm_torchtpu.offload.raiden_connector": _patch_raiden,
    "vllm.v1.executor.multiproc_executor": _patch_executor,
    "vllm_torchtpu.runner.tpu_runner": _patch_runner,
}
_pending: set = set()


def _run(name, module) -> None:
    _pending.discard(name)
    try:
        _TARGETS[name](module)
        logger.info("HOSTPROF patched %s", name)
    except Exception:
        logger.warning("HOSTPROF failed to patch %s", name, exc_info=True)


class _PostImportHook(importlib.abc.MetaPathFinder):
    """Runs a target's patch right after its module executes."""

    def find_spec(self, name, path, target=None):
        if name not in _pending:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _run(name, module)

        loader.exec_module = exec_module
        return spec


_installed = False


def install_hostprof() -> None:
    global _installed
    if _installed or os.environ.get("VLLM_TPU_HOSTPROF", "0") in ("", "0"):
        return
    _installed = True
    for name in _TARGETS:
        mod = sys.modules.get(name)
        if mod is not None:
            _run(name, mod)
        else:
            _pending.add(name)
    if _pending:
        sys.meta_path.insert(0, _PostImportHook())
    logger.info("HOSTPROF installed (pid %d)", os.getpid())
