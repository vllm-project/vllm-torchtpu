# SPDX-License-Identifier: Apache-2.0
"""Perfetto slices for engine startup, as cdk's trace generator reads them.

With ``VLLM_TPU_STARTUP_TRACE=1`` the startup path of every worker prints
``trace-start-event`` / ``trace-end-event`` lines (the format of
cloud-devkit-recipes' ``cdk.trace``), which cdk turns into Perfetto slices
after the job: one process per server (``pid=vllm-<role>``, the role from
``VLLM_TPU_STARTUP_TRACE_ROLE``, else ``TPU_RAIDEN_JOB_NAME``, else
``server``) and one thread per worker rank (``tid=rank<N>``). Nested calls
nest. Covered:

  * TPUWorker: init_device, load_model, get_kv_cache_spec,
    determine_available_memory, initialize_from_config,
    compile_or_warm_up_model;
  * TPUModelRunner: load_model, profile_run, initialize_kv_cache,
    capture_model and every ``_precompile_*`` / ``_warmup_*`` method;
  * the model loader: load_model (construction + weights) and load_weights;
  * TpuCompilerAdaptor: ``xla-compile-<range>`` for each compile and
    ``aot-load-<range>`` for each artifact rebuilt from the compile cache;
  * the Qwen4Exp PD connector worker: register_runner and the loaded-state
    identity.

Off (the default) nothing is wrapped. The lines go to stdout, unbuffered.
"""

import functools
import os
import threading
import time

ENV = "VLLM_TPU_STARTUP_TRACE"

_installed: set[str] = set()
# Cleared when worker.compile_or_warm_up_model returns (see _wrap).
_active = [True]


def enabled() -> bool:
    return os.environ.get(ENV, "0").lower() in ("1", "true", "yes", "on")


def _pid() -> str:
    role = (
        os.environ.get("VLLM_TPU_STARTUP_TRACE_ROLE")
        or os.environ.get("TPU_RAIDEN_JOB_NAME")
        or "server"
    )
    return "vllm-" + "".join(c if c.isalnum() or c in "-_." else "_" for c in role)


def _tid() -> str:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank is not None:
        return f"rank{rank}"
    return f"proc{os.getpid()}-{threading.get_ident()}"


def _clean(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.:()," else "_" for c in name)


def _emit(kind: str, name: str) -> None:
    print(
        f"trace-{kind}-event: name={_clean(name)} ts={time.time():.6f} "
        f"pid={_pid()} tid={_tid()}",
        flush=True,
    )


class span:
    """``with span("name"):`` one slice around the block (closed on error too)."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if enabled():
            _emit("start", self.name)
        return self

    def __exit__(self, *exc):
        if enabled():
            _emit("end", self.name)
        return False


def _wrap(cls, attr: str, name=None) -> None:
    """Wrap ``cls.attr`` (defined on cls itself) in a slice; name may be a
    callable of the call's (self, args, kwargs) for per-call names."""
    fn = cls.__dict__.get(attr)
    if fn is None or not callable(fn) or getattr(fn, "_startup_traced", False):
        return
    label = name or f"{cls.__name__}.{attr}"

    @functools.wraps(fn)
    def traced(self, *args, **kwargs):
        # Startup only: once compile_or_warm_up_model has returned, the wrapped
        # methods that also run per step (compiles, sampler calls) stay silent.
        if not _active[0]:
            return fn(self, *args, **kwargs)
        try:
            n = label(self, args, kwargs) if callable(label) else label
        except Exception:  # noqa: BLE001 - a name must never break startup
            n = f"{cls.__name__}.{attr}"
        _emit("start", n)
        stacks = n in _stack_spans() and _stack_dumps_start()
        try:
            return fn(self, *args, **kwargs)
        finally:
            if stacks:
                _stack_dumps_stop()
            _emit("end", n)
            if n == "worker.compile_or_warm_up_model":
                _active[0] = False

    traced._startup_traced = True  # type: ignore[attr-defined]
    setattr(cls, attr, traced)


def _stack_spans() -> set[str]:
    """Spans that get stack dumps (VLLM_TPU_STARTUP_STACKS_SPANS, comma-separated;
    default the weight load).

    faulthandler reads other threads' state without locks: sampling every 20 s
    through the compile phase killed a prefill worker mid-lowering
    (j-969ef035-4526-43b5-8166, VllmWorker-1 died with no exit code), so keep it to
    short diagnostic runs and prefer the weight load, where it ran safely."""
    return set(
        os.environ.get("VLLM_TPU_STARTUP_STACKS_SPANS", "loader.load_weights").split(",")
    )


def _stack_dumps_start() -> bool:
    """VLLM_TPU_STARTUP_STACKS=<seconds>: on local ranks 0 and 1, dump every
    thread's Python stack to stderr every <seconds> while weights load, to see
    what a straggling rank is doing that its peers are not."""
    try:
        interval = float(os.environ.get("VLLM_TPU_STARTUP_STACKS", "0") or 0)
    except ValueError:
        return False
    if interval <= 0 or os.environ.get("LOCAL_RANK", "0") not in ("0", "1"):
        return False
    import faulthandler
    import sys

    print(
        f"startup-stacks: rank {os.environ.get('LOCAL_RANK')} dumping every {interval}s",
        file=sys.stderr,
        flush=True,
    )
    faulthandler.dump_traceback_later(interval, repeat=True, file=sys.stderr)
    return True


def _stack_dumps_stop() -> None:
    import faulthandler

    faulthandler.cancel_dump_traceback_later()


def _range_name(prefix: str, compile_range) -> str:
    try:
        start, end = compile_range.start, compile_range.end
        return f"{prefix}-{start}" if start == end else f"{prefix}-{start}-{end}"
    except Exception:  # noqa: BLE001
        return prefix


def install_worker(cls) -> None:
    if not enabled() or "worker" in _installed:
        return
    _installed.add("worker")
    for attr in (
        "init_device",
        "load_model",
        "get_kv_cache_spec",
        "determine_available_memory",
        "initialize_from_config",
        "compile_or_warm_up_model",
    ):
        _wrap(cls, attr, f"worker.{attr}")


def install_runner(cls) -> None:
    if not enabled() or "runner" in _installed:
        return
    _installed.add("runner")
    for attr in ("load_model", "profile_run", "initialize_kv_cache", "capture_model"):
        _wrap(cls, attr, f"runner.{attr}")
    for attr in list(vars(cls)):
        if (
            attr.startswith(("_precompile_", "_warmup_"))
            and attr != "_precompile_timed"
            and callable(vars(cls)[attr])
        ):
            _wrap(cls, attr, attr.lstrip("_"))
    # The compiled spec-verify preludes the rejection-sampler precompile warms
    # per (padded_logits_length, num_reqs) bucket, named by their logits shape.
    for attr in (
        "spec_bonus_and_target_logits",
        "spec_bonus_and_target_tokens",
        "spec_gather_bonus_and_target_logits",
    ):
        _wrap(cls, attr, lambda s, a, k, _n=attr: _shape_name(_n, a))
    try:
        from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
            RejectionSampler,
        )

        _wrap(
            RejectionSampler,
            "__call__",
            lambda s, a, k: "rejection_sampler"
            + ("-sampling" if k.get("do_sampling") else "-greedy")
            + ("-topk" if k.get("topk_candidates") is not None else "")
            + _shape_suffix(k.get("draft_token_ids")),
        )
    except Exception:  # noqa: BLE001
        pass
    install_loaders()
    install_compiler()


def _shape_suffix(t) -> str:
    try:
        return "-" + "x".join(str(int(d)) for d in t.shape)
    except Exception:  # noqa: BLE001
        return ""


def _shape_name(name: str, args) -> str:
    return name + (_shape_suffix(args[0]) if args else "")


def install_loaders() -> None:
    if not enabled() or "loaders" in _installed:
        return
    _installed.add("loaders")
    try:
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader

        _wrap(BaseModelLoader, "load_model", "loader.load_model")
    except Exception:  # noqa: BLE001
        pass
    for mod, name in (
        ("vllm.model_executor.model_loader.runai_streamer_loader", "RunaiModelStreamerLoader"),
        ("vllm.model_executor.model_loader.default_loader", "DefaultModelLoader"),
    ):
        try:
            module = __import__(mod, fromlist=[name])
            _wrap(getattr(module, name), "load_weights", "loader.load_weights")
        except Exception:  # noqa: BLE001
            pass


def install_compiler() -> None:
    if not enabled() or "compiler" in _installed:
        return
    _installed.add("compiler")
    try:
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
    except Exception:  # noqa: BLE001
        return

    def compile_name(self, args, kwargs):
        r = kwargs.get("compile_range", args[3] if len(args) > 3 else None)
        return _range_name("xla-compile", r)

    def load_name(self, args, kwargs):
        r = kwargs.get("compile_range", args[4] if len(args) > 4 else None)
        return _range_name("aot-load", r)

    _wrap(TpuCompilerAdaptor, "compile", compile_name)
    _wrap(TpuCompilerAdaptor, "load", load_name)


def install_pd_worker(cls) -> None:
    if not enabled() or "pd" in _installed:
        return
    _installed.add("pd")
    _wrap(cls, "register_runner", "pd.register_runner")
    _wrap(cls, "_collect_pd_semantics", "pd.loaded_state_identity")
