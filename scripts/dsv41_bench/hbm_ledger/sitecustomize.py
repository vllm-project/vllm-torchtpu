"""HBM ledger probe for DSv4.1 serving runs, gated by DSV41_HBM_LEDGER=1.

Put this directory on PYTHONPATH. Every interpreter (incl. spawned vLLM
workers) imports `sitecustomize` at startup; when the gate is set we wrap
`TPUWorker` methods to log, per rank:
  * HBM used/limit (torch.accelerator.get_memory_info) after load_model,
    determine_available_memory, initialize_from_config, warmup, and every
    DSV41_HBM_LEDGER_EVERY execute_model calls (default 16);
  * peak stats from torch.accelerator.memory_stats when the backend has them;
  * parameter + buffer bytes grouped by module-name category after load.
No source under src/ is touched, so the measured baseline stays unmodified.
The next `sitecustomize` on sys.path (e.g. src/sitecustomize.py, which sets
torch_tpu env defaults) is chained so it is not shadowed.
"""

import os


def _chain_next_sitecustomize():
    import importlib.machinery
    import importlib.util
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


_chain_next_sitecustomize()

if os.environ.get("DSV41_HBM_LEDGER") == "1":
    import importlib.abc
    import importlib.util
    import sys

    _TARGET = "vllm_torchtpu.worker.tpu_worker"
    _GIB = float(1 << 30)

    def _mem_line(worker, stage):
        import torch

        devs = getattr(worker, "devices", None) or [None]
        dev = devs[0]
        try:
            free, limit = torch.accelerator.get_memory_info(dev)
            msg = f"used={(limit - free) / _GIB:.3f}GiB limit={limit / _GIB:.3f}GiB"
        except Exception as e:  # noqa: BLE001 - diagnostic only
            msg = f"get_memory_info failed: {e!r}"
        try:
            stats = torch.accelerator.memory_stats(dev)
            keys = [k for k in stats if "peak" in k or "in_use" in k]
            msg += " " + " ".join(f"{k}={stats[k] / _GIB:.3f}GiB" for k in keys)
        except Exception as e:  # noqa: BLE001
            msg += f" memory_stats n/a ({type(e).__name__})"
        print(
            f"[HBM_LEDGER] rank={getattr(worker, 'rank', '?')} stage={stage} {msg}",
            flush=True,
        )

    def _category(name):
        n = name.lower()
        for key, cat in (
            ("engram", "engram"),
            ("experts", "routed_experts"),
            ("shared_expert", "shared_experts"),
            ("embed", "embed"),
            ("lm_head", "lm_head"),
            ("indexer", "indexer"),
            ("compressor", "compressor"),
            ("attn", "attn"),
            ("mhc", "mhc"),
            ("hc_", "mhc"),
            ("gate", "router"),
            ("norm", "norm"),
        ):
            if key in n:
                return cat
        return "other"

    def _param_ledger(worker):
        model = getattr(getattr(worker, "model_runner", None), "model", None)
        if model is None:
            print("[HBM_LEDGER] param ledger: model_runner.model missing", flush=True)
            return
        totals = {}
        seen = set()
        for kind, it in (
            ("param", model.named_parameters(remove_duplicate=False)),
            ("buffer", model.named_buffers(remove_duplicate=False)),
        ):
            for name, t in it:
                try:
                    key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
                except Exception:  # noqa: BLE001
                    key = id(t)
                if key in seen:
                    continue
                seen.add(key)
                cat = f"{kind}:{_category(name)}"
                if t.device.type != "tpu":
                    cat += f"@{t.device.type}"
                totals[cat] = totals.get(cat, 0) + t.numel() * t.element_size()
        total = sum(v for k, v in totals.items() if "@" not in k)
        body = " ".join(f"{k}={v / _GIB:.3f}" for k, v in sorted(totals.items()))
        print(
            f"[HBM_LEDGER] rank={worker.rank} stage=param_ledger "
            f"tpu_total={total / _GIB:.3f}GiB {body}",
            flush=True,
        )

    def _patch(mod):
        cls = mod.TPUWorker
        every = int(os.environ.get("DSV41_HBM_LEDGER_EVERY", "16"))

        def wrap(name, after):
            orig = getattr(cls, name)

            def inner(self, *a, **k):
                out = orig(self, *a, **k)
                try:
                    after(self, out)
                except Exception as e:  # noqa: BLE001
                    print(f"[HBM_LEDGER] {name} probe failed: {e!r}", flush=True)
                return out

            inner.__name__ = name
            setattr(cls, name, inner)

        def after_load(self, _):
            _mem_line(self, "after_load_model")
            _param_ledger(self)

        max_steps = int(os.environ.get("DSV41_HBM_LEDGER_MAX_STEPS", "64"))
        all_ranks = os.environ.get("DSV41_HBM_LEDGER_ALL_RANKS") == "1"

        def after_exec(self, _):
            if not all_ranks and getattr(self, "rank", 0) != 0:
                return
            self._hbm_ledger_n = getattr(self, "_hbm_ledger_n", 0) + 1
            if self._hbm_ledger_n <= max_steps and self._hbm_ledger_n % every == 0:
                _mem_line(self, f"execute_model#{self._hbm_ledger_n}")

        wrap("load_model", after_load)

        def after_avail(self, out):
            _mem_line(self, f"after_determine_avail(kv={out / _GIB:.3f}GiB)")

        wrap("determine_available_memory", after_avail)
        wrap("initialize_from_config", lambda s, _: _mem_line(s, "after_kv_alloc"))
        wrap("compile_or_warm_up_model", lambda s, _: _mem_line(s, "after_warmup"))
        wrap("execute_model", after_exec)

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            _patch(module)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
