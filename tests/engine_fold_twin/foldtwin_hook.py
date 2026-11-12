# SPDX-License-Identifier: Apache-2.0
"""Engine-core hooks for the fold-on/off twin (test-only; loaded from a .pth).

FOLDTWIN_LOG=<path>: after every Scheduler.schedule, append one JSON line with
  the scheduled tokens per request, the CoW block copies, and every scheduled
  request's block table per KV-cache group (KVCacheManager.get_block_ids).
FOLDTWIN_BURN=<n>: right after the KV cache manager is built, take n blocks off
  the head of the free queue and never return them, so the twin's requests get
  pages that no warmup / dummy run ever wrote (the pool starts zeroed).
"""

import importlib.abc
import importlib.util
import json
import os
import sys

_LOG = os.environ.get("FOLDTWIN_LOG")
_BURN = int(os.environ.get("FOLDTWIN_BURN", "0") or 0)
_NOCKPT = os.environ.get("FOLDTWIN_NOCKPT") == "1"
_SCHED = "vllm.v1.core.sched.scheduler"
_FOLD = "vllm_torchtpu.core.mamba_tail_fold"


def _install_nockpt(module):
    """FOLDTWIN_NOCKPT=1: the worker mirror reports no checkpoint, so the GDN / PLE kernels
    write none; the scheduler and the manager still fold (fold geometry, no in-step write)."""
    import numpy as np

    def no_checkpoints(num_computed, num_scheduled, *a, **k):
        z = np.zeros(len(num_scheduled), dtype=np.int32)
        return z, z.copy()

    module.prefill_checkpoints = no_checkpoints
    print("FOLDTWIN_NOCKPT: prefill_checkpoints returns no checkpoint", file=sys.stderr,
          flush=True)


def _install():
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler.schedule, "_foldtwin", False):
        return
    orig_init = KVCacheManager.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        if _BURN > 0:
            pool = self.block_pool
            n = min(_BURN, pool.get_num_free_blocks() // 2)
            self._foldtwin_burned = pool.get_new_blocks(n)
            ids = [b.block_id for b in self._foldtwin_burned]
            print(f"FOLDTWIN burned {n} blocks [{min(ids)}..{max(ids)}]", file=sys.stderr,
                  flush=True)

    orig_schedule = Scheduler.schedule
    state = {"step": 0}

    def schedule(self, *a, **k):
        out = orig_schedule(self, *a, **k)
        if _LOG:
            try:
                sched = dict(getattr(out, "num_scheduled_tokens", {}) or {})
                copies = [(int(c.src_block_id), int(c.dst_block_id))
                          for c in (getattr(out, "kv_cache_block_copies", None) or [])]
                tables = {}
                computed = {}
                for rid in sched:
                    try:
                        tables[rid] = [list(map(int, g))
                                       for g in self.kv_cache_manager.get_block_ids(rid)]
                    except Exception as e:  # noqa: BLE001
                        tables[rid] = repr(e)
                    req = self.requests.get(rid)
                    if req is not None:
                        computed[rid] = int(req.num_computed_tokens)
                rec = {"step": state["step"], "sched": sched, "copies": copies,
                       "computed_after": computed, "tables": tables}
                with open(_LOG, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:  # noqa: BLE001
                print("FOLDTWIN log failed:", repr(e), file=sys.stderr, flush=True)
        if getattr(out, "num_scheduled_tokens", None):
            state["step"] += 1
        return out

    schedule._foldtwin = True
    KVCacheManager.__init__ = init
    Scheduler.schedule = schedule
    print("FOLDTWIN hooks installed (log=%s burn=%d)" % (_LOG, _BURN), file=sys.stderr,
          flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        try:
            if module.__name__ == _FOLD:
                _install_nockpt(module)
                return
            _install()
        except Exception as e:  # noqa: BLE001
            print("FOLDTWIN install failed:", repr(e), file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    """Patch the target modules right after they execute."""

    targets = (({_SCHED} if (_LOG or _BURN) else set()) | ({_FOLD} if _NOCKPT else set()))

    def find_spec(self, name, path, target=None):
        if name not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if (_LOG or _BURN or _NOCKPT) and not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
