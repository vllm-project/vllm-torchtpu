"""q38-cache-tail-i1 mutant arm X only (put on PYTHONPATH by q38_cache_tail_i1.py; never on a served engine).

With Q38_TAIL_MUTANT=drop_cow, every process that imports the TPU runner gets a TPUModelRunner whose page-local
copy-on-write block copies are dropped, so a partial-tail prefix hit resumes from an unfilled block. The identity
check must then see different tokens; if it does not, it cannot detect a broken tail. Chains to any other
sitecustomize on sys.path afterwards."""
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "vllm_torchtpu.runner.tpu_runner"


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            pass
        if spec is None or spec.loader is None:
            return spec
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            runner = module.TPUModelRunner

            def _dropped_copies(self, copies):
                if not getattr(runner, "_q38_mutant_logged", False):
                    runner._q38_mutant_logged = True
                    print("[q38 mutant] TPUModelRunner._apply_kv_cache_block_copies dropped "
                          f"({len(copies)} pairs and all later ones)", flush=True)

            runner._apply_kv_cache_block_copies = _dropped_copies

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("Q38_TAIL_MUTANT") == "drop_cow":
    sys.meta_path.insert(0, _PatchOnImport())

_here = os.path.dirname(os.path.abspath(__file__))
for _p in list(sys.path):
    if not _p or os.path.abspath(_p) == _here:
        continue
    _f = os.path.join(_p, "sitecustomize.py")
    if os.path.exists(_f):
        exec(compile(open(_f).read(), _f, "exec"), {"__name__": "sitecustomize_chained", "__file__": _f})
        break
