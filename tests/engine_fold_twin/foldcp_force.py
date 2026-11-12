# SPDX-License-Identifier: Apache-2.0
"""Loaded from foldcp_force.pth in every Python process of the pod. With
FOLDCP_FORCE=1 it makes vllm_torchtpu.core.mamba_tail_fold's
fold_blocked_by_context_parallel return None (the fold stays on under
main-KV DCP, as VLLM_TPU_MAMBA_TAIL_FOLD_DCP=1 did on za6). Test-only."""

import importlib.abc
import os
import sys

TARGET = "vllm_torchtpu.core.mamba_tail_fold"


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        module.fold_blocked_by_context_parallel = lambda cfg, main_kv_dcp: None
        print("FOLDCP_FORCE: fold_blocked_by_context_parallel bypassed", file=sys.stderr,
              flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            import importlib.util

            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("FOLDCP_FORCE") == "1" and not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
