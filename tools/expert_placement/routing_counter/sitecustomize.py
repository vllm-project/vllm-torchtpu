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
"""Arms the expert routing counter in every Python process that loads vLLM.

Put this directory first on ``PYTHONPATH`` and set ``EP_ROUTING_COUNTER=1``
(both inherited by the workers). Python imports only the first
``sitecustomize`` on ``sys.path``, so this file first runs the one it shadows
(vllm_torchtpu ships one that sets environment defaults torch_tpu reads at
import). It then installs an import hook that wraps the counter's targets as
they are imported; it imports neither torch nor vLLM itself, so processes
that never load them (Ray agents, CLIs) stay as they were.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# The runner only: its wrapped load_model wraps the routing entry points.
_TARGETS = ("vllm_torchtpu.runner.tpu_runner",)


def _chain_shadowed_sitecustomize() -> None:
    """Runs the next ``sitecustomize`` on sys.path (the one this shadows)."""
    here = os.path.realpath(_THIS_DIR)
    rest = [p for p in sys.path if os.path.realpath(p or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None or spec.origin is None:
        return
    if os.path.realpath(spec.origin) == os.path.realpath(__file__):
        return
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules["_ep_routing_counter_shadowed_sitecustomize"] = module
        spec.loader.exec_module(module)
    except Exception as e:  # same contract as Python's own: report, go on
        print(f"[ep_routing_counter] {spec.origin} failed: {e!r}", file=sys.stderr)


class _PostExecLoader(importlib.abc.Loader):
    """Runs the real loader, then the counter's patch, before the import
    statement binds any name (so ``from m import f`` gets the wrapper)."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        if _THIS_DIR not in sys.path:
            sys.path.insert(0, _THIS_DIR)
        import ep_routing_counter as counter

        counter.install_runner_patch(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder(importlib.abc.MetaPathFinder):
    """Wraps the loader of the target modules; finds nothing itself."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _PostExecLoader(spec.loader)
        return spec


_chain_shadowed_sitecustomize()  # first: it must run before torch is imported
_ON = os.getenv("EP_ROUTING_COUNTER", "0").strip().lower() in ("1", "true", "yes")
if _ON and not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
