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
"""Process-wide defaults required before torch_tpu autoloads."""

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import types
from collections.abc import Callable, Sequence
from typing import Any

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")


class _PostExecLoader(importlib.abc.Loader):
    """Wraps a module loader to attach compatibility aliases after execution."""

    def __init__(
        self,
        loader: importlib.abc.Loader,
        post_exec: Callable[[Any], None],
    ) -> None:
        self._loader = loader
        self._post_exec = post_exec

    def create_module(
        self, spec: importlib.machinery.ModuleSpec
    ) -> types.ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: types.ModuleType) -> None:
        module.__loader__ = self._loader
        if module.__spec__ is not None:
            module.__spec__.loader = self._loader
        self._loader.exec_module(module)
        self._post_exec(module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _VllmOmniCompatFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Lazy compatibility shim for vllm-omni==0.32.0rc1 against vLLM main HEAD.

    vllm-omni 0.32.0rc1 references three vLLM paths that were relocated in
    upstream vLLM after v0.32.0:
    1. `vllm.entrypoints.launchers.cli_args` -> moved to
       `vllm.entrypoints.common.cli_args` in vllm#62151 (imported by
       `vllm_omni.config.omni_config`, `vllm_omni.entrypoints.cli.serve`, and
       `vllm_omni.entrypoints.openai.api_server`).
    2. `vllm.v1.core.kv_cache_utils.get_kv_cache_configs` -> moved to
       `vllm.v1.core.kv_cache_planning` in vllm#61183 (imported by
       `vllm_omni.diffusion.diffusion_kv.initialization` in both the
       orchestrator and spawned worker/stage processes).
    3. `vllm.entrypoints.serve.utils.api_utils` (`VLLM_SUBCMD_PARSER_EPILOG`,
       `cli_env_setup`) -> moved to `vllm.entrypoints.common.startup` in
       vllm#62151 (imported by `vllm_omni.entrypoints.cli.{main,serve}` and
       `vllm_omni.entrypoints.cli.benchmark.main`).

    TODO: Remove _VllmOmniCompatFinder once vllm-omni cuts a release beyond
    0.32.0rc1 compatible with vllm > 0.32.0.
    """

    _LAUNCHERS_PKG = "vllm.entrypoints.launchers"
    _LAUNCHERS_CLI_ARGS = "vllm.entrypoints.launchers.cli_args"
    _KV_CACHE_UTILS = "vllm.v1.core.kv_cache_utils"
    _SERVE_API_UTILS = "vllm.entrypoints.serve.utils.api_utils"

    @staticmethod
    def _patch_kv_cache_utils(module: Any) -> None:
        from vllm.v1.core import kv_cache_planning

        if not hasattr(module, "get_kv_cache_configs"):
            module.get_kv_cache_configs = kv_cache_planning.get_kv_cache_configs

    @staticmethod
    def _patch_serve_api_utils(module: Any) -> None:
        from vllm.entrypoints.common import startup

        if not hasattr(module, "VLLM_SUBCMD_PARSER_EPILOG"):
            module.VLLM_SUBCMD_PARSER_EPILOG = startup.VLLM_SUBCMD_PARSER_EPILOG
        if not hasattr(module, "cli_env_setup"):
            module.cli_env_setup = startup.cli_env_setup

    @classmethod
    def apply_to_loaded_modules(cls) -> None:
        kv_cache_utils = sys.modules.get(cls._KV_CACHE_UTILS)
        if kv_cache_utils is not None:
            cls._patch_kv_cache_utils(kv_cache_utils)
        serve_api_utils = sys.modules.get(cls._SERVE_API_UTILS)
        if serve_api_utils is not None:
            cls._patch_serve_api_utils(serve_api_utils)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == self._LAUNCHERS_CLI_ARGS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=False)
        if fullname == self._KV_CACHE_UTILS:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PostExecLoader(spec.loader, self._patch_kv_cache_utils)
            return spec
        if fullname == self._SERVE_API_UTILS:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PostExecLoader(spec.loader, self._patch_serve_api_utils)
            return spec
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        target = importlib.import_module("vllm.entrypoints.common.cli_args")
        sys.modules[module.__name__] = target
        launchers_pkg = sys.modules.get(self._LAUNCHERS_PKG)
        if launchers_pkg is not None:
            launchers_pkg_any: Any = launchers_pkg
            launchers_pkg_any.cli_args = target


if not any(type(f).__name__ == "_VllmOmniCompatFinder" for f in sys.meta_path):
    sys.meta_path.insert(0, _VllmOmniCompatFinder())
_VllmOmniCompatFinder.apply_to_loaded_modules()
