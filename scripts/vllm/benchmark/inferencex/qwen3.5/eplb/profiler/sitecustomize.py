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
"""Installs the MoE routing profiler at interpreter start.

Put this directory first on PYTHONPATH, in front of the existing entries
(never instead of them), and set VLLM_TPU_ENABLE_MOE_PROFILER=1. Python then
imports this module before vLLM starts. It also runs the next
sitecustomize.py on sys.path (vllm-torchtpu's own, which sets process-wide
torch_tpu defaults), which this module would otherwise shadow.
"""

import importlib.util
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_shadowed_sitecustomize() -> None:
    """Executes the first other sitecustomize.py on sys.path, if any."""
    for entry in sys.path:
        path = os.path.join(entry or os.getcwd(), "sitecustomize.py")
        same_dir = os.path.abspath(os.path.dirname(path)) == _THIS_DIR
        if same_dir or not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location("_shadowed_sitecustomize", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return


def _install_profiler() -> None:
    if os.getenv("VLLM_TPU_ENABLE_MOE_PROFILER", "0").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    try:
        from hook_patch import apply_runtime_hook  # pylint: disable=g-import-not-at-top

        ok = apply_runtime_hook()
        state = "installed" if ok else "NOT installed"
        print(
            f"[sitecustomize] EPLB MoE profiler hook {state} in PID {os.getpid()}",
            flush=True,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            "[sitecustomize] Error installing the EPLB MoE profiler hook in PID"
            f" {os.getpid()}: {e}",
            flush=True,
        )


_run_shadowed_sitecustomize()
_install_profiler()
