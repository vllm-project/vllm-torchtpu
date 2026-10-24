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
import importlib.util
import subprocess
import sys

import pytest


def test_package_imports_before_vllm():
    result = subprocess.run(
        [sys.executable, "-c", "import vllm_torchtpu; import vllm"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_submodule_imports_before_vllm_initialization_completes():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_torchtpu.layers.adapter.quantization.fp8; import vllm",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_sitecustomize_cold_import_kv_cache_configs_subprocess():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from vllm.v1.core.kv_cache_utils import get_kv_cache_configs; "
                "assert 'vllm.entrypoints.launchers.cli_args' not in sys.modules; "
                "from vllm.v1.core import kv_cache_planning; "
                "assert get_kv_cache_configs is kv_cache_planning.get_kv_cache_configs"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_sitecustomize_cold_import_launchers_cli_args_subprocess():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import vllm.entrypoints.launchers as launchers_pkg; "
                "from vllm.entrypoints.launchers.cli_args import FrontendArgs; "
                "from vllm.entrypoints.common import cli_args as common_cli_args; "
                "assert FrontendArgs is common_cli_args.FrontendArgs; "
                "mod = sys.modules['vllm.entrypoints.launchers.cli_args']; "
                "assert mod is common_cli_args; "
                "assert launchers_pkg.cli_args is common_cli_args; "
                "spec_name = common_cli_args.__spec__.name; "
                "assert spec_name == 'vllm.entrypoints.common.cli_args'"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_sitecustomize_cold_import_serve_api_utils_subprocess():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from vllm.entrypoints.serve.utils.api_utils import "
                "VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup; "
                "from vllm.entrypoints.common import startup; "
                "epilog = startup.VLLM_SUBCMD_PARSER_EPILOG; "
                "assert VLLM_SUBCMD_PARSER_EPILOG is epilog; "
                "assert cli_env_setup is startup.cli_env_setup"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module_name",
    [
        "vllm_omni.diffusion.diffusion_engine",
        "vllm_omni.entrypoints.openai.api_server",
        "vllm_omni.entrypoints.cli.main",
    ],
)
def test_sitecustomize_cold_import_vllm_omni_entrypoints_subprocess(
    module_name: str,
):
    if importlib.util.find_spec("vllm_omni") is None:
        pytest.skip("vllm_omni is not installed in this environment")
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
