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
import subprocess
import sys

import pytest


def _run_python(code: str, timeout: int = 60) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr


def test_package_imports_before_vllm():
    _run_python("import vllm_torchtpu; import vllm")


def test_submodule_imports_before_vllm_initialization_completes():
    _run_python("import vllm_torchtpu.layers.adapter.quantization.fp8; import vllm")


@pytest.mark.parametrize(
    "imports",
    [
        (
            "from vllm.v1.core.kv_cache_utils import get_kv_cache_configs; "
            "from vllm.v1.core import kv_cache_planning; "
        ),
        # kv_cache_planning imports kv_cache_utils while it is still executing.
        (
            "from vllm.v1.core import kv_cache_planning; "
            "from vllm.v1.core.kv_cache_utils import get_kv_cache_configs; "
        ),
    ],
    ids=["utils_first", "planning_first"],
)
def test_sitecustomize_cold_import_kv_cache_configs_subprocess(imports: str):
    _run_python(
        "import sys; "
        f"{imports}"
        "assert 'vllm.entrypoints.launchers.cli_args' not in sys.modules; "
        "assert get_kv_cache_configs is kv_cache_planning.get_kv_cache_configs"
    )


def test_sitecustomize_cold_import_launchers_cli_args_subprocess():
    _run_python(
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
    )


def test_sitecustomize_cold_import_serve_api_utils_subprocess():
    _run_python(
        "from vllm.entrypoints.serve.utils.api_utils import "
        "VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup; "
        "from vllm.entrypoints.common import startup; "
        "epilog = startup.VLLM_SUBCMD_PARSER_EPILOG; "
        "assert VLLM_SUBCMD_PARSER_EPILOG is epilog; "
        "assert cli_env_setup is startup.cli_env_setup"
    )


def test_sitecustomize_wraps_startup_cli_env_setup_subprocess():
    _run_python(
        "from vllm.entrypoints.common.startup import cli_env_setup; "
        "assert getattr(cli_env_setup, '_rust_wrapped', False)"
    )


def test_sitecustomize_rust_cli_delegation_subprocess():
    import os

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "m",
            "--invalid-option-for-rvllm",
        ],
        capture_output=True,
        text=True,
        env={
            **dict(os.environ),
            "VLLM_TPU_USE_RUST": "1",
            "VLLM_TPU_RVLLM_BINARY": "/bin/true",
        },
        timeout=30,
    )
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "rvllm: error:" in result.stderr
