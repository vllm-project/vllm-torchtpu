# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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
"""Idempotent installer to wire agentic telemetry into a vllm-torchtpu tree."""

import sys
from pathlib import Path

PINNED_OTEL_PACKAGES: tuple[str, ...] = (
    "opentelemetry-api==1.27.0",
    "opentelemetry-sdk==1.27.0",
    "opentelemetry-exporter-otlp-proto-http==1.27.0",
    "opentelemetry-exporter-otlp-proto-grpc==1.27.0",
)


def install_into_tree(pkg_root: Path | None = None) -> None:
    """Wire telemetry env vars and `install_agentic_telemetry_patches` in-place.

    Raises `RuntimeError` if target files or expected anchors are missing.
    """
    if pkg_root is None:
        pkg_root = Path(__file__).resolve().parent.parent

    envs_path = pkg_root / "envs.py"
    if not envs_path.exists():
        raise RuntimeError(f"Missing target file: {envs_path}")
    envs_text = envs_path.read_text(encoding="utf-8")
    if '"VLLM_TPU_AGENTIC_TELEMETRY"' not in envs_text:
        anchor = "environment_variables: dict[str, Callable[[], Any]] = {\n"
        if anchor not in envs_text:
            raise RuntimeError(
                f"Failed to locate environment_variables anchor in {envs_path}"
            )
        addition = (
            '    "VLLM_TPU_AGENTIC_TELEMETRY": '
            'env_bool("VLLM_TPU_AGENTIC_TELEMETRY", default=False),\n'
            '    "VLLM_TPU_AGENTIC_SPAN_LOG": '
            'env_str("VLLM_TPU_AGENTIC_SPAN_LOG", default=""),\n'
            '    "VLLM_REPLICA_ID": '
            'env_str("VLLM_REPLICA_ID", default=""),\n'
            '    "VLLM_TPU_AGENTIC_BLOCK_SIZE": '
            'env_nonnegative_int("VLLM_TPU_AGENTIC_BLOCK_SIZE", 0),\n'
        )
        envs_path.write_text(
            envs_text.replace(anchor, anchor + addition, 1),
            encoding="utf-8",
        )

    reg_path = pkg_root / "patch_registry.py"
    if not reg_path.exists():
        raise RuntimeError(f"Missing target file: {reg_path}")
    reg_text = reg_path.read_text(encoding="utf-8")
    target_str = "vllm_torchtpu.telemetry.vllm_hooks:install_agentic_telemetry_patches"
    if target_str not in reg_text:
        reload_fn = (
            "vllm_torchtpu.platforms.tpu_platform:"
            "_patch_api_server_kernel_reload_endpoint"
        )
        anchor = f'        "{reload_fn}",\n        _ACTIVE,\n    ),\n'
        if anchor not in reg_text:
            raise RuntimeError(f"Failed to locate PATCHES anchor in {reg_path}")
        addition = f'    Patch(\n        "{target_str}",\n        _ENGINE,\n    ),\n'
        reg_path.write_text(
            reg_text.replace(anchor, anchor + addition, 1),
            encoding="utf-8",
        )

    compiler_path = pkg_root / "compilation" / "tpu_compiler.py"
    if not compiler_path.exists():
        raise RuntimeError(f"Missing target file: {compiler_path}")
    compiler_text = compiler_path.read_text(encoding="utf-8")
    ignored_env_names = (
        "VLLM_TPU_AGENTIC_TELEMETRY",
        "VLLM_TPU_AGENTIC_SPAN_LOG",
        "VLLM_REPLICA_ID",
        "VLLM_TPU_AGENTIC_BLOCK_SIZE",
    )
    if any(f'"{name}"' not in compiler_text for name in ignored_env_names):
        anchor = "_TPU_COMPILE_ENV_IGNORED = {\n"
        if anchor not in compiler_text:
            raise RuntimeError(
                f"Failed to locate _TPU_COMPILE_ENV_IGNORED anchor in {compiler_path}"
            )
        missing_lines = "".join(
            f'    "{env_name}",\n'
            for env_name in ignored_env_names
            if f'"{env_name}"' not in compiler_text
        )
        compiler_path.write_text(
            compiler_text.replace(anchor, anchor + missing_lines, 1),
            encoding="utf-8",
        )


if __name__ == "__main__":
    install_into_tree()
    sys.stdout.write("CUJ-2 agentic telemetry wired into vllm_torchtpu.\n")
