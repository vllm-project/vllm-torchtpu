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
"""Rollback-safe startup preflight validation for TPU runtime environment."""

import difflib
import warnings
from dataclasses import dataclass, field
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

EXTERNAL_TPU_RUNTIME_ENVS: frozenset[str] = frozenset(
    {
        "TPU_ACCELERATOR_TYPE",
        "TPU_ALLOW_UNSUPPORTED_CHIPS",
        "TPU_BACKEND_PORT",
        "TPU_BACKEND_TYPE",
        "TPU_CHIPS_PER_HOST",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_CHIP_TOPOLOGY",
        "TPU_COORDINATOR_ADDRESS",
        "TPU_COORDINATOR_CLIENT_ID",
        "TPU_COORDINATOR_PORT",
        "TPU_COORDINATOR_SERVICE_ACCOUNT",
        "TPU_DEVICE_ORDER",
        "TPU_DEVICE_PATHS",
        "TPU_DONT_WAIT_FOR_TOPOLOGY",
        "TPU_ENABLE_CUSTOM_BARRIER",
        "TPU_FAST_CHIP_INIT",
        "TPU_HOST_BOUNDS",
        "TPU_HOST_COUNT",
        "TPU_LIBRARY_PATH",
        "TPU_LOAD_LIBRARY_OVERRIDE",
        "TPU_LOG_DIR",
        "TPU_MAX_CORES_PER_PROCESS",
        "TPU_MEGACORE",
        "TPU_MESH_CONTROLLER_ADDRESS",
        "TPU_MESH_CONTROLLER_PORT",
        "TPU_MIN_LOG_LEVEL",
        "TPU_MULTIPROCESS_DP",
        "TPU_NAME",
        "TPU_NODE_ID",
        "TPU_NUM_DEVICES",
        "TPU_NUM_HOSTS",
        "TPU_PCP8_160K_PROFILE_DIR",
        "TPU_PCP8_160K_RESULT_DIR",
        "TPU_PCP8_PROFILE_SESSION_KEY",
        "TPU_PREMAPPED_BUFFER_SIZE",
        "TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES",
        "TPU_PROCESS_ADDRESSES",
        "TPU_PROCESS_BOUNDS",
        "TPU_PROCESS_PORT",
        "TPU_PROJECT",
        "TPU_RUN_PCP8_160K_BENCHMARK",
        "TPU_RUNTIME_METRICS_PORTS",
        "TPU_SKIP_INIT",
        "TPU_SKIP_MDS_QUERY",
        "TPU_SLICE_BUILDER_DUMP_PATH",
        "TPU_STDERR_LOG_LEVEL",
        "TPU_TEST_ONLY_KNOB",
        "TPU_TOPOLOGY",
        "TPU_TOPOLOGY_ALT",
        "TPU_TOPOLOGY_WRAP",
        "TPU_VERSION",
        "TPU_VISIBLE_CHIPS",
        "TPU_VISIBLE_DEVICES",
        "TPU_VISIBLE_DEVICE_PATHS",
        "TPU_VM_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_WORKER_ID",
        "TPU_ZONE",
        "VLLM_TPU_PROFILE_DELAY_MS",
        "VLLM_TPU_PROFILE_DURATION_MS",
        "VLLM_TPU_USING_PATHWAYS",
        "VLLM_TPU_WEIGHT_LOAD_SYNC_INTERVAL",
    }
)

SECRET_ENV_KEYS: frozenset[str] = frozenset(
    {
        "RAIDEN_SHM_KEY",
        "VLLM_TORCHTPU_IPC_KEY",
    }
)


def _is_secret_key(key: str) -> bool:
    return key in SECRET_ENV_KEYS or key.endswith(
        ("_TOKEN", "_SECRET", "_PASSWORD", "_API_KEY", "_IPC_KEY", "_SHM_KEY")
    )


@dataclass
class PreflightReport:
    """Startup validation summary for TPU runtime environment configuration."""

    configured_envs: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compile_factors_digest: str = ""


def preflight_check(
    vllm_config: Any = None,
    *,
    expected_compile_digest: str | None = None,
) -> PreflightReport:
    """Validate TPU environment configuration at startup."""
    del vllm_config
    fatal_errors: list[str] = []
    warnings_list: list[str] = []

    try:
        strict_env = bool(envs._uncached_get("VLLM_TPU_STRICT_ENV"))
    except ValueError:
        strict_env = False

    env_snapshot = envs.raw_environ_snapshot()

    for key, tombstone_msg in envs.REMOVED_ENVS.items():
        if key in env_snapshot:
            fatal_errors.append(tombstone_msg)

    for legacy_key, new_key in sorted(envs.DEPRECATED_ENVS.items()):
        if legacy_key in env_snapshot:
            val = env_snapshot[legacy_key]
            msg = (
                f"Environment variable '{legacy_key}' is deprecated and will be "
                f"removed in a future release. Use '{new_key}' instead."
            )
            if strict_env:
                fatal_errors.append(msg)
            else:
                warnings.warn(msg, FutureWarning, stacklevel=2)
                warnings_list.append(msg)
                if new_key not in env_snapshot:
                    envs.set_env_override(new_key, val)
                    env_snapshot[new_key] = val

    for name in sorted(envs.environment_variables):
        if name in env_snapshot:
            try:
                envs._uncached_get(name)
            except ValueError as exc:
                fatal_errors.append(f"Invalid value for {name}: {exc}")

    candidates = (
        list(envs.environment_variables.keys())
        + list(envs.DEPRECATED_ENVS.keys())
        + list(envs.REMOVED_ENVS.keys())
    )
    for k in sorted(env_snapshot):
        if not (
            k.startswith("TPU_")
            or k.startswith("VLLM_TPU_")
            or k.startswith("VLLM_TORCHTPU_")
        ):
            continue
        if k in envs.environment_variables:
            continue
        if k in envs.DEPRECATED_ENVS:
            continue
        if k in envs.REMOVED_ENVS:
            continue
        if k in EXTERNAL_TPU_RUNTIME_ENVS:
            continue
        if k.startswith("TPU_MEGASCALE_"):
            continue

        matches = difflib.get_close_matches(k, candidates, n=1, cutoff=0.5)
        if matches:
            warn_msg = (
                f"Unknown TPU environment variable '{k}'. Did you mean '{matches[0]}'?"
            )
        else:
            warn_msg = f"Unknown TPU environment variable '{k}'."

        if strict_env:
            fatal_errors.append(warn_msg)
        else:
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
            warnings_list.append(warn_msg)

    current_digest = ""
    if not fatal_errors:
        current_digest = envs.compile_factors_digest()
        expected_digest = (
            expected_compile_digest
            if expected_compile_digest is not None
            else envs._uncached_get("VLLM_TORCHTPU_COMPILE_FACTORS_DIGEST")
        )
        if expected_digest and expected_digest != current_digest:
            fatal_errors.append(
                f"TPU compile factors digest mismatch: expected '{expected_digest}', "
                f"got '{current_digest}'."
            )

    if fatal_errors:
        header = "TPU startup preflight check failed:\n- "
        raise ValueError(header + "\n- ".join(fatal_errors))

    configured_envs: list[tuple[str, str]] = []
    for k in sorted(envs.environment_variables):
        if k in env_snapshot:
            val = "***" if _is_secret_key(k) else env_snapshot[k]
            configured_envs.append((k, val))

    if envs.is_rank_zero_env():
        summary = ", ".join(f"{k}={v}" for k, v in configured_envs) or "defaults"
        logger.info(
            "TPU env preflight: compile_factors_digest=%s configured=[%s]",
            current_digest,
            summary,
        )

    return PreflightReport(
        configured_envs=configured_envs,
        warnings=warnings_list,
        compile_factors_digest=current_digest,
    )
