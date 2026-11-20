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
import os
import re
from typing import Any

import vllm_torchtpu.envs as envs

_STANDARD_KEYS = frozenset(
    {
        "host_tracer_level",
        "device_tracer_level",
        "python_tracer_level",
    }
)

_TORCH_PROFILER_KEYS = frozenset(
    {
        "record_shapes",
        "profile_memory",
        "with_stack",
        "with_flops",
        "with_modules",
    }
)

_ALLOWED_VALUE_CHARS = re.compile(r"^[a-zA-Z0-9_./,:\-\s]*$")
_OPTION_PATTERN = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$")


def _parse_option_value(key: str, val: str) -> Any:
    val_lower = val.lower()
    if val_lower == "true":
        return True
    if val_lower == "false":
        return False
    try:
        return int(val)
    except ValueError:
        if not _ALLOWED_VALUE_CHARS.match(val):
            raise ValueError(
                f"Invalid characters in option value '{val}' for key '{key}'"
            ) from None
        return val


def parse_profile_options(
    profile_prefix: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not profile_prefix or not profile_prefix.strip():
        return {}, {}

    if not profile_prefix.isascii():
        raise ValueError("profile_prefix contains non-ASCII characters")

    standard_opts = {}
    advanced_opts = {}

    parts = profile_prefix.split(";")

    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = _OPTION_PATTERN.match(part)
        if not match:
            raise ValueError(
                f"Invalid profile option format in '{part}'. "
                "Expected 'key:value' format."
            )
        key = match.group(1)
        val = match.group(2)

        parsed_val = _parse_option_value(key, val)

        if key in _STANDARD_KEYS:
            standard_opts[key] = parsed_val
        else:
            advanced_opts[key] = parsed_val

    return standard_opts, advanced_opts


def get_default_standard_opts() -> dict[str, Any]:
    return {
        "host_tracer_level": envs.HOST_TRACER_LEVEL,
        "device_tracer_level": envs.DEVICE_TRACER_LEVEL,
        "python_tracer_level": envs.PYTHON_TRACER_LEVEL,
    }


def get_default_advanced_opts() -> dict[str, Any]:
    return {
        "tpu_trace_mode": envs.TPU_TRACE_MODE,
        "tpu_num_sparse_cores_to_trace": envs.TPU_NUM_SPARSE_CORES_TO_TRACE,
        "tpu_num_sparse_core_tiles_to_trace": envs.TPU_NUM_SPARSE_CORE_TILES_TO_TRACE,
    }


def extract_torch_profiler_kwargs(
    profiler_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extracts standard ``torch.profiler.profile`` boolean flags."""
    if not profiler_kwargs:
        return {}
    torch_kwargs: dict[str, Any] = {}
    for key in _TORCH_PROFILER_KEYS:
        if key in profiler_kwargs:
            val = profiler_kwargs[key]
            if isinstance(val, bool):
                torch_kwargs[key] = val
            else:
                parsed = _parse_option_value(key, str(val))
                if not isinstance(parsed, bool):
                    raise ValueError(f"Expected boolean value for '{key}', got '{val}'")
                torch_kwargs[key] = parsed
    return torch_kwargs


def _merge_profiler_option(
    key: str,
    val: Any,
    standard_opts: dict[str, Any],
    advanced_opts: dict[str, Any],
) -> None:
    if isinstance(val, (bool, int)):
        parsed_val: Any = val
    else:
        parsed_val = _parse_option_value(str(key), str(val))
    if key in _STANDARD_KEYS:
        standard_opts[key] = parsed_val
    else:
        advanced_opts[key] = parsed_val


def resolve_profile_dir_and_opts(
    base_dir: str,
    profile_prefix: str | None = None,
    profiler_kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """
    Resolves the target profiling directory and parses dynamic tracing options.

    Args:
        base_dir: The base directory for trace outputs.
        profile_prefix: A string. If it contains a semicolon or colon, it's
            parsed as structured configuration (e.g., "host_tracer_level:1;").
            Otherwise, it acts solely as a subdirectory name appending to base_dir.
        profiler_kwargs: Optional dictionary of profiler options passed via
            start_profile (e.g., {"host_tracer_level": 3,
            "e2e_enable_fw_throttle_event": True}).

    Returns:
        A tuple of (profile_dir, standard_opts, advanced_opts) where the opts
        dictionaries are pre-populated with baseline defaults and merged with
        any overrides specified in profile_prefix and profiler_kwargs.
    """
    standard_opts = get_default_standard_opts()
    advanced_opts = get_default_advanced_opts()
    profile_dir = base_dir

    effective_prefix = profile_prefix
    if (
        not effective_prefix
        and profiler_kwargs
        and isinstance(profiler_kwargs.get("profile_prefix"), str)
    ):
        effective_prefix = profiler_kwargs["profile_prefix"]

    if effective_prefix:
        if ":" in effective_prefix or ";" in effective_prefix:
            parsed_standard, parsed_advanced = parse_profile_options(effective_prefix)
            standard_opts.update(parsed_standard)
            advanced_opts.update(parsed_advanced)
        else:
            profile_dir = os.path.join(base_dir, effective_prefix)

    if profiler_kwargs:
        for key, val in profiler_kwargs.items():
            if key == "profile_prefix" or key in _TORCH_PROFILER_KEYS:
                continue
            if key == "experimental_options" and isinstance(val, dict):
                for exp_key, exp_val in val.items():
                    _merge_profiler_option(
                        str(exp_key), exp_val, standard_opts, advanced_opts
                    )
                continue
            _merge_profiler_option(str(key), val, standard_opts, advanced_opts)

    return profile_dir, standard_opts, advanced_opts
