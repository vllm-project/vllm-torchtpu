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
"""Runtime constants and low-level parsers for bucket policy and step eligibility."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NoReturn

from .contract import PAR_DP_LOCKSTEP

DEFAULT_BUCKETS: tuple[int, ...] = (8, 16, 32)
DEFAULT_VERIFY_BUCKETS: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)
MTP_VERIFY_TOKENS: int = 4
MTP_SPECULATIVE_TOKENS: int = MTP_VERIFY_TOKENS - 1
MAX_BUCKET_ROWS: int = 64
MIN_VERIFY_TOKENS: int = 2
FALSE_STRS: tuple[str, ...] = ("0", "false", "no", "off", "none", "")
_MIN_OUT_ROW_MULTIPLE: int = 16

STARTUP_FLAG_RULES: tuple[tuple[str, str], ...] = (
    ("unaligned_cache_layout", "requires unified KV and recurrent cache layout"),
    (
        "multimodal_inputs",
        "requires text-only token IDs (multimodal inputs unsupported)",
    ),
    ("lora_enabled", "does not support LoRA adapters"),
    ("pipeline_parallel", "requires single-stage pipeline execution"),
    ("context_parallel", "does not support context parallelism"),
    (PAR_DP_LOCKSTEP, "does not support data-parallel lockstep"),
    ("aux_hidden_states", "does not support auxiliary hidden-state capture"),
)


def _raise_invalid(label: str, val: object) -> NoReturn:
    """Raise a standardized `ValueError` for an invalid policy argument."""
    msg = f"Megakernel: invalid {label}={val!r}"
    raise ValueError(msg)


def _is_exact_int(item: object) -> bool:
    """Return whether `item` is an `int` and not a `bool`."""
    return type(item) is int


def _parse_bucket_list(
    val: str | Iterable[int],
    *,
    valid_fn: Callable[[int], bool],
    label: str,
) -> tuple[int, ...]:
    """Parse, validate, and sort unique integer buckets from `val`."""
    if not isinstance(val, str):
        if isinstance(val, (bytes, bytearray)) or not isinstance(val, Iterable):
            _raise_invalid(label, val)
        parsed_buckets = list(val)
        if any(not _is_exact_int(b) or not valid_fn(b) for b in parsed_buckets) or len(
            set(parsed_buckets)
        ) != len(parsed_buckets):
            _raise_invalid(label, val)
        return tuple(sorted(parsed_buckets))
    raw = val.strip()
    if raw.lower() in FALSE_STRS:
        return ()
    parts = [part.strip() for part in raw.replace(":", ",").split(",") if part.strip()]
    try:
        parsed_buckets = [int(part) for part in parts]
    except ValueError:
        msg = f"Megakernel: invalid {label}={val!r}"
        raise ValueError(msg) from None
    if (
        not parsed_buckets
        or len(set(parsed_buckets)) != len(parsed_buckets)
        or any(not valid_fn(b) for b in parsed_buckets)
    ):
        _raise_invalid(label, val)
    return tuple(sorted(parsed_buckets))


def _parse_verify_tokens(val: Iterable[int] | None) -> tuple[int, ...]:
    """Validate and sort unique verify-step token widths (`t >= 2`)."""
    if val is None:
        return (MTP_VERIFY_TOKENS,)
    if isinstance(val, (str, bytes, bytearray)) or not isinstance(val, Iterable):
        _raise_invalid("verify_tokens", val)
    widths = list(val)
    if (
        not widths
        or any(
            not _is_exact_int(width)
            or width < MIN_VERIFY_TOKENS
            or width > MAX_BUCKET_ROWS
            for width in widths
        )
        or len(set(widths)) != len(widths)
    ):
        _raise_invalid("verify_tokens", val)
    return tuple(sorted(widths))


__all__ = [
    "DEFAULT_BUCKETS",
    "DEFAULT_VERIFY_BUCKETS",
    "FALSE_STRS",
    "MAX_BUCKET_ROWS",
    "MIN_VERIFY_TOKENS",
    "MTP_SPECULATIVE_TOKENS",
    "MTP_VERIFY_TOKENS",
    "STARTUP_FLAG_RULES",
]
