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
"""Low-level validation and bucket-list parsers for runtime policy."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NoReturn

from .constants import FALSE_STRS, MAX_BUCKET_ROWS, MIN_VERIFY_TOKENS


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


def _parse_verify_tokens(val: Iterable[int]) -> tuple[int, ...]:
    """Validate and sort unique verify-step token widths (`t >= 2`)."""
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
