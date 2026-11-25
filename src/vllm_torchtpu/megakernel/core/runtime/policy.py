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
"""Bucket policy parsing and selection for decode and verify batches.

Handles the `T=1` decode and `T>=2` speculative verify bucket lists.

Framework-free: imports no jax, torch, or vLLM (stdlib only at run time).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from .constants import (
    DEFAULT_BUCKETS,
    DEFAULT_VERIFY_BUCKETS,
    FALSE_STRS,
    MAX_BUCKET_ROWS,
    MIN_VERIFY_TOKENS,
    MTP_SPECULATIVE_TOKENS,
    MTP_VERIFY_TOKENS,
)

if TYPE_CHECKING:
    from ..profile import Profile

_MIN_OUT_ROW_MULTIPLE: int = 16


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


def parse_mk_buckets(val: str | Iterable[int]) -> tuple[int, ...]:
    """Parse `T=1` decode batch buckets from a string or iterable (`'0'` disables).

    Args:
        val: Comma- or colon-separated bucket string, `'0'`/`'false'`/`'none'`
            to disable decode buckets, or an iterable of unique positive ints.

    Returns:
        Sorted tuple of unique validated `T=1` decode bucket row counts.

    Raises:
        ValueError: If `val` is malformed, contains duplicates, or has entries
            outside `[1, MAX_BUCKET_ROWS]`.
    """
    return _parse_bucket_list(
        val,
        valid_fn=lambda b: 1 <= b <= MAX_BUCKET_ROWS,
        label="decode buckets",
    )


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


def parse_mk_verify_buckets(
    val: str | Iterable[int],
    verify_tokens: Sequence[int] = (MTP_VERIFY_TOKENS,),
) -> tuple[int, ...]:
    """Parse verify row buckets from a string or iterable (`'0'` disables).

    Args:
        val: Comma- or colon-separated bucket string, `'0'`/`'false'`/`'none'`
            to disable verify buckets, or an iterable of unique positive ints.
        verify_tokens: Allowed verify-step token widths (`t >= 2`). Every
            bucket in `val` must be divisible by at least one width.

    Returns:
        Sorted tuple of unique validated verify bucket row counts.

    Raises:
        ValueError: If `verify_tokens` or `val` is invalid or unaligned.
    """
    widths = _parse_verify_tokens(verify_tokens)
    min_width = widths[0]
    return _parse_bucket_list(
        val,
        valid_fn=lambda b: (
            min_width <= b <= MAX_BUCKET_ROWS
            and any(b % width == 0 for width in widths)
        ),
        label="verify_buckets",
    )


def select_bucket(num_tokens: int, buckets: Sequence[int] = DEFAULT_BUCKETS) -> int:
    """Return the smallest compiled bucket `>= num_tokens`.

    Args:
        num_tokens: Active token rows requiring a compiled bucket.
        buckets: Sorted ascending sequence of compiled bucket row counts.

    Returns:
        Smallest bucket in `buckets` greater than or equal to `num_tokens`.

    Raises:
        ValueError: If `buckets` is empty or `num_tokens` exceeds every bucket.
    """
    for bucket in buckets:
        if bucket >= num_tokens:
            return bucket
    msg = f"num_tokens={num_tokens} exceeds max bucket in {buckets}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BucketPolicy:
    """Resolved `T=1` decode and `T>=2` verify buckets and output row paddings.

    Attributes:
        buckets: Sorted `T=1` decode bucket row counts (`()` disables decode).
        verify_buckets: Sorted `T>=2` verify bucket row counts (`()` disables).
        token_paddings: Sorted runner output-row padding sizes.
        verify_tokens: Sorted allowed verify-step token widths (`t >= 2`).
    """

    buckets: tuple[int, ...] = DEFAULT_BUCKETS
    verify_buckets: tuple[int, ...] = DEFAULT_VERIFY_BUCKETS
    token_paddings: tuple[int, ...] = ()
    verify_tokens: tuple[int, ...] = (MTP_VERIFY_TOKENS,)

    def __post_init__(self) -> None:
        """Validate and canonicalize bucket tuples on construction."""
        resolved_widths = _parse_verify_tokens(self.verify_tokens)
        object.__setattr__(self, "verify_tokens", resolved_widths)
        object.__setattr__(self, "buckets", parse_mk_buckets(self.buckets))
        object.__setattr__(
            self,
            "verify_buckets",
            _parse_bucket_list(
                self.verify_buckets,
                valid_fn=lambda b: (
                    1 <= b <= MAX_BUCKET_ROWS
                    and any(b % width == 0 for width in resolved_widths)
                ),
                label="verify_buckets",
            ),
        )
        raw_paddings = self.token_paddings
        if isinstance(raw_paddings, (str, bytes, bytearray)) or not isinstance(
            raw_paddings, Iterable
        ):
            _raise_invalid("token_paddings", raw_paddings)
        paddings = list(raw_paddings)
        if any(not _is_exact_int(pad) or pad < 1 for pad in paddings):
            _raise_invalid("token_paddings", raw_paddings)
        object.__setattr__(self, "token_paddings", tuple(sorted(set(paddings))))

    @classmethod
    def resolve(
        cls,
        profile: Profile,
        buckets: Sequence[int] | None = None,
        verify_buckets: Sequence[int] | None = None,
        *,
        speculative_tokens: int | None = None,
        buckets_explicit: bool | None = None,
        verify_buckets_explicit: bool | None = None,
        token_paddings: Sequence[int] = (),
        verify_tokens: Sequence[int] | None = None,
    ) -> BucketPolicy:
        """Resolve `(buckets, verify_buckets)` from explicit overrides or `profile`.

        Args:
            profile: Active megakernel profile supplying default bucket lists.
            buckets: Optional decode bucket sequence; `None` uses `profile` or
                `DEFAULT_BUCKETS`.
            verify_buckets: Optional verify bucket sequence; `None` uses
                `profile` or `DEFAULT_VERIFY_BUCKETS`.
            speculative_tokens: Number of draft tokens (`K`), or `None` when
                unspecified (`0` disables speculative verify; `K >= 1` implies
                verify width `K + 1` unless `verify_tokens` overrides it).
            buckets_explicit: Whether `buckets` came from an explicit user
                override; `None` infers `buckets is not None`.
            verify_buckets_explicit: Whether `verify_buckets` came from an
                explicit user override; `None` infers `verify_buckets is not None`.
            token_paddings: Runner output-row padding sizes.
            verify_tokens: Allowed verify widths (`t >= 2`); `None` derives
                `(speculative_tokens + 1,)` when `speculative_tokens >= 1`,
                otherwise `(MTP_VERIFY_TOKENS,)`.

        Returns:
            Canonical `BucketPolicy` filtered to the active decode/verify mode.

        Raises:
            ValueError: If bucket values, `verify_tokens`, or explicit
                `verify_buckets` alignment are invalid.
        """
        if verify_tokens is None:
            if speculative_tokens is not None and speculative_tokens >= 1:
                resolved_verify_tokens = (speculative_tokens + 1,)
            else:
                resolved_verify_tokens = (MTP_VERIFY_TOKENS,)
        else:
            resolved_verify_tokens = _parse_verify_tokens(verify_tokens)

        default_buckets = (
            DEFAULT_BUCKETS if profile.buckets is None else profile.buckets
        )
        default_verify_buckets = (
            DEFAULT_VERIFY_BUCKETS
            if profile.verify_buckets is None
            else profile.verify_buckets
        )
        sorted_buckets = parse_mk_buckets(
            default_buckets if buckets is None else buckets
        )
        raw_verify_buckets = _parse_bucket_list(
            default_verify_buckets if verify_buckets is None else verify_buckets,
            valid_fn=lambda b: 1 <= b <= MAX_BUCKET_ROWS,
            label="verify_buckets",
        )
        sorted_verify_buckets = tuple(
            b
            for b in raw_verify_buckets
            if any(b % width == 0 for width in resolved_verify_tokens)
        )

        is_buckets_explicit = (
            (buckets is not None) if buckets_explicit is None else buckets_explicit
        )
        is_verify_buckets_explicit = (
            (verify_buckets is not None)
            if verify_buckets_explicit is None
            else verify_buckets_explicit
        )
        must_align_all = is_verify_buckets_explicit or (
            verify_buckets is None and profile.verify_buckets is not None
        )
        if must_align_all:
            unaligned = [
                b for b in raw_verify_buckets if b not in sorted_verify_buckets
            ]
            if unaligned:
                msg = (
                    f"Megakernel: verify_buckets {unaligned} not divisible "
                    f"by verify_tokens={resolved_verify_tokens}"
                )
                raise ValueError(msg)

        if speculative_tokens is not None:
            active_width = speculative_tokens + 1
            if active_width in resolved_verify_tokens and not is_buckets_explicit:
                sorted_buckets = ()
            elif (
                active_width not in resolved_verify_tokens
                and not is_verify_buckets_explicit
            ):
                sorted_verify_buckets = ()

        return cls(
            sorted_buckets,
            sorted_verify_buckets,
            tuple(token_paddings),
            resolved_verify_tokens,
        )

    @property
    def max_bucket(self) -> int:
        """Return the largest `T=1` decode bucket, or `0` when empty."""
        return self.buckets[-1] if self.buckets else 0

    @property
    def max_verify_bucket(self) -> int:
        """Return the largest `T>=2` verify bucket, or `0` when empty."""
        return self.verify_buckets[-1] if self.verify_buckets else 0

    def verify_buckets_for(self, tokens_per_seq: int) -> tuple[int, ...]:
        """Return sorted buckets aligned to `tokens_per_seq` (`1` or verify width)."""
        if tokens_per_seq == 1:
            return self.buckets
        if tokens_per_seq in self.verify_tokens:
            return tuple(
                bucket for bucket in self.verify_buckets if bucket % tokens_per_seq == 0
            )
        return ()

    def select(
        self,
        active_rows: int,
        tokens_per_seq: int = 1,
        *,
        clamp: bool = False,
    ) -> int:
        """Select the smallest bucket for `(tokens_per_seq, active_rows)`.

        Args:
            active_rows: Total active token rows (`num_reqs * tokens_per_seq`).
            tokens_per_seq: Uniform tokens per request (`1` for decode).
            clamp: When `True`, clamp `active_rows` into `[tokens_per_seq, max]`.

        Returns:
            Selected compiled bucket row count.
        """
        valid_buckets = self.verify_buckets_for(tokens_per_seq)
        max_bucket_rows = valid_buckets[-1] if valid_buckets else 0
        target = (
            max(tokens_per_seq, min(active_rows, max_bucket_rows))
            if clamp
            else active_rows
        )
        return select_bucket(target, valid_buckets)

    def out_rows(self, b: int, m_rows: int | None = None) -> int:
        """Return runner output row count for bucket `b` under `token_paddings`.

        Args:
            b: Compiled megakernel bucket row count.
            m_rows: Explicit fallback row count when `token_paddings` is empty;
                `None` rounds `b` up to a multiple of `16`.

        Returns:
            Padded output tensor row count expected by the runner.
        """
        default_rows = (
            max(
                _MIN_OUT_ROW_MULTIPLE,
                (b + _MIN_OUT_ROW_MULTIPLE - 1)
                // _MIN_OUT_ROW_MULTIPLE
                * _MIN_OUT_ROW_MULTIPLE,
            )
            if m_rows is None
            else m_rows
        )
        return next(
            (pad for pad in self.token_paddings if pad >= b),
            default_rows,
        )

    def iter_buckets(self) -> Iterator[tuple[int, int]]:
        """Return an iterator over `(tokens_per_seq, bucket_rows)` pairs."""
        pairs: list[tuple[int, int]] = [(1, bucket) for bucket in self.buckets]
        for width in self.verify_tokens:
            pairs.extend((width, bucket) for bucket in self.verify_buckets_for(width))
        return iter(pairs)


__all__ = [
    "DEFAULT_BUCKETS",
    "DEFAULT_VERIFY_BUCKETS",
    "FALSE_STRS",
    "MAX_BUCKET_ROWS",
    "MIN_VERIFY_TOKENS",
    "MTP_SPECULATIVE_TOKENS",
    "MTP_VERIFY_TOKENS",
    "BucketPolicy",
    "parse_mk_buckets",
    "parse_mk_verify_buckets",
    "select_bucket",
]
