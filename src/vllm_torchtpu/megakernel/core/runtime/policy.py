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
"""The bucket policy: the `T=1` decode and `T=4` MTP verify bucket lists,
their parsing, and bucket selection (`BucketPolicy`).

Framework-free: imports no jax, torch or vLLM (stdlib only at run time).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..profile import Profile

DEFAULT_BUCKETS: tuple[int, ...] = (8, 16, 32)
DEFAULT_VERIFY_BUCKETS: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)
MTP_VERIFY_TOKENS: int = 4
MTP_SPECULATIVE_TOKENS: int = MTP_VERIFY_TOKENS - 1
_FALSE_STRS = ("0", "false", "no", "off", "none", "")


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Megakernel: {msg}")


def _parse_bucket_list(
    val: str | Iterable[int], *, valid_fn: Callable[[int], bool], label: str
) -> tuple[int, ...]:
    if not isinstance(val, str):
        if not isinstance(val, Iterable) or isinstance(val, (bytes, bytearray)):
            raise ValueError(f"Megakernel: invalid {label}={val!r}")
        bs = list(val)
        _need(
            all(
                isinstance(b, int) and not isinstance(b, bool) and valid_fn(b)
                for b in bs
            )
            and len(set(bs)) == len(bs),
            f"invalid {label}={val!r}",
        )
        return tuple(sorted(bs))
    raw = val.strip()
    if raw.lower() in _FALSE_STRS:
        return ()
    parts = [x.strip() for x in raw.replace(":", ",").split(",") if x.strip()]
    try:
        bs = [int(x) for x in parts]
    except ValueError:
        raise ValueError(f"Megakernel: invalid {label}={val!r}") from None
    _need(
        bool(bs) and len(set(bs)) == len(bs) and all(valid_fn(b) for b in bs),
        f"invalid {label}={val!r}",
    )
    return tuple(sorted(bs))


def parse_mk_buckets(val: str | Iterable[int]) -> tuple[int, ...]:
    """Parse `T=1` decode batch buckets from a string or iterable (`'0'` disables)."""
    return _parse_bucket_list(
        val, valid_fn=lambda b: 1 <= b <= 64, label="decode buckets"
    )


def parse_mk_verify_buckets(val: str | Iterable[int]) -> tuple[int, ...]:
    """Parse `T=4` MTP verify row buckets from a string or iterable (`'0'` disables)."""
    return _parse_bucket_list(
        val, valid_fn=lambda b: 4 <= b <= 64 and b % 4 == 0, label="verify_buckets"
    )


def select_bucket(num_tokens: int, buckets: Sequence[int] = DEFAULT_BUCKETS) -> int:
    """Return the smallest compiled bucket `>= num_tokens`."""
    for b in buckets:
        if b >= num_tokens:
            return b
    raise ValueError(f"num_tokens={num_tokens} exceeds max bucket in {buckets}")


@dataclass(frozen=True)
class BucketPolicy:
    """Resolved `T=1` decode and `T=4` verify buckets and output-padding rules."""

    buckets: tuple[int, ...] = DEFAULT_BUCKETS
    verify_buckets: tuple[int, ...] = DEFAULT_VERIFY_BUCKETS
    token_paddings: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "buckets", parse_mk_buckets(self.buckets))
        object.__setattr__(
            self, "verify_buckets", parse_mk_verify_buckets(self.verify_buckets)
        )
        pads = self.token_paddings
        if isinstance(pads, (str, bytes, bytearray)) or not isinstance(pads, Iterable):
            raise ValueError(f"Megakernel: invalid token_paddings={pads!r}")
        items = list(pads)
        _need(
            all(
                isinstance(p, int) and not isinstance(p, bool) and p >= 1 for p in items
            ),
            f"invalid token_paddings={pads!r}",
        )
        object.__setattr__(self, "token_paddings", tuple(sorted(set(items))))

    @property
    def max_bucket(self) -> int:
        return max(self.buckets) if self.buckets else 0

    @property
    def max_verify_bucket(self) -> int:
        return max(self.verify_buckets) if self.verify_buckets else 0

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
    ) -> BucketPolicy:
        """Resolve `(buckets, verify_buckets)` from explicit args or `profile`."""
        def_b = DEFAULT_BUCKETS if profile.buckets is None else profile.buckets
        p_ver = profile.verify_buckets
        def_v = DEFAULT_VERIFY_BUCKETS if p_ver is None else p_ver
        b_sorted = parse_mk_buckets(def_b if buckets is None else buckets)
        v_sorted = parse_mk_verify_buckets(
            def_v if verify_buckets is None else verify_buckets
        )
        if speculative_tokens is not None:
            b_ok = buckets is not None
            v_ok = verify_buckets is not None
            b_exp = b_ok if buckets_explicit is None else buckets_explicit
            v_exp = v_ok if verify_buckets_explicit is None else verify_buckets_explicit
            if speculative_tokens == MTP_SPECULATIVE_TOKENS and not b_exp:
                b_sorted = ()
            elif speculative_tokens != MTP_SPECULATIVE_TOKENS and not v_exp:
                v_sorted = ()
        return cls(b_sorted, v_sorted, tuple(token_paddings))

    def select(
        self, active_rows: int, tokens_per_seq: int = 1, *, clamp: bool = False
    ) -> int:
        """Select the smallest bucket for `(tokens_per_seq, active_rows)`."""
        is_v = tokens_per_seq == MTP_VERIFY_TOKENS
        b_list = self.verify_buckets if is_v else self.buckets
        max_b = self.max_verify_bucket if is_v else self.max_bucket
        target = max(tokens_per_seq, min(active_rows, max_b)) if clamp else active_rows
        return select_bucket(target, b_list)

    def out_rows(self, b: int, m_rows: int | None = None) -> int:
        """Return output row count for bucket `b` given `token_paddings`."""
        eff_m = max(16, (b + 15) // 16 * 16) if m_rows is None else m_rows
        return next((p for p in self.token_paddings if p >= b), eff_m)

    def iter_buckets(self) -> Iterator[tuple[int, int]]:
        """Yield `(tokens_per_seq, bucket_t)` across decode and verify buckets."""
        for b in self.buckets:
            yield (1, b)
        for b in self.verify_buckets:
            yield (MTP_VERIFY_TOKENS, b)
