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
"""Request-count bucketing for attention and GDN metadata tensors.

When ``TPU_METADATA_REQ_BUCKETING=1`` is enabled, ``TPUModelRunner`` keeps the
global ``max_num_seqs`` capacity and scheduler admission unchanged while sizing
per-step request metadata tensors (``seq_lens``, ``query_start_loc``,
``block_tables``, ``mamba_state_indices``, and ``mamba_prefill_ckpt_offsets``)
to the smallest configured request bucket ``R`` covering the step's active
requests and padded token bucket. Each ``R`` owns an isolated compiled wrapper
keyed via ``metadata_req_bucket_scope(R)`` in ``compute_tpu_compilation_hash``.
"""

from __future__ import annotations

import bisect
import contextlib
import threading
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

import torch

_tls = threading.local()
_STEP_COUNTS: Counter[tuple[int, int, bool]] = Counter()
_ACTIVE_MAX_SEQS_SEEN: dict[str, int] = {}


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an int, got {value!r} ({type(value).__name__})."
        )
    return int(value)


def _require_positive_int(value: Any, name: str) -> int:
    iv = _require_int(value, name)
    if iv <= 0:
        raise ValueError(f"{name} must be > 0, got {iv}.")
    return iv


@contextlib.contextmanager
def metadata_req_bucket_scope(num_reqs: int | None) -> Iterator[None]:
    """Bind the active metadata request bucket on this thread during compile."""
    if num_reqs is not None:
        num_reqs = _require_positive_int(num_reqs, "num_reqs")
    prev = getattr(_tls, "active_req_bucket", None)
    _tls.active_req_bucket = num_reqs
    try:
        yield
    finally:
        _tls.active_req_bucket = prev


def get_active_metadata_req_bucket() -> int | None:
    return getattr(_tls, "active_req_bucket", None)


def resolve_req_buckets(raw_buckets: Sequence[int], max_reqs_cap: int) -> list[int]:
    """Return sorted, deduplicated request buckets capped at ``max_reqs_cap``."""
    cap = _require_positive_int(max_reqs_cap, "max_reqs_cap")
    cleaned_set: set[int] = set()
    for raw in raw_buckets:
        b = _require_positive_int(raw, "req_bucket")
        if b < cap:
            cleaned_set.add(b)
    cleaned = sorted(cleaned_set)
    cleaned.append(cap)
    return cleaned


def _first_ge(buckets: Sequence[int], value: int) -> int:
    idx = bisect.bisect_left(buckets, value)
    if idx >= len(buckets):
        return int(buckets[-1])
    return int(buckets[idx])


def select_target_num_reqs(
    *,
    num_tokens_padded: int,
    min_token_bucket: int,
    active_num_reqs: int,
    is_decode_only: bool,
    req_buckets: Sequence[int],
    max_reqs_cap: int,
) -> int:
    """Select the request metadata bucket ``target_num_reqs`` for a chunk.

    Fails closed with ``ValueError`` if ``active_num_reqs`` is negative or
    exceeds either ``max_reqs_cap`` or ``num_tokens_padded``, guaranteeing
    ``active_num_reqs <= target_num_reqs <= max_reqs_cap``.
    """
    cap = _require_positive_int(max_reqs_cap, "max_reqs_cap")
    tok_padded = _require_positive_int(num_tokens_padded, "num_tokens_padded")
    min_tok = _require_positive_int(min_token_bucket, "min_token_bucket")
    active = _require_int(active_num_reqs, "active_num_reqs")
    if active < 0:
        raise ValueError(f"active_num_reqs must be >= 0, got {active}.")
    if active > cap:
        raise ValueError(f"active_num_reqs ({active}) exceeds max_reqs_cap ({cap}).")
    if active > tok_padded:
        raise ValueError(
            f"active_num_reqs ({active}) exceeds num_tokens_padded ({tok_padded})."
        )

    buckets = resolve_req_buckets(req_buckets, cap)
    token_req_cap = _first_ge(buckets, max(1, min(tok_padded, cap)))
    if bool(is_decode_only) and tok_padded == min_tok:
        req_bucket = _first_ge(buckets, max(1, min(active, cap)))
        return max(active, min(req_bucket, token_req_cap))
    return max(active, token_req_cap)


def build_warmup_req_groups(
    token_paddings: Sequence[int],
    req_buckets: Sequence[int],
    max_reqs_cap: int,
) -> list[tuple[int, list[int]]]:
    """Group backbone ``token_paddings`` by the ``req_cap`` wrapper they use."""
    cap = _require_positive_int(max_reqs_cap, "max_reqs_cap")
    paddings = [_require_positive_int(t, "token_padding") for t in token_paddings]
    if not paddings:
        return []
    buckets = resolve_req_buckets(req_buckets, cap)
    sorted_paddings = sorted(set(paddings))
    min_tok = sorted_paddings[0]
    min_tok_req_cap = _first_ge(buckets, max(1, min(min_tok, cap)))

    by_req: dict[int, list[int]] = {}
    for r in buckets:
        if r <= min_tok_req_cap:
            by_req.setdefault(r, []).append(min_tok)
    for tok in sorted_paddings[1:]:
        r = _first_ge(buckets, max(1, min(tok, cap)))
        if tok not in by_req.setdefault(r, []):
            by_req[r].append(tok)

    return [(r, sorted(by_req[r])) for r in sorted(by_req)]


def find_compile_wrappers(model: torch.nn.Module) -> list[Any]:
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    if isinstance(model, TorchCompileWithNoGuardsWrapper):
        return [model]
    return [
        m for m in model.modules() if isinstance(m, TorchCompileWithNoGuardsWrapper)
    ]


def capture_wrapper_states(model: torch.nn.Module) -> list[tuple[Any, dict[str, Any]]]:
    states: list[tuple[Any, dict[str, Any]]] = []
    for w in find_compile_wrappers(model):
        states.append(
            (
                w,
                {
                    "aot_compiled_fn": getattr(w, "aot_compiled_fn", None),
                    "was_aot_compile_fn_loaded_from_disk": getattr(
                        w, "was_aot_compile_fn_loaded_from_disk", False
                    ),
                    "compiled": getattr(w, "compiled", False),
                    "first_compile": getattr(w, "first_compile", False),
                    "_compiled_callable": getattr(w, "_compiled_callable", None),
                    "_compiled_bytecode": getattr(w, "_compiled_bytecode", None),
                },
            )
        )
    return states


def activate_wrapper_states(states: list[tuple[Any, dict[str, Any]]]) -> None:
    for w, st in states:
        for attr, val in st.items():
            setattr(w, attr, val)


def record_metadata_step(
    *,
    num_tokens_padded: int,
    target_num_reqs: int,
    is_decode_only: bool,
    max_num_seqs_cap: int,
) -> None:
    _STEP_COUNTS[
        (int(num_tokens_padded), int(target_num_reqs), bool(is_decode_only))
    ] += 1
    _ACTIVE_MAX_SEQS_SEEN["max_num_seqs_cap"] = int(max_num_seqs_cap)


def get_metadata_req_bucketing_stats() -> dict[str, Any]:
    return {
        "step_counts": {
            f"tok{t}_req{r}_{'dec' if d else 'mix'}": int(c)
            for (t, r, d), c in sorted(_STEP_COUNTS.items())
        },
        "max_num_seqs_cap": _ACTIVE_MAX_SEQS_SEEN.get("max_num_seqs_cap"),
    }


def reset_metadata_req_bucketing_stats() -> None:
    _STEP_COUNTS.clear()
    _ACTIVE_MAX_SEQS_SEEN.clear()
