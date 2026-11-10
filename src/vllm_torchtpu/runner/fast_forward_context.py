# SPDX-License-Identifier: Apache-2.0
"""Cached vLLM forward context for steady decode steps.

`vllm.forward_context.set_forward_context` rebuilds the same ForwardContext on
every call: DP metadata (a CPU tensor assert, `DPMetadata.make`), the platform's
`set_additional_forward_context` (a host `.item()` / `.tolist()` plus a dict
lookup) and the dataclass itself. At decode batch 1-2 the worker runs it three
times per step (target, draft step 0, draft loop) for ~0.06 ms each, all host
time on the critical path.

`fast_set_forward_context` builds that context once per
(vllm_config, num_tokens, num_tokens_across_dp, cudagraph mode) and on later
calls installs a shallow copy carrying the new `attn_metadata`. Every other
field of the cached context is a function of the key only. Batch-size tracking
(VLLM_LOG_BATCHSIZE_INTERVAL) and every non-default argument fall back to the
stock path. Default off on this branch (the unified gates ran without it); on with
VLLM_TPU_FAST_FORWARD_CONTEXT=1, as the hostov stack (band H j-81e043bf) ran it.
"""

from __future__ import annotations

import contextlib
import copy
import os
from typing import Any

import vllm.forward_context as _fc

_ENABLED = os.environ.get("VLLM_TPU_FAST_FORWARD_CONTEXT", "0") == "1"
_CACHE: dict[tuple, Any] = {}
_MAX = 256


def _key(vllm_config: Any, num_tokens: int | None, num_tokens_across_dp: Any) -> tuple | None:
    if num_tokens_across_dp is None:
        dp = None
    elif hasattr(num_tokens_across_dp, "tolist"):
        if num_tokens_across_dp.device.type != "cpu" or num_tokens_across_dp.numel() > 64:
            return None
        dp = tuple(num_tokens_across_dp.tolist())
    else:
        return None
    return (id(vllm_config), num_tokens, dp)


@contextlib.contextmanager
def _stock(attn_metadata, vllm_config, num_tokens, num_tokens_across_dp, **kwargs):
    with _fc.set_forward_context(
        attn_metadata,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        **kwargs,
    ):
        yield


def fast_set_forward_context(
    attn_metadata: Any,
    vllm_config: Any,
    num_tokens: int | None = None,
    num_tokens_across_dp: Any = None,
    **kwargs: Any,
):
    """Drop-in for `set_forward_context(attn_metadata, vllm_config, num_tokens=...,
    num_tokens_across_dp=...)`; any other keyword argument takes the stock path."""
    if not _ENABLED or kwargs or getattr(_fc, "track_batchsize", False):
        return _stock(attn_metadata, vllm_config, num_tokens, num_tokens_across_dp, **kwargs)
    key = _key(vllm_config, num_tokens, num_tokens_across_dp)
    if key is None:
        return _stock(attn_metadata, vllm_config, num_tokens, num_tokens_across_dp)
    base = _CACHE.get(key)
    if base is None:
        return _build_and_cache(key, attn_metadata, vllm_config, num_tokens, num_tokens_across_dp)
    fc = copy.copy(base)
    fc.attn_metadata = attn_metadata
    return _fc.override_forward_context(fc)


@contextlib.contextmanager
def _build_and_cache(key, attn_metadata, vllm_config, num_tokens, num_tokens_across_dp):
    with _fc.set_forward_context(
        attn_metadata,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        fc = _fc.get_forward_context()
        if len(_CACHE) >= _MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = copy.copy(fc)
        yield
