# SPDX-License-Identifier: Apache-2.0
"""Cached vLLM forward context for steady decode steps (q38 L6 lever ``fctx``).

Ported from the Qwen3.5 PD lane: shriharim's 14bd07701 "[Runner] Cached forward
context for steady decode steps" and 13117d25c "[Runner] Fast forward context:
with the switch off, call sites use their own set_forward_context" (branch
shriharim/qwen35-pd/PR11-batch1-host-trims; source commits 5b8013baf, b2916cac6).
The Q38 port changes only the switch: instead of an import-time environment
variable, the L6 step-time lever ``fctx`` (``VLLM_TPU_L6_FAST_FWD_CTX``, read
per step through ``core/l6_ab.py``, so it can follow the in-process A/B
windows), and each copy gets fresh ``slot_mapping`` / ``additional_kwargs``
containers and its own ``DPMetadata`` object, so nothing a forward writes into
its context is seen by the next step.

`vllm.forward_context.set_forward_context` rebuilds the same ForwardContext on
every call: DP metadata (`DPMetadata.make`, and the TPU platform's
`set_additional_forward_context`, which moves the DP token counts to the TPU),
and the dataclass itself. The decode worker runs it several times per step
(connector setup, target forward, draft step 0, draft loop), all host time on
the critical path.

`fast_set_forward_context` builds that context once per
(vllm_config, num_tokens, num_tokens_across_dp) and on later calls installs a
shallow copy carrying the new `attn_metadata`. Every other field of the cached
context is a function of the key only (it is captured before the forward runs,
so per-forward counters such as `moe_layer_index` start from their initial
values). Batch-size tracking (VLLM_LOG_BATCHSIZE_INTERVAL) and every non-default
argument fall back to the stock path.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Any

import vllm.forward_context as _fc

from vllm_torchtpu.core import l6_ab as _l6_ab

_CACHE: dict[tuple, Any] = {}
_MAX = 256


def enabled() -> bool:
    """Whether call sites should use fast_set_forward_context this step. Off, they
    call their own set_forward_context exactly as before the port."""
    return _l6_ab.on("fctx")


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
    if kwargs or getattr(_fc, "track_batchsize", False):
        return _stock(attn_metadata, vllm_config, num_tokens, num_tokens_across_dp, **kwargs)
    key = _key(vllm_config, num_tokens, num_tokens_across_dp)
    if key is None:
        _l6_ab.note("fctx", "stock")
        return _stock(attn_metadata, vllm_config, num_tokens, num_tokens_across_dp)
    base = _CACHE.get(key)
    if base is None:
        _l6_ab.note("fctx", "build")
        return _build_and_cache(key, attn_metadata, vllm_config, num_tokens, num_tokens_across_dp)
    _l6_ab.note("fctx", "hit")
    return _fc.override_forward_context(_fresh(base, attn_metadata))


def _fresh(base: Any, attn_metadata: Any) -> Any:
    """A per-call copy of a cached context: its own containers and its own DPMetadata
    object (whose `local_sizes` a forward may set), so nothing written during one
    forward is seen by the next."""
    fc = copy.copy(base)
    fc.attn_metadata = attn_metadata
    fc.slot_mapping = {}
    fc.additional_kwargs = dict(base.additional_kwargs)
    if base.dp_metadata is not None:
        fc.dp_metadata = copy.copy(base.dp_metadata)
    return fc


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
        _CACHE[key] = _fresh(fc, None)
        yield
