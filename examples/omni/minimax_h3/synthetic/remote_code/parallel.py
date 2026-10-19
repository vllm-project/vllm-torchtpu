# SPDX-License-Identifier: Apache-2.0
"""Mutable parallel-state dict for the synthetic H3 video VAE.

vllm-omni's ``MiniMaxH3VideoVAE._native_parallel_state`` imports
``<remote package>.parallel`` and calls ``get_parallel_state()``; it then
``clear()``s and ``update()``s the returned dict in place, so the identity of
the dict must stay stable for the lifetime of the process.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_STATE: dict[str, Any] = {
    "group_size": 1,
    "group_rank": 0,
    "local_process_group": None,
    "sp_size": 1,
    "sp_rank": 0,
    "sp_enabled": False,
    "sp_process_group": None,
    "tp_size": 1,
    "tp_rank": 0,
}

_PARALLEL_STATE: dict[str, Any] = dict(_DEFAULT_STATE)


def get_parallel_state() -> dict[str, Any]:
    """Return the process-wide mutable parallel-state dict."""
    return _PARALLEL_STATE


def ensure_parallel_state() -> dict[str, Any]:
    """Fill any missing keys with single-rank defaults and return the dict."""
    for key, value in _DEFAULT_STATE.items():
        _PARALLEL_STATE.setdefault(key, value)
    return _PARALLEL_STATE


__all__ = ["ensure_parallel_state", "get_parallel_state"]
