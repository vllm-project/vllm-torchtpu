# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Decoupled process group provider for llm_tpu_core.

Allows distributed acceleration primitives to obtain process groups without
depending on vLLM's parallel_state.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from typing import Any, Protocol


class ProcessGroupProvider(Protocol):
    """Protocol for providing distributed process groups across model axes."""

    def get_tp_group(self) -> Any: ...
    def get_pp_group(self) -> Any: ...
    def get_pcp_group(self) -> Any: ...
    def get_dcp_group(self) -> Any: ...
    def get_ep_group(self) -> Any: ...
    def get_world_group(self) -> Any: ...


class DefaultGroupProvider:
    """Default fallback group provider when running standalone or testing."""

    def get_tp_group(self) -> Any:
        return None

    def get_pp_group(self) -> Any:
        return None

    def get_pcp_group(self) -> Any:
        return None

    def get_dcp_group(self) -> Any:
        return None

    def get_ep_group(self) -> Any:
        return None

    def get_world_group(self) -> Any:
        return None


_CURRENT_PROVIDER: ProcessGroupProvider = DefaultGroupProvider()


def set_group_provider(provider: ProcessGroupProvider) -> None:
    """Register a custom process group provider (e.g. from vLLM)."""
    global _CURRENT_PROVIDER
    _CURRENT_PROVIDER = provider


def get_group_provider() -> ProcessGroupProvider:
    """Return the active process group provider."""
    return _CURRENT_PROVIDER


def get_tp_group() -> Any:
    return _CURRENT_PROVIDER.get_tp_group()


def get_pp_group() -> Any:
    return _CURRENT_PROVIDER.get_pp_group()


def get_pcp_group() -> Any:
    return _CURRENT_PROVIDER.get_pcp_group()


def get_dcp_group() -> Any:
    return _CURRENT_PROVIDER.get_dcp_group()


def get_ep_group() -> Any:
    return _CURRENT_PROVIDER.get_ep_group()


def get_world_group() -> Any:
    return _CURRENT_PROVIDER.get_world_group()
