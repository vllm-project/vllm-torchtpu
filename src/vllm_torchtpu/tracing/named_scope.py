# SPDX-License-Identifier: Apache-2.0
"""XProf name scopes for model code.

Wraps TorchTPU's ``torch.tpu.profiler.named_scope``, which labels the TPU ops
issued inside a scope in eager mode and under ``torch.compile``. XProf shows
the labels as levels of the "Framework Name Scope" hierarchy, interleaved
with the ``nn.Module`` path of each op.

Scopes are no-ops until ``initialize_named_scopes()`` runs. Models call it in
``__init__``, as they call their other ``initialize_*`` helpers: that is after
the platform has imported TorchTPU and before any forward is traced, so no
traced frame imports a module or reads the environment. Without TorchTPU's
API (releases before 0.3.0) or with ``TPU_PROFILER_NAMED_SCOPES=0``, scopes
stay no-ops.
"""

import contextlib
import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

_scope: Callable[[str], contextlib.AbstractContextManager] | None = None


def _import_scope() -> Callable[[str], contextlib.AbstractContextManager] | None:
    try:
        import torch_tpu  # noqa: F401  # Installs the `torch.tpu` namespace.
        from torch.tpu.profiler import named_scope as scope

        return scope
    except (ImportError, AttributeError):
        pass
    try:
        from torch_tpu._internal.profiler.named_scope import named_scope as scope

        return scope
    except ImportError:
        return None


def initialize_named_scopes() -> bool:
    """Enable the scopes if TorchTPU provides them; returns whether it did.

    Call outside traced code. Repeated calls are cheap.
    """
    global _scope
    if _scope is None and envs.TPU_PROFILER_NAMED_SCOPES:
        _scope = _import_scope()
        if _scope is not None:
            logger.info_once("TorchTPU named scopes label model phases in XProf.")
    return _scope is not None


def named_scopes_enabled() -> bool:
    """Whether ``initialize_named_scopes()`` found TorchTPU's API."""
    return _scope is not None


def named_scope(name: str) -> contextlib.AbstractContextManager:
    """Label the TPU ops issued inside the context with ``name``.

    Create the scope where it is entered: a scope built outside a compiled
    region and entered inside it leaves the ops unlabeled. ``name`` must not
    contain ``@`` or ``:``; a ``/`` adds a level.
    """
    if _scope is None:
        return contextlib.nullcontext()
    return _scope(name)


def scoped(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator form of ``named_scope``; each call enters a new scope."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(function)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if _scope is None:
                return function(*args, **kwargs)
            with _scope(name):
                return function(*args, **kwargs)

        return wrapper

    return decorate
