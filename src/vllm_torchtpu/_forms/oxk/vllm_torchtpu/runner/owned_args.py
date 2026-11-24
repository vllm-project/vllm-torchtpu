# SPDX-License-Identifier: Apache-2.0
"""Owned tensors at compiled-program boundaries.

torch_tpu hands a compiled program the whole backing buffer of a view it is
given through an attribute source (vLLM's AOT forward lifts forward-context
and module attributes as graph inputs). A slice of a larger buffer then
reaches a program compiled for the slice's size as the larger buffer, and the
launch fails with ``RuntimeProgramInputMismatch ... expected s32[n] got
s32[N]`` (exos j-dc33c5ec: a ``[37, 16]`` stack row). An explicit argument
that is a view costs an ``execute_as_strided`` program instead.

`is_owned` / `ensure_owned` keep such views away from compiled programs.
`static_shape_args_owned` strips dynamo's ``mark_dynamic`` stamp without the
``a[:]`` view `eagle3._static_shape_args` uses: a ``detach()`` alias shares
the whole storage (no offset, no larger base), and a mutated argument is
returned as the caller's own object.
"""

import functools
from collections.abc import Callable
from typing import Any

import torch

# Attributes `torch._dynamo.mark_dynamic` (and its unbacked/hint variants)
# stamps onto the tensor object; any of them makes dynamo trace a dynamic dim.
_DYNAMO_STAMPS = (
    "_dynamo_dynamic_indices",
    "_dynamo_unbacked_indices",
    "_dynamo_strict_unbacked_indices",
    "_dynamo_weak_dynamic_indices",
    "_dynamo_hint_overrides",
    "_dynamo_dynamic_range",
    "_has_dynamo_dim_marking",
)


def is_owned(t: torch.Tensor) -> bool:
    """True when `t` is exactly its storage: contiguous, offset 0, and the
    storage holds no more bytes than `t` (no larger base buffer behind it)."""
    if t._base is not None or t.storage_offset() != 0 or not t.is_contiguous():
        return False
    return t.untyped_storage().nbytes() == t.numel() * t.element_size()


def ensure_owned(t: torch.Tensor) -> torch.Tensor:
    """`t` if it owns its storage, else a contiguous copy of the same values."""
    return t if is_owned(t) else t.clone(memory_format=torch.contiguous_format)


def is_stamped(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and any(
        getattr(t, name, None) for name in _DYNAMO_STAMPS
    )


def static_shape_args_owned(fn: Callable) -> Callable:
    """Call `fn` with every stamped top-level tensor argument replaced by a
    stamp-free alias that owns its whole storage.

    An owned stamped argument becomes ``t.detach()`` (same storage, so an
    in-place write lands in the caller's tensor); a stamped view becomes an
    owned copy. A result that is one of the substituted aliases is returned
    as the caller's original tensor when the alias shared its storage, so the
    caller gets back exactly the object it passed, as without the wrapper.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        back: dict[int, torch.Tensor] = {}

        def strip(a):
            if not is_stamped(a):
                return a
            if is_owned(a):
                alias = a.detach()
                back[id(alias)] = a
                return alias
            return a.clone(memory_format=torch.contiguous_format)

        new_args = [strip(a) for a in args]
        new_kwargs = {k: strip(v) for k, v in kwargs.items()}
        out = fn(*new_args, **new_kwargs)
        if not back:
            return out
        if isinstance(out, torch.Tensor):
            return back.get(id(out), out)
        if isinstance(out, tuple):
            return tuple(
                back.get(id(o), o) if isinstance(o, torch.Tensor) else o for o in out
            )
        return out

    return wrapper
