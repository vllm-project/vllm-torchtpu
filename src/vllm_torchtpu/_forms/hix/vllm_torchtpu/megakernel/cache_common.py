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
"""Shared helpers of the pooled cache steps (argument checks, pool I/O choice)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .pool.config import PoolBackend, cache_exec
from .spec import ModelSpec


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ModeTag:
    """Zero-leaf pytree wrapper for string modes like `'from_ckpt'` in `jax.jit`."""

    mode: str

    def tree_flatten(self) -> tuple[tuple[()], str]:
        return (), self.mode

    @classmethod
    def tree_unflatten(cls, aux_data: str, children: tuple[()]) -> ModeTag:
        del children
        return cls(aux_data)


def wrap_active_seqs(active_seqs: Any) -> Any:
    if isinstance(active_seqs, str):
        if active_seqs not in ("from_ckpt", "all"):
            raise ValueError(
                f"Unsupported active_seqs mode {active_seqs!r}; "
                "expected 'from_ckpt', 'all', None, or an int32/bool mask array"
            )
        return ModeTag(active_seqs)
    return active_seqs


def unwrap_active_seqs(active_seqs: Any) -> Any:
    if isinstance(active_seqs, ModeTag):
        return active_seqs.mode
    return active_seqs


def resolve_active_seqs_mode(
    active_seqs: Any,
    *,
    default_mode: str = "from_ckpt",
) -> tuple[str, jax.Array | None]:
    if active_seqs is None:
        return default_mode, None
    if isinstance(active_seqs, str):
        if active_seqs in ("from_ckpt", "all"):
            return active_seqs, None
        raise ValueError(
            f"Unsupported active_seqs mode {active_seqs!r}; "
            "expected 'from_ckpt', 'all', None, or an int32/bool mask array"
        )
    return "ref", jnp.asarray(active_seqs, dtype=jnp.int32)


def resolve_model_spec(g_or_spec: Any) -> ModelSpec:
    if isinstance(g_or_spec, ModelSpec):
        return g_or_spec
    if hasattr(g_or_spec, "to_model_spec"):
        return g_or_spec.to_model_spec()
    raise TypeError(
        "Expected a ModelSpec or a geometry config implementing .to_model_spec(), "
        f"got {type(g_or_spec).__name__}"
    )


def require_pool_backend(model_spec: ModelSpec) -> PoolBackend:
    if model_spec.cache_backend is None:
        raise ValueError(
            f"ModelSpec {model_spec.name!r} does not configure a cache_backend"
        )
    return model_spec.cache_backend


def validate_page_table_shape(page_table: Any, ex: ExecSpec) -> None:
    ns, r = ex.NS, ex.R
    if page_table.ndim == 2:
        if page_table.shape[0] not in (ns, r * ns) or page_table.shape[1] < 1:
            raise ValueError(
                f"page_table has shape {page_table.shape}, expected "
                f"({ns}, >=1) or ({r * ns}, >=1)"
            )
    elif page_table.ndim == 3:
        if page_table.shape[:2] not in ((1, ns), (r, ns)) or page_table.shape[2] < 1:
            raise ValueError(
                f"page_table has shape {page_table.shape}, expected ({r}, {ns}, >=1)"
            )
    else:
        raise ValueError(f"page_table must be 2D or 3D, got shape {page_table.shape}")


class PooledStepFn:
    """Callable step function that unpacks as `(step_fn, info)` and has `.lower()`."""

    def __init__(self, call_fn: Any, lower_fn: Any, info: dict[str, Any]):
        self._call_fn = call_fn
        self._lower_fn = lower_fn
        self.info = info

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_fn(*args, **kwargs)

    def lower(self, *args: Any, **kwargs: Any) -> Any:
        return self._lower_fn(*args, **kwargs)

    def __iter__(self) -> Any:
        yield self
        yield self.info


def should_use_pool_gather(
    ms: ModelSpec,
    pb: PoolBackend,
    ex: ExecSpec,
    *,
    use_fused: bool,
) -> bool:
    """Return True when non-fused verify should gather GDN states in-kernel (`GPR`)."""
    if use_fused or cache_exec(ex).pool_gather in (False, "split"):
        return False
    if not pb.supports_pool_gather or not pb.has_recurrent_pool:
        if cache_exec(ex).pool_gather is True:
            raise MegakernelConfigError(
                f"ModelSpec {ms.name!r} cache_backend does not support pool_gather"
            )
        return False
    if pb.can_pool_gather is not None:
        ok = bool(pb.can_pool_gather(ex))
        if cache_exec(ex).pool_gather is True and not ok:
            raise MegakernelConfigError(
                f"pool_gather=True is not supported for ExecSpec(B={ex.B}, T={ex.T}, "
                f"page_size={cache_exec(ex).page_size})"
            )
        return ok
    return bool(cache_exec(ex).pool_gather)


def require_fused_donation(ms: ModelSpec, use_fused: bool, donate_pools: bool) -> None:
    """Fail closed: fused pool I/O is only correct with outer pool donation.

    On TPU the fused megakernel writes the pools through unaliased HBM DMAs and
    returns the donated inputs as outputs (see `cache_experimental`). Without
    `donate_pools=True` XLA may hand the kernel a copy, silently dropping writes.
    """
    if use_fused and not donate_pools:
        raise MegakernelConfigError(
            f"fused_pool_io for ModelSpec {ms.name!r} requires donate_pools=True "
            "(the fused kernel relies on outer buffer donation for its pool writes)"
        )


__all__ = [
    "ModeTag",
    "PooledStepFn",
    "require_fused_donation",
    "require_pool_backend",
    "resolve_active_seqs_mode",
    "resolve_model_spec",
    "should_use_pool_gather",
    "unwrap_active_seqs",
    "validate_page_table_shape",
    "wrap_active_seqs",
]
