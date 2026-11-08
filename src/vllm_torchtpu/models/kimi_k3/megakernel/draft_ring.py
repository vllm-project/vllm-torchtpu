# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/draft_ring.py
"""The DSpark draft's ring computed by the prefill program.

The decoder's draft (:mod:`draft`) attends a ring of the last 2048 positions'
projected keys and values of the target's residual streams entering the aux
layers (24, 48, 72, 88, 92).
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger

from .program import kernel_modules
from .runtime_shim.bridge import (
    OpRegistrationSpec,
    hbm,
    materialize,
    register_op,
    upload,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = init_logger(__name__)

AUX_LAYERS = (24, 48, 72, 88, 92)
WEIGHT_ARRAYS = ("fc", "kv", "kvnorm", "hidden_norm")
_GIB_BYTES = 2**30
_REV_PREFIX_LEN = 8
_NUM_WEIGHT_ARRAYS = 4


class PrefillRing:
    """The prefill side of the DSpark ring."""

    def __init__(
        self,
        mesh: Mesh,
        rank: int,
        axis: str,
        checkpoint: Path,
        presharded: Path,
    ) -> None:
        """Initialize ring projector for ``mesh``, ``rank``, and ``checkpoint``."""
        (self.dspark,) = kernel_modules("dspark")
        self.config = self.dspark.DraftConfig.from_checkpoint(checkpoint)
        if tuple(self.config.aux_layers) != AUX_LAYERS:
            msg = (
                f"the prefill ring taps aux layers {AUX_LAYERS}; the draft "
                f"{checkpoint} reads {self.config.aux_layers}"
            )
            raise NotImplementedError(msg)
        self.mesh, self.rank, self.axis = mesh, rank, axis
        rev = Path(checkpoint).resolve().name[:_REV_PREFIX_LEN]
        self._directory = Path(presharded) / f"dspark-{rev}"
        self.weights: tuple[torch.Tensor, ...] = ()
        self._ops: dict[
            tuple[object, ...], Callable[..., tuple[torch.Tensor, ...]]
        ] = {}
        self._zero: tuple[torch.Tensor, torch.Tensor] | None = None

    def load_weights(self) -> None:
        """Read this rank's four draft shard arrays and gather replicated weights."""
        started = time.perf_counter()
        index = json.loads((self._directory / "index.json").read_text())
        if index.get("format") != self.dspark.KERNEL_SHARD_FORMAT:
            msg = f"{self._directory}: not a DSpark kernel shard set"
            raise ValueError(msg)
        raw = np.memmap(
            self._directory / f"rank{self.rank:02d}.bin",
            dtype=np.uint8,
            mode="r",
        )
        shards: list[torch.Tensor] = []
        for name in WEIGHT_ARRAYS:
            spec = index["arrays"][name]
            dtype = (
                np.dtype(jnp.bfloat16)
                if spec["dtype"] == "bfloat16"
                else np.dtype(spec["dtype"])
            )
            count = int(np.prod(spec["shape"]))
            offset = int(spec["offset"])
            value = np.array(raw[offset : offset + count * dtype.itemsize])
            shards.append(upload(value.view(dtype).reshape(spec["shape"])))
        cfg, axis = self.config, self.axis

        def gather(
            fc: jax.Array,
            kv: jax.Array,
            kvnorm: jax.Array,
            hidden_norm: jax.Array,
        ) -> tuple[jax.Array, ...]:
            """Gather per-rank draft ring weight shards across ``axis``."""
            out = self.dspark.ring_weights_from_shards(
                cfg,
                fc[0],
                kv[0],
                kvnorm[0],
                hidden_norm[0],
                axis,
            )
            return tuple(out[name] for name in ("fc", "kv", "k_norm", "hidden_norm"))

        op = register_op(
            "pallas::k3_prefill_draft_ring_weights",
            self.mesh,
            gather,
            (P(axis),) * _NUM_WEIGHT_ARRAYS,
            (P(),) * _NUM_WEIGHT_ARRAYS,
        )
        self.weights = tuple(op(*shards))
        materialize(self.weights[0])
        if self.rank == 0:
            gib = sum(t.numel() * t.element_size() for t in self.weights) / _GIB_BYTES
            logger.info(
                "K3 prefill: DSpark ring weights resident (%.2f GiB per rank) in "
                "%.1f s; HBM %s",
                gib,
                time.perf_counter() - started,
                hbm(self.rank),
            )

    def empty(self, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an empty ``(cache, meta)`` ring starting at token ``start``."""
        cfg = self.config
        cache = torch.zeros(
            (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
            dtype=torch.bfloat16,
        ).to("tpu")
        meta = torch.tensor([start, start], dtype=torch.int32).to("tpu")
        return cache, meta

    def zero(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a shared empty ring for unused window sequence slots."""
        if self._zero is None:
            self._zero = self.empty(0)
        return self._zero

    def update(
        self,
        blocks: torch.Tensor,
        *window_inputs: object,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Update ``rings`` from one packed prefill window's residual streams."""
        (
            prefix,
            middle,
            positions_obj,
            sequences_obj,
            ends,
            stops,
            rings_obj,
        ) = window_inputs
        positions = np.ascontiguousarray(positions_obj, np.int32)
        sequences = np.ascontiguousarray(sequences_obj, np.int32)
        rings = cast("Sequence[tuple[torch.Tensor, torch.Tensor]]", rings_obj)
        slots, window = len(rings), int(positions.size)

        def to_device(values: object) -> torch.Tensor:
            """Upload int32 array-like ``values`` to TPU."""
            return torch.from_numpy(np.ascontiguousarray(values, np.int32)).to("tpu")

        out = self._update_op(window, slots)(
            blocks,
            prefix,
            middle,
            to_device(positions),
            to_device(sequences),
            to_device(ends),
            to_device(stops),
            *self.weights,
            *(cache for cache, _ in rings),
            *(meta for _, meta in rings),
        )
        after = list(zip(out[:slots], out[2 * slots : 3 * slots], strict=True))
        at = list(zip(out[slots : 2 * slots], out[3 * slots :], strict=True))
        return after, at

    def update_op(
        self,
        window: int,
        count: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Return the cached ring-update custom op for ``window`` and ``count``."""
        return self._update_op(window, count)

    def _update_op(
        self,
        window: int,
        slots: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the cached custom op for ``window`` and ``slots``."""
        key = ("update", window, slots)
        if key not in self._ops:
            cfg, axis, dspark = self.config, self.axis, self.dspark

            def body(
                blocks: jax.Array,
                *args: jax.Array,
            ) -> tuple[jax.Array, ...]:
                """Project window residual streams into each sequence's draft ring."""
                prefix, middle, positions, sequences, ends, stops = args[:6]
                rest = args[6:]
                weights = dict(
                    zip(
                        ("fc", "kv", "k_norm", "hidden_norm"),
                        rest[:_NUM_WEIGHT_ARRAYS],
                        strict=True,
                    )
                )
                caches = rest[_NUM_WEIGHT_ARRAYS : _NUM_WEIGHT_ARRAYS + slots]
                metas = rest[_NUM_WEIGHT_ARRAYS + slots :]
                aux = jnp.stack(
                    (blocks[0, 2], blocks[0, 4], blocks[0, 6], middle[0], prefix[0])
                )
                after, at = dspark.prefill_ring_update(
                    cfg,
                    weights,
                    aux,
                    positions,
                    sequences,
                    jnp.stack([cache[0] for cache in caches]),
                    ends,
                    stops,
                    axis,
                )
                firsts = [meta[1] for meta in metas]
                return (
                    *(after[s][None] for s in range(slots)),
                    *(at[s][None] for s in range(slots)),
                    *(jnp.stack((ends[s], firsts[s])) for s in range(slots)),
                    *(jnp.stack((stops[s], firsts[s])) for s in range(slots)),
                )

            self._ops[key] = register_op(
                f"pallas::k3_prefill_draft_ring_w{window}_s{slots}",
                self.mesh,
                body,
                OpRegistrationSpec(
                    in_specs=(P(axis),) * 3
                    + (P(),) * _NUM_WEIGHT_ARRAYS
                    + (P(),) * _NUM_WEIGHT_ARRAYS
                    + (P(axis),) * slots
                    + (P(),) * slots,
                    out_specs=(P(axis),) * (2 * slots) + (P(),) * (2 * slots),
                    export_key=repr(
                        ("prefill_draft_ring", window, slots, dataclasses.astuple(cfg))
                    ),
                ),
            )
        return self._ops[key]
