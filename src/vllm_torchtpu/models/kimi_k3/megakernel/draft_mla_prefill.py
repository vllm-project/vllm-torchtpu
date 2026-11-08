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
# src/vllm_torchtpu/megakernel/kimi_k3/draft_mla_prefill.py
"""The MLA DSpark draft's context latents computed from a prefill window.

Every prompt token the prefill runs gets, in each of the draft's five layers,
the latent ``[kv_a_layernorm(kv[:512]) | rope(kv[512:576])]`` projected from
the residual streams entering target layers 3, 24, 48, 72, and 90.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger

from .program import kernel_modules
from .runtime_shim.bridge import OpRegistrationSpec, hbm, register_op, upload

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = init_logger(__name__)

BLOCK_TAPS = {24: 2, 48: 4, 72: 6}
EXPECTED_AUX_LAYERS = (3, 24, 48, 72, 90)
_REV_PREFIX_LEN = 8
_ROPE_POS_SLACK = 96
_HOST_COUNT = 4
_LANES_PER_HOST = 8
_LATENT_PAD = 64
_SHARDED_LATENT_INPUTS = 8


def _set_tensor_in_place(target: torch.Tensor, source: torch.Tensor) -> None:
    """Replace ``target`` storage with ``source`` in place."""
    attr_name = "set_"
    getattr(target, attr_name)(source)


class PrefillDraftLatents:
    """Project and write MLA DSpark draft context latents during prefill."""

    def __init__(
        self,
        mesh: Mesh,
        rank: int,
        axis: str,
        *config_args: object,
    ) -> None:
        """Initialize prefill draft latent projector for ``mesh`` and ``rank``."""
        checkpoint_obj, presharded_obj, max_pos_obj = config_args
        checkpoint = Path(str(checkpoint_obj))
        presharded = Path(str(presharded_obj))
        max_position = int(str(max_pos_obj))
        (self.mla,) = kernel_modules("dspark_mla")
        self.mesh, self.rank, self.axis = mesh, rank, axis
        self.config = self.mla.MlaDraftConfig.from_checkpoint(checkpoint)
        if tuple(self.config.aux_layers) != EXPECTED_AUX_LAYERS:
            msg = (
                "the K3 prefill taps aux layers 3, 24, 48, 72, 90, not "
                f"{self.config.aux_layers}"
            )
            raise NotImplementedError(msg)
        rev = checkpoint.resolve().name[:_REV_PREFIX_LEN]
        self.directory = presharded / f"dspark-{rev}"
        self.max_position = max_position
        self.weights: dict[str, torch.Tensor] = {}
        self.pools: list[torch.Tensor] = []
        self._ops: dict[
            tuple[object, ...], Callable[..., tuple[torch.Tensor, ...]]
        ] = {}

    def load_weights(self) -> None:
        """Load per-rank MLA draft shards and gather replicated context weights."""
        cfg = self.config
        shards = self.mla.load_rank_shards(self.directory, self.rank)
        lanes = self.mla.LANES
        self.weights["kv_a"] = upload(
            np.ascontiguousarray(shards["qkv_a"][:, :, cfg.q_lora : cfg.q_lora + lanes])
        )
        self.weights["kv_norm"] = upload(np.ascontiguousarray(shards["kv_norm"]))
        self.weights["context_norm"] = upload(
            np.ascontiguousarray(shards["context_norm"])
        )
        self.weights["rope"] = upload(
            self.mla.rope_table(cfg, self.max_position + _ROPE_POS_SLACK)
        )
        fc = upload(np.ascontiguousarray(shards["fc"]))
        chunk, fc_in = self.mla.CHUNK, self.mla.FC_IN
        axis = self.axis

        def gather(fc_shard: jax.Array) -> tuple[jax.Array]:
            """All-gather ``fc_shard`` across ``axis`` into full projection matrix."""
            full = jax.lax.all_gather(fc_shard[0], axis)
            full = (
                full.reshape(_HOST_COUNT, _LANES_PER_HOST, fc_in, chunk)
                .transpose(0, 2, 1, 3)
                .reshape(_HOST_COUNT * fc_in, _LANES_PER_HOST * chunk)
            )
            return (full[None],)

        op = register_op(
            "pallas::k3_dspark_mla_gather_context_proj",
            self.mesh,
            gather,
            OpRegistrationSpec(
                in_specs=(P(axis),),
                out_specs=(P(axis),),
                export_key=repr(("dspark_mla_gather_fc", fc_in, chunk)),
            ),
        )
        (self.weights["context_proj"],) = op(fc)
        if self.rank == 0:
            logger.info(
                "K3 prefill: MLA DSpark context weights resident; HBM %s",
                hbm(self.rank),
            )

    def bind(self, pools: Sequence[torch.Tensor]) -> None:
        """Bind the draft's KV cache layer pool tensors."""
        if len(pools) != self.config.layers:
            msg = (
                f"K3 prefill: the MLA DSpark draft needs {self.config.layers} draft KV "
                f"cache layers, found {len(pools)}"
            )
            raise RuntimeError(msg)
        self.pools = list(pools)

    def copy_blocks(self, pairs: Sequence[tuple[int, int]], per_call: int) -> None:
        """Execute copy-on-write block copies across all draft layer pools."""
        for first in range(0, len(pairs), per_call):
            chunk = list(pairs[first : first + per_call])
            chunk += [(0, 0)] * (per_call - len(chunk))
            source = torch.tensor([a for a, _ in chunk], dtype=torch.int32).to("tpu")
            destination = torch.tensor([b for _, b in chunk], dtype=torch.int32)
            updated = self._copy_op(per_call)(
                *self.pools,
                source,
                destination.to("tpu"),
            )
            for old, fresh in zip(self.pools, updated, strict=True):
                if fresh is not old:
                    _set_tensor_in_place(old, fresh)

    def _copy_op(
        self,
        per_call: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the cached block-copy custom op for ``per_call``."""
        key = ("copy", per_call)
        if key in self._ops:
            return self._ops[key]
        layers = self.config.layers

        def body(*args: jax.Array) -> tuple[jax.Array, ...]:
            """Copy ``source`` blocks to ``destination`` blocks in each pool."""
            pools, source, destination = (
                args[:layers],
                args[layers],
                args[layers + 1],
            )
            return tuple(pool.at[destination].set(pool[source]) for pool in pools)

        self._ops[key] = register_op(
            f"pallas::k3_dspark_mla_prefill_copy_blocks_{per_call}",
            self.mesh,
            body,
            OpRegistrationSpec(
                in_specs=(P(self.axis),) * layers + (P(), P()),
                out_specs=(P(self.axis),) * layers,
                donate_argnums=tuple(range(layers)),
                export_key=repr(("dspark_mla_prefill_copy_blocks", layers, per_call)),
            ),
        )
        return self._ops[key]

    def _latents_op(
        self,
        window: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the cached latent-projection op for ``window``."""
        key = ("latents", window)
        if key in self._ops:
            return self._ops[key]
        mla, cfg, axis = self.mla, self.config, self.axis
        layers = cfg.layers

        def body(
            blocks: jax.Array,
            *args: jax.Array,
        ) -> tuple[jax.Array]:
            """Project auxiliary residual streams into per-layer MLA latents."""
            (
                tap3,
                tap90,
                context_proj,
                kv_a,
                kv_norm,
                context_norm,
                rope,
                positions,
                inverse,
            ) = args
            rank = jax.lax.axis_index(axis)
            local = blocks.shape[2]
            own = jnp.maximum(
                jax.lax.dynamic_slice_in_dim(positions, rank * local, local),
                0,
            )
            streams = {
                3: tap3[0],
                90: tap90[0],
                **{layer: blocks[0, b] for layer, b in BLOCK_TAPS.items()},
            }
            features = jnp.concatenate(
                [streams[layer] for layer in cfg.aux_layers],
                axis=-1,
            )
            context = jnp.dot(
                features,
                context_proj[0],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            context = mla.rms_norm(context, context_norm[0][0], cfg.eps)
            table = rope[0][own]
            cos, sin = table[:, : cfg.rope : 2], table[:, cfg.rope :: 2]
            rows = []
            for layer in range(layers):
                kv = jnp.dot(
                    context,
                    kv_a[0, layer],
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                c_kv = mla.rms_norm(
                    kv[:, : cfg.kv_lora],
                    kv_norm[0, layer, 0, : cfg.kv_lora],
                    cfg.eps,
                )
                k_pe = mla.apply_rope(kv[:, cfg.kv_lora : cfg.latent], cos, sin)
                rows.append(jnp.concatenate((c_kv, k_pe), axis=-1))
            latents = jnp.stack(rows)
            latents = jax.lax.all_gather(latents, axis, axis=1, tiled=True)
            return (jnp.take(latents, inverse, axis=1)[None],)

        self._ops[key] = register_op(
            f"pallas::k3_dspark_mla_prefill_latents_w{window}",
            self.mesh,
            body,
            OpRegistrationSpec(
                in_specs=(P(axis),) * _SHARDED_LATENT_INPUTS + (P(), P()),
                out_specs=(P(axis),),
                export_key=repr(
                    (
                        "dspark_mla_prefill_latents",
                        window,
                        layers,
                        cfg.aux_layers,
                        self.max_position,
                    )
                ),
            ),
        )
        return self._ops[key]

    def latents(
        self,
        window: int,
        *window_inputs: object,
    ) -> torch.Tensor:
        """Compute the window's draft latents ``[1, layers, W, 576]``."""
        blocks, tap3, tap90, positions_obj, order_obj = window_inputs
        positions = np.asarray(positions_obj, dtype=np.int32)
        order = np.asarray(order_obj, dtype=np.int32)
        inverse = np.argsort(order).astype(np.int32)
        (latents_out,) = self._latents_op(window)(
            blocks,
            tap3,
            tap90,
            *(
                self.weights[n]
                for n in (
                    "context_proj",
                    "kv_a",
                    "kv_norm",
                    "context_norm",
                    "rope",
                )
            ),
            torch.from_numpy(
                np.ascontiguousarray(positions[order]).astype(np.int32)
            ).to("tpu"),
            torch.from_numpy(inverse).to("tpu"),
        )
        return latents_out

    def _write_op(
        self,
        window: int,
        block_shape: tuple[int, ...],
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the cached pool-write custom op for ``window``."""
        key = ("write", window, block_shape)
        if key in self._ops:
            return self._ops[key]
        layers = self.config.layers
        pages = window // (block_shape[0] * block_shape[1])

        def body(
            latents: jax.Array,
            table: jax.Array,
            *pools: jax.Array,
        ) -> tuple[jax.Array, ...]:
            """Write window latent pages into each draft layer pool."""
            table_row, latents_row, pool_list = table[0], latents[0], list(pools)
            for layer in range(layers):
                rows = (
                    jnp.pad(latents_row[layer], ((0, 0), (0, _LATENT_PAD)))
                    .astype(pool_list[layer].dtype)
                    .reshape(pages, *block_shape)
                )

                def write_page(
                    page_idx: int,
                    pool: jax.Array,
                    *,
                    layer_rows: jax.Array = rows,
                ) -> jax.Array:
                    """Write ``page_idx`` of ``layer_rows`` into ``pool``."""
                    return jax.lax.dynamic_update_index_in_dim(
                        pool,
                        jax.lax.dynamic_index_in_dim(
                            layer_rows,
                            page_idx,
                            keepdims=False,
                        ),
                        table_row[page_idx],
                        axis=0,
                    )

                pool_list[layer] = jax.lax.fori_loop(
                    0,
                    pages,
                    write_page,
                    pool_list[layer],
                )
            return tuple(pool_list)

        self._ops[key] = register_op(
            f"pallas::k3_dspark_mla_prefill_write_w{window}",
            self.mesh,
            body,
            OpRegistrationSpec(
                in_specs=(P(self.axis),) * (2 + layers),
                out_specs=(P(self.axis),) * layers,
                donate_argnums=tuple(range(2, 2 + layers)),
                export_key=repr(
                    ("dspark_mla_prefill_write", window, block_shape, layers)
                ),
            ),
        )
        return self._ops[key]

    def write(
        self,
        latents: torch.Tensor,
        table: torch.Tensor,
        block_shape: tuple[int, ...],
    ) -> None:
        """Write ``latents`` into the draft pools using ``table``."""
        window = int(latents.shape[2])
        updated = self._write_op(window, tuple(block_shape))(
            latents,
            table,
            *self.pools,
        )
        for old, fresh in zip(self.pools, updated, strict=True):
            if fresh is not old:
                _set_tensor_in_place(old, fresh)
