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
# src/vllm_torchtpu/megakernel/kimi_k3/decode_ops.py
"""Kimi-K3 decode's device side: weights, KDA state slots, and step ops.

:class:`DecodeOps` holds this rank's decoder weights and KDA slots and builds
the ops a decode step runs: the plain step per compiled batch size (powers of
two up to ``max_num_seqs``; a step pads to the next), the single-request
DSpark verify and the batched verify. :class:`Control` is the layout of the
replicated control vector those ops read and return updated.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger

from . import config as kernel_config
from . import sampling
from .layout import CONV_HEAD_ROWS, NUM_LAYERS
from .pools import (
    DP_BATCH_SIZES,
    KDA_LAYERS,
    ROUTED_RS_BATCHES,
    TOKEN_SCATTER_BATCHES,
    dp_batch_sizes,
    dp_group_rows,
    dp_row_owners,
    group_slots,
    select_verify_kda_rows,
)
from .program import (
    TP_SIZE,
    VOCAB,
    kernel_modules,
    recurrent_state_dtype,
)
from .runtime_shim.bridge import (
    OpRegistrationSpec,
    hbm,
    kernel_mesh,
    materialize,
    register_op,
    report_load_stats,
    stream_to_device,
)
from .runtime_shim.compilation import CompileCall

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = init_logger(__name__)

AXIS = "tp"
_POWER_OF_TWO_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
_NO_HLO_CALL_BATCHES = (8, 16, 32, 64, 96, 128)
_SMALL_DP_BATCHES = (2, 4)
_DEFAULT_ROW_OWNERS = 4
_MAX_WHOLE_CACHE_TILE_BATCH = 8
_LARGE_CACHE_TILE_MIN_BATCH = 32
_LARGE_CACHE_TILE = 2
_MID_CACHE_TILE = 4
_MIN_ALIAS_EXPERT_SCALES_BATCH = 6
_CONV_CHANNELS = 3
_CONV_TAPS = 3
_HEAD_DIM = 128
_WORDS_DIVISOR = 32
_SEED_MASK = 0x7FFFFFFF

__all__ = [
    "AXIS",
    "DP_BATCH_SIZES",
    "GREEDY",
    "Control",
    "DecodeOps",
    "decode_batch_sizes",
    "decode_options",
    "dp_batch_sizes",
    "dp_group_rows",
    "dp_row_owners",
]


class _InPlaceSetTensor(Protocol):
    """Protocol for in-place ``Tensor.set_`` storage adoption."""

    def set_(self, source: torch.Tensor) -> torch.Tensor:
        """Rebind storage to ``source`` in place."""
        ...


def decode_batch_sizes(max_num_seqs: int) -> tuple[int, ...]:
    """Return powers of two below ``max_num_seqs`` and ``max_num_seqs`` itself."""
    return tuple(
        sorted({b for b in _POWER_OF_TWO_BATCHES if b < max_num_seqs} | {max_num_seqs})
    )


def decode_options(batch_size: int) -> dict[str, int | bool | str | tuple[int, ...]]:
    """Return the kernel's tuned decode recipe for ``batch_size`` rows."""
    dp = kernel_config.active().attention_dp
    scatter = dp > 1 or batch_size in TOKEN_SCATTER_BATCHES
    if batch_size <= _MAX_WHOLE_CACHE_TILE_BATCH:
        tile = batch_size
    elif batch_size >= _LARGE_CACHE_TILE_MIN_BATCH:
        tile = _LARGE_CACHE_TILE
    else:
        tile = math.gcd(_MID_CACHE_TILE, batch_size)
    owners = dp_row_owners(batch_size)
    attention_dp_opts: dict[str, int | bool | str] = (
        {
            "attention_dp": dp,
            **(
                {"sequence_parallel_size": owners}
                if owners != _DEFAULT_ROW_OWNERS
                else {}
            ),
        }
        if dp > 1
        else {}
    )
    return {
        **attention_dp_opts,
        "gate_first_kda_projection": True,
        "residual_reduction_order": "relaxed",
        "mla_cache_batch_tile_size": min(tile, batch_size),
        "alias_expert_scales": batch_size >= _MIN_ALIAS_EXPERT_SCALES_BATCH,
        "attention_token_scatter": scatter,
        "moe_token_scatter_output": scatter,
        "moe_routed_reduce_scatter": batch_size in ROUTED_RS_BATCHES,
        "sequence_parallel_pre_attention": scatter,
    }


# Sampling parameters of a greedy plan entry: temperature, top_p, top_k, seed.
GREEDY = (0.0, 1.0, 0, 0)


class Control:
    """Layout of the decoder's replicated int32 device control state."""

    FIELDS = 8
    SLOT, RESET, TOKEN, POSITION, TEMPERATURE, TOP_P, TOP_K, SEED = range(FIELDS)
    BANK = 7

    @classmethod
    def total_slots(cls, slots: int) -> int:
        """Return total persistent KDA slots required for ``slots`` regular slots."""
        return slots

    def state_slot(self, slot: jax.Array, row: jax.Array) -> jax.Array:
        """Return persistent KDA state slot index broadcast over ``row``."""
        return slot + jnp.zeros_like(row)

    def __init__(self, slots: int, rows: int, pages: int) -> None:
        """Initialize control vector offsets for ``slots``, ``rows``, and ``pages``."""
        self.slots, self.rows, self.pages = int(slots), int(rows), int(pages)
        self.plan_start = 3 * self.slots
        self.tables_start = self.plan_start + self.rows * self.FIELDS
        self.size = self.tables_start + self.rows * self.pages

    def key(self) -> tuple[int, int, int]:
        """Return structural cache key ``(slots, rows, pages)``."""
        return self.slots, self.rows, self.pages

    def plan(
        self,
        entries: Sequence[tuple[int, bool, int, int, tuple[float, float, int, int]]],
        tables: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Pack host ``entries`` and block ``tables`` into the plan slice."""
        fields = np.zeros((self.rows, self.FIELDS), np.int32)
        fields[:, self.RESET] = 1
        fields[:, self.TEMPERATURE] = np.float32(0).view(np.int32)
        fields[:, self.TOP_P] = np.float32(1).view(np.int32)
        table = np.zeros((self.rows, self.pages), np.int32)
        for r, (
            (slot, reset, tok, position, (temperature, top_p, top_k, seed)),
            row,
        ) in enumerate(zip(entries, tables, strict=True)):
            fields[r] = (
                slot,
                int(reset),
                tok if reset else 0,
                position if reset else 0,
                np.float32(temperature).view(np.int32),
                np.float32(top_p).view(np.int32),
                top_k,
                seed & _SEED_MASK,
            )
            width = min(self.pages, len(row))
            table[r, :width] = row[:width]
        return np.concatenate((fields.reshape(-1), table.reshape(-1)))

    def consumed(self, plan: np.ndarray, rows: int) -> np.ndarray:
        """Return ``plan`` after clearing reset flags of ``rows`` live rows."""
        updated = plan.copy()
        fields = updated[: self.rows * self.FIELDS].reshape(self.rows, self.FIELDS)
        live = np.zeros(self.rows, bool)
        live[:rows] = fields[:rows, self.SLOT] != 0
        fields[live, self.RESET] = 0
        fields[live, self.TOKEN] = 0
        fields[live, self.POSITION] = 0
        return updated

    def fields(
        self,
        control: jax.Array,
        batch: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Extract ``(plan, tables)`` views from ``control`` for ``batch``."""
        plan = control[self.plan_start : self.plan_start + batch * self.FIELDS].reshape(
            batch, self.FIELDS
        )
        tables = control[
            self.tables_start : self.tables_start + batch * self.pages
        ].reshape(batch, self.pages)
        return plan, tables

    def inputs(
        self,
        control: jax.Array,
        batch: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return ``(slot, token, position, row, plan, tables)`` for ``batch`` rows."""
        plan, tables = self.fields(control, batch)
        slot, reset = plan[:, self.SLOT], plan[:, self.RESET] != 0
        tok = jnp.where(reset, plan[:, self.TOKEN], control[slot])
        position = jnp.where(
            reset,
            plan[:, self.POSITION],
            control[self.slots + slot],
        )
        row = jnp.where(reset, 0, control[2 * self.slots + slot])
        return slot, tok, position, row, plan, tables

    def sampling(
        self,
        plan: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return ``(temperature, top_p, top_k, seed)`` arrays per plan row."""

        def as_float(bits: jax.Array) -> jax.Array:
            """Bitcast int32 ``bits`` back to float32."""
            return jax.lax.bitcast_convert_type(bits, jnp.float32)

        return (
            as_float(plan[:, self.TEMPERATURE]),
            as_float(plan[:, self.TOP_P]),
            plan[:, self.TOP_K],
            plan[:, self.SEED],
        )

    def advance(
        self,
        control: jax.Array,
        batch: int,
        slot: jax.Array,
        *step_state: jax.Array,
    ) -> jax.Array:
        """Record each row's next token, position, and state row in ``control``."""
        tok, position, row = step_state
        control = control.at[slot].set(tok)
        control = control.at[self.slots + slot].set(position)
        control = control.at[2 * self.slots + slot].set(row)
        resets = self.plan_start + jnp.arange(batch) * self.FIELDS + self.RESET
        return control.at[resets].set(jnp.where(slot != 0, 0, control[resets]))


class DecodeOps:
    """Decoder weights, KDA slots, and step ops over vLLM's unified pools."""

    VERIFY_ROWS = 8

    def __init__(
        self,
        presharded: Path | str,
        num_slots: int,
        batch_sizes: tuple[int, ...] = (1,),
        mamba_ssm_cache_dtype: str = "auto",
    ) -> None:
        """Initialize decoder state and op caches for ``presharded``."""
        self.recurrent_dtype: torch.dtype = getattr(
            torch,
            recurrent_state_dtype(mamba_ssm_cache_dtype),
        )
        self.presharded = Path(presharded)
        (self.paged, self.rank_loader) = kernel_modules("paged", "rank_loader")
        self.mesh, self.rank = kernel_mesh(AXIS, TP_SIZE)
        self.attention_dp = kernel_config.active().attention_dp
        if self.attention_dp > 1:
            batch_sizes = dp_batch_sizes(batch_sizes, self.attention_dp)
        self.batch_sizes = tuple(sorted(batch_sizes))
        self.num_slots = num_slots
        self.weights: dict[str, torch.Tensor] = {}
        self.names: tuple[str, ...] = ()
        self.slots: list[torch.Tensor] = []
        self.unified_pools: list[torch.Tensor] = []
        self.pool_map: tuple[int, ...] | None = None
        self.owner_table: tuple[int, ...] | None = None
        self._ops: dict[
            tuple[object, ...], Callable[..., tuple[torch.Tensor, ...]]
        ] = {}
        self._shared: dict[object, object] = {}

    def cached_op(self, key: object, build: Callable[[], object]) -> object:
        """Return or build the shared object cached under ``key``."""
        if key not in self._shared:
            self._shared[key] = build()
        return self._shared[key]

    def batch_for(self, rows: int) -> int:
        """Return the smallest compiled batch size holding ``rows`` rows."""
        for batch in self.batch_sizes:
            if batch >= rows:
                return batch
        msg = (
            f"K3 decode: {rows} rows exceed the compiled batch sizes {self.batch_sizes}"
        )
        raise ValueError(msg)

    def load_weights(self) -> float:
        """Stream this rank's decoder weights to TPU and allocate KDA state slots."""
        if self.weights:
            return 0.0
        presharded = self.presharded
        stats = stream_to_device(
            self.rank_loader.iter_decode_rank(
                presharded,
                self.rank,
                attention_dp=self.attention_dp,
            ),
            lambda _group, name, tensor: self.weights.__setitem__(name, tensor),
        )
        self.names = tuple(sorted(self.weights))
        materialize(self.weights[self.names[-1]])
        state_slots = group_slots(self.num_slots, self.attention_dp) * self.attention_dp
        conv = torch.zeros(
            (
                1,
                state_slots,
                KDA_LAYERS,
                _CONV_CHANNELS,
                _CONV_TAPS,
                CONV_HEAD_ROWS,
                _HEAD_DIM,
            ),
            dtype=torch.bfloat16,
        ).to("tpu")
        recurrent = torch.zeros(
            (1, state_slots, KDA_LAYERS, _CONV_CHANNELS, _HEAD_DIM, _HEAD_DIM),
            dtype=self.recurrent_dtype,
        ).to("tpu")
        self.slots = [conv, recurrent]
        if self.rank == 0:
            logger.info(
                "K3 decode: rank 0 weights resident (%.1f GiB, %d families) "
                "in %.1f s from %s; HBM %s",
                stats["gib"],
                len(self.names),
                stats["wall"],
                presharded,
                hbm(self.rank),
            )
        report_load_stats("decode", self.rank, stats)
        return float(stats["wall"])

    def use_unified_pools(
        self,
        pools: list[torch.Tensor],
        pool_map: list[int],
        owner_table: Sequence[int] | None = None,
    ) -> None:
        """Bind unified-pool tensors ``pools`` and layer index ``pool_map``."""
        if self._ops:
            msg = "K3 decode: bind the unified pools before building the step ops"
            raise RuntimeError(msg)
        self.unified_pools = list(pools)
        self.pool_map = tuple(int(i) for i in pool_map)
        self.owner_table = (
            None if owner_table is None else tuple(int(i) for i in owner_table)
        )
        if self.attention_dp > 1 and self.owner_table is not None:
            group = TP_SIZE // self.attention_dp
            for start in range(0, TP_SIZE, group):
                held = {o % group for o in self.owner_table[start : start + group]}
                if len(held) != group:
                    msg = (
                        "K3 decode: each DP group needs every DCP page owner "
                        f"once: {self.owner_table}"
                    )
                    raise RuntimeError(msg)

    def _operands(self) -> tuple[torch.Tensor, ...]:
        """Return flat tuple of weight, KDA slot, and unified pool tensors."""
        return (
            *(self.weights[n] for n in self.names),
            *self.slots,
            *self.unified_pools,
        )

    def _adopt(
        self,
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        pools: Sequence[torch.Tensor],
    ) -> None:
        """Update KDA slot references and rebind unified pool storages in place."""
        self.slots = [conv, recurrent]
        for old, fresh in zip(self.unified_pools, pools, strict=True):
            if fresh is not old:
                cast("_InPlaceSetTensor", old).set_(fresh)

    def _build_plain(
        self,
        batch: int,
        control: Control,
        sub_base: int,
        *,
        sampled: bool,
        masked: bool,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return cached plain decode op for ``batch`` rows."""
        key = ("owned", batch, control.key(), sub_base, sampled, masked)
        if key in self._ops:
            return self._ops[key]
        if batch not in self.batch_sizes:
            msg = (
                f"K3 decode: batch {batch} is not among "
                f"the compiled sizes {self.batch_sizes}"
            )
            raise ValueError(msg)
        paged, names = self.paged, self.names
        count = len(names)
        options = decode_options(batch)
        pool_map, owner_table = self.pool_map, self.owner_table
        pools = len(self.unified_pools)
        width = self.VERIFY_ROWS

        def body(*args: jax.Array) -> tuple[jax.Array, ...]:
            """Execute one plain paged decode step inside ``shard_map``."""
            weights = {name: args[i][0] for i, name in enumerate(names)}
            conv, recurrent = args[count][0], args[count + 1][0]
            pool = args[count + 2 : count + 2 + pools]
            state = args[count + 2 + pools]
            mask: jax.Array | None = None
            if masked:
                state, mask = (
                    state[: control.size],
                    state[control.size :].reshape(batch, -1),
                )
            slot, tok, position, row, plan, tables = control.inputs(state, batch)
            state_slot = control.state_slot(slot, row)
            logits, conv, recurrent, pool = paged.decode_paged(
                weights,
                conv,
                recurrent,
                pool,
                tok,
                position,
                tables,
                state_slot,
                batch_size=batch,
                layers=NUM_LAYERS,
                dcp_pool_map=pool_map,
                dcp_owner_table=owner_table,
                **options,
            )
            if mask is not None:
                logits = sampling.apply_grammar_mask(logits, mask)
            chosen = sampling.greedy(logits)
            if sampled:
                temperature, top_p, top_k, seed = control.sampling(plan)
                draws = sampling.sample_rows(
                    logits,
                    jnp.where(temperature > 0, temperature, 1.0),
                    top_p,
                    top_k,
                    seed,
                    position,
                )
                chosen = jnp.where(temperature > 0, draws, chosen)
            emitted = jnp.full((batch, width), -1, jnp.int32).at[:, 0].set(chosen)
            state = control.advance(state, batch, slot, chosen, position + 1, row)
            return (emitted, state, conv[None], recurrent[None], *pool)

        in_specs = (P(AXIS),) * (count + 2 + pools) + (P(),)
        out_specs = (P(), P(), P(AXIS), P(AXIS)) + (P(AXIS),) * pools
        name = (
            f"pallas::k3_decode_owned_b{batch}"
            f"{'_sampled' if sampled else ''}{'_masked' if masked else ''}"
        )
        export_key = repr(
            (
                "owned",
                batch,
                NUM_LAYERS,
                names,
                sorted(options.items()),
                pool_map,
                owner_table,
                control.key(),
                sub_base,
                sampled,
            )
            + (("masked",) if masked else ())
        )
        self._ops[key] = register_op(
            name,
            self.mesh,
            body,
            OpRegistrationSpec(
                in_specs=in_specs,
                out_specs=out_specs,
                donate_argnums=tuple(range(count, count + 2 + pools)),
                export_key=export_key,
            ),
        )
        return self._ops[key]

    def compilation_calls(
        self,
        control: Control,
        sub_base: int,
        aux_layers: tuple[int, ...],
        *,
        spec_alone: bool,
        spec_shapes: tuple[tuple[int, int], ...],
        **compile_options: object,
    ) -> list[CompileCall]:
        """Build precompilation calls for plain and speculative verify decoder ops."""
        synthetic_rates = compile_options.get("synthetic_rates")
        masked_ops = bool(compile_options.get("masked_ops", False))
        operands = self._operands()
        calls: list[CompileCall] = []

        def add(
            op: Callable[..., tuple[object, ...]],
            rows: int,
            size: int,
            *,
            sampled: bool = False,
        ) -> None:
            """Append a ``CompileCall`` for ``op`` at ``rows`` and ``size``."""
            calls.append(
                CompileCall(
                    op,
                    (*operands, torch.empty(size, dtype=torch.int32)),
                    self._compiler_options(rows),
                    priority=2 * rows + int(sampled),
                )
            )

        masked_variants = (False, True) if masked_ops else (False,)
        words = VOCAB // _WORDS_DIVISOR
        for masked in masked_variants:
            for sampled in (False, True):
                for batch in self.batch_sizes:
                    if masked:
                        calls.append(
                            CompileCall(
                                self._build_mask_pack(control, batch),
                                (
                                    torch.empty(control.size, dtype=torch.int32),
                                    torch.empty((batch, words), dtype=torch.int32),
                                ),
                                {},
                                priority=0,
                            )
                        )
                    add(
                        self._build_plain(
                            batch,
                            control,
                            sub_base,
                            sampled=sampled,
                            masked=masked,
                        ),
                        batch,
                        control.size + batch * words if masked else control.size,
                        sampled=sampled,
                    )

        def add_verify(
            requests: int,
            rows_per: int,
            *,
            single: bool = False,
        ) -> None:
            """Append metadata and verifier ``CompileCall`` entries."""
            rows = requests * rows_per
            ids_size = (
                3 * rows + 1 + rows * control.pages
                if single
                else 4 * rows + requests * control.pages
            )
            calls.append(
                CompileCall(
                    self._build_verify_metadata(control, requests),
                    (
                        torch.empty(ids_size, dtype=torch.int32),
                        torch.empty(control.size, dtype=torch.int32),
                        torch.empty(requests, dtype=torch.int32),
                    ),
                    {},
                    priority=0,
                )
            )
            rates = (
                synthetic_rates(rows_per - 1)
                if callable(synthetic_rates)
                else synthetic_rates
            )
            rates_tuple = cast("tuple[float, ...] | None", rates)
            op = (
                self._build_verify(aux_layers, control, rates_tuple)
                if single
                else self._build_verify_batched(
                    aux_layers,
                    requests,
                    rows_per,
                    control,
                    rates_tuple,
                )
            )
            add(op, rows, ids_size + requests * (control.FIELDS + 1))

        if spec_alone:
            add_verify(1, self.VERIFY_ROWS, single=True)
        for requests, spec_rows in spec_shapes:
            add_verify(requests, spec_rows)
        return calls

    @staticmethod
    def _compiler_options(rows: int) -> dict[str, str]:
        """Return XLA compiler flag overrides for ``rows`` decode batch size."""
        config = kernel_config.active()
        dp_rows = rows in _SMALL_DP_BATCHES and config.attention_dp > 1
        if (rows in _NO_HLO_CALL_BATCHES or dp_rows) and config.decode_no_hlo_calls:
            return {"xla_tpu_enable_deduplicated_calls": "DISABLED"}
        return {}

    def _build_mask_pack(
        self,
        control: Control,
        batch: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build op packing control vector and ``batch`` grammar bitmasks."""
        key = ("mask_pack", control.key(), batch)
        if key in self._ops:
            return self._ops[key]

        def body(state: jax.Array, mask: jax.Array) -> tuple[jax.Array]:
            """Concatenate ``state`` and flattened ``mask``."""
            return (jnp.concatenate((state, mask.reshape(-1))),)

        self._ops[key] = register_op(
            f"pallas::k3_mask_pack_{batch}",
            self.mesh,
            body,
            OpRegistrationSpec(
                in_specs=(P(), P()), out_specs=(P(),), export_key=repr(key)
            ),
        )
        return self._ops[key]

    def plain_step(
        self,
        state: torch.Tensor,
        step_config: tuple[int, Control, int],
        *,
        sampled: bool,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one plain decode step for ``step_config`` and adopt updated pools."""
        batch, control, sub_base = step_config
        op = self._build_plain(
            batch,
            control,
            sub_base,
            sampled=sampled,
            masked=mask is not None,
        )
        if mask is not None:
            (state,) = self._build_mask_pack(control, batch)(state, mask)
        outputs = self._run(op, batch, *self._operands(), state)
        emitted, next_state, conv, recurrent, *pools = outputs
        self._adopt(conv, recurrent, pools)
        return emitted, next_state

    @staticmethod
    def _run(
        op: Callable[..., tuple[torch.Tensor, ...]],
        rows: int,
        *args: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Invoke ``op(*args)`` with row-appropriate XLA compiler options."""
        options = DecodeOps._compiler_options(rows)
        if options:
            compiler_opts = importlib.import_module(
                "torch_tpu._internal.compiler_options"
            )
            sync_mod = importlib.import_module("torch_tpu._internal.sync")
            with compiler_opts.custom_compiler_options(options):
                outputs = op(*args)
                sync_mod.synchronize(list(outputs), wait=False)
            return outputs
        return op(*args)

    def _build_verify(
        self,
        aux_layers: tuple[int, ...],
        control: Control,
        synthetic_rates: tuple[float, ...] | None = None,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build single-request speculative verification op."""
        key = ("verify", aux_layers, control.key(), synthetic_rates)
        if key in self._ops:
            return self._ops[key]
        rows, paged, names = self.VERIFY_ROWS, self.paged, self.names
        count = len(names)
        options = decode_options(rows)
        options["defer_kda_commit"] = self.attention_dp == 1
        options.update(
            moe_routed_reduce_scatter=True,
            moe_token_scatter_output=True,
            attention_token_scatter=True,
            sequence_parallel_pre_attention=True,
            sequence_rows=rows,
            aux_hidden_layers=tuple(aux_layers),
        )
        pool_map, owner_table = self.pool_map, self.owner_table
        pools = len(self.unified_pools)

        def body(*args: jax.Array) -> tuple[jax.Array, ...]:
            """Run single-request verify step and commit accepted KDA state."""
            weights = {name: args[i][0] for i, name in enumerate(names)}
            conv, recurrent = args[count][0], args[count + 1][0]
            pool = args[count + 2 : count + 2 + pools]
            (packet,) = args[count + 2 + pools :]
            ids, plan, modes = self._unpack_verify_metadata(packet, control, 1)
            old_conv, old_recurrent = conv, recurrent
            tokens, positions, slots = (
                ids[:rows],
                ids[rows : 2 * rows],
                ids[2 * rows : 3 * rows],
            )
            state_row, table = ids[3 * rows], ids[3 * rows + 1 :]
            logits, aux, conv, recurrent, pool = paged.decode_paged(
                weights,
                conv,
                recurrent,
                pool,
                tokens,
                positions,
                table.reshape(rows, -1),
                slots,
                batch_size=rows,
                layers=NUM_LAYERS,
                dcp_pool_map=pool_map,
                dcp_owner_table=owner_table,
                state_row=state_row,
                **options,
            )
            decisions, conv, recurrent = self._finish_verify(
                control,
                (old_conv, old_recurrent, conv, recurrent),
                (logits, tokens, positions, slots),
                (plan, modes, synthetic_rates, True),
            )
            return (logits[None], aux, decisions, conv[None], recurrent[None], *pool)

        in_specs = (P(AXIS),) * (count + 2 + pools) + (P(),)
        out_specs = (P(AXIS), P(), P(), P(AXIS), P(AXIS)) + (P(AXIS),) * pools
        export_key = repr(
            (
                "verify",
                rows,
                NUM_LAYERS,
                names,
                sorted(options.items()),
                control.key(),
                synthetic_rates,
                pool_map,
                owner_table,
            )
        )
        self._ops[key] = register_op(
            f"pallas::k3_decode_verify{rows}",
            self.mesh,
            body,
            OpRegistrationSpec(
                in_specs=in_specs,
                out_specs=out_specs,
                donate_argnums=tuple(range(count, count + 2 + pools)),
                export_key=export_key,
            ),
        )
        return self._ops[key]

    def verify_packed(
        self,
        verify_inputs: tuple[torch.Tensor, Sequence[int]],
        control: Control,
        state_and_modes: tuple[torch.Tensor, torch.Tensor],
        *,
        synthetic_rates: tuple[float, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run single-request verify op, commit accepted state, and return outputs."""
        ids, aux_layers = verify_inputs
        state, modes = state_and_modes
        (packet,) = self._build_verify_metadata(control, 1)(ids, state, modes)
        op = self._build_verify(tuple(aux_layers), control, synthetic_rates)
        sharded, aux, decisions, conv, recurrent, *pools = self._run(
            op,
            self.VERIFY_ROWS,
            *self._operands(),
            packet,
        )
        self._adopt(conv, recurrent, pools)
        return sharded, aux, decisions

    def _build_verify_batched(
        self,
        aux_layers: tuple[int, ...],
        requests: int,
        rows_per: int,
        control: Control,
        synthetic_rates: tuple[float, ...] | None = None,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build batched speculative verification op for ``requests x rows_per``."""
        key = ("verify", aux_layers, requests, rows_per, control.key(), synthetic_rates)
        if key in self._ops:
            return self._ops[key]
        rows, paged, names = requests * rows_per, self.paged, self.names
        count = len(names)
        options = decode_options(rows)
        options["defer_kda_commit"] = True
        options.update(
            sequence_rows=rows_per,
            aux_hidden_layers=tuple(aux_layers),
        )
        pool_map, owner_table = self.pool_map, self.owner_table
        pools = len(self.unified_pools)
        placed = order = None
        if self.attention_dp > 1:
            placed = dp_group_rows(rows, self.attention_dp)
            order = np.argsort(placed)

        def body(*args: jax.Array) -> tuple[jax.Array, ...]:
            """Run batched verify step and commit accepted KDA states."""
            weights = {name: args[i][0] for i, name in enumerate(names)}
            conv, recurrent = args[count][0], args[count + 1][0]
            pool = args[count + 2 : count + 2 + pools]
            (packet,) = args[count + 2 + pools :]
            ids, plan, modes = self._unpack_verify_metadata(packet, control, requests)
            old_conv, old_recurrent = conv, recurrent
            tokens, positions, slots = (
                ids[:rows],
                ids[rows : 2 * rows],
                ids[2 * rows : 3 * rows],
            )
            tables = jnp.repeat(
                ids[4 * rows :].reshape(requests, -1),
                rows_per,
                axis=0,
            )
            operands = (tokens, positions, tables, slots)
            if order is not None:
                operands = tuple(operand[order] for operand in operands)
            logits, aux, conv, recurrent, pool = paged.decode_paged(
                weights,
                conv,
                recurrent,
                pool,
                *operands,
                batch_size=rows,
                layers=NUM_LAYERS,
                dcp_pool_map=pool_map,
                dcp_owner_table=owner_table,
                **options,
            )
            if placed is not None:
                logits, aux = logits[placed], aux[:, placed]
            decisions, conv, recurrent = self._finish_verify(
                control,
                (old_conv, old_recurrent, conv, recurrent),
                (logits, tokens, positions, slots),
                (plan, modes, synthetic_rates, False),
            )
            return (logits[None], aux, decisions, conv[None], recurrent[None], *pool)

        in_specs = (P(AXIS),) * (count + 2 + pools) + (P(),)
        out_specs = (P(AXIS), P(), P(), P(AXIS), P(AXIS)) + (P(AXIS),) * pools
        export_key = repr(
            (
                "verify_batched",
                requests,
                rows_per,
                NUM_LAYERS,
                names,
                sorted(options.items()),
                control.key(),
                synthetic_rates,
                pool_map,
                owner_table,
            )
        )
        self._ops[key] = register_op(
            f"pallas::k3_decode_verify{requests}x{rows_per}",
            self.mesh,
            body,
            OpRegistrationSpec(
                in_specs=in_specs,
                out_specs=out_specs,
                donate_argnums=tuple(range(count, count + 2 + pools)),
                export_key=export_key,
            ),
        )
        return self._ops[key]

    def verify_batched(
        self,
        verify_inputs: tuple[torch.Tensor, Sequence[int], tuple[int, int]],
        control: Control,
        state_and_modes: tuple[torch.Tensor, torch.Tensor],
        *,
        synthetic_rates: tuple[float, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run batched verify op, commit accepted states, and return outputs."""
        ids, aux_layers, (requests, rows_per) = verify_inputs
        state, modes = state_and_modes
        (packet,) = self._build_verify_metadata(control, requests)(ids, state, modes)
        op = self._build_verify_batched(
            tuple(aux_layers),
            requests,
            rows_per,
            control,
            synthetic_rates,
        )
        sharded, aux, decisions, conv, recurrent, *pools = self._run(
            op,
            requests * rows_per,
            *self._operands(),
            packet,
        )
        self._adopt(conv, recurrent, pools)
        return sharded, aux, decisions

    def _build_verify_metadata(
        self,
        control: Control,
        requests: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build op packing verifier ``ids``, ``plan``, and ``modes``."""
        key = ("verify_metadata", control.key(), requests)
        if key not in self._ops:

            def body(
                ids: jax.Array,
                state: jax.Array,
                modes: jax.Array,
            ) -> tuple[jax.Array]:
                """Pack verifier metadata fields for ``requests`` sequences."""
                plan, _ = control.fields(state, requests)
                return (jnp.concatenate((ids, plan.reshape(-1), modes)),)

            self._ops[key] = register_op(
                f"pallas::k3_verify_metadata_{requests}",
                self.mesh,
                body,
                OpRegistrationSpec(
                    in_specs=(P(), P(), P()), out_specs=(P(),), export_key=repr(key)
                ),
            )
        return self._ops[key]

    @staticmethod
    def _unpack_verify_metadata(
        packet: jax.Array,
        control: Control,
        requests: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Unpack ``(ids, plan, modes)`` from packed verifier ``packet``."""
        size = requests * (control.FIELDS + 1)
        return (
            packet[:-size],
            packet[-size:-requests].reshape(requests, control.FIELDS),
            packet[-requests:],
        )

    def _finish_verify(
        self,
        control: Control,
        banks: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        verify_tensors: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        plan_and_options: tuple[
            jax.Array,
            jax.Array,
            tuple[float, ...] | None,
            bool,
        ],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute acceptance decisions and commit accepted KDA post-states."""
        conv, recurrent, new_conv, new_recurrent = banks
        logits, tokens, positions, slots = verify_tensors
        plan, modes, synthetic_rates, single = plan_and_options
        requests = int(modes.shape[0])
        rows_per = int(tokens.shape[0]) // requests
        rates_arr = (
            None
            if synthetic_rates is None
            else jnp.asarray(synthetic_rates, dtype=jnp.float32)
        )
        decisions = sampling.verify_acceptance(
            logits,
            (
                tokens.reshape(requests, rows_per),
                positions.reshape(requests, rows_per)[:, 0],
                modes,
            ),
            control.sampling(plan),
            synthetic_rates=rates_arr,
            single=single,
        )
        sources, targets = select_verify_kda_rows(
            slots,
            decisions[:, 0],
            requests=requests,
            rows_per=rows_per,
            attention_dp=self.attention_dp,
            bank_slots=int(conv.shape[0]),
        )
        commit_fn = vars(self.paged)["_commit_kda_states"]
        conv, recurrent = commit_fn(
            conv,
            recurrent,
            new_conv,
            new_recurrent,
            targets,
            source_rows=sources,
        )
        return decisions, conv, recurrent
