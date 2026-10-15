# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 decode's device side: the weights, the KDA state slots and the
step ops over vLLM's unified pools.

:class:`DecodeOps` holds this rank's decoder weights and KDA slots and builds
the ops a decode step runs: the plain step per compiled batch size (powers of
two up to ``max_num_seqs``; a step pads to the next), the single-request
DSpark verify and the batched verify. :class:`Control` is the layout of the
replicated control vector those ops read and return updated. Which op a step
runs, and the requests' state between steps, are :class:`KimiDecoder`'s
(``decode``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.bridge import (
    hbm,
    kernel_mesh,
    materialize,
    register_op,
    report_load_stats,
    stream_to_device,
)
from vllm_torchtpu.megakernel.kimi_k3 import config as kernel_config
from vllm_torchtpu.megakernel.kimi_k3.layout import CONV_HEAD_ROWS, NUM_LAYERS
from vllm_torchtpu.megakernel.kimi_k3.pools import (
    KDA_LAYERS,
    group_slots,
)
from vllm_torchtpu.megakernel.kimi_k3.program import TP_SIZE, VOCAB, kernel_modules

logger = init_logger(__name__)

AXIS = "tp"


def decode_batch_sizes(max_num_seqs: int) -> tuple[int, ...]:
    """The powers of two below ``--max-num-seqs``, and itself."""
    return tuple(
        sorted(
            {b for b in (1, 2, 4, 8, 16, 32, 64, 128) if b < max_num_seqs}
            | {max_num_seqs}
        )
    )


# Attention DP's compiled plain-decode batches (DP row ownership).
DP_BATCH_SIZES = (2, 4, 8, 16, 32, 64, 96, 128)


def dp_batch_sizes(requested: tuple[int, ...], attention_dp: int) -> tuple[int, ...]:
    """Attention DP's compiled sizes for ``requested`` batches: the
    ``DP_BATCH_SIZES`` (B128 with DP2 only) from the one holding the smallest
    requested batch to the one holding the largest; a step pads to the next."""
    sizes = tuple(b for b in DP_BATCH_SIZES if b <= 96 or attention_dp == 2)
    first = sum(b < min(requested) for b in sizes)
    last = sum(b < max(requested) for b in sizes)
    return sizes[first : last + 1]


def dp_row_owners(batch_size: int) -> int:
    """Attention DP's local row owners: the four rank pairs, or at B2 and B4
    (one or two rows per DP group) one: every rank holds every row."""
    return 1 if batch_size <= 4 else 4


def decode_options(batch_size: int) -> dict:
    """The kernel's tuned decode recipe for ``batch_size`` rows: from B8,
    rows are token-scattered over sequence-parallel owners with the routed
    reduce-scatter (attention DP's B2 and B4 too, to one owner, without the
    routed reduce-scatter); the MLA cache tile is the whole batch up to B8, 4
    below B32, 2 from B32."""
    dp = kernel_config.active().attention_dp
    scatter = dp > 1 or batch_size in (8, 12, 16, 20, 24, 28, 32, 64, 96, 128)
    tile = (
        batch_size
        if batch_size <= 8
        else 2
        if batch_size >= 32
        else math.gcd(4, batch_size)
    )
    owners = dp_row_owners(batch_size)
    attention_dp = (
        {
            "attention_dp": dp,
            "attention_dp_ownership": True,
            "attention_state_pipeline": True,
            "dp_moe_gather": "split",
            **({"sequence_parallel_size": owners} if owners != 4 else {}),
        }
        if dp > 1
        else {}
    )
    return {
        **attention_dp,
        "gate_first_kda_projection": True,
        "residual_reduction_order": "relaxed",
        "mla_cache_batch_tile_size": min(tile, batch_size),
        "alias_expert_scales": batch_size >= 6,
        "attention_token_scatter": scatter,
        "moe_token_scatter_output": scatter,
        "moe_routed_reduce_scatter": batch_size
        in (6, 8, 12, 16, 20, 24, 28, 32, 64, 96, 128),
        "sequence_parallel_pre_attention": scatter,
    }


class DecodeOps:
    """Decoder weights, KDA slots and the step ops over vLLM's unified pools.

    ``slots`` are the KDA state slots ``[conv, recurrent]``; MLA layer ``m``'s
    latent pages are in ``unified_pools[pool_map[m]]`` (blocks per
    ``layout.cache_layout()``), block-table entry ``j`` holding each rank's
    page of logical block ``j``. Plain decode donates pools and slots. A
    verifier donates the pools but keeps the input state banks alive until the
    post-acceptance commit donates and re-adopts them.
    """

    VERIFY_ROWS = 8  # speculative verify: the anchor and 7 drafts of one sequence

    def __init__(
        self,
        presharded: Path,
        num_slots: int,
        batch_sizes: tuple[int, ...] = (1,),
    ) -> None:
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
        self.unified_pools: list[torch.Tensor] = []  # bound by use_unified_pools
        self.pool_map: tuple[int, ...] | None = None  # MLA layer -> unified pool index
        # Mesh position -> page owner (vLLM TP rank).
        self.owner_table: tuple[int, ...] | None = None
        self._ops: dict[Any, Any] = {}  # the step ops (over the bound pools)
        self._shared: dict[Any, Any] = {}  # cached_op's

    def cached_op(self, key: Any, build: Callable[[], Any]) -> Any:
        """The op (or op-registering object) kept under ``key``, built with
        ``build()`` the first time. Torch op names are process-global -- a
        second registration replaces the first -- so decoders sharing these
        ops share their small ops too."""
        if key not in self._shared:
            self._shared[key] = build()
        return self._shared[key]

    def batch_for(self, rows: int) -> int:
        """The smallest compiled batch size holding ``rows`` rows."""
        for batch in self.batch_sizes:
            if batch >= rows:
                return batch
        raise ValueError(
            f"K3 decode: {rows} rows exceed the compiled batch sizes {self.batch_sizes}"
        )

    def load_weights(self) -> float:
        """Stream this rank's decoder weights to the device and allocate the
        KDA slots (once: decoders sharing these ops load them once)."""
        if self.weights:
            return 0.0
        presharded = self.presharded
        stats = stream_to_device(
            self.rank_loader.iter_decode_rank(
                presharded,
                self.rank,
                split_experts=True,
                attention_dp=self.attention_dp,
            ),
            lambda _group, name, tensor: self.weights.__setitem__(name, tensor),
        )
        self.names = tuple(sorted(self.weights))
        materialize(self.weights[self.names[-1]])
        # Attention DP: bank entry s + k * n holds held shard k's heads of its
        # group's slot s (pools.group_slots).
        state_slots = group_slots(self.num_slots, self.attention_dp) * self.attention_dp
        conv = torch.zeros(
            (1, state_slots, KDA_LAYERS, 3, 3, CONV_HEAD_ROWS, 128),
            dtype=torch.bfloat16,
        ).to("tpu")
        recurrent = torch.zeros(
            (1, state_slots, KDA_LAYERS, 3, 128, 128), dtype=torch.float32
        ).to("tpu")
        self.slots = [conv, recurrent]
        if self.rank == 0:
            logger.info(
                "K3 decode: rank 0 weights resident (%.1f GiB, %d families) "
                "in %.1f s from %s; "
                "HBM %s",
                stats["gib"],
                len(self.names),
                stats["wall"],
                presharded,
                hbm(self.rank),
            )
        report_load_stats("decode", self.rank, stats)
        return stats["wall"]

    def use_unified_pools(
        self,
        pools: list[torch.Tensor],
        pool_map: list[int],
        owner_table: Sequence[int] | None = None,
    ) -> None:
        """Decode from vLLM's unified-pool tensors: MLA layer ``m`` is in
        ``pools[pool_map[m]]``. ``owner_table[p]`` is the page-owner index
        of mesh position ``p`` (default: the mesh position): global page
        ``g`` is read and committed by the rank whose owner index is
        ``g % 32``."""
        if self._ops:
            raise RuntimeError(
                "K3 decode: bind the unified pools before building the step ops"
            )
        self.unified_pools, self.pool_map = list(pools), tuple(int(i) for i in pool_map)
        self.owner_table = (
            None if owner_table is None else tuple(int(i) for i in owner_table)
        )
        if self.attention_dp > 1 and self.owner_table is not None:
            group = TP_SIZE // self.attention_dp
            for start in range(0, TP_SIZE, group):
                held = {o % group for o in self.owner_table[start : start + group]}
                if len(held) != group:
                    raise RuntimeError(
                        "K3 decode: each DP group needs every DCP page owner "
                        f"once: {self.owner_table}"
                    )

    def _operands(self) -> tuple[torch.Tensor, ...]:
        return (
            *(self.weights[n] for n in self.names),
            *self.slots,
            *self.unified_pools,
        )

    def _adopt(self, conv, recurrent, pools) -> None:
        self.slots = [conv, recurrent]
        for old, fresh in zip(self.unified_pools, pools, strict=True):
            if fresh is not old:
                old.set_(fresh)  # keep vLLM's and the connector's tensor identity

    def _build_plain(
        self, batch: int, control: Any, sub_base: int, sampled: bool, masked: bool
    ) -> Any:
        """The plain step for ``batch`` rows: the rows' tokens, positions and
        KDA state slots from the control (:class:`Control`), the decoder, the
        next tokens chosen in the op (greedy; with ``sampled`` rows with a
        temperature draw theirs), and the control advanced. With ``masked``
        the operand is the control and the rows' grammar bitmasks packed by
        :meth:`_build_mask_pack` (one operand: Torch ops take at most 64),
        and the op chooses among the allowed tokens (xgrammar's bitmask: bit
        ``j`` of word ``w`` is token ``32 * w + j``'s allowed-ness). Returns
        ``(tokens [batch, VERIFY_ROWS] (column 0; the rest -1), control,
        slots, pools)``."""
        key = ("owned", batch, control.key(), sub_base, sampled, masked)
        if key in self._ops:
            return self._ops[key]
        if batch not in self.batch_sizes:
            raise ValueError(
                f"K3 decode: batch {batch} is not among "
                f"the compiled sizes {self.batch_sizes}"
            )
        from vllm_torchtpu.megakernel.kimi_k3 import sampling

        paged, names = self.paged, self.names
        count = len(names)
        options = decode_options(batch)
        pool_map, owner_table = self.pool_map, self.owner_table
        pools = len(self.unified_pools)
        width = self.VERIFY_ROWS

        def body(*args):
            weights = {name: args[i][0] for i, name in enumerate(names)}
            conv, recurrent = args[count][0], args[count + 1][0]
            pool = args[count + 2 : count + 2 + pools]
            state = args[count + 2 + pools]
            mask = None
            if masked:
                state, mask = (
                    state[: control.size],
                    state[control.size :].reshape(batch, -1),
                )
            slot, token, position, row, plan, tables = control.inputs(state, batch)
            state_slot = control.state_slot(slot, row)
            logits, conv, recurrent, pool = paged.decode_paged(
                weights,
                conv,
                recurrent,
                pool,
                token,
                position,
                tables,
                state_slot,
                batch_size=batch,
                layers=NUM_LAYERS,
                dcp_pool_map=pool_map,
                dcp_owner_table=owner_table,
                **options,
            )
            if masked:
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
            in_specs,
            out_specs,
            donate_argnums=tuple(range(count, count + 2 + pools)),
            export_key=export_key,
        )
        return self._ops[key]

    def compilation_calls(
        self,
        control: Any,
        sub_base: int,
        aux_layers: tuple[int, ...],
        spec_alone: bool,
        spec_shapes: tuple[tuple[int, int], ...],
        synthetic_rates=None,
        masked_ops: bool = False,
    ) -> list:
        """The configured primary decoder graphs, using shape-only operands.
        ``synthetic_rates``: the rates, or ``drafts -> rates`` (per draft
        width), the verify ops are built with. ``masked_ops``: also compile
        the grammar-masked plain steps (structured outputs; with DSpark, the
        steps where a structured request decodes)."""
        from vllm_torchtpu.megakernel.compilation import CompileCall

        operands = self._operands()
        calls = []

        def add(op, rows, size, sampled=False):
            calls.append(
                CompileCall(
                    op,
                    (*operands, torch.empty(size, dtype=torch.int32)),
                    self._compiler_options(rows),
                    priority=2 * rows + sampled,
                )
            )

        masked_variants = (False, True) if masked_ops else (False,)
        for masked in masked_variants:
            for sampled in (False, True):
                for batch in self.batch_sizes:
                    if masked:
                        words = VOCAB // 32
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
                        self._build_plain(batch, control, sub_base, sampled, masked),
                        batch,
                        control.size + batch * (VOCAB // 32)
                        if masked
                        else control.size,
                        sampled,
                    )

        def add_verify(requests, rows_per, single=False):
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
            op = (
                self._build_verify(aux_layers, control, rates)
                if single
                else self._build_verify_batched(
                    aux_layers, requests, rows_per, control, rates
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
        config = kernel_config.active()
        dp_rows = rows in (2, 4) and config.attention_dp > 1
        if (rows in (8, 16, 32, 64, 96, 128) or dp_rows) and config.decode_no_hlo_calls:
            return {"xla_tpu_enable_deduplicated_calls": "DISABLED"}
        return {}

    def _build_mask_pack(self, control: Any, batch: int) -> Any:
        """Keep the masked plain step within Torch's 64-operand limit: pack
        the control and the rows' grammar bitmasks into one operand (the
        verifier's metadata op does the same)."""
        key = ("mask_pack", control.key(), batch)
        if key in self._ops:
            return self._ops[key]

        def body(state, mask):
            return (jnp.concatenate((state, mask.reshape(-1))),)

        self._ops[key] = register_op(
            f"pallas::k3_mask_pack_{batch}",
            self.mesh,
            body,
            (P(), P()),
            (P(),),
            export_key=repr(key),
        )
        return self._ops[key]

    def plain_step(
        self,
        state: torch.Tensor,
        batch: int,
        control: Any,
        sub_base: int,
        sampled: bool,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run :meth:`_build_plain`; returns
        ``(tokens [batch, VERIFY_ROWS], state)``. ``mask``: the rows' grammar
        bitmasks ``[batch, VOCAB // 32]`` int32 (all-ones rows unconstrained)."""
        op = self._build_plain(batch, control, sub_base, sampled, mask is not None)
        if mask is not None:
            (state,) = self._build_mask_pack(control, batch)(state, mask)
        outputs = self._run(op, batch, *self._operands(), state)
        emitted, state, conv, recurrent, *pools = outputs
        self._adopt(conv, recurrent, pools)
        return emitted, state

    @staticmethod
    def _run(op: Any, rows: int, *args: Any) -> Any:
        """``op(*args)`` for a ``rows``-row step: at 8, 16, 32, 64, 96 and 128 rows
        (and attention DP's 2 and 4) with XLA's deduplicated calls off
        (``KernelConfig.decode_no_hlo_calls``)."""
        options = DecodeOps._compiler_options(rows)
        if options:
            from torch_tpu._internal.compiler_options import custom_compiler_options
            from torch_tpu._internal.sync import synchronize

            # Deduplicated calls prevent jumbo instruction overlays and split
            # the B8/B16/B32 expert loop. Capture this option when the lazy
            # graph is materialized, then restore the caller's compiler
            # configuration.
            with custom_compiler_options(options):
                outputs = op(*args)
                synchronize(list(outputs), wait=False)
            return outputs
        return op(*args)

    def _build_verify(
        self, aux_layers: tuple[int, ...], control, synthetic_rates=None
    ) -> Any:
        """The speculative verify: one sequence's ``VERIFY_ROWS`` tokens at
        consecutive positions (kernel ``sequence_rows``), each row with its
        own temporary post-state, with input state from the request slot. Its
        packed operand is ``[tokens | positions | slots | state_row |
        table]``. Returns the vocab-sharded logits (each worker holds its
        ``[1, rows, V/32]``), the aux hidden states ``[L, rows, hidden]``
        (replicated), acceptance decisions, persistent KDA banks and pools.
        Temporary post-states stay inside the op. The input banks are donated
        only to the non-communicating commit after all native readers finish."""
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

        def body(*args):
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
                old_conv,
                old_recurrent,
                conv,
                recurrent,
                logits,
                tokens,
                positions,
                slots,
                plan,
                modes,
                synthetic_rates,
                True,
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
            in_specs,
            out_specs,
            donate_argnums=tuple(range(count, count + 2 + pools)),
            export_key=export_key,
        )
        return self._ops[key]

    def verify_packed(
        self, ids, aux_layers, control, state, modes, synthetic_rates=None
    ):
        """Verify, decide acceptance and commit states before returning to Torch."""
        (packet,) = self._build_verify_metadata(control, 1)(ids, state, modes)
        op = self._build_verify(tuple(aux_layers), control, synthetic_rates)
        sharded, aux, decisions, conv, recurrent, *pools = self._run(
            op, self.VERIFY_ROWS, *self._operands(), packet
        )
        self._adopt(conv, recurrent, pools)
        return sharded, aux, decisions

    def _build_verify_batched(
        self,
        aux_layers: tuple[int, ...],
        requests: int,
        rows_per: int,
        control,
        synthetic_rates=None,
    ) -> Any:
        """A speculative verify of ``requests`` sequences of ``rows_per``
        tokens each (kernel ``sequence_rows``): each group starts from its
        request's persistent state and emits temporary post-states. The packed
        operand is ``[tokens | positions | slots | indices | tables]`` (per
        row, except one block-table row per request; ``indices`` are for the
        draft op). Returns what :meth:`_build_verify` does."""
        key = ("verify", aux_layers, requests, rows_per, control.key(), synthetic_rates)
        if key in self._ops:
            return self._ops[key]
        rows, paged, names = requests * rows_per, self.paged, self.names
        count = len(names)
        options = decode_options(rows)
        options["defer_kda_commit"] = self.attention_dp == 1
        # The batch recipe: one request with four verify rows (1x4) uses the
        # plain B4 layout; token-scattered attention/MoE requires at least B8.
        options.update(
            sequence_rows=rows_per,
            aux_hidden_layers=tuple(aux_layers),
        )
        pool_map, owner_table = self.pool_map, self.owner_table
        pools = len(self.unified_pools)

        def body(*args):
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
            tables = ids[4 * rows :].reshape(requests, -1)
            logits, aux, conv, recurrent, pool = paged.decode_paged(
                weights,
                conv,
                recurrent,
                pool,
                tokens,
                positions,
                jnp.repeat(tables, rows_per, axis=0),
                slots,
                batch_size=rows,
                layers=NUM_LAYERS,
                dcp_pool_map=pool_map,
                dcp_owner_table=owner_table,
                **options,
            )
            decisions, conv, recurrent = self._finish_verify(
                control,
                old_conv,
                old_recurrent,
                conv,
                recurrent,
                logits,
                tokens,
                positions,
                slots,
                plan,
                modes,
                synthetic_rates,
                False,
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
            in_specs,
            out_specs,
            donate_argnums=tuple(range(count, count + 2 + pools)),
            export_key=export_key,
        )
        return self._ops[key]

    def verify_batched(
        self,
        ids,
        aux_layers,
        requests,
        rows_per,
        control,
        state,
        modes,
        synthetic_rates=None,
    ):
        """Batched verify/accept/commit with only persistent state live out."""
        (packet,) = self._build_verify_metadata(control, requests)(ids, state, modes)
        op = self._build_verify_batched(
            tuple(aux_layers), requests, rows_per, control, synthetic_rates
        )
        sharded, aux, decisions, conv, recurrent, *pools = self._run(
            op, requests * rows_per, *self._operands(), packet
        )
        self._adopt(conv, recurrent, pools)
        return sharded, aux, decisions

    def _build_verify_metadata(self, control, requests):
        """Keep the verifier within Torch's 64-operand limit."""
        key = ("verify_metadata", control.key(), requests)
        if key not in self._ops:

            def body(ids, state, modes):
                plan, _ = control.fields(state, requests)
                return (jnp.concatenate((ids, plan.reshape(-1), modes)),)

            self._ops[key] = register_op(
                f"pallas::k3_verify_metadata_{requests}",
                self.mesh,
                body,
                (P(), P(), P()),
                (P(),),
                export_key=repr(key),
            )
        return self._ops[key]

    @staticmethod
    def _unpack_verify_metadata(packet, control, requests):
        size = requests * (control.FIELDS + 1)
        return (
            packet[:-size],
            packet[-size:-requests].reshape(requests, control.FIELDS),
            packet[-requests:],
        )

    def _finish_verify(
        self,
        control,
        conv,
        recurrent,
        new_conv,
        new_recurrent,
        logits,
        tokens,
        positions,
        slots,
        plan,
        modes,
        synthetic_rates,
        single,
    ):
        from vllm_torchtpu.megakernel.kimi_k3 import sampling

        requests = modes.shape[0]
        rows_per = tokens.shape[0] // requests
        decisions = sampling.verify_acceptance(
            logits,
            tokens.reshape(requests, rows_per),
            positions.reshape(requests, rows_per)[:, 0],
            modes,
            control.sampling(plan),
            synthetic_rates=synthetic_rates,
            single=single,
        )
        targets = slots.reshape(requests, rows_per)[:, 0]
        # The native custom call finishes before this consumer can write the
        # donated banks. It never aliases its temporary outputs to the banks.
        # Padding (null-slot) requests commit the last verifier row.
        sources = jnp.arange(requests) * rows_per + jnp.where(
            targets == 0, rows_per - 1, decisions[:, 0]
        )
        conv, recurrent = self.paged._commit_kda_states(
            conv, recurrent, new_conv, new_recurrent, targets, source_rows=sources
        )
        return decisions, conv, recurrent


# Sampling parameters of a greedy plan entry: temperature, top_p, top_k, seed.
GREEDY = (0.0, 1.0, 0, 0)


class Control:
    """The layout of the decoder's device control state: one replicated int32
    vector every op reads and returns updated, so nothing is uploaded per step
    while the host's tokens and positions lag (async scheduling).

    ``[tokens | positions | rows]`` (one per regular KDA slot), then the plan:
    ``rows x FIELDS`` per-row fields and the rows' block tables (``rows x
    pages``). The ops clear the reset flags of the live rows they consumed
    (not on the null slot 0, which padding rows use). A slot's ``row`` records
    the verifier entry acceptance selected; only its post-state is committed."""

    FIELDS = 8
    SLOT, RESET, TOKEN, POSITION, TEMPERATURE, TOP_P, TOP_K, SEED = range(FIELDS)
    BANK = 7  # logical verifier entries after the anchor

    @classmethod
    def total_slots(cls, slots: int) -> int:
        """One persistent KDA state per regular slot, including null slot 0;
        verifier post-states are temporary, so speculation adds no slots."""
        return slots

    def state_slot(self, slot, row):
        """All verifier rows read the accepted state; ``row`` only records the
        selected entry (broadcast over it keeps the operand shape)."""
        return slot + jnp.zeros_like(row)

    def __init__(self, slots: int, rows: int, pages: int):
        self.slots, self.rows, self.pages = int(slots), int(rows), int(pages)
        self.plan_start = 3 * self.slots
        self.tables_start = self.plan_start + self.rows * self.FIELDS
        self.size = self.tables_start + self.rows * self.pages

    def accepted_state_rows(self, ids, state, requests, rows_per, *, single=False):
        """Native source rows and bank destinations after DP1 acceptance.

        The batched entry permutes bank entries so its first row reads the
        previous accepted state. Control records the selected bank entry, not
        its row in this verifier. Padding commits the final verifier row,
        independent of its accepted count.
        """
        total = requests * rows_per
        plan, _ = self.fields(state, requests)
        slots = plan[:, self.SLOT]
        accepted_entry = state[2 * self.slots + slots]
        entries = (
            jnp.arange(rows_per, dtype=jnp.int32)[None, :]
            if single
            else ids[3 * total : 4 * total].reshape(requests, rows_per)
        )
        selected = jnp.argmax(entries == accepted_entry[:, None], axis=1)
        source = jnp.arange(requests, dtype=jnp.int32) * rows_per + jnp.where(
            slots == 0, rows_per - 1, selected
        )
        return source, ids[2 * total + source]

    def key(self) -> tuple[int, int, int]:
        return self.slots, self.rows, self.pages

    # ----- host
    def plan(self, entries, tables) -> np.ndarray:
        """The plan part for ``entries`` of ``(slot, reset, token, position,
        (temperature, top_p, top_k, seed))``; the rest are padding rows."""
        fields = np.zeros((self.rows, self.FIELDS), np.int32)
        fields[:, self.RESET] = 1
        fields[:, self.TEMPERATURE] = np.float32(0).view(np.int32)
        fields[:, self.TOP_P] = np.float32(1).view(np.int32)
        table = np.zeros((self.rows, self.pages), np.int32)
        for r, (
            (slot, reset, token, position, (temperature, top_p, top_k, seed)),
            row,
        ) in enumerate(zip(entries, tables, strict=True)):
            fields[r] = (
                slot,
                int(reset),
                token if reset else 0,
                position if reset else 0,
                np.float32(temperature).view(np.int32),
                np.float32(top_p).view(np.int32),
                top_k,
                seed & 0x7FFFFFFF,
            )
            width = min(self.pages, len(row))
            table[r, :width] = row[:width]
        return np.concatenate((fields.reshape(-1), table.reshape(-1)))

    def consumed(self, plan: np.ndarray, rows: int) -> np.ndarray:
        """``plan`` as the device holds it after an op consumed its first
        ``rows`` rows' resets (live rows only)."""
        plan = plan.copy()
        fields = plan[: self.rows * self.FIELDS].reshape(self.rows, self.FIELDS)
        live = np.zeros(self.rows, bool)
        live[:rows] = fields[:rows, self.SLOT] != 0
        fields[live, self.RESET] = 0
        fields[live, self.TOKEN] = 0
        fields[live, self.POSITION] = 0
        return plan

    # ----- in the ops (jax)
    def fields(self, control, batch: int):
        plan = control[self.plan_start : self.plan_start + batch * self.FIELDS].reshape(
            batch, self.FIELDS
        )
        tables = control[
            self.tables_start : self.tables_start + batch * self.pages
        ].reshape(batch, self.pages)
        return plan, tables

    def inputs(self, control, batch: int):
        """``(slot, token, position, row, plan, tables)`` of the first ``batch``
        plan rows."""
        import jax.numpy as jnp

        plan, tables = self.fields(control, batch)
        slot, reset = plan[:, self.SLOT], plan[:, self.RESET] != 0
        token = jnp.where(reset, plan[:, self.TOKEN], control[slot])
        position = jnp.where(reset, plan[:, self.POSITION], control[self.slots + slot])
        row = jnp.where(reset, 0, control[2 * self.slots + slot])
        return slot, token, position, row, plan, tables

    def sampling(self, plan):
        """``(temperature, top_p, top_k, seed)`` per plan row."""
        import jax
        import jax.numpy as jnp

        def as_float(bits):
            return jax.lax.bitcast_convert_type(bits, jnp.float32)

        return (
            as_float(plan[:, self.TEMPERATURE]),
            as_float(plan[:, self.TOP_P]),
            plan[:, self.TOP_K],
            plan[:, self.SEED],
        )

    def advance(self, control, batch: int, slot, token, position, row):
        """Record each row's next token, position and state row; clear the
        live rows' reset flags."""
        import jax.numpy as jnp

        control = control.at[slot].set(token)
        control = control.at[self.slots + slot].set(position)
        control = control.at[2 * self.slots + slot].set(row)
        resets = self.plan_start + jnp.arange(batch) * self.FIELDS + self.RESET
        return control.at[resets].set(jnp.where(slot != 0, 0, control[resets]))
