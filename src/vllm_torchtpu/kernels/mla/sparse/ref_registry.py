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
"""Declarative trace-time Pallas ref registry and typed views for SparseMLA."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse.config import (
    Q_ROPE_DIM,
    SEND_PAD_ROWS,
    SEND_SLAB_ROW0,
    AttentionKernelConfig,
    BodyExecutionStrategy,
    HbmPinRole,
    align_to,
    send_layout_lanes,
    send_layout_shape,
)


def _hbm_block_spec() -> pl.BlockSpec:
    return pl.BlockSpec(memory_space=pltpu.HBM)


def _smem_block_spec() -> pl.BlockSpec:
    return pl.BlockSpec(memory_space=pltpu.SMEM)


def _chunk_index_map(chunk_idx: Any, _token_end: Any) -> tuple[Any, int]:
    return (chunk_idx, 0)


def _zero_2d_index_map(_chunk_idx: Any, _token_end: Any) -> tuple[int, int]:
    return (0, 0)


@dataclasses.dataclass(frozen=True)
class SlotBinding:
    """Static mapping from a logical ref name to a positional slice in ``*raw_refs``."""

    name: str
    start: int
    count: int = 1
    is_group: bool = False


class BoundRefs:
    """Trace-time name-to-ref view constructed inside a Pallas kernel body."""

    __slots__ = ("_refs",)

    def __init__(self, refs: dict[str, Any]) -> None:
        self._refs = refs

    def __getitem__(self, name: str) -> Any:
        return self._refs[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._refs.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self._refs

    def __len__(self) -> int:
        return len(self._refs)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._refs)


@dataclasses.dataclass(frozen=True)
class RefLayout:
    """Frozen, hashable metadata describing positional Pallas kernel ref slots.

    Safe to close over via ``functools.partial`` into ``pl.pallas_call`` because
    it holds only static names and integer slot offsets, never ``jax.Array``
    operands.
    """

    bindings: tuple[SlotBinding, ...]
    total_slots: int
    num_prefetches: int
    num_inputs: int
    num_outputs: int
    num_scratch: int

    def bind(self, *raw_refs: Any) -> BoundRefs:
        """Binds positional Pallas kernel ``*raw_refs`` to their registered names."""
        if len(raw_refs) != self.total_slots:
            raise ValueError(
                f"Expected {self.total_slots} Pallas refs "
                f"(prefetches={self.num_prefetches}, inputs={self.num_inputs}, "
                f"outputs={self.num_outputs}, scratch={self.num_scratch}), "
                f"got {len(raw_refs)}."
            )
        bound: dict[str, Any] = {}
        for slot in self.bindings:
            if slot.is_group:
                bound[slot.name] = tuple(
                    raw_refs[slot.start : slot.start + slot.count]
                )
            else:
                bound[slot.name] = raw_refs[slot.start]
        return BoundRefs(bound)


@dataclasses.dataclass(frozen=True)
class _PrefetchEntry:
    name: str
    value: jax.Array


@dataclasses.dataclass(frozen=True)
class _InputEntry:
    name: str
    value: jax.Array
    spec: pl.BlockSpec
    pin_role: HbmPinRole | None
    alias_output: str | None


@dataclasses.dataclass(frozen=True)
class _OutputEntry:
    name: str
    shape_dtypes: tuple[jax.ShapeDtypeStruct, ...]
    specs: tuple[pl.BlockSpec, ...]
    is_group: bool


@dataclasses.dataclass(frozen=True)
class _ScratchEntry:
    name: str
    shape_spec: Any


class PallasRefRegistry:
    """Declarative trace-time builder for ``PrefetchScalarGridSpec`` slots.

    Binds scalar prefetches, inputs, outputs (single or grouped), and scratch
    allocations by name while preserving exact Pallas positional ordering and
    deriving ``input_output_aliases`` without manual index arithmetic.
    """

    def __init__(self) -> None:
        self._prefetches: list[_PrefetchEntry] = []
        self._inputs: list[_InputEntry] = []
        self._outputs: list[_OutputEntry] = []
        self._scratches: list[_ScratchEntry] = []
        self._seen_names: set[str] = set()

    def _register_name(self, name: str) -> None:
        if name in self._seen_names:
            raise ValueError(f"Duplicate Pallas ref name: {name!r}")
        self._seen_names.add(name)

    def add_prefetch(self, name: str, value: jax.Array) -> None:
        """Registers a scalar prefetch array (SMEM)."""
        self._register_name(name)
        self._prefetches.append(_PrefetchEntry(name=name, value=value))

    def add_input(
        self,
        name: str,
        value: jax.Array,
        spec: pl.BlockSpec | None = None,
        *,
        pin_role: HbmPinRole | None = None,
        alias_output: str | None = None,
    ) -> None:
        """Registers an input operand, defaulting ``spec`` to HBM."""
        self._register_name(name)
        resolved_spec = _hbm_block_spec() if spec is None else spec
        self._inputs.append(
            _InputEntry(
                name=name,
                value=value,
                spec=resolved_spec,
                pin_role=pin_role,
                alias_output=alias_output,
            )
        )

    def add_output(
        self,
        name: str,
        shape_dtype: jax.ShapeDtypeStruct,
        spec: pl.BlockSpec | None = None,
    ) -> None:
        """Registers a single output tensor, defaulting ``spec`` to HBM."""
        self._register_name(name)
        resolved_spec = _hbm_block_spec() if spec is None else spec
        self._outputs.append(
            _OutputEntry(
                name=name,
                shape_dtypes=(shape_dtype,),
                specs=(resolved_spec,),
                is_group=False,
            )
        )

    def add_output_group(
        self,
        name: str,
        shape_dtypes: Sequence[jax.ShapeDtypeStruct],
        specs: Sequence[pl.BlockSpec],
    ) -> None:
        """Registers a contiguous group of outputs bound as a tuple under ``name``."""
        shape_tuple = tuple(shape_dtypes)
        spec_tuple = tuple(specs)
        if not shape_tuple:
            raise ValueError(f"Output group {name!r} must be non-empty.")
        if len(shape_tuple) != len(spec_tuple):
            raise ValueError(
                f"Output group {name!r} has {len(shape_tuple)} shapes but "
                f"{len(spec_tuple)} BlockSpecs."
            )
        self._register_name(name)
        self._outputs.append(
            _OutputEntry(
                name=name,
                shape_dtypes=shape_tuple,
                specs=spec_tuple,
                is_group=True,
            )
        )

    def add_scratch(self, name: str, shape_spec: Any) -> None:
        """Registers a VMEM/SMEM/semaphore scratch allocation."""
        self._register_name(name)
        self._scratches.append(_ScratchEntry(name=name, shape_spec=shape_spec))

    @property
    def layout(self) -> RefLayout:
        """Returns frozen, hashable slot metadata for ``pl.pallas_call`` closure."""
        bindings: list[SlotBinding] = []
        offset = 0
        for p in self._prefetches:
            bindings.append(SlotBinding(name=p.name, start=offset))
            offset += 1
        for inp in self._inputs:
            bindings.append(SlotBinding(name=inp.name, start=offset))
            offset += 1
        num_outputs = 0
        for out in self._outputs:
            count = len(out.shape_dtypes)
            bindings.append(
                SlotBinding(
                    name=out.name,
                    start=offset,
                    count=count,
                    is_group=out.is_group,
                )
            )
            offset += count
            num_outputs += count
        for scr in self._scratches:
            bindings.append(SlotBinding(name=scr.name, start=offset))
            offset += 1
        return RefLayout(
            bindings=tuple(bindings),
            total_slots=offset,
            num_prefetches=len(self._prefetches),
            num_inputs=len(self._inputs),
            num_outputs=num_outputs,
            num_scratch=len(self._scratches),
        )

    def _flat_outputs(
        self,
    ) -> tuple[list[jax.ShapeDtypeStruct], list[pl.BlockSpec]]:
        shapes: list[jax.ShapeDtypeStruct] = []
        specs: list[pl.BlockSpec] = []
        for out in self._outputs:
            shapes.extend(out.shape_dtypes)
            specs.extend(out.specs)
        if not shapes:
            raise ValueError("At least one output must be registered.")
        return shapes, specs

    def build_grid_spec(self, grid: tuple[Any, ...]) -> pltpu.PrefetchScalarGridSpec:
        """Builds ``PrefetchScalarGridSpec`` preserving single-output convention."""
        _, flat_out_specs = self._flat_outputs()
        out_specs: Any = (
            flat_out_specs[0] if len(flat_out_specs) == 1 else list(flat_out_specs)
        )
        return pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(self._prefetches),
            in_specs=[entry.spec for entry in self._inputs],
            out_specs=out_specs,
            scratch_shapes=[entry.shape_spec for entry in self._scratches],
            grid=grid,
        )

    @property
    def out_shape(self) -> Any:
        """Single ``ShapeDtypeStruct`` if 1 output, else ``list[ShapeDtypeStruct]``."""
        flat_shapes, _ = self._flat_outputs()
        return flat_shapes[0] if len(flat_shapes) == 1 else flat_shapes

    @property
    def input_output_aliases(self) -> dict[int, int]:
        """Computes exact ``{prefetch_count + input_idx: output_idx}`` alias map."""
        output_indices: dict[str, int] = {}
        group_names: set[str] = set()
        out_idx = 0
        for out in self._outputs:
            if out.is_group:
                group_names.add(out.name)
                out_idx += len(out.shape_dtypes)
            else:
                output_indices[out.name] = out_idx
                out_idx += 1

        aliases: dict[int, int] = {}
        aliased_outputs: dict[int, str] = {}
        num_prefetches = len(self._prefetches)
        for input_idx, inp in enumerate(self._inputs):
            if inp.alias_output is None:
                continue
            target = inp.alias_output
            if target in group_names:
                raise ValueError(
                    f"Input {inp.name!r} cannot alias output group {target!r}."
                )
            if target not in output_indices:
                raise ValueError(
                    f"Input {inp.name!r} aliases unknown output {target!r}."
                )
            target_idx = output_indices[target]
            if target_idx in aliased_outputs:
                raise ValueError(
                    f"Output {target!r} (index {target_idx}) is aliased by both "
                    f"{aliased_outputs[target_idx]!r} and {inp.name!r}."
                )
            aliased_outputs[target_idx] = inp.name
            aliases[num_prefetches + input_idx] = target_idx
        return aliases

    def call_args(
        self,
        pin_fn: Callable[[HbmPinRole | None], bool] | None = None,
    ) -> tuple[Any, ...]:
        """Returns ``(*prefetches, *inputs)``, pinning inputs where ``pin_fn`` holds."""
        prefetch_vals = [entry.value for entry in self._prefetches]
        input_vals = [
            pltpu.with_memory_space_constraint(entry.value, pltpu.HBM)
            if pin_fn is not None and pin_fn(entry.pin_role)
            else entry.value
            for entry in self._inputs
        ]
        return (*prefetch_vals, *input_vals)

    def bind(self, *raw_refs: Any) -> BoundRefs:
        """Binds positional ``*raw_refs`` using this registry's ``RefLayout``."""
        return self.layout.bind(*raw_refs)


@dataclasses.dataclass(frozen=True)
class BoundsLayout:
    """Frozen, hashable mapping from named scalar-bounds slots to integer indices."""

    names: tuple[str, ...]

    def index(self, name: str) -> int:
        """Returns the integer offset of ``name``, raising ``KeyError`` if absent."""
        try:
            return self.names.index(name)
        except ValueError as err:
            raise KeyError(
                f"Bounds slot {name!r} is not registered; available: {self.names}"
            ) from err

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.names)


class BoundsBuilder:
    """Builds the int32 bounds prefetch vector alongside its ``BoundsLayout``."""

    def __init__(self) -> None:
        self._names: list[str] = []
        self._values: list[int | jax.Array] = []
        self._seen: set[str] = set()

    def add(self, name: str, value: int | jax.Array) -> None:
        """Appends a named int32 scalar to the bounds prefetch vector."""
        if name in self._seen:
            raise ValueError(f"Duplicate bounds slot name: {name!r}")
        self._seen.add(name)
        self._names.append(name)
        self._values.append(value)

    def build(self) -> tuple[jax.Array, BoundsLayout]:
        """Materializes ``jnp.array(values, jnp.int32)`` and frozen ``BoundsLayout``."""
        if not self._names:
            raise ValueError("BoundsBuilder requires at least one slot.")
        return jnp.array(self._values, jnp.int32), BoundsLayout(tuple(self._names))


@dataclasses.dataclass(frozen=True)
class BoundsRef:
    """Named trace-time accessor over the scalar-prefetch bounds ref."""

    raw: Any
    layout: BoundsLayout

    def _slot(self, name: str) -> Any:
        return self.raw[self.layout.index(name)]

    def __getitem__(self, name: str) -> Any:
        return self._slot(name)

    def __contains__(self, name: str) -> bool:
        return name in self.layout

    @property
    def start_seq_idx(self) -> Any:
        return self._slot("start_seq_idx")

    @property
    def end_seq_idx(self) -> Any:
        return self._slot("end_seq_idx")

    @property
    def prefix_end(self) -> Any:
        return self._slot("prefix_end")

    @property
    def peer_rank(self) -> Any:
        return self._slot("peer_rank")

    @property
    def peer_lane(self) -> Any:
        return self._slot("peer_lane")

    @property
    def real_tokens(self) -> Any:
        return self._slot("real_tokens")


@dataclasses.dataclass(frozen=True)
class DmaSemaphores:
    """Named view over the attention kernel's ``DMA[(5 if has_lse else 4, 2)]``."""

    raw: Any
    has_lse: bool

    def kv_nope(self, slot: Any) -> Any:
        return self.raw.at[0, slot]

    def q(self, slot: Any) -> Any:
        return self.raw.at[1, slot]

    def o(self, slot: Any) -> Any:
        return self.raw.at[2, slot]

    def kv_rope(self, slot: Any) -> Any:
        return self.raw.at[3, slot]

    def lse(self, slot: Any) -> Any:
        assert self.has_lse, "LSE semaphore requested with return_lse=False."
        return self.raw.at[4, slot]


@dataclasses.dataclass(frozen=True)
class OverflowFixupRefs:
    """Scratch refs for capped-window overflow online-softmax fix-up."""

    nope_x2: Any
    rope_x2: Any
    m: Any
    l: Any
    acc: Any
    sems: Any


@dataclasses.dataclass(frozen=True)
class LseFillRefs:
    """Scratch refs for kernel-owned LSE fill, send word header, or exchange sems."""

    fill: Any
    sem: Any
    send_wbuf: Any = None
    xchg_sems: Any = None


@dataclasses.dataclass(frozen=True)
class AttentionRefs:
    """Typed view of all Pallas refs passed into ``_attention_kernel``."""

    # Prefetches
    kv_lens: Any
    bounds: BoundsRef
    sem_ids: Any
    group_tab: Any
    # Inputs
    q: Any
    cache_kv_nope: Any
    cache_kv_rope: Any
    q_pe: Any
    q_rope_table: Any
    live_nope: Any
    live_rope: Any
    # Outputs
    out: Any
    lse: Any
    # Core double-buffered scratch
    bkv_nope_x2: Any
    bkv_rope_x2: Any
    bq_x2: Any
    bo_x2: Any
    blse_x2: Any
    sems: DmaSemaphores
    # Optional scratch
    fixup: OverflowFixupRefs | None
    lse_fill: LseFillRefs | None
    bq_pe_x2: Any
    bq_tab_x2: Any
    kv_stage: Any
    recv_v: Any

    @classmethod
    def from_bound(
        cls,
        bound: BoundRefs,
        bounds_layout: BoundsLayout,
        return_lse: bool,
    ) -> AttentionRefs:
        """Constructs ``AttentionRefs`` from a ``BoundRefs`` view."""
        fixup: OverflowFixupRefs | None = None
        if "fix_nope_x2" in bound:
            fixup = OverflowFixupRefs(
                nope_x2=bound["fix_nope_x2"],
                rope_x2=bound["fix_rope_x2"],
                m=bound["fix_m"],
                l=bound["fix_l"],
                acc=bound["fix_acc"],
                sems=bound["fix_sems"],
            )
        lse_fill_refs: LseFillRefs | None = None
        if "fill" in bound:
            lse_fill_refs = LseFillRefs(
                fill=bound["fill"],
                sem=bound["fill_sem"],
                send_wbuf=bound.get("send_wbuf"),
                xchg_sems=bound.get("xchg_sems"),
            )
        return cls(
            kv_lens=bound["kv_lens"],
            bounds=BoundsRef(bound["bounds"], bounds_layout),
            sem_ids=bound["sem_ids"],
            group_tab=bound.get("group_tab"),
            q=bound["q"],
            cache_kv_nope=bound["cache_kv_nope"],
            cache_kv_rope=bound["cache_kv_rope"],
            q_pe=bound.get("q_pe"),
            q_rope_table=bound.get("q_rope_table"),
            live_nope=bound.get("live_nope"),
            live_rope=bound.get("live_rope"),
            out=bound["out"],
            lse=bound.get("lse"),
            bkv_nope_x2=bound["bkv_nope_x2"],
            bkv_rope_x2=bound["bkv_rope_x2"],
            bq_x2=bound["bq_x2"],
            bo_x2=bound["bo_x2"],
            blse_x2=bound.get("blse_x2"),
            sems=DmaSemaphores(bound["dma_sems"], has_lse=return_lse),
            fixup=fixup,
            lse_fill=lse_fill_refs,
            bq_pe_x2=bound.get("bq_pe_x2"),
            bq_tab_x2=bound.get("bq_tab_x2"),
            kv_stage=bound.get("kv_stage"),
            recv_v=bound.get("recv_v"),
        )


@dataclasses.dataclass(frozen=True)
class LiveFusedPrepRefs:
    """Typed view of all Pallas refs passed to ``_capped_window_live_fused_kernel``."""

    token_end: Any
    windows: Any
    topk: Any
    chunk_outs: tuple[Any, ...]
    kv: Any
    tab: Any
    live: Any
    stage: Any
    phys: Any
    cw: Any
    ab: Any
    sems: Any

    @classmethod
    def from_bound(cls, bound: BoundRefs) -> LiveFusedPrepRefs:
        """Constructs ``LiveFusedPrepRefs`` from a ``BoundRefs`` view."""
        return cls(
            token_end=bound["token_end"],
            windows=bound["windows"],
            topk=bound["topk"],
            chunk_outs=bound["chunk_outs"],
            kv=bound["kv"],
            tab=bound["tab"],
            live=bound["live"],
            stage=bound["stage"],
            phys=bound["phys"],
            cw=bound["cw"],
            ab=bound["ab"],
            sems=bound["sems"],
        )


def _validate_attention_registry_args(
    cfg: AttentionKernelConfig,
    *,
    q: jax.Array,
    group_tab: jax.Array | None,
    q_pe: jax.Array | None,
    q_rope_table: jax.Array | None,
    live_views: tuple[jax.Array, jax.Array] | None,
    lse: jax.Array | None,
    send_recv_in: jax.Array | None,
    out_piece: tuple[int, int, jax.Array | None] | None,
) -> tuple[int, jax.Array | None, int, tuple[int, ...] | None]:
    """Validates attention ref inputs and derives output/head/LSE geometry."""
    num_tokens, num_q_heads, o_dim = q.shape
    out_rows, out_offset, out_carried = (
        (num_tokens, 0, None) if out_piece is None else out_piece
    )

    if cfg.split_q and q_pe is None:
        raise ValueError("cfg.split_q=True requires q_pe.")
    if cfg.q_rope and q_rope_table is None:
        raise ValueError("cfg.q_rope=True requires q_rope_table.")
    if cfg.has_group_tab != (live_views is not None):
        raise ValueError(
            f"cfg.has_group_tab ({cfg.has_group_tab}) must match "
            f"live_views presence ({live_views is not None})."
        )
    if live_views is not None and group_tab is None:
        raise ValueError("live_views requires group_tab.")
    if cfg.recv_in != (send_recv_in is not None):
        raise ValueError(
            f"cfg.recv_in ({cfg.recv_in}) must match "
            f"send_recv_in presence ({send_recv_in is not None})."
        )
    if cfg.out_in != (out_carried is not None):
        raise ValueError(
            f"cfg.out_in ({cfg.out_in}) must match "
            f"out_carried presence ({out_carried is not None})."
        )
    if out_piece is not None:
        if not (cfg.return_lse and not cfg.lse_fill and not cfg.preserve_suffix):
            raise ValueError(
                "out_piece requires return_lse=True, lse_fill=False, "
                "and preserve_suffix=False."
            )
        if lse is None or lse.shape[0] != out_rows:
            raise ValueError(
                f"out_piece requires lse with leading dim {out_rows}, "
                f"got {None if lse is None else lse.shape}."
            )
        if out_offset != cfg.out_row_offset:
            raise ValueError(
                f"out_piece offset ({out_offset}) != cfg.out_row_offset "
                f"({cfg.out_row_offset})."
            )

    if cfg.split_q:
        assert q_pe is not None
        pe_dim = Q_ROPE_DIM if cfg.q_rope else q_pe.shape[-1]
        head_dim = align_to(o_dim + pe_dim, 128)
    else:
        head_dim = o_dim

    lse_shape: tuple[int, ...] | None = (
        (num_tokens, num_q_heads)
        if cfg.lse_fill
        else (None if lse is None else lse.shape)
    )
    return out_rows, out_carried, head_dim, lse_shape


def _register_attention_prefetches(
    reg: PallasRefRegistry,
    cfg: AttentionKernelConfig,
    *,
    kv_lens: jax.Array,
    start_seq_idx: int | jax.Array,
    end_seq_idx: int | jax.Array,
    token_end: jax.Array | None,
    group_tab: jax.Array | None,
    has_live_views: bool,
    send_peer: jax.Array | tuple[jax.Array, ...] | None,
) -> BoundsLayout:
    """Registers scalar prefetches and builds the named ``BoundsLayout``."""
    bounds_builder = BoundsBuilder()
    bounds_builder.add("start_seq_idx", start_seq_idx)
    bounds_builder.add("end_seq_idx", end_seq_idx)
    if cfg.preserve_suffix:
        if token_end is None:
            raise ValueError("cfg.preserve_suffix=True requires token_end.")
        bounds_builder.add("prefix_end", token_end)
    if cfg.xchg_enabled:
        if cfg.preserve_suffix or send_peer is None:
            raise ValueError(
                "cfg.send_exchange requires preserve_suffix=False and send_peer."
            )
        expected_peer = 3 if cfg.xchg_merge else 2
        if isinstance(send_peer, (tuple, list)):
            if len(send_peer) < expected_peer:
                raise ValueError(
                    f"send_peer needs at least {expected_peer} entries, "
                    f"got {len(send_peer)}."
                )
        elif len(send_peer.shape) != 1 or send_peer.shape[0] < expected_peer:
            raise ValueError(
                f"send_peer array must be 1D with at least {expected_peer} "
                f"entries, got shape {send_peer.shape}."
            )
        bounds_builder.add("peer_rank", send_peer[0])
        bounds_builder.add("peer_lane", send_peer[1])
        if cfg.xchg_merge:
            bounds_builder.add("real_tokens", send_peer[2])
    bounds_arr, bounds_layout = bounds_builder.build()

    reg.add_prefetch("kv_lens", kv_lens)
    reg.add_prefetch("bounds", bounds_arr)
    reg.add_prefetch("sem_ids", jnp.zeros((2,), jnp.int32))
    if has_live_views:
        assert group_tab is not None
        reg.add_prefetch("group_tab", group_tab)
    return bounds_layout


def _register_attention_inputs(
    reg: PallasRefRegistry,
    cfg: AttentionKernelConfig,
    *,
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    q_pe: jax.Array | None,
    q_rope_table: jax.Array | None,
    live_views: tuple[jax.Array, jax.Array] | None,
    lse: jax.Array | None,
    send_recv_in: jax.Array | None,
    has_out_piece: bool,
    out_carried: jax.Array | None,
) -> None:
    """Registers HBM inputs with exact pin roles and output aliases."""
    reg.add_input(
        "q",
        q,
        pin_role=HbmPinRole.Q,
        alias_output=None if has_out_piece else "out",
    )
    reg.add_input("cache_kv_nope", cache_kv_nope, pin_role=HbmPinRole.KV)
    reg.add_input("cache_kv_rope", cache_kv_rope, pin_role=HbmPinRole.KV)
    if cfg.split_q:
        assert q_pe is not None
        reg.add_input("q_pe", q_pe, pin_role=HbmPinRole.Q_PE)
    if cfg.q_rope:
        assert q_rope_table is not None
        reg.add_input("q_rope_table", q_rope_table, pin_role=HbmPinRole.TAB)
    if live_views is not None:
        live_nope, live_rope = live_views
        reg.add_input("live_nope", live_nope, pin_role=HbmPinRole.KV)
        reg.add_input("live_rope", live_rope, pin_role=HbmPinRole.KV)
    if cfg.return_lse:
        if not cfg.lse_fill:
            if lse is None:
                raise ValueError("return_lse=True with lse_fill=False requires lse.")
            reg.add_input("lse_in", lse, pin_role=None, alias_output="lse")
        elif send_recv_in is not None:
            reg.add_input(
                "recv_in", send_recv_in, pin_role=HbmPinRole.RECV, alias_output="lse"
            )
    if out_carried is not None:
        reg.add_input("out_carried", out_carried, pin_role=None, alias_output="out")


def _register_attention_outputs(
    reg: PallasRefRegistry,
    cfg: AttentionKernelConfig,
    *,
    q: jax.Array,
    out_rows: int,
    lse_shape: tuple[int, ...] | None,
) -> None:
    """Registers HBM outputs (``out`` and optional ``lse`` / send buffer)."""
    num_tokens, num_q_heads, o_dim = q.shape
    reg.add_output(
        "out", jax.ShapeDtypeStruct(shape=(out_rows, *q.shape[1:]), dtype=q.dtype)
    )
    if not cfg.return_lse:
        return
    if cfg.xchg_merge:
        lse_out = jax.ShapeDtypeStruct(
            shape=(num_tokens * num_q_heads // cfg.send_world, o_dim),
            dtype=jnp.int16,
        )
    elif cfg.xchg_enabled:
        send_shape = send_layout_shape(num_tokens, num_q_heads, cfg.send_world, o_dim)
        lse_out = jax.ShapeDtypeStruct(
            shape=(send_shape[0] * send_shape[1], o_dim), dtype=jnp.int16
        )
    elif cfg.send_enabled:
        lse_out = jax.ShapeDtypeStruct(
            shape=send_layout_shape(num_tokens, num_q_heads, cfg.send_world, o_dim),
            dtype=jnp.int16,
        )
    else:
        assert lse_shape is not None
        lse_out = jax.ShapeDtypeStruct(shape=lse_shape, dtype=jnp.float32)
    reg.add_output("lse", lse_out)


def _register_optional_attention_scratches(
    reg: PallasRefRegistry,
    cfg: AttentionKernelConfig,
    *,
    num_tokens: int,
    num_q_heads: int,
    o_dim: int,
    head_dim: int,
    send_lanes: int,
    send_step_rows: int,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    lse_shape: tuple[int, ...] | None,
    q_pe: jax.Array | None,
    q_rope_table: jax.Array | None,
    send_recv_in: jax.Array | None,
) -> None:
    """Registers optional feature scratches in exact kernel registration order."""
    bsz = cfg.batch_size
    vmem, dma_sem = pltpu.VMEM, pltpu.SemaphoreType.DMA
    if cfg.capped_windows is not None:
        nope_tile, rope_tile = cache_kv_nope.shape[1:], cache_kv_rope.shape[1:]
        reg.add_scratch("fix_nope_x2", vmem((2, *nope_tile), cache_kv_nope.dtype))
        reg.add_scratch("fix_rope_x2", vmem((2, *rope_tile), cache_kv_rope.dtype))
        reg.add_scratch("fix_m", vmem((num_q_heads, 1), jnp.float32))
        reg.add_scratch("fix_l", vmem((num_q_heads, 1), jnp.float32))
        reg.add_scratch("fix_acc", vmem((num_q_heads, head_dim), jnp.float32))
        reg.add_scratch("fix_sems", dma_sem((2, 2)))

    if cfg.lse_fill and cfg.send_enabled:
        reg.add_scratch(
            "fill", vmem((num_tokens // bsz, bsz, send_lanes), jnp.float32)
        )
        if cfg.xchg_enabled:
            reg.add_scratch("xchg_sems", dma_sem((3,)))
        else:
            reg.add_scratch(
                "send_wbuf", vmem((cfg.send_world, SEND_PAD_ROWS, o_dim), jnp.int16)
            )
        reg.add_scratch("fill_sem", dma_sem(()))
    elif cfg.lse_fill:
        assert lse_shape is not None
        reg.add_scratch("fill", vmem(lse_shape, jnp.float32))
        reg.add_scratch("fill_sem", dma_sem(()))

    if cfg.q_rope:
        assert q_rope_table is not None
        reg.add_scratch(
            "bq_tab_x2", vmem((2, bsz, *q_rope_table.shape[1:]), jnp.float32)
        )
    if cfg.split_q:
        assert q_pe is not None
        pe_lanes = 128 if cfg.body.narrow_q_rope_dma else q_pe.shape[-1]
        reg.add_scratch("bq_pe_x2", vmem((2, bsz, num_q_heads, pe_lanes), q_pe.dtype))
    if cfg.body.uses_kv_stage:
        kv_stage_shape = (
            (1, bsz, cache_kv_nope.shape[1], head_dim)
            if cfg.body.strategy == BodyExecutionStrategy.QUAD
            else (2, 2, cache_kv_nope.shape[1], head_dim)
        )
        reg.add_scratch("kv_stage", vmem(kv_stage_shape, jnp.float8_e4m3fn))
    if cfg.xchg_merge:
        if send_recv_in is not None:
            raise ValueError("merge mode has no receive buffer")
        reg.add_scratch(
            "recv_v",
            vmem((cfg.send_world, SEND_SLAB_ROW0 + send_step_rows, o_dim), jnp.int16),
        )


def _register_attention_scratches(
    reg: PallasRefRegistry,
    cfg: AttentionKernelConfig,
    *,
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    head_dim: int,
    lse_shape: tuple[int, ...] | None,
    q_pe: jax.Array | None,
    q_rope_table: jax.Array | None,
    send_recv_in: jax.Array | None,
) -> None:
    """Registers VMEM and DMA-semaphore scratch buffers in exact kernel order."""
    bsz = cfg.batch_size
    num_tokens, num_q_heads, o_dim = q.shape
    vmem = pltpu.VMEM
    bq_buf = vmem((2, bsz, num_q_heads, o_dim), q.dtype)
    send_step_rows = bsz * num_q_heads // cfg.send_world if cfg.send_enabled else 0
    bo_buf = (
        vmem((2, cfg.send_world, cfg.slab_row0 + send_step_rows, o_dim), jnp.int16)
        if cfg.send_enabled
        else bq_buf
    )
    send_lanes = (
        send_layout_lanes(num_tokens, num_q_heads, cfg.send_world)
        if cfg.send_enabled
        else 0
    )

    reg.add_scratch(
        "bkv_nope_x2", vmem((2, bsz, *cache_kv_nope.shape[1:]), cache_kv_nope.dtype)
    )
    reg.add_scratch(
        "bkv_rope_x2", vmem((2, bsz, *cache_kv_rope.shape[1:]), cache_kv_rope.dtype)
    )
    reg.add_scratch("bq_x2", bq_buf)
    reg.add_scratch("bo_x2", bo_buf)
    if cfg.return_lse:
        blse_shape = (
            (2, bsz, send_lanes or num_q_heads)
            if cfg.lse_lane_dense
            else (2, bsz, num_q_heads, 1)
        )
        reg.add_scratch("blse_x2", vmem(blse_shape, jnp.float32))
    reg.add_scratch(
        "dma_sems", pltpu.SemaphoreType.DMA((5 if cfg.return_lse else 4, 2))
    )

    _register_optional_attention_scratches(
        reg,
        cfg,
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        o_dim=o_dim,
        head_dim=head_dim,
        send_lanes=send_lanes,
        send_step_rows=send_step_rows,
        cache_kv_nope=cache_kv_nope,
        cache_kv_rope=cache_kv_rope,
        lse_shape=lse_shape,
        q_pe=q_pe,
        q_rope_table=q_rope_table,
        send_recv_in=send_recv_in,
    )


def build_attention_ref_registry(
    cfg: AttentionKernelConfig,
    *,
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    kv_lens: jax.Array,
    start_seq_idx: int | jax.Array,
    end_seq_idx: int | jax.Array,
    token_end: jax.Array | None = None,
    group_tab: jax.Array | None = None,
    q_pe: jax.Array | None = None,
    q_rope_table: jax.Array | None = None,
    live_views: tuple[jax.Array, jax.Array] | None = None,
    lse: jax.Array | None = None,
    send_recv_in: jax.Array | None = None,
    out_piece: tuple[int, int, jax.Array | None] | None = None,
    send_peer: jax.Array | tuple[jax.Array, ...] | None = None,
) -> tuple[PallasRefRegistry, BoundsLayout]:
    """Builds the ``PallasRefRegistry`` and ``BoundsLayout`` for ``_attention_kernel``.

    Preserves the exact ``PrefetchScalarGridSpec`` registration order, scratch
    buffer shapes, output shapes, and ``input_output_aliases`` of
    ``run_mla_kernel``.
    """
    out_rows, out_carried, head_dim, lse_shape = _validate_attention_registry_args(
        cfg,
        q=q,
        group_tab=group_tab,
        q_pe=q_pe,
        q_rope_table=q_rope_table,
        live_views=live_views,
        lse=lse,
        send_recv_in=send_recv_in,
        out_piece=out_piece,
    )
    reg = PallasRefRegistry()
    bounds_layout = _register_attention_prefetches(
        reg,
        cfg,
        kv_lens=kv_lens,
        start_seq_idx=start_seq_idx,
        end_seq_idx=end_seq_idx,
        token_end=token_end,
        group_tab=group_tab,
        has_live_views=live_views is not None,
        send_peer=send_peer,
    )
    _register_attention_inputs(
        reg,
        cfg,
        q=q,
        cache_kv_nope=cache_kv_nope,
        cache_kv_rope=cache_kv_rope,
        q_pe=q_pe,
        q_rope_table=q_rope_table,
        live_views=live_views,
        lse=lse,
        send_recv_in=send_recv_in,
        has_out_piece=out_piece is not None,
        out_carried=out_carried,
    )
    _register_attention_outputs(reg, cfg, q=q, out_rows=out_rows, lse_shape=lse_shape)
    _register_attention_scratches(
        reg,
        cfg,
        q=q,
        cache_kv_nope=cache_kv_nope,
        cache_kv_rope=cache_kv_rope,
        head_dim=head_dim,
        lse_shape=lse_shape,
        q_pe=q_pe,
        q_rope_table=q_rope_table,
        send_recv_in=send_recv_in,
    )
    return reg, bounds_layout


def build_live_fused_prep_ref_registry(
    *,
    token_end: jax.Array,
    windows: jax.Array,
    topk_indices: jax.Array,
    chunk_size: int,
    batch_size: int,
    topk: int,
    topk_cap: int,
) -> PallasRefRegistry:
    """Builds the ``PallasRefRegistry`` for ``_capped_window_live_fused_kernel``.

    Preserves the exact ``PrefetchScalarGridSpec`` registration order of
    ``prepare_capped_window_live_fused``.
    """
    num_tokens, padded_pps = windows.shape
    num_windows = topk // topk_cap
    num_chunks = num_tokens // chunk_size
    gpc = chunk_size // batch_size
    region_rows = chunk_size * topk_cap // 128
    row_stride = topk // 128

    hbm = _hbm_block_spec()
    smem = _smem_block_spec()

    reg = PallasRefRegistry()
    reg.add_prefetch("token_end", jnp.reshape(token_end, (1,)).astype(jnp.int32))
    reg.add_input(
        "windows",
        windows,
        spec=pl.BlockSpec((chunk_size, padded_pps), _chunk_index_map),
    )
    reg.add_input(
        "topk",
        topk_indices.reshape(num_tokens * row_stride, 128),
        spec=pl.BlockSpec((chunk_size * row_stride, 128), _chunk_index_map),
    )
    reg.add_output_group(
        "chunk_outs",
        shape_dtypes=[
            jax.ShapeDtypeStruct((num_windows * region_rows, 128), jnp.int32)
        ]
        * num_chunks,
        specs=[hbm] * num_chunks,
    )
    reg.add_output(
        "kv",
        jax.ShapeDtypeStruct((num_tokens // 128, 128), jnp.int32),
        spec=pl.BlockSpec((num_tokens // 128, 128), _zero_2d_index_map),
    )
    reg.add_output(
        "tab",
        jax.ShapeDtypeStruct((2 * num_tokens // batch_size,), jnp.int32),
        spec=smem,
    )
    reg.add_output(
        "live",
        jax.ShapeDtypeStruct((num_chunks,), jnp.int32),
        spec=smem,
    )
    reg.add_scratch(
        "stage",
        pltpu.VMEM((2, num_windows * region_rows, 128), jnp.int32),
    )
    reg.add_scratch(
        "phys",
        pltpu.VMEM((topk_cap // 128 * chunk_size, 128), jnp.int32),
    )
    reg.add_scratch("cw", pltpu.SMEM((num_chunks,), jnp.int32))
    reg.add_scratch("ab", pltpu.SMEM((2 * gpc,), jnp.int32))
    reg.add_scratch("sems", pltpu.SemaphoreType.DMA((2 * num_windows,)))
    return reg
