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
"""The Qwen3.5 MTP draft layer on an attention-only megakernel step.

The MTP draft is one full-attention decoder layer (`mtp.layers.0`) followed by
`mtp.norm`. `Qwen35DraftProgram` packs both as a one-layer attention-only stack
of this folder's `Qwen35` model (`L=1`, `att_every=1`: GQA + MoE, no GDN) and
runs it through the library's pooled step inside the drafter's compiled
programs:

- draft step 0 when every request verifies `K + 1` tokens: all `K + 1` rows of
  each request run attention and write their K/V, and only the rows gathered
  for the next draft token route through the experts (the MoE row mask,
  `MoEDebug.row_mask`);
- the draft loop (steps `1..K-1`): one row per request.

The new K/V rows land in the draft layer's pool at the slots ragged paged
attention writes, so steps on either path read each other's cache. Steps that
don't fit the buckets, mixed prefill steps and position overrides keep the
drafter's own (XLA) programs.

Drafter seam. The program never edits the drafter. `Qwen35Program.attach_draft`
builds and binds this program and hands it to the drafter's
`attach_draft_megakernel(program) -> bool`; the drafter builds its own
megakernel programs around it and routes the draft layer to `forward_layer`
while one of them runs. What the drafter may read: `loop_buckets`,
`verify_buckets`, `padded_rows`, `bound`, `layer_name`, `use_for_step`,
`note_step` and `forward_layer`. `detach_draft_megakernel()` undoes the attach
and `check_draft_megakernel_warm()` raises if a megakernel draft program was
left uncompiled by the warmup.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.adapter import (
    build_jax_step_op,
    get_routed_experts,
    load_vllm_weights,
)
from vllm_torchtpu.megakernel.core import (
    CacheExec,
    ExecSpec,
    MoEDebug,
    Profile,
    make_pooled_step,
    planner,
    with_cache,
)
from vllm_torchtpu.megakernel.core.runtime import stats as mk_stats

from .model import Qwen35

logger = init_logger(__name__)

# Rows per draft step: one per request in the loop, K + 1 = 4 per request at
# step 0. Larger steps keep the XLA draft layer.
DEFAULT_LOOP_BUCKETS: tuple[int, ...] = (4, 8, 16)
DEFAULT_VERIFY_BUCKETS: tuple[int, ...] = (4, 8, 16, 32)
NUM_SPEC_TOKENS = 3
_MOE = "moe"


class DraftUnsupported(ValueError):
    """The MTP drafter, its layer or its cache can't run on this program."""


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise DraftUnsupported(msg)


def _unwrap(x: Any) -> Any:
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def parse_draft_buckets(raw: Any, default: Sequence[int]) -> tuple[int, ...]:
    """Row buckets of a draft step from an option value: a string of ints
    separated by `,` or `:` (`:` inside a top-level `key=value,...` config
    string), an int or a sequence of ints; `None` takes `default`, and `0` /
    `"off"` (or an empty sequence) turns the step off. Each bucket is a
    multiple of 4 in `4..64`."""
    if raw is None:
        return tuple(default)
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("0", "none", "off", "false", ""):
            return ()
        vals = [int(x) for x in text.replace(":", ",").split(",") if x.strip()]
    elif isinstance(raw, int) and not isinstance(raw, bool):
        vals = [] if raw == 0 else [raw]
    else:
        vals = [int(x) for x in raw]
    buckets = tuple(sorted(set(vals)))
    if not all(4 <= b <= 64 and b % 4 == 0 for b in buckets):
        raise ValueError(
            f"draft buckets {raw!r}: each must be a multiple of 4 in 4..64"
        )
    return buckets


def padded(n: int, paddings: Sequence[int]) -> int:
    """`n` rounded up to the runner's token padding (`n` itself past the largest)."""
    return next((p for p in sorted(paddings) if p >= n), n)


def covering_buckets(
    raw: Sequence[int], cap: int, paddings: Sequence[int]
) -> tuple[int, ...]:
    """The buckets of `raw` up to the first one that holds `cap` rows after the
    runner pads them (a step of `cap` rows arrives padded, e.g. 28 requests as
    32 rows)."""
    if not raw:
        return ()
    need = padded(cap, paddings)
    keep = [b for b in raw if b <= need]
    if not keep or keep[-1] < need:
        keep += [b for b in raw if b > need][:1]
    return tuple(keep) or (raw[0],)


def select_bucket(rows: int, buckets: Sequence[int]) -> int:
    """The smallest bucket holding `rows` rows."""
    for b in buckets:
        if b >= rows:
            return b
    raise ValueError(f"{rows} rows exceed the largest draft bucket in {buckets}")


def _fit(x: jax.Array, rows: int) -> jax.Array:
    """The first `rows` rows of `x`, zero-padded when `x` is shorter."""
    if x.shape[0] >= rows:
        return x[:rows]
    return jnp.pad(x, [(0, rows - x.shape[0])] + [(0, 0)] * (x.ndim - 1))


def step_operands(
    ex: ExecSpec,
    hidden: jax.Array,
    block_tables: jax.Array,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    gather_indices: jax.Array | None,
) -> tuple[jax.Array, ...]:
    """One rank's pooled-step operands `(pos, hidden, page_table, active_seqs,
    moe_rows)` of a draft step in the bucket of `ex`.

    Sequence `s` is active when its `query_start_loc` span is `ex.T` rows: its
    rows `s * T .. s * T + T - 1` are those hidden rows, and its tokens sit at
    `seq_lens[s] - T .. seq_lens[s] - 1`, the slots ragged paged attention
    writes. Inactive sequences write no K/V and route no experts, and their
    outputs are discarded. With `gather_indices`, only the gathered rows of
    the active requests route through the experts.
    """
    T, NS, B, M = ex.T, ex.NS, ex.B, ex.M
    knr = seq_lens.shape[0]
    pages = block_tables.reshape(knr, -1)
    live_all = (query_start_loc[1 : knr + 1] - query_start_loc[:knr]) == T
    live = _fit(live_all, NS)
    page_table = jnp.where(live[:, None], _fit(pages, NS), 0).astype(jnp.int32)
    lens = _fit(seq_lens.astype(jnp.int32), NS)
    j = jnp.arange(T, dtype=jnp.int32)
    pos = jnp.where(live[:, None], lens[:, None] - T + j, j).reshape(B)
    live_rows = jnp.repeat(live, T)
    rows = min(hidden.shape[0], M)
    hid = jnp.zeros((M, hidden.shape[1]), hidden.dtype).at[:rows].set(hidden[:rows])
    hid = jnp.where(_fit(live_rows, M)[:, None], hid, jnp.zeros_like(hid))
    if gather_indices is None:
        moe_rows = live_rows.astype(jnp.int32)
    else:
        num_reqs = jnp.sum(live_all.astype(jnp.int32))
        g = gather_indices.astype(jnp.int32)
        g = jnp.where(jnp.arange(g.shape[0]) < num_reqs, g, B)
        moe_rows = jnp.zeros((B,), jnp.int32).at[g].set(1, mode="drop")
    return pos, hid, page_table, live.astype(jnp.int32), moe_rows


def draft_geometry_of(g: Any) -> Any:
    """The draft layer's geometry from the target's `g`: one attention layer,
    EP experts placed contiguously by rank (no redundant copies)."""
    return dataclasses.replace(g, L=1, att_every=1, moe_tp=False, e_slots=0)


def draft_geometry(model_def: type[Qwen35] = Qwen35, hf_config: Any = None) -> Any:
    """The draft layer's geometry of `model_def`'s checkpoint (its default HF
    config unless `hf_config` is given)."""
    return draft_geometry_of(
        model_def.derive_geometry(
            hf_config if hf_config is not None else model_def.default_hf_config
        )
    )


def check_draft_layer(layer: Any, *, g: Any, R: int, tp_rank: int) -> None:
    """Raise `DraftUnsupported` unless `layer` is a full-attention MTP layer
    with FP8 experts placed contiguously by rank and plain KV scales."""
    _need(
        getattr(layer, "layer_type", "full_attention") == "full_attention",
        f"layer_type={getattr(layer, 'layer_type', None)!r}",
    )
    _need(not getattr(layer, "layer_scale", False), "layer_scale")
    attn = layer.self_attn.attn
    for name in ("_k_scale_float", "_v_scale_float"):
        _need(float(getattr(attn, name, 1.0)) == 1.0, f"{name} != 1")
    impl = getattr(attn, "impl", None)
    _need(not getattr(impl, "_bm_kernel", False), "block-major KV pool")
    _need(getattr(impl, "sinks", None) is None, "attention sinks")
    _need(getattr(impl, "sliding_window", None) is None, "sliding window")
    experts = get_routed_experts(layer.mlp)
    w13, w2 = experts.w13_weight, experts.w2_weight
    _need(
        w13.dtype == torch.float8_e4m3fn and w2.dtype == torch.float8_e4m3fn,
        f"expert weights {w13.dtype}/{w2.dtype} are not FP8",
    )
    er = g.E // R
    _need(
        int(w13.shape[0]) == er,
        f"{int(w13.shape[0])} expert slots per rank != {er}",
    )
    for name in ("_ecopy", "_ecopy_cores"):
        _need(getattr(experts, name, None) is None, f"experts.{name} is set")
    expert_map = getattr(experts, "expert_map", None)
    if isinstance(expert_map, torch.Tensor):
        want = torch.full((g.E,), -1, dtype=torch.int64)
        lo = tp_rank * er
        want[lo : lo + er] = torch.arange(er)
        _need(
            torch.equal(expert_map.detach().cpu().to(torch.int64), want),
            "experts are not placed contiguously by rank",
        )
    if getattr(layer.mlp, "_fused_gate_gu_fp8_weight", None) is None:
        gate_dtype = layer.mlp.gate.weight.dtype
        _need(
            gate_dtype in (torch.bfloat16, torch.float32),
            f"unfused router weight dtype {gate_dtype}",
        )


class Qwen35DraftProgram:
    """Runs the Qwen3.5 MTP draft layer through an attention-only megakernel
    step (one rank). `Qwen35Program.attach_draft` builds it."""

    display_name = "Qwen3.5 MTP draft"

    def __init__(
        self,
        layer: Any,
        norm: Any,
        mesh: Any,
        *,
        profile: Profile,
        tp_rank: int,
        dcp_size: int = 1,
        loop_buckets: Sequence[int] = DEFAULT_LOOP_BUCKETS,
        verify_buckets: Sequence[int] = DEFAULT_VERIFY_BUCKETS,
        max_reqs: int = 0,
        token_paddings: Sequence[int] = (),
        model_def: type[Qwen35] = Qwen35,
        geom: Any = None,
        base_exec: ExecSpec | None = None,
        stats_period_s: float = 60.0,
    ) -> None:
        self.mesh = mesh
        self.layer, self.norm = layer, norm
        self.profile = profile
        self.tp_rank = int(tp_rank)
        R = int(mesh.devices.size)
        self.R = R
        self.dcp_size = int(dcp_size)
        self.model_def = model_def
        g = geom if geom is not None else draft_geometry(model_def)
        _need(g.L == 1 and g.NGDN == 0, f"expected one attention layer, got {g}")
        check_draft_layer(layer, g=g, R=R, tp_rank=self.tp_rank)
        self.geom = g
        self.spec = model_def.to_model_spec(g)
        self._paddings = tuple(int(x) for x in token_paddings)
        raw_loop, raw_verify = tuple(loop_buckets), tuple(verify_buckets)
        if max_reqs > 0:
            self.loop_buckets = covering_buckets(raw_loop, max_reqs, self._paddings)
            self.verify_buckets = covering_buckets(
                raw_verify, (NUM_SPEC_TOKENS + 1) * max_reqs, self._paddings
            )
        else:
            self.loop_buckets, self.verify_buckets = raw_loop, raw_verify
        _need(
            bool(self.loop_buckets or self.verify_buckets),
            "both draft bucket lists are empty",
        )
        if base_exec is None:
            b_max = max((*self.loop_buckets, *self.verify_buckets, 8))
            base_exec = ExecSpec(
                R=R, B=b_max, T=1, stack_e=False, cache=CacheExec(paged_kv=True)
            )
        # The MoE row mask (a `moe_rows` kernel input): step 0 routes only the
        # gathered rows through the experts. Other debug settings (`skip`,
        # `scopes`) pass through.
        dbg = dataclasses.replace(
            base_exec.debug,
            modules={**dict(base_exec.debug.modules), _MOE: MoEDebug(row_mask=True)},
        )
        self.base_exec = dataclasses.replace(base_exec, R=R, stack_e=False, debug=dbg)
        self.tk = planner.plan(self.spec, self.base_exec).tk

        with torch.no_grad():
            stack = types.SimpleNamespace(layers=[layer], norm=norm)
            weights = load_vllm_weights(
                model_def, stack, self.base_exec, g=g, tp_rank=self.tp_rank
            )
        self.w_names: tuple[str, ...] = tuple(weights["plan"].w_names)
        self.w_flat: list[torch.Tensor] = list(weights["w_flat"])
        experts = get_routed_experts(layer.mlp)
        # The expert leaves alias the layer's own parameters (no copy); the
        # compiled programs take the parameters themselves (`_leaves`).
        self._expert_idx: dict[str, int] = {}
        idx = 0
        for k in self.w_names:
            val = weights[k]
            n = len(val) if isinstance(val, (tuple, list)) else 1
            if k in ("e13", "e2"):
                _need(n == 1, f"{k}: {n} leaves for a one-layer stack")
                self._expert_idx[k] = idx
            idx += n
        # Leaves per weight name (a tuple of one per layer, or one array).
        self._w_lens = {
            k: len(weights[k]) if isinstance(weights[k], (tuple, list)) else 0
            for k in self.w_names
        }
        _need(
            set(self._expert_idx) == {"e13", "e2"}
            and self.w_flat[self._expert_idx["e13"]].data_ptr()
            == experts.w13_weight.data_ptr()
            and self.w_flat[self._expert_idx["e2"]].data_ptr()
            == experts.w2_weight.data_ptr(),
            "expert leaves do not alias w13_weight / w2_weight",
        )
        packed = [
            w for i, w in enumerate(self.w_flat) if i not in self._expert_idx.values()
        ]
        if packed and packed[0].device.type == "tpu":
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors(packed)

        self.bound = False
        self.layer_name: str = layer.self_attn.attn.layer_name
        self.num_pages = 0
        self.page_size = 0
        self._steps: dict[tuple[int, int, int], tuple[ExecSpec, Any]] = {}
        # Registered before any draft program is built; they only run once
        # `bind` has set the pool.
        self._loop_op: Any = self._build_op(1) if self.loop_buckets else None
        self._verify_op: Any = (
            self._build_op(NUM_SPEC_TOKENS + 1) if self.verify_buckets else None
        )
        self.step_mix_tracker = mk_stats.StepMixTracker(period_s=stats_period_s)
        logger.info(
            "%s megakernel initialized: loop_buckets=%s, verify_buckets=%s, "
            "%.1f MiB packed weights (tp_rank=%d/%d, dcp=%d, profile=%s)",
            self.display_name,
            self.loop_buckets,
            self.verify_buckets,
            sum(t.numel() * t.element_size() for t in packed) / 2**20,
            self.tp_rank,
            R,
            self.dcp_size,
            profile.name,
        )

    # --- bind ---------------------------------------------------------------

    def bind(self) -> None:
        """Bind the draft layer's KV pool; until then no step runs here."""
        if self.bound:
            return
        pool = _unwrap(self.layer.self_attn.attn.kv_cache)
        _need(
            isinstance(pool, torch.Tensor) and pool.dtype == torch.float8_e4m3fn,
            f"draft KV cache {type(pool)} {getattr(pool, 'dtype', None)} is not "
            "an FP8 tensor",
        )
        kvh = max(1, self.geom.NKV // self.R)
        want = (2 * kvh, self.geom.HD // 4, 4)
        _need(
            pool.dim() == 5 and tuple(pool.shape[1:4]) == want,
            f"draft KV cache shape {tuple(pool.shape)} != (pages, *{want}, page)",
        )
        _need(
            pool.is_contiguous() and pool.storage_offset() == 0,
            "draft KV cache is not a base buffer",
        )
        self.num_pages = int(pool.shape[0])
        self.page_size = int(pool.shape[-1])
        self.bound = True
        logger.info(
            "%s megakernel bound to %s (num_pages=%d, page_size=%d)",
            self.display_name,
            self.layer_name,
            self.num_pages,
            self.page_size,
        )

    # --- steps --------------------------------------------------------------

    def step_exec(self, T: int, B: int, width: int) -> ExecSpec:
        """The paged `ExecSpec` of `T`-row sequences in a `B`-row bucket over
        `width`-page block tables (the program's profile picks the tiles)."""
        base = with_cache(
            self.base_exec,
            paged_kv=True,
            num_pages=self.num_pages,
            page_size=self.page_size,
            max_pages_per_seq=width,
            dcp=self.dcp_size,
        )
        base = dataclasses.replace(base, ctx=width * self.page_size)
        ex = self.profile.exec_for(base, B=B, T=T, g=self.geom)
        # Split pool I/O: the KV scatter after the kernel writes the active
        # sequences' new tokens (there is no recurrent state to gather).
        return with_cache(ex, fused_pool_io=False, pool_gather=False)

    def step(self, T: int, B: int, width: int) -> tuple[ExecSpec, Any]:
        """`(ExecSpec, pooled step)` of a draft step; built once per shape."""
        key = (T, B, width)
        if key not in self._steps:
            ex = self.step_exec(T, B, width)
            _need(
                planner.plan(self.spec, ex).tk == self.tk,
                f"t{T}b{B} tiles change the packed weight tiling",
            )
            fn = make_pooled_step(
                self.mesh, self.spec, ex, donate_pools=True, fused_pool_io=False
            )
            self._steps[key] = (ex, fn)
        return self._steps[key]

    def _weights(self, leaves: Sequence[jax.Array]) -> dict[str, Any]:
        """The pooled step's weight dict of the flat leaves (`w_flat` order)."""
        out: dict[str, Any] = {}
        i = 0
        for k in self.w_names:
            n = self._w_lens[k]
            out[k] = tuple(leaves[i : i + n]) if n else leaves[i]
            i += max(n, 1)
        return out

    def draft_fn(self, T: int):
        """The JAX function of `T`-row draft steps over rank-folded global
        operands `(hidden, block_tables, seq_lens, query_start_loc,
        [gather_indices], *w_flat, pool)` (each rank's copy stacked on dim 0);
        returns `(rows, new_pool)`. It picks the bucket and the block-table
        width from the shapes it is traced with."""
        R = self.R
        A = P(self.mesh.axis_names[0])
        buckets = self.loop_buckets if T == 1 else self.verify_buckets
        gather = T > 1
        n_meta = 5 if gather else 4
        n_w = len(self.w_flat)

        def impl(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
            hidden, block_tables, seq_lens = args[:3]
            gidx = args[4] if gather else None
            leaves = args[n_meta : n_meta + n_w]
            pool = args[-1]
            rows = hidden.shape[0] // R
            knr = seq_lens.shape[0] // R
            width = block_tables.size // R // knr
            _need(width * knr * R == block_tables.size, "ragged block tables")
            ex, step = self.step(T, select_bucket(rows, buckets), width)
            _need(rows <= ex.M, f"{rows} rows exceed the {ex.M}-row step")

            def prep(*xs: jax.Array) -> tuple[jax.Array, ...]:
                return step_operands(ex, *xs[:4], xs[4] if gather else None)

            pos, hid, page_table, live, moe_rows = jax.shard_map(
                prep,
                mesh=self.mesh,
                in_specs=(A,) * n_meta,
                out_specs=(P(), A, P(), P(), P()),
                check_vma=False,
            )(*args[:n_meta])
            out, new_pools = step(
                pos, hid, self._weights(leaves), (pool,), page_table, moe_rows, live
            )

            def post(o: jax.Array, *g: jax.Array) -> jax.Array:
                if not g:
                    return o[:rows]
                return jnp.take(
                    o, jnp.clip(g[0].astype(jnp.int32), 0, ex.M - 1), axis=0
                )

            post_args = (out, gidx) if gather else (out,)
            out = jax.shard_map(
                post,
                mesh=self.mesh,
                in_specs=(A,) * len(post_args),
                out_specs=A,
                check_vma=False,
            )(*post_args)
            return out, new_pools[0]

        return impl

    def _build_op(self, T: int) -> Any:
        """The torch callable running `T`-row draft steps (a custom op over
        the library's JAX bridge; donates and updates the pool in place)."""
        A = P(self.mesh.axis_names[0])
        n_meta = 5 if T > 1 else 4
        n_in = n_meta + len(self.w_flat) + 1
        return build_jax_step_op(
            self.draft_fn(T),
            mesh=self.mesh,
            in_specs=(A,) * n_in,
            out_specs=(A, A),
            donate={1: n_in - 1},
            name=f"qwen35_mtp_draft_t{T}",
        )

    def padded_rows(self, rows: int) -> int:
        """The rows a draft step of `rows` rows arrives at after the runner pads it.

        `use_for_step` admits step-0 steps up to `padded_rows((K + 1) *
        num_reqs)` rows, so the drafter's step-0 program compiles every token
        bucket up to the padded count of its widest step.
        """
        return padded(rows, self._paddings)

    def use_for_step(self, attn_metadata: Any, num_rows: int, *, step0: bool) -> bool:
        """Whether a draft step of `num_rows` rows runs on the megakernel.

        `step0` steps must be pure decode-verify steps (`K + 1` rows per
        request), which the caller checks. Counts the decision.
        """
        buckets = self.verify_buckets if step0 else self.loop_buckets
        md = (
            attn_metadata.get(self.layer_name)
            if isinstance(attn_metadata, dict)
            else None
        )
        # A step-0 step arrives padded to the runner's token bucket: its rows
        # past the requests' K + 1 tokens each are inert (no live sequence, no
        # gather index).
        b_max = buckets[-1] if buckets else 0
        max_rows = (
            min(
                b_max,
                self.padded_rows(int(md.seq_lens.shape[0]) * (NUM_SPEC_TOKENS + 1)),
            )
            if (step0 and md is not None)
            else b_max
        )
        ok = bool(
            self.bound
            and buckets
            and num_rows <= max_rows
            and md is not None
            and md.query_start_loc.shape[0] == md.seq_lens.shape[0] + 1
            and md.block_tables.numel() % md.seq_lens.shape[0] == 0
        )
        return self.note_step(f"{'s0' if step0 else 'loop'}_n{num_rows}", ok)

    def note_step(self, key: str, run: bool) -> bool:
        """Count a draft-step decision (`mk_<key>` / `xla_<key>`); log the
        megakernel / XLA mix periodically."""
        mix_key = f"{'mk' if run else 'xla'}_{key}"
        _, summary = self.step_mix_tracker.note_step(mix_key, run)
        if summary is not None and self.tp_rank == 0:
            elapsed, mk_cnt, total, pct, snap = summary
            logger.info(
                "%s megakernel step mix over %.0fs: %d/%d MK (%.1f%%) %s",
                self.display_name,
                elapsed,
                mk_cnt,
                total,
                pct,
                snap,
            )
        return run

    def _leaves(self, layer: Any) -> list[torch.Tensor]:
        experts = get_routed_experts(layer.mlp)
        leaves = list(self.w_flat)
        leaves[self._expert_idx["e13"]] = experts.w13_weight
        leaves[self._expert_idx["e2"]] = experts.w2_weight
        return leaves

    def forward_layer(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        gather_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """`mtp.norm(layer(hidden_states))` for the post-`fc` rows, traced into
        the calling draft program; with `gather_indices` (draft step 0), only
        those rows are returned."""
        from vllm.forward_context import get_forward_context

        attn = layer.self_attn.attn
        md = get_forward_context().attn_metadata[attn.layer_name]
        pool = _unwrap(attn.kv_cache)
        meta = (md.block_tables, md.seq_lens, md.query_start_loc)
        if gather_indices is None:
            out, _ = self._loop_op(hidden_states, *meta, *self._leaves(layer), pool)
        else:
            out, _ = self._verify_op(
                hidden_states,
                *meta,
                gather_indices.to(torch.int32),
                *self._leaves(layer),
                pool,
            )
        return out

    def reset_stats(self) -> None:
        self.step_mix_tracker.reset()


__all__ = [
    "DEFAULT_LOOP_BUCKETS",
    "DEFAULT_VERIFY_BUCKETS",
    "NUM_SPEC_TOKENS",
    "DraftUnsupported",
    "Qwen35DraftProgram",
    "check_draft_layer",
    "covering_buckets",
    "draft_geometry",
    "draft_geometry_of",
    "padded",
    "parse_draft_buckets",
    "select_bucket",
    "step_operands",
]
