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
"""Serving glue for the DeepSeek-V4 decode/verify megakernel.

The runner asks three things of `DSv4Megakernel` each step:

1. `classify` (host counts only, no device sync): is this rank's step all
   decode or all verify windows of exactly q = 1 + K rows, one forward chunk,
   within the largest bucket? An empty rank is `idle`.
2. `decide`: in TP mode (one engine) the local answer stands. In DP mode the
   kernel's collectives span the DP group, so it is all-or-nothing: each rank
   contributes a vote to the DP all-reduce the runner already does, and the
   step runs the megakernel only if no busy rank voted against it. Every rank
   then picks the same bucket from the reduced max token count, and idle ranks
   run the same program with every request inactive.
3. `forward`: run the decided path. The per-rank operand preparation (slicing
   the runner's metadata to the bucket) is compiled apart from the kernel step
   so every rank runs an identical kernel executable whatever its own batch.

With `VLLM_TPU_DSV4_MK_AB=1` eligible steps alternate per bucket between the
megakernel and the stock forward (identically on every rank) and both are
timed with a device sync; calibration only.
"""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
import functools
import json
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import torch
import torch.nn.functional as F
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_callable
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
from vllm_torchtpu.kernels.deepseek_v4_megakernel import weight_map as WM
from vllm_torchtpu.kernels.deepseek_v4_megakernel.diag import vllm_adapter as SD
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MODE_DP = "dp"
MODE_TP = "tp"
_HEAD_OPERANDS = 4  # meta, input_ids, positions, seq_lens


class Unsupported(ValueError):
    pass


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise Unsupported(f"DeepSeek-V4 megakernel: {why}")


# ---------------------------------------------------------------- buckets


def parse_row_buckets(
    raw: str,
    q: int,
    reqs: int,
    max_ctx: int | None = None,
    extra_qs: Sequence[int] = (),
) -> tuple[KI.Bucket, ...]:
    """`VLLM_TPU_DSV4_MK_BUCKETS`: comma-separated global row counts T (the
    sum of every DP rank's active rows), each optionally `T[@q][:ctx]` with its
    own verify query length `q` and longest served context (default `max_ctx`);
    `reqs` slots per rank."""
    ctx_of: dict[tuple[int, int], int | None] = {}
    default_qs = tuple(dict.fromkeys((q, *(int(x) for x in extra_qs if int(x) > 0))))
    for item in (x.strip() for x in raw.split(",")):
        if item:
            rc, _, c = item.partition(":")
            r_str, has_q, q_str = rc.partition("@")
            r = int(r_str)
            ctx = int(c) if c else max_ctx
            qs = (int(q_str),) if has_q else default_qs
            for bq in qs:
                ctx_of[(r, bq)] = ctx
    entries = sorted(ctx_of, key=lambda rq: (rq[0], -rq[1]))
    _need(bool(entries), f"no buckets in VLLM_TPU_DSV4_MK_BUCKETS={raw!r}")
    for r, bq in entries:
        _need(
            r >= bq,
            f"bucket {r} rows holds no {bq}-row request "
            f"(VLLM_TPU_DSV4_MK_BUCKETS={raw!r})",
        )
    return tuple(
        KI.Bucket(q=bq, rows=r, reqs=reqs, max_ctx=ctx_of[(r, bq)]) for r, bq in entries
    )


def _serves_ctx(b: KI.Bucket, ctx: int) -> bool:
    return b.max_ctx is None or ctx <= b.max_ctx


def select_bucket(
    rows: int,
    buckets: Sequence[KI.Bucket],
    ctx: int = 0,
    q: int | None = None,
) -> KI.Bucket | None:
    """The smallest bucket with at least `rows` global rows that serves a
    `ctx`-token context (and query length `q`, when specified), or None."""
    for b in buckets:
        if (q is None or b.q == q) and b.rows >= rows and _serves_ctx(b, ctx):
            return b
    return None


# ------------------------------------------------------- A/B calibration


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


class _LatencySeries:
    """Exact count/mean, and up to `cap` samples (ring) for quantiles."""

    __slots__ = ("cap", "count", "sum_ms", "samples")

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.count = 0
        self.sum_ms = 0.0
        self.samples: list[float] = []

    def add(self, dt_ms: float) -> None:
        if len(self.samples) < self.cap:
            self.samples.append(dt_ms)
        else:
            self.samples[self.count % self.cap] = dt_ms
        self.count += 1
        self.sum_ms += dt_ms

    def summary(self) -> dict[str, float]:
        s = sorted(self.samples)
        return {
            "n": self.count,
            "mean": round(self.sum_ms / max(self.count, 1), 3),
            "p10": round(_percentile(s, 0.10), 3),
            "p50": round(_percentile(s, 0.50), 3),
            "p90": round(_percentile(s, 0.90), 3),
        }


class MKABCalibrator:
    """Deterministic per-bucket megakernel/stock alternation and latency log.

    `choose` alternates only eligible steps, by a per-key counter, so ranks
    that see the same sequence of decisions make the same choices.
    """

    def __init__(self, enabled: bool, period_s: float, cap: int = 4000) -> None:
        self.enabled = enabled
        self.period_s = period_s
        self.cap = cap
        self.counts: dict[str, int] = {}
        self.last: tuple[str, str] | None = None
        self._window: dict[str, dict[str, _LatencySeries]] = {}
        self._cumulative: dict[str, dict[str, _LatencySeries]] = {}
        self._t0 = time.monotonic()

    def reset(self) -> None:
        self.counts.clear()
        self.last = None
        self._window.clear()
        self._cumulative.clear()
        self._t0 = time.monotonic()

    def choose(self, key: str, eligible: bool) -> tuple[bool, str]:
        """`(run_megakernel, mix_key)`; on the A/B stock half of an eligible
        bucket `mk_<b>` the mix key is `ab_stock_<b>`."""
        if not eligible:
            self.last = (key, "stock")
            return False, key
        if not self.enabled:
            self.last = (key, "mk")
            return True, key
        n = self.counts.get(key, 0)
        self.counts[key] = n + 1
        if n % 2 == 0:
            self.last = (key, "mk")
            return True, key
        self.last = (key, "stock")
        return False, "ab_stock_" + key.removeprefix("mk_")

    def _add(self, table, key: str, sub: str, dt_ms: float) -> None:
        series = table.setdefault(key, {})
        if sub not in series:
            series[sub] = _LatencySeries(self.cap)
        series[sub].add(dt_ms)

    def record(
        self, dt_ms: float, now: float | None = None
    ) -> tuple[float, str, str] | None:
        """Record `dt_ms` against the last choice; every `period_s` return
        `(elapsed_s, window_json, cumulative_json)` and start a new window."""
        if not self.enabled or self.last is None:
            return None
        key, path = self.last
        for table in (self._window, self._cumulative):
            self._add(table, key, path, dt_ms)
        now = time.monotonic() if now is None else now
        elapsed = now - self._t0
        if elapsed < self.period_s:
            return None
        out = elapsed, self.format_json(window=True), self.format_json()
        self._window.clear()
        self._t0 = now
        return out

    def summary_dict(self, window: bool = False) -> dict[str, dict[str, Any]]:
        src = self._window if window else self._cumulative
        return {
            k: {sub: src[k][sub].summary() for sub in sorted(src[k])}
            for k in sorted(src)
        }

    def format_json(self, window: bool = False) -> str:
        return json.dumps(self.summary_dict(window=window), sort_keys=True)


# -------------------------------------------------------- step decisions

IDLE = "idle"
ELIGIBLE = "eligible"
STOCK = "stock"


@dataclasses.dataclass(frozen=True)
class LocalStep:
    kind: str  # IDLE, ELIGIBLE or STOCK
    reason: str
    rows: int = 0  # global rows this rank's requests occupy in the step
    ctx: int = 0  # longest request context incl. this step's window
    q: int = 0  # non-zero when the step uses a non-default verify width

    @property
    def vote(self) -> int:
        """This rank's veto: 1 sends the whole DP group to stock."""
        return int(self.kind == STOCK)


def dp_entries(local: LocalStep | None, dp_rank: int, dp_size: int) -> list[int]:
    """This rank's entries of the DP all-reduce (MAX): its veto, its longest
    context, then its active rows in its own slot of a `dp_size` vector, so
    the reduced vector holds every rank's rows (summed on the host; MAX
    cannot sum) and the group's longest context."""
    rows = [0] * dp_size
    if local is None:
        return [0, 0, *rows]
    rows[dp_rank] = local.rows
    return [local.vote, local.ctx, *rows]


@dataclasses.dataclass(frozen=True)
class DPReduced:
    """What every rank reads back from the DP all-reduce."""

    max_tokens: int
    max_chunks: int
    any_prefill: bool
    any_veto: bool
    total_rows: int  # sum of every rank's active rows
    max_ctx: int = 0  # the group's longest request context
    q: int = 0  # non-zero when the step uses a non-default verify width


@dataclasses.dataclass(frozen=True)
class StepDecision:
    run: bool
    key: str  # step-mix key: mk_*, ab_stock_* or a stock_* reason
    bucket: KI.Bucket | None = None
    idle: bool = False
    parity: bool = False  # run the stock forward too, then compare (run only)


def classify_local_step(
    *,
    num_reqs: int,
    num_decode: int,
    num_windowed: int,
    num_static_verify: int,
    num_tokens: int,
    num_chunks: int,
    has_encoder_inputs: bool,
    q: int,
    buckets: Sequence[KI.Bucket],
    max_reqs: int,
    max_seq_len: int = 0,
    step_q: int | None = None,
) -> LocalStep:
    """Classify this rank's step from the runner's host-side counts.

    The counts are `_reorder_batch_for_rpa`'s: `num_decode` 1-token requests
    lead, then `num_static_verify` verify windows of exactly q tokens, up to
    `num_windowed`; anything after is prefill. When `step_q` is given (for
    width-aware DSpark steps with a uniform alternate verify width), the step
    is matched against buckets of width `step_q`.
    """
    if num_reqs == 0 or num_tokens == 0:
        return LocalStep(IDLE, IDLE)
    if has_encoder_inputs:
        return LocalStep(STOCK, "stock_encoder")
    if num_windowed < num_reqs:
        return LocalStep(STOCK, "stock_prefill")
    if num_chunks != 1:
        return LocalStep(STOCK, "stock_chunks")
    eff_q = q if step_q is None else step_q
    if step_q is not None and step_q != q:
        uniform = num_decode == 0 and any(b.q == step_q for b in buckets)
    else:
        uniform = num_decode == num_reqs if q == 1 else num_static_verify == num_reqs
    if not uniform:
        return LocalStep(STOCK, "stock_mixed_qlen")
    if num_tokens != num_reqs * eff_q:
        raise AssertionError(
            f"{num_reqs} requests of {eff_q} rows scheduled {num_tokens} tokens"
        )
    if num_reqs > max_reqs:
        return LocalStep(STOCK, "stock_own_reqs")
    # max_seq_len: the longest request's tokens incl. this step's window
    multi_q = any(b.q != q for b in buckets)
    want_q = eff_q if (multi_q or (step_q is not None and step_q != q)) else None
    if not any(
        (want_q is None or b.q == want_q) and _serves_ctx(b, max_seq_len)
        for b in buckets
    ):
        return LocalStep(STOCK, "stock_ctx")
    rows = num_reqs * KI.request_block_rows(eff_q)
    if select_bucket(rows, buckets, max_seq_len, q=want_q) is None:
        return LocalStep(STOCK, f"stock_rows{rows}")
    out_q = eff_q if (step_q is not None and step_q != q) else 0
    return LocalStep(ELIGIBLE, ELIGIBLE, rows, max_seq_len, q=out_q)


def decide_step(
    local: LocalStep,
    reduced: DPReduced | None,
    buckets: Sequence[KI.Bucket],
) -> tuple[str, KI.Bucket | None]:
    """`(key, bucket)`; bucket None means stock. `reduced` is None in TP mode.

    In DP mode every input to the bucket choice is DP-reduced, so all ranks
    pick the same bucket: the smallest T holding the group's total rows that
    serves the group's longest context.
    """
    if reduced is None:
        if local.kind != ELIGIBLE:
            return local.reason, None
        rows, ctx, step_q = local.rows, local.ctx, local.q
    else:
        if reduced.any_veto:
            if local.kind == STOCK:
                return local.reason, None
            return (
                "stock_dp_prefill" if reduced.any_prefill else "stock_dp_peer"
            ), None
        if reduced.max_tokens == 0 or reduced.max_chunks != 1:
            return "stock_dp_idle", None
        rows, ctx, step_q = reduced.total_rows, reduced.max_ctx, reduced.q or local.q
    want_q = step_q if step_q > 0 else None
    bucket = select_bucket(rows, buckets, ctx, q=want_q)
    if bucket is None:
        if reduced is None:
            raise AssertionError(f"no bucket for {rows} rows after an eligible vote")
        if select_bucket(rows, buckets, q=want_q) is not None:
            return f"stock_ctx_rows{rows}", None
        return f"stock_total_rows{rows}", None
    return bucket.key, bucket


def reachable_out_rows(bucket: KI.Bucket, token_paddings: Sequence[int]) -> list[int]:
    """The runner's padded row counts for the steps `bucket` serves (the
    first token padding holding each whole-request row count)."""
    paddings = sorted(token_paddings)
    out = set()
    for n in range(1, bucket.reqs + 1):
        out.update(
            p for p in paddings[bisect.bisect_left(paddings, n * bucket.q) :][:1]
        )
    return sorted(out)


# ---------------------------------------------------------- cache binding


def _live(t: Any, what: str) -> torch.Tensor:
    _need(
        isinstance(t, torch.Tensor) and t.numel() > 0,
        f"{what}: expected a bound cache tensor, got {type(t).__name__}",
    )
    return t


def layer_cache_roles(attn: Any, layer: int) -> tuple[str, int, dict[str, Any]]:
    """`(kind, compress_ratio, {role: (kv-cache layer name, tensor)})` of one
    decoder layer's attention, read the way the stock ops read them."""
    ratio = int(attn.compress_ratio)
    swa = attn.swa_cache_layer
    roles: dict[str, Any] = {
        "swa": (swa.prefix, _live(swa.kv_cache, f"layer {layer} swa")),
    }
    if ratio <= 1:  # SWA-only (DSpark draft layers): the window cache alone
        return KI.SWA, ratio, roles
    kind = KI.CSA if ratio == 4 else KI.HCA
    state = attn.compressor.state_cache
    main_name = attn.custom_prefix  # its static_forward_context key
    if kind == KI.CSA:
        entry = attn.kv_cache
        _need(
            isinstance(entry, tuple) and len(entry) == 2,
            f"layer {layer}: CSA cache is not a (nope, rope) pair",
        )
        # The runner binds a CSA compressor state to its compressed-KV
        # layer's entry, i.e. the same (nope, rope) pair; the state rows live
        # in the NoPE array (kv_cache_dsv4: state and KV groups never share a
        # block id).
        st = state.kv_cache
        _need(
            isinstance(st, tuple) and len(st) == 2 and st[0] is entry[0],
            f"layer {layer} state: expected the layer's (nope, rope) pair, "
            f"got {type(st).__name__}",
        )
        roles["state"] = (state.prefix, _live(st[0], f"layer {layer} state"))
        roles["kv"] = (main_name, _live(entry[0], f"layer {layer} kv"))
        roles["kv_rope"] = (main_name, _live(entry[1], f"layer {layer} kv_rope"))
        idx = attn.indexer
        roles["indexer_k"] = (
            idx.k_cache.prefix,
            _live(idx.k_cache.kv_cache, f"layer {layer} indexer_k"),
        )
        ist = idx.compressor.state_cache
        roles["indexer_state"] = (
            ist.prefix,
            _live(ist.kv_cache, f"layer {layer} indexer_state"),
        )
    else:
        roles["state"] = (state.prefix, _live(state.kv_cache, f"layer {layer} state"))
        roles["kv"] = (main_name, _live(attn.kv_cache, f"layer {layer} kv"))
    return kind, ratio, roles


@dataclasses.dataclass
class BoundCaches:
    layers: tuple[KI.LayerCaches, ...]
    arrays: list[torch.Tensor]  # each distinct array once, in operand order
    group_ids: list[int]  # KV-cache group id per block-table operand
    group_layer_names: list[str]  # a layer of each group, to find its metadata


def bind_layer_caches(
    decoder_layers: Sequence[Any], group_of: dict[str, int]
) -> BoundCaches:
    """Collect every cache the layers use, each distinct array once.

    Arrays are deduplicated by storage: the runner overlays SWA caches and
    compressor states onto other layers' arrays, and an array donated twice
    would be two aliasing outputs.
    """
    arrays: list[torch.Tensor] = []
    array_of_ptr: dict[int, int] = {}
    per_layer = []
    for i, layer in enumerate(decoder_layers):
        kind, ratio, roles = layer_cache_roles(layer.attn, i)
        refs = {}
        for role, (name, t) in roles.items():
            _need(name in group_of, f"layer {i} {role}: {name!r} is in no KV group")
            ptr = t.data_ptr()
            if ptr not in array_of_ptr:
                array_of_ptr[ptr] = len(arrays)
                arrays.append(t)
            refs[role] = (array_of_ptr[ptr], group_of[name], name)
        per_layer.append((kind, ratio, refs))
    group_ids = sorted({gid for _, _, refs in per_layer for _, gid, _ in refs.values()})
    operand_of_gid = {gid: i for i, gid in enumerate(group_ids)}
    group_layer_names = [""] * len(group_ids)
    layers = []
    for kind, ratio, refs in per_layer:
        cache_refs = {}
        for role, (arr, gid, name) in refs.items():
            group_layer_names[operand_of_gid[gid]] = name
            cache_refs[role] = KI.CacheRef(array=arr, group=operand_of_gid[gid])
        layers.append(KI.LayerCaches(kind=kind, compress_ratio=ratio, **cache_refs))
    return BoundCaches(tuple(layers), arrays, group_ids, group_layer_names)


# --------------------------------------------------------- weight operands


def _attr(owner: Any, path: str, what: str) -> Any:
    for name in path.split("."):
        if name.isdigit():  # a ModuleList index
            owner = owner[int(name)]
            continue
        _need(
            hasattr(owner, name), f"{what}.{path}: no attribute {name!r}{_hint(name)}"
        )
        owner = getattr(owner, name)
    return owner


def _hint(name: str) -> str:
    if name.startswith("_tpu_fused_"):
        return (
            " (the fused EP MoE v2 op is not armed: it needs "
            "USE_MOE_FUSED_EP_KERNEL=1, MOE_FUSED_EP_ENABLE_W4A8=1 and expert "
            "parallelism; the grouped-matmul layout packs fp4 differently)"
        )
    return ""


def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


def read_served(
    owner: Any, spec: dict[str, WM.Served], what: str
) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
    """The tensors `spec` names under `owner`, validated; and per key whether
    it is stored [n, k] (only where `Served.nk_ok`). Nothing is copied."""
    tensors, transposed = {}, {}
    for key, sv in spec.items():
        t = _attr(owner, sv.path, what)
        where = f"{what}.{sv.path}"
        _need(isinstance(t, torch.Tensor), f"{where}: {type(t).__name__}, not a tensor")
        shape = tuple(t.shape)

        def fits(want):
            return len(want) == len(shape) and all(
                a in (-1, b) for a, b in zip(want, shape)
            )

        flipped = sv.nk_ok and not fits(sv.shape) and fits(sv.shape[::-1])
        _need(
            (fits(sv.shape) or flipped) and _dtype_name(t) == sv.dtype,
            f"{where}: served {shape} {_dtype_name(t)}, expected {sv.shape} {sv.dtype}",
        )
        _need(t.is_contiguous(), f"{where}: not contiguous (stride {t.stride()})")
        tensors[key], transposed[key] = t, flipped
    return tensors, transposed


def read_draft_served(
    draft_model: Any, g: WM.Geometry, n_aux: int
) -> tuple[
    dict[str, torch.Tensor], list[dict[str, torch.Tensor]], list[dict[str, bool]]
]:
    """The DSpark draft's served tensors (`WM.served_draft_spec`), validated:
    model-level, per draft layer, and the per-layer [n, k] flags. `draft_model`
    is the proposer's `draft_model` (`DSparkDeepseekV4ForCausalLM`)."""
    model = draft_model.model
    layers = list(model.layers)
    local_experts = int(layers[0].ffn.experts.routed_experts.w13_weight.shape[0])
    glob_spec, layer_specs = WM.served_draft_spec(g, len(layers), n_aux, local_experts)
    glob, _ = read_served(model, glob_spec, "draft")
    served, flags = [], []
    for i, (layer, spec) in enumerate(zip(layers, layer_specs)):
        _need(
            int(layer.attn.compress_ratio) <= 1,
            f"draft layer {i}: compress_ratio {layer.attn.compress_ratio}, not SWA",
        )
        t, f = read_served(layer, spec, f"draft.layers.{i}")
        served.append(t)
        flags.append(f)
    return glob, served, flags


def build_weight_operands(
    causal_lm: Any,
    spec: KI.ModelSpec,
    rank: int,
    run_jax: Callable[..., torch.Tensor],
    dense: Any = None,
    logits: bool = False,
    shared: dict[str, torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """This rank's weight operands in `weight_manifest(spec, logits=)` order
    (`shared`: the "placement" operands, by name):
    served tensors zero-copy, load-time copies through `run_jax(name, fn,
    *tensors)` one operand at a time. `rank` is the mesh index; under vLLM TP
    (`spec.attn_tp`) the served tensors are its TP shard (`WM.tp_local`)."""
    g = WM.Geometry.from_hf(causal_lm.config, ratios=spec.compress_ratios)
    if spec.attn_tp > 1:
        _need(not logits, "logits mode under vLLM TP (the lm_head is a TP shard)")
        g = WM.tp_local(g, spec.attn_tp)
    layers = list(causal_lm.model.layers)
    served, flags = [], []
    for i, layer in enumerate(layers):
        # a static expert placement may give layers different slot counts
        local_experts = int(layer.ffn.experts.routed_experts.w13_weight.shape[0])
        t, f = read_served(
            layer, WM.served_layer_spec(g, i, local_experts), f"layers.{i}"
        )
        served.append(t)
        flags.append(f)
    glob, _ = read_served(
        causal_lm, WM.served_global_spec(g, spec.attn_tp > 1), "model"
    )
    shard = WM.head_shard(rank, spec.num_ranks, spec.attn_tp)
    builds = WM.plan_operands(spec, flags, g, shard, dense,
                              manifest=functools.partial(KI.weight_manifest,
                                                         logits=logits))  # fmt: skip
    return _materialize(builds, glob, served, run_jax, shared)


def build_draft_weight_operands(
    draft_model: Any,
    spec: KI.ModelSpec,
    g: WM.Geometry,
    n_aux: int,
    rank: int,
    run_jax: Callable[..., torch.Tensor],
    dense: Any = None,
    logits: bool = False,
    shared: dict[str, torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """This rank's DSpark draft-step weight operands, in
    `KI.draft_weight_manifest(spec, logits=)` (= `step_impl.draft_weight_names`)
    order. `spec` is the draft's ModelSpec, `g` the target's geometry;
    `shared` operands are taken as given (the target's lm_head shard)."""
    glob, served, flags = read_draft_served(draft_model, g, n_aux)
    gd = WM.draft_geometry(g, len(served))
    manifest = functools.partial(KI.draft_weight_manifest, logits=logits)
    builds = WM.plan_operands(spec, flags, gd, WM.Shard(rank, spec.num_ranks), dense,
                              manifest=manifest)  # fmt: skip
    return _materialize(builds, glob, served, run_jax, shared or {})


def _materialize(
    builds: Sequence[WM.Build],
    glob: dict[str, torch.Tensor],
    served: list[dict[str, torch.Tensor]],
    run_jax: Callable[..., torch.Tensor],
    shared: dict[str, torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """The operand tensors of `builds`: served tensors zero-copy (default
    layout), load-time copies through `run_jax(name, fn, *tensors)`,
    `shared` ones by entry name as given."""
    out = []
    # A rotary table served once under two names (cos_sin, comp_cos_sin) is
    # packed once.
    packed_rope: dict[int, torch.Tensor] = {}
    # served tensors the runner relaid out, copied back to the default layout
    relaid: list[str] = []
    for b in builds:
        if shared and b.entry.name in shared:
            out.append(shared[b.entry.name])
            continue
        tensors = [
            glob[x.key] if x.layer is None else served[x.layer][x.key]
            for x in b.sources
        ]
        if b.fn is None:
            t = tensors[0]
            got = settled_layout(t)
            nbytes = t.numel() * t.element_size()
            if got is False or (got is None and nbytes <= _VIEW_COPY_MAX_BYTES):
                t = t.cpu().to(t.device, copy=True)  # fresh: default layout
                relaid.append(b.entry.name)
        elif b.entry.rule == "rope" and tensors[0].data_ptr() in packed_rope:
            t = packed_rope[tensors[0].data_ptr()]
        elif not tensors:
            a = jax.eval_shape(b.fn)
            t = torch.zeros(
                a.shape, dtype=getattr(torch, a.dtype.name), device=_device(served)
            )
        else:
            t = run_jax(f"dsv4_mk_w_{b.entry.name}", b.fn, *tensors)
            if b.entry.rule == "rope":
                packed_rope[tensors[0].data_ptr()] = t
        _need(
            WM.entry_matches(b.entry, t.shape, _dtype_name(t)),
            f"operand {b.entry.name}: built {tuple(t.shape)} {t.dtype}, manifest "
            f"wants {b.entry.shape} {b.entry.dtype}",
        )
        out.append(t)
    if relaid:
        logger.info(
            "DeepSeek-V4 megakernel weights: %d served operands are views or "
            "not in the default device layout (runner relayout), copied: %s",
            len(relaid),
            relaid,
        )
    return out


def _device(served: list[dict[str, torch.Tensor]]) -> torch.device:
    return next(iter(served[0].values())).device


def torch_tpu_run_jax(name: str, fn: Callable, *tensors: torch.Tensor) -> torch.Tensor:
    """Run `fn` on this rank's device over torch tensors (one-time, eager).

    The JaxCallable is called directly, not registered as a torch op: a
    packed stream reads one tensor per layer (884 for xproj over 61 layers),
    past the dispatcher's 64-argument schema limit."""
    from torch_tpu._internal.pallas import pallas

    args = [f"x{i}" for i in range(len(tensors))]
    src = (
        f"def {name}({', '.join(a + ': jax.Array' for a in args)}) -> jax.Array:\n"
        f"    return _fn({', '.join(args)})\n"
    )
    ns: dict[str, Any] = {"jax": jax, "_fn": fn}
    exec(compile(src, f"<{name}>", "exec"), ns)  # noqa: S102 - arity is data
    call = pallas.custom_jax_kernel(
        ns[name], f"dsv4_mk_weights::{name}", static_argnums=(), warn_deprecated=False
    )
    return call(*tensors)


# ---------------------------------------------------------- runner access


def _parity_config(model: Any) -> SD.ParityConfig | None:
    """`VLLM_TPU_DSV4_MK_PARITY*`, with this rank's lm_head vocabulary shard."""
    if not (envs.VLLM_TPU_DSV4_MK_PARITY or envs.VLLM_TPU_DSV4_MK_PARITY_WARMUP):
        return None
    head = model.lm_head
    shard = head.shard_indices
    return SD.ParityConfig(
        head=SD.LocalHead(
            weight=head.weight,
            vocab_start=int(shard.org_vocab_start_index),
            num_tokens=int(shard.num_org_elements),
        ),
        served_budget=envs.VLLM_TPU_DSV4_MK_PARITY,
        every=envs.VLLM_TPU_DSV4_MK_PARITY_EVERY,
        warmup_budget=envs.VLLM_TPU_DSV4_MK_PARITY_WARMUP,
        tol=envs.VLLM_TPU_DSV4_MK_PARITY_TOL,
        pairs=tuple(
            tuple(p.split(",")) for p in envs.VLLM_TPU_DSV4_MK_PARITY_PAIR.split(";")
        ),
    )


def _find_backbone(model: Any) -> Any:
    backbone = model.model
    _need(
        hasattr(backbone, "layers") and hasattr(backbone, "aux_hidden_state_layers"),
        f"{type(model).__name__}.model is not a DeepseekV4Model",
    )
    return backbone


def _current_attn_metadata() -> dict[str, Any]:
    from vllm.forward_context import get_forward_context

    md = get_forward_context().attn_metadata
    if not isinstance(md, dict):
        raise AssertionError(f"per-layer attention metadata expected, got {type(md)}")
    return md


def default_layout(t: torch.Tensor) -> bool | None:
    """Whether `t` has XLA's default device layout for its shape, the layout
    the exported step is compiled for (None: torch_tpu reports no layout,
    e.g. a view or a not yet materialized result). Host tensors: True.

    A served table the runner relaid out (`relayout_rope_caches`,
    `relayout_hash_tables`: row-major `{1,0}`) fails the step's compile when
    the kernel reads it; a non-default cache would be copied around the
    kernel, its in-place writes landing in the copy."""
    if t.device.type != "tpu":
        return True
    from torch_tpu._internal.compile import tpu_torch_compile as ttc

    got = ttc.get_device_layout_if_materialized(t)
    if got is None:
        return None
    return _layout_key(got) == _layout_key(ttc.get_default_layout(t.dtype, t.shape))


def settled_layout(t: torch.Tensor) -> bool | None:
    """`default_layout` after materializing `t` if torch_tpu reported none;
    still None means a view, which torch_tpu copies into a temporary on every
    kernel call (a dtype view of a whole [24, 7168, 3072] fp4 tensor: 11 ms a
    call, and in-place writes lost; view_operand_probe p5_fp4)."""
    got = default_layout(t)
    if got is None:
        _sync([t])
        got = default_layout(t)
    return got


# a served operand reported as a view is copied up to this size; past it the
# copy would cost real HBM and bind() refuses the operand instead
_VIEW_COPY_MAX_BYTES = 64 * 2**20


def _layout_key(layout: Any) -> tuple:
    minor_to_major, tiles, bits = layout
    return tuple(minor_to_major), tuple(tuple(x) for x in tiles), int(bits)


def _sync(tensors: Sequence[torch.Tensor] | None) -> None:
    from vllm_torchtpu.utils import synchronize_device, synchronize_tensors

    synchronize_tensors(tensors)
    synchronize_device()


def _mesh_and_rank(mode: str) -> tuple[jax.sharding.Mesh, int, int]:
    """The EP mesh (ascending device id), this rank's index in it, and the EP
    rank whose routed experts this rank's served expert tensors hold.

    In DP mode it spans the DP group, in TP mode the TP group; the kernel
    reads both from `meta`, never from the mesh position of a constant. The
    two differ on hosts whose device ids are not in rank order (every tpu7x
    host; see `ep_mesh.ep_rank_order`).
    """
    from vllm_torchtpu.distributed.ep_mesh import (
        build_ep_mesh,
        ep_mesh_index,
        ep_rank_order,
    )

    mesh = build_ep_mesh()
    _need(mesh is not None, f"{mode} mode needs expert parallelism over the ranks")
    rank = ep_mesh_index()
    order = ep_rank_order()
    assert rank is not None and order is not None
    return mesh, int(rank), int(order[rank])


@dataclasses.dataclass(frozen=True)
class GroupMember:
    """One EP rank as `_engine_mesh_and_rank` gathers it."""

    key: int  # dp_rank * attn_tp + tp_rank: its mesh index
    device_id: int
    ep_rank: int  # whose routed experts it serves


def engine_mesh_order(members: Sequence[GroupMember]) -> list[int]:
    """Device ids by mesh index (= `key`), refusing unless the keys are
    exactly 0..R-1 and the device ids distinct."""
    keys = sorted(m.key for m in members)
    _need(keys == list(range(len(members))), f"engine mesh keys {keys}")
    ids = [m.device_id for m in members]
    _need(len(set(ids)) == len(ids), f"two EP ranks share a device id: {ids}")
    return [m.device_id for m in sorted(members, key=lambda m: m.key)]


def _engine_mesh_and_rank(attn_tp: int) -> tuple[jax.sharding.Mesh, int, int]:
    """vLLM TP inside DP engines: a mesh over the EP group in which TP rank j
    of DP engine g sits at index g * attn_tp + j (the kernel's attention groups
    are consecutive mesh indices), this rank's index, and its EP rank."""
    import numpy as np
    from jax.sharding import Mesh
    from torch_tpu._internal.pallas import get_pallas_mesh
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME, get_ep_group
    from vllm_torchtpu.distributed.mesh_utils import _get_tpu_global_device_id

    ep, tp, dp = get_ep_group(), get_tp_group(), get_dp_group()
    _need(ep is not None, "DP mode needs expert parallelism over the ranks")
    _need(
        int(tp.world_size) == attn_tp and int(ep.world_size) == attn_tp * dp.world_size,
        f"EP {ep.world_size} is not TP {tp.world_size} x DP {dp.world_size}",
    )
    _need(
        int(ep.world_size) == int(torch.distributed.get_world_size()),
        "the EP group is not the whole world (get_pallas_mesh spans the world)",
    )
    mine = GroupMember(
        key=int(dp.rank_in_group) * attn_tp + int(tp.rank_in_group),
        device_id=_get_tpu_global_device_id(),
        ep_rank=int(ep.rank_in_group),
    )
    members: list[Any] = [None] * int(ep.world_size)
    torch.distributed.all_gather_object(members, mine, group=ep.cpu_group)
    device_ids = engine_mesh_order(members)
    by_id = {
        int(d.id): d for d in get_pallas_mesh(axis_names=(EP_AXIS_NAME,)).devices.flat
    }
    _need(
        sorted(by_id) == sorted(device_ids),
        f"pallas mesh devices {sorted(by_id)} vs the EP group's {device_ids}",
    )
    mesh = Mesh(np.asarray([by_id[d] for d in device_ids]), axis_names=(EP_AXIS_NAME,))
    logger.info(
        "DeepSeek-V4 megakernel engine mesh: device ids %s (index g*%d + j), "
        "this rank index %d, EP rank %d",
        device_ids,
        attn_tp,
        mine.key,
        mine.ep_rank,
    )
    return mesh, mine.key, mine.ep_rank


def _fit_rows(x: torch.Tensor, rows: int, dim: int = 0) -> torch.Tensor:
    n = x.shape[dim]
    if n >= rows:
        return x.narrow(dim, 0, rows)
    pad = [0, 0] * (x.ndim - 1 - dim) + [0, rows - n]
    return F.pad(x, pad)


def copied_bytes(model: Any, operands: Sequence[torch.Tensor]) -> int:
    """Bytes of the operands that are not storage of a served tensor."""
    served = {t.data_ptr() for t in model.parameters()}
    served |= {t.data_ptr() for t in model.buffers()}
    return sum(
        t.numel() * t.element_size() for t in operands if t.data_ptr() not in served
    )


def export_step(
    fn: Callable,
    mesh: Any,
    n_in: int,
    n_out: int,
    fn_name: str,
    op_name: str,
    donate_argnums: tuple[int, ...] = (),
) -> Any:
    """`fn` (per-rank operands -> outputs) under a shard_map over the mesh
    axis, exported as a torch_tpu JaxCallable. Each rank's operands are its
    shard of a global array concatenated along axis 0, so the body sees
    per-rank shapes unmodified."""
    axis = P(mesh.axis_names[0])
    in_specs, out_specs = (axis,) * n_in, (axis,) * n_out

    def body(*args):
        return tuple(fn(*args))

    smap = jax.shard_map(
        body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )
    args = [f"x{i}" for i in range(n_in)]
    ret = "tuple[" + ", ".join(["jax.Array"] * n_out) + "]"
    src = (
        f"def {fn_name}({', '.join(a + ': jax.Array' for a in args)}) -> {ret}:\n"
        f"    return _smap({', '.join(args)})\n"
    )
    ns: dict[str, Any] = {"jax": jax, "_smap": smap}
    exec(compile(src, f"<{fn_name}>", "exec"), ns)  # noqa: S102 - arity is data
    return sharded_jax_callable(
        op_name,
        ns[fn_name],
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
        donate_argnums=donate_argnums,
    )


def inplace_call(flat: Any, n_head: int, direct: bool) -> Callable:
    """call(head, weights, caches) -> outputs of an in-place exported step."""
    if direct:
        return DirectStepCall(flat, n_head)

    def call(head, weights, caches):  # a stand-in callable (unit tests)
        return flat(*head, *weights, *caches)

    return call


class DirectStepCall:
    """The in-place step's exported kernel, called without the JaxCallable
    front end after the first call.

    JaxCallable.__call__ rebuilds a string key from the shape and dtype of
    every operand (~900 here) on each call to find the registered kernel; that
    and its per-operand checks cost ~0.8 ms of host time per step. A bucket's
    operands never change shape (prep emits fixed shapes, the weight and cache
    lists are bound once), so the first call goes through `flat` -- export,
    registration, the output placeholders -- and later calls dispatch the
    registered kernel directly, checking only what can change: the head
    operands' shapes and that the weight / cache lists are the bound ones."""

    def __init__(self, flat: Any, n_head: int) -> None:
        self.flat, self.n_head = flat, n_head
        self._entry: tuple | None = None

    def __call__(self, head: Sequence[torch.Tensor], weights: list, caches: list):
        args = (*head, *weights, *caches)
        if self._entry is None:
            outs = self.flat(*args)
            self._bind(head, weights, caches, args)
            return outs
        fp, out_shapes, out_tree, devices, sig, w_ref, c_ref = self._entry
        _need(
            weights is w_ref and caches is c_ref,
            "megakernel step called with weight / cache lists it was not bound to",
        )
        _need(
            tuple((t.shape, t.dtype) for t in head) == sig,
            f"megakernel step head operands changed shape: {sig}",
        )
        from torch_tpu._internal.pallas import tpu_torch_pallas

        results = tpu_torch_pallas.call_custom_kernel(
            self.flat.name,
            fp,
            inputs=list(args),
            output_shapes=out_shapes,
            donate_argnums=self.flat.donate_argnums,
            device_assignment=devices,
        )
        return out_tree.unflatten(results)

    def _bind(self, head, weights, caches, args) -> None:
        from torch_tpu._internal.pallas import pallas as pallas_impl

        flat = self.flat
        _need(not flat.donate_argnums, "direct dispatch is for the in-place step")
        key = pallas_impl._get_kernel_invocation_key(
            flat.trace_key, args, {}, flat.static_argnums
        )
        out_shapes, out_tree = flat.output_shapes[key]
        self._entry = (
            flat.kernel_key_to_mlir_fingerprint[key],
            out_shapes,
            out_tree,
            [int(d.id) for d in flat.mesh.devices.flat],
            tuple((t.shape, t.dtype) for t in head),
            weights,
            caches,
        )


def placement_operands(
    spec: KI.ModelSpec, ep_rank: int, device: Any, placement: Any = None
) -> tuple[KI.ModelSpec, dict[str, torch.Tensor]]:
    """The static expert placement (`expert_placement.current()` unless
    given): `spec` with its moe_placement / expert_slots, and this rank's
    rows of `moe_phase.placement_tables` (EP rank `ep_rank`) as the
    "placement" weight operands. No placement: `spec` as is, no operands."""
    from vllm_torchtpu import expert_placement
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_phase

    placement = placement or expert_placement.current()
    if placement is None:
        return spec, {}
    E, R = spec.n_routed_experts, spec.num_ranks
    _need(
        (placement.experts, placement.shards) == (E, R),
        f"expert placement for {placement.experts} experts over {placement.shards} "
        f"ranks, the model has {E} over {R}",
    )
    held, copy = moe_phase.placement_tables(placement, spec.num_layers, R, E)
    spec = dataclasses.replace(
        spec,
        moe_placement="perm" if copy is None else "replicas",
        expert_slots=int(held.shape[2]),
    )
    rows = {"moe_held": held} | ({} if copy is None else {"moe_copy": copy})
    return spec, {k: torch.from_numpy(v[ep_rank]).to(device) for k, v in rows.items()}


class RunTrace:
    """stderr trace of the first `budget` megakernel executions of this rank
    (target and draft, busy and idle), tagged with the DP step count. With
    `sync`, each is followed by a device sync and its end is traced: a rank
    whose last line is a begin is stuck in that kernel."""

    def __init__(self, rank: int, budget: int, sync: bool) -> None:
        self.rank, self.left, self.sync = rank, budget, sync
        self.step = 0  # DP decisions so far
        self.n = 0

    @contextlib.contextmanager
    def __call__(self, what: str):
        if self.left <= 0:
            yield
            return
        self.left -= 1
        self.n += 1
        tag = f"[dsv4-megakernel rank {self.rank}] run {self.n} step {self.step} {what}"
        sys.stderr.write(f"{tag} begin\n")
        sys.stderr.flush()
        t0 = time.perf_counter()
        yield
        if self.sync:
            _sync(None)
            ms = (time.perf_counter() - t0) * 1e3
            sys.stderr.write(f"{tag} end {ms:.1f} ms\n")
            sys.stderr.flush()


class ServingKernel:
    """The served `KernelImpl`: weights mapped from the served module tree
    (`weight_map`, this rank's slices), the program from `step_impl`."""

    def __init__(
        self,
        rank: int,
        logits: bool = False,
        shared: dict[str, torch.Tensor] | None = None,
        **step_kw: Any,
    ) -> None:
        self.rank = rank
        self.shared = shared or {}  # build_weight_operands(shared=)
        # the steps return own-row logits (this rank's lm_head shard)
        self.logits = logits
        # Pallas interpret mode copies its inputs, so it cannot write in place.
        self.cache_inplace = not step_kw.get("interpret")
        self.step_kw = step_kw

    def collect_weights(self, model: Any, spec: KI.ModelSpec) -> list[torch.Tensor]:
        return build_weight_operands(
            model, spec, self.rank, torch_tpu_run_jax, logits=self.logits,
            shared=self.shared,
        )  # fmt: skip

    def build_step(self, cfg: KI.StepConfig, bucket: KI.Bucket):
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl

        kw = dict(self.step_kw)
        if bucket.max_ctx is not None:
            kw["max_ctx"] = bucket.max_ctx  # the attention scratch sizes with it
        return step_impl.build_step(
            cfg, bucket, cache_inplace=self.cache_inplace, logits=self.logits, **kw
        )

    def collect_draft_weights(
        self,
        draft_model: Any,
        spec: KI.ModelSpec,
        g: WM.Geometry,
        n_aux: int,
        shared: dict[str, torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        return build_draft_weight_operands(
            draft_model, spec, g, n_aux, self.rank, torch_tpu_run_jax,
            logits=self.logits, shared=shared,
        )  # fmt: skip

    def build_draft_step(self, cfg: KI.StepConfig, bucket: KI.Bucket, ctx_q: int):
        from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl

        kw = dict(self.step_kw)
        if bucket.max_ctx is not None:
            kw["max_ctx"] = bucket.max_ctx
        return step_impl.build_draft_step(
            cfg, bucket, ctx_q, cache_inplace=self.cache_inplace, logits=self.logits,
            **kw,
        )  # fmt: skip


# ------------------------------------------------------------- the driver


class DSv4Megakernel:
    """Built at `load_model` (weights), bound to the caches in `capture_model`."""

    # the in-place step dispatches its registered kernel directly (DirectStepCall)
    direct_dispatch = True

    @classmethod
    def maybe_create(
        cls, runner: Any, kernel: KI.KernelImpl | None = None
    ) -> DSv4Megakernel | None:
        if not envs.VLLM_TPU_DSV4_MEGAKERNEL:
            return None
        return cls.from_runner(runner, kernel)

    @classmethod
    def from_runner(
        cls, runner: Any, kernel: KI.KernelImpl | None = None
    ) -> DSv4Megakernel:
        """`kernel` None: the served step kernel (`ServingKernel`)."""
        hf = runner.model_config.hf_config
        _need(
            getattr(hf, "model_type", None) == "deepseek_v4",
            f"model_type {getattr(hf, 'model_type', None)!r}",
        )
        # The indexer K cache is allocated with the same shape in both layouts,
        # so a layout mismatch would trace and run with misread keys.
        layout = envs.VLLM_TPU_DSV4_INDEXER_KV_LAYOUT
        _need(
            layout == "sal",
            f"VLLM_TPU_DSV4_INDEXER_KV_LAYOUT={layout!r}: the kernel reads and "
            "writes the CSA indexer K cache in the seq-along-lane layout only; "
            "set VLLM_TPU_DSV4_INDEXER_KV_LAYOUT=sal",
        )
        pc = runner.parallel_config
        _need(pc.pipeline_parallel_size == 1, "pipeline parallelism")
        # vLLM TP inside each DP engine (TP8 x DP2): the kernel's attention
        # groups are the engines (StepKernelConfig.attn_tp)
        attn_tp = 1
        if runner._dp_lockstep_enabled():
            mode = MODE_DP
            attn_tp = int(pc.tensor_parallel_size)
            _need(
                attn_tp == 1 or not envs.VLLM_TPU_DSV4_MK_LOGITS,
                "logits mode under vLLM TP (stock TP compute_logits serves it)",
            )
        else:
            mode = MODE_TP
            _need(pc.data_parallel_size == 1, "DP without lockstep")
        spec_cfg = runner.speculative_config
        k = int(spec_cfg.num_speculative_tokens) if spec_cfg is not None else 0
        q = k + 1
        if k:
            _need(
                runner._static_verify_q_len == q,
                f"verify windows are not a static {q} rows",
            )
        drafter = getattr(runner, "drafter", None)
        _need(
            not envs.VLLM_TPU_DSV4_MK_LOGITS
            or drafter is None
            or hasattr(drafter, "_sample_from_hidden"),
            f"logits mode with the {type(drafter).__name__} drafter, which reads "
            "the target hidden states",
        )
        backbone = _find_backbone(runner.model)
        if attn_tp > 1:
            mesh, rank, expert_block = _engine_mesh_and_rank(attn_tp)
            vocab_start = int(backbone.embed_tokens.shard_indices.org_vocab_start_index)
        else:
            mesh, rank, expert_block = _mesh_and_rank(mode)
            vocab_start = 0
        spec = KI.ModelSpec(
            hidden_size=int(hf.hidden_size),
            num_layers=len(backbone.layers),
            compress_ratios=tuple(
                int(layer.attn.compress_ratio) for layer in backbone.layers
            ),
            dtype=str(runner.model_config.dtype).removeprefix("torch."),
            q=q,
            mode=mode,
            axis_name=mesh.axis_names[0],
            num_ranks=int(mesh.devices.size),
            aux_layers=tuple(backbone.aux_hidden_state_layers),
            sliding_window=int(hf.sliding_window),
            rms_eps=float(hf.rms_norm_eps),
            hc_eps=float(hf.hc_eps),
            vocab_size=int(hf.vocab_size),
            num_hash_layers=int(hf.num_hash_layers),
            n_routed_experts=int(hf.n_routed_experts),
            attn_tp=attn_tp,
        )
        spec, placed = placement_operands(spec, expert_block, runner.device)
        if kernel is None:
            # the kernel sizes its attention for the context classify() admits
            kernel = ServingKernel(
                rank,
                max_ctx=envs.VLLM_TPU_DSV4_MK_MAX_CTX,
                logits=envs.VLLM_TPU_DSV4_MK_LOGITS,
                shared=placed,
            )
        # Before the KV cache is sized (load_model): the copies come out of
        # what the cache would otherwise get.
        with torch.no_grad():
            weights = [w.detach() for w in kernel.collect_weights(runner.model, spec)]
        if weights and weights[0].device.type == "tpu":
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors(weights)
        logger.info(
            "DeepSeek-V4 megakernel weights: %d operands, %.2f GiB of load-time "
            "copies on this rank (the rest are the served tensors)",
            len(weights),
            copied_bytes(runner.model, weights) / 2**30,
        )
        per_bs = getattr(spec_cfg, "num_speculative_tokens_per_batch_size", None)
        extra_qs: tuple[int, ...] = ()
        if per_bs:
            vals = per_bs.values() if hasattr(per_bs, "values") else per_bs
            extra_qs = tuple(int(v) + 1 for v in vals if int(v) + 1 != q)
        mk = cls(
            spec=spec,
            buckets=parse_row_buckets(
                envs.VLLM_TPU_DSV4_MK_BUCKETS,
                q,
                envs.VLLM_TPU_DSV4_MK_OWN_REQS,
                envs.VLLM_TPU_DSV4_MK_MAX_CTX,
                extra_qs=extra_qs,
            ),
            kernel=kernel,
            mesh=mesh,
            rank=rank,
            expert_block=expert_block,
            vocab_start=vocab_start,
            weights=weights,
            decoder_layers=list(backbone.layers),
            device=runner.device,
            ab_enabled=envs.VLLM_TPU_DSV4_MK_AB,
            stats_period_s=envs.VLLM_TPU_DSV4_MK_STATS_PERIOD_S,
            trace_steps=envs.VLLM_TPU_DSV4_MK_TRACE_STEPS,
            nonfinite_stats=envs.VLLM_TPU_DSV4_MK_NONFINITE_STATS,
            parity=_parity_config(runner.model),
        )
        from vllm_torchtpu.kernels.deepseek_v4_megakernel.draft.vllm_adapter import (
            DraftMegakernel,
        )

        mk.draft = DraftMegakernel.maybe_create(runner, mk)
        return mk

    def __init__(
        self,
        *,
        spec: KI.ModelSpec,
        buckets: Sequence[KI.Bucket],
        kernel: KI.KernelImpl,
        mesh: Any,
        rank: int,
        expert_block: int,
        weights: list[torch.Tensor],
        decoder_layers: list[Any],
        vocab_start: int = 0,
        device: torch.device | str,
        ab_enabled: bool = False,
        stats_period_s: float = 60.0,
        trace_steps: int = 0,
        compile_fn: Callable[[Callable], Callable] | None = None,
        op_namespace: str = "dsv4_megakernel",
        nonfinite_stats: bool = False,
        parity: SD.ParityConfig | None = None,
    ) -> None:
        self.spec = spec
        self.buckets = tuple(buckets)
        self.kernel = kernel
        self.mesh = mesh
        self.rank = rank
        self.weights = weights
        self.decoder_layers = decoder_layers
        self.device = torch.device(device)
        self.op_namespace = op_namespace
        self._compile = compile_fn or self._default_compile
        # A runtime operand, not a graph constant: every rank must run the
        # identical executable.
        meta = torch.zeros(KI.META_LEN, dtype=torch.int32)
        meta[KI.META_RANK] = rank
        meta[KI.META_Q] = spec.q
        meta[KI.META_EBLK] = expert_block
        meta[KI.META_VSTART] = vocab_start
        self.meta = meta.to(self.device)
        self._metas: dict[int, torch.Tensor] = {spec.q: self.meta}
        self.cfg: KI.StepConfig | None = None
        self.caches: BoundCaches | None = None
        self._steps: dict[KI.Bucket, Callable] = {}
        self._preps: dict[KI.Bucket, Callable] = {}
        self._fits: dict[tuple[KI.Bucket, int], Callable] = {}
        self._idle_operands: dict[KI.Bucket, list[torch.Tensor]] = {}
        self._decision: StepDecision | None = None
        # the bucket the last forward ran on the megakernel (None: stock)
        self.last_run: KI.Bucket | None = None
        self.draft: Any = None  # draft.vllm_adapter.DraftMegakernel
        # forward returns the step rows' logits [rows, V] in place of the hidden
        self.logits = bool(getattr(kernel, "logits", False))
        self.ab = MKABCalibrator(enabled=ab_enabled, period_s=stats_period_s)
        self._step_mix: dict[str, int] = {}
        self._step_mix_t0 = time.monotonic()
        self._stats_period_s = stats_period_s
        self._trace_left = trace_steps
        self._trace_local = ""
        self.trace = RunTrace(
            rank, envs.VLLM_TPU_DSV4_MK_TRACE_RUNS, envs.VLLM_TPU_DSV4_MK_TRACE_SYNC
        )
        # False until the runner's warm-up is over (`mark_serving`)
        self.serving = False
        self._local_real_rows = 0  # this rank's scheduled rows (classify)
        self._local_ctx = 0  # and its longest request context
        self._nonfinite = (
            SD.NonfiniteStats(rank, spec.q, self.device, stats_period_s, self._compile)
            if nonfinite_stats
            else None
        )
        self._parity = None
        if parity is not None:
            _need(not ab_enabled, "parity probe with VLLM_TPU_DSV4_MK_AB")
            _need(not self.logits, "parity probe in logits mode")
            self._parity = SD.ParityProbe(rank, parity, self._compile)
        logger.info(
            "DeepSeek-V4 megakernel: mode=%s attn_tp=%d q=%d buckets=%s "
            "aux_layers=%s %d weight operands, mesh index %d of %d, ab=%s",
            spec.mode,
            spec.attn_tp,
            spec.q,
            [b.rows for b in self.buckets],
            spec.aux_layers,
            len(weights),
            rank,
            spec.num_ranks,
            ab_enabled,
        )

    @staticmethod
    def _default_compile(fn: Callable) -> Callable:
        return torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)

    # ------------------------------------------------------------ binding

    def bind_caches(self, runner: Any) -> None:
        groups = runner.kv_cache_config.kv_cache_groups
        group_of = {n: gid for gid, g in enumerate(groups) for n in g.layer_names}
        caches = bind_layer_caches(self.decoder_layers, group_of)
        widths = [
            int(runner.input_batch.block_table[gid].max_num_blocks_per_req)
            for gid in caches.group_ids
        ]
        pages = {
            int(lyr.attn.swa_cache_layer.block_size) for lyr in self.decoder_layers
        }
        _need(len(pages) == 1, f"SWA page sizes differ across layers: {pages}")
        self.bind(caches, widths, swa_page=pages.pop())
        if self.draft is not None:
            self.draft.bind_caches(runner, self.buckets)
        if self._nonfinite is not None:
            self._nonfinite.self_check()

    def bind(
        self, caches: BoundCaches, group_widths: Sequence[int], swa_page: int = 128
    ) -> None:
        """Fix the cache layout and build every bucket's kernel step."""
        # Every operand the step reads zero-copy: a view or a non-default layout
        # is copied by torch_tpu / XLA around the kernel on every call (writes
        # to a cache lost), or fails the step's compile.
        for what, tensors in (
            ("cache array", caches.arrays),
            ("weight operand", self.weights),
        ):
            for i, t in enumerate(tensors):
                got = settled_layout(t)
                _need(
                    got is True,
                    f"{what} {i} {tuple(t.shape)} {t.dtype}: "
                    + ("a view" if got is None else "not in the default device layout")
                    + ", the exported step reads it zero-copy",
                )
        if self.kernel.cache_inplace:
            # the kernel writes whole arrays from their base address
            for t in caches.arrays:
                _need(
                    t.is_contiguous() and t.storage_offset() == 0,
                    f"in-place cache {tuple(t.shape)} is a view",
                )
        self.caches = caches
        self.cfg = KI.StepConfig(
            swa_page=swa_page,
            model=self.spec,
            layers=caches.layers,
            cache_shapes=tuple(
                (tuple(t.shape), str(t.dtype).removeprefix("torch."))
                for t in caches.arrays
            ),
            group_widths=tuple(group_widths),
        )
        for b in self.buckets:
            self._steps[b] = self._build_step(b)
            self._preps[b] = self._build_prep(b)
            self._idle_operands[b] = self._zero_operands(b)
        logger.info(
            "DeepSeek-V4 megakernel bound: %d layers, %d cache arrays, "
            "KV groups %s (widths %s)",
            len(caches.layers),
            len(caches.arrays),
            caches.group_ids,
            list(group_widths),
        )

    @property
    def bound(self) -> bool:
        return self.cfg is not None

    def reachable_rows(self, token_paddings: Sequence[int]) -> list[int]:
        """The runner's padded row counts of the steps the buckets serve."""
        return sorted(
            {p for b in self.buckets for p in reachable_out_rows(b, token_paddings)}
        )

    def _build_step(self, bucket: KI.Bucket) -> Callable:
        """The kernel step for `bucket`.

        `kernel.cache_inplace`: the kernel DMA-writes the bound caches, which
        are plain inputs; the step is the exported program called eagerly,
        with no torch op around it, so nothing in torch (functionalization,
        copy-back) touches the caches. Otherwise the caches are donated,
        come back as outputs and are copied into the bound tensors inside a
        compiled step (a communicating pallas_call gets compiler-owned
        outputs, so XLA then copies every cache in and out per step)."""
        inplace = self.kernel.cache_inplace
        cfg = self.cfg
        caches = self.caches
        assert cfg is not None and caches is not None
        fn = self.kernel.build_step(cfg, bucket)
        n_bt = len(cfg.group_widths)
        n_w = len(self.weights)
        n_c = len(caches.arrays)
        n_aux = len(self.spec.aux_layers)
        n_head_out = 1 + n_aux
        n_in = _HEAD_OPERANDS + n_bt + n_w + n_c
        tag = f"q{bucket.q}_t{bucket.rows}"
        op_name = f"{self.op_namespace}::step_{tag}"
        n_out = n_head_out + (0 if inplace else n_c)
        donate = () if inplace else tuple(range(n_in - n_c, n_in))
        flat = export_step(
            fn, self.mesh, n_in, n_out, f"dsv4_megakernel_{tag}", op_name, donate
        )
        if inplace:
            call = inplace_call(flat, _HEAD_OPERANDS + n_bt, self.direct_dispatch)

            def inplace_step(meta, input_ids, positions, seq_lens, block_tables,
                             weights, caches):  # fmt: skip
                head = (meta, input_ids, positions, seq_lens, *block_tables)
                outs = call(head, weights, caches)
                return outs[0], tuple(outs[1:n_head_out])

            return inplace_step

        hidden_shape = (bucket.own_rows, self.spec.hidden_size)
        out_dtype = getattr(torch, self.spec.dtype)

        # The dispatcher caps a schema at 64 arguments; the weights and caches
        # go in as Tensor[] arguments and reach the callable positionally.
        @torch.library.custom_op(op_name, mutates_args=())
        def op(
            meta: torch.Tensor,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            seq_lens: torch.Tensor,
            block_tables: list[torch.Tensor],
            weights: list[torch.Tensor],
            caches: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            return list(
                flat(
                    meta,
                    input_ids,
                    positions,
                    seq_lens,
                    *block_tables,
                    *weights,
                    *caches,
                )
            )

        @op.register_fake
        def _(meta, input_ids, positions, seq_lens, block_tables, weights, caches):
            head = [
                input_ids.new_empty(hidden_shape, dtype=out_dtype)
                for _ in range(n_head_out)
            ]
            return head + [torch.empty_like(c) for c in caches]

        def step(meta, input_ids, positions, seq_lens, block_tables, weights, caches):
            outs = op(
                meta, input_ids, positions, seq_lens, block_tables, weights, caches
            )
            for dst, new in zip(caches, outs[n_head_out:]):
                dst.copy_(new)
            return outs[0], outs[1:n_head_out]

        step.__code__ = step.__code__.replace(co_name=f"dsv4_mk_step_{tag}")
        return self._compile(step)

    def _build_prep(self, bucket: KI.Bucket) -> Callable:
        """Slice this rank's runner metadata to the bucket's fixed shapes.

        Requests past `request_distribution[2]` (the step's request count) are
        zeroed: the runner leaves stale seq_lens there.
        """
        q, B, M = bucket.q, bucket.reqs, bucket.own_rows
        widths = self.cfg.group_widths

        def prep(input_ids, positions, seq_lens, request_distribution, *block_tables):
            active = (
                torch.arange(B, dtype=torch.int32, device=seq_lens.device)
                < request_distribution[2]
            )
            row_active = _fit_rows(active.repeat_interleave(q), M)
            zero_rows = torch.zeros(M, dtype=torch.int32, device=seq_lens.device)
            ids = torch.where(row_active, _fit_rows(input_ids, M).int(), zero_rows)
            pos = torch.where(row_active, _fit_rows(positions, M).int(), zero_rows)
            seq = torch.where(active, _fit_rows(seq_lens, B).int(), zero_rows[:B])
            num_reqs_padded = seq_lens.shape[0]
            tables = []
            for bt, width in zip(block_tables, widths):
                bt = _fit_rows(bt.reshape(num_reqs_padded, -1), B)
                bt = _fit_rows(bt, width, dim=1).int()
                tables.append(torch.where(active[:, None], bt, torch.zeros_like(bt)))
            return (ids, pos, seq, *tables)

        prep.__code__ = prep.__code__.replace(
            co_name=f"dsv4_mk_prep_q{q}_t{bucket.rows}"
        )
        return self._compile(prep)

    def _zero_operands(self, bucket: KI.Bucket) -> list[torch.Tensor]:
        """An idle rank's prepared operands: every request inactive."""
        z = torch.zeros
        ops = [
            z(bucket.own_rows, dtype=torch.int32),
            z(bucket.own_rows, dtype=torch.int32),
            z(bucket.reqs, dtype=torch.int32),
        ]
        ops += [z((bucket.reqs, w), dtype=torch.int32) for w in self.cfg.group_widths]
        return [t.to(self.device) for t in ops]

    def _fit(self, bucket: KI.Bucket, out_rows: int) -> Callable:
        """Kernel rows -> the runner's padded rows: free when they agree,
        leading-row views when the runner wants fewer (a compiled graph costs
        ~0.8 ms of host time per call, in-server xprof), else one small
        compiled graph per (bucket, rows) that pads."""
        if out_rows == bucket.own_rows:
            return lambda hidden, aux: (hidden, list(aux) or None)
        if out_rows < bucket.own_rows:
            return lambda hidden, aux: (
                hidden.narrow(0, 0, out_rows),
                [a.narrow(0, 0, out_rows) for a in aux] or None,
            )
        key = (bucket, out_rows)
        fit = self._fits.get(key)
        if fit is None:

            def fit_rows(hidden, aux):
                aux = [_fit_rows(a, out_rows).clone() for a in aux]
                return _fit_rows(hidden, out_rows), aux or None

            fit_rows.__code__ = fit_rows.__code__.replace(
                co_name=f"dsv4_mk_fit_q{bucket.q}_t{bucket.rows}_p{out_rows}"
            )
            fit = self._fits[key] = self._compile(fit_rows)
        return fit

    # -------------------------------------------------------- per step

    def classify(
        self,
        runner: Any,
        scheduler_output: Any,
        num_decode: int,
        num_windowed: int,
        num_chunks: int,
    ) -> LocalStep:
        """This rank's step, from host-side counts after the batch reorder."""
        num_reqs = runner.input_batch.num_reqs
        scheduled = len(scheduler_output.num_scheduled_tokens)
        local = self._classify(runner, scheduler_output, num_reqs, num_decode,
                               num_windowed, num_chunks, scheduled)  # fmt: skip
        self._local_real_rows = scheduler_output.total_num_scheduled_tokens
        self._local_ctx = local.ctx
        if self._trace_left > 0:
            self._trace_local = (
                f"batch_reqs={num_reqs} scheduled_reqs={scheduled} "
                f"tokens={scheduler_output.total_num_scheduled_tokens} "
                f"decode={num_decode} windowed={num_windowed} "
                f"static_verify={runner._num_static_verify_reqs} chunks={num_chunks} "
                f"local={local.kind}/{local.reason} rows={local.rows} ctx={local.ctx}"
            )
        return local

    def _meta_for(self, bucket: KI.Bucket) -> torch.Tensor:
        meta = self._metas.get(bucket.q)
        if meta is None:
            m = self.meta.detach().clone()
            m[KI.META_Q] = bucket.q
            meta = self._metas[bucket.q] = m.to(self.device)
        return meta

    def _classify(self, runner, scheduler_output, num_reqs, num_decode,
                  num_windowed, num_chunks, scheduled) -> LocalStep:  # fmt: skip
        if num_reqs != scheduled:
            # the persistent batch holds requests this step does not schedule
            return LocalStep(STOCK, "stock_unscheduled_reqs")
        step_q = None
        if (
            num_reqs > 0
            and num_decode == 0
            and num_windowed == num_reqs
            and runner._num_static_verify_reqs < num_reqs
            and any(b.q != self.spec.q for b in self.buckets)
        ):
            counts = set(scheduler_output.num_scheduled_tokens.values())
            if len(counts) == 1:
                step_q = int(next(iter(counts)))
        return classify_local_step(
            num_reqs=num_reqs,
            num_decode=num_decode,
            num_windowed=num_windowed,
            num_static_verify=runner._num_static_verify_reqs,
            num_tokens=scheduler_output.total_num_scheduled_tokens,
            num_chunks=num_chunks,
            has_encoder_inputs=bool(scheduler_output.scheduled_encoder_inputs),
            q=self.spec.q,
            buckets=self.buckets,
            max_reqs=self.buckets[0].reqs,
            max_seq_len=self._max_seq_len(runner, scheduler_output),
            step_q=step_q,
        )

    @staticmethod
    def _max_seq_len(runner: Any, scheduler_output: Any) -> int:
        """Longest request context after this step: computed + scheduled."""
        n = runner.input_batch.num_reqs
        if n == 0:
            return 0
        computed = runner.input_batch.num_computed_tokens_cpu[:n]
        sched = scheduler_output.num_scheduled_tokens
        ids = runner.input_batch.req_ids[:n]
        return int(max(int(c) + int(sched.get(r, 0)) for c, r in zip(computed, ids)))

    def decide(
        self, local: LocalStep, reduced: DPReduced | None = None
    ) -> StepDecision:
        """Fix this step's path; `forward` consumes it once."""
        if not self.bound:
            key, bucket = "stock_unbound", None
        else:
            key, bucket = decide_step(local, reduced, self.buckets)
        run, mix_key = self.ab.choose(key, bucket is not None)
        self.last_run = None
        parity = run and self._parity is not None and self._parity.select(self.phase)
        self._decision = StepDecision(
            run=run, key=mix_key, bucket=bucket, idle=local.kind == IDLE, parity=parity
        )
        self._note_step(mix_key)
        self.trace.step += 1
        if self._trace_left > 0:
            self._trace_left -= 1
            r = reduced
            dp = "" if r is None else (
                f" | dp: veto={int(r.any_veto)} prefill={int(r.any_prefill)} "
                f"total_rows={r.total_rows} max_ctx={r.max_ctx} "
                f"max_tokens={r.max_tokens} chunks={r.max_chunks}"
            )  # fmt: skip
            local_s = self._trace_local if local.kind != IDLE else "idle"
            self._emit(f"step: {local_s}{dp} -> {mix_key} run={int(run)}")
            self._trace_local = ""
        return self._decision

    def _emit(self, msg: str) -> None:
        # stderr: Ray drops worker logger.info lines while serving
        sys.stderr.write(f"[dsv4-megakernel rank {self.rank}] {msg}\n")
        sys.stderr.flush()

    @staticmethod
    def dp_entries(local: LocalStep | None, dp_rank: int, dp_size: int) -> list[int]:
        """This rank's trailing entries of the DP MAX all-reduce (`dp_entries`);
        `local` None is an idle rank (`execute_dummy_batch`)."""
        return dp_entries(local, dp_rank, dp_size)

    def decide_dp(
        self,
        local: LocalStep | None,
        *,
        max_tokens: int,
        max_chunks: int,
        any_prefill: bool,
        entries: Sequence[int],
    ) -> StepDecision:
        """`entries`: the reduced `dp_entries` (veto, longest context, then
        rows per rank)."""
        return self.decide(
            local if local is not None else LocalStep(IDLE, IDLE),
            DPReduced(
                max_tokens,
                max_chunks,
                any_prefill,
                any_veto=bool(entries[0]),
                total_rows=int(sum(entries[2:])),
                max_ctx=int(entries[1]),
            ),
        )

    def owns_step(self) -> bool:
        """Whether the next forward is the megakernel's (the runner then skips
        stock-only per-step setup such as the MoE padding mask)."""
        d = self._decision
        return d is not None and d.run and not d.parity

    def forward(
        self,
        stock_forward: Callable,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: Any,
    ) -> tuple[Any, Any]:
        decision, self._decision = self._decision, None
        # what the runner and the drafter read off this forward's output
        self.last_run = (
            decision.bucket if decision is not None and decision.run else None
        )
        if decision is None:
            return stock_forward(
                input_ids, positions, inputs_embeds, intermediate_tensors
            )
        if decision.parity:
            return self._forward_parity(
                decision,
                stock_forward,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
        if self.ab.enabled:
            _sync(None)
            t0 = time.perf_counter()
        if decision.run:
            out = self._run(
                decision, input_ids, positions, inputs_embeds, intermediate_tensors
            )
        else:
            out = stock_forward(
                input_ids, positions, inputs_embeds, intermediate_tensors
            )
        if self.ab.enabled:
            _sync([out[0]])
            self._record_ab((time.perf_counter() - t0) * 1e3)
        return out

    def _forward_parity(
        self,
        decision: StepDecision,
        stock_forward: Callable,
        *inputs: Any,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """The parity pair's two forwards on the same inputs (by default the
        stock forward, then the megakernel step), each synced before the next.
        The pair's megakernel output is served (the second of two; a
        stock-stock pair is followed by a served megakernel step). Every
        forward writes only the step's own cache slots (KV, compressor states,
        indexer records) and the megakernel writes them before reading them,
        so each sees the caches as if it ran alone and, when the megakernel
        runs last, the served caches are as after the megakernel alone (not
        so for the "mk,stock" control)."""
        pair = self._parity.pair
        served = 1 if pair[1] == "mk" else 0 if pair[0] == "mk" else None
        outs = []
        for i, path in enumerate(pair):
            if path == "stock":
                outs.append(stock_forward(*inputs))
            else:
                outs.append(self._run(decision, *inputs, count=i == served))
            _sync(None)  # this forward's cache writes land before the next reads
        out = self._run(decision, *inputs) if served is None else outs[served]
        if not decision.idle:
            self._parity.compare(
                self.phase,
                f"step {self.trace.step} {decision.bucket.key} ctx {self._local_ctx}",
                outs[0],
                outs[1],
                self._local_real_rows,
            )
        return out

    def _run(
        self,
        decision: StepDecision,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: Any,
        count: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """`count`: feed the non-finite counter (a parity step counts only the
        served run)."""
        if inputs_embeds is not None or intermediate_tensors is not None:
            raise AssertionError("megakernel step with embeddings or PP tensors")
        bucket = decision.bucket
        assert bucket is not None and self.caches is not None
        if decision.idle:
            ops = self._idle_operands[bucket]
        else:
            md = _current_attn_metadata()
            names = self.caches.group_layer_names
            first = md[names[0]]
            if first.sequence_layout_kind != "all":
                raise AssertionError(
                    f"sequence layout {first.sequence_layout_kind!r}: rows are "
                    "not request-major over the whole batch"
                )
            ops = self._preps[bucket](
                input_ids,
                positions,
                first.seq_lens,
                first.request_distribution,
                *(md[n].block_tables for n in names),
            )
        with self.trace(f"target {bucket.key} {'idle' if decision.idle else 'busy'}"):
            hidden, aux = self._steps[bucket](
                self._meta_for(bucket),
                ops[0],
                ops[1],
                ops[2],
                list(ops[3:]),
                self.weights,
                self.caches.arrays,
            )
        if count and self._nonfinite is not None:
            if decision.idle:
                self._nonfinite.count_idle(self.phase)
            else:
                self._nonfinite.count(self.phase, hidden, aux,
                                      first.request_distribution,
                                      self._local_real_rows)  # fmt: skip
        return self._fit(bucket, int(input_ids.shape[0]))(hidden, aux)

    # ----------------------------------------------------------- warmup

    def warmup(self, runner: Any) -> None:
        """Compile every bucket's kernel step (run idle: reads and writes
        nothing), and its prep and output fit at the runner's reachable padded
        row counts with max-model-len metadata shapes. Every rank runs the
        same sequence, so in-kernel collectives pair."""
        paddings = sorted(int(p) for p in runner.num_tokens_paddings)
        num_reqs = int(runner.num_reqs_max_model_len)
        md_like = [torch.zeros(num_reqs, dtype=torch.int32)]
        md_like.append(torch.tensor([0, 0, 0], dtype=torch.int32))
        md_like += [
            torch.zeros(num_reqs * w, dtype=torch.int32) for w in self.cfg.group_widths
        ]
        md_like = [t.to(self.device) for t in md_like]
        for b in self.buckets:
            outs = reachable_out_rows(b, paddings)
            with torch.no_grad():
                prepped = []
                for p in outs:
                    rows = torch.zeros(p, dtype=torch.int32, device=self.device)
                    prepped.append(self._preps[b](rows, rows, *md_like))
                # A busy rank feeds the step prep's outputs, an idle rank the
                # idle operands: run both (all requests inactive), and sync,
                # since torch_tpu compiles on the first sync, not at the call.
                for ops in (prepped[-1], self._idle_operands[b]):
                    hidden, aux = self._steps[b](
                        self._meta_for(b),
                        ops[0],
                        ops[1],
                        ops[2],
                        list(ops[3:]),
                        self.weights,
                        self.caches.arrays,
                    )
                    _sync([hidden, *aux])
                for p in outs:
                    _sync([self._fit(b, p)(hidden, aux)[0]])
                self._warm_diagnostics(b, outs, hidden, aux, md_like[1])
            logger.info(
                "DeepSeek-V4 megakernel: warmed bucket %s (runner rows %s)", b.key, outs
            )
        if self.draft is not None:
            self.draft.warmup(getattr(runner, "drafter", None))
        self.reset_stats()

    def _warm_diagnostics(self, b, outs, hidden, aux, request_distribution) -> None:
        """Compile the diagnostics' graphs now: compiling one at serving time
        stalls the busy ranks while their peers wait in a collective."""
        if self._nonfinite is not None:
            self._nonfinite.warm(hidden, aux, request_distribution)
            _sync(None)
        if self._parity is not None:
            for p in outs:
                fit_h, fit_aux = self._fit(b, p)(hidden, aux)
                self._parity.warm(fit_h, fit_aux or [])
            _sync(None)

    # ------------------------------------------------------------- stats

    @property
    def phase(self) -> int:
        """Index into `diag.vllm_adapter.PHASES`."""
        return int(self.serving)

    def mark_serving(self) -> None:
        """The runner's warm-up is over: later steps are served traffic."""
        if self._nonfinite is not None:
            self._nonfinite.log(self.phase)
        if self._parity is not None:
            for pair in self._parity.pairs:
                self._parity.pair = pair
                self._emit(self._parity.summary(self.phase))
        self.serving = True

    def reset_stats(self) -> None:
        self._step_mix.clear()
        self._step_mix_t0 = time.monotonic()
        self.ab.reset()

    def _note_step(self, key: str) -> None:
        if self._nonfinite is not None:
            self._nonfinite.maybe_log(self.phase)
        self._step_mix[key] = self._step_mix.get(key, 0) + 1
        now = time.monotonic()
        if now - self._step_mix_t0 < self._stats_period_s:
            return
        if self.rank == 0:
            total = sum(self._step_mix.values())
            mk = sum(v for k, v in self._step_mix.items() if k.startswith("mk_"))
            self._emit(
                f"step mix over {now - self._step_mix_t0:.0f}s: {mk}/{total} "
                f"megakernel {dict(sorted(self._step_mix.items()))}"
            )
        self._step_mix = {}
        self._step_mix_t0 = now

    def _record_ab(self, dt_ms: float) -> None:
        res = self.ab.record(dt_ms)
        if res is not None and self.rank == 0:
            elapsed, win, cum = res
            self._emit(f"AB latency (window {elapsed:.0f}s): {win}")
            self._emit(f"AB latency (cumulative): {cum}")
