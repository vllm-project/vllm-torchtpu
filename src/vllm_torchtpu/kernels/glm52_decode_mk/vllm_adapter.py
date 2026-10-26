# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Serving glue for the GLM-5.2 decode megakernel (kernels/glm52_decode_mk), wired into TPUModelRunner (stage-B lane
L5: the same-harness twin A/B; WHITEBOARD.v3 §6.5).

Port of glm5_mk_dcp's adapter (vllm_adapter.py @ d41e70f6d; its rebased copy ran on main 9ffdb1eb4 in G0, j-31f97434)
to this kernel's operand plan and deployment: TP = DCP = world, DP 1, **EP off** (TP16 MoE: the served routed experts are
main's moe_tp_decode TP layout w13 [E, H, 2I/R], w2 [E, I/R, H], s13 [E, 1, 1, 2I/R], s2 [E, 1, 1, H], passed per layer,
zero-copy = the served operand form), q_b: this rank's HR heads and their scale slice, taken as they are at qrep 0
(column-parallel q_b_proj) or sliced out of the replicated q_b at qrep 1 (VLLM_DCP_Q_REPLICATE=1, q group = the world;
`q_b_local`) into the megakernel's own stacked copy, so the kernel is the same either way, MTP verify rows. Enabled by
GLM52_DECODE_MK=1.

Runner contract (runner/tpu_runner.py): `maybe_create` at load_model, `bind_caches` + `warmup` at capture_model,
`step_ok` per scheduled step (sets runner._glm52_mk_step to the step's row bucket, or None), `try_call` inside
forward_model, which consumes that decision (one forward per decision) and returns None to fall back to stock.
`enabled` is the runtime A/B toggle (twin harness R8: flipped by collective_rpc between interleaved stock / megakernel
batches); `repack()` re-reads the router weight and bias after the harness re-draws them (R3).

The step_ok guard (WHITEBOARD.v3 §8 W4). The megakernel engages only when every precondition holds; otherwise the step
runs the stock graphs, and every decline is counted under a named reason (no silent decline):
- load time (else no megakernel is built; logged): GLM-5.2 DSA, TP == world == 16 and DCP == 16 with the DCP group = the
  TP group = ranks 0..15 in rank order (the megakernel's mesh position, DCP KV ownership and rank operand coincide), or
  DCP == 8 as V16 (stageB/v16/DESIGN-ADDENDUM.md: DCP groups {0..7}, {8..15}, rank_in_group = rank % 8; the two
  replicas of each DCP shard split it by page parity, so the kernel's work and collectives stay 16-way),
  interleave 4, EP off, q_b column-parallel (qrep 0) or replicated over a q group = the world (qrep 1, sliced per
  layer: `qrep_layer` / `q_b_local`), MTP 3, block 256, one rope
  pair layout for every layer, MLA and indexer alike (`rope_layout`: glm-agentx's NeoX permutes -> MKConfig.rope_neox),
  and the
  A0-10 pool guard (a max_model_len request's rank-0 pages <= Tiles.pool, else the DSA wave overflows: `check_pool`);
- bind time (else every step declines 'unbound'; logged): the served cache form. The runner-stacked groups
  (`runner.glm5_kv_stack_groups`, GLM5_MKD_STACK_C; allocated by default when this megakernel is built) in equal
  groups of consecutive layers, nope / rope grouped alike, every layer bound to its group row; or per-layer caches
  only up to GLM52_DECODE_MK_PERLAYER_MAX_L layers (per-layer groups at 78 layers are 263,806 bundles, blocked;
  stageB/L3-served-form/RESULTS.md §0.5);
- step time (`classify_step`, from the scheduler output and the runner's request bookkeeping, both identical on
  every rank, so every rank decides alike): one chunk, no encoder inputs,
  every request a verify request (1 + its scheduled draft tokens, >= 1 draft, past its prompt) with 2..QMAX rows,
  <= BMAX requests, and a row bucket holding the tokens and the requests (<= bucket // QMAX); prefill and mixed steps
  decline;
- call time (`try_call`): the forward carries no inputs_embeds / intermediate tensors, the token operands are 1-D
  int32 with 1..bucket rows (a runner bucket below the megakernel's, TPU_TOKEN_BUCKET_EXTRA=4,8, is padded up outside
  the compiled step, so only the warmed 16 / 32-row programs ever run) and the attention metadata has the shapes the
  bucket was compiled for (else 'embeds' / 'shape': stock, counted, the shape reason logged as a warning).
Counts: `counts` (per reason, mk per bucket), logged by rank 0 on the first step of each new reason and every
GLM52_DECODE_MK_LOG_EVERY decisions (default 2000); `status()` returns them.

Env:
- GLM52_DECODE_MK=1            build the megakernel at load (else the runner never sees it);
- GLM52_DECODE_MK_ROWS=16,32   one compiled step per row bucket; a step takes the smallest bucket that holds it;
- GLM52_DECODE_MK_TRACE=1      trace build (named scopes 'L_norm' ... per layer; A1-N3: +0.05-0.14% time);
- GLM52_DECODE_MK_OFF=1        built and warmed, but `enabled` starts False (the harness turns it on per arm);
- GLM52_DECODE_MK_LOG_EVERY=n  engagement log cadence (decisions; 0 = first-of-reason lines only);
- GLM52_DECODE_MK_PERLAYER_MAX_L=10  the largest model served from per-layer caches (the 10-layer twin compiled);
- GLM52_DECODE_MK_LV_FULL=1    with lever LV (GLM52_DECODE_MK_LEVERS=B,LV): also build each bucket's program WITHOUT
                               LV and run it on steps whose live rows fill the bucket after LV's 4-row rounding
                               (ceil(T / 4) * 4 == bucket: LV moves every row there, and its extra issue sites cost
                               0.01-0.14 ms per step, stageB/lv-prod D1); "0" = the LV program on every step;
- GLM5_MKD_STACK_C             unset: stacked caches when the megakernel is built; "0" keeps per-layer caches.
"""

import dataclasses
import os
import time
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

from . import kernel as K
from . import layout as LY
from . import presets as PR
from .config import FULL, rup

logger = init_logger(__name__)

AXIS = "mk52"


class Unsupported(Exception):
    pass


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise Unsupported(why)


def enabled() -> bool:
    return os.environ.get("GLM52_DECODE_MK", "0") == "1"


def _rows_env() -> list[int]:
    v = os.environ.get("GLM52_DECODE_MK_ROWS", "16")
    rows = sorted({int(x) for x in v.split(",") if x.strip()})
    _need(bool(rows) and all(r in (16, 32) for r in rows), f"GLM52_DECODE_MK_ROWS={v}")
    return rows


def rank0_pages(seq: int, *, page: int, R: int, ilv: int, vr: int = 1) -> int:
    """Rank 0's local page count for a request of `seq` tokens under DCP interleave ownership: the kernel's per-request
    wave weight (dsa.py step_init `req[b, 11]`, nl0 = (seq // (R * ilv)) * ilv + min(seq % (R * ilv), ilv)). R is the
    cache's DCP size; vr > 1 (V16): the replica's tiles, ceil(pages / vr) (the most any replica of any shard takes)."""
    g = R * ilv
    nl0 = (seq // g) * ilv + min(seq % g, ilv)
    return -(-(-(-nl0 // page)) // vr)


def check_pool(max_model_len: int, pools, *, page: int, R: int, ilv: int, vr: int = 1) -> int:
    """The A0-10 pool guard (stageA/a0/RESULTS.md §8): step_init opens a new DSA wave only when the pool is already
    non-empty, so ONE request with more rank-0 pages than `Tiles.pool` overflows b_keys / b_blk (UB, no in-kernel
    check). vLLM never schedules seq > max_model_len and rank0_pages is non-decreasing in seq, so bounding a
    max_model_len request at load time bounds every step. -> that page count; raises Unsupported (no megakernel)."""
    need = rank0_pages(max_model_len, page=page, R=R, ilv=ilv, vr=vr)
    for t, pool in pools.items():
        _need(
            need <= int(pool),
            f"pool guard (A0-10): a {max_model_len}-token request has {need} rank-0 pages > "
            f"Tiles.pool {pool} (b{t}): the DSA wave would overflow",
        )
    return need


def _unwrap(x):
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x


def _kn(weight: torch.Tensor, k: int, n: int, what: str) -> torch.Tensor:
    if tuple(weight.shape) == (k, n):
        return weight
    _need(
        tuple(weight.shape) == (n, k),
        f"{what}: shape {tuple(weight.shape)}, want ({k}, {n})",
    )
    return weight.t().contiguous()


def _pad_n(w: torch.Tensor, n: int) -> torch.Tensor:
    k = w.shape[-1]
    return torch.nn.functional.pad(w, (0, n - k)) if k != n else w


def _scale(layer, n: int, what: str) -> torch.Tensor:
    for name in ("weight_scale", "weight_scale_inv"):
        s = getattr(layer, name, None)
        if s is not None:
            _need(
                s.numel() == n,
                f"{what}.{name}: {tuple(s.shape)} is not per-channel [{n}]",
            )
            return s.reshape(-1).float()
    raise Unsupported(f"{what}: no weight_scale")


def _own(t: torch.Tensor) -> torch.Tensor:
    """An operand with storage of exactly its own shape: a view operand is re-materialized by torch_tpu on every
    call (2.2 ms/step of copies, j-725a55a4)."""
    out = torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out


def _rope_cache_operand(cs: torch.Tensor) -> torch.Tensor:
    """The rope cache as a step operand. The served buffer when it owns its storage: glm-agentx's
    TPU_ROPE_CACHE_ROW_MAJOR re-materializes it {1,0} and pins that entry layout for every graph input of its
    (dtype, shape), so a fresh default-layout copy would fail at the step's entry (RuntimeProgramInputMismatch). A view
    (e.g. a truncated cache) is copied once (`_own`), re-laid out {1,0} when that (dtype, shape) is pinned."""
    if _owns_storage(cs):
        return cs
    out = _own(cs)
    try:
        from vllm_torchtpu.runner import utils as runner_utils
    except Exception:  # noqa: BLE001 - a tree (or host) without the relayout: nothing is pinned
        return out
    pinned = getattr(runner_utils, "_PINNED_ENTRY_LAYOUTS", {})
    if (out.dtype, tuple(out.shape)) in pinned:
        out = runner_utils._to_row_major(out)
    return out


def _owns_storage(t: torch.Tensor) -> bool:
    try:
        return (
            t.storage_offset() == 0
            and t.is_contiguous()
            and t.untyped_storage().nbytes() == t.numel() * t.element_size()
        )
    except Exception:  # noqa: BLE001
        return False


def _dt(x) -> str:
    """dtype name of a torch tensor / torch dtype / jnp dtype ('float8_e4m3fn', 'bfloat16', 'uint32', ...)."""
    d = x.dtype if isinstance(x, torch.Tensor) else x
    if isinstance(d, torch.dtype):
        return str(d).replace("torch.", "")
    import numpy as np

    return np.dtype(d).name


def _find_backbone(model):
    for m in model.modules():
        if hasattr(m, "layers") and hasattr(m, "embed_tokens") and hasattr(m, "norm"):
            return m
    raise Unsupported("no decoder backbone (layers/embed_tokens/norm) in the model")


def _attn_module(self_attn):
    for m in self_attn.modules():
        if hasattr(m, "W_UK_T") and hasattr(m, "W_UV") and hasattr(m, "kv_cache"):
            return m
    raise Unsupported("no MLA attention module with W_UK_T / W_UV / kv_cache")


def rope_layout(layers) -> bool:
    """The rope pair layout the served weights and caches use -> MKConfig.rope_neox. glm-agentx permutes the rope
    output columns at load (TPU_MLA_ROPE_NEOX_PERMUTE on the MLA wrapper: q_b / fused_qkv_a k_pe columns, cache k_pe;
    TPU_INDEXER_ROPE_NEOX_PERMUTE on the indexer: wq_b / wk / k_norm, cache k) and marks each module
    `_rope_neox_permuted`; an older tree has no marker (= interleaved). The kernel takes one layout for every layer and
    for MLA and indexer alike (one rope table), so a mixed model declines."""
    modes = set()
    for l, ly in enumerate(layers):
        sa = ly.self_attn
        idx = getattr(sa, "indexer", None)
        mla = [
            m
            for m in sa.modules()
            if hasattr(m, "_rope_neox_permuted") and m is not idx
        ]
        _need(
            len(mla) <= 1,
            f"layer {l}: {len(mla)} MLA modules carry a rope-permute marker",
        )
        modes.add(("mla", bool(mla[0]._rope_neox_permuted) if mla else False))
        if idx is not None:
            modes.add(("indexer", bool(getattr(idx, "_rope_neox_permuted", False))))
    vals = {v for _, v in modes}
    _need(
        len(vals) == 1,
        f"rope layouts differ (MLA / indexer, neox-permuted?): {sorted(modes)}; the megakernel takes one",
    )
    return vals.pop()


def _routed(mlp):
    exp = mlp.experts
    return getattr(exp, "routed_experts", exp)


def q_b_local(
    weight: torch.Tensor, scale: torch.Tensor, cfg, rank: int, qrep: bool = False
):
    """This rank's q_b operand: (weight [QL, HR*DQ] fp8, per-channel scale [HR*DQ] f32), the megakernel's own heads.

    - qrep 0 (column-parallel q_b_proj at TP16): the served weight holds this rank's HR heads -> as is.
    - qrep 1 at DCP == TP (VLLM_DCP_Q_REPLICATE=1: DCPGroupColumnParallelLinear over a group = the TP world, tp_size
      1): the served weight holds all NH heads in head order, and the column-parallel shard of rank r is heads
      [r*HR, (r+1)*HR) = columns [r*HR*DQ, (r+1)*HR*DQ) of the weight and of the scale (vLLM's `_local_view` slices
      q the same way). Every load step acts per output channel or per head: the block-scale dequant + per-channel
      requant (fp8.py `_process_fp8_linear_weights`) and the NeoX rope permute (mla_attention_op
      `permute_rope_to_half_split`, per head over all NH heads). So the slice is byte-identical to the qrep 0 shard
      (tests/kernels/test_glm52_decode_mk_qrep.py). The megakernel copies it into its own stacked qb operand: the
      kernel, its operands and its HBM stream are those of qrep 0.

    weight: either orientation ([QL, N] flipped, or [N, QL]); scale: per-channel, N elements. The served N must be
    the form `qrep` names (HR*DQ for qrep 0, NH*DQ for qrep 1), else Unsupported (no megakernel).

    V16 (DCP = R / 2, stageB/v16/DESIGN-ADDENDUM.md): the q group is the DCP group (D ranks, rank_in_group = rank % D)
    and the replicated weight holds that group's D * HR heads in head order (lane rank // D holds heads
    [lane * D * HR, (lane + 1) * D * HR)); rank r's column-parallel heads [r * HR, (r + 1) * HR) are the group's heads
    [(r % D) * HR, +HR): columns [(r % D) * HR * DQ, +HR * DQ)."""
    nqb, nall = cfg.HR * cfg.DQ, cfg.HR * cfg.D * cfg.DQ
    want = nall if qrep else nqb
    n = weight.numel() // cfg.QL
    _need(
        n == want and n * cfg.QL == weight.numel(),
        f"q_b_proj {tuple(weight.shape)}: want N {want} "
        + ("(qrep 1: the q group's D * HR heads)" if qrep else "(qrep 0: this rank's HR heads)"),
    )
    _need(
        scale.numel() == n, f"q_b scale {tuple(scale.shape)} is not per-channel [{n}]"
    )
    s = scale.reshape(-1).float()
    if not qrep:
        return _kn(weight, cfg.QL, n, "q_b_proj"), s
    c0 = (rank % cfg.D) * nqb
    if tuple(weight.shape) == (cfg.QL, n):
        w = weight[:, c0 : c0 + nqb]
    else:
        _need(
            tuple(weight.shape) == (n, cfg.QL),
            f"q_b_proj {tuple(weight.shape)} is not [{cfg.QL}, {n}] / [{n}, {cfg.QL}]",
        )
        w = weight[c0 : c0 + nqb]
    # views of the served tensors: _collect_weights stacks them into its own copy (no per-layer copy is kept)
    return _kn(w, cfg.QL, nqb, "q_b_proj (qrep slice)"), s[c0 : c0 + nqb]


def qrep_layer(q_b_proj, R: int, tp_rank: int, layer: int, dcp: int | None = None) -> bool:
    """Is this layer's q_b_proj replicated (vLLM sets `qrep_active` on DCPGroupColumnParallelLinear)? Accepted only
    over a q group that is the DCP group in rank order (group_size = DCP, rank_in_group = this TP rank % DCP; at
    DCP = TP that is the whole world): then the replicated weight is sliceable to this rank's column-parallel heads
    (`q_b_local`). Anything else raises."""
    if not bool(getattr(q_b_proj, "qrep_active", False)):
        return False
    D = int(dcp or R)
    gs = int(getattr(q_b_proj, "group_size", 0) or 0)
    rig = int(getattr(q_b_proj, "rank_in_group", -1))
    _need(
        gs == D and rig == tp_rank % D,
        f"layer {layer}: q_b_proj is replicated (qrep_active) over a q group of {gs} at rank_in_group {rig}; the "
        f"megakernel slices this rank's heads only from a group = the DCP group ({D}, rank {tp_rank % D})",
    )
    return True


def index_pattern(has_indexer: list[bool], freq0: int, skip0: int) -> tuple[int, int]:
    """(idx_freq, idx_skip) reproducing the model's ACTUAL per-layer fresh-indexer pattern under the kernel's rule
    (config.full_indexer: max(l - skip + 1, 0) % freq == 0), the hf (freq, skip) first. R1 (twin-plan §5.1): the twin
    sets `index_topk_pattern` [F, F, S], which the hf freq/skip (4, 3) do not describe; (4, 2) does."""
    L = len(has_indexer)

    def pat(fr, sk):
        return [max(l - sk + 1, 0) % fr == 0 for l in range(L)]

    cands = (
        [(freq0, skip0)]
        + [(freq0, sk) for sk in range(0, L + 1)]
        + [(fr, sk) for fr in range(1, max(L, 4) + 1) for sk in range(0, L + 1)]
    )
    for fr, sk in cands:
        if fr >= 1 and pat(fr, sk) == list(has_indexer):
            return fr, sk
    raise Unsupported(
        f"indexer pattern {has_indexer} is not max(l - skip + 1, 0) % freq == 0 for any (freq, skip)"
    )


PERLAYER_BLOCKED = (
    "per-layer cache groups at 78 layers are 263,806 bundles against the 65,536 limit (stageB/"
    "L3-served-form RESULTS §0.5): serve from runner-stacked caches (GLM5_MKD_STACK_C)"
)
MAX_CACHE_GROUPS = 4  # the timed served form is nope / rope 2 groups, idx 1 (probe 15 bench_mkb FORM=srv)


def stacked_groups_plan(
    bindings: dict, sizes: dict, max_groups: int = MAX_CACHE_GROUPS
) -> dict:
    """Validate the runner-stacked cache groups against the kernel's group contract (kernel.cache_groups / csel):
    per kind, equal groups of consecutive layers, the kind's i-th layer at row i % gs of group i // gs; nope and rope
    grouped alike. bindings: {kind: [(group index, row) of each layer of the kind, in layer order]}; sizes: {kind:
    [layers per group]}. -> {kind: n_groups}; raises Unsupported."""
    G = {}
    for kind in ("nope", "rope", "idx"):
        _need(kind in bindings and kind in sizes, f"stacked caches: no {kind} groups")
        sz, bd = list(sizes[kind]), list(bindings[kind])
        _need(
            bool(sz) and len(set(sz)) == 1,
            f"stacked caches: {kind} groups are not equal: {sz}",
        )
        gs, n = sz[0], len(sz)
        _need(
            n <= max_groups,
            f"stacked caches: {kind} in {n} groups (> {max_groups}, untimed)",
        )
        _need(
            len(bd) == gs * n,
            f"stacked caches: {kind} {len(bd)} layers bound, groups hold {gs * n}",
        )
        for i, (gi, row) in enumerate(bd):
            _need(
                (gi, row) == (i // gs, i % gs),
                f"stacked caches: {kind} layer #{i} bound to group {gi} row {row}, want {i // gs} row {i % gs}",
            )
        G[kind] = n
    _need(
        G["nope"] == G["rope"] and list(sizes["nope"]) == list(sizes["rope"]),
        f"stacked caches: nope groups {sizes['nope']} != rope groups {sizes['rope']}",
    )
    return G


# ------------------------------------------------------------------ the step guard (pure: CPU-tested)

REASONS = (
    "mk",
    "off",
    "unbound",
    "multi_chunk",
    "encoder",
    "prefill",
    "mixed",
    "no_draft",
    "q_len",
    "over_reqs",
    "over_bucket",
    "embeds",
    "shape",
)


def classify_step(
    reqs, *, num_chunks: int, has_encoder: bool, qmax: int, rows, bmax: int
):
    """The step-time preconditions, from the scheduler's view (identical on every rank).
    reqs: [(scheduled tokens, scheduled draft tokens, still in its prompt)] per request.
    -> (row bucket or None, reason, detail). A request is a verify request iff it has >= 1 draft token, its scheduled
    tokens are exactly 1 + its drafts and it is past its prompt. Prefill chunks (also a 2..4-token prompt tail after
    a prefix-cache hit), plain decode without drafts and mixed steps decline."""
    if num_chunks != 1:
        return None, "multi_chunk", f"num_chunks {num_chunks}"
    if has_encoder:
        return None, "encoder", ""
    n_ver = n_pref = n_nodraft = 0
    big = 0
    for n, k, in_prompt in reqs:
        if k >= 1 and n == 1 + k and not in_prompt:
            n_ver += 1
            if not 2 <= n <= qmax:
                big = max(big, n)
        elif in_prompt or n != 1 + k:
            n_pref += 1
        else:
            n_nodraft += 1
    nreq = len(reqs)
    total = sum(n for n, _, _ in reqs)
    detail = f"{nreq} reqs, {total} tokens: {n_ver} verify, {n_pref} prefill, {n_nodraft} decode w/o draft"
    if nreq == 0:
        return None, "no_draft", detail
    if n_ver and (n_pref or n_nodraft):
        return None, "mixed", detail
    if n_pref:
        return None, "prefill", detail
    if n_nodraft:
        return None, "no_draft", detail
    if big:
        return None, "q_len", f"{detail}; a request has {big} rows > QMAX {qmax}"
    if nreq > bmax:
        return None, "over_reqs", f"{detail}; > BMAX {bmax}"
    for t in rows:
        if total <= t and nreq <= t // qmax:
            return t, "mk", f"b{t}"
    return None, "over_bucket", f"{detail}; buckets {list(rows)}"


class StepBucket(int):
    """step_ok's decision: the row bucket (an int, which is all the runner passes along) plus `full` = the step's live
    rows fill the bucket after LV's 4-row rounding (ceil(T / 4) * 4 == bucket), so the adapter runs the bucket's
    no-LV program (GLM52_DECODE_MK_LV_FULL). A plain int decision (harness paths) runs the LV program."""

    def __new__(cls, t: int, full: bool = False):
        o = int.__new__(cls, t)
        o.full = bool(full)
        return o


def fills_bucket(total_tokens: int, bucket: int) -> bool:
    """LV moves ceil(T / 4) * 4 all-reduce rows (>= its floor), q-AG rows 4 / ceil(T / 8) * 8 and ceil(T / 4) A2A
    chunks: every collective moves the whole bucket iff ceil(T / 4) * 4 == bucket."""
    return -(-int(total_tokens) // 4) * 4 >= int(bucket)


def sched_reqs(runner, scheduler_output) -> list:
    """classify_step's per-request view of a SchedulerOutput (+ the runner's prompt bookkeeping after _update_states)."""
    ns = scheduler_output.num_scheduled_tokens
    spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    ib = runner.input_batch
    out = []
    for req_id, n in ns.items():
        i = ib.req_id_to_index[req_id]
        in_prompt = int(ib.num_computed_tokens_cpu[i]) < int(ib.num_prompt_tokens[i])
        out.append((int(n), len(spec.get(req_id) or ()), in_prompt))
    return out


def fit_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    """x's leading dim cut or zero-padded to n rows (the step's pad-to-TMAX)."""
    k = x.shape[0]
    if k >= n:
        return x[:n]
    return torch.nn.functional.pad(x, [0, 0] * (x.dim() - 1) + [0, n - k])


def to_bucket_rows(x: torch.Tensor, t: int) -> torch.Tensor:
    """input_ids / positions as the compiled step's [t] operand. The step is torch.compile(dynamic=False): a
    different input length is a different program, so the runner's own token buckets below the megakernel's
    (TPU_TOKEN_BUCKET_EXTRA=4,8) are padded here, outside the compiled region, with the zeros the step's pad
    inserted before (`fit_rows`); the step then only ever sees t rows (its `fit` is the identity). n == t passes the
    runner's tensor untouched."""
    if x.dim() == 1 and x.shape[0] == t:
        return x
    return fit_rows(x.reshape(-1), t).contiguous()


def token_rows_mismatch(bucket: int, input_ids, positions) -> str:
    """The call-time tripwire on the token operands (rank-uniform: every rank gets the same padded token count).
    `to_bucket_rows` maps any 1..bucket rows onto the bucket's warmed [bucket] int32 program; anything else would
    compile a new 78-layer program mid-serving (or cut real rows), so it declines ('shape') with this reason."""
    shp = tuple(input_ids.shape)
    if len(shp) != 1:
        return f"b{bucket}: input_ids shape {shp} is not 1-D"
    if positions is not None and tuple(positions.shape) != shp:
        return f"b{bucket}: positions shape {tuple(positions.shape)} != input_ids {shp}"
    n = shp[0]
    if not 1 <= n <= bucket:
        return (
            f"b{bucket}: {n} token rows not in 1..{bucket} (the runner padded past the megakernel bucket; "
            "only smaller runner buckets are padded up)"
        )
    for name, x in (("input_ids", input_ids), ("positions", positions)):
        if x is not None and x.dtype != torch.int32:
            return f"b{bucket}: {name} dtype {x.dtype} != int32 (the warmed program's)"
    return ""


N_FIX = 10  # pos [TMAX], seq [S], qsl [S+1], bt [S, W], cs rows [TMAX, DR], h0 [M, H], ltab, dtab (replicated),
# rank i32[1], smalls (per rank)


def build_spmd(cfg, mesh, *, max_pages, nb, nbi, tiles, groups, stacked=False):
    """The SPMD step program, jax only (also lowered by the lane's AOT script, stageB/L5-twin-harness/aot_l5.py):
    shard_map over the one-axis R-rank mesh of (metadata + rope table from wrapper.py -> megakernel -> writer).
    Inputs: N_FIX fixed (8 replicated + smalls per rank), then the dense stacks, the per-layer served expert tensors
    and the caches, all sharded on dim 0 (= this rank's own tensors). Caches: `stacked` -> the runner's stacked groups
    [n_layers / G, nb, ...] as they are (the kernel's group operands); else per-layer [nb, ...], each reshaped to the
    kernel's one-layer group [1, ...] inside the body (a bitcast). Outputs: out [M, H] bf16 and the written caches.
    -> (smap, in_specs, out_specs, info)."""
    import jax
    from jax.sharding import PartitionSpec as P

    from . import wrapper as W
    from .writer import build_writer

    call, info = K.build(
        cfg,
        max_pages=W.kernel_pages(cfg, max_pages),
        nb=nb,
        nbi=nbi,
        tiles=tiles,
        stack_e=False,
        groups=groups,
    )
    writer, winfo = build_writer(
        cfg, max_pages=max_pages, nb=nb, nbi=nbi, groups=groups
    )  # the writer takes the FULL local tables (V16: the replica writes every token of its shard)
    assert len(info["c_names"]) == len(winfo["c_names"])
    n_d, n_e, n_c = len(info["dkinds"]), len(info["e_names"]), len(info["c_names"])
    TMAX = cfg.TMAX
    rep, sh = P(), P(AXIS)
    in_specs = (rep,) * (N_FIX - 2) + (sh, sh) + (sh,) * (n_d + n_e) + (sh,) * n_c
    out_specs = (sh,) + (sh,) * n_c

    def body(*a):
        pos, seq, qsl, btv, csr, h0, ltab_, dtab_, rank, smalls = a[:N_FIX]
        dw = a[N_FIX : N_FIX + n_d]
        ew = a[N_FIX + n_d : N_FIX + n_d + n_e]
        cs = a[N_FIX + n_d + n_e :]
        r = rank[
            0
        ]  # an operand, not lax.axis_index: partition-id does not lower in torch_tpu's export
        scal, rmeta, qmeta, bt, bti, rvec = W.metadata(
            cfg, TMAX, pos, seq, qsl, btv, nb, nbi, max_pages, r
        )
        wbt, wbti = bt, bti
        if cfg.VR > 1:  # V16: this replica's half tables for the megakernel
            bt, bti = W.v16_tables(cfg, bt, bti, r)
        tab = W.rope_table(cfg, csr)
        c1 = (
            list(cs) if stacked else [c[None] for c in cs]
        )  # per-layer: [1, ...], a one-layer group
        outs = call(
            scal,
            rmeta,
            qmeta,
            bt,
            bti,
            ltab_,
            dtab_,
            rvec,
            h0,
            tab,
            smalls,
            *dw,
            *ew,
            *c1,
        )
        out, nrow, rrow, irec = outs[:4]
        new = writer(scal, rmeta, wbt, wbti, nrow, rrow, irec, *c1)
        return (out, *(new if stacked else [c[0] for c in new]))

    smap = jax.shard_map(
        body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )
    return smap, in_specs, out_specs, info


class Glm52DecodeMK:
    """Built at load_model (weights), bound to the caches at capture_model."""

    @classmethod
    def maybe_create(cls, runner) -> "Glm52DecodeMK | None":
        if not enabled():
            return None
        try:
            return cls(runner)
        except Unsupported as e:
            logger.warning(
                "GLM-5.2 decode megakernel disabled (not built; every step runs the stock graphs): %s",
                e,
            )
            return None

    def __init__(self, runner):
        import torch.distributed as dist

        hf = runner.model_config.hf_config
        pc = runner.parallel_config
        _need(
            type(hf).__name__.startswith("GlmMoeDsa")
            or getattr(hf, "model_type", "") == "glm_moe_dsa",
            f"model_type {hf.model_type}",
        )
        _need(pc.pipeline_parallel_size == 1, "pipeline parallel")
        _need(getattr(pc, "data_parallel_size", 1) == 1, "data parallel")
        R = int(pc.tensor_parallel_size)
        _need(
            R == FULL.R,
            f"TP {R}: the megakernel is built and verified for TP = DCP = {FULL.R} only",
        )
        dcp = int(getattr(pc, "decode_context_parallel_size", 1))
        _need(
            dcp in (R, R // 2),
            f"DCP {dcp}: the megakernel is built for DCP = TP = {R} or, as V16 (virtual DCP {R} on the DCP {R // 2} "
            f"cache: the two replicas of each DCP shard split it by page parity), DCP = {R // 2}",
        )
        self.dcp = dcp
        _need(dist.is_initialized() and dist.get_world_size() == R, "world != TP")
        _need(
            int(getattr(pc, "cp_kv_cache_interleave_size", 1)) == FULL.ILV,
            "interleave != 4",
        )
        _need(
            not bool(getattr(pc, "enable_expert_parallel", False)),
            "EP is on (stage A is TP16 MoE, EP off)",
        )
        # qrep is judged per layer below (q_b_proj.qrep_active), not from parallel_config.dcp_q_replicate: vLLM's GLM
        # config hook sets that field True even when VLLM_DCP_Q_REPLICATE=0 turns runtime q-replication off.
        self.rank_layout = self._check_rank_layout(R, dcp)
        spec = runner.speculative_config
        _need(spec is not None and getattr(spec, "method", "") == "mtp", "MTP is off")
        qmax = 1 + int(spec.num_speculative_tokens)
        _need(qmax == 4, f"{qmax} rows per request")
        _need(getattr(hf, "scoring_func", None) == "sigmoid", "router scoring")
        _need(
            getattr(hf, "n_group", 1) in (1, None)
            and getattr(hf, "topk_group", 1) in (1, None),
            "grouped routing",
        )
        _need(bool(getattr(hf, "norm_topk_prob", True)), "unnormalized top-k")
        _need(
            bool(getattr(hf, "rope_interleave", False))
            and bool(getattr(hf, "indexer_rope_interleave", False)),
            "non-interleaved rope",
        )
        rp = getattr(hf, "rope_parameters", None) or {}
        _need(rp.get("rope_type", "default") == "default", f"rope {rp}")
        _need(runner.block_size == FULL.page, f"block_size {runner.block_size}")
        self.rows = _rows_env()
        bb = _find_backbone(runner.model)
        self.backbone = bb
        L = int(hf.num_hidden_layers)
        _need(len(bb.layers) == L, f"{len(bb.layers)} layers")
        has_idx = [
            getattr(bb.layers[l].self_attn, "indexer", None) is not None
            for l in range(L)
        ]
        freq, skip = index_pattern(
            has_idx, int(hf.index_topk_freq), int(hf.index_skip_topk_offset)
        )
        rope_neox = rope_layout(bb.layers)
        base = dataclasses.replace(
            FULL,
            R=R,
            L=L,
            first_dense=hf.first_k_dense_replace,
            H=hf.hidden_size,
            NH=hf.num_attention_heads,
            QL=hf.q_lora_rank,
            idx_freq=freq,
            idx_skip=skip,
            eps=hf.rms_norm_eps,
            routed_scale=float(hf.routed_scaling_factor),
            QMAX=qmax,
            TMAX=self.rows[-1],
            BM=int(runner.scheduler_config.max_num_seqs),
            rope_neox=rope_neox,
            DCP=0 if dcp == R else dcp,
        )
        for got, want, what in (
            (hf.kv_lora_rank, base.KL, "kv_lora_rank"),
            (hf.qk_nope_head_dim, base.DN, "qk_nope"),
            (hf.qk_rope_head_dim, base.DR, "qk_rope"),
            (hf.v_head_dim, base.DV, "v_head_dim"),
            (hf.n_routed_experts, base.E, "experts"),
            (hf.num_experts_per_tok, base.TOPK, "top-k"),
            (hf.moe_intermediate_size, base.I, "moe_intermediate"),
            (
                hf.moe_intermediate_size * hf.n_shared_experts,
                base.IS,
                "shared intermediate",
            ),
            (hf.intermediate_size, base.ID, "dense intermediate"),
            (hf.index_n_heads, base.IH, "index heads"),
            (hf.index_head_dim, base.IDIM, "index dim"),
            (hf.index_topk, base.KT, "index_topk"),
        ):
            _need(got == want, f"{what} {got} != kernel {want}")
        self.cfgs = {}
        for t in self.rows:
            c = dataclasses.replace(base, TMAX=t)
            try:
                c.check()
            except AssertionError as e:
                raise Unsupported(f"admission (TMAX {t}): {e}") from e
            self.cfgs[t] = c
        cfg = self.cfg = self.cfgs[self.rows[-1]]
        for l in range(L):
            _need(has_idx[l] == cfg.full_indexer(l), f"layer {l}: indexer pattern")
        # qrep 1 (VLLM_DCP_Q_REPLICATE=1): each such layer's replicated q_b is sliced to this rank's heads at load
        self.qrep_layers = [
            l
            for l in range(L)
            if qrep_layer(
                bb.layers[l].self_attn.q_b_proj, R, int(self.rank_layout["tp"]), l, dcp
            )
        ]
        self.rank = int(dist.get_rank())
        self.counts = {r: 0 for r in REASONS}
        self.decisions = 0
        self.consumed = 0
        self.log_every = int(os.environ.get("GLM52_DECODE_MK_LOG_EVERY", "2000") or 0)
        self.unbound = "capture_model has not bound the caches"
        self.cache_form = None
        self.call_shapes = {}
        self.warm_shapes = {}
        self.enabled = os.environ.get("GLM52_DECODE_MK_OFF", "0") != "1"
        self.calls = 0
        from torch_tpu._internal.pallas import get_pallas_mesh

        self.mesh = get_pallas_mesh(axis_names=(AXIS,), mesh_shape=(R,))
        mml = int(runner.model_config.max_model_len)
        self.max_pages = -(-mml // (cfg.page * dcp))  # full local tables (V16: the kernel takes half of them)
        rot = bb.layers[0].self_attn.rotary_emb
        _need(rot.cos_sin_cache.shape[-1] == cfg.DR, "rope cache width")
        with torch.no_grad():
            self.cos_sin = _rope_cache_operand(rot.cos_sin_cache.detach())
            self.k_scales = [
                float(
                    getattr(_attn_module(bb.layers[l].self_attn), "_k_scale_float", 1.0)
                )
                for l in range(cfg.L)
            ]
            self.dense, self.experts = self._collect_weights()
            self.smalls = self._build_smalls().detach()
        self.tiles = K.Tiles(trace=os.environ.get("GLM52_DECODE_MK_TRACE", "0") == "1")
        # stage-B levers (presets.py): GLM52_DECODE_MK_LEVERS = "" / "A" (stage A) | "B" (all) | "L4,L1,..."
        self.levers = PR.parse(os.environ.get("GLM52_DECODE_MK_LEVERS", ""))
        self.tiles_by_rows = {
            t: dataclasses.replace(self.tiles, **PR.tiles_kwargs(t, self.levers, self.rows))
            for t in self.rows
        }
        # LV full-bucket steps run the no-LV program (GLM52_DECODE_MK_LV_FULL, default on with lever LV)
        self.lv_full = "LV" in self.levers and os.environ.get("GLM52_DECODE_MK_LV_FULL", "1") != "0"
        self.tiles_full_by_rows = (
            {
                t: dataclasses.replace(
                    self.tiles, **PR.tiles_kwargs(t, tuple(x for x in self.levers if x != "LV"), self.rows)
                )
                for t in self.rows
            }
            if self.lv_full
            else {}
        )
        self.pool_pages = check_pool(
            mml,
            {t: tl.pool for t, tl in self.tiles_by_rows.items()},
            page=cfg.page,
            R=dcp,
            ilv=cfg.ILV,
            vr=cfg.VR,
        )
        self.caches = None
        self._steps = {}
        self._steps_full = {}
        self.mla_layer_name = _attn_module(bb.layers[0].self_attn).layer_name
        logger.info(
            "GLM-5.2 decode megakernel: built for rows %s, L %d (fresh indexer layers %s, freq %d skip %d), "
            "rope cache rows %d (max_model_len %d), k_scales %s, trace %s, enabled %s, levers %s, rope %s, "
            "q_b %s, dcp %s",
            self.rows,
            cfg.L,
            LY.layers_of(cfg, "iwq"),
            freq,
            skip,
            int(self.cos_sin.shape[0]),
            mml,
            sorted(set(self.k_scales)),
            self.tiles.trace,
            self.enabled,
            ",".join(self.levers) or "stage A",
            "neox (half-split)" if cfg.rope_neox else "interleaved",
            f"qrep 1 (this rank's heads sliced, {len(self.qrep_layers)}/{cfg.L} layers)"
            if self.qrep_layers
            else "qrep 0 (column-parallel)",
            f"{dcp} (V16-p: virtual DCP {R} on the DCP {dcp} cache, page-parity replica split)"
            if cfg.VR > 1
            else f"{dcp} (= TP)",
        )

    @staticmethod
    def _check_rank_layout(R: int, dcp: int | None = None) -> dict:
        """The megakernel addresses peers and owns DCP KV positions by its mesh position, which get_pallas_mesh
        numbers by torch rank (distributed/dcp.py), and takes the rank as an operand: all of that is the stock DCP
        ownership only when the TP group is ranks 0..R-1 in rank order and the DCP group is the rank's block of dcp
        consecutive ranks (dcp = R: the same group; V16 dcp = R / 2: {0..D-1}, {D..R-1}) with rank_in_group = rank % dcp
        (distributed/dcp.py: global_rank == lane * dcp + dcp_rank)."""
        import torch.distributed as dist
        from vllm.distributed import parallel_state as ps

        D = int(dcp or R)
        rank = int(dist.get_rank())
        out = {"rank": rank}
        for name, get, n in (("tp", ps.get_tp_group, R), ("dcp", ps.get_dcp_group, D)):
            g = get()
            ranks = tuple(int(x) for x in g.ranks)
            lo = rank // n * n
            _need(
                ranks == tuple(range(lo, lo + n)),
                f"{name} group ranks {ranks} != {lo}..{lo + n - 1}",
            )
            _need(
                int(g.rank_in_group) == rank % n,
                f"{name} rank_in_group {g.rank_in_group} != rank % {n} = {rank % n}",
            )
            out[name] = int(g.rank_in_group)
        return out

    # ---------------------------------------------------------------- weights

    def _layer(self, l: int) -> dict:
        cfg, r = self.cfg, self.rank
        H = cfg.H
        Ly = self.backbone.layers[l]
        sa = Ly.self_attn
        at = _attn_module(sa)
        qb_w, qb_s = q_b_local(
            sa.q_b_proj.weight,
            _scale(sa.q_b_proj, sa.q_b_proj.weight.numel() // cfg.QL, "q_b_proj"),
            cfg,
            r,
            qrep=l in self.qrep_layers,
        )
        d = {
            "qkva": _pad_n(
                _kn(sa.fused_qkv_a_proj.weight, H, cfg.QKVA, "fused_qkv_a_proj"),
                cfg.QKVP,
            ),
            "qb": qb_w,
            "q_b_s": qb_s,
            "wuk": at.W_UK_T,
            "wuv": at.W_UV,
            "wo": _kn(sa.o_proj.weight, cfg.KR, H, "o_proj"),
        }
        idx = getattr(sa, "indexer", None)
        _need((idx is not None) == cfg.full_indexer(l), f"layer {l}: indexer pattern")
        if idx is not None:
            d["iwq"] = _kn(idx.wq_b.weight, cfg.QL, cfg.IH * cfg.IDIM, "indexer.wq_b")
            d["iwk"] = _pad_n(
                _kn(
                    idx.wk_weights_proj.weight,
                    H,
                    cfg.IDIM + cfg.IH,
                    "indexer.wk_weights_proj",
                ),
                cfg.IKW,
            )
        mlp = Ly.mlp
        if cfg.is_moe(l):
            d["gate"] = self._gate_kn(l)
            rt = _routed(mlp)
            d["ew13"], d["ew2"] = rt.w13_weight, rt.w2_weight
            d["es13"], d["es2"] = rt.w13_weight_scale_inv, rt.w2_weight_scale_inv
            for k_, shp in (
                ("ew13", (cfg.E, H, 2 * cfg.IR)),
                ("ew2", (cfg.E, cfg.IR, H)),
                ("es13", (cfg.E, 1, 1, 2 * cfg.IR)),
                ("es2", (cfg.E, 1, 1, H)),
            ):
                _need(
                    tuple(d[k_].shape) == shp,
                    f"layer {l} {k_}: {tuple(d[k_].shape)} != {shp} (TP16 layout)",
                )
            for k_ in ("ew13", "ew2", "es13", "es2"):
                _need(
                    _owns_storage(d[k_]),
                    f"layer {l} {k_}: a view operand (would be re-materialized per call)",
                )
            se = mlp.shared_experts
            d["sh13"] = _kn(
                se.gate_up_proj.weight, H, 2 * cfg.ISR, "shared.gate_up_proj"
            )
            d["sh2"] = _kn(se.down_proj.weight, cfg.ISR, H, "shared.down_proj")
        else:
            d["dw13"] = _kn(
                mlp.gate_up_proj.weight, H, 2 * cfg.IDR, "dense.gate_up_proj"
            )
            d["dw2"] = _kn(mlp.down_proj.weight, cfg.IDR, H, "dense.down_proj")
        return d

    def _gate_kn(self, l: int) -> torch.Tensor:
        cfg = self.cfg
        return _pad_n(
            _kn(self.backbone.layers[l].mlp.gate.weight, cfg.H, cfg.E, "gate"), cfg.EPAD
        )

    def _collect_weights(self):
        cfg = self.cfg
        per = {l: self._layer(l) for l in range(cfg.L)}
        self._per_layer = per
        dsh = LY.dense_shapes(cfg)
        dense, self._dense_kind = [], {}
        for k in LY.DKINDS:
            ls = LY.layers_of(cfg, k)
            if not ls:
                continue
            ts = [per[l][k].detach() for l in ls]
            for l, t in zip(ls, ts):
                _need(
                    tuple(t.shape) == tuple(dsh[k][0]),
                    f"{k}[{l}]: {tuple(t.shape)} != {dsh[k][0]}",
                )
                _need(
                    _dt(t) == _dt(dsh[k][1]),
                    f"{k}[{l}]: dtype {t.dtype} != {dsh[k][1]}",
                )
            self._dense_kind[k] = len(dense)
            dense.append(_own(torch.stack(ts)))  # load-time stacked copy
        experts = []
        for l in LY.layers_of(cfg, "gate"):  # served tensors, per layer, zero-copy
            experts += [per[l]["ew13"], per[l]["ew2"], per[l]["es13"], per[l]["es2"]]
        return dense, experts

    def _build_smalls(self) -> torch.Tensor:
        cfg = self.cfg
        lay, S = LY.smalls_layout(cfg)
        rows = []
        for l in range(cfg.L + 1):
            f = (
                self._small_fields(l)
                if l < cfg.L
                else {"in_ln": self.backbone.norm.weight}
            )
            dev = next(iter(f.values())).device
            parts = []
            for name, (_, n) in lay.items():
                v = f.get(name)
                v = (
                    torch.zeros(n, dtype=torch.float32, device=dev)
                    if v is None
                    else v.reshape(-1).float()
                )
                if name == "k_scale":
                    v = v[:1].expand(n)
                _need(v.numel() == n, f"smalls {name}[{l}]: {v.numel()} != {n}")
                parts.append(torch.nn.functional.pad(v, (0, rup(n) - n)))
            rows.append(torch.cat(parts))
        return _own(torch.stack(rows).reshape(cfg.L + 1, 1, S))

    def _small_fields(self, l: int) -> dict:
        cfg = self.cfg
        Ly = self.backbone.layers[l]
        sa = Ly.self_attn
        dev = Ly.input_layernorm.weight.device
        f = {
            "in_ln": Ly.input_layernorm.weight,
            "post_ln": Ly.post_attention_layernorm.weight,
            "q_a_ln": sa.q_a_layernorm.weight,
            "kv_a_ln": sa.kv_a_layernorm.weight,
            "qkv_a_s": torch.nn.functional.pad(
                _scale(sa.fused_qkv_a_proj, cfg.QKVA, "fused_qkv_a_proj"),
                (0, cfg.QKVP - cfg.QKVA),
            ),
            "q_b_s": self._per_layer[l]["q_b_s"],
            "o_proj_s": _scale(sa.o_proj, cfg.H, "o_proj"),
            "k_scale": torch.full(
                (1,), self.k_scales[l], dtype=torch.float32, device=dev
            ),
        }
        idx = getattr(sa, "indexer", None)
        if idx is not None:
            f["idx_wq_s"] = _scale(idx.wq_b, cfg.IH * cfg.IDIM, "indexer.wq_b")
            f["idx_kn_w"] = idx.k_norm.weight
            f["idx_kn_b"] = idx.k_norm.bias
        mlp = Ly.mlp
        if cfg.is_moe(l):
            gb = torch.full((cfg.EPAD,), -1e30, dtype=torch.float32, device=dev)
            gb[: cfg.E] = mlp.gate.e_score_correction_bias.float()
            f["gate_bias"] = gb
            f["sw13_s"] = _scale(
                mlp.shared_experts.gate_up_proj, 2 * cfg.ISR, "shared.gate_up_proj"
            )
            f["sw2_s"] = _scale(mlp.shared_experts.down_proj, cfg.H, "shared.down_proj")
        else:
            f["w13_s"] = _scale(mlp.gate_up_proj, 2 * cfg.IDR, "dense.gate_up_proj")
            f["w2_s"] = _scale(mlp.down_proj, cfg.H, "dense.down_proj")
        return f

    @torch.no_grad()
    def repack(self) -> dict:
        """R3: after the router weights / bias are re-drawn in place (twin harness `_randomize_router`), copy them
        into the megakernel's operands (the stacked `gate` kind and the smalls rows), in place, then check on device
        that every operand equals its module. Returns the max |diff| per check (0.0 = by-reference equivalent)."""
        cfg = self.cfg
        gi = self._dense_kind["gate"]
        g = self.dense[gi]
        for j, l in enumerate(LY.layers_of(cfg, "gate")):
            g[j].copy_(self._gate_kn(l))
        self.smalls.copy_(self._build_smalls())
        lay, _ = LY.smalls_layout(cfg)
        off = 0
        o_gb = None
        for name, (_, n) in lay.items():
            if name == "gate_bias":
                o_gb = (off, n)
            off += rup(n)
        gate_d = bias_d = 0.0
        for j, l in enumerate(LY.layers_of(cfg, "gate")):
            gate_d = max(
                gate_d,
                float((g[j].float() - self._gate_kn(l).float()).abs().max().cpu()),
            )
            if o_gb is not None:
                b = self.smalls[l, 0, o_gb[0] : o_gb[0] + cfg.E]
                bias_d = max(
                    bias_d,
                    float(
                        (
                            b
                            - self.backbone.layers[
                                l
                            ].mlp.gate.e_score_correction_bias.float()
                        )
                        .abs()
                        .max()
                        .cpu()
                    ),
                )
        return {
            "gate_maxdiff": gate_d,
            "bias_maxdiff": bias_d,
            "moe_layers": len(LY.layers_of(cfg, "gate")),
        }

    # ----------------------------------------------------------------- caches

    def bind_caches(self, runner) -> None:
        """Bind the served cache form (W4 precondition; L3-c). Never raises: on any mismatch the megakernel stays
        unbound, every step declines 'unbound' and the reason is logged (and returned by status()).

        - runner-stacked (runner.glm5_kv_stack_groups non-empty; GLM5_MKD_STACK_C): the groups ARE the kernel's
          cache-group operands [n_layers / G, nb, ...], validated by `stacked_groups_plan` (equal consecutive
          groups, nope / rope alike, each layer's module bound to its group tensor at its row); the stock DCP ops
          address the same tensors through `kv_stack`. GLM-5.2 served: nope / rope 2 x 39, idx 1 x 21 (the
          SparseCore 16 GiB per-tensor split) = the form probe 15 timed.
        - per-layer: one operand per (kind, layer), the runner's own tensors, each reshaped to a one-layer group in
          the body; only up to GLM52_DECODE_MK_PERLAYER_MAX_L layers (PERLAYER_BLOCKED)."""
        try:
            self._bind_caches(runner)
            self.unbound = None
        except Unsupported as e:
            self.unbound = str(e)
            self.caches = None
            self._steps = {}
            self._steps_full = {}
            logger.warning(
                "GLM-5.2 decode megakernel: NOT bound, every step will run stock ('unbound'): %s",
                e,
            )

    def _module_caches(self, l: int):
        sa = self.backbone.layers[l].self_attn
        at = _attn_module(sa)
        kv = _unwrap(at.kv_cache)
        _need(
            isinstance(kv, (tuple, list)) and len(kv) == 2,
            f"layer {l}: MLA cache is not (nope, rope)",
        )
        out = {
            "nope": (kv[0], getattr(at, "kv_stack", None)),
            "rope": (kv[1], getattr(at, "kv_stack", None)),
        }
        idx = getattr(sa, "indexer", None)
        if idx is not None:
            out["idx"] = (
                _unwrap(idx.k_cache.kv_cache),
                getattr(idx.k_cache, "kv_stack", None),
            )
        return out

    def _bind_caches(self, runner) -> None:
        cfg = self.cfg
        spec = getattr(
            _attn_module(self.backbone.layers[0].self_attn), "mla_kv_spec", None
        )
        _need(
            not (spec and getattr(spec[1], "packed_rope", False)),
            "TPU_SPARSE_MLA_PACKED_ROPE: the rope cache is 64 B/token packed; the megakernel reads 128 B rows",
        )
        per = {l: self._module_caches(l) for l in range(cfg.L)}
        groups = getattr(runner, "glm5_kv_stack_groups", None) or {}
        stacked = any(v[1] is not None for d in per.values() for v in d.values())
        caches, views = [], []
        if groups or stacked:
            _need(
                bool(groups)
                and all(v[1] is not None for d in per.values() for v in d.values()),
                f"caches partly stacked (runner groups {sorted(groups)}; module kv_stack on "
                "some layers only)",
            )
            g0 = groups["nope"][0][0]
            nb = int(g0.shape[1])
            nbi = int(groups["idx"][0][0].shape[1]) if groups.get("idx") else nb
            ck = LY.cache_kinds(cfg, nb, nbi)
            bindings, sizes = {}, {}
            for kind in ("nope", "rope", "idx"):
                _need(
                    bool(groups.get(kind)),
                    f"stacked caches: the runner stacked no {kind}",
                )
                tens = [t for t, _ in groups[kind]]
                sizes[kind] = [int(t.shape[0]) for t in tens]
                bd = []
                for l in ck[kind][0]:
                    t, st = per[l][kind]
                    gi = next((i for i, g in enumerate(tens) if g is t), None)
                    _need(
                        gi is not None,
                        f"stacked caches: layer {l} {kind} is not bound to a runner group tensor",
                    )
                    _need(
                        st is not None and int(st[1]) == int(t.shape[1]),
                        f"stacked caches: layer {l} {kind} kv_stack {st} vs group nb {int(t.shape[1])}",
                    )
                    bd.append((gi, int(st[0])))
                bindings[kind] = bd
                for t in tens:
                    want = (sizes[kind][0],) + tuple(ck[kind][1])
                    _need(
                        tuple(t.shape) == want,
                        f"stacked {kind} group {tuple(t.shape)} != {want}",
                    )
                    _need(
                        _dt(t) == _dt(ck[kind][2]),
                        f"stacked {kind} dtype {t.dtype} != {ck[kind][2]}",
                    )
                    if not _owns_storage(t):
                        views.append(f"{kind}-group")
                    caches.append(t)
            G = stacked_groups_plan(bindings, sizes)
            form = "stacked"
        else:
            maxl = int(os.environ.get("GLM52_DECODE_MK_PERLAYER_MAX_L", "10"))
            _need(
                maxl >= cfg.L,
                f"{cfg.L} layers on per-layer caches (> {maxl}): {PERLAYER_BLOCKED}",
            )
            nb = int(per[0]["nope"][0].shape[0])
            nbi = int(next(d["idx"][0] for d in per.values() if "idx" in d).shape[0])
            ck = LY.cache_kinds(cfg, nb, nbi)
            G = {}
            for kind in ("nope", "rope", "idx"):
                ls = ck[kind][0]
                G[kind] = len(ls)
                for l in ls:
                    t = per[l][kind][0]
                    _need(
                        tuple(t.shape) == tuple(ck[kind][1]),
                        f"cache {kind}[{l}]: {tuple(t.shape)} != {ck[kind][1]}",
                    )
                    _need(
                        _dt(t) == _dt(ck[kind][2]),
                        f"cache {kind}[{l}]: dtype {t.dtype} != {ck[kind][2]}",
                    )
                    if not _owns_storage(t):
                        views.append(f"{kind}{l}")
                    caches.append(t)
            form = "per-layer"
        _need(
            not views,
            f"view cache operands {views} (re-materialized by torch_tpu on every call)",
        )
        self.caches, self.groups, self.nb, self.nbi = caches, G, nb, nbi
        self.cache_form, self.cache_views = form, views
        logger.info(
            "GLM-5.2 decode megakernel: %d dense + %d expert (served per-layer) + %d cache operands, cache form "
            "%s, groups %s, %d blocks (idx %d), max_pages %d",
            len(self.dense),
            len(self.experts),
            len(caches),
            form,
            G,
            nb,
            nbi,
            self.max_pages,
        )
        for t in self.rows:
            self._steps[t] = self._build_op(self.cfgs[t])
            if self.lv_full:
                self._steps_full[t] = self._build_op(self.cfgs[t], full=True)

    def _build_op(self, cfg, full=False):
        """The compiled per-step torch function for row bucket cfg.TMAX: embed (torch, vocab-parallel) -> one SPMD
        program shard_map(metadata + rope table (wrapper.py, the instruments-tested arithmetic) -> megakernel ->
        writer) over the R-rank mesh -> caches copied back (donated, aliased in place). full: the bucket's no-LV
        program (tiles_full_by_rows; GLM52_DECODE_MK_LV_FULL)."""
        import jax

        from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

        n_d, n_e, n_c = len(self.dense), len(self.experts), len(self.caches)
        smap, in_specs, out_specs, info = build_spmd(
            cfg,
            self.mesh,
            max_pages=self.max_pages,
            nb=self.nb,
            nbi=self.nbi,
            tiles=(self.tiles_full_by_rows if full else self.tiles_by_rows)[cfg.TMAX],
            groups=self.groups,
            stacked=self.cache_form == "stacked",
        )
        assert list(info["dkinds"]) == list(self._dense_kind), (
            info["dkinds"],
            list(self._dense_kind),
        )
        assert len(info["e_names"]) == n_e, (len(info["e_names"]), n_e)
        assert len(info["c_names"]) == n_c
        TMAX, M = cfg.TMAX, cfg.M
        dev = self.smalls.device
        ltab = torch.tensor(info["ltab"], dtype=torch.int32).to(dev)
        dtab = torch.tensor(info["dtab"], dtype=torch.int32).to(dev)
        rank_t = torch.tensor([self.rank], dtype=torch.int32).to(dev)
        n_fix = N_FIX
        n_in = n_fix + n_d + n_e + n_c
        args = [f"x{i}" for i in range(n_in)]
        ret = "tuple[" + ", ".join(["jax.Array"] * len(out_specs)) + "]"
        fname = f"glm52_decode_mk_b{TMAX}{'f' if full else ''}"
        src = (
            f"def {fname}({', '.join(a + ': jax.Array' for a in args)}) -> {ret}:\n"
            f"    return _smap({', '.join(args)})\n"
        )
        ns: dict[str, Any] = {"jax": jax, "_smap": smap}
        exec(compile(src, f"<{fname}>", "exec"), ns)  # noqa: S102 - arity is data
        name = f"glm52_decode_mk::step_b{TMAX}{'f' if full else ''}"
        # unregistered callable (sharded_jax_op register=False): the operand count exceeds a torch op schema's 64;
        # the op below takes the weights and caches as Tensor[] arguments (glm5_mk_dcp)
        flat = sharded_jax_op(
            name + "_flat",
            ns[fname],
            mesh=self.mesh,
            input_partition_specs=in_specs,
            output_partition_specs=out_specs,
            donate_argnums=tuple(range(n_in - n_c, n_in)),
            register=False,
        )

        @torch.library.custom_op(name, mutates_args=())
        def op(
            pos: torch.Tensor,
            seq: torch.Tensor,
            qsl: torch.Tensor,
            btv: torch.Tensor,
            csr: torch.Tensor,
            h0: torch.Tensor,
            ltab_: torch.Tensor,
            dtab_: torch.Tensor,
            rank: torch.Tensor,
            smalls: torch.Tensor,
            weights: list[torch.Tensor],
            caches: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            return list(
                flat(
                    pos,
                    seq,
                    qsl,
                    btv,
                    csr,
                    h0,
                    ltab_,
                    dtab_,
                    rank,
                    smalls,
                    *weights,
                    *caches,
                )
            )

        @op.register_fake
        def _(pos, seq, qsl, btv, csr, h0, ltab_, dtab_, rank, smalls, weights, caches):
            return [torch.empty((M, cfg.H), dtype=torch.bfloat16, device=h0.device)] + [
                torch.empty_like(c) for c in caches
            ]

        fit = fit_rows  # identity here: __call__ hands the step ids / positions already at TMAX rows

        cos_sin = self.cos_sin
        n_pos = int(cos_sin.shape[0])

        def step(ids, positions, qsl, seq_lens, bt_flat, smalls, weights, caches):
            i32 = torch.int32
            nq = qsl.shape[0] - 1
            btv = bt_flat.reshape(nq, -1).to(i32)
            p = fit(positions.reshape(-1).to(i32), TMAX)
            T = qsl[-1].to(i32)
            valid = torch.arange(TMAX, device=p.device, dtype=i32) < T
            pc = torch.clamp(p * valid.to(i32), 0, n_pos - 1)
            csr = (
                cos_sin[pc.long()].to(torch.bfloat16).float()
            )  # [TMAX, DR], bf16 values
            h0 = self.backbone.embed_tokens(fit(ids.reshape(-1), TMAX)).float()
            outs = op(
                p,
                seq_lens.to(i32),
                qsl.to(i32),
                btv,
                csr,
                h0,
                ltab,
                dtab,
                rank_t,
                smalls,
                list(weights),
                list(caches),
            )
            for dst, src in zip(caches, outs[1 : 1 + len(caches)]):
                dst.copy_(src)
            return outs[0][:TMAX]

        logger.info(
            "GLM-5.2 decode megakernel: step b%d%s built (%d operands)",
            TMAX,
            " (full-bucket, no LV)" if full else "",
            n_in,
        )
        return torch.compile(step, backend="tpu", fullgraph=True, dynamic=False)

    # ------------------------------------------------------------------- run

    def bucket_for(self, num_tokens: int) -> int | None:
        for t in self.rows:
            if num_tokens <= t:
                return t
        return None

    def step_ok(self, runner, scheduler_output, num_chunks: int) -> int | None:
        """The W4 step guard (module docstring): the row bucket this step runs the megakernel at, or None (stock).
        Decided from the scheduler output and static facts only, so every rank decides alike (a rank-local decline
        of a collective kernel hangs the others). Every decision is counted under its reason."""
        bucket, why, detail = self._decide(runner, scheduler_output, num_chunks)
        self._count(why, detail, bucket)
        return bucket

    def decision(self, reqs, num_chunks: int = 1, has_encoder: bool = False):
        """step_ok's decision from classify_step's per-request view [(scheduled tokens, drafts, in prompt)]: None or a
        StepBucket (full = the step fills its bucket and the no-LV program is built). Not counted (step_ok counts);
        public for harnesses that emulate step_ok."""
        cfg = self.cfg
        bucket, why, detail = classify_step(
            reqs,
            num_chunks=num_chunks,
            has_encoder=has_encoder,
            qmax=cfg.QMAX,
            rows=[t for t in self.rows if t in self._steps],
            bmax=cfg.BMAX,
        )
        if bucket is None:
            return None, why, detail
        total = sum(int(n) for n, _, _ in reqs)
        full = self.lv_full and bucket in self._steps_full and fills_bucket(total, bucket)
        return StepBucket(bucket, full), why, detail

    def _decide(self, runner, scheduler_output, num_chunks: int):
        if not self.enabled:
            return None, "off", ""
        if self.unbound is not None or not self._steps:
            return None, "unbound", str(self.unbound)
        return self.decision(
            sched_reqs(runner, scheduler_output),
            num_chunks=num_chunks,
            has_encoder=bool(scheduler_output.scheduled_encoder_inputs),
        )

    def _count(self, why: str, detail: str = "", bucket: int | None = None) -> None:
        c = self.counts
        first = c.get(why, 0) == 0
        c[why] = c.get(why, 0) + 1
        if bucket is not None:
            c[f"mk_b{int(bucket)}"] = c.get(f"mk_b{int(bucket)}", 0) + 1
            if getattr(bucket, "full", False):
                c[f"mk_b{int(bucket)}_full"] = c.get(f"mk_b{int(bucket)}_full", 0) + 1
        if why != "embeds" and why != "shape":
            self.decisions += 1
        if self.rank == 0 and first:
            logger.info(
                "GLM-5.2 decode megakernel: first '%s' step (%s)%s",
                why,
                detail,
                "" if why == "mk" else ": runs stock",
            )
        if (
            self.rank == 0
            and self.log_every
            and self.decisions % self.log_every == 0
            and why not in ("embeds", "shape")
        ):
            self.log_counts()

    def log_counts(self) -> None:
        c = {k: v for k, v in self.counts.items() if v}
        logger.info(
            "GLM-5.2 decode megakernel engagement: %d decisions, mk %d (%s), declined %s; calls %d, "
            "decisions consumed %d",
            self.decisions,
            c.get("mk", 0),
            ", ".join(f"{k} {v}" for k, v in c.items() if k.startswith("mk_b")) or "-",
            {k: v for k, v in c.items() if k != "mk" and not k.startswith("mk_b")}
            or "none",
            self.calls,
            self.consumed,
        )

    def rows_ok(self, q_lens: list[int], num_tokens: int) -> bool:
        """The same predicate from host-known rows (the twin harness, which calls forward_model directly): every row
        group a verify request, through classify_step; the twin also needs its padded size to be the bucket."""
        if (
            not self.enabled
            or self.unbound is not None
            or not self._steps
            or not q_lens
        ):
            return False
        cfg = self.cfg
        bucket, _, _ = classify_step(
            [(int(q), int(q) - 1, False) for q in q_lens],
            num_chunks=1,
            has_encoder=False,
            qmax=cfg.QMAX,
            rows=[t for t in self.rows if t in self._steps],
            bmax=cfg.BMAX,
        )
        return bucket is not None and bucket == num_tokens

    def _metadata(self):
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        if isinstance(md, dict):
            md = md.get(self.mla_layer_name)
        assert md is not None and hasattr(md, "query_start_loc"), type(md)
        return md

    def try_call(
        self,
        runner,
        decision,
        input_ids,
        positions,
        inputs_embeds=None,
        intermediate_tensors=None,
        harness=False,
    ):
        """forward_model's hook. `decision` is step_ok's bucket for this step (consumed by the caller; `harness`:
        the twin harness's rows_ok bucket, not a step decision and not counted). -> the backbone hidden states
        [n, H], or None = run stock. Counts the call-time declines ('embeds' / 'shape')."""
        if not harness:
            self.consumed += 1
        if decision is None:
            return None
        if inputs_embeds is not None or intermediate_tensors is not None:
            self._count("embeds", "inputs_embeds / intermediate tensors", None)
            return None
        md = self._metadata()
        why = self._shape_mismatch(decision, input_ids, md, positions)
        if why:
            self._count("shape", why, None)
            warned = self.__dict__.setdefault("_shape_warned", set())
            if self.rank == 0 and why not in warned:
                warned.add(why)
                logger.warning(
                    "GLM-5.2 decode megakernel: step declined to stock, an unexpected shape would recompile "
                    "the step program: %s",
                    why,
                )
            return None
        return self(runner, input_ids, positions, bucket=decision, md=md)

    def _shape_mismatch(self, bucket: int, input_ids, md, positions=None) -> str:
        """The step program is compiled with static shapes (torch.compile dynamic=False). Token operands: any 1..bucket
        rows are padded to the bucket outside the compiled step (`to_bucket_rows`: the runner's smaller buckets,
        TPU_TOKEN_BUCKET_EXTRA=4,8), anything else declines (`token_rows_mismatch`). Metadata: the first served call
        of a bucket fixes its metadata shapes (logged, with whether they equal the warmup's: if not, that call
        compiles once); a later call with other shapes declines ('shape') instead of recompiling mid-serving. The
        runner's shapes are config-static (query_start_loc is max_num_reqs + 1 under use_max_model_len), so this is a
        tripwire, rank-uniform like the shapes themselves."""
        if bucket not in self._steps:
            return f"b{bucket} not built"
        why = token_rows_mismatch(bucket, input_ids, positions)
        if why:
            return why
        got = (
            tuple(md.query_start_loc.shape),
            tuple(md.seq_lens.shape),
            int(md.block_tables.numel()),
        )
        want = self.call_shapes.get(bucket)
        if want is None:
            self.call_shapes[bucket] = got
            warm = self.warm_shapes.get(bucket)
            if self.rank == 0:
                (logger.info if got == warm else logger.warning)(
                    "GLM-5.2 decode megakernel: b%d served metadata shapes %s (warmup %s%s)",
                    bucket,
                    got,
                    warm,
                    "" if got == warm else ": this first call compiles once",
                )
            return ""
        if got != want:
            return (
                f"b{bucket}: qsl / seq_lens / block_tables {got} != the bucket's {want}"
            )
        return ""

    @torch.no_grad()
    def _run(self, step, ids, pos, qsl, seq, bt) -> torch.Tensor:
        """The one place a compiled step program is called, by serving (`__call__`)
        and `warmup` alike, so both run it in the same global state. Dynamo guards
        the torch.compile'd step on the grad mode (GLOBAL_STATE). Serving runs here
        under no_grad; capture_model (which calls `warmup`) runs with grad enabled.
        A warmup that called the programs directly compiled entries serving never
        hit, and every program re-traced on its first served call
        (stageB/best-ab-lv: 0.8-1.7 s decode stalls inside the measured window)."""
        weights = self.dense + self.experts
        return step(ids, pos, qsl, seq, bt, self.smalls, weights, self.caches)

    @torch.no_grad()
    def __call__(
        self,
        runner,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        bucket=None,
        md=None,
    ) -> torch.Tensor:
        n = input_ids.shape[0]
        t = bucket if bucket is not None else self.bucket_for(n)
        assert t is not None, (n, self.rows)
        md = md if md is not None else self._metadata()
        steps = self._steps_full if (getattr(bucket, "full", False) and t in self._steps_full) else self._steps
        t = int(t)
        out = self._run(
            steps[t],
            to_bucket_rows(input_ids, t),
            to_bucket_rows(positions, t),
            md.query_start_loc,
            md.seq_lens,
            md.block_tables,
        )
        self.calls += 1
        if n <= out.shape[0]:
            return out[:n]
        return torch.nn.functional.pad(out, (0, 0, 0, n - out.shape[0]))

    def status(self) -> dict:
        return {
            "rank": self.rank,
            "enabled": self.enabled,
            "rows": self.rows,
            "built": sorted(self._steps),
            "built_full": sorted(self._steps_full),
            "lv_full": self.lv_full,
            "L": self.cfg.L,
            "fresh_layers": LY.layers_of(self.cfg, "iwq"),
            "idx_freq_skip": [self.cfg.idx_freq, self.cfg.idx_skip],
            "n_dense": len(self.dense),
            "n_experts": len(self.experts),
            "n_caches": len(self.caches or []),
            "nb": getattr(self, "nb", None),
            "max_pages": self.max_pages,
            "dcp": getattr(self, "dcp", None),
            "v16": self.cfg.VR > 1,
            "pool_pages": getattr(self, "pool_pages", None),
            "trace": self.tiles.trace,
            "levers": list(self.levers),
            "calls": self.calls,
            "cache_views": getattr(self, "cache_views", None),
            "k_scales": sorted(set(self.k_scales)),
            "rope_neox": self.cfg.rope_neox,
            "qrep_layers": len(getattr(self, "qrep_layers", []) or []),
            "counts": dict(self.counts),
            "decisions": self.decisions,
            "consumed": self.consumed,
            "cache_form": self.cache_form,
            "groups": dict(getattr(self, "groups", {}) or {}),
            "unbound": self.unbound,
            "rank_layout": self.rank_layout,
            "call_shapes": {str(k): v for k, v in self.call_shapes.items()},
            "warm_shapes": {str(k): v for k, v in self.warm_shapes.items()},
            "warm_replay_compiles": getattr(self, "warm_replay_compiles", None),
        }

    def warmup(self, runner) -> None:
        """Compile and run every built step program once, through `_run` (serving's
        call and grad mode): each bucket's program and, with lever LV, its
        full-bucket no-LV program (`_steps_full`). Input: one request of QMAX rows at
        positions 0..QMAX-1 on block 0 (the null block; nothing is served from it
        before the first step). Then replay every program once more and count dynamo
        compiles: a served-mode call after warmup must compile nothing (logged;
        `status()['warm_replay_compiles']`)."""
        from torch._dynamo.utils import counters

        from vllm_torchtpu.utils import synchronize_tensors

        cfg = self.cfg
        dev = self.smalls.device
        if self.unbound is not None:
            return
        programs = []
        for t in self.rows:
            ids = torch.zeros(t, dtype=torch.int32, device=dev)
            pos = torch.arange(t, dtype=torch.int32, device=dev) % cfg.QMAX
            S = cfg.BMAX
            qsl = torch.tensor([0] + [cfg.QMAX] * S, dtype=torch.int32, device=dev)
            seq = torch.tensor(
                [cfg.QMAX] + [0] * (S - 1), dtype=torch.int32, device=dev
            )
            bt = torch.zeros(S * self.max_pages, dtype=torch.int32, device=dev)
            for tag, steps in (("", self._steps), (" (full-bucket, no LV)", self._steps_full)):
                if t not in steps:
                    continue
                args = (steps[t], ids, pos, qsl, seq, bt)
                t0 = time.monotonic()
                synchronize_tensors([self._run(*args)])
                self.warm_shapes[t] = (tuple(qsl.shape), tuple(seq.shape), int(bt.numel()))
                programs.append((t, tag, args))
                logger.info(
                    "GLM-5.2 decode megakernel: b%d%s compiled and warmed in %.1f s",
                    t,
                    tag,
                    time.monotonic() - t0,
                )
        g0 = counters["stats"]["unique_graphs"]
        for _, _, args in programs:
            synchronize_tensors([self._run(*args)])
        self.warm_replay_compiles = counters["stats"]["unique_graphs"] - g0
        (logger.info if self.warm_replay_compiles == 0 else logger.warning)(
            "GLM-5.2 decode megakernel: warmed %d step programs (%s); "
            "served-mode replay compiled %d new graphs",
            len(programs),
            ", ".join(f"b{t}{'f' if tag else ''}" for t, tag, _ in programs),
            self.warm_replay_compiles,
        )
