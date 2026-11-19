# SPDX-License-Identifier: Apache-2.0
"""q38 L4 SEEDFIX (VLLM_TPU_SEEDED_SAMPLING, default on; host-only, in _TPU_COMPILE_ENV_IGNORED).

Requests with their own ``SamplingParams.seed`` draw their sampling uniforms from their per-request generator
(vLLM ``InputBatch.generators``: batch index -> torch.Generator, created from the seed on every TP rank), so their
output is reproducible by seed. Every chunk still draws its uniforms from the runner's one sampling generator
(tpu_runner ``_get_sampling_generator``) with today's shapes and order; this module then overwrites the seeded rows'
slices of those arrays (an overlay), so unseeded rows read exactly the uniforms they read without it.

Per step (``build_plan``, called once by sample_tokens only when ``input_batch.generators`` is non-empty):
  * the seeded rows that sample this step: every seeded batch index except partial (chunked-prefill) requests, whose
    sampled token is discarded (the same test as sample_tokens' discard loop). Partial rows draw nothing, and
    sample_tokens then skips its per-request ``set_offset(-4)`` rewind for them (nothing was drawn to rewind);
  * each such request draws ONE fixed set from its own generator, in this order, whatever its chunk does this step:
      bonus [V], accept [K], recover [K, V]       (V = runner.vocab_size, K = num_speculative_tokens; K = 0: bonus only)
      cbonus [W], crecover [K, W]                  (only under VLLM_TPU_VPS_DRAWS=candidate; W = TP * VP_KC)
    The set depends only on the config, never on the verify route, the request's draft count, the chunk's padding,
    the fallback-row count F or the other requests, so a request's generator advances by the same amount on every
    sampled step and its n-th step's uniforms are a function of (seed, n) alone. Ranks agree because the plan is a
    function of the scheduler output and every rank's generator for a request is seeded identically.

Overlay (per seeded row, fixed shapes; ``index_put_`` with index tensors cached on device by (start, length), so
steady decode uploads nothing):
  * bonus-like arrays [B_pad, V] (or [B_pad, W] with candidate draws): row b = batch index - chunk start <- bonus
    (cbonus). The same piece serves the non-spec first-token draw (``u``), the verify bonus draw and the stock path;
  * accept [T_pad] and recover [T_pad, V] (or [T_pad, W]): the request's draft slots t0 .. t0+d-1, contiguous by
    construction (t0 = sum of the chunk's earlier draft lengths, speculative_decoding_manager), <- accept[:d] and
    recover[:d] (crecover[:d]);
  * fallback rows fb_u [F, V] (candidate draws only): the request's range verify rows (top_k outside [1, VP_KC]) sit
    at slots the host can compute (the verify sorts range rows first, stably, bonus rows then target rows); each
    covered slot <- the row's full piece (bonus, or recover[j]), so a seeded range row computes the stock arithmetic
    on the uniforms the stock route would give it.
Limits (DRAWS=candidate only; the served env is DRAWS=full): an exact-tie row's slot is decided in-graph, so a seeded
tie row reads the shared fallback uniform; in-range rows take per-candidate (VPB) uniforms on the VPS route and the
full pieces on the stock route, so their token can depend on which route the chunk took.

Warm-up (``warm_rows`` / ``warm_bucket``): every eager program the overlay can issue (the piece draws on a scratch
generator, the index uploads, the index_put_ / narrow / index_select at every steady bucket width) runs during the
runner's precompile, so a first seeded request in serving compiles nothing. Flag off: base behaviour bit for bit
(the platform rejects seeded requests again, no plan, no warm-up, the rewind as before).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SAMPLING_EPS = 1e-5  # tpu_runner / vp_sampler SAMPLING_EPS (vllm's)
LOG_EVERY = 1000  # seeded steps between SEEDFIX count lines
_IDX_CACHE_MAX = 8192

_counts = {"steps": 0, "rows": 0, "partial": 0, "rows_ovl": 0, "tgt_ovl": 0, "fb_ovl": 0, "fb_tie_or_uncovered": 0,
           "width_skip": 0}
_logged_first = False


def enabled() -> bool:
    return bool(envs.VLLM_TPU_SEEDED_SAMPLING)


def counts() -> dict:
    return dict(_counts)


def _reset_for_tests() -> None:
    global _logged_first
    for k in _counts:
        _counts[k] = 0
    _logged_first = False


def _note(runner: Any, site: str, key: tuple) -> None:
    note = getattr(runner, "_eager_shape_note", None)
    if note is not None:
        note(site, key)


def dev_index(runner: Any, start: int, length: int, device: torch.device) -> torch.Tensor:
    """int64 [length] = start .. start+length-1 on `device`, uploaded once per (start, length, device) and cached."""
    cache = runner.__dict__.get("_seedfix_idx")
    if cache is None:
        cache = {}
        runner._seedfix_idx = cache
    key = (start, length, str(device))
    t = cache.get(key)
    if t is None:
        if len(cache) >= _IDX_CACHE_MAX:
            cache.clear()
        _note(runner, "seedfix_h2d", (length,))
        t = torch.arange(start, start + length, dtype=torch.int64).to(device)
        cache[key] = t
    return t


class SeedPlan:
    """One step's seeded rows and their pieces (drawn at build time)."""

    __slots__ = ("rows", "partial", "vocab", "k", "cw", "pieces", "chunk")

    def __init__(self, rows: list[int], partial: list[int], vocab: int, k: int, cw: int) -> None:
        self.rows = rows  # sorted batch indices that sample this step
        self.partial = partial
        self.vocab, self.k, self.cw = vocab, k, cw
        self.pieces: dict[int, tuple] = {}
        self.chunk: tuple | None = None  # (start, end, md, logits dtype) of the VPS chunk being verified

    def in_chunk(self, start: int, end: int) -> list[int]:
        return [i for i in self.rows if start <= i < end]


def draw_pieces(gen: torch.Generator, vocab: int, k: int, cw: int, device: torch.device) -> tuple:
    """The fixed per-step draw set of one seeded request, in its fixed order."""
    f32 = torch.float32
    bonus = torch.rand((vocab,), dtype=f32, device=device, generator=gen)
    accept = recover = cbonus = crecover = None
    if k:
        accept = torch.rand((k,), dtype=f32, device=device, generator=gen)
        recover = torch.rand((k, vocab), dtype=f32, device=device, generator=gen)
        if cw:
            cbonus = torch.rand((cw,), dtype=f32, device=device, generator=gen)
            crecover = torch.rand((k, cw), dtype=f32, device=device, generator=gen)
    return bonus, accept, recover, cbonus, crecover


def _cand_width(runner: Any) -> int:
    vh = getattr(runner, "_vps_head", None)
    if vh is None or not getattr(runner, "_vps_cand_draws", False):
        return 0
    from vllm_torchtpu.layers.adapter.sample import vp_sampler

    return int(vh.tp_size) * vp_sampler.VP_KC


def _num_spec(runner: Any) -> int:
    sc = getattr(runner, "speculative_config", None)
    return int(sc.num_speculative_tokens) if sc is not None else 0


def build_plan(runner: Any, scheduler_output: Any) -> SeedPlan | None:
    """The step's plan, or None (flag off, or no seeded request samples this step). Draws every seeded row's
    pieces now, in batch-index order."""
    global _logged_first
    if not enabled():
        return None
    batch = runner.input_batch
    gens = batch.generators
    num_reqs = batch.num_reqs
    rows: list[int] = []
    partial: list[int] = []
    for i in sorted(gens):
        if i >= num_reqs:
            continue
        req_id = batch.req_ids[i]
        if req_id is None:
            continue
        rs = runner.requests[req_id]
        n_sched = scheduler_output.num_scheduled_tokens.get(req_id)
        if n_sched is None:
            continue
        if rs.num_computed_tokens + n_sched < rs.num_tokens:
            partial.append(i)
        else:
            rows.append(i)
    _counts["partial"] += len(partial)
    if not rows:
        return None
    plan = SeedPlan(rows, partial, int(runner.vocab_size), _num_spec(runner), _cand_width(runner))
    device = runner.device
    for i in rows:
        plan.pieces[i] = draw_pieces(gens[i], plan.vocab, plan.k, plan.cw, device)
    _counts["steps"] += 1
    _counts["rows"] += len(rows)
    if not _logged_first or _counts["steps"] % LOG_EVERY == 0:
        _logged_first = True
        logger.info(
            "q38 SEEDFIX steps=%d rows=%d partial=%d rows_ovl=%d tgt_ovl=%d fb_ovl=%d fb_tie_or_uncovered=%d "
            "width_skip=%d (V=%d K=%d W=%d)",
            _counts["steps"], _counts["rows"], _counts["partial"], _counts["rows_ovl"], _counts["tgt_ovl"],
            _counts["fb_ovl"], _counts["fb_tie_or_uncovered"], _counts["width_skip"], plan.vocab, plan.k, plan.cw,
        )
    return plan


def overlay_rows(runner: Any, plan: SeedPlan, u: torch.Tensor, start: int, end: int, cand: bool = False) -> None:
    """Bonus-like array u [B_pad, width]: each seeded row's row <- its bonus (cbonus) piece."""
    for i in plan.in_chunk(start, end):
        src = plan.pieces[i][3 if cand else 0]
        if src is None or src.shape[0] != u.shape[-1] or i - start >= u.shape[0]:
            _counts["width_skip"] += 1
            continue
        _note(runner, "seedfix_rows", (tuple(u.shape), int(src.shape[0])))
        u.index_put_((dev_index(runner, i - start, 1, u.device),), src.view(1, -1))
        _counts["rows_ovl"] += 1


def overlay_targets(
    runner: Any,
    plan: SeedPlan,
    accept_u: torch.Tensor | None,
    recover_u: torch.Tensor | None,
    start: int,
    end: int,
    md: Any,
    cand: bool = False,
) -> None:
    """accept [T_pad] / recover [T_pad, width]: each seeded row's draft slots t0 .. t0+d-1 <- accept[:d] /
    recover[:d] (crecover[:d])."""
    if plan.k == 0:
        return
    dl = md.draft_lengths_cpu
    for i in plan.in_chunk(start, end):
        b = i - start
        if b >= len(dl):
            continue
        d = min(int(dl[b]), plan.k)
        if d <= 0:
            continue
        t0 = int(np.sum(dl[:b]))
        p = plan.pieces[i]
        if accept_u is None and recover_u is None:
            return
        dev = accept_u.device if accept_u is not None else recover_u.device
        idx = dev_index(runner, t0, d, dev)
        if accept_u is not None:
            _note(runner, "seedfix_acc", (tuple(accept_u.shape), d))
            accept_u.index_put_((idx,), p[1][:d])
        if recover_u is not None:
            src = p[4] if cand else p[2]
            if src is None or src.shape[1] != recover_u.shape[-1]:
                _counts["width_skip"] += 1
            else:
                _note(runner, "seedfix_rec", (tuple(recover_u.shape), d))
                recover_u.index_put_((idx,), src[:d])
        _counts["tgt_ovl"] += 1


def overlay_verify(runner: Any, plan: SeedPlan, bonus_u, accept_u, recover_u, start: int, end: int, md: Any,
                   cand: bool = False) -> None:
    if bonus_u is not None:
        overlay_rows(runner, plan, bonus_u, start, end, cand)
    overlay_targets(runner, plan, accept_u, recover_u, start, end, md, cand)


def fallback_slots(runner: Any, start: int, end: int, md: Any, dtype: torch.dtype, kc: int) -> tuple[dict, dict]:
    """Host copy of vp_sampler.vp_verify_fb's range-row slot order for the chunk [start, end): ({bonus row b: slot},
    {target slot t: slot}). Range rows (non-greedy, top_k outside [1, kc] after `_padded_sampling_params`' >= vocab
    -> 0) take prio 0 in a stable sort over [bonus rows, target rows], so their slots are their ranks in that order."""
    n = end - start
    temps = runner.input_batch.temperature_cpu_tensor[start:end].to(dtype)
    k = runner.input_batch.top_k_cpu_tensor[start:end]
    k = torch.where(k >= runner.vocab_size, torch.zeros_like(k), k)
    in_range = (k >= 1) & (k <= kc)
    range_b = (~(temps <= SAMPLING_EPS) & ~in_range).tolist()
    range_t = ((temps != 0) & ~in_range).tolist()
    dl = [int(x) for x in md.draft_lengths_cpu[:n]]
    slot_b, slot_t, s = {}, {}, 0
    for b in range(n):
        if range_b[b]:
            slot_b[b] = s
            s += 1
    t = 0
    for b in range(n):
        for _ in range(dl[b]):
            if range_t[b]:
                slot_t[t] = s
                s += 1
            t += 1
    return slot_b, slot_t


def overlay_fb(runner: Any, plan: SeedPlan, fb_u: torch.Tensor, fallback_rows: int) -> None:
    """fb_u [F, V] (DRAWS=candidate): each covered range verify row of a seeded request <- its full piece."""
    if plan.chunk is None or plan.k == 0:
        return
    start, end, md, dtype = plan.chunk
    rows = plan.in_chunk(start, end)
    if not rows:
        return
    from vllm_torchtpu.layers.adapter.sample import vp_sampler

    slot_b, slot_t = fallback_slots(runner, start, end, md, dtype, vp_sampler.VP_KC)
    dl = md.draft_lengths_cpu
    dev = fb_u.device
    for i in rows:
        b = i - start
        p = plan.pieces[i]
        if p[0].shape[0] != fb_u.shape[-1]:
            _counts["width_skip"] += 1
            continue
        targets = []
        if b in slot_b:
            targets.append((slot_b[b], None))
        d = min(int(dl[b]), plan.k) if b < len(dl) else 0
        t0 = int(np.sum(dl[:b])) if d else 0
        for j in range(d):
            if t0 + j in slot_t:
                targets.append((slot_t[t0 + j], j))
        for slot, j in targets:
            if slot >= fallback_rows:
                _counts["fb_tie_or_uncovered"] += 1
                continue
            _note(runner, "seedfix_fb", (tuple(fb_u.shape),))
            if j is None:
                src = p[0].view(1, -1)
            else:
                src = torch.index_select(p[2], 0, dev_index(runner, j, 1, dev))
            fb_u.index_put_((dev_index(runner, slot, 1, dev),), src)
            _counts["fb_ovl"] += 1


# ------------------------------------------------------------------------------------------------ warm-up


def _warm_keys(runner: Any) -> set:
    s = runner.__dict__.get("_seedfix_warmed")
    if s is None:
        s = set()
        runner._seedfix_warmed = s
    return s


def warm_pieces(runner: Any) -> None:
    """The piece draws on a fresh scratch generator (a new request's generator is fresh too), 4 rounds so every
    generator-state phase is covered (torch_tpu materializes the state every 4 updates; a step makes 3 or 5 draws),
    and the index uploads of every length the overlay uses (1 .. K). The serving generators are never touched."""
    done = _warm_keys(runner)
    if "pieces" in done:
        return
    done.add("pieces")
    vocab, k, cw = int(runner.vocab_size), _num_spec(runner), _cand_width(runner)
    gen = torch.Generator(device=runner.device)
    gen.manual_seed(0)
    for _ in range(4):
        draw_pieces(gen, vocab, k, cw, runner.device)
    for length in range(1, max(k, 1) + 1):
        dev_index(runner, 0, length, runner.device)


def _warm_rows(runner: Any, b_pad: int, width: int) -> None:
    key = ("rows", b_pad, width)
    done = _warm_keys(runner)
    if key in done:
        return
    done.add(key)
    dev = runner.device
    u = torch.empty((b_pad, width), dtype=torch.float32, device=dev)
    src = torch.empty((width,), dtype=torch.float32, device=dev)
    _note(runner, "seedfix_rows", ((b_pad, width), width))
    u.index_put_((dev_index(runner, 0, 1, dev),), src.view(1, -1))


def _warm_targets(runner: Any, t_pad: int, width: int, k: int) -> None:
    key = ("tgt", t_pad, width)
    done = _warm_keys(runner)
    if key in done:
        return
    done.add(key)
    dev = runner.device
    acc = torch.empty((t_pad,), dtype=torch.float32, device=dev)
    rec = torch.empty((t_pad, width), dtype=torch.float32, device=dev)
    acc_src = torch.empty((k,), dtype=torch.float32, device=dev)
    rec_src = torch.empty((k, width), dtype=torch.float32, device=dev)
    for d in range(1, min(k, t_pad) + 1):
        idx = dev_index(runner, 0, d, dev)
        _note(runner, "seedfix_acc", ((t_pad,), d))
        acc.index_put_((idx,), acc_src[:d])
        _note(runner, "seedfix_rec", ((t_pad, width), d))
        rec.index_put_((idx,), rec_src[:d])


def _warm_fb(runner: Any, f: int, vocab: int, k: int) -> None:
    key = ("fb", f, vocab)
    done = _warm_keys(runner)
    if key in done or f <= 0:
        return
    done.add(key)
    dev = runner.device
    fb = torch.empty((f, vocab), dtype=torch.float32, device=dev)
    bonus = torch.empty((vocab,), dtype=torch.float32, device=dev)
    rec = torch.empty((k, vocab), dtype=torch.float32, device=dev)
    idx = dev_index(runner, 0, 1, dev)
    _note(runner, "seedfix_fb", ((f, vocab),))
    fb.index_put_((idx,), bonus.view(1, -1))
    fb.index_put_((idx,), torch.index_select(rec, 0, idx))


def warm_rows(runner: Any, b_pad: int, width: int) -> None:
    """Warm-up of a non-spec sampling bucket (u [b_pad, width]). No-op with the flag off."""
    if not enabled():
        return
    warm_pieces(runner)
    _warm_rows(runner, b_pad, width)


def warm_bucket(runner: Any, num_tokens: int, num_reqs: int, fb_rows: list[int] | tuple = ()) -> None:
    """Warm-up of one spec-verify bucket (T_pad = num_tokens, B_pad = num_reqs): the overlay's programs at the full
    width (stock routes and VPS full draws), at the candidate width under DRAWS=candidate, and fb_u at each fallback
    row count. No-op with the flag off."""
    if not enabled():
        return
    warm_pieces(runner)
    vocab, k, cw = int(runner.vocab_size), _num_spec(runner), _cand_width(runner)
    _warm_rows(runner, num_reqs, vocab)
    if k:
        _warm_targets(runner, num_tokens, vocab, k)
    if cw:
        _warm_rows(runner, num_reqs, cw)
        if k:
            _warm_targets(runner, num_tokens, cw, k)
        for f in fb_rows:
            _warm_fb(runner, f, vocab, k)
