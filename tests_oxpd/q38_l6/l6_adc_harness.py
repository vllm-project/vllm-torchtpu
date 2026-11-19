# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever adcarry: a CPU harness over the CARRY harness (l6_carry_harness.py)
that also drives the real Qwen3.8 adapter as ``execute_model`` does:
``Qwen4ExpAdapter.begin_step``, then ``TPUModelRunner._prepare_inputs``, then
``Qwen4ExpAdapter.prepare_inputs`` on the step's input ids (the fused
program's substituted ids when it ran).

Works on the base tree too (the lever-off identity proof imports nothing the
base lacks): ``runner.adapt_carry`` is imported only when the lever is set.

* Every ``Tensor.to(<torch.device>)`` is a counted copy (``H.Uploads``), so a
  reused device tensor keeps its values and the adapter's window / scalars
  uploads are counted with the runner's.
* The compiled programs run as the eager functions they wrap (the post-forward
  and carry programs, ``fused_step_inputs`` and, with the lever, the two merged
  programs).
* ``AdcSim.step`` returns one digest per device input of the forward: every
  ``_prepare_inputs`` output (attention-metadata context, positions, input ids
  and drafts, logits / substitution indices, spec metadata), every adapter
  kwarg (n-gram context, state slots, context lengths, valid tokens, query
  starts, block tables), both n-gram records and both row maps.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace as NS

import numpy as np
import torch

import l6_carry_harness as H
from vllm_torchtpu.core import l6_ab

K = H.K
STRIDE = H.STRIDE
ATTN = "model.layers.3.self_attn"
ADAPTER_ENV = ("VLLM_TPU_QWEN4_FUSED_PREP", "VLLM_TPU_QWEN4_NGRAM_WINDOW",
               "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL", "VLLM_TPU_QWEN4_FUSED_PREP_CHECK")


def adapter_env(window: bool = True, functional: bool = True) -> dict[str, str]:
    """The served adapter configuration (FUSED_PREP, NGRAM_WINDOW=1, FUNCTIONAL)."""
    return {
        "VLLM_TPU_QWEN4_FUSED_PREP": "1",
        "VLLM_TPU_QWEN4_NGRAM_WINDOW": "1" if window else "",
        "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL": "1" if functional else "0",
    }


def install_eager_programs():
    """The CARRY harness's eager programs, plus the merged programs -> their
    eager functions. Returns an undo callable."""
    undo = [H.install_eager_programs()]
    try:
        from vllm_torchtpu.runner import adapt_carry
    except ImportError:  # the base tree
        adapt_carry = None
    if adapt_carry is not None:
        prev = (adapt_carry.program_postfwd, adapt_carry.program_carry)
        adapt_carry.program_postfwd = adapt_carry.merged_postfwd
        adapt_carry.program_carry = adapt_carry.merged_carry
        undo.append(lambda: (setattr(adapt_carry, "program_postfwd", prev[0]),
                             setattr(adapt_carry, "program_carry", prev[1])))

    def run_undo():
        for u in reversed(undo):
            u()

    return run_undo


def make_adapter(r, *, pooled: bool, ngram: int, eos: int = 7):
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter, fused_step_inputs

    a = object.__new__(Qwen4ExpAdapter)
    a.runner = r
    a.pooled = pooled
    a.config = NS(ngram_size=ngram, eos_token_id=eos)
    a.block_size = 16
    a.attention_prefix = "model.layers.3"
    a.cache = None
    a.token_store = None
    a.spec_window = K + 1
    a._recent = []
    a._previous_rows = {}
    a._current_rows = {}
    a._host_ple = None
    a._fused = fused_step_inputs  # the compiled program's function, eager
    r.model_adapter = a
    r.supports_mm_inputs = False
    return a


class AdcSim(H.Sim):
    """H.Sim with the real adapter (``prompt`` is every initial request's
    context; a small one gives short-context rows). Extra events:
    ``join_short`` (a request with a one-token context joins), ``preempt`` (request 0 leaves, keeping its id)
    and ``resume`` (the oldest preempted request rejoins at the end with the
    same id), ``idle_begin`` (an adapter ``begin_step`` with no chunk before
    the step), ``hold2`` / ``hold2_off`` (request 0 schedules 2 tokens from now
    on / again 1 + K)."""

    def __init__(self, runner, *, nreq, seed, prompt=40, pooled=False, ngram=3):
        self.adapter = make_adapter(runner, pooled=pooled, ngram=ngram)
        self.preempted: list[tuple[str, int]] = []
        self.force_rej: dict[int, int] = {}
        self.hold2: str | None = None
        self.region_ns: list[int] = []  # wall time of begin_step + _prepare_inputs + prepare_inputs
        super().__init__(runner, nreq=nreq, seed=seed, prompt=prompt, pooled=pooled)
        tbl = torch.arange(runner.max_num_reqs * 4, dtype=torch.int32)

        def build_attention_metadata(**_kwargs):
            md = {ATTN: NS(block_tables=tbl), ATTN + ".compressed_k": NS(block_tables=tbl + 1000)}
            md["layer.0"] = runner._attn_metadata_builder_ctx
            return md, None

        runner._build_attention_metadata = build_attention_metadata

    def _add_named(self, rid: str, computed: int) -> str:
        r, ib = self.r, self.r.input_batch
        i = ib.num_reqs
        ib.req_ids.append(rid)
        ib.req_id_to_index[rid] = i
        ib.num_reqs += 1
        ib.num_prompt_tokens[i] = self.prompt
        ib.num_tokens_no_spec[i] = self.prompt
        ib.token_ids_cpu[i] = self.rng.integers(1, 30000, ib.token_ids_cpu.shape[1]).astype(np.int32)
        r.requests[rid] = NS(req_id=rid, output_token_ids=[])
        self.computed[rid] = computed
        return rid

    def step(self, events=()):
        from vllm_torchtpu.runner.model_adapters import ModelInputSnapshot

        ev = set(events)
        r, ib, a = self.r, self.r.input_batch, self.adapter
        rng = self.rng
        join_rid = None
        if "join" in ev:
            join_rid = self._add(self.prompt - 5)
        if "join_short" in ev:
            join_rid = self._add(1)
        if "preempt" in ev:
            rid = ib.req_ids[0]
            self.preempted.append((rid, self.computed[rid]))
            self.leave(0)
        if "resume" in ev and self.preempted:
            rid, c = self.preempted.pop(0)
            join_rid = self._add_named(rid, max(c - 5, 1))
        if "leave" in ev:
            self.leave(0)
        if "swap" in ev:
            self.swap(0, ib.num_reqs - 1)
        if "skew" in ev:
            self.computed[ib.req_ids[0]] += 1
        if "flush" in ev:
            r._pre_async_results = None
        if "hold2" in ev:
            self.hold2 = ib.req_ids[0]
        if "hold2_off" in ev:
            self.hold2 = None
        n = ib.num_reqs
        req_ids = list(ib.req_ids[:n])
        nsched, nd = {}, {}
        for i, rid in enumerate(req_ids):
            if rid == join_rid:
                nsched[rid], nd[rid] = 5, 0  # the prompt tail, no drafts
            elif "nonspec" in ev:
                nsched[rid], nd[rid] = 1, 0
            elif ("partial" in ev and i == 0) or rid == self.hold2:
                nsched[rid], nd[rid] = 2, 1
            else:
                nsched[rid], nd[rid] = STRIDE, K
            ib.num_computed_tokens_cpu[i] = self.computed[rid]
        r._dp_target_bucket = max(r.num_tokens_paddings) if "pad" in ev else None
        so = NS(
            total_num_scheduled_tokens=sum(nsched.values()),
            num_scheduled_tokens=dict(nsched),
            scheduled_spec_decode_tokens={rid: [0] * nd[rid] for rid in req_ids if nd[rid]},
        )
        if l6_ab.ACTIVE:
            l6_ab.begin_step()
        t0 = time.perf_counter_ns()
        with H.Uploads() as up:
            if "idle_begin" in ev:
                a.begin_step()
            store = a.begin_step()
            out = r._prepare_inputs(so, 0, 0, n)
            fused = r.__dict__.pop("_postfwd_fused_out", None)
            spec = out[7]
            if fused is not None and (fused[0] != 0 or spec is None):
                fused = None
            ids = fused[1] if fused is not None else r.input_ids
            snap = ModelInputSnapshot(store, 0, n, int(ids.shape[0]), out[0], ids)
            kw = a.prepare_inputs(snap)
        self.region_ns.append(time.perf_counter_ns() - t0)
        self.uploads.append(up.n)
        dig = self._digest_adc(out, fused, kw)
        rej = [int(rng.integers(0, nd[rid] + 1)) for rid in req_ids]
        if self.n in self.force_rej:
            rej = [min(self.force_rej[self.n], nd[rid]) for rid in req_ids]
        r._modify_prev_results()
        discard = [n - 1] if "discard" in ev else []
        src_pos = list(range(n))
        if "remap" in ev:
            src_pos = [(i + 1) % n for i in range(n)]
        self._results(nd=[nd[rid] for rid in req_ids], rej=rej, discard=discard, src_pos=src_pos)
        for i, rid in enumerate(req_ids):
            self.computed[rid] += nsched[rid] - self.last_rej.get(rid, 0)
        self.last_rej = {rid: rej[i] for i, rid in enumerate(req_ids)}
        r._dp_target_bucket = None
        self.n += 1
        return dig

    def _digest_adc(self, out, fused, kw) -> dict[str, bytes]:
        r, a = self.r, self.adapter
        b = H._b
        (_md, logits, padded_num_reqs, num_reqs, end_index, cur, pre, spec, rm_ids, _plan) = out
        ctx = r._attn_metadata_builder_ctx
        d = {
            "seq_lens": b(ctx.seq_lens),
            "query_start_loc": b(ctx.query_start_loc),
            "request_distribution": b(ctx.request_distribution),
            "mamba_state_indices": b(ctx.mamba_state_indices),
            "mamba_request_distribution": b(ctx.mamba_request_distribution),
            "positions": b(r.position_ids),
            "logits_indices": b(logits),
            "counts": b(np.array([padded_num_reqs, num_reqs, end_index])),
            "cur_input_indices": b(cur),
            "pre_next_tokens_indices": b(pre),
            "request_major_input_ids": b(rm_ids),
        }
        if spec is not None:
            d["spec_final_logits_indices"] = b(spec.final_logits_indices)
            d["spec_target_logits_indices"] = b(spec.target_logits_indices)
            d["spec_draft_lengths"] = b(spec.draft_lengths)
        if fused is not None:
            d["fused"] = b(np.array([fused[0]]))
            d["input_ids"] = b(fused[1])
            d["draft_token_ids"] = b(fused[2])
        else:
            d["raw_input_ids"] = b(r.input_ids)
        for k in sorted(kw):
            d["adapter_" + k] = b(kw[k])
        for j, rec in enumerate(a._recent):
            d[f"adapter_record{j}"] = b"".join(b(t) for t in rec)
        d["adapter_current_rows"] = repr(sorted(a._current_rows.items())).encode()
        d["adapter_previous_rows"] = repr(sorted(a._previous_rows.items())).encode()
        return d


def warm(sim) -> None:
    """End of warm-up, as capture_model + the worker do it: the warmed lever
    programs, then ``warmup_done``."""
    from vllm_torchtpu.runner import decode_carry

    r = sim.r
    if decode_carry.flag_set():
        decode_carry.precompile(r)
    try:
        from vllm_torchtpu.runner import adapt_carry
    except ImportError:
        adapt_carry = None
    if adapt_carry is not None and adapt_carry.flag_set():
        adapt_carry.precompile(r)
    r.warmup_done = True


_RUNNER_KEYS = ("max_num_reqs", "paddings", "mamba", "comb", "max_len")


def _sim(shape: dict, seed: int, nreq: int) -> AdcSim:
    r = H.make_runner(seed=seed, **{k: v for k, v in shape.items() if k in _RUNNER_KEYS})
    return AdcSim(r, nreq=nreq, seed=seed, prompt=shape.get("prompt", 40), pooled=shape.get("pooled", False),
                  ngram=shape.get("ngram", 3))


def prewarm_keys(shape: dict, seed: int) -> set:
    """The post-forward shapes a served warm-up compiles (capture_model's
    ``_warmup_spec_decode`` runs real spec-decode steps per bucket): two
    steady steps at every request count 1..max_num_reqs on scratch runners,
    with every L6 lever unset (and l6_ab re-resolved after)."""
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("VLLM_TPU_L6_")}
    l6_ab._resolve()
    keys: set = set()
    try:
        for nreq in range(1, shape.get("max_num_reqs", 8) + 1):
            sim = _sim(shape, seed, nreq)
            for _ in range(2):
                sim.step()
            keys |= set(sim.r.__dict__.get("_postfwd_warm_keys", ()))
    finally:
        os.environ.update(saved)
        l6_ab._resolve()
    return keys


def run(shape: dict, steps: int, *, seed: int, events: dict[int, tuple] | None = None, on_step=None,
        warm_at: int | None = 2, force_rej: dict[int, int] | None = None, prewarm: bool = True):
    """Run one engine for `steps` steps; warm-up ends before step `warm_at`
    (the pre-warmed post-forward shapes join the runner's, the lever programs
    are precompiled, then ``warmup_done``). Returns (digests, sim)."""
    events = events or {}
    keys = prewarm_keys(shape, seed) if (prewarm and warm_at is not None) else set()
    sim = _sim(shape, seed, shape.get("nreq", 3))
    sim.force_rej = dict(force_rej or {})
    digests = []
    for _ in range(steps):
        if warm_at is not None and sim.n == warm_at:
            sim.r.__dict__.setdefault("_postfwd_warm_keys", set()).update(keys)
            warm(sim)
        if on_step is not None:
            on_step(sim)
        digests.append(sim.step(events.get(sim.n, ())))
    return digests, sim


digest_all = H.digest_all
first_difference = H.first_difference
