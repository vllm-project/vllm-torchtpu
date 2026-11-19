# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever carry: a CPU harness that drives the real
``TPUModelRunner._prepare_inputs``, ``_modify_prev_results`` and
``_update_placeholder`` through async speculative decode steps, with the
device side simulated (random rejections, random draft sources).

Works on the base tree too (the lever-off identity proof imports nothing
the base lacks): ``runner.decode_carry`` is only imported by tests that turn
the lever on.

* ``Uploads`` makes every ``Tensor.to(<torch.device>)`` return a copy and
  counts them. On CPU such a call would return the host staging tensor
  itself, so a "device" tensor would silently follow later host writes and
  hide a stale reuse. With it, a reused upload keeps the values it had.
* The compiled fused programs are replaced by the eager functions they wrap
  (``subtract_substitute_extract`` and, with the lever, the carry form).
* ``Sim.step`` returns one digest per output that leaves ``_prepare_inputs``
  for the device: the attention-metadata context (seq_lens, query_start_loc,
  request distributions, mamba state indices), positions, the substituted
  input ids and drafts (or the raw ids when the fused program did not run),
  the logits / substitution indices, the spec metadata and the Qwen3.8
  adapter's fused input prep (n-gram / PLE context, state slots, valid
  tokens: what the QSA and PLE layers read) computed from them.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace as NS

import numpy as np
import torch

from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.layers.core.sequence_layout import AllSequenceLayoutPlanner
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.speculative_decoding_manager import SpeculativeDecodingManager
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncPreResults,
    subtract_substitute_extract,
)

K = 3
STRIDE = 1 + K
NGRAM = 3


class Uploads:
    """Context manager: ``Tensor.to(<torch.device>)`` copies (like an H2D)
    and is counted."""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        self._had = "to" in torch.Tensor.__dict__
        self._prev = torch.Tensor.__dict__.get("to")
        orig = torch.Tensor.to
        me = self

        def to(t, *args, **kwargs):
            out = orig(t, *args, **kwargs)
            dev = args[0] if args else kwargs.get("device")
            if isinstance(dev, torch.device):
                me.n += 1
                if out is t:
                    out = t.clone()
            return out

        torch.Tensor.to = to
        return self

    def __exit__(self, *exc):
        if self._had:
            torch.Tensor.to = self._prev
        else:
            del torch.Tensor.to
        return False


def install_eager_programs():
    """The compiled fused programs -> the eager functions they wrap. Returns
    an undo callable."""
    prev = tpu_runner._postfwd_subst_fused
    tpu_runner._postfwd_subst_fused = subtract_substitute_extract
    undo = [lambda: setattr(tpu_runner, "_postfwd_subst_fused", prev)]
    try:
        from vllm_torchtpu.runner import decode_carry
    except ImportError:  # the base tree
        decode_carry = None
    if decode_carry is not None:
        prev_c = decode_carry.carry_fused
        decode_carry.carry_fused = decode_carry.carry_subtract_substitute_extract
        undo.append(lambda: setattr(decode_carry, "carry_fused", prev_c))

    def run_undo():
        for u in reversed(undo):
            u()

    return run_undo


def make_runner(
    *,
    nreq: int = 3,
    max_num_reqs: int = 8,
    paddings: tuple[int, ...] = (16, 32, 64),
    mamba: bool = True,
    comb: bool = True,
    prompt: int = 40,
    max_len: int = 1024,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    r = object.__new__(TPUModelRunner)
    r.device = torch.device("cpu")
    r.most_model_len = None
    r.num_reqs_max_model_len = max_num_reqs
    r.num_reqs_most_model_len = None
    r.num_tokens_paddings = list(paddings)
    r.max_num_reqs = max_num_reqs
    r.max_num_tokens = max(paddings)
    r.max_model_len = max_len
    r.uses_mrope = False
    r.supports_mm_inputs = False
    r.lora_config = None
    r.reorder_batch_threshold = 1
    r.is_pooling_model = False
    r._pcp_enabled = False
    r._mtp_enabled = False
    r.speculative_config = NS(num_speculative_tokens=K)
    r.scheduler_config = NS(async_scheduling=True)
    r.parallel_config = NS(
        prefill_context_parallel_size=1,
        decode_context_parallel_size=4,
        pipeline_parallel_size=1,
        cp_kv_cache_interleave_size=16,
    )
    r.vllm_config = NS(
        parallel_config=r.parallel_config,
        scheduler_config=r.scheduler_config,
        speculative_config=r.speculative_config,
        kv_transfer_config=None,
        compilation_config=NS(static_forward_context={}),
    )
    r.sequence_layout_planner = AllSequenceLayoutPlanner()
    T = max(paddings)
    r.input_ids_cpu = torch.zeros(T, dtype=torch.int32)
    r.positions_cpu = torch.zeros(T, dtype=torch.int32)
    r.positions_np = r.positions_cpu.numpy()
    r.query_start_loc_cpu = torch.zeros(T + 1, dtype=torch.int32)
    r.query_start_loc_np = r.query_start_loc_cpu.numpy()
    r.seq_lens_cpu = torch.zeros(T, dtype=torch.int32)
    r.seq_lens_np = r.seq_lens_cpu.numpy()
    r.arange_np = np.arange(T, dtype=np.int64)
    r._request_distribution_cpu = torch.zeros(3, dtype=torch.int32)
    r._combined_request_distribution_cpu = torch.zeros(6, dtype=torch.int32)
    r._decode_device_cache_key = None
    r._cached_query_start_loc = None
    r._cached_logits_indices = None
    r._cached_request_distribution = None
    r._input_staging_fence = None
    r._h2d_reuse = {}
    r._dp_target_bucket = None
    r._attention_capacity = {}
    r._attention_runs_batched_kernel = False
    r._fast_token_substitution = False
    r._unified_kv_layout = False
    r._mamba_ckpt_window = 1
    r._kv_cache_bundle = None
    r._mamba_copy_plan = []
    r.empty_slot_mappings = {0: torch.empty(0)}
    r._has_mamba_state = bool(mamba)
    r.mamba_state_indices_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
    r._mamba_slot_by_req_id = {}
    r._free_mamba_slots = [1 + g * STRIDE for g in reversed(range(4 * max_num_reqs))]
    r._mamba_offset_seeded = set()
    r.mamba_slot_read_offsets = torch.zeros(64, dtype=torch.int32) if comb else None
    r._pre_async_results = None
    r._last_sequence_layout_plan = None
    r.warmup_done = False

    def build_attention_metadata(**_kwargs):
        return {"layer.0": r._attn_metadata_builder_ctx}, None

    r._build_attention_metadata = build_attention_metadata
    tok = torch.from_numpy(rng.integers(1, 30000, (max_num_reqs + 4, max_len)).astype(np.int32))
    r.input_batch = NS(
        num_reqs=0,
        req_ids=[],
        req_id_to_index={},
        num_computed_tokens_cpu=np.zeros(max_num_reqs + 4, dtype=np.int32),
        num_prompt_tokens=np.zeros(max_num_reqs + 4, dtype=np.int32),
        num_tokens_no_spec=np.zeros(max_num_reqs + 4, dtype=np.int32),
        token_ids_cpu=tok.numpy(),
        token_ids_cpu_tensor=tok,
    )
    r.requests = {}
    r.spec_decode_manager = SpeculativeDecodingManager(r)
    return r


def _b(x) -> bytes:
    if x is None:
        return b"None"
    if isinstance(x, torch.Tensor):
        a = x.detach().contiguous().numpy()
    else:
        a = np.ascontiguousarray(np.asarray(x))
    return f"{a.dtype}{tuple(a.shape)}|".encode() + a.tobytes()


class Sim:
    """One engine's async-spec decode loop over a runner from make_runner.

    Scheduler model: request r schedules ``1 + nd`` tokens (``nd`` drafts) at
    ``num_computed = previous + previous scheduled - rejected count of the
    step before`` (the scheduler learns a step's rejections one step late),
    as vLLM's async scheduler does with one step in flight."""

    def __init__(self, runner, *, nreq: int, seed: int, prompt: int = 40, pooled: bool = False):
        self.r = runner
        self.rng = np.random.default_rng(1000 + seed)
        self.prompt = prompt
        self.pooled = pooled
        self.n = 0
        self.computed: dict[str, int] = {}
        self.last_rej: dict[str, int] = {}
        self.next_id = 0
        self.uploads: list[int] = []
        rows = runner.max_num_reqs
        self.record = (
            torch.zeros((rows, NGRAM - 1 + STRIDE), dtype=torch.int64),
            torch.zeros(rows, dtype=torch.int64),
            torch.zeros(rows, dtype=torch.int64),
        )
        for _ in range(nreq):
            self._add(self.prompt)
        self._bootstrap()

    # -- batch edits (applied identically to every run) --
    def _add(self, computed: int) -> str:
        r, ib = self.r, self.r.input_batch
        rid = f"q{self.next_id}"
        self.next_id += 1
        i = ib.num_reqs
        ib.req_ids.append(rid)
        ib.req_id_to_index[rid] = i
        ib.num_reqs += 1
        ib.num_prompt_tokens[i] = self.prompt
        ib.num_tokens_no_spec[i] = self.prompt  # the whole prompt is on the host
        ib.token_ids_cpu[i] = self.rng.integers(1, 30000, ib.token_ids_cpu.shape[1]).astype(np.int32)
        r.requests[rid] = NS(req_id=rid, output_token_ids=[])
        self.computed[rid] = computed
        return rid

    def _move(self, src: int, dst: int) -> None:
        ib = self.r.input_batch
        rid = ib.req_ids[src]
        ib.req_ids[dst] = rid
        ib.req_id_to_index[rid] = dst
        ib.token_ids_cpu[dst] = ib.token_ids_cpu[src]
        ib.num_tokens_no_spec[dst] = ib.num_tokens_no_spec[src]
        ib.num_prompt_tokens[dst] = ib.num_prompt_tokens[src]

    def leave(self, j: int) -> None:
        r, ib = self.r, self.r.input_batch
        rid = ib.req_ids[j]
        last = ib.num_reqs - 1
        if j != last:
            self._move(last, j)
        ib.req_ids.pop()
        ib.num_reqs -= 1
        del ib.req_id_to_index[rid]
        del r.requests[rid]
        del self.computed[rid]

    def swap(self, i: int, j: int) -> None:
        ib = self.r.input_batch
        a, b = ib.req_ids[i], ib.req_ids[j]
        ib.req_ids[i], ib.req_ids[j] = b, a
        ib.req_id_to_index[a], ib.req_id_to_index[b] = j, i
        for arr in (ib.token_ids_cpu, ib.num_tokens_no_spec, ib.num_prompt_tokens):
            tmp = arr[i].copy()
            arr[i] = arr[j]
            arr[j] = tmp

    def _bootstrap(self) -> None:
        """The prompt's prefill step: one sampled token, no drafts."""
        ib = self.r.input_batch
        n = ib.num_reqs
        self._results(
            nd=[0] * n,
            rej=[0] * n,
            discard=[],
            src_pos=list(range(n)),
        )

    def _results(self, *, nd, rej, discard, src_pos) -> None:
        r, ib = self.r, self.r.input_batch
        n = ib.num_reqs
        req_ids = list(ib.req_ids[:n])
        sampled = torch.full((n, STRIDE), INVALID_TOKEN_ID, dtype=torch.int32)
        for i in range(n):
            k = 1 + nd[i] - rej[i]
            sampled[i, :k] = torch.from_numpy(self.rng.integers(1, 30000, k).astype(np.int32))
        request_seq_lens = [(i, r.requests[rid], 0, rid) for i, rid in enumerate(req_ids)]
        nti = {i: src_pos[i] for i in range(n)}
        ndr = {i: int(nd[i]) for i in range(n)}
        mapping = r._update_placeholder(discard, request_seq_lens, nti, ndr)
        src = self.rng.integers(1, 30000, r.max_num_reqs * STRIDE).astype(np.int32)
        rej_dev = torch.zeros(r.max_num_reqs, dtype=torch.int32)
        for i in range(n):
            rej_dev[src_pos[i]] = int(rej[i])
        r._pre_async_results = AsyncPreResults(
            req_ids=req_ids,
            next_tokens_tpu=torch.from_numpy(src),
            request_seq_lens=request_seq_lens,
            discard_sampled_tokens_req_indices=list(discard),
            req_id_to_index_copy=mapping,
            copy_state=NS(wait=lambda: None, sampled_token_ids_cpu=sampled),
            spec_decode_num_rejected_tokens=rej_dev,
            num_draft_per_req=ndr,
        )

    def step(self, events=()):
        """One engine step; returns {output name: bytes}."""
        ev = set(events)
        r, ib = self.r, self.r.input_batch
        rng = self.rng
        if l6_ab.ACTIVE:
            # execute_model: the step's lever states, then (q38 L6-GATE, only with a gate set) the
            # batch-size gates, both before _update_states applies this step's admissions and
            # finishes to the batch (below). The gate hook reads the scheduler output: here the
            # set of requests this step schedules (only the set matters to the gate).
            l6_ab.begin_step()
            if getattr(l6_ab, "GATE", 0):
                scheduled = list(ib.req_ids[: ib.num_reqs])
                if "leave" in ev:
                    scheduled.pop(0)
                if "join" in ev:
                    scheduled.append(f"q{self.next_id}")
                r._l6_gate_step(NS(num_scheduled_tokens=dict.fromkeys(scheduled, 1)))
        join_rid = None
        if "join" in ev:
            join_rid = self._add(self.prompt - 5)
        if "leave" in ev:
            self.leave(0)
        if "swap" in ev:
            self.swap(0, ib.num_reqs - 1)
        if "skew" in ev:
            self.computed[ib.req_ids[0]] += 1
        if "flush" in ev:
            r._pre_async_results = None
        n = ib.num_reqs
        req_ids = list(ib.req_ids[:n])
        nsched, nd = {}, {}
        for i, rid in enumerate(req_ids):
            if rid == join_rid:
                nsched[rid], nd[rid] = 5, 0  # the prompt tail, no drafts
            elif "nonspec" in ev:
                nsched[rid], nd[rid] = 1, 0
            elif "partial" in ev and i == 0:
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
        with Uploads() as up:
            out = r._prepare_inputs(so, 0, 0, n)
        self.uploads.append(up.n)
        dig = self._digest(out, n)
        r.__dict__.pop("_postfwd_fused_out", None)  # execute_model takes it
        # sample_tokens: roll back the previous results, then park this step's
        rej = [int(rng.integers(0, nd[rid] + 1)) for rid in req_ids]
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

    def _digest(self, out, n) -> dict[str, bytes]:
        r = self.r
        (_md, logits, padded_num_reqs, num_reqs, end_index, cur, pre, spec, rm_ids, _plan) = out
        ctx = r._attn_metadata_builder_ctx
        fused = r.__dict__.get("_postfwd_fused_out")
        d = {
            "seq_lens": _b(ctx.seq_lens),
            "query_start_loc": _b(ctx.query_start_loc),
            "request_distribution": _b(ctx.request_distribution),
            "mamba_state_indices": _b(ctx.mamba_state_indices),
            "mamba_request_distribution": _b(ctx.mamba_request_distribution),
            "positions": _b(r.position_ids),
            "logits_indices": _b(logits),
            "counts": _b(np.array([padded_num_reqs, num_reqs, end_index])),
            "cur_input_indices": _b(cur),
            "pre_next_tokens_indices": _b(pre),
            "request_major_input_ids": _b(rm_ids),
        }
        if spec is not None:
            d["spec_final_logits_indices"] = _b(spec.final_logits_indices)
            d["spec_target_logits_indices"] = _b(spec.target_logits_indices)
            d["spec_draft_lengths"] = _b(spec.draft_lengths)
        if fused is not None:
            d["fused"] = _b(np.array([fused[0]]))
            d["input_ids"] = _b(fused[1])
            d["draft_token_ids"] = _b(fused[2])
            ids = fused[1]
        else:
            d["raw_input_ids"] = _b(r.input_ids)
            ids = r.input_ids
        d.update(self._adapter(ctx, ids, n))
        return d

    def _adapter(self, ctx, ids, n) -> dict[str, bytes]:
        """The Qwen3.8 adapter's fused input prep over this step's metadata."""
        from vllm_torchtpu.models.vllm.qwen4_exp.inputs import fused_step_inputs

        rows = int(ctx.seq_lens.shape[0])
        starts = ctx.query_start_loc[: rows + 1]
        state = ctx.mamba_state_indices if ctx.mamba_state_indices is not None else starts[:rows]
        scalars = torch.tensor([0, n] + list(range(rows)), dtype=torch.int64)
        out = fused_step_inputs(
            starts,
            ctx.seq_lens[:rows],
            state,
            self.r.input_batch.token_ids_cpu_tensor,
            scalars,
            self.record,
            self.record,
            ids,
            window=False,
            pooled=self.pooled,
            spec_window=STRIDE,
            eos_token_id=0,
            ngram_size=NGRAM,
            functional=True,
        )
        self.record = out[4]
        names = ("adapter_context", "adapter_history", "adapter_slots", "adapter_valid_tokens")
        d = {k: _b(v) for k, v in zip(names, out[:4])}
        d["adapter_record"] = b"".join(_b(t) for t in out[4])
        return d


def run(shape: dict, steps: int, *, seed: int, events: dict[int, tuple] | None = None,
        on_step=None):
    """Run one engine for `steps` steps; returns (digests, sim)."""
    events = events or {}
    r = make_runner(seed=seed, **{k: v for k, v in shape.items() if k != "pooled"})
    sim = Sim(r, nreq=shape.get("nreq", 3), seed=seed, pooled=shape.get("pooled", False))
    digests = []
    for _ in range(steps):
        if on_step is not None:
            on_step(sim)
        digests.append(sim.step(events.get(sim.n, ())))
    return digests, sim


def digest_all(digests) -> str:
    h = hashlib.sha256()
    for d in digests:
        for k in sorted(d):
            h.update(k.encode())
            h.update(d[k])
    return h.hexdigest()


def first_difference(a, b):
    """(step, name) of the first differing output, or None."""
    if len(a) != len(b):
        return ("len", len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if x.keys() != y.keys():
            return (i, "keys", sorted(set(x) ^ set(y)))
        for k in sorted(x):
            if x[k] != y[k]:
                return (i, k)
    return None
