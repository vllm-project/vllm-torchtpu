# SPDX-License-Identifier: Apache-2.0
"""q38 L1-DSTASH CPU harness (tests only): the real vLLM AsyncScheduler (with or without the real DECODE_TRIM of
core/decode_trim.py installed) drives the TPU runner's real async speculative-decoding bookkeeping on CPU, with a
deterministic fake model in place of the forward.

Engine order is vLLM's batch-queue order: schedule(t+1), runner step t+1, update(t), schedule(t+2), ... so a request
held back at t+1 re-enters at t+2 with exact positions and all_token_ids, exactly as in serving.

Real runner code used per step (TPUModelRunner on an object.__new__ instance carrying only the attributes these
methods read): _update_states (incl. vLLM's GPUModelRunner._update_states and the stash capture),
_flush_disjoint_async_results, _dstash_apply (called exactly where _prepare_inputs calls it: right after the input ids
are uploaded), _prepare_async_token_substitution_indices, _apply_async_token_substitution, _build_mamba_state_indices,
_reset_read_offsets_for_new_requests, _update_mamba_slot_read_offsets, _modify_prev_results (incl. the stash counter),
_assemble_async_spec_substitution / _assemble_async_prefill_bootstrap, _update_placeholder, and the real
subtract_num_rejected_tokens. Copied (not callable on its own): the host token / position gather of _prepare_inputs and
its inline rejection-correction index loop (tpu_runner.py, the block after "Async spec: seq_lens/positions were advanced
optimistically"), and sample_tokens' request_seq_lens loop and the stride-(1+K) source pad. The compiled
_substitute_placeholder_token runs as its eager body (the "tpu" backend cannot trace CPU tensors) behind a recorder.

Fake model: a hash-chain "LM" whose state is checkpointed per input slot in the request's GDN slot group, read back at
the runner's mamba read offset, so outputs depend only on the request's own consumed (token, position) stream and are
independent of the drafts (as in real speculative decoding); wrong positions or a wrong read offset change them. The
drafter proposes the true continuation with a deterministic subset corrupted, so acceptance varies.

Scheduler setup: L1-C48's tests/core/test_decode_trim.py `_real` / `_req` (copied, credit L1-C48), with a local OPT
config directory tried first (no network), then facebook/opt-125m.
Works on trees without the stash (BASE): the stash hooks are only called where the real code would call them.
"""

from __future__ import annotations

import contextlib
import inspect
import os
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

K = 3
STRIDE = 1 + K
MAX_REQS = 8
BLOCK = 16
MAX_MODEL_LEN = 512
PROMPT = 16
VOCAB = 32000
TOKEN_PADDINGS = [16, 32, 64, 128]
HERE = os.path.dirname(os.path.abspath(__file__))
_P = 2**31 - 1

# DECODE_TRIM tables (rows per request = 1 + K = 4): which scenario holds how many requests back per step.
TABLE_NEVER = "1024:1"  # installed, never trims (0 holds)
TABLE_ONE = "8:1/16:3/al:1/reentry:0"  # 3 requests: 2 served (8 rows), 1 held per step (C48 R4's table)
TABLE_MANY = "8:1/16:3/24:5/al:1/reentry:0"  # 6 requests: 2 served / 4 held, then the 4 forced back / 2 held, ...
SCENARIOS = {
    "S0": dict(table=None, n=3, max_tokens=[30, 24, 27]),
    "S0i": dict(table=TABLE_NEVER, n=3, max_tokens=[30, 24, 27]),
    "S1": dict(table=TABLE_ONE, n=3, max_tokens=[30, 24, 27]),
    "S2": dict(table=TABLE_MANY, n=6, max_tokens=[30, 24, 36, 28, 33, 26]),
}


# ------------------------------------------------------------------------------------------------ fake model
def mix(s: int, x: int, p: int) -> int:
    return (s * 1000003 + (int(x) + 7) * 31 + int(p) * 7 + 12345) % _P


def out(s: int) -> int:
    return 50 + s % 30000


def corrupt(rid: str, q: int, i: int) -> bool:
    return (sum(map(ord, rid)) * 31 + q * 17 + i * 5) % 3 == 0


class FakeModel:
    """Per GDN slot group, one state checkpoint per verify window position."""

    def __init__(self):
        self.ckpt: dict[int, list[int]] = {}
        self.started: set[str] = set()

    def prefill(self, slot, xs, ps):
        s = 0
        for x, p in zip(xs, ps):
            s = mix(s, x, p)
        self.ckpt[slot] = [s] + [0] * K
        return [out(s)], s, int(ps[-1]) + 1

    def verify(self, slot, off, xs, ps):
        s = self.ckpt[slot][off]
        ys, states = [], []
        for x, p in zip(xs, ps):
            s = mix(s, x, p)
            states.append(s)
            ys.append(out(s))
        a = 0
        while a < len(xs) - 1 and int(xs[a + 1]) == ys[a]:
            a += 1
        self.ckpt[slot] = states + [0] * (STRIDE - len(states))
        return ys[: a + 1], states[a], int(ps[a]) + 1

    @staticmethod
    def draft(rid, state, y_last, q):
        s = mix(state, y_last, q)
        ds = []
        for i in range(K):
            c = out(s)
            ds.append(c + 1 if corrupt(rid, q, i) else c)
            s = mix(s, c, q + 1 + i)
        return ds


# ------------------------------------------------------------------------------------------------ scheduler (C48)
_MODEL_CONFIG = None


def model_config():
    """ModelConfig for the scheduler: the local OPT config directory first (no network), then facebook/opt-125m."""
    global _MODEL_CONFIG
    if _MODEL_CONFIG is None:
        from vllm.config import ModelConfig

        errs = []
        for m in (os.path.join(HERE, "tiny_opt"), "facebook/opt-125m"):
            try:
                _MODEL_CONFIG = ModelConfig(model=m, tokenizer=m, dtype="float16", seed=42, max_model_len=2048,
                                            skip_tokenizer_init=True)
                print(f"Q38DS model_config from {m}", flush=True)
                break
            except Exception as e:  # report every attempt; fail only if none works
                errs.append(f"{m}: {type(e).__name__}: {e}")
        if _MODEL_CONFIG is None:
            raise RuntimeError("no ModelConfig: " + " | ".join(errs))
    return _MODEL_CONFIG


def make_scheduler(spec_k=K, num_blocks=256, max_num_seqs=MAX_REQS):
    from vllm.config import CacheConfig, DeviceConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=2048, max_model_len=2048,
            enable_chunked_prefill=True, is_encoder_decoder=False, watermark=0.0, async_scheduling=True,
        ),
        model_config=model_config(),
        cache_config=CacheConfig(block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto",
                                 enable_prefix_caching=False),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=64, dtype=torch.float16))],
    )
    cfg.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(cfg)
    s = AsyncScheduler(vllm_config=cfg, kv_cache_config=kv,
                       structured_output_manager=StructuredOutputManager(cfg), block_size=BLOCK)
    s.num_spec_tokens = spec_k
    s._spec_token_placeholders = [-1] * spec_k
    init_none_hash(sha256)
    return s, get_request_block_hasher(BLOCK, sha256)


def make_request(rid, hasher, seed, max_tokens):
    from vllm import SamplingParams
    from vllm.v1.request import Request

    p = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    p.update_from_generation_config({}, 2)
    return Request(rid, [seed * 1000 + i for i in range(PROMPT)], p, None, block_hasher=hasher)


# ------------------------------------------------------------------------------------------------ runner
def make_runner():
    from vllm.v1.worker.gpu_input_batch import InputBatch

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    rn = object.__new__(TPUModelRunner)
    rn.device = torch.device("cpu")
    rn.speculative_config = SimpleNamespace(num_speculative_tokens=K, use_ngram_gpu=lambda: False,
                                            use_eagle=lambda: True, method="mtp", draft_tensor_parallel_size=1)
    rn.scheduler_config = SimpleNamespace(async_scheduling=True)
    rn.parallel_config = SimpleNamespace(data_parallel_size=1, enable_expert_parallel=False)
    rn.use_async_scheduling = True
    rn.use_async_spec_decode = True
    rn.prev_num_draft_tokens = SimpleNamespace(np=np.zeros(MAX_REQS, dtype=np.int32))
    rn.requests = {}
    rn.num_prompt_logprobs = {}
    rn._pre_async_results = None
    rn._mamba_state_pos = {}
    rn._mamba_offset_seeded = set()
    rn._read_offset_scratch = {}
    rn._last_sequence_layout_plan = None
    rn._fast_token_substitution = False
    rn._pcp_enabled = False
    rn._pp_is_first = True
    rn._pp_is_last = True
    rn.max_num_reqs = MAX_REQS
    rn.num_reqs_paddings = [8]
    rn.num_tokens_paddings = list(TOKEN_PADDINGS)
    rn.input_batch = InputBatch(
        max_num_reqs=MAX_REQS, max_model_len=MAX_MODEL_LEN, max_num_batched_tokens=2048, device=torch.device("cpu"),
        vocab_size=VOCAB, block_sizes=[BLOCK], kernel_block_sizes=[BLOCK],
        max_num_blocks_per_req=[MAX_MODEL_LEN // BLOCK], num_spec_tokens=K,
    )
    rn.late_interaction_runner = SimpleNamespace(on_requests_finished=lambda _ids: None,
                                                 register_request=lambda _rid, _params: None)
    rn.encoder_cache = {}
    rn.is_pooling_model = False
    rn.uses_mrope = False
    rn.uses_xdrope_dim = 0
    rn._may_reorder_batch = lambda _output: None
    rn._zero_block_ids = lambda _ids: None
    rn._update_num_xla_graphs = lambda _name: None
    # Compact GDN slot pool, one group of 1 + K slots per request.
    rn._mamba_slot_stride = STRIDE
    rn._mamba_ckpt_window = STRIDE
    rn._mamba_copy_plan = []
    nblk = 1 + 2 * MAX_REQS * STRIDE
    rn.mamba_slot_read_offsets = torch.zeros(nblk, dtype=torch.int32)
    rn.mamba_state_indices_cpu = torch.zeros(MAX_REQS, dtype=torch.int32)
    rn._init_mamba_slot_pool(nblk)
    return rn


def sig(*ts):
    return tuple((tuple(t.shape), str(t.dtype)) for t in ts)


class Recorder:
    """Eager stand-in for the compiled _substitute_placeholder_token; records every call's caller and signature."""

    def __init__(self, eager):
        self.eager = eager
        self.calls: list[tuple[str, tuple]] = []

    def __call__(self, input_ids, cur, pre, next_tokens):
        caller = inspect.currentframe().f_back.f_code.co_name
        self.calls.append((caller, sig(input_ids, cur, pre, next_tokens)))
        return self.eager(input_ids, cur, pre, next_tokens)


@contextlib.contextmanager
def patched(env: dict[str, str | None]):
    """Patch the compiled substitution program (eager body + recorder), the pipeline group and synchronize_tensors;
    set / clear env keys; restore everything on exit."""
    from vllm.v1.worker import gpu_model_runner

    from vllm_torchtpu.runner import tpu_runner as r

    comp = r._substitute_placeholder_token
    eager = getattr(comp, "_torchdynamo_orig_callable", None) or getattr(comp, "__wrapped__", None)
    assert eager is not None and eager is not comp, "no eager body for _substitute_placeholder_token"
    rec = Recorder(eager)
    saved = {k: os.environ.get(k) for k in env}
    old = (r._substitute_placeholder_token, gpu_model_runner.get_pp_group, r.synchronize_tensors)
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        r._substitute_placeholder_token = rec
        gpu_model_runner.get_pp_group = lambda: SimpleNamespace(is_last_rank=True)
        r.synchronize_tensors = lambda *a, **k: None
        yield rec
    finally:
        r._substitute_placeholder_token, gpu_model_runner.get_pp_group, r.synchronize_tensors = old
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def env_for(stash: bool, table: str | None, warmed: bool = True, log_every: int = 0) -> dict[str, str | None]:
    return {
        "VLLM_TPU_DECODE_TRIM_DRAFT_STASH": "1" if stash else None,
        "VLLM_TPU_DECODE_TRIM": table,
        "VLLM_TPU_WARMED_SUBSTITUTE_SOURCE": "1" if warmed else None,
        "VLLM_TPU_DECODE_TRIM_LOG": str(log_every),
        "VLLM_TPU_POSTFWD_FUSED": None,
        "VLLM_TPU_SPEC_TAIL_COMPILED": None,
        "VLLM_TPU_SPEC_TAIL_REJ_PAD": None,
        "VLLM_TPU_PINNED_TOKEN_COPY": None,
    }


# ------------------------------------------------------------------------------------------------ one runner step
class Step:
    """The runner side of one engine step (execute_model + sample_tokens), real methods where callable."""

    def __init__(self, rn, fm, rec):
        self.rn, self.fm, self.rec = rn, fm, rec
        self.records: dict[str, list] = {}
        self.produced: dict[str, int] = {}

    def __call__(self, so, step_no, log):
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_torchtpu.runner.tpu_runner_async_output import (
            AsyncPreResults,
            AsyncTPUCopyState,
            subtract_num_rejected_tokens,
        )

        rn, fm = self.rn, self.fm
        # requests with a row in the previous results that this step does not schedule and that stay live: the runner's view
        # of "held back" (DECODE_TRIM's holds, plus vLLM's own skip of a request whose last step already reached max_tokens)
        inval = set(so.finished_req_ids) | set(so.preempted_req_ids or ()) | set(so.scheduled_cached_reqs.resumed_req_ids)
        pre0 = rn._pre_async_results
        self.last_unsched = sorted(r for r in (pre0.req_id_to_index_copy if pre0 is not None else {})
                                   if r not in inval and r not in so.num_scheduled_tokens and r in rn.requests)
        rn._update_states(so)
        rn._flush_disjoint_async_results()
        n = rn.input_batch.num_reqs
        ids = [rn.input_batch.req_ids[i] for i in range(n)]
        if so.total_num_scheduled_tokens == 0 or n == 0:
            return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
        nsched = np.array([so.num_scheduled_tokens[r] for r in ids], dtype=np.int32)
        spec = {r: len(so.scheduled_spec_decode_tokens.get(r, ())) for r in ids}
        prefill = [r for r in ids if r not in fm.started]
        if prefill and len(prefill) != n:
            raise AssertionError(f"mixed prefill/decode step {step_no}: harness covers pure steps only")
        total = int(nsched.sum())
        padded = next(b for b in TOKEN_PADDINGS if b >= total)
        # _prepare_inputs: host gather of token ids and positions (copied), then the upload.
        tok = np.zeros(padded, dtype=np.int32)
        pos = np.zeros(padded, dtype=np.int32)
        seq = np.zeros(MAX_REQS, dtype=np.int32)
        acc = 0
        for i, rid in enumerate(ids):
            nc = int(rn.input_batch.num_computed_tokens_cpu[i])
            m = int(nsched[i])
            tok[acc : acc + m] = rn.input_batch.token_ids_cpu[i, nc : nc + m]
            pos[acc : acc + m] = nc + np.arange(m)
            seq[i] = nc + m
            acc += m
        rn.positions_cpu = torch.from_numpy(pos.copy())
        rn.input_ids = torch.from_numpy(tok.copy())
        host_tok = tok.copy()
        if "_dstash" in rn.__dict__:  # the call _prepare_inputs makes right after the upload
            rn._dstash_apply(so, 0, n, nsched)
        positions = rn.positions_cpu.clone()
        seq_lens = torch.from_numpy(seq)
        pre = rn._pre_async_results
        pre_map = dict(pre.req_id_to_index_copy) if pre is not None else {}
        if pre is not None and pre.spec_decode_num_rejected_tokens is not None and pre.req_id_to_index_copy:
            # inline rejection-correction index loop of _prepare_inputs (copied)
            seq_idx = np.full(MAX_REQS, -1, dtype=np.int32)
            pos_idx = np.full(padded, -1, dtype=np.int32)
            acc = 0
            for i, rid in enumerate(ids):
                m = int(nsched[i])
                p = pre.req_id_to_index_copy.get(rid)
                if p is not None:
                    seq_idx[i] = p
                    pos_idx[acc : acc + m] = p
                acc += m
            seq_lens, positions = subtract_num_rejected_tokens(
                seq_lens, positions, pre.spec_decode_num_rejected_tokens,
                torch.from_numpy(seq_idx), torch.from_numpy(pos_idx))
        cur, src = rn._prepare_async_token_substitution_indices(0, n, nsched)
        input_ids = rn._apply_async_token_substitution(rn.input_ids, cur, src)
        indices = rn._build_mamba_state_indices(0, n, MAX_REQS)
        rn._attn_metadata_builder_ctx = SimpleNamespace(mamba_state_indices=indices, unified_mamba_state_indices=None)
        rn._reset_read_offsets_for_new_requests(0, n)
        # fake forward + verify + draft
        is_spec = any(spec.values())
        width = STRIDE if is_spec else 1
        sampled = torch.full((n, width), -1, dtype=torch.int32)
        drafts = torch.zeros((n, K), dtype=torch.int32)
        valid = []
        acc = 0
        x_all = input_ids.tolist()
        p_all = positions.tolist()
        for i, rid in enumerate(ids):
            m = int(nsched[i])
            xs, ps = x_all[acc : acc + m], p_all[acc : acc + m]
            slot = int(indices[i])
            off = int(rn.mamba_slot_read_offsets[slot])
            if rid in prefill:
                ys, st, q = fm.prefill(slot, xs, ps)
                fm.started.add(rid)
            else:
                ys, st, q = fm.verify(slot, off, xs, ps)
            ds = fm.draft(rid, st, ys[-1], q)
            sampled[i, : len(ys)] = torch.tensor(ys, dtype=torch.int32)
            drafts[i] = torch.tensor(ds, dtype=torch.int32)
            before = self.produced.get(rid, 0)
            self.records.setdefault(rid, []).append(
                dict(before=before, x=list(map(int, xs)), host=list(map(int, host_tok[acc : acc + m])),
                     pos=list(map(int, ps)), off=off if rid not in prefill else -1, out=list(ys), drafts=ds,
                     in_pre=rid in pre_map, step=step_no))
            self.produced[rid] = before + len(ys)
            valid.append(list(ys))
            acc += m
        rn._update_mamba_slot_read_offsets([indices], sampled, n)
        # sample_tokens bookkeeping (request_seq_lens loop copied), real rollback / assembly / placeholder update
        req_ids = list(ids)
        request_seq_lens, discard = [], []
        for i, rid in enumerate(req_ids):
            st = rn.requests[rid]
            seq_len = st.num_computed_tokens + so.num_scheduled_tokens[rid]
            if seq_len < st.num_tokens:
                discard.append(i)
            else:
                request_seq_lens.append((i, st, seq_len, rid))
        copy_state = AsyncTPUCopyState(chunks_cpu=[sampled.clone()], chunk_real_lens=[n],
                                       copy_ready_event=SimpleNamespace(synchronize=lambda: None),
                                       chunks_tpu=[sampled])
        rn._modify_prev_results()
        if is_spec:
            dl = np.array([spec[r] for r in ids], dtype=np.int32)
            state = SimpleNamespace(num_reqs_list=[n], spec_decode_metadata_list=[
                SimpleNamespace(draft_lengths=torch.from_numpy(dl), draft_lengths_cpu=dl)])
            chunks, nti, rej, counts = rn._assemble_async_spec_substitution(drafts, [sampled], state)
        else:
            chunks, nti, rej, counts = rn._assemble_async_prefill_bootstrap(drafts, [sampled], [n])
        mapping = rn._update_placeholder(discard, request_seq_lens, nti, counts)
        nt = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        if rej is not None:  # sample_tokens' stride-(1+K) pad to the bound (copied)
            bound = MAX_REQS * STRIDE
            if nt.shape[0] < bound:
                nt = torch.nn.functional.pad(nt, (0, bound - nt.shape[0]))
        rn._pre_async_results = AsyncPreResults(
            req_ids=req_ids, next_tokens_tpu=nt, request_seq_lens=request_seq_lens,
            discard_sampled_tokens_req_indices=discard, req_id_to_index_copy=mapping, copy_state=copy_state,
            spec_decode_num_rejected_tokens=rej, num_draft_per_req=counts)
        for i in discard:
            valid[i] = []
        return ModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                 sampled_token_ids=valid)


# ------------------------------------------------------------------------------------------------ the engine loop
class _Grab(__import__("logging").Handler):
    def __init__(self):
        super().__init__(level=0)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


LOGGERS = ("vllm.vllm_torchtpu.runner.draft_stash", "vllm.vllm_torchtpu.runner.tpu_runner")


def run(name: str, *, stash: bool, warmed: bool = True, log_every: int = 0, table: str | None = "scenario",
        trim_env: bool = True, max_steps: int = 600):
    """Run scenario `name`; return a dict of records, final outputs, per-step log and the stash / call evidence.
    table: "scenario" = the scenario's DECODE_TRIM table, None = no trim (the reference). trim_env: also export the
    table as VLLM_TPU_DECODE_TRIM (the runner-side gate); False = the trim runs on the scheduler but the runner's env
    does not say so."""
    import logging

    from vllm_torchtpu.core import decode_trim as dt

    sc = SCENARIOS[name]
    tbl = sc["table"] if table == "scenario" else table
    env = env_for(stash, tbl if trim_env else None, warmed, log_every)
    grab = _Grab()
    lgs = [logging.getLogger(n) for n in LOGGERS]
    levels = [lg.level for lg in lgs]
    for lg in lgs:
        lg.addHandler(grab)
        lg.setLevel(logging.INFO)
    with contextlib.ExitStack() as stack:
        stack.callback(lambda: [(lg.removeHandler(grab), lg.setLevel(lv)) for lg, lv in zip(lgs, levels)])
        rec = stack.enter_context(patched(env))
        s, h = make_scheduler()
        trim = dt.install(s, dt.TrimTable.parse(tbl), log_every) if tbl else None
        reqs = [make_request(f"r{i}", h, i + 1, sc["max_tokens"][i]) for i in range(sc["n"])]
        for q in reqs:
            s.add_request(q)
        rn = make_runner()
        fm = FakeModel()
        step = Step(rn, fm, rec)
        pending: deque = deque()
        log = []
        for step_no in range(max_steps):
            if not s.has_requests() and not pending:
                break
            if s.has_requests():
                so = s.schedule()
                held = sorted(trim.skipped_prev) if trim is not None else []
                ncalls = len(rec.calls)
                mro = step(so, step_no, log)
                st = rn.__dict__.get("_dstash")
                entry = dict(step=step_no, sched=sorted(so.num_scheduled_tokens), held=held, unsched=step.last_unsched,
                             calls=[c for c in rec.calls[ncalls:]],
                             applied=sorted(st.last_applied) if st is not None else None,
                             entries=sorted(st.entries) if st is not None else None,
                             dstash_attrs=sorted(k for k in rn.__dict__ if k.startswith("_dstash")))
                log.append(entry)
                pending.append((so, mro))
            if len(pending) >= 2 or (pending and not s.has_requests()):
                so0, mro0 = pending.popleft()
                s.update_from_output(so0, mro0)
        else:
            raise AssertionError(f"{name}: did not finish in {max_steps} steps")
        finals = {q.request_id: list(q.output_token_ids) for q in reqs}
        st = rn.__dict__.get("_dstash")
        return dict(
            name=name, stash=stash, warmed=warmed, table=tbl, trim_env=trim_env, records=step.records, finals=finals,
            log=log, calls=list(rec.calls), totals=(dict(st.totals) if st is not None else None),
            trim_totals=(dict(trim.totals) if trim is not None else None),
            dstash_attrs=sorted(k for k in rn.__dict__ if k.startswith("_dstash")),
            refused=rn.__dict__.get("_dstash_refused"), lines=[x for x in grab.lines if "DSTASH" in x],
            pending_accept=(sorted(st.pending_accept) if st is not None else None),
            live_entries=(sorted(st.entries) if st is not None else None), runner=rn,
        )


def trimmed_records(res, max_tokens):
    """Per request, the served-step records produced before the request had all its tokens (a step scheduled before
    the scheduler saw the request finish carries no committed output)."""
    mt = dict(zip([f"r{i}" for i in range(len(max_tokens))], max_tokens))
    return {rid: [{k: v for k, v in e.items() if k not in ("step", "in_pre", "host")} for e in recs if e["before"] < mt[rid]]
            for rid, recs in res["records"].items()}


def reentries(res):
    """(step, request) pairs where a request ran after being held back in the previous step."""
    out_ = []
    prev_held: set = set()
    for e in res["log"]:
        for rid in e["sched"]:
            if rid in prev_held:
                out_.append((e["step"], rid))
        prev_held = set(e["held"])
    return out_


def warmed_signatures():
    """Signatures the real _precompile_substitute_placeholder_token warms on the harness runner."""
    with patched(env_for(False, None, True)) as rec:
        rn = make_runner()
        rn._precompile_substitute_placeholder_token()
        return {c[1] for c in rec.calls}


def dump(res) -> dict:
    """JSON-able view (no runner object)."""
    return {k: v for k, v in res.items() if k != "runner"}
