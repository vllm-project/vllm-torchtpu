# SPDX-License-Identifier: Apache-2.0
"""Shared fakes for the q38 L4 SEEDFIX CPU proofs (test_seedfix.py, q38_seedfix_xtree.py, q38_seedfix_bench.py).

Tree-agnostic: it imports only modules every tree has, so q38_seedfix_xtree.py can drive the base tree (no
runner/seedfix.py) and the head tree through the same steps. The runner's REAL sampling methods (sample_tokens,
_sample_spec_verify_chunk / _fused / _vps, _vps_verify_fb, _padded_sampling_params, _vps_fallback_rows, ...) run on
a fake runner; compiled programs are replaced by their eager bodies and the TP all-gather by the identity (TP1).

A request has fixed logits per step (a function of its logit seed and the step), so a seeded request placed in
different batches sees identical logits and only its RNG decides its tokens.
"""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace as NS

import numpy as np
import torch

import vllm_torchtpu.runner.tpu_runner as tr
from vllm_torchtpu.runner import l4_vpshost  # v5 composition: VPSHOST T-1(i)'s (idle) async-copy state
from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rsm

K = 3
TV = 10240  # 64 VP_BLOCKs of 160: a TP1 VPS head (the fallback path needs >= VP_KC + 1 blocks)
CPU = torch.device("cpu")


class Stop(Exception):
    """Raised by the AsyncTPUCopyState stand-in: the step's sampling and its discard / rewind loop are done."""


CAP = threading.local()  # CAP.last: the step's per-chunk sampled tokens (trimmed to the real rows)


class _CopyStop:
    """tpu_runner.AsyncTPUCopyState stand-in: records combined_selected_tokens (the step's result) and stops the
    step right after the discard / rewind loop."""

    @staticmethod
    def from_device_chunks(chunks, lens, spec_logprobs=None):
        CAP.last = [c[:n].clone() for c, n in zip(chunks, lens)]
        raise Stop()


class Patcher:
    """monkeypatch-like setattr with undo (for scripts without pytest)."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)
        self._undo.clear()


REC: dict[str, list] = {"vps": [], "rs": [], "sfl": [], "fused": []}


def clear_rec():
    for v in REC.values():
        v.clear()


def patch_all(mp):
    """Compiled programs -> eager bodies (recording the uniforms each receives); identity all-gather;
    AsyncTPUCopyState -> _CopyStop."""
    import vllm.distributed

    def vps_fb(*a):
        # (local, shard_offset, bonus_idx, target_idx, temps, top_k, top_p, bonus_u, accept_u, recover_u, ...,
        #  n_active, F, cand, fb_u, [ro...])
        REC["vps"].append(dict(bonus_u=a[7].clone(), accept_u=a[8].clone(), recover_u=a[9].clone(),
                               fb_u=None if a[20] is None else a[20].clone(), F=a[18]))
        return tr._vps_verify_fb_impl(*a)

    def vps_plain(*a):
        REC["vps"].append(dict(bonus_u=a[7].clone(), accept_u=a[8].clone(), recover_u=a[9].clone(), fb_u=None, F=None))
        return tr._vps_verify_impl(*a)

    def fused(*a):
        REC["fused"].append(dict(bonus_u=a[6].clone(), accept_u=a[7].clone(), recover_u=a[8].clone()))
        return tr._verify_fused_impl(*a)

    def rrs(*a):
        REC["rs"].append(dict(accept_u=a[10].clone(), recover_u=a[11].clone()))
        return rsm._random_rejection_sample_impl(*a)

    mp.setattr(tr, "_vps_verify_fb_compiled", vps_fb)
    mp.setattr(tr, "_vps_verify_compiled", vps_plain)
    mp.setattr(tr, "_verify_fused_compiled", fused)
    mp.setattr(rsm, "_random_rejection_sample_with_segment", rrs)
    mp.setattr(vllm.distributed, "tensor_model_parallel_all_gather", lambda x, dim=-1: x)
    mp.setattr(tr, "AsyncTPUCopyState", _CopyStop)


def eager_sample(logits, temperatures, u, top_k, top_p, all_greedy=False, topk_candidates=0):
    REC["sfl"].append(dict(u=u.clone()))
    if all_greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)
    return tr._sample_from_logits_impl(logits, temperatures, u, top_k, top_p, topk_candidates)


class Req:
    """One request: its seed (None = unseeded), sampling params, draft count, logit seed; its generator persists."""

    def __init__(self, rid, seed, temp=0.8, top_k=20, top_p=0.95, logit_seed=0, d=K, partial=False, vocab=TV):
        self.rid, self.seed, self.temp, self.top_k, self.top_p = rid, seed, temp, top_k, top_p
        self.logit_seed, self.d, self.partial, self.vocab = logit_seed, d, partial, vocab
        self.gen = None
        if seed is not None:
            self.gen = torch.Generator()
            self.gen.manual_seed(int(seed))

    def rows(self, step):
        """(K + 1, vocab) bf16 logits (K target rows, then the bonus row) and K draft ids, fixed per (request, step)."""
        g = torch.Generator().manual_seed(self.logit_seed * 100003 + step * 7919 + 17)
        x = torch.randn(K + 1, self.vocab, generator=g) * 2.0
        idx = torch.randint(0, self.vocab, (K + 1, 8), generator=g)
        x.scatter_(1, idx, torch.tensor([5.0, 4.6, 4.2, 3.8, 3.2, 2.6, 2.0, 1.5]).expand(K + 1, 8))
        x = x.bfloat16()
        am = x[:K].float().argmax(-1)
        second = torch.sort(x[:K].float(), dim=-1, descending=True, stable=True).indices[:, 1]
        pick = torch.rand(K, generator=g)
        drafts = torch.where(pick < 0.5, am, torch.where(pick < 0.8, second, torch.randint(0, self.vocab, (K,), generator=g)))
        return x, drafts.to(torch.int32)


class FakeBatch:
    def __init__(self, reqs):
        self.req_ids = [r.rid for r in reqs]
        self.num_reqs = len(reqs)
        self.temperature_cpu_tensor = torch.tensor([r.temp for r in reqs], dtype=torch.float32)
        self.top_k_cpu_tensor = torch.tensor([r.top_k for r in reqs], dtype=torch.int32)
        self.top_p_cpu_tensor = torch.tensor([r.top_p for r in reqs], dtype=torch.float32)
        self.generators = {i: r.gen for i, r in enumerate(reqs) if r.gen is not None}
        self.max_num_logprobs = None
        self.all_greedy = all(r.temp <= 0 for r in reqs)


def compose_spec(reqs, step, b_pad=None, t_pad=None):
    """One spec-verify chunk holding `reqs` (in order): logits [t_pad, V], SpecDecodeMetadata-like md (the layout of
    speculative_decoding_manager.get_spec_decode_metadata, padding included)."""
    n = len(reqs)
    dl = [r.d for r in reqs]
    total = sum(d + 1 for d in dl)
    t_pad = t_pad or max(16, 1 << (total - 1).bit_length())
    b_pad = b_pad or max(n + 1, 8)
    assert t_pad >= total and b_pad > n, (t_pad, total, b_pad, n)
    vocab = reqs[0].vocab
    logits = torch.zeros(t_pad, vocab, dtype=torch.bfloat16)
    bonus_idx, target_idx, drafts, seg, grp = [], [], [], [], []
    cu = 0
    for i, r in enumerate(reqs):
        x, dr = r.rows(step)
        for j in range(r.d):
            logits[cu + j] = x[j]
            target_idx.append(cu + j)
            drafts.append(int(dr[j]))
            seg.append(i)
            grp.append(j)
        logits[cu + r.d] = x[K]
        bonus_idx.append(cu + r.d)
        cu += r.d + 1

    def pad(vals, length, fill, dtype):
        out = torch.full((length,), fill, dtype=dtype)
        if vals:
            out[: len(vals)] = torch.tensor(vals, dtype=dtype)
        return out

    md = NS(
        draft_token_ids=pad(drafts, t_pad, 0, torch.int32),
        draft_lengths=pad(dl, b_pad, 0, torch.int32),
        draft_lengths_cpu=np.array(dl, dtype=np.int32),
        target_logits_indices=pad(target_idx, t_pad, 0, torch.int32),
        bonus_logits_indices=pad(bonus_idx, b_pad, 0, torch.int32),
        final_logits_indices=torch.zeros(t_pad, dtype=torch.int32),
        segment_ids=pad(seg, t_pad, n, torch.int64),
        group_indices=pad(grp, t_pad, 0, torch.int32),
        deferred_draft_token_ids_cpu=None,
    )
    return logits, md


def compose_plain(reqs, step, b_pad=None):
    """One non-spec chunk: logits [b_pad, V] (each request's bonus row)."""
    n = len(reqs)
    b_pad = b_pad or max(n + 1, 8)
    logits = torch.zeros(b_pad, reqs[0].vocab, dtype=torch.bfloat16)
    for i, r in enumerate(reqs):
        logits[i] = r.rows(step)[0][K]
    return logits


def sched_for(reqs, spec=True, plain_rids=()):
    reqs_state, sched = {}, {}
    for r in reqs:
        n = (r.d + 1) if spec and r.rid not in plain_rids else 1
        total = 1000 if r.partial else 100
        reqs_state[r.rid] = NS(num_computed_tokens=100 - n, num_tokens=total)
        sched[r.rid] = n
    return reqs_state, NS(num_scheduled_tokens=sched)


class _Stage:
    @contextlib.contextmanager
    def stage(self, name):
        yield


class Runner:
    """The runner's real sampling methods on a fake runner (TP1 VPS head of width TV when vps=True)."""

    sample_tokens = tr.TPUModelRunner.sample_tokens
    _sample_spec_verify_chunk = tr.TPUModelRunner._sample_spec_verify_chunk
    _sample_spec_verify_fused = tr.TPUModelRunner._sample_spec_verify_fused
    _sample_spec_verify_vps = tr.TPUModelRunner._sample_spec_verify_vps
    _vps_verify_fb = tr.TPUModelRunner._vps_verify_fb
    _vps_fallback_rows = tr.TPUModelRunner._vps_fallback_rows
    _padded_sampling_params = tr.TPUModelRunner._padded_sampling_params
    _build_padded_sampling_params = tr.TPUModelRunner._build_padded_sampling_params
    _sampler_topk_candidates = tr.TPUModelRunner._sampler_topk_candidates
    _topk_candidate_blocks = tr.TPUModelRunner._topk_candidate_blocks
    _upload_unless_unchanged = tr.TPUModelRunner._upload_unless_unchanged
    _upload_keyed = tr.TPUModelRunner._upload_keyed
    _greedy_sample = tr.TPUModelRunner._greedy_sample
    _get_sampling_generator = tr.TPUModelRunner._get_sampling_generator
    _spec_verify_bucket_reachable = tr.TPUModelRunner._spec_verify_bucket_reachable
    # v5 composition (V5COMP-BUILD; tests only): VPSHOST T-1(i)'s VPS_FALLBACK helper (the stock _vps_verify_fb uses it)
    _log_vps_fallback = tr.TPUModelRunner._log_vps_fallback

    def __init__(self, vps=True, cand=False, fb_max=8, tiegap=0, model_seed=0, vocab=TV, k=K):
        self.device = CPU
        self.vocab_size = vocab
        self.model_config = NS(seed=model_seed)
        self._sampling_generator = None
        self.speculative_config = NS(num_speculative_tokens=k) if k else None
        self.rejection_sampler = rsm.RejectionSampler()
        self._vps_head = NS(full_vocab=vocab, tp_size=1, shard_offset=0, shard_width=vocab) if vps else None
        self._vps_fb_path = vps and (fb_max > 0 or tiegap > 0 or cand)
        self._vps_fb_max, self._vps_tiegap, self._vps_cand_draws = fb_max, tiegap, cand
        self._vps_fb_stats, self._vps_fb_chunks, self._vps_checks_left = None, 0, 0
        self._vps_fb_async = l4_vpshost.AsyncCounts()  # v5 composition: statsasync off, nothing in flight
        self._verify_fused_checks_left = 0
        self._fused_read_offsets = (None, 0)
        self._fused_read_offsets_applied = False
        self.mamba_slot_read_offsets = None
        self._h2d_reuse = {}
        self._stage_timing = _Stage()
        self.scheduler_config = NS(async_scheduling=False)
        self.is_pooling_model = False
        self._pp_is_last = True
        self._pp_wave = None
        self._is_async_drafter = False
        self.structured_decoding_manager = None
        self.execute_model_state = None
        self.sample_from_logits_func = eager_sample
        self.input_batch = None
        self.requests = {}
        self.outs = []
        # composed trees (q38-l4-samp2-vfy): no VFY-ONE, no VRNG controller
        self._vfy = None
        self._vrng = None
        self.warmup_done = True

    def spec_gather_bonus_and_target_logits(self, logits, bonus_idx, target_idx):
        return torch.index_select(logits, 0, bonus_idx), torch.index_select(logits, 0, target_idx)

    def _update_mamba_slot_read_offsets(self, *a, **k):
        pass

    def _update_num_xla_graphs(self, *a, **k):
        pass

    def _check_vps(self, *a, **k):
        pass

    def set_batch(self, reqs, spec=True, plain_rids=()):
        self.input_batch = FakeBatch(reqs)
        self.requests, sched = sched_for(reqs, spec, plain_rids)
        return sched


def run_step(runner, chunks, route="vps", spec=True, step=0, pads=None, plain_chunks=()):
    """One sample_tokens step over `chunks` (a list of request lists, in batch order) on route "vps" (VPS where the
    chunk is eligible, else the stock route, as execute_model decides), "stock" or "plain" (non-spec step).
    `plain_chunks`: indices of non-verify chunks inside a spec step (their requests finish prefill this step).
    Returns (per-chunk [num_reqs, K+1] or [num_reqs, 1] token tensors, per-chunk route names)."""
    reqs = [r for c in chunks for r in c]
    plain_rids = {r.rid for ci in plain_chunks for r in chunks[ci]}
    sched = runner.set_batch(reqs, spec, plain_rids)
    logits_l, nreq_l, md_l, vps_l, fb_l, routes = [], [], [], [], [], []
    start = 0
    for ci, c in enumerate(chunks):
        bp, tp = (pads[ci] if pads else (None, None))
        if spec and ci in plain_chunks:
            logits_l.append(compose_plain(c, step, b_pad=bp))
            md_l.append(None)
            vps_l.append(False)
            fb_l.append(0)
            routes.append("spec_plain")
            nreq_l.append(len(c))
            start += len(c)
            continue
        if spec:
            logits, md = compose_spec(c, step, b_pad=bp, t_pad=tp)
            use_vps, F = False, 0
            if route == "vps" and runner._vps_head is not None:
                if runner._vps_fb_path:
                    f = runner._vps_fallback_rows(start, start + len(c))
                    use_vps, F = f is not None, f or 0
                else:
                    use_vps = all(1 <= r.top_k <= 32 for r in c)
            md_l.append(md)
            vps_l.append(use_vps)
            fb_l.append(F)
            routes.append("vps" if use_vps else "stock")
        else:
            logits = compose_plain(c, step, b_pad=bp)
            md_l.append(None)
            routes.append("plain")
        logits_l.append(logits)
        nreq_l.append(len(c))
        start += len(c)
    runner.execute_model_state = NS(
        scheduler_output=sched, draft_chunks=None, top_tokens=False,
        vps_chunks=vps_l if spec else None, vps_fallback=fb_l if spec else None,
        spec_decode_metadata_list=md_l, logits_list=logits_l, num_reqs_list=nreq_l, mamba_state_indices_list=None,
    )
    CAP.last = None
    try:
        runner.sample_tokens(None)
    except Stop:
        pass
    assert CAP.last is not None, "the step did not reach the sampled-token copy"
    return CAP.last, routes
