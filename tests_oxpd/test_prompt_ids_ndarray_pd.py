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
"""PD port of MM's P15 (VLLM_TPU_PROMPT_IDS_NDARRAY, default off; MM q38mm-dm-pta 32cf47e55..8a577bc14): PD's schedulers
substitute through vLLM's base Scheduler.schedule; the Stage-3 producer's tail truncation reuses the arrival array's prefix
(both orders); vLLM 0.31's NewRequestData survives the pickle round trip identically; vLLM 0.31's worker (CachedRequestState
+ InputBatch, built as GPUModelRunner._update_states builds them) holds byte-identical rows for the arrival array; the flag
is not a compile factor. Assertion messages start with the names the CDK CPU proof's mutants are matched against
(PTA-ENGAGE, PTA-KEYCHECK, PTA-SLICE, PTA-IDENT, PTA-WORKER, PTA-ADMP, PTA-KEY). PTA-ADMP: a decode engine's ADM-P
PromptTail (VLLM_TPU_PD_PROMPT_TAIL) ships untouched."""

import os
import pickle
import random
from types import SimpleNamespace

import numpy as np

from vllm_torchtpu.core import prompt_ids_ndarray as P


def _req(rid, n, seed=1):
    rnd = random.Random(seed)
    return SimpleNamespace(request_id=rid, prompt_token_ids=[rnd.randrange(0, 151_000) for _ in range(n)])


def _out(*nrds):
    return SimpleNamespace(scheduled_new_reqs=list(nrds))


def test_pd_schedulers_substitute_through_the_base_schedule(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    req = _req("r0", 300)
    nrd = SimpleNamespace(req_id="r0", prompt_token_ids=req.prompt_token_ids)
    monkeypatch.setattr(Scheduler, "schedule", lambda self, *a, **k: _out(nrd), raising=False)
    monkeypatch.setattr(Scheduler, P._SCHED_FLAG, False, raising=False)
    monkeypatch.setattr(P, "install_engine_hook", lambda: False)
    monkeypatch.setenv(P.ENV, "1")
    P.patch_pd_schedulers()
    out = Scheduler.schedule(SimpleNamespace(requests={"r0": req}))
    got = out.scheduled_new_reqs[0].prompt_token_ids
    assert isinstance(got, np.ndarray) and got.dtype == np.int32, f"PTA-ENGAGE: prompt must ship as int32, got {type(got)}"
    assert got.tolist() == req.prompt_token_ids, "PTA-ENGAGE: values"
    assert isinstance(req.prompt_token_ids, list), "PTA-ENGAGE: the Request keeps its list"


def test_flag_off_leaves_the_scheduler_alone(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    sentinel = lambda self, *a, **k: "orig"  # noqa: E731
    monkeypatch.setattr(Scheduler, "schedule", sentinel, raising=False)
    monkeypatch.setattr(Scheduler, P._SCHED_FLAG, False, raising=False)
    monkeypatch.delenv(P.ENV, raising=False)
    P.patch_pd_schedulers()
    assert Scheduler.schedule is sentinel, "PTA-KEYCHECK: flag off must not wrap Scheduler.schedule"


def test_producer_tail_truncation_reuses_the_arrival_array_in_both_orders():
    """tpu_connector's Stage-3 P side pops the prompt's tail after arrival. Attached before the pop: the arrival array's
    prefix ships (no rebuild). Attached after: it matches directly."""
    for order in ("attach_then_pop", "pop_then_attach"):
        req = _req("r1", 1000, seed=3)
        if order == "attach_then_pop":
            P.attach(req)
        while len(req.prompt_token_ids) > 640:
            req.prompt_token_ids.pop()
        if order == "pop_then_attach":
            P.attach(req)
        rebuilt, sliced = P.stats["rebuilt"], P.stats.get("sliced", 0)
        nrd = SimpleNamespace(req_id="r1", prompt_token_ids=req.prompt_token_ids)
        P.substitute(SimpleNamespace(requests={"r1": req}), _out(nrd))
        got = nrd.prompt_token_ids
        assert isinstance(got, np.ndarray) and got.tolist() == req.prompt_token_ids, f"PTA-SLICE: {order} values"
        assert P.stats["rebuilt"] == rebuilt, f"PTA-SLICE: {order} must not rebuild on the busy loop"
        assert P.stats.get("sliced", 0) == sliced + (order == "attach_then_pop"), f"PTA-SLICE: {order} slice count"


def test_new_request_data_pickle_roundtrip_is_identical():
    """vLLM 0.31's NewRequestData (PD's pinned vLLM): pickled and loaded with the int32 array, every field equals the
    list version's and the prompt values are identical."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    rnd = random.Random(9)
    ids = [rnd.randrange(0, 151_000) for _ in range(94_000)]
    r = Request("rq", list(ids), SamplingParams(max_tokens=1), None)
    off = NewRequestData.from_request(r, ([1, 2, 3],))
    on = NewRequestData.from_request(r, ([1, 2, 3],))
    P.substitute(SimpleNamespace(requests={"rq": r}), _out(on))
    a, b = pickle.loads(pickle.dumps(off)), pickle.loads(pickle.dumps(on))
    assert isinstance(b.prompt_token_ids, np.ndarray) and b.prompt_token_ids.dtype == np.int32, "PTA-IDENT: dtype"
    assert b.prompt_token_ids.tolist() == a.prompt_token_ids == ids, "PTA-IDENT: prompt values differ"
    da, db = dict(vars(a)), dict(vars(b))
    da.pop("prompt_token_ids")
    db.pop("prompt_token_ids")
    assert da == db, "PTA-IDENT: other NewRequestData fields differ"
    assert r.prompt_token_ids == ids, "PTA-IDENT: the Request's list is untouched"


def _worker_031(nrd):
    """vLLM 0.31's worker side for one new request: CachedRequestState as GPUModelRunner._update_states builds it from
    NewRequestData (output_token_ids=[]), then InputBatch.add_request. 0.31's InputBatch requires kernel_block_sizes
    (MM's 0.32 helper omits it, which is why MM's two InputBatch tests fail on PD's tree)."""
    import torch
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    batch = InputBatch(max_num_reqs=4, max_model_len=131_072, max_num_batched_tokens=8192, device=torch.device("cpu"),
                       vocab_size=151_936, block_sizes=[1024], kernel_block_sizes=[1024],
                       max_num_blocks_per_req=[128], num_spec_tokens=3)
    state = CachedRequestState(req_id=nrd.req_id, prompt_token_ids=nrd.prompt_token_ids,
                               prompt_embeds=nrd.prompt_embeds, prompt_is_token_ids=nrd.prompt_is_token_ids,
                               mm_features=nrd.mm_features, sampling_params=nrd.sampling_params,
                               pooling_params=nrd.pooling_params, generator=None, block_ids=nrd.block_ids,
                               num_computed_tokens=nrd.num_computed_tokens, output_token_ids=[],
                               lora_request=nrd.lora_request)
    return batch, state, batch.add_request(state)


def test_worker_input_batch_identity_on_vllm_031():
    """The served order on PD's vLLM 0.31: arrival conversion (chunked, 94,000 > CHUNK), schedule-time substitution,
    the executor's pickle broadcast, then the worker's request state and input batch. Every worker-visible quantity
    equals the list path's, and the identity-log row hash is the sha1 of the prompt's int32 bytes."""
    import hashlib

    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    from vllm_torchtpu.core import q38mm_tokhash as H

    rnd = random.Random(11)
    ids = [rnd.randrange(0, 151_000) for _ in range(94_000)]
    r = Request("rw", list(ids), SamplingParams(temperature=0.0, max_tokens=16), None)
    P.attach(r)
    rebuilt = P.stats["rebuilt"]
    off = NewRequestData.from_request(r, ([0, 1, 2],))
    on = NewRequestData.from_request(r, ([0, 1, 2],))
    P.substitute(SimpleNamespace(requests={"rw": r}), _out(on))
    assert P.stats["rebuilt"] == rebuilt, "PTA-WORKER: the arrival array must ship (no schedule-time rebuild)"
    off, on = pickle.loads(pickle.dumps(off)), pickle.loads(pickle.dumps(on))
    assert isinstance(on.prompt_token_ids, np.ndarray), "PTA-WORKER: the array path must reach the worker"
    ba, sa, ia = _worker_031(off)
    bb, sb, ib = _worker_031(on)
    n = len(ids)
    assert ia == ib, "PTA-WORKER: batch index"
    assert int(ba.num_prompt_tokens[ia]) == int(bb.num_prompt_tokens[ib]) == n, "PTA-WORKER: num_prompt_tokens"
    assert sa.num_prompt_tokens == sb.num_prompt_tokens == n, "PTA-WORKER: request state prompt length"
    assert int(ba.num_tokens_no_spec[ia]) == int(bb.num_tokens_no_spec[ib]) == n, "PTA-WORKER: num_tokens_no_spec"
    assert np.array_equal(ba.token_ids_cpu[ia, :n], bb.token_ids_cpu[ib, :n]), "PTA-WORKER: token_ids_cpu row differs"
    assert ba.token_ids_cpu[ia, :n].tolist() == ids, "PTA-WORKER: token_ids_cpu row is not the prompt"
    assert np.array_equal(ba.is_token_ids[ia, :n], bb.is_token_ids[ib, :n]), "PTA-WORKER: is_token_ids"
    for idx in (0, 1, 8191, 8192, 12_345, n - 1, n, n + 1):
        assert int(sa.get_token_id(idx)) == int(sb.get_token_id(idx)), f"PTA-WORKER: get_token_id({idx})"
    ha, hb = H.row_hash(ba.token_ids_cpu, ia, n), H.row_hash(bb.token_ids_cpu, ib, n)
    assert ha == hb == hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()[:16], "PTA-WORKER: row hash"


class _TailStandIn:
    """ADM-P's PromptTail contract (runner/prompt_tail.py, 26d59ed31), for trees without ADM-P: len() is the full
    length, reads at or above start return the tail, and reads below start, iteration and numpy conversion raise."""

    def __init__(self, length, start, tail):
        self.length, self.start, self.tail = length, start, list(tail)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if isinstance(i, slice):
            raise RuntimeError(f"PromptTailError: slice {i}")
        i = int(i) + (self.length if int(i) < 0 else 0)
        if i < self.start:
            raise RuntimeError(f"PromptTailError: index {i}")
        return self.tail[i - self.start]

    def __iter__(self):
        raise RuntimeError("PromptTailError: iteration")

    def __array__(self, *args, **kwargs):
        raise RuntimeError("PromptTailError: numpy conversion")


def _adm_p_tail(prompt, start):
    """The prompt as a decode engine with ADM-P ships it. Where ADM-P exists (q38-stack-v4) this is its real engine half
    (pd_prompt_tail.tail_new_request_data, window 64) with the real PromptTail; elsewhere the stand-in."""
    try:
        from vllm_torchtpu.core import pd_prompt_tail as pt
        from vllm_torchtpu.runner.prompt_tail import PromptTail
    except ImportError:
        return _TailStandIn(len(prompt), start, prompt[start:]), "stand-in"
    sp = SimpleNamespace(presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, prompt_logprobs=None)
    nrd = SimpleNamespace(prompt_token_ids=prompt, num_computed_tokens=start + 64, prompt_embeds=None,
                          prompt_is_token_ids=None, mm_features=[], sampling_params=sp)
    pt.tail_new_request_data(nrd, 64, PromptTail)
    assert isinstance(nrd.prompt_token_ids, PromptTail), "PTA-ADMP: setup: ADM-P's engine half did not trim"
    return nrd.prompt_token_ids, "real ADM-P"


def test_adm_p_prompt_tails_ship_untouched():
    """A decode engine with ADM-P (VLLM_TPU_PD_PROMPT_TAIL; in q38-stack-v4) has already replaced the prompt by a
    PromptTail inside NewRequestData.from_request when the base schedule returns. P15 must ship it untouched, with or
    without an arrival array: matching it to the array would put the whole prompt back on the wire (ADM-P undone, the
    worker's tail count reads 0), and a rebuild would iterate it (PromptTailError on the engine core)."""
    for attached in (True, False):
        req = _req("rt", 5000, seed=5)
        if attached:
            P.attach(req)
        tail, kind = _adm_p_tail(req.prompt_token_ids, 4000)
        if os.environ.get("Q38_P15_EXPECT_ADMP") == "1":
            assert kind == "real ADM-P", f"PTA-ADMP: this tree must exercise the real ADM-P path, got {kind}"
        assert len(tail) == 5000 and int(tail[-1]) == int(req.prompt_token_ids[-1]), "PTA-ADMP: setup: tail contract"
        nrd = SimpleNamespace(req_id="rt", prompt_token_ids=tail)
        try:
            n = P.substitute(SimpleNamespace(requests={"rt": req}), _out(nrd))
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"PTA-ADMP: attached={attached}: substitute raised {type(e).__name__}: {e}") from e
        assert nrd.prompt_token_ids is tail, f"PTA-ADMP: attached={attached}: the PromptTail must ship untouched"
        assert n == 0, f"PTA-ADMP: attached={attached}: a PromptTail is not a substitution (n={n})"
        assert isinstance(req.prompt_token_ids, list) and len(req.prompt_token_ids) == 5000, "PTA-ADMP: Request list"


def test_the_flag_is_not_a_compile_factor(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler as tc

    monkeypatch.delenv(P.ENV, raising=False)
    off = tc._tpu_compile_env_factors()
    monkeypatch.setenv(P.ENV, "1")
    assert tc._tpu_compile_env_factors() == off, "PTA-KEY: the flag must not change the compile-env factors (host-only)"
