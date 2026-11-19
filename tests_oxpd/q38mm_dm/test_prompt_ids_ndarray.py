# SPDX-License-Identifier: Apache-2.0
"""CPU tests for VLLM_TPU_PROMPT_IDS_NDARRAY (q38mm DM lane, P15), against the image's vLLM 0.32 classes.

Identity: the shipped array equals the request's prompt list, the request's list is untouched, a pickle round trip
through the shm broadcast's pickler gives the same tokens, and the worker's InputBatch ends with identical
token_ids_cpu / num_prompt_tokens / get_token_id for list and array prompts. Engagement: the scheduler hook
substitutes only when the flag is on, and the engine hook caches the array on the Request at arrival.
"""

import copyreg
import io
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.core import prompt_ids_ndarray as P

VOCAB = 151_936


def _prompt(n, seed=0):
    rnd = random.Random(seed)
    return [rnd.randrange(0, VOCAB) for _ in range(n)]


def _nrd(req_id, prompt, n_groups=4):
    from vllm.v1.core.sched.output import NewRequestData

    blocks = tuple(list(range(g * 1000, g * 1000 + max(1, len(prompt) // 512))) for g in range(n_groups))
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=blocks,
        num_computed_tokens=max(0, len(prompt) - 2048),
        lora_request=None,
    )


def _so(nrds):
    from vllm.v1.core.sched.output import SchedulerOutput

    so = SchedulerOutput.make_empty()
    so.scheduled_new_reqs = list(nrds)
    so.num_scheduled_tokens = {n.req_id: 2048 for n in nrds}
    so.total_num_scheduled_tokens = 2048 * len(nrds)
    return so


def _shm_pickle(obj):
    """The shm broadcast's serialization (MessageQueue.enqueue / dequeue, vLLM 0.32)."""
    bufs = [b""]

    def oob(buf):
        raw = buf.raw()
        if len(raw) < 1024 * 1024:
            return True
        bufs.append(raw)
        return False

    with io.BytesIO() as bio:
        p = pickle.Pickler(bio, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=oob)
        p.dispatch_table = dict(copyreg.dispatch_table)
        p.dump(obj)
        bufs[0] = bio.getvalue()
    return pickle.loads(bufs[0], buffers=[bytearray(b) for b in bufs[1:]])


# --------------------------------------------------------------------------------------------- conversion
def test_to_array_int32_equal():
    ids = _prompt(5000)
    arr = P.to_array(ids)
    assert arr.dtype == np.int32 and arr.shape == (5000,)
    assert arr.tolist() == ids


def test_to_array_refuses_out_of_range():
    with pytest.raises(OverflowError):
        P.to_array([1, 2, 2**31])


@pytest.mark.parametrize("n", [0, 1, 8191, 8192, 8193, 94_000])
def test_chunked_equals_unchunked(n):
    ids = _prompt(n, 9)
    a, b = P.to_array(ids), P.to_array_chunked(ids)
    assert b.dtype == np.int32 and np.array_equal(a, b) and b.tolist() == ids


def test_chunked_refuses_out_of_range():
    ids = _prompt(20_000, 10)
    ids[15_000] = 2**31
    with pytest.raises(OverflowError):
        P.to_array_chunked(ids)


def test_vocab_max_survives():
    ids = [0, VOCAB - 1, 2**31 - 1]
    assert P.to_array(ids).tolist() == ids


# --------------------------------------------------------------------------------------------- scheduler side
def test_substitute_replaces_with_cached_array_and_keeps_request_list():
    ids = _prompt(94_000, 1)
    req = SimpleNamespace(prompt_token_ids=ids)
    P.attach(req)
    nrd = _nrd("r0", req.prompt_token_ids)
    sched = SimpleNamespace(requests={"r0": req})
    so = _so([nrd])
    assert P.substitute(sched, so) == 1
    got = so.scheduled_new_reqs[0].prompt_token_ids
    assert isinstance(got, np.ndarray) and got.dtype == np.int32
    assert got is getattr(req, P.ATTR)
    assert got.tolist() == ids
    assert req.prompt_token_ids is ids and isinstance(req.prompt_token_ids, list)
    assert so.scheduled_new_reqs[0].prompt_len == len(ids)


@pytest.mark.parametrize("stale", ["length", "last", "missing"])
def test_substitute_rebuilds_a_stale_cache(stale):
    ids = _prompt(3000, 2)
    req = SimpleNamespace(prompt_token_ids=ids)
    if stale == "length":
        setattr(req, P.ATTR, P.to_array(ids[:-1]))
    elif stale == "last":
        bad = list(ids)
        bad[-1] = (bad[-1] + 1) % VOCAB
        setattr(req, P.ATTR, P.to_array(bad))
    so = _so([_nrd("r0", ids)])
    P.substitute(SimpleNamespace(requests={"r0": req}), so)
    assert so.scheduled_new_reqs[0].prompt_token_ids.tolist() == ids


def test_substitute_without_request_object_still_converts():
    ids = _prompt(100, 3)
    so = _so([_nrd("ghost", ids)])
    assert P.substitute(SimpleNamespace(requests={}), so) == 1
    assert so.scheduled_new_reqs[0].prompt_token_ids.tolist() == ids


def test_substitute_leaves_none_prompts():
    nrd = _nrd("e", [1, 2, 3])
    nrd.prompt_token_ids = None
    so = _so([nrd])
    assert P.substitute(SimpleNamespace(requests={}), so) == 0
    assert so.scheduled_new_reqs[0].prompt_token_ids is None


def test_shm_pickle_roundtrip_identity():
    ids = _prompt(94_000, 4)
    off = _so([_nrd("r0", list(ids))])
    on = _so([_nrd("r0", list(ids))])
    P.substitute(SimpleNamespace(requests={}), on)
    a, b = _shm_pickle(off), _shm_pickle(on)
    na, nb = a.scheduled_new_reqs[0], b.scheduled_new_reqs[0]
    assert isinstance(nb.prompt_token_ids, np.ndarray)
    assert list(map(int, nb.prompt_token_ids)) == na.prompt_token_ids == ids
    da, db = dict(vars(na)), dict(vars(nb))
    da.pop("prompt_token_ids")
    db.pop("prompt_token_ids")
    assert da == db
    assert a.num_scheduled_tokens == b.num_scheduled_tokens


def test_scheduler_hook_substitutes_only_when_on(monkeypatch):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

    ids = _prompt(2000, 5)
    for on in (False, True):
        so = _so([_nrd("r0", ids)])
        monkeypatch.setattr(AsyncScheduler, "schedule", lambda self, throttle_prefills=False, _so=so: _so)
        s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
        s._q38_prompt_np = on
        s.requests = {"r0": SimpleNamespace(prompt_token_ids=ids)}
        out = s.schedule()
        got = out.scheduled_new_reqs[0].prompt_token_ids
        if on:
            assert isinstance(got, np.ndarray) and got.tolist() == ids
        else:
            assert got is ids


def test_engine_hook_caches_at_arrival(monkeypatch):
    from vllm.v1.engine import core as core_mod

    ids = _prompt(1000, 6)
    req = SimpleNamespace(prompt_token_ids=ids)
    monkeypatch.setattr(core_mod.EngineCore, "preprocess_add_request", lambda self, r: (req, 0))
    monkeypatch.setattr(P, "_installed", False)
    assert P.install_engine_hook()
    out = core_mod.EngineCore.preprocess_add_request(None, object())
    assert out == (req, 0)
    assert getattr(req, P.ATTR).tolist() == ids
    assert P.install_engine_hook()  # idempotent


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(P.ENV, raising=False)
    assert not P.enabled()
    monkeypatch.setenv(P.ENV, "1")
    assert P.enabled()


# --------------------------------------------------------------------------------------------- worker side
def _input_batch():
    from vllm.v1.worker.gpu_input_batch import InputBatch

    return InputBatch(max_num_reqs=4, max_model_len=131_072, max_num_batched_tokens=8192,
                      device=torch.device("cpu"), vocab_size=VOCAB, block_sizes=[1024],
                      max_num_blocks_per_req=[128], num_spec_tokens=3)


def _state(req_id, prompt):
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    from vllm.sampling_params import SamplingParams

    return CachedRequestState(req_id=req_id, prompt_token_ids=prompt, mm_features=[],
                              sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
                              generator=None, block_ids=([0, 1, 2],), num_computed_tokens=0,
                              output_token_ids=[7, 8])


def test_worker_input_batch_identity():
    ids = _prompt(94_000, 7)
    ba, bb = _input_batch(), _input_batch()
    sa, sb = _state("r0", list(ids)), _state("r0", P.to_array(ids))
    ia, ib = ba.add_request(sa), bb.add_request(sb)
    assert ia == ib
    n = len(ids) + 2
    assert np.array_equal(ba.token_ids_cpu[ia, :n], bb.token_ids_cpu[ib, :n])
    assert ba.token_ids_cpu[ia, :len(ids)].tolist() == ids
    assert int(ba.num_prompt_tokens[ia]) == int(bb.num_prompt_tokens[ib]) == len(ids)
    assert sa.num_prompt_tokens == sb.num_prompt_tokens == len(ids)
    for idx in (0, 1, len(ids) - 1, len(ids), len(ids) + 1, len(ids) + 5):
        assert int(sa.get_token_id(idx)) == int(sb.get_token_id(idx))
    assert np.array_equal(ba.is_token_ids[ia, :n], bb.is_token_ids[ib, :n])


# --------------------------------------------------------------------------------------------- identity log (A/B gate)
def test_tokhash_identical_for_list_and_array(monkeypatch):
    from vllm_torchtpu.core import q38mm_tokhash as H

    ids = _prompt(94_000, 8)
    ba, bb = _input_batch(), _input_batch()
    ba.add_request(_state("r0", list(ids)))
    bb.add_request(_state("r0", P.to_array(ids)))
    ha = H.row_hash(ba.token_ids_cpu, ba.req_id_to_index["r0"], len(ids))
    hb = H.row_hash(bb.token_ids_cpu, bb.req_id_to_index["r0"], len(ids))
    assert ha == hb
    import hashlib

    assert ha == hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()[:16]  # the whole prompt row
    changed = list(ids)
    changed[12_345] = (changed[12_345] + 1) % VOCAB
    bc = _input_batch()
    bc.add_request(_state("r0", changed))
    assert H.row_hash(bc.token_ids_cpu, 0, len(ids)) != ha
    lines = []
    monkeypatch.setattr(H, "_ON", True)
    monkeypatch.setattr(H.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    assert H.log_new_requests(bb, _so([_nrd("r0", ids)])) == 1
    assert lines == [f"Q38MM_TOKHASH rid=r0 n={len(ids)} sha1={ha}"]
    monkeypatch.setattr(H, "_ON", False)
    assert H.log_new_requests(bb, _so([_nrd("r0", ids)])) == 0
