# SPDX-License-Identifier: Apache-2.0
"""TPU_COMPACT_NEW_REQ_TOKENS: new requests' prompt ids travel to the workers
as int32 arrays. Off, nothing is patched; on, the worker sees the same token
values (pickled through the scheduler-output path) and fills the same
InputBatch row.
"""

import io
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine.core import EngineCore
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_torchtpu import envs
from vllm_torchtpu.core import compact_prompt_tokens as cpt

_PATCHED = (
    (EngineCore, "preprocess_add_request"),
    (NewRequestData, "from_request"),
    (CachedRequestState, "get_token_id"),
)


@pytest.fixture
def restore_classes():
    saved = [(cls, name, cls.__dict__[name]) for cls, name in _PATCHED]
    yield
    for cls, name, orig in saved:
        setattr(cls, name, orig)
        if "_tpu_compact_prompt_tokens" in cls.__dict__:
            delattr(cls, "_tpu_compact_prompt_tokens")


def _request(ids, num_computed=0):
    return SimpleNamespace(
        request_id="r0",
        prompt_token_ids=ids,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=num_computed,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        replay_start=0,
    )


def _shm_roundtrip(obj):
    """Pickle like shm_broadcast.MessageQueue: protocol 5, >=1 MiB out of band."""
    oob = []

    def cb(buf):
        raw = buf.raw()
        if len(raw) < 1024 * 1024:
            return True
        oob.append(raw)
        return False

    bio = io.BytesIO()
    pickle.Pickler(bio, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=cb).dump(obj)
    return pickle.loads(bio.getvalue(), buffers=[bytearray(b) for b in oob]), len(oob)


def test_off_patches_nothing(monkeypatch, restore_classes):
    monkeypatch.setattr(envs, "TPU_COMPACT_NEW_REQ_TOKENS", False, raising=False)
    before = [cls.__dict__[name] for cls, name in _PATCHED]
    cpt.install_compact_prompt_tokens()
    assert [cls.__dict__[name] for cls, name in _PATCHED] == before
    ids = list(range(10000))
    data = NewRequestData.from_request(_request(ids), ([1, 2],))
    assert data.prompt_token_ids is ids


def test_on_sends_int32_array_with_same_values(monkeypatch, restore_classes):
    monkeypatch.setattr(envs, "TPU_COMPACT_NEW_REQ_TOKENS", True, raising=False)
    monkeypatch.setattr(envs, "TPU_COMPACT_NEW_REQ_TOKENS_MIN", 4096, raising=False)
    cpt.install_compact_prompt_tokens()
    cpt.install_compact_prompt_tokens()  # idempotent
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 200_064, size=300_000).tolist()
    req = _request(ids, num_computed=290_000)
    setattr(req, cpt._ATTR, cpt.to_compact(ids, 4096))
    data = NewRequestData.from_request(req, ([7, 8, 9],))
    assert isinstance(data.prompt_token_ids, np.ndarray)
    assert data.prompt_token_ids.dtype == np.int32
    assert data.block_ids == ([7, 8, 9],) and data.num_computed_tokens == 290_000

    got, n_oob = _shm_roundtrip(data)
    assert n_oob == 1  # 1.2 MB of ids: one out-of-band buffer
    assert got.prompt_token_ids.tolist() == ids

    # Worker side: the InputBatch row fill and get_token_id see the same ids.
    row_list = np.zeros(400_000, dtype=np.int32)
    row_arr = np.zeros(400_000, dtype=np.int32)
    row_list[: len(ids)] = ids
    row_arr[: len(ids)] = got.prompt_token_ids
    assert np.array_equal(row_list, row_arr)
    state = CachedRequestState(
        req_id="r0",
        prompt_token_ids=got.prompt_token_ids,
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=([7, 8, 9],),
        num_computed_tokens=290_000,
        output_token_ids=[5],
    )
    assert state.num_prompt_tokens == len(ids)
    tok = state.get_token_id(123)
    assert type(tok) is int and tok == ids[123]
    assert state.get_token_id(len(ids)) == 5


def test_on_keeps_lists_when_short_or_changed(monkeypatch, restore_classes):
    monkeypatch.setattr(envs, "TPU_COMPACT_NEW_REQ_TOKENS", True, raising=False)
    monkeypatch.setattr(envs, "TPU_COMPACT_NEW_REQ_TOKENS_MIN", 4096, raising=False)
    cpt.install_compact_prompt_tokens()
    assert cpt.to_compact(list(range(100)), 4096) is None
    assert cpt.to_compact([2**40] * 5000, 4096) is None  # does not fit int32
    short = list(range(100))
    assert NewRequestData.from_request(_request(short), ([1],)).prompt_token_ids is short
    # The prompt changed after the conversion (e.g. a P/D pop): send the list.
    ids = list(range(5000))
    req = _request(ids)
    setattr(req, cpt._ATTR, cpt.to_compact(ids, 4096))
    ids.pop()
    assert NewRequestData.from_request(req, ([1],)).prompt_token_ids is ids


def test_preprocess_add_request_attaches_array(monkeypatch, restore_classes):
    monkeypatch.setattr(envs, "TPU_COMPACT_NEW_REQ_TOKENS", True, raising=False)
    monkeypatch.setattr(envs, "TPU_COMPACT_NEW_REQ_TOKENS_MIN", 4096, raising=False)
    ids = list(range(8192))
    req = _request(ids)
    monkeypatch.setattr(
        EngineCore, "preprocess_add_request", lambda self, request: (req, 3)
    )
    cpt.install_compact_prompt_tokens()
    out, wave = EngineCore.preprocess_add_request(None, object())
    assert out is req and wave == 3
    arr = getattr(req, cpt._ATTR)
    assert arr.dtype == np.int32 and arr.tolist() == ids
