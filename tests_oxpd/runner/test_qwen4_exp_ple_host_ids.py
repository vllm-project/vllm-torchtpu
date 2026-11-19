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
"""VLLM_TPU_PLE_HOST_IDS: stage 0's PLE table rows computed on the host from the
host arrays the step's device inputs come from, identical to the device ids."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_torchtpu.models.vllm.qwen4_exp import ple_offload
from vllm_torchtpu.models.vllm.qwen4_exp.inputs import (
    Qwen4ExpAdapter,
    ngram_context_from_store,
)
from vllm_torchtpu.models.vllm.qwen4_exp.ple import (
    PLEInputs,
    Qwen4ExpNGramEmbedding,
    head_vocab_sizes,
    layer_multipliers,
)
from vllm_torchtpu.models.vllm.qwen4_exp.ple_storage import HostPLETable
from vllm_torchtpu.models.vllm.qwen4_exp.pooled_ple import Qwen4ExpPooledPLE

EOS = 7
NGRAM = 3
HEADS_PER_NGRAM = 4
VOCAB = 151_936


def _ple(across_dp: bool = False):
    """A PLE layer double: the real hash constants, compute_ids and host_ids."""
    heads = (NGRAM - 1) * HEADS_PER_NGRAM
    sizes = head_vocab_sizes(20_000_000, heads, 0)
    emb = SimpleNamespace(
        layer_multipliers=torch.tensor(layer_multipliers(VOCAB, NGRAM, 0, 1234)),
        ngram_heads_vocab_sizes=torch.tensor(sizes),
        ngram_heads_offsets=torch.tensor([sum(sizes[:h]) for h in range(heads)]),
        ngram_size=NGRAM,
        eos_token_id=EOS,
        heads_per_ngram=HEADS_PER_NGRAM,
    )
    emb.compute_ids = Qwen4ExpNGramEmbedding.compute_ids.__get__(emb)
    ple = SimpleNamespace(
        ple_embedding=emb,
        parallel=SimpleNamespace(across_dp=across_dp, gather_ids=lambda ids: ids),
        prefix="model.layers.0.ple",
    )
    ple.host_ids = Qwen4ExpPooledPLE.host_ids.__get__(ple)
    ple._lookup_ids = Qwen4ExpPooledPLE._lookup_ids.__get__(ple)
    return ple


def _step(seed: int = 0):
    """Host arrays of one step: 3 real requests (one fresh, two with history,
    an EOS inside a history), padded to 5 request rows and 64 tokens."""
    g = torch.Generator().manual_seed(seed)
    rows, max_len, num_tokens = 5, 200, 64
    store = torch.randint(10, VOCAB, (8, max_len), generator=g, dtype=torch.int32)
    store[1, 40] = EOS  # inside request 1's history
    qlens = [20, 17, 9]  # real tokens 46 of 64
    computed = [0, 42, 120]
    qsl = torch.zeros(rows + 1, dtype=torch.int32)
    qsl[1:4] = torch.tensor(qlens).cumsum(0)
    qsl[4:] = qsl[3]  # padding rows repeat the total
    seq = torch.zeros(rows, dtype=torch.int32)
    seq[:3] = torch.tensor(computed) + torch.tensor(qlens)
    input_ids = torch.zeros(num_tokens, dtype=torch.int32)
    for r in range(3):
        a, b = int(qsl[r]), int(qsl[r + 1])
        input_ids[a:b] = store[r, computed[r] : computed[r] + qlens[r]]
    input_ids[46:] = 0  # the runner zeroes the padded tail
    return SimpleNamespace(
        store=store, qsl=qsl, seq=seq, input_ids=input_ids, rows=rows, num_tokens=num_tokens
    )


def _device_ids(ple, st):
    """What the device graph computes (_prepare_inputs_eager's history, then
    _lookup_ids), run on CPU copies of the uploaded arrays."""
    context = (st.seq - (st.qsl[1:] - st.qsl[:-1])).to(torch.int64)
    row_ids = (torch.arange(st.rows, dtype=torch.int64)).clamp(max=st.store.shape[0] - 1)
    history = ngram_context_from_store(st.store.clone(), row_ids, context, EOS, NGRAM)
    return ple._lookup_ids(PLEInputs(st.input_ids.clone(), st.qsl.clone(), history), st.qsl[-1:])


def _runner(st, *, num_reqs=3, async_scheduling=False):
    return SimpleNamespace(
        query_start_loc_cpu=st.qsl,
        seq_lens_cpu=st.seq,
        input_ids_cpu=st.input_ids,
        input_batch=SimpleNamespace(token_ids_cpu_tensor=st.store, num_reqs=num_reqs),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )


def _adapter(runner, spec_window=0):
    a = object.__new__(Qwen4ExpAdapter)
    a.runner = runner
    a.config = SimpleNamespace(eos_token_id=EOS, ngram_size=NGRAM)
    a.spec_window = spec_window
    return a


@pytest.fixture(autouse=True)
def _clean():
    ple_offload.clear_host_ids()
    for k in ple_offload._host_ids_stats:
        ple_offload._host_ids_stats[k] = 0
    yield
    ple_offload.clear_host_ids()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_host_ids_equal_the_device_ids(seed):
    ple, st = _ple(), _step(seed)
    snapshot = SimpleNamespace(row_start=0, num_requests=3, num_tokens=st.num_tokens)
    _adapter(_runner(st))._stash_host_ple_ids(ple, snapshot, st.rows)
    host = ple_offload._host_ids[ple.prefix]
    device = _device_ids(ple, st)
    assert host.dtype == device.dtype == torch.int32
    assert host.shape == device.shape == (64, (NGRAM - 1) * HEADS_PER_NGRAM)
    assert torch.equal(host, device)
    # Padded tokens are -1 and real tokens are real rows.
    assert (host[46:] == -1).all() and (host[:46] >= 0).all()


def test_eos_in_history_resets_the_ngram():
    """Request 1's history crosses an EOS: its first token's older head reads EOS."""
    ple, st = _ple(), _step(0)
    st.store[1, 41] = EOS  # the token just before request 1's chunk
    snapshot = SimpleNamespace(row_start=0, num_requests=3, num_tokens=st.num_tokens)
    _adapter(_runner(st))._stash_host_ple_ids(ple, snapshot, st.rows)
    assert torch.equal(ple_offload._host_ids[ple.prefix], _device_ids(ple, st))


@pytest.mark.parametrize(
    "change",
    [
        {"row_start": 1},
        {"num_requests": 2},
        {"spec_window": 2},
        {"async_scheduling": True},
        {"across_dp": True},
    ],
)
def test_unsupported_steps_stash_nothing(change):
    ple, st = _ple(across_dp=change.get("across_dp", False)), _step(0)
    snapshot = SimpleNamespace(
        row_start=change.get("row_start", 0),
        num_requests=change.get("num_requests", 3),
        num_tokens=st.num_tokens,
    )
    runner = _runner(st, async_scheduling=change.get("async_scheduling", False))
    _adapter(runner, spec_window=change.get("spec_window", 0))._stash_host_ple_ids(
        ple, snapshot, st.rows
    )
    assert ple.prefix not in ple_offload._host_ids


def test_an_unsupported_step_clears_an_earlier_stash():
    ple, st = _ple(), _step(0)
    ple_offload.stash_host_ids(ple.prefix, torch.zeros(64, 8, dtype=torch.int32))
    snapshot = SimpleNamespace(row_start=0, num_requests=2, num_tokens=st.num_tokens)
    _adapter(_runner(st))._stash_host_ple_ids(ple, snapshot, st.rows)
    assert ple.prefix not in ple_offload._host_ids


def test_a_stash_is_used_once_and_must_match():
    ids = torch.zeros(4, 8, dtype=torch.int32)
    ple_offload.stash_host_ids("p", torch.ones(4, 8, dtype=torch.int32))
    got = ple_offload.take_host_ids("p", ids)
    assert got is not None and int(got.sum()) == 32
    assert ple_offload.take_host_ids("p", ids) is None  # used once
    ple_offload.stash_host_ids("p", torch.ones(5, 8, dtype=torch.int32))
    assert ple_offload.take_host_ids("p", ids) is None  # shape differs
    ple_offload.stash_host_ids("p", torch.ones(4, 8, dtype=torch.int64))
    assert ple_offload.take_host_ids("p", ids) is None  # dtype differs
    assert ple_offload._host_ids_stats["fallback"] == 2
    assert ple_offload._host_ids_stats["used"] == 1


def test_check_mode_counts_differing_ids(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_PLE_HOST_IDS_CHECK", "2")
    ids = torch.zeros(4, 8, dtype=torch.int32)
    ple_offload.stash_host_ids("p", torch.zeros(4, 8, dtype=torch.int32))
    ple_offload.take_host_ids("p", ids)
    ple_offload.stash_host_ids("p", torch.ones(4, 8, dtype=torch.int32))
    ple_offload.take_host_ids("p", ids)
    assert ple_offload._host_ids_stats["checked"] == 2
    assert ple_offload._host_ids_stats["bad"] == 1


@pytest.mark.parametrize("timing", [False, True])
@pytest.mark.parametrize("stashed", [False, True])
def test_host_lookup_reads_the_stash_instead_of_the_device_ids(monkeypatch, timing, stashed):
    monkeypatch.setattr(ple_offload.envs, "VLLM_TPU_PLE_HOST_TIMING", timing, raising=False)
    table = MagicMock(spec=HostPLETable)
    seen = []

    def lookup_host(host_ids):
        seen.append(host_ids.clone())
        return torch.zeros(host_ids.shape[0], 8, dtype=torch.int8)

    table.lookup_host.side_effect = lookup_host
    layer = SimpleNamespace(ple_embedding=SimpleNamespace(ngram_embedding=table))
    ctx = SimpleNamespace(no_compile_layers={"p": layer})
    device_ids = torch.full((4, 2), 3, dtype=torch.int32)
    if stashed:
        ple_offload.stash_host_ids("p", torch.full((4, 2), 9, dtype=torch.int32))
    with patch.object(ple_offload, "get_forward_context", return_value=ctx):
        out = ple_offload.host_lookup(device_ids, 4, 8, 1, True, "p")
    assert out.shape == (1, 4, 8)
    assert len(seen) == 1
    assert int(seen[0][0, 0]) == (9 if stashed else 3)


def test_bind_model_finds_the_host_ple_layer_only_when_on(monkeypatch):
    ple = SimpleNamespace(prefix="model.layers.0.ple")
    backbone = SimpleNamespace(host_ple_index=0, layers=[SimpleNamespace(ple=ple)])
    other = SimpleNamespace(host_ple_index=None)
    model = SimpleNamespace(modules=lambda: iter([other, backbone]))
    for value, want in (("1", ple), ("0", None)):
        monkeypatch.setenv("VLLM_TPU_PLE_HOST_IDS", value)
        a = object.__new__(Qwen4ExpAdapter)
        a.runner = SimpleNamespace()
        a.pooled = False
        with patch("vllm_torchtpu.models.vllm.qwen4_exp.inputs.Qwen4ExpCache", MagicMock()):
            a.bind_model(model)
        assert a._host_ple is want
