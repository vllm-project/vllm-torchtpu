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
"""Host critical-path switches of the decode loop (all default off):
TPU_HOST_STAGE_TIMING_HOST_ONLY, TPU_HOST_TOKEN_PADDING_REUSE,
TPU_HOST_SAMPLING_PARAMS_REUSE and TPU_HOST_SKIP_GUARD_EVAL_AFTER_STEPS.
Each must leave every value the device sees unchanged."""

import inspect
import random
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState
from vllm_torchtpu.runner import host_path, stage_timing
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

# --- stage timing ---------------------------------------------------------


def test_host_only_stage_timing_records_no_device_event(monkeypatch):
    def no_event(**_):
        raise AssertionError("host-only timing must not create a TpuEvent")

    monkeypatch.setattr(stage_timing, "TpuEvent", no_event)
    timing = stage_timing.RunnerStageTiming(2, device_events=False)
    with timing.stage("main_forward"):
        pass
    with pytest.raises(ValueError):
        with timing.stage("sample"):
            raise ValueError("body failed")
    cpu, device = timing.take_samples()
    assert [(r, s) for r, s, _ in cpu] == [("2", "main_forward"), ("2", "sample")]
    assert all(t >= 0 for _, _, t in cpu)
    assert device == []


def test_runner_wires_stage_timing_switch():
    src = TPUModelRunner.__init__.__code__.co_names
    assert "TPU_HOST_STAGE_TIMING_HOST_ONLY" in src


# --- token padding --------------------------------------------------------


class _CountingMask:
    """A bool mask tensor that counts the writes `update` issues."""

    def __init__(self, n):
        self.t = torch.zeros(n, dtype=torch.bool)
        self.writes = 0

    def state(self, reuse):
        state = TokenPaddingState(local_padding_mask=self.t, reuse=reuse)
        orig = self.t.copy_

        def counted(src):
            self.writes += 1
            return orig(src)

        state.local_padding_mask = SimpleNamespace(
            shape=self.t.shape, copy_=counted, __getitem__=self.t.__getitem__
        )
        return state


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_token_padding_reuse_writes_the_same_mask(seed):
    rng = random.Random(seed)
    # Steady decode repeats a few (valid, padded) pairs: target then drafter.
    pairs = [(8, 8), (8, 8), (40, 48), (40, 48), (0, 16), (24, 24), (3, 6)]
    seq = [rng.choice(pairs) for _ in range(200)]
    ref = TokenPaddingState(local_padding_mask=torch.zeros(64, dtype=torch.bool))
    fast_mask = _CountingMask(64)
    fast = fast_mask.state(reuse=True)
    for valid, padded in seq:
        ref.update(valid, padded)
        fast.update(valid, padded)
        assert torch.equal(fast_mask.t, ref.local_padding_mask)
    changes = 1 + sum(a != b for a, b in zip(seq, seq[1:]))
    assert fast_mask.writes == changes < len(seq)


def test_token_padding_without_reuse_writes_every_update():
    mask = _CountingMask(16)
    state = mask.state(reuse=False)
    for _ in range(5):
        state.update(8, 8)
    assert mask.writes == 5


def test_token_padding_create_reads_switch(monkeypatch):
    monkeypatch.setenv("TPU_HOST_TOKEN_PADDING_REUSE", "1")
    assert TokenPaddingState.create(8, "cpu").reuse
    monkeypatch.setenv("TPU_HOST_TOKEN_PADDING_REUSE", "0")
    assert not TokenPaddingState.create(8, "cpu").reuse


def test_token_padding_consumers_trace_after_skipped_updates():
    """The skipped update keeps the same mask tensor, so the traced MoE
    padding ops see one stable graph input (fake tensors, symbolic T)."""
    state = TokenPaddingState(
        local_padding_mask=torch.zeros(32, dtype=torch.bool), reuse=True
    )
    token_padding.set_padding_state(state)
    try:
        state.update(5, 8)
        mask_id = id(state.local_padding_mask)
        state.update(5, 8)
        assert id(state.local_padding_mask) == mask_id

        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def route(ids, weights, logits):
            ids, weights = token_padding.zero_routing_weights_for_padding(
                ids, weights, is_local_tensor=True
            )
            return ids, weights, token_padding.mask_router_logits_for_padding(logits)

        ids = torch.zeros(8, 2, dtype=torch.int32)
        weights = torch.ones(8, 2)
        logits = torch.zeros(8, 4)
        _, w, lg = route(ids, weights, logits)
        assert w[:5].eq(1).all() and w[5:].eq(0).all()
        assert torch.isnan(lg[5:]).all() and not torch.isnan(lg[:5]).any()

        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            fids = mode.from_tensor(torch.zeros(6, 2, dtype=torch.int32))
            fw = mode.from_tensor(torch.ones(6, 2))
            _, w2 = token_padding.zero_routing_weights_for_padding(
                fids, fw, is_local_tensor=True
            )
            assert tuple(w2.shape) == (6, 2)
    finally:
        token_padding.set_padding_state(None)


# --- sampling params ------------------------------------------------------


def _sampling_runner(num_reqs=6, vocab=1000):
    batch = SimpleNamespace(
        temperature_cpu_tensor=torch.rand(num_reqs) + 0.1,
        top_k_cpu_tensor=torch.randint(0, 2 * vocab, (num_reqs,), dtype=torch.int32),
        top_p_cpu_tensor=torch.rand(num_reqs),
    )
    return SimpleNamespace(
        input_batch=batch, vocab_size=vocab, _sampling_params_reuse=None
    )


def _params(runner, start, end, padded, dtype=torch.float32):
    return TPUModelRunner._padded_sampling_params(
        runner, start, end, padded, dtype, torch.device("cpu")
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sampling_params_reuse_matches_fresh_build(monkeypatch, dtype):
    runner = _sampling_runner()
    rng = random.Random(0)
    steps = []
    for i in range(60):
        if i % 7 == 3:  # a request's params change in place
            runner.input_batch.temperature_cpu_tensor[rng.randrange(6)] = rng.random()
        if i % 11 == 5:
            runner.input_batch.top_k_cpu_tensor[rng.randrange(6)] = rng.randrange(50)
        if i % 13 == 6:
            runner.input_batch.top_p_cpu_tensor[rng.randrange(6)] = rng.random()
        start, end = rng.choice([(0, 4), (0, 4), (0, 6), (2, 5)])
        padded = 8 if end - start > 4 else 4
        steps.append((start, end, padded))

        monkeypatch.setattr(envs, "TPU_HOST_SAMPLING_PARAMS_REUSE", False, raising=False)
        ref = _params(runner, start, end, padded, dtype)
        monkeypatch.setattr(envs, "TPU_HOST_SAMPLING_PARAMS_REUSE", True, raising=False)
        got = _params(runner, start, end, padded, dtype)
        for r, g in zip(ref, got):
            assert r.dtype == g.dtype and torch.equal(r, g)


def test_sampling_params_reuse_returns_previous_upload(monkeypatch):
    monkeypatch.setattr(envs, "TPU_HOST_SAMPLING_PARAMS_REUSE", True, raising=False)
    runner = _sampling_runner()
    first = _params(runner, 0, 4, 4)
    again = _params(runner, 0, 4, 4)
    assert all(a is b for a, b in zip(first, again))
    other_bucket = _params(runner, 0, 4, 8)
    assert other_bucket[0].shape[0] == 8
    runner.input_batch.top_p_cpu_tensor[1] = 0.123
    changed = _params(runner, 0, 8 // 2, 8)
    assert changed[2][1].item() == pytest.approx(0.123)
    assert changed[2] is not other_bucket[2]


def test_sampling_params_reuse_off_builds_every_call(monkeypatch):
    monkeypatch.setattr(envs, "TPU_HOST_SAMPLING_PARAMS_REUSE", False, raising=False)
    runner = _sampling_runner()
    a = _params(runner, 0, 4, 4)
    b = _params(runner, 0, 4, 4)
    assert all(x is not y for x, y in zip(a, b))
    assert runner._sampling_params_reuse is None


# --- guard eval stance ----------------------------------------------------


@pytest.fixture
def restore_stance():
    yield
    torch.compiler.set_stance("default", skip_guard_eval_unsafe=False)


def test_guard_skipper_off_never_arms(monkeypatch, restore_stance):
    skipper = host_path.GuardEvalSkipper(0)
    assert not skipper.pending
    assert not skipper.step()
    assert not skipper.armed


def test_guard_skipper_arms_after_quiet_steps(monkeypatch, restore_stance):
    graphs = iter([5, 5, 6, 6, 6, 6])
    monkeypatch.setattr(host_path, "_dynamo_graph_count", lambda: next(graphs))
    skipper = host_path.GuardEvalSkipper(3)
    # 5 (new), 5 (quiet 1), 6 (new graph resets), 6, 6, 6 -> arms on quiet 3.
    armed = [skipper.step() for _ in range(6)]
    assert armed == [False, False, False, False, False, True]
    assert skipper.armed and not skipper.pending
    assert not skipper.step()
    skipper.disarm()
    assert not skipper.armed and skipper.pending


def test_guard_skipper_keeps_compiled_outputs(restore_stance):
    """Armed after warmup over the buckets, the compiled function returns the
    same values for every warm bucket; a cold specialization raises (Dynamo
    refuses to recompile under the stance) instead of returning wrong data."""
    torch._dynamo.reset()

    class Owner:
        def __init__(self):
            self.scale = torch.tensor(3.0)

        @torch.compile(backend="eager", fullgraph=True, dynamic=False)
        def step(self, x, idx):
            return (x[idx] * self.scale).sum(-1)

    owner = Owner()
    buckets = (4, 8, 16)
    inputs = {n: (torch.randn(n, 5), torch.arange(n).flip(0)) for n in buckets}
    expected = {n: (x[i] * 3.0).sum(-1) for n, (x, i) in inputs.items()}
    for n in buckets:
        torch.testing.assert_close(owner.step(*inputs[n]), expected[n])
    skipper = host_path.GuardEvalSkipper(2)
    while not skipper.step():
        pass
    for _ in range(3):
        for n in buckets:
            got = owner.step(*inputs[n])
            assert torch.equal(got, owner.step.__wrapped__(owner, *inputs[n]))
    with pytest.raises(RuntimeError, match="skip_guard_eval_unsafe"):
        owner.step(torch.randn(32, 5), torch.arange(32))
    skipper.disarm()
    assert owner.step(torch.randn(32, 5), torch.arange(32)).shape == (32,)


def test_runner_ticks_guard_skipper_only_when_pending():
    src = inspect.getsource(inspect.unwrap(TPUModelRunner.execute_model))
    assert "if self._guard_eval_skipper.pending:" in src
