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
"""CPU check of DSv4 attention intra-chip TP against an unsharded reference.

Two threads stand in for the two cores of a chip, joined by a fake chip-pair
group, so the module's own ``attention_impl``/``_o_proj`` run unchanged; only
the projections, attention and the o-projection op are small stand-ins.
"""

import importlib.util
import pathlib
import sys
import threading
import types

import pytest
import torch

_MODULE = (
    pathlib.Path(__file__).parents[3]
    / "src/vllm_torchtpu/models/vllm/deepseek_v4/attn_intra_chip.py"
)

NUM_TOKENS = 3
NUM_HEADS = 16
HEAD_DIM = 8
Q_LORA_RANK = 6
NUM_GROUPS = 4
O_LORA_RANK = 5
HIDDEN_SIZE = 7


class _FakeBaseAttention:
    """Stands in for VllmDeepseekV4MLAAttention: only what the subclass calls."""

    def __init__(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, act_dtype):
        self.seen_counts = (self.n_local_heads, self.n_local_groups)


def _load_module(monkeypatch):
    """Load attn_intra_chip.py by path, without the TPU-only package imports."""
    stubs = {
        name: types.ModuleType(name)
        for name in (
            "vllm",
            "vllm.distributed",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.linear",
            "vllm_torchtpu",
            "vllm_torchtpu.logger",
            "vllm_torchtpu.models",
            "vllm_torchtpu.models.vllm",
            "vllm_torchtpu.models.vllm.deepseek_v4",
            "vllm_torchtpu.models.vllm.deepseek_v4.attention",
        )
    }
    attention = stubs["vllm_torchtpu.models.vllm.deepseek_v4.attention"]
    attention.VllmDeepseekV4MLAAttention = _FakeBaseAttention
    stubs["vllm.distributed"].parallel_state = types.SimpleNamespace()
    stubs["vllm.model_executor.layers.linear"].ColumnParallelLinear = object
    stubs["vllm.model_executor.layers.linear"].RowParallelLinear = object
    logger = types.SimpleNamespace(
        info=lambda *a: None, info_once=lambda *a: None, warning=lambda *a: None
    )
    stubs["vllm_torchtpu.logger"].init_logger = lambda name: logger
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("attn_intra_chip_under_test", _MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PairGroup:
    """Both threads exchange through a shared buffer behind a barrier."""

    # GroupCoordinator.world_size for one chip's group: its two cores,
    # independent of how many DP ranks the job has.
    world_size = 2

    def __init__(self, shared, rank):
        self.shared, self.rank_in_group = shared, rank

    def exchange(self, x):
        buf, barrier = self.shared
        buf[self.rank_in_group] = x
        barrier.wait()
        out = [buf[0], buf[1]]
        barrier.wait()
        return out

    def all_gather(self, x, dim):
        return torch.cat(self.exchange(x), dim)

    def reduce_scatter(self, x, dim):
        return sum(self.exchange(x)).chunk(2, dim)[self.rank_in_group]


def test_matches_unsharded_reference(monkeypatch):
    aic = _load_module(monkeypatch)
    torch.manual_seed(0)
    wq = torch.randn(Q_LORA_RANK, NUM_HEADS * HEAD_DIM)
    wa = torch.randn(NUM_GROUPS, (NUM_HEADS // NUM_GROUPS) * HEAD_DIM, O_LORA_RANK)
    wb = torch.randn(NUM_GROUPS * O_LORA_RANK, HIDDEN_SIZE)
    pos_scale = torch.randn(1000)

    def attn(q, pos):  # mixes all heads of a token, like real attention needs
        return q * pos_scale[pos][:, None, None] + q.sum(1, keepdim=True)

    def reference(qr, pos):
        o = attn((qr @ wq).view(-1, NUM_HEADS, HEAD_DIM), pos)
        o = (
            o.reshape(-1, NUM_GROUPS, (NUM_HEADS // NUM_GROUPS) * HEAD_DIM)
            * pos_scale[pos][:, None, None]
        )
        return (
            torch.einsum("tgk,gko->tgo", o, wa).reshape(-1, NUM_GROUPS * O_LORA_RANK)
            @ wb
        )

    cls = aic.VllmDeepseekV4MLAAttentionIntraChip

    shared = ({}, threading.Barrier(2))

    def make(rank):
        layer = object.__new__(cls)
        object.__setattr__(layer, "_chip_pair", _PairGroup(shared, rank))
        heads = slice(
            rank * NUM_HEADS // 2 * HEAD_DIM, (rank + 1) * NUM_HEADS // 2 * HEAD_DIM
        )
        groups = slice(rank * NUM_GROUPS // 2, (rank + 1) * NUM_GROUPS // 2)
        rows = slice(
            rank * NUM_GROUPS // 2 * O_LORA_RANK,
            (rank + 1) * NUM_GROUPS // 2 * O_LORA_RANK,
        )
        layer.wq_b = lambda x: x @ wq[:, heads]
        layer.mla_attn = types.SimpleNamespace(
            n_local_heads=NUM_HEADS, head_dim=HEAD_DIM
        )
        layer.n_local_heads, layer.head_dim = NUM_HEADS, HEAD_DIM
        layer.rotary_emb = types.SimpleNamespace(cos_sin_cache=None)
        layer.o_proj_op = lambda o, p, c, w, s: torch.einsum(
            "tgk,gko->tgo",
            o.reshape(-1, NUM_GROUPS // 2, (NUM_HEADS // NUM_GROUPS) * HEAD_DIM)
            * pos_scale[p][:, None, None],
            wa[groups],
        ).reshape(-1, NUM_GROUPS // 2 * O_LORA_RANK)
        layer.wo_b = lambda z: z @ wb[rows]
        layer.wo_a = types.SimpleNamespace(weight=None, weight_scale=None)
        return layer

    qr = [torch.randn(NUM_TOKENS, Q_LORA_RANK) for _ in range(2)]
    pos = [torch.randint(0, 1000, (NUM_TOKENS,)) for _ in range(2)]
    out = [None, None]
    errors = []

    def run(rank):
        try:
            layer = make(rank)
            # The base attention_impl runs attention between these two hooks.
            o = attn(layer._project_q(qr[rank]), pos[rank])
            out[rank] = layer._o_proj(o, pos[rank])
        except BaseException as error:  # noqa: BLE001 - surfaced below
            errors.append(error)
            shared[1].abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    for rank in range(2):
        torch.testing.assert_close(
            out[rank], reference(qr[rank], pos[rank]), rtol=1e-4, atol=1e-4
        )


class _PastLockstepGuard(Exception):
    pass


@pytest.mark.parametrize(
    "dp_size,expert_parallel,allowed",
    [(16, True, True), (1, True, False), (16, False, False), (1, False, False)],
    ids=["dp16-ep", "dp1", "no-ep", "dp1-no-ep"],
)
def test_requires_dp_lockstep(monkeypatch, dp_size, expert_parallel, allowed):
    """Without DP lockstep the chip pair can run different steps; refuse to build."""
    aic = _load_module(monkeypatch)
    aic.parallel_state.get_tensor_model_parallel_world_size = lambda: 1
    utils = types.ModuleType("vllm_torchtpu.utils")
    utils.get_dp_size = lambda parallel_config: parallel_config.data_parallel_size
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.utils", utils)

    def past_guard():
        raise _PastLockstepGuard

    monkeypatch.setattr(aic, "_chip_pair_group", past_guard)
    vllm_config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(
            data_parallel_size=dp_size, enable_expert_parallel=expert_parallel
        )
    )
    expected = (
        pytest.raises(_PastLockstepGuard)
        if allowed
        else pytest.raises(RuntimeError, match="DP lockstep")
    )
    with expected:
        aic.VllmDeepseekV4MLAAttentionIntraChip(vllm_config)


def test_process_weights_sees_half_counts(monkeypatch):
    aic = _load_module(monkeypatch)
    layer = object.__new__(aic.VllmDeepseekV4MLAAttentionIntraChip)
    object.__setattr__(layer, "_chip_pair", types.SimpleNamespace(world_size=2))
    layer.n_local_heads, layer.n_local_groups = NUM_HEADS, NUM_GROUPS
    layer.process_weights_after_loading(torch.bfloat16)
    assert layer.seen_counts == (NUM_HEADS // 2, NUM_GROUPS // 2)
    assert (layer.n_local_heads, layer.n_local_groups) == (NUM_HEADS, NUM_GROUPS)


def test_load_current_chiplet_slice_points_layer_and_params(monkeypatch):
    aic = _load_module(monkeypatch)
    layer = torch.nn.Linear(4, 4)
    layer.weight.tp_rank, layer.weight.tp_size = 0, 1  # like a vLLM block-fp8 param
    aic._load_current_chiplet_slice(layer, 1, 2)
    assert (layer.tp_rank, layer.tp_size) == (1, 2)
    assert (layer.weight.tp_rank, layer.weight.tp_size) == (1, 2)
    assert not hasattr(layer.bias, "tp_rank")
