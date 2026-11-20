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


class _FakeJaxOp:
    """Stands in for a torch_tpu pallas.jax_op: records how it was built."""

    def __init__(self, name, fn, **kwargs):
        self.name, self.fn, self.kwargs = name, fn, kwargs

    def register_fake(self, fake):
        self.fake = fake


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
            "vllm_torchtpu.kernels",
            "vllm_torchtpu.kernels.deepseek_v4",
            "vllm_torchtpu.kernels.deepseek_v4.collective_ids",
            "vllm_torchtpu.kernels.deepseek_v4.o_projection",
            "vllm_torchtpu.kernels.deepseek_v4.q_send",
            "vllm_torchtpu.kernels.deepseek_v4.rope",
            "vllm_torchtpu.models.vllm.vllm_model_wrapper_context",
            "torch_tpu",
            "torch_tpu._internal",
        )
    }
    stubs["vllm_torchtpu.kernels.deepseek_v4.collective_ids"].Q_SEND_COLLECTIVE_ID = 4
    # `from package import module` reads the module off the (stubbed) package.
    kernels = stubs["vllm_torchtpu.kernels.deepseek_v4"]
    kernels.q_send = stubs["vllm_torchtpu.kernels.deepseek_v4.q_send"]
    stubs["vllm_torchtpu.kernels.deepseek_v4.rope"].qnorm_rope_pair = None
    o_projection = stubs["vllm_torchtpu.kernels.deepseek_v4.o_projection"]
    o_projection.O_FETCH_MAX_TOKENS = 512
    o_projection.fused_reverse_rope_wo_a_projection_o_fetch = None
    stubs["torch_tpu._internal"].pallas = types.SimpleNamespace(jax_op=_FakeJaxOp)
    stubs["vllm_torchtpu.models.vllm.vllm_model_wrapper_context"].get_pallas_op_mesh = (
        lambda: None
    )
    attention = stubs["vllm_torchtpu.models.vllm.deepseek_v4.attention"]
    attention.VllmDeepseekV4MLAAttention = _FakeBaseAttention
    attention._pallas_op_cache = {}
    attention._name_float = lambda value: str(value).replace(".", "p")
    stubs["vllm.distributed"].parallel_state = types.SimpleNamespace()
    stubs["vllm.model_executor.layers.linear"].ColumnParallelLinear = object
    stubs["vllm.model_executor.layers.linear"].RowParallelLinear = object
    logger = types.SimpleNamespace(
        info=lambda *a: None,
        info_once=lambda *a: None,
        warning=lambda *a: None,
        # Like the real one, which is lru_cached: the arguments must hash.
        warning_once=lambda *a: hash(a),
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


@pytest.mark.parametrize("use_fused", [False, True], ids=["exchange", "fused"])
def test_matches_unsharded_reference(monkeypatch, use_fused):
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
        group = _PairGroup(shared, rank)
        object.__setattr__(layer, "_chip_pair", group)
        layer._use_fused_o_fetch = use_fused
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
            n_local_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            rotary_emb=types.SimpleNamespace(cos_sin_cache=None),
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

        def fused_o_fetch(o, p, c, w, s):
            # Stand-in for the kernel: gets this chiplet's unswapped `o` (all
            # heads) and returns wo_a over both chiplets' tokens for its heads.
            o_both = group.all_gather(o, dim=0)[:, heads_half]
            return layer.o_proj_op(o_both, p, c, w, s)

        heads_half = slice(rank * NUM_HEADS // 2, (rank + 1) * NUM_HEADS // 2)
        object.__setattr__(layer, "_fused_o_fetch", fused_o_fetch)

        def q_send_rope(q_half, p, c):
            # Stand-in for _QSendRope: swap the blocks between the chiplets and
            # assemble all heads for the current chiplet's tokens; the reference
            # has no q-norm + RoPE.
            neighbor = aic._swap_with_neighbor_chiplet(
                group, q_half[rank], q_half[1 - rank]
            )
            pieces = [q_half[rank], neighbor][:: 1 if rank == 0 else -1]
            return torch.cat(pieces, dim=-1).view(-1, NUM_HEADS, HEAD_DIM)

        object.__setattr__(layer, "_q_send_rope", q_send_rope)
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
            o = attn(layer._project_q_rope(qr[rank], pos[rank]), pos[rank])
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
    monkeypatch.setattr(aic, "_QSendRope", lambda *args: None)
    layer = object.__new__(aic.VllmDeepseekV4MLAAttentionIntraChip)
    object.__setattr__(layer, "_chip_pair", types.SimpleNamespace(world_size=2))
    layer.n_local_heads, layer.n_local_groups = NUM_HEADS, NUM_GROUPS
    layer._use_fused_o_fetch = False
    layer.wo_b = types.SimpleNamespace(weight=torch.zeros(1))
    layer.mla_attn = types.SimpleNamespace(eps=1e-6, head_dim=HEAD_DIM)
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


def _fused_o_fetch_config(max_num_batched_tokens, compile_sizes):
    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens
        ),
        compilation_config=types.SimpleNamespace(compile_sizes=compile_sizes),
    )


@pytest.mark.parametrize(
    "max_num_batched_tokens,compile_sizes,supported",
    [
        (256, [16, 24, 128, 256], True),
        (264, [16, 264], False),
        (256, [16, 256, 512], False),
        (256, [16, 20, 256], False),
    ],
    ids=["fused", "too-many-tokens", "size-above-budget", "unaligned-bucket"],
)
def test_fused_o_fetch_falls_back_when_unsupported(
    monkeypatch, max_num_batched_tokens, compile_sizes, supported
):
    """Configs the kernel cannot run take the exchange path instead."""
    aic = _load_module(monkeypatch)
    warnings = []
    monkeypatch.setattr(
        aic.logger, "warning_once", lambda *a: warnings.append(a) or hash(a)
    )
    config = _fused_o_fetch_config(max_num_batched_tokens, compile_sizes)
    assert aic._check_fused_o_fetch_supported(config) is supported
    # An automatic fallback warns once (R-5.3).
    assert len(warnings) == (0 if supported else 1)


def test_fused_o_fetch_finds_neighbor_and_chiplet_index(monkeypatch):
    """The kernel's two scalars: the neighbor's index on the EP mesh (from the
    chip topology) and this chiplet's index in its pair."""
    aic = _load_module(monkeypatch)
    # Mesh order differs from device id order, as under PCP/DCP.
    mesh_ids = [2, 3, 1, 0]
    mesh = types.SimpleNamespace(
        devices=types.SimpleNamespace(
            flat=[types.SimpleNamespace(id=i) for i in mesh_ids]
        ),
        axis_names=("ep",),
    )
    modules = {
        "vllm_torchtpu.distributed": types.ModuleType("vllm_torchtpu.distributed"),
        "vllm_torchtpu.distributed.chip_topology": types.ModuleType("chip_topology"),
        "vllm_torchtpu.distributed.ep_mesh": types.ModuleType("ep_mesh"),
        "vllm_torchtpu.distributed.mesh_utils": types.ModuleType("mesh_utils"),
    }
    modules["vllm_torchtpu.distributed.chip_topology"].get_chip_topology = (
        lambda: types.SimpleNamespace(chips=[(0, 1), (2, 3)])
    )
    modules["vllm_torchtpu.distributed.ep_mesh"].build_ep_mesh = lambda: mesh
    modules["vllm_torchtpu.distributed.mesh_utils"]._get_tpu_global_device_id = (
        lambda: 1
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(aic, "fused_reverse_rope_wo_a_projection_o_fetch", lambda: None)
    op = types.SimpleNamespace(register_fake=lambda fn: None)
    monkeypatch.setattr(aic, "pallas", types.SimpleNamespace(jax_op=lambda *a, **k: op))
    group = types.SimpleNamespace(rank_in_group=1)
    fetch = aic._FusedOFetch(group, "cpu", HEAD_DIM)
    # Device 1's neighbor is device 0, at index 3 of the mesh.
    assert fetch.neighbor_mesh_index.tolist() == [3]
    assert fetch.chiplet_index.tolist() == [1]


def _half_count_layer(aic):
    layer = object.__new__(aic.VllmDeepseekV4MLAAttentionIntraChip)
    object.__setattr__(
        layer, "_chip_pair", types.SimpleNamespace(world_size=2, rank_in_group=0)
    )
    layer.n_local_heads, layer.n_local_groups = NUM_HEADS, NUM_GROUPS
    layer._use_fused_o_fetch = False
    return layer


def test_q_send_rope_built_at_weight_load(monkeypatch):
    aic = _load_module(monkeypatch)
    monkeypatch.setattr(aic, "_QSendRope", lambda *args: args)
    layer = _half_count_layer(aic)
    layer.wo_b = types.SimpleNamespace(weight=torch.zeros(1))
    layer.mla_attn = types.SimpleNamespace(eps=1e-6, head_dim=HEAD_DIM)
    layer.process_weights_after_loading(torch.bfloat16)
    built = layer.__dict__["_q_send_rope"]
    assert built[1:] == (torch.zeros(1).device, HEAD_DIM, 1e-6)


def _q_send_rope(aic, monkeypatch, chiplet_index):
    """A real _QSendRope on stand-in ops; the neighbor is at mesh index 3."""
    mesh = types.SimpleNamespace(axis_names=("ep",))
    monkeypatch.setattr(aic, "_chip_pair_mesh", lambda: (mesh, 1, 3))
    group = types.SimpleNamespace(rank_in_group=chiplet_index)
    return aic._QSendRope(group, "cpu", HEAD_DIM, 1e-6)


def test_q_send_rope_ops_and_their_fakes(monkeypatch):
    aic = _load_module(monkeypatch)
    q = _q_send_rope(aic, monkeypatch, chiplet_index=1)
    # The kernels' device data: (neighbor mesh index, chiplet index).
    assert q.neighbor_mesh_index.tolist() == [3]
    assert q.chiplet_index.tolist() == [1]
    assert q._rope.fn.keywords == {"eps": 1e-6}
    half = NUM_HEADS // 2
    q_half = torch.empty(2, NUM_TOKENS, half * HEAD_DIM)
    qh, neighbor = q._send.fake(q_half, q.neighbor_mesh_index, q.chiplet_index)
    assert qh.shape == (2, NUM_TOKENS, half, HEAD_DIM)
    assert neighbor.shape == (NUM_TOKENS, half, HEAD_DIM)
    current = qh.view(2 * NUM_TOKENS, half, HEAD_DIM)
    neighbor = neighbor.view(NUM_TOKENS, half, HEAD_DIM)
    assert q._rope.fake(current, neighbor, None, None).shape == (
        NUM_TOKENS,
        NUM_HEADS,
        HEAD_DIM,
    )


def test_q_send_rope_ropes_both_blocks_in_place(monkeypatch):
    aic = _load_module(monkeypatch)
    q = _q_send_rope(aic, monkeypatch, chiplet_index=0)
    half = NUM_HEADS // 2
    sends, ropes = [], []

    def send(sent, neighbor_mesh_index, chiplet_index):
        sends.append((sent, neighbor_mesh_index, chiplet_index))
        return sent.view(2, NUM_TOKENS, half, HEAD_DIM), torch.ones(
            NUM_TOKENS, half, HEAD_DIM
        )

    q._send, q._rope = send, lambda *a: ropes.append(a) or "q"
    q_half = torch.randn(2, NUM_TOKENS, half * HEAD_DIM)
    pos, cache = torch.arange(NUM_TOKENS), torch.randn(64, HEAD_DIM)
    assert q(q_half, pos, cache) == "q"
    ((sent, neighbor_mesh_index, chiplet_index),) = sends
    assert sent.shape == (2, NUM_TOKENS, half * HEAD_DIM)
    assert neighbor_mesh_index is q.neighbor_mesh_index
    assert chiplet_index is q.chiplet_index
    ((current, neighbor, positions, cos_sin_cache, rope_chiplet_index),) = ropes
    # Both blocks go to q-norm + RoPE as [rows, half the heads, head_dim], in place,
    # with the chiplet index as data (one op for both chiplets).
    assert current.shape == (2 * NUM_TOKENS, half, HEAD_DIM)
    assert neighbor.shape == (NUM_TOKENS, half, HEAD_DIM)
    assert positions is pos and cos_sin_cache is cache
    assert rope_chiplet_index is q.chiplet_index


def test_project_q_rope_sends_both_chiplets_rows(monkeypatch):
    aic = _load_module(monkeypatch)
    layer = _half_count_layer(aic)
    width = NUM_HEADS // 2 * HEAD_DIM
    w = torch.randn(Q_LORA_RANK, width)
    layer.wq_b = lambda x: x @ w
    neighbor_qr = torch.randn(NUM_TOKENS, Q_LORA_RANK)
    object.__setattr__(
        layer,
        "_chip_pair",
        types.SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            all_gather=lambda x, dim: torch.cat([x, neighbor_qr], dim),
        ),
    )
    cache = torch.randn(64, HEAD_DIM)
    layer.mla_attn = types.SimpleNamespace(
        head_dim=HEAD_DIM,
        n_local_heads=NUM_HEADS,
        rotary_emb=types.SimpleNamespace(cos_sin_cache=cache),
    )
    calls = []
    layer.__dict__["_q_send_rope"] = lambda *a: calls.append(a) or "q"
    qr, pos = torch.randn(NUM_TOKENS, Q_LORA_RANK), torch.arange(NUM_TOKENS)
    assert layer._project_q_rope(qr, pos) == "q"
    ((q_half, positions, cos_sin_cache),) = calls
    # [2, T, W]: chiplet 0's tokens, then chiplet 1's; the current chiplet's heads.
    assert q_half.shape == (2, NUM_TOKENS, width)
    torch.testing.assert_close(q_half[0], qr @ w)
    torch.testing.assert_close(q_half[1], neighbor_qr @ w)
    assert positions is pos and cache is cos_sin_cache
