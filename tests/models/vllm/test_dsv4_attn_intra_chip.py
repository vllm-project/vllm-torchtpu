# SPDX-License-Identifier: Apache-2.0
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

import torch

_MODULE = (
    pathlib.Path(__file__).parents[3]
    / "src/vllm_torchtpu/models/vllm/deepseek_v4/attn_intra_chip.py"
)

# tokens, heads, head_dim, q_lora, groups, o_lora, hidden
T, H, D, R, G, OL, HID = 3, 16, 8, 6, 4, 5, 7


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
    wq = torch.randn(R, H * D)
    wa = torch.randn(G, (H // G) * D, OL)
    wb = torch.randn(G * OL, HID)
    pos_scale = torch.randn(1000)

    def attn(q, pos):  # mixes all heads of a token, like real attention needs
        return q * pos_scale[pos][:, None, None] + q.sum(1, keepdim=True)

    def reference(qr, pos):
        o = attn((qr @ wq).view(-1, H, D), pos)
        o = o.reshape(-1, G, (H // G) * D) * pos_scale[pos][:, None, None]
        return torch.einsum("tgk,gko->tgo", o, wa).reshape(-1, G * OL) @ wb

    cls = aic.VllmDeepseekV4MLAAttentionIntraChip

    shared = ({}, threading.Barrier(2))

    def make(rank):
        layer = object.__new__(cls)
        object.__setattr__(layer, "_chip_pair", _PairGroup(shared, rank))
        heads = slice(rank * H // 2 * D, (rank + 1) * H // 2 * D)
        groups = slice(rank * G // 2, (rank + 1) * G // 2)
        rows = slice(rank * G // 2 * OL, (rank + 1) * G // 2 * OL)
        layer.wq_b = lambda x: x @ wq[:, heads]
        layer.mla_attn = types.SimpleNamespace(
            n_local_heads=H, head_dim=D, compressor=None
        )
        layer.n_local_heads, layer.head_dim = H, D
        layer.indexer = None
        layer.qnorm_rope = lambda q, p: q
        layer.kv_rope = lambda kv, p: kv
        layer.forward_mqa = lambda q, kv, p, _, topk_indices=None: attn(q, p)
        layer.rotary_emb = types.SimpleNamespace(cos_sin_cache=None)
        layer.o_proj_op = lambda o, p, c, w, s: torch.einsum(
            "tgk,gko->tgo",
            o.reshape(-1, G // 2, (H // G) * D) * pos_scale[p][:, None, None],
            wa[groups],
        ).reshape(-1, G // 2 * OL)
        layer.wo_b = lambda z: z @ wb[rows]
        layer.wo_a = types.SimpleNamespace(weight=None, weight_scale=None)
        return layer

    qr = [torch.randn(T, R) for _ in range(2)]
    pos = [torch.randint(0, 1000, (T,)) for _ in range(2)]
    out = [None, None]
    errors = []

    def run(rank):
        try:
            layer = make(rank)
            o = layer.attention_impl(None, qr[rank], None, None, pos[rank], None)
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


def test_process_weights_sees_half_counts(monkeypatch):
    aic = _load_module(monkeypatch)
    layer = object.__new__(aic.VllmDeepseekV4MLAAttentionIntraChip)
    object.__setattr__(layer, "_chip_pair", types.SimpleNamespace(world_size=2))
    layer.n_local_heads, layer.n_local_groups = H, G
    layer.process_weights_after_loading(torch.bfloat16)
    assert layer.seen_counts == (H // 2, G // 2)
    assert (layer.n_local_heads, layer.n_local_groups) == (H, G)


def test_use_my_slice_points_layer_and_params(monkeypatch):
    aic = _load_module(monkeypatch)
    layer = torch.nn.Linear(4, 4)
    layer.weight.tp_rank, layer.weight.tp_size = 0, 1  # like a vLLM block-fp8 param
    aic._use_my_slice(layer, 1, 2)
    assert (layer.tp_rank, layer.tp_size) == (1, 2)
    assert (layer.weight.tp_rank, layer.weight.tp_size) == (1, 2)
    assert not hasattr(layer.bias, "tp_rank")
