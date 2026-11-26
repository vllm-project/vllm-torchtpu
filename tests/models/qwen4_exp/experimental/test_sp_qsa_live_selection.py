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
"""Sequence-parallel QSA live-selection summary tests for Qwen4Exp."""

from collections.abc import Mapping
from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config.compilation import CompilationMode

from tests.models.qwen4_exp.experimental.qwen38_test_utils import pin_plan
from tests.models.qwen4_exp.experimental.test_sequence_parallel import (
    PartialAttention,
    RowwiseMoE,
    SimulatedTP,
)
from vllm_torchtpu.models.qwen4_exp.experimental import model as qwen4_model
from vllm_torchtpu.models.qwen4_exp.experimental import sequence_parallel as sp
from vllm_torchtpu.models.qwen4_exp.experimental.hyperconnection import (
    HyperConnectionConfig,
)
from vllm_torchtpu.models.qwen4_exp.experimental.ops.rows_update_op import (
    ROWS_UPDATE_OP,
)

HIDDEN, HC, LOWRANK, VOCAB, TOKENS = 32, 2, 8, 64, 32


@pytest.fixture(autouse=True)
def cpu_default():
    with torch.device("cpu"):
        yield


def _make_sp(backbone: object) -> qwen4_model.Qwen4ExpBackboneSP:
    cfg = NS(compilation_config=NS(mode=CompilationMode.NONE))
    return qwen4_model.Qwen4ExpBackboneSP(vllm_config=cfg, backbone=backbone)


def _hc(*, use_combine: bool = True) -> qwen4_model.GatedResidual:
    return qwen4_model.GatedResidual(
        HyperConnectionConfig(
            hc_count=HC,
            hidden_size=HIDDEN,
            hc_lowrank=LOWRANK,
            params_dtype=torch.float32,
            hc_per_branch_norm=True,
        ),
        use_combine=use_combine,
    )


def _build_backbone(
    tp: Any,
    layer_types: list[str],
    selections: Mapping[int, torch.Tensor | None] | None = None,
    *,
    seed: int = 0,
) -> Any:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    backbone: Any = qwen4_model.Qwen4ExpBackbone.__new__(qwen4_model.Qwen4ExpBackbone)
    nn.Module.__init__(backbone)
    object.__setattr__(backbone, "do_not_compile", True)
    backbone.hc_count = HC
    backbone.host_ple_index = None
    backbone.emit_mtp_hidden = False
    backbone.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
    backbone.layers = nn.ModuleList()
    for idx, kind in enumerate(layer_types):
        layer: Any = qwen4_model.Qwen4ExpDecoderLayer.__new__(
            qwen4_model.Qwen4ExpDecoderLayer
        )
        nn.Module.__init__(layer)
        layer.ple = None
        layer.layer_type = kind
        attn_w = torch.randn((HIDDEN, HIDDEN), generator=gen) / HIDDEN**0.5
        moe_w = torch.randn((HIDDEN, HIDDEN), generator=gen) / HIDDEN**0.5
        sel = None if selections is None else selections.get(idx)
        sub = PartialAttention(tp, attn_w, kind, selection=sel)
        if kind == "linear_attention":
            layer.linear_attn = sub
        else:
            layer.self_attn = sub
        layer.mlp = RowwiseMoE(moe_w)
        layer.attn_hyper_connection = _hc()
        layer.mlp_hyper_connection = _hc()
        backbone.layers.append(layer)
    backbone.hyper_connection_mixer = _hc(use_combine=False)
    with torch.no_grad():
        for param in backbone.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.1)
    return backbone


def _step_inputs(tokens: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    input_ids = torch.arange(tokens, dtype=torch.int64) % VOCAB
    positions = torch.arange(tokens, dtype=torch.int64)
    valid = torch.arange(tokens) < (tokens - 4)
    kwargs: dict[str, object] = dict(
        query_start_loc=torch.tensor([0, 4, tokens], dtype=torch.int32),
        ngram_context=torch.zeros((2, 2), dtype=torch.int64),
        state_slots=torch.tensor([0, 1], dtype=torch.int32),
        context_lens=torch.tensor([516, 0], dtype=torch.int32),
        valid_tokens=valid,
        main_block_table=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        compressed_block_table=torch.tensor([[5, 6], [7, 8]], dtype=torch.int32),
    )
    return input_ids, positions, kwargs


_HYBRID_BLOCK = ["linear_attention"] * 3 + ["full_attention"]


@pytest.mark.parametrize(
    ("backbone", "expected"),
    [
        (NS(layers=[NS(layer_type=t) for t in _HYBRID_BLOCK * 12]), 12),
        (NS(layers=[NS(layer_type=t) for t in _HYBRID_BLOCK * 23]), 23),
        (NS(layers=[NS(layer_type="linear_attention") for _ in range(8)]), 0),
        (NS(layers=[]), 0),
    ],
)
def test_count_qsa_layers_across_hybrid_geometries(backbone, expected):
    assert qwen4_model._count_qsa_layers(backbone) == expected


@pytest.mark.parametrize(
    "transition",
    [
        "parent_to_empty",
        "solo_to_empty_and_dtype",
        "meta_dtype_stays_meta",
    ],
)
def test_qsa_sel_buf_registration_and_apply_lifecycle(transition: str):
    if transition == "parent_to_empty":
        tp1 = NS(rank=0, size=1)
        backbone = _build_backbone(
            tp1, ["linear_attention", "full_attention", "full_attention"]
        )
        model_sp = _make_sp(backbone)
        assert model_sp._qsa_sel_buf is not None
        assert model_sp._qsa_sel_buf.shape == (2, 1)
        assert model_sp._qsa_sel_buf.dtype == torch.int32
        assert "_qsa_sel_buf" not in model_sp.state_dict()
        assert list(model_sp.named_parameters()) == []

        # Default nn.Module._apply skips integer buffers on dtype casts while
        # still applying transforms to any floating buffers/children.
        orig_buf = model_sp._qsa_sel_buf
        orig_buf.fill_(7)
        model_sp.register_buffer(
            "_extra_fp32", torch.ones(1, dtype=torch.float32), persistent=False
        )
        wrapper = nn.Module()
        wrapper.model = backbone
        wrapper.model_sp = model_sp
        wrapper.to(dtype=torch.bfloat16)
        wrapper.bfloat16()
        wrapper.to(device="cpu")
        assert backbone.embed_tokens.weight.dtype == torch.bfloat16
        assert model_sp._extra_fp32.dtype == torch.bfloat16
        assert model_sp._qsa_sel_buf is orig_buf
        assert model_sp._qsa_sel_buf.dtype == torch.int32
        assert torch.equal(
            model_sp._qsa_sel_buf, torch.full((2, 1), 7, dtype=torch.int32)
        )

        gdn_sp = _make_sp(_build_backbone(tp1, ["linear_attention"]))
        assert gdn_sp._qsa_sel_buf is None
        gdn_sp.to(dtype=torch.bfloat16)
        assert gdn_sp._qsa_sel_buf is None

    with torch.device("meta"):
        parent: Any = nn.Module()
        meta_bb: Any = nn.Module()
        meta_bb.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        meta_bb.register_shared_gdn_verify_buffers = lambda _owner: False
        layers = [nn.Module(), nn.Module(), nn.Module()]
        object.__setattr__(layers[0], "layer_type", "linear_attention")
        object.__setattr__(layers[1], "layer_type", "full_attention")
        object.__setattr__(layers[2], "layer_type", "full_attention")
        meta_bb.layers = nn.ModuleList(layers)
        parent.model = meta_bb
        parent.model_sp = _make_sp(meta_bb)
        assert parent.model_sp._qsa_sel_buf is not None
        assert parent.model_sp._qsa_sel_buf.is_meta
        assert parent.model_sp._qsa_sel_buf.dtype == torch.int32

    if transition == "parent_to_empty":
        parent.to_empty(device="cpu")
    elif transition == "solo_to_empty_and_dtype":
        parent.model_sp.to_empty(device="cpu")
        parent.model_sp.to(dtype=torch.bfloat16)
    elif transition == "meta_dtype_stays_meta":
        parent.to(dtype=torch.bfloat16)
        assert parent.model_sp._qsa_sel_buf is not None
        assert parent.model_sp._qsa_sel_buf.is_meta
        assert parent.model_sp._qsa_sel_buf.dtype == torch.int32
        return

    buf: torch.Tensor | None = parent.model_sp._qsa_sel_buf
    assert buf is not None
    assert not buf.is_meta
    assert buf.device.type == "cpu"
    assert buf.dtype == torch.int32
    assert buf.shape == (2, 1)


def test_backbone_sp_updates_qsa_sel_buf_with_24_row_cap_and_matches_replicated(
    monkeypatch,
):
    tp2 = SimulatedTP(2)
    monkeypatch.setattr(sp, "shard", tp2.shard)
    monkeypatch.setattr(sp, "gather", tp2.gather)
    monkeypatch.setattr(sp, "reduce", tp2.reduce)
    monkeypatch.setattr(sp, "tp_world_size", lambda: tp2.size)
    pin_plan(
        monkeypatch,
        QWEN4_EXP_PREFILL_SP=True,
        QWEN4_EXP_PREFILL_SP_MOE=False,
        QWEN4_EXP_PREFILL_SP_SHARED_LOCAL=False,
        VLLM_TPU_QSA_INDEXER_TOKEN_SHARD=False,
        VLLM_TPU_GDN_HOIST_METADATA=False,
    )

    layer_types = [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
        "full_attention",
    ]

    def _make_sel(head_min: int) -> torch.Tensor:
        sel = torch.full((TOKENS, 6), head_min + 50, dtype=torch.int32)
        sel[3, 2] = head_min
        sel[24:, :] = -999
        return sel

    selections = {1: _make_sel(105), 3: _make_sel(215), 4: _make_sel(325)}
    input_ids, positions, kwargs = _step_inputs(TOKENS)
    valid = kwargs["valid_tokens"]
    assert isinstance(valid, torch.Tensor)

    def _run_replicated(_rank: int):
        bb = _build_backbone(tp2, layer_types, selections, seed=7)
        bb_sp = _make_sp(bb)
        out = bb(input_ids, positions, **kwargs)
        assert bb_sp._qsa_sel_buf is not None
        assert torch.equal(bb_sp._qsa_sel_buf, torch.zeros((3, 1), dtype=torch.int32))
        return out

    def _run_sp(_rank: int):
        bb = _build_backbone(tp2, layer_types, selections, seed=7)
        bb.emit_mtp_hidden = True
        bb_sp = _make_sp(bb)
        mixed_out, mtp_hidden = bb_sp(input_ids, positions, **kwargs)
        assert mtp_hidden.shape == (TOKENS, HC * HIDDEN)
        assert bb_sp._qsa_sel_buf is not None
        return mixed_out, bb_sp._qsa_sel_buf.clone()

    rep_outs = tp2.run(_run_replicated)
    sp_results = tp2.run(_run_sp)

    expected_sel = torch.tensor([[105], [215], [325]], dtype=torch.int32)
    for rank in range(2):
        sp_out, sel_buf = sp_results[rank]
        assert sel_buf.dtype == torch.int32
        assert torch.equal(sel_buf, expected_sel)
        torch.testing.assert_close(
            sp_out[valid], rep_outs[rank][valid], rtol=1e-5, atol=1e-5
        )


class _FakeTPUBuffer(torch.Tensor):
    """CPU-backed Tensor reporting device.type == 'tpu' to test update_rows_ routing."""

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        return torch.Tensor._make_subclass(cls, data, False)

    @property
    def device(self):  # type: ignore[override]
        return NS(type="tpu")


def test_backbone_sp_tpu_branch_dispatches_update_rows(monkeypatch):
    tp1 = SimulatedTP(1)
    monkeypatch.setattr(sp, "shard", tp1.shard)
    monkeypatch.setattr(sp, "gather", tp1.gather)
    monkeypatch.setattr(sp, "reduce", tp1.reduce)
    monkeypatch.setattr(sp, "tp_world_size", lambda: 1)
    pin_plan(
        monkeypatch,
        QWEN4_EXP_PREFILL_SP=True,
        QWEN4_EXP_PREFILL_SP_MOE=False,
        QWEN4_EXP_PREFILL_SP_SHARED_LOCAL=False,
        VLLM_TPU_QSA_INDEXER_TOKEN_SHARD=False,
        VLLM_TPU_GDN_HOIST_METADATA=False,
    )

    recorded: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _fake_update_rows(buf: torch.Tensor, rows: torch.Tensor) -> None:
        recorded.append((buf, rows.clone()))
        torch.Tensor.copy_(buf[: rows.shape[0]], rows)

    monkeypatch.setattr(qwen4_model, "update_rows_", _fake_update_rows)

    sel0 = torch.full((16, 4), 42, dtype=torch.int32)
    sel1 = torch.full((16, 4), 17, dtype=torch.int32)
    bb = _build_backbone(
        tp1, ["full_attention", "full_attention"], {0: sel0, 1: sel1}, seed=11
    )
    bb_sp = _make_sp(bb)
    tpu_buf = _FakeTPUBuffer(torch.zeros((2, 1), dtype=torch.int32))
    bb_sp.register_buffer("_qsa_sel_buf", tpu_buf, persistent=False)

    input_ids, positions, kwargs = _step_inputs(16)
    tp1.run(lambda _r: bb_sp(input_ids, positions, **kwargs))

    assert len(recorded) == 1
    written_buf, written_rows = recorded[0]
    assert written_buf is tpu_buf
    assert written_rows.dtype == torch.int32
    assert written_rows.is_contiguous()
    assert torch.equal(written_rows, torch.tensor([[42], [17]], dtype=torch.int32))

    # Real ROWS_UPDATE_OP fake registration accepts the int32 buffer and rows.
    with FakeTensorMode():
        fake_out = ROWS_UPDATE_OP(
            torch.empty(written_buf.shape, dtype=written_buf.dtype),
            torch.empty(written_rows.shape, dtype=written_rows.dtype),
        )
        assert fake_out.shape == written_buf.shape
        assert fake_out.dtype == torch.int32

    recorded.clear()
    gdn_bb = _build_backbone(tp1, ["linear_attention"], seed=13)
    gdn_sp = _make_sp(gdn_bb)
    assert gdn_sp._qsa_sel_buf is None
    tp1.run(lambda _r: gdn_sp(input_ids, positions, **kwargs))
    assert recorded == []


def test_backbone_sp_fullgraph_compiles_once_across_dynamic_token_counts(monkeypatch):
    monkeypatch.setattr(sp, "shard", lambda x: x)
    monkeypatch.setattr(sp, "gather", lambda x, _n: x)
    monkeypatch.setattr(sp, "reduce", lambda x, _s: x)
    monkeypatch.setattr(sp, "check_divisible", lambda _n: None)
    pin_plan(
        monkeypatch,
        QWEN4_EXP_PREFILL_SP=True,
        QWEN4_EXP_PREFILL_SP_MOE=False,
        QWEN4_EXP_PREFILL_SP_SHARED_LOCAL=False,
        VLLM_TPU_QSA_INDEXER_TOKEN_SHARD=False,
        VLLM_TPU_GDN_HOIST_METADATA=False,
    )

    sel0 = torch.arange(128 * 4, dtype=torch.int32).reshape(128, 4) + 12
    sel1 = torch.arange(128 * 4, dtype=torch.int32).reshape(128, 4) + 28
    sel0[24:, :] = -9999
    sel1[24:, :] = -9999
    bb = _build_backbone(
        NS(rank=0, size=1),
        ["linear_attention", "full_attention", "full_attention"],
        {1: sel0, 2: sel1},
        seed=19,
    )
    bb_sp = _make_sp(bb)

    torch._dynamo.reset()
    compile_calls = 0

    def _counting_backend(gm, example_inputs):
        del example_inputs
        nonlocal compile_calls
        compile_calls += 1
        return gm.forward

    compiled_sp = torch.compile(
        bb_sp.forward, backend=_counting_backend, fullgraph=True, dynamic=True
    )
    for num_tokens in (8, 16, 32, 64):
        input_ids, positions, kwargs = _step_inputs(num_tokens)
        out = compiled_sp(input_ids, positions, **kwargs)
        assert out.shape == (num_tokens, HIDDEN)
        assert bb_sp._qsa_sel_buf is not None
        assert bb_sp._qsa_sel_buf.dtype == torch.int32
        assert torch.equal(
            bb_sp._qsa_sel_buf,
            torch.tensor([[12], [28]], dtype=torch.int32),
        )
    assert compile_calls == 1
