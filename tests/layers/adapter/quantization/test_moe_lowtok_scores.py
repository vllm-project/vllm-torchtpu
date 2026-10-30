# SPDX-License-Identifier: Apache-2.0
"""Load-time selection and forward wiring of the small-batch MoE scores entry."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import pytest
import torch

from vllm_torchtpu.layers.adapter import fused_moe_ep
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization import moe_lowtok_scores as scores

pytestmark = pytest.mark.cpu_test


def make_layer(experts=64, hidden=512, inter=256, use_ep=True):
    def meta(*shape, dtype=torch.float8_e4m3fn):
        return torch.empty(shape, dtype=dtype, device="meta")

    return SimpleNamespace(
        layer_name="model.layers.0.mlp.experts",
        hidden_size=hidden,
        global_num_experts=512,
        renormalize=True,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        apply_router_weight_on_input=False,
        moe_config=SimpleNamespace(
            experts_per_token=10,
            ep_size=8,
            in_dtype=torch.bfloat16,
            moe_parallel_config=SimpleNamespace(use_ep=use_ep),
        ),
        w13_weight=meta(experts, hidden, 2 * inter),
        w2_weight=meta(experts, inter, hidden),
        w13_weight_scale_inv=meta(experts, 1, 1, 2 * inter, dtype=torch.float32),
        w2_weight_scale_inv=meta(experts, 1, 1, hidden, dtype=torch.float32),
        _experts_start=torch.tensor(192, dtype=torch.int32),
    )


def method(owner=None):
    return SimpleNamespace(_use_adaptive_fused_moe=owner == "adaptive", owner=owner)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("TPU_MOE_LOWTOK", "1")
    monkeypatch.setattr(
        scores, "enable_pipelined_collective_and_compute", lambda: False
    )
    monkeypatch.setattr(scores, "_LAYERS", [])
    monkeypatch.setattr(
        fused_moe_ep, "fused_moe_ep_supported", lambda m: m.owner == "v2"
    )
    op = Mock(name="op")
    monkeypatch.setattr(scores, "_build_op", Mock(return_value=op))
    return op


def test_disabled_builds_and_logs_nothing(monkeypatch):
    monkeypatch.setenv("TPU_MOE_LOWTOK", "0")
    monkeypatch.setattr(scores, "_LAYERS", [])
    assert scores.prebuild(method(), make_layer(use_ep=False), "gelu") is None
    scores.log_summary()


def test_plain_softmax_router_moves_into_the_kernel(enabled):
    assert scores.prebuild(method(), make_layer(), "silu") is enabled
    scores._build_op.assert_called_once_with(10)


@pytest.mark.parametrize(
    "layer, activation, match",
    [
        (make_layer(use_ep=False), "silu", "expert parallelism"),
        (make_layer(experts=129), "silu", "envelope"),
        (make_layer(inter=128), "silu", "envelope"),
        (make_layer(), "gelu", "envelope"),
    ],
)
def test_layer_the_kernel_cannot_serve_is_refused_at_the_summary(
    enabled, layer, activation, match
):
    assert scores.prebuild(method(), layer, activation) is None
    with pytest.raises(NotImplementedError, match=f"TPU_MOE_LOWTOK.*{match}"):
        scores.log_summary()


@pytest.mark.parametrize("owner", ["adaptive", "v2"])
def test_layer_a_fused_path_owns_is_declined_not_refused(enabled, owner):
    scores.prebuild(method(owner), make_layer(use_ep=False), "silu")
    scores.log_summary()


@pytest.mark.parametrize(
    "option, value",
    [
        ("custom_routing_function", lambda **kw: None),
        ("scoring_func", "sigmoid"),
        ("renormalize", False),
        ("e_score_correction_bias", torch.zeros(512)),
        ("use_grouped_topk", True),
        ("hash_indices_table", torch.zeros(8, 10)),
        ("routed_scaling_factor", 2.5),
        ("apply_router_weight_on_input", True),
    ],
)
def test_routing_the_kernel_does_not_reproduce_keeps_the_ids_entry(
    enabled, option, value
):
    layer = make_layer()
    setattr(layer, option, value)
    if option == "use_grouped_topk":
        layer.num_expert_group, layer.topk_group = 8, 4
    assert scores.prebuild(method(), layer, "silu") is None


def test_chunk_pipelining_keeps_the_ids_entry(enabled, monkeypatch):
    monkeypatch.setattr(scores, "enable_pipelined_collective_and_compute", lambda: True)
    assert scores.prebuild(method(), make_layer(), "silu") is None


def test_a_layer_outside_the_backbone_graph_arms_the_scores_entry(enabled, monkeypatch):
    """The row cap runs inside the JAX op per compiled shape, so layers outside
    the "backbone" tag (such as the MTP draft under "eagle_head") also arm the
    scores entry without forking the torch trace."""
    from vllm.compilation import backends

    info = Mock()
    monkeypatch.setattr(scores.logger, "info", info)
    with backends.set_model_tag("eagle_head"):
        assert scores.prebuild(method(), make_layer(), "silu") is enabled
    assert scores.prebuild(method(), make_layer(), "silu") is enabled
    scores.log_summary()
    line = info.call_args.args[0] % info.call_args.args[1:]
    assert line.startswith("moe_lowtok: 2/2 MoE layers armed (scores 2, ids 0)")


def test_summary_counts_entries_and_declines(enabled, monkeypatch):
    info = Mock()
    monkeypatch.setattr(scores.logger, "info", info)
    for _ in range(3):
        scores.prebuild(method(), make_layer(), "silu")
    biased = make_layer()
    biased.e_score_correction_bias = torch.zeros(512)
    scores.prebuild(method(), biased, "silu")
    scores.prebuild(method("v2"), make_layer(), "silu")
    scores.log_summary()
    line = info.call_args.args[0] % info.call_args.args[1:]
    assert line.startswith("moe_lowtok: 4/5 MoE layers armed (scores 3, ids 1)")
    assert "1 x the fused EP kernel owns the layer" in line
    assert "1 x scores declined, ids armed: e_score_correction_bias" in line
    assert scores._LAYERS == []


@pytest.mark.parametrize("rows", [1, 128, 129, 256])
def test_serves_checks_only_op_presence(rows):
    """The row-count fork lives inside the JAX op so torch tracing stays uniform."""
    x = torch.empty(rows, 8)
    assert scores.serves(Mock(), x)
    assert not scores.serves(None, x)


@pytest.mark.parametrize("has_copies", [False, True])
@pytest.mark.parametrize("rows, lowtok", [(64, True), (65, False)])
def test_in_op_row_cap_dispatches_lowtok_or_fused_moe_func(
    monkeypatch, rows, lowtok, has_copies
):
    monkeypatch.setenv("TPU_MOE_LOWTOK_MAX_TOKENS", "64")
    monkeypatch.setenv("TPU_MOE_ROUTER_TOPK", "sort")
    monkeypatch.setattr(scores, "moe_lowtok", Mock(return_value="lowtok"))
    monkeypatch.setattr(scores, "fused_moe_func", Mock(return_value="gmm"))
    monkeypatch.setattr(scores.expert_placement, "has_copies", lambda: has_copies)
    monkeypatch.setattr(scores.expert_placement, "owned", lambda: 64)
    x = jnp.zeros((rows, 8), dtype=jnp.bfloat16)
    s = jnp.ones((rows, 16), dtype=jnp.float32) / 16.0
    w13 = jnp.zeros((4, 8, 16), dtype=jnp.float8_e4m3fn)
    w2 = jnp.zeros((4, 8, 8), dtype=jnp.float8_e4m3fn)
    s13 = jnp.ones((4, 1, 1, 16), dtype=jnp.float32)
    s2 = jnp.ones((4, 1, 1, 8), dtype=jnp.float32)
    start = jnp.int32(0)
    out = scores._moe_scores_fn(x, s, w13, w2, s13, s2, start, topk=2)
    assert out == ("lowtok" if lowtok else "gmm")
    if lowtok:
        scores.moe_lowtok.assert_called_once()
        scores.fused_moe_func.assert_not_called()
    else:
        scores.moe_lowtok.assert_not_called()
        scores.fused_moe_func.assert_called_once()
        assert scores.fused_moe_func.call_args.kwargs["num_primary_experts"] == (
            64 if has_copies else None
        )


@pytest.mark.parametrize("skip_padded", [False, True])
def test_apply_passes_softmax_scores_and_the_grouped_operand_order(
    monkeypatch, skip_padded
):
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1" if skip_padded else "0")
    padded = torch.tensor([False, True, False])
    monkeypatch.setattr(
        scores.token_padding,
        "padding_mask_for",
        lambda rows, is_local_tensor=False: padded,
    )
    layer = make_layer()
    op = Mock(return_value="out")
    x = torch.randn(3, 512, dtype=torch.bfloat16)
    logits = torch.randn(3, 512, dtype=torch.bfloat16)
    assert scores.apply(op, layer, x, logits) == "out"
    args = op.call_args.args
    want = logits.clone()
    if skip_padded:
        want[1] = float("-inf")
    assert args[0] is x
    torch.testing.assert_close(args[1], want)
    assert args[2:] == (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        layer._experts_start,
    )


@pytest.mark.parametrize("armed", [True, False])
def test_fp8_forward_takes_the_scores_entry_when_armed(monkeypatch, armed):
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0")
    monkeypatch.setattr(fp8, "enable_pipelined_collective_and_compute", lambda: False)
    monkeypatch.setattr(scores, "apply", Mock(return_value="lowtok"))
    monkeypatch.setattr(fp8.moe_routing, "route", Mock(return_value=(None, None)))
    monkeypatch.setattr(fp8, "fused_moe_gmm", Mock(return_value="gmm"))
    method = SimpleNamespace(
        _tpu_activation_str="silu",
        _use_adaptive_fused_moe=False,
        _moe_lowtok_scores_op=Mock() if armed else None,
    )
    layer = make_layer()
    layer.w13_bias = layer.w2_bias = None
    x = torch.empty(129, 512)
    out = fp8.VllmFp8MoEMethodTPU.apply_monolithic(
        method, layer, x, torch.empty(129, 512)
    )
    assert out == ("lowtok" if armed else "gmm")


def test_op_is_built_once_per_topk_and_fakes_the_row_shape():
    from torch._subclasses.fake_tensor import FakeTensorMode

    op = scores._build_op(10)
    assert scores._build_op(10) is op
    layer = make_layer()
    with FakeTensorMode():
        x = torch.empty(4, 512, dtype=torch.bfloat16)
        w = [
            torch.empty(t.shape, dtype=t.dtype)
            for t in (
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
            )
        ]
        out = op(x, torch.empty(4, 512), *w, torch.empty((), dtype=torch.int32))
    assert tuple(out.shape) == (4, 512) and out.dtype == torch.bfloat16


def test_copy_enabled_placement_arms_60_backbone_plus_1_mtp_draft_layer(
    enabled, monkeypatch, tmp_path
):
    """With a copy-enabled placement table (e.g. 68 slots = 64 owned + 4 copies),
    the 60 placed backbone layers arm the copy-aware scores op while the unplaced
    61st MTP draft layer (mtp.layers.0.mlp.experts, 64 slots) arms the 0-copy
    scores op, giving 61/61 scores armed."""
    import json

    table = {
        "shards": 8,
        "experts": 512,
        "owned_per_core": 64,
        "extra_slots": 4,
        "slots_per_core": 68,
        "copies": True,
        "layers": {
            str(lidx): {
                "cores": [
                    list(range(r * 64, (r + 1) * 64))
                    + [((r + 1) % 8) * 64 + j for j in range(4)]
                    for r in range(8)
                ],
                "copies": {
                    str(((r + 1) % 8) * 64 + j): r for r in range(8) for j in range(4)
                },
            }
            for lidx in range(60)
        },
    }
    pfile = tmp_path / "opt_copies32.json"
    pfile.write_text(json.dumps(table))
    monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(pfile))

    copies_op = Mock(name="copies_op")
    monkeypatch.setattr(scores, "_build_copies_op", Mock(return_value=copies_op))
    info = Mock()
    monkeypatch.setattr(scores.logger, "info", info)

    for lidx in range(60):
        layer = make_layer(experts=68)
        layer.layer_name = f"language_model.model.layers.{lidx}.mlp.experts"
        layer.moe_config.ep_rank = 3
        assert scores.prebuild(method(), layer, "silu") is copies_op
        assert layer._lowtok_has_copies is True
        assert layer._lowtok_owned == 64
        assert layer._lowtok_prim.shape == (512,)
        assert layer._lowtok_copy.shape == (512,)
        assert layer._lowtok_hot1h.shape == (512, 128)
        assert layer._lowtok_tables[0].item() == 3 * 64

    mtp_layer = make_layer(experts=64)
    mtp_layer.layer_name = "mtp.layers.0.mlp.experts"
    assert scores.prebuild(method(), mtp_layer, "silu") is enabled
    assert mtp_layer._lowtok_has_copies is False

    scores.log_summary()
    line = info.call_args.args[0] % info.call_args.args[1:]
    assert line.startswith("moe_lowtok: 61/61 MoE layers armed (scores 61, ids 0)")


@pytest.mark.parametrize("rows, lowtok", [(64, True), (65, False)])
def test_in_op_row_cap_dispatches_copy_aware_lowtok_or_fused_moe_func(
    monkeypatch, rows, lowtok
):
    monkeypatch.setenv("TPU_MOE_LOWTOK_MAX_TOKENS", "64")
    monkeypatch.setenv("TPU_MOE_ROUTER_TOPK", "sort")
    monkeypatch.setattr(scores, "moe_lowtok", Mock(return_value="lowtok_copies"))
    monkeypatch.setattr(scores, "fused_moe_func", Mock(return_value="gmm_copies"))
    x = jnp.zeros((rows, 8), dtype=jnp.bfloat16)
    s = jnp.ones((rows, 512), dtype=jnp.float32) / 512.0
    w13 = jnp.zeros((68, 8, 16), dtype=jnp.float8_e4m3fn)
    w2 = jnp.zeros((68, 8, 8), dtype=jnp.float8_e4m3fn)
    s13 = jnp.ones((68, 1, 1, 16), dtype=jnp.float32)
    s2 = jnp.ones((68, 1, 1, 8), dtype=jnp.float32)
    start = jnp.int32(0)
    tables = jnp.zeros((1024,), dtype=jnp.int32)
    prim = jnp.zeros((512,), dtype=jnp.int32)
    copy = jnp.full((512,), -1, dtype=jnp.int32)
    hot1h = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    out = scores._moe_scores_fn(
        x,
        s,
        w13,
        w2,
        s13,
        s2,
        start,
        tables,
        prim,
        copy,
        hot1h,
        topk=10,
        ranks=8,
        owned=64,
        has_copies=True,
    )
    assert out == ("lowtok_copies" if lowtok else "gmm_copies")
    if lowtok:
        scores.moe_lowtok.assert_called_once()
        assert scores.moe_lowtok.call_args.kwargs == {
            "topk": 10,
            "ranks": 8,
            "owned": 64,
            "has_copies": True,
            "from_logits": True,
        }
        scores.fused_moe_func.assert_not_called()
    else:
        scores.moe_lowtok.assert_not_called()
        scores.fused_moe_func.assert_called_once()
        assert scores.fused_moe_func.call_args.kwargs["num_primary_experts"] == 64


def test_apply_passes_holder_tables_when_layer_has_copies(monkeypatch):
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0")
    layer = make_layer(experts=68)
    layer._lowtok_has_copies = True
    layer._lowtok_tables = torch.zeros(1024, dtype=torch.int32)
    layer._lowtok_prim = torch.zeros(512, dtype=torch.int32)
    layer._lowtok_copy = torch.full((512,), -1, dtype=torch.int32)
    layer._lowtok_hot1h = torch.zeros((512, 128), dtype=torch.bfloat16)
    op = Mock(return_value="out_copies")
    x = torch.randn(3, 512, dtype=torch.bfloat16)
    logits = torch.randn(3, 512, dtype=torch.bfloat16)
    assert scores.apply(op, layer, x, logits) == "out_copies"
    args = op.call_args.args
    assert args[0] is x
    torch.testing.assert_close(args[1], logits)
    assert args[2:] == (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        layer._experts_start,
        layer._lowtok_tables,
        layer._lowtok_prim,
        layer._lowtok_copy,
        layer._lowtok_hot1h,
    )
