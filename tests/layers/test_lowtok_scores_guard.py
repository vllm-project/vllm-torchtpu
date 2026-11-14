"""The few-token kernel's scores entry takes the served layer: the FP8 checkpoint keeps its MoE scales under
``w13_weight_scale`` / ``w2_weight_scale`` (only block-quantised ones use ``_inv``).
Both scale names must engage the kernel. Runs on the CPU under fake tensors."""
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def _layer(scale_name):
    tables = {
        k: torch.zeros(512, dtype=torch.int32)
        for k in ("_lowtok_gid", "_lowtok_prim", "_lowtok_copy")
    }
    tables.update({
        k: torch.full((128, ), -1, dtype=torch.int32)
        for k in ("_lowtok_hot", "_lowtok_hprim", "_lowtok_hcopy")
    })
    tables["_lowtok_hot1h"] = torch.zeros(512, 128, dtype=torch.bfloat16)
    tables["_lowtok_slot"] = torch.full((512, ), -1, dtype=torch.int32)
    tables["_lowtok_tables"] = torch.full((1024, ), -1, dtype=torch.int32)
    layer = SimpleNamespace(
        renormalize=True,
        scoring_func="softmax",
        custom_routing_function=None,
        w13_bias=None,
        w2_bias=None,
        _experts_start=torch.tensor(192, dtype=torch.int32),
        _lowtok_owned=64,
        moe_config=SimpleNamespace(
            experts_per_token=10,
            ep_size=8,
            in_dtype=torch.bfloat16,
            moe_parallel_config=SimpleNamespace(use_ep=True)),
        hidden_size=4096,
        **tables)
    layer.w13_weight = torch.empty(64, 4096, 2048, dtype=torch.float8_e4m3fn)
    layer.w2_weight = torch.empty(64, 1024, 4096, dtype=torch.float8_e4m3fn)
    setattr(layer, f"w13_{scale_name}", torch.empty(64, 1, 1, 2048))
    setattr(layer, f"w2_{scale_name}", torch.empty(64, 1, 1, 4096))
    return layer


@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
def test_scores_entry_takes_the_served_layer(monkeypatch, scale_name):
    import vllm_torchtpu.layers.vllm.moe_lowtok_scores_op as scores_op
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.vllm.quantization import fp8
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_FROM_SCORES", 1, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_MAX_TOKENS", 32, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", 1,
                        raising=False)  # supported per-channel scale layout
    scores_op.prebuild(10, 8, 64)
    with FakeTensorMode():
        layer = _layer(scale_name)
        x = torch.empty(4, 4096, dtype=torch.bfloat16)
        logits = torch.empty(4, 512, dtype=torch.bfloat16)
        out = fp8._lowtok_scores_output(layer, x, logits, "silu")
        assert out is not None and tuple(out.shape) == (4, 4096)
        ids = torch.zeros(4, 10, dtype=torch.int32)
        assert tuple(fp8._lowtok_lift_tables(layer, ids).shape) == (
            4, 10)  # the grouped side lifts the tables too
        big = torch.empty(64, 4096, dtype=torch.bfloat16)
        assert fp8._lowtok_scores_output(
            layer, big, torch.empty(64, 512, dtype=torch.bfloat16),
            "silu") is None
    assert fp8._lowtok_scores_static_refusal(layer, "silu") is None
    assert fp8._lowtok_scores_static_refusal(layer,
                                             "gelu") == "activation gelu"


def _capture(fn, *args):
    """The graph dynamo hands the compile flow for fn, traced with the token count dynamic as the served forward is;
    the captured graph is not run (the kernel has no CPU path)."""
    seen = []

    def backend(gm, example_inputs):
        seen.append(gm)
        return lambda *a: [torch.zeros(0)]

    torch._dynamo.reset()
    try:
        torch.compile(fn, backend=backend, dynamic=True)(*args)
    except Exception:
        pass
    assert seen, "dynamo captured no graph"
    return seen[0]


def _lowtok_inputs(gm):
    """The few-token tables among the placeholders, as the compile flow's hand-over check lists them."""
    return {
        n.name
        for n in gm.graph.nodes if n.op == "placeholder" and (
            "_lowtok_" in n.name or "experts_start" in n.name)
    }


def test_both_sides_of_the_row_count_fork_lift_the_same_tables(monkeypatch):
    """Bucket handoff requires identical lifted inputs in both traces.

    The grouped path must read the same tables consumed by the few-token op.
    """
    import vllm_torchtpu.layers.vllm.moe_lowtok_scores_op as scores_op
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.vllm.quantization import fp8
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_FROM_SCORES", 1, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_MAX_TOKENS", 32, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", 1, raising=False)
    scores_op.prebuild(10, 8, 64)
    layer = _layer(
        "weight_scale")  # real (empty) tensors: dynamo takes no fake inputs
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    logits = torch.zeros(4, 512, dtype=torch.bfloat16)
    ids = torch.zeros(4, 10, dtype=torch.int32)
    below = _capture(
        lambda x, logits: fp8._lowtok_scores_output(layer, x, logits, "silu"),
        x, logits)
    # the grouped route: the retain-read plus its own read of the experts_start buffer
    above = _capture(
        lambda ids: fp8._lowtok_lift_tables(layer, ids) + layer._experts_start
        * 0, ids)
    assert len(_lowtok_inputs(below)) == 5, _lowtok_inputs(below)
    assert _lowtok_inputs(above) == _lowtok_inputs(below), (
        _lowtok_inputs(above), _lowtok_inputs(below))


def test_the_traced_guard_has_no_logging():
    """Dynamo refuses logging calls inside the compiled forward."""
    import inspect

    from vllm_torchtpu.layers.vllm.quantization import fp8
    for fn in (fp8._lowtok_scores_output, fp8._lowtok_scores_static_refusal):
        assert "logger." not in inspect.getsource(
            fn) and "print(" not in inspect.getsource(fn)


def test_scores_entry_defers_to_routing_simulation(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.vllm import moe_routing
    from vllm_torchtpu.layers.vllm.quantization import fp8

    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_FROM_SCORES", 1)
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_MAX_TOKENS", 32)
    with FakeTensorMode():
        layer = _layer("weight_scale")
        monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)
        assert fp8._lowtok_scores_static_refusal(layer, "silu") is None
        monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", object())
        assert fp8._lowtok_scores_static_refusal(
            layer, "silu") == "routing simulation"
        x = torch.empty(4, 4096, dtype=torch.bfloat16)
        logits = torch.empty(4, 512, dtype=torch.bfloat16)
        assert fp8._lowtok_scores_output(layer, x, logits, "silu") is None


@pytest.mark.parametrize("option, value, refusal", [
    ("e_score_correction_bias", torch.zeros(512), "expert-selection bias"),
    ("routed_scaling_factor", 2.5, "routed scaling 2.5"),
    ("use_grouped_topk", True, "grouped top-k"),
    ("apply_router_weight_on_input", True, "router weight on input"),
])
def test_routing_options_the_kernel_does_not_apply_are_refused(
        monkeypatch, option, value, refusal):
    """The kernel's router is the plain softmax top-k; a layer with an
    expert-selection bias, a routed scaling factor, grouped top-k or the
    router weights on the input stays on the routing path that applies them."""
    import vllm_torchtpu.layers.vllm.moe_lowtok_scores_op as scores_op
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.vllm.quantization import fp8
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_FROM_SCORES", 1, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_MAX_TOKENS", 32, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", 1, raising=False)
    scores_op.prebuild(10, 8, 64)
    with FakeTensorMode():
        layer = _layer("weight_scale")
        assert fp8._lowtok_scores_static_refusal(layer, "silu") is None
        setattr(layer, option, value)
        assert fp8._lowtok_scores_static_refusal(layer, "silu") == refusal
        x = torch.empty(4, 4096, dtype=torch.bfloat16)
        logits = torch.empty(4, 512, dtype=torch.bfloat16)
        assert fp8._lowtok_scores_output(layer, x, logits, "silu") is None


def test_the_ids_hook_terms_are_stated_at_weight_loading(monkeypatch):
    """The grouped path's few-token entry decides inside the jitted function;
    its static terms on the served layer are evaluated once for the boot log
    and mirror the hook's condition."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.vllm.quantization import fp8
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_MAX_TOKENS", 256, raising=False)
    with FakeTensorMode():
        layer = _layer("weight_scale_inv")
        assert fp8._lowtok_ids_static_refusal(layer, "silu") is None
        assert fp8._lowtok_ids_static_refusal(layer,
                                              "gelu") == "activation gelu"
        layer.w13_weight_scale_inv = torch.empty(64, 4096, 2048)
        assert fp8._lowtok_ids_static_refusal(
            layer, "silu") == "w13 scale shape (64, 4096, 2048)"
        layer = _layer("weight_scale")  # no _inv scales: the hook's own term
        assert fp8._lowtok_ids_static_refusal(
            layer, "silu") == "no scales under weight_scale_inv"
        layer = _layer("weight_scale_inv")
        layer.moe_config.moe_parallel_config.use_ep = False
        assert fp8._lowtok_ids_static_refusal(
            layer, "silu") == "no expert parallelism"
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_MAX_TOKENS", 0, raising=False)
    assert fp8._lowtok_ids_static_refusal(
        layer, "silu") == "TPU_MOE_LOWTOK_MAX_TOKENS unset"


@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [4, 256])
def test_scores_entry_preserves_activation_dtype(monkeypatch, dtype, rows):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.vllm import moe_lowtok_scores_op as scores_op
    from vllm_torchtpu.layers.vllm.quantization import fp8

    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_FROM_SCORES", 1)
    monkeypatch.setattr(envs, "TPU_MOE_LOWTOK_MAX_TOKENS", 256)
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", 0)
    calls = []

    def kernel(hidden, *args, **kwargs):
        calls.append(hidden)
        return hidden

    monkeypatch.setattr(scores_op, "moe_lowtok_scores_op", kernel)
    with FakeTensorMode():
        layer = _layer("weight_scale_inv")
        hidden = torch.empty(rows, 4096, dtype=dtype)
        logits = torch.empty(rows, 512, dtype=dtype)
        result = fp8._lowtok_scores_output(layer, hidden, logits, "silu")
        if dtype == torch.bfloat16:
            assert result is hidden and calls == [hidden]
        else:
            assert result is None and not calls
