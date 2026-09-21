"""The hoisted norm weight (layers/vllm/norm_weight_hoist.py): after install, every GemmaRMSNorm carries (1 + w)
in f32 as a buffer and its native forward gives exactly what it gave before, with and without a residual."""
import pytest
import torch


def _norms():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    torch.manual_seed(0)
    with set_current_vllm_config(
            VllmConfig()):  # CustomOp reads the config at construction
        a, b = GemmaRMSNorm(64, eps=1e-6), GemmaRMSNorm(64, eps=1e-6)
    for m in (a, b):
        with torch.no_grad():
            m.weight.copy_(torch.randn(64) * 0.1)
    return a, b


@pytest.fixture(autouse=True)
def _restore_the_class():
    """install() patches GemmaRMSNorm for the process; put it back."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    forward = GemmaRMSNorm.__dict__.get("forward_native")
    flag = GemmaRMSNorm.__dict__.get("_tpu_norm_weight_hoisted")
    yield
    if forward is not None:
        GemmaRMSNorm.forward_native = forward
    if flag is None:
        if hasattr(GemmaRMSNorm, "_tpu_norm_weight_hoisted"):
            delattr(GemmaRMSNorm, "_tpu_norm_weight_hoisted")


def test_hoisted_forward_matches_and_buffers_exist():
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    from vllm_torchtpu.layers.adapter import norm_weight_hoist
    a, b = _norms()
    x = torch.randn(8, 64)
    r = torch.randn(8, 64)
    ref_plain = GemmaRMSNorm.forward_native(a, x.clone())
    ref_res = GemmaRMSNorm.forward_native(a, x.clone(), r.clone())
    model = torch.nn.Sequential(a, b)
    n = norm_weight_hoist.install(model)
    assert n == 2 and hasattr(
        a, "_tpu_weight_plus_one"
    ) and a._tpu_weight_plus_one.dtype == torch.float32
    torch.testing.assert_close(a._tpu_weight_plus_one,
                               a.weight.float() + 1.0,
                               rtol=0,
                               atol=0)
    out_plain = a.forward_native(x.clone())
    out_res = a.forward_native(x.clone(), r.clone())
    torch.testing.assert_close(out_plain, ref_plain, rtol=0, atol=0)
    for o, e in zip(out_res, ref_res):
        torch.testing.assert_close(o, e, rtol=0, atol=0)


def test_install_is_idempotent_and_gated():
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import norm_weight_hoist
    a, b = _norms()
    model = torch.nn.Sequential(a, b)
    assert norm_weight_hoist.install(model) == 2
    assert norm_weight_hoist.install(
        model) == 2  # buffers refreshed in place, no duplicates
    assert len([k for k, _ in a.named_buffers()]) == 1
    if not envs.TPU_NORM_WEIGHT_HOIST:
        assert norm_weight_hoist.install_if_enabled(model) == 0
