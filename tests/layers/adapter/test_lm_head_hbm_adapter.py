# SPDX-License-Identifier: Apache-2.0
"""TPU_LM_HEAD_HBM at the torch level.

CPU (a stand-in op): which head slots each mode rewires -- "all" swaps the target
head's method (shared draft slots follow), "drafter" rebinds only the draft's slots to
a copy sharing the same weight, refusals leave everything untouched, and a fault
(rebinding nothing) is detectable by the returned count.
TPU (the real op): the method vs the F.linear it replaces, eager and compiled: row
counts above the kernel's MAX_M bit for bit; decode counts within a few bf16 ulp,
few logits off."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: E402
    UnquantizedEmbeddingMethod,
)

from vllm_torchtpu.kernels import lm_head_hbm as lh  # noqa: E402
from vllm_torchtpu.layers.adapter import lm_head_hbm as adapter  # noqa: E402

H, V = 256, 512


class _Head(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16, device="cpu", bias=False):
        super().__init__()
        w = (0.02 * torch.randn(V, H)).to(dtype).to(device)
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(V)) if bias else None
        self.quant_method = UnquantizedEmbeddingMethod()


class _SharedHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head


class _Target(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.lm_head = head


class _Draft(torch.nn.Module):
    """DeepSeek-MTP shape: model.layers.<i>.shared_head.head -> the target's head."""

    def __init__(self, head, n_layers=1):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleDict(
            {str(78 + i): torch.nn.Module() for i in range(n_layers)}
        )
        for layer in self.model.layers.values():
            layer.shared_head = _SharedHead(head)


def _fake_op(x2, w):
    return x2 @ w.t() + 1.0  # distinguishable from F.linear


def test_all_mode_swaps_target_and_shared_slots():
    head = _Head()
    target, draft = _Target(head), _Draft(head, n_layers=2)
    n = adapter.install_lm_head_hbm(target, draft, "all", op=_fake_op)
    assert n == 3
    assert isinstance(head.quant_method, adapter.HbmLmHeadMethod)
    assert isinstance(
        head.quant_method, UnquantizedEmbeddingMethod
    )  # head_dtype checks
    for layer in draft.model.layers.values():
        assert layer.shared_head.head is head
    x = torch.randn(3, H).to(torch.bfloat16)
    torch.testing.assert_close(
        head.quant_method.apply(head, x), _fake_op(x, head.weight)
    )


def test_drafter_mode_leaves_target_and_shares_weight():
    head = _Head()
    target, draft = _Target(head), _Draft(head, n_layers=2)
    n = adapter.install_lm_head_hbm(target, draft, "drafter", op=_fake_op)
    assert n == 2
    assert type(head.quant_method) is UnquantizedEmbeddingMethod  # target untouched
    proxies = [layer.shared_head.head for layer in draft.model.layers.values()]
    assert proxies[0] is proxies[1] and proxies[0] is not head
    p = proxies[0]
    assert isinstance(p.quant_method, adapter.HbmLmHeadMethod)
    assert p.weight is head.weight  # same Parameter, no second copy
    assert target.lm_head is head
    x = torch.randn(4, H).to(torch.bfloat16)
    torch.testing.assert_close(p.quant_method.apply(p, x), _fake_op(x, head.weight))
    # the target's own path is still F.linear
    torch.testing.assert_close(
        head.quant_method.apply(head, x), torch.nn.functional.linear(x, head.weight)
    )


def test_method_falls_back_for_bias_or_non_bf16():
    m = adapter.HbmLmHeadMethod(_fake_op)
    h32 = _Head(dtype=torch.float32)
    x32 = torch.randn(2, H)
    torch.testing.assert_close(
        m.apply(h32, x32), torch.nn.functional.linear(x32, h32.weight)
    )
    hb = _Head()
    xb = torch.randn(2, H).to(torch.bfloat16)
    b = torch.ones(V, dtype=torch.bfloat16)
    torch.testing.assert_close(
        m.apply(hb, xb, b), torch.nn.functional.linear(xb, hb.weight, b)
    )
    # 3-D input keeps its leading dims
    x3 = torch.randn(2, 3, H).to(torch.bfloat16)
    assert m.apply(hb, x3).shape == (2, 3, V)


@pytest.mark.parametrize(
    "make",
    [
        lambda: _Head(dtype=torch.float32),
        lambda: _Head(bias=True),
    ],
)
def test_refusals_install_nothing(make):
    head = make()
    target, draft = _Target(head), _Draft(head)
    for mode in ("all", "drafter"):
        assert adapter.install_lm_head_hbm(target, draft, mode, op=_fake_op) == 0
        assert type(head.quant_method) is UnquantizedEmbeddingMethod
        assert draft.model.layers["78"].shared_head.head is head


def test_quantized_head_refused_and_modes_validated():
    head = _Head()

    class _Q(UnquantizedEmbeddingMethod):
        pass

    head.quant_method = _Q()
    assert adapter.install_lm_head_hbm(_Target(head), None, "all", op=_fake_op) == 0
    assert type(head.quant_method) is _Q
    with pytest.raises(ValueError):
        adapter.install_lm_head_hbm(_Target(_Head()), None, "everything", op=_fake_op)
    assert adapter.install_lm_head_hbm(_Target(_Head()), None, "", op=_fake_op) == 0


def test_drafter_mode_without_shared_slots_is_a_noop():
    head = _Head()
    other = _Head()
    assert (
        adapter.install_lm_head_hbm(
            _Target(head), _Draft(other), "drafter", op=_fake_op
        )
        == 0
    )
    assert adapter.install_lm_head_hbm(_Target(head), None, "drafter", op=_fake_op) == 0
    assert type(other.quant_method) is UnquantizedEmbeddingMethod


# ---------------------------------------------------------------- TPU: the real op


# The TPU lane's marker, not a probe: probing at import (a tensor moved to
# "tpu") creates the PJRT client in the pytest process during collection,
# which then holds the chips, so every test that spawns TPU workers fails.
tpu = pytest.mark.tpu_test

KH, KV = 6144, 9680


class _TpuHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w = (0.02 * torch.randn(KV, KH)).to(torch.bfloat16)
        self.weight = torch.nn.Parameter(w.to("tpu"), requires_grad=False)
        self.bias = None
        self.quant_method = UnquantizedEmbeddingMethod()

    def forward(self, x):
        return self.quant_method.apply(self, x, None)


def _ulps(a, b):
    a = a.view(torch.int16).to(torch.int32)
    b = b.view(torch.int16).to(torch.int32)
    a = torch.where(a < 0, -32768 - a, a)
    b = torch.where(b < 0, -32768 - b, b)
    return (a - b).abs()


@tpu
@pytest.mark.parametrize("compiled", [False, True])
def test_method_vs_flinear_on_tpu(compiled):
    torch.manual_seed(0)
    head = _TpuHead()
    rows = (1, 4, 8, 12, 16, lh.MAX_M, lh.MAX_M + 1, 128)
    xs = {t: torch.randn(t, KH).to(torch.bfloat16).to("tpu") for t in rows}
    torch._dynamo.reset()
    run = torch.compile(head, fullgraph=True, dynamic=False) if compiled else head
    ref = {t: run(x).cpu() for t, x in xs.items()}
    assert adapter.install_lm_head_hbm(_Target(head), None, "all") == 1
    torch._dynamo.reset()
    run = torch.compile(head, fullgraph=True, dynamic=False) if compiled else head
    for t, x in xs.items():
        got = run(x).cpu()
        d = _ulps(got, ref[t])
        n = int((d > 0).sum())
        r = ref[t].float()
        ulp = torch.exp2(torch.floor(torch.log2(r.abs().clamp_min(2.0**-6))) - 7)
        d_rel = float(((got.float() - r).abs() / ulp).max())
        print(
            f"LMHEADPIN t={t} compiled={compiled} off={n}/{d.numel()} "
            f"max_ulp_floored={d_rel:.1f}"
        )
        if t > lh.MAX_M:
            assert n == 0  # XLA's dot inside the op: same bits as F.linear
        else:
            assert d_rel <= 8 and n <= max(1, t * KV // 1000)
            assert (got.float().argmax(1) == ref[t].float().argmax(1)).all()
