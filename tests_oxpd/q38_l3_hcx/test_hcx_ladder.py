# SPDX-License-Identifier: Apache-2.0
"""HCX through vLLM's real compile pipeline and PD's shape-variant hand-over
(q38 L3-HCX; CPU image only: needs vLLM).

A two-HC layer (attn mix -> combine_mix -> mlp combine) of the real
GatedResidual, with FINAL's decode tiles (norm 128 / up 128 / full 256) as
recorded custom ops whose names contain "wave", runs the precompile ladder of
tests/compilation/test_shape_variants.py (``@support_torch_compile`` ->
VllmBackend -> PiecewiseBackend -> the real TpuCompilerAdaptor and
shape_variants hand-over, fake torch_tpu backend). Each bucket must be compiled
exactly once by a trace that accepts it, nothing refused, no source-order
mismatch, and the executable serving each bucket must hold the expected number
of tile ops: off 4 at 128 / 256 / 512; HCX 256 none at <= 256 and 4 at 512;
HCX 512 none anywhere (that case needs the combine to skip the prefill row test,
which would otherwise pin the token symbol to 512).

Also: compute_tpu_compilation_hash moves with the value and only when set.
"""

import os
import sys
from functools import partial

import hcx_load
import pytest
import torch

if hcx_load.mode() != "image":
    pytest.skip("needs vLLM's compile pipeline (the CDK CPU image)", allow_module_level=True)

sys.path.insert(0, os.path.join(hcx_load.TREE, "tests", "compilation"))

import test_shape_variants as sv  # noqa: E402
from torch import nn  # noqa: E402
from vllm.compilation.decorators import support_torch_compile  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_torchtpu.compilation import shape_variants  # noqa: E402

ENV = "VLLM_TPU_HC_XLA_MAX_ROWS"
HC, H, LR = 4, 8, 4
BUCKETS = [64, 96, 128, 256, 512]

_lib = torch.library.Library("q38_l3_hcx_test", "FRAGMENT")
_lib.define("wave_norm(Tensor x, Tensor w) -> Tensor")
_lib.impl("wave_norm", lambda x, w: (x.float() * 0.5).to(x.dtype), "CPU")
torch.library.register_fake("q38_l3_hcx_test::wave_norm", lambda x, w: torch.empty_like(x))
_lib.define("wave_up(Tensor g, Tensor n, Tensor w) -> Tensor")
_lib.impl("wave_up", lambda g, n, w: g.new_zeros(g.shape[0], w.shape[0] // HC), "CPU")
torch.library.register_fake(
    "q38_l3_hcx_test::wave_up", lambda g, n, w: g.new_empty(g.shape[0], w.shape[0] // HC)
)
_lib.define("wave_full(Tensor x, Tensor w, Tensor d, Tensor u) -> (Tensor, Tensor)")
_lib.impl("wave_full", lambda x, w, d, u: (x.new_zeros(x.shape[0], x.shape[1] // HC), x.clone()), "CPU")
torch.library.register_fake(
    "q38_l3_hcx_test::wave_full",
    lambda x, w, d, u: (x.new_empty(x.shape[0], x.shape[1] // HC), torch.empty_like(x)),
)
_lib.define("wave_comb(Tensor r, Tensor b, Tensor g) -> Tensor")
_lib.impl("wave_comb", lambda r, b, g: r.clone(), "CPU")
torch.library.register_fake("q38_l3_hcx_test::wave_comb", lambda r, b, g: torch.empty_like(r))
OPS = torch.ops.q38_l3_hcx_test


def _tiled(seed):
    m = hcx_load.load()
    cfg = m.hc.HyperConnectionConfig(
        hc_count=HC, hidden_size=H, params_dtype=torch.bfloat16, hc_lowrank=LR,
        rms_norm_eps=1e-6, hc_per_branch_norm=True,
    )
    mod = m.hc.GatedResidual(cfg)
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed)
        for p in mod.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * 0.4).to(p.dtype))
    mod.prefill_norm_tile, mod.prefill_up_tile, mod.prefill_full_tile = 128, 128, 256
    mod._prefill_tile_step = 512
    mod._prefill_norm_op = OPS.wave_norm
    mod._prefill_up_op = OPS.wave_up
    mod._prefill_full_op = OPS.wave_full
    mod._prefill_combine_op = OPS.wave_comb
    return mod


@support_torch_compile(dynamic_arg_dims={"x": 0})
class HcxLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.attn = _tiled(0)
        self.mlp = _tiled(5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block, residual = self.attn.mix(x)
        block, residual = self.attn.combine_mix(block, residual, self.mlp)
        return self.mlp.combine(block, residual)


def _x(rows):
    return torch.zeros(rows, HC * H, dtype=torch.bfloat16)


def _ladder(cache_root, hcx, expect):
    if hcx is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = hcx
    sv.BUCKETS[:] = BUCKETS
    config, backend = sv.prepare(cache_root)
    model = sv.build(HcxLayer, config)
    assert model.attn._hcx_max_rows == (int(hcx) if hcx else 0)
    retraces = 0
    with set_current_vllm_config(config), shape_variants.warmup() as tracked:
        for rows in BUCKETS:
            if tracked.needs_retrace(rows):
                shape_variants.retrace(model, config)
                retraces += 1
            with set_forward_context({}, config):
                model(_x(rows))
        assert not tracked.refused, tracked.refused
        mismatches = tracked.source_mismatches
    assert mismatches == 0, mismatches
    assert sorted(backend.compiled) == sorted(expect.items()), backend.compiled
    served = {}
    with set_current_vllm_config(config):
        for rows in BUCKETS:
            sv.CALLS.clear()
            with set_forward_context({}, config):
                model(_x(rows))
            served[rows] = sv.CALLS[-1]
    assert served == {rows: (rows, waves) for rows, waves in expect.items()}, served
    print(f"HCX-LADDER hcx={hcx} retraces={retraces} compiled={sorted(backend.compiled)}", flush=True)


CASES = {
    "off": (None, {64: 0, 96: 0, 128: 4, 256: 4, 512: 4}),
    "hcx256": ("256", {64: 0, 96: 0, 128: 0, 256: 0, 512: 4}),
    "hcx512": ("512", {64: 0, 96: 0, 128: 0, 256: 0, 512: 0}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_ladder_compiles_each_bucket_once_with_the_expected_tile_ops(tmp_path, case):
    hcx, expect = CASES[case]
    sv.in_child(partial(_ladder, str(tmp_path), hcx, expect))


def test_compile_hash_keys_hcx_only_when_set():
    import test_tpu_compile_key_registration as reg

    off = reg._hash_with({ENV: None})
    assert reg._hash_with({ENV: ""}) == off
    on = reg._hash_with({ENV: "256"})
    assert on != off
    assert reg._hash_with({ENV: "128"}) not in (on, off)
