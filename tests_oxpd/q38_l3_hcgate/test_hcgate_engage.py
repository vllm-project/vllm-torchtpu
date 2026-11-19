# SPDX-License-Identifier: Apache-2.0
"""HCGATE engagement per bucket from the traced graph (q38 L3-HCGATE; deviceless, 0 chips).

A two-HC layer (attn mix -> attn combine_mix into the mlp HC -> mlp combine), the
pattern every decoder layer traces, is captured with Dynamo (fullgraph; the backend
stops right after capture, so nothing executes) at each decode bucket, with the token
dimension marked dynamic as the engine traces it. Counted in each captured FX graph:

* ``hc_decode``: calls of the fused decode ops (``qwen4_exp::hc_decode_*`` and
  ``qwen4_exp::hc_decode_combine_*``; their Pallas kernels are ``qwen4exp_hc_decode*``);
* tile ops: ``hc_norm`` / ``hc_up_epilogue`` / ``hc_full`` / ``gated_residual``.

Ops: ``real`` (CDK CPU image) = the GatedResidual constructor builds the real torch_tpu
jax_ops from TPU_QWEN4_HC_DECODE_FUSED / _UP_T and FINAL's tiles 128 / 128 / 256;
``standin`` (any host with torch) = torch.library ops of the same names attached after
construction (checks the capture and counting logic where torch_tpu is absent).

Arms: final (HC fused, gate unset), nohcf (FINAL - HC fused), gate128 (VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS
=128), gate128_hcx256 (+ VLLM_TPU_HC_XLA_MAX_ROWS=256), nohcf_hcx256, band (=128:256, the car's
value: the 256 route kept fused). Required:
gate128 has 2 hc_decode calls at 16 / 96 / 128 and 0 at 160 / 192 / 224 / 256; its graph
equals final's node for node at <= 128 and nohcf's at > 128 (with HCX: nohcf_hcx256's);
the token symbol is never specialized where final's is not.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q38_l3_hcx"))
import hcx_load  # noqa: E402

ENV = "VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS"
HCX_ENV = "VLLM_TPU_HC_XLA_MAX_ROWS"
FUSED_KEYS = ("TPU_QWEN4_HC_DECODE_FUSED", "TPU_QWEN4_HC_DECODE_UP_T")
HC, H, LR = 4, 128, 8
BUCKETS = (16, 96, 128, 160, 192, 224, 256)
ARMS = {
    "final": dict(fused=True),
    "nohcf": dict(fused=False),
    "gate128": dict(fused=True, gate="128"),
    "gate128_hcx256": dict(fused=True, gate="128", hcx="256"),
    "band": dict(fused=True, gate="128:256"),
    "nohcf_hcx256": dict(fused=False, hcx="256"),
}
TILE_NAMES = ("hc_norm", "hc_up_epilogue", "hc_full", "gated_residual")


def _ops_mode():
    forced = os.environ.get("Q38_L3_HCGATE_OPS", "").strip()
    if forced:
        return forced
    try:
        import torch_tpu  # noqa: F401
    except Exception:  # noqa: BLE001
        return "standin"
    return "real"


# ---------------------------------------------------------------- stand-in ops (same names)
_LIB = None


def _standin_ops():
    global _LIB
    if _LIB is None:
        lib = torch.library.Library("q38_l3_hcgate_test", "FRAGMENT")

        def define(name, schema, impl, fake):
            lib.define(f"{name}{schema}")
            lib.impl(name, impl, "CPU")
            torch.library.register_fake(f"q38_l3_hcgate_test::{name}", fake)

        def mixed(x):
            return x.new_zeros(x.shape[0], H)

        define("hc_decode_inj_upt", "(Tensor x, Tensor w, Tensor d, Tensor u, Tensor i) -> (Tensor, Tensor)",
               lambda x, w, d, u, i: (mixed(x), x.new_zeros(x.shape[0], HC, dtype=torch.float32)),
               lambda x, w, d, u, i: (x.new_empty(x.shape[0], H), x.new_empty(x.shape[0], HC, dtype=torch.float32)))
        define("hc_decode_combine_upt",
               "(Tensor r, Tensor b, Tensor g, Tensor w, Tensor d, Tensor u, Tensor i) -> (Tensor, Tensor, Tensor)",
               lambda r, b, g, w, d, u, i: (mixed(r), r.new_zeros(r.shape[0], HC, dtype=torch.float32), r.clone()),
               lambda r, b, g, w, d, u, i: (r.new_empty(r.shape[0], H),
                                            r.new_empty(r.shape[0], HC, dtype=torch.float32), torch.empty_like(r)))
        define("hc_norm_t", "(Tensor x, Tensor w) -> Tensor", lambda x, w: x.clone(),
               lambda x, w: torch.empty_like(x))
        define("hc_up_epilogue_t", "(Tensor g, Tensor n, Tensor w) -> Tensor", lambda g, n, w: mixed(g),
               lambda g, n, w: g.new_empty(g.shape[0], H))
        define("hc_full_t", "(Tensor x, Tensor w, Tensor d, Tensor u) -> (Tensor, Tensor)",
               lambda x, w, d, u: (mixed(x), x.clone()),
               lambda x, w, d, u: (x.new_empty(x.shape[0], H), torch.empty_like(x)))
        define("gated_residual_t", "(Tensor r, Tensor b, Tensor g) -> Tensor", lambda r, b, g: r.clone(),
               lambda r, b, g: torch.empty_like(r))
        _LIB = lib
    return torch.ops.q38_l3_hcgate_test


def _attach_standin(mod, fused):
    ops = _standin_ops()
    mod.prefill_norm_tile, mod.prefill_up_tile, mod.prefill_full_tile = 128, 128, 256
    mod._prefill_tile_step = 512
    mod._prefill_norm_op = ops.hc_norm_t
    mod._prefill_up_op = ops.hc_up_epilogue_t
    mod._prefill_full_op = ops.hc_full_t
    mod._prefill_combine_op = ops.gated_residual_t
    if fused:
        mod._decode_mix_op = ops.hc_decode_inj_upt
        mod._decode_combine_mix_op = ops.hc_decode_combine_upt
        mod._decode_up_t = True


# ---------------------------------------------------------------- builders
def _clear(monkeypatch, envs):
    for key in (ENV, HCX_ENV, *FUSED_KEYS):
        monkeypatch.delenv(key, raising=False)
        if key in vars(envs):  # a frozen module attribute would shadow the env (delete, never write back)
            monkeypatch.delattr(envs, key)


def _build(m, monkeypatch, seed, ops, fused=True, gate=None, hcx=None):
    _clear(monkeypatch, m.envs)
    if gate:
        monkeypatch.setenv(ENV, gate)
    if hcx:
        monkeypatch.setenv(HCX_ENV, hcx)
    if ops == "real" and fused:
        for key in FUSED_KEYS:
            monkeypatch.setenv(key, "1")
    cfg = m.hc.HyperConnectionConfig(
        hc_count=HC, hidden_size=H, params_dtype=torch.bfloat16, hc_lowrank=LR,
        rms_norm_eps=1e-6, hc_per_branch_norm=True,
    )
    if ops == "real":
        mod = m.hc.GatedResidual(cfg, prefill_up_tile=128, prefill_norm_tile=128, prefill_full_tile=256)
        assert (mod._decode_mix_op is not None) == fused
    else:
        mod = m.hc.GatedResidual(cfg)
        _attach_standin(mod, fused)
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed)
        for p in mod.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * 0.4).to(p.dtype))
    lo, _, hi = (gate or "0").partition(":")
    assert mod._hcf_max_rows == (int(lo), int(hi or 0))
    assert mod._hcx_max_rows == (int(hcx) if hcx else 0)
    _clear(monkeypatch, m.envs)
    return mod


class _Captured(Exception):
    pass


def _capture(attn, mlp, rows, dynamic=True):
    """Dynamo-capture one decoder layer's HC pattern at ``rows``; nothing executes.
    Returns (node list [(op, target, name)] or None, error text or ''). An op is counted
    by its target or its node name; graphs are compared on (op, target) only (generated
    names carry counters that a dead shape read can advance). The nodes are read
    inside the backend: Dynamo may clear the graph after the backend raises."""
    seen = {}

    def backend(gm, example_inputs):
        seen["gm"] = [(n.op, str(n.target), n.name) for n in gm.graph.nodes]
        raise _Captured()

    def layer(x, block):
        mixed, residual = attn.mix(x)
        mixed2, residual2 = attn.combine_mix(block, residual, mlp)
        return mixed, mixed2, mlp.combine(block, residual2)

    g = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, HC * H, generator=g).bfloat16()
    block = torch.randn(rows, H, generator=g).bfloat16()
    if dynamic:
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(block, 0)
    torch._dynamo.reset()
    try:
        torch.compile(layer, backend=backend, fullgraph=True)(x, block)
    except Exception as e:  # noqa: BLE001 - BackendCompilerFailed wraps _Captured
        if "gm" in seen and (isinstance(e, _Captured) or isinstance(getattr(e, "inner_exception", None), _Captured)):
            return seen["gm"], ""
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
    return None, "the capture backend never ran"


def _count(nodes, names):
    return sum(1 for op, target, name in nodes
               if op == "call_function" and any(k in target or k in name for k in names))


def _shape(nodes):
    return [(op, target) for op, target, _ in nodes]


@pytest.fixture(scope="module")
def table():
    """{(arm, rows): (decode calls, tile calls, node list, dynamic error)}, built once."""
    m = hcx_load.load()
    ops = _ops_mode()
    mp = pytest.MonkeyPatch()
    out = {}
    try:
        for arm, kw in ARMS.items():
            attn = _build(m, mp, 0, ops, **kw)
            mlp = _build(m, mp, 5, ops, **kw)
            for rows in BUCKETS:
                gm, err = _capture(attn, mlp, rows, dynamic=True)
                dyn_err = err
                if gm is None:  # record why, then capture static so the counts are still read
                    gm, err = _capture(attn, mlp, rows, dynamic=False)
                    assert gm is not None, (arm, rows, err)
                assert len(gm) > 2, (arm, rows, gm)
                dec, til = _count(gm, ("hc_decode",)), _count(gm, TILE_NAMES)
                out[(arm, rows)] = (dec, til, _shape(gm), dyn_err)
                print(f"HCGATE-ENGAGE ops={ops} arm={arm} rows={rows} hc_decode={dec} tiles={til} "
                      f"dynamic={'ok' if not dyn_err else 'SPECIALIZED/ERR ' + dyn_err}", flush=True)
    finally:
        mp.undo()
    return out


def test_ops_mode_is_real_in_the_image():
    if hcx_load.mode() == "image":
        assert _ops_mode() == "real", "the CDK CPU image must use the real torch_tpu jax_ops"


def test_final_fuses_every_bucket(table):
    for rows in BUCKETS:
        assert table[("final", rows)][:2] == (2, 0), rows


def test_gate128_counts_per_bucket(table):
    for rows in BUCKETS:
        dec, til = table[("gate128", rows)][:2]
        if rows <= 128:
            assert (dec, til) == (2, 0), rows
        else:
            assert dec == 0, rows
    # Without HCX: 160 / 192 / 224 take XLA (no tile op); 256 takes the norm + up tiles (2 per HC).
    assert [table[("gate128", r)][1] for r in (160, 192, 224, 256)] == [0, 0, 0, 4]


def test_gate128_graph_equals_final_below_and_nohcf_above(table):
    for rows in BUCKETS:
        ref = "final" if rows <= 128 else "nohcf"
        assert table[("gate128", rows)][2] == table[(ref, rows)][2], (rows, ref)


def test_gate128_with_hcx_takes_xla_above(table):
    for rows in BUCKETS:
        dec, til, nodes, _ = table[("gate128_hcx256", rows)]
        if rows <= 128:
            assert (dec, til) == (2, 0) and nodes == table[("final", rows)][2], rows
        else:
            assert (dec, til) == (0, 0) and nodes == table[("nohcf_hcx256", rows)][2], rows


def test_the_token_symbol_is_specialized_no_more_than_finals(table):
    for arm in ARMS:
        for rows in BUCKETS:
            if table[("final", rows)][3] or table[("nohcf", rows)][3]:
                continue  # the control itself specializes there: uncompared (printed above)
            assert not table[(arm, rows)][3], (arm, rows, table[(arm, rows)][3])


def test_band_keeps_256_fused_and_gates_160_to_224(table):
    """The car's value 128:256: graph == final's at 16 / 96 / 128 / 256, == nohcf's at 160 / 192 / 224."""
    for rows in BUCKETS:
        dec, til, nodes, _ = table[("band", rows)]
        if 128 < rows < 256:
            assert (dec, til) == (0, 0) and nodes == table[("nohcf", rows)][2], rows
        else:
            assert (dec, til) == (2, 0) and nodes == table[("final", rows)][2], rows
