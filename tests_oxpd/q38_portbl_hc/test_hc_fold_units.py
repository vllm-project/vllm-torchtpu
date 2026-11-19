# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL HC: structure, routing, engagement and tile-rule tests (CPU; 0 chips).

TPU_QWEN4_HC_DECODE_FOLD / TPU_QWEN4_HC_DECODE_TILE_ROWS, ported from Benliang Wang <benliangw@google.com> (commits
d523c2ce0 / 28419c4a5 on benliangw/q38fh-integ-pin-5b44702aea; his tile-table test is reused in
test_kernel_tile_rows_table). The bitwise legs are in test_hc_fold_bitwise.py.
"""

import ast
import inspect
import logging
import os
import sys
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hc_fold_stage as S  # noqa: E402
from tests.compilation.dynamo_input_order import parameters_by_first_use
from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import hc_decode as D
from vllm_torchtpu.models.vllm.qwen4_exp import hc_engagement as E
from vllm_torchtpu.models.vllm.qwen4_exp import hyperconnection as H
from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpBackbone, Qwen4ExpDecoderLayer

SMALL = S.DIMS["small"]


@pytest.fixture(autouse=True)
def _reset_engagement():
    E.reset()
    yield
    E.reset()


# ---------------------------------------------------------------- tile rule (TPU_QWEN4_HC_DECODE_TILE_ROWS)
@pytest.mark.parametrize(
    "max_tile, expected",
    [
        (0, {96: 96, 128: 128, 192: 64, 256: 128, 320: 64, 384: 128, 448: 64}),
        (256, {96: 96, 112: 112, 128: 128, 144: 144, 192: 192, 208: 208, 256: 256,
               320: 160, 384: 192, 448: 224, 512: 256}),
        (224, {192: 192, 256: 128, 320: 160, 384: 192, 448: 224}),
        (512, {160: 160, 192: 192, 224: 224, 256: 256, 320: 320, 448: 448}),
    ],
)
def test_kernel_tile_rows_table(max_tile, expected):
    """Benliang's table (test_qwen4_exp_hc_decode.py on 28419c4a5) plus the v8 buckets 160 / 224 at 512."""
    for tokens, tile in expected.items():
        assert D.decode_kernel_tile(tokens, max_tile) == tile, tokens


@pytest.mark.parametrize("max_tile", [0] + list(range(128, 513, 16)))
def test_kernel_tile_rows_properties(max_tile):
    """Every count decode_token_tile accepts: the tile divides it; with a limit and >= 128 rows the tile is a multiple
    of 16, at most max(limit, decode_token_tile), and the count is the fewest such; the limit never changes which
    counts are accepted (the torch-side gate stays decode_token_tile); 0 is decode_token_tile exactly."""
    for tokens in range(1, 1025):
        tile = D.decode_kernel_tile(tokens, max_tile)
        base = D.decode_token_tile(tokens)
        assert (tile is None) == (base is None), tokens
        if tile is None:
            continue
        assert tokens % tile == 0 and tile >= base, (tokens, tile)
        if not max_tile or tokens < 128:
            assert tile == base, (tokens, tile)
            continue
        assert tile % 16 == 0, (tokens, tile)
        assert tile <= max(max_tile, base), (tokens, tile)
        fewer = [c for c in range(1, tokens // tile) if tokens % c == 0 and (tokens // c) % 16 == 0
                 and tokens // c <= max_tile]
        assert not fewer, (tokens, tile, fewer)


@pytest.mark.parametrize("value, ok", [(0, True), (128, True), (512, True), (144, True), (496, True),
                                       (64, False), (100, False), (130, False), (528, False), (1024, False)])
def test_tile_rows_validation(value, ok):
    if ok:
        assert D.decode_tile_rows(value) == value
    else:
        with pytest.raises(ValueError, match="TPU_QWEN4_HC_DECODE_TILE_ROWS"):
            D.decode_tile_rows(value)


def test_bad_tile_rows_fails_at_build():
    with S.env_state(**{S.FUSED: "1", S.UPT: "1", S.TILE: "100"}), S.core_builders():
        with pytest.raises(ValueError, match="TPU_QWEN4_HC_DECODE_TILE_ROWS"):
            S.build_hc(SMALL, 1)


# ---------------------------------------------------------------- keys
def test_keys_registered_hashed_and_default_off(monkeypatch):
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    for k in (S.FOLD, S.TILE):
        monkeypatch.delenv(k, raising=False)
        assert k in envs.environment_variables
        assert k not in _TPU_COMPILE_ENV_IGNORED       # program-changing: hashed
    assert envs.environment_variables[S.FOLD]() is False
    assert envs.environment_variables[S.TILE]() == 0
    src = open(inspect.getsourcefile(envs)).read()
    for k in (S.FOLD, S.TILE):
        assert src.count(f'"{k}":') == 1, k


# ---------------------------------------------------------------- GatedResidual pieces
def test_unset_keys_build_todays_ops():
    """Fold / tile unset: the same builder calls and op names as v8 (no no-injection op, no tile tag)."""
    S.BUILT.clear()
    with S.env_state(**S.SERVED), S.core_builders():
        S.build_hc(SMALL, 1)
        mixer = S.build_hc(SMALL, 2, inject=False)
    names = sorted(n for n in S.BUILT if "hc_decode" in n)
    assert names and all("_noinj" not in n and not re.search(r"_t[0-9]+_v1$", n) for n in names), names
    assert mixer._decode_combine_mix_op is None and mixer._decode_mix_op is not None
    print(f"\nPBL_HC UNSET ops {names}")


def test_fold_builds_the_mixer_op_with_up_t_and_tile():
    S.BUILT.clear()
    with S.env_state(**dict(S.SERVED, **{S.FOLD: "1", S.TILE: "512"})), S.core_builders():
        mixer = S.build_hc(SMALL, 2, inject=False)
    noinj = [n for n in S.BUILT if "_noinj" in n]
    assert mixer._decode_combine_mix_op is not None
    assert len(noinj) == 1 and noinj[0].endswith("_noinj_upt_t512_v1"), noinj


class _Boom:
    """A token count that must never be read."""

    def __index__(self):
        raise AssertionError("token count read")

    __int__ = __index__

    def __le__(self, other):
        raise AssertionError("token count compared")

    __lt__ = __ge__ = __gt__ = __mod__ = __rmod__ = __le__


def test_decode_fold_rows_rule():
    with S.env_state(**dict(S.SERVED, **{S.FOLD: "1"})), S.core_builders():
        attn, mixer = S.build_hc(SMALL, 1), S.build_hc(SMALL, 2, inject=False)
        for m in (attn, mixer):
            assert [m.decode_fold_rows(r, torch.bfloat16) for r in (16, 64, 80, 96, 112, 128)] == [True] * 6
            assert [m.decode_fold_rows(r, torch.bfloat16) for r in (160, 192, 224, 256, 512, 1024)] == [False] * 6
            assert not m.decode_fold_rows(96, torch.float32)
    with S.env_state(**{S.FUSED: "1", S.UPT: "1", S.FOLD: "1"}), S.core_builders():
        ungated = S.build_hc(SMALL, 3)
        assert [ungated.decode_fold_rows(r, torch.bfloat16) for r in (160, 192, 224, 256, 512)] == [True] * 5
        assert not ungated.decode_fold_rows(136, torch.bfloat16) and not ungated.decode_fold_rows(1024, torch.bfloat16)
    # Unarmed (the MTP draft after disable_decode_fused; or no fused ops): never reads the row count.
    with S.env_state(**dict(S.SERVED, **{S.FOLD: "1"})), S.core_builders():
        draft_mixer = S.build_hc(SMALL, 4, inject=False)
    assert draft_mixer._decode_combine_mix_op is not None
    draft_mixer.disable_decode_fused()
    assert draft_mixer._decode_combine_mix_op is None and draft_mixer._decode_mix_op is None
    assert draft_mixer.decode_fold_rows(_Boom(), torch.bfloat16) is False
    with S.env_state(), S.core_builders():
        plain = S.build_hc(SMALL, 5)
    assert plain.decode_fold_rows(_Boom(), torch.bfloat16) is False


def test_combine_mix_noinj_branch_and_prefill_fused_off():
    calls = []
    me = SimpleNamespace(hc_count=4, combine=lambda b, r: calls.append("combine") or r[0])
    residual = torch.zeros(8, 16, dtype=torch.bfloat16)
    gate = torch.zeros(8, 4, dtype=torch.float32)
    block = torch.zeros(8, 4, dtype=torch.bfloat16)
    mixer = SimpleNamespace(
        _decode_combine_mix_op=lambda *a: calls.append("noinj") or ("M", "C"), _decode_fused=lambda x: True,
        _decode_weights=lambda: (),
    )
    assert H.GatedResidual.combine_mix(me, block, (residual, gate), mixer) == ("M", ("C", None))
    prefill_next = SimpleNamespace(
        _decode_combine_mix_op=None, _prefill_combine_mix_op=lambda *a: calls.append("prefill") or (1, 2, 3),
        _prefill_fused_mix=lambda x: True, _decode_weights=lambda: (), _hcx_xla=lambda x: False,
        mix=lambda h: calls.append("mix") or ("m", (h, None)),
    )
    H.GatedResidual.combine_mix(me, block, (residual, gate), prefill_next)
    H.GatedResidual.combine_mix(me, block, (residual, gate), prefill_next, prefill_fused=False)
    assert calls == ["noinj", "prefill", "combine", "mix"]


# ---------------------------------------------------------------- the stage loop with exact stand-ins
class _HC:
    """combine(moe, (r, g)) = r + moe; mix(h) = (2h, (h, None)) (GatedResidual's contract); folds where ``fold_ok``."""

    def __init__(self, log, name, fold_ok=True):
        self.log, self.name, self.fold_ok = log, name, fold_ok

    def mix(self, hidden):
        return hidden * 2, (hidden, None)

    def combine(self, moe, residual):
        return residual[0] + moe

    def decode_fold_rows(self, tokens, dtype):
        self.log.append(("fold_rows", self.name, int(tokens)))
        return self.fold_ok

    def combine_mix(self, moe, residual, next_hc, reread=False, **kw):
        self.log.append(("combine_mix", self.name, next_hc.name, tuple(sorted(kw.items()))))
        return next_hc.mix(self.combine(moe, residual))


class _Layer:
    def __init__(self, log, i, ple=False, fold_ok=True):
        self.log, self.i = log, i
        self.ple = object() if ple else None
        self.attn_hyper_connection = _HC(log, f"attn{i}", fold_ok)
        self.mlp_hyper_connection = _HC(log, f"mlp{i}", fold_ok)

    def __call__(self, hidden, *args, pending=None, defer=False, **kw):
        self.log.append(("layer", self.i, pending is not None, defer, (("decode_fold", kw.get("decode_fold", "absent")),)))
        if pending is not None:
            assert hidden is None and self.ple is None, "a hand-off into a PLE layer"
            moe_prev, residual_prev, hc_prev = pending
            extra = {"prefill_fused": False} if kw.get("decode_fold") else {}
            block, residual = hc_prev.combine_mix(moe_prev, residual_prev, self.attn_hyper_connection, **extra)
        else:
            if self.ple is not None:
                hidden = hidden + 0.5
            block, residual = self.attn_hyper_connection.mix(hidden)
        moe = block * (self.i + 1) + 1.0
        if defer:
            return moe, residual, self.mlp_hyper_connection
        return self.mlp_hyper_connection.combine(moe, residual)


def _run_loop(env, stage, ple_at, last_stage=True, fold_ok=True, rows=2):
    log = []
    layers = {i: _Layer(log, i, ple=(i == ple_at), fold_ok=fold_ok) for i in stage}
    mixer = _HC(log, "mixer", fold_ok)
    fake = SimpleNamespace(
        host_ple_index=None, stage=range(stage[0], stage[-1] + 1), first_stage=False, last_stage=last_stage,
        layers=layers, hc_count=4, sp_pad_rows=0, embed_tokens=None, hyper_connection_mixer=mixer,
        emit_mtp_hidden=True,
    )
    hidden = torch.arange(rows * 4, dtype=torch.float64).reshape(rows, 4)
    with S.env_state(**env):
        out = Qwen4ExpBackbone._forward_impl(
            fake, torch.zeros(rows, dtype=torch.int64), torch.zeros(rows, dtype=torch.int64), None,
            sequence_parallel=False, intermediate_tensors={"hidden_states": hidden},
            query_start_loc=None, ngram_context=None, state_slots=None, context_lens=None,
            valid_tokens=None, main_block_table=None, compressed_block_table=None,
        )
    return out, log


@pytest.mark.parametrize("stage,ple_at", [([0, 1, 2, 3, 4, 5], 1), ([6, 7, 8, 9], None), ([0, 1, 2], 2)])
def test_fold_loop_skips_ple_and_folds_the_mixer(stage, ple_at):
    (ref_m, ref_h), log0 = _run_loop({}, stage, ple_at)
    (m, h), log1 = _run_loop({S.FOLD: "1"}, stage, ple_at)
    assert torch.equal(ref_m, m) and torch.equal(ref_h, h)
    assert not any(e[0] in ("combine_mix", "fold_rows") for e in log0)   # unset: the predicate is never asked
    assert all(e[4] == (("decode_fold", False),) for e in log0 if e[0] == "layer")
    calls = [e for e in log1 if e[0] == "layer"]
    for k, (_, i, took, deferred, kw) in enumerate(calls):
        nxt = stage[k + 1] if k + 1 < len(stage) else None
        assert deferred == (nxt is None or nxt != ple_at), (i, deferred)
        assert took == (k > 0 and calls[k - 1][3]), (i, took)
        assert dict(kw)["decode_fold"] == took, (i, kw)
    hands = [e for e in log1 if e[0] == "combine_mix"]
    want = [("combine_mix", f"mlp{stage[k]}", f"attn{stage[k + 1]}", (("prefill_fused", False),))
            for k in range(len(stage) - 1) if stage[k + 1] != ple_at]
    want.append(("combine_mix", f"mlp{stage[-1]}", "mixer", (("prefill_fused", False),)))
    assert hands == want


def test_fold_loop_not_at_uncovered_rows_and_not_across_stages():
    (ref, _), _ = _run_loop({}, [0, 1, 2, 3], None, fold_ok=False)
    (out, _), log = _run_loop({S.FOLD: "1"}, [0, 1, 2, 3], None, fold_ok=False)
    assert torch.equal(ref, out)
    assert not any(e[0] == "combine_mix" for e in log) and all(not e[3] for e in log if e[0] == "layer")
    # A non-last pipeline stage hands its streams on: its last layer combines (no mixer on this stage).
    out, log = _run_loop({S.FOLD: "1"}, [0, 1, 2], None, last_stage=False)
    assert [e[3] for e in log if e[0] == "layer"] == [True, True, False]
    assert not any(e[0] == "fold_rows" and e[1] == "mixer" for e in log)
    ref, _ = _run_loop({}, [0, 1, 2], None, last_stage=False)
    assert torch.equal(ref["hidden_states"], out["hidden_states"])


def test_defer_d_alone_is_unchanged_and_keeps_precedence():
    """DEFER-D alone: no fold predicate, no decode_fold hand-off, combine_mix called exactly as on v8 (no
    prefill_fused keyword). DEFER-D + fold: DEFER-D's hand-offs keep their routing; the fold only adds the mixer."""
    _, log = _run_loop({S.DEFER: "1"}, [0, 1, 2, 3], 1)
    assert not any(e[0] == "fold_rows" for e in log)
    assert all(e[3] == () for e in log if e[0] == "combine_mix")
    assert all(dict(e[4])["decode_fold"] is False for e in log if e[0] == "layer")
    assert [e[3] for e in log if e[0] == "layer"] == [False, True, True, False]
    _, log = _run_loop({S.DEFER: "1", S.FOLD: "1"}, [0, 1, 2, 3], 1)
    hands = [e for e in log if e[0] == "combine_mix"]
    assert hands == [("combine_mix", "mlp1", "attn2", ()), ("combine_mix", "mlp2", "attn3", ()),
                     ("combine_mix", "mlp3", "mixer", (("prefill_fused", False),))]


class _Stop(Exception):
    pass


@pytest.mark.parametrize("decode_fold", [False, True])
def test_decoder_layer_hand_off_keywords(decode_fold):
    """The REAL decoder-layer forward: a fold hand-off asks combine_mix for prefill_fused=False; any other hand-off
    makes exactly v8's call (reread only)."""
    seen = []

    class _Rec:
        def combine_mix(self, moe, residual, next_hc, **kw):
            seen.append(kw)
            raise _Stop

    me = SimpleNamespace(ple=None, attn_hyper_connection=object())
    with S.env_state(), pytest.raises(_Stop):
        Qwen4ExpDecoderLayer.forward(me, None, None, torch.zeros(4, dtype=torch.int64), None, None, None, None,
                                     None, None, None, pending=(None, None, _Rec()), decode_fold=decode_fold)
    want = {"reread": False}
    if decode_fold:
        want["prefill_fused"] = False
    assert seen == [want]


# ---------------------------------------------------------------- engagement (VLLM_TPU_HC_DECODE_COUNT)
def test_register_hc_fold_counts_handoffs():
    def stage(fold, last=True):
        with S.env_state(**dict(S.SERVED, **({S.FOLD: "1"} if fold else {}))), S.core_builders():
            st = S.FoldStage(SMALL, layers=4, hooks=False)
        st.last_stage = last
        return st

    E.reset()
    Qwen4ExpBackbone._register_hc_fold(stage(False))
    assert E.fold_handoffs() == 0
    with S.env_state(**dict(S.SERVED, **{S.FOLD: "1"})):
        Qwen4ExpBackbone._register_hc_fold(stage(True))
        assert E.fold_snapshot()["layers"] == 2 and E.fold_snapshot()["final"] == 1
        Qwen4ExpBackbone._register_hc_fold(stage(True, last=False))
        assert E.fold_snapshot()["layers"] == 2 and E.fold_snapshot()["final"] == 0


def test_fold_not_counted_without_registration():
    with S.env_state(**S.SERVED):
        E.record_built(True)
        assert E.count_step(96, 0, mk=False) is True
        assert E.fold_snapshot()["steps"] == 0 and E.fold_handoffs() == 0


def test_fold_count_and_log_line(caplog):
    with S.env_state(**S.SERVED):
        E.record_built(True)
        E.register_fold(2, 1)
        with caplog.at_level(logging.INFO):
            for rows, mk in ((96, False), (128, False), (64, True), (160, False), (256, False), (96, False),
                             (80, False), (48, False)):
                E.count_step(rows, 8, mk=mk)
    snap = E.fold_snapshot()
    assert snap["folded"] == 5 and snap["mk"] == 1 and snap["steps"] == 8 and snap["handoffs"] == 15
    assert snap["by_rows"] == {48: 1, 80: 1, 96: 2, 128: 1}
    lines = [r.getMessage() for r in caplog.records if "DEC-HC" in r.getMessage()]
    assert lines == [
        "DEC-HC engagement: fused decode HC on 1 HC modules (UP_T 1); fused-eligible forwards 6 of 8 on this rank",
        "DEC-HC fold engagement: folded forwards 5 of 8 target forwards on this rank (megakernel 1); hand-offs per "
        "folded forward 3 (2 layer + 1 final mixer); folded hand-offs 15; folded forwards by rows "
        "{48: 1, 80: 1, 96: 2, 128: 1}",
    ], lines


def test_no_fold_line_without_registration(caplog):
    E.record_built(True)
    with S.env_state(**S.SERVED), caplog.at_level(logging.INFO):   # HCGATE 128: 160 rows not fused-eligible
        for rows in (96, 160):
            E.count_step(rows, 2)
    lines = [r.getMessage() for r in caplog.records if "DEC-HC" in r.getMessage()]
    assert lines == ["DEC-HC engagement: fused decode HC on 1 HC modules (UP_T 1); fused-eligible forwards 1 of 2 "
                     "on this rank"]
    assert E.snapshot() == {"modules": 1, "up_t": 1, "steps": 2, "eligible": 1}


# ---------------------------------------------------------------- D14 / MTP / megakernel untouched
def _names(src):
    tree = ast.parse(src)
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
        {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
        {n.arg for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg} | \
        {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


FOLD_NAMES = {"TPU_QWEN4_HC_DECODE_FOLD", "decode_fold", "decode_fold_rows", "fold_here", "prefill_fused"}


def test_d14_mtp_and_megakernel_paths_never_read_the_fold():
    import textwrap

    import vllm_torchtpu
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    for fn in (Qwen4ExpBackbone.forward_segments, Qwen4ExpBackbone.forward_wholestep, Qwen4ExpDecoderLayer.forward_rows):
        hit = _names(textwrap.dedent(inspect.getsource(fn))) & FOLD_NAMES
        assert not hit, (fn.__name__, hit)
    assert not _names(open(inspect.getsourcefile(mtp)).read()) & FOLD_NAMES
    root = os.path.join(os.path.dirname(vllm_torchtpu.__file__), "megakernel_runtime")
    n = 0
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".py"):
                n += 1
                assert not _names(open(os.path.join(dp, f)).read()) & FOLD_NAMES, f
    assert n > 10
    # The stock loop is the only reader of the key among the model's forwards.
    assert "TPU_QWEN4_HC_DECODE_FOLD" in _names(textwrap.dedent(inspect.getsource(Qwen4ExpBackbone._forward_impl)))


def test_mtp_style_modules_unaffected_with_the_key_on():
    """The MTP draft's HCs (built under the key, then disable_decode_fused) keep the unfused mix and no combine-mix
    op; its layer is called without pending / decode_fold, so its mix and combine are today's."""
    with S.env_state(**dict(S.SERVED, **{S.FOLD: "1"})), S.core_builders():
        layer_hc, mixer = S.build_hc(SMALL, 7), S.build_hc(SMALL, 8, inject=False)
    with S.env_state(**S.SERVED), S.core_builders():
        ref_hc, ref_mixer = S.build_hc(SMALL, 7), S.build_hc(SMALL, 8, inject=False)
    for m in (layer_hc, mixer, ref_hc, ref_mixer):
        m.disable_decode_fused()
        assert m._decode_mix_op is None and m._decode_combine_mix_op is None
    x = torch.randn(96, 4 * SMALL["hidden"], generator=torch.Generator().manual_seed(3)).bfloat16()
    with S.env_state(**dict(S.SERVED, **{S.FOLD: "1"})), torch.no_grad():
        a, (_, ga) = layer_hc.mix(x)
        b = mixer.mix(x)[0]
    with S.env_state(**S.SERVED), torch.no_grad():
        ra, (_, rga) = ref_hc.mix(x)
        rb = ref_mixer.mix(x)[0]
    assert torch.equal(a, ra) and torch.equal(ga, rga) and torch.equal(b, rb)


# ---------------------------------------------------------------- traced graphs
def _graph(env, rows, dynamic):
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm.code)
        return gm

    st, _ = S.build_stage(env, SMALL, layers=4, hooks=False)
    x, ids, valid = S.stage_inputs(rows, SMALL["hidden"])
    torch._dynamo.reset()
    with S.env_state(**env), torch.no_grad():
        if dynamic:
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(ids, 0)
            torch._dynamo.mark_dynamic(valid, 0)
        torch.compile(st, backend=backend, dynamic=dynamic)(x, ids, valid)
    torch._dynamo.reset()
    return graphs


@pytest.mark.parametrize("rows", [160, 192, 224, 256])
def test_xla_rows_trace_the_unfolded_graph(rows):
    """Served gates (HCGATE 128, HCX 256): at 160-256 rows the fold is an exact no-op, op for op (static trace)."""
    off = _graph(S.SERVED, rows, False)
    on = _graph(dict(S.SERVED, **{S.FOLD: "1"}), rows, False)
    print(f"\nPBL_HC GRAPH rows {rows}: {len(off)} graph(s) off, {len(on)} on; identical {on == off}; "
          f"ops {sum(g.count(chr(10)) for g in off)}")
    assert off and on == off


def test_dynamic_trace_at_160_is_unfolded_info():
    """INFO: the dynamic-shape trace at 160 rows (the trace that serves 160-256 under the gates) is the same."""
    off = _graph(S.SERVED, 160, True)
    on = _graph(dict(S.SERVED, **{S.FOLD: "1"}), 160, True)
    print(f"\nPBL_HC GRAPH dynamic rows 160: identical {on == off} ({len(off)} / {len(on)} graphs)")
    assert on == off


def _stub(mod):
    """Traceable torch stand-ins with the registered ops' contracts (same weights, same order)."""
    hidden = mod.hidden_size

    def mix_op(x, norm, down, up, inject=None):
        normed = x * (1.0 + norm)
        mixed = F.linear(F.silu(F.linear(normed, down)), up)[:, :hidden]
        return mixed if inject is None else (mixed, F.linear(normed, inject).float())

    def combine_mix_op(residual, block, gate, *weights):
        combined = residual + block.repeat(1, 4) * gate.repeat_interleave(hidden, dim=-1).to(block.dtype)
        if len(weights) == 3:
            return mix_op(combined, *weights), combined
        mixed, injection = mix_op(combined, *weights)
        return mixed, injection, combined

    mod._decode_mix_op = mix_op
    mod._decode_combine_mix_op = combine_mix_op
    return mod


def test_folded_trace_uses_weights_in_one_order():
    """With the fold, a fused decode bucket and an unfused bucket still first use every weight in one order through
    the REAL stage loop (Benliang's test_folded_traces_use_weights_in_one_order, on our loop). The shape-variant
    hand-over refuses a bucket whose subgraph inputs moved (dynamo_input_order.py)."""
    rows = 8
    with S.env_state(**dict(S.SERVED, **{S.FOLD: "1"})), S.core_builders():
        fused = S.FoldStage(SMALL, layers=4, tiles=False, hooks=False)
    for m in fused.modules():
        if isinstance(m, H.GatedResidual):
            _stub(m)
    with S.env_state(), S.core_builders():
        unfused = S.FoldStage(SMALL, layers=4, tiles=False, hooks=False)
    x, ids, valid = S.stage_inputs(rows, SMALL["hidden"])
    with S.env_state(**dict(S.SERVED, **{S.FOLD: "1"})):
        assert fused.hyper_connection_mixer.decode_fold_rows(rows, torch.bfloat16)
        a = parameters_by_first_use(fused, x, ids, valid)
    with S.env_state():
        b = parameters_by_first_use(unfused, x, ids, valid)
    print(f"\nPBL_HC ORDER {len(a)} parameters, same order {a == b}")
    assert len(a) == 4 * (2 * 4 + 2) + 3 and a == b


def test_runner_passes_megakernel_hits_to_the_fold_count():
    """The runner (host, log-only) keeps its megakernel-hit mark under TPU_QWEN4_HC_DECODE_FOLD and hands it to
    hc_engagement.count_step, so forwards the megakernel took (D14) are not counted as folded."""
    import vllm_torchtpu

    src = open(os.path.join(os.path.dirname(vllm_torchtpu.__file__), "runner", "tpu_runner.py")).read()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "count_step" and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "hc_engagement"]
    assert len(calls) == 1, len(calls)
    mk = [k for k in calls[0].keywords if k.arg == "mk"]
    assert mk and "_q38_mk_hit" in ast.unparse(mk[0].value), ast.unparse(calls[0])
    marks = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "q38_mk_mark" for t in n.targets)]
    assert len(marks) == 1 and "TPU_QWEN4_HC_DECODE_FOLD" in ast.unparse(marks[0].value), [ast.unparse(m) for m in marks]
    sets = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.If) and "q38_mk_mark" in ast.unparse(n.test)]
    assert len(sets) == 2 and any("_q38_mk_hit = True" in s for s in sets) and any("_q38_mk_hit = False" in s
                                                                                  for s in sets), sets
