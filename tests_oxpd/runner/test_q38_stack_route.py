# SPDX-License-Identifier: Apache-2.0
"""q38 L2 STACK-MARK (VLLM_TPU_Q38_STACK_ROUTE_COUNT): per-bucket route markers, host only, log-only.

  - engagement: at 96 / 128 / 160 / 192 / 224 / 256 rows under HCGATE 128:256 + O1s (min rows 48) + GDN window tile,
    the counts equal both the code's own predicates (hc_engagement.would_fuse, Qwen4ExpQSA._request_shard_applies,
    the window-tile env) and a hand-written table; flags-off cases count 0; every bucket logs; megakernel forwards
    count in fwd / mk only; a forward without context_lens reads o1s n/a;
  - q38 L1-SPLIT merge (v5 x L2 stk64d, both lines): the table is v5's 8 / 16 / 32 / 48 / 64 (D14 buckets) + 96..256 +
    1024 rows, with stk64d's L3-HCGATE + T95u fix present: hcfused = modules > 0 and decode_fused_allows (band 128:256:
    rows <= 128 or rows % 128 == 0) and a decode token tile (<= hc_decode.MAX_TOKENS 512) -> 1 at 8..128 and 256, 0 at
    160 / 192 / 224 and 1024; the setup line reads "HCGATE limits (128, 256)". O1s (Qwen4ExpQSA._request_shard_applies
    -> qsa_op.request_shard_applies: world > 1, rows % world == 0, rows <= slots * max_query_len, rows >= min_rows): at
    TP 8, max_query_len 4, 64 slots, min rows 48 -> 0 at 8 / 16 / 32 rows, 1 at 48..256, 0 at 1024;
  - the runner calls the counter once, after the target forward's dispatch, gated by the env;
  - identity: every log-only env here (LOG_ONLY_ENVS: STACK-MARK's now, others later) is compile-ignored and leaves the
    TPU compile hash and a small decode graph's lowered text byte-identical.
"""

import importlib.util
import inspect
import pathlib
import re
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_torchtpu.models.vllm.qwen4_exp import hc_engagement, stack_route
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

ROWS = (8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 1024)
SLOTS = 64  # request slots of the forward (context_lens.shape[0])
LOG_ONLY_ENVS = [
    ("VLLM_TPU_Q38_STACK_ROUTE_COUNT", "50"),  # q38 L2 STACK-MARK
    ("VLLM_TPU_LOG_STEP_BUCKETS", "50"),  # step-bucket log (STEPBUCKETS2's period needs it on)
    ("VLLM_TPU_LOG_STEP_BUCKETS_PERIOD", "1"),  # q38 L5 STEPBUCKETS2
]
# q38 L1-SPLIT hand table (HCGATE 128:256 with the T95u fix), O1s min rows 48 (TP 8, max_query_len 4, 64 slots), window
# tile on: (o1s, hcfused, gdnwt) per forward. Band: rows <= 128 or rows % 128 == 0; decode_token_tile: R <= 128 -> R;
# 256 -> 128; 1024 > MAX_TOKENS 512 -> None. O1s: rows >= 48 and rows <= 64 * 4 = 256. Written out, not computed.
WANT_ON = {8: (0, 1, 1), 16: (0, 1, 1), 32: (0, 1, 1), 48: (1, 1, 1), 64: (1, 1, 1), 96: (1, 1, 1), 128: (1, 1, 1),
           160: (1, 0, 1), 192: (1, 0, 1), 224: (1, 0, 1), 256: (1, 1, 1), 1024: (0, 0, 1)}
# HCGATE unset: every bucket with a decode tile fuses (all but 1024 here).
WANT_HC_UNSET = {r: int(r != 1024) for r in ROWS}


@pytest.fixture(autouse=True)
def _clean():
    stack_route.reset()
    hc_engagement.reset()
    yield
    stack_route.reset()
    hc_engagement.reset()


def qsa_layer(mode=True, min_rows=48, tp=8, max_query_len=4):
    """A target QSA layer as bound (no weights): its own _request_shard_applies decides O1s."""
    layer = object.__new__(Qwen4ExpQSA)
    layer.__dict__.update(decode_request_shard=mode, tp_size=tp, max_query_len=max_query_len,
                          request_shard_min_rows=min_rows, _request_shard_dcp_op=object() if mode else None)
    return layer


class Model:
    def __init__(self, layer):
        self.layer = layer

    def named_modules(self):
        yield "model", self
        yield "mtp.0.self_attn", qsa_layer(True, 0)  # the draft never takes O1s: must be skipped
        if self.layer is not None:
            yield "model.layers.3.self_attn", self.layer


def kwargs(slots=SLOTS):
    return {"context_lens": SimpleNamespace(shape=(slots,)), "input_ids": None}


def drive(monkeypatch, *, hc_band="128:256", modules=48, window_tile="1", layer=None, steps=2, every=1000,
          model_kwargs=None):
    if hc_band is None:
        monkeypatch.delenv("VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS", hc_band)
    if window_tile is None:
        monkeypatch.delenv("VLLM_TPU_GDN_WINDOW_TILE", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_GDN_WINDOW_TILE", window_tile)
    for _ in range(modules):
        hc_engagement.record_built(False)
    model = Model(qsa_layer() if layer is None else layer)
    for _ in range(steps):
        for rows in ROWS:
            stack_route.count_step(model, kwargs() if model_kwargs is None else model_kwargs, rows, every)
    return model


def test_counts_match_the_predicates_and_the_table(monkeypatch):
    model = drive(monkeypatch)
    snap = stack_route.snapshot()
    assert sorted(snap) == list(ROWS)
    for rows in ROWS:
        fwd, o1s, hc, wt, mk = snap[rows]
        pred = (int(model.layer._request_shard_applies(SimpleNamespace(shape=(rows,)),
                                                      SimpleNamespace(context_lens=SimpleNamespace(shape=(SLOTS,))))),
                int(hc_engagement.would_fuse(rows)), 1)
        assert fwd == 2 and mk == 0, (rows, snap[rows])
        assert (o1s, hc, wt) == (2 * pred[0], 2 * pred[1], 2 * pred[2]), (rows, snap[rows], pred)
        assert (o1s, hc, wt) == tuple(2 * x for x in WANT_ON[rows]), (rows, snap[rows])


@pytest.mark.parametrize("case", ["o1s_off", "o1s_gate_above", "hc_off", "hc_unset", "wt_off"])
def test_flags_off_cases_count_zero(monkeypatch, case):
    layer = {"o1s_off": qsa_layer(mode=False), "o1s_gate_above": qsa_layer(min_rows=512)}.get(case)
    drive(monkeypatch, layer=layer, modules=0 if case == "hc_off" else 48,
          hc_band=None if case == "hc_unset" else "128:256", window_tile=None if case == "wt_off" else "1")
    snap = stack_route.snapshot()
    col = {"o1s_off": 1, "o1s_gate_above": 1, "hc_off": 2, "wt_off": 3}.get(case)
    for rows in ROWS:
        assert snap[rows][0] == 2
        if col is not None:
            assert snap[rows][col] == 0, (case, rows, snap[rows])
    if case == "hc_unset":  # HCGATE off: every bucket with a decode tile fuses (all but 1024 here)
        assert all(snap[r][2] == 2 * WANT_HC_UNSET[r] for r in ROWS), snap


def test_line_lists_every_bucket_and_the_setup(monkeypatch, caplog):
    lines = []
    monkeypatch.setattr(stack_route, "_logger", lambda: SimpleNamespace(info=lambda fmt, *a: lines.append(fmt % a)))
    drive(monkeypatch, steps=1, every=len(ROWS))
    assert len(lines) == 2 and lines[0].startswith("q38 STACK route rank ") and " setup: o1s layer " in lines[0]
    assert "model.layers.3.self_attn (mode True, tp 8, max_query_len 4, min_rows 48, op True)" in lines[0]
    assert "HCGATE limits (128, 256)" in lines[0] and "window_tile env 1" in lines[0]
    want = " | ".join("rows %d: fwd 1 mk 0 o1s %d hcfused %d gdnwt(env) %d" % ((r,) + WANT_ON[r]) for r in ROWS)
    assert re.fullmatch(r"q38 STACK route rank -?\d+: " + re.escape(want), lines[1]), lines[1]


def test_megakernel_forwards_count_in_fwd_and_mk_only(monkeypatch):
    drive(monkeypatch, steps=0)
    model = Model(qsa_layer())
    for rows in (96, 256):
        assert stack_route.count_step(model, kwargs(), rows, 1000, mk=True) == (0, 0, 0)
    assert stack_route.snapshot() == {96: [1, 0, 0, 0, 1], 256: [1, 0, 0, 0, 1]}


def test_no_context_lens_reads_na(monkeypatch):
    drive(monkeypatch, steps=1, model_kwargs={})
    assert all(v[1] == stack_route.O1S_NA for v in stack_route.snapshot().values())


def test_zero_forwards_logs_nothing_and_counts_nothing(monkeypatch):
    lines = []
    monkeypatch.setattr(stack_route, "_logger", lambda: SimpleNamespace(info=lambda fmt, *a: lines.append(fmt % a)))
    assert stack_route.count_step(Model(qsa_layer()), kwargs(), 96, 0) is None  # env 0 = off
    assert stack_route.snapshot() == {} and lines == []


def test_runner_counts_once_after_dispatch_gated_by_the_env():
    from vllm_torchtpu.runner import tpu_runner

    src = inspect.getsource(tpu_runner)
    calls = [m.start() for m in re.finditer(r"stack_route\.count_step\(", src)]
    assert len(calls) == 1, calls
    gate = src.rfind("if envs.VLLM_TPU_Q38_STACK_ROUTE_COUNT:", 0, calls[0])
    dispatch = src.rfind("hidden_states, aux_hidden_states = self.forward_model(", 0, calls[0])
    assert 0 <= dispatch < gate < calls[0] and calls[0] - gate < 400, (dispatch, gate, calls[0])
    fwd = inspect.getsource(tpu_runner.TPUModelRunner.forward_model)
    assert "self._q38_mk_hit = True" in fwd and fwd.index("self._q38_mk_hit = False") < fwd.index("try_forward(")


@pytest.mark.parametrize("name,value", LOG_ONLY_ENVS)
def test_log_only_env_keeps_the_compile_hash(name, value, monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert name in envs.environment_variables and name in _TPU_COMPILE_ENV_IGNORED
    spec = importlib.util.spec_from_file_location(
        "q38_stackmark_ckey", pathlib.Path(__file__).parents[1] / "compilation/test_tpu_compile_key_registration.py")
    ckey = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ckey)
    assert ckey._hash_with({name: value}) == ckey._hash_with({name: None})


def _lowered_decode_step():
    """A small decode graph: one pooled QSA verify layer (indexer / attention stubbed as the fused-write tests do),
    lowered for TPU7x deviceless."""
    import jax
    import jax.numpy as jnp
    from jax._src.mesh import AbstractDevice, AbstractMesh
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

    spec = importlib.util.spec_from_file_location(
        "q38_stackmark_fkw", pathlib.Path(__file__).parents[1] / "kernels/test_qwen4_exp_fused_kv_write.py")
    fkw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fkw)
    dtype, options = fkw.LAYOUTS["fp8_paired_slot1_packed_k"]
    args = fkw.make_step(np.random.default_rng(5), *fkw.CASES["verify"], dtype)
    shapes = [jax.ShapeDtypeStruct(np.shape(a), a.dtype) for a in args]
    jax.clear_caches()
    step = jax.jit(pooled_qsa.pooled_qsa_step, static_argnames=(
        "compress_ratio", "token_topk", "norm_eps", "rotary_dim", "k_scale", "v_scale", "compressed_slot",
        "main_slot", "max_query_len"))
    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        return step.trace(*shapes, compress_ratio=fkw.RATIO, token_topk=fkw.TOPK, norm_eps=1e-6,
                          rotary_dim=fkw.ROTARY, max_query_len=4, k_scale=0.375, v_scale=2.5,
                          **options).lower(lowering_platforms=("tpu",)).as_text()


@pytest.mark.parametrize("name,value", LOG_ONLY_ENVS)
def test_log_only_env_keeps_a_decode_graph_byte_identical(name, value, monkeypatch):
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(pooled_qsa, "select_blocks",
                        lambda query, *a, block_topk, **k: jnp.zeros((len(query), block_topk), jnp.int32))
    monkeypatch.setattr(pooled_qsa, "qsa_attention",
                        lambda q, gate, blocks, lengths, table, pool, *a, **k: (jnp.zeros(q.shape, jnp.bfloat16), pool))
    monkeypatch.delenv(name, raising=False)
    off = _lowered_decode_step()
    monkeypatch.setenv(name, value)
    on = _lowered_decode_step()
    assert len(off) > 1000 and on == off


def test_all_log_only_envs_together_keep_the_compile_hash():
    """The composed car sets every log-only env at once."""
    spec = importlib.util.spec_from_file_location(
        "q38_stackmark_ckey_all", pathlib.Path(__file__).parents[1] / "compilation/test_tpu_compile_key_registration.py")
    ckey = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ckey)
    on = {name: value for name, value in LOG_ONLY_ENVS}
    assert ckey._hash_with(on) == ckey._hash_with({name: None for name in on})


def test_all_log_only_envs_together_keep_a_decode_graph_byte_identical(monkeypatch):
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(pooled_qsa, "select_blocks",
                        lambda query, *a, block_topk, **k: jnp.zeros((len(query), block_topk), jnp.int32))
    monkeypatch.setattr(pooled_qsa, "qsa_attention",
                        lambda q, gate, blocks, lengths, table, pool, *a, **k: (jnp.zeros(q.shape, jnp.bfloat16), pool))
    for name, _ in LOG_ONLY_ENVS:
        monkeypatch.delenv(name, raising=False)
    off = _lowered_decode_step()
    for name, value in LOG_ONLY_ENVS:
        monkeypatch.setenv(name, value)
    on = _lowered_decode_step()
    assert len(off) > 1000 and on == off
