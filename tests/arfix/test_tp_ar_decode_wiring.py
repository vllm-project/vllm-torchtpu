# SPDX-License-Identifier: Apache-2.0
"""q38 AR-FIX: wiring of the decode TP all-reduce kernel (VLLM_TPU_DECODE_TP_AR), CPU, no TPU.

- Flag off (the default): mode '', no compile-key factor, ``arm`` is a no-op, and both call sites
  (``sequence_parallel.reduce`` on the replicated path, ``Qwen4ExpQSA.forward_with_indices(reduce=True)``) return
  exactly what the stock ``tensor_model_parallel_all_reduce`` returns (the same object), so the traced graph is
  unchanged.
- Flag set and armed: bf16 2-D [rows, hidden] partials go to the armed op; any other dtype, rank or width keeps the
  stock call; the sequence-parallel reduce-scatter is never touched.
- The compile key: unset leaves ``_tpu_compile_env_factors`` unchanged; set adds exactly this lever's two factors.
- Arming refuses (raises) on TP world != 8, EP off, EP ranks != TP ranks, and ranks that disagree; a successful arm
  builds one jax_op over the EP mesh named ``pallas::q38_tp_ar_decode_<digest>``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.kernels.collectives import tp8_ar_decode_fn as fnmod
from vllm_torchtpu.layers.adapter import tp_ar_decode as ar
from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp

ENV, CAP = fnmod.ENV, fnmod.MAX_ROWS_ENV


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(CAP, raising=False)
    monkeypatch.setattr(ar, "_ARMED", None)
    yield


def _stock(monkeypatch, module):
    calls = []

    def stock(x):
        calls.append(x)
        return ("stock", x)

    monkeypatch.setattr(module, "tensor_model_parallel_all_reduce", stock)
    return calls


def _armed(monkeypatch, hidden=2560):
    calls = []

    def op(x):
        calls.append(x)
        return ("kernel", x)

    monkeypatch.setattr(ar, "_ARMED", (op, hidden))
    return calls


# ---------------------------------------------------------------- flag off


def test_flag_off_is_inert(monkeypatch):
    assert fnmod.mode() == ""
    assert fnmod.compile_factors() == {}
    assert ar.arm(None) is False
    assert ar._ARMED is None
    assert ar.use_kernel(torch.zeros(16, 2560, dtype=torch.bfloat16)) is False


def test_flag_off_sp_reduce_is_the_stock_call(monkeypatch):
    calls = _stock(monkeypatch, sp)
    monkeypatch.setattr(sp, "tp_world_size", lambda: 8)
    x = torch.ones(16, 2560, dtype=torch.bfloat16)
    out = sp.reduce(x, False)
    assert out == ("stock", x) and calls == [x]


def test_sp_reduce_scatter_untouched_even_when_armed(monkeypatch):
    _armed(monkeypatch)
    monkeypatch.setattr(sp, "tp_world_size", lambda: 8)
    seen = []
    monkeypatch.setattr(sp, "tensor_model_parallel_reduce_scatter", lambda x, d: seen.append(d) or ("rs", x))
    x = torch.ones(16, 2560, dtype=torch.bfloat16)
    assert sp.reduce(x, True) == ("rs", x) and seen == [0]


def test_sp_reduce_tp1_returns_input(monkeypatch):
    _armed(monkeypatch)
    monkeypatch.setattr(sp, "tp_world_size", lambda: 1)
    x = torch.ones(16, 2560, dtype=torch.bfloat16)
    assert sp.reduce(x, False) is x


# ---------------------------------------------------------------- flag on (armed)


@pytest.mark.parametrize("rows", [16, 48, 64, 24, 1])
def test_armed_sp_reduce_takes_the_kernel_for_bf16_rows(monkeypatch, rows):
    stock = _stock(monkeypatch, sp)
    kern = _armed(monkeypatch)
    monkeypatch.setattr(sp, "tp_world_size", lambda: 8)
    x = torch.ones(rows, 2560, dtype=torch.bfloat16)
    assert sp.reduce(x, False) == ("kernel", x)
    assert kern == [x] and stock == []


@pytest.mark.parametrize(
    "shape,dtype",
    [((16, 2560), torch.float32), ((16, 1024), torch.bfloat16), ((2, 16, 2560), torch.bfloat16), ((2560,), torch.bfloat16)],
)
def test_armed_other_tensors_keep_the_stock_call(monkeypatch, shape, dtype):
    stock = _stock(monkeypatch, sp)
    kern = _armed(monkeypatch)
    monkeypatch.setattr(sp, "tp_world_size", lambda: 8)
    x = torch.ones(*shape, dtype=dtype)
    assert sp.reduce(x, False) == ("stock", x)
    assert kern == [] and stock == [x]


def _qsa_call(monkeypatch, reduce=True, tp_size=8, proj_merge=False):
    from vllm_torchtpu.models.vllm.qwen4_exp import qsa

    stock = _stock(monkeypatch, qsa)
    part = torch.ones(16, 2560, dtype=torch.bfloat16)
    # BLSTACK-H8 (port onto Benliang Wang's d523c2ce0): his forward_with_indices passes project_shares= to
    # attend_with_indices and, under VLLM_TPU_QSA_DCP_MERGE=proj (_proj_merge), projects with o_proj_group_weight.
    fake = SimpleNamespace(
        attend_with_indices=lambda *a, **k: (part, "sel"),
        o_proj=lambda t: t,
        tp_size=tp_size,
    )
    if proj_merge:
        fake._proj_merge = True
        fake.o_proj_group_weight = torch.eye(2560, dtype=torch.bfloat16)
    out = qsa.Qwen4ExpQSA.forward_with_indices(fake, None, None, None, reduce=reduce)
    return out, stock, part


def test_flag_off_qsa_reduce_is_the_stock_call(monkeypatch):
    (out, sel), stock, part = _qsa_call(monkeypatch)
    assert out == ("stock", part) and sel == "sel" and stock == [part]


def test_armed_qsa_reduce_takes_the_kernel(monkeypatch):
    kern = _armed(monkeypatch)
    (out, sel), stock, part = _qsa_call(monkeypatch)
    assert out == ("kernel", part) and stock == [] and kern == [part]


def test_flag_off_qsa_proj_merge_reduce_is_the_stock_call(monkeypatch):
    (out, sel), stock, part = _qsa_call(monkeypatch, proj_merge=True)
    assert out[0] == "stock" and torch.equal(out[1], part) and len(stock) == 1 and sel == "sel"


def test_armed_qsa_proj_merge_reduce_takes_the_kernel(monkeypatch):
    """His served decode runs VLLM_TPU_QSA_DCP_MERGE=proj: the TP sum of the o_proj_group_weight shares goes to H8."""
    kern = _armed(monkeypatch)
    (out, sel), stock, part = _qsa_call(monkeypatch, proj_merge=True)
    assert out[0] == "kernel" and torch.equal(out[1], part) and stock == [] and len(kern) == 1


def test_armed_qsa_partial_path_untouched(monkeypatch):
    kern = _armed(monkeypatch)
    (out, sel), stock, part = _qsa_call(monkeypatch, reduce=False)
    assert out is part and stock == [] and kern == []


# ---------------------------------------------------------------- flag parsing and compile key


@pytest.mark.parametrize("value", ["h9", "1", "on", "H8"])
def test_bad_mode_raises(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError):
        fnmod.mode()


@pytest.mark.parametrize("value", ["24", "0", "512", "x"])
def test_bad_cap_raises(monkeypatch, value):
    monkeypatch.setenv(CAP, value)
    with pytest.raises(ValueError):
        fnmod.max_rows()


def test_compile_key_unset_unchanged_set_moves(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler as tc

    base = tc._tpu_compile_env_factors()
    assert ENV not in base and CAP not in base
    monkeypatch.setenv(ENV, "h8")
    on = tc._tpu_compile_env_factors()
    assert on.pop(ENV) == "h8" and on.pop(CAP) == "256"
    assert on == base
    monkeypatch.setenv(ENV, "h8x")
    monkeypatch.setenv(CAP, "64")
    on2 = tc._tpu_compile_env_factors()
    assert (on2[ENV], on2[CAP]) == ("h8x", "64")


# ---------------------------------------------------------------- arming


class _Group(SimpleNamespace):
    pass


def _arm_env(monkeypatch, tp_world=8, ep_ranks=None, signatures_differ=False):
    pytest.importorskip("torch_tpu._internal.pallas")
    import torch_tpu._internal.pallas as ttp
    import vllm.distributed as vd

    from vllm_torchtpu.distributed import ep_mesh

    monkeypatch.setenv(ENV, "h8")
    tp = _Group(world_size=tp_world, ranks=list(range(tp_world)), cpu_group=None)
    monkeypatch.setattr(vd, "get_tp_group", lambda: tp)
    ep = None if ep_ranks == () else _Group(world_size=8, ranks=list(ep_ranks or range(8)))
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: ep)
    order = (0, 1, 4, 5, 6, 7, 2, 3)

    class _Dev:
        def __init__(self, i):
            self.id = i

    class _Mesh:
        shape = {ep_mesh.EP_AXIS_NAME: 8}
        axis_names = (ep_mesh.EP_AXIS_NAME,)

        class devices:  # noqa: N801
            flat = [_Dev(i) for i in order]

    monkeypatch.setattr(ep_mesh, "build_ep_mesh", lambda: _Mesh())
    monkeypatch.setattr(ep_mesh, "ep_device_ids", lambda: order)
    monkeypatch.setattr(ep_mesh, "_ep_mesh_order", lambda ids: tuple(range(len(ids))))

    def gather(out, sig, group=None):
        for i in range(len(out)):
            out[i] = sig if not (signatures_differ and i == 3) else ("other",)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    built = []

    class _Op:
        def __init__(self, name, fn, **kw):
            built.append((name, fn, kw))

        def register_fake(self, f):
            self.fake = f

        def __call__(self, x):
            return ("kernel", x)

    monkeypatch.setattr(ttp, "jax_op", _Op)
    runner = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=2560)))
    return runner, built


def test_arm_success_builds_one_op(monkeypatch):
    runner, built = _arm_env(monkeypatch)
    assert ar.arm(runner) is True
    assert len(built) == 1 and built[0][0].startswith("pallas::q38_tp_ar_decode")
    assert ar._ARMED is not None and ar._ARMED[1] == 2560
    assert ar.use_kernel(torch.zeros(16, 2560, dtype=torch.bfloat16)) is True


@pytest.mark.parametrize(
    "kw,msg",
    [({"tp_world": 4}, "TP world 4"), ({"ep_ranks": ()}, "EP is off"), ({"ep_ranks": tuple(range(8, 16))}, "not the TP ranks")],
)
def test_arm_refuses(monkeypatch, kw, msg):
    runner, built = _arm_env(monkeypatch, **kw)
    with pytest.raises(RuntimeError, match=msg):
        ar.arm(runner)
    assert ar._ARMED is None and built == []


def test_arm_refuses_when_ranks_disagree(monkeypatch):
    runner, built = _arm_env(monkeypatch, signatures_differ=True)
    with pytest.raises(RuntimeError, match="disagree"):
        ar.arm(runner)
    assert ar._ARMED is None
