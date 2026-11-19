# SPDX-License-Identifier: Apache-2.0
"""q38 L2 (decode plan v1 D34): VLLM_TPU_QSA_DCP_HEAD_A2A=xla on the FINAL tree, where ``_dcp_finish`` also carries
VLLM_TPU_QSA_DCP_ONE_GATHER (in FINAL's decode env). Routing: "xla" takes the head-slice all-to-all and never gathers
(ONE_GATHER or not); "" keeps FINAL's gather path exactly (ONE_GATHER on -> one packed gather). Every DCP op of the
decode engine (target ``build_pooled_qsa_dcp_op``, the MTP draft's DCP ops, O1s's request-shard and plain ops) ends in
``_dcp_finish``, so one switch covers them. Numerics are DEC-COLL's ``test_dcp_head_a2a.py`` (bitwise today's)."""

import inspect

import pytest


def _wire(monkeypatch, head_a2a, one_gather):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    calls = []
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_DCP_HEAD_A2A", head_a2a, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_DCP_ONE_GATHER", one_gather, raising=False)

    def pack(out, lse, gate):
        calls.append(("pack", tuple(out.shape), tuple(lse.shape), tuple(gate.shape)))
        return torch.zeros(4 * out.shape[0], 8)

    def merge(recv, gate):
        calls.append(("merge", tuple(recv.shape)))
        return torch.zeros(gate.shape, dtype=torch.bfloat16)

    def a2a(x, group):
        calls.append(("a2a", tuple(x.shape), group))
        return x

    def gather(out, lse, group, one):
        calls.append(("gather", one))
        return out.unsqueeze(0).repeat(4, 1, 1, 1), lse.unsqueeze(0).repeat(4, 1, 1)

    def combine(out_pair, lse_pair, gate, rank):
        calls.append(("combine", tuple(out_pair.shape)))
        return torch.zeros(gate.shape, dtype=torch.bfloat16)

    monkeypatch.setattr(qsa_op, "_dcp_head_a2a_ops", lambda: (pack, merge))
    monkeypatch.setattr(qsa_op, "_all_to_all", a2a)
    monkeypatch.setattr(qsa_op, "_dcp_gather_partials", gather)
    tokens, heads, hd = 16, 3, 4
    out = torch.zeros(tokens, 4 * heads, hd)
    lse = torch.zeros(tokens, 4 * heads)
    qgate = torch.zeros(tokens, heads * 2 * hd)
    res = qsa_op._dcp_finish(out, lse, qgate, hd, torch.zeros(1, dtype=torch.int32), "PAIR", combine)
    assert tuple(res.shape) == (tokens, heads * hd)
    return [c[0] for c in calls], calls


def test_xla_takes_the_all_to_all_with_one_gather_on(monkeypatch):
    names, calls = _wire(monkeypatch, "xla", True)
    assert names == ["pack", "a2a", "merge"], names
    assert calls[1][2] == "PAIR"  # the DCP pair group, not the TP group


def test_xla_takes_the_all_to_all_with_one_gather_off(monkeypatch):
    names, _ = _wire(monkeypatch, "xla", False)
    assert names == ["pack", "a2a", "merge"], names


@pytest.mark.parametrize("one_gather", [True, False])
def test_unset_keeps_finals_gather_path(monkeypatch, one_gather):
    names, calls = _wire(monkeypatch, "", one_gather)
    assert names == ["gather", "combine"], names
    assert calls[0][1] is one_gather


def test_every_decode_dcp_op_ends_in_dcp_finish():
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    for build in (qsa_op.build_pooled_qsa_dcp_op, qsa_op.build_pooled_qsa_draft_dcp_op,
                  qsa_op.build_pooled_qsa_request_shard_dcp_op, qsa_op.build_pooled_qsa_request_plain_dcp_op):
        assert "_dcp_finish(" in inspect.getsource(build), build.__name__


def test_flag_registered_default_empty(monkeypatch):
    from vllm_torchtpu import envs as E

    name = "VLLM_TPU_QSA_DCP_HEAD_A2A"
    assert name in E.environment_variables
    monkeypatch.delenv(name, raising=False)
    assert E.environment_variables[name]() == ""
    monkeypatch.setenv(name, "xla")
    assert E.environment_variables[name]() == "xla"


def test_dcp_head_a2a_jax_ops_build_with_real_torch_tpu():
    """The pack / merge jax_ops build with the real torch_tpu: pallas.jax_op evaluates each step function's annotations
    (eval_str) at build time, and dcp_head_a2a.py uses ``from __future__ import annotations`` - a name in an annotation
    that is not a module global would raise NameError at engine init (the H8 failure on 5d368f166, J0b allonk)."""
    pytest.importorskip("torch_tpu")
    import inspect

    from vllm_torchtpu.kernels.collectives import dcp_head_a2a as A
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    for fn in (A.dcp_a2a_pack_step, A.dcp_a2a_merge_step):
        inspect.signature(fn, eval_str=True)  # the annotations resolve in the module's globals
    qsa_op._dcp_head_a2a_ops.cache_clear()
    pack, merge = qsa_op._dcp_head_a2a_ops()
    assert pack is not None and merge is not None
    assert qsa_op._dcp_head_a2a_ops() == (pack, merge)
