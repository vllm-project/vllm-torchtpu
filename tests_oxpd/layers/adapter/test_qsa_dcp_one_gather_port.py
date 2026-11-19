"""Port checks of D-MK's VLLM_TPU_QSA_DCP_ONE_GATHER (99df0bbfe, picked from 80b612e33's 36624958c) onto the P-c /
lever-3 chain (q38 DEC-FB-3 lever 4). D-MK's test_qsa_dcp_one_gather.py proves on == off bit for bit through
_dcp_finish with 2 -> 1 collectives; this adds:
1. flag off, _dcp_gather_partials runs exactly the two all-gathers _dcp_finish ran before the port (AST, against
   aa83ece72's lines 979-980, verbatim);
2. _dcp_finish passes bool(envs.VLLM_TPU_QSA_DCP_ONE_GATHER) (read while the DCP op is traced: hashed, see
   tests/compilation/test_tpu_compile_key_registration.py);
3. mixed out / lse dtypes fall back to the two gathers even with the flag on (the packing needs one dtype)."""
import ast
import inspect
import textwrap

import torch

from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

OLD = ("out_pair = pair_group.all_gather(out.unsqueeze(0).contiguous(), dim=0)\n"
       "lse_pair = pair_group.all_gather(lse.unsqueeze(0).contiguous(), dim=0)\n")


def _fn(f):
    return ast.parse(textwrap.dedent(inspect.getsource(f))).body[0]


def test_flag_off_branch_is_the_old_two_gathers():
    fn = _fn(qsa_op._dcp_gather_partials)
    first = next(n for n in fn.body if isinstance(n, ast.If))
    assert ast.unparse(first.test) == "not one_gather or out.dtype != lse.dtype"
    old = ast.parse(OLD).body
    assert [ast.dump(n) for n in first.body[:2]] == [ast.dump(n) for n in old]
    assert ast.unparse(first.body[2]) == "return (out_pair, lse_pair)"


def test_finish_passes_the_flag():
    fn = _fn(qsa_op._dcp_finish)
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_dcp_gather_partials"]
    assert len(calls) == 1
    assert ast.unparse(calls[0]) == "_dcp_gather_partials(out, lse, pair_group, bool(envs.VLLM_TPU_QSA_DCP_ONE_GATHER))"


class _Count:
    def __init__(self):
        self.n = 0

    def all_gather(self, x, dim=0):
        self.n += 1
        return torch.cat([x, x], dim=dim)


def test_mixed_dtypes_fall_back_to_two_gathers():
    out = torch.randn(4, 8, 128)
    lse = torch.randn(4, 8).to(torch.bfloat16)
    g = _Count()
    out_pair, lse_pair = qsa_op._dcp_gather_partials(out, lse, g, True)
    assert g.n == 2
    assert out_pair.dtype == torch.float32 and lse_pair.dtype == torch.bfloat16
    assert torch.equal(lse_pair[0], lse) and torch.equal(out_pair[1], out)
    g2 = _Count()
    qsa_op._dcp_gather_partials(out, lse.float(), g2, True)
    assert g2.n == 1
