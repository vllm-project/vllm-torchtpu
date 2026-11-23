# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-HC (rule 12): the TPU_QWEN4_HC_PREFILL_FOLD / _DOWN_ASM op variants
are built by the REAL torch_tpu pallas.jax_op (signature check, schema,
torch.library registration), not a fake. Needs torch_tpu (a CDK CPU pod has it;
elsewhere the module skips, and the q38 proof harness fails on any skip).

Flag-off names are pinned: the builders called without the new keywords must
register exactly today's op names.
"""

import hashlib

import pytest

pytest.importorskip("torch_tpu")

import torch  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (  # noqa: E402
    build_prefill_combine_mix_op,
    build_prefill_inj_mix_op,
)

ARGS = (4, 2560, 1e-6, 256, True, "bf16ep64")  # the S3 prefill build (C, render)
TAG = hashlib.sha1(repr(1e-6).encode()).hexdigest()[:8]


def _schema(name):
    ns, op = name.split("::")
    return getattr(getattr(torch.ops, ns), op).default._schema


def _check(name, n_args, n_rets):
    s = _schema(name)
    assert [str(a.type) for a in s.arguments] == ["Tensor"] * n_args, s
    assert len(s.returns) == n_rets, s


def test_flag_off_names_are_unchanged():
    build_prefill_inj_mix_op(*ARGS)
    build_prefill_combine_mix_op(*ARGS)
    _check(f"qwen4_exp::hc_prefill_inj_c4_h2560_m256_e{TAG}_p_nbf16ep64_v1", 5, 2)
    _check(f"qwen4_exp::hc_prefill_combine_c4_h2560_m256_e{TAG}_p_nbf16ep64_v1", 7, 3)


@pytest.mark.parametrize(
    "builder, kwargs, suffix, n_args, n_rets",
    [
        (build_prefill_inj_mix_op, dict(down_asm=True), "_da", 5, 2),
        (build_prefill_combine_mix_op, dict(reread=True), "_rr", 7, 3),
        (build_prefill_combine_mix_op, dict(down_asm=True), "_da", 7, 3),
        (build_prefill_combine_mix_op, dict(reread=True, down_asm=True), "_rr_da", 7, 3),
    ],
)
def test_variant_ops_build_through_the_real_jax_op(builder, kwargs, suffix, n_args, n_rets):
    builder(*ARGS, **kwargs)
    kind = "inj" if builder is build_prefill_inj_mix_op else "combine"
    _check(f"qwen4_exp::hc_prefill_{kind}_c4_h2560_m256_e{TAG}_p_nbf16ep64{suffix}_v1", n_args, n_rets)
