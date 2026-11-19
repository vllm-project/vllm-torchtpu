# SPDX-License-Identifier: Apache-2.0
"""q38 STALL-HUNT: VLLM_TPU_VFY_WARM_ALIAS with the real vllm_torchtpu package (reports/v8_stall_hunt_1116.md).
CDK CPU pod only. The harness-level proofs (S1-S5) are in tests/q38_stallhunt/test_vfy_warm_alias.py.

S6 DRAFT-ASM (v8 serves DRAFT_ASM_ONE / FOLD / CAP with VFY_ONE=2, so its superset seed is the one folded into the
   verify): the serving superset's num_draft is chunk.draft_lengths (the verify's draft_lengths) and base_args[0] is
   the same object; the warm superset after tpu_runner._vfy_warm_alias takes the given tensor in args[0] and
   base_args[0], with keys equal to serving's.
S7 registration (methodology rule 13): the key is in envs.environment_variables, parses "1" -> True, "0" / unset ->
   False, is in _TPU_COMPILE_ENV_IGNORED and not in the hashed factors; tpu_runner reads it.
Run: python -m pytest -q -p no:cacheprovider --noconftest tests/q38_stallhunt/test_vfy_warm_alias_real.py
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "q38_draftasm"))
import draftasm_harness as dh  # noqa: E402
from vllm_torchtpu.runner import vfy_one  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm  # noqa: E402

KEY = "VLLM_TPU_VFY_WARM_ALIAS"


def _alias_fn():
    """tpu_runner._vfy_warm_alias without importing tpu_runner (torch_tpu / jax at import): its source, exec'd."""
    import ast
    import dataclasses
    import textwrap
    from typing import Any

    path = os.path.join(HERE, "..", "..", "src", "vllm_torchtpu", "runner", "tpu_runner.py")
    src = open(path).read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_vfy_warm_alias")
    ns = {"torch": torch, "dataclasses": dataclasses, "Any": Any}
    exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    return ns["_vfy_warm_alias"]


@pytest.mark.parametrize("R", [5, 16, 40, 64])
def test_s6_draft_asm_superset_seed_alias(monkeypatch, R):
    alias = _alias_fn()
    dh.install(monkeypatch, fold=True)
    fp = dh.make_proposer(True)
    s = dh.make_step(7600 + R, R)
    live = fp.vfy_seed_spec(s.chunk, R)
    assert live is not None and live.fn is draft_asm._async_seed_inputs_asm_impl
    assert live.args[0] is s.chunk.draft_lengths, "serving superset: num_draft is chunk.draft_lengths"
    assert live.base_args[0] is live.args[0]
    T, B = s.chunk.input_ids.shape[0], s.chunk.draft_lengths.shape[0]
    warm = fp.vfy_seed_warm_spec(T, B, s.chunk.attn_ctx.seq_lens.shape[0], R)
    assert warm.args[0] is not s.chunk.draft_lengths and warm.base_args[0] is warm.args[0]
    dl = torch.zeros(B, dtype=s.chunk.draft_lengths.dtype)
    got, ok = alias(warm, dl)
    assert ok and got.args[0] is dl and got.base_args[0] is dl
    assert got.args[1:] == warm.args[1:] or all(a is b for a, b in zip(got.args[1:], warm.args[1:]))
    assert vfy_one.key(got.fn, *got.args) == vfy_one.key(live.fn, *live.args)
    assert vfy_one.key(*got.base_args) == vfy_one.key(*live.base_args)


@pytest.mark.parametrize("raw,want", [(None, False), ("0", False), ("1", True), ("true", True)])
def test_s7_registration(monkeypatch, raw, want):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED, _tpu_compile_env_factors

    if raw is None:
        monkeypatch.delenv(KEY, raising=False)
    else:
        monkeypatch.setenv(KEY, raw)
    assert KEY in envs.environment_variables
    assert envs.environment_variables[KEY]() is want
    assert getattr(envs, KEY) is want
    assert KEY in _TPU_COMPILE_ENV_IGNORED and KEY not in _tpu_compile_env_factors()
    runner_src = open(os.path.join(HERE, "..", "..", "src", "vllm_torchtpu", "runner", "tpu_runner.py")).read()
    assert f'"{KEY}"' in runner_src
