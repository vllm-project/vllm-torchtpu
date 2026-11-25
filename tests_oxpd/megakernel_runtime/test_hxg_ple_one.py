# SPDX-License-Identifier: Apache-2.0
"""q38 HIX-GLUE PF0 (VLLM_TPU_HXG_PLE_ONE, CPU): the resident PLE lookup of a megakernel bucket in one gather.

Rows:
  P1 ple_min_rows: off -> the env min rows unchanged, no line; on -> 0 and one per-rank bind line per call; on with chunk tokens < 64
     -> ValueError naming VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS.
  P2 resident_lookup_raw: chunk 8 (today's 16-row program) vs chunk 512 (PF0) -> bit-identical rows, int8 and int16 tables, ids inside
     and outside the shard, 1..64 tokens; a wrong vocab_start differs (control).
  P3 call site (AST): ResidentPLETable._finish_storage passes min_rows through ple_min_rows before build_raw_lookup_op.
  P4 registered and hashed.
Run: PYTHONPATH=src pytest tests_oxpd/megakernel_runtime/test_hxg_ple_one.py
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import hixglue
from vllm_torchtpu.kernels.qwen4_exp.ple import resident_lookup_raw

KEY = "VLLM_TPU_HXG_PLE_ONE"


def test_p1_gate(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=hixglue.logger.name)
    monkeypatch.setenv(KEY, "0")
    assert hixglue.ple_min_rows(80, 512, 2) == 80
    assert not [r for r in caplog.records if KEY in r.getMessage()]
    monkeypatch.setenv(KEY, "1")
    assert hixglue.ple_min_rows(80, 512, 2) == 0 and hixglue.ple_min_rows(80, 512, 7) == 0
    lines = [r.getMessage() for r in caplog.records if KEY + " bound rank" in r.getMessage()]
    assert len(lines) == 2 and "rank 2:" in lines[0] and "rank 7:" in lines[1]
    with pytest.raises(ValueError, match="VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS"):
        hixglue.ple_min_rows(80, 8, 2)


@pytest.mark.parametrize("dt", [jnp.int8, jnp.int16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_p2_one_gather_bitwise(dt, head_dim):
    rng = np.random.default_rng(head_dim)
    local_rows, heads = 300, 4
    table = jnp.asarray(rng.integers(-100, 100, size=(-(-local_rows * head_dim // 128), 128)), dtype=dt)
    for T in (1, 8, 12, 16, 33, 64):
        ids = jnp.asarray(rng.integers(-5, 2 * local_rows, size=(T, heads)), dtype=jnp.int32)
        a = resident_lookup_raw(table, ids, 250, local_rows=local_rows, head_dim=head_dim, chunk_tokens=8)
        b = resident_lookup_raw(table, ids, 250, local_rows=local_rows, head_dim=head_dim, chunk_tokens=512)
        assert a.shape == b.shape and a.dtype == b.dtype and bool(jnp.array_equal(a, b))
    c = resident_lookup_raw(table, ids, 251, local_rows=local_rows, head_dim=head_dim, chunk_tokens=512)
    assert not bool(jnp.array_equal(b, c))


def test_p3_call_site():
    from vllm_torchtpu.models.vllm.qwen4_exp import ple_storage

    src = textwrap.dedent(inspect.getsource(ple_storage.ResidentPLETable._finish_storage))
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = [getattr(n.func, "attr", getattr(n.func, "id", "")) for n in calls]
    assert "ple_min_rows" in names and "build_raw_lookup_op" in names
    assert src.index("ple_min_rows(") < src.index("build_raw_lookup_op(")


def test_p4_registered_and_hashed():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert KEY in envs.environment_variables and KEY not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
