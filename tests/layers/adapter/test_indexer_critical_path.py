# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verification suite for Indexer Critical-Path Optimizations:
1. `TPU_DCP_INDEXER_A2A_SMALL_COLL` (default False)
2. `TPU_INDEXER_OVERLAP_QB_ABSORB` (default False)

Runs pure-JAX bit-identity + end-to-end `streamindex_topk_dcp` 4-way matrix
equivalence tests on CPU (including CPU-only environments without `torch`), plus
PyTorch adapter plumbing tests whenever `torch` is installed.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_module
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    dcp_sc_compact,
    streamindex_topk_dcp,
)
from vllm_torchtpu.kernels.mla.absorb_q import absorb_q
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor as jax_quantize_tensor,
)
from vllm_torchtpu.kernels.quantized_matmul.util import (
    xla_prequantized_matmul,
    xla_quantized_matmul,
)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# 1. Environment flag defaults & parsing
# --------------------------------------------------------------------------- #


def test_critical_path_env_flags_default_off(monkeypatch):
    monkeypatch.delenv("TPU_DCP_INDEXER_A2A_SMALL_COLL", raising=False)
    monkeypatch.delenv("TPU_DCP_INDEXER_A2A_SMALL_COLL_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TPU_INDEXER_OVERLAP_QB_ABSORB", raising=False)
    monkeypatch.delenv("TPU_INDEXER_OVERLAP_QB_ABSORB_MAX_TOKENS", raising=False)
    assert envs.TPU_DCP_INDEXER_A2A_SMALL_COLL is False
    assert envs.TPU_DCP_INDEXER_A2A_SMALL_COLL_MAX_TOKENS == 128
    assert envs.TPU_INDEXER_OVERLAP_QB_ABSORB is False
    assert envs.TPU_INDEXER_OVERLAP_QB_ABSORB_MAX_TOKENS == 64

    monkeypatch.setenv("TPU_DCP_INDEXER_A2A_SMALL_COLL", "1")
    monkeypatch.setenv("TPU_DCP_INDEXER_A2A_SMALL_COLL_MAX_TOKENS", "64")
    monkeypatch.setenv("TPU_INDEXER_OVERLAP_QB_ABSORB", "1")
    monkeypatch.setenv("TPU_INDEXER_OVERLAP_QB_ABSORB_MAX_TOKENS", "32")
    assert envs.TPU_DCP_INDEXER_A2A_SMALL_COLL is True
    assert envs.TPU_DCP_INDEXER_A2A_SMALL_COLL_MAX_TOKENS == 64
    assert envs.TPU_INDEXER_OVERLAP_QB_ABSORB is True
    assert envs.TPU_INDEXER_OVERLAP_QB_ABSORB_MAX_TOKENS == 32


# --------------------------------------------------------------------------- #
# 2. GLM-5.2 T=64 q_b (quantized + prequantized) + W_UK absorb bit-identity
# --------------------------------------------------------------------------- #


def _absorb_q_for_host(q_nope: jax.Array, w_uk_t: jax.Array) -> jax.Array:
    """Calls `absorb_q` directly on TPU; on XLA CPU (`DotThunk` lacks
    `BF16 x BF16 = F32`), promotes the `bfloat16` inputs bit-exactly to
    `float32` before calling `absorb_q` and casts back to `q_nope.dtype`."""
    if jax.default_backend() == "cpu" and q_nope.dtype == jnp.bfloat16:
        return absorb_q(q_nope.astype(jnp.float32), w_uk_t.astype(jnp.float32)).astype(
            q_nope.dtype
        )
    return absorb_q(q_nope, w_uk_t.astype(q_nope.dtype))


def test_glm52_qb_and_absorb_q_bit_exact():
    """Verifies that fused `(q_b -> slice -> absorb_q)` on GLM-5.2 decode shapes
    (`T=64, q_lora_rank=1536, H=64, nope=192, rope=64, kv_lora_rank=512`) is
    bit-for-bit (`uint16`) identical to standalone `q_b` followed by `absorb_q`,
    for both prequantized `q_c_fp8` (`TPU_FUSED_NORM_QKVA`) and unquantized `q_c`.
    """
    t = 64
    q_lora_rank = 1536
    heads, nope, rope, kv_lora_rank = 64, 192, 64, 512
    q_out_dim = heads * (nope + rope)

    k0, k1, k2, k3 = jax.random.split(jax.random.key(17), 4)
    q_c = jax.random.normal(k0, (t, q_lora_rank), jnp.bfloat16)
    qb_w = jax.random.normal(k1, (q_lora_rank, q_out_dim), jnp.float32).astype(
        jnp.float8_e4m3fn
    )
    qb_s = jax.random.uniform(k2, (q_out_dim,), jnp.float32, minval=0.01, maxval=0.2)
    w_uk_t = jax.random.normal(k3, (heads, nope, kv_lora_rank), jnp.bfloat16)

    q_c_fp8, q_c_scale = jax_quantize_tensor(q_c, jnp.float8_e4m3fn, dim=-1)

    # Standalone reference.
    ref_q_xla = xla_quantized_matmul(q_c, qb_w, qb_s)
    ref_q_pq = xla_prequantized_matmul(q_c_fp8, q_c_scale, qb_w, qb_s, jnp.bfloat16)
    ref_q_nope = ref_q_pq.reshape(t, heads, nope + rope)[:, :, :nope]
    ref_ql_nope = _absorb_q_for_host(ref_q_nope, w_uk_t)

    # Prequantized must match xla_quantized_matmul bit for bit.
    assert np.array_equal(
        np.asarray(ref_q_pq).view(np.uint16), np.asarray(ref_q_xla).view(np.uint16)
    )

    # Overlapped `merge_overlap_fn` exactness.
    def _overlapped_pq(xq, xs, wq, ws, wuk):
        q_out = xla_prequantized_matmul(xq, xs, wq, ws, jnp.bfloat16)
        h, nd, _ = wuk.shape
        hd = q_out.shape[1] // h
        qn = q_out.reshape(q_out.shape[0], h, hd)[..., :nd]
        return q_out, _absorb_q_for_host(qn, wuk)

    got_q, got_ql_nope = jax.jit(_overlapped_pq)(q_c_fp8, q_c_scale, qb_w, qb_s, w_uk_t)
    assert np.array_equal(
        np.asarray(got_q).view(np.uint16), np.asarray(ref_q_pq).view(np.uint16)
    )
    assert np.array_equal(
        np.asarray(got_ql_nope).view(np.uint16),
        np.asarray(ref_ql_nope).view(np.uint16),
    )

    # Positive control: perturbing one W_UK_T entry changes ql_nope bits.
    w_uk_bad = w_uk_t.at[0, 0, 0].set(w_uk_t[0, 0, 0] + jnp.bfloat16(1.0))
    _, bad_ql_nope = jax.jit(_overlapped_pq)(q_c_fp8, q_c_scale, qb_w, qb_s, w_uk_bad)
    assert not np.array_equal(
        np.asarray(bad_ql_nope).view(np.uint16),
        np.asarray(ref_ql_nope).view(np.uint16),
    )


# --------------------------------------------------------------------------- #
# 3. End-to-end 4-way matrix equivalence of streamindex_topk_dcp on CPU Pallas
# --------------------------------------------------------------------------- #


def test_end_to_end_4way_matrix_equivalence_on_cpu_pallas():
    """Runs all 4 combinations `(a2a_small_coll in {False, True}) x
    (qb_absorb_overlap in {False, True})` end-to-end on 4 CPU devices with
    Pallas TPU interpret mode + race detection, verifying bit-for-bit equality
    of `(local_topk, q_out, ql_nope)` across all 4 modes and zero races."""
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count=4",
    )
    proc = subprocess.run(
        [sys.executable, __file__, "worker4"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT4 ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    res = json.loads(lines[-1][len("RESULT4 ") :])
    for mode_name, stats in res.items():
        assert stats["topk_exact"], (mode_name, stats)
        assert stats["q_exact"], (mode_name, stats)
        assert stats["ql_nope_exact"], (mode_name, stats)
        assert not stats["races"], (mode_name, stats)


def _worker4():
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call  # noqa: F401
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    dcp = 4
    t, k = 16, 1024
    h_idx, d_idx = 8, 128
    q_lora_rank, heads, nope, rope, kv_lora_rank = 256, 8, 192, 64, 128
    q_out_dim = heads * (nope + rope)

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == dcp
    mesh = Mesh(np.asarray(devices).reshape(1, dcp), ("dcp_lane", DCP_AXIS_NAME))
    rep = NamedSharding(mesh, P())
    sharded = NamedSharding(mesh, P(DCP_AXIS_NAME))

    rng = np.random.default_rng(123)
    # Deterministic distinct scores per (rank, token, slot) so top-k has no ties.
    raw_scores = rng.permutation(dcp * t * k).astype(np.int32).reshape(dcp, t, k)
    raw_idxs = np.arange(dcp * t * k, dtype=np.int32).reshape(dcp, t, k) % 4096

    def _scoring_stub(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        *,
        cp_rank=None,
        final_sc_tie=(),
        **kwargs,
    ):
        del q, indexer_weights, seq_lens, page_indices, cu_q_lens, distribution, kwargs
        r = jnp.asarray(cp_rank, jnp.int32)
        all_scores = jnp.asarray(raw_scores)
        all_idxs = jnp.asarray(raw_idxs)
        scores = all_scores[r] + cache_kv[0, 0, 0, 0].astype(jnp.int32) * 0
        idxs = all_idxs[r]
        if final_sc_tie:
            return (
                idxs,
                scores,
                tuple(lax.optimization_barrier(x) for x in final_sc_tie),
            )
        return idxs, scores

    def _sc_topk_exact(scores, k_, row_lengths, **kwargs):
        del row_lengths
        vals, slots = lax.top_k(scores, k_)
        slots = slots.astype(jnp.int32)
        return (vals, slots) if kwargs.get("return_scores", True) else slots

    def _pack_exact(slots, idxs=None, dcp_rank=None):
        assert idxs is not None and dcp_rank is not None
        _, width = slots.shape
        r = jnp.asarray(dcp_rank, jnp.int32)
        lo = r * width
        hi = lo + width
        owned = (slots >= lo) & (slots < hi)
        local_col = jnp.clip(slots - lo, 0, width - 1)
        gathered = jnp.take_along_axis(idxs, local_col, axis=1)
        vals = jnp.where(owned, gathered, -1)
        col = jnp.arange(width, dtype=jnp.int32)[None, :]
        key = jnp.where(owned, col, width + col)
        order = jnp.argsort(key, axis=1)
        return jnp.take_along_axis(vals, order, axis=1)

    sc_module._sc_topk_direct = _sc_topk_exact
    dcp_sc_compact.pack_owned_winners = _pack_exact
    st_mod = sys.modules[
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    ]
    st_mod.streamindex_topk = _scoring_stub

    q = jax.device_put(jnp.zeros((t, h_idx, d_idx), jnp.float8_e4m3fn), rep)
    weights = jax.device_put(jnp.ones((t, h_idx), jnp.bfloat16), rep)
    cache_kv = jax.device_put(jnp.zeros((dcp * 4, 16, 4, 256), jnp.uint8), sharded)
    seq_lens = jax.device_put(jnp.full((t,), 2048, jnp.int32), rep)
    page_indices = jax.device_put(jnp.arange(t * 4, dtype=jnp.int32), rep)
    cu_q_lens = jax.device_put(jnp.arange(t + 1, dtype=jnp.int32), rep)
    dist = jax.device_put(jnp.array([t, t, t], jnp.int32), rep)
    rank_ids = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), sharded)
    lane_ids = jax.device_put(jnp.zeros((dcp,), dtype=jnp.int32), sharded)

    k0, k1, k2, k3 = jax.random.split(jax.random.key(99), 4)
    q_c = jax.device_put(jax.random.normal(k0, (t, q_lora_rank), jnp.bfloat16), rep)
    qb_xq, qb_xs = jax_quantize_tensor(q_c, jnp.float8_e4m3fn, dim=-1)
    qb_xq = jax.device_put(qb_xq, rep)
    qb_xs = jax.device_put(qb_xs, rep)
    qb_w = jax.device_put(
        jax.random.normal(k1, (q_lora_rank, q_out_dim), jnp.float32).astype(
            jnp.float8_e4m3fn
        ),
        rep,
    )
    qb_s = jax.device_put(
        jax.random.uniform(k2, (q_out_dim,), jnp.float32, minval=0.01, maxval=0.2), rep
    )
    w_uk_t = jax.device_put(
        jax.random.normal(k3, (heads, nope, kv_lora_rank), jnp.bfloat16), rep
    )

    def _qb_pq(xq, xs, wq, ws):
        return xla_prequantized_matmul(xq, xs, wq, ws, jnp.bfloat16)

    def _absorb_fn(q_out, wuk):
        qn = q_out.reshape(q_out.shape[0], heads, nope + rope)[..., :nope]
        return q_out, _absorb_q_for_host(qn, wuk)

    def _run_interp(fn):
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc

        params = pltpu.InterpretParams(detect_races=True, dma_execution_mode="eager")
        pltpu.reset_tpu_interpret_mode_state()
        with (
            pltpu.force_tpu_interpret_mode(params),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            out = jax.tree.map(lambda v: np.asarray(jax.block_until_ready(v)), fn())
        return out, bool(ipc.races is not None and ipc.races.races_found)

    # Mode 0: Baseline (XLA a2a + XLA ag, sequential q_b + absorb_q).
    base_topk = np.asarray(
        jax.jit(
            lambda: streamindex_topk_dcp(
                q,
                weights,
                cache_kv,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                mesh=mesh,
                k=k,
                compression_ratio=1,
                dcp_size=dcp,
                interleave_size=1,
                a2a_merge=True,
                a2a_small_coll=False,
                dcp_rank_ids=rank_ids,
                dcp_lane_ids=lane_ids,
            )
        )()
    )
    base_q, base_ql_nope = jax.tree.map(
        np.asarray,
        jax.jit(lambda: _absorb_fn(_qb_pq(qb_xq, qb_xs, qb_w, qb_s), w_uk_t))(),
    )

    out_report = {}
    for small_coll in (False, True):
        for overlap in (False, True):
            name = f"small_coll={int(small_coll)}_overlap={int(overlap)}"
            jax.clear_caches()
            if not overlap:
                topk_got, races = _run_interp(
                    jax.jit(
                        lambda sc=small_coll: streamindex_topk_dcp(
                            q,
                            weights,
                            cache_kv,
                            seq_lens,
                            page_indices,
                            cu_q_lens,
                            dist,
                            mesh=mesh,
                            k=k,
                            compression_ratio=1,
                            dcp_size=dcp,
                            interleave_size=1,
                            a2a_merge=True,
                            a2a_small_coll=sc,
                            dcp_rank_ids=rank_ids,
                            dcp_lane_ids=lane_ids,
                        )
                    )
                )
                q_got, ql_got = base_q, base_ql_nope
            else:
                (topk_got, (q_got, ql_got)), races = _run_interp(
                    jax.jit(
                        lambda sc=small_coll: streamindex_topk_dcp(
                            q,
                            weights,
                            cache_kv,
                            seq_lens,
                            page_indices,
                            cu_q_lens,
                            dist,
                            mesh=mesh,
                            k=k,
                            compression_ratio=1,
                            dcp_size=dcp,
                            interleave_size=1,
                            a2a_merge=True,
                            a2a_small_coll=sc,
                            dcp_rank_ids=rank_ids,
                            dcp_lane_ids=lane_ids,
                            overlap_fn=_qb_pq,
                            overlap_args=(qb_xq, qb_xs, qb_w, qb_s),
                            merge_overlap_fn=_absorb_fn,
                            merge_overlap_args=(w_uk_t,),
                            overlap_pin_max_tokens=64,
                            overlap_out_specs=(P(), P()),
                        )
                    )
                )
            out_report[name] = dict(
                topk_exact=bool(np.array_equal(topk_got, base_topk)),
                q_exact=bool(
                    np.array_equal(q_got.view(np.uint16), base_q.view(np.uint16))
                ),
                ql_nope_exact=bool(
                    np.array_equal(ql_got.view(np.uint16), base_ql_nope.view(np.uint16))
                ),
                races=races,
            )
    print("RESULT4 " + json.dumps(out_report), flush=True)


# --------------------------------------------------------------------------- #
# 4. Adapter helper & attention bypass verification (runs on CPU even w/o torch)
# --------------------------------------------------------------------------- #


def test_adapter_helpers_and_attention_bypass_without_torch(monkeypatch):
    """Executes `_qb_absorb_operands` and `_w_uk_t_for_absorb` extracted from
    `mla_attention_op.py` (so they are exercised even in CPU-only envs without
    `torch`) and verifies `PallasMLAttentionBackendImpl.forward` bypasses
    `_absorb_q` when `ql_nope_absorbed` is supplied."""
    import ast
    import types
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    mla_src = (
        repo_root / "src/vllm_torchtpu/layers/adapter/custom_ops/mla_attention_op.py"
    ).read_text()
    attn_src = (repo_root / "src/vllm_torchtpu/layers/adapter/attention.py").read_text()

    tree = ast.parse(mla_src)
    extracted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_qb_absorb_operands", "_w_uk_t_for_absorb")
    ]
    assert len(extracted) == 2

    class _FakeFp8Method:
        pass

    class _FakeTorch:
        float8_e4m3fn = "fp8"
        float8_e4m3fnuz = "fp8uz"

        class Tensor:
            pass

    if torch is None:
        fake_layers = types.ModuleType("vllm_torchtpu.layers")
        fake_adapter = types.ModuleType("vllm_torchtpu.layers.adapter")
        fake_quant = types.ModuleType("vllm_torchtpu.layers.adapter.quantization")
        fake_fp8 = types.ModuleType("vllm_torchtpu.layers.adapter.quantization.fp8")
        fake_fp8.VllmFp8LinearMethodTPU = _FakeFp8Method
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.layers", fake_layers)
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.layers.adapter", fake_adapter)
        monkeypatch.setitem(
            sys.modules, "vllm_torchtpu.layers.adapter.quantization", fake_quant
        )
        monkeypatch.setitem(
            sys.modules, "vllm_torchtpu.layers.adapter.quantization.fp8", fake_fp8
        )
        expected_fp8_cls = _FakeFp8Method
        expected_fp8_dtype = "fp8"
    else:
        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            VllmFp8LinearMethodTPU,
        )

        expected_fp8_cls = VllmFp8LinearMethodTPU
        expected_fp8_dtype = torch.float8_e4m3fn

    ns = {
        "Any": object,
        "torch": _FakeTorch if torch is None else torch,
        "KEEP_VLLM_LAYOUT_ATTR": "_tpu_keep_vllm_weight_layout",
        "FP8_PALLAS_WEIGHT": "_tpu_fp8_pallas_weight",
        "FP8_PALLAS_N_ATTR": "_tpu_fp8_pallas_n",
        "FP8_PALLAS_HBM_PIN_ATTR": "_tpu_fp8_pallas_hbm_pin",
    }
    mod_ast = ast.Module(body=extracted, type_ignores=[])
    exec(compile(mod_ast, "<mla_helpers>", "exec"), ns)  # noqa: S102
    qb_absorb_operands = ns["_qb_absorb_operands"]
    w_uk_t_for_absorb = ns["_w_uk_t_for_absorb"]

    class _FakeWeight:
        def __init__(self, dtype=expected_fp8_dtype, ndim=2, shape=(128, 256)):
            self.dtype = dtype
            self.ndim = ndim
            self.shape = shape

        def dim(self):
            return self.ndim

    def make_layer(**overrides):
        layer = MagicMock(spec=[])
        layer.quant_method = expected_fp8_cls.__new__(expected_fp8_cls)
        layer.bias = None
        layer.gather_output = False
        layer.weight = _FakeWeight()
        layer.weight_scale = _FakeWeight(dtype="f32", ndim=1, shape=(256,))
        layer._tpu_fp8_pallas_n = 256
        for k, v in overrides.items():
            setattr(layer, k, v)
        return layer

    assert qb_absorb_operands(make_layer())[-1] == "pallas"
    assert (
        qb_absorb_operands(make_layer(_tpu_fp8_pallas_hbm_pin=True))[-1] == "pallas_hbm"
    )
    assert qb_absorb_operands(make_layer(), use_prequant=True)[-1] == "prequant"
    assert qb_absorb_operands(make_layer(_tpu_fp8_pallas_n=None))[-1] == "xla"
    assert qb_absorb_operands(make_layer(_tpu_keep_vllm_weight_layout=True)) is None
    assert qb_absorb_operands(make_layer(bias="bias")) is None
    assert qb_absorb_operands(make_layer(_tpu_fp8_pallas_weight="skinny")) is None
    assert qb_absorb_operands(make_layer(gather_output=True)) is None

    attn = MagicMock(spec=[])
    attn.W_UK_T = _FakeWeight(dtype="bf16", ndim=3, shape=(8, 192, 512))
    attn.W_UK_T_scale = None
    attn.W_UK_T_dcp_qrep = _FakeWeight(dtype="bf16", ndim=3, shape=(64, 192, 512))
    attn.W_UK_T_dcp_qrep_scale = None
    assert w_uk_t_for_absorb(attn, False) is attn.W_UK_T
    assert w_uk_t_for_absorb(attn, True) is attn.W_UK_T_dcp_qrep
    attn.W_UK_T_dcp_qrep_scale = "fp8_scale"
    assert w_uk_t_for_absorb(attn, True) is None

    # Verify attention.py wires `ql_nope_absorbed` bypass before `_absorb_q`.
    assert 'ql_nope_absorbed = kwargs.get("ql_nope_absorbed")' in attn_src
    assert "if ql_nope_absorbed is not None:" in attn_src


@pytest.mark.skipif(torch is None, reason="torch not installed in this venv")
def test_qb_absorb_operands_and_w_uk_t_helpers_torch():
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
    from vllm_torchtpu.layers.adapter.linear_common import (
        FP8_PALLAS_HBM_PIN_ATTR,
        FP8_PALLAS_N_ATTR,
        FP8_PALLAS_WEIGHT,
        KEEP_VLLM_LAYOUT_ATTR,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    def make_layer(**overrides):
        layer = torch.nn.Module()
        layer.quant_method = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
        layer.weight = torch.nn.Parameter(
            torch.zeros(128, 256).to(torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(torch.ones(256), requires_grad=False)
        layer.bias = None
        setattr(layer, FP8_PALLAS_N_ATTR, 256)
        for k, v in overrides.items():
            setattr(layer, k, v)
        return layer

    assert mla_op._qb_absorb_operands(make_layer())[-1] == "pallas"
    assert (
        mla_op._qb_absorb_operands(make_layer(**{FP8_PALLAS_HBM_PIN_ATTR: True}))[-1]
        == "pallas_hbm"
    )
    assert mla_op._qb_absorb_operands(make_layer(), use_prequant=True)[-1] == "prequant"
    assert (
        mla_op._qb_absorb_operands(make_layer(**{FP8_PALLAS_N_ATTR: None}))[-1] == "xla"
    )
    for bad in (
        {KEEP_VLLM_LAYOUT_ATTR: True},
        {"bias": torch.nn.Parameter(torch.zeros(256))},
        {FP8_PALLAS_WEIGHT: torch.nn.Parameter(torch.zeros(1))},
        {"gather_output": True},
        {"quant_method": object()},
    ):
        assert mla_op._qb_absorb_operands(make_layer(**bad)) is None

    attn = MagicMock()
    attn.W_UK_T = torch.ones(8, 192, 512)
    attn.W_UK_T_scale = None
    attn.W_UK_T_dcp_qrep = torch.ones(64, 192, 512)
    attn.W_UK_T_dcp_qrep_scale = None
    assert mla_op._w_uk_t_for_absorb(attn, False) is attn.W_UK_T
    assert mla_op._w_uk_t_for_absorb(attn, True) is attn.W_UK_T_dcp_qrep
    attn.W_UK_T_dcp_qrep_scale = torch.ones(64)
    assert mla_op._w_uk_t_for_absorb(attn, True) is None


if __name__ == "__main__" and sys.argv[1:] == ["worker4"]:
    _worker4()
