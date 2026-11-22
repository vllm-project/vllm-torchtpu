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
"""CPU Pallas interpret + race-detector + TPU_7X MLIR lowering tests for
`TPU_DCP_INDEXER_A2A_SMALL_COLL` (`small_indexer_score_all_to_all` +
`small_all_gather` in `streamindex_topk_dcp`).
"""

import contextlib
import io
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_module
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    dcp_sc_compact,
    streamindex_topk_dcp,
)
from vllm_torchtpu.kernels.experimental.small_collectives import (
    all_gather as small_ag,
)
from vllm_torchtpu.kernels.experimental.small_collectives import (
    indexer_a2a,
)
from vllm_torchtpu.kernels.mla.absorb_q import absorb_q
from vllm_torchtpu.kernels.quantized_matmul.util import xla_prequantized_matmul

N = 16
AXIS = "dcp"
LANE_AXIS = "dcp_lane"


def test_supports_indexer_score_all_to_all():
    assert indexer_a2a.supports_indexer_score_all_to_all(64, 16, 2048, 128)
    assert indexer_a2a.supports_indexer_score_all_to_all(16, 16, 2048, 128)
    assert indexer_a2a.supports_indexer_score_all_to_all(128, 16, 2048, 128)
    # Exceeds max_tokens cap.
    assert not indexer_a2a.supports_indexer_score_all_to_all(256, 16, 2048, 128)
    # Not divisible by num_cores.
    assert not indexer_a2a.supports_indexer_score_all_to_all(24, 16, 2048, 128)
    # k not a multiple of 1024 (8 * 128).
    assert not indexer_a2a.supports_indexer_score_all_to_all(64, 16, 512, 128)
    assert not indexer_a2a.supports_indexer_score_all_to_all(64, 1, 2048, 128)


def test_validation_rejects_bad_shapes_and_dtypes():
    with pytest.raises(ValueError, match="32-bit"):
        indexer_a2a.small_indexer_score_all_to_all(
            jnp.zeros((64 * 2048,), jnp.bfloat16),
            num_cores=16,
            tokens_per_core=4,
            k=2048,
            axis=AXIS,
        )
    with pytest.raises(ValueError, match="1-D"):
        indexer_a2a.small_indexer_score_all_to_all(
            jnp.zeros((64, 2048), jnp.int32),
            num_cores=16,
            tokens_per_core=4,
            k=2048,
            axis=AXIS,
        )
    with pytest.raises(ValueError, match="multiple of 1024"):
        indexer_a2a.small_indexer_score_all_to_all(
            jnp.zeros((16 * 512,), jnp.int32),
            num_cores=16,
            tokens_per_core=1,
            k=512,
            axis=AXIS,
        )
    with pytest.raises(ValueError, match="lane requires lane_axis"):
        indexer_a2a.small_indexer_score_all_to_all(
            jnp.zeros((64 * 2048,), jnp.int32),
            num_cores=16,
            tokens_per_core=4,
            k=2048,
            axis=AXIS,
            lane=jnp.zeros((1,), jnp.int32),
        )
    with pytest.raises(ValueError, match="x length"):
        indexer_a2a.small_indexer_score_all_to_all(
            jnp.zeros((32 * 2048,), jnp.int32),
            num_cores=16,
            tokens_per_core=4,
            k=2048,
            axis=AXIS,
        )


def test_numpy_reference_matches_token_major_all_to_all():
    rng = np.random.default_rng(7)
    raw = rng.integers(-100000, 100000, size=(16, 64, 2048), dtype=np.int32)
    got = indexer_a2a.reference(raw)
    assert got.shape == (16, 4, 16 * 2048)
    for dst in range(16):
        for t in range(4):
            tok = dst * 4 + t
            for src in range(16):
                sl = got[dst, t, src * 2048 : (src + 1) * 2048]
                assert np.array_equal(sl, raw[src, tok])


def _run_worker():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N}"
        " --xla_allow_excess_precision=false",
    )
    proc = subprocess.run(
        [sys.executable, __file__, "worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def on_16_devices():
    return _run_worker()


def test_race_detector_is_live(on_16_devices):
    assert on_16_devices["race_control"] == {"racy": True, "clean": False}


@pytest.mark.parametrize("tokens_per_core", [1, 4], ids=["t16_direct_hbm", "t64_glm52"])
def test_score_a2a_and_stage3_ag_match_xla_bit_for_bit(on_16_devices, tokens_per_core):
    case = on_16_devices["dcp16"][str(tokens_per_core)]
    for mode in ("on_wait", "eager"):
        got = case[mode]
        assert got["a2a_exact"], got
        assert got["ag_exact"], got
        assert not got["a2a_races"], got
        assert not got["ag_races"], got


def test_multi_lane_2x8_mesh_isolates_lanes_with_zero_races(on_16_devices):
    got = on_16_devices["lane2_dcp8"]
    for mode in ("on_wait", "eager"):
        assert got[mode]["a2a_exact"], got[mode]
        assert got[mode]["ag_exact"], got[mode]
        assert not got[mode]["races"], got[mode]


def test_axis_index_without_rank_operand_is_exact(on_16_devices):
    got = on_16_devices["axis_index"]
    assert got["a2a_exact"] and not got["races"], got


# --------------------------------------------------------------------------- #
# TPU_7X Mosaic MLIR lowering & full streamindex_topk_dcp export verification
# --------------------------------------------------------------------------- #


@pytest.fixture
def tpu7x_info(monkeypatch):
    from jax._src.pallas.mosaic import tpu_info

    kind = jax.devices()[0].device_kind
    if tpu_info.chip_version_from_device_kind(kind) is None:
        monkeypatch.setitem(
            tpu_info.registry,
            kind,
            lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1),
        )
    tpu_info.get_tpu_info.cache_clear()
    yield
    tpu_info.get_tpu_info.cache_clear()


def _asm(exported) -> str:
    from jax.interpreters import mlir

    with mlir.make_ir_context():
        module = mlir.ir.Module.parse(exported.mlir_module())
        return str(module.operation.get_asm(enable_debug_info=False))


@pytest.mark.parametrize("tokens_per_core", [1, 4])
@pytest.mark.parametrize("hbm_operand", [True, False])
def test_tpu7x_lowering_of_indexer_score_a2a(tpu7x_info, tokens_per_core, hbm_operand):
    from jax.sharding import AbstractMesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    k = 2048
    t = N * tokens_per_core
    mesh = AbstractMesh((1, N), (LANE_AXIS, AXIS))

    def fn(scores, rank, lane):
        return jax.shard_map(
            lambda s, r, ln: indexer_a2a.small_indexer_score_all_to_all(
                s,
                num_cores=N,
                tokens_per_core=tokens_per_core,
                k=k,
                axis=AXIS,
                rank=r,
                lane_axis=LANE_AXIS,
                lane=ln,
                hbm_operand=hbm_operand,
            ),
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )(scores, rank, lane)

    sharded = NamedSharding(mesh, P(AXIS))
    exported = jax.export.export(jax.jit(fn), platforms=["tpu"])(
        jax.ShapeDtypeStruct((N * t * k,), jnp.int32, sharding=sharded),
        jax.ShapeDtypeStruct((N,), jnp.int32, sharding=sharded),
        jax.ShapeDtypeStruct((N,), jnp.int32, sharding=sharded),
    )
    mlir_text = _asm(exported)
    assert f"small_idx_score_a2a_n{N}_t{tokens_per_core}_k{k}" in mlir_text
    assert "all_to_all" not in mlir_text


def test_tpu7x_export_full_streamindex_topk_dcp_small_coll_and_qb_absorb(
    monkeypatch, tpu7x_info
):
    """Exports the complete GLM-5.2 DCP=16, T=64 `streamindex_topk_dcp` pipeline
    to `TPU_7X` MLIR with both `a2a_small_coll=True` and overlapped prequantized
    `q_b` + `absorb_q` enabled."""
    from jax.sharding import AbstractMesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    jax.clear_caches()

    def _sc_topk_stub(scores, k_, row_lengths, **kwargs):
        del row_lengths
        out = jnp.zeros((scores.shape[0], k_), jnp.int32)
        return (out, out) if kwargs.get("return_scores", True) else out

    def _pack_stub(slots, idxs=None, dcp_rank=None):
        del idxs, dcp_rank
        return slots

    def _scoring_stub(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        *,
        k,
        final_sc_tie=(),
        **kwargs,
    ):
        del (
            indexer_weights,
            cache_kv,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            kwargs,
        )
        b = q.shape[0]
        idxs = jnp.zeros((b, k), jnp.int32)
        scores = jnp.zeros((b, k), jnp.int32)
        if final_sc_tie:
            return (
                idxs,
                scores,
                tuple(lax.optimization_barrier(x) for x in final_sc_tie),
            )
        return idxs, scores

    stk_mod = sys.modules[
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    ]
    monkeypatch.setattr(sc_module, "_sc_topk_direct", _sc_topk_stub)
    monkeypatch.setattr(dcp_sc_compact, "pack_owned_winners", _pack_stub)
    monkeypatch.setattr(stk_mod, "streamindex_topk", _scoring_stub)

    t, h_idx, d_idx, topk = 64, 32, 128, 2048
    q_lora_rank, h_qrep, nope_dim, rope_dim, kv_lora_rank = 1536, 64, 192, 64, 512
    q_out_dim = h_qrep * (nope_dim + rope_dim)
    mesh = AbstractMesh((1, N), (LANE_AXIS, DCP_AXIS_NAME))
    rep = NamedSharding(mesh, P())
    sharded = NamedSharding(mesh, P(DCP_AXIS_NAME))

    def _absorb_fn(q_out, w_uk_t):
        q_nope = q_out.reshape(q_out.shape[0], h_qrep, nope_dim + rope_dim)[
            ..., :nope_dim
        ]
        return q_out, absorb_q(q_nope, w_uk_t.astype(q_nope.dtype))

    def run_pipeline(
        q,
        weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        dist,
        rank_ids,
        lane_ids,
        qb_xq,
        qb_xs,
        qb_w,
        qb_s,
        w_uk_t,
    ):
        return streamindex_topk_dcp(
            q,
            weights,
            cache_kv,
            seq_lens,
            page_indices,
            cu_q_lens,
            dist,
            mesh=mesh,
            k=topk,
            compression_ratio=1,
            dcp_size=N,
            interleave_size=1,
            a2a_merge=True,
            a2a_small_coll=True,
            dcp_rank_ids=rank_ids,
            dcp_lane_ids=lane_ids,
            overlap_fn=lambda xq, xs, wq, ws: xla_prequantized_matmul(
                xq, xs, wq, ws, jnp.bfloat16
            ),
            overlap_args=(qb_xq, qb_xs, qb_w, qb_s),
            merge_overlap_fn=_absorb_fn,
            merge_overlap_args=(w_uk_t,),
            overlap_pin_max_tokens=64,
            overlap_out_specs=(P(), P()),
        )

    S = jax.ShapeDtypeStruct
    exported = jax.export.export(jax.jit(run_pipeline), platforms=["tpu"])(
        S((t, h_idx, d_idx), jnp.float8_e4m3fn, sharding=rep),
        S((t, h_idx), jnp.bfloat16, sharding=rep),
        S((N * 16, 16, 4, 256), jnp.uint8, sharding=sharded),
        S((t,), jnp.int32, sharding=rep),
        S((t * 8,), jnp.int32, sharding=rep),
        S((t + 1,), jnp.int32, sharding=rep),
        S((3,), jnp.int32, sharding=rep),
        S((N,), jnp.int32, sharding=sharded),
        S((N,), jnp.int32, sharding=sharded),
        S((t, q_lora_rank), jnp.float8_e4m3fn, sharding=rep),
        S((t, 1), jnp.float32, sharding=rep),
        S((q_lora_rank, q_out_dim), jnp.float8_e4m3fn, sharding=rep),
        S((q_out_dim,), jnp.float32, sharding=rep),
        S((h_qrep, nope_dim, kv_lora_rank), jnp.bfloat16, sharding=rep),
    )
    mlir_text = _asm(exported)
    assert "small_idx_score_a2a_n16_t4_k2048" in mlir_text
    assert "small_all_gather_n16_l8192" in mlir_text
    assert "stablehlo.all_to_all" not in mlir_text
    assert "stablehlo.all_gather" not in mlir_text
    assert mlir_text.count("stablehlo.optimization_barrier") >= 3
    out_avals = exported.out_avals
    assert len(out_avals) == 3
    assert out_avals[0].shape == (N * t, topk)
    assert out_avals[1].shape == (t, q_out_dim)
    assert out_avals[2].shape == (t, h_qrep, kv_lora_rank)


# --------------------------------------------------------------------------- #
# 16-CPU-device subprocess worker (Pallas TPU interpret + race detector)
# --------------------------------------------------------------------------- #


def _interpret(fn, *args, dma_execution_mode="on_wait"):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    params = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_execution_mode
    )
    pltpu.reset_tpu_interpret_mode_state()
    with (
        pltpu.force_tpu_interpret_mode(params),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        out = jax.tree.map(lambda x: np.asarray(jax.block_until_ready(x)), fn(*args))
    return out, bool(interpret_pallas_call.races.races_found)


def _race_control():
    from jax.experimental import pallas as pl

    def make(read_early):
        def body(x_hbm, o_ref, buf, sem):
            copy = pltpu.make_async_copy(x_hbm, buf, sem)
            copy.start()
            if read_early:
                o_ref[...] = buf[...]
            copy.wait()
            if not read_early:
                o_ref[...] = buf[...]

        return pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((8, 128), jnp.int32),
                pltpu.SemaphoreType.DMA(()),
            ],
        )

    x = jnp.ones((8, 128), jnp.int32)
    return {
        "racy": _interpret(lambda v: make(True)(v), x)[1],
        "clean": _interpret(lambda v: make(False)(v), x)[1],
    }


def _worker():
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == N, devices
    rng = np.random.default_rng(42)

    # 1. Full DCP=16 (1x16 mesh), k=2048, tokens_per_core in {1, 4}.
    mesh16 = Mesh(np.asarray(devices).reshape(1, N), (LANE_AXIS, AXIS))
    spec16 = NamedSharding(mesh16, P(AXIS))
    rank16 = jax.device_put(jnp.arange(N, dtype=jnp.int32), spec16)
    lane16 = jax.device_put(jnp.zeros((N,), dtype=jnp.int32), spec16)

    dcp16_cases = {}
    k = 2048
    for tpc in (1, 4):
        t = N * tpc
        raw_scores = rng.integers(-(2**28), 2**28, size=(N, t, k), dtype=np.int32)
        want_a2a = indexer_a2a.reference(raw_scores)  # [N, tpc, N * k]
        # Stage-2 exact top-k slot indices per rank: [N, tpc, k].
        raw_slots = np.argpartition(-want_a2a, kth=k - 1, axis=-1)[..., :k].astype(
            np.int32
        )
        want_ag = np.broadcast_to(raw_slots.reshape(N * tpc, k), (N, N * tpc, k)).copy()

        scores_in = jax.device_put(jnp.asarray(raw_scores.reshape(N * t * k)), spec16)
        slots_in = jax.device_put(jnp.asarray(raw_slots.reshape(N * tpc * k)), spec16)

        a2a_fn = jax.jit(
            jax.shard_map(
                lambda s, r, ln, tpc_=tpc: indexer_a2a.small_indexer_score_all_to_all(
                    s,
                    num_cores=N,
                    tokens_per_core=tpc_,
                    k=k,
                    axis=AXIS,
                    rank=r,
                    lane_axis=LANE_AXIS,
                    lane=ln,
                    hbm_operand=True,
                ),
                mesh=mesh16,
                in_specs=(P(AXIS), P(AXIS), P(AXIS)),
                out_specs=P(AXIS),
                check_vma=False,
            )
        )
        ag_fn = jax.jit(
            jax.shard_map(
                lambda sl, r, ln: small_ag.small_all_gather(
                    sl,
                    num_cores=N,
                    axis=AXIS,
                    rank=r,
                    lane_axis=LANE_AXIS,
                    lane=ln,
                    hbm_operand=True,
                ),
                mesh=mesh16,
                in_specs=(P(AXIS), P(AXIS), P(AXIS)),
                out_specs=P(AXIS),
                check_vma=False,
            )
        )

        # Also cross-check against XLA lax.all_to_all and lax.all_gather!
        xla_a2a = np.asarray(
            jax.shard_map(
                lambda s, t_=t: lax.all_to_all(
                    s.reshape(t_, k), AXIS, split_axis=0, concat_axis=1, tiled=True
                ),
                mesh=mesh16,
                in_specs=(P(AXIS),),
                out_specs=P(AXIS),
                check_vma=False,
            )(scores_in)
        ).reshape(N, tpc, N * k)
        assert np.array_equal(xla_a2a, want_a2a)

        per_mode = {}
        for mode in ("on_wait", "eager"):
            got_a2a, a2a_races = _interpret(
                a2a_fn, scores_in, rank16, lane16, dma_execution_mode=mode
            )
            got_ag, ag_races = _interpret(
                ag_fn, slots_in, rank16, lane16, dma_execution_mode=mode
            )
            per_mode[mode] = dict(
                a2a_exact=bool(
                    np.array_equal(got_a2a.reshape(N, tpc, N * k), want_a2a)
                ),
                ag_exact=bool(np.array_equal(got_ag.reshape(N, t, k), want_ag)),
                a2a_races=a2a_races,
                ag_races=ag_races,
            )
        dcp16_cases[str(tpc)] = per_mode

    # 2. Multi-lane 2x8 mesh (2 independent TP lanes x DCP=8, T=32 -> tpc=4).
    lanes, dcp8, tpc8 = 2, 8, 4
    t8 = dcp8 * tpc8
    mesh2x8 = Mesh(np.asarray(devices).reshape(lanes, dcp8), (LANE_AXIS, AXIS))
    spec2x8 = NamedSharding(mesh2x8, P(LANE_AXIS, AXIS))
    raw_2x8 = rng.integers(-(2**28), 2**28, size=(lanes, dcp8, t8, k), dtype=np.int32)
    want_a2a_2x8 = np.stack([indexer_a2a.reference(raw_2x8[ln]) for ln in range(lanes)])
    raw_slots_2x8 = rng.integers(
        0, dcp8 * k, size=(lanes, dcp8, tpc8, k), dtype=np.int32
    )
    want_ag_2x8 = np.stack(
        [
            np.broadcast_to(raw_slots_2x8[ln].reshape(t8, k), (dcp8, t8, k)).copy()
            for ln in range(lanes)
        ]
    )
    scores_2x8 = jax.device_put(
        jnp.asarray(raw_2x8.reshape(lanes, dcp8 * t8 * k)), spec2x8
    )
    slots_2x8 = jax.device_put(
        jnp.asarray(raw_slots_2x8.reshape(lanes, dcp8 * tpc8 * k)), spec2x8
    )
    rank_2x8 = jax.device_put(
        jnp.broadcast_to(jnp.arange(dcp8, dtype=jnp.int32), (lanes, dcp8)), spec2x8
    )
    lane_2x8 = jax.device_put(
        jnp.broadcast_to(jnp.arange(lanes, dtype=jnp.int32)[:, None], (lanes, dcp8)),
        spec2x8,
    )

    both_2x8 = jax.jit(
        jax.shard_map(
            lambda s, sl, r, ln: (
                indexer_a2a.small_indexer_score_all_to_all(
                    s[0],
                    num_cores=dcp8,
                    tokens_per_core=tpc8,
                    k=k,
                    axis=AXIS,
                    rank=r[0],
                    lane_axis=LANE_AXIS,
                    lane=ln[0],
                    hbm_operand=True,
                )[None],
                small_ag.small_all_gather(
                    sl[0],
                    num_cores=dcp8,
                    axis=AXIS,
                    rank=r[0],
                    lane_axis=LANE_AXIS,
                    lane=ln[0],
                    hbm_operand=True,
                )[None],
            ),
            mesh=mesh2x8,
            in_specs=(
                P(LANE_AXIS, AXIS),
                P(LANE_AXIS, AXIS),
                P(LANE_AXIS, AXIS),
                P(LANE_AXIS, AXIS),
            ),
            out_specs=(P(LANE_AXIS, AXIS), P(LANE_AXIS, AXIS)),
            check_vma=False,
        )
    )
    lane2_dcp8 = {}
    for mode in ("on_wait", "eager"):
        (got_a2a, got_ag), races = _interpret(
            both_2x8, scores_2x8, slots_2x8, rank_2x8, lane_2x8, dma_execution_mode=mode
        )
        lane2_dcp8[mode] = dict(
            a2a_exact=bool(
                np.array_equal(
                    got_a2a.reshape(lanes, dcp8, tpc8, dcp8 * k), want_a2a_2x8
                )
            ),
            ag_exact=bool(
                np.array_equal(got_ag.reshape(lanes, dcp8, t8, k), want_ag_2x8)
            ),
            races=races,
        )

    # 3. Single-axis mesh without rank operand (axis_index path).
    mesh1d = Mesh(np.asarray(devices), (AXIS,))
    spec1d = NamedSharding(mesh1d, P(AXIS))
    raw_1d = rng.integers(-(2**28), 2**28, size=(N, 32, 1024), dtype=np.int32)
    want_1d = indexer_a2a.reference(raw_1d)
    scores_1d = jax.device_put(jnp.asarray(raw_1d.reshape(N * 32 * 1024)), spec1d)
    fn_1d = jax.jit(
        jax.shard_map(
            lambda s: indexer_a2a.small_indexer_score_all_to_all(
                s, num_cores=N, tokens_per_core=2, k=1024, axis=AXIS, rank=None
            ),
            mesh=mesh1d,
            in_specs=(P(AXIS),),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )
    got_1d, races_1d = _interpret(fn_1d, scores_1d)
    axis_index = dict(
        a2a_exact=bool(np.array_equal(got_1d.reshape(N, 2, N * 1024), want_1d)),
        races=races_1d,
    )

    result = dict(
        race_control=_race_control(),
        dcp16=dcp16_cases,
        lane2_dcp8=lane2_dcp8,
        axis_index=axis_index,
    )
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["worker"]:
    _worker()
