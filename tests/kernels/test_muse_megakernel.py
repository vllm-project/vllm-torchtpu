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
"""Tests for the Muse Spark Open 816B A42B single-dispatch Pallas decode megakernel."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.muse_megakernel import (
    KV_GROUP,
    PAGE_TOKENS,
    SW_RING,
    TP_AXIS_NAME,
    MuseMegakernelConfig,
    canonical_test_weights,
    empty_hybrid_kv_cache,
    logical_mesh,
    make_hybrid_decode_step,
)
from vllm_torchtpu.models.vllm.muse_spark.megakernel_fastpath import MuseDecodeFastPath

_W = 256  # kvwidth per rank: 2 KV heads x (K, V) x 64


def _writeback_reference(pool, kv_sw, kv_gl, kv_draft, meta, lmeta):
    """NumPy reference for writeback_blocks_local."""
    out = pool.copy()
    n_kp = kv_gl.shape[2]
    gl_flat = kv_gl.reshape(1, kv_gl.shape[1], n_kp * KV_GROUP, 128, _W)
    for j in range(meta.shape[1]):
        src, slot, roff, draft_ok = meta[:4, j]
        for kind, idx, g_row in lmeta:
            blk = meta[4 + g_row, j]
            if kind < 0 or blk < 0 or blk >= pool.shape[0]:
                continue
            if kind == 0:
                if src < 0:
                    continue
                vals = gl_flat[0, idx, src]
            else:
                if slot < 0 or (kind == 2 and draft_ok == 0):
                    continue
                ring = kv_sw if kind == 1 else kv_draft
                vals = ring[0, idx, slot, roff : roff + 128]
            out[blk] = vals.reshape(pool.shape[1:])
    return out


def _writeback_case(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_slots, n_kp, num_blocks = 3, 2, 24
    kv_sw = rng.standard_normal((1, 3, n_slots, SW_RING, _W)).astype(np.float32)
    kv_gl = rng.standard_normal((1, 2, n_kp, PAGE_TOKENS, _W)).astype(np.float32)
    kv_draft = rng.standard_normal((1, 1, n_slots, SW_RING, _W)).astype(np.float32)
    # Garbage-filled pool: entries the writeback skips must keep it bit for bit.
    pool = rng.standard_normal((num_blocks, 128, 1, 2, 128)).astype(np.float32)
    # Groups: row 0 global, row 1 sliding, row 2 draft.
    meta = np.array(
        [
            # src kv_gl subpage, slot, ring offset, draft ready, blk g0, g1, g2
            [5, 0, 0, 1, 3, 4, 5],
            [17, 1, 1920, 0, 6, 7, 8],  # draft not ready: draft layer skipped
            [-1, 2, 128, 1, 9, 10, 11],  # no superpage: global layer skipped
            [2, 1, 256, 1, -1, 12, -1],  # padding block ids
            [-1, -1, -1, 0, -1, -1, -1],  # bucket padding
        ],
        dtype=np.int32,
    ).T
    lmeta = np.array([[0, 1, 0], [1, 2, 1], [2, 0, 2], [-1, -1, -1]], np.int32)
    return pool, kv_sw, kv_gl, kv_draft, meta, lmeta


def test_writeback_blocks_local_matches_reference():
    """Whole-block writeback into a garbage-filled pool is bit-exact and skips
    padding, missing superpages and draft rows that drafting has not reached.
    """
    from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
        writeback_blocks_local,
    )

    pool, kv_sw, kv_gl, kv_draft, meta, lmeta = _writeback_case()
    got = np.asarray(
        jax.jit(writeback_blocks_local)(
            jnp.asarray(pool),
            jnp.asarray(kv_sw),
            jnp.asarray(kv_gl),
            jnp.asarray(kv_draft),
            jnp.asarray(meta),
            jnp.asarray(lmeta),
        )
    )
    want = _writeback_reference(pool, kv_sw, kv_gl, kv_draft, meta, lmeta)
    np.testing.assert_array_equal(got, want)
    # Spot checks against the sources themselves.
    np.testing.assert_array_equal(
        got[3].reshape(128, _W), kv_gl[0, 1, 0, 5 * 128 : 6 * 128]
    )
    np.testing.assert_array_equal(got[7].reshape(128, _W), kv_sw[0, 2, 1, 1920:2048])
    np.testing.assert_array_equal(got[8], pool[8])  # draft not ready
    np.testing.assert_array_equal(got[9], pool[9])  # no superpage
    np.testing.assert_array_equal(got[11].reshape(128, _W), kv_draft[0, 0, 2, 128:256])
    untouched = sorted(set(range(pool.shape[0])) - {3, 4, 5, 6, 7, 10, 11, 12})
    np.testing.assert_array_equal(got[untouched], pool[untouched])


class _FakeBlockTable:
    def __init__(self, rows: np.ndarray):
        self._rows = torch.from_numpy(rows)

    def get_cpu_tensor(self) -> torch.Tensor:
        return self._rows


def _writeback_fastpath(monkeypatch: pytest.MonkeyPatch, seed: int = 1):
    """A MuseDecodeFastPath with two requests' rings, superpages and block tables
    on CPU, and the real flush path running writeback_blocks_local through JAX.
    """
    import vllm_torchtpu.utils as vt_utils
    from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
        writeback_blocks_local,
    )
    from vllm_torchtpu.models.vllm.muse_spark.megakernel_fastpath import SlotTable

    monkeypatch.setattr(vt_utils, "synchronize_tensors", lambda *a, **k: None)
    rng = np.random.default_rng(seed)
    fp = object.__new__(MuseDecodeFastPath)
    fp.block_size = 128
    fp.n_slots = 2
    fp.max_kp = 2
    fp.spec = True
    fp.kv_writeback = True
    fp.prefix_cache_enabled = True
    fp.ready = False
    fp.wb_same_step_hits = 0
    fp._dev = torch.device("cpu")
    fp._ops = {}
    fp._wb_pending = {}
    fp.wb_flushes = fp.wb_blocks = 0
    fp.slots = SlotTable(fp.n_slots)
    fp.kpt_host = np.array([[3, 0], [1, 2]], np.int32)
    fp.kv_sw = torch.from_numpy(
        rng.standard_normal((1, 2, 2, SW_RING, _W)).astype(np.float32)
    )
    fp.kv_gl = torch.from_numpy(
        rng.standard_normal((1, 1, 4, PAGE_TOKENS, _W)).astype(np.float32)
    )
    fp.kv_draft = torch.from_numpy(
        rng.standard_normal((1, 1, 2, SW_RING, _W)).astype(np.float32)
    )
    # Two pools; draft layer 0 shares pool 1 with sliding layer 1 (group 1).
    fp._wb_gids = [0, 1, 2]
    garbage = [
        rng.standard_normal((40, 128, 1, 2, 128)).astype(np.float32) for _ in range(2)
    ]
    fp._wb_pools = [torch.from_numpy(g.copy()) for g in garbage]
    fp._wb_lmeta = [
        torch.tensor([[0, 0, 0], [1, 0, 1], [-1, -1, -1]], dtype=torch.int32),
        torch.tensor([[1, 1, 1], [2, 0, 2], [-1, -1, -1]], dtype=torch.int32),
    ]
    fp._wb_has_draft = True
    fp._wb_draft_dummy = None

    fp.wb_buckets_run = []

    def fake_fn(bucket):
        # Stands in for the compiled multi-pool program: same per-pool op, pools
        # updated in place.
        def run(pools, kv_sw, kv_gl, kv_draft, meta, lmetas):
            assert meta.shape[1] == bucket
            fp.wb_buckets_run.append(bucket)
            for pool, lmeta in zip(pools, lmetas, strict=True):
                out = writeback_blocks_local(
                    *(jnp.asarray(t.numpy()) for t in (pool, kv_sw, kv_gl, kv_draft)),
                    jnp.asarray(meta.numpy()),
                    jnp.asarray(lmeta.numpy()),
                )
                pool.copy_(torch.from_numpy(np.array(out)))

        return run

    fp._get_writeback_fn = fake_fn
    # Distinct block ids per (request, logical block, group).
    bt = {
        g: (np.arange(40, dtype=np.int32).reshape(2, 20) * 3 + g) % 40 for g in range(3)
    }
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table={g: _FakeBlockTable(bt[g]) for g in bt})
    )
    return fp, runner, bt, garbage


def test_writeback_queue_flush_and_release(monkeypatch: pytest.MonkeyPatch):
    """Filled blocks queue at fill time and reach vLLM's pools bit-exact on flush;
    unqueued blocks keep their garbage.
    """
    from vllm_torchtpu.models.vllm.muse_spark import megakernel_fastpath as mk_fp

    fp, runner, bt, garbage = _writeback_fastpath(monkeypatch)
    st_a = fp.slots.acquire("a")
    st_b = fp.slots.acquire("b")
    assert st_a is not None and st_b is not None
    # Request a took over at 200 tokens (block 0 prefilled by vLLM) and decoded to
    # 390: block 1 and 2 filled. Request b decoded 0 -> 2050 with drafts lagging.
    st_a.last_cached_tokens = 128
    st_a.draft_upto = 390
    st_b.last_cached_tokens = 2048 - 128
    st_b.draft_upto = 1900
    bts: dict[int, object] = {}
    assert fp._wb_queue_filled_blocks(runner, 0, "a", st_a, 390, bts) == 2
    assert fp._wb_queue_filled_blocks(runner, 1, "b", st_b, 2050, bts) == 1
    assert st_a.last_cached_tokens == 384 and st_b.last_cached_tokens == 2048
    # No new full block: nothing queued.
    assert fp._wb_queue_filled_blocks(runner, 0, "a", st_a, 400, bts) == 2

    # A step without a prefix hit does not flush; one with a hit does.
    no_hit = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        scheduled_new_reqs=[SimpleNamespace(num_computed_tokens=0)],
    )
    fp.before_vllm_forward(no_hit)
    assert sum(len(q) for q in fp._wb_pending.values()) == 3
    hit = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        scheduled_new_reqs=[SimpleNamespace(num_computed_tokens=256)],
    )
    fp.before_vllm_forward(hit)
    assert not fp._wb_pending and fp.wb_flushes == 1 and fp.wb_blocks == 3

    pool0, pool1 = (p.numpy() for p in fp._wb_pools)
    kv_sw, kv_gl, kv_draft = fp.kv_sw.numpy(), fp.kv_gl.numpy(), fp.kv_draft.numpy()
    written0, written1 = set(), set()
    for row, slot, lp, draft_ready in (
        (0, 0, 1, True),
        (0, 0, 2, True),
        (1, 1, 15, False),
    ):
        kp = fp.kpt_host[slot, lp // KV_GROUP]
        sub = lp % KV_GROUP
        gl = kv_gl[0, 0, kp, sub * 128 : (sub + 1) * 128]
        r0 = (lp * 128) % SW_RING
        np.testing.assert_array_equal(pool0[bt[0][row, lp]].reshape(128, _W), gl)
        np.testing.assert_array_equal(
            pool0[bt[1][row, lp]].reshape(128, _W), kv_sw[0, 0, slot, r0 : r0 + 128]
        )
        np.testing.assert_array_equal(
            pool1[bt[1][row, lp]].reshape(128, _W), kv_sw[0, 1, slot, r0 : r0 + 128]
        )
        written0 |= {bt[0][row, lp], bt[1][row, lp]}
        written1.add(bt[1][row, lp])
        if draft_ready:
            np.testing.assert_array_equal(
                pool1[bt[2][row, lp]].reshape(128, _W),
                kv_draft[0, 0, slot, r0 : r0 + 128],
            )
            written1.add(bt[2][row, lp])
    for pool, g, written in (
        (pool0, garbage[0], written0),
        (pool1, garbage[1], written1),
    ):
        keep = sorted(set(range(40)) - written)
        np.testing.assert_array_equal(pool[keep], g[keep])

    # release() writes the released request's queued blocks before freeing them.
    fp._free_slot_kpages = lambda slot: None
    fp._kp_reserved = 0
    fp._prefetched_step = None
    fp.req_mk_tokens = {}
    assert fp._wb_queue_filled_blocks(runner, 0, "a", st_a, 520, {}) == 1
    fp.release(["b"])
    assert fp._wb_pending  # b had nothing queued
    fp.release(["a"])
    assert not fp._wb_pending and fp.wb_flushes == 2

    # A request flushes before its oldest queued block can leave the ring or be
    # freed by vLLM's sliding-window manager.
    assert mk_fp.SPEC_MAX_ROWS < 128
    assert mk_fp.WB_BUCKETS[-1] == mk_fp.MAX_SLOTS * mk_fp.WB_FLUSH_BLOCKS
    assert (
        mk_fp.WB_FLUSH_BLOCKS * 128 + 2 * mk_fp.SPEC_MAX_ROWS
        < mk_fp.wb_min_window(128)
        <= SW_RING
    )


def test_writeback_warmup_covers_every_serving_flush(monkeypatch: pytest.MonkeyPatch):
    """warmup() runs the serving flush routine for every row bucket, and no flush
    size uses a bucket that warmup did not compile.
    """
    from vllm_torchtpu.models.vllm.muse_spark import megakernel_fastpath as mk_fp

    fp, _, _, garbage = _writeback_fastpath(monkeypatch)
    fp._warmup_writeback()
    warmed = set(fp.wb_buckets_run)
    assert warmed == set(mk_fp.WB_BUCKETS)
    # All block ids are -1: warmup writes nothing.
    for pool, g in zip(fp._wb_pools, garbage):
        np.testing.assert_array_equal(pool.numpy(), g)
    row = np.full((mk_fp.WB_META_ROWS + len(fp._wb_gids),), -1, np.int32)
    for n in range(1, 2 * mk_fp.WB_BUCKETS[-1] + 2):
        fp.wb_buckets_run = []
        fp._writeback_rows([row] * n)
        assert set(fp.wb_buckets_run) <= warmed
        assert len(fp.wb_buckets_run) == -(-n // mk_fp.WB_BUCKETS[-1])


def test_writeback_window_check():
    """Deferred writeback rejects a KV group that frees blocks before a flush."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    from vllm_torchtpu.models.vllm.muse_spark import megakernel_fastpath as mk_fp

    def spec(window):
        kw = dict(block_size=128, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
        if window is None:
            return FullAttentionSpec(**kw)
        return SlidingWindowSpec(sliding_window=window, **kw)

    fp = object.__new__(MuseDecodeFastPath)
    fp.block_size = 128
    fp._wb_gids = [0, 1, 2]
    # Production: global group, 2048-token sliding group, full-attention draft.
    fp._gid_specs = {0: spec(None), 1: spec(2048), 2: spec(None)}
    fp._check_writeback_windows()
    fp._gid_specs[2] = spec(mk_fp.wb_min_window(128))
    fp._check_writeback_windows()
    fp._gid_specs[2] = spec(1024)
    with pytest.raises(NotImplementedError, match="group 2 keeps 1024"):
        fp._check_writeback_windows()
    # Groups the writeback never touches are not checked.
    fp._wb_gids = [0, 1]
    fp._check_writeback_windows()
    # Unknown or block-freeing spec types and missing specs are rejected.
    from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec

    fp._wb_gids = [0, 1, 2]
    kw = dict(block_size=128, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    for bad in (ChunkedLocalAttentionSpec(attention_chunk_size=8192, **kw), None):
        fp._gid_specs[2] = bad
        with pytest.raises(NotImplementedError, match="group 2 has"):
            fp._check_writeback_windows()


def test_writeback_rejects_aliased_pools(monkeypatch: pytest.MonkeyPatch):
    """Two pools that are views of one buffer would be aliased mutated inputs of
    the flush program, so binding rejects them.
    """
    fp, _, _, _ = _writeback_fastpath(monkeypatch)
    fp._gid_specs = {}
    fp.spec = False
    fp._draft_pools, fp._draft_gids = [], []
    fp._sw_pools, fp._sw_gids = [], []
    fp._gl_gids = [0, 0]
    fp._check_writeback_windows = lambda: None
    buf = torch.zeros((2, 40, 128, 1, 2, 128))
    fp._gl_pools = [buf[0], torch.zeros((40, 128, 1, 2, 128))]
    fp._bind_writeback_pools(_W)
    assert len(fp._wb_pools) == 2
    fp._gl_pools = [buf[0], buf[1]]
    with pytest.raises(NotImplementedError, match="own its storage"):
        fp._bind_writeback_pools(_W)


def test_same_step_prefix_hit_detection(monkeypatch: pytest.MonkeyPatch):
    """A new request hitting the block an owned request completes in this step is
    counted; hits on older blocks are not.
    """
    fp, runner, bt, _ = _writeback_fastpath(monkeypatch)
    fp.ready = True
    fp._gl_group_id = 0
    st_a = fp.slots.acquire("a")
    st_b = fp.slots.acquire("b")
    assert st_a is not None and st_b is not None
    st_a.owned = st_b.owned = True
    runner.requests = {}
    fp.runner = runner
    # a computes position 383 (block 2's last token) this step; b is mid-block.
    runner.input_batch.num_reqs = 2
    runner.input_batch.req_ids = ["a", "b"]
    runner.input_batch.num_computed_tokens_cpu = np.array([383, 300], np.int32)

    def step(blocks, n_computed):
        return SimpleNamespace(
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="n",
                    num_computed_tokens=n_computed,
                    block_ids=(blocks, [], []),
                )
            ],
        )

    fp.before_vllm_forward(step([int(bt[0][0, j]) for j in range(2)], 256))
    assert fp.wb_same_step_hits == 0
    fp.before_vllm_forward(step([int(bt[0][0, j]) for j in range(4)], 384))
    assert fp.wb_same_step_hits == 1
    # b's open block is not cached by vLLM yet, so it is not counted.
    fp.before_vllm_forward(step([int(bt[0][1, j]) for j in range(3)], 384))
    assert fp.wb_same_step_hits == 1


def test_from_checkpoint_rejects_unsupported_layer_layout(tmp_path):
    import json

    layers = 8
    cfg = {
        "num_hidden_layers": layers,
        "layer_types": [
            "full_attention" if i % 4 == 1 else "sliding_attention"
            for i in range(layers)
        ],
        "hidden_size": 8192,
        "num_attention_heads": 128,
        "num_key_value_heads": 16,
        "head_dim": 64,
        "vocab_size": 202048,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "moe_hidden_size": 4096,
        "moe_intermediate_size": 4096,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "qk_scale_factor": 5.47,
        "output_multiplier": 0.18,
        "final_logit_softcapping": 20.0,
        "residual_gate_temperature": 0.3,
        "sliding_window": SW_RING,
        "rope_parameters": {"rope_theta": 500000.0},
        "eos_token_id": [200001, 200008, 200007],
    }

    def load(**overrides):
        (tmp_path / "config.json").write_text(json.dumps({**cfg, **overrides}))
        return MuseMegakernelConfig.from_checkpoint(tmp_path)

    c = load()
    assert c.full_attention == (1, 5)
    assert c.eos == (200001, 200008, 200007)
    assert load(eos_token_id=200001).eos == (200001,)
    shifted = ["full_attention" if i % 4 == 0 else "x" for i in range(layers)]
    with pytest.raises(NotImplementedError, match="i % 4 == 1"):
        load(layer_types=shifted)
    with pytest.raises(NotImplementedError, match="sliding_window"):
        load(sliding_window=4096)


@pytest.mark.multichip
def test_hybrid_decode_step_single_custom_call_and_128k_streaming():
    """Verify single tpu_custom_call lowering and hybrid KV execution at 1K and 120K
    positions.
    """
    if jax.default_backend() != "tpu" or len(jax.devices()) != 8:
        pytest.skip("Requires 8 TPU TensorCores")
    c = MuseMegakernelConfig(
        dim=8192,
        layers=4,
        heads=128,
        kv_heads=16,
        head_dim=64,
        vocab=202048,
        moe_hidden=4096,
        moe_intermediate=4096,
        experts=256,
        top_k=8,
        full_attention=(1,),
        bf16_expert_layers=(0,),
    )
    mesh = logical_mesh(8)
    tp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(TP_AXIS_NAME))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    max_slots = 16
    max_kp = 64  # 64 * 2048 = 131,072 tokens (128K)
    num_kpages = max_slots * max_kp + 2

    w = canonical_test_weights(mesh, c, seed=42)
    kv = empty_hybrid_kv_cache(
        mesh, c, max_slots=max_slots, num_kpages=num_kpages, max_kp=max_kp
    )

    for b in (1, 2, 4, 8, 16):
        step = make_hybrid_decode_step(
            mesh,
            c,
            max_slots=max_slots,
            num_kpages=num_kpages,
            max_kp=max_kp,
            batch_size=b,
        )
        w_shapes = {
            k: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=tp) for k, v in w.items()
        }
        sw_s = jax.ShapeDtypeStruct(kv["kv_sw"].shape, kv["kv_sw"].dtype, sharding=tp)
        gl_s = jax.ShapeDtypeStruct(kv["kv_gl"].shape, kv["kv_gl"].dtype, sharding=tp)
        tok_s = jax.ShapeDtypeStruct((b,), jnp.int32, sharding=rep)
        ctrl_s = jax.ShapeDtypeStruct((2, b), jnp.int32, sharding=rep)
        kpt_s = jax.ShapeDtypeStruct(kv["kpt"].shape, jnp.int32, sharding=rep)
        hlo = step.lower(w_shapes, sw_s, gl_s, tok_s, ctrl_s, kpt_s).as_text()
        assert hlo.count("tpu_custom_call") == 1, (
            f"Expected 1 tpu_custom_call for batch={b}"
        )

    step1 = make_hybrid_decode_step(
        mesh,
        c,
        max_slots=max_slots,
        num_kpages=num_kpages,
        max_kp=max_kp,
        batch_size=1,
    )
    for pos in (0, 1024, 4096, 32768, 122880):
        tok = jax.device_put(np.asarray([100], np.int32), rep)
        ctrl = jax.device_put(np.asarray([[pos], [0]], np.int32), rep)
        targets, aux, kv["kv_sw"], kv["kv_gl"] = step1(
            w, kv["kv_sw"], kv["kv_gl"], tok, ctrl, kv["kpt"]
        )
        assert targets.shape == (1,)
        assert np.all(np.isfinite(np.asarray(aux, dtype=np.float32)))


@pytest.mark.multichip
def test_hybrid_decode_head_covers_full_vocab_and_returns_logits():
    """The last vocab column of the last shard is reachable; return_logits agrees with
    the
    argmax.
    """
    if jax.default_backend() != "tpu" or len(jax.devices()) != 8:
        pytest.skip("Requires 8 TPU TensorCores")
    c = MuseMegakernelConfig(layers=2, full_attention=(1,), bf16_expert_layers=(0,))
    mesh = logical_mesh(8)
    tp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(TP_AXIS_NAME))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    vshard = c.vocab // 8
    w = canonical_test_weights(mesh, c, seed=7)
    kv = empty_hybrid_kv_cache(mesh, c, max_slots=1, num_kpages=3, max_kp=2)
    step = make_hybrid_decode_step(
        mesh, c, max_slots=1, num_kpages=3, max_kp=2, batch_size=1, return_logits=True
    )
    shapes = [
        {k: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=tp) for k, v in w.items()},
        jax.ShapeDtypeStruct(kv["kv_sw"].shape, kv["kv_sw"].dtype, sharding=tp),
        jax.ShapeDtypeStruct(kv["kv_gl"].shape, kv["kv_gl"].dtype, sharding=tp),
        jax.ShapeDtypeStruct((1,), jnp.int32, sharding=rep),
        jax.ShapeDtypeStruct((2, 1), jnp.int32, sharding=rep),
        jax.ShapeDtypeStruct(kv["kpt"].shape, jnp.int32, sharding=rep),
    ]
    assert step.lower(*shapes).as_text().count("tpu_custom_call") == 1

    tok = jax.device_put(np.asarray([100], np.int32), rep)
    ctrl = jax.device_put(np.asarray([[0], [0]], np.int32), rep)
    targets, aux, kv["kv_sw"], kv["kv_gl"], logits = step(
        w, kv["kv_sw"], kv["kv_gl"], tok, ctrl, kv["kpt"]
    )
    logits = np.asarray(logits)
    assert logits.shape == (1, c.vocab)
    assert np.all(np.abs(logits) <= c.soft_cap)
    assert int(np.asarray(targets)[0]) == int(np.argmax(logits[0]))

    # Align the head column of token vocab - 1 (rank 7, last real column) with the final
    # hidden state.
    hidden = np.asarray(aux, np.float32).reshape(-1)[: c.dim]
    column = jnp.asarray(hidden / np.linalg.norm(hidden) * 4.0, jnp.bfloat16)
    w["lmw"] = jax.jit(lambda a: a.at[7, :, vshard - 1].set(column), out_shardings=tp)(
        w["lmw"]
    )
    targets, _, kv["kv_sw"], kv["kv_gl"], logits = step(
        w, kv["kv_sw"], kv["kv_gl"], tok, ctrl, kv["kpt"]
    )
    assert int(np.asarray(targets)[0]) == c.vocab - 1
    assert int(np.argmax(np.asarray(logits)[0])) == c.vocab - 1


@pytest.mark.multichip
def test_hybrid_decode_b1_b8_b16_bit_exact():
    """Verify batch_size=1, 8, 16 (with force_rows8=False) produce bit-identical
    hidden states and logits.
    """
    if jax.default_backend() != "tpu" or len(jax.devices()) != 8:
        pytest.skip("Requires 8 TPU TensorCores")
    c = MuseMegakernelConfig(layers=2, full_attention=(1,), bf16_expert_layers=(0,))
    mesh = logical_mesh(8)
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    w = canonical_test_weights(mesh, c, seed=11)
    max_slots = 16
    max_kp = 4
    num_kpages = max_slots * max_kp + 2

    outs_by_b = {}
    for b in (1, 8, 16):
        kv = empty_hybrid_kv_cache(
            mesh, c, max_slots=max_slots, num_kpages=num_kpages, max_kp=max_kp
        )
        step = make_hybrid_decode_step(
            mesh,
            c,
            max_slots=max_slots,
            num_kpages=num_kpages,
            max_kp=max_kp,
            batch_size=b,
            force_rows8=False,
            return_logits=True,
        )
        tok = jax.device_put(np.full((b,), 321, np.int32), rep)
        ctrl = np.full((2, b), -1, np.int32)
        ctrl[0, 0] = 0
        ctrl[1, :] = np.arange(b, dtype=np.int32) % max_slots
        ctrl_d = jax.device_put(ctrl, rep)
        targets, aux, _, _, logits = step(
            w, kv["kv_sw"], kv["kv_gl"], tok, ctrl_d, kv["kpt"]
        )
        outs_by_b[b] = (
            int(np.asarray(targets)[0]),
            np.asarray(aux, np.float32)[0],
            np.asarray(logits, np.float32)[0],
        )

    assert outs_by_b[1][0] == outs_by_b[8][0] == outs_by_b[16][0]
    np.testing.assert_array_equal(outs_by_b[1][1], outs_by_b[8][1])
    np.testing.assert_array_equal(outs_by_b[1][1], outs_by_b[16][1])
    np.testing.assert_array_equal(outs_by_b[1][2], outs_by_b[8][2])
    np.testing.assert_array_equal(outs_by_b[1][2], outs_by_b[16][2])

    # Also verify spec grouped num_reqs=1, batch_size=8 matches sequential b=1
    kv_seq = empty_hybrid_kv_cache(
        mesh, c, max_slots=max_slots, num_kpages=num_kpages, max_kp=max_kp
    )
    step1 = make_hybrid_decode_step(
        mesh,
        c,
        max_slots=max_slots,
        num_kpages=num_kpages,
        max_kp=max_kp,
        batch_size=1,
        return_logits=True,
    )
    seq_toks = [101, 202, 303, 404]
    seq_logits = []
    for k_idx, tk in enumerate(seq_toks):
        tok_1 = jax.device_put(np.asarray([tk], np.int32), rep)
        ctrl_1 = jax.device_put(np.asarray([[4096 + k_idx], [0]], np.int32), rep)
        _, _, kv_seq["kv_sw"], kv_seq["kv_gl"], lg_1 = step1(
            w, kv_seq["kv_sw"], kv_seq["kv_gl"], tok_1, ctrl_1, kv_seq["kpt"]
        )
        seq_logits.append(np.asarray(lg_1, np.float32)[0])

    kv_grp = empty_hybrid_kv_cache(
        mesh, c, max_slots=max_slots, num_kpages=num_kpages, max_kp=max_kp
    )
    step_grp = make_hybrid_decode_step(
        mesh,
        c,
        max_slots=max_slots,
        num_kpages=num_kpages,
        max_kp=max_kp,
        batch_size=8,
        num_reqs=1,
        return_logits=True,
    )
    tok_8 = np.full((8,), seq_toks[0], np.int32)
    tok_8[:4] = seq_toks
    ctrl_8 = np.full((2, 8), -1, np.int32)
    ctrl_8[0, :4] = [4096 + k_idx for k_idx in range(4)]
    ctrl_8[1, :] = 0
    _, _, _, _, lg_8 = step_grp(
        w,
        kv_grp["kv_sw"],
        kv_grp["kv_gl"],
        jax.device_put(tok_8, rep),
        jax.device_put(ctrl_8, rep),
        kv_grp["kpt"],
    )
    lg_8_np = np.asarray(lg_8, np.float32)
    for k_idx in range(4):
        np.testing.assert_array_equal(seq_logits[k_idx], lg_8_np[k_idx])


@pytest.mark.tpu_test
def test_ring_rows_coalesced_read_write_bit_exact():
    """Verify coalesced 16-row block DMAs in ring_rows_read/write_local."""
    if jax.default_backend() != "tpu" or len(jax.devices()) < 1:
        pytest.skip("Requires TPU")
    from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
        ring_rows_read_local,
        ring_rows_write_local,
    )

    dev0 = jax.devices()[0]
    rng = np.random.default_rng(123)
    kv_np = rng.standard_normal((1, 5, 4, 64, 256), dtype=np.float32)
    kv_j = jax.device_put(jnp.asarray(kv_np, dtype=jnp.bfloat16), dev0)
    kv_ref = np.asarray(kv_j, dtype=np.float32).copy()

    # Cross 16-row boundaries, switch slots, include duplicates and masked-out rows.
    slots = np.asarray([0, 0, 0, 0, 1, 1, 0, 2], dtype=np.int32)
    ridx = np.asarray([14, 15, 16, 17, 3, 3, 15, 31], dtype=np.int32)
    mask = np.asarray([1, 1, 0, 1, 1, 1, 1, 0], dtype=np.int32)

    got_rd = np.asarray(
        jax.jit(ring_rows_read_local)(
            kv_j, jax.device_put(slots, dev0), jax.device_put(ridx, dev0)
        ),
        dtype=np.float32,
    )
    exp_rd = np.stack(
        [kv_ref[0, :, slots[j], ridx[j], :] for j in range(len(slots))], axis=1
    )[None]
    np.testing.assert_array_equal(got_rd, exp_rd)

    vals_np = np.asarray(
        jnp.asarray(
            rng.standard_normal((1, 5, len(slots), 256), dtype=np.float32),
            dtype=jnp.bfloat16,
        ),
        dtype=np.float32,
    )
    kv_out = jax.jit(ring_rows_write_local)(
        kv_j,
        jax.device_put(vals_np, dev0),
        jax.device_put(slots, dev0),
        jax.device_put(ridx, dev0),
        jax.device_put(mask, dev0),
    )
    exp_kv = kv_ref.copy()
    for j in range(len(slots)):
        if mask[j] != 0:
            exp_kv[0, :, slots[j], ridx[j], :] = vals_np[0, :, j, :]
    np.testing.assert_array_equal(np.asarray(kv_out, dtype=np.float32), exp_kv)

    # All-zero mask must leave KV untouched.
    kv_zero = jax.jit(ring_rows_write_local)(
        kv_out,
        jax.device_put(vals_np, dev0),
        jax.device_put(slots, dev0),
        jax.device_put(ridx, dev0),
        jax.device_put(np.zeros_like(mask), dev0),
    )
    np.testing.assert_array_equal(np.asarray(kv_zero, dtype=np.float32), exp_kv)


def _megakernel_vllm_cfg(
    *,
    tp: int = 8,
    dp: int = 1,
    pp: int = 1,
    kv_transfer_config: object = None,
    spec_method: str | None = None,
    async_scheduling: bool | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        cache_config=SimpleNamespace(block_size=128, user_specified_block_size=True),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=True,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=(
            None
            if spec_method is None
            else SimpleNamespace(method=spec_method, draft_model_config=None)
        ),
        kv_transfer_config=kv_transfer_config,
    )


def test_megakernel_config_disables_async_scheduling_and_validates_tp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jax.experimental.pallas import tpu as pltpu

    import vllm_torchtpu.tpu_info as tpu_info
    from vllm_torchtpu.models.vllm.muse_spark import megakernel_fastpath as mk_fp
    from vllm_torchtpu.models.vllm.muse_spark import text_model as tm

    monkeypatch.setenv("VLLM_TPU_MUSE_MEGAKERNEL", "1")
    monkeypatch.setattr(tpu_info, "get_chip_version", lambda: pltpu.ChipVersion.TPU_7X)
    # The production setup: TP=8, DP=1, expert parallel, DSpark.
    for initial_async in (None, True):
        vllm_cfg = _megakernel_vllm_cfg(
            spec_method="dspark", async_scheduling=initial_async
        )
        tm.TPUMuseSparkConfig.verify_and_update_config(vllm_cfg)
        assert vllm_cfg.scheduler_config.async_scheduling is False

    for kwargs, match in (
        ({"tp": 4}, "tensor_parallel_size=8"),
        ({"dp": 2}, "data_parallel_size"),
        ({"pp": 2}, "pipeline_parallel_size"),
        ({"kv_transfer_config": object()}, "KV connectors"),
        ({"spec_method": "eagle3"}, "dspark"),
        ({"spec_method": "mtp"}, "dspark"),
    ):
        with pytest.raises(NotImplementedError, match=match):
            tm.TPUMuseSparkConfig.verify_and_update_config(
                _megakernel_vllm_cfg(**kwargs)
            )
    monkeypatch.setattr(tpu_info, "get_chip_version", lambda: pltpu.ChipVersion.TPU_V6E)
    with pytest.raises(NotImplementedError, match="v7x"):
        tm.TPUMuseSparkConfig.verify_and_update_config(_megakernel_vllm_cfg())

    assert [mk_fp.resolve_max_slots(n) for n in (1, 3, 8, 12, 32, 256)] == [
        1,
        4,
        8,
        16,
        32,
        32,
    ]
    assert mk_fp.resolve_max_slots(64, spec=True) == 32
