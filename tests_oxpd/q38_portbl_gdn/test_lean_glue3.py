# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL GDN: VLLM_TPU_GDN_LEAN_GLUE3 gives today's bytes with less glue.

Ported from Benliang Wang's tests/kernels/gdn/v3/test_lean_glue.py (commit aa8e6b237 on
benliangw/q38fh-integ-pin-5b44702aea; credit), for the three pieces that keep the SELECT checkpoint layout D14 shares:
stacked slot rows, zero fill by DMA, b and a packed into one operand. His fourth piece (no checkpoint-index operand)
needs the GDN window log and is not ported.

Legs (each prints one line; the CPU proof job reads them):

1. Kernel pieces on dense states (Pallas TPU interpreter on CPU, uninitialized memory reads as NaN, so a row the
   kernel misses shows up): zero_fill, ba_packed and slot_rows each byte-identical to the flag-off kernel.
2. BW: the REAL pooled path (the adapter's jax-op function from ``VllmGatedDeltaNetAttention._build_pooled_gdn_op``
   -> layers/core run_jax_gdn_attention_pooled -> v3 wrapper -> kernels in interpret mode) at 80 / 96 / 128 / 192 /
   256 rows, Qwen3.8 Flash TP8 per-rank geometry (2 kq / 6 v heads, fp8 NHD pool; the served 1664-token pool pages at
   80-128 rows, 128-token kernel pages at 192 / 256 to bound CPU memory), MTP K = 3 (4 rows per request), with the
   served GDN keys (WINDOW_TILE, CONV_WEIGHT_KMAJOR, dynamic tiling, QK pair layout off). Three GDN layers per step,
   each with its own pool and its own block ids (each Qwen3.8 GDN layer is its own KV-cache group); the stacks come
   from the adapter's own ``_stacked_slot_rows`` over the forward context. Every layer's output (padding rows
   included, which today are zeros) and every pool byte (recurrent state and checkpoint blocks) must be equal flag
   on vs flag off. Cases: a verify step (full 4-row windows, a fresh request, bucket padding), a mixed step (1-4 row
   windows incl. a 1-token request), and at 80 rows a verify + prefill-chunk step (PER_SEQ by absolute request
   index, HOIST_READ_OFFSETS subsumed) also run with VLLM_TPU_GDN_HOIST_READ_OFFSETS on.
   The read offsets differ per layer here (in serving a request's offset is the same in every layer), so a kernel
   that read another layer's row would differ.
3. ENG: with the flag on the stacked gather is built in every layer's call and the zero-fill / packed / slot-row
   kernels are traced; with it off none is.
4. Wiring: forward passes (stacks, row) only with the flag; the flag-off op keeps today's name and 14 operands; the
   engagement line; the per-step gather count in compiled XLA (CPU) for a three-layer step.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, PartitionSpec, use_abstract_mesh
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import metadata, wrapper
from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op as op
from vllm_torchtpu.layers.adapter.custom_ops import gdn_lean_engagement as eng
from vllm_torchtpu.layers.core import gdn_attention as core
from vllm_torchtpu.models.vllm.qwen4_exp.gdn import Qwen4ExpGatedDeltaNetAttention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context

# The pooled kernels need test_gdn_v3_window_tile's interpreter view shim to run on CPU (its module docstring);
# its pools and geometries are the Flash ones used there.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
import test_gdn_v3_window_tile as wt  # noqa: E402

W = 4  # MTP K = 3
N_KQ, N_V, DK = 2, 6, 128  # Qwen3.8 Flash per rank at TP8
DIM = 2 * N_KQ * DK + N_V * DK  # 1280
LAYERS = 3


# ---------------------------------------------------------------------------
# 1. Kernel pieces on dense states (adapted from his test_lean_glue.py).
# ---------------------------------------------------------------------------

_N_KQ1, _N_V1, _D1, _KS1 = 1, 2, 128, 4
_DIM1 = 2 * _N_KQ1 * _D1 + _N_V1 * _D1


def _run_dense(lean, lengths, num_decode, rows, slot_rows_row=None):
    num_seqs = len(lengths)
    rngs = iter(jax.random.split(jax.random.key(7), 9))
    qkv = jax.random.normal(next(rngs), (rows, _DIM1), dtype=jnp.bfloat16)
    b = jax.random.normal(next(rngs), (rows, _N_V1), dtype=jnp.bfloat16)
    a = jax.random.normal(next(rngs), (rows, _N_V1), dtype=jnp.bfloat16)
    conv_w = jax.random.normal(next(rngs), (_DIM1, 1, _KS1), dtype=jnp.bfloat16)
    conv_b = jax.random.normal(next(rngs), (_DIM1,), dtype=jnp.bfloat16)
    a_log = jax.random.normal(next(rngs), (_N_V1,), dtype=jnp.float32)
    dt_bias = jax.random.normal(next(rngs), (_N_V1,), dtype=jnp.float32)
    init_conv = jax.random.normal(next(rngs), (num_seqs, _KS1 - 1, _DIM1), dtype=jnp.bfloat16)
    init_rec = jax.random.normal(next(rngs), (num_seqs, _N_V1, _D1, _D1), dtype=jnp.float32)
    q_loc = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), jnp.int32)
    state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
    slot_rows = None
    if slot_rows_row is not None:
        # Three layers; the other rows name other sequences' state slots.
        stack = np.stack([np.roll(np.arange(num_seqs), k + 1) for k in range(3)])
        stack[slot_rows_row] = np.arange(num_seqs)
        slot_rows = metadata.GdnSlotRows(
            jnp.asarray(stack, jnp.int32), jnp.zeros((3, num_seqs), jnp.int32),
            jnp.asarray([slot_rows_row], jnp.int32),
        )
        # The per-layer array gives only the shape: poison it.
        state_indices = jnp.full((num_seqs,), num_seqs - 1, jnp.int32)
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    jax.clear_caches()
    pltpu.reset_tpu_interpret_mode_state()
    with (
        use_abstract_mesh(mesh),
        mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
        pltpu.force_tpu_interpret_mode(
            pltpu.InterpretParams(dma_execution_mode="eager", skip_floating_point_ops=False,
                                  uninitialized_memory="nan")
        ),
    ):
        (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
            qkv=qkv, b=b, a=a, conv_state=init_conv, recurrent_state=init_rec, conv_weight=conv_w,
            conv_bias=conv_b, a_log=a_log, dt_bias=dt_bias, query_start_loc=q_loc, state_indices=state_indices,
            distribution=jnp.asarray([num_decode, num_seqs, num_seqs], jnp.int32),
            seq_lens=jnp.full((num_seqs,), 11, dtype=jnp.int32), n_kq=_N_KQ1, n_v=_N_V1, d_k=_D1, d_v=_D1,
            kernel_size=_KS1, decode_tile_size=2, mixed_tile_size=1, lean_glue=lean, slot_rows=slot_rows,
        )
        jax.block_until_ready((conv_out, rec_out, out))
    return np.asarray(conv_out), np.asarray(rec_out), np.asarray(out)


def _same_bytes(off, on):
    return all(np.array_equal(np.asarray(x).view(np.uint8), np.asarray(y).view(np.uint8)) for x, y in zip(off, on))


@pytest.mark.parametrize(
    "lengths,num_decode,rows",
    [([1] * 6, 6, 9), ([1] * 4 + [1, 1], 4, 10), ([1] * 6, 6, 6)],
    ids=["decode_pad", "decode_perseq_pad", "no_pad"],
)
def test_zero_fill_is_byte_identical(lengths, num_decode, rows):
    off = _run_dense(False, lengths, num_decode, rows)
    before = wrapper.LEAN_GLUE_TRACES["zero_fill"]
    on = _run_dense(("zero_fill",), lengths, num_decode, rows)
    ok = _same_bytes(off, on)
    print(f"PBL_GDN_K zero_fill {lengths} rows {rows}: {'EQUAL' if ok else 'DIFF'} traced +"
          f"{wrapper.LEAN_GLUE_TRACES['zero_fill'] - before}")
    assert ok
    assert wrapper.LEAN_GLUE_TRACES["zero_fill"] > before
    assert np.all(on[2][sum(lengths):] == 0)
    assert np.isfinite(on[2].astype(np.float32)).all()


_LAYOUTS = [([1] * 6, 6, 9), ([1] * 4 + [1, 1], 4, 10)]


@pytest.mark.parametrize("lengths,num_decode,rows", _LAYOUTS, ids=["decode", "decode_perseq"])
def test_ba_packed_is_byte_identical(lengths, num_decode, rows):
    before = wrapper.LEAN_GLUE_TRACES["ba_packed"]
    ok = _same_bytes(_run_dense(False, lengths, num_decode, rows), _run_dense(("ba_packed",), lengths, num_decode, rows))
    print(f"PBL_GDN_K ba_packed {lengths}: {'EQUAL' if ok else 'DIFF'}")
    assert ok
    assert wrapper.LEAN_GLUE_TRACES["ba_packed"] > before


@pytest.mark.parametrize("row", [0, 2])
@pytest.mark.parametrize("lengths,num_decode,rows", _LAYOUTS, ids=["decode", "decode_perseq"])
def test_slot_rows_are_byte_identical(lengths, num_decode, rows, row):
    before = wrapper.LEAN_GLUE_TRACES["slot_rows"]
    ok = _same_bytes(_run_dense(False, lengths, num_decode, rows),
                     _run_dense(False, lengths, num_decode, rows, slot_rows_row=row))
    print(f"PBL_GDN_K slot_rows row {row} {lengths}: {'EQUAL' if ok else 'DIFF'}")
    assert ok
    assert wrapper.LEAN_GLUE_TRACES["slot_rows"] > before


def test_per_seq_slot_rows_need_absolute_index():
    cfg = SimpleNamespace(batch_size=8, chunk_size=4, tile_size=4, seq_tile_size=1)
    rows = metadata.GdnSlotRows(jnp.zeros((2, 4), jnp.int32), jnp.zeros((2, 4), jnp.int32),
                                jnp.zeros((1,), jnp.int32))
    with pytest.raises(ValueError, match="absolute request index"):
        metadata.compute_per_seq_metadata(
            cfg, jnp.ones(4, jnp.int32), jnp.arange(5, dtype=jnp.int32), jnp.zeros(4, jnp.int32),
            jnp.zeros(4, jnp.int32), jnp.int32(0), jnp.int32(4), slot_rows=rows)


# ---------------------------------------------------------------------------
# 2-3. BW + ENG through the real pooled op path.
# ---------------------------------------------------------------------------


def _identity_shard_map(f, *, mesh, in_specs, out_specs, check_vma=True):
    """A one-device "model" axis: shard_map is the identity on whole arrays. Checks that every operand has a spec
    and that None operands (and only those) have None specs, which is what the edited in_specs must keep."""
    del mesh, out_specs, check_vma

    def call(*args):
        specs = (in_specs,) if isinstance(in_specs, PartitionSpec) else tuple(in_specs)
        assert len(specs) == len(args), (len(specs), len(args))
        for i, (spec, arg) in enumerate(zip(specs, args)):
            assert (spec is None) == (arg is None), (i, spec, type(arg))
        return f(*args)

    return call


@pytest.fixture
def served_gdn(monkeypatch):
    """Served GDN keys (v8 record env), the interpreter (with the pooled view shim), a one-device mesh."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    for name, value in (
        ("TPU_ENABLE_GDN_DYNAMIC_TILING", True), ("VLLM_TPU_GDN_WINDOW_TILE", True),
        ("VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR", True), ("TPU_GDN_CONV_QK_PAIR_LAYOUT", False),
        ("VLLM_TPU_GDN_HOIST_READ_OFFSETS", False), ("VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT", False),
        ("VLLM_TPU_GDN_PREFILL_BF16_QKV_2D", False), ("VLLM_TPU_GDN_PREFILL_QKV_FROM_PROJ", False),
        ("VLLM_TPU_BLOCK_MAJOR_KV", False), ("VLLM_TPU_GDN_LEAN_GLUE3", False),
    ):
        monkeypatch.setattr(envs, name, value)
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    monkeypatch.setattr(jax, "shard_map", _identity_shard_map)
    monkeypatch.setattr(op, "_POOLED_GDN_PREFIXES", set())
    eng.reset()
    wt._install_interpret_views(monkeypatch)
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    jax.clear_caches()
    with (
        set_current_vllm_config(config),
        use_abstract_mesh(mesh),
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(uninitialized_memory="nan")),
    ):
        yield monkeypatch
    jax.clear_caches()


def _layer(prefix: str):
    attn = Qwen4ExpGatedDeltaNetAttention.__new__(Qwen4ExpGatedDeltaNetAttention)
    attn.prefix = prefix
    attn.num_k_heads, attn.num_v_heads, attn.tp_size = N_KQ, N_V, 1
    attn.head_k_dim = attn.head_v_dim = DK
    attn.conv_kernel_size, attn.num_spec = 4, W - 1
    attn.gqa_interleaved_layout = False
    attn.conv1d = SimpleNamespace(bias=None)
    attn.model_config = SimpleNamespace(dtype=torch.bfloat16, architecture="Qwen3_5MoeForCausalLM")
    attn.cache_config = SimpleNamespace(mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="float32")
    attn._pooled_conv_state_dtype = torch.float32
    return attn


def _build(monkeypatch, lean: bool, pool_block_tokens: int):
    """Builds LAYERS layers' pooled ops with the flag on or off; returns (layers, jax-op function, op name)."""
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_LEAN_GLUE3", lean)
    captured = []

    def fake_jax_op(name, fn, **kw):
        captured.append((name, fn, kw))
        return mock.MagicMock()

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, tensor_parallel_size=1,
                                        cp_kv_cache_interleave_size=16, data_parallel_size=1),
        cache_config=SimpleNamespace(block_size=pool_block_tokens, mamba_block_size=4096,
                                     mamba_page_size_padded=None, mamba_cache_mode=None,
                                     use_kda_recoverssm=False, use_replayssm=False),
        speculative_config=None, num_speculative_tokens=W - 1,
    )
    layers = [_layer(f"model.layers.{i}.linear_attn") for i in range(LAYERS)]
    with (
        mock.patch.dict(op._shared_gdn_ops, {}, clear=True),
        set_vllm_model_wrapper_context(mesh=SimpleNamespace(shape={"model": 1}), vllm_config=vllm_config),
        mock.patch.object(op.pallas, "jax_op", side_effect=fake_jax_op),
    ):
        for attn in layers:
            attn._build_pooled_gdn_op()
            attn._register_lean_glue3()
    (name, fn, kw), = captured  # one shared op for the identical layers
    assert kw.get("donate_argnums") == (3,)
    return layers, fn, name


def _geo(rows):
    return wt.TP8_P1664 if rows <= 128 else wt.TP8


def _case(rows: int, kind: str, seed: int):
    """One step for LAYERS layers: per-layer block ids (global, distinct), per-block read offsets."""
    slots = {80: 24, 96: 24, 128: 32, 192: 48, 256: 64}[rows]
    if kind == "verify":
        lens = [W] * min(slots - 1, rows // W - 1)
        n_win = len(lens)
    elif kind == "mixed":
        lens = []
        for k in range(slots - 1):
            ln = [4, 2, 4, 1, 3, 4, 4, 2][k % 8]
            if sum(lens) + ln > rows - 3:
                break
            lens.append(ln)
        n_win = len(lens)
    elif kind == "prefill":
        lens = [[4, 2, 4, 1, 3, 4, 4, 2][k % 8] for k in range(slots - 8)]
        n_win = len(lens)
        lens.append(rows - sum(lens) - 3)  # one prefill chunk (PER_SEQ), 3 padding rows
    else:
        raise ValueError(kind)
    n_live = len(lens)
    rng = np.random.default_rng(seed)
    lens_a = np.zeros(slots, np.int64)
    lens_a[:n_live] = lens
    live = lens_a > 0
    qsl = np.zeros(slots + 1, np.int32)
    qsl[1:] = np.cumsum(lens_a)
    assert qsl[-1] < rows, (qsl[-1], rows)  # bucket padding rows exist
    num_blocks = 1 + LAYERS * slots * W
    ids = rng.permutation(np.arange(1, num_blocks)).reshape(LAYERS, slots, W)
    ckpt = np.where(live[None, :, None], ids, 0).astype(np.int32)
    history = np.full(slots, 1000)
    history[min(2, n_live - 1)] = 0  # a fresh request (no initial state)
    seq_lens = np.where(live, history + lens_a, 0).astype(np.int32)
    offsets = rng.integers(0, W, num_blocks).astype(np.int32)
    offsets[0] = 0
    keys = iter(jax.random.split(jax.random.key(seed), 8 * LAYERS))
    per_layer = []
    for layer in range(LAYERS):
        per_layer.append(dict(
            qkv=jax.random.normal(next(keys), (rows, DIM)).astype(jnp.bfloat16),
            b=jax.random.normal(next(keys), (rows, N_V)).astype(jnp.bfloat16),
            a=jax.random.normal(next(keys), (rows, N_V)).astype(jnp.bfloat16),
            conv_weight=(0.5 * jax.random.normal(next(keys), (4, 1, DIM))).astype(jnp.float32),
            a_log=jax.random.normal(next(keys), (N_V,)),
            dt_bias=jax.random.normal(next(keys), (N_V,)),
            state=jnp.asarray(ckpt[layer, :, 0]),
            ckpt=jnp.asarray(ckpt[layer]),
        ))
    return dict(
        slots=slots, n_live=n_live, live_rows=int(qsl[-1]), num_blocks=num_blocks, layers=per_layer,
        qsl=jnp.asarray(qsl), dist=jnp.asarray([n_win, n_live, n_live], jnp.int32),
        seq_lens=jnp.asarray(seq_lens), offsets=jnp.asarray(offsets),
        state_np=ckpt[:, :, 0].copy(),
    )


def _forward_context(attn_layers, state_np):
    md = {a.prefix: SimpleNamespace(mamba_state_indices=torch.from_numpy(state_np[i].copy()))
          for i, a in enumerate(attn_layers)}
    md["model.layers.3.self_attn"] = SimpleNamespace(mamba_state_indices=None)  # not a GDN layer
    return SimpleNamespace(attn_metadata=md)


def _run_step(attn_layers, fn, case, pools):
    """Every layer's pooled call of one step through the adapter's jax-op function, passing the stacks exactly when
    the layer was built lean (as forward does); returns (outputs, pools)."""
    outs, new_pools = [], []
    ctx = _forward_context(attn_layers, case["state_np"])
    for i, attn in enumerate(attn_layers):
        L = case["layers"][i]
        args = [L["qkv"], L["b"], L["a"], jnp.array(pools[i], copy=True), L["conv_weight"], None, L["a_log"],
                L["dt_bias"], L["state"], case["qsl"], case["dist"], case["seq_lens"], case["offsets"], L["ckpt"]]
        if attn._lean_glue3:
            with mock.patch.object(op, "get_forward_context", return_value=ctx):
                stacked, row = attn._stacked_slot_rows(torch.from_numpy(case["state_np"][i].copy()))
            args += [jnp.asarray(stacked.numpy()), jnp.asarray(row.numpy())]
        pool, out = fn(*args)
        jax.block_until_ready((pool, out))
        outs.append(np.asarray(out))
        new_pools.append(np.asarray(pool))
    return outs, new_pools


def _counters():
    return dict(wrapper.LEAN_GLUE_TRACES, gather=core.STACKED_GATHER_TRACES[0])


def _delta(after, before):
    return {k: after[k] - before[k] for k in after}


BW_CASES = [(r, k) for r in (80, 96, 128, 192, 256) for k in ("verify", "mixed")] + [(80, "prefill")]


@pytest.mark.parametrize("rows,kind", BW_CASES, ids=[f"r{r}-{k}" for r, k in BW_CASES])
def test_pooled_bw_and_engagement(served_gdn, rows, kind):
    monkeypatch = served_gdn
    geo = _geo(rows)
    case = _case(rows, kind, seed=rows + len(kind))
    pools = [wt.make_pool(geo, 100 * rows + i, case["num_blocks"]) for i in range(LAYERS)]

    off_layers, off_fn, off_name = _build(monkeypatch, False, geo.manager_tokens)
    c0 = _counters()
    out_off, pool_off = _run_step(off_layers, off_fn, case, pools)
    d_off = _delta(_counters(), c0)

    on_layers, on_fn, on_name = _build(monkeypatch, True, geo.manager_tokens)
    built_on = dict(name=on_name, lean_layers=sum(a._lean_glue3 for a in on_layers),
                    registered=eng.snapshot()["layers"])
    c1 = _counters()
    out_on, pool_on = _run_step(on_layers, on_fn, case, pools)
    d_on = _delta(_counters(), c1)

    arms = [("lean", out_on, pool_on, d_on)]
    if kind == "prefill":
        # VLLM_TPU_GDN_HOIST_READ_OFFSETS is subsumed: on top of the flag it changes nothing.
        monkeypatch.setattr(envs, "VLLM_TPU_GDN_HOIST_READ_OFFSETS", True)
        jax.clear_caches()
        c2 = _counters()
        out_h, pool_h = _run_step(on_layers, on_fn, case, pools)
        arms.append(("lean+hoist", out_h, pool_h, _delta(_counters(), c2)))
        monkeypatch.setattr(envs, "VLLM_TPU_GDN_HOIST_READ_OFFSETS", False)

    bw_ok, eng_ok = True, True
    for label, out_x, pool_x, d_x in arms:
        out_eq = [np.array_equal(a.view(np.uint8), b.view(np.uint8)) for a, b in zip(out_off, out_x)]
        pool_eq = [np.array_equal(a.view(np.uint8), b.view(np.uint8)) for a, b in zip(pool_off, pool_x)]
        bw_ok &= all(out_eq) and all(pool_eq)
        eng_x = d_x["gather"] == LAYERS and d_x["zero_fill"] > 0 and d_x["ba_packed"] > 0 and d_x["slot_rows"] > 0
        eng_ok &= eng_x
        ndiff = [int((a.view(np.uint8) != b.view(np.uint8)).sum()) for a, b in zip(pool_off, pool_x)]
        print(f"PBL_GDN_BW rows {rows} case {kind} arm {label} geo {geo.name} layers {LAYERS} slots "
              f"{case['slots']} live_rows {case['live_rows']} pad_rows {rows - case['live_rows']}: "
              f"out {'EQUAL' if all(out_eq) else 'DIFF ' + str(out_eq)} pool {'EQUAL' if all(pool_eq) else 'DIFF bytes ' + str(ndiff)}"
              f" | eng on {d_x} off {d_off}")
    eng_off_ok = all(v == 0 for v in d_off.values()) and re.fullmatch(r"pallas::gdn_attention_pooled_\d+", off_name)
    eng_ok = bool(eng_ok and re.fullmatch(r"pallas::gdn_attention_pooled_lg3_\d+", on_name)
                  and built_on["lean_layers"] == LAYERS and built_on["registered"] == LAYERS)
    print(f"PBL_GDN_ENG rows {rows} case {kind}: on gather {d_on['gather']} zero_fill {d_on['zero_fill']} ba_packed "
          f"{d_on['ba_packed']} slot_rows {d_on['slot_rows']} built {built_on} | off {d_off} op {off_name} -> "
          f"{'OK' if eng_ok and eng_off_ok else 'BAD'}")

    # Positive controls: the step wrote state and real outputs; padding rows are today's zeros.
    for i in range(LAYERS):
        assert not np.array_equal(pool_off[i].view(np.uint8), np.asarray(pools[i]).view(np.uint8)), "no state written"
        live = out_off[i][: case["live_rows"]].astype(np.float32)
        assert np.isfinite(live).all() and np.abs(live).sum() > 0
        assert np.all(out_off[i][case["live_rows"]:].astype(np.float32) == 0)
    assert bw_ok, "flag on differs from flag off"
    assert eng_ok and eng_off_ok, (d_on, d_off)


# ---------------------------------------------------------------------------
# 4. Wiring, flag-off identity, engagement line, per-step gather count.
# ---------------------------------------------------------------------------


def _forward_layer(prefix, lean, num_spec=W - 1):
    attn = _layer(prefix)
    attn.num_spec = num_spec
    attn._lean_glue3 = lean
    attn._pooled_pass_qkvz = False
    attn._pcp_streaming_configured = False
    attn.kv_cache = (torch.ones(4, 8),)
    attn.conv1d = SimpleNamespace(weight=torch.randn(1), bias=None)
    attn.conv_weight_kmajor = torch.randn(4, 1, 64)
    attn.A_log, attn.dt_bias = torch.randn(1), torch.randn(1)
    attn.num_v_heads, attn.head_v_dim = 4, 16
    attn.in_proj_qkv = mock.MagicMock(return_value=(torch.randn(2, 96), None))
    attn.in_proj_z = mock.MagicMock(return_value=(torch.randn(2, 64), None))
    attn.in_proj_ba = mock.MagicMock(return_value=(torch.randn(2, 8), None))
    attn.norm = mock.MagicMock(return_value=torch.randn(2, 4, 16))
    attn.out_proj = mock.MagicMock(return_value=(torch.ones(2, 64), None))
    attn.gdn_pooled_op = mock.MagicMock(return_value=torch.randn(2, 4, 16))
    return attn


@pytest.mark.parametrize("lean,offsets", [(False, True), (True, True), (True, False)],
                         ids=["off", "lean", "lean-no-offsets"])
def test_forward_passes_stacks_only_with_the_flag(monkeypatch, lean, offsets):
    monkeypatch.setattr(op, "_POOLED_GDN_PREFIXES", set())
    prefixes = [f"model.layers.{i}.linear_attn" for i in (0, 2, 1)]
    layers = [_forward_layer(p, lean, W - 1 if offsets else 0) for p in prefixes]
    if lean:
        op._POOLED_GDN_PREFIXES.update(prefixes)
    states = {p: torch.tensor([3 + 10 * i, 5 + 10 * i], dtype=torch.int32) for i, p in enumerate(prefixes)}
    md = {p: SimpleNamespace(
        mamba_state_indices=states[p], request_distribution=torch.zeros(3, dtype=torch.int32),
        mamba_slot_read_offsets=torch.zeros(64, dtype=torch.int32) if offsets else None,
        mamba_request_distribution=None,
        mamba_ckpt_indices=torch.zeros((2, W), dtype=torch.int32), query_start_loc=torch.zeros(3),
        seq_lens=torch.ones(2)) for p in prefixes}
    with mock.patch.object(op, "get_forward_context", return_value=SimpleNamespace(attn_metadata=md)):
        for attn in layers:
            attn.forward(torch.randn(2, 64))
    order = sorted(prefixes)
    for attn in layers:
        args = attn.gdn_pooled_op.call_args.args
        assert len(args) == (16 if lean else 14), len(args)
        if lean and offsets:
            stacked, row = args[14], args[15]
            assert torch.equal(stacked, torch.stack([states[p] for p in order]))
            assert row.tolist() == [order.index(attn.prefix)]
            assert torch.equal(stacked[row[0]], args[8])  # this layer's own state indices
        elif lean:
            assert args[14] is None and args[15] is None  # no read offsets: no stacks, the kernel pieces only
    print(f"PBL_GDN_WIRE lean {lean} offsets {offsets}: forward operands "
          f"{len(layers[0].gdn_pooled_op.call_args.args)}")


def test_flag_off_build_is_todays_op(served_gdn):
    layers, fn, name = _build(served_gdn, False, wt.TP8_P1664.manager_tokens)
    import inspect

    params = list(inspect.signature(fn).parameters)
    assert re.fullmatch(r"pallas::gdn_attention_pooled_\d+", name)
    assert len(params) == 14 and params[-1] == "ckpt_indices", params
    assert not any(a._lean_glue3 for a in layers) and not op._POOLED_GDN_PREFIXES
    assert eng.snapshot()["layers"] == 0
    on_layers, on_fn, on_name = _build(served_gdn, True, wt.TP8_P1664.manager_tokens)
    on_params = list(inspect.signature(on_fn).parameters)
    assert on_params[-2:] == ["stacked_state_indices", "layer_row"] and len(on_params) == 16
    assert sorted(op._POOLED_GDN_PREFIXES) == sorted(a.prefix for a in on_layers)
    print(f"PBL_GDN_WIRE flag off op {name} ({len(params)} operands); flag on op {on_name} ({len(on_params)})")


def test_lean_needs_the_per_region_pool(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_LEAN_GLUE3", True)
    assert op.lean_glue3_applies(False)
    assert not op.lean_glue3_applies(True)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_LEAN_GLUE3", False)
    assert not op.lean_glue3_applies(False)


def test_engagement_line(monkeypatch, caplog):
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT", False)
    eng.reset()
    assert not eng.count_step(96, 0)  # no lean layer built
    for _ in range(LAYERS):
        eng.record_built()
    assert eng.count_step(96, 0)
    assert not eng.count_step(64, 0, mk=True)  # the megakernel took it
    assert eng.count_step(96, 0)
    line = eng.format_line()
    print("PBL_GDN_ENGLINE", line)
    assert line == ("q38 GDN-LEAN3 engagement: lean pooled GDN layers 3; stock forwards 2 of 4 on this rank "
                    "(zero fill 2) | rows 64: 0/1 | rows 96: 2/3")
    eng.reset()


def test_runner_marks_megakernel_forwards_for_the_count():
    """The runner's megakernel mark (_q38_mk_hit) is set with VLLM_TPU_GDN_LEAN_GLUE3 too, not only with
    VLLM_TPU_Q38_STACK_ROUTE_COUNT, so the engagement line never counts a megakernel forward as a lean one."""
    import inspect

    from vllm_torchtpu.runner import tpu_runner

    src = inspect.getsource(tpu_runner)
    # q38 PORT-BL COMPOSE: the composed tree ORs every port key into the one mark, so check the assignment's
    # parsed value (it names both STACK_ROUTE_COUNT and LEAN_GLUE3), not one exact source line.
    import ast

    marks = [ast.unparse(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "q38_mk_mark" for t in n.targets)]
    assert len(marks) == 1, marks
    assert "envs.VLLM_TPU_Q38_STACK_ROUTE_COUNT" in marks[0] and "envs.VLLM_TPU_GDN_LEAN_GLUE3" in marks[0], marks
    assert src.count("if q38_mk_mark:\n            self._q38_mk_hit = False") == 1
    assert src.count("if hit and q38_mk_mark:\n                self._q38_mk_hit = True") == 1
    hook = src[src.index("if envs.VLLM_TPU_GDN_LEAN_GLUE3 and envs.VLLM_TPU_GDN_LEAN_GLUE3_COUNT:"):][:600]
    assert 'mk=bool(getattr(self, "_q38_mk_hit", False))' in hook
    print("PBL_GDN_WIRE runner mark set with the lean key; count hook reads it")


def test_stacked_gather_is_one_computation_per_step(monkeypatch):
    """Compiled XLA (CPU) of a three-layer step with the kernels stubbed: off, one read-offset gather per layer;
    on, the three layers' identical stack gathers merge into one."""
    seen = {}

    def stub(*args, read_offsets=None, slot_rows=None, **kw):
        seen["stub"] = True
        ro = args[13] if len(args) > 13 else read_offsets
        src = (jax.lax.dynamic_index_in_dim(slot_rows.read_offsets, slot_rows.row[0], keepdims=False)
               if slot_rows is not None else ro)
        return kw["state_source"], jnp.broadcast_to(src.sum().astype(jnp.bfloat16), (args[0].shape[0], N_V, DK))

    monkeypatch.setattr(core.gdn_v3_wrapper, "fused_conv1d_gdn", stub)
    monkeypatch.setattr(jax, "shard_map", _identity_shard_map)
    monkeypatch.setattr(core, "_build_v3_pool_state_plan", lambda *a, **k: None)
    slots, rows = 16, 64
    rng = np.random.default_rng(0)
    states = jnp.asarray(rng.permutation(np.arange(1, 1 + LAYERS * slots)).reshape(LAYERS, slots), jnp.int32)
    offsets = jnp.asarray(rng.integers(0, W, 1 + LAYERS * slots), jnp.int32)

    def step(lean, states, offsets, pool):
        outs = []
        for i in range(LAYERS):
            extra = (states, jnp.asarray([i], jnp.int32)) if lean else ()
            _, out = core.run_jax_gdn_attention_pooled(
                jnp.zeros((rows, DIM), jnp.bfloat16), jnp.zeros((rows, N_V)), jnp.zeros((rows, N_V)), pool,
                jnp.zeros((4, 1, DIM), jnp.float32), None, jnp.zeros(N_V), jnp.zeros(N_V), states[i],
                jnp.zeros(slots + 1, jnp.int32), jnp.zeros(3, jnp.int32), jnp.ones(slots, jnp.int32), offsets,
                jnp.zeros((slots, W), jnp.int32), *extra, n_kq=N_KQ, n_v=N_V, d_k=DK, d_v=DK, kernel_size=4,
                pool_block_tokens=128, mesh=SimpleNamespace(shape={"model": 1}), num_spec_tokens=W - 1,
                conv_weight_kmajor=True)
            outs.append(out)
        return outs

    pool = jnp.zeros((4, 8), jnp.float32)
    counts = {}
    for lean in (False, True):
        text = jax.jit(lambda s, o, p: step(lean, s, o, p)).lower(states, offsets, pool).compile().as_text()
        counts[lean] = len(re.findall(r"\bgather\(", text))
    print(f"PBL_GDN_CSE compiled XLA CPU read-offset gathers per {LAYERS}-layer step: off {counts[False]} "
          f"on {counts[True]}")
    assert seen.get("stub")
    assert counts[False] == LAYERS and counts[True] == 1, counts
