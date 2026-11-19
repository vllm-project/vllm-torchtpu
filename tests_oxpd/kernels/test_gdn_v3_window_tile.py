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
"""GDN v3 window_tile (VLLM_TPU_GDN_WINDOW_TILE, PR 2821 d8340b973) against
today's inner_kernel, on the pooled verify call, outputs and every pool byte.

Adapted from the PR's tests/kernels/test_gdn_v3_verify_tile.py
(test_window_tile_matches_inner_kernel and its schedules), at Qwen3.8 Flash
per-rank geometry: the NHD fp8 pool (token rows [1, 4, 256], 1 KiB),
FP32 recurrent and conv states, 128-token kernel blocks, MTP K = 3 (window
4), at TP8 (2 kq / 6 v heads, dim 1280: 10 conv rows per token, padded to
16 inside the tile) and TP2 (8 kq / 24 v heads, dim 5120). The PR's own
geometry (SEQ_ALONG_LANE pool, BF16 states, QK pair-blocked conv rows) is a
secondary case.

Backends. By default the kernels run in the Pallas TPU interpreter on CPU
(`Q38_GDN_WT_BACKEND=tpu` runs them on a TPU instead, with no shim and no
patch; that is the on-device bitwise gate). The CPU interpreter needs two
test-only aids:

* `_interpret_views`: the TPU interpreter's get / swap callbacks take only
  index transforms, but every pooled state access goes through a typed view
  (ref bitcast + reshape; "'BitcastTransform' object has no attribute
  'indices'"), so the pooled kernel, today's included, cannot run on CPU at
  all. The shim reads the indexed region and applies the remaining views in
  numpy, with jax's own `state.utils.bitcast` semantics (validated below
  against jax, and the whole pooled path against the dense-state kernel).
* `state_load_barrier`: an identity after the recurrent-state load of both
  paths; XLA CPU otherwise contracts the update of a BF16-loaded state into
  multiply-adds differently per fusion context (the PR's BF16 case only).
* `fixed_order_l2`: XLA CPU reduces `compute_gdn.l2_norm`'s 128-lane sum in
  an order that depends on the array's shape (see
  test_cpu_l2_norm_reduction_order_depends_on_shape): inner_kernel normalizes
  a [seq, num_kq_heads, chunk, 1, 128] array, window_tile one [1, 128] row,
  and at TP2's 8 kq heads some rows round differently (M, Mac, dev: 15 of 64
  rows). TP2 is therefore compared bit for bit with `l2_norm` replaced, in
  both paths alike, by a fixed-order pairwise lane sum (`_tree_l2_norm`), and
  unpatched under `_assert_confined`. On TPU a 128-lane reduce is one
  vector-register row and does not depend on the row count (I).

Mac dev numbers quoted here are labelled (M, Mac, dev); the CDK CPU job is
the record.
"""

import dataclasses
import inspect
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.gdn_pool_layout import (
    derive_pooled_gdn_state_layout,
    pooled_gdn_conv_state_bytes,
    pooled_gdn_ssm_state_bytes,
)
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, window_tile, wrapper

ON_TPU = os.environ.get("Q38_GDN_WT_BACKEND", "cpu") == "tpu"
D, KERNEL_SIZE, W = 128, 4, 4  # head dim, conv taps, verify window (K = 3)
QSA_BLOCK = 128  # Flash's kernel pages (qwen4_exp.qsa.QSA_BLOCK_SIZE)

# Unpatched TP2 confinement bound: a value head's state may differ by at most
# this many ulps of that head's largest |state| at that checkpoint.
# Measured worst (M, Mac, dev): 4.0, on 4 / 9 / 12 sequences.
CONFINE_HEAD_ULPS = 8
# And every output within this many bf16 ulps of the largest |output| of its
# (token, value head) row. Measured worst (M, Mac, dev): 0.25. Per element,
# near-zero outputs reach 5 bf16 ulps, which says nothing (DEC-FB-2 approved
# this in place of "outputs within 1 bf16 ulp" per element).
CONFINE_OUT_ROW_ULPS = 1


# --------------------------------------------------------------------------
# Test-only interpreter aids.
# --------------------------------------------------------------------------


def np_bitcast(x: np.ndarray, dtype) -> np.ndarray:
    """jax._src.state.utils.bitcast in numpy: a bitwidth change packs or
    unpacks along the second-minor dim (TPU semantics)."""
    dtype = np.dtype(dtype)
    xb, yb = x.dtype.itemsize * 8, dtype.itemsize * 8
    shape = list(x.shape)
    if xb != yb:
        assert len(shape) >= 2 and shape[-2] * xb % yb == 0, (x.shape, x.dtype, dtype)
    shape[-2] = shape[-2] * xb // yb
    if xb < yb:
        ratio = yb // xb
        x = x.reshape(*x.shape[:-2], x.shape[-2] // ratio, ratio, x.shape[-1])
        return np.ascontiguousarray(x.swapaxes(-1, -2)).view(dtype)[..., 0]
    if xb > yb:
        ratio = xb // yb
        y = np.ascontiguousarray(x).view(dtype).reshape(*x.shape, ratio)
        return np.ascontiguousarray(y.swapaxes(-1, -2).reshape(shape))
    return np.ascontiguousarray(x).view(dtype)


def _np_index(t):
    return tuple(
        i
        if isinstance(i, int | np.integer)
        else slice(int(i.start), int(i.start) + int(i.size) * int(i.stride), int(i.stride))
        for i in t.indices
    )


def _view_apply(arr, transforms):
    from jax._src.state import indexing, types

    for t in transforms:
        if isinstance(t, indexing.NDIndexer):
            arr = arr[_np_index(t)]
        elif isinstance(t, types.BitcastTransform):
            arr = np_bitcast(arr, t.dtype)
        elif isinstance(t, types.ReshapeTransform):
            arr = arr.reshape(t.shape)
        else:
            raise NotImplementedError(type(t))
    return arr


def _view_write(arr, transforms, val):
    from jax._src.state import indexing, types

    if not transforms:
        val = np.asarray(val)
        assert val.shape == arr.shape, (val.shape, arr.shape)
        return val.astype(arr.dtype)
    t, rest = transforms[0], transforms[1:]
    if isinstance(t, indexing.NDIndexer):
        idx = _np_index(t)
        out = np.array(arr, copy=True)
        out[idx] = _view_write(arr[idx], rest, val)
        return out
    if isinstance(t, types.BitcastTransform):
        return np_bitcast(_view_write(np_bitcast(arr, t.dtype), rest, val), arr.dtype)
    if isinstance(t, types.ReshapeTransform):
        return _view_write(arr.reshape(t.shape), rest, val).reshape(arr.shape)
    raise NotImplementedError(type(t))


def _install_interpret_views(monkeypatch):
    """Lets the TPU interpreter's get / swap callbacks take ref bitcast and
    reshape views: the indexed prefix is read (and written) as usual, the
    rest is applied in numpy. DMAs keep the stock path (index-only)."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
    from jax._src.state import indexing

    orig_get, orig_swap = ipc.get, ipc.swap
    calls = {"get": 0, "swap": 0}

    def split(transforms):
        for i, t in enumerate(transforms):
            if not isinstance(t, indexing.NDIndexer):
                return i
        return None

    def get(token, device_id, local_core_id, memory_space, buffer_id, transforms, *a, **k):
        at = split(transforms)
        if at is None:
            return orig_get(token, device_id, local_core_id, memory_space, buffer_id, transforms, *a, **k)
        calls["get"] += 1
        token, region = orig_get(
            token, device_id, local_core_id, memory_space, buffer_id, tuple(transforms[:at]), *a, **k
        )
        return token, _view_apply(np.asarray(region), jax.tree.map(int, tuple(transforms[at:])))

    def swap(token, device_id, local_core_id, memory_space, buffer_id, transforms, val, mask, **k):
        at = split(transforms)
        if at is None:
            return orig_swap(token, device_id, local_core_id, memory_space, buffer_id, transforms, val, mask, **k)
        calls["swap"] += 1
        assert mask is None
        prefix, rest = tuple(transforms[:at]), jax.tree.map(int, tuple(transforms[at:]))
        token, region = orig_get(
            token, device_id, local_core_id, memory_space, buffer_id, prefix,
            source_info=k.get("source_info"),
        )
        region = np.asarray(region)
        old = np.array(_view_apply(region, rest), copy=True)
        token, _ = orig_swap(
            token, device_id, local_core_id, memory_space, buffer_id, prefix,
            _view_write(region, rest, val), None, **k,
        )
        return token, old

    # DMA reads call get.__wrapped__ (index transforms only).
    get.__wrapped__ = orig_get.__wrapped__
    monkeypatch.setattr(ipc, "get", get)
    monkeypatch.setattr(ipc, "swap", swap)
    return calls


def _tree_l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """compute_gdn.l2_norm with a fixed-order (pairwise, halving) lane sum,
    so XLA CPU rounds it the same for every array shape. Test-only. The
    squares sit behind an optimization_barrier: fused into the first add,
    XLA CPU contracts them into multiply-adds, again depending on the shape
    (M, Mac, dev: 8 of 64 sums differ without it)."""
    sq = jax.lax.optimization_barrier(x * x)
    while sq.shape[-1] > 1:
        half = sq.shape[-1] // 2
        sq = sq[..., :half] + sq[..., half:]
    return x / jnp.sqrt(sq + eps)


@pytest.fixture
def kernels(monkeypatch):
    """Kernel environment: the Pallas TPU interpreter on CPU (with the view
    shim), or the TPU itself. Records window_tile.run traces."""
    traced = []
    run = window_tile.run

    def recorded(*a, **k):
        traced.append(k["cfg"])
        return run(*a, **k)

    monkeypatch.setattr(window_tile, "run", recorded)
    monkeypatch.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False)
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    jax.clear_caches()
    if ON_TPU:
        yield traced
        jax.clear_caches()
        return
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    _install_interpret_views(monkeypatch)
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh), pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
        yield traced
    jax.clear_caches()


@pytest.fixture
def fixed_order_l2(monkeypatch):
    """Test-only: compute_gdn.l2_norm -> _tree_l2_norm for BOTH paths
    (window_tile and inner_kernel's recurrent_gdn both look it up there)."""
    if ON_TPU:
        pytest.skip("the TPU runs the real l2_norm")
    monkeypatch.setattr(compute_gdn, "l2_norm", _tree_l2_norm)


@pytest.fixture
def state_load_barrier(monkeypatch):
    """Test-only: an optimization_barrier (an identity) after the recurrent
    state load of BOTH paths (vmem_ldst._load_recurrent_state for
    inner_kernel, vmem_ldst.load_state_head for window_tile). Fused with the
    BF16 -> FP32 convert, XLA CPU contracts the state update into
    multiply-adds differently per fusion context (M, Mac, dev: 21,907 of
    65,536 FP32 words of one batched vs per-head update differ; 0 with the
    barrier, and 0 with an FP32 initial state)."""
    if ON_TPU:
        pytest.skip("the TPU runs the real loads")
    from vllm_torchtpu.kernels.gdn.v3 import vmem_ldst

    load_all, load_head = vmem_ldst._load_recurrent_state, vmem_ldst.load_state_head
    monkeypatch.setattr(
        vmem_ldst, "_load_recurrent_state",
        lambda *a, **k: jax.lax.optimization_barrier(load_all(*a, **k)),
    )
    monkeypatch.setattr(
        vmem_ldst, "load_state_head",
        lambda *a, **k: jax.lax.optimization_barrier(load_head(*a, **k)),
    )


# --------------------------------------------------------------------------
# Geometry, pools, steps.
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Geo:
    name: str
    n_kq: int
    n_v: int
    page: tuple[int, ...]  # one kernel block, after the block index
    pool_dtype: object
    ssm_dtype: object
    conv_dtype: object
    hnd: bool = False
    qk_pair: bool = False
    manager_grained: bool = False

    @property
    def dim(self):
        return 2 * self.n_kq * D + self.n_v * D

    @property
    def token_bytes(self):
        if self.hnd:
            return int(np.prod(self.page[:-1])) * np.dtype(self.pool_dtype).itemsize
        return int(np.prod(self.page[1:])) * np.dtype(self.pool_dtype).itemsize

    def layout(self):
        return derive_pooled_gdn_state_layout(
            ssm_bytes=pooled_gdn_ssm_state_bytes(
                num_v_heads=self.n_v, head_k_dim=D, head_v_dim=D, dtype=jnp.dtype(self.ssm_dtype)
            ),
            conv_bytes=pooled_gdn_conv_state_bytes(
                kernel_size=KERNEL_SIZE, conv_dim=self.dim, dtype=jnp.dtype(self.conv_dtype)
            ),
            token_bytes=self.token_bytes,
        )

    @property
    def manager_tokens(self):
        if self.manager_grained:
            # The pool tensor's blocks are whole pool pages (served Flash:
            # [blocks, 1664, 1, 4, 256]); the state sits at their start.
            return self.page[0]
        # Else the scheduler page rounds the state up to whole QSA pages.
        return -(-self.layout().required_tokens // QSA_BLOCK) * QSA_BLOCK

    @property
    def split(self):
        block = self.page[-1] if self.hnd else self.page[0]
        return self.manager_tokens // block


FLASH_PAGE = (QSA_BLOCK, 1, 4, 256)  # NHD fp8: [tokens, 1, 4 head slots, 256]
TP8 = Geo("tp8", 2, 6, FLASH_PAGE, jnp.float8_e4m3fn, jnp.float32, jnp.float32)
TP2 = Geo("tp2", 8, 24, FLASH_PAGE, jnp.float8_e4m3fn, jnp.float32, jnp.float32)
# The same TP8 state in a pool whose kernel blocks are whole manager blocks
# (row-ranged SSM region of 384 tokens, conv region at row 384).
TP8_MGR = dataclasses.replace(TP8, name="tp8_mgr", page=(512, 1, 4, 256), manager_grained=True)
# The served pool tensors (C: reports/lane_dec_idx.md, lane_tp2page.md):
# fp8 [blocks, 1664, 1, 4, 256], one 1664-token pool page per block.
SERVED_PAGE = (1664, 1, 4, 256)
TP8_P1664 = dataclasses.replace(TP8, name="tp8_p1664", page=SERVED_PAGE, manager_grained=True)
TP2_P1664 = dataclasses.replace(TP2, name="tp2_p1664", page=SERVED_PAGE, manager_grained=True)
# The PR's own case, reduced: SEQ_ALONG_LANE fp8 pages (2 heads x 64 packed
# rows x 4 x 128 lanes), BF16 states (uint32 recurrent view), QK pair rows.
HND_BF16 = Geo(
    "hnd_bf16", 2, 4, (2, 64, 4, 128), jnp.float8_e4m3fn, jnp.bfloat16, jnp.bfloat16,
    hnd=True, qk_pair=True,
)
# The same layout with FP32 states: bitwise on CPU with no aid.
HND_F32 = dataclasses.replace(HND_BF16, name="hnd_f32", ssm_dtype=jnp.float32, conv_dtype=jnp.float32)
# Dense-comparison pool: Flash's TP8 token rows as FP32 words (no bitcast).
TP8_F32POOL = dataclasses.replace(TP8, name="tp8_f32pool", page=(QSA_BLOCK, 1, 1, 256), pool_dtype=jnp.float32)


def plan_for(geo: Geo, pool):
    """The pooled caller's plan (layers/core/gdn_attention.py
    _build_v3_pool_state_plan), built from the same production helpers."""
    lay = geo.layout()
    # pool_adapters reads the KV layout (whole-block DMA) when it plans.
    saved = os.environ.get("VLLM_KV_CACHE_LAYOUT")
    os.environ["VLLM_KV_CACHE_LAYOUT"] = "HND" if geo.hnd else "NHD"
    try:
        return _plan(geo, pool, lay)
    finally:
        if saved is None:
            del os.environ["VLLM_KV_CACHE_LAYOUT"]
        else:
            os.environ["VLLM_KV_CACHE_LAYOUT"] = saved


def _plan(geo, pool, lay):
    return pool_adapters.v3_state_source(
        pool,
        split=geo.split,
        ssm_ntok=lay.ssm_tokens,
        conv_tok0=lay.ssm_tokens,
        conv_ntok=lay.conv_tokens,
        conv_dim=geo.dim,
        n_v=geo.n_v,
        d_k=D,
        d_v=D,
        kernel_size=KERNEL_SIZE,
        qk_pair_layout=geo.qk_pair,
        recurrent_state_dtype=jnp.dtype(geo.ssm_dtype),
        conv_state_dtype=jnp.dtype(geo.conv_dtype),
    )


def make_pool(geo: Geo, seed: int, num_manager_blocks: int):
    """A pool whose state views read small finite values: filled in the view
    space of the state dtype, then cast back to the pool's bytes."""
    rng = np.random.default_rng(seed)
    proto = np.zeros((num_manager_blocks * geo.split, *geo.page), np.dtype(geo.pool_dtype))
    space = np.float32 if jnp.dtype(geo.ssm_dtype) == jnp.float32 else jnp.bfloat16
    vals = np_bitcast(proto, space)
    vals = (0.1 * rng.standard_normal(vals.shape)).astype(space)
    return jnp.asarray(np_bitcast(vals, proto.dtype))


def make_weights(geo: Geo, seed: int, bias: bool = False):
    keys = iter(jax.random.split(jax.random.key(seed), 4))
    return dict(
        conv_weight=(0.5 * jax.random.normal(next(keys), (geo.dim, 1, KERNEL_SIZE))).astype(jnp.bfloat16),
        conv_bias=(0.1 * jax.random.normal(next(keys), (geo.dim,))).astype(jnp.bfloat16) if bias else None,
        a_log=jax.random.normal(next(keys), (geo.n_v,)),
        dt_bias=jax.random.normal(next(keys), (geo.n_v,)),
    )


def make_step(geo, seed, num_spec, lens, dist, fresh=(), offsets=None, num_tokens=None, ordered=False):
    """One call's activations and metadata; `lens` are tokens per sequence.
    Checkpoint blocks of a sequence are deliberately not in window order
    unless `ordered` (the dense comparison addresses them as base + t)."""
    window = num_spec + 1
    seqs = len(lens)
    lens = np.asarray(lens)
    live = np.arange(seqs) < dist[-1]
    qsl = np.zeros(seqs + 1, np.int32)
    qsl[1:] = np.cumsum(np.where(live, lens, 0))
    num_tokens = int(qsl[-1]) if num_tokens is None else num_tokens
    history = np.where(np.isin(np.arange(seqs), fresh), 0, 1000)
    keys = iter(jax.random.split(jax.random.key(seed), 3))
    step = dict(
        qkv=jax.random.normal(next(keys), (num_tokens, geo.dim)).astype(jnp.bfloat16),
        b=jax.random.normal(next(keys), (num_tokens, geo.n_v)).astype(jnp.bfloat16),
        a=jax.random.normal(next(keys), (num_tokens, geo.n_v)).astype(jnp.bfloat16),
        query_start_loc=jnp.asarray(qsl),
        distribution=jnp.asarray(dist, jnp.int32),
        seq_lens=jnp.asarray(np.where(live, history + lens, 0), jnp.int32),
        state_indices=jnp.asarray(1 + np.arange(seqs) * window, jnp.int32),
        num_spec_tokens=num_spec,
    )
    if num_spec:
        order = np.arange(window) if ordered else np.arange(window)[::-1]
        step["ckpt_indices"] = jnp.asarray(1 + np.arange(seqs)[:, None] * window + order, jnp.int32)
        step["read_offsets"] = jnp.asarray(np.zeros(seqs) if offsets is None else offsets, jnp.int32)
    return step


def pooled_call(geo, pool, step, weights, wt, tile=None, **kw):
    if tile is not None:
        kw["decode_tile_size"] = tile
    return wrapper.fused_conv1d_gdn(
        conv_state=None,
        recurrent_state=None,
        state_source=jnp.array(pool, copy=True),
        state_plan=plan_for(geo, pool),
        n_kq=geo.n_kq,
        n_v=geo.n_v,
        d_k=D,
        d_v=D,
        kernel_size=KERNEL_SIZE,
        window_tile_mode=wt,
        **step,
        **weights,
        **kw,
    )


def _bytes(x):
    return np.asarray(x).view(np.uint8)


def assert_bitwise(got, want, what):
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(_bytes(g), _bytes(w), err_msg=what)


# --------------------------------------------------------------------------
# Schedules (the PR's, plus Flash batch sizes and tiles).
# --------------------------------------------------------------------------

SCHEDULES = {
    # Full windows, then mixed acceptance rolls each sequence back to a
    # different checkpoint, then full windows again.
    "verify_full": (
        6,
        [
            dict(lens=[W] * 6, dist=(6,) * 3),
            dict(lens=[W] * 6, dist=(6,) * 3, offsets=np.arange(6) % W),
            dict(lens=[W] * 6, dist=(6,) * 3, offsets=np.arange(6) * 3 % W),
        ],
    ),
    # 1-4 token windows, a fresh slot, and padded slots past the live ones.
    "verify_partial": (
        9,
        [
            dict(lens=[4, 1, 3, 2, 4, 4, 1, 2, 3], dist=(7, 7, 7), fresh=(4,), num_tokens=36),
            dict(lens=[2, 4, 1, 4, 3, 1, 4, 2, 4], dist=(8, 8, 8), offsets=[3, 0, 2, 1, 2, 0, 0, 1, 3],
                 num_tokens=36),
        ],
    ),
    # Verify windows followed by prefill chunks in the PER_SEQ segment.
    "verify_prefill": (
        6,
        [
            dict(lens=[4, 2, 1, 3, 9, 17], dist=(4, 6, 6), fresh=(5,), num_tokens=48),
            dict(lens=[4, 4, 2, 1, 4, 12], dist=(5, 6, 6), offsets=[1, 3, 0, 2, 0, 0], num_tokens=48),
        ],
    ),
    # An odd live count leaves a padded slot in the last tile.
    "verify_odd": (8, [dict(lens=[W] * 8, dist=(7,) * 3, offsets=np.arange(8) % W, num_tokens=32)]),
    # A lone live sequence with fewer tokens than the window pads the only tile.
    "verify_one": (4, [dict(lens=[3, 1, 1, 1], dist=(1, 1, 1), offsets=[2, 0, 0, 0], num_tokens=4)]),
    # Served-size tiles, under dynamic tiling as served
    # (TPU_ENABLE_GDN_DYNAMIC_TILING=1; tile "dyn"): 24 sequences give TP8 a
    # tile of 8 (its served tile at B28-64; the VMEM fit caps it there) and
    # TP2 one of 2. Mixed acceptance, a fresh slot, padded slots. Like the
    # served request arrays, the per-sequence arrays cover whole tiles (a
    # padded slot past them reads metadata out of bounds, which the
    # interpreter refuses).
    "verify_wide": (
        24,
        [
            dict(lens=[4, 2, 4, 1, 3, 4, 4, 2] * 3, dist=(22, 22, 22), fresh=(5,), num_tokens=96),
            dict(lens=[4] * 24, dist=(23, 23, 23), offsets=np.arange(24) % W, num_tokens=96),
        ],
    ),
}


def run_schedule(geo, schedule, kernels, tile, check, seed=0):
    """Runs a schedule with window_tile on and off, each from the same pool,
    and calls check((out_on, pool_on), (out_off, pool_off), label) per step.
    tile "dyn" sizes the tiles as served (dynamic tiling)."""
    seqs, steps = SCHEDULES[schedule]
    want_tile = tile
    if tile == "dyn":
        envs.TPU_ENABLE_GDN_DYNAMIC_TILING = True  # restored by the kernels fixture
        tile = None
        # The served tiles (tiling.get_tile_sizes at TPU7x's 64 MiB VMEM).
        want_tile = {"tp8": 8, "tp2": 2}[geo.name]
    weights = make_weights(geo, 1 + seed, bias=geo.name == "tp8")
    pool = make_pool(geo, 2 + seed, 1 + seqs * W + 1)
    pool_on, pool_off = pool, jnp.array(pool, copy=True)
    for i, spec in enumerate(steps):
        step = make_step(geo, 10 * seed + 10 + i, 3, **spec)
        before = len(kernels)
        prev = (pool_on, pool_off)
        pool_off, out_off = pooled_call(geo, pool_off, step, weights, False, tile)
        assert len(kernels) == before, "flag off traced window_tile"
        pool_on, out_on = pooled_call(geo, pool_on, step, weights, True, tile)
        if i == 0:
            assert len(kernels) > before, "flag on never reached window_tile"
            assert all(cfg.window_tile for cfg in kernels[before:])
            tiles = {cfg.seq_tile_size for cfg in kernels[before:]}
            assert tiles == {want_tile}, f"ran at tile {tiles}, not {want_tile}"
        assert not np.array_equal(_bytes(pool_off), _bytes(pool)), "the call wrote no state"
        check((out_on, pool_on), (out_off, pool_off), f"{geo.name} {schedule} tile {want_tile} step {i}",
              prev=prev, step=step)


# Strict (real l2_norm) on CPU: inner_kernel's batched q/k array must stay
# under 32 rows (seq tile x kq heads x window), where XLA CPU switches its
# 128-lane reduce (test_cpu_l2_norm_reduction_order_depends_on_shape).
CASES = [
    (TP8, "verify_full", 2),
    (TP8, "verify_partial", 3),
    (TP8, "verify_prefill", 2),
    (TP8, "verify_odd", 2),
    (TP8, "verify_one", 1),
    (TP8_MGR, "verify_partial", 3),
    (TP8_P1664, "verify_partial", 3),
    (HND_F32, "verify_partial", 2),
    (HND_F32, "verify_full", 2),
]
# The PR's BF16 states: bit for bit under state_load_barrier on CPU.
HND_BF16_CASES = [(HND_BF16, "verify_partial", 2), (HND_BF16, "verify_full", 2)]
# 32 or more q/k rows per tile: bit for bit with the fixed-order l2_norm,
# confined without it.
WIDE_CASES = [
    (TP8, "verify_partial", 4),
    (TP8, "verify_wide", "dyn"),
    # TP2's FP32 states fit one sequence per tile on the static path (the
    # wrapper's VMEM sizing); its served tile of 2 comes from dynamic tiling.
    (TP2, "verify_full", 1),
    (TP2, "verify_partial", 1),
    (TP2, "verify_prefill", 1),
    (TP2, "verify_odd", 1),
    (TP2, "verify_wide", "dyn"),
    (TP2_P1664, "verify_full", 1),
]


def _id(case):
    geo, schedule, tile = case
    return f"{geo.name}-{schedule}-t{tile}"


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_window_tile_bitwise(case, kernels):
    """TP8 (padded rows), its manager-grained pool and the PR's HND/BF16
    case: bit for bit with the real l2_norm."""
    geo, schedule, tile = case
    run_schedule(geo, schedule, kernels, tile, lambda got, want, what, **_: assert_bitwise(got, want, what))


@pytest.mark.parametrize("case", HND_BF16_CASES, ids=_id)
def test_window_tile_bitwise_bf16_states(case, kernels, state_load_barrier):
    """The PR's own geometry (SEQ_ALONG_LANE pool, BF16 states in a uint32
    view, QK pair-blocked conv rows) bit for bit under state_load_barrier
    (M, Mac, dev: without it 19 of 7.5 M pool bytes differ, single low bytes
    of BF16 state values; outputs and conv bytes bitwise)."""
    geo, schedule, tile = case
    run_schedule(geo, schedule, kernels, tile, lambda got, want, what, **_: assert_bitwise(got, want, what))


@pytest.mark.parametrize("case", CASES[:2] + WIDE_CASES, ids=_id)
def test_window_tile_bitwise_fixed_order_l2(case, kernels, fixed_order_l2, state_load_barrier):
    """TP2 and TP8 at tile 4 (and two TP8 strict cases again) bit for bit,
    with compute_gdn.l2_norm replaced in both paths by `_tree_l2_norm` and
    the state_load_barrier identity (see the module docstring)."""
    geo, schedule, tile = case
    run_schedule(geo, schedule, kernels, tile, lambda got, want, what, **_: assert_bitwise(got, want, what))


CONFINED_CASES = [
    (TP8, "verify_partial", 4),
    (TP2, "verify_full", 1),
    (TP2, "verify_partial", 1),
    (TP2, "verify_wide", "dyn"),
]


@pytest.mark.skipif(ON_TPU, reason="on TPU these are in test_window_tile_bitwise_on_tpu")
@pytest.mark.parametrize("case", CONFINED_CASES, ids=_id)
def test_window_tile_wide_unpatched_confined(case, kernels):
    """Wide tiles (TP8 at tile 4, TP2) with the real l2_norm on CPU: the only
    differences are those of l2_norm's shape-dependent lane-sum order
    (_assert_confined). Scale-aware bounds, approved by DEC-FB-2 and
    DECODE-LEAD in place of per-word / per-element ulp bounds, which near-zero
    values make meaningless (state words reach 0.5-2.5 M ulps there; outputs
    5 bf16 ulps); (4') and (2')'s inherited-group exemption approved by
    DEC-FB-2.

    (M, Mac, dev), these 4 cases, every step: state ratio max 5.0 (4.0 at
    step 0; bound 8), max |diff| 4.8e-7; outputs max 0.25 row bf16 ulp
    (bound 1); conv checkpoints bitwise; first divergences of equal-start
    groups whole-group in 100% (328 of 328), with 0-67 inherited groups per
    step. Earlier single-step smoke (4 / 12 / 9 sequences):
    ratio 4.0 in 3 of 3, 66 of 66 whole-group. A run beyond a bound fails;
    no bound is raised."""
    geo, schedule, tile = case
    run_schedule(
        geo, schedule, kernels, tile, lambda got, want, what, **kw: _assert_confined(geo, got, want, what, **kw)
    )


def _bf16_ordinal(x):
    u = np.asarray(x).view(np.uint16).astype(np.int64)
    return np.where(u & 0x8000, -(u & 0x7FFF), u)


def _state_regions(geo, pool):
    """Per manager block: (ssm [n_v, 128, 128], conv [3, dim]) in the state
    dtypes, decoded the way the kernel's typed views read them (FP32 states)."""
    lay = geo.layout()
    f32 = np_bitcast(np.asarray(pool), np.float32)
    rows = f32.reshape(-1, geo.manager_tokens * f32.shape[-1] * int(np.prod(f32.shape[2:-1])))
    ssm = rows[:, : geo.n_v * D * D].reshape(-1, geo.n_v, D, D)
    lanes = f32.shape[-1] * int(np.prod(f32.shape[2:-1]))
    conv0 = lay.ssm_tokens * lanes
    conv = rows[:, conv0 : conv0 + (KERNEL_SIZE - 1) * geo.dim].reshape(-1, KERNEL_SIZE - 1, geo.dim)
    return ssm, conv


def _bf16_row_ulp(x):
    """The bf16 ulp of each [.., v_head_dim] row's largest |value|."""
    top = np.abs(x).max(-1, keepdims=True).astype(np.float32)
    return np.where(top > 0, np.exp2(np.floor(np.log2(np.maximum(top, 1e-30))) - 7), np.inf)


def _assert_confined(geo, got, want, what, prev, step):
    """Unpatched wide tiles on CPU, (c) of the scheme:
    (1) conv checkpoint bytes bitwise;
    (2) a (sequence, key head) whose value heads all start from bitwise-equal
        states diverges, at its first differing checkpoint, in all of its
        value heads (its k feeds every one of them); groups whose initial
        state already differs (inherited from an earlier step) are exempt;
    (3) every value head's state within CONFINE_HEAD_ULPS ulps of that head's
        largest |state| at that checkpoint;
    (4) every output within CONFINE_OUT_ROW_ULPS bf16 ulps of the largest
        |output| of its (token, value head) row.
    Prints the measured numbers (Q38_CONFINE) so each run records its own."""
    assert jnp.dtype(geo.ssm_dtype) == jnp.float32 and not geo.hnd
    (out_on, pool_on), (out_off, pool_off) = got, want
    oa = np.asarray(out_on.astype(jnp.float32)).reshape(-1, geo.n_v, D)
    ob = np.asarray(out_off.astype(jnp.float32)).reshape(-1, geo.n_v, D)
    out_ratio = float((np.abs(oa - ob) / _bf16_row_ulp(ob)).max())
    out_elem = np.abs(_bf16_ordinal(out_on) - _bf16_ordinal(out_off))
    ssm_on, conv_on = _state_regions(geo, pool_on)
    ssm_off, conv_off = _state_regions(geo, pool_off)
    init_on, _ = _state_regions(geo, prev[0])
    init_off, _ = _state_regions(geo, prev[1])
    ckpts = np.asarray(step["ckpt_indices"])
    offsets = np.asarray(step["read_offsets"])
    group = geo.n_v // geo.n_kq
    seqs = (ssm_on.shape[0] - 2) // W
    worst_ratio, worst_abs, firsts, whole, inherited = 0.0, 0.0, 0, 0, 0
    problems = []
    for s in range(seqs):
        read = int(ckpts[s, offsets[s]])
        start_differs = np.abs(init_on[read] - init_off[read]).reshape(geo.n_v, -1).max(-1) > 0
        seen = set()
        for t in range(W):
            blk = int(ckpts[s, t])
            a, b = ssm_on[blk].reshape(geo.n_v, -1), ssm_off[blk].reshape(geo.n_v, -1)
            diff = np.abs(a - b).max(-1)
            ratio = diff / np.spacing(np.abs(b).max(-1).astype(np.float32))
            worst_ratio = max(worst_ratio, float(ratio.max()))
            worst_abs = max(worst_abs, float(diff.max()))
            bad = np.nonzero(ratio > CONFINE_HEAD_ULPS)[0]
            if bad.size:
                problems.append(f"seq {s} ckpt {t} heads {bad.tolist()} ratio {ratio.max():.1f}")
            differing = {int(h) for h in np.nonzero(diff)[0]}
            for g in range(geo.n_kq):
                heads = set(range(g * group, (g + 1) * group))
                if g in seen or not heads & differing:
                    continue
                seen.add(g)
                if start_differs[sorted(heads)].any():
                    inherited += 1
                    continue
                firsts += 1
                if heads <= differing:
                    whole += 1
                else:
                    problems.append(
                        f"seq {s} ckpt {t} key head {g} first diverges in value heads "
                        f"{sorted(heads & differing)} only"
                    )
    print(
        f"Q38_CONFINE {what}: max head-ulp ratio {worst_ratio:.2f} (bound {CONFINE_HEAD_ULPS}), "
        f"max |diff| {worst_abs:.3g}, outputs differing {int((out_elem > 0).sum())} "
        f"(max {int(out_elem.max())} element bf16 ulp; row-ulp ratio {out_ratio:.2f}, "
        f"bound {CONFINE_OUT_ROW_ULPS}), first divergences whole-group {whole}/{firsts} "
        f"(+{inherited} inherited)"
    )
    np.testing.assert_array_equal(conv_on.view(np.uint32), conv_off.view(np.uint32), err_msg=f"{what}: conv")
    assert out_ratio <= CONFINE_OUT_ROW_ULPS, f"{what}: outputs beyond {CONFINE_OUT_ROW_ULPS} row bf16 ulps"
    assert not problems, f"{what}: " + "; ".join(problems[:8])


TPU_CASES = [c for c in CASES if c[0] is TP8] + WIDE_CASES + HND_BF16_CASES


@pytest.mark.skipif(not ON_TPU, reason="the on-device bitwise gate needs a TPU")
@pytest.mark.parametrize("case", TPU_CASES, ids=_id)
def test_window_tile_bitwise_on_tpu(case, kernels):
    """On TPU, unpatched: TP8 at tiles 1-4 and its served tile 8 (dynamic
    tiling; 16 is out of reach, the VMEM fit caps TP8 at 8), TP2 (served
    tile 2) and the PR's BF16 case, bit for bit (the microbench's validity
    gate)."""
    geo, schedule, tile = case
    run_schedule(geo, schedule, kernels, tile, lambda got, want, what, **_: assert_bitwise(got, want, what))


def _ref_reshape_ranks(fn):
    """Runs fn() with jax's ref reshape (state.types.TransformedRef.reshape,
    which every Ref.reshape goes through) recording the rank of each ref it
    reshapes, and returns those ranks."""
    from jax._src.state import types as state_types

    orig = state_types.TransformedRef.reshape
    ranks = []

    def recorded(self, *shape):
        ranks.append(len(self.shape))
        return orig(self, *shape)

    state_types.TransformedRef.reshape = recorded
    try:
        fn()
    finally:
        state_types.TransformedRef.reshape = orig
    return ranks


@pytest.mark.skipif(ON_TPU, reason="trace-level CPU guard (the device runs the real lowering)")
@pytest.mark.parametrize("geo", [TP8, TP2], ids=lambda g: g.name)
def test_window_tile_reshapes_no_rank1_ref(geo, kernels):
    """Mosaic cannot reshape a rank-1 ref ("Reshape 1D ref is not supported":
    T44 GDNWT j-bae75c90, the TP8 conv bias); the x86 interpreter and the
    view shim can, so the CPU suite passed it. Traces the window_tile call
    (conv bias on, at TP8 and TP2) and requires every ref reshape in it to
    act on a ref of rank >= 2. The deviceless TPU AOT lowering
    (q38_gdn_tpu/q38_gdn_wt_aot.py) is the lowering-level check."""
    weights = make_weights(geo, 1, bias=True)
    pool = make_pool(geo, 2, 1 + 6 * W + 1)
    step = make_step(geo, 10, 3, **SCHEDULES["verify_full"][1][0])
    ranks = _ref_reshape_ranks(
        lambda: jax.make_jaxpr(lambda p: pooled_call(geo, p, step, weights, True, 2)[0])(pool)
    )
    assert kernels and all(cfg.window_tile for cfg in kernels), "window_tile was not traced"
    assert ranks, "no ref reshape recorded (the hook missed jax's ref reshape)"
    assert min(ranks) >= 2, f"rank-1 ref reshape in the traced kernel (ranks {sorted(set(ranks))})"


def test_decode_window_one_keeps_inner_kernel(kernels):
    """Plain decode (window 1) never takes window_tile, flag or not."""
    geo = TP8
    weights = make_weights(geo, 3)
    pool = make_pool(geo, 4, 6)
    step = make_step(geo, 5, 0, [1, 1, 1, 1], (4, 4, 4))
    p_off, o_off = pooled_call(geo, pool, step, weights, False, 2)
    p_on, o_on = pooled_call(geo, pool, step, weights, True, 2)
    assert kernels == []
    assert_bitwise((o_on, p_on), (o_off, p_off), "decode")


# --------------------------------------------------------------------------
# Why (b) patches l2_norm, and the shim's validity.
# --------------------------------------------------------------------------


def _l2_both(shape, fn):
    x = jax.random.normal(jax.random.key(3), shape, jnp.float32)
    lead = shape[:-2]

    @jax.jit
    def both(x):
        rows = x.reshape(-1, 1, shape[-1])
        per_row = jnp.stack([fn(rows[i]) for i in range(rows.shape[0])]).reshape(shape)
        return fn(x), per_row

    batched, per_row = (np.asarray(v) for v in both(x))
    return int((batched.view(np.uint32) != per_row.view(np.uint32)).reshape(int(np.prod(lead)), -1).any(-1).sum())


@pytest.mark.skipif(ON_TPU, reason="XLA CPU behaviour")
def test_cpu_l2_norm_reduction_order_depends_on_shape():
    """On XLA CPU, compute_gdn.l2_norm of inner_kernel's batched q/k layout
    [seq, num_kq_heads, chunk, 1, 128] and of window_tile's [1, 128] rows can
    round differently: the 128-lane jnp.sum reduces in an order that depends
    on the array's row count. (M, Mac, dev), jitted, batched against per-row:
    8, 16 and 24 rows bitwise ([1|2|3, 2, 4, 1, 128]: TP8 at tiles 1-3);
    32 rows, 9 of 32 rows differ ([4, 2, 4, 1, 128]: TP8 at tile 4; and
    [1, 8, 4, 1, 128]: TP2 at tile 1); [2, 8, 4, 1, 128], 15 of 64. The raw
    sums differ even with a barrier on the squares (32 of 64), so it is the
    reduction order itself. `_tree_l2_norm` (the test-only patch of
    test_window_tile_bitwise_fixed_order_l2) rounds alike at every shape; it
    needs its optimization_barrier, without which XLA CPU contracts the
    squares into the first add, again shape-dependently (8 of 64 sums
    differ)."""
    shapes = [(1, 2, W, 1, 128), (2, 2, W, 1, 128), (3, 2, W, 1, 128), (4, 2, W, 1, 128),
              (1, 8, W, 1, 128), (2, 8, W, 1, 128)]
    real = [_l2_both(shape, compute_gdn.l2_norm) for shape in shapes]
    tree = [_l2_both(shape, _tree_l2_norm) for shape in shapes]
    print(f"Q38_L2_ORDER rows differing, real / tree, at {[int(np.prod(s_[:-1])) for s_ in shapes]} rows: "
          f"{real} / {tree}")
    assert real[0] == real[1] == real[2] == 0, real
    assert real[3] > 0 and real[4] > 0 and real[5] > 0, real
    assert not any(tree), tree


def _update_both(initial, barrier):
    """One recurrent update of 4 value heads, batched (inner_kernel's form)
    and per head (window_tile's), jitted together on XLA CPU; returns the
    number of FP32 state words that differ."""
    nv, dims = 4, (((2,), (1,)), ((0,), (0,)))
    keys = iter(jax.random.split(jax.random.key(0), 8))
    state = (0.1 * jax.random.normal(next(keys), (nv, D, D))).astype(initial)
    k = jax.random.normal(next(keys), (nv, 1, D))
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v = jax.random.normal(next(keys), (nv, 1, D))
    g = jax.nn.sigmoid(jax.random.normal(next(keys), (nv, 1, 1)))
    b = jax.nn.sigmoid(jax.random.normal(next(keys), (nv, 1, 1)))

    def load(x):
        x = x.astype(jnp.float32)
        return jax.lax.optimization_barrier(x) if barrier else x

    def batched(state, k, v, g, b):
        decayed = load(state) * g
        v_upd = jax.lax.dot(k, decayed, dimension_numbers=dims, preferred_element_type=jnp.float32)
        return decayed + jnp.swapaxes(k, -1, -2) * (b * (v - v_upd))

    def per_head(state, k, v, g, b):
        rows = []
        for h in range(nv):
            decayed = load(state[h]) * g[h]
            v_upd = jax.lax.dot(
                k[h][None], decayed[None], dimension_numbers=dims, preferred_element_type=jnp.float32
            )[0]
            rows.append(decayed + jnp.swapaxes(k[h], -1, -2) * (b[h] * (v[h] - v_upd)))
        return jnp.stack(rows)

    both = jax.jit(lambda *a: (batched(*a), per_head(*a)))
    x, y = (np.asarray(t) for t in both(state, k, v, g, b))
    return int((x.view(np.uint32) != y.view(np.uint32)).sum())


@pytest.mark.skipif(ON_TPU, reason="XLA CPU behaviour")
def test_cpu_bf16_state_update_contraction_depends_on_fusion():
    """Why `state_load_barrier` exists. One update of a BF16-loaded state,
    batched against per head on XLA CPU (M, Mac, dev): 21,907 of 65,536 FP32
    words differ (multiply-adds contracted differently per fusion context);
    0 with an optimization_barrier after the BF16 -> FP32 convert; 0 with an
    FP32 initial state (no convert to fuse)."""
    counts = (
        _update_both(jnp.bfloat16, barrier=False),
        _update_both(jnp.bfloat16, barrier=True),
        _update_both(jnp.float32, barrier=False),
    )
    print(f"Q38_CONTRACTION bf16 / bf16+barrier / f32: {counts} of {4 * D * D}")
    assert counts[0] > 0 and counts[1] == 0 and counts[2] == 0, counts


@pytest.mark.skipif(ON_TPU, reason="CPU shim")
@pytest.mark.parametrize(
    "src,dst,shape",
    [
        (jnp.float8_e4m3fn, np.float32, (3, 2, 4, 256)),
        (jnp.float8_e4m3fn, jnp.bfloat16, (2, 8, 128)),
        (np.float32, jnp.float8_e4m3fn, (2, 3, 2, 256)),
        (jnp.bfloat16, np.uint32, (4, 6, 128)),
        (np.uint32, jnp.bfloat16, (5, 128)),
        (np.float32, np.uint32, (2, 3, 128)),
    ],
)
def test_shim_bitcast_matches_jax(src, dst, shape):
    """The shim's numpy bitcast equals jax's state discharge bitcast (the
    TPU semantics jax uses for ref bitcasts), both ways."""
    from jax._src.state import utils as state_utils

    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=int(np.prod(shape)) * np.dtype(src).itemsize, dtype=np.uint8)
    x = raw.view(np.dtype(src)).reshape(shape)
    want = np.asarray(state_utils.bitcast(jnp.asarray(x), dst))
    got = np_bitcast(x, dst)
    assert got.shape == want.shape and got.dtype == want.dtype
    np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))
    np.testing.assert_array_equal(np_bitcast(got, np.dtype(src)).view(np.uint8), x.view(np.uint8))


@pytest.mark.skipif(ON_TPU, reason="CPU shim")
@pytest.mark.parametrize("geo", [TP8_F32POOL, TP8], ids=lambda g: g.name)
def test_pooled_flag_off_matches_dense_kernel(geo, kernels):
    """Today's pooled path through the shim equals the dense-state kernel
    (no ref views at all) on the same states: outputs, and every checkpoint
    written back. On the FP32 pool the views are reshapes only; on the fp8
    pool they also bitcast."""
    # Whole tiles: the dense kernel's metadata read of a padded tile slot is
    # out of bounds, which the interpreter refuses (TPU kernels run with
    # bounds checks off).
    seqs = 4
    slots = 1 + seqs * W + 1
    weights = make_weights(geo, 7, bias=True)
    pool = make_pool(geo, 8, slots)
    ssm, conv = _state_regions(geo, pool)
    dense = (jnp.asarray(conv), jnp.asarray(ssm))
    spec = dict(lens=[4, 1, 3, 2], dist=(4, 4, 4), fresh=(3,), offsets=[0, 0, 2, 1])
    step = make_step(geo, 9, 3, ordered=True, **spec)
    new_pool, out_pooled = pooled_call(geo, pool, step, weights, False, 2)
    dense_step = {k: v for k, v in step.items() if k != "ckpt_indices"}
    (conv_d, ssm_d), out_dense = wrapper.fused_conv1d_gdn(
        conv_state=dense[0], recurrent_state=dense[1], n_kq=geo.n_kq, n_v=geo.n_v, d_k=D, d_v=D,
        kernel_size=KERNEL_SIZE, decode_tile_size=2, **dense_step, **weights,
    )
    assert_bitwise(out_pooled, out_dense, "outputs")
    ssm_p, conv_p = _state_regions(geo, new_pool)
    np.testing.assert_array_equal(ssm_p.view(np.uint32), np.asarray(ssm_d).view(np.uint32))
    np.testing.assert_array_equal(conv_p.view(np.uint32), np.asarray(conv_d).view(np.uint32))
    assert not np.array_equal(ssm_p, ssm), "no checkpoint written"


def test_test_plan_is_the_pooled_callers_plan():
    """plan_for builds the plan layers/core/gdn_attention.py builds."""
    gdn_attention = pytest.importorskip("vllm_torchtpu.layers.core.gdn_attention")
    for geo in (TP8, TP2, TP8_MGR, TP8_P1664, TP2_P1664):
        pool = jax.ShapeDtypeStruct((geo.split * 4, *geo.page), geo.pool_dtype)
        want = gdn_attention._build_v3_pool_state_plan(
            pool, conv_dim=geo.dim, n_v=geo.n_v, d_k=D, d_v=D, kernel_size=KERNEL_SIZE,
            pool_block_tokens=geo.manager_tokens, qk_pair_layout=False,
            recurrent_state_dtype=jnp.float32, conv_state_dtype=jnp.float32,
        )
        assert plan_for(geo, pool) == want, geo.name


# --------------------------------------------------------------------------
# Gate, flag and flags-off identity.
# --------------------------------------------------------------------------


def _cfg(geo, mode=config.GDNMode.BATCHED, window=W, **kw):
    pool = jax.ShapeDtypeStruct((geo.split * 4, *geo.page), geo.pool_dtype)
    plan = plan_for(geo, pool)
    cfg = config.GDNConfig(
        mode=mode, batch_size=16, kernel_size=KERNEL_SIZE, tile_size=4, window_size=window,
        dim_size=geo.dim, num_kq_heads=geo.n_kq, num_v_heads=geo.n_v, kq_head_dim=D, v_head_dim=D,
        dtypes=config.Dtypes(jnp.float32, jnp.bfloat16, jnp.float32, jnp.float32, jnp.float32),
        state_plan=plan, **kw,
    )
    block = geo.page if plan.whole_block_dma else (plan.recurrent.nrows, *geo.page[1:])
    return cfg, block


@pytest.fixture
def tpu7x_info(monkeypatch):
    if not ON_TPU:
        from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

        mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
        with use_abstract_mesh(mesh):
            yield
    else:
        yield


@pytest.mark.parametrize("geo", [TP8, TP2, TP8_MGR, TP8_P1664, TP2_P1664, HND_BF16], ids=lambda g: g.name)
def test_supported_on_flash_and_pr_geometries(geo, tpu7x_info, monkeypatch):
    if geo.hnd:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    else:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    cfg, block = _cfg(geo)
    assert window_tile.supported(cfg, block, jnp.dtype(geo.pool_dtype))
    assert cfg.state_plan.whole_block_dma is geo.hnd
    rows = geo.dim // 128
    assert cfg.window_tile_rows == -(-rows // 8) * 8
    if geo is TP8:
        assert (rows, cfg.window_tile_rows) == (10, 16)


def test_supported_refuses_what_it_does_not_implement(tpu7x_info, monkeypatch):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    cfg, block = _cfg(TP8)
    dtype = jnp.dtype(TP8.pool_dtype)
    assert not window_tile.supported(dataclasses.replace(cfg, window_size=1), block, dtype)
    assert not window_tile.supported(dataclasses.replace(cfg, mode=config.GDNMode.PER_SEQ), block, dtype)
    kda = dataclasses.replace(cfg, attention_mode=config.AttentionMode.KDA)
    assert not window_tile.supported(kda, block, dtype)
    assert not window_tile.supported(dataclasses.replace(cfg, state_plan=None), block, dtype)
    # TP8's 10 rows with a BF16 conv view (taps at unaligned packed rows).
    bf16_conv, block16 = _cfg(dataclasses.replace(TP8, conv_dtype=jnp.bfloat16))
    assert not window_tile.supported(bf16_conv, block16, dtype)
    # A permuted conv layout in 256-lane typed rows.
    plan = cfg.state_plan
    perm = dataclasses.replace(plan.conv, rows_perm=tuple(range(plan.conv.rows_used))[::-1])
    assert not window_tile.supported(
        dataclasses.replace(cfg, state_plan=dataclasses.replace(plan, conv=perm)), block, dtype
    )


def test_flag_registered_default_off_and_compile_hashed(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_GDN_WINDOW_TILE", raising=False)
    assert "VLLM_TPU_GDN_WINDOW_TILE" in envs.environment_variables
    assert envs.VLLM_TPU_GDN_WINDOW_TILE is False
    monkeypatch.setenv("VLLM_TPU_GDN_WINDOW_TILE", "1")
    assert envs.VLLM_TPU_GDN_WINDOW_TILE is True
    tpu_compiler = pytest.importorskip("vllm_torchtpu.compilation.tpu_compiler")
    assert "VLLM_TPU_GDN_WINDOW_TILE" not in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_flag_off_config_metadata_and_scratch_unchanged(tpu7x_info, monkeypatch):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    cfg, _ = _cfg(TP8)
    assert cfg.window_tile is False
    assert not any("window_tile" in key for key in cfg.get_metadata())
    assert "_wt" not in cfg.get_kernel_name()
    assert "window_scratch_refs" not in cfg.get_scratch_shape_dict()
    on = dataclasses.replace(cfg, window_tile=True)
    assert on.get_metadata()["window_tile"] == 1
    assert on.get_kernel_name().endswith("_wt")
    assert len(on.get_scratch_shape_dict()["window_scratch_refs"]) == 3


def test_flag_off_runs_no_new_code(kernels, monkeypatch):
    """With the env flag unset, the verify call never consults window_tile."""
    monkeypatch.delenv("VLLM_TPU_GDN_WINDOW_TILE", raising=False)

    def boom(*a, **k):
        raise AssertionError("window_tile consulted with the flag off")

    monkeypatch.setattr(window_tile, "supported", boom)
    monkeypatch.setattr(window_tile, "run", boom)
    geo = TP8
    step = make_step(geo, 11, 3, [4, 2, 4, 1], (4, 4, 4), offsets=[1, 0, 3, 0])
    pool, out = pooled_call(geo, make_pool(geo, 12, 18), step, make_weights(geo, 13), None, 2)
    jax.block_until_ready(out)


def test_env_flag_reaches_the_dispatch(kernels, monkeypatch):
    """window_tile_mode=None follows VLLM_TPU_GDN_WINDOW_TILE."""
    geo = TP8
    step = make_step(geo, 14, 3, [4, 3, 4, 1], (4, 4, 4))
    weights, pool = make_weights(geo, 15), make_pool(geo, 16, 18)
    monkeypatch.setenv("VLLM_TPU_GDN_WINDOW_TILE", "0")
    p_off, o_off = pooled_call(geo, pool, step, weights, None, 2)
    assert kernels == []
    monkeypatch.setenv("VLLM_TPU_GDN_WINDOW_TILE", "1")
    jax.clear_caches()
    p_on, o_on = pooled_call(geo, pool, step, weights, None, 2)
    assert kernels and all(cfg.window_tile for cfg in kernels)
    assert_bitwise((o_on, p_on), (o_off, p_off), "env flag")


def test_dispatch_is_gated_by_the_config_flag_only():
    """inner_kernel takes window_tile exactly when cfg.window_tile is set."""
    src = inspect.getsource(wrapper.inner_kernel)
    assert "if cfg.window_tile:" in src
    assert src.index("if cfg.window_tile:") < src.index("if cfg.dense_conv:")
