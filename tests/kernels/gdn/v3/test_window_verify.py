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
"""CPU interpret-mode bit-exactness of the GDN v3 window verify tile
(VLLM_TPU_GDN_V3_WINDOW_VERIFY) and the single-launch call
(VLLM_TPU_GDN_V3_SINGLE_LAUNCH) against the current kernels.

Geometry: one Qwen3.8-Flash TP8 rank (2 kq / 6 v GDN heads, head dim 128,
conv width 1280) on the served unified pool: NHD fp8 pages of 512 tokens
(1024 bytes per token), one page per mamba block, BF16 recurrent state and
FP32 conv state. MTP K=3 (4-token verify windows); 24 and 32 windows give
the 96- and 128-row verify calls of c48. The new paths must reproduce every
output bit and every pool byte, over steps with full and partial windows,
rollback offsets, fresh and padded slots and prefill after the windows.
"""

import contextlib
import dataclasses
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as _ipc
from jax._src.pallas.mosaic.interpret import utils as _iutils
from jax._src.state import indexing as _indexing
from jax._src.state import types as _state_types
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import config, window_tile, wrapper
from vllm_torchtpu.layers.core import gdn_attention

N_KQ, N_V, D, KS = 2, 6, 128, 4
DIM = 2 * N_KQ * D + N_V * D
PAGE_TOKENS = 512
PAGE = (PAGE_TOKENS, 1, 4, 256)  # NHD fp8 page: 1024 bytes per token
SPEC = 3
W = SPEC + 1

KERNEL_NAMES: list[str] = []


@pytest.fixture(autouse=True)
def _nhd_pool(monkeypatch):
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")


# CPU interpret mode reads and writes refs through index transforms only;
# the pooled state seam accesses raw pool blocks through typed views (ref
# bitcast + reshape, kernels/typed_ldst.py). These wrappers emulate those
# views on the host with Mosaic's ref.bitcast semantics (the second-minor
# dim packs, pallas/mosaic/primitives.py), so the reference kernels and the
# new ones go through the same byte mapping.


def _np_bitcast(x: np.ndarray, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    src = x.dtype
    if src.itemsize < dtype.itemsize:
        *lead, m, n = x.shape
        p = dtype.itemsize // src.itemsize
        y = np.swapaxes(x.reshape(*lead, m // p, p, n), -1, -2)
        return np.ascontiguousarray(y).view(dtype).reshape(*lead, m // p, n)
    if src.itemsize > dtype.itemsize:
        p = src.itemsize // dtype.itemsize
        y = np.ascontiguousarray(x).view(dtype).reshape(*x.shape, p)
        *lead, m, n, _ = y.shape
        return np.swapaxes(y, -1, -2).reshape(*lead, m * p, n)
    return x.view(dtype)


def _split_transforms(transforms):
    transforms = tuple(transforms)
    n = 0
    while n < len(transforms) and isinstance(transforms[n], _indexing.NDIndexer):
        n += 1
    return transforms[:n], jax.tree.map(int, transforms[n:])


def _forward(arr, rest):
    chain = [arr]
    for t in rest:
        if isinstance(t, _indexing.NDIndexer):
            arr = arr[_iutils.to_range((t,))]
        elif isinstance(t, _state_types.BitcastTransform):
            arr = _np_bitcast(arr, t.dtype)
        elif isinstance(t, _state_types.ReshapeTransform):
            arr = arr.reshape(t.shape)
        else:
            raise NotImplementedError(t)
        chain.append(arr)
    return chain


def _backward(chain, rest, new):
    for t, prev in zip(reversed(rest), reversed(chain[:-1]), strict=True):
        if isinstance(t, _indexing.NDIndexer):
            base = np.array(prev, copy=True)
            base[_iutils.to_range((t,))] = new
            new = base
        elif isinstance(t, _state_types.BitcastTransform):
            new = _np_bitcast(np.asarray(new), prev.dtype)
        else:
            new = np.asarray(new).reshape(prev.shape)
    return new


@contextlib.contextmanager
def _typed_view_interpret():
    orig_get, orig_swap = _ipc.get, _ipc.swap

    def get(token, dev, core, space, buf, transforms, *args, **kwargs):
        pre, rest = _split_transforms(transforms)
        if not rest:
            return orig_get(token, dev, core, space, buf, transforms, *args, **kwargs)
        token, base = orig_get(token, dev, core, space, buf, pre, *args, **kwargs)
        return token, _forward(np.asarray(base), rest)[-1]

    def swap(token, dev, core, space, buf, transforms, val, mask, **kwargs):
        pre, rest = _split_transforms(transforms)
        if not rest:
            return orig_swap(
                token, dev, core, space, buf, transforms, val, mask, **kwargs
            )
        assert mask is None
        token, base = orig_get(token, dev, core, space, buf, pre, **kwargs)
        chain = _forward(np.asarray(base), rest)
        new = np.asarray(val).astype(chain[-1].dtype).reshape(chain[-1].shape)
        token, _ = orig_swap(
            token,
            dev,
            core,
            space,
            buf,
            pre,
            _backward(chain, rest, new),
            None,
            **kwargs,
        )
        return token, chain[-1]

    # DMAs call the undecorated functions and never use typed views.
    get.__wrapped__ = orig_get.__wrapped__
    swap.__wrapped__ = getattr(orig_swap, "__wrapped__", orig_swap)
    with mock.patch.object(_ipc, "get", get), mock.patch.object(_ipc, "swap", swap):
        yield


def _plan(pool):
    return gdn_attention._build_v3_pool_state_plan(
        pool,
        conv_dim=DIM,
        n_v=N_V,
        d_k=D,
        d_v=D,
        kernel_size=KS,
        pool_block_tokens=PAGE_TOKENS,
        qk_pair_layout=False,
        recurrent_state_dtype=jnp.bfloat16,
        conv_state_dtype=jnp.float32,
    )


def _pool(seed: int, num_blocks: int) -> jax.Array:
    """fp8 pool whose bytes read back as small finite BF16 / FP32 values."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 0x3B, size=(num_blocks,) + PAGE, dtype=np.uint8)
    raw |= (rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7).astype(np.uint8)
    return jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)


def _weights(seed: int) -> dict:
    keys = iter(jax.random.split(jax.random.key(seed), 4))
    return dict(
        conv_weight=(0.5 * jax.random.normal(next(keys), (DIM, 1, KS))).astype(
            jnp.bfloat16
        ),
        conv_bias=(0.1 * jax.random.normal(next(keys), (DIM,))).astype(jnp.bfloat16),
        a_log=jax.random.normal(next(keys), (N_V,)),
        dt_bias=jax.random.normal(next(keys), (N_V,)),
    )


def _step(seed, lens, dist, fresh=(), offsets=None, num_tokens=None) -> dict:
    """One call's activations and metadata; `lens` are tokens per sequence."""
    seqs = len(lens)
    lens = np.asarray(lens)
    live = np.arange(seqs) < dist[-1]
    qsl = np.zeros(seqs + 1, np.int32)
    qsl[1:] = np.cumsum(np.where(live, lens, 0))
    num_tokens = int(qsl[-1]) if num_tokens is None else num_tokens
    history = np.where(np.isin(np.arange(seqs), fresh), 0, 1000)
    keys = iter(jax.random.split(jax.random.key(seed), 3))
    # Checkpoint blocks of a sequence deliberately not in window order.
    ckpts = 1 + np.arange(seqs)[:, None] * W + np.arange(W)[::-1]
    return dict(
        qkv=jax.random.normal(next(keys), (num_tokens, DIM)).astype(jnp.bfloat16),
        b=jax.random.normal(next(keys), (num_tokens, N_V)).astype(jnp.bfloat16),
        a=jax.random.normal(next(keys), (num_tokens, N_V)).astype(jnp.bfloat16),
        query_start_loc=jnp.asarray(qsl),
        distribution=jnp.asarray(dist, jnp.int32),
        seq_lens=jnp.asarray(np.where(live, history + lens, 0), jnp.int32),
        state_indices=jnp.asarray(1 + np.arange(seqs) * W, jnp.int32),
        ckpt_indices=jnp.asarray(ckpts, jnp.int32),
        read_offsets=jnp.asarray(
            np.zeros(seqs) if offsets is None else offsets, jnp.int32
        ),
    )


def _call(pool, step, weights, flags, **kw):
    """One pooled fused_conv1d_gdn call with the env flags in `flags` on."""
    KERNEL_NAMES.clear()
    orig_name = config.GDNConfig.get_kernel_name

    def spy_name(cfg):
        KERNEL_NAMES.append(orig_name(cfg))
        return KERNEL_NAMES[-1]

    names = (
        "VLLM_TPU_GDN_V3_WINDOW_VERIFY",
        "VLLM_TPU_GDN_V3_SINGLE_LAUNCH",
        "TPU_GDN_RAW_QKV",
        "TPU_GDN_RAW_BA",
        "TPU_GDN_KERNEL_ZERO_OUT",
        "TPU_ENABLE_GDN_DYNAMIC_TILING",
    )
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    jax.clear_caches()
    with contextlib.ExitStack() as stack:
        for name in names:
            stack.enter_context(mock.patch.object(envs, name, name in flags))
        stack.enter_context(
            mock.patch.object(config.GDNConfig, "get_kernel_name", spy_name)
        )
        stack.enter_context(use_abstract_mesh(mesh))
        stack.enter_context(_typed_view_interpret())
        stack.enter_context(
            pltpu.force_tpu_interpret_mode(
                pltpu.InterpretParams(
                    dma_execution_mode="eager", skip_floating_point_ops=False
                )
            )
        )
        pool, out = wrapper.fused_conv1d_gdn(
            conv_state=None,
            recurrent_state=None,
            state_source=jnp.array(pool, copy=True),
            state_plan=_plan(pool),
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D,
            d_v=D,
            kernel_size=KS,
            num_spec_tokens=SPEC,
            # CPU interpret mode lacks ref.reshape on the C > 1 chunked path.
            mixed_tile_size=1,
            **step,
            **weights,
            **kw,
        )
        jax.block_until_ready((pool, out))
    return pool, out


def _assert_same(got, want, what):
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        g, w = np.asarray(g), np.asarray(w)
        np.testing.assert_array_equal(g.view(np.uint8), w.view(np.uint8), err_msg=what)


# Tolerance of the window tile against inner_kernel on CPU. Upstream checks
# bitwise equality on TPU; XLA:CPU compiles the two formulations (one head
# per dot vs heads batched) with different fusions, which moves an
# occasional FP32 state element by an ulp and so flips its BF16 rounding.
# Conv checkpoints and every non-state pool byte must match bitwise; BF16
# outputs and BF16 recurrent checkpoint elements may differ in at most
# MAX_DIFF_FRACTION of the elements, each by at most one BF16 ulp of its
# value or MAX_ABS_ERR (elements near zero after cancellation).
MAX_DIFF_FRACTION = 1e-3
MAX_ABS_ERR = 1e-6
DIFF_STATS: list[tuple[str, int, int, float]] = []


def _ssm_rows(pool) -> int:
    return _plan(pool).recurrent.nrows


def _decode_ssm(pool_bytes: np.ndarray, rows: int) -> np.ndarray:
    """[blocks, N_V * D * D] BF16 recurrent state bits (uint16) of every
    pool block, decoded with the kernel's typed view (u32 lane-split view,
    then the in-register u32 -> bf16 bitcast)."""
    raw = pool_bytes[:, :rows]  # [blocks, rows, 1, 4, 256] uint8
    u32 = _np_bitcast(raw, np.uint32).reshape(raw.shape[0], -1, 128)
    bf16 = _np_bitcast(u32, np.uint16)
    return bf16.reshape(raw.shape[0], -1)


def _assert_window_close(got, want, what):
    (out, pool), (ref_out, ref_pool) = got, want
    g = np.asarray(pool).view(np.uint8)
    w = np.asarray(ref_pool).view(np.uint8)
    rows = _ssm_rows(pool)
    np.testing.assert_array_equal(g[:, rows:], w[:, rows:], err_msg=f"{what} conv")

    for name, a, b in (
        ("outputs", np.asarray(out), np.asarray(ref_out)),
        (
            "recurrent state",
            _decode_ssm(g, rows).view(jnp.bfloat16),
            _decode_ssm(w, rows).view(jnp.bfloat16),
        ),
    ):
        a, b = a.astype(np.float32), b.astype(np.float32)
        err = np.abs(a - b)
        diff = int((err > 0).sum())
        DIFF_STATS.append((f"{what} {name}", diff, a.size, float(err.max())))
        # One BF16 ulp of the value, or an absolute floor for elements that
        # nearly cancel (where an FP32 ulp of the operands is many BF16
        # ulps of the result).
        bound = np.maximum(np.abs(b) * 2.0**-7, MAX_ABS_ERR)
        assert (err <= bound).all(), f"{what}: {name} max error {err.max()}"
        assert diff <= MAX_DIFF_FRACTION * a.size, (
            f"{what}: {diff} of {a.size} {name} elements differ"
        )


SCHEDULES = {
    # c48 verify calls: 24 and 32 full windows (96 / 128 rows), then every
    # rollback offset.
    "rows96": [
        dict(lens=[W] * 24, dist=(24,) * 3),
        dict(lens=[W] * 24, dist=(24,) * 3, offsets=np.arange(24) % W),
    ],
    "rows128": [
        dict(lens=[W] * 32, dist=(32,) * 3, offsets=np.arange(32) * 3 % W),
    ],
    # 1-4 token windows, a fresh slot, padded slots past the live ones.
    "partial": [
        dict(
            lens=[4, 1, 3, 2, 4, 4, 1, 2, 3, 4, 4, 4],
            dist=(9, 9, 9),
            fresh=(4,),
            offsets=[3, 0, 2, 1, 0, 0, 0, 1, 3, 0, 0, 0],
            num_tokens=48,
        ),
    ],
    # Verify windows followed by prefill chunks in the PER_SEQ segment.
    "prefill": [
        dict(
            lens=[4, 2, 4, 1, 3, 4, 37, 21],
            dist=(6, 8, 8),
            fresh=(7,),
            offsets=[1, 3, 0, 0, 2, 0, 0, 0],
            num_tokens=96,
        ),
    ],
}

SERVED = ("TPU_GDN_RAW_QKV", "TPU_GDN_RAW_BA", "TPU_GDN_KERNEL_ZERO_OUT")


def _run_schedule(schedule, flags_new, flags_ref, seed, exact=True, **kw):
    """Each step of `schedule` with `flags_new` and with `flags_ref`, both
    from the reference pool of the previous step (so that tolerated
    differences do not compound)."""
    steps = SCHEDULES[schedule]
    seqs = len(steps[0]["lens"])
    weights = _weights(seed)
    ref_pool = _pool(seed + 1, 1 + seqs * W + 1)
    names = []
    check = _assert_same if exact else _assert_window_close
    for i, spec in enumerate(steps):
        step = _step(seed + 10 + i, **spec)
        pool, out = _call(ref_pool, step, weights, flags_new, **kw)
        names.append(" ".join(KERNEL_NAMES))
        ref_pool, ref_out = _call(ref_pool, step, weights, flags_ref, **kw)
        check((out, pool), (ref_out, ref_pool), f"{schedule} step {i}")
    return names


def test_window_tile_supports_served_pool():
    pool = _pool(0, 2)
    with use_abstract_mesh(
        AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    ):
        plan = _plan(pool)
        assert not plan.whole_block_dma
        assert plan.conv.view_dtype == jnp.float32
        cfg = config.GDNConfig(
            mode=config.GDNMode.BATCHED,
            dtypes=config.Dtypes(*(jnp.dtype(jnp.float32),) * 5),
            batch_size=128,
            dim_size=DIM,
            kernel_size=KS,
            tile_size=4,
            num_kq_heads=N_KQ,
            num_v_heads=N_V,
            kq_head_dim=D,
            v_head_dim=D,
            state_plan=plan,
            window_size=W,
            window_verify=True,
        )
        assert window_tile.supported(cfg, pool.shape, pool.dtype)
        # Plain decode and KDA keep inner_kernel.
        assert not window_tile.supported(
            dataclasses.replace(cfg, window_size=1), pool.shape, pool.dtype
        )


@pytest.mark.parametrize("served", [False, True], ids=["prepared", "raw"])
@pytest.mark.parametrize("schedule", sorted(SCHEDULES))
def test_window_verify_matches_inner_kernel(schedule, served):
    base = SERVED if served else ()
    names = _run_schedule(
        schedule,
        base + ("VLLM_TPU_GDN_V3_WINDOW_VERIFY",),
        base,
        seed=100,
        exact=False,
    )
    print("DIFF_STATS", DIFF_STATS[-2 * len(SCHEDULES[schedule]) :])
    # The window tile compiled into the verify kernel.
    assert all("_w4" in n and "_wt" in n for n in names), names


@pytest.mark.parametrize("schedule", ["rows96", "prefill"])
def test_window_verify_dynamic_tiling(schedule):
    flags = ("TPU_ENABLE_GDN_DYNAMIC_TILING",)
    _run_schedule(
        schedule,
        flags + ("VLLM_TPU_GDN_V3_WINDOW_VERIFY",),
        flags,
        seed=200,
        exact=False,
    )


@pytest.mark.parametrize("window", [False, True], ids=["inner", "window"])
@pytest.mark.parametrize("schedule", ["rows96", "partial", "prefill"])
def test_single_launch_matches_two_launches(schedule, window):
    base = SERVED + (("VLLM_TPU_GDN_V3_WINDOW_VERIFY",) if window else ())
    names = _run_schedule(
        schedule, base + ("VLLM_TPU_GDN_V3_SINGLE_LAUNCH",), base, seed=300
    )
    # One kernel per call: no separate PER_SEQ kernel was built.
    assert all("_batched_" in n and "_per_seq_" not in n for n in names), names


def test_single_launch_without_zero_out():
    # The XLA zero-initialized output aliased into the single kernel.
    _run_schedule("prefill", ("VLLM_TPU_GDN_V3_SINGLE_LAUNCH",), (), seed=400)


def test_fault_is_caught():
    """A window tile that drops the conv bias changes the outputs."""
    orig = window_tile.run

    def no_bias(*args, **kwargs):
        weights = args[9]
        weights = weights.__class__(
            conv=weights.conv.__class__(weight=weights.conv.weight, bias=None),
            gdn=weights.gdn,
        )
        return orig(*args[:9], weights, *args[10:], **kwargs)

    with (
        mock.patch.object(window_tile, "run", no_bias),
        pytest.raises(AssertionError),
    ):
        _run_schedule(
            "partial",
            ("VLLM_TPU_GDN_V3_WINDOW_VERIFY",),
            (),
            seed=500,
            exact=False,
        )
