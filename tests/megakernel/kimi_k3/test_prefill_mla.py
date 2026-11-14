# SPDX-License-Identifier: Apache-2.0
"""The prefill MLA layer's window all-gather on CPU: the half-ring gather of
the ranks' sequence-parallel rows, the attention kernel that gathers the
window while it runs, and the layer with
``KernelConfig.prefill_mla_flash_gather`` against the layer that takes the
gathered window. Needs 32 devices:
XLA_FLAGS=--xla_force_host_platform_device_count=32 (interpreting 32 devices
takes minutes)."""

import contextlib
from collections import namedtuple

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels

pytestmark = pytest.mark.cpu_test

FakeDevice = namedtuple("FakeDevice", "id coords core_on_chip")
HIDDEN = 7168
PAGE = 512


def _mesh():
    if len(jax.devices()) < 32:
        pytest.skip(
            "needs 32 devices: XLA_FLAGS=--xla_force_host_platform_device_count=32"
        )
    return Mesh(np.asarray(jax.devices()[:32]), ("k3_sp",))


def _ring_tables():
    """The ring tables of a 2x2x4 slice whose device ids follow the cores."""
    devices = [
        FakeDevice(((x * 2 + y) * 4 + z) * 2 + core, (x, y, z), core)
        for x in range(2)
        for y in range(2)
        for z in range(4)
        for core in range(2)
    ]
    return kernels.ring_tables(sorted(devices, key=lambda d: d.id), tuple(range(32)))


@contextlib.contextmanager
def _interpret():
    # A ring step waits for its incoming copy through its own outgoing copy's
    # descriptor: copies must run when they start, not when waited.
    with pltpu.force_tpu_interpret_mode(
        pltpu.InterpretParams(dma_execution_mode="eager", detect_races=True)
    ):
        yield


def _sharded(mesh, local, out_specs):
    return jax.jit(
        jax.shard_map(
            local, mesh=mesh, in_specs=P("k3_sp"), out_specs=out_specs, check_vma=False
        )
    )


def _window(local_rows):
    """The window of the ranks' rank-major local rows, in token order."""
    window = local_rows.shape[0] * local_rows.shape[1]
    expected = np.zeros((window, local_rows.shape[2]))
    expected[kernels.sp_order(window)] = local_rows.reshape(window, -1)
    return expected


@pytest.mark.parametrize("window", [512, 4096])
def test_window_all_gather_lays_the_ranks_rows_out_in_token_order(window):
    """512: one block of 16-row stripes; 4096: two blocks of 64-row stripes."""
    mesh = _mesh()
    width = 384
    rng = np.random.default_rng(window)
    local_rows = rng.integers(-64, 65, (32, window // 32, width))  # exact in BF16
    tables = jnp.asarray(_ring_tables(), jnp.int32)

    def local(rows):
        return kernels.window_all_gather(rows[0], tables, collective_id=39)[None]

    with _interpret():
        gathered = _sharded(mesh, local, P("k3_sp"))(
            jnp.asarray(local_rows, jnp.bfloat16)
        )
    expected = _window(local_rows)
    for rank in range(32):
        np.testing.assert_array_equal(np.asarray(gathered[rank], np.float64), expected)


@pytest.mark.parametrize(
    "window,q_tile",
    [
        (512, 1024),  # one query tile per head: 3 grid steps run all 17 ring steps
        (2048, 256),  # 24 grid steps, at most one ring step each
    ],
)
def test_the_attention_kernel_gathers_the_window_while_it_attends(
    monkeypatch, window, q_tile
):
    monkeypatch.setattr(
        kernels, "_config", lambda: KernelConfig(mla_flash_qtile=q_tile)
    )
    mesh = _mesh()
    width = 384
    rng = np.random.default_rng(window + 1)
    query = rng.standard_normal((32, 3, window, 256)) * 0.3
    key = rng.standard_normal((32, 3, window, 256))
    value = rng.standard_normal((32, 3, window, 128))
    local_rows = rng.integers(-64, 65, (32, window // 32, width))
    length = window - 37
    segments = jnp.asarray([0, length, length], jnp.int32)
    tables = jnp.asarray(_ring_tables(), jnp.int32)

    def local(query, key, value, rows):
        args = (query[0], key[0], value[0], jnp.asarray([length], jnp.int32), segments)
        alone = kernels.mla_flash_prefill_packed(*args, count=1)
        attended, gathered = kernels.mla_flash_prefill_packed(
            *args, count=1, gather=(rows[0], tables), collective_id=39
        )
        return alone[None], attended[None], gathered[None]

    operands = [jnp.asarray(x, jnp.bfloat16) for x in (query, key, value, local_rows)]
    with _interpret():
        alone, attended, gathered = _sharded(mesh, local, (P("k3_sp"),) * 3)(*operands)
    np.testing.assert_array_equal(np.asarray(attended), np.asarray(alone))
    expected = _window(local_rows)
    for rank in range(32):
        np.testing.assert_array_equal(np.asarray(gathered[rank], np.float64), expected)


def _mla_operands(rng, window, prefix_tokens, blocks_per_pool):
    """One MLA layer's operands (family slot 0) on every rank: the residual
    rows, snapshots, mixture weights and active count, the MLA families, a
    one-sequence descriptor after ``prefix_tokens`` cached tokens and the
    rank's FP8 pool pages (page k on rank k % 32), ids and arena bases."""
    rows = window // 32

    def bf16(shape, scale):
        return jnp.asarray(rng.standard_normal(shape) * scale, jnp.bfloat16)

    families = {
        "m_qa": bf16((32, 1, HIDDEN, 1536), HIDDEN**-0.5),
        "m_qb": bf16((32, 1, 1536, 768), 1536**-0.5),
        "m_ka": bf16((32, 1, HIDDEN, 640), HIDDEN**-0.5),
        "m_kb": bf16((32, 1, 512, 768), 512**-0.5),
        "m_gate": bf16((32, 1, HIDDEN, 384), HIDDEN**-0.5),
        "m_qnorm": jnp.asarray(
            1 + 0.1 * rng.standard_normal((32, 1, 1, 1536)), jnp.bfloat16
        ),
        "m_knorm": jnp.asarray(
            1 + 0.1 * rng.standard_normal((32, 1, 1, 512)), jnp.bfloat16
        ),
        "m_o": bf16((32, 1, 384, HIDDEN), (96 * 128) ** -0.5),
    }
    pools = np.zeros((32, blocks_per_pool, 128, 4, 640), ml_dtypes.float8_e4m3fn)
    pages = prefix_tokens // PAGE
    ids = np.zeros((32, 3), np.int32)
    local = np.zeros((32, 2), np.int32)
    for k in range(pages):
        rank, block = k % 32, 1 + k // 32
        pools[rank, block] = (rng.standard_normal((128, 4, 640)) * 0.5).astype(
            ml_dtypes.float8_e4m3fn
        )
        ids[rank, 1 + ids[rank, 0]] = block
        ids[rank, 0] += 1
        local[rank, 1] += PAGE
    return (
        bf16((32, rows, HIDDEN), 3.0),
        bf16((32, 8, rows, HIDDEN), 2.0),
        jnp.asarray(rng.standard_normal((32, 1, HIDDEN)) * 0.05, jnp.float32),
        jnp.asarray(1 + 0.2 * rng.standard_normal((32, 1, HIDDEN)), jnp.bfloat16),
        jnp.asarray([4], jnp.int32),
        families,
        jnp.asarray([0, window, window, 0, prefix_tokens], jnp.int32),
        jnp.asarray([window - 21], jnp.int32),
        jnp.asarray(pools.reshape(32 * blocks_per_pool, 128, 4, 640)),
        jnp.asarray(ids),
        jnp.asarray(local),
    )


def _expand_prefix_pages(pages, order, key_value_up):
    """``expand_prefix_pages`` in XLA (its kernel bitcasts a VMEM ref, which
    the interpreter cannot run), up to the token order within a page."""
    heads, lanes = key_value_up.shape[1] // 256, pages.shape[-1]
    live = jnp.arange(order.shape[0] - 1) < order[0]
    rows = jnp.where(live[:, None, None, None], pages[order[1:]].astype(jnp.float32), 0)
    rows = rows.reshape(-1, lanes).astype(jnp.bfloat16)
    expanded = jnp.dot(
        rows[:, :512], key_value_up, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    rope = jnp.pad(rows[:, 512:576], ((0, 0), (0, 64)))
    keys = jnp.stack(
        [
            jnp.concatenate((expanded[:, 256 * h : 256 * h + 128], rope), axis=1)
            for h in range(heads)
        ]
    )
    values = jnp.stack(
        [expanded[:, 256 * h + 128 : 256 * (h + 1)].T for h in range(heads)]
    )
    dtype = jnp.float8_e4m3fn if kernels._config().prefill_prefix_fp8 else jnp.bfloat16
    return keys.astype(dtype)[None], values.astype(dtype)[None]


def test_mla_layer_gathering_under_the_attention_matches_the_gathered_window(
    monkeypatch,
):
    """The whole layer as the prefill groups run it, flag off and on: the new
    residual rows and the compressed KV bit for bit."""
    mesh = _mesh()
    tables = _ring_tables()
    monkeypatch.setattr(kernels, "ring_tables", lambda devices, token_ranks: tables)
    monkeypatch.setattr(kernels, "expand_prefix_pages", _expand_prefix_pages)
    window, capacity = 512, 4096
    operands = _mla_operands(np.random.default_rng(5), window, 2 * PAGE, 2)

    def layer(config):
        monkeypatch.setattr(kernels, "_config", lambda: config)
        # The op factories build their Pallas calls: interpret them too.
        with _interpret():
            mla_input = kernels.make_mla_input_sp(mesh, seq_len=window)
            mla = kernels.make_indexed_mla_layer_sp_ring(
                mesh,
                seq_len=window,
                segments=1,
                prefix_capacity=capacity,
                flash_tile=256,
            )

        def run(
            prefix, blocks, res, norm, active, families, desc, length, pool, ids, local
        ):
            counts, order = kernels.cp_prefix_meta(
                mesh, pool, ids, local, segments=1, capacity=capacity
            )
            arena = kernels.CpPrefix((pool,), ids, local, counts, order)
            hidden = mla_input(prefix, blocks, res, norm, active)
            return mla(
                hidden,
                prefix,
                families,
                desc,
                length,
                jnp.asarray([0], jnp.int32),
                arena,
            )

        with _interpret():
            return jax.jit(run)(*operands)

    gathered = layer(KernelConfig())
    fused = layer(KernelConfig(prefill_mla_flash_gather=True))
    for before, after in zip(gathered, fused, strict=True):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
