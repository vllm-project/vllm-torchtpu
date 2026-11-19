# SPDX-License-Identifier: Apache-2.0
"""Compressed keys once per decode host (qwen4_exp pd_shared_compressed).

The reference is today's broadcast path: every rank stages every producer
compressed page and the repack copies ``compressed_rows_np`` rows into its
pool. The shared path lands page ``b`` of each tag's pair space on rank
``b % G`` only (``producer_runs``, applied as Raiden would: dense per-unit
coverage, one per-tag clip), then all-gathers and scatters
(``gather_passes``). Every rank's pool must come out byte-identical to the
reference, whole pool compared, prefix pages included.

The 8-device test runs the landing as one SPMD program (pack, all-gather,
scatter) on a CPU mesh: run with
``XLA_FLAGS=--xla_force_host_platform_device_count=8``.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).parents[2] / "src/vllm_torchtpu/models/vllm/qwen4_exp"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sc = _load("pd_shared_compressed")
st = _load("pd_staging")

G, RATIO, SRC_BLOCK, DST_BLOCK = 8, 4, 3200, 512
GEO = sc.SharedGeometry(group=G, ratio=RATIO, src_block=SRC_BLOCK)


# --- a load: producer pages, decode tables, today's reference -------------


def make_load(num_tokens, layers, width, skip=0, packed_slots=1, seed=0):
    """Producer compressed pages, decode tables and the reference pools.

    ``packed_slots`` > 1 packs consecutive layers into one owner group per
    ``compressed_destinations`` (slot s at row offset s * 64 of the page).
    Decode tables list every page from token 0 (the prefix pages are shared
    cache pages the load must not write); pools start random.
    """
    rng = np.random.default_rng(seed)
    n_src = -(-num_tokens // SRC_BLOCK)
    produced = [
        rng.integers(0, 256, (n_src, GEO.src_rows, width), dtype=np.uint8)
        for _ in range(layers)
    ]
    n_dst = -(-num_tokens // DST_BLOCK)
    groups = -(-layers // packed_slots)
    pool_pages = 1 + groups * n_dst + 3
    perm = rng.permutation(np.arange(1, pool_pages))  # page 0: the null page
    tables = [perm[g * n_dst : (g + 1) * n_dst] for g in range(groups)]
    member_rows = DST_BLOCK // (2 * RATIO)
    dests = [(tables[l // packed_slots], (l % packed_slots) * member_rows) for l in range(layers)]
    pool0 = rng.integers(0, 256, (pool_pages * DST_BLOCK, width), dtype=np.uint8)
    return produced, dests, pool0


def reference_pool(num_tokens, produced, dests, pool0, skip=0, seed=1):
    """Today's path: broadcast staging in the producer geometry (source pages
    from the first one not wholly inside the prefix) and the repack's
    ``compressed_rows_np`` copy rows."""
    rng = np.random.default_rng(seed)
    first = skip // SRC_BLOCK
    pool = pool0.copy()
    for pages, (table, offset) in zip(produced, dests):
        staged = pages[first:]
        ids = rng.permutation(np.arange(1, 1 + 2 * len(staged)))[: len(staged)]
        carrier = np.zeros((1 + 2 * len(staged), SRC_BLOCK, pages.shape[-1]), np.uint8)
        carrier[ids, : GEO.src_rows] = staged
        src, dst = st.compressed_rows_np(
            num_tokens, RATIO, ids, SRC_BLOCK, table, DST_BLOCK,
            start=skip, src_first=first, dst_offset=offset,
        )
        pool[dst] = carrier.reshape(-1, pages.shape[-1])[src]
    return pool


# --- the shared path's landing, applied as Raiden's planner would -------


def check_raiden_rules(runs, num_tokens):
    """Dense per-unit coverage of the same length, runs inside one producer
    page, declared bytes == coverage (what SplitGlobalSpaceSpans /
    ValidateSpanCoverage enforce)."""
    n_src = -(-num_tokens // SRC_BLOCK)
    length = GEO.unit_pages(num_tokens) * GEO.page_rows
    assert (runs[:, 4] > 0).all()
    assert (runs[:, 3] + runs[:, 4] <= GEO.src_rows).all()
    assert (runs[:, 2] < n_src).all() and (runs[:, 2] >= 0).all()
    for unit in range(G):
        mine = runs[runs[:, 0] == unit]
        mine = mine[np.argsort(mine[:, 1])]
        ends = mine[:, 1] + mine[:, 4]
        assert mine[0, 1] == 0 and (mine[1:, 1] == ends[:-1]).all(), "coverage not dense"
        assert ends[-1] == length, f"unit {unit} covers {ends[-1]} rows, not {length}"
    assert runs[:, 4].sum() == G * length


def land(num_tokens, produced, skip, rng):
    """Every rank's compact staging after the transfer: the producer runs,
    clipped by the per-tag dst_skip (whole pages, the same on every unit),
    at staging pages allocated identically on every rank."""
    clip_rows = GEO.clip_pages(skip) * GEO.page_rows
    count = GEO.staged_pages(num_tokens, skip)
    width = produced[0].shape[-1]
    carrier_pages = 1 + len(produced) * count + 5
    free = rng.permutation(np.arange(1, carrier_pages))
    staging = np.zeros((G, carrier_pages, GEO.page_rows, width), np.uint8)
    tag_pages = []
    for t, pages in enumerate(produced):
        ids = free[t * count : (t + 1) * count]
        tag_pages.append(tuple(int(i) for i in ids))
        runs = sc.producer_runs(num_tokens, GEO)
        check_raiden_rules(runs, num_tokens)
        for unit, dst, page, row, rows in runs:
            j = dst - clip_rows
            if j < 0:
                continue  # clipped by dst_skip_bytes (whole pages)
            k, r0 = divmod(j, GEO.page_rows)
            assert r0 + rows <= GEO.page_rows  # a run stays in one compact page
            staging[unit, ids[k], r0 : r0 + rows] = pages[page, row : row + rows]
    return staging, tag_pages


def shared_pool_numpy(num_tokens, produced, dests, pool0, skip=0, seed=2):
    """The landing gather emulated: every rank packs its own staging pages,
    the all-gather concatenates the send buffers rank-major, and the scatter
    writes the pool. The gathered buffer and both index lists are the same
    on every rank, so one pool stands for all of them here; the SPMD test
    below runs the ranks separately."""
    rng = np.random.default_rng(seed)
    staging, tag_pages = land(num_tokens, produced, skip, rng)
    tags = [sc.SharedTag(p, tuple(table), off) for p, (table, off) in zip(tag_pages, dests)]
    passes = sc.gather_passes(num_tokens, tags, DST_BLOCK, GEO, skip_tokens=skip)
    pool = pool0.copy()
    width = pool0.shape[-1]
    for p in passes:
        send = staging[:, p.pack]  # [G, Q, 256, W]: each rank packs its own
        gathered = send.reshape(-1, width)  # all-gather: rank-major
        pool[p.scatter_dst] = gathered[p.scatter_src]
    return pool, passes


def cases():
    out = []
    for num_tokens in (16_385, 20_000, 40_000, 2 * 16_384 + 1):
        for skip in (0, 2048, 16_384):
            if skip < num_tokens - 2048:
                out.append((num_tokens, skip))
    return out


@pytest.mark.parametrize("num_tokens,skip", cases())
@pytest.mark.parametrize("packed", [1, 2])
def test_shared_pool_is_byte_identical_full_rows(num_tokens, skip, packed):
    # Real 1 KiB rows ([4 sublanes, 256 lanes] of bytes), 3 layers.
    produced, dests, pool0 = make_load(num_tokens, 3, 1024, skip, packed)
    ref = reference_pool(num_tokens, produced, dests, pool0, skip)
    pool, _ = shared_pool_numpy(num_tokens, produced, dests, pool0, skip)
    assert np.array_equal(pool, ref)
    assert not np.array_equal(ref, pool0)  # the load wrote something


@pytest.mark.parametrize("skip", [0, 2048, 34_816, 65_536])
@pytest.mark.parametrize("packed", [1, 8])
def test_shared_pool_served_geometry(skip, packed):
    # An 88K-token request: 12 QSA layers; narrow rows keep it small (the
    # addressing does not depend on the row width).
    num_tokens = 88_000
    produced, dests, pool0 = make_load(num_tokens, 12, 16, skip, packed)
    ref = reference_pool(num_tokens, produced, dests, pool0, skip)
    pool, passes = shared_pool_numpy(num_tokens, produced, dests, pool0, skip)
    assert np.array_equal(pool, ref)
    if skip == 0:
        # 11,000 pairs -> 43 pages -> 6 per rank (5 or 6 own + padding);
        # 72 pages per rank in 5 passes of <= 16.
        assert GEO.unit_pages(num_tokens) == 6
        assert [p.bucket for p in passes] == [16, 16, 16, 16, 8]


def test_bytes_per_rank_at_88k():
    num_tokens, row = 88_000, 1024
    per_tag = GEO.unit_pages(num_tokens) * GEO.page_rows * row
    shared = 12 * per_tag
    # Broadcast (S12 compact rows): every rank gets every producer page's
    # 400 live rows.
    broadcast = 12 * -(-num_tokens // SRC_BLOCK) * GEO.src_rows * row
    assert shared == 18_874_368  # 18.9 MB per rank, 151 MB per host
    assert broadcast == 137_625_600  # 137.6 MB per rank, 1.10 GB per host
    assert broadcast / shared > 7.29


def test_short_loads_stay_broadcast():
    # 7 pages (14,336 tokens) or fewer: some rank would land no page.
    assert not GEO.shared(14_336)
    assert GEO.shared(14_337)
    with pytest.raises(ValueError, match="stays broadcast"):
        sc.producer_runs(14_336, GEO)


def test_owner_rule_is_head_unique_round_robin():
    num_tokens = 40_000  # 5,000 pairs -> 20 pages
    pages = GEO.pages(num_tokens)
    assert pages == 20
    for unit in range(G):
        own = [GEO.unit_page_source(num_tokens, unit, k) for k in range(GEO.unit_pages(num_tokens))]
        real = [b for k, b in enumerate(own) if GEO.is_real(num_tokens, unit, k)]
        assert real == list(range(unit, pages, G))  # head_unique.owned_page_indices
        assert all(GEO.owner(b) == unit for b in own)


@pytest.mark.parametrize("num_tokens", [14_337, 16_384, 25_600, 25_601, 88_000, 131_072, 190_001])
def test_producer_runs_meet_raiden_rules(num_tokens):
    check_raiden_rules(sc.producer_runs(num_tokens, GEO), num_tokens)


@pytest.mark.parametrize("skip", [2048, 14_336, 16_384, 30_720, 32_768])
def test_prefix_clip_is_equal_whole_pages_on_every_rank(skip):
    # The clip removes only pairs below the prefix, from every rank alike.
    num_tokens = 88_000
    clip = GEO.clip_pages(skip)
    assert clip == skip // 16_384
    for unit in range(G):
        for k in range(clip):
            b = GEO.unit_page_source(num_tokens, unit, k)
            assert (b + 1) * GEO.page_rows * GEO.pair_tokens <= skip


def test_passes_are_rank_agnostic_and_tile_aligned():
    num_tokens = 88_000
    produced, dests, _ = make_load(num_tokens, 12, 1, 0, 1)
    tags = [
        sc.SharedTag(tuple(range(1 + 6 * t, 7 + 6 * t)), tuple(table), off)
        for t, (table, off) in enumerate(dests)
    ]
    for p in sc.gather_passes(num_tokens, tags, DST_BLOCK, GEO):
        n = G * p.bucket * GEO.page_rows
        assert len(p.scatter_src) == len(p.scatter_dst) == n
        tiles = p.scatter_dst.reshape(-1, sc.TILE)
        runs = (tiles[:, 1:] - tiles[:, :-1] == 1) | (tiles[:, 1:] == tiles[:, :-1])
        assert runs.all(), "a tile is not one contiguous destination run"


# --- the SPMD program on an 8-device CPU mesh ---------------------------------


def _mesh():
    jax = pytest.importorskip("jax")
    if len(jax.devices()) < G:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")
    from jax.sharding import Mesh

    return Mesh(np.array(jax.devices()[:G]), ("tp",))


def shared_landing_program(mesh, bucket, width):
    """Pack, all-gather and scatter of one pass, as one SPMD program over the
    TP mesh: what each rank's landing does, with the rank never a constant."""
    import jax
    from jax.sharding import PartitionSpec as P

    def body(pool, staging, pack, src, dst):
        send = staging[0][pack]  # [Q, 256, W]: this rank's compact pages
        gathered = jax.lax.all_gather(send, "tp", axis=0, tiled=True)  # [G * Q, 256, W]
        rows = gathered.reshape(-1, width)[src]
        return pool.at[0, dst].set(rows)

    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P(), P(), P()),
            out_specs=P("tp"),
        ),
        donate_argnums=(0,),
    )


@pytest.mark.parametrize(
    "num_tokens,layers,width,skip,packed",
    [(40_000, 3, 1024, 2048, 2), (88_000, 12, 64, 0, 8), (88_000, 12, 64, 34_816, 1)],
)
def test_spmd_landing_on_cpu_mesh_is_byte_identical(num_tokens, layers, width, skip, packed):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = _mesh()
    produced, dests, pool0 = make_load(num_tokens, layers, width, skip, packed)
    ref = reference_pool(num_tokens, produced, dests, pool0, skip)
    staging, tag_pages = land(num_tokens, produced, skip, np.random.default_rng(2))
    tags = [sc.SharedTag(p, tuple(t), o) for p, (t, o) in zip(tag_pages, dests)]
    passes = sc.gather_passes(num_tokens, tags, DST_BLOCK, GEO, skip_tokens=skip)
    shard = NamedSharding(mesh, P("tp"))
    pools = jax.device_put(np.broadcast_to(pool0, (G,) + pool0.shape).copy(), shard)
    staged = jax.device_put(staging, shard)
    programs = {}
    for p in passes:
        fn = programs.setdefault(p.bucket, shared_landing_program(mesh, p.bucket, width))
        pools = fn(
            pools, staged,
            jnp.asarray(p.pack, jnp.int32),
            jnp.asarray(p.scatter_src, jnp.int32),
            jnp.asarray(p.scatter_dst, jnp.int32),
        )
    out = np.asarray(pools)
    for rank in range(G):
        assert np.array_equal(out[rank], ref), f"rank {rank} differs"


def test_refuses_a_decode_without_the_synchronized_repack():
    sc.check_admissible(tp_size=8, sync_repack=True, sync_repack_ab_s=0)
    for kwargs in (
        dict(tp_size=8, sync_repack=False, sync_repack_ab_s=0),
        dict(tp_size=8, sync_repack=True, sync_repack_ab_s=60),
    ):
        with pytest.raises(ValueError, match="QWEN4_EXP_PD_SYNC_REPACK=1"):
            sc.check_admissible(**kwargs)
    with pytest.raises(ValueError, match="TP>1"):
        sc.check_admissible(tp_size=1, sync_repack=True, sync_repack_ab_s=0)
