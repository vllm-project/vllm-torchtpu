# SPDX-License-Identifier: Apache-2.0
"""The single-layer fused-MoE bench's host side (``bench moe``): one layer's
weights read alone match the prefill loader's, routing cases keep their
counts, the load model is the kernel's tiling, the synthesized inputs route
as designed through the kernel's own mixture and Top16, and the reference's
expert rows match the kernel's tiles."""

import importlib.util
import json
import pathlib
import sys
import types

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3 import kernels as model
from vllm_torchtpu.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k
from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader as rl
from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe import fused

pytestmark = pytest.mark.cpu_test

_PATH = pathlib.Path(__file__).resolve().parents[3] / "tools/megakernel/moe_layer.py"
_SPEC = importlib.util.spec_from_file_location("moe_layer", _PATH)
ml = importlib.util.module_from_spec(_SPEC)
sys.modules["moe_layer"] = ml
_SPEC.loader.exec_module(ml)

# ----------------------------------------------------------------- weights

LAYERS = 3  # two MoE layers
EXPERT_SHAPES = {  # per TP4 quarter; gate/up: two interleave chunks per half
    "expert_gu": (2, 112, 2, 512),
    "expert_gus": (2, 112, 1, 512),
    "expert_down": (2, 112, 6, 4),
    "expert_ds": (2, 112, 2, 4),
}
RANGES = [(0, 1), (1, 2)]
CACHE = ("latent_down", "latent_up", "shared_gu_0", "shared_gu_1")
CACHE += ("shared_down_0", "shared_down_1")


def _write(directory, name, arrays):
    """Families stored back to back in one file: their index entries."""
    families, blob, offset = {}, [], 0
    for family, value in arrays.items():
        families[family] = {
            "file": name,
            "offset": offset,
            "shape": list(value.shape),
            "dtype": "float32",
        }
        blob.append(value.tobytes())
        offset += value.nbytes
    (directory / name).write_bytes(b"".join(blob))
    return families


@pytest.fixture(scope="module")
def preshard(tmp_path_factory):
    """A small 32-rank pre-shard whose every layer and rank differ."""
    root = tmp_path_factory.mktemp("preshard")
    rng = np.random.default_rng(0)
    names = rl.PREFILL_IO + rl.PREFILL_COMMON + rl.PREFILL_KDA + rl.PREFILL_MLA
    for rank in range(32):
        directory = root / f"rank{rank:02d}"
        directory.mkdir()
        dense = {
            n: rng.standard_normal((LAYERS, 1, 8)).astype(np.float32) for n in names
        }
        dense.update(
            k_o=rng.standard_normal((LAYERS, 768, 3)).astype(np.float32),
            m_o=rng.standard_normal((LAYERS, 768, 3)).astype(np.float32),
            latent_norm=rng.standard_normal((2, 1, 8)).astype(np.float32),
            router=rng.standard_normal((2, 5, 896)).astype(np.float32),
            router_bias=rng.standard_normal((2, 1, 896)).astype(np.float32),
        )
        families = _write(directory, "dense.bin", dense)
        for name, shape in EXPERT_SHAPES.items():
            files = []
            for index, (lo, hi) in enumerate(RANGES):
                part = rng.standard_normal((hi - lo, *shape[1:])).astype(np.float32)
                files.append(f"{name}_{index}.bin")
                part.tofile(directory / files[-1])
            families[name] = {
                "shape": list(shape),
                "dtype": "float32",
                "files": files,
                "layer_ranges": [list(r) for r in RANGES],
            }
        index = {"rank": rank, "layers": LAYERS, "families": families}
        (directory / "index.json").write_text(json.dumps(index))
    cache = root / "prefill-sp-cache"
    cache.mkdir()
    layout = {}
    for name in CACHE:
        rng.standard_normal((2, 4, 6)).astype(np.float32).tofile(cache / f"{name}.bin")
        layout[name] = {"file": f"{name}.bin", "shape": [2, 4, 6], "dtype": "float32"}
    (cache / "layout.json").write_text(
        json.dumps({"layers": LAYERS, "families": layout})
    )
    order = np.stack([rng.permutation(896) for _ in range(LAYERS - 1)])
    np.savez(root / "plan.npz", order=order.astype(np.int32))
    return root


@pytest.mark.parametrize("plan", [True, False])
@pytest.mark.parametrize("rank", [0, 1, 6, 31])
def test_read_moe_layer_is_a_layer_of_the_prefill_loader(
    monkeypatch, preshard, rank, plan
):
    config = KernelConfig(expert_plan=str(preshard / "plan.npz") if plan else "")
    monkeypatch.setattr(rl, "_config", lambda: config)
    whole = {}
    for group, name, value in rl.iter_prefill_rank(
        preshard, preshard / "prefill-sp-cache", rank, layers=LAYERS
    ):
        family, _, part = name.partition("@")
        whole.setdefault((group, family), []).append((part, value))
    whole = {
        key: np.concatenate([value for _, value in sorted(parts)])
        for key, parts in whole.items()
    }
    for layer in range(LAYERS - 1):
        alone = rl.read_moe_layer(preshard, rank, layer, layers=LAYERS)
        assert set(alone) == set(k.MOE_FAMILIES) | {"mlp_res", "ffn_norm"}
        for name in k.MOE_FAMILIES:
            np.testing.assert_array_equal(
                alone[name], whole[("moe", name)][layer : layer + 1], err_msg=name
            )
        for name in ("mlp_res", "ffn_norm"):
            np.testing.assert_array_equal(
                alone[name], whole[("common", name)][layer + 1]
            )


# ----------------------------------------------------------------- routing


def test_counted_routes_keep_their_counts_and_distinct_experts():
    rng = np.random.default_rng(3)
    counts = rng.multinomial(16 * 4096, rng.dirichlet(np.full(896, 2.0)))
    sets = ml.sets_from_counts(counts, seed=7)
    assert sets.shape == (4096, 16)
    np.testing.assert_array_equal(np.bincount(sets.ravel(), minlength=896), counts)
    assert all(len(set(row)) == 16 for row in sets.tolist())
    np.testing.assert_array_equal(sets, ml.sets_from_counts(counts, seed=7))
    crowded = np.zeros(896, np.int64)  # 64 tokens; one expert holds 65 routes
    crowded[0], crowded[1:15], crowded[15] = 65, 64, 63
    with pytest.raises(ValueError, match="holds 65 routes"):
        ml.sets_from_counts(crowded, seed=0)


def test_uniform_sets_are_distinct_and_seeded():
    sets = ml.uniform_sets(2048, seed=5)
    assert sets.shape == (2048, 16) and sets.min() >= 0 and sets.max() < 896
    assert all(len(set(row)) == 16 for row in sets.tolist())
    np.testing.assert_array_equal(sets, ml.uniform_sets(2048, seed=5))
    assert not np.array_equal(sets, ml.uniform_sets(2048, seed=6))


def test_parse_places_served_counts_by_the_plan(tmp_path):
    rng = np.random.default_rng(4)
    logical = np.stack(  # distinct loads: 1024, 4096 and 16384 tokens
        [rng.multinomial(16 * 4**i * 1024, np.ones(896) / 896) for i in range(3)]
    )
    np.savez(
        tmp_path / "routes.npz",
        counts=logical.astype(np.int16),
        window=np.array([7, 7, 9]),
        layer=np.array([2, 5, 2]),
    )
    order = np.stack([rng.permutation(896) for _ in range(92)])
    np.save(tmp_path / "slots.npy", logical[:2])
    cases = ml.parse(
        f"uniform:3,serving:9:2,sweep:2,uniform:3,counts:{tmp_path / 'slots.npy'}",
        order=order,
        routes_file=str(tmp_path / "routes.npz"),
    )
    labels = [c.label for c in cases]
    assert labels[:2] == ["uniform:3", "serving:9:2"] and cases[0].seed == 3
    assert cases[1].fits(16384) and not cases[1].fits(8192) and cases[0].fits(512)
    np.testing.assert_array_equal(cases[1].counts, logical[2][order[2]])
    assert (cases[1].window, cases[1].layer) == (9, 2)
    assert labels[2] == "serving:7:2"  # sweep's lightest; its heaviest is a repeat
    assert len(labels) == len(set(labels)) == 5
    assert labels[-2:] == [f"counts:{tmp_path / 'slots.npy'}#{i}" for i in (0, 1)]
    with pytest.raises(ValueError, match="no window 8"):
        ml.parse("serving:8:2", order=order, routes_file=str(tmp_path / "routes.npz"))


def _kernel_metadata(counts, tile):
    """The routed loop's tiles as the kernel's ``fill_metadata`` cuts them."""
    counts = np.asarray(counts, np.int32)
    cfgs = types.SimpleNamespace(
        dims=types.SimpleNamespace(size_group=56, size_lhs_sublane=16),
        tiles=types.SimpleNamespace(tile_m=tile),
    )
    size = 56 + -(-int(counts.sum()) // tile) + 1

    def kernel(counts_ref, offset_ref, offsets_ref, groups_ref, count_ref):
        metadata = k.MetadataRef(
            gm_id_to_group_id=groups_ref, gm_id_to_m_offset=offsets_ref
        )
        count_ref[0] = k.fill_metadata(counts_ref, offset_ref, metadata, cfgs=cfgs)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    offsets, _, count = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((size + 1,), jnp.int32),
            jax.ShapeDtypeStruct((size,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        in_specs=[smem, smem],
        out_specs=(smem, smem, smem),
        interpret=True,
    )(jnp.asarray(counts), jnp.zeros((1,), jnp.int32))
    return np.asarray(offsets)[: int(count[0]) + 1]


@pytest.mark.parametrize("tile", [256, 128])
def test_chip_loads_follow_the_kernels_tiles_and_buckets(tile):
    rng = np.random.default_rng(8 + tile)
    counts = rng.poisson(300 if tile == 256 else 150, 896)
    counts[rng.choice(896, 40, replace=False)] = 0
    counts[rng.choice(896, 40, replace=False)] = rng.integers(1, 20, 40)
    loads = ml.chip_loads(counts, tile)
    first_bucket = 1 if tile == 256 else 0  # fused._gu_inner
    for chip in (0, 5, 15):
        offsets = _kernel_metadata(counts[56 * chip : 56 * chip + 56], tile)
        start, end = offsets[:-1], offsets[1:]
        last = end - (start - start % 16)
        gu = 64 * (
            np.maximum((np.maximum(last - 1, 0)) // 64 - first_bucket, 0)
            + first_bucket
            + 1
        )
        down = 64 * (np.maximum(last - 1, 0) // 64 + 1)
        assert loads["visits"][chip] == len(start)
        assert loads["routed"][chip] == counts[56 * chip : 56 * chip + 56].sum()
        assert loads["gu_rows"][chip] == gu.sum()
        assert loads["down_rows"][chip] == down.sum()


def test_compound_launch_loads_sum_their_parts():
    sets = ml.uniform_sets(10240, seed=1)
    loads = ml.case_loads(sets, 10240, fused._COMPOUND_PARTS[10240])
    big = ml.launch_loads(np.bincount(sets[:8192].ravel(), minlength=896), 8192)
    small = ml.launch_loads(np.bincount(sets[8192:].ravel(), minlength=896), 2048)
    assert loads["visits"] == big["visits"].max() + small["visits"].max()
    assert loads["gu_rows"] == big["gu_rows"].max() + small["gu_rows"].max()


@pytest.mark.parametrize(
    ("layout", "index"),
    [
        ((1, 1, 0), 0),  # the bench's default: the layer alone at slot 0
        ((3, 92, 2), 2),  # the prefill role's first bank (MoE layers 0..2)
        ((8, 92, 47), 4),  # a regular bank: layers 43..50
        ((1, 92, 91), 0),  # the final bank
        ((92, 92, 91), 91),  # the aggregate's families
        ((14, 14, 13), 13),
    ],
)
def test_layouts_index_the_layer_as_the_kernel_does(layout, index):
    assert ml.Layout(*layout).expert_index == index
    assert ml.Layout(*layout).problem() == ""
    value = np.arange(6, dtype=np.uint32).reshape(1, 2, 3) + 1
    held = ml.held_family(value, layout[0], index)
    assert held.shape == (layout[0], 2, 3)
    np.testing.assert_array_equal(held[index], value[0])
    assert held.sum() == value.sum()


def test_layouts_refuse_slots_outside_their_families():
    assert "dense layers" in ml.Layout(92, 1, 5).problem()
    assert "expert layers" in ml.Layout(13, 14, 13).problem()


def test_layout_entries_place_experts_by_bank_and_the_rest_by_slot():
    entries = ml.Layout(8, 92, 47).entries(("router", "expert_gu", "expert_ds"))
    assert entries == {"router": (92, 47), "expert_gu": (8, 4), "expert_ds": (8, 4)}


@pytest.mark.parametrize(("routes", "over"), [(65000, False), (70000, True)])
def test_capacity_cases_put_their_routes_on_one_chip(routes, over):
    (case,) = ml.parse(f"capacity:{routes}:3", order=np.tile(np.arange(896), (92, 1)))
    assert case.fits(16384) and not case.fits(routes // 16 - 1)
    sets = case.sets(case.tokens(16384, None))
    counts = np.bincount(sets.ravel(), minlength=896)
    assert counts[56 * 3 : 56 * 4].sum() == routes and counts.sum() == 16 * 16384
    assert counts.max() - counts[56 * 3 : 56 * 4].min() <= 1  # even on the chip
    assert all(len(set(row)) == 16 for row in sets[:4096].tolist())
    loads = ml.case_loads(sets, 16384, (16384,))
    assert loads["routed"] == routes and loads["overflow"] is over
    assert fused._route_capacity(16384) == 65536


_OVERLAY_LOG = """\
I1113 overlay.cc:7142] OverlayProgram(), input program bundle count: 1297546
I1113 overlay.cc:7218] OverlayProgram(), continuation_program_info, offset: 1297546
I1113 overlay_partitioner.cc:1419] Partitions:
I1113 overlay_partitioner.cc:1419] OverlayPartitioner (regular/jumbo/hlo overlay \
size 16264/16264/48792, prefix size 9, trampoline size 13):
I1113 overlay_partitioner.cc:1419] OverlayPartition 7: [0, 16100):16100, kind=0
I1113 overlay_partitioner.cc:1419] OverlayPartition 8: [16100, 30000):13900, kind=1
I1113 overlay_partitioner.cc:1419] OverlayPartition 9: [30000, 46000):16000, kind=0
I1113 overlay.cc:7142] OverlayProgram(), input program bundle count: 412
"""


def test_overlays_follow_the_bench_compile_markers():
    log = "\n".join(
        [
            "[   10.0s r00] bench moe compile begins: w16384",
            _OVERLAY_LOG,
            "[  400.0s r00] bench moe compile ends: w16384: first launch 390.0 s",
            "I1113 overlay.cc:7142] OverlayProgram(), input program bundle count: 5",
            "[  401.0s r00] bench moe compile begins: w8192",
            "[  402.0s r00] bench moe compile ends: w8192: first launch 1.0 s",
        ]
    )
    by_window = ml.overlays_by_window(log)
    assert list(by_window) == [16384, 8192]
    assert [p["bundles"] for p in by_window[16384]] == [1297546, 412]
    assert by_window[8192] == []  # a cached executable leaves no records


def test_overlay_programs_read_the_compile_log():
    first, second = ml.overlay_programs(_OVERLAY_LOG)
    assert first == {
        "bundles": 1297546,
        "sizes": "16264/16264/48792",
        "partitions": 3,
        "kind0": 2,
        "kind1": 1,
    }
    assert second == {"bundles": 412, "sizes": "", "partitions": 0}
    assert ml.overlay_programs("no compile here") == []


# ----------------------------------------------------------------- inputs


@pytest.fixture(scope="module")
def router_layer():
    """Router, bias and norm weights of real shape and plausible scale."""
    rng = np.random.default_rng(11)
    router = ml.bf16(rng.standard_normal((7168, 896)) * 0.02)
    norm = ml.bf16(rng.uniform(0.05, 1.0, 7168))
    bias = rng.uniform(-0.25, 0.25, 896).astype(np.float32)
    return router, norm, bias, ml.HiddenSolver(router, norm)


def test_mixture_of_equal_blocks_returns_the_designed_hidden_state(router_layer):
    router, norm, bias, solver = router_layer
    rng = np.random.default_rng(12)
    sets = ml.uniform_sets(256, seed=2)
    valid = np.ones(256, bool)
    logits = ml.target_logits(sets, valid, bias, rng, margin=0.1)
    residual, _ = solver.residual(logits, rng)
    mixture = k.make_mixture_norm_tiled(seq_len=256, tile=128, interpret=True)
    rows = jnp.asarray(residual.astype(ml_dtypes.bfloat16))
    hidden = mixture(
        rows,
        jnp.broadcast_to(rows, (8, *rows.shape)),
        jnp.asarray(rng.standard_normal((1, 7168)), jnp.float32),
        jnp.asarray(norm.astype(ml_dtypes.bfloat16))[None],
        jnp.asarray([5], jnp.int32),
    )
    hidden = np.asarray(hidden).astype(np.float32)
    expected = ml.post_norm(residual, norm)
    ulp = np.abs(expected) * 2.0**-7
    assert np.all(np.abs(hidden - expected) <= ulp)
    assert np.mean(hidden == expected) > 0.999
    ml.check_selection(hidden, router, bias, sets, valid)


@pytest.mark.parametrize("spread", [0.0, 0.5])
def test_synthesized_rows_select_their_experts_in_the_kernels_top16(
    router_layer, spread
):
    router, norm, bias, solver = router_layer
    bias = bias * (spread / 0.5)
    rng = np.random.default_rng(13)
    counts = rng.multinomial(16 * 16384, rng.dirichlet(np.full(896, 5.0)))
    sets = ml.sets_from_counts(counts, seed=1)[:512]
    valid = np.arange(512) < 480
    logits = ml.target_logits(sets, valid, bias, rng, margin=0.1)
    residual, scaled = solver.residual(logits, rng)
    assert scaled == 0
    np.testing.assert_allclose(np.sqrt(np.mean(residual**2, axis=1)), 1, atol=1e-2)
    hidden = ml.post_norm(residual, norm)
    margin = ml.check_selection(hidden, router, bias, sets, valid)
    assert margin > 0.05
    scores = jnp.asarray(hidden @ router)
    selected, weights = fused._Top16(scores, jnp.asarray(bias)[None]).finish(
        jnp.asarray(valid)[:, None]
    )
    selected = np.sort(np.asarray(selected), axis=1)
    np.testing.assert_array_equal(selected[valid], np.sort(sets[valid], axis=1))
    assert np.all(np.asarray(weights)[~valid] == 0)
    ours, _ = ml.top16(hidden @ router, bias)
    np.testing.assert_array_equal(np.sort(ours, axis=1), selected)


def test_selection_needs_a_gap_between_the_16th_and_17th_scores(router_layer):
    router, norm, bias, solver = router_layer
    rng = np.random.default_rng(16)
    sets = ml.uniform_sets(8, seed=3)
    valid = np.ones(8, bool)
    logits = ml.target_logits(sets, valid, bias, rng, margin=0.1)
    hidden = ml.post_norm(solver.residual(logits, rng)[0], norm)
    margin = ml.check_selection(hidden, router, bias, sets, valid)
    with pytest.raises(ValueError, match="apart"):
        ml.check_selection(hidden, router, bias, sets, valid, least=margin + 1e-3)


def test_inputs_digest_tells_any_change_apart():
    rows = ml.bf16(np.random.default_rng(17).standard_normal((4, 8)))
    digest = ml.inputs_digest(16384, rows, rows)
    assert digest == ml.inputs_digest(16384, rows.copy(), rows.copy())
    changed = rows.copy()
    changed[2, 3] = np.nextafter(changed[2, 3], np.float32(np.inf))
    assert digest != ml.inputs_digest(16384, rows, changed)
    assert digest != ml.inputs_digest(16383, rows, rows)


def test_unreachable_selection_fails_loudly():
    bias = np.zeros(896, np.float32)
    bias[0] = 1.5  # expert 0 outscores any expert without it, whatever the logits
    sets = np.arange(1, 17)[None]
    with pytest.raises(ValueError, match="router bias spans"):
        ml.target_logits(
            sets, np.ones(1, bool), bias, np.random.default_rng(0), margin=0.1
        )


# ----------------------------------------------------------------- reference


def _mxfp4(rng, rows, cols):
    """Random packed MXFP4 weights [rows / 8, cols] and E8M0 scales."""
    codes = rng.integers(0, 16, size=(rows // 8, 8, cols)).astype(np.uint32)
    packed = np.bitwise_or.reduce(
        codes << (4 * np.arange(8, dtype=np.uint32))[:, None], 1
    )
    scales = rng.integers(122, 130, size=(rows // 32, cols)).astype(np.uint8)
    return packed.astype(np.uint32), scales


def test_dequantize_is_the_checkpoint_unpacking():
    packed, scales = _mxfp4(np.random.default_rng(14), 256, 384)
    np.testing.assert_array_equal(
        ml.dequantize(packed, scales),
        np.asarray(model.unpack_mxfp4(jnp.asarray(packed), jnp.asarray(scales))).astype(
            np.float32
        ),
    )


class _Ref:
    """A numpy array behind the ref slicing the kernel's tile code uses."""

    def __init__(self, array):
        self.array = array
        self.shape, self.dtype = array.shape, array.dtype

    @property
    def at(self):
        array = self.array

        class _At:
            def __getitem__(self, index):
                index = tuple(
                    slice(int(i.start), int(i.start) + int(i.size))
                    if hasattr(i, "size") and not isinstance(i, slice)
                    else i
                    for i in (index if isinstance(index, tuple) else (index,))
                )
                return _Ref(array[index])

        return _At()

    def reshape(self, *shape):
        return _Ref(self.array.reshape(*shape))

    def __getitem__(self, index):
        return jnp.asarray(self.array[index])


def _cfgs(*, tile_k, fuse_act):
    return types.SimpleNamespace(
        dims=types.SimpleNamespace(size_k=tile_k),
        tiles=types.SimpleNamespace(tile_m=256, tile_k=tile_k),
        rhs_cfgs=types.SimpleNamespace(has_bias=False, quant_block_size=32),
        lhs_cfgs=types.SimpleNamespace(dtype=jnp.bfloat16),
        acc_dtype=jnp.float32,
        fuse_act=fuse_act,
    )


def _fp4(packed, scales):
    """Packed MXFP4 as the kernel's refs see it: FP4 codes and E8M0 bytes."""
    codes = (packed[:, None, :] >> (4 * np.arange(8, dtype=np.uint32))[:, None]) & 15
    values = ml.FP4[codes.reshape(-1, packed.shape[-1])]
    return values.astype(ml_dtypes.float4_e2m1fn), scales


@pytest.fixture(autouse=True)
def tpu_info(monkeypatch):
    info = types.SimpleNamespace(mxu_column_size=256, num_lanes=128)
    monkeypatch.setattr(fused.pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(k.pltpu, "get_tpu_info", lambda: info)


def test_expert_rows_match_the_kernels_tiles():
    rng = np.random.default_rng(15)
    rows, latent, mid, width = 256, 512, 512, 384
    gu_packed, gu_scales = _mxfp4(rng, latent, 2 * mid)
    down_packed, down_scales = _mxfp4(rng, mid, width)
    lhs = ml.bf16(rng.standard_normal((rows, latent)))
    route = rng.uniform(0.02, 0.1, rows).astype(np.float32)
    gu_cfgs = _cfgs(tile_k=latent, fuse_act="situ:4:25")
    weights, scales = _fp4(gu_packed, gu_scales)
    rhs = k.WeightsRef(weight=_Ref(weights), scale=_Ref(scales), bias=None)
    panels = fused._product_panels(
        _Ref(lhs.astype(ml_dtypes.bfloat16)),
        rhs,
        cfgs=gu_cfgs,
        bucket_rows=rows,
        overlap_work=lambda _: None,
    )
    middle = np.concatenate(
        [np.asarray(k.apply_act_fn(value, gu_cfgs.fuse_act)) for _, value in panels], 1
    )
    middle = ml.bf16(middle * route[:, None])
    weights, scales = _fp4(down_packed, down_scales)
    rhs = k.WeightsRef(weight=_Ref(weights), scale=_Ref(scales), bias=None)
    kernel = fused._expert_product(
        _Ref(middle.astype(ml_dtypes.bfloat16)),
        rhs,
        cfgs=_cfgs(tile_k=mid, fuse_act=None),
        bucket_rows=rows,
        overlap_work=lambda _: None,
    )
    kernel = ml.bf16(np.asarray(kernel))
    ours = ml.expert_rows(
        lhs,
        route,
        ml.dequantize(gu_packed, gu_scales),
        ml.dequantize(down_packed, down_scales),
    )
    np.testing.assert_allclose(
        ours, kernel, rtol=2e-2, atol=2e-2 * np.abs(kernel).max()
    )
    assert np.mean(ours == kernel) > 0.9
