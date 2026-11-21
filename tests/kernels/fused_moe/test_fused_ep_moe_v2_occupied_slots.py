"""CPU tests: an expert layout's empty staging slots do not change the build.

Dynamic EPLB gives each shard spare staging slots that never hold rows (48
active + 1 replica + 7 staging = 56 for DeepSeek-V4.1 at EP8). The fused EP v2
build's density estimates -- the tile-height ladder and the VMEM push -- used
to count those slots, so a 56-slot layout got a different build than the
48-slot one at the 1024-token bucket (short-tile ladder + VMEM push), which a
TPU microbenchmark measured 28-32% slower under skewed routing.
"""

import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import kernel

TILE_M, EP = 128, 8

# (rows_alloc at 48 slots, rows_alloc at 56 slots): the target builds both
# DSv4.1 serving runs logged (the slab grows by the extra slots' alignment).
TARGET_BUILDS = [
    (1280, 1408),
    (2048, 2176),
    (3584, 3712),
    (6656, 6784),
    (7424, 7552),
    (12800, 12928),
    (25088, 25216),
    (49664, 49792),
]
# The DSpark drafter's builds: 16 logical experts per shard, 24 slots.
DRAFT_BUILDS = [(1280, 1280), (2176, 2304), (4096, 4224), (7936, 8064)]


@pytest.mark.parametrize("g_local", [8, 16, 48, 56])
@pytest.mark.parametrize("rows", [1280, 2048, 7424, 12800, 49664, 65536])
def test_without_occupied_slots_the_choices_are_unchanged(g_local, rows):
    rungs, push = kernel.density_choices(TILE_M, rows, g_local, EP)
    assert rungs == kernel.tile_height_rungs(TILE_M, rows, g_local, EP)
    assert push == (rows <= EP * g_local * TILE_M)


@pytest.mark.parametrize(
    "logical, slots, builds", [(48, 56, TARGET_BUILDS), (16, 24, DRAFT_BUILDS)]
)
def test_staging_slots_get_the_unplaced_build(logical, slots, builds):
    """With the occupied slots and the unplaced slab, a placed build makes
    exactly the unplaced build's choices at every logged bucket."""
    for rows_plain, rows_placed in builds:
        plain = kernel.density_choices(TILE_M, rows_plain, logical, EP)
        placed = kernel.density_choices(
            TILE_M,
            rows_placed,
            slots,
            EP,
            occupied_slots=logical,
            density_rows_alloc=rows_plain,
        )
        assert placed == plain, (rows_plain, rows_placed, plain, placed)


def test_occupied_slots_alone_fix_the_push_at_the_boundary_bucket():
    """At 8192 tokens x top-6 the routed rows (49152) sit exactly on one tile
    per expert at 48 slots (8 x 48 x 128). Occupied slots alone already keep
    the VMEM push off there; the tile ladder can still differ, because the
    placed slab's extra rows move the routed-row bound past it (which ladder
    that gives depends on the branch's short-tile rule), so the unplaced slab
    is passed too (test_staging_slots_get_the_unplaced_build)."""
    plain = kernel.density_choices(TILE_M, 49664, 48, EP)
    occupied_only = kernel.density_choices(TILE_M, 49792, 56, EP, occupied_slots=48)
    assert plain[1] is False and occupied_only[1] is False


def test_the_1024_token_bucket_differed_before():
    """The regression this guards: counting staging slots flips the build."""
    plain = kernel.density_choices(TILE_M, 49664, 48, EP)
    counted = kernel.density_choices(TILE_M, 49792, 56, EP)
    assert plain != counted
    assert plain[1] is False and counted[1] is True  # VMEM push only when counted


def test_occupied_slots_never_exceed_the_shard():
    assert kernel.density_choices(TILE_M, 49664, 48, EP, occupied_slots=99) == (
        kernel.density_choices(TILE_M, 49664, 48, EP)
    )
