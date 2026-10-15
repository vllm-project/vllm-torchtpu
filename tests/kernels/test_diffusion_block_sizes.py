# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the diffusion flash-attention block-size search."""

import dataclasses

import pytest

from vllm_torchtpu.kernels.diffusion_attention import block_sizes as bs

# A 128 MiB-VMEM spec standing in for v6e, which has not been measured.
_GEN6_SPEC = dataclasses.replace(
    bs.V7_SPEC, name="v6", generation=6, vmem_bytes=128 * 2**20
)

# (q_len, kv_len, head_dim): Wan 2.x 480p/720p 81 frames on 1, 4 (Ulysses)
# and 8 (CFG 2 x Ulysses 4) devices, Wan 480p 17 frames, FLUX, Z-Image,
# LTX-2 and a narrow-head case; self- and cross-attention.
_SHAPES = [
    (q, kv, hd)
    for q, hd in [
        (75600, 128),
        (32760, 128),
        (18900, 128),
        (8190, 128),
        (7800, 128),
        (4608, 128),
        (4128, 128),
        (30720, 128),
        (4096, 64),
    ]
    for kv in (q, 512, 1024)
]


@pytest.mark.parametrize("spec", [bs.V7_SPEC, _GEN6_SPEC], ids=lambda s: s.name)
@pytest.mark.parametrize("mode", ["online", "fixedm"])
@pytest.mark.parametrize("q_len,kv_len,head_dim", _SHAPES)
def test_block_sizes_fit_vmem(spec, mode, q_len, kv_len, head_dim):
    blocks = bs.autotune_block_sizes(q_len, head_dim, 2, spec, kv_len, mode)
    footprint = bs.vmem_footprint_bytes(
        blocks["block_q"], blocks["block_kv_compute"], head_dim, blocks["block_kv"]
    )
    assert footprint <= spec.vmem_bytes - bs._VMEM_RESERVE_BYTES


def test_block_sizes_v7_picks():
    spec = bs.V7_SPEC
    # Fixed-m self-attention picks (head_dim 128) are unchanged by the cap.
    for seq, block_q in [(32760, 4096), (8190, 4096), (75600, 3840), (18900, 3840)]:
        assert bs.autotune_block_sizes(seq, 128, 2, spec, seq, "fixedm") == {
            "block_q": block_q,
            "block_kv": 2048,
            "block_kv_compute": 2048,
        }
    # Wan 720p single-device cross-attention picked block_q 18944, which
    # needs 64.89 MiB of VMEM on v7x (physical 64 MiB) and failed to compile.
    assert bs.autotune_block_sizes(75600, 128, 2, spec, 512, "online") == {
        "block_q": 15360,
        "block_kv": 512,
        "block_kv_compute": 512,
    }
