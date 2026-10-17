# SPDX-License-Identifier: Apache-2.0
"""FP8 MLA pages retain BF16/FP32 state bytes and explicit transfer encoding."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    lower_dcp_page_spans,
    lower_glm_row_spans,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
    measured_kimi_k3_layout_fingerprint,
)

from .raiden_test_utils import FakeTensor


def _manifest(fp8):
    packing, itemsize, dtype = (4, 1, "float8_e4m3fn") if fp8 else (2, 2, "bfloat16")
    pool = FakeTensor((4, 128, packing, 640), itemsize, dtype=f"torch.{dtype}")
    attention, state = "model.layers.3.attn", "model.layers.0.kda"
    groups = [
        SimpleNamespace(
            layer_names=[attention],
            kv_cache_spec=SimpleNamespace(
                block_size=128 * packing, page_size_bytes=327680
            ),
        ),
        SimpleNamespace(
            layer_names=[state],
            kv_cache_spec=SimpleNamespace(
                shapes=((3, 3, 3, 128), (3, 128, 128)),
                dtypes=("torch.float32", "torch.float32"),
                page_size_bytes=327680,
            ),
        ),
    ]
    return rpm.build_kimi_k3_pool_manifest(
        named_kv_caches={attention: pool, state: [pool]},
        kv_cache_groups=groups,
        raw_tensors=[pool],
        mamba_group_ordinal_by_layer={state: 0},
        extra_state_rows=15,
    )


def test_fp8_manifest_preserves_state_extents_and_distinguishes_encoding():
    references = []
    for fp8 in (False, True):
        manifest = _manifest(fp8)
        packing = 4 if fp8 else 2
        references.append(
            measured_kimi_k3_layout_fingerprint(
                manifest,
                page_tokens=128 * packing,
                state_fragment_heads=3,
                state_fragments=1,
                layout_getter=lambda _, packing=packing: (
                    [3, 2, 1, 0],
                    [[packing, 128], [packing, 1]],
                    0,
                ),
                package_version=lambda _: "test",
            )
        )
    assert references[0][0] != references[1][0]
    for a, b in zip(_manifest(False).pools, _manifest(True).pools, strict=True):
        assert a.tag == b.tag
        assert (a.base_offset_bytes, a.regions, a.block_stride_bytes) == (
            b.base_offset_bytes,
            b.regions,
            b.block_stride_bytes,
        )
    regions = {p.tag: p for p in _manifest(True).pools}
    assert regions["gdn.ssm.g0"].live_bytes_per_block == 3 * 128 * 128 * 4
    assert regions["gdn.conv.g0"].base_offset_bytes == 77 * 2560
    assert regions["gdn.conv.g0"].live_bytes_per_block == 19 * 2560


@pytest.mark.parametrize("tail", [1, 3, 4, 5, 511, 512])
def test_fp8_transfer_tail_and_dcp_prefix_bytes(tail):
    # The ordinary replicated-prefix transfer rounds up to four-token rows.
    partial = lower_glm_row_spans(
        tag="fa",
        num_tokens=512 + tail,
        transfer_rank=0,
        parallelism=1,
        page_tokens=512,
        live_bytes_per_block=327680,
        row_bytes=2560,
        block_ids=[7, 8],
    )
    assert partial.spans[-1].size_bytes == ((tail + 3) // 4) * 2560
    # Megakernel DCP ships complete local pages. Prefix cuts use these page
    # byte offsets, and each destination receives its own rank's bytes.
    dcp = lower_dcp_page_spans(
        tag="fa",
        transfer_rank=17,
        live_bytes_per_block=327680,
        block_ids=[7, 8],
    )
    assert dcp.declared_bytes == 2 * 327680
    assert [s.dst_offset_bytes for s in dcp.spans] == [0, 327680]
    assert all(s.dst_unit_ordinal == 17 and s.size_bytes == 327680 for s in dcp.spans)
