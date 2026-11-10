# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# Source provenance: inferact/k3-megakernel@30c9ba966b
#   tests/megakernel/kimi_k3/test_dspark_attention_dp.py
"""Tests for DSpark under attention DP across verify-row placement and schedules."""

from __future__ import annotations

import base64
import hashlib
import re
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir as mlir_ir
from jax.experimental import topologies
from jax.experimental.mosaic.dialects import tpu as mosaic_tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.decode import KimiDecoder
from vllm_torchtpu.models.kimi_k3.megakernel.decode_ops import Control, dp_group_rows
from vllm_torchtpu.models.kimi_k3.megakernel.draft import DsparkDraft, _batched_fn
from vllm_torchtpu.models.kimi_k3.megakernel.draft_mla import MlaDsparkDraft
from vllm_torchtpu.models.kimi_k3.megakernel.draft_mla import (
    _batched_fn as _mla_batched_fn,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    dspark,
    dspark_mla,
    dspark_mla_kernel,
)
from vllm_torchtpu.models.kimi_k3.megakernel.program import spec_tokens_per_batch
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)

_MAX_PART_REQUESTS = 8
_EXPECTED_VERIFY8_HLO_BYTES, _EXPECTED_VERIFY8_HLO_WORDS = (
    (
        31311490,
        (
            0x5E9AB81B4163A692,
            0xD94A76DD4CF323F6,
            0x9CB49B39F54795F1,
            0x1E0A9697BB08D348,
        ),
    )
    if jax.__version__ == "0.11.1"
    else (
        # Computed by test_dspark_mla_verify8_canonical_hlo_sha256 under
        # jax==0.11.2 in vllm/vllm-torchtpu:nightly.
        31305070,
        (
            0xFD49D00140D69533,
            0x5916FB39C3485231,
            0x5273B02AF283A360,
            0xDA9CB02B1711A061,
        ),
    )
)
_EXPECTED_VERIFY8_CUSTOM_CALL_WORDS = (
    0xA8B5879A021C2C39,
    0xE891F080D41C41A9,
    0xCCEA188EC672C723,
    0x4EF02DADF690C693,
)


def _sha256_words(text: str) -> tuple[int, int, int, int]:
    """Return the 256-bit SHA256 digest of text as four big-endian 64-bit ints."""
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    return (
        int.from_bytes(raw[0:8], "big"),
        int.from_bytes(raw[8:16], "big"),
        int.from_bytes(raw[16:24], "big"),
        int.from_bytes(raw[24:32], "big"),
    )


def _normalize_mosaic_body(hlo_text: str) -> str:
    """Replace base64 Mosaic MLIR bodies with debug-stripped textual MLIR."""

    def _decode_body(match: re.Match[str]) -> str:
        """Decode one base64 custom_call_config.body and strip debug locations."""
        raw_bytes = base64.b64decode(match.group(1))
        with mlir_ir.Context() as ctx:
            mosaic_tpu.register_dialect(ctx)
            ctx.allow_unregistered_dialects = True
            module: Any = mlir_ir.Module.parse(raw_bytes)
            asm_text = str(module.operation.get_asm(enable_debug_info=False))
        return f'"body": "{asm_text}"'

    return re.sub(r'"body":\s*"([^"]+)"', _decode_body, hlo_text)


@pytest.mark.parametrize("batch", [8, 16, 32, 64, 96, 128])
def test_verify_rows_are_the_rows_each_group_attends(batch: int) -> None:
    """Verify dp_group_rows matches KimiDecoder group row ordering."""
    rows = dp_group_rows(batch, 2)
    decoder = types.SimpleNamespace(groups=2) if False else object.__new__(KimiDecoder)
    decoder.groups = 2
    expected = [
        KimiDecoder._group_rows(decoder, batch, group_idx) for group_idx in range(2)
    ]
    assert rows.tolist() == expected[0] + expected[1]


def _stub_fused(
    _weights: object,
    caches: jax.Array,
    positions: jax.Array,
    aux: jax.Array,
    logits: jax.Array,
    _vectors: jax.Array,
    bases: jax.Array,
    count: jax.Array,
    anchor: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Deterministic stub for DSpark fused_draft in multi-part DP tests."""
    out = jnp.zeros((bases.shape[0], 128), jnp.int32)
    out = out.at[:, dspark.OUT_COUNT].set(count).at[:, dspark.OUT_BONUS].set(anchor)
    # Outputs that depend on each request's own verify rows.
    out = out.at[:, 40].set(aux[0, :: dspark.ROWS, 0].astype(jnp.int32))
    out = out.at[:, 41].set(logits[0, :, 0, 0].astype(jnp.int32))
    return out, caches + 1, positions + bases[None, :, None]


@pytest.mark.parametrize("verified", [24, 32])
def test_drafting_in_parts_matches_one_call(verified: int) -> None:
    """Verify multi-part DSpark drafting matches a single full-batch call."""
    rng = np.random.default_rng(verified)
    rows_per, pages = 4, 3
    control = Control(slots=65, rows=64, pages=pages)
    slots = rng.permutation(np.arange(1, 65))[:verified]
    slots[[3, verified - 2]] = 0  # padding requests share the null slot
    entries = [
        (
            int(slot_id),
            idx % 3 == 0,
            int(rng.integers(5, 999)),
            int(rng.integers(9, 9000)),
            (0.0, 1.0, 0, 0),
        )
        for idx, slot_id in enumerate(slots)
    ]
    plan = control.plan(entries, [np.arange(pages, dtype=np.int32)] * verified)
    state = np.concatenate([rng.integers(0, 50, 3 * control.slots), plan]).astype(
        np.int32
    )
    total = verified * rows_per
    ids = jnp.asarray(rng.integers(0, 1000, 4 * total + verified * pages), jnp.int32)
    aux = jnp.asarray(rng.integers(0, 99, (5, total, 16)), jnp.bfloat16)
    logits = jnp.asarray(rng.integers(0, 99, (32, total, 32)), jnp.float32)
    decisions = jnp.asarray(
        np.stack(
            [rng.integers(0, rows_per, verified), rng.integers(0, 999, verified)], 1
        ),
        jnp.int32,
    )

    def draft(requests: int, first: int, state_arr: jax.Array) -> tuple[jax.Array, ...]:
        """Run one _batched_fn slice over [first, first + requests)."""
        fn = _batched_fn(
            dspark, _stub_fused, control, (), requests, rows_per, first, verified
        )
        return fn(
            jnp.zeros((1, requests, 2, 4, 8), jnp.bfloat16),
            jnp.zeros((1, requests, 4), jnp.int32),
            jnp.zeros((1, requests, 128), jnp.int32),
            aux,
            logits,
            ids,
            state_arr,
            decisions,
        )

    whole = draft(verified, 0, jnp.asarray(state))
    parts, part_state = [], jnp.asarray(state)
    for first, requests in DsparkDraft.parts(verified, rows_per):
        out = draft(requests, first, part_state)
        parts.append(out)
        part_state = out[4]
    np.testing.assert_array_equal(
        np.concatenate([p[0] for p in parts], axis=1), whole[0]
    )
    np.testing.assert_array_equal(np.concatenate([p[3] for p in parts]), whole[3])
    np.testing.assert_array_equal(part_state, whole[4])


def _stub_mla_fused(
    _weights: object,
    _pools: object,
    aux: jax.Array,
    logits: jax.Array,
    previous: jax.Array,
    bases: jax.Array,
    count: jax.Array,
    anchor: jax.Array,
    tables: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Deterministic stub for MLA fused_draft_local in multi-part DP tests."""
    out = jnp.zeros((bases.shape[0], 128), jnp.int32)
    out = out.at[:, dspark.OUT_COUNT].set(count).at[:, dspark.OUT_BONUS].set(anchor)
    # Outputs that depend on each request's own operands.
    out = out.at[:, 40].set(aux[0, :: logits.shape[1], 0].astype(jnp.int32))
    out = out.at[:, 41].set(logits[:, 0, 0].astype(jnp.int32))
    out = out.at[:, 42].set(tables[:, 0]).at[:, 43].set(previous[:, 0])
    rows = aux[0, :, :1].astype(jnp.float32)
    return out, jnp.stack([rows, rows + 1])  # [layers, rows, 1] context latents


def _stub_commit(
    pools: tuple[jax.Array, ...],
    latents: jax.Array,
    _tables: object,
    positions: jax.Array,
) -> tuple[jax.Array, ...]:
    """Deterministic stub for MLA latent pool commit."""
    return tuple(
        pool.at[positions].set(latents[layer, :, 0]) for layer, pool in enumerate(pools)
    )


@pytest.mark.parametrize(("verified", "dp"), [(16, 1), (16, 2), (32, 2)])
def test_mla_drafting_in_parts_matches_one_call(verified: int, dp: int) -> None:
    """Verify multi-part MLA drafting matches single-call MLA drafting."""
    rng = np.random.default_rng(verified + dp)
    rows_per, pages = 4, 3
    control = Control(slots=65, rows=64, pages=pages)
    slots = rng.permutation(np.arange(1, 65))[:verified]
    entries = [
        (int(slot_id), idx % 3 == 0, int(rng.integers(5, 999)), 0, (0.0, 1.0, 0, 0))
        for idx, slot_id in enumerate(slots)
    ]
    plan = control.plan(entries, [np.arange(pages, dtype=np.int32)] * verified)
    state = jnp.asarray(
        np.concatenate([rng.integers(0, 50, 3 * control.slots), plan]), jnp.int32
    )
    total = verified * rows_per
    bases = 100 * rng.permutation(verified)  # distinct positions per request
    positions = (bases[:, None] + np.arange(rows_per)[None, :]).reshape(-1)
    ids = np.concatenate(
        [
            rng.integers(0, 1000, total),
            positions,
            rng.integers(0, 1000, 2 * total),
            rng.integers(0, 1000, verified * pages),
        ]
    )
    ids_arr = jnp.asarray(ids, jnp.int32)
    aux = jnp.asarray(rng.integers(0, 99, (5, total, 16)), jnp.bfloat16)
    logits = jnp.asarray(rng.integers(0, 99, (1, total, 32)), jnp.float32)
    decisions = jnp.asarray(
        np.stack(
            [rng.integers(0, rows_per, verified), rng.integers(0, 999, verified)], 1
        ),
        jnp.int32,
    )
    previous = jnp.asarray(rng.integers(0, 999, (verified, 128)), jnp.int32)
    pools = (jnp.zeros(100 * verified, jnp.float32),) * 2

    def draft(
        index: tuple[int, ...], state_arr: jax.Array, pools_tuple: tuple[jax.Array, ...]
    ) -> tuple[Any, Any, Any, Any]:
        """Run one _mla_batched_fn slice over the selected request indices."""
        fn = _mla_batched_fn(
            dspark,
            _stub_mla_fused,
            _stub_commit,
            control,
            (),
            2,
            index,
            verified,
            rows_per,
        )
        num_active = len(index)
        out = fn(
            *pools_tuple,
            *(previous[j : j + 1] for j in index),
            aux,
            logits,
            ids_arr,
            state_arr,
            decisions,
        )
        return (
            out[:num_active],
            out[num_active : num_active + 2],
            out[num_active + 2],
            out[num_active + 3],
        )

    whole = draft(tuple(range(verified)), state, pools)
    vectors: list[Any] = [None] * verified
    emitted: list[Any] = []
    order: list[int] = []
    part_state, part_pools = state, pools
    for index in MlaDsparkDraft.parts(verified, dp):
        # At most eight a call; under DP as many of each group's, group-major.
        assert len(index) <= _MAX_PART_REQUESTS
        assert [j // (verified // dp) for j in index] == sorted(
            list(range(dp)) * (len(index) // dp)
        )
        part_vectors, part_pools, part_emitted, part_state = draft(
            index, part_state, part_pools
        )
        for j, vector in zip(index, part_vectors, strict=True):
            vectors[j] = vector
        emitted.append(part_emitted)
        order.extend(index)
    assert sorted(order) == list(range(verified))
    np.testing.assert_array_equal(np.concatenate(vectors), np.concatenate(whole[0]))
    for part_pool, whole_pool in zip(part_pools, whole[1], strict=True):
        np.testing.assert_array_equal(part_pool, whole_pool)
    np.testing.assert_array_equal(np.concatenate(emitted)[np.argsort(order)], whole[2])
    np.testing.assert_array_equal(part_state, whole[3])


def _config(
    attention_dp: int,
    schedule: tuple[tuple[int, int, int], ...],
    max_num_seqs: int = 32,
) -> MegakernelEngineConfig:
    """Build a test MegakernelEngineConfig for speculative schedule checks."""
    return MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32 // attention_dp,
        block_size=512,
        kv_cache_dtype="fp8",
        max_num_seqs=max_num_seqs,
        max_model_len=1 << 17,
        max_num_batched_tokens=512,
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=schedule,
        attention_dp=attention_dp,
    )


def test_attention_dp2_speculates_up_to_32_requests() -> None:
    """Verify spec_tokens_per_batch schedule resolution under attention_dp=2."""
    schedule = ((1, 2, 7), (3, 32, 3))
    dense = spec_tokens_per_batch(_config(2, schedule))
    assert dense is not None
    assert dense[1:3] == (7, 7)
    assert set(dense[3:]) == {3}
    with pytest.raises(NotImplementedError):
        spec_tokens_per_batch(_config(1, schedule))
    with pytest.raises(NotImplementedError):
        spec_tokens_per_batch(_config(2, ((1, 16, 7),)))


def test_dspark_mla_verify8_canonical_hlo_sha256() -> None:
    """Verify DSpark public surface and verify8 canonical HLO parity."""
    for helper in (dspark.bf16, dspark.f32, dspark.tile_rms_norm, dspark.select_rows):
        assert callable(helper)

    topo = topologies.get_topology_desc(topology_name="tpu7x_2x2x4", platform="tpu")
    assert topo is not None
    mesh_tp = Mesh(topo.devices, ("tp",))
    c_draft = dspark_mla.MlaDraftConfig()
    num_requests = 2
    rows = 8
    pages = 64
    block = 7
    attention_dp = 1
    total_rows = num_requests * rows

    fused_draft = dspark_mla_kernel.fused_draft_local(
        c_draft,
        requests=num_requests,
        pages=pages,
        block=block,
        rows=rows,
        attention_dp=attention_dp,
    )
    weights = {
        "ln1": jax.ShapeDtypeStruct((c_draft.layers, 1, c_draft.hidden), jnp.bfloat16),
        "ln2": jax.ShapeDtypeStruct((c_draft.layers, 1, c_draft.hidden), jnp.bfloat16),
        "q_norm": jax.ShapeDtypeStruct(
            (c_draft.layers, 1, c_draft.q_lora), jnp.bfloat16
        ),
        "kv_norm": jax.ShapeDtypeStruct((c_draft.layers, 1, 640), jnp.bfloat16),
        "context_norm": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "final_norm": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "mask_embed": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "rot": jax.ShapeDtypeStruct((c_draft.rope, c_draft.rope), jnp.bfloat16),
        "qkv_tile": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.hidden, 128), jnp.bfloat16
        ),
        "context_tile": jax.ShapeDtypeStruct((c_draft.hidden, 128), jnp.bfloat16),
        "q_b": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.q_lora, 512), jnp.bfloat16
        ),
        "w_uk": jax.ShapeDtypeStruct(
            (c_draft.layers, 2, c_draft.nope, c_draft.kv_lora), jnp.bfloat16
        ),
        "w_uv": jax.ShapeDtypeStruct(
            (c_draft.layers, 2, c_draft.kv_lora, c_draft.v_head), jnp.bfloat16
        ),
        "o": jax.ShapeDtypeStruct(
            (c_draft.layers, 2 * c_draft.v_head, c_draft.hidden), jnp.bfloat16
        ),
        "gate": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.hidden, 512), jnp.bfloat16
        ),
        "up": jax.ShapeDtypeStruct((c_draft.layers, c_draft.hidden, 512), jnp.bfloat16),
        "down": jax.ShapeDtypeStruct(
            (c_draft.layers, 512, c_draft.hidden), jnp.bfloat16
        ),
        "fc": jax.ShapeDtypeStruct((8960, 896), jnp.bfloat16),
        "w2t": jax.ShapeDtypeStruct((256, c_draft.vocab // 32), jnp.bfloat16),
        "w1": jax.ShapeDtypeStruct((c_draft.vocab, 256), jnp.bfloat16),
        "rope": jax.ShapeDtypeStruct((512, 2 * c_draft.rope), jnp.float32),
        "lm_head": jax.ShapeDtypeStruct(
            (c_draft.hidden, c_draft.vocab // 32), jnp.bfloat16
        ),
        "embed": jax.ShapeDtypeStruct(
            (c_draft.vocab // 32, c_draft.hidden), jnp.bfloat16
        ),
    }
    pools = tuple(
        jax.ShapeDtypeStruct((16, 128, 4, 640), jnp.float8_e4m3fn)
        for _ in range(c_draft.layers)
    )
    aux = jax.ShapeDtypeStruct(
        (len(c_draft.aux_layers), total_rows, c_draft.hidden), jnp.bfloat16
    )
    logits = jax.ShapeDtypeStruct(
        (num_requests, rows, c_draft.vocab // 32), jnp.float32
    )
    previous = jax.ShapeDtypeStruct((num_requests, 128), jnp.int32)
    position = jax.ShapeDtypeStruct((num_requests,), jnp.int32)
    forced_count = jax.ShapeDtypeStruct((num_requests,), jnp.int32)
    forced_anchor = jax.ShapeDtypeStruct((num_requests,), jnp.int32)
    tables = jax.ShapeDtypeStruct((num_requests, pages), jnp.int32)

    part_a_sharded = jax.shard_map(
        fused_draft,
        mesh=mesh_tp,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    lowered_draft = jax.jit(part_a_sharded).lower(
        weights,
        pools,
        aux,
        logits,
        previous,
        position,
        forced_count,
        forced_anchor,
        tables,
    )
    opts = xla_client.hlo.HloPrintOptions.canonical()
    opts.print_backend_config = True
    opts.print_metadata = False
    hlo_comp: Any = lowered_draft.compiler_ir(dialect="hlo")
    assert hlo_comp is not None
    canonical_hlo = _normalize_mosaic_body(hlo_comp.get_hlo_module().to_string(opts))
    assert "tpu_custom_call" in canonical_hlo
    assert len(canonical_hlo.encode("utf-8")) == _EXPECTED_VERIFY8_HLO_BYTES
    assert _sha256_words(canonical_hlo) == _EXPECTED_VERIFY8_HLO_WORDS
    custom_call_lines = "\n".join(
        line for line in canonical_hlo.splitlines() if "tpu_custom_call" in line
    )
    assert _sha256_words(custom_call_lines) == _EXPECTED_VERIFY8_CUSTOM_CALL_WORDS
