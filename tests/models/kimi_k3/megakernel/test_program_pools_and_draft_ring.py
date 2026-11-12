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
"""Unit tests for program, pools, draft_ring, and draft_mla_prefill contracts."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.kimi_k3.megakernel import (
    draft,
    draft_mla,
    draft_mla_prefill,
    draft_ring,
    pools,
    program,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import dspark

if TYPE_CHECKING:
    from pathlib import Path

    from jax.sharding import Mesh


@pytest.mark.parametrize(
    ("batch", "drafts", "attention_dp", "expected"),
    [
        (1, 0, 1, True),
        (8, 7, 1, True),
        (9, 7, 1, False),
        (16, 3, 1, True),
        (17, 3, 1, False),
        (8, 7, 2, True),
        (32, 3, 2, True),
        (33, 3, 2, False),
        (0, 3, 1, False),
    ],
)
def test_program_supports_spec_tokens_boundaries(
    batch: int,
    drafts: int,
    attention_dp: int,
    expected: bool,
) -> None:
    """Verify speculative width support matrix for DP=1 and DP=2."""
    assert program.supports_spec_tokens(batch, drafts, attention_dp) is expected


def test_program_and_pools_explicit_exports() -> None:
    """Verify explicit __all__ contracts on program and pools modules."""
    for name in program.__all__:
        assert name in program.__dict__
    for name in pools.__all__:
        assert name in pools.__dict__
    assert draft.DsparkDraft.uses_unified_pools is False
    assert draft_mla.MlaDsparkDraft.uses_unified_pools is True


def test_select_verify_kda_rows_explicit_parameters() -> None:
    """Verify select_verify_kda_rows with explicit DP1 parameters and null slots."""
    slots = jnp.asarray(
        [
            [2, 2, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    ).reshape(-1)
    accepted = jnp.asarray([1, 0], dtype=jnp.int32)
    sources, targets = pools.select_verify_kda_rows(
        slots,
        accepted,
        requests=2,
        rows_per=4,
        attention_dp=1,
        bank_slots=64,
    )
    assert sources.tolist() == [1, 7]
    assert targets.tolist() == [2, 0]


def test_draft_mla_prefill_set_tensor_in_place() -> None:
    """Verify _set_tensor_in_place rebinds storage via InPlaceSetTensor."""
    target = torch.zeros((2, 3), dtype=torch.float32)
    source = torch.full((2, 3), 7.0, dtype=torch.float32)
    draft_mla_prefill._set_tensor_in_place(target, source)
    assert torch.equal(target, source)


@dataclasses.dataclass
class _FakeDraftConfigLoader:
    """Stub DraftConfig loader returning default DraftConfig."""

    @staticmethod
    def from_checkpoint(_directory: Path) -> dspark.DraftConfig:
        """Return default DraftConfig matching AUX_LAYERS."""
        return dspark.DraftConfig()


@dataclasses.dataclass
class _FakeDsparkKernel:
    """Minimal dspark module double for PrefillRing.load_weights."""

    KERNEL_SHARD_FORMAT: int = dspark.KERNEL_SHARD_FORMAT
    DraftConfig: type[_FakeDraftConfigLoader] = _FakeDraftConfigLoader

    def ring_weights_from_shards(self, *args: object) -> dict[str, object]:
        """Unused placeholder for register_op registration."""
        _ = args
        return {}


def test_prefill_ring_load_weights_mmap_roundtrip(tmp_path: Path) -> None:
    """Verify PrefillRing.load_weights reads shard payloads via mmap context."""
    ckpt = tmp_path / "ckpt_rev12345"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 7168,
                "num_hidden_layers": 1,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "head_dim": 128,
                "vocab_size": 163840,
            }
        )
    )
    shard_dir = tmp_path / "dspark-ckpt_rev"
    shard_dir.mkdir()
    arrays = {
        name: np.full((1, 2), idx + 1, dtype=np.float32)
        for idx, name in enumerate(draft_ring.WEIGHT_ARRAYS)
    }
    offset = 0
    specs = {}
    with (shard_dir / "rank00.bin").open("wb") as handle:
        for name, arr in arrays.items():
            raw_bytes = arr.tobytes()
            handle.write(raw_bytes)
            specs[name] = {
                "dtype": "float32",
                "shape": list(arr.shape),
                "offset": offset,
            }
            offset += len(raw_bytes)
    (shard_dir / "index.json").write_text(
        json.dumps({"format": dspark.KERNEL_SHARD_FORMAT, "arrays": specs})
    )

    with patch.object(
        draft_ring,
        "kernel_modules",
        return_value=(_FakeDsparkKernel(),),
    ):
        ring = draft_ring.PrefillRing(
            mesh=cast("Mesh", object()),
            rank=0,
            axis="k3_sp",
            checkpoint=ckpt,
            presharded=tmp_path,
        )
    uploaded: list[np.ndarray] = []
    fake_weights = [torch.ones((1, 2), dtype=torch.float32) for _ in range(4)]

    def _fake_upload(arr: np.ndarray) -> np.ndarray:
        uploaded.append(arr.copy())
        return arr

    with (
        patch.object(draft_ring, "upload", side_effect=_fake_upload),
        patch.object(
            draft_ring,
            "register_op",
            return_value=lambda *_shards: tuple(fake_weights),
        ),
        patch.object(draft_ring, "materialize"),
    ):
        ring.load_weights()

    assert len(uploaded) == len(draft_ring.WEIGHT_ARRAYS)
    for idx, name in enumerate(draft_ring.WEIGHT_ARRAYS):
        np.testing.assert_array_equal(uploaded[idx], arrays[name])

    (shard_dir / "index.json").write_text(json.dumps({"format": 999, "arrays": {}}))
    with pytest.raises(ValueError, match="not a DSpark kernel shard set"):
        ring.load_weights()
