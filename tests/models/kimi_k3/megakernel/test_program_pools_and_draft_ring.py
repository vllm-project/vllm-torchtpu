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
import importlib
import json
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu.models.kimi_k3.megakernel import (
    decode,
    decode_ops,
    draft,
    draft_mla,
    draft_mla_prefill,
    draft_ring,
    pools,
    prefill,
    program,
)
from vllm_torchtpu.models.kimi_k3.megakernel import layout as k3_layout
from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_SP,
    AXIS_TP,
    EXPERT_FAMILIES,
)
from vllm_torchtpu.models.kimi_k3.megakernel.common.protocols import (
    set_tensor_in_place,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import dspark
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import bridge
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.program import KVGroup

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


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
    *,
    expected: bool,
) -> None:
    """Verify speculative width support matrix for DP=1 and DP=2."""
    assert program.supports_spec_tokens(batch, drafts, attention_dp) is expected


def test_program_and_pools_explicit_exports() -> None:
    """Verify explicit __all__ contracts on program and pools modules."""
    for name in program.__all__:
        assert hasattr(program, name)
    for name in pools.__all__:
        assert hasattr(pools, name)
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
        pools.VerifyKdaSelectionConfig(
            requests=2,
            rows_per=4,
            attention_dp=1,
            bank_slots=64,
        ),
    )
    assert sources.tolist() == [1, 7]
    assert targets.tolist() == [2, 0]


def test_protocols_set_tensor_in_place() -> None:
    """Verify set_tensor_in_place rebinds storage via InPlaceSetTensor."""
    target = torch.zeros((2, 3), dtype=torch.float32)
    source = torch.full((2, 3), 7.0, dtype=torch.float32)
    set_tensor_in_place(target, source)
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
    DsparkRingWeightShards: type[dspark.DsparkRingWeightShards] = (
        dspark.DsparkRingWeightShards
    )

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
            mesh=_cpu_mesh(AXIS_SP),
            rank=0,
            axis="k3_sp",
            checkpoint=ckpt,
            presharded=tmp_path,
        )
    uploaded: list[np.ndarray] = []
    fake_weights = [torch.ones((1, 2), dtype=torch.float32) for _ in range(4)]

    def _fake_upload(arr: np.ndarray) -> np.ndarray:
        """Record a copy of each uploaded shard array."""
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


def test_bridge_synchronize_tpu_forwards_wait_flag_and_exports() -> None:
    """Verify bridge.synchronize_tpu delegates wait flag and is listed in __all__."""
    assert {"RegisteredOp", "synchronize_tpu"}.issubset(set(bridge.__all__))
    calls: list[tuple[list[torch.Tensor], bool]] = []
    sync_mod = importlib.import_module("torch_tpu._internal.sync")

    def _record_sync(tensors: Sequence[torch.Tensor], *, wait: bool = True) -> None:
        """Record synchronized tensor lists and wait flags."""
        calls.append((list(tensors), wait))

    first = torch.zeros((1,), dtype=torch.int32)
    second = torch.ones((1,), dtype=torch.int32)
    with patch.object(sync_mod, "synchronize", side_effect=_record_sync):
        bridge.synchronize_tpu((first, second))
        bridge.synchronize_tpu((second,), wait=True)

    assert calls == [([first, second], False), ([second], True)]


def _cpu_mesh(axis: str) -> Mesh:
    """Build a real 1-device CPU JAX Mesh on ``axis``."""
    return Mesh(np.array([jax.devices()[0]]), (axis,))


def _write_mla_draft_checkpoint(
    directory: Path,
    *,
    rms_norm_eps: float = 1e-6,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
) -> Path:
    """Create a valid minimal MLA DSpark checkpoint directory under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "k3_dspark",
                "hidden_size": 7168,
                "num_hidden_layers": 5,
                "num_attention_heads": 64,
                "q_lora_rank": 1536,
                "kv_lora_rank": kv_lora_rank,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": qk_rope_head_dim,
                "v_head_dim": 128,
                "intermediate_size": 18432,
                "vocab_size": 163840,
                "target_layer_ids": [2, 23, 47, 71, 89],
                "mask_token_id": 163839,
                "rms_norm_eps": rms_norm_eps,
                "markov_rank": 128,
                "block_size": 7,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": 10000.0,
                    "factor": 32.0,
                    "original_max_position_embeddings": 4096,
                },
            }
        ),
        encoding="utf-8",
    )
    return directory


def _write_ring_draft_checkpoint(directory: Path) -> Path:
    """Create a valid minimal ring DSpark checkpoint directory under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "aux_hidden_state_layer_ids": list(draft_ring.AUX_LAYERS),
                "mask_token_id": 163839,
                "markov_rank": 128,
                "transformer_layer_config": {
                    "hidden_size": 7168,
                    "num_hidden_layers": 5,
                    "num_attention_heads": 64,
                    "num_key_value_heads": 32,
                    "head_dim": 128,
                    "intermediate_size": 18432,
                    "vocab_size": 163840,
                    "sliding_window": 2048,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {"rope_theta": 10000.0},
                },
            }
        ),
        encoding="utf-8",
    )
    return directory


def _make_sharded_op_spy() -> tuple[
    list[tuple[str, str | None]],
    Callable[..., Callable[..., tuple[torch.Tensor, ...]]],
]:
    """Return a registration log and stub ``_register_sharded_op``."""
    registered: list[tuple[str, str | None]] = []

    def _spy(
        name: str,
        _fn: object,
        _mesh: Mesh,
        spec: bridge.OpRegistrationSpec,
        _configure_callable: object = None,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Record ``(name, export_key)`` and return a unique callable."""
        index = len(registered)
        registered.append((name, spec.export_key))

        def _runner(*_args: object) -> tuple[torch.Tensor, ...]:
            """Return placeholder output tensors matching ``spec.out_specs``."""
            return tuple(
                torch.full((1,), index, dtype=torch.int32) for _ in spec.out_specs
            )

        return _runner

    return registered, _spy


_OPS_PER_WINDOW = 3
_TWO_WINDOW_BUILDS = 6
_PREFILL_EXPERT_BANKS = 13


@dataclasses.dataclass
class _FakePrefillKernels:
    """Minimal prefill_megakernel double for build_window_ops unit testing."""

    MOE_FAMILIES: tuple[str, ...] = EXPERT_FAMILIES

    def make_embedding_sp(self, *_args: object, **_kwargs: object) -> object:
        """Return stub embedding callable."""
        return lambda *_a, **_kw: jnp.zeros((1,))

    def make_prefill_stack_sp_indexed(
        self, *_args: object, **_kwargs: object
    ) -> object:
        """Return stub first-stack callable."""
        return lambda *_a, **_kw: (jnp.zeros((1,)),) * 5

    def make_regular_group8_sp_indexed(
        self, *_args: object, **_kwargs: object
    ) -> object:
        """Return stub regular-group callable."""
        return lambda *_a, **_kw: (jnp.zeros((1,)),) * 6

    def make_final_mla_sp_indexed(self, *_args: object, **_kwargs: object) -> object:
        """Return stub final-MLA callable."""
        return lambda *_a, **_kw: (
            jnp.zeros((1,)),
            jnp.zeros((1,)),
            (jnp.zeros((1,)),),
        )

    def make_prefill_logits_sp(self, *_args: object, **_kwargs: object) -> object:
        """Return stub logits callable."""
        return lambda *_a, **_kw: jnp.zeros((1,))


@dataclasses.dataclass
class _FakeRankLoader:
    """Minimal rank_loader double for PrefillProgram tests."""

    PREFILL_IO: tuple[str, ...] = (
        "embedding",
        "output_res",
        "final_norm",
        "lm_head",
    )
    PREFILL_COMMON: tuple[str, ...] = ("common_w",)
    PREFILL_KDA: tuple[str, ...] = ("kda_w",)
    PREFILL_MLA: tuple[str, ...] = ("mla_w",)

    def iter_prefill_rank(self, *_args: object, **_kwargs: object) -> object:
        """Return an empty iterator overridden by fake stream_to_device."""
        return iter(())


@dataclasses.dataclass
class _WarmupDraftState:
    """Minimal draft state carrying a replicated draft vector."""

    vector: torch.Tensor


class _WarmupDraftDouble:
    """Minimal draft double driving KimiDecoder.warmup op registrations."""

    uses_unified_pools: bool = False

    def __init__(self) -> None:
        """Initialize stub draft config and placeholder weights."""
        self.weights: dict[str, torch.Tensor] = {"stub": torch.zeros(1)}
        self.config = dspark.DraftConfig()
        self.aux_layers: tuple[int, ...] = ()

    def load_weights(self) -> float:
        """No-op weight loader."""
        return 0.0

    def new_state(self, _zero: torch.Tensor) -> _WarmupDraftState:
        """Return a fresh stub draft state."""
        return _WarmupDraftState(vector=torch.zeros(1, dtype=torch.int32))

    def from_ring(
        self,
        _cache: torch.Tensor,
        _positions: torch.Tensor,
        _zero: torch.Tensor,
    ) -> _WarmupDraftState:
        """Return a stub draft state from an offered ring."""
        return _WarmupDraftState(vector=torch.zeros(1, dtype=torch.int32))

    def synthetic_rates_for(self, _drafts: int) -> tuple[float, ...] | None:
        """Return no synthetic acceptance rates."""
        return None

    def step_single(
        self,
        state: _WarmupDraftState,
        inputs: decode_ops.DraftStepInputs,
    ) -> tuple[_WarmupDraftState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return placeholder single-step draft outputs."""
        return (
            state,
            torch.zeros((1, 8), dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            inputs.control_state,
        )

    def step_batched(
        self,
        states: Sequence[_WarmupDraftState],
        inputs: decode_ops.BatchedDraftStepInputs,
    ) -> tuple[list[_WarmupDraftState], torch.Tensor, torch.Tensor]:
        """Return placeholder batched draft outputs."""
        return (
            list(states),
            torch.zeros((len(states), 8), dtype=torch.int32),
            inputs.control_state,
        )

    def compilation_calls(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> list[decode.CompileCall]:
        """Return no extra draft compile calls."""
        return []


@dataclasses.dataclass
class _WarmupImporterDouble:
    """Minimal KDA importer double for KimiDecoder.bind_kv_caches warmup."""

    raw_pools: list[torch.Tensor]
    mla_raw: list[int] = dataclasses.field(
        default_factory=lambda: [0] * pools.MLA_LAYERS
    )
    mla_layer_names: tuple[str, ...] = ("model.layers.0.self_attn",)
    kda_layer_names: tuple[str, ...] = ("model.layers.1.self_attn",)

    def bind(self, _caches: object, _dtype: str) -> int:
        """Return fixed block count."""
        return 4

    def import_state(
        self,
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        _blocks: torch.Tensor,
        _slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unchanged state tensors."""
        return conv, recurrent

    def import_ring(
        self,
        _draft_config: object,
        _blocks: torch.Tensor,
        _computed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return placeholder ring cache and positions."""
        return torch.zeros(1), torch.zeros(1, dtype=torch.int32)


def test_read_rings_op_keys_on_draft_config() -> None:
    """Verify PoolWriter.read_rings_op distinguishes different draft configs."""
    mesh = _cpu_mesh(AXIS_SP)
    raw_pool = torch.zeros((2, *k3_layout.LAYOUT.block_shape), dtype=torch.bfloat16)
    writer = prefill.PoolWriter(
        mesh,
        0,
        [raw_pool],
        prefill.BoundPoolIndices(
            mla_raw=[0] * pools.MLA_LAYERS,
            kda_raw=[0] * pools.KDA_LAYERS,
            state_perm=None,
        ),
    )
    registered, spy = _make_sharded_op_spy()
    cfg_a = dspark.DraftConfig(window=2048)
    cfg_b = dspark.DraftConfig(window=1024)

    with patch.object(bridge, "_register_sharded_op", side_effect=spy):
        op_a = writer.read_rings_op(cfg_a, 2)
        op_a_cached = writer.read_rings_op(cfg_a, 2)
        op_b = writer.read_rings_op(cfg_b, 2)

    expected_count = 2
    assert op_a is op_a_cached
    assert op_a is not op_b
    assert len(registered) == expected_count
    assert registered[0][0] != registered[1][0]
    assert registered[0][1] != registered[1][1]


def test_mla_dspark_draft_bind_pools_and_owner_table_cache_keys(tmp_path: Path) -> None:
    """Verify MlaDsparkDraft.bind_pools validates/clears ops and keys on owner_table."""
    ckpt = _write_mla_draft_checkpoint(tmp_path / "mla_rev12345")
    mesh = _cpu_mesh(AXIS_TP)
    with patch.object(draft_mla, "kernel_mesh", return_value=(mesh, 0)):
        drafter = draft_mla.MlaDsparkDraft(4096, str(ckpt), tmp_path)
    drafter.weights = {name: torch.zeros(1) for name in drafter.names}

    template = torch.zeros((2, 16, 16, 640), dtype=torch.bfloat16)
    valid_pools = [torch.zeros_like(template) for _ in range(drafter.config.layers)]

    with pytest.raises(RuntimeError, match="expected 5 draft pools"):
        drafter.bind_pools(template, None, valid_pools[:-1])
    with pytest.raises(RuntimeError, match="expected 5 draft pools"):
        drafter.bind_pools(
            template,
            None,
            [*valid_pools[:-1], torch.zeros((2, 8, 16, 640), dtype=torch.bfloat16)],
        )
    with pytest.raises(RuntimeError, match="expected 5 draft pools"):
        drafter.bind_pools(
            template,
            None,
            [*valid_pools[:-1], torch.zeros_like(template, dtype=torch.float16)],
        )

    control = decode_ops.Control(slots=5, rows=2, pages=4)
    registered, spy = _make_sharded_op_spy()
    owner_a = tuple(range(program.TP_SIZE))
    owner_b = tuple(reversed(range(program.TP_SIZE)))

    with (
        patch.object(bridge, "_register_sharded_op", side_effect=spy),
        patch.object(
            drafter.kernel,
            "fused_draft_local",
            return_value=lambda *_a, **_kw: (
                jnp.zeros((1, 128)),
                jnp.zeros((5, 8, 640)),
            ),
        ),
    ):
        drafter.bind_pools(template, owner_a, valid_pools)
        calls_a = drafter.compilation_calls(control, [(1, 4)], spec_alone=True)
        calls_a_cached = drafter.compilation_calls(control, [(1, 4)], spec_alone=True)
        assert calls_a[0].op is calls_a_cached[0].op
        assert calls_a[1].op is calls_a_cached[1].op

        drafter.bind_pools(template, owner_b, valid_pools)
        calls_b = drafter.compilation_calls(control, [(1, 4)], spec_alone=True)

        # Even if owner_table changes without rebinding, cache keys distinguish it.
        drafter.owner_table = owner_a
        calls_a2 = drafter.compilation_calls(control, [(1, 4)], spec_alone=True)

    expected_registrations = 6
    assert len(registered) == expected_registrations
    assert calls_a[0].op is not calls_b[0].op
    assert calls_a[1].op is not calls_b[1].op
    assert calls_a2[0].op is not calls_b[0].op
    assert calls_a2[1].op is not calls_b[1].op
    assert registered[0][0] != registered[2][0]
    assert registered[1][0] != registered[3][0]

    # Rebinding without explicit pools reallocates when template dtype changes.
    fp16_template = torch.zeros_like(template, dtype=torch.float16)
    drafter.bind_pools(fp16_template, owner_a)
    assert len(drafter.pools) == drafter.config.layers
    assert all(p.dtype == torch.float16 for p in drafter.pools)


def test_prefill_draft_latents_bind_and_dtype_cache_keys(tmp_path: Path) -> None:
    """Verify PrefillDraftLatents.bind validates pools/clears ops and keys on dtype."""
    ckpt = _write_mla_draft_checkpoint(tmp_path / "mla_prefill_rev12345")
    mesh = _cpu_mesh(AXIS_SP)
    latents = draft_mla_prefill.PrefillDraftLatents(
        mesh,
        0,
        AXIS_SP,
        ckpt,
        tmp_path,
        4096,
    )
    bf16_pools = [
        torch.zeros((2, 16, 16, 640), dtype=torch.bfloat16)
        for _ in range(latents.config.layers)
    ]
    fp16_pools = [
        torch.zeros((2, 16, 16, 640), dtype=torch.float16)
        for _ in range(latents.config.layers)
    ]

    with pytest.raises(RuntimeError, match="draft KV cache layers"):
        latents.bind(bf16_pools[:-1])
    with pytest.raises(RuntimeError, match="shape"):
        latents.bind(
            [*bf16_pools[:-1], torch.zeros((2, 8, 16, 640), dtype=torch.bfloat16)]
        )
    with pytest.raises(RuntimeError, match="dtype"):
        latents.bind([*bf16_pools[:-1], fp16_pools[-1]])

    registered, spy = _make_sharded_op_spy()
    block_shape = (16, 16, 640)
    latents_in = torch.zeros((1, latents.config.layers, 256, 576), dtype=torch.bfloat16)
    table = torch.zeros((1, 1), dtype=torch.int32)
    with (
        patch.object(bridge, "_register_sharded_op", side_effect=spy),
        patch.object(draft_mla_prefill, "set_tensor_in_place", lambda _a, _b: None),
    ):
        latents.bind(bf16_pools)
        latents.write(latents_in, table, block_shape)
        latents.write(latents_in, table, block_shape)
        assert len(registered) == 1

        # Rebinding clears cached ops even for matching dtype.
        latents.bind([torch.zeros_like(p) for p in bf16_pools])
        latents.write(latents_in, table, block_shape)
        rebind_count = 2
        assert len(registered) == rebind_count

        latents.bind(fp16_pools)
        latents.write(latents_in, table, block_shape)

        # Even if pools change dtype directly, the write cache key misses.
        latents.pools = bf16_pools
        latents.write(latents_in, table, block_shape)

    expected_registrations = 4
    assert len(registered) == expected_registrations
    assert registered[0][0] != registered[2][0]
    assert registered[0][1] != registered[2][1]


def test_prefill_draft_latents_op_keys_on_full_mla_draft_config(
    tmp_path: Path,
) -> None:
    """Verify PrefillDraftLatents keys op names and export keys on eps/kv_lora/rope."""
    mesh = _cpu_mesh(AXIS_SP)
    variants = (
        ("base", _write_mla_draft_checkpoint(tmp_path / "mla_base_rev12345")),
        (
            "eps",
            _write_mla_draft_checkpoint(
                tmp_path / "mla_eps_rev12345",
                rms_norm_eps=1e-5,
            ),
        ),
        (
            "split",
            _write_mla_draft_checkpoint(
                tmp_path / "mla_split_rev12345",
                kv_lora_rank=448,
                qk_rope_head_dim=128,
            ),
        ),
    )
    registered, spy = _make_sharded_op_spy()
    placeholder = torch.zeros((1,), dtype=torch.bfloat16)
    positions = np.arange(4, dtype=np.int32)
    order = np.arange(4, dtype=np.int32)
    with patch.object(bridge, "_register_sharded_op", side_effect=spy):
        for _label, ckpt in variants:
            latents = draft_mla_prefill.PrefillDraftLatents(
                mesh,
                0,
                AXIS_SP,
                ckpt,
                tmp_path,
                4096,
            )
            latents.weights = dict.fromkeys(
                (
                    "context_proj",
                    "kv_a",
                    "kv_norm",
                    "context_norm",
                    "rope",
                ),
                placeholder,
            )
            latents.latents(
                512,
                placeholder,
                placeholder,
                placeholder,
                positions,
                order,
            )
    assert len(registered) == len(variants)
    assert len({name for name, _ in registered}) == len(variants)
    assert len({export_key for _, export_key in registered}) == len(variants)


def test_prefill_program_window_ops_invalidates_on_expert_banks(
    tmp_path: Path,
) -> None:
    """Verify PrefillProgram.build_window_ops keys on banks and clears on load."""
    mesh = _cpu_mesh(AXIS_SP)
    registered, spy = _make_sharded_op_spy()
    with (
        patch.object(
            prefill,
            "kernel_modules",
            return_value=(_FakePrefillKernels(), _FakeRankLoader()),
        ),
        patch.object(prefill, "kernel_mesh", return_value=(mesh, 0)),
        patch.object(bridge, "_register_sharded_op", side_effect=spy),
    ):
        prog = prefill.PrefillProgram((2048,), 4096, 4)
        prog.weights = {"io": {"embedding": torch.zeros(1)}}
        ops_global = prog.build_window_ops(2048)
        assert prog.build_window_ops(2048) is ops_global
        assert len(registered) == _OPS_PER_WINDOW
        assert all("moe-global" in (ek or "") for _, ek in registered)

        def _fake_stream(
            _iterator: object,
            store: Callable[[str, str, torch.Tensor], None],
        ) -> dict[str, float]:
            """Populate all required weight groups and expert banks on CPU."""
            for group, names in prog.names.items():
                if group == "moe":
                    continue
                for name in names:
                    store(group, name, torch.zeros(1))
            for bank_idx in range(_PREFILL_EXPERT_BANKS):
                for family in EXPERT_FAMILIES:
                    store("moe", f"{family}#bank{bank_idx}", torch.zeros(1))
            return {"wall": 0.1, "wait": 0.0, "upload": 0.0, "gib": 0.1}

        with (
            patch.object(prefill, "stream_to_device", side_effect=_fake_stream),
            patch.object(prefill, "materialize"),
            patch.object(prefill, "report_load_stats"),
        ):
            prog.load_weights(tmp_path)

        ops_banked = prog.build_window_ops(2048)
        assert ops_banked is not ops_global
        assert prog.build_window_ops(2048) is ops_banked
        assert len(registered) == _TWO_WINDOW_BUILDS
        assert all("moe-bank8" in (ek or "") for _, ek in registered[_OPS_PER_WINDOW:])
        assert {name for name, _ in registered[:_OPS_PER_WINDOW]}.isdisjoint(
            {name for name, _ in registered[_OPS_PER_WINDOW:]}
        )


def _assert_decode_op_names_disambiguate(
    tmp_path: Path,
    raw_pool_bf16: torch.Tensor,
    registered: list[tuple[str, str | None]],
) -> None:
    """Assert DecodeOps and KimiDecoder op names disambiguate across Control layouts."""
    ops = decode_ops.DecodeOps(tmp_path, num_slots=5, batch_sizes=(1, 2))
    ops.names = ("w0",)
    ops.weights = {"w0": torch.zeros(1)}
    ops.slots = [torch.zeros(1), torch.zeros(1)]
    ops.use_unified_pools([raw_pool_bf16], [0] * pools.MLA_LAYERS)
    ctrl_a = decode_ops.Control(slots=5, rows=2, pages=4)
    ctrl_b = decode_ops.Control(slots=5, rows=2, pages=8)

    start_a = len(registered)
    ops.compilation_calls(
        ctrl_a, 5, (3,), spec_alone=True, spec_shapes=((2, 4),), masked_ops=True
    )
    decode_names_a = {name for name, _ in registered[start_a:]}
    start_b = len(registered)
    ops.compilation_calls(
        ctrl_b, 5, (3,), spec_alone=True, spec_shapes=((2, 4),), masked_ops=True
    )
    decode_names_b = {name for name, _ in registered[start_b:]}
    assert decode_names_a
    assert decode_names_a.isdisjoint(decode_names_b)

    decode_cfg = MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=1,
        block_size=128,
        max_num_seqs=2,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        num_speculative_tokens=7,
    )
    importer = _WarmupImporterDouble(raw_pools=[raw_pool_bf16])
    draft_stub = _WarmupDraftDouble()

    def _noop(*_args: object, **_kwargs: object) -> None:
        """No-op warmup hook."""

    def _run_decoder_warmup(pages: int) -> set[str]:
        """Warm one KimiDecoder for ``pages`` and return newly registered op names."""
        before = len(registered)
        warm_ops = decode_ops.DecodeOps(tmp_path, num_slots=5, batch_sizes=(1, 2))
        warm_ops.names = ("w0",)
        warm_ops.weights = {"w0": torch.zeros(1)}
        warm_ops.slots = [torch.zeros(1), torch.zeros(1)]
        warm_ops.cached_op(("kda_importer", None), lambda: importer)
        decoder = decode.KimiDecoder(decode_cfg, warm_ops, draft_stub, tp_rank=0)
        with (
            patch.object(
                warm_ops,
                "plain_step",
                side_effect=lambda state, *_a, **_kw: (torch.zeros(1), state),
            ),
            patch.object(
                warm_ops,
                "verify_packed",
                return_value=(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
            ),
            patch.object(
                warm_ops,
                "verify_batched",
                return_value=(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
            ),
            patch.object(warm_ops, "compilation_calls", return_value=[]),
        ):
            decoder.bind_kv_caches(
                {},
                [
                    KVGroup(
                        layer_names=("model.layers.0.self_attn",),
                        block_size=128,
                        table_width=pages,
                    ),
                    KVGroup(
                        layer_names=("model.layers.1.self_attn",),
                        block_size=128,
                        table_width=pages,
                    ),
                ],
            )
            decoder.warmup()
        return {name for name, _ in registered[before:]}

    with (
        patch.object(decode, "tp_rank_permutation", return_value=None),
        patch.object(decode, "prepare_distributed", _noop),
        patch.object(decode, "compile_ops", _noop),
        patch.object(decode, "materialize", _noop),
        patch.object(decode, "synchronize_tpu", _noop),
    ):
        dec_names_a = _run_decoder_warmup(4)
        dec_names_b = _run_decoder_warmup(8)
    expected_decoder_ops = 3
    assert len(dec_names_a) == expected_decoder_ops
    assert len(dec_names_b) == expected_decoder_ops
    assert dec_names_a.isdisjoint(dec_names_b)


def _assert_pool_writer_op_names_disambiguate(
    mesh_sp: Mesh,
    raw_pool_bf16: torch.Tensor,
    raw_pool_fp8: torch.Tensor,
    registered: list[tuple[str, str | None]],
) -> None:
    """Assert all 5 PoolWriter op builders disambiguate across layout/perm/config."""
    writer_a = prefill.PoolWriter(
        mesh_sp,
        0,
        [raw_pool_bf16],
        prefill.BoundPoolIndices([0] * pools.MLA_LAYERS, [0] * pools.KDA_LAYERS, None),
    )
    writer_b = prefill.PoolWriter(
        mesh_sp,
        0,
        [raw_pool_fp8],
        prefill.BoundPoolIndices(
            [0] * pools.MLA_LAYERS,
            [0] * pools.KDA_LAYERS,
            [(i, (i + 1) % program.TP_SIZE) for i in range(program.TP_SIZE)],
        ),
    )
    cfg_1 = dspark.DraftConfig(window=2048)
    cfg_2 = dspark.DraftConfig(window=1024)
    writer_a.copy_op()
    writer_a.write_op(256, 1, 1)
    writer_a.rings_op(1)
    writer_a.read_rings_op(cfg_1, 1)
    writer_a.read_states_op(1)
    pw_names_a = [name for name, _ in registered[-5:]]

    writer_b.copy_op()
    writer_b.write_op(256, 1, 1)
    writer_b.rings_op(1)
    writer_b.read_rings_op(cfg_2, 1)
    writer_b.read_states_op(1)
    pw_names_b = [name for name, _ in registered[-5:]]
    for name_a, name_b in zip(pw_names_a, pw_names_b, strict=True):
        assert name_a != name_b


def _assert_dspark_and_ring_op_names_disambiguate(
    mesh_sp: Mesh,
    ring_ckpt: Path,
    tmp_path: Path,
    registered: list[tuple[str, str | None]],
) -> None:
    """Assert DsparkDraft (3 sites) and PrefillRing (1 site) disambiguate op names."""
    ctrl = decode_ops.Control(slots=5, rows=2, pages=4)
    dspark_a = draft.DsparkDraft(4096, str(ring_ckpt), tmp_path)
    dspark_b = draft.DsparkDraft(4096, str(ring_ckpt), tmp_path)
    dspark_b.config = dspark.DraftConfig(window=1024)
    for drafter in (dspark_a, dspark_b):
        drafter.names = ("w0",)
        drafter.weights = {"w0": torch.zeros(1)}
    batch_inputs = decode_ops.BatchedDraftStepInputs(
        aux=torch.zeros(1),
        logits=torch.zeros(1),
        ids=torch.zeros(1),
        control_state=torch.zeros(1),
        decisions=torch.zeros(1),
        control=ctrl,
        rows_per=4,
    )
    large_states = [
        draft.DraftState(torch.zeros(1), torch.zeros(1), torch.zeros(1))
        for _ in range(16)
    ]
    with patch.object(
        dspark_a.dspark,
        "make_fused_draft",
        return_value=lambda *_a, **_kw: (jnp.zeros((1,)),) * 3,
    ):
        start_a = len(registered)
        dspark_a.compilation_calls(ctrl, [(2, 4)], spec_alone=True)
        dspark_a.step_batched(large_states, batch_inputs)
        dr_names_a = [name for name, _ in registered[start_a:]]

        start_b = len(registered)
        dspark_b.compilation_calls(ctrl, [(2, 4)], spec_alone=True)
        dspark_b.step_batched(large_states, batch_inputs)
        dr_names_b = [name for name, _ in registered[start_b:]]
    assert any("metadata" in name for name in dr_names_a)
    for name_a, name_b in zip(dr_names_a, dr_names_b, strict=True):
        assert name_a != name_b

    ring_a = draft_ring.PrefillRing(mesh_sp, 0, AXIS_SP, ring_ckpt, tmp_path)
    ring_b = draft_ring.PrefillRing(mesh_sp, 0, AXIS_SP, ring_ckpt, tmp_path)
    ring_b.config = dspark.DraftConfig(window=1024)
    ring_a.update_op(256, 1)
    ring_b.update_op(256, 1)
    assert registered[-2][0] != registered[-1][0]


def _assert_mla_draft_op_names_disambiguate(
    mesh_sp: Mesh,
    mla_ckpt: Path,
    tmp_path: Path,
    registered: list[tuple[str, str | None]],
) -> None:
    """Assert MlaDsparkDraft attach_target and PrefillDraftLatents disambiguate."""
    mla_a = draft_mla.MlaDsparkDraft(4096, str(mla_ckpt), tmp_path)
    mla_b = draft_mla.MlaDsparkDraft(4096, str(mla_ckpt), tmp_path)
    mla_b.config = dataclasses.replace(mla_b.config, mask_token=163838)
    mla_a.attach_target({"lm_head": torch.zeros(1), "embedding": torch.zeros(1)})
    mla_b.attach_target({"lm_head": torch.zeros(1), "embedding": torch.zeros(1)})
    assert registered[-2][0] != registered[-1][0]

    lat_a = draft_mla_prefill.PrefillDraftLatents(
        mesh_sp, 0, AXIS_SP, mla_ckpt, tmp_path, 4096
    )
    lat_b = draft_mla_prefill.PrefillDraftLatents(
        mesh_sp, 0, AXIS_SP, mla_ckpt, tmp_path, 8192
    )
    lat_b.config = dataclasses.replace(lat_b.config, layers=4)
    weight_names = ("context_proj", "kv_a", "kv_norm", "context_norm", "rope")
    for lat in (lat_a, lat_b):
        lat.weights = {name: torch.zeros(1) for name in weight_names}
    lat_a.bind([torch.zeros((2, 16, 16, 640), dtype=torch.bfloat16)] * 5)
    lat_b.bind([torch.zeros((2, 16, 16, 640), dtype=torch.float16)] * 4)
    positions = np.zeros(256, dtype=np.int32)
    order = np.arange(256, dtype=np.int32)
    dummy = torch.zeros(1)
    table = torch.zeros((1, 1), dtype=torch.int32)
    with patch.object(draft_mla_prefill, "set_tensor_in_place", lambda _a, _b: None):
        lat_a.copy_blocks([(0, 0)], 8)
        lat_a.latents(256, dummy, dummy, dummy, positions, order)
        lat_a.write(torch.zeros((1, 5, 256, 576)), table, (16, 16, 640))
        lat_names_a = [name for name, _ in registered[-3:]]

        lat_b.copy_blocks([(0, 0)], 8)
        lat_b.latents(256, dummy, dummy, dummy, positions, order)
        lat_b.write(torch.zeros((1, 4, 256, 576)), table, (16, 16, 640))
        lat_names_b = [name for name, _ in registered[-3:]]
    for name_a, name_b in zip(lat_names_a, lat_names_b, strict=True):
        assert name_a != name_b


def test_decode_and_pool_writer_op_names_disambiguate_configs(
    tmp_path: Path,
) -> None:
    """Verify op_key_suffix disambiguates op names across all program builders."""
    suffix_a = bridge.op_key_suffix(("cfg", 1))
    suffix_b = bridge.op_key_suffix(("cfg", 2))
    expected_suffix_len = 8
    assert len(suffix_a) == expected_suffix_len
    assert suffix_a == bridge.op_key_suffix(("cfg", 1))
    assert suffix_a != suffix_b

    mesh_tp = _cpu_mesh(AXIS_TP)
    mesh_sp = _cpu_mesh(AXIS_SP)
    registered, spy = _make_sharded_op_spy()
    ring_ckpt = _write_ring_draft_checkpoint(tmp_path / "ring_rev12345")
    mla_ckpt = _write_mla_draft_checkpoint(tmp_path / "mla_rev67890")
    raw_pool_bf16 = torch.zeros(
        (2, *k3_layout.LAYOUT.block_shape), dtype=torch.bfloat16
    )
    raw_pool_fp8 = torch.zeros(
        (2, *k3_layout.FP8_LAYOUT.block_shape), dtype=torch.float8_e4m3fn
    )

    with (
        patch.object(bridge, "_register_sharded_op", side_effect=spy),
        patch.object(bridge, "kernel_mesh", return_value=(mesh_tp, 0)),
        patch.object(decode_ops, "kernel_mesh", return_value=(mesh_tp, 0)),
        patch.object(draft, "kernel_mesh", return_value=(mesh_tp, 0)),
        patch.object(draft_mla, "kernel_mesh", return_value=(mesh_tp, 0)),
    ):
        _assert_decode_op_names_disambiguate(tmp_path, raw_pool_bf16, registered)
        _assert_pool_writer_op_names_disambiguate(
            mesh_sp, raw_pool_bf16, raw_pool_fp8, registered
        )
        _assert_dspark_and_ring_op_names_disambiguate(
            mesh_sp, ring_ckpt, tmp_path, registered
        )
        _assert_mla_draft_op_names_disambiguate(mesh_sp, mla_ckpt, tmp_path, registered)


def test_program_modules_reject_nonlocal_uris(tmp_path: Path) -> None:
    """Verify all program entrypoints reject remote and collapsed URIs upfront."""
    ring_ckpt = _write_ring_draft_checkpoint(tmp_path / "ring_uri_rev")
    mla_ckpt = _write_mla_draft_checkpoint(tmp_path / "mla_uri_rev")
    mesh_tp = _cpu_mesh(AXIS_TP)
    mesh_sp = _cpu_mesh(AXIS_SP)
    decode_cfg = MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=2,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        num_speculative_tokens=7,
    )
    prefill_cfg = MegakernelEngineConfig(
        role="prefill",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=2,
        max_model_len=4096,
        max_num_batched_tokens=4096,
    )

    with (
        patch.object(bridge, "kernel_mesh", return_value=(mesh_tp, 0)),
        patch.object(decode_ops, "kernel_mesh", return_value=(mesh_tp, 0)),
        patch.object(draft, "kernel_mesh", return_value=(mesh_tp, 0)),
        patch.object(prefill, "kernel_mesh", return_value=(mesh_sp, 0)),
        patch.object(prefill, "tp_rank_permutation", return_value=None),
    ):
        for bad_uri in ("s3://bucket/prefix", "s3:/bucket/collapsed"):
            with pytest.raises(ValueError, match="local filesystem path"):
                draft.draft_class(bad_uri)
            with pytest.raises(ValueError, match="local filesystem path"):
                draft.DsparkDraft(4096, bad_uri, tmp_path)
            with pytest.raises(ValueError, match="local filesystem path"):
                draft.DsparkDraft(4096, str(ring_ckpt), cast("Path", bad_uri))
            with pytest.raises(ValueError, match="local filesystem path"):
                draft_mla.MlaDsparkDraft(4096, bad_uri, tmp_path)
            with pytest.raises(ValueError, match="local filesystem path"):
                draft_mla.MlaDsparkDraft(4096, str(mla_ckpt), cast("Path", bad_uri))
            with pytest.raises(ValueError, match="local filesystem path"):
                draft_mla_prefill.PrefillDraftLatents(
                    mesh_sp, 0, AXIS_SP, bad_uri, tmp_path, 4096
                )
            with pytest.raises(ValueError, match="local filesystem path"):
                draft_mla_prefill.PrefillDraftLatents(
                    mesh_sp, 0, AXIS_SP, mla_ckpt, bad_uri, 4096
                )
            with pytest.raises(ValueError, match="local filesystem path"):
                draft_ring.PrefillRing(
                    mesh_sp, 0, AXIS_SP, cast("Path", bad_uri), tmp_path
                )
            with pytest.raises(ValueError, match="local filesystem path"):
                draft_ring.PrefillRing(
                    mesh_sp, 0, AXIS_SP, ring_ckpt, cast("Path", bad_uri)
                )
            with pytest.raises(ValueError, match="local filesystem path"):
                program.build(decode_cfg, cast("Path", bad_uri), tp_rank=0)
            with pytest.raises(ValueError, match="local filesystem path"):
                program.build(
                    decode_cfg,
                    tmp_path,
                    cast("Path", bad_uri),
                    tp_rank=0,
                )
            with pytest.raises(ValueError, match="local filesystem path"):
                decode_ops.DecodeOps(bad_uri, num_slots=2)
            with pytest.raises(ValueError, match="local filesystem path"):
                prefill.KimiPrefiller(prefill_cfg, cast("Path", bad_uri), tp_rank=0)
            with pytest.raises(ValueError, match="local filesystem path"):
                prefill.KimiPrefiller(
                    prefill_cfg,
                    tmp_path,
                    cast("Path", bad_uri),
                    tp_rank=0,
                )
