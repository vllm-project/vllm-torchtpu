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

import json
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.models.kimi_k3.megakernel import runner
from vllm_torchtpu.models.kimi_k3.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import aggregate as aggregate_mod
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import prefill as prefill_mod
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import program as program_mod
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.aggregate import KimiAggregate
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import (
    EP8_DEFAULT_EXPERT_PLAN,
    EP8_DEFAULT_REPLICA_PLAN,
    EP16_DEFAULT_EXPERT_PLAN,
    EP16_DEFAULT_REPLICA_PLAN,
    KernelConfig,
    resolve_expert_parallel_settings,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    decode_megakernel as dm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import rank_loader as rl

LAYERS = 3
EXPERT_SHAPES = {
    "expert_gu": (2, 112, 4, 8),
    "expert_gus": (2, 112, 1, 8),
    "expert_down": (2, 112, 6, 4),
    "expert_ds": (2, 112, 2, 4),
}
LAYER_RANGES = ((0, 1), (1, 2))


@pytest.fixture(scope="module")
def preshard_32(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("preshard32")
    expert_raw: dict[tuple[int, str, int], np.ndarray] = {}
    router_raw = np.arange(2 * 4 * 896, dtype=np.float32).reshape(2, 4, 896) + 1.25
    router_bias_raw = np.arange(2 * 1 * 896, dtype=np.float32).reshape(2, 1, 896) - 3.5

    for rank in range(32):
        rank_dir = root / f"rank{rank:02d}"
        rank_dir.mkdir()
        families: dict[str, object] = {}
        dense_shapes: dict[str, tuple[int, ...]] = {
            name: (2, 3) for name in rl.PREFILL_IO + rl.PREFILL_COMMON
        }
        dense_shapes.update({name: (2, 3) for name in rl.PREFILL_KDA + rl.PREFILL_MLA})
        dense_shapes["latent_norm"] = (2, 1, 4096)
        dense_shapes["router"] = (2, 4, 896)
        dense_shapes["router_bias"] = (2, 1, 896)
        dense_shapes["k_o"] = (2, 768, 3)
        dense_shapes["m_o"] = (2, 768, 3)

        blobs = []
        offset = 0
        for name, shape in dense_shapes.items():
            if name == "router":
                arr = router_raw.copy()
            elif name == "router_bias":
                arr = router_bias_raw.copy()
            else:
                arr = np.full(shape, float(rank + 1), dtype=np.float32)
            families[name] = {
                "file": "dense.bin",
                "offset": offset,
                "shape": list(shape),
                "dtype": "float32",
            }
            blobs.append(arr.tobytes())
            offset += arr.nbytes
        (rank_dir / "dense.bin").write_bytes(b"".join(blobs))

        for name, shape in EXPERT_SHAPES.items():
            files = []
            for part_idx, (lo, hi) in enumerate(LAYER_RANGES):
                part_shape = (hi - lo, *shape[1:])
                base = (
                    (rank + 1) * 1_000_000
                    + (part_idx + 1) * 100_000
                    + np.arange(int(np.prod(part_shape)), dtype=np.int32).reshape(
                        part_shape
                    )
                ).astype(np.float32)
                expert_raw[(rank, name, lo)] = base
                filename = f"{name}_{part_idx}.bin"
                base.tofile(rank_dir / filename)
                files.append(filename)
            families[name] = {
                "shape": list(shape),
                "dtype": "float32",
                "files": files,
                "layer_ranges": [list(r) for r in LAYER_RANGES],
            }
        (rank_dir / "index.json").write_text(
            json.dumps({"rank": rank, "layers": LAYERS, "families": families})
        )

    cache = root / "prefill-sp-cache"
    cache.mkdir()
    cache_families = {}
    for name in (
        "latent_down",
        "latent_up",
        "shared_gu_0",
        "shared_gu_1",
        "shared_down_0",
        "shared_down_1",
    ):
        np.full((2, 2), 0.5, dtype=np.float32).tofile(cache / f"{name}.bin")
        cache_families[name] = {
            "file": f"{name}.bin",
            "shape": [2, 2],
            "dtype": "float32",
        }
    (cache / "layout.json").write_text(
        json.dumps({"layers": LAYERS, "families": cache_families})
    )

    rng = np.random.default_rng(20261112)
    order = np.stack([rng.permutation(896), rng.permutation(896)], axis=0).astype(
        np.int64
    )
    plan_path = root / "custom_ep16_plan.npz"
    np.savez(plan_path, order=order)

    return {
        "root": root,
        "cache": cache,
        "expert_raw": expert_raw,
        "router_raw": router_raw,
        "router_bias_raw": router_bias_raw,
        "order": order,
        "plan_path": str(plan_path),
    }


def _extract_decode_half(
    name: str, family_entry: np.ndarray, slot: tuple[int, int], half: int
) -> np.ndarray:
    extracted = np.zeros(slot, dtype=family_entry.dtype)
    for axis, start, size, dst_offset in dm.expert_half_windows(name, slot, half):
        if axis == 1:
            extracted[:, dst_offset : dst_offset + size] = family_entry[
                :, start : start + size
            ]
        else:
            extracted[:] = family_entry[start : start + size]
    return extracted


def test_should_default_k3_unified_ep16_env_to_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("K3_UNIFIED_EP16", raising=False)
    assert envs.environment_variables["K3_UNIFIED_EP16"]() is True
    monkeypatch.setenv("K3_UNIFIED_EP16", "0")
    assert envs.environment_variables["K3_UNIFIED_EP16"]() is False


def test_should_resolve_unified_ep16_settings_when_enabled() -> None:
    assert resolve_expert_parallel_settings(
        aggregate=True,
        unified_ep16=True,
        aggregate_ep16=False,
        raw_expert_plan="",
        raw_replica_plan="replica_plan_ep8_real_plain4.npz",
        raw_hot_plan="",
    ) == (16, 16, EP16_DEFAULT_EXPERT_PLAN, EP16_DEFAULT_REPLICA_PLAN)

    assert resolve_expert_parallel_settings(
        aggregate=True,
        unified_ep16=True,
        aggregate_ep16=False,
        raw_expert_plan=EP8_DEFAULT_EXPERT_PLAN,
        raw_replica_plan="",
        raw_hot_plan="",
    ) == (16, 16, EP16_DEFAULT_EXPERT_PLAN, EP16_DEFAULT_REPLICA_PLAN)

    assert resolve_expert_parallel_settings(
        aggregate=True,
        unified_ep16=True,
        aggregate_ep16=False,
        raw_expert_plan="plan_custom_ep16.npz",
        raw_replica_plan="none",
        raw_hot_plan="",
    ) == (16, 16, "plan_custom_ep16.npz", "")

    assert resolve_expert_parallel_settings(
        aggregate=True,
        unified_ep16=True,
        aggregate_ep16=False,
        raw_expert_plan="off",
        raw_replica_plan="",
        raw_hot_plan="hot.npz",
    ) == (16, 16, "", "")

    assert resolve_expert_parallel_settings(
        aggregate=True,
        unified_ep16=True,
        aggregate_ep16=False,
        raw_expert_plan="",
        raw_replica_plan="custom_ep16_replica.npz",
        raw_hot_plan="",
    ) == (16, 16, EP16_DEFAULT_EXPERT_PLAN, "custom_ep16_replica.npz")


def test_should_preserve_ep8_and_disagg_settings_when_disabled() -> None:
    assert resolve_expert_parallel_settings(
        aggregate=False,
        unified_ep16=True,
        aggregate_ep16=False,
        raw_expert_plan="custom.npz",
        raw_replica_plan="rep.npz",
        raw_hot_plan="",
    ) == (16, 8, "custom.npz", "rep.npz")

    assert resolve_expert_parallel_settings(
        aggregate=True,
        unified_ep16=False,
        aggregate_ep16=False,
        raw_expert_plan="",
        raw_replica_plan="",
        raw_hot_plan="",
    ) == (8, 8, EP8_DEFAULT_EXPERT_PLAN, EP8_DEFAULT_REPLICA_PLAN)

    assert resolve_expert_parallel_settings(
        aggregate=True,
        unified_ep16=False,
        aggregate_ep16=False,
        raw_expert_plan="custom_ep8.npz",
        raw_replica_plan="",
        raw_hot_plan="hot.npz",
    ) == (8, 8, "custom_ep8.npz", "")

    assert resolve_expert_parallel_settings(
        aggregate=True,
        unified_ep16=False,
        aggregate_ep16=False,
        raw_expert_plan="none",
        raw_replica_plan="0",
        raw_hot_plan="",
    ) == (8, 8, "", "")

    assert resolve_expert_parallel_settings(
        aggregate=True,
        unified_ep16=False,
        aggregate_ep16=True,
        raw_expert_plan="",
        raw_replica_plan="",
        raw_hot_plan="",
    ) == (16, 8, "", "")


def test_should_validate_decode_expert_parallel_in_kernel_config() -> None:
    assert KernelConfig().decode_expert_parallel == 8
    cfg = KernelConfig(
        decode_expert_parallel=16,
        prefill_expert_parallel=16,
        kd_replica_plan=EP16_DEFAULT_REPLICA_PLAN,
    )
    assert cfg.decode_expert_parallel == 16
    assert cfg.kd_replica_plan == EP16_DEFAULT_REPLICA_PLAN

    with pytest.raises(ValueError, match="decode_expert_parallel"):
        KernelConfig(decode_expert_parallel=12)
    with pytest.raises(ValueError, match="kd_hot_plan"):
        KernelConfig(decode_expert_parallel=16, kd_hot_plan="hot.npz")


def test_should_balance_load_and_preserve_hotfirst_order_in_ep16_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            prefill_expert_plan=EP16_DEFAULT_EXPERT_PLAN,
        ),
    )
    ep16_order = rl.prefill_expert_plan(93)
    assert ep16_order is not None
    assert ep16_order.shape == (92, 896)

    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=8,
            decode_expert_parallel=8,
            prefill_expert_plan=EP8_DEFAULT_EXPERT_PLAN,
        ),
    )
    ep8_order = rl.prefill_expert_plan(93)
    assert ep8_order is not None

    zipf_curve = 1.0 / (np.arange(1, 113, dtype=np.float64) ** 0.6)
    for layer in (0, 45, 91):
        weights = np.empty(896, dtype=np.float64)
        for group in range(8):
            block = ep8_order[layer, 112 * group : 112 * (group + 1)]
            weights[block] = zipf_curve
        naive_chip_loads = np.array(
            [weights[ep8_order[layer, 56 * c : 56 * (c + 1)]].sum() for c in range(16)]
        )
        balanced_chip_loads = np.array(
            [weights[ep16_order[layer, 56 * c : 56 * (c + 1)]].sum() for c in range(16)]
        )
        assert naive_chip_loads.max() / naive_chip_loads.mean() > 1.35
        assert balanced_chip_loads.max() / balanced_chip_loads.mean() < 1.01
        for chip in range(16):
            chip_weights = weights[ep16_order[layer, 56 * chip : 56 * (chip + 1)]]
            assert np.all(chip_weights[:-1] >= chip_weights[1:] - 1e-12)


@pytest.mark.parametrize("rank", [0, 1, 2, 7, 19])
def test_should_match_decode_and_prefill_ep16_experts_when_unplanned(
    monkeypatch: pytest.MonkeyPatch,
    preshard_32: dict[str, object],
    rank: int,
) -> None:
    root = Path(str(preshard_32["root"]))
    cache = Path(str(preshard_32["cache"]))
    expert_raw = preshard_32["expert_raw"]
    assert isinstance(expert_raw, dict)

    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            prefill_expert_plan="",
        ),
    )
    decode_joined = {
        name: arr
        for _, name, arr in rl.iter_decode_rank(
            root, rank, layers=LAYERS, split_experts=False
        )
    }
    decode_split = [
        (name, arr)
        for _, name, arr in rl.iter_decode_rank(
            root, rank, layers=LAYERS, split_experts=True
        )
        if "@" in name
    ]
    assert len(decode_split) == len(EXPERT_SHAPES) * len(LAYER_RANGES)

    prefill_items = {
        name: arr
        for _, name, arr in rl.iter_prefill_rank(
            root, cache, rank, layers=LAYERS, split_experts=False
        )
    }
    for name in EXPERT_SHAPES:
        np.testing.assert_array_equal(decode_joined[name], prefill_items[name])
        slot = dm.expert_slot_shape(name, decode_joined[name].shape[1:])
        slot_ep8 = dm.expert_slot_shape(name, decode_joined[name].shape[1:], halves=1)
        assert slot_ep8 == decode_joined[name].shape[2:]
        for layer in range(LAYERS - 1):
            for expert in (0, 23, 55):
                for half in (0, 1):
                    got = _extract_decode_half(
                        name, decode_joined[name][layer, expert], slot, half
                    )
                    global_expert = 56 * (rank // 2) + expert
                    src_rank = 4 * (global_expert // 112) + 2 * (rank % 2) + half
                    want = expert_raw[(src_rank, name, layer)][0, global_expert % 112]
                    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("rank", [0, 1, 6, 31])
def test_should_permute_experts_and_router_identically_when_ep16_plan_active(
    monkeypatch: pytest.MonkeyPatch,
    preshard_32: dict[str, object],
    rank: int,
) -> None:
    root = Path(str(preshard_32["root"]))
    cache = Path(str(preshard_32["cache"]))
    expert_raw = preshard_32["expert_raw"]
    router_raw = preshard_32["router_raw"]
    router_bias_raw = preshard_32["router_bias_raw"]
    order = preshard_32["order"]
    plan_path = str(preshard_32["plan_path"])
    assert isinstance(expert_raw, dict)
    assert isinstance(router_raw, np.ndarray)
    assert isinstance(router_bias_raw, np.ndarray)
    assert isinstance(order, np.ndarray)

    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            prefill_expert_plan=plan_path,
        ),
    )
    decode_items = {
        name: arr
        for _, name, arr in rl.iter_decode_rank(
            root, rank, layers=LAYERS, split_experts=False
        )
    }
    prefill_items = {
        name: arr
        for _, name, arr in rl.iter_prefill_rank(
            root, cache, rank, layers=LAYERS, split_experts=False
        )
    }
    expected_router = np.take_along_axis(router_raw, order[:, None, :], axis=-1)
    expected_bias = np.take_along_axis(router_bias_raw, order[:, None, :], axis=-1)
    np.testing.assert_array_equal(decode_items["router"], expected_router)
    np.testing.assert_array_equal(decode_items["router_bias"], expected_bias)
    np.testing.assert_array_equal(decode_items["router"], prefill_items["router"])
    np.testing.assert_array_equal(
        decode_items["router_bias"], prefill_items["router_bias"]
    )

    for name in EXPERT_SHAPES:
        np.testing.assert_array_equal(decode_items[name], prefill_items[name])
        slot = dm.expert_slot_shape(name, decode_items[name].shape[1:])
        for layer in range(LAYERS - 1):
            for expert in (0, 17, 55):
                logical_expert = int(order[layer, 56 * (rank // 2) + expert])
                for half in (0, 1):
                    got = _extract_decode_half(
                        name, decode_items[name][layer, expert], slot, half
                    )
                    src_rank = 4 * (logical_expert // 112) + 2 * (rank % 2) + half
                    want = expert_raw[(src_rank, name, layer)][0, logical_expert % 112]
                    np.testing.assert_array_equal(got, want)


def test_should_allow_shared_expert_skip_in_prefill_when_decode_ep_matches(
    monkeypatch: pytest.MonkeyPatch,
    preshard_32: dict[str, object],
) -> None:
    root = Path(str(preshard_32["root"]))
    cache = Path(str(preshard_32["cache"]))

    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            prefill_expert_plan=str(preshard_32["plan_path"]),
        ),
    )
    remaining = [
        name
        for _, name, _ in rl.iter_prefill_rank(
            root,
            cache,
            0,
            layers=LAYERS,
            split_experts=True,
            skip=frozenset(rl.PREFILL_SHARED),
        )
    ]
    assert sorted(remaining) == sorted(
        ["k_o", "m_o", "latent_down", "latent_up", "shared_gu", "shared_down"]
    )

    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=8,
            decode_expert_parallel=16,
        ),
    )
    with pytest.raises(ValueError, match="EP16"):
        list(
            rl.iter_prefill_rank(
                root,
                cache,
                0,
                layers=LAYERS,
                skip=frozenset(rl.EXPERT_FAMILIES),
            )
        )


def test_should_share_all_prefill_families_in_aggregate_when_unified_ep16(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eng_cfg = MegakernelEngineConfig(
        role="aggregate",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=512,
        kv_cache_dtype="fp8",
        max_num_seqs=4,
        max_model_len=1 << 20,
        max_num_batched_tokens=8256,
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=((1, 1, 7), (2, 4, 3)),
    )
    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=8,
            decode_expert_parallel=16,
        ),
    )
    with pytest.raises(NotImplementedError, match="prefill_expert_parallel 16"):
        KimiAggregate(eng_cfg, tmp_path, tp_rank=0)

    shared_calls: list[object] = []
    weights = {name: object() for name in (*rl.PREFILL_SHARED, "k_o", "latent_down")}
    monkeypatch.setattr(
        program_mod,
        "build_decoder",
        lambda cfg, presharded, draft, *, tp_rank: types.SimpleNamespace(
            rank=tp_rank,
            load_weights=lambda: shared_calls.append("decode"),
            ops=types.SimpleNamespace(weights=weights),
        ),
    )
    monkeypatch.setattr(
        prefill_mod,
        "KimiPrefiller",
        lambda cfg, presharded, draft, *, tp_rank, compute_logits=True: (
            types.SimpleNamespace(
                rank=tp_rank,
                compute_logits=compute_logits,
                load_weights=lambda shared: shared_calls.append(shared),
            )
        ),
    )

    monkeypatch.setattr(
        aggregate_mod,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            prefill_expert_plan=EP16_DEFAULT_EXPERT_PLAN,
        ),
    )
    unified_engine = KimiAggregate(eng_cfg, tmp_path, tp_rank=2)
    unified_engine.load_weights()
    assert shared_calls[0] == "decode"
    assert shared_calls[1] == {n: weights[n] for n in rl.PREFILL_SHARED}

    shared_calls.clear()
    monkeypatch.setattr(
        aggregate_mod,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=8,
            prefill_expert_plan=EP16_DEFAULT_EXPERT_PLAN,
        ),
    )
    split_engine = KimiAggregate(eng_cfg, tmp_path, tp_rank=2)
    split_engine.load_weights()
    excluded = {*rl.EXPERT_FAMILIES, "router", "router_bias"}
    assert shared_calls[1] == {
        n: weights[n] for n in rl.PREFILL_SHARED if n not in excluded
    }


def test_should_expand_physical_routes_to_ordered_local_halves() -> None:
    experts = jnp.array([[3, 9, 0, 0], [55, 0, 0, 0]], dtype=jnp.int32)
    probabilities = jnp.array(
        [[0.61, 0.27, 0.0, 0.0], [0.83, 0.0, 0.0, 0.0]], dtype=jnp.float32
    )
    counts = jnp.array([2, 1], dtype=jnp.int32)
    local_experts, local_probs, local_counts = dm.expert_half_routes(
        experts, probabilities, counts
    )
    assert local_experts.shape == (2, 8)
    np.testing.assert_array_equal(local_experts[0, :4], [6, 7, 18, 19])
    np.testing.assert_array_equal(local_experts[1, :2], [110, 111])
    np.testing.assert_allclose(
        local_probs[0, :4], [0.61, 0.61, 0.27, 0.27], rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(local_probs[1, :2], [0.83, 0.83], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(local_counts, [4, 2])


def test_should_install_unified_ep16_kernel_config_in_build_program(
    monkeypatch: pytest.MonkeyPatch,
    preshard_32: dict[str, object],
) -> None:
    root = Path(str(preshard_32["root"]))
    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=8,
            decode_expert_parallel=8,
            prefill_expert_plan="",
        ),
    )
    ep8_decode = {
        name: arr
        for _, name, arr in rl.iter_decode_rank(
            root, 0, layers=LAYERS, split_experts=False
        )
    }
    assert ep8_decode["expert_gu"].shape == (2, 112, 4, 8)

    installed: list[KernelConfig] = []
    monkeypatch.setattr(config, "install", installed.append)
    monkeypatch.setattr(
        runner.k3,
        "build",
        lambda cfg, presharded, draft, *, tp_rank: ("built", cfg.role, tp_rank),
    )
    monkeypatch.setenv("K3_UNIFIED_EP16", "1")
    monkeypatch.setenv("K3_PREFILL_EXPERT_PLAN", EP8_DEFAULT_EXPERT_PLAN)

    fake_vllm_cfg = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                architectures=["KimiK3ForConditionalGeneration"]
            ),
            hf_text_config=types.SimpleNamespace(
                num_hidden_layers=93,
                hidden_size=7168,
            ),
            max_model_len=32768,
        ),
        parallel_config=types.SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=32,
            decode_context_parallel_size=32,
            cp_kv_cache_interleave_size=1,
        ),
        cache_config=types.SimpleNamespace(
            block_size=512,
            cache_dtype="fp8",
            mamba_ssm_cache_dtype="auto",
        ),
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=16384,
            async_scheduling=True,
        ),
        load_config=types.SimpleNamespace(
            load_format="megakernel",
            model_loader_extra_config={"presharded": str(root)},
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )
    built = runner.build_program(fake_vllm_cfg, tp_rank=3)
    assert built == ("built", "aggregate", 3)
    assert len(installed) == 1
    assert installed[0].decode_expert_parallel == 16
    assert installed[0].prefill_expert_parallel == 16
    assert installed[0].prefill_expert_plan == EP16_DEFAULT_EXPERT_PLAN
    assert installed[0].kd_replica_plan == EP16_DEFAULT_REPLICA_PLAN
    assert installed[0].decode_expert_banks


def test_should_validate_shipped_ep16_replica_plan_and_reject_invalid_plans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            prefill_expert_plan=EP16_DEFAULT_EXPERT_PLAN,
            kd_replica_plan=EP16_DEFAULT_REPLICA_PLAN,
        ),
    )
    expert_order = rl.prefill_expert_plan(93)
    assert expert_order is not None
    loaded = rl.replica_plan(0, 93, expert_order=expert_order)
    assert loaded is not None
    rank0_slots, experts, order = loaded
    assert rank0_slots.shape == (92, 4)
    assert experts.shape == (92, 16, 4)
    assert order.shape == (92, 64)
    for layer in range(92):
        assert sorted(order[layer].tolist()) == list(range(64))
        flat = experts[layer].reshape(-1).tolist()
        assert len(set(flat)) == 64
        for group in range(16):
            row = experts[layer, group].tolist()
            assert all(0 <= e < 896 and e // 56 != group for e in row)

    rank31 = rl.replica_plan(31, 93, expert_order=expert_order)
    assert rank31 is not None
    np.testing.assert_array_equal(rank31[0], experts[:, 15, :])

    too_many_slots = tmp_path / "bad_slots.npz"
    np.savez(
        too_many_slots,
        experts=np.zeros((2, 16, 9), dtype=np.int32),
        order=np.tile(np.arange(144, dtype=np.int32), (2, 1)),
    )
    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            kd_replica_plan=str(too_many_slots),
        ),
    )
    with pytest.raises(ValueError, match="1 <= S <= 8"):
        rl.replica_plan(0, 3)

    bad_order = tmp_path / "bad_order.npz"
    np.savez(
        bad_order,
        experts=np.full((2, 16, 2), -1, dtype=np.int32),
        order=np.zeros((2, 32), dtype=np.int32),
    )
    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            kd_replica_plan=str(bad_order),
        ),
    )
    with pytest.raises(ValueError, match="order must permute"):
        rl.replica_plan(0, 3)

    dup_expert = tmp_path / "dup_expert.npz"
    dup_arr = np.full((2, 16, 2), -1, dtype=np.int32)
    dup_arr[0, 0, 0] = 7
    dup_arr[0, 1, 0] = 7
    np.savez(
        dup_expert,
        experts=dup_arr,
        order=np.tile(np.arange(32, dtype=np.int32), (2, 1)),
    )
    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            kd_replica_plan=str(dup_expert),
        ),
    )
    with pytest.raises(ValueError, match="invalid or repeated"):
        rl.replica_plan(0, 3)


@pytest.mark.parametrize("use_expert_plan", [False, True])
@pytest.mark.parametrize("rank", [0, 1, 6, 31])
def test_should_regroup_ep16_replicas_and_populate_replica_table_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
    preshard_32: dict[str, object],
    tmp_path: Path,
    use_expert_plan: bool,
    rank: int,
) -> None:
    root = Path(str(preshard_32["root"]))
    cache = Path(str(preshard_32["cache"]))
    expert_raw = preshard_32["expert_raw"]
    order = preshard_32["order"]
    plan_path = str(preshard_32["plan_path"]) if use_expert_plan else ""
    assert isinstance(expert_raw, dict)
    assert isinstance(order, np.ndarray)

    rep_experts = np.full((2, 16, 2), -1, dtype=np.int32)
    for layer in range(2):
        for g in range(16):
            peer_g = (g + 5) % 16
            src_slot = 56 * peer_g + 3
            same_slot = 56 * g + 11
            rep_experts[layer, g, 0] = (
                int(order[layer, src_slot]) if use_expert_plan else src_slot
            )
            if g < 15:
                rep_experts[layer, g, 1] = (
                    int(order[layer, same_slot]) if use_expert_plan else same_slot
                )
    rep_order = np.stack(
        [np.arange(32, dtype=np.int32)[::-1], np.arange(32, dtype=np.int32)],
        axis=0,
    )
    rep_path = tmp_path / f"rep_ep16_{int(use_expert_plan)}.npz"
    np.savez(rep_path, experts=rep_experts, order=rep_order)

    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            prefill_expert_plan=plan_path,
            kd_replica_plan=str(rep_path),
        ),
    )
    monkeypatch.setattr(
        rl.RankDir,
        "read_dense_all",
        lambda self, _orig=rl.RankDir.read_dense_all: {
            k: (
                np.pad(v[..., :3584], ((0, 0), (0, 0), (0, 512)))
                if k == "latent_norm"
                else v
            )
            for k, v in _orig(self).items()
        },
    )

    decode_items = {
        name: arr
        for _, name, arr in rl.iter_decode_rank(
            root, rank, layers=LAYERS, split_experts=False
        )
    }
    split_parts = [
        (name, arr)
        for _, name, arr in rl.iter_decode_rank(
            root, rank, layers=LAYERS, split_experts=True
        )
        if "@" in name
    ]
    assert len(split_parts) == len(EXPERT_SHAPES) * len(LAYER_RANGES)
    prefill_items = {
        name: arr
        for _, name, arr in rl.iter_prefill_rank(
            root, cache, rank, layers=LAYERS, split_experts=False
        )
    }

    group = rank // 2
    for name in EXPERT_SHAPES:
        assert decode_items[name].shape[1] == 58
        np.testing.assert_array_equal(decode_items[name][:, :56], prefill_items[name])
        slot = dm.expert_slot_shape(name, decode_items[name].shape[1:])
        for layer in range(LAYERS - 1):
            for j in range(2):
                logical_e = int(rep_experts[layer, group, j])
                if logical_e < 0:
                    np.testing.assert_array_equal(
                        decode_items[name][layer, 56 + j], 0.0
                    )
                    continue
                for half in (0, 1):
                    got = _extract_decode_half(
                        name, decode_items[name][layer, 56 + j], slot, half
                    )
                    src_rank = 4 * (logical_e // 112) + 2 * (rank % 2) + half
                    want = expert_raw[(src_rank, name, layer)][0, logical_e % 112]
                    np.testing.assert_array_equal(got, want)

    ln = decode_items["latent_norm"]
    np.testing.assert_array_equal(ln[:, 0, :3584], float(rank + 1))
    inv_order = np.argsort(order, axis=1) if use_expert_plan else None
    for layer in range(LAYERS - 1):
        for i, flat in enumerate(rep_order[layer]):
            g, j = divmod(int(flat), 2)
            log_e = int(rep_experts[layer, g, j])
            phys_e = (
                -1
                if log_e < 0
                else (int(inv_order[layer, log_e]) if inv_order is not None else log_e)
            )
            assert ln[layer, 0, 3584 + i] == float(phys_e)
            assert ln[layer, 0, 3584 + 32 + i] == float(g if phys_e >= 0 else -1)
            assert ln[layer, 0, 3584 + 64 + i] == float(56 + j if phys_e >= 0 else -1)


def test_should_dispatch_high_group_ep16_routes_to_cool_replica_holders() -> None:
    winners = np.zeros((4, 16), dtype=np.int32)
    for b in range(4):
        winners[b, 0] = 728
        winners[b, 1:8] = np.arange(729, 736, dtype=np.int32)
        winners[b, 8:15] = np.arange(784, 791, dtype=np.int32)
        winners[b, 15] = 504
    counts = np.bincount(winners.reshape(-1), minlength=896)[None, :].astype(np.float32)

    entries = 2
    table = np.zeros((1, 256), dtype=np.float32)
    table[0, 0] = 729
    table[0, 1] = 728
    table[0, entries + 0] = 14
    table[0, entries + 1] = 2
    table[0, 2 * entries + 0] = 56
    table[0, 2 * entries + 1] = 57

    to_copy, holder, slot = dm.replica_dispatch(
        jnp.asarray(winners),
        jnp.asarray(table),
        entries,
        56,
        jnp.asarray(counts),
    )
    to_copy_np = np.asarray(to_copy)
    holder_np = np.asarray(holder)
    slot_np = np.asarray(slot)

    assert not np.any(to_copy_np[:, 1])
    assert np.all(to_copy_np[:, 0])
    np.testing.assert_array_equal(holder_np[:, 0], 2)
    np.testing.assert_array_equal(slot_np[:, 0], 57)


def test_should_preserve_ep8_replicas_and_reject_dirty_latent_norm_padding(
    monkeypatch: pytest.MonkeyPatch,
    preshard_32: dict[str, object],
    tmp_path: Path,
) -> None:
    root = Path(str(preshard_32["root"]))
    expert_raw = preshard_32["expert_raw"]
    assert isinstance(expert_raw, dict)

    rep_ep8 = np.full((2, 8, 1), -1, dtype=np.int32)
    for layer in range(2):
        for g in range(8):
            rep_ep8[layer, g, 0] = 112 * ((g + 1) % 8) + 5
    rep_ord = np.tile(np.arange(8, dtype=np.int32), (2, 1))
    rep_path = tmp_path / "rep_ep8.npz"
    np.savez(rep_path, experts=rep_ep8, order=rep_ord)

    monkeypatch.setattr(
        config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=8,
            decode_expert_parallel=8,
            prefill_expert_plan="",
            kd_replica_plan=str(rep_path),
        ),
    )
    with pytest.raises(ValueError, match="latent_norm padding lanes"):
        list(rl.iter_decode_rank(root, 2, layers=LAYERS, split_experts=False))

    monkeypatch.setattr(
        rl.RankDir,
        "read_dense_all",
        lambda self, _orig=rl.RankDir.read_dense_all: {
            k: (
                np.pad(v[..., :3584], ((0, 0), (0, 0), (0, 512)))
                if k == "latent_norm"
                else v
            )
            for k, v in _orig(self).items()
        },
    )
    loaded = {
        name: arr
        for _, name, arr in rl.iter_decode_rank(
            root, 2, layers=LAYERS, split_experts=False
        )
    }
    assert loaded["expert_gu"].shape[1] == 113
    np.testing.assert_array_equal(
        loaded["expert_gu"][0, 112], expert_raw[(6, "expert_gu", 0)][0, 5]
    )


@pytest.mark.parametrize("disabled", ["", "0", "none", "off", "false", " NONE "])
def test_resolve_expert_parallel_settings_isolates_1p1d_and_disabled_plans(
    disabled: str,
) -> None:
    # 1P1D (aggregate=False): prefill=16, decode=8; disabled tokens never leak.
    assert resolve_expert_parallel_settings(
        aggregate=False,
        unified_ep16=True,
        aggregate_ep16=True,
        raw_expert_plan=disabled,
        raw_replica_plan=disabled,
        raw_hot_plan="",
    ) == (16, 8, "", "")

    # 1P1D never hands EP16 replica plan or EP8 default expert plan to EP=8 decode.
    assert resolve_expert_parallel_settings(
        aggregate=False,
        unified_ep16=False,
        aggregate_ep16=False,
        raw_expert_plan=EP8_DEFAULT_EXPERT_PLAN,
        raw_replica_plan=EP16_DEFAULT_REPLICA_PLAN,
        raw_hot_plan="",
    ) == (16, 8, "", "")

    # Aggregate unified EP16 respects explicit disable tokens when non-empty.
    if disabled.strip():
        assert resolve_expert_parallel_settings(
            aggregate=True,
            unified_ep16=True,
            aggregate_ep16=True,
            raw_expert_plan=disabled,
            raw_replica_plan=disabled,
            raw_hot_plan="",
        ) == (16, 16, "", "")
        assert resolve_expert_parallel_settings(
            aggregate=True,
            unified_ep16=False,
            aggregate_ep16=False,
            raw_expert_plan=disabled,
            raw_replica_plan=disabled,
            raw_hot_plan="",
        ) == (8, 8, "", "")


def _fake_megakernel_vllm_config(
    *, role: str, attention_dp: int = 1, dcp: int = 32
) -> types.SimpleNamespace:
    extra: dict[str, object] = {"presharded": "/tmp/unused-preshard"}
    if role == "decode" and attention_dp > 1:
        extra["attention_dp"] = attention_dp
    if role == "aggregate":
        kv_cfg = None
    elif role == "prefill":
        kv_cfg = types.SimpleNamespace(
            is_kv_producer=True, is_kv_consumer=False, kv_role="kv_producer"
        )
    else:
        kv_cfg = types.SimpleNamespace(
            is_kv_producer=False, is_kv_consumer=True, kv_role="kv_consumer"
        )
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                architectures=["KimiK3ForConditionalGeneration"]
            ),
            hf_text_config=types.SimpleNamespace(
                num_hidden_layers=93, hidden_size=7168
            ),
            max_model_len=32768,
        ),
        parallel_config=types.SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=32,
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=512,
        ),
        cache_config=types.SimpleNamespace(
            block_size=512,
            cache_dtype="fp8",
            mamba_ssm_cache_dtype="auto",
        ),
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=16384,
            async_scheduling=True,
        ),
        load_config=types.SimpleNamespace(
            load_format="megakernel",
            model_loader_extra_config=extra,
        ),
        kv_transfer_config=kv_cfg,
        speculative_config=None,
    )


def test_build_program_and_kernel_config_isolate_1p1d_and_aggregate_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("K3_"):
            monkeypatch.delenv(key, raising=False)

    # 1. Bare KernelConfig() never pairs decode_expert_parallel=8 with EP16 plan,
    # and strips/normalizes disabled tokens across all plan fields.
    bare = KernelConfig()
    assert bare.decode_expert_parallel == 8
    assert bare.kd_replica_plan == ""
    monkeypatch.setattr(config, "active", lambda: bare)
    assert rl.replica_plan(rank=0, layers=92) is None

    normed = KernelConfig(
        kd_replica_plan=" NONE ",
        prefill_replica_plan=" NONE ",
        prefill_expert_plan=" none ",
        kd_hot_plan=" OFF ",
    )
    assert normed.kd_replica_plan == ""
    assert normed.prefill_replica_plan == ""
    assert normed.prefill_expert_plan == ""
    assert normed.kd_hot_plan == ""

    installed: list[KernelConfig] = []
    monkeypatch.setattr(config, "install", installed.append)
    monkeypatch.setattr(
        runner.k3,
        "build",
        lambda cfg, presharded, draft, *, tp_rank: ("built", cfg.role, tp_rank),
    )

    # 2. 1P1D decode (dp=1, default envs): installs EP8 decode with empty plans.
    installed.clear()
    runner.build_program(_fake_megakernel_vllm_config(role="decode"), tp_rank=0)
    assert len(installed) == 1
    assert installed[0].decode_expert_parallel == 8
    assert installed[0].kd_replica_plan == ""
    assert installed[0].prefill_replica_plan == ""
    assert installed[0].prefill_expert_plan == ""

    # 3. 1P1D decode (dp=2, dcp=16) + K3_HOT_PLAN: no prefill_replica_plan clash.
    installed.clear()
    monkeypatch.setenv("K3_HOT_PLAN", "hot.npz")
    runner.build_program(
        _fake_megakernel_vllm_config(role="decode", attention_dp=2, dcp=16),
        tp_rank=0,
    )
    assert installed[0].kd_hot_plan == "hot.npz"
    assert installed[0].kd_replica_plan == ""
    assert installed[0].prefill_replica_plan == ""
    monkeypatch.delenv("K3_HOT_PLAN")

    # 4. 1P1D decode (dp=2, dcp=16) + EP8 K3_REPLICA_PLAN: no mismatch crash.
    installed.clear()
    monkeypatch.setenv("K3_REPLICA_PLAN", EP8_DEFAULT_REPLICA_PLAN)
    runner.build_program(
        _fake_megakernel_vllm_config(role="decode", attention_dp=2, dcp=16),
        tp_rank=0,
    )
    assert installed[0].kd_replica_plan == EP8_DEFAULT_REPLICA_PLAN
    assert installed[0].prefill_replica_plan == ""
    monkeypatch.delenv("K3_REPLICA_PLAN")

    # 5. 1P1D prefill (default envs and K3_PREFILL_EXPERT_PLAN=none).
    installed.clear()
    runner.build_program(_fake_megakernel_vllm_config(role="prefill"), tp_rank=0)
    assert installed[0].prefill_expert_parallel == 16
    assert installed[0].prefill_expert_plan == EP16_DEFAULT_EXPERT_PLAN
    assert installed[0].prefill_replica_plan == EP16_DEFAULT_REPLICA_PLAN
    assert installed[0].kd_replica_plan == ""

    installed.clear()
    monkeypatch.setenv("K3_PREFILL_EXPERT_PLAN", "none")
    runner.build_program(_fake_megakernel_vllm_config(role="prefill"), tp_rank=0)
    assert installed[0].prefill_expert_plan == ""
    monkeypatch.delenv("K3_PREFILL_EXPERT_PLAN")

    # 6. Aggregate EP8 (K3_UNIFIED_EP16=0, K3_AGGREGATE_EP16=0).
    installed.clear()
    monkeypatch.setenv("K3_UNIFIED_EP16", "0")
    monkeypatch.setenv("K3_AGGREGATE_EP16", "0")
    runner.build_program(_fake_megakernel_vllm_config(role="aggregate"), tp_rank=0)
    assert installed[0].prefill_expert_parallel == 8
    assert installed[0].decode_expert_parallel == 8
    assert installed[0].kd_replica_plan == EP8_DEFAULT_REPLICA_PLAN
    assert installed[0].prefill_replica_plan == ""

    # 7. Aggregate unified EP16 with K3_REPLICA_PLAN=none and custom plan.
    installed.clear()
    monkeypatch.setenv("K3_UNIFIED_EP16", "1")
    monkeypatch.setenv("K3_REPLICA_PLAN", "none")
    runner.build_program(_fake_megakernel_vllm_config(role="aggregate"), tp_rank=0)
    assert installed[0].kd_replica_plan == ""
    assert installed[0].prefill_replica_plan == ""

    installed.clear()
    monkeypatch.setenv("K3_REPLICA_PLAN", "custom_ep16.npz")
    runner.build_program(_fake_megakernel_vllm_config(role="aggregate"), tp_rank=0)
    assert installed[0].kd_replica_plan == "custom_ep16.npz"
    assert installed[0].prefill_replica_plan == "custom_ep16.npz"
