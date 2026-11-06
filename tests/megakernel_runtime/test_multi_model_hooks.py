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
"""Unit and 2-rank Gloo tests for multi-model megakernel runtime hooks."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest import mock

import jax
import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.runtime.contract import (
    DOMAIN_DCP,
    DOMAIN_KERNEL_MESH,
    DOMAIN_TP,
    DOMAIN_TP_X_DCP,
    PAR_DCP,
    PAR_DP,
    PAR_EP_IN_TP,
    PAR_EP_WIDER_THAN_TP,
    PAR_PCP,
    PAR_PP,
    PAR_TP,
    LayoutRequest,
    TpuMonoSpec,
)
from vllm_torchtpu.megakernel_runtime import (
    WEIGHT_CONSUMED,
    testkit,
)
from vllm_torchtpu.megakernel_runtime import (
    agreement as agreement_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    config as config_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    driver as driver_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    fingerprint as fp_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    guards as guards_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    hooks as hooks_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    salt as salt_mod,
)
from vllm_torchtpu.megakernel_runtime.errors import MegakernelPluginError
from vllm_torchtpu.megakernel_runtime.plugin import RunnerView, Shorthand
from vllm_torchtpu.megakernel_runtime.testing import fake_runner, fake_vllm_config

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


@dataclasses.dataclass(frozen=True)
class _PluginLayoutDataclass:
    """Dataclass stand-in returned by a plugin `layout_request` hook."""

    block_size: int
    cp_kv_cache_interleave_size: int
    mamba_slot_tokens: int


class _FakeAttentionBackend:
    """Deterministic attention backend for cache-slot derivation tests."""

    @staticmethod
    def get_name() -> str:
        """Return backend identifier."""
        return "TEST_BACKEND"

    @staticmethod
    def get_min_page_size(_vllm_config: object) -> int:
        """Return minimum supported page size in tokens."""
        return 16

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        """Return supported kernel block granularity."""
        return [16]

    @staticmethod
    def get_kv_cache_page_size_bytes(
        block_size: int, *_args: object, **_kwargs: object
    ) -> int:
        """Return linear 1 KiB-per-token physical slot byte size."""
        bytes_per_token = 1024
        return block_size * bytes_per_token


class _RecordingModel(torch.nn.Module):
    """Minimal module recording `(name, tensor)` pairs reaching `load_weights`."""

    def __init__(self) -> None:
        """Initialize empty weight store."""
        super().__init__()
        self.loaded: dict[str, torch.Tensor] = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Consume weight iterator into `self.loaded`."""
        for name, tensor in weights:
            self.loaded[name] = tensor
        return set(self.loaded)


class _GetAllWeightsLoader:
    """Loader exposing `get_all_weights`, matching `DefaultModelLoader`."""

    def __init__(self, raw: list[tuple[str, torch.Tensor]]) -> None:
        """Store raw checkpoint weight pairs."""
        self.raw = raw

    def get_all_weights(
        self, _model_config: object, _model: object
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield raw checkpoint weights before post-load processing."""
        yield from self.raw

    def load_model(self, vllm_config: object, model_config: object) -> _RecordingModel:
        """Load weights and drop transient scale tensor like quantized MoE layers."""
        _ = vllm_config
        model = _RecordingModel()
        model.load_weights(self.get_all_weights(model_config, model))
        model.loaded.pop("layer.0.w13_weight_scale_2", None)
        return model


class _LoadWeightsOnlyLoader:
    """Custom `BaseModelLoader`-style loader implementing `load_weights` only."""

    def __init__(self, raw: list[tuple[str, torch.Tensor]]) -> None:
        """Store raw checkpoint weight pairs."""
        self.raw = raw

    def load_weights(self, model: _RecordingModel, _model_config: object) -> None:
        """Feed raw weights directly into `model.load_weights`."""
        model.load_weights(iter(self.raw))

    def load_model(self, vllm_config: object, model_config: object) -> _RecordingModel:
        """Instantiate `_RecordingModel` and populate via `self.load_weights`."""
        _ = vllm_config
        model = _RecordingModel()
        self.load_weights(model, model_config)
        return model


@contextlib.contextmanager
def _patched_plugin(
    spec: TpuMonoSpec, options: dict[str, object] | None = None
) -> Iterator[None]:
    """Patch `hooks` plugin selection and import to return `spec`."""
    opts = {} if options is None else dict(options)
    selected = (SimpleNamespace(enabled=True), (config_mod.PluginRef("dummy"), opts))
    with (
        mock.patch.object(hooks_mod, "select_plugin", return_value=selected),
        mock.patch.object(
            hooks_mod, "import_plugin", return_value=(spec, ModuleType("dummy"))
        ),
    ):
        yield


@pytest.mark.parametrize(
    ("domain", "topology", "expected"),
    [
        (DOMAIN_TP, {"tp_size": 8, "dp_size": 2}, 8),
        (DOMAIN_DCP, {"tp_size": 1, "dcp_size": 8, "pcp_size": 8}, 8),
        (DOMAIN_TP_X_DCP, {"tp_size": 32, "dcp_size": 8}, 32),
        (DOMAIN_TP_X_DCP, {"tp_size": 1, "dcp_size": 8, "pcp_size": 8}, 8),
        (DOMAIN_TP_X_DCP, {"tp_size": 8, "dcp_size": 8}, 8),
        (DOMAIN_TP_X_DCP, {"tp_size": 4, "dcp_size": 8, "pcp_size": 2}, 8),
        (DOMAIN_KERNEL_MESH, {"tp_size": 16, "dp_size": 2}, 32),
        (DOMAIN_KERNEL_MESH, {"tp_size": 8, "dp_size": 4}, 32),
        (
            DOMAIN_KERNEL_MESH,
            {"tp_size": 4, "dp_size": 2, "pp_size": 2, "pcp_size": 2},
            32,
        ),
    ],
)
def test_domain_world_size_for_nested_and_mesh_topologies(
    domain: str,
    topology: dict[str, int],
    expected: int,
) -> None:
    """Compute expected mesh device counts across all four collective domains."""
    runner = fake_runner(
        tp_size=topology.get("tp_size", 1),
        dp_size=topology.get("dp_size", 1),
        pp_size=topology.get("pp_size", 1),
        dcp_size=topology.get("dcp_size", 1),
        pcp_size=topology.get("pcp_size", 1),
    )
    view = RunnerView(runner, tp_rank=0)
    assert guards_mod.domain_world_size(domain, runner, view) == expected


def test_kernel_mesh_domain_builds_two_device_mesh_and_resolves_groups() -> None:
    """Verify `DOMAIN_KERNEL_MESH` mesh construction and coordinator routing."""
    devs = jax.devices()
    devs_pair = [devs[0], devs[1]] if len(devs) > 1 else [devs[0], devs[0]]
    fallback_mesh = jax.sharding.Mesh(np.asarray(devs_pair), ("tp",))
    runner_mesh = fake_runner(tp_size=1, dp_size=2, mesh=fallback_mesh)
    view_mesh = RunnerView(runner_mesh, tp_rank=0)
    expected_devices = 2
    mesh_ranks = guards_mod.domain_world_size(
        DOMAIN_KERNEL_MESH, runner_mesh, view_mesh
    )
    assert mesh_ranks == expected_devices
    built_mesh = vars(driver_mod)["_build_tp_rank_mesh"](
        mesh_ranks, fallback_mesh=fallback_mesh
    )
    assert driver_mod.check_mesh_size(mesh_ranks, int(built_mesh.devices.size)) is None

    sentinel_tp, sentinel_dcp, sentinel_world = object(), object(), object()
    runner_tp_nested = fake_runner(tp_size=32, dcp_size=8)
    runner_dcp_nested = fake_runner(tp_size=1, dcp_size=8)
    with (
        mock.patch.object(torch.distributed, "is_available", return_value=True),
        mock.patch.object(torch.distributed, "is_initialized", return_value=True),
        mock.patch(
            "vllm.distributed.get_tp_group",
            return_value=SimpleNamespace(
                world_size=32, cpu_group=sentinel_tp, ranks=list(range(32))
            ),
        ),
        mock.patch(
            "vllm.distributed.get_dcp_group",
            return_value=SimpleNamespace(
                world_size=8, cpu_group=sentinel_dcp, ranks=list(range(8))
            ),
        ),
        mock.patch(
            "vllm.distributed.get_world_group",
            return_value=SimpleNamespace(world_size=2, cpu_group=sentinel_world),
        ),
    ):
        assert agreement_mod.agreement_group(DOMAIN_TP, runner_tp_nested) is sentinel_tp
        assert (
            agreement_mod.agreement_group(DOMAIN_DCP, runner_dcp_nested) is sentinel_dcp
        )
        assert (
            agreement_mod.agreement_group(DOMAIN_TP_X_DCP, runner_tp_nested)
            is sentinel_tp
        )
        assert (
            agreement_mod.agreement_group(DOMAIN_TP_X_DCP, runner_dcp_nested)
            is sentinel_dcp
        )
        assert (
            agreement_mod.agreement_group(DOMAIN_KERNEL_MESH, runner_mesh)
            is sentinel_world
        )


@pytest.mark.parametrize(
    ("tp_size", "pcp_size", "dcp_size"),
    [
        (6, 1, 4),
        (4, 3, 6),
        (4, 4, 4),
        (8, 2, 2),
        (2, 4, 4),
    ],
)
def test_non_nested_tp_x_dcp_declines_in_check_spec_and_rejects_in_agreement(
    tp_size: int, pcp_size: int, dcp_size: int
) -> None:
    """Decline crossed or non-divisible TPxDCP at `check_spec` and fail closed."""
    runner = fake_runner(tp_size=tp_size, pcp_size=pcp_size, dcp_size=dcp_size)
    view = RunnerView(runner, tp_rank=0)
    spec = TpuMonoSpec(
        "s_dcp",
        "1",
        create="m:c",
        collective_domain=DOMAIN_TP_X_DCP,
        parallelism=frozenset({PAR_TP, PAR_DCP, PAR_PCP}),
    )
    decline = guards_mod.check_spec(runner, spec, view=view)
    assert decline is not None
    assert decline.code == f"parallelism_unsupported:{DOMAIN_TP_X_DCP}"
    with pytest.raises(MegakernelConfigError, match="Incompatible TP="):
        guards_mod.domain_world_size(DOMAIN_TP_X_DCP, runner, view)
    with pytest.raises(MegakernelConfigError, match="Incompatible TP="):
        agreement_mod.agreement_group(DOMAIN_TP_X_DCP, runner)


@pytest.mark.parametrize(
    ("tp_size", "pcp_size", "dcp_size"),
    [
        (4, 1, 4),
        (8, 1, 2),
        (2, 1, 1),
    ],
)
def test_domain_dcp_requires_unit_tp_size(
    tp_size: int, pcp_size: int, dcp_size: int
) -> None:
    """Accept `DOMAIN_DCP` when `TP=1` and decline/reject when `TP > 1`."""
    pure_dcp_runner = fake_runner(tp_size=1, pcp_size=8, dcp_size=8)
    pure_view = RunnerView(pure_dcp_runner, tp_rank=0)
    spec_dcp = TpuMonoSpec(
        "s_pure_dcp",
        "1",
        create="m:c",
        collective_domain=DOMAIN_DCP,
        parallelism=frozenset({PAR_TP, PAR_DCP, PAR_PCP}),
    )
    assert guards_mod.check_spec(pure_dcp_runner, spec_dcp, view=pure_view) is None

    hybrid_tp_dcp_runner = fake_runner(
        tp_size=tp_size, pcp_size=pcp_size, dcp_size=dcp_size
    )
    hybrid_view = RunnerView(hybrid_tp_dcp_runner, tp_rank=0)
    decline = guards_mod.check_spec(hybrid_tp_dcp_runner, spec_dcp, view=hybrid_view)
    assert decline is not None
    assert decline.code == f"parallelism_unsupported:{DOMAIN_DCP}"
    with pytest.raises(MegakernelConfigError, match="pure DCP requires TP=1"):
        guards_mod.domain_world_size(DOMAIN_DCP, hybrid_tp_dcp_runner, hybrid_view)
    with pytest.raises(MegakernelConfigError, match="pure DCP requires TP=1"):
        agreement_mod.agreement_group(DOMAIN_DCP, hybrid_tp_dcp_runner)


def test_unknown_collective_domain_and_uninitialized_cpu_group_errors() -> None:
    """Reject unknown domains, undeclared wide EP, and uninitialized cpu_group."""
    runner = fake_runner(tp_size=4)
    view = RunnerView(runner, tp_rank=0)
    with pytest.raises(
        MegakernelConfigError, match="Unsupported collective_domain 'dp'"
    ):
        guards_mod.domain_world_size("dp", runner, view)
    with pytest.raises(MegakernelPluginError, match="has no agreement group"):
        agreement_mod.agreement_group("dp", runner)

    ep_runner = fake_runner(tp_size=8, dp_size=2, enable_expert_parallel=True)
    ep_view = RunnerView(ep_runner, tp_rank=0)
    spec_no_ep = TpuMonoSpec(
        "s_no_ep",
        "1",
        create="m:c",
        parallelism=frozenset({PAR_TP, PAR_DP}),
    )
    ep_decline = guards_mod.check_spec(ep_runner, spec_no_ep, view=ep_view)
    assert ep_decline is not None
    assert ep_decline.code == f"parallelism_unsupported:{PAR_EP_WIDER_THAN_TP}"

    with (
        mock.patch.object(torch.distributed, "is_available", return_value=True),
        mock.patch.object(torch.distributed, "is_initialized", return_value=True),
        mock.patch(
            "vllm.distributed.get_tp_group",
            return_value=SimpleNamespace(world_size=4, cpu_group=None),
        ),
        pytest.raises(RuntimeError, match="TP group cpu_group is not initialized"),
    ):
        agreement_mod.agreement_group(DOMAIN_TP, runner)


@pytest.mark.parametrize(
    ("topology", "expected_kinds"),
    [
        ({"tp_size": 8}, []),
        ({"tp_size": 8, "ep": 1}, [PAR_EP_IN_TP]),
        ({"tp_size": 32, "dcp_size": 8, "ep": 1}, [PAR_DCP, PAR_EP_IN_TP]),
        (
            {"tp_size": 16, "dp_size": 2, "dcp_size": 16, "ep": 1},
            [PAR_DP, PAR_EP_WIDER_THAN_TP, PAR_DCP],
        ),
        (
            {"tp_size": 8, "pcp_size": 2, "ep": 1},
            [PAR_EP_WIDER_THAN_TP, PAR_PCP],
        ),
        (
            {"tp_size": 4, "dp_size": 2, "pp_size": 2, "pcp_size": 2},
            [PAR_DP, PAR_PP, PAR_PCP],
        ),
        (
            {"tp_size": 8, "dp_size": 2, "dp_lockstep": 1},
            [PAR_DP, "dp_lockstep"],
        ),
    ],
)
def test_detect_parallelism_across_combined_regimes(
    topology: dict[str, int], expected_kinds: list[str]
) -> None:
    """Detect combined `(TP, DCP, PCP, DP, PP, EP, DP_LOCKSTEP)` regimes."""
    tp_size = topology.get("tp_size", 1)
    runner = fake_runner(
        tp_size=tp_size,
        dp_size=topology.get("dp_size", 1),
        pp_size=topology.get("pp_size", 1),
        dcp_size=topology.get("dcp_size", 1),
        pcp_size=topology.get("pcp_size", 1),
        enable_expert_parallel=bool(topology.get("ep", 0)),
        dp_lockstep=bool(topology.get("dp_lockstep", 0)),
    )
    assert guards_mod.detect_parallelism(runner, tp_size) == expected_kinds


def test_disjoint_live_tp_and_dcp_coordinator_ranks_raise_plugin_error() -> None:
    """Reject live `GroupCoordinator` rank sets when neither nests inside the other."""
    runner = fake_runner(tp_size=4, dcp_size=2)
    with (
        mock.patch.object(torch.distributed, "is_available", return_value=True),
        mock.patch.object(torch.distributed, "is_initialized", return_value=True),
        mock.patch(
            "vllm.distributed.get_tp_group",
            return_value=SimpleNamespace(
                world_size=4, cpu_group=object(), ranks=[0, 1, 2, 3]
            ),
        ),
        mock.patch(
            "vllm.distributed.get_dcp_group",
            return_value=SimpleNamespace(
                world_size=2, cpu_group=object(), ranks=[2, 6]
            ),
        ),
        pytest.raises(MegakernelPluginError, match="not nested"),
    ):
        agreement_mod.agreement_group(DOMAIN_TP_X_DCP, runner)


@pytest.mark.parametrize(
    ("domain", "parallelism", "runner_kwargs"),
    [
        (
            DOMAIN_DCP,
            f"frozenset({{{PAR_TP!r}, {PAR_DCP!r}}})",
            {"tp_size": 1, "dcp_size": 2},
        ),
        (
            DOMAIN_TP_X_DCP,
            f"frozenset({{{PAR_TP!r}, {PAR_DCP!r}}})",
            {"tp_size": 1, "dcp_size": 2},
        ),
        (
            DOMAIN_TP_X_DCP,
            f"frozenset({{{PAR_TP!r}, {PAR_DCP!r}}})",
            {"tp_size": 2, "dcp_size": 1},
        ),
        (
            DOMAIN_KERNEL_MESH,
            f"frozenset({{{PAR_TP!r}, {PAR_DP!r}}})",
            {"tp_size": 1, "dp_size": 2},
        ),
    ],
)
def test_two_rank_gloo_consensus_across_collective_domains(
    tmp_path: Path,
    domain: str,
    parallelism: str,
    runner_kwargs: dict[str, int],
) -> None:
    """Run real 2-rank Gloo startup/bind consensus across non-TP domains."""
    old_spec = (
        "SPEC = TpuMonoSpec(\n"
        '    "toy", "1", create=create, decline_codes=("toy_declined",)\n'
        ")"
    )
    new_spec = (
        "SPEC = TpuMonoSpec(\n"
        f'    "toy", "1", create=create, collective_domain={domain!r},\n'
        f"    parallelism={parallelism}, decline_codes=('toy_declined',)\n"
        ")"
    )
    plugin_src = testkit.TOY_PLUGIN.replace(old_spec, new_spec)
    served = testkit.run_two_ranks(
        tmp_path / "serve",
        sources=[plugin_src, plugin_src],
        runner_kwargs=runner_kwargs,
    )
    assert [r["type"] for r in served] == ["ProgramDriver", "ProgramDriver"]

    declined = testkit.assert_rank_uniform_declines(
        tmp_path / "decline",
        sources=[plugin_src, plugin_src],
        options={"decline_rank": 1},
        runner_kwargs=runner_kwargs,
    )
    assert declined[0]["code"] == "peer_declined:toy_declined"
    assert declined[1]["code"] == "toy_declined"


@pytest.mark.parametrize(
    (
        "raw",
        "expected_block",
        "expected_interleave",
        "expected_mamba",
        "expected_floor",
    ),
    [
        ({}, None, None, None, None),
        (LayoutRequest(block_size=128), 128, None, None, 128),
        (
            _PluginLayoutDataclass(
                block_size=128, cp_kv_cache_interleave_size=64, mamba_slot_tokens=640
            ),
            128,
            64,
            640,
            640,
        ),
        (
            {
                "page_size": 128,
                "dcp_interleave_size": 128,
                "block_size": 128,
                "mamba_slot_tokens": 640,
            },
            128,
            128,
            640,
            640,
        ),
        ({"block_size": 128, "dcp_kv_cache_interleave_size": 64}, 128, 64, None, 128),
    ],
)
def test_layout_request_from_raw_normalizes_aliases(
    raw: object,
    expected_block: int | None,
    expected_interleave: int | None,
    expected_mamba: int | None,
    expected_floor: int | None,
) -> None:
    """Normalize `LayoutRequest`, dataclass, and downstream alias dicts."""
    parsed = salt_mod.parse_layout_request(raw)
    assert parsed.block_size == expected_block
    assert parsed.cp_kv_cache_interleave_size == expected_interleave
    assert parsed.mamba_slot_tokens == expected_mamba
    assert parsed.slot_tokens_floor == expected_floor


@pytest.mark.parametrize(
    ("bad_raw", "match"),
    [
        (["not_a_mapping"], "LayoutRequest, Mapping, or dataclass"),
        (_PluginLayoutDataclass, "LayoutRequest, Mapping, or dataclass"),
        ({"unknown_field": 128}, "unknown key 'unknown_field'"),
        ({"page_size": 64, "block_size": 128}, "conflicting values for block_size"),
        (
            {"cp_kv_cache_interleave_size": 64, "dcp_interleave_size": 128},
            "conflicting values for cp_kv_cache_interleave_size",
        ),
        ({"block_size": 0}, "positive int"),
        ({"block_size": -32}, "positive int"),
        ({"mamba_slot_tokens": True}, "positive int"),
        ({"block_size": 64, "cp_kv_cache_interleave_size": 128}, "not divisible by"),
    ],
)
def test_layout_request_from_raw_rejects_invalid_inputs(
    bad_raw: object, match: str
) -> None:
    """Reject malformed keys, types, booleans, and non-divisible interleave sizes."""
    with pytest.raises(MegakernelConfigError, match=match):
        salt_mod.parse_layout_request(bad_raw)


def test_compile_cache_salt_and_resolve_layout_request_positive_and_negative() -> None:
    """Verify deterministic salt across `LayoutRequest`, dataclass, dict, and errors."""
    cfg = fake_vllm_config(megakernel={"enabled": True})
    expected_digest_len = 16
    req_typed = LayoutRequest(
        block_size=128, cp_kv_cache_interleave_size=64, mamba_slot_tokens=640
    )
    req_dc = _PluginLayoutDataclass(
        block_size=128, cp_kv_cache_interleave_size=64, mamba_slot_tokens=640
    )
    req_dict = {
        "block_size": 128,
        "cp_kv_cache_interleave_size": 64,
        "mamba_slot_tokens": 640,
    }
    salts: list[str] = []
    for ret_val in (req_typed, req_dc, req_dict):
        spec = TpuMonoSpec(
            "salt_eq", "1", create="m:c", layout_request=lambda *_a, v=ret_val: v
        )
        selected = (SimpleNamespace(enabled=True), (config_mod.PluginRef("dummy"), {}))
        with (
            mock.patch.object(salt_mod, "select_plugin", return_value=selected),
            mock.patch.object(
                salt_mod, "import_plugin", return_value=(spec, ModuleType("dummy"))
            ),
        ):
            digest = salt_mod.compile_cache_salt(cfg)
            assert digest is not None
            assert len(digest) == expected_digest_len
            salts.append(digest)
    assert salts[0] == salts[1] == salts[2]

    spec_none = TpuMonoSpec(
        "salt_none", "1", create="m:c", layout_request=lambda *_a: None
    )
    selected = (SimpleNamespace(enabled=True), (config_mod.PluginRef("dummy"), {}))
    with (
        mock.patch.object(salt_mod, "select_plugin", return_value=selected),
        mock.patch.object(
            salt_mod, "import_plugin", return_value=(spec_none, ModuleType("dummy"))
        ),
    ):
        assert salt_mod.compile_cache_salt(cfg) is None

    spec_bad = TpuMonoSpec(
        "salt_bad", "1", create="m:c", layout_request=lambda *_a: "invalid"
    )
    with (
        mock.patch.object(salt_mod, "select_plugin", return_value=selected),
        mock.patch.object(
            salt_mod, "import_plugin", return_value=(spec_bad, ModuleType("dummy"))
        ),
        pytest.raises(
            MegakernelConfigError, match="LayoutRequest, Mapping, or dataclass instance"
        ),
    ):
        salt_mod.resolve_layout_request(cfg)

    with (
        mock.patch.object(salt_mod, "select_plugin", return_value=selected),
        mock.patch.object(salt_mod, "missing_template", return_value="arch"),
    ):
        assert salt_mod.resolve_plugin_hook(cfg, "layout_request") is None


def test_apply_layout_request_updates_cache_and_parallel_configs() -> None:
    """Apply `LayoutRequest` onto `vllm_config`, preserving user overrides safely."""
    req = {
        "page_size": 128,
        "dcp_interleave_size": 128,
        "mamba_slot_tokens": 640,
    }
    spec = TpuMonoSpec("lay", "1", create="m:c", layout_request=lambda *_a: req)
    cfg = fake_vllm_config(
        megakernel={"enabled": True},
        block_size=16,
        cp_kv_cache_interleave_size=16,
    )
    expected_block = 128
    expected_mamba_floor = 640
    with _patched_plugin(spec):
        applied = hooks_mod.apply_layout_request(cfg)
        applied_again = hooks_mod.apply_layout_request(cfg)
    assert applied == LayoutRequest(
        block_size=expected_block,
        cp_kv_cache_interleave_size=expected_block,
        mamba_slot_tokens=expected_mamba_floor,
    )
    assert applied_again == applied
    assert cfg.cache_config.block_size == expected_block
    assert cfg.parallel_config.cp_kv_cache_interleave_size == expected_block
    assert cfg.parallel_config.dcp_kv_cache_interleave_size == expected_block
    assert "_megakernel_layout_applied" not in vars(cfg)

    user_block = 32
    default_interleave = 1
    user_cfg = fake_vllm_config(
        megakernel={"enabled": True},
        block_size=user_block,
        user_specified_block_size=True,
        cp_kv_cache_interleave_size=default_interleave,
    )
    with _patched_plugin(spec):
        hooks_mod.apply_layout_request(user_cfg)
    assert user_cfg.cache_config.block_size == user_block
    assert user_cfg.parallel_config.cp_kv_cache_interleave_size == default_interleave

    conflict_cfg = fake_vllm_config(
        megakernel={"enabled": True},
        block_size=16,
        cp_kv_cache_interleave_size=32,
    )
    with _patched_plugin(spec), pytest.raises(MegakernelConfigError, match="conflicts"):
        hooks_mod.apply_layout_request(conflict_cfg)

    no_par_cfg = fake_vllm_config(megakernel={"enabled": True}, block_size=16)
    no_par_cfg.parallel_config = None
    with (
        _patched_plugin(spec),
        pytest.raises(MegakernelConfigError, match="parallel_config is None"),
    ):
        hooks_mod.apply_layout_request(no_par_cfg)

    disabled_block = 16
    disabled_cfg = fake_vllm_config(
        megakernel={"enabled": False}, block_size=disabled_block
    )
    assert hooks_mod.apply_layout_request(disabled_cfg) is None
    assert disabled_cfg.cache_config.block_size == disabled_block


@pytest.mark.parametrize(
    ("override_mode", "initial_block", "expected_final_block"),
    [
        ("plugin_floor", 16, 640),
        ("user_override", 32, 32),
    ],
)
def test_block_slot_derivation_applies_plugin_slot_tokens_floor(
    override_mode: str, initial_block: int, expected_final_block: int
) -> None:
    """Exercise `apply_layout_request` + `update_tpu_block_size_and_slot_config`."""
    pytest.importorskip("vllm")
    block_utils_mod = importlib.import_module(
        "vllm_torchtpu.platforms.tpu_block_size_utils"
    )
    spec = TpuMonoSpec(
        "lay_slot",
        "1",
        create="m:c",
        layout_request=lambda *_a: LayoutRequest(block_size=128, mamba_slot_tokens=640),
    )
    cfg = fake_vllm_config(
        megakernel={"enabled": True},
        block_size=initial_block,
        user_specified_block_size=(override_mode == "user_override"),
        kv_cache_dtype=torch.bfloat16,
    )
    with _patched_plugin(spec):
        layout_req = hooks_mod.apply_layout_request(cfg)
        block_utils_mod.update_tpu_block_size_and_slot_config(
            cfg, _FakeAttentionBackend, layout_request=layout_req
        )
    assert cfg.cache_config.block_size == expected_final_block


class _HookObserver:
    """Top-level weight-hook test spy supporting 2-arg, 3-arg, `*args`, and kw-only."""

    def __init__(self) -> None:
        """Initialize recorded call log."""
        self.seen: list[str] = []

    def two_arg(self, name: str, _tensor: torch.Tensor) -> None:
        """Observe `(name, tensor)` without options."""
        self.seen.append(name)

    def three_arg(
        self, name: str, _tensor: torch.Tensor, opts: dict[str, object]
    ) -> None:
        """Observe `(name, tensor, opts)`."""
        self.seen.append(f"{name}:{opts.get('tag', 'none')}")

    def varargs(self, *args: object) -> None:
        """Observe positional `*args` including options."""
        expected_arity = 3
        assert len(args) == expected_arity
        self.seen.append(str(args[0]))

    def kwonly(self, name: str, _tensor: torch.Tensor, *, flag: bool = True) -> None:
        """Observe 2 positional args when 3rd parameter is keyword-only."""
        assert flag
        self.seen.append(name)

    def consume_w0(
        self, name: str, _tensor: torch.Tensor, _opts: dict[str, object]
    ) -> object:
        """Consume `w0` and pass remaining weights through."""
        self.seen.append(name)
        return WEIGHT_CONSUMED if name == "w0" else None

    def consume_shard(
        self, name: str, _tensor: torch.Tensor, _opts: dict[str, object]
    ) -> object:
        """Consume `consumed_shard` and pass remaining weights through."""
        self.seen.append(name)
        return WEIGHT_CONSUMED if name == "consumed_shard" else None


@pytest.mark.parametrize(
    ("hook_kind", "expected_remaining", "expected_w0_val"),
    [
        ("two_arg", ["w0", "w1"], 1.0),
        ("three_arg", ["w0", "w1"], 1.0),
        ("varargs", ["w0", "w1"], 1.0),
        ("kwonly", ["w0", "w1"], 1.0),
        ("consume_w0", ["w1"], None),
    ],
)
def test_load_tensor_stream_signatures_and_weight_consumed_sentinel(
    hook_kind: str,
    expected_remaining: list[str],
    expected_w0_val: float | None,
) -> None:
    """Support 2-arg, 3-arg, `*args`, kw-only, and `WEIGHT_CONSUMED`."""
    observer = _HookObserver()
    hook_fn = getattr(observer, hook_kind)
    spec = TpuMonoSpec("obs", "1", create="m:c", load_tensor=hook_fn)
    cfg = fake_vllm_config(megakernel={"enabled": True})
    weights = [("w0", torch.tensor([1.0])), ("w1", torch.tensor([2.0]))]
    with _patched_plugin(spec, options={"tag": "ok"}):
        pairs = dict(hooks_mod.load_tensor_stream(cfg, weights))
    assert list(pairs) == expected_remaining
    assert len(observer.seen) == len(weights)
    if expected_w0_val is not None:
        assert float(pairs["w0"].item()) == expected_w0_val


def test_load_tensor_spec_laziness_and_fingerprint() -> None:
    """Enforce lazy string `load_tensor` and `layout_request` and hash fingerprint."""
    spec_lazy = TpuMonoSpec(
        "obs_test",
        "1",
        create="obs_mod:create",
        layout_request=".obs_mod:layout_req",
        load_tensor=".obs_mod:load_tensor",
    )
    spec_plain = TpuMonoSpec("obs_test", "1", create="obs_mod:create")
    testkit.check_spec(spec_lazy, lazy_create=True)

    spec_eager_load = TpuMonoSpec(
        "obs_test",
        "1",
        create="obs_mod:create",
        load_tensor=lambda *_a: None,
    )
    with pytest.raises(AssertionError, match="load_tensor"):
        testkit.check_spec(spec_eager_load, lazy_create=True)

    spec_eager_layout = TpuMonoSpec(
        "obs_test",
        "1",
        create="obs_mod:create",
        layout_request=lambda *_a: None,
    )
    with pytest.raises(AssertionError, match="layout_request"):
        testkit.check_spec(spec_eager_layout, lazy_create=True)

    fp_plain = fp_mod.plugin_fingerprint(
        module="m",
        attr="SPEC",
        plugin=spec_plain,
        source_hash="h",
        config={},
        raw_env="",
    )
    fp_lazy = fp_mod.plugin_fingerprint(
        module="m",
        attr="SPEC",
        plugin=spec_lazy,
        source_hash="h",
        config={},
        raw_env="",
    )
    assert fp_plain != fp_lazy


@pytest.mark.parametrize("loader_kind", ["get_all_weights", "load_weights_only"])
def test_observe_model_loader_wraps_and_restores_loaders(loader_kind: str) -> None:
    """Intercept both `get_all_weights` and `load_weights` loaders cleanly."""
    observer = _HookObserver()
    raw = [
        ("layer.0.w13_weight_scale_2", torch.tensor([0.5])),
        ("consumed_shard", torch.tensor([1.5])),
        ("layer.0.weight", torch.tensor([2.5])),
    ]
    loader: _GetAllWeightsLoader | _LoadWeightsOnlyLoader = (
        _GetAllWeightsLoader(raw)
        if loader_kind == "get_all_weights"
        else _LoadWeightsOnlyLoader(raw)
    )
    disabled_cfg = fake_vllm_config(megakernel={"enabled": False})
    with hooks_mod.observe_model_loader(disabled_cfg, loader):
        assert vars(loader) == {"raw": raw}

    spec_no_hook = TpuMonoSpec("obs_none", "1", create="m:c", load_tensor=None)
    enabled_cfg = fake_vllm_config(megakernel={"enabled": True})
    with (
        _patched_plugin(spec_no_hook),
        hooks_mod.observe_model_loader(enabled_cfg, loader),
    ):
        assert vars(loader) == {"raw": raw}

    spec = TpuMonoSpec("obs", "1", create="m:c", load_tensor=observer.consume_shard)
    with _patched_plugin(spec):
        with hooks_mod.observe_model_loader(enabled_cfg, loader):
            model = loader.load_model(enabled_cfg, enabled_cfg.model_config)
        assert vars(loader) == {"raw": raw}

    assert observer.seen == [
        "layer.0.w13_weight_scale_2",
        "consumed_shard",
        "layer.0.weight",
    ]
    assert "consumed_shard" not in model.loaded
    assert "layer.0.weight" in model.loaded


@pytest.mark.parametrize("loader_kind", ["get_all_weights", "load_weights_only"])
def test_observe_model_loader_restores_attributes_when_load_raises(
    loader_kind: str,
) -> None:
    """Restore loader and model attributes via `finally` when weight loading fails."""
    raw = [("w0", torch.tensor([1.0]))]
    loader: _GetAllWeightsLoader | _LoadWeightsOnlyLoader = (
        _GetAllWeightsLoader(raw)
        if loader_kind == "get_all_weights"
        else _LoadWeightsOnlyLoader(raw)
    )

    def _raise_on_w0(
        name: str, _tensor: torch.Tensor, _opts: dict[str, object]
    ) -> None:
        """Raise a RuntimeError when observing `w0`."""
        msg = f"boom on {name}"
        raise RuntimeError(msg)

    spec = TpuMonoSpec("obs_err", "1", create="m:c", load_tensor=_raise_on_w0)
    enabled_cfg = fake_vllm_config(megakernel={"enabled": True})
    with _patched_plugin(spec):
        with (
            pytest.raises(RuntimeError, match="boom on w0"),
            hooks_mod.observe_model_loader(enabled_cfg, loader),
        ):
            loader.load_model(enabled_cfg, enabled_cfg.model_config)
        assert vars(loader) == {"raw": raw}


def test_shorthand_for_arch_prioritizes_constrained_hf_fields_and_rejects_bad() -> None:
    """Prefer matching `hf_fields` across insertion orders and reject mismatches."""
    sh_base = Shorthand(
        module="m.base",
        architectures=("ArchA",),
        model_types=(),
        legacy_keys=frozenset(),
        hf_fields=(),
    )
    sh_spec = Shorthand(
        module="m.spec",
        architectures=("ArchA",),
        model_types=(),
        legacy_keys=frozenset(),
        hf_fields=(("hidden_size", 8192), ("num_hidden_layers", 92)),
    )
    for mapping in (
        {"base": sh_base, "special": sh_spec},
        {"special": sh_spec, "base": sh_base},
    ):
        with mock.patch.dict(config_mod.SHORTHANDS, mapping, clear=True):
            hf_match = SimpleNamespace(
                architectures=["ArchA"], hidden_size=8192, num_hidden_layers=92
            )
            assert config_mod.shorthand_for_arch(hf_match) == "special"

            hf_nested = {
                "architectures": ["ArchA"],
                "text_config": {"hidden_size": 8192, "num_hidden_layers": 92},
            }
            assert config_mod.shorthand_for_arch(hf_nested) == "special"

            hf_fallback = SimpleNamespace(
                architectures=["ArchA"], hidden_size=4096, num_hidden_layers=60
            )
            assert config_mod.shorthand_for_arch(hf_fallback) == "base"

    with mock.patch.dict(config_mod.SHORTHANDS, {"special": sh_spec}, clear=True):
        assert config_mod.shorthand_for_arch(None) is None
        assert (
            config_mod.shorthand_for_arch(SimpleNamespace(architectures=["ArchA"]))
            is None
        )
        assert (
            config_mod.shorthand_for_arch(
                SimpleNamespace(
                    architectures=["ArchA"], hidden_size=4096, num_hidden_layers=92
                )
            )
            is None
        )

    with pytest.raises(MegakernelConfigError, match="hf_fields"):
        Shorthand(
            module="m.bad",
            architectures=("ArchA",),
            model_types=(),
            legacy_keys=frozenset(),
            hf_fields=(("", 92),),
        )


@pytest.mark.parametrize(
    "domain",
    [DOMAIN_TP, DOMAIN_DCP, DOMAIN_TP_X_DCP, DOMAIN_KERNEL_MESH],
)
def test_spec_accepts_all_collective_domains_and_optional_hooks(
    domain: str,
) -> None:
    """Accept all four collective domains and optional lazy hook references."""
    spec = TpuMonoSpec(
        "t",
        "1",
        create="m:c",
        collective_domain=domain,
        layout_request="pkg.mod:layout_req",
        load_tensor="pkg.mod:load_tensor",
    )
    assert spec.collective_domain == domain
    assert spec.layout_request == "pkg.mod:layout_req"
    assert spec.load_tensor == "pkg.mod:load_tensor"


@pytest.mark.parametrize(
    ("bad_field", "bad_val", "match"),
    [
        ("collective_domain", "dp", "collective_domain='dp'"),
        ("collective_domain", "", "collective_domain=''"),
        (
            "layout_request",
            "bad_ref",
            "layout_request must be a callable or 'module:attr'",
        ),
        (
            "load_tensor",
            "bad_ref",
            "load_tensor must be a callable or 'module:attr'",
        ),
        ("layout_request", 123, "layout_request must be a callable"),
        ("load_tensor", 123, "load_tensor must be a callable"),
    ],
)
def test_spec_rejects_invalid_domain_and_hook_fields(
    bad_field: str, bad_val: object, match: str
) -> None:
    """Reject unknown collective domains and malformed hook fields on TpuMonoSpec."""
    kw: dict[str, Any] = {bad_field: bad_val}
    with pytest.raises(MegakernelConfigError, match=match):
        TpuMonoSpec("t", "1", create="m:c", **kw)


@pytest.mark.parametrize(
    ("kwargs", "expected_floor", "expected_dict"),
    [
        ({}, None, {}),
        ({"block_size": 128}, 128, {"block_size": 128}),
        ({"mamba_slot_tokens": 512}, 512, {"mamba_slot_tokens": 512}),
        (
            {
                "block_size": 128,
                "cp_kv_cache_interleave_size": 64,
                "mamba_slot_tokens": 640,
            },
            640,
            {
                "block_size": 128,
                "cp_kv_cache_interleave_size": 64,
                "mamba_slot_tokens": 640,
            },
        ),
    ],
)
def test_layout_request_validates_fields_and_computes_slot_tokens_floor(
    kwargs: dict[str, int],
    expected_floor: int | None,
    expected_dict: dict[str, int],
) -> None:
    """Validate positive LayoutRequest fields, floor computation, and to_dict."""
    req = LayoutRequest(**kwargs)
    assert req.slot_tokens_floor == expected_floor
    assert req.to_dict() == expected_dict
    field_name = "block_size"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(req, field_name, 64)


@pytest.mark.parametrize(
    ("bad_kwargs", "match"),
    [
        ({"block_size": 0}, "positive int"),
        ({"block_size": -16}, "positive int"),
        ({"block_size": True}, "positive int"),
        ({"cp_kv_cache_interleave_size": False}, "positive int"),
        ({"mamba_slot_tokens": "640"}, "positive int"),
        ({"block_size": 128, "cp_kv_cache_interleave_size": 48}, "not divisible by"),
    ],
)
def test_layout_request_rejects_invalid_fields(
    bad_kwargs: dict[str, Any], match: str
) -> None:
    """Reject non-positive, boolean, non-int, or non-divisible LayoutRequest fields."""
    with pytest.raises(MegakernelConfigError, match=match):
        LayoutRequest(**bad_kwargs)


_ORIG_TORCH_DEVICE = torch.device


def _cpu_device(dev: object) -> torch.device:
    """Map `"tpu"` device strings to CPU during unit tests."""
    return (
        _ORIG_TORCH_DEVICE("cpu") if str(dev) == "tpu" else _ORIG_TORCH_DEVICE(str(dev))
    )


def test_tpu_model_runner_load_model_wires_observe_model_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `TPUModelRunner.load_model` wires `observe_model_loader`."""
    pytest.importorskip("vllm")
    if "vllm_torchtpu.runner.mm_encoder_manager" not in sys.modules:
        monkeypatch.setitem(
            sys.modules, "vllm_torchtpu.runner.mm_encoder_manager", mock.MagicMock()
        )
    backends_utils = importlib.import_module("vllm.v1.attention.backends.utils")
    with mock.patch.object(
        backends_utils, "get_kv_cache_layout", create=True, return_value=None
    ):
        tpu_runner_mod = importlib.import_module("vllm_torchtpu.runner.tpu_runner")
    observer = _HookObserver()
    raw = [
        ("layer.0.w13_weight_scale_2", torch.tensor([0.5])),
        ("consumed_shard", torch.tensor([1.5])),
        ("layer.0.weight", torch.tensor([2.5])),
    ]
    loader = _GetAllWeightsLoader(raw)
    spec = TpuMonoSpec(
        "runner_obs", "1", create="m:c", load_tensor=observer.consume_shard
    )
    cfg = fake_vllm_config(megakernel={"enabled": True})
    cfg.load_config = SimpleNamespace(load_format="auto")
    cfg.model_config.quantization = None
    runner_stub = SimpleNamespace(
        vllm_config=cfg,
        model_config=cfg.model_config,
        load_config=cfg.load_config,
        device_config=SimpleNamespace(device=SimpleNamespace(type="cpu")),
        mesh=None,
        is_pooling_model=False,
        _is_async_drafter=False,
        _initialize_pallas_kernels=lambda: None,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    with (
        mock.patch("torch.device", side_effect=_cpu_device),
        mock.patch.object(tpu_runner_mod, "get_model_loader", return_value=loader),
        mock.patch.object(
            tpu_runner_mod, "get_tpu_quantization_config", return_value=None
        ),
        mock.patch.object(tpu_runner_mod, "validate_online_fp8", return_value=None),
        _patched_plugin(spec),
    ):
        tpu_runner_mod.TPUModelRunner.load_model(runner_stub)
    assert observer.seen == [
        "layer.0.w13_weight_scale_2",
        "consumed_shard",
        "layer.0.weight",
    ]
    assert "layer.0.w13_weight_scale_2" not in runner_stub.model.loaded
    assert "consumed_shard" not in runner_stub.model.loaded
    assert "layer.0.weight" in runner_stub.model.loaded
