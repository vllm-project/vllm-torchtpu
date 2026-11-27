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
"""Tests for the Qwen3.5 config hook (models/qwen3_5/hooks.py)."""

from __future__ import annotations

import copy
import json
import pathlib
from types import SimpleNamespace

import pytest
import yaml
from jax.experimental.pallas import tpu as pltpu
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    Qwen3_5ForConditionalGenerationConfig,
)

from vllm_torchtpu import additional_config_sections, tpu_info
from vllm_torchtpu.models.qwen3_5 import hooks, settings
from vllm_torchtpu.models.qwen3_5.hooks import TPUQwen35MoeConfig
from vllm_torchtpu.models.vllm import register_models

# The real lookup, before the autouse fixture replaces it.
_REAL_CHIP_FAMILY = hooks._chip_family
TPU7X = pltpu.ChipVersion.TPU_7X.value

QWEN35 = "Qwen3_5MoeForConditionalGeneration"


@pytest.fixture
def restore_config_map():
    saved = dict(MODELS_CONFIG_MAP)
    yield
    MODELS_CONFIG_MAP.clear()
    MODELS_CONFIG_MAP.update(saved)


def test_hook_is_registered_for_qwen3_5_only(restore_config_map):
    register_models()
    assert frozenset({QWEN35}) == settings.ARCHITECTURES
    assert MODELS_CONFIG_MAP[QWEN35] is TPUQwen35MoeConfig
    # vLLM's own Qwen3.5 hook stays in charge of the sibling architectures.
    for sibling in (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
    ):
        assert not issubclass(MODELS_CONFIG_MAP[sibling], TPUQwen35MoeConfig)


def test_hook_extends_vllms_qwen3_5_hook():
    assert issubclass(TPUQwen35MoeConfig, Qwen3_5ForConditionalGenerationConfig)


# Qwen3.5-397B-A17B's text config: the shape the hook checks.
HIDDEN, EXPERTS, LAYERS = 4096, 512, 60


def _vllm_config(
    arch: str = QWEN35,
    *,
    kv_role: str | None = None,
    kv_cache_memory_bytes: int | None = None,
    prefix_match_unit: int | None = None,
    tp: int = 1,
    dp: int = 1,
    dcp: int = 1,
    pcp: int = 1,
    pp: int = 1,
    ep: bool = False,
    max_num_seqs: int = 8,
    additional: dict | None = None,
    scheduler_cls: str | None = None,
) -> SimpleNamespace:
    """A Qwen3.5-397B-A17B engine config as the hook sees it; by default one
    engine without KV transfer or any parallelism."""
    text = SimpleNamespace(
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        num_hidden_layers=LAYERS,
        mamba_ssm_dtype="float32",
        rope_parameters={"mrope_section": [11, 11, 10], "rope_theta": 1e7},
    )
    model_config = SimpleNamespace(
        architecture=arch,
        architectures=[arch],
        hf_text_config=text,
        config_updated=False,
        is_hybrid=False,
        convert_type="none",
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            prefix_match_unit=prefix_match_unit,
            prefix_cache_retention_interval=0,
            mamba_ssm_cache_dtype="auto",
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            scheduler_cls=scheduler_cls,
            enable_chunked_prefill=True,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
        ),
        kv_transfer_config=(
            None
            if kv_role is None
            else SimpleNamespace(
                kv_role=kv_role,
                # vLLM's role properties: kv_both is both.
                is_kv_producer=kv_role in ("kv_producer", "kv_both"),
                is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
            )
        ),
        load_config=SimpleNamespace(load_format="auto"),
        additional_config={} if additional is None else additional,
    )


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    """The defaults were measured on TPU7x; tests of another device override."""
    monkeypatch.setattr(hooks, "_chip_family", lambda: TPU7X)


def _run_hook(cfg) -> SimpleNamespace:
    TPUQwen35MoeConfig.verify_and_update_config(cfg)
    return cfg


# Main's nightly Qwen3.5-397B-A17B configs (models/qwen3_5/README.md): C1, a
# TP1 x DP8 engine with expert parallelism, and C3's TP1 x PCP8 prefill and
# TP1 x DP8 decode legs. None of them was measured with the default.
NIGHTLY_CONFIGS = {
    "c1-interleaved": dict(kv_role=None, tp=1, dp=8, dcp=1, ep=True, max_num_seqs=64),
    "c3-prefill": dict(
        kv_role="kv_producer", tp=1, pcp=8, dcp=1, ep=True, max_num_seqs=8
    ),
    "c3-decode": dict(
        kv_role="kv_consumer", tp=1, dp=8, dcp=1, ep=True, max_num_seqs=32
    ),
}


@pytest.mark.parametrize("name", sorted(NIGHTLY_CONFIGS))
def test_mains_nightly_configs_keep_mains_defaults(name):
    cfg = _run_hook(_vllm_config(**NIGHTLY_CONFIGS[name]))
    assert cfg.cache_config.kv_cache_memory_bytes is None
    assert cfg.cache_config.prefix_match_unit is None
    assert "tpu_pp_scheduler" not in cfg.additional_config


def test_kv_producer_gets_a_128_token_prefix_match_unit():
    cfg = _run_hook(_measured_prefill_leg())
    assert cfg.cache_config.prefix_match_unit == 128


@pytest.mark.parametrize("kv_role", ["kv_consumer", None, "kv_both"])
def test_other_roles_keep_the_default_prefix_match_unit(kv_role):
    cfg = _run_hook(_vllm_config(kv_role=kv_role))
    assert cfg.cache_config.prefix_match_unit is None


@pytest.mark.parametrize("unit", [1, 64, 256])
def test_explicit_prefix_match_unit_is_kept(unit):
    # On the measured prefill leg, where the hook would otherwise set 128.
    cfg = _run_hook(_measured_prefill_leg(prefix_match_unit=unit))
    assert cfg.cache_config.prefix_match_unit == unit


def test_retention_interval_and_scheduler_are_not_touched():
    cfg = _vllm_config(kv_role="kv_producer")
    cfg.cache_config.prefix_cache_retention_interval = 1280
    _run_hook(cfg)
    assert cfg.cache_config.prefix_cache_retention_interval == 1280
    assert cfg.scheduler_config.scheduler_cls is None


def test_vllms_qwen3_5_check_still_runs():
    cfg = _run_hook(_vllm_config())
    assert cfg.cache_config.mamba_ssm_cache_dtype == "float32"


def test_unknown_setting_fails_at_engine_start():
    with pytest.raises(ValueError, match="Unknown Qwen3.5 setting"):
        _run_hook(_vllm_config(additional={"qwen3_5": {"typo": True}}))


def _snapshot(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _snapshot(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _snapshot(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_snapshot(v) for v in obj]
    return obj


# The Qwen3.5 siblings vLLM itself registers: the closest architectures our
# registration could capture by mistake.
OTHER_ARCHS = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
)


@pytest.mark.parametrize("arch", OTHER_ARCHS)
@pytest.mark.parametrize("kv_role", ["kv_consumer", "kv_producer", None])
def test_other_models_resolve_exactly_as_on_main(restore_config_map, arch, kv_role):
    """Base vs head: vLLM's own dispatch (VllmConfig.try_verify_and_update_config)
    over the config map main has (vLLM's Qwen3.5 hook for our architecture) and
    over the map after register_models(). A model that is not ours must come
    out identical, field for field."""
    from vllm.config import VllmConfig

    register_models()
    head_map = dict(MODELS_CONFIG_MAP)
    base_map = dict(head_map)
    for ours in settings.ARCHITECTURES:
        base_map[ours] = Qwen3_5ForConditionalGenerationConfig

    results = []
    for config_map in (base_map, head_map):
        MODELS_CONFIG_MAP.clear()
        MODELS_CONFIG_MAP.update(config_map)
        cfg = _vllm_config(arch, kv_role=kv_role)
        VllmConfig.try_verify_and_update_config(cfg)
        results.append(_snapshot(cfg))
    assert results[0] == results[1]


def test_our_model_resolves_through_the_hook(restore_config_map):
    from vllm.config import VllmConfig

    register_models()
    cfg = _measured_prefill_leg()
    VllmConfig.try_verify_and_update_config(cfg)
    assert cfg.cache_config.prefix_match_unit == 128


# Names no change will register: real section names (tpu_scheduler, ...) stop
# being "unread" once their change lands in the same tree.
@pytest.mark.parametrize("key", ["tpu_typo", "tpu_not_a_section"])
def test_unread_tpu_section_fails_at_engine_start(key):
    """A tpu_* section no reader in this build claims would be ignored by
    vLLM; the hook refuses it and names the sections it reads."""
    assert key not in additional_config_sections.KNOWN_SECTIONS
    with pytest.raises(ValueError, match=rf"section\(s\) \['{key}'\].*qwen3_5"):
        _run_hook(_vllm_config(additional={key: {"x": 1}}))


def test_known_and_foreign_sections_pass():
    cfg = _measured_prefill_leg(
        additional={
            "qwen3_5": {},
            "tpu_online_fp8": {},
            # Outside the qwen3_5 / tpu_* namespaces: main's keys and other
            # plugins' keys keep vLLM's pass-through behaviour.
            "abort_on_preemption": False,
            "megakernel": {},
            "phased_profiler_decode_only_kv_len_threshold": 1,
            "some_other_plugin": {"a": 1},
        }
    )
    _run_hook(cfg)
    assert cfg.cache_config.prefix_match_unit == 128


@pytest.mark.parametrize("arch", OTHER_ARCHS)
def test_other_models_are_not_checked(restore_config_map, arch):
    from vllm.config import VllmConfig

    register_models()
    cfg = _vllm_config(arch, additional={"tpu_typo": {}})
    VllmConfig.try_verify_and_update_config(cfg)


@pytest.mark.parametrize(
    "value, family",
    [
        ("tpu7x-8", TPU7X),
        ("v7x-16", TPU7X),
        ("v6e-8", "v6e"),
        ("v5litepod-4", "v5e"),
        ("tpu-unknown-8", None),
        ("", None),
        (None, None),
    ],
)
def test_chip_family_reads_the_accelerator_type(accelerator_type, value, family):
    """tpu_info resolves the chip, and an unknown one stays unknown (no TPU7x
    default), so it never gets the calibrated default."""
    accelerator_type(value)
    assert _REAL_CHIP_FAMILY() == family


def test_the_chip_lookup_is_read_once(accelerator_type, monkeypatch):
    calls = []
    monkeypatch.setattr(tpu_info, "get_tpu_type", lambda: calls.append(1) or "tpu7x-8")
    tpu_info.find_chip_version.cache_clear()
    assert [_REAL_CHIP_FAMILY() for _ in range(3)] == [TPU7X] * 3
    assert len(calls) == 1


# The 128-token prefix match unit applies only where it was measured: the
# validation entries' prefill legs (kv_producer, PP8, no PCP or DCP) of
# Qwen3.5-397B-A17B on TPU7x. Any other engine keeps main's unset unit, so the
# platform resolves the hash block size from the KV cache groups (whole
# scheduler blocks on main's PCP disagg integration runs).


def _measured_prefill_leg(**overrides) -> SimpleNamespace:
    leg = dict(kv_role="kv_producer", tp=1, dcp=1, pcp=1, pp=8, max_num_seqs=32)
    leg.update(overrides)
    return _vllm_config(**leg)


def _retention(cfg) -> bool | None:
    return cfg.additional_config.get("tpu_pp_scheduler", {}).get(
        "hybrid_state_retention"
    )


def test_measured_prefill_leg_keeps_hybrid_state_snapshots():
    cfg = _run_hook(_measured_prefill_leg())
    assert _retention(cfg) is True
    # The other options of the section are kept.
    cfg = _run_hook(
        _measured_prefill_leg(additional={"tpu_pp_scheduler": {"prefix_aware": True}})
    )
    assert cfg.additional_config["tpu_pp_scheduler"] == {
        "prefix_aware": True,
        "hybrid_state_retention": True,
    }


@pytest.mark.parametrize("value", [True, False])
def test_an_explicit_retention_value_is_kept(value):
    section = {"hybrid_state_retention": value}
    cfg = _run_hook(_measured_prefill_leg(additional={"tpu_pp_scheduler": section}))
    assert _retention(cfg) is value


@pytest.mark.parametrize(
    "overrides",
    [
        dict(pp=1),
        dict(pcp=2),
        dict(dcp=2),
        dict(kv_role="kv_consumer"),
        dict(kv_role="kv_both"),
        dict(kv_role=None),
        # Another scheduler class: no pipeline chunk scheduler to keep them.
        dict(scheduler_cls="x.Y"),
    ],
)
def test_other_engines_keep_no_hybrid_state_snapshots(overrides):
    assert _retention(_run_hook(_measured_prefill_leg(**overrides))) is None


def test_a_pp4_prefill_leg_of_another_checkpoint_keeps_no_snapshots():
    """A PP4 prefill leg of a smaller Qwen3.5 MoE checkpoint (the shape of
    Qwen3.5-35B-A3B) was not measured with retention."""
    cfg = _measured_prefill_leg(pp=4)
    text = cfg.model_config.hf_text_config
    text.num_experts, text.hidden_size, text.num_hidden_layers = 256, 2048, 40
    assert _retention(_run_hook(cfg)) is None


def test_other_device_keeps_no_hybrid_state_snapshots(monkeypatch):
    monkeypatch.setattr(hooks, "_chip_family", lambda: "v6e")
    assert _retention(_run_hook(_measured_prefill_leg())) is None


@pytest.mark.parametrize("pcp", [2, 4, 8])
def test_pcp_prefill_leg_keeps_mains_prefix_match_unit(pcp):
    cfg = _run_hook(_measured_prefill_leg(pcp=pcp, pp=1))
    assert cfg.cache_config.prefix_match_unit is None


@pytest.mark.parametrize("dcp", [2, 4])
def test_dcp_prefill_leg_keeps_mains_prefix_match_unit(dcp):
    cfg = _run_hook(_measured_prefill_leg(dcp=dcp))
    assert cfg.cache_config.prefix_match_unit is None


@pytest.mark.parametrize(
    "field, value",
    [("num_experts", 256), ("hidden_size", 2048), ("num_hidden_layers", 40)],
)
def test_other_checkpoint_prefill_leg_keeps_mains_prefix_match_unit(field, value):
    cfg = _measured_prefill_leg()
    setattr(cfg.model_config.hf_text_config, field, value)
    assert _run_hook(cfg).cache_config.prefix_match_unit is None


@pytest.mark.parametrize("chip", ["v6e", "v5e", "v5p", "7", None])
def test_other_device_prefill_leg_keeps_mains_prefix_match_unit(monkeypatch, chip):
    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    assert _run_hook(_measured_prefill_leg()).cache_config.prefix_match_unit is None


def test_explicit_unit_on_a_pcp_prefill_leg_is_kept():
    cfg = _run_hook(_measured_prefill_leg(pcp=4, pp=1, prefix_match_unit=256))
    assert cfg.cache_config.prefix_match_unit == 256


@pytest.fixture
def restore_pcp_patch_targets():
    """Undoes main's PCP block-size patch after the test. The patch rebinds
    functions and methods in these vLLM namespaces for the whole process."""
    import sys

    from vllm.v1.core import (
        kv_cache_coordinator,
        kv_cache_manager,
        kv_cache_planning,
        kv_cache_utils,
    )
    from vllm.v1.engine import core as engine_core
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    namespaces = [
        kv_cache_coordinator,
        kv_cache_manager,
        kv_cache_planning,
        kv_cache_utils,
        engine_core,
        engine_core.EngineCoreProc,
        FullAttentionSpec,
    ]
    if "vllm.v1.kv_offload.base" in sys.modules:
        namespaces.append(sys.modules["vllm.v1.kv_offload.base"])
    saved = [(ns, dict(vars(ns))) for ns in namespaces]
    yield
    for ns, before in saved:
        for name in set(vars(ns)) - set(before):
            delattr(ns, name)
        for name, value in before.items():
            if vars(ns).get(name) is not value:
                setattr(ns, name, value)


@pytest.mark.parametrize(
    "shape",
    [
        {},
        # Qwen3.5-35B-A3B, the checkpoint main's PCP disagg integration runs use.
        {"num_experts": 256, "hidden_size": 2048, "num_hidden_layers": 40},
    ],
    ids=["397b", "35b"],
)
@pytest.mark.usefixtures("restore_pcp_patch_targets")
def test_pcp4_prefill_leg_hashes_in_whole_scheduler_blocks(shape):
    """Main's PCP4 disagg integration check: after the hook, the platform's PCP
    block-size resolution gives hash_block_size == scheduler_block_size on a
    Qwen3.5 prefill leg (768-token groups at PCP4 resolve to 3072 and 3072)."""
    import torch
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout

    from vllm_torchtpu import _patch_vllm_hybrid_pcp_block_sizes

    cfg = _measured_prefill_leg(pcp=4, pp=1)
    for field, value in shape.items():
        setattr(cfg.model_config.hf_text_config, field, value)
    _run_hook(cfg)
    cfg.cache_config.block_size = 768
    cfg.cache_config.enable_prefix_caching = True
    cfg.cache_config.num_gpu_blocks_override = None
    cfg.cache_config.get_resolved_kv_cache_layout = lambda: KVCacheLayout.LBNHC
    cfg.attention_config = SimpleNamespace(hisparse_config=None)

    _patch_vllm_hybrid_pcp_block_sizes()
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

    attention = FullAttentionSpec(
        block_size=768, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
    )
    gdn = MambaSpec(
        block_size=768,
        shapes=((3, 8192),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["gdn"], gdn),
        ],
    )
    assert resolve_kv_cache_block_sizes(kv_cache_config, cfg) == (3072, 3072)


VALIDATION = pathlib.Path(hooks.__file__).parent / "validation"
POINTS = sorted(VALIDATION.glob("p[0-9][0-9]-*.yaml"))


def _flag(args: list[str], name: str, default: str | None) -> str | None:
    return args[args.index(name) + 1] if name in args else default


def _leg_config(leg: dict) -> SimpleNamespace:
    """The hook's view of one validation-entry leg, from its serve arguments
    and its additional_config."""
    args = leg["serve_args"]
    transfer = _flag(args, "--kv-transfer-config", None)
    explicit = _flag(args, "--kv-cache-memory-bytes", None)
    return _vllm_config(
        kv_role=None if transfer is None else json.loads(transfer)["kv_role"],
        kv_cache_memory_bytes=None if explicit is None else int(explicit),
        tp=int(_flag(args, "--tensor-parallel-size", "1")),
        dp=int(_flag(args, "--data-parallel-size", "1")),
        dcp=int(_flag(args, "--decode-context-parallel-size", "1")),
        pcp=int(_flag(args, "--prefill-context-parallel-size", "1")),
        pp=int(_flag(args, "--pipeline-parallel-size", "1")),
        ep="--enable-expert-parallel" in args,
        max_num_seqs=int(_flag(args, "--max-num-seqs", None)),
        additional=copy.deepcopy(leg.get("additional_config", {})),
    )


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_validation_points_keep_their_defaults(path):
    """Every point's prefill leg gets the 128-token unit; every decode leg
    keeps the default unit. The KV cache size stays what each leg passes: p01's
    decode leg its --kv-cache-memory-bytes, every other leg none (its
    --gpu-memory-utilization)."""
    with path.open() as f:
        legs = yaml.safe_load(f)["legs"]
    prefill = _run_hook(_leg_config(legs["prefill"]))
    decode = _run_hook(_leg_config(legs["decode"]))
    assert prefill.kv_transfer_config.kv_role == "kv_producer"
    assert prefill.cache_config.prefix_match_unit == 128
    # The prefill legs keep hybrid-state snapshots, as they ran, next to the
    # options their section sets; the decode legs' sections are unchanged.
    assert prefill.additional_config["tpu_pp_scheduler"] == {
        **legs["prefill"].get("additional_config", {}).get("tpu_pp_scheduler", {}),
        "hybrid_state_retention": True,
    }
    assert decode.additional_config == legs["decode"].get("additional_config", {})
    assert "tpu_pp_scheduler" not in decode.additional_config
    assert prefill.cache_config.kv_cache_memory_bytes is None
    assert decode.kv_transfer_config.kv_role == "kv_consumer"
    assert decode.cache_config.prefix_match_unit is None
    explicit = _flag(legs["decode"]["serve_args"], "--kv-cache-memory-bytes", None)
    assert decode.cache_config.kv_cache_memory_bytes == (
        None if explicit is None else int(explicit)
    )
