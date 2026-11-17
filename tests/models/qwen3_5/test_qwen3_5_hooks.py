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

# The real lookup, before the autouse fixture below replaces it.
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


# Qwen3.5-397B-A17B's text config. One FP8 expert slot in one MoE layer takes
# 3 x hidden x moe_intermediate = 3 x 4096 x 1024 B.
HIDDEN, INTER, EXPERTS, LAYERS = 4096, 1024, 512, 60
SLOT_LAYER_BYTES = 3 * HIDDEN * INTER
# Main's placement only permutes experts: 512 / EP8 = 64 slots per rank, 16
# fewer than the 80 the budget was measured with, in each of the 60 layers:
# 960 slot-layers, so a TP8 + DCP4 decode leg gets 33.5e9 B + 960 x 3 x 4096 x
# 1024 B = 45,579,595,520 B here. (The validation entries' record runs placed
# 68 slots per rank with an expert table that has copies, a later change, and
# ran with 42,509,364,992 B.)
CREDIT_EP8 = (80 - 64) * LAYERS * SLOT_LAYER_BYTES


def _vllm_config(
    arch: str = QWEN35,
    *,
    kv_role: str | None = "kv_consumer",
    kv_cache_memory_bytes: int | None = None,
    prefix_match_unit: int | None = None,
    tp: int = 8,
    dp: int = 1,
    dcp: int = 4,
    pcp: int = 1,
    pp: int = 1,
    ep: bool = True,
    max_num_seqs: int = 8,
    additional: dict | None = None,
) -> SimpleNamespace:
    text = SimpleNamespace(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTER,
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
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs, scheduler_cls=None),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
        ),
        kv_transfer_config=(
            None if kv_role is None else SimpleNamespace(kv_role=kv_role)
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


def test_tp8_dcp4_decode_leg_gets_the_calibrated_budget():
    cfg = _run_hook(_vllm_config())
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + CREDIT_EP8
    assert cfg.cache_config.kv_cache_memory_bytes == 45_579_595_520


def test_dcp1_with_more_than_16_seqs_uses_the_smaller_base():
    cfg = _run_hook(_vllm_config(tp=2, dp=4, dcp=1, max_num_seqs=28))
    # EP = TP x DP = 8, so the same 64 slots per rank.
    assert cfg.cache_config.kv_cache_memory_bytes == 32_500_000_000 + CREDIT_EP8


@pytest.mark.parametrize("dcp,seqs", [(1, 16), (4, 28), (1, 1)])
def test_other_shapes_use_the_larger_base(dcp, seqs):
    cfg = _run_hook(_vllm_config(dcp=dcp, max_num_seqs=seqs))
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + CREDIT_EP8


@pytest.mark.parametrize(
    "layout",
    [
        # A pipeline-parallel engine: every rank was charged for all 60 layers,
        # so PP8 x TP1 holds 512 slots per rank and got about -293.6e9 B.
        dict(kv_role=None, tp=1, pp=8, dcp=1, max_num_seqs=32),
        dict(kv_role="kv_consumer", tp=1, pp=8, dcp=1, max_num_seqs=32),
        dict(kv_role=None, tp=4, pp=2, dcp=1, max_num_seqs=32),
        dict(tp=8, pp=2),
        # Experts over 4 ranks: 128 slots per rank, about -2.7e9 B.
        dict(tp=4, dcp=1),
        dict(tp=2, dp=2, dcp=1),
        # Experts over 16 ranks.
        dict(tp=8, dp=2),
        # Main's nightly TP1 x DP8 decode leg.
        dict(tp=1, dp=8, dcp=1, max_num_seqs=32),
        # No expert parallelism: the slot count does not apply.
        dict(ep=False),
        # PCP: vLLM folds it into the EP size.
        dict(tp=8, pcp=2),
    ],
    ids=[
        "pp8-tp1-mixed",
        "pp8-tp1-decode",
        "pp2-tp4-mixed",
        "pp2-tp8-decode",
        "ep4",
        "dp2-tp2",
        "ep16",
        "dp8-tp1",
        "no-ep",
        "pcp2-tp8",
    ],
)
def test_unmeasured_layouts_keep_mains_behaviour(layout):
    """The budget was measured on TP8 and DP4 x TP2 decode legs with expert
    parallelism and no PP or PCP. Any other layout stays on
    gpu_memory_utilization, where the slot credit could go negative."""
    cfg = _run_hook(_vllm_config(**layout))
    assert cfg.cache_config.kv_cache_memory_bytes is None


@pytest.mark.parametrize("dcp,seqs", [(1, 28), (1, 16), (4, 16), (4, 1)])
@pytest.mark.parametrize("tp,dp", [(8, 1), (2, 4)])
def test_measured_layouts_get_a_positive_budget(tp, dp, dcp, seqs):
    cfg = _run_hook(_vllm_config(tp=tp, dp=dp, dcp=dcp, max_num_seqs=seqs))
    assert cfg.cache_config.kv_cache_memory_bytes > 0


@pytest.mark.parametrize("kv_role", ["kv_consumer", None, "kv_producer", "kv_both"])
def test_explicit_kv_cache_memory_bytes_is_kept(kv_role):
    cfg = _run_hook(_vllm_config(kv_role=kv_role, kv_cache_memory_bytes=39_500_000_000))
    assert cfg.cache_config.kv_cache_memory_bytes == 39_500_000_000


def test_kv_producer_stays_on_gpu_memory_utilization():
    cfg = _run_hook(_vllm_config(kv_role="kv_producer", tp=1, dcp=1, max_num_seqs=32))
    assert cfg.cache_config.kv_cache_memory_bytes is None


@pytest.mark.parametrize("kv_role", [None, "kv_both"])
def test_mixed_engine_keeps_mains_behaviour(kv_role):
    """The budget was measured on decode legs only."""
    cfg = _run_hook(_vllm_config(kv_role=kv_role))
    assert cfg.cache_config.kv_cache_memory_bytes is None


def test_a_text_config_without_moe_fields_fails_loudly():
    """A calibrated checkpoint whose text config lacks the MoE fields fails at
    engine start rather than getting a budget without the slot credit."""
    cfg = _vllm_config()
    del cfg.model_config.hf_text_config.moe_intermediate_size
    with pytest.raises(AttributeError, match="moe_intermediate_size"):
        _run_hook(cfg)


# Main's nightly Qwen3.5-397B-A17B configs (models/qwen3_5/README.md): C1, a
# TP1 x DP8 engine with expert parallelism, and C3's TP1 x PCP8 prefill and
# TP1 x DP8 decode legs. None of them was measured with the defaults.
NIGHTLY_CONFIGS = {
    "c1-interleaved": dict(kv_role=None, tp=1, dp=8, dcp=1, max_num_seqs=256),
    "c3-prefill": dict(kv_role="kv_producer", tp=1, pcp=8, dcp=1, max_num_seqs=8),
    "c3-decode": dict(kv_role="kv_consumer", tp=1, dp=8, dcp=1, max_num_seqs=32),
}


@pytest.mark.parametrize("name", sorted(NIGHTLY_CONFIGS))
def test_mains_nightly_configs_keep_mains_defaults(name):
    cfg = _run_hook(_vllm_config(**NIGHTLY_CONFIGS[name]))
    assert cfg.cache_config.kv_cache_memory_bytes is None
    assert cfg.cache_config.prefix_match_unit is None


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
    cfg = _vllm_config()
    VllmConfig.try_verify_and_update_config(cfg)
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + CREDIT_EP8


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
    cfg = _vllm_config(
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
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + CREDIT_EP8


@pytest.mark.parametrize("arch", OTHER_ARCHS)
def test_other_models_are_not_checked(restore_config_map, arch):
    from vllm.config import VllmConfig

    register_models()
    cfg = _vllm_config(arch, additional={"tpu_typo": {}})
    VllmConfig.try_verify_and_update_config(cfg)


# The calibrated budget applies only where it was measured: Qwen3.5-397B-A17B's
# checkpoint shape on TPU7x. Anything else keeps main's gpu_memory_utilization.


def test_calibrated_shape_on_tpu7x_gets_the_budget():
    cfg = _run_hook(_vllm_config())
    assert cfg.cache_config.kv_cache_memory_bytes == 45_579_595_520


@pytest.mark.parametrize(
    "field, value",
    [("num_experts", 256), ("hidden_size", 2048), ("num_hidden_layers", 40)],
)
def test_other_checkpoint_of_the_architecture_keeps_mains_behaviour(field, value):
    cfg = _vllm_config()
    setattr(cfg.model_config.hf_text_config, field, value)
    assert _run_hook(cfg).cache_config.kv_cache_memory_bytes is None


@pytest.mark.parametrize("chip", ["v6e", "v5e", "v5p", "7", None])
def test_smaller_or_unknown_device_keeps_mains_behaviour(monkeypatch, chip):
    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    assert _run_hook(_vllm_config()).cache_config.kv_cache_memory_bytes is None


def test_explicit_budget_is_kept_on_any_device(monkeypatch):
    monkeypatch.setattr(hooks, "_chip_family", lambda: "v6e")
    cfg = _run_hook(_vllm_config(kv_cache_memory_bytes=12_000_000_000))
    assert cfg.cache_config.kv_cache_memory_bytes == 12_000_000_000


@pytest.fixture
def accelerator_type(monkeypatch):
    """Sets the accelerator type tpu_info reads; its chip lookup is cached."""

    def use(value):
        monkeypatch.setattr(tpu_info, "get_tpu_type", lambda: value)
        tpu_info.find_chip_version.cache_clear()

    yield use
    tpu_info.find_chip_version.cache_clear()


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
    default), so it never gets the calibrated budget."""
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


@pytest.mark.parametrize("chip", ["v6e", "v5p", None])
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
    """The hook's view of one validation-entry leg, from its serve arguments."""
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
    )


# The KV budget each point's decode leg gets on this tree: TP8 + DCP4 legs the
# auto budget with main's permutation placement; #1's DP4 x TP2 leg passes
# --kv-cache-memory-bytes, which is kept.
DECODE_KV_BYTES = {
    "p01": 39_500_000_000,
    "p05": 33_500_000_000 + CREDIT_EP8,
    "p08": 33_500_000_000 + CREDIT_EP8,
    "p10": 33_500_000_000 + CREDIT_EP8,
    "p12": 33_500_000_000 + CREDIT_EP8,
}


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_validation_points_keep_their_defaults(path):
    """Every point's prefill leg gets the 128-token unit and stays on
    gpu_memory_utilization; every decode leg keeps the default unit and gets
    its budget: the measured legs are unchanged."""
    with path.open() as f:
        legs = yaml.safe_load(f)["legs"]
    prefill = _run_hook(_leg_config(legs["prefill"]))
    decode = _run_hook(_leg_config(legs["decode"]))
    assert prefill.kv_transfer_config.kv_role == "kv_producer"
    assert prefill.cache_config.prefix_match_unit == 128
    assert prefill.cache_config.kv_cache_memory_bytes is None
    assert decode.kv_transfer_config.kv_role == "kv_consumer"
    assert decode.cache_config.prefix_match_unit is None
    point = path.name.split("-")[0]
    assert decode.cache_config.kv_cache_memory_bytes == DECODE_KV_BYTES[point]
