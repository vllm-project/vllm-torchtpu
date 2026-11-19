# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp/prefix_granularity.py admission rules, without vLLM (loaded by path)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = (
    Path(__file__).resolve().parents[3]
    / "src" / "vllm_torchtpu" / "models" / "vllm" / "qwen4_exp" / "prefix_granularity.py"
)
_spec = importlib.util.spec_from_file_location("_test_prefix_granularity", _PATH)
G = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = G
_spec.loader.exec_module(G)

SERVED = dict(enable_prefix_caching=True, mamba_cache_mode="align", pooled=True,
              compress_ratio=4, num_speculative_tokens=0, dcp=1, pcp=1)


def test_unset_is_always_fine():
    assert G.prefix_match_unit_problems(None, **{**SERVED, "pooled": False}) == []


@pytest.mark.parametrize("unit", [128, 256, 640, 3200])
def test_served_prefill_admits_kernel_page_multiples(unit):
    assert G.prefix_match_unit_problems(unit, **SERVED) == []


@pytest.mark.parametrize("unit,frag", [(64, "QSA kernel page"), (96, "QSA kernel page"),
                                        (0, "positive"), (-128, "positive")])
def test_rejects_sub_page_units(unit, frag):
    problems = G.prefix_match_unit_problems(unit, **SERVED)
    assert problems and any(frag in p for p in problems)


@pytest.mark.parametrize("override,frag", [
    ({"enable_prefix_caching": False}, "enable-prefix-caching"),
    ({"mamba_cache_mode": "none"}, "mamba-cache-mode"),
    ({"pooled": False}, "unified pool"),
    ({"num_speculative_tokens": 3}, "MTP"),
    ({"dcp": 4}, "context parallelism"),
    ({"pcp": 2}, "context parallelism"),
    ({"compress_ratio": 256}, "indexer compression group"),
])
def test_rejects_unvalidated_engines(override, frag):
    problems = G.prefix_match_unit_problems(128, **{**SERVED, **override})
    assert any(frag in p for p in problems), problems


def _cfg(unit, **kw):
    return SimpleNamespace(
        cache_config=SimpleNamespace(prefix_match_unit=unit, enable_prefix_caching=True,
                                     mamba_cache_mode="align"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=kw.get("dcp", 1),
                                        prefill_context_parallel_size=1),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(indexer_compress_ratio=4)),
    )


def test_validate_raises_with_every_reason():
    G.validate_prefix_match_unit(_cfg(128), pooled=True, num_speculative_tokens=0)
    G.validate_prefix_match_unit(_cfg(None), pooled=False, num_speculative_tokens=3)
    with pytest.raises(ValueError, match="MTP"):
        G.validate_prefix_match_unit(_cfg(128), pooled=True, num_speculative_tokens=3)
    with pytest.raises(ValueError, match="context parallelism"):
        G.validate_prefix_match_unit(_cfg(128, dcp=4), pooled=True, num_speculative_tokens=0)


# q38 CACHE-3: F2's prefill-only MTP producer (mtp, K = 1) with VLLM_TPU_PREFIX_UNIT_ALLOW_F2_PRODUCER=1.

def test_f2_producer_admits_k1_only():
    assert G.prefix_match_unit_problems(128, **{**SERVED, "num_speculative_tokens": 1}, f2_producer=True) == []
    k3 = G.prefix_match_unit_problems(128, **{**SERVED, "num_speculative_tokens": 3}, f2_producer=True)
    assert any("MTP" in p for p in k3)


def test_non_producer_mtp_still_refused():
    problems = G.prefix_match_unit_problems(128, **{**SERVED, "num_speculative_tokens": 1})
    assert any("MTP" in p for p in problems)


def test_f2_producer_keeps_every_other_rule():
    for override, frag in (({"pooled": False}, "unified pool"), ({"dcp": 2}, "context parallelism"),
                           ({"mamba_cache_mode": "none"}, "mamba-cache-mode")):
        problems = G.prefix_match_unit_problems(128, **{**SERVED, "num_speculative_tokens": 1, **override},
                                                f2_producer=True)
        assert any(frag in p for p in problems), (override, problems)
    assert any("QSA kernel page" in p
               for p in G.prefix_match_unit_problems(64, **{**SERVED, "num_speculative_tokens": 1}, f2_producer=True))


def test_validate_passes_the_producer_flag_through():
    G.validate_prefix_match_unit(_cfg(128), pooled=True, num_speculative_tokens=1, f2_producer=True)
    with pytest.raises(ValueError, match="is not validated together with MTP on the same engine"):
        G.validate_prefix_match_unit(_cfg(128), pooled=True, num_speculative_tokens=1)


def test_config_wires_the_producer_flag_to_the_env_and_the_f2_predicate():
    """Source check (config.py needs vLLM to import): the call passes the env flag AND pd_prefill_mtp_producer."""
    src = (Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu" / "models" / "vllm" / "qwen4_exp" / "config.py").read_text()
    assert "f2_producer=bool(envs.VLLM_TPU_PREFIX_UNIT_ALLOW_F2_PRODUCER) and pd_prefill_mtp_producer(cfg)" in src
