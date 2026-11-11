# SPDX-License-Identifier: Apache-2.0
"""Served-setup contract: every field the megakernel setup reads directly from a vLLM
class exists in the INSTALLED vLLM.

The set6 CPU tests ran in an image with vLLM 0.32, while the served images carry the
pinned vLLM build. A field that only the newer class sets (``reduce_results``, read by
VLLM_TPU_MK_PLE_TRIM) raised inside the whole-step trace, the megakernel was disabled
and za8d served on the stock path (GSM8K 0.004). These tests read the installed classes'
sources, so run them in the served image:

    docker run ... <served za image> -c 'pip install -q pytest; cd /ov &&
        PYTHONPATH=/ov/src:/ov python3 -m pytest tests/megakernel_runtime/test_served_setup_smoke.py'
"""

from __future__ import annotations

import importlib
import inspect
import re

import pytest


def _fields(cls) -> set[str]:
    """Instance fields set in any class of the MRO (plain or annotated assignment,
    register_parameter / register_buffer), submodule / attribute names, and members."""
    have: set[str] = set()
    for klass in cls.__mro__:
        try:
            src = inspect.getsource(klass)
        except (OSError, TypeError):
            continue
        have |= set(re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=\n]+)?=(?!=)", src))
        have |= set(re.findall(r"register_(?:parameter|buffer)\(\s*[\"']([A-Za-z0-9_]+)", src))
        have |= set(re.findall(r"setattr\(\s*self\s*,\s*[\"']([A-Za-z0-9_]+)", src))
    have |= {n for n, _ in inspect.getmembers(cls)}
    return have


def _cls(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


# (class, fields read directly by the megakernel setup / served forward, where)
VLLM_CONTRACT = [
    (
        "vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding",
        {"tp_size", "shard_indices", "quant_method"},
        "qwen4_exp/model.py Qwen4ExpBackbone._merged_head_embed (VLLM_TPU_MK_PLE_TRIM)",
    ),
    (
        "vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead",
        {"tp_size", "shard_indices", "quant_method"},
        "megakernel_runtime/qwen38_draft.py MkDraft (lm_head; org_vocab_size via getattr)",
    ),
    (
        "vllm.model_executor.models.qwen3_next.Qwen3NextSparseMoeBlock",
        {"gate", "experts", "shared_expert", "shared_expert_gate"},
        "megakernel_runtime/qwen38_*weights.py, weight_collector.py (target + MTP MoE)",
    ),
    (
        "vllm.model_executor.models.qwen3_next.Qwen3NextMLP",
        {"gate_up_proj", "down_proj"},
        "megakernel_runtime/qwen38_weights.py _shared, weight_collector.py",
    ),
]


@pytest.mark.parametrize("path,needed,where", VLLM_CONTRACT, ids=[c[0].rsplit(".", 1)[1] for c in VLLM_CONTRACT])
def test_installed_vllm_class_has_the_fields_the_megakernel_reads(path, needed, where):
    missing = needed - _fields(_cls(path))
    assert not missing, f"{path} lacks {sorted(missing)} (read in {where})"


def test_installed_fp8_moe_registers_the_expert_tensors_the_packer_views():
    """`pack_expert_views_layer` aliases `experts.w13_weight` / `w2_weight` and their
    `*_scale_inv` (or `*_scale`) block scales: names the installed FP8 MoE method must
    register."""
    from vllm.model_executor.layers.quantization import fp8

    src = inspect.getsource(fp8)
    for name in ("w13_weight", "w2_weight"):
        assert re.search(rf"[\"']{name}[\"']", src), name
    # Literal, or `f"w13_{self.weight_scale_name}"` with weight_scale(_inv) as the name.
    assert re.search(r"[\"']w13_weight_scale(_inv)?[\"']", src) or (
        "w13_{self.weight_scale_name}" in src
        and re.search(r"weight_scale_name\s*=\s*\(?[^)]{0,300}?weight_scale", src)
    )


def test_megakernel_reads_optional_vllm_fields_through_getattr():
    """Fields some vLLM releases lack are read with getattr and the pinned default."""
    from vllm_torchtpu.megakernel_runtime import qwen38_draft
    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    merged = inspect.getsource(qwen4_model.Qwen4ExpBackbone._merged_head_embed)
    for field in ("reduce_results", "parallel_group", "use_fused_embedding"):
        assert f'getattr(emb, "{field}"' in merged, field
        assert f"emb.{field}" not in merged, field
    draft = inspect.getsource(qwen38_draft)
    assert "lm.org_vocab_size" not in draft


def test_served_hc_classes_pack_with_and_without_the_combine_term():
    """The final mixers (target tail, MTP) are GatedResidual(use_combine=False): no
    `block_inject_weight`. The packer must not read it unconditionally (4440019257)."""
    from vllm_torchtpu.megakernel_runtime import qwen38_weights

    src = inspect.getsource(qwen38_weights._hc_mix_operands)
    direct = re.findall(r"hc\.block_inject_weight\b", src)
    guarded = re.findall(r"getattr\(hc, \"block_inject_weight\"", src)
    assert not direct or guarded or "use_combine" in src, (
        "_hc_mix_operands reads block_inject_weight without a combine-term guard"
    )
