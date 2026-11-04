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
"""Fast runtime loader for preprocessed raw checkpoint arrays.

Loads pre-quantized, K-major transposed MoE expert tensors from ``experts/array_*``
and merged PLE n-gram embedding tables from ``misc/array_*`` directly into
model parameters using single-seek contiguous ``pread`` slices per rank.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.model_loader.preprocessed_format import (
    MANIFEST_FILENAME,
    find_matching_leaf,
    resolve_and_read_shard,
)

logger = logging.getLogger("vllm." + __name__)

# Upload at most 256 MiB per H2D transfer slice (matches ExpertParamStager).
_UPLOAD_CHUNK_BYTES = 256 << 20


def is_preprocessed_checkpoint_dir(path: str | Path | None) -> bool:
    """Return True if ``path`` is a local directory containing preprocessed arrays."""
    if not path:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    if (p / MANIFEST_FILENAME).is_file():
        return True
    experts_dir = p / "experts"
    if experts_dir.is_dir():
        for child in experts_dir.iterdir():
            if child.is_dir() and child.name.startswith("array_"):
                return True
    misc_dir = p / "misc"
    if misc_dir.is_dir():
        for child in misc_dir.iterdir():
            if child.is_dir() and child.name.startswith("array_"):
                return True
    return False


def resolve_preprocessed_checkpoint_dir(model_path: str | Path | None) -> Path | None:
    """Resolve the active preprocessed checkpoint directory from env or model_path."""
    if not envs.VLLM_TPU_ENABLE_PREPROCESSED_LOADER:
        return None
    override = envs.VLLM_TPU_PREPROCESSED_CHECKPOINT_DIR
    if override and is_preprocessed_checkpoint_dir(override):
        return Path(override)
    if model_path and is_preprocessed_checkpoint_dir(model_path):
        return Path(model_path)
    return None


def _upload_tensor_to_device(
    cpu_tensor: torch.Tensor,
    target_device: torch.device,
) -> torch.Tensor:
    """Transfer ``cpu_tensor`` to ``target_device`` in chunks and sync on TPU."""
    if target_device.type == "cpu" or cpu_tensor.device == target_device:
        return cpu_tensor.contiguous()
    if cpu_tensor.ndim == 0 or cpu_tensor.numel() == 0:
        dev_t = cpu_tensor.to(target_device)
    else:
        per_row = max(1, cpu_tensor[0].numel() * cpu_tensor.element_size())
        step = max(1, _UPLOAD_CHUNK_BYTES // per_row)
        chunks = [
            cpu_tensor[start : start + step].to(target_device)
            for start in range(0, cpu_tensor.shape[0], step)
        ]
        dev_t = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        del chunks
    if target_device.type == "tpu":
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors([dev_t], wait=True)
    return dev_t


def _is_mtp_draft_model(model: nn.Module) -> bool:
    """Return True if ``model`` is a standalone MTP draft model whose
    ``model.layers.<i>`` corresponds to checkpoint ``mtp.layers.<i>``.
    """
    cls_name = type(model).__name__
    inner_cls_name = type(getattr(model, "model", None)).__name__
    if cls_name.endswith("MTP") or "MultiTokenPredictor" in inner_cls_name:
        return True
    for m in model.modules():
        prefix = getattr(m, "prefix", None)
        if isinstance(prefix, str) and prefix.startswith("mtp."):
            return True
    return False


def _candidate_expert_prefixes(module_name: str, *, is_mtp: bool = False) -> list[str]:
    """Candidate ``array_<prefix>`` names for a ``RoutedExperts`` module."""
    candidates = [module_name]
    if module_name.endswith(".experts.routed_experts"):
        short = module_name[: -len(".routed_experts")]
        candidates.insert(0, short)
    if module_name.startswith("model.language_model."):
        stripped = "model." + module_name[len("model.language_model.") :]
        if stripped not in candidates:
            candidates.append(stripped)
        if stripped.endswith(".experts.routed_experts"):
            short = stripped[: -len(".routed_experts")]
            if short not in candidates:
                candidates.insert(0, short)
    if is_mtp:
        mtp_candidates: list[str] = []
        for cand in candidates:
            if cand.startswith("model.layers."):
                mtp_cand = "mtp.layers." + cand[len("model.layers.") :]
                if mtp_cand not in mtp_candidates:
                    mtp_candidates.append(mtp_cand)
        candidates = mtp_candidates + [c for c in candidates if c not in mtp_candidates]
    return candidates


def _is_routed_experts_module(module: nn.Module) -> bool:
    """Return True if `module` is a `RoutedExperts` instance (real or stand-in)."""
    try:
        from vllm.model_executor.layers.fused_moe import RoutedExperts

        if isinstance(module, RoutedExperts):
            return True
    except ImportError:
        pass
    return (
        type(module).__name__ == "RoutedExperts"
        or hasattr(module, "moe_config")
        and hasattr(module, "w13_weight")
        and hasattr(module, "w2_weight")
    )


def _load_routed_experts_from_preprocessed(
    model: nn.Module,
    ckpt_dir: Path,
) -> set[str]:
    """Load all ``RoutedExperts`` modules from ``<ckpt_dir>/experts/array_*``."""
    experts_root = ckpt_dir / "experts"
    if not experts_root.is_dir():
        return set()

    loaded_params: set[str] = set()
    num_modules = 0
    total_bytes = 0
    t0 = time.perf_counter()
    is_mtp = _is_mtp_draft_model(model)

    for module_name, module in model.named_modules():
        if not _is_routed_experts_module(module):
            continue

        matched_prefix: str | None = None
        for prefix in _candidate_expert_prefixes(module_name, is_mtp=is_mtp):
            if (experts_root / f"array_{prefix}.w13_weight").is_dir():
                matched_prefix = prefix
                break
        if matched_prefix is None:
            continue

        p_cfg = module.moe_config.moe_parallel_config
        ep_size = p_cfg.ep_size if p_cfg.use_ep else 1
        ep_rank = p_cfg.ep_rank if p_cfg.use_ep else 0
        tp_size = p_cfg.tp_size
        tp_rank = p_cfg.tp_rank

        if ep_size > 1:
            if envs.TPU_MOE_HIERARCHICAL_EP:
                raise ValueError(
                    "Preprocessed MoE expert arrays require linear expert placement "
                    "and are incompatible with TPU_MOE_HIERARCHICAL_EP=1."
                )
            if getattr(module.moe_config, "enable_eplb", False) or getattr(
                p_cfg, "enable_eplb", False
            ):
                raise ValueError(
                    "Preprocessed MoE expert arrays are incompatible with "
                    "--enable-eplb."
                )
            placement = getattr(
                module.moe_config,
                "expert_placement_strategy",
                getattr(p_cfg, "expert_placement_strategy", "linear"),
            )
            if placement != "linear":
                raise ValueError(
                    f"Preprocessed MoE expert arrays require "
                    f"expert_placement_strategy='linear', got {placement!r}."
                )

        target_device = module.w13_weight.data.device

        w13_dir = experts_root / f"array_{matched_prefix}.w13_weight"
        w2_dir = experts_root / f"array_{matched_prefix}.w2_weight"
        w13_s_dir = experts_root / f"array_{matched_prefix}.w13_weight_scale_inv"
        w2_s_dir = experts_root / f"array_{matched_prefix}.w2_weight_scale_inv"

        w13_cpu = resolve_and_read_shard(
            w13_dir,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            split_mode="paired_columns",
        )
        w2_cpu = resolve_and_read_shard(
            w2_dir,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            split_mode="rows",
        )
        w13_s_cpu = resolve_and_read_shard(
            w13_s_dir,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            split_mode="paired_columns",
        )
        w2_s_cpu = resolve_and_read_shard(
            w2_s_dir,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            split_mode="rows",
        )

        # Scales are stored as 3D [E_local, num_blocks, N] on disk; reshape to
        # the 4D [E_local, num_blocks, 1, N] layout expected by the GMM kernel.
        if w13_s_cpu.ndim == 3:
            w13_s_cpu = w13_s_cpu.unsqueeze(2).contiguous()
        if w2_s_cpu.ndim == 3:
            w2_s_cpu = w2_s_cpu.unsqueeze(2).contiguous()

        expected_e = module.w13_weight.shape[0]
        expected_h = module.w13_weight.shape[2]
        if w13_cpu.shape[0] != expected_e or w13_cpu.shape[1] != expected_h:
            raise ValueError(
                f"Preprocessed w13_weight shape {tuple(w13_cpu.shape)} in {w13_dir} "
                f"does not match RoutedExperts ({expected_e=}, {expected_h=})."
            )
        if w2_cpu.shape[0] != expected_e or w2_cpu.shape[2] != expected_h:
            raise ValueError(
                f"Preprocessed w2_weight shape {tuple(w2_cpu.shape)} in {w2_dir} "
                f"does not match RoutedExperts ({expected_e=}, {expected_h=})."
            )
        requant_block = envs.MOE_REQUANTIZE_BLOCK_SIZE
        if requant_block is not None:
            num_blocks_w13 = w13_s_cpu.shape[1]
            if num_blocks_w13 * requant_block != expected_h:
                raise ValueError(
                    f"Preprocessed scale shape {tuple(w13_s_cpu.shape)} in "
                    f"{w13_s_dir} is incompatible with "
                    f"MOE_REQUANTIZE_BLOCK_SIZE={requant_block} "
                    f"(expected {expected_h // requant_block} blocks along "
                    f"K={expected_h})."
                )

        total_bytes += (
            w13_cpu.numel() * w13_cpu.element_size()
            + w2_cpu.numel() * w2_cpu.element_size()
            + w13_s_cpu.numel() * w13_s_cpu.element_size()
            + w2_s_cpu.numel() * w2_s_cpu.element_size()
        )

        w13_dev = _upload_tensor_to_device(w13_cpu, target_device)
        w2_dev = _upload_tensor_to_device(w2_cpu, target_device)
        w13_s_dev = _upload_tensor_to_device(w13_s_cpu, target_device)
        w2_s_dev = _upload_tensor_to_device(w2_s_cpu, target_device)
        del w13_cpu, w2_cpu, w13_s_cpu, w2_s_cpu

        if target_device.type == "tpu":
            from vllm_torchtpu.utils import fresh_buffer

            w13_s_dev = fresh_buffer(w13_s_dev)
            w2_s_dev = fresh_buffer(w2_s_dev)

        for old_scale in ("w13_weight_scale", "w2_weight_scale"):
            if hasattr(module, old_scale):
                delattr(module, old_scale)

        module.w13_weight = nn.Parameter(w13_dev, requires_grad=False)
        module.w2_weight = nn.Parameter(w2_dev, requires_grad=False)
        module.w13_weight_scale_inv = nn.Parameter(w13_s_dev, requires_grad=False)
        module.w2_weight_scale_inv = nn.Parameter(w2_s_dev, requires_grad=False)
        module._tpu_preprocessed_loaded = True

        param_prefix = f"{module_name}." if module_name else ""
        loaded_params.update(
            {
                f"{param_prefix}w13_weight",
                f"{param_prefix}w2_weight",
                f"{param_prefix}w13_weight_scale_inv",
                f"{param_prefix}w2_weight_scale_inv",
            }
        )
        num_modules += 1

    if num_modules > 0:
        elapsed = time.perf_counter() - t0
        logger.info(
            "[preprocessed-loader] Loaded %d RoutedExperts modules (%.2f GiB) "
            "from %s in %.2f s (%.2f GiB/s)",
            num_modules,
            total_bytes / 2**30,
            experts_root,
            elapsed,
            (total_bytes / 2**30) / max(elapsed, 1e-9),
        )
    return loaded_params


def _load_ple_from_preprocessed(
    model: nn.Module,
    ckpt_dir: Path,
) -> set[str]:
    """Load merged PLE n-gram embedding tables from ``<ckpt_dir>/misc/array_*``."""
    misc_root = ckpt_dir / "misc"
    if not misc_root.is_dir():
        return set()

    loaded_params: set[str] = set()
    t0 = time.perf_counter()
    total_bytes = 0
    num_tables = 0

    for module_name, module in model.named_modules():
        table = getattr(module, "ngram_embedding", None)
        if table is None or not hasattr(table, "shard_indices"):
            continue
        # Only match Qwen4ExpNGramEmbedding modules (holding .ngram_embedding)
        if not hasattr(module, "layer_multipliers"):
            continue

        candidate_prefixes = [module_name]
        if module_name.startswith("model.language_model."):
            candidate_prefixes.append(
                "model." + module_name[len("model.language_model.") :]
            )
        matched_prefix: str | None = None
        for prefix in candidate_prefixes:
            if (misc_root / f"array_{prefix}.ngram_embedding.weight").is_dir():
                matched_prefix = prefix
                break
        if matched_prefix is None:
            continue

        weight_dir = misc_root / f"array_{matched_prefix}.ngram_embedding.weight"
        tp_size = int(getattr(table, "tp_size", 1))
        start_row = int(table.shard_indices.org_vocab_start_index)
        end_row = int(table.shard_indices.org_vocab_end_index)
        rows_per_rank = end_row - start_row
        tp_rank = start_row // rows_per_rank if rows_per_rank > 0 and tp_size > 1 else 0

        shard_cpu = resolve_and_read_shard(
            weight_dir,
            ep_size=1,
            ep_rank=0,
            tp_size=tp_size,
            tp_rank=tp_rank,
            row_sharded_tp=True,
        )
        total_bytes += shard_cpu.numel() * shard_cpu.element_size()

        weight_param = table.get_parameter("weight")
        # If table uses ResidentPLETable/HostPLETable (which tracks _owned_ranges),
        # route through weight_loader or copy directly into weight_param.
        if hasattr(table, "_owned_ranges") and hasattr(table, "_loaded_ranges"):
            table.weight_loader(
                weight_param,
                shard_cpu[: table.shard_indices.num_org_elements],
                checkpoint_start=start_row,
            )
        else:
            target_device = weight_param.data.device
            if target_device.type == "cpu":
                weight_param.data.copy_(shard_cpu)
            else:
                weight_param.data = _upload_tensor_to_device(shard_cpu, target_device)
        del shard_cpu

        param_prefix = f"{module_name}." if module_name else ""
        loaded_params.add(f"{param_prefix}ngram_embedding.weight")

        # Load optional weight_scale if present.
        scale_dir = misc_root / f"array_{matched_prefix}.ngram_embedding.weight_scale"
        if scale_dir.is_dir() and find_matching_leaf(scale_dir) is not None:
            scale_cpu = resolve_and_read_shard(scale_dir)
            if "weight_scale" in dict(table.named_parameters(recurse=False)):
                scale_param = table.get_parameter("weight_scale")
                scale_param.data.copy_(scale_cpu.reshape(scale_param.shape))
                loaded_params.add(f"{param_prefix}ngram_embedding.weight_scale")

        # Load optional hash buffers if exported to misc/array_*.
        for buf_name in (
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
        ):
            buf_dir = misc_root / f"array_{matched_prefix}.{buf_name}"
            if buf_dir.is_dir() and find_matching_leaf(buf_dir) is not None:
                buf_cpu = resolve_and_read_shard(buf_dir)
                buf = module.get_buffer(buf_name)
                buf.copy_(buf_cpu.reshape(buf.shape))
                loaded_params.add(f"{param_prefix}{buf_name}")

        # Mark parent Qwen4ExpPooledPLE as having received these weights if applicable.
        parent_ple_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
        if parent_ple_name:
            parent_ple = model.get_submodule(parent_ple_name)
            if hasattr(parent_ple, "_loaded") and isinstance(parent_ple._loaded, set):
                parent_ple._loaded.update(
                    {
                        "ple_embedding.ngram_embedding.weight",
                        "ple_embedding.ngram_embedding.weight_scale",
                        "ple_embedding.layer_multipliers",
                        "ple_embedding.ngram_heads_vocab_sizes",
                        "ple_embedding.ngram_heads_offsets",
                    }
                )

        num_tables += 1

    if num_tables > 0:
        elapsed = time.perf_counter() - t0
        logger.info(
            "[preprocessed-loader] Loaded %d PLE n-gram tables (%.2f GiB) "
            "from %s in %.2f s (%.2f GiB/s)",
            num_tables,
            total_bytes / 2**30,
            misc_root,
            elapsed,
            (total_bytes / 2**30) / max(elapsed, 1e-9),
        )
    return loaded_params


def load_preprocessed_arrays_into_model(
    model: nn.Module,
    checkpoint_dir: str | Path,
) -> set[str]:
    """Load preprocessed expert and PLE raw arrays into ``model``."""
    ckpt_path = Path(checkpoint_dir)
    loaded = _load_routed_experts_from_preprocessed(model, ckpt_path)
    loaded |= _load_ple_from_preprocessed(model, ckpt_path)
    return loaded
