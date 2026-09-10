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
"""Load-time configurable FP8 for selected unquantized TPU Linear layers."""

import json
from dataclasses import dataclass, field
from fnmatch import fnmatchcase

import torch
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)

from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass
class OnlineFp8Policy:
    targets: list[str]
    exclude: list[str]
    weight_scheme: str = "per_channel"
    weight_block_size: tuple[int, int] | None = None
    matched: set[str] = field(default_factory=set)
    selected: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("tpu_online_fp8 must be an object")
        unknown = set(value) - {
            "enabled", "weight_scheme", "weight_block_size", "targets",
            "exclude"
        }
        if unknown:
            raise ValueError(
                f"Unknown tpu_online_fp8 fields: {sorted(unknown)}")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("tpu_online_fp8.enabled must be a boolean")
        if not enabled:
            return None
        scheme = value.get("weight_scheme", "per_channel")
        if scheme not in ("per_channel", "per_tensor", "per_block"):
            raise ValueError("tpu_online_fp8.weight_scheme must be "
                             "per_channel, per_tensor, or per_block")
        block = value.get("weight_block_size")
        if scheme == "per_block":
            if (not isinstance(block, list) or len(block) != 2
                    or any(type(v) is not int or v <= 0 for v in block)):
                raise ValueError("tpu_online_fp8.weight_block_size requires "
                                 "two positive integers [N_out, K_in]")
            if block[1] % 128:
                raise ValueError("Online FP8 K block size must be a multiple "
                                 "of 128 for TPU projection kernels")
            block = tuple(block)
        elif "weight_block_size" in value:
            raise ValueError("weight_block_size is only valid for per_block")
        targets = value.get("targets", [])
        exclude = value.get("exclude", [])
        for name, patterns in (("targets", targets), ("exclude", exclude)):
            if not isinstance(patterns, list) or any(
                    not isinstance(p, str) or not p or p.startswith("re:")
                    for p in patterns):
                raise ValueError(f"tpu_online_fp8.{name} requires a list of "
                                 "nonempty exact names or fnmatch patterns")
        if not targets:
            raise ValueError("tpu_online_fp8.targets must not be empty")
        return cls(list(dict.fromkeys(targets)), list(dict.fromkeys(exclude)),
                   scheme, block)

    def apply_mapper(self, mapper):
        self.targets = mapper.apply_list(self.targets)
        self.exclude = mapper.apply_list(self.exclude)

    def select(self, prefix, packed_mapping):
        parent, _, leaf = prefix.rpartition(".")
        aliases = [
            f"{parent}.{part}" if parent else part
            for part in packed_mapping.get(leaf, [leaf])
        ]
        choices = []
        for alias in aliases:
            names = (prefix, alias)
            hits = {
                p
                for p in self.targets if any(fnmatchcase(n, p) for n in names)
            }
            self.matched.update(hits)
            excluded = any(
                fnmatchcase(n, p) for n in names for p in self.exclude)
            choices.append(bool(hits) and not excluded)
        if any(choices) and not all(choices):
            raise ValueError(
                f"Partial fused-layer online FP8 selection: {prefix}; "
                f"select or exclude all of {aliases}")
        return all(choices), aliases


def attach_online_fp8(config, vllm_config):
    """Keep the checkpoint config intact; attach independent per-model state."""
    policy = OnlineFp8Policy.from_dict(
        vllm_config.additional_config.get("tpu_online_fp8"))
    if policy is not None:
        if config.get_name() not in ("fp8", "modelopt_fp4"):
            raise ValueError("tpu_online_fp8 supports FP8 and ModelOpt NVFP4 "
                             "checkpoint configurations")
        config.online_fp8_policy = policy
    return config


def map_online_fp8(config, mapper):
    policy = getattr(config, "online_fp8_policy", None)
    if policy is not None:
        policy.apply_mapper(mapper)


def resolve_online_fp8(config, layer, prefix, base_method):
    policy = getattr(config, "online_fp8_policy", None)
    if policy is None:
        return base_method
    selected, aliases = policy.select(prefix, config.packed_modules_mapping)
    if not selected:
        return base_method
    if not isinstance(layer, LinearBase):
        raise ValueError(
            f"Online FP8 target is not a supported Linear: {prefix}")
    if not isinstance(base_method, UnquantizedLinearMethod):
        raise ValueError(f"Online FP8 target {prefix} already has checkpoint "
                         f"method {type(base_method).__name__}")
    from vllm_torchtpu.layers.adapter.quantization.fp8 import (
        VllmFp8Config, VllmFp8LinearMethodTPU)
    online_config = VllmFp8Config(is_checkpoint_fp8_serialized=False,
                                  activation_scheme="dynamic",
                                  weight_block_size=None)
    method = VllmFp8LinearMethodTPU(online_config, prefix=prefix)
    method.online_fp8 = True
    method.online_fp8_policy = policy
    policy.selected[prefix] = aliases
    return method


def validate_online_fp8(model, config):
    """Audit once after loading, before compilation or accepting requests."""
    policy = getattr(config, "online_fp8_policy", None)
    if policy is None:
        return
    validated = set()
    for name, layer in model.named_modules():
        prefix = getattr(layer, "prefix", name)
        selected, aliases = policy.select(prefix,
                                          config.packed_modules_mapping)
        if not selected:
            continue
        method = getattr(layer, "quant_method", None)
        if not getattr(method, "online_fp8", False):
            raise ValueError(
                f"Online FP8 target {prefix} bypassed quantization "
                "dispatch or is an unsupported module")
        if not getattr(layer, "_tpu_online_fp8_processed", False):
            raise ValueError(f"Online FP8 target {prefix} was not processed")
        weight, scale = layer.weight, layer.weight_scale
        keep_layout = getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False)
        n_out = weight.shape[0] if keep_layout else weight.shape[1]
        k_in = weight.shape[1] if keep_layout else weight.shape[0]
        if policy.weight_scheme == "per_tensor":
            expected_shape = (1, )
            expected_block = (n_out, k_in)
        elif policy.weight_scheme == "per_block":
            block_n, block_k = policy.weight_block_size
            if n_out % block_n or k_in % block_k:
                raise ValueError(f"Invalid online FP8 block size for {prefix}")
            expected_shape = (1, k_in // block_k, 1, n_out)
            expected_block = (block_n, block_k)
        else:
            expected_shape = (n_out, )
            expected_block = (1, k_in)
        if (weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32
                or scale.shape != expected_shape
                or layer.weight_block_size != expected_block):
            raise ValueError(f"Invalid online FP8 runtime format for {prefix}")
        validated.add(prefix)
        logger.info(
            "TPU online FP8: %s",
            json.dumps({
                "module": prefix,
                "checkpoint_aliases": aliases,
                "checkpoint_dtype": "bfloat16",
                "runtime_dtype": "float8_e4m3fn",
                "weight_shape": list(weight.shape),
                "scale_shape": list(scale.shape),
                "weight_scheme": policy.weight_scheme,
                "weight_block_size": list(layer.weight_block_size),
                "converted": True,
            }))
    unmatched = set(policy.targets) - policy.matched
    missing = set(policy.selected) - validated
    if unmatched or missing or not validated:
        raise ValueError("Invalid online FP8 selection: "
                         f"unmatched targets={sorted(unmatched)}, "
                         f"unprocessed modules={sorted(missing)}, "
                         f"converted={len(validated)}")
    logger.info("TPU online FP8 validated %d runtime Linear layers",
                len(validated))


def quantize_online_fp8(weight, policy):
    """Quantize a loaded local [N, K] shard; expand only scales along N."""
    from vllm_torchtpu.layers.core.quantization import quantize_tensor

    n_out, k_in = weight.shape
    scheme = policy.weight_scheme
    weight = weight.to(torch.float32)
    if scheme == "per_tensor":
        from vllm_torchtpu.layers.core.quantization import _safe_inverse_scale

        # Keep the matrix layout: flattening tens of millions of elements
        # into one TPU lane dimension makes load-time compilation expensive.
        abs_max = weight.abs().amax(dim=1, keepdim=True).amax(dim=0,
                                                              keepdim=True)
        scale = abs_max * torch.tensor(
            1.0 / 448, dtype=torch.float32, device=weight.device)
        quantized = (weight * _safe_inverse_scale(scale)).clamp(-448, 448)
        return quantized.to(torch.float8_e4m3fn), scale.reshape(1), (n_out,
                                                                     k_in)
    if scheme == "per_channel":
        quantized, scale = quantize_tensor(weight, torch.float8_e4m3fn)
        return quantized, scale.reshape(n_out), (1, k_in)
    block_n, block_k = policy.weight_block_size
    if n_out % block_n or k_in % block_k:
        raise ValueError(
            f"Online FP8 block {policy.weight_block_size} must divide local "
            f"weight shape {(n_out, k_in)} after tensor-parallel sharding")
    grid = weight.reshape(n_out // block_n, block_n, k_in // block_k,
                          block_k).permute(0, 2, 1, 3)
    quantized, scale = quantize_tensor(
        grid.reshape(n_out // block_n, k_in // block_k, -1),
        torch.float8_e4m3fn)
    quantized = quantized.reshape(grid.shape).permute(0, 2, 1,
                                                      3).reshape(n_out, k_in)
    scale = scale.squeeze(-1).repeat_interleave(block_n, dim=0)
    scale = scale.t().unsqueeze(0).unsqueeze(2).contiguous()
    return quantized, scale, (block_n, block_k)
