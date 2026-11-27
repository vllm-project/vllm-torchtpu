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
"""TPU hardware, torus collective, architecture, dtype, and solver constants."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from collections.abc import Mapping

# Collective & TPU topology constants.
AR_PHASES: int = 2
DEFAULT_CORES_PER_CHIP: int = int(
    pltpu.ChipVersion.TPU_7X.num_physical_tensor_cores_per_chip
)
MIN_TORUS_DIM: int = 4
WRAPPED_LANES_PER_AXIS: int = 2
DEFAULT_ICI_LINKS_PER_CHIP: int = 6

DEFAULT_ICI_LINK_BW: float = 100.0e9
DEFAULT_ICI_HOP_LATENCY_S: float = 1.0e-6
# Nominal 32-bit element lane rate relative to BF16 matrix peak (1/128).
# Strict lower bounds divide by the bracket fast end (1/64) unless explicitly
# overridden on ChipSpec, guaranteeing bound <= estimate across the bracket.
DEFAULT_VECTOR_TO_BF16_RATIO: float = 1.0 / 128.0
VECTOR_TO_BF16_RATIO_BRACKET: tuple[float, float] = (1.0 / 256.0, 1.0 / 64.0)
# Elements per 32-bit vector word at 8-bit packing.
VECTOR_MAX_ELEMENTS_PER_WORD: int = 4
# Elements per 32-bit vector word for weights stored below one byte (4-bit elements).
VECTOR_SUB_BYTE_ELEMENTS_PER_WORD: int = 8
SUBCHANNEL_BLOCK_K_FLOOR: int = 512
DEFAULT_D2D_TO_ICI_LINK_MULT: float = 6.0
D2D_BW_BRACKET: tuple[float, float] = (600.0e9, 1200.0e9)
TPU7X_FP8_FLOPS_PER_CORE_BRACKET: tuple[float, float] = (2.300e15, 2.307e15)
TPU7X_HBM_BYTES_PER_CORE_MAX: int = 96 * (1 << 30)
DEFAULT_D2D_BW: float = DEFAULT_D2D_TO_ICI_LINK_MULT * DEFAULT_ICI_LINK_BW
DEFAULT_ICI_BW: float = (
    float(DEFAULT_ICI_LINKS_PER_CHIP)
    * DEFAULT_ICI_LINK_BW
    / float(DEFAULT_CORES_PER_CHIP)
)
DEFAULT_DCN_HOST_GBIT_PER_S: float = 400.0
DEFAULT_CHIPS_PER_HOST: int = 4
DEFAULT_PCIE_BW: float = 64.0e9
BITS_PER_BYTE: float = 8.0
DEFAULT_DCN_BW: float = (DEFAULT_DCN_HOST_GBIT_PER_S * 1.0e9 / BITS_PER_BYTE) / float(
    DEFAULT_CHIPS_PER_HOST
)
ICI_HOP_LATENCY_BRACKET_S: tuple[float, float] = (0.5e-6, 2.0e-6)
PCIE_BRACKET_BW: tuple[float, float] = (32.0e9, 64.0e9)
MXU_M_TILE_BRACKET: tuple[float, float] = (128.0, 256.0)
DCN_BRACKET_GBIT_PER_S: tuple[float, float] = (200.0, 800.0)
CHIP_COUNT_BRACKET: tuple[float, float] = (4.0, 4.0)

# Supported Cloud TPU7x 3D physical slice shapes (minimum provisionable slice is 2x2x1).
TOPOLOGY_AXES: dict[str, tuple[int, int, int]] = {
    "2x2x1": (2, 2, 1),
    "2x2x2": (2, 2, 2),
    "2x2x4": (2, 2, 4),
    "2x4x4": (2, 4, 4),
    "4x4x4": (4, 4, 4),
}

VALID_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "full_attention",
        "indexed_attention",
        "sparse_gqa",
        "attention",
        "gqa",
        "mla",
        "mla_attention",
        "csa2",
        "dsa",
        "dsa_attention",
        "deepseek_sparse_attention",
        "sparse_mla",
        "sliding_attention",
        "swa",
        "sliding_window_attention",
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
        "dense_mlp",
        "mlp",
        "moe",
        "mtp",
        "ple",
        "ffn_only",
    }
)

LINEAR_ATTENTION_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
    }
)

LAYER_TYPE_MAP: dict[str, str] = {
    "mla": "mla",
    "mla_attention": "mla",
    "csa2": "mla",
    "dsa": "dsa",
    "dsa_attention": "dsa",
    "deepseek_sparse_attention": "dsa",
    "sparse_mla": "dsa",
    "sliding_attention": "swa",
    "swa": "swa",
    "sliding_window_attention": "swa",
    "kda": "kda",
    "kimi_linear": "kda",
    "kda_attention": "kda",
    "linear_attention": "gdn",
    "gdn": "gdn",
    "gated_deltanet": "gdn",
    "mamba": "gdn",
    "gqa": "gqa",
    "sparse_gqa": "gqa",
    "indexed_attention": "gqa",
}

NON_FULL_ATTN_TOKENS: frozenset[str] = frozenset(
    LINEAR_ATTENTION_LAYER_TOKENS
    | {layer_tok for layer_tok, fam in LAYER_TYPE_MAP.items() if fam == "swa"}
    | {"ffn_only"}
)

# Unit scales and math constants.
ACTIVATION_BYTES_BF16: float = 2.0
MATMUL_FLOP_PER_MAC: float = 2.0
CAUSAL_MIDPOINT_HALF: float = 0.5
# Draft fc weight maps concatenated token embedding and target hidden state
# (two hidden-size vectors) to one hidden vector.
DRAFT_PROJECTION_HIDDEN_VECTORS: float = 2.0
UNIT_KILO: float = 1.0e3
UNIT_MEGA: float = 1.0e6
UNIT_GIGA: float = 1.0e9
UNIT_TERA: float = 1.0e12
UNIT_PETA: float = 1.0e15
UNIT_MILLI: float = 1.0e-3
UNIT_MILLIS_PER_SECOND: float = 1.0e3
UNIT_MICRO: float = 1.0e-6
BIN_KIBI: float = 1024.0
BIN_MEBI: float = 1048576.0
UNIT_GIBI: float = float(1 << 30)
PERCENT_SCALE: float = 100.0

# HF config loader & quantization constants.
FOUR_BITS: int = 4
EIGHT_BITS: int = 8
SIXTEEN_BITS: int = 16
WEIGHT_BLOCK_DIM_LEN: int = 2
FP8_BLOCK_SCALE_BYTES: float = 4.0
HTTP_OK: int = 200
HTTP_TIMEOUT_S: float = 10.0
DEFAULT_HTTP_PORT: int = 80
DEFAULT_HTTPS_PORT: int = 443

FP32_BYTES: float = 4.0
BF16_BYTES: float = 2.0
FP8_BYTES: float = 1.0
INT8_BYTES: float = 1.0
DEFAULT_SSM_STATE_BYTES: float = 4.0
FP4_BYTES: float = 0.5
MXFP4_BYTES: float = 0.5 + 1.0 / 32.0
NVFP4_BYTES: float = 0.5 + 1.0 / 16.0
MXFP8_BYTES: float = FP8_BYTES + FP8_BYTES / 32.0
FP8_UE8M0_BYTES: float = MXFP8_BYTES
FP8_UE8M0_32X32_BYTES: float = FP8_BYTES + FP8_BYTES / (32.0 * 32.0)
_INT4_PTH_PARTS: tuple[str, ...] = ("int4", "per", "token", "head")
INT4_PER_TOKEN_HEAD_KV_DTYPE: str = "_".join(_INT4_PTH_PARTS)
# Per-token 4-bit KV scale cache stores 8 FP32 scales per rank (32 B/tok)
# plus one FP32 scale per token for 4-bit sparse index keys (4 B/tok).
INT4_PER_TOKEN_HEAD_SCALE_BYTES: float = 8.0 * FP32_BYTES
INT4_INDEX_SCALE_BYTES: float = FP32_BYTES
# CSA2 packed indexer key cache: 128 B FP8 key plus 4 B UE8M0 scale per token.
CSA2_INDEX_RECORD_BYTES: float = 128.0 * FP8_BYTES + FP32_BYTES

BASE_DTYPE_BYTES: dict[str, float] = {
    "float32": FP32_BYTES,
    "fp32": FP32_BYTES,
    "bfloat16": BF16_BYTES,
    "bf16": BF16_BYTES,
    "float16": BF16_BYTES,
    "fp16": BF16_BYTES,
    "float8_e4m3fn": FP8_BYTES,
    "float8_e5m2": FP8_BYTES,
    "float8_e8m0fnu": FP8_BYTES,
    "e8m0": FP8_BYTES,
    "fp8": FP8_BYTES,
    "int8": INT8_BYTES,
    "uint8": INT8_BYTES,
}

WEIGHT_DTYPE_BYTES: dict[str, float] = {
    "bf16": BF16_BYTES,
    "fp8": FP8_BYTES,
    "mxfp8": MXFP8_BYTES,
}
EXPERT_DTYPE_BYTES: dict[str, float] = {
    "bf16": BF16_BYTES,
    "fp8": FP8_BYTES,
    "nvfp4": NVFP4_BYTES,
    "mxfp4": MXFP4_BYTES,
}
KV_DTYPE_BYTES: dict[str, float] = {
    "bf16": BF16_BYTES,
    "fp8": FP8_BYTES,
    "int8": INT8_BYTES,
    "fp4": FP4_BYTES,
    "int4": FP4_BYTES,
    INT4_PER_TOKEN_HEAD_KV_DTYPE: FP4_BYTES,
}
STATE_DTYPE_BYTES: dict[str, float] = {"fp32": FP32_BYTES, "bf16": BF16_BYTES}

WEIGHT_ONLY_QUANT_METHODS: frozenset[str] = frozenset(
    {"awq", "gptq", "gptq_marlin", "awq_marlin", "bitsandbytes", "hqq", "eetq"}
)

# Solver, HBM capacity, and chip hardware constants.
HBM_PACKING_NUMERATOR: float = 4.0
HBM_MIN_PACKING_DIVISOR: float = 0.5
SPARSE_MLA_SCALE_BYTES: float = 4.0
HEAD_DIM_PACKING_THRESHOLD: int = 64
GLU_MATMUL_COUNT: int = 3
NON_GATED_MATMUL_COUNT: int = 2
DEFAULT_MAX_INSTANCES: int = 16
DEFAULT_HBM_RESERVE_GB: float = 0.0
DEFAULT_PREFILL_CHUNK: int = 2048
EP_PAIR_RANKS: int = 2
MIN_INSTANCES: int = 1
MIN_DISAGG_INSTANCES: int = 2

EXAMPLE_CONCURRENCIES: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 256)
EXAMPLE_SLO_TIERS: dict[str, tuple[float | None, float | None]] = {
    "none": (None, None),
    "interactive": (5000.0, 33.33),
    "throughput": (5000.0, 66.67),
}


# Warning a serving plan carries when no layout fits the chip budget and HBM.
NO_FEASIBLE_LAYOUT_WARNING: str = (
    "No feasible layout met memory and latency constraints."
)


type SloVerdict = Literal["ruled_out", "estimate_meets", "undecided"]

SLO_VERDICT_RULED_OUT: SloVerdict = "ruled_out"
SLO_VERDICT_UNDECIDED: SloVerdict = "undecided"
SLO_VERDICT_ESTIMATE_MEETS: SloVerdict = "estimate_meets"
SLO_VERDICT_RANK: dict[str, int] = {
    SLO_VERDICT_RULED_OUT: 0,
    SLO_VERDICT_UNDECIDED: 1,
    SLO_VERDICT_ESTIMATE_MEETS: 2,
}
FRONTIER_2D_TIE_TOLERANCE: float = 1e-9

# Versioned solve JSON payload and top-K diversity selection constants.
SOLVE_SCHEMA: str = "vllm_torchtpu.roofline.solve/v2"
INSPECT_SCHEMA: str = "vllm_torchtpu.roofline.inspect/v1"
CHIP_SPEC_SCHEMA: str = "vllm_torchtpu.roofline.chip_spec/v1"
VALID_CHIP_PROVENANCE: frozenset[str] = frozenset(
    {"public_spec", "jax_tpu_info", "measured_microbench", "projected"}
)
MIN_TOP_LAYOUTS_FOR_FASTEST: int = 2
MIN_TOP_LAYOUTS_FOR_BOUND_BEST: int = 3
DEFAULT_TOP_K: int = 5
MIN_TOP_K: int = 1

PREFIX_RANGE_PAIR_LEN: int = 2


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Canonical 1:1 HuggingFace repo specification for a modeled architecture."""

    canonical_id: str
    hf_repos: tuple[str, ...]
    model_types: tuple[str, ...]
    architectures: tuple[str, ...]
    allowed_layer_kinds: frozenset[str]


# Closed-world registry of verified public HuggingFace models with authentic fixtures.
SUPPORTED_MODELS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "qwen3_5_moe": ModelSpec(
            canonical_id="qwen3_5_moe",
            hf_repos=(
                "Qwen/Qwen3.5-397B-A17B-FP8",
                "nvidia/Qwen3.5-397B-A17B-NVFP4",
            ),
            model_types=("qwen3_5_moe", "qwen3_5_moe_text"),
            architectures=(
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3_5MoeForCausalLM",
            ),
            allowed_layer_kinds=frozenset({"gqa", "gdn"}),
        ),
        "glm_moe_dsa": ModelSpec(
            canonical_id="glm_moe_dsa",
            hf_repos=(
                "zai-org/GLM-5.2-FP8",
                "nvidia/GLM-5.2-NVFP4",
            ),
            model_types=("glm_moe_dsa",),
            architectures=("GlmMoeDsaForCausalLM",),
            allowed_layer_kinds=frozenset({"mla", "dsa"}),
        ),
        "deepseek_v41": ModelSpec(
            canonical_id="deepseek_v41",
            hf_repos=("deepseek-ai/DeepSeek-V4.1-Flash",),
            model_types=("deepseek_v41", "deepseek_v41_text"),
            architectures=("DeepseekV41ForCausalLM",),
            allowed_layer_kinds=frozenset({"csa2", "mla", "dsa"}),
        ),
        "minimax_m3_vl": ModelSpec(
            canonical_id="minimax_m3_vl",
            hf_repos=("MiniMaxAI/MiniMax-M3.1-Flash",),
            model_types=("minimax_m3_vl", "minimax_m3_vl_text"),
            architectures=(
                "MiniMaxM3SparseForConditionalGeneration",
                "MiniMaxM3SparseForCausalLM",
            ),
            allowed_layer_kinds=frozenset({"gqa"}),
        ),
    }
)

SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    mt for spec in SUPPORTED_MODELS.values() for mt in spec.model_types
)

HF_REPO_TO_CANONICAL_ID: Mapping[str, str] = MappingProxyType(
    {
        repo.lower(): spec.canonical_id
        for spec in SUPPORTED_MODELS.values()
        for repo in spec.hf_repos
    }
)

MULTIMODAL_KEYS: tuple[str, ...] = (
    "vision_config",
    "audio_config",
    "video_config",
    "mm_projector_type",
    "image_token_index",
    "image_token_id",
    "video_token_id",
    "video_token_index",
    "vision_start_token_id",
    "vision_end_token_id",
    "language_model_only",
    "img_token_compression_config",
    "image_grid_pinpoints",
    "image_seq_length",
    "multimodal_projector_bias",
    "process_image_mode",
    "projector_hidden_act",
    "projector_hidden_size",
    "vision_feature_layer",
    "vision_feature_select_strategy",
)
BENIGN_HF_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "_name_or_path",
        "attention_bias",
        "attention_dropout",
        "auto_map",
        "bos_token_id",
        "cls_token_id",
        "compress_rope_theta",
        "dspark_noise_token_id",
        "dtype",
        "engram_compressed_vocab_size",
        "engram_pad_token_id",
        "engram_vocab_size",
        "eos_token_id",
        "ep_size",
        "hc_eps",
        "hc_sinkhorn_iters",
        "hidden_act",
        "id2label",
        "index_skip_topk_offset",
        "index_topk_freq",
        "index_topk_pattern",
        "indexer_rope_interleave",
        "initializer_range",
        "label2id",
        "layer_norm_eps",
        "mamba_ssm_dtype",
        "mlp_bias",
        "model_max_length",
        "moe_router_dtype",
        "n_group",
        "norm_topk_prob",
        "num_reward_heads",
        "output_router_logits",
        "pad_token_id",
        "partial_rotary_factor",
        "pretraining_tp",
        "qk_head_dim",
        "qk_norm_type",
        "rms_norm_eps",
        "rope_interleave",
        "rope_parameters",
        "rope_scaling",
        "rope_theta",
        "rotary_dim",
        "routed_scaling_factor",
        "router_aux_loss_coef",
        "scoring_func",
        "sep_token_id",
        "swiglu_alpha",
        "swiglu_limit",
        "tokenizer_class",
        "topk_group",
        "topk_method",
        "torch_dtype",
        "transformers_version",
        "unk_token_id",
        "use_cache",
        "use_gemma_norm",
        "use_qk_norm",
        "use_routing_bias",
    }
)
HYBRID_PATTERN_MAP: dict[str, str] = {
    "*": "full_attention",
    "M": "linear_attention",
    "-": "ffn_only",
    "E": "ffn_only",
}
STEP_FLOOR_CACHE_MAXSIZE: int = 32768
UNIT_COLL_CACHE_MAXSIZE: int = 32768
BLOCK_PARALLEL_ENABLED: int = 2

# Informational warning emitted when closed-loop think_time_s is zero.


UNDECIDED_SLO_WARNING: str = (
    "Recommended layout is UNDECIDED: physical lower bounds meet SLOs, "
    "but the uncalibrated serial estimate exceeds at least one SLO."
)

UNKNOWN_THINK_TIME_WARNING: str = (
    "Think time omitted: closed-loop concurrency throttling is not applied."
)

ACTIVATION_HULL_SCHEMA: str = "vllm_torchtpu.roofline.moe_activation_hull/v1"
# Float64 slope rounding scales as ulp(active) / spacing (~1e-13 for 512
# experts); 1e-9 experts/token absorbs JSON float formatting slack.
HULL_CONCAVITY_SLOPE_TOLERANCE: float = 1e-9
# Absorbs 6-decimal JSON rounding (0.5e-6 half-ULP with 2x margin,
# unit: experts) when comparing exported knots to the Bernoulli cap.
HULL_BERNOULLI_CAP_TOLERANCE: float = 1e-6

_SOLVER_ASSUMPTION_ROWS: tuple[tuple[str, str, str, object, str, str, str], ...] = (
    (
        "bound_ttft_envelope_formula",
        "",
        "assumption",
        "hit-rate-adjusted uncached prefill step floor",
        "formula",
        "TTFT bounds evaluate unpadded prefill at scenario prefix hit rate.",
        "solver",
    ),
    (
        "bound_tpot_residency",
        "",
        "assumption",
        "least fixed-point steady-state residency",
        "formula",
        "TPOT bounds evaluate bottom-up from batch 1 to physical HBM batch cap.",
        "solver",
    ),
    (
        "bound_capacity_ceiling",
        "",
        "assumption",
        "unpadded linear work-conservation",
        "req/s",
        (
            "pool_capacity_bound divides unpadded HBM, matmul, vector, "
            "and per-link-axis cuts."
        ),
        "solver",
    ),
    (
        "eff_prefill_rounding",
        "",
        "assumption",
        "floor((1 - h) * isl) for bounds; ceil(...) for estimates",
        "tokens",
        "Strict bounds floor uncached tokens; estimates include chunk overhead.",
        "modules",
    ),
    (
        "hbm_reserve_gb",
        "--hbm-reserve-gb",
        "constraint",
        DEFAULT_HBM_RESERVE_GB,
        "GB",
        "Optional HBM reserve subtracted from physical per-core memory.",
        "cli",
    ),
    (
        "prefill_chunk_tokens",
        "",
        "assumption",
        DEFAULT_PREFILL_CHUNK,
        "tokens",
        "Default chunk size for chunked prefill under mixed concurrency.",
        "constants",
    ),
    (
        "collective_latency_us",
        "--collective-latency-us",
        "calibration",
        0.0,
        "us",
        "Optional per-hop ICI calibration latency added only to estimates "
        "(0 means none added).",
        "cli",
    ),
    (
        "ep_uniform_routing",
        "--moe-activation",
        "constraint",
        "bernoulli",
        "mode",
        "Default routed MoE activation unless a supplied hull is active.",
        "occupancy",
    ),
)

SOLVER_ASSUMPTIONS: tuple[dict[str, object], ...] = tuple(
    {
        "name": name,
        "flag": flag,
        "kind": kind,
        "value": value,
        "unit": unit,
        "description": desc,
        "source": src,
    }
    for name, flag, kind, value, unit, desc, src in _SOLVER_ASSUMPTION_ROWS
)
