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

from typing import TYPE_CHECKING, Literal

from jax.experimental.pallas import tpu as pltpu

if TYPE_CHECKING:
    from .chips import ChipSpec

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
DEFAULT_VECTOR_TO_BF16_RATIO: float = 1.0 / 128.0
VECTOR_TO_BF16_RATIO_BRACKET: tuple[float, float] = (1.0 / 256.0, 1.0 / 64.0)
# Elements per 32-bit vector word at the widest packing (8-bit elements).
VECTOR_MAX_ELEMENTS_PER_WORD: int = 4
# Elements per 32-bit vector word for weights stored below one byte (4-bit elements).
VECTOR_SUB_BYTE_ELEMENTS_PER_WORD: int = 8
SUBCHANNEL_BLOCK_K_FLOOR: int = 512
TPU7X_ICI_LINKS_PER_CHIP: int = DEFAULT_ICI_LINKS_PER_CHIP
DEFAULT_D2D_BW: float = float(TPU7X_ICI_LINKS_PER_CHIP) * DEFAULT_ICI_LINK_BW
DEFAULT_ICI_BW: float = DEFAULT_D2D_BW / float(DEFAULT_CORES_PER_CHIP)
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
PAGE_TABLE_BRACKET_BYTES: tuple[float, float] = (262_144.0, 1_048_576.0)
RPA_PAGE_SIZE_TOKENS: int = 64
RPA_PAGE_SIZE_BRACKET_TOKENS: tuple[float, float] = (16.0, 64.0)
GPU_MEM_UTIL_BRACKET: tuple[float, float] = (0.80, 0.95)

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
        "attention",
        "gqa",
        "mla",
        "mla_attention",
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

WEIGHT_DTYPE_BYTES: dict[str, float] = {"bf16": BF16_BYTES, "fp8": FP8_BYTES}
EXPERT_DTYPE_BYTES: dict[str, float] = {
    "bf16": BF16_BYTES,
    "fp8": FP8_BYTES,
    "nvfp4": NVFP4_BYTES,
    "mxfp4": MXFP4_BYTES,
}
KV_DTYPE_BYTES: dict[str, float] = {"bf16": BF16_BYTES, "fp8": FP8_BYTES}
STATE_DTYPE_BYTES: dict[str, float] = {"fp32": FP32_BYTES, "bf16": BF16_BYTES}

WEIGHT_ONLY_QUANT_METHODS: frozenset[str] = frozenset(
    {"awq", "gptq", "gptq_marlin", "awq_marlin", "bitsandbytes", "hqq", "eetq"}
)

# Solver, HBM capacity, and SMEM page-table constants.
DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.92
MAX_CTX_TOKENS_FOR_SMALL_PAGE: int = 8192
# Reserve half of SMEM for KV page tables and half for compiler/kernel scratch.
SMEM_PAGE_TABLE_DIVISOR: int = 2
BYTES_PER_PAGE_ENTRY: int = 4
MIN_MOE_INTERMEDIATE_ALIGNMENT: int = int(
    pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).num_lanes
)
RPA_MIN_PAGE_TOKENS: int = 16
RPA_MAX_PAGE_TOKENS: int = 256
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

# Three-valued SLO feasibility verdicts and ascending rank order.
type SloVerdict = Literal["ruled_out", "estimate_meets", "undecided"]

SLO_VERDICT_RULED_OUT: SloVerdict = "ruled_out"
SLO_VERDICT_UNDECIDED: SloVerdict = "undecided"
SLO_VERDICT_ESTIMATE_MEETS: SloVerdict = "estimate_meets"

SLO_VERDICT_RANK: dict[str, int] = {
    SLO_VERDICT_RULED_OUT: 0,
    SLO_VERDICT_UNDECIDED: 1,
    SLO_VERDICT_ESTIMATE_MEETS: 2,
}

# Warning a serving plan carries when the recommended layout's SLO verdict is undecided.
UNDECIDED_SLO_WARNING: str = (
    "Recommended layout has SLO verdict 'undecided': its modeled estimate "
    "misses the latency SLO, though the physical lower bound does not rule "
    "the layout out."
)

# Informational warning emitted when closed-loop think_time_s is zero.
ZERO_THINK_TIME_WARNING: str = (
    "Closed-loop scenario evaluated with zero think time (think_time_s = 0.0): "
    "steady-state residency follows active request arrival bounded by HBM, "
    "SMEM, and prefill capacity limits."
)

# Versioned solve JSON payload and top-K diversity selection constants.
SOLVE_SCHEMA: str = "vllm_torchtpu.roofline.solve/v2"
CHIP_SPEC_SCHEMA: str = "vllm_torchtpu.roofline.chip_spec/v1"
INSPECT_SCHEMA: str = "vllm_torchtpu.roofline.inspect/v1"
VALID_CHIP_PROVENANCE: frozenset[str] = frozenset(
    {"public_spec", "jax_tpu_info", "measured_microbench", "projected"}
)
ACTIVATION_HULL_SCHEMA: str = "vllm_torchtpu.roofline.moe_activation_hull/v1"
# Float64 slope rounding scales as ulp(active) / spacing (~1e-13 for 512
# experts); 1e-9 experts/token absorbs JSON float formatting slack.
HULL_CONCAVITY_SLOPE_TOLERANCE: float = 1e-9
# Absorbs 6-decimal JSON rounding (0.5e-6 half-ULP with 2x margin,
# unit: experts) when comparing exported knots to the Bernoulli cap.
HULL_BERNOULLI_CAP_TOLERANCE: float = 1e-6
MIN_TOP_LAYOUTS_FOR_FASTEST: int = 2
MIN_TOP_LAYOUTS_FOR_BOUND_BEST: int = 3
DEFAULT_TOP_K: int = 5
MIN_TOP_K: int = 1
MIN_DIVERSE_TOP_LAYOUTS: int = 4
DEEP_PP_THRESHOLD: int = 8

# Self-contained HTML report constants.
MAX_EMBEDDED_CANDIDATES: int = 200
MAX_SVG_BG_POINTS: int = 1200
SVG_GRID_CELL_PX: float = 4.0
MIN_FRONTIER_POINTS_FOR_POLYLINE: int = 2
PREFIX_RANGE_PAIR_LEN: int = 2

# Closed-world supported model_type families.
SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        # Core & LLaMA families
        "llama",
        "llama4",
        "llama4_text",
        # Qwen families
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_next",
        "qwen3_moe",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_8",
        "qwen3_8_moe",
        "qwen4_exp",
        "qwen4_exp_text",
        # DeepSeek families
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "deepseek_v4",
        # GLM families
        "glm4",
        "glm4_moe",
        "glm4_moe_lite",
        "glm5",
        "glm_moe",
        "glm_moe_dsa",
        # Kimi / Moonshot families
        "kimi_k2",
        "kimi_k3",
        "kimi_linear",
        # Mistral / Mixtral families
        "mistral",
        "mixtral",
        # Gemma families
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "gemma4",
        "gemma4_text",
        # Open source & other modeled causal LMs
        "gpt_oss",
        "opt",
        "minimax_text_01",
        # Supported hybrid state-space families (requiring GDN/KDA head keys)
        "nemotron_h",
        "bamba",
        "falcon_h1",
    }
)

HYBRID_SSM_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "nemotron_h",
        "bamba",
        "falcon_h1",
    }
)

MULTIMODAL_KEYS: tuple[str, ...] = (
    "vision_config",
    "audio_config",
    "video_config",
    "mm_projector_type",
    "image_token_index",
)
NON_GATED_MLP_MODEL_TYPES: frozenset[str] = frozenset({"opt", "gpt2", "gpt_neox"})
BENIGN_HF_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "_name_or_path",
        "architectures",
        "attention_bias",
        "attention_dropout",
        "auto_map",
        "bos_token_id",
        "cls_token_id",
        "eos_token_id",
        "id2label",
        "initializer_range",
        "label2id",
        "layer_norm_eps",
        "max_position_embeddings",
        "mlp_bias",
        "model_max_length",
        "pad_token_id",
        "rms_norm_eps",
        "rope_scaling",
        "rope_theta",
        "sep_token_id",
        "tokenizer_class",
        "transformers_version",
        "unk_token_id",
        "use_cache",
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

# Shared structured solver assumption catalog consumed by CLI --assumptions,
# solve JSON, and HTML reports.
SOLVER_ASSUMPTIONS: tuple[dict[str, object], ...] = (
    {
        "name": "bound_ttft_envelope_formula",
        "flag": "",
        "kind": "assumption",
        "value": "hit-rate-adjusted uncached prefill step floor",
        "unit": "formula",
        "description": (
            "Hard and envelope TTFT lower bounds evaluate unpadded prefill at "
            "the scenario prefix hit rate without heuristic residency scaling."
        ),
        "source": "solver",
    },
    {
        "name": "bound_tpot_residency",
        "flag": "",
        "kind": "assumption",
        "value": "least fixed-point steady-state residency",
        "unit": "formula",
        "description": (
            "Decode TPOT lower bounds evaluate bottom-up from batch 1 up to the "
            "HBM/SMEM residency cap using closed-loop steady-state arrival."
        ),
        "source": "solver",
    },
    {
        "name": "bound_capacity_ceiling",
        "flag": "",
        "kind": "assumption",
        "value": "unpadded linear work-conservation",
        "unit": "req/s",
        "description": (
            "Pool capacity ceiling X_cap_pool divides unpadded per-token HBM, "
            "matmul, vector, and collective cut bounds without padding bloat."
        ),
        "source": "solver",
    },
    {
        "name": "eff_prefill_rounding",
        "flag": "",
        "kind": "assumption",
        "value": "floor((1 - h) * isl) for bounds; ceil(...) for estimates",
        "unit": "tokens",
        "description": (
            "Strict resource bounds use floored uncached prefill tokens while "
            "serial estimates include minimum non-empty chunk/logit overhead."
        ),
        "source": "modules",
    },
    {
        "name": "hbm_reserve_gb",
        "flag": "--hbm-reserve-gb",
        "kind": "constraint",
        "value": DEFAULT_HBM_RESERVE_GB,
        "unit": "GB",
        "description": (
            "Additional host/framework HBM reserve subtracted from usable "
            "device memory when computing weights and KV page capacity."
        ),
        "source": "cli",
    },
    {
        "name": "gpu_memory_utilization",
        "flag": "",
        "kind": "assumption",
        "value": DEFAULT_GPU_MEMORY_UTILIZATION,
        "unit": "fraction",
        "description": (
            "Usable HBM fraction before weights, runtime buffers, and KV cache."
        ),
        "source": "constants",
    },
    {
        "name": "smem_page_table_budget",
        "flag": "",
        "kind": "assumption",
        "value": (
            int(
                pltpu.get_tpu_info_for_chip(
                    pltpu.ChipVersion.TPU_7X, 1
                ).smem_capacity_bytes
            )
            // SMEM_PAGE_TABLE_DIVISOR
        ),
        "unit": "bytes",
        "description": (
            "On-chip SMEM byte cap constraining resident page-table entries "
            "and maximum concurrent sequence residency."
        ),
        "source": "constants",
    },
    {
        "name": "prefill_chunk_tokens",
        "flag": "",
        "kind": "assumption",
        "value": DEFAULT_PREFILL_CHUNK,
        "unit": "tokens",
        "description": (
            "Default chunk size used when evaluating chunked prefill schedules "
            "under steady-state mixed concurrency."
        ),
        "source": "constants",
    },
    {
        "name": "collective_latency_us",
        "flag": "--collective-latency-us",
        "kind": "calibration",
        "value": DEFAULT_ICI_HOP_LATENCY_S / UNIT_MICRO,
        "unit": "us",
        "description": (
            "Optional per-hop ICI collective calibration latency added only to "
            "serial_estimate_s without relaxing physical bounds."
        ),
        "source": "cli",
    },
    {
        "name": "ep_uniform_routing",
        "flag": "--moe-activation",
        "kind": "constraint",
        "value": "uniform",
        "unit": "mode",
        "description": (
            "Default routed MoE activation assumption unless superseded by a "
            "supplied empirical hull or bounded load multiplier."
        ),
        "source": "occupancy",
    },
)


def get_solver_assumptions(
    chip: ChipSpec | None = None,
) -> list[dict[str, object]]:
    """Return a fresh list of structured solver assumption entries."""
    entries = [dict(entry) for entry in SOLVER_ASSUMPTIONS]
    if chip is not None:
        for entry in entries:
            if entry["name"] == "smem_page_table_budget":
                entry["value"] = chip.smem_page_table_bytes
    return entries
