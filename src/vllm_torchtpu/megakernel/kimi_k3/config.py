# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kernel settings for the Kimi-K3 megakernels.

Kernel modules read their settings from :func:`active` when they are
imported (most settings are module constants), so a non-default config must
be installed with :func:`install` before the first kernel import. Serving and
the standalone tools install overrides before building the program.
"""

import dataclasses
import hashlib
import json
from typing import Any

from vllm_torchtpu.megakernel.config import set_active_kernel_config

LOAD_FIELDS = ("read_streams", "load_inflight_gb", "expert_reshard_cache")


@dataclasses.dataclass(frozen=True)
class KernelConfig:
    # Decode context parallel attention.
    dcp_group: int = 32
    # Attention data parallelism of the decoder (TP32 / dp heads per DP group
    # of 32 / dp ranks, each with its own rows); needs dcp_group = 32 / dp.
    attention_dp: int = 1
    # Block tables: "local" (vLLM's rank-local DCP tables) or "global" (global
    # page ids).
    dcp_tables: str = "local"
    # MLA decode key tile (tokens per page fold). 2048 is opt-in and applies
    # only to FP8 plain decode at attention_dp=2, dcp_group=16 with 32 rows per
    # DP group (B64); everything else uses 1024.
    mla_decode_ktile: int = 1024
    # The target's chunked DCP attention (DCP32, attention DP2's DCP16: groups
    # of whole hosts) combines in two stages (each host merges its partials,
    # BF16, before one exchange across hosts), opt-in.
    dcp_host_combine: bool = False
    # The decode layer's attention cond with the delta-attention (KDA) branch
    # as its first region and the latent-attention (MLA) branch second: the
    # same computation, another code order in the rolled layer loop.
    kda_branch_first: bool = False
    # The compile sizes (requests: rows / rows per sequence) that take it.
    kda_branch_first_batches: tuple[int, ...] = (1, 2, 4)
    # The DCP query all-gathers' wire (target and MLA DSpark draft): "bf16"
    # sends the BF16-rounded queries and Kimi-K3.1's index fields (FP8 values,
    # exact in BF16) and each receiver quantizes the queries to FP8 exactly as
    # the sender would; "fp8" sends FP8 query/scale packets and FP8 field codes.
    dcp_query_wire: str = "bf16"
    # Routed experts (MXFP4, the pre-shard format).
    mxfp4_block: int = 2048
    mxfp4_pair_tile: int = 512
    mxfp4_static_tile: int = 0
    # None: 1024 at B96, 512 at power-of-two B1-B128, else unsplit (0).
    mxfp4_dot_tile: int | None = None
    pooled_weights: bool = True
    shared_shards: bool | None = None  # None: at B128
    compact_only: bool = False
    singleton_block: bool | None = None  # None: at B16-B128
    stage_blocks: bool | None = None  # None: at B16-B128
    b16_compact_experts: bool = True
    shared_expert_finishers: bool = True
    compact_overflow: bool = True
    # Routed experts: each expert of a pair at its own row capacity, in programs
    # of expert_own_capacity_rows rows (32, 64). off, gate_up (the gate/up dots)
    # or gate_up_finish (also the activation, down dot and output selection).
    expert_own_capacity: str = "off"
    expert_own_capacity_rows: tuple[int, ...] = (32,)
    # Decode expert weights sharded EP2 x TP16: 448 experts x 192 channels a rank,
    # eight a loop iteration, at most 16 rows (a pre-shard with "expert_layout":
    # "ep2_tp16"); off: EP8 x TP4.
    expert_weights_ep2_tp16: bool = False
    b4_attention_gather: bool = True
    fused_cache_commit: bool = True
    # Kimi-K3.1: the fused cache commit also writes the index keys (key row
    # and exponent byte); off: two XLA scatters after it.
    fused_index_commit: bool = True
    decode_no_hlo_calls: bool = True  # no deduplicated HLO calls at B8..B128, DP B2/B4
    shard_routing: bool = False
    kda_staged: bool = False
    weight_streams: tuple[tuple[str, tuple], ...] = ()  # (weight, (axis, streams))
    # VMEM and compiler settings.
    vmem_limit_mb: int = 64
    vmem_spill_reserve_mb: float = 5.0
    internal_scratch_kb: int = 0
    bounds_checks: bool = False
    # The MLA DSpark draft's DCP attention over an FP8 cache: "fp8" quantizes
    # the queries per head row and the probabilities as the target's verify
    # does; "bf16" keeps both BF16 against the FP8 pages (converted exactly).
    draft_attention_operands: str = "fp8"
    # Prefill attention.
    # Prefill MLA flash tiles (queries x keys per step).
    mla_flash_qtile: int = 1024
    mla_flash_ktile: int = 2048
    # Both prefill MoE rings: BF16 halves transfer bytes, rounding every hop.
    # FP32 runs the fused MoE at every window; BF16 the packed MoE at 4K/8K
    # and the SparseCore gather/combine below 4K.
    moe_reduce_dtype: str = "float32"
    # The most row-tokens (rows x local tokens) one decode score-and-select
    # pass holds scores for; a call of 32 or more rows above it scores and
    # selects its rows in two halves. 64 rows at 688K tokens of context fit
    # the attention allocation's free VMEM.
    dsa_select_row_tokens: int = 64 * 22528
    # Kimi-K3.1's select at 16 rows and more: virtual rows in descending
    # order of their sequences' positions (one permutation a step), so each
    # 8-row block's passes run to bounds of similar rows.
    dsa_select_sort: bool = False
    # Kimi-K3.1's select at eight rows: each round's digit counts go out as
    # 16-bit halves of int32 lanes (half the bytes) in a flat exchange, the
    # receiver unpacking every member's packet before the sum; while a row's
    # local tokens (and so every count) stay below 2^16, else today's form.
    dsa_select_pack16: bool = False
    # Kimi-K3.1's select at eight rows: the digit width, 6, 7 or 8 bits a round
    # (6 and 7 leave a last round of the remaining 2 or 4 bits); 0 keeps
    # dsa_select.SELECT_FORMS' 8.
    dsa_select_digit_bits: int = 0
    # Kimi-K3.1's select: stop the rounds once every row is decided (the count
    # at or above the lower edge of its current bin is exactly k, so every key
    # in the bin is selected), checked from the round that has resolved 20 bits.
    dsa_select_stop: bool = False
    # Kimi-K3.1's select at eight rows and below (packed counts): key planes a
    # loop trip. 1 keeps today's loop; 2 or 4 also make the targets and shift
    # once a block, count an odd plane in a tail, and fold the 8-bit fields into
    # 16-bit lane pairs before the row sums.
    dsa_select_count_unroll: int = 1
    # Kimi-K3.1's select at 16 and 32 rows: 0 keeps a compare per threshold; 2
    # or 4 count the digits in packed 8-bit counters (four a lane), that many
    # key planes a loop trip (as dsa_select_count_unroll's reworked pass).
    dsa_select_packed_unroll: int = 0
    # The decode scorer's new tokens: in groups of 8 rows in the scoring order
    # (a call of a multiple of 8 rows whose verify sequences divide 8), one MXU
    # dot of the group's queries against its 8 new keys scores every row's own
    # and earlier rows' tokens; off, each row scores them with one dot per
    # token. The scores are the same bits either way.
    dsa_group_new_tokens: bool = True
    # The decode scorer's cached pages, block-major (calls of 8 rows and up
    # with grouped new tokens): a loop step scores one page of each of up to 8
    # verify sequences (or two pages of a call's only sequence), every read
    # before any store, so the MXU feed of one page overlaps the drain of
    # another; one store per 8-row tile. The scores are the same bits.
    dsa_score_block_major: bool = False
    # Kimi-K3.1's prefill top-k selection: the Pallas kernel bounded by each
    # sequence's context (dsa_prefill_select), or the XLA one over the arena.
    dsa_prefill_select: str = "kernel"
    # The selection kernel's index-score products: BF16 operands (exact for
    # FP8 values), or FP8 x FP8 (the GPU indexer's; FP32 sums in another order).
    dsa_index_dots: str = "bf16"
    # The selection kernel's rows per grid step (16 halves the index-page
    # reads when the scores fit VMEM), its weighted head sums (VPU in the
    # decoder's order; butterfly: the same order, every row's sublane
    # reduction in one shared roll tree; or one MXU product in another FP32
    # order) and its threshold search (32 bisection passes; only the bits
    # below the common prefix of bounds from each 4096-column group's lane
    # maxima; or over each lane's 32 largest keys, sorted and merged per
    # group, with the full search when a lane may have dropped one: all exact).
    dsa_select_rows: int = 8
    dsa_head_sums: str = "vpu"
    dsa_bisect: str = "full"
    # Kimi-K3.1's prefill attention: "off" expands the arena per head and
    # masks the flash kernel by the selection; "heads" attends each row's
    # selected keys in latent form at the rank's heads (mla_sparse_prefill;
    # stages only); "rows" attends a rank's own rows at all 96 heads (stages
    # and the TP32 slice, which then hold every head's query and KV
    # up-projections); "rows_fast" is "rows" with the head-major selection
    # settings (dsa_head_sums major, dsa_bisect lanes, dsa_select_rows 32; the
    # same words as the decoder-order selection), set in __post_init__.
    mla_sparse_prefill: str = "off"
    # "rows": rows per union block. A block of consecutive rows gathers the
    # union of their selected tokens once and masks each row to its own (1:
    # every row gathers its own tokens).
    mla_sparse_block: int = 1
    # The SparseCore gathers the FP8 gather table directly (no int8 view,
    # which XLA materializes as a copy of the whole table each MLA layer).
    mla_sparse_fp8_gather: bool = False
    # "rows": windows whose longest context (cached + new tokens) is at most
    # this many tokens take the masked flash instead (0: never); windows of
    # up to 2048 tokens, then wider ones. From the slice's crossovers (k3p
    # runtime, fast selection): 1856- and 640-token windows near 54-61K,
    # 8192-token windows near 32K.
    mla_sparse_dense_below: int = 57344
    mla_sparse_dense_below_wide: int = 32768
    # "rows" (per-row gathers): row groups gathered and attended in turn, so
    # the SparseCore gather of one group can overlap the attention of the
    # previous one (1: one gather for all of a rank's rows).
    mla_sparse_groups: int = 1
    # "rows": the absorbed queries in two BF16 parts (no BF16 rounding of
    # q_nope W_UK^T in the scores; the score half of the kernel doubles).
    mla_sparse_query_split: bool = True
    # Weight loading.
    read_streams: int = 4
    load_inflight_gb: float = 16.0
    # Host-local directory (tmpfs) for the source files of expert weights sharded
    # EP2 x TP16 at load: a host's ranks share each file instead of each faulting
    # it in from network storage. Empty: read in place.
    expert_reshard_cache: str = ""
    # Diagnostics (off in serving): dcp_skip and debug_skip skip named work, so
    # results are invalid (timing or hang bisection only); the rest add trace
    # scopes, VMEM reports or VMEM padding.
    dcp_skip: tuple[str, ...] = ()
    debug_skip: tuple[str, ...] = ()
    phase_scopes: bool = False
    pool_report: bool = False
    vmem_report: bool = False
    vmem_pad_kb: int = 0

    def __post_init__(self) -> None:
        if self.mla_sparse_prefill == "rows_fast":
            for name, value in (
                ("mla_sparse_prefill", "rows"),
                ("dsa_head_sums", "major"),
                ("dsa_bisect", "lanes"),
                ("dsa_select_rows", 32),
            ):
                object.__setattr__(self, name, value)
        if self.moe_reduce_dtype not in ("float32", "bfloat16"):
            raise ValueError("moe_reduce_dtype is float32 or bfloat16")
        if self.expert_own_capacity not in ("off", "gate_up", "gate_up_finish"):
            raise ValueError("expert_own_capacity is off, gate_up or gate_up_finish")
        if not set(self.expert_own_capacity_rows) <= {32, 64}:
            raise ValueError("expert_own_capacity_rows holds 32 and 64 only")
        if self.dsa_prefill_select not in ("kernel", "xla"):
            raise ValueError("dsa_prefill_select is kernel or xla")
        if self.dsa_index_dots not in ("bf16", "fp8"):
            raise ValueError("dsa_index_dots is bf16 or fp8")
        if self.dsa_select_digit_bits not in (0, 6, 7, 8):
            raise ValueError("dsa_select_digit_bits is 0, 6, 7 or 8")
        if self.dsa_select_count_unroll not in (1, 2, 4):
            raise ValueError("dsa_select_count_unroll is 1, 2 or 4")
        if self.dsa_select_packed_unroll not in (0, 2, 4):
            raise ValueError("dsa_select_packed_unroll is 0, 2 or 4")
        if self.dsa_select_rows not in (8, 16, 32):
            raise ValueError("dsa_select_rows is 8, 16 or 32")
        if self.dsa_head_sums not in ("vpu", "butterfly", "mxu", "major", "bf16"):
            raise ValueError("dsa_head_sums is vpu, butterfly, mxu, major or bf16")
        if self.dsa_bisect not in ("full", "narrow", "lanes"):
            raise ValueError("dsa_bisect is full, narrow or lanes")
        if self.mla_sparse_prefill not in ("off", "heads", "rows"):
            raise ValueError("mla_sparse_prefill is off, heads or rows")
        if self.mla_sparse_groups not in (1, 2, 4, 8):
            raise ValueError("mla_sparse_groups is 1, 2, 4 or 8")
        if self.mla_sparse_block not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("mla_sparse_block is a power of two up to 64")
        if self.draft_attention_operands not in ("fp8", "bf16"):
            raise ValueError("draft_attention_operands is fp8 or bf16")
        if self.mla_decode_ktile not in (1024, 2048):
            raise ValueError("mla_decode_ktile supports 1024 or experimental 2048")
        if self.mla_decode_ktile == 2048 and (
            self.attention_dp != 2 or self.dcp_group != 16
        ):
            raise ValueError(
                "mla_decode_ktile=2048 requires attention_dp=2 and dcp_group=16"
            )

    def key(self) -> str:
        """A stable digest of every field that shapes a kernel, for caches of
        compiled kernels (the weight-loading settings change none)."""
        fields = {
            name: value
            for name, value in dataclasses.asdict(self).items()
            if name not in LOAD_FIELDS
        }
        blob = json.dumps(fields, sort_keys=True, default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def replace(self, **fields: Any) -> "KernelConfig":
        return dataclasses.replace(
            self, **{name: _coerce(self, name, value) for name, value in fields.items()}
        )


def _coerce(config: KernelConfig, name: str, value: Any) -> Any:
    """Parse a command-line string into the field's type."""
    current = getattr(config, name)
    if not isinstance(value, str):
        return value
    if current is None:  # an optional override: parse by the declared type
        if value.lower() in ("", "none"):
            return None
        kind = str(KernelConfig.__dataclass_fields__[name].type)
        if kind.startswith("bool"):
            return value.lower() in ("1", "true", "yes", "on")
        if kind.startswith("int"):
            return int(value)
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if name == "weight_streams":
        return tuple(sorted((k, tuple(v)) for k, v in json.loads(value).items()))
    if isinstance(current, tuple):
        items = tuple(filter(None, value.split(",")))
        return tuple(int(item) for item in items) if current and isinstance(current[0], int) else items
    return value


_ACTIVE: KernelConfig | None = None


def install(config: KernelConfig) -> KernelConfig:
    """Fix the process's kernel config. Kernel modules read it at import, so
    installing a different config after the first import is an error."""
    global _ACTIVE
    if _ACTIVE is not None and config != _ACTIVE:
        raise RuntimeError(
            f"K3 kernel config is already fixed ({_ACTIVE}); install {config} before "
            "the first kernel import"
        )
    _ACTIVE = config
    set_active_kernel_config(config)
    return config


def active() -> KernelConfig:
    """The installed config, or the defaults (installed on first use)."""
    return _ACTIVE if _ACTIVE is not None else install(KernelConfig())
