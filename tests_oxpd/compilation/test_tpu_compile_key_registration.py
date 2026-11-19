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

"""Compile-key registration of the q38-stack-v4 flags.

Rule 1: a flag that changes a traced graph or a kernel lowering is registered
in ``envs.environment_variables`` and is NOT in ``_TPU_COMPILE_ENV_IGNORED``,
so setting it moves ``compute_tpu_compilation_hash`` (and with it the vLLM
torch.compile cache directory).

Rule 2: a host-only flag (scheduler, staging, counters, timers, logs) is
registered and IS in ``_TPU_COMPILE_ENV_IGNORED``, so a served arm that sets it
still reuses the compile cache of the arm that does not.

The lists mirror the q38-stack-v4 flag registry (q38_stack_v4_flags.tsv, owned
by INT-3). A flag that joins the stack joins one of these lists. Each value is
a valid non-default setting: the registry's ``on`` value where that differs
from the default, otherwise another value the parser accepts.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# (flag, a valid non-default value, the default envs.py must resolve to)
HASHED_FLAGS = [
    # q38 L4 X-H14-AOTSKIP: hashed on purpose (lead ruling): its arm must compile its own
    # AOT forwards, not load a twin arm's artifact from the shared in-pod cache.
    ("VLLM_TPU_AOT_SKIP_GUARD_CHECK", "1", False),
    # q38 L4 X-D14-DRAFT-2 (a): the draft embedding all-reduce through H8 (traced draft graph).
    ("VLLM_TPU_DRAFT_EMBED_H8", "1", False),
    # q38 L4 X-D14-DRAFT-2 (c): the draft embeds from a replicated table (traced draft graph).
    ("VLLM_TPU_DRAFT_EMBED_REPL", "1", False),
    # q38 L4 EMB-REP-TGT: the target's embedding looks up the replicated table (traced target graph).
    ("VLLM_TPU_TARGET_EMBED_REPL", "1", False),
    # q38 L4 S1: exact top-kc on tie-free keys in VPS local_candidates (traced graph).
    ("VLLM_TPU_VPS_TOPK_KEYS", "1", False),
    # q38 L4 S2: exact top-k cutoff in the VPS candidate mask (traced graph).
    ("VLLM_TPU_VPS_TOPK_CUTOFF", "1", False),
    # q38 L4 VRNG-VPS: in-graph VPS verify uniforms (adds programs).
    ("VLLM_TPU_VPS_INGRAPH_RNG", "1", False),
    ("VLLM_TPU_QSA_PREFILL_UNION_SC_BITS_PLAN", "1", False),  # S2
    ("VLLM_TPU_PLE_FUSED", "1", False),  # F
    ("VLLM_TPU_PLE_FUSED_TAP_MODE", "single", "slice"),  # F sub
    ("VLLM_TPU_PLE_FUSED_TILE", "256", 128),  # F sub (registry on = default)
    ("VLLM_TPU_PLE_FUSED_PRECISION", "xla", "torch"),  # F sub
    ("VLLM_TPU_PLE_FUSED_NORM", "mean", "sum"),  # F sub (never set)
    ("VLLM_TPU_PLE_CONV_ONEHOT", "1", False),  # conv1h
    ("TPU_QWEN4_HC_PREFILL_DEFER_WHOLE", "1", False),  # D
    ("TPU_QWEN4_HC_PREFILL_FOLD", "1", False),  # forbidden in serving
    ("TPU_QWEN4_HC_PREFILL_DEFER_COMBINE", "1", False),  # forbidden in serving
    ("Q38_PROBE_PLE_ONEHOT_MUTANT", "1", False),  # probe, forbidden in serving
    ("VLLM_TPU_QSA_DECODE_REQUEST_SHARD", "1", False),  # O1s
    ("VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS", "48", 0),  # O1s min-rows gate (q38 L2 D21)
    ("VLLM_TPU_QSA_DCP_HEAD_A2A", "xla", ""),  # DCP head-slice all-to-all (DEC-COLL DCB1; q38 L2 D34)
    ("VLLM_TPU_QSA_DCP_Q_ONLY_GATHER", "1", False),  # DCP q-only pair gather (q38 L2 D27a)
    ("VLLM_TPU_GDN_WINDOW_TILE", "1", False),  # GDNWT window tile (DEC-FB-2 port; T95f O1 KEEP-ELIGIBLE; q38 L2 D51)
    ("VLLM_TPU_PLE_DECODE_BOUNDED", "1", "0"),  # PLE conv bounded to the bucket's requests (q38 L1 D07-b2)
    ("VLLM_TPU_PLE_DECODE_BOUNDED", "2", "0"),  # + static window slices (q38 L1 D07-b2)
    ("VLLM_TPU_PLE_D14_FOLD", "1", False),  # PLE conv + checkpoints in one Pallas program (q38 L1-PLE)
    ("VLLM_TPU_PLE_WRDEFER", "4", "0"),  # PLE checkpoint writer, 4 DMAs in flight (q38 L1 PLE-WRDEFER)
    ("VLLM_TPU_PLE_WRDEFER", "2", "0"),  # 2 DMAs in flight (q38 L1 PLE-WRDEFER)
    ("VLLM_TPU_QSA_SHARED_RANK_OPERANDS", "1", False),  # O1s-HOIST H-A shared rank operands (MM DSTOCK-B port)
    ("VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT", "1", False),  # O1s-HOIST port 2: decode indexer without the early-exit cond
    ("VLLM_TPU_QSA_INDEXER_STRIDED_KEYS", "1", False),  # D24 word-strided index key reads (q38 L2)
    ("VLLM_TPU_QSA_INDEXER_KEY_LANES", "128", 0),  # L11 128-lane index key copies (DEC-FB lever 11; X-IDX-4 port onto D24)
    # q38 PORT-BL IDX (ported from Benliang Wang, 2ce02bb1a): decode indexer dense step and tail skip (scoring kernel).
    ("VLLM_TPU_QSA_INDEXER_DENSE_STEP", "1", False),
    ("VLLM_TPU_QSA_INDEXER_TAIL_SKIP", "8", 0),
    # q38 PORT-BL COMPOSE: the other ported stock-path keys (Benliang Wang's mechanisms), hashed.
    ("TPU_QWEN4_HC_DECODE_FOLD", "1", False),  # PORT-BL HC (d523c2ce0 + 28419c4a5)
    ("TPU_QWEN4_HC_DECODE_TILE_ROWS", "512", 0),  # PORT-BL HC (28419c4a5)
    ("VLLM_TPU_GDN_LEAN_GLUE3", "1", False),  # PORT-BL GDN (aa8e6b237, 3 of 4 pieces)
    ("FUSED_MOE_DECODE_EXACT_WIDTH", "1", False),  # PORT-BL MOE (b0c1453ba)
    ("VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS", "80", 0),  # PORT-BL MOE (stock-bucket gate for 28419c4a5's chunk)
    # q38 HIGHC-HC3CAR COMPOSE: PORT-BL UNION's keys (Benliang Wang, 2ce02bb1a; q38-L1-v9-union fe2e4a0fa), hashed:
    # the union RPA verify attention and the HBM-typed RPA schedule on the O1s target attend (traced programs).
    ("VLLM_TPU_QSA_DECODE_RPA_UNION_O1S", "640", 0),
    ("VLLM_TPU_RPA_SCHEDULE_HBM_O1S", "1", False),
    # FR10b base composition (F2), hashed the same way.
    ("VLLM_TPU_MTP_DRAFT_ROWS", "1", False),
    ("VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", "1", False),
    ("VLLM_TPU_QSA_MTP_KV_DCP", "1", False),
    # q38 L4 X-D14-DRAFT-1: the draft's shared-selection ops without indexer state.
    ("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE", "1", False),
    # q38 DEC-FB-3 lever 3 (dongruizi 48745d9b0 / 7a0a3958e): new compiled programs and step-0 wiring.
    ("VLLM_TPU_DRAFT_ROWS_FRESH", "1", False),
    ("VLLM_TPU_DRAFT_CARRY_ROWS_FRESH", "1", False),
    # q38 DEC-FB-3 lever 2 (MM DSTOCK 6dcff75dc): the draft token programs change.
    ("VLLM_TPU_DRAFT_TOP_TOKENS", "1", False),
    # q38 DEC-FB-3 lever 4 (D-MK 99df0bbfe / 36624958c): read while the DCP decode op is traced.
    ("VLLM_TPU_QSA_DCP_ONE_GATHER", "1", False),
    # MM DSTOCK D12 RPAd fast fetch (ported, DEC-O4-4): changes the kernel (suffix -f).
    ("VLLM_TPU_QSA_DECODE_RPA_FETCH", "fast", ""),
    # Imported megakernel tree (MM 37b960ea9, c32 DCP-2 path): trace-time knobs, hashed.
    ("VLLM_TPU_MK_HOST_PLE", "1", False),
    ("VLLM_TPU_MK_GDN_CHUNK_GATHER", "0", "auto"),
    ("VLLM_TPU_MK_GDN_HM", "0", "1"),
    ("VLLM_TPU_MK_GDN_REC_F32", "1", "0"),
    ("VLLM_TPU_MK_HIER", "2", ""),
    ("VLLM_TPU_MK_MOE_ABLATE", "acc", ""),
    ("VLLM_TPU_MK_RINGS", "6:4", ""),
    ("VLLM_TPU_MK_SCOPES", "1", "0"),
    ("VLLM_TPU_MK_UNION_STACK", "0", "1"),
    # q38 L4 EAGER-FIX E-a: the spec tail program's rejected-count output shape.
    ("VLLM_TPU_SPEC_TAIL_REJ_PAD", "1", False),
    # q38 L1-D08S1: D08 S1 "UNION-LEAN" stages of the whole step (read at trace time by indexed_options).
    ("VLLM_TPU_MK_UNION_BOUNDED", "1", False),
    ("VLLM_TPU_MK_UNION_LEAN", "1", False),
    ("VLLM_TPU_MK_SLOT_BATCH", "1", False),
    # q38 L2-UAISSUE: the union walk's DMA descriptors (read at trace time by indexed_options).
    ("VLLM_TPU_MK_UNION_ISSUE", "2", 0),
    # q38 L2-UNPACK: the union tile body's ring read (read at trace time by indexed_options).
    ("VLLM_TPU_MK_UNION_UNPACK", "1", False),
    ("VLLM_TPU_MK_UNION_PIPE", "3", 0),  # q38 L2-UAPIPE X-UA-PIPE (traced whole step)
    # q38 L1-D07B1: the whole-step call's GDN metadata operands and block-id glue change.
    ("VLLM_TPU_MK_GDN_META_DEDUP", "1", False),
    # q38 L1-GLUEFIX: per-bucket gates of D07-b1 / D07-b2 (which program a bucket compiles / traces).
    ("VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS", "32", 0),
    ("VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS", "32", 0),
    # q38 L4 D83 (P3): the VPS verify program also returns the read-offset scatter operands.
    ("VLLM_TPU_VPS_FUSED_READ_OFFSETS", "1", False),
    # q38 L1-IDX: the in-kernel indexer package of the whole step (read at build / trace time by indexed_options,
    # build_wholestep_call and wholestep_tiles).
    ("VLLM_TPU_MK_SCORE_CHUNK", "128", "0"),  # S2P chunk pages
    ("VLLM_TPU_MK_SCORE_LANES", "128", "0"),  # S2P carrier lanes
    ("VLLM_TPU_MK_SCORE_MP", "4", "0"),  # S2P manager pages (+ manager-page key table)
    ("VLLM_TPU_MK_SELECT_RADIX", "2", "1"),  # S2P radix-2 thresholds
    ("VLLM_TPU_MK_UNION_QSPLIT", "1", "0"),  # QSPLIT strided rows
    ("VLLM_TPU_MK_UNION_QSPLIT", "2", "0"),  # QSPLIT row-gather matmul
    ("VLLM_TPU_MK_TABLE_SLIM", "1", "0"),  # TSLIM column slimming
    ("VLLM_TPU_MK_TABLE_SLIM", "2", "0"),  # TSLIM + next-layer prefetch
    ("VLLM_TPU_MK_RING_S1", "32=7:10,64=6:4", ""),  # RING-S1 per-bucket rings
    # q38 X-K-LIVE: the whole-step kernel takes the live-row mask and skips pad rows (operand list + traced program).
    ("VLLM_TPU_MK_LIVE_ROWS", "1", False),
    # q38 X-K-LIVE GDN compute sub-mode (ExecSpec.live_rows = 2).
    ("VLLM_TPU_MK_LIVE_GDN_COMPUTE", "1", False),
    # q38 X-GLUE-1-D14: in-kernel GDN checkpoint write (kernel outputs / inputs and the call's scatter change).
    ("VLLM_TPU_MK_GDN_CKPT_INK", "1", False),
    # q38 L1-G1 GDN-COMMIT1: the whole-step call skips the checkpoint scatter and returns the checkpoints (traced program).
    ("VLLM_TPU_MK_GDN_COMMIT1", "1", False),
    # q38 L1-QX: X-DCP-QX-EARLY, the DCP q exchange starts after the QSA projection (kernel option qx_early, traced program).
    ("VLLM_TPU_MK_DCP_QX_EARLY", "1", "0"),
    # q38 L4 DRAFT-ASM: the folded seed / carry / draft-loop programs.
    ("VLLM_TPU_DRAFT_ASM_FOLD", "1", False),
    # q38 L4 VFY-ONE: one VPS verify program (1) and + the draft seed (2); both precompiled when on.
    ("VLLM_TPU_VPS_VFY_ONE", "1", False),
    # q38 L4 DRAFT1-B: stage A's carry-loop and DH-D's one draft program.
    ("VLLM_TPU_DRAFT_ASM_CARRY_LOOP", "1", False),
    ("VLLM_TPU_DRAFT_ASM_ONE", "1", False),
    # q38 L4 DRAFT-CAP: DH-D's programs at R_d = min(p, capacity) request slots.
    ("VLLM_TPU_DRAFT_ASM_CAP", "1", False),
    # q38 L1-L47F: model layer 47 in the whole step (kernel L 45, the call's sink export, the skipped stock path).
    ("VLLM_TPU_MK_L47_FOLD", "1", False),
    # q38 L1-H3: the draft's fused-loop attention as one megakernel call per pass (draft program + operands).
    ("VLLM_TPU_MK_DRAFT_ATTN", "1", False),
    # q38 L1-GDNINK: in-kernel GDN variants (traced program).
    ("VLLM_TPU_MK_GDN_INK", "1", "0"),
    # q38 L4 LMH-STREAM stage 0: the draft / verify LM head's weight operand is [hidden, vocab shard].
    ("VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT", "1", False),
    ("VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY", "1", False),
    # q38 L4 T1-K1FUSE: the verify + draft program per static shape signature.
    ("VLLM_TPU_K1FUSE", "1", False),
    # q38 L1-SELLOOP: the whole step's selection loops over owned request slots only (traced program).
    ("VLLM_TPU_MK_SELECT_OWNED", "1", "0"),
    ("VLLM_TPU_MK_SELECT_OWNED", "2", "0"),
    # q38 L2 KVW-ALL: the whole step's QSA cache writes as one all-layer Pallas call (traced graph).
    ("VLLM_TPU_MK_QSA_WRITE_ALL", "1", False),
    # q38 L2 WFOLD: the whole-step call returns the kernel's QSA cache values and the writer takes them (traced graph).
    ("VLLM_TPU_MK_QSA_WRITE_KVAL", "1", False),
    # q38 L3 D60 (lever 5c): pipelined MoE compaction at buckets >= N rows (kernel -cr_<R>p).
    ("FUSED_MOE_DECODE_COMPACT_PIPE", "192", 0),
    ("FUSED_MOE_DECODE_COMPACT_PIPE_ROWS", "48", 32),
]

IGNORED_FLAGS = [
    ("VLLM_TPU_GDN_LEAN_GLUE3_COUNT", "500", 0),  # q38 PORT-BL COMPOSE: PORT-BL GDN engagement period (log-only)
    # q38 L4 SEEDFIX: seeded-request admission + an eager overlay on the shared draws (default on).
    ("VLLM_TPU_SEEDED_SAMPLING", "0", True),
    ("VLLM_TPU_Q38_STACK_ROUTE_COUNT", "50", 0),  # q38 L2 STACK-MARK: host route counter, log-only
    ("VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS", "32", 0),  # warm-up sweep cap
    ("VLLM_TPU_DRAFT_ASM_CHECK_STEPS", "200", 0),  # DRAFT-ASM CHECK: which path runs
    ("VLLM_TPU_DRAFT_ASM_AB", "ab:1+3", ""),  # DRAFT-ASM l6_ab window switch
    ("VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB", "ab:1+3", ""),  # DRAFT1-B stage A window switch
    ("VLLM_TPU_DRAFT_ASM_ONE_AB", "ab:2", ""),  # DRAFT1-B DH-D window switch
    ("VLLM_TPU_K1FUSE_AB", "ab:1", ""),  # T1-K1FUSE window switch
    ("VLLM_TPU_K1FUSE_CHECK", "0", 256),  # T1-K1FUSE CHECK window: which path serves
    ("VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY", "1", False),  # warm-up served-only
    ("VLLM_TPU_PDENSE_COUNT", "200", 0),  # host counters after dispatch
    ("VLLM_TPU_PD_PROMPT_TAIL", "64", 0),  # ADM-P: prompt tokens shipped
    ("VLLM_TPU_CPU_PROBE_S", "30", 0.0),  # q38 L1 STACK-D v3: CPUPROBE thread-placement log, host-only
    ("VLLM_TPU_KV_TWIN_GROUPS", "1", False),  # D1k: scheduler bookkeeping
    ("VLLM_TPU_BLOCK_TABLE_STAGE_INC", "1", False),  # BT-INC: host pack
    ("VLLM_TPU_BLOCK_TABLE_STAGE_INC_CHECK", "200", 0),  # BT-INC host check
    ("VLLM_TPU_ENGINE_PHASES", "200", 0),  # D1k engine timers
    ("VLLM_TPU_HOST_PHASES_SLOW_MS", "35", 0.0),  # HOSTSLOW log lines
    ("VLLM_TPU_HOST_PHASES_SUBS", "1", False),  # q38 L4: HOSTSUBS log lines
    ("VLLM_TPU_HOST_PHASES_GC", "0", True),  # q38 L6-INSTR: GCCENSUS log lines
    ("VLLM_TPU_STAGE_TIMING_OFF", "1", False),  # D1b: no timing events
    ("VLLM_TPU_DRAFT_TOP_TOKENS_CHECK", "200", 0),  # lever 2: host-side check
    ("VLLM_TPU_FAST_METADATA_BUILD", "1", False),  # MM FAST_METADATA_BUILD: host walk, same metadata
    # q38 DEC-HOST-4 D1-norm L-GDNCOPY: packed seed-copy uploads (same values, same programs).
    ("VLLM_TPU_GDN_SEED_COPY_PACKED", "1", False),
    ("VLLM_TPU_GDN_SEED_COPY_PACKED_CHECK", "200", 0),
    ("VLLM_TPU_D1N_AB_PERIOD_MS", "20000", 0),  # D1-norm windowed A/B clock
    # q38 L4 D98 / X-H14-D98R: host-side dispatch of the same compiled programs.
    ("VLLM_TPU_FAST_COMPILED_CALL", "1", False),
    ("VLLM_TPU_FAST_CALL_GUARD_MEMO", "check", ""),
    ("VLLM_TPU_FAST_CALL_DROP_GLOBAL_STATE", "1", False),
    ("VLLM_TPU_FAST_CALL_LOG_EVERY", "64", 200),
    # q38 L6-HOST step-time host levers and their A/B clock (core/l6_ab.py).
    ("VLLM_TPU_L6_AB_PERIOD_MS", "20000", 0),
    ("VLLM_TPU_L6_AB_CLASSES", "4", 2),
    ("VLLM_TPU_L6_AB_STEPS", "200", 0),
    ("VLLM_TPU_L6_AB_CONFIRM", "32", 16),
    ("VLLM_TPU_L6_AB_DIR", "/tmp/q38_l6ab_test", ""),
    ("VLLM_TPU_L6_CHECK_STEPS", "30000", 0),
    ("VLLM_TPU_L6_ENVS_CACHE", "1", ""),
    ("VLLM_TPU_L6_ENVS_CACHE_CHECK", "200", 0),
    ("VLLM_TPU_L6_ENVS_CENSUS", "200", 0),
    ("VLLM_TPU_L6_FAST_FWD_CTX", "1", ""),
    ("VLLM_TPU_L6_GC_FREEZE", "1", ""),
    ("VLLM_TPU_L6_SWITCH", "1", ""),
    ("VLLM_TPU_L6_SWITCH_US", "500", 200),
    ("VLLM_TPU_L6_MAIN_PIN", "1", ""),
    ("VLLM_TPU_L6_MAIN_PIN_CPU", "3", -1),
    # q38 L6 MAINPIN-2 / NUMA bind / SYNC-GC (core/l6_jitter.py, worker/numa_bind.py): host placement and GC timing.
    ("VLLM_TPU_L6_MAIN_PIN2", "1", ""),
    ("VLLM_TPU_L6_NUMA_BIND", "1", False),
    ("VLLM_TPU_L6_SYNC_GC", "1", ""),
    ("VLLM_TPU_L6_SYNC_GC_EVERY", "50", 100),
    ("VLLM_TPU_L6_SYNC_GC_VALVE", "20000", 100000),
    ("VLLM_TPU_L6_DEFER_PREV", "1", ""),
    ("VLLM_TPU_L6_STEADY_STEP", "1", ""),  # L6 steady-step fast path (runner/l6_steady.py)
    ("VLLM_TPU_L6_STEADY_STEP_CHECK", "200", 0),
    ("VLLM_TPU_L6_DECODE_CARRY", "1", ""),  # L6 lever carry (runner/decode_carry.py)
    ("VLLM_TPU_L6_DECODE_CARRY_CHECK", "200", 0),
    ("VLLM_TPU_L6_CARRY_MIN_REQS", "8", 0),  # L6-GATE batch-size gates (core/l6_ab.py GATES)
    ("VLLM_TPU_L6_STEADY_MIN_REQS", "8", 0),
    # q38 L6 lever adcarry (runner/adapt_carry.py): switch between precompiled programs and its check cadence.
    ("VLLM_TPU_L6_ADAPT_CARRY", "1", ""),
    ("VLLM_TPU_L6_ADAPT_CARRY_CHECK", "200", 0),
    # q38 L6 lever engalloc (core/l6_engine_alloc.py): engine-core KV slot fast path and its check cadence.
    ("VLLM_TPU_L6_ENGINE_ALLOC_FAST", "1", ""),
    ("VLLM_TPU_L6_ENGINE_ALLOC_FAST_CHECK", "200", 0),
    # q38 L6 lever setupfree (runner/l6_setupfree.py) and its check cadence.
    ("VLLM_TPU_L6_SETUP_FREE", "1", ""),
    ("VLLM_TPU_L6_SETUP_FREE_CHECK", "200", 0),
    # q38 L6-H2 levers (overflow unit l6_h2_cheap): fence skip, token lists once, AOT guard skip windows.
    ("VLLM_TPU_L6_FENCE_SKIP", "1", ""),
    ("VLLM_TPU_L6_TOKEN_LISTS", "1", ""),
    ("VLLM_TPU_L6_TOKEN_LISTS_CHECK", "200", 0),
    ("VLLM_TPU_L6_AOT_SKIP", "1", ""),
    ("VLLM_TPU_L6_AOT_SKIP_CHECK", "200", 0),
    # q38 L6 lever gcadmit (core/l6_gcadmit.py): gc.freeze() after admissions, its cap and check cadence.
    ("VLLM_TPU_L6_GC_ADMIT_FREEZE", "1", ""),
    ("VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP", "100000", 1000000),
    ("VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK", "16", 64),
    ("VLLM_TPU_L6_GC_CENSUS_TYPES", "1", False),  # L6 gen-2 heap type census (log-only)
    ("VLLM_TPU_L6_GC_CENSUS_MS", "500", 1000),
    ("VLLM_TPU_L6_GC_CENSUS_MAX", "2", 4),
    ("VLLM_TPU_L6_EOS_CANARY", "248044", ""),  # L6 end-token canary (log-only)
    # q38 L6 lever ptrims (runner/l6_ptrims.py): input-prep host trims and their check cadence.
    ("VLLM_TPU_L6_PREP_TRIMS", "1", ""),
    ("VLLM_TPU_L6_PREP_TRIMS_CHECK", "200", 0),
    ("VLLM_TPU_L6_ADMIT_INC", "1", ""),  # L6 lever admitinc (ADMIT-INC + batched row writes)
    ("VLLM_TPU_L6_ADMIT_INC_CHECK", "200", 0),
    # q38 L6-MK levers (runner/l6_mk.py, core/l6_fuse_rpc.py): host call paths only.
    ("VLLM_TPU_L6_MK_CTX", "1", ""),
    ("VLLM_TPU_L6_MK_CTX_CHECK", "200", 0),
    ("VLLM_TPU_L6_MK_THIN_TT", "1", ""),
    ("VLLM_TPU_L6_MK_THIN_TT_CHECK", "200", 0),
    ("VLLM_TPU_L6_MK_DIRECT", "1", ""),
    ("VLLM_TPU_L6_MK_DIRECT_CHECK", "200", 0),
    ("VLLM_TPU_L6_FUSE_RPC", "1", ""),
    ("VLLM_TPU_L6_OUT_KVONLY", "1", ""),
    ("VLLM_TPU_L6_OUT_KVONLY_CHECK", "200", 0),
    # q38 L6 levers clean / pshort (runner/l6_carry_lean.py, runner/l6_prep_short.py) and their check cadences.
    ("VLLM_TPU_L6_CARRY_LEAN", "1", ""),
    ("VLLM_TPU_L6_CARRY_LEAN_CHECK", "200", 0),
    ("VLLM_TPU_L6_PREP_SHORT", "1", ""),
    ("VLLM_TPU_L6_PREP_SHORT_CHECK", "200", 0),
    # q38 L6 P-METER (runner/l6_pmeter.py): log-only stamps and the DELAY-INJECT host spin.
    ("VLLM_TPU_L6_PMETER", "1", ""),
    ("VLLM_TPU_L6_PMETER_WINDOW", "64", 200),
    ("VLLM_TPU_L6_INJECT", "ab:1", ""),
    ("VLLM_TPU_L6_INJECT_US", "1000", 250),
    ("VLLM_TPU_L6_INJECT_RANK", "3", 0),
    ("VLLM_TPU_L6_INJECT_ALL", "ab:2", ""),
    ("VLLM_TPU_L6_INJECT_ALL_US", "2000", 1500),
    ("VLLM_TPU_L6_NODEFER_CENSUS", "1", ""),  # q38 L6 NODEFER census (runner/l6_nodefer.py), log-only
    ("VLLM_TPU_L6_NODEFER_EVERY", "16", 64),
    ("VLLM_TPU_MK_OPERAND_DUMP", "1", False),  # imported MK: host operand dump
    ("VLLM_TPU_MK_STEP_MIX_INTERVAL_S", "30", 10.0),  # imported MK: log interval
    ("CDK_OUTPUT_DIR", "/tmp/x", "/tmp/cdk_output"),  # imported MK: dump directory
    ("VLLM_TPU_Q38_CTX_CENSUS", "200", 0),  # q38 L5 CTXCENSUS: host context census, log-only
    # q38 L4 EAGER-FIX: host-side copies, warm-up draws and the shape log.
    ("VLLM_TPU_SAMPLING_COPY_FULL", "1", False),
    ("VLLM_TPU_EAGER_SHAPE_LOG", "50", 0),
    ("VLLM_TPU_VPS_WARM_DRAWS", "4", 0),
    # q38 L4 VPSHOST host levers (l6_ab classes; read per step, never traced).
    ("VLLM_TPU_L4_SAMP_FB_ORDER", "1", ""),
    ("VLLM_TPU_L4_SAMP_SNAP", "1", ""),
    ("VLLM_TPU_L4_SAMP_SNAP_CHECK", "64", 0),
    ("VLLM_TPU_L4_VPS_STATS_ASYNC", "1", ""),
    ("VLLM_TPU_LOG_STEP_BUCKETS_PERIOD", "1", False),  # q38 L5 STEPBUCKETS2: host step periods, log-only
    ("VLLM_TPU_SERVED_AB", "whole:4000:64", ""),  # q38 L5 SAB: host routing between existing programs
    ("VLLM_TPU_DECODE_TRIM", "64:13.38/80:17.47", ""),  # q38 L1-C48 DECODE-TRIM: engine-core scheduling only
    ("VLLM_TPU_DECODE_TRIM_LOG", "500", 2000),  # q38 L1-C48 DECODE-TRIM window line period
    ("VLLM_TPU_DECODE_TRIM_DRAFT_STASH", "1", False),  # q38 L1-DSTASH: runner host-side stash, warmed program
    ("QWEN4_EXP_PD_REPACK_L2", "1", False),  # q38 L2 D74: its own landing program, none of the model's
    # q38 L4 D83: the fold's check (separate update on a copy, eager) and its positive control.
    ("VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK", "200", 0),
    ("VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT", "2", 0),
    # q38 L4 VFY-ONE: its CHECK count and its window switch (l6_ab lever "vfyone").
    ("VLLM_TPU_VPS_VFY_ONE_CHECK", "50", 0),
    ("VLLM_TPU_VPS_VFY_ONE_AB", "ab:1", ""),
    ("VLLM_TPU_VPS_VFY_ONE_KEYLITE", "1", False),  # q38 L4 KEYLITE: the warm-key test's computation, host-only
    ("VLLM_TPU_VPS_INGRAPH_RNG_AB", "alt:400", ""),  # q38 L4 VRNG-VPS path switch (host only)
    ("VLLM_TPU_S3_DEFER_GATE_MIN_REQS", "4", 0),  # q38 L1-S3DGATE: which L6 windows run lever s3defer (host only)
]

_ALL_FLAGS = HASHED_FLAGS + IGNORED_FLAGS


def _config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.9,
            num_gpu_blocks=1024,
            num_gpu_blocks_override=None,
            enable_prefix_caching=False,
            mamba_cache_mode="none",
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048,
            max_num_seqs=16,
        ),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=None,
    )


def _hash_with(env: dict[str, str | None]) -> str:
    """The TPU compile hash with each named flag set to its value (None = unset)."""
    from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

    with patch(
        "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
        return_value=[],
    ):
        with patch.dict(os.environ):
            for name, value in env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            return compute_tpu_compilation_hash(_config())


@pytest.mark.parametrize(("name", "value", "default"), _ALL_FLAGS)
def test_flag_is_registered_and_parses(name, value, default):
    """The compile-key collector skips a getter that raises, so an unparseable
    value would drop the flag from the hash and look like a hash change. Pin
    that the value parses, to something other than the default."""
    from vllm_torchtpu import envs

    assert name in envs.environment_variables
    getter = envs.environment_variables[name]
    with patch.dict(os.environ):
        os.environ.pop(name, None)
        assert getter() == default
        os.environ[name] = value
        assert getter() != default


@pytest.mark.parametrize(("name", "value", "default"), HASHED_FLAGS)
def test_graph_flag_moves_the_compile_hash(name, value, default):
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert name not in _TPU_COMPILE_ENV_IGNORED
    assert _hash_with({name: value}) != _hash_with({name: None})


@pytest.mark.parametrize(("name", "value", "default"), IGNORED_FLAGS)
def test_host_only_flag_keeps_the_compile_hash(name, value, default):
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert name in _TPU_COMPILE_ENV_IGNORED
    assert _hash_with({name: value}) == _hash_with({name: None})


def test_all_host_only_flags_together_keep_the_compile_hash():
    """A stacked arm sets every host-only lever at once."""
    on = {name: value for name, value, _ in IGNORED_FLAGS}
    off = {name: None for name, _, _ in IGNORED_FLAGS}
    assert _hash_with(on) == _hash_with(off)


def test_vfy_one_values_share_one_compile_hash():
    """q38 L4 VFY-ONE: 1 and 2 precompile the same programs, so they key the same cache (the seed is
    switched on by value); on differs from off; anything but 0 / 1 / 2 raises."""
    from vllm_torchtpu import envs

    n = "VLLM_TPU_VPS_VFY_ONE"
    assert _hash_with({n: "1"}) == _hash_with({n: "2"}) != _hash_with({n: None})
    assert _hash_with({n: "0"}) == _hash_with({n: None})
    with patch.dict(os.environ):
        for raw, want in (("", 0), ("0", 0), ("1", 1), ("2", 2), (" 2 ", 2)):
            os.environ[n] = raw
            assert envs.vfy_one_value() == want and envs.environment_variables[n]() == (want > 0)
        for bad in ("3", "on", "true", "-1"):
            os.environ[n] = bad
            with pytest.raises(ValueError):
                envs.vfy_one_value()
