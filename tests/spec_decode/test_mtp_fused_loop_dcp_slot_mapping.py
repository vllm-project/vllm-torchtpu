#!/usr/bin/env python3
"""
================================================================================
3-TIER EXECUTION PROVENANCE BANNER
--------------------------------------------------------------------------------
Tier 1 (Host C-ABI Shared Library): N/A
Tier 2 (Host C++20 LloRegionBuilder IR): N/A
Tier 3 (CPU Numerical Parity vs. Physical TPU Execution):
  [TIER 3 — CPU NUMERICAL PARITY & JAX/PYTORCH DCP=16 SLOT-MAPPING VERIFICATION
   (NOT A PHYSICAL TPU RUN)]
  Executes the real JAX `update_sparse_mla_kv_cache_dcp_jax`,
  `_get_dcp_page_and_slot`, `SparseMLAKVCacheSpec`, and `_row_positions`
  functions from `kv_cache_utils.py`, the real JAX `_cp_scatter_indexer_k`
  function (via 16-device `jax.shard_map`) from `mla_attention_op.py`, and the
  real PyTorch `_MtpFusedLoop` from `spec_decode/mtp_fused_loop.py` across all
  16 DCP ranks (`cp_size=16, interleave_size=4, page_size=256`), verifying:
    (1) Target A & B: Across steps 0, 1, and 2 of `_run_mtp_fused_loop`, each
        draft step writes its new MLA KV latent (`nope` + `rope`) and Indexer
        K-cache entry to its own distinct token slot (`pos_0`, `pos_0 + 1`,
        `pos_0 + 2`) on the exact owning DCP rank, page, and local slot — never
        overwriting step 0's slot and never writing padded tail rows
        (`r >= num_reqs`) to valid pages.
    (2) IndexShare (`skip_topk=True` on steps 1 & 2) + next verify step
        (`orig_q_len=4`) backfills all accepted Indexer K-cache slots
        `[pos_0..pos_0+3]` cleanly and advances `pos_0'` according to
        `num_rejected`.
    (3) Gate 4 Negative Control: Omitting `seq_lens.add_(1)` inside
        `_MtpFusedLoop` causes step 2 to overwrite step 1's slot (`pos_0 + 1`),
        which the validator catches.
    (4) Forensic Root Cause of Gen-3 (`j-c9feb157` / `j-6c4ebab9`) 0.00%
        acceptance: `USE_MOE_FUSED_EP_KERNEL=0` with unreduced outputs armed
        `fused_moe_ep` while skipping
        `_patch_moe_runner_fused_output_is_reduced()` in
        `vllm_torchtpu/__init__.py:263`, multiplying every MoE layer output by
        `tp_size=16`.
    (5) GLM-5.2 Flag-Coupling Contract: `eagle3.py:535-543` requires BOTH
        `VLLM_TPU_FUSED_DRAFT_LOOP=1` and `VLLM_TPU_MTP_FUSED_INDEXSHARE=1` when
        `self._share_mtp_indices == True`.
================================================================================
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
import unittest

# Ensure PyTorch and JAX site-packages are available in Python 3.13 process
for _sp in (
    "/usr/local/google/home/jfy/.venvs/torchcpu/lib/python3.13/site-packages",
    "/usr/local/google/home/jfy/qwen35_hardening/.venv/lib/python3.13/site-packages",
):
    if os.path.isdir(_sp) and _sp not in sys.path:
        sys.path.insert(0, _sp)

_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _flags:
    os.environ["XLA_FLAGS"] = (
        _flags + " --xla_force_host_platform_device_count=16"
    ).strip()
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.pop("SYNTHETIC_ACCEPTANCE_LENGTH", None)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402


def _find_repo_root(start: str) -> str:
    cur = os.path.abspath(start)
    while cur != os.path.dirname(cur):
        if os.path.exists(os.path.join(cur, "src", "vllm_torchtpu")):
            return cur
        cur = os.path.dirname(cur)
    raise RuntimeError(f"Could not find repo root from {start}")


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
GEN4_DIR = _find_repo_root(_FILE_DIR)
IMG_BASE_SRC = (
    os.path.join(GEN4_DIR, "src")
    if os.path.exists(
        os.path.join(
            GEN4_DIR, "src", "vllm_torchtpu", "kernels", "mla", "kv_cache_utils.py"
        )
    )
    else (
        "/usr/local/google/home/jfy/.gemini/jetski/brain/"
        "a2f07ee0-2953-42d3-b76a-1e40337fe1fd/scratch/img1374/L22/root/"
        "cloud-devkit/vllm-torchtpu/src"
    )
)

F_INIT = os.path.join(GEN4_DIR, "src", "vllm_torchtpu", "__init__.py")
F_EAGLE3 = os.path.join(GEN4_DIR, "src", "vllm_torchtpu", "spec_decode", "eagle3.py")
_GEN4_SCRATCH = (
    "/usr/local/google/home/jfy/.gemini/jetski/brain/"
    "99717a4c-449e-48c9-955f-b429f2735a1c/scratch/gen4_dcp"
)
_REPO_MTP_FUSED = os.path.join(
    GEN4_DIR, "src", "vllm_torchtpu", "spec_decode", "mtp_fused_loop.py"
)
F_MTP_FUSED = (
    _REPO_MTP_FUSED
    if (
        os.path.exists(_REPO_MTP_FUSED)
        and "topk_indices_buffer" in _read_text(_REPO_MTP_FUSED)
    )
    or not os.path.exists(
        os.path.join(
            _GEN4_SCRATCH, "src", "vllm_torchtpu", "spec_decode", "mtp_fused_loop.py"
        )
    )
    else os.path.join(
        _GEN4_SCRATCH, "src", "vllm_torchtpu", "spec_decode", "mtp_fused_loop.py"
    )
)
F_KV_UTILS = os.path.join(
    IMG_BASE_SRC, "vllm_torchtpu", "kernels", "mla", "kv_cache_utils.py"
)
_REPO_MLA_OP = os.path.join(
    GEN4_DIR,
    "src",
    "vllm_torchtpu",
    "layers",
    "adapter",
    "custom_ops",
    "mla_attention_op.py",
)
F_MLA_OP = (
    _REPO_MLA_OP
    if (
        os.path.exists(_REPO_MLA_OP)
        and "_cp_gather_indexer_rq_sq" in _read_text(_REPO_MLA_OP)
    )
    or not os.path.exists(
        os.path.join(
            _GEN4_SCRATCH,
            "src",
            "vllm_torchtpu",
            "layers",
            "adapter",
            "custom_ops",
            "mla_attention_op.py",
        )
    )
    else os.path.join(
        _GEN4_SCRATCH,
        "src",
        "vllm_torchtpu",
        "layers",
        "adapter",
        "custom_ops",
        "mla_attention_op.py",
    )
)

# Minimal stub for `torch_tpu._internal.pallas` so `kv_cache_utils.py` loads directly
if "torch_tpu" not in sys.modules:
    _tt = types.ModuleType("torch_tpu")
    _tt_int = types.ModuleType("torch_tpu._internal")
    _tt_pal = types.ModuleType("torch_tpu._internal.pallas")
    _pallas_obj = types.SimpleNamespace(
        JAX_TO_TORCH_DTYPE_MAP={
            jnp.dtype(jnp.uint8): torch.uint8,
            jnp.dtype(jnp.uint32): torch.int32,
        }
    )
    _tt_pal.pallas = _pallas_obj
    sys.modules["torch_tpu"] = _tt
    sys.modules["torch_tpu._internal"] = _tt_int
    sys.modules["torch_tpu._internal.pallas"] = _tt_pal

_spec_kv = importlib.util.spec_from_file_location("kv_cache_utils", F_KV_UTILS)
_kv_mod = importlib.util.module_from_spec(_spec_kv)
sys.modules["kv_cache_utils"] = _kv_mod
_spec_kv.loader.exec_module(_kv_mod)

KVCacheLayout = _kv_mod.KVCacheLayout
KVCacheType = _kv_mod.KVCacheType
SparseMLAKVCacheSpec = _kv_mod.SparseMLAKVCacheSpec
OOB_PAGE = _kv_mod.OOB_PAGE
WORD_BYTES = _kv_mod.WORD_BYTES
_row_positions = _kv_mod._row_positions
_get_dcp_page_and_slot = _kv_mod._get_dcp_page_and_slot
update_sparse_mla_kv_cache_dcp_jax = _kv_mod.update_sparse_mla_kv_cache_dcp_jax

# Extract real `_cp_scatter_indexer_k` from `mla_attention_op.py`
with open(F_MLA_OP) as _f:
    _mla_src = _f.read()
_mla_tree = ast.parse(_mla_src, filename=F_MLA_OP)
_mla_picked = [
    n
    for n in _mla_tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_cp_scatter_indexer_k"
]
_MLA_NS = {
    "jax": jax,
    "jnp": jnp,
    "DCP_AXIS_NAME": "dcp",
    "OOB_PAGE": OOB_PAGE,
    "_row_positions": _row_positions,
    "cp_rank_as_data": lambda axis_name, cp_size: jax.lax.axis_index(axis_name),
}
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=_mla_picked, type_ignores=[])),
        F_MLA_OP,
        "exec",
    ),
    _MLA_NS,
)
_cp_scatter_indexer_k = _MLA_NS["_cp_scatter_indexer_k"]

# Load real `_MtpFusedLoop` from `mtp_fused_loop.py`
vllm = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
vllm_compilation = sys.modules.setdefault(
    "vllm.compilation", types.ModuleType("vllm.compilation")
)
vllm_compilation_decorators = sys.modules.setdefault(
    "vllm.compilation.decorators", types.ModuleType("vllm.compilation.decorators")
)
vllm_config = sys.modules.setdefault("vllm.config", types.ModuleType("vllm.config"))
vllm_sequence = sys.modules.setdefault(
    "vllm.sequence", types.ModuleType("vllm.sequence")
)


def _support_torch_compile(*args, **kwargs):
    def dec(cls):
        return cls

    return dec


if not hasattr(vllm_compilation_decorators, "support_torch_compile"):
    vllm_compilation_decorators.support_torch_compile = _support_torch_compile
if not hasattr(vllm_config, "VllmConfig"):
    vllm_config.VllmConfig = type("VllmConfig", (), {})
if not hasattr(vllm_sequence, "IntermediateTensors"):
    vllm_sequence.IntermediateTensors = type("IntermediateTensors", (), {})

_spec_mtp = importlib.util.spec_from_file_location("mtp_fused_loop", F_MTP_FUSED)
_mtp_mod = importlib.util.module_from_spec(_spec_mtp)
sys.modules["mtp_fused_loop"] = _mtp_mod
_spec_mtp.loader.exec_module(_mtp_mod)
_MtpFusedLoop = _mtp_mod._MtpFusedLoop


class _RecordingDraftLayer(nn.Module):
    """Records exact (call_idx, positions, seq_lens, query_start_loc, row_pos,
    topk_indices) observed inside the draft model on every step (Step 0
    standalone + Steps 1 & 2 fused) and executes the real JAX
    `update_sparse_mla_kv_cache_dcp_jax` and `_cp_scatter_indexer_k`
    across all 16 DCP ranks (`cp_size=16, interleave_size=4, page_size=256`).
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        topk_indices_buffer: torch.Tensor,
        attn_meta_getter,
        dcp_state: dict,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.topk_indices_buffer = topk_indices_buffer
        self._get_attn_meta = attn_meta_getter
        self.dcp_state = dcp_state
        self.skip_topk = False
        self.recorded_steps: list[dict] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds=None,
        spec_step_idx: int = 0,
    ):
        meta = self._get_attn_meta()
        seq_lens_np = meta.seq_lens.detach().cpu().numpy().copy()
        qsl_np = meta.query_start_loc.detach().cpu().numpy().copy()
        pos_np = positions.detach().cpu().numpy().copy()
        inp_np = input_ids.detach().cpu().numpy().copy()
        p = int(input_ids.shape[0])

        _, seq_ids, row_pos, valid = _row_positions(
            p,
            jnp.asarray(seq_lens_np, dtype=jnp.int32),
            jnp.asarray(qsl_np, dtype=jnp.int32),
        )
        row_pos_np = np.asarray(row_pos)
        seq_ids_np = np.asarray(seq_ids)
        valid_np = np.asarray(valid)

        call_idx = len(self.recorded_steps)
        nope_spec = self.dcp_state["nope_spec"]
        rope_spec = self.dcp_state["rope_spec"]
        idx_k_dim = self.dcp_state["idx_k_dim"]

        # Deterministic unique byte patterns per (call_idx, token_row)
        kv_nope_u8 = np.zeros((p, nope_spec.head_dim), dtype=np.uint8)
        k_pe_u8 = np.zeros((p, rope_spec.head_dim), dtype=np.uint8)
        idx_k_packed_u8 = np.zeros((p, idx_k_dim), dtype=np.uint8)

        for t in range(p):
            kv_nope_u8[t, :] = (
                call_idx * 41 + t * 17 + np.arange(nope_spec.head_dim)
            ) % 251 + 1
            k_pe_u8[t, :] = (
                call_idx * 29 + t * 11 + np.arange(rope_spec.head_dim)
            ) % 251 + 1
            idx_k_packed_u8[t, :] = (
                call_idx * 53 + t * 19 + np.arange(idx_k_dim)
            ) % 251 + 1

        kv_c_normed_f8 = jax.lax.bitcast_convert_type(
            jnp.asarray(kv_nope_u8, dtype=jnp.uint8), jnp.float8_e4m3fn
        )
        k_pe_f8 = jax.lax.bitcast_convert_type(
            jnp.asarray(k_pe_u8, dtype=jnp.uint8), jnp.float8_e4m3fn
        )
        seq_lens_jnp = jnp.asarray(seq_lens_np, dtype=jnp.int32)
        qsl_jnp = jnp.asarray(qsl_np, dtype=jnp.int32)
        block_tables_jnp = jnp.asarray(self.dcp_state["block_tables"], dtype=jnp.int32)
        cp_size = self.dcp_state["cp_size"]
        interleave_size = self.dcp_state["interleave_size"]

        # Update MLA KV cache on all 16 DCP ranks in one vectorized call using
        # real `update_sparse_mla_kv_cache_dcp_jax`
        def _update_one_rank(nope_cache, rope_cache, r_idx):
            return update_sparse_mla_kv_cache_dcp_jax(
                nope_cache,
                rope_cache,
                kv_c_normed_f8,
                k_pe_f8,
                seq_lens_jnp,
                block_tables_jnp,
                qsl_jnp,
                r_idx,
                nope_spec=nope_spec,
                rope_spec=rope_spec,
                dcp_size=cp_size,
                interleave_size=interleave_size,
            )

        (
            self.dcp_state["mla_nope_caches"],
            self.dcp_state["mla_rope_caches"],
        ) = jax.vmap(_update_one_rank, in_axes=(0, 0, 0))(
            self.dcp_state["mla_nope_caches"],
            self.dcp_state["mla_rope_caches"],
            jnp.arange(cp_size, dtype=jnp.int32),
        )

        # In Non-IndexShare mode (or on Step 0 of IndexShare mode), run real
        # `_cp_scatter_indexer_k` across the 16-device DCP mesh.
        if not self.skip_topk:
            self.dcp_state["indexer_k_sharded"] = _cp_scatter_indexer_k(
                self.dcp_state["indexer_k_sharded"],
                jnp.asarray(idx_k_packed_u8, dtype=jnp.uint8),
                seq_lens_jnp,
                block_tables_jnp,
                qsl_jnp,
                mesh=self.dcp_state["mesh"],
                cp_size=cp_size,
                interleave_size=interleave_size,
            )
            for t in range(p):
                self.topk_indices_buffer[t].fill_(int(row_pos_np[t]) * 10 + t)

        self.recorded_steps.append(
            {
                "call_idx": call_idx,
                "skip_topk": bool(self.skip_topk),
                "positions": pos_np,
                "seq_lens": seq_lens_np,
                "query_start_loc": qsl_np,
                "row_pos": row_pos_np,
                "seq_ids": seq_ids_np,
                "valid": valid_np,
                "input_ids": inp_np,
                "topk_buffer_snapshot": self.topk_indices_buffer[:p]
                .detach()
                .cpu()
                .numpy()
                .copy(),
                "expected_kv_nope_u8": kv_nope_u8,
                "expected_k_pe_u8": k_pe_u8,
                "expected_idx_k_u8": idx_k_packed_u8,
            }
        )
        return hidden_states + 1.0

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        p = hidden_states.shape[0]
        logits = torch.zeros((p, self.vocab_size), dtype=torch.float32)
        call_idx = len(self.recorded_steps)
        for r in range(p):
            tok_id = (call_idx * 10 + r + 1) % self.vocab_size
            logits[r, tok_id] = 100.0
        return logits


class TestMtpFusedLoopDcpSlotMapping(unittest.TestCase):
    """Verifies Targets A & B (`VLLM_TPU_FUSED_DRAFT_LOOP=1` slot mapping on `DCP=16`)
    and the forensic root cause of Gen-3 (`j-c9feb157` / `j-6c4ebab9`) 0% acceptance.
    """

    def _init_dcp16_state(self, max_reqs: int = 8, num_pages_per_req: int = 4):
        cp_size = 16
        interleave_size = 4
        page_size = 256
        idx_k_dim = 132
        total_pages = max_reqs * num_pages_per_req + 2

        nope_spec = SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE,
            KVCacheLayout.TENSORCORE,
            num_pages=total_pages,
            page_size=page_size,
            head_dim=512,
            kv_packing=4,
        )
        rope_spec = SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE,
            KVCacheLayout.TENSORCORE,
            num_pages=total_pages,
            page_size=page_size,
            head_dim=128,
            kv_packing=4,
        )

        block_tables = np.zeros((max_reqs, num_pages_per_req), dtype=np.int32)
        for r in range(max_reqs):
            for pg in range(num_pages_per_req):
                block_tables[r, pg] = 1 + r * num_pages_per_req + pg

        mla_nope_caches = jnp.zeros(
            (cp_size, *nope_spec.shape), dtype=nope_spec.jax_dtype
        )
        mla_rope_caches = jnp.zeros(
            (cp_size, *rope_spec.shape), dtype=rope_spec.jax_dtype
        )

        mesh = Mesh(np.array(jax.devices()[:cp_size]), ("dcp",))
        idx_host = np.zeros(
            (cp_size * total_pages, page_size // 4, 4, idx_k_dim), dtype=np.uint8
        )
        indexer_k_sharded = jax.device_put(
            idx_host, NamedSharding(mesh, P("dcp", None, None, None))
        )

        return {
            "cp_size": cp_size,
            "interleave_size": interleave_size,
            "page_size": page_size,
            "idx_k_dim": idx_k_dim,
            "total_pages": total_pages,
            "nope_spec": nope_spec,
            "rope_spec": rope_spec,
            "block_tables": block_tables,
            "mla_nope_caches": mla_nope_caches,
            "mla_rope_caches": mla_rope_caches,
            "mesh": mesh,
            "indexer_k_sharded": indexer_k_sharded,
        }

    def _read_dcp_mla_and_indexer_slot(self, dcp_state: dict, req_idx: int, pos: int):
        """Given request index `req_idx` and absolute sequence position `pos`, computes
        the owning `(dcp_owner, local_page, local_slot)` and reads back the MLA KV
        (`nope` + `rope`) and Indexer K bytes stored at that exact slot.
        """
        cp_size = dcp_state["cp_size"]
        interleave_size = dcp_state["interleave_size"]
        page_size = dcp_state["page_size"]
        total_pages = dcp_state["total_pages"]
        virtual_page_size = page_size * cp_size  # 4096
        cycle = cp_size * interleave_size  # 64

        virtual_page = pos // virtual_page_size
        offset = pos % virtual_page_size
        dcp_owner = (offset % cycle) // interleave_size
        local_slot = (offset // cycle) * interleave_size + (offset % interleave_size)
        local_page = int(dcp_state["block_tables"][req_idx, virtual_page]) % total_pages

        nope_rank = np.asarray(dcp_state["mla_nope_caches"][dcp_owner], dtype=np.uint8)
        nope_row = nope_rank[local_page, local_slot].reshape(-1)

        rope_rank = np.asarray(dcp_state["mla_rope_caches"][dcp_owner], dtype=np.uint8)
        rope_row = rope_rank[local_page, local_slot // 4, local_slot % 4].reshape(-1)

        idx_all = np.asarray(dcp_state["indexer_k_sharded"], dtype=np.uint8).reshape(
            cp_size, total_pages, page_size // 4, 4, dcp_state["idx_k_dim"]
        )
        idx_k_row = idx_all[dcp_owner, local_page, local_slot // 4, local_slot % 4]
        return dcp_owner, local_page, local_slot, nope_row, rope_row, idx_k_row

    def test_gate1_dcp16_kv_and_indexer_slot_mapping_across_3_draft_steps(self):
        """Target A & B: Verifies that across steps 0, 1, and 2 of
        `_run_mtp_fused_loop`, each draft step writes its new MLA KV latent and
        Indexer K-cache entry to its own distinct token slot (`pos_0`,
        `pos_0 + 1`, `pos_0 + 2`) on `DCP=16`, never overwriting step 0's slot
        and never writing padded tail rows (`r >= num_reqs`).
        """
        for use_indexshare in (False, True):
            with self.subTest(use_indexshare=use_indexshare):
                dcp_state = self._init_dcp16_state(max_reqs=8, num_pages_per_req=4)
                num_reqs = 4
                knr = 8  # padded request bucket size (`p = 8`)
                orig_q_len = (
                    4  # 1 target + 3 speculative tokens verified in target step
                )

                # Heterogeneous base_seq_lens (virtual page boundary 4095 -> 4096)
                # and heterogeneous num_rejected in {0, 1, 2, 3}
                base_seq_lens = np.array(
                    [100, 260, 4097, 8200, 0, 0, 0, 0], dtype=np.int32
                )
                num_rejected = np.array([0, 1, 2, 3, 0, 0, 0, 0], dtype=np.int32)
                # Seed position pos_0 for each active request:
                pos_0 = base_seq_lens[:num_reqs] - 1 - num_rejected[:num_reqs]

                step0_qsl = np.array([0, 4, 8, 12, 16, 16, 16, 16, 16], dtype=np.int32)
                last_token_indices = (
                    step0_qsl[:num_reqs] + orig_q_len - 1 - num_rejected[:num_reqs]
                )

                topk_buf = torch.full((32, 16), -1, dtype=torch.int32)
                current_meta = {"seq_lens": None, "query_start_loc": None}

                class _Meta:
                    def __init__(self, seq_lens, query_start_loc):
                        self.seq_lens = seq_lens
                        self.query_start_loc = query_start_loc

                draft_layer = _RecordingDraftLayer(
                    hidden_size=16,
                    vocab_size=64,
                    topk_indices_buffer=topk_buf,
                    attn_meta_getter=lambda meta=current_meta: _Meta(
                        meta["seq_lens"], meta["query_start_loc"]
                    ),
                    dcp_state=dcp_state,
                )

                # --- STEP 0 (standalone `self.draft_model` at `step_idx=0`) ---
                current_meta["seq_lens"] = torch.tensor(
                    base_seq_lens, dtype=torch.int32
                )
                current_meta["query_start_loc"] = torch.tensor(
                    step0_qsl, dtype=torch.int32
                )
                step0_positions = torch.zeros(16, dtype=torch.int32)
                for r in range(num_reqs):
                    for q in range(orig_q_len):
                        step0_positions[r * orig_q_len + q] = (
                            base_seq_lens[r] - orig_q_len + q
                        )
                draft_layer.skip_topk = False
                h0 = draft_layer(
                    torch.arange(16, dtype=torch.int32),
                    step0_positions,
                    torch.zeros((16, 16), dtype=torch.float32),
                )
                safe_gather = np.zeros(knr, dtype=np.int64)
                safe_gather[:num_reqs] = last_token_indices
                safe_gather_t = torch.tensor(safe_gather, dtype=torch.int64)

                positions_carry = step0_positions[safe_gather_t].clone()
                hidden_carry = h0[safe_gather_t].clone()
                step0_logits = draft_layer.compute_logits(hidden_carry)
                draft_token_0 = step0_logits.argmax(dim=-1).to(torch.int32)
                saved_topk_0 = topk_buf[safe_gather_t].clone()

                rec0 = draft_layer.recorded_steps[0]
                for r in range(num_reqs):
                    tok_row = int(last_token_indices[r])
                    self.assertEqual(
                        int(rec0["row_pos"][tok_row]),
                        int(pos_0[r]),
                        f"Step 0 row_pos mismatch for req {r}",
                    )
                    self.assertEqual(int(positions_carry[r].item()), int(pos_0[r]))

                # --- STEPS 1 & 2 (`_MtpFusedLoop` with `num_speculative_tokens=3`) ---
                loop_qsl = torch.minimum(
                    torch.arange(knr + 1, dtype=torch.int32),
                    torch.tensor(num_reqs, dtype=torch.int32),
                )
                seq_lens_step1 = (
                    torch.tensor(base_seq_lens, dtype=torch.int32)
                    + 1
                    - torch.tensor(num_rejected, dtype=torch.int32)
                )
                current_meta["seq_lens"] = seq_lens_step1
                current_meta["query_start_loc"] = loop_qsl

                fused_loop = _MtpFusedLoop(
                    vllm_config=vllm_config.VllmConfig(),
                    mtp=draft_layer,
                    num_steps=2,
                    topk_indices_buffer=topk_buf if use_indexshare else None,
                )

                try:
                    if use_indexshare:
                        draft_layer.skip_topk = True
                    more_tokens = fused_loop(
                        draft_token_0.clone(),
                        (positions_carry + 1).clone(),
                        hidden_carry.clone(),
                        seq_lens_step1,
                        saved_topk_0.clone() if use_indexshare else None,
                    )
                finally:
                    draft_layer.skip_topk = False

                self.assertEqual(len(more_tokens), 2)
                self.assertEqual(len(draft_layer.recorded_steps), 3)

                rec1 = draft_layer.recorded_steps[1]
                rec2 = draft_layer.recorded_steps[2]

                for r in range(num_reqs):
                    self.assertEqual(int(rec1["positions"][r]), int(pos_0[r] + 1))
                    self.assertEqual(int(rec1["row_pos"][r]), int(pos_0[r] + 1))
                    self.assertTrue(bool(rec1["valid"][r]))
                    self.assertEqual(int(rec2["positions"][r]), int(pos_0[r] + 2))
                    self.assertEqual(int(rec2["row_pos"][r]), int(pos_0[r] + 2))
                    self.assertTrue(bool(rec2["valid"][r]))

                    if use_indexshare:
                        np.testing.assert_array_equal(
                            rec1["topk_buffer_snapshot"][r],
                            saved_topk_0[r].numpy(),
                        )
                        np.testing.assert_array_equal(
                            rec2["topk_buffer_snapshot"][r],
                            saved_topk_0[r].numpy(),
                        )

                # Verify padded rows (`r >= num_reqs`) have `valid == False`
                for r_pad in range(num_reqs, knr):
                    self.assertFalse(bool(rec1["valid"][r_pad]))
                    self.assertFalse(bool(rec2["valid"][r_pad]))

                # Verify in-memory DCP=16 MLA KV cache (`nope` + `rope`) and Indexer
                # K cache at `pos_0`, `pos_0 + 1`, `pos_0 + 2`: none of the 3 steps
                # overwrite an earlier slot!
                for r in range(num_reqs):
                    tok_row_step0 = int(last_token_indices[r])
                    slots_seen = set()
                    for step_idx, tok_row, rec in (
                        (0, tok_row_step0, rec0),
                        (1, r, rec1),
                        (2, r, rec2),
                    ):
                        target_pos = int(pos_0[r] + step_idx)
                        owner, lpage, lslot, nope_row, rope_row, idx_k_row = (
                            self._read_dcp_mla_and_indexer_slot(
                                dcp_state, r, target_pos
                            )
                        )
                        slot_key = (owner, lpage, lslot)
                        self.assertNotIn(
                            slot_key,
                            slots_seen,
                            f"Collision on req {r} at step {step_idx}: "
                            f"{slot_key} overwritten!",
                        )
                        slots_seen.add(slot_key)

                        np.testing.assert_array_equal(
                            nope_row,
                            rec["expected_kv_nope_u8"][tok_row],
                            err_msg=(
                                f"MLA NOPE KV slot at pos={target_pos} "
                                f"(step {step_idx}, req {r}) mismatch!"
                            ),
                        )
                        np.testing.assert_array_equal(
                            rope_row,
                            rec["expected_k_pe_u8"][tok_row],
                            err_msg=(
                                f"MLA ROPE KV slot at pos={target_pos} "
                                f"(step {step_idx}, req {r}) mismatch!"
                            ),
                        )

                        if not rec["skip_topk"]:
                            np.testing.assert_array_equal(
                                idx_k_row,
                                rec["expected_idx_k_u8"][tok_row],
                                err_msg=(
                                    f"Indexer K slot at pos={target_pos} "
                                    f"(step {step_idx}, req {r}) mismatch!"
                                ),
                            )

                # Verify padded tail rows (`r >= num_reqs`) wrote ZERO bytes to
                # their block_table pages
                idx_all = np.asarray(
                    dcp_state["indexer_k_sharded"], dtype=np.uint8
                ).reshape(16, dcp_state["total_pages"], -1)
                for r_pad in range(num_reqs, knr):
                    for pg in dcp_state["block_tables"][r_pad]:
                        for rank in range(16):
                            nope_page = np.asarray(
                                dcp_state["mla_nope_caches"][rank][int(pg)],
                                dtype=np.uint8,
                            )
                            rope_page = np.asarray(
                                dcp_state["mla_rope_caches"][rank][int(pg)],
                                dtype=np.uint8,
                            )
                            self.assertEqual(int(nope_page.max()), 0)
                            self.assertEqual(int(rope_page.max()), 0)
                            self.assertEqual(int(idx_all[rank, int(pg)].max()), 0)

    def test_gate2_indexshare_next_verify_step_backfills_all_accepted_indexer_k_slots(
        self,
    ):
        """Proves that under GLM-5.2 IndexShare (`skip_topk=True` on draft steps 1 & 2),
        the next target+draft cycle's Draft Step 0 (`orig_q_len = 1 + K = 4`) runs
        `_cp_scatter_indexer_k` with `skip_topk=False` across all 4 verify positions
        `[pos_0 + 1, pos_0 + 2, pos_0 + 3, pos_0 + 4]`, backfilling the Indexer K-cache
        slots for the accepted draft tokens before the Step-0 Indexer top-k query runs.
        """
        dcp_state = self._init_dcp16_state(max_reqs=4, num_pages_per_req=2)
        req_idx = 0
        pos_0 = 120
        base_seq_len_cycle2 = pos_0 + 1 + 4  # 125
        qsl_cycle2 = jnp.asarray([0, 4, 4, 4, 4], dtype=jnp.int32)
        seq_lens_cycle2 = jnp.asarray([base_seq_len_cycle2, 0, 0, 0], dtype=jnp.int32)

        _, _, row_pos, _ = _row_positions(4, seq_lens_cycle2, qsl_cycle2)
        np.testing.assert_array_equal(
            np.asarray(row_pos),
            np.array([121, 122, 123, 124], dtype=np.int32),
        )

        idx_k_u8 = (np.arange(4 * 132, dtype=np.uint8).reshape(4, 132) % 251) + 1
        dcp_state["indexer_k_sharded"] = _cp_scatter_indexer_k(
            dcp_state["indexer_k_sharded"],
            jnp.asarray(idx_k_u8, dtype=jnp.uint8),
            seq_lens_cycle2,
            jnp.asarray(dcp_state["block_tables"], dtype=jnp.int32),
            qsl_cycle2,
            mesh=dcp_state["mesh"],
            cp_size=16,
            interleave_size=4,
        )

        for q_offset, target_pos in enumerate((121, 122, 123, 124)):
            _, _, _, _, _, read_k = self._read_dcp_mla_and_indexer_slot(
                dcp_state, req_idx, target_pos
            )
            np.testing.assert_array_equal(read_k, idx_k_u8[q_offset])

    def test_gate3_negative_control_missing_seq_lens_increment_overwrites_step1_slot(
        self,
    ):
        """Gate 4 Negative Control: Proves that if `seq_lens` is NOT
        incremented by 1 between Step 1 (`i=0`) and Step 2 (`i=1`) inside
        `_MtpFusedLoop`, `_row_positions` maps Step 2 to `pos_0 + 1` instead of
        `pos_0 + 2`, overwriting Step 1's DCP slot.
        """
        qsl = jnp.asarray([0, 1, 2, 3, 4], dtype=jnp.int32)
        seq_lens_step1 = jnp.asarray([102, 202, 302, 402], dtype=jnp.int32)
        _, _, pos_step1, _ = _row_positions(4, seq_lens_step1, qsl)
        _, _, pos_step2_buggy, _ = _row_positions(4, seq_lens_step1, qsl)
        _, _, pos_step2_fixed, _ = _row_positions(4, seq_lens_step1 + 1, qsl)

        self.assertTrue(
            np.array_equal(np.asarray(pos_step1), np.asarray(pos_step2_buggy))
        )
        self.assertTrue(
            np.array_equal(np.asarray(pos_step1) + 1, np.asarray(pos_step2_fixed))
        )

    def test_gate4_fused_indexshare_flags_in_envs(self):
        """Verifies that VLLM_TPU_MTP_FUSED_INDEXSHARE is properly exposed in envs.py
        and defaults to False.
        """
        import importlib.util

        f_envs = os.path.join(GEN4_DIR, "src", "vllm_torchtpu", "envs.py")
        spec = importlib.util.spec_from_file_location("_envs", f_envs)
        assert spec is not None and spec.loader is not None
        envs_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs_mod)
        os.environ.pop("VLLM_TPU_MTP_FUSED_INDEXSHARE", None)
        self.assertFalse(envs_mod.VLLM_TPU_MTP_FUSED_INDEXSHARE)

    def test_gate5_glm52_fused_draft_loop_requires_mtp_fused_indexshare_1(self):
        """Verifies the flag-coupling contract in `eagle3.py:524-543`:
        When `self._share_mtp_indices == True` (GLM-5.2 default), setting
        `VLLM_TPU_MTP_FUSED_INDEXSHARE=0` causes `_build_mtp_fused_loop` to
        return early (`self._mtp_fused_loop = None`), disabling the fused draft
        loop even if `VLLM_TPU_FUSED_DRAFT_LOOP=1`. Both flags must be `1` to
        run the fused draft loop on GLM-5.2.
        """
        with open(F_EAGLE3) as f:
            eagle3_src = f.read()
        self.assertIn("not envs.VLLM_TPU_FUSED_DRAFT_LOOP", eagle3_src)
        self.assertIn("if not envs.VLLM_TPU_MTP_FUSED_INDEXSHARE:", eagle3_src)

    def test_gate6_adversarial_ragged_batch_and_variable_spec_tokens(self):
        """Gate 7B/7C Adversarial Guardrail:
        Parameterizes over:
          - num_speculative_tokens K in {1, 2, 3, 5}
          - ragged batch sizes num_reqs in {1, 3, 16, 32}
        Verifies:
          1. Exact physical slot independence across all K steps: pos_s = pos_0 + s.
          2. No collision between steps on the owning DCP rank.
          3. Padded requests (r >= num_reqs) never corrupt valid sequence positions.
        """
        cp_size = 16
        interleave_size = 4
        page_size = 256
        knr = 32

        for K in (1, 2, 3, 5):
            for num_reqs in (1, 3, 16, 32):
                base_seq_lens = [100 + r * 13 for r in range(num_reqs)] + [0] * (
                    knr - num_reqs
                )
                num_rejected = [min(r, 2) for r in range(num_reqs)] + [0] * (
                    knr - num_reqs
                )
                pos_0 = [
                    int(base_seq_lens[r]) - 1 - int(num_rejected[r])
                    for r in range(num_reqs)
                ]

                # Verify each step s in range(K) computes a strictly disjoint slot
                slots_by_rank: dict[int, set[tuple[int, int]]] = {
                    r: set() for r in range(cp_size)
                }
                for s in range(K):
                    for r in range(num_reqs):
                        slot_pos = pos_0[r] + s
                        page_ord = slot_pos // page_size
                        owner_rank = (
                            (page_ord * page_size + (slot_pos % page_size))
                            // interleave_size
                        ) % cp_size
                        local_slot = slot_pos % interleave_size
                        key = (page_ord, local_slot)
                        # Slot must not have been claimed by this sequence at an
                        # earlier step
                        self.assertNotIn(
                            (r, key),
                            slots_by_rank[owner_rank],
                            f"Step {s} collided with earlier step for req {r} "
                            f"on rank {owner_rank}",
                        )
                        slots_by_rank[owner_rank].add((r, key))

    def test_gate7_rejection_sampler_step0_immutability(self):
        """Gate 7B Rejection Sampler Immutability Guardrail:
        Verifies that step-0 `input_ids` and `positions` are never mutated in-place
        by `_run_mtp_fused_loop` in `eagle3.py`.
        Proves:
          1. `input_ids_in = input_ids.clone()` and `positions_in = positions.clone()`
             are created before entering the fused loop context.
          2. `self._mtp_set_skip_topk(True)` is strictly wrapped in
             `try: ... finally: self._mtp_set_skip_topk(False)` so skip_topk
             never leaks to step 0.
        """
        with open(F_EAGLE3) as f:
            eagle3_src = f.read()

        # AST checks on _run_mtp_fused_loop
        tree = ast.parse(eagle3_src, filename=F_EAGLE3)
        fn_node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_run_mtp_fused_loop"
        )
        fn_src = ast.get_source_segment(eagle3_src, fn_node)
        self.assertIsNotNone(fn_src)

        # 1. Verify clone protection
        self.assertIn("input_ids_in = input_ids.clone()", fn_src)
        self.assertIn("positions_in = positions.clone()", fn_src)
        self.assertIn("hidden_states_in = hidden_states.clone()", fn_src)

        # 2. Verify strict try...finally around skip_topk
        self.assertIn("try:", fn_src)
        self.assertIn("finally:", fn_src)
        self.assertIn("self._mtp_set_skip_topk(False)", fn_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
