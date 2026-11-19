# SPDX-License-Identifier: Apache-2.0
"""The megakernel tree imported from dongruizi / MM 37b960ea9 (block import a5df8c4ab, c32 DCP-2 path; DEC-O4-4) is byte for
byte MM's, except the named, pinned exceptions below. CPU only, no vllm import: it hashes files.

Manifest: mm_37b960ea9_megakernel_sha256.json = sha256 of every file under src/vllm_torchtpu/megakernel and
src/vllm_torchtpu/megakernel_runtime at 37b960ea9 (115 files), written from `git show 37b960ea9:<path>`.
Exceptions:
  - megakernel_runtime/qwen38_driver.py carries two marked insertions: the main-KV DCP guard call (commit 1e8e7cd34) and the
    '+'-join normalization of VLLM_TPU_MEGAKERNEL (INT-4 cb677f4ad, ported from this base's 80b612e33; cherry-picked onto
    q38-L1-d14serve by L1-SERVE for the served D14 string); with both blocks cut out, the file must hash to MM's exactly;
  - megakernel_runtime/qwen38_guards.py is this base's own file (not in 37b960ea9);
  - q38 L1-D08S1 (D08 S1 UNION-LEAN): megakernel/wholestep/qsa_full_lean.py is a new file; spec.py and
    qwen38_wholestep_call.py carry the pinned lines of D08S1_EDITS (and the latter one block appended at the end), undone
    before D14's below;
  - q38 L1-D14 (the DCP-4 group path): megakernel/wholestep/qsa_full_c.py, spec.py and megakernel_runtime/
    qwen38_wholestep_call.py carry the pinned in-place lines of D14_EDITS and (qsa_full_c.py, qwen38_wholestep_call.py)
    one block appended at the end; with those undone, each file must hash to MM's exactly (qsa_full_b.py is MM's,
    unchanged). The driver's D14 lines sit inside the marked guard insertion above;
  - q38 L1-D07B1 (VLLM_TPU_MK_GDN_META_DEDUP): megakernel_runtime/qwen38_segment_call.py, qwen38_wholestep_call.py,
    qwen38_wholestep.py and qwen38_driver.py carry D07B1_EDITS: a pinned number of blocks between "# q38 L1-D07B1 begin"
    / "# q38 L1-D07B1 end" lines plus pinned in-place lines; with those undone first (then D14's / the guard insertion),
    each file must hash to MM's exactly, and no other line may mention L1-D07B1;
  - q38 L1-V3 (STACK-D v3 composed tree): the driver also carries HOST_DRIVER_BLOCKS, the exact unmarked additions of
    L4 HOSTSUBS (host_phases sub-marks around the whole-step call) and L5 SAB (served A/B windows); each occurs exactly
    once and is cut before the marked insertions; megakernel_runtime/served_ab.py (L5 SAB) is a new file.
  - q38 L1-IDX (the in-kernel indexer package): megakernel/wholestep/qsa_full_idx.py is a new file; spec.py,
    megakernel_runtime/qwen38_wholestep_call.py and qwen38_wholestep_inputs.py carry the pinned in-place lines of IDX_EDITS
    and (the latter two) one block appended at the end; strip_idx undoes them FIRST (before D07B1's / D14's / D08S1's), and
    no other line of those files (nor qsa_full_b / qsa_full_c / union_attn / indexer / qsa_full_lean) may mention L1-IDX.
  - q38 L5 CTXCENSUS (VLLM_TPU_Q38_CTX_CENSUS, log-only): the driver also carries exactly CENSUS_BLOCKS marked blocks of
    whole lines (from a MARK3 line through the next END3 line); they are cut first, and with them cut the driver is
    v3c's driver byte for byte, so the steps above apply unchanged.
  - q38 X-K-LIVE (VLLM_TPU_MK_LIVE_ROWS; L1-LIVE GDN half + L3-LIVE MoE half; STACK-D v4): megakernel/live_rows.py and
    megakernel/blocks/moe/trace.py are new files; the files of LIVE_EDITS carry its pinned in-place edits and (some) one
    appended block (the whole-step call's after the L1-IDX block); strip_live undoes them first (before strip_idx; after strip_glue), and no
    other file of the manifest may mention X-K-LIVE.
  - q38 L2 D73 gate (a4333fb7d; STACK-D v4): megakernel_runtime/qwen38_wholestep_cache.py carries the pinned lines of
    D73_EDITS, undone by strip_d73 (with strip_live, strip_v4).
  - q38 X-GLUE-1 (VLLM_TPU_MK_GDN_CKPT_INK, in-kernel GDN checkpoint write): megakernel/models/qwen38_gdn_ckpt_ink.py is a
    new file; blocks/gdn/states.py, models/qwen38.py, wholestep/spec.py, megakernel_runtime/qwen38_wholestep_call.py and
    qwen38_segment_call.py carry the pinned in-place lines of GLUE_EDITS (and the last two one block appended at the end);
    strip_glue undoes them after strip_l47f and strip_qx (strip_v5), and no other line of those files may mention X-GLUE-1.
  - q38 L1-QX (X-DCP-QX-EARLY, VLLM_TPU_MK_DCP_QX_EARLY; cycle 6): megakernel_runtime/qwen38_wholestep_call.py carries the
    pinned in-place line of QX_EDITS and one block appended at the end (after X-K-LIVE's and X-GLUE-1's); strip_qx undoes them (inside
    strip_v5, after strip_l47f, before strip_glue; STACK-D v5), and no other file of the manifest may mention L1-QX. (qsa_full_idx.py, where the kernel half
    lives, is L1-IDX's new file, outside the manifest.)
  - q38 L1-L47F (VLLM_TPU_MK_L47_FOLD, L47-FOLD-S1): megakernel/wholestep/qsa_full_l47f.py and
    megakernel_runtime/qwen38_l47fold.py are new files; planner.py and models/qwen38.py carry the partial-last-cycle port
    of dongruizi a18f2ea44 and spec.py / qwen38_wholestep_call.py / qwen38_wholestep.py the fold's edits, all pinned in
    L47F_EDITS (in-place texts, the call module's block appended at the end after X-K-LIVE's, and qwen38_wholestep.py's
    L47F_BLOCKS marked blocks); strip_l47f undoes them FIRST (inside strip_v5; STACK-D v5: its block is the last of the
    whole-step call), and no other file of the manifest may mention L1-L47F.
  - q38 L1-H3 (H3-DRAFT-ATTN-D14, VLLM_TPU_MK_DRAFT_ATTN): megakernel/wholestep/qsa_draft.py and
    megakernel_runtime/qwen38_draft_attn.py are new files; no file of the manifest is edited (no manifest file mentions
    L1-H3).
  - q38 L1-GDNINK (VLLM_TPU_MK_GDN_INK; branch q38-L1-gdnink): megakernel/blocks/gdn/recurrence.py and megakernel/stream.py
    carry the pinned in-place edits of GINK_EDITS (gdn_rec's X-K-LIVE dispatch line / dmatmul's K-tile loop line, each
    extended on the same line) and one block appended at the end of each (in recurrence.py after X-K-LIVE's); strip_gink
    undoes them FIRST (inside strip_v4, before strip_live), and no other file of the manifest may mention L1-GDNINK.
  - q38 L1-GDNINK (VLLM_TPU_MK_GDN_INK; branch q38-L1-gdnink): megakernel/blocks/gdn/recurrence.py and
    megakernel_runtime/qwen38_wholestep_call.py carry the pinned in-place edits of GINK_EDITS (gdn_rec's X-K-LIVE dispatch
    line / wholestep_tiles' return line, each extended on the same line) and one block appended at the end of each (after
    X-K-LIVE's); strip_gink undoes them FIRST (inside strip_v4, before strip_live), and no other file of the manifest may
    mention L1-GDNINK.
  - q38 L1-SELLOOP (VLLM_TPU_MK_SELECT_OWNED; owned-slot selection loop): megakernel_runtime/qwen38_wholestep_call.py carries
    the pinned in-place line of SELLOOP_EDITS and one block appended at the very end (after X-K-LIVE's); strip_selloop undoes
    them FIRST (inside strip_v6, which strip_v5 calls first; STACK-D v6 / q38 L1-V6: its block follows L1-L47F's and ends the call module), and no other file of the manifest may mention L1-SELLOOP. The kernel
    half lives in the base's own files qsa_full_lean.py / qsa_full_idx.py (BASE_ONLY).
  - q38 L1-KVW (KVW-ALL + QSA-WRITE-FOLD stage A, VLLM_TPU_MK_QSA_WRITE_ALL / _KVAL; wired by q38 L1-V6, STACK-D v6):
    megakernel_runtime/qwen38_wholestep_call.py and qwen38_wholestep_cache.py carry the pinned in-place lines of KVW_EDITS
    and one block appended at the very end of each (in the call after L1-SELLOOP's); strip_kvw undoes them FIRST (inside
    strip_v6), and no other file of the manifest may mention L1-KVW. The writer itself (qwen38_qsa_write_all.py) is a new
    file outside the manifest.
  - q38 L2-UAISSUE (X-UA-ISSUE): megakernel/wholestep/union_issue.py is a new file; union_attn.py carries UAISSUE_EDITS marked
    blocks between "# q38 L2-UAISSUE begin" / "# q38 L2-UAISSUE end" lines (cut first; then the file must hash to MM's exactly
    and no other line may mention L2-UAISSUE); its qwen38_wholestep_call.py lines sit inside D08S1's appended block.
  - q38 L2-UNPACK (X-UA-UNPACK): megakernel/wholestep/union_unpack.py is a new file; union_attn.py carries UNPACK_BLOCKS marked
    blocks between "# q38 L2-UNPACK begin" / "# q38 L2-UNPACK end" lines (cut before UAISSUE's), and qwen38_wholestep_call.py
    the pinned line of UNPACK_LINES (undone before D08S1's edits; its functions sit inside D08S1's appended block).
  - q38 L2-UAPIPE (X-UA-PIPE): union_attn.py gains one block appended after MM's last line, spec.py and qwen38_wholestep_call.py
    carry the pinned lines of UAPIPE_EDITS (and the latter one block appended at its end); undone first of all.
  - q38 L2-UAMERGE (X-UA-ISSUE + X-UA-UNPACK + X-UA-PIPE (ii) composed, q38-l2-uamerge): the composition lives inside
    L2-UAPIPE's appended union_attn.py block and in qsa_full_lean.py (a new file); no pinned line beyond the three above.
  - q38 L2-XUAV5 (the X-UA port onto STACK-D v5 F-v5c 31de11eac, q38-l2-xuav5): on the whole-step call the X-UA-ISSUE,
    X-UA-UNPACK and X-UA-PIPE functions are one tail appended after L1-L47F's block (from XUAV5_CALL_MARK to the end),
    and _d08s1_options carries the pinned lines of XUAV5_CALL_LINES; strip_xuav5 undoes every X-UA edit FIRST of all
    (inside strip_v6, before L1-KVW's and L1-SELLOOP's; q38 L1-V6, STACK-D v6: the X-UA tail follows L1-KVW's block), so each X-UA-touched file is its F-v5c text again and every v5 / v4 / IDX /
    D07B1 / D14 / D08S1 step below applies unchanged.
  - q38 L1-G1 (GDN-COMMIT1, VLLM_TPU_MK_GDN_COMMIT1): megakernel_runtime/qwen38_gdn_commit.py is a new file;
    qwen38_wholestep_call.py, qwen38_wholestep.py, qwen38_segment_call.py and qwen38_driver.py carry the pinned in-place
    lines of G1_EDITS and one block appended at the very end of each (in the call after the X-UA tail, in the segment call
    after X-GLUE-1's block, in the driver after __all__); strip_g1 undoes them FIRST of all (inside strip_v6, before
    strip_xuav5; the driver check calls it before the census cut), and no other file of the manifest may mention L1-G1.
Any other added, missing or changed file fails."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MANIFEST = os.path.join(HERE, "mm_37b960ea9_megakernel_sha256.json")
DRIVER = "src/vllm_torchtpu/megakernel_runtime/qwen38_driver.py"
BASE_ONLY = {"src/vllm_torchtpu/megakernel_runtime/qwen38_guards.py",
             "src/vllm_torchtpu/megakernel/wholestep/qsa_full_lean.py",  # q38 L1-D08S1
             "src/vllm_torchtpu/megakernel_runtime/served_ab.py",  # q38 L5 SAB (L1-V3 composition)
             "src/vllm_torchtpu/megakernel/wholestep/qsa_full_idx.py",  # q38 L1-IDX
             "src/vllm_torchtpu/megakernel/live_rows.py",  # q38 X-K-LIVE (STACK-D v4)
             "src/vllm_torchtpu/megakernel/blocks/moe/trace.py",  # q38 X-K-LIVE (STACK-D v4)
             "src/vllm_torchtpu/megakernel/models/qwen38_gdn_ckpt_ink.py",  # q38 X-GLUE-1
             "src/vllm_torchtpu/megakernel/wholestep/qsa_full_l47f.py",  # q38 L1-L47F
             "src/vllm_torchtpu/megakernel_runtime/qwen38_l47fold.py",  # q38 L1-L47F
             "src/vllm_torchtpu/megakernel/wholestep/qsa_draft.py",  # q38 L1-H3 (H3-DRAFT-ATTN-D14)
             "src/vllm_torchtpu/megakernel_runtime/qwen38_draft_attn.py",  # q38 L1-H3
             "src/vllm_torchtpu/megakernel_runtime/qwen38_qsa_write_all.py",  # q38 L2 KVW-ALL / WFOLD (q38 L1-V6, STACK-D v6)
             "src/vllm_torchtpu/megakernel/wholestep/union_issue.py",  # q38 L2-UAISSUE
             "src/vllm_torchtpu/megakernel/wholestep/union_unpack.py",  # q38 L2-UNPACK
             "src/vllm_torchtpu/megakernel_runtime/qwen38_gdn_commit.py"}  # q38 L1-G1 (GDN-COMMIT1)
MARK = "        # q38 DEC-O4-4 (c32 DCP-2 port; not in 37b960ea9)"
END = '            logger.warning("Qwen3.8 megakernel: %s", refusal)\n'
WS = "src/vllm_torchtpu/megakernel/wholestep/"
RT = "src/vllm_torchtpu/megakernel_runtime/"
MKD = "src/vllm_torchtpu/megakernel/"
# q38 L1-D14: file -> ([(D14 line, MM's line)], appended-block marker or None); each D14 line occurs exactly once and the
# appended block runs from its marker to the end of the file.
D14_EDITS = {
    WS + "qsa_full_c.py": (
        [
            ('                qs = [jnp.concatenate([scr["qf_q"][r], scr["qd_q"][r]], axis=0) if self.dcp == 2 else '
             'self._group_q(ctx, r) for r in rs]\n',
             '                qs = [jnp.concatenate([scr["qf_q"][r], scr["qd_q"][r]], axis=0) for r in rs]\n'),
            ("                for r, (att, lse) in zip(rs, res) if self.dcp == 2 else ():\n",
             "                for r, (att, lse) in zip(rs, res):\n"),
            ("                return outs if self.dcp == 2 else self._group_stage(ctx, rs, res, outs)\n",
             "                return outs\n"),
        ],
        "\n\n# ------------------------------------------- main-KV DCP over a group of 4 (q38 L1-D14)\n",
    ),
    WS + "spec.py": (
        [
            ("        from .qsa_full_c import QSAFullStepBlock, QSAFullStepGroupBlock\n",
             "        from .qsa_full_c import QSAFullStepBlock\n"),
            ('        qsa_block: Any = (QSAFullStepGroupBlock if int(indexed.get("dcp", 1) or 1) > 2 else QSAFullStepBlock)'
             '(**indexed, **(qsa or {}))  # q38 L1-D14\n',
             "        qsa_block: Any = QSAFullStepBlock(**indexed, **(qsa or {}))\n"),
        ],
        None,
    ),
    RT + "qwen38_wholestep_call.py": (
        [
            ("    t = _group_dcp_rings(_wholestep_tiles(B, max_groups), B)  # q38 L1-D14\n",
             "    t = _wholestep_tiles(B, max_groups)\n"),
        ],
        "\n\n# q38 L1-D14: main-KV DCP 4 (the whole step's group path",
    ),
}
MARK2 = "        # q38 INT-4 (c32 '+' option joins; not in 37b960ea9)"
END2 = "        # q38 INT-4 end\n"
# q38 L5 CTXCENSUS: the census hooks in the driver (attribute, helpers, the stock_ctx and whole-step calls).
MARK3 = "# q38 L5 CTXCENSUS (not in 37b960ea9)"
END3 = "# q38 L5 CTXCENSUS end\n"
CENSUS_BLOCKS = 4


# q38 L1-D08S1: file -> ([(D08S1 text, D14 text)], appended-block marker or None), undone before D14's edits.
D08S1_EDITS = {
    WS + "spec.py": (
        [
            ("        from .qsa_full_c import QSAFullStepBlock, QSAFullStepGroupBlock\n"
             "        from .qsa_full_lean import lean_step_block  # q38 L1-D08S1: None unless a D08 S1 option (or the test "
             "probe) is set\n",
             "        from .qsa_full_c import QSAFullStepBlock, QSAFullStepGroupBlock\n"),
            ('        qsa_block: Any = lean_step_block(indexed, qsa) or (QSAFullStepGroupBlock if int(indexed.get("dcp", 1) '
             'or 1) > 2 else QSAFullStepBlock)(**indexed, **(qsa or {}))  # q38 L1-D14\n',
             '        qsa_block: Any = (QSAFullStepGroupBlock if int(indexed.get("dcp", 1) or 1) > 2 else QSAFullStepBlock)'
             '(**indexed, **(qsa or {}))  # q38 L1-D14\n'),
        ],
        None,
    ),
    RT + "qwen38_wholestep_call.py": (
        [
            ("        compressed_dcp=compressed_dcp(), **_d08s1_options(),  # q38 L1-D08S1\n",
             "        compressed_dcp=compressed_dcp(),\n"),
        ],
        "\n\n# q38 L1-D08S1: D08 package S1",
    ),
}


# q38 L1-IDX: file -> ([(IDX line, the line before IDX)], appended-block marker or None); undone before every other strip.
IDX_EDITS = {
    WS + "spec.py": (
        [
            ("        from .qsa_full_idx import lean_step_block  # q38 L1-D08S1 / L1-IDX: None unless a D08 S1 / L1-IDX option "
             "(or the test probe) is set\n",
             "        from .qsa_full_lean import lean_step_block  # q38 L1-D08S1: None unless a D08 S1 option (or the test "
             "probe) is set\n"),
        ],
        None,
    ),
    RT + "qwen38_wholestep_call.py": (
        [
            ("        compressed_dcp=compressed_dcp(), **_d08s1_options(), **_idx_options(),  # q38 L1-D08S1 / L1-IDX\n",
             "        compressed_dcp=compressed_dcp(), **_d08s1_options(),  # q38 L1-D08S1\n"),
            ('    rings = os.environ.get("VLLM_TPU_MK_RINGS", "") or _ring_s1(B)  # q38 L1-IDX RING-S1 (per bucket, S1 on)\n',
             '    rings = os.environ.get("VLLM_TPU_MK_RINGS", "")\n'),
            ("    pps = _idx_pps(max_groups, layout)  # q38 L1-IDX TSLIM (max_groups * 4 // 128 when off)\n",
             "    pps = max_groups * 4 // 128\n"),
            ("        d = _idx_step_inputs(qsa_step_inputs)(  # q38 L1-IDX S2P / TSLIM (qsa_step_inputs itself when off)\n",
             "        d = qsa_step_inputs(\n"),
        ],
        "\n\n# q38 L1-IDX: the in-kernel indexer package on the whole step",
    ),
    RT + "qwen38_wholestep_inputs.py": ([], "\n\n# q38 L1-IDX: S2P's manager-page key table"),
}


# q38 X-GLUE-1 (VLLM_TPU_MK_GDN_CKPT_INK, in-kernel GDN checkpoint write): file -> ([(GLUE line, the line before GLUE)],
# appended-block marker or None); undone FIRST (before L1-IDX's / D07B1's / D14's / D08S1's). Each GLUE line keeps every
# expression start column of the line it replaces (flag-off lowering identity).
MK = "src/vllm_torchtpu/megakernel/"
GLUE_MARK = "# q38 X-GLUE-1 begin\n"
GLUE_EDITS = {
    MK + "blocks/gdn/states.py": (
        [
            ('    if out and "rec_chunk_out" in ctx.ST: return ctx.ST["rec_chunk_out"](ctx, gp, o, j, s)  # noqa: E701  Issue DMA for row chunk `j` of GDN layer `o` and ring buffer `s`; q38 X-GLUE-1 dispatch, same line\n',
             '    """Issue DMA for row chunk `j` of GDN layer `o` and ring buffer `s`."""\n'),
        ],
        None,
    ),
    MK + "models/qwen38.py": (
        [
            ('        from .qwen38_gdn_ckpt_ink import pool_gdn_cls; gdn_cls = pool_gdn_cls()  # noqa: E702  q38 X-GLUE-1: Qwen38PoolGDN unless VLLM_TPU_MK_GDN_CKPT_INK\n',
             '        from .qwen38_gdn_pool import Qwen38PoolGDN as gdn_cls\n'),
        ],
        None,
    ),
    WS + "spec.py": (
        [
            ('        from ..models.qwen38_gdn_ckpt_ink import pool_gdn_cls; gdn_cls = pool_gdn_cls()  # noqa: E702  q38 X-GLUE-1: Qwen38PoolGDN unless VLLM_TPU_MK_GDN_CKPT_INK\n',
             '        from ..models.qwen38_gdn_pool import Qwen38PoolGDN as gdn_cls\n'),
        ],
        None,
    ),
    RT + "qwen38_wholestep_call.py": (
        [
            ('        extra_outs=(KV_NEW, KR_NEW, IR_NEW) + ((UNION_STATS,) if stats else ()) + _ink_extra_outs(spec, interpret, layout),  # q38 X-GLUE-1\n',
             '        extra_outs=(KV_NEW, KR_NEW, IR_NEW) + ((UNION_STATS,) if stats else ()),\n'),
            ('        pool = scatter_gdn_ckpts_fast(pool, conv_ck, rec_ck, wb, layout=layout, interpret=interpret) if (conv_ck.shape, conv_ck.dtype) != (pool.shape, pool.dtype) else (conv_ck if interpret else pool)  # q38 X-GLUE-1: in-kernel checkpoints pass the pool through\n',
             '        pool = scatter_gdn_ckpts_fast(pool, conv_ck, rec_ck, wb, layout=layout, interpret=interpret)\n'),
            ('        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32)); xs.update(_live_xs(ex, meta[3], NS, T)); xs.update(_ink_xs(fn, wb))  # noqa: E702  q38 X-GLUE-1\n',
             '        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32)); xs.update(_live_xs(ex, meta[3], NS, T))\n'),
            ('        outs = fn(pos0, x, *(xs[k] for k in fn.extra_names), *w); outs = _ink_outs(fn, outs, pool, interpret)  # noqa: E702  q38 X-GLUE-1\n',
             '        outs = fn(pos0, x, *(xs[k] for k in fn.extra_names), *w)\n'),
            ('        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32)); xs.update(_live_xs(ex, qsl, NS, T)); xs.update(_ink_xs(fn, wb))  # noqa: E702  q38 X-GLUE-1\n',
             '        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32)); xs.update(_live_xs(ex, qsl, NS, T))\n'),
            ('        outs = fn(pos0, x, *(xs[k] for k in fn.extra_names), *w); outs = _ink_outs(fn, outs, pool, interpret)  # noqa: E702  q38 X-GLUE-1 (call_shared)\n',
             '        outs = fn(pos0, x, *(xs[k] for k in fn.extra_names), *w)\n'),
        ],
        GLUE_MARK,
    ),
    RT + "qwen38_segment_call.py": (
        [
            ('    fn, _, _, n_w = make_flat_segment_step(mesh, spec, ex, extra_outs=_ink_extra_outs(spec, interpret, layout))  # q38 X-GLUE-1\n',
             '    fn, _, _, n_w = make_flat_segment_step(mesh, spec, ex)\n'),
            ('        ) if (c.shape, c.dtype) != (pool.shape, pool.dtype) else (c if interpret else pool),  # q38 X-GLUE-1: in-kernel checkpoints pass the pool through\n',
             '        ),\n'),
            ('                named.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32)); named.update(_ink_xs(fn, wb))  # noqa: E702  q38 X-GLUE-1\n',
             '                named.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32))\n'),
            ('        outs = fn(pos, x, *xs, *w, *states); outs = _ink_outs(fn, outs, pool, interpret)  # noqa: E702  q38 X-GLUE-1\n',
             '        outs = fn(pos, x, *xs, *w, *states)\n'),
        ],
        GLUE_MARK,
    ),
}


def strip_glue(path: str, text: str) -> str:
    """Undo X-GLUE-1's pinned lines and cut its appended block (from GLUE_MARK to the end of the file; each present exactly
    once); no-op for other files. The result must not mention X-GLUE-1."""
    if path not in GLUE_EDITS:
        return text
    pairs, mark = GLUE_EDITS[path]
    if mark is not None:
        assert text.count(mark) == 1, (path, "X-GLUE-1 block count != 1")
        text = text[: text.index(mark)]
    for new, old in pairs:
        assert text.count(new) == 1, (path, new)
        text = text.replace(new, old)
    assert "X-GLUE-1" not in text, (path, "an X-GLUE-1 line outside the pinned edits")
    return text


def strip_idx(path: str, text: str) -> str:
    """Undo L1-IDX's pinned lines and cut its appended block (each present exactly once); no-op for other files. The
    result must not mention L1-IDX."""
    if path not in IDX_EDITS:
        return text
    pairs, mark = IDX_EDITS[path]
    for new, old in pairs:
        assert text.count(new) == 1, (path, new)
        text = text.replace(new, old)
    if mark is not None:
        assert text.count(mark) == 1, (path, "L1-IDX block count != 1")
        text = text[: text.index(mark)]
    assert "L1-IDX" not in text, (path, "an L1-IDX line outside the pinned edits")
    return text


# q38 X-K-LIVE / L2 D73 gate (STACK-D v4): file -> ([(member text, the text before it)], appended-block mark or None); each
# member text occurs exactly once and an appended block runs from its mark to the end of the file. Undone before every
# other strip (the whole-step call's X-K-LIVE block follows the L1-IDX block, which strip_idx cuts to the end of the file).
LIVE_EDITS = {
    MKD + 'blocks/gdn/recurrence.py': (
        [
            ('    if int(ctx.ex.live_rows) >= 2: return gdn_rec_live(ctx, gp, u, ybuf, slot)  # noqa: E701  Gated delta rule on key-major `recb [B * vh, KD, VD]` (`GCH=False`); q38 X-K-LIVE dispatch, same line\n',
             '    """Gated delta rule on key-major `recb [B * vh, KD, VD]` (`GCH=False`)."""\n'),
            ('    if int(ctx.ex.live_rows) >= 2: return gdn_rec_chunked_live(ctx, gp, o, slot)  # noqa: E701  Gated delta rule over `NRB`-buffered row chunks (`GCH=True`); q38 X-K-LIVE dispatch, same line\n',
             '    """Gated delta rule over `NRB`-buffered row chunks (`GCH=True`)."""\n'),
        ],
        '\n\n# q38 X-K-LIVE begin (L1 GDN half): pad requests skip the recurrence. Appended after every existing line, and the\n',
    ),
    MKD + 'blocks/gdn/states.py': (
        [
            ('    if int(ctx.ex.live_rows) >= 2 and ctx.ex.VER: return rec_in_live(ctx, gp, o)  # noqa: E701  Whole-state `rec` reads of layer `o` (`GCH=False`; per sequence in verify); q38 X-K-LIVE dispatch, same line\n',
             '    """Whole-state `rec` reads of layer `o` (`GCH=False`; per sequence in verify)."""\n'),
        ],
        "\n\n# q38 X-K-LIVE begin (L1 GDN half): appended; `rec_in`'s dispatch replaces its docstring line (no line moves).\n",
    ),
    MKD + 'blocks/moe/block.py': (
        [
        ],
        '\n    # q38 X-K-LIVE (L3-LIVE): the `MoEDebug.hit_trace` outputs (tests only); appended after every traced method.\n',
    ),
    MKD + 'blocks/moe/dedup.py': (
        [
            ('        if m.dbg.row_mask or ex.live_rows:  # q38 X-K-LIVE (L3-LIVE): + the production live-row mask\n            on = _row_on(m, b)\n',
             '        if m.dbg.row_mask:\n            on = ctx.extra["moe_rows"][b] != 0\n'),
        ],
        '\n\n# q38 X-K-LIVE begin (L3-LIVE). Appended so the flag-off lowering of every function above keeps its source locations.\n',
    ),
    MKD + 'blocks/moe/params.py': (
        [
            ("    weighted output of the experts in that list).\n\n    `hit_trace` (q38 X-K-LIVE, L3-LIVE; tests only) adds no input but three per-layer outputs that copy what the\n    routing decided (`blocks/moe/trace.py`): the rows' picks (`hitv`), the hit list (`elist`) and the expert-unit\n    count (`st` = n_hit * TE, the units `route_fetch` and the refills issue as expert-weight DMAs).\n",
             '    weighted output of the experts in that list).\n'),
            ('    row_mask: bool = False\n    hit_trace: bool = False\n',
             '    row_mask: bool = False\n'),
            ('    assert not (dbg.row_mask and ecopy), "row_mask: identity expert placement only"\n    if ex.live_rows and ecopy:  # q38 X-K-LIVE (L3-LIVE): exclusive with X-K-ECOPY until composed\n        raise MegakernelConfigError(\n            "live_rows (X-K-LIVE) with redundant expert copies (ECOPY, e_slots > E / R) is not composed yet"\n        )\n    if ex.live_rows and dbg.row_mask:  # one row-mask source per kernel\n        raise MegakernelConfigError("live_rows and the MoEDebug.row_mask debug input are exclusive")\n',
             '    assert not (dbg.row_mask and ecopy), "row_mask: identity expert placement only"\n'),
        ],
        None,
    ),
    MKD + 'blocks/moe/routing.py': (
        [
            ('                )\n    if m.dbg.hit_trace:  # q38 X-K-LIVE (L3-LIVE): debug engagement trace (tests only)\n        from .trace import hit_trace\n\n        hit_trace(m, l_idx)\n',
             '                )\n'),
        ],
        None,
    ),
    MKD + 'exec_spec.py': (
        [
            ('    cache: Any = None\n    # q38 X-K-LIVE: the kernel takes the live-row mask `live_rows` (SMEM int32 [B]; planner.extra_inputs) and\n    # blocks may skip pad rows (megakernel/live_rows.py). Only the whole-step call sets it.\n    live_rows: bool = False\n',
             '    cache: Any = None\n'),
        ],
        None,
    ),
    MKD + 'models/qwen38_gdn_pool.py': (
        [
            ('        """Layer `o`\'s pool states -> `pg_rec` (`REC`) / `pg_conv` (f32 history rows). `direct`: the states go\n        to each request\'s first `recb` row group instead (the slot `rec_in` would copy them to), once the previous\n        GDN layer\'s `recb` write-back has drained; `gdn_in` then skips both. `ssm=False` (chunk gather): only the\n        conv history; the chunks read their states (`_ChunkIn`). q38 X-K-LIVE: with `ctx.ex.live_rows` the next\n        line hands over to `_gather_live` (pad requests skipped); it replaces a docstring line (no line moves)."""\n        if ctx.ex.live_rows: return self._gather_live(ctx, o, direct, ssm)  # noqa: E701\n',
             '        """Layer `o`\'s pool states -> `pg_rec` (`REC`) / `pg_conv` (f32 history rows).\n\n        `direct`: the states go to each request\'s first `recb` row group instead\n        (the slot `rec_in` would copy them to), once the previous GDN layer\'s\n        `recb` write-back has drained; `gdn_in` then skips both. `ssm=False` (chunk\n        gather): only the conv history; the chunks read their states (`_ChunkIn`)."""\n'),
            ('            st["rec_chunk_in"] = _ChunkInLive if ctx.ex.live_rows else _ChunkIn  # q38 X-K-LIVE\n',
             '            st["rec_chunk_in"] = _ChunkIn\n'),
        ],
        '\n    # q38 X-K-LIVE begin (L1 GDN half): appended to the class (no existing line moves).\n',
    ),
    MKD + 'planner.py': (
        [
            ('        out.extend(b.extra_inputs(g, ex))\n    if getattr(ex, "live_rows", False):  # q38 X-K-LIVE: one model-level input, read by any block\n        from .live_rows import live_rows_input\n\n        out.append(live_rows_input(ex))\n',
             '        out.extend(b.extra_inputs(g, ex))\n'),
        ],
        None,
    ),
    RT + 'qwen38_wholestep_call.py': (
        [
            ('        interpret=interpret, debug=DebugOptions(scopes=scopes), live_rows=_live_rows_on(B),\n',
             '        interpret=interpret, debug=DebugOptions(scopes=scopes),\n'),
            ('        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32)); xs.update(_live_xs(ex, meta[3], NS, T))\n',
             '        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32))\n'),
            ('        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32)); xs.update(_live_xs(ex, qsl, NS, T))\n',
             '        xs.update(gdn_pool=pool, gdn_rb=rb, gdn_hi=hi.astype(jnp.int32))\n'),
        ],
        '# q38 X-K-LIVE begin\n',
    ),
}
D73_EDITS = {
    RT + 'qwen38_wholestep_cache.py': (
        [
            ('\nfrom vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (\n',
             '\nfrom vllm_torchtpu import envs\nfrom vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (\n'),
            ("      `VLLM_TPU_QSA_FUSED_KV_WRITE` (D73) does not change this choice (q38 L2 D73 gate): D73's\n      per-layer fused writer is for stock steps only; on the whole step it regressed at 64 rows\n      (MM, +0.095 ms), so `batched` always takes `_batched_writes`. `batched=False` keeps the\n      per-layer stock sequence (where `qsa_cache_writes` still honours the env).\n",
             '      The fused-writer path (`VLLM_TPU_QSA_FUSED_KV_WRITE`) keeps the per-layer sequence.\n'),
            ('    if batched:  # q38 L2 D73 gate: D14 ignores VLLM_TPU_QSA_FUSED_KV_WRITE (stock steps only)\n',
             '    if batched and not envs.VLLM_TPU_QSA_FUSED_KV_WRITE:\n'),
        ],
        None,
    ),
}


def _strip_pinned(edits: dict, tag: str, path: str, text: str) -> str:
    if path not in edits:
        return text
    pairs, mark = edits[path]
    for new, old in pairs:
        assert text.count(new) == 1, (path, tag, new[:80], text.count(new))
        text = text.replace(new, old)
    if mark is not None:
        assert text.count(mark) == 1, (path, tag + " block count != 1")
        text = text[: text.index(mark)]
    return text


# q38 L1-GDNINK (VLLM_TPU_MK_GDN_INK): file -> ([(GDNINK text, the X-K-LIVE text before it)], appended-block mark). Undone before
# strip_live (the GDNINK block follows the X-K-LIVE block, which strip_live cuts to the end of the file).
GINK_EDITS = {
    MKD + 'blocks/gdn/recurrence.py': (
        [
            ('    if int(ctx.ex.live_rows) >= 2 or gdnink_ws(ctx): return (gdn_rec_live if int(ctx.ex.live_rows) >= 2 else gdn_rec_wshm)(ctx, gp, u, ybuf, slot)  # noqa: E701  Gated delta rule on key-major `recb [B * vh, KD, VD]` (`GCH=False`); q38 X-K-LIVE dispatch, same line; q38 L1-GDNINK dispatch, same line\n',
             '    if int(ctx.ex.live_rows) >= 2: return gdn_rec_live(ctx, gp, u, ybuf, slot)  # noqa: E701  Gated delta rule on key-major `recb [B * vh, KD, VD]` (`GCH=False`); q38 X-K-LIVE dispatch, same line\n'),
        ],
        '\n\n# q38 L1-GDNINK begin (GDN-REC-WSHM, VLLM_TPU_MK_GDN_INK=1): the head-major state update on the whole-state path (<= 16 rows,\n',
    ),
    RT + 'qwen38_wholestep_call.py': (
        [
            ('    return t if not _gdnink_rel_fetch(B) else dataclasses.replace(t, rel_fetch=True)  # q38 L1-GDNINK (GDN-REL-FETCH), same line\n',
             '    return t\n'),
        ],
        "\n\n# q38 L1-GDNINK begin (GDN-REL-FETCH, VLLM_TPU_MK_GDN_INK bit 2): the whole step's dense weight-stream refills decoded at trace\n",
    ),
}


def strip_gink(path: str, text: str) -> str:
    """q38 L1-GDNINK: undo its pinned edit and cut its appended block (each present exactly once); no-op for other files. The
    result must not mention L1-GDNINK."""
    text = _strip_pinned(GINK_EDITS, "L1-GDNINK", path, text)
    if path in GINK_EDITS:
        assert "L1-GDNINK" not in text, (path, "an L1-GDNINK line outside the pinned edits")
    return text


def strip_live(path: str, text: str) -> str:
    """q38 X-K-LIVE: undo its pinned edits and cut its appended block (each present exactly once); no-op for other files. The
    result must not mention X-K-LIVE."""
    text = _strip_pinned(LIVE_EDITS, "X-K-LIVE", path, text)
    if path in LIVE_EDITS:
        assert "X-K-LIVE" not in text, (path, "an X-K-LIVE line outside the pinned edits")
    return text


def strip_d73(path: str, text: str) -> str:
    """q38 L2 D73 gate: undo its pinned lines (each present exactly once); no-op for other files."""
    text = _strip_pinned(D73_EDITS, "D73", path, text)
    if path in D73_EDITS:
        assert "D73" not in text, (path, "a D73 line outside the pinned edits")
    return text


# q38 L1-QX (X-DCP-QX-EARLY): file -> ([(L1-QX text, the text before it)], appended-block mark); undone first.
QX_EDITS = {
    RT + "qwen38_wholestep_call.py": (
        [
            ("    return dict(out, **_qx_options())  # q38 L1-QX (qx_early; end of module)\n",
             "    return out\n"),
        ],
        "\n\n# q38 L1-QX begin\n",
    ),
}


# q38 L1-SELLOOP: file -> ([(SELLOOP line, the line before it)], appended-block mark); the block runs from its mark to the end of
# the file (after X-K-LIVE's block), so it is cut before strip_live cuts X-K-LIVE's.
SELLOOP_EDITS = {
    RT + "qwen38_wholestep_call.py": (
        [
            ("        compressed_dcp=compressed_dcp(), **_d08s1_options(), **_idx_options(), **_selloop_options(),  # q38 L1-D08S1 / L1-IDX / L1-SELLOOP\n",
             "        compressed_dcp=compressed_dcp(), **_d08s1_options(), **_idx_options(),  # q38 L1-D08S1 / L1-IDX\n"),
        ],
        "# q38 L1-SELLOOP begin\n",
    ),
}


def strip_qx(path: str, text: str) -> str:
    """q38 L1-QX: undo its pinned line and cut its appended block (each present exactly once); no-op for other files. The
    result must not mention L1-QX."""
    text = _strip_pinned(QX_EDITS, "L1-QX", path, text)
    if path in QX_EDITS:
        assert "L1-QX" not in text, (path, "an L1-QX line outside the pinned edits")
    return text


# q38 L1-L47F (L47-FOLD-S1): file -> ([(L47F text, the text before it)], appended-block mark or None); undone first
# (inside strip_v5). qwen38_wholestep.py also carries L47F_BLOCKS marked blocks ("# q38 L1-L47F begin" .. "end" lines,
# cut first). Generated from the diff vs ecf4cc34a (scratch/overflow/l1_l47f_work/gen_l47f_block.py).
L47F_EDITS = {
    MKD + 'planner.py': (
        [
            ('    n_cycles, rem = divmod(L, period)  # q38 L1-L47F (port of dongruizi a18f2ea44): a partial last cycle\n    assert rem == 0 or period > 1, (L, period)  # q38 L1-L47F: = the first `rem` layers of the cycle (branch order)\n',
             '    assert L % period == 0, (L, period)\n    n_cycles = L // period\n'),
            ('                    block_layer_counts[b.name] = n_cycles * c_b + min(c_b, max(0, rem - sum(counts[:b_idx])))  # q38 L1-L47F\n',
             '                    block_layer_counts[b.name] = n_cycles * c_b\n'),
            ('    T_TOTAL = (g.L // period) * t_cycle + sum(  # q38 L1-L47F: + the partial last cycle\'s tiles (0 when L % period == 0)\n        min(c_b, max(0, g.L % period - sum(branch_counts[:i]))) * t_b for i, (c_b, t_b) in enumerate(zip(branch_counts, branch_tiles_list)))\n    assert g.L % period == 0 or not tiles.rel_fetch, "q38 L1-L47F: a partial last cycle needs rel_fetch off (fetch_rel stops at L // period)"\n',
             '    T_TOTAL = (g.L // period) * t_cycle\n'),
        ],
        None,
    ),
    MKD + 'models/qwen38.py': (
        [
            ('    # q38 L1-L47F (port of dongruizi a18f2ea44): a partial last cycle (`L % att_every == 1`) is one QSA layer, the whole\n    # step\'s cycles starting at their QSA layer (`first="qsa"`, model layers 3..47 = L 45: 12 QSA + 33 GDN layers).\n    @property\n    def NATT(self) -> int:  # type: ignore[override]\n        n, rem = divmod(self.L, self.att_every)\n        return n + int(rem == 1 and self.att_every <= self.L)\n\n    @property\n    def NGDN(self) -> int:  # type: ignore[override]\n        return self.L - self.NATT\n    # q38 L1-L47F end\n\n',
             ''),
            ('        assert self.att_every > self.L or self.L % self.att_every in (0, 1)  # q38 L1-L47F: (0, 1), was == 0\n',
             '        assert self.att_every > self.L or self.L % self.att_every == 0\n'),
        ],
        None,
    ),
    WS + 'spec.py': (
        [
            ('    assert L % g.att_every == 0 or (first == "qsa" and L % g.att_every == 1), (L, g.att_every)  # q38 L1-L47F: L 45 too\n',
             '    assert L % g.att_every == 0 or (L == 1 and first == "qsa"), (L, g.att_every)\n'),
            ('        from .qsa_full_l47f import lean_step_block  # q38 L1-D08S1 / L1-IDX / L1-L47F: None unless a D08 S1 / L1-IDX option (or the test probe) is set\n',
             '        from .qsa_full_idx import lean_step_block  # q38 L1-D08S1 / L1-IDX: None unless a D08 S1 / L1-IDX option (or the test probe) is set\n'),
        ],
        None,
    ),
    RT + 'qwen38_wholestep_call.py': (
        [
            ("    n_att = -(-L // g.att_every)  # q38 L1-L47F: L 45's partial last cycle is its QSA layer (= L // att_every at L 44)\n",
             '    n_att = L // g.att_every\n'),
            ('            n_layers=n_att, **_l47f_options(L, g.att_every),  # q38 L1-L47F: the sink export ({} when off)\n',
             '            n_layers=n_att,\n'),
            ('    stats = union_stats_on() or _l47f_sink_out(L, g.att_every)  # q38 L1-L47F: outs[6] = the sink export (not with stats)\n',
             '    stats = union_stats_on()\n'),
            ('        extra_outs=(KV_NEW, KR_NEW, IR_NEW) + ((UNION_STATS,) if union_stats_on() else ()) + _l47f_outs(L, g.att_every) + _ink_extra_outs(spec, interpret, layout),  # q38 X-GLUE-1 / L1-L47F\n',
             '        extra_outs=(KV_NEW, KR_NEW, IR_NEW) + ((UNION_STATS,) if stats else ()) + _ink_extra_outs(spec, interpret, layout),  # q38 X-GLUE-1\n'),  # STACK-D v5: the composed line -> X-GLUE-1's
        ],
        '\n\n# q38 L1-L47F begin\n',  # STACK-D v5: with its two leading blank lines, so the QX block before it ends the stripped text
    ),
    RT + 'qwen38_wholestep.py': (
        [
            ('            from .qwen38_l47fold import build_wholestep_call_l47f as build_wholestep_call  # q38 L1-L47F (itself when off)\n',
             '            from .qwen38_wholestep_call import build_wholestep_call\n'),
            ('                L=L_WHOLE + int(self.l47f),  # q38 L1-L47F: 45 with the fold\n',
             '                L=L_WHOLE,\n'),
            ('        ids = list(range(FIRST_LAYER, LAST_LAYER + int(l47f_on())))  # q38 L1-L47F: + model layer 47 (the last QSA layer)\n',
             '        ids = list(range(FIRST_LAYER, LAST_LAYER))\n'),
            ('            L=len(ids),  # q38 L1-L47F: L_WHOLE when off\n',
             '            L=L_WHOLE,\n'),
        ],
        None,
    ),
}
L47F_BEGIN, L47F_END = "# q38 L1-L47F begin", "# q38 L1-L47F end"
L47F_BLOCKS = {RT + "qwen38_wholestep.py": 6}


def strip_l47f(path: str, text: str) -> str:
    """q38 L1-L47F: cut its marked blocks (the pinned count, never nested; marker lines included), undo its pinned texts and
    cut its appended block (each present exactly once); no-op for other files. The result must not mention L1-L47F."""
    if path in L47F_BLOCKS:
        out, inside, seen = [], False, 0
        for line in text.splitlines(keepends=True):
            tag = line.strip()
            if tag == L47F_BEGIN:
                assert not inside, (path, "nested L47F block")
                inside, seen = True, seen + 1
            elif tag == L47F_END:
                assert inside, (path, "L47F end without begin")
                inside = False
            elif not inside:
                out.append(line)
        assert not inside and seen == L47F_BLOCKS[path], (path, "L47F blocks", seen, "want", L47F_BLOCKS[path])
        text = "".join(out)
    text = _strip_pinned(L47F_EDITS, "L1-L47F", path, text)
    if path in L47F_EDITS:
        assert "L1-L47F" not in text, (path, "an L1-L47F line outside the pinned edits")
    return text


def strip_selloop(path: str, text: str) -> str:
    """q38 L1-SELLOOP: undo its pinned line and cut its appended block (each present exactly once); no-op for other files.
    The result must not mention L1-SELLOOP."""
    text = _strip_pinned(SELLOOP_EDITS, "L1-SELLOOP", path, text)
    if path in SELLOOP_EDITS:
        assert "L1-SELLOOP" not in text, (path, "an L1-SELLOOP line outside the pinned edits")
    return text


# q38 L1-KVW (KVW-ALL + QSA-WRITE-FOLD stage A, wired by q38 L1-V6; STACK-D v6): file -> ([(KVW text, the text before it)],
# appended-block mark). Each block runs from its mark to the end of the file (in the whole-step call after L1-SELLOOP's),
# so strip_kvw runs first (inside strip_v6). The cache module's pinned dispatch line is D73's line extended in place.
KVW_EDITS = {
    RT + "qwen38_wholestep_call.py": (
        [
            ('        extra_outs=(KV_NEW, KR_NEW, IR_NEW) + ((UNION_STATS,) if union_stats_on() else ()) + _l47f_outs(L, g.att_every) + _kvw_extra_outs() + _ink_extra_outs(spec, interpret, layout),  # q38 X-GLUE-1 / L1-L47F / L1-KVW\n',
             '        extra_outs=(KV_NEW, KR_NEW, IR_NEW) + ((UNION_STATS,) if union_stats_on() else ()) + _l47f_outs(L, g.att_every) + _ink_extra_outs(spec, interpret, layout),  # q38 X-GLUE-1 / L1-L47F\n'),
            ('            num_requests=M // T, dcp=dcp, compressed_dcp=dcp if compressed_dcp() else None, **_kvw_kw(kv, rank[int(dcp_size > 1):], Bq),  # q38 L1-KVW\n',
             '            num_requests=M // T, dcp=dcp, compressed_dcp=dcp if compressed_dcp() else None,\n'),
            ('        + ((tp,) if dcp_size > 1 else ()) + _kvw_specs(tp),  # q38 L1-KVW: + (ck_new, ring_new) under KVAL\n',
             '        + ((tp,) if dcp_size > 1 else ()),\n'),
            ('                      trig, norms, *rank, *_kvw_outs(outs))  # q38 L1-KVW\n',
             '                      trig, norms, *rank)\n'),
            ('                      trig, norms, *rank, *_kvw_outs(outs))  # q38 L1-KVW (call_shared)\n',
             '                      trig, norms, *rank)\n'),
        ],
        "# q38 L1-KVW begin\n",
    ),
    RT + "qwen38_wholestep_cache.py": (
        [
            ('    compressed_dcp: tuple[Any, int] | None = None, kernel_values: Any = None,  # q38 L2 WFOLD / L1-KVW (end of module)\n',
             '    compressed_dcp: tuple[Any, int] | None = None,\n'),
            ('    if batched or _kvw_unbatched(kernel_values):  # q38 L2 D73 gate: D14 ignores VLLM_TPU_QSA_FUSED_KV_WRITE (stock steps only); q38 L1-KVW\n        return _kvw_writer(nreq, kernel_values, _batched_writes)(  # q38 L1-KVW: `_batched_writes` unless VLLM_TPU_MK_QSA_WRITE_ALL\n',
             '    if batched:  # q38 L2 D73 gate: D14 ignores VLLM_TPU_QSA_FUSED_KV_WRITE (stock steps only)\n        return _batched_writes(\n'),
        ],
        "\n\n# q38 L1-KVW begin\n",
    ),
}


def strip_kvw(path: str, text: str) -> str:
    """q38 L1-KVW: undo its pinned lines and cut its appended block (each present exactly once); no-op for other files.
    The result must not mention L1-KVW."""
    text = _strip_pinned(KVW_EDITS, "L1-KVW", path, text)
    if path in KVW_EDITS:
        assert "L1-KVW" not in text, (path, "an L1-KVW line outside the pinned edits")
    return text


def strip_v4(path: str, text: str) -> str:
    """q38 STACK-D v4: X-K-LIVE's and the D73 gate's edits, undone before strip_idx."""
    return strip_d73(path, strip_live(path, strip_gink(path, text)))  # q38 L1-GDNINK first


def strip_v5(path: str, text: str) -> str:
    """q38 STACK-D v5: the members appended after v4's blocks, undone before strip_v4 in reverse append order (each cuts
    from its mark to the end of the file): L1-L47F, L1-QX, then X-GLUE-1; STACK-D v6's members (strip_v6) first."""
    return strip_glue(path, strip_qx(path, strip_l47f(path, strip_v6(path, text))))


def strip_v6(path: str, text: str) -> str:
    """q38 STACK-D v6 (q38 L1-V6): the members appended after v5's blocks, undone first in reverse append order (each cuts
    from its mark to the end of the file): L2-XUAV5's X-UA tail (the last block of the whole-step call; strip_xuav5 also
    undoes X-UA's union_attn.py / spec.py edits), then L1-KVW (the last block of the cache module), then L1-SELLOOP.
    q38 L1-G1 (on v6): its pinned lines and appended blocks are undone before all of them (strip_g1)."""
    return strip_selloop(path, strip_kvw(path, strip_xuav5(path, strip_g1(path, text))))


# q38 L1-G1 (GDN-COMMIT1, VLLM_TPU_MK_GDN_COMMIT1): file -> ([(G1 text, the text before it)], appended-block mark). Each
# block runs from its mark to the end of the file (in the whole-step call after the X-UA tail), so strip_g1 runs first.
G1_EDITS = {
    RT + "qwen38_wholestep_call.py": (
        [
            ("        pool = _g1_scatter(scatter_gdn_ckpts_fast, write)(pool, conv_ck, rec_ck, wb, layout=layout, interpret=interpret) if (conv_ck.shape, conv_ck.dtype) != (pool.shape, pool.dtype) else (conv_ck if interpret else pool)  # q38 X-GLUE-1: in-kernel checkpoints pass the pool through; q38 L1-G1: no scatter with VLLM_TPU_MK_GDN_COMMIT1\n",
             "        pool = scatter_gdn_ckpts_fast(pool, conv_ck, rec_ck, wb, layout=layout, interpret=interpret) if (conv_ck.shape, conv_ck.dtype) != (pool.shape, pool.dtype) else (conv_ck if interpret else pool)  # q38 X-GLUE-1: in-kernel checkpoints pass the pool through\n"),
            ("        return _g1_ret((unpad(res), pool, outs[6]) if stats else (unpad(res), pool), write, conv_ck, rec_ck, wb, meta[0::GDN_META], NS, R)  # q38 L1-G1\n",
             "        return (unpad(res), pool, outs[6]) if stats else (unpad(res), pool)\n"),
            ("        return _g1_ret((unpad(res), pool, outs[6]) if stats else (unpad(res), pool), write, conv_ck, rec_ck, wb, sts, NS, R)  # q38 L1-G1 (call_shared)\n",
             "        return (unpad(res), pool, outs[6]) if stats else (unpad(res), pool)\n"),
            ("        return call_shared, in_specs, _g1_specs((tp, tp, tp) if stats else (tp, tp), write, tp, rep), (1 + n_w,)  # q38 L1-G1 (call_shared)\n",
             "        return call_shared, in_specs, (tp, tp, tp) if stats else (tp, tp), (1 + n_w,)\n"),
            ("    return call, in_specs, _g1_specs((tp, tp, tp) if stats else (tp, tp), write, tp, rep), (1 + n_w,)  # q38 L1-G1\n",
             "    return call, in_specs, (tp, tp, tp) if stats else (tp, tp), (1 + n_w,)\n"),
        ],
        "\n\n# q38 L1-G1 begin",
    ),
    RT + "qwen38_wholestep.py": (
        [
            ("            return _g1_stash(self, list(self.callable_for(int(hidden.shape[0]))(hidden, *args)), int(hidden.shape[0]))  # q38 L1-G1\n",
             "            return list(self.callable_for(int(hidden.shape[0]))(hidden, *args))\n"),
        ],
        "\n\n# q38 L1-G1 begin",
    ),
    RT + "qwen38_segment_call.py": (
        [
            ("        lambda pool, c, r, wb: _g1_scatter(scatter_fn, write and head)(  # q38 L1-G1: no head scatter with VLLM_TPU_MK_GDN_COMMIT1\n",
             "        lambda pool, c, r, wb: scatter_fn(\n"),
            ("        return _g1_ret((unpad(res), scatter(pool, conv_ck, rec_ck, wb)), write and head, conv_ck, rec_ck, wb, meta[0::GDN_META], NS, R)  # q38 L1-G1\n",
             "        return unpad(res), scatter(pool, conv_ck, rec_ck, wb)\n"),
            ("        return call, in_specs, _g1_specs((tp, tp), write and head, tp, rep), (n_lead + n_w,)  # q38 L1-G1\n",
             "        return call, in_specs, (tp, tp), (n_lead + n_w,)\n"),
        ],
        "\n\n# q38 L1-G1 begin",
    ),
    DRIVER: (
        [
            ("                return _g1_stash(self, list(call(hidden, *weights, *gdn_io)), B)  # q38 L1-G1\n",
             "                return list(call(hidden, *weights, *gdn_io))\n"),
            ("                _g1_stash(self, list(call(h, *weights, *io)), b)  # q38 L1-G1\n",
             "                call(h, *weights, *io)\n"),
        ],
        "\n\n# q38 L1-G1 begin",
    ),
}


def strip_g1(path: str, text: str) -> str:
    """q38 L1-G1: undo its pinned lines and cut its appended block (each present exactly once); no-op for other files and
    for text that no longer mentions L1-G1. The result must not mention L1-G1."""
    if path not in G1_EDITS or "L1-G1" not in text:
        return text
    text = _strip_pinned(G1_EDITS, "L1-G1", path, text)
    assert "L1-G1" not in text, (path, "an L1-G1 line outside the pinned edits")
    return text


# q38 L2-UAPIPE (X-UA-PIPE, VLLM_TPU_MK_UNION_PIPE): file -> ([(UAPIPE text, D08S1 text)], appended-block marker or None),
# undone before D08S1's (its edits sit on top of them); union_attn.py only gains a block appended after MM's last line.
# q38 L2-XUAV5: on F-v5c the whole-step call's pinned pair is the L1-IDX line (UAPIPE's option joins it on the same line);
# q38 L1-V6 (STACK-D v6): L1-SELLOOP's option sits before it on that line.
UAPIPE_EDITS = {
    WS + "union_attn.py": ([], "\n\n# ===================================================================== q38 L2-UAPIPE (X-UA-PIPE"),
    WS + "spec.py": (
        [
            ("        from .qsa_full_lean import pipe_step_block  # q38 L2-UAPIPE: None unless `union_pipe` is set "
             "(X-UA-PIPE)\n", ""),
            ("        qsa_block: Any = pipe_step_block(indexed, qsa) or lean_step_block(indexed, qsa) or (",
             "        qsa_block: Any = lean_step_block(indexed, qsa) or ("),
        ],
        None,
    ),
    RT + "qwen38_wholestep_call.py": (
        [
            ("        compressed_dcp=compressed_dcp(), **_d08s1_options(), **_idx_options(), **_selloop_options(), "
             "**_union_pipe_options(),  # q38 L1-D08S1 / L1-IDX / L1-SELLOOP, q38 L2-UAPIPE\n",
             "        compressed_dcp=compressed_dcp(), **_d08s1_options(), **_idx_options(), **_selloop_options(),  # q38 "
             "L1-D08S1 / L1-IDX / L1-SELLOOP\n"),  # q38 L1-V6: STACK-D v6, on L1-SELLOOP's line (undone by strip_selloop next)
        ],
        "\n\n# q38 L2-UAPIPE: X-UA-PIPE",
    ),
}


def strip_uapipe(path: str, text: str) -> str:
    """Undo L2-UAPIPE's pinned lines and cut its appended block (each present exactly once); no-op for other files and
    for text already stripped (no L2-UAPIPE line left; test_uapipe_files_differ_... asserts the edits are live)."""
    if path not in UAPIPE_EDITS or "L2-UAPIPE" not in text:
        return text
    pairs, mark = UAPIPE_EDITS[path]
    for new, old in pairs:
        assert text.count(new) == 1, (path, new)
        text = text.replace(new, old)
    if mark is not None:
        assert text.count(mark) == 1, (path, "UAPIPE block count != 1")
        text = text[: text.index(mark)].rstrip("\n") + "\n"  # the block was appended after the file's last line
    assert "L2-UAPIPE" not in text, (path, "an L2-UAPIPE line outside the pinned edits")
    return text


def strip_d08s1(path: str, text: str) -> str:
    """Undo D08S1's pinned lines and cut its appended block (each present exactly once); no-op for other files."""
    text = strip_uapipe(path, text)  # q38 L2-UAPIPE: its edits sit on top of D08S1's (and L2-UNPACK's)
    text = strip_unpack(path, text)  # q38 L2-UNPACK: its pinned line sits on top of D08S1's / D14's text
    if path not in D08S1_EDITS:
        return text
    pairs, mark = D08S1_EDITS[path]
    for new, old in pairs:
        assert text.count(new) == 1, (path, new)
        text = text.replace(new, old)
    if mark is not None:
        assert text.count(mark) == 1, (path, "D08S1 block count != 1")
        text = text[: text.index(mark)]
    return text


# q38 L1-D07B1: file -> (number of marked blocks, [(D07B1 line, MM's line)]); each D07B1 line occurs exactly once.
D07B1_BEGIN, D07B1_END = "# q38 L1-D07B1 begin", "# q38 L1-D07B1 end"
D07B1_EDITS = {
    RT + "qwen38_segment_call.py": (1, []),
    RT + "qwen38_wholestep_call.py": (3, []),
    RT + "qwen38_wholestep.py": (
        8,  # q38 L1-GLUEFIX: + the callable_for expansion block and the bind-time n_gdn block
        [("        for i in () if self.gdn_shared else self.gdn_ids:  # q38 L1-D07B1: shared below\n",
          "        for i in self.gdn_ids:\n")],
    ),
    DRIVER: (2, []),
}


# q38 L2-UAISSUE: file -> number of marked blocks (marker lines included, never nested).
UAISSUE_BEGIN, UAISSUE_END = "# q38 L2-UAISSUE begin", "# q38 L2-UAISSUE end"
UAISSUE_EDITS = {WS + "union_attn.py": 3}
# q38 L2-UNPACK: union_attn.py -> number of marked blocks (cut before UAISSUE's); qwen38_wholestep_call.py -> its pinned
# [(UNPACK line, D08S1 / D14 line)], undone before D08S1's edits.
UNPACK_BEGIN, UNPACK_END = "# q38 L2-UNPACK begin", "# q38 L2-UNPACK end"
UNPACK_BLOCKS = {WS + "union_attn.py": 2}
UNPACK_LINES = {
    RT + "qwen38_wholestep_call.py": [
        ("    slots = _uaunpack_slots([(lay.main_slot, lay.compressed_slot) for lay in qsa_layers])  # q38 L2-UNPACK\n",
         "    slots = [(lay.main_slot, lay.compressed_slot) for lay in qsa_layers]\n"),
    ],
}


def strip_unpack(path: str, text: str) -> str:
    """Cut L2-UNPACK's marked blocks / undo its pinned lines (each exactly as pinned); no-op for other files and for text
    already stripped (test_unpack_files_differ_... asserts the edits are live). union_attn.py: no line outside the blocks may
    mention L2-UNPACK; qwen38_wholestep_call.py: its functions sit inside D08S1's appended block (cut there)."""
    if path in UNPACK_BLOCKS and UNPACK_BEGIN in text:
        out, inside, seen = [], False, 0
        for line in text.splitlines(keepends=True):
            tag = line.strip()
            if tag == UNPACK_BEGIN:
                assert not inside, (path, "nested UNPACK block")
                inside, seen = True, seen + 1
            elif tag == UNPACK_END:
                assert inside, (path, "UNPACK end without begin")
                inside = False
            elif not inside:
                out.append(line)
        assert not inside and seen == UNPACK_BLOCKS[path], (path, "UNPACK blocks", seen, "want", UNPACK_BLOCKS[path])
        text = "".join(out)
        assert "L2-UNPACK" not in text, (path, "an L2-UNPACK line outside the marked blocks")
    if path in UNPACK_LINES and UNPACK_LINES[path][0][0] in text:
        for new, old in UNPACK_LINES[path]:
            assert text.count(new) == 1, (path, new)
            text = text.replace(new, old)
    return text


def strip_uaissue(path: str, text: str) -> str:
    """Cut UAISSUE's marked blocks (the pinned count, never nested); no line outside them may mention L2-UAISSUE."""
    text = strip_uapipe(path, text)  # q38 L2-UAMERGE: L2-UAPIPE's appended block (after MM's last line) is cut first
    text = strip_unpack(path, text)  # q38 L2-UNPACK: its blocks sit beside UAISSUE's
    if path not in UAISSUE_EDITS:
        return text
    out, inside, seen = [], False, 0
    for line in text.splitlines(keepends=True):
        tag = line.strip()
        if tag == UAISSUE_BEGIN:
            assert not inside, (path, "nested UAISSUE block")
            inside, seen = True, seen + 1
        elif tag == UAISSUE_END:
            assert inside, (path, "UAISSUE end without begin")
            inside = False
        elif not inside:
            out.append(line)
    assert not inside and seen == UAISSUE_EDITS[path], (path, "UAISSUE blocks", seen, "want", UAISSUE_EDITS[path])
    text = "".join(out)
    assert "L2-UAISSUE" not in text, (path, "an L2-UAISSUE line outside the marked blocks")
    return text


# q38 L2-XUAV5 (X-UA on F-v5c): the whole-step call's X-UA tail (X-UA-ISSUE's, X-UA-UNPACK's and X-UA-PIPE's functions,
# appended after L1-L47F's block) and its pinned lines inside _d08s1_options (D08S1's block): [(X-UA text, F-v5c text)].
XUAV5_CALL_MARK = "\n\n# q38 L2-UAISSUE: X-UA-ISSUE"
XUAV5_CALL_LINES = [
    ("    issue = {**_uaissue_options(), **_uaunpack_options()}  # q38 L2-UAISSUE, q38 L2-UNPACK: {} when off\n", ""),
    ("    if not any(on.values()):\n        return issue\n", "    if not any(on.values()):\n        return {}\n"),
    ("    return {**on, **issue}\n", "    return on\n"),
]
XUA_TAGS = ("L2-UAISSUE", "L2-UNPACK", "L2-UAPIPE")


def strip_xuav5(path: str, text: str) -> str:
    """Undo every X-UA edit (q38 L2-XUAV5): union_attn.py's UAPIPE block and UNPACK / UAISSUE marked blocks, spec.py's
    UAPIPE lines, the whole-step call's UAPIPE / UNPACK pinned lines, its X-UA tail (XUAV5_CALL_MARK to the end) and the
    XUAV5_CALL_LINES. No-op for other files and for text with no X-UA tag left; afterwards no X-UA tag remains."""
    if not any(tag in text for tag in XUA_TAGS):
        return text
    text = strip_uaissue(path, text)  # = strip_uapipe + strip_unpack (+ UAISSUE's marked blocks in union_attn.py)
    if path == RT + "qwen38_wholestep_call.py":
        assert text.count(XUAV5_CALL_MARK) == 1, (path, "X-UA tail count != 1")
        text = text[: text.index(XUAV5_CALL_MARK)].rstrip("\n") + "\n"  # the tail was appended after L1-L47F's block
        for new, old in XUAV5_CALL_LINES:
            assert text.count(new) == 1, (path, new)
            text = text.replace(new, old)
    for tag in XUA_TAGS:
        assert tag not in text, (path, "an X-UA line outside the pinned edits", tag)
    return text


# q38 L1-V3: L4 HOSTSUBS (a0c341406) and L5 SAB (3d5b5b677) add these exact, unmarked blocks to the driver (pure
# additions vs 965b3c6e7); each occurs exactly once and is cut before the marked insertions are stripped.
HOST_DRIVER_BLOCKS = (
    ('L4 HOSTSUBS', 'from vllm_torchtpu.runner import host_phases\n'),
    ('L4 HOSTSUBS', '                # q38 L4 host sub-marks (VLLM_TPU_HOST_PHASES_SUBS): split\n                # forward_dispatch around the whole-step call (no-ops when off).\n                host_phases.sub("fd_pre_whole")\n'),
    ('L4 HOSTSUBS', '                host_phases.sub("fd_whole_call")\n'),
    ('L5 SAB', '        # q38 L5 SAB (VLLM_TPU_SERVED_AB=whole:P[:D]): whole-step ON / OFF in alternating windows of P\n        # serving calls, OFF = the wholedry route; served_ab.py. None = off.\n        self.sab: Any = None\n'),
    ('L5 SAB', '            if envs.VLLM_TPU_SERVED_AB:\n                from vllm.distributed.parallel_state import get_tp_group\n\n                from . import served_ab\n\n                tp = get_tp_group()\n                self.sab = served_ab.from_env(envs.VLLM_TPU_SERVED_AB, tp.rank_in_group, tp.world_size)\n'),
    ('L5 SAB', '        if getattr(self, "sab", None) is not None:\n            self.sab.tick(runner)\n'),
    ('L5 SAB', '        if (\n            self.whole_program is not None\n            and getattr(self, "sab", None) is not None\n            and not self.sab.whole_on()\n        ):\n            # q38 L5 SAB OFF window: the wholedry route for this step (stock, precompiled).\n            self._note(f"ab_off_b{rows}")\n            self._count("drafts_stock", n_req * (T_VERIFY - 1))\n            return False, None\n'),
)


# q38 L6-V6 (q38-l6-v6): L6-MK (q38-l6-mk 7060ee3f0, lever mkctx) edits the driver's try_forward. Its exact edits,
# as (composed text, MM / b488e1cb0 text) pairs, each composed text present exactly once; strip_l6_mk undoes them
# first, so the driver checks below still compare the rest of the driver with MM's (37b960ea9) byte for byte.
L6_MK_DRIVER_EDITS = (
    ('from vllm_torchtpu.runner import host_phases\nfrom vllm_torchtpu.runner import l6_mk as _l6_mk\n\n',
     'from vllm_torchtpu.runner import host_phases\n\n'),
    ('            return False, None\n        # q38 L6 MK-CTX (runner/l6_mk.py, VLLM_TPU_L6_MK_CTX): with the lever on this call reuses the\n        # eligibility computed here, resolves check_real only where the check branch reads it, enters\n        # the whole-step contexts from cached factories and imports the call module once.\n        elig = None\n        if (\n',
     '            return False, None\n        if (\n'),
    ('            and self.whole_above == "stock"\n            and not (elig := self._whole_eligible(runner, n_req, rows))\n        ):\n',
     '            and self.whole_above == "stock"\n            and not self._whole_eligible(runner, n_req, rows)\n        ):\n'),
    ('            return False, None\n        mkc = _l6_mk.mkctx_on(self)  # read past the stock-above-whole route (it never uses the lever)\n        if (\n',
     '            return False, None\n        if (\n'),
    ('        try:\n            if not mkc:\n                check_real = bool(getattr(runner, "warmup_done", False)) and _real_traffic(kw, runner, n_req)\n            elif self.check_left > 0 and self.shadow is not None and (not self.check_rows or rows in self.check_rows):\n                check_real = bool(getattr(runner, "warmup_done", False)) and _real_traffic(kw, runner, n_req)\n            else:\n                check_real = False  # not read: the check branch below needs the condition just tested\n            if (\n',
     '        try:\n            check_real = bool(getattr(runner, "warmup_done", False)) and _real_traffic(kw, runner, n_req)\n            if (\n'),
    ('                and rows in self.wholecheck_rows\n                and (elig if (mkc and elig is not None) else (elig := self._whole_eligible(runner, n_req, rows)))\n                and (\n',
     '                and rows in self.wholecheck_rows\n                and self._whole_eligible(runner, n_req, rows)\n                and (\n'),
    ('                out = self._whole_shadow_step(runner, input_ids, positions, kw, rows, real, n_req)\n            elif elig if (mkc and elig is not None) else (elig := self._whole_eligible(runner, n_req, rows)):\n                self.whole_calls += 1\n',
     '                out = self._whole_shadow_step(runner, input_ids, positions, kw, rows, real, n_req)\n            elif self._whole_eligible(runner, n_req, rows):\n                self.whole_calls += 1\n'),
    ('                host_phases.sub("fd_pre_whole")\n                if mkc:\n                    _l6_mk.note_whole()\n                    if _l6_mk.check_turn(_l6_mk.MKCTX):\n                        _l6_mk.mkctx_check(self, runner, n_req, rows, elig)\n                with (\n                    _l6_mk.whole_contexts(self, runner)\n                    if mkc\n                    else self._contexts(runner, "model_mk_whole")\n                ):\n                    out = self.whole_program(\n',
     '                host_phases.sub("fd_pre_whole")\n                with self._contexts(runner, "model_mk_whole"):\n                    out = self.whole_program(\n'),
    ('                host_phases.sub("fd_whole_call")\n                if mkc:\n                    wsc = _l6_mk.wsc_module()\n                else:\n                    from . import qwen38_wholestep_call as wsc\n\n',
     '                host_phases.sub("fd_whole_call")\n                from . import qwen38_wholestep_call as wsc\n\n'),
)


def strip_l6_mk(text: str) -> str:
    """q38 L6-V6: undo L6-MK's driver edits (each present exactly once)."""
    for new, old in L6_MK_DRIVER_EDITS:
        assert text.count(new) == 1, (new[:80], text.count(new))
        text = text.replace(new, old)
    return text


def strip_host_driver(text: str) -> str:
    """q38 L1-V3: cut L4 HOSTSUBS's and L5 SAB's exact driver blocks (each present exactly once)."""
    for tag, block in HOST_DRIVER_BLOCKS:
        assert text.count(block) == 1, (tag, block[:60], text.count(block))
        text = text.replace(block, "")
    return text


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tree_files():
    man = json.load(open(MANIFEST))
    out = set()
    for d in man["dirs"]:
        for dp, dns, fns in os.walk(os.path.join(ROOT, d)):
            dns[:] = [x for x in dns if x != "__pycache__"]
            for f in fns:
                if f.endswith(".pyc"):
                    continue
                out.add(os.path.relpath(os.path.join(dp, f), ROOT))
    return man, out


def strip_insertion(text: str, mark: str = MARK, end: str = END) -> str:
    """Cut one marked block (from mark through end, inclusive); exactly one such block must be present."""
    assert text.count(mark) == 1, f"marked insertion count {text.count(mark)} != 1"
    i = text.index(mark)
    j = text.index(end, i) + len(end)
    return text[:i] + text[j:]


def strip_census(text: str) -> str:
    """Cut the CENSUS_BLOCKS marked census blocks, each from the start of its MARK3 line through its END3 line; the
    blocks must not nest and their count is pinned."""
    assert text.count(MARK3) == CENSUS_BLOCKS and text.count(END3) == CENSUS_BLOCKS, (
        text.count(MARK3), text.count(END3))
    while MARK3 in text:
        k = text.index(MARK3)
        i = text.rfind("\n", 0, k) + 1
        j = text.index(END3, k) + len(END3)
        assert MARK3 not in text[k + len(MARK3) : j], "nested census block"
        text = text[:i] + text[j:]
    return text


def strip_d14(path: str, text: str) -> str:
    """Undo D14's pinned in-place lines and cut its appended block; each must be present exactly once."""
    text = strip_d08s1(path, text)  # q38 L1-D08S1: its edits sit on top of D14's
    pairs, mark = D14_EDITS[path]
    for new, old in pairs:
        assert text.count(new) == 1, (path, new)
        text = text.replace(new, old)
    if mark is not None:
        assert text.count(mark) == 1, (path, "D14 block count != 1")
        text = text[: text.index(mark)]
    return text


def strip_d07b1(path: str, text: str) -> str:
    """Cut D07B1's marked blocks (marker lines included; the pinned count, never nested) and undo its pinned lines."""
    n, pairs = D07B1_EDITS[path]
    out, inside, seen = [], False, 0
    for line in text.splitlines(keepends=True):
        tag = line.strip()
        if tag == D07B1_BEGIN:
            assert not inside, (path, "nested D07B1 block")
            inside, seen = True, seen + 1
        elif tag == D07B1_END:
            assert inside, (path, "D07B1 end without begin")
            inside = False
        elif not inside:
            out.append(line)
    assert not inside and seen == n, (path, "D07B1 blocks", seen, "want", n)
    text = "".join(out)
    for new, old in pairs:
        assert text.count(new) == 1, (path, new)
        text = text.replace(new, old)
    assert "L1-D07B1" not in text, (path, "a D07B1 line outside the pinned edits")
    return text


def test_manifest_covers_mm_tree():
    man = json.load(open(MANIFEST))
    assert man["source"] == "dongruizi / MM 37b960ea9"
    assert len(man["sha256"]) == 115


def test_no_added_or_missing_files():
    man, files = _tree_files()
    want = set(man["sha256"])
    assert files - want == BASE_ONLY, sorted(files - want)
    assert want - files == set(), sorted(want - files)


def test_imported_files_are_byte_identical_to_mm():
    man, files = _tree_files()
    bad = []
    for f, h in sorted(man["sha256"].items()):
        if f == DRIVER:
            continue
        data = strip_idx(f, strip_v4(f, strip_v5(f, open(os.path.join(ROOT, f), "rb").read().decode("utf-8")))).encode("utf-8")  # q38 L1-IDX / v4 / v5 (strip_v5 undoes X-UA first: q38 L2-XUAV5)
        if f in D07B1_EDITS:
            data = strip_d07b1(f, data.decode("utf-8")).encode("utf-8")
        if f in D14_EDITS:
            data = strip_d14(f, data.decode("utf-8")).encode("utf-8")
        elif f in UAPIPE_EDITS:  # q38 L2-UAPIPE (union_attn.py)
            data = strip_uapipe(f, data.decode("utf-8")).encode("utf-8")
        if _sha(data) != h:
            bad.append(f)
    assert not bad, bad


def test_d14_files_differ_from_mm_only_by_the_pinned_d14_edits():
    """q38 L1-D14: both files differ from MM's (the edits are live) and hash to MM's once they are undone."""
    man = json.load(open(MANIFEST))
    for f in D14_EDITS:
        text = strip_idx(f, strip_v4(f, strip_v5(f, open(os.path.join(ROOT, f), encoding="utf-8").read())))  # q38 L1-IDX / v4 / v5
        if f in D07B1_EDITS:
            text = strip_d07b1(f, text)
        assert _sha(text.encode("utf-8")) != man["sha256"][f], f
        assert _sha(strip_d14(f, text).encode("utf-8")) == man["sha256"][f], f


def test_driver_differs_from_mm_only_by_the_marked_insertions():
    man = json.load(open(MANIFEST))
    text = strip_d07b1(DRIVER, strip_l6_mk(strip_census(strip_g1(DRIVER, open(os.path.join(ROOT, DRIVER), encoding="utf-8").read()))))  # q38 L1-V6: census + L6-MK; q38 L1-G1 first
    text = strip_insertion(strip_insertion(strip_host_driver(text)), MARK2, END2)
    assert _sha(text.encode("utf-8")) == man["sha256"][DRIVER]


def test_d08s1_files_differ_from_d14_only_by_the_pinned_d08s1_edits():
    """q38 L1-D08S1: spec.py / qwen38_wholestep_call.py carry its live edits, undone exactly by strip_d08s1; qsa_full_b.py,
    qsa_full_c.py and union_attn.py carry none (with every S1 option off the traced lines are D14's / MM's)."""
    for f in D08S1_EDITS:
        text = strip_idx(f, strip_v4(f, strip_v5(f, open(os.path.join(ROOT, f), encoding="utf-8").read())))  # q38 L1-IDX / v4 / v5
        assert strip_d08s1(f, text) != text, f
        for new, _ in D08S1_EDITS[f][0]:
            assert text.count(new) == 1, (f, new)
    for f in ("qsa_full_b.py", "qsa_full_c.py", "union_attn.py", "indexer.py"):
        assert "L1-D08S1" not in open(os.path.join(ROOT, WS + f), encoding="utf-8").read(), f


def test_d07b1_files_differ_from_mm_only_by_the_pinned_d07b1_edits():
    """q38 L1-D07B1: each file differs from MM's (the edits are live) and hashes to MM's once they (then D14's or the driver's
    guard insertion) are undone."""
    man = json.load(open(MANIFEST))
    for f in D07B1_EDITS:
        text = strip_idx(f, strip_v4(f, strip_v5(f, open(os.path.join(ROOT, f), encoding="utf-8").read())))  # q38 L1-IDX / v4 / v5
        assert "L1-D07B1" in text, f
        assert _sha(text.encode("utf-8")) != man["sha256"][f], f
        if f == DRIVER:
            text = strip_l6_mk(text)  # q38 L6-V6: L6-MK's driver edits first
        text = strip_d07b1(f, text)
        if f in D14_EDITS:
            text = strip_d14(f, text)
        if f == DRIVER:
            text = strip_insertion(strip_insertion(strip_host_driver(strip_census(text))), MARK2, END2)
        assert _sha(text.encode("utf-8")) == man["sha256"][f], f


def test_idx_files_differ_only_by_the_pinned_idx_edits():
    """q38 L1-IDX: each IDX file carries its live edits (strip_idx changes it, and every pinned line occurs exactly once),
    the stripped text mentions L1-IDX nowhere, and MM's kernel files and the D08 S1 lean file carry no L1-IDX line."""
    for f in IDX_EDITS:
        text = strip_v5(f, open(os.path.join(ROOT, f), encoding="utf-8").read())  # q38 STACK-D v5 (X-GLUE-1, L1-QX)
        assert "L1-IDX" in text, f
        stripped = strip_idx(f, text)
        assert stripped != text, f
        for new, _ in IDX_EDITS[f][0]:
            assert text.count(new) == 1, (f, new)
    for f in ("qsa_full_b.py", "qsa_full_c.py", "union_attn.py", "indexer.py", "qsa_full_lean.py"):
        assert "L1-IDX" not in open(os.path.join(ROOT, WS + f), encoding="utf-8").read(), f


def test_live_and_d73_files_differ_only_by_their_pinned_edits():
    """q38 STACK-D v4: each X-K-LIVE / D73 file carries its live edits (strip_live / strip_d73 changes it and every pinned
    text occurs exactly once), and no other file of the manifest mentions X-K-LIVE or the D73 gate."""
    man = json.load(open(MANIFEST))
    for edits, strip, word in ((LIVE_EDITS, strip_live, "X-K-LIVE"), (D73_EDITS, strip_d73, "D73")):
        for f in edits:
            text = strip_gink(f, strip_v5(f, open(os.path.join(ROOT, f), encoding="utf-8").read()))  # q38 STACK-D v5 (X-GLUE-1, L1-QX, L1-L47F), then L1-GDNINK
            assert word in text, f
            assert strip(f, text) != text, f
            for new, _ in edits[f][0]:
                assert text.count(new) == 1, (f, new[:80])
        for f in sorted(set(man["sha256"]) - set(edits)):
            assert word not in open(os.path.join(ROOT, f), encoding="utf-8").read(), (f, word)


def test_glue_files_differ_only_by_the_pinned_glue_edits():
    """q38 X-GLUE-1: each GLUE file carries its live edits (every pinned line exactly once, the appended block once where
    pinned), strip_glue changes it and the stripped text mentions X-GLUE-1 nowhere; no other megakernel file mentions it
    except the new module."""
    for f, (pairs, mark) in GLUE_EDITS.items():
        text = strip_l47f(f, strip_v6(f, open(os.path.join(ROOT, f), encoding="utf-8").read()))  # STACK-D v5: L1-L47F's lines over X-GLUE-1's (STACK-D v6 first)
        assert "X-GLUE-1" in text, f
        for new, _ in pairs:
            assert text.count(new) == 1, (f, new)
        if mark is not None:
            assert text.count(mark) == 1, f
        assert strip_glue(f, text) != text, f
    man, files = _tree_files()
    for f in sorted(files):
        if f in GLUE_EDITS or f.endswith("qwen38_gdn_ckpt_ink.py") or not f.endswith(".py"):
            continue
        assert "X-GLUE-1" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), f


def test_qx_files_differ_only_by_their_pinned_edits():
    """q38 L1-QX: the whole-step call carries its live edits (strip_qx changes it, the pinned line occurs exactly once, the
    appended block ends the file), and no other file of the manifest mentions L1-QX."""
    man = json.load(open(MANIFEST))
    for f in QX_EDITS:
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert "L1-QX" in text, f
        stripped = strip_qx(f, text)
        assert stripped != text, f
        for new, _ in QX_EDITS[f][0]:
            assert text.count(new) == 1, (f, new[:80])
        assert strip_l47f(f, strip_v6(f, text)).endswith("# q38 L1-QX end\n"), f  # STACK-D v5: L1-L47F's block follows (v6's after it)
    for f in sorted(set(man["sha256"]) - set(QX_EDITS)):
        assert "L1-QX" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), (f, "L1-QX")


def test_l47f_files_differ_only_by_their_pinned_edits():
    """q38 L1-L47F: each L47F file carries its live edits (strip_l47f changes it; every pinned text occurs exactly once; the
    call module's appended block ends the file; qwen38_wholestep.py's marked blocks are the pinned count), the stripped
    texts mention L1-L47F nowhere, and no other file of the manifest mentions L1-L47F."""
    man = json.load(open(MANIFEST))
    for f in L47F_EDITS:
        text = strip_v6(f, open(os.path.join(ROOT, f), encoding="utf-8").read())  # STACK-D v6: L1-KVW extends a pinned line
        assert "L1-L47F" in text, f
        assert strip_l47f(f, text) != text, f
        for new, _ in L47F_EDITS[f][0]:
            assert text.count(new) == 1, (f, new[:80])
    assert strip_v6(RT + "qwen38_wholestep_call.py", open(os.path.join(ROOT, RT + "qwen38_wholestep_call.py"), encoding="utf-8").read()).endswith("# q38 L1-L47F end\n")  # STACK-D v6: v6 blocks follow
    for f in sorted(set(man["sha256"]) - set(L47F_EDITS)):
        assert "L1-L47F" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), (f, "L1-L47F")


def test_h3_adds_files_only():
    """q38 L1-H3: the draft attention kind lives in its two new files (BASE_ONLY); no manifest file mentions L1-H3."""
    man = json.load(open(MANIFEST))
    for f in ("src/vllm_torchtpu/megakernel/wholestep/qsa_draft.py",
              "src/vllm_torchtpu/megakernel_runtime/qwen38_draft_attn.py"):
        assert f in BASE_ONLY and "L1-H3" in open(os.path.join(ROOT, f), encoding="utf-8").read(), f
    for f in sorted(man["sha256"]):
        assert "L1-H3" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), f


def test_gdnink_files_differ_only_by_their_pinned_edits():
    """q38 L1-GDNINK: recurrence.py carries its live edits (strip_gink changes it, every pinned text occurs exactly once, and
    with them undone strip_live / strip_d73 find X-K-LIVE's pinned text), and no other file of the manifest mentions L1-GDNINK."""
    man = json.load(open(MANIFEST))
    for f in GINK_EDITS:
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert "L1-GDNINK" in text, f
        stripped = strip_gink(f, text)
        assert stripped != text, f
        for new, old in GINK_EDITS[f][0]:
            assert text.count(new) == 1, (f, new[:80])
            assert stripped.count(old) >= 1, (f, old[:80])  # a generic restored line (e.g. "    return t") may recur
        assert "gdn_rec_wshm" not in stripped and "gdnink" not in stripped and "rel_fetch=True" not in stripped, f
    for f in sorted(set(man["sha256"]) - set(GINK_EDITS)):
        assert "L1-GDNINK" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), f


def test_selloop_files_differ_only_by_the_pinned_selloop_edits():
    """q38 L1-SELLOOP: the whole-step call carries its live edits (strip_selloop changes it, the pinned line occurs exactly
    once, the appended block ends the file), the stripped text mentions L1-SELLOOP nowhere, and no other file of the manifest
    mentions L1-SELLOOP."""
    man = json.load(open(MANIFEST))
    for f, (pairs, mark) in SELLOOP_EDITS.items():
        text = strip_kvw(f, strip_xuav5(f, open(os.path.join(ROOT, f), encoding="utf-8").read()))  # STACK-D v6: later blocks first
        assert "L1-SELLOOP" in text, f
        assert strip_selloop(f, text) != text, f
        for new, _ in pairs:
            assert text.count(new) == 1, (f, new[:80])
        assert text.count(mark) == 1 and text.endswith("# q38 L1-SELLOOP end\n"), f  # STACK-D v6: after L1-KVW's / X-UA's blocks are cut
    for f in sorted(set(man["sha256"]) - set(SELLOOP_EDITS)):
        assert "L1-SELLOOP" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), (f, "L1-SELLOOP")


def test_kvw_files_differ_only_by_their_pinned_edits():
    """q38 L1-KVW (STACK-D v6): the whole-step call and the cache module carry their live edits (strip_kvw changes each,
    every pinned text occurs exactly once, the appended block ends the file and, cut, leaves L1-SELLOOP's block / the D73
    file's last line last), the stripped texts mention L1-KVW nowhere, and no other file of the manifest mentions L1-KVW."""
    man = json.load(open(MANIFEST))
    for f, (pairs, mark) in KVW_EDITS.items():
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert "L1-KVW" in text, f
        stripped = strip_kvw(f, text)
        assert stripped != text and "L1-KVW" not in stripped, f
        for new, _ in pairs:
            assert text.count(new) == 1, (f, new[:80])
        assert text.count(mark) == 1 and strip_xuav5(f, text).endswith("# q38 L1-KVW end\n"), f  # STACK-D v6: the X-UA tail follows in the call
    assert strip_kvw(RT + "qwen38_wholestep_call.py", open(os.path.join(ROOT, RT + "qwen38_wholestep_call.py"),
                                                         encoding="utf-8").read()).endswith("# q38 L1-SELLOOP end\n")
    for f in sorted(set(man["sha256"]) - set(KVW_EDITS)):
        assert "L1-KVW" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), (f, "L1-KVW")


def test_uaissue_files_differ_from_mm_only_by_the_marked_blocks():
    """q38 L2-UAISSUE: union_attn.py differs from MM's (the blocks are live) and hashes to MM's once they are cut;
    union_issue.py is the new file, and no MM-pinned file other than union_attn.py mentions L2-UAISSUE."""
    man, files = _tree_files()
    for f in UAISSUE_EDITS:
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert text.count("# q38 L2-UAISSUE begin") == UAISSUE_EDITS[f], f
        assert _sha(text.encode("utf-8")) != man["sha256"][f], f
        assert _sha(strip_uaissue(f, text).encode("utf-8")) == man["sha256"][f], f
    assert "src/vllm_torchtpu/megakernel/wholestep/union_issue.py" in files
    for f in sorted(set(man["sha256"]) - set(UAISSUE_EDITS) - {RT + "qwen38_wholestep_call.py"}):
        assert "L2-UAISSUE" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), f


def test_unpack_files_differ_only_by_the_pinned_unpack_edits():
    """q38 L2-UNPACK: union_attn.py carries its 2 live marked blocks (hashing to MM's once they and UAISSUE's are cut) and
    qwen38_wholestep_call.py its pinned line; union_unpack.py is the new file; no other MM-pinned file mentions L2-UNPACK."""
    man, files = _tree_files()
    ua = WS + "union_attn.py"
    text = open(os.path.join(ROOT, ua), encoding="utf-8").read()
    assert text.count(UNPACK_BEGIN) == UNPACK_BLOCKS[ua] and strip_unpack(ua, text) != text
    assert _sha(strip_uaissue(ua, text).encode("utf-8")) == man["sha256"][ua]
    for f, pairs in UNPACK_LINES.items():
        t = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert all(t.count(new) == 1 for new, _ in pairs) and strip_unpack(f, t) != t, f
    assert "src/vllm_torchtpu/megakernel/wholestep/union_unpack.py" in files
    for f in sorted(set(man["sha256"]) - set(UNPACK_BLOCKS) - set(UNPACK_LINES)):
        assert "L2-UNPACK" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), f


def test_uapipe_files_differ_only_by_the_pinned_uapipe_edits():
    """q38 L2-UAPIPE: union_attn.py, spec.py and qwen38_wholestep_call.py carry its live edits (union_attn.py hashes to
    MM's once its appended block and L2-UNPACK's / L2-UAISSUE's marked blocks are cut); qsa_full_b.py / qsa_full_c.py /
    indexer.py carry none. q38 L2-UAMERGE: X-UA-ISSUE / X-UA-UNPACK inside the pipe live in the appended block too."""
    man = json.load(open(MANIFEST))
    for f in UAPIPE_EDITS:
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert "L2-UAPIPE" in text and strip_uapipe(f, text) != text, f
    ua = WS + "union_attn.py"
    text = open(os.path.join(ROOT, ua), encoding="utf-8").read()
    assert _sha(strip_uaissue(ua, strip_uapipe(ua, text)).encode("utf-8")) == man["sha256"][ua]
    for f in ("qsa_full_b.py", "qsa_full_c.py", "indexer.py"):
        assert "L2-UAPIPE" not in open(os.path.join(ROOT, WS + f), encoding="utf-8").read(), f


def test_xuav5_files_differ_only_by_the_x_ua_edits():
    """q38 L2-XUAV5: the three X-UA-touched manifest files carry live X-UA edits, strip_xuav5 removes every X-UA tag (the
    whole-step call then ends with L1-KVW's block, q38 L1-V6 / STACK-D v6, and with L1-L47F's once strip_v6 is done; its X-UA tail present exactly once), it is a no-op on its own
    output, and no other manifest file mentions an X-UA tag."""
    man = json.load(open(MANIFEST))
    xf = (WS + "union_attn.py", WS + "spec.py", RT + "qwen38_wholestep_call.py")
    for f in xf:
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert any(tag in text for tag in XUA_TAGS), f
        stripped = strip_xuav5(f, text)
        assert stripped != text and not any(tag in stripped for tag in XUA_TAGS), f
        assert strip_xuav5(f, stripped) == stripped, f
    call = open(os.path.join(ROOT, RT + "qwen38_wholestep_call.py"), encoding="utf-8").read()
    assert call.count(XUAV5_CALL_MARK) == 1 and call.index(XUAV5_CALL_MARK) > call.index("# q38 L1-L47F end\n")
    assert strip_xuav5(RT + "qwen38_wholestep_call.py", call).endswith("# q38 L1-KVW end\n")  # q38 L1-V6: STACK-D v6 order
    assert strip_v6(RT + "qwen38_wholestep_call.py", call).endswith("# q38 L1-L47F end\n")
    for f in sorted(set(man["sha256"]) - set(xf)):
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert not any(tag in text for tag in XUA_TAGS), f


def test_g1_files_differ_only_by_their_pinned_edits():
    """q38 L1-G1 (GDN-COMMIT1): the whole-step call and the torch op module carry their live edits (strip_g1 changes each,
    every pinned text occurs exactly once, the appended block ends the file and, cut, leaves the X-UA tail / the module's
    flag-off text last), the stripped texts mention L1-G1 nowhere, qwen38_gdn_commit.py is a new file (BASE_ONLY), and no
    other file of the manifest mentions L1-G1."""
    man, files = _tree_files()
    for f, (pairs, mark) in G1_EDITS.items():
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert "L1-G1" in text, f
        stripped = strip_g1(f, text)
        assert stripped != text and "L1-G1" not in stripped, f
        for new, _ in pairs:
            assert text.count(new) == 1, (f, new[:80])
        assert text.count(mark) == 1 and text.endswith("# q38 L1-G1 end\n"), f
        assert strip_g1(f, stripped) == stripped, f
    call = open(os.path.join(ROOT, RT + "qwen38_wholestep_call.py"), encoding="utf-8").read()
    assert call.index(XUAV5_CALL_MARK) < call.index(G1_EDITS[RT + "qwen38_wholestep_call.py"][1])  # after the X-UA tail
    assert "src/vllm_torchtpu/megakernel_runtime/qwen38_gdn_commit.py" in files & BASE_ONLY
    for f in sorted(set(man["sha256"]) - set(G1_EDITS)):
        assert "L1-G1" not in open(os.path.join(ROOT, f), encoding="utf-8").read(), (f, "L1-G1")
