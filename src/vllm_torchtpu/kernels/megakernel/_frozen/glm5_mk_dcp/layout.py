"""Host-side layouts shared by the TP x DCP megakernel, its adapter and its tests.

`smalls_layout`: the per-layer f32 vector pack, one row per layer (+ row L: the final norm in `in_ln`),
every field 128-lane aligned. Under TP the pack differs per rank: the fields in PER_RANK hold this
rank's slice (q_b, shared and dense gate_up columns) or a scale computed over this rank's local K
(o_proj, shared down, dense down -- main requantizes those per rank, main-weights-runner.md §1).
"""

PER_RANK = ("q_b_s", "o_proj_s", "sw13_s", "sw2_s", "w13_s", "w2_s")


def rup(x, a=128):
    return (x + a - 1) // a * a


def smalls_layout(cfg):
    """name -> (offset, length); returns (dict, total width)."""
    fields = [
        ("in_ln", cfg.H), ("post_ln", cfg.H), ("q_a_ln", cfg.QL), ("kv_a_ln", cfg.KL),
        ("qkv_a_s", rup(cfg.QKVA)), ("q_b_s", cfg.HR * cfg.DQ), ("o_proj_s", cfg.H),
        ("idx_wq_s", cfg.IH * cfg.IDIM), ("idx_kn_w", cfg.IDIM), ("idx_kn_b", cfg.IDIM),
        ("gate_bias", cfg.E), ("sw13_s", 2 * cfg.ISR), ("sw2_s", cfg.H),
        ("w13_s", 2 * cfg.IDR), ("w2_s", cfg.H),
    ]
    if getattr(cfg, "V", 0):
        # MTP drafter: the shared-head / embed / hidden norms, in every pass row (the glue after
        # pass l reads them from pass l's slot)
        fields += [("shead", cfg.H), ("enorm", cfg.H), ("hnorm", cfg.H)]
    if getattr(cfg, "KREP", 0):
        # decode replica table, one entry per slot in greedy order (-1 = empty): rep_e = kernel
        # expert, rep_s = slot = holder * KREP + q (replicas.table). Appended last.
        fields += [("rep_e", 128), ("rep_s", 128)]
    off, lay = 0, {}
    for name, n in fields:
        lay[name] = (off, n)
        off += rup(n)
    return lay, off
