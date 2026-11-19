# SPDX-License-Identifier: Apache-2.0
"""q38 L4 SEEDFIX cross-tree identity (CDK CPU pod only).

Usage: cd <tree>; PYTHONPATH=src python3 <head>/tests/q38_l4_seedfix/q38_seedfix_xtree.py <tag>
       python3 q38_seedfix_xtree.py --compare <base.log> <head_on.log> <head_off.log>

Drives the SAME steps (sf_fakes: the tree's real sample_tokens / verify methods on a fake runner) on whichever tree is
on PYTHONPATH and prints one sha per (scenario, part):
  all_tok / unseeded_tok / seeded_tok   the step tokens (all rows, unseeded rows, seeded rows), 4 steps;
  unseeded_u / all_u                    every uniform array the verify / sampler received (unseeded slices / whole);
  gen                                   the shared sampling generator's final state.
--compare applies the rules (fixed before the run):
  head flag off == base on every part of every scenario (base bitwise, seeded rows included);
  head flag on  == base on unseeded_tok, unseeded_u and gen everywhere, and on every part of all-unseeded scenarios;
  head flag on  != base on seeded_tok of every mixed scenario (the fix engages).
Last line: XTREE_SUMMARY PASS|FAIL ...
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _sha(ts):
    h = hashlib.sha1()
    for t in ts:
        t = t.contiguous()
        h.update(str((tuple(t.shape), str(t.dtype))).encode())
        h.update(t.view(-1).view(__import__("torch").uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()[:12]


def run(tag):
    sys.path.insert(0, HERE)
    import torch

    import sf_fakes as F
    import vllm_torchtpu.runner.tpu_runner as tr
    from vllm_torchtpu.layers.adapter.sample import vp_sampler as vp

    try:
        from vllm_torchtpu.runner import seedfix as sfm
    except ImportError:
        sfm = None
    for k in list(os.environ):
        if k.startswith(("VLLM_TPU_L6_", "VLLM_TPU_L4_", "VLLM_TPU_VPS_")):
            os.environ.pop(k)
    mp = F.Patcher()
    F.patch_all(mp)
    root = os.path.realpath("src")
    print(f"XTREE_ORIGIN {tag} tpu_runner={os.path.realpath(tr.__file__)} inside={os.path.realpath(tr.__file__).startswith(root + os.sep)} "
          f"seedfix={'yes' if sfm else 'no'} flag={os.environ.get('VLLM_TPU_SEEDED_SAMPLING', '<unset>')}")

    def mixed(tag2, n, seeded, range_rows=(), d=None):
        return [F.Req(f"{tag2}{i}", 1000 + i if i in seeded else None, top_k=0 if i in range_rows else 20,
                      temp=[0.8, 0.6, 1.0][i % 3], top_p=[0.95, 1.0, 0.9][i % 3], logit_seed=i,
                      d=F.K if d is None else d[i]) for i in range(n)]

    def runner(route):
        return {"vps_fb": lambda: F.Runner(vps=True, fb_max=8), "vps_nofb": lambda: F.Runner(vps=True, fb_max=8),
                "vps_cand": lambda: F.Runner(vps=True, cand=True, fb_max=8),
                "vps_plainprog": lambda: F.Runner(vps=True, fb_max=0)}.get(route, lambda: F.Runner(vps=False))()

    D = [3, 2, 3, 1, 3, 3, 0]
    n_lines = 0
    for route in ("vps_fb", "vps_nofb", "vps_cand", "vps_plainprog", "stock", "fused", "plain", "spec_plain"):
        for kind in ("all_unseeded", "mixed"):
            seeded = () if kind == "all_unseeded" else (0, 3, 5)
            rng = (1,) if route in ("vps_fb", "vps_cand") else ()
            F.clear_rec()
            r = runner(route)
            chunk = mixed("x", 7, seeded, rng, D if route not in ("plain",) else None)
            toks = []
            for s in range(4):
                if route == "fused":
                    sched = r.set_batch(chunk)
                    logits, md = F.compose_spec(chunk, s)
                    r._seed_plan = (sfm.build_plan(r, sched) if sfm is not None and r.input_batch.generators else None)
                    out = r._sample_spec_verify_fused(logits, md, 0, len(chunk), r._get_sampling_generator())[:7]
                    r._seed_plan = None
                    toks.append(out)
                elif route == "spec_plain":
                    verify = mixed("v", 3, (), (), [3, 3, 3])
                    o, rts = F.run_step(r, [verify, chunk], step=s, plain_chunks=(1,))
                    assert rts == ["stock", "spec_plain"], rts
                    toks.append(o[1].reshape(7, -1))
                else:
                    o, rts = F.run_step(r, [chunk], route={"stock": "stock", "plain": "plain"}.get(route, "vps"),
                                        spec=route != "plain", step=s)
                    toks.append(o[0].reshape(7, -1))
            sb = {b for b, q in enumerate(chunk) if q.seed is not None}
            st, t = set(), 0
            for b, q in enumerate(chunk):
                if q.seed is not None and route not in ("plain", "spec_plain"):
                    st.update(range(t, t + q.d))
                t += q.d
            keep = [i for i in range(7) if i not in sb]
            seedr = sorted(sb)
            parts = {
                "all_tok": _sha(toks),
                "unseeded_tok": _sha([x[keep] for x in toks]),
                "seeded_tok": _sha([x[seedr] for x in toks]) if seedr else "-",
                "gen": _sha([r._sampling_generator.get_state()]),
            }
            allu, unu = [], []
            fbx = set()
            if route == "vps_cand" and seeded and sfm is not None:
                md0 = F.compose_spec(chunk, 0)[1]
                sbs, sts = sfm.fallback_slots(r, 0, 7, md0, torch.bfloat16, vp.VP_KC)
                fbx = {sbs[b] for b in sb if b in sbs} | {sts[x] for x in st if x in sts}
            for key_list, keys in (("vps", ("bonus_u", "accept_u", "recover_u", "fb_u")),
                                   ("fused", ("bonus_u", "accept_u", "recover_u")), ("rs", ("accept_u", "recover_u")),
                                   ("sfl", ("u",))):
                for rec in F.REC[key_list]:
                    for key in keys:
                        a = rec.get(key)
                        if a is None:
                            continue
                        allu.append(a)
                        excl = sb if key in ("bonus_u", "u") else (fbx if key == "fb_u" else st)
                        if route in ("plain", "spec_plain"):
                            # spec_plain: the verify chunk is all unseeded; its bonus u loses rows sb too (harmless)
                            excl = sb if key == "u" else set()
                        unu.append(a[[i for i in range(a.shape[0]) if i not in excl]])
            parts["all_u"] = _sha(allu)
            parts["unseeded_u"] = _sha(unu)
            parts["n_u"] = str(len(allu))
            for part, v in parts.items():
                print(f"XTREE {tag} {route}:{kind} {part} {v}")
                n_lines += 1
    mp.undo()
    print(f"XTREE_DONE {tag} lines={n_lines}")


def compare(base_log, on_log, off_log):
    def load(p):
        d = {}
        for ln in open(p, errors="replace"):
            if ln.startswith("XTREE ") and len(ln.split()) == 5:
                _, _, sc, part, v = ln.split()
                d[(sc, part)] = v
        return d

    b, on, off = load(base_log), load(on_log), load(off_log)
    fails = []
    if not b or set(b) != set(on) or set(b) != set(off):
        fails.append(f"key sets differ: base {len(b)} on {len(on)} off {len(off)}")
    n_ok = 0
    for (sc, part), v in sorted(b.items()):
        kind = sc.split(":")[1]
        if off.get((sc, part)) != v:
            fails.append(f"OFF!=BASE {sc} {part} {off.get((sc, part))} {v}")
        else:
            n_ok += 1
        if part in ("unseeded_tok", "unseeded_u", "gen", "n_u") or kind == "all_unseeded":
            if on.get((sc, part)) != v:
                fails.append(f"ON!=BASE {sc} {part} {on.get((sc, part))} {v}")
            else:
                n_ok += 1
        if kind == "mixed" and part == "seeded_tok":
            if on.get((sc, part)) == v:
                fails.append(f"ON==BASE seeded {sc} (no engagement)")
            else:
                n_ok += 1
    for f in fails:
        print(f"XTREE_CMP FAIL {f}")
    n_sc = len({k[0] for k in b})
    print(f"XTREE_SUMMARY {'PASS' if not fails else 'FAIL'} scenarios={n_sc} checks_ok={n_ok} fails={len(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(*sys.argv[2:5]))
    run(sys.argv[1])
