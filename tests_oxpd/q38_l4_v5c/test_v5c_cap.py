# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v5 composition (q38-l4-v5c with DRAFT-CAP): DRAFT-CAP inside DH-D (VLLM_TPU_DRAFT_ASM_CAP) with VFY-ONE
value 2's seed taken from the verify program, on the AGREE engine-step clock. CDK CPU pod only. V5COMP-BUILD.

Through DRAFT-CAP's own harness (tests/q38_draftcap/draftcap_harness.py over DRAFT1-B's and DRAFT-ASM's: the real
Eagle3Proposer.propose, every compiled program a counted region, the R_d program as the "one" region, per-group
tables). The verify's stash is built as in tests/q38_l4_v5c/test_v5c_drafter.py.

K1  8 serving steps at 8 / 32 / 48 / 64 rows, fallback rows off and on: DH-D folds every step through ONE R_d program
    (DRAFT-CAP engaged) with the verify's superset seed taken (0 seed programs), bitwise the flag-off base path
    (drafts, source, rejected count, index dicts, step 1's positions / seq_lens / first token).
Run: python -m pytest -q -p no:cacheprovider -p no:logging -s tests/q38_l4_v5c/test_v5c_cap.py
"""

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q38_draftcap"))
import draftcap_harness as hc  # noqa: E402
from vllm_torchtpu.runner import vfy_one  # noqa: E402
from vllm_torchtpu.spec_decode import draft_one  # noqa: E402

SEEDS = ("seed", "seed_asm", "seed_cl")
COUNTS: dict = {}


def _inject(p, step):
    runner = p.runner
    if runner.__dict__.get("_vfy") is None:
        runner._vfy = vfy_one.VfyOne(2, 0, "")
    spec = p.vfy_seed_spec(step.chunk, step.R)
    assert spec is not None and spec.fn is draft_one._seed_cl_impl, spec
    outs = tuple(spec.fn(step.nt, *spec.args))
    runner._vfy_seed = (spec, step.nt, outs, vfy_one.versions((step.nt, *spec.args, *spec.base_args)))


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("R", [8, 32, 48, 64])
def test_k1_cap_dhd_with_the_verify_seed_is_bitwise_base_over_8_steps(monkeypatch, R, fallback):
    seeds = [61000 + 100 * R + 10 * fallback + i for i in range(8)]
    base, _ = hc.serve(monkeypatch, "base", seeds, R, step_kw={"fallback": fallback})
    p = hc.variant(monkeypatch, "cap")
    assert p._draft_asm.one_cap
    hc.run_step(p, hc.make_step(seeds[0] - 1, R, fallback=fallback), serving=False)
    for i, (sd, b) in enumerate(zip(seeds, base)):
        s = hc.make_step(sd, R, fallback=fallback)
        _inject(p, s)
        o = hc.run_step(p, s, serving=True)
        assert o.folded and o.calls["one"] == 1, (i, o.calls, dict(p._draft_asm.reasons))
        assert sum(o.calls[k] for k in SEEDS) == 0 and o.calls["draft_fwd0"] == 0, (i, o.calls)
        hc.same_out(o, b, f"DRAFT-CAP+VFY vs base R={R} step {i}")
    assert p._draft_asm.counts()["folded_one"] == 8 and p.runner._vfy.stats["seed_taken"] == 8
    COUNTS["k1_cases"] = COUNTS.get("k1_cases", 0) + 1
    print(f"\nV5C_CAP K1 R={R} fallback={int(fallback)} steps=8 R_d program 8/8 with the verify seed taken 8/8 "
          f"(seed programs 0); bitwise base")


def test_zz_counts():
    print("\nV5C_CAP_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
