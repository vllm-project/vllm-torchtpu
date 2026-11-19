# SPDX-License-Identifier: Apache-2.0
"""q38 L1-D14: the whole step's ring depths under main-KV DCP 4. Only the 64-row bucket changes (expert ring 6 -> 4:
l1-d14-aot-i1 j-7d24d656 measured the default b64 rings at 66.09M of 63.94M scoped VMEM, 5 slots also over, 4 fits);
every other bucket at DCP 4 equals DCP 2's, and DCP 1 / 2 are untouched. CPU only (no compile)."""

ROWS = (4, 8, 16, 24, 32, 48, 64)


def _rings(monkeypatch, dcp):
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    monkeypatch.delenv("VLLM_TPU_MK_RINGS", raising=False)
    monkeypatch.setattr(wsc, "main_kv_dcp", lambda: dcp)
    return {B: (wsc.wholestep_tiles(B, 65536).nbe, wsc.wholestep_tiles(B, 65536).nbd) for B in ROWS}


def test_group_dcp_rings_trim_only_the_64_row_bucket(monkeypatch):
    one, two, four = (_rings(monkeypatch, d) for d in (1, 2, 4))
    assert four[64] == (4, 4), four
    assert two[64] == one[64] == (6, 4), (one, two)
    for B in ROWS[:-1]:
        assert four[B] == two[B], (B, four, two)


def test_group_dcp_rings_helper(monkeypatch):
    import dataclasses

    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    t = wsc._wholestep_tiles(64, 65536)
    for dcp, B, want in ((4, 64, 4), (4, 48, t.nbe), (4, 96, t.nbe), (2, 64, t.nbe), (1, 64, t.nbe)):
        monkeypatch.setattr(wsc, "main_kv_dcp", lambda d=dcp: d)
        assert wsc._group_dcp_rings(t, B).nbe == want, (dcp, B)
    monkeypatch.setattr(wsc, "main_kv_dcp", lambda: 4)
    small = dataclasses.replace(t, nbe=3)
    assert wsc._group_dcp_rings(small, 64).nbe == 3  # never raises a smaller ring
