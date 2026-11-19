# SPDX-License-Identifier: Apache-2.0
"""q38 L1-SERVE: the served D14 option string reaches the imported (MM 37b960ea9) Qwen3.8 driver intact.

Transport (C): the served disagg recipe turns DECODE_EXTRA_SERVER_ENV into the decode container env with
    parse(s) = [std.splitLimit(kv, '=', 1) for kv in std.filter(function(x) x != '', std.split(s, ','))]
    { [p[0]]: p[1] for p in pairs }
(infx-qwen38-flash-agentx-disagg.jsonnet a420f95d, extraServerEnv, :585-594), and the STACK-D composer joins the decode env
items with ','. So the D14 option list must be joined with '+', and the driver must turn '+' back into ','
(INT-4 cb677f4ad's marked block in Qwen38SegmentDriver.maybe_create).

These tests run the recipe's split (re-implemented line for line below) and then the driver's OWN lines (from
`raw = str(envs.VLLM_TPU_MEGAKERNEL ...` through '# q38 INT-4 end', taken from the source) and its own parsers. They pass
only if the served string yields exactly one VLLM_TPU_MEGAKERNEL env item, no stray env names, and the D14 options
(whole / wholeshard / wholeunion / wholeloop on, wholerows 64, wholecap 262144, buckets 8:16:32:48:64). With SERVED in
the comma form, or with the driver's replace line deleted, test_served_d14_string_transport fails (the mutation checks of
the L1-SERVE proof job). CPU only, no TPU."""
import inspect
import textwrap
import types

from vllm_torchtpu.megakernel_runtime import qwen38_driver as d

MK = "VLLM_TPU_MEGAKERNEL"
# The served value of record (D14 STEP T98f arm d14, joined with '+' for the recipe; prereg a61ce361 + A1 d5c71ec6).
SERVED = "whole=1+wholeshard=1+wholeunion=1+wholeloop=1+wholerows=64+wholecap=262144+buckets=8:16:32:48:64"
COMMA = SERVED.replace("+", ",")
WANT = {"whole": True, "wholeshard": True, "wholeunion": True, "wholeloop": True, "wholerows": 64,
        "wholecap": 262144, "buckets": (8, 16, 32, 48, 64)}
# Other decode env items around it, as the composer writes them (names only matter here).
OTHERS = ["VLLM_TPU_QSA_MAIN_KV_DCP=4", "VLLM_TPU_DECODE_TP_AR=h8", "VLLM_TPU_VPS_DRAWS=full"]


def recipe_env(decode_extra_server_env: str) -> dict:
    """The recipe's extraServerEnv for the decode leg (EXTRA_SERVER_ENV empty), with its NAME=VALUE assert."""
    pairs = [kv.split("=", 1) for kv in decode_extra_server_env.split(",") if kv != ""]
    assert all(len(p) == 2 and p[0] != "" for p in pairs), "EXTRA_SERVER_ENV entries must look like NAME=VALUE"
    return {p[0]: p[1] for p in pairs}


def driver_raw(value):
    src = textwrap.dedent(inspect.getsource(d.Qwen38SegmentDriver.maybe_create.__func__)).split("\n")
    start = next(i for i, ln in enumerate(src) if "raw = str(envs.VLLM_TPU_MEGAKERNEL" in ln)
    end = next(i for i, ln in enumerate(src) if ln.strip() == "# q38 INT-4 end")
    ns = {"envs": types.SimpleNamespace(VLLM_TPU_MEGAKERNEL=value)}
    exec(textwrap.dedent("\n".join(src[start : end + 1])), ns)  # noqa: S102 - the driver's own lines
    return ns["raw"]


def driver_opts(raw):
    return {
        "whole": d._parse_flag(raw, "whole", False),
        "wholeshard": d._parse_flag(raw, "wholeshard", False),
        "wholeunion": d._parse_flag(raw, "wholeunion", False),
        "wholeloop": d._parse_flag(raw, "wholeloop", False),
        "wholerows": d._parse_int(raw, "wholerows"),
        "wholecap": d._parse_int(raw, "wholecap"),
        "buckets": d._parse_buckets(raw),
    }


def transport(value):
    """Composer join -> recipe split -> engine env -> driver normalization -> parsed options."""
    env = recipe_env(",".join(OTHERS + [MK + "=" + value]))
    stray = sorted(k for k in env if k.startswith(("whole", "buckets")) or k in WANT)
    return env, stray, driver_opts(driver_raw(env.get(MK, "")))


def test_served_d14_string_transport():
    env, stray, opts = transport(SERVED)
    assert env[MK] == SERVED
    assert stray == []
    assert sum(1 for k in env if k == MK) == 1
    assert opts == WANT


def test_comma_form_does_not_survive_the_recipe():
    # Documents the failure the '+' form avoids: the comma form leaves VLLM_TPU_MEGAKERNEL=whole=1 plus stray env vars.
    env, stray, opts = transport(COMMA)
    assert env[MK] == "whole=1"
    assert "wholeunion" in stray and "buckets" in stray
    assert opts != WANT and opts["wholeunion"] is False


def test_driver_alone_accepts_both_joins():
    assert driver_opts(driver_raw(SERVED)) == WANT
    assert driver_opts(driver_raw(COMMA)) == WANT
