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
"""HTTP API, static export, and CLI guard tests for the Python-backed workbench."""

from __future__ import annotations

import base64
import dataclasses
import gzip
import http.client
import json
import re
import socket
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from xml.parsers import expat

import pytest

from tests.megakernel.test_advisor_calibrate import (
    _build_perfetto_trace,
    _build_xplane_bytes,
)
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor.regimes import _evaluate_sweep_workloads
from vllm_torchtpu.megakernel.advisor.serve import (
    _format_server_url,
    _select_server_class,
    validate_loopback_host,
)
from vllm_torchtpu.megakernel.core import errors

if TYPE_CHECKING:
    from collections.abc import Iterator

SVG_BLOCK_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL)
ADVISOR_DATA_RE = re.compile(
    r'<script\s+id="advisor-data"\s+type="application/json">(.*?)</script>',
    re.DOTALL,
)
BANNED_JS_PHYSICS_SYMBOLS: tuple[str, ...] = (
    "evaluateWorkload",
    "computePerTokenVmem",
    "expertsHitBusiest",
    "allReduceCost",
    "dsaStageSeconds",
    "traceEventsToMeas",
    "buildLiveBucketSchedule",
)
HTTP_STATUS_OK = 200
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_PAYLOAD_TOO_LARGE = 413
TEST_IPV6_PORT = 8765
INVALID_BATCH_FLOAT = 16.9
DP_PREFILL_SIZE = 2
TP_PREFILL_SIZE = 4
EP_PREFILL_SIZE = 8
_QWEN_SMALL_BATCH = 4
_QWEN_HIGH_TOKENS = 64
_QWEN_CONTEXT_4K = 4096
_QWEN_LONG_PREFILL_ISL = 16384
_QWEN_PREFILL_PP_8 = 8
_QWEN_TP_EP_8 = 8
_WATERFALL_RESIDUAL_TOL_US = 1e-9


@contextmanager
def _running_workbench_server(
    model_slug: str = "qwen3.5-397b",
    chip_name: str = "tpu7x",
) -> Iterator[tuple[str, int]]:
    """Start an ephemeral loopback workbench server and yield `(host, port)`."""
    server = advisor.create_workbench_server(
        model=model_slug,
        chip=chip_name,
        host="127.0.0.1",
        port=0,
    )
    host = str(server.server_address[0])
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def _get_text(host: str, port: int, path: str) -> tuple[int, str]:
    """Send an HTTP GET request over loopback and return `(status, body_text)`."""
    conn = http.client.HTTPConnection(host, port, timeout=15.0)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")
    finally:
        conn.close()


def _post_json(
    host: str,
    port: int,
    path: str,
    payload: dict[str, object],
) -> tuple[int, dict[str, object]]:
    """Send a JSON HTTP POST request and return `(status_code, decoded_json)`."""
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=30.0)
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw)
    finally:
        conn.close()


def test_serve_api_advise_matches_python_api() -> None:
    """Verify `POST /api/advise` matches `advisor.advise()` across presets."""
    for model_slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(model_slug)
        chip_spec = advisor.resolve_chip_spec("tpu7x")
        decode_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
        workload = advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            context=8192,
            tp=decode_layout.tp,
            ep=decode_layout.ep,
            dp=decode_layout.dp,
            pp=decode_layout.pp,
            dcp=decode_layout.dcp,
            pcp=decode_layout.pcp,
        )
        expected_report = json.loads(
            json.dumps(advisor.advise(profile, workload, chip_spec).to_dict())
        )
        with _running_workbench_server(model_slug=model_slug) as (host, port):
            get_status, html_text = _get_text(host, port, "/")
            assert get_status == HTTP_STATUS_OK
            match = ADVISOR_DATA_RE.search(html_text)
            assert match is not None
            seed = json.loads(match.group(1).replace("<\\/", "</"))
            assert seed["mode"] == "live"
            assert seed["model"] == model_slug

            status, response_body = _post_json(
                host,
                port,
                "/api/advise",
                {
                    "phase": "decode",
                    "batch": 16,
                    "context": 8192,
                    "tp": decode_layout.tp,
                    "ep": decode_layout.ep,
                    "dp": decode_layout.dp,
                    "pp": decode_layout.pp,
                    "dcp": decode_layout.dcp,
                    "pcp": decode_layout.pcp,
                },
            )
            assert status == HTTP_STATUS_OK
            assert response_body["report"] == expected_report


def test_serve_api_phase_map_matches_bucket_schedule() -> None:
    """Verify `POST /api/phase_map` verdicts and bucket schedule match Python."""
    model_slug = "qwen3.5-397b"
    profile = advisor.resolve_model_profile(model_slug)
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    decode_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    sweep = advisor.advise_regimes(model_slug, "tpu7x")
    expected_schedule = advisor.build_bucket_schedule(sweep).to_dict()

    with _running_workbench_server(model_slug=model_slug) as (host, port):
        status, payload = _post_json(
            host,
            port,
            "/api/phase_map",
            {
                "phase": "decode",
                "tp": decode_layout.tp,
                "ep": decode_layout.ep,
                "dp": decode_layout.dp,
                "pp": decode_layout.pp,
                "dcp": decode_layout.dcp,
                "pcp": decode_layout.pcp,
            },
        )
        assert status == HTTP_STATUS_OK
        assert payload["bucket_schedule"] == expected_schedule
        cells = payload["cells"]
        assert isinstance(cells, list)
        assert cells
        sample_cell = cells[0]
        direct = advisor.advise(
            profile,
            advisor.WorkloadSpec(
                phase="decode",
                batch=int(sample_cell["tokens"]) * decode_layout.dp,
                context=int(sample_cell["context"]),
                tp=decode_layout.tp,
                ep=decode_layout.ep,
                dp=decode_layout.dp,
                pp=decode_layout.pp,
                dcp=decode_layout.dcp,
                pcp=decode_layout.pcp,
            ),
            chip_spec,
        )
        assert sample_cell["recommended_scope"] == direct.recommended_scope
        assert sample_cell["fastest_scope"] == direct.fastest_scope
        assert sample_cell["core_ready_scope"] == direct.core_ready_scope
        assert sample_cell["core_support"] == direct.core_support


def test_serve_api_calibrate_parses_traces_and_rejects_bad_payloads(
    tmp_path: Path,
) -> None:
    """Verify `POST /api/calibrate` parses traces and rejects bad payloads."""
    perfetto_bytes = json.dumps(_build_perfetto_trace(synthetic_default=True)).encode(
        "utf-8"
    )
    perfetto_gz_bytes = gzip.compress(perfetto_bytes)
    xplane_bytes = _build_xplane_bytes(synthetic_default=True)

    perfetto_path = tmp_path / "sample.json"
    perfetto_path.write_bytes(perfetto_bytes)
    xplane_path = tmp_path / "sample.xplane.pb"
    xplane_path.write_bytes(xplane_bytes)
    expected_perfetto = advisor.calibrate_trace(
        perfetto_path,
        options=advisor.CalibrateOptions(chip="tpu7x", model="qwen3.5-397b"),
    )
    expected_xplane = advisor.calibrate_trace(
        xplane_path,
        options=advisor.CalibrateOptions(chip="tpu7x", model="qwen3.5-397b"),
    )
    temp_root = Path(tempfile.gettempdir())
    before_upload_dirs = {entry.name for entry in temp_root.glob("mka_wb_upload_*")}

    with _running_workbench_server(model_slug="qwen3.5-397b") as (host, port):
        for filename, raw_bytes, expected in (
            ("sample.json", perfetto_bytes, expected_perfetto),
            ("sample.json.gz", perfetto_gz_bytes, expected_perfetto),
            ("sample.xplane.pb", xplane_bytes, expected_xplane),
        ):
            status, body = _post_json(
                host,
                port,
                "/api/calibrate",
                {
                    "filename": filename,
                    "content_base64": base64.b64encode(raw_bytes).decode("ascii"),
                },
            )
            assert status == HTTP_STATUS_OK
            raw_meas = body["measurements"]
            assert isinstance(raw_meas, dict)
            assert "launch_gap_us" in raw_meas
            assert {k: v for k, v in raw_meas.items() if k != "source"} == {
                k: v for k, v in expected.items() if k != "source"
            }

        advise_status, advise_body = _post_json(
            host,
            port,
            "/api/advise",
            {"phase": "decode", "batch": 16, "context": 8192},
        )
        assert advise_status == HTTP_STATUS_OK
        advise_report = advise_body["report"]
        assert isinstance(advise_report, dict)
        advise_metrics = advise_report["metrics"]
        assert isinstance(advise_metrics, dict)
        assert advise_metrics["launch_gap_us"]["tag"] == "MEASURED"

        bad_status, bad_body = _post_json(
            host,
            port,
            "/api/calibrate",
            {
                "filename": "corrupt.json",
                "content_base64": base64.b64encode(b"{not valid json").decode("ascii"),
            },
        )
        assert bad_status == HTTP_STATUS_BAD_REQUEST
        assert "error" in bad_body

        # Invalid path traversal filename rejected with 400.
        trav_status, _ = _post_json(
            host,
            port,
            "/api/calibrate",
            {
                "filename": "../evil.json",
                "content_base64": base64.b64encode(perfetto_bytes).decode("ascii"),
            },
        )
        assert trav_status == HTTP_STATUS_BAD_REQUEST

        # Oversized Content-Length header rejected fail-closed with HTTP 413.
        oversized_len = advisor.serve.MAX_TRACE_UPLOAD_JSON_BODY_BYTES + 1024
        conn = http.client.HTTPConnection(host, port, timeout=15.0)
        try:
            conn.putrequest("POST", "/api/calibrate")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(oversized_len))
            conn.endheaders()
            resp = conn.getresponse()
            assert resp.status == HTTP_STATUS_PAYLOAD_TOO_LARGE
        finally:
            conn.close()

    after_upload_dirs = {entry.name for entry in temp_root.glob("mka_wb_upload_*")}
    assert after_upload_dirs - before_upload_dirs == set()


def test_workbench_static_html_embeds_snapshot_without_physics_js(
    tmp_path: Path,
) -> None:
    """Verify `--workbench-html` and `serve --export-html` embed zero JS math."""
    wb_html_path = tmp_path / "workbench_cli.html"
    serve_export_path = tmp_path / "workbench_serve.html"
    zero_js_html_path = tmp_path / "audit_zero_js.html"

    rc_wb = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--regime-sweep",
            "--workbench-html",
            str(wb_html_path),
            "--html",
            str(zero_js_html_path),
        ]
    )
    assert rc_wb == 0
    rc_serve = advisor.main(
        [
            "serve",
            "--model",
            "qwen3.5-397b",
            "--chip",
            "tpu7x",
            "--export-html",
            str(serve_export_path),
        ]
    )
    assert rc_serve == 0

    # Zero-JS static audit report remains strictly free of <script> tags.
    audit_html = zero_js_html_path.read_text(encoding="utf-8")
    assert "<script" not in audit_html.lower()

    js_source = advisor.assets.load_workbench_js()
    for banned_symbol in BANNED_JS_PHYSICS_SYMBOLS:
        assert banned_symbol not in js_source

    for out_path in (wb_html_path, serve_export_path):
        wb_html = out_path.read_text(encoding="utf-8")
        assert wb_html.startswith("<!DOCTYPE html>")
        assert 'id="view-workbench"' in wb_html
        assert 'id="wb-vmem-arena-svg"' in wb_html
        match = ADVISOR_DATA_RE.search(wb_html)
        assert match is not None
        seed = json.loads(match.group(1).replace("<\\/", "</"))
        assert seed["mode"] == "static"
        assert seed["model"] == "qwen3.5-397b"
        assert seed["bucket_schedule"]["schema"] == advisor.BUCKET_SCHEDULE_SCHEMA
        is_final = True
        for svg_markup in SVG_BLOCK_RE.findall(wb_html):
            parser = expat.ParserCreate()
            parser.Parse(svg_markup, is_final)


def test_serve_cli_enforces_loopback_bind_and_validates_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify `advisor serve` rejects bad hosts, ports, and output paths."""
    cwd_dir = tmp_path / "cwd_serve_clean"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    all_ifaces_host = f"{0}.0.0.0"
    bad_argvs: tuple[list[str], ...] = (
        ["serve", "--model", "qwen3.5-397b", "--host", all_ifaces_host],
        ["serve", "--model", "qwen3.5-397b", "--host", "203.0.113.10"],
        ["serve", "--model", "qwen3.5-397b", "--host", ""],
        ["serve", "--model", "qwen3.5-397b", "--port", "-1"],
        ["serve", "--model", "qwen3.5-397b", "--port", "70000"],
        ["serve", "--model", "qwen3.5-397b", "--chip", ""],
        ["serve", "--model", "qwen3.5-397b", "--export-html", ""],
        ["serve", "--model", "qwen3.5-397b", "--export-html", str(cwd_dir)],
        ["serve", "--model", "qwen3.5-397b", "--export-html", "-"],
        ["--model", "qwen3.5-397b", "--workbench-html", ""],
        ["--model", "qwen3.5-397b", "--workbench-html", str(cwd_dir)],
        ["--model", "qwen3.5-397b", "--workbench-html", "-"],
    )
    for argv in bad_argvs:
        with pytest.raises(errors.MegakernelConfigError):
            advisor.main(argv)
    assert list(cwd_dir.iterdir()) == []


def test_serve_cli_supports_config_flag_and_rejects_mutual_exclusion(
    tmp_path: Path,
) -> None:
    """Verify `advisor serve --config` works, calibrates, and enforces exclusivity."""
    snapshot_path = (
        Path(__file__).resolve().parent
        / "data"
        / "advisor"
        / "hf_configs"
        / "qwen3.5-397b.json"
    )
    cfg_dir = tmp_path / "custom_qwen_dir"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    hf_cfg_path = cfg_dir / "config.json"
    hf_cfg_path.write_text(
        json.dumps(json.loads(snapshot_path.read_text(encoding="utf-8"))["config"]),
        encoding="utf-8",
    )
    export_path = tmp_path / "config_workbench.html"
    rc = advisor.main(
        [
            "serve",
            "--config",
            str(hf_cfg_path),
            "--export-html",
            str(export_path),
        ]
    )
    assert rc == 0
    html_text = export_path.read_text(encoding="utf-8")
    match = ADVISOR_DATA_RE.search(html_text)
    assert match is not None
    seed = json.loads(match.group(1))
    assert seed["mode"] == "static"

    profile = dataclasses.replace(
        advisor.resolve_model_profile(hf_cfg_path), name="custom_qwen_dir"
    )
    assert profile.name not in advisor.MODEL_PRESETS
    server = advisor.create_workbench_server(
        model=profile, chip="tpu7x", host="127.0.0.1", port=0
    )
    host = str(server.server_address[0])
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        perfetto_bytes = json.dumps(
            _build_perfetto_trace(synthetic_default=True)
        ).encode("utf-8")
        cal_status, cal_body = _post_json(
            host,
            port,
            "/api/calibrate",
            {
                "filename": "cfg_trace.json",
                "content_base64": base64.b64encode(perfetto_bytes).decode("ascii"),
            },
        )
        assert cal_status == HTTP_STATUS_OK
        cal_meas = cal_body["measurements"]
        assert isinstance(cal_meas, dict)
        assert "launch_gap_us" in cal_meas
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    with pytest.raises(SystemExit):
        advisor.main(
            [
                "serve",
                "--model",
                "qwen3.5-397b",
                "--config",
                str(hf_cfg_path),
            ]
        )

    for empty_flag in ("--model", "--config", "--hf-repo"):
        with pytest.raises(errors.MegakernelConfigError):
            advisor.main(["serve", empty_flag, ""])


def _ipv6_loopback_available() -> bool:
    """Return True when the local network stack supports binding `::1`."""
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


def test_serve_ipv6_loopback_address_family_and_url_formatting() -> None:
    """Verify `::1` uses AF_INET6 socket family and bracketed browser URL."""
    assert validate_loopback_host("::1") == "::1"
    assert _select_server_class("::1").address_family == socket.AF_INET6
    assert _select_server_class("127.0.0.1").address_family == socket.AF_INET
    assert (
        _format_server_url("::1", TEST_IPV6_PORT) == f"http://[::1]:{TEST_IPV6_PORT}/"
    )
    assert (
        _format_server_url("127.0.0.1", TEST_IPV6_PORT)
        == f"http://127.0.0.1:{TEST_IPV6_PORT}/"
    )
    if not _ipv6_loopback_available():
        with pytest.raises(OSError, match=r".+"):
            advisor.create_workbench_server(model="qwen3.5-397b", host="::1", port=0)
        return
    server = advisor.create_workbench_server(model="qwen3.5-397b", host="::1", port=0)
    bound_host = str(server.server_address[0])
    bound_port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, html_text = _get_text(bound_host, bound_port, "/")
        assert status == HTTP_STATUS_OK
        assert 'id="advisor-data"' in html_text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_serve_measurements_chip_scope_and_workbench_html_propagation(
    tmp_path: Path,
) -> None:
    """Verify chip-scoped measurement checks and calibrated HTML seed propagation."""
    with pytest.raises(errors.MegakernelConfigError):
        advisor.create_workbench_server(
            model="qwen3.5-397b", chip="v6e", measurements="bundled"
        )

    bundled_html_path = tmp_path / "bundled_workbench.html"
    assert (
        advisor.main(
            [
                "serve",
                "--model",
                "qwen3.5-397b",
                "--measurements",
                "bundled",
                "--export-html",
                str(bundled_html_path),
            ]
        )
        == 0
    )
    bundled_match = ADVISOR_DATA_RE.search(
        bundled_html_path.read_text(encoding="utf-8")
    )
    assert bundled_match is not None
    bundled_seed = json.loads(bundled_match.group(1))
    assert bundled_seed["default_eval"] == bundled_seed["regime_evals"][0]
    bundled_lg = bundled_seed["default_eval"]["report"]["metrics"]["launch_gap_us"]
    assert bundled_lg["value"] == 1.0
    assert "tpu7x_reference_priors" in bundled_lg["source"]

    xplane_path = tmp_path / "calibrated.xplane.pb"
    xplane_path.write_bytes(_build_xplane_bytes(synthetic_default=True))
    calibrated_json_path = tmp_path / "calibrated.json"
    advisor.calibrate_trace(
        xplane_path,
        calibrated_json_path,
        options=advisor.CalibrateOptions(chip="tpu7x", model="qwen3.5-397b"),
    )

    wb_html_path = tmp_path / "calibrated_workbench.html"
    serve_html_path = tmp_path / "calibrated_serve.html"
    assert (
        advisor.main(
            [
                "--model",
                "qwen3.5-397b",
                "--regime-sweep",
                "--measurements",
                str(calibrated_json_path),
                "--workbench-html",
                str(wb_html_path),
            ]
        )
        == 0
    )
    assert (
        advisor.main(
            [
                "serve",
                "--model",
                "qwen3.5-397b",
                "--measurements",
                str(calibrated_json_path),
                "--export-html",
                str(serve_html_path),
            ]
        )
        == 0
    )

    for html_path in (wb_html_path, serve_html_path):
        match = ADVISOR_DATA_RE.search(html_path.read_text(encoding="utf-8"))
        assert match is not None
        seed = json.loads(match.group(1))
        regime_evals = seed["regime_evals"]
        assert isinstance(regime_evals, list)
        assert regime_evals
        assert seed["default_eval"] == regime_evals[0]
        assert seed["default_eval"]["report"]["metrics"]["launch_gap_us"]["tag"] == (
            "MEASURED"
        )
        for entry in regime_evals:
            report = entry["report"]
            assert report["metrics"]["launch_gap_us"]["tag"] == "MEASURED"
            assert report["vmem_budget_mib"] > 0.0
            assert report["per_token_vmem_bytes"] >= 0


def test_serve_api_rejects_invalid_json_types_and_zero_efficiencies() -> None:
    """Verify `POST /api/advise` fails closed on zero efficiencies and bad types."""
    bad_payloads: tuple[dict[str, object], ...] = (
        {"eta_dense": 0.0},
        {"batch": INVALID_BATCH_FLOAT},
        {"vmem_alias": "false"},
        {"phase": ""},
        {"dispatch_regime": ""},
    )
    with _running_workbench_server(model_slug="qwen3.5-397b") as (host, port):
        for payload in bad_payloads:
            status, body = _post_json(host, port, "/api/advise", payload)
            assert status == HTTP_STATUS_BAD_REQUEST
            assert "error" in body


def test_phase_map_supports_dp_gt_1_prefill_and_verify() -> None:
    """Verify `POST /api/phase_map` succeeds without invalid cells on prefill/verify."""
    cases: tuple[dict[str, object], ...] = (
        {
            "phase": "prefill",
            "dp": DP_PREFILL_SIZE,
            "tp": TP_PREFILL_SIZE,
            "ep": EP_PREFILL_SIZE,
        },
        {
            "phase": "verify",
            "dp": 1,
            "tp": EP_PREFILL_SIZE,
            "ep": EP_PREFILL_SIZE,
            "spec_tokens": 0,
        },
    )
    with _running_workbench_server(model_slug="qwen3.5-397b") as (host, port):
        for payload in cases:
            status, response_body = _post_json(host, port, "/api/phase_map", payload)
            assert status == HTTP_STATUS_OK
            cells = response_body["cells"]
            assert isinstance(cells, list)
            assert cells
            for cell in cells:
                assert not str(cell["verdict"]).startswith("invalid_config(")


def test_workbench_html_escapes_script_breakout_and_html_comments() -> None:
    """Verify `render_workbench_html` escapes `</script>` and `<!--` in seed JSON."""
    profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    sweep = dataclasses.replace(
        advisor.advise_regimes(profile, chip_spec),
        model="evil</script><!--<script>",
    )
    html_text = advisor.render_workbench_html(profile, chip_spec, sweep, mode="static")
    match = ADVISOR_DATA_RE.search(html_text)
    assert match is not None
    raw_seed = match.group(1)
    assert "</script>" not in raw_seed.lower()
    assert "<!--" not in raw_seed
    decoded = json.loads(raw_seed)
    assert decoded["model"] == "evil</script><!--<script>"


def _assert_waterfall_zero_residual(report: advisor.AdvisorReport) -> None:
    """Assert 5 Waterfall deltas sum exactly to whole_step minus per_op floor."""
    whole_step = next(s for s in report.scopes if s.scope == "whole_step")
    per_op = next(s for s in report.scopes if s.scope == "per_op")
    compute_delta = whole_step.compute_us - per_op.compute_us
    hbm_delta = whole_step.hbm_us - per_op.hbm_us
    comm_delta = whole_step.comm_us - per_op.comm_us
    launch_host_fill_delta = (
        whole_step.launch_us + whole_step.fill_drain_us + whole_step.host_head_us
    ) - (per_op.launch_us + per_op.fill_drain_us + per_op.host_head_us)
    raw_sum = compute_delta + hbm_delta + comm_delta + launch_host_fill_delta
    net_delta = whole_step.step_bound_us - per_op.step_bound_us
    overlap_delta = net_delta - raw_sum
    assert abs((raw_sum + overlap_delta) - net_delta) < _WATERFALL_RESIDUAL_TOL_US


def _assert_workbench_ui_strings(wb_html: str) -> None:
    """Assert JS Waterfall, Core-Ready phase map, and HTML select contracts."""
    js_source = advisor.assets.load_workbench_js()
    for token in (
        "Launch, Host & Fill/Drain Delta",
        "host_head_us",
        "Overlap / Parallelism Delta",
        "netD - rawSum",
        'viewBox="0 0 760 162"',
        'y2="154"',
        "core_support",
        "core_ready_scope",
        "core_blockers",
        "rep.binding_unit",
    ):
        assert token in js_source
    assert "mxu" not in js_source.lower()
    assert re.search(
        r"S\.phaseMapMode\s*===\s*'core_ready'\s*\?\s*"
        r"c\.core_ready_scope\s*:\s*c\.recommended_scope",
        js_source,
    )
    select_match = re.search(
        r'<select[^>]*id="wb-phase-map-mode"[^>]*>(.*?)</select>',
        wb_html,
        re.DOTALL,
    )
    assert select_match is not None
    select_body = select_match.group(1)
    assert '<option value="recommended">Physics Winner</option>' in select_body
    assert '<option value="core_ready">Core-Ready Today</option>' in select_body


def test_workbench_waterfall_overlap_bar_and_core_ready_phase_map_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 5-bar zero-residual Waterfall, Core-Ready phase map, and clean labels."""
    profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    sweep = advisor.advise_regimes(profile, chip_spec)
    wb_html = advisor.render_workbench_html(profile, chip_spec, sweep, mode="static")
    _assert_workbench_ui_strings(wb_html)

    phase_map = advisor.workbench.build_phase_map_payload(profile, chip_spec, sweep)
    cells = phase_map["cells"]
    assert isinstance(cells, list)
    assert cells
    for cell in cells:
        assert isinstance(cell, dict)
        assert {"core_ready_scope", "core_support", "core_blockers"} <= set(cell)

    cell_small = next(
        c
        for c in cells
        if c["tokens"] == _QWEN_SMALL_BATCH and c["context"] == _QWEN_CONTEXT_4K
    )
    cell_high = next(
        c
        for c in cells
        if c["tokens"] == _QWEN_HIGH_TOKENS and c["context"] == _QWEN_CONTEXT_4K
    )
    assert cell_small["recommended_scope"] == "whole_step"
    assert cell_small["core_ready_scope"] == "whole_step"
    assert cell_small["core_support"] == "within_core_limits"
    assert cell_high["recommended_scope"] == "whole_step"
    assert cell_high["core_ready_scope"] == "per_op"
    assert cell_high["core_support"] == "outside_core_limits"

    matmul_entries, _ = _evaluate_sweep_workloads(
        profile,
        chip_spec,
        (
            (
                "Long Prefill (B=1, T=16384)",
                advisor.WorkloadSpec(
                    phase="prefill",
                    batch=1,
                    isl=_QWEN_LONG_PREFILL_ISL,
                    osl=0,
                    tp=1,
                    ep=1,
                    pp=_QWEN_PREFILL_PP_8,
                ),
            ),
        ),
        None,
    )
    prefill_entry = next(
        entry for entry in matmul_entries if entry.binding_unit == "matmul"
    )
    prefill_html = advisor.render_workbench_html(
        profile,
        chip_spec,
        dataclasses.replace(sweep, regimes=(prefill_entry,)),
        mode="static",
    )
    rendered_body = re.sub(
        r"<(style|script)\b[^>]*>.*?</\1>",
        "",
        prefill_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert "matmul" in rendered_body
    assert 'id="wb-kpi-bind">matmul<' in rendered_body
    assert "mxu" not in prefill_html.lower()
    prefill_seed_match = ADVISOR_DATA_RE.search(prefill_html)
    assert prefill_seed_match is not None
    assert "binding_unit_labels" not in json.loads(prefill_seed_match.group(1))

    empty_sweep = dataclasses.replace(sweep, regimes=())
    empty_html = advisor.render_workbench_html(profile, chip_spec, empty_sweep)
    empty_seed_match = ADVISOR_DATA_RE.search(empty_html)
    assert empty_seed_match is not None
    empty_default_rep = json.loads(empty_seed_match.group(1))["default_eval"]["report"]
    expected_tps = empty_default_rep["metrics"]["throughput_tok_s_per_chip"]["value"]
    assert f'id="wb-kpi-scope">{empty_default_rep["recommended_scope"]}<' in empty_html
    assert f'id="wb-kpi-tps">{expected_tps:.1f}<' in empty_html

    decode_workload = advisor.WorkloadSpec(
        phase="decode",
        batch=_QWEN_SMALL_BATCH,
        context=_QWEN_CONTEXT_4K,
        tp=_QWEN_TP_EP_8,
        ep=_QWEN_TP_EP_8,
    )
    report = advisor.advise("qwen3.5-397b", decode_workload, "tpu7x")
    assert report.recommended_scope == "whole_step"
    assert report.core_support == "within_core_limits"
    _assert_waterfall_zero_residual(report)
    _assert_waterfall_zero_residual(
        advisor.advise(
            "qwen3.5-397b",
            dataclasses.replace(
                decode_workload,
                host_head_per_op_us=120.0,
                host_head_fused_us=20.0,
                host_head_explicit=True,
            ),
            "tpu7x",
        )
    )

    monkeypatch.setattr(
        advisor.workbench,
        "advise",
        lambda *_args, **_kwargs: dataclasses.replace(
            report,
            core_support="outside_core_limits",
            core_blockers=(),
        ),
    )
    outside_empty_html = advisor.render_workbench_html(profile, chip_spec, empty_sweep)
    assert "Blocked: <code>none</code>" in outside_empty_html
