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
"""Loopback-only stdlib HTTP server backing the interactive Fusion Workbench."""

from __future__ import annotations

import base64
import binascii
import json
import socket
import tempfile
import threading
import webbrowser
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_args

from vllm_torchtpu.megakernel.core import errors

from .calibrate import CalibrateOptions, calibrate_trace
from .chips import ChipSpec, resolve_chip_spec
from .constants import DEFAULT_CHIP, MEASUREMENTS_SCHEMA
from .models import (
    MODEL_PRESETS,
    VALID_EXECUTION_MODES,
    ModelProfile,
    resolve_model_profile,
)
from .regimes import RegimeSweepOptions, advise_regimes
from .specs import (
    DispatchRegime,
    EfficiencySpec,
    ExecutionMode,
    MeasurementsData,
    Phase,
    WorkloadSpec,
    as_mapping,
    as_scalar_number,
    load_measurements,
    parse_host_table_hbm_overrides,
    scalar_bool,
    scalar_int,
)
from .workbench import (
    PhaseMapOptions,
    build_phase_map_payload,
    evaluate_workbench_payload,
    render_workbench_html,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .specs import AdvisorSweepReport

__all__ = [
    "DEFAULT_SERVE_HOST",
    "DEFAULT_SERVE_PORT",
    "LOOPBACK_HOSTS",
    "MAX_JSON_BODY_BYTES",
    "MAX_TRACE_UPLOAD_BYTES",
    "MAX_TRACE_UPLOAD_JSON_BODY_BYTES",
    "WorkbenchServerOptions",
    "create_workbench_server",
    "run_workbench_server",
    "validate_loopback_host",
]

DEFAULT_SERVE_HOST = "127.0.0.1"
DEFAULT_SERVE_PORT = 8765
MAX_TCP_PORT = 65535
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_JSON_BODY_BYTES = 1 * 1024 * 1024
MAX_TRACE_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_TRACE_UPLOAD_JSON_BODY_BYTES = (MAX_TRACE_UPLOAD_BYTES * 4) // 3 + 65536
ALLOWED_UPLOAD_SUFFIXES: tuple[str, ...] = (".json", ".json.gz", ".xplane.pb")
VALID_PHASES: frozenset[str] = frozenset(get_args(Phase))
VALID_DISPATCH_REGIMES: frozenset[str] = frozenset(get_args(DispatchRegime))


@dataclass(frozen=True)
class WorkbenchServerOptions:
    """Configuration options for running the loopback workbench HTTP server."""

    host: str = DEFAULT_SERVE_HOST
    port: int = DEFAULT_SERVE_PORT
    measurements: str | Path | MeasurementsData | None = None
    open_browser: bool = False


def validate_loopback_host(host: str) -> str:
    """Validate that `host` is a strict loopback address."""
    cleaned = host.strip()
    if cleaned not in LOOPBACK_HOSTS:
        msg = f"--host must be loopback ({sorted(LOOPBACK_HOSTS)}), got {host!r}"
        raise errors.MegakernelConfigError(msg)
    return cleaned


def _opt_num(
    mapping: Mapping[str, object], key: str, default: float | None
) -> float | None:
    """Extract an optional finite float from `mapping[key]`, failing closed."""
    raw = mapping.get(key)
    if raw is None:
        return default
    numeric = as_scalar_number(raw)
    if numeric is None:
        msg = f"Field {key!r} must be numeric, got {raw!r}"
        raise errors.MegakernelConfigError(msg)
    return float(numeric)


def _req_num(mapping: Mapping[str, object], key: str, default: float) -> float:
    """Extract an optional finite float from `mapping[key]`, keeping `0.0`."""
    extracted = _opt_num(mapping, key, default)
    return default if extracted is None else extracted


def _opt_int(mapping: Mapping[str, object], key: str, default: int) -> int:
    """Extract an optional strict integer from `mapping[key]`, failing closed."""
    raw = mapping.get(key)
    if raw is None:
        return default
    return scalar_int(raw, key)


def _opt_enum_str(
    mapping: Mapping[str, object],
    key: str,
    default: str,
    valid: frozenset[str],
) -> str:
    """Extract and validate an optional enum string field, failing closed."""
    if key not in mapping or mapping[key] is None:
        return default
    raw_value = mapping[key]
    if type(raw_value) is not str:
        msg = f"Field {key!r} must be a string, got {type(raw_value).__name__}"
        raise errors.MegakernelConfigError(msg)
    cleaned = raw_value.strip().lower()
    if cleaned not in valid:
        msg = f"Unsupported {key}: {raw_value!r}"
        raise errors.MegakernelConfigError(msg)
    return cleaned


def _workload_from_json(payload: Mapping[str, object]) -> WorkloadSpec:
    """Construct validated `WorkloadSpec` (with `EfficiencySpec`) from JSON."""
    phase = cast("Phase", _opt_enum_str(payload, "phase", "decode", VALID_PHASES))
    defaults = WorkloadSpec(phase=phase)
    dispatch_regime = cast(
        "DispatchRegime",
        _opt_enum_str(
            payload,
            "dispatch_regime",
            defaults.dispatch_regime,
            VALID_DISPATCH_REGIMES,
        ),
    )
    default_efficiency = EfficiencySpec()
    efficiency = EfficiencySpec(
        eta_dense=_req_num(payload, "eta_dense", default_efficiency.eta_dense),
        eta_moe_grouped=_req_num(
            payload, "eta_moe_grouped", default_efficiency.eta_moe_grouped
        ),
        eta_hbm=_req_num(payload, "eta_hbm", default_efficiency.eta_hbm),
        eta_ici=_req_num(payload, "eta_ici", default_efficiency.eta_ici),
    )
    raw_alias = payload.get("vmem_alias")
    vmem_alias = (
        defaults.vmem_alias
        if raw_alias is None
        else scalar_bool(raw_alias, "vmem_alias")
    )
    raw_execution = payload.get("execution")
    execution = (
        cast(
            "ExecutionMode",
            _opt_enum_str(payload, "execution", "full", VALID_EXECUTION_MODES),
        )
        if raw_execution is not None
        else defaults.execution
    )
    raw_table_overrides = payload.get("host_table_hbm_overrides")
    host_table_hbm_overrides = (
        parse_host_table_hbm_overrides(raw_table_overrides)
        if raw_table_overrides is not None
        else defaults.host_table_hbm_overrides
    )
    return WorkloadSpec(
        phase=phase,
        batch=_opt_int(payload, "batch", defaults.batch),
        isl=_opt_int(payload, "isl", defaults.isl),
        osl=_opt_int(payload, "osl", defaults.osl),
        context=_opt_int(payload, "context", defaults.context),
        spec_tokens=_opt_int(payload, "spec_tokens", defaults.spec_tokens),
        chunk_size=_opt_int(payload, "chunk_size", defaults.chunk_size),
        tp=_opt_int(payload, "tp", defaults.tp),
        ep=_opt_int(payload, "ep", defaults.ep),
        dp=_opt_int(payload, "dp", defaults.dp),
        pp=_opt_int(payload, "pp", defaults.pp),
        dcp=_opt_int(payload, "dcp", defaults.dcp),
        pcp=_opt_int(payload, "pcp", defaults.pcp),
        vmem_budget_mib=_opt_num(payload, "vmem_budget_mib", defaults.vmem_budget_mib),
        dma_buffers=_opt_int(payload, "dma_buffers", defaults.dma_buffers),
        vmem_alias=vmem_alias,
        dispatch_regime=dispatch_regime,
        efficiency=efficiency,
        execution=execution,
        host_table_hbm_overrides=host_table_hbm_overrides,
        layout_execution=defaults.layout_execution,
        layout_host_table_hbm_overrides=defaults.layout_host_table_hbm_overrides,
        kv_page_size=_opt_int(payload, "kv_page_size", defaults.kv_page_size),
    )


def _validate_upload_filename(raw_name: object) -> str:
    """Validate uploaded trace filename against path traversal and suffixes."""
    if type(raw_name) is not str or not raw_name.strip():
        msg = "Upload requires a non-empty filename"
        raise errors.MegakernelConfigError(msg)
    cleaned = raw_name.strip()
    if (
        "/" in cleaned
        or "\\" in cleaned
        or ".." in cleaned
        or Path(cleaned).name != cleaned
    ):
        msg = f"Invalid upload filename: {raw_name!r}"
        raise errors.MegakernelConfigError(msg)
    if not cleaned.endswith(ALLOWED_UPLOAD_SUFFIXES):
        msg = (
            f"Unsupported upload extension {cleaned!r}; "
            f"expected {ALLOWED_UPLOAD_SUFFIXES}"
        )
        raise errors.MegakernelConfigError(msg)
    return cleaned


class _WorkbenchServerState:
    """Mutable session state held by the loopback workbench HTTP server."""

    def __init__(
        self,
        profile: ModelProfile,
        chip_spec: ChipSpec,
        *,
        measurements: MeasurementsData | None = None,
    ) -> None:
        """Initialize server state with model, chip, and optional calibration."""
        self.profile = profile
        self.chip_spec = chip_spec
        self._lock = threading.Lock()
        self.measurements = measurements
        self.sweep = advise_regimes(
            profile, chip_spec, options=RegimeSweepOptions(measurements=measurements)
        )

    def snapshot(self) -> tuple[MeasurementsData | None, AdvisorSweepReport]:
        """Return a thread-safe `(measurements, sweep)` snapshot pair."""
        with self._lock:
            return self.measurements, self.sweep

    def update_measurements(self, measurements: MeasurementsData) -> None:
        """Replace active measurements and swap the refreshed sweep atomically."""
        updated_sweep = advise_regimes(
            self.profile,
            self.chip_spec,
            options=RegimeSweepOptions(measurements=measurements),
        )
        with self._lock:
            self.measurements = measurements
            self.sweep = updated_sweep


def _calibrate_uploaded_bytes(
    state: _WorkbenchServerState, filename: str, raw_bytes: bytes
) -> dict[str, object]:
    """Calibrate uploaded trace or measurements bytes in a temporary folder."""
    with tempfile.TemporaryDirectory(prefix="mka_wb_upload_") as tmp_dir:
        upload_path = Path(tmp_dir) / filename
        upload_path.write_bytes(raw_bytes)
        if filename.endswith(".json"):
            try:
                parsed_doc = json.loads(raw_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                msg = f"Invalid JSON upload {filename}: {exc}"
                raise errors.MegakernelConfigError(msg) from exc
            if (
                type(parsed_doc) is dict
                and parsed_doc.get("schema") == MEASUREMENTS_SCHEMA
            ):
                state.update_measurements(
                    load_measurements(upload_path, chip_name=state.chip_spec.name)
                )
                return dict(parsed_doc)
        out_json = Path(tmp_dir) / "calibrated.json"
        is_preset = state.profile.name in MODEL_PRESETS
        payload = calibrate_trace(
            upload_path,
            out_json,
            options=CalibrateOptions(
                chip=state.chip_spec.name,
                model=state.profile.name if is_preset else None,
                num_layers=None if is_preset else state.profile.num_layers,
            ),
        )
        state.update_measurements(
            load_measurements(out_json, chip_name=state.chip_spec.name)
        )
        return dict(payload)


def _dispatch_post_route(
    state: _WorkbenchServerState, route: str, body: Mapping[str, object]
) -> tuple[HTTPStatus, dict[str, object]]:
    """Route a decoded JSON POST payload to the matching Python advisor API."""
    if route == "/api/advise":
        workload = _workload_from_json(body)
        active_measurements, _ = state.snapshot()
        return HTTPStatus.OK, evaluate_workbench_payload(
            state.profile,
            workload,
            state.chip_spec,
            measurements=active_measurements,
        )
    if route == "/api/phase_map":
        workload = _workload_from_json(body)
        active_measurements, active_sweep = state.snapshot()
        return HTTPStatus.OK, build_phase_map_payload(
            state.profile,
            state.chip_spec,
            active_sweep,
            options=PhaseMapOptions(
                phase=workload.phase,
                base_workload=workload,
                measurements=active_measurements,
            ),
        )
    if route == "/api/calibrate":
        filename = _validate_upload_filename(body.get("filename"))
        encoded = body.get("content_base64")
        if type(encoded) is not str:
            msg = "Upload requires string content_base64"
            raise errors.MegakernelConfigError(msg)
        try:
            raw_bytes = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            msg = f"Invalid base64 payload: {exc}"
            raise errors.MegakernelConfigError(msg) from exc
        if len(raw_bytes) > MAX_TRACE_UPLOAD_BYTES:
            return HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {
                "error": f"Decoded trace exceeds {MAX_TRACE_UPLOAD_BYTES} byte limit"
            }
        return HTTPStatus.OK, {
            "measurements": _calibrate_uploaded_bytes(state, filename, raw_bytes),
            "active": True,
        }
    return HTTPStatus.NOT_FOUND, {"error": "Unknown route"}


class _WorkbenchHTTPServer(ThreadingHTTPServer):
    """Loopback `ThreadingHTTPServer` holding typed workbench session state."""

    def __init__(
        self,
        server_address: tuple[str, int],
        state: _WorkbenchServerState,
    ) -> None:
        """Bind server socket and attach `state`."""
        self.workbench_state = state
        super().__init__(server_address, _WorkbenchHandler)


class _WorkbenchHTTPServerV6(_WorkbenchHTTPServer):
    """IPv6 loopback `ThreadingHTTPServer` bound to `::1`."""

    address_family = socket.AF_INET6


def _select_server_class(host: str) -> type[_WorkbenchHTTPServer]:
    """Select IPv6 or IPv4 loopback HTTP server class matching `host`."""
    return _WorkbenchHTTPServerV6 if host == "::1" else _WorkbenchHTTPServer


def _format_server_url(host: str, port: int) -> str:
    """Format an HTTP URL bracketing literal IPv6 host addresses."""
    formatted_host = f"[{host}]" if ":" in host else host
    return f"http://{formatted_host}:{port}/"


class _WorkbenchHandler(BaseHTTPRequestHandler):
    """HTTP handler serving `/` and `/api/{advise,phase_map,calibrate}`."""

    def log_message(self, *args: object, **kwargs: object) -> None:
        """Silence default stderr request logging during CLI/test runs."""
        _ = (args, kwargs)

    def _state(self) -> _WorkbenchServerState:
        """Return typed workbench session state attached to the server."""
        return cast("_WorkbenchHTTPServer", self.server).workbench_state

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        """Write an HTTP response with explicit status, type, and length."""
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        """Serialize `payload` as UTF-8 JSON and transmit."""
        self._send_bytes(
            status,
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _read_json_body(
        self, max_bytes: int
    ) -> tuple[HTTPStatus | None, dict[str, object]]:
        """Read and parse a size-bounded JSON POST body."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"}
        if content_length < 0:
            return HTTPStatus.BAD_REQUEST, {"error": "Negative Content-Length"}
        if content_length > max_bytes:
            return HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {
                "error": f"Payload exceeds {max_bytes} byte limit"
            }
        try:
            decoded = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {exc}"}
        mapped = as_mapping(decoded)
        if mapped is None:
            return HTTPStatus.BAD_REQUEST, {"error": "Expected JSON object body"}
        return None, dict(mapped)

    def do_GET(self) -> None:
        """Serve the pre-hydrated live Fusion Workbench HTML document."""
        if self.path not in ("/", "/index.html"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        state = self._state()
        active_measurements, active_sweep = state.snapshot()
        html_doc = render_workbench_html(
            state.profile,
            state.chip_spec,
            active_sweep,
            mode="live",
            measurements=active_measurements,
        )
        self._send_bytes(
            HTTPStatus.OK, html_doc.encode("utf-8"), "text/html; charset=utf-8"
        )

    def do_POST(self) -> None:
        """Dispatch JSON POST requests to live Python advisor endpoints."""
        max_bytes = (
            MAX_TRACE_UPLOAD_JSON_BODY_BYTES
            if self.path == "/api/calibrate"
            else MAX_JSON_BODY_BYTES
        )
        err_status, body = self._read_json_body(max_bytes)
        if err_status is not None:
            self._send_json(err_status, body)
            return
        try:
            status, payload = _dispatch_post_route(self._state(), self.path, body)
            self._send_json(status, payload)
        except (errors.MegakernelConfigError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def create_workbench_server(
    *,
    model: str | ModelProfile,
    chip: str | ChipSpec = DEFAULT_CHIP,
    host: str = DEFAULT_SERVE_HOST,
    port: int = DEFAULT_SERVE_PORT,
    measurements: str | Path | MeasurementsData | None = None,
) -> ThreadingHTTPServer:
    """Create a loopback-bound `ThreadingHTTPServer` for the workbench."""
    safe_host = validate_loopback_host(host)
    if port < 0 or port > MAX_TCP_PORT:
        msg = f"--port must be in [0, {MAX_TCP_PORT}], got {port}"
        raise errors.MegakernelConfigError(msg)
    profile = resolve_model_profile(model)
    chip_spec = resolve_chip_spec(chip)
    loaded_measurements = (
        measurements
        if measurements is None or type(measurements) is MeasurementsData
        else load_measurements(measurements, chip_name=chip_spec.name)
    )
    server_cls = _select_server_class(safe_host)
    return server_cls(
        (safe_host, port),
        _WorkbenchServerState(profile, chip_spec, measurements=loaded_measurements),
    )


def run_workbench_server(
    *,
    model: str | ModelProfile,
    chip: str | ChipSpec = DEFAULT_CHIP,
    options: WorkbenchServerOptions | None = None,
) -> int:
    """Start the loopback Fusion Workbench server until interrupted."""
    server_options = options or WorkbenchServerOptions()
    server = create_workbench_server(
        model=model,
        chip=chip,
        host=server_options.host,
        port=server_options.port,
        measurements=server_options.measurements,
    )
    bound_host, bound_port = server.server_address[:2]
    if server_options.open_browser:
        webbrowser.open(_format_server_url(str(bound_host), int(bound_port)))
    try:
        with suppress(KeyboardInterrupt):
            server.serve_forever()
    finally:
        server.server_close()
    return 0
