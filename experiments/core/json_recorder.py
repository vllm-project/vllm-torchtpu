"""Thread-safe recorder for persisting experiment results to sweep.json.

Complies with the formal Experiment Sweep JSON schema v1.0.0 and optionally
uploads artifacts (such as xplane traces) and sweep metadata to GCS.
"""

import getpass
import glob
import json
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.interfaces import IResultRecorder


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer fallback for non-serializable objects."""
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, os.PathLike):
        return str(obj)
    if hasattr(obj, "__dict__") and obj.__dict__:
        return obj.__dict__
    return str(obj)


def _env_flag(name: str) -> bool:
    """Returns True if the env var is set to a truthy value (not 0/false)."""
    value = os.environ.get(name, "").strip().lower()
    return value not in ("", "0", "false")


def _default_user() -> str:
    """Best-effort owner name, resolved lazily; never raises."""
    try:
        return os.environ.get("USER") or getpass.getuser() or "unknown"
    except Exception:
        return "unknown"


class JsonResultRecorder(IResultRecorder):
    """Records experiment trials into a structured JSON file adhering to schema v1.0.0."""

    def __init__(
        self,
        json_file: str,
        sweep_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        user: str = "",
        hardware: Optional[Dict[str, Any]] = None,
        upload_gcs: bool = True,
    ):
        self.upload_gcs = upload_gcs and not _env_flag(
            "EXPERIMENTS_NO_UPLOAD_GCS")
        self.json_file = os.path.abspath(json_file)
        self.json_dir = os.path.dirname(self.json_file)
        if self.json_dir:
            os.makedirs(self.json_dir, exist_ok=True)

        self._lock = threading.Lock()
        self.sweep_id = (sweep_id or os.path.basename(self.json_dir) or
                         datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        self.user = (user or (metadata.get("user") if metadata else None)
                     or _default_user())

        raw_meta = dict(metadata or {})
        raw_meta.setdefault("description", "")
        raw_meta.setdefault("mode", "profile")

        with self._lock:
            if os.path.exists(self.json_file) and os.path.getsize(
                    self.json_file) > 0:
                with open(self.json_file, "r") as f:
                    self.data = json.load(f)
                self.data["status"] = "RUNNING"
                if metadata:
                    self.data.setdefault("metadata", {}).update(raw_meta)
            else:
                now_str = datetime.now(
                    timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                self.data: Dict[str, Any] = {
                    "schema_version": "1.0.0",
                    "sweep_id": self.sweep_id,
                    "created_at": now_str,
                    "completed_at": None,
                    "status": "RUNNING",
                    "user": self.user,
                    "metadata": raw_meta,
                    "experiments": [],
                }
                if hardware:
                    self.data["hardware"] = hardware
                self._flush_unlocked()

    @property
    def gcs_sweep_dir(self) -> str:
        bucket = os.environ.get("MARC_GCS_STAGING_BUCKET",
                                "cloud-tpu-inference-eval-staging")
        base_prefix = os.environ.get("MARC_GCS_INCOMING_PREFIX", "incoming")
        user = self.user or "unknown"
        return f"gs://{bucket}/{base_prefix}/{user}/{self.sweep_id}"

    def record(
        self,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        reproduction_cmd: str = "",
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Thread-safely appends an experiment trial record."""
        with self._lock:
            trial_record = self._new_trial(metadata, metrics, reproduction_cmd,
                                           artifacts)
            self.data["experiments"].append(trial_record)
            self._flush_unlocked()
            print(f">>> [JSON RECORDER] Appended trial "
                  f"{trial_record['experiment_id']} to "
                  f"{os.path.basename(self.json_file)}")

    def _new_trial(
        self,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        reproduction_cmd: str,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Builds a COMPLETED trial record; caller must hold the lock."""
        trial_idx = len(self.data["experiments"])
        trial_record: Dict[str, Any] = {
            "experiment_id": f"{self.sweep_id}_trial_{trial_idx:03d}",
            "trial_index": trial_idx,
            "status": "COMPLETED",
            "timestamp":
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "parameters": dict(metadata),
            "metrics": dict(metrics),
        }
        if reproduction_cmd:
            trial_record["reproduction_command"] = reproduction_cmd
        if artifacts:
            trial_record["artifacts"] = dict(artifacts)
        return trial_record

    def record_failure(
        self,
        metadata: Dict[str, Any],
        error_message: str,
        reproduction_cmd: str = "",
    ) -> None:
        """
        Appends a FAILED experiment trial carrying the runner error message.
        """
        with self._lock:
            trial_record = self._new_trial(metadata, {}, reproduction_cmd)
            trial_record["status"] = "FAILED"
            trial_record["error_message"] = error_message
            self.data["experiments"].append(trial_record)
            self._flush_unlocked()
            print(f">>> [JSON RECORDER] Recorded FAILED trial "
                  f"{trial_record['experiment_id']}: {error_message}")

    def _is_tracing_enabled(self) -> bool:
        """Returns True if tracing was requested in sweep metadata or any trial parameter."""
        meta = self.data.get("metadata", {})
        if meta.get("trace") is True or meta.get("trace_configs"):
            return True
        if meta.get("config", {}).get("trace_configs") or meta.get(
                "config", {}).get("trace") is True:
            return True
        return any(
            exp.get("parameters", {}).get("trace") is True
            for exp in self.data.get("experiments", []))

    def _verify_traces_saved(self,
                             traces_dir: str,
                             timeout_s: float = 30.0) -> List[str]:
        """Waits for active trace dump markers to clear and returns non-empty trace paths."""
        if not os.path.exists(traces_dir):
            return []
        start = time.time()
        while time.time() - start < timeout_s:
            markers = [
                f for f in os.listdir(traces_dir)
                if f.startswith(".canonical_ts_") and f.endswith(".tmp")
            ]
            if not markers:
                break
            time.sleep(0.2)
        else:
            return []

        saved = []
        for root, _, files in os.walk(traces_dir):
            for f in files:
                if f.endswith((".xplane.pb", ".pt.trace.json.gz")):
                    fp = os.path.join(root, f)
                    if os.path.getsize(fp) > 0:
                        saved.append(fp)
        return saved

    def finalize(self,
                 status: str = "COMPLETED",
                 error_message: Optional[str] = None) -> None:
        """
        Marks sweep completion (or failure), stamps completed_at, and uploads
        traces and sweep.json to GCS.
        """
        with self._lock:
            self.data["status"] = status
            self.data["completed_at"] = datetime.now(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if error_message is not None:
                self.data["error_message"] = error_message

            traces_dir = os.path.join(self.json_dir, "traces")
            if self.upload_gcs and self._is_tracing_enabled(
            ) and os.path.isdir(traces_dir):
                self._upload_traces_unlocked(traces_dir)

            self._flush_unlocked()
            if self.upload_gcs:
                self._upload_to_gcs_unlocked()

    def _upload_traces_unlocked(self,
                                traces_dir: Optional[str] = None) -> List[str]:
        """Uploads traces directory to GCS and maps trace files to trials."""
        if traces_dir is None:
            traces_dir = os.path.join(self.json_dir, "traces")

        trace_files = self._verify_traces_saved(traces_dir)
        if not trace_files:
            return []

        cmd = [
            "gsutil", "-m", "-q", "cp", "-r", traces_dir,
            f"{self.gcs_sweep_dir}/"
        ]
        try:
            res = subprocess.run(cmd,
                                 capture_output=True,
                                 text=True,
                                 timeout=300)
            if res.returncode == 0:
                print(
                    f">>> [GCS UPLOAD] Successfully uploaded {len(trace_files)} trace file(s) to {self.gcs_sweep_dir}/traces/"
                )
            else:
                print(
                    f">>> [GCS UPLOAD WARNING] Traces upload failed (code {res.returncode}): {res.stderr.strip()}"
                )
                return []
        except Exception as e:
            print(
                f">>> [GCS UPLOAD ERROR] Exception during traces upload: {e}")
            return []

        trace_uris = [
            f"{self.gcs_sweep_dir}/traces/{os.path.relpath(fp, traces_dir)}"
            for fp in trace_files
        ]

        self._assign_trace_paths(trace_uris, traces_dir)
        return trace_uris

    def _assign_trace_paths(self, trace_uris: List[str],
                            traces_dir: str) -> None:
        """Attaches trace URIs only to trials recorded with trace=True.

        Timestamped profile sessions under plugins/profile/ are matched 1:1,
        in chronological order, with traced trials in record order. Trials
        that were not traced always get an empty trace_paths list.
        """
        experiments = self.data.get("experiments", [])
        for exp in experiments:
            exp.setdefault("artifacts", {}).setdefault("trace_paths", [])

        traced_exps = [
            e for e in experiments
            if e.get("parameters", {}).get("trace") is True
        ]
        if not traced_exps:
            return

        profile_dirs = sorted(
            glob.glob(os.path.join(traces_dir, "plugins", "profile", "*")))
        if len(profile_dirs) != len(traced_exps):
            print(f">>> [GCS UPLOAD WARNING] {len(profile_dirs)} profile "
                  f"session(s) for {len(traced_exps)} traced trial(s); "
                  "attaching all trace URIs to every traced trial")
            for exp in traced_exps:
                exp["artifacts"]["trace_paths"] = list(trace_uris)
            return

        for exp, pdir in zip(traced_exps, profile_dirs):
            ts = os.path.basename(pdir)
            exp["artifacts"]["trace_paths"] = [
                u for u in trace_uris
                if f"/plugins/profile/{ts}/" in u and u.endswith(".xplane.pb")
            ]

    def _upload_to_gcs_unlocked(self) -> None:
        """Uploads sweep.json to GCS staging bucket."""
        gcs_uri = f"{self.gcs_sweep_dir}/sweep.json"
        try:
            res = subprocess.run(
                ["gsutil", "-q", "cp", self.json_file, gcs_uri],
                capture_output=True,
                text=True,
                timeout=30)
            if res.returncode == 0:
                print(
                    f">>> [GCS UPLOAD] Successfully uploaded sweep JSON to {gcs_uri}"
                )
            else:
                print(
                    f">>> [GCS UPLOAD WARNING] Upload failed (code {res.returncode}): {res.stderr.strip()}"
                )
        except Exception as e:
            print(f">>> [GCS UPLOAD ERROR] Exception during GCS upload: {e}")

    def _flush_unlocked(self) -> None:
        """Atomically writes JSON to disk via temporary file rename."""
        temp_fd, temp_path = tempfile.mkstemp(dir=self.json_dir, suffix=".tmp")
        try:
            with os.fdopen(temp_fd, "w") as f:
                json.dump(self.data, f, indent=2, default=_json_serializer)
            os.replace(temp_path, self.json_file)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


class CompositeResultRecorder(IResultRecorder):
    """Composite recorder broadcasting calls to multiple IResultRecorder instances."""

    def __init__(self, recorders: List[IResultRecorder]):
        self.recorders = recorders

    def record(
        self,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        reproduction_cmd: str = "",
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> None:
        for recorder in self.recorders:
            try:
                recorder.record(metadata,
                                metrics,
                                reproduction_cmd,
                                artifacts=artifacts)
            except TypeError:
                recorder.record(metadata, metrics, reproduction_cmd)

    def record_failure(self,
                       metadata: Dict[str, Any],
                       error_message: str,
                       reproduction_cmd: str = "") -> None:
        for recorder in self.recorders:
            recorder.record_failure(metadata, error_message, reproduction_cmd)

    def finalize(self,
                 status: str = "COMPLETED",
                 error_message: Optional[str] = None) -> None:
        for recorder in self.recorders:
            recorder.finalize(status, error_message)
