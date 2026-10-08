"""
json_recorder.py
----------------
Robust, thread-safe, schema-adhering JSON recorder implementing IResultRecorder.
Persists hierarchical Sweep -> Experiments data matching the MARC JSON Schema.
"""

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.interfaces import IResultRecorder


def _json_serializer(obj: Any) -> Any:
    """Fallback serializer for non-primitive types (numpy, torch, pathlib, set)."""
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "__fspath__"):
        return os.fspath(obj)
    return str(obj)


class JsonResultRecorder(IResultRecorder):
    """
    Hierarchical JSON recorder that manages a single `sweep.json` file.
    Holds top-level sweep metadata and appends individual experiment trial records.
    Guarantees thread-safe recording and atomic disk persistence.
    """

    def __init__(
        self,
        json_file: str,
        sweep_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        user_ldap: str = "",
        hardware: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the recorder and writes/loads the root sweep JSON document.
        """
        self.json_file = os.path.abspath(json_file)
        self.json_dir = os.path.dirname(self.json_file)
        if self.json_dir:
            os.makedirs(self.json_dir, exist_ok=True)

        self._lock = threading.Lock()
        self.sweep_id = (sweep_id or os.path.basename(self.json_dir) or
                         datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        self.user_ldap = (user_ldap or os.environ.get("USER")
                          or os.environ.get("LOGNAME") or "unknown")

        # Normalize top-level metadata dict ensuring required schema fields
        raw_meta = dict(metadata or {})
        raw_meta.setdefault("description", "")
        raw_meta.setdefault("mode", "profile")

        with self._lock:
            # If resuming an existing sweep, load existing document; otherwise initialize fresh
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
                    "user_ldap": self.user_ldap,
                    "metadata": raw_meta,
                    "experiments": [],
                }
                if hardware:
                    self.data["hardware"] = hardware
                self._flush_unlocked()

    def record(
        self,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        reproduction_cmd: str = "",
    ) -> None:
        """
        Thread-safely appends an individual completed experiment trial to the sweep.
        """
        with self._lock:
            trial_record = self._new_trial(metadata, metrics, reproduction_cmd)
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

    def finalize(self,
                 status: str = "COMPLETED",
                 error_message: Optional[str] = None) -> None:
        """
        Marks sweep completion (or failure) and stamps completed_at.
        """
        with self._lock:
            self.data["status"] = status
            self.data["completed_at"] = datetime.now(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if error_message is not None:
                self.data["error_message"] = error_message
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """
        Atomically writes JSON to disk via temporary file rename.
        """
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
    """
    Composite recorder that broadcasts record calls to multiple IResultRecorder instances.
    Enables dual-persistence (e.g. CsvResultRecorder + JsonResultRecorder).
    """

    def __init__(self, recorders: List[IResultRecorder]):
        self.recorders = recorders

    def record(
        self,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        reproduction_cmd: str = "",
    ) -> None:
        for recorder in self.recorders:
            recorder.record(metadata, metrics, reproduction_cmd)

    def finalize(self,
                 status: str = "COMPLETED",
                 error_message: Optional[str] = None) -> None:
        for recorder in self.recorders:
            recorder.finalize(status, error_message)
