"""
csv_recorder.py
---------------
Thread-safe, schema-evolving CSV recorder for persisting benchmark sweep results.
Dynamically handles schema evolution when new metrics or tasks are added during a sweep.
"""

import csv
import os
from typing import Any, Dict, List

from core.interfaces import IResultRecorder


class CsvResultRecorder(IResultRecorder):
    """
    Schema-evolving CSV recorder that writes benchmark metrics to disk.
    Automatically updates the CSV header if new metrics appear in subsequent coordinates.
    """

    def __init__(self, csv_file: str):
        """Initializes the recorder and ensures parent directory exists."""
        self.csv_file = csv_file
        csv_dir = os.path.dirname(os.path.abspath(csv_file))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

    def record(self,
               metadata: Dict[str, Any],
               metrics: Dict[str, Any],
               reproduction_cmd: str = "") -> None:
        """
        Records a completed trial with its sweep metadata, evaluation metrics, and CLI reproduction command.
        Rewrites header if new metrics are encountered.
        """
        # Combine sweep metadata and evaluation metrics into a single row dictionary
        record_row = {**metadata, **metrics}
        if reproduction_cmd:
            record_row["Reproduction_Command"] = reproduction_cmd

        # Check if the target CSV file already exists on disk
        file_exists = os.path.exists(self.csv_file) and os.path.getsize(
            self.csv_file) > 0
        existing_rows: List[Dict[str, Any]] = []
        fieldnames: List[str] = []

        if file_exists:
            # Read existing fieldnames and rows
            with open(self.csv_file, "r", newline="") as f_read:
                reader = csv.DictReader(f_read)
                fieldnames = list(reader.fieldnames or [])
                existing_rows = list(reader)
        else:
            fieldnames = list(record_row.keys())

        # Check if new metric keys need to be added to the CSV header
        needs_rewrite = False
        for key in record_row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
                needs_rewrite = True

        # Rewrite CSV with expanded headers if schema evolved, otherwise append row
        if needs_rewrite and file_exists:
            with open(self.csv_file, "w", newline="") as f_write:
                writer = csv.DictWriter(f_write,
                                        fieldnames=fieldnames,
                                        extrasaction="ignore",
                                        restval="")
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow(r)
                writer.writerow(record_row)
        else:
            with open(self.csv_file, "a", newline="") as f_append:
                writer = csv.DictWriter(f_append,
                                        fieldnames=fieldnames,
                                        extrasaction="ignore",
                                        restval="")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record_row)

        print(f">>> [RECORDER] Saved metrics to {self.csv_file}")
