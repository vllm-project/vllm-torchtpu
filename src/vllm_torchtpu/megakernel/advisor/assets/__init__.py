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
"""Static report and workbench CSS/JS assets for the megakernel advisor."""

from __future__ import annotations

import functools
import importlib.resources

__all__ = ["load_report_css", "load_workbench_css", "load_workbench_js"]


def _read_asset(filename: str) -> str:
    """Read a UTF-8 static asset bundled inside the `assets` package."""
    package_name = __package__ or __name__
    return (
        importlib.resources.files(package_name)
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )


@functools.lru_cache(maxsize=1)
def load_report_css() -> str:
    """Load the standalone HTML report stylesheet as a UTF-8 string."""
    return _read_asset("report.css")


@functools.lru_cache(maxsize=1)
def load_workbench_css() -> str:
    """Load the workbench CSS asset bundled with the advisor package."""
    return _read_asset("workbench.css")


@functools.lru_cache(maxsize=1)
def load_workbench_js() -> str:
    """Load the thin UI/SVG workbench JavaScript asset."""
    return _read_asset("workbench.js")
