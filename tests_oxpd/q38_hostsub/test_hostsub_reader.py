# SPDX-License-Identifier: Apache-2.0
"""The HOSTSUBS census reader's selftest, run as a test (fresh python -I)."""

import os
import subprocess
import sys


def test_hostsub_reader_selftest():
    reader = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q38_hostsub_read.py")
    p = subprocess.run([sys.executable, "-I", reader, "--selftest"], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and "HOSTSUB_SELFTEST PASS (32/32 checks)" in p.stdout, p.stdout + p.stderr
