#!/usr/bin/env python3
"""
run_coverage.py
---------------
Single-command test and coverage orchestration utility.
Initializes runtime dependencies before starting coverage, executes the full test suite,
prints detailed metric breakdowns, and generates HTML reports.
"""

import os
import sys
import unittest

os.environ["COVERAGE_RUN"] = "1"

import coverage

EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(EXPERIMENTS_DIR)

if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)


def main():
    print("=======================================================")
    print(">>> [TEST SUITE] Executing Unit & Integration Tests")
    print(f">>> Working Directory: {EXPERIMENTS_DIR}")
    print("=======================================================\n")

    # 2. Start coverage measurement for target modules
    cov = coverage.Coverage(source=[
        "core", "resolvers", "runners", "run_sweep_orchestrator", "sparsity"
    ],
                            config_file=".coveragerc")
    cov.start()

    # 3. Discover and run test suite
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)

    # 4. Stop and save coverage
    cov.stop()
    cov.save()

    print("\n=======================================================")
    print(">>> [COVERAGE REPORT] Computing Line Coverage Breakdown")
    print("=======================================================")

    cov.report(show_missing=True)
    cov.html_report(directory="htmlcov")
    print(
        f"\n>>> [REPORT] Interactive HTML coverage report: {os.path.join(EXPERIMENTS_DIR, 'htmlcov', 'index.html')}"
    )

    sys.exit(0 if test_result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
