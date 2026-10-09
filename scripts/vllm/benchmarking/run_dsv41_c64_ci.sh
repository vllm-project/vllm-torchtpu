#!/bin/bash
# Benchmark first; publish only a complete, valid C64 result. Upload errors
# fail the job while run_in_docker.sh still preserves the local artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR=/perf_eval_results/deepseek-v41-flash-tp8-ep-100k-decode-c64-dspark-offload

bash "$SCRIPT_DIR/run_decode_benchmark.sh" \
    --config deepseek-v41-flash-tp8-ep-100k-decode-c64-dspark-offload \
    --results-dir "$RESULTS_DIR" --rounds 1

python3 "$SCRIPT_DIR/upload_decode_sheet.py" \
    --summary "$RESULTS_DIR/decode/isl98000_osl5000_c64/summary.json" \
    --spreadsheet-id 1A2nrPEJgRd6wYGSqW5IZwaHDYph6viYH40kuRaBMpzw \
    --sheet-id 1542472756
