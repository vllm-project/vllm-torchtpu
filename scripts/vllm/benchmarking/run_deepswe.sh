#!/bin/bash
# Run DeepSWE v1.1 tasks with mini-swe-agent against a vLLM server and report
# the resolve rate.
#
# Usage:
#   ./scripts/vllm/benchmarking/run_deepswe.sh [--config CONFIG_NAME]
#       [--host HOST] [--port PORT]
#       [--tasks "TASK ..." | --n-tasks N [--sample-seed S]]
#       [--results-dir DIR] [--work-dir DIR] [--api-host ADDR]
#       [--n-concurrent N] [--agent-timeout-multiplier X]
#
# The server is started with `run_benchmarks.sh --config CONFIG_NAME
# --serve-only` (default deepseek-v41-flash-tp8-ep-deepswe: 1M context, the
# deepseek_v41 tool-call and reasoning parsers, reasoning_effort=max,
# temperature 1.0, top_p 0.95) and stopped at exit. With --host, an already
# running server is used instead; the config still names the served model.
#
# The harness is the one in the DeepSeek-V4.1-Flash model card's
# evaluation/README.md: datacurve-ai/pier with evaluation/dsh-minimal.patch,
# datacurve-ai/deep-swe, and mini-swe-agent with reasoning_effort=max. pier and
# deep-swe are cloned into --work-dir once and reused. All 113 tasks run by
# default; --tasks picks tasks by name, --n-tasks a seeded random sample.
#
# Requirements on this host: docker with the compose plugin, uv, git, and
# internet access. Each task runs in its own container behind a squid egress
# proxy, and pier builds a verifier image per task; mini-swe-agent installs
# inside each task container. Task containers reach the server at --api-host
# (default: --host, or this host's first IP for localhost). The CI container
# (run_in_docker.sh) has no docker daemon, so run this on the host.
#
# A task's agent steps run one after another, so the wall time is that of the
# slowest task: 1-2 h per task on tpu7x-8 without prefix caching.
#
# Results land in <results-dir>/: pier's job (jobs/deepswe/, with per-task
# trajectories and verifier logs), pier.log and summary.txt, plus serve.log and
# server.log when the script starts the server. Exits with pier's status.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PIER_COMMIT=0c802fc067a425345b24d1c69411aa98acf61a1d
DEEPSWE_COMMIT=0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea
MINI_SWE_AGENT_VERSION=2.2.6
PATCH_URL=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/evaluation/dsh-minimal.patch

CONFIG_NAME=deepseek-v41-flash-tp8-ep-deepswe
HOST=""
PORT="${PORT:-8000}"
TASKS=""
N_TASKS=""
SAMPLE_SEED=0
RESULTS_DIR=""
WORK_DIR="${DEEPSWE_WORK_DIR:-$HOME/.cache/vllm-torchtpu-deepswe}"
API_HOST=""
N_CONCURRENT=8
AGENT_TIMEOUT_MULTIPLIER=2

usage() {
    echo "Usage: $0 [--config CONFIG_NAME] [--host HOST] [--port PORT] [--tasks \"TASK ...\" | --n-tasks N [--sample-seed S]] [--results-dir DIR] [--work-dir DIR] [--api-host ADDR] [--n-concurrent N] [--agent-timeout-multiplier X]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG_NAME="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --n-tasks) N_TASKS="$2"; shift 2 ;;
        --sample-seed) SAMPLE_SEED="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --api-host) API_HOST="$2"; shift 2 ;;
        --n-concurrent) N_CONCURRENT="$2"; shift 2 ;;
        --agent-timeout-multiplier) AGENT_TIMEOUT_MULTIPLIER="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

CONFIG_FILE="$SCRIPT_DIR/configs/${CONFIG_NAME}.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
for tool in docker uv git curl python3; do
    if ! command -v "$tool" > /dev/null; then
        echo "ERROR: $tool is required on this host."
        exit 1
    fi
done
if ! docker compose version > /dev/null 2>&1; then
    echo "ERROR: the docker compose plugin is required on this host."
    exit 1
fi

# Read the served model and startup budget from the config in a subshell, so
# its exports do not leak into this script.
eval "$(
    bash -c '
        source "$1" > /dev/null
        : "${SERVER_READY_WAIT_MIN:=90}"
        declare -p MODEL SERVER_READY_WAIT_MIN
    ' _ "$CONFIG_FILE"
)"

if [ -z "$API_HOST" ]; then
    if [ -z "$HOST" ] || [ "$HOST" = localhost ] || [ "$HOST" = 127.0.0.1 ]; then
        API_HOST=$(hostname -I | awk '{print $1}')
    else
        API_HOST="$HOST"
    fi
fi
SERVER_URL="http://${HOST:-localhost}:$PORT"
API_BASE="http://$API_HOST:$PORT/v1"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../../../benchmark_runs/deepswe_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR" "$WORK_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

SERVE_PID=""

# Invoked by the EXIT trap below; shellcheck 0.10 reports that as SC2317 and
# later versions as SC2329.
# shellcheck disable=SC2317,SC2329
stop_server() {
    if [ -n "$SERVE_PID" ]; then
        echo "Stopping the server..."
        # run_benchmarks.sh stops vllm serve from its own EXIT trap.
        kill -TERM "$SERVE_PID" 2>/dev/null || true
        wait "$SERVE_PID" 2>/dev/null || true
        bash "$SCRIPT_DIR/cleanup_server.sh" > /dev/null 2>&1 || true
        SERVE_PID=""
    fi
}

if [ -z "$HOST" ]; then
    trap stop_server EXIT INT TERM
    bash "$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" \
        --results-dir "$RESULTS_DIR" --port "$PORT" --serve-only \
        > "$RESULTS_DIR/serve.log" 2>&1 &
    SERVE_PID=$!
    echo "Starting $CONFIG_NAME (pid=$SERVE_PID); log: $RESULTS_DIR/serve.log, $RESULTS_DIR/server.log"
fi

# Set up the harness while the server loads. The marker records what the checkout was prepared for, so a changed pin or
# port redoes it.
SETUP_MARKER="pier-$PIER_COMMIT-port$PORT deep-swe-$DEEPSWE_COMMIT"
if [ "$(cat "$WORK_DIR/.setup" 2>/dev/null)" != "$SETUP_MARKER" ]; then
    echo "Setting up pier and deep-swe in $WORK_DIR"
    rm -rf "$WORK_DIR/pier" "$WORK_DIR/deep-swe" "$WORK_DIR/.setup"
    git clone -q https://github.com/datacurve-ai/pier.git "$WORK_DIR/pier"
    git -C "$WORK_DIR/pier" checkout -q "$PIER_COMMIT"
    curl -fsSL -o "$WORK_DIR/dsh-minimal.patch" "$PATCH_URL"
    git -C "$WORK_DIR/pier" apply "$WORK_DIR/dsh-minimal.patch"
    # pier's egress proxy only allows ports 80 and 443; the agent in each
    # task container reaches the server through it.
    sed -i "s/^acl Safe_ports port 80 443\$/acl Safe_ports port 80 443 $PORT/" \
        "$WORK_DIR/pier/src/pier/environments/agent_setup.py"
    grep -q "^acl Safe_ports port 80 443 $PORT\$" "$WORK_DIR/pier/src/pier/environments/agent_setup.py"
    (cd "$WORK_DIR/pier" && uv sync -q)
    git clone -q https://github.com/datacurve-ai/deep-swe.git "$WORK_DIR/deep-swe"
    git -C "$WORK_DIR/deep-swe" checkout -q "$DEEPSWE_COMMIT"
    echo "$SETUP_MARKER" > "$WORK_DIR/.setup"
fi

if [ -n "$SERVE_PID" ]; then
    max_wait=$(( SERVER_READY_WAIT_MIN * 60 ))
    waited=0
    until curl -s -o /dev/null --connect-timeout 1 "$SERVER_URL/health"; do
        if ! kill -0 "$SERVE_PID" 2>/dev/null; then
            echo "ERROR: the server exited during startup."
            tail -50 "$RESULTS_DIR/serve.log"
            [ -f "$RESULTS_DIR/server.log" ] && tail -200 "$RESULTS_DIR/server.log"
            exit 1
        fi
        if [ "$waited" -ge "$max_wait" ]; then
            echo "ERROR: the server did not become ready within $SERVER_READY_WAIT_MIN minutes."
            exit 1
        fi
        sleep 10
        waited=$(( waited + 10 ))
    done
    echo "Server ready after ${waited}s"
fi
if ! curl -sf -o /dev/null --connect-timeout 5 "$API_BASE/models"; then
    echo "ERROR: no server at $API_BASE; set --api-host to an address task containers can reach."
    exit 1
fi

SELECT=()
if [ -n "$TASKS" ]; then
    read -r -a TASK_LIST <<< "$TASKS"
    for task in "${TASK_LIST[@]}"; do SELECT+=(-i "$task"); done
elif [ -n "$N_TASKS" ]; then
    SELECT=(--n-tasks "$N_TASKS" --sample-seed "$SAMPLE_SEED")
fi

# litellm's deepseek provider talks to the server; pier allowlists the
# api_base host in each task's egress proxy.
cat > "$RESULTS_DIR/mini-swe-agent.yaml" <<EOF
model:
  model_kwargs:
    api_base: $API_BASE
EOF

rm -rf "$RESULTS_DIR/jobs/deepswe"
echo "Running pier against $API_BASE ($MODEL), $N_CONCURRENT concurrent trials; log: $RESULTS_DIR/pier.log"
start=$(date +%s)
pier_rc=0
(
    cd "$WORK_DIR"
    uv run --project "$WORK_DIR/pier" pier run -p deep-swe/tasks "${SELECT[@]}" \
        --agent mini-swe-agent --model "deepseek/$MODEL" \
        --ak version="$MINI_SWE_AGENT_VERSION" \
        --ak reasoning_effort=max --ak cost_limit=0 \
        --ak config_file="$RESULTS_DIR/mini-swe-agent.yaml" \
        --ae DEEPSEEK_API_KEY=unused \
        --agent-timeout-multiplier "$AGENT_TIMEOUT_MULTIPLIER" \
        -n "$N_CONCURRENT" --no-delete -o "$RESULTS_DIR/jobs" --job-name deepswe -y
) > "$RESULTS_DIR/pier.log" 2>&1 || pier_rc=$?
echo "pier exited with $pier_rc after $(( ($(date +%s) - start) / 60 )) min"

python3 - "$RESULTS_DIR/jobs/deepswe" <<'EOF' | tee "$RESULTS_DIR/summary.txt"
import json
import pathlib
import sys

rows = []
for path in sorted(pathlib.Path(sys.argv[1]).glob("*/result.json")):
    result = json.loads(path.read_text())
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    exception = (result.get("exception_info") or {}).get("exception_type")
    rows.append((result["task_name"].rsplit("/", 1)[-1], int(rewards.get("reward") or 0), exception))
resolved = sum(reward for _, reward, _ in rows)
print(f"DeepSWE: {resolved}/{len(rows)} resolved" + (f" ({resolved / len(rows):.1%})" if rows else ""))
for task, reward, exception in rows:
    print(f"  {reward} {task}" + (f" ({exception})" if exception else ""))
EOF
exit "$pier_rc"
