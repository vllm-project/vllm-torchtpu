#!/usr/bin/env bash
# InferenceX AgentX（aiperf --scenario inferencex-agentx-mvp）打 2P1D proxy，每 30s 抓 P0 / P1 / D 的 /metrics。
# 参数和 scripts/dsv41_pd/run_agentx.sh 相同（同一个 seed / 数据集 / scenario），只是地址从 GKE 现查、多抓一个 P。
#   source c512.env && TAG=gke2p1d bash run_agentx.sh
# 在开发机上跑（224 核；dsv41-bench pod 只有 16 vCPU，c512 下客户端可能成瓶颈）。结果：/zzlfs/temu/results/agentx-c<CONC>-<TAG>-<T>/
set -uo pipefail
source "$(dirname "$0")/common.sh"
P0_IP=$(pod_ip "$(role_pod p0)"); P1_IP=$(pod_ip "$(role_pod p1)"); D_IP=$(pod_ip "$(role_pod d)")
PROXY_URL=${PROXY_URL:-http://$P0_IP:8000}
CONC=${CONC:-512}; DURATION=${DURATION:-3600}; TAG=${TAG:-gke2p1d}; T=$(date +%m%d-%H%M)
TOKENIZER=${TOKENIZER:-/zzlfs/temu/tools/dsv41-tokenizer}
OUT=/zzlfs/temu/results/agentx-c${CONC}-${TAG}-$T; mkdir -p "$OUT/metrics"
{ echo "proxy=$PROXY_URL p0=$P0_IP p1=$P1_IP d=$D_IP conc=$CONC duration=$DURATION"
  echo "P_ENV=${P_ENV:-} D_ENV=${D_ENV:-}"; env | grep -E '^PROXY_' | sort; } > "$OUT/config.txt"
export HF_HOME=/zzlfs/temu/hf-cache AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
( while true; do ts=$(date +%s)
    curl -s --max-time 10 "$P0_IP:8100/metrics" > "$OUT/metrics/p0-$ts.prom"
    curl -s --max-time 10 "$P1_IP:8100/metrics" > "$OUT/metrics/p1-$ts.prom"
    curl -s --max-time 10 "$D_IP:8200/metrics"  > "$OUT/metrics/d-$ts.prom"
    sleep 30; done ) &
SCRAPER=$!; trap 'kill $SCRAPER 2>/dev/null' EXIT
echo "aiperf c$CONC ${DURATION}s -> $OUT"
cd "$OUT" && /zzlfs/temu/.venv-aiperf/bin/aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "$PROXY_URL" \
  --model deepseek-v41 \
  --tokenizer "$TOKENIZER" \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --endpoint-type chat \
  --concurrency "$CONC" \
  --benchmark-duration "$DURATION" \
  --random-seed 2026110501 \
  --use-server-token-count \
  --streaming \
  --extra-inputs ignore_eos:true \
  --cache-bust first_turn_prefix \
  --system-idle-gap-cap-seconds 10 \
  --trajectory-start-min-ratio 0.0 \
  --trajectory-start-max-ratio 1.0 \
  --ui simple \
  --output-artifact-dir "$OUT/artifacts" ${AIPERF_EXTRA:-} > "$OUT/aiperf.log" 2>&1
rc=$?
echo "aiperf rc=$rc out=$OUT"
/zzlfs/temu/.venv-epic/bin/python -I "$REPO/scripts/dsv41_pd/agentx_summary.py" "$OUT" 2>/dev/null || true
exit $rc
