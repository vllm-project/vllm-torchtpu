#!/usr/bin/env bash
# InferenceX AgentX（aiperf --scenario inferencex-agentx-mvp）打 1P1D proxy，同时每 30s 抓 P / D 的 /metrics。
#   CONC=256 DURATION=3600 TAG=base bash run_agentx.sh
set -uo pipefail
CONC=${CONC:-256}; DURATION=${DURATION:-3600}; TAG=${TAG:-run}; T=$(date +%m%d-%H%M)
OUT=/zzlfs/temu/results/agentx-c${CONC}-${TAG}-$T; mkdir -p $OUT/metrics
export HF_HOME=/zzlfs/temu/hf-cache AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
( while true; do ts=$(date +%s); curl -s --max-time 10 10.128.0.100:8100/metrics > $OUT/metrics/p-$ts.prom; curl -s --max-time 10 10.128.0.156:8200/metrics > $OUT/metrics/d-$ts.prom; sleep 30; done ) &
SCRAPER=$!; trap 'kill $SCRAPER 2>/dev/null' EXIT
cd $OUT && /zzlfs/temu/.venv-aiperf/bin/aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url http://10.128.0.100:8000 \
  --model deepseek-v41 \
  --tokenizer /tmp/dsv41-tokenizer \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --endpoint-type chat \
  --concurrency $CONC \
  --benchmark-duration $DURATION \
  --random-seed 2026110501 \
  --use-server-token-count \
  --streaming \
  --extra-inputs ignore_eos:true \
  --cache-bust first_turn_prefix \
  --system-idle-gap-cap-seconds 10 \
  --trajectory-start-min-ratio 0.0 \
  --trajectory-start-max-ratio 1.0 \
  --ui simple \
  --output-artifact-dir $OUT/artifacts ${AIPERF_EXTRA:-} > $OUT/aiperf.log 2>&1
echo "aiperf rc=$? out=$OUT"
