#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Layout presets for the decode hill-climb. Source after env.sh:
#   source ./layout.sh            # reads LAYOUT, GLOBAL_SEQS, MTP_K, MNBT, HOST_RANK, HOST_IPS
# Sets: PAR_ARGS (array), HEADLESS_ARGS (array), SEQS_PER_ENGINE, COMPILE_SIZES, ENGINES (count)
# shellcheck disable=SC2034  # arrays are consumed by serve_sweep.sh after sourcing
#
# Four-host presets cover the 16 chips / 32 TensorCores of the 2x2x4 slice.
#   dp8tp4      : baseline of the 2026-10-30 run (2 engines per host, TP4 inside a host)
#   dp4tp8dcp2  : one engine per host, TP8 over the host's 4 chips, KV pages striped over 2 cores
#   tp32dcp8    : one engine over all 4 hosts, TP32, KV pages striped over 8 cores (production decode shape)
#   dp2tp16dcp4 : two engines of 2 hosts each (experimental launch form, see README)
# Eight- and sixteen-host presets (2x4x4 = 32 chips, 4x4x4 = 64 chips): two engines of TP4 per host, as dp8tp4.
#   dp16tp4     : 16 engines of TP4 over 8 hosts (EP over 64 ranks, 16 members per TP rank's DP group)
#   dp32tp4     : 32 engines of TP4 over 16 hosts (EP over 128 ranks, 32 members per TP rank's DP group)
# Single-host presets cover the 4 chips / 8 TensorCores of one 2x2x1 host (NATIVE_HOST_IPS has one address).
#   tp8         : one engine, TP8, no DCP
#   tp8xdcp2    : one engine, TP8, KV pages striped over 2 cores
#   dp2xtp4     : two engines of TP4 on the host
#   dp4xtp2     : four engines of TP2 on the host (EP over 4 DP members per TP rank)
#   dp8xtp1     : eight engines of TP1 on the host (EP over 8 DP members)
LAYOUT=${LAYOUT:-dp8tp4}
GLOBAL_SEQS=${GLOBAL_SEQS:-96}
MTP_K=${MTP_K:-3}
MNBT=${MNBT:-1024}
HEADLESS_ARGS=()
case "$LAYOUT" in
  dp8tp4|dp16tp4|dp32tp4)
    ENGINES=${LAYOUT%tp4}; ENGINES=${ENGINES#dp}
    PAR_ARGS=(--data-parallel-size "$ENGINES" --data-parallel-size-local 2 --nnodes 1
              --tensor-parallel-size 4 --decode-context-parallel-size 1
              --data-parallel-start-rank $((HOST_RANK * 2)) --data-parallel-hybrid-lb --master-port 29500
              --data-parallel-address "$MASTER_ADDRESS" --data-parallel-rpc-port 29650) ;;
  dp4tp8dcp2|dp4tp8)
    ENGINES=4
    DCP=2; [[ "$LAYOUT" == dp4tp8 ]] && DCP=1
    PAR_ARGS=(--data-parallel-size 4 --data-parallel-size-local 1 --nnodes 1
              --tensor-parallel-size 8 --decode-context-parallel-size "$DCP"
              --data-parallel-start-rank "$HOST_RANK" --data-parallel-hybrid-lb --master-port 29500
              --data-parallel-address "$MASTER_ADDRESS" --data-parallel-rpc-port 29650) ;;
  tp32dcp8|tp32dcp4)
    ENGINES=1
    DCP=8; [[ "$LAYOUT" == tp32dcp4 ]] && DCP=4
    PAR_ARGS=(--tensor-parallel-size 32 --decode-context-parallel-size "$DCP"
              --nnodes 4 --node-rank "$HOST_RANK" --master-addr "$MASTER_ADDRESS" --master-port 29500)
    [[ "$HOST_RANK" != 0 ]] && HEADLESS_ARGS=(--headless) ;;
  dp2tp16dcp4)
    ENGINES=2
    GROUP_HEAD=${HOST_IPS[$((HOST_RANK / 2 * 2))]}
    PAR_ARGS=(--data-parallel-size 2 --data-parallel-size-local 1 --nnodes 4 --node-rank "$HOST_RANK"
              --tensor-parallel-size 16 --decode-context-parallel-size 4
              --data-parallel-start-rank $((HOST_RANK / 2))
              --data-parallel-address "$MASTER_ADDRESS" --data-parallel-rpc-port 29650
              --master-addr "$GROUP_HEAD" --master-port 29500)
    [[ "$HOST_RANK" != 0 ]] && HEADLESS_ARGS=(--headless) ;;
  tp8|tp8xdcp2)
    ENGINES=1
    DCP=1; [[ "$LAYOUT" == tp8xdcp2 ]] && DCP=2
    PAR_ARGS=(--tensor-parallel-size 8 --decode-context-parallel-size "$DCP" --nnodes 1 --master-port 29500) ;;
  dp2xtp4)
    ENGINES=2
    PAR_ARGS=(--data-parallel-size 2 --data-parallel-size-local 2 --nnodes 1
              --tensor-parallel-size 4 --decode-context-parallel-size 1 --master-port 29500
              --data-parallel-address "$MASTER_ADDRESS" --data-parallel-rpc-port 29650) ;;
  dp4xtp2|dp8xtp1)
    ENGINES=${LAYOUT:2:1}
    PAR_ARGS=(--data-parallel-size "$ENGINES" --data-parallel-size-local "$ENGINES" --nnodes 1
              --tensor-parallel-size $((8 / ENGINES)) --decode-context-parallel-size 1 --master-port 29500
              --data-parallel-address "$MASTER_ADDRESS" --data-parallel-rpc-port 29650) ;;
  *) echo "unknown LAYOUT=$LAYOUT" >&2; return 2 ;;
esac
case "$LAYOUT" in
  tp8|tp8xdcp2|dp2xtp4|dp4xtp2|dp8xtp1) LAYOUT_HOSTS=1 ;;
  dp16tp4|dp32tp4) LAYOUT_HOSTS=$((ENGINES / 2)) ;;
  *) LAYOUT_HOSTS=4 ;;
esac
if [[ "${#HOST_IPS[@]}" != "$LAYOUT_HOSTS" ]]; then
  echo "LAYOUT=$LAYOUT needs $LAYOUT_HOSTS host address(es), got ${#HOST_IPS[@]}" >&2; return 2
fi
SEQS_PER_ENGINE=$((GLOBAL_SEQS / ENGINES))
VERIFY_TOKENS=$((SEQS_PER_ENGINE * (MTP_K + 1)))
COMPILE_SIZES=$(python3 -c "print(sorted({$SEQS_PER_ENGINE, $VERIFY_TOKENS, $MNBT}))" | tr -d ' ')
export LAYOUT GLOBAL_SEQS MTP_K MNBT ENGINES SEQS_PER_ENGINE COMPILE_SIZES
