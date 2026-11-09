#!/bin/bash
# The recipe's points, one place for a sweep and for a standalone launch: per (ISL, OSL, users) the layout; for the
# data-parallel points the slots per engine, the global prompt budget and the release group; for the tensor-parallel
# points the attention decode batch and the placement with copies. sweep.sh and server.sh both resolve a point here,
# and a value the caller has already set in the environment is kept (an explicit override, including an empty
# TPU_EXPERT_PLACEMENT for no placement file).
declare -A POINT_TABLE=(
  [8192,1024,4]="TP8_EP"
  [8192,1024,8]="TP8_EP"
  [8192,1024,16]="TP8_EP"
  [8192,1024,32]="TP8_EP"
  [8192,1024,64]="DP8_EP 16 8192 8"
  [8192,1024,128]="DP8_EP 20 8192 16"
  [8192,1024,256]="DP8_EP 40 3072 8"
  [8192,1024,512]="DP8_EP 80 6144 16"
  [8192,1024,1024]="DP8_EP 144 16384 16"
)
declare -A POINT_FLAGS=(
  # Tensor-parallel prompt passes use 512-row bucket spacing to limit padding.
  # The default extra buckets (4, 8, 48) remain available for decode. These points
  # use vocabulary-parallel greedy argmax and build GDN decode tables in-kernel.
  [8192,1024,4]="TPU_RPA_DECODE_BATCH=4 TPU_TOKEN_BUCKET_STEP=512 TPU_GREEDY_TOP_TOKENS=1 TPU_GDN_TABLES_IN_KERNEL=1 TPU_GATED_NORM_TILE=256"
  [8192,1024,8]="TPU_RPA_DECODE_BATCH=8 TPU_TOKEN_BUCKET_STEP=512 TPU_GREEDY_TOP_TOKENS=1 TPU_GDN_TABLES_IN_KERNEL=1 TPU_GATED_NORM_TILE=256"
  [8192,1024,16]="TPU_RPA_DECODE_BATCH=8 TPU_TOKEN_BUCKET_STEP=512 TPU_GREEDY_TOP_TOKENS=1 TPU_GDN_TABLES_IN_KERNEL=1 TPU_GATED_NORM_TILE=256"
  [8192,1024,32]="TPU_RPA_DECODE_BATCH=8 TPU_TOKEN_BUCKET_STEP=512 TPU_GREEDY_TOP_TOKENS=1 TPU_GDN_TABLES_IN_KERNEL=1 TPU_GATED_NORM_TILE=256"
  # At 64 and 128 users, the scores entry selects one holder per copied expert.
  # DP uses a 64-row norm tile because each rank carries the full head count.
  [8192,1024,64]="TPU_MOE_LOWTOK_FROM_SCORES=1 TPU_MOE_LOWTOK_SCORES_STRICT=1 TPU_GDN_TABLES_IN_KERNEL=1 TPU_GATED_NORM_TILE=64"
  # At 256 users and above, use the ids entry and placement without copies.
  # Slot-count buckets avoid padding partially occupied decode steps to the
  # next power of two. Preserve the default extras when adding 20, 80 or 144.
  [8192,1024,128]="TPU_MOE_LOWTOK_FROM_SCORES=1 TPU_MOE_LOWTOK_SCORES_STRICT=1 TPU_GDN_TABLES_IN_KERNEL=1 TPU_GATED_NORM_TILE=64 TPU_TOKEN_BUCKET_EXTRA=4,8,20,48"
  [8192,1024,256]="TPU_GDN_TABLES_IN_KERNEL=1 TPU_GATED_NORM_TILE=64"
  [8192,1024,512]="TPU_GDN_TABLES_IN_KERNEL=1 TPU_GATED_NORM_TILE=64 TPU_TOKEN_BUCKET_EXTRA=4,8,48,80"
  [8192,1024,1024]="TPU_GDN_TABLES_IN_KERNEL=1 TPU_GATED_NORM_TILE=64 TPU_TOKEN_BUCKET_EXTRA=4,8,48,144"
)

# point_settings ISL OSL CONC: the point's KEY=VALUE lines, in the order they apply; nothing for an unknown point.
point_settings() {
  local isl=$1 osl=$2 conc=$3 sharding seqs budget group f
  local spec="${POINT_TABLE[$isl,$osl,$conc]:-}"
  [ -z "$spec" ] && return 0
  read -r sharding seqs budget group <<< "$spec"
  echo "SHARDING=$sharding"
  if [ -n "${seqs:-}" ]; then
    echo "MAX_NUM_SEQS=$seqs"; echo "GLOBAL_MAX_NUM_BATCHED_TOKENS=$budget"; echo "DP_SCHED_BUFFER_THRESHOLD=$group"
  fi
  for f in ${POINT_FLAGS[$isl,$osl,$conc]:-}; do echo "$f"; done
  if [ "$isl,$osl" = "8192,1024" ]; then
    case "$conc" in
      4|8|16|32|64|128) echo "TPU_EXPERT_PLACEMENT=$(dirname "${BASH_SOURCE[0]}")/expert_placement_dp8_ep8_copies128.json" ;;
    esac
  fi
}

# apply_point_defaults ISL OSL CONC: export the point's settings for every key the environment does not already set.
apply_point_defaults() {
  local line key value
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    key="${line%%=*}"; value="${line#*=}"
    if [ -z "${!key+x}" ]; then export "$key=$value"; fi
  done < <(point_settings "$1" "$2" "$3")
}
