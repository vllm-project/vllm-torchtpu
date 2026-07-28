#!/bin/bash
set -o pipefail

# Text Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Define Experiments
# Format: "model tp_size extra_args -- batch_size input_len output_len [max_model_len] [max_num_batched_tokens]"
# extra_args can be empty or flags like --enable-expert-parallel
EXPERIMENTS=(
    # --- Small model smoke tests (commented out) ---
    # "Qwen/Qwen3-0.6B 1  -- 1 32 128"
    # "Qwen/Qwen3-0.6B 4  -- 1 32 128"
    # "Qwen/Qwen3-0.6B 4  -- 8 32 128"

    # --- 30B MoE FP8 (commented out) ---
    # "Qwen/Qwen3-30B-A3B-FP8 4  -- 1 32 128"
    # "Qwen/Qwen3-30B-A3B-FP8 4  -- 8 1024 1"
    # "Qwen/Qwen3-30B-A3B-FP8 8 --enable-expert-parallel -- 1 1024 1"
    # "Qwen/Qwen3-30B-A3B-FP8 8 --enable-expert-parallel -- 8 1024 1"

    # --- Qwen-30B 480B-FP8 ---
    # Configs used by Google for performance evaluation
    # "Qwen/Qwen3-Coder-30B-A3B-Instruct 1 --no-enable-prefix-caching -- 8 1024 1024"
    # "Qwen/Qwen3-Coder-30B-A3B-Instruct 1 --no-enable-prefix-caching -- 8 1024 8192"
    # "Qwen/Qwen3-Coder-30B-A3B-Instruct 1 --no-enable-prefix-caching -- 8 8192 1024"
    # "Qwen/Qwen3-Coder-30B-A3B-Instruct 8 --enable-expert-parallel --no-enable-prefix-caching -- 8 1024 1024"
    # "Qwen/Qwen3-Coder-30B-A3B-Instruct 8 --enable-expert-parallel --no-enable-prefix-caching -- 8 1024 8192"
    # "Qwen/Qwen3-Coder-30B-A3B-Instruct 8 --enable-expert-parallel --no-enable-prefix-caching -- 8 8192 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel --no-enable-prefix-caching --kv-cache-dtype fp8 -- 8 1024 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel --no-enable-prefix-caching --kv-cache-dtype fp8 -- 8 1024 8192"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel --no-enable-prefix-caching --kv-cache-dtype fp8 -- 8 8192 1024"


    # --- 480B FP8 EP ---
    # Matching fbcode benchmark_configs.json settings as closely as possible.
    # Benchmark defaults: prefill input_lens=1024,8192 batch_sizes=1,2,4,8
    #                     decode context_lens=1024,8192 batch_sizes=1,8,16,32,64,128,256
    #                     max_model_len=16384 (covers 8192*1.8 + 1024*1.8 = 16590 worst case)
    #
    # Constraint: --max-num-batched-tokens 1024 required to limit compile buckets to [16..1024].
    # Without this, XLA compilation of 8 workers in parallel OOMs the 944GB host RAM container.
    # Chunked prefill handles longer sequences by splitting into 1024-token chunks.
    # Compile cache (symlinked to /mnt/hyperdisk/_cache_vllm) must be warm for cold-start to survive.

    # Prefill profiles — full benchmark matrix: input_lens={1024,8192} x batch_sizes={1,2,4,8}
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 1 1024 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 2 1024 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 4 1024 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 8 1024 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 1 8192 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 2 8192 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 4 8192 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 8 8192 1 16384 1024"

    # Decode profiles — full benchmark matrix: context_lens={1024,8192} x batch_sizes={1,8,16,32,64,128,256}
    # context_len is simulated by input_len (prefills that many tokens, then decodes 1)
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 1 1 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 8 1 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 16 1 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 32 1 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 64 1 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 128 1 1 16384 1024"
    # "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 8 --enable-expert-parallel -- 256 1 1 16384 1024"
)

# 2. Execution Loop
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED_EXPERIMENTS=()

echo -e "${GREEN}Starting Profiling Suite - Start Time: $TIMESTAMP${NC}"
echo "--------------------------------------------------------"

for experiment in "${EXPERIMENTS[@]}"; do
    # Split on " -- " to separate model/tp/flags from batch config
    model_part="${experiment%% -- *}"
    config_part="${experiment##* -- }"

    # Parse model part: "model tp_size [extra_args...]"
    read -r model tp_size extra_args <<< "$model_part"

    # Parse config part: "batch_size input_len output_len [max_model_len] [max_num_batched_tokens]"
    read -r batch_size input_len output_len max_model_len_override max_num_batched_tokens <<< "$config_part"
    max_model_len=$((input_len + output_len))
    # Use override if provided, otherwise use computed value with 1024 floor
    if [[ -n "$max_model_len_override" ]]; then
        max_model_len=$max_model_len_override
    elif [[ $max_model_len -lt 1024 ]]; then
        max_model_len=1024
    fi

    safe_model_name="${model//\//_}"
    suffix=""
    [[ "$extra_args" == *"--enable-expert-parallel"* ]] && suffix="_ep"
    output_dir="profiles/${safe_model_name}_tp${tp_size}${suffix}_bs${batch_size}_in${input_len}_out${output_len}_${TIMESTAMP}"

    echo -e "${CYAN}Running Experiment:${NC}"
    echo "  Model:       $model"
    echo "  TP:          $tp_size"
    echo "  Extra args:  ${extra_args:-(none)}"
    echo "  Config:      BS=$batch_size, In=$input_len, Out=$output_len"
    echo "  MaxLen:      $max_model_len"
    echo "  Artifacts:   $output_dir"

    # Build command
    cmd=(python3 examples/tpu_profiling.py
        --model "$model"
        --batch-size "$batch_size"
        --input-len "$input_len"
        --output-len "$output_len"
        --max-model-len "$max_model_len"
        --profile-result-dir "$output_dir"
        --tensor-parallel-size "$tp_size"
    )
    [[ -n "$max_num_batched_tokens" ]] && cmd+=(--max-num-batched-tokens "$max_num_batched_tokens")
    # shellcheck disable=SC2206
    [[ -n "$extra_args" ]] && cmd+=($extra_args)

    if "${cmd[@]}"; then
        echo -e "${GREEN}[SUCCESS] Experiment completed.${NC}"
    else
        echo -e "${RED}[FAILURE] Experiment failed!${NC}"
        FAILED_EXPERIMENTS+=("$model tp=$tp_size ${extra_args} (BS=$batch_size, In=$input_len, Out=$output_len)")
    fi

    echo "--------------------------------------------------------"
done

echo ""
if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    echo -e "${GREEN}All experiments finished successfully!${NC}"
else
    echo -e "${RED}Some experiments failed:${NC}"
    for fail in "${FAILED_EXPERIMENTS[@]}"; do
        echo -e "  - $fail"
    done
    exit 1
fi
