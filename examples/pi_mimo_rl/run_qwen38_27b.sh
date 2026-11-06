#!/usr/bin/env bash
# Qwen3.8-27B on a six-node Ray cluster: 3 training nodes and 3 inference nodes.
# Arguments after the script follow these in train.py's command line, so they override them.
set -euo pipefail

SLIME_DIR=${SLIME_DIR:?Set SLIME_DIR to slime at 859938dc}
EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HF_CHECKPOINT=${HF_CHECKPOINT:?Set HF_CHECKPOINT to Qwen3.8-27B}
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT to a fresh shared experiment directory}
LOG_ROOT=${LOG_ROOT:-$RUN_ROOT}
export RAY_ADDRESS=${RAY_ADDRESS:?Set RAY_ADDRESS to the existing Ray cluster}
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the Ray dashboard URL}
PROMPT_DATA=${PROMPT_DATA:?Set PROMPT_DATA to the MiMo training rows}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLIME_DIR:${MEGATRON_DIR:?Set MEGATRON_DIR to your Megatron-LM checkout}:$EXAMPLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GOMAXPROCS=2
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME:?Set SOCKET_IFNAME to the inter-node network interface}
export NCCL_SOCKET_IFNAME=$SOCKET_IFNAME TP_SOCKET_IFNAME=$SOCKET_IFNAME
export MIMO_TASKS=${MIMO_TASKS:?Set MIMO_TASKS to the MiMo Harbor tasks directory}
export MIMO_TEMPLATES=${MIMO_TEMPLATES:-}
export HARBOR_DIR=${HARBOR_DIR:?Set HARBOR_DIR to tokin[harbor] installed with pip --target}
export PI_ARCHIVE=${PI_ARCHIVE:?Set PI_ARCHIVE to the prebuilt archive pi.py builds}
export TRIALS_DIR=${TRIALS_DIR:-$RUN_ROOT/trials}
export TOKIN_CHAT_TEMPLATE=${TOKIN_CHAT_TEMPLATE:-qwen38}
export E2B_API_KEY=${E2B_API_KEY:?Set E2B_API_KEY}
export RAY_USE_UVLOOP=0

test -s "$PROMPT_DATA"
test -d "$MIMO_TASKS"
test -s "$PI_ARCHIVE"
if [[ -e "$LOG_ROOT/run.log" ]]; then
    echo 'LOG_ROOT/run.log already exists; choose a new attempt log directory.' >&2
    exit 1
fi
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

source "$SLIME_DIR/scripts/models/qwen3.5-27B.sh"
TRACKING_ARGS=()
CHECKPOINT_ARGS=()
if (( SAVE_INTERVAL > 0 )); then
    CHECKPOINT_ARGS=(--save "$RUN_ROOT/checkpoints" --save-interval "$SAVE_INTERVAL")
fi
if [[ -n ${WANDB_PROJECT:-} ]]; then
    TRACKING_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT"
        --wandb-group "${WANDB_NAME:-qwen-agent-rl}" --disable-wandb-random-suffix --wandb-dir "$LOG_ROOT")
fi

# Runtime credentials belong in a private file, never in argv or shell tracing.
umask 077
RUNTIME_ENV=$(mktemp "$RUN_ROOT/runtime-env.XXXXXX.json")
trap 'rm -f -- "$RUNTIME_ENV"' EXIT
python - "$RUNTIME_ENV" <<'PY'
import json, os, sys
keys = {
    "PYTHONPATH", "PYTHONUNBUFFERED", "CUDA_DEVICE_MAX_CONNECTIONS",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "GOMAXPROCS", "GLOO_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
    "NCCL_NVLS_ENABLE",
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY",
    "MIMO_TASKS", "MIMO_TEMPLATES", "HARBOR_DIR", "PI_ARCHIVE", "TRIALS_DIR", "TOKIN_CHAT_TEMPLATE", "RAY_USE_UVLOOP",
}
keys.update(k for k in os.environ if k.startswith(("E2B_", "WANDB_")))
with open(sys.argv[1], "w") as stream:
    json.dump({"env_vars": {k: os.environ[k] for k in keys if k in os.environ}}, stream)
PY

cd "$SLIME_DIR"
ray job submit --address="$RAY_DASHBOARD_ADDRESS" --runtime-env="$RUNTIME_ENV" -- \
    python -u "$SLIME_DIR/train.py" \
    --actor-num-nodes 3 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --rollout-num-gpus 24 \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" --load "${LOAD_CHECKPOINT:-$HF_CHECKPOINT}" \
    "${CHECKPOINT_ARGS[@]}" \
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async \
    --fully-async-group-prefetch-factor 2 \
    --custom-generate-function-path generate.generate \
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata \
    --rollout-shuffle --apply-chat-template-kwargs '{"reasoning_effort":"medium"}' \
    --num-rollout 200 --rollout-batch-size 8 --n-samples-per-prompt 8 \
    --global-batch-size 64 --num-steps-per-rollout 1 --micro-batch-size 1 \
    --rollout-max-context-len 131072 --rollout-max-response-len 8192 \
    --rollout-temperature 1 --rollout-top-p 0.95 --rollout-stop-token-ids 248046 248044 \
    --use-score-centering --use-tis --tis-clip 2 --tis-clip-low 0 \
    --partial-rollout --flush-cache-interval 3 \
    --tensor-model-parallel-size 4 --pipeline-model-parallel-size 3 \
    --decoder-first-pipeline-num-layers 22 --decoder-last-pipeline-num-layers 20 \
    --context-parallel-size 2 --sequence-parallel \
    --qwen-gdn-implementation megatron \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu 32768 \
    --log-probs-chunk-size 1024 --balance-data --use-distributed-optimizer \
    --pg-loss-type reinforce --disable-grpo-std-normalization --calculate-per-token-loss \
    --advantage-estimator grpo --kl-loss-coef 0 --kl-coef 0 --entropy-coef 0 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --optimizer-cpu-offload --optimizer-offload-fraction 1 --overlap-cpu-optimizer-d2h-h2d \
    --use-precision-aware-optimizer --use-stateless-adam --no-save-optim \
    --rollout-num-gpus-per-engine 2 \
    --router-policy consistent_hashing \
    --sglang-server-concurrency 14 \
    --sglang-enable-hierarchical-cache --sglang-hicache-ratio 2 \
    --sglang-hicache-mem-layout page_first_direct --sglang-hicache-io-backend direct \
    --sglang-mem-fraction-static 0.8 --sglang-context-length 131072 \
    --sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps 3 \
    --sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens 4 \
    --sglang-mamba-radix-cache-strategy extra_buffer --sglang-mamba-full-memory-ratio 0.5 \
    --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3 \
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 --attention-backend flash \
    --rollout-data-transport straw --rollout-data-dir "$RUN_ROOT/rollout_data" --rollout-queue-online-gc \
    --rollout-storage-profile local \
    "${TRACKING_ARGS[@]}" "$@" \
    2>&1 | tee "$LOG_ROOT/run.log"
