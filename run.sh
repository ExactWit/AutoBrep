#!/usr/bin/env bash
# exp_launcher entry — AutoBrep (unconditional infer + PC-conditioned train/infer).
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${HOME}/software/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-autobrep}"

MODE="${1:?usage: $0 <mode> [--exp-dir ...]}"
shift

EXP_DIR_ARG=""
OUTPUT_DIR_ARG=""
DATA_DIR_ARG=""
DATASPLIT_ARG=""
DATASET_ARG=""
TASK_ARG=""
CHECKPOINT_ARG=""
WEIGHT_FOLDER="${WEIGHT_FOLDER:-/data/hdd/outputs/AutoBrep}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_BATCHES="${NUM_BATCHES:-10}"
COMPLEXITY="${COMPLEXITY:-medium}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
VERTEX_THRESHOLD="${VERTEX_THRESHOLD:-0.002}"
SEWING_TOLERANCE="${SEWING_TOLERANCE:-0.002}"
Z_THRESHOLD="${Z_THRESHOLD:-0}"
SEED="${SEED:-689447}"
USE_SEED="${USE_SEED:-0}"
DEBUG="${DEBUG:-1}"
FORMAT_ARG="${FORMAT:-step}"
RESUME_FROM_ARG=""
MAX_EPOCHS="${MAX_EPOCHS:-5}"
LR="${LR:-0.0001}"
LIMIT_TRAIN="${LIMIT_TRAIN:-50000}"
LIMIT_VAL="${LIMIT_VAL:-500}"
PC_NUM_POINTS="${PC_NUM_POINTS:-2048}"
PC_NUM_LATENTS="${PC_NUM_LATENTS:-64}"
ACCUM_GRAD="${ACCUM_GRAD:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PC_CONDITIONED="${PC_CONDITIONED:-0}"
POINT_CLOUD="${POINT_CLOUD:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-dir) EXP_DIR_ARG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR_ARG="$2"; shift 2 ;;
    --data-dir|--dataset-dir) DATA_DIR_ARG="$2"; shift 2 ;;
    --datasplit) DATASPLIT_ARG="$2"; shift 2 ;;
    --dataset) DATASET_ARG="$2"; shift 2 ;;
    --task) TASK_ARG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT_ARG="$2"; shift 2 ;;
    --weight-folder) WEIGHT_FOLDER="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-batches) NUM_BATCHES="$2"; shift 2 ;;
    --complexity) COMPLEXITY="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --vertex-threshold) VERTEX_THRESHOLD="$2"; shift 2 ;;
    --sewing-tolerance) SEWING_TOLERANCE="$2"; shift 2 ;;
    --z-threshold) Z_THRESHOLD="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --use-seed) USE_SEED="$2"; shift 2 ;;
    --debug) DEBUG="$2"; shift 2 ;;
    --format) FORMAT_ARG="$2"; shift 2 ;;
    --resume-from) RESUME_FROM_ARG="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --limit-train) LIMIT_TRAIN="$2"; shift 2 ;;
    --limit-val) LIMIT_VAL="$2"; shift 2 ;;
    --pc-num-points) PC_NUM_POINTS="$2"; shift 2 ;;
    --pc-num-latents) PC_NUM_LATENTS="$2"; shift 2 ;;
    --accumulate-grad-batches) ACCUM_GRAD="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --pc-conditioned) PC_CONDITIONED="$2"; shift 2 ;;
    --point-cloud) POINT_CLOUD="$2"; shift 2 ;;
    --index|--sample-id) shift 2 ;;
    --) shift; break ;;
    -*) echo "[run.sh] unknown option: $1" >&2; exit 1 ;;
    *) break ;;
  esac
done

activate_env() {
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
  fi
  # AutoBrep deps partially in ~/.local; do not set PYTHONNOUSERSITE=1.
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="${REPO_DIR}/core/src:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  cd "${REPO_DIR}"
}

case "${MODE}" in
  capabilities)
    exec cat <<'JSON'
{
  "modes": ["capabilities", "train", "infer"],
  "datasets": ["abc", "abc-1m"],
  "tasks": {"abc": ["gen"], "abc-1m": ["gen"]},
  "env_file": "environment.yml",
  "checkpoints": {
    "ar": "ar.ckpt",
    "surf_fsq": "surf-fsq.ckpt",
    "edge_fsq": "edge-fsq.ckpt",
    "pc_cond": "last.ckpt"
  },
  "infer_requires_sample": false,
  "defaults": {
    "train": {
      "weight_folder": "/data/hdd/outputs/AutoBrep",
      "batch_size": 1,
      "max_epochs": 5,
      "lr": 0.0001,
      "limit_train": 50000,
      "limit_val": 500,
      "pc_num_points": 2048,
      "pc_num_latents": 64,
      "accumulate_grad_batches": 4,
      "num_workers": 2
    },
    "infer": {
      "weight_folder": "/data/hdd/outputs/AutoBrep",
      "batch_size": 8,
      "num_batches": 10,
      "complexity": "medium",
      "temperature": 1.0,
      "top_p": 0.9,
      "vertex_threshold": 0.002,
      "sewing_tolerance": 0.002,
      "z_threshold": 0,
      "seed": 689447,
      "use_seed": false,
      "debug": true,
      "pc_conditioned": false
    }
  },
  "args": {
    "train": [
      "--weight-folder",
      "--batch-size",
      "--max-epochs",
      "--lr",
      "--limit-train",
      "--limit-val",
      "--pc-num-points",
      "--pc-num-latents",
      "--accumulate-grad-batches",
      "--num-workers",
      "--gpu",
      "--resume-from"
    ],
    "infer": [
      "--weight-folder",
      "--batch-size",
      "--num-batches",
      "--complexity",
      "--temperature",
      "--top-p",
      "--vertex-threshold",
      "--sewing-tolerance",
      "--z-threshold",
      "--seed",
      "--use-seed",
      "--debug",
      "--checkpoint",
      "--gpu",
      "--pc-conditioned",
      "--point-cloud"
    ]
  },
  "arg_fields": {
    "train": [
      {"key": "weight_folder", "flag": "--weight-folder", "label": "预训练权重目录", "type": "text"},
      {"key": "batch_size", "flag": "--batch-size", "label": "batch_size", "type": "number"},
      {"key": "max_epochs", "flag": "--max-epochs", "label": "max_epochs", "type": "number"},
      {"key": "lr", "flag": "--lr", "label": "learning rate", "type": "number"},
      {"key": "limit_train", "flag": "--limit-train", "label": "limit_train", "type": "number"},
      {"key": "limit_val", "flag": "--limit-val", "label": "limit_val", "type": "number"},
      {"key": "pc_num_points", "flag": "--pc-num-points", "label": "点云点数", "type": "number"},
      {"key": "pc_num_latents", "flag": "--pc-num-latents", "label": "条件 token 数", "type": "number"},
      {"key": "accumulate_grad_batches", "flag": "--accumulate-grad-batches", "label": "梯度累积", "type": "number"},
      {"key": "num_workers", "flag": "--num-workers", "label": "num_workers", "type": "number"}
    ],
    "infer": [
      {"key": "weight_folder", "flag": "--weight-folder", "label": "权重目录", "type": "text"},
      {"key": "batch_size", "flag": "--batch-size", "label": "batch_size", "type": "number"},
      {"key": "num_batches", "flag": "--num-batches", "label": "采样批次数", "type": "number"},
      {"key": "complexity", "flag": "--complexity", "label": "复杂度", "type": "select", "choices": ["random", "easy", "medium", "hard"]},
      {"key": "temperature", "flag": "--temperature", "label": "temperature", "type": "number"},
      {"key": "top_p", "flag": "--top-p", "label": "top_p", "type": "number"},
      {"key": "vertex_threshold", "flag": "--vertex-threshold", "label": "vertex_threshold", "type": "number"},
      {"key": "sewing_tolerance", "flag": "--sewing-tolerance", "label": "sewing_tolerance", "type": "number"},
      {"key": "z_threshold", "flag": "--z-threshold", "label": "z_threshold", "type": "number"},
      {"key": "seed", "flag": "--seed", "label": "seed", "type": "number"},
      {"key": "use_seed", "flag": "--use-seed", "label": "固定随机种子", "type": "bool"},
      {"key": "debug", "flag": "--debug", "label": "debug 图/点云", "type": "bool"},
      {"key": "pc_conditioned", "flag": "--pc-conditioned", "label": "点云条件推理", "type": "bool"},
      {"key": "point_cloud", "flag": "--point-cloud", "label": "点云 .npy (N,3)", "type": "text"}
    ]
  }
}
JSON
    ;;

  train)
    activate_env
    if [[ -z "${EXP_DIR_ARG}" ]]; then
      echo "[run.sh] ERROR: --exp-dir is required for train" >&2
      exit 1
    fi
    DATA_ROOT="${DATA_DIR_ARG:-/data/hdd/datasets/ABC-1M}"
    mkdir -p "${EXP_DIR_ARG}/checkpoints" "${EXP_DIR_ARG}/metrics" "${EXP_DIR_ARG}/tensorboard"
    TRAIN_ARGS=(
      --exp-dir "${EXP_DIR_ARG}"
      --data-dir "${DATA_ROOT}"
      --dataset "${DATASET_ARG:-abc-1m}"
      --task "${TASK_ARG:-gen}"
      --weight-folder "${WEIGHT_FOLDER}"
      --gpu "${GPU}"
      --batch-size "${BATCH_SIZE}"
      --max-epochs "${MAX_EPOCHS}"
      --lr "${LR}"
      --limit-train "${LIMIT_TRAIN}"
      --limit-val "${LIMIT_VAL}"
      --pc-num-points "${PC_NUM_POINTS}"
      --pc-num-latents "${PC_NUM_LATENTS}"
      --accumulate-grad-batches "${ACCUM_GRAD}"
      --num-workers "${NUM_WORKERS}"
    )
    if [[ -n "${OUTPUT_DIR_ARG}" ]]; then
      TRAIN_ARGS+=(--output-dir "${OUTPUT_DIR_ARG}")
    fi
    if [[ -n "${DATASPLIT_ARG}" ]]; then
      TRAIN_ARGS+=(--datasplit "${DATASPLIT_ARG}")
    fi
    if [[ -n "${RESUME_FROM_ARG}" ]]; then
      TRAIN_ARGS+=(--resume-from "${RESUME_FROM_ARG}")
    fi
    echo "[run.sh] train data=${DATA_ROOT} weight=${WEIGHT_FOLDER} → ${EXP_DIR_ARG}" >&2
    exec python -u "${REPO_DIR}/scripts/train_pc_pipeline.py" "${TRAIN_ARGS[@]}"
    ;;

  infer)
    activate_env
    if [[ -z "${OUTPUT_DIR_ARG}" ]]; then
      echo "[run.sh] ERROR: --output-dir is required for infer" >&2
      exit 1
    fi
    if [[ -z "${EXP_DIR_ARG}" ]]; then
      echo "[run.sh] ERROR: --exp-dir is required for infer" >&2
      exit 1
    fi
    mkdir -p "${EXP_DIR_ARG}/checkpoints" "${EXP_DIR_ARG}/metrics" "${OUTPUT_DIR_ARG}/infer"

    if [[ -z "${WEIGHT_FOLDER}" || ! -d "${WEIGHT_FOLDER}" ]]; then
      if [[ -f "${EXP_DIR_ARG}/checkpoints/ar.ckpt" ]]; then
        WEIGHT_FOLDER="${EXP_DIR_ARG}/checkpoints"
      elif [[ -d /data/hdd/outputs/AutoBrep ]]; then
        WEIGHT_FOLDER="/data/hdd/outputs/AutoBrep"
      fi
    fi

    INFER_ARGS=(
      --exp-dir "${EXP_DIR_ARG}"
      --output-dir "${OUTPUT_DIR_ARG}"
      --dataset "${DATASET_ARG:-abc}"
      --task "${TASK_ARG:-gen}"
      --weight-folder "${WEIGHT_FOLDER}"
      --gpu "${GPU}"
      --batch-size "${BATCH_SIZE}"
      --num-batches "${NUM_BATCHES}"
      --complexity "${COMPLEXITY}"
      --temperature "${TEMPERATURE}"
      --top-p "${TOP_P}"
      --vertex-threshold "${VERTEX_THRESHOLD}"
      --sewing-tolerance "${SEWING_TOLERANCE}"
      --z-threshold "${Z_THRESHOLD}"
      --seed "${SEED}"
      --use-seed "${USE_SEED}"
      --debug "${DEBUG}"
      --format "${FORMAT_ARG}"
      --pc-conditioned "${PC_CONDITIONED}"
    )
    if [[ -n "${DATA_DIR_ARG}" ]]; then
      INFER_ARGS+=(--data-dir "${DATA_DIR_ARG}")
    fi
    if [[ -n "${DATASPLIT_ARG}" ]]; then
      INFER_ARGS+=(--datasplit "${DATASPLIT_ARG}")
    fi
    if [[ -n "${CHECKPOINT_ARG}" ]]; then
      INFER_ARGS+=(--checkpoint "${CHECKPOINT_ARG}")
    fi
    if [[ -n "${POINT_CLOUD}" ]]; then
      INFER_ARGS+=(--point-cloud "${POINT_CLOUD}")
    fi
    echo "[run.sh] infer weight_folder=${WEIGHT_FOLDER} pc=${PC_CONDITIONED} → ${OUTPUT_DIR_ARG}/infer" >&2
    exec python -u "${REPO_DIR}/scripts/infer_pipeline.py" "${INFER_ARGS[@]}"
    ;;

  *)
    echo "[run.sh] unknown mode: ${MODE} (supported: capabilities, train, infer)" >&2
    exit 1
    ;;
esac
