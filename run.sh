#!/usr/bin/env bash
# exp_launcher entry — AutoBrep (unconditional + autocomplete infer).
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
AUTOCOMPLETE="${AUTOCOMPLETE:-0}"
CONDITION_JSON="${CONDITION_JSON:-}"
ABC_STEM="${ABC_STEM:-}"
FACE_IDS="${FACE_IDS:-}"
NUM_CONDITION_FACES="${NUM_CONDITION_FACES:-}"
CONDITION_MODE="${CONDITION_MODE:-random}"
ABC_SPLIT="${ABC_SPLIT:-train}"

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
    --autocomplete|--mode) AUTOCOMPLETE="$2"; shift 2 ;;
    --condition-json) CONDITION_JSON="$2"; shift 2 ;;
    --abc-stem) ABC_STEM="$2"; shift 2 ;;
    --face-ids) FACE_IDS="$2"; shift 2 ;;
    --num-condition-faces) NUM_CONDITION_FACES="$2"; shift 2 ;;
    --condition-mode) CONDITION_MODE="$2"; shift 2 ;;
    --abc-split) ABC_SPLIT="$2"; shift 2 ;;
    --index|--sample-id|--resume-from|--max-steps|--val-check-interval|--limit-val-batches|--lr|--pc-num-points|--pc-num-latents|--accumulate-grad-batches|--num-workers|--pc-conditioned|--point-cloud|--max-epochs|--limit-train|--limit-val)
      shift 2 ;;
    --) shift; break ;;
    -*) echo "[run.sh] unknown option: $1" >&2; exit 1 ;;
    *) break ;;
  esac
done

# --mode autocomplete (string) → enable flag
if [[ "${AUTOCOMPLETE}" == "autocomplete" ]]; then
  AUTOCOMPLETE=1
fi

activate_env() {
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
  fi
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="${REPO_DIR}/core/src:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  cd "${REPO_DIR}"
}

case "${MODE}" in
  capabilities)
    exec cat <<'JSON'
{
  "modes": ["capabilities", "infer"],
  "datasets": ["abc", "abc-1m"],
  "tasks": {"abc": ["gen"], "abc-1m": ["gen"]},
  "env_file": "environment.yml",
  "checkpoints": {
    "ar": "ar.ckpt",
    "surf_fsq": "surf-fsq.ckpt",
    "edge_fsq": "edge-fsq.ckpt"
  },
  "infer_requires_sample": false,
  "defaults": {
    "infer": {
      "weight_folder": "/data/hdd/outputs/AutoBrep",
      "batch_size": 4,
      "num_batches": 1,
      "complexity": "medium",
      "temperature": 1.0,
      "top_p": 0.9,
      "vertex_threshold": 0.002,
      "sewing_tolerance": 0.002,
      "z_threshold": 0,
      "seed": 689447,
      "use_seed": false,
      "debug": true,
      "autocomplete": false,
      "condition_mode": "random",
      "abc_split": "train"
    }
  },
  "args": {
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
      "--autocomplete",
      "--condition-json",
      "--abc-stem",
      "--face-ids",
      "--num-condition-faces",
      "--condition-mode",
      "--abc-split",
      "--data-dir"
    ]
  },
  "arg_fields": {
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
      {"key": "autocomplete", "flag": "--autocomplete", "label": "启用 BOGEOM 补全", "type": "bool"},
      {"key": "condition_json", "flag": "--condition-json", "label": "条件 JSON 路径", "type": "text"},
      {"key": "abc_stem", "flag": "--abc-stem", "label": "ABC stem", "type": "text"},
      {"key": "face_ids", "flag": "--face-ids", "label": "条件面 ids (逗号分隔)", "type": "text"},
      {"key": "num_condition_faces", "flag": "--num-condition-faces", "label": "随机条件面数", "type": "number"},
      {"key": "condition_mode", "flag": "--condition-mode", "label": "条件面选取", "type": "select", "choices": ["random", "constraint"]},
      {"key": "abc_split", "flag": "--abc-split", "label": "ABC split", "type": "select", "choices": ["train", "val", "test"]},
      {"key": "data_dir", "flag": "--data-dir", "label": "ABC-1M 根目录", "type": "text"}
    ]
  }
}
JSON
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

    # Autocomplete defaults to ABC-1M when data-dir omitted
    if [[ -z "${DATA_DIR_ARG}" ]]; then
      if [[ "${AUTOCOMPLETE}" == "1" || "${AUTOCOMPLETE}" == "true" || -n "${CONDITION_JSON}" || -n "${ABC_STEM}" ]]; then
        DATA_DIR_ARG="/data/hdd/datasets/ABC-1M"
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
      --autocomplete "${AUTOCOMPLETE}"
      --condition-mode "${CONDITION_MODE}"
      --abc-split "${ABC_SPLIT}"
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
    if [[ -n "${CONDITION_JSON}" ]]; then
      INFER_ARGS+=(--condition-json "${CONDITION_JSON}")
    fi
    if [[ -n "${ABC_STEM}" ]]; then
      INFER_ARGS+=(--abc-stem "${ABC_STEM}")
    fi
    if [[ -n "${FACE_IDS}" ]]; then
      INFER_ARGS+=(--face-ids "${FACE_IDS}")
    fi
    if [[ -n "${NUM_CONDITION_FACES}" ]]; then
      INFER_ARGS+=(--num-condition-faces "${NUM_CONDITION_FACES}")
    fi
    echo "[run.sh] infer weight_folder=${WEIGHT_FOLDER} autocomplete=${AUTOCOMPLETE} → ${OUTPUT_DIR_ARG}/infer" >&2
    exec python -u "${REPO_DIR}/scripts/infer_pipeline.py" "${INFER_ARGS[@]}"
    ;;

  *)
    echo "[run.sh] unknown mode: ${MODE} (supported: capabilities, infer)" >&2
    exit 1
    ;;
esac
