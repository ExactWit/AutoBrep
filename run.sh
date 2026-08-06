#!/usr/bin/env bash
# exp_launcher entry — AutoBrep (PC-cond + ECCV view-cond SFT).
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
SPLIT_ARG=""
DATASET_ARG=""
TASK_ARG=""
CHECKPOINT_ARG=""
WEIGHT_FOLDER="${WEIGHT_FOLDER:-/data/hdd/outputs/AutoBrep}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_BATCHES="${NUM_BATCHES:-10}"
COMPLEXITY="${COMPLEXITY:-from_condition}"
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
MAX_STEPS="${MAX_STEPS:-10000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-100}"
LR="${LR:-0.0001}"
PC_NUM_POINTS="${PC_NUM_POINTS:-2048}"
PC_NUM_LATENTS="${PC_NUM_LATENTS:-64}"
VIEW_NUM_LATENTS="${VIEW_NUM_LATENTS:-64}"
ACCUM_GRAD="${ACCUM_GRAD:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PC_CONDITIONED="${PC_CONDITIONED:-0}"
VIEW_CONDITIONED="${VIEW_CONDITIONED:-0}"
POINT_CLOUD="${POINT_CLOUD:-}"
SAMPLE_ID_ARG=""
INFER_SPLIT_NAME="${INFER_SPLIT_NAME:-val}"
MAX_FACE="${MAX_FACE:-200}"
SHARD_SIZE="${SHARD_SIZE:-100}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
# legacy (accepted but ignored by train_*_pipeline)
MAX_EPOCHS="${MAX_EPOCHS:--1}"
LIMIT_TRAIN="${LIMIT_TRAIN:--1}"
LIMIT_VAL="${LIMIT_VAL:--1}"
OFFICIAL_VAL_SAMPLES="${OFFICIAL_VAL_SAMPLES:--1}"
OFFICIAL_VAL_SAMPLES_MID="${OFFICIAL_VAL_SAMPLES_MID:-24}"
OFFICIAL_VAL_GEN_BATCH="${OFFICIAL_VAL_GEN_BATCH:-1}"
OFFICIAL_VAL_EVERY="${OFFICIAL_VAL_EVERY:-0}"
OFFICIAL_VAL_EPOCH_FRAC="${OFFICIAL_VAL_EPOCH_FRAC:-0.25}"
NO_OFFICIAL_VAL="${NO_OFFICIAL_VAL:-0}"
EVAL_PY="${EVAL_PY:-/data/hdd/datasets/eccv2026ws-cad-data/examples/min_eval/eval.py}"
EVAL_GEN_BATCH="${EVAL_GEN_BATCH:-1}"
GEN_RETRIES="${GEN_RETRIES:-4}"
GEN_RERANK="${GEN_RERANK:-0}"
MAKE_SUBMISSION_ZIP="${MAKE_SUBMISSION_ZIP:-0}"
# ECCV view-cond default: epoch schedule on small set (override with --max-steps)
ECCV_MAX_EPOCHS="${ECCV_MAX_EPOCHS:-50}"
USE_PRIM_SEQ_ENCODER="${USE_PRIM_SEQ_ENCODER:-}"
PRIM_D_MODEL="${PRIM_D_MODEL:-512}"
PRIM_N_LAYERS="${PRIM_N_LAYERS:-0}"
PRIM_MAX_SEQ="${PRIM_MAX_SEQ:-384}"
PRIM_PREFIX_MODE="${PRIM_PREFIX_MODE:-prefix_lm}"
USE_TOPO_SKETCH="${USE_TOPO_SKETCH:-0}"
TOPO_SKETCH_MAX="${TOPO_SKETCH_MAX:-64}"
TOPO_COUNT_WEIGHT="${TOPO_COUNT_WEIGHT:-0}"
COND_DROPOUT="${COND_DROPOUT:-0.1}"
UNFREEZE_DECODER_LAYERS="${UNFREEZE_DECODER_LAYERS:-0}"
USE_DECODER_CROSS_ATTN="${USE_DECODER_CROSS_ATTN:-}"
DECODER_XATTN_HEADS="${DECODER_XATTN_HEADS:-8}"
ENABLE_AUX_VIEW_BBOX="${ENABLE_AUX_VIEW_BBOX:-0}"
AUX_VIEW_BBOX_WEIGHT="${AUX_VIEW_BBOX_WEIGHT:-0.1}"
ENABLE_AUX_SURF_TYPE="${ENABLE_AUX_SURF_TYPE:-0}"
AUX_SURF_TYPE_WEIGHT="${AUX_SURF_TYPE_WEIGHT:-0.1}"
FSQ_UPGRADE="${FSQ_UPGRADE:-0}"
POSTPROCESS_ANALYTIC="${POSTPROCESS_ANALYTIC:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-dir) EXP_DIR_ARG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR_ARG="$2"; shift 2 ;;
    --data-dir|--dataset-dir) DATA_DIR_ARG="$2"; shift 2 ;;
    --datasplit) DATASPLIT_ARG="$2"; shift 2 ;;
    --split) SPLIT_ARG="$2"; shift 2 ;;
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
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --val-check-interval) VAL_CHECK_INTERVAL="$2"; shift 2 ;;
    --limit-val-batches) LIMIT_VAL_BATCHES="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --pc-num-points) PC_NUM_POINTS="$2"; shift 2 ;;
    --pc-num-latents) PC_NUM_LATENTS="$2"; shift 2 ;;
    --view-num-latents) VIEW_NUM_LATENTS="$2"; shift 2 ;;
    --accumulate-grad-batches) ACCUM_GRAD="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --pc-conditioned) PC_CONDITIONED="$2"; shift 2 ;;
    --view-conditioned) VIEW_CONDITIONED="$2"; shift 2 ;;
    --point-cloud) POINT_CLOUD="$2"; shift 2 ;;
    --sample-id|--index) SAMPLE_ID_ARG="$2"; shift 2 ;;
    --infer-split-name) INFER_SPLIT_NAME="$2"; shift 2 ;;
    --max-face) MAX_FACE="$2"; shift 2 ;;
    --shard-size) SHARD_SIZE="$2"; shift 2 ;;
    --limit-samples) LIMIT_SAMPLES="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --limit-train) LIMIT_TRAIN="$2"; shift 2 ;;
    --limit-val) LIMIT_VAL="$2"; shift 2 ;;
    --official-val-samples) OFFICIAL_VAL_SAMPLES="$2"; shift 2 ;;
    --official-val-samples-mid) OFFICIAL_VAL_SAMPLES_MID="$2"; shift 2 ;;
    --official-val-gen-batch) OFFICIAL_VAL_GEN_BATCH="$2"; shift 2 ;;
    --official-val-every) OFFICIAL_VAL_EVERY="$2"; shift 2 ;;
    --official-val-epoch-frac) OFFICIAL_VAL_EPOCH_FRAC="$2"; shift 2 ;;
    --no-official-val) NO_OFFICIAL_VAL="$2"; shift 2 ;;
    --eval-py) EVAL_PY="$2"; shift 2 ;;
    --gen-batch|--eval-gen-batch) EVAL_GEN_BATCH="$2"; shift 2 ;;
    --gen-retries) GEN_RETRIES="$2"; shift 2 ;;
    --gen-rerank) GEN_RERANK="$2"; shift 2 ;;
    --make-submission-zip) MAKE_SUBMISSION_ZIP="$2"; shift 2 ;;
    --use-prim-seq-encoder) USE_PRIM_SEQ_ENCODER="$2"; shift 2 ;;
    --prim-d-model) PRIM_D_MODEL="$2"; shift 2 ;;
    --prim-n-layers) PRIM_N_LAYERS="$2"; shift 2 ;;
    --prim-max-seq) PRIM_MAX_SEQ="$2"; shift 2 ;;
    --prim-prefix-mode) PRIM_PREFIX_MODE="$2"; shift 2 ;;
    --use-topo-sketch) USE_TOPO_SKETCH="$2"; shift 2 ;;
    --topo-sketch-max) TOPO_SKETCH_MAX="$2"; shift 2 ;;
    --topo-count-weight) TOPO_COUNT_WEIGHT="$2"; shift 2 ;;
    --cond-dropout) COND_DROPOUT="$2"; shift 2 ;;
    --unfreeze-decoder-layers) UNFREEZE_DECODER_LAYERS="$2"; shift 2 ;;
    --use-decoder-cross-attn) USE_DECODER_CROSS_ATTN="$2"; shift 2 ;;
    --decoder-xattn-heads) DECODER_XATTN_HEADS="$2"; shift 2 ;;
    --enable-aux-view-bbox) ENABLE_AUX_VIEW_BBOX="$2"; shift 2 ;;
    --aux-view-bbox-weight) AUX_VIEW_BBOX_WEIGHT="$2"; shift 2 ;;
    --enable-aux-surf-type) ENABLE_AUX_SURF_TYPE="$2"; shift 2 ;;
    --aux-surf-type-weight) AUX_SURF_TYPE_WEIGHT="$2"; shift 2 ;;
    --fsq-upgrade) FSQ_UPGRADE="$2"; shift 2 ;;
    --postprocess-analytic) POSTPROCESS_ANALYTIC="$2"; shift 2 ;;
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
  "modes": ["capabilities", "preprocess", "train", "test", "infer", "public_infer"],
  "launch_modes": ["preprocess", "train", "resume", "test", "infer", "public_infer"],
  "datasets": ["abc", "abc-1m", "eccv2026ws-cad-data"],
  "tasks": {"abc": ["gen"], "abc-1m": ["gen"], "eccv2026ws-cad-data": ["gen", "cad"]},
  "env_file": "environment.yml",
  "checkpoints": {
    "ar": "ar.ckpt",
    "surf_fsq": "surf-fsq.ckpt",
    "edge_fsq": "edge-fsq.ckpt",
    "pc_cond": "last.ckpt",
    "eccv_view": "last.ckpt"
  },
  "infer_requires_sample": false,
  "defaults": {
    "preprocess": {
      "data_dir": "/data/hdd/datasets/eccv2026ws-cad-data",
      "max_face": 200,
      "shard_size": 100,
      "limit_samples": 0,
      "num_workers": 4
    },
    "train": {
      "weight_folder": "/data/hdd/outputs/AutoBrep",
      "data_dir": "/data/hdd/datasets/eccv2026ws-cad-data",
      "batch_size": 2,
      "max_epochs": 50,
      "max_steps": -1,
      "val_check_interval": 500,
      "limit_val_batches": 100,
      "lr": 0.0001,
      "view_num_latents": 64,
      "accumulate_grad_batches": 2,
      "num_workers": 2,
      "official_val_samples": -1,
      "official_val_samples_mid": 24,
      "official_val_gen_batch": 1,
      "official_val_every": 0,
      "official_val_epoch_frac": 0.25,
      "no_official_val": false,
      "complexity": "from_condition",
      "use_prim_seq_encoder": 1,
      "prim_n_layers": 0,
      "prim_max_seq": 384,
      "prim_prefix_mode": "prefix_lm",
      "use_topo_sketch": 0,
      "topo_sketch_max": 64,
      "topo_count_weight": 0,
      "cond_dropout": 0.1,
      "unfreeze_decoder_layers": 0,
      "use_decoder_cross_attn": 0,
      "enable_aux_view_bbox": 0,
      "enable_aux_surf_type": 0,
      "fsq_upgrade": 0
    },
    "test": {
      "weight_folder": "/data/hdd/outputs/AutoBrep",
      "data_dir": "/data/hdd/datasets/eccv2026ws-cad-data",
      "split": "test",
      "complexity": "from_condition",
      "temperature": 1.0,
      "top_p": 0.9,
      "gen_batch": 1,
      "gen_retries": 4,
      "gen_rerank": 0,
      "postprocess_analytic": 1
    },
    "infer": {
      "weight_folder": "/data/hdd/outputs/AutoBrep",
      "data_dir": "/data/hdd/datasets/eccv2026ws-cad-data",
      "batch_size": 1,
      "num_batches": 10,
      "complexity": "from_condition",
      "temperature": 1.0,
      "top_p": 0.9,
      "vertex_threshold": 0.002,
      "sewing_tolerance": 0.002,
      "z_threshold": 0,
      "seed": 689447,
      "use_seed": false,
      "debug": true,
      "pc_conditioned": false,
      "view_conditioned": false,
      "infer_split_name": "val"
    },
    "public_infer": {
      "weight_folder": "/data/hdd/outputs/AutoBrep",
      "data_dir": "/data/hdd/datasets/eccv2026ws-cad-data",
      "complexity": "from_condition",
      "temperature": 1.0,
      "top_p": 0.9,
      "gen_batch": 1,
      "gen_retries": 4,
      "gen_rerank": 0,
      "make_submission_zip": true
    }
  },
  "args": {
    "preprocess": [
      "--data-dir",
      "--max-face",
      "--shard-size",
      "--limit-samples",
      "--num-workers",
      "--datasplit"
    ],
    "train": [
      "--weight-folder",
      "--data-dir",
      "--batch-size",
      "--max-epochs",
      "--max-steps",
      "--val-check-interval",
      "--limit-val-batches",
      "--lr",
      "--view-num-latents",
      "--accumulate-grad-batches",
      "--num-workers",
      "--gpu",
      "--resume-from",
      "--official-val-samples",
      "--official-val-samples-mid",
      "--official-val-gen-batch",
      "--official-val-every",
      "--official-val-epoch-frac",
      "--no-official-val",
      "--eval-py",
      "--complexity",
      "--use-prim-seq-encoder",
      "--prim-d-model",
      "--prim-n-layers",
      "--prim-max-seq",
      "--prim-prefix-mode",
      "--use-topo-sketch",
      "--topo-sketch-max",
      "--topo-count-weight",
      "--cond-dropout",
      "--unfreeze-decoder-layers",
      "--use-decoder-cross-attn",
      "--decoder-xattn-heads",
      "--enable-aux-view-bbox",
      "--aux-view-bbox-weight",
      "--enable-aux-surf-type",
      "--aux-surf-type-weight",
      "--fsq-upgrade"
    ],
    "test": [
      "--weight-folder",
      "--data-dir",
      "--datasplit",
      "--split",
      "--checkpoint",
      "--gpu",
      "--complexity",
      "--temperature",
      "--top-p",
      "--gen-batch",
      "--gen-retries",
      "--gen-rerank",
      "--eval-py",
      "--postprocess-analytic"
    ],
    "infer": [
      "--weight-folder",
      "--data-dir",
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
      "--view-conditioned",
      "--point-cloud",
      "--sample-id",
      "--infer-split-name",
      "--datasplit"
    ],
    "public_infer": [
      "--weight-folder",
      "--data-dir",
      "--datasplit",
      "--checkpoint",
      "--gpu",
      "--complexity",
      "--temperature",
      "--top-p",
      "--gen-batch",
      "--gen-retries",
      "--gen-rerank",
      "--make-submission-zip"
    ]
  },
  "arg_fields": {
    "preprocess": [
      {"key": "data_dir", "flag": "--data-dir", "label": "ECCV 数据集根目录", "type": "text"},
      {"key": "max_face", "flag": "--max-face", "label": "最大面数", "type": "number"},
      {"key": "shard_size", "flag": "--shard-size", "label": "parquet shard 行数", "type": "number"},
      {"key": "limit_samples", "flag": "--limit-samples", "label": "限制样本数(0=全部)", "type": "number"},
      {"key": "num_workers", "flag": "--num-workers", "label": "并行进程数", "type": "number"}
    ],
    "train": [
      {"key": "weight_folder", "flag": "--weight-folder", "label": "预训练权重目录", "type": "text"},
      {"key": "data_dir", "flag": "--data-dir", "label": "数据根目录", "type": "text"},
      {"key": "batch_size", "flag": "--batch-size", "label": "batch_size", "type": "number"},
      {"key": "max_epochs", "flag": "--max-epochs", "label": "训练 epoch 数（小数据集重复遍历）", "type": "number"},
      {"key": "max_steps", "flag": "--max-steps", "label": "max_steps（-1=按 epoch；仅当 max_epochs≤0）", "type": "number"},
      {"key": "val_check_interval", "flag": "--val-check-interval", "label": "step 模式下每 N batch 验证", "type": "number"},
      {"key": "limit_val_batches", "flag": "--limit-val-batches", "label": "每次验证 batch 上限", "type": "number"},
      {"key": "lr", "flag": "--lr", "label": "learning rate", "type": "number"},
      {"key": "view_num_latents", "flag": "--view-num-latents", "label": "视图条件 token 数", "type": "number"},
      {"key": "accumulate_grad_batches", "flag": "--accumulate-grad-batches", "label": "梯度累积", "type": "number"},
      {"key": "num_workers", "flag": "--num-workers", "label": "num_workers", "type": "number"},
      {"key": "official_val_samples", "flag": "--official-val-samples", "label": "末 epoch 全量 STEP 数(≤0=全~694)", "type": "number"},
      {"key": "official_val_samples_mid", "flag": "--official-val-samples-mid", "label": "中途 STEP 固定子集大小", "type": "number"},
      {"key": "official_val_gen_batch", "flag": "--official-val-gen-batch", "label": "STEP AR 批大小(吃显存)", "type": "number"},
      {"key": "official_val_every", "flag": "--official-val-every", "label": "STEP 每 N epoch（0=用 frac 里程碑）", "type": "number"},
      {"key": "official_val_epoch_frac", "flag": "--official-val-epoch-frac", "label": "STEP 里程碑比例(0.25→25/50/75/100%)", "type": "number"},
      {"key": "no_official_val", "flag": "--no-official-val", "label": "关闭官方 STEP 评测", "type": "bool"},
      {"key": "complexity", "flag": "--complexity", "label": "复杂度 token（训练仍用 GT 面数；推理/官方val）", "type": "select", "choices": ["from_condition", "easy", "medium", "hard", "random"]}
    ],
    "infer": [
      {"key": "weight_folder", "flag": "--weight-folder", "label": "权重目录", "type": "text"},
      {"key": "data_dir", "flag": "--data-dir", "label": "数据根目录", "type": "text"},
      {"key": "batch_size", "flag": "--batch-size", "label": "batch_size", "type": "number"},
      {"key": "num_batches", "flag": "--num-batches", "label": "采样批次数", "type": "number"},
      {"key": "complexity", "flag": "--complexity", "label": "复杂度", "type": "select", "choices": ["from_condition", "random", "easy", "medium", "hard"]},
      {"key": "temperature", "flag": "--temperature", "label": "temperature", "type": "number"},
      {"key": "top_p", "flag": "--top-p", "label": "top_p", "type": "number"},
      {"key": "vertex_threshold", "flag": "--vertex-threshold", "label": "vertex_threshold", "type": "number"},
      {"key": "sewing_tolerance", "flag": "--sewing-tolerance", "label": "sewing_tolerance", "type": "number"},
      {"key": "z_threshold", "flag": "--z-threshold", "label": "z_threshold", "type": "number"},
      {"key": "seed", "flag": "--seed", "label": "seed", "type": "number"},
      {"key": "use_seed", "flag": "--use-seed", "label": "固定随机种子", "type": "bool"},
      {"key": "debug", "flag": "--debug", "label": "debug 图/点云", "type": "bool"},
      {"key": "pc_conditioned", "flag": "--pc-conditioned", "label": "点云条件推理", "type": "bool"},
      {"key": "view_conditioned", "flag": "--view-conditioned", "label": "多视图条件推理", "type": "bool"},
      {"key": "point_cloud", "flag": "--point-cloud", "label": "点云 .npy (N,3)", "type": "text"},
      {"key": "sample_id", "flag": "--sample-id", "label": "ECCV sample id", "type": "text"},
      {"key": "infer_split_name", "flag": "--infer-split-name", "label": "infer split", "type": "select", "choices": ["train", "val", "test", "public_test"]}
    ],
    "test": [
      {"key": "weight_folder", "flag": "--weight-folder", "label": "预训练权重目录", "type": "text"},
      {"key": "data_dir", "flag": "--data-dir", "label": "数据根目录", "type": "text"},
      {"key": "complexity", "flag": "--complexity", "label": "复杂度", "type": "select", "choices": ["from_condition", "easy", "medium", "hard", "random"]},
      {"key": "temperature", "flag": "--temperature", "label": "temperature", "type": "number"},
      {"key": "top_p", "flag": "--top-p", "label": "top_p", "type": "number"},
      {"key": "gen_batch", "flag": "--gen-batch", "label": "STEP AR 批大小", "type": "number"}
    ],
    "public_infer": [
      {"key": "weight_folder", "flag": "--weight-folder", "label": "预训练权重目录", "type": "text"},
      {"key": "data_dir", "flag": "--data-dir", "label": "数据根目录", "type": "text"},
      {"key": "complexity", "flag": "--complexity", "label": "复杂度", "type": "select", "choices": ["from_condition", "easy", "medium", "hard", "random"]},
      {"key": "temperature", "flag": "--temperature", "label": "temperature", "type": "number"},
      {"key": "top_p", "flag": "--top-p", "label": "top_p", "type": "number"},
      {"key": "gen_batch", "flag": "--gen-batch", "label": "STEP AR 批大小", "type": "number"},
      {"key": "make_submission_zip", "flag": "--make-submission-zip", "label": "打包 submission.zip", "type": "bool"}
    ]
  }
}
JSON
    ;;

  preprocess)
    activate_env
    DATA_ROOT="${DATA_DIR_ARG:-/data/hdd/datasets/eccv2026ws-cad-data}"
    PRE_ARGS=(
      --data-dir "${DATA_ROOT}"
      --max-face "${MAX_FACE}"
      --shard-size "${SHARD_SIZE}"
      --num-workers "${NUM_WORKERS}"
    )
    if [[ -n "${DATASPLIT_ARG}" ]]; then
      PRE_ARGS+=(--datasplit "${DATASPLIT_ARG}")
    fi
    if [[ "${LIMIT_SAMPLES}" != "0" ]]; then
      PRE_ARGS+=(--limit-samples "${LIMIT_SAMPLES}")
    fi
    echo "[run.sh] preprocess data=${DATA_ROOT}" >&2
    exec python -u "${REPO_DIR}/scripts/preprocess_eccv_autobrep.py" "${PRE_ARGS[@]}"
    ;;

  train)
    activate_env
    if [[ -z "${EXP_DIR_ARG}" ]]; then
      echo "[run.sh] ERROR: --exp-dir is required for train" >&2
      exit 1
    fi
    DATASET="${DATASET_ARG:-eccv2026ws-cad-data}"
    mkdir -p "${EXP_DIR_ARG}/checkpoints" "${EXP_DIR_ARG}/metrics" "${EXP_DIR_ARG}/tensorboard"

    if [[ "${DATASET}" == "eccv2026ws-cad-data" || "${DATASET}" == "eccv2026ws-cad" || "${DATASET}" == "eccv" ]]; then
      DATA_ROOT="${DATA_DIR_ARG:-/data/hdd/datasets/eccv2026ws-cad-data}"
      # Prefer epoch schedule for small ECCV set (repeat parquet passes).
      ECCV_EPOCHS="${MAX_EPOCHS}"
      if [[ "${ECCV_EPOCHS}" == "-1" || -z "${ECCV_EPOCHS}" ]]; then
        ECCV_EPOCHS="${ECCV_MAX_EPOCHS}"
      fi
      TRAIN_ARGS=(
        --exp-dir "${EXP_DIR_ARG}"
        --data-dir "${DATA_ROOT}"
        --dataset "${DATASET}"
        --task "${TASK_ARG:-gen}"
        --weight-folder "${WEIGHT_FOLDER}"
        --gpu "${GPU}"
        --batch-size "${BATCH_SIZE}"
        --max-epochs "${ECCV_EPOCHS}"
        --max-steps "${MAX_STEPS}"
        --val-check-interval "${VAL_CHECK_INTERVAL}"
        --limit-val-batches "${LIMIT_VAL_BATCHES}"
        --lr "${LR}"
        --view-num-latents "${VIEW_NUM_LATENTS}"
        --accumulate-grad-batches "${ACCUM_GRAD}"
        --num-workers "${NUM_WORKERS}"
        --official-val-samples "${OFFICIAL_VAL_SAMPLES}"
        --official-val-samples-mid "${OFFICIAL_VAL_SAMPLES_MID}"
        --official-val-gen-batch "${OFFICIAL_VAL_GEN_BATCH}"
        --official-val-every "${OFFICIAL_VAL_EVERY}"
        --official-val-epoch-frac "${OFFICIAL_VAL_EPOCH_FRAC}"
        --eval-py "${EVAL_PY}"
        --complexity "${COMPLEXITY}"
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
      if [[ "${NO_OFFICIAL_VAL}" == "1" || "${NO_OFFICIAL_VAL}" == "true" || "${NO_OFFICIAL_VAL}" == "True" ]]; then
        TRAIN_ARGS+=(--no-official-val)
      fi
      # P1/P2 flags: only forward when explicitly enabled (1).
      # Launcher may inject capability defaults of 0; older tips lack those argparse opts.
      if [[ "${USE_PRIM_SEQ_ENCODER}" == "1" ]]; then
        TRAIN_ARGS+=(
          --use-prim-seq-encoder 1
          --prim-d-model "${PRIM_D_MODEL}"
          --prim-n-layers "${PRIM_N_LAYERS}"
          --prim-max-seq "${PRIM_MAX_SEQ}"
          --prim-prefix-mode "${PRIM_PREFIX_MODE}"
        )
        if [[ "${USE_TOPO_SKETCH}" == "1" ]]; then
          TRAIN_ARGS+=(
            --use-topo-sketch 1
            --topo-sketch-max "${TOPO_SKETCH_MAX}"
            --topo-count-weight "${TOPO_COUNT_WEIGHT}"
          )
        fi
        TRAIN_ARGS+=(--cond-dropout "${COND_DROPOUT}")
        TRAIN_ARGS+=(--unfreeze-decoder-layers "${UNFREEZE_DECODER_LAYERS}")
      fi
      if [[ "${USE_DECODER_CROSS_ATTN}" == "1" ]]; then
        TRAIN_ARGS+=(
          --use-decoder-cross-attn 1
          --decoder-xattn-heads "${DECODER_XATTN_HEADS}"
        )
      fi
      if [[ "${ENABLE_AUX_VIEW_BBOX}" == "1" ]]; then
        TRAIN_ARGS+=(--enable-aux-view-bbox 1 --aux-view-bbox-weight "${AUX_VIEW_BBOX_WEIGHT}")
      fi
      if [[ "${ENABLE_AUX_SURF_TYPE}" == "1" ]]; then
        TRAIN_ARGS+=(--enable-aux-surf-type 1 --aux-surf-type-weight "${AUX_SURF_TYPE_WEIGHT}")
      fi
      if [[ "${FSQ_UPGRADE}" == "1" ]]; then
        TRAIN_ARGS+=(--fsq-upgrade 1)
      fi
      echo "[run.sh] train eccv data=${DATA_ROOT} weight=${WEIGHT_FOLDER} → ${EXP_DIR_ARG}" >&2
      exec python -u "${REPO_DIR}/scripts/train_eccv_pipeline.py" "${TRAIN_ARGS[@]}"
    fi

    DATA_ROOT="${DATA_DIR_ARG:-/data/hdd/datasets/ABC-1M}"
    TRAIN_ARGS=(
      --exp-dir "${EXP_DIR_ARG}"
      --data-dir "${DATA_ROOT}"
      --dataset "${DATASET}"
      --task "${TASK_ARG:-gen}"
      --weight-folder "${WEIGHT_FOLDER}"
      --gpu "${GPU}"
      --batch-size "${BATCH_SIZE}"
      --max-steps "${MAX_STEPS}"
      --val-check-interval "${VAL_CHECK_INTERVAL}"
      --limit-val-batches "${LIMIT_VAL_BATCHES}"
      --lr "${LR}"
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
    echo "[run.sh] train pc data=${DATA_ROOT} weight=${WEIGHT_FOLDER} → ${EXP_DIR_ARG}" >&2
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
      --view-conditioned "${VIEW_CONDITIONED}"
      --infer-split-name "${INFER_SPLIT_NAME}"
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
    if [[ -n "${SAMPLE_ID_ARG}" ]]; then
      INFER_ARGS+=(--sample-id "${SAMPLE_ID_ARG}")
    fi
    echo "[run.sh] infer weight_folder=${WEIGHT_FOLDER} pc=${PC_CONDITIONED} view=${VIEW_CONDITIONED} → ${OUTPUT_DIR_ARG}/infer" >&2
    exec python -u "${REPO_DIR}/scripts/infer_pipeline.py" "${INFER_ARGS[@]}"
    ;;

  test)
    activate_env
    if [[ -z "${EXP_DIR_ARG}" ]]; then
      echo "[run.sh] ERROR: --exp-dir is required for test" >&2
      exit 1
    fi
    DATA_ROOT="${DATA_DIR_ARG:-/data/hdd/datasets/eccv2026ws-cad-data}"
    OUT="${OUTPUT_DIR_ARG:-${EXP_DIR_ARG}}"
    mkdir -p "${EXP_DIR_ARG}/metrics" "${OUT}"
    EVAL_ARGS=(
      --exp-dir "${EXP_DIR_ARG}"
      --output-dir "${OUT}"
      --data-dir "${DATA_ROOT}"
      --weight-folder "${WEIGHT_FOLDER}"
      --gpu "${GPU}"
      --split "${SPLIT_ARG:-test}"
      --complexity "${COMPLEXITY}"
      --temperature "${TEMPERATURE}"
      --top-p "${TOP_P}"
      --gen-batch "${EVAL_GEN_BATCH}"
      --gen-retries "${GEN_RETRIES:-1}"
      --gen-rerank "${GEN_RERANK:-0}"
      --eval-py "${EVAL_PY}"
    )
    if [[ -n "${DATASPLIT_ARG}" ]]; then
      EVAL_ARGS+=(--datasplit "${DATASPLIT_ARG}")
    fi
    if [[ -n "${CHECKPOINT_ARG}" ]]; then
      EVAL_ARGS+=(--checkpoint "${CHECKPOINT_ARG}")
    fi
    echo "[run.sh] test official split → ${EXP_DIR_ARG}/metrics/test.json" >&2
    exec python -u "${REPO_DIR}/scripts/eval_eccv_split.py" "${EVAL_ARGS[@]}"
    ;;

  public_infer)
    activate_env
    if [[ -z "${EXP_DIR_ARG}" ]]; then
      echo "[run.sh] ERROR: --exp-dir is required for public_infer" >&2
      exit 1
    fi
    if [[ -z "${OUTPUT_DIR_ARG}" ]]; then
      echo "[run.sh] ERROR: --output-dir is required for public_infer" >&2
      exit 1
    fi
    DATA_ROOT="${DATA_DIR_ARG:-/data/hdd/datasets/eccv2026ws-cad-data}"
    mkdir -p "${EXP_DIR_ARG}/metrics" "${OUTPUT_DIR_ARG}/predictions"
    EVAL_ARGS=(
      --exp-dir "${EXP_DIR_ARG}"
      --output-dir "${OUTPUT_DIR_ARG}"
      --data-dir "${DATA_ROOT}"
      --weight-folder "${WEIGHT_FOLDER}"
      --gpu "${GPU}"
      --split public_test
      --complexity "${COMPLEXITY}"
      --temperature "${TEMPERATURE}"
      --top-p "${TOP_P}"
      --gen-batch "${EVAL_GEN_BATCH}"
      --gen-retries "${GEN_RETRIES:-1}"
      --gen-rerank "${GEN_RERANK:-0}"
      --make-submission-zip 1
    )
    if [[ -n "${DATASPLIT_ARG}" ]]; then
      EVAL_ARGS+=(--datasplit "${DATASPLIT_ARG}")
    fi
    if [[ -n "${CHECKPOINT_ARG}" ]]; then
      EVAL_ARGS+=(--checkpoint "${CHECKPOINT_ARG}")
    fi
    echo "[run.sh] public_infer → ${OUTPUT_DIR_ARG}/submission.zip" >&2
    exec python -u "${REPO_DIR}/scripts/eval_eccv_split.py" "${EVAL_ARGS[@]}"
    ;;

  *)
    echo "[run.sh] unknown mode: ${MODE} (supported: capabilities, preprocess, train, test, infer, public_infer)" >&2
    exit 1
    ;;
esac
