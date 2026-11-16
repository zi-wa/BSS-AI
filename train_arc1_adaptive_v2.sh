#!/usr/bin/env bash
# ARC-1 training with the z keep/cut head, version 2: trm_diffusion_adaptive_v2 (models/recursive_reasoning/trm_diffusion_adaptive_v2.py).
# vs the first version (trm_diffusion_adaptive.py): the cut/keep decision is made at the start of a step from the carried z_H
# (no 50% random cuts at init: z_head weights 0, cut bias -5 -> starts as "always keep"), one extra no-grad forward per step
# (the branch not taken) for the label instead of two, and the z loss only counts samples that continue a puzzle
# (models/losses.py: z_cut_valid, train/z_cut_rate, train/z_label_cut_rate).
# Hyperparameters = the standard ARC-1 canvas run trm-uniform-diffusion-arc-1-bs768 (checkpoints/.../all_config.yaml),
# written out explicitly so later default changes in config/ cannot change this run. Only the model module differs.
#
# Usage: bash train_arc1_adaptive_v2.sh [--gpus 0,1,2,3] [--run-name NAME] [--skip-build] [-- extra hydra overrides]
#   --gpus       CUDA device ids (default 0,1,2,3). global batch stays 768 whatever the GPU count.
#   --run-name   default trm-adaptive-v2-arc-1-bs768 -> checkpoints/<run-name>
#   --skip-build do not build data/arc1concept-aug-1000 if it is missing
# Env: Z_LABEL_PROB (fraction of steps with the cut/keep label, default 1.0; 0.5 = ~1.3x instead of ~1.6x step time),
#      CONDA_ENV_BIN (default ./env/dr_trm/bin if present, else torchrun from PATH / the active env), CKPT_ROOT (default ./checkpoints), WANDB_MODE (default offline)
set -euo pipefail
cd "$(dirname "$0")"

GPU_IDS=0,1,2,3; RUN_NAME=trm-adaptive-v2-arc-1-bs768; SKIP_BUILD=0; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPU_IDS="$2"; shift 2;;
    --run-name) RUN_NAME="$2"; shift 2;;
    --skip-build) SKIP_BUILD=1; shift;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) sed -n 2,15p "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

# python env: ./env/dr_trm if present (packed env of the share copy), CONDA_ENV_BIN=<env>/bin, or the active env
CONDA_ENV_BIN=${CONDA_ENV_BIN:-$PWD/env/dr_trm/bin}
if [[ -x $CONDA_ENV_BIN/torchrun ]]; then export PATH=$CONDA_ENV_BIN:$PATH
else echo "warning: $CONDA_ENV_BIN/torchrun not found, using torchrun from PATH: $(command -v torchrun)" >&2; fi

DATA_DIR=data/arc1concept-aug-1000
if [[ $SKIP_BUILD == 0 && ! -d $DATA_DIR ]]; then
  echo ">> building dataset -> $DATA_DIR"
  python -m dataset.build_arc_dataset --input-file-prefix kaggle/combined/arc-agi --output-dir "$DATA_DIR" \
    --subsets training evaluation concept --test-set-name evaluation
fi

export CUDA_VISIBLE_DEVICES=$GPU_IDS
NGPU=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=${WANDB_DIR:-$PWD}
CKPT_ROOT=${CKPT_ROOT:-$PWD/checkpoints}
echo ">> run=$RUN_NAME arch=trm_diffusion_adaptive_v2 z_label_prob=${Z_LABEL_PROB:-1.0} gpus=$GPU_IDS ($NGPU)"

exec torchrun --nproc-per-node "$NGPU" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm_diffusion_adaptive_v2 arch.z_label_prob=${Z_LABEL_PROB:-1.0} \
  data_paths="[$DATA_DIR]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  arch.hidden_size=512 arch.num_heads=8 arch.expansion=4 \
  arch.puzzle_emb_ndim=512 arch.puzzle_emb_len=16 \
  arch.halt_max_steps=16 arch.halt_exploration_prob=0.1 arch.forward_dtype=bfloat16 \
  global_batch_size=768 epochs=100000 eval_interval=10000 \
  lr=1e-4 lr_min_ratio=1.0 lr_warmup_steps=2000 beta1=0.9 beta2=0.95 \
  weight_decay=0.1 puzzle_emb_lr=1e-2 puzzle_emb_weight_decay=0.1 \
  ema=True ema_rate=0.999 seed=0 \
  +run_name="$RUN_NAME" +checkpoint_path="$CKPT_ROOT/$RUN_NAME" \
  ${EXTRA[@]+"${EXTRA[@]}"}
