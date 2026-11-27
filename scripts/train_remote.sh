#!/usr/bin/env bash
# Train one policy of the study on a fresh cloud GPU instance, upload its checkpoints, shut down.
#
# On the instance (Ubuntu with an NVIDIA driver, e.g. AWS "Deep Learning Base OSS Nvidia Driver
# GPU AMI (Ubuntu 22.04)"), inside tmux so a dropped SSH connection does not stop it:
#     curl -sLO https://raw.githubusercontent.com/goelankur1996/who-moves-the-arm/main/scripts/train_remote.sh
#     HF_TOKEN=hf_... bash train_remote.sh B
#
# Runs (recipe from the study plan; B and C must stay identical apart from the dataset):
#   B      SmolVLA from lerobot/smolvla_base on ankurcw/wmta_demos_v2, batch 64, 20k steps
#   C      the same, on ankurcw/wmta_demos_v2_marked (the demos with arm C's rings)
#   C-ACT  ACT on ankurcw/wmta_demos_v2_marked, no language input, batch 8, 30k steps, mixed precision
# C and C-ACT can run at the same time on one instance, each in its own tmux window: start the
# second once the first has printed "training". Whichever finishes first leaves the instance up.
# Checkpoints every 5k steps; all of them go to https://huggingface.co/ankurcw/wmta_<run>
# under checkpoints/<step>/. The instance shuts down at the end, also after a failure
# (an EBS-backed EC2 instance stops on shutdown, so its disk is kept).
set -euo pipefail  # pipefail: a failed lerobot-train piped into tee still counts as failed

RUN=${1:?usage: HF_TOKEN=... bash train_remote.sh B|C|C-ACT}
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token with write access}"
LEROBOT_COMMIT=8bf6056d
WORK=$HOME/wmta
OUT=$WORK/outputs/$RUN
LOG=$WORK/train_$RUN.log
TRAINING_STARTED=0
NO_SHUTDOWN=${NO_SHUTDOWN:-0}   # NO_SHUTDOWN=1 keeps the instance running (for debugging)

case "$RUN" in
  B)
    POLICY_ARGS=(--policy.path=lerobot/smolvla_base)
    DATASET=ankurcw/wmta_demos_v2
    REPO=ankurcw/wmta_B_smolvla
    BATCH=${BATCH:-64}; STEPS=${STEPS:-20000}
    EXTRA=(
           "--rename_map={\"observation.images.scene\": \"observation.images.camera1\", \"observation.images.wrist\": \"observation.images.camera2\"}")
    ;;
  C)
    POLICY_ARGS=(--policy.path=lerobot/smolvla_base)
    DATASET=ankurcw/wmta_demos_v2_marked
    REPO=ankurcw/wmta_C_smolvla
    BATCH=${BATCH:-64}; STEPS=${STEPS:-20000}
    EXTRA=(
           "--rename_map={\"observation.images.scene\": \"observation.images.camera1\", \"observation.images.wrist\": \"observation.images.camera2\"}")
    ;;
  C-ACT)
    POLICY_ARGS=(--policy.type=act --policy.use_amp=true)
    DATASET=ankurcw/wmta_demos_v2_marked
    REPO=ankurcw/wmta_C-ACT_act
    BATCH=${BATCH:-8}; STEPS=${STEPS:-30000}
    EXTRA=()
    ;;
  *) echo "unknown run '$RUN'"; exit 2 ;;
esac

mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1
echo "== $(date -u) run $RUN on $(hostname); GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv

finish() {
  code=$?
  echo "== $(date -u) training exited with code $code; uploading checkpoints"
  export PATH="$HOME/.local/bin:$PATH"
  if [ -d "$OUT/checkpoints" ]; then
    for ck in "$OUT"/checkpoints/[0-9]*; do
      [ -d "$ck/pretrained_model" ] || continue
      "$WORK/lerobot/.venv/bin/hf" upload "$REPO" "$ck/pretrained_model" "checkpoints/$(basename "$ck")" \
        --repo-type model --commit-message "$RUN checkpoint $(basename "$ck")" || echo "upload of $ck failed"
    done
  fi
  "$WORK/lerobot/.venv/bin/hf" upload "$REPO" "$LOG" "logs/train_$RUN.log" --repo-type model || true
  echo "== $(date -u) done (exit code $code)"
  # Shut down only once real training has started; a failure in setup or the memory check
  # leaves the instance running so it can be fixed in place.
  # Another run still training on this instance (C and C-ACT together): leave it up for that one.
  if [ "$NO_SHUTDOWN" != "1" ] && [ "$TRAINING_STARTED" = "1" ]; then
    if pgrep -f lerobot-train >/dev/null; then echo "== another run is still training; not shutting down"
    else sudo shutdown -h now; fi
  fi
  if [ "$TRAINING_STARTED" != "1" ]; then echo "== instance left running (failed before training started)"; fi
}
trap finish EXIT

# System packages and uv
sudo apt-get update -qq && sudo apt-get install -y -qq git ffmpeg >/dev/null
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"

# LeRobot at the study's pinned commit
if [ ! -d "$WORK/lerobot" ]; then
  git clone -q https://github.com/huggingface/lerobot.git "$WORK/lerobot"
fi
cd "$WORK/lerobot"
if [ ! -x .venv/bin/lerobot-train ]; then   # set up once; a second run on the same instance reuses it
  git checkout -q "$LEROBOT_COMMIT"
  uv venv -q --python 3.12 .venv
  VIRTUAL_ENV=$PWD/.venv uv pip install -q -e ".[smolvla,training]"   # training = datasets, av, accelerate, wandb
fi
source .venv/bin/activate
echo "== lerobot $(git rev-parse --short HEAD), torch $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

hf auth login --token "$HF_TOKEN" >/dev/null
hf repo create "$REPO" --repo-type model --exist-ok >/dev/null || true

train() {  # train <batch> <steps> <output dir> [extra args]
  local batch=$1 steps=$2 out=$3; shift 3
  lerobot-train \
    "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$DATASET" \
    --dataset.video_backend=pyav \
    "${EXTRA[@]}" \
    --batch_size="$batch" --steps="$steps" \
    --log_freq=100 \
    --seed=1000 \
    --output_dir="$out" \
    --job_name="wmta_$RUN" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false "$@"
}

# Memory check: 10 steps at the planned batch, nothing saved. Only a CUDA out-of-memory error
# switches to batch 32 for twice the steps (same passes over the data); any other error stops here.
PRE="$WORK/preflight_${RUN}_$BATCH.log"
if ! train "$BATCH" 10 "$WORK/preflight_${RUN}_$BATCH" --save_checkpoint=false 2>&1 | tee "$PRE"; then
  if [ "$BATCH" -gt 32 ] && grep -qiE "out of memory|OutOfMemoryError" "$PRE"; then
    echo "== batch $BATCH does not fit on this GPU; using batch 32, steps $((STEPS * BATCH / 32))"
    STEPS=$((STEPS * BATCH / 32)); BATCH=32
    train "$BATCH" 10 "$WORK/preflight_${RUN}_$BATCH" --save_checkpoint=false
  else
    echo "== the memory check failed for a reason other than memory; see above"; exit 1
  fi
fi
echo "== $(date -u) training $RUN: batch $BATCH, steps $STEPS, checkpoints every 5000"
TRAINING_STARTED=1
train "$BATCH" "$STEPS" "$OUT" --save_freq=5000
