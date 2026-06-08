#!/bin/bash
#SBATCH -J fluidsim-efficiency
#SBATCH -p rtxp6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_logs/fluidsim-efficiency-%j.out
#SBATCH -e slurm_logs/fluidsim-efficiency-%j.err

set -euo pipefail

REPO="/mnt/home/jjian/workplace/qz/FluidSim"
PY="$REPO/.micromamba_fluidsim/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ROOT="/tmp/${USER}/fluidsim-efficiency-${JOB_ID}"
RAW_DIR="$REPO/output/benchmark_efficiency/raw/job_${JOB_ID}"
OUT_DIR="$REPO/output/benchmark_efficiency"

cd "$REPO"
mkdir -p slurm_logs "$RUN_ROOT" "$RAW_DIR" "$OUT_DIR"

export TI_ARCH=cuda
export MPLBACKEND=Agg
export PATH="$REPO/.micromamba_fluidsim/bin:$PATH"

echo "[job] host=$(hostname) job=${JOB_ID} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] start=$(date --iso-8601=seconds)"
echo "[job] repo=$(pwd)"
"$PY" - <<'PY'
import taichi as ti
import numpy as np
print("[job] taichi", ti.__version__)
print("[job] numpy", np.__version__)
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

declare -A branches=(
  [flip]="main"
  [apic]="apic-branch"
  [polypic]="polypic"
)

for alg in flip apic polypic; do
  branch="${branches[$alg]}"
  target="$RUN_ROOT/$alg"
  rm -rf "$target"
  mkdir -p "$target"
  git archive "$branch" framework.py | tar -x -C "$target"
  echo "[prepare] $alg from $branch commit=$(git rev-parse --short "$branch")"
done

for rep in 1 2 3; do
  for alg in flip apic polypic; do
    echo "[run] rep=${rep} alg=${alg} $(date --iso-8601=seconds)"
    mkdir -p "$RAW_DIR/rep_${rep}/$alg"
    "$PY" "$REPO/scripts/run_branch_benchmark.py" \
      --repo "$RUN_ROOT/$alg" \
      --algorithm "$alg" \
      --out "$RAW_DIR/rep_${rep}/$alg" \
      --ratio 0.97 \
      2>&1 | tee "$RAW_DIR/rep_${rep}/$alg/run.log"
  done
done

"$PY" "$REPO/scripts/summarize_efficiency.py" \
  --raw-dir "$RAW_DIR" \
  --out-dir "$OUT_DIR" \
  --doc "$REPO/docs/final_report/efficiency_benchmark.md" \
  --warmup-frames 5

{
  echo "job_id=${JOB_ID}"
  echo "host=$(hostname)"
  echo "start_branch=$(git branch --show-current)"
  echo "current_commit=$(git rev-parse --short HEAD)"
  echo "flip_commit=$(git rev-parse --short main)"
  echo "apic_commit=$(git rev-parse --short apic-branch)"
  echo "polypic_commit=$(git rev-parse --short polypic)"
  echo "raw_dir=$RAW_DIR"
  echo "ti_arch=${TI_ARCH}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
} > "$OUT_DIR/benchmark_metadata.txt"

echo "[job] end=$(date --iso-8601=seconds)"
