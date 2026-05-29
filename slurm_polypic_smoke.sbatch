#!/usr/bin/env bash
#SBATCH --job-name=fluidsim-polypic-smoke
#SBATCH --partition=rtxp6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/home/jjian/workplace/qz/FluidSim
#SBATCH --output=/mnt/home/jjian/workplace/qz/FluidSim/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/home/jjian/workplace/qz/FluidSim/slurm_logs/%x-%j.err

set -euo pipefail

cd /mnt/home/jjian/workplace/qz/FluidSim

echo "[job] host=$(hostname) job=${SLURM_JOB_ID:-manual} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] start=$(date -Is)"

export TI_ARCH=cuda
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export FLUIDSIM_SMOKE_FRAMES="${FLUIDSIM_SMOKE_FRAMES:-30}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${PWD}/.micromamba_root}"

PYTHON_BIN="python3"
if ! "${PYTHON_BIN}" - <<'DEPCHECK' >/dev/null 2>&1
import taichi, PIL, matplotlib, numpy
DEPCHECK
then
  if ! command -v micromamba >/dev/null 2>&1; then
    echo "[job] ERROR: dependencies missing and micromamba is unavailable" >&2
    exit 2
  fi
  ENV_PREFIX="${PWD}/.micromamba_fluidsim"
  if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "[job] creating micromamba env at ${ENV_PREFIX}"
    micromamba create -y -p "${ENV_PREFIX}" -c conda-forge python=3.10 pip
  fi
  PYTHON_BIN="${ENV_PREFIX}/bin/python"
  export PATH="${ENV_PREFIX}/bin:${PATH}"
  if ! "${PYTHON_BIN}" - <<'DEPCHECK2' >/dev/null 2>&1
import taichi, PIL, matplotlib, numpy
DEPCHECK2
  then
    echo "[job] installing FluidSim dependencies into ${ENV_PREFIX}"
    "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
    "${PYTHON_BIN}" -m pip install 'taichi==1.7.4' pillow matplotlib numpy
  fi
else
  echo "[job] system python already has FluidSim dependencies"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v micromamba >/dev/null 2>&1; then
    echo "[job] installing ffmpeg into ${PWD}/.micromamba_fluidsim"
    micromamba install -y -p "${PWD}/.micromamba_fluidsim" -c conda-forge ffmpeg
    export PATH="${PWD}/.micromamba_fluidsim/bin:${PATH}"
  else
    echo "[job] WARNING: ffmpeg unavailable and micromamba missing; video compile may fail" >&2
  fi
fi

"${PYTHON_BIN}" - <<'ENVINFO'
import taichi as ti
import numpy as np
print('[job] python ok')
print('[job] taichi', ti.__version__)
print('[job] numpy', np.__version__)
ENVINFO

"${PYTHON_BIN}" -u run_polypic_smoke.py   --scene dam_break   --ratio 0.97   --method polypic   --frames "${FLUIDSIM_SMOKE_FRAMES}"   --output-root "output/slurm_smoke/${SLURM_JOB_ID:-manual}"

echo "[job] end=$(date -Is)"
