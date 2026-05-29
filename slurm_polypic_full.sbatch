#!/usr/bin/env bash
#SBATCH --job-name=fluidsim-polypic-full
#SBATCH --partition=rtxp6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
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
export FLUIDSIM_FORCE_RENDER=1
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${PWD}/.micromamba_root}"

ENV_PREFIX="${PWD}/.micromamba_fluidsim"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  if ! command -v micromamba >/dev/null 2>&1; then
    echo "[job] ERROR: ${ENV_PREFIX}/bin/python missing and micromamba unavailable" >&2
    exit 2
  fi
  echo "[job] creating micromamba env at ${ENV_PREFIX}"
  micromamba create -y -p "${ENV_PREFIX}" -c conda-forge python=3.10 pip
fi

export PATH="${ENV_PREFIX}/bin:${PATH}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

if ! "${PYTHON_BIN}" - <<'DEPCHECK' >/dev/null 2>&1
import taichi, PIL, matplotlib, numpy
DEPCHECK
then
  echo "[job] installing FluidSim Python dependencies"
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install 'taichi==1.7.4' pillow matplotlib numpy
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  if ! command -v micromamba >/dev/null 2>&1; then
    echo "[job] ERROR: ffmpeg missing and micromamba unavailable" >&2
    exit 3
  fi
  echo "[job] installing ffmpeg into ${ENV_PREFIX}"
  micromamba install -y -p "${ENV_PREFIX}" -c conda-forge ffmpeg
fi

"${PYTHON_BIN}" - <<'ENVINFO'
import taichi as ti
import numpy as np
print('[job] taichi', ti.__version__)
print('[job] numpy', np.__version__)
ENVINFO

"${PYTHON_BIN}" -u framework.py all 0.97 polypic

echo "[job] output summary"
find output/polypic/ratio_970 -maxdepth 3 -type f -printf '%p %s bytes
' | sort || true

echo "[job] end=$(date -Is)"
