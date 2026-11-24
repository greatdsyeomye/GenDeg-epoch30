#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
run_dir="${repo}/logs/2026-10-14T18-56-32_gendeg_init_no_llr_512_ft5_gpu0"
log="${repo}/logs/long_runs/gendeg_init_no_llr_512_to30_gpu0.log"

cd "${repo}"
export TZ=Asia/Seoul
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${repo}:${repo}/stable_diffusion:${repo}/NAFNet:${repo}/third_party/taming-transformers${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

checkpoint="$(find "${run_dir}/checkpoints" -maxdepth 1 -type f -name 'last*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "No resumable checkpoint under ${run_dir}/checkpoints" >&2
  exit 1
fi

exec "${python_bin}" stable_diffusion/main.py \
  --resume "${checkpoint}" \
  --base configs/continue_gendeg_no_llr_512_to30.yaml \
  --scale_lr false --train --no-test \
  2>&1 | tee -a "${log}"
