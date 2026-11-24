#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
run_name="gendeg_init_no_llr_512_ft5_gpu0"
long_log="${repo}/logs/long_runs/${run_name}.log"
clean_root="${BSDS500_CLEAN_ROOT:-${repo}/data/BSDS500/images/test}"
resume_dir="${1:-}"

cd "${repo}"
mkdir -p logs/long_runs
export TZ=Asia/Seoul
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${repo}:${repo}/stable_diffusion:${repo}/NAFNet:${repo}/third_party/taming-transformers${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

if [[ -n "${resume_dir}" ]]; then
  resume_checkpoint="$(find "${resume_dir}/checkpoints" -maxdepth 1 -type f -name 'last*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "${resume_checkpoint}" || ! -f "${resume_checkpoint}" ]]; then
    echo "Resume checkpoint not found under ${resume_dir}/checkpoints" >&2
    exit 1
  fi
  "${python_bin}" stable_diffusion/main.py \
    --resume "${resume_checkpoint}" --scale_lr false --train --no-test \
    2>&1 | tee -a "${long_log}"
  run_dir="${resume_dir}"
else
  "${python_bin}" stable_diffusion/main.py \
    --name "${run_name}" \
    --logdir logs \
    --base \
      configs/train.yaml \
      configs/train_full_latent_50k_60ep.yaml \
      configs/train_factorwise_stats.yaml \
      configs/finetune_gendeg_no_llr_512_5ep.yaml \
    --scale_lr false \
    --train \
    --no-test \
    2>&1 | tee -a "${long_log}"
  run_dir="$(find logs -maxdepth 1 -type d -name "*_${run_name}" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
checkpoint="$(find "${run_dir}/checkpoints" -maxdepth 1 -type f -name 'last*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
project_config="$(find "${run_dir}/configs" -type f -name '*-project.yaml' | sort | tail -1)"
if [[ ! -f "${checkpoint}" || ! -f "${project_config}" ]]; then
  echo "Training completed but checkpoint/config was not found under ${run_dir}" >&2
  exit 1
fi

run_inference() {
  local name="$1"
  local text_cfg="$2"
  local image_cfg="$3"
  local raw_dir="${repo}/results/${name}"
  local scm_dir="${repo}/results/${name}_scm"
  mkdir -p "${raw_dir}" "${scm_dir}"

  "${python_bin}" scripts/infer_bsds500_simultaneous_combinations.py \
    --ckpt "${checkpoint}" --project-config "${project_config}" \
    --clean-root "${clean_root}" --output-dir "${raw_dir}" \
    --device cuda:0 --resolution 512 --steps 100 \
    --cfg-text "${text_cfg}" --cfg-image "${image_cfg}" \
    --seed 2026 --max-order 2 --image-ids 290035 --donor-id 05201 \
    2>&1 | tee "${raw_dir}/run.log"

  "${python_bin}" scripts/apply_scm_to_composite_results.py \
    --input-dir "${raw_dir}" --output-dir "${scm_dir}" \
    --scm-ckpt checkpoints/nafnet_sd.ckpt --device cuda:0 \
    2>&1 | tee "${scm_dir}/run.log"
}

run_inference "bsds500_gendeg_init_no_llr_512_ft5_cfg1x1_290035" 1.0 1.0
run_inference "bsds500_gendeg_init_no_llr_512_ft5_cfg7p5x1p5_290035" 7.5 1.5

"${python_bin}" scripts/build_cfg_scm_comparison.py \
  --low-raw results/bsds500_gendeg_init_no_llr_512_ft5_cfg1x1_290035 \
  --low-scm results/bsds500_gendeg_init_no_llr_512_ft5_cfg1x1_290035_scm \
  --full-raw results/bsds500_gendeg_init_no_llr_512_ft5_cfg7p5x1p5_290035 \
  --full-scm results/bsds500_gendeg_init_no_llr_512_ft5_cfg7p5x1p5_290035_scm \
  --output-dir results/bsds500_gendeg_init_no_llr_512_ft5_cfg_scm_comparison \
  --image-id 290035
