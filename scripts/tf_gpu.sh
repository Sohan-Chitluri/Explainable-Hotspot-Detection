#!/usr/bin/env bash
# Enable pip TensorFlow GPU on NixOS: system libcuda + pip nvidia-* CUDA/cuDNN wheels.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
SITE="${VENV}/lib/python3.11/site-packages"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing venv at ${VENV}" >&2
  exit 1
fi

libs=("/run/opengl-driver/lib")
for pkg in cuda_runtime cublas cudnn cufft curand cusolver cusparse cuda_nvrtc nvjitlink nccl cuda_cupti nvtx; do
  d="${SITE}/nvidia/${pkg}/lib"
  if [[ -d "${d}" ]]; then
    libs+=("${d}")
  fi
done
# Some wheels use slightly different folder names
for d in "${SITE}"/nvidia/*/lib; do
  [[ -d "${d}" ]] || continue
  libs+=("${d}")
done

# de-duplicate
LD_LIBRARY_PATH="$(printf '%s\n' "${libs[@]}" | awk 'NF && !seen[$0]++' | paste -sd: -)"
export LD_LIBRARY_PATH
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_cuda_data_dir=${SITE}/nvidia/cuda_nvcc}"

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >&2
exec "${VENV}/bin/python" "$@"
