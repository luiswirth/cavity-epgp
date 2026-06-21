#!/bin/bash
set -euo pipefail
if [[ $# -eq 0 ]]; then set -- ellipse sphere; fi
mkdir -p out/logs
uv sync --quiet
for geom in "$@"; do
  rm -rf out/noise/$geom
  sbatch --array=1-5%4 euler/run.sbatch "$geom" noise euler/noise.txt
done
