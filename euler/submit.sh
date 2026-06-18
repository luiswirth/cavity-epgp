#!/bin/bash
set -euo pipefail
mkdir -p out/logs
for geom in "${@:-ellipse sphere}"; do
  sbatch --array=1-60 euler/run.sbatch "$geom"
done
