#!/bin/bash
set -euo pipefail
mkdir -p out/logs
for geom in "${@:-ellipse sphere}"; do
  sbatch --array=1 euler/run.sbatch "$geom" ref euler/epgp_ref.txt
done
