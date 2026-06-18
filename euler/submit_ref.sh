#!/bin/bash
set -euo pipefail
if [[ $# -eq 0 ]]; then set -- ellipse sphere; fi
mkdir -p out/logs
uv sync --quiet
for geom in "$@"; do
  rm -f out/ref/$geom/manifest.csv out/ref/$geom/provenance.csv
  sbatch --array=1 euler/run.sbatch "$geom" ref euler/ref.txt
done
