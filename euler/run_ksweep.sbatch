#!/bin/bash
#SBATCH --job-name=epgp-ksweep
#SBATCH --account=ls_math
#SBATCH --output=out/logs/slurm-%A.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
GEOM=$1
ROOT=$(pwd)
R=$ROOT/out/ksweep/$GEOM
rm -rf "$R"; mkdir -p "$R" "$ROOT/out/logs"
srun --cpu-bind=cores uv run --project "$ROOT" epgp-operator ksweep \
  --config "$ROOT/res/config_${GEOM}.txt" --n-spectral 256 --n-boundary 4096 \
  --kmin 0.25 --kmax 4.0 --nk 200 --outdir "$R"
echo "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown),$(hostname),$(date -Is)" \
  > "$R/provenance.csv"
