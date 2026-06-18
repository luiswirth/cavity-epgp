#!/bin/bash
#SBATCH --job-name=epgp
#SBATCH --account=ls_math
#SBATCH --output=out/logs/slurm-%A_%a.log
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=EPYC_7742
#
#   euler/submit_grid.sh [geom]   # convergence grid
#   euler/submit_ref.sh  [geom]   # single high-fidelity reference run
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
GEOM=$1; MODE=$2; TASKFILE=$3
ROOT=$(pwd); R=$ROOT/out/$MODE/$GEOM; mkdir -p "$R" "$ROOT/out/logs"
read -r NS NB < <(grep -vE '^\s*#|^\s*$' "$ROOT/$TASKFILE" | sed -n "${SLURM_ARRAY_TASK_ID}p")
work=$R/work/ns${NS}_nb${NB}; rm -rf "$work"; mkdir -p "$work"; cd "$work"
t0=$SECONDS
srun --cpu-bind=cores /usr/bin/time -v -o time.txt \
  uv run --project "$ROOT" epgp-operator operator \
  --config "$ROOT/res/config_${GEOM}.txt" --n-spectral "$NS" --n-boundary "$NB" --outdir "$R" \
  > run.log 2>&1
secs=$((SECONDS - t0)); log=$(cat run.log); echo "$log"
dofs=$(echo "$log" | grep -oE 'dofs=[0-9]+' | grep -oE '[0-9]+')
cond=$(echo "$log" | grep -oE 'cond=[0-9.eE+-]+' | grep -oE '[0-9.eE+-]+$')
mem_kb=$(grep -i 'Maximum resident set size' time.txt | grep -oE '[0-9]+' || true)
# manifest columns: n_spectral,n_boundary,dofs,secs,mem_kb,cond
echo "${NS},${NB},${dofs},${secs},${mem_kb},${cond}" >> "$R/manifest.csv"
echo "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown),$(hostname),${NS},${NB},$(date -Is)" >> "$R/provenance.csv"
echo "done ns${NS}nb${NB}: dofs=$dofs cond=$cond secs=$secs mem_kb=$mem_kb"
