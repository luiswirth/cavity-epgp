# cavity-epgp

EP-GP solver for the interior PEC cavity reaction operator.
Thin layer over maxwellgp. Runs on ETH Euler.

## Euler

    euler/submit_grid.sh [geom]     # sbatch 2D (N_s, N_b) convergence grid
    euler/submit_ref.sh  [geom]     # sbatch single high-fidelity reference run

## Field slice

    uv run epgp-operator field --config res/config_ellipse.txt --out out/grid/ellipse/field.npz
    uv run epgp-operator field --config res/config_sphere.txt  --out out/grid/sphere/field.npz

Output: `out/{grid,ref}/{shape}/T_ns{ns}_nb{nb}.npy`, `manifest.csv`, `provenance.csv`.
Pull results into cavity-benchmark with `./pull-euler.sh`.
