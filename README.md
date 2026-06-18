# cavity-epgp

EP-GP solver for the interior PEC cavity reaction operator.
Thin layer over maxwellgp. Runs on ETH Euler.

## Euler

    euler/submit.sh                 # sbatch 2D (N_s, N_b) grid -> out/{ellipse,sphere}/

## Field slice

    uv run epgp-operator field --config res/config_ellipse.txt --out out/ellipse/field.npz
    uv run epgp-operator field --config res/config_sphere.txt  --out out/sphere/field.npz

Output: `out/{shape}/T_ns{ns}_nb{nb}.npy`, `manifest.csv`, `provenance.csv`.
Pull results into cavity-benchmark with `./pull-euler.sh`.
