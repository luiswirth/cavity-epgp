# cavity-epgp

Ehrenpreis-Palamodov Gaussian-process solver for the interior PEC cavity
reaction operator. A thin cavity-specific layer over `maxwellgp`: it builds the
boundary data from the analytic dipole field, conditions the Maxwell-constrained
GP on the tangential trace, and reads off the dipole reaction operator T.

Mirrors `cavity-bem`: a solver plus an `euler/` array runner over a grid file,
producing operators under `out/`. The benchmarking (cross-validation against the
BEM reference, figures) lives in `cavity-benchmark`.

## Solver

    uv run epgp-operator operator --config res/config_ellipse.txt \
        --n-spectral 256 --n-boundary 1200 --outdir out/ellipse
    uv run epgp-operator field --config res/config_ellipse.txt \
        --source 0 0 1 --pol 1 0 0 --out out/ellipse/field.npz

The reaction-operator CLI saves `out/<shape>/T_ns<ns>_nb<nb>.npy` and prints
`dofs=`, `cond=` for the run harness to record.

## Euler

The convergence grid is `(n_spectral, n_boundary)` in `euler/epgp_grid.txt`
(10 x 6 = 60 points). One array task per grid point:

    sbatch --array=1-60 euler/run.sbatch ellipse
    sbatch --array=1-60 euler/run.sbatch sphere

Each task wraps the solver in `/usr/bin/time -v` and appends a row to
`out/<shape>/manifest.csv` (n_spectral,n_boundary,dofs,secs,mem_kb,cond). The
config is the shared benchmark config produced by `cavity-benchmark gen-config`.
