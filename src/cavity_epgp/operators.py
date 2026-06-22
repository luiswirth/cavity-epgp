"""EPGP reaction-operator assembly for the PEC ellipsoidal cavity.

Drives the maxwellgp Maxwell-constrained Gaussian process: builds the boundary
data from the analytic incident field, conditions the GP on the tangential
trace, and reads off the dipole reaction operator T. The plane-wave directions
and boundary collocation are the solver's private discretization; the shared
benchmark geometry (Lambda points, k, semi-axes) comes from the config file.
"""

import argparse
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from maxwellgp import GaussianProcess, MaxwellKernel
from maxwellgp.utils import fibonacci_sphere

from .analytic import incident_field_batch

jax.config.update("jax_enable_x64", True)

JITTER = 1e-8


@dataclass
class GPConfig:
    """Maxwell-GP conditioning hyperparameters (the solver's private settings)."""

    n_boundary: int = 1200
    log_noise: float = -12.0
    opt_noise: bool = False
    opt_steps: int = 200

    @classmethod
    def from_args(cls, args):
        return cls(args.n_boundary, args.log_noise, args.opt_noise, args.opt_steps)


# --- geometry / boundary data -------------------------------------------------

def load_config(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    k, a, b, c, _n = lines[0].split()
    semiaxes = np.array([float(a), float(b), float(c)])
    data = np.array([[float(v) for v in ln.split()] for ln in lines[1:]])
    return float(k), semiaxes, data[:, 0:3], data[:, 3:6], data[:, 6:9]


def boundary_collocation(semiaxes, n):
    u = np.asarray(fibonacci_sphere(n))
    points = u * semiaxes
    normals = points / semiaxes**2
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def tangential_trace(Ei, normals):
    """Tangential trace of the (negated) incident field: PEC boundary data."""
    En = np.sum(Ei * normals, axis=1, keepdims=True)
    return -(Ei - En * normals)


# --- conditioning (shared fit) ------------------------------------------------

def optimize_log_noise(kernel, log_noise0, X_train, Y, steps, lr=0.05):
    ln = jnp.asarray(log_noise0)
    opt = optax.adam(lr)
    state = opt.init(ln)

    def loss_of(ln):
        return GaussianProcess(kernel, log_noise=ln).nlml(X_train, Y)

    @jax.jit
    def step(ln, state):
        loss, g = jax.value_and_grad(loss_of)(ln)
        updates, state = opt.update(g, state)
        ln = jnp.clip(optax.apply_updates(ln, updates), -12.0, 0.0)
        return ln, state, loss

    loss = loss_of(ln)
    for _ in range(steps):
        ln, state, loss = step(ln, state)
    return float(ln), float(loss)


def fit(cfg, semiaxes, k, Y, n_spectral):
    """Build the boundary, condition the Maxwell-GP, return (model, posterior)."""
    bnd_points, bnd_normals = boundary_collocation(semiaxes, cfg.n_boundary)
    X_train = jnp.asarray(np.concatenate([bnd_points, bnd_normals], axis=1))

    kernel = MaxwellKernel(n_spectral=n_spectral, wavenumber=k, trace="tangential")
    log_noise = cfg.log_noise
    if cfg.opt_noise:
        log_noise, _ = optimize_log_noise(kernel, log_noise, X_train, Y, cfg.opt_steps)
        print(f"tuned log_noise = {log_noise:.4f} (eps={np.exp(log_noise):.3e})")
    model = GaussianProcess(kernel, log_noise=log_noise)
    return model, model.condition(X_train, Y, jitter=JITTER)


# --- reaction operator assembly -----------------------------------------------

def _nlml(post, model, Y):
    # Marginal likelihood reusing the conditioned factor (no re-factorization):
    # Phi_Y = A mu_w = L (L^H mu_w), and the nlml fit term is <Phi_Y, mu_w>.
    L, w = post.L, post.mu_w
    M, J = Y.shape
    Phi_Y = L @ (L.conj().T @ w)
    data_fit = 0.5 * (jnp.vdot(Y, Y).real / jnp.exp(model.log_noise)
                      - jnp.sum((Phi_Y.conj() * w).real))
    logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L).real))
    logdet_C = logdet_A + M * model.log_noise - jnp.sum(model.kernel.log_weights)
    return float(data_fit + J * (0.5 * logdet_C + 0.5 * M * jnp.log(2.0 * jnp.pi)))


def assemble_operator(cfg, semiaxes, k, points, e1, e2, n_spectral):
    """Assemble the dipole reaction operator T for one (k, n_spectral).

    Returns (T, Sigma, nlml, posterior, model); Sigma is the posterior covariance
    of the operator entries (transmitter-independent), nlml the marginal
    likelihood of the boundary fit, the posterior carries the conditioned factor
    used for the system condition number, the model the noise level.
    """
    configs = []
    for i in range(len(points)):
        n = points[i] / np.linalg.norm(points[i])
        configs.append((points[i], n, e1[i]))
        configs.append((points[i], n, e2[i]))
    n_cfg = len(configs)

    bnd_points, bnd_normals = boundary_collocation(semiaxes, cfg.n_boundary)
    cols = [tangential_trace(incident_field_batch(bnd_points, z, k, p), bnd_normals).reshape(-1)
            for z, _, p in configs]
    Y = jnp.asarray(np.stack(cols, axis=1))

    model, post = fit(cfg, semiaxes, k, Y, n_spectral)

    X_query = jnp.asarray(np.stack([np.concatenate([x, nrm]) for x, nrm, _ in configs]))
    Q = jnp.asarray(np.stack([q for _, _, q in configs]))
    # polarization-projected query map: Psi[:, i] sums the 3 tangential
    # components at receiver i weighted by its polarization Q[i].
    Phi_q = model.kernel.features(X_query).reshape(-1, n_cfg, 3)
    Psi = jnp.einsum("fic,ic->fi", Phi_q, Q)
    T = np.asarray(post.mean(Psi))
    Sigma = np.asarray(post.cov(Psi))
    nlml = _nlml(post, model, Y)
    return T, Sigma, nlml, post, model


# --- subcommand: reaction operator --------------------------------------------

def run_operator(args):
    """One grid point: assemble T at (n_spectral, n_boundary), save it, and print
    dofs and cond for the run harness (secs and mem come from /usr/bin/time)."""
    ns, nb = args.n_spectral, args.n_boundary
    k, semiaxes, points, e1, e2 = load_config(args.config)
    cfg = GPConfig.from_args(args)
    T, Sigma, nlml, post, model = assemble_operator(cfg, semiaxes, k, points, e1, e2, ns)

    cond = float(np.linalg.cond(np.asarray(post.L @ post.L.conj().T)))
    recip = np.linalg.norm(T - T.T) / np.linalg.norm(T)
    std = np.sqrt(np.clip(np.real(np.diag(Sigma)), 0.0, None))
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"T_ns{ns}_nb{nb}.npy")
    np.save(out, T)
    np.save(os.path.join(args.outdir, f"Sigma_ns{ns}_nb{nb}.npy"), Sigma)
    print(f"dofs={2 * ns}")
    print(f"cond={cond:.6e}")
    print(f"recip={recip:.3e}")
    print(f"mean_std={std.mean():.6e}")
    print(f"log_noise={float(model.log_noise):.6f}")
    print(f"nlml={nlml:.6e}")
    print(f"wrote {out}")


# --- subcommand: field slice --------------------------------------------------

def run_field(args):
    k, semiaxes, *_ = load_config(args.config)
    a, b, c = semiaxes
    z = np.array(args.source, dtype=float)
    p = np.array(args.pol, dtype=float)

    cfg = GPConfig.from_args(args)
    bnd_points, bnd_normals = boundary_collocation(semiaxes, cfg.n_boundary)
    y = jnp.asarray(tangential_trace(incident_field_batch(bnd_points, z, k, p),
                                     bnd_normals).reshape(-1, 1))

    model, post = fit(cfg, semiaxes, k, y, args.n_spectral)

    xs = np.linspace(-1.05 * a, 1.05 * a, args.ngrid)
    zs = np.linspace(-1.05 * c, 1.05 * c, args.ngrid)
    XX, ZZ = np.meshgrid(xs, zs)
    pts = np.stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()], axis=1)

    mean_chunks, var_chunks = [], []
    for i in range(0, len(pts), args.batch):
        phi = model.kernel.feature_map.full(jnp.asarray(pts[i : i + args.batch]))
        mean_chunks.append(np.asarray(post.mean(phi)))
        var_chunks.append(np.asarray(post.var(phi)))
    field6 = np.concatenate(mean_chunks).reshape(-1, 6)
    var6 = np.concatenate(var_chunks).reshape(-1, 6)
    Escat = field6[:, :3]
    Escat_std = np.sqrt(var6[:, :3])  # posterior std of the scattered E field
    Einc = incident_field_batch(pts, z, k, p)
    Etot = Einc + Escat

    inside = (pts[:, 0] ** 2 / a**2 + pts[:, 1] ** 2 / b**2 + pts[:, 2] ** 2 / c**2) <= 1.0
    ng = args.ngrid
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        xs=xs, zs=zs,
        Escat=Escat.reshape(ng, ng, 3),
        Escat_std=Escat_std.reshape(ng, ng, 3),
        Einc=Einc.reshape(ng, ng, 3),
        Etot=Etot.reshape(ng, ng, 3),
        mask=inside.reshape(ng, ng),
        semiaxes=semiaxes, source=z, pol=p, k=k,
    )
    print(f"wrote {args.out}  (slice {ng}x{ng}, source={z.tolist()}, pol={p.tolist()})")


# --- subcommand: wavenumber sweep ---------------------------------------------

def run_ksweep(args):
    """Sweep the wavenumber at fixed resolution, recording the conditioning
    system's condition number per k (a resonance indicator). No operator solve:
    the factor depends only on kernel, boundary, k and noise, not on the data."""
    _, semiaxes, *_ = load_config(args.config)
    cfg = GPConfig.from_args(args)
    bnd_points, bnd_normals = boundary_collocation(semiaxes, cfg.n_boundary)
    X_train = jnp.asarray(np.concatenate([bnd_points, bnd_normals], axis=1))
    Y = jnp.zeros((3 * cfg.n_boundary, 1))  # dummy: L is data-independent
    ks = np.linspace(args.kmin, args.kmax, args.nk)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "ksweep.csv")
    with open(out, "w") as f:
        f.write("k,cond\n")
        for k in ks:
            kernel = MaxwellKernel(n_spectral=args.n_spectral, wavenumber=float(k),
                                   trace="tangential")
            post = GaussianProcess(kernel, log_noise=cfg.log_noise).condition(
                X_train, Y, jitter=JITTER)
            cond = float(np.linalg.cond(np.asarray(post.L @ post.L.conj().T)))
            f.write(f"{k:.6f},{cond:.6e}\n")
            print(f"k={k:.4f} cond={cond:.6e}")
    print(f"wrote {out}")


# --- CLI ----------------------------------------------------------------------

def add_common(sp):
    sp.add_argument("--config", default="res/config_ellipse.txt")
    sp.add_argument("--n-spectral", type=int, default=256)
    sp.add_argument("--n-boundary", type=int, default=1200)
    sp.add_argument("--log-noise", type=float, default=-12.0)
    sp.add_argument("--opt-noise", action=argparse.BooleanOptionalAction, default=False)
    sp.add_argument("--opt-steps", type=int, default=200)


def main():
    ap = argparse.ArgumentParser(description="Maxwell-GP PEC ellipsoidal cavity")
    sub = ap.add_subparsers(dest="cmd", required=True)

    op = sub.add_parser("operator", help="assemble the dipole reaction operator T")
    add_common(op)
    op.add_argument("--outdir", default="out/ellipse")
    op.set_defaults(func=run_operator)

    fld = sub.add_parser("field", help="evaluate the field on a slice for one dipole")
    add_common(fld)
    fld.add_argument("--source", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    fld.add_argument("--pol", type=float, nargs=3, required=True, metavar=("PX", "PY", "PZ"))
    fld.add_argument("--ngrid", type=int, default=400)
    fld.add_argument("--batch", type=int, default=4000)
    fld.add_argument("--out", default="out/ellipse/field.npz")
    fld.set_defaults(func=run_field)

    ks = sub.add_parser("ksweep", help="sweep wavenumber, record conditioning")
    add_common(ks)
    ks.add_argument("--kmin", type=float, default=0.25)
    ks.add_argument("--kmax", type=float, default=4.0)
    ks.add_argument("--nk", type=int, default=200)
    ks.add_argument("--outdir", default="out/ksweep/ellipse")
    ks.set_defaults(func=run_ksweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
