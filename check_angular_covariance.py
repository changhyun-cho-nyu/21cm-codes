"""
check_angular_covariance.py
===========================
How good is "shell-averaged emissivity x direction-averaged transmission"?

compute_Jalpha_He.c / compute_Jalpha_AGN.c (with TRANSMISSION_DIR) build J_alpha^He at x as
    J_avg(x) = sum_R w(R) <eps(x + R n)>_n  <T(x, R; n)>_n ,
the product of the shell average of the emissivity and the direction average of the IGM
transmission (compute_transmission_He.c). The true sum is direction by direction,
    J_dir(x) = sum_R w(R) <eps(x + R n) T(x, R; n)>_n .
They differ where sources and absorbers are correlated on the sky of x (sources sit in
ionized regions, so the directions toward them tend to be the transparent ones).

This script traces, from sampled points x, the same rays as compute_transmission_He.c:
the same opacity files (n_HI, n_HeI), light cone (each distance uses the opacity and the
source box of the snapshot nearest its emission redshift), Verner+96 cross sections at
the photon's energy E_n2 (1+z')/(1+z) (the n = 2 window, the largest part of J), Fibonacci
directions, periodic box, out to the Ly-beta horizon. Along every ray it also samples the
source box, and it computes J_dir and J_avg with the same rays, so their ratio isolates the
angular covariance (the J code's exact shell average and the ray sampling cancel).
w(R) dR = (E'/E_n2)^-alpha dR, the emitted spectrum across the window (E' = E_n2 (1+z')/(1+z)),
reported for several alpha; the ratio barely depends on it.

Sources, separately (each ratio is independent of the normalisation):
  stars : SFR_eff He-band boxes   <sfr_dir>/StarFormationRate/StarFormationRate_NNN.hdf5
  agn   : LuminosityAGN renders   <render_dir>/LuminosityAGN/LuminosityAGN_NNN.hdf5 (two dirs: <= 428, > 428)

Output: the ratio J_dir/J_avg, J-weighted over the sampled points (the effect on the box-mean
J) and its percentiles, for ionized points (n_HI < 1% <n_H>, as the mean-free-path modes) and
for all; and the covariance factor <eps T>/(<eps><T>) in bins of R.

Usage
-----
  python check_angular_covariance.py --snap 300 --opa .../opacity_He_sfmask --sfr .../SFR_eff/He \\
         --render .../Flagship_new/postprocessing/ren_640 .../Flagship_below_z_4p75_new/postprocessing/ren_640 \\
         --zcsv .../SFR_eff/SFR_eff_summary.csv [--n-points 1000] [--ndir 192] [--sources stars agn]
"""
import argparse
import glob
import os
import re
import sys
import time

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

C_KMS, MPC_CM = 2.99792458e5, 3.0856776e24
OM, OL, H0 = 0.30964144, 0.6902672, 67.66
OB = 0.04897
E_LL_HE = 54.418
NH_MEAN_COM = 0.76 * OB * 3.0 * (0.6766 * 100 * 3.241e-20)**2 / (8 * np.pi * 6.674e-8) / 1.67262192e-24


def verner(E, E0, s0, ya, P, yw, y0, y1):
    x = E / E0 - y0
    y = np.sqrt(x * x + y1 * y1)
    return s0 * 1e-18 * ((x - 1)**2 + yw * yw) * y**(0.5 * P - 5.5) * (1 + np.sqrt(y / ya))**(-P)


def sigma_HI(E):
    return np.where(E < 13.598, 0.0, verner(E, 0.4298, 5.475e4, 32.88, 2.963, 0.0, 0.0, 0.0))


def sigma_HeI(E):
    return np.where(E < 24.587, 0.0, verner(E, 13.61, 949.2, 1.469, 3.188, 2.039, 0.4434, 2.136))


def fib_dirs(n):
    ga = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    zc = 1.0 - 2.0 * (i + 0.5) / n
    s = np.sqrt(1.0 - zc * zc)
    return np.stack([s * np.cos(ga * i), s * np.sin(ga * i), zc], axis=1)


def comoving_table(z0, z1, n=8192):
    z = np.linspace(z0, z1, n)
    Hz = H0 * np.sqrt(OM * (1 + z)**3 + OL)
    r = np.concatenate([[0.0], np.cumsum(0.5 * (C_KMS / Hz[1:] + C_KMS / Hz[:-1]) * np.diff(z))])
    return z, r                                              # cMpc


def pick(snaps, zs, z_lo, z_hi, kmax):
    """snapshots spanning [z_lo, z_hi] (plus the nearest beyond), at most kmax, evenly chosen"""
    inside = [s for s in snaps if z_lo - 1e-9 <= zs[s] <= z_hi + 1e-9]
    beyond = [s for s in snaps if zs[s] > z_hi + 1e-9]
    if beyond:
        inside.append(min(beyond, key=lambda s: zs[s]))
    inside.sort(key=lambda s: -zs[s])
    if len(inside) > kmax:
        idx = np.round(np.linspace(0, len(inside) - 1, kmax)).astype(int)
        inside = [inside[i] for i in idx]
    return inside


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--opa", required=True, help="opacity directory (opacity_NNN.hdf5, as used for the transmission)")
    p.add_argument("--sfr", help="SFR_eff He directory (contains StarFormationRate/)")
    p.add_argument("--render", nargs="+", help="render directories with LuminosityAGN/ (searched in order)")
    p.add_argument("--zcsv", required=True, help="SFR_eff_summary.csv (snap, redshift of every snapshot)")
    p.add_argument("--sources", nargs="+", default=["stars", "agn"], choices=["stars", "agn"])
    p.add_argument("--n-points", type=int, default=1000)
    p.add_argument("--ndir", type=int, default=192)
    p.add_argument("--kmax", type=int, default=8, help="snapshots kept per field along the light cone")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.5, 3.0])
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    zs = {}
    with open(a.zcsv) as f:
        next(f)
        for line in f:
            c = line.split(",")
            zs[int(c[0])] = float(c[1])
    z0 = zs[a.snap]
    z_max = (1 + z0) * (1 - 1 / 9) / (1 - 1 / 4) - 1                      # Ly-beta horizon
    opa_snaps = sorted(int(re.search(r"opacity_(\d+)\.hdf5", q).group(1)) for q in glob.glob(os.path.join(a.opa, "opacity_*.hdf5")))
    opa_snaps = [s for s in opa_snaps if s in zs]
    z_first = max(zs[s] for s in opa_snaps)
    z_max = min(z_max, z_first)
    zt, rt = comoving_table(z0, z_max)
    R_max = rt[-1]

    os_ = pick(opa_snaps, zs, z0, z_max, a.kmax)
    opa = []
    for s in os_:
        with h5py.File(os.path.join(a.opa, f"opacity_{s:03d}.hdf5"), "r") as f:
            opa.append((zs[s], f["n_HI"][()], f["n_HeI"][()]))
            BOX = float(f.attrs["BoxSize_cMpc"]); NO = f["n_HI"].shape[0]
            sfm = int(f.attrs.get("sf_mask", 0))
    k0 = int(np.argmin([abs(o[0] - z0) for o in opa]))
    cell_o = BOX / NO
    ds = 0.5 * cell_o
    nst = int(np.ceil(R_max / ds))
    R = (np.arange(nst) + 0.5) * ds
    zj = np.interp(R, rt, zt)
    sc = (1 + zj) / (1 + z0)
    E2 = 0.75 * E_LL_HE
    fac = (1 + zj)**2 * ds * MPC_CM
    wH, wHe = sigma_HI(E2 * sc) * fac, sigma_HeI(E2 * sc) * fac
    fo = np.array([np.argmin([abs(o[0] - zz) for o in opa]) for zz in zj])
    print(f"snap {a.snap} z = {z0:.4f}: rays to R = {R_max:.0f} cMpc (z' = {z_max:.3f}), {nst} steps of {ds:.3f} cMpc; "
          f"opacity {NO}^3 from snapshots {os_} (sf_mask {sfm}); {a.ndir} directions, {a.n_points} points")

    src = {}
    for kind in a.sources:
        if kind == "stars":
            if not a.sfr:
                sys.exit("--sfr needed for stars")
            avail = [s for s in zs if os.path.exists(os.path.join(a.sfr, "StarFormationRate", f"StarFormationRate_{s:03d}.hdf5"))]
            path = lambda s: os.path.join(a.sfr, "StarFormationRate", f"StarFormationRate_{s:03d}.hdf5")
            dset = "StarFormationRate"
        else:
            if not a.render:
                sys.exit("--render needed for agn")
            def path(s):
                for d in a.render:
                    q = os.path.join(d, "LuminosityAGN", f"LuminosityAGN_{s:03d}.hdf5")
                    if os.path.exists(q):
                        return q
                return None
            avail = [s for s in zs if path(s)]
            dset = "LuminosityAGN"
        ss = pick(sorted(avail), zs, z0, z_max, a.kmax)
        boxes = []
        for s in ss:
            with h5py.File(path(s), "r") as f:
                b = np.maximum(f[dset][()].astype(np.float32), 0.0)
            boxes.append((zs[s], b))
        fs = np.array([np.argmin([abs(b[0] - zz) for b in boxes]) for zz in zj])
        src[kind] = (boxes, fs)
        print(f"  {kind}: {boxes[0][1].shape[0]}^3 boxes from snapshots {ss}")

    rng = np.random.default_rng(a.seed)
    dirs = fib_dirs(a.ndir)
    edges = np.array([0, 5, 10, 20, 50, 100, 200, 400, 1e9])
    rbin = np.digitize(R, edges) - 1
    acc = {k: {"ion": [], "num_dir": {al: [] for al in a.alphas}, "num_avg": {al: [] for al in a.alphas},
               "covR_n": np.zeros(len(edges) - 1), "covR_d": np.zeros(len(edges) - 1)} for k in src}
    t0 = time.time()
    for ip in range(a.n_points):
        x0 = rng.uniform(0, BOX, 3)
        # ionized point: every opacity cell touching it below 1% of <n_H> (as the mean-free-path modes)
        n0 = opa[k0][1]                                                       # the target snapshot's field
        corners = x0[None, :] + 0.25 * cell_o * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        ci = np.floor(corners / cell_o).astype(int) % NO
        ionized = bool(np.all(n0[ci[:, 0], ci[:, 1], ci[:, 2]] < 1e-2 * NH_MEAN_COM))
        pos = x0[None, None, :] + R[None, :, None] * dirs[:, None, :]          # (ndir, nst, 3)
        io = (np.floor(pos / cell_o).astype(np.int64)) % NO
        dtau = np.empty((a.ndir, nst))
        for k in range(len(opa)):
            m = fo == k
            if m.any():
                ix = io[:, m]
                dtau[:, m] = opa[k][1][ix[..., 0], ix[..., 1], ix[..., 2]] * wH[m] + opa[k][2][ix[..., 0], ix[..., 1], ix[..., 2]] * wHe[m]
        tau = np.cumsum(dtau, axis=1) - 0.5 * dtau
        T = np.exp(-tau)                                                       # (ndir, nst)
        for kind, (boxes, fs) in src.items():
            Ns = boxes[0][1].shape[0]
            isrc = (np.floor(pos / (BOX / Ns)).astype(np.int64)) % Ns
            eps = np.empty((a.ndir, nst), np.float64)
            for k in range(len(boxes)):
                m = fs == k
                if m.any():
                    ix = isrc[:, m]
                    eps[:, m] = boxes[k][1][ix[..., 0], ix[..., 1], ix[..., 2]]
            eT = (eps * T).mean(axis=0)
            e_m, T_m = eps.mean(axis=0), T.mean(axis=0)
            A = acc[kind]
            A["ion"].append(ionized)
            for al in a.alphas:
                w = sc**(-al) * ds
                A["num_dir"][al].append(float((w * eT).sum()))
                A["num_avg"][al].append(float((w * e_m * T_m).sum()))
            np.add.at(A["covR_n"], rbin, eT * ds)
            np.add.at(A["covR_d"], rbin, e_m * T_m * ds)
        if ip == 9:
            print(f"  ~{(time.time() - t0) / 10 * a.n_points / 60:.1f} min for {a.n_points} points", flush=True)

    for kind, A in acc.items():
        ion = np.array(A["ion"])
        print(f"\n{kind}:  {ion.sum()} of {len(ion)} points ionized")
        for al in a.alphas:
            d, v = np.array(A["num_dir"][al]), np.array(A["num_avg"][al])
            ok = v > 0
            for lab, m in (("ionized", ion & ok), ("all", ok)):
                if m.sum() == 0:
                    continue
                r = d[m] / v[m]
                print(f"  alpha {al:3.1f}, {lab:7s}: J_dir/J_avg  J-weighted {d[m].sum() / v[m].sum():.4f};  "
                      f"per point median {np.median(r):.4f}, 16-84% {np.percentile(r, 16):.4f}-{np.percentile(r, 84):.4f}, "
                      f"1-99% {np.percentile(r, 1):.4f}-{np.percentile(r, 99):.4f}")
        cov = A["covR_n"] / np.where(A["covR_d"] > 0, A["covR_d"], np.nan)
        share = A["covR_d"] / A["covR_d"].sum()
        print("  covariance factor <eps T>/(<eps><T>) by R [cMpc] (share of J_avg):  " +
              "  ".join(f"{edges[i]:.0f}-{edges[i + 1]:.0f}: {cov[i]:.3f} ({share[i]:.2f})"
                        for i in range(len(cov)) if np.isfinite(cov[i]) and edges[i] < R_max))


if __name__ == "__main__":
    main()
