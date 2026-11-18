"""
check_J_mean.py
===============
Which of the two box means in the H-band stellar J_alpha files is off, and by how much?
compute_Jalpha.c writes J_mean (mean of the 640^3 map: 200 log shells in R plus the inner shell, then negative voxels
clipped to 0) and J_scalar (step 8: box-mean SFR, midpoint rule over the snapshot redshifts). Production J_mean/J_scalar
is 0.927 at z = 27.2, 0.987-0.993 at z = 12-20, 1.003-1.009 at z = 4-5.6 (JALPHA_NOTES §4.4). Both are quadratures of
the same integral

  <J>(z) = (1+z)^2/(4 pi) sum_{n=2}^{30} f_rec(n) int_z^{z_max(n)} dz' c/H(z') eps(nu'_n, z') <SFR>(z') SFR_UNIT / m_p

with the box mean of the shell windows exact (W(0) = 1), so the mean of the unclipped map is the shell sum with the
box-mean SFR. Computed here, from the same inputs as the C code (the make_SFR_eff.py boxes, BPASS table, Z weights)
and with its constants (C 2.9979e10, m_p 1.6738e-24, Omega_m 0.30964144, Omega_L 0.6902672, h 0.6766) so that only the
quadrature differs:
  scalar_rep : the C code's J_scalar rule, re-implemented.  scalar_rep / J_scalar(file) = 1 checks the integrand here;
  shell_rep  : the C code's shell sum with box-mean SFR (its R grid, its 60-step R_to_z).  J_mean / shell_rep - 1 is
               the clipping shift (+ float32 accumulation);
  conv       : Gauss-Legendre (order 6) on intervals split at every kink of the integrand: snapshot redshifts (SFR
               linear in z between them), BPASS frequency nodes mapped to z' for each line (eps linear in nu), and
               Z-weight rows (linear in z). Each piece is then smooth; conv_lo (order 3) shows the convergence.
<SFR>(z'): box mean of each SFR_eff box with negatives set to 0 (as read_sfr_box), linear in z between snapshots
(interpolating two non-negative boxes and averaging commutes, so this is the C code's box mean exactly).

Usage
-----
  python check_J_mean.py means   --sfr /orcd/data/mvogelsb/004/chcho/SFR_eff/H --csv sfr_means_H.csv     (I/O: 709 boxes)
  python check_J_mean.py compare --csv sfr_means_H.csv --sfr .../SFR_eff/H --jdir .../Jalpha \\
         --spectrum <BPASS hdf5> --weights <Z_weights.txt> [--snaps 5 100 620] [--out J_mean_check.csv]
'means' is resumable (rows already in the CSV are skipped). 'compare' takes seconds.
"""
import argparse
import csv
import glob
import os
import re
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

# constants of compute_Jalpha.c (deliberately the same: only the quadrature is under test)
C_CGS, MSUN_CGS, YR_CGS, MP_CGS, CMPC_CGS, NU_LL = 2.9979e10, 1.989e33, 3.15576e7, 1.6738e-24, 3.0857e24, 3.289e15
OMEGA_M, OMEGA_L, H0_CGS = 0.30964144, 0.6902672, 0.6766 * 100.0 * 3.241e-20
PLANCK, MSOL2GRAM, GYR2SEC = 6.62607015e-27, 1.989e33, 3.15576e16
N_SNAPS, N_MAX, N_SHELLS = 709, 30, 200
F_REC = [0.0, 0.0, 1.0000, 0.0000, 0.2609, 0.3078, 0.3259, 0.3353, 0.3410, 0.3448, 0.3476, 0.3496, 0.3512, 0.3524,
         0.3535, 0.3543, 0.3550, 0.3556, 0.3561, 0.3565, 0.3569, 0.3572, 0.3575, 0.3578, 0.3580, 0.3582, 0.3584,
         0.3586, 0.3587, 0.3589, 0.3590]
DSET = "StarFormationRate"


def H_cgs(z):
    return H0_CGS * np.sqrt(OMEGA_M * (1.0 + z)**3 + OMEGA_L)


def z_max_lyman(z_obs, n):
    return (1.0 + z_obs) * (1.0 - (n + 1.0)**-2) / (1.0 - float(n)**-2) - 1.0


def nu_emitted(z_e, z_obs, n):
    return NU_LL * (1.0 - 1.0 / (n * n)) * (1.0 + z_e) / (1.0 + z_obs)


def comoving_Mpc(z1, z2, nsteps):
    """the C trapezoid, vectorised over z2"""
    z2 = np.asarray(z2, dtype=float)
    dz = (z2 - z1) / nsteps
    i = np.arange(nsteps + 1)
    w = np.where((i == 0) | (i == nsteps), 0.5, 1.0)
    z = z1 + i[None, :] * dz.reshape(-1, 1)
    s = (w * C_CGS / H_cgs(z)).sum(axis=1)
    out = s * dz.ravel() / CMPC_CGS
    return np.where(z2.ravel() <= z1, 0.0, out).reshape(z2.shape)


def R_to_z(z_obs, z_max, R):
    """the C bisection (60 iterations, 60-step distances), vectorised over R"""
    R = np.asarray(R, dtype=float)
    za, zb = np.full(R.shape, z_obs), np.full(R.shape, z_max)
    for _ in range(60):
        zm = 0.5 * (za + zb)
        lo = comoving_Mpc(z_obs, zm, 60) < R
        za, zb = np.where(lo, zm, za), np.where(lo, zb, zm)
    return 0.5 * (za + zb)


# ---------------------------------------------------------------- inputs
def sfr_path(d, s):
    return os.path.join(d, "All", f"All_{s:03d}.hdf5")


def cmd_means(a):
    done = set()
    if os.path.exists(a.csv):
        with open(a.csv) as f:
            done = {int(r["snap"]) for r in csv.DictReader(f)}
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as fo:
        w = csv.writer(fo)
        if new:
            w.writerow(["snap", "z", "mean_floor0", "n_neg", "min", "complete"])
        for s in range(N_SNAPS):
            if s in done:
                continue
            p = sfr_path(a.sfr, s)
            if not os.path.exists(p):
                sys.exit(f"ERROR: missing {p}")
            with h5py.File(p, "r") as f:
                z = float(f["Header"].attrs["Redshift"])
                d = f[DSET]
                comp = int(d.attrs.get("complete", -1))
                tot, nneg, mn = 0.0, 0, np.inf
                for i0 in range(0, d.shape[0], a.planes):
                    x = d[i0:i0 + a.planes]
                    nneg += int((x < 0).sum())
                    mn = min(mn, float(x.min()))
                    tot += np.maximum(x, 0.0).sum(dtype=np.float64)
                mean = tot / float(np.prod(d.shape))
            w.writerow([s, repr(float(z)), repr(float(mean)), nneg, repr(float(mn)), comp])
            fo.flush()
            print(f"  {s:3d}  z = {z:8.4f}  <SFR_eff> = {mean:.6e} Msun/yr/voxel  neg {nneg}  complete {comp}", flush=True)


def read_means(fn):
    with open(fn) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["snap"]))
    snaps = np.array([int(r["snap"]) for r in rows])
    if not np.array_equal(snaps, np.arange(N_SNAPS)):
        sys.exit(f"ERROR: {fn} has {len(rows)} snapshots, need 0..{N_SNAPS - 1} (run 'means' to completion)")
    bad = [int(r["snap"]) for r in rows if int(r["complete"]) not in (1, -1)]
    if bad:
        sys.exit(f"ERROR: SFR_eff boxes not complete: {bad[:10]}")
    z = np.array([float(r["z"]) for r in rows])
    if np.any(np.diff(z) >= 0):
        sys.exit("ERROR: snapshot redshifts not decreasing")
    return z, np.array([float(r["mean_floor0"]) for r in rows])


def sfr_unit(sfr_dir):
    with h5py.File(sfr_path(sfr_dir, 0), "r") as f:
        hd = f["Header"].attrs
        box, h, ul, N = float(hd["BoxSize"]), float(hd["HubbleParam"]), float(hd["UnitLength_in_cm"]), int(hd["NumPixels"])
    cell = box * ul / h / N
    return (MSUN_CGS / YR_CGS) / cell**3, box / h / 1000.0, N


class Emissivity:
    """eps_mix of compute_Jalpha.c: sum_j w_j(z) N_nu(nu, Z_j), N_nu linear in nu (edges held), w linear in z (held)"""
    def __init__(self, spectrum, weights):
        with h5py.File(spectrum, "r") as f:
            Nf, Na, Nm = (int(np.asarray(f[k][()]).ravel()[0]) for k in ("Number_of_freq", "N_age", "N_metallicity"))
            freq, age, met = f["freq"][()].ravel(), f["age"][()].ravel(), f["metallicity"][()].ravel()
            Lnu = f["Lnu"][()].reshape(Nm, Na, Nf)                        # C reads it flat: ((iz*N_age + ia)*Nfreq + ifreq)
        if not np.all(np.diff(freq) > 0):
            sys.exit("ERROR: BPASS freq not increasing: compute_Jalpha.c eps_mix assumes increasing frequencies")
        la = np.log10(np.maximum(age, 1e-6))
        dlog = la[Na // 2] - la[Na // 2 - 1] if Na > 1 else 0.1
        hi = 10.0**(la + 0.5 * dlog)
        lo = np.where(np.arange(Na) == 0, 0.0, 10.0**(la - 0.5 * dlog))
        dt = (hi - lo) * GYR2SEC
        self.freq = freq
        self.N_nu = np.einsum("maf,a->mf", Lnu / (PLANCK * freq) * (MP_CGS / MSOL2GRAM), dt)
        self.logZ = np.log10(met)
        rows = []
        with open(weights) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                v = np.array(line.split(), dtype=float)
                if len(v) != Nm + 1:
                    sys.exit(f"ERROR: {weights}: row with {len(v) - 1} weights, BPASS has {Nm} metallicities")
                if v[1:].sum() > 0:
                    rows.append(np.concatenate([[v[0]], v[1:] / v[1:].sum()]))
        rows = np.array(sorted(rows, key=lambda r: r[0]))
        self.zw_z, self.zw_w = rows[:, 0], rows[:, 1:]

    def weights(self, z):
        return np.stack([np.interp(z, self.zw_z, self.zw_w[:, j]) for j in range(self.zw_w.shape[1])], axis=-1)

    def __call__(self, nu, z):
        w = self.weights(z)
        return sum(w[..., j] * np.interp(nu, self.freq, self.N_nu[j]) for j in range(self.N_nu.shape[0]))


# ---------------------------------------------------------------- the three box means
class BoxMeanJ:
    def __init__(self, zs, sfr, eps, unit, box_mpc, N):
        self.zs, self.sfr, self.eps, self.unit, self.box_mpc, self.N = zs, sfr, eps, unit, box_mpc, N
        self.za, self.sa = zs[::-1], sfr[::-1]                          # ascending for np.interp (ends held, as the C)

    def mean_sfr(self, z):
        return np.interp(z, self.za, self.sa)

    def integrand(self, z, z_obs, n):
        return ((1.0 + z_obs)**2 / (4.0 * np.pi) * F_REC[n] * self.eps(nu_emitted(z, z_obs, n), z)
                * C_CGS / H_cgs(z) * self.mean_sfr(z) * self.unit / MP_CGS)

    def lines(self, z_obs):
        for n in range(2, N_MAX + 1):
            if F_REC[n] == 0.0:
                continue
            zmax = min(z_max_lyman(z_obs, n), self.zs[0])
            if zmax > z_obs:
                yield n, zmax

    def scalar_rep(self, z_obs):
        J = 0.0
        for n, zmax in self.lines(z_obs):
            pts = np.sort(np.concatenate([[z_obs], self.zs[(self.zs > z_obs) & (self.zs <= zmax)], [zmax]]))
            zm, dz = 0.5 * (pts[1:] + pts[:-1]), np.diff(pts)
            J += np.sum(self.integrand(zm, z_obs, n) * dz)
        return J

    def shell_rep(self, z_obs):
        J = 0.0
        for n, zmax in self.lines(z_obs):
            Rmax = float(comoving_Mpc(z_obs, np.array([zmax]), 500)[0])
            Rmin = self.box_mpc / self.N
            Rin = Rmax if Rmax < Rmin else Rmin
            nlog = N_SHELLS if Rmax > Rmin else 0
            dlogR = (np.log(Rmax) - np.log(Rmin)) / nlog if nlog > 0 else 0.0
            i = np.arange(-1, nlog)
            Rlo = np.where(i < 0, 0.0, np.exp(np.log(Rmin) + i * dlogR))
            Rhi = np.where(i < 0, Rin, np.exp(np.log(Rmin) + (i + 1.0) * dlogR))
            zp = R_to_z(z_obs, zmax, 0.5 * (Rlo + Rhi))
            zlo = np.where(i < 0, z_obs, R_to_z(z_obs, zmax, Rlo))
            zhi = np.where(i == nlog - 1, zmax, R_to_z(z_obs, zmax, Rhi))
            J += np.sum(self.integrand(zp, z_obs, n) * (zhi - zlo))
        return J

    def conv(self, z_obs, order):
        x, wq = np.polynomial.legendre.leggauss(order)
        J = 0.0
        for n, zmax in self.lines(z_obs):
            zf = (1.0 + z_obs) * self.eps.freq / (NU_LL * (1.0 - 1.0 / (n * n))) - 1.0   # BPASS nodes in z'
            br = np.concatenate([[z_obs, zmax], self.zs, self.eps.zw_z, zf])
            br = np.unique(br[(br >= z_obs) & (br <= zmax)])
            mid, half = 0.5 * (br[1:] + br[:-1]), 0.5 * np.diff(br)
            z = mid[:, None] + half[:, None] * x[None, :]
            J += np.sum(self.integrand(z, z_obs, n) * wq[None, :] * half[:, None])
        return J


def file_means(jdir, s):
    with h5py.File(os.path.join(jdir, f"Jalpha_H_{s}.hdf5"), "r") as f:
        at = f["J_alpha"].attrs
        return float(at["Redshift"]), float(at["J_mean"]), float(at["J_scalar"])


def cmd_compare(a):
    zs, sfr = read_means(a.csv)
    unit, box_mpc, N = sfr_unit(a.sfr)
    eps = Emissivity(a.spectrum, a.weights)
    B = BoxMeanJ(zs, sfr, eps, unit, box_mpc, N)
    print(f"SFR_UNIT = {unit:.4e} g/s/cm^3 per Msun/yr/voxel, box {box_mpc:.2f} cMpc, N = {N}; "
          f"BPASS {len(eps.freq)} frequencies, {len(eps.logZ)} metallicities; Z weights {len(eps.zw_z)} rows")
    if a.snaps:
        snaps = a.snaps
    else:
        snaps = sorted(int(m.group(1)) for p in glob.glob(os.path.join(a.jdir, "Jalpha_H_*.hdf5"))
                       if (m := re.fullmatch(r"Jalpha_H_(\d+)\.hdf5", os.path.basename(p))))
    rows = []
    print("snap       z    J_mean/conv  J_scalar/conv  shell_rep/conv  J_mean/shell_rep-1  scalar_rep/J_scalar-1  "
          "conv_lo/conv-1")
    for s in snaps:
        zf, jm, js = file_means(a.jdir, s)
        z = zs[s]
        if abs(zf - z) > 1e-6:
            sys.exit(f"ERROR: snapshot {s}: J file z = {zf}, SFR_eff z = {z}")
        sc, sh, cv, cl = B.scalar_rep(z), B.shell_rep(z), B.conv(z, 6), B.conv(z, 3)
        r = dict(snap=s, z=z, J_mean=jm, J_scalar=js, scalar_rep=sc, shell_rep=sh, conv=cv, conv_lo=cl)
        rows.append(r)
        if cv == 0:
            print(f"{s:4d} {z:8.4f}   J = 0 (no emission before z = {zs[0]:.4f} in the SFR inputs); "
                  f"file J_mean {jm:.3e}, J_scalar {js:.3e}")
            continue
        f = lambda x, y: x / y if y else np.nan
        print(f"{s:4d} {z:8.4f}   {f(jm, cv):.6f}     {f(js, cv):.6f}      {f(sh, cv):.6f}       "
              f"{f(jm, sh) - 1:+.3e}          {f(sc, js) - 1:+.3e}            {f(cl, cv) - 1:+.1e}")
    if a.out:
        with open(a.out, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("means", help="box means of the SFR_eff boxes (negatives set to 0) -> CSV, resumable")
    m.add_argument("--sfr", required=True, help="make_SFR_eff.py output for the H band (holds All/All_NNN.hdf5)")
    m.add_argument("--csv", required=True)
    m.add_argument("--planes", type=int, default=64)
    c = sub.add_parser("compare", help="file J_mean, J_scalar against the re-implemented and converged box means")
    c.add_argument("--csv", required=True, help="output of 'means'")
    c.add_argument("--sfr", required=True, help="SFR_eff H directory (Header of snapshot 0 for SFR_UNIT)")
    c.add_argument("--jdir", required=True, help="J directory with Jalpha_H_NNN.hdf5")
    c.add_argument("--spectrum", required=True, help="BPASS table the C code was compiled with (SPECTRUM_FILE)")
    c.add_argument("--weights", required=True, help="Z weights file the C code was compiled with (Z_WEIGHTS_FILE)")
    c.add_argument("--snaps", type=int, nargs="+", help="default: every Jalpha_H_NNN.hdf5 in --jdir")
    c.add_argument("--out", help="CSV of all values")
    a = p.parse_args()
    cmd_means(a) if a.cmd == "means" else cmd_compare(a)


if __name__ == "__main__":
    main()
