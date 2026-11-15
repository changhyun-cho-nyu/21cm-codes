"""
check_wf_variants.py
====================
How much the 21 cm maps change if the Wouthuysen-Field coupling is computed more accurately. For one render
snapshot, delta T_b is recomputed over the whole box, in the production configuration (real space, mass-weighted
spin-temperature inputs, J = stars + AGN), with four prescriptions that add one change at a time:

  V0  production (PS_21cm_mpi.c): S_a = exp(-0.803 T^-2/3 (1e-6 tau)^1/3), tau = 3e5 x_HI ((1+z)/7)^1.5, T_c = T_K
  V1  + exact Gunn-Peterson coefficient: tau = [pi e^2 f lambda / (m_e c)] <n_H>(z) x_HI / H(z)
      (Lumina cosmology, H with Lambda; 1.30 x the 3e5 form at z >= 10)
  V2  + local density: tau = ... x_HI (1 + delta), 1 + delta = rho / <rho> (the gas mean, as the maps)
  V3  + Hirata (2006) S~_a(T_K, T_S, tau) and colour temperature
        T_c^-1 = T_K^-1 + 0.405535 T_K^-1 (T_S^-1 - T_K^-1), T_S solved iteratively (as 21cmFAST)
        S~_a = (1 - 0.0631789/T + 0.115995/T^2 - 0.401403/(T_S T) + 0.336463/(T_S T^2))
               / (1 + 2.98394 xi + 1.53583 xi^2 + 3.85289 xi^3),  xi = (1e-7 tau)^1/3 T^-2/3
        T_S^-1 = (T_g^-1 + x~_a T_c^-1 + x_c T_K^-1) / (1 + x~_a + x_c),  x~_a = 1.7928e11/(1+z) S~_a J
The x_alpha prefactor, x_c, the delta T_b prefactor and the signal x_HI (1+delta) are those of tb_physics.Line
(checked against the C maps by check_Tb_maps.py). Not included: the velocity gradient in tau (n_HI / (dv/dr)), and the
reduced CMB coupling x_CMB = (1 - exp(-tau_21))/tau_21 (Venumadhav+18) that 21cmFAST also puts in T_S. Below
T_K ~ 2 K the Hirata fit is at the edge of its range: the share of the H I there is printed.
Fiducial maps only (the no-heating reference is not treated here).

Output: the volume mean <dT_b> of each variant (the global signal, as plot_global_Tb.py) and, with --ps, the
power spectra Delta^2(k) = k^3 P/(2 pi^2) of each variant in log k bins, the ratios to V0 and the cross-correlation
r(k) of V3 with V0. With --rin, V0 is also compared with the production map (mean ratio should be 1 to ~1e-5).
A CSV per snapshot goes to --out.

Usage
-----
  python check_wf_variants.py --dir <ren_640> --snap 156 --J Jalpha_H_156.hdf5 --J Jalpha_AGN_H_156.hdf5 \\
         --rin Tb_maps_prod/H_real/RIN_156.hdf5 --ps --out wf_variants/
  python check_wf_variants.py --summary wf_variants/      (table of all wf_global*.csv in the directory)
"""
import argparse
import os
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from tb_physics import (C, E_CGS, F_ALPHA, G, K_EH, K_HH, K_PH, M_E, M_H, MPC, OMEGA_L_DEFAULT, T_E, T_HH, X_H,
                        Line, kappa)
from check_Tb_maps import field_path, find_3d

LAMBDA_LYA = 1215.67e-8          # cm
NAMES = ("V0 production", "V1 +exact tau_GP coeff", "V2 +(1+delta) in tau_GP", "V3 +Hirata S~, T_c")


def S_fp(T, tau):
    return np.exp(-0.803 * T**(-2.0 / 3.0) * (1e-6 * tau)**(1.0 / 3.0))


def ts_simple(Tg, T, xa, xc):
    return (1.0 + xc + xa) / (1.0 / Tg + (xc + xa) / T)


def ts_hirata(Tg, T, A, tau, xc, Ts0, tol=1e-9, itmax=200):
    """T_S with Hirata (2006) S~_a and T_c; A = x_alpha prefactor x J. Returns T_S and the number not converged."""
    Ti = 1.0 / T
    xi = (1e-7 * tau)**(1.0 / 3.0) * Ti**(2.0 / 3.0)
    den = 1.0 + 2.98394 * xi + 1.53583 * xi**2 + 3.85289 * xi**3
    Ts = Ts0.copy()
    todo = np.ones(Ts.shape, bool)
    for _ in range(itmax):
        Tsi = 1.0 / Ts
        S = (1.0 - 0.0631789 * Ti + 0.115995 * Ti**2 - 0.401403 * Tsi * Ti + 0.336463 * Tsi * Ti**2) / den
        xa = A * S
        Tci = Ti + 0.405535 * Ti * (Tsi - Ti)
        new = (1.0 + xa + xc) / (1.0 / Tg + xa * Tci + xc * Ti)
        todo = np.abs(new - Ts) > tol * np.abs(Ts)
        Ts = new
        if not todo.any():
            break
    return Ts, int(todo.sum())


def summary(path):
    import glob
    rows = [np.loadtxt(fn, delimiter=",", skiprows=1) for fn in glob.glob(os.path.join(path, "wf_global_*.csv"))]
    if not rows:
        sys.exit(f"no wf_global_*.csv in {path}")
    t = np.array(sorted(rows, key=lambda r: -r[1]))
    print("global <dT_b> [mK], V0 and the change of V1, V2, V3 relative to V0 (mK and %)")
    print("  snap      z        V0        dV1             dV2             dV3")
    for r in t:
        v0 = r[2]
        print(f"  {int(r[0]):4d}  {r[1]:7.3f}  {v0:+9.4f}  " + "  ".join(
            f"{r[2 + i] - v0:+8.4f} ({(r[2 + i] / v0 - 1) * 100 if v0 else 0:+6.2f}%)" for i in (1, 2, 3)))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--summary":
        return summary(sys.argv[2])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="render directory of the snapshot")
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--J", action="append", required=True, help="J_alpha files (stars, AGN), summed")
    p.add_argument("--rin", help="production RIN_NNN.hdf5, to check V0 against")
    p.add_argument("--ps", action="store_true", help="also the power spectra (needs ~20 GB)")
    p.add_argument("--nbins", type=int, default=24)
    p.add_argument("--nplanes", type=int, default=16)
    p.add_argument("--out", default="wf_variants")
    a = p.parse_args()

    d, s = a.dir, a.snap
    files = {}

    def fld(name):
        if name not in files:
            files[name] = h5py.File(field_path(d, name, s), "r")
        return files[name][name]

    with h5py.File(os.path.join(d, "All", f"All_{s:03d}.hdf5"), "r") as f:
        hd = dict(f["Header"].attrs)
    z = float(hd["Redshift"]) if "Redshift" in hd else 1.0 / float(hd["Time"]) - 1.0
    h = float(hd["HubbleParam"])
    om, ob = float(hd.get("Omega0", 0.30964144)), float(hd.get("OmegaBaryon", 0.048975))
    ol = float(hd.get("OmegaLambda", OMEGA_L_DEFAULT))
    rho_unit = float(hd["UnitMass_in_g"]) * h**2 / float(hd["UnitLength_in_cm"])**3
    box = float(hd["BoxSize"])                                         # ckpc/h
    N = fld("Density").shape[0]
    L = Line("H", z, h, om, ob, ol)
    Tg = L.Tg

    H0 = 100.0 * h * 1e5 / MPC
    Hz = H0 * np.sqrt(om * (1.0 + z)**3 + ol)
    nH_mean = X_H * ob * 3.0 * H0**2 / (8.0 * np.pi * G) * (1.0 + z)**3 / M_H
    tau_bar = np.pi * E_CGS**2 * F_ALPHA * LAMBDA_LYA / (M_E * C) * nH_mean / Hz
    tau_old = 3e5 * ((1.0 + z) / 7.0)**1.5
    print(f"snap {s}, z = {z:.4f}, N = {N}; tau_GP(x_HI = 1, mean density): exact {tau_bar:.4e}, code {tau_old:.4e}, "
          f"ratio {tau_bar / tau_old:.4f}")

    rho_mean = sum(fld("Density")[i:i + 64].astype(np.float64).sum() for i in range(0, N, 64)) / float(N)**3

    Jd = [h5py.File(fn, "r") for fn in a.J]
    Jd = [f[find_3d(f)] for f in Jd]

    maps = [np.empty((N, N, N), np.float32) for _ in NAMES]
    nbad, rin_num, rin_den, rin_worst = 0, 0.0, 0.0, 0.0
    xa_sum, nslab, hi_cold, hi_all = 0.0, 0, 0.0, 0.0
    for i0 in range(0, N, a.nplanes):
        sl = slice(i0, min(N, i0 + a.nplanes))
        rho = fld("Density")[sl].astype(np.float64)
        T = fld("Temperature")[sl].astype(np.float64)
        xHII = fld("HII_Fraction")[sl].astype(np.float64)
        fII = fld("HeII_Fraction")[sl].astype(np.float64)
        fIII = fld("HeIII_Fraction")[sl].astype(np.float64)
        J = sum(dd[sl].astype(np.float64) for dd in Jd)
        nH = X_H * rho * rho_unit * (1.0 + z)**3 / M_H
        xHI = 1.0 - xHII
        ne = nH * (xHII + fII + 2.0 * fIII)
        xc = L.Tst / (L.A * Tg) * (kappa(T, T_HH, K_HH) * xHI * nH + kappa(T, T_E, K_EH) * ne
                                   + kappa(T, T_E, K_PH) * xHII * nH)
        A = L.xa_c * J
        sig = L.P * rho * xHI / rho_mean
        one_d = rho / rho_mean

        xa0 = A * S_fp(T, tau_old * xHI)
        Ts = [ts_simple(Tg, T, xa0, xc),
              ts_simple(Tg, T, A * S_fp(T, tau_bar * xHI), xc)]
        tau2 = tau_bar * xHI * one_d
        Ts.append(ts_simple(Tg, T, A * S_fp(T, tau2), xc))
        T3, nb = ts_hirata(Tg, T, A, tau2, xc, Ts[2])
        Ts.append(T3)
        nbad += nb
        for m, t in zip(maps, Ts):
            m[sl] = sig * (1.0 - Tg / t)
        xa_sum += xa0.sum(); nslab += xa0.size
        hi_cold += sig[T < 2.0].sum(); hi_all += sig.sum()        # H I below 2 K: edge of the Hirata fit
        if a.rin:
            with h5py.File(a.rin, "r") as f:
                c = f["rin"][sl].astype(np.float64)
            v0 = maps[0][sl].astype(np.float64)
            rin_num += (c * v0).sum(); rin_den += (v0**2).sum()
            rin_worst = max(rin_worst, float((np.abs(c - v0) / (np.abs(v0) + 1e-6 * np.sqrt(np.mean(v0**2)))).max()))

    if nbad:
        print(f"WARNING: Hirata T_S iteration not converged in {nbad} cells")
    if a.rin:
        r = rin_num / rin_den
        good = abs(r - 1) < 1e-4 and rin_worst < 1e-4
        print(f"V0 vs production map {a.rin}: best-fit ratio {r:.6f}, worst relative difference per cell {rin_worst:.2e} "
              + ("OK" if good else "MISMATCH: the variants are not measured against the production maps"))
    means = [float(m.mean(dtype=np.float64)) for m in maps]
    print(f"<x_alpha> (V0) = {xa_sum / nslab:.4e};  H I (signal) in cells with T_K < 2 K: {hi_cold / hi_all:.4f}")
    print("global signal (volume mean) [mK]:")
    for n, mu in zip(NAMES, means):
        print(f"  {n:26s} {mu:+11.5f}   change {mu - means[0]:+9.5f} mK ({(mu / means[0] - 1) * 100 if means[0] else 0:+7.3f}%)")

    os.makedirs(a.out, exist_ok=True)
    tag = ""
    with open(os.path.join(a.out, f"wf_global{tag}_{s}.csv"), "w") as f:
        f.write("snap,z," + ",".join(f"mean_V{i}" for i in range(len(NAMES))) + "\n")
        f.write(f"{s},{z:.6f}," + ",".join(f"{m:.8e}" for m in means) + "\n")

    if not a.ps:
        return
    kf = 2.0 * np.pi / (box / 1e3)                                     # h/cMpc
    ii = np.fft.fftfreq(N, 1.0 / N)
    k = np.sqrt((ii[:, None, None] * kf)**2 + (ii[None, :, None] * kf)**2
                + (np.arange(N // 2 + 1)[None, None, :] * kf)**2).astype(np.float32)
    edges = np.geomspace(kf, kf * N / 2, a.nbins + 1)
    b = np.digitize(k, edges) - 1
    del k
    ok = (b >= 0) & (b < a.nbins)
    bb = b[ok]
    cnt = np.bincount(bb, minlength=a.nbins).astype(np.float64)
    V = (box / 1e3)**3
    F0 = np.fft.rfftn(maps[0] - means[0])
    P, X = [], None
    for i, m in enumerate(maps):
        F = F0 if i == 0 else np.fft.rfftn(m - means[i])
        P.append(np.bincount(bb, (np.abs(F)**2)[ok], minlength=a.nbins) / np.maximum(cnt, 1) * V / float(N)**6)
        if i == len(maps) - 1:
            X = np.bincount(bb, (F0 * np.conj(F)).real[ok], minlength=a.nbins) / np.maximum(cnt, 1) * V / float(N)**6
        if i:
            del F
    kc = np.sqrt(edges[:-1] * edges[1:])
    D2 = [kc**3 * q / (2 * np.pi**2) for q in P]
    r = X / np.sqrt(P[0] * P[-1])
    print("power spectra: Delta^2 [mK^2] of V0, ratios Vi/V0, r(V0, V3)")
    print("  k [h/cMpc]   Delta2_V0     V1/V0    V2/V0    V3/V0    r")
    rows = []
    for j in range(a.nbins):
        if cnt[j] == 0:
            continue
        rows.append([kc[j], cnt[j]] + [q[j] for q in D2] + [r[j]])
        print(f"  {kc[j]:9.4f}  {D2[0][j]:11.4e}  " + "  ".join(f"{D2[i][j] / D2[0][j]:7.4f}" for i in range(1, len(D2)))
              + f"  {r[j]:7.5f}")
    np.savetxt(os.path.join(a.out, f"wf_ps{tag}_{s}.csv"), np.array(rows), delimiter=",",
               header="k_h_cMpc,modes," + ",".join(f"D2_V{i}" for i in range(len(NAMES))) + ",r_V0_V3", comments="")


if __name__ == "__main__":
    main()
