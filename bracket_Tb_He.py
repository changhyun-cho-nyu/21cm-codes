"""
bracket_Tb_He.py
================
How much does the 3.46 cm signal depend on J_alpha^He at all?

J_alpha^He from compute_Jalpha_He.c / compute_Jalpha_AGN.c ignores absorption
by H I and He I in the IGM, so it is an upper limit. The true delta T_b^He lies
between two cases computed here voxel by voxel from the same render:
    upper : x_alpha from the J_alpha^He boxes as they are
    zero  : x_alpha = 0 (no radiative coupling; T_S set by collisions alone)
If the two agree, IGM absorption of the He II Lyman band does not matter.

Physics as in global_snaps_clean.ipynb (compute_temperatures, He part):
    x_alpha = 16 pi^2 T*_He e^2 f_alpha / (9 A10_He T_g m_e c) * S_alpha * J
    S_alpha = exp(-0.803 T_K^(-2/3) (1e-6 tau_GP)^(1/3)), tau_GP = 6e3 x_HeII ((1+z)/7)^1.5
    x_c     = T*_He kappa / (A10_He T_g),  kappa = n_e sqrt(8 k T_K / (pi m_e c^2)) c (14.3 eV / k T_K) a0^2
    T_S     = (1 + x_alpha + x_c) / (1/T_g + (x_alpha + x_c)/T_K)            (T_C = T_K)
    dT_b    = P(z) x_HeII (rho/<rho>) (1 - T_g/T_S),  P(z) = 1e6 c^3 h A10 [3He/H] <n_H> / (32 pi k nu^2 H(z) (1+z)) uK
              (the exact optically thin form, H(z) with Lambda, [3He/H] = 1e-5, as PS_3cm_mpi.c)
x_HeII is per He nucleus; the render's HeII_Fraction is taken per H nucleus
(as means_HeII in the notebook) unless its maximum exceeds 0.2.

The prefactor x_HeII (rho/<rho>) is the same in both cases, so the ratio of the
two signals depends only on the couplings; the absolute normalisation of
dT_b does not affect the conclusion.

Usage
-----
  python bracket_Tb_He.py --render .../All/All_NNN.hdf5 \
         --J .../Jalpha_He_NNN.hdf5 [--J .../Jalpha_AGN_He_NNN.hdf5] [--write out.hdf5]
Each --J file adds to J_alpha^He (stars, AGN); its 3-D dataset is found automatically.
"""
import argparse

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

# constants (cgs), as in PS_3cm_mpi.c
K_B, M_E, C, E_CGS = 1.380649e-16, 9.1093837e-28, 2.99792458e10, 4.80320425e-10
M_P = 1.6735575e-24                          # g, hydrogen atom (as the C codes)
H_PL = 6.62607015e-27
A0 = 5.29177e-9
NU_HE = 8.665649867e9                        # Hz
T_STAR_HE = H_PL * NU_HE / K_B               # K
A10_HE = 1.95434e-12                         # s^-1, 3 (nu_He/nu_21)^3 A_21
HE3_H = 1.0e-5
G_GRAV, MPC_CM = 6.6743e-8, 3.0856776e24
OMEGA_L = 0.6902672
F_ALPHA = 0.4162
X_H, Y_HE = 0.76, 0.24 / (4.0 * 0.76)        # H mass fraction, He/H by number
RHO_CONV = 3.0991811197281837e-22            # code density -> g/cm^3 (comoving); used only if the header lacks the units
OMEGA_M, OMEGA_B, H_LITTLE = 0.30964144, 0.048975, 0.6766

def find_3d(f):
    names = [k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3]
    if len(names) != 1:
        raise SystemExit(f"ERROR: expected one 3-D dataset in {f.filename}, found {names}")
    return names[0]

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render", required=True, help="All_NNN.hdf5 render (Density, Temperature, *_Fraction)")
    p.add_argument("--J", action="append", required=True, help="J_alpha^He file(s), summed")
    p.add_argument("--planes", type=int, default=32)
    p.add_argument("--write", help="optional output file with both dT_b^He boxes [uK]")
    args = p.parse_args()

    fr = h5py.File(args.render, "r")
    z = float(fr["Header"].attrs["Redshift"])
    hdr = fr["Header"].attrs
    rho_conv = (float(hdr["UnitMass_in_g"]) * float(hdr["HubbleParam"])**2 / float(hdr["UnitLength_in_cm"])**3
                if all(k in hdr for k in ("UnitMass_in_g", "UnitLength_in_cm", "HubbleParam")) else RHO_CONV)
    N = fr["Density"].shape[0]
    Jf = [h5py.File(fn, "r") for fn in args.J]
    Jd = [f[find_3d(f)] for f in Jf]
    for f, d in zip(Jf, Jd):
        zj = float(d.attrs["Redshift"]) if "Redshift" in d.attrs else float("nan")
        print(f"J file {f.filename} [{d.name}], z = {zj:.4f}, mean = {d.attrs.get('J_mean', float('nan')):.4e}")
        if abs(zj - z) > 1e-3:
            print(f"  WARNING: J redshift {zj:.4f} differs from render redshift {z:.4f}")
        if d.shape != (N, N, N):
            raise SystemExit(f"ERROR: {f.filename} has shape {d.shape}, render is {N}^3")

    rho_mean = 0.0
    hemax = 0.0
    for i0 in range(0, N, args.planes):
        rho_mean += fr["Density"][i0:i0 + args.planes].astype(np.float64).sum()
        hemax = max(hemax, float(fr["HeII_Fraction"][i0:i0 + args.planes].max()))
    rho_mean /= N**3
    per_he = hemax > 0.2
    print(f"render {args.render}\n  z = {z:.4f}, N = {N}, max HeII_Fraction = {hemax:.3f} "
          f"-> taken per {'He' if per_he else 'H'} nucleus")

    Tg = 2.7255 * (1.0 + z)
    pre_a = 16.0 * np.pi**2 * T_STAR_HE * E_CGS**2 * F_ALPHA / (9.0 * A10_HE * Tg * M_E * C)
    H0 = 100.0 * H_LITTLE * 1e5 / MPC_CM
    Hz = H0 * np.sqrt(OMEGA_M * (1.0 + z)**3 + OMEGA_L)
    nH_mean = X_H * OMEGA_B * 3.0 * H0**2 / (8.0 * np.pi * G_GRAV) * (1.0 + z)**3 / M_P
    amp = 1e6 * C**3 * H_PL * A10_HE * HE3_H * nH_mean / (32.0 * np.pi * K_B * NU_HE**2 * Hz * (1.0 + z))   # uK

    out = h5py.File(args.write, "w") if args.write else None
    if out:
        for name in ("Tb_He_upper", "Tb_He_zero"):
            out.create_dataset(name, (N, N, N), "f4", chunks=(min(16, N), N, N), compression="gzip")
        out.attrs["Redshift"] = z

    s = {k: 0.0 for k in ("up", "zero", "up2", "zero2", "w", "w_rad")}
    for i0 in range(0, N, args.planes):
        sl = slice(i0, min(N, i0 + args.planes))
        rho = fr["Density"][sl].astype(np.float64)
        T = np.maximum(fr["Temperature"][sl].astype(np.float64), 1.0)
        xHII = fr["HII_Fraction"][sl].astype(np.float64)
        fII = fr["HeII_Fraction"][sl].astype(np.float64)
        fIII = fr["HeIII_Fraction"][sl].astype(np.float64)
        if per_he:                                   # convert to per H nucleus
            fII, fIII = fII * Y_HE, fIII * Y_HE
        xHeII = fII / Y_HE                           # per He nucleus
        J = sum(d[sl].astype(np.float64) for d in Jd)
        J = np.maximum(J, 0.0)

        n_H = X_H * rho * rho_conv * (1.0 + z)**3 / M_P          # proper cm^-3
        n_e = n_H * (xHII + fII + 2.0 * fIII)
        kappa = n_e * np.sqrt(8.0 * K_B * T / (np.pi * M_E * C**2)) * C * (2.291112e-11 / (K_B * T)) * A0**2
        x_c = T_STAR_HE * kappa / (A10_HE * Tg)
        tau = 6.0e3 * xHeII * ((1.0 + z) / 7.0)**1.5
        S = np.exp(-0.803 * T**(-2.0 / 3.0) * (1e-6 * tau)**(1.0 / 3.0))
        x_a = pre_a * S * J

        pref = amp * xHeII * rho / rho_mean
        TS_up = (1.0 + x_a + x_c) / (1.0 / Tg + (x_a + x_c) / T)
        TS_0 = (1.0 + x_c) / (1.0 / Tg + x_c / T)
        Tb_up = pref * (1.0 - Tg / TS_up)
        Tb_0 = pref * (1.0 - Tg / TS_0)

        s["up"] += Tb_up.sum(); s["zero"] += Tb_0.sum()
        s["up2"] += (Tb_up**2).sum(); s["zero2"] += (Tb_0**2).sum()
        w = np.abs(Tb_up)                            # where the signal is
        s["w"] += w.sum(); s["w_rad"] += (w * x_a / np.maximum(x_a + x_c, 1e-300)).sum()
        if out:
            out["Tb_He_upper"][sl] = Tb_up.astype(np.float32)
            out["Tb_He_zero"][sl] = Tb_0.astype(np.float32)

    n = float(N)**3
    m_up, m_0 = s["up"] / n, s["zero"] / n
    r_up = np.sqrt(max(s["up2"] / n - m_up**2, 0.0)); r_0 = np.sqrt(max(s["zero2"] / n - m_0**2, 0.0))
    print(f"\nz = {z:.4f}   (J_alpha^He = sum of {len(Jd)} file(s))")
    print(f"  mean dT_b^He : upper {m_up:.4e} uK   zero {m_0:.4e} uK   zero/upper = {m_0 / m_up if m_up else float('nan'):.4f}")
    print(f"  rms  dT_b^He : upper {r_up:.4e} uK   zero {r_0:.4e} uK   zero/upper = {r_0 / r_up if r_up else float('nan'):.4f}")
    print(f"  signal-weighted radiative share x_a/(x_a+x_c) = {s['w_rad'] / s['w'] if s['w'] else float('nan'):.4f}")
    print("  -> the true signal lies between 'zero' and 'upper'; a ratio near 1 means IGM absorption does not matter")
    if out:
        out.close()
        print(f"  wrote {args.write} (Tb_He_upper, Tb_He_zero) for the power-spectrum code")

if __name__ == "__main__":
    main()
