"""
check_Tb_maps.py
================
Independent end-to-end check of the delta T_b maps of PS_21cm_mpi.c / PS_3cm_mpi.c.
For a few slabs of planes, delta T_b is recomputed here from the render fields and the
J_alpha files with separate code (astropy constants when available) and compared cell
by cell with the RIN_NNN.hdf5 map the C code wrote. It checks the whole chain at once:
units of the render density, n_H, n_e, x_c, x_alpha (J_alpha units), S_alpha, T_S,
the exact optically thin prefactor, and the signal convention.

Physics (what the C codes are meant to do; T_c = T_K):
  T_S = (1 + x_c + x_a) / (1/T_g + (x_c + x_a)/T_K),  T_g = 2.7255 (1+z) K
  21 cm : x_a = 16 pi^2 T* e^2 f / (27 A10 T_g m_e c) S_a J,  S_a = exp(-0.803 T^-2/3 (1e-6 tau_GP)^1/3),
          tau_GP = 3e5 x_HI ((1+z)/7)^1.5;  x_c = T*/(A10 T_g) (k_HH n_HI + k_eH n_e + k_pH n_p)
          (Furlanetto, Oh & Briggs 2006 tables, log-log interpolation, held beyond the ends)
          dT_b = P21 [x_HI (1+delta)] (1 - T_g/T_S),  [x_HI(1+delta)] = SpectralDensityHI/0.76/<Density>
          (redshift space) or Density (1 - HII_Fraction)/<Density> (real space)
  3.46cm: x_a as above with 9, A10_He, T*_He, tau_GP = 6e3 x_HeII ((1+z)/7)^1.5;
          x_c = T*_He/(A10_He T_g) n_e sqrt(8kT/(pi m_e c^2)) c (14.3 eV/kT) a0^2   (McQuinn & Switzer 2009)
          dT_b = P_He x_HeII (1+delta) (1 - T_g/T_S), real space (or HeII_Fraction/y x SpectralDensity, --he-rs)
  P = tau T_S/(1+z) at <n_H>, exact, with H(z) including Lambda: 3 c^3 h A10 n / (32 pi k nu^2 H) for H,
      c^3 h A10 [3He/H] n / (32 pi k nu^2 H) for 3He+ (inverted levels).

Usage
-----
  python check_Tb_maps.py --line H  --dir <ren_640> --snap 300 --J Jalpha_H_300.hdf5 --J Jalpha_AGN_H_300.hdf5 \\
         --rin Tb_maps_prod/H/RIN_300.hdf5 [--real-space] [--ts-volume] [--tk-ref 1e-3]
  python check_Tb_maps.py --line He --dir <ren_640> --snap 500 --J Jalpha_He_500_tr_sfmask.hdf5 \\
         --J Jalpha_AGN_He_500_tr_sfmask.hdf5 --rin Tb_maps_prod/He/RIN_500.hdf5 [--he-rs] [--ts-volume]
The switches must match how the map was made (H: H_REDSHIFT_SPACE, default redshift space;
He: HE_REDSHIFT_SPACE, run_3cm.sh uses real space; both: TS_VOLUME_WEIGHTED).
Exit status 1 if any compared cell differs by more than --tol relative to its own value.
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

from tb_physics import SRC, X_H, Y_HE, M_H, OMEGA_L_DEFAULT, Line


def field_path(d, name, snap):
    p = os.path.join(d, name, f"{name}_{snap:03d}.hdf5")
    return p if os.path.exists(p) else os.path.join(d, "All", f"All_{snap:03d}.hdf5")


def find_3d(f, prefer=None):
    names = [k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3]
    if prefer and prefer in names:
        return prefer
    if len(names) != 1:
        sys.exit(f"ERROR: expected one 3-D dataset in {f.filename}, found {names}")
    return names[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--line", choices=["H", "He"], required=True)
    p.add_argument("--dir", required=True, help="render directory (<field>/<field>_NNN.hdf5 or All/All_NNN.hdf5)")
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--J", action="append", required=True, help="J_alpha files of the line (stars, AGN), summed")
    p.add_argument("--rin", required=True, help="RIN_NNN.hdf5 written by the C code")
    p.add_argument("--real-space", action="store_true", help="H: map made with -DH_REDSHIFT_SPACE=0")
    p.add_argument("--he-rs", action="store_true", help="He: map made with the code default HE_REDSHIFT_SPACE=1")
    p.add_argument("--ts-volume", action="store_true", help="map made with -DTS_VOLUME_WEIGHTED=1")
    p.add_argument("--tk-ref", type=float, metavar="THRESH",
                   help="H: map made with -DTK_REFERENCE -DTK_REF_XTHRESH=THRESH (no-heating reference: "
                        "T = T_CMB,0 (1+z)^2/(1+124.8) (1+delta)^(2/3) where x_HII < THRESH)")
    p.add_argument("--slabs", type=int, nargs="+", default=[0, 317, 636], help="first plane of each 4-plane slab")
    p.add_argument("--nplanes", type=int, default=4)
    p.add_argument("--tol", type=float, default=1e-4, help="max |C - here| / |here| allowed per cell (float32 maps: ~1e-7)")
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
    rho_unit = float(hd["UnitMass_in_g"]) * h**2 / float(hd["UnitLength_in_cm"])**3   # code density -> g cm^-3 comoving
    N = fld("Density").shape[0]
    print(f"constants: {SRC};  snap {s}, z = {z:.4f}, N = {N}, h = {h}, Om = {om}, Ob = {ob}, OL = {ol}"
          f"{'' if 'OmegaLambda' in hd else ' (OmegaLambda not in header: default)'}")

    # mean gas density over the whole box (the C codes' global_gas)
    rho_mean = 0.0
    for i0 in range(0, N, 32):
        rho_mean += fld("Density")[i0:i0 + 32].astype(np.float64).sum()
    rho_mean /= float(N)**3

    L = Line(a.line, z, h, om, ob, ol)
    unit = L.unit
    print(f"  T* = {L.Tst:.7f} K, A10 = {L.A:.5e} s^-1, prefactor = {L.P:.6e} {unit}, x_alpha const = {L.xa_c:.5e} / (S J)")
    ts = {"HII": "HII_VolumeFraction", "T": "TemperatureVolumeWeighted", "HeII": "HeII_VolumeFraction",
          "HeIII": "HeIII_VolumeFraction"} if a.ts_volume else \
         {"HII": "HII_Fraction", "T": "Temperature", "HeII": "HeII_Fraction", "HeIII": "HeIII_Fraction"}

    Jf = [h5py.File(fn, "r") for fn in a.J]
    Jd = [f[find_3d(f)] for f in Jf]
    rf = h5py.File(a.rin, "r")
    rin = rf[find_3d(rf, "rin")]
    if rin.shape != (N, N, N):
        sys.exit(f"ERROR: {a.rin} has shape {rin.shape}")
    print(f"  map {a.rin}: content = {rf[rin.name].attrs.get('content', b'?')}")

    worst, num, den, cnt = 0.0, 0.0, 0.0, 0
    for i0 in a.slabs:
        sl = slice(i0, min(N, i0 + a.nplanes))
        rho = fld("Density")[sl].astype(np.float64)
        T = fld(ts["T"])[sl].astype(np.float64)
        xHII = fld(ts["HII"])[sl].astype(np.float64)
        fII = fld(ts["HeII"])[sl].astype(np.float64)
        fIII = fld(ts["HeIII"])[sl].astype(np.float64)
        J = sum(dd[sl].astype(np.float64) for dd in Jd)
        nH = X_H * rho * rho_unit * (1.0 + z)**3 / M_H                     # proper cm^-3
        if a.tk_ref is not None:
            if a.line != "H":
                sys.exit("--tk-ref is an option of the 21 cm code only")
            ref = (xHII < a.tk_ref)
            T = np.where(ref, L.t_ref(rho, rho_mean), T)
            print(f"  planes {sl.start}-{sl.stop - 1}: reference T in {ref.mean():.4f} of the cells")
        if a.line == "H":
            if a.real_space:
                sig = rho * (1.0 - fld("HII_Fraction")[sl].astype(np.float64)) / rho_mean
            else:
                sig = fld("SpectralDensityHI")[sl].astype(np.float64) / X_H / rho_mean
        else:
            x_sig = fld("HeII_Fraction")[sl].astype(np.float64) / Y_HE        # mass-weighted for the signal
            dens = fld("SpectralDensity")[sl].astype(np.float64) if a.he_rs else rho
            sig = x_sig * dens / rho_mean
        Tb, xc, xa = L.tb(sig, nH, T, xHII, fII, fIII, J)
        Cm = rin[sl].astype(np.float64)
        diff = np.abs(Cm - Tb)
        rms_here = np.sqrt(np.mean(Tb**2))
        rel = diff / (np.abs(Tb) + 1e-6 * rms_here)                         # per cell, relative to its own value
        print(f"  planes {sl.start}-{sl.stop - 1}: <dT_b> here {Tb.mean():+.6e}, C {Cm.mean():+.6e}  (ratio {Cm.mean() / Tb.mean():.7f});"
              f"  max rel diff per cell = {rel.max():.2e}, max|diff| = {diff.max():.2e} {unit} ({diff.max() / rms_here:.1e} rms);  "
              f"<x_c> {xc.mean():.3e}, <x_a> {xa.mean():.3e}, non-finite here {int((~np.isfinite(Tb)).sum())}")
        worst = max(worst, rel.max())
        num += (Cm * Tb).sum(); den += (Tb * Tb).sum(); cnt += Tb.size
    print(f"overall: {cnt} cells, best-fit C/here = {num / den:.7f}, worst relative diff per cell = {worst:.2e} (tolerance {a.tol:g})")
    if not np.isfinite(worst) or worst > a.tol:
        print("MISMATCH: check that the switches match how the map was made")
        sys.exit(1)
    print("OK: the C map equals the independent calculation")


if __name__ == "__main__":
    main()
