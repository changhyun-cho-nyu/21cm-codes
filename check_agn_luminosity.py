"""
check_agn_luminosity.py
=======================
Does the rendered LuminosityAGN field (read by compute_Jalpha_AGN.c) hold the
intrinsic bolometric luminosity, or is an escape fraction / obscuration already
applied?

It compares the box total of the render with the same quantity rebuilt from
the black holes of the snapshot at the same redshift:
    L_bol,i = eps_r (1 - eps_f) Mdot_i c^2        (manuscript, Black holes)
In code units (mass/time x velocity^2) this is the code luminosity unit, the
same unit as the render if LUM_UNIT in compute_Jalpha_AGN.c is right.

    R = sum(render) / sum_i L_bol,i
      ~ 1                        -> intrinsic L_bol, nothing applied
      ~ 1 / (1 - eps_f)          -> eps_r Mdot c^2, the (1 - eps_f) factor not applied
      ~ <f_esc>_L (printed)      -> f_esc = 0.3 (L/1e46)^0.07 applied
      below that                 -> obscuration (and/or f_esc) applied as well
A render deposits each black hole's luminosity on the grid, so smoothing does
not change the box total.

Usage
-----
  python check_agn_luminosity.py --render .../LuminosityAGN_NNN.hdf5 \
         --snap ".../snapdir_MMM/snap_MMM.*.hdf5" [--eps-r 0.2 --eps-f 0.1]
Pick a render NNN and snapshot MMM at the same redshift (both are printed).
eps_r and eps_f are read from the snapshot's Parameters group when present.
"""
import argparse
import glob
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

C_KMS = 2.99792458e5
LUM_UNIT_C = 6.4478e36          # erg/s per code unit, as in compute_Jalpha_AGN.c
PARAM_KEYS = {"eps_r": ["BlackHoleRadiativeEfficiency"],
              "eps_f": ["BlackHoleFeedbackFactor", "QuasarFeedbackFactor"]}

def param(f, names):
    for grp in ("Parameters", "Config", "Header"):
        if grp in f:
            for n in names:
                if n in f[grp].attrs:
                    return float(f[grp].attrs[n])
    return None

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render", required=True, help="AGN render file")
    p.add_argument("--dset", default="LuminosityAGN")
    p.add_argument("--snap", required=True, help="snapshot file(s), glob in quotes")
    p.add_argument("--eps-r", type=float, help="radiative efficiency (default: from snapshot)")
    p.add_argument("--eps-f", type=float, help="quasar-mode feedback factor (default: from snapshot)")
    args = p.parse_args()

    with h5py.File(args.render, "r") as f:
        z_ren = float(f["Header"].attrs["Redshift"]) if "Header" in f else float("nan")
        d = f[args.dset]
        L_ren = 0.0
        for i0 in range(0, d.shape[0], 32):
            L_ren += np.clip(d[i0:i0 + 32].astype(np.float64), 0, None).sum()
        attrs = dict(d.attrs)
    print(f"render   : {args.render}\n  z = {z_ren:.4f}, sum({args.dset}) = {L_ren:.6e} code units"
          f" = {L_ren * LUM_UNIT_C:.4e} erg/s" + (f"\n  dataset attributes: {attrs}" if attrs else ""))

    files = sorted(glob.glob(args.snap))
    if not files:
        sys.exit(f"ERROR: no files match {args.snap}")
    mdot, eps_r, eps_f, z_snap, units = [], args.eps_r, args.eps_f, None, {}
    for fn in files:
        with h5py.File(fn, "r") as f:
            if z_snap is None:
                z_snap = float(f["Header"].attrs["Redshift"])
                eps_r = eps_r if eps_r is not None else param(f, PARAM_KEYS["eps_r"])
                eps_f = eps_f if eps_f is not None else param(f, PARAM_KEYS["eps_f"])
                for k in ("UnitMass_in_g", "UnitLength_in_cm", "UnitVelocity_in_cm_per_s"):
                    for grp in ("Parameters", "Header"):
                        if grp in f and k in f[grp].attrs:
                            units[k] = float(f[grp].attrs[k])
            if "PartType5" in f and "BH_Mdot" in f["PartType5"]:
                mdot.append(f["PartType5/BH_Mdot"][()].astype(np.float64))
    if eps_r is None or eps_f is None:
        sys.exit("ERROR: eps_r / eps_f not found in the snapshot; pass --eps-r and --eps-f")
    mdot = np.concatenate(mdot) if mdot else np.zeros(0)
    if mdot.size == 0:
        sys.exit("ERROR: no PartType5/BH_Mdot in the snapshot files")

    lum_unit = LUM_UNIT_C
    if len(units) == 3:
        lum_unit = units["UnitMass_in_g"] * units["UnitVelocity_in_cm_per_s"]**3 / units["UnitLength_in_cm"]
    L_bh = eps_r * (1.0 - eps_f) * np.clip(mdot, 0, None) * C_KMS**2      # code units
    L_cgs = L_bh * lum_unit
    print(f"snapshot : {len(files)} file(s), z = {z_snap:.4f}, {mdot.size} BHs ({np.count_nonzero(mdot > 0)} accreting)")
    print(f"  eps_r = {eps_r:g}, eps_f = {eps_f:g}; code luminosity unit = {lum_unit:.5e} erg/s "
          f"(compute_Jalpha_AGN.c uses {LUM_UNIT_C:.5e})")
    print(f"  sum eps_r (1-eps_f) Mdot c^2 = {L_bh.sum():.6e} code units = {L_cgs.sum():.4e} erg/s")
    if abs(z_ren - z_snap) > 1e-3:
        print(f"  WARNING: render z = {z_ren:.4f} and snapshot z = {z_snap:.4f} differ; pick matching files")

    R = L_ren / L_bh.sum()
    fesc = np.where(L_cgs > 0, 0.3 * (np.maximum(L_cgs, 1e-30) / 1e46)**0.07, 0.0)
    fesc_L = (fesc * L_cgs).sum() / L_cgs.sum()
    print(f"\nR = render / intrinsic L_bol = {R:.4f}")
    print(f"  compare: 1 (intrinsic), {1 / (1 - eps_f):.4f} (no (1-eps_f) factor), "
          f"{fesc_L:.4f} (luminosity-weighted f_esc = 0.3 (L/1e46)^0.07)")
    top = np.sort(L_cgs)[::-1]
    print(f"  brightest BHs [erg/s]: " + " ".join(f"{x:.2e}" for x in top[:5]) +
          f"; top 10 hold {top[:10].sum() / top.sum():.1%} of the total")

if __name__ == "__main__":
    main()
