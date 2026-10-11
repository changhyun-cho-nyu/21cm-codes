"""
make_Z_snapshots.py
===================
Stellar metallicity per full LUMINA snapshot, as in metallicity.ipynb, but with
the snapshot number kept and the young-star metallicity added.

For every snapshot in each run directory (above and below z = 4.75) it reads
snap_NNN.hdf5 (Header: Redshift, Time) and PartType4/PartType4_NNN.hdf5 in
chunks, drops wind particles (GFM_StellarFormationTime <= 0) and computes

  Z_mean_mw    mass-weighted mean Z of ALL stars (Masses)       [old definition]
  Z_young_W    initial-mass-weighted mean birth Z of stars younger than W Myr
               (W = 10, 20, 50, 100). This is the SFR-weighted metallicity of
               the stars forming now, the one that sets the BPASS Lyman-band
               output (most photons are emitted within ~10-20 Myr).

Outputs
-------
  Z_snapshots.csv : one row per snapshot, all runs in one file, ordered from
                    high to low redshift. Columns:
                    run, snap, redshift, a, N_star, N_wind, Z_min, Z_max,
                    Z_mean, Z_mean_mw, Z_young_10, Z_young_20, Z_young_50,
                    Z_young_100, N_young_20, aform_ok
  Z_history.txt   : two columns "redshift  Z" for compute_Jalpha.c and
                    compute_Jalpha_He.c (Z_HISTORY_FILE), from the column
                    chosen with --column (default Z_young_20). Snapshots with
                    no star in that window are left out; the C codes
                    interpolate in log Z between rows and hold the end values.

Usage
-----
  python make_Z_snapshots.py                         # both LUMINA runs, defaults
  python make_Z_snapshots.py --column Z_young_10     # other window for Z_history
  python make_Z_snapshots.py --from-csv Z_snapshots.csv --column Z_young_50
                                                     # rewrite Z_history only
  python make_Z_snapshots.py --runs above=/path/a below=/path/b
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np
import pandas as pd

try:                                       # registers extra HDF5 filters if needed
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

RUNS = {
    "above": "/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots",
    "below": "/orcd/data/mvogelsb/005/Lumina/Lumina_below_z_4p75/snapshots",
}
WINDOWS_MYR = (10, 20, 50, 100)
CHUNK = 50_000_000                         # particles read per pass

# LUMINA cosmology (as in compute_Jalpha*.c)
H0_S = 0.6766 * 100.0 * 3.241e-20          # s^-1
OM, OL = 0.30964144, 0.6902672
MYR_S = 3.15576e13

def t_of_a(a):
    """Cosmic time [Myr] at scale factor a (flat LCDM, radiation neglected)."""
    return (2.0 / (3.0 * H0_S * np.sqrt(OL)) / MYR_S
            * np.arcsinh(np.sqrt(OL / OM) * np.asarray(a, float)**1.5))

def snapshot_row(run, snapdir, snap):
    with h5py.File(os.path.join(snapdir, f"snap_{snap:03d}.hdf5"), "r") as f:
        z_snap = float(f["Header"].attrs["Redshift"])
        a_snap = float(f["Header"].attrs["Time"])
    t_snap = t_of_a(a_snap)

    n_star = n_wind = 0
    s_m = s_mZ = s_Z = 0.0
    z_min, z_max, af_max = np.inf, -np.inf, 0.0
    young = {w: [0.0, 0.0, 0] for w in WINDOWS_MYR}          # sum m0, sum m0*Z, count
    with h5py.File(os.path.join(snapdir, "PartType4", f"PartType4_{snap:03d}.hdf5"), "r") as f:
        n = f["GFM_StellarFormationTime"].shape[0]
        m0key = "GFM_InitialMass" if "GFM_InitialMass" in f else "Masses"
        for i in range(0, n, CHUNK):
            af = f["GFM_StellarFormationTime"][i:i + CHUNK].astype(np.float64)
            ok = af > 0                                      # drop GFM wind particles
            n_wind += int((~ok).sum())
            if not ok.any():
                continue
            af = af[ok]
            Z  = f["GFM_Metallicity"][i:i + CHUNK].astype(np.float64)[ok]
            m  = f["Masses"][i:i + CHUNK].astype(np.float64)[ok]
            m0 = f[m0key][i:i + CHUNK].astype(np.float64)[ok]
            n_star += Z.size
            s_m += m.sum(); s_mZ += (m * Z).sum(); s_Z += Z.sum()
            z_min, z_max = min(z_min, Z.min()), max(z_max, Z.max())
            af_max = max(af_max, af.max())
            age = t_snap - t_of_a(af)                        # Myr
            for w in WINDOWS_MYR:
                y = age < w
                young[w][0] += m0[y].sum(); young[w][1] += (m0[y] * Z[y]).sum(); young[w][2] += int(y.sum())

    nan = float("nan")
    row = {
        "run": run, "snap": snap, "redshift": z_snap, "a": a_snap,
        "N_star": n_star, "N_wind": n_wind,
        "Z_min": z_min if n_star else nan, "Z_max": z_max if n_star else nan,
        "Z_mean": s_Z / n_star if n_star else nan,
        "Z_mean_mw": s_mZ / s_m if s_m > 0 else nan,
    }
    for w in WINDOWS_MYR:
        M, MZ, _ = young[w]
        row[f"Z_young_{w}"] = MZ / M if M > 0 else nan
    row["N_young_20"] = young[20][2]
    row["aform_ok"] = bool(af_max <= a_snap * (1 + 1e-6))
    if m0key != "GFM_InitialMass":
        print(f"  WARNING snap {snap:03d}: GFM_InitialMass missing, young stars weighted by Masses")
    return row

def write_history(df, column, path):
    d = df[["redshift", column]].dropna().sort_values("redshift")
    np.savetxt(path, d.to_numpy(), fmt="%.10e",
               header=f"redshift  {column}  (from make_Z_snapshots.py; clamped to the BPASS grid "
                      f"inside the C codes)")
    print(f"wrote {path}: {len(d)} rows from column {column}, z = {d.redshift.min():.3f} .. "
          f"{d.redshift.max():.3f} ({len(df) - len(d)} snapshots without young stars left out)")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=None, help="name=snapdir pairs (default: both LUMINA runs)")
    p.add_argument("--out-csv", default="Z_snapshots.csv")
    p.add_argument("--out-history", default="Z_history.txt")
    p.add_argument("--column", default="Z_young_20", help="column written to the Z_history file")
    p.add_argument("--from-csv", default=None, help="skip the particles; rewrite Z_history from this CSV")
    args = p.parse_args()

    if args.from_csv:
        write_history(pd.read_csv(args.from_csv), args.column, args.out_history)
        return

    runs = dict(r.split("=", 1) for r in args.runs) if args.runs else RUNS
    rows = []
    for run, snapdir in runs.items():
        snaps = sorted(int(re.search(r"snap_(\d{3})\.hdf5$", q).group(1))
                       for q in glob.glob(os.path.join(snapdir, "snap_*.hdf5")))
        print(f"{run}: {len(snaps)} snapshots in {snapdir}")
        for snap in snaps:
            r = snapshot_row(run, snapdir, snap)
            rows.append(r)
            print(f"  {run:5s} snap {snap:03d}  z = {r['redshift']:7.4f}  N_star = {r['N_star']:>13,d}  "
                  f"Z_mean_mw = {r['Z_mean_mw']:.4e}  Z_young_10 = {r['Z_young_10']:.4e}  "
                  f"Z_young_20 = {r['Z_young_20']:.4e}  (N = {r['N_young_20']:,d})", flush=True)

    df = pd.DataFrame(rows).sort_values("redshift", ascending=False).reset_index(drop=True)
    dup = df["snap"].duplicated(keep=False)
    if dup.any():
        print("WARNING: the same snapshot number appears in more than one run:",
              sorted(df.loc[dup, "snap"].unique()))
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}: {len(df)} snapshots")
    write_history(df, args.column, args.out_history)

if __name__ == "__main__":
    main()
