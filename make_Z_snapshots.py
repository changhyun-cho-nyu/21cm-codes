"""
make_Z_snapshots.py
===================
Stellar metallicity per full LUMINA snapshot, as in metallicity.ipynb, but with
the snapshot number kept, the young-star metallicity added, and a test of
whether a single metallicity is good enough for the BPASS yield.

For every snapshot in each run directory (above and below z = 4.75) it reads
snap_NNN.hdf5 (Header: Redshift, Time) and PartType4/PartType4_NNN.hdf5 in
chunks, drops wind particles (GFM_StellarFormationTime <= 0) and computes

  Z_mean_mw    mass-weighted mean Z of ALL stars (Masses)       [old definition]
  Z_young_W    initial-mass-weighted mean birth Z of stars younger than W Myr
               (W = 10, 20, 50, 100): the metallicity of the stars forming now.
  w_00..w_12   (window --window, default 20 Myr) fraction of the young-star
               initial mass at each BPASS metallicity grid point. Each star's
               mass is split between its two neighbouring grid points with the
               log-Z weights that eps_b() uses in the C codes (Z <= Z_grid[0],
               including Z = 0, goes to the first point, Z >= Z_grid[-1] to
               the last), so sum_j w_j eps_b(nu, Z_j) is EXACTLY the
               young-star average of eps_b(nu, Z_p).

Band-ratio check (needs the BPASS file, --bpass)
-------------------------------------------------
For the H (10.2-13.6 eV) and He II (40.8-54.4 eV) Lyman bands, the lifetime
photon yield per baryon N_band(Z) is built from BPASS exactly as in
load_spectrum() of the C codes. Per snapshot:
  Level 1 = N_band(Z_young_W)                 (one metallicity, what the C codes use)
  Level 2 = sum_j w_j N_band(Z_j)             (average over the young stars' metallicities)
  ratio_band = Level 2 / Level 1
A ratio near 1 means one metallicity is enough; a ratio above 1 means the
metallicity spread of the young stars raises the yield (metal-poor stars emit
more, and the arithmetic mean Z under-weights them).

Outputs
-------
  Z_snapshots.csv : one row per snapshot, all runs in one file, ordered from
                    high to low redshift: run, snap, redshift, a, N_star,
                    N_wind, Z_min, Z_max, Z_mean, Z_mean_mw, Z_young_10/20/50/100,
                    N_young_<W>, aform_ok, w_00..w_12, and with --bpass also
                    NH_L1, NH_L2, ratio_H, NHe_L1, NHe_L2, ratio_He
  Z_weights.txt   : "redshift  w_00 ... w_12", read by compute_Jalpha.c and
                    compute_Jalpha_He.c (Z_WEIGHTS_FILE): the yield is the
                    young-star average sum_j w_j eps_b(nu, Z_j).
  Z_history.txt   : "redshift  Z" from --column (default Z_young_20): the single
                    young-star metallicity, for plots and for the paper; the C
                    codes no longer read it.
  Snapshots with no star in the window are left out of both text files.

Usage
-----
  python make_Z_snapshots.py                         # both LUMINA runs, defaults
  python make_Z_snapshots.py --bpass none            # skip the band-ratio check
  python make_Z_snapshots.py --from-csv Z_snapshots.csv --column Z_young_10
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
BPASS_FILE = ("/orcd/data/mvogelsb/004/ozier/ZOOM_XL/sharedData/"
              "bpass_spectra_bin_chab100_v2.2.1.hdf5")
WINDOWS_MYR = (10, 20, 50, 100)
CHUNK = 50_000_000                         # particles read per pass

# LUMINA cosmology (as in compute_Jalpha*.c)
H0_S = 0.6766 * 100.0 * 3.241e-20          # s^-1
OM, OL = 0.30964144, 0.6902672
MYR_S = 3.15576e13

# BPASS conversion, as in load_spectrum() of compute_Jalpha.c
PLANCK, MP_CGS, MSOL2GRAM, GYR2SEC = 6.62607015e-27, 1.6738e-24, 1.989e33, 3.15576e16
NU_LL_H, NU_LL_HE = 3.289e15, 1.31583e16   # Lyman limits [Hz], as in the C codes
BANDS = {"H": (0.75 * NU_LL_H, NU_LL_H), "He": (0.75 * NU_LL_HE, NU_LL_HE)}

def t_of_a(a):
    """Cosmic time [Myr] at scale factor a (flat LCDM, radiation neglected)."""
    return (2.0 / (3.0 * H0_S * np.sqrt(OL)) / MYR_S
            * np.arcsinh(np.sqrt(OL / OM) * np.asarray(a, float)**1.5))

def logZ_split(Z, logZ_grid):
    """Lower grid index j and weight w of the upper point, linear in log Z and
    clamped to the grid, exactly as eps_b() in the C codes: value = (1-w) X_j + w X_{j+1}."""
    lz = np.log10(np.maximum(np.asarray(Z, float), 10**logZ_grid[0]))
    lz = np.clip(lz, logZ_grid[0], logZ_grid[-1])
    j = np.clip(np.searchsorted(logZ_grid, lz) - 1, 0, len(logZ_grid) - 2)
    w = (lz - logZ_grid[j]) / (logZ_grid[j + 1] - logZ_grid[j])
    return j, w

def at_Z(Z, logZ_grid, values):
    """Grid quantity interpolated to metallicity Z (same rule as eps_b())."""
    j, w = logZ_split(Z, logZ_grid)
    return (1.0 - w) * values[j] + w * values[j + 1]

def bpass_band_yields(path):
    """Metallicity grid and lifetime photon yield per baryon in each band,
    N_band[j] = Int_band N_nu(nu, Z_j) dnu, built as in load_spectrum()."""
    with h5py.File(path, "r") as f:
        nf, na, nz = (int(f[k][()]) for k in ("Number_of_freq", "N_age", "N_metallicity"))
        freq = f["freq"][()]
        age = np.maximum(f["age"][()], 1e-6)                  # Gyr
        Zg = f["metallicity"][()]
        Lnu = f["Lnu"][()].reshape(nz, na, nf)                # erg/s/Hz/Msun
    log_age = np.log10(age)
    dlog = log_age[na // 2] - log_age[na // 2 - 1] if na > 1 else 0.1
    hi, lo = 10**(log_age + 0.5 * dlog), 10**(log_age - 0.5 * dlog)
    lo[0] = 0.0                                               # first bin from t = 0
    dt = (hi - lo) * GYR2SEC
    yields = {}
    for band, (nu_lo, nu_hi) in BANDS.items():
        m = (freq >= nu_lo) & (freq <= nu_hi)
        if m.sum() < 2:
            raise ValueError(f"BPASS table has {m.sum()} points in the {band} band")
        rate = np.trapezoid(Lnu[:, :, m] / (PLANCK * freq[m]) * (MP_CGS / MSOL2GRAM), freq[m], axis=2)
        yields[band] = (rate * dt[None, :]).sum(axis=1)       # photons / baryon, per Z
    return Zg, yields

def snapshot_row(run, snapdir, snap, logZ_grid, window):
    with h5py.File(os.path.join(snapdir, f"snap_{snap:03d}.hdf5"), "r") as f:
        z_snap = float(f["Header"].attrs["Redshift"])
        a_snap = float(f["Header"].attrs["Time"])
    t_snap = t_of_a(a_snap)
    windows = sorted(set(WINDOWS_MYR) | {window})

    n_star = n_wind = 0
    s_m = s_mZ = s_Z = 0.0
    z_min, z_max, af_max = np.inf, -np.inf, 0.0
    young = {w: [0.0, 0.0, 0] for w in windows}              # sum m0, sum m0*Z, count
    wgrid = np.zeros(len(logZ_grid))                          # young mass per grid point
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
            for w in windows:
                y = age < w
                young[w][0] += m0[y].sum(); young[w][1] += (m0[y] * Z[y]).sum(); young[w][2] += int(y.sum())
            y = age < window
            if y.any():
                j, wt = logZ_split(Z[y], logZ_grid)
                wgrid += np.bincount(j, weights=m0[y] * (1.0 - wt), minlength=len(logZ_grid))
                wgrid += np.bincount(j + 1, weights=m0[y] * wt, minlength=len(logZ_grid))

    nan = float("nan")
    row = {
        "run": run, "snap": snap, "redshift": z_snap, "a": a_snap,
        "N_star": n_star, "N_wind": n_wind,
        "Z_min": z_min if n_star else nan, "Z_max": z_max if n_star else nan,
        "Z_mean": s_Z / n_star if n_star else nan,
        "Z_mean_mw": s_mZ / s_m if s_m > 0 else nan,
    }
    for w in windows:
        M, MZ, _ = young[w]
        row[f"Z_young_{w}"] = MZ / M if M > 0 else nan
    row[f"N_young_{window}"] = young[window][2]
    row["aform_ok"] = bool(af_max <= a_snap * (1 + 1e-6))
    tot = wgrid.sum()
    for k in range(len(logZ_grid)):
        row[f"w_{k:02d}"] = wgrid[k] / tot if tot > 0 else nan
    if m0key != "GFM_InitialMass":
        print(f"  WARNING snap {snap:03d}: GFM_InitialMass missing, young stars weighted by Masses")
    return row

def add_band_ratios(df, logZ_grid, yields, window):
    """Level 1 (yield at Z_young_W) vs Level 2 (young-star average of the yield)."""
    wcols = [f"w_{k:02d}" for k in range(len(logZ_grid))]
    W = df[wcols].to_numpy()
    Zy = df[f"Z_young_{window}"].to_numpy()
    for band, Nj in yields.items():
        L1 = np.where(np.isfinite(Zy), at_Z(np.nan_to_num(Zy, nan=0.0), logZ_grid, Nj), np.nan)
        L2 = W @ Nj
        df[f"N{band}_L1"], df[f"N{band}_L2"], df[f"ratio_{band}"] = L1, L2, L2 / L1
    return df

def write_history(df, column, path):
    d = df[["redshift", column]].dropna().sort_values("redshift")
    np.savetxt(path, d.to_numpy(), fmt="%.10e",
               header=f"redshift  {column}  (from make_Z_snapshots.py; clamped to the BPASS grid "
                      f"inside the C codes)")
    print(f"wrote {path}: {len(d)} rows from column {column}, z = {d.redshift.min():.3f} .. "
          f"{d.redshift.max():.3f} ({len(df) - len(d)} snapshots without young stars left out)")

def write_weights(df, logZ_grid, window, path):
    wcols = [f"w_{k:02d}" for k in range(len(logZ_grid))]
    d = df[["redshift"] + wcols].dropna().sort_values("redshift")
    np.savetxt(path, d.to_numpy(), fmt="%.8e",
               header=(f"redshift  w_00..w_{len(logZ_grid) - 1:02d}: fraction of the initial mass of stars "
                       f"younger than {window:g} Myr at each BPASS metallicity\n"
                       "Z_grid = " + " ".join(f"{10**x:.4g}" for x in logZ_grid)))
    print(f"wrote {path}: {len(d)} rows")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=None, help="name=snapdir pairs (default: both LUMINA runs)")
    p.add_argument("--bpass", default=BPASS_FILE, help="BPASS table for the band-ratio check ('none' to skip)")
    p.add_argument("--window", type=float, default=20.0, help="young-star window [Myr] for the weights and ratios")
    p.add_argument("--out-csv", default="Z_snapshots.csv")
    p.add_argument("--out-history", default="Z_history.txt")
    p.add_argument("--out-weights", default="Z_weights.txt")
    p.add_argument("--column", default="Z_young_20", help="column written to the Z_history file")
    p.add_argument("--from-csv", default=None, help="skip the particles; rewrite Z_history from this CSV")
    args = p.parse_args()
    window = int(args.window) if float(args.window).is_integer() else args.window

    if args.from_csv:
        write_history(pd.read_csv(args.from_csv), args.column, args.out_history)
        return

    if args.bpass.lower() == "none":
        yields = None
        Zg = np.array([1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3, 1e-2, 1.4e-2, 2e-2, 3e-2, 4e-2])
        print("no BPASS file: weights on the standard BPASS v2.2.1 metallicity grid, no band ratios")
    else:
        Zg, yields = bpass_band_yields(args.bpass)
        print("BPASS Z grid:", " ".join(f"{z:.3g}" for z in Zg))
        for band, Nj in yields.items():
            print(f"  lifetime yield, {band:2s} band [photons/baryon]:", " ".join(f"{x:.4g}" for x in Nj))
    logZ_grid = np.log10(Zg)

    runs = dict(r.split("=", 1) for r in args.runs) if args.runs else RUNS
    rows = []
    for run, snapdir in runs.items():
        snaps = sorted(int(re.search(r"snap_(\d{3})\.hdf5$", q).group(1))
                       for q in glob.glob(os.path.join(snapdir, "snap_*.hdf5")))
        print(f"{run}: {len(snaps)} snapshots in {snapdir}")
        for snap in snaps:
            r = snapshot_row(run, snapdir, snap, logZ_grid, window)
            rows.append(r)
            print(f"  {run:5s} snap {snap:03d}  z = {r['redshift']:7.4f}  N_star = {r['N_star']:>13,d}  "
                  f"Z_mean_mw = {r['Z_mean_mw']:.4e}  Z_young_{window} = {r[f'Z_young_{window}']:.4e}  "
                  f"(N = {r[f'N_young_{window}']:,d})", flush=True)

    df = pd.DataFrame(rows).sort_values("redshift", ascending=False).reset_index(drop=True)
    dup = df["snap"].duplicated(keep=False)
    if dup.any():
        print("WARNING: the same snapshot number appears in more than one run:",
              sorted(df.loc[dup, "snap"].unique()))
    if yields is not None:
        df = add_band_ratios(df, logZ_grid, yields, window)
        print(f"\nBand-ratio check (Level 2 / Level 1, young stars < {window} Myr):")
        print(f"  {'snap':>5} {'z':>8} {'Z_young':>10} {'ratio_H':>8} {'ratio_He':>9}")
        for _, r in df.iterrows():
            print(f"  {r.snap:5d} {r.redshift:8.3f} {r[f'Z_young_{window}']:10.3e} "
                  f"{r.ratio_H:8.3f} {r.ratio_He:9.3f}")
    df.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}: {len(df)} snapshots")
    write_history(df, args.column, args.out_history)
    write_weights(df, logZ_grid, window, args.out_weights)

if __name__ == "__main__":
    main()
