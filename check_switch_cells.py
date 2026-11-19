"""
check_switch_cells.py
=====================
What changes in the gas between full snapshots 121 (z = 4.753, the last of the z > 4.75 run) and 126 (z = 4.495, the
first full snapshot of the z < 4.75 run, where H and He I come from the uniform FG09 UVB and only AGN are RT sources)?
The gas-cell 3.46 cm mean of the gas denser than 1+delta = 10 steps up by 9% there while the IGM part is smooth
(JALPHA_NOTES 7.1, 2026-11-19). The guess to test: the UVB, with no self-shielding before z = 3.35, ionizes the dense
gas more than the stellar RT did, which raises n_e and the collisional coupling. Not checked before this script.

Input: the cube cells kept by test_cells_Tb.py scan for the He run of each snapshot (cell_test_<snap>/scan_He_<snap>_*.hdf5,
64 cubes of 16^3 of the 320^3 pixels, ~0.8% of the volume). The cubes come from a fixed seed, so they are the same
regions at every snapshot. Per cell: m, rho, V, T, xHII, fII, fIII (per H nucleus; star-forming cells set to 0 by the
render convention) and tb (the cell's delta T_b^He [uK], from test_cells_Tb.py's physics).

Per bin of the cell's 1+delta = rho / rho_mean (and separately star-forming gas):
  mass share; non-star-forming gas, mass-weighted: x_HII, x_HeII = fII/y, x_HeIII = fIII/y, n_e/n_H = xHII + fII + 2 fIII,
  T; and the bin's contribution to the cubes' <delta T_b^He> = sum V tb / sum V (all cells), which adds up to the mean.

  python -u check_switch_cells.py scan --snap 126            # writes cell_test_126/switch_cells_126.csv
  python check_switch_cells.py compare --snaps 116 121 126 131
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd

Y_HE = 0.0789                        # n_He / n_H (the per-H He fractions: x_HeII = fII / y), as tb_physics
EDGES = [0, 0.5, 1, 3, 10, 30, 100, 1000, np.inf]
NH_REF, Z_REF = 1.468e-5, 3.25014    # mean proper n_H at full snapshot 152 (as check_neutral_cells.py), ∝ (1+z)^3
ROOT = "/orcd/data/mvogelsb/004/chcho"
LABELS = [f"{lo:g}-{hi:g}" for lo, hi in zip(EDGES[:-1], EDGES[1:])] + ["star-forming"]


def scan(a):
    d = a.dir or f"{ROOT}/cell_test_{a.snap}"
    files = sorted(f for f in glob.glob(os.path.join(d, f"scan_He_{a.snap:03d}_*.hdf5")) if not f.endswith(".part"))
    if not files:
        sys.exit(f"no scan files in {d}")
    with h5py.File(files[0], "r") as f:
        A = dict(f.attrs)
    z, rm = float(A["z"]), float(A["rho_mean"])
    nHbar = NH_REF * ((1 + z) / (1 + Z_REF))**3
    print(f"snapshot {a.snap}, z = {z:.4f}, {len(files)} scan files, cubes: n_cube {A.get('n_cube')}, side "
          f"{A.get('cube_side')}, seed {A.get('seed')}; mean n_H {nHbar:.3e} cm^-3 (proper)", flush=True)
    nb = len(EDGES)                               # density bins + star-forming
    keys = ("m", "m_xHII", "m_xHeII", "m_xHeIII", "m_ne", "m_T", "V_tb")
    S = {k: np.zeros(nb) for k in keys}
    Vtot, ncell, ncube = 0.0, 0, 0
    for t, fn in enumerate(files):
        with h5py.File(fn, "r") as f:
            for name in f:
                if not name.startswith("cube_"):
                    continue
                g = f[name]
                m, rho, V, T, x, f2, f3, tb = (g[k][()].astype(np.float64)
                                               for k in ("m", "rho", "V", "T", "xHII", "fII", "fIII", "tb"))
                sf = (x == 0) & (f2 == 0) & (f3 == 0)
                i = np.where(sf, nb - 1, np.digitize(rho / rm, EDGES) - 1)
                for k, w in (("m", m), ("m_xHII", m * x), ("m_xHeII", m * f2 / Y_HE), ("m_xHeIII", m * f3 / Y_HE),
                             ("m_ne", m * (x + f2 + 2 * f3)), ("m_T", m * T), ("V_tb", V * tb)):
                    S[k] += np.bincount(i, w, nb)
                Vtot += V.sum(); ncell += len(m); ncube += 1
        print(f"  file {t + 1}/{len(files)}: {ncell:,} cells", flush=True)
    if not np.all(np.isfinite([S[k].sum() for k in keys])):
        sys.exit("STOP: non-finite sums")
    M = S["m"]
    df = pd.DataFrame({
        "snap": a.snap, "z": z, "bin": LABELS,
        "nH_lo": [e * nHbar for e in EDGES[:-1]] + [np.nan], "nH_hi": [e * nHbar for e in EDGES[1:]] + [np.nan],
        "mass_share": M / M.sum(),
        "x_HII": S["m_xHII"] / M, "x_HeII": S["m_xHeII"] / M, "x_HeIII": S["m_xHeIII"] / M,
        "ne_per_nH": S["m_ne"] / M, "T": S["m_T"] / M,
        "dTb_contrib_uK": S["V_tb"] / Vtot})
    out = a.out or os.path.join(d, f"switch_cells_{a.snap:03d}.csv")
    df.to_csv(out + ".part", index=False)
    os.replace(out + ".part", out)
    pd.set_option("display.width", 200)
    print(f"\n{a.snap} z = {z:.4f}: {ncell:,} cells in {ncube} cube pieces; cubes' <dT_b^He> = "
          f"{S['V_tb'].sum() / Vtot:.5f} uK (star-forming cells: x fractions are the render convention, 0)")
    print(df.drop(columns=["snap", "z"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"wrote {out}")


def compare(a):
    dfs = []
    for s in a.snaps:
        p = os.path.join(a.root, f"cell_test_{s}", f"switch_cells_{s:03d}.csv")
        if not os.path.exists(p):
            sys.exit(f"no {p}: run 'scan --snap {s}' first")
        dfs.append(pd.read_csv(p))
    zs = [float(d["z"][0]) for d in dfs]
    print("columns: " + "  ".join(f"{s} (z = {z:.3f})" for s, z in zip(a.snaps, zs)))
    for col, fmt in (("mass_share", "{:.4f}"), ("x_HII", "{:.5f}"), ("ne_per_nH", "{:.5f}"), ("x_HeII", "{:.4f}"),
                     ("x_HeIII", "{:.4f}"), ("T", "{:.4g}"), ("dTb_contrib_uK", "{:.5f}")):
        print(f"\n{col}  (1+delta bin; non-star-forming gas except mass_share and dTb_contrib)")
        for j, lab in enumerate(dfs[0]["bin"]):
            v = [d[col][j] for d in dfs]
            print(f"  {lab:>13s}  " + "  ".join(fmt.format(x).rjust(10) for x in v))
    print("\ncubes' <dT_b^He> [uK]: " + "  ".join(f"{d['dTb_contrib_uK'].sum():.5f}" for d in dfs))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("scan")
    q.add_argument("--snap", type=int, required=True)
    q.add_argument("--dir", default="", help=f"default {ROOT}/cell_test_<snap>")
    q.add_argument("--out", default="", help="default <dir>/switch_cells_<snap>.csv")
    q = sub.add_parser("compare")
    q.add_argument("--snaps", type=int, nargs="+", required=True)
    q.add_argument("--root", default=ROOT)
    a = p.parse_args()
    {"scan": scan, "compare": compare}[a.mode](a)


if __name__ == "__main__":
    main()
