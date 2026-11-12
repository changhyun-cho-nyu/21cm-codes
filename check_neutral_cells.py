"""
check_neutral_cells.py
======================
Where is the neutral hydrogen of a full snapshot, by proper n_H, split into star-forming gas (neutral by the render
convention) and the rest? Reads the cube cells kept by test_cells_Tb.py scan (cell_test_<snap>/scan_He_<snap>_*.hdf5;
star-forming cells have x_HII = f_HeII = f_HeIII = 0 there).

Made for the discontinuity between renders 617 and 618 (z = 3.35; JALPHA_NOTES 6.1 item 7): the mass-weighted
neutral fraction doubles while the volume-weighted one does not. Compare full snapshots 146 / 150 (before) with
152 / 155 / 158 (after).

  python -u check_neutral_cells.py --snap 152 [--dir /orcd/data/mvogelsb/004/chcho/cell_test_152]
"""
import argparse
import glob
import os

import h5py
import numpy as np

EDGES = [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.106, np.inf]
NH_REF, Z_REF = 1.468e-5, 3.25014       # mean proper n_H at snapshot 152 (decompose_cells_He.py log), scales as (1+z)^3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--dir", default="")
    a = p.parse_args()
    d = a.dir or f"/orcd/data/mvogelsb/004/chcho/cell_test_{a.snap}"
    files = sorted(f for f in glob.glob(os.path.join(d, f"scan_He_{a.snap:03d}_*.hdf5")) if not f.endswith(".part"))
    if not files:
        raise SystemExit(f"no scan files in {d}")
    with h5py.File(files[0], "r") as f:
        z, rm = float(f.attrs["z"]), float(f.attrs["rho_mean"])
    nHbar = NH_REF * ((1 + z) / (1 + Z_REF))**3
    print(f"snapshot {a.snap}, z = {z:.4f}, {len(files)} scan files, mean n_H {nHbar:.3e} cm^-3", flush=True)
    nb = len(EDGES) - 1
    M = np.zeros((4, nb))                                   # gas, neutral, star-forming, neutral non-SF
    ncell = 0
    for t, fn in enumerate(files):
        with h5py.File(fn, "r") as f:
            for name in f:
                if not name.startswith("cube_"):
                    continue
                g = f[name]
                m, rho, x, f2, f3 = (g[k][()].astype(np.float64) for k in ("m", "rho", "xHII", "fII", "fIII"))
                i = np.digitize(rho / rm * nHbar, EDGES) - 1
                sf = (x == 0) & (f2 == 0) & (f3 == 0)
                for r, w in enumerate((m, m * (1 - x), m * sf, m * (1 - x) * ~sf)):
                    M[r] += np.bincount(i, w, nb)
                ncell += len(m)
        print(f"  file {t + 1}/{len(files)}, {ncell:,} cells", flush=True)
    T = M[0].sum()
    print(f"\n{a.snap} z = {z:.4f}: neutral {M[1].sum() / T:.5f} = SF {M[2].sum() / T:.5f} + non-SF {M[3].sum() / T:.5f}")
    print("  n_H bins [cm^-3, proper]:", EDGES)
    for r, lab in enumerate(("gas", "neutral", "SF", "neutral non-SF")):
        print(f"  {lab:15s}", " ".join(f"{v:.3e}" for v in M[r] / T))
    print("  x_HI non-SF   ", " ".join(f"{v:.3e}" for v in M[3] / np.maximum(M[0] - M[2], 1e-300)))


if __name__ == "__main__":
    main()
