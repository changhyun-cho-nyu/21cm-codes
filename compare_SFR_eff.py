"""
compare_SFR_eff.py
==================
Convergence test of the linear-in-time SFR interpolation in make_SFR_eff.py:
compare a run on every snapshot (REF) with a run on every S-th snapshot (TEST,
--stride S), on the snapshots and planes both runs share.

For each band and common snapshot:
  bias = sum(TEST) / sum(REF) - 1          (box-mean change)
  L1   = sum|TEST - REF| / sum(REF)        (voxel-by-voxel change)

If the interpolation error scales as dt^2, the error of the full-cadence run is
about L1 / (S^2 - 1), i.e. L1/3 for --stride 2. Check the scaling with a
--stride 4 run: L1(4) / L1(2) = 5 for dt^2 (3 for dt).

Usage
-----
  python compare_SFR_eff.py REF_DIR TEST_DIR [--bands H He]
"""
import argparse
import glob
import os

import h5py
import numpy as np

from make_SFR_eff import DSET, OUT_PATTERN

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref")
    p.add_argument("test")
    p.add_argument("--bands", nargs="+", default=["H", "He"], choices=["H", "He"])
    args = p.parse_args()

    for band in args.bands:
        pat = OUT_PATTERN[band]
        files = sorted(glob.glob(os.path.join(args.test, pat.replace("{:03d}", "[0-9][0-9][0-9]"))))
        rows = []
        for ft in files:
            fr = os.path.join(args.ref, os.path.relpath(ft, args.test))
            if not os.path.exists(fr):
                continue
            with h5py.File(ft, "r") as a, h5py.File(fr, "r") as b:
                p0, p1 = a[DSET].attrs["plane_range"]
                q0, q1 = b[DSET].attrs["plane_range"]
                if not (q0 <= p0 and p1 <= q1):
                    raise SystemExit(f"ERROR: {fr} does not cover planes {p0}..{p1 - 1}")
                stride = (int(b[DSET].attrs["stride"]), int(a[DSET].attrs["stride"]))
                T = a[DSET][p0:p1].astype(np.float64)
                R = b[DSET][p0:p1].astype(np.float64)
                z = float(a["Header"].attrs["Redshift"])
            sR = R.sum()
            if sR > 0:
                rows.append((z, T.sum() / sR - 1.0, np.abs(T - R).sum() / sR))
        if not rows:
            print(f"{band}: no common snapshots with non-zero SFR_eff")
            continue
        rows = np.array(sorted(rows, reverse=True))
        print(f"{band} band: stride {stride[0]} (REF) vs {stride[1]} (TEST), {len(rows)} common snapshots, "
              f"planes {p0}..{p1 - 1}")
        print("      z      bias        L1")
        ks = sorted({int(np.argmin(np.abs(rows[:, 0] - zc))) for zc in (20, 15, 12, 10, 8, 6, 5, 4, 3)})
        for k in ks:
            print(f"  {rows[k, 0]:6.3f}  {rows[k, 1]:+.2e}  {rows[k, 2]:.2e}")
        print(f"  all snapshots: max |bias| = {np.abs(rows[:, 1]).max():.2e}, "
              f"L1 median = {np.median(rows[:, 2]):.2e}, max = {rows[:, 2].max():.2e} "
              f"(at z = {rows[np.argmax(rows[:, 2]), 0]:.3f})")
        r = stride[1] / stride[0]
        if r > 1:
            print(f"  -> full-cadence error ~ L1 / {r**2 - 1:g} if it scales as dt^2")

if __name__ == "__main__":
    main()
