"""
cell_density_map.py
===================
The gas density at the gas-cell level on the 640^3 grid of the gas-cell 3.46 cm maps (user, 2026-11-19), so that the
density power spectrum and the 3.46 cm power spectrum come from the same cells, pixels and weighting.

From the scan task files of test_cells_Tb.py (scan_<line>_NNN_TTT.hdf5, --grids full: per-pixel sums over the cells
whose centres fall in the pixel, render index order):
    rho_cell(p) = sum m / sum V          (volume-weighted mean of the cells' densities, the weighting of the
                                          delta T_b cell map, sum V dT_b / sum V)
    delta(p)    = rho_cell(p) / <rho_cell> - 1   (<.> = pixel mean, as ps_from_map.py --overdensity)
Also printed, not stored: the NGP mass density sum m / V_pixel (mass-conserving; differs where the cells' volumes do
not fill the pixel) and its correlation with rho_cell. Units: the snapshot's comoving code density.
The task files are checked: --ntask files, one snapshot and axis order, rows contiguous from 0 without gaps or
overlaps (a missing last task is caught by the file count). z and the box size come from the snapshot's
RIN_cell / cellmap file in the same directory (z must agree with the task files).

Output: <out> with dataset "rho" (float32, comoving code density), attrs z, box_ckpc_h, order, snap_full, render_snap,
line, content. Then the spectrum:
    python ps_from_map.py --map <out> --dset rho --overdensity --out PS_cell_delta_NNN.out

Usage
-----
  python cell_density_map.py --scandir /orcd/data/mvogelsb/004/chcho/cell_test_150 --snap 150 \\
      --out /orcd/data/mvogelsb/004/chcho/cell_test_150/cellrho_150.hdf5
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scandir", required=True, help="directory with scan_<line>_NNN_TTT.hdf5")
    ap.add_argument("--snap", type=int, required=True, help="full snapshot")
    ap.add_argument("--line", default="He", help="the scan's line (the grids m and V do not depend on it)")
    ap.add_argument("--ntask", type=int, default=32, help="expected number of task files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.scandir, f"scan_{a.line}_{a.snap:03d}_*.hdf5"))
                   if not f.endswith(".part"))
    if len(files) != a.ntask:
        sys.exit(f"ERROR: {len(files)} task files in {a.scandir}, expected --ntask {a.ntask}")
    with h5py.File(files[0], "r") as f:
        A = dict(f.attrs)
    order = tuple(int(x) for x in A["order"])
    rows, m, V = [], None, None
    for fn in files:
        with h5py.File(fn, "r") as f:
            fa = f.attrs
            if int(fa["snap"]) != a.snap or tuple(int(x) for x in fa["order"]) != order:
                sys.exit(f"ERROR: {fn}: snapshot {int(fa['snap'])} / order {tuple(fa['order'])} differ from the first file")
            if "m" not in f or "V" not in f:
                sys.exit(f"ERROR: {fn} has no m / V grids (scanned with --grids slim?)")
            rows.append(tuple(int(x) for x in fa["rows"]))
            if m is None:
                m = f["m"][()].astype(np.float64); V = f["V"][()].astype(np.float64)
            else:
                m += f["m"][()]; V += f["V"][()]
        print(f"  {os.path.basename(fn)}", flush=True)
    rows.sort()
    gaps = [(rows[i][1], rows[i + 1][0]) for i in range(len(rows) - 1) if rows[i][1] != rows[i + 1][0]]
    if rows[0][0] != 0 or gaps:
        sys.exit(f"ERROR: the task files do not cover the snapshot exactly once: first row {rows[0][0]}, gaps/overlaps {gaps[:5]}")
    n_cells = rows[-1][1]

    N = m.shape[0]
    # the box size is not in the task files; the cell map of the same snapshot has it
    cm = sorted(glob.glob(os.path.join(a.scandir, f"RIN_cell_*_{a.snap:03d}.hdf5")) +
                glob.glob(os.path.join(a.scandir, f"cellmap_*_{a.snap:03d}.hdf5")))
    if not cm:
        sys.exit(f"ERROR: no RIN_cell / cellmap file of snapshot {a.snap} in {a.scandir} (for z and the box size)")
    with h5py.File(cm[0], "r") as f:
        ca = dict(f.attrs)
    box = float(ca.get("box_ckpc_h", ca.get("box")))
    z = float(ca["z"])
    if abs(z - float(A["z"])) > 1e-6:
        sys.exit(f"ERROR: z of {cm[0]} ({z}) differs from the task files ({float(A['z'])})")
    vpix = (box / N)**3

    empty = V <= 0
    rho = np.where(empty, 0.0, m / np.where(empty, 1.0, V))
    ngp = m / vpix
    print(f"\nfull snapshot {a.snap}, z = {z:.4f}, {n_cells:,} cells, N = {N}, box {box:g} ckpc/h, axis order {order}")
    print(f"  gas volume / box volume {V.sum() / box**3:.6f}; empty pixels {int(empty.sum()):,} (rho = 0 there)")
    print(f"  <sum m / sum V> = {rho.mean():.6e}; <sum m / V_pix> = {ngp.mean():.6e} (= total mass / box volume)")
    lr, ln = np.log(rho[~empty]), np.log(np.maximum(ngp[~empty], 1e-300))
    print(f"  corr(log sum m/sum V, log sum m/V_pix) = {np.corrcoef(lr, ln)[0, 1]:.5f}; "
          f"var(delta): {np.var(rho / rho.mean() - 1):.4f} / {np.var(ngp / ngp.mean() - 1):.4f}")
    print(f"  1+delta percentiles (sum m / sum V): 1% {np.percentile(rho, 1) / rho.mean():.3f}, 50% "
          f"{np.median(rho) / rho.mean():.3f}, 99% {np.percentile(rho, 99) / rho.mean():.2f}, max {rho.max() / rho.mean():.1f}")
    del ngp, m, V
    tmp = a.out + ".part"
    with h5py.File(tmp, "w") as f:
        f["rho"] = rho.astype(np.float32)
        f["rho"].attrs["content"] = np.bytes_("gas density at the gas-cell level: sum of cell masses / sum of cell volumes "
                                              "over the cells whose centres fall in the pixel (comoving code units), "
                                              "render index order; cell_density_map.py")
        f.attrs.update({"z": z, "box_ckpc_h": box, "order": order, "snap_full": a.snap,
                        "render_snap": int(A["render_snap"]), "line": a.line, "n_empty_pixels": int(empty.sum())})
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
