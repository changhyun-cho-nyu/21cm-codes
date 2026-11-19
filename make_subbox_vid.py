"""
make_subbox_vid.py
==================
Sub-volume VIDs for the cosmic-variance figure of the paper (fig:subbox, figs/subbox_VID.png; figure cell of
subvol_VID.ipynb). Each 640^3 map (500 cMpc) is cut into 5^3 = 125 non-overlapping 128^3 sub-boxes of 100 cMpc, and
the VID of the full box and of every sub-box is computed with make_vid.vid_logbins: the estimator of the VID figure
(VID.ipynb), 100 log-spaced bins per side, normalised to the voxels of the map it is computed on.

Inputs (user, 2026-11-19: the figure from the current results):
  H : the production 21 cm map, SPIN_MODEL 1: <maps>/RIN_<snap>.hdf5 (dataset rin), z from <maps>/PS_<snap>.out;
      the figure uses render 177 (<x_HII>_V = 0.1, z = 9.17).
  He: the gas-cell 3.46 cm map (every He statistic from the gas cells, 2026-11-18), found as in make_vid.py
      (make_vid.cell_maps: <root>/cell_test*/RIN_cell_He_NNN.hdf5, else cellmap_He_NNN.hdf5), z from the file;
      the figure uses full snapshot 116 (<x_HeIII>_V = 0.0997, z = 5.00).
Was (old figure): cell_by_cell/hydrogen/Tb_H/RIN_177 and cell_by_cell/helium/Tb_He/RIN_406 (640^3 He), with the
notebook's own copy of the estimator (H positive-side floor 1e-3 mK there, 1e-9 in make_vid.py; --floor reproduces it).

Check: the full-box VID written here must equal the VID figure's file for the same map (VID/H_wf/VID_H_177.csv,
VID/He_cell/VID_He_116.csv), when --ref is given and the floor is make_vid's.

Outputs in <out>/<line>_<snap>/ (written as .part, then renamed):
  VID_full.csv, VID_sub_<ix>_<iy>_<iz>.csv   Tb, vid_prob, vid_phys, side (as make_vid.py)
  stats.csv                                  one row per map (full first), with the make_vid statistics

Usage (from the repository directory on ORCD; ~1 min per map):
  python make_subbox_vid.py --line H  --snap 177 --maps /orcd/data/mvogelsb/004/chcho/Tb_maps_prod/H_real_wf \
         --ref /orcd/data/mvogelsb/004/chcho/VID/H_wf --out /orcd/data/mvogelsb/004/chcho/VID/subbox
  python make_subbox_vid.py --line He --snap 116 --cell-root /orcd/data/mvogelsb/004/chcho \
         --ref /orcd/data/mvogelsb/004/chcho/VID/He_cell --out /orcd/data/mvogelsb/004/chcho/VID/subbox
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

import make_vid
from make_vid import BOX_CMPC, cell_maps, get_redshift, write_vid

N_SPLIT = 5                       # 500 cMpc / 100 cMpc
MAX_DCOUNT = 3                    # voxels per bin allowed to differ from the reference VID (bin-edge round-off)


def voxel_counts(df, nvox):
    """voxels per bin from a VID file: count = vid_prob * N_voxels * width; for log-spaced edges e_i = e_0 q^i the
    centre is c_i = e_i (1+q)/2 and the width e_i (q-1), so width = 2 c (q-1)/(q+1), q = c_{i+1}/c_i (per side)"""
    out = np.zeros(len(df))
    for side in ("neg", "pos"):
        idx = np.flatnonzero(df["side"].values == side)
        if len(idx) == 0:
            continue
        c = np.abs(df["Tb"].values[idx])
        q = c[1] / c[0] if len(c) > 1 else 1.01
        out[idx] = df["vid_prob"].values[idx] * nvox * 2.0 * c * (q - 1.0) / (q + 1.0)
    return out


def load_map(a):
    if a.cell_root:
        cells = cell_maps(a.cell_root, a.line)
        if a.snap not in cells:
            sys.exit(f"ERROR: no gas-cell {a.line} map for full snapshot {a.snap} under {a.cell_root}/cell_test*")
        fn, ds = cells[a.snap]
        with h5py.File(fn, "r") as f:
            return f[ds][()], float(f.attrs["z"]), f"{fn}:{ds}"
    fn = os.path.join(a.maps, f"RIN_{a.snap}.hdf5")
    with h5py.File(fn, "r") as f:
        m = f["rin"][()]
    return m, get_redshift(os.path.join(a.maps, f"PS_{a.snap}.out")), f"{fn}:rin"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--line", choices=["H", "He"], required=True)
    p.add_argument("--snap", type=int, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--maps", help="map directory (RIN_<snap>.hdf5, PS_<snap>.out)")
    g.add_argument("--cell-root", help="gas-cell maps under <root>/cell_test*/")
    p.add_argument("--out", required=True)
    p.add_argument("--ref", help="directory of the VID figure's VIDs (VID_<line>_NNN.csv): the full-box VID must match")
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--floor", type=float, default=None,
                   help=f"positive-side floor (map units); default make_vid's {make_vid.FLOOR:g}")
    a = p.parse_args()
    if a.line == "He" and not a.cell_root:
        sys.exit("ERROR: He from the gas-cell maps only (--cell-root)")
    if a.floor is not None:
        make_vid.FLOOR = a.floor

    m, z, src = load_map(a)
    N = m.shape[0]
    if m.shape != (N,) * 3 or N % N_SPLIT:
        sys.exit(f"ERROR: map shape {m.shape}, need a cube divisible by {N_SPLIT}")
    if not np.all(np.isfinite(m)):
        sys.exit(f"ERROR: {np.sum(~np.isfinite(m))} non-finite voxels")
    n = N // N_SPLIT
    res = BOX_CMPC / N
    out = os.path.join(a.out, f"{a.line}_{a.snap:03d}")
    os.makedirs(out, exist_ok=True)
    print(f"{a.line} snapshot {a.snap}, z = {z:.4f}: {src}, {N}^3; {N_SPLIT}^3 sub-boxes of {n}^3 "
          f"({n * res:.1f} cMpc); positive-side floor {make_vid.FLOOR:g}")

    rows = []
    pos, neg, st = make_vid.vid_logbins(m.ravel(), res, a.bins)
    write_vid(pos, neg, os.path.join(out, "VID_full.csv"))
    rows.append({"box": "full", "snap": a.snap, "z": z, **st})
    for ix in range(N_SPLIT):
        for iy in range(N_SPLIT):
            for iz in range(N_SPLIT):
                v = m[ix * n:(ix + 1) * n, iy * n:(iy + 1) * n, iz * n:(iz + 1) * n].ravel()
                pos, neg, st = make_vid.vid_logbins(v, res, a.bins)
                write_vid(pos, neg, os.path.join(out, f"VID_sub_{ix}_{iy}_{iz}.csv"))
                rows.append({"box": f"{ix}_{iy}_{iz}", "snap": a.snap, "z": z, **st})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "stats.csv.part"), index=False)
    os.replace(os.path.join(out, "stats.csv.part"), os.path.join(out, "stats.csv"))

    sub = df[df["box"] != "full"]
    full = df.iloc[0]
    print(f"  full box: mean {full['mean']:+.5e}, std {full['std']:.5e}, skew {full['skewness']:+.3f}, "
          f"kurt {full['kurtosis']:+.3f}")
    print(f"  {len(sub)} sub-boxes: mean {sub['mean'].min():+.4e} .. {sub['mean'].max():+.4e} "
          f"(rms scatter / full {sub['mean'].std() / abs(full['mean']):.3f}), <sub means> / full "
          f"{sub['mean'].mean() / full['mean']:.6f}; std {sub['std'].min():.4e} .. {sub['std'].max():.4e}")

    if a.ref:
        ref = os.path.join(a.ref, f"VID_{a.line}_{a.snap:03d}.csv")
        if not os.path.exists(ref):
            print(f"  check: no {ref}, full-box VID not compared")
        elif a.floor is not None:
            print("  check: --floor differs from make_vid's: not compared with the VID figure")
        else:
            # Not bit-identical in general: the maps are float32 and numpy's float32 log10 (the bin edges) is CPU-dispatched,
            # so edges computed on another machine can differ by ~1e-6 and move a voxel lying on an edge (2026-11-19:
            # render 177, VID/H_wf from compute nodes vs the login node: edges 2.2e-6, one voxel in the top bin). So:
            # same bins (centres to 1e-5) and the same voxel count in every bin to within MAX_DCOUNT.
            r, h = pd.read_csv(ref), pd.read_csv(os.path.join(out, "VID_full.csv"))
            ok = len(r) == len(h) and list(r["side"]) == list(h["side"])
            if ok:
                dtb = float(np.max(np.abs(r["Tb"] - h["Tb"]) / np.abs(r["Tb"])))
                dc = np.abs(voxel_counts(r, N**3) - voxel_counts(h, N**3))
                ok = dtb < 1e-5 and dc.max() <= MAX_DCOUNT
                print(f"  check: full-box VID vs {ref}: bin centres max rel diff {dtb:.1e}, voxel counts max |diff| "
                      f"{dc.max():.2f} (total {dc.sum():.2f} of {N**3:,}): {'same' if ok else 'DIFFERENT'}")
            else:
                print(f"  check: full-box VID vs {ref}: {len(r)} vs {len(h)} rows or different sides: DIFFERENT")
            if not ok:
                sys.exit("STOP: the full-box VID differs from the VID figure's (different map or estimator)")
    print(f"wrote {out}: VID_full.csv, {len(sub)} VID_sub_*.csv, stats.csv")


if __name__ == "__main__":
    main()
