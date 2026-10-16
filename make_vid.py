"""
make_vid.py
===========
Voxel intensity distributions (VIDs) and one-point statistics of the delta T_b maps (RIN_NNN.hdf5 of
PS_21cm_mpi.c / PS_3cm_mpi.c), for the figures of VID.ipynb.

The calculation is that of calculate_vid_logbins in one-stat.ipynb (the old notebook), unchanged: for each map,
  positive voxels: `bins` log-spaced edges from max(min, FLOOR) to max; negative voxels: the same for -dT_b;
  vid_prob = counts / (N_voxels * bin width) (normalised to all voxels, so each side integrates to its voxel
  fraction), vid_phys = counts / (bin width * total volume) [per (map unit) per cMpc^3, voxel = BOX_CMPC / N];
  bin centres = arithmetic mean of the edges; and the statistics (mean, median, variance, std, mad, iqr,
  skewness, excess kurtosis, p05/p16/p84/p95, min, max, f_negative/f_positive/f_zero, cv, bimodality
  BC = (skew^2 + 1)/(kurt + 3)).
z is read from the first line of PS_NNN.out next to the map (as the notebook did).

Outputs in --out:
  VID_<line>_NNN.csv        Tb, vid_prob, vid_phys, side (neg rows first, then pos), as before
  VID_<line>_NNN_stats.csv  one row: snap, z and the statistics
  stats_<line>.csv          (mode table) all snapshots' statistics, sorted by z (high to low)

Usage
-----
  python make_vid.py vid   --line H --maps /orcd/data/mvogelsb/004/chcho/Tb_maps_prod/H_real --out <dir> [--task 0 --ntask 8]
  python make_vid.py table --line H --out <dir>
Snapshots that already have both outputs are skipped (resumable).
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

BOX_CMPC = 500.0          # Lumina box [cMpc]; the notebook used voxel = 500/640 cMpc
FLOOR = 1.0e-9            # positive-side floor of the old notebook (map units)


def get_redshift(ps_file):
    with open(ps_file) as f:
        return float(f.readline().strip())


def vid_logbins(voxels, resolution_cMpc, bins):
    """calculate_vid_logbins of one-stat.ipynb, on an array of voxel values"""
    v = voxels
    med = np.median(v)
    stats = {
        "mean": np.mean(v), "median": med, "variance": np.var(v), "std": np.std(v),
        "mad": np.median(np.abs(v - med)), "iqr": np.subtract(*np.percentile(v, [75, 25])),
        "skewness": skew(v), "kurtosis": kurtosis(v),
        "p05": np.percentile(v, 5), "p16": np.percentile(v, 16), "p84": np.percentile(v, 84), "p95": np.percentile(v, 95),
        "min": v.min(), "max": v.max(),
        "f_negative": float((v < 0).mean()), "f_positive": float((v > 0).mean()), "f_zero": float((v == 0).mean()),
    }
    stats["cv"] = stats["std"] / stats["mean"] if abs(stats["mean"]) > 1e-12 else np.nan
    stats["bimodality"] = (stats["skewness"]**2 + 1.0) / (stats["kurtosis"] + 3.0)
    total_volume = len(v) * resolution_cMpc**3

    def side(x, sign, floor):
        if len(x) == 0:
            return np.array([]), np.array([]), np.array([])
        vmin = max(x.min(), floor) if floor is not None else x.min()
        vmax = x.max()
        if vmin == vmax:
            vmax = vmin * 1.01
        counts, edges = np.histogram(x, bins=np.logspace(np.log10(vmin), np.log10(vmax), bins))
        centers = sign * 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        widths = np.where(widths > 0, widths, 1e-20)
        return centers, counts / (len(v) * widths), counts / (widths * total_volume)

    pos = side(v[v > 0], 1.0, FLOOR)
    neg = side(-v[v < 0], -1.0, None)
    return pos, neg, stats


def write_vid(pos, neg, path):
    df_pos = pd.DataFrame({"Tb": pos[0], "vid_prob": pos[1], "vid_phys": pos[2], "side": "pos"})
    df_neg = pd.DataFrame({"Tb": neg[0], "vid_prob": neg[1], "vid_phys": neg[2], "side": "neg"})
    parts = [d for d in (df_neg, df_pos) if len(d)]
    (pd.concat(parts, ignore_index=True) if parts else df_pos).to_csv(path + ".part", index=False)
    os.replace(path + ".part", path)


def snaps_in(maps):
    out = []
    for f in glob.glob(os.path.join(maps, "RIN_*.hdf5")):
        m = re.search(r"RIN_(\d+)\.hdf5$", f)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def vid(a):
    os.makedirs(a.out, exist_ok=True)
    snaps = [s for s in snaps_in(a.maps) if a.smin <= s <= a.smax][a.task::a.ntask]
    print(f"{a.line}: {len(snaps)} snapshots for task {a.task}/{a.ntask} from {a.maps}")
    nfail = 0
    for s in snaps:
        f_vid = os.path.join(a.out, f"VID_{a.line}_{s:03d}.csv")
        f_st = os.path.join(a.out, f"VID_{a.line}_{s:03d}_stats.csv")
        if os.path.exists(f_vid) and os.path.exists(f_st):
            continue
        ps = os.path.join(a.maps, f"PS_{s:03d}.out")
        if not os.path.exists(ps):
            print(f"  snap {s:03d}: no {ps}, skipped"); nfail += 1; continue
        z = get_redshift(ps)
        with h5py.File(os.path.join(a.maps, f"RIN_{s:03d}.hdf5"), "r") as f:
            m = f["rin"][()]
        N = m.shape[0]
        pos, neg, st = vid_logbins(m.ravel(), BOX_CMPC / N, a.bins)
        write_vid(pos, neg, f_vid)
        row = {"snap": s, "z": z}
        row.update(st)
        pd.DataFrame([row]).to_csv(f_st + ".part", index=False)
        os.replace(f_st + ".part", f_st)
        print(f"  snap {s:03d} z = {z:.4f}: mean {st['mean']:+.4e}, std {st['std']:.4e}, skew {st['skewness']:+.3f}, "
              f"kurt {st['kurtosis']:+.3f}", flush=True)
        del m
    if nfail:
        sys.exit(f"{nfail} snapshots without PS file")


def table(a):
    rows = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(a.out, f"VID_{a.line}_*_stats.csv")))]
    if not rows:
        sys.exit(f"no VID_{a.line}_*_stats.csv in {a.out}")
    df = pd.concat(rows, ignore_index=True).sort_values("z", ascending=False)
    path = os.path.join(a.out, f"stats_{a.line}.csv")
    df.to_csv(path, index=False)
    print(f"wrote {path}: {len(df)} snapshots, z {df['z'].max():.3f}-{df['z'].min():.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("vid")
    q.add_argument("--line", choices=["H", "He"], required=True)
    q.add_argument("--maps", required=True, help="directory with RIN_NNN.hdf5 and PS_NNN.out")
    q.add_argument("--out", required=True)
    q.add_argument("--bins", type=int, default=100, help="bin edges per side (the notebook used 100)")
    q.add_argument("--smin", type=int, default=0)
    q.add_argument("--smax", type=int, default=10**6)
    q.add_argument("--task", type=int, default=0)
    q.add_argument("--ntask", type=int, default=1)
    q = sub.add_parser("table")
    q.add_argument("--line", choices=["H", "He"], required=True)
    q.add_argument("--out", required=True)
    a = p.parse_args()
    vid(a) if a.mode == "vid" else table(a)


if __name__ == "__main__":
    main()
