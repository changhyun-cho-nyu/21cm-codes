"""
check_mfp_outputs.py
====================
Which field made each MFP output, and is it complete? (user, 2026-11-19: can the earlier volume-weighted outputs,
made with the 32-bit build whose source was not kept, be trusted?)

Every start cell with x > 0.5 sends exactly NumDirections (192) rays and each ray is counted once in mfp_hist, so
    sum(mfp_hist) = NumDirections x N(x > 0.5)
holds exactly for the field the run used. For each mfp_NNN.hdf5 this counts N(x > 0.5) in the render of that
snapshot for the volume-weighted field (*_VolumeFraction) and the mass-weighted one (*_Fraction), He divided by y with
the float32 factor of the code, and says which one the ray count matches. Also:
  - wrapped 32-bit bins (negative counts) are restored (+2^32) and reported;
  - with --compare-dir: the same snapshot made by another build (e.g. the 64-bit rerun of mfp/run_mfp.sh with OUT_ROOT)
    must agree bin by bin in mfp_hist and the 2D histograms (the ray tracing is deterministic);
  - the median and 84th percentile of the ray lengths [cMpc].

Usage (ORCD, ~1 min and 2 GB per snapshot; needs hdf5plugin for the compressed renders):
  python mfp/check_mfp_outputs.py --line H  --dir /orcd/data/mvogelsb/004/chcho/mfp-bubbles-thesan/output/LUMINA_hydrogen_Vol
  python mfp/check_mfp_outputs.py --line He --dir .../LUMINA_helium_Vol [--snaps 450 596] [--compare-dir .../mfp_rerun/LUMINA_helium_Vol]
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

R = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
SCALE_HE = np.float32(4.0 * 0.76 / 0.24)          # = float32(12.666667), the factor of both builds


def render_dir(s):
    return f"{R}/Flagship_new/postprocessing/ren_640" if s <= 428 else f"{R}/Flagship_below_z_4p75_new/postprocessing/ren_640"


def set_root(root):
    global R
    R = root


def read_field(ren, name, s):
    p = f"{ren}/{name}/{name}_{s:03d}.hdf5"
    if not os.path.exists(p):
        p = f"{ren}/All/All_{s:03d}.hdf5"
    with h5py.File(p, "r") as f:
        return f[name][()].astype(np.float32)


def hists(path):
    with h5py.File(path, "r") as f:
        H = dict(f["Header"].attrs)
        keys = ("mfp_hist", "mfp_hist_xHII", "mfp_hist_logT", "mfp_hist_logD")
        is32 = {k: f[k].dtype.itemsize == 4 for k in keys}
        out = {k: f[k][()].astype(np.int64) for k in keys}
    wrapped = {k: int((v < 0).sum()) for k, v in out.items()}
    for k, v in out.items():
        if wrapped[k] and not is32[k]:
            sys.exit(f"ERROR: {path}: negative counts in the 64-bit {k} (corrupt file)")
        v[v < 0] += 2**32                 # 32-bit build: one wrap at most (<= 5.0e10 rays in all)
    return H, out, {k: n for k, n in wrapped.items() if n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--line", choices=["H", "He"], required=True)
    ap.add_argument("--dir", required=True, help="directory with mfp_NNN.hdf5")
    ap.add_argument("--snaps", type=int, nargs="*", help="render snapshots (default: all files in --dir)")
    ap.add_argument("--compare-dir", help="directory with the same snapshots from another build")
    ap.add_argument("--render-root", default=R, help="Thesan-XL directory holding the two render trees")
    a = ap.parse_args()
    set_root(a.render_root)

    files = sorted(glob.glob(os.path.join(a.dir, "mfp_*.hdf5")))
    snaps = [int(re.search(r"mfp_(\d+)\.hdf5$", f).group(1)) for f in files]
    if a.snaps:
        snaps = [s for s in snaps if s in a.snaps]
    if not snaps:
        sys.exit(f"no mfp files in {a.dir}")
    fv, fm = (("HII_VolumeFraction", "HII_Fraction") if a.line == "H" else ("HeIII_VolumeFraction", "HeIII_Fraction"))
    sc = np.float32(1.0) if a.line == "H" else SCALE_HE
    bad = 0
    print(f"{'snap':>5} {'z':>7} {'rays/NumDir':>13} {'N(x_V>0.5)':>12} {'N(x_M>0.5)':>12}  field   "
          f"{'median':>8} {'p84':>8} [cMpc]  notes")
    for s in snaps:
        H, h, wrapped = hists(os.path.join(a.dir, f"mfp_{s:03d}.hdf5"))
        nd = int(H["NumDirections"])
        rays = int(h["mfp_hist"].sum())
        ren = render_dir(s)
        nV = int(np.count_nonzero(read_field(ren, fv, s) * sc > np.float32(0.5)))
        nM = int(np.count_nonzero(read_field(ren, fm, s) * sc > np.float32(0.5)))
        ok_v, ok_m = rays == nd * nV, rays == nd * nM
        field = "VOLUME" if ok_v and not ok_m else "mass" if ok_m and not ok_v else "both" if ok_v else "NEITHER"
        notes = []
        if wrapped:
            notes.append(f"restored wrapped bins {wrapped}")
        if rays % nd:
            notes.append(f"ray count not a multiple of {nd}")
        if field != "VOLUME" and field != "both":
            bad += 1
            notes.append(f"rays/NumDir - N(x_V>0.5) = {rays / nd - nV:+.0f}")
        cell = float(H["BoxSize"]) / float(H["HubbleParam"]) / 1e3 / int(H["NumPixels"])
        R_ = (np.arange(len(h["mfp_hist"])) + 0.5) / int(H["NumBinsPerCell"]) * cell
        c = np.cumsum(h["mfp_hist"]) / max(rays, 1)
        med, p84 = R_[np.searchsorted(c, 0.5)], R_[np.searchsorted(c, 0.84)]
        if a.compare_dir:
            q = os.path.join(a.compare_dir, f"mfp_{s:03d}.hdf5")
            if os.path.exists(q):
                _, h2, _ = hists(q)
                same = [k for k in h if np.array_equal(h[k], h2[k])]
                diff = [k for k in h if k not in same]
                if diff:
                    bad += 1
                    d = h["mfp_hist"] - h2["mfp_hist"]
                    notes.append(f"DIFFERS from {q} in {diff} (mfp_hist: {int(np.abs(d).sum())} rays moved)")
                else:
                    notes.append("identical to the comparison build")
        print(f"{s:5d} {float(H['Redshift']):7.3f} {rays / nd:13.0f} {nV:12d} {nM:12d}  {field:7s} {med:8.2f} {p84:8.2f}  "
              + "; ".join(notes), flush=True)
    print(f"\n{len(snaps)} files, {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
