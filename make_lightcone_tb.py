"""
make_lightcone_tb.py
====================
delta T_b of the 21 cm and 3.46 cm lines on the whole Lumina light cone (640 x 640 pixels, 5896 slices, z 30 ->
2.987), with the production physics (lightcone_tb.cone_tb) and the production J_alpha (stars + AGN) sampled onto the
cone (lightcone_J.cone_J). Settings decided by the user (2026-10-27): time="linear" between J snapshots,
outside="hold" beyond the J snapshots' z range (H: z < 2.9981 and z > 29.4285; He: z < 2.9981 and z > 10.014;
check_cone_outside.py: hold vs J = 0 <= 0.4% at the high ends, J = 0 would be -20% at the low end).

Works in chunks of slices (memory), and resumes: each output file has a per-slice "done" flag.
Output, one file per line in --out: lightcone_dTb_<line>.hdf5 with
  dTb (640, 640, nslices) float32 [mK for H, uK for He], line of sight last, slices in the cone's order (z decreasing);
  mean (nslices): the pixel mean of each slice; z_edges, chi_edges [cMpc/h], z (slice centres); done (nslices);
  attributes: line, unit, time, outside, J directory and tag, CONE_ORIGIN, the light-cone header, git commit.
Same limits as the maps (real space; star-forming gas neutral; 21 cm at z >~ 10 and 3.46 cm resolution-dependent,
here at the cone's pixel size: 0.43 cMpc/h at z = 3 to 0.73 at z = 20, against 0.53 for the maps).

At the end it prints the chi^2 d chi-weighted means of the bands z +- 0.1 that lie inside the processed slices, to
compare with check_lightcone_tb.py (J cone column).

Usage
-----
  python make_lightcone_tb.py --out DIR [--lines H He] [--chunk 128] [--slices I0 I1 | --zrange ZLO ZHI] [--save-J]
"""
import argparse
import os
import subprocess

import h5py
import numpy as np

from lightcone_J import JDIR, cone_J
from lightcone_tb import CONE_ORIGIN, cone_tb, read_cone

BASE = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
CONES = [f"{BASE}/Flagship_new/postprocessing/rlc_640", f"{BASE}/Flagship_below_z_4p75_new/postprocessing/rlc_640"]
UNITS = {"H": "mK", "He": "uK"}
BANDS = (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 11.0, 13.0, 15.0)


def cone_edges(dirs):
    """z and chi [cMpc/h] slice edges of the joined cone (the shared edge once), as read_cone joins them"""
    ze, ce = None, None
    for d in dirs:
        with h5py.File(os.path.join(d, "All.hdf5"), "r") as f:
            z, c = f["Redshifts"][()], f["Distances"][()] / 1e3
        if ze is None:
            ze, ce = z, c
        else:
            if not np.isclose(z[0], ze[-1], rtol=0, atol=1e-9):
                raise ValueError(f"the light cones do not meet: z {ze[-1]} vs {z[0]}")
            ze, ce = np.concatenate([ze, z[1:]]), np.concatenate([ce, c[1:]])
    return ze, ce


def open_out(path, line, shape, ze, ce, hd, a):
    new = not os.path.exists(path)
    f = h5py.File(path, "a")
    ch = tuple(min(64, n) for n in shape)
    if new:
        f.create_dataset("dTb", shape=shape, dtype=np.float32, chunks=ch)
        f.create_dataset("mean", shape=(shape[-1],), dtype=np.float64)
        f.create_dataset("done", data=np.zeros(shape[-1], bool))
        f["z_edges"], f["chi_edges"], f["z"] = ze, ce, 0.5 * (ze[:-1] + ze[1:])
        if a.save_J:
            f.create_dataset("J", shape=shape, dtype=np.float32, chunks=ch)
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        except OSError:
            commit = "unknown"
        f.attrs.update(line=line, unit=UNITS[line], time="linear", outside="hold", jdir=a.jdir, tag=a.tag,
                       cone_origin=np.array(CONE_ORIGIN), commit=commit, cones=";".join(a.cones),
                       J_units="proper photons s^-1 cm^-2 Hz^-1 sr^-1")
        for k, v in hd.items():
            f.attrs[f"header_{k}"] = v
    else:
        if "done" not in f:
            raise ValueError(f"{path} is incomplete (no 'done'); move it away")
        if f["dTb"].shape != shape or not np.allclose(f["z_edges"][()], ze, rtol=0, atol=1e-12):
            raise ValueError(f"{path} exists with a different cone; move it away")
        if a.save_J and "J" not in f:
            raise ValueError(f"{path} was started without --save-J")
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lines", nargs="+", default=["H", "He"], choices=["H", "He"])
    ap.add_argument("--chunk", type=int, default=128, help="slices per chunk")
    ap.add_argument("--slices", type=int, nargs=2, default=None, help="only slices I0 <= i < I1 (a test)")
    ap.add_argument("--zrange", type=float, nargs=2, default=None, help="only the slices lying in ZLO <= z <= ZHI (a test)")
    ap.add_argument("--save-J", action="store_true", help="also store the sampled J (another 9.7 GB per line)")
    ap.add_argument("--cones", nargs="+", default=CONES)
    ap.add_argument("--jdir", default=JDIR)
    ap.add_argument("--tag", default="tr_sfmask")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ze, ce = cone_edges(a.cones)
    ns = len(ze) - 1
    i0, i1 = a.slices if a.slices else (0, ns)
    if a.zrange:
        sel = np.flatnonzero((ze[:-1] <= max(a.zrange)) & (ze[1:] >= min(a.zrange)))
        i0, i1 = int(sel[0]), int(sel[-1]) + 1
    probe = read_cone(a.cones, fields=("Density",), sel=np.s_[:1, :1], zmin=ze[1], zmax=ze[0])
    hd = probe["header"]
    N = int(hd["NumPixels"])
    print(f"cone: {ns} slices, z {ze[0]:.4f} - {ze[-1]:.5f}, {N} x {N} pixels; processing slices {i0} - {i1 - 1}; "
          f"lines {a.lines}; time linear, outside hold; out {a.out}", flush=True)

    files = {line: open_out(os.path.join(a.out, f"lightcone_dTb_{line}.hdf5"), line, (N, N, ns), ze, ce, hd, a)
             for line in a.lines}
    try:
        for c0 in range(i0, i1, a.chunk):
            c1 = min(c0 + a.chunk, i1)
            todo = [line for line in a.lines if not files[line]["done"][c0:c1].all()]
            if not todo:
                continue
            cone = read_cone(a.cones, sel=np.s_[:, :], zmin=ze[c1], zmax=ze[c0])
            if cone["Density"].shape[-1] != c1 - c0 or not np.allclose(cone["z_edges"], ze[c0:c1 + 1], rtol=0, atol=1e-12):
                raise RuntimeError(f"slices {c0} - {c1 - 1}: read_cone returned {cone['Density'].shape[-1]} slices")
            for line in todo:
                J, _ = cone_J(cone, line, np.s_[:, :], time="linear", outside="hold", jdir=a.jdir, tag=a.tag, verbose=False)
                tb = cone_tb(cone, line, J)
                f = files[line]
                f["dTb"][:, :, c0:c1] = tb
                f["mean"][c0:c1] = tb.mean(axis=(0, 1), dtype=np.float64)
                if a.save_J:
                    f["J"][:, :, c0:c1] = J
                f["done"][c0:c1] = True
                f.flush()
                print(f"  slices {c0:4d} - {c1 - 1:4d} (z {ze[c0]:.4f} - {ze[c1]:.4f}) {line}: <dT_b> "
                      f"{float(f['mean'][c0:c1].mean()):.5g} {UNITS[line]}", flush=True)
                del J, tb
            del cone

        chi = 0.5 * (ce[:-1] + ce[1:])
        w = chi**2 * np.abs(np.diff(ce))
        zc = 0.5 * (ze[:-1] + ze[1:])
        for line in a.lines:
            f = files[line]
            done, mean = f["done"][()], f["mean"][()]
            print(f"\n{line}: {done.sum()} / {ns} slices done; band means (chi^2 d chi weighted, slices lying in z +- 0.1):")
            for zb in BANDS:
                m = (ze[:-1] <= zb + 0.1) & (ze[1:] >= zb - 0.1)
                if m.any() and done[m].all():
                    print(f"  z = {zb:5.2f} ({m.sum()} slices, mean z {np.sum(w[m] * zc[m]) / w[m].sum():.4f}): "
                          f"{np.sum(w[m] * mean[m]) / w[m].sum():.5g} {UNITS[line]}")
    finally:
        for f in files.values():
            f.close()


if __name__ == "__main__":
    main()
