#!/usr/bin/env python3
"""
assemble_renders.py
===================
Rebuild the 3D render cubes of the SonjaXrayPaper runs (50 cMpc, Lumina model, with / without MRT_XRAY) in the
layout the pipeline codes read (PS_21cm_mpi.c, compute_Jalpha*.c, make_SFR_eff.py): <out>/All/All_NNN.hdf5 with
/Header (attributes copied from the render header) and one N^3 float32 dataset per field at the root.

Input (read only): <renderings>/render_NNN/render_NNN.<i>.hdf5, i = 0 ... NumFiles-1. Each file holds a
contiguous piece of the flat pixel list (N^3 = 128^3 pixels in total; only file 0 has the Header). Fields are
(n,) per pixel, (n, 3) / (n, 6) vectors, or (n, 8) for the 2x2x2 sub-cells of Density, HII_Fraction,
SpectralDensityHI, IonEnergy. Sub-cell fields are reduced to N^3 here:
  Density, SpectralDensityHI   mean of the 8 sub-cells (equal sub-cell volumes)
  HII_Fraction                 sum(rho_i x_i) / sum(rho_i)  (mass-weighted, as the pixel field is defined)
The flat list is assumed to be the C-order pixel cube; the checks below test it.

Checks (one CSV row per render, <out>/checks/part_TT.csv):
  mass        mean comoving gas density / (Omega_b rho_crit,0), header Omega_b and h (Lumina: 1.0042)
  spec        mean SpectralDensity / mean Density (the redshift-space remap conserves mass: 1 to float precision)
  hi          sum SpectralDensityHI / (X sum Density (1 - HII_Fraction)) over the sub-cells (Lumina: 1 to 1e-5)
  c0 c1 c2    correlation of log Density with its neighbour along axis 0, 1, 2 (a wrong layout gives ~0 along
              some axis; the right one gives similar, high values on all three)
  losK        corr(SpectralDensity - Density, -d_K(rho v_K)) for K = 0, 1, 2: the remap axis (Lumina: +z = axis 2)
              has a large positive value, the others ~0
  vs_axis     the K with the largest losK

Usage (one run per call; tasks split the renders):
  python assemble_renders.py --src <run>/output/movie/renderings --out /orcd/data/mvogelsb/004/chcho/xray/<run>/ren_128 \
      [--snaps 0-404] [--task 0 --ntask 8] [--fields default|all|F1,F2,...] [--overwrite]
  python assemble_renders.py report --out <dir>     (merges and summarises the check CSVs)
"""
import argparse
import csv
import glob
import os
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

X_H = 0.76
RHO_CRIT0_H2 = 1.87834e-29          # g cm^-3 / h^2  (3 H0^2 / 8 pi G with H0 = 100 h km/s/Mpc, G = 6.6743e-8)
DEFAULT_FIELDS = ["Density", "HII_Fraction", "HII_VolumeFraction", "Temperature", "TemperatureVolumeWeighted",
                  "HeII_Fraction", "HeII_VolumeFraction", "HeIII_Fraction", "HeIII_VolumeFraction",
                  "StarFormationRate", "SpectralDensity", "SpectralDensityHI", "LuminosityAGN",
                  "LyaLuminosityCol", "LyaLuminosityRec", "Volume"]
NEED_FOR_CHECKS = ["Density", "HII_Fraction", "SpectralDensity", "SpectralDensityHI", "Velocities"]
OUT_ROOT = "/orcd/data/mvogelsb/004/chcho"   # the only place outputs may go (user rule, 2026-11-15); inputs are read only
COLS = ["render", "z", "mass", "spec", "hi", "c0", "c1", "c2", "los0", "los1", "los2", "vs_axis"]


def parse_snaps(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def render_files(src, n):
    fs = glob.glob(f"{src}/render_{n:03d}/render_{n:03d}.*.hdf5")
    return sorted(fs, key=lambda p: int(p.rsplit(".", 2)[-2]))


def read_render(files, fields):
    """Concatenate each field over the pieces; return header attrs, header datasets, field arrays, field attrs."""
    with h5py.File(files[0], "r") as f:
        hdr = dict(f["Header"].attrs)
        hdr_ds = {k: f["Header"][k][()] for k in f["Header"]}
        nfiles = int(hdr["NumFiles"])
        avail = [k for k in f if isinstance(f[k], h5py.Dataset)]
        attrs = {k: dict(f[k].attrs) for k in avail}
    if len(files) != nfiles:
        raise RuntimeError(f"{len(files)} files, header NumFiles = {nfiles}")
    missing = [k for k in fields if k not in avail]
    if missing:
        raise RuntimeError(f"fields not in the render: {missing}")
    parts = {k: [] for k in fields}
    for p in files:
        with h5py.File(p, "r") as f:
            for k in fields:
                parts[k].append(f[k][()])
    return hdr, hdr_ds, {k: np.concatenate(v) for k, v in parts.items()}, attrs


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def assemble_one(src, out, n, fields, overwrite):
    dst = f"{out}/All/All_{n:03d}.hdf5"
    if os.path.exists(dst) and not overwrite:
        return None
    files = render_files(src, n)
    if not files:
        raise RuntimeError(f"no files for render {n}")
    read = list(dict.fromkeys(fields + NEED_FOR_CHECKS))
    hdr, hdr_ds, raw, attrs = read_render(files, read)
    N = int(hdr["NumPixels"])
    npix = N ** 3
    for k, v in raw.items():
        if v.shape[0] != npix:
            raise RuntimeError(f"{k}: {v.shape[0]} pixels, expected {npix}")

    # sub-cell fields -> pixel fields (keep the sub-cell values for the hi check)
    rho8 = raw["Density"].astype(np.float64)
    x8 = raw["HII_Fraction"].astype(np.float64)
    hi8 = raw["SpectralDensityHI"].astype(np.float64)
    if rho8.ndim != 2 or rho8.shape[1] != 8:
        raise RuntimeError(f"Density has shape {rho8.shape}, expected (N^3, 8)")
    cube = {}
    for k, v in raw.items():
        if v.ndim == 2 and v.shape[1] == 8:
            if k == "HII_Fraction":
                v = (rho8 * x8).sum(1) / rho8.sum(1)
            else:
                v = v.astype(np.float64).mean(1)
        cube[k] = np.asarray(v).reshape((N, N, N) + v.shape[1:])

    # checks
    z = float(hdr["Redshift"])
    to_cgs = float(attrs["Density"]["to_cgs"])   # code density x to_cgs x h^2 = g/cm^3 (comoving); rho_crit has h^2 too
    rho = cube["Density"]
    mass = rho.mean() * to_cgs / (float(hdr["OmegaBaryon"]) * RHO_CRIT0_H2)
    spec = cube["SpectralDensity"].mean() / rho.mean()
    hi = hi8.sum() / (X_H * (rho8 * (1.0 - x8)).sum())
    lr = np.log(rho)
    cs = [corr(lr, np.roll(lr, 1, axis=a)) for a in range(3)]
    d = cube["SpectralDensity"] - rho
    vel = cube["Velocities"].astype(np.float64)
    los = [corr(d, -(np.roll(rho * vel[..., K], -1, K) - np.roll(rho * vel[..., K], 1, K))) for K in range(3)]
    row = dict(render=n, z=z, mass=mass, spec=spec, hi=hi, c0=cs[0], c1=cs[1], c2=cs[2],
               los0=los[0], los1=los[1], los2=los[2], vs_axis=int(np.argmax(los)))

    # write (to a temporary name, then rename: a killed job leaves no half file)
    os.makedirs(f"{out}/All", exist_ok=True)
    tmp = dst + ".tmp"
    with h5py.File(tmp, "w") as f:
        g = f.create_group("Header")
        for k, v in hdr.items():
            g.attrs[k] = v
        g.attrs["NumPixels"] = np.int32(N)
        for k, v in hdr_ds.items():
            g[k] = v
        g.attrs["assembled_from"] = np.bytes_(os.path.realpath(f"{src}/render_{n:03d}"))
        g.attrs["subcell_reduction"] = np.bytes_("Density, SpectralDensityHI: mean of 8; HII_Fraction: mass-weighted")
        for k in fields:
            ds = f.create_dataset(k, data=cube[k].astype(np.float32))
            for a, v in attrs[k].items():
                ds.attrs[a] = v
            ds.attrs["Redshift"] = z
    os.replace(tmp, dst)
    return row


def report(out):
    rows = []
    for p in sorted(glob.glob(f"{out}/checks/part_*.csv")):
        with open(p) as fp:
            rows += list(csv.DictReader(fp))
    if not rows:
        sys.exit(f"no check files in {out}/checks")
    rows.sort(key=lambda r: int(r["render"]))
    have = sorted(int(os.path.basename(p)[4:7]) for p in glob.glob(f"{out}/All/All_*.hdf5"))
    print(f"{len(rows)} renders checked, {len(have)} All files ({have[0]}-{have[-1]})")
    gaps = sorted(set(range(have[0], have[-1] + 1)) - set(have))
    print(f"missing All files: {gaps if gaps else 'none'}")
    for c in COLS[2:]:
        v = np.array([float(r[c]) for r in rows])
        print(f"  {c:8s} min {v.min():.6g}  median {np.median(v):.6g}  max {v.max():.6g}")
    with open(f"{out}/checks/all.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}/checks/all.csv")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        p = argparse.ArgumentParser()
        p.add_argument("cmd")
        p.add_argument("--out", required=True)
        report(p.parse_args().out)
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="<run>/output/movie/renderings (read only)")
    p.add_argument("--out", required=True, help=f"output directory, under {OUT_ROOT}")
    p.add_argument("--out-root", default=OUT_ROOT, help=argparse.SUPPRESS)   # tests only
    p.add_argument("--snaps", default=None, help="e.g. 0-404 or 154,309 (default: every render_NNN in --src)")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--ntask", type=int, default=1)
    p.add_argument("--fields", default="default", help="default | all | comma-separated list")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    out = os.path.realpath(a.out)
    root = os.path.realpath(a.out_root)
    if not out.startswith(root + os.sep):
        sys.exit(f"refusing to write to {out}: outputs go under {root} only")
    if a.snaps:
        snaps = parse_snaps(a.snaps)
    else:
        snaps = sorted(int(d.rsplit("_", 1)[-1]) for d in glob.glob(f"{a.src}/render_*"))
    snaps = snaps[a.task::a.ntask]
    if a.fields == "default":
        fields = DEFAULT_FIELDS
    elif a.fields == "all":
        with h5py.File(render_files(a.src, snaps[0])[0], "r") as f:
            fields = [k for k in f if isinstance(f[k], h5py.Dataset)]
    else:
        fields = a.fields.split(",")

    os.makedirs(f"{out}/checks", exist_ok=True)
    part = f"{out}/checks/part_{a.task:02d}.csv"
    done = set()
    if os.path.exists(part):
        with open(part) as fp:
            done = {int(r["render"]) for r in csv.DictReader(fp)}
    new = not os.path.exists(part)
    with open(part, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=COLS)
        if new:
            w.writeheader()
        for n in snaps:
            row = assemble_one(a.src, out, n, fields, a.overwrite)
            if row is None:
                print(f"render {n:3d}: exists, skipped", flush=True)
                continue
            if n not in done or a.overwrite:
                w.writerow(row)
                fp.flush()
            print(f"render {n:3d} z={row['z']:.4f} mass {row['mass']:.5f} spec {row['spec']:.7f} hi {row['hi']:.6f} "
                  f"corr {row['c0']:.3f} {row['c1']:.3f} {row['c2']:.3f} "
                  f"los {row['los0']:+.3f} {row['los1']:+.3f} {row['los2']:+.3f} (axis {row['vs_axis']})", flush=True)


if __name__ == "__main__":
    main()
