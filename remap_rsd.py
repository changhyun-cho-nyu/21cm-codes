"""
remap_rsd.py
============
He II density in redshift space, made the way the renders make
SpectralDensity / SpectralDensityHI: every cell's mass is moved along +z
(the 3rd grid axis) by its line-of-sight peculiar velocity,
    s = x + v_z / (a H(z))            (comoving),
and deposited with cloud-in-cell weights (periodic).

The renders remap the gas cells themselves; here the 640^3 grid cells are
remapped with their (mass-weighted) mean velocity, so the result is first
validated against the render's own remapped fields:
    remap(Density)                         vs  SpectralDensity
    remap(0.76 Density (1 - HII_Fraction)) vs  SpectralDensityHI
(the same SpectralDensityHI is also compared with the shortcut
0.76 x_HI(real) x SpectralDensity, the construction this replaces)
The velocity unit convention of the Velocities render is not assumed: the
script tries v (peculiar km/s), v sqrt(a) and v / sqrt(a) and reports the
power ratio and cross-correlation r(k) for each.

Usage
-----
  python remap_rsd.py --dir <render_dir> --snap 220 --validate [--write-hi hi_220.hdf5]
  python remap_rsd.py --dir <render_dir> --snap 220 --out <dir> [--conv km_s|sqrt_a|inv_sqrt_a]
The second form writes <dir>/SpectralDensityHeII/SpectralDensityHeII_NNN.hdf5:
dataset SpectralDensityHeII = remap(Density x HeII_Fraction), in Density units
(HeII_Fraction is n_HeII/n_H, so x_HeII (1+delta)_s = SpectralDensityHeII / (y <Density>)),
read by PS_3cm_mpi.c compiled with -DHE_RS_DIR='"<dir>"'.
"""
import argparse
import os

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from check_tb_choices import load, spectra

OM, OL, H0 = 0.30964144, 0.6902672, 67.66          # Lumina; H0 in km/s/Mpc
CONVS = {"km_s": lambda a: 1.0, "sqrt_a": lambda a: np.sqrt(a), "inv_sqrt_a": lambda a: 1.0 / np.sqrt(a)}


def header(d, s):
    with h5py.File(os.path.join(d, "All", f"All_{s:03d}.hdf5"), "r") as h:
        hd = h["Header"].attrs
        z = float(hd["Redshift"]) if "Redshift" in hd else 1.0 / float(hd["Time"]) - 1.0
        return z, float(hd["BoxSize"]), float(hd["HubbleParam"]), int(hd["NumPixels"])


def vz_cells(d, s, z, box, h, n, conv):
    """line-of-sight displacement in grid cells"""
    a = 1.0 / (1.0 + z)
    Hz = H0 * np.sqrt(OM * (1 + z)**3 + OL)                          # km/s/Mpc
    p = os.path.join(d, "Velocities", f"Velocities_{s:03d}.hdf5")
    with h5py.File(p, "r") as f:
        vz = f["Velocities"][..., 2].astype(np.float32)              # km/s (convention to be tested)
    shift_ckpc_h = vz * CONVS[conv](a) / (a * Hz) * 1000.0 * h       # comoving kpc/h
    return shift_ckpc_h / (box / n)


def remap(m, dz):
    """CIC deposit of each cell's content at k + dz along the last axis (periodic)"""
    n0, n1, n = m.shape
    out = np.empty(m.shape, np.float32)
    for i0 in range(0, n0, 16):                                      # slabs: the shift is along z only
        i1 = min(n0, i0 + 16)
        pos = np.arange(n)[None, None, :] + dz[i0:i1]
        k0 = np.floor(pos).astype(np.int64); w = pos - k0
        mm = m[i0:i1].astype(np.float64)
        base = ((np.arange(i1 - i0)[:, None] * n1 + np.arange(n1)[None, :]) * n)[:, :, None]
        acc = np.zeros((i1 - i0) * n1 * n, np.float64)
        for kk, ww in ((k0, 1.0 - w), (k0 + 1, w)):
            acc += np.bincount((base + kk % n).ravel(), (mm * ww).ravel(), minlength=acc.size)
        out[i0:i1] = acc.reshape(i1 - i0, n1, n)
    return out


def compare(name, test, ref, box):
    kc, cnt, P, X = spectra([ref, test], box / 1000.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = X[0] / np.sqrt(P[0] * P[1])
    sel = np.flatnonzero((cnt > 0) & (P[0] > 0))
    pick = sorted({int(sel[np.argmin(abs(kc[sel] - q))]) for q in (0.05, 0.1, 0.3, 1.0, 3.0)})
    rows = "  ".join(f"k={kc[i]:.2f}: {P[1][i] / P[0][i]:.3f}/{r[i]:.3f}" for i in pick)
    cell = np.sqrt(((test - ref)**2).mean(dtype=np.float64)) / np.sqrt(((ref - ref.mean(dtype=np.float64))**2).mean(dtype=np.float64))
    print(f"    {name:44s} P_test/P_render / r(k):  {rows}   cell rms err {cell:.3f}")
    return float(r[sel[np.argmin(abs(kc[sel] - 1.0))]])          # r near k = 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--out", help="write SpectralDensityHeII here")
    p.add_argument("--write-hi", metavar="FILE", help="also save the reproduced SpectralDensityHI (best convention) here")
    p.add_argument("--conv", choices=list(CONVS), default=None,
                   help="velocity convention (default: the best one from the validation)")
    a = p.parse_args()
    d, s = a.dir, a.snap
    z, box, h, n = header(d, s)
    with h5py.File(os.path.join(d, "Velocities", f"Velocities_{s:03d}.hdf5"), "r") as f:
        vh = dict(f["Header"].attrs) if "Header" in f else {}
    print(f"snap {s}  z = {z:.3f}  N = {n}  box {box / 1000:.1f} cMpc/h; Velocities header: "
          + ", ".join(f"{k}={vh[k]}" for k in vh if any(t in k.lower() for t in ("vel", "unit", "time", "redshift"))))
    rho = load(d, "Density", s, 1)
    best = a.conv
    if a.validate or best is None:
        rho_s = load(d, "SpectralDensity", s, 1)
        hi = 0.76 * rho * (1.0 - load(d, "HII_Fraction", s, 1))
        hi_s = load(d, "SpectralDensityHI", s, 1)
        score = {}
        print("  reproducing SpectralDensityHI (all in Density units x 0.76; same k bins):")
        compare("H I: shortcut 0.76 x_HI(real) x SpectralDensity", 0.76 * (1.0 - load(d, "HII_Fraction", s, 1)) * rho_s, hi_s, box)
        for conv in CONVS:
            dz = vz_cells(d, s, z, box, h, n, conv)
            print(f"  convention {conv}: rms shift {np.sqrt((dz**2).mean()):.2f} cells, max {np.abs(dz).max():.1f}")
            score[conv] = compare("gas:  remap(Density) vs SpectralDensity", remap(rho, dz), rho_s, box)
            compare("H I: remap(0.76 Density x_HI) vs SpectralDensityHI", remap(hi, dz), hi_s, box)
            del dz
        best = best or max(score, key=lambda c: score[c])
        print(f"  best convention (highest r near k = 1 for the gas): {best}")
        if a.write_hi:
            hi_r = remap(hi, vz_cells(d, s, z, box, h, n, best))
            with h5py.File(a.write_hi, "w") as f:
                for nm, arr in (("SpectralDensityHI_remap", hi_r), ("SpectralDensityHI_render", hi_s)):
                    f.create_dataset(nm, data=arr, chunks=(16, n, n), compression="gzip", compression_opts=4)
                f.attrs["velocity_convention"] = best; f.attrs["Redshift"] = z; f.attrs["snap"] = s
            print(f"  wrote {a.write_hi} (SpectralDensityHI_remap and SpectralDensityHI_render, same units)")
            del hi_r
        del rho_s, hi, hi_s
    if a.out:
        dz = vz_cells(d, s, z, box, h, n, best)
        he = remap(rho * load(d, "HeII_Fraction", s, 1), dz)
        od = os.path.join(a.out, "SpectralDensityHeII"); os.makedirs(od, exist_ok=True)
        fn = os.path.join(od, f"SpectralDensityHeII_{s:03d}.hdf5")
        with h5py.File(fn, "w") as f:
            ds = f.create_dataset("SpectralDensityHeII", data=he, chunks=(16, n, n), compression="gzip", compression_opts=4)
            ds.attrs["content"] = ("remap along +z of Density x HeII_Fraction (He II per H, Density units) "
                                   "with the Velocities render, CIC, periodic")
            ds.attrs["velocity_convention"] = best
            ds.attrs["Redshift"] = z
        print(f"  wrote {fn}  (<SpectralDensityHeII>/<Density x HeII_Fraction> = "
              f"{he.mean(dtype=np.float64) / (rho * load(d, 'HeII_Fraction', s, 1)).mean(dtype=np.float64):.6f})")


if __name__ == "__main__":
    main()
