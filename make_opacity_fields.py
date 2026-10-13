"""
make_opacity_fields.py
======================
Absorber densities for the He II Lyman-band transmission (compute_transmission_He.c).

Photons between He II Ly-alpha and the He II Lyman limit (40.8-54.4 eV) are
absorbed by H I and He I photoionization. For every render snapshot this
script writes the comoving number densities of the two absorbers, block-
averaged onto a coarser grid:

    n_HI  = n_H (1 - x_HII)                 n_H = X rho / m_p   (comoving cm^-3)
    n_HeI = n_He (1 - x_HeII - x_HeIII)      n_He = y n_H, ionized fractions per He

HII_Fraction / HeII_Fraction / HeIII_Fraction are the mass-weighted render
fields, so rho x_HI is the H I mass in a cell and the block average is the
H I column per unit length through the block. The He fractions are stored per
H nucleus in the renders (max HeII_Fraction = 0.079 = y); --he-per-he switches
to fractions per He nucleus.

Star-forming gas: Lumina's radiative transfer treats gas above the star-formation
threshold as transparent, and the escape fractions (stellar f_esc = 0.18, the
AGN f_esc in LuminosityAGN) already describe the absorption inside it. Counting
its H I again would absorb the same photons twice. Every run prints the share of
the H I and He I that sits in render cells with StarFormationRate > 0; with
--sf-mask those cells get n_HI = n_HeI = 0 before the block average. A 640^3 cell
is 0.78 cMpc, so the mask also removes whatever circumgalactic/IGM H I shares a
cell with the star-forming gas: the masked and unmasked runs bracket the effect.
Write the masked fields to their own directory.

Output: <out>/opacity_NNN.hdf5 with datasets n_HI, n_HeI (float32, comoving
cm^-3, shape (N/f)^3) and attributes Redshift, BoxSize_cMpc, factor, sf_mask,
HI_share_SF_cells, HeI_share_SF_cells.

Usage
-----
  python make_opacity_fields.py --out DIR [--factor 2] [--every 1] [--snaps 0 708] [--sf-mask]
"""
import argparse
import os
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401  (compressed renders)
except ImportError:
    pass

RENDER_DIRS = [
    "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640",
    "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640",
]
X_H = 0.76
Y_HE = 0.24 / (4.0 * 0.76)               # n_He / n_H
M_P = 1.67262192e-24
MSUN_1E10 = 1.98847e43                   # default UnitMass_in_g (1e10 Msun / h)

def block_mean(a, f):
    if f == 1:
        return a
    n0, n1, n2 = a.shape
    return a.reshape(n0 // f, f, n1 // f, f, n2 // f, f).mean(axis=(1, 3, 5))

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--dirs", nargs="+", default=RENDER_DIRS)
    p.add_argument("--factor", type=int, default=2, help="block-average factor (640 -> 320 for 2)")
    p.add_argument("--snaps", type=int, nargs=2, default=[0, 708], metavar=("FIRST", "LAST"))
    p.add_argument("--every", type=int, default=1, help="keep every k-th snapshot")
    p.add_argument("--he-per-he", action="store_true", help="He fractions are per He nucleus, not per H")
    p.add_argument("--sf-mask", action="store_true",
                   help="n_HI = n_HeI = 0 in cells with StarFormationRate > 0 (transparent ISM, as in Lumina's RT)")
    p.add_argument("--planes", type=int, default=32)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    f = args.factor
    planes = max(f, args.planes // f * f)

    for s in range(args.snaps[0], args.snaps[1] + 1, args.every):
        path = next((q for q in (os.path.join(d, f"All/All_{s:03d}.hdf5") for d in args.dirs) if os.path.exists(q)), None)
        if path is None:
            print(f"snap {s:03d}: not found, skipped"); continue
        out = os.path.join(args.out, f"opacity_{s:03d}.hdf5")
        with h5py.File(path, "r") as fr:
            hd = fr["Header"].attrs
            z = float(hd["Redshift"]); h = float(hd["HubbleParam"])
            ulen = float(hd["UnitLength_in_cm"]); umass = float(hd.get("UnitMass_in_g", MSUN_1E10))
            N = fr["Density"].shape[0]
            if N % f:
                sys.exit(f"ERROR: N = {N} not divisible by factor {f}")
            box_cmpc = float(hd["BoxSize"]) / h / 1000.0
            rho_unit = umass * h**2 / ulen**3                     # code density -> g cm^-3 (comoving)
            Nc = N // f
            nHI = np.empty((Nc, Nc, Nc), np.float32); nHeI = np.empty_like(nHI)
            fmax = 0.0
            sum_HI = sum_HeI = sf_HI = sf_HeI = 0.0; n_sf = 0
            for i0 in range(0, N, planes):
                sl = slice(i0, min(N, i0 + planes))
                nH = X_H * fr["Density"][sl].astype(np.float64) * rho_unit / M_P
                xHI = np.clip(1.0 - fr["HII_Fraction"][sl].astype(np.float64), 0.0, 1.0)
                fion = fr["HeII_Fraction"][sl].astype(np.float64) + fr["HeIII_Fraction"][sl].astype(np.float64)
                fmax = max(fmax, float(fion.max()))
                xHeI = np.clip(1.0 - (fion if args.he_per_he else fion / Y_HE), 0.0, 1.0)
                hi, hei = nH * xHI, Y_HE * nH * xHeI
                sf = fr["StarFormationRate"][sl] > 0
                sum_HI += hi.sum(); sum_HeI += hei.sum()
                sf_HI += hi[sf].sum(); sf_HeI += hei[sf].sum(); n_sf += int(sf.sum())
                if args.sf_mask:
                    hi[sf] = 0.0; hei[sf] = 0.0
                c = slice(i0 // f, sl.stop // f)
                nHI[c] = block_mean(hi, f)
                nHeI[c] = block_mean(hei, f)
        if not args.he_per_he and fmax > 1.05 * Y_HE:
            sys.exit(f"ERROR: snap {s}: max(HeII+HeIII fraction) = {fmax:.3f} > y = {Y_HE:.3f}; "
                     "the He fractions look per He nucleus: rerun with --he-per-he")
        with h5py.File(out, "w") as fo:
            for name, arr in (("n_HI", nHI), ("n_HeI", nHeI)):
                d = fo.create_dataset(name, data=arr, chunks=(min(16, Nc), Nc, Nc), compression="gzip", compression_opts=4)
                d.attrs["units"] = "comoving cm^-3"
            fo.attrs["Redshift"] = z
            fo.attrs["BoxSize_cMpc"] = box_cmpc
            fo.attrs["factor"] = f
            fo.attrs["sf_mask"] = int(args.sf_mask)
            fo.attrs["HI_share_SF_cells"] = sf_HI / sum_HI if sum_HI > 0 else 0.0
            fo.attrs["HeI_share_SF_cells"] = sf_HeI / sum_HeI if sum_HeI > 0 else 0.0
            fo.attrs["source"] = path
        print(f"snap {s:03d}  z = {z:7.3f}  <n_HI> = {nHI.mean():.3e}  <n_HeI> = {nHeI.mean():.3e} "
              f"(comoving cm^-3)  SFR>0 cells: {100.0 * n_sf / N**3:.3f}% of cells, "
              f"{100.0 * sf_HI / max(sum_HI, 1e-300):.2f}% of H I, {100.0 * sf_HeI / max(sum_HeI, 1e-300):.2f}% of He I "
              f"({'masked' if args.sf_mask else 'kept'})  -> {out}", flush=True)

if __name__ == "__main__":
    main()
