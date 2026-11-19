"""
ps_from_map.py
==============
Power spectrum of a 640^3 delta T_b map written as PS_NNN.out in the format of PS_21cm_mpi.c / PS_3cm_mpi.c
(output_PS), so that the notebooks' load_ps reads it unchanged. For the gas-cell-level 3.46 cm maps of
test_cells_Tb.py merge (RIN_cell_He_NNN.hdf5, dataset "rin"; cellmap_He_NNN.hdf5, datasets "dTb_cell" and
"dTb_production"), which have no PS files.

Estimator and bins: those of check_ps.py, which reproduces the C output to 1e-6 (Delta^2_C / Delta^2_numpy over
1240 bins at snapshot 156): P = |FFT(m - <m>)|^2 L^3 / N^6 on the rfft half-space, 2000 log bins between
k_f = 2 pi / L and k_f N / 2, k at the bin centre, Delta^2 = k^3 P / (2 pi^2).
File: 5 header lines (z, 2000, 1, 1, N^3), then "k[h/ckpc] Delta^2 P CountModes 0" per bin. Lines 3 and 4 are 1, so
load_ps's factor (line 3 / line 4)^2 is 1 and column 2 is Delta^2 itself [map units^2]; shot noise is not
defined for a map and is written as 0. z and L come from the map file's attributes (z; box_ckpc_h or box).

Usage
-----
  python ps_from_map.py --map RIN_cell_He_131.hdf5 --out PS_cell_He_131.out
  python ps_from_map.py --map cellmap_He_136.hdf5 --dset dTb_cell --out PS_cell_He_136.out
  python ps_from_map.py --map Tb_maps_prod/He/RIN_500.hdf5 --out PS_test.out --check Tb_maps_prod/He/PS_500.out
  python ps_from_map.py --map ren_640/HeIII_VolumeFraction/HeIII_VolumeFraction_406.hdf5 --dset HeIII_VolumeFraction \
         --scale 12.6667 --header ren_640/All/All_406.hdf5 --out PS_xHeIII_406.out      # x_HeIII = field / y
--scale: multiply the map first (Delta^2 scales as scale^2), e.g. 1/y for the per-H He fractions.
--header: an HDF5 file with a Header group (a render's All_NNN.hdf5): z = Redshift, L = BoxSize, used when the map
file has no z / box attributes (the render field files).
--check: compare with an existing C PS file of the same map, bin by bin (Delta^2 and mode counts); exit 1 if
it differs by more than 1e-3 in any bin with modes (the C file has 6 significant digits).
"""
import argparse
import sys

import h5py
import numpy as np

BINS_PS = 2000


def spectrum(m, L):
    """k [1/L units], Delta^2, P, CountModes in the C code's 2000 log bins"""
    N = m.shape[0]
    if m.shape != (N, N, N):
        raise ValueError(f"map is {m.shape}, not N^3")
    F = np.fft.rfftn(m - m.mean())
    P = np.abs(F)**2 * L**3 / float(N)**6
    del F
    kf = 2 * np.pi / L
    ii = np.abs(np.fft.fftfreq(N, 1.0 / N))
    kxy2 = (ii[:, None]**2 + ii[None, :]**2) * kf**2
    kz2 = (np.arange(N // 2 + 1) * kf)**2
    Kmin, Kmax = kf, kf * N / 2
    binfac = BINS_PS / (np.log(Kmax) - np.log(Kmin))
    cnt = np.zeros(BINS_PS)
    Ps = np.zeros(BINS_PS)
    for i in range(N):                                    # one x plane at a time (memory)
        k = np.sqrt(kxy2[i][:, None] + kz2[None, :])
        ok = (k >= Kmin) & (k < Kmax)
        b = (np.log(k[ok] / Kmin) * binfac).astype(np.int64)
        cnt += np.bincount(b, minlength=BINS_PS)[:BINS_PS]
        Ps += np.bincount(b, P[i][ok], minlength=BINS_PS)[:BINS_PS]
    Pb = Ps / np.maximum(cnt, 1)
    kc = np.exp((np.arange(BINS_PS) + 0.5) / binfac) * Kmin
    return kc, kc**3 * Pb / (2 * np.pi**2), Pb, cnt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True, help="HDF5 file with a 3-D map")
    p.add_argument("--dset", default="rin", help="dataset (rin; dTb_cell or dTb_production in cellmap files)")
    p.add_argument("--out", required=True)
    p.add_argument("--z", type=float, help="redshift, if the file has no z attribute")
    p.add_argument("--box", type=float, help="BoxSize [ckpc/h], if the file has no box attribute (Lumina: 338300)")
    p.add_argument("--scale", type=float, default=1.0, help="multiply the map by this first (e.g. 1/y)")
    p.add_argument("--header", help="HDF5 file with a Header group (Redshift, BoxSize), e.g. the render's All_NNN.hdf5")
    p.add_argument("--check", help="a C PS file of the same map to compare with")
    a = p.parse_args()

    with h5py.File(a.map, "r") as f:
        m = f[a.dset][()].astype(np.float64) * a.scale
        at = dict(f.attrs)
    if a.header:
        with h5py.File(a.header, "r") as f:
            hd = dict(f["Header"].attrs)
        at.setdefault("z", hd.get("Redshift"))
        at.setdefault("box", hd.get("BoxSize"))
    z = a.z if a.z is not None else at.get("z")
    L = a.box if a.box is not None else at.get("box_ckpc_h", at.get("box"))
    if z is None or L is None:
        sys.exit(f"ERROR: {a.map} has no z / box attributes: pass --z / --box")
    z, L = float(z), float(L)
    if not np.all(np.isfinite(m)):
        sys.exit(f"ERROR: {a.map}:{a.dset} has {np.count_nonzero(~np.isfinite(m))} non-finite values")
    N = m.shape[0]
    kc, D2, Pb, cnt = spectrum(m, L)
    with open(a.out, "w") as fo:
        fo.write(f"{z:g}\n{BINS_PS}\n1\n1\n{N**3:15d}\n")
        for i in range(BINS_PS):
            fo.write(f"{kc[i]:g} {D2[i]:g} {Pb[i]:g} {cnt[i]:g} 0\n")
    sel = cnt > 0
    print(f"{a.map}:{a.dset}{f' x {a.scale:g}' if a.scale != 1 else ''}: z = {z:g}, N = {N}, L = {L:g} ckpc/h, <map> = {m.mean():.5e}, "
          f"{int(sel.sum())} bins with modes -> {a.out}")

    if a.check:
        with open(a.check) as f:
            head = [next(f) for _ in range(5)]
        t = np.loadtxt(a.check, skiprows=5)
        D2C = t[:, 1] * (float(head[2]) / float(head[3]))**2
        use = (t[:, 3] > 0) & sel
        same_counts = np.array_equal(t[use, 3], cnt[use]) and np.array_equal(t[:, 3] > 0, sel)
        r = D2C[use] / D2[use]
        print(f"  vs {a.check}: mode counts {'identical' if same_counts else 'DIFFER'}; Delta^2_C / Delta^2_here "
              f"median {np.median(r):.6f}, range {r.min():.6f} - {r.max():.6f} ({use.sum()} bins); "
              f"z {float(head[0]):g} vs {z:g}")
        if not same_counts or np.max(np.abs(r - 1)) > 1e-3:
            sys.exit(1)


if __name__ == "__main__":
    main()
