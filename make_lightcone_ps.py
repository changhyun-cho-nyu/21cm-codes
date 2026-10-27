"""
make_lightcone_ps.py
====================
Power spectra Delta^2(k) of the light-cone delta T_b (make_lightcone_tb.py output, lightcone_dTb_<line>.hdf5) in
bands of comoving depth --depth around each --z (lightcone_ps.lightcone_ps: exact line-of-sight sum over the
unequal slices, each slice's mean removed, Blackman-Harris along the line of sight, Tukey 0.2 across the sky), and
the same k bins of the production 640^3 maps (RIN_NNN.hdf5), interpolated linearly in z to the band centre between
the two maps that bracket it.

The maps' Delta^2 is computed here from RIN (P = |FFT(dT_b - <dT_b>)|^2 L^3 / N^6, the convention of PS_NNN.out,
checked by check_ps.py) in exactly the cone's k bins (same k_min = 2 pi / max(L), 10 per decade, same k_max), so
the ratio cone / maps is bin by bin. Expected differences:
  - the cone pixel size (0.43 cMpc/h at z = 3 to 0.73 at z = 20, maps 0.53): 3.46 cm at all z, 21 cm at z >~ 10
    (as in the means, check_lightcone_tb.py);
  - the light-cone effect: a band of depth 338.3 cMpc/h spans Delta z = 0.71 / 1.42 / 2.29 / 4.0 at z = 4 / 7 /
    10 / 15 (150 cMpc/h: 0.31 / 0.63 / 1.02 / 1.8), the maps are coeval at the band centre;
  - window leakage: usable for k >~ 6 x 2 pi / depth (lightcone_ps tests), marked in the table;
  - sample variance (one cone band vs the whole box).
Bands whose slices are not all done in the cone file are skipped (so it runs on a partial file).

Output: --out/lightcone_ps_<line>.hdf5, group d<depth>/z<z>: k, D2, nmodes (cone), D2_maps, nmodes_maps, attributes
(the band's z and chi range, slices, pixel, k_use = 6 x 2 pi / depth, the map snapshots and weights). Prints
cone / maps in k ranges (median of the bins, the number of bins).

Usage
-----
  python make_lightcone_ps.py --cone-dir DIR --out DIR [--lines H He] [--z 3.5 ... 15] [--depth 338.3 150]
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd

from check_lightcone_tb import bracket, map_list
from lightcone_ps import lightcone_ps

USER = "/orcd/data/mvogelsb/004/chcho"
UNITS = {"H": "mK^2", "He": "uK^2"}
KRANGES = ((None, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, None))     # k_use is the lower limit of the first range


def map_power(path, L):
    """|k| and P(k) of all rfftn modes of a RIN map (box L cMpc/h), the PS_NNN.out normalisation"""
    with h5py.File(path, "r") as f:
        m = f["rin"][()].astype(np.float64)
    N = m.shape[0]
    F = np.fft.rfftn(m - m.mean())
    del m
    P = (np.abs(F)**2 * (L**3 / float(N)**6)).ravel()
    del F
    kf = 2 * np.pi / L
    i = np.abs(np.fft.fftfreq(N, 1.0 / N))
    k = kf * np.sqrt(i[:, None, None]**2 + i[None, :, None]**2 + np.arange(N // 2 + 1)[None, None, :]**2).ravel()
    return k, P


def bin_power(k, P, kmin, kmax, bpd, nbins):
    """Delta^2 and mode counts in the bins of lightcone_ps (index floor(log10(k / kmin) bpd + 1e-9))"""
    ok = (k >= kmin) & (k < kmax)
    b = np.floor(np.log10(k[ok] / kmin) * bpd + 1e-9).astype(int)
    n = np.bincount(b, minlength=nbins)[:nbins]
    s = np.bincount(b, k[ok]**3 * P[ok], minlength=nbins)[:nbins]
    return s / np.maximum(n, 1) / (2 * np.pi**2), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cone-dir", default=f"{USER}/lightcone", help="make_lightcone_tb.py output")
    ap.add_argument("--out", default=f"{USER}/lightcone")
    ap.add_argument("--lines", nargs="+", default=["H", "He"], choices=["H", "He"])
    ap.add_argument("--z", type=float, nargs="+", default=[3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 11.0, 13.0, 15.0])
    ap.add_argument("--depth", type=float, nargs="+", default=[338.3, 150.0], help="band depths [cMpc/h]")
    ap.add_argument("--maps-H", default=f"{USER}/Tb_maps_prod/H_real")
    ap.add_argument("--maps-He", default=f"{USER}/Tb_maps_prod/He")
    ap.add_argument("--ion", default=f"{USER}/ion_history/ionization_history.csv")
    ap.add_argument("--box", type=float, default=None, help="map box [cMpc/h]; default the cone header BoxSize")
    ap.add_argument("--window-los", default="blackmanharris", choices=["blackmanharris", "none"])
    ap.add_argument("--taper-perp", type=float, default=0.2)
    ap.add_argument("--band-mean", action="store_true", help="remove the band's mean instead of each slice's (a test)")
    ap.add_argument("--bins-per-decade", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ion = pd.read_csv(a.ion, comment="#").set_index("snap")

    for line in a.lines:
        src = os.path.join(a.cone_dir, f"lightcone_dTb_{line}.hdf5")
        mdir = a.maps_H if line == "H" else a.maps_He
        maps = map_list(mdir, ion)
        with h5py.File(src, "r") as f, h5py.File(os.path.join(a.out, f"lightcone_ps_{line}.hdf5"), "a") as g:
            ze, ce, done = f["z_edges"][()], f["chi_edges"][()], f["done"][()]
            theta = float(f.attrs["header_OpeningAngle"])
            L = a.box if a.box else float(f.attrs["header_BoxSize"]) / 1e3
            centre = 0.5 * (ce[:-1] + ce[1:])
            oz = np.argsort(ze)
            print(f"\n===== {line}: {src}, {done.sum()} / {len(done)} slices done; maps {len(maps)} in {mdir}; "
                  f"box {L:.2f} cMpc/h; depths {a.depth}", flush=True)
            for zt in a.z:
                chi0 = float(np.interp(zt, ze[oz], ce[oz]))
                bands = {}
                for depth in a.depth:
                    sel = np.flatnonzero((centre >= chi0 - 0.5 * depth) & (centre < chi0 + 0.5 * depth))
                    if len(sel) < 8 or chi0 - 0.5 * depth < ce.min() or chi0 + 0.5 * depth > ce.max():
                        print(f"  z = {zt}, depth {depth}: band outside the cone or < 8 slices; skipped")
                    elif not done[sel].all():
                        print(f"  z = {zt}, depth {depth}: {np.sum(~done[sel])} of {len(sel)} slices not done; skipped")
                    else:
                        bands[depth] = sel
                if not bands:
                    continue
                br = bracket(maps, zt)
                mp = []
                if br is None:
                    print(f"  z = {zt}: no maps bracket it; cone only")
                else:
                    (za, sa), (zb, sb) = br
                    wb = 0.0 if zb == za else (zt - za) / (zb - za)
                    mp = [(sa, 1.0 - wb)] + ([(sb, wb)] if sb != sa else [])
                    mpow = {s: map_power(os.path.join(mdir, f"RIN_{s}.hdf5"), L) for s, _ in mp}
                for depth, sel in bands.items():
                    i0, i1 = max(sel.min() - 1, 0), min(sel.max() + 2, len(centre))   # one slice of margin each side
                    tb = f["dTb"][:, :, i0:i1]
                    k, D2, n, info = lightcone_ps(tb, ce[i0:i1 + 1], ze[i0:i1 + 1], zt, depth, theta,
                                                  window_los=a.window_los, taper_perp=a.taper_perp,
                                                  slice_mean=not a.band_mean, bins_per_decade=a.bins_per_decade)
                    del tb
                    if info["nslices"] != len(sel):
                        raise RuntimeError(f"lightcone_ps used {info['nslices']} slices, expected {len(sel)}")
                    kmin = 2 * np.pi / max(info["L"])
                    kmax = np.pi / max(info["pix"][0], info["pix"][1], info["L"][2] / info["nslices"])
                    idx = np.floor(np.log10(k / kmin) * a.bins_per_decade + 1e-9).astype(int)
                    kuse = 6 * 2 * np.pi / depth
                    name = f"d{depth:g}/z{zt:g}"
                    if name in g:
                        del g[name]
                    h = g.create_group(name)
                    h["k"], h["D2"], h["nmodes"] = k, D2, n
                    h.attrs.update(z_centre=zt, depth=depth, z_range=info["z_range"], chi_range=info["chi_range"],
                                   nslices=info["nslices"], pix=info["pix"], L=info["L"], k_use=kuse,
                                   window_los=a.window_los, taper_perp=a.taper_perp, slice_mean=not a.band_mean,
                                   unit=UNITS[line], source=src)
                    head = (f"  z = {zt:5.2f}, depth {depth:5.1f}: z {info['z_range'][0]:.3f} - {info['z_range'][1]:.3f}, "
                            f"{info['nslices']} slices, pixel {info['pix'][0]:.3f} cMpc/h, k_use {kuse:.3f}")
                    if not mp:
                        print(head + f"; Delta^2 at k_use..1: {np.median(D2[(k >= kuse) & (k < 1)]):.4g} {UNITS[line]}")
                        continue
                    D2m = np.zeros(idx.max() + 1)
                    nm = None
                    for s, w in mp:
                        d, nm = bin_power(*mpow[s], kmin, kmax, a.bins_per_decade, idx.max() + 1)
                        D2m += w * d
                    D2m, nm = D2m[idx], nm[idx]
                    h["D2_maps"], h["nmodes_maps"] = D2m, nm
                    h.attrs.update(map_snaps=[s for s, _ in mp], map_weights=[w for _, w in mp])
                    r = D2 / np.where(D2m > 0, D2m, np.nan)
                    parts = []
                    for lo, hi in KRANGES:
                        lo = kuse if lo is None else max(lo, kuse)
                        if hi and lo >= hi:
                            continue
                        m = (k >= lo) & (k < (hi if hi else np.inf))
                        parts.append(f"[{lo:.2f},{hi if hi else 'max'}) {np.nanmedian(r[m]):.3f} ({m.sum()})" if m.any() else
                                     f"[{lo:.2f},{hi if hi else 'max'}) -")
                    print(head + f"; maps {'+'.join(str(s) for s, _ in mp)}; cone/maps " + ", ".join(parts), flush=True)
                if mp:
                    del mpow


if __name__ == "__main__":
    main()
