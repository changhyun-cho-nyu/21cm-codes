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

q method (--qtest): the render has the redshift-space H II mass exactly,
    rho_HII,s = 0.76 SpectralDensity - SpectralDensityHI
(sub-cell velocities included, star-forming gas neutral), so the He II can be
placed as
    rho_HeII,s = q_s x rho_HII,s,   q = He II / H II = HeII_Fraction / (0.76 HII_Fraction),
    q_s = remap(Density x HeII_Fraction) / remap(0.76 Density x HII_Fraction)
and the grid-mean velocities then enter only through q, which varies much less
inside a pixel than the density does. --qtest runs two checks:
  1. the same construction on H I (remapped fraction x exact redshift-space
     density), against the exact SpectralDensityHI; x_HI varies more inside a
     pixel than q does, so this bounds the q-method error from above;
  2. how smooth q is: its H II-mass-weighted distribution (q/y) and pixel-to-pixel
     change, the He II mass the method keeps, and the He power from q_s vs from
     the real-space q (no remap of q at all), next to the plain remap and the shortcut.

Usage
-----
  python remap_rsd.py --dir <render_dir> --snap 220 --validate [--write-hi hi_220.hdf5]
  python remap_rsd.py --dir <render_dir> --snap 220 --qtest [--rsd]
--rsd reports how large the redshift-space effect is (the He signal of the T_b maps is
real space): P_rs/P_real and r(k) for the gas and H I (exact, from the render) and for
He II from the q method, the plain remap and the shortcut, and P_rs/P_real for modes
across (mu < 0.3) and along (mu > 0.7) the line of sight.
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

from check_tb_choices import Y, load, spectra

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


def compare(name, test, ref, box, refname="render"):
    kc, cnt, P, X = spectra([ref, test], box / 1000.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = X[0] / np.sqrt(P[0] * P[1])
    sel = np.flatnonzero((cnt > 0) & (P[0] > 0))
    pick = sorted({int(sel[np.argmin(abs(kc[sel] - q))]) for q in (0.05, 0.1, 0.3, 1.0, 3.0)})
    rows = "  ".join(f"k={kc[i]:.2f}: {P[1][i] / P[0][i]:.3f}/{r[i]:.3f}" for i in pick)
    cell = np.sqrt(((test - ref)**2).mean(dtype=np.float64)) / np.sqrt(((ref - ref.mean(dtype=np.float64))**2).mean(dtype=np.float64))
    print(f"    {name:44s} P_test/P_{refname} / r(k):  {rows}   cell rms err {cell:.3f}")
    return float(r[sel[np.argmin(abs(kc[sel] - 1.0))]])          # r near k = 1


def ratio(num, den):
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def wpercentiles(x, w, ps=(0.01, 0.16, 0.5, 0.84, 0.99)):
    """weighted percentiles of x > 0 from a fine log histogram (values below 1e-4 go in the first bin)"""
    edges = np.logspace(-4, 2, 1201)
    hst, _ = np.histogram(np.clip(x, edges[0], edges[-1] * 0.999), edges, weights=w)
    c = np.cumsum(hst) / hst.sum()
    return [float(edges[min(np.searchsorted(c, p) + 1, edges.size - 1)]) for p in ps]


def mu_split(name, rs, real, box, nbins=24):
    """P_rs/P_real for modes across (mu < 0.3) and along (mu > 0.7) the line of sight (3rd axis)"""
    n = real.shape[0]
    Fa = np.fft.rfftn(real - real.mean(dtype=np.float64))
    Fb = np.fft.rfftn(rs - rs.mean(dtype=np.float64))
    kx = np.fft.fftfreq(n, 1.0 / n) * 2 * np.pi / box
    kz = np.arange(n // 2 + 1) * 2 * np.pi / box
    k = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kz[None, None, :]**2)
    mu = np.divide(np.broadcast_to(kz[None, None, :], k.shape), k, out=np.zeros_like(k), where=k > 0)
    edges = np.logspace(np.log10(2 * np.pi / box), np.log10(np.pi * n / box), nbins + 1)
    b = np.digitize(k, edges) - 1
    kc = np.sqrt(edges[1:] * edges[:-1])
    del k
    out = []
    for lo, hi in ((0.0, 0.3), (0.7, 1.01)):
        m = (mu >= lo) & (mu < hi) & (b >= 0) & (b < nbins)
        bb = b[m]
        Pa = np.bincount(bb, np.abs(Fa[m])**2, minlength=nbins)
        Pb = np.bincount(bb, np.abs(Fb[m])**2, minlength=nbins)
        sel = np.flatnonzero(Pa > 0)
        pick = sorted({int(sel[np.argmin(abs(kc[sel] - q))]) for q in (0.05, 0.1, 0.3, 1.0)})
        out.append(f"mu {lo:.1f}-{min(hi, 1.0):.1f}: " + " ".join(f"k={kc[i]:.2f}:{Pb[i] / Pa[i]:.3f}" for i in pick))
    print(f"    {name:44s} P_rs/P_real  " + "   ".join(out))


def q_test(d, s, z, box, h, n, conv, rho, checks=True, rsd=False):
    """the q method and its two checks (see the module docstring); all densities in Density units.
    rsd: also the size of the redshift-space effect, P_rs/P_real, for gas and H I (exact) and He II"""
    dz = vz_cells(d, s, z, box, h, n, conv)
    rho_s = load(d, "SpectralDensity", s, 1)
    hi_s = load(d, "SpectralDensityHI", s, 1)
    xhii = load(d, "HII_Fraction", s, 1)
    hii_s = 0.76 * rho_s - hi_s                                       # exact redshift-space H II mass
    print(f"  q method, convention {conv}: min(0.76 SpectralDensity - SpectralDensityHI)/mean = "
          f"{hii_s.min() / hii_s.mean(dtype=np.float64):.2e} (>= 0 up to rounding; clipped at 0)")
    np.maximum(hii_s, 0.0, out=hii_s)
    hi = 0.76 * rho * (1.0 - xhii)

    if checks:
        print("  check 1, H I analogue: remapped fraction x exact redshift-space density, vs SpectralDensityHI")
        hi_r = remap(hi, dz)
        compare("H I: shortcut 0.76 x_HI(real) x SpectralDensity", 0.76 * (1.0 - xhii) * rho_s, hi_s, box)
        compare("H I: plain remap(0.76 Density x_HI)", hi_r, hi_s, box)
        compare("H I: SpectralDensity x remap(H I)/remap(gas)", ratio(hi_r, remap(rho, dz)) * rho_s, hi_s, box)
        del hi_r

    he = rho * load(d, "HeII_Fraction", s, 1)                         # He II (per H) x Density
    hii = 0.76 * rho * xhii                                           # real-space H II mass
    del xhii
    q_re = ratio(he, hii)                                             # = HeII_Fraction / (0.76 HII_Fraction)
    if checks:
        sw = hii.sum(dtype=np.float64)
        print("  check 2, smoothness of q, shown as n_HeII/(y n_HII) = HeII_Fraction/(y HII_Fraction) "
              "(1 = all He as He II where H is ionized):")
        print(f"    H II-mass-weighted mean = {he.sum(dtype=np.float64) / sw * 0.76 / Y:.4f};  "
              "percentiles 1/16/50/84/99%: "
              + " / ".join(f"{v:.3g}" for v in wpercentiles(q_re * (0.76 / Y), hii)))
        for lo in (0.1, 0.5):
            m = hii < lo * 0.76 * rho
            print(f"    He II mass in pixels with x_HII < {lo}: {he[m].sum(dtype=np.float64) / he.sum(dtype=np.float64):.4f} "
                  f"(H II mass there {hii[m].sum(dtype=np.float64) / sw:.4f})")
        del m
        dq = np.roll(q_re, -1, axis=2) - q_re
        dhe = np.roll(he, -1, axis=2) - he
        wq = 0.5 * (hii + np.roll(hii, -1, axis=2))
        sq = np.sqrt((wq * dq**2).sum(dtype=np.float64) / wq.sum(dtype=np.float64)) / (he.sum(dtype=np.float64) / sw)
        she = np.sqrt((dhe.astype(np.float64)**2).mean()) / he.mean(dtype=np.float64)
        print(f"    rms change between neighbouring pixels along z, over the mean: q {sq:.3f} (H II-weighted), "
              f"He II density {she:.3f}")
        del dq, dhe, wq

    he_r = remap(he, dz)
    q_rs = ratio(he_r, remap(hii, dz))
    he_q = q_rs * hii_s                                               # the q method
    del q_rs, hii
    he_sc = ratio(he, rho) * rho_s                                    # shortcut HeII_Fraction x SpectralDensity
    if checks:
        tot = he.sum(dtype=np.float64)
        print(f"    He II mass: q method / real = {he_q.sum(dtype=np.float64) / tot:.5f}, "
              f"q_real x exact H II_s / real = {(q_re * hii_s).sum(dtype=np.float64) / tot:.5f}, "
              f"plain remap / real = {he_r.sum(dtype=np.float64) / tot:.5f}")
        print("    He fields against the q method (no exact reference exists; differences show the sensitivity):")
        compare("He: q_real x exact H II_s (q not remapped)", q_re * hii_s, he_q, box, "q")
        compare("He: plain remap(Density x HeII_Fraction)", he_r, he_q, box, "q")
        compare("He: shortcut HeII_Fraction x SpectralDensity", he_sc, he_q, box, "q")
    del q_re

    if rsd:
        print("  size of the redshift-space effect: redshift-space field vs the same field in real space")
        print("   (P_rs/P_real / r(k); gas and H I exact from the render, He II from the three constructions)")
        compare("gas:   SpectralDensity vs Density", rho_s, rho, box, "real")
        compare("H I:   SpectralDensityHI vs 0.76 Density x_HI", hi_s, hi, box, "real")
        compare("He II: q method vs Density x HeII_Fraction", he_q, he, box, "real")
        compare("He II: plain remap", he_r, he, box, "real")
        compare("He II: shortcut", he_sc, he, box, "real")
        print("   by angle to the line of sight (mu = k_z/k):")
        mu_split("H I (exact)", hi_s, hi, box / 1000.0)
        mu_split("He II (q method)", he_q, he, box / 1000.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--qtest", action="store_true", help="test the q method (exact H II_s x remapped He II/H II)")
    p.add_argument("--rsd", action="store_true", help="size of the redshift-space effect, P_rs/P_real, for gas, "
                   "H I (exact) and He II (q method, plain remap, shortcut)")
    p.add_argument("--out", help="write SpectralDensityHeII here")
    p.add_argument("--write-hi", metavar="FILE", help="also save the reproduced SpectralDensityHI (best convention) here")
    p.add_argument("--conv", choices=list(CONVS), default="sqrt_a",
                   help="velocity convention: v_pec = Velocities x sqrt(a) km/s (AREPO internal units) is the one "
                        "that reproduces SpectralDensity / SpectralDensityHI on Lumina (P ratio 1.000, r = 1.000 "
                        "for k < 1 h/cMpc at z = 7.6); --validate tests all three")
    a = p.parse_args()
    d, s = a.dir, a.snap
    z, box, h, n = header(d, s)
    with h5py.File(os.path.join(d, "Velocities", f"Velocities_{s:03d}.hdf5"), "r") as f:
        vh = dict(f["Header"].attrs) if "Header" in f else {}
    print(f"snap {s}  z = {z:.3f}  N = {n}  box {box / 1000:.1f} cMpc/h; Velocities header: "
          + ", ".join(f"{k}={vh[k]}" for k in vh if any(t in k.lower() for t in ("vel", "unit", "time", "redshift"))))
    rho = load(d, "Density", s, 1)
    best = a.conv
    if a.validate:
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
        auto = max(score, key=lambda c: score[c])
        print(f"  best convention (highest r near k = 1 for the gas): {auto}"
              + ("" if auto == best else f"   WARNING: --conv is {best}"))
        if a.write_hi:
            hi_r = remap(hi, vz_cells(d, s, z, box, h, n, best))
            with h5py.File(a.write_hi, "w") as f:
                for nm, arr in (("SpectralDensityHI_remap", hi_r), ("SpectralDensityHI_render", hi_s)):
                    f.create_dataset(nm, data=arr, chunks=(16, n, n), compression="gzip", compression_opts=4)
                f.attrs["velocity_convention"] = best; f.attrs["Redshift"] = z; f.attrs["snap"] = s
            print(f"  wrote {a.write_hi} (SpectralDensityHI_remap and SpectralDensityHI_render, same units)")
            del hi_r
        del rho_s, hi, hi_s
    if a.qtest or a.rsd:
        q_test(d, s, z, box, h, n, best, rho, checks=a.qtest, rsd=a.rsd)
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
