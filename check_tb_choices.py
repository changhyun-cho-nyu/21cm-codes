"""
check_tb_choices.py
===================
Measures, on one render snapshot, the two choices left open in the T_b maps
(PS_21cm_mpi.c / PS_3cm_mpi.c):

1. Redshift space for He. The renders have the H I density mapped to redshift
   space (SpectralDensityHI) but no He II equivalent, so the He map uses
   x_HeII(real) x SpectralDensity. The same construction can be made for H I,
   x_HI(real) x SpectralDensity, and compared with the exact SpectralDensityHI:
   that error bounds the He approximation (He III fronts are wider and smoother
   than H II fronts, so the He error should be smaller). Reported as
   P_approx/P_exact and the cross-correlation r(k) of the two fields, and the
   size of the redshift-space effect itself on He, P_rs/P_real.
   The comparison uses every cell; the first output line confirms that
   SpectralDensityHI and HII_Fraction describe the same H I (sum ratio 1).

2. Mass- vs volume-weighted spin-temperature inputs. They differ only in cells
   that are partly ionized, so the script reports which share of the signal
   (H I mass for 21 cm, He II mass for 3.46 cm) sits in cells with
   0.1 < x_HII < 0.9 (HII_Fraction, mass-weighted), and the signal-weighted
   mean of T_mass/T_vol overall and in those cells.

Only densities and fractions are used (no spin factor): the questions are about
the fields that multiply it.

Usage
-----
  python check_tb_choices.py --dir <render_dir> --snap 300 [--factor 2]
<render_dir> holds <field>/<field>_NNN.hdf5 and/or All/All_NNN.hdf5 (as the C codes).
--factor 2 block-averages to 320^3 first (8x less memory; k <= the 320^3 Nyquist).
"""
import argparse
import os

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

Y = 0.24 / (4.0 * 0.76)


def load(d, field, snap, f):
    p = os.path.join(d, field, f"{field}_{snap:03d}.hdf5")
    if not os.path.exists(p):
        p = os.path.join(d, "All", f"All_{snap:03d}.hdf5")
    with h5py.File(p, "r") as h:
        a = h[field][()].astype(np.float32)
    if f > 1:
        n = a.shape[0] // f
        a = a.reshape(n, f, n, f, n, f).mean(axis=(1, 3, 5), dtype=np.float64).astype(np.float32)
    return a


def spectra(fields, box, nbins=24):
    """auto spectra of each field and cross spectra with the first, in log k bins (k in 1/box units)"""
    n = fields[0].shape[0]
    F = [np.fft.rfftn(a - a.mean()) for a in fields]
    kx = np.fft.fftfreq(n, 1.0 / n) * 2 * np.pi / box
    kz = np.arange(n // 2 + 1) * 2 * np.pi / box
    k = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kz[None, None, :]**2)
    kmin, kmax = 2 * np.pi / box, np.pi * n / box
    edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    b = np.digitize(k.ravel(), edges) - 1
    ok = (b >= 0) & (b < nbins)
    bb = b[ok]
    cnt = np.bincount(bb, minlength=nbins)
    def binned(x):
        return np.bincount(bb, x.ravel()[ok], minlength=nbins) / np.maximum(cnt, 1)
    P = [binned(np.abs(f)**2) for f in F]
    X = [binned((F[0] * np.conj(f)).real) for f in F[1:]]
    kc = np.sqrt(edges[1:] * edges[:-1])
    return kc, cnt, P, X


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--factor", type=int, default=1)
    a = p.parse_args()
    d, s, f = a.dir, a.snap, a.factor
    with h5py.File(os.path.join(d, "All", f"All_{s:03d}.hdf5"), "r") as h:
        hd = h["Header"].attrs
        z = float(hd["Redshift"]) if "Redshift" in hd else 1.0 / float(hd["Time"]) - 1.0
        box = float(hd["BoxSize"]) / 1000.0                          # cMpc/h
    rho = load(d, "Density", s, f)
    rho_s = load(d, "SpectralDensity", s, f)
    hi_s = load(d, "SpectralDensityHI", s, f)
    xm = load(d, "HII_Fraction", s, f)
    he2 = load(d, "HeII_Fraction", s, f)
    mean = rho.mean(dtype=np.float64)
    print(f"snap {s}  z = {z:.3f}  grid {rho.shape[0]}^3  box {box:.1f} cMpc/h")
    print(f"  checks: <SpectralDensity>/<Density> = {rho_s.mean(dtype=np.float64) / mean:.5f} (1), "
          f"<SpectralDensityHI>/0.76/<Density x_HI> = {hi_s.mean(dtype=np.float64) / 0.76 / (rho * (1 - xm)).mean(dtype=np.float64):.5f} (1), "
          f"max HeII_Fraction = {he2.max():.4f} (<= y = {Y:.4f}: per H nucleus)")

    # ---- 1. redshift space -------------------------------------------------
    exact = hi_s / 0.76 / mean                                 # x_HI (1+delta), redshift space, exact
    approx = (1 - xm) * rho_s / mean                           # the He construction applied to H I
    he_real = he2 / Y * rho / mean
    he_rs = he2 / Y * rho_s / mean
    kc, cnt, P, X = spectra([exact, approx], box)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = X[0] / np.sqrt(P[0] * P[1])
    kc2, _, P2, _ = spectra([he_real, he_rs], box)
    print("\n1. redshift space (k in h/cMpc)")
    print("   H I test of the He construction: x_HI(real) x SpectralDensity vs SpectralDensityHI;  He: rs / real")
    print(f"   {'k':>8s} {'modes':>8s} {'P_approx/P_exact':>17s} {'r(k)':>8s} {'P_He,rs/P_He,real':>18s}")
    for i in range(len(kc)):
        if cnt[i] == 0 or P[0][i] <= 0 or P2[0][i] <= 0:
            continue
        print(f"   {kc[i]:8.3f} {cnt[i]:8d} {P[1][i] / P[0][i]:17.4f} {r[i]:8.4f} {P2[1][i] / P2[0][i]:18.4f}")
    print(f"   field level: rms(approx - exact)/rms(exact - 1) = "
          f"{np.sqrt(((approx - exact)**2).mean()) / np.sqrt(((exact - exact.mean())**2).mean()):.4f}")

    # ---- 2. weighting -------------------------------------------------------
    T_m = load(d, "Temperature", s, f)
    T_v = load(d, "TemperatureVolumeWeighted", s, f)
    xv = load(d, "HII_VolumeFraction", s, f)
    part = (xm > 0.1) & (xm < 0.9)
    w_h = rho * (1 - xm)                                       # H I mass per cell
    w_he = rho * he2                                           # He II mass per cell
    ratio = T_m / np.maximum(T_v, 1e-30)
    print("\n2. mass- vs volume-weighted spin-temperature inputs")
    for name, w in (("21 cm (H I-weighted)", w_h), ("3.46 cm (He II-weighted)", w_he)):
        W = w.sum(dtype=np.float64)
        if W <= 0:
            print(f"   {name}: no signal"); continue
        fp = w[part].sum(dtype=np.float64) / W
        tr = (w * ratio).sum(dtype=np.float64) / W
        tp = (w[part] * ratio[part]).sum(dtype=np.float64) / max(w[part].sum(dtype=np.float64), 1e-300)
        dx = (w * np.abs(xm - xv)).sum(dtype=np.float64) / W
        print(f"   {name:26s}: share in 0.1 < x_HII < 0.9 cells = {100 * fp:6.2f}%   "
              f"<T_mass/T_vol> = {tr:.4f} (all), {tp:.4f} (those cells)   <|x_mass - x_vol|> = {dx:.4f}")


if __name__ == "__main__":
    main()
