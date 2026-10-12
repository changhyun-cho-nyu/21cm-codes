"""
check_sfr_eff_spectrum.py
=========================
How good is the one approximation behind SFR_eff?

make_SFR_eff.py weights each voxel's SFR history with the BAND-integrated
BPASS photon rate, and the J_alpha codes then multiply SFR_eff by the
LIFETIME spectrum N_nu(nu). This is exact only if the spectral shape inside
the band does not change with stellar age. Young populations are bluer, so for
a history dominated by young stars the true emissivity is tilted towards the
Lyman limit relative to the lifetime shape.

For one metallicity and a few SFR histories, this script compares
    exact  : eps(nu) = Int_0^tau_max SFR(t0 - tau) P_nu(tau) dtau
    approx : SFR_eff x N_nu,  SFR_eff = Int SFR(t0 - tau) P_band(tau) dtau / N_band
(P = BPASS photon rate per Hz per baryon, constant within its age bins, as in
load_spectrum() and make_SFR_eff.py) and reports approx/exact
  * at the middle of a few Lyman-n windows (nu_n + nu_n+1)/2 (the line
    frequencies themselves sit in stellar absorption lines, whose depth
    changes strongly with age, so single points there are misleading), and
  * weighted as J_alpha weights the band: sum_n f_rec(n) Int_{nu_n}^{nu_n+1} eps dnu
    (ignores the slow change of c/H and of the SFR across one Lyman window).

Usage
-----
  python check_sfr_eff_spectrum.py [--bpass FILE] [--Z 0.001 0.004 0.02]
"""
import argparse

import h5py
import numpy as np

from make_Z_snapshots import BANDS, BPASS_FILE, GYR2SEC, MP_CGS, MSOL2GRAM, MYR_S, PLANCK

TAU_MAX = {"H": 300.0, "He": 100.0}
F_REC = [0.0, 0.0, 1.0, 0.0, 0.2609, 0.3078, 0.3259, 0.3353, 0.3410, 0.3448, 0.3476,
         0.3496, 0.3512, 0.3524, 0.3535, 0.3543, 0.3550, 0.3556, 0.3561, 0.3565, 0.3569,
         0.3572, 0.3575, 0.3578, 0.3580, 0.3582, 0.3584, 0.3586, 0.3587, 0.3589, 0.3590]
HISTORIES = {                                   # SFR(t0 - tau), tau = lookback [Myr]
    "constant":            lambda tau: np.ones_like(tau),
    "rising, e-fold 300":  lambda tau: np.exp(-tau / 300.0),
    "rising, e-fold 100":  lambda tau: np.exp(-tau / 100.0),
    "rising, e-fold 30":   lambda tau: np.exp(-tau / 30.0),
    "burst 0-10 Myr ago":  lambda tau: (tau < 10.0).astype(float),
    "burst 20-30 Myr ago": lambda tau: ((tau >= 20.0) & (tau < 30.0)).astype(float),
}

def load(path):
    with h5py.File(path, "r") as f:
        nf, na, nz = (int(f[k][()]) for k in ("Number_of_freq", "N_age", "N_metallicity"))
        freq = f["freq"][()]
        age = np.maximum(f["age"][()], 1e-6)
        Zg = f["metallicity"][()]
        Lnu = f["Lnu"][()].reshape(nz, na, nf)
    la = np.log10(age)
    dlog = la[na // 2] - la[na // 2 - 1] if na > 1 else 0.1
    hi, lo = 10**(la + 0.5 * dlog), 10**(la - 0.5 * dlog)
    lo[0] = 0.0
    return freq, Zg, Lnu, lo * GYR2SEC / MYR_S, hi * GYR2SEC / MYR_S

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bpass", default=BPASS_FILE)
    p.add_argument("--Z", type=float, nargs="+", default=[1e-3, 4e-3, 0.02])
    args = p.parse_args()
    freq, Zg, Lnu, lo, hi = load(args.bpass)
    dtau = 0.01
    for band, (nu_lo, nu_hi) in BANDS.items():
        m = (freq >= nu_lo) & (freq <= nu_hi)
        nu = freq[m]
        nu_LL = nu_hi
        tau = np.arange(0.5 * dtau, TAU_MAX[band], dtau)
        ib = np.clip(np.searchsorted(hi, tau, side="right"), 0, hi.size - 1)
        print(f"\n=== {band} band ({nu_lo:.4e} - {nu_hi:.4e} Hz, {m.sum()} table points), "
              f"tau_max = {TAU_MAX[band]:g} Myr ===")
        for Z in args.Z:
            j = int(np.argmin(np.abs(np.log10(Zg) - np.log10(Z))))
            P = Lnu[j][:, m] / (PLANCK * nu) * (MP_CGS / MSOL2GRAM)          # (N_age, N_nu) per s per Hz
            P_band = np.trapezoid(P, nu, axis=1)                              # (N_age,)
            dt_s = (hi - lo) * MYR_S
            N_nu = (P * dt_s[:, None]).sum(0)                                 # lifetime spectrum
            N_band = (P_band * dt_s).sum()
            print(f"  Z = {Zg[j]:g}")
            print(f"    {'history':22s} {'J-weighted':>10s}   approx/exact mid-window:"
                  + "".join(f"  n={n:<4d}" for n in (2, 4, 6, 10)))
            for name, h in HISTORIES.items():
                S = np.bincount(ib, weights=h(tau) * dtau * MYR_S, minlength=hi.size)   # per age bin
                exact = S @ P
                sfr_eff = (S @ P_band) / N_band
                approx = sfr_eff * N_nu
                num = den = 0.0
                for n in range(2, 31):
                    if F_REC[n] == 0.0:
                        continue
                    a, b = nu_LL * (1 - n**-2.0), nu_LL * (1 - (n + 1)**-2.0)
                    w = (nu >= a) & (nu <= b)
                    if w.sum() < 2:
                        continue
                    num += F_REC[n] * np.trapezoid(approx[w], nu[w])
                    den += F_REC[n] * np.trapezoid(exact[w], nu[w])
                mid = [0.5 * nu_LL * ((1 - n**-2.0) + (1 - (n + 1)**-2.0)) for n in (2, 4, 6, 10)]
                at = [np.interp(v, nu, approx) / np.interp(v, nu, exact) for v in mid]
                print(f"    {name:22s} {num / den:10.4f}   " + "".join(f"  {x:6.3f}" for x in at))
    print("\nJ-weighted = 1 means SFR_eff x N_nu reproduces the J_alpha-weighted emissivity exactly.")

if __name__ == "__main__":
    main()
