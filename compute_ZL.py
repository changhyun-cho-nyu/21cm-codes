"""
compute_ZL.py
=============
Luminosity-weighted stellar metallicity Z_L in the H I (10.2-13.6 eV) and He II (40.8-54.4 eV) Lyman bands, and how
well a single metallicity reproduces the band photon output of the young-star metallicity mix that the J_alpha codes
use (user, 2026-11-24: people ask whether our metallicity is luminosity-weighted).

Inputs: the BPASS table and the band photon rates built exactly as in the C codes (make_Z_snapshots.bpass_band_rates),
the young-star weights Z_weights.txt (w_j(z): initial-mass fraction of the stars younger than 20 Myr at each BPASS
metallicity Z_j), and optionally Z_snapshots.csv (Z_young_20, Z_mean_mw) and the production SFR_eff_summary.csv
(box-mean SFR history).

Two definitions, per band b, with N_j the lifetime yield and l_j(tau) the rate at age tau:
  birth    (at each weight row z): the light of the stars born at z over their lifetime
             Z_L = sum_j w_j N_j Z_j / sum_j w_j N_j,         <N> = sum_j w_j N_j
  now      (with --sfr-summary, at each snapshot t): the light emitted at t by all earlier populations, each with its
           birth weights w_j(t') and current age t - t' (the global SFR history, linear in t between snapshots, zero
           before the first; the same model as make_SFR_eff.py, without the window truncation)
             Z_L(t) = Int SFR(t') sum_j w_j(t') l_j(t-t') Z_j dt' / Int SFR(t') sum_j w_j(t') l_j(t-t') dt'
Z_L is computed on the grid metallicities Z_j (each star split between its two neighbours linearly in log Z, as the
weights are built), so it is the luminosity-weighted mean of the grid representation.

Single-metallicity tests (what one Z gives for the band photon output, relative to the mix; 1 = exact):
  N_b(Z) / <N_b>   for Z = Z_L of the same band, Z_L of the other band, Z_young_20 (initial-mass-weighted mean of the
                   stars < 20 Myr), Z_mean_mw (all stars, the old code)
  Z_eff,b: the single Z with N_b(Z_eff) = <N_b> (log-Z interpolation as eps_b() in the C codes; nan if no root)

Outputs (--out, default ZL/): ZL_birth.csv (one row per weight row) and, with --sfr-summary, ZL_now.csv (one row per
snapshot). Printed: a table at selected redshifts.

  python compute_ZL.py --weights Z_weights.txt --zsnap Z_snapshots.csv \\
      --sfr-summary /orcd/data/mvogelsb/004/chcho/SFR_eff/SFR_eff_summary.csv --out /orcd/data/mvogelsb/004/chcho/ZL
  python compute_ZL.py --selftest
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from make_Z_snapshots import BPASS_FILE, at_Z, bpass_band_rates, t_of_a

BANDS = ("H", "He")
Z_PRINT = (20.0, 15.0, 12.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0)


def read_weights(path):
    """Z_weights.txt -> (z ascending, W [n_row, n_Z], Z grid from the header or None)"""
    zgrid = None
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if "Z_grid" in s:
                    zgrid = np.array([float(x) for x in s.split("=", 1)[1].split()])
                continue
            rows.append([float(x) for x in s.split()])
    a = np.array(rows)
    o = np.argsort(a[:, 0])
    return a[o, 0], a[o, 1:], zgrid


def weights_at(z, zw, W):
    """w_j(z): linear in z between rows, end rows held outside (as Z_weights_at() in compute_Jalpha.c)"""
    z = np.atleast_1d(np.asarray(z, float))
    out = np.empty((len(z), W.shape[1]))
    for j in range(W.shape[1]):
        out[:, j] = np.interp(z, zw, W[:, j])          # np.interp holds the end values outside
    return out


def z_eff(target, logZ, N):
    """single Z with at_Z(Z) = target on the grid (log-Z linear segments); nan if none, the lowest-Z root if several"""
    for k in range(len(logZ) - 1):
        a, b = N[k] - target, N[k + 1] - target
        if a == 0:
            return 10**logZ[k]
        if a * b < 0:
            return 10**(logZ[k] + (logZ[k + 1] - logZ[k]) * a / (a - b))
    return 10**logZ[-1] if N[-1] == target else np.nan


def birth_table(zw, W, Zg, N, zsnap):
    logZ = np.log10(Zg)
    out = {"redshift": zw}
    for b in BANDS:
        L = W * N[b][None, :]                           # light per unit initial mass, by grid metallicity
        tot = L.sum(axis=1)
        out[f"ZL_{b}"] = (L * Zg[None, :]).sum(axis=1) / tot
        out[f"N_mix_{b}"] = tot
        out[f"Zeff_{b}"] = np.array([z_eff(t, logZ, N[b]) for t in tot])
    out["ZM_grid"] = (W * Zg[None, :]).sum(axis=1)     # initial-mass-weighted mean on the grid
    df = pd.DataFrame(out)
    if zsnap is not None:
        d = zsnap.sort_values("redshift")
        for c in ("Z_young_20", "Z_mean_mw"):
            df[c] = np.interp(df["redshift"], d["redshift"], d[c]) if c in d else np.nan
    for b in BANDS:
        o = "He" if b == "H" else "H"
        tests = {"ZL": df[f"ZL_{b}"], f"ZL{o}": df[f"ZL_{o}"]}
        for c in ("Z_young_20", "Z_mean_mw"):
            if c in df:
                tests[c] = df[c]
        for name, Zs in tests.items():
            df[f"single_{name}_{b}"] = [at_Z(z, logZ, N[b]) for z in Zs] / df[f"N_mix_{b}"]
    return df


def now_table(summary, zw, W, Zg, tau_lo, tau_hi, rates, dtau=0.2):
    """light emitted at each snapshot by all earlier populations (global SFR history)"""
    s = pd.read_csv(summary).sort_values("snap")
    z, t, sfr = s["redshift"].to_numpy(), s["t_Myr"].to_numpy(), np.maximum(s["mean_SFR"].to_numpy(), 0.0)
    if np.any(np.diff(t) <= 0):
        sys.exit(f"ERROR: {summary}: times not increasing")
    Zg = np.asarray(Zg)
    rows = []
    for k in range(len(t)):
        if sfr[:k + 1].max() <= 0:
            rows.append({"snap": int(s["snap"].iloc[k]), "redshift": z[k], "t_Myr": t[k]})
            continue
        age = np.arange(0.5 * dtau, t[k] - t[0], dtau)              # midpoints, cohorts formed after t[0]
        tf = t[k] - age
        S = np.interp(tf, t, sfr, left=0.0)                          # SFR linear in t, zero before the first snapshot
        zf = np.interp(tf, t, z)                                     # formation redshift
        w = weights_at(zf, zw, W)                                    # birth weights
        ib = np.clip(np.searchsorted(tau_hi, age, side="right"), 0, len(tau_hi) - 1)
        row = {"snap": int(s["snap"].iloc[k]), "redshift": z[k], "t_Myr": t[k]}
        for b in BANDS:
            l = rates[b][:, ib].T                                    # (n_age, n_Z), constant within each BPASS bin
            l = np.where(age[:, None] < tau_hi[-1], l, 0.0)
            light = (S[:, None] * w * l).sum(axis=0)                 # per grid Z
            tot = light.sum()
            row[f"ZL_{b}"] = (light * Zg).sum() / tot if tot > 0 else np.nan
            row[f"young100_share_{b}"] = (S[:, None] * w * l)[age < 100].sum() / tot if tot > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def show(df, cols, fmt="{:.3g}"):
    d = df.sort_values("redshift", ascending=False)
    pick = sorted({int(np.abs(d["redshift"] - zz).to_numpy().argmin()) for zz in Z_PRINT})
    d = d.iloc[pick]
    print("  " + "  ".join(f"{c:>14s}" for c in cols))
    for _, r in d.iterrows():
        print("  " + "  ".join(f"{fmt.format(r[c]) if pd.notna(r[c]) else 'nan':>14s}" for c in cols))


def selftest():
    """planted BPASS-like rates with known integrals; checks the definitions against hand values"""
    Zg = np.array([1e-4, 1e-3, 1e-2])
    tau_lo = np.array([0.0, 10.0, 100.0]); tau_hi = np.array([10.0, 100.0, 1000.0])   # Myr
    rates = {"H": np.array([[3.0, 1.0, 0.1], [2.0, 1.0, 0.1], [1.0, 1.0, 0.1]]),
             "He": np.array([[9.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
    MYR = 3.15576e13
    N = {b: (r * (tau_hi - tau_lo)[None, :] * MYR).sum(axis=1) for b, r in rates.items()}
    zw = np.array([3.0, 10.0]); W = np.array([[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]])
    df = birth_table(zw, W, Zg, N, None)
    for b in BANDS:
        hand = (0.5 * N[b][0] * 1e-4 + 0.5 * N[b][2] * 1e-2) / (0.5 * N[b][0] + 0.5 * N[b][2])
        assert np.allclose(df[f"ZL_{b}"], hand, rtol=1e-12), (b, df[f"ZL_{b}"].iloc[0], hand)
        ze = df[f"Zeff_{b}"].iloc[0]
        assert np.isclose(at_Z(ze, np.log10(Zg), N[b]), df[f"N_mix_{b}"].iloc[0], rtol=1e-10)
    assert np.allclose(df["ZM_grid"], 0.5 * 1e-4 + 0.5 * 1e-2)
    # "now" with constant SFR and constant weights over a long time -> the birth value (Int l dtau = N)
    t = np.arange(0.0, 3000.0 + 1e-9, 2.5)
    a = np.linspace(0.1, 0.3, len(t))
    zz = 1.0 / a - 1.0
    tmp = os.path.join(os.environ.get("TMPDIR", "."), "ZL_selftest_summary.csv")
    pd.DataFrame({"snap": np.arange(len(t)), "redshift": zz, "t_Myr": t, "mean_SFR": 1.0}).to_csv(tmp, index=False)
    nw = now_table(tmp, zw, W, Zg, tau_lo, tau_hi, rates, dtau=0.1)
    os.remove(tmp)
    last = nw.iloc[-1]
    for b in BANDS:
        assert np.isclose(last[f"ZL_{b}"], df[f"ZL_{b}"].iloc[0], rtol=1e-3), (b, last[f"ZL_{b}"], df[f"ZL_{b}"].iloc[0])
    print("selftest: birth Z_L = hand values (1e-12), Z_eff closes, Z_M grid, now -> birth for steady SFR (1e-3): OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bpass", default=BPASS_FILE)
    p.add_argument("--weights", default="Z_weights.txt")
    p.add_argument("--zsnap", default="Z_snapshots.csv", help="for Z_young_20 and Z_mean_mw ('none' to skip)")
    p.add_argument("--sfr-summary", default="", help="production SFR_eff_summary.csv for the 'now' definition")
    p.add_argument("--out", default="ZL")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    Zg, tau_lo, tau_hi, rates = bpass_band_rates(a.bpass)
    MYR = 3.15576e13
    N = {b: (rates[b] * ((tau_hi - tau_lo) * MYR)[None, :]).sum(axis=1) for b in BANDS}
    zw, W, zgrid = read_weights(a.weights)
    if W.shape[1] != len(Zg) or (zgrid is not None and not np.allclose(zgrid, Zg, rtol=1e-6)):
        sys.exit(f"ERROR: weights have {W.shape[1]} columns / Z_grid {zgrid}, BPASS grid {Zg}")
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-6):
        sys.exit("ERROR: weight rows do not sum to 1")
    zsnap = None if a.zsnap.lower() == "none" else pd.read_csv(a.zsnap)
    os.makedirs(a.out, exist_ok=True)

    print("BPASS grid Z:", " ".join(f"{z:.3g}" for z in Zg))
    for b in BANDS:
        print(f"lifetime yield {b:2s} [photons/baryon]:", " ".join(f"{x:.4g}" for x in N[b]))
    bt = birth_table(zw, W, Zg, N, zsnap)
    fn = os.path.join(a.out, "ZL_birth.csv")
    bt.to_csv(fn, index=False)
    print(f"\nbirth (stars < 20 Myr at z, lifetime light), {len(bt)} rows -> {fn}")
    cols = ["redshift", "Z_mean_mw", "Z_young_20", "ZM_grid", "ZL_H", "ZL_He", "Zeff_H", "Zeff_He"]
    show(bt, [c for c in cols if c in bt])
    print("\nsingle-Z band output / mix (1 = exact)")
    cols = ["redshift"] + [c for c in bt.columns if c.startswith("single_")]
    show(bt, cols, "{:.4f}")
    if a.sfr_summary:
        nt = now_table(a.sfr_summary, zw, W, Zg, tau_lo, tau_hi, rates)
        fn = os.path.join(a.out, "ZL_now.csv")
        nt.to_csv(fn, index=False)
        print(f"\nnow (light emitted at t by all earlier populations, global SFR history) -> {fn}")
        show(nt, ["redshift", "ZL_H", "ZL_He", "young100_share_H", "young100_share_He"])


if __name__ == "__main__":
    main()
