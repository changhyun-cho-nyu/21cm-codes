"""
jeans_floor_check.py
====================
Does AREPO's Jeans pressure floor (ENFORCE_JEANS_STABILITY_OF_CELLS) change the stored gas temperature of the cold
neutral IGM? (user, 2026-11-25.) Both X-ray test runs and Lumina are compiled with it and with _EEOS.

What the code does (AREPO, src/hydro/update_primitive_variables.cc:374-397): P = max(P_th, (gamma-1) rho u_J),
u_J = 2 G m / (a eps), eps = ForceSoftening of the cell's adaptive softening type (2.8 x MinimumComovingHydroSoftening
x AdaptiveHydroSofteningSpacing^k, k from GasSoftFactor x the cell radius, grav_softening.cc:136-148). Only Pressure is
changed, not Utherm, but the HLLC flux takes the energy density as press/(gamma-1) + KE (riemann_hllc.cc:263) and
Utherm = (E - KE)/m, so the floor does pdV work on the thermal energy. Define T_J = (gamma-1) u_J mu m_p / k (the
cell's own mu, as for T; T_J / T = P_J / P_th). Hubble cooling is removed by X = T / (1+z)^2. Then
  adiabatic:             dX = (2/3) X     dln(1+delta)
  floor (T_J >> T):      dX = (2/3) X_J   dln(1+delta)     (reversible: compression heats, expansion cools)
  shocks, numerical dissipation: entropy can only rise.   Radiative terms in neutral metal-free gas at T < 1e4 K:
  negligible; Compton only heats while T < T_CMB. The 5 K clamp (MinGasTemp, RT_thermochem.cc:325, 575) only adds.
So a neutral, metal-free, non-star-forming cell that expands and ends BELOW its adiabatic temperature is the floor's
signature (or mixing by mass exchange between cells: reported by mass-change class).

scan (one run, two full snapshots s1 < s2 in time; a batch job, ~2.2e8 cells per snapshot, whole box):
  1. per snapshot, neutral (x_HII < 1e-3) non-star-forming cells by 1+delta, H I weight X m (1 - x_HII): share with the
     floor active (T_J > T), median T_J / T, median T_J and T, share below 5.5 K.
  2. cells followed by ParticleIDs from s1 to s2 (IDs present once in each), kept if non-star-forming and neutral at
     both, metal-free at both (if GFM_Metallicity exists), |m2/m1 - 1| < 0.1 (refinement / derefinement excluded).
     lr = ln(rho2/rho1); T2ad = T1 ((1+z2)/(1+z1))^2 (rho2/rho1)^(2/3); E = T2 - T2ad (non-adiabatic change at s2);
     F = (2/3) TJbar lr, TJbar = mean of the endpoint X_J, at z2. Predictions with the 5 K clamp:
       E_ad = max(T2ad, 5) - T2ad          E_fl = max(T2ad + F, 5) - T2ad
     Per lr bin (H I weight at s2): median T1, T2ad, T2, E (16/50/84), E_ad, E_fl, F, T_J; share with T2 < 0.8 T2ad
     among cells with T2ad > 7 K ("cooled below adiabatic"), observed and as E_fl predicts; share at the clamp;
     median E/F over the cells where neither T2ad nor T2ad + F is below 5.5 K (|lr| >= 0.1; the clamp would
     otherwise set E: with T_J ~ 300 K, expanding cells mostly fall in that case, so for them the cooled share is
     the test).
     Mixing check: the cooled-below-adiabatic share of expanding cells for |m2/m1 - 1| < 0.02 and 0.02-0.1.
  3. at s2, cells at T > 5.5 K, in 0.05-dex bins of 1+delta: H I-weighted slope of log T on log m. Floor-heated gas
     follows T_J ~ m^(2/3) at fixed density (slope of log T_J printed for comparison); shocks and adiabatic: no m.
  4. (added 2026-11-25 after the first run) the same matched cells split by their start state: at the clamp
     (T1 < 5.5 K), floor on (T1 < T_J1) or floor off (T1 >= T_J1), in bins of ln(rho2/rho1) and of 1+delta at s2.
     The floor's extra work is F' = (2/3) max(TJbar - Tpath, 0) ln(rho2/rho1), Tpath = (T1 g2 + T2ad)/2 (zero above
     T_J; section 2's F uses (2/3) TJbar and overestimates it). At the same compression and density: if the floor
     adds its work, E - F' (floor on) = E (floor off); if it does not, E (floor on) = E (floor off). Clamp-start
     cells show whether gas leaves the clamp by the floor's work (E ~ F') or otherwise; a floor + clamp ratchet
     within the interval would give E > F' there.
  writes <out>/<run>/jeans_floor/jeans_<s1>_<s2>.npz and the tables on stdout.
report: the tables of every pair found for a run, and data/jeans_floor_<run>.png.

  python jeans_floor_check.py scan --run withoutXrays --snaps 24 36
  python jeans_floor_check.py report --run withoutXrays
Inputs are read only (Oliver's runs); outputs only under /orcd/data/mvogelsb/004/chcho.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pair_cells_xray import (SRC, OUT, RUNS, X_H, GAMMA_M1, K_B, M_P, XNEU, check_out, wpct, unique_sorted,  # noqa: E402
                             snap_files)

G_CGS = 6.6743e-8
T_FLOOR, T_FLOOR_TOL, T_COLD_MIN = 5.0, 5.5, 7.0
MASS_TOL = 0.1
CBINS = (0.0, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0, np.inf)               # 1+delta
LR_EDGES = np.round(np.arange(-1.5, 2.51, 0.1), 2)                  # ln(rho2/rho1)
LD_SLOPE = np.round(np.arange(-0.5, 0.51, 0.05), 2)                 # log10(1+delta), mass-slope test
FLAGS = ("ENFORCE_JEANS_STABILITY_OF_CELLS", "ENFORCE_JEANS_STABILITY_OF_CELLS_EEOS", "ADAPTIVE_HYDRO_SOFTENING",
         "USE_ENTROPY_FOR_COLD_FLOWS", "JEANS_TOTPRESSURE_LIMIT", "UVB_OFF", "MRT_XRAY")
SPLIT_LR = (-0.5, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)       # ln(rho2/rho1), section 4
SPLIT_D = (0.0, 1.0, 2.0, 3.0, 10.0, np.inf)                        # 1+delta at s2, section 4
SPLIT_COLS = ("n", "w", "d50", "T1", "TJ", "F50", "E16", "E50", "E84", "EmF50", "EoverF50", "lnT2T2ad50", "fell")
NSOFTTYPES_HYDRO = 64
INK, INK2, GRID = "#1f1f1e", "#6b6b68", "#e6e6e3"
PAIR_COLORS = ("#2a78d6", "#eb6834", "#2e9e5b", "#8a55c7", "#c23b54", "#6b6b68")


def value(hd, par, key, default=None):
    for d in (hd, par):
        if key in d:
            return float(np.ravel(d[key])[0])
    if default is None:
        sys.exit(f"STOP: {key} in neither Header nor Parameters")
    return default


def read_cells(src, run, snap):
    fs = snap_files(src, run, snap)
    with h5py.File(fs[0], "r") as f:
        hd = dict(f["Header"].attrs)
        par = dict(f["Parameters"].attrs) if "Parameters" in f else {}
        cfg = set(f["Config"].attrs.keys()) if "Config" in f else None
    z, box = value(hd, par, "Redshift"), value(hd, par, "BoxSize")
    nfiles = int(value(hd, par, "NumFilesPerSnapshot", len(fs)))
    ntot = int(hd["NumPart_Total"][0]) + (int(hd["NumPart_Total_HighWord"][0]) << 32 if "NumPart_Total_HighWord" in hd else 0)
    if len(fs) != nfiles:
        sys.exit(f"STOP: {run} {snap}: {len(fs)} files found, the header says {nfiles}")
    uv = value(hd, par, "UnitVelocity_in_cm_per_s", 1e5)
    ul, um = value(hd, par, "UnitLength_in_cm"), value(hd, par, "UnitMass_in_g")
    meta = dict(snap=snap, z=z, box=box, uv=uv, G=G_CGS * um / (ul * uv**2),
                gsf=value(hd, par, "GasSoftFactor"), smin=value(hd, par, "MinimumComovingHydroSoftening", 0.0),
                sp=value(hd, par, "AdaptiveHydroSofteningSpacing", 0.0), Tmin=value(hd, par, "MinGasTemp", np.nan))
    if cfg is None:
        sys.exit(f"STOP: no Config group in {fs[0]}: cannot tell which floor was compiled")
    meta["flags"] = {k: (k in cfg) for k in FLAGS}
    meta["adaptive"] = meta["flags"]["ADAPTIVE_HYDRO_SOFTENING"]
    if meta["adaptive"] and (meta["smin"] <= 0 or meta["sp"] < 1):
        sys.exit("STOP: ADAPTIVE_HYDRO_SOFTENING without MinimumComovingHydroSoftening / AdaptiveHydroSofteningSpacing")
    print(f"{run} snapshot {snap}: z = {z:.5f}, BoxSize {box}, {len(fs)} files, {ntot:,} gas cells; "
          f"G = {meta['G']:.6g} (code), GasSoftFactor {meta['gsf']}, MinimumComovingHydroSoftening {meta['smin']}, "
          f"spacing {meta['sp']}, MinGasTemp {meta['Tmin']}", flush=True)
    print("  Config: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in meta["flags"].items()), flush=True)

    keys = ("id", "m", "rho", "u", "x", "ne", "sf", "Z")
    C = {k: [] for k in keys}
    hasZ = None
    nread = 0
    for p in fs:
        with h5py.File(p, "r") as f:
            if "PartType0" not in f:
                continue
            g = f["PartType0"]
            n = g["Masses"].shape[0]
            if n == 0:
                continue
            for k in ("ParticleIDs", "Masses", "Density", "InternalEnergy", "HII_Fraction"):
                if k not in g:
                    sys.exit(f"STOP: {k} missing in {p}; available: {sorted(g.keys())}")
            if hasZ is None:
                hasZ = "GFM_Metallicity" in g
                print(f"  GFM_Metallicity {'present (metal-free cut on)' if hasZ else 'MISSING (no metal-free cut)'}; "
                      f"StarFormationRate {'present' if 'StarFormationRate' in g else 'MISSING (no SF cut)'}; "
                      f"Ne from {'ElectronAbundance' if 'ElectronAbundance' in g else 'HII + HeII + 2 HeIII'}", flush=True)
            x = g["HII_Fraction"][()].astype(np.float32)
            if "ElectronAbundance" in g:
                ne = g["ElectronAbundance"][()].astype(np.float32)
            else:
                f2 = g["HeII_Fraction"][()].astype(np.float32) if "HeII_Fraction" in g else 0.0
                f3 = g["HeIII_Fraction"][()].astype(np.float32) if "HeIII_Fraction" in g else 0.0
                ne = (x + f2 + 2.0 * f3).astype(np.float32)
            C["id"].append(g["ParticleIDs"][()].astype(np.uint64))
            C["m"].append(g["Masses"][()].astype(np.float32))
            C["rho"].append(g["Density"][()].astype(np.float32))
            C["u"].append(g["InternalEnergy"][()].astype(np.float32))
            C["x"].append(x)
            C["ne"].append(ne)
            C["sf"].append((g["StarFormationRate"][()] > 0) if "StarFormationRate" in g else np.zeros(n, bool))
            C["Z"].append((g["GFM_Metallicity"][()] > 0) if hasZ else np.zeros(n, bool))
            nread += n
    if nread != ntot:
        sys.exit(f"STOP: {run} {snap}: read {nread:,} cells, header says {ntot:,}")
    C = {k: np.concatenate(v) for k, v in C.items()}
    meta["hasZ"] = bool(hasZ)
    meta["rho_mean"] = C["m"].astype(np.float64).sum() / box**3
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * C["ne"].astype(np.float64)) * M_P
    tfac = GAMMA_M1 * uv**2 * mu / K_B                              # T = tfac u
    C["T"] = (tfac * C["u"]).astype(np.float32)
    C["TJ"] = (tfac * jeans_u(C["m"], C["rho"], meta)).astype(np.float32)
    del mu, tfac
    print(f"  read {nread:,} cells; mean gas density {meta['rho_mean']:.6g} (code); "
          f"metal-enriched {C['Z'].mean():.3e}, star-forming {C['sf'].mean():.3e} of the cells", flush=True)
    return C, meta


def jeans_u(m, rho, meta):
    """u_J = 2 G m / (a eps), eps as AREPO assigns it (code units, physical specific energy like InternalEnergy)."""
    m = m.astype(np.float64)
    r = (3.0 * m / rho.astype(np.float64) / (4.0 * np.pi))**(1.0 / 3.0)    # comoving cell radius
    soft = meta["gsf"] * r
    if meta["adaptive"]:
        f0 = 2.8 * meta["smin"]
        k = np.where(soft <= f0, 0, np.floor(0.5 + np.log(np.maximum(soft, f0) / f0) / np.log(meta["sp"])))
        eps = f0 * meta["sp"]**np.minimum(k, NSOFTTYPES_HYDRO - 1)
    else:
        eps = soft
    return 2.0 * meta["G"] * m * (1.0 + meta["z"]) / eps


def snapshot_table(C, meta):
    one = C["rho"].astype(np.float64) / meta["rho_mean"]
    sel = (~C["sf"]) & (C["x"] < XNEU)
    w = X_H * C["m"].astype(np.float64) * (1.0 - C["x"])
    tot = w[sel].sum()
    rows = []
    print(f"\n  snapshot {meta['snap']} (z = {meta['z']:.3f}), neutral non-star-forming cells by 1+delta (H I weight):")
    print(f"  {'1+delta':>10} {'HI share':>9} {'floor on':>9} {'TJ/T p50':>9} {'TJ p50':>8} {'T p50':>8} {'T<5.5K':>7}")
    for lo, hi in zip(CBINS[:-1], CBINS[1:]):
        s = sel & (one >= lo) & (one < hi)
        ws = w[s]
        if ws.sum() <= 0:
            rows.append([lo, hi] + [np.nan] * 6)
            continue
        T, TJ = C["T"][s].astype(np.float64), C["TJ"][s].astype(np.float64)
        row = [lo, hi, ws.sum() / tot, ws[TJ > T].sum() / ws.sum(), wpct(TJ / T, ws, [50])[0], wpct(TJ, ws, [50])[0],
               wpct(T, ws, [50])[0], ws[T < T_FLOOR_TOL].sum() / ws.sum()]
        rows.append(row)
        print(f"  {lo:4.1f}-{hi:<5.1f} {row[2]:9.4f} {row[3]:9.4f} {row[4]:9.2f} {row[5]:8.1f} {row[6]:8.1f} {row[7]:7.3f}")
    return np.array(rows)


def slope_table(C, meta):
    """H I-weighted slope of log T (and of log T_J) on log m at fixed 1+delta, cells at T > 5.5 K."""
    ld = np.log10(C["rho"].astype(np.float64) / meta["rho_mean"])
    sel = (~C["sf"]) & (C["x"] < XNEU) & (~C["Z"]) & (C["T"] > T_FLOOR_TOL)
    rows = []
    print(f"\n  snapshot {meta['snap']}: slope d log T / d log m at fixed 1+delta (T > 5.5 K; floor-heated: ~2/3, "
          f"shocks / adiabatic: 0)")
    print(f"  {'log(1+d)':>12} {'cells':>11} {'m p5-p95 / median':>18} {'slope T':>8} {'slope TJ':>9}")
    for lo, hi in zip(LD_SLOPE[:-1], LD_SLOPE[1:]):
        s = sel & (ld >= lo) & (ld < hi)
        n = int(s.sum())
        if n < 1000:
            rows.append([lo, hi, n, np.nan, np.nan, np.nan])
            continue
        lm = np.log10(C["m"][s].astype(np.float64))
        w = X_H * C["m"][s].astype(np.float64) * (1.0 - C["x"][s])
        out = []
        for y in (np.log10(C["T"][s].astype(np.float64)), np.log10(C["TJ"][s].astype(np.float64))):
            mx, my = np.average(lm, weights=w), np.average(y, weights=w)
            out.append(np.sum(w * (lm - mx) * (y - my)) / np.sum(w * (lm - mx)**2))
        p5, p50, p95 = wpct(lm, w, [5, 50, 95])
        rows.append([lo, hi, n, 10**(p95 - p5), out[0], out[1]])
        print(f"  {lo:+5.2f}..{hi:+5.2f} {n:11,d} {10**(p95 - p5):10.2f} / {10**p50:.3g} {out[0]:8.3f} {out[1]:9.3f}")
    return np.array(rows)


def split_table(cls, lr, one2, E, Ftrue, T1, TJbar, T2, T2ad, w, z2):
    """Section 4: the non-adiabatic change E by start state (clamp / floor on / floor off), ln(rho2/rho1) and 1+delta2.

    The floor's extra work is Ftrue = (2/3) max(TJbar - Tpath, 0) lnrho (zero where the gas is above T_J). If the floor
    adds its work, E - Ftrue of floor-on cells matches E of floor-off cells (and clamp cells) at the same compression
    and density; if it does not, E itself matches. Rows: (class, lr bin, density bin); plus an aggregate over
    0.1 <= lnrho < 0.5 per density bin (last lr index)."""
    nL, nD = len(SPLIT_LR) - 1, len(SPLIT_D) - 1
    lb = np.digitize(lr, SPLIT_LR) - 1
    db = np.digitize(one2, SPLIT_D) - 1
    out = np.full((3, nL + 1, nD, len(SPLIT_COLS)), np.nan)
    wtot = w.sum()
    unc = (np.minimum(T2ad, T2ad + Ftrue) > T_FLOOR_TOL) & (np.abs(lr) >= 0.1) & (Ftrue != 0)
    cold = T2ad > T_COLD_MIN
    agg = (lr >= 0.1) & (lr < 0.5)
    for c in range(3):
        for L in range(nL + 1):
            sel_l = agg if L == nL else (lb == L)
            for D in range(nD):
                j = np.flatnonzero((cls == c) & sel_l & (db == D))
                if len(j) < 200:
                    continue
                wj = w[j]
                row = dict(n=len(j), w=wj.sum() / wtot)
                row["d50"], row["T1"], row["TJ"], row["F50"] = (wpct(v[j], wj, [50])[0] for v in (one2, T1, TJbar, Ftrue))
                row["E16"], row["E50"], row["E84"] = wpct(E[j], wj, [16, 50, 84])
                row["EmF50"] = wpct(E[j] - Ftrue[j], wj, [50])[0]
                u = unc[j]
                row["EoverF50"] = wpct(E[j][u] / Ftrue[j][u], wj[u], [50])[0] if u.sum() >= 100 else np.nan
                row["lnT2T2ad50"] = wpct(np.log(T2[j] / T2ad[j]), wj, [50])[0]
                cc = cold[j]
                row["fell"] = (wj[cc][T2[j][cc] < 0.8 * T2ad[j][cc]].sum() / wj[cc].sum()) if cc.sum() >= 100 else np.nan
                out[c, L, D] = [row[k] for k in SPLIT_COLS]

    def g(c, L, D, k):
        return out[c, L, D, SPLIT_COLS.index(k)]

    print(f"\n4. split by the start state: clamp (T1 < {T_FLOOR_TOL} K), floor on (T1 < T_J1), floor off (T1 >= T_J1)"
          f"\n   F' = (2/3) max(T_J - T, 0) ln(rho2/rho1): the floor's extra work (0 above T_J). If the floor adds its work,"
          f"\n   E - F' (floor on) = E (floor off) at the same compression and density; if not, E (floor on) = E (floor off)."
          f"\n   Medians, H I weight, T in K at z = {z2:.2f}; n = cells.")
    print(f"   {'lnrho':>11} {'1+d2':>8} | {'clamp: n':>10} {'E':>7} {'F_':>6} | {'on: n':>10} {'T1':>6} {'T_J':>5} {'E':>7} "
          f"{'F_':>6} {'E-F_':>7} {'E/F_':>6} | {'off: n':>10} {'T1':>6} {'E':>7}")
    for L in list(range(nL)) + [nL]:
        lab = "0.10..0.50" if L == nL else f"{SPLIT_LR[L]:+.2f}..{SPLIT_LR[L + 1]:+.2f}"
        if L == nL:
            print("   aggregate over 0.1 <= lnrho < 0.5:")
        for D in range(nD):
            if all(np.isnan(g(c, L, D, "n")) for c in range(3)):
                continue
            dl = f"{SPLIT_D[D]:g}-{SPLIT_D[D + 1]:g}"

            def f(c, k, fmt):
                v = g(c, L, D, k)
                return format(v, fmt) if np.isfinite(v) else " " * len(format(0.0, fmt))
            print(f"   {lab:>11} {dl:>8} | {f(0, 'n', '10,.0f')} {f(0, 'E50', '+7.1f')} {f(0, 'F50', '+6.1f')} | "
                  f"{f(1, 'n', '10,.0f')} {f(1, 'T1', '6.0f')} {f(1, 'TJ', '5.0f')} {f(1, 'E50', '+7.1f')} {f(1, 'F50', '+6.1f')} "
                  f"{f(1, 'EmF50', '+7.1f')} {f(1, 'EoverF50', '+6.2f')} | {f(2, 'n', '10,.0f')} {f(2, 'T1', '6.0f')} "
                  f"{f(2, 'E50', '+7.1f')}")
    return out


def scan(a):
    od = os.path.join(a.out, a.run, "jeans_floor")
    check_out(od, a.allow_any_out)
    s1, s2 = a.snaps
    A, ma = read_cells(a.src, a.run, s1)
    B, mb = read_cells(a.src, a.run, s2)
    if mb["z"] >= ma["z"]:
        sys.exit(f"STOP: snapshot {s2} (z = {mb['z']}) is not later than {s1} (z = {ma['z']})")
    if not (ma["flags"]["ENFORCE_JEANS_STABILITY_OF_CELLS"] and mb["flags"]["ENFORCE_JEANS_STABILITY_OF_CELLS"]):
        print("WARNING: ENFORCE_JEANS_STABILITY_OF_CELLS not compiled: T_J is computed anyway, but no floor acts")

    print("\n1. floor against the thermal pressure, per snapshot")
    tabA, tabB = snapshot_table(A, ma), snapshot_table(B, mb)

    print("\n3. mass dependence at s2")
    slope = slope_table(B, mb)

    # 2. follow cells by ParticleIDs
    sa, ia, da = unique_sorted(A["id"])
    sb, ib, db = unique_sorted(B["id"])
    pos = np.minimum(np.searchsorted(sb, sa), len(sb) - 1)
    hit = sb[pos] == sa
    i1, i2 = ia[hit], ib[pos[hit]]
    del sa, sb, ia, ib, pos, hit
    nA, nB = len(A["id"]), len(B["id"])
    wB = X_H * B["m"].astype(np.float64) * (1.0 - B["x"])
    print(f"\n2. cells followed by ParticleIDs {s1} -> {s2}: {len(i1):,} of {nA:,} / {nB:,}; duplicated IDs dropped "
          f"{da:,} / {db:,}; matched share of the s2 H I {wB[i2].sum() / wB.sum():.4f}")
    q = {k: A[k][i1] for k in ("m", "rho", "T", "TJ", "x", "sf", "Z")}
    r = {k: B[k][i2] for k in ("m", "rho", "T", "TJ", "x", "sf", "Z")}
    del A, B, i1, i2
    neutral = (~q["sf"]) & (~r["sf"]) & (q["x"] < XNEU) & (r["x"] < XNEU) & (~q["Z"]) & (~r["Z"])
    dm = np.abs(r["m"].astype(np.float64) / q["m"] - 1.0)
    w = X_H * r["m"].astype(np.float64) * (1.0 - r["x"])
    wn = w[neutral].sum()
    keep = neutral & (dm < MASS_TOL)
    print(f"   neutral, non-star-forming, metal-free at both: {neutral.sum():,} cells; H I share with |dm/m| < "
          f"{MASS_TOL}: {w[keep].sum() / wn:.4f} (< 0.02: {w[neutral & (dm < 0.02)].sum() / wn:.4f})")
    q = {k: v[keep] for k, v in q.items()}
    r = {k: v[keep] for k, v in r.items()}
    w, dm = w[keep], dm[keep]
    g2 = ((1.0 + mb["z"]) / (1.0 + ma["z"]))**2
    ratio = r["rho"].astype(np.float64) / q["rho"]
    lr = np.log(ratio)
    T1, T2 = q["T"].astype(np.float64), r["T"].astype(np.float64)
    T2ad = T1 * g2 * ratio**(2.0 / 3.0)
    TJbar = 0.5 * (q["TJ"].astype(np.float64) * g2 + r["TJ"])
    F = GAMMA_M1 * TJbar * lr
    E = T2 - T2ad
    Ead = np.maximum(T2ad, T_FLOOR) - T2ad
    Efl = np.maximum(T2ad + F, T_FLOOR) - T2ad
    cold = T2ad > T_COLD_MIN
    unclamped = (np.minimum(T2ad, T2ad + F) > T_FLOOR_TOL) & (np.abs(lr) >= 0.1)   # E/F only where the clamp is not hit
    # split by the start state (section 4): 0 = at the clamp, 1 = floor on (T1 < T_J1), 2 = floor off (T1 >= T_J1)
    TJ1 = q["TJ"].astype(np.float64)
    cls = np.where(T1 < T_FLOOR_TOL, 0, np.where(T1 < TJ1, 1, 2)).astype(np.int8)
    Tpath = 0.5 * (T1 * g2 + T2ad)                                  # gas T along the path, at z2
    Ftrue = GAMMA_M1 * np.maximum(TJbar - Tpath, 0.0) * lr          # the floor's extra work only where T < T_J
    one2 = r["rho"].astype(np.float64) / mb["rho_mean"]
    del q, r, ratio, TJ1

    nb = len(LR_EDGES) - 1
    cols = ("n", "w", "T1", "T2ad", "T2", "E16", "E50", "E84", "Ead50", "Efl50", "F50", "TJ50",
            "fell_obs", "fell_fl", "clamp_obs", "EoverF50")
    tab = {c: np.full(nb, np.nan) for c in cols}
    k = np.digitize(lr, LR_EDGES) - 1
    o = np.argsort(k, kind="stable")
    ks = k[o]
    starts = np.searchsorted(ks, np.arange(nb + 1))
    print(f"\n   per ln(rho2/rho1) bin (H I weight at s2; T in K at z = {mb['z']:.2f}; fell = T2 < 0.8 T2ad among "
          f"T2ad > {T_COLD_MIN:g} K; clamp = T2 < {T_FLOOR_TOL} K among T2ad > {T_COLD_MIN:g} K)")
    print(f"   {'lr':>11} {'HI share':>8} {'T1':>7} {'T2ad':>7} {'T2':>7} {'E p16/p50/p84':>24} {'E_ad':>6} "
          f"{'E_fl':>8} {'T_J':>6} {'fell obs':>8} {'fell fl':>8} {'clamp':>6} {'E/F':>6}  (E/F where unclamped)")
    wtot = w.sum()
    for b in range(nb):
        j = o[starts[b]:starts[b + 1]]
        if len(j) < 100:
            continue
        wj = w[j]
        tab["n"][b], tab["w"][b] = len(j), wj.sum() / wtot
        tab["T1"][b], tab["T2ad"][b], tab["T2"][b] = (wpct(v[j], wj, [50])[0] for v in (T1, T2ad, T2))
        tab["E16"][b], tab["E50"][b], tab["E84"][b] = wpct(E[j], wj, [16, 50, 84])
        tab["Ead50"][b], tab["Efl50"][b], tab["F50"][b], tab["TJ50"][b] = (wpct(v[j], wj, [50])[0]
                                                                            for v in (Ead, Efl, F, TJbar))
        c = cold[j]
        if wj[c].sum() > 0:
            wc = wj[c]
            tab["fell_obs"][b] = wc[T2[j][c] < 0.8 * T2ad[j][c]].sum() / wc.sum()
            tab["fell_fl"][b] = wc[(T2ad[j][c] + Efl[j][c]) < 0.8 * T2ad[j][c]].sum() / wc.sum()
            tab["clamp_obs"][b] = wc[T2[j][c] < T_FLOOR_TOL].sum() / wc.sum()
        u = unclamped[j]
        if abs(0.5 * (LR_EDGES[b] + LR_EDGES[b + 1])) >= 0.1 and u.sum() >= 100:
            tab["EoverF50"][b] = wpct(E[j][u] / F[j][u], wj[u], [50])[0]
        print(f"   {LR_EDGES[b]:+5.2f}..{LR_EDGES[b + 1]:+5.2f} {tab['w'][b]:8.4f} {tab['T1'][b]:7.1f} "
              f"{tab['T2ad'][b]:7.1f} {tab['T2'][b]:7.1f} {tab['E16'][b]:+8.1f}/{tab['E50'][b]:+7.1f}/{tab['E84'][b]:+7.1f} "
              f"{tab['Ead50'][b]:6.1f} {tab['Efl50'][b]:+8.1f} {tab['TJ50'][b]:6.0f} {tab['fell_obs'][b]:8.3f} "
              f"{tab['fell_fl'][b]:8.3f} {tab['clamp_obs'][b]:6.3f} {tab['EoverF50'][b]:+6.2f}")

    print("\n   summary (H I weight):")
    summ = []
    for name, s in (("expanding lr < -0.1", lr < -0.1), ("|lr| <= 0.1", np.abs(lr) <= 0.1), ("compressed lr > 0.1", lr > 0.1)):
        sc = s & cold
        fo = w[sc & (T2 < 0.8 * T2ad)].sum() / max(w[sc].sum(), 1e-300)
        ff = w[sc & ((T2ad + Efl) < 0.8 * T2ad)].sum() / max(w[sc].sum(), 1e-300)
        su = s & unclamped
        ef = wpct(E[su] / F[su], w[su], [50])[0] if su.sum() >= 100 else np.nan
        summ.append([w[s].sum() / wtot, fo, ff, ef])
        print(f"   {name:22s} H I share {w[s].sum() / wtot:.3f}; cooled below 0.8 T2ad (T2ad > {T_COLD_MIN:g} K): "
              f"observed {fo:.3f}, floor predicts {ff:.3f}, adiabatic + shocks 0; median E/F {ef:+.2f} "
              f"(cells where neither T2ad nor T2ad + F is below 5.5 K; floor alone 1, adiabatic 0)")
    mix = []
    for lo, hi in ((0.0, 0.02), (0.02, MASS_TOL)):
        sc = (lr < -0.1) & cold & (dm >= lo) & (dm < hi)
        fo = w[sc & (T2 < 0.8 * T2ad)].sum() / max(w[sc].sum(), 1e-300)
        mix.append([lo, hi, w[sc].sum() / wtot, fo])
        print(f"   mixing check, expanding cells with {lo:.2f} <= |dm/m| < {hi:.2f}: H I share {w[sc].sum() / wtot:.3f}, "
              f"cooled below 0.8 T2ad {fo:.3f}")

    split = split_table(cls, lr, one2, E, Ftrue, T1, TJbar, T2, T2ad, w, mb["z"])

    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"jeans_{s1:03d}_{s2:03d}.npz")
    np.savez(fn, run=a.run, s1=s1, s2=s2, z1=ma["z"], z2=mb["z"], lr_edges=LR_EDGES, cbins=np.array(CBINS),
             tabA=tabA, tabB=tabB, slope=slope, summary=np.array(summ), mix=np.array(mix),
             cols=np.array(cols), **{f"bin_{c}": tab[c] for c in cols},
             split=split, split_lr=np.array(SPLIT_LR), split_d=np.array(SPLIT_D), split_cols=np.array(SPLIT_COLS),
             flags=np.array([f"{k}={v}" for k, v in mb["flags"].items()]))
    print(f"\n   wrote {fn}")


def report(a):
    fs = sorted(glob.glob(os.path.join(a.out, a.run, "jeans_floor", "jeans_*_*.npz")))
    if not fs:
        sys.exit(f"no jeans_*.npz under {os.path.join(a.out, a.run, 'jeans_floor')}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, fn in enumerate(fs):
        d = np.load(fn)
        c = PAIR_COLORS[i % len(PAIR_COLORS)]
        x = 0.5 * (d["lr_edges"][1:] + d["lr_edges"][:-1])
        lab = f"{int(d['s1']):03d}->{int(d['s2']):03d} (z {float(d['z1']):.1f}->{float(d['z2']):.1f})"
        print(f"\n{a.run} {lab}: [expanding, |lr| <= 0.1, compressed] H I share, cooled below 0.8 T2ad observed / "
              f"floor-predicted, median E/F")
        for nm, row in zip(("expanding", "static", "compressed"), d["summary"]):
            print(f"   {nm:10s} {row[0]:.3f}  {row[1]:.3f} / {row[2]:.3f}  {row[3]:+.2f}")
        if "split" in d.files:
            S, sc, sd = d["split"], list(d["split_cols"]), d["split_d"]
            print("   split, compressed 0.1 <= lnrho < 0.5, medians [K]: 1+d2 | clamp E, F' | on E, F', E-F' | off E")
            for D in range(len(sd) - 1):
                v = lambda c, k: S[c, -1, D, sc.index(k)]   # noqa: E731
                if np.isnan(v(1, "n")) and np.isnan(v(2, "n")):
                    continue
                print(f"   {sd[D]:g}-{sd[D + 1]:g} | {v(0, 'E50'):+7.1f} {v(0, 'F50'):+6.1f} | {v(1, 'E50'):+7.1f} "
                      f"{v(1, 'F50'):+6.1f} {v(1, 'EmF50'):+7.1f} | {v(2, 'E50'):+7.1f}")
        ok = np.isfinite(d["bin_E50"])
        ax[0].fill_between(x[ok], d["bin_E16"][ok], d["bin_E84"][ok], color=c, alpha=0.15, lw=0)
        ax[0].plot(x[ok], d["bin_E50"][ok], color=c, lw=1.8, label=f"{lab}: observed")
        ax[0].plot(x[ok], d["bin_Efl50"][ok], color=c, lw=1.2, ls="--", label="floor prediction")
        ax[0].plot(x[ok], d["bin_Ead50"][ok], color=c, lw=1.0, ls=":", label="adiabatic + 5 K clamp")
        ax[1].plot(x[ok], d["bin_fell_obs"][ok], color=c, lw=1.8, label=f"{lab}: observed")
        ax[1].plot(x[ok], d["bin_fell_fl"][ok], color=c, lw=1.2, ls="--", label="floor prediction")
    ax[0].axhline(0, color=INK2, lw=0.8)
    ax[0].set_xlabel(r"$\ln(\rho_2/\rho_1)$")
    ax[0].set_ylabel(r"$T_2 - T_{2,\rm ad}$ [K] (median, 16-84%)")
    ax[0].set_yscale("symlog", linthresh=10)
    ax[1].set_xlabel(r"$\ln(\rho_2/\rho_1)$")
    ax[1].set_ylabel(r"H I share with $T_2 < 0.8\,T_{2,\rm ad}$ ($T_{2,\rm ad} > 7$ K)")
    for x_ in ax:
        x_.grid(color=GRID, lw=0.6)
        x_.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{a.run}: neutral metal-free gas cells followed by ID, non-adiabatic change vs the Jeans-floor pdV work",
                 fontsize=10, color=INK)
    fig.tight_layout()
    os.makedirs(a.data, exist_ok=True)
    out = os.path.join(a.data, f"jeans_floor_{a.run}.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--run", choices=RUNS, required=True)
    s.add_argument("--snaps", type=int, nargs=2, required=True, metavar=("S1", "S2"))
    s.add_argument("--src", default=SRC)
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)   # mock tests only
    r = sp.add_parser("report")
    r.add_argument("--run", choices=RUNS, required=True)
    r.add_argument("--out", default=os.environ.get("BASE", OUT))
    r.add_argument("--data", default="data")
    a = p.parse_args()
    {"scan": scan, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
