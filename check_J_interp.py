"""
check_J_interp.py
=================
How good is the H-band J_alpha interpolated in time between the production J snapshots, as lightcone_J.cone_J does on
the light cone, at a render snapshot that is not in the J list?

The production H J (stars + AGN) exists at the 305 render snapshots of run_21cm.sh: every render to 155, then gaps of
2-10 renders (14 once, 596 -> 610). J computed directly at a render inside a gap (the production codes,
run_Jalpha_prod.sh with ONLY= and J_OUT= a separate directory, so that cone_J never sees it) is compared with
  J_lin  = (1 - w) J(a) + w J(b),  w = (z - z_a) / (z_b - z_a)   (cone_J time="linear"; a, b the bracketing J snapshots)
  J_near = the J snapshot nearer in z                             (cone_J time="nearest")
over the whole box (the cone samples box cells, so the box answers the cone's question for that z).

Reports
  J, for stars, AGN and the total: <J_x>/<J_true>; per cell e = J_x/J_true - 1: percentiles, J-weighted rms, volume
    fractions with |e| > 1, 5, 10 %; power ratio P_x/P_true and r(k) = P_xt/sqrt(P_xx P_tt) in log k bins.
  delta T_b (21 cm, real space, mass-weighted T_S inputs, as run_21cm.sh; render fields of the test snapshot for all
    three, so J is the only difference): <dT_b> ratio, rms(dT_b_x - dT_b_true) / std(dT_b_true), max |diff|, P ratio
    and r(k); <x_alpha> to show how strongly dT_b depends on J at that z.
Power spectra: mean removed, |FFT|^2 binned in |k| (the rfft half-space, each mode once); only ratios and r are
reported, so the normalisation cancels.

Usage
-----
  python check_J_interp.py --snap 625 --jtest /orcd/data/mvogelsb/004/chcho/Jalpha_interp_test \\
         --jprod /orcd/data/mvogelsb/004/chcho/Jalpha --dir <ren_640 that holds render 625> [--spin-model 1] [--out DIR]
  (run_J_interp_test.sh does the J and the comparison for the test snapshots)
Memory ~40 GB at 640^3 (J of three snapshots, three dT_b maps, FFTs). Exit status 1 on inconsistent inputs.
"""
import argparse
import os
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

try:
    import scipy.fft as sfft
    def rfftn(a):
        return sfft.rfftn(a, workers=os.cpu_count())
except ImportError:
    rfftn = np.fft.rfftn

from check_Tb_maps import field_path
from lightcone_J import j_files
from tb_physics import M_H, OMEGA_L_DEFAULT, X_H, Line

DSET = {"stars": "J_alpha", "agn": "Jalpha_AGN_H"}


def read_J(path, name):
    with h5py.File(path, "r") as f:
        d = f[name]
        return d[()].astype(np.float32), float(d.attrs["Redshift"]), dict(d.attrs)


def cell_stats(x, t):
    """statistics of e = x/t - 1 over the cells with t > 0 (float64 sums in slabs)"""
    m = t > 0
    e = (x[m].astype(np.float64) / t[m]) - 1.0
    w = t[m].astype(np.float64)
    out = {"mean_ratio": float(x.sum(dtype=np.float64) / t.sum(dtype=np.float64)),
           "pct": np.percentile(e, [1, 16, 50, 84, 99]),
           "rms_Jw": float(np.sqrt(np.sum(w * e * e) / np.sum(w))),
           "frac": [float(np.mean(np.abs(e) > c)) for c in (0.01, 0.05, 0.10)],
           "n_excl": int(m.size - m.sum())}
    return out


class Spectra:
    """auto and cross power of fields against a reference, in log |k| bins (k in h/cMpc)"""
    def __init__(self, N, box, per_dex=4):
        k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=box / N)
        kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=box / N)
        kk = np.sqrt(k1[:, None, None]**2 + k1[None, :, None]**2 + kz[None, None, :]**2)
        kmin, kmax = 2.0 * np.pi / box, np.pi * N / box
        self.edges = kmin * 10.0**(np.arange(0.0, np.log10(kmax / kmin) + 1.0 / per_dex, 1.0 / per_dex))
        self.idx = np.digitize(kk, self.edges).ravel().astype(np.int32)   # 0: k < kmin (the k = 0 mode), > nb: beyond
        self.nb = len(self.edges) - 1
        self.nmodes = np.bincount(self.idx, minlength=self.nb + 2)[1:self.nb + 1]

    def bin(self, a):
        return np.bincount(self.idx, weights=a.ravel(), minlength=self.nb + 2)[1:self.nb + 1]

    def compare(self, true, others):
        """{name: (P_x/P_t, r)} per bin"""
        Ft = rfftn(true - true.mean(dtype=np.float64))
        Pt = self.bin(np.abs(Ft)**2)
        out = {}
        for name, x in others.items():
            Fx = rfftn(x - x.mean(dtype=np.float64))
            Px, Pxt = self.bin(np.abs(Fx)**2), self.bin((Fx * np.conj(Ft)).real)
            out[name] = (Px / Pt, Pxt / np.sqrt(Px * Pt))
            del Fx
        return out

    def centres(self):
        return np.sqrt(self.edges[:-1] * self.edges[1:])


def print_spectra(sp, res, label):
    kc = sp.centres()
    names = list(res)
    print(f"  {label}: k [h/cMpc], modes, then P_x/P_true and r for " + ", ".join(names))
    for i in range(sp.nb):
        if sp.nmodes[i] == 0:
            continue
        cols = "  ".join(f"{res[n][0][i]:.5f} {res[n][1][i]:.6f}" for n in names)
        print(f"    {kc[i]:7.4f} {sp.nmodes[i]:10d}  {cols}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap", type=int, required=True, help="render snapshot of the test J (not in the production J list)")
    p.add_argument("--jtest", required=True, help="directory with Jalpha_H_NNN.hdf5 and Jalpha_AGN_H_NNN.hdf5 of --snap")
    p.add_argument("--jprod", required=True, help="production J directory (the J snapshots cone_J uses)")
    p.add_argument("--dir", required=True, help="render directory (ren_640) that holds --snap")
    p.add_argument("--spin-model", type=int, choices=[0, 1], required=True,
                   help="21 cm spin model of dT_b (1: the production maps from 2026-11-15)")
    p.add_argument("--planes", type=int, default=16, help="planes per slab for dT_b (memory)")
    p.add_argument("--out", default=None, help="directory for J_interp_NNN.npz (binned spectra and statistics)")
    a = p.parse_args()

    s = a.snap
    # --- test J and the bracketing production J snapshots (cone_J's rule)
    test = {c: read_J(os.path.join(a.jtest, f), DSET[c])
            for c, f in (("stars", f"Jalpha_H_{s}.hdf5"), ("agn", f"Jalpha_AGN_H_{s}.hdf5"))}
    z = test["stars"][1]
    if abs(test["agn"][1] - z) > 1e-6:
        sys.exit(f"ERROR: test stars z = {z}, AGN z = {test['agn'][1]}")
    files = j_files("H", a.jprod)
    if s in files:
        sys.exit(f"ERROR: snapshot {s} is a production J snapshot: nothing to interpolate")
    snaps = np.array(sorted(files, key=lambda q: files[q][2]))          # z increasing
    zs = np.array([files[q][2] for q in snaps])
    b = int(np.searchsorted(zs, z))
    if b == 0 or b == len(zs) or zs[b] == z:
        sys.exit(f"ERROR: z = {z} is not strictly inside the production J range {zs[0]} - {zs[-1]}")
    sa, sb = int(snaps[b - 1]), int(snaps[b])                           # sa: lower z (later), sb: higher z
    if not sb < s < sa:
        sys.exit(f"ERROR: bracketing J snapshots {sb}, {sa} (by z) do not bracket render {s} in order")
    w = (z - zs[b - 1]) / (zs[b] - zs[b - 1])
    near = sb if w >= 0.5 else sa
    print(f"snapshot {s}, z = {z:.6f}: production J snapshots {sb} (z = {zs[b]:.6f}) and {sa} (z = {zs[b - 1]:.6f}); "
          f"gap {sa - sb} renders; weights {sa}: {1 - w:.4f}, {sb}: {w:.4f}; nearest {near}")
    for c in ("stars", "agn"):
        at = test[c][2]
        print(f"  test {c}: J_mean {at.get('J_mean', np.nan):.6e}, J_scalar {at.get('J_scalar', np.nan):.6e}"
              + (f", SFR_input {at['SFR_input']!r}, tau_max_Myr {at.get('tau_max_Myr')}" if c == "stars" else ""))

    prod = {}
    for q in (sa, sb):
        for c, k in (("stars", 0), ("agn", 1)):
            prod[(q, c)], zq, at = read_J(files[q][k], DSET[c])
            if c == "stars":
                tag_t, tag_p = test["stars"][2].get("SFR_input"), at.get("SFR_input")
                if tag_t != tag_p or test["stars"][2].get("tau_max_Myr") != at.get("tau_max_Myr"):
                    sys.exit(f"ERROR: test and production stellar J made from different inputs: {tag_t!r} vs {tag_p!r}")
    N = test["stars"][0].shape[0]

    # --- J statistics
    comps = {}
    for c in ("stars", "agn"):
        t = test[c][0]
        lin = ((1.0 - w) * prod[(sa, c)] + w * prod[(sb, c)]).astype(np.float32)
        comps[c] = (t, lin, prod[(near, c)])
    tot = tuple(comps["stars"][i] + comps["agn"][i] for i in range(3))
    comps["total"] = tot

    with h5py.File(os.path.join(a.dir, "All", f"All_{s:03d}.hdf5"), "r") as f:
        hd = dict(f["Header"].attrs)
    zr = float(hd["Redshift"]) if "Redshift" in hd else 1.0 / float(hd["Time"]) - 1.0
    if abs(zr - z) > 1e-4 * (1.0 + z):
        sys.exit(f"ERROR: render z = {zr}, test J z = {z}")
    h = float(hd["HubbleParam"])
    box = float(hd["BoxSize"]) / 1000.0 if "BoxSize" in hd else float(test["stars"][2]["BoxSize_cMpc"]) * h   # cMpc/h
    sp = Spectra(N, box)
    rec = {"snap": s, "z": z, "sa": sa, "sb": sb, "w": w, "near": near, "k": sp.centres(), "nmodes": sp.nmodes}

    print("\nJ_alpha (proper photons s^-1 cm^-2 Hz^-1 sr^-1):")
    print("  component  method  <J_x>/<J_true>   e = J_x/J_true - 1: p1 p16 p50 p84 p99   J-weighted rms   "
          "|e| > 1% 5% 10%   (cells with J_true <= 0 excluded)")
    for c, (t, lin, nr) in comps.items():
        for name, x in (("linear", lin), ("nearest", nr)):
            st = cell_stats(x, t)
            pc = " ".join(f"{v:+.4f}" for v in st["pct"])
            fr = " ".join(f"{v:.4f}" for v in st["frac"])
            print(f"  {c:6s} {name:8s} {st['mean_ratio']:.6f}   {pc}   {st['rms_Jw']:.5f}   {fr}   ({st['n_excl']})")
            rec[f"J_{c}_{name}"] = np.array([st["mean_ratio"], *st["pct"], st["rms_Jw"], *st["frac"]])
        print(f"  {c:6s} <J_true> = {t.mean(dtype=np.float64):.6e}")
    for c in ("stars", "agn", "total"):
        t, lin, nr = comps[c]
        res = sp.compare(t, {"linear": lin, "nearest": nr})
        print_spectra(sp, res, f"P_J {c}")
        for n, (pr, r) in res.items():
            rec[f"PJ_{c}_{n}"], rec[f"rJ_{c}_{n}"] = pr, r
    Jt, Jl, Jn = comps["total"]
    del comps, prod, test, tot

    # --- delta T_b with each J (render fields of snapshot s)
    om, ob = float(hd.get("Omega0", 0.30964144)), float(hd.get("OmegaBaryon", 0.048975))
    ol = float(hd.get("OmegaLambda", OMEGA_L_DEFAULT))
    rho_unit = float(hd["UnitMass_in_g"]) * h**2 / float(hd["UnitLength_in_cm"])**3
    L = Line("H", z, h, om, ob, ol, spin_model=a.spin_model)
    fh = {}
    def fld(name):
        if name not in fh:
            fh[name] = h5py.File(field_path(a.dir, name, s), "r")
        return fh[name][name]
    if fld("Density").shape != Jt.shape:
        sys.exit(f"ERROR: render {fld('Density').shape}, J {Jt.shape}")
    rho_mean = sum(fld("Density")[i:i + 32].astype(np.float64).sum() for i in range(0, N, 32)) / float(N)**3
    Tb = {n: np.empty(Jt.shape, np.float32) for n in ("true", "linear", "nearest")}
    xa_sum = 0.0
    for i0 in range(0, N, a.planes):
        sl = slice(i0, min(N, i0 + a.planes))
        rho = fld("Density")[sl].astype(np.float64)
        T = fld("Temperature")[sl].astype(np.float64)
        xHII = fld("HII_Fraction")[sl].astype(np.float64)
        fII = fld("HeII_Fraction")[sl].astype(np.float64)
        fIII = fld("HeIII_Fraction")[sl].astype(np.float64)
        nH = X_H * rho * rho_unit * (1.0 + z)**3 / M_H
        sig = rho * (1.0 - xHII) / rho_mean
        for n, J in (("true", Jt), ("linear", Jl), ("nearest", Jn)):
            tb, _, xa = L.tb(sig, nH, T, xHII, fII, fIII, J[sl].astype(np.float64))
            Tb[n][sl] = tb
            if n == "true":
                xa_sum += xa.sum()
    for f in fh.values():
        f.close()
    t = Tb["true"].astype(np.float64)
    std_t = t.std()
    print(f"\ndelta T_b (21 cm, spin model {a.spin_model}, real space; mK): <dT_b_true> = {t.mean():+.6e}, "
          f"std {std_t:.6e}, <x_alpha> (true J) = {xa_sum / Jt.size:.4e}")
    print("  method   <dT_b_x>/<dT_b_true>   rms(diff)/std(true)   max|diff| [mK]")
    for n in ("linear", "nearest"):
        d = Tb[n].astype(np.float64) - t
        mr = Tb[n].mean(dtype=np.float64) / t.mean()
        print(f"  {n:8s} {mr:.6f}   {np.sqrt(np.mean(d * d)) / std_t:.3e}   {np.abs(d).max():.3e}")
        rec[f"Tb_{n}"] = np.array([mr, np.sqrt(np.mean(d * d)) / std_t, np.abs(d).max()])
        del d
    del t
    res = sp.compare(Tb["true"], {"linear": Tb["linear"], "nearest": Tb["nearest"]})
    print_spectra(sp, res, "P_dTb")
    for n, (pr, r) in res.items():
        rec[f"PTb_{n}"], rec[f"rTb_{n}"] = pr, r

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"J_interp_{s:03d}.npz")
        np.savez(fn, **rec)
        print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
