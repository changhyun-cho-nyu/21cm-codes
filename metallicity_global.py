"""
metallicity_global.py
=====================
Global delta T_b of the 21 cm and 3.46 cm lines for the 13 BPASS metallicities, each applied to every star at
every redshift (Appendix sec:metallicity; user, 2026-11-23: global evolution only, the only difference from the
production run is the metallicity). Everything else is the production pipeline: effective SFR (make_SFR_eff.py),
f_esc = 0.18 and Lumina's IGM transmission in the He band, the AGN J unchanged, spin model 1, real space,
mass-weighted fields.

Method (approximation: J_star,j(x) = R_j(z) J_star,fid(x) in every pixel)
-----------------------------------------------------------------------
1. R   : R_j(z) = <J_star,j>/<J_star,fid> per line, from the box means alone, exactly:
         SFR_eff is linear in SFR, so <SFR_eff_j>(t_k) = sum_m W^j_km <SFR>(t_m), with W^j the make_SFR_eff.py
         weights for all stars at Z_j (W^fid from the production Z_weights.txt; checked against the production
         SFR_eff_summary.csv), and <J> is the converged integral of check_J_mean.py (H) or its He-band version with
         f_esc and the box-mean IGM transmission <T>(R) of the snapshot's transmission file (as J_scalar_attenuated).
         The quadrature offset of the C shell sum (<= 0.8%, JALPHA_NOTES 4.4) cancels in the ratio.
         Neglected: the spatial variation of J_star,j/J_star,fid (each voxel's own SFR history inside the 300/100 Myr
         window; the spectral shape changes the weight of near and far shells). It enters the global mean only at
         second order (through the non-linear T_S for H, through its correlation with dT_b/dJ for He).
2. map640 --snap N : 21 cm, 640^3 render pixels (low z): dT_b for J = R_j J_star + J_AGN in every pixel, all 14
         variants (fiducial + 13 Z) in one read of the fields; the fiducial mean is checked against the production RIN
         map (--rin).
3. box --line L --snap N --task T : gas cells (H at z >= 8, He at z <= 10), EVERY cell of the full snapshot (user,
         2026-11-24: whole box, no cubes or other samples). The rows are the gas-cell scan's row blocks
         (test_cells_Tb.py, cell_test[_H]_NN/scan_L_NNN_*.hdf5), each split into --split tasks; J of each cell =
         the scan's J at its pixel (the same files, float32 operations, render axis order and z interpolation) +
         (R_j - 1) J_star. The J files are identified by reproducing J stored by the scan exactly (a file check).
         Two more variants (user, 2026-11-25), for the radiative share of the coupling: 'no_star' (J = J_AGN only:
         the AGN + collision floor) and 'no_J' (J = 0: collisions only). Radiative share = 1 - <dT_b>_no_J/<dT_b>_fid.
   box-merge --line L --snap N : sums all tasks (they must tile the snapshot once) -> whole-box <dT_b> per variant;
         check: every scan block's fiducial sum V and sum V dT_b must equal the gas-cell scan's (same cells, J,
         physics), to --tol-sum.
4. table : every box-merge result and map640 row -> one CSV (+ PNG with --png).

Usage (ORCD; job scripts run_metal_*.sh)
-----
  python metallicity_global.py R --sfr-summary .../SFR_eff/SFR_eff_summary.csv --weights .../Z_weights.txt \\
         --jdir .../Jalpha [.../Jalpha_fullsnap ...] --out .../metallicity/R_Z.csv [--snaps ...]
  python metallicity_global.py map640 --snap 300 --dir <ren_640> --rcsv R_Z.csv --jdir .../Jalpha \\
         --rin .../Tb_maps_prod/H_real_wf/RIN_300.hdf5 --out .../metallicity
  python metallicity_global.py box --line He --snap 116 --snapdir <snapshots> --scan-dir .../cell_test_116 \\
         --rcsv R_Z.csv --jdir .../Jalpha --task 0 --split 4 --out .../metallicity   [--max-rows 4000000: timing]
  python metallicity_global.py box-merge --line He --snap 116 --scan-dir .../cell_test_116 --out .../metallicity
  python metallicity_global.py table --out .../metallicity [--png]
"""
import argparse
import csv
import glob
import os
import sys
import time
from collections import defaultdict

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import check_J_mean as CJ

LINES = ("H", "He")
NU_LL = {"H": CJ.NU_LL, "He": 1.31583e16}           # compute_Jalpha.c / compute_Jalpha_He.c
F_ESC = {"H": 1.0, "He": 0.18}                       # F_ESC_STAR_HE (cancels in R; kept for the absolute check)
TAU_MAX = {"H": 300.0, "He": 100.0}                  # make_SFR_eff.py windows
J_NAME = {"H": "Jalpha_H_{}.hdf5", "He": "Jalpha_He_{}_{}.hdf5"}
AGN_NAME = {"H": "Jalpha_AGN_H_{}.hdf5", "He": "Jalpha_AGN_He_{}_{}.hdf5"}
EXTRA = ("no_star", "no_J")                           # box variants after the 13 Z: J = J_AGN only, J = 0


def variant_J(v, nZ, Jc, Jstar, Jagn, R):
    """J of box variant v in the cells: 0 = the scan's J (fiducial), 1..nZ = stars at Z_j (Jc + sum (R_j - 1) J_star),
    nZ+1 = 'no_star' (the AGN J of the files), nZ+2 = 'no_J' (J = 0). Jstar, Jagn: per-side weighted arrays; R: per-side
    R arrays (index 0 = fiducial = 1)."""
    if v == 0:
        return Jc
    if v <= nZ:
        return Jc + sum((r[v] - 1.0) * js for r, js in zip(R, Jstar))
    if v == nZ + 1:
        return sum(Jagn)
    if v == nZ + 2:
        return np.zeros_like(Jc)
    raise ValueError(f"no variant {v}")


def zlabel(Z):
    return f"{Z:.0e}" if Z < 1e-3 else f"{Z:g}"


# ---------------------------------------------------------------------------------------------------------------------
# 1. R_j(z)
# ---------------------------------------------------------------------------------------------------------------------
def read_summary(fn):
    """production SFR_eff_summary.csv: per snapshot z, t, box-mean raw SFR (floored at 0), <SFR_eff>/<SFR> per band"""
    with open(fn) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["snap"]))
    snaps = np.array([int(r["snap"]) for r in rows])
    if not np.array_equal(snaps, np.arange(CJ.N_SNAPS)):
        sys.exit(f"ERROR: {fn} has {len(rows)} rows, need snapshots 0..{CJ.N_SNAPS - 1} (stride 1, all planes)")
    z = np.array([float(r["redshift"]) for r in rows])
    if np.any(np.diff(z) >= 0):
        sys.exit(f"ERROR: {fn}: redshifts not decreasing with the snapshot number")
    sfr = np.array([float(r["mean_SFR"]) for r in rows])
    ratio = {b: np.array([float(r[f"ratio_{b}"]) for r in rows]) for b in LINES}
    return z, sfr, ratio


def one_hot_emissivity(base, j):
    """the C eps_mix with all weight at BPASS metallicity j at every z"""
    e = CJ.Emissivity.__new__(CJ.Emissivity)
    e.freq, e.N_nu, e.logZ = base.freq, base.N_nu, base.logZ
    e.zw_z = np.array([0.0])
    e.zw_w = np.zeros((1, base.N_nu.shape[0]))
    e.zw_w[0, j] = 1.0
    return e


def trans_mean_vec(R, Rg, m):
    """jalpha_transmission.h trans_mean, vectorised: linear in R below the first radius (T = 1 at R = 0), linear in
    ln R between the stored radii, held beyond the last"""
    R = np.asarray(R, float)
    out = np.interp(np.log(np.maximum(R, Rg[0])), np.log(Rg), m)
    return np.where(R <= Rg[0], 1.0 + (m[0] - 1.0) * R / Rg[0], out)


class BandMeanJ:
    """box-mean J of one line: the converged integral of check_J_mean.BoxMeanJ.conv, with the line's Lyman limit,
    f_esc and (He) the box-mean IGM transmission of the snapshot (as the C J_scalar_attenuated)"""
    def __init__(self, line, zs, sfr, eps, unit, trans=None):
        self.line, self.zs, self.sfr, self.eps, self.unit, self.trans = line, zs, sfr, eps, unit, trans
        self.za, self.sa = zs[::-1], sfr[::-1]
        self.nu_ll = NU_LL[line]

    def integrand(self, z, z_obs, n):
        nu = self.nu_ll * (1.0 - 1.0 / (n * n)) * (1.0 + z) / (1.0 + z_obs)
        v = ((1.0 + z_obs)**2 / (4.0 * np.pi) * CJ.F_REC[n] * F_ESC[self.line] * self.eps(nu, z)
             * CJ.C_CGS / CJ.H_cgs(z) * np.interp(z, self.za, self.sa) * self.unit / CJ.MP_CGS)
        if self.trans is not None:
            Rg, m2, m4 = self.trans
            v = v * trans_mean_vec(CJ.comoving_Mpc(z_obs, z, 60), Rg, m2 if n == 2 else m4)
        return v

    def conv(self, z_obs, order=6):
        x, wq = np.polynomial.legendre.leggauss(order)
        J = 0.0
        for n in range(2, CJ.N_MAX + 1):
            if CJ.F_REC[n] == 0.0:
                continue
            zmax = min(CJ.z_max_lyman(z_obs, n), self.zs[0])
            if zmax <= z_obs:
                continue
            zf = (1.0 + z_obs) * self.eps.freq / (self.nu_ll * (1.0 - 1.0 / (n * n))) - 1.0   # BPASS nodes in z'
            br = [np.array([z_obs, zmax]), self.zs, self.eps.zw_z, zf]
            if self.trans is not None:                                                     # kinks of <T>(R)
                Rg = self.trans[0]
                Rmax = float(CJ.comoving_Mpc(z_obs, np.array([zmax]), 500)[0])
                Rin = Rg[Rg < Rmax]
                if Rin.size:
                    br.append(CJ.R_to_z(z_obs, zmax, Rin))
            br = np.concatenate(br)
            br = np.unique(br[(br >= z_obs) & (br <= zmax)])
            mid, half = 0.5 * (br[1:] + br[:-1]), 0.5 * np.diff(br)
            z = mid[:, None] + half[:, None] * x[None, :]
            J += np.sum(self.integrand(z, z_obs, n) * wq[None, :] * half[:, None])
        return J


def find_file(dirs, name):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def CJ_first_3d(f):
    return f[next(k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3)]


def _dec(v):
    if isinstance(v, np.ndarray) and v.dtype.kind in "SO" and v.size:
        v = v.ravel()[0]
    if isinstance(v, bytes):
        return v.decode(errors="replace").strip("\x00 ")
    return v


def j_attrs(path):
    with h5py.File(path, "r") as f:
        return {k: _dec(v) for k, v in CJ_first_3d(f).attrs.items()}


def read_trans(path):
    with h5py.File(path, "r") as f:
        return (f["R_cMpc"][()].astype(float), f["mean_T_n2"][()].astype(float), f["mean_T_n4"][()].astype(float),
                float(f.attrs["Redshift"]))


def cmd_R(a):
    from make_SFR_eff import build_W, t_of_z
    from make_Z_snapshots import bpass_band_rates, BPASS_FILE
    bpass = a.bpass or BPASS_FILE
    zs, msfr, ratio = read_summary(a.sfr_summary)
    t = t_of_z(zs)
    Zg, tau_lo, tau_hi, rates = bpass_band_rates(bpass)
    from make_SFR_eff import read_weights
    wz, ww = read_weights(a.weights, Zg)
    wfile = {"H": a.weights, "He": a.weights_he or a.weights}
    bases = {line: CJ.Emissivity(bpass, wfile[line]) for line in LINES}
    if not np.allclose(10**bases["H"].logZ, Zg, rtol=1e-6):
        sys.exit("ERROR: BPASS metallicity grids differ between the two readers")
    if wfile["He"] != wfile["H"]:
        same = np.array_equal(bases["H"].zw_z, bases["He"].zw_z) and np.allclose(bases["H"].zw_w, bases["He"].zw_w, atol=0)
        print(f"He-band eps weights: {wfile['He']} ({'identical rows to' if same else 'DIFFERENT from'} {wfile['H']})")
    nZ = Zg.size
    unit = None
    if a.unit_from:
        unit = CJ.sfr_unit(a.unit_from)[0]
    print(f"BPASS {bpass}: {nZ} metallicities {', '.join(zlabel(z) for z in Zg)}; SFR_eff and H weights {a.weights} "
          f"({wz.size} rows)")
    print(f"SFR summary {a.sfr_summary}: {zs.size} snapshots, z = {zs[0]:.3f} .. {zs[-1]:.3f}")
    meanE = {}
    for line in a.lines:
        Wf = build_W(t, zs, rates[line], tau_lo, tau_hi, wz, ww, TAU_MAX[line])
        Ef = Wf @ msfr
        ok = msfr > 0
        dev = np.abs(Ef[ok] / (ratio[line][ok] * msfr[ok]) - 1.0)
        print(f"  {line}: W_fid <SFR> vs production <SFR_eff> (summary, 6 digits): max |rel diff| {dev.max():.2e} "
              f"over {ok.sum()} snapshots")
        if dev.max() > a.tol_w:
            sys.exit(f"STOP: the fiducial W does not reproduce the production SFR_eff box means (> {a.tol_w:g}); "
                     "wrong weights file, BPASS table or summary")
        E = [Ef]
        for j in range(nZ):
            oh = np.zeros((1, nZ)); oh[0, j] = 1.0
            E.append(build_W(t, zs, rates[line], tau_lo, tau_hi, np.array([0.0]), oh, TAU_MAX[line]) @ msfr)
        meanE[line] = np.array(E)                                   # (1 + nZ, 709)
        k = [int(np.argmin(np.abs(zs - zc))) for zc in (15, 10, 6, 4)]
        print(f"  {line}: <SFR_eff_j>/<SFR_eff_fid> at z = " + ", ".join(f"{zs[i]:.2f}" for i in k) + ": " +
              "; ".join(f"Z={zlabel(Zg[j])} " + "/".join(f"{meanE[line][1 + j, i] / Ef[i]:.3f}" if Ef[i] > 0 else "-"
                                                         for i in k) for j in (0, 6, 12)))
    out_rows = []
    for line in a.lines:
        if a.snaps:
            snaps = a.snaps
        else:
            pat = J_NAME[line].format("*", a.tr_tag) if line == "He" else J_NAME[line].format("*")
            snaps = sorted({int(os.path.basename(p).split("_")[2].split(".")[0]) for d in a.jdir for p in glob.glob(os.path.join(d, pat))})
        print(f"\n{line}: {len(snaps)} snapshots")
        print("  snap      z     <J>_conv,fid/J_file  " + " ".join(f"{zlabel(z):>7s}" for z in Zg))
        for s in snaps:
            z_obs = zs[s]
            name = J_NAME[line].format(s, a.tr_tag) if line == "He" else J_NAME[line].format(s)
            jf = find_file(a.jdir, name)
            at = j_attrs(jf) if jf else {}
            if jf and abs(float(at.get("Redshift", z_obs)) - z_obs) > 1e-4:
                sys.exit(f"STOP: {jf} is at z = {at['Redshift']}, snapshot {s} at z = {z_obs}")
            trans = None
            if line == "He":
                tp = None
                if a.trans_dir:
                    tp = os.path.join(a.trans_dir, f"transmission_He_{s:03d}.hdf5")
                elif jf and str(at.get("IGM_transmission", "none")).strip().startswith("/"):
                    tp = str(at["IGM_transmission"]).strip()
                if tp and os.path.exists(tp):
                    Rg, m2, m4, zt = read_trans(tp)
                    if abs(zt - z_obs) > 1e-4:
                        sys.exit(f"STOP: {tp} is at z = {zt}, snapshot {s} at z = {z_obs}")
                    trans = (Rg, m2, m4)
                elif not a.he_no_trans:
                    sys.exit(f"STOP: He snapshot {s}: no transmission file (J file {jf}, attribute "
                             f"{at.get('IGM_transmission')!r}); give --trans-dir, or --he-no-trans")
            base = bases[line]
            if jf and "Z_weights_obs" in at:          # the weights the J code used at z_obs, as it recorded them
                wj = np.asarray(at["Z_weights_obs"], float).ravel()
                wm = np.asarray(base.weights(z_obs), float).ravel()
                if wj.size != wm.size or np.max(np.abs(wj - wm)) > 1e-6:
                    sys.exit(f"STOP: {jf} recorded Z weights {np.round(wj, 5)} at z_obs, the weights file gives "
                             f"{np.round(wm, 5)}: give the weights file the J code used (--weights / --weights-he)")
            Jv = []
            u = unit if unit is not None else 1.0
            for v in range(1 + nZ):
                eps = base if v == 0 else one_hot_emissivity(base, v - 1)
                Jv.append(BandMeanJ(line, zs, meanE[line][v], eps, u, trans).conv(z_obs, a.order))
            Jv = np.array(Jv)
            R = np.where(Jv[0] > 0, Jv / np.where(Jv[0] > 0, Jv[0], 1.0), 1.0)
            ref = at.get("J_scalar_attenuated" if line == "He" and trans is not None else "J_mean")
            chk = Jv[0] / float(ref) if (unit is not None and ref not in (None, 0) and float(ref) > 0) else np.nan
            print(f"  {s:4d} {z_obs:8.4f}   {chk:10.5f}          " + " ".join(f"{r:7.4f}" for r in R[1:]), flush=True)
            out_rows.append(dict(line=line, snap=s, z=repr(float(z_obs)), J_file=jf or "", J_file_ref=ref if ref is not None else "",
                                 J_conv_fid=repr(float(Jv[0])), conv_over_file=chk, transmission=int(trans is not None),
                                 **{f"R_{j:02d}": repr(float(R[1 + j])) for j in range(nZ)}))
    if not out_rows:
        sys.exit("STOP: no snapshots (give --snaps or a --jdir with stellar J files)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fo:
        fo.write("# metallicity_global.py R: R_jj = <J_star>(all stars at Z_grid[jj]) / <J_star>(production Z mix)\n")
        fo.write("# Z_grid = " + " ".join(f"{z:g}" for z in Zg) + "\n")
        w = csv.DictWriter(fo, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {a.out}")


def read_R(fn):
    Zg = None
    rows = {}
    with open(fn) as f:
        lines = f.readlines()
    for l in lines:
        if l.startswith("# Z_grid ="):
            Zg = np.array(l.split("=")[1].split(), float)
    for r in csv.DictReader([l for l in lines if not l.startswith("#")]):
        rows[(r["line"], int(r["snap"]))] = (float(r["z"]), np.array([float(r[f"R_{j:02d}"]) for j in range(len(Zg))]))
    if Zg is None:
        sys.exit(f"ERROR: {fn} has no Z_grid line")
    return Zg, rows


def R_of(Rtab, line, snap, z_check=None):
    if (line, snap) not in Rtab:
        sys.exit(f"STOP: no R for {line} render snapshot {snap} in the R table (run 'R' with --snaps including it)")
    z, R = Rtab[(line, snap)]
    if z_check is not None and abs(z - z_check) > 1e-4:
        sys.exit(f"STOP: R table z = {z} for {line} {snap}, file z = {z_check}")
    return np.concatenate([[1.0], R])                              # variant 0 = fiducial


def load_3d(path, expect_z=None):
    with h5py.File(path, "r") as f:
        d = CJ_first_3d(f)
        zj = d.attrs.get("Redshift")
        arr = d[()].astype(np.float32)
    if expect_z is not None and zj is not None and abs(float(zj) - expect_z) > 1e-4:
        sys.exit(f"STOP: {path} is at z = {float(zj)}, expected {expect_z}")
    return arr


# ---------------------------------------------------------------------------------------------------------------------
# 2. 21 cm on the 640^3 render pixels
# ---------------------------------------------------------------------------------------------------------------------
def cmd_map640(a):
    from check_Tb_maps import field_path
    from tb_physics import Line, X_H, M_H, OMEGA_L_DEFAULT
    Zg, Rtab = read_R(a.rcsv)
    d, s = a.dir, a.snap
    with h5py.File(os.path.join(d, "All", f"All_{s:03d}.hdf5"), "r") as f:
        hd = dict(f["Header"].attrs)
    z = float(hd["Redshift"]) if "Redshift" in hd else 1.0 / float(hd["Time"]) - 1.0
    h = float(hd["HubbleParam"])
    om, ob = float(hd.get("Omega0", 0.30964144)), float(hd.get("OmegaBaryon", 0.048975))
    ol = float(hd.get("OmegaLambda", OMEGA_L_DEFAULT))
    rho_unit = float(hd["UnitMass_in_g"]) * h**2 / float(hd["UnitLength_in_cm"])**3
    R = R_of(Rtab, "H", s, z)
    js = a.J_star or find_file(a.jdir, J_NAME["H"].format(s))
    ja = a.J_agn or find_file(a.jdir, AGN_NAME["H"].format(s))
    if not js or not ja:
        sys.exit(f"STOP: J files for H render {s} not found in {a.jdir}")
    files = {}

    def fld(name):
        if name not in files:
            files[name] = h5py.File(field_path(d, name, s), "r")
        return files[name][name]
    N = fld("Density").shape[0]
    rho_mean = 0.0
    for i0 in range(0, N, 32):
        rho_mean += fld("Density")[i0:i0 + 32].astype(np.float64).sum()
    rho_mean /= float(N)**3
    L = Line("H", z, h, om, ob, ol, spin_model=1)
    Js, Ja = load_3d(js, z), load_3d(ja, z)
    if Js.shape != (N,) * 3 or Ja.shape != (N,) * 3:
        sys.exit("ERROR: J shape differs from the render")
    st = 1                                             # every pixel (no subsampling)
    print(f"render {s} (z = {z:.4f}, N = {N}) in {d}; J {js} + {ja}; stride {st}; R = " +
          " ".join(f"{r:.4f}" for r in R[1:]), flush=True)
    nv = len(R)
    S = np.zeros(nv)
    npix = 0
    t0 = time.time()
    for i0 in range(0, N, a.planes):
        sl = slice(i0, min(N, i0 + a.planes))
        sub = (slice(None, None, st), slice(None, None, st)) if st > 1 else (slice(None), slice(None))
        pl = np.arange(sl.start, sl.stop)
        pl = pl[pl % st == 0]
        if pl.size == 0:
            continue
        take = lambda arr: arr[pl.min():pl.max() + 1:st][(slice(None),) + sub].astype(np.float64)
        rho = take(fld("Density"))
        T = take(fld("Temperature"))
        xHII = take(fld("HII_Fraction"))
        fII = take(fld("HeII_Fraction"))
        fIII = take(fld("HeIII_Fraction"))
        js_, ja_ = take(Js), take(Ja)
        sig = rho * (1.0 - xHII) / rho_mean
        nH = X_H * rho * rho_unit * (1.0 + z)**3 / M_H
        for v in range(nv):
            S[v] += L.tb(sig, nH, T, xHII, fII, fIII, R[v] * js_ + ja_)[0].sum()
        npix += rho.size
        if (i0 // a.planes) % 5 == 0:
            el = time.time() - t0
            print(f"  planes {sl.start}-{sl.stop - 1}: {el / 60:.1f} min, ~{el * N / sl.stop / 60:.1f} min total; "
                  f"<dT_b> so far fid {S[0] / npix:+.5f} mK", flush=True)
    m = S / npix
    mmap = np.nan
    if a.rin:
        with h5py.File(a.rin, "r") as f:
            dd = f["rin"] if "rin" in f else CJ_first_3d(f)
            c = str(dd.attrs.get("content", b"").decode() if isinstance(dd.attrs.get("content", b""), bytes) else dd.attrs.get("content", ""))
            if "SPIN_MODEL 1" not in c:
                sys.exit(f"STOP: {a.rin} is not a SPIN_MODEL 1 map: {c!r}")
            tot = 0.0
            for i0 in range(0, N, 32):
                tot += dd[i0:i0 + 32].astype(np.float64).sum()
            mmap = tot / float(N)**3
        print(f"  check: production map mean {mmap:+.6f} mK, fiducial here {m[0]:+.6f} mK "
              f"({'same pixels' if st == 1 else f'every {st}th pixel'}): ratio {m[0] / mmap:.6f}")
        if st == 1 and abs(m[0] / mmap - 1.0) > a.tol_map:
            sys.exit(f"STOP: the fiducial recomputed here differs from the map by more than {a.tol_map:g}")
    # every pixel: the means themselves
    glob_ = (mmap + (m - m[0])) if (st > 1 and np.isfinite(mmap)) else m
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"H640_{s:03d}.csv")
    with open(fn, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["line", "source", "snap", "z", "stride", "npix", "map_mean", "fid_here"] +
                   [f"dTb_{j:02d}" for j in range(len(Zg))] + [f"R_{j:02d}" for j in range(len(Zg))])
        w.writerow(["H", "map640", s, repr(z), st, npix, repr(float(mmap)), repr(float(m[0]))] +
                   [repr(float(x)) for x in glob_[1:]] + [repr(float(x)) for x in R[1:]])
    print(f"  <dT_b> [mK]: fiducial {glob_[0]:+.5f}; " + ", ".join(f"Z={zlabel(Zg[j])} {glob_[1 + j]:+.5f}" for j in range(len(Zg))))
    print(f"wrote {fn} ({(time.time() - t0) / 60:.1f} min)")


# ---------------------------------------------------------------------------------------------------------------------
# 3. every gas cell of the full snapshot (whole box)
# ---------------------------------------------------------------------------------------------------------------------
def scan_files(a):
    d = a.scan_dir
    fs = sorted(f for f in glob.glob(os.path.join(d, f"scan_{a.line}_{a.snap:03d}_*.hdf5")) if not f.endswith(".part"))
    if not fs:
        sys.exit(f"no scan_{a.line}_{a.snap:03d}_*.hdf5 in {d}")
    A = [dict(h5py.File(f, "r").attrs) for f in fs]
    nt = {int(x["ntask"]) for x in A if "ntask" in x}
    if nt and len(fs) != nt.pop():
        sys.exit(f"STOP: {len(fs)} scan files, the scan had ntask = {int(A[0]['ntask'])}: incomplete")
    for k in ("order", "render_snap", "n_cube", "z", "rho_mean"):
        vals = {tuple(np.atleast_1d(x[k]).tolist()) for x in A}
        if len(vals) != 1:
            sys.exit(f"STOP: scan files differ in {k}: {vals}")
    rows = sorted(tuple(int(v) for v in x["rows"]) for x in A)
    if rows[0][0] != 0 or any(rows[i][1] != rows[i + 1][0] for i in range(len(rows) - 1)):
        sys.exit(f"STOP: the scan files' row blocks do not tile the snapshot: {rows[:3]} ...")
    return fs, A[0], A


def sval(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def jnames(line, r, tag):
    if line == "He":
        return J_NAME[line].format(r, tag), AGN_NAME[line].format(r, tag)
    return J_NAME[line].format(r), AGN_NAME[line].format(r)


def jsnap_of(basename, line):
    """render number of a stellar J file name (Jalpha_H_NNN.hdf5, Jalpha_He_NNN_<tag>.hdf5), else None"""
    import re
    m = re.fullmatch(rf"Jalpha_{line}_(\d+)(_.*)?\.hdf5", basename.strip())
    return int(m.group(1)) if m else None


def stored_J_sample(fs, nmax=200_000):
    """positions and J of cells the gas-cell scan stored (its cube groups); used ONLY to identify which J files the scan
    used (a file check, not a result)"""
    for f_ in fs:
        with h5py.File(f_, "r") as f:
            for q in sorted(k for k in f if k.startswith("cube_")):
                g = f[q]
                if g["m"].shape[0] > 0:
                    n = min(nmax, g["m"].shape[0])
                    return {"pos": g["pos"][:n], "J": g["J"][:n]}
    sys.exit("STOP: no stored cells in the scan files to identify the J files")


def rel_diff(Jrep, Jstored):
    Jst = Jstored.astype(np.float32).astype(np.float64)
    return float(np.max(np.abs(Jrep.astype(np.float64) - Jst) / np.maximum(np.abs(Jst), 1e-300)))


def resolve_J(a, A, fs, S, order, Rtab):
    """the J files the gas-cell scan used (recorded names, else the J render near render_snap whose J reproduces the
    stored J exactly), loaded, with R of each side; also whether the scan interpolated J in z between two renders"""
    from test_cells_Tb import render_index
    ra = int(A["render_snap"])
    interp = int(A.get("J_interp", 0)) == 1
    sides = [[ra, 1.0, a.jsnap]]
    if interp:
        rb, wb = int(A["render_snap_b"]), float(A["J_weight_b"])
        sides = [[ra, 1.0 - wb, a.jsnap], [rb, wb, a.jsnap_b if a.jsnap_b is not None else rb]]
    for i, sd in enumerate(sides):
        if sd[2] is None and "J_files" in A:
            rec = sval(A["J_files"] if i == 0 else A["J_b_files"]).split(";")
            sd[2] = next(jsnap_of(x, a.line) for x in rec if jsnap_of(x, a.line) is not None)
    if interp and sides[0][2] is None:
        sides[0][2] = ra                               # run_cells_Tb.sh: with RSNAP_B, J at the renders themselves

    def cands(js_fixed, r):
        rs = [js_fixed] if js_fixed is not None else [r] + [r + d for k in (1, 2, 3) for d in (-k, k)]
        out = []
        for js in rs:
            nm_s, nm_a = jnames(a.line, js, a.tr_tag)
            for d in a.jdir:
                ps, pa = os.path.join(d, nm_s), os.path.join(d, nm_a)
                if os.path.exists(ps) and os.path.exists(pa):
                    out.append((js, ps, pa))
        return out
    C = [cands(sd[2], sd[0]) for sd in sides]
    for sd, c in zip(sides, C):
        if not c:
            sys.exit(f"STOP: no J files for render {sd[0]} (J render {sd[2]}) in {a.jdir}")
    import itertools
    c0 = stored_J_sample(fs)
    wb_ = sides[1][1] if interp else 0.0
    tried, chosen = [], None
    for combo in itertools.product(*C):
        arr = [(load_3d(ps), load_3d(pa)) for _, ps, pa in combo]
        ijk = render_index(c0["pos"].astype(np.float64), S.box, arr[0][0].shape[0], order)
        tot = [np.float32(0) + Js_[ijk] + Ja_[ijk] for Js_, Ja_ in arr]
        Jrep = tot[0] if not interp else ((1.0 - wb_) * tot[0] + wb_ * tot[1]).astype(np.float32)
        d_ = rel_diff(Jrep, c0["J"])
        tried.append(f"{' + '.join(os.path.relpath(ps, os.path.dirname(os.path.dirname(ps))) for _, ps, _ in combo)}: {d_:.1e}")
        if d_ <= a.tol_j:
            chosen = (combo, arr)
            break
        del arr
    if chosen is None:
        sys.exit(f"STOP: no J files reproduce the stored J (max rel diff per candidate: {tried}); check --jdir / --jsnap")
    if len(tried) > 1 or sides[0][2] is None:
        print(f"J files identified from the scan's stored J; tried: {tried}")
    J = []
    for (r, w, _), (js, ps, pa), (Js_, Ja_) in zip(sides, chosen[0], chosen[1]):
        zr = Rtab[(a.line, js)][0] if (a.line, js) in Rtab else None
        at = j_attrs(ps)
        if zr is not None and "Redshift" in at and abs(float(at["Redshift"]) - zr) > 1e-4:
            sys.exit(f"STOP: {ps} is at z = {at['Redshift']}, R table z = {zr} for render {js}")
        J.append(dict(render=js, w=w, R=R_of(Rtab, a.line, js, zr), Js=Js_, Ja=Ja_, ps=ps, pa=pa))
        print(f"J of render {r}: file render {js} (weight {w:.4f}): {ps} + {pa}\n    stellar file: SFR_input "
              f"{at.get('SFR_input', '?')!r}, IGM_transmission {at.get('IGM_transmission', '-')!r}, f_esc_star "
              f"{at.get('f_esc_star', '-')}; R = " + " ".join(f"{x:.4f}" for x in J[-1]["R"][1:]), flush=True)
    return J, interp


def solve_ts_illinois(Tg, T, A, xc, tauGP, tauTs, tol=1e-12, itmax=200):
    """the T_S root of tb_physics.solve_ts_wf (same F, same bracket, same input and 1/T_c checks), found by the
    Illinois variant of regula falsi in ln T_S instead of bisection: ~2x faster; laptop test on 5e6 random cells,
    z = 8-18: max |dT_S/T_S| 5e-14 vs bisection. Used by 'box' for the 21 cm line; the fiducial of every scan block is
    checked against the gas-cell scan (bisection) in box-merge. Since 2026-11-25 'box' uses tb_physics.solve_ts_wf_fast
    by default (--solver fast); this one stays as --solver illinois."""
    import tb_physics as TP
    Tg, T, A, xc, tauGP, tauTs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (Tg, T, A, xc, tauGP, tauTs)))
    bad = ~(np.isfinite(T) & np.isfinite(A) & np.isfinite(xc) & np.isfinite(tauGP) & np.isfinite(tauTs) & (T > 0))
    if np.any(bad):
        raise ValueError(f"T_S: {int(bad.sum())} cells with non-finite or non-positive inputs (T_K, J, x_c, tau)")
    Tg, T, A, xc, tauGP, tauTs = (np.ascontiguousarray(v).ravel() for v in (Tg, T, A, xc, tauGP, tauTs))

    def F(lTs, ix):
        Ts = np.exp(lTs)
        S_, Tci, xcmb = TP.wf_terms(Ts, Tg[ix], T[ix], A[ix], tauGP[ix], tauTs[ix])
        xa = A[ix] * S_
        return 1.0 / Ts - (xcmb / Tg[ix] + xa * Tci + xc[ix] / T[ix]) / (xcmb + xa + xc[ix])
    n = T.size
    ix = np.arange(n)
    a, b = np.log(0.25 * np.minimum(Tg, T)), np.log(4.0 * np.maximum(Tg, T))
    fa, fb = F(a, ix), F(b, ix)
    if np.any(fa <= 0) or np.any(fb >= 0):
        raise ValueError(f"T_S not bracketed in {int(np.sum(fa <= 0) + np.sum(fb >= 0))} cells")
    x = 0.5 * (a + b)
    side = np.zeros(n, np.int8)
    act = ix
    for _ in range(itmax):
        aa, bb, ffa, ffb = a[act], b[act], fa[act], fb[act]
        c = (aa * ffb - bb * ffa) / (ffb - ffa)
        c = np.where((c > aa) & (c < bb), c, 0.5 * (aa + bb))       # stay strictly inside the bracket
        fc = F(c, act)
        x[act] = c
        pos = fc > 0                                               # root in [c, b]
        sa = side[act]
        a[act] = np.where(pos, c, aa); fa[act] = np.where(pos, fc, ffa)
        b[act] = np.where(pos, bb, c); fb[act] = np.where(pos, ffb, fc)
        kb, ka = act[pos & (sa == 1)], act[~pos & (sa == -1)]
        fb[kb] *= 0.5; fa[ka] *= 0.5                               # Illinois: same end kept twice
        side[act] = np.where(pos, 1, -1)
        act = act[~((b[act] - a[act] < tol) | (fc == 0))]
        if act.size == 0:
            break
    else:
        raise ValueError(f"T_S: {act.size} cells not converged in {itmax} iterations")
    Ts = np.exp(x)
    _, Tci, _ = TP.wf_terms(np.maximum(Ts, Tg), Tg, T, A, tauGP, tauTs)
    if np.any(Tci <= 0):
        raise ValueError(f"T_S: 1/T_c <= 0 in {int(np.sum(Tci <= 0))} cells (T_K min {float(T.min()):.3g} K)")
    return Ts


def task_rows(A_all, task, split):
    """task -> (block = the gas-cell scan's task whose rows it covers, sub-block, rows r0..r1)"""
    blocks = sorted(tuple(int(v) for v in x["rows"]) for x in A_all)
    b, sub = divmod(task, split)
    if b >= len(blocks):
        sys.exit(f"ERROR: --task {task} >= {len(blocks)} blocks x --split {split}")
    e = np.linspace(blocks[b][0], blocks[b][1], split + 1).astype(np.int64)
    return b, sub, int(e[sub]), int(e[sub + 1]), blocks[b]


def cmd_box(a):
    """one task: rows r0..r1 of the full snapshot (all cells there), dT_b for the fiducial J, the 13 Z, J_AGN only and
    J = 0"""
    from test_cells_Tb import Snapshot, Phys, prep_cells, render_index
    Zg, Rtab = read_R(a.rcsv)
    fs, A, A_all = scan_files(a)
    S = Snapshot(a.snapdir, a.snap)
    sm = int(A.get("spin_model", 1))
    if a.line == "H" and sm != 1:
        sys.exit(f"STOP: the H scan used spin model {sm}")
    if a.line == "H" and a.solver == "illinois":
        import tb_physics
        tb_physics.solve_ts_wf = solve_ts_illinois             # this process only; checked in box-merge
    P = Phys(S, a.line, sm, solver="fast" if a.solver == "fast" else "bisect")
    if abs(S.z - float(A["z"])) > 1e-6 or abs(P.rho_mean / float(A["rho_mean"]) - 1) > 1e-9:
        sys.exit(f"STOP: snapshot z {S.z} / rho_mean {P.rho_mean} vs scan files {A['z']} / {A['rho_mean']}")
    if int(sum(int(x.get("n_cells", 0)) for x in A_all)) not in (0, S.n):
        sys.exit(f"STOP: the scan files hold {sum(int(x.get('n_cells', 0)) for x in A_all)} cells, the snapshot {S.n}")
    if "J_files" in A and sval(A["J_files"]).strip() == "none":
        sys.exit("STOP: this scan used J = 0 (--J none): no metallicity dependence, nothing to do")
    order = tuple(int(x) for x in np.atleast_1d(A["order"]))
    b, sub, r0, r1, blk = task_rows(A_all, a.task, a.split)
    if a.max_rows:
        r1 = min(r1, r0 + a.max_rows)
    J, interp = resolve_J(a, A, fs, S, order, Rtab)
    N = J[0]["Js"].shape[0]
    nZ = len(Zg)
    nv = 1 + nZ + len(EXTRA)
    names = ["fid"] + [f"Z_{j:02d}" for j in range(nZ)] + list(EXTRA)
    print(f"{a.line} snapshot {a.snap}: z = {S.z:.5f}, spin model {sm}, T_S solver {a.solver if a.line == 'H' else '-'}, render order {order}, "
          f"J {'linear in z between 2 renders' if interp else 'at one render'}; task {a.task} = scan block {b} "
          f"(rows {blk[0]:,}-{blk[1]:,}), part {sub}/{a.split}: rows {r0:,}-{r1:,} ({r1 - r0:,} cells)"
          f"{'  [TIMING RUN: --max-rows, no output written]' if a.max_rows else ''}", flush=True)
    SV = 0.0
    SVtb = np.zeros(nv)
    ncell = 0
    t0 = time.time()
    for i0 in range(r0, r1, a.rows):
        i1 = min(r1, i0 + a.rows)
        c = prep_cells(S, i0, i1)
        i, j, k = render_index(c["pos"], S.box, N, order)
        del c["pos"]
        # the scan's J (test_cells_Tb load_J / J_at float32 operations), then the stellar part per side
        tot = [np.float32(0) + s_["Js"][i, j, k] + s_["Ja"][i, j, k] for s_ in J]
        Jc = (tot[0] if not interp else ((1.0 - J[1]["w"]) * tot[0] + J[1]["w"] * tot[1]).astype(np.float32)).astype(np.float64)
        Jstar = [s_["w"] * s_["Js"][i, j, k].astype(np.float64) for s_ in J]
        Jagn = [s_["w"] * s_["Ja"][i, j, k].astype(np.float64) for s_ in J]
        del i, j, k, tot
        V = c["V"]
        SV += V.sum()
        for v in range(nv):
            Jv = variant_J(v, nZ, Jc, Jstar, Jagn, [s_["R"] for s_ in J])
            for c0 in range(0, len(V), a.chunk):
                sl = slice(c0, c0 + a.chunk)
                tb = P.tb(c["rho"][sl], c["T"][sl], c["xHII"][sl], c["fII"][sl], c["fIII"][sl], Jv[sl])
                if not np.all(np.isfinite(tb)):
                    sys.exit(f"STOP: non-finite dT_b in rows {i0}-{i1}, variant {v}")
                SVtb[v] += np.sum(V[sl] * tb)
        ncell += i1 - i0
        el = time.time() - t0
        print(f"  rows {i0 - r0 + (i1 - i0):,}/{r1 - r0:,}: {el / 60:.1f} min; {el / ncell * 1e6:.2f} us per cell "
              f"({nv} variants); <dT_b> fid {SVtb[0] / SV:+.6g}, Z={zlabel(Zg[0])} {SVtb[1] / SV:+.6g}, "
              f"Z={zlabel(Zg[-1])} {SVtb[nZ] / SV:+.6g}, no_star {SVtb[nZ + 1] / SV:+.6g}, no_J {SVtb[nZ + 2] / SV:+.6g}",
              flush=True)
    rate = (time.time() - t0) / max(ncell, 1)
    if a.max_rows:
        nall = S.n
        print(f"TIMING: {rate * 1e6:.2f} us per cell for {nv} variants on this node -> {rate * (blk[1] - blk[0]) / a.split / 3600:.1f} h "
              f"per task (--split {a.split}), {rate * nall / 3600:.0f} task-hours for the whole snapshot ({nall:,} cells)")
        return
    od = os.path.join(a.out, f"box_{a.line}")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"box_{a.snap:03d}_{a.task:04d}.npz")
    np.savez(fn + ".part.npz", SV=SV, SVtb=SVtb, ncell=ncell, rows=np.array([r0, r1]), block=b, sub=sub, split=a.split,
             task=a.task, z=S.z, snap=a.snap, line=a.line, solver=a.solver if a.line == "H" else "-", Z_grid=Zg, R=np.array([s_["R"] for s_ in J]), variants=np.array(names),
             renders=np.array([s_["render"] for s_ in J]), weights=np.array([s_["w"] for s_ in J]),
             J_files=";".join(f"{s_['ps']};{s_['pa']}" for s_ in J), scan_dir=os.path.abspath(a.scan_dir))
    os.replace(fn + ".part.npz", fn)
    print(f"wrote {fn} ({(time.time() - t0) / 3600:.2f} h, {rate * 1e6:.2f} us per cell)")


def cmd_box_merge(a):
    od = os.path.join(a.out, f"box_{a.line}")
    fs = sorted(f for f in glob.glob(os.path.join(od, f"box_{a.snap:03d}_*.npz")) if not f.endswith(".part.npz"))
    if not fs:
        sys.exit(f"no task files for {a.line} {a.snap} in {od}")
    D = [np.load(f) for f in fs]
    split = {int(d["split"]) for d in D}
    if len(split) != 1:
        sys.exit(f"STOP: task files with different --split {sorted(split)}")
    split = split.pop()
    for k in ("R", "renders", "weights", "Z_grid"):
        if any(not np.array_equal(d[k], D[0][k]) for d in D):
            sys.exit(f"STOP: the task files differ in {k}")
    has_v = {"variants" in d.files for d in D}
    if len(has_v) != 1:
        sys.exit("STOP: task files with and without the extra variants (no_star, no_J): rerun the old tasks")
    nZ = len(D[0]["Z_grid"])
    if has_v.pop():
        names = [str(x) for x in D[0]["variants"]]
        if any([str(x) for x in d["variants"]] != names for d in D) or names != ["fid"] + [f"Z_{j:02d}" for j in range(nZ)] + list(EXTRA):
            sys.exit(f"STOP: unexpected variant lists in the task files: {names}")
    else:
        names = ["fid"] + [f"Z_{j:02d}" for j in range(nZ)]          # files from before the extra variants
    if any(len(d["SVtb"]) != len(names) for d in D):
        sys.exit("STOP: SVtb length does not match the variant list")
    _, _, A_all = scan_files(a)
    blocks = sorted(tuple(int(v) for v in x["rows"]) for x in A_all)
    got = sorted(int(d["task"]) for d in D)
    if got != list(range(len(blocks) * split)):
        miss = sorted(set(range(len(blocks) * split)) - set(got))
        sys.exit(f"STOP: {len(D)} task files, need {len(blocks) * split}; missing tasks {miss[:20]}")
    rows = sorted(tuple(int(v) for v in d["rows"]) for d in D)
    S_ = int(sum(r[1] - r[0] for r in rows))
    if rows[0][0] != 0 or any(rows[i][1] != rows[i + 1][0] for i in range(len(rows) - 1)) or S_ != blocks[-1][1]:
        sys.exit("STOP: the task rows do not tile the snapshot exactly once")
    # check: every block's fiducial sums equal the gas-cell scan's (same cells, same J, same physics)
    by_block = defaultdict(lambda: [0.0, 0.0])
    for d in D:
        by_block[int(d["block"])][0] += float(d["SV"]); by_block[int(d["block"])][1] += float(d["SVtb"][0])
    worst_v = worst_t = 0.0
    for x in A_all:
        b = blocks.index(tuple(int(v) for v in x["rows"]))
        sv, svt = by_block[b]
        worst_v = max(worst_v, abs(sv / float(x["sum_V"]) - 1.0))
        worst_t = max(worst_t, abs(svt - float(x["sum_V_tb"])) / max(abs(float(x["sum_V_tb"])), 1e-300))
    SV = sum(float(d["SV"]) for d in D)
    SVtb = sum(d["SVtb"] for d in D)
    m = SVtb / SV
    box_scan = sum(float(x["sum_V_tb"]) for x in A_all) / sum(float(x["sum_V"]) for x in A_all)
    Zg = D[0]["Z_grid"]
    z = float(D[0]["z"])
    unit = "mK" if a.line == "H" else "uK"
    print(f"{a.line} snapshot {a.snap}, z = {z:.4f}: {len(D)} task files, {int(sum(int(d['ncell']) for d in D)):,} cells "
          f"(every cell of the snapshot)")
    print(f"  check per scan block (fiducial vs the gas-cell scan): max |sum V ratio - 1| {worst_v:.1e}, "
          f"max |sum V dT_b ratio - 1| {worst_t:.1e}; whole box fiducial {m[0]:+.7g} {unit} (scan {box_scan:+.7g})")
    if worst_v > 1e-9 or worst_t > a.tol_sum:
        sys.exit(f"STOP: the fiducial does not reproduce the gas-cell scan (tolerance {a.tol_sum:g})")
    for j in range(len(Zg)):
        print(f"    Z = {zlabel(Zg[j]):>6s}: {m[1 + j]:+.6g} {unit}  (change {m[1 + j] - m[0]:+.4g})")
    ex = {nm: (float(m[names.index(nm)]) if nm in names else float("nan")) for nm in EXTRA}
    if all(np.isfinite(v) for v in ex.values()):
        print(f"    J_AGN only (no stars): {ex['no_star']:+.6g} {unit}; J = 0 (collisions only): {ex['no_J']:+.6g} {unit}")
        print(f"    radiative share 1 - <dT_b>_noJ/<dT_b>_fid = {1.0 - ex['no_J'] / m[0]:.4f}; stellar part "
              f"(<dT_b>_fid - <dT_b>_nostar)/<dT_b>_fid = {(m[0] - ex['no_star']) / m[0]:.4f}")
    fn = os.path.join(od, f"box_{a.line}_{a.snap:03d}.csv")
    with open(fn, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["line", "source", "snap", "z", "box_fid", "check_sum_V_tb"] + [f"dTb_{j:02d}" for j in range(len(Zg))]
                   + [f"dTb_{nm}" for nm in EXTRA])
        w.writerow([a.line, "cells", a.snap, repr(z), repr(float(m[0])), repr(worst_t)] + [repr(float(x)) for x in m[1:1 + nZ]]
                   + [repr(ex[nm]) for nm in EXTRA])
    print(f"wrote {fn}")


# ---------------------------------------------------------------------------------------------------------------------
# 4. table
# ---------------------------------------------------------------------------------------------------------------------
def cmd_table(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(a.out, "H640_*.csv")) + glob.glob(os.path.join(a.out, "box_*", "box_*_[0-9]*.csv"))):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    if not rows:
        sys.exit(f"nothing in {a.out}")
    nZ = sum(1 for k in rows[0] if k.startswith("dTb_"))
    out = []
    for r in rows:
        fid = float(r["map_mean"]) if r["source"] == "map640" else float(r["box_fid"])
        if r["source"] == "map640" and not np.isfinite(fid):
            fid = float(r["fid_here"])
        out.append(dict(line=r["line"], source=r["source"], snap=int(r["snap"]), z=float(r["z"]), fiducial=fid,
                        **{f"dTb_{j:02d}": float(r[f"dTb_{j:02d}"]) for j in range(nZ)},
                        **{f"dTb_{nm}": float(r.get(f"dTb_{nm}") or "nan") for nm in EXTRA}))
    out.sort(key=lambda r: (r["line"], -r["z"]))
    fn = os.path.join(a.out, "metallicity_global.csv")
    with open(fn, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {fn}: {len(out)} rows")
    for r in out:
        print(f"  {r['line']:2s} {r['source']:6s} {r['snap']:4d} z = {r['z']:7.3f}: fid {r['fiducial']:+.4g}, "
              f"Z=1e-5 {r['dTb_00']:+.4g}, Z=0.04 {r[f'dTb_{nZ - 1:02d}']:+.4g}")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        cmap = plt.get_cmap("coolwarm")
        for i, line in enumerate(LINES):
            rr = [r for r in out if r["line"] == line]
            if not rr:
                continue
            z = np.array([r["z"] for r in rr])
            for j in range(nZ):
                ax[i].plot(z, [r[f"dTb_{j:02d}"] for r in rr], "-", color=cmap(j / max(nZ - 1, 1)), lw=1)
            ax[i].plot(z, [r["fiducial"] for r in rr], "k-", lw=2, label="fiducial (young-star Z mix)")
            for nm, ls, lab in (("no_star", "--", "AGN only (no stars)"), ("no_J", ":", r"$J_\alpha = 0$")):
                y = np.array([r[f"dTb_{nm}"] for r in rr])
                if np.any(np.isfinite(y)):
                    ax[i].plot(z, y, "k" + ls, lw=1.5, label=lab)
            ax[i].set_xlabel("z"); ax[i].set_ylabel(r"$\delta T_b$ [" + ("mK" if line == "H" else r"$\mu$K") + "]")
            ax[i].set_title("21 cm" if line == "H" else "3.46 cm"); ax[i].legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(a.out, "metallicity_global.png")
        fig.savefig(png, dpi=150)
        print(f"wrote {png}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("R", help="R_j(z) = <J_star,j>/<J_star,fid> per line and render snapshot")
    r.add_argument("--sfr-summary", required=True, help="production SFR_eff_summary.csv (make_SFR_eff.py)")
    r.add_argument("--weights", required=True, help="Z_weights.txt of the production SFR_eff (run_SFR_eff.sh) and the H J")
    r.add_argument("--weights-he", help="Z_weights.txt the He J code was compiled with (default: --weights)")
    r.add_argument("--bpass", help="BPASS table (default: make_Z_snapshots.BPASS_FILE)")
    r.add_argument("--jdir", nargs="+", default=[], help="J directories, searched in order (file z, J_mean, transmission)")
    r.add_argument("--unit-from", help="SFR_eff H directory (Header of snapshot 0) for the absolute check vs the J files")
    r.add_argument("--lines", nargs="+", default=list(LINES), choices=LINES)
    r.add_argument("--snaps", type=int, nargs="+", help="render snapshots (default: every stellar J file in --jdir)")
    r.add_argument("--tr-tag", default="tr_sfmask")
    r.add_argument("--trans-dir", help="He: transmission_He_NNN.hdf5 directory (default: the J file's IGM_transmission)")
    r.add_argument("--he-no-trans", action="store_true", help="He without transmission where no file is found")
    r.add_argument("--order", type=int, default=6)
    r.add_argument("--tol-w", type=float, default=1e-4)
    r.add_argument("--out", required=True)
    m = sub.add_parser("map640", help="21 cm global on the 640^3 pixels, 14 variants")
    m.add_argument("--snap", type=int, required=True)
    m.add_argument("--dir", required=True, help="ren_640 directory of the snapshot")
    m.add_argument("--rcsv", required=True)
    m.add_argument("--jdir", nargs="+", default=[])
    m.add_argument("--J-star"); m.add_argument("--J-agn")
    m.add_argument("--rin", help="production RIN_N.hdf5 (SPIN_MODEL 1) to check the fiducial mean")
    m.add_argument("--planes", type=int, default=8)
    m.add_argument("--tol-map", type=float, default=1e-4)
    m.add_argument("--out", required=True)
    for nm in ("box", "box-merge"):
        c = sub.add_parser(nm, help="every gas cell of a full snapshot, 16 variants (fid, 13 Z, no_star, no_J)" if nm == "box" else
                           "sum the box task files, check against the gas-cell scan, write the global values")
        c.add_argument("--line", choices=LINES, required=True)
        c.add_argument("--snap", type=int, required=True)
        c.add_argument("--scan-dir", required=True, help="the gas-cell scan of this snapshot (row blocks, J files, checks)")
        c.add_argument("--out", required=True)
        if nm == "box":
            c.add_argument("--snapdir", required=True)
            c.add_argument("--rcsv", required=True)
            c.add_argument("--jdir", nargs="+", required=True)
            c.add_argument("--tr-tag", default="tr_sfmask")
            c.add_argument("--task", type=int, required=True, help="0 .. (scan blocks x --split) - 1")
            c.add_argument("--split", type=int, default=1, help="tasks per scan row block")
            c.add_argument("--rows", type=int, default=2_000_000, help="rows read at a time")
            c.add_argument("--chunk", type=int, default=1_000_000, help="cells per dT_b call")
            c.add_argument("--max-rows", type=int, help="timing run: only this many rows, nothing written")
            c.add_argument("--tol-j", type=float, default=1e-6)
            c.add_argument("--jsnap", type=int)
            c.add_argument("--jsnap-b", type=int)
            c.add_argument("--solver", choices=["fast", "illinois", "bisect"], default="fast",
                           help="21 cm T_S solver: tb_physics.solve_ts_wf_fast (quadratic in 1/T_S, default, 2026-11-25), "
                                "solve_ts_illinois, or tb_physics.solve_ts_wf (bisection)")
        else:
            c.add_argument("--tol-sum", type=float, default=1e-6)
    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    t.add_argument("--png", action="store_true")
    a = p.parse_args()
    {"R": cmd_R, "map640": cmd_map640, "box": cmd_box, "box-merge": cmd_box_merge, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
