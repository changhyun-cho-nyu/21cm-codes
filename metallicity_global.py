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
2. map640 --snap N : 21 cm, 640^3 render pixels (low z): dT_b for J = R_j J_star + J_AGN, every pixel (or every
         --stride-th pixel per axis), all 14 variants (fiducial + 13 Z) in one read of the fields; the fiducial mean is
         checked against the production RIN map (--rin).
3. cubes --line L --snap N : gas cells (H at z >= 8, He at z <= 10): the 64 random cubes stored by the gas-cell scans
         (test_cells_Tb.py, cell_test[_H]_NN/scan_L_NNN_*.hdf5), J of each cell = stored J + (R_j - 1) J_star at the
         cell's pixel (linear in z between two renders where the scan interpolated). The stored J is reproduced from
         the given J files bit for bit before anything else (checks files, render axis order and weights); the
         fiducial dT_b is recomputed and compared with the stored one.
   merge --line L --snap N : cube sums -> global_j = whole-box gas-cell fiducial (sum_V_tb / sum_V of the scan
         files, exact) + (<dT_b>_cubes,j - <dT_b>_cubes,fid), jackknife over the cubes (the cube sample estimates the
         change without bias; the ratio form box x cubes_j/cubes_fid is written too, for comparison).
         map640 --stride > 1 likewise: production map mean + the change on the pixel subset.
4. table : every merged result and map640 row -> one CSV (+ PNG with --png).

Usage (ORCD; job scripts run_metal_*.sh)
-----
  python metallicity_global.py R --sfr-summary .../SFR_eff/SFR_eff_summary.csv --weights .../Z_weights.txt \\
         --jdir .../Jalpha [.../Jalpha_fullsnap ...] --out .../metallicity/R_Z.csv [--snaps ...]
  python metallicity_global.py map640 --snap 300 --dir <ren_640> --rcsv R_Z.csv --jdir .../Jalpha \\
         --rin .../Tb_maps_prod/H_real_wf/RIN_300.hdf5 --out .../metallicity
  python metallicity_global.py cubes --line He --snap 116 --snapdir <snapshots> --scan-dir .../cell_test_116 \\
         --rcsv R_Z.csv --jdir .../Jalpha --task 0 --ntask 8 --out .../metallicity
  python metallicity_global.py merge --line He --snap 116 --scan-dir .../cell_test_116 --out .../metallicity
  python metallicity_global.py table --out .../metallicity [--png]
"""
import argparse
import csv
import glob
import os
import sys
import time

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
    st = a.stride
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
    # every pixel: the means themselves; a pixel subset (--stride > 1): production map mean + the subset's change
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
# 3. gas cells of the scan cubes
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


def read_cube(fs, q, keys, nmax=None):
    parts, n = [], 0
    for f_ in fs:
        with h5py.File(f_, "r") as f:
            g = f.get(f"cube_{q:03d}")
            if g is not None and g["m"].shape[0] > 0:
                parts.append({k: g[k][()] for k in keys})
                n += len(parts[-1][keys[0]])
        if nmax and n >= nmax:
            break
    if not parts:
        sys.exit(f"STOP: cube {q} has no cells")
    c = {k: np.concatenate([p_[k] for p_ in parts]) for k in keys}
    return {k: v[:nmax] for k, v in c.items()} if nmax else c


def rel_diff(Jrep, Jstored):
    Jst = Jstored.astype(np.float32).astype(np.float64)
    return float(np.max(np.abs(Jrep.astype(np.float64) - Jst) / np.maximum(np.abs(Jst), 1e-300)))


def cmd_cubes(a):
    from test_cells_Tb import Snapshot, Phys, render_index
    Zg, Rtab = read_R(a.rcsv)
    fs, A, _ = scan_files(a)
    S = Snapshot(a.snapdir, a.snap)
    sm = int(A.get("spin_model", 1))
    if a.line == "H" and sm != 1:
        sys.exit(f"STOP: the H scan used spin model {sm}")
    P = Phys(S, a.line, sm)
    if abs(S.z - float(A["z"])) > 1e-6 or abs(P.rho_mean / float(A["rho_mean"]) - 1) > 1e-9:
        sys.exit(f"STOP: snapshot z {S.z} / rho_mean {P.rho_mean} vs scan files {A['z']} / {A['rho_mean']}")
    if "J_files" in A and sval(A["J_files"]).strip() == "none":
        sys.exit("STOP: this scan used J = 0 (--J none): no metallicity dependence, nothing to do")
    order = tuple(int(x) for x in np.atleast_1d(A["order"]))
    ra = int(A["render_snap"])
    interp = int(A.get("J_interp", 0)) == 1
    sides = [[ra, 1.0, a.jsnap]]
    if interp:
        rb, wb = int(A["render_snap_b"]), float(A["J_weight_b"])
        sides = [[ra, 1.0 - wb, a.jsnap], [rb, wb, a.jsnap_b if a.jsnap_b is not None else rb]]   # J_b at render b
    for i, sd in enumerate(sides):                     # the J render of each side
        if sd[2] is None and "J_files" in A:           # scans from 2026-11-19 on record the J file names
            rec = sval(A["J_files"] if i == 0 else A["J_b_files"]).split(";")
            sd[2] = next(jsnap_of(x, a.line) for x in rec if jsnap_of(x, a.line) is not None)
    if interp and sides[0][2] is None:
        sides[0][2] = ra                               # run_cells_Tb.sh: with RSNAP_B, J at the renders themselves

    def cands(js_fixed, r):
        """(J render, stellar file, AGN file) for every allowed J render and every --jdir, in order; an unknown J
        render (older scans) is searched among r, r -+ 1, r -+ 2, r -+ 3"""
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
    # which files: the pair (or pairs) that reproduce the stored J of the first cube's cells exactly
    import itertools
    c0 = read_cube(fs, 0, ("pos", "J"), nmax=200_000)
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
        print(f"J files found from the stored J; tried: {tried}")
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
    N = J[0]["Js"].shape[0]
    ncube = int(A["n_cube"])
    mine = [q for q in range(ncube) if q % a.ntask == a.task]
    nv = 1 + len(Zg)
    print(f"{a.line} snapshot {a.snap}: z = {S.z:.5f}, spin model {sm}, {len(fs)} scan files, render order {order}, "
          f"J {'linear in z between 2 renders' if interp else 'at one render'}; task {a.task}/{a.ntask}: cubes {mine}",
          flush=True)
    cube = np.zeros((ncube, 2 + nv))                       # n, V, V tb_v
    chk = np.zeros(4)                                      # J: max rel diff; tb: n, max |d|, sum |tb|
    t0 = time.time()
    for q in mine:
        c = read_cube(fs, q, ("pos", "rho", "V", "T", "xHII", "fII", "fIII", "J", "tb"))
        i, j, k = render_index(c["pos"].astype(np.float64), S.box, N, order)
        # the scan's J, reproduced with its float32 operations (test_cells_Tb.load_J and J_at)
        tot = [np.float32(0) + s_["Js"][i, j, k] + s_["Ja"][i, j, k] for s_ in J]
        Jrep = tot[0] if not interp else ((1.0 - J[1]["w"]) * tot[0] + J[1]["w"] * tot[1]).astype(np.float32)
        d_ = rel_diff(Jrep, c["J"])
        chk[0] = max(chk[0], d_)
        if d_ > a.tol_j:
            sys.exit(f"STOP: cube {q}: the stored J is not reproduced from these J files (max rel diff {d_:.2e}): "
                     f"wrong files, axis order or weights")
        Jc = c["J"].astype(np.float64)
        Jstar = [s_["w"] * s_["Js"][i, j, k].astype(np.float64) for s_ in J]
        rho, T, xHII = c["rho"].astype(np.float64), c["T"].astype(np.float64), c["xHII"].astype(np.float64)
        fII, fIII, V = c["fII"].astype(np.float64), c["fIII"].astype(np.float64), c["V"].astype(np.float64)
        cube[q, 0] += len(V); cube[q, 1] += V.sum()
        for v in range(nv):
            Jv = Jc + sum((s_["R"][v] - 1.0) * js for s_, js in zip(J, Jstar))
            tb = np.empty(len(V))
            for c0 in range(0, len(V), a.chunk):
                sl = slice(c0, c0 + a.chunk)
                tb[sl] = P.tb(rho[sl], T[sl], xHII[sl], fII[sl], fIII[sl], Jv[sl])
            if not np.all(np.isfinite(tb)):
                sys.exit(f"STOP: non-finite dT_b in cube {q}, variant {v}")
            if v == 0:
                d_ = np.abs(tb - c["tb"].astype(np.float64))
                chk[1] += len(tb); chk[2] = max(chk[2], float(d_.max())); chk[3] += float(np.abs(tb).sum())
            cube[q, 2 + v] += np.sum(V * tb)
        print(f"  cube {q:2d}: {len(V):,} cells, <dT_b> fid {cube[q, 2] / cube[q, 1]:+.5g}, Z={zlabel(Zg[0])} "
              f"{cube[q, 3] / cube[q, 1]:+.5g}, Z={zlabel(Zg[-1])} {cube[q, 2 + nv - 1] / cube[q, 1]:+.5g}; J max rel diff "
              f"{chk[0]:.1e}, fid tb max |recomputed - stored| {chk[2]:.2e} ({time.time() - t0:.0f} s)", flush=True)
        del c
    od = os.path.join(a.out, f"cells_{a.line}")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"cubes_{a.snap:03d}_{a.task:03d}.npz")
    np.savez(fn + ".part.npz", cube=cube, check=chk, mine=np.array(mine), z=S.z, snap=a.snap, line=a.line, task=a.task,
             ntask=a.ntask, n_cube=ncube, Z_grid=Zg, R=np.array([s_["R"] for s_ in J]),
             renders=np.array([s_["render"] for s_ in J]), weights=np.array([s_["w"] for s_ in J]),
             J_files=";".join(f"{s_['ps']};{s_['pa']}" for s_ in J), scan_dir=os.path.abspath(a.scan_dir))
    os.replace(fn + ".part.npz", fn)
    print(f"wrote {fn} ({(time.time() - t0) / 60:.1f} min)")


def cmd_merge(a):
    od = os.path.join(a.out, f"cells_{a.line}")
    fs = sorted(f for f in glob.glob(os.path.join(od, f"cubes_{a.snap:03d}_*.npz")) if not f.endswith(".part.npz"))
    if not fs:
        sys.exit(f"no task files for {a.line} {a.snap} in {od}")
    D = [np.load(f) for f in fs]
    nt = {int(d["ntask"]) for d in D}
    if len(nt) != 1 or len(D) != list(nt)[0] or sorted(int(d["task"]) for d in D) != list(range(len(D))):
        sys.exit(f"STOP: {len(D)} task files with ntask {sorted(nt)}: incomplete or mixed")
    for k in ("R", "renders", "weights", "Z_grid"):
        if any(not np.array_equal(d[k], D[0][k]) for d in D):
            sys.exit(f"STOP: the task files differ in {k}")
    cube = sum(d["cube"] for d in D)
    if np.any(cube[:, 0] == 0):
        sys.exit(f"STOP: cubes without cells: {np.flatnonzero(cube[:, 0] == 0)}")
    Jmax = max(float(d["check"][0]) for d in D)
    ntb = sum(float(d["check"][1]) for d in D)
    dmax = max(float(d["check"][2]) for d in D)
    stb = sum(float(d["check"][3]) for d in D)
    Zg = D[0]["Z_grid"]
    z = float(D[0]["z"])
    V, Vtb = cube[:, 1], cube[:, 2:]
    n = len(V)
    m = Vtb.sum(axis=0) / V.sum()
    jk = np.array([(Vtb.sum(axis=0) - Vtb[i]) / (V.sum() - V[i]) for i in range(n)])     # (n, nv) jackknife means
    err = lambda x: np.sqrt((n - 1) / n * np.sum((x - x.mean(axis=0))**2, axis=0))
    diff, ediff = m - m[0], err(jk - jk[:, :1])                    # change of the cube mean, Z_j - fiducial
    ratio = m / m[0] if m[0] != 0 else np.full_like(m, np.nan)
    eratio = err(jk / jk[:, :1]) if np.all(jk[:, 0] != 0) else np.full_like(m, np.nan)
    fs_scan = sorted(f for f in glob.glob(os.path.join(a.scan_dir, f"scan_{a.line}_{a.snap:03d}_*.hdf5"))
                     if not f.endswith(".part"))
    sv = svt = 0.0
    for f_ in fs_scan:
        with h5py.File(f_, "r") as f:
            sv += float(f.attrs["sum_V"]); svt += float(f.attrs["sum_V_tb"])
    box = svt / sv
    glob_ = box + diff                         # whole-box fiducial + the cubes' change (unbiased for the change)
    unit = "mK" if a.line == "H" else "uK"
    print(f"{a.line} snapshot {a.snap}, z = {z:.4f}: {len(D)} task files, {int(cube[:, 0].sum()):,} cells in {n} cubes")
    print(f"  checks: stored J reproduced to max rel {Jmax:.1e}; fiducial dT_b recomputed vs stored on {int(ntb):,} cells: "
          f"max |diff| {dmax:.2e} {unit} (mean |dT_b| {stb / max(ntb, 1):.3e})")
    print(f"  whole box (exact, {len(fs_scan)} scan files): fiducial {box:+.6g} {unit}; cubes fiducial {m[0]:+.6g}")
    print("  global(Z_j) = whole box + (cubes(Z_j) - cubes(fid)); the ratio estimate box x cubes(Z_j)/cubes(fid) for comparison")
    for j in range(len(Zg)):
        print(f"    Z = {zlabel(Zg[j]):>6s}: change {diff[1 + j]:+.5g} +- {ediff[1 + j]:.2g} {unit} -> {glob_[1 + j]:+.6g} {unit}"
              f"   (ratio {ratio[1 + j]:.5f} +- {eratio[1 + j]:.5f} -> {box * ratio[1 + j]:+.6g})")
    fn = os.path.join(od, f"cells_{a.line}_{a.snap:03d}.csv")
    nZ = len(Zg)
    with open(fn, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["line", "source", "snap", "z", "box_fid", "cubes_fid"] + [f"dTb_{j:02d}" for j in range(nZ)] +
                   [f"err_{j:02d}" for j in range(nZ)] + [f"ratio_{j:02d}" for j in range(nZ)] +
                   [f"err_ratio_{j:02d}" for j in range(nZ)])
        w.writerow([a.line, "cells", a.snap, repr(z), repr(box), repr(float(m[0]))] +
                   [repr(float(x)) for x in glob_[1:]] + [repr(float(x)) for x in ediff[1:]] +
                   [repr(float(x)) for x in ratio[1:]] + [repr(float(x)) for x in eratio[1:]])
    print(f"wrote {fn}")


# ---------------------------------------------------------------------------------------------------------------------
# 4. table
# ---------------------------------------------------------------------------------------------------------------------
def cmd_table(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(a.out, "H640_*.csv")) + glob.glob(os.path.join(a.out, "cells_*", "cells_*.csv"))):
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
                        **{f"dTb_{j:02d}": float(r[f"dTb_{j:02d}"]) for j in range(nZ)}))
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
    m.add_argument("--stride", type=int, default=1)
    m.add_argument("--planes", type=int, default=8)
    m.add_argument("--tol-map", type=float, default=1e-4)
    m.add_argument("--out", required=True)
    for nm in ("cubes", "merge"):
        c = sub.add_parser(nm)
        c.add_argument("--line", choices=LINES, required=True)
        c.add_argument("--snap", type=int, required=True)
        c.add_argument("--scan-dir", required=True)
        c.add_argument("--out", required=True)
        if nm == "cubes":
            c.add_argument("--snapdir", required=True)
            c.add_argument("--rcsv", required=True)
            c.add_argument("--jdir", nargs="+", required=True)
            c.add_argument("--tr-tag", default="tr_sfmask")
            c.add_argument("--task", type=int, default=0)
            c.add_argument("--ntask", type=int, default=1)
            c.add_argument("--chunk", type=int, default=4_000_000)
            c.add_argument("--tol-j", type=float, default=1e-6)
            c.add_argument("--jsnap", type=int, help="render of the J files (default: recorded in the scan files, "
                           "else detected: the render near render_snap whose J reproduces the stored J)")
            c.add_argument("--jsnap-b", type=int, help="second J render (interpolated scans; default render_snap_b)")
    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    t.add_argument("--png", action="store_true")
    a = p.parse_args()
    {"R": cmd_R, "map640": cmd_map640, "cubes": cmd_cubes, "merge": cmd_merge, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
