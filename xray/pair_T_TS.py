"""
pair_T_TS.py
============
Whole-box gas and spin temperatures of the 50 cMpc pair (Lumina-50 = withXrays, Lumina-50-NoX = withoutXrays) with the
same statistics as the Lumina no-X scan (lumina_noX.py, Sums.ts; user, 2026-11-24: figure of T_gas, T_S and dT_b for
Lumina, Lumina without X-rays, the pair and T_CMB). Every gas cell of one run at one full snapshot:
    dT_b (tb_physics.Line spin model 1, tau_GP from the cell's n_HI, J_alpha of the run at the render; as
          closure_noX_xray.py and the "cell" option of cells_Tb_xray.py: 1+delta relative to the run's gas mean,
          star-forming cells neutral),
    H I weight w = X_H m (1 - x_HII) (SF cells x = 0), in two selections (lumina_noX.TS_SEL):
          neutral non-SF (x_HII < 1e-3) and all H I (every cell, the selection of cells_Tb_xray.py's HI_spin),
    sums W, W/T, W (1 - T_g/T_S); box <dT_b> = sum V dT_b / sum V.
Derived: <1/T>^-1 = W / sum(W/T), T_S,eff = T_g / (1 - <1 - T_g/T_S>) = <1/T_S>^-1 (H I-weighted harmonic means).
Checks: the box <dT_b> against cells_Tb_xray.py (mean_cells_cell) and the all-H I <1 - T_g/T_S> against its HI_spin_cell
where that file exists (bisection solver there; --solver fast here).

  python pair_T_TS.py --run withXrays --snap 57          (render from SNAP_RENDER; J from <out>/<run>/Jalpha)
Writes <out>/<run>/cells_TTS/cells_TTS_NNN.npz (a few kB). Inputs read only; outputs only under /orcd/data/mvogelsb/004/chcho.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from pair_cells_xray import SRC, OUT, RUNS, SNAP_RENDER, X_H, XNEU, read_render, read_snap, axis_order, check_out, snap_files  # noqa: E402
from cells_Tb_xray import hdr_value, load_3d, Line, M_H  # noqa: E402

TS_SEL = ("neutral non-SF", "all H I")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", choices=RUNS, required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--render", type=int, default=None)
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", default=os.environ.get("BASE", OUT))
    p.add_argument("--chunk", type=int, default=4_000_000)
    p.add_argument("--solver", choices=["fast", "bisect"], default="fast")
    p.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    render = a.render if a.render is not None else SNAP_RENDER.get(a.snap)
    if render is None:
        sys.exit(f"no render known for snapshot {a.snap}: pass --render")
    od = os.path.join(a.out, a.run, "cells_TTS")
    check_out(od, a.allow_any_out)
    t0 = time.time()

    fs = snap_files(a.src, a.run, a.snap)
    with h5py.File(fs[0], "r") as f:
        hd = dict(f["Header"].attrs)
        par = dict(f["Parameters"].attrs) if "Parameters" in f else {}
    z = float(hd["Redshift"])
    h = hdr_value(hd, par, "HubbleParam")
    om, ob = hdr_value(hd, par, "Omega0"), hdr_value(hd, par, "OmegaBaryon")
    ol = hdr_value(hd, par, "OmegaLambda", 0.6902672)
    UL = hdr_value(hd, par, "UnitLength_in_cm", 3.085678e21)
    UM = hdr_value(hd, par, "UnitMass_in_g", 1.989e43)
    nfac = X_H * UM * h**2 / UL**3 * (1.0 + z)**3 / M_H
    L = Line("H", z, h, om, ob, ol, spin_model=1, solver=a.solver)
    Tg = L.Tg
    print(f"{a.run}, snapshot {a.snap}, render {render}: z = {z:.5f}, T_gamma = {Tg:.3f} K, solver {a.solver}", flush=True)

    zr, N, R = read_render(a.src, a.run, render, ["Density"])
    if abs(zr - z) > 1e-3:
        sys.exit(f"STOP: render z {zr} vs snapshot z {z}")
    C, M, dep_m, _ = read_snap(a.src, a.run, a.snap, N)
    o, co = axis_order(dep_m.reshape(N, N, N), R["Density"])
    del dep_m, R
    idx = ((C["ip"].astype(np.uint64) * np.uint64(N)) >> np.uint64(32)).astype(np.int64)
    flat = np.ravel_multi_index((idx[:, o[0]], idx[:, o[1]], idx[:, o[2]]), (N,) * 3)
    del idx
    jd = os.path.join(a.out, a.run, "Jalpha")
    J = (load_3d(os.path.join(jd, f"Jalpha_H_{render}.hdf5"), "J_alpha")
         + load_3d(os.path.join(jd, f"Jalpha_AGN_H_{render}.hdf5"), "Jalpha_AGN_H")).ravel()
    if J.size != N**3:
        sys.exit(f"STOP: J has {J.size} pixels, render N = {N}")
    rho_bar = M["rho_mean"]
    n = len(C["m"])
    print(f"  {n:,} cells, axis order {o}, <J> {J.mean():.4e}", flush=True)

    ts = np.zeros((2, 3))                          # [TS_SEL][W, W/T, W (1 - T_g/T_S)]
    sV = sVt = 0.0
    nsf = 0
    for i0 in range(0, n, a.chunk):
        sl = slice(i0, min(n, i0 + a.chunk))
        sf = C["sf"][sl]
        m = C["m"][sl].astype(np.float64)
        rho = C["rho"][sl].astype(np.float64)
        T = C["T"][sl].astype(np.float64)
        x = np.where(sf, 0.0, np.clip(C["x"][sl].astype(np.float64), 0.0, 1.0))
        f2 = np.where(sf, 0.0, C["f2"][sl].astype(np.float64))
        f3 = np.where(sf, 0.0, C["f3"][sl].astype(np.float64))
        try:
            tb, _, _, ex = L.tb(rho * (1.0 - x) / rho_bar, nfac * rho, T, x, f2, f3, J[flat[sl]], full=True)
        except ValueError as e:
            sys.exit(f"STOP: cells {sl.start}-{sl.stop}: {e}")
        sp = 1.0 - Tg / ex["TS"]
        if not (np.all(np.isfinite(tb)) and np.all(np.isfinite(sp))):
            sys.exit(f"STOP: non-finite dT_b or T_S in cells {sl.start}-{sl.stop}")
        V = m / rho
        sV += V.sum(); sVt += np.sum(V * tb)
        w = X_H * m * (1.0 - x)
        for i, s_ in enumerate((~sf & (x < XNEU), slice(None))):
            ts[i] += [w[s_].sum(), np.sum(w[s_] / T[s_]), np.sum(w[s_] * sp[s_])]
        nsf += int(sf.sum())
    mean = sVt / sV
    print(f"  box <dT_b> {mean:+.5f} mK ({n:,} cells, {nsf:,} star-forming; {time.time() - t0:.0f} s)")
    res = dict(z=z, snap=a.snap, render=render, run=a.run, Tgamma=Tg, ts=ts, mean_tb=mean, sum_V=sV, n_cells=n,
               solver=a.solver, sel=np.array(TS_SEL))
    for i, lab in enumerate(TS_SEL):
        W = ts[i, 0]
        Th, s = W / ts[i, 1], ts[i, 2] / W
        print(f"  {lab:15s}: <1/T>^-1 {Th:9.3f} K | <1 - T_g/T_S> {s:+.4f} | T_S,eff {Tg / (1.0 - s):9.3f} K "
              f"(H I share {W / ts[1, 0]:.4f})")
        sfx = "" if i == 0 else "_all"
        res.update({f"T{sfx}": Th, f"spin{sfx}": s, f"TS{sfx}": Tg / (1.0 - s)})
    pc = os.path.join(a.out, a.run, "cells_Tb", f"cells_Tb_{a.snap:03d}.npz")
    if os.path.exists(pc):
        P = np.load(pc)
        print(f"  check vs {pc}: <dT_b> {mean:+.6f} vs {float(P['mean_cells_cell']):+.6f} mK; all-H I <1 - T_g/T_S> "
              f"{res['spin_all']:+.6f} vs {float(P['HI_spin_cell']):+.6f}")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"cells_TTS_{a.snap:03d}.npz")
    np.savez(fn + ".part.npz", **res)
    os.replace(fn + ".part.npz", fn)
    print(f"wrote {fn} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
