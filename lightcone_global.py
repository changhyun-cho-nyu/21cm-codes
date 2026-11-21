"""
lightcone_global.py
===================
Global history of the 21 cm signal along the Lumina light cone, slice by slice (640 x 640 pixels, 5896 slices,
z 30 -> 2.987): <delta T_b>, its rms, spin and kinetic temperatures, ionized fractions, x_alpha, x_c, J_alpha.

delta T_b, T_S, x_c, x_alpha are recomputed with the settings of an existing make_lightcone_tb.py H file (read from
its attributes: cones, J directory and tag, time, outside, spin model), through the same cone_tb call (extras=True),
and the recomputed delta T_b is compared with that file chunk by chunk: it stops if any pixel differs by more than
1e-3 mK + 1e-5 x the chunk's max |delta T_b| (0 on the node that made the file; numpy's math is CPU-dispatched,
so another node can differ in the last bit). Needs the
21 cm spin model 1 (the T_S of tb_physics.Line.tb full=True).

Per slice (one row each; z decreasing along the cone; all means over the slice's pixels):
  z, z_hi, z_lo, chi_lo, chi_hi [cMpc/h], T_gamma [K]
  dTb_mean, dTb_rms, f_emit (fraction of pixels with dT_b > 0) [mK]; dTb_mean_file (the cone file's 'mean')
  delta_mean = <rho> / (Omega_b rho_crit)
  xHII_V  = sum(Volume HII_VolumeFraction) / sum(Volume)      (as ion_history/ionization_history.csv xHII_V)
  xHII_M  = sum(rho HII_Fraction) / sum(rho);  xHeII_V, xHeIII_V (Volume-weighted *_VolumeFraction / y), xHeIII_M
  TK_M    = sum(rho T) / sum(rho)  (T = Temperature, mass-weighted per pixel);  TK_V = sum(Volume T_V) / sum(Volume)
  TK_HI   = sum(rho_HI T) / sum(rho_HI);  TK_HI_harm = sum(rho_HI) / sum(rho_HI / T)   (rho_HI = rho (1 - HII_Fraction))
  spin_HI = sum(rho_HI (1 - T_gamma/T_S)) / sum(rho_HI): <dT_b> = P(z) <(1 + delta) x_HI (1 - T_gamma/T_S) x_CMB>, so
            apart from x_CMB (>= 0.987) <dT_b> = P(z) (<rho_HI> / rho_mean) spin_HI
  TS_eff  = T_gamma / (1 - spin_HI): the single T_S that gives the H I-weighted spin factor (what sets <dT_b>)
  TS_HI   = sum(rho_HI T_S) / sum(rho_HI) (arithmetic), TS_med (median over pixels)
  xa_HI, xc_HI = H I-weighted x_alpha (with S~_alpha) and x_c;  xa_V, xc_V pixel means;  J_V pixel mean J_alpha
Pixels within a slice have equal volume to 0.3% (tan-angle pixels: (1 + t_x^2 + t_y^2)^-3/2 >= 0.997), so pixel
means are volume means.

Output: CSV (--out), appended chunk by chunk; a rerun skips the slices already in it (resume after a time-out).
Same limits as the cone delta T_b (JALPHA_NOTES 4.7): real space, star-forming gas neutral, cone pixels 0.43-0.73
cMpc/h; 21 cm at z >~ 10 is resolution-dependent (temperature mixing in a pixel; the gas-cell runs, 6.1 item 11).

Usage
-----
  python lightcone_global.py --cone-file DIR/lightcone_dTb_H.hdf5 --out DIR/lightcone_global_H.csv [--chunk 64]
                             [--slices I0 I1 | --zrange ZLO ZHI]
"""
import argparse
import os
import subprocess

import h5py
import numpy as np
import pandas as pd

from lightcone_J import cone_J
from lightcone_tb import CONE_ORIGIN, cone_tb, read_cone, units
from tb_physics import Y_HE

FIELDS = ("Density", "Temperature", "HII_Fraction", "HeII_Fraction", "HeIII_Fraction", "Volume",
          "HII_VolumeFraction", "HeII_VolumeFraction", "HeIII_VolumeFraction", "TemperatureVolumeWeighted")
COLS = ["slice", "z", "z_hi", "z_lo", "chi_lo", "chi_hi", "T_gamma", "dTb_mean", "dTb_rms", "f_emit", "dTb_mean_file",
        "delta_mean", "xHII_V", "xHII_M", "xHeII_V", "xHeIII_V", "xHeIII_M", "TK_M", "TK_V", "TK_HI", "TK_HI_harm",
        "spin_HI", "TS_eff", "TS_HI", "TS_med", "xa_HI", "xa_V", "xc_HI", "xc_V", "J_V"]


def slice_stats(cone, tb, ex, J, rho_mean):
    """one dict per slice of the chunk (arrays (N, N, nc))"""
    ax = (0, 1)
    f64 = np.float64
    rho = cone["Density"].astype(f64)
    V = cone["Volume"].astype(f64)
    rhoHI = rho * (1.0 - cone["HII_Fraction"].astype(f64))
    T = cone["Temperature"].astype(f64)
    Tg = 2.7255 * (1.0 + cone["z"])
    TS = ex["TS"].astype(f64)
    sr, sV, sHI = rho.sum(ax), V.sum(ax), rhoHI.sum(ax)
    spin = (rhoHI * (1.0 - Tg / TS)).sum(ax) / sHI
    tb64 = tb.astype(f64)
    return dict(
        T_gamma=Tg,
        dTb_mean=tb64.mean(ax), dTb_rms=tb64.std(ax), f_emit=(tb > 0).mean(ax),
        delta_mean=rho.mean(ax) / rho_mean,
        xHII_V=(V * cone["HII_VolumeFraction"]).sum(ax) / sV,
        xHII_M=(rho * cone["HII_Fraction"]).sum(ax) / sr,
        xHeII_V=(V * cone["HeII_VolumeFraction"]).sum(ax) / sV / Y_HE,
        xHeIII_V=(V * cone["HeIII_VolumeFraction"]).sum(ax) / sV / Y_HE,
        xHeIII_M=(rho * cone["HeIII_Fraction"]).sum(ax) / sr / Y_HE,
        TK_M=(rho * T).sum(ax) / sr,
        TK_V=(V * cone["TemperatureVolumeWeighted"]).sum(ax) / sV,
        TK_HI=(rhoHI * T).sum(ax) / sHI,
        TK_HI_harm=sHI / (rhoHI / T).sum(ax),
        spin_HI=spin, TS_eff=Tg / (1.0 - spin),
        TS_HI=(rhoHI * TS).sum(ax) / sHI,
        TS_med=np.median(TS.reshape(-1, TS.shape[-1]), axis=0),
        xa_HI=(rhoHI * ex["xa"]).sum(ax) / sHI, xa_V=ex["xa"].astype(f64).mean(ax),
        xc_HI=(rhoHI * ex["xc"]).sum(ax) / sHI, xc_V=ex["xc"].astype(f64).mean(ax),
        J_V=J.astype(f64).mean(ax),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cone-file", required=True, help="make_lightcone_tb.py output, lightcone_dTb_H.hdf5")
    ap.add_argument("--out", required=True, help="CSV")
    ap.add_argument("--chunk", type=int, default=64, help="slices per chunk")
    ap.add_argument("--slices", type=int, nargs=2, default=None, help="only slices I0 <= i < I1 (a test)")
    ap.add_argument("--zrange", type=float, nargs=2, default=None, help="only the slices lying in ZLO <= z <= ZHI")
    a = ap.parse_args()

    with h5py.File(a.cone_file, "r") as f:
        at = dict(f.attrs)
        ze, ce, done, fmean = f["z_edges"][()], f["chi_edges"][()], f["done"][()], f["mean"][()]
    if at.get("line") != "H" or int(at.get("spin_model", 0)) != 1:
        raise ValueError(f"{a.cone_file}: line {at.get('line')}, spin model {at.get('spin_model', 0)}; needs H, model 1")
    if not np.allclose(at["cone_origin"], CONE_ORIGIN):
        raise ValueError(f"cone origin {at['cone_origin']} differs from lightcone_tb.CONE_ORIGIN {CONE_ORIGIN}")
    cones = str(at["cones"]).split(";")
    ns = len(ze) - 1
    i0, i1 = a.slices if a.slices else (0, ns)
    if a.zrange:
        s = np.flatnonzero((ze[:-1] <= max(a.zrange)) & (ze[1:] >= min(a.zrange)))
        i0, i1 = int(s[0]), int(s[-1]) + 1
    if not done[i0:i1].all():
        raise ValueError(f"{np.sum(~done[i0:i1])} of the slices {i0} - {i1 - 1} are not done in {a.cone_file}")

    have = set()
    if os.path.exists(a.out):
        have = set(pd.read_csv(a.out, comment="#")["slice"].astype(int))
    else:
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        except OSError:
            commit = "unknown"
        with open(a.out, "w") as fo:
            fo.write(f"# lightcone_global.py (commit {commit}); cone file {a.cone_file} (commit {at.get('commit')}); "
                     f"J {at['jdir']} tag {at['tag']}, time {at['time']}, outside {at['outside']}, spin model 1\n")
            fo.write("# per slice; pixel = volume means; dT_b in mK, temperatures in K, chi in cMpc/h, J in proper "
                     "photons s^-1 cm^-2 Hz^-1 sr^-1; see the script's docstring for each column\n")
            fo.write(",".join(COLS) + "\n")
    print(f"cone file {a.cone_file}: {ns} slices; processing {i0} - {i1 - 1} in chunks of {a.chunk}; "
          f"{len(have)} slices already in {a.out}", flush=True)

    worst = 0.0
    for c0 in range(i0, i1, a.chunk):
        c1 = min(c0 + a.chunk, i1)
        if all(j in have for j in range(c0, c1)):
            continue
        cone = read_cone(cones, fields=FIELDS, sel=np.s_[:, :], zmin=ze[c1], zmax=ze[c0])
        if cone["Density"].shape[-1] != c1 - c0 or not np.allclose(cone["z_edges"], ze[c0:c1 + 1], rtol=0, atol=1e-12):
            raise RuntimeError(f"slices {c0} - {c1 - 1}: read_cone returned {cone['Density'].shape[-1]} slices")
        J, _ = cone_J(cone, "H", np.s_[:, :], time=str(at["time"]), outside=str(at["outside"]), jdir=str(at["jdir"]),
                      tag=str(at["tag"]), verbose=False)
        tb, ex = cone_tb(cone, "H", J, spin_model=1, extras=True)
        with h5py.File(a.cone_file, "r") as f:
            ref = f["dTb"][:, :, c0:c1]
        d = np.abs(tb - ref)
        diff, ndiff, tol = float(d.max()), int(np.count_nonzero(d)), 1e-3 + 1e-5 * float(np.abs(ref).max())
        del ref, d
        worst = max(worst, diff)
        if diff > tol:
            raise RuntimeError(f"slices {c0} - {c1 - 1}: recomputed dT_b differs from {a.cone_file} by up to {diff:.3g} "
                               f"mK (> {tol:.3g}; different code or inputs since the cone file was made)")
        st = slice_stats(cone, tb, ex, J, units(cone["header"])[1])
        rows = pd.DataFrame(st)
        rows.insert(0, "slice", np.arange(c0, c1))
        rows.insert(1, "z", cone["z"])
        rows.insert(2, "z_hi", ze[c0:c1])
        rows.insert(3, "z_lo", ze[c0 + 1:c1 + 1])
        rows.insert(4, "chi_lo", np.minimum(ce[c0:c1], ce[c0 + 1:c1 + 1]))
        rows.insert(5, "chi_hi", np.maximum(ce[c0:c1], ce[c0 + 1:c1 + 1]))
        rows["dTb_mean_file"] = fmean[c0:c1]
        rows = rows[COLS]
        rows.to_csv(a.out, mode="a", header=False, index=False, float_format="%.8g")
        print(f"  slices {c0:4d} - {c1 - 1:4d} (z {ze[c0]:.4f} - {ze[c1]:.4f}): <dT_b> {rows.dTb_mean.mean():+.4g} mK, "
              f"T_S,eff {rows.TS_eff.mean():.4g} K, T_K,HI harm {rows.TK_HI_harm.mean():.4g} K, x_HII,V "
              f"{rows.xHII_V.mean():.4g}, x_a,HI {rows.xa_HI.mean():.3g}; dT_b vs file: max |diff| {diff:.1g} mK, {ndiff} pixels differ", flush=True)
        del cone, J, tb, ex
    print(f"done; max |recomputed dT_b - cone file| over the processed chunks {worst:.1g} mK")


if __name__ == "__main__":
    main()
