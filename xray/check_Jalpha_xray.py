"""
check_Jalpha_xray.py
====================
Sanity check of the J_alpha outputs of the X-ray test runs (run_Jalpha_xray.sh), from the J_mean and Redshift
attributes only (fast): every render of both runs readable, finite and >= 0, stars and AGN at the same z, z strictly
decreasing and identical in both runs; the renders where J_stars and J_AGN first become > 0; and, at selected renders,
the AGN share, the ratio withXrays / withoutXrays, and withoutXrays / Lumina (production J, log-interpolated in z).
Run from anywhere:  python check_Jalpha_xray.py      (JBASE=<dir> overrides /orcd/data/mvogelsb/004/chcho)
Tested on mock files with planted ratios (2026-11-16).
"""
import glob, os, h5py, numpy as np
B = os.environ.get("JBASE", "/orcd/data/mvogelsb/004/chcho")
def attrs(f, d):
    with h5py.File(f, "r") as h:
        a = h[d].attrs
        return float(a["Redshift"]), float(a["J_mean"])
lum = {}
for f in glob.glob(f"{B}/Jalpha/Jalpha_H_*.hdf5"):
    s = int(os.path.basename(f)[len("Jalpha_H_"):-len(".hdf5")])
    try:
        z, js = attrs(f, "J_alpha")
        _, ja = attrs(f"{B}/Jalpha/Jalpha_AGN_H_{s}.hdf5", "Jalpha_AGN_H")
    except Exception as e:
        print(f"Lumina render {s} skipped: {e}"); continue
    lum[z] = js + ja
zl = np.array(sorted(lum)); jl = np.array([lum[z] for z in zl])
print(f"Lumina J files: {len(zl)}, z = {zl[0]:.3f} .. {zl[-1]:.3f}")
res = {}
for run in ("withoutXrays", "withXrays"):
    J = f"{B}/xray/{run}/Jalpha"; rows, bad = [], []
    for s in range(405):
        try:
            z, js = attrs(f"{J}/Jalpha_H_{s}.hdf5", "J_alpha")
            z2, ja = attrs(f"{J}/Jalpha_AGN_H_{s}.hdf5", "Jalpha_AGN_H")
        except Exception as e:
            bad.append((s, f"unreadable: {e}")); continue
        if abs(z - z2) > 1e-6: bad.append((s, f"z stars {z} vs AGN {z2}"))
        if not (np.isfinite(js) and np.isfinite(ja)) or js < 0 or ja < 0: bad.append((s, f"J_mean {js} {ja}"))
        rows.append((s, z, js, ja))
    if not rows:
        raise SystemExit(f"{run}: no render readable")
    res[run] = np.array(rows)
    print(f"{run}: {len(rows)} renders read, {len(bad)} problems {bad[:5]}")
    zz = res[run][:, 1]
    print(f"  z = {zz[0]:.4f} .. {zz[-1]:.4f}, strictly decreasing: {bool(np.all(np.diff(zz) < 0))}; "
          f"first render with J_stars > 0: {int(res[run][np.argmax(res[run][:, 2] > 0), 0]) if np.any(res[run][:, 2] > 0) else 'none'}, "
          f"first with J_AGN > 0: {int(res[run][np.argmax(res[run][:, 3] > 0), 0]) if np.any(res[run][:, 3] > 0) else 'none'}")
a, b = res["withoutXrays"], res["withXrays"]
print("same z grid in both runs:", bool(a.shape == b.shape and np.allclose(a[:, 1], b[:, 1], rtol=0, atol=1e-6)))
print("render      z    J_tot(noX)   AGN share(noX)  J_tot withX/noX  J_tot noX/Lumina")
for s in (0, 30, 60, 71, 100, 112, 154, 182, 207, 254, 302, 344, 404):
    i = np.flatnonzero(a[:, 0] == s); k = np.flatnonzero(b[:, 0] == s)
    if not len(i) or not len(k):
        continue
    _, z, js, ja = a[i[0]]; jt = js + ja; jb = b[k[0], 2] + b[k[0], 3]
    jL = np.exp(np.interp(z, zl, np.log(np.maximum(jl, 1e-300)))) if zl[0] <= z <= zl[-1] else np.nan
    f = lambda x, y: f"{x / y:9.4f}" if y > 0 else "      n/a"
    print(f"{s:6d} {z:8.3f}  {jt:11.4e}   {f(ja, jt)}       {f(jb, jt)}       {f(jt, jL)}")
