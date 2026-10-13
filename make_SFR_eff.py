"""
make_SFR_eff.py
===============
Effective star formation rate boxes for compute_Jalpha.c (H band) and
compute_Jalpha_He.c (He II band), so that each voxel emits according to its
OWN recent star formation history instead of "lifetime yield x current SFR".

The emissivity of a voxel at time t_k is the sum over the stars formed in it
at earlier times t, each emitting at its current age t_k - t:
    eps(x, t_k) = (1/m_p) Int SFR(x, t) sum_j w_j(t) l_j(t_k - t) dt,
with l_j(tau) the BPASS band photon rate per unit mass at age tau and
metallicity Z_j, and w_j(t) the young-star metallicity weights of the stars
formed at t (Z_weights.txt). The J_alpha codes use eps = SFR x eps_mix(nu, z),
eps_mix = sum_j w_j(z) N_j(nu) the lifetime yield, so they give the right
emissivity if they read
    SFR_eff(x, t_k) = Int SFR(x, t) sum_j w_j(t) l_j(t_k - t) dt / sum_j w_j(t_k) N_j
                    = sum_m W_km SFR(x, t_m)
instead of SFR. For an SFR constant over the window, SFR_eff = SFR times the
fraction of the lifetime yield emitted within the window.

How it is computed
------------------
* Times: each render file's Header/Redshift (the attribute the C codes read)
  is converted to cosmic time with the LUMINA cosmology. No regular spacing
  is assumed anywhere.
* SFR(x, t) is linear in t between consecutive render snapshots and zero
  before the first snapshot. (The C codes interpolate the boxes they read
  linearly in z; for the 2-5 Myr snapshot spacing the interpolation weights
  differ by <= 1.2e-3 at z <= 10, 8e-3 at z = 25: negligible.)
* Only stars younger than TAU_MAX contribute: 300 Myr for the H band, 100 Myr
  for the He II band (>= 94% and >= 96% of the lifetime yield). Older stars
  are dropped, so SFR_eff is slightly low for an old, steady population.
* W_km is built once per band by integrating the BPASS age profile (constant
  within each BPASS age bin, as in load_spectrum) against the linear-in-time
  SFR, on an age grid of D_AGE_MYR, with the metallicity weights at each
  cohort's formation time.
* The boxes are then processed a slab of planes at a time, stepping forward
  in time and keeping in memory only the past slabs still inside a window;
  every input box is read once per slab and in total once.

Outputs (same Header attributes as the inputs, dataset name StarFormationRate
so the C codes need no change; attributes 'content', 'band', 'tau_max_Myr', 'stride',
'plane_range' and 'complete' (0 until every slab is written)):
  <out>/H/All/All_NNN.hdf5                              -> compute_Jalpha.c   (sfr_dir argument)
  <out>/He/StarFormationRate/StarFormationRate_NNN.hdf5 -> compute_Jalpha_He.c (-DSFR_DIR_OLD=...)
  <out>/SFR_eff_summary.csv: per snapshot z, t, window size and box means.

Usage
-----
  python make_SFR_eff.py --out /path/to/SFR_eff --weights Z_weights.txt [--planes 32]

Convergence test of the linear-in-time interpolation (compare_SFR_eff.py):
  --stride S            keep every S-th snapshot (in time order), starting from the first
  --plane-range I0 I1   process only planes I0 <= i < I1 (the rest of each box stays zero)
e.g. on the same 64 planes:
  python make_SFR_eff.py --out test/s1 --weights W --plane-range 0 64
  python make_SFR_eff.py --out test/s2 --weights W --plane-range 0 64 --stride 2
  python compare_SFR_eff.py test/s1 test/s2
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

try:                                       # registers extra HDF5 filters if needed
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from make_Z_snapshots import BPASS_FILE, H0_S, MYR_S, OL, OM, bpass_band_rates, logZ_split

RENDER_DIRS = [
    "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640",
    "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640",
]
IN_PATTERN = "All/All_{:03d}.hdf5"         # as read by compute_Jalpha.c
DSET = "StarFormationRate"
N_SNAPS = 709
TAU_MAX_MYR = {"H": 300.0, "He": 100.0}
OUT_PATTERN = {"H": "H/All/All_{:03d}.hdf5",
               "He": "He/StarFormationRate/StarFormationRate_{:03d}.hdf5"}
D_AGE_MYR = 0.02                           # age-integration step (first BPASS bin is 1.12 Myr wide)

_PREF = 2.0 / (3.0 * H0_S * np.sqrt(OL)) / MYR_S   # Myr
_A = np.sqrt(OL / OM)

def t_of_z(z):
    """Cosmic time [Myr] at redshift z (flat LCDM, as in the C codes' H(z))."""
    return _PREF * np.arcsinh(_A * (1.0 + np.asarray(z, float))**-1.5)

def z_of_t(t):
    return (np.sinh(np.asarray(t, float) / _PREF) / _A)**(-2.0 / 3.0) - 1.0

# ---------------------------------------------------------------- inputs
def find_snapshots(dirs, n_snaps):
    """(snap, path, z) for every render snapshot found (first directory first),
    ordered in time; the redshift is read from each file's header."""
    out = []
    for s in range(n_snaps):
        path = next((p for p in (os.path.join(d, IN_PATTERN.format(s)) for d in dirs) if os.path.exists(p)), None)
        if path is None:
            print(f"  WARNING: snapshot {s:03d} not found in any directory, skipped")
            continue
        with h5py.File(path, "r") as f:
            out.append((s, path, float(f["Header"].attrs["Redshift"])))
    out.sort(key=lambda r: -r[2])                                       # high z (early) first
    z = np.array([r[2] for r in out])
    if np.any(np.diff(z) >= 0):
        sys.exit("ERROR: two snapshots have the same redshift")
    if [r[0] for r in out] != sorted(r[0] for r in out):
        print("  WARNING: snapshot numbers are not in time order; using the header redshifts")
    return out

def read_weights(path, Zg):
    """Young-star metallicity weights, same rules as load_Z_weights() in the C codes."""
    zs, rows, grid_ok = [], [], False
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if "Z_grid =" in line:
                    g = np.array(line.split("Z_grid =")[1].split(), float)
                    if g.size != Zg.size or np.any(np.abs(np.log10(g) - np.log10(Zg)) > 1e-3 * (np.abs(np.log10(Zg)) + 1)):
                        sys.exit(f"ERROR: Z_grid in {path} does not match the BPASS metallicities")
                    grid_ok = True
                continue
            v = line.split()
            if len(v) < 1 + Zg.size:
                continue
            w = np.array(v[1:1 + Zg.size], float)
            if w.sum() > 0:
                zs.append(float(v[0])); rows.append(w / w.sum())
    if not zs:
        sys.exit(f"ERROR: no usable rows in {path}")
    if not grid_ok:
        print(f"  WARNING: {path} has no Z_grid line to check")
    o = np.argsort(zs)
    return np.array(zs)[o], np.array(rows)[o]

def weights_at(zs, W, z):
    """(len(z), N_Z) weights: linear in z between rows, end rows held."""
    z = np.atleast_1d(np.asarray(z, float))
    return np.stack([np.interp(z, zs, W[:, j]) for j in range(W.shape[1])], axis=1)

# ---------------------------------------------------------------- weights
def build_W(t, z_snap, rates, tau_lo, tau_hi, wz, ww, tau_max):
    """W[k, m]: contribution of SFR at snapshot m to SFR_eff at snapshot k.
    rates: (N_Z, N_age) band photon rate per baryon, constant within age bins."""
    n = t.size
    N_life = (rates * ((tau_hi - tau_lo) * MYR_S)[None, :]).sum(axis=1)  # (N_Z,) lifetime yield
    tau = np.arange(0.5 * D_AGE_MYR, tau_max, D_AGE_MYR)                  # age-grid midpoints [Myr]
    ib = np.clip(np.searchsorted(tau_hi, tau, side="right"), 0, tau_hi.size - 1)
    W = np.zeros((n, n))
    for k in range(n):
        tf = t[k] - tau                                                   # formation times
        ok = tf >= t[0]                                                   # no stars before snapshot 0
        if not ok.any():
            continue
        tf, a = tf[ok], ib[ok]
        lam = np.einsum("vj,jv->v", weights_at(wz, ww, z_of_t(tf)), rates[:, a])   # photons/s/baryon
        m = np.clip(np.searchsorted(t, tf, side="right") - 1, 0, n - 2)    # t[m] <= tf < t[m+1]
        u = (tf - t[m]) / (t[m + 1] - t[m])                                # linear SFR between snapshots
        c = lam * (D_AGE_MYR * MYR_S)
        W[k] = (np.bincount(m, weights=c * (1 - u), minlength=n)
                + np.bincount(m + 1, weights=c * u, minlength=n))
        W[k] /= float(weights_at(wz, ww, z_snap[k])[0] @ N_life)
    return W

# ---------------------------------------------------------------- boxes
def create_outputs(snaps, out_root, bands, N, in_attrs, stride, plane_range):
    for band in bands:
        for (s, path, z) in snaps:
            p = os.path.join(out_root, OUT_PATTERN[band].format(s))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with h5py.File(p, "w") as f:
                g = f.create_group("Header")
                for k, v in in_attrs[s].items():
                    g.attrs[k] = v
                c = (min(16, N), min(128, N), min(128, N))
                d = f.create_dataset(DSET, shape=(N, N, N), dtype="f4", chunks=c,
                                     compression="gzip", compression_opts=4, fillvalue=0.0)
                d.attrs["content"] = ("effective SFR [Msun/yr per voxel]: SFR history weighted by the "
                                      "BPASS band photon rate at each age, normalised by the lifetime yield")
                d.attrs["band"] = band
                d.attrs["tau_max_Myr"] = TAU_MAX_MYR[band]
                d.attrs["stride"] = stride
                d.attrs["plane_range"] = plane_range
                d.attrs["complete"] = 0             # set to 1 only after every slab is written

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", default=RENDER_DIRS, help="render directories, searched in order")
    p.add_argument("--out", required=True, help="output root directory")
    p.add_argument("--weights", required=True, help="Z_weights.txt from make_Z_snapshots.py")
    p.add_argument("--bpass", default=BPASS_FILE)
    p.add_argument("--bands", nargs="+", default=["H", "He"], choices=["H", "He"])
    p.add_argument("--planes", type=int, default=32, help="planes per slab (memory ~ window x planes x N^2 x 4 B)")
    p.add_argument("--n-snaps", type=int, default=N_SNAPS)
    p.add_argument("--tau-max-H", type=float, default=TAU_MAX_MYR["H"])
    p.add_argument("--tau-max-He", type=float, default=TAU_MAX_MYR["He"])
    p.add_argument("--stride", type=int, default=1, help="keep every S-th snapshot (convergence test)")
    p.add_argument("--plane-range", type=int, nargs=2, metavar=("I0", "I1"), help="process planes I0..I1-1 only")
    args = p.parse_args()
    TAU_MAX_MYR.update({"H": args.tau_max_H, "He": args.tau_max_He})
    if args.stride < 1:
        sys.exit("ERROR: --stride must be >= 1")

    print("finding snapshots ...")
    snaps = find_snapshots(args.dirs, args.n_snaps)
    if args.stride > 1:
        snaps = snaps[::args.stride]
        print(f"  --stride {args.stride}: keeping every {args.stride}th snapshot, {len(snaps)} left")
    z = np.array([r[2] for r in snaps]); t = t_of_z(z)
    dt = np.diff(t)
    print(f"  {len(snaps)} snapshots, z = {z[0]:.3f} .. {z[-1]:.3f}, t = {t[0]:.1f} .. {t[-1]:.1f} Myr, "
          f"spacing {dt.min():.2f} .. {dt.max():.2f} Myr (median {np.median(dt):.2f})")

    Zg, tau_lo, tau_hi, rates = bpass_band_rates(args.bpass)
    wz, ww = read_weights(args.weights, Zg)
    print(f"  BPASS: {Zg.size} metallicities, {tau_lo.size} age bins up to {tau_hi[-1]:.0f} Myr; "
          f"weights: {wz.size} rows, z = {wz[0]:.2f} .. {wz[-1]:.2f}")

    Wb = {}
    for band in args.bands:
        t0 = time.time()
        Wb[band] = build_W(t, z, rates[band], tau_lo, tau_hi, wz, ww, TAU_MAX_MYR[band])
        win = (Wb[band] != 0).sum(axis=1)
        print(f"  W[{band}]: tau_max = {TAU_MAX_MYR[band]:g} Myr, window {win.min()} .. {win.max()} snapshots "
              f"(median {int(np.median(win))}), built in {time.time() - t0:.1f} s")

    n = len(snaps)
    nz_any = np.zeros((n, n), bool)
    for W in Wb.values():
        nz_any |= W != 0
    first_m = np.array([np.flatnonzero(nz_any[k])[0] if nz_any[k].any() else k for k in range(n)])
    last_use = np.array([np.flatnonzero(nz_any[:, m]).max() if nz_any[:, m].any() else m for m in range(n)])

    in_attrs = {}
    for (s, path, _) in snaps:
        with h5py.File(path, "r") as f:
            in_attrs[s] = dict(f["Header"].attrs)
            N = f[DSET].shape[0]
            chunks = f[DSET].chunks
    print(f"  boxes: {N}^3, input chunks {chunks}, slab = {args.planes} planes; "
          f"peak window {int((last_use - np.arange(n)).max()) + 1} slabs "
          f"~ {((last_use - np.arange(n)).max() + 1) * args.planes * N * N * 4 / 1e9:.1f} GB")
    p0, p1 = args.plane_range if args.plane_range else (0, N)
    if not 0 <= p0 < p1 <= N:
        sys.exit(f"ERROR: --plane-range must satisfy 0 <= I0 < I1 <= {N}")
    if (p0, p1) != (0, N):
        print(f"  --plane-range: planes {p0}..{p1 - 1} only")
    create_outputs(snaps, args.out, args.bands, N, in_attrs, args.stride, np.array([p0, p1], np.int32))

    sum_sfr = np.zeros(n); sum_eff = {b: np.zeros(n) for b in args.bands}
    t_start = time.time()
    for i0 in range(p0, p1, args.planes):
        i1 = min(p1, i0 + args.planes)
        buf = {}
        for k, (s, path, _) in enumerate(snaps):
            with h5py.File(path, "r") as f:
                sl = f[DSET][i0:i1].astype(np.float32)
            np.maximum(sl, 0.0, out=sl)                                    # as the C codes do
            sum_sfr[k] += sl.sum(dtype=np.float64)
            flat = sl.ravel()
            nzi = np.flatnonzero(flat)                                     # SFR is sparse: keep non-zeros only
            buf[k] = (nzi, flat[nzi].astype(np.float64)) if nzi.size else None
            for band, W in Wb.items():
                idx, val = [], []
                for m in range(first_m[k], k + 1):
                    if W[k, m] != 0.0 and buf.get(m) is not None:
                        idx.append(buf[m][0]); val.append(W[k, m] * buf[m][1])
                if idx:
                    E = np.bincount(np.concatenate(idx), weights=np.concatenate(val),
                                    minlength=flat.size).astype(np.float32).reshape(sl.shape)
                else:
                    E = np.zeros_like(sl)
                sum_eff[band][k] += E.sum(dtype=np.float64)
                with h5py.File(os.path.join(args.out, OUT_PATTERN[band].format(s)), "r+") as f:
                    f[DSET][i0:i1] = E
            for m in [m for m in buf if last_use[m] <= k]:
                del buf[m]
        el = time.time() - t_start
        print(f"  planes {i0:4d}-{i1 - 1:4d} done, {el / 60:.1f} min elapsed, "
              f"~{el / (i1 - p0) * (p1 - i1) / 60:.1f} min left", flush=True)

    # every slab is written: mark the files complete (the J_alpha codes refuse complete = 0)
    for band in args.bands:
        for (s, _, _) in snaps:
            with h5py.File(os.path.join(args.out, OUT_PATTERN[band].format(s)), "r+") as f:
                f[DSET].attrs["complete"] = 1

    csv = os.path.join(args.out, "SFR_eff_summary.csv")
    with open(csv, "w") as f:
        f.write("snap,redshift,t_Myr," + ",".join(f"window_{b},ratio_{b}" for b in args.bands) + ",mean_SFR\n")
        for k, (s, _, zk) in enumerate(snaps):
            cols = []
            for b in args.bands:
                r = sum_eff[b][k] / sum_sfr[k] if sum_sfr[k] > 0 else float("nan")
                cols += [str(int((Wb[b][k] != 0).sum())), f"{r:.6g}"]
            f.write(f"{s},{zk:.6f},{t[k]:.4f}," + ",".join(cols) + f",{sum_sfr[k] / ((p1 - p0) * N * N):.6e}\n")
    print(f"wrote {csv}")
    print("box-mean SFR_eff / SFR (compare with the R(z) of the global age test):")
    for zc in (20, 15, 12, 10, 8, 6, 5, 4, 3):
        k = int(np.argmin(np.abs(z - zc)))
        print(f"  z = {z[k]:6.3f}: " + "  ".join(
            f"{b}: {sum_eff[b][k] / sum_sfr[k]:.3f}" if sum_sfr[k] > 0 else f"{b}: n/a" for b in args.bands))

if __name__ == "__main__":
    main()
