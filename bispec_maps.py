"""
bispec_maps.py
==============
Bispectra of the production maps with BiFFT (Watkinson et al. 2017; ~/apps/bifft on ORCD), in h/cMpc.

Fields (--field):
  H       21 cm delta T_b, Tb_maps_prod/H_real_wf/RIN_<snap>.hdf5 (dataset rin; SPIN_MODEL 1, real space);
          <snap> = render snapshot
  He      3.46 cm delta T_b, the gas-cell maps <cell-root>/cell_test*/RIN_cell_He_NNN.hdf5 (rin), else
          cellmap_He_NNN.hdf5 (dTb_cell), found as make_vid.py does; <snap> = FULL snapshot (102 ... 158)
  xHII    HII_VolumeFraction of the 640^3 renders                 ; <snap> = render snapshot
  xHI     1 - HII_VolumeFraction                                   ; render snapshot
  xHeIII  HeIII_VolumeFraction / y  (y = n_He/n_H = 0.0789)        ; render snapshot
  xHeII   HeII_VolumeFraction / y                                  ; render snapshot
  Tests of the early He sign flip (2026-11-23), all render snapshots:
  xnotHeIII      1 - HeIII_VolumeFraction / y  (= x_HeII + x_HeI: x_HeII without the He I dips); B = -B(xHeIII)
  xnotHeIII_bin  1 - [x_HeIII > 0.5]  (the neutral/ionized split the He MFP uses: mfp/lumina-mfp-hist-grid.cc)
  xHI_bin        1 - [x_HII > 0.5]    (the same for H)
Ionization fields are volume-weighted (CLAUDE.md, 2026-11-19). The normalised B is unchanged by a constant
factor (so / y does not matter) but changes SIGN between x and 1 - x: B(xHI) = -B(xHII) exactly, and
B(xHeII) ~ -B(xHeIII) where He I is negligible (z < 4.75). He II is to the 3.46 cm line what H I is to 21 cm,
so the matching pairs are (xHI, xHeII) (the emitting species; 3.46 cm work uses xHeII, user 2026-11-19) or
(xHII, xHeIII) (the ionized species); a mixed pair gives opposite-sign evolution by construction.

Box and k units: LEN = --box = 338.3 cMpc/h (500 cMpc x h, h = 0.6766), so every k (input and output) is in
h/cMpc, k_F = 2 pi / 338.3 = 0.0186 and k_Nyq = 5.94 h/cMpc on 640^3. For render fields the header BoxSize is
checked against --box. B is in [map unit]^3 (cMpc/h)^6, P in [map unit]^2 (cMpc/h)^3.

Configurations (--config), each a BiFFT function (normalise = 1 as in the old get_*.py):
  iso   calculate_isoc_bipec(k_1 = k1): k1 = k2, theta/pi in {0.01, 0.05, 0.1, 0.2, 1/3, 0.4, 0.5, 0.6, 0.7,
        0.85, 0.95} (theta = the angle between k1 and k2 inside the triangle: 1/3 equilateral, -> 0 squeezed,
        -> 1 flattened); a bin is kept only if k3 >= 2.25 k_F and k3 < k_Nyq
  gen   calculate_general_bipec(k_1 = k1, k_2 = k2) (--k1 and --k2 paired)
  equi  calculate_equi_bipec(K_FAC = kfac): equilateral, log bins from k_F * K_FAC by factors K_FAC
BiFFT must be the patched build (double-precision sums, 2026-11-19): the original float32 sums returned
P ~ 0.5x and normalised B ~ 2.8x at 640^3. `selftest` checks it.

Output: <out>/<field>/BS_<config>_<field>_<snap:03d>_<k tag>.txt, '#' header (source, z, box, units), columns
  k3  B_norm  B_imag  k1  k2  P1  P2  P3  N_tri  theta_over_pi  B
  B_norm = B / sqrt(P1 P2 P3 / (k1 k2 k3)) (Watkinson+18, dimensionless); B_imag unnormalised (BiFFT returns it so);
  B = B_norm * sqrt(P1 P2 P3 / (k1 k2 k3)), the unnormalised bispectrum. The first nine columns are those of the
  old get_*.py files (their 10th was the row index).
Existing outputs are skipped (resumable); --task/--ntask split the (snapshot, k) list.

Usage
-----
  python bispec_maps.py run --field H --config iso --snaps 177 203 220 --k1 0.1 0.3 0.6 --out <dir> [--task 0 --ntask 4]
  python bispec_maps.py run --field He --config iso --snaps 126 131 136 --k1 0.1 0.3 --out <dir>
  python bispec_maps.py run --field xHII --config gen --snaps 242 --k1 0.15 --k2 0.3 --out <dir>
  python bispec_maps.py run --field H --config equi --snaps 242 --kfac 1.1 --out <dir>
  python bispec_maps.py selftest --n 640 --k1 0.2       # patched-BiFFT check, needs ~70 GB (run_bispec.sh SELFTEST=1)
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

import numpy as np
import h5py

try:
    import hdf5plugin  # noqa: F401   (compressed renders)
except ImportError:
    pass

U = "/orcd/data/mvogelsb/004/chcho"
REN = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
DIR_A = f"{REN}/Flagship_new/postprocessing/ren_640"                # render snapshots <= 428
DIR_B = f"{REN}/Flagship_below_z_4p75_new/postprocessing/ren_640"   # render snapshots > 428
SNAP_SWITCH = 428
BOX_MPCH = 338.3                                  # 500 cMpc x 0.6766
X_H = 0.76
Y_HE = (1.0 - X_H) / (4.0 * X_H)
RENDER_FIELDS = {"xHII": ("HII_VolumeFraction", 1.0, False, None), "xHI": ("HII_VolumeFraction", 1.0, True, None),
                 "xHeIII": ("HeIII_VolumeFraction", Y_HE, False, None), "xHeII": ("HeII_VolumeFraction", Y_HE, False, None),
                 # tests of the early He sign flip (2026-11-23): x_HeII without the He I dips, and the field the MFP sees
                 "xnotHeIII": ("HeIII_VolumeFraction", Y_HE, True, None),
                 "xnotHeIII_bin": ("HeIII_VolumeFraction", Y_HE, True, 0.5),
                 "xHI_bin": ("HII_VolumeFraction", 1.0, True, 0.5)}
# (render field, divisor, use 1 - x, threshold): x = field / divisor; with a threshold x -> 1 where x > threshold,
# else 0 (the ionized cells of mfp/lumina-mfp-hist-grid.cc, which starts rays where x > 0.5 and stops them at 0.5)
ISO_THETA = np.array([0.01, 0.05, 0.1, 0.2, 0.33333333333333333, 0.4, 0.5, 0.6, 0.7, 0.85, 0.95], np.float32)
COLS = "k3 B_norm B_imag k1 k2 P1 P2 P3 N_tri theta_over_pi B"


# ----------------------------------------------------------------------------------------------- BiFFT
def load_bifft():
    """import bifft, with np.complex restored and the per-element Python loop of complex_to_two_floats replaced by
    the identical vectorised copy (same C-order flattening, float64), which takes minutes at 640^3"""
    if not hasattr(np, "complex"):
        np.complex = complex
    import bifft
    ba = sys.modules["bifft.bispec_analysis"]

    def complex_to_two_floats(complex_data):
        c = np.asarray(complex_data)
        return (np.ascontiguousarray(c.real, dtype=np.float64).ravel(),
                np.ascontiguousarray(c.imag, dtype=np.float64).ravel())

    ba.complex_to_two_floats = complex_to_two_floats
    return bifft, ba


def bifft_build(ba):
    """where the C libraries are and when they were built (for the output header)"""
    src = getattr(ba, "SOURCE_FOLDER", "?")
    so = os.path.join(src, "Compute_21cmBS_isoceles.so")
    t = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(so))) if os.path.exists(so) else "?"
    return f"{src} (isoceles.so built {t})"


def run_config(bifft, cfg, m, box, k1, k2, kfac, num_th):
    N = m.shape[0]
    common = dict(data=m, DIM=[N], LEN=[box], normalise=1, num_th=num_th)
    if cfg == "iso":
        r = bifft.calculate_isoc_bipec(k_1=k1, **common)
    elif cfg == "gen":
        r = bifft.calculate_general_bipec(k_1=k1, k_2=k2, **common)
    else:
        r = bifft.calculate_equi_bipec(K_FAC=kfac, **common)
    r = {k: np.asarray(v, dtype=np.float64) for k, v in r.items()}
    r["B"] = r["BS"] * np.sqrt(r["P1"] * r["P2"] * r["P3"] / (r["k1"] * r["k2"] * r["k3"]))
    return r


# ------------------------------------------------------------------------------------------- the maps
def cell_maps(root, line):
    """as make_vid.cell_maps: {full snapshot: (file, dataset)}; RIN_cell_<line>_NNN wins over cellmap_<line>_NNN"""
    out = {}
    for pat, ds in ((f"cellmap_{line}_*.hdf5", "dTb_cell"), (f"RIN_cell_{line}_*.hdf5", "rin")):
        seen = {}
        for f in sorted(glob.glob(os.path.join(root, "cell_test*", pat))):
            s = int(re.search(r"_(\d+)\.hdf5$", f).group(1))
            if s in seen:
                sys.exit(f"ERROR: full snapshot {s} has two gas-cell maps: {seen[s]} and {f}")
            seen[s] = f
            out[s] = (f, ds)
    return out


def text_attr(a, key):
    v = a.get(key)
    return v.decode() if isinstance(v, bytes) else (str(v) if v is not None else "")


def load_field(a, s):
    """(map as float32 (N,N,N), z, source file, description, BoxSize [cMpc/h] or None)"""
    if a.field == "H":
        fn = os.path.join(a.h_maps, f"RIN_{s}.hdf5")
        with h5py.File(fn, "r") as f:
            d = f["rin"]
            z = float(d.attrs["Redshift"]) if "Redshift" in d.attrs else float(f.attrs.get("Redshift", np.nan))
            return d[()].astype(np.float32), z, fn, text_attr(d.attrs, "content") or "21 cm delta T_b [mK]", None
    if a.field == "He":
        cells = cell_maps(a.cell_root, "He")
        if s not in cells:
            raise FileNotFoundError(f"no gas-cell He map for full snapshot {s} under {a.cell_root}/cell_test*")
        fn, ds = cells[s]
        with h5py.File(fn, "r") as f:
            d = f[ds]
            return (d[()].astype(np.float32), float(f.attrs["z"]), fn,
                    text_attr(d.attrs, "content") or "3.46 cm delta T_b (gas-cell level)", None)
    name, div, comp, thr = RENDER_FIELDS[a.field]
    d0 = a.dir_a if s <= SNAP_SWITCH else a.dir_b
    for fn in (os.path.join(d0, name, f"{name}_{s:03d}.hdf5"), os.path.join(d0, "All", f"All_{s:03d}.hdf5")):
        if os.path.exists(fn):
            with h5py.File(fn, "r") as f:
                if name not in f:
                    continue
                hd = f["Header"].attrs if "Header" in f else {}
                z = float(hd["Redshift"]) if "Redshift" in hd else np.nan
                box = float(hd["BoxSize"]) / 1000.0 if "BoxSize" in hd else None
                m = f[name][()] / div
                x = f"{name}{' / y' if div != 1 else ''}"
                if thr is not None:
                    m = (m > thr).astype(np.float64)
                    x = f"[{x} > {thr:g}]"
                m = (1.0 - m if comp else m).astype(np.float32)
                desc = f"{'1 - ' if comp else ''}{x} (volume-weighted fraction{', binary' if thr is not None else ''})"
                return m, z, fn, desc, box
    raise FileNotFoundError(f"no {name} for render snapshot {s} in {d0}")


def ktag(cfg, k1, k2, kfac):
    if cfg == "iso":
        return f"k{k1:g}"
    if cfg == "gen":
        return f"k{k1:g}_{k2:g}"
    return f"kfac{kfac:g}"


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or "?"
    except OSError:
        return "?"


def run(a):
    if a.config in ("iso", "gen") and not a.k1:
        sys.exit("--k1 is needed for iso and gen")
    if a.config == "gen" and (not a.k2 or len(a.k2) != len(a.k1)):
        sys.exit("gen needs --k2 with as many values as --k1 (paired)")
    if a.config == "equi":
        ks = [(None, None, f) for f in a.kfac]
    else:
        ks = [(k, a.k2[i] if a.config == "gen" else k, None) for i, k in enumerate(a.k1)]
    jobs = [(s, k) for s in a.snaps for k in ks][a.task::a.ntask]
    bifft, ba = load_bifft()
    build = bifft_build(ba)
    odir = os.path.join(a.out, a.field)
    os.makedirs(odir, exist_ok=True)
    print(f"{a.field} {a.config}: {len(jobs)} (snapshot, k) jobs for task {a.task}/{a.ntask}; box {a.box} cMpc/h; "
          f"BiFFT {build}; num_th {a.num_th}", flush=True)
    nfail, cur = 0, (None, None)
    for s, (k1, k2, kfac) in jobs:
        fo = os.path.join(odir, f"BS_{a.config}_{a.field}_{s:03d}_{ktag(a.config, k1, k2, kfac)}.txt")
        if os.path.exists(fo):
            continue
        if cur[0] != s:
            cur = (None, None)
            try:
                cur = (s, load_field(a, s))
            except (OSError, KeyError) as e:
                print(f"  snap {s}: {e}; skipped", flush=True); nfail += 1; continue
        m, z, src, desc, box_hdr = cur[1]
        if box_hdr is not None and abs(box_hdr / a.box - 1) > 1e-4:
            sys.exit(f"ERROR: {src} header BoxSize {box_hdr} cMpc/h != --box {a.box}")
        if m.ndim != 3 or len(set(m.shape)) != 1:
            sys.exit(f"ERROR: {src}: map shape {m.shape} is not a cube")
        if not np.all(np.isfinite(m)):
            sys.exit(f"ERROR: {src}: non-finite values in the map")
        t0 = time.time()
        r = run_config(bifft, a.config, m, a.box, k1, k2, kfac, a.num_th)
        N = m.shape[0]
        hdr = [f"bispec_maps.py (commit {git_commit()}), {time.strftime('%Y-%m-%d %H:%M')}; BiFFT {build}",
               f"field {a.field}: {desc}; file {src}; snapshot {s}; z = {z:.6f}; N = {N}; mean {m.mean():.6e}",
               f"config {a.config} ({ktag(a.config, k1, k2, kfac)}); box {a.box} cMpc/h, k_F = {2*np.pi/a.box:.5f}, "
               f"k_Nyq = {np.pi*N/a.box:.4f} h/cMpc; normalise = 1",
               "units: k [h/cMpc]; P [map unit^2 (cMpc/h)^3]; B, B_imag [map unit^3 (cMpc/h)^6]; B_norm dimensionless",
               COLS]
        rows = np.column_stack([r["k3"], r["BS"], r["BSimag"], r["k1"], r["k2"], r["P1"], r["P2"], r["P3"],
                                r["Ntri"], r["theta_over_pi"], r["B"]]) if len(r["k3"]) else np.zeros((0, 11))
        np.savetxt(fo + ".part", rows, fmt="%.8e", delimiter="\t", header="\n".join(hdr))
        os.replace(fo + ".part", fo)
        print(f"  snap {s:03d} z = {z:.4f} {ktag(a.config, k1, k2, kfac)}: {len(rows)} bins, "
              f"{time.time()-t0:.0f} s -> {fo}", flush=True)
    if nfail:
        sys.exit(f"{nfail} snapshots could not be read")


# ------------------------------------------------------------------------------------------- selftest
def corner_range(N, KF):
    """per-axis squared extent of BiFFT's pixel cube [n-1.051, n+1.0] k_F (bispec_k3_bin, same arithmetic):
    returns (min of squares, max of squares, n k_F) for n = 0..N-1 in FFTW order (n > N/2 -> n - N)"""
    n = np.arange(N, dtype=np.float64)
    n = np.where(n > N // 2, n - N, n)
    lo_n = n - 1.051
    hi = (lo_n + 2.051) * KF
    lo = lo_n * KF
    return np.minimum(lo**2, hi**2), np.maximum(lo**2, hi**2), n * KF


def bifft_masks(N, L, K1, K3min, K3max):
    """the k1 (= k2) shell and the k3 bin as bispec_k3_bin selects them, and |k| of every pixel"""
    KF = 2.0 * np.pi / np.float64(np.float32(L))            # X_LEN is a float in the C code
    mn, mx, nk = corner_range(N, KF)
    pmin = np.sqrt(mn[:, None, None] + mn[None, :, None] + mn[None, None, :])
    pmax = np.sqrt(mx[:, None, None] + mx[None, :, None] + mx[None, None, :])
    m1 = (pmin <= K1) & (K1 <= pmax)
    m3 = (((pmax >= K3max) & (pmin <= K3max)) | ((pmax <= K3max) & (pmin >= K3min)) |
          ((pmax >= K3min) & (pmin <= K3min)))
    del pmin, pmax
    kmag = np.sqrt(nk[:, None, None]**2 + nk[None, :, None]**2 + nk[None, None, :]**2)
    return m1, m3, kmag


def reference_iso(field, L, k1, theta_over_pi):
    """P1, P3, B, N_tri, k3 of one isosceles bin, computed as bispec_k3_bin defines them but in float64 with
    pairwise (numpy) sums: P = V/N sum_x d_n^2 / N_k, B = V^2/N sum_x d1 d2 d3 / N_tri"""
    try:
        from scipy import fft as sfft
        fftn = lambda x: sfft.fftn(x, workers=os.cpu_count())
        ifftn = lambda x: sfft.ifftn(x, workers=os.cpu_count())
    except ImportError:
        fftn, ifftn = np.fft.fftn, np.fft.ifftn
    N = field.shape[0]
    Ntot = float(N)**3
    V = float(np.float32(L))**3
    K1 = np.float64(np.float32(k1))
    t = np.float64(np.float32(theta_over_pi))
    k3 = lambda th: np.sqrt(K1**2 + K1**2 - 2.0 * K1 * K1 * np.cos(th))
    K3min, K3max = k3((t - t * 0.025) * np.pi), k3((t + t * 0.025) * np.pi)
    m1, m3, kmag = bifft_masks(N, L, K1, K3min, K3max)
    k3avg = kmag[m3].mean()
    del kmag
    F = fftn(field.astype(np.float64)) / Ntot
    d1 = ifftn(np.where(m1, F, 0)) * Ntot
    d3 = ifftn(np.where(m3, F, 0)) * Ntot
    del F
    i1 = ifftn(m1.astype(np.complex128)) * Ntot
    i3 = ifftn(m3.astype(np.complex128)) * Ntot
    N1 = np.sum(i1 * i1).real / Ntot
    N3 = np.sum(i3 * i3).real / Ntot
    Ntri = np.sum(i1 * i1 * i3).real / Ntot
    del i1, i3
    P1 = (V / Ntot) * np.sum(d1 * d1).real / N1
    P3 = (V / Ntot) * np.sum(d3 * d3).real / N3
    B = (V**2 / Ntot) * np.sum(d1 * d1 * d3).real / Ntri
    return dict(P1=P1, P3=P3, B=B, Ntri=Ntri, k3=k3avg, Bnorm=B / np.sqrt(P1 * P1 * P3 / (K1 * K1 * k3avg)))


def selftest(a):
    """a skewed white-noise field (so B != 0) through BiFFT's isosceles code vs reference_iso"""
    rng = np.random.default_rng(a.seed)
    g = rng.standard_normal((a.n, a.n, a.n), dtype=np.float32)
    field = 10.0 * (g + np.float32(0.3) * (g * g - np.float32(1.0)))
    del g
    bifft, ba = load_bifft()
    print(f"selftest: N = {a.n}, box {a.box} cMpc/h, k1 = {a.k1[0]} h/cMpc, theta/pi = {a.theta}; "
          f"BiFFT {bifft_build(ba)}", flush=True)
    t0 = time.time()
    r = run_config(bifft, "iso", field, a.box, a.k1[0], None, None, a.num_th)
    print(f"  BiFFT: {time.time()-t0:.0f} s, theta/pi bins {np.round(r['theta_over_pi'], 3).tolist()}", flush=True)
    j = np.where(np.abs(r["theta_over_pi"] - a.theta) < 1e-3)[0]
    if not len(j):
        sys.exit(f"theta/pi = {a.theta} is not among BiFFT's bins for this k1 (k3 too small or above Nyquist)")
    j = j[0]
    t0 = time.time()
    ref = reference_iso(field, a.box, a.k1[0], a.theta)
    print(f"  reference: {time.time()-t0:.0f} s", flush=True)
    got = dict(P1=r["P1"][j], P3=r["P3"][j], B=r["B"][j], Ntri=r["Ntri"][j], k3=r["k3"][j], Bnorm=r["BS"][j])
    bad = 0
    for q in ("Ntri", "k3", "P1", "P3", "B", "Bnorm"):
        rat = got[q] / ref[q]
        ok = abs(rat - 1) < a.tol
        bad += not ok
        print(f"  {q:6s} BiFFT {got[q]: .6e}  reference {ref[q]: .6e}  ratio {rat:.6f}  {'ok' if ok else 'FAIL'}")
    if bad:
        sys.exit(f"selftest FAILED ({bad} quantities off by more than {a.tol}); the unpatched float32 build "
                 f"gives P ratio ~0.5 at 640^3")
    print("selftest passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("run")
    q.add_argument("--field", choices=["H", "He"] + list(RENDER_FIELDS), required=True)
    q.add_argument("--config", choices=["iso", "gen", "equi"], required=True)
    q.add_argument("--snaps", type=int, nargs="+", required=True,
                   help="render snapshots (H, x fields) or full snapshots (He gas-cell maps)")
    q.add_argument("--k1", type=float, nargs="+", help="k1 [h/cMpc] (iso, gen)")
    q.add_argument("--k2", type=float, nargs="+", help="k2 [h/cMpc] (gen; paired with --k1)")
    q.add_argument("--kfac", type=float, nargs="+", default=[1.1], help="K_FAC (equi)")
    q.add_argument("--out", required=True)
    q.add_argument("--h-maps", default=f"{U}/Tb_maps_prod/H_real_wf")
    q.add_argument("--cell-root", default=U)
    q.add_argument("--dir-a", default=DIR_A)
    q.add_argument("--dir-b", default=DIR_B)
    q.add_argument("--task", type=int, default=0)
    q.add_argument("--ntask", type=int, default=1)
    for r in (q, sub.add_parser("selftest")):
        r.add_argument("--box", type=float, default=BOX_MPCH, help="box side [cMpc/h]")
        r.add_argument("--num-th", type=int, default=1, help="BiFFT threads (the old jobs used 1)")
    t = sub.choices["selftest"]
    t.add_argument("--n", type=int, default=640)
    t.add_argument("--k1", type=float, nargs=1, default=[0.2])
    t.add_argument("--theta", type=float, default=0.3333333, help="theta/pi of the bin to check")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--tol", type=float, default=1e-3)
    a = p.parse_args()
    (run if a.mode == "run" else selftest)(a)


if __name__ == "__main__":
    main()
