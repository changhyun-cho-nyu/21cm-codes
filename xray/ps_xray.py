"""
ps_xray.py
==========
21 cm power spectra of the two X-ray test runs (withoutXrays, withXrays), computed from the delta T_b maps
(Tb_21cm/RIN_N.hdf5 of run_21cm_xray.sh), render by render:
  Delta^2(k) = k^3 P(k) / (2 pi^2) [mK^2] of each run, P = |FFT(dT_b - <dT_b>)|^2 V / N^6 (the convention of
  ../check_ps.py, checked against PS_21cm_mpi.c), in log bins of k from k_f to k_Nyquist [h/cMpc],
  the ratio Delta^2_withX / Delta^2_noX, and the cross-correlation r(k) = P_x / sqrt(P_noX P_withX).
The two runs have the same ICs and the same render times (checked here: the map redshifts must agree).

Outputs (in --out, default data/):
  ps_xray.npz                  all renders: z, k, D2_noX, D2_X, r, mean dT_b of each run, independent modes per bin
  ps_xray_table.csv            the same, one row per render and k bin
  ps_xray_k.png                Delta^2(k), ratio and r(k) at selected redshifts
  ps_xray_z.png                Delta^2(z) and ratio at three k
Run from xray/:  python ps_xray.py          (BASE=<dir> overrides /orcd/data/mvogelsb/004/chcho/xray)
"""
import argparse
import glob
import os

import h5py
import numpy as np

RUNS = ("withoutXrays", "withXrays")
LABELS = {"withoutXrays": "no X-rays", "withXrays": "with X-rays"}
COLORS = {"withoutXrays": "#2a78d6", "withXrays": "#eb6834"}     # categorical slots 1, 2 (validated pair)
INK, INK2, GRID = "#1f1f1e", "#6b6b68", "#e6e6e3"


def box_size(base, run):
    """BoxSize [ckpc/h] and N from the run's first assembled render"""
    f = sorted(glob.glob(f"{base}/{run}/ren_128/All/All_*.hdf5"))[0]
    with h5py.File(f, "r") as h:
        return float(h["Header"].attrs["BoxSize"]), int(h["Header"].attrs["NumPixels"])


def renders(base, run):
    fs = glob.glob(f"{base}/{run}/Tb_21cm/RIN_*.hdf5")
    return sorted(int(os.path.basename(f)[4:-5]) for f in fs)


def spectra(m1, m2, L, edges):
    """binned P1, P2, cross [mK^2 (cMpc/h)^3], mode-weighted k and mode counts; maps in mK, L in cMpc/h"""
    N = m1.shape[0]
    F1 = np.fft.rfftn(m1 - m1.mean())
    F2 = np.fft.rfftn(m2 - m2.mean())
    kf = 2.0 * np.pi / L
    ii = np.fft.fftfreq(N, 1.0 / N)
    k = np.sqrt((ii[:, None, None] * kf)**2 + (ii[None, :, None] * kf)**2
                + (np.arange(N // 2 + 1)[None, None, :] * kf)**2)
    b = np.digitize(k, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    bb, kk = b[ok], k[ok]
    nb = len(edges) - 1
    cnt = np.bincount(bb, minlength=nb).astype(float)              # rfft entries (averaging weights)
    kz = np.broadcast_to(np.arange(N // 2 + 1)[None, None, :], k.shape)
    half = np.where((kz == 0) | (kz == N // 2), 0.5, 1.0)          # those planes hold +-k twice
    nind = np.bincount(bb, half[ok], minlength=nb)                 # independent complex modes
    norm = L**3 / float(N)**6
    P1 = np.bincount(bb, (np.abs(F1)**2)[ok], minlength=nb) / np.maximum(cnt, 1) * norm
    P2 = np.bincount(bb, (np.abs(F2)**2)[ok], minlength=nb) / np.maximum(cnt, 1) * norm
    X = np.bincount(bb, (F1 * np.conj(F2)).real[ok], minlength=nb) / np.maximum(cnt, 1) * norm
    kc = np.bincount(bb, kk, minlength=nb) / np.maximum(cnt, 1)
    return P1, P2, X, kc, nind


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=os.environ.get("BASE", "/orcd/data/mvogelsb/004/chcho/xray"))
    p.add_argument("--out", default="data")
    p.add_argument("--nbins", type=int, default=10)
    p.add_argument("--zk", type=float, nargs="+", default=[15, 13, 12, 11, 10, 8],
                   help="redshifts of the Delta^2(k) panels and the table")
    p.add_argument("--kz", type=float, nargs="+", default=[0.3, 1.0, 3.0],
                   help="k [h/cMpc] of the Delta^2(z) panels (nearest bin)")
    a = p.parse_args()

    Lk, N = box_size(a.base, RUNS[0])
    Lk2, N2 = box_size(a.base, RUNS[1])
    if (Lk, N) != (Lk2, N2):
        raise SystemExit(f"the runs differ in BoxSize/N: {Lk}/{N} vs {Lk2}/{N2}")
    L = Lk / 1e3                                                   # cMpc/h
    kf = 2.0 * np.pi / L
    r0, r1 = renders(a.base, RUNS[0]), renders(a.base, RUNS[1])
    common = sorted(set(r0) & set(r1))
    if not common:
        raise SystemExit("no render has a map in both runs")
    with h5py.File(f"{a.base}/{RUNS[0]}/Tb_21cm/RIN_{common[0]}.hdf5", "r") as h:
        Nm = h["rin"].shape[0]
    if Nm != N:
        print(f"maps are {Nm}^3 (renders {N}^3): k bins from the maps")
        N = Nm
    edges = np.geomspace(kf, kf * N / 2, a.nbins + 1)
    print(f"BoxSize {L:.3f} cMpc/h, N = {N}, k_f = {kf:.4f}, k_Nyq = {kf * N / 2:.3f} h/cMpc; "
          f"maps: {len(r0)} / {len(r1)}, {len(common)} in both")
    if len(common) < max(len(r0), len(r1)):
        print(f"WARNING: renders in only one run: {sorted(set(r0) ^ set(r1))[:20]}")

    nr = len(common)
    z = np.empty(nr); D2 = np.empty((2, nr, a.nbins)); R = np.empty((nr, a.nbins)); mu = np.empty((2, nr))
    for j, s in enumerate(common):
        maps, zz = [], []
        for run in RUNS:
            with h5py.File(f"{a.base}/{run}/Tb_21cm/RIN_{s}.hdf5", "r") as h:
                maps.append(h["rin"][()].astype(np.float64)); zz.append(float(h["rin"].attrs["Redshift"]))
        if abs(zz[0] - zz[1]) > 1e-5:
            raise SystemExit(f"render {s}: z differs between the runs ({zz[0]} vs {zz[1]})")
        z[j] = zz[0]
        P1, P2, X, kc, cnt = spectra(maps[0], maps[1], L, edges)
        D2[0, j], D2[1, j] = kc**3 * P1 / (2 * np.pi**2), kc**3 * P2 / (2 * np.pi**2)
        with np.errstate(invalid="ignore", divide="ignore"):
            R[j] = X / np.sqrt(P1 * P2)
        mu[0, j], mu[1, j] = maps[0].mean(), maps[1].mean()

    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "ps_xray.npz"), render=np.array(common), z=z, k=kc, modes=cnt,
             D2_noX=D2[0], D2_X=D2[1], r=R, mean_noX=mu[0], mean_X=mu[1], box_cMpc_h=L, N=N)
    with open(os.path.join(a.out, "ps_xray_table.csv"), "w") as f:
        f.write("render,z,k_h_cMpc,independent_modes,D2_noX_mK2,D2_withX_mK2,ratio_withX_noX,r,mean_noX_mK,mean_withX_mK\n")
        for j, s in enumerate(common):
            for i in range(a.nbins):
                f.write(f"{s},{z[j]:.5f},{kc[i]:.5f},{int(cnt[i])},{D2[0, j, i]:.6e},{D2[1, j, i]:.6e},"
                        f"{D2[1, j, i] / D2[0, j, i]:.6e},{R[j, i]:.6f},{mu[0, j]:.6e},{mu[1, j]:.6e}\n")

    # ---- table at the selected redshifts
    ik = [int(np.argmin(abs(np.log(kc / kq)))) for kq in a.kz]
    print("\nDelta^2 [mK^2] no X-rays / with X-rays (ratio), at k = " + ", ".join(f"{kc[i]:.2f}" for i in ik) + " h/cMpc;"
          " r(k) min over k; mean dT_b [mK]")
    jz = [int(np.argmin(abs(z - zq))) for zq in a.zk]
    for zq in sorted(set(np.round(np.r_[[29, 25, 20, 17], a.zk, [9, 7, 6, 5]], 1)), reverse=True):
        j = int(np.argmin(abs(z - zq)))
        cells = "  ".join(f"{D2[0, j, i]:9.3e} / {D2[1, j, i]:9.3e} ({D2[1, j, i] / D2[0, j, i]:6.3f})" for i in ik)
        print(f"  z = {z[j]:6.3f} (render {common[j]:3d}):  {cells}   r_min {np.nanmin(R[j]):6.3f}   "
              f"mean {mu[0, j]:+7.3f} / {mu[1, j]:+7.3f}")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "legend.frameon": False})

    nc = len(jz)
    fig, ax = plt.subplots(3, nc, figsize=(2.6 * nc, 7.2), sharex=True, sharey="row", squeeze=False,
                           gridspec_kw=dict(height_ratios=[2.2, 1.2, 1.0]))
    for c, j in enumerate(jz):
        for q, run in enumerate(RUNS):
            ax[0, c].loglog(kc, D2[q, j], color=COLORS[run], lw=2, marker="o", ms=4, label=LABELS[run])
        ax[0, c].set_title(f"z = {z[j]:.2f}", color=INK, fontsize=10)
        ax[1, c].semilogx(kc, D2[1, j] / D2[0, j], color=INK, lw=2, marker="o", ms=4)
        ax[1, c].axhline(1, color=INK2, lw=0.8)
        ax[1, c].set_yscale("log")
        ax[2, c].semilogx(kc, R[j], color=INK, lw=2, marker="o", ms=4)
        ax[2, c].set_xlabel(r"$k$ [$h$ cMpc$^{-1}$]")
    kt = [t for t in (0.2, 0.5, 1, 2, 5, 10) if edges[0] <= t <= edges[-1]]
    for axx in ax[2]:
        axx.xaxis.set_major_locator(FixedLocator(kt))
        axx.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axx.xaxis.set_minor_formatter(NullFormatter())
    ax[0, 0].set_ylabel(r"$\Delta^2_{21}(k)$ [mK$^2$]")
    ax[1, 0].set_ylabel("with / without X-rays")
    ax[2, 0].set_ylabel(r"$r(k)$")
    ax[2, 0].set_ylim(min(-0.05, np.nanmin(R[jz]) - 0.05), 1.05)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2)
    fig.suptitle("21 cm power spectrum of the X-ray test runs (50 cMpc, 128³)", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(a.out, "ps_xray_k.png"), dpi=200)

    fig, ax = plt.subplots(2, len(ik), figsize=(3.2 * len(ik), 5.4), sharex=True, sharey="row", squeeze=False,
                           gridspec_kw=dict(height_ratios=[2, 1.1]))
    for c, i in enumerate(ik):
        for q, run in enumerate(RUNS):
            ax[0, c].semilogy(z, D2[q, :, i], color=COLORS[run], lw=2, label=LABELS[run])
        ax[0, c].set_title(f"k = {kc[i]:.2f} $h$ cMpc$^{{-1}}$", color=INK, fontsize=10)
        ax[1, c].semilogy(z, D2[1, :, i] / D2[0, :, i], color=INK, lw=2)
        ax[1, c].axhline(1, color=INK2, lw=0.8)
        ax[1, c].set_xlabel(r"$z$")
    for c in range(len(ik)):
        ax[1, c].invert_xaxis() if c == 0 else None
    ax[0, 0].set_ylabel(r"$\Delta^2_{21}$ [mK$^2$]")
    ax[1, 0].set_ylabel("with / without X-rays")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2)
    fig.suptitle("21 cm power spectrum vs redshift, X-ray test runs", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(a.out, "ps_xray_z.png"), dpi=200)
    print(f"\nwrote {a.out}/ps_xray.npz, ps_xray_table.csv, ps_xray_k.png, ps_xray_z.png")


if __name__ == "__main__":
    main()
