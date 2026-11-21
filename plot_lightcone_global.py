"""
plot_lightcone_global.py
========================
Figures of the 21 cm light cone (2026-11-20):
  lightcone_global.png : per slice along the cone, from lightcone_global.py's CSV:
      (a) <delta T_b> and its pixel rms; (b) T_gamma, T_K (mass-weighted; H I-weighted harmonic), T_S,eff (the
      H I-weighted spin factor as one T_S) and the pixel median T_S; (c) x_HII (volume, mass) and x_HeIII (volume),
      with the box values of the renders (ion_history CSV) as a check; (d) H I-weighted x_alpha and x_c.
  lightcone_ps_evolution.png : from make_lightcone_ps.py --tile (cone only):
      (a) Delta^2 at fixed k against the band centre z, each depth only where k >= k_use = 6 x 2 pi / depth;
      (b) Delta^2(k) of every few bands of one depth.
  lightcone_ps_evolution.csv : the points of (a).
Marks: z = 4.75 (run restart, render 428), z = 3.35 (self-shielding step, renders 617/618), and z > 10 shaded
(21 cm resolution-dependent there: the cone delta T_b is an upper limit, JALPHA_NOTES 4.7 and 6.1 item 11).

Usage
-----
  python plot_lightcone_global.py --global-csv DIR/lightcone_global_H.csv --ps DIR/ps_tiles/lightcone_ps_H.hdf5 \
      --ion /orcd/data/mvogelsb/004/chcho/ion_history/ionization_history.csv --outdir DIR/figs
"""
import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

KS = (0.15, 0.3, 0.5, 1.0, 2.0)            # h/cMpc


def marks(ax, zmax):
    for z0, lab in ((4.75, "restart"), (3.35, "self-shielding")):
        ax.axvline(z0, color="0.6", lw=0.7, ls=":")
    ax.axvspan(10.0, zmax, color="0.92", zorder=0)


def global_fig(df, ion, out):
    df = df.sort_values("z")
    z = df.z.values
    zmax = z.max()
    fig, axs = plt.subplots(4, 1, figsize=(7, 11), sharex=True)
    ax = axs[0]
    ax.fill_between(z, df.dTb_mean - df.dTb_rms, df.dTb_mean + df.dTb_rms, color="C0", alpha=0.2, lw=0,
                    label="± pixel rms")
    ax.plot(z, df.dTb_mean, "C0", lw=1, label=r"$\langle\delta T_b\rangle$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"$\delta T_b$ [mK]")
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(z, df.T_gamma, "k", lw=1, label=r"$T_\gamma$")
    ax.plot(z, df.TK_M, "C3", lw=1, label=r"$T_K$, mass-weighted")
    ax.plot(z, df.TK_HI_harm, "C1", lw=1, label=r"$T_K$, H I harmonic")
    ax.plot(z, df.TS_eff, "C0", lw=1, label=r"$T_{S,\rm eff}=T_\gamma/(1-\langle 1-T_\gamma/T_S\rangle_{\rm HI})$")
    ax.plot(z, df.TS_med, "C2", lw=1, ls="--", label=r"$T_S$, pixel median")
    ax.set_yscale("log")
    ax.set_ylabel("T [K]")
    ax.legend(fontsize=7)
    ax = axs[2]
    ax.plot(z, df.xHII_V, "C0", lw=1, label=r"$x_{\rm HII}$, volume")
    ax.plot(z, df.xHII_M, "C0", lw=1, ls="--", label=r"$x_{\rm HII}$, mass")
    ax.plot(z, df.xHeIII_V, "C3", lw=1, label=r"$x_{\rm HeIII}$, volume")
    if ion is not None:
        ax.plot(ion.z, ion.xHII_V, "k.", ms=2, label="box (renders), volume")
        ax.plot(ion.z, ion.xHeIII_V, "k.", ms=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.5)
    ax.set_ylabel("ionized fraction")
    ax.legend(fontsize=7)
    ax = axs[3]
    ax.plot(z, df.xa_HI, "C0", lw=1, label=r"$x_\alpha$ (with $\tilde S_\alpha$), H I-weighted")
    ax.plot(z, df.xc_HI, "C3", lw=1, label=r"$x_c$, H I-weighted")
    ax.set_yscale("log")
    ax.set_ylabel("coupling")
    ax.set_xlabel("z (light-cone slice)")
    ax.legend(fontsize=7)
    for ax in axs:
        marks(ax, zmax)
    axs[0].set_xlim(z.min(), zmax)
    axs[0].set_title("Lumina 21 cm light cone, per slice (shaded: z > 10, resolution-limited)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def read_tiles(path):
    rows = []
    with h5py.File(path, "r") as f:
        for dn in f:
            for zn in f[dn]:
                g = f[dn][zn]
                rows.append(dict(depth=float(g.attrs["depth"]), z=float(g.attrs["z_centre"]), k=g["k"][()],
                                 D2=g["D2"][()], kuse=float(g.attrs["k_use"]), zr=tuple(g.attrs["z_range"])))
    return sorted(rows, key=lambda r: (r["depth"], r["z"]))


def ps_fig(tiles, out_png, out_csv, depth_curves):
    pts = []
    for t in tiles:
        k, D2 = t["k"], t["D2"]
        ok = (D2 > 0) & np.isfinite(D2)
        for kk in KS:
            if kk < t["kuse"] or kk < k[ok].min() or kk > k[ok].max():
                continue
            pts.append(dict(depth=t["depth"], z=t["z"], z_lo=min(t["zr"]), z_hi=max(t["zr"]), k=kk,
                            D2=float(10**np.interp(np.log10(kk), np.log10(k[ok]), np.log10(D2[ok])))))
    pts = pd.DataFrame(pts)
    pts.to_csv(out_csv, index=False, float_format="%.6g")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axs[0]
    for i, kk in enumerate(KS):
        for depth, ls in zip(sorted(pts.depth.unique()), ("-", "--", ":")):
            p = pts[(pts.k == kk) & (pts.depth == depth)].sort_values("z")
            if len(p):
                ax.plot(p.z, p.D2, ls, color=f"C{i}", marker="o", ms=3,
                        label=f"k = {kk} h/cMpc, depth {depth:g}" if True else None)
    ax.set_yscale("log")
    ax.set_xlabel("band centre z")
    ax.set_ylabel(r"$\Delta^2_{21}$ [mK$^2$]")
    marks(ax, pts.z.max() if len(pts) else 30)
    ax.legend(fontsize=6, ncol=2)
    ax = axs[1]
    tl = [t for t in tiles if t["depth"] == depth_curves]
    step = max(1, len(tl) // 8)
    cmap = plt.get_cmap("viridis")
    for j, t in enumerate(tl[::step]):
        m = (t["k"] >= t["kuse"]) & (t["D2"] > 0)
        ax.loglog(t["k"][m], t["D2"][m], color=cmap(j / max(1, len(tl[::step]) - 1)),
                  label=f"z = {t['z']:.2f} ({min(t['zr']):.2f}-{max(t['zr']):.2f})")
    ax.set_xlabel("k [h/cMpc]")
    ax.set_ylabel(r"$\Delta^2_{21}$ [mK$^2$]")
    ax.set_title(f"bands of depth {depth_curves:g} cMpc/h, k >= 6 x 2 pi / depth", fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--global-csv", help="lightcone_global.py output")
    ap.add_argument("--ps", help="make_lightcone_ps.py --tile output (lightcone_ps_H.hdf5)")
    ap.add_argument("--ion", help="ion_history/ionization_history.csv (box check of the fractions)")
    ap.add_argument("--depth-curves", type=float, default=150.0, help="depth of the bands in the Delta^2(k) panel")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.global_csv:
        df = pd.read_csv(a.global_csv, comment="#")
        ion = pd.read_csv(a.ion, comment="#") if a.ion else None
        global_fig(df, ion, os.path.join(a.outdir, "lightcone_global.png"))
        d = df.dTb_mean - df.dTb_mean_file
        print(f"global: {len(df)} slices, z {df.z.min():.3f} - {df.z.max():.3f}; max |<dT_b> - cone file mean| "
              f"{np.abs(d).max():.2g} mK")
    if a.ps:
        ps_fig(read_tiles(a.ps), os.path.join(a.outdir, "lightcone_ps_evolution.png"),
               os.path.join(a.outdir, "lightcone_ps_evolution.csv"), a.depth_curves)
    print(f"figures in {a.outdir}")


if __name__ == "__main__":
    main()
