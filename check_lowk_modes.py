"""
check_lowk_modes.py
===================
What makes the feature at k = 0.03-0.04 h/cMpc in every power spectrum? In the 338.3 cMpc/h box that range holds
only two shells of Fourier modes: |n|^2 = 3 (k = 0.0322, the 4 body diagonals) and |n|^2 = 4 (k = 0.0371, the 3
axis modes), so each point there is the mean of 3-4 modes, the same modes in every snapshot and every field.

For each map: the power of every independent mode with |n|^2 <= NMAX, divided by a power law in k fitted to the
shell means over |n|^2 = 5 .. 60 (k = 0.04-0.14 h/cMpc). For a Gaussian field the ratio of one mode is
exponentially distributed (mean 1, so 0.1-3 is normal); a shell mean of m modes scatters by ~1/sqrt(m).
  - the same modes high (or low) in the density field and in the delta T_b / x maps, at every z:
    the large-scale structure of this realization (sample variance), not the pipeline;
  - only the axis modes (|n|^2 = 1, 4) off, or only in the derived maps: an artifact to look into.

Usage
-----
  python check_lowk_modes.py --map <file>[:dataset] [--map ...] [--box 338300] [--nmax 8]
e.g. the density render, the 21 cm map and the 3.46 cm map at one snapshot:
  python check_lowk_modes.py --map .../ren_640/Density/Density_406.hdf5:Density \\
      --map /orcd/data/mvogelsb/004/chcho/Tb_maps_prod/H_real_wf/RIN_406.hdf5:rin \\
      --map /orcd/data/mvogelsb/004/chcho/Tb_maps_prod/He/RIN_406.hdf5:rin
"""
import argparse

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


def load(spec):
    path, _, dset = spec.partition(":")
    with h5py.File(path, "r") as f:
        if not dset:
            names = [k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3]
            if len(names) != 1:
                raise SystemExit(f"{path}: give the dataset as {path}:<name> (3-D datasets: {names})")
            dset = names[0]
        return f[dset][()].astype(np.float64), f"{path.split('/')[-1]}:{dset}"


def modes(m, L, nmax, fit=(5, 60)):
    N = m.shape[0]
    F = np.fft.rfftn(m - m.mean())
    P = np.abs(F)**2 * L**3 / float(N)**6
    kf = 2 * np.pi / L
    r = int(np.ceil(np.sqrt(fit[1]))) + 1
    rows = []                                     # (n^2, (nx, ny, nz), P) for the independent modes (rfft half-space, nz >= 0)
    for nx in range(-r, r + 1):
        for ny in range(-r, r + 1):
            for nz in range(0, r + 1):
                n2 = nx * nx + ny * ny + nz * nz
                if n2 == 0 or n2 > fit[1]:
                    continue
                if nz == 0 and (nx < 0 or (nx == 0 and ny < 0)):
                    continue                      # its conjugate is counted
                rows.append((n2, (nx, ny, nz), P[nx % N, ny % N, nz]))
    rows.sort(key=lambda x: (x[0], x[1]))         # by shell, the same order for every map
    n2s = np.array([x[0] for x in rows])
    Ps = np.array([x[2] for x in rows])
    sh = np.unique(n2s[(n2s >= fit[0]) & (n2s <= fit[1])])
    shell_mean = np.array([Ps[n2s == s].mean() for s in sh])
    c = np.polyfit(np.log(kf * np.sqrt(sh)), np.log(shell_mean), 1)
    model = lambda n2: np.exp(np.polyval(c, np.log(kf * np.sqrt(n2))))
    return [(n2, v, p / model(n2)) for n2, v, p in rows if n2 <= nmax], c[0], kf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", action="append", required=True, help="file[:dataset]; repeat")
    ap.add_argument("--box", type=float, default=338300.0, help="BoxSize [ckpc/h]")
    ap.add_argument("--nmax", type=int, default=8, help="largest |n|^2 listed")
    a = ap.parse_args()
    L = a.box / 1e3                               # cMpc/h -> k in h/cMpc
    res = []
    for spec in a.map:
        m, name = load(spec)
        rows, slope, kf = modes(m, L, a.nmax)
        res.append((name, rows, slope))
        print(f"{name}: N = {m.shape[0]}, power-law slope of P(k) over k = {kf * np.sqrt(5):.3f}-{kf * np.sqrt(60):.3f}: {slope:+.2f}")
    print(f"\nP(mode) / power-law fit, per independent mode (|n|^2 <= {a.nmax}); shell mean in [ ]")
    print(f"{'|n|^2':>5} {'k':>7} {'mode':>12}  " + "  ".join(f"{r[0][:22]:>22}" for r in res))
    rows0 = res[0][1]
    for i, (n2, v, _) in enumerate(rows0):
        print(f"{n2:5d} {kf * np.sqrt(n2):7.4f} {str(v):>12}  " + "  ".join(f"{r[1][i][2]:22.2f}" for r in res))
        last = i == len(rows0) - 1 or rows0[i + 1][0] != n2
        if last:
            sel = [j for j, x in enumerate(rows0) if x[0] == n2]
            print(f"{'':5} {'':7} {'[shell mean]':>12}  " + "  ".join(
                f"{np.mean([r[1][j][2] for j in sel]):22.2f}" for r in res) + f"   ({len(sel)} modes, ~{1 / np.sqrt(len(sel)):.0%} scatter)")
    if len(res) > 1:
        print("\ncorrelation of log(P/fit) across the listed modes, map 1 vs each other map:")
        x = np.log([q[2] for q in rows0])
        for r in res[1:]:
            y = np.log([q[2] for q in r[1]])
            print(f"  {r[0]}: r = {np.corrcoef(x, y)[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
