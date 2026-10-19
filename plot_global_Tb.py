"""
plot_global_Tb.py
=================
Global 21 cm signal <dT_b>(z) of the production maps against the no-heating (adiabatic) reference, read from the
PS_21cm_mpi.c logs: each snapshot prints "DIAGNOSTIC: rin variable audit   z = ..." and then the "rin (T_b)" line
with the mean over all cells of the map (mK), which is the volume-weighted global signal.

Usage
-----
  python plot_global_Tb.py --log logs/PS_21cm_25884455.out --label "Lumina (fiducial)" \
                           --log logs/PS_21cm_25971016.out --label "no heating (T_ref, x_HII < 1e-3)" \
                           --out global_Tb.png
Writes the figure, a CSV per log (same name as the figure, _<n>.csv) and prints a table at a few redshifts.
"""
import argparse
import re

import numpy as np


def read_log(path):
    z_cur, rows = None, {}
    with open(path, errors="replace") as f:
        for line in f:
            m = re.search(r"DIAGNOSTIC: rin variable audit\s+z = ([0-9.eE+-]+)", line)
            if m:
                z_cur = float(m.group(1)); continue
            if z_cur is not None and line.strip().startswith("rin (T_b)"):
                rows[z_cur] = float(line.split()[2])                   # mean over all cells [mK]
                # (the line's min/max columns are per-cell dT_b x V_cell, not normalised: not used)
                z_cur = None
    z = np.array(sorted(rows, reverse=True))
    return z, np.array([rows[k] for k in z])[:, None]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", action="append", required=True)
    p.add_argument("--label", action="append", required=True)
    p.add_argument("--out", default="global_Tb.png")
    a = p.parse_args()
    if len(a.log) != len(a.label):
        raise SystemExit("one --label per --log")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    for i, (lg, lb) in enumerate(zip(a.log, a.label)):
        z, v = read_log(lg)
        if len(z) == 0:
            raise SystemExit(f"no 'rin (T_b)' entries in {lg}")
        data.append((lb, z, v))
        np.savetxt(a.out.rsplit(".", 1)[0] + f"_{i}.csv", np.column_stack([z, v[:, 0]]), delimiter=",",
                   header="z,mean_dTb_mK", comments="")
        ax.plot(z, v[:, 0], lw=2.2, label=f"{lb} ({len(z)} snapshots)", ls="-" if i == 0 else "--")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\langle \delta T_b^{\rm H} \rangle$ [mK]")
    ax.invert_xaxis()
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")
    zs = [29, 25, 20, 17, 15, 13, 12, 11, 10, 9, 8, 7, 6]
    print("  z    " + "  ".join(f"{lb[:28]:>28}" for lb, _, _ in data))
    for zq in zs:
        cells = []
        for lb, z, v in data:
            j = int(np.argmin(abs(z - zq)))
            if abs(z[j] - zq) < 0.5:
                cells.append(f"{v[j, 0]:+10.3f} mK (z={z[j]:.2f})")
            else:
                cells.append(f"{'-':>28}")
        print(f"  {zq:<4} " + "  ".join(f"{c:>28}" for c in cells))


if __name__ == "__main__":
    main()
