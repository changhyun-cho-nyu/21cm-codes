"""
make_Z_young_history.py
=======================
Write Z_history.txt for compute_Jalpha.c / compute_Jalpha_He.c using the
metallicity of YOUNG stars: the initial-mass-weighted birth metallicity of star
particles formed within the last WINDOW_MYR (default 20 Myr) before each time.
This equals the SFR-weighted metallicity of the stars forming at that time,
which is what sets the BPASS Lyman-band output (most photons are emitted
within ~10-20 Myr of formation).

One late snapshot is enough: GFM_Metallicity is fixed at birth and
GFM_StellarFormationTime / GFM_InitialMass are kept, so the particles of the
latest snapshot carry the full birth-metallicity history of the box.

Usage
-----
    python make_Z_young_history.py <snapdir> <snapnum> <out.txt> [window_Myr] [check_z ...]

    snapdir : directory with snap_NNN.hdf5 and PartType4/PartType4_NNN.hdf5
    snapnum : latest full snapshot (it must postdate every redshift you need)
    check_z : optional redshifts at which to print Z_young (to compare with a
              direct measurement from an earlier snapshot)

Output: two columns "redshift  Z" (ascending z), the format read by
load_Z_history() in the C codes. Z is not clamped here; the C codes clamp it to
the BPASS grid. Time bins with no young stars in the window are skipped.
"""
import sys
import h5py
import numpy as np

# LUMINA cosmology (as in compute_Jalpha*.c)
H0_S  = 0.6766 * 100.0 * 3.241e-20          # s^-1
OM, OL = 0.30964144, 0.6902672
MYR_S = 3.15576e13
BIN_MYR = 1.0                               # formation-time histogram resolution
CHUNK = 50_000_000                          # particles read per pass

_pref = 2.0 / (3.0 * H0_S * np.sqrt(OL)) / MYR_S      # Myr
_A = np.sqrt(OL / OM)

def t_of_a(a):
    """Cosmic time [Myr] at scale factor a (flat LCDM, radiation neglected)."""
    return _pref * np.arcsinh(_A * np.asarray(a, float)**1.5)

def a_of_t(t):
    return (np.sinh(np.asarray(t, float) / _pref) / _A)**(2.0 / 3.0)

def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    snapdir, snap, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    window = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
    check_z = [float(x) for x in sys.argv[5:]]

    with h5py.File(f"{snapdir}/snap_{snap:03d}.hdf5", "r") as f:
        a_snap = float(f["Header"].attrs["Time"])
    t_snap = t_of_a(a_snap)

    edges = np.arange(0.0, t_snap + BIN_MYR, BIN_MYR)
    M = np.zeros(len(edges) - 1)            # initial mass formed per bin
    MZ = np.zeros(len(edges) - 1)           # initial mass x birth Z per bin
    n_used = n_wind = 0
    with h5py.File(f"{snapdir}/PartType4/PartType4_{snap:03d}.hdf5", "r") as f:
        n = f["GFM_StellarFormationTime"].shape[0]
        mkey = "GFM_InitialMass" if "GFM_InitialMass" in f else "Masses"
        if mkey != "GFM_InitialMass":
            print("WARNING: GFM_InitialMass missing, weighting by current Masses")
        for i in range(0, n, CHUNK):
            af = f["GFM_StellarFormationTime"][i:i + CHUNK].astype(np.float64)
            ok = af > 0                                    # drop wind particles
            n_wind += int((~ok).sum()); n_used += int(ok.sum())
            tf = t_of_a(af[ok])
            m = f[mkey][i:i + CHUNK].astype(np.float64)[ok]
            Z = f["GFM_Metallicity"][i:i + CHUNK].astype(np.float64)[ok]
            M += np.histogram(tf, edges, weights=m)[0]
            MZ += np.histogram(tf, edges, weights=m * Z)[0]

    # running sums over the last `window` Myr, evaluated at each bin's upper edge
    nwin = max(1, int(round(window / BIN_MYR)))
    cM, cMZ = np.r_[0.0, np.cumsum(M)], np.r_[0.0, np.cumsum(MZ)]
    j = np.arange(1, len(edges))
    lo = np.maximum(j - nwin, 0)
    Mw, MZw = cM[j] - cM[lo], cMZ[j] - cMZ[lo]
    t_out = edges[1:]
    good = Mw > 0
    z_out = 1.0 / a_of_t(t_out[good]) - 1.0
    Z_out = MZw[good] / Mw[good]
    o = np.argsort(z_out)
    np.savetxt(out, np.c_[z_out[o], Z_out[o]], fmt="%.8e",
               header=(f"redshift  Z_young  (initial-mass-weighted birth Z of stars formed in the "
                       f"last {window:g} Myr; from snap {snap:03d}, {n_used} star particles, "
                       f"{n_wind} wind particles skipped)"))
    print(f"snap {snap:03d}: z = {1/a_snap - 1:.4f}, {n_used} stars used, {n_wind} wind particles skipped")
    print(f"wrote {out}: {good.sum()} rows, z = {z_out.min():.3f} .. {z_out.max():.3f}, window = {window:g} Myr")
    for zc in check_z:
        print(f"  Z_young(z = {zc:.4f}) = {10**np.interp(zc, z_out[o], np.log10(Z_out[o])):.4e}")

if __name__ == "__main__":
    main()
