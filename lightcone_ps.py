"""
lightcone_ps.py
===============
Spherically averaged power spectrum Delta^2(k) = k^3 P(k) / (2 pi^2) of a Lumina light-cone field (e.g. delta T_b)
in sub-bands of comoving depth `depth` around z_centre (a light cone is not periodic and evolves along the line
of sight, so P(k) is measured band by band).

Geometry (AREPO src/reionization_lightcone, rlc_320 header): fields are (NumPixels, NumPixels, NumDepth), the
last axis the line of sight; pixels are angular (OpeningAngle = BoxSize / chi(TargetRedshift), 0.0628 rad, so the
cone is one box wide only at z = 5 and wider above, where the transverse field holds repeated copies of the box);
slices are spherical shells between the edges `Distances` (ckpc/h), 1.03-1.57 cMpc/h thick, not uniform (6%
jumps between neighbours).

Method, per band (the slices whose centres lie within chi_centre +- depth/2):
  - each slice's mean is removed (slice_mean=True; else the band's mean), which also removes the evolution of the
    mean signal across the band;
  - flat sky: transverse comoving pixel = (OpeningAngle / NumPixels) x chi_centre;
  - a Tukey taper across the sky (taper_perp = Tukey alpha, the fraction of each axis inside the cosine edges;
    0.2 tapers the outer 10% on each side), because the field is not periodic across the sky;
  - a Blackman-Harris window along the line of sight (window_los), against the leakage from the band edges;
  - transverse FFT, then along the line of sight an exact Fourier sum over the slices with their own centres and
    widths (no interpolation onto a uniform grid), at k_par = 2 pi m / L_par, m = 0 .. n/2;
  - P = |FT|^2 / V / <w^2>, binned in log k (the normalisation of PS_21cm_mpi.c and tb_power for a periodic cube).
Modes on the k_par = 0 plane are counted once per (k_x, k_y), as rfftn does. The 6-46% of the transverse field
that repeats the box above z = 5 makes the mode counts (and so naive error bars) overstate the information.

Usage
-----
  import h5py
  from lightcone_ps import lightcone_ps
  f = h5py.File(".../rlc_320/All.hdf5", "r")            # or the per-field files; delta T_b computed from them
  chi_edges = f["Distances"][()] / 1e3                   # cMpc/h
  z_edges = f["Redshifts"][()]
  theta = f["Header"].attrs["OpeningAngle"]
  k, D2, nmodes, info = lightcone_ps(tb, chi_edges, z_edges, z_centre=7.0, depth=150.0, opening_angle=theta)
Units: k [h/cMpc], Delta^2 [field units^2, mK^2 for delta T_b in mK].

Tests (2026-10-26, Gaussian fields with P ~ k^-2 exp(-(k/k_c)^2)):
  - a periodic cube posing as a cone (uniform slices, no windows, band mean) gives the cube PS to 1.4e-15, slices
    in either order;
  - an angular cone through a periodic 200 cMpc/h box built as RT_lightcone_rendering.cc does (replicas without
    shifts, cone 1.11 boxes wide as at z = 7, slices of the pixel width x (1 +- 5%), depth 180 cMpc/h, 3
    realisations, divided by the trilinear-sampling transfer): with the defaults 0.96-1.00 for
    6 x 2 pi / depth < k < 0.75 k_Nyquist (0.99-1.00 at k = 1-2.6 for k_c = 2), 0.95 at 4.6 x 2 pi / depth,
    0.84 +- 0.05 at 3.6 x; without the taper 0.81-0.84 at 4.6-5.8 x 2 pi / depth. For a very steep spectrum
    (k_c = 1, a 10^5 range in P) the sidelobes leak low-k power to k > 1.6 (+2% at 1.6, +10% at 2, +34% at 2.6).
  Use k >~ 6 x 2 pi / depth (0.25 h/cMpc for depth = 150, 0.11 for 338.3).
"""
import numpy as np
from scipy.signal.windows import tukey

BH = (0.35875, 0.48829, 0.14128, 0.01168)               # Blackman-Harris coefficients


def blackman_harris(t):
    """Blackman-Harris window at t = x / L in [0, 1] (continuous, for slices of unequal width)."""
    return BH[0] - BH[1] * np.cos(2 * np.pi * t) + BH[2] * np.cos(4 * np.pi * t) - BH[3] * np.cos(6 * np.pi * t)


def lightcone_ps(tb, chi_edges, z_edges, z_centre, depth, opening_angle, window_los="blackmanharris",
                 taper_perp=0.2, slice_mean=True, bins_per_decade=10):
    """
    tb            : (nx, ny, nz) field, the last axis the line of sight
    chi_edges     : (nz + 1,) comoving distances of the slice edges [cMpc/h] (Distances / 1e3), in either order
    z_edges       : (nz + 1,) redshifts of the same edges (only to place z_centre)
    z_centre      : centre of the band; depth: its comoving depth [cMpc/h]
    opening_angle : the cone's opening angle [rad] (header OpeningAngle)
    window_los    : "blackmanharris" or "none"; taper_perp: Tukey alpha across the sky (0 = none)
    Returns k [h/cMpc], Delta^2, modes per bin, and a dict describing the band.
    """
    nx, ny, nz = tb.shape
    chi_edges = np.asarray(chi_edges, dtype=np.float64)[:nz + 1]
    z_edges = np.asarray(z_edges, dtype=np.float64)[:nz + 1]
    oz = np.argsort(z_edges)
    chi0 = np.interp(z_centre, z_edges[oz], chi_edges[oz])
    centre = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    width = np.abs(np.diff(chi_edges))
    lo, hi = chi0 - 0.5 * depth, chi0 + 0.5 * depth
    sel = np.where((centre >= lo) & (centre < hi))[0]
    if len(sel) < 8:
        raise ValueError(f"only {len(sel)} slices in the band: increase depth or check z_centre")
    if lo < chi_edges.min() or hi > chi_edges.max():
        raise ValueError(f"band {lo:.1f}-{hi:.1f} cMpc/h is outside the light cone "
                         f"({chi_edges.min():.1f}-{chi_edges.max():.1f})")
    blo = (centre[sel] - 0.5 * width[sel]).min()        # the band: the selected slices, edge to edge
    Lz = (centre[sel] + 0.5 * width[sel]).max() - blo
    x = centre[sel] - blo                                # slice centres from the band edge
    dw = width[sel]

    cube = tb[:, :, sel].astype(np.float64)
    if slice_mean:
        cube -= cube.mean(axis=(0, 1), keepdims=True)
    else:
        cube -= np.sum(cube.mean(axis=(0, 1)) * dw) / dw.sum()
    if window_los == "blackmanharris":
        wz = blackman_harris(x / Lz)
    elif window_los == "none":
        wz = np.ones(len(sel))
    else:
        raise ValueError(f"unknown window_los {window_los}")
    wx = tukey(nx, taper_perp) if taper_perp > 0 else np.ones(nx)
    wy = tukey(ny, taper_perp) if taper_perp > 0 else np.ones(ny)
    cube *= wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    norm = np.mean(wx**2) * np.mean(wy**2) * np.sum(wz**2 * dw) / Lz

    dx = opening_angle / nx * chi0                       # flat sky at the band centre
    dy = opening_angle / ny * chi0
    Lx, Ly = nx * dx, ny * dy

    # transverse FFT, then the exact line-of-sight Fourier sum over the slices (centres x, widths dw)
    G = np.fft.fft2(cube, axes=(0, 1)).reshape(nx * ny, len(sel))
    del cube
    m = np.arange(len(sel) // 2 + 1)
    kz = 2 * np.pi * m / Lz
    E = dw[:, None] * np.exp(-1j * kz[None, :] * x[:, None])
    F = (G @ E).reshape(nx, ny, len(m)) * (dx * dy)
    del G
    P = np.abs(F)**2 / (Lx * Ly * Lz) / norm

    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    k = np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
    kmin = 2 * np.pi / max(Lx, Ly, Lz)
    kmax = np.pi / max(dx, dy, Lz / len(sel))            # the lowest Nyquist (mean slice width along the line of sight)
    ok = (k >= kmin) & (k < kmax)
    k, P = k[ok], P[ok]
    b = np.floor(np.log10(k / kmin) * bins_per_decade + 1e-9).astype(int)   # a shell on a bin edge goes up
    n = np.bincount(b)
    use = n > 0
    kb = np.bincount(b, k)[use] / n[use]
    D2 = np.bincount(b, k**3 * P)[use] / n[use] / (2 * np.pi**2)
    zs = z_edges[np.argsort(chi_edges)]
    zlo, zhi = np.interp([blo, blo + Lz], np.sort(chi_edges), zs)
    info = dict(z_range=(zlo, zhi), chi_range=(blo, blo + Lz), nslices=len(sel), L=(Lx, Ly, Lz), pix=(dx, dy),
                slice_width=(dw.min(), dw.max()), window_los=window_los, taper_perp=taper_perp,
                slice_mean=slice_mean)
    return kb, D2, n[use], info
