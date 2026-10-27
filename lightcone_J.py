"""
lightcone_J.py
==============
J_alpha (stars + AGN) of the production box snapshots, sampled onto the light cone at the box cells given by the
cone -> box mapping (lightcone_tb.CONE_ORIGIN, check_cone_mapping.py): pixel (ix, iy) of the slice at centre
distance r is at P = O + (t_x p, t_y p, p) mod BoxSize, p = r / sqrt(1 + t_x^2 + t_y^2), nearest J cell.

J exists only at the render snapshots of the J production (H: the SNAPS of run_21cm.sh; He: those of run_3cm.sh,
z <= 10), each at one time. Each light-cone slice is at its own z, so in time:
  time="nearest": J of the J snapshot nearest in z;
  time="linear" : linear in z between the two J snapshots that bracket the slice, each sampled at the slice's cells.
Slices outside the J snapshots' z range: an error unless outside="zero" (J = 0) or "hold" (the end snapshot's J at
the slice's cells). Both are modelling choices (the user's). The J snapshots' z is the dataset's Redshift attribute.
Only the 64^3 chunks touched are read (a fan-diagram row touches ~30 per file and J snapshot).

Usage
-----
  from lightcone_tb import read_cone, cone_tb
  from lightcone_J import cone_J
  cone = read_cone([...], sel=np.s_[64, :], zmax=20.0)
  J_H, info = cone_J(cone, "H", sel=np.s_[64, :], time="linear")     # same sel as read_cone
  tb21 = cone_tb(cone, "H", J_H)
"""
import glob
import os
import re

import h5py
import numpy as np

from lightcone_tb import CONE_ORIGIN

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

JDIR = "/orcd/data/mvogelsb/004/chcho/Jalpha"
DSETS = {"H": ("J_alpha", "Jalpha_AGN_H"), "He": ("J_alpha_He", "Jalpha_AGN_He")}


def j_files(band, jdir=JDIR, tag="tr_sfmask"):
    """{snap: (stars file, AGN file, z)} of the J snapshots with both files"""
    if band == "H":
        pat, agn = r"Jalpha_H_(\d+)\.hdf5", "Jalpha_AGN_H_{}.hdf5"
    else:
        pat, agn = rf"Jalpha_He_(\d+)_{tag}\.hdf5", "Jalpha_AGN_He_{}_" + tag + ".hdf5"
    out = {}
    for p in glob.glob(os.path.join(jdir, "Jalpha_*.hdf5")):
        m = re.fullmatch(pat, os.path.basename(p))
        if not m:
            continue
        s = int(m.group(1))
        pa = os.path.join(jdir, agn.format(s))
        if not os.path.exists(pa):
            continue
        with h5py.File(p, "r") as f:
            z = float(f[DSETS[band][0]].attrs["Redshift"])
        out[s] = (p, pa, z)
    if not out:
        raise FileNotFoundError(f"no {band} J files in {jdir}")
    return out


def box_mean(files, band):
    """z (increasing) and the box-mean J (stars + AGN) of the J snapshots, from the J_mean attributes"""
    rows = []
    for s, (p1, p2, z) in files.items():
        with h5py.File(p1, "r") as f1, h5py.File(p2, "r") as f2:
            m1, m2 = f1[DSETS[band][0]].attrs.get("J_mean"), f2[DSETS[band][1]].attrs.get("J_mean")
        if m1 is not None and m2 is not None:
            rows.append((z, float(m1) + float(m2)))
    rows.sort()
    return np.array(rows).T if rows else None


def cone_cells(cone, sel, Nb, origin=CONE_ORIGIN):
    """flat J-grid cell index (Nb^3 grid) of every selected pixel of every slice, shape cone["Density"].shape"""
    hd = cone["header"]
    N, theta, L = int(hd["NumPixels"]), float(hd["OpeningAngle"]), float(hd["BoxSize"]) / 1e3   # cMpc/h
    t = ((np.arange(N) + 0.5) / N - 0.5) * theta
    tx, ty = np.meshgrid(t, t, indexing="ij")
    tx, ty = tx[sel], ty[sel]
    r = 0.5 * (cone["chi_edges"][:-1] + cone["chi_edges"][1:])            # cMpc/h
    p = r / np.sqrt(1.0 + tx[..., None]**2 + ty[..., None]**2)            # (..., nslices)
    dx = L / Nb
    idx = [np.floor(((o + q) % L) / dx).astype(np.int64) % Nb
           for o, q in zip(origin, (tx[..., None] * p, ty[..., None] * p, p))]
    out = (idx[0] * Nb + idx[1]) * Nb + idx[2]
    if out.shape != cone["Density"].shape:
        raise ValueError(f"sel gives {out.shape}, the cone fields are {cone['Density'].shape}: pass read_cone's sel")
    return out


def gather(dset, cells):
    """dset values at flat cell indices, reading only the chunks touched (the whole dataset if most are)"""
    Nb = dset.shape[0]
    shape, cells = cells.shape, cells.ravel()
    i, j, k = cells // (Nb * Nb), (cells // Nb) % Nb, cells % Nb
    cx, cy, cz = dset.chunks or (64, 64, 64)
    cid = ((i // cx) * ((Nb + cy - 1) // cy) + j // cy) * ((Nb + cz - 1) // cz) + k // cz
    u, inv = np.unique(cid, return_inverse=True)
    nchunk = ((Nb + cx - 1) // cx) * ((Nb + cy - 1) // cy) * ((Nb + cz - 1) // cz)
    if len(u) > nchunk // 2:
        return dset[()][i, j, k].reshape(shape)
    out = np.empty(cells.shape, np.float32)
    ny, nz = (Nb + cy - 1) // cy, (Nb + cz - 1) // cz
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(u) + 1))
    for n, c in enumerate(u):
        m = order[bounds[n]:bounds[n + 1]]
        a, b, d = c // (ny * nz), (c // nz) % ny, c % nz
        blk = dset[a * cx:(a + 1) * cx, b * cy:(b + 1) * cy, d * cz:(d + 1) * cz]
        out[m] = blk[i[m] - a * cx, j[m] - b * cy, k[m] - d * cz]
    return out.reshape(shape)


def cone_J(cone, band, sel, time="linear", outside=None, jdir=JDIR, tag="tr_sfmask", origin=CONE_ORIGIN, verbose=True):
    """
    J_alpha [proper photons s^-1 cm^-2 Hz^-1 sr^-1], stars + AGN, on the cone cells (shape cone["Density"].shape).
    sel: the transverse selection passed to read_cone. Returns (J, info): info["weights"][slice] = {snap: weight}.
    """
    if time not in ("nearest", "linear"):
        raise ValueError("time must be 'nearest' or 'linear'")
    files = j_files(band, jdir, tag)
    snaps = np.array(sorted(files, key=lambda s: files[s][2]))               # z increasing
    zs = np.array([files[s][2] for s in snaps])
    zc = cone["z"]
    lo, hi = zc < zs[0], zc > zs[-1]
    if (lo | hi).any() and outside not in ("zero", "hold"):
        raise ValueError(f"{(lo | hi).sum()} slices (z {zc[lo | hi].min():.3f} - {zc[lo | hi].max():.3f}) outside the "
                         f"{band} J snapshots (z {zs[0]:.3f} - {zs[-1]:.3f}): choose outside='zero' or 'hold'")
    weights = []
    for z in zc:
        if z < zs[0] or z > zs[-1]:
            weights.append({} if outside == "zero" else {int(snaps[0] if z < zs[0] else snaps[-1]): 1.0})
            continue
        b = int(np.searchsorted(zs, z))                      # zs[b-1] < z <= zs[b]
        if zs[b] == z or b == 0:
            weights.append({int(snaps[b]): 1.0})
            continue
        a = b - 1
        w = (z - zs[a]) / (zs[b] - zs[a])
        if time == "nearest":
            weights.append({int(snaps[b] if w >= 0.5 else snaps[a]): 1.0})
        else:
            weights.append({int(snaps[a]): 1.0 - w, int(snaps[b]): w})

    J = np.zeros(cone["Density"].shape, np.float32)
    needed = sorted({s for w in weights for s in w})
    cells = None
    for n, s in enumerate(needed):
        js = [jj for jj, w in enumerate(weights) if s in w]
        wv = np.array([weights[jj][s] for jj in js], np.float32)
        with h5py.File(files[s][0], "r") as f1, h5py.File(files[s][1], "r") as f2:
            d1, d2 = f1[DSETS[band][0]], f2[DSETS[band][1]]
            if cells is None:
                cells = cone_cells(cone, sel, d1.shape[0], origin)
            c = cells[..., js]
            J[..., js] += wv * (gather(d1, c) + gather(d2, c))
        if verbose and (n % 25 == 0 or n == len(needed) - 1):
            print(f"  {band} J: snapshot {s} (z = {files[s][2]:.4f}), {len(js)} slices; {n + 1}/{len(needed)}", flush=True)
    return J, {"weights": weights, "files": files}
