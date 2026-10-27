"""
lightcone_tb.py
===============
delta T_b of the 21 cm and 3.46 cm lines on the Lumina light cone (rlc_*/All.hdf5), slice by slice, with the
physics of the production maps (tb_physics.Line, checked against PS_21cm_mpi.c / PS_3cm_mpi.c by check_Tb_maps.py):
exact tau T_S prefactor with H(z) including Lambda, S_alpha in x_alpha, the Furlanetto+06 collision rates,
x_c^He of McQuinn & Switzer (2009), n_e = n_H (x_HII + f_HeII + 2 f_HeIII) with the per-H He fractions; cosmology
and units from the light-cone header.

  - read_cone joins the light cones of the two runs (z > 4.75 and z < 4.75) at their shared edge without
    duplicating it, after checking that they are the same cone (NumPixels, OpeningAngle) and that the edges meet.
    The two runs' Center differ (the below-z = 4.75 run's box coordinates are shifted), but the cone is continuous
    across the join: corr(log rho) of the last outer and first inner slices 0.567, of two outer neighbours 0.605
    (2026-10-26).
  - cone_tb: as test_cells_Tb.py, the mean density is Omega_b rho_crit (the production maps normalise by the map's
    gas mean: -0.4% to +0.7% in delta T_b, JALPHA_NOTES §6.2). J_alpha is whatever the caller passes, in proper
    photons s^-1 cm^-2 Hz^-1 sr^-1 (the light cone has no J_alpha field).
Same limits as the 640^3 maps: real space; star-forming gas neutral; 3.46 cm amplitude a lower limit (resolution).

Usage
-----
  from lightcone_tb import read_cone, cone_tb, units
  base = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
  cone = read_cone([f"{base}/Flagship_new/postprocessing/rlc_640", f"{base}/Flagship_below_z_4p75_new/postprocessing/rlc_640"],
                   sel=np.s_[64, :], zmax=20.0)       # one row of pixels (the fan diagram); sel=np.s_[:, :] for all
  tb21 = cone_tb(cone, "H", J_H)                      # mK, shape of cone["Density"] (..., nslices)
  tbHe = cone_tb(cone, "He", J_He)                    # uK
  cone["z"] (slice centres), cone["z_edges"], cone["chi_edges"] [cMpc/h], cone["header"]
  delta = cone["Density"] / units(cone["header"])[1]  # 1 + delta about Omega_b rho_crit
"""
import os

import h5py
import numpy as np

from tb_physics import G, M_H, MPC, OMEGA_L_DEFAULT, X_H, Y_HE, Line

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

FIELDS = ("Density", "Temperature", "HII_Fraction", "HeII_Fraction", "HeIII_Fraction")
HDR_KEYS = ("HubbleParam", "Omega0", "OmegaBaryon", "OmegaLambda", "UnitLength_in_cm", "UnitMass_in_g",
            "NumPixels", "OpeningAngle", "BoxSize")


def read_cone(dirs, fields=FIELDS, sel=np.s_[:, :], zmax=None, zmin=None):
    """
    dirs  : light-cone directories, high z first (e.g. Flagship_new/.../rlc_640, Flagship_below_z_4p75_new/.../rlc_640)
    sel   : selection of the two transverse axes (np.s_[64, :] is the row of the fan diagram)
    zmax, zmin : keep only the slices lying entirely within zmin <= z <= zmax (None: no limit)
    Returns a dict: the fields (..., nslices), z_edges, chi_edges [cMpc/h], z (slice centres), header.
    """
    hds, parts = [], []
    for d in dirs:
        with h5py.File(os.path.join(d, "All.hdf5"), "r") as f:
            hds.append({k: f["Header"].attrs[k] for k in HDR_KEYS if k in f["Header"].attrs})
            ze, ce = f["Redshifts"][()], f["Distances"][()]   # edges, z decreasing; slice i spans ze[i] .. ze[i+1]
            keep = np.ones(len(ze) - 1, bool)
            if zmax is not None:
                keep &= ze[:-1] <= zmax
            if zmin is not None:
                keep &= ze[1:] >= zmin
            if not keep.any():
                continue                                      # no slice of this run in the range
            i0, i1 = np.flatnonzero(keep)[[0, -1]]
            data = {k: f[k][sel + (slice(i0, i1 + 1),)] for k in fields}
        parts.append((ze[i0:i1 + 2], ce[i0:i1 + 2], data))
    if not parts:
        raise ValueError(f"no slices with {zmin} <= z <= {zmax}")

    hd0 = hds[0]
    for hd in hds[1:]:
        for k in ("NumPixels", "OpeningAngle", "BoxSize", "HubbleParam", "Omega0", "OmegaBaryon"):
            if k in hd0 and not np.isclose(hd[k], hd0[k], rtol=1e-12):
                raise ValueError(f"{k} differs between the light cones: {hd0[k]} vs {hd[k]}")
    ze, ce, data = parts[0]
    for ze2, ce2, d2 in parts[1:]:
        if not (np.isclose(ze2[0], ze[-1], rtol=0, atol=1e-9) and np.isclose(ce2[0], ce[-1], rtol=0, atol=1e-3)):
            raise ValueError(f"the light cones do not meet: z {ze[-1]} vs {ze2[0]}, D {ce[-1]} vs {ce2[0]}")
        ze, ce = np.concatenate([ze, ze2[1:]]), np.concatenate([ce, ce2[1:]])      # the shared edge once
        data = {k: np.concatenate([data[k], d2[k]], axis=-1) for k in data}
    nz = next(iter(data.values())).shape[-1]
    if len(ze) != nz + 1:
        raise ValueError(f"{len(ze)} edges for {nz} slices")
    out = dict(data)
    out.update(z_edges=ze, chi_edges=ce / 1e3, z=0.5 * (ze[:-1] + ze[1:]), header=hd0)
    return out


def units(hd):
    """(code density -> g cm^-3 comoving, Omega_b rho_crit in code density units) from a light-cone header"""
    h, ob = float(hd["HubbleParam"]), float(hd["OmegaBaryon"])
    UL, UM = float(hd.get("UnitLength_in_cm", 3.085678e21)), float(hd.get("UnitMass_in_g", 1.989e43))
    rho_unit = UM * h**2 / UL**3
    H0 = 100.0 * h * 1e5 / MPC
    return rho_unit, ob * 3.0 * H0**2 / (8.0 * np.pi * G) / rho_unit


def cone_tb(cone, line, J, chunk=256):
    """delta T_b [mK for line="H", uK for "He"] of every light-cone cell; J broadcastable to cone["Density"]"""
    hd = cone["header"]
    h, om, ob = float(hd["HubbleParam"]), float(hd["Omega0"]), float(hd["OmegaBaryon"])
    ol = float(hd.get("OmegaLambda", OMEGA_L_DEFAULT))
    rho_unit, rho_mean = units(hd)

    rho, T, xHII = cone["Density"], cone["Temperature"], cone["HII_Fraction"]
    fII, fIII = cone["HeII_Fraction"], cone["HeIII_Fraction"]
    for name, a in (("Density", rho), ("Temperature", T)):
        if np.any(a <= 0):
            raise ValueError(f"{np.count_nonzero(a <= 0)} cells with {name} <= 0")
    if max(fII.max(), fIII.max()) > 1.05 * Y_HE:
        raise ValueError(f"He fractions up to {max(fII.max(), fIII.max()):.4f} > 1.05 y = {1.05 * Y_HE:.4f}: "
                         "not per H nucleus as assumed")
    J = np.broadcast_to(J, rho.shape)

    out = np.empty(rho.shape, dtype=np.float32)
    z = cone["z"]
    for j0 in range(0, len(z), chunk):                            # chunks of slices (memory)
        s = np.s_[..., j0:j0 + chunk]
        L = Line(line, z[j0:j0 + chunk], h, om, ob, ol)           # z per slice, broadcast along the last axis
        r, xh = rho[s].astype(np.float64), xHII[s].astype(np.float64)
        f2, f3 = fII[s].astype(np.float64), fIII[s].astype(np.float64)
        nH = X_H * r * rho_unit * (1.0 + L.z)**3 / M_H           # proper cm^-3
        sig = r * (1.0 - xh) / rho_mean if line == "H" else (f2 / Y_HE) * r / rho_mean
        out[s] = L.tb(sig, nH, T[s].astype(np.float64), xh, f2, f3, J[s].astype(np.float64))[0]
    return out
