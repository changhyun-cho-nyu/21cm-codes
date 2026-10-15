"""
test_cells_Tb.py
================
delta T_b from the Voronoi gas cells of a full Lumina snapshot, against the same gas averaged in pixels:
the resolution (clumping) test of the 3.46 cm signal, and of 21 cm at high z, at the simulation's own resolution.

Why. The maps compute delta T_b from pixel averages (mass-weighted T and fractions, mean density). delta T_b^He
~ n_e n_HeII f(T) is two-body, so the average of the cell values exceeds the value of the averages; the render
test (test_resolution_Tb.py) found no convergence up to 2560^3. Here the limit is the gas cells themselves.

Method. A full snapshot has ~2e11 gas cells, so it reads K blocks of B consecutive cells spread through the
file (AREPO writes cells in Peano-Hilbert domain order, so a block is a compact region; `inspect` checks it).
In each block, for every cell (render conventions: star-forming gas neutral with He II = He III = 0; T from u
and Ne as in RT_movie_3D_rendering.cc; J_alpha of the 640^3 file at the cell's pixel):
  cell   delta T_b of each cell (tb_physics.py, the physics of the C codes)
  Ng     cells grouped by their centre into Ng^3 pixels (Ng = 320 ... 10240); per pixel the gas density
         sum(m)/sum(V), mass-weighted T, x_HII, f_HeII, f_HeIII, volume-weighted J; delta T_b of those averages,
         given to every cell of the pixel. Ng = 640 is the production calculation; Ng -> infinity is `cell`.
Only cells in complete 320^3 pixels (gas volume within --complete of the pixel volume) are used, so every
pixel at every level is a whole pixel. Reported, for each level: the volume-weighted <dT_b> and its ratio to
the cell level (jackknife over blocks), and the variance of the map smoothed to 320^3 and 640^3 pixels
(~ the power-spectrum amplitude at k <~ 2 and 4 h/cMpc) with its correlation with the cell-level map.
With --render: the 640^3 render at the same pixels (axis order, Density, T, fractions, and dT_b from the
render, which is the production map) against the cells grouped at 640, as a check of the whole mapping.

Usage
-----
  python test_cells_Tb.py inspect --snapdir <.../snapshots> [--snap 20] [--renders A B] [--jdir <Jalpha>]
  python test_cells_Tb.py run --snapdir <.../snapshots> --snap 20 --line He --J <J stars> --J <J AGN> \\
         [--render <ren_640> --render-snap 406] [--blocks 32 --block-size 40000000]
Memory: ~12 GB per 4e7-cell block plus the 640^3 J (1 GB).
"""
import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from tb_physics import SRC, X_H, Y_HE, M_H, G, MPC, OMEGA_L_DEFAULT, Line

# AREPO constants (RT_movie_3D_rendering.cc uses them for the render temperature)
GAMMA_MINUS1, BOLTZMANN, PROTONMASS, HYDROGEN_MASSFRAC = 2.0 / 3.0, 1.38065e-16, 1.67262178e-24, 0.76
GAS_FIELDS = ("Coordinates", "Masses", "Density", "InternalEnergy", "HII_Fraction", "HeII_Fraction", "HeIII_Fraction",
              "StarFormationRate")
# Lumina's repackaged snapshots store IntCoordinates (32-bit integers, x = int * BoxSize / 2^32, Zier+26 Sec. 9.1)
# and no ElectronAbundance. AREPO-RT always sets Ne = HII + HeII + 2 HeIII (MRT/RT_update_primitive_variables.cc,
# MRT/RT.cc; star-forming gas: set_full_ionization_mrt, Ne = (1+X)/2X = 1 + 2y), so Ne is rebuilt from the stored
# fractions (before the render's neutral convention for star-forming gas), as the render used it for T.
LEVELS = (320, 640, 1280, 2560, 5120, 10240)
SCALES = (320, 640)


class Snapshot:
    """header and PartType0 fields of a full snapshot (columnar layout or the snap_NNN.hdf5 virtual file)"""
    def __init__(self, snapdir, snap, gas_file=None):
        self.path = os.path.join(snapdir, f"snap_{snap:03d}.hdf5")
        if not os.path.exists(self.path):
            sys.exit(f"ERROR: no {self.path}")
        with h5py.File(self.path, "r") as f:
            self.hd = {}
            for grp in ("Parameters", "Config", "Header"):          # Header last: it wins
                if grp in f:
                    self.hd.update({k: v for k, v in f[grp].attrs.items()})
        cands = [gas_file] if gas_file else [os.path.join(snapdir, "PartType0", f"PartType0_{snap:03d}.hdf5"), self.path]
        self.f, self.g = None, None
        for c in cands:
            if c and os.path.exists(c):
                f = h5py.File(c, "r")
                g = f["PartType0"] if "PartType0" in f else f
                if "Density" in g:
                    self.f, self.g, self.gas_path = f, g, c
                    break
                f.close()
        if self.g is None:
            sys.exit(f"ERROR: no PartType0 fields in {cands}")
        self.pos_name = "Coordinates" if "Coordinates" in self.g else "IntCoordinates"
        self.has_ne = "ElectronAbundance" in self.g
        missing = [k for k in GAS_FIELDS if k != "Coordinates" and k not in self.g]
        if self.pos_name not in self.g:
            missing.append("Coordinates / IntCoordinates")
        if missing:
            print(f"ERROR: {self.gas_path} lacks {missing}. What is there:")
            print(f"  {self.gas_path} [{self.g.name}]: " + ", ".join(
                f"{k} {self.g[k].shape if isinstance(self.g[k], h5py.Dataset) else '(group)'}" for k in self.g))
            d0 = os.path.join(snapdir, "PartType0")
            if os.path.isdir(d0):
                ents = sorted(os.listdir(d0))
                mine = [e for e in ents if f"{snap:03d}" in e] or ents[:20]
                print(f"  {d0}: {len(ents)} entries; for snapshot {snap}: {mine[:40]}")
                for e in mine[:3]:
                    q = os.path.join(d0, e)
                    if os.path.isfile(q) and e.endswith(".hdf5"):
                        with h5py.File(q, "r") as f:
                            f.visititems(lambda n, o: print(f"    {e}:{n} {o.shape} {o.dtype}")
                                         if isinstance(o, h5py.Dataset) else None)
                    elif os.path.isdir(q):
                        print(f"    {e}/: {sorted(os.listdir(q))[:10]}")
            sys.exit(1)
        self.n = self.g["Density"].shape[0]
        self.z = float(self.hd["Redshift"]) if "Redshift" in self.hd else 1.0 / float(self.hd["Time"]) - 1.0
        self.box = float(self.hd["BoxSize"])
        self.h = float(self.hd["HubbleParam"])
        pd = self.g[self.pos_name]
        if np.issubdtype(pd.dtype, np.integer) and not np.issubdtype(pd.dtype, np.unsignedinteger):
            sys.exit(f"STOP: {self.pos_name} is {pd.dtype} (signed); the int -> position mapping for signed integers is "
                     f"not documented. Report the dtype and min/max of a few rows before going on.")

    def unit(self, name, default):
        return float(self.hd.get(name, default))

    def read(self, name, i0, i1):
        if name == "Coordinates":
            d = self.g[self.pos_name]
            a = d[i0:i1].astype(np.float64)
            if np.issubdtype(d.dtype, np.integer):
                a *= self.box / 2.0**int(8 * d.dtype.itemsize)        # unsigned 32-bit: int * BoxSize / 2^32
            return a
        if name == "ElectronAbundance" and not self.has_ne:
            return (self.g["HII_Fraction"][i0:i1].astype(np.float64) + self.g["HeII_Fraction"][i0:i1].astype(np.float64)
                    + 2.0 * self.g["HeIII_Fraction"][i0:i1].astype(np.float64))
        return self.g[name][i0:i1].astype(np.float64)


def first_3d(f):
    return f[next(k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3)]


def render_redshifts(dirs):
    out = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "All", "All_*.hdf5")):
            s = int(os.path.basename(p)[4:7])
            try:
                with h5py.File(p, "r") as f:
                    out[s] = float(f["Header"].attrs["Redshift"])
            except Exception:
                pass
    return out


def inspect(a):
    snaps = sorted(int(os.path.basename(p)[5:8]) for p in glob.glob(os.path.join(a.snapdir, "snap_*.hdf5")))
    print(f"{a.snapdir}: {len(snaps)} snapshots; top level: {sorted(os.listdir(a.snapdir))[:12]}")
    rz = render_redshifts(a.renders) if a.renders else {}
    print(f"{'snap':>5} {'z':>9}" + (f" {'render':>7} {'z_render':>9} {'dz':>9}  J files" if rz else ""))
    for s in snaps:
        with h5py.File(os.path.join(a.snapdir, f"snap_{s:03d}.hdf5"), "r") as f:
            z = float(f["Header"].attrs["Redshift"])
        line = f"{s:5d} {z:9.5f}"
        if rz:
            r = min(rz, key=lambda k: abs(rz[k] - z))
            have = []
            if a.jdir:
                for tag, fs in (("H", (f"Jalpha_H_{r}.hdf5", f"Jalpha_AGN_H_{r}.hdf5")),
                                ("He", (f"Jalpha_He_{r}_{a.tr_tag}.hdf5", f"Jalpha_AGN_He_{r}_{a.tr_tag}.hdf5"))):
                    if all(os.path.exists(os.path.join(a.jdir, x)) for x in fs):
                        have.append(tag)
            line += f" {r:7d} {rz[r]:9.5f} {z - rz[r]:+9.2e}  {','.join(have) or '-'}"
        print(line)
    if a.snap is None:
        return
    S = Snapshot(a.snapdir, a.snap, a.gas_file)
    print(f"\nsnapshot {a.snap}: z = {S.z:.5f}, BoxSize {S.box}, h {S.h}, {S.n:,} gas cells, fields from {S.gas_path}")
    for k in ("Omega0", "OmegaBaryon", "OmegaLambda", "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s",
              "NumFilesPerSnapshot"):
        print(f"  {k}: {S.hd.get(k, '(absent)')}")
    print(f"  positions: {S.pos_name}; ElectronAbundance stored: {S.has_ne}"
          + ("" if S.has_ne else " (rebuilt as HII + HeII + 2 HeIII)"))
    for k in [S.pos_name] + [f for f in GAS_FIELDS if f != "Coordinates"]:
        d = S.g[k]
        print(f"  {k:18s} {str(d.shape):22s} {str(d.dtype):8s} chunks {d.chunks} compression {d.compression} "
              f"virtual {d.is_virtual}")
    B = a.block_size
    print(f"\ncompactness of blocks of {B:,} consecutive cells (a Peano-Hilbert ordered file gives compact regions):")
    for i0 in (0, S.n // 3, 2 * S.n // 3):
        t0 = time.time()
        raw = S.g[S.pos_name][i0:i0 + B]
        dt = time.time() - t0
        x = S.read("Coordinates", i0, i0 + B)
        print(f"  {S.pos_name} rows {i0:,}+: raw min {raw.min(axis=0)}, max {raw.max(axis=0)}; "
              f"as positions {x.min():.4g}-{x.max():.4g} (BoxSize {S.box})")
        ext = []
        for ax in range(3):                                        # periodic extent: largest gap complement
            c = np.sort(x[:, ax])
            gaps = np.diff(np.concatenate([c, [c[0] + S.box]]))
            ext.append(S.box - gaps.max())
        side = (B * S.box**3 / S.n)**(1.0 / 3.0)
        print(f"  cells {i0:,}+: extent {ext[0] / 1e3:.1f} x {ext[1] / 1e3:.1f} x {ext[2] / 1e3:.1f} cMpc/h "
              f"(a cube of the same mean volume: {side / 1e3:.1f}); read {raw.nbytes / dt / 1e9:.2f} GB/s")
        # what the test needs: are the cells in whole 320^3 pixels (compact patches), or scattered?
        V = S.read("Masses", i0, i0 + B) / S.read("Density", i0, i0 + B)
        for Ng in (40, 320):
            _, inv, _ = group(x, S.box, Ng)
            cover = np.bincount(inv, V) / (S.box / Ng)**3
            full = np.abs(cover - 1.0) < a.complete
            print(f"    {Ng}^3 pixels ({S.box / Ng / 1e3:.2f} cMpc/h): {len(cover):,} touched, {full.sum():,} complete "
                  f"(|gas volume / pixel - 1| < {a.complete}); cells in complete pixels {full[inv].mean():.3f}; "
                  f"cover quartiles {np.percentile(cover, [25, 50, 75]).round(3).tolist()}")


class Acc:
    """volume-weighted sums per level, per block; and smoothed-map pixel values"""
    def __init__(self, names):
        self.names = names
        self.blk = []                                   # per block: {name: (sum V tb, sum V)}
        self.maps = {S: {n: [] for n in names} for S in SCALES}


def group(pos, box, Ng):
    ix = np.floor(pos * (Ng / box)).astype(np.int64) % Ng
    key = (ix[:, 0] * Ng + ix[:, 1]) * Ng + ix[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    return uk, inv.ravel(), ix


def run(a):
    S = Snapshot(a.snapdir, a.snap, a.gas_file)
    z, box, h = S.z, S.box, S.h
    om = float(S.hd.get("Omega0", 0.30964144)); ob = float(S.hd.get("OmegaBaryon", 0.048975))
    ol = float(S.hd.get("OmegaLambda", OMEGA_L_DEFAULT))
    UL, UM, UV = S.unit("UnitLength_in_cm", 3.085678e21), S.unit("UnitMass_in_g", 1.989e43), S.unit("UnitVelocity_in_cm_per_s", 1e5)
    rho_unit = UM * h**2 / UL**3                                     # code density -> g cm^-3 comoving
    H0 = 100.0 * h * 1e5 / MPC
    rho_mean = ob * 3.0 * H0**2 / (8.0 * np.pi * G) / rho_unit       # mean baryon density, code units
    L = Line(a.line, z, h, om, ob, ol)
    print(f"constants: {SRC}; snapshot {a.snap}, z = {z:.5f}, line {a.line}, {S.n:,} gas cells, prefactor {L.P:.6e} {L.unit}")
    print(f"  units: length {UL:.6e} cm, mass {UM:.6e} g, velocity {UV:.3e} cm/s; <rho_b> = {rho_mean:.6e} (cosmology)")

    R = None
    if a.render:
        R = {}
        with h5py.File(os.path.join(a.render, "All", f"All_{a.render_snap:03d}.hdf5"), "r") as f:
            zr = float(f["Header"].attrs["Redshift"])
        print(f"  render {a.render} snapshot {a.render_snap}: z = {zr:.5f} (dz = {z - zr:+.2e})")
        for k in ("Density", "Temperature", "HII_Fraction", "HeII_Fraction", "HeIII_Fraction"):
            p = os.path.join(a.render, k, f"{k}_{a.render_snap:03d}.hdf5")
            if not os.path.exists(p):
                p = os.path.join(a.render, "All", f"All_{a.render_snap:03d}.hdf5")
            R[k] = h5py.File(p, "r")[k]
        m6 = 0.0
        for i0 in range(0, R["Density"].shape[0], 32):
            m6 += R["Density"][i0:i0 + 32].astype(np.float64).sum()
        m6 /= float(R["Density"].shape[0])**3
        print(f"  <Density> of the render = {m6:.6e} ({m6 / rho_mean - 1:+.2e} vs the cosmological mean, which is used here)")

    Jf = [h5py.File(fn, "r") for fn in a.J]
    J = sum(first_3d(f)[()].astype(np.float32) for f in Jf)
    NJ = J.shape[0]
    print(f"  J_alpha: {a.J}, {NJ}^3, mean {J.astype(np.float64).mean():.4e}")

    def tb(rho, T, xHII, fII, fIII, Jx):
        nH = X_H * rho * rho_unit * (1.0 + z)**3 / M_H
        sig = rho * (1.0 - xHII) / rho_mean if a.line == "H" else (fII / Y_HE) * rho / rho_mean
        return L.tb(sig, nH, T, xHII, fII, fIII, Jx)[0]

    names = [str(n) for n in LEVELS] + ["cell"]
    acc = Acc(names)
    order = None if R is not None else (0, 1, 2)
    if order is not None:
        print("  no --render: J sampled with the render axes = (x, y, z), unchecked")
    val = {"n": 0, "rho": [], "T": [], "fII": [], "xHII": [], "tb_r": 0.0, "tb_640": 0.0, "tb_cell": 0.0}
    dshare_edges = [0, 1, 10, 100, np.inf]
    dshare = np.zeros(len(dshare_edges) - 1)
    ne_chk = []
    K, B = a.blocks, a.block_size
    starts = ((np.arange(K) + 0.5) * S.n / K - B / 2).astype(np.int64).clip(0, S.n - B)
    vpix320 = (box / 320)**3
    t0 = time.time()
    for ib, s0 in enumerate(starts):
        tb0 = time.time()
        F = {k: S.read(k, s0, s0 + B) for k in GAS_FIELDS + ("ElectronAbundance",)}
        pos, m, rho = F["Coordinates"], F["Masses"], F["Density"]
        sf = F["StarFormationRate"] > 0
        xHII = np.where(sf, 0.0, F["HII_Fraction"])
        fII = np.where(sf, 0.0, F["HeII_Fraction"])
        fIII = np.where(sf, 0.0, F["HeIII_Fraction"])
        mu = 4.0 / (1.0 + 3.0 * HYDROGEN_MASSFRAC + 4.0 * HYDROGEN_MASSFRAC * F["ElectronAbundance"]) * PROTONMASS
        T = GAMMA_MINUS1 * F["InternalEnergy"] * UV**2 / BOLTZMANN * mu
        nsf = ~sf
        ne_chk.append(np.median(np.abs(F["ElectronAbundance"][nsf] - (F["HII_Fraction"] + F["HeII_Fraction"]
                                                                       + 2 * F["HeIII_Fraction"])[nsf])))
        V = m / rho
        del F
        # complete 320^3 pixels
        uk, inv, _ = group(pos, box, 320)
        cover = np.bincount(inv, V) / vpix320
        keep = (np.abs(cover - 1.0) < a.complete)[inv]
        frac_keep = keep.mean()
        pos, m, rho, xHII, fII, fIII, T, V = (q[keep] for q in (pos, m, rho, xHII, fII, fIII, T, V))
        if len(m) == 0:
            print(f"  block {ib}: no complete 320^3 pixel (block not compact?), skipped"); continue
        # render validation and axis order (first block decides the order)
        g640 = group(pos, box, 640)
        if R is not None:
            uk6, inv6, ix6 = g640
            M6 = np.bincount(inv6, m)
            V6 = np.bincount(inv6, V)
            first = np.zeros(len(uk6), np.int64); first[inv6[::-1]] = np.arange(len(inv6))[::-1]
            pix = ix6[first]                                          # (x, y, z) pixel of each group
            vp6 = (box / 640)**3
            full6 = np.abs(V6 / vp6 - 1.0) < a.complete
            cand = [(0, 1, 2), (2, 1, 0)] if order is None else [order]
            best = None
            for o in cand:
                ia = pix[:, o[0]]
                ua = np.unique(ia)
                Rd = R["Density"][ua.tolist()]
                rd = Rd[np.searchsorted(ua, ia), pix[:, o[1]], pix[:, o[2]]]
                with np.errstate(invalid="ignore", divide="ignore"):
                    cc = np.corrcoef(np.log(M6[full6] / vp6), np.log(np.maximum(rd[full6], 1e-30)))[0, 1]
                cc = -1.0 if not np.isfinite(cc) else cc
                if best is None or cc > best[1]:
                    best = (o, cc)
                if order is None:
                    print(f"  axis order render[i0,i1,i2] = coords {o}: corr(log rho) = {cc:.4f}")
            order = best[0]
            o = order
            ia = pix[:, o[0]]; ua = np.unique(ia); sel = np.searchsorted(ua, ia)
            rf = {k: R[k][ua.tolist()][sel, pix[:, o[1]], pix[:, o[2]]].astype(np.float64) for k in R}
            avg = lambda q: np.bincount(inv6, m * q) / M6
            Tm, xm, f2, f3 = avg(T), avg(xHII), avg(fII), avg(fIII)
            w = full6
            val["n"] += int(w.sum())
            val["rho"].append(np.median(rf["Density"][w] / (M6[w] / vp6)))
            val["T"].append(np.median(rf["Temperature"][w] / Tm[w]))
            val["fII"].append(np.median(np.abs(rf["HeII_Fraction"][w] - f2[w])) / Y_HE)
            val["xHII"].append(np.median(np.abs(rf["HII_Fraction"][w] - xm[w])))
            Jp = J[pix[:, o[0]] * NJ // 640, pix[:, o[1]] * NJ // 640, pix[:, o[2]] * NJ // 640].astype(np.float64)
            tr = tb(rf["Density"], rf["Temperature"], rf["HII_Fraction"], rf["HeII_Fraction"], rf["HeIII_Fraction"], Jp)
            val["tb_r"] += tr[w].sum()
            val["_w"] = w
        # J at each cell (render axes)
        ixJ = np.floor(pos * (NJ / box)).astype(np.int64) % NJ
        Jc = J[ixJ[:, order[0]], ixJ[:, order[1]], ixJ[:, order[2]]].astype(np.float64)
        del ixJ
        tbc = tb(rho, T, xHII, fII, fIII, Jc)
        # density shares of the cell-level mean
        dlt = rho / rho_mean
        dshare += np.bincount(np.clip(np.digitize(dlt, dshare_edges) - 1, 0, len(dshare) - 1), V * tbc, minlength=len(dshare))
        sinv = {Sg: (g640[1] if Sg == 640 else group(pos, box, Sg)[1]) for Sg in SCALES}
        res = {}
        for lv in names:
            if lv == "cell":
                v = tbc
            else:
                Ng = int(lv)
                inv = g640[1] if Ng == 640 else group(pos, box, Ng)[1]
                Mp, Vp = np.bincount(inv, m), np.bincount(inv, V)
                avg = lambda q: np.bincount(inv, m * q) / Mp
                v = tb(Mp / Vp, avg(T), avg(xHII), avg(fII), avg(fIII), np.bincount(inv, V * Jc) / Vp)[inv]
            res[lv] = ((V * v).sum(), V.sum())
            for Sg in SCALES:
                si = sinv[Sg]
                acc.maps[Sg][lv].append(np.bincount(si, V * v) / np.bincount(si, V))
            if R is not None and lv in ("640", "cell"):
                w = val["_w"]
                inv6 = g640[1]
                val["tb_640" if lv == "640" else "tb_cell"] += (np.bincount(inv6, V * v) / np.bincount(inv6, V))[w].sum()
        acc.blk.append(res)
        r640 = res["640"][0] / res["cell"][0] if res["cell"][0] else np.nan
        print(f"  block {ib:2d} cells {s0:,}+: {frac_keep:.2f} in complete 320^3 pixels, <dT_b> cell "
              f"{res['cell'][0] / res['cell'][1]:+.4e}, 640/cell {r640:.4f}  ({time.time() - tb0:.0f} s)", flush=True)

    if not acc.blk:
        sys.exit("no usable block")
    print(f"\nsnapshot {a.snap}, z = {z:.4f}, line {a.line}: {len(acc.blk)} blocks, {time.time() - t0:.0f} s")
    if S.has_ne:
        print(f"  median |ElectronAbundance - (x_HII + f_HeII + 2 f_HeIII)| (non-SF cells) = {np.median(ne_chk):.2e}")
    else:
        print("  ElectronAbundance not stored: Ne = HII + HeII + 2 HeIII (the render Temperature check below tests it)")
    tot = {lv: sum(b[lv][0] for b in acc.blk) / sum(b[lv][1] for b in acc.blk) for lv in names}
    rat = np.array([[b[lv][0] / b["cell"][0] for lv in names] for b in acc.blk])
    nb = len(acc.blk)
    jk = []                                                          # jackknife of the pooled ratio
    for i in range(nb):
        num = {lv: sum(acc.blk[j][lv][0] for j in range(nb) if j != i) for lv in names}
        jk.append([num[lv] / num["cell"] for lv in names])
    jk = np.array(jk)
    jerr = np.sqrt((nb - 1) / nb * ((jk - jk.mean(0))**2).sum(0))
    print(f"\n  <dT_b> [{L.unit}] (volume-weighted over the cells used) and ratio to the cell level (jackknife error):")
    for i, lv in enumerate(names):
        print(f"    {lv:>6}: {tot[lv]:+.5e}   ratio {tot[lv] / tot['cell']:.4f} +- {jerr[i]:.4f}"
              f"   (blocks: median {np.median(rat[:, i]):.4f}, range {rat[:, i].min():.4f}-{rat[:, i].max():.4f})")
    print(f"  share of the cell-level <dT_b> by 1+delta: " + ", ".join(
        f"{dshare_edges[i]:g}-{dshare_edges[i + 1]:g}: {dshare[i] / dshare.sum():.3f}" for i in range(len(dshare))))
    for Sg in SCALES:
        ref = np.concatenate(acc.maps[Sg]["cell"])
        print(f"  map smoothed to {Sg}^3 pixels ({len(ref):,} pixels): variance ratio to cell level / correlation")
        print("    " + "  ".join(f"{lv}: {np.var(np.concatenate(acc.maps[Sg][lv])) / np.var(ref):.4f}/"
                                   f"{np.corrcoef(np.concatenate(acc.maps[Sg][lv]), ref)[0, 1]:.4f}" for lv in names[:-1]))
    if R is not None:
        print(f"\n  render check on {val['n']:,} complete 640^3 pixels (axis order {order}): render / cells grouped at 640:")
        print(f"    Density median ratio {np.median(val['rho']):.4f}, Temperature {np.median(val['T']):.4f}, "
              f"median |dx_HII| {np.median(val['xHII']):.2e}, median |dx_HeII| {np.median(val['fII']):.2e}")
        print(f"    <dT_b> on these pixels: render (production map) {val['tb_r'] / val['n']:+.5e}, cells at 640 "
              f"{val['tb_640'] / val['n']:+.5e}, cell level {val['tb_cell'] / val['n']:+.5e} {L.unit}")
        print(f"    => production map / cell level on these pixels: {val['tb_r'] / val['tb_cell']:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("inspect", "run"):
        q = sub.add_parser(name)
        q.add_argument("--snapdir", required=True)
        q.add_argument("--gas-file", help="file with the PartType0 fields (default: PartType0/PartType0_NNN.hdf5, else snap_NNN.hdf5)")
        q.add_argument("--block-size", type=int, default=40_000_000)
    q = sub.choices["inspect"]
    q.add_argument("--snap", type=int)
    q.add_argument("--complete", type=float, default=0.05)
    q.add_argument("--renders", nargs="*", help="ren_640 directories, to match render snapshots by z")
    q.add_argument("--jdir", help="J_alpha directory, to say which matched snapshots have J files")
    q.add_argument("--tr-tag", default="tr_sfmask")
    q = sub.choices["run"]
    q.add_argument("--snap", type=int, required=True)
    q.add_argument("--line", choices=["H", "He"], default="He")
    q.add_argument("--J", action="append", required=True, help="640^3 J_alpha files of the line (stars, AGN), summed")
    q.add_argument("--render", help="ren_640 directory for the check against the production inputs")
    q.add_argument("--render-snap", type=int)
    q.add_argument("--blocks", type=int, default=32)
    q.add_argument("--complete", type=float, default=0.05, help="|gas volume / pixel volume - 1| for a complete 320^3 pixel")
    a = p.parse_args()
    if a.mode == "run" and a.render and a.render_snap is None:
        sys.exit("--render needs --render-snap")
    inspect(a) if a.mode == "inspect" else run(a)


if __name__ == "__main__":
    main()
